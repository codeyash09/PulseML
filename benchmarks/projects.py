"""A real project directory for a scenario, so shell access has something to find.

The curve benchmarks hand the agent numbers and nothing else. That is the right test for
detection, and the wrong one for tools: a shell in an empty directory can only report
that the directory is empty, and "tools did not help" would be a fact about the harness.

So each scenario gets a directory that looks like the run it describes -- a training
script, a config, a metrics file at full resolution, a log. Where the fault has a cause
that lives in code, the code contains it: `lr_to_zero` really does build a schedule that
reaches zero at the wrong epoch, `shuffled_labels` really does shuffle the label column,
`eval_in_train_mode` really is missing its `model.eval()`.

Where the fault has no cause in the training script -- a GPU throttling, one rank slower
than the others, a disk that filled -- the script is clean, and that is not an oversight.
An agent that finds a bug in those is confabulating, and the benchmark should be able to
tell.
"""
import json
import os

BASE = '''"""{title}

Training script for the {name} experiment.
"""
import argparse
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger("train")

CONFIG = {config}


def set_seed(seed):
{seed_body}


def build_dataset(split):
    """Loads the shard for `split` and returns a TensorDataset of (inputs, labels)."""
    shard = np.load(os.path.join(CONFIG["data_dir"], f"{{split}}.npz"))
    inputs = torch.from_numpy(shard["inputs"]).float()
    labels = torch.from_numpy(shard["labels"]).long()
{label_body}
    return TensorDataset(inputs, labels)


def build_model():
    hidden = CONFIG["hidden"]
    return nn.Sequential(
        nn.Linear(CONFIG["features"], hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Dropout(CONFIG["dropout"]),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden, momentum=CONFIG["bn_momentum"]),
        nn.ReLU(),
        nn.Linear(hidden, CONFIG["num_classes"]),
    )


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        eps=CONFIG["adam_eps"],
    )


def build_scheduler(optimizer, steps_per_epoch):
{scheduler_body}


def evaluate(model, loader, device):
{eval_body}


def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_dataset("train")
    val_set = build_dataset("val")
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle={shuffle})
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"])

    model = build_model().to(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["amp"])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        running = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=CONFIG["amp"]):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels,
                                       label_smoothing=CONFIG["label_smoothing"])
{loss_body}
            scaler.scale(loss).backward()
{grad_body}
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        log.info("epoch=%d loss=%.6f val_loss=%.6f val_accuracy=%.4f lr=%.3e",
                 epoch, train_loss, val_loss, val_accuracy,
                 optimizer.param_groups[0]["lr"])
{checkpoint_body}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
'''

DEFAULT_CONFIG = {
    "data_dir": "./data", "features": 256, "hidden": 512, "num_classes": 10,
    "batch_size": 128, "epochs": 60, "lr": 1e-3, "weight_decay": 0.01,
    "adam_eps": 1e-8, "dropout": 0.1, "bn_momentum": 0.1, "label_smoothing": 0.0,
    "warmup_epochs": 2, "amp": True, "seed": 1234,
}

PARTS = {
    "seed_body": '    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)',
    "label_body": "",
    "shuffle": "True",
    "scheduler_body": (
        '    total = CONFIG["epochs"] * steps_per_epoch\n'
        '    warmup = CONFIG["warmup_epochs"] * steps_per_epoch\n\n'
        '    def curve(step):\n'
        '        if step < warmup:\n'
        '            return step / max(1, warmup)\n'
        '        progress = (step - warmup) / max(1, total - warmup)\n'
        '        return 0.5 * (1.0 + math.cos(math.pi * progress))\n\n'
        '    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)'),
    "eval_body": (
        '    model.eval()\n'
        '    total_loss, correct, seen = 0.0, 0, 0\n'
        '    with torch.no_grad():\n'
        '        for inputs, labels in loader:\n'
        '            inputs, labels = inputs.to(device), labels.to(device)\n'
        '            logits = model(inputs)\n'
        '            total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)\n'
        '            correct += (logits.argmax(dim=1) == labels).sum().item()\n'
        '            seen += labels.size(0)\n'
        '    return total_loss / max(1, seen), correct / max(1, seen)'),
    "loss_body": "",
    "grad_body": (
        '            scaler.unscale_(optimizer)\n'
        '            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_norm"])'),
    "checkpoint_body": (
        '        torch.save({"model": model.state_dict(), "epoch": epoch},\n'
        '                   os.path.join(CONFIG["checkpoint_dir"], f"epoch{epoch}.pt"))'),
}


# Where the fault genuinely lives in the training code. Each entry replaces one part of
# the template and/or overrides config values. Families absent from here get a clean
# script, because their cause is not in the script.
CODE_FAULTS = {
    "lr_to_zero": {"config": {"warmup_epochs": 2},
                   "scheduler_body": (
                       '    # Linear decay over a horizon that is NOT the run length.\n'
                       '    total = 20 * steps_per_epoch\n\n'
                       '    def curve(step):\n'
                       '        return max(0.0, 1.0 - step / total)\n\n'
                       '    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)')},
    "lr_jump": {"scheduler_body": (
        '    total = CONFIG["epochs"] * steps_per_epoch\n\n'
        '    def curve(step):\n'
        '        # Reloads the base LR halfway through instead of continuing the decay.\n'
        '        if step > total // 2:\n'
        '            return CONFIG["restart_multiplier"]\n'
        '        return 1.0 - step / total\n\n'
        '    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)'),
        "config": {"restart_multiplier": 40.0}},
    "lr_not_restored": {"config": {"resume_from": "checkpoints/epoch29.pt"},
                        "scheduler_body": (
                            '    # NOTE: built fresh on resume, so the LR restarts at its initial\n'
                            '    # value rather than continuing from the checkpointed step.\n'
                            '    total = CONFIG["epochs"] * steps_per_epoch\n\n'
                            '    def curve(step):\n'
                            '        return 1.0 - 0.9 * step / total\n\n'
                            '    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)')},
    "resume_regression": {"config": {"resume_from": "checkpoints/epoch29.pt",
                                     "restore_optimizer": False}},
    "adam_epsilon_too_large": {"config": {"adam_eps": 1.0}},
    "bn_stats_frozen": {"config": {"bn_momentum": 0.0}},
    "label_smoothing_floor": {"config": {"label_smoothing": 0.2}},
    "grad_clip_always_active": {"config": {"clip_norm": 0.05}},
    "gradient_accumulation_double": {
        "config": {"accumulation_steps": 8},
        "grad_body": (
            '            # Accumulates over N steps without scaling the loss or the LR.\n'
            '            if (step + 1) % CONFIG["accumulation_steps"] != 0:\n'
            '                continue\n'
            '            scaler.unscale_(optimizer)\n'
            '            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_norm"])')},
    "shuffled_labels": {"label_body": (
        '    # Label column is permuted independently of the inputs.\n'
        '    labels = labels[torch.randperm(labels.size(0))]')},
    "duplicate_batches": {"shuffle": "False",
                          "label_body": (
                              '    # Only the first batch worth of rows is ever used.\n'
                              '    inputs = inputs[:CONFIG["batch_size"]].repeat(64, 1)\n'
                              '    labels = labels[:CONFIG["batch_size"]].repeat(64)')},
    "data_ordering_bias": {"shuffle": "False",
                           "label_body": (
                               '    # Rows arrive sorted by class and the loader does not shuffle.\n'
                               '    order = torch.argsort(labels)\n'
                               '    inputs, labels = inputs[order], labels[order]')},
    "val_equals_train": {"label_body": (
        '    # Both splits read the same shard.\n'
        '    if split == "val":\n'
        '        shard = np.load(os.path.join(CONFIG["data_dir"], "train.npz"))\n'
        '        inputs = torch.from_numpy(shard["inputs"]).float()\n'
        '        labels = torch.from_numpy(shard["labels"]).long()')},
    "eval_in_train_mode": {"eval_body": (
        '    total_loss, correct, seen = 0.0, 0, 0\n'
        '    with torch.no_grad():\n'
        '        for inputs, labels in loader:\n'
        '            inputs, labels = inputs.to(device), labels.to(device)\n'
        '            logits = model(inputs)\n'
        '            total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)\n'
        '            correct += (logits.argmax(dim=1) == labels).sum().item()\n'
        '            seen += labels.size(0)\n'
        '    return total_loss / max(1, seen), correct / max(1, seen)')},
    "eval_set_too_small": {"config": {"val_size": 8}},
    "loss_not_averaged": {"loss_body": (
        '                loss = F.cross_entropy(logits, labels, reduction="sum")')},
    "loss_constant_zero": {"loss_body": (
        '                # Every label is masked out, so the reduction is over nothing.\n'
        '                mask = labels < 0\n'
        '                loss = (loss * mask).sum() / mask.sum().clamp(min=1)')},
    "metric_off_by_one_epoch": {"checkpoint_body": (
        '        # Validation for THIS epoch is written against the previous index.\n'
        '        history.setdefault("val_loss", []).append(val_loss)\n'
        '        log.info("logged val_loss for epoch %d", epoch - 1)')},
    "metric_wrong_axis": {"eval_body": (
        '    model.eval()\n'
        '    total_loss, correct, seen = 0.0, 0, 0\n'
        '    with torch.no_grad():\n'
        '        for inputs, labels in loader:\n'
        '            inputs, labels = inputs.to(device), labels.to(device)\n'
        '            logits = model(inputs)\n'
        '            total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)\n'
        '            correct += (logits.argmax(dim=0) == labels).sum().item()\n'
        '            seen += labels.size(0)\n'
        '    return total_loss / max(1, seen), correct / max(1, seen)')},
    "tokenizer_mismatch": {"config": {"vocab_size": 50000, "tokenizer": "bpe-32k"},
                           "label_body": (
                               '    # Tokenizer vocabulary and the embedding table disagree.\n'
                               '    labels = labels.clamp(max=CONFIG["vocab_size"] - 1)')},
    "sequence_truncation": {"config": {"max_length": 128, "observed_p99_length": 1024}},
    "padding_fraction_growth": {"config": {"bucket_by_length": False, "max_length": 1024}},
    "oov_rate_growth": {"config": {"tokenizer": "bpe-32k", "shards": "v2 (retokenised)"}},
    "class_imbalance_collapse": {"config": {"class_weights": None, "majority_share": 0.95}},
    "seed_not_fixed": {"seed_body": (
        '    # Only Python\'s RNG is seeded; numpy and torch are left to the clock.\n'
        '    random.seed(seed)')},
    "silent_dtype_downcast": {"loss_body": (
        '                loss = loss.half().float()')},
    "loss_scale_collapse": {"config": {"amp": True, "init_scale": 65536.0}},
    "momentum_dead": {"grad_body": (
        '            scaler.unscale_(optimizer)\n'
        '            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_norm"])\n'
        '            # Wipes the optimizer state every step.\n'
        '            optimizer.state.clear()')},
    "norm_collapse": {"config": {"weight_decay": 2.0}},
    "dead_relu": {"config": {"lr": 0.5}},
    "checkpoint_not_saving": {"checkpoint_body": (
        '        if epoch % CONFIG["save_every"] == 0 and CONFIG["checkpoint_dir"]:\n'
        '            pass  # TODO: torch.save(...)'),
        "config": {"save_every": 5}},
    "kl_collapse": {"loss_body": (
        '                kl = kl_divergence(posterior, prior).mean()\n'
        '                loss = loss + CONFIG["kl_weight"] * kl'),
        "config": {"kl_weight": 0.0001}},
    "aux_loss_dominates": {"loss_body": (
        '                aux = auxiliary_penalty(model)\n'
        '                loss = loss + CONFIG["aux_weight"] * aux'),
        "config": {"aux_weight": 100.0}},
    "distill_teacher_ignored": {"loss_body": (
        '                with torch.no_grad():\n'
        '                    teacher_logits = teacher(inputs)\n'
        '                kl = F.kl_div(F.log_softmax(logits, -1),\n'
        '                              F.softmax(teacher_logits, -1), reduction="batchmean")\n'
        '                loss = loss + CONFIG["distill_weight"] * kl'),
        "config": {"distill_weight": 0.0}},
    "gradient_not_synced": {"config": {"world_size": 64, "grad_average_divisor": 8}},
    "moe_load_imbalance": {"config": {"num_experts": 8, "capacity_factor": 1.0,
                                      "load_balance_weight": 0.0}},
    "moe_expert_collapse": {"config": {"num_experts": 8, "router_z_loss": 0.0,
                                       "load_balance_weight": 0.0}},
    "temperature_collapse": {"config": {"learnable_temperature": True,
                                        "temperature_min": None}},
    "catastrophic_forgetting": {"config": {"replay_fraction": 0.0,
                                           "finetune_task": "task_b"}},
    "reward_hacking": {"config": {"reward_shaping": "terminal_only",
                                  "episode_length_penalty": 0.0}},
}


def script_for(family, name):
    """The training script for this scenario, with its fault in it where there is one."""
    parts = dict(PARTS)
    config = dict(DEFAULT_CONFIG)
    config["clip_norm"] = 1.0
    config["checkpoint_dir"] = "./checkpoints"

    fault = CODE_FAULTS.get(family, {})
    config.update(fault.get("config") or {})
    for key, value in fault.items():
        if key != "config":
            parts[key] = value

    return BASE.format(title=f"{family} experiment", name=name,
                       config=json.dumps(config, indent=4).replace("null", "None")
                                                          .replace("true", "True")
                                                          .replace("false", "False"),
                       **parts)


def write_project(directory, case, histories):
    """Lay out a directory that looks like the run this scenario describes."""
    os.makedirs(directory, exist_ok=True)
    family, name = case["family"], case["name"]

    with open(os.path.join(directory, "train.py"), "w", encoding="utf-8") as handle:
        handle.write(script_for(family, name))

    config = dict(DEFAULT_CONFIG)
    config.update((CODE_FAULTS.get(family, {}).get("config") or {}))
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    # Full resolution, every reading, in the order they happened.
    columns = sorted(histories)
    with open(os.path.join(directory, "metrics.csv"), "w", encoding="utf-8") as handle:
        handle.write("epoch," + ",".join(columns) + "\n")
        length = max(len(v) for v in histories.values())
        for i in range(length):
            row = [str(i)]
            for column in columns:
                series = histories[column]
                row.append(repr(series[i]) if i < len(series) else "")
            handle.write(",".join(row) + "\n")

    os.makedirs(os.path.join(directory, "checkpoints"), exist_ok=True)
    with open(os.path.join(directory, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# {name}\n\nRun launched with `python train.py`.\n"
                     f"Metrics are appended to metrics.csv each epoch.\n")
    return directory
