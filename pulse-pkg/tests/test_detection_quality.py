"""What the detector must and must not say, on runs whose answer we know.

test_detect.py checks the checks one at a time. This one asks the question a user
asks: given a training run, does Pulse call it correctly? Every curve here is
labelled -- broken runs must be caught, healthy runs must be left alone -- and both
halves matter, because a detector that interrupts healthy runs gets switched off and
then detects nothing at all.

The curves are the small version of the 37-scenario benchmark in detectbench; the
numbers in the assertions are that benchmark's result, so a change that regresses
detection quality fails here rather than six months later on someone's real run.
"""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pulse import pulse_detect as detect  # noqa: E402

EPOCHS = 60


def noise(seed, sigma):
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(400)]


def descending(n=EPOCHS, start=2.0, floor=0.05, sigma=0.01, seed=0):
    noises = noise(seed, sigma)
    return [floor + (start - floor) * math.exp(-3.0 * i / n) + noises[i] for i in range(n)]


def flat(n=EPOCHS, value=0.693, sigma=0.0005, seed=2):
    noises = noise(seed, sigma)
    return [value + noises[i] for i in range(n)]


# ------------------------------------------------------------------ broken runs
BROKEN = {
    "nan midrun": ({"loss": descending()[:25] + [float("nan")] * 35}, None),
    "inf midrun": ({"loss": descending()[:30] + [float("inf")] * 30}, None),
    "exploding": ({"loss": descending()[:20] + [0.3 * (2.2 ** i) for i in range(40)]}, None),
    "diverging slowly": ({"loss": [0.6 * (1.02 ** i) for i in range(EPOCHS)]}, None),
    "frozen from the start": ({"loss": flat(), "accuracy": [0.5] * EPOCHS}, None),
    "learns then stops": ({"loss": descending(n=15, start=2.0, floor=1.2)
                                   + flat(n=45, value=1.2, sigma=0.001, seed=7)}, None),
    "gradient explosion": ({"loss": descending(),
                            "grad_norm": [1.0 * (1.6 ** i) for i in range(EPOCHS)]}, None),
    "gradient vanishing": ({"loss": flat(value=0.69),
                            "grad_norm": [max(1e-12, 0.5 * (0.5 ** i)) for i in range(EPOCHS)]}, None),
    "learning rate jumped": ({"loss": descending()[:30] + [0.4 * (1.05 ** i) for i in range(30)],
                              "lr": [0.001] * 30 + [0.5] * 30}, None),
    "overfitting": ({"loss": descending(start=1.5, floor=0.01),
                     "val_loss": [0.6 + 0.02 * max(0, i - 20) for i in range(EPOCHS)]}, None),
    "validation regression": ({"loss": descending(),
                               "val_loss": [0.5 + 0.01 * i for i in range(EPOCHS)]}, None),
    "accuracy at chance": ({"loss": flat(value=0.693), "accuracy": [0.5] * EPOCHS}, None),
    "perfect validation": ({"loss": descending(), "val_accuracy": [1.0] * EPOCHS}, None),
    "nan weights": ({"loss": descending()}, {"dense/kernel": {"nan": 12, "inf": 0}}),
    "slow crawl": ({"loss": [2.0 - 0.001 * i for i in range(EPOCHS)]}, None),
    "sustained spike": ({"loss": descending()[:25] + [9.0 + 0.2 * i for i in range(35)]}, None),
    "metric frozen": ({"loss": descending(), "accuracy": [0.62] * EPOCHS}, None),
}

# ------------------------------------------------------------------ healthy runs
HEALTHY = {
    "smooth": {"loss": descending(), "val_loss": descending(seed=3, floor=0.08)},
    "noisy": {"loss": descending(sigma=0.12, seed=4),
              "val_loss": descending(sigma=0.18, seed=5, floor=0.1)},
    "converged and flat": {"loss": descending(n=20, start=2.0, floor=0.02)
                                   + flat(n=40, value=0.02, sigma=0.001, seed=6)},
    "warmup": {"loss": [0.9, 1.1, 1.3, 1.25] + descending(n=56, start=1.2, floor=0.05)},
    "learning rate schedule": {"loss": descending(), "lr": [0.01] * 30 + [0.001] * 30},
    "one transient spike": {"loss": [v * 6 if i == 30 else v
                                     for i, v in enumerate(descending())]},
    "fine-tuning": {"loss": [0.050 - 0.00015 * i for i in range(EPOCHS)],
                    "accuracy": [0.970 + 0.0002 * i for i in range(EPOCHS)]},
    "short run": {"loss": descending(n=8)},
    "noisy validation": {"loss": descending(),
                         "val_loss": [0.9 * math.exp(-0.9 * i / EPOCHS) + n
                                      for i, n in enumerate(noise(9, 0.12)[:EPOCHS])]},
    "adversarial (GAN)": {"g_loss": [1.2 + 0.35 * math.sin(i / 2.0) + n
                                     for i, n in enumerate(noise(10, 0.05)[:EPOCHS])],
                          "d_loss": [0.7 + 0.2 * math.cos(i / 2.0) + n
                                     for i, n in enumerate(noise(11, 0.05)[:EPOCHS])]},
    "step-level readings": {"loss": [2.0 * math.exp(-3.0 * i / 600) + abs(n)
                                     for i, n in enumerate(noise(12, 0.25) * 2)][:600]},
    "custom metric names": {"train_objective": descending(),
                            "eval_objective": descending(seed=13, floor=0.09)},
    "gaps in the history": {"loss": descending(),
                            "val_loss": [descending(seed=14, floor=0.08)[i] if i % 5 == 0 else None
                                         for i in range(EPOCHS)]},
    "tiny values": {"loss": [1e-7 * math.exp(-i / 20) for i in range(EPOCHS)]},
    "large values": {"loss": [5e6 * math.exp(-3.0 * i / EPOCHS) for i in range(EPOCHS)]},
}


def replay(histories, tensor_stats=None, sensitivity=0.3):
    """Feed a run to a fresh engine one reading at a time, as the monitor does."""
    engine = detect.DetectionEngine(sensitivity=sensitivity, confirmations=2)
    length = max(len(v) for v in histories.values())
    actionable = []
    for step in range(1, length + 1):
        window = {name: values[:step] for name, values in histories.items()}
        stats = tensor_stats if (tensor_stats and step > 3) else None
        for finding in engine.update(window, step=step, tensor_stats=stats)["raised"]:
            if finding.severity in (detect.CRITICAL, detect.WARNING):
                actionable.append(finding)
    return actionable


@pytest.mark.parametrize("label", sorted(BROKEN))
def test_broken_runs_are_caught(label):
    histories, tensor_stats = BROKEN[label]
    found = replay(histories, tensor_stats)
    assert found, "%s went undetected for the whole run" % label


@pytest.mark.parametrize("label", sorted(HEALTHY))
def test_healthy_runs_are_left_alone(label):
    found = replay(HEALTHY[label])
    assert not found, "%s was interrupted: %s" % (label, found[0].message if found else "")


def test_sensitivity_trades_recall_for_quiet_in_the_right_direction():
    """Turning the dial up must not lose detections, and may cost false alarms."""
    for sensitivity in (0.1, 0.3, 0.5, 0.7):
        caught = sum(1 for label in BROKEN
                     if replay(BROKEN[label][0], BROKEN[label][1], sensitivity))
        assert caught == len(BROKEN), "sensitivity %.1f missed %d broken runs" % (
            sensitivity, len(BROKEN) - caught)


def test_info_findings_do_not_interrupt_training():
    """Only critical and warning findings are worth pausing a run for."""
    engine = detect.DetectionEngine(sensitivity=0.3, confirmations=2)
    raised = []
    values = [0.970 + 0.0002 * i for i in range(EPOCHS)]
    for step in range(1, EPOCHS + 1):
        raised += engine.update({"accuracy": values[:step]}, step=step)["raised"]
    assert raised, "a metric that never moves is still worth recording"
    assert all(f.severity == detect.INFO for f in raised), \
        "a fine-tune holding 97% accuracy must not be raised as a problem"
