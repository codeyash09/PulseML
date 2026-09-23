"""Labelled training curves for benchmarking Pulse's detection.

Each scenario is a run replayed epoch by epoch into the engine. `expect` is the
check that should raise (None = nothing should raise: these are the false-alarm
tests, and they matter as much as the positives -- a detector that fires on a
healthy run gets turned off).

Curves are deterministic (seeded) so a result is reproducible.
"""
import math

import numpy as np

EPOCHS = 60


# Every curve's noise is seeded so a result is reproducible. Shifting this offset
# redraws all of them, which is how the sweep checks that a threshold was fitted to
# the shape of a run and not to one lucky draw of its noise.
SEED_OFFSET = 0


def set_seed_offset(offset):
    global SEED_OFFSET
    SEED_OFFSET = int(offset)


def _rng(seed):
    return np.random.default_rng(seed + SEED_OFFSET * 1009)


def descending(n=EPOCHS, start=2.0, floor=0.05, noise=0.01, seed=0):
    """A healthy loss: fast early progress, slowing, small noise."""
    r = _rng(seed)
    return [float(floor + (start - floor) * math.exp(-3.0 * i / n) + r.normal(0, noise)) for i in range(n)]


def rising(n=EPOCHS, start=0.5, rate=0.03, seed=1):
    r = _rng(seed)
    return [float(start * (1 + rate) ** i + r.normal(0, 0.01)) for i in range(n)]


def flat(n=EPOCHS, value=0.693, noise=0.0005, seed=2):
    r = _rng(seed)
    return [float(value + r.normal(0, noise)) for i in range(n)]


def accuracy_from(loss, chance=0.5, best=0.97):
    lo, hi = min(loss), max(loss)
    span = max(hi - lo, 1e-9)
    return [float(chance + (best - chance) * (hi - v) / span) for v in loss]


# --------------------------------------------------------------------------- faults


def nan_midrun():
    loss = descending()
    loss[25:] = [float("nan")] * (len(loss) - 25)
    return {"loss": loss}, None


def inf_midrun():
    loss = descending()
    loss[30:] = [float("inf")] * (len(loss) - 30)
    return {"loss": loss}, None


def exploding_loss():
    loss = descending()[:20] + [float(0.3 * (2.2 ** i)) for i in range(40)]
    return {"loss": loss}, None


def diverging_slowly():
    return {"loss": rising(start=0.6, rate=0.02)}, None


def frozen_from_start():
    """Dropout(1.0), a zeroed initializer: the loss never moves at all."""
    loss = flat()
    return {"loss": loss, "accuracy": [0.5] * EPOCHS}, None


def learns_then_stops():
    """Real progress for 15 epochs, then nothing for 45 at a bad value."""
    head = descending(n=15, start=2.0, floor=1.2)
    return {"loss": head + flat(n=45, value=1.2, noise=0.001, seed=7)}, None


def gradient_explosion():
    return {"loss": descending(), "grad_norm": [float(1.0 * (1.6 ** i)) for i in range(EPOCHS)]}, None


def gradient_vanishing():
    return {"loss": flat(value=0.69),
            "grad_norm": [float(max(1e-12, 0.5 * (0.5 ** i))) for i in range(EPOCHS)]}, None


def lr_jumped_up():
    lr = [0.001] * 30 + [0.5] * 30
    return {"loss": descending()[:30] + rising(n=30, start=0.4, rate=0.05), "lr": lr}, None


def overfitting_classic():
    train = descending(start=1.5, floor=0.01)
    val = [float(0.6 + 0.02 * max(0, i - 20)) for i in range(EPOCHS)]
    return {"loss": train, "val_loss": val}, None


def validation_regression():
    val = [float(0.5 + 0.01 * i) for i in range(EPOCHS)]
    return {"loss": descending(), "val_loss": val}, None


def oscillating_loss():
    r = _rng(11)
    return {"loss": [float(1.0 + 0.6 * math.sin(i * 2.4) + r.normal(0, 0.05)) for i in range(EPOCHS)]}, None


def accuracy_at_chance():
    return {"loss": flat(value=0.693), "accuracy": [0.5] * EPOCHS, "val_accuracy": [0.502] * EPOCHS}, None


def perfect_validation():
    """A leak: validation is perfect from the first epoch."""
    return {"loss": descending(), "val_accuracy": [1.0] * EPOCHS, "accuracy": accuracy_from(descending())}, None


def nan_weights():
    return {"loss": descending()}, {"dense/kernel": {"nan": 12, "inf": 0, "shape": [64, 32]}}


def slow_crawl():
    """Technically improving, far slower than it should: 2.0 -> 1.94 in 60 epochs."""
    return {"loss": [float(2.0 - 0.001 * i) for i in range(EPOCHS)]}, None


def loss_spike_sustained():
    """A spike that does not recover -- divergence, not one bad batch."""
    return {"loss": descending()[:25] + [float(9.0 + 0.2 * i) for i in range(35)]}, None


def metric_frozen_while_loss_moves():
    return {"loss": descending(), "accuracy": [0.62] * EPOCHS}, None


# --------------------------------------------------------------------------- healthy


def healthy_smooth():
    loss = descending()
    return {"loss": loss, "val_loss": descending(seed=3, floor=0.08),
            "accuracy": accuracy_from(loss), "val_accuracy": accuracy_from(loss, best=0.95)}, None


def healthy_noisy():
    """Small dataset: the same descent with a lot of noise on it."""
    loss = descending(noise=0.12, seed=4)
    return {"loss": loss, "val_loss": descending(noise=0.18, seed=5, floor=0.1)}, None


def healthy_converged_flat():
    """Descends, then sits at a good value for 40 epochs. The classic false alarm."""
    return {"loss": descending(n=20, start=2.0, floor=0.02) + flat(n=40, value=0.02, noise=0.001, seed=6)}, None


def healthy_warmup():
    """Learning-rate warmup: the loss rises for the first few epochs by design."""
    return {"loss": [0.9, 1.1, 1.3, 1.25] + descending(n=56, start=1.2, floor=0.05)}, None


def healthy_lr_schedule():
    """A step schedule drops the learning rate 10x at epoch 30 -- deliberate."""
    lr = [0.01] * 30 + [0.001] * 30
    return {"loss": descending(), "lr": lr}, None


def healthy_transient_spike():
    """One bad batch at epoch 30, recovered by the next epoch."""
    loss = descending()
    loss[30] = loss[30] * 6
    return {"loss": loss}, None


def healthy_finetune():
    """Fine-tuning: already good, improving by tiny amounts."""
    return {"loss": [float(0.050 - 0.00015 * i) for i in range(EPOCHS)],
            "accuracy": [float(0.970 + 0.0002 * i) for i in range(EPOCHS)]}, None


def healthy_short_run():
    loss = descending(n=8)
    return {"loss": loss, "val_loss": descending(n=8, seed=8, floor=0.1)}, None


def healthy_noisy_validation():
    """Validation on a small split bounces around while training improves."""
    r = _rng(9)
    val = [float(0.4 + r.normal(0, 0.12)) for _ in range(EPOCHS)]
    return {"loss": descending(), "val_loss": val}, None


def healthy_gan_oscillation():
    """Adversarial training: the generator loss legitimately oscillates."""
    r = _rng(10)
    return {"g_loss": [float(1.2 + 0.35 * math.sin(i / 2.0) + r.normal(0, 0.05)) for i in range(EPOCHS)],
            "d_loss": [float(0.7 + 0.2 * math.cos(i / 2.0) + r.normal(0, 0.05)) for i in range(EPOCHS)]}, None


def healthy_late_breakthrough():
    """Flat for 30 epochs, then learns (grokking). Flagging the flat part is
    defensible, so this is scored separately rather than counted as a failure."""
    return {"loss": flat(n=30, value=0.69, noise=0.002) + descending(n=30, start=0.69, floor=0.05)}, None


def healthy_perfect_easy_task():
    """A genuinely easy task: accuracy really is 1.0. Pulse warns here by
    design ('suspiciously perfect'), so this is scored separately too."""
    return {"loss": descending(floor=0.001), "val_accuracy": [1.0] * EPOCHS}, None


def healthy_step_level():
    """Per-batch values rather than per-epoch: noisier, many more points."""
    r = _rng(12)
    return {"loss": [float(2.0 * math.exp(-3.0 * i / 600) + abs(r.normal(0, 0.25))) for i in range(600)]}, None


def healthy_custom_names():
    """Nothing is called 'loss': does the engine still recognise the run?"""
    loss = descending()
    return {"train_objective": loss, "eval_objective": descending(seed=13, floor=0.09),
            "dice": accuracy_from(loss, chance=0.3, best=0.9)}, None


def healthy_missing_values():
    """Gaps in the history: a metric only evaluated every few epochs."""
    loss = descending()
    val = [descending(seed=14, floor=0.08)[i] if i % 5 == 0 else None for i in range(EPOCHS)]
    return {"loss": loss, "val_loss": val}, None


def healthy_tiny_values():
    return {"loss": [float(1e-7 * math.exp(-i / 20)) for i in range(EPOCHS)]}, None


def healthy_large_values():
    return {"loss": [float(5e6 * math.exp(-3.0 * i / EPOCHS)) for i in range(EPOCHS)]}, None


# --------------------------------------------------------------------------- faults, round 2
# Failure modes a real run hits that the first 18 did not cover. Several of these are
# here precisely because it was not clear any check would catch them.


def loss_collapses_to_zero():
    """Loss hits exactly 0 and stays: a collapsed model, or a target equal to the input."""
    return {"loss": descending(n=15, start=1.5, floor=0.4) + [0.0] * 45}, None


def late_data_leak():
    """The leak is not obvious in the first epochs -- accuracy creeps to 1.0 by epoch 12."""
    acc = [min(1.0, 0.55 + 0.04 * i) for i in range(EPOCHS)]
    return {"loss": descending(start=0.9, floor=1e-5), "accuracy": acc,
            "val_accuracy": [min(1.0, 0.56 + 0.04 * i) for i in range(EPOCHS)]}, None


def lr_decayed_to_zero():
    """A schedule bug drives the learning rate to 0 at epoch 20; learning stops dead."""
    lr = [0.01 * (0.5 ** i) for i in range(20)] + [0.0] * 40
    head = descending(n=20, start=2.0, floor=0.8)
    return {"loss": head + flat(n=40, value=head[-1], noise=0.0008, seed=21), "lr": lr}, None


def lr_zero_from_the_start():
    return {"loss": flat(value=2.3026, noise=0.0005, seed=22), "lr": [0.0] * EPOCHS}, None


def zero_gradient_norm():
    """requires_grad=False, or a detached graph: nothing flows back at all."""
    return {"loss": flat(value=0.693, noise=0.0004, seed=23), "grad_norm": [0.0] * EPOCHS}, None


def validation_only_nan():
    """Train is fine; validation goes NaN -- an empty split, or a metric dividing by zero."""
    return {"loss": descending(), "val_loss": descending(seed=24)[:20] + [float("nan")] * 40}, None


def one_metric_nan():
    return {"loss": descending(), "accuracy": accuracy_from(descending())[:25] + [float("nan")] * 35}, None


def accuracy_collapses_to_prior():
    """The model stops predicting anything but the majority class."""
    return {"loss": descending(n=20, start=0.9, floor=0.5) + flat(n=40, value=0.68, noise=0.002, seed=25),
            "accuracy": [0.82 - 0.02 * i if i < 14 else 0.55 for i in range(EPOCHS)]}, None


def accuracy_at_chance_while_loss_falls():
    """Shuffled labels: the loss comes down by memorising, accuracy never moves."""
    return {"loss": descending(start=2.3, floor=0.2), "accuracy": [0.1 + 0.001 * (i % 3) for i in range(EPOCHS)]}, None


def loss_falls_past_zero_forever():
    """Falls without bound past zero. A sign error looks exactly like this -- and so
    does a perfectly healthy density model, whose negative log-likelihood really is
    unbounded below, so this one is reported rather than scored."""
    return {"loss": [0.8 - 0.12 * i for i in range(EPOCHS)]}, None


def negative_loss_climbing():
    """A negative objective going the wrong way. Every ratio test in the detector
    changes direction when the sign does, so this is a fault that positive-only
    reasoning cannot see."""
    r = _rng(46)
    return {"loss": [float(-4.0 + 0.05 * i + r.normal(0, 0.02)) for i in range(EPOCHS)]}, None


def negative_loss_frozen():
    """An ELBO parked at the same value from the first epoch to the last."""
    return {"elbo_loss": flat(value=-3.2, noise=0.0008, seed=47)}, None


def oscillation_growing():
    """Instability setting in: the swings get bigger every epoch."""
    r = _rng(26)
    return {"loss": [float(1.0 + (0.02 * i) * math.sin(i * 2.4) + r.normal(0, 0.01)) for i in range(EPOCHS)]}, None


def weight_norm_unbounded():
    return {"loss": descending(floor=0.3), "weight_norm": [float(5.0 + 2.0 * i) for i in range(EPOCHS)]}, None


def broken_metric_perfect_but_loss_high():
    """accuracy == 1.0 while the loss says otherwise: the metric is measuring nothing."""
    return {"loss": flat(value=2.3026, noise=0.001, seed=27), "accuracy": [1.0] * EPOCHS}, None


def dataloader_exhausted():
    """The same batch over and over: the metrics repeat an identical short cycle."""
    cycle = [0.61, 0.58, 0.63, 0.59, 0.60]
    return {"loss": descending(n=10, start=1.2, floor=0.6) + cycle * 10}, None


def train_val_gap_enormous():
    """Train 0.01, validation 5.0 and flat: an eval-mode or normalisation bug."""
    return {"loss": descending(start=1.2, floor=0.01),
            "val_loss": flat(value=5.0, noise=0.01, seed=28)}, None


def loss_resets_every_epoch():
    """A state reset bug: each epoch starts over from the same value."""
    values = []
    for _ in range(12):
        values.extend([2.0, 1.7, 1.5, 1.4, 1.35])
    return {"loss": values}, None


def stuck_at_uniform_prediction():
    """Loss parked at ln(10): the model predicts a uniform distribution over 10 classes."""
    return {"loss": descending(n=8, start=2.9, floor=2.3026)
                    + flat(n=52, value=2.3026, noise=0.0006, seed=29),
            "accuracy": [0.1] * EPOCHS}, None


def diverges_after_real_progress():
    """Thirty good epochs, then it comes apart -- the hardest kind to catch late."""
    return {"loss": descending(n=30, start=2.0, floor=0.2)
                    + [float(0.2 * (1.09 ** i)) for i in range(30)]}, None


def validation_worse_from_the_first_epoch():
    return {"loss": descending(), "val_loss": [float(0.7 + 0.03 * i) for i in range(EPOCHS)]}, None


def step_time_growing():
    """A leak: every step takes longer than the last. Not a loss, not a metric."""
    return {"loss": descending(), "step_time": [float(0.1 + 0.01 * i) for i in range(EPOCHS)]}, None


# --------------------------------------------------------------------------- healthy, round 2


def healthy_cyclical_lr():
    """OneCycle/SGDR: the learning rate is supposed to jump around."""
    lr = [float(0.001 + 0.009 * abs(math.sin(i * math.pi / 10))) for i in range(EPOCHS)]
    return {"loss": descending(), "lr": lr}, None


def healthy_warm_restarts():
    """SGDR: the loss jumps up at each restart and then goes lower than before."""
    values = []
    for cycle, start in enumerate((2.0, 1.2, 0.7)):
        values.extend(descending(n=20, start=start, floor=start * 0.35, seed=30 + cycle))
    return {"loss": values}, None


def healthy_amp_loss_scale():
    """Mixed precision: the reported loss is scaled by 65536. Big, and perfectly fine."""
    return {"loss": [float(v * 65536.0) for v in descending(seed=33)]}, None


def healthy_curriculum():
    """The data gets harder at epoch 30 by design, so the loss rises before falling again."""
    return {"loss": descending(n=30, start=1.5, floor=0.3)
                    + descending(n=30, start=1.1, floor=0.15, seed=34)}, None


def healthy_multi_task():
    """Four losses on wildly different scales, all of them healthy."""
    return {"loss": descending(seed=35), "bbox_loss": [float(v * 120) for v in descending(seed=36)],
            "cls_loss": [float(v * 0.004) for v in descending(seed=37)],
            "mask_loss": descending(seed=38, start=0.9, floor=0.2)}, None


def healthy_rl_run():
    """Reinforcement learning: reward climbs, and the 'loss' means very little."""
    r = _rng(39)
    return {"reward": [float(10 + 2.0 * i + r.normal(0, 3)) for i in range(EPOCHS)],
            "policy_loss": [float(0.3 * math.sin(i / 3.0) + r.normal(0, 0.08)) for i in range(EPOCHS)],
            "value_loss": descending(seed=40, start=5.0, floor=1.2)}, None


def healthy_val_better_than_train():
    """Dropout and augmentation are on for training only, so validation looks better."""
    train = descending(start=1.4, floor=0.30)
    return {"loss": train, "val_loss": [float(v * 0.75) for v in descending(seed=41, start=1.4, floor=0.30)]}, None


def healthy_imbalanced_high_accuracy():
    """95% of the labels are one class, so accuracy starts high. The loss still improves."""
    return {"loss": descending(start=0.4, floor=0.05, seed=42),
            "accuracy": [float(min(0.985, 0.95 + 0.0008 * i)) for i in range(EPOCHS)]}, None


def healthy_epoch_boundary_spikes():
    """Per-step loss with a spike at each epoch boundary, where the data reshuffles."""
    r = _rng(43)
    values = []
    for step in range(400):
        base = 2.0 * math.exp(-3.0 * step / 400) + abs(r.normal(0, 0.06))
        values.append(float(base * (2.6 if step % 50 == 0 else 1.0)))
    return {"loss": values}, None


def healthy_short_early_stop():
    return {"loss": descending(n=12, start=1.4, floor=0.4),
            "val_loss": descending(n=12, start=1.5, floor=0.5, seed=44)}, None


def healthy_two_fit_calls():
    """Two models trained in one script: the history of the second starts high again."""
    return {"loss": descending(n=30, start=2.0, floor=0.1)
                    + descending(n=30, start=1.8, floor=0.08, seed=45)}, None


# name, builder, expected check (None = nothing should raise), tags
SCENARIOS = [
    # ---- must fire
    ("nan_midrun", nan_midrun, "nonfinite", "fault"),
    ("inf_midrun", inf_midrun, "nonfinite", "fault"),
    ("exploding_loss", exploding_loss, "loss_spike", "fault"),
    ("diverging_slowly", diverging_slowly, "any", "fault"),
    ("frozen_from_start", frozen_from_start, "any", "fault"),
    ("learns_then_stops", learns_then_stops, "any", "fault"),
    ("gradient_explosion", gradient_explosion, "norm_explosion", "fault"),
    ("gradient_vanishing", gradient_vanishing, "any", "fault"),
    ("lr_jumped_up", lr_jumped_up, "any", "fault"),
    ("overfitting_classic", overfitting_classic, "any", "fault"),
    ("validation_regression", validation_regression, "any", "fault"),
    ("oscillating_loss", oscillating_loss, "any", "fault"),
    ("accuracy_at_chance", accuracy_at_chance, "any", "fault"),
    ("perfect_validation", perfect_validation, "suspiciously_perfect", "fault"),
    ("nan_weights", nan_weights, "tensor_nonfinite", "fault"),
    ("slow_crawl", slow_crawl, "any", "fault"),
    ("loss_spike_sustained", loss_spike_sustained, "any", "fault"),
    ("metric_frozen_while_loss_moves", metric_frozen_while_loss_moves, "any", "fault"),
    # ---- must stay quiet
    ("healthy_smooth", healthy_smooth, None, "healthy"),
    ("healthy_noisy", healthy_noisy, None, "healthy"),
    ("healthy_converged_flat", healthy_converged_flat, None, "healthy"),
    ("healthy_warmup", healthy_warmup, None, "healthy"),
    ("healthy_lr_schedule", healthy_lr_schedule, None, "healthy"),
    ("healthy_transient_spike", healthy_transient_spike, None, "healthy"),
    ("healthy_finetune", healthy_finetune, None, "healthy"),
    ("healthy_short_run", healthy_short_run, None, "healthy"),
    ("healthy_noisy_validation", healthy_noisy_validation, None, "healthy"),
    ("healthy_gan_oscillation", healthy_gan_oscillation, None, "healthy"),
    ("healthy_step_level", healthy_step_level, None, "healthy"),
    ("healthy_custom_names", healthy_custom_names, None, "healthy"),
    ("healthy_missing_values", healthy_missing_values, None, "healthy"),
    ("healthy_tiny_values", healthy_tiny_values, None, "healthy"),
    ("healthy_large_values", healthy_large_values, None, "healthy"),
    # ---- round 2: must fire
    ("loss_collapses_to_zero", loss_collapses_to_zero, "any", "fault"),
    ("late_data_leak", late_data_leak, "any", "fault"),
    ("lr_decayed_to_zero", lr_decayed_to_zero, "any", "fault"),
    ("lr_zero_from_the_start", lr_zero_from_the_start, "any", "fault"),
    ("zero_gradient_norm", zero_gradient_norm, "any", "fault"),
    ("validation_only_nan", validation_only_nan, "nonfinite", "fault"),
    ("one_metric_nan", one_metric_nan, "nonfinite", "fault"),
    ("accuracy_collapses_to_prior", accuracy_collapses_to_prior, "any", "fault"),
    ("accuracy_at_chance_while_loss_falls", accuracy_at_chance_while_loss_falls, "any", "fault"),
    ("negative_loss_climbing", negative_loss_climbing, "any", "fault"),
    ("negative_loss_frozen", negative_loss_frozen, "any", "fault"),
    ("oscillation_growing", oscillation_growing, "any", "fault"),
    ("weight_norm_unbounded", weight_norm_unbounded, "any", "fault"),
    ("broken_metric_perfect_but_loss_high", broken_metric_perfect_but_loss_high, "any", "fault"),
    ("dataloader_exhausted", dataloader_exhausted, "any", "fault"),
    ("train_val_gap_enormous", train_val_gap_enormous, "any", "fault"),
    ("loss_resets_every_epoch", loss_resets_every_epoch, "any", "fault"),
    ("stuck_at_uniform_prediction", stuck_at_uniform_prediction, "any", "fault"),
    ("diverges_after_real_progress", diverges_after_real_progress, "any", "fault"),
    ("validation_worse_from_the_first_epoch", validation_worse_from_the_first_epoch, "any", "fault"),
    ("step_time_growing", step_time_growing, "any", "fault"),
    # ---- round 2: must stay quiet
    ("healthy_cyclical_lr", healthy_cyclical_lr, None, "healthy"),
    ("healthy_warm_restarts", healthy_warm_restarts, None, "healthy"),
    ("healthy_amp_loss_scale", healthy_amp_loss_scale, None, "healthy"),
    ("healthy_curriculum", healthy_curriculum, None, "healthy"),
    ("healthy_multi_task", healthy_multi_task, None, "healthy"),
    ("healthy_rl_run", healthy_rl_run, None, "healthy"),
    ("healthy_val_better_than_train", healthy_val_better_than_train, None, "healthy"),
    ("healthy_imbalanced_high_accuracy", healthy_imbalanced_high_accuracy, None, "healthy"),
    ("healthy_epoch_boundary_spikes", healthy_epoch_boundary_spikes, None, "healthy"),
    ("healthy_short_early_stop", healthy_short_early_stop, None, "healthy"),
    # ---- judgement calls: reported, not counted as failures either way
    ("healthy_late_breakthrough", healthy_late_breakthrough, None, "debatable"),
    ("healthy_perfect_easy_task", healthy_perfect_easy_task, None, "debatable"),
    ("healthy_two_fit_calls", healthy_two_fit_calls, None, "debatable"),
    ("loss_falls_past_zero_forever", loss_falls_past_zero_forever, "any", "debatable"),
]
