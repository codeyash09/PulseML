"""Where does Pulse's detection stop working?

run_detectbench.py asks whether Pulse catches a broken run. It answers yes, 39/39, at
every sensitivity -- which tells us the cases are all comfortably inside the envelope
and nothing about where the envelope ends.

This asks the harder question. Every failure family here is generated as a *ladder*: the
same fault, from blatant down to nearly invisible, holding everything else fixed. A
family's score is not "caught / missed" but the rung it stops at -- the weakest version
of that fault Pulse still sees. That is the number that tells you whether a real run
would be caught, because real faults do not arrive at benchmark strength.

The ladders are built so that the *only* thing changing is the fault's strength. Noise,
length, starting loss and the healthy part of the curve are identical across a family's
rungs, so a difference in outcome is a difference in detectability and not in luck.

Healthy runs are scored here too, and a false alarm counts exactly like a miss: a
detector that cries wolf on a good run gets turned off, and then it detects nothing.
The healthy set is deliberately adversarial -- warmup bumps, cosine restarts, grokking,
curriculum steps -- because those are the shapes that look like faults and are not.

    python3 limits.py                     # the whole thing
    python3 limits.py --family divergence_rate
    python3 limits.py --sensitivity 0.5 --json out.json
"""
import math

import numpy as np

EPOCHS = 60
SEED_OFFSET = 0


_RNG_CACHE = {}


def set_seed_offset(offset):
    global SEED_OFFSET
    SEED_OFFSET = int(offset)
    _RNG_CACHE.clear()


def reset_rngs():
    """Start every case from the same draw. The runner calls this before each builder,
    so a case is reproducible on its own and independent of what ran before it."""
    _RNG_CACHE.clear()


def _rng(seed):
    """One generator per seed, reused across calls so successive draws advance it.

    This used to build a fresh generator every call, which meant `_rng(5).normal(...)`
    inside a loop returned the *same* number on every iteration: what looked like noise
    was a constant offset, and curves meant to be noisy were byte-identical. Two healthy
    fixtures then tripped the `frozen` check and were scored as false alarms against
    Pulse, which was correct behaviour on the data it was given.
    """
    key = (seed, SEED_OFFSET)
    if key not in _RNG_CACHE:
        _RNG_CACHE[key] = np.random.default_rng(seed + SEED_OFFSET * 1009)
    return _RNG_CACHE[key]


# --------------------------------------------------------------------------- shapes
# One healthy backbone, reused by every ladder, so that a fault is the only difference
# between a broken run and its healthy twin.

def healthy_loss(n=EPOCHS, start=2.0, floor=0.05, noise=0.01, seed=0):
    r = _rng(seed)
    return [float(floor + (start - floor) * math.exp(-3.0 * i / n) + r.normal(0, noise))
            for i in range(n)]


def healthy_val(train, gap=0.06, noise=0.012, seed=7):
    """Validation loss: a little above train, a little noisier. Normal, not overfitting."""
    r = _rng(seed)
    return [float(v + gap + r.normal(0, noise)) for v in train]


def accuracy_from(loss, chance=0.5, best=0.97):
    lo, hi = min(loss), max(loss)
    span = max(hi - lo, 1e-9)
    return [float(chance + (best - chance) * (hi - v) / span) for v in loss]


def grad_norm_for(loss, scale=0.5, noise=0.05, seed=11):
    r = _rng(seed)
    return [float(max(1e-9, v * scale + r.normal(0, noise * scale))) for v in loss]


CASES = []


def case(name, family, rung, expect, tag):
    """Register one run. `rung` orders a family from easiest to hardest to see."""
    def register(builder):
        CASES.append({"name": name, "family": family, "rung": rung,
                      "expect": expect, "tag": tag, "builder": builder})
        return builder
    return register


def ladder(family, rungs, expect, tag="fault", label=None):
    """Register one case per rung. `rungs` is [(label_value, difficulty_index), ...]."""
    def register(build):
        for value, index in rungs:
            name = f"{family}[{label(value) if label else value}]"
            CASES.append({"name": name, "family": family, "rung": index,
                          "expect": expect, "tag": tag,
                          "builder": (lambda v=value: build(v))})
        return build
    return register


def rungs(values):
    """Hardest last: index 0 is the blatant version."""
    return [(v, i) for i, v in enumerate(values)]


# =========================================================================== faults
# Each ladder walks one knob from obvious to subtle. The comment on each says what the
# hardest rung corresponds to in a real run.

# --- divergence ------------------------------------------------------------------

@ladder("divergence_rate", rungs([0.50, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004]),
        expect="any")
def _divergence_rate(rate):
    """Loss turns around at epoch 20 and grows by `rate` per epoch.
    The last rung is 0.4%/epoch: a 25% worse loss over the back half of the run."""
    loss = healthy_loss()
    turn = 20
    base = loss[turn]
    for i in range(turn, EPOCHS):
        loss[i] = float(base * (1 + rate) ** (i - turn) + _rng(3).normal(0, 0.005))
    return {"loss": loss}, None


@ladder("divergence_onset", rungs([5, 15, 25, 35, 45, 50, 54, 57]), expect="any")
def _divergence_onset(onset):
    """A clear 12%/epoch divergence, but starting later and later: at the last rung
    there are only three epochs of evidence before the run ends."""
    loss = healthy_loss()
    base = loss[onset]
    for i in range(onset, EPOCHS):
        loss[i] = float(base * 1.12 ** (i - onset))
    return {"loss": loss}, None


@ladder("divergence_under_noise", rungs([0.005, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80]),
        expect="any")
def _divergence_under_noise(noise):
    """A fixed 6%/epoch divergence buried under growing noise. The fault never changes;
    only how well it hides. This is the signal-to-noise limit."""
    r = _rng(31)
    loss = healthy_loss(noise=0.0)
    turn = 20
    base = loss[turn]
    for i in range(turn, EPOCHS):
        loss[i] = float(base * 1.06 ** (i - turn))
    return {"loss": [float(v + r.normal(0, noise)) for v in loss]}, None


# --- overfitting -----------------------------------------------------------------

@ladder("overfit_gap", rungs([2.0, 1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015]), expect="any")
def _overfit_gap(gap):
    """Train keeps falling, val turns up at epoch 25 and ends `gap` above its best.
    The last rung is a 0.015 regression -- inside the noise of most val curves."""
    train = healthy_loss()
    val = healthy_val(train)
    turn = 25
    best = val[turn]
    for i in range(turn, EPOCHS):
        frac = (i - turn) / (EPOCHS - turn - 1)
        val[i] = float(best + gap * frac + _rng(5).normal(0, 0.008))
    return {"loss": train, "val_loss": val}, None


@ladder("overfit_flat_val", rungs([0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02]), expect="any")
def _overfit_flat_val(train_drop):
    """Val stops improving entirely while train keeps dropping by `train_drop` overall.
    No val regression at all -- only the widening gap says anything is wrong."""
    train = healthy_loss()
    val = healthy_val(train)
    turn = 25
    frozen = val[turn]
    r = _rng(6)
    for i in range(turn, EPOCHS):
        val[i] = float(frozen + r.normal(0, 0.01))
        train[i] = float(train[turn] - train_drop * (i - turn) / (EPOCHS - turn - 1))
    return {"loss": train, "val_loss": val}, None


@ladder("val_accuracy_regression", rungs([0.30, 0.18, 0.10, 0.05, 0.025, 0.012, 0.006]),
        expect="any")
def _val_accuracy_regression(drop):
    """Train accuracy climbs, val accuracy peaks at epoch 28 then falls by `drop`."""
    train = healthy_loss()
    acc = accuracy_from(train)
    val_acc = [float(a - 0.03) for a in acc]
    peak = 28
    top = val_acc[peak]
    for i in range(peak, EPOCHS):
        frac = (i - peak) / (EPOCHS - peak - 1)
        val_acc[i] = float(top - drop * frac + _rng(8).normal(0, 0.004))
    return {"loss": train, "accuracy": acc, "val_accuracy": val_acc}, None


# --- non-finite ------------------------------------------------------------------

@ladder("nan_onset", rungs([1, 5, 20, 40, 52, 57, 59]), expect="nonfinite")
def _nan_onset(onset):
    """Loss becomes NaN at `onset` and stays. The last rung leaves one NaN epoch."""
    loss = healthy_loss()
    for i in range(onset, EPOCHS):
        loss[i] = float("nan")
    return {"loss": loss}, None


@ladder("nan_intermittent", rungs([30, 15, 8, 4, 2, 1]), expect="nonfinite")
def _nan_intermittent(count):
    """`count` scattered NaN epochs in an otherwise healthy run -- a bad batch, not a
    dead run. One NaN in sixty is the hardest rung and the most realistic."""
    loss = healthy_loss()
    r = _rng(13)
    for i in r.choice(EPOCHS, size=count, replace=False):
        loss[int(i)] = float("nan")
    return {"loss": loss}, None


@ladder("inf_onset", rungs([10, 30, 45, 55, 58]), expect="nonfinite")
def _inf_onset(onset):
    loss = healthy_loss()
    for i in range(onset, EPOCHS):
        loss[i] = float("inf")
    return {"loss": loss}, None


@ladder("overflow_to_inf", rungs([1.8, 1.5, 1.35, 1.25, 1.18, 1.12]), expect="any")
def _overflow_to_inf(growth):
    """Loss grows geometrically until it overflows float64 and becomes inf. Slower
    growth reaches the overflow later, leaving fewer finite epochs to notice it."""
    loss = healthy_loss()[:15]
    v = loss[-1]
    for _ in range(EPOCHS - 15):
        v = v * growth if v < 1e308 else float("inf")
        loss.append(float(v))
    return {"loss": loss}, None


@ladder("nan_in_gradients", rungs([1, 5, 20, 40, 55]), expect="any")
def _nan_in_gradients(onset):
    """The loss stays finite and healthy-looking; the gradient norm is NaN. A run that
    looks fine on the metric everybody watches."""
    loss = healthy_loss()
    grad = grad_norm_for(loss)
    for i in range(onset, EPOCHS):
        grad[i] = float("nan")
    return {"loss": loss, "grad_norm": grad}, None


# --- gradients -------------------------------------------------------------------

@ladder("grad_vanish", rungs([1e-14, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]),
        expect="any")
def _grad_vanish(floor):
    """Gradient norm decays to `floor` by the end. Pulse's vanishing threshold is 1e-8,
    so this ladder walks straight across it."""
    loss = healthy_loss()
    start = 0.8
    grad = [float(start * (floor / start) ** (i / (EPOCHS - 1))) for i in range(EPOCHS)]
    # Loss stops improving once the gradients die.
    for i in range(EPOCHS):
        if grad[i] < 1e-6:
            loss[i] = loss[max(0, i - 1)]
    return {"loss": loss, "grad_norm": grad}, None


@ladder("grad_explode", rungs([1e12, 1e9, 1e6, 1e4, 1e3, 1e2, 30, 10]), expect="any")
def _grad_explode(peak):
    """Gradient norm climbs from 0.8 to `peak` over the back half."""
    loss = healthy_loss()
    grad = grad_norm_for(loss)
    start = 25
    base = grad[start]
    for i in range(start, EPOCHS):
        frac = (i - start) / (EPOCHS - start - 1)
        grad[i] = float(base * (peak / base) ** frac)
    return {"loss": loss, "grad_norm": grad}, None


@ladder("grad_cliff", rungs([1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 0.05, 0.2]), expect="any")
def _grad_cliff(after):
    """Gradient norm falls off a cliff in a single epoch -- a layer detached from the
    graph, or a frozen backbone -- and sits at `after` from then on."""
    loss = healthy_loss()
    grad = grad_norm_for(loss)
    for i in range(30, EPOCHS):
        grad[i] = float(after + _rng(17).normal(0, after * 0.02))
        loss[i] = loss[29]
    return {"loss": loss, "grad_norm": grad}, None


@ladder("dead_relu", rungs([0.0, 1e-6, 1e-4, 0.01, 0.05, 0.15]), expect="any")
def _dead_relu(activation):
    """Mean activation collapses to `activation`: the units stopped firing."""
    loss = healthy_loss()
    act = [float(0.4 - (0.4 - activation) * min(1.0, i / 25.0)) for i in range(EPOCHS)]
    for i in range(25, EPOCHS):
        loss[i] = loss[25]
    return {"loss": loss, "activation_mean": act, "grad_norm": grad_norm_for(loss)}, None


# --- stuck / not learning ---------------------------------------------------------

@ladder("plateau_length", rungs([55, 45, 35, 25, 18, 12, 8, 5]), expect="any")
def _plateau_length(length):
    """Healthy start, then flat for the last `length` epochs."""
    loss = healthy_loss()
    start = EPOCHS - length
    held = loss[start]
    for i in range(start, EPOCHS):
        loss[i] = float(held + _rng(19).normal(0, 0.0004))
    return {"loss": loss}, None


@ladder("stagnation_slope", rungs([0.0, 1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 6e-3, 1e-2]),
        expect="any")
def _stagnation_slope(slope):
    """Loss still falls, but by `slope` per epoch -- technically improving, in practice
    finished. The last rung is a real if unimpressive 0.6/epoch improvement."""
    loss = healthy_loss()[:20]
    v = loss[-1]
    for _ in range(EPOCHS - 20):
        v = max(0.0, v - slope)
        loss.append(float(v + _rng(23).normal(0, 0.0003)))
    return {"loss": loss}, None


@ladder("never_learned", rungs([1.0, 0.999, 0.995, 0.99, 0.97, 0.94, 0.90]), expect="any")
def _never_learned(ratio):
    """Loss ends at `ratio` of where it started: the model never learned anything.
    At 0.90 it improved by 10%, which is bad but not obviously a bug."""
    r = _rng(29)
    start = 0.693
    return {"loss": [float(start * (1 - (1 - ratio) * i / (EPOCHS - 1)) + r.normal(0, 0.002))
                     for i in range(EPOCHS)]}, None


@ladder("frozen_metric", rungs([60, 40, 25, 15, 10, 6, 4]), expect="any")
def _frozen_metric(length):
    """A metric computed once and reused: byte-identical for `length` epochs."""
    loss = healthy_loss()
    start = EPOCHS - length
    for i in range(start, EPOCHS):
        loss[i] = loss[start]
    return {"loss": loss}, None


@ladder("repeating_cycle", rungs([2, 3, 4, 6, 10, 15]), expect="any")
def _repeating_cycle(period):
    """The same `period` values repeat forever -- a data loader serving one batch."""
    loss = healthy_loss()
    cycle = loss[20:20 + period]
    for i in range(20, EPOCHS):
        loss[i] = cycle[(i - 20) % period]
    return {"loss": loss}, None


# --- instability ------------------------------------------------------------------

@ladder("oscillation_amp", rungs([2.0, 1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015]),
        expect="any")
def _oscillation_amp(amp):
    """Loss alternates up/down by `amp` around a flat trend: the learning rate is too
    high. Small amplitudes are indistinguishable from noise by eye."""
    loss = healthy_loss()
    for i in range(20, EPOCHS):
        loss[i] = float(loss[20] + amp * (1 if i % 2 else -1) + _rng(37).normal(0, 0.004))
    return {"loss": loss}, None


@ladder("loss_spike", rungs([1000, 100, 20, 8, 4, 2.5, 1.6, 1.25]), expect="any")
def _loss_spike(multiple):
    """One epoch `multiple` times the surrounding loss, then straight back to normal."""
    loss = healthy_loss()
    loss[35] = float(loss[35] * multiple)
    return {"loss": loss}, None


@ladder("sawtooth", rungs([0.8, 0.5, 0.3, 0.15, 0.08, 0.04]), expect="any")
def _sawtooth(amp):
    """Loss climbs for five epochs then resets -- an optimizer state bug, not a
    schedule. Distinguishing this from warm restarts is the hard part."""
    loss = healthy_loss()
    for i in range(20, EPOCHS):
        loss[i] = float(loss[20] + amp * ((i - 20) % 5) / 4.0)
    return {"loss": loss}, None


# --- learning rate ----------------------------------------------------------------

@ladder("lr_jump", rungs([1e4, 1e3, 100, 40, 20, 12, 8, 5]), expect="any")
def _lr_jump(ratio):
    """The learning rate is multiplied by `ratio` at epoch 30 -- a schedule that reset,
    or a config reloaded wrong."""
    loss = healthy_loss()
    lr = [0.001] * 30 + [0.001 * ratio] * (EPOCHS - 30)
    for i in range(30, EPOCHS):
        loss[i] = float(loss[i - 1] * (1 + 0.02 * math.log10(max(ratio, 1.1))))
    return {"loss": loss, "lr": lr}, None


@ladder("lr_to_zero", rungs([5, 12, 20, 30, 42, 52]), expect="any")
def _lr_to_zero(onset):
    """The LR schedule reaches zero at `onset` and training silently stops. The loss
    does not get worse -- it just stops moving, which is easy to read as convergence."""
    loss = healthy_loss()
    lr = [float(0.001 * max(0.0, 1 - i / onset)) for i in range(EPOCHS)]
    for i in range(onset, EPOCHS):
        loss[i] = float(loss[onset] + _rng(41).normal(0, 0.0003))
    return {"loss": loss, "lr": lr}, None


# --- weights ----------------------------------------------------------------------

@ladder("norm_explosion", rungs([1e9, 1e6, 1e4, 500, 100, 30, 10, 4]), expect="any")
def _norm_explosion(peak):
    """Weight norm grows from 10 to `peak`: no regularization, or a runaway residual."""
    loss = healthy_loss()
    start = 20
    norm = [10.0] * start
    for i in range(start, EPOCHS):
        frac = (i - start) / (EPOCHS - start - 1)
        norm.append(float(10.0 * (peak / 10.0) ** frac))
    return {"loss": loss, "weight_norm": norm}, None


@ladder("norm_collapse", rungs([0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1]), expect="any")
def _norm_collapse(floor):
    """Weight norm decays to `floor` -- weight decay far too strong, or a dying model."""
    loss = healthy_loss()
    norm = [float(max(floor, 10.0 * (max(floor, 1e-300) / 10.0) ** (i / (EPOCHS - 1))))
            for i in range(EPOCHS)]
    for i in range(30, EPOCHS):
        loss[i] = loss[30]
    return {"loss": loss, "weight_norm": norm}, None


# --- leakage and impossible values ------------------------------------------------

@ladder("suspiciously_perfect", rungs([1.0, 0.9999, 0.999, 0.998, 0.995, 0.99]),
        expect="any")
def _suspiciously_perfect(top):
    """Validation accuracy reaches `top` within a few epochs: the label is in the
    features. Perfect is obvious; 0.99 is just a good model, and that is the point."""
    acc = [float(min(top, 0.5 + (top - 0.5) * min(1.0, i / 6.0))) for i in range(EPOCHS)]
    loss = [float(max(1e-6, 2.0 * (1 - a))) for a in acc]
    return {"loss": loss, "val_accuracy": acc}, None


@ladder("val_equals_train", rungs([0.0, 1e-9, 1e-6, 1e-4, 1e-3, 0.01]), expect="any")
def _val_equals_train(jitter):
    """Validation is being computed on the training set: the two curves agree to
    `jitter`. At 0.01 that is merely a suspiciously small generalization gap."""
    train = healthy_loss()
    r = _rng(43)
    return {"loss": train,
            "val_loss": [float(v + r.normal(0, jitter)) for v in train]}, None


@ladder("impossible_value", rungs([1e300, 1e30, 100.0, 2.0, 1.05, 1.001]),
        expect="any")
def _impossible_value(top):
    """An accuracy above 1.0. Something is normalizing by the wrong count."""
    acc = accuracy_from(healthy_loss())
    for i in range(30, EPOCHS):
        acc[i] = float(top)
    return {"loss": healthy_loss(), "accuracy": acc}, None


@ladder("negative_loss", rungs([-1e6, -1e3, -10.0, -1.0, -0.1, -0.001]), expect="any")
def _negative_loss(floor):
    """Cross-entropy going negative: the labels or the log are wrong."""
    loss = healthy_loss()
    for i in range(30, EPOCHS):
        frac = (i - 30) / (EPOCHS - 31)
        loss[i] = float(loss[30] + (floor - loss[30]) * frac)
    return {"loss": loss}, None


# --- numerical precision ----------------------------------------------------------

@ladder("quantized_loss", rungs([2, 4, 8, 16, 64, 256]), expect="any")
def _quantized_loss(levels):
    """The loss only ever takes `levels` distinct values -- fp16 underflow, or a metric
    accumulated in the wrong dtype."""
    loss = healthy_loss()
    lo, hi = min(loss), max(loss)
    step = (hi - lo) / max(1, levels - 1)
    return {"loss": [float(lo + round((v - lo) / step) * step) for v in loss]}, None


@ladder("loss_scale_collapse", rungs([1.0, 16.0, 256.0, 4096.0, 16384.0]), expect="any")
def _loss_scale_collapse(floor):
    """The fp16 loss scaler keeps halving: every step is overflowing and being skipped,
    so the model is not actually training."""
    scale = [float(max(floor, 65536.0 / (2 ** (i / 3.0)))) for i in range(EPOCHS)]
    loss = healthy_loss()
    for i in range(20, EPOCHS):
        loss[i] = loss[20]
    return {"loss": loss, "loss_scale": scale}, None


# --- throughput and resources -----------------------------------------------------

@ladder("throughput_decay", rungs([0.0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.85]), expect="any")
def _throughput_decay(end_frac):
    """Tokens per second falls to `end_frac` of where it started: a leak, thermal
    throttling, or a dataloader that is falling behind."""
    loss = healthy_loss()
    tok = [float(12000 * (1 - (1 - end_frac) * i / (EPOCHS - 1))) for i in range(EPOCHS)]
    return {"loss": loss, "tokens_per_sec": tok}, None


@ladder("memory_growth", rungs([64.0, 16.0, 8.0, 4.0, 2.0, 1.4, 1.15]), expect="any")
def _memory_growth(multiple):
    """Allocated memory grows `multiple`x over the run: tensors retained every step."""
    loss = healthy_loss()
    mem = [float(4000 * multiple ** (i / (EPOCHS - 1))) for i in range(EPOCHS)]
    return {"loss": loss, "gpu_mem_mb": mem}, None


@ladder("step_time_growth", rungs([50.0, 10.0, 4.0, 2.0, 1.5, 1.2, 1.08]), expect="any")
def _step_time_growth(multiple):
    """Step time grows `multiple`x -- a list being appended to every iteration."""
    loss = healthy_loss()
    t = [float(0.25 * multiple ** (i / (EPOCHS - 1))) for i in range(EPOCHS)]
    return {"loss": loss, "step_time": t}, None


# --- short runs: the latency limit -------------------------------------------------

@ladder("short_run_nan", rungs([30, 20, 12, 8, 5, 4, 3, 2]), expect="nonfinite")
def _short_run_nan(length):
    """A run that is only `length` epochs long and goes NaN halfway. Detection that
    needs a long window cannot help a run that is about to be killed."""
    loss = healthy_loss(n=length)
    for i in range(length // 2, length):
        loss[i] = float("nan")
    return {"loss": loss}, None


@ladder("short_run_diverge", rungs([40, 25, 15, 10, 7, 5, 4]), expect="any")
def _short_run_diverge(length):
    """A short run diverging hard from its midpoint."""
    loss = healthy_loss(n=length)
    turn = length // 2
    base = loss[turn]
    for i in range(turn, length):
        loss[i] = float(base * 1.5 ** (i - turn))
    return {"loss": loss}, None


# --- many variables ----------------------------------------------------------------

@ladder("needle_in_haystack", rungs([1, 5, 20, 60, 150, 400]), expect="any")
def _needle_in_haystack(decoys):
    """One diverging metric among `decoys` healthy ones. Real runs log a lot; a
    detector that only looks at `loss` will miss a fault in `val_loss_task3`."""
    histories = {"loss": healthy_loss()}
    for d in range(decoys):
        histories[f"metric_{d}"] = healthy_loss(seed=100 + d, noise=0.02)
    broken = healthy_loss(seed=999)
    for i in range(20, EPOCHS):
        broken[i] = float(broken[20] * 1.15 ** (i - 20))
    histories["val_loss"] = broken
    return histories, None


# --- tensors -----------------------------------------------------------------------

@ladder("tensor_nonfinite", rungs([65536, 1024, 64, 4, 1]), expect="tensor_nonfinite")
def _tensor_nonfinite(count):
    """`count` NaN elements in a 256x256 weight. The real stats dict reports nan/inf as
    counts and computes mean/std over the finite values only, so a tensor that is 99%
    NaN still reports a perfectly ordinary mean -- the count is the only signal."""
    stats = {"layer1.weight": {"shape": [256, 256], "device": "cuda:0", "dtype": "float32",
                               "mean": 0.01, "std": 0.02, "min": -0.1, "max": 0.1,
                               "nan": count, "inf": 0}}
    return {"loss": healthy_loss()}, stats


@ladder("tensor_inf", rungs([4096, 256, 16, 1]), expect="tensor_nonfinite")
def _tensor_inf(count):
    stats = {"layer1.weight": {"shape": [256, 256], "device": "cuda:0", "dtype": "float32",
                               "mean": 0.01, "std": 0.02, "min": -0.1, "max": 0.1,
                               "nan": 0, "inf": count}}
    return {"loss": healthy_loss()}, stats


@ladder("tensor_norm_blowup", rungs([1e12, 1e8, 1e5, 1e3, 100.0, 20.0]), expect="any")
def _tensor_norm_blowup(scale):
    """Weight std grows to `scale` with no NaN in it. All finite, all enormous."""
    stats = {"layer1.weight": {"shape": [256, 256], "device": "cuda:0", "dtype": "float32",
                               "mean": 0.0, "std": scale, "min": -scale, "max": scale,
                               "nan": 0, "inf": 0}}
    return {"loss": healthy_loss()}, stats


# ========================================================================== healthy
# Shapes that look like faults and are not. A false alarm here is scored as a failure.

@case("healthy_textbook", "healthy", 0, None, "healthy")
def _healthy_textbook():
    train = healthy_loss()
    return {"loss": train, "val_loss": healthy_val(train),
            "accuracy": accuracy_from(train), "grad_norm": grad_norm_for(train)}, None


@case("healthy_noisy", "healthy", 0, None, "healthy")
def _healthy_noisy():
    return {"loss": healthy_loss(noise=0.06)}, None


@case("healthy_very_noisy", "healthy", 0, None, "healthy")
def _healthy_very_noisy():
    """Small-batch RL-style noise. Loud, but the trend is down."""
    return {"loss": healthy_loss(noise=0.15)}, None


@case("healthy_warmup_bump", "healthy", 0, None, "healthy")
def _healthy_warmup_bump():
    """Loss rises for the first five epochs while the LR warms up. Real, and normal."""
    loss = healthy_loss()
    for i in range(5):
        loss[i] = float(loss[0] * (1 + 0.15 * i))
    lr = [float(0.001 * min(1.0, (i + 1) / 5)) for i in range(EPOCHS)]
    return {"loss": loss, "lr": lr}, None


@case("healthy_cosine_restarts", "healthy", 0, None, "healthy")
def _healthy_cosine_restarts():
    """SGDR: the loss jumps at every restart and then beats its previous best."""
    loss = []
    for cycle in range(3):
        base = 2.0 * (0.45 ** cycle)
        for i in range(20):
            loss.append(float(base * math.exp(-2.5 * i / 20) + _rng(51).normal(0, 0.008)))
    lr = []
    for cycle in range(3):
        for i in range(20):
            lr.append(float(0.001 * 0.5 * (1 + math.cos(math.pi * i / 20))))
    return {"loss": loss, "lr": lr}, None


@case("healthy_grokking", "healthy", 0, None, "healthy")
def _healthy_grokking():
    """Flat for forty epochs, then it suddenly generalizes. Indistinguishable from a
    stuck run until the moment it is not."""
    val = [float(0.69 + _rng(53).normal(0, 0.003)) for _ in range(40)]
    val += [float(0.69 * math.exp(-0.5 * i) + 0.01) for i in range(EPOCHS - 40)]
    return {"loss": healthy_loss(), "val_loss": val}, None


@case("healthy_double_descent", "healthy", 0, None, "healthy")
def _healthy_double_descent():
    """Val gets worse in the middle and then better than before -- the shape
    overfitting detection is built to flag, on a run that is fine."""
    train = healthy_loss()
    val = healthy_val(train)
    for i in range(20, 38):
        val[i] = float(val[20] + 0.25 * (i - 20) / 18)
    for i in range(38, EPOCHS):
        val[i] = float(val[38] * math.exp(-2.0 * (i - 38) / (EPOCHS - 38)))
    return {"loss": train, "val_loss": val}, None


@case("healthy_curriculum_steps", "healthy", 0, None, "healthy")
def _healthy_curriculum_steps():
    """The loss steps up whenever harder data is mixed in, then resumes falling."""
    loss = healthy_loss()
    for boundary in (20, 40):
        for i in range(boundary, EPOCHS):
            loss[i] = float(loss[i] + 0.35 * math.exp(-0.4 * (i - boundary)))
    return {"loss": loss}, None


@case("healthy_finetune_flat", "healthy", 0, None, "healthy")
def _healthy_finetune_flat():
    """Fine-tuning from a good checkpoint: the loss starts at 0.08 and barely moves,
    because there is very little left to learn. Not a stuck run."""
    r = _rng(57)
    return {"loss": [float(0.08 - 0.02 * i / EPOCHS + r.normal(0, 0.002))
                     for i in range(EPOCHS)]}, None


@case("healthy_early_convergence", "healthy", 0, None, "healthy")
def _healthy_early_convergence():
    """Converged and flat afterwards, because it is done.

    Flattening at 0.06 rather than at 0.96: a run that stops at half its starting loss
    has not converged, it is stuck, and `stagnation` firing on it is correct. The
    fixture has to be a genuinely finished run for a raise to count as a false alarm.
    """
    r = _rng(59)
    loss = healthy_loss(n=25) + [float(0.055 + r.normal(0, 0.008)) for _ in range(EPOCHS - 25)]
    return {"loss": loss, "val_loss": healthy_val(loss)}, None


@case("healthy_normal_gap", "healthy", 0, None, "healthy")
def _healthy_normal_gap():
    """Val sits about 0.25 above train the whole way. A generalization gap is not
    overfitting unless it widens.

    With its own noise: a val curve equal to train plus a constant to the last decimal
    is a synthetic artefact, not a healthy run, and anything reading it closely is right
    to say so.
    """
    r = _rng(63)
    train = healthy_loss()
    return {"loss": train, "val_loss": [float(v + 0.25 + r.normal(0, 0.014)) for v in train]}, None


@case("healthy_lr_decay", "healthy", 0, None, "healthy")
def _healthy_lr_decay():
    """Step decay: the LR drops 10x twice, which is a 10x jump ratio in the wrong
    direction and must not read as an lr_jump fault."""
    lr = [0.001] * 25 + [0.0001] * 20 + [0.00001] * 15
    return {"loss": healthy_loss(), "lr": lr}, None


@case("healthy_short", "healthy", 0, None, "healthy")
def _healthy_short():
    return {"loss": healthy_loss(n=6)}, None


@case("healthy_tiny", "healthy", 0, None, "healthy")
def _healthy_tiny():
    return {"loss": healthy_loss(n=2)}, None


@case("healthy_single_epoch", "healthy", 0, None, "healthy")
def _healthy_single_epoch():
    return {"loss": healthy_loss(n=1)}, None


@case("healthy_high_lr_stable", "healthy", 0, None, "healthy")
def _healthy_high_lr_stable():
    """Large but stable oscillation around a falling trend -- large-batch training."""
    loss = healthy_loss()
    return {"loss": [float(v * (1 + 0.04 * (1 if i % 2 else -1)))
                     for i, v in enumerate(loss)]}, None


@case("healthy_accuracy_plateau", "healthy", 0, None, "healthy")
def _healthy_accuracy_plateau():
    """Accuracy saturates at 0.96 while the loss keeps improving. Normal."""
    train = healthy_loss()
    acc = [float(min(0.96, 0.5 + 0.5 * i / 20)) for i in range(EPOCHS)]
    return {"loss": train, "accuracy": acc}, None


@case("healthy_many_metrics", "healthy", 0, None, "healthy")
def _healthy_many_metrics():
    """Forty healthy metrics. Forty chances to raise something."""
    h = {"loss": healthy_loss()}
    for d in range(40):
        h[f"metric_{d}"] = healthy_loss(seed=200 + d, noise=0.03)
    return h, None


@case("healthy_constant_lr", "healthy", 0, None, "healthy")
def _healthy_constant_lr():
    return {"loss": healthy_loss(), "lr": [0.001] * EPOCHS}, None


@case("healthy_zero_loss_target", "healthy", 0, None, "healthy")
def _healthy_zero_loss_target():
    """An autoencoder driving loss to nearly zero. Small numbers, not broken ones.

    Noisy, because a loss that is a clean exponential to fifteen significant figures is
    not something a training run produces.
    """
    r = _rng(65)
    return {"loss": [float(max(1e-12, 1e-3 * math.exp(-4.0 * i / EPOCHS) + 1e-9
                               + r.normal(0, 3e-5))) for i in range(EPOCHS)]}, None


@case("healthy_rising_accuracy_only", "healthy", 0, None, "healthy")
def _healthy_rising_accuracy_only():
    """Only an accuracy is logged, and it rises. Nothing named `loss` at all."""
    return {"accuracy": accuracy_from(healthy_loss())}, None


@case("healthy_reward_rising", "healthy", 0, None, "healthy")
def _healthy_reward_rising():
    """RL: reward goes UP, and a rising curve must not be read as divergence."""
    r = _rng(61)
    return {"reward": [float(-100 + 180 * (1 - math.exp(-3.0 * i / EPOCHS)) + r.normal(0, 4))
                       for i in range(EPOCHS)]}, None


@case("healthy_perplexity_falling", "healthy", 0, None, "healthy")
def _healthy_perplexity_falling():
    loss = healthy_loss(start=4.0, floor=1.8)
    return {"loss": loss, "perplexity": [float(math.exp(v)) for v in loss]}, None


# ==================================================================== blind spots
# Everything above walks a knob on a fault Pulse has a check for, to find where the
# check gives up. These are different: real failures with no corresponding check at
# all, included precisely because they should fail. A benchmark that only tests what a
# system was built to catch measures its thresholds, not its coverage.
#
# Most are relationships between variables rather than the shape of one curve, which is
# the structural gap: Pulse's checks are mostly univariate.

BLIND_FAMILIES = set()


def blind(family):
    BLIND_FAMILIES.add(family)
    return family


# --- data pipeline ----------------------------------------------------------------

@ladder(blind("shuffled_labels"), rungs([10, 100, 1000, 50000]), expect="any")
def _shuffled_labels(classes):
    """Labels shuffled: the loss falls to exactly ln(classes) and sits there, because
    the model learns the prior and nothing else. The giveaway is the *value* -- chance
    level for this many classes -- not the shape, which is a normal convergence."""
    chance = math.log(classes)
    r = _rng(71)
    return {"loss": [float(chance + (math.log(classes) * 0.3) * math.exp(-3.0 * i / EPOCHS)
                           + r.normal(0, 0.004)) for i in range(EPOCHS)],
            "num_classes": [float(classes)] * EPOCHS}, None


@ladder(blind("class_imbalance_collapse"), rungs([0.99, 0.95, 0.9, 0.8, 0.7]), expect="any")
def _class_imbalance_collapse(majority):
    """The model predicts the majority class for everything. Accuracy equals the class
    prior exactly and the loss looks converged; only a per-class metric would show it."""
    r = _rng(73)
    acc = [float(min(majority, 0.3 + (majority - 0.3) * min(1.0, i / 10.0)) + r.normal(0, 0.002))
           for i in range(EPOCHS)]
    return {"loss": healthy_loss(start=1.0, floor=float(-math.log(majority))),
            "accuracy": acc, "f1_macro": [float(2 * (1 - majority)) for _ in range(EPOCHS)]}, None


@ladder(blind("loss_not_averaged"), rungs([1024, 256, 64, 16, 4]), expect="any")
def _loss_not_averaged(batch):
    """The loss is summed rather than averaged, so it scales with batch size. Training
    is fine; every number reported is `batch` times too big, and any threshold tuned on
    a per-sample loss is now meaningless."""
    return {"loss": [float(v * batch) for v in healthy_loss()],
            "batch_size": [float(batch)] * EPOCHS}, None


@ladder(blind("padding_fraction_growth"), rungs([0.95, 0.8, 0.6, 0.4, 0.25]), expect="any")
def _padding_fraction_growth(end_frac):
    """Sequence bucketing degrades and the batches fill up with padding: by the end
    `end_frac` of every batch is padding, so most of the compute trains on nothing.
    The loss curve is untroubled -- it is measured on the real tokens."""
    return {"loss": healthy_loss(),
            "padding_fraction": [float(0.05 + (end_frac - 0.05) * i / (EPOCHS - 1))
                                 for i in range(EPOCHS)],
            "tokens_per_sec": [float(12000 * (1 - 0.5 * i / EPOCHS)) for i in range(EPOCHS)]}, None


@ladder(blind("duplicate_batches"), rungs([1, 2, 4, 8]), expect="any")
def _duplicate_batches(unique):
    """The sampler serves only `unique` distinct batches. The loss falls beautifully --
    the model is memorizing them -- while val stays flat."""
    train = [float(2.0 * math.exp(-6.0 * i / EPOCHS) + 1e-4) for i in range(EPOCHS)]
    val = [float(0.69 + _rng(77).normal(0, 0.004)) for _ in range(EPOCHS)]
    return {"loss": train, "val_loss": val,
            "unique_batches": [float(unique)] * EPOCHS}, None


# --- distributed ------------------------------------------------------------------

@ladder(blind("rank_desync"), rungs([2.0, 1.0, 0.4, 0.15, 0.05]), expect="any")
def _rank_desync(spread):
    """Ranks report losses `spread` apart: the all-reduce is not happening and each
    rank is training its own model. Every individual curve looks healthy."""
    base = healthy_loss()
    h = {"loss": base}
    for rank in range(4):
        offset = spread * (rank - 1.5) / 1.5
        h[f"loss_rank{rank}"] = [float(v + offset * (0.3 + 0.7 * i / EPOCHS))
                                 for i, v in enumerate(base)]
    return h, None


@ladder(blind("straggler_rank"), rungs([20.0, 8.0, 3.0, 1.8, 1.25]), expect="any")
def _straggler_rank(slowdown):
    """One rank takes `slowdown`x as long per step, so every other rank waits at the
    barrier. Throughput is a fraction of what the hardware can do and nothing is wrong
    with the loss."""
    h = {"loss": healthy_loss()}
    for rank in range(4):
        t = 0.25 * (slowdown if rank == 2 else 1.0)
        h[f"step_time_rank{rank}"] = [float(t)] * EPOCHS
    return h, None


@ladder(blind("gradient_not_synced"), rungs([4, 8, 16, 64]), expect="any")
def _gradient_not_synced(world_size):
    """Gradients are averaged over the wrong world size, so the effective learning rate
    is `world_size`x off. It still trains, just badly, and no curve says why."""
    h = {"loss": healthy_loss(floor=0.45)}
    h["world_size"] = [float(world_size)] * EPOCHS
    h["grad_norm"] = [float(v * world_size) for v in grad_norm_for(h["loss"])]
    return h, None


# --- checkpoint and resume --------------------------------------------------------

@ladder(blind("resume_regression"), rungs([2.0, 1.0, 0.5, 0.2, 0.08]), expect="any")
def _resume_regression(jump):
    """A resumed run jumps back up by `jump` at the restart: the optimizer state was
    not restored. It recovers, so the shape is a spike followed by re-convergence --
    exactly like a legitimate warm restart."""
    loss = healthy_loss()
    for i in range(30, EPOCHS):
        loss[i] = float(loss[i] + jump * math.exp(-0.25 * (i - 30)))
    return {"loss": loss, "epoch_resumed": [0.0] * 30 + [1.0] * (EPOCHS - 30)}, None


@ladder(blind("lr_not_restored"), rungs([100.0, 10.0, 3.0, 1.5]), expect="any")
def _lr_not_restored(ratio):
    """After a resume the LR is back at its initial value instead of its decayed one.
    The loss degrades gently; the LR curve is the only evidence and it looks like a
    schedule, not a bug."""
    lr = [float(0.001 * (1 - 0.9 * i / 30)) for i in range(30)]
    lr += [float(0.001 * ratio * (1 - 0.9 * (i - 30) / 30)) for i in range(30, EPOCHS)]
    loss = healthy_loss()
    for i in range(30, EPOCHS):
        loss[i] = float(loss[29] * (1 + 0.004 * ratio * (i - 30)))
    return {"loss": loss, "lr": lr}, None


# --- evaluation correctness -------------------------------------------------------

@ladder(blind("eval_in_train_mode"), rungs([0.30, 0.15, 0.08, 0.04, 0.02]), expect="any")
def _eval_in_train_mode(noise):
    """model.eval() was never called: dropout is still on during validation, so val is
    noisy and pessimistic by a constant amount. Nothing diverges; the number is simply
    wrong, and every decision made from it is wrong too."""
    train = healthy_loss()
    r = _rng(79)
    return {"loss": train,
            "val_loss": [float(v + 0.12 + abs(r.normal(0, noise))) for v in train]}, None


@ladder(blind("bn_stats_frozen"), rungs([0.5, 0.25, 0.1, 0.05]), expect="any")
def _bn_stats_frozen(gap):
    """BatchNorm running stats never updated, so val is `gap` worse than it should be
    and the gap is flat. A constant offset reads as a normal generalization gap."""
    train = healthy_loss()
    return {"loss": train, "val_loss": [float(v + gap) for v in train],
            "bn_momentum": [0.0] * EPOCHS}, None


@ladder(blind("metric_wrong_axis"), rungs([0.5, 0.2, 0.1, 0.05]), expect="any")
def _metric_wrong_axis(offset):
    """Accuracy is argmaxed over the wrong axis. It correlates with nothing: the loss
    improves while accuracy wanders. Neither curve is individually abnormal."""
    train = healthy_loss()
    r = _rng(83)
    return {"loss": train,
            "accuracy": [float(0.5 + r.normal(0, offset)) for _ in range(EPOCHS)]}, None


# --- RL and generative ------------------------------------------------------------

@ladder(blind("reward_hacking"), rungs([0.02, 0.1, 0.25, 0.5, 0.75]), expect="any")
def _reward_hacking(length_frac):
    """Reward climbs while episode length collapses to `length_frac`: the policy found
    a degenerate strategy. The metric everyone watches is going the right way."""
    r = _rng(89)
    return {"reward": [float(-50 + 150 * (1 - math.exp(-3.0 * i / EPOCHS)) + r.normal(0, 3))
                       for i in range(EPOCHS)],
            "episode_length": [float(500 * (1 - (1 - length_frac) * i / (EPOCHS - 1)))
                               for i in range(EPOCHS)]}, None


@ladder(blind("gan_mode_collapse"), rungs([1e-6, 1e-3, 0.02, 0.1, 0.25]), expect="any")
def _gan_mode_collapse(d_floor):
    """The discriminator wins completely: D loss to `d_floor`, G loss climbing. Two
    curves moving in opposite directions is the definition of GAN training right up
    until it is the definition of failure."""
    d = [float(max(d_floor, 0.69 * math.exp(-4.0 * i / EPOCHS))) for i in range(EPOCHS)]
    g = [float(0.69 * math.exp(2.5 * i / EPOCHS)) for i in range(EPOCHS)]
    return {"d_loss": d, "g_loss": g}, None


@ladder(blind("kl_collapse"), rungs([0.0, 1e-6, 1e-3, 0.01, 0.05]), expect="any")
def _kl_collapse(floor):
    """VAE posterior collapse: the KL term goes to `floor` and the latent is ignored.
    Total loss improves, which is what makes it hard -- the model is getting better at
    the wrong objective."""
    kl = [float(max(floor, 5.0 * math.exp(-5.0 * i / EPOCHS))) for i in range(EPOCHS)]
    recon = healthy_loss(start=3.0, floor=0.8)
    return {"loss": [float(a + b) for a, b in zip(kl, recon)],
            "kl_loss": kl, "recon_loss": recon}, None


@ladder(blind("catastrophic_forgetting"), rungs([0.9, 0.6, 0.35, 0.18, 0.08]), expect="any")
def _catastrophic_forgetting(drop):
    """Fine-tuning on task B: task B accuracy rises, task A falls by `drop`. The
    headline metric improves throughout."""
    r = _rng(97)
    return {"loss": healthy_loss(),
            "acc_task_b": [float(0.3 + 0.65 * min(1.0, i / 25.0) + r.normal(0, 0.005))
                           for i in range(EPOCHS)],
            "acc_task_a": [float(0.92 - drop * min(1.0, i / 30.0) + r.normal(0, 0.005))
                           for i in range(EPOCHS)]}, None


# --- determinism and silent corruption ---------------------------------------------

@ladder(blind("seed_not_fixed"), rungs([0.5, 0.2, 0.08, 0.03]), expect="any")
def _seed_not_fixed(spread):
    """Three "identical" runs land `spread` apart. Nothing is wrong with any single
    curve; the fault is only visible by comparing runs, which a per-run detector
    cannot do by construction."""
    h = {}
    for k in range(3):
        r = _rng(101 + k)
        h[f"loss_run{k}"] = [float(0.05 + (2.0 - 0.05) * math.exp(-3.0 * i / EPOCHS)
                                   + spread * (k - 1) * i / EPOCHS + r.normal(0, 0.01))
                             for i in range(EPOCHS)]
    h["loss"] = h["loss_run0"]
    return h, None


@ladder(blind("silent_dtype_downcast"), rungs([1e-3, 1e-4, 1e-5, 1e-6]), expect="any")
def _silent_dtype_downcast(resolution):
    """Something downcast to fp16 mid-pipeline: the loss is quantized to `resolution`
    and stops improving below it, but the curve above that floor is perfect."""
    loss = healthy_loss(floor=resolution * 2)
    return {"loss": [float(round(v / resolution) * resolution) for v in loss]}, None


@ladder(blind("gradient_accumulation_double"), rungs([8, 4, 2]), expect="any")
def _gradient_accumulation_double(steps):
    """Gradients accumulated `steps` times but the LR was not divided, so the effective
    step is `steps`x too large. Training is unstable in a way that looks like an
    ordinary high learning rate."""
    r = _rng(103)
    loss = [float(1.2 + 0.5 * math.sin(i / 2.0) * (steps / 8.0) + r.normal(0, 0.05))
            for i in range(EPOCHS)]
    return {"loss": loss, "accumulation_steps": [float(steps)] * EPOCHS}, None


@ladder(blind("data_ordering_bias"), rungs([1.0, 0.5, 0.2, 0.08]), expect="any")
def _data_ordering_bias(amplitude):
    """The data is sorted by class and never shuffled, so the loss oscillates with the
    epoch boundary. Periodic, but tied to the epoch rather than alternating step to
    step -- which is what the oscillation check looks for."""
    base = healthy_loss()
    return {"loss": [float(v + amplitude * math.sin(2 * math.pi * i / 10.0))
                     for i, v in enumerate(base)]}, None


# --- hardware ---------------------------------------------------------------------

@ladder(blind("gpu_util_collapse"), rungs([0.02, 0.1, 0.3, 0.55, 0.75]), expect="any")
def _gpu_util_collapse(util):
    """GPU utilization falls to `util` -- the dataloader cannot keep up. The run is
    correct and costing several times what it should."""
    return {"loss": healthy_loss(),
            "gpu_util": [float(0.95 - (0.95 - util) * min(1.0, i / 20.0))
                         for i in range(EPOCHS)],
            "data_wait_ms": [float(5 + 400 * min(1.0, i / 20.0)) for i in range(EPOCHS)]}, None


@ladder(blind("thermal_throttle"), rungs([0.4, 0.6, 0.75, 0.88]), expect="any")
def _thermal_throttle(clock_frac):
    """Clocks drop to `clock_frac` as the card heats up."""
    return {"loss": healthy_loss(),
            "gpu_clock_mhz": [float(1900 * (1 - (1 - clock_frac) * min(1.0, i / 25.0)))
                              for i in range(EPOCHS)],
            "gpu_temp_c": [float(55 + 33 * min(1.0, i / 25.0)) for i in range(EPOCHS)]}, None


# ================================================================= more failure types
# Breadth rather than depth: shorter ladders, many more distinct faults. These are the
# things that actually go wrong in training runs, written as the curves they produce.

def _flat(v, n=EPOCHS):
    return [float(v)] * n


# --- initialization and scale ------------------------------------------------------

@ladder(blind("bad_weight_init"), rungs([1e6, 1e3, 50.0, 12.0]), expect="any")
def _bad_weight_init(start):
    """Initial loss of `start` instead of ln(vocab): the init scale is wrong. It
    recovers and converges, so only the first few epochs ever showed it."""
    loss = [float(start * math.exp(-1.2 * i) + 0.7) for i in range(EPOCHS)]
    return {"loss": loss}, None


@ladder(blind("logit_explosion"), rungs([1e5, 1e3, 100.0, 25.0]), expect="any")
def _logit_explosion(peak):
    """Logits grow without bound while the loss looks fine -- the softmax saturates and
    gradients quietly stop flowing."""
    return {"loss": healthy_loss(),
            "logit_max": [float(5.0 * (peak / 5.0) ** (i / (EPOCHS - 1))) for i in range(EPOCHS)],
            "grad_norm": [float(0.5 * math.exp(-4.0 * i / EPOCHS)) for i in range(EPOCHS)]}, None


@ladder(blind("softmax_saturation"), rungs([1e-8, 1e-5, 1e-3, 0.02]), expect="any")
def _softmax_saturation(entropy):
    """Output entropy collapses to `entropy`: the model is maximally confident and
    wrong, which a loss average hides."""
    return {"loss": healthy_loss(floor=0.4),
            "output_entropy": [float(max(entropy, 2.3 * math.exp(-6.0 * i / EPOCHS)))
                               for i in range(EPOCHS)]}, None


@ladder(blind("attention_entropy_collapse"), rungs([1e-4, 0.01, 0.1, 0.4]), expect="any")
def _attention_entropy_collapse(floor):
    """Every attention head attends to one token. Common with too-high LR on long
    context, and invisible in the loss until much later."""
    return {"loss": healthy_loss(),
            "attn_entropy": [float(max(floor, 3.5 * math.exp(-5.0 * i / EPOCHS)))
                             for i in range(EPOCHS)]}, None


# --- optimizer ---------------------------------------------------------------------

@ladder(blind("momentum_dead"), rungs([0.0, 1e-8, 1e-5, 1e-3]), expect="any")
def _momentum_dead(value):
    """Optimizer momentum buffers are zeroed every step -- a state_dict loaded wrong.
    Training still descends, just far more slowly than it should."""
    return {"loss": healthy_loss(floor=0.9), "momentum_norm": _flat(value)}, None


@ladder(blind("grad_clip_always_active"), rungs([1.0, 0.5, 0.2, 0.05]), expect="any")
def _grad_clip_always_active(clip):
    """The gradient norm is pinned at exactly the clip value on every step: the clip is
    doing all the work and the real gradient scale is unknown."""
    return {"loss": healthy_loss(floor=0.8), "grad_norm": _flat(clip),
            "clip_fraction": _flat(1.0)}, None


@ladder(blind("adam_epsilon_too_large"), rungs([1.0, 1e-2, 1e-4]), expect="any")
def _adam_epsilon_too_large(eps):
    """Adam's epsilon swamps the second moment, so the effective step size collapses
    and the run converges to a worse place, smoothly."""
    return {"loss": healthy_loss(floor=0.55 + 0.1 * math.log10(max(eps, 1e-8)) / 4),
            "adam_eps": _flat(eps)}, None


@ladder(blind("ema_diverges"), rungs([2.0, 0.8, 0.3, 0.1]), expect="any")
def _ema_diverges(gap):
    """The EMA copy of the weights drifts `gap` away from the live model -- the decay
    is wrong, and the checkpoint that gets shipped is not the model that was trained."""
    live = healthy_loss()
    return {"loss": live, "ema_loss": [float(v + gap * i / EPOCHS) for i, v in enumerate(live)]}, None


# --- mixture of experts -------------------------------------------------------------

@ladder(blind("moe_expert_collapse"), rungs([1e-4, 0.05, 0.3, 0.8]), expect="any")
def _moe_expert_collapse(entropy):
    """Routing entropy falls to `entropy`: one expert takes every token and the rest of
    the parameters are dead weight. The loss keeps improving."""
    return {"loss": healthy_loss(),
            "router_entropy": [float(max(entropy, 2.0 * math.exp(-4.0 * i / EPOCHS)))
                               for i in range(EPOCHS)],
            "expert_0_frac": [float(min(0.99, 0.125 + 0.87 * i / EPOCHS)) for i in range(EPOCHS)]}, None


@ladder(blind("moe_load_imbalance"), rungs([50.0, 12.0, 4.0, 1.8]), expect="any")
def _moe_load_imbalance(ratio):
    """Max/min expert load reaches `ratio`. Tokens get dropped at capacity and the
    throughput cost never shows up in the loss."""
    return {"loss": healthy_loss(),
            "expert_load_ratio": [float(1.0 + (ratio - 1.0) * i / (EPOCHS - 1)) for i in range(EPOCHS)],
            "dropped_token_frac": [float(0.3 * i / EPOCHS) for i in range(EPOCHS)]}, None


# --- distillation and multi-objective -----------------------------------------------

@ladder(blind("distill_teacher_ignored"), rungs([5.0, 2.0, 0.8, 0.3]), expect="any")
def _teacher_student_divergence(kl):
    """Distillation is not the fault -- this is a distillation run where the teacher
    is being ignored. Student KL to the teacher grows to `kl` while its own CE falls:
    the KL weight is zero, or the wrong teacher was loaded. Training looks healthy and
    the expensive teacher is contributing nothing."""
    return {"loss": healthy_loss(),
            "kl_to_teacher": [float(0.2 + kl * i / (EPOCHS - 1)) for i in range(EPOCHS)]}, None


@ladder(blind("aux_loss_dominates"), rungs([1000.0, 100.0, 10.0, 3.0]), expect="any")
def _aux_loss_dominates(weight):
    """An auxiliary term is `weight`x the main loss, so the total is optimizing the
    regularizer. The total falls; the thing you care about does not."""
    main = [float(0.69 - 0.02 * i / EPOCHS) for i in range(EPOCHS)]
    aux = [float(weight * math.exp(-4.0 * i / EPOCHS)) for i in range(EPOCHS)]
    return {"loss": [float(a + b) for a, b in zip(main, aux)],
            "main_loss": main, "aux_loss": aux}, None


@ladder(blind("aux_loss_nan_only"), rungs([5, 25, 45, 57]), expect="any")
def _aux_loss_nan_only(onset):
    """Only the auxiliary loss is NaN. The total is computed elsewhere and stays
    finite, so the headline number never shows it."""
    aux = [float(0.1)] * EPOCHS
    for i in range(onset, EPOCHS):
        aux[i] = float("nan")
    return {"loss": healthy_loss(), "aux_loss": aux}, None


# --- data and tokenization -----------------------------------------------------------

@ladder(blind("tokenizer_mismatch"), rungs([50000, 32000, 8000, 1000]), expect="any")
def _tokenizer_mismatch(vocab):
    """The loss sits at ln(vocab) forever: the tokenizer does not match the embedding
    table, so every prediction is uniform over a vocabulary it cannot address."""
    chance = math.log(vocab)
    r = _rng(107)
    return {"loss": [float(chance + r.normal(0, 0.003)) for _ in range(EPOCHS)],
            "vocab_size": _flat(vocab)}, None


@ladder(blind("oov_rate_growth"), rungs([0.6, 0.3, 0.1, 0.03]), expect="any")
def _oov_rate_growth(rate):
    """Out-of-vocabulary rate climbs to `rate` as the data shifts: later shards were
    written with a different tokenizer."""
    return {"loss": healthy_loss(floor=0.3),
            "oov_rate": [float(0.001 + rate * i / (EPOCHS - 1)) for i in range(EPOCHS)]}, None


@ladder(blind("sequence_truncation"), rungs([0.8, 0.5, 0.25, 0.1]), expect="any")
def _sequence_truncation(frac):
    """`frac` of every batch is being truncated away: max_length is too small and the
    model never sees the end of anything."""
    return {"loss": healthy_loss(floor=0.6),
            "truncated_frac": [float(frac)] * EPOCHS,
            "mean_seq_len": _flat(512.0)}, None


@ladder(blind("eval_set_too_small"), rungs([2, 8, 32, 128]), expect="any")
def _eval_set_too_small(n):
    """The val set has `n` examples, so val loss is mostly sampling noise. Every early
    stopping decision made from it is a coin flip."""
    train = healthy_loss()
    r = _rng(109)
    noise = 1.2 / math.sqrt(n)
    return {"loss": train, "val_loss": [float(v + 0.1 + r.normal(0, noise)) for v in train],
            "val_size": _flat(n)}, None


# --- silent stops --------------------------------------------------------------------

@ladder(blind("metrics_stop_updating"), rungs([10, 25, 40, 52]), expect="any")
def _metrics_stop_updating(onset):
    """Every metric freezes at `onset`: the training process is alive but the step loop
    is wedged on a barrier or a dataloader worker. Nothing errors."""
    h = {"loss": healthy_loss(), "grad_norm": grad_norm_for(healthy_loss()),
         "tokens_per_sec": _flat(12000.0)}
    for name, values in h.items():
        for i in range(onset, EPOCHS):
            values[i] = values[onset]
    return h, None


@ladder(blind("throughput_to_zero"), rungs([5, 20, 40, 55]), expect="any")
def _throughput_to_zero(onset):
    """Throughput goes to zero at `onset` while the loss keeps its last value: the run
    has hung, and a loss-only detector sees a plateau at worst."""
    loss = healthy_loss()
    tok = _flat(12000.0)
    for i in range(onset, EPOCHS):
        loss[i] = loss[onset]
        tok[i] = 0.0
    return {"loss": loss, "tokens_per_sec": tok}, None


@ladder(blind("checkpoint_not_saving"), rungs([0, 1]), expect="any")
def _checkpoint_not_saving(saved):
    """`checkpoints_written` never increments -- the path is wrong or the disk is full.
    Forty hours of training with nothing to show for it."""
    return {"loss": healthy_loss(), "checkpoints_written": _flat(saved),
            "disk_free_gb": [float(max(0.0, 50 - 0.9 * i)) for i in range(EPOCHS)]}, None


# --- metric correctness ---------------------------------------------------------------

@ladder(blind("accuracy_frozen_while_loss_moves"), rungs([30, 20, 12, 6]), expect="any")
def _accuracy_frozen_while_loss_moves(length):
    """Loss improves; accuracy has not changed in `length` epochs. One of them is being
    computed wrong and they cannot both be right."""
    loss = healthy_loss()
    acc = accuracy_from(loss)
    for i in range(EPOCHS - length, EPOCHS):
        acc[i] = acc[EPOCHS - length]
    return {"loss": loss, "accuracy": acc}, None


@ladder(blind("perplexity_loss_mismatch"), rungs([100.0, 10.0, 2.0, 1.3]), expect="any")
def _perplexity_loss_mismatch(factor):
    """Reported perplexity is not exp(loss) -- it is `factor` off. One of the two is
    lying, and both are on the dashboard."""
    loss = healthy_loss(start=4.0, floor=1.8)
    return {"loss": loss, "perplexity": [float(math.exp(v) * factor) for v in loss]}, None


@ladder(blind("metric_off_by_one_epoch"), rungs([5, 3, 1]), expect="any")
def _metric_off_by_one_epoch(shift):
    """Validation is logged `shift` epochs late, so every comparison between train and
    val is misaligned. Both curves look perfect."""
    train = healthy_loss()
    val = healthy_val(train)
    return {"loss": train, "val_loss": val[shift:] + val[-1:] * shift}, None


@ladder(blind("loss_constant_zero"), rungs([20, 40, 55]), expect="any")
def _loss_constant_zero(onset):
    """Loss becomes exactly 0.0 -- masked-out labels, or an empty batch reduction.
    Zero is the best possible value, so nothing that looks for "worse" will fire."""
    loss = healthy_loss()
    for i in range(onset, EPOCHS):
        loss[i] = 0.0
    return {"loss": loss}, None


@ladder("label_smoothing_floor", rungs([0.4, 0.2, 0.1, 0.05]), expect=None, tag="healthy")
def _label_smoothing_floor(floor):
    """The loss cannot go below `floor` because of label smoothing. This is not a
    fault -- it is the objective doing what it was configured to do -- so it is scored
    as a healthy run that a stagnation check must not flag."""
    return {"loss": [float(floor + (2.0 - floor) * math.exp(-3.0 * i / EPOCHS))
                     for i in range(EPOCHS)]}, None


# --- contrastive / multimodal -----------------------------------------------------------

@ladder(blind("temperature_collapse"), rungs([1e-4, 0.01, 0.05, 0.12]), expect="any")
def _temperature_collapse(temp):
    """The learned contrastive temperature collapses to `temp`, so the loss is tiny and
    the embeddings are degenerate."""
    return {"loss": [float(max(1e-5, 2.0 * temp / 0.07 * math.exp(-2.0 * i / EPOCHS)))
                     for i in range(EPOCHS)],
            "temperature": [float(max(temp, 0.07 * math.exp(-4.0 * i / EPOCHS)))
                            for i in range(EPOCHS)]}, None


@ladder(blind("embedding_norm_drift"), rungs([1000.0, 100.0, 10.0, 3.0]), expect="any")
def _embedding_norm_drift(ratio):
    """Text and image embedding norms drift `ratio` apart, so one modality dominates
    every similarity. Both encoders are training fine on their own terms."""
    return {"loss": healthy_loss(floor=0.5),
            "text_emb_norm": _flat(1.0),
            "image_emb_norm": [float(1.0 * ratio ** (i / (EPOCHS - 1))) for i in range(EPOCHS)]}, None


# --- tensors ------------------------------------------------------------------------------

@ladder(blind("tensor_dtype_change"), rungs(["int8", "float16", "bfloat16"]), expect="any")
def _tensor_dtype_change(dtype):
    """A weight silently changes dtype mid-run -- an autocast boundary in the wrong
    place. Precision is lost and no scalar shows it.

    The stats are a *sequence*: float32 first, then `dtype`. A fixture that reported the
    same dtype from the first reading was not testing a change at all."""
    stats = [{"layer1.weight": {"shape": [256, 256], "device": "cuda:0", "dtype": d,
                                "mean": 0.01, "std": 0.02, "min": -0.1, "max": 0.1,
                                "nan": 0, "inf": 0}}
             for d in (["float32"] * 30 + [dtype] * 30)]
    return {"loss": healthy_loss()}, stats


@ladder(blind("tensor_moved_to_cpu"), rungs(["cpu"]), expect="any")
def _tensor_moved_to_cpu(device):
    """A parameter ends up on the CPU, so every step round-trips over PCIe. Correct,
    and fifty times slower."""
    stats = {"layer1.weight": {"shape": [256, 256], "device": device, "dtype": "float32",
                               "mean": 0.01, "std": 0.02, "min": -0.1, "max": 0.1,
                               "nan": 0, "inf": 0}}
    return {"loss": healthy_loss(),
            "step_time": [float(0.25 * (1 + 20 * min(1.0, i / 10))) for i in range(EPOCHS)]}, stats


@ladder(blind("tensor_all_zeros"), rungs([0.0]), expect="any")
def _tensor_all_zeros(value):
    """A weight tensor is entirely zero: a layer that was never initialized, or was
    zeroed by a bad load. Its gradients are zero too, so it never recovers."""
    stats = {"layer3.weight": {"shape": [512, 512], "device": "cuda:0", "dtype": "float32",
                               "mean": value, "std": value, "min": value, "max": value,
                               "nan": 0, "inf": 0}}
    return {"loss": healthy_loss(floor=0.7)}, stats
