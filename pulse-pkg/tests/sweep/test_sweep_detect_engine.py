"""Bug sweep: the detection engine (src/pulse/pulse_detect.py) and how the CLI feeds it.

Every test_bug_* asserts the CORRECT behaviour, so it fails on the current code and
passes once the bug is fixed. Every test_ok_* is regression coverage for behaviour that
was verified to work.
"""
import io
import contextlib
import math
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.join(HERE, "..", "..", "src"), os.path.join(HERE, "..", "src")):
    if os.path.isdir(candidate):
        sys.path.insert(0, os.path.abspath(candidate))
        break

from pulse import pulse_detect as D  # noqa: E402
from pulse.pulse_detect import DetectionEngine  # noqa: E402


# ----------------------------------------------------------------------------- helpers

def feed(histories, start=3, sensitivity=0.3, engine=None, **kw):
    """Grow every history one reading at a time, as a live run does, and collect every
    finding raised along the way as (check, variable)."""
    engine = engine or DetectionEngine(sensitivity=sensitivity, **kw)
    longest = max(len(v) for v in histories.values())
    raised = []
    for k in range(start, longest + 1):
        out = engine.update({n: list(v[:k]) for n, v in histories.items()}, step=k)
        raised.extend((f.check, f.variable) for f in out["raised"])
    return raised


def fired(histories, sensitivity=0.3):
    """Every finding the checks produce on this exact state, before confirmation."""
    engine = DetectionEngine(sensitivity=sensitivity, confirmations=1)
    return [(f.check, f.variable) for f in engine.update(histories)["raised"]]


def checks_of(pairs):
    return {c for c, _ in pairs}


def noisy(values, sigma, seed=0):
    rng = random.Random(seed)
    return [v + rng.gauss(0.0, sigma) for v in values]


def make_cli():
    from pulse import pulse_cli
    cli = pulse_cli.PulseCLI(watch_locals={}, pdf_dir=os.path.join("/tmp", "_sweep_detect_pdfs"))
    cli.sensitivity = 0.3
    cli.epoch_scalar_histories = {}
    cli.batch_scalar_histories = {}
    return cli


# ============================================================================ BUGS

# ---- CLI wiring ------------------------------------------------------------------

def test_bug_sampled_scalar_dedup_makes_frozen_loss_invisible():
    """PulseCLI.update appends a sampled scalar to scalar_histories only when it differs
    from the previous reading (`changed = not hist or not _values_equal(hist[-1], v)`).
    The detector reads those histories, so a loss that is stuck at one value -- a
    detached graph, an optimizer that never steps -- produces a history of ONE reading
    no matter how many steps pass. `frozen` (6 identical readings), `plateau`,
    `counter_stalled` and `hyperparameter_out_of_range` can therefore never fire on a
    sampled (non-Keras) run. Correct: the detector must see one reading per step."""
    cli = make_cli()
    cli.continuous = True
    cli.auto_intervene = False
    cli.tracked_vars = ["loss"]
    buf = io.StringIO()
    losses = [1.0, 0.9, 0.8, 0.7] + [0.65] * 20
    with contextlib.redirect_stdout(buf):
        for i, value in enumerate(losses):
            cli.watch_locals = {"loss": value}
            cli.update(step=i)
    history = cli._detector_histories().get("loss", [])
    assert len(history) >= 10, f"24 steps sampled, detector sees only {history}"


def test_bug_confirmation_satisfied_by_rechecking_the_same_epoch():
    """_check_for_trouble runs on every PulseCLI.update (every batch) but Keras epoch
    histories only grow once per epoch. The engine counts each update() as a new
    confirmation, so a finding seen on ONE epoch reading is 'confirmed' by the next
    batch without any new evidence -- `confirmations=2` does nothing. Correct: a repeat
    evaluation of unchanged histories must not advance the confirmation streak."""
    cli = make_cli()
    # A val_loss whose last epoch is the fourth rise in a row -- fires val_regression
    # (a confirmation-gated check) on exactly this state.
    val = [0.50, 0.49, 0.51, 0.50, 0.49, 0.50, 0.51, 0.50, 0.49, 0.50, 0.52, 0.55, 0.58, 0.62]
    train = [0.60 - 0.01 * i for i in range(len(val))]
    for epoch, (t, v) in enumerate(zip(train, val)):
        cli._record_keras_logs({"loss": t, "val_loss": v}, epoch=epoch)
    assert ("val_regression", "val_loss") in fired(cli._detector_histories())
    first = cli._check_for_trouble()          # first batch after the epoch
    second = cli._check_for_trouble()         # next batch, same epoch data
    assert first is None
    assert second is None, f"raised on a re-check of unchanged data: {second}"


# ---- crashes -----------------------------------------------------------------------

def test_bug_overflow_in_pair_check_kills_detection():
    """_check_pairs computes the validation noise as `(v - mean) ** 2` on raw values
    (pulse_detect.py:1349), unlike _stdev which rescales. Once val_loss explodes past
    ~1e154 that raises OverflowError out of update(); the CLI swallows it and returns
    None, so the exact moment the run blows up is the moment detection goes silent.
    Correct: update() never raises and still reports the explosion."""
    train = [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.66, 0.64, 0.63, 0.62, 5e199, 1e200]
    val = [1.1, 1.0, 0.9, 0.85, 0.8, 0.78, 0.76, 0.75, 0.74, 0.73, 5e199, 1e200]
    engine = DetectionEngine(sensitivity=0.3)
    out = engine.update({"loss": train, "val_loss": val})  # must not raise
    assert "loss_spike" in {f.check for f in out["raised"]}


# ---- role / name classification ----------------------------------------------------

def test_bug_cross_entropy_loss_reported_as_entropy_collapse():
    """ENTROPY_NAME_HINTS contains "entropy", which is a substring of `cross_entropy`,
    `binary_crossentropy`, `sparse_categorical_crossentropy` (Keras metric names). A
    healthy classification loss falling from 2.3 to 0.2 is reported as "has collapsed
    ... it is now going to one place". Correct: a loss is not an entropy-role variable."""
    loss = [2.3 * math.exp(-0.12 * i) + 0.15 for i in range(40)]
    for name in ("cross_entropy", "sparse_categorical_crossentropy"):
        raised = feed({name: loss})
        assert ("entropy_collapse", name) not in raised, raised


def test_bug_loss_value_and_value_loss_classified_as_validation():
    """is_validation() matches `startswith("val")` and `"_val" in name`, so
    `loss_value` (the usual name for loss.item()) and RL's `value_loss` are treated as
    VALIDATION losses. Then _check_pairs/_check_relations pair them against the train
    loss, and a script with `loss` and `loss_value = loss.item()` gets "validation is
    being computed on the training data". Correct: neither is a validation name."""
    assert not D.is_validation("loss_value")
    assert not D.is_validation("value_loss")
    loss = noisy([2.0 * math.exp(-0.1 * i) + 0.2 for i in range(20)], 0.01)
    raised = feed({"loss": loss, "loss_value": list(loss)})
    assert "identical_series" not in checks_of(raised), raised


def test_bug_value_loss_gets_val_regression_despite_moving_target():
    """`value_loss` is a MOVING_TARGET (direction carries no information) but because
    is_validation("value_loss") is true, the val_regression / val_drift block in
    _check_loss runs on it -- that block is not gated on `directionless`. An RL critic
    loss that rises for four PPO iterations is reported as a regression. Correct: no
    directional check fires on a moving-target loss."""
    value_loss = [0.5, 0.45, 0.6, 0.4, 0.55, 0.42, 0.5, 0.6, 0.35, 0.3, 0.4, 0.55, 0.7, 0.9]
    raised = fired({"value_loss": value_loss})
    assert not ({"val_regression", "val_drift"} & checks_of(raised)), raised


def test_bug_kl_weight_annealing_reported_as_divergence():
    """LOSS_NAME_HINTS contains "kl", so the KL-annealing coefficient every VAE ramps
    from 0 to 1 (`kl_weight`, `kl_beta`) is treated as a loss, and its schedule is
    reported CRITICAL: "kl_weight is climbing, not falling". Correct: a coefficient /
    weight / beta is not a loss."""
    ramp = [min(1.0, 0.01 + i / 30.0) for i in range(40)]
    raised = feed({"kl_weight": ramp})
    assert ("divergence", "kl_weight") not in raised, raised


def test_bug_grad_accum_counter_treated_as_metric():
    """METRIC_NAME_HINTS contains "acc", so `accum_step`/`grad_accum_steps` look like an
    accuracy. A gradient-accumulation counter cycling 0,1,2,3 is then reported as
    "repeating the same 4 readings exactly: the same data is going through the model".
    Correct: looks_like_metric("accum_step") is False and nothing fires."""
    assert not D.looks_like_metric("grad_accum_steps")
    counter = [float(i % 4) for i in range(24)]
    raised = feed({"accum_step": counter})
    assert "repeating" not in checks_of(raised), raised


def test_bug_clip_eps_flagged_as_optimizer_epsilon():
    """SANE_RANGES["eps"] is (1e-12, 1e-4) and matches the whole token `eps`, so PPO's
    `clip_eps = 0.2` (and epsilon-greedy `eps`) is reported as "an optimizer epsilon this
    large swamps the second moment". Correct: only optimizer epsilons (adam_eps, or
    plain eps) should be range-checked -- not clip_eps."""
    reward = [float(i) for i in range(12)]
    raised = fired({"clip_eps": [0.2] * 12, "reward": reward})
    assert ("hyperparameter_out_of_range", "clip_eps") not in raised, raised


def test_bug_error_rate_on_eval_grid_reported_as_quantised():
    """`val_error` (classification error, k wrong out of N) is loss-like by the "err"
    hint, and the quantisation check runs on every loss. k/N lands on a 1/N grid by
    arithmetic -- the same reason the check exempts scores -- so a healthy eval error on
    a 200-example set is reported as "being rounded to a grid". Correct: no finding."""
    rng = random.Random(3)
    wrong = 80
    values = []
    for i in range(40):
        wrong = max(10, wrong + rng.choice([-4, -3, -2, -1, 1, 2]))
        values.append(wrong / 200.0)
    raised = fired({"val_error": values})
    assert ("quantised", "val_error") not in raised, raised


def test_bug_error_rate_reported_as_wasted_compute():
    """WASTE_NAME_HINTS contains "error_rate", so a classification error rate that sits
    at 45% on a hard task is reported as "that share of every batch is being thrown
    away". An error rate is a quality measure, not a waste fraction. Correct: no
    waste_* finding for an error rate."""
    values = noisy([0.6 - 0.15 * (1 - math.exp(-i / 5)) for i in range(20)], 0.005)
    raised = fired({"error_rate": values})
    assert not {c for c in checks_of(raised) if c.startswith("waste")}, raised


# ---- the metric checks ---------------------------------------------------------------

def test_bug_suspiciously_perfect_on_percent_scale_metric():
    """suspiciously_perfect tests `max(values) >= 0.999` with no notion of scale. Any
    metric reported in percent or on an open scale -- SQuAD F1 (0-100), BLEU, mAP in
    percent -- is "perfect" by its third reading: "f1 reached 50.1 within 3 readings,
    which usually means the labels are reachable from the inputs". Correct: only a
    0..1 metric at its ceiling is perfect."""
    raised = fired({"f1": [31.2, 45.3, 50.1]})
    raised += fired({"val_bleu": [12.0, 18.5, 21.0]})
    assert "suspiciously_perfect" not in checks_of(raised), raised


def test_bug_suspiciously_perfect_on_one_lucky_reading():
    """The early-run branch uses `max(values)`, so a single per-batch accuracy of 1.0 on
    a small batch -- followed by 0.75 -- is "accuracy reached 1 within 4 readings, which
    usually means the labels are reachable". A leak pins the score at the ceiling; one
    lucky batch does not. Correct: require the latest readings to be at the ceiling."""
    raised = feed({"accuracy": [0.25, 0.5, 1.0, 0.75, 0.625]}, start=2)
    assert "suspiciously_perfect" not in checks_of(raised), raised


def test_bug_identical_series_on_a_trivial_dataset():
    """identical_series fires whenever a train and a validation series agree to 1e-9
    for 10 readings -- including when both are simply CONSTANT, e.g. accuracy and
    val_accuracy both 1.0 on an easy dataset. That is reported as "validation is being
    computed on the training data". Two constants agreeing is not evidence of that.
    Correct: require the series to actually vary before calling them identical."""
    loss = [0.5 * math.exp(-0.3 * i) + 0.001 for i in range(14)]
    val = [0.55 * math.exp(-0.28 * i) + 0.002 for i in range(14)]
    raised = feed({"loss": loss, "val_loss": val,
                   "accuracy": [1.0] * 14, "val_accuracy": [1.0] * 14})
    assert "identical_series" not in checks_of(raised), raised


def test_bug_repeating_on_quantised_converged_val_metric():
    """The frozen check exempts a converged, quantised metric ("197 right out of 200 is
    exactly 0.985 every time"), but the `repeating` check right above it does not: a
    converged val_accuracy flipping between two neighbouring grid values (0.985, 0.99)
    is reported as "the same data is going through the model every time". Correct: the
    same quantisation exemption applies."""
    acc = [0.6, 0.8, 0.9, 0.95, 0.97, 0.98] + [0.985, 0.99] * 6
    raised = feed({"val_accuracy": acc})
    assert "repeating" not in checks_of(raised), raised


# ---- sign / scale blind spots -----------------------------------------------------

def test_bug_never_learned_blind_for_negative_objective():
    """The comment at pulse_detect.py:1204 says never_learned was made to work for
    objectives that report a negative number ("an ELBO ... flat at -3.2 forever"), but
    `precise_enough = 3.0 * se < start * 0.10` is never true when start < 0. A negative
    NLL/ELBO sitting flat for 40 readings is never reported. Correct: compare against
    abs(start)."""
    values = noisy([-3.2] * 40, 0.01, seed=1)
    positive = [-v for v in values]
    assert ("never_learned", "loss") in fired({"loss": positive})   # control
    assert ("never_learned", "loss") in fired({"loss": values})


def test_bug_loss_spike_blind_for_negative_loss():
    """loss_spike requires `baseline > 0`, so a loss that lives below zero (Gaussian NLL,
    negative ELBO, -log-likelihood of a continuous density) can jump from -3 to +500 and
    loss_spike never fires. Correct: measure the jump against the loss's own spread."""
    values = noisy([-3.0] * 30, 0.02) + [500.0]
    raised = fired({"loss": values})
    assert ("loss_spike", "loss") in raised, raised


def test_bug_score_regression_blind_for_negative_reward():
    """score_regression requires `peak > 0`. RL rewards are routinely negative
    (Pendulum, MountainCar, cost-shaped rewards), so an episode reward that climbs from
    -1500 to -200 and collapses back to -1200 is never reported. Correct: measure the
    fall against the range the score has covered, not against its sign."""
    up = [-1500 + 1300 * (1 - math.exp(-i / 5)) for i in range(20)]
    down = [-200 - 1000 * (1 - math.exp(-i / 2)) for i in range(1, 10)]
    raised = fired({"episode_reward": noisy(up + down, 5.0)})
    assert ("score_regression", "episode_reward") in raised, raised


def test_bug_loss_spike_on_near_zero_floor():
    """loss_spike compares the latest reading to a multiple of the recent floor. When the
    floor is near zero (a regression MAE converged to 3e-4) a move to 6e-3 -- 0.3% of
    where the run started -- is "spiked to 20x its recent floor", CRITICAL. This was a
    false alarm on a real benchmark run. Correct: a spike has to be large relative to the
    scale the variable has lived on, not only to a near-zero floor."""
    head = [1.0 * math.exp(-i / 6) + 3e-4 for i in range(40)]
    tail = noisy([3e-4] * 50, 2e-5, seed=4)
    values = head + tail + [6e-3]
    raised = fired({"val_mae": values})
    assert ("loss_spike", "val_mae") not in raised, raised


def test_bug_analysis_window_makes_long_converged_run_never_learned():
    """ANALYSIS_WINDOW (1500) truncates histories before every check, and never_learned
    and stagnation compare the *start of the window* to the end. On a step-level run
    past 1500 readings the true start is gone: a loss that came down from 2.1 to 0.1 and
    converged is reported CRITICAL "loss is no better than when it started ... this run
    is not learning". The comment on ANALYSIS_WINDOW claims "early still means early on
    any realistic run". Correct: no never_learned on a run that learned."""
    values = noisy([2.0 * math.exp(-i / 100) + 0.1 for i in range(3000)], 0.003, seed=5)
    raised = fired({"loss": values})
    assert ("never_learned", "loss") not in raised, raised


def test_bug_plateau_fires_on_converged_run():
    """stagnation deliberately stays quiet on a run that came down >90% and then sat
    still ("a run that came down 99% and then sat still has converged"), but `plateau`
    has no such guard: a full-batch run that converged from 2.05 to 0.05 is reported
    "loss has barely moved". Correct: a converged run is not a plateau."""
    values = [2.0 * math.exp(-0.5 * i) + 0.05 for i in range(60)]
    raised = feed({"loss": values})
    assert ("plateau", "loss") not in raised, raised


# ---- pairing / relations -----------------------------------------------------------

def test_bug_train_val_pairing_depends_on_dict_order():
    """_check_pairs, _check_widening_gap and _check_eval_noise take the FIRST loss-like
    non-validation name as `train` and the FIRST validation one as `val`, by dict order.
    The CLI builds histories from a set (_detector_histories), so the order is hash
    order. With `val_mae` ahead of `val_loss`, a val_loss that rises while loss falls
    (plain overfitting) is compared as val_mae-vs-loss and never reported. Correct:
    pair each val_X with X (or at least prefer val_loss/loss)."""
    n = 16
    loss = [1.0 * math.exp(-i / 5) + 0.1 for i in range(n)]
    val_loss = [0.9 - 0.05 * i if i < 6 else 0.6 + 0.08 * (i - 6) for i in range(n)]
    flat = noisy([0.3] * n, 0.002, seed=6)
    ordered_bad = {"loss": loss, "val_mae": flat, "val_loss": val_loss, "mae": flat}
    ordered_good = {"loss": loss, "val_loss": val_loss, "mae": flat, "val_mae": flat}
    good = {c for c, v in fired(ordered_good) if v == "val_loss"}
    bad = {c for c, v in fired(ordered_bad) if v == "val_loss"}
    assert {"overfitting", "widening_gap"} & good          # control: found in Keras order
    assert good == bad, (good, bad)


def test_bug_lr_warmup_counts_as_warm_restart():
    """_learning_rate_restarts returns True if the LR ever more than doubles between two
    readings -- which every warmup does. That flag disables progress_lost for the whole
    run, so a run with a warmup that later loses its progress (e.g. a resume that dropped
    optimizer state) is never reported. Correct: only a return to a level the LR has
    already been at counts as a restart."""
    lr = [1e-5, 1e-4, 1e-3] + [1e-3 * (0.97 ** i) for i in range(1, 38)]
    loss = [2.0 * math.exp(-i / 5) + 0.1 for i in range(28)] + [1.2] * 12
    loss = noisy(loss, 0.005, seed=7)
    control = fired({"loss": loss, "lr": [1e-3 * (0.97 ** i) for i in range(40)]})
    assert ("progress_lost", "loss") in control
    raised = fired({"loss": loss, "lr": lr})
    assert ("progress_lost", "loss") in raised, raised


def test_bug_perplexity_relation_uses_wrong_loss():
    """_check_known_relations takes the FIRST loss-like non-validation series as `loss`
    and the FIRST name containing "perplexity" as the perplexity. "perplexity"/"ppl" are
    themselves LOSS_NAME_HINTS, and val perplexity is not a train quantity, so on a
    correct run it compares perplexity to exp(perplexity), or val_perplexity to
    exp(train loss), and reports "perplexity is not exp(loss) ... one of the two is being
    computed wrong". Correct: pair perplexity with the loss of the same split, and never
    with itself."""
    loss = [3.0 - 0.1 * i for i in range(12)]
    val = [3.1 - 0.05 * i for i in range(12)]
    a = fired({"perplexity": [math.exp(v) for v in loss], "loss": loss})
    b = fired({"loss": loss, "val_loss": val, "val_perplexity": [math.exp(v) for v in val]})
    assert "relationship_broken" not in checks_of(a), a
    assert "relationship_broken" not in checks_of(b), b


def test_bug_component_collapse_on_keras_metrics():
    """_check_component_collapse treats every loss-like, non-validation series except a
    few exact names as a *term of the objective*. Keras metrics `mse` and `mae` are
    loss-like, so a healthy regression run compiled with loss="huber",
    metrics=["mse", "mae"] -- where MSE falls quadratically faster than MAE -- is told
    "mse was 50% of the objective and is now 10%: this term has stopped contributing".
    Correct: metrics are not components; no component_collapse here."""
    n = 30
    mae = [0.9 * math.exp(-i / 6) + 0.05 for i in range(n)]
    mse = [m * m * 1.2 for m in mae]
    huber = [0.5 * v for v in mse]
    raised = fired({"loss": huber, "mse": mse, "mae": mae})
    assert "component_collapse" not in checks_of(raised), raised


# ---- tensors ---------------------------------------------------------------------------

def test_bug_dtype_change_on_rebound_loop_variable():
    """The CLI feeds every tracked tensor local's stats as tensor_stats, keyed by the
    Python name. A loop variable reused for different arrays -- `for key, val in
    batch.items()` over float64 features and int32 labels -- is reported as "val changed
    dtype from float64 to int32 mid-run: precision is being lost". (This fired on a real
    benchmark.) The shape changed too, so it is a different object bound to the same
    name. Correct: no dtype_change when the shape changed as well."""
    engine = DetectionEngine(sensitivity=0.3)
    engine.update({}, tensor_stats={"val": {"dtype": "float64", "shape": (32, 10), "device": "cpu",
                                            "mean": 0.1, "std": 1.0, "nan": 0, "inf": 0}})
    out = engine.update({}, tensor_stats={"val": {"dtype": "int32", "shape": (32,), "device": "cpu",
                                                  "mean": 3.0, "std": 2.0, "nan": 0, "inf": 0}})
    assert "tensor_dtype_change" not in {f.check for f in out["raised"]}, out


def test_bug_all_zero_integer_labels_called_uninitialised_layer():
    """tensor_all_zeros fires on any tensor with mean 0 and std 0: a batch of int64
    labels that are all class 0, a padding mask, a freshly zero-initialised bias (the
    Keras default). It says "a layer that was never initialised". An integer tensor is
    never a layer. Correct: no tensor_all_zeros for integer dtypes."""
    engine = DetectionEngine(sensitivity=0.3, confirmations=1)
    out = engine.update({}, tensor_stats={"labels": {"dtype": "int64", "shape": (16,), "device": "cpu",
                                                     "mean": 0.0, "std": 0.0, "nan": 0, "inf": 0}})
    assert "tensor_all_zeros" not in {f.check for f in out["raised"]}, out


def test_bug_norm_explosion_masks_dtype_change():
    """_check_tensor_shape records the new dtype in _tensor_seen and then returns early on
    tensor_norm_explosion, so a dtype change that happens on the same reading as a norm
    explosion is lost for good (next reading compares new-to-new). Correct: the dtype
    change is still reported, on this reading or the next."""
    engine = DetectionEngine(sensitivity=0.3)
    base = {"shape": (64, 64), "device": "cuda:0", "mean": 0.0, "nan": 0, "inf": 0}
    engine.update({}, tensor_stats={"w": dict(base, dtype="float32", std=0.02)})
    a = engine.update({}, tensor_stats={"w": dict(base, dtype="float16", std=1e7)})
    b = engine.update({}, tensor_stats={"w": dict(base, dtype="float16", std=0.02)})
    raised = {f.check for f in a["raised"] + b["raised"]}
    assert "tensor_dtype_change" in raised, raised


# ============================================================================ OK

def test_ok_nan_is_immediate_and_critical():
    out = DetectionEngine().update({"loss": [1.0, 0.9, float("nan")]})
    assert [(f.check, f.severity) for f in out["raised"]] == [("nonfinite", D.CRITICAL)]


def test_ok_numpy_float32_nan_detected():
    np = pytest.importorskip("numpy")
    out = DetectionEngine().update({"loss": [np.float32(1.0), np.float32(0.5), np.float32("nan")]})
    assert "nonfinite" in {f.check for f in out["raised"]}


def test_ok_inf_reported_as_infinite():
    out = DetectionEngine().update({"loss": [1.0, float("inf")]})
    assert "infinite" in out["raised"][0].message


@pytest.mark.parametrize("history", [[], [1.0], [None], [True, False], ["0.5", "0.4"],
                                     [None, None, 0.5], [1.0, None, 0.8, "x", 0.7]])
def test_ok_degenerate_histories_do_not_crash(history):
    engine = DetectionEngine()
    for _ in range(3):
        engine.update({"loss": history, "val_loss": history, "accuracy": history})


def test_ok_empty_update_and_no_histories():
    engine = DetectionEngine()
    assert engine.update({}) == {"raised": [], "cleared": []}
    assert engine.update(None) == {"raised": [], "cleared": []}


def test_ok_huge_magnitudes_single_series_do_not_crash():
    engine = DetectionEngine()
    for _ in range(2):
        engine.update({"loss": [1e300 * (1 + 0.01 * i) for i in range(40)],
                       "grad_norm": [1e290 * (1 + 0.1 * i) for i in range(40)]})


def test_ok_healthy_decaying_run_is_quiet():
    n = 40
    hist = {"loss": noisy([2.0 * math.exp(-i / 8) + 0.1 for i in range(n)], 0.01),
            "val_loss": noisy([2.1 * math.exp(-i / 9) + 0.15 for i in range(n)], 0.015, seed=2),
            "accuracy": [min(0.97, 0.3 + 0.02 * i) for i in range(n)]}
    assert feed(hist) == []


def test_ok_confirmation_needed_for_warning_checks():
    engine = DetectionEngine(confirmations=2)
    frozen = {"loss": [1.0, 0.9, 0.8] + [0.7] * 6}
    assert engine.update(frozen)["raised"] == []
    assert [f.check for f in engine.update(frozen)["raised"]] == ["frozen"]


def test_ok_finding_clears_after_rearm_and_can_fire_again():
    engine = DetectionEngine(confirmations=1, rearm_after=2)
    bad = {"loss": [1.0, 0.9, 0.8] + [0.7] * 6}
    good = {"loss": [1.0, 0.9, 0.8] + [0.7] * 5 + [0.6]}
    assert engine.update(bad)["raised"]
    assert engine.update(good)["cleared"] == []
    assert [f.check for f in engine.update(good)["cleared"]] == ["frozen"]
    assert [f.check for f in engine.update(bad)["raised"]] == ["frozen"]


def test_ok_loss_spike_fires_immediately():
    values = noisy([1.0] * 30, 0.01) + [50.0]
    engine = DetectionEngine()
    assert ("loss_spike", "loss") in [(f.check, f.variable) for f in engine.update({"loss": values})["raised"]]


def test_ok_val_accuracy_drop_is_score_regression():
    acc = [0.5 + 0.04 * i for i in range(12)] + [0.9, 0.7, 0.5, 0.4, 0.35]
    assert ("score_regression", "val_accuracy") in fired({"val_accuracy": acc})


def test_ok_divergence_detected():
    values = noisy([0.5 + 0.05 * i for i in range(30)], 0.005)
    assert ("divergence", "loss") in fired({"loss": values})


def test_ok_moving_target_loss_skips_directional_checks():
    values = noisy([0.5 + 0.05 * i for i in range(30)], 0.005)
    assert ("divergence", "g_loss") not in fired({"g_loss": values})


def test_ok_lr_warm_restart_not_reported_as_jump():
    """A restart back to a peak the schedule has already used is not an lr_jump."""
    cycle = [1e-3 * (0.5 ** i) for i in range(10)]
    engine = DetectionEngine(confirmations=1)
    out = engine.update({"lr": cycle * 2 + cycle[:1]})
    assert "lr_jump" not in {f.check for f in out["raised"]}


def test_ok_steps_is_not_an_optimizer_eps():
    assert ("hyperparameter_out_of_range", "steps") not in fired({"steps": [5.0] * 12})


def test_ok_adam_eps_out_of_range_detected():
    raised = fired({"adam_eps": [0.1] * 10, "loss": [1.0 - 0.05 * i for i in range(10)]})
    assert ("hyperparameter_out_of_range", "adam_eps") in raised


def test_ok_perplexity_relation_checked():
    loss = [3.0 - 0.1 * i for i in range(10)]
    good = [math.exp(v) for v in loss]
    bad = [2 * math.exp(v) for v in loss]
    assert ("relationship_broken", "perplexity") not in fired({"loss": loss, "perplexity": good})
    assert ("relationship_broken", "perplexity") in fired({"loss": loss, "perplexity": bad})


def test_ok_thresholds_clamped_and_monotone():
    lo, hi = D.thresholds(-5), D.thresholds(7)
    assert lo == D.thresholds(0.0) and hi == D.thresholds(1.0)
    assert lo["explosion_multiplier"] > hi["explosion_multiplier"]
    assert lo["plateau_range_frac"] < hi["plateau_range_frac"]


def test_ok_overrides_take_precedence():
    values = noisy([1.0] * 30, 0.01) + [3.0]
    assert ("loss_spike", "loss") not in fired({"loss": values})
    engine = DetectionEngine(confirmations=1, overrides={"explosion_multiplier": 2.0})
    assert "loss_spike" in {f.check for f in engine.update({"loss": values})["raised"]}


def test_ok_group_divergence_across_ranks():
    n = 20
    a = [1.0 - 0.02 * i for i in range(n)]
    b = [1.0 - 0.02 * i for i in range(10)] + [0.8 + 0.1 * i for i in range(10)]
    raised = fired({"loss_rank0": a, "loss_rank1": b})
    assert "group_diverged" in checks_of(raised)


def test_ok_numbered_metrics_are_not_a_group():
    assert DetectionEngine._group_key("metric_3") is None
    assert DetectionEngine._group_key("loss_rank3") == "loss_rank"


def test_ok_tensor_nan_and_device_move():
    engine = DetectionEngine()
    out = engine.update({}, tensor_stats={"w": {"nan": 3, "inf": 0}})
    assert [f.check for f in out["raised"]] == ["tensor_nonfinite"]
    engine = DetectionEngine()
    engine.update({}, tensor_stats={"w": {"device": "cuda:0", "dtype": "float32"}})
    out = engine.update({}, tensor_stats={"w": {"device": "cpu", "dtype": "float32"}})
    assert [f.check for f in out["raised"]] == ["tensor_device_change"]


def test_ok_analysis_window_bounds_cost():
    values = [1.0 / (1 + i) for i in range(20000)]
    engine = DetectionEngine()
    engine.update({"loss": values})     # completes; histories truncated to ANALYSIS_WINDOW


def test_ok_summarise_orders_by_severity():
    a = D.Finding("x", "v", D.WARNING, "warn", confidence=0.9)
    b = D.Finding("y", "v", D.CRITICAL, "crit", confidence=0.1)
    assert D.summarise([a, b]) == "crit; warn"
    assert D.summarise([]) == ""


def test_ok_keras_logs_nan_reaches_cli_detector():
    cli = make_cli()
    for epoch in range(5):
        cli._record_keras_logs({"loss": 1.0 / (epoch + 1), "val_loss": 1.1 / (epoch + 1)}, epoch=epoch)
    assert cli._check_for_trouble() is None
    cli._record_keras_logs({"loss": float("nan"), "val_loss": float("nan")}, epoch=5)
    problem = cli._check_for_trouble()
    assert problem and "NaN" in problem


# ============================================================================ DET ledger items
# confirmed by reading (no test in the sweep); each one below asserts the fixed behaviour.

def test_bug_keras_loss_finding_reported_twice_via_train_loss_alias():
    """_record_keras_logs stores "train_loss" as an alias of Keras's "loss" (for the legacy
    detector); the engine saw two copies of one series and every loss finding came twice.
    Correct: the detector gets the loss once."""
    cli = make_cli()
    for epoch in range(3):
        cli._record_keras_logs({"loss": 1.0 / (epoch + 1)}, epoch=epoch)
    histories = cli._detector_histories()
    assert "loss" in histories and "train_loss" not in histories, sorted(histories)
    cli._record_keras_logs({"loss": float("nan")}, epoch=3)
    problem = cli._check_for_trouble()
    assert problem and problem.count("NaN") == 1, problem


def test_bug_finding_step_freezes_at_history_cap():
    """_check_for_trouble passed step=max(len(history)), which stops at the 2,000-reading
    cap. Correct: the run's own step."""
    cli = make_cli()
    cli.step = 5000
    cli._detector_scalar_histories = {"loss": [0.5] * 1999 + [float("nan")]}
    cli.scalar_histories = {"loss": [0.5, float("nan")]}
    cli._check_for_trouble()
    assert [f.step for f in cli._detection_engine().current()] == [5000]


def test_bug_logit_scale_collapse_message_is_backwards():
    """logit_scale multiplies the logits (an inverse temperature): shrinking flattens the
    softmax towards uniform, not towards a hard argmax."""
    values = [4.6 * math.exp(-i / 5) for i in range(20)]
    engine = DetectionEngine(confirmations=1)
    found = [f for f in engine.update({"logit_scale": values})["raised"]
             if f.check == "temperature_collapse"]
    assert found and "uniform" in found[0].message and "argmax" not in found[0].message


def test_bug_throughput_hint_matches_unrelated_names():
    """THROUGHPUT hint `it_s` matched init_std, split_size and logit_scale."""
    for name in ("init_std", "split_size", "logit_scale", "clips"):
        assert not D.looks_like_throughput(name), name
    for name in ("it_s", "train_it_s", "samples_per_sec", "fps"):
        assert D.looks_like_throughput(name), name


def test_bug_numpy_array_histories_raise():
    """`if not history` on a numpy array raises ValueError out of update()."""
    np = pytest.importorskip("numpy")
    out = DetectionEngine().update({"loss": np.array([1.0, 0.9, np.nan])})
    assert [f.check for f in out["raised"]] == ["nonfinite"]
    DetectionEngine().update({"loss": np.array([])})


def test_bug_cpu_data_batch_called_a_layer_left_on_cpu():
    """A data tensor seen first on the CPU (before .to(device)) while weights are on the
    GPU was reported as a layer on the wrong device. Only parameters count."""
    engine = DetectionEngine()
    engine.update({}, tensor_stats={"w": {"device": "cuda:0", "dtype": "float32"}})
    out = engine.update({}, tensor_stats={"x": {"device": "cpu", "dtype": "float32"}})
    assert "tensor_device_change" not in {f.check for f in out["raised"]}
    out = engine.update({}, tensor_stats={"b": {"device": "cpu", "dtype": "float32",
                                                "requires_grad": True}})
    assert "tensor_device_change" in {f.check for f in out["raised"]}


def test_ok_epoch_level_local_is_not_frozen_by_repeated_sampling():
    """The per-sample detector history must not turn an epoch-level local that is simply
    not reassigned between samples into a 'frozen' series."""
    cli = make_cli()
    cli.continuous = True
    cli.auto_intervene = False
    cli.tracked_vars = ["val_loss"]
    val_loss = 0.8
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(20):
            cli.watch_locals = {"val_loss": val_loss}      # the same object every sample
            cli.update()
    assert len(cli._detector_histories().get("val_loss", [])) == 1
