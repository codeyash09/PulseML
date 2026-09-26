"""Bug sweep 2: the detection engine (pulse_detect.py) and the CLI's detector wiring.

Every test_bug_* asserts the CORRECT behaviour, so it fails on the current code and
passes once the bug is fixed. Every test_ok_* is regression coverage for behaviour that
was verified to work.
"""
import contextlib
import io
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

def feed(histories, start=3, engine=None, **kw):
    """Grow every history one reading at a time, as a live run does; collect every
    finding raised along the way as (check, variable)."""
    engine = engine or DetectionEngine(**kw)
    longest = max(len(v) for v in histories.values())
    raised = []
    for k in range(start, longest + 1):
        out = engine.update({n: list(v[:k]) for n, v in histories.items()}, step=k)
        raised.extend((f.check, f.variable) for f in out["raised"])
    return raised


def fired(histories, sensitivity=0.3):
    engine = DetectionEngine(sensitivity=sensitivity, confirmations=1)
    return [(f.check, f.variable) for f in engine.update(histories)["raised"]]


def checks_of(pairs, variable=None):
    return {c for c, v in pairs if variable is None or v == variable}


def noisy(values, sigma, seed=0):
    rng = random.Random(seed)
    return [v + rng.gauss(0.0, sigma) for v in values]


def make_cli():
    from pulse import pulse_cli
    cli = pulse_cli.PulseCLI(watch_locals={}, pdf_dir=os.path.join("/tmp", "_s2_detect_pdfs"))
    cli.sensitivity = 0.3
    cli.epoch_scalar_histories = {}
    cli.batch_scalar_histories = {}
    return cli


def run_updates(cli, frames):
    """Drive the real PulseCLI.update() the way the tracer does (no explicit step)."""
    cli.continuous = True
    cli.auto_intervene = False
    with contextlib.redirect_stdout(io.StringIO()):
        for frame in frames:
            cli.watch_locals.update(frame)
            cli.update()


# A steadily climbing training loss: every name for "the training loss" must see it.
DIVERGING = noisy([0.5 + 0.02 * i for i in range(60)], 0.01, seed=1)


# ============================================================================ BUGS

def test_bug_running_and_avg_loss_are_treated_as_gan_losses():
    """MOVING_TARGET_HINTS is matched as a SUBSTRING (pulse_detect.py looks_like_moving_target),
    so "g_loss" matches runnin[g_loss], av[g_loss], trainin[g_loss], lo[g_loss]; "d_loss"
    matches embe[d_loss], pre[d_loss], weighte[d_loss]; "q_loss" matches se[q_loss]. The
    canonical PyTorch names for the training loss (running_loss, avg_loss) and fastai's
    validation loss (vali[d_loss]) are therefore classed as GAN/RL
    objectives and divergence, plateau, stagnation, never_learned, progress_lost,
    oscillation and val_regression are all switched off for them: a running_loss that
    climbs for 60 readings raises nothing, while the identical curve named `loss` raises
    divergence. Correct: match the hints as whole name tokens / suffix words."""
    assert checks_of(feed({"loss": DIVERGING})) >= {"divergence"}
    for name in ("running_loss", "avg_loss", "training_loss", "valid_loss", "reg_loss",
                 "seg_loss", "log_loss", "seq_loss", "embed_loss", "pred_loss", "weighted_loss"):
        assert "divergence" in checks_of(feed({name: DIVERGING})), name
        assert not D.looks_like_moving_target(name), name


def test_bug_frozen_loss_never_confirmed_once_history_hits_the_cap():
    """The require_new_data fingerprint (DetectionEngine._evidence) is
    (len(history), repr(history[-1])). The CLI caps every detector history at 2,000
    readings, so once a run is past that the length never changes -- and a loss that has
    FROZEN repeats its last value, so the fingerprint never changes either. `frozen`
    fires, gets streak 1, and every later check is treated as "the same data again":
    it is never confirmed. A step-level run whose loss freezes after ~2,000 samples
    (dead ReLUs, a detached graph after a code path change) is never reported, although
    the same freeze before the cap is. Correct: the fingerprint must advance with every
    new reading (e.g. count appends, not the capped length)."""
    rng = random.Random(0)
    healthy = [2.0 * math.exp(-i / 400) + 0.3 + rng.gauss(0, 0.02) for i in range(2000)]
    cli = make_cli()
    cli._detector_scalar_histories = {"loss": list(healthy)}
    cli.scalar_histories = {"loss": list(healthy)}
    problems = []
    for _ in range(60):
        # loss.item() on a stuck loss: a new float object with the same value each step
        cli._record_detector_scalar("loss", float(repr(0.65)), source=None)
        problem = cli._check_for_trouble()
        if problem:
            problems.append(problem)
    assert len(cli._detector_scalar_histories["loss"]) == 2000
    engine = cli._detection_engine()
    assert engine._streak.get(("frozen", "loss")) >= 1      # it fires on every check...
    assert problems, "a loss frozen at 0.65 for 60 new readings was never reported"


def test_ok_frozen_loss_before_the_cap_is_confirmed():
    """Same freeze, history not yet at the cap: confirmed on the next new reading."""
    rng = random.Random(0)
    healthy = [2.0 * math.exp(-i / 400) + 0.3 + rng.gauss(0, 0.02) for i in range(1500)]
    engine = DetectionEngine(require_new_data=True)
    raised = []
    history = list(healthy)
    for _ in range(12):
        history.append(0.65)
        raised += [f.check for f in engine.update({"loss": history})["raised"]]
    assert "frozen" in raised


def test_bug_opening_level_survives_a_history_reset():
    """DetectionEngine._openings is written once per name when a history first outgrows
    ANALYSIS_WINDOW and is never forgotten -- not when the history is cleared
    (/delete, /delete all, a second fit() into a fresh history) and restarts short. The
    stale opening then drives `learned_earlier`, `converged` and stagnation's
    `progress` for the NEW run: a run whose loss sits flat at 1.0 is not reported as
    never_learned because a previous run once opened at 1.37. Correct: a history that
    got shorter than it was is a new series; its opening is forgotten."""
    rng = random.Random(3)
    first_run = [2.0 * math.exp(-i / 300) + 0.1 + rng.gauss(0, 0.01) for i in range(1600)]
    flat = [1.0 + rng.gauss(0, 0.005) for _ in range(200)]
    fresh = checks_of(feed({"loss": flat}))
    assert "never_learned" in fresh          # what a fresh engine says about this run
    engine = DetectionEngine()
    engine.update({"loss": first_run})
    after_reset = checks_of(feed({"loss": flat}, engine=engine))
    assert "never_learned" in after_reset, after_reset


def test_bug_opening_level_survives_cli_delete_all():
    """The same leak through the CLI: /delete all clears the histories but keeps the
    engine (self._detector) and its openings, so a re-tracked `loss` inherits the old
    run's opening."""
    cli = make_cli()
    rng = random.Random(4)
    cli.tracked_vars = ["loss"]
    cli._detector_scalar_histories = {"loss": [2.0 * math.exp(-i / 300) + 0.1 + rng.gauss(0, 0.01)
                                               for i in range(1600)]}
    cli.scalar_histories = {"loss": list(cli._detector_scalar_histories["loss"])}
    cli._check_for_trouble()
    with contextlib.redirect_stdout(io.StringIO()):
        cli._cmd_delete("all")
    assert "loss" not in cli._detection_engine()._openings


def test_bug_dcgan_generator_loss_flagged_as_divergence():
    """A healthy GAN's generator loss rises as the discriminator improves; that is why
    g_loss/gen_loss/generator_loss are moving targets. The names the PyTorch DCGAN tutorial
    (errD/errG) and CycleGAN/pix2pix (loss_G/loss_D) use are not in MOVING_TARGET_HINTS, and
    neither is lossG/lossD, so the same healthy curve that is silent as `g_loss` raises
    divergence and progress_lost. Correct: these are moving targets too."""
    n = 40
    d_curve = noisy([1.3 * math.exp(-i / 15) + 0.4 for i in range(n)], 0.03, 1)
    g_curve = noisy([1.5 + 2.5 * (1 - math.exp(-i / 15)) for i in range(n)], 0.05, 2)
    assert feed({"d_loss": d_curve, "g_loss": g_curve}) == []
    for d_name, g_name in (("errD", "errG"), ("loss_D", "loss_G"), ("lossD", "lossG")):
        raised = feed({d_name: d_curve, g_name: g_curve})
        assert raised == [], (g_name, raised)


def test_bug_rl_value_and_policy_loss_names_flagged_as_divergence():
    """The value loss of a healthy PPO/A2C run grows as returns grow. `value_loss` is a
    moving target, but RLlib's `vf_loss`, CleanRL's `v_loss` and SB3's
    `policy_gradient_loss` are not, so the same curve raises a divergence. Correct:
    all of them are directionless objectives."""
    n = 40
    value_curve = noisy([5 + 20 * (1 - math.exp(-i / 12)) for i in range(n)], 0.5, 4)
    assert feed({"value_loss": value_curve}) == []
    for name in ("vf_loss", "v_loss", "policy_gradient_loss"):
        assert "divergence" not in checks_of(feed({name: value_curve})), name
        assert D.looks_like_moving_target(name), name


def test_bug_dpo_rejected_reward_flagged_as_score_regression():
    """TRL's DPOTrainer logs rewards/rejected, which is SUPPOSED to fall (the policy
    pushes the rejected completion's implicit reward down; margins grow). "reward" is a
    score hint, so a healthy DPO run gets a CRITICAL score_regression. Correct: a
    healthy DPO run -- margins rising, loss falling -- raises nothing."""
    n = 40
    run = {"rewards/chosen": noisy([0.02 + 0.3 * i / n for i in range(n)], 0.03, 7),
           "rewards/rejected": noisy([0.02 - 4.0 * i / n for i in range(n)], 0.05, 8),
           "rewards/margins": noisy([4.3 * i / n for i in range(n)], 0.05, 9),
           "loss": noisy([0.69 * math.exp(-i / 15) + 0.1 for i in range(n)], 0.01, 10)}
    raised = feed(run)
    assert ("score_regression", "rewards/rejected") not in raised, raised


def test_bug_rlhf_kl_to_reference_flagged_as_divergence():
    """`kl` is a loss hint, so TRL PPO's objective/kl (KL from the reference policy, which
    grows by design as the policy moves) and PPO's approx_kl diagnostic are classed as
    losses: a KL that rises from 0 to 6 over a healthy RLHF run is reported as a loss
    "climbing, not falling". Correct: a KL diagnostic is not an objective that should
    fall; no divergence."""
    n = 40
    kl = noisy([6.0 * (1 - math.exp(-i / 20)) + 0.05 for i in range(n)], 0.05, 11)
    for name in ("objective/kl", "approx_kl"):
        raised = checks_of(feed({name: kl}))
        assert "divergence" not in raised and "progress_lost" not in raised, (name, raised)


def test_bug_keras_numpy_nan_loss_reported_twice_via_alias():
    """_detector_histories drops the train_loss alias only when
    `histories["train_loss"] == histories["loss"]`. _record_keras_logs converts the same
    log value twice with float(); for a numpy scalar (np.float32 / a tf tensor's .numpy())
    each float() is a new object, and a NaN is not equal to another NaN object, so the
    lists compare unequal from the first NaN on and every finding is reported twice
    ("train_loss is NaN; ... loss is NaN"). Correct: the alias is dropped by
    construction (it is a copy), and NaN is reported once."""
    np = pytest.importorskip("numpy")
    cli = make_cli()
    for epoch in range(4):
        cli._record_keras_logs({"loss": np.float32(1.0 / (epoch + 1))}, epoch=epoch)
    cli._record_keras_logs({"loss": np.float32("nan")}, epoch=4)
    assert "train_loss" not in cli._detector_histories()
    problem = cli._check_for_trouble()
    assert problem and problem.count("NaN") == 1, problem


def test_bug_tracer_mode_counter_stalled_can_never_fire():
    """_record_detector_scalar skips a repeated value when the local is the same object
    and no explicit step was passed -- which is always the case for a counter that has
    stopped counting under the tracer (update() is called without a step, and an int
    that is not reassigned is the same object; so is any small int recomputed with
    len()). The counter's detector history therefore stays at ONE reading however long
    the run goes on, and counter_stalled ("40 hours and nothing on disk") can never fire
    on a `pulse run` script. Correct: a stalled counter is seen as stalled while the
    rest of the run moves."""
    from pulse import pulse_cli  # noqa: F401
    cli = make_cli()
    cli.tracked_vars = ["loss", "checkpoints_saved"]
    checkpoints_saved = 3
    frames = [{"loss": float(repr(2.0 * math.exp(-i / 20) + 0.1)),
               "checkpoints_saved": checkpoints_saved} for i in range(40)]
    run_updates(cli, frames)
    assert len(cli._detector_histories()["loss"]) >= 30        # the run is moving
    assert ("counter_stalled", "checkpoints_saved") in fired(cli._detector_histories())


def test_bug_tracer_mode_hyperparameter_check_can_never_fire():
    """Same root cause: a configured constant (adam_eps = 1e-2, dropout = 0.95) is one
    unchanging object, so its detector history is one reading, and
    _check_relations only looks at series with >= 8 readings. The
    hyperparameter_out_of_range check can never fire under the tracer. Correct: a
    tracked constant outside its sane range is reported."""
    cli = make_cli()
    cli.tracked_vars = ["loss", "adam_eps"]
    adam_eps = 1e-2
    frames = [{"loss": float(repr(2.0 * math.exp(-i / 20) + 0.1)), "adam_eps": adam_eps}
              for i in range(40)]
    run_updates(cli, frames)
    assert ("hyperparameter_out_of_range", "adam_eps") in fired(cli._detector_histories())


def test_bug_static_all_zero_tensor_can_never_be_confirmed_in_cli():
    """The CLI passes _matrix_cache stats to the engine with require_new_data=True. A
    tensor's fingerprint is repr(its stats); statistics() of a weight that stays all-zero
    (never initialised, or gradients cannot reach it -- exactly what tensor_all_zeros
    describes) is identical on every probe, so the WARNING gets streak 1 and is never
    confirmed however many new probes and new loss readings arrive. The check can
    never fire from the CLI. Correct: a condition that persists across new probes /
    new data is confirmed."""
    np = pytest.importorskip("numpy")
    from pulse.pulse_backend import statistics
    cli = make_cli()
    rng = random.Random(5)
    loss = [2.0 * math.exp(-i / 10) + 0.2 + rng.gauss(0, 0.005) for i in range(30)]
    cli._detector_scalar_histories = {"loss": loss[:20]}
    cli.scalar_histories = {"loss": loss[:20]}
    problems = []
    for i in range(10):
        # a fresh probe of the (still all-zero) weight, and a new loss reading
        cli._matrix_cache = {"fc2.weight": {"base_name": "fc2.weight",
                                            "stats": statistics(np.zeros((64, 32), np.float32))}}
        cli._record_detector_scalar("loss", loss[20 + i], source=None)
        problem = cli._check_for_trouble()
        if problem:
            problems.append(problem)
    engine = cli._detection_engine()
    assert engine._streak.get(("tensor_all_zeros", "fc2.weight")) >= 1   # fires every time...
    assert any("entirely zero" in p for p in problems), problems


def test_bug_slash_named_learning_rate_is_not_a_learning_rate():
    """looks_like_lr only accepts `lr`, `*_lr`, `lr_*` (after '-' -> '_'), so the names a
    run mirrors from W&B/TensorBoard -- train/lr, optim/lr, train/learning_rate -- are not
    learning rates: a 1000x LR jump raises nothing, and _learning_rate_restarts never
    sees an SGDR schedule. Correct: '/' separates words like '_' does."""
    lr = [1e-3] * 10 + [1.0] * 3
    assert "lr_jump" in checks_of(feed({"lr": lr}))
    for name in ("train/lr", "optim/lr", "train/learning_rate"):
        assert "lr_jump" in checks_of(feed({name: lr})), name
        assert D.looks_like_lr(name), name


def test_bug_auroc_is_not_a_score():
    """AUROC (torchmetrics' name, `val_auroc`) matches no metric hint -- `auc` is matched
    as a whole token -- so a validation AUROC that falls from 0.95 to 0.60 raises no
    score_regression, while the same curve named val_auc does. Correct: AUROC/AUPRC are
    scores."""
    n = 30
    curve = [0.95 - (0.35 * (i - 15) / 15 if i > 15 else 0.0) for i in range(n)]
    curve = noisy([min(0.95, 0.6 + 0.35 * i / 10) if i <= 15 else c for i, c in enumerate(curve)], 0.003, 12)
    assert "score_regression" in checks_of(feed({"val_auc": curve}))
    assert "score_regression" in checks_of(feed({"val_auroc": curve}))


def test_bug_normal_start_anchor_never_expires():
    """NORMAL_START is announced as "a spike-detection baseline until real data
    accumulates" (pulse_cli.py _apply_directives), but _check_loss uses
    `baseline = min(floor, anchor)` forever. When the agent's estimate is lower than
    where a healthy run actually lives (MSE on unnormalised targets: estimated 1.0,
    real ~50), every ordinary 4-sigma bump of an otherwise smooth loss is a CRITICAL
    "spiked to 30, 30x its recent floor of 1" -- hundreds of readings in. Correct:
    once the run has its own history the anchor stops overriding it."""
    rng = random.Random(6)
    loss = [50.0 * math.exp(-i / 400) + rng.gauss(0, 0.2) for i in range(300)]
    loss.append(loss[-1] + 6.0)          # a +25% bump on one epoch, nothing more
    assert ("loss_spike", "loss") not in feed({"loss": loss}, start=len(loss) - 3)
    engine = DetectionEngine()
    engine.set_baseline("loss", 1.0)
    raised = feed({"loss": loss}, start=len(loss) - 3, engine=engine)
    assert ("loss_spike", "loss") not in raised, raised


# ============================================================================ OK

@pytest.mark.parametrize("name", ["acc@1", "top1_acc", "f1_macro", "trainAcc", "valAccuracy",
                                  "mAP50", "mIoU", "val_acc", "test_acc"])
def test_ok_metric_names_are_metrics(name):
    assert D.looks_like_metric(name) and not D.looks_like_loss(name)


@pytest.mark.parametrize("name", ["grad_accum_steps", "indices", "heatmap", "layer2", "previous",
                                  "kl_weight", "kl_beta", "loss_scale"])
def test_ok_short_hints_do_not_match_inside_words(name):
    assert not D.looks_like_metric(name)
    assert not D.looks_like_loss(name)


def test_ok_val_pairing_uses_matching_stem():
    names = ["loss", "mae", "val_mae", "val_loss"]
    assert dict((v, t) for t, v in D._train_val_pairs(names)) == {"val_loss": "loss", "val_mae": "mae"}


def test_ok_named_gan_losses_are_quiet():
    n = 40
    d_curve = noisy([1.3 * math.exp(-i / 15) + 0.4 for i in range(n)], 0.03, 1)
    g_curve = noisy([1.5 + 2.5 * (1 - math.exp(-i / 15)) for i in range(n)], 0.05, 2)
    assert feed({"d_loss": d_curve, "g_loss": g_curve}) == []


def test_ok_contrastive_loss_near_zero_is_quiet():
    loss = noisy([2.0 * math.exp(-i / 5) + 0.001 for i in range(60)], 0.0005, 11)
    assert feed({"loss": loss}) == []


def test_ok_negative_r2_climbing_is_quiet():
    r2 = noisy([-2 + 2.85 * (1 - math.exp(-i / 5)) for i in range(40)], 0.01, 13)
    assert feed({"val_r2": r2}) == []


def test_ok_recheck_of_unchanged_epoch_does_not_confirm():
    engine = DetectionEngine(require_new_data=True)
    hist = {"loss": [0.6931] * 10}
    out1 = engine.update(hist)["raised"]
    out2 = engine.update(hist)["raised"]
    assert not out1 and not out2
    out3 = engine.update({"loss": [0.6931] * 11})["raised"]
    assert "frozen" in {f.check for f in out3}


def test_ok_checks_are_isolated_on_error():
    engine = DetectionEngine()

    def boom(*a, **k):
        raise RuntimeError("x")
    engine._check_pairs = boom
    out = engine.update({"loss": [1.0, 0.9, float("nan")]})
    assert [f.check for f in out["raised"]] == ["nonfinite"]
    assert engine.last_error and "RuntimeError" in engine.last_error


def test_bug_second_keras_fit_reads_as_a_score_regression():
    """PulseKerasTracker.on_train_begin does nothing when epoch_scalar_histories already
    exists, so a second model.fit() in the same script -- k-fold cross-validation, a
    hyperparameter loop, `for fold in range(5): model = build(); model.fit(...)` -- is
    appended to the first fit's history. The detector then sees fold 1's accuracy of 0.95
    "fall" to a fresh model's 0.36 and raises CRITICAL score_regression on accuracy and
    val_accuracy two epochs into fold 2 -- pausing a perfectly healthy script for the
    agent (and progress_lost etc. follow on the concatenated loss).
    Correct: a new fit() starts a new series (histories and engine state), so the
    second fold's opening epoch is not a spike."""
    from pulse import pulse_cli

    class _Base:
        def __init__(self):
            pass

    saved = pulse_cli._PULSE_KERAS_TRACKER_CLS
    pulse_cli._PULSE_KERAS_TRACKER_CLS = None
    try:
        tracker_cls = pulse_cli._build_keras_tracker_class(_Base)
    finally:
        pulse_cli._PULSE_KERAS_TRACKER_CLS = saved
    cli = make_cli()
    cli.auto_intervene = False
    cli.continuous = True
    tracker = tracker_cls(cli)
    tracker.model, tracker.params = None, {"epochs": 20}
    problems = []
    with contextlib.redirect_stdout(io.StringIO()):
        for fold in range(2):
            tracker.on_train_begin()
            for epoch in range(20):
                logs = {"loss": 2.0 * math.exp(-epoch / 4) + 0.3 + 0.002 * fold,
                        "accuracy": min(0.95, 0.3 + 0.04 * epoch),
                        "val_loss": 2.0 * math.exp(-epoch / 4) + 0.35,
                        "val_accuracy": min(0.93, 0.28 + 0.037 * epoch)}
                tracker.on_epoch_end(epoch, logs)
                problem = cli._check_for_trouble()
                if problem:
                    problems.append((fold, epoch, problem))
    assert not problems, problems


def test_bug_vae_kl_term_rising_reported_as_divergence():
    """In a healthy VAE the KL term RISES while the reconstruction term and the total fall
    -- the encoder starts using its latent (and with KL annealing it rises by design).
    `kl_loss` is checked as a standalone loss, so every such run gets divergence (CRITICAL
    once the term doubles) and progress_lost on kl_loss: 20/20 seeds in a replay.
    Correct: a component of a composite objective whose total is improving is not
    diverging; no divergence/progress_lost on the KL term."""
    n = 100
    rng = random.Random(3000)
    run = {"kl_loss": [5 + 15 * (1 - math.exp(-i / 30)) + rng.gauss(0, 0.3) for i in range(n)],
           "recon_loss": [120 - 60 * (1 - math.exp(-i / 40)) + rng.gauss(0, 1) for i in range(n)],
           "loss": [125 - 45 * (1 - math.exp(-i / 40)) + rng.gauss(0, 1) for i in range(n)]}
    raised = checks_of(feed(run), "kl_loss")
    assert not raised & {"divergence", "progress_lost"}, raised


# ============================================================================ FROZEN vs CONVERGED
#
# The confirmation fix above lets `frozen` (and plateau/stagnation, which share its
# fingerprint) be confirmed past the 2,000-reading cap. A run that is genuinely
# converging can repeat a value exactly too -- rounded logging, float32, a loss that
# has reached 0.0, a fixed point -- and must never be told it is frozen, before or after
# the cap. Every run below is a long, realistic, healthy curve fed through the CLI the
# way a live run is (one reading at a time, capped histories, the engine kept across
# checks) and also straight into an uncapped engine.

def _cli_findings(values, stride=7, keras=False):
    """Every (check, severity) the CLI's engine held at any point of the run."""
    np = pytest.importorskip("numpy")
    cli = make_cli()
    cli.tracked_vars = ["loss"]
    seen = set()
    with contextlib.redirect_stdout(io.StringIO()):
        for i, value in enumerate(values):
            if keras:
                cli._record_keras_logs({"loss": np.float32(value)}, epoch=i)
            else:
                cli._record_detector_scalar("loss", float(value), source=None)
            if i >= 3 and (i % stride == 0 or i == len(values) - 1):
                cli._check_for_trouble()
                seen |= {(f.check, f.severity) for f in cli._detection_engine().active.values()}
    return seen


def _engine_findings(values, stride=13):
    engine = DetectionEngine(require_new_data=True)
    seen = set()
    for k in range(4, len(values) + 1, stride):
        engine.update({"loss": values[:k]}, step=k)
        seen |= {(f.check, f.severity) for f in engine.active.values()}
    return seen


def _descent(n, start=2.3, floor=0.5, tau=150, sigma=1e-3, seed=0):
    rng = random.Random(seed)
    return [floor + (start - floor) * math.exp(-i / tau) + rng.gauss(0, sigma) for i in range(n)]


def _assert_not_frozen(values, keras=False):
    for seen in (_cli_findings(values, keras=keras), _engine_findings(values)):
        checks = {c for c, _ in seen}
        assert not checks & {"frozen", "repeating"}, sorted(seen)


def test_ok_loss_creeping_down_1e6_per_step_is_not_frozen():
    rng = random.Random(20)
    values = _descent(1000, seed=20)
    level = values[-1]
    for _ in range(9000):
        level *= 1 - 1e-6
        values.append(level * (1 + rng.gauss(0, 1e-9)))
    _assert_not_frozen(values)


def test_ok_converged_loss_jittering_on_its_floor_is_not_frozen():
    rng = random.Random(21)
    values = [0.3 + 2.0 * math.exp(-i / 200) + rng.gauss(0, 0.002) for i in range(10000)]
    _assert_not_frozen(values)


@pytest.mark.parametrize("jitter", [2e-6, 3e-5])
def test_ok_loss_logged_rounded_to_4_decimals_is_not_frozen(jitter):
    """round(loss, 4): once the run improves by less than 1e-4 per step the logged value
    repeats for hundreds of readings while the model is still learning -- or flickers
    between two neighbouring values when it sits on a rounding boundary."""
    rng = random.Random(22)
    values = [round(0.3 + 2.0 * math.exp(-i / 400) + rng.gauss(0, jitter), 4) for i in range(10000)]
    _assert_not_frozen(values)
    # Past the cap the fingerprint now confirms what repeats; a range of zero that is
    # only the rounding is not a plateau, and rounding jitter is not a cycle.
    assert not {c for c, _ in _cli_findings(values)} & {"plateau", "periodic"}


def test_ok_loss_printed_as_float32_and_parsed_back_is_not_frozen():
    np = pytest.importorskip("numpy")
    rng = random.Random(23)
    values = _descent(1000, seed=23)
    level = values[-1]
    for _ in range(7000):
        level *= 1 - 2e-8
        values.append(float("%.8g" % np.float32(level * (1 + rng.gauss(0, 1e-10)))))
    _assert_not_frozen(values)


def test_ok_keras_epoch_loss_converged_onto_a_fixed_point_is_not_frozen():
    """Full-batch Keras, no shuffling: the float32 epoch loss converges onto one value and
    stays on it, bit for bit, for thousands of epochs -- longer than the whole analysis
    window. It converged; it did not freeze."""
    values = [0.05 + 0.5 * math.exp(-e / 30) for e in range(3000)]
    _assert_not_frozen(values, keras=True)


def test_ok_keras_epoch_loss_converged_with_shuffling_noise_is_not_frozen():
    rng = random.Random(24)
    values = [0.05 + 0.5 * math.exp(-e / 30) + rng.gauss(0, 1e-4) for e in range(3000)]
    _assert_not_frozen(values, keras=True)


@pytest.mark.parametrize("shape", ["hinge", "underflow"])
def test_ok_loss_at_exactly_zero_after_fitting_a_tiny_dataset_is_not_frozen(shape):
    rng = random.Random(25)
    if shape == "hinge":
        values = [max(0.0, 0.5 - 0.001 * i + rng.gauss(0, 0.002)) if i < 600 else 0.0
                  for i in range(5000)]
    else:                               # MSE on an exactly fittable set, decaying to 0.0
        values, level = [], 1.0
        for _ in range(8000):
            level *= 0.9
            values.append(level if level > 1e-300 else 0.0)
    _assert_not_frozen(values)


def test_ok_lr_decayed_finetune_barely_moving_is_not_frozen():
    np = pytest.importorskip("numpy")
    rng = random.Random(26)
    steps = [0.25 - 0.002 * (1 - 0.5 * (1 + math.cos(math.pi * i / 10000))) + rng.gauss(0, 0.01)
             for i in range(10000)]
    _assert_not_frozen(steps)
    epochs, level = [], 0.25
    for e in range(3000):
        lr = 1e-5 * 0.5 * (1 + math.cos(math.pi * e / 3000))
        level -= lr * 0.01
        epochs.append(float(np.float32(level + rng.gauss(0, 1e-7) * lr / 1e-5)))
    _assert_not_frozen(epochs, keras=True)


def test_ok_frozen_after_rounded_logging_is_still_frozen():
    """Rounded logging does not hide a real freeze: a loss that was moving by many grid
    steps per reading and then stops dead is frozen, past the cap too."""
    rng = random.Random(27)
    values = [round(0.3 + 2.0 * math.exp(-i / 400) + rng.gauss(0, 0.02), 4) for i in range(2500)]
    values += [values[-1]] * 40
    checks = {c for c, _ in _cli_findings(values, stride=3)}
    assert "frozen" in checks


def test_ok_loss_stuck_at_zero_from_the_start_is_frozen():
    raised = feed({"loss": [0.0] * 12})
    assert "frozen" in checks_of(raised)


def test_ok_keras_epoch_loss_frozen_past_the_cap_is_confirmed():
    """The fingerprint counts readings for Keras epoch histories too."""
    np = pytest.importorskip("numpy")
    rng = random.Random(28)
    cli = make_cli()
    problems = []
    with contextlib.redirect_stdout(io.StringIO()):
        for epoch in range(2030):
            loss = 0.3 + 2.0 * math.exp(-epoch / 400) + rng.gauss(0, 0.02) if epoch < 2010 else 0.65
            cli._record_keras_logs({"loss": np.float32(loss)}, epoch=epoch)
            if epoch >= 2000:
                problems.append(cli._check_for_trouble())
    assert len(cli.epoch_scalar_histories["loss"]) == 2000
    assert any(p and "exactly" in p for p in problems), problems


def test_bug_a_check_that_throws_is_never_mentioned():
    """DetectionEngine records a failing check in last_error, and nothing ever read it: a
    check could crash on every update and be silently gone for the whole run. Correct:
    the CLI says so once per distinct failure (and logs it)."""
    cli = make_cli()
    cli.tracked_vars = ["loss"]
    cli._detector_scalar_histories = {"loss": [1.0, 0.9, 0.8, 0.7, 0.6]}
    engine = cli._detection_engine()

    def boom(*a, **k):
        raise RuntimeError("pairs exploded")
    engine._check_pairs = boom
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli._check_for_trouble()
        cli._record_detector_scalar("loss", 0.5)
        cli._check_for_trouble()
    text = out.getvalue()
    assert "pairs exploded" in text, text
    assert text.count("pairs exploded") == 1, text


def test_ok_counter_ticking_once_per_epoch_is_not_stalled_under_the_tracer():
    """The per-tick reading for counters is per tick on which the run moved on, and the
    counter that does move is seen moving."""
    cli = make_cli()
    cli.tracked_vars = ["loss", "checkpoints_saved"]
    frames = [{"loss": float(repr(2.0 * math.exp(-i / 20) + 0.1)), "checkpoints_saved": i // 4}
              for i in range(40)]
    run_updates(cli, frames)
    assert ("counter_stalled", "checkpoints_saved") not in fired(cli._detector_histories())


def test_bug_single_variable_delete_keeps_its_opening():
    """/delete <name> drops the history; the engine's opening for it must go too, or a
    re-tracked series inherits it."""
    cli = make_cli()
    rng = random.Random(29)
    cli.tracked_vars = ["loss"]
    cli._matrix_cached_vars = set()
    cli._detector_scalar_histories = {"loss": [2.0 * math.exp(-i / 300) + 0.1 + rng.gauss(0, 0.01)
                                               for i in range(1600)]}
    cli._check_for_trouble()
    assert "loss" in cli._detection_engine()._openings
    with contextlib.redirect_stdout(io.StringIO()):
        cli._cmd_delete("loss")
    assert "loss" not in cli._detection_engine()._openings


def test_ok_objective_namespace_reward_is_not_a_loss():
    """TRL logs objective/rlhf_reward, which rises by design: a score, not a loss."""
    assert not D.looks_like_loss("objective/rlhf_reward")
    assert D.looks_like_score("objective/rlhf_reward")
    assert D.looks_like_loss("kl_loss") and D.looks_like_loss("objective")
