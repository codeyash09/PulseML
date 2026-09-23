"""Can the detector be crashed, hung, or confused by the data a real run hands it?

A detector that raises an exception takes the training run down with it, which is
strictly worse than not detecting anything. Every case here is something a real
callback has produced: a metric that is None until the first validation pass, a
history of one point, a loss that is legitimately negative, names with unicode in
them, 50k step-level readings, a tensor summary with missing keys.

Nothing here asserts a particular finding: the contract is that the engine keeps
running, keeps its promises about types, and stays fast.

    python3 edge_cases.py
"""
import math
import os
import sys
import time

# Point PULSE_SRC at a Pulse checkout, or rely on an installed pulse package.
sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))

from pulse import pulse_detect as detect  # noqa: E402

PASS, FAIL = [], []


def check(label, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - that is the point of the test
        FAIL.append("%s -> %s: %s" % (label, type(exc).__name__, exc))
    else:
        PASS.append(label)


def engine():
    return detect.DetectionEngine(sensitivity=0.3, confirmations=2)


def feed(histories, **kw):
    """One update must always return the two lists, whatever it was given."""
    out = engine().update(histories, **kw)
    assert isinstance(out, dict) and "raised" in out and "cleared" in out, out
    assert isinstance(out["raised"], list) and isinstance(out["cleared"], list)
    for finding in out["raised"]:
        assert finding.severity in (detect.CRITICAL, detect.WARNING, detect.INFO)
        assert isinstance(finding.message, str) and finding.message
        assert 0.0 <= finding.confidence <= 1.0, finding.confidence
    return out


# ------------------------------------------------------------------ empty and tiny
check("no histories at all", lambda: feed({}))
check("None instead of histories", lambda: feed(None))
check("a variable with an empty history", lambda: feed({"loss": []}))
check("a single reading", lambda: feed({"loss": [1.0]}))
check("two readings", lambda: feed({"loss": [1.0, 0.9]}))
check("every value identical", lambda: feed({"loss": [0.5] * 100}))
check("all zeros", lambda: feed({"loss": [0.0] * 100}))

# ------------------------------------------------------------------ wrong types
check("None values in the history", lambda: feed({"loss": [1.0, None, 0.8, None, 0.6] * 8}))
check("all values None", lambda: feed({"loss": [None] * 40}))
check("strings in the history", lambda: feed({"loss": ["0.5", "oops", 0.4] * 10}))
check("booleans in the history", lambda: feed({"acc": [True, False, True] * 10}))
check("a nested list", lambda: feed({"loss": [[1.0], [0.9], 0.8] * 10}))
check("a dict as a history", lambda: feed({"loss": {"a": 1}}))
check("a numpy-ish object", lambda: feed({"loss": [type("F", (), {"__float__": lambda s: 1.0})()] * 12}))

# ------------------------------------------------------------------ hostile numbers
check("negative loss throughout", lambda: feed({"loss": [-1.0 - 0.01 * i for i in range(60)]}))
check("loss crossing zero", lambda: feed({"loss": [1.0 - 0.05 * i for i in range(60)]}))
check("denormal magnitudes", lambda: feed({"loss": [1e-300 * (0.9 ** i) for i in range(40)]}))
check("enormous magnitudes", lambda: feed({"loss": [1e300 * (0.9 ** i) for i in range(40)]}))
check("nan from the first reading", lambda: feed({"loss": [float("nan")] * 30}))
check("inf and -inf mixed", lambda: feed({"loss": [float("inf"), float("-inf")] * 20}))
check("nan then recovery", lambda: feed({"loss": [1.0, float("nan"), 0.9, 0.8] * 10}))

# ------------------------------------------------------------------ shapes and names
check("histories of different lengths",
      lambda: feed({"loss": [1.0] * 60, "val_loss": [0.9] * 3, "acc": [0.5] * 31}))
check("empty variable name", lambda: feed({"": [1.0] * 20}))
check("unicode names", lambda: feed({"perte_d'entraînement": [1.0] * 20, "précision": [0.5] * 20}))
check("a name that is every hint at once", lambda: feed({"val_loss_accuracy_lr_grad_norm": [1.0] * 20}))
check("two hundred variables",
      lambda: feed({("m%d" % i): [1.0 - 0.001 * j for j in range(30)] for i in range(200)}))

# ------------------------------------------------------------------ steps
check("no step given", lambda: feed({"loss": [1.0] * 20}))
check("negative step", lambda: feed({"loss": [1.0] * 20}, step=-5))
check("huge step", lambda: feed({"loss": [1.0] * 20}, step=2 ** 62))
check("float step", lambda: feed({"loss": [1.0] * 20}, step=3.7))

# ------------------------------------------------------------------ tensor stats
check("tensor stats missing keys", lambda: feed({"loss": [1.0] * 20}, tensor_stats={"w": {}}))
check("tensor stats wrong types",
      lambda: feed({"loss": [1.0] * 20}, tensor_stats={"w": {"nan": "lots", "inf": None}}))
check("tensor stats not a dict", lambda: feed({"loss": [1.0] * 20}, tensor_stats={"w": 5}))
check("tensor stats None", lambda: feed({"loss": [1.0] * 20}, tensor_stats=None))


# ------------------------------------------------------------------ the state machine
def state_machine():
    eng = engine()
    broken = [0.693] * 40                      # frozen: should raise, once confirmed
    raised = []
    for i in range(12, 41):
        raised += eng.update({"loss": broken[:i]}, step=i)["raised"]
    assert raised, "a frozen loss must raise something"
    keys = {f.key for f in raised}
    assert len(raised) == len(keys), "the same finding must not be raised twice while it is active"
    assert eng.current(), "an active finding must show up in current()"
    # it gets fixed: the loss starts moving again
    fixed = broken + [0.693 * (0.9 ** i) for i in range(1, 40)]
    cleared = []
    for i in range(41, len(fixed) + 1):
        cleared += eng.update({"loss": fixed[:i]}, step=i)["cleared"]
    assert cleared, "a finding must clear once the run recovers"


check("raise once, clear on recovery", state_machine)


def immediate_vs_confirmed():
    eng = engine()
    out = eng.update({"loss": [1.0, 0.9, float("nan")]}, step=3)
    assert any(f.check == "nonfinite" for f in out["raised"]), "nan must not wait for a second opinion"
    slow = detect.DetectionEngine(sensitivity=0.3, confirmations=3)
    first = slow.update({"loss": [0.693] * 30}, step=30)["raised"]
    assert not first, "an ordinary finding must wait for its confirmations"


check("nan raises at once, the rest wait", immediate_vs_confirmed)


def idempotent():
    eng = engine()
    hist = {"loss": [0.693] * 40}
    for _ in range(10):
        eng.update(hist, step=40)
    active = eng.current()
    assert len({f.key for f in active}) == len(active), "duplicate findings for the same key"


check("the same data repeated does not pile up findings", idempotent)


def sensitivity_is_monotone():
    """Turning the dial up must never make the engine notice less."""
    curve = {"loss": [1.0 - 0.0005 * i for i in range(60)]}
    seen = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        eng = detect.DetectionEngine(sensitivity=s, confirmations=2)
        fired = set()
        for i in range(5, 61):
            for f in eng.update({"loss": curve["loss"][:i]}, step=i)["raised"]:
                fired.add(f.check)
        seen.append(fired)
    for lower, higher in zip(seen, seen[1:]):
        assert lower <= higher or len(higher) >= len(lower), (lower, higher)


check("higher sensitivity never detects less", sensitivity_is_monotone)


def thresholds_are_sane():
    for s in (0.0, 0.5, 1.0, -5.0, 99.0):
        t = detect.thresholds(s)
        assert t["explosion_multiplier"] > 1.0, t
        assert t["stagnation_frac"] > 0, t
        assert -1.0 <= t["oscillation_correlation"] <= 0.0, t
        assert t["oscillation_flip_threshold"] >= 1, t


check("thresholds stay sane, even off the end of the dial", thresholds_are_sane)


# ------------------------------------------------------------------ cost
def fast_on_long_runs():
    eng = engine()
    values = [2.0 * math.exp(-3.0 * i / 50000) for i in range(50000)]
    start = time.time()
    for i in range(49900, 50001, 10):
        eng.update({"loss": values[:i], "acc": values[:i]}, step=i)
    elapsed = time.time() - start
    assert elapsed < 5.0, "50k-reading history took %.1fs for 11 updates" % elapsed


check("a 50k-step history stays fast", fast_on_long_runs)


def no_mutation():
    """The engine must not edit the caller's training history."""
    original = {"loss": [1.0, float("nan"), 0.8] + [0.5] * 30, "acc": [0.1] * 33}
    copy = {k: list(v) for k, v in original.items()}
    engine().update(original, step=33)
    assert original == copy, "the engine modified the history it was given"


check("the caller's history is never modified", no_mutation)

# ------------------------------------------------------------------ the types a run really reports
# `isinstance(v, float)` is true for numpy's float64 (it subclasses float) and false
# for float32, which is the default dtype in Keras and everywhere in mixed precision.
# That one asymmetry made the detector silently discard every reading of a float32 run.


def numeric_types_are_understood():
    import numpy as np

    from pulse.pulse_detect import _finite
    cases = {
        "np.float64": np.float64(0.5), "np.float32": np.float32(0.5),
        "np.float16": np.float16(0.5), "np.int32": np.int32(1), "np.int64": np.int64(1),
        "0-dim array": np.array(0.5), "python int": 1, "python float": 0.5,
    }
    for label, value in cases.items():
        assert _finite([value]), "%s was discarded" % label
    import decimal
    import fractions
    assert _finite([decimal.Decimal("0.5")]), "Decimal was discarded"
    assert _finite([fractions.Fraction(1, 2)]), "Fraction was discarded"

    class TorchLike:                      # a 0-dim tensor converts through __float__
        def __float__(self):
            return 0.5

    assert _finite([TorchLike()]), "a tensor-like scalar was discarded"
    # and the things that must stay out
    assert not _finite(["0.5"]), "a string was accepted as a reading"
    assert not _finite([b"0.5"]), "bytes were accepted as a reading"
    assert not _finite([True, False]), "a bool was accepted as a reading"
    assert not _finite([None]), "None was accepted as a reading"
    assert not _finite([complex(1, 2)]), "a complex number was accepted as a reading"

    class Exploding:
        def __float__(self):
            raise RuntimeError("no")

    assert not _finite([Exploding()]), "an object whose __float__ raises was accepted"


check("every numeric type a training loop reports", numeric_types_are_understood)


def float32_run_is_checked():
    """The whole point: a run reported in float32 must be detected like any other."""
    import numpy as np
    eng = engine()
    frozen = [np.float32(0.6931)] * 30
    fired = set()
    for i in range(10, 31):
        for f in eng.update({"loss": frozen[:i]}, step=i)["raised"]:
            fired.add(f.check)
    assert fired, "a frozen float32 run raised nothing at all"
    nan_engine = engine()
    history = [np.float32(1.0), np.float32(0.9), np.float32("nan")]
    raised = nan_engine.update({"loss": history}, step=3)["raised"]
    assert any(f.check == "nonfinite" for f in raised), "a float32 NaN was not detected"


check("a float32 run is detected like any other", float32_run_is_checked)


# ------------------------------------------------------------------ how histories really arrive
def sliding_window_history():
    """The brain caps history, so the engine sees a window that slides, not a run
    that starts at step 1. Nothing may assume values[0] is the first epoch."""
    eng = engine()
    full = [2.0 * math.exp(-0.05 * i) for i in range(400)]
    for end in range(50, 401, 10):
        window = full[max(0, end - 200):end]          # the last 200 readings only
        eng.update({"loss": window}, step=end)
    assert True


check("a history window that slides", sliding_window_history)


def history_shrinks():
    eng = engine()
    eng.update({"loss": [0.693] * 40}, step=40)
    eng.update({"loss": [0.693] * 5}, step=41)        # a cap kicked in, or a restart
    eng.update({"loss": [0.693] * 2}, step=42)
    eng.update({"loss": []}, step=43)


check("a history that gets shorter", history_shrinks)


def variable_disappears():
    """A metric stops being reported -- validation every 5 epochs, or a crash."""
    eng = engine()
    for i in range(12, 40):
        eng.update({"loss": [0.693] * i, "val_loss": [0.7] * i}, step=i)
    before = {f.key for f in eng.current()}
    for i in range(40, 50):
        eng.update({"loss": [0.693] * i}, step=i)     # val_loss is gone
    assert before, "nothing was active to begin with"
    # The contract is simply that it does not crash and does not invent findings
    # about a variable it can no longer see.
    for finding in eng.current():
        assert finding.variable in ("loss", "val_loss")


check("a variable that stops being reported", variable_disappears)


def variable_appears_late():
    eng = engine()
    for i in range(5, 30):
        eng.update({"loss": [2.0 * math.exp(-0.1 * j) for j in range(i)]}, step=i)
    for i in range(30, 45):
        eng.update({"loss": [2.0 * math.exp(-0.1 * j) for j in range(i)],
                    "val_loss": [0.5] * (i - 29)}, step=i)


check("a variable that appears mid-run", variable_appears_late)


def distributed_ranks():
    """Four ranks reporting the same metric under their own names."""
    eng = engine()
    histories = {("loss_rank%d" % rank): [0.693] * 30 for rank in range(4)}
    raised = eng.update(histories, step=30)["raised"]
    eng.update(histories, step=31)
    assert all(isinstance(f.variable, str) for f in raised)


check("the same metric from four ranks", distributed_ranks)


def name_case_variants():
    eng = engine()
    eng.update({"Loss": [0.693] * 30, "loss": [0.693] * 30, "LOSS": [0.693] * 30}, step=30)
    eng.update({"Loss": [0.693] * 31, "loss": [0.693] * 31, "LOSS": [0.693] * 31}, step=31)
    keys = {f.key for f in eng.current()}
    assert len(keys) == len(eng.current()), "case variants collided into one key"


check("Loss, loss and LOSS at the same time", name_case_variants)


def steps_out_of_order():
    eng = engine()
    for step in (10, 9, 10, 100, 1, 0, -3, 10):
        eng.update({"loss": [0.693] * 20}, step=step)


check("steps that repeat, go backwards or reset", steps_out_of_order)


check("a 4000-character variable name", lambda: feed({"x" * 4000: [1.0] * 20}))


# ------------------------------------------------------------------ the state machine, harder
def problem_returns_after_being_fixed():
    """A fix lands, the finding clears, and then the run goes bad the same way again.
    The old detector keyed on its own message and could never re-report this."""
    eng = engine()
    for i in range(12, 40):
        eng.update({"loss": [0.693] * i}, step=i)
    assert eng.current(), "the frozen run was never raised"
    recovering = [0.693] * 40 + [0.693 * (0.85 ** i) for i in range(1, 30)]
    for i in range(41, len(recovering) + 1):
        eng.update({"loss": recovering[:i]}, step=i)
    assert not [f for f in eng.current() if f.check == "frozen"], "it never cleared"
    broken_again = recovering + [recovering[-1]] * 30
    raised_again = []
    for i in range(len(recovering) + 1, len(broken_again) + 1):
        raised_again += eng.update({"loss": broken_again[:i]}, step=i)["raised"]
    assert any(f.check == "frozen" for f in raised_again), \
        "the same problem returning was never raised a second time"


check("a problem that returns after a fix is raised again", problem_returns_after_being_fixed)


def confirmations_edge_values():
    for confirmations in (0, -5, 1, 1000):
        eng = detect.DetectionEngine(sensitivity=0.3, confirmations=confirmations)
        for i in range(12, 40):
            eng.update({"loss": [0.693] * i}, step=i)
    # confirmations=1000 must simply never raise, not crash or raise early
    slow = detect.DetectionEngine(sensitivity=0.3, confirmations=1000)
    raised = []
    for i in range(12, 60):
        raised += slow.update({"loss": [0.693] * i}, step=i)["raised"]
    assert not raised, "a 1000-confirmation engine raised something"


check("confirmations of 0, -5, 1 and 1000", confirmations_edge_values)


def baselines_can_be_nonsense():
    for value in (0.0, -1.0, float("nan"), float("inf"), 1e300):
        eng = engine()
        eng.set_baseline("loss", value)
        for i in range(5, 30):
            eng.update({"loss": [2.0 * math.exp(-0.1 * j) for j in range(i)]}, step=i)


check("an agent-supplied baseline of 0, -1, NaN or inf", baselines_can_be_nonsense)


def findings_survive_json():
    import json
    eng = engine()
    for i in range(12, 40):
        eng.update({"loss": [0.693] * i}, step=i)
    for finding in eng.current():
        text = json.dumps(finding.to_dict())
        back = json.loads(text)
        assert back["check"] and back["variable"] is not None
        assert isinstance(back["values"], dict)


check("every finding survives a JSON round trip", findings_survive_json)


def state_does_not_grow_without_bound():
    eng = engine()
    history = {"loss": [0.693] * 40}
    for _ in range(1000):
        eng.update(history, step=40)
    assert len(eng.active) < 50, "active findings grew to %d" % len(eng.active)
    assert len(eng._streak) < 100, "streak bookkeeping grew to %d" % len(eng._streak)


check("a thousand identical updates do not grow the engine", state_does_not_grow_without_bound)


def many_variables_long_history():
    eng = engine()
    histories = {("m%d" % v): [1.0 - 0.0001 * i for i in range(2000)] for v in range(50)}
    start = time.time()
    eng.update(histories, step=2000)
    elapsed = time.time() - start
    assert elapsed < 5.0, "50 variables x 2000 readings took %.1fs" % elapsed


check("50 variables with 2000 readings each", many_variables_long_history)


# ------------------------------------------------------------------ valid but unusual values
check("a loss that is negative and frozen",
      lambda: feed({"elbo_loss": [-3.2] * 30}))
check("a loss that is an integer every step", lambda: feed({"loss": [3] * 30}))
check("a loss of exactly zero throughout", lambda: feed({"loss": [0.0] * 30}))
check("a metric above 1.0 (a sum, not a rate)", lambda: feed({"score": [float(i) for i in range(30)]}))
check("a learning rate of zero throughout", lambda: feed({"loss": [0.693] * 30, "lr": [0.0] * 30}))
check("a learning rate that is negative", lambda: feed({"loss": [0.693] * 30, "lr": [-0.01] * 30}))

print("\n".join("PASS  " + p for p in PASS))
if FAIL:
    print("\n".join("FAIL  " + f for f in FAIL))
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
