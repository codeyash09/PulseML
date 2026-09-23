"""Does detection reach the detector on a real Keras run?

This is the failure the curve benchmark cannot see. In sixteen Deep4ge runs Pulse
raised nothing at all, and the detectors were fine: the metrics never arrived. Keras
does not expose loss or val_loss as locals, they come through a callback's `logs`
dict, and the code that read histories looked somewhere else.

So this drives the real ingestion path -- PulseCLI._record_keras_logs, the function
the injected callback calls on every epoch end -- with the log dicts Keras actually
emits, and then asks the real detector whether it noticed. No TensorFlow required:
what is under test is Pulse's plumbing, not Keras'.

    python3 wiring.py
"""
import math
import os
import sys

# Point PULSE_SRC at a Pulse checkout, or rely on an installed pulse package.
sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legacy import make_cli  # noqa: E402

PASS, FAIL = [], []


def run_epochs(cli, logs_per_epoch):
    """Feed epochs the way the injected Keras callback does, and collect verdicts."""
    seen = []
    for epoch, logs in enumerate(logs_per_epoch):
        cli._record_keras_logs(logs, epoch=epoch)
        trouble = cli._check_for_trouble()
        if trouble:
            seen.append((epoch, trouble))
    return seen


def case(label, logs_per_epoch, should_fire):
    cli = make_cli(0.3)
    try:
        seen = run_epochs(cli, logs_per_epoch)
    except Exception as exc:  # noqa: BLE001
        FAIL.append("%s -> crashed: %s: %s" % (label, type(exc).__name__, exc))
        return
    fired = bool(seen)
    if fired != should_fire:
        FAIL.append("%s -> %s (%s)" % (label, "fired" if fired else "silent",
                                       seen[0][1][:90] if seen else "nothing raised"))
    else:
        PASS.append("%s%s" % (label, (" @epoch %d: %s" % (seen[0][0], seen[0][1][:60])) if seen else ""))


# A Keras run whose loss goes NaN at epoch 12 -- the commonest crash in JunoBench.
nan_run = [{"loss": 2.0 * math.exp(-0.2 * i), "accuracy": 0.5 + 0.03 * i,
            "val_loss": 2.1 * math.exp(-0.18 * i), "val_accuracy": 0.5 + 0.028 * i}
           for i in range(12)]
nan_run += [{"loss": float("nan"), "accuracy": float("nan"),
             "val_loss": float("nan"), "val_accuracy": float("nan")} for _ in range(8)]
case("keras logs, loss goes NaN", nan_run, True)

# A healthy Keras run: this must stay silent, or every run gets interrupted.
healthy = [{"loss": 2.0 * math.exp(-0.15 * i) + 0.05, "accuracy": min(0.99, 0.5 + 0.02 * i),
            "val_loss": 2.0 * math.exp(-0.13 * i) + 0.09, "val_accuracy": min(0.97, 0.5 + 0.019 * i)}
           for i in range(30)]
case("keras logs, healthy run", healthy, False)

# Frozen: Dropout(1.0) or a zeroed learning rate. Loss identical every epoch.
case("keras logs, loss frozen",
     [{"loss": 0.6931, "accuracy": 0.5, "val_loss": 0.6932, "val_accuracy": 0.5} for _ in range(25)],
     True)

# Overfitting: train keeps falling, validation turns around at epoch 10.
case("keras logs, validation turns around",
     [{"loss": 1.5 * math.exp(-0.25 * i) + 0.01,
       # Never exactly constant: a flat float repeated is caught by the frozen check,
       # which would pass this test for a reason that has nothing to do with overfitting.
       "val_loss": (0.55 if i < 10 else 0.55 + 0.06 * (i - 10)) + 0.004 * math.sin(i * 1.7),
       "accuracy": min(0.999, 0.6 + 0.03 * i)} for i in range(30)],
     True)

# Keras names its metrics after the loss function, and custom metrics keep their own
# names. Detection must not depend on a metric being called "accuracy".
case("keras logs, custom metric names",
     [{"sparse_categorical_crossentropy": 0.6931, "val_sparse_categorical_crossentropy": 0.6931,
       "mean_io_u": 0.33} for _ in range(25)],
     True)

# Metrics that appear only from epoch 5 (validation_freq), and a None for an epoch
# Keras did not evaluate: a real log stream is ragged.
ragged = []
for i in range(30):
    logs = {"loss": 2.0 * math.exp(-0.2 * i) + 0.02, "accuracy": min(0.98, 0.5 + 0.02 * i)}
    if i >= 5 and i % 3 == 0:
        logs["val_loss"] = 2.0 * math.exp(-0.18 * i) + 0.06
    if i == 7:
        logs["val_loss"] = None
    ragged.append(logs)
case("keras logs, ragged validation", ragged, False)

# Empty and None logs happen (a callback firing before the first metric exists).
case("keras logs, empty dicts", [{} for _ in range(10)] + [None] * 5, False)


def engine_is_the_one_running():
    """The default path must be the benchmarked engine, not the old detector."""
    cli = make_cli(0.3)
    for i in range(20):
        cli._record_keras_logs({"loss": 0.6931, "accuracy": 0.5}, epoch=i)
    cli._check_for_trouble()
    assert getattr(cli, "_detector", None) is not None, \
        "_check_for_trouble did not go through the DetectionEngine"
    os.environ["PULSE_LEGACY_DETECTOR"] = "1"
    try:
        other = make_cli(0.3)
        for i in range(20):
            other._record_keras_logs({"loss": 0.6931, "accuracy": 0.5}, epoch=i)
        assert other._check_for_trouble(), "the legacy escape hatch stopped working"
        assert getattr(other, "_detector", None) is None
    finally:
        os.environ.pop("PULSE_LEGACY_DETECTOR", None)


try:
    engine_is_the_one_running()
    PASS.append("the default path runs the engine; PULSE_LEGACY_DETECTOR=1 restores the old one")
except AssertionError as exc:
    FAIL.append("detector selection -> %s" % exc)

print("\n".join("PASS  " + p for p in PASS))
if FAIL:
    print("\n".join("FAIL  " + f for f in FAIL))
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
