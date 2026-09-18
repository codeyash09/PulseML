"""Do a run's metrics actually reach the detector?

The detectors being right is not enough. In sixteen Deep4ge runs Pulse raised nothing
while training diverged, and every check was working: Keras delivers loss and val_loss
through a callback's `logs` dict rather than as locals, and the code that assembled
histories for the detector read somewhere those never reached.

So this drives the real ingestion path -- _record_keras_logs, which the injected
callback calls at every epoch end -- and then asks the real detector. TensorFlow is
not needed and not wanted here: what is under test is Pulse's own plumbing.
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from pulse import pulse_cli  # noqa: E402


def make_cli():
    cli = pulse_cli.PulseCLI(watch_locals={}, pdf_dir=os.path.join(HERE, "_wiring_out"))
    cli.sensitivity = 0.3
    cli.epoch_scalar_histories = {}
    cli.batch_scalar_histories = {}
    return cli


def feed(cli, logs_per_epoch):
    """Epochs arrive one at a time, and the detector runs after each, as in a real fit()."""
    raised = []
    for epoch, logs in enumerate(logs_per_epoch):
        cli._record_keras_logs(logs, epoch=epoch)
        trouble = cli._check_for_trouble()
        if trouble:
            raised.append((epoch, trouble))
    return raised


class KerasLogsReachTheDetector(unittest.TestCase):
    def test_nan_loss_is_detected(self):
        logs = [{"loss": 2.0 * math.exp(-0.2 * i), "accuracy": 0.5 + 0.03 * i,
                 "val_loss": 2.1 * math.exp(-0.18 * i)} for i in range(12)]
        logs += [{"loss": float("nan"), "val_loss": float("nan")} for _ in range(8)]
        raised = feed(make_cli(), logs)
        self.assertTrue(raised, "a Keras run whose loss went NaN was never detected")
        self.assertIn("nan", raised[0][1].lower())

    def test_a_healthy_keras_run_is_not_interrupted(self):
        logs = [{"loss": 2.0 * math.exp(-0.15 * i) + 0.05,
                 "accuracy": min(0.99, 0.5 + 0.02 * i),
                 "val_loss": 2.0 * math.exp(-0.13 * i) + 0.09,
                 "val_accuracy": min(0.97, 0.5 + 0.019 * i)} for i in range(30)]
        self.assertEqual(feed(make_cli(), logs), [])

    def test_a_frozen_loss_is_detected(self):
        logs = [{"loss": 0.6931, "accuracy": 0.5, "val_loss": 0.6932} for _ in range(25)]
        self.assertTrue(feed(make_cli(), logs), "a loss frozen for 25 epochs was not detected")

    def test_detection_does_not_depend_on_a_metric_being_called_accuracy(self):
        logs = [{"sparse_categorical_crossentropy": 0.6931,
                 "val_sparse_categorical_crossentropy": 0.6931,
                 "mean_io_u": 0.33} for _ in range(25)]
        self.assertTrue(feed(make_cli(), logs), "custom Keras metric names were not checked")

    def test_ragged_logs_are_tolerated(self):
        """validation_freq, a metric that starts late, a None for an unevaluated epoch."""
        logs = []
        for i in range(30):
            entry = {"loss": 2.0 * math.exp(-0.2 * i) + 0.02,
                     "accuracy": min(0.98, 0.5 + 0.02 * i)}
            if i >= 5 and i % 3 == 0:
                entry["val_loss"] = 2.0 * math.exp(-0.18 * i) + 0.06
            if i == 7:
                entry["val_loss"] = None
            logs.append(entry)
        self.assertEqual(feed(make_cli(), logs), [])

    def test_empty_logs_are_harmless(self):
        self.assertEqual(feed(make_cli(), [{} for _ in range(10)] + [None] * 5), [])


class TheRightDetectorRuns(unittest.TestCase):
    """The default path must use the engine that the benchmark measures."""

    def tearDown(self):
        os.environ.pop(pulse_cli.PulseCLI._LEGACY_DETECTOR_ENV, None)

    def test_default_path_uses_the_detection_engine(self):
        cli = make_cli()
        feed(cli, [{"loss": 0.6931, "accuracy": 0.5} for _ in range(20)])
        self.assertIsNotNone(getattr(cli, "_detector", None),
                             "_check_for_trouble did not go through the DetectionEngine")

    def test_the_old_detector_is_still_reachable(self):
        os.environ[pulse_cli.PulseCLI._LEGACY_DETECTOR_ENV] = "1"
        cli = make_cli()
        self.assertTrue(feed(cli, [{"loss": 0.6931, "accuracy": 0.5} for _ in range(20)]))
        self.assertIsNone(getattr(cli, "_detector", None))


if __name__ == "__main__":
    unittest.main()
