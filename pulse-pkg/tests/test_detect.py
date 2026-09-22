"""Tests for the detection engine: each check's firing condition, and the state machine."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pulse.pulse_detect import CRITICAL, WARNING, DetectionEngine, summarise  # noqa: E402


def run(engine, histories, rounds=3, **kwargs):
    """Feed the same histories a few times so confirmation-gated checks can fire.

    Accumulates across rounds: a finding is raised in exactly one round, so returning
    only the last round's result would hide it.
    """
    result = {"raised": [], "cleared": []}
    for _ in range(rounds):
        out = engine.update(histories, **kwargs)
        result["raised"].extend(out["raised"])
        result["cleared"].extend(out["cleared"])
    return result


def checks(findings):
    return sorted(f.check for f in findings)


class ChecksTest(unittest.TestCase):
    def setUp(self):
        self.engine = DetectionEngine(sensitivity=0.3)

    def test_nan_fires_immediately_without_waiting_for_confirmation(self):
        out = self.engine.update({"loss": [1.0, 0.9, float("nan")]})
        self.assertEqual(checks(out["raised"]), ["nonfinite"])
        self.assertEqual(out["raised"][0].severity, CRITICAL)

    def test_nan_suppresses_other_checks_on_the_same_variable(self):
        out = self.engine.update({"loss": [1.0] * 12 + [float("nan")]})
        self.assertEqual(checks(out["raised"]), ["nonfinite"])

    def test_loss_spike(self):
        history = [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.66]
        out = run(self.engine, {"loss": history + [40.0]})
        self.assertIn("loss_spike", checks(out["raised"]))

    def test_frozen_loss(self):
        out = run(self.engine, {"loss": [0.6931] * 8})
        self.assertIn("frozen", checks(self.engine.current()))

    def test_never_learned(self):
        out = run(self.engine, {"loss": [2.30, 2.31, 2.29, 2.30, 2.31, 2.30, 2.29, 2.31, 2.30, 2.30, 2.31, 2.30]})
        raised = checks(self.engine.current())
        self.assertIn("never_learned", raised)

    def test_a_healthy_run_raises_nothing(self):
        history = [1.0 / (i + 1) ** 0.5 for i in range(60)]
        out = run(self.engine, {"loss": history}, rounds=5)
        self.assertEqual(self.engine.current(), [], f"false positives: {summarise(self.engine.current())}")

    def test_gradient_norm_explosion_and_collapse(self):
        base = [1.0, 1.1, 0.95, 1.05, 1.0, 1.02]
        out = run(self.engine, {"grad_norm": base + [95.0]})
        self.assertIn("norm_explosion", checks(out["raised"]))

        other = DetectionEngine(sensitivity=0.3)
        out = run(other, {"grad_norm": base + [1e-9]})
        self.assertIn("norm_collapse", checks(other.current()))

    def test_learning_rate_jump(self):
        out = run(self.engine, {"lr": [0.001, 0.001, 0.001, 0.6]})
        self.assertIn("lr_jump", checks(self.engine.current()))

    def test_oscillation(self):
        history = [1.0 + (0.5 if i % 2 else -0.5) for i in range(24)]
        out = run(self.engine, {"loss": history})
        self.assertIn("oscillation", checks(self.engine.current()))

    def test_overfitting_needs_both_curves(self):
        train = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.48]
        val = [1.0, 0.95, 0.97, 1.05, 1.15, 1.25, 1.35, 1.45]
        out = run(self.engine, {"train_loss": train, "val_loss": val})
        self.assertIn("overfitting", checks(self.engine.current()))

    def test_suspiciously_perfect_metric_fires_beyond_two_points(self):
        # The old check compared len(history) to exactly 2, so it was dead from the
        # third reading onward. Four readings must still trip it.
        out = run(self.engine, {"accuracy": [0.9997, 0.9998, 0.9999, 1.0]})
        self.assertIn("suspiciously_perfect", checks(self.engine.current()))

    def test_keras_style_metric_names_are_checked(self):
        # These arrive as callback log keys, never as script locals: the old detector
        # iterated tracked_vars only and so never looked at them.
        out = run(self.engine, {"val_loss": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
        self.assertTrue(self.engine.current(), "no finding on a Keras-style val_loss history")

    def test_tensor_nan_from_a_probe(self):
        out = self.engine.update({}, tensor_stats={"weights": {"nan": 4, "inf": 0}})
        self.assertEqual(checks(out["raised"]), ["tensor_nonfinite"])


class StateMachineTest(unittest.TestCase):
    def test_a_gradual_finding_still_needs_two_evaluations(self):
        """The confirmation gate, on a check whose evidence is still there next time."""
        engine = DetectionEngine(sensitivity=0.3, confirmations=2)
        rising = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52]
        first = engine.update({"val_loss": rising})
        self.assertEqual(first["raised"], [], "fired on a single evaluation")
        second = engine.update({"val_loss": rising})
        self.assertTrue(checks(second["raised"]), "never confirmed on the second look")

    def test_a_spike_is_reported_the_moment_it_happens(self):
        """A transient cannot wait for confirmation, because it is gone by then.

        The gate asks for the same finding on two consecutive evaluations. That is
        right for a trend, which is still there next time, and impossible for a spike:
        the engine is only re-run when new readings arrive, and by then the spike is
        no longer the latest value, so the finding never comes back to be confirmed.
        Measured on a ladder of single-epoch spikes, nothing from 1.25x to 1000x was
        ever reported -- 0 of 8 -- until this was made immediate.
        """
        engine = DetectionEngine(sensitivity=0.3, confirmations=2)
        history = [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.66]
        raised = engine.update({"loss": history + [40.0]})["raised"]
        self.assertEqual(checks(raised), ["loss_spike"])

    def test_ordinary_noise_is_not_a_spike(self):
        """The reason the gate was put there in the first place, still held."""
        engine = DetectionEngine(sensitivity=0.3, confirmations=2)
        history = [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.66]
        for wobble in (0.72, 0.80, 0.95, 1.1):
            engine = DetectionEngine(sensitivity=0.3, confirmations=2)
            raised = engine.update({"loss": history + [wobble]})["raised"]
            self.assertEqual(raised, [], f"fired on a single reading of {wobble}")

    def test_a_finding_is_raised_once_not_every_round(self):
        engine = DetectionEngine(sensitivity=0.3)
        history = [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.66, 40.0]
        run(engine, {"loss": history}, rounds=2)
        again = engine.update({"loss": history})
        self.assertEqual(again["raised"], [])
        self.assertEqual(len(engine.current()), 1)
        self.assertGreater(engine.current()[0].count, 1)

    def test_numbers_changing_do_not_make_it_a_new_problem(self):
        # The old detector keyed on the formatted message, so "spiked to 4.12" and
        # "spiked to 4.13" were two unrelated problems and both escalated.
        engine = DetectionEngine(sensitivity=0.3)
        base = [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.66]
        run(engine, {"loss": base + [40.0]}, rounds=2)
        for spike in (40.1, 41.7, 39.2):
            out = engine.update({"loss": base + [spike]})
            self.assertEqual(out["raised"], [], f"re-raised for a different number ({spike})")
        self.assertEqual(len(engine.current()), 1)

    def test_a_cleared_problem_can_fire_again_later(self):
        engine = DetectionEngine(sensitivity=0.3, confirmations=2, rearm_after=2)
        base = [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.66]
        run(engine, {"loss": base + [40.0]}, rounds=2)
        self.assertEqual(len(engine.current()), 1)

        healthy = [1.0 / (i + 1) ** 0.5 for i in range(40)]
        cleared = []
        for _ in range(3):
            cleared.extend(engine.update({"loss": healthy})["cleared"])
        self.assertTrue(any(f.check == "loss_spike" for f in cleared))
        self.assertEqual(engine.current(), [])

        again = run(engine, {"loss": base + [40.0]}, rounds=2)
        self.assertIn("loss_spike", checks(again["raised"]), "the same problem could not fire a second time")

    def test_findings_are_ordered_worst_first(self):
        engine = DetectionEngine(sensitivity=0.3)
        run(engine, {"loss": [0.6931] * 10, "accuracy": [0.5] * 12}, rounds=3)
        engine.update({"loss": [float("nan")]})
        severities = [f.severity for f in engine.current()]
        self.assertEqual(severities[0], CRITICAL)

    def test_findings_serialise(self):
        engine = DetectionEngine(sensitivity=0.3)
        out = engine.update({"loss": [1.0, 0.9, float("inf")]})
        payload = out["raised"][0].to_dict()
        self.assertEqual(payload["check"], "nonfinite")
        self.assertEqual(payload["variable"], "loss")
        self.assertIn("confidence", payload)


class SensitivityTest(unittest.TestCase):
    def test_dial_changes_what_fires(self):
        history = [1.0, 0.99, 1.01, 1.0, 0.995, 1.005, 1.0, 0.998, 1.002, 1.0, 1.0, 0.999]
        quiet = DetectionEngine(sensitivity=0.0)
        loud = DetectionEngine(sensitivity=1.0)
        run(quiet, {"loss": history}, rounds=3)
        run(loud, {"loss": history}, rounds=3)
        self.assertLessEqual(len(quiet.current()), len(loud.current()))

    def test_learning_rate_threshold_is_on_the_dial(self):
        # It used to be hardcoded at 10x, so the dial could not reach it.
        from pulse.pulse_detect import thresholds
        self.assertNotEqual(thresholds(0.0)["lr_jump_ratio"], thresholds(1.0)["lr_jump_ratio"])
        self.assertNotEqual(thresholds(0.0)["never_learned_ratio"], thresholds(1.0)["never_learned_ratio"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
