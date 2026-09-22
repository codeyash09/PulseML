"""Detection beyond the loss curve: what a variable means, and how variables relate.

Every check used to key off two questions -- does this look like a loss, does it look
like a score -- which covers the run's own arithmetic and nothing else. A benchmark of
95 failure families found 35 of them invisible for that reason: a GPU at 2% utilisation,
a router collapsed onto one expert, a rank training its own copy of the model, a
checkpoint counter that stopped moving. None of those are in the loss, and all of them
are logged by the runs they happen to.

These tests cover the two layers added for that: a *role* per variable, which says which
direction is bad for that kind of number, and *relations* between variables, because most
of what goes wrong in a real run is a relationship coming apart while every individual
curve stays unremarkable.

Each test also has to stay quiet on the healthy shape it is closest to, which is the
harder half: a detector that fires on good runs gets switched off.
"""
import math
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from pulse.pulse_detect import CRITICAL, DetectionEngine, WARNING  # noqa: E402

EPOCHS = 40


def falling(n=EPOCHS, start=2.0, floor=0.05):
    return [floor + (start - floor) * math.exp(-3.0 * i / n) for i in range(n)]


def flat(value, n=EPOCHS):
    return [float(value)] * n


def ramp(start, end, n=EPOCHS):
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def raised(histories, sensitivity=0.3, confirmations=2, tensor_stats=None, steps=None):
    """Replay a run into a fresh engine and collect every check that fired."""
    engine = DetectionEngine(sensitivity=sensitivity, confirmations=confirmations)
    length = steps or max(len(v) for v in histories.values())
    found = []
    for step in range(1, length + 1):
        window = {k: v[:step] for k, v in histories.items()}
        stats = None
        if tensor_stats is not None:
            stats = (tensor_stats[min(step - 1, len(tensor_stats) - 1)]
                     if isinstance(tensor_stats, list) else tensor_stats)
        result = engine.update(window, step=step, tensor_stats=stats)
        found.extend(f.check for f in result["raised"])
    return found


class DirectionTest(unittest.TestCase):
    """Worse means up for a loss and down for a score, and the checks only knew one."""

    def test_a_collapsing_validation_accuracy_is_reported(self):
        # val_regression and val_drift both test `b > a`. That is right for val_loss
        # and backwards for val_accuracy: this exact run raised nothing at all.
        found = raised({"loss": falling(), "val_accuracy": ramp(0.94, 0.34)})
        self.assertIn("score_regression", found,
                      f"a validation accuracy falling 0.94 -> 0.34 raised {found}")

    def test_the_equivalent_val_loss_regression_still_fires(self):
        found = raised({"loss": falling(), "val_loss": ramp(0.30, 0.90)})
        self.assertTrue({"val_regression", "val_drift", "divergence"} & set(found), found)

    def test_a_rising_reward_is_not_a_regression(self):
        # RL logs a score that goes up. Reading it as a loss makes success a fault.
        found = raised({"reward": ramp(-100.0, 80.0)})
        self.assertNotIn("score_regression", found, found)
        self.assertNotIn("divergence", found, found)

    def test_an_accuracy_that_merely_plateaus_is_not_a_regression(self):
        found = raised({"loss": falling(),
                        "accuracy": [min(0.93, 0.5 + 0.5 * i / 15) for i in range(EPOCHS)]})
        self.assertNotIn("score_regression", found, found)


class TransientTest(unittest.TestCase):
    def test_a_single_epoch_spike_is_reported(self):
        loss = falling()
        loss[25] = loss[25] * 50
        self.assertIn("loss_spike", raised({"loss": loss}))

    def test_a_smooth_curve_has_no_spike(self):
        self.assertNotIn("loss_spike", raised({"loss": falling()}))


class RoleTest(unittest.TestCase):
    """One check per kind of number, rather than per variable name."""

    def test_entropy_collapsing_onto_one_choice(self):
        found = raised({"loss": falling(),
                        "router_entropy": [2.0 * math.exp(-4.0 * i / EPOCHS) for i in range(EPOCHS)]})
        self.assertIn("entropy_collapse", found, found)

    def test_padding_that_eats_most_of_every_batch(self):
        found = raised({"loss": falling(), "padding_fraction": ramp(0.05, 0.85)})
        self.assertIn("waste_rising", found, found)

    def test_a_steady_high_waste_fraction_counts_too(self):
        # Never rose, so "rising" never fired, and 80% of the compute still went nowhere.
        found = raised({"loss": falling(), "truncated_frac": flat(0.8)})
        self.assertTrue({"waste_high", "waste_rising"} & set(found), found)

    def test_a_small_waste_fraction_is_left_alone(self):
        found = raised({"loss": falling(), "oov_rate": flat(0.01)})
        self.assertEqual([c for c in found if c.startswith("waste")], [], found)

    def test_an_idle_accelerator(self):
        found = raised({"loss": falling(), "gpu_util": ramp(0.95, 0.05)})
        self.assertIn("utilisation_drop", found, found)

    def test_throughput_falling_away(self):
        found = raised({"loss": falling(), "tokens_per_sec": ramp(12000.0, 3000.0)})
        self.assertIn("throughput_drop", found, found)

    def test_throughput_reaching_zero_is_critical(self):
        engine = DetectionEngine()
        # Reaches zero and stays there: a ramp whose last point is zero is a run
        # slowing down, not one that has stopped.
        histories = {"loss": falling(),
                     "tokens_per_sec": ramp(12000.0, 0.0, n=EPOCHS // 2) + flat(0.0, EPOCHS // 2)}
        severities = []
        for step in range(1, EPOCHS + 1):
            result = engine.update({k: v[:step] for k, v in histories.items()}, step=step)
            severities += [f.severity for f in result["raised"] if f.check == "throughput_stopped"]
        self.assertIn(CRITICAL, severities)

    def test_memory_climbing_towards_an_oom(self):
        found = raised({"loss": falling(),
                        "gpu_mem_mb": [4000 * 8.0 ** (i / (EPOCHS - 1)) for i in range(EPOCHS)]})
        self.assertIn("memory_growth", found, found)

    def test_steady_memory_is_not_a_leak(self):
        found = raised({"loss": falling(), "gpu_mem_mb": flat(4000.0)})
        self.assertNotIn("memory_growth", found, found)

    def test_a_counter_that_stopped_counting(self):
        # The loss keeps moving, so the run is alive; nothing is reaching disk.
        found = raised({"loss": falling(), "checkpoints_written": flat(0.0)})
        self.assertIn("counter_stalled", found, found)

    def test_a_counter_on_a_finished_run_is_not_news(self):
        # Everything has stopped, including the counter. That is the run being over.
        found = raised({"loss": flat(0.05), "checkpoints_written": flat(3.0)})
        self.assertNotIn("counter_stalled", found, found)

    def test_an_optimizer_state_that_was_never_anything_but_zero(self):
        found = raised({"loss": falling(floor=0.9), "momentum_norm": flat(0.0)})
        self.assertIn("norm_collapse", found, found)

    def test_a_learned_temperature_collapsing(self):
        found = raised({"loss": falling(),
                        "temperature": [0.07 * math.exp(-5.0 * i / EPOCHS) for i in range(EPOCHS)]})
        self.assertIn("temperature_collapse", found, found)

    def test_a_thermometer_is_not_a_softmax_temperature(self):
        found = raised({"loss": falling(), "gpu_temp_c": ramp(55.0, 92.0)})
        self.assertIn("thermal_risk", found, found)
        self.assertNotIn("temperature_collapse", found, found)


class GridAndCycleTest(unittest.TestCase):
    def test_a_loss_rounded_onto_a_grid(self):
        # With noise, as a real loss has: a perfectly smooth curve lands on a grid
        # trivially because its consecutive differences are all equal anyway, and
        # reporting that would flag every synthetic straight line.
        import random
        random.seed(1)
        grid = 0.001
        noisy = [v + random.gauss(0, 0.01) for v in falling(n=60)]
        found = raised({"loss": [round(v / grid) * grid for v in noisy]})
        self.assertIn("quantised", found, found)

    def test_the_same_curve_unrounded_is_not_flagged(self):
        import random
        random.seed(1)
        found = raised({"loss": [v + random.gauss(0, 0.01) for v in falling(n=60)]})
        self.assertNotIn("quantised", found, found)

    def test_an_accuracy_on_a_grid_is_just_arithmetic(self):
        # k correct out of 200 lands on multiples of 0.005 and always will.
        found = raised({"loss": falling(n=60),
                        "accuracy": [round((0.5 + 0.45 * i / 59) * 200) / 200 for i in range(60)]})
        self.assertNotIn("quantised", found, found)

    def test_data_that_is_never_shuffled(self):
        base = falling(n=60)
        cycled = [v + 0.35 * math.sin(2 * math.pi * i / 8.0) for i, v in enumerate(base)]
        self.assertIn("periodic", raised({"loss": cycled}))

    def test_an_ordinary_noisy_curve_has_no_cycle(self):
        import random
        random.seed(7)
        noisy = [v + random.gauss(0, 0.12) for v in falling(n=60)]
        self.assertNotIn("periodic", raised({"loss": noisy}))


class ProgressTest(unittest.TestCase):
    def test_a_run_that_gave_back_what_it_had_gained(self):
        # The run reached 0.1 and now sits at 0.8 and stays there: no single step is a
        # big enough jump to be a spike, and it is plainly worse than it already was.
        loss = falling() [:25] + [0.8 + 0.01 * (i % 3) for i in range(EPOCHS - 25)]
        self.assertIn("progress_lost", raised({"loss": loss}))

    def test_a_curriculum_step_is_not_lost_progress(self):
        # The data gets harder on purpose: the loss jumps and then descends again from
        # the higher point. Same shape as a regression, and not one.
        loss = falling(n=30, start=1.5, floor=0.3) + falling(n=30, start=1.1, floor=0.15)
        self.assertNotIn("progress_lost", raised({"loss": loss}))

    def test_a_warm_restart_is_not_lost_progress(self):
        loss = []
        for start in (2.0, 1.2, 0.7):
            loss.extend(falling(n=20, start=start, floor=start * 0.35))
        self.assertNotIn("progress_lost", raised({"loss": loss}))

    def test_one_bad_batch_is_a_spike_not_lost_progress(self):
        # Well past the explosion multiple, so it is unambiguously a spike. What must
        # not happen is the *same* reading also being called lost progress: the run is
        # back to normal on the next epoch and has given up nothing.
        loss = falling()
        loss[30] = loss[30] * 20
        found = raised({"loss": loss})
        self.assertIn("loss_spike", found)
        self.assertNotIn("progress_lost", found, found)

    def test_a_spike_below_the_bar_is_left_alone_entirely(self):
        loss = falling()
        loss[30] = loss[30] * 6          # under the explosion multiple at this dial
        found = raised({"loss": loss})
        self.assertNotIn("progress_lost", found, found)

    def test_a_noisy_flat_run_has_not_lost_progress(self):
        import random
        random.seed(3)
        self.assertNotIn("progress_lost",
                         raised({"loss": [0.12 + random.gauss(0, 0.01) for _ in range(EPOCHS)]}))


class RelationTest(unittest.TestCase):
    """Faults that no single curve carries."""

    def test_ranks_that_started_together_and_came_apart(self):
        base = falling()
        histories = {"loss": base}
        for rank in range(4):
            offset = 0.8 * (rank - 1.5) / 1.5
            histories[f"loss_rank{rank}"] = [v + offset * i / EPOCHS for i, v in enumerate(base)]
        self.assertIn("group_diverged", raised(histories))

    def test_one_worker_holding_up_the_others(self):
        histories = {"loss": falling()}
        for rank in range(4):
            histories[f"step_time_rank{rank}"] = flat(5.0 if rank == 2 else 0.25)
        self.assertIn("group_outlier", raised(histories))

    def test_ranks_that_agree_are_not_a_finding(self):
        base = falling()
        histories = {"loss": base}
        for rank in range(4):
            histories[f"loss_rank{rank}"] = [v * (1 + 0.005 * rank) for v in base]
        found = raised(histories)
        self.assertNotIn("group_diverged", found, found)
        self.assertNotIn("group_outlier", found, found)

    def test_many_unrelated_metrics_are_not_a_shard_group(self):
        # metric_0 .. metric_39 are forty different numbers that happen to be numbered.
        import random
        random.seed(11)
        histories = {"loss": falling()}
        for k in range(40):
            histories[f"metric_{k}"] = [random.gauss(k, 0.1) for _ in range(EPOCHS)]
        found = raised(histories)
        self.assertNotIn("group_diverged", found, found)
        self.assertNotIn("group_outlier", found, found)

    def test_an_ema_drifting_from_the_weights_it_shadows(self):
        live = falling()
        histories = {"loss": live,
                     "ema_loss": [v + 1.5 * i / EPOCHS for i, v in enumerate(live)]}
        self.assertIn("pair_drift", raised(histories))

    def test_validation_computed_on_the_training_set(self):
        train = falling()
        self.assertIn("identical_series", raised({"loss": train, "val_loss": list(train)}))

    def test_a_normal_generalisation_gap_is_not_identical(self):
        train = falling()
        found = raised({"loss": train, "val_loss": [v + 0.2 for v in train]})
        self.assertNotIn("identical_series", found, found)

    def test_perplexity_that_is_not_exp_of_the_loss(self):
        loss = falling(start=4.0, floor=1.8)
        found = raised({"loss": loss, "perplexity": [math.exp(v) * 5.0 for v in loss]})
        self.assertIn("relationship_broken", found, found)

    def test_perplexity_that_matches_is_left_alone(self):
        loss = falling(start=4.0, floor=1.8)
        found = raised({"loss": loss, "perplexity": [math.exp(v) for v in loss]})
        self.assertNotIn("relationship_broken", found, found)

    def test_a_reward_climbing_while_the_episodes_collapse(self):
        found = raised({"reward": ramp(-50.0, 100.0), "episode_length": ramp(500.0, 20.0)})
        self.assertIn("objective_gamed", found, found)

    def test_training_pulling_away_while_validation_stands_still(self):
        train = falling()
        val = [0.55] * EPOCHS
        found = raised({"loss": train, "val_loss": val})
        self.assertIn("widening_gap", found, found)

    def test_a_loss_settled_at_chance_for_its_class_count(self):
        chance = math.log(1000)
        # Settles onto chance rather than still approaching it at the last epoch.
        found = raised({"loss": [chance + 2.0 * math.exp(-8.0 * i / EPOCHS) for i in range(EPOCHS)],
                        "num_classes": flat(1000.0)})
        self.assertIn("chance_level", found, found)

    def test_a_loss_below_chance_is_learning(self):
        found = raised({"loss": falling(start=6.9, floor=0.8), "num_classes": flat(1000.0)})
        self.assertNotIn("chance_level", found, found)

    def test_a_loss_term_that_stopped_contributing(self):
        kl = [5.0 * math.exp(-5.0 * i / EPOCHS) for i in range(EPOCHS)]
        recon = falling(start=3.0, floor=0.8)
        found = raised({"loss": [a + b for a, b in zip(kl, recon)],
                        "kl_loss": kl, "recon_loss": recon})
        self.assertIn("component_collapse", found, found)

    def test_validation_noisier_than_training(self):
        import random
        random.seed(5)
        train = falling()
        found = raised({"loss": train,
                        "val_loss": [v + 0.1 + random.gauss(0, 0.45) for v in train]})
        self.assertIn("eval_unstable", found, found)

    def test_a_normally_noisy_validation_curve_is_fine(self):
        import random
        random.seed(5)
        train = [v + random.gauss(0, 0.01) for v in falling()]
        found = raised({"loss": train,
                        "val_loss": [v + 0.06 + random.gauss(0, 0.015) for v in train]})
        self.assertNotIn("eval_unstable", found, found)


class HyperparameterTest(unittest.TestCase):
    def test_an_epsilon_that_swamps_the_optimizer(self):
        found = raised({"loss": falling(floor=0.9), "adam_eps": flat(1.0)})
        self.assertIn("hyperparameter_out_of_range", found, found)

    def test_batchnorm_that_never_updates_its_statistics(self):
        found = raised({"loss": falling(), "bn_momentum": flat(0.0)})
        self.assertIn("hyperparameter_out_of_range", found, found)

    def test_a_step_count_is_not_an_epsilon(self):
        # `eps` is a substring of `steps`, and matching it there told a healthy run its
        # step count was an optimizer epsilon out of range.
        for name in ("steps", "STEPS", "total_steps", "num_steps"):
            found = raised({"loss": falling(), name: ramp(0.0, 600.0)})
            self.assertNotIn("hyperparameter_out_of_range", found, f"{name}: {found}")

    def test_sane_values_are_not_reported(self):
        found = raised({"loss": falling(), "adam_eps": flat(1e-8),
                        "dropout": flat(0.1), "weight_decay": flat(0.01)})
        self.assertNotIn("hyperparameter_out_of_range", found, found)

    def test_a_scheduled_value_is_not_a_misconfiguration(self):
        # A temperature being annealed on purpose takes many values; a config does not.
        found = raised({"loss": falling(), "temperature": ramp(1.0, 0.2)})
        self.assertNotIn("hyperparameter_out_of_range", found, found)


class TensorTest(unittest.TestCase):
    @staticmethod
    def stats(**over):
        base = {"shape": [256, 256], "device": "cuda:0", "dtype": "float32",
                "mean": 0.01, "std": 0.02, "min": -0.1, "max": 0.1, "nan": 0, "inf": 0}
        base.update(over)
        return {"layer1.weight": base}

    def test_a_weight_that_is_finite_and_enormous(self):
        found = raised({"loss": falling()}, tensor_stats=self.stats(std=1e12, max=1e12))
        self.assertIn("tensor_norm_explosion", found, found)

    def test_a_weight_that_is_entirely_zero(self):
        found = raised({"loss": falling()}, tensor_stats=self.stats(mean=0.0, std=0.0))
        self.assertIn("tensor_all_zeros", found, found)

    def test_a_dtype_that_changes_mid_run(self):
        sequence = [self.stats(dtype="float32")] * 20 + [self.stats(dtype="float16")] * 20
        self.assertIn("tensor_dtype_change", raised({"loss": falling()}, tensor_stats=sequence))

    def test_a_parameter_that_moves_to_the_cpu(self):
        sequence = [self.stats(device="cuda:0")] * 20 + [self.stats(device="cpu")] * 20
        self.assertIn("tensor_device_change", raised({"loss": falling()}, tensor_stats=sequence))

    def test_an_ordinary_weight_says_nothing(self):
        found = raised({"loss": falling()}, tensor_stats=self.stats())
        self.assertEqual([c for c in found if c.startswith("tensor")], [], found)

    def test_nan_counts_still_take_priority(self):
        found = raised({"loss": falling()}, tensor_stats=self.stats(nan=17, std=1e12))
        self.assertIn("tensor_nonfinite", found, found)


class CostTest(unittest.TestCase):
    """Detection runs on the training thread in cli mode, so its cost is the run's cost."""

    def test_a_long_heavily_instrumented_run_stays_cheap(self):
        import random
        random.seed(0)
        steps = 20000
        histories = {"loss": [2.0 * math.exp(-3.0 * i / steps) for i in range(steps)]}
        histories["val_loss"] = [v + 0.06 for v in histories["loss"]]
        for k in range(60):
            histories[f"metric_{k}"] = [random.gauss(1.0, 0.1) for _ in range(steps)]

        engine = DetectionEngine()
        engine.update(histories, step=steps)              # warm any caches
        started = time.perf_counter()
        for _ in range(3):
            engine.update(histories, step=steps)
        each = (time.perf_counter() - started) / 3
        # Generous: the point is to catch a return to re-reading every value in every
        # history on every update, which cost most of a second per call.
        self.assertLess(each, 0.25, f"one update took {each * 1000:.0f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
