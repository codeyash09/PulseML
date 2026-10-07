"""Does each path that runs the detector give it what its checks need?

An audit drove the same synthetic runs through the engine directly, through a real
Monitor -> stream -> Brain, through a real PulseCLI.update(), and through the Tk
dashboard's manifest, and found checks that worked on one path and could not fire (or
fired falsely) on another. One test per wiring fault it found.
"""
import json
import math
import os
import random
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from pulse import pulse_detect as detect                 # noqa: E402
from pulse import pulse_monitor                          # noqa: E402
from pulse import pulse_stream as stream                 # noqa: E402
from pulse.pulse_brain import Brain                      # noqa: E402
from pulse.pulse_monitor import Monitor                  # noqa: E402


def decay(i, a=2.0, b=0.3, tau=60.0):
    return b + a * math.exp(-i / tau)


def fresh(value):
    """A new float object, the way a loop that recomputes a value produces one."""
    return float(repr(value))


class Pipe:
    """A real Monitor writing a real stream that a real Brain reads, one poll per step."""

    def __init__(self, directory, readings_per_poll=1):
        self.monitor = Monitor(directory=str(directory), session_id="wiring",
                               interval=0.0, tensor_interval=0.0)
        self.brain = Brain(str(directory))
        self.brain.schedule.due = lambda *a, **k: False          # no audits, no model
        self.brain.tensor_probe_interval = 0.0
        self.raised = []
        self.brain.on_finding = lambda found: self.raised.extend(found)
        self.per_poll = readings_per_poll
        self.count = 0

    def sample(self, values, step):
        self.monitor._last_control_poll = -1e9                   # answer requests at once
        self.monitor.observe_locals(values, step=step)
        self.count += 1
        if self.count % self.per_poll == 0:
            self.poll()

    def poll(self):
        self.monitor.writer.flush()
        self.brain.poll_once()

    def close(self):
        self.monitor.close()
        self.brain.poll_once()

    def checks(self):
        return {f.check for f in self.raised}


# ---------------------------------------------------------------- monitor: what a reading is

def test_monitor_marks_the_same_object_read_again_as_a_repeat(tmp_path):
    monitor = Monitor(directory=str(tmp_path), session_id="m", interval=0.0, tensor_interval=0.0)
    val_loss = fresh(0.75)
    monitor.observe_locals({"val_loss": val_loss}, step=1)
    monitor.observe_locals({"val_loss": val_loss}, step=2)            # nobody reassigned it
    monitor.observe_locals({"val_loss": fresh(0.75)}, step=3)         # recomputed, same number
    monitor.close()
    frames = [f for f in stream.StreamReader(str(tmp_path)).poll() if f.get("kind") == stream.KIND_SCALARS]
    assert [("val_loss" in (f.get("repeats") or ())) for f in frames] == [False, True, False]


def test_monitor_counts_a_constant_counter_or_lr_on_every_step(tmp_path):
    monitor = Monitor(directory=str(tmp_path), session_id="m", interval=0.0, tensor_interval=0.0)
    lr, saved = fresh(1e-3), 7
    for step in range(3):
        monitor.observe_locals({"lr": lr, "checkpoints_saved": saved}, step=step)
    monitor.close()
    frames = [f for f in stream.StreamReader(str(tmp_path)).poll() if f.get("kind") == stream.KIND_SCALARS]
    assert not any(f.get("repeats") for f in frames)


# ---------------------------------------------------------------- brain: readings, not samples

def test_brain_does_not_take_an_epoch_level_val_loss_for_a_frozen_one(tmp_path):
    """A per-epoch val_loss sampled every step raised frozen, never_learned (CRITICAL),
    plateau and stagnation on a healthy run, and the app started an agent turn on it."""
    val_losses = [fresh(decay(25 * k, a=1.5, b=0.4)) for k in range(10)]
    val_accs = [fresh(0.5 + 0.4 * (1 - math.exp(-k / 4))) for k in range(10)]
    pipe, r = Pipe(tmp_path), random.Random(0)
    for i in range(250):
        pipe.sample({"loss": decay(i) + 0.01 * r.random(), "val_loss": val_losses[i // 25],
                     "val_acc": val_accs[i // 25]}, step=i)
    pipe.close()
    assert pipe.raised == [], [(f.check, f.variable) for f in pipe.raised]
    assert len(pipe.brain.histories["val_loss"]) == 10


def test_brain_still_sees_a_loss_that_froze(tmp_path):
    pipe, r = Pipe(tmp_path), random.Random(1)
    for i in range(200):
        pipe.sample({"loss": fresh(decay(i) + 0.01 * r.random() if i < 150 else 0.9123456789)}, step=i)
    pipe.close()
    assert "frozen" in pipe.checks()


def test_brain_does_not_confirm_a_finding_with_another_variables_frame(tmp_path):
    """require_new_data: a one-reading grad_norm spike, logged in its own frame, was
    confirmed as soon as the loss's frame arrived -- the same spike, looked at twice."""
    pipe, r = Pipe(tmp_path), random.Random(2)
    for i in range(240):
        pipe.monitor.observe("grad_norm", 100.0 if i == 200 else 1.0 + 0.05 * r.random(), step=i)
        pipe.poll()
        pipe.monitor.observe("loss", decay(i) + 0.005 * r.random(), step=i)
        pipe.poll()
    pipe.close()
    assert "norm_explosion" not in pipe.checks()


def test_brain_confirms_a_sustained_explosion(tmp_path):
    pipe, r = Pipe(tmp_path), random.Random(3)
    for i in range(230):
        pipe.sample({"loss": decay(i) + 0.005 * r.random(),
                     "grad_norm": (100.0 if i >= 200 else 1.0) + 0.05 * r.random()}, step=i)
    pipe.close()
    assert "norm_explosion" in pipe.checks()


@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("kind", ["loss_spike", "nonfinite"])
def test_brain_sees_a_transient_that_is_not_the_last_reading_of_a_poll(tmp_path, kind, offset):
    pipe, r = Pipe(tmp_path, readings_per_poll=3), random.Random(4)
    for i in range(240):
        value = decay(i) + 0.005 * r.random()
        if i == 201 + offset:
            value = value * 30 if kind == "loss_spike" else float("nan")
        pipe.sample({"loss": value}, step=i)
    pipe.close()
    assert kind in pipe.checks()


# ---------------------------------------------------------------- brain: tensors

def test_brain_runs_the_tensor_checks(tmp_path):
    torch = pytest.importorskip("torch")
    pipe = Pipe(tmp_path)
    for i in range(40):
        weight = torch.randn(8, 8)
        if i >= 20:
            weight = weight.half()                   # metadata alone shows this
        if i == 30:
            weight[0, 0] = float("nan")              # only statistics show this
        pipe.sample({"loss": decay(i), "weight": weight}, step=i)
    pipe.close()
    assert {"tensor_dtype_change", "tensor_nonfinite"} <= pipe.checks()
    # The probe's numpy dtype ("float16") must not read as a change from "torch.float16".
    assert [f.check for f in pipe.raised].count("tensor_dtype_change") == 1


def test_brain_asks_for_statistics_of_host_tensors_only(tmp_path):
    brain = Brain(str(tmp_path))
    brain.tensors = {"w_host": {"device": "cpu", "elements": 64},
                     "w_gpu": {"device": "cuda:0", "elements": 64},
                     "huge": {"device": "cpu", "elements": 10 ** 9}}
    brain._maybe_probe_tensors(now=100.0)
    brain._maybe_probe_tensors(now=101.0)            # not due again yet
    with open(brain.reader.control_path) as handle:
        messages = [json.loads(line) for line in handle]
    assert len(messages) == 1
    assert messages[0]["action"] == stream.CONTROL_SNAPSHOT and messages[0]["auto"] is True
    assert messages[0]["names"] == ["w_host"]


def test_monitor_skips_an_unprompted_probe_of_a_big_tensor(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(pulse_monitor, "AUTO_PROBE_MAX_ELEMENTS", 10)
    monitor = Monitor(directory=str(tmp_path), session_id="m", interval=0.0, tensor_interval=0.0)
    monitor._handle_message({"action": stream.CONTROL_SNAPSHOT, "names": ["big", "small"], "auto": True})
    monitor.observe_locals({"big": np.zeros((8, 8)), "small": np.zeros(4)}, step=1)
    monitor.close()
    frames = stream.StreamReader(str(tmp_path)).poll()
    probed = {f["name"] for f in frames if f.get("kind") == stream.KIND_TENSOR and f.get("stats")}
    skipped = {f["name"] for f in frames if f.get("event") == "probe_skipped"}
    assert probed == {"small"} and skipped == {"big"}


# ---------------------------------------------------------------- CLI: constants are readings

def make_cli():
    from pulse import pulse_cli
    cli = pulse_cli.PulseCLI(watch_locals={"_": 0}, pdf_dir=os.path.join(HERE, "_wiring_out"))
    cli.sensitivity = 0.3
    cli.continuous = True
    cli.generate_pdfs = False
    cli._maybe_periodic_checkin = lambda: None
    cli._prime_with_agent_if_needed = lambda: None
    problems = []
    # what the detector found, acted on or only shown (pulse_detect.acts_on decides which)
    act = cli._act_on_detection

    def record(problem):
        problems.extend(f.message for f in (cli._last_detector_findings or []))
        act(problem)
    cli._act_on_detection = record
    cli._escalate_training_problem = lambda problem: None
    return cli, problems


def drive_cli(cli, steps):
    import contextlib
    import io
    names = sorted({name for values in steps for name in values})
    cli.tracked_vars = names
    for values in steps:
        cli.watch_locals.clear()
        cli.watch_locals.update(values)
        with contextlib.redirect_stdout(io.StringIO()):
            cli.update()


def test_cli_sees_an_lr_jump_when_the_lr_is_the_optimizers_own_float():
    """`lr = opt.param_groups[0]["lr"]` is one object until the schedule moves it: kept
    only when it changed, the LR had two readings at its jump and lr_jump needs three."""
    cli, problems = make_cli()
    low, high, r = fresh(1e-4), fresh(1e-1), random.Random(5)
    drive_cli(cli, [{"loss": fresh(decay(i) + 0.005 * r.random()), "lr": low if i < 60 else high}
                    for i in range(70)])
    assert any("lr changed by" in p for p in problems), problems


def test_cli_sees_chance_level_with_an_int_class_count():
    cli, problems = make_cli()
    r = random.Random(6)
    drive_cli(cli, [{"loss": fresh((decay(i, a=1.0, b=math.log(10)) if i < 60 else math.log(10))
                                   + 1e-4 * r.random()), "num_classes": 10} for i in range(100)])
    assert any("ln(10)" in p for p in problems), problems


def test_counters_lrs_and_class_counts_record_every_tick():
    for name in ("lr", "learning_rate", "num_classes", "checkpoints_saved", "adam_eps"):
        assert detect.records_every_tick(name), name
    for name in ("loss", "val_loss", "accuracy"):
        assert not detect.records_every_tick(name), name


# ---------------------------------------------------------------- engine: waste_pinned

def test_waste_pinned_is_reported_for_as_long_as_it_is_pinned():
    engine = detect.DetectionEngine()
    history = []
    seen = set()
    for i in range(40):
        history.append(1.0)
        engine.update({"clip_fraction": list(history)}, step=i)
        seen |= {f.check for f in engine.current()}
    assert "waste_pinned" in {f.check for f in engine.current()}
    assert "waste_high" not in seen


# ---------------------------------------------------------------- Tk dashboard

def dashboard_check(manifest, sensitivity=0.3, detector=None):
    from pulse import pulse as dashboard_module
    fake = types.SimpleNamespace(_detector=detector, explosion_multiplier=None, sensitivity=sensitivity)
    result = dashboard_module.Dashboard._check_for_trouble(fake, json.loads(json.dumps(manifest)))
    return result, fake._detector


def scalar_entry(points):
    """What the manifest writer keeps: changes only, the last twenty as `recent`."""
    return {"kind": "scalar", "latest_value": points[-1][1], "recent": [v for _, v in points[-20:]],
            "history": [list(p) for p in points[-2000:]]}


def test_dashboard_leaves_a_healthy_descent_alone():
    """With only the 20-value window, a gently descending loss read as "no better than
    when it started" -- a CRITICAL that paused the run -- about 290 steps in."""
    r, points, detector, problems = random.Random(7), [], None, []
    for i in range(320):
        points.append((i, decay(i) + 0.005 * r.random()))
        problem, detector = dashboard_check({"loss": scalar_entry(points)}, detector=detector)
        if problem:
            problems.append(problem)
    assert problems == []


def test_dashboard_runs_the_tensor_checks_on_the_full_statistics():
    stats = {"kind": "matrix", "shape": [16, 16], "dtype": "float32", "device": "cpu",
             "mean": 0.0, "std": 3e8, "min": -1e9, "max": 1e9, "nan": 0, "inf": 0, "updated": 1.0}
    problem, detector = dashboard_check({"weight": stats})
    assert problem is None                       # one look; it is confirmed on the next probe
    problem, detector = dashboard_check({"weight": stats}, detector=detector)
    assert problem is None                       # the same manifest re-read is not a new look
    problem, _ = dashboard_check({"weight": dict(stats, updated=2.0)}, detector=detector)
    assert problem and "standard deviation" in problem


def test_dashboard_reads_the_configured_sensitivity(tmp_path, monkeypatch):
    from pulse import pulse as dashboard_module
    config = tmp_path / "pulse_config.json"
    config.write_text(json.dumps({"Sensitivity": "tight"}))
    monkeypatch.delenv("PULSE_SENSITIVITY", raising=False)
    monkeypatch.setenv("PULSE_CONFIG", str(config))
    assert dashboard_module._configured_sensitivity() == 0.75
    monkeypatch.setenv("PULSE_SENSITIVITY", "0.9")
    assert dashboard_module._configured_sensitivity() == 0.9
    monkeypatch.delenv("PULSE_SENSITIVITY")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("PULSE_CONFIG")
    monkeypatch.chdir(empty)
    assert dashboard_module._configured_sensitivity(default=0.3) == 0.3


def test_parse_sensitivity():
    assert detect.parse_sensitivity("loose") == 0.2
    assert detect.parse_sensitivity(2) == 1.0
    assert detect.parse_sensitivity("nonsense", 0.4) == 0.4
    assert detect.parse_sensitivity(None) == 0.3
