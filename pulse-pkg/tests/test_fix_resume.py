"""Checkpoint when a fix starts, resume from it after the fixed script restarts."""
import json
import os

import numpy as np
import pytest

keras = pytest.importorskip("keras")

import pulse.pulse_cli as pc
from pulse.pulse_cli import PulseCLI


def small_model(hidden=4, seed=0):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(3,)),
                              keras.layers.Dense(hidden, activation="relu"),
                              keras.layers.Dense(1, activation="sigmoid")])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def checkpointing_cli(tmp_path, model, epoch=4, epochs=10):
    cli = PulseCLI.__new__(PulseCLI)
    cli.script_path = str(tmp_path / "train.py")
    cli._keras_model, cli._keras_epoch, cli._keras_epochs = model, epoch, epochs
    return cli


def test_checkpoint_is_saved_with_where_training_was(tmp_path):
    cli = checkpointing_cli(tmp_path, small_model())
    cli._save_fix_checkpoint()
    meta = json.load(open(cli._fix_checkpoint))
    assert meta["epoch"] == 4 and meta["epochs"] == 10 and meta["finite"] is True
    assert os.path.exists(meta["weights"])
    assert cli._resume_after_fix is True


def test_no_model_means_no_checkpoint(tmp_path):
    cli = PulseCLI.__new__(PulseCLI)
    cli.script_path = str(tmp_path / "train.py")
    cli._save_fix_checkpoint()
    assert cli._fix_checkpoint is None


def test_restarted_fit_resumes_the_weights_at_the_next_epoch(tmp_path, monkeypatch):
    trained = small_model(seed=1)
    cli = checkpointing_cli(tmp_path, trained)
    cli._save_fix_checkpoint()
    monkeypatch.setenv(pc._RESUME_ENV, cli._fix_checkpoint)

    fresh = small_model(seed=2)
    kwargs = {"epochs": 10}
    pc._resume_from_fix_checkpoint(fresh, (np.zeros((2, 3)), np.zeros(2)), kwargs)
    for a, b in zip(trained.get_weights(), fresh.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert kwargs["initial_epoch"] == 5
    assert pc._RESUME_ENV not in os.environ            # only the first fit() resumes


def test_resume_always_leaves_at_least_one_epoch(tmp_path, monkeypatch):
    cli = checkpointing_cli(tmp_path, small_model(), epoch=9, epochs=10)
    cli._save_fix_checkpoint()
    monkeypatch.setenv(pc._RESUME_ENV, cli._fix_checkpoint)
    kwargs = {"epochs": 10}
    pc._resume_from_fix_checkpoint(small_model(seed=3), (), kwargs)
    assert kwargs["initial_epoch"] == 9


def test_a_changed_architecture_does_not_break_the_restart(tmp_path, monkeypatch):
    cli = checkpointing_cli(tmp_path, small_model(hidden=4))
    cli._save_fix_checkpoint()
    monkeypatch.setenv(pc._RESUME_ENV, cli._fix_checkpoint)
    widened = small_model(hidden=8, seed=4)
    fresh = [w.copy() for w in widened.get_weights()]
    kwargs = {"epochs": 10}
    pc._resume_from_fix_checkpoint(widened, (), kwargs)   # must not raise
    # half-old, half-new is neither the trained model nor a clean start: start clean
    for a, b in zip(fresh, widened.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert "initial_epoch" not in kwargs


def test_non_finite_weights_restart_from_scratch(tmp_path, monkeypatch):
    model = small_model()
    weights = model.get_weights()
    weights[0][0, 0] = np.nan
    model.set_weights(weights)
    cli = checkpointing_cli(tmp_path, model)
    cli._save_fix_checkpoint()
    assert json.load(open(cli._fix_checkpoint))["finite"] is False


def test_restart_passes_the_checkpoint_on(tmp_path):
    cli = checkpointing_cli(tmp_path, small_model())
    cli._save_fix_checkpoint()
    env = cli._restart_env(depth=0)
    assert env.get(pc._RESUME_ENV) == cli._fix_checkpoint
    assert env[pc._RESTART_CHILD_ENV] == "1"


def test_a_fix_that_asks_for_a_fresh_start_gets_one(tmp_path):
    cli = checkpointing_cli(tmp_path, small_model())
    cli._save_fix_checkpoint()
    cli._resume_after_fix = False
    assert pc._RESUME_ENV not in cli._restart_env(depth=0)


def test_no_checkpoint_restarts_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv(pc._RESUME_ENV, "/stale/from/an/earlier/restart.json")
    cli = PulseCLI.__new__(PulseCLI)
    assert pc._RESUME_ENV not in cli._restart_env(depth=0)


def test_the_fix_can_ask_for_a_fresh_start():
    assert "resume: OPTIONAL" in pc.SYSTEM_PROMPT


def test_keras_steps_are_batches_across_the_whole_run():
    class FakePulse:
        auto_intervene = True
        def __init__(self):
            self.steps_seen = []
        def _record_keras_batch_logs(self, logs): pass
        def _record_keras_logs(self, logs, epoch=None): pass
        def update(self, step=None):
            self.steps_seen.append(step)
    tracker_cls = pc._build_keras_tracker_class(keras.callbacks.Callback)
    pulse = FakePulse()
    for fit_call in range(2):
        tracker = tracker_cls(pulse)
        tracker.set_model(small_model())
        tracker.set_params({"epochs": 2})
        tracker.on_train_begin()
        for epoch in range(2):
            for batch in range(30):
                tracker.on_train_batch_end(batch)
            tracker.on_epoch_end(epoch, {"loss": 0.5})
    assert pulse.steps_seen == [30, 60, 90, 120]


class _SessionFreeCLI(PulseCLI):
    """PulseCLI without a live session: any state _restart_process reads that a real run
    would have set up is simply absent (None)."""
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def test_after_a_resumed_rerun_fails_the_next_attempt_starts_fresh(tmp_path, monkeypatch):
    cli = _SessionFreeCLI.__new__(_SessionFreeCLI)
    cli.script_path = str(tmp_path / "train.py")
    cli._keras_model, cli._keras_epoch, cli._keras_epochs = small_model(), 4, 10
    open(cli.script_path, "w").write("pass\n")
    cli._save_fix_checkpoint()
    cli._log_incident = lambda *a, **k: None
    cli._finalize_agent_downtime = lambda: None
    cli._resolve_restart_interpreter = lambda: "python3"
    verdicts = iter([(False, "val_loss still stuck"), (True, "")])
    cli._confirm_fix_did_its_job = lambda result: next(verdicts)
    launched = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(argv, capture_output=True, text=True, env=None, **kw):
        launched.append(env.get(pc._RESUME_ENV))
        return Done()

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    monkeypatch.setattr(pc.time, "sleep", lambda s: None)
    monkeypatch.delenv("PULSE_AUTO_RESTART", raising=False)
    with pytest.raises(SystemExit):
        cli._restart_process()
    assert launched[0] == cli._fix_checkpoint      # first attempt resumes
    assert launched[1] is None                     # it did not work: the next starts fresh
