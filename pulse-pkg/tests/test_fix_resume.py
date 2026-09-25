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
    kwargs = {"epochs": 10}
    pc._resume_from_fix_checkpoint(widened, (), kwargs)   # must not raise
    assert all(np.all(np.isfinite(w)) for w in widened.get_weights())


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
