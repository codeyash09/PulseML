"""Sweep 2 -- Pulse's Keras fit() hook against REAL Keras 3 (CPU only).

test_bug_* assert the CORRECT behaviour and fail on the current code.
"""
import os
import signal

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

keras = pytest.importorskip("keras")

import pulse.pulse_cli as pc
from pulse.pulse_cli import PulseCLI


@pytest.fixture
def hooked(tmp_path, monkeypatch):
    """A PulseCLI with Pulse's fit hook installed on the real keras.Model (restored after)."""
    old_sigint = signal.getsignal(signal.SIGINT)
    monkeypatch.chdir(tmp_path)
    for name in (pc._RESUME_ENV, pc._RESTART_CHILD_ENV, "PULSE_CONFIG", "PULSE_AUTO_RESTART"):
        monkeypatch.delenv(name, raising=False)
    cli = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdf"))
    cli.continuous = True
    cli.auto_intervene = False
    cli._start_primed = cli._start_primed_with_agent = True
    cli._ensure_retry_ticker = lambda: None
    monkeypatch.setattr(pc, "_PULSE_ACTIVE_INSTANCE", cli)
    monkeypatch.setattr(pc, "_PULSE_KERAS_HOOK_INSTALLED", False)
    monkeypatch.setattr(pc, "_PULSE_KERAS_TRACKER_CLS", None)
    model_cls = pc._keras_model_class()
    assert model_cls is not None
    original_fit = model_cls.__dict__.get("fit")
    pc._install_keras_fit_hook()
    assert pc._PULSE_KERAS_HOOK_INSTALLED
    yield cli
    if original_fit is not None:
        model_cls.fit = original_fit
    elif "fit" in model_cls.__dict__:
        delattr(model_cls, "fit")                      # back to the inherited Trainer.fit
    signal.signal(signal.SIGINT, old_sigint)


def _data(n=320):
    rng = np.random.default_rng(0)
    return rng.random((n, 4), dtype=np.float32), rng.random((n, 1), dtype=np.float32)


def _model(**compile_kw):
    m = keras.Sequential([keras.Input((4,)), keras.layers.Dense(1)])
    m.compile("sgd", "mse", **compile_kw)
    return m


@pytest.mark.parametrize("form", ["single_callback", "callback_list"])
def test_bug_single_callback_object_breaks_fit(hooked, form):
    """Keras 3 flattens `callbacks` (tree.flatten) and also takes a ready CallbackList, so
    `model.fit(x, y, callbacks=EarlyStopping(...))` (one callback, no list) or
    `callbacks=keras.callbacks.CallbackList([...])` trains fine without Pulse. pulse_fit does
    `list(callbacks)`, which raises TypeError: '<Callback>' object is not iterable -- Pulse
    breaks a working fit() call. Correct: such a value is wrapped ([callbacks, tracker]) and
    fit() runs with Pulse's tracker attached."""
    x, y = _data()
    es = keras.callbacks.EarlyStopping(monitor="loss", patience=5)
    cbs = es if form == "single_callback" else keras.callbacks.CallbackList([es])
    _model().fit(x, y, epochs=1, verbose=0, callbacks=cbs)


def test_bug_steps_per_execution_undercounts_training_steps(hooked):
    """on_train_batch_end adds 1 to _keras_steps per call, but with compile(steps_per_execution=N)
    Keras calls it once per N batches (batch = end_step). 20 batches at N=10 are counted as 2
    steps: step-scheduled check-ins fire 10x later than the agent asked (a 20-step check-in
    after 200 batches). Correct: 2 epochs x 10 batches -> 20 steps."""
    seen = []
    hooked.update = lambda step=None, **kw: seen.append(step)
    x, y = _data()
    _model(steps_per_execution=10).fit(x, y, batch_size=32, epochs=2, verbose=0)
    assert seen and seen[-1] == 20, f"20 training batches were counted as steps {seen}"


def test_ok_real_keras_fit_counts_batches_and_records_epoch_logs(hooked):
    seen = []
    hooked.update = lambda step=None, **kw: seen.append(step)
    x, y = _data()
    m = _model()
    m.fit(x, y, batch_size=32, epochs=2, verbose=0)
    m.fit(x, y, 32, 1, 0, [keras.callbacks.History()])     # positional callbacks merge
    assert seen == [10, 20, 30]
    assert len(hooked.epoch_scalar_histories["loss"]) == 3
    assert hooked._keras_model is m and hooked._keras_fit_count == 2


# ---- fix-round tests for the entries confirmed by reading ---------------------------------

def _checkpoint(cli, tmp_path, model, epoch, epochs):
    cli.script_path = str(tmp_path / "train.py")
    cli._keras_model, cli._keras_epoch, cli._keras_epochs = model, epoch, epochs
    cli._keras_fit_count = 1
    cli._save_fix_checkpoint()
    assert cli._fix_checkpoint
    return cli._fix_checkpoint


class _Epochs(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.ran = []

    def on_epoch_begin(self, epoch, logs=None):
        self.ran.append(epoch)


def test_bug_positional_initial_epoch_conflicts_with_resume(hooked, tmp_path, monkeypatch):
    """_resume_from_fix_checkpoint wrote kwargs["initial_epoch"] even when the script passed
    initial_epoch positionally (fit's 12th parameter): TypeError 'got multiple values for
    argument initial_epoch' on the resumed run. Correct: it resumes at the next epoch."""
    x, y = _data()
    monkeypatch.setenv(pc._RESUME_ENV, _checkpoint(hooked, tmp_path, _model(), 2, 5))
    monkeypatch.setattr(pc, "_RESUME_FITS_SEEN", {})
    seen = _Epochs()
    _model().fit(x, y, 32, 5, 0, [seen], 0.0, None, True, None, None, 0)
    assert seen.ran == [3, 4]


def test_bug_user_callback_raising_in_on_epoch_end_leaves_the_checkpoint_an_epoch_behind(hooked):
    """Pulse's tracker ran after the user's callbacks, so one raising in on_epoch_end left
    _keras_epoch at the previous epoch: the resume trained the finished epoch again on the
    updated weights. Correct: the finished epoch is recorded before any user callback runs."""
    class Raises(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 1:
                raise RuntimeError("checkpoint dir is full")
    x, y = _data()
    m = _model()
    with pytest.raises(RuntimeError):
        m.fit(x, y, epochs=4, verbose=0, callbacks=[Raises()])
    assert hooked._keras_epoch == 1 and hooked._keras_model is m


def test_bug_second_fit_is_a_new_detector_segment_but_the_history_keeps_both(hooked):
    x, y = _data()
    m = _model()
    m.fit(x, y, epochs=2, verbose=0)
    _model().fit(x, y, epochs=1, verbose=0)
    assert len(hooked.epoch_scalar_histories["loss"]) == 3
    assert len(hooked._history_for_detector("loss")) == 1


class _NoSession(PulseCLI):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def test_bug_fix_checkpoints_are_left_in_the_project_and_the_judge_is_told_run_from_the_start(
        tmp_path, monkeypatch):
    """After a resumed re-run that did its job, the newest weight files stayed in
    .pulse_checkpoints/ (gigabytes for a large model), and PASS 6 was told the program was
    're-run from the start' while its output covered only the resumed epochs.
    Correct: the judge hears it resumed from a checkpoint; the checkpoints are deleted."""
    cli = _NoSession.__new__(_NoSession)
    open(tmp_path / "train.py", "w").write("pass\n")
    _checkpoint(cli, tmp_path, _model(), 2, 5)
    cli.agent_provider, cli.agent_key = next(iter(pc.PROVIDERS)), "k"
    cli._log_incident = lambda *a, **k: None
    cli._finalize_agent_downtime = lambda: None
    cli._resolve_restart_interpreter = lambda: "python3"
    prompts = []
    cli._call_model = lambda prompt, **k: (prompts.append(prompt), '{"resolved": true, "reason": "ok"}')[1]

    class Done:
        returncode, stdout, stderr = 0, "TRAINED", ""

    monkeypatch.setattr(pc.subprocess, "run", lambda argv, **kw: Done())
    monkeypatch.setattr(pc.time, "sleep", lambda s: None)
    monkeypatch.delenv("PULSE_AUTO_RESTART", raising=False)
    with pytest.raises(SystemExit):
        cli._restart_process()
    assert prompts and "resuming training from a checkpoint" in prompts[0]
    assert "re-run from the start" not in prompts[0]
    assert not (tmp_path / ".pulse_checkpoints").exists()
