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
