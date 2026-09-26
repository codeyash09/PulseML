"""Sweep 2 -- PulseCLI's training-loop integration (src/pulse/pulse_cli.py).

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_*  are regression coverage for behaviour verified to work.

No real model calls: agent calls are stubbed, subprocess.run is faked for restarts, and
any retry-ticker thread started here is parked (daemon) so it cannot outlive the test's
stubs in a harmful way.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time as _real_time
import types

import numpy as np
import pytest

import pulse.pulse_cli as pc
from pulse.pulse_cli import PulseCLI


# --------------------------------------------------------------------------- helpers
# (copied from tests/sweep/test_sweep_cli_loop_integration.py so this file is self-contained)

class FakeTime:
    def __init__(self, now=1000.0, sleep=None):
        self.now = now
        self._sleep = sleep

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if self._sleep is not None:
            self._sleep(seconds)

    def __getattr__(self, name):
        return getattr(_real_time, name)


@pytest.fixture
def env_guard(monkeypatch):
    saved_env = dict(os.environ)
    saved_providers = dict(pc.PROVIDERS)
    for name in ("PULSE_WEBHOOK_URL", pc._RESTART_CHILD_ENV, pc._RESTART_DEPTH_ENV, pc._RESUME_ENV,
                 "PULSE_AUTO_RESTART", "PULSE_AUTO_PROVIDER", "PULSE_AGENT_LOG", "PULSE_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    pc.PROVIDERS.clear()
    pc.PROVIDERS.update(saved_providers)


@pytest.fixture
def make_cli(tmp_path, monkeypatch, env_guard):
    old_sigint = signal.getsignal(signal.SIGINT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pc.cloud, "save_cached_profile", lambda **k: None, raising=False)

    def factory(watch_locals=None, **attrs):
        cli = PulseCLI(watch_locals=watch_locals if watch_locals is not None else {},
                       pdf_dir=str(tmp_path / "pdf"))
        cli.continuous = True
        cli.auto_intervene = False
        cli._start_primed = True
        cli._start_primed_with_agent = True
        cli._ensure_retry_ticker = lambda: None

        def no_real_model_calls(*a, **k):
            raise AssertionError("a test reached a real model call")

        cli._call_model = no_real_model_calls
        for key, value in attrs.items():
            setattr(cli, key, value)
        return cli

    yield factory
    signal.signal(signal.SIGINT, old_sigint)


def _with_agent(cli, tmp_path, code="x = 1\n"):
    script = tmp_path / "train.py"
    script.write_text(code)
    cli.script_path = str(script)
    cli.code_text = code
    cli.agent_provider = "DeepSeek" if "DeepSeek" in pc.PROVIDERS else next(iter(pc.PROVIDERS))
    cli.agent_key = "test-key"
    return cli


def _stub_restart_side_effects(cli, monkeypatch):
    cli._flush_cloud_now = lambda: None
    monkeypatch.setattr(pc, "_stdout_is_tty", lambda: False)


class FakeKerasModel:
    def __init__(self, fill=0.0):
        self.built = True
        self._w = [np.full((3, 2), fill), np.full((2,), fill)]
        self.ran_epochs = []

    def get_weights(self):
        return [w.copy() for w in self._w]

    def set_weights(self, ws):
        self._w = [np.array(w) for w in ws]

    def save_weights(self, path):
        with open(path, "wb") as f:
            np.savez(f, *self._w)

    def load_weights(self, path):
        with np.load(path) as data:
            loaded = [data[k] for k in sorted(data.files, key=lambda s: int(s.split("_")[1]))]
        if [w.shape for w in loaded] != [w.shape for w in self._w]:
            raise ValueError("shape mismatch")
        self._w = loaded

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
            initial_epoch=0, **kwargs):
        callbacks = list(callbacks or [])
        for cb in callbacks:
            cb.model, cb.params = self, {"epochs": epochs}
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin()
        for epoch in range(initial_epoch, epochs):
            for cb in callbacks:
                if hasattr(cb, "on_train_batch_end"):
                    cb.on_train_batch_end(0, {"loss": 1.0})
            self.ran_epochs.append(epoch)
            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch, {"loss": 1.0 / (epoch + 1)})
        return callbacks


class FakeCallback:
    def __init__(self):
        self.model = None
        self.params = {}


@pytest.fixture
def fake_keras(monkeypatch):
    module = types.ModuleType("keras")
    model_cls = type("Model", (FakeKerasModel,), {})
    module.Model = model_cls
    module.callbacks = types.SimpleNamespace(Callback=FakeCallback)
    monkeypatch.setitem(sys.modules, "keras", module)
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr(pc, "_PULSE_KERAS_HOOK_INSTALLED", False)
    monkeypatch.setattr(pc, "_PULSE_KERAS_TRACKER_CLS", None)
    monkeypatch.setattr(pc, "_RESUME_FITS_SEEN", {})
    return module


class FakeCudaTensor:
    """Looks like a torch CUDA tensor to Pulse: .device.type == 'cuda', detach().cpu()."""
    copies = 0

    def __init__(self, shape=(4, 4)):
        self.device = types.SimpleNamespace(type="cuda")
        self.shape = shape
        self._host = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    def detach(self):
        return self

    def cpu(self):
        FakeCudaTensor.copies += 1
        return self._host.copy()


# =========================================================================== BUGS

# ---- GPU tracking cadence ---------------------------------------------------------------

@pytest.mark.parametrize("never_copied", ["not_in_locals", "cpu_resident"])
def test_bug_one_uncopyable_gputracked_var_forces_device_copies_every_update(make_cli, monkeypatch,
                                                                                never_copied):
    """update(): probe_gpu = interval elapsed OR any gpu-tracked var not in _gpu_probed_vars.
    A var is only added to _gpu_probed_vars when Pulse actually host-copies it. A GPU-tracked
    name that is never copied -- not (yet) in watch_locals (/gputrack accepts any discovered
    name), or a CPU value / one with a CPU mirror -- keeps probe_gpu True on EVERY update, so
    every other GPU-tracked tensor gets a forced device-to-host sync every tick instead of
    every gpu_probe_interval (600 s): exactly the slowdown GPU tracking promises to avoid.
    Correct: 5 updates within one interval -> one host copy of the CUDA tensor."""
    clock = FakeTime(now=5000.0)
    monkeypatch.setattr(pc, "time", clock)
    watch = {"w": FakeCudaTensor()}
    if never_copied == "cpu_resident":
        watch["other"] = np.ones((3, 3))
    cli = make_cli(watch)
    cli.discovered = {"w": (4, 4), "other": (3, 3)}
    assert cli._cmd_gputrack("w", quiet=True) == "w"
    assert cli._cmd_gputrack("other", quiet=True) == "other"
    FakeCudaTensor.copies = 0
    for _ in range(5):
        clock.now += 1.0
        cli.update()
    assert FakeCudaTensor.copies == 1, (
        f"the CUDA tensor was copied to the host {FakeCudaTensor.copies} times in 5 s "
        f"(gpu_probe_interval={cli.gpu_probe_interval}s)")


# ---- step counting ----------------------------------------------------------------------

def test_bug_strided_minibatch_counter_counts_samples_as_steps(make_cli):
    """_detect_loop_step_delta credits a counter's raw increase. The most common NumPy /
    from-scratch minibatch loop is `for i in range(0, len(X), batch_size):` -- `i` moves by
    batch_size per step, so every step is counted batch_size times (32x here, as long as a
    tick's jump stays <= 200). Step-scheduled check-ins (paid model calls) then fire 32x as
    often as the agent asked. Correct: 10 minibatches -> ~10 steps."""
    watch = {"i": 0}
    cli = make_cli(watch)
    cli.update()                      # first sighting
    base = cli.step
    for i in range(32, 352, 32):      # 10 more minibatches
        watch["i"] = i
        cli.update()
    assert cli.step - base <= 12, f"10 minibatches of 32 counted as {cli.step - base} steps"


# ---- check-in answer parsing ------------------------------------------------------------

def test_bug_markdown_nextcheck_and_checknote_are_dropped(make_cli, tmp_path):
    """The sweep-1 fix made the VERDICT markdown-tolerant (`**VERDICT:** ok` is ok), but
    NEXTCHECK and CHECKNOTE are still matched on the raw answer with ^\\s*NEXTCHECK:. The same
    bolded answer loses the agent's chosen cadence and its note to the next check-in.
    Correct: `**NEXTCHECK:** 2000` sets the interval to 2000 and the bold CHECKNOTE is kept."""
    cli = _with_agent(make_cli(), tmp_path)
    cli.checkin_interval_steps = 500
    answer = ("**VERDICT:** ok\n**NEXTCHECK:** 2000\n"
              "**CHECKNOTE:** re-check val_loss after the LR drop at epoch 30")
    cli._finish_periodic_checkin(answer, None, "prompt", [])
    assert cli.checkin_interval_steps == 2000
    assert "val_loss" in (cli._checkin_note or "")


def test_bug_verdict_no_issues_is_escalated_as_a_problem(make_cli, tmp_path):
    """_checkin_verdict reads only the verdict's first word; 'no' is not in the ok list, so
    `VERDICT: no issues found` -> problem -> _escalate_training_problem -> the paid fix pipeline
    on a healthy run. The legacy STATUS path lists 'no issues'/'no problems' as ok
    (_CHECKIN_OK_STATUSES); the VERDICT path does not. Correct: read as ok."""
    for answer in ("VERDICT: no issues found\nNEXTCHECK: 500",
                   "VERDICT: no problems\nNEXTCHECK: 500"):
        is_problem, _ = PulseCLI._checkin_verdict(answer)
        assert is_problem is False, f"{answer.splitlines()[0]!r} was read as a problem"


# ---- exit handler -----------------------------------------------------------------------

def test_bug_stale_checkin_from_before_a_fix_restart_is_escalated_at_exit(make_cli, tmp_path, monkeypatch):
    """A check-in started on the OLD code is still in flight when an escalation applies a fix
    and _restart_process() runs the fixed script to the end, then sys.exit()s. Nothing clears
    _checkin_call, so _finish_background_calls_at_exit (SystemExit is not an interrupt or a
    crash) applies that pre-fix verdict: a 'problem' about already-fixed code starts a second
    fix chain and a second full restart of a run that just finished and was confirmed fixed.
    Correct: a check-in about code that has since been fixed and re-run is not escalated."""
    cli = _with_agent(make_cli(), tmp_path)
    cli.auto_intervene = True
    _stub_restart_side_effects(cli, monkeypatch)
    cli._checkin_call = types.SimpleNamespace(       # finished while the fixed child ran
        done=True, error=None, prompt="the run",
        result=("VERDICT: problem\nPROBLEM: learning rate 10.0 makes the loss diverge (train.py:1)\n"
                "NEXTCHECK: 500\nCHECKNOTE: none", []))
    monkeypatch.setattr(pc.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "val_acc 0.97", ""))
    cli._confirm_fix_did_its_job = lambda result: (True, "accuracy is now 0.97")
    with pytest.raises(SystemExit):
        cli._restart_process()                   # the fixed re-run finished and was confirmed
    escalated = []
    cli._escalate_training_problem = escalated.append
    cli._finish_background_calls_at_exit()       # interpreter exit after that SystemExit
    assert escalated == [], f"a pre-fix check-in was escalated after the fixed run: {escalated}"


def test_bug_exit_after_ctrl_c_still_waits_ten_seconds_for_a_discarded_checkin(make_cli, tmp_path,
                                                                                 monkeypatch):
    """_finish_background_calls_at_exit waits up to _EXIT_WAIT_SECONDS for running model calls
    and only THEN checks _interrupted / sys.last_value, so after Ctrl+C (or a crash) the
    process sits for 10 s at exit for a check-in answer it is going to throw away.
    Correct: an interrupted run skips the wait (the answer is never used)."""
    cli = _with_agent(make_cli(), tmp_path)
    clock = FakeTime(now=1000.0)
    clock._sleep = lambda s: setattr(clock, "now", clock.now + s)
    monkeypatch.setattr(pc, "time", clock)
    cli._interrupted = True
    cli._checkin_call = types.SimpleNamespace(done=False, result=None, error=None, prompt="p")
    start = clock.now
    cli._finish_background_calls_at_exit()
    assert clock.now - start < 1.0, f"waited {clock.now - start:.1f}s at exit after Ctrl+C"


# ---- restart ----------------------------------------------------------------------------

def test_bug_ctrl_c_during_restarted_run_is_treated_as_a_crash_and_relaunched(make_cli, tmp_path,
                                                                              monkeypatch):
    """Ctrl+C in the terminal reaches the whole process group: the restarted child (unattended,
    so it raises KeyboardInterrupt at once) dies with -SIGINT and a 'Traceback ... KeyboardInterrupt'
    on stderr; the interactive parent's handler only latches a pause. _restart_process classifies
    returncode < 0 as a crash, feeds 'the restarted training process crashed' to the agent (a
    paid pipeline that may edit code over a KeyboardInterrupt) and relaunches the whole script
    -- the user's Ctrl+C restarts training instead of stopping it.
    Correct: a child ended by SIGINT/KeyboardInterrupt is the user stopping the run: no agent
    call, no relaunch."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    launches, asked = [], []

    def fake_run(argv, **kw):
        launches.append(argv)
        if len(launches) == 1:
            cli._sigint_handler(signal.SIGINT, None)      # the parent got the same Ctrl+C
            return subprocess.CompletedProcess(
                argv, -signal.SIGINT, "epoch 3/50 ...",
                "Traceback (most recent call last):\n  File \"train.py\", line 9, in <module>\n"
                "    time.sleep(1)\nKeyboardInterrupt\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    monkeypatch.setattr(pc, "time", FakeTime(sleep=lambda s: None))

    def agent(question, include_code=False, _depth=0):
        asked.append(question)
        cli._fix_applied_this_turn = True

    cli._ask_agent_impl = agent
    cli._confirm_fix_did_its_job = lambda result: (True, "")
    cli._auto_rollback_after_failed_restarts = lambda commit: False
    try:
        cli._restart_process()
    except (SystemExit, KeyboardInterrupt):
        pass
    assert asked == [], "the user's Ctrl+C was handed to the agent as a crash to fix"
    assert len(launches) == 1, f"training was relaunched {len(launches) - 1} time(s) after Ctrl+C"


def test_bug_restart_from_retry_ticker_runs_beside_the_old_training(make_cli, tmp_path, monkeypatch):
    """The retry ticker (a queued restart, or a retried ask_agent whose fix restarts) calls
    _restart_process() on its own thread; the replacement script then runs start-to-finish while
    the training thread keeps running the OLD, buggy code: two full trainings at once on the same
    GPU/data/output files (the child can OOM next to its parent, and that 'crash' is then fed
    to the agent and ends in rolling back a good fix). Correct: the old training does not
    proceed (update() does not return) while the replacement run is running."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    parked = threading.Event()

    def sleep(seconds):
        if parked.is_set():
            _real_time.sleep(3600)          # park the daemon ticker after the test
        _real_time.sleep(0.01)

    monkeypatch.setattr(pc, "time", FakeTime(sleep=sleep))
    child_running, release = threading.Event(), threading.Event()

    def fake_run(argv, **kw):
        child_running.set()
        release.wait(5)
        return subprocess.CompletedProcess(argv, 1, "", "Traceback (most recent call last):\nOSError: x\n")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    cli._ask_agent_impl = lambda q, include_code=False, _depth=0: None
    cli._auto_rollback_after_failed_restarts = lambda commit: False
    cli._pending_restart_retry = {"next_attempt": 0.0, "backoff": 60.0}
    PulseCLI._ensure_retry_ticker(cli)
    overlapped = False
    try:
        if child_running.wait(5):
            t = threading.Thread(target=cli.update, daemon=True)   # the training thread's next step
            t.start()
            t.join(1.0)
            overlapped = not t.is_alive()
    finally:
        release.set()
        _real_time.sleep(0.3)
        parked.set()
    assert not overlapped, "the old training kept stepping while the fixed replacement run was running"


# ---- Keras fit index / resume -----------------------------------------------------------

def test_bug_crash_in_first_epoch_of_second_fit_resumes_with_first_fits_epoch(make_cli, fake_keras,
                                                                             tmp_path, monkeypatch):
    """_keras_model/_keras_epoch are only set in on_epoch_end and never reset when a new fit()
    starts, while the fit index is bumped at fit() start. A crash in the first epoch of fit #2
    (fine-tune stage bug at batch 1) checkpoints fit #1's model and last epoch (2) tagged as fit #2.
    After the fix-restart the fine-tune fit loads the pretrain model's weights and starts at
    epoch 3 of 5 -- 3 of its 5 epochs silently skipped. Correct: fit #2 resumes at its epoch 0
    (its own, current weights)."""
    cli = _with_agent(make_cli(), tmp_path)
    monkeypatch.setattr(pc, "_PULSE_ACTIVE_INSTANCE", cli)
    pc._install_keras_fit_hook()
    Model = fake_keras.Model
    pre, fine = Model(0.1), Model(0.9)
    pre.fit([[0.0] * 3], [0.0], epochs=3)

    class Boom(FakeCallback):
        def on_train_batch_end(self, batch, logs=None):
            raise RuntimeError("shape mismatch in the fine-tune head")

    with pytest.raises(RuntimeError):
        fine.fit([[0.0] * 3], [0.0], epochs=5, callbacks=[Boom()])
    cli.handle_crash("Traceback (most recent call last):\nRuntimeError: shape mismatch\n")
    assert cli._fix_checkpoint
    monkeypatch.setenv(pc._RESUME_ENV, cli._fix_checkpoint)
    monkeypatch.setattr(pc, "_RESUME_FITS_SEEN", {})
    pre2, fine2 = Model(0.0), Model(0.0)            # the restarted, fixed script
    pre2.fit([[0.0] * 3], [0.0], epochs=3)
    fine2.fit([[0.0] * 3], [0.0], epochs=5)
    assert fine2.ran_epochs == [0, 1, 2, 3, 4], f"fine-tune fit ran only epochs {fine2.ran_epochs}"
    assert not np.allclose(fine2.get_weights()[0], 0.1), "fine-tune model got the pretrain model's weights"


# ---- check-in scheduling ----------------------------------------------------------------

def test_bug_checkin_that_outlasts_its_nextcheck_starts_the_next_one_immediately(make_cli, tmp_path,
                                                                                  monkeypatch):
    """_last_checkin_step is set when a check-in STARTS; its answer (seconds to minutes of tool
    rounds later, while training runs on) sets checkin_interval_steps and prints 'next check-in
    for N steps from now'. If the loop took more than N steps meanwhile -- 10 ms/step and a 60 s
    investigation is 6000 steps -- the very next update() starts another check-in: back-to-back
    paid check-ins for the rest of a fast run, whatever spacing the agent asked for.
    Correct: NEXTCHECK counts from when the answer is applied ('from now')."""
    cli = _with_agent(make_cli(), tmp_path)
    cli.checkin_interval_steps = 10 ** 9            # (no check-in of its own below)
    cli.update(step=1100)                           # training ran 1000 steps meanwhile ...
    cli._last_checkin_step = 100                    # ... since the check-in started at step 100
    cli._checkin_call = types.SimpleNamespace(
        done=True, error=None, prompt="p",
        result=("VERDICT: ok\nNEXTCHECK: 200\nCHECKNOTE: none", []))
    cli.update(step=1101)                           # the answer lands: next one in 200 steps
    assert cli.checkin_interval_steps == 200
    monkeypatch.setenv("PULSE_ASYNC_MODEL_CALLS", "0")
    started = []
    cli._build_agent_context = lambda include_code=False: "snapshot"
    cli._run_checkin = lambda prompt: (started.append(cli.step), ("VERDICT: ok\nNEXTCHECK: 200", []))[1]
    cli.update(step=1102)                           # one step after the answer
    assert started == [], f"a new check-in started {1102 - 1101} step after the agent asked for 200"


# ---- Ctrl+C then an auto-restart --------------------------------------------------------

def test_bug_pending_ctrl_c_does_not_stop_the_auto_restart(make_cli, tmp_path, monkeypatch):
    """Ctrl+C while the fix pipeline runs inside update() (minutes of model calls) only latches
    _stop_requested ('Pausing after the current training step...'); the 5 s re-delivery timer
    sees _update_running and gives up. When the fix lands, _restart_process() launches a whole
    new, unattended training run anyway, ignoring the pending stop -- and a second Ctrl+C then
    hits that child too (which the parent treats as a crash, see the test above).
    Correct: with a Ctrl+C pending, Pulse does not launch the replacement run (pause/ask first)."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    launches = []

    def fake_run(argv, **kw):
        launches.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    cli._confirm_fix_did_its_job = lambda result: (True, "")
    cli._update_running = True                      # inside update(): the escalation's pipeline
    cli._sigint_handler(signal.SIGINT, None)        # the user presses Ctrl+C during it
    assert cli._stop_requested
    try:
        cli._restart_process()                      # the fix landed: ask_agent restarts
    except SystemExit:
        pass
    assert launches == [], "a new full training run was launched although Ctrl+C was pending"


def test_bug_ctrl_c_in_a_keras_epoch_longer_than_5s_kills_training_instead_of_pausing(make_cli,
                                                                                    monkeypatch):
    """In a Keras run update() comes only at epoch end (the tracer cannot see inside fit()).
    Ctrl+C mid-epoch prints 'Pausing after the current training step...', but the re-delivery
    timer fires after a fixed _SIGINT_GRACE_SECONDS (5 s) and, since no update() came yet,
    raises KeyboardInterrupt in the middle of the epoch -- any epoch (or PyTorch step) longer
    than 5 s can never be paused, and the run dies with a traceback instead. Correct: while
    steps are known to arrive every ~30 s, a Ctrl+C 5 s into that gap is not re-delivered."""
    clock = FakeTime(now=1000.0)
    monkeypatch.setattr(pc, "time", clock)
    timers, kills = [], []

    class FakeTimer:
        def __init__(self, interval, fn, args=()):
            timers.append((interval, fn, args))
            self.daemon = True

        def start(self):
            pass

    monkeypatch.setattr(pc.threading, "Timer", FakeTimer)
    monkeypatch.setattr(pc.signal, "pthread_kill", lambda tid, sig: kills.append(sig))
    cli = make_cli({"loss": 1.0}, tracked_vars=["loss"], var_states={"loss": "track"})
    # Under pytest-xdist (or a runner started in the background) SIGINT starts out ignored, so
    # PulseCLI leaves it alone and the re-delivery timer bails out: the test then passed for
    # the wrong reason. Install the handler as it is in a normal run (make_cli restores it).
    signal.signal(signal.SIGINT, cli._sigint_handler)
    for epoch in range(3):                   # epoch-end updates every 30 s
        clock.now += 30.0
        cli.watch_locals["loss"] = 1.0 / (epoch + 1)
        cli.update(step=(epoch + 1) * 100)
    clock.now += 5.0                         # Ctrl+C 5 s into the 4th epoch
    cli._sigint_handler(signal.SIGINT, None)
    assert timers, "no re-delivery timer was scheduled"
    interval, fn, args = timers[-1]
    if 5.0 + interval < 30.0:                # the timer fires before the next epoch end
        clock.now += interval
        fn(*args)
    assert kills == [], (f"Ctrl+C was re-delivered as KeyboardInterrupt {interval:g}s later, "
                         "mid-epoch, although the next epoch end (the pause point) was ~25 s away")


def test_bug_restart_drops_interpreter_options(make_cli, tmp_path, monkeypatch):
    """_restart_process builds argv as [python, script] + sys.argv[1:] (or -m module): the
    interpreter's own options -- `python -O train.py` (asserts off), `-W error`, `-X utf8`,
    `-u` -- are lost, so the fixed re-run is a different program (an assert the user disabled
    now fires and is taken for the fix failing). sys.orig_argv has them. Correct: the child is
    started with the same interpreter options."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    launched = []
    monkeypatch.setattr(pc.subprocess, "run",
                        lambda argv, **kw: (launched.append(list(argv)),
                                            subprocess.CompletedProcess(argv, 0, "", ""))[1])
    cli._confirm_fix_did_its_job = lambda result: (True, "")
    monkeypatch.setattr(sys, "argv", [cli.script_path, "--epochs", "3"])
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-O", "-W", "error", cli.script_path,
                                           "--epochs", "3"], raising=False)
    with pytest.raises(SystemExit):
        cli._restart_process()
    argv = launched[0]
    assert "-O" in argv and "-W" in argv, f"restarted with {argv[1:]}"


# =========================================================================== OK

def test_ok_gputrack_single_cuda_var_copies_once_per_interval(make_cli, monkeypatch):
    clock = FakeTime(now=5000.0)
    monkeypatch.setattr(pc, "time", clock)
    cli = make_cli({"w": FakeCudaTensor()})
    cli.discovered = {"w": (4, 4)}
    cli._cmd_gputrack("w", quiet=True)
    FakeCudaTensor.copies = 0
    for _ in range(5):
        clock.now += 1.0
        cli.update()
    assert FakeCudaTensor.copies == 1
    clock.now += cli.gpu_probe_interval + 1
    cli.update()
    assert FakeCudaTensor.copies == 2


def test_ok_nextcheck_plain_and_bold_verdict(make_cli, tmp_path):
    cli = _with_agent(make_cli(), tmp_path)
    cli._finish_periodic_checkin("VERDICT: **ok**\nNEXTCHECK: 1,500\nCHECKNOTE: none", None, "p", [])
    assert cli.checkin_interval_steps == 1500
    assert cli._checkin_note == ""


def test_ok_explicit_keras_steps_are_monotonic_across_fits(make_cli):
    cli = make_cli()
    for s in (10, 20, 30):
        cli.update(step=s)
    for s in (5, 15):                 # a second counter restarting lower: new segment
        cli.update(step=s)
    assert cli.step == 30 + 1 + 10


def test_ok_restart_child_output_streams_large_and_binary_output(make_cli, tmp_path):
    """_run_restart_child with a real child: 4 MB on stdout + invalid UTF-8 on stderr,
    no deadlock, everything kept."""
    cli = make_cli()
    code = ("import sys\nsys.stdout.write('x' * (4 * 1024 * 1024)); sys.stdout.flush()\n"
            "sys.stderr.buffer.write(b'\\xff\\xfe bad bytes\\n')\n")
    devnull = open(os.devnull, "w")
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = devnull
    try:
        t0 = _real_time.monotonic()
        result, streamed = cli._run_restart_child([sys.executable, "-c", code], dict(os.environ), None)
    finally:
        sys.stdout, sys.stderr = saved
        devnull.close()
    assert streamed and result.returncode == 0
    assert len(result.stdout) == 4 * 1024 * 1024
    assert "bad bytes" in result.stderr
    assert _real_time.monotonic() - t0 < 30


def test_ok_restart_child_that_leaves_a_grandchild_holding_the_pipe_returns(make_cli):
    """A child that forks a long-lived process holding stdout: _run_restart_child returns
    within ~the 5 s reader join instead of hanging until the grandchild exits."""
    cli = make_cli()
    code = ("import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])\n"
            "print('done')\n")
    devnull = open(os.devnull, "w")
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = devnull
    try:
        t0 = _real_time.monotonic()
        result, _ = cli._run_restart_child([sys.executable, "-c", code], dict(os.environ), None)
        took = _real_time.monotonic() - t0
    finally:
        sys.stdout, sys.stderr = saved
        devnull.close()
    assert result.returncode == 0 and "done" in result.stdout
    assert took < 15, f"took {took:.1f}s"
