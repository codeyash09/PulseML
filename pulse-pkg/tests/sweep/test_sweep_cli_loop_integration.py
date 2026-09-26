"""Bug sweep: PulseCLI's training-loop integration (src/pulse/pulse_cli.py).

Covers update()/step tracking, matrix probing, the Keras hook, crash handling,
restart + checkpoint-resume, the periodic check-in, the start-of-run prime, the
end-of-run review, the retry ticker and config/provider hand-off across a restart.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_*  are regression coverage for behaviour verified to work.

No real model calls: every agent call is stubbed, litellm.completion is mocked
where the real _call_model path is exercised, and the retry ticker is stubbed
wherever a real one could outlive the test.
"""
import atexit
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

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(pc.__file__)))


# --------------------------------------------------------------------------- helpers

class FakeTime:
    """Stands in for pulse_cli's `time` module: a controllable monotonic clock and
    an optional replacement sleep; everything else is the real module."""

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
    """Snapshot/restore os.environ and the PROVIDERS table (restart code mutates both)."""
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
    """A real PulseCLI (constructor and all), with the process-global side effects of
    the constructor (SIGINT handler, atexit review) undone afterwards."""
    old_sigint = signal.getsignal(signal.SIGINT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pc.cloud, "save_cached_profile", lambda **k: None, raising=False)
    created = []

    def factory(watch_locals=None, **attrs):
        cli = PulseCLI(watch_locals=watch_locals if watch_locals is not None else {},
                       pdf_dir=str(tmp_path / "pdf"))
        atexit.unregister(cli._end_of_run_review)
        cli.continuous = True
        cli.auto_intervene = False
        cli._start_primed = True
        cli._start_primed_with_agent = True
        cli._ensure_retry_ticker = lambda: None      # never let a real retry thread outlive a test
        for key, value in attrs.items():
            setattr(cli, key, value)
        created.append(cli)
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
    """Enough of keras.Model for the fit hook, the tracker callback, and checkpoint/resume."""

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
    """A fake `keras` module in sys.modules, with the hook globals reset around the test."""
    module = types.ModuleType("keras")
    model_cls = type("Model", (FakeKerasModel,), {})
    module.Model = model_cls
    module.callbacks = types.SimpleNamespace(Callback=FakeCallback)
    monkeypatch.setitem(sys.modules, "keras", module)
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr(pc, "_PULSE_KERAS_HOOK_INSTALLED", False)
    monkeypatch.setattr(pc, "_PULSE_KERAS_TRACKER_CLS", None)
    return module


# =========================================================================== BUGS

# ---- check-in worker thread vs the terminal / print lock ---------------------------

def test_bug_checkin_terminal_confirmation_prompts_on_worker_and_freezes_training_prints(
        make_cli, tmp_path, monkeypatch):
    """A check-in runs on a worker thread while training continues. If it asks for a
    TERMINAL command the classifier flags (any '>' counts, e.g. `python -c "print(2>1)"`,
    or `2>&1`), _run_terminal -> _confirm_terminal_command calls input() FROM THE WORKER.
    safe_input holds _io_lock for as long as it waits, and every print() in the process
    (builtins.print = safe_print) needs that lock -- so the training script's next print
    blocks until somebody answers a y/n nobody is watching (unattended / restarted runs).
    Correct: a check-in never prompts; flagged commands are refused (or run) without input(),
    and the training thread's print() is never blocked by it."""
    cli = _with_agent(make_cli(), tmp_path)
    asked, release = threading.Event(), threading.Event()

    def blocking_input(prompt=""):
        asked.set()
        release.wait(10)
        return "n"

    monkeypatch.setattr(pc, "_original_input", blocking_input)
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    monkeypatch.setattr(pc._ui, "enabled", lambda: False)
    worker = threading.Thread(target=cli._checkin_service_tools,
                              args=('TERMINAL: python -c "print(2>1)"',), daemon=True)
    worker.start()
    asked.wait(3)
    printed = threading.Event()
    trainer = threading.Thread(target=lambda: (print("epoch 3 loss 0.41"), printed.set()), daemon=True)
    trainer.start()
    print_finished = printed.wait(2)
    release.set()
    worker.join(15)
    trainer.join(5)
    assert not asked.is_set(), "the background check-in prompted for input() on its worker thread"
    assert print_finished, "the training thread's print() blocked behind the check-in's pending prompt"


# ---- matrix probing cadence ----------------------------------------------------------

def test_bug_track_matrix_probe_is_gated_by_ten_minute_gpu_interval(make_cli, monkeypatch):
    """update(): probe_track = ... or (gpu_ready and now-_last_matrix_probe >= matrix_probe_interval),
    where gpu_ready needs gpu_probe_interval (600 s) since _last_gpu_probe -- and
    _last_gpu_probe is reset on every matrix probe. So a 'track' matrix (documented as probed
    every matrix_probe_interval = 1 s) is re-read only once per 10 minutes: a weight matrix that
    turns NaN is invisible to the display and to the detector's tensor_stats for ~10 min.
    Correct: 'track' matrices are re-probed every matrix_probe_interval seconds."""
    clock = FakeTime(now=1000.0)
    monkeypatch.setattr(pc, "time", clock)
    watch = {"W": np.ones((4, 4))}
    cli = make_cli(watch, tracked_vars=["W"], var_states={"W": "track"})
    cli.update()
    clock.now += 2
    cli.update()                                       # second probe resets _last_gpu_probe
    watch["W"] = np.full((4, 4), np.nan)
    for _ in range(10):
        clock.now += 3                                 # 30 s later, 1 s interval
        cli.update()
    stats = cli._matrix_cache["W"]["stats"]
    assert stats.get("nan"), f"W went NaN 30 s ago but the cached stats are still {stats}"


# ---- step tracking -------------------------------------------------------------------

def test_bug_explicit_step_that_resets_each_epoch_stops_all_checkins(make_cli, tmp_path, monkeypatch):
    """update(step=...) assigns self.step = step. A per-epoch batch index (the docstring's own
    example: "a Keras callback's batch/epoch index") makes self.step go 0..29, 0..29, ...
    _maybe_periodic_checkin compares self.step - _last_checkin_step, which after the first
    check-in (at 20) never reaches the interval again. 120 real steps at a 20-step interval
    get 1 check-in instead of ~6. Correct: a backwards explicit step does not freeze scheduling
    (treat it as a new segment / keep a monotonic global count)."""
    monkeypatch.setenv("PULSE_ASYNC_MODEL_CALLS", "0")
    cli = _with_agent(make_cli(), tmp_path)
    cli.checkin_interval_steps = 20
    cli._build_agent_context = lambda include_code=False: "snapshot"
    started = []
    cli._run_checkin = lambda prompt: (started.append(cli.step), ("VERDICT: ok", []))[1]
    for _epoch in range(4):
        for batch in range(30):
            cli.update(step=batch)
    assert len(started) >= 4, f"only {len(started)} check-in(s) over 120 steps at a 20-step interval"


def test_bug_epoch_only_loop_counter_counts_epochs_not_steps(make_cli):
    """The most common PyTorch loop -- `for epoch in ...: for x, y in loader:` -- has only
    `epoch` among _LOOP_VAR_CANDIDATES. _detect_loop_step_delta then returns a 'confident' 0 on
    every tick inside an epoch, which suppresses the loss-changed fallback, so self.step counts
    EPOCHS. A 50-epoch run never reaches the 500-step check-in (or the 20-step minimum over 15
    epochs) -- the very failure the Keras batch counting was added to fix.
    Correct: when the only counter is coarse (epoch) and the loss moves every tick, steps
    advance per tick (≈100 here)."""
    watch = {"epoch": 0, "loss": 1.0}
    cli = make_cli(watch, tracked_vars=["loss"], var_states={"loss": "track"})
    n = 0
    for epoch in range(4):
        watch["epoch"] = epoch
        for _batch in range(25):
            n += 1
            watch["loss"] = 1.0 / n
            cli.update()
    assert cli.step >= 90, f"100 loss-changing training steps were counted as {cli.step}"


def test_bug_inner_loop_counter_wrap_drops_a_step_each_epoch(make_cli):
    """_detect_loop_step_delta: when `i` wraps (4 -> 0 at an epoch boundary) it returns 0, but a
    real iteration happened (batch 0 of the next epoch). With 5 batches/epoch that loses 20% of
    all steps. Correct: 15 iterations -> step 15."""
    watch = {"i": 0}
    cli = make_cli(watch)
    for _epoch in range(3):
        for i in range(5):
            watch["i"] = i
            cli.update()
    assert cli.step == 15, f"15 loop iterations counted as {cli.step} steps"


# ---- end-of-run review ----------------------------------------------------------------

def test_bug_end_of_run_review_runs_in_restarted_child_and_starts_a_second_fix_chain(
        make_cli, tmp_path, monkeypatch):
    """A fix-restarted child (PULSE_RESTART_CHILD=1) that crashes before step 1: pulse.py's
    excepthook returns early for children WITHOUT calling handle_crash (so _crash_seen stays
    False), then atexit -> _end_of_run_review sees step == 0 and escalates "finished without
    training, nothing crashed visibly" -- the child runs its own agent fix + restart, while the
    parent is about to feed the same failure to the agent too. That is the nested chain
    _RESTART_CHILD_ENV exists to prevent. Correct: a restart child never escalates at exit."""
    cli = _with_agent(make_cli(), tmp_path)
    cli.auto_intervene = True
    monkeypatch.setenv(pc._RESTART_CHILD_ENV, "1")
    escalated = []
    cli._escalate_training_problem = escalated.append
    cli._end_of_run_review()
    assert escalated == []


def test_bug_end_of_run_review_escalates_after_ctrl_c(make_cli, tmp_path, monkeypatch):
    """The user presses Ctrl+C twice (e.g. during data loading, before step 1). pulse.py's
    excepthook returns early for KeyboardInterrupt without marking the crash, so at exit
    _end_of_run_review treats it as "an error may have been caught" and hands it to the agent,
    which may rewrite the code and restart the script the user just killed.
    Correct: a KeyboardInterrupt exit is never escalated."""
    pulse_mod = pytest.importorskip("pulse.pulse")
    cli = _with_agent(make_cli(), tmp_path)
    cli.auto_intervene = True
    escalated = []
    cli._escalate_training_problem = escalated.append
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(pulse_mod, "_stop_cli_tracing", lambda: None, raising=False)
    pulse_mod._install_cli_excepthook(cli)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    cli._end_of_run_review()
    assert escalated == [], "a Ctrl+C exit was escalated to the agent as a silent failure"


def test_bug_checkin_problem_that_lands_after_training_is_dropped(make_cli, tmp_path):
    """A check-in's answer is applied only by the next update(). One that finishes after the
    last training step (or during a long evaluation with no update() calls) is lost: at exit
    _end_of_run_review returns as soon as step > 0 and never looks at _checkin_call. A
    'problem' verdict (e.g. validation set == training set) silently disappears.
    Correct: at exit, a finished (or briefly-awaited) check-in is applied, so its problem is
    escalated/reported."""
    cli = _with_agent(make_cli(), tmp_path)
    cli.auto_intervene = True
    cli.step = 800
    cli._checkin_call = types.SimpleNamespace(
        done=True, error=None, prompt="the run",
        result=("VERDICT: problem\nPROBLEM: validation_data is the training set (train.py:12)\n"
                "NEXTCHECK: 500\nCHECKNOTE: none", []))
    escalated = []
    cli._escalate_training_problem = escalated.append
    cli._end_of_run_review()
    assert escalated and "validation_data" in escalated[0]


# ---- restart ----------------------------------------------------------------------------

def _meta_checkpoint(tmp_path):
    meta = tmp_path / "fix.json"
    meta.write_text(json.dumps({"weights": str(tmp_path / "w.h5"), "epoch": 3, "finite": True}))
    return str(meta)


def test_bug_restart_retry_keeps_resuming_after_second_fix_asks_for_fresh_start(
        make_cli, tmp_path, monkeypatch):
    """_restart_process builds child_env once, before the retry loop. When a relaunch crashes,
    the failure goes back to the agent, whose new fix may say "resume": false (an initializer /
    data / label fix -- the saved weights carry the bug); _apply_code_fix sets
    _resume_after_fix = False, but the next attempt still launches with PULSE_RESUME_CHECKPOINT
    and resumes the bug-carrying weights. Correct: the env is rebuilt after each in-loop fix."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    cli._fix_checkpoint = _meta_checkpoint(tmp_path)
    cli._resume_after_fix = True
    envs = []
    outcomes = iter([1, 0])

    def fake_run(argv, **kw):
        envs.append(dict(kw.get("env") or {}))
        return subprocess.CompletedProcess(argv, next(outcomes), "out", "Traceback: boom")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)

    def agent_fixes_with_fresh_start(question, include_code=False, _depth=0):
        cli._resume_after_fix = False               # what _apply_code_fix does for "resume": false
        cli._fix_applied_this_turn = True

    cli._ask_agent_impl = agent_fixes_with_fresh_start
    cli._confirm_fix_did_its_job = lambda result: (True, "")
    with pytest.raises(SystemExit):
        cli._restart_process()
    assert pc._RESUME_ENV in envs[0]
    assert pc._RESUME_ENV not in envs[1], "second attempt resumed although the new fix asked for a fresh start"


def test_bug_restart_reruns_identical_script_after_clean_exit_with_no_new_fix(
        make_cli, tmp_path, monkeypatch):
    """A relaunch that exits 0 but the post-fix check says 'did not do its job': the failure is
    fed to the agent; if the agent changes nothing, the loop relaunches the byte-identical
    script -- a full training run -- up to 5 times, then rolls the fix back. Re-running an
    unchanged program that already finished cleanly cannot change the verdict.
    Correct: no relaunch unless a new fix was actually written."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    runs = []

    def fake_run(argv, **kw):
        runs.append(argv)
        return subprocess.CompletedProcess(argv, 0, "val_acc 0.50", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    cli._confirm_fix_did_its_job = lambda result: (False, "accuracy still at chance")
    cli._fix_applied_this_turn = False

    def agent_changes_nothing(question, include_code=False, _depth=0):
        cli._fix_applied_this_turn = False

    cli._ask_agent_impl = agent_changes_nothing
    cli._auto_rollback_after_failed_restarts = lambda commit: False
    cli._restart_process()
    assert len(runs) == 1, f"the unchanged script was re-run {len(runs)} times"


def test_bug_restart_child_runs_in_whatever_directory_the_script_chdir_ed_into(
        make_cli, tmp_path, monkeypatch):
    """_restart_process launches the child with no cwd=, so it inherits the parent's CURRENT
    directory. A script that did os.chdir("outputs/run1") (common before saving artifacts) is
    restarted from there, and every relative path in it ("data/train.csv") breaks -- the fixed
    run crashes for a reason that has nothing to do with the fix. Correct: the child runs from
    the directory the original process was launched in."""
    launch_dir = os.getcwd()
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    moved = tmp_path / "outputs" / "run1"
    moved.mkdir(parents=True)
    monkeypatch.chdir(moved)
    seen = {}

    def fake_run(argv, **kw):
        seen["cwd"] = kw.get("cwd") or os.getcwd()
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    cli._confirm_fix_did_its_job = lambda result: (True, "")
    with pytest.raises(SystemExit):
        cli._restart_process()
    assert os.path.realpath(seen["cwd"]) == os.path.realpath(launch_dir)


def test_bug_restart_drops_python_dash_m_invocation(make_cli, tmp_path, monkeypatch):
    """`python -m mypkg.train --lr 1`: sys.argv[0] is the file path, so the restart runs
    `python /abs/mypkg/train.py --lr 1` -- relative imports (`from .model import Net`) fail with
    "attempted relative import with no known parent package", and every restart attempt
    crashes. Correct: a script started with -m is restarted with -m <module>."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    monkeypatch.setattr(sys, "argv", [cli.script_path, "--lr", "1"])
    monkeypatch.setattr(sys.modules["__main__"], "__spec__",
                        types.SimpleNamespace(name="mypkg.train", parent="mypkg"), raising=False)
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    cli._confirm_fix_did_its_job = lambda result: (True, "")
    with pytest.raises(SystemExit):
        cli._restart_process()
    assert "-m" in seen["argv"] and "mypkg.train" in seen["argv"], seen["argv"]


def test_bug_failed_restart_leaves_unattended_flags_in_parent_environment(make_cli, tmp_path, monkeypatch):
    """_restart_process writes PULSE_AUTO_RESTART=1 (and PULSE_AUTO_PROVIDER, ...) into the
    PARENT's os.environ, not just the child env. When every launch fails and the parent keeps
    running, anything it starts afterwards -- DataLoader spawn workers re-importing the script,
    a TERMINAL: `python train.py` check, the user's own subprocess -- inherits
    PULSE_AUTO_RESTART=1 and silently switches to unattended mode. The code's own comment says
    only the child should get the marker. Correct: the parent's environment is unchanged."""
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    monkeypatch.setattr(pc, "time", FakeTime())

    def cannot_launch(argv, **kw):
        raise OSError("exec format error")

    monkeypatch.setattr(pc.subprocess, "run", cannot_launch)
    cli._queue_restart_retry = lambda: None
    cli._restart_process()
    assert "PULSE_AUTO_RESTART" not in os.environ


def test_bug_retry_ticker_calls_restart_again_every_tick_forever(make_cli, monkeypatch):
    """_ensure_retry_ticker never clears _pending_restart_retry. Once next_attempt has passed,
    every 15 s tick calls _restart_process() again -- if that returns without re-queueing
    (depth cap reached, script missing, ...) it is retried forever; if the relaunch runs, a new
    full training run starts every tick. Correct: a queued restart is attempted once per queue
    entry."""
    real_sleep = _real_time.sleep
    monkeypatch.setattr(pc, "time", FakeTime(sleep=lambda s: real_sleep(0.01)))
    cli = make_cli()
    del cli._ensure_retry_ticker                     # use the real ticker for this test
    calls = []
    cli._restart_process = lambda: calls.append(1)
    cli._pending_restart_retry = {"next_attempt": 0.0, "backoff": 60.0}
    cli._ensure_retry_ticker()
    real_sleep(0.6)
    cli._pending_restart_retry = None                # stop the daemon thread's work
    assert len(calls) == 1, f"one queued restart was attempted {len(calls)} times"


def test_bug_restart_from_retry_ticker_thread_cannot_end_the_old_process():
    """A queued restart (and an agent retry that applies a fix) runs _restart_process on the
    retry-ticker daemon thread. On success it calls sys.exit(0) -- which on a non-main thread only
    ends that thread. The old process keeps training the old code (it was training concurrently
    for the whole child run, too). Correct: a successful restart ends the old process."""
    code = f"""
import sys, time, threading
sys.path.insert(0, {SRC_DIR!r})
import pulse.pulse_cli as pc
cli = pc.PulseCLI.__new__(pc.PulseCLI)
cli._retry_ticker_started = False
cli._retry_ticker_lock = threading.RLock()
cli._pending_agent_retry = None
cli._pending_restart_retry = {{"next_attempt": 0, "backoff": 60}}
def restart_that_succeeded():
    sys.exit(0)              # what _restart_process does once the fixed run finished
cli._restart_process = restart_that_succeeded
real_sleep = time.sleep
class T:
    def __getattr__(self, n): return getattr(time, n)
    def sleep(self, s): real_sleep(0.01)
pc.time = T()
cli._ensure_retry_ticker()
real_sleep(1.5)
print("OLD PROCESS STILL TRAINING", flush=True)
"""
    env = dict(os.environ, PULSE_LOGGING="0")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert "OLD PROCESS STILL TRAINING" not in out.stdout, out.stdout + out.stderr


# ---- config / provider hand-off across a restart ---------------------------------------

def test_bug_config_custom_agent_api_key_lost_after_restart(make_cli, tmp_path, monkeypatch):
    """pulse_config.json {"agent": "anthropic/claude-x", "api_key": "sk-..."} (no api_key_env):
    the custom branch keeps the key only in self.agent_key -- it is not put in the environment.
    After a fix restart, _agent_setup's PULSE_AUTO_PROVIDER path re-registers the custom provider
    with env_key None and sets agent_key = "local" (sent as api_key=None), ignoring the config
    key -- so every agent call in the restarted run fails authentication.
    Correct: the restarted run uses the same key."""
    parent = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(parent, monkeypatch)
    parent.agent_provider = parent.agent_key = None
    parent.non_interactive = True
    parent.config = {"agent": "anthropic/claude-x", "api_key": "sk-from-config"}
    assert parent._select_agent_provider_and_key(initial=True)
    assert parent.agent_key == "sk-from-config"
    child_env = {}

    def fake_run(argv, **kw):
        child_env.update(kw.get("env") or {})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    parent._confirm_fix_did_its_job = lambda result: (True, "")
    with pytest.raises(SystemExit):
        parent._restart_process()

    os.environ.clear()
    os.environ.update(child_env)
    pc.PROVIDERS.pop("Custom: anthropic/claude-x", None)   # a fresh process has no such entry
    child = make_cli()
    child.config = {"agent": "anthropic/claude-x", "api_key": "sk-from-config"}
    child.non_interactive = True
    child._agent_setup()
    assert child.agent_key == "sk-from-config", f"restarted run's key: {child.agent_key!r}"


def test_bug_config_null_value_disables_autofix(make_cli, tmp_path, monkeypatch):
    """_config_bool: a JSON null ({"autofix": null}, a common 'unset') becomes str(None) =
    'none' -> False, silently turning auto-fix OFF instead of using the default (True).
    Correct: null means 'use the default'."""
    cfg = tmp_path / "pulse_config.json"
    cfg.write_text(json.dumps({"autofix": None}))
    cli = make_cli()
    cli.set_code_text("x = 1\n", script_path=str(tmp_path / "train.py"))
    assert cli.config_path == str(cfg)
    assert cli.auto_intervene is True


def test_bug_explicit_pulse_config_path_missing_is_silent(make_cli, tmp_path, monkeypatch, capsys):
    """PULSE_CONFIG=/path/typo.json that does not exist is skipped without a word; Pulse falls
    back to interactive setup (or to some other pulse_config.json it finds), so an unattended job
    blocks on a prompt. Correct: an explicitly named config that is missing is reported."""
    monkeypatch.setenv("PULSE_CONFIG", str(tmp_path / "typo_config.json"))
    cli = make_cli()
    capsys.readouterr()
    cli.set_code_text("x = 1\n", script_path=str(tmp_path / "train.py"))
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "typo_config.json" in out


# ---- check-in parsing --------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "VERDICT: **ok**\nNEXTCHECK: 500\nCHECKNOTE: none",
    "**VERDICT:** ok\nNEXTCHECK: 500\nCHECKNOTE: none",
    "VERDICT: `ok`\nNEXTCHECK: 500",
])
def test_bug_markdown_formatted_ok_verdict_is_not_ok(answer):
    """_checkin_verdict: models routinely bold/backtick the value. 'VERDICT: **ok**' has first
    word '**ok**' -> treated as a PROBLEM -> a full auto-intervention (fix pipeline, code edit,
    restart) on a healthy run; '**VERDICT:** ok' does not match _VERDICT_RE at all (inconclusive,
    and an extra model call). Correct: markdown decoration is ignored -> ok."""
    assert PulseCLI._checkin_verdict(answer)[0] is False


@pytest.mark.parametrize("text, expected", [("NEXTCHECK: 1,000", 1000), ("NEXTCHECK: 2_000", 2000)])
def test_bug_nextcheck_with_thousands_separator(text, expected):
    """_NEXTCHECK_RE captures only leading digits, so 'NEXTCHECK: 1,000' parses as 1 and is
    clamped to 20 -- check-ins 50x more often than asked (each is several paid model calls).
    Correct: 1,000 -> 1000."""
    assert PulseCLI._parse_nextcheck_steps(text) == expected


def test_bug_checkin_stat_tools_fail_on_history_with_unreadable_step(make_cli):
    """update() appends None to a scalar history whenever the variable is unreadable for a step.
    _scalar_history returns the list with the None in it, and OUTLIER/CORR/HISTOGRAM do arithmetic
    on it -> TypeError; in a check-in that surfaces as 'OUTLIER loss: failed (TypeError ...)', so one
    unreadable step makes the statistics tools useless for the rest of the run.
    Correct: None entries are skipped."""
    cli = make_cli()
    cli.scalar_histories = {"loss": [1.0, None, 0.9, 0.85, 0.8, 0.82, 40.0, 0.78, 0.77, 0.76, 0.75, 0.74]}
    results, _ = cli._checkin_service_tools("OUTLIER: loss")
    assert "failed" not in results and "40" in results, results


# ---- /sensitivity --------------------------------------------------------------------------

def test_bug_sensitivity_oscillation_inf_raises_into_training_loop(make_cli):
    """_cmd_sensitivity: 'oscillation inf' -> int(float('inf')) raises OverflowError; only
    ValueError is caught, so the typed command (or an agent SENSITIVITY: directive) raises out of
    the interactive prompt / update() into the user's training loop.
    Correct: a usage message, no exception."""
    cli = make_cli()
    msg = cli._cmd_sensitivity("oscillation inf", quiet=True)
    assert msg and "Usage" in msg


# ---- start-of-run prime --------------------------------------------------------------------

def test_bug_mllint_finding_runs_the_agent_fix_pipeline_twice_at_start(make_cli, tmp_path, monkeypatch):
    """With an agent configured before step 1 (the normal case, and every restarted run),
    update() calls _prime_at_start -- which runs MLLINT and, on a finding, ask_agent() -- and then
    _prime_with_agent_if_needed, which is guarded only by _start_primed_with_agent, re-runs MLLINT
    and calls ask_agent() AGAIN for the same finding (plus a second deterministic auto-fix pass).
    Two full paid fix pipelines for one finding. Correct: one."""
    cli = _with_agent(make_cli(), tmp_path)
    cli._start_primed = False
    cli._start_primed_with_agent = False
    monkeypatch.setattr(pc, "_mllint_scan", lambda trees: [("train.py", 3, "softmax output into a from_logits loss")])
    cli._iter_ast_trees = lambda: []
    cli._mllint_auto_fix = lambda findings: False
    cli._start_start_prime = lambda deferred: None
    asks = []
    cli.ask_agent = lambda question, include_code=False, **kw: asks.append(question)
    cli._prime_at_start()
    cli._prime_with_agent_if_needed()
    assert len(asks) == 1, f"the same MLLINT finding was sent through the fix pipeline {len(asks)} times"


# ---- crash handling -------------------------------------------------------------------------

def test_bug_offer_known_fix_reports_success_when_nothing_was_reapplied(make_cli, tmp_path):
    """offer_known_fix returns True ('already handled, skip the agent') after _apply_code_fix
    whether or not anything matched. A known fix whose snippet is no longer in the file (the code
    changed since, or it was already applied) applies nothing -- and the crash is then never
    diagnosed at all. Correct: return False when nothing was re-applied, so the agent is asked."""
    cli = _with_agent(make_cli(), tmp_path, code="a = 1\nprint(a)\n")
    cli.auto_intervene = True
    sig = "deadbeef0001"
    cli._known_fixes = {sig: {"old": ["b = 2"], "new": ["b = 3"], "files": [None], "explanation": "e"}}
    cli._sync_agent_turn = lambda **kw: None
    restarts = []
    cli._restart_process = lambda: restarts.append(1)
    cli._fix_applied_this_turn = False
    assert cli.offer_known_fix(sig) is False


def test_bug_offer_known_fix_restarts_on_stale_fix_flag(make_cli, tmp_path):
    """_fix_applied_this_turn is only reset inside _ask_agent_impl. If an earlier fix this run
    left it True (e.g. its restart hit the depth cap or gave up), a later known-fix re-apply that
    changes nothing still triggers _restart_process(). Correct: restart only when this re-apply
    wrote something."""
    cli = _with_agent(make_cli(), tmp_path, code="a = 1\nprint(a)\n")
    cli.auto_intervene = True
    sig = "deadbeef0002"
    cli._known_fixes = {sig: {"old": ["b = 2"], "new": ["b = 3"], "files": [None], "explanation": "e"}}
    cli._sync_agent_turn = lambda **kw: None
    restarts = []
    cli._restart_process = lambda: restarts.append(1)
    cli._fix_applied_this_turn = True              # left over from an earlier fix this run
    cli.offer_known_fix(sig)
    assert restarts == []


def test_bug_traceback_signature_changes_with_object_address(make_cli):
    """_traceback_signature hashes the last 4 lines verbatim. Exception messages routinely embed
    per-run values (object reprs '<Foo object at 0x7f..>'), so the same bug gets a new signature
    every run: dedup, _known_fixes and the "already fixed this" warning never match.
    Correct: volatile addresses do not change the signature."""
    tb = ('Traceback (most recent call last):\n  File "/w/train.py", line 12, in <module>\n'
          '    out = model(batch)\nTypeError: unsupported operand for <Batch object at {addr}>\n')
    a = PulseCLI._traceback_signature(tb.format(addr="0x7f3a1c2b9d30"))
    b = PulseCLI._traceback_signature(tb.format(addr="0x7fe01122aa10"))
    assert a == b


def test_bug_non_transient_agent_failure_is_queued_for_endless_retry(make_cli, tmp_path, monkeypatch):
    """_ask_agent_impl sets _last_call_failed_transiently = True for EVERY AgentRequestFailed,
    including authentication errors / bad model names / context too long, which _call_model
    deliberately does not retry. ask_agent then queues the same question for retry every
    60..600 s for the rest of the run (and the crash hook sleeps 5+20+60 s re-asking).
    Correct: only transient failures are queued."""
    import litellm
    cli = _with_agent(make_cli(), tmp_path)

    def auth_fail(**kwargs):
        raise litellm.AuthenticationError(message="invalid x-api-key", llm_provider="deepseek",
                                          model="deepseek-chat")

    monkeypatch.setattr(litellm, "completion", auth_fail)
    cli.ask_agent("why is my loss flat?", include_code=False)
    assert cli._pending_agent_retry is None, "an authentication failure was queued for retry"


# ---- Keras hook ------------------------------------------------------------------------------

def test_bug_keras_fit_with_positional_callbacks_raises_typeerror(make_cli, fake_keras, monkeypatch):
    """pulse_fit always sets kwargs['callbacks']. `model.fit(x, y, 32, 5, 0, [cb])` (callbacks is
    fit's 6th positional parameter) then raises "got multiple values for argument 'callbacks'" --
    Pulse breaks a valid fit() call. Correct: positional callbacks are merged."""
    cli = make_cli()
    monkeypatch.setattr(pc, "_PULSE_ACTIVE_INSTANCE", cli)
    pc._install_keras_fit_hook()
    mine = object()
    cbs = fake_keras.Model().fit([[0.0] * 3], [0.0], 32, 1, 0, [mine])
    assert mine in cbs and any(isinstance(c, pc._PULSE_KERAS_TRACKER_CLS) for c in cbs)


def test_bug_checkpoint_from_second_fit_is_resumed_into_the_first_fit(make_cli, fake_keras, tmp_path, monkeypatch):
    """_resume_from_fix_checkpoint resumes into the FIRST fit() of the restarted script and the
    checkpoint records no fit index. Script: pretrain fit (3 epochs), then fine-tune fit; the
    problem is caught in the fine-tune at epoch 1 -> after the restart the PRETRAIN fit gets the
    fine-tune weights and starts at epoch 2 (skipping pretraining), and the fine-tune fit, where
    the checkpoint belongs, starts from scratch. Correct: the checkpoint is applied to the fit()
    it was taken in (here: the first fit runs all its epochs untouched)."""
    cli = make_cli()
    cli.script_path = str(tmp_path / "train.py")
    monkeypatch.setattr(pc, "_PULSE_ACTIVE_INSTANCE", cli)
    pc._install_keras_fit_hook()
    Model = fake_keras.Model
    pre, fine = Model(0.1), Model(0.1)
    pre.fit([[0.0] * 3], [0.0], epochs=3)
    fine.fit([[0.0] * 3], [0.0], epochs=2)              # ...problem detected here, epoch 1 of 5
    cli._keras_epochs = 5
    cli._save_fix_checkpoint()
    assert cli._fix_checkpoint
    monkeypatch.setenv(pc._RESUME_ENV, cli._fix_checkpoint)
    pre2, fine2 = Model(0.0), Model(0.0)                 # the restarted script
    pre2.fit([[0.0] * 3], [0.0], epochs=3)
    fine2.fit([[0.0] * 3], [0.0], epochs=5)
    assert pre2.ran_epochs == [0, 1, 2], f"pretraining fit was resumed: ran {pre2.ran_epochs}"


def test_bug_keras_run_with_autofix_off_gets_no_update_during_fit(make_cli, fake_keras, monkeypatch):
    """PulseKerasTracker.on_epoch_end calls pulse.update() only `if auto_intervene`. The tracer
    cannot call update() while fit() is running (no user-frame lines execute), so with /autofix off
    a Keras run gets no step count, no periodic check-in (which is designed to run with autofix off
    -- it 'flags it for you to look at') and no Ctrl+C pause for the whole fit.
    Correct: update() still runs at epoch end; auto_intervene only gates the escalation."""
    cli = make_cli()
    cli.auto_intervene = False
    monkeypatch.setattr(pc, "_PULSE_ACTIVE_INSTANCE", cli)
    pc._install_keras_fit_hook()
    steps = []
    cli.update = lambda step=None, **kw: steps.append(step)
    fake_keras.Model().fit([[0.0] * 3], [0.0], epochs=3)
    assert steps, "update() was never called during a 3-epoch fit with autofix off"


# ---- escalation ------------------------------------------------------------------------------

def test_bug_escalation_without_agent_still_writes_weight_checkpoint(make_cli, tmp_path):
    """_escalate_training_problem saves a full weights checkpoint into .pulse_checkpoints/ before
    checking whether an agent exists. With no agent nothing can be fixed or restarted, yet every
    new detection writes another model-sized file that is never cleaned up (a disk leak on long
    runs with big models). Correct: no checkpoint when no fix can follow."""
    cli = make_cli()
    cli.script_path = str(tmp_path / "train.py")
    cli._keras_model, cli._keras_epoch, cli._keras_epochs = FakeKerasModel(0.5), 2, 10
    cli._escalate_training_problem("loss went NaN at epoch 3")
    folder = tmp_path / ".pulse_checkpoints"
    assert not folder.exists() or not list(folder.iterdir())


# =========================================================================== OK (coverage)

def test_ok_parse_nextcheck_clamps_and_rejects():
    assert PulseCLI._parse_nextcheck_steps("NEXTCHECK: 5") == PulseCLI._CHECKIN_MIN_STEPS
    assert PulseCLI._parse_nextcheck_steps("NEXTCHECK: 999999") == PulseCLI._CHECKIN_MAX_STEPS
    assert PulseCLI._parse_nextcheck_steps("VERDICT: ok\nNEXTCHECK: 300") == 300
    assert PulseCLI._parse_nextcheck_steps("NEXTCHECK: 0") is None
    assert PulseCLI._parse_nextcheck_steps("no line") is None
    assert PulseCLI._parse_nextcheck_steps(None) is None


def test_ok_loop_counter_advances_step_by_its_delta(make_cli):
    watch = {"global_step": 0}
    cli = make_cli(watch)
    cli.update()                                 # baseline (fallback +1: no loss, no counter yet)
    base = cli.step
    for gs in (1, 2, 5):
        watch["global_step"] = gs
        cli.update()
    assert cli.step - base == 5
    watch["global_step"] = 5000                  # implausible jump -> not trusted
    cli._detect_loop_step_delta()
    assert cli._loop_var_last_seen["global_step"] == 5000


def test_ok_explicit_step_and_time_per_step_ema(make_cli, monkeypatch):
    clock = FakeTime(now=100.0)
    monkeypatch.setattr(pc, "time", clock)
    cli = make_cli()
    cli._last_step_time = 100.0
    clock.now = 101.0
    cli._record_step_advance(10)
    assert cli._time_per_step_ema == pytest.approx(0.1)
    clock.now = 102.0
    cli._record_step_advance(1)
    assert cli._time_per_step_ema == pytest.approx(0.2 * 1.0 + 0.8 * 0.1)
    assert cli._format_time_per_step() == "~280ms"
    cli.update(step=42)
    assert cli.step == 42


def test_ok_loss_change_fallback_counts_steps(make_cli):
    watch = {"loss": 1.0}
    cli = make_cli(watch, tracked_vars=["loss"], var_states={"loss": "track"})
    for k in range(1, 11):
        watch["loss"] = 1.0 / k
        cli.update()
    cli.update()                                  # unchanged loss: no step
    assert cli.step == 10
    assert cli.scalar_histories["loss"][-1] == pytest.approx(0.1)


def test_ok_finished_background_checkin_applied_on_next_update(make_cli, tmp_path):
    cli = _with_agent(make_cli(), tmp_path)
    cli._checkin_call = types.SimpleNamespace(
        done=True, error=None, prompt="p",
        result=("VERDICT: ok\nNEXTCHECK: 300\nCHECKNOTE: watch val_loss after epoch 5", []))
    cli.update()
    assert cli._checkin_call is None
    assert cli.checkin_interval_steps == 300
    assert cli._checkin_note == "watch val_loss after epoch 5"


def test_ok_checkin_problem_escalates_once_per_distinct_problem(make_cli, tmp_path):
    cli = _with_agent(make_cli(), tmp_path)
    cli.auto_intervene = True
    asked = []
    cli.ask_agent = lambda q, include_code=False, **kw: asked.append(q)
    cli._save_fix_checkpoint = lambda: None
    answer = "VERDICT: problem\nPROBLEM: scaler fit on the full dataset (train.py:9)\nNEXTCHECK: 100"
    cli._finish_periodic_checkin(answer, None, "prompt", [])
    cli._finish_periodic_checkin(answer, None, "prompt", [])
    assert len(asked) == 1 and "scaler fit" in asked[0]
    assert cli.checkin_interval_steps == 100


def test_ok_checkin_request_failure_is_skipped_not_raised(make_cli):
    cli = make_cli()
    cli._finish_periodic_checkin(None, pc.AgentRequestFailed("rate limited"), "p", [])
    with pytest.raises(KeyError):
        cli._finish_periodic_checkin(None, KeyError("bug"), "p", [])


def test_ok_background_model_call_captures_result_and_error():
    ok = pc._BackgroundModelCall(lambda: "answer", tag="x")
    bad = pc._BackgroundModelCall(lambda: (_ for _ in ()).throw(ValueError("boom")))
    ok._thread.join(5)
    bad._thread.join(5)
    assert ok.done and ok.result == "answer" and ok.tag == "x" and ok.error is None
    assert bad.done and isinstance(bad.error, ValueError)


def test_ok_restart_env_marks_child_and_resume(make_cli, tmp_path):
    cli = make_cli()
    cli._fix_checkpoint = _meta_checkpoint(tmp_path)
    cli._resume_after_fix = True
    env = cli._restart_env(1)
    assert env[pc._RESTART_CHILD_ENV] == "1" and env[pc._RESTART_DEPTH_ENV] == "2"
    assert env[pc._RESUME_ENV] == cli._fix_checkpoint
    cli._resume_after_fix = False
    assert pc._RESUME_ENV not in cli._restart_env(1)


def test_ok_restart_depth_cap_does_not_launch(make_cli, tmp_path, monkeypatch):
    cli = _with_agent(make_cli(), tmp_path)
    monkeypatch.setenv(pc._RESTART_DEPTH_ENV, str(pc._MAX_RESTART_DEPTH))
    monkeypatch.setattr(pc.subprocess, "run", lambda *a, **k: pytest.fail("launched past the depth cap"))
    cli._restart_process()


def test_ok_restart_passes_script_args(make_cli, tmp_path, monkeypatch):
    cli = _with_agent(make_cli(), tmp_path)
    _stub_restart_side_effects(cli, monkeypatch)
    monkeypatch.setattr(sys, "argv", [cli.script_path, "--epochs", "3"])
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["env"] = list(argv), kw.get("env")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    cli._confirm_fix_did_its_job = lambda r: (True, "")
    with pytest.raises(SystemExit):
        cli._restart_process()
    assert seen["argv"][1:] == [os.path.abspath(cli.script_path), "--epochs", "3"]
    assert seen["env"][pc._RESTART_CHILD_ENV] == "1"
    assert seen["env"]["PULSE_AUTO_PROVIDER"] == cli.agent_provider


def test_ok_agent_setup_restores_named_provider_after_restart(make_cli, monkeypatch):
    name = next(n for n, info in pc.PROVIDERS.items() if info.get("env_key") and not info.get("local"))
    env_key = pc.PROVIDERS[name]["env_key"]
    monkeypatch.setenv("PULSE_AUTO_PROVIDER", name)
    monkeypatch.setenv(env_key, "sk-carried")
    cli = make_cli()
    cli._agent_setup()
    assert cli.agent_provider == name and cli.agent_key == "sk-carried"
    assert "PULSE_AUTO_PROVIDER" not in os.environ


def test_ok_fit_epochs_positional_and_keyword():
    assert pc._fit_epochs((1, 2, 32, 7), {}) == 7
    assert pc._fit_epochs(("ds",), {"epochs": 4}) == 4
    assert pc._fit_epochs(("ds",), {}) == 1


def test_ok_resume_into_fit_with_dataset_and_no_y(make_cli, fake_keras, tmp_path, monkeypatch):
    cli = make_cli()
    cli.script_path = str(tmp_path / "train.py")
    trained = fake_keras.Model(0.7)
    cli._keras_model, cli._keras_epoch, cli._keras_epochs = trained, 3, 10
    cli._save_fix_checkpoint()
    monkeypatch.setenv(pc._RESUME_ENV, cli._fix_checkpoint)
    fresh = fake_keras.Model(0.0)
    kwargs = {"x": "dataset", "epochs": 10}
    pc._resume_from_fix_checkpoint(fresh, (), kwargs)
    assert kwargs["initial_epoch"] == 4
    np.testing.assert_allclose(fresh.get_weights()[0], 0.7)


def test_ok_keras_hook_counts_batches_across_fits_and_records_epoch_logs(make_cli, fake_keras, monkeypatch):
    cli = make_cli()
    cli.auto_intervene = True
    monkeypatch.setattr(pc, "_PULSE_ACTIVE_INSTANCE", cli)
    pc._install_keras_fit_hook()
    steps = []
    cli.update = lambda step=None, **kw: steps.append(step)
    model = fake_keras.Model()
    user_cbs = []
    model.fit([[0.0] * 3], [0.0], epochs=2, callbacks=user_cbs)
    model.fit([[0.0] * 3], [0.0], epochs=2)
    assert user_cbs == []                                    # the user's list is not mutated
    assert steps == [1, 2, 3, 4]
    assert cli.epoch_scalar_histories["loss"] == [1.0, 0.5, 1.0, 0.5]
    assert cli.epoch_scalar_histories["train_loss"] == [1.0, 0.5, 1.0, 0.5]
    assert cli._keras_model is model and cli._keras_epochs == 2


def test_ok_record_keras_logs_keeps_nan_and_skips_none(make_cli):
    cli = make_cli()
    cli._record_keras_logs({"loss": float("nan"), "val_loss": None, "acc": np.float32(0.5)}, epoch=0)
    assert np.isnan(cli.epoch_scalar_histories["loss"][0])
    assert "val_loss" not in cli.epoch_scalar_histories
    assert cli.epoch_scalar_histories["acc"] == [0.5]


def test_ok_handle_crash_dedup_and_location(make_cli, tmp_path):
    cli = _with_agent(make_cli(), tmp_path)
    tb = (f'Traceback (most recent call last):\n  File "{cli.script_path}", line 7, in <module>\n'
          '    step()\nValueError: shapes (3,) and (4,) not aligned\n')
    s1 = cli.handle_crash(tb)
    s2 = cli.handle_crash(tb)
    assert s1 == s2 and cli._traceback_signatures_seen[s1] == 2
    assert cli._last_crash_location == (os.path.abspath(cli.script_path), 7)
    assert cli._crash_seen


def test_ok_end_of_run_review_escalates_silent_no_training_run(make_cli, tmp_path):
    cli = _with_agent(make_cli(), tmp_path)
    cli.auto_intervene = True
    escalated = []
    cli._escalate_training_problem = escalated.append
    cli._end_of_run_review()
    cli._end_of_run_review()                                  # once only
    assert len(escalated) == 1 and "without training" in escalated[0]


def test_ok_agent_log_path_is_pinned_absolute(env_guard, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PULSE_AGENT_LOG", "1")
    monkeypatch.setattr(pc, "_agent_log_resolved", {})
    first = pc._agent_log_path()
    assert first == str(tmp_path / "pulse_agent.log")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert pc._agent_log_path() == first and os.environ["PULSE_AGENT_LOG"] == first
    pc._agent_log_event("TEST EVENT", "line one\nline two")
    text = open(first, encoding="utf-8").read()
    assert "TEST EVENT" in text and "    line two" in text
    monkeypatch.setenv("PULSE_AGENT_LOG", "off")
    assert pc._agent_log_path() is None


def test_ok_config_file_selects_unattended_mode(make_cli, tmp_path):
    (tmp_path / "pulse_config.json").write_text(json.dumps({"Auto-Fix": "off", "sensitivity": "tight",
                                                           "PDFs": True}))
    cli = make_cli()
    cli.continuous = False
    cli.set_code_text("x = 1\n", script_path=str(tmp_path / "train.py"))
    assert cli.non_interactive and cli.continuous
    assert cli.auto_intervene is False
    assert cli.sensitivity == pc.PulseCLI._SENSITIVITY_PRESETS["tight"]
    assert cli.generate_pdfs is True


def test_ok_sensitivity_command_variants(make_cli):
    cli = make_cli()
    cli._cmd_sensitivity("0.8", quiet=True)
    assert cli.sensitivity == 0.8
    cli._cmd_sensitivity("7", quiet=True)
    assert cli.sensitivity == 1.0
    cli._cmd_sensitivity("spike 4", quiet=True)
    assert cli.explosion_multiplier == 4.0
    cli._cmd_sensitivity("spike auto", quiet=True)
    assert cli.explosion_multiplier is None
    assert "Usage" in cli._cmd_sensitivity("banana", quiet=True)


def test_ok_start_prime_answer_sets_first_checkin_and_note(make_cli, tmp_path):
    cli = _with_agent(make_cli(), tmp_path)
    cli._finish_start_prime("SENSITIVITY: 0.6\nNORMAL_START: none\nGPUTRACK: none\nNEXTCHECK: 60\n"
                            "CHECKNOTE: train.py:14 fits the scaler before the split", None, False)
    assert cli.sensitivity == pytest.approx(0.6)
    assert cli.checkin_interval_steps == 60
    assert "scaler" in cli._checkin_note and cli._start_prime_answered


def test_ok_start_prime_failure_retries_once_deferred(make_cli, tmp_path):
    cli = _with_agent(make_cli(), tmp_path)
    calls = []
    cli._start_start_prime = lambda deferred: calls.append(deferred)
    cli._finish_start_prime(None, pc.AgentRequestFailed("timeout"), False)
    cli._finish_start_prime(None, pc.AgentRequestFailed("timeout"), False)
    assert calls == [True]
