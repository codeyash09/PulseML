"""Sweep 2 fix round, area restart: tests for the ledger entries that were confirmed by
reading only (fix-restart, the exit hook, Ctrl+C / SIGTERM handling in pulse_cli.py).

No real model calls, no real restarts of a training script: subprocess.run is faked where a
restart would launch one, and the one real child below is a tiny `python -c` sleeper.
"""
import os
import signal
import subprocess
import sys
import threading
import time as _real_time
import types

import pytest

import pulse.pulse_cli as pc
from pulse.pulse_cli import PulseCLI


class FakeTime:
    def __init__(self, now=1000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def __getattr__(self, name):
        return getattr(_real_time, name)


@pytest.fixture
def make_cli(tmp_path, monkeypatch):
    for name in (pc._RESTART_CHILD_ENV, pc._RESTART_DEPTH_ENV, pc._RESUME_ENV, "PULSE_AUTO_RESTART",
                 "PULSE_CONFIG", "PULSE_WEBHOOK_URL", "PULSE_AGENT_LOG"):
        monkeypatch.delenv(name, raising=False)
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
        cli._flush_cloud_now = lambda: None

        def no_real_model_calls(*a, **k):
            raise AssertionError("a test reached a real model call")

        cli._call_model = no_real_model_calls
        for key, value in attrs.items():
            setattr(cli, key, value)
        return cli

    yield factory
    signal.signal(signal.SIGINT, old_sigint)


def _with_script(cli, tmp_path):
    script = tmp_path / "train.py"
    script.write_text("x = 1\n")
    cli.script_path = str(script)
    cli.code_text = "x = 1\n"
    return cli


# ---- SIGTERM isn't passed on to the re-run ----------------------------------------------

def test_sigterm_is_passed_on_to_the_restarted_run(make_cli):
    """Killing the Pulse parent (SIGTERM from a scheduler or `kill`) while it waits for the
    fixed re-run must stop the re-run too, not leave it training as an orphan."""
    cli = make_cli()
    code = "import time\nfor _ in range(600):\n    time.sleep(0.1)\n"
    handlers = []

    def deliver_sigterm():
        # What the kernel would do on SIGTERM: run the handler installed for it. (Called
        # directly rather than with os.kill, so a broken hand-off cannot kill the test runner.)
        for _ in range(100):
            handler = signal.getsignal(signal.SIGTERM)
            if callable(handler):
                handlers.append(handler)
                _real_time.sleep(0.3)             # the child is running by now
                handler(signal.SIGTERM, None)
                return
            _real_time.sleep(0.05)

    old = signal.getsignal(signal.SIGTERM)
    threading.Thread(target=deliver_sigterm, daemon=True).start()
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
    assert handlers, "no SIGTERM handler was installed while the re-run was running"
    assert result.returncode == -signal.SIGTERM, result.returncode
    assert took < 20, f"the re-run kept running {took:.0f}s after SIGTERM"
    assert cli._restart_child_stopped_by == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == old, "the SIGTERM handler was not restored"


def test_a_sigterm_stopped_rerun_is_not_fed_to_the_agent(make_cli, tmp_path, monkeypatch):
    cli = _with_script(make_cli(), tmp_path)
    cli.agent_provider, cli.agent_key = next(iter(pc.PROVIDERS)), "k"
    monkeypatch.setattr(pc, "_stdout_is_tty", lambda: False)
    asked = []
    cli._ask_agent_impl = lambda q, include_code=False, _depth=0: asked.append(q)
    monkeypatch.setattr(pc.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, -signal.SIGTERM, "", ""))
    with pytest.raises(SystemExit) as exc:
        cli._restart_process()
    assert asked == [] and exc.value.code == 128 + signal.SIGTERM


# ---- A stray 'TERM environment variable not set' line -----------------------------------

def test_restart_does_not_run_clear_without_term(make_cli, tmp_path, monkeypatch):
    cli = _with_script(make_cli(), tmp_path)
    monkeypatch.setattr(pc, "_stdout_is_tty", lambda: True)
    monkeypatch.delenv("TERM", raising=False)
    calls = []

    def fake_run(argv, **kw):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pc.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        cli._restart_process()
    assert ["clear"] not in calls, calls


# ---- The exit hook gives up on a long multi-round check-in ------------------------------

def test_exit_hook_waits_for_a_multi_round_checkin(make_cli, tmp_path, monkeypatch):
    """A check-in of several tool rounds can take well over 10 s; its answer is still applied."""
    cli = _with_script(make_cli(), tmp_path)
    cli.auto_intervene = True
    cli.agent_provider, cli.agent_key = next(iter(pc.PROVIDERS)), "k"
    clock = FakeTime()
    monkeypatch.setattr(pc, "time", clock)
    finish_at = clock.now + 45.0
    answer = ("VERDICT: problem\nPROBLEM: labels are shuffled apart from the features (train.py:1)\n"
              "NEXTCHECK: 500\nCHECKNOTE: none", [])

    class SlowCall:
        error, prompt = None, "p"

        @property
        def done(self):
            return clock.now >= finish_at

        @property
        def result(self):
            return answer if self.done else None

    cli._checkin_call = SlowCall()
    escalated = []
    cli._escalate_training_problem = escalated.append
    cli._finish_background_calls_at_exit()
    assert escalated and "labels are shuffled" in escalated[0]


def test_exit_hook_wait_ends_on_ctrl_c(make_cli, tmp_path, monkeypatch):
    cli = _with_script(make_cli(), tmp_path)
    clock = FakeTime()
    start = clock.now

    def sleep(seconds):
        clock.now += seconds
        if clock.now - start > 20:
            cli._interrupted = True               # Ctrl+C while the notice is up

    clock.sleep = sleep
    monkeypatch.setattr(pc, "time", clock)
    cli._checkin_call = types.SimpleNamespace(done=False, result=None, error=None, prompt="p")
    cli._finish_background_calls_at_exit()
    assert clock.now - start < 25


# ---- Ctrl+C at the pause prompt exits only after 5 s ------------------------------------

def test_ctrl_c_at_the_pause_prompt_stops_at_once(make_cli, monkeypatch):
    cli = make_cli()
    cli.continuous = False
    timers = []
    monkeypatch.setattr(pc.threading, "Timer", lambda *a, **k: timers.append(a) or types.SimpleNamespace(
        daemon=True, start=lambda: None))
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)

    def prompt(text=""):
        cli._sigint_handler(signal.SIGINT, None)   # Ctrl+C while Pulse waits at its prompt
        return ""                                  # (Enter, had the handler only latched)

    monkeypatch.setattr("builtins.input", prompt)
    with pytest.raises(KeyboardInterrupt):
        cli.update(step=1)
    assert timers == [], "the Ctrl+C was latched for a later pause instead of stopping"


# ---- `_update_running` isn't reset if `update()` raises ----------------------------------

def test_update_running_is_reset_when_update_raises(make_cli, monkeypatch):
    cli = make_cli({"loss": 1.0}, tracked_vars=["loss"], var_states={"loss": "track"})

    def boom(name):
        raise RuntimeError("inside update()")

    monkeypatch.setattr(pc, "_looks_like_loss", boom)
    with pytest.raises(RuntimeError):
        cli.update(step=1)
    assert cli._update_running is False and cli._in_update is False


# ---- A restart the retry ticker asked for runs on the training thread --------------------

def test_deferred_restart_runs_at_the_next_update(make_cli):
    cli = make_cli()
    restarted = []
    cli._restart_process = lambda: restarted.append(threading.current_thread())
    cli._restart_deferred = True
    cli.update(step=1)
    assert restarted == [threading.current_thread()] and cli._restart_deferred is False
