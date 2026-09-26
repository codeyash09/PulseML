"""Sweep 2, static area -- shared state between the training thread, the background
check-in worker, the restart path and the new interpreter-exit handler.

Each test_bug_* asserts the CORRECT behaviour, so it fails on the code as found.

Run:  PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/sweep2/test_s2_static_state.py
"""
import json
import os
import subprocess
import sys
import textwrap
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("PULSE_ASYNC_MODEL_CALLS", "0")
os.environ.setdefault("PULSE_LOGGING", "0")

from pulse import pulse_cli  # noqa: E402
from pulse.pulse_cli import PulseCLI  # noqa: E402


def _cli(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('training')\n")
    cli = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdfs"))
    cli.script_path = str(script)
    cli.code_text = script.read_text()
    cli.agent_provider = next(iter(pulse_cli.PROVIDERS))
    cli.agent_key = "sk-test-not-real"
    cli.auto_intervene = True
    cli._flush_cloud_now = lambda *a, **k: None
    cli._sync_agent_turn = lambda *a, **k: None
    cli._log_incident = lambda *a, **k: None
    return cli


class _DoneCall:
    """A _BackgroundModelCall whose worker has already finished."""

    def __init__(self, answer):
        self.done = True
        self.result = (answer, [])
        self.error = None
        self.prompt = "[Automatic check-in ...]"


_PROBLEM_ANSWER = ("VERDICT: problem\nPROBLEM: learning rate 5.0 at train.py:1 diverges the loss\n"
                   "NEXTCHECK: 500\nCHECKNOTE: none")


# =====================================================================================
# 1. A check-in that was running when a fix restarted the script is applied at the
#    parent's exit, after the fixed child run has already finished.
# =====================================================================================

def test_bug_stale_pre_fix_checkin_is_applied_after_the_fixed_restart_finished(tmp_path, monkeypatch):
    """A periodic check-in runs on a worker thread. If the deterministic detector escalates
    meanwhile and the fix restarts the script, _restart_process runs the fixed child to
    completion and then sys.exit()s the parent. Interpreter exit then runs
    _finish_background_calls_at_exit, which applies the parent's still-held _checkin_call:
    a verdict about the OLD, pre-fix code. A 'problem' verdict escalates again -- a new
    paid agent pipeline that may edit the already-fixed code and restart once more, after
    the fixed run is over. Correct: a restart discards (or marks stale) any check-in taken
    on the old process, so nothing from it is applied at exit."""
    cli = _cli(tmp_path)
    asked = []
    cli.ask_agent = lambda question, *a, **k: asked.append(question)
    cli._checkin_call = _DoneCall(_PROBLEM_ANSWER)          # started before the fix, landed later

    monkeypatch.setattr(cli, "_resolve_restart_interpreter", lambda: sys.executable)
    monkeypatch.setattr(cli, "_run_restart_child",
                        lambda argv, env, cwd: (subprocess.CompletedProcess(argv, 0, "", ""), True))
    monkeypatch.setattr(cli, "_confirm_fix_did_its_job", lambda result: (True, ""))
    monkeypatch.setattr(pulse_cli, "_stdout_is_tty", lambda: False)
    monkeypatch.delenv(pulse_cli._RESTART_DEPTH_ENV if hasattr(pulse_cli, "_RESTART_DEPTH_ENV")
                       else "PULSE_RESTART_DEPTH", raising=False)
    with pytest.raises(SystemExit):
        cli._restart_process()               # the fixed child ran to the end; parent exits

    cli._finish_background_calls_at_exit()   # what threading's exit hook runs next
    assert asked == [], ("a check-in of the pre-fix process was applied after the fixed run "
                         f"finished and re-escalated: {asked}")


# =====================================================================================
# 2. A restart triggered from the interpreter-exit hook loses the fixed run's exit code.
# =====================================================================================

def test_bug_restart_from_exit_hook_loses_the_fixed_runs_exit_status(tmp_path):
    """_finish_background_calls_at_exit applies a check-in that landed after the last step;
    a 'problem' verdict can escalate, apply a fix and _restart_process, which ends with
    sys.exit(child_returncode). The exit hook catches that SystemExit and `pass`es, so the
    process exits with the ORIGINAL run's status (0) -- a fixed run that failed with
    sys.exit(3) looks successful to the shell/CI/scheduler. (Raising it would not help
    either: exceptions from threading atexit hooks are only printed.) The retry ticker
    handles the same situation with os._exit(code). Correct: the process exits with the
    fixed child's status."""
    code = textwrap.dedent(f"""
        import os, sys, types
        sys.path.insert(0, {SRC!r})
        os.environ["PULSE_ASYNC_MODEL_CALLS"] = "0"
        os.environ["PULSE_LOGGING"] = "0"
        from pulse import pulse_cli
        from pulse.pulse_cli import PulseCLI

        class Done:
            done = True
            result = ("VERDICT: problem\\nPROBLEM: x", [])
            error = None
            prompt = "p"

        stub = types.SimpleNamespace(_start_prime_call=None, _checkin_call=Done(),
                                     _interrupted=False, _EXIT_WAIT_SECONDS=1.0)
        def finish(answer, error, prompt, transcript):
            sys.exit(3)          # _restart_process: the fixed child exited with status 3
        stub._finish_periodic_checkin = finish
        stub._finish_background_calls_at_exit = types.MethodType(
            PulseCLI._finish_background_calls_at_exit, stub)
        pulse_cli._PULSE_ACTIVE_INSTANCE = stub
        pulse_cli._register_exit_hook()
        print("script finished normally")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONPATH=SRC))
    assert "script finished normally" in proc.stdout, proc.stderr
    assert proc.returncode == 3, (f"exit status {proc.returncode}: the restarted run's status 3 was "
                                  f"swallowed at exit. stderr: {proc.stderr[-500:]}")


# =====================================================================================
# 3. A fix the user asks for resumes from a stale escalation checkpoint.
# =====================================================================================

class _FakeKerasModel:
    def __init__(self):
        self.weights = [np.zeros(3)]

    def save_weights(self, path):
        with open(path, "w") as f:
            f.write("weights")

    def get_weights(self):
        return list(self.weights)


def test_bug_user_requested_fix_resumes_from_a_stale_escalation_checkpoint(tmp_path):
    """The fix prompt promises 'Training resumes from the weights it had when this fix
    started'. _save_fix_checkpoint is only called from handle_crash and
    _escalate_training_problem. When an escalation at epoch 2 ended without a fix, the run
    trained on to epoch 8, and the user then paused (Ctrl+C) and asked for a fix, the
    restart's PULSE_RESUME_CHECKPOINT still points at the epoch-2 weights: six epochs of
    training are silently thrown away (and a fix asked for with no earlier escalation
    restarts from scratch). Correct: a top-level ask_agent that applies a fix resumes from
    the weights as they are when that fix started."""
    cli = _cli(tmp_path)
    cli._keras_model = _FakeKerasModel()
    cli._keras_epoch = 2
    cli._save_fix_checkpoint()                 # the epoch-2 escalation (no fix came of it)
    assert cli._fix_checkpoint

    cli._keras_epoch = 8                       # training went on
    cli._last_call_failed_transiently = False

    def fake_pipeline(question, include_code=False, _depth=0):
        cli._fix_applied_this_turn = True      # the user's fix was applied
        cli._last_applied_fix = {"old": ["a"], "new": ["b"], "explanation": "lower lr"}
        return "fixed"
    cli._ask_agent_impl = fake_pipeline
    cli.ask_agent("the learning rate is too high, please fix it", include_code=False)

    env = cli._restart_env(0)
    meta_path = env.get(pulse_cli._RESUME_ENV)
    assert meta_path, "no checkpoint to resume from for the user's fix"
    with open(meta_path) as f:
        epoch = json.load(f)["epoch"]
    assert epoch == 8, f"the restart resumes from epoch {epoch}'s weights, not the current epoch 8"


# =====================================================================================
# ok: the exit handler applies a check-in answer that landed after the last step
# =====================================================================================

def test_ok_exit_handler_applies_a_late_checkin_once(tmp_path):
    cli = _cli(tmp_path)
    asked = []
    cli.ask_agent = lambda question, *a, **k: asked.append(question)
    cli._escalate_training_problem = lambda problem: asked.append(problem)
    cli._checkin_call = _DoneCall(_PROBLEM_ANSWER)
    cli._finish_background_calls_at_exit()
    cli._finish_background_calls_at_exit()     # idempotent
    assert len(asked) == 1 and "learning rate 5.0" in asked[0]
    assert cli._checkin_call is None


def test_ok_exit_handler_skips_the_checkin_after_ctrl_c(tmp_path):
    cli = _cli(tmp_path)
    asked = []
    cli._escalate_training_problem = lambda problem: asked.append(problem)
    cli._checkin_call = _DoneCall(_PROBLEM_ANSWER)
    cli._interrupted = True
    cli._finish_background_calls_at_exit()
    assert asked == []


def test_ok_background_call_captures_base_exceptions():
    call = pulse_cli._BackgroundModelCall(lambda: (_ for _ in ()).throw(SystemExit(2)), prompt="p")
    call._thread.join(5)
    assert call.done and isinstance(call.error, SystemExit) and call.prompt == "p"
