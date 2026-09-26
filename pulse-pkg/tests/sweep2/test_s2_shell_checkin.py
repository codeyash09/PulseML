"""Sweep 2 -- periodic check-in verdict parsing and the agent log."""
import os

import pytest

import pulse.pulse_cli as pc
from pulse.pulse_cli import PulseCLI


@pytest.mark.parametrize("answer", [
    "VERDICT: no problems found\nNEXTCHECK: 500\nCHECKNOTE: none",
    "VERDICT: no issues -- metrics in the expected range\nNEXTCHECK: 500",
    "VERDICT: nothing wrong\nNEXTCHECK: 500",
])
def test_bug_negated_ok_verdict_escalates_as_a_problem(answer):
    """_checkin_verdict decides on the verdict's FIRST WORD only, so 'VERDICT: no problems
    found' ('no'), 'no issues', 'nothing wrong' are read as a problem: training is
    auto-paused and an investigation starts on a healthy run (the old STATUS: path even
    lists 'no issues'/'no problems' in _CHECKIN_OK_STATUSES). Correct: accept the same
    ok phrases for VERDICT, or at least treat 'no/nothing' + problem/issue as ok."""
    is_problem, _ = PulseCLI._checkin_verdict(answer)
    assert is_problem is False, answer


def test_bug_problem_text_loses_asterisks_and_backticks():
    """_checkin_verdict strips every '*' and '`' from the WHOLE answer before extracting
    PROBLEM, so the description handed to the investigation is corrupted: 'lr * 10'
    becomes 'lr  10', 'x ** 2' becomes 'x  2', `y_true` loses its code marks. Correct:
    strip markdown only around the field names/verdict word, not inside PROBLEM."""
    is_problem, problem = PulseCLI._checkin_verdict(
        "VERDICT: problem\nPROBLEM: train.py:12 scales the target twice: y = y * scale ** 2\nNEXTCHECK: 100")
    assert is_problem is True
    assert "y * scale ** 2" in problem, problem


def test_ok_verdict_forms():
    assert PulseCLI._checkin_verdict("**VERDICT:** ok -- all finite")[0] is False
    assert PulseCLI._checkin_verdict("VERDICT: problem\nPROBLEM: leak at a.py:3\nNEXTCHECK: 50")[1] == "leak at a.py:3"
    assert PulseCLI._checkin_verdict("TERMINAL: ls")[0] is None


def test_ok_agent_log_scrubs_and_pins_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PULSE_AGENT_LOG", "1")
    monkeypatch.setattr(pc, "_agent_log_resolved", {})
    key = "sk-or-v1-" + "d" * 40
    pc._agent_log_event("TEST", f"key {key}")
    path = tmp_path / "pulse_agent.log"
    assert path.exists() and key not in path.read_text()
    assert os.environ["PULSE_AGENT_LOG"] == str(path)


# ---- fix-round additions ------------------------------------------------------------------

import signal


@pytest.mark.parametrize("answer", [
    "VERDICT: problem: no clear improvement over 3 epochs\nPROBLEM: lr too low (train.py:4)",
    "VERDICT: no clear improvement\nNEXTCHECK: 50",
    "VERDICT: not ok\nPROBLEM: leak",
    "VERDICT: nothing wrong with the data, but the labels are shifted by one",
])
def test_ok_negations_that_are_problems_stay_problems(answer):
    assert PulseCLI._checkin_verdict(answer)[0] is True, answer


def test_bug_checknote_swallows_later_fields_and_tool_lines():
    """CHECKNOTE ran to the end of the answer unless NEXTCHECK/VERDICT/PROBLEM/GPU followed,
    so SENSITIVITY / NORMAL_START / tool lines became part of the note fed to the next check-in."""
    answer = ("NEXTCHECK: 100\nCHECKNOTE: watch val_loss after epoch 3\n"
              "SENSITIVITY: medium\nNORMAL_START: loss=2.3\nTERMINAL: ls")
    assert PulseCLI._parse_checknote(answer) == "watch val_loss after epoch 3"
    assert PulseCLI._parse_checknote("- **CHECKNOTE:** `x * 2` drifts") == "`x * 2` drifts"
    assert PulseCLI._parse_checknote("CHECKNOTE: none") == ""


@pytest.fixture
def real_cli(tmp_path, monkeypatch):
    old_sigint = signal.getsignal(signal.SIGINT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pc.cloud, "save_cached_profile", lambda **k: None, raising=False)
    for name in (pc._RESTART_CHILD_ENV, pc._RESUME_ENV, "PULSE_AGENT_LOG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PULSE_ASYNC_MODEL_CALLS", "0")
    cli = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdf"))
    cli._ensure_retry_ticker = lambda: None
    cli.code_text = "x = 1\n"
    cli.agent_provider = next(iter(pc.PROVIDERS))
    cli.agent_key = "test-key"
    cli._build_agent_context = lambda include_code=False: "snapshot"
    yield cli
    signal.signal(signal.SIGINT, old_sigint)


def test_bug_sync_start_check_that_fails_twice_is_asked_a_third_time(real_cli):
    calls = []

    def failing(*a, **k):
        calls.append(k.get("purpose"))
        raise pc.AgentRequestFailed("provider down")

    real_cli._call_model = failing
    real_cli._prime_at_start()
    real_cli._prime_with_agent_if_needed()
    assert len(calls) == 2, calls


def test_bug_sync_model_call_errors_leak_into_training(real_cli):
    """PULSE_ASYNC_MODEL_CALLS=0 caught only AgentRequestFailed: any other error in a
    start check or a check-in was raised into the user's training loop."""
    def broken(*a, **k):
        raise RuntimeError("pulse bug")

    real_cli._call_model = broken
    real_cli._prime_at_start()                      # must not raise
    real_cli._run_checkin = broken
    real_cli.checkin_interval_steps = 20
    real_cli.step = 25
    real_cli._maybe_periodic_checkin()              # must not raise
    assert real_cli._last_checkin_step == 25
