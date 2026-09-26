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
