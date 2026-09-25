"""The periodic check-in: its own multi-round conversation, its own toolset, the
VERDICT format, and the note each scheduler leaves for the next check-in."""
import pytest

from pulse.pulse_cli import PulseCLI


def make_cli(replies, tools=None):
    """A PulseCLI with no training attached: _call_model answers from `replies`,
    and the tools are stubs that record what they were asked."""
    cli = PulseCLI.__new__(PulseCLI)
    cli.calls = []
    cli.ran = []
    replies = list(replies)

    def call_model(message, max_tokens=0, *, system=None, history=None, purpose=None):
        cli.calls.append({"message": message, "system": system, "history": list(history or [])})
        return replies.pop(0)

    cli._call_model = call_model
    for name in ("_run_grep", "_run_view", "_run_terminal", "_run_corr", "_run_outlier",
                 "_run_diffstats", "_run_histogram"):
        setattr(cli, name, (lambda n: lambda arg: cli.ran.append((n, arg)) or f"<{n} {arg}>")(name))
    cli._run_mllint = lambda: cli.ran.append(("_run_mllint", "")) or "<mllint>"
    return cli


@pytest.mark.parametrize("answer, expected", [
    ("VERDICT: ok\nNEXTCHECK: 5", False),
    ("VERDICT: OK.\n", False),
    ("VERDICT: ok -- everything finite, scaler fitted on train only", False),
    ("VERDICT: problem\nPROBLEM: validation_data is the training set (train.py:41)", True),
    # the old format, including the explained 'ok' that used to escalate every time
    ("STATUS: ok — data prep looks healthy (std 0.0013)", False),
    ("STATUS: looks healthy", False),
    ("STATUS: val_loss rising while loss falls", True),
])
def test_verdict(answer, expected):
    assert PulseCLI._checkin_verdict(answer)[0] is expected


def test_no_verdict_is_inconclusive():
    assert PulseCLI._checkin_verdict("I think it is fine")[0] is None


def test_problem_text_stops_at_the_next_field():
    answer = ("VERDICT: problem\nPROBLEM: scaler fitted before the split\n(train.py:22)\n"
              "NEXTCHECK: 3\nCHECKNOTE: confirm val accuracy drops once fixed")
    is_problem, problem = PulseCLI._checkin_verdict(answer)
    assert is_problem and problem == "scaler fitted before the split\n(train.py:22)"


def test_investigates_over_several_rounds_then_answers():
    cli = make_cli([
        "Let me look.\nGREP: StandardScaler",
        "TERMINAL: python -c 'print(1)'\nVIEW: 20-30",
        "VERDICT: problem\nPROBLEM: leak\nNEXTCHECK: 5\nCHECKNOTE: none",
    ])
    answer, transcript = cli._run_checkin("the run")
    assert answer.startswith("VERDICT: problem")
    assert [name for name, _ in cli.ran] == ["_run_grep", "_run_view", "_run_terminal"]
    assert len(cli.calls) == 3 and len(transcript) == 3
    # results go back in the next message, the conversation carries forward
    assert "<_run_grep StandardScaler>" in cli.calls[1]["message"]
    assert len(cli.calls[2]["history"]) == 4
    # its own system prompt, never the investigation agent's
    assert all(c["system"].startswith("You are Pulse's periodic check-in") for c in cli.calls)


def test_last_round_says_so_and_stops():
    rounds = PulseCLI._CHECKIN_MAX_ROUNDS
    cli = make_cli(["GREP: x"] * rounds + ["VERDICT: ok"])
    answer, _ = cli._run_checkin("the run")
    assert answer == "VERDICT: ok"
    assert len(cli.calls) == rounds + 1
    assert "last tool round" in cli.calls[-1]["message"]
    assert "last tool round" not in cli.calls[-2]["message"]


def test_live_object_tools_are_refused_not_run():
    cli = make_cli(["REPL: model.layers[0]\nDRYRUN: model(x)", "VERDICT: ok"])
    cli._run_exec_repl = lambda arg: pytest.fail("REPL must not run on the check-in thread")
    cli._run_exec_dryrun = lambda arg: pytest.fail("DRYRUN must not run on the check-in thread")
    cli._run_checkin("the run")
    assert "Not available during a check-in" in cli.calls[1]["message"]


def test_a_failing_tool_does_not_end_the_checkin():
    cli = make_cli(["OUTLIER: loss", "VERDICT: ok"])
    cli._run_outlier = lambda arg: 1 / 0
    answer, _ = cli._run_checkin("the run")
    assert answer == "VERDICT: ok"
    assert "OUTLIER loss: failed (ZeroDivisionError" in cli.calls[1]["message"]


def finished_cli():
    cli = make_cli([])
    cli.checkin_interval = 180.0
    cli.auto_intervene = True
    cli.agent_history = []
    cli.escalated = []
    cli._escalate_training_problem = cli.escalated.append
    cli._apply_directives = lambda *a, **k: ""
    return cli


def test_note_is_kept_for_the_next_checkin_and_problem_escalates():
    cli = finished_cli()
    cli._finish_periodic_checkin(
        "VERDICT: problem\nPROBLEM: relu output into binary_crossentropy (train.py:32)\n"
        "NEXTCHECK: 4\nCHECKNOTE: after the fix, val_accuracy should leave 0.48",
        None, "prompt", ["round 1: GREP relu"])
    assert cli.escalated == ["[periodic check-in] relu output into binary_crossentropy (train.py:32)"]
    assert cli._checkin_note == "after the fix, val_accuracy should leave 0.48"
    assert cli.checkin_interval == 240.0


def test_ok_does_not_escalate_and_none_clears_the_note():
    cli = finished_cli()
    cli._checkin_note = "old"
    cli._finish_periodic_checkin("VERDICT: ok -- all finite\nNEXTCHECK: 10\nCHECKNOTE: none",
                                 None, "prompt", [])
    assert cli.escalated == [] and cli._checkin_note == ""


def test_history_summary_samples_evenly():
    cli = make_cli([])
    cli._detector_histories = lambda: {"val_loss": list(range(100)), "empty": []}
    text = cli._checkin_history_summary(max_points=5)
    assert text == "- val_loss (100 readings): 0, 25, 50, 74, 99"


def test_prompt_formats():
    PulseCLI._CHECKIN_SYSTEM_PROMPT.format(min_mins=2, max_mins=60)
    PulseCLI._PERIODIC_CHECKIN_PROMPT.format(mins=3, tracked="(none)", snapshot="s", history="h", note="n")
    assert "CHECKNOTE:" in PulseCLI._START_PRIME_PROMPT


def test_missing_verdict_is_asked_for_once_more():
    cli = make_cli(["GREP: x", "I looked around and it seems fine overall.", "VERDICT: ok"])
    answer, transcript = cli._run_checkin("the run")
    assert answer == "VERDICT: ok"
    assert len(cli.calls) == 3
    assert "no VERDICT line" in cli.calls[-1]["message"]
    assert transcript[-1].startswith("verdict: asked again")


def fake_response(text):
    class Usage:
        prompt_tokens, completion_tokens, total_tokens = 10, 5, 15
    class Msg:
        content = text
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]
        usage = Usage()
    return Resp()


def real_call_cli(monkeypatch, tmp_path, replies):
    import pulse.pulse_cli as pc
    monkeypatch.setenv("PULSE_AGENT_LOG", str(tmp_path / "pulse_agent.log"))
    monkeypatch.setattr(pc.time, "sleep", lambda s: None)
    replies = list(replies)
    monkeypatch.setattr(pc.litellm, "completion", lambda **kw: fake_response(replies.pop(0)))
    cli = PulseCLI.__new__(PulseCLI)
    cli.agent_model_string, cli.agent_provider, cli.agent_api_base, cli.agent_key = "m", "p", None, "k"
    cli.agent_history = []
    cli._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "cost": 0.0}
    return cli, tmp_path / "pulse_agent.log"


def test_an_empty_reply_is_retried(monkeypatch, tmp_path):
    cli, log = real_call_cli(monkeypatch, tmp_path, ["", "VERDICT: ok"])
    assert cli._call_model("hello", purpose="periodic check-in, round 1") == "VERDICT: ok"
    text = log.read_text()
    assert "FAILED" in text and "empty response" in text and "VERDICT: ok" in text


def test_agent_log_shows_what_the_model_saw_and_answered(monkeypatch, tmp_path):
    cli, log = real_call_cli(monkeypatch, tmp_path, ["first answer", "second answer"])
    cli._call_model("look at train.py", system="SYS", history=[], purpose="periodic check-in, round 1")
    cli._call_model("again", system="SYS", history=[], purpose="periodic check-in, round 2")
    text = log.read_text()
    assert "CALL #" in text and "periodic check-in, round 1" in text
    assert "look at train.py" in text and "first answer" in text
    # the system prompt is written once, then referred to
    assert text.count("SYS\n") == 1 and "same as before" in text


def test_agent_log_is_off_unless_asked_for(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli, _log = real_call_cli(monkeypatch, tmp_path, ["an answer"])
    monkeypatch.delenv("PULSE_AGENT_LOG")
    cli._call_model("hello")
    assert list(tmp_path.iterdir()) == []


def test_run_flag_turns_the_agent_log_on(monkeypatch, tmp_path):
    from pulse import cli as pulse_cli_entry
    monkeypatch.delenv("PULSE_AGENT_LOG", raising=False)
    monkeypatch.chdir(tmp_path)
    pulse_cli_entry._parse_run_args(["--agent-log", "train.py"])
    assert __import__("os").environ["PULSE_AGENT_LOG"] == str(tmp_path / "pulse_agent.log")
    pulse_cli_entry._parse_run_args(["--agent-log=logs/a.log", "train.py", "--agent-log"])
    assert __import__("os").environ["PULSE_AGENT_LOG"] == str(tmp_path / "logs" / "a.log")


def test_pulse_itself_is_not_sent_as_a_project_file(tmp_path):
    from pulse.pulse import _discover_project_files
    (tmp_path / "helpers.py").write_text("X = 1\n")
    script = tmp_path / "train.py"
    script.write_text("from pulse import auto_track\nimport helpers\n")
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        found = _discover_project_files(str(script))
    finally:
        sys.path.remove(str(tmp_path))
    assert found == [str(tmp_path / "helpers.py")]
