"""The agent can talk to the person while it keeps working: `message_user` in Pulse Code's
native tool loop, and the `MESSAGE:` directive for the debugger, the check-in, the text
pipeline and the dashboard. A message is shown at once, goes back to the model as nothing
(so a reply that only carries a message is not a tool round), and inside the Pulse app it
is an entry of its own."""
import json
import os
import re
import sys
import types

import pytest

import pulse.pulse_cli as pc
from pulse import pulse as core
from pulse import pulse_app as appmod
from pulse import pulse_code
from pulse import pulse_code_agent as native
from pulse import pulse_supabase as cloud
from pulse import pulse_tui as tui
from pulse import pulse_ui as ui

PLAIN = re.compile(r"\033\[[0-9;]*m")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    saved_env, cwd = dict(os.environ), os.getcwd()
    for name in ("OPENROUTER_API_KEY", "PULSE_PROVIDER", "PULSE_CONFIG", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "pulsehome" / "credentials.json")
    monkeypatch.setattr(cloud, "save_cached_profile", lambda **k: None)
    monkeypatch.setattr(ui, "_UNICODE", True)
    yield
    ui.set_host(None)
    pc.set_agent_observer(None)
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(saved_env)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "train.py").write_text("lr = 0.3\n\nfor step in range(10):\n    loss = step * lr\n")
    return root


@pytest.fixture
def cli(project, monkeypatch):
    c = appmod._new_cli(str(project), [str(project / "train.py")])
    c.agent_provider = next(n for n, info in pc.PROVIDERS.items() if info.get("model") and info.get("env_key"))
    c.agent_key = "sk-test"
    c._sync_agent_turn = lambda *a, **k: None
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: pytest.fail("no model call expected here"))
    return c


# ---- the plain terminal and the app -----------------------------------------------------

def test_a_message_prints_with_a_bar_on_a_plain_terminal(capsys):
    ui.message("Looking at the loss first.\nThen the scheduler.")
    out = PLAIN.sub("", capsys.readouterr().out)
    assert "│ Looking at the loss first." in out and "│ Then the scheduler." in out
    ui.message("   ")
    assert capsys.readouterr().out == ""


def test_inside_the_app_a_message_is_its_own_entry(cli):
    app = appmod.App(cli, cli._project_root)
    app.install()
    try:
        ui.message("Checking lr now")
    finally:
        app.uninstall()
    assert [e.kind for e in app.view.entries] == ["say"]
    lines = [PLAIN.sub("", line) for line in tui.entry_lines(app.view.entries[0], 40, False)]
    assert lines == ["│ Checking lr now"]


# ---- the debugger's directive -------------------------------------------------------------

def test_message_is_a_directive_the_debugger_knows():
    cleaned, requests = pc.PulseCLI._extract_new_directives("Let me look.\nMESSAGE: the loss floor is 0.5\nGREP: lr")
    assert requests["message"] == ["the loss floor is 0.5"] and requests["grep"] == ["lr"] if "grep" in requests else True
    assert "MESSAGE:" not in cleaned
    assert "message" in pc.PulseCLI._CHECKIN_TOOL_NAMES


def test_a_message_alone_is_shown_and_is_not_a_tool_round(cli, monkeypatch):
    shown = []
    monkeypatch.setattr(ui, "message", shown.append)
    echoed = []
    monkeypatch.setattr(cli, "_echo_directives", lambda requests: echoed.append(dict(requests)))
    assert cli._service_tool_requests("MESSAGE: the floor is the +0.5 on line 4") == ""
    assert shown == ["the floor is the +0.5 on line 4"] and echoed == [{}]


def test_a_message_beside_a_tool_is_shown_once_and_the_tool_still_runs(cli, monkeypatch):
    shown = []
    monkeypatch.setattr(ui, "message", shown.append)
    notes = cli._service_tool_requests("MESSAGE: looking at lr\nGREP: lr")
    assert shown == ["looking at lr"] and "lr = 0.3" in notes


def test_a_no_change_answer_may_carry_a_message():
    answer = "MESSAGE: nothing to change here\nNo code change needed: the value is already correct."
    assert pc.PulseCLI._parse_no_change(answer) is not None


def test_the_check_in_shows_a_message_and_sends_nothing_back(cli, monkeypatch):
    shown = []
    monkeypatch.setattr(ui, "message", shown.append)
    results, summary = cli._checkin_service_tools("MESSAGE: still healthy so far\nGREP: lr")
    assert shown == ["still healthy so far"] and "lr = 0.3" in results
    assert "MESSAGE still healthy so far" in summary
    results, summary = cli._checkin_service_tools("MESSAGE: only a message")
    assert results == "" and shown[-1] == "only a message"


def test_the_prompts_tell_every_agent_about_it():
    assert "MESSAGE: <one line>" in pc.SYSTEM_PROMPT
    assert "MESSAGE: <one line>" in pc.PulseCLI._CHECKIN_SYSTEM_PROMPT
    assert "MESSAGE: <one line>" in pulse_code.CODE_SYSTEM_PROMPT
    assert "MESSAGE: <one line>" in core.SYSTEM_PROMPT
    assert "message_user" in native.SYSTEM_PROMPT
    assert any(t["function"]["name"] == "message_user" for t in native.TOOLS)


# ---- the text pipeline ---------------------------------------------------------------------

def test_the_code_agents_text_pipeline_allows_it_and_hides_the_line(cli, monkeypatch):
    assert "message" in pulse_code._ALLOWED_TOOLS
    assert pulse_code.tool_calls_in("MESSAGE: hi\nGREP: lr") == ["GREP: lr"]
    assert pulse_code._plain_text("MESSAGE: hi\nThe answer.\nNO_CHANGES") == "The answer."


def reply(text, reasoning=None):
    message = types.SimpleNamespace(content=text, reasoning_content=reasoning, tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
                                 usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))


def test_in_the_app_the_text_pipeline_shows_the_message_then_the_tools(cli, monkeypatch):
    cli._native_off = True
    answers = iter([reply("MESSAGE: I will check where lr is used.\nGREP: lr"),
                    reply("The learning rate is 0.3.\nNO_CHANGES")])
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: next(answers))
    app = appmod.App(cli, cli._project_root)
    app.install()
    try:
        outcome = pulse_code.run_turn(cli, "where is lr used?")
    finally:
        app.uninstall()
    assert outcome == "answered"
    kinds = [e.kind for e in app.view.entries]
    assert kinds.index("say") < kinds.index("tool")
    assert app.view.entries[kinds.index("say")].text == "I will check where lr is used."
    assert "MESSAGE:" not in "\n".join(e.text for e in app.view.entries)


# ---- the native tool loop ------------------------------------------------------------------

def test_message_user_shows_the_message_and_is_not_listed_as_a_tool(cli):
    state = native._State(cli, "why?")
    app = appmod.App(cli, cli._project_root)
    app.install()
    try:
        result = native._execute(state, {"name": "message_user", "arguments": json.dumps({"message": "Reading train.py"})})
        empty = native._execute(state, {"name": "message_user", "arguments": "{}"})
    finally:
        app.uninstall()
    assert result == "Shown to the user." and "empty" in empty
    assert [e.kind for e in app.view.entries] == ["say"] and app.view.entries[0].text == "Reading train.py"


def call(name, arguments, cid="c1"):
    return types.SimpleNamespace(id=cid, function=types.SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def native_reply(text, calls=()):
    message = types.SimpleNamespace(content=text, tool_calls=list(calls) or None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
                                 usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))


def test_the_native_loop_messages_and_works_in_the_same_turn(cli, monkeypatch):
    sent = []
    answers = iter([
        native_reply("", [call("message_user", {"message": "Looking at where lr is used first."}),
                          call("grep", {"pattern": "lr"}, "c2")]),
        native_reply("lr is set to 0.3 on line 1 and used in the loop."),
    ])

    def completion(**kw):
        sent.append(kw)
        return next(answers)

    monkeypatch.setattr(native.litellm, "completion", completion)
    app = appmod.App(cli, cli._project_root)
    app.install()
    try:
        outcome = native.run_native_turn(cli, "where is lr used?")
    finally:
        app.uninstall()
    assert outcome == "answered"
    kinds = [e.kind for e in app.view.entries]
    assert kinds[:2] == ["say", "tool"] and "lr is set to 0.3" in app.view.entries[-1].text
    assert app.view.entries[0].text == "Looking at where lr is used first."
    assert app.view.entries[1].calls == ["GREP: lr"]
    # the model got "Shown to the user." for the message and the real grep result
    tool_results = [m for m in sent[1]["messages"] if m.get("role") == "tool"]
    assert [m["content"] for m in tool_results][0] == "Shown to the user."
    assert "lr = 0.3" in tool_results[1]["content"]


# ---- the dashboard -------------------------------------------------------------------------

def test_the_dashboard_puts_a_message_in_the_chat(monkeypatch):
    fake_tk = types.SimpleNamespace(Frame=type("Frame", (object,), {}), TclError=Exception)
    monkeypatch.setattr(core, "tk", fake_tk)
    monkeypatch.setattr(core, "HAS_TK", True)
    monkeypatch.setattr(core, "_CHAT_PANEL_CLS", None)
    cls = core._chat_panel_class()
    panel = cls.__new__(cls)
    panel.appended = []
    panel.after = lambda ms, fn=None: fn() if fn else None
    panel._append = lambda who, text: panel.appended.append((who, text))
    _cleaned, requests = core._extract_new_directives("MESSAGE: the batch looks wrong\nDEPGRAPH:")
    assert requests["message"] == ["the batch looks wrong"]
    assert panel._apply_new_directives({"message": ["the batch looks wrong"]}) == ""
    assert panel.appended == [("Pulse", "the batch looks wrong")]
    core._CHAT_PANEL_CLS = None


# ---- live thinking: the model's reasoning as it streams ------------------------------------

def chunk(content=None, reasoning=None):
    delta = types.SimpleNamespace(content=content, reasoning_content=reasoning)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


def streaming_model(monkeypatch, pieces, final_text, final_reasoning):
    """A provider that streams `pieces` (reasoning or text deltas) when asked to."""
    calls = []

    def completion(**kw):
        calls.append(kw)
        if kw.get("stream"):
            return iter(pieces)
        return reply(final_text, final_reasoning)

    def builder(chunks, messages=None):
        return reply(final_text, final_reasoning)

    monkeypatch.setattr(pc.litellm, "completion", completion)
    monkeypatch.setattr(pc.litellm, "stream_chunk_builder", builder)
    return calls


def test_reasoning_streams_into_a_live_thinking_entry(cli, monkeypatch):
    pieces = [chunk(reasoning="The floor "), chunk(reasoning="is 0.5."), chunk(content="The loss "),
              chunk(content="is flat.")]
    calls = streaming_model(monkeypatch, pieces, "The loss is flat.", "The floor is 0.5.")
    app = appmod.App(cli, cli._project_root)
    seen = []
    app.install()
    try:
        original = app._observe

        def spy(event, **data):
            seen.append((event, data.get("text"), [ (e.kind, e.live, e.text) for e in app.view.entries]))
            original(event, **data)

        pc.set_agent_observer(spy)
        answer = cli._call_model("why?")
    finally:
        app.uninstall()
    assert answer == "The loss is flat."
    assert calls[0]["stream"] is True
    events = [e for e, _t, _s in seen]
    assert events == ["stream_start", "reasoning_delta", "reasoning_delta", "content_delta", "content_delta",
                      "stream_end"]
    # while streaming, the thinking entry was live and growing; the answer was a live line
    _e, _t, during = seen[4]
    assert ("thinking", True, "The floor is 0.5.") in during and ("text", True, "The loss ") in during
    # afterwards: the thinking is a normal (folded) entry and the live answer line is gone
    kinds = [(e.kind, e.live, e.text) for e in app.view.entries]
    assert kinds == [("thinking", False, "The floor is 0.5.")]


def test_a_live_thinking_entry_shows_its_last_lines_while_folded():
    entry = tui.Entry("thinking", "one\ntwo\nthree\nfour\nfive\nsix", live=True)
    lines = [PLAIN.sub("", line) for line in tui.entry_lines(entry, 40, False)]
    assert lines[0].startswith("✻ Thinking") and lines[1:] == ["  three", "  four", "  five", "  six"]
    entry.live = False
    entry.touch()
    assert len(tui.entry_lines(entry, 40, False)) == 1


def test_streaming_is_only_used_when_something_is_watching(cli, monkeypatch):
    calls = streaming_model(monkeypatch, [], "answer", None)
    assert cli._call_model("plain") == "answer"
    assert "stream" not in calls[0]                      # no observer: a plain call
    monkeypatch.setenv("PULSE_STREAMING", "0")
    app = appmod.App(cli, cli._project_root)
    app.install()
    try:
        assert cli._call_model("plain") == "answer"
    finally:
        app.uninstall()
    assert "stream" not in calls[1]


def test_a_provider_that_answers_in_full_when_asked_to_stream_is_fine(cli, monkeypatch):
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: reply("whole", "thought"))
    app = appmod.App(cli, cli._project_root)
    app.install()
    try:
        assert cli._call_model("q") == "whole"
    finally:
        app.uninstall()
    assert [(e.kind, e.text) for e in app.view.entries] == [("thinking", "thought")]


def test_the_native_loop_streams_through_the_same_path(cli, monkeypatch):
    sent = []

    def completion(**kw):
        sent.append(kw)
        if kw.get("stream"):
            return iter([chunk(reasoning="Reading the loop."), chunk(content="lr is 0.3.")])
        return native_reply("lr is 0.3.")

    monkeypatch.setattr(native.litellm, "completion", completion)
    monkeypatch.setattr(native.litellm, "stream_chunk_builder", lambda chunks, messages=None: native_reply("lr is 0.3."))
    app = appmod.App(cli, cli._project_root)
    app.install()
    try:
        assert native.run_native_turn(cli, "where is lr?") == "answered"
    finally:
        app.uninstall()
    assert sent[0]["stream"] is True and sent[0]["tools"]
    assert [e.kind for e in app.view.entries][:1] == ["thinking"]
    assert app.view.entries[0].text == "Reading the loop."


def test_the_app_is_allowed_on_windows_when_the_console_can_draw(monkeypatch):
    monkeypatch.setattr(ui, "enabled", lambda: True)
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((120, 40)))
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(tui, "windows_console_ready", lambda: False)
    assert appmod.usable() is False
    monkeypatch.setattr(tui, "windows_console_ready", lambda: True)
    assert appmod.usable() is True
