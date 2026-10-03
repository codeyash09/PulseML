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


# =========================================================================================
# The agent acts on the request: run control, the request path, no stopping at a plan
# =========================================================================================

def test_the_agents_run_tools_are_the_apps_run_control(cli, monkeypatch):
    app = appmod.App(cli, cli._project_root)
    seen = []
    app._launch = lambda arg: seen.append(("launch", arg)) or setattr(app, "console", object()) \
        or setattr(app, "session", {"session_id": "s1", "script": "/p/train.py"})
    state = native._State(cli, "run it")
    app.install()
    try:
        out = native._execute(state, {"name": "start_run", "arguments": json.dumps({"script": "train.py", "args": "--epochs 2"})})
    finally:
        app.uninstall()
    assert seen == [("launch", "train.py --epochs 2")]
    assert "Started train.py under Pulse" in out and "run_status" in out
    assert [e.kind for e in app.view.entries] == ["tool"] and app.view.entries[0].calls == ["START_RUN: train.py --epochs 2"]


def test_run_tools_outside_the_app_say_so(cli):
    state = native._State(cli, "run it")
    out = native._execute(state, {"name": "start_run", "arguments": json.dumps({"script": "train.py"})})
    assert "only available inside the Pulse app" in out and "pulse run --stream" in out
    assert "only available inside the Pulse app" in native._execute(state, {"name": "run_status", "arguments": "{}"})


def test_run_status_and_restart_go_through_the_open_run(cli, tmp_path, monkeypatch):
    app = appmod.App(cli, cli._project_root)
    assert "No run is open" in app._agent_run_status()
    assert "No run is open" in app._agent_restart_run()
    app.console = types.SimpleNamespace(brain=types.SimpleNamespace(
        evidence=lambda include_code=False: {"x": 1}, render_evidence=lambda pack: "step 10, loss 0.5"))
    app.session = {"session_id": "s9", "script": str(tmp_path / "train.py")}
    assert "EVIDENCE FROM THE RUN (train.py" in app._agent_run_status() and "loss 0.5" in app._agent_run_status()
    assert "not started from here" in app._agent_restart_run()


def test_stop_run_asks_the_person_first(cli, monkeypatch):
    app = appmod.App(cli, cli._project_root)
    sent = []
    app.console = types.SimpleNamespace(send_control=lambda action, **f: sent.append(action))
    app.session = {"session_id": "s1", "script": "/p/train.py"}
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: False)
    assert "did not allow" in app._agent_stop_run() and sent == []
    monkeypatch.setattr(ui, "confirm", lambda *a, **k: True)
    assert "Asked train.py to stop" in app._agent_stop_run() and sent == ["stop"]


def test_a_training_command_that_times_out_points_at_start_run(cli, monkeypatch):
    state = native._State(cli, "run it")
    monkeypatch.setattr(cli, "_run_terminal", lambda arg, **k: "TERMINAL: python train.py\nTimed out after 120s\nExit code: -9")
    out = native._execute(state, {"name": "run_command", "arguments": json.dumps({"command": "python train.py"})})
    assert "start_run" in out
    monkeypatch.setattr(cli, "_run_terminal", lambda arg, **k: "Exit code: 0\nok")
    assert "start_run" not in native._execute(state, {"name": "run_command", "arguments": json.dumps({"command": "pytest -q"})})


def test_the_text_pipeline_has_run_directives(cli, monkeypatch):
    app = appmod.App(cli, cli._project_root)
    launched = []
    app._launch = lambda arg: launched.append(arg) or setattr(app, "console", object()) \
        or setattr(app, "session", {"session_id": "s1", "script": "/p/train.py"})
    app.install()
    try:
        notes = pulse_code._service(cli, "I will start it.\nRUN: train.py --lr 0.1")
    finally:
        app.uninstall()
    assert launched == ["train.py --lr 0.1"] and "Started train.py under Pulse" in notes
    assert pulse_code.tool_calls_in("RUN: train.py\nRUNSTATUS:") == ["RUN: train.py", "RUNSTATUS"]
    assert "RUN:" not in pulse_code._plain_text("RUN: train.py\nStarting it.\nNO_CHANGES")
    assert "only available inside the Pulse app" in pulse_code._service(cli, "RESTART:")
    assert "RUN: <script>" in pulse_code.CODE_SYSTEM_PROMPT and "start_run" in native.SYSTEM_PROMPT


def test_a_reply_that_only_announces_a_plan_is_nudged_to_act(cli, monkeypatch):
    answers = iter([native_reply("I'll start by reading train.py and then make the change."),
                    native_reply("", [call("read_file", {"path": "train.py"})]),
                    native_reply("Done: lr is set on line 1; nothing to change.")])
    sent = []

    def completion(**kw):
        sent.append(kw)
        return next(answers)

    monkeypatch.setattr(native.litellm, "completion", completion)
    assert native.run_native_turn(cli, "check lr") == "answered"
    nudge = sent[1]["messages"][-1]["content"]
    assert "did not do it" in nudge and "tools" in nudge
    assert len(sent) == 3


def test_a_plain_answer_is_not_nudged(cli, monkeypatch):
    sent = []

    def completion(**kw):
        sent.append(kw)
        return native_reply("lr is 0.3, set on line 1.")

    monkeypatch.setattr(native.litellm, "completion", completion)
    assert native.run_native_turn(cli, "what is lr?") == "answered" and len(sent) == 1


def test_the_debuggers_pause_prompt_handles_a_question_as_a_question(cli, monkeypatch):
    """A person's own request is not run through the locate/diagnose passes."""
    cli._native_off = True
    sent = []

    def completion(**kw):
        sent.append(str(kw["messages"][-1]["content"]))
        return reply("The learning rate is 0.3 (line 1).")

    monkeypatch.setattr(pc.litellm, "completion", completion)
    answer = cli.ask_agent("what is the learning rate?", include_code=True, from_user=True)
    assert answer == "The learning rate is 0.3 (line 1)."
    assert len(sent) == 1 and "what is the learning rate?" in sent[0] and "PASS 1" not in sent[0]
    assert "Do what they asked" in sent[0]


def test_the_debuggers_pause_prompt_implements_a_change_the_person_asked_for(cli, monkeypatch):
    cli._native_off = True
    cli.auto_intervene = True                 # no y/N: the pipeline applies the fix itself
    cli._restart_process = lambda *a, **k: None
    sent = []
    fix = {"old": ["lr = 0.3"], "new": ["lr = 0.05"], "files": ["train.py"], "explanation": "lower lr"}

    def completion(**kw):
        last = str(kw["messages"][-1]["content"])
        sent.append(last)
        if "Do what they asked" in last:
            return reply("I will change lr = 0.3 to lr = 0.05 on line 1.\nCHANGE_NEEDED")
        if "IMPLEMENT: Make exactly the change" in last:
            return reply(json.dumps(fix))
        if '"passes"' in last or "VERIFY" in last.upper():
            return reply('{"passes": true, "reason": "fine"}')
        return reply('{"resolved": true, "reason": "ok"}')

    monkeypatch.setattr(pc.litellm, "completion", completion)
    cli.ask_agent("lower the learning rate to 0.05", include_code=True, from_user=True)
    assert "lr = 0.05" in open(os.path.join(cli._project_root, "train.py")).read()
    assert any("The person asked for this change" in m and "lower the learning rate" in m for m in sent)
    assert not any("PASS 1 -- LOCATE" in m for m in sent)


def test_pulses_own_problems_still_take_the_diagnose_path(cli, monkeypatch):
    cli._native_off = True
    sent = []

    def completion(**kw):
        sent.append(str(kw["messages"][-1]["content"]))
        return reply("- train.py line 1")

    monkeypatch.setattr(pc.litellm, "completion", completion)
    cli.ask_agent("Pulse detected a problem: loss is NaN", include_code=True)
    assert any("PASS 1 -- LOCATE" in m for m in sent)


def test_the_dashboard_chat_handles_the_persons_request_as_asked(monkeypatch):
    fake_tk = types.SimpleNamespace(Frame=type("Frame", (object,), {}), TclError=Exception)
    monkeypatch.setattr(core, "tk", fake_tk)
    monkeypatch.setattr(core, "HAS_TK", True)
    monkeypatch.setattr(core, "_CHAT_PANEL_CLS", None)
    cls = core._chat_panel_class()
    panel = cls.__new__(cls)
    panel.appended, panel.history = [], []
    panel.after = lambda ms, fn=None: fn() if fn else None
    panel._append = lambda who, text: panel.appended.append((who, text))
    panel._set_stage = lambda label: None
    panel.promote_fn = lambda name: None
    panel._apply_directives = lambda *a, **k: ("", "", [])
    panel._apply_new_directives = lambda requests: ""
    sent = []
    panel._call_model = lambda model, instruction, images=None, max_tokens=0: sent.append(instruction) or "lr is 0.3."
    text, change = panel._user_request_turn("m", "context", "what is lr?", None)
    assert text == "lr is 0.3." and change is False
    assert "what is lr?" in sent[0] and "Do what they asked" in sent[0]
    assert ("Pulse", "lr is 0.3.") in panel.appended
    core._CHAT_PANEL_CLS = None


# =========================================================================================
# The screen: the latest exchange, the cursor, a run in the background
# =========================================================================================

def test_unscrolled_the_transcript_shows_only_the_latest_exchange():
    view = tui.View()
    for i in range(3):
        view.entries += [tui.Entry("user", f"question {i}"), tui.Entry("text", f"answer {i}")]
    shown = "\n".join(PLAIN.sub("", line) for line in tui.compose(view, 80, 24)[0])
    assert "question 2" in shown and "answer 2" in shown and "question 1" not in shown
    view.scroll = 3
    scrolled = "\n".join(PLAIN.sub("", line) for line in tui.compose(view, 80, 24)[0])
    assert "question 1" in scrolled or "question 0" in scrolled


def test_arrows_scroll_when_nothing_is_typed(cli):
    app = appmod.App(cli, cli._project_root)
    app.view.entries = [tui.Entry("text", f"line {i}") for i in range(80)]
    app.on_key("up")
    assert app.view.scroll == 3
    app.on_key("down")
    assert app.view.scroll == 0
    app.on_key("x")
    app.on_key("up")                         # with text typed the arrows are the history
    assert app.view.scroll == 0


def test_the_cursor_sits_right_after_the_typed_text_in_the_split_view():
    view = tui.View()
    view.side = ["train.py  ● live", "step 10"]
    view.editor.set("abc")
    frame, row, col = tui.compose(view, 100, 24)
    line = PLAIN.sub("", frame[row])
    assert line[col - 3:col] == "abc" and line[col] == " "


def test_the_screen_asks_for_an_orange_bar_cursor_and_puts_it_back(monkeypatch):
    written = []
    screen = tui.Screen(fd=1)
    monkeypatch.setattr(screen, "write", written.append)
    monkeypatch.setattr(screen._console, "enter", lambda **k: True)
    with screen:
        pass
    assert "\033]12;#ff8700\007" in written[0] and "\033[5 q" in written[0]
    assert "\033]112\007" in written[-1] and "\033[0 q" in written[-1]


def test_a_run_can_be_left_watching_in_the_background(cli, tmp_path, monkeypatch):
    from pulse import pulse_console as con
    from pulse.pulse_monitor import Monitor
    project = tmp_path / "runproj"
    project.mkdir()
    (project / "train.py").write_text("lr = 0.3\n")
    directory = str(project / ".pulse_stream" / "run1")
    monitor = Monitor(directory=directory, session_id="run1", script_path=str(project / "train.py"),
                      interval=0.0, tensor_interval=0.0)
    monitor.observe_locals({"loss": 0.5, "step": 3})
    monitor.snapshot_state({})
    monitor.writer.close()
    session = con._describe_session(directory)
    app = appmod.App(cli, cli._project_root)
    app._open_run(session)
    assert app.view.area == "DEBUG" and app.view.side is not None
    app._background_run()
    assert app.background and app.view.side is None and app.view.area == ""
    assert "watching train.py" in app.view.context and app.console is not None
    assert "EVIDENCE FROM THE RUN" in app._evidence()        # requests still carry the run
    monkeypatch.setattr(con, "discover", lambda *a, **k: [session])
    monkeypatch.setattr(con, "unmonitored_python_processes", lambda: [])
    app._pick_run("")                                        # /monitor brings it back
    assert not app.background and app.view.area == "DEBUG"
    app._close_run(quiet=True)
    assert app.console is None and app.cli._project_root == cli._project_root


def test_esc_goes_to_the_background_and_back(cli, tmp_path):
    app = appmod.App(cli, cli._project_root)
    app.console = types.SimpleNamespace(workdir=str(tmp_path), stop=lambda join=False: None,
                                        brain=types.SimpleNamespace(agent=None))
    app.session = {"session_id": "s1", "script": str(tmp_path / "train.py")}
    app.on_key("esc")
    import time as _t
    deadline = _t.time() + 5
    while _t.time() < deadline and not app.background:
        _t.sleep(0.02)
    assert app.background
    deadline = _t.time() + 5
    while _t.time() < deadline and app.view.busy:
        _t.sleep(0.02)
    app.on_key("esc")
    deadline = _t.time() + 5
    while _t.time() < deadline and app.background:
        _t.sleep(0.02)
    assert not app.background


# =========================================================================================
# In auto mode the reviewer model, not the person, answers "apply this change?"
# =========================================================================================

def reviewer(monkeypatch, verdict):
    seen = []

    def review(settings, request, explanation, diff, cwd, completion=None):
        seen.append((request, explanation, diff))
        if isinstance(verdict, Exception):
            raise verdict
        return pc._approver.Decision(approved=verdict.startswith("APPROVE"), reason=verdict.split(":", 1)[1].strip())

    monkeypatch.setattr(pc._approver, "review_change", review)
    monkeypatch.setattr(pc._approver, "settings_from", lambda *a, **k: pc._approver.ApproverSettings(model="judge"))
    return seen


def test_the_native_loops_edit_is_reviewed_by_the_model_not_the_person(cli, monkeypatch):
    seen = reviewer(monkeypatch, "APPROVE: does what was asked")
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: pytest.fail("the person must not be asked"))
    state = native._State(cli, "lower the lr")
    native._execute(state, {"name": "read_file", "arguments": json.dumps({"path": "train.py"})})
    out = native._execute(state, {"name": "edit_file", "arguments": json.dumps(
        {"path": "train.py", "old_string": "lr = 0.3", "new_string": "lr = 0.05"})})
    assert "NOT APPLIED" not in out and "lr = 0.05" in open(os.path.join(cli._project_root, "train.py")).read()
    assert seen and seen[0][0] == "lower the lr" and "+lr = 0.05" in seen[0][2]


def test_a_denied_edit_tells_the_agent_why(cli, monkeypatch):
    reviewer(monkeypatch, "DENY: that rewrites the whole file")
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: pytest.fail("the person must not be asked"))
    state = native._State(cli, "lower the lr")
    native._execute(state, {"name": "read_file", "arguments": json.dumps({"path": "train.py"})})
    out = native._execute(state, {"name": "edit_file", "arguments": json.dumps(
        {"path": "train.py", "old_string": "lr = 0.3", "new_string": "lr = 0.05"})})
    assert "NOT APPLIED" in out and "rewrites the whole file" in out
    assert "lr = 0.3" in open(os.path.join(cli._project_root, "train.py")).read()
    assert state.edits_declined == 1


def test_when_the_reviewer_cannot_answer_the_person_is_asked(cli, monkeypatch):
    reviewer(monkeypatch, pc._approver.ApproverUnavailable("timeout"))
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: "y")
    state = native._State(cli, "lower the lr")
    native._execute(state, {"name": "read_file", "arguments": json.dumps({"path": "train.py"})})
    out = native._execute(state, {"name": "edit_file", "arguments": json.dumps(
        {"path": "train.py", "old_string": "lr = 0.3", "new_string": "lr = 0.05"})})
    assert "NOT APPLIED" not in out


def test_the_text_pipeline_change_is_reviewed_too(cli, monkeypatch):
    cli._native_off = True
    seen = reviewer(monkeypatch, "APPROVE: fine")
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: pytest.fail("the person must not be asked"))
    fix = {"old": ["lr = 0.3"], "new": ["lr = 0.05"], "files": ["train.py"], "create": [], "explanation": "lower lr"}
    answers = iter([reply("Plan: change lr."), reply(json.dumps(fix)), reply('{"passes": true, "reason": "ok"}'),
                    reply('{"verified": true, "reason": "ok"}'), reply("Done.\nTASK_COMPLETE")])
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: next(answers))
    assert pulse_code.run_turn(cli, "lower the learning rate") == "applied"
    assert "lr = 0.05" in open(os.path.join(cli._project_root, "train.py")).read()
    assert seen and seen[0][1] == "lower lr"
