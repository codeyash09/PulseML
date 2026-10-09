"""The Pulse app: one full-screen layout for the agent and the runs on this machine.

Three layers are tested here, none of them against a real terminal:
  * pulse_tui -- pure: text folding, keys, the editor, and compose() (a View -> one frame);
  * the hosting in pulse_app.App -- the existing pipelines print, ask and run stages
    inside the app without knowing it;
  * the run view -- opening a run, the run pane, the commands, going back.
No model is called: litellm.completion is replaced in every test that reaches it."""
import builtins
import getpass
import json
import os
import re
import sys
import threading
import time
import types

import pytest

import pulse.pulse_cli as pc
from pulse import cli as entry
from pulse import pulse_app as appmod
from pulse import pulse_code
from pulse import pulse_console as con
from pulse import pulse_supabase as cloud
from pulse import pulse_tui as tui
from pulse import pulse_ui as ui
from pulse.pulse_monitor import Monitor

PLAIN = re.compile(r"\033\[[0-9;]*m")


def plain(text):
    return PLAIN.sub("", text)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    saved_env = dict(os.environ)
    saved_providers = dict(pc.PROVIDERS)
    cwd = os.getcwd()
    for name in ("OPENROUTER_API_KEY", "PULSE_PROVIDER", "PULSE_APPROVER", "PULSE_CONFIG", "PULSE_CLASSIC",
                 "PULSE_MODEL", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PULSE_HOME", str(tmp_path / "pulsehome"))
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "pulsehome" / "credentials.json")
    monkeypatch.setattr(cloud, "save_cached_profile", lambda **k: None)
    monkeypatch.setattr(ui, "_UNICODE", True)
    yield
    ui.set_host(None)
    pc.set_agent_observer(None)
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(saved_env)
    pc.PROVIDERS.clear()
    pc.PROVIDERS.update(saved_providers)


# =========================================================================================
# pulse_tui: text
# =========================================================================================

def test_wrap_breaks_at_words_and_never_exceeds_the_width():
    lines = tui.wrap("the quick brown fox jumps over the lazy dog", 12)
    assert lines == ["the quick", "brown fox", "jumps over", "the lazy dog"]
    assert tui.wrap("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]
    assert tui.wrap("", 10) == [""]


def test_wrap_carries_colour_across_a_fold_and_closes_it():
    lines = tui.wrap("\033[31mred words that go on and on\033[0m plain tail", 14)
    assert all(tui.visible_len(line) <= 14 for line in lines)
    assert lines[0].startswith("\033[31m") and lines[0].endswith(tui.RESET)
    assert lines[1].startswith("\033[31m")              # still red after the fold
    assert "\033[31m" not in lines[-1]                  # and not after the reset
    assert plain(" ".join(lines)) == "red words that go on and on plain tail"


def test_wrap_indents_continuation_lines():
    lines = tui.wrap("    result: a long explanation that folds", 20, indent="    ")
    assert all(line.startswith("    ") for line in lines) and len(lines) > 1


def test_clean_keeps_colour_and_drops_every_other_escape():
    dirty = "\033[2K\rhello \033[32mgreen\033[0m\033[?25l\x07\033[10;3H tab\there\033]0;title\x07"
    assert tui.clean(dirty) == "\rhello \033[32mgreen\033[0m tab    here"


def test_pad_is_exactly_the_width():
    assert tui.visible_len(tui.pad("abc", 10)) == 10
    assert tui.visible_len(tui.pad("\033[1mabcdefghijklmnop\033[0m", 10)) == 10


# =========================================================================================
# pulse_tui: keys and the editor
# =========================================================================================

def test_parse_keys_names_what_a_terminal_sends():
    keys, rest = tui.parse_keys("ab\033[A\033[5~\x0f\r\x7f\033[200~pasted\ntext\033[201~\033[1;2B")
    assert keys == ["a", "b", "up", "pgup", "ctrl+o", "enter", "backspace", "paste:pasted\ntext", "pgdn"]
    assert rest == ""


def test_parse_keys_waits_for_the_rest_of_a_sequence():
    assert tui.parse_keys("a\033") == (["a"], "\033")
    assert tui.parse_keys("\033[") == ([], "\033[")
    assert tui.parse_keys("\033[200~half") == ([], "\033[200~half")
    assert tui.parse_keys("\x03") == (["ctrl+c"], "")


def test_editor_edits_one_line_and_remembers_history():
    editor = tui.Editor()
    for key in "hello":
        editor.handle(key)
    editor.handle("left"); editor.handle("left"); editor.handle("X")
    assert (editor.text, editor.pos) == ("helXlo", 4)
    editor.handle("backspace"); editor.handle("end"); editor.handle("!")
    assert editor.text == "hello!"
    editor.handle("paste:two\nlines")
    assert editor.text == "hello!two lines"                      # a pasted newline never submits
    editor.handle("ctrl+w")
    assert editor.text == "hello!two "
    assert editor.take() == "hello!two " and editor.text == ""
    editor.handle("z")
    editor.handle("up")
    assert editor.text == "hello!two "
    editor.handle("down")
    assert editor.text == "z"                                    # the draft comes back


# =========================================================================================
# pulse_tui: entries and frames
# =========================================================================================

def test_thinking_is_one_line_until_expanded():
    entry_ = tui.Entry("thinking", "the scheduler maybe\nsteps too often")
    folded = [plain(line) for line in tui.entry_lines(entry_, 60, False)]
    assert folded == ["Thought (6 words)"]
    entry_.seconds = 4.2
    entry_.touch()
    assert [plain(line) for line in tui.entry_lines(entry_, 60, False)] == ["Thought for 4.2s"]
    opened = [plain(line) for line in tui.entry_lines(entry_, 60, True)]
    assert opened[0] == "Thought for 4.2s" and "the scheduler maybe" in opened[1]
    assert "steps too often" in opened[2]


def test_tool_calls_read_as_what_was_done_and_expand_to_the_output():
    entry_ = tui.Entry("tool", calls=["GREP: lr_scheduler", "VIEW: train.py:40-80"], output="a\nb\nc")
    folded = [plain(line) for line in tui.entry_lines(entry_, 60, False)]
    assert folded == ["Searched for lr_scheduler · Read train.py:40-80  · 3 lines"]
    opened = [plain(line) for line in tui.entry_lines(entry_, 60, True)]
    assert opened[0].startswith("Searched for lr_scheduler") and "GREP: lr_scheduler" in opened[0]
    assert [line.strip().lstrip("⎿ ").strip() for line in opened[2:]] == ["a", "b", "c"]
    assert tui.describe_call("TERMINAL: pytest -q") == "Ran pytest -q"
    assert tui.describe_call("EDIT_FILE: train.py") == "Edited train.py"
    assert tui.describe_call("RUN_STATUS") == "Checked the run"


def test_a_finding_keeps_its_severity_and_folds_under_it():
    entry_ = tui.Entry("finding", "loss has barely moved over its last 20 readings", severity="warning")
    lines = [plain(line) for line in tui.entry_lines(entry_, 30, False)]
    assert lines[0].startswith("▲ WARNING ") and all(len(line) <= 30 for line in lines)
    assert lines[1].startswith(" " * len("▲ WARNING "))


def demo_view():
    view = tui.View()
    view.context, view.agent = "~/proj", "agent: DeepSeek"
    view.commands = [("/monitor", "watch a run"), ("/help", "commands")]
    view.entries += [
        tui.Entry("user", "why is the loss flat?"),
        tui.Entry("thinking", "maybe the scheduler"),
        tui.Entry("tool", calls=["GREP: lr"], output="a\nb"),
        tui.Entry("text", "The scheduler steps every batch, so the learning rate is zero by step 900."),
    ]
    return view


@pytest.mark.parametrize("width, height", [(100, 22), (120, 40), (84, 12), (60, 24), (40, 12)])
def test_a_frame_fills_the_terminal_but_its_last_row(width, height):
    """One row shorter than the screen: some terminals scroll when the last row is written
    to, and the input line was what disappeared."""
    view = demo_view()
    view.side = ["train.py  live", "step 10", "loss 0.5"]
    frame, row, col = tui.compose(view, width, height)
    assert len(frame) == height - 1
    assert all(tui.visible_len(line) <= width for line in frame)
    assert 0 <= row < height and 0 <= col < width
    assert plain(frame[row]).rstrip().endswith("❯")              # the cursor sits on the input line


def test_the_screen_splits_only_when_a_run_is_open_and_there_is_room():
    view = demo_view()
    home = [plain(line) for line in tui.compose(view, 100, 22)[0]]
    assert not any("│" in line for line in home)
    view.side = ["train.py  ● live", "~/experiments", "", "loss  0.0431"]
    wide = [plain(line) for line in tui.compose(view, 100, 22)[0]]
    body = wide[2:]
    assert all("│" in line for line in body)                     # run | agent, all the way down
    assert body[0].split("│")[0].strip() == "train.py  ● live"
    assert any("why is the loss flat?" in line.split("│")[1] for line in body)
    view.side_brief = ["train.py  ● live", "step 10 · loss 0.04"]
    narrow = [plain(line) for line in tui.compose(view, 70, 22)[0]]
    assert not any("│" in line for line in narrow)               # no room: the run is a strip on top
    assert narrow[2].strip() == "train.py  ● live" and "step 10" in narrow[3]


def test_ctrl_o_state_changes_what_the_frame_shows():
    view = demo_view()
    folded = "\n".join(plain(line) for line in tui.compose(view, 90, 24)[0])
    assert "maybe the scheduler" not in folded and "Thought" in folded
    view.expanded = True
    opened = "\n".join(plain(line) for line in tui.compose(view, 90, 24)[0])
    assert "maybe the scheduler" in opened


def test_typing_a_slash_shows_the_matching_commands():
    view = demo_view()
    view.editor.set("/mo")
    frame = "\n".join(plain(line) for line in tui.compose(view, 90, 24)[0])
    assert "/monitor" in frame and "watch a run" in frame and "/help" not in frame.split("❯ /mo")[0][-200:]


def test_a_question_and_a_list_take_over_the_input_block():
    view = demo_view()
    view.question, view.secret = "API key", True
    view.editor.set("secret")
    frame = "\n".join(plain(line) for line in tui.compose(view, 90, 24)[0])
    assert "API key" in frame and "secret" not in frame and "••••••" in frame
    view.secret, view.question = False, "Runs on this machine"
    view.options = [("train.py", "step 10", "live"), ("eval.py", "step 3", "finished")]
    view.option_ids, view.option_index = [0, 1], 1
    view.editor.set("")
    frame = [plain(line) for line in tui.compose(view, 90, 24)[0]]
    chosen = next(line for line in frame if "eval.py" in line)
    assert chosen.strip().startswith("›") and any("train.py" in line and "›" not in line for line in frame)


def test_the_home_bar_says_pulse_and_a_run_says_debug():
    view = demo_view()
    top = plain(tui.compose(view, 80, 24)[0][0])
    assert top.startswith(" PULSE   ~/proj") and "/" not in top.split("~/proj")[0]
    view.area = "DEBUG"
    assert plain(tui.compose(view, 80, 24)[0][0]).startswith(" PULSE / DEBUG")


def test_a_row_is_never_written_into_the_last_column():
    assert tui.cut("─" * 80, 79) == "─" * 79
    assert tui.cut("\033[1mabcdef\033[0m", 3) == "\033[1mabc"
    assert tui.cut("short", 10) == "short"


def test_scrolling_back_shows_older_lines_and_says_so():
    view = tui.View()
    view.entries = [tui.Entry("text", f"line {i}") for i in range(60)]
    newest = "\n".join(plain(line) for line in tui.compose(view, 80, 20)[0])
    assert "line 59" in newest
    view.scroll = 30
    older = "\n".join(plain(line) for line in tui.compose(view, 80, 20)[0])
    assert "line 59" not in older and "scrolled back" in older
    view.scroll = 10 ** 6
    tui.compose(view, 80, 20)
    assert view.scroll < 200                                     # clamped to what exists


# =========================================================================================
# The app as a host
# =========================================================================================

class FakeCli:
    agent_provider = None
    agent_key = None
    agent_model_string = None
    agent_api_base = None
    focus = []
    _project_root = None


@pytest.fixture
def app(tmp_path):
    cli_ = FakeCli()
    cli_._project_root = str(tmp_path)
    return appmod.App(cli_, str(tmp_path))


def text_of(app_):
    return "\n".join(plain(e.text) for e in app_.view.entries)


def on_thread(work):
    box = {}

    def run():
        try:
            box["value"] = work()
        except BaseException as exc:      # noqa: BLE001 -- the test asserts on it
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def wait_for(condition, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def press(app_, *keys):
    for key in keys:
        if len(key) > 1 and key not in ("enter", "esc", "up", "down", "tab", "backspace", "pgup", "pgdn", "end") \
                and not key.startswith(("ctrl+", "paste:")):
            for ch in key:
                app_.on_key(ch)
        else:
            app_.on_key(key)


def test_captured_output_becomes_transcript_lines(app):
    app.feed("first line\nsecond ")
    assert text_of(app) == "first line" and app.view.partial == "second "
    app.feed("half\n\n\n\nthird\n")
    assert text_of(app) == "first line\nsecond half\n\nthird"     # blank runs collapse to one
    app.feed("Planning... /\rPlanning... -\r\033[K\033[32mdone\033[0m\n")
    assert text_of(app).endswith("third\ndone")                  # a carriage return restarts the line
    assert "\033[32m" in app.view.entries[-1].text               # colour survives


def test_a_hosted_question_is_answered_in_the_input_line(app):
    thread, box = on_thread(lambda: app.ask("Model name"))
    assert wait_for(lambda: app.view.question == "Model name")
    press(app, "qwen-9", "enter")
    thread.join(5)
    assert box["value"] == "qwen-9" and app.view.question == ""
    assert "Model name  qwen-9" in text_of(app)


def test_a_secret_answer_is_never_written_to_the_transcript(app):
    thread, box = on_thread(lambda: app.ask("API key", secret=True))
    assert wait_for(lambda: app.view.secret)
    press(app, "sk-very-secret", "enter")
    thread.join(5)
    assert box["value"] == "sk-very-secret" and "sk-very-secret" not in text_of(app)
    assert app.view.secret is False


def test_a_draft_survives_a_question(app):
    press(app, "half typed")
    thread, box = on_thread(lambda: app.ask("Continue? (Y/n)"))
    assert wait_for(lambda: app.view.question != "")
    assert app.view.editor.text == ""
    press(app, "y", "enter")
    thread.join(5)
    assert box["value"] == "y" and app.view.editor.text == "half typed"


def test_ctrl_c_or_esc_cancels_a_question_like_input_would(app):
    for key in ("ctrl+c", "esc"):
        thread, box = on_thread(lambda: app.ask("Anything"))
        assert wait_for(lambda: app.view.question != "")
        press(app, key)
        thread.join(5)
        assert isinstance(box.get("error"), KeyboardInterrupt)


def options(*labels):
    return [ui.Option(label, detail=f"about {label}") for label in labels]


def test_a_hosted_list_is_picked_with_arrows_or_by_typing(app):
    thread, box = on_thread(lambda: app.choose(options("alpha", "beta", "gamma"), title="Pick"))
    assert wait_for(lambda: app.view.options is not None)
    press(app, "down", "down", "enter")
    thread.join(5)
    assert box["value"] == 2 and app.view.options is None

    thread, box = on_thread(lambda: app.choose(options("alpha", "beta", "gamma")))
    assert wait_for(lambda: app.view.options is not None)
    press(app, "bet")
    assert app.view.option_ids == [1]
    press(app, "enter")
    thread.join(5)
    assert box["value"] == 1

    thread, box = on_thread(lambda: app.choose(options("alpha", "beta")))
    assert wait_for(lambda: app.view.options is not None)
    press(app, "esc")
    thread.join(5)
    assert box["value"] is None


def test_a_list_honours_row_shortcuts_and_extra_hotkeys(app):
    rows = [ui.Option("Sign up", key="s"), ui.Option("Log in", key="l")]
    thread, box = on_thread(lambda: app.choose(rows, hotkeys={"c": "create"}))
    assert wait_for(lambda: app.view.options is not None)
    press(app, "l")
    thread.join(5)
    assert box["value"] == 1
    thread, box = on_thread(lambda: app.choose(rows, hotkeys={"c": "create"}))
    assert wait_for(lambda: app.view.options is not None)
    press(app, "c")
    thread.join(5)
    assert box["value"] == "create"


def test_install_makes_the_app_the_host_and_uninstall_puts_everything_back(app):
    before = (sys.stdout, sys.stderr, builtins.input, getpass.getpass)
    app.install()
    try:
        assert ui.host() is app and ui.enabled() and pc._AGENT_OBSERVER is not None
        print("captured by the app")
        sys.stderr.write("and so is stderr\n")
        assert builtins.input is not before[2]
    finally:
        app.uninstall()
    assert (sys.stdout, sys.stderr, builtins.input, getpass.getpass) == before
    assert ui.host() is None and pc._AGENT_OBSERVER is None
    assert "captured by the app\nand so is stderr" in text_of(app)


def test_the_ui_primitives_go_through_the_host(app):
    app.install()
    try:
        thread, box = on_thread(lambda: ui.ask("\033[1mName\033[0m"))
        assert wait_for(lambda: app.view.question == "Name")
        press(app, "x", "enter")
        thread.join(5)
        assert box["value"] == "x"

        thread, box = on_thread(lambda: input("Fix other errors? (y/n) > "))
        assert wait_for(lambda: app.view.question.startswith("Fix other errors?"))
        press(app, "n", "enter")
        thread.join(5)
        assert box["value"] == "n"

        thread, box = on_thread(lambda: ui.confirm("Stop the training run?", default=False))
        assert wait_for(lambda: "Stop the training run?" in app.view.question)
        press(app, "enter")
        thread.join(5)
        assert box["value"] is False

        with ui.Stage("Planning"):
            assert app.view.live == "Planning"
        assert app.view.live == ""
    finally:
        app.uninstall()


def test_tool_progress_is_held_back_unless_a_question_needs_it(app):
    with app.hush() as held:
        app.feed("-> /grep lr\n")
    assert "".join(held) == "-> /grep lr\n" and text_of(app) == ""

    def tool_that_asks():
        with app.hush():
            app.feed("This command deletes files: rm -rf out\n")
            return app.ask("Run it anyway? (y/N)")

    thread, box = on_thread(tool_that_asks)
    assert wait_for(lambda: app.view.question != "")
    assert "rm -rf out" in text_of(app)                          # the context of the question is shown
    press(app, "n", "enter")
    thread.join(5)
    assert box["value"] == "n"


def test_reasoning_and_tools_become_foldable_entries(app):
    app._observe("reasoning", text="first I look at the scheduler")
    app._observe("reasoning", text="")
    app.tool(["GREP: lr"], "train.py:4: lr = 0.3\n")
    kinds = [e.kind for e in app.view.entries]
    assert kinds == ["thinking", "tool"]
    assert app.view.entries[1].calls == ["GREP: lr"]


def test_ctrl_o_scrolling_and_leaving(app):
    press(app, "ctrl+o")
    assert app.view.expanded is True
    press(app, "ctrl+o", "pgup")
    assert app.view.expanded is False and app.view.scroll > 0
    press(app, "end")
    assert app.view.scroll == 0
    press(app, "ctrl+c")
    assert not app.done and "again" in app.view.status
    press(app, "ctrl+c")
    assert app.done


def test_a_job_that_fails_does_not_end_the_app(app):
    def boom(line):
        raise RuntimeError("nope")

    app._handle = boom
    press(app, "anything", "enter")
    assert wait_for(lambda: not app.view.busy and "nope" in text_of(app))
    assert app.view.entries[0].kind == "user" and not app.done


def test_enter_while_working_does_not_start_a_second_job(app):
    gate = threading.Event()
    app._handle = lambda line: gate.wait(5)
    press(app, "first", "enter")
    assert wait_for(lambda: app.view.busy)
    press(app, "second", "enter")
    assert app.view.editor.text == "second" and "still working" in app.view.status
    gate.set()
    assert wait_for(lambda: not app.view.busy)


# =========================================================================================
# The real agent pipeline, hosted
# =========================================================================================

def reply(text, reasoning=None):
    message = types.SimpleNamespace(content=text, reasoning_content=reasoning)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                 usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))


TRAIN = "import time\n\nlr = 0.3\n\ndef train():\n    loss = 1.0\n    for step in range(10):\n        loss = loss * 0.9\n"


@pytest.fixture
def real_app(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text(TRAIN)
    cli_ = appmod._new_cli(str(project), [str(project / "train.py")])
    cli_.agent_provider = next(n for n, info in pc.PROVIDERS.items() if info.get("model") and info.get("env_key"))
    cli_.agent_key = "sk-test"
    cli_._sync_agent_turn = lambda *a, **k: None
    cli_._native_off = True     # these script the text pipeline; the native tool loop has its own tests
    # Nothing in these tests may reach a provider: any call a test did not script (a
    # scheduled audit, say) gets a harmless canned answer.
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: reply(
        '{"status": "ok", "risk": "low", "findings": [], "next_check_minutes": 30}\nLooks fine.'))
    app_ = appmod.App(cli_, str(project))
    yield app_
    app_.uninstall()


def hosted(app_):
    """Make the app the host now. Done inside the test body, not the fixture: pytest swaps
    sys.stdout between a fixture's setup and the test's call."""
    if ui.host() is not app_ or not isinstance(sys.stdout, appmod._Capture):
        if app_._saved:
            app_.uninstall()
        app_.install()
    return app_


def script_model(monkeypatch, replies):
    calls = []

    def completion(**kw):
        # a copy: the pipeline shrinks its own messages once a turn is over
        kw = dict(kw, messages=[dict(m) for m in kw["messages"]])
        calls.append(kw)
        instruction = str(kw["messages"][-1]["content"])
        for needle, answer in replies:
            if needle in instruction:
                if callable(answer):
                    return answer(len([c for c in calls if needle in str(c["messages"][-1]["content"])]))
                return answer
        return reply("ok")

    monkeypatch.setattr(pc.litellm, "completion", completion)
    return calls


def run_line(app_, line):
    hosted(app_)
    press(app_, line, "enter")
    assert wait_for(lambda: app_.view.busy)
    return app_


def test_a_question_shows_thinking_tools_and_the_answer(real_app, monkeypatch):
    def plan(n):
        if n == 1:
            return reply("I need to see where lr is used.\nGREP: lr", "the loss is flat, so look at lr first")
        return reply("The learning rate is fixed at 0.3.\nNO_CHANGES", "line 3 sets it")

    script_model(monkeypatch, [("STEP 1 -- PLAN", plan)])
    run_line(real_app, "why is the loss flat?")
    assert wait_for(lambda: not real_app.view.busy, 20)
    kinds = [e.kind for e in real_app.view.entries]
    assert kinds[0] == "user" and kinds.count("thinking") == 2 and "tool" in kinds
    tool = next(e for e in real_app.view.entries if e.kind == "tool")
    assert tool.calls == ["GREP: lr"] and "lr = 0.3" in tool.output
    transcript = text_of(real_app)
    assert "The learning rate is fixed at 0.3." in transcript
    assert "I need to see where lr is used." in transcript       # what it said before its tools
    assert "GREP: lr" not in transcript                          # ...without the directive lines
    assert "-> /grep" not in transcript and "[tool results]" not in transcript   # folded, not printed
    assert real_app.view.live == ""


def test_a_fix_is_confirmed_in_the_input_line_and_applied(real_app, monkeypatch):
    fix = {"old": ["lr = 0.3"], "new": ["lr = 0.05"], "files": ["train.py"], "create": [],
           "explanation": "Lower the learning rate."}
    script_model(monkeypatch, [
        ("STEP 1 -- PLAN", reply("Plan: change lr to 0.05 in train.py.")),
        ("STEP 2 -- IMPLEMENT", reply(json.dumps(fix))),
        ('"passes"', reply('{"passes": true, "reason": "ok"}')),
        ('"verified"', reply('{"verified": true, "reason": "the file compiles"}')),
    ])
    path = os.path.join(real_app.home_root, "train.py")
    run_line(real_app, "lower the learning rate")
    assert wait_for(lambda: "Apply these changes" in real_app.view.question, 20)
    assert "lr = 0.3" in open(path).read()                       # nothing is written before the answer
    edit = next(e for e in real_app.view.entries if e.kind == "edit")
    assert edit.calls == ["EDIT: train.py"] and "+lr = 0.05" in edit.output   # the diff is in the transcript
    assert edit.open is True                                     # opened, so the person sees what they answer
    press(real_app, "enter")
    assert wait_for(lambda: not real_app.view.busy, 20)
    assert "lr = 0.05" in open(path).read()
    assert edit.text.startswith("applied") and "/undo to revert" in edit.text
    assert [plain(l) for l in tui.entry_lines(edit, 80, False)] == ["Edited train.py  · +1 −1  · applied"]


def test_declining_the_diff_changes_nothing(real_app, monkeypatch):
    fix = {"old": ["lr = 0.3"], "new": ["lr = 0.05"], "files": ["train.py"], "create": [], "explanation": "x"}
    script_model(monkeypatch, [("STEP 1 -- PLAN", reply("Plan: change lr.")),
                               ("STEP 2 -- IMPLEMENT", reply(json.dumps(fix))),
                               ('"passes"', reply('{"passes": true, "reason": "ok"}'))])
    run_line(real_app, "lower the learning rate")
    assert wait_for(lambda: "Apply these changes" in real_app.view.question, 20)
    press(real_app, "n", "enter")
    assert wait_for(lambda: not real_app.view.busy, 20)
    assert "lr = 0.3" in open(os.path.join(real_app.home_root, "train.py")).read()


def test_without_an_agent_a_request_says_how_to_get_one(real_app, monkeypatch):
    real_app.cli.agent_provider = real_app.cli.agent_key = None
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: pytest.fail("no agent, no call"))
    run_line(real_app, "explain train.py")
    assert wait_for(lambda: not real_app.view.busy)
    assert "/agent" in text_of(real_app) and "OpenRouter" in text_of(real_app)


def test_pulse_code_commands_work_inside_the_app(real_app):
    run_line(real_app, "/files")
    assert wait_for(lambda: not real_app.view.busy)
    assert "Focus" in text_of(real_app) and "train.py" in text_of(real_app)
    run_line(real_app, "/help")
    assert wait_for(lambda: not real_app.view.busy)
    assert "/monitor" in text_of(real_app) and "/run" in text_of(real_app)
    run_line(real_app, "/exit")
    assert wait_for(lambda: real_app.done)


def test_the_model_call_reports_reasoning_to_an_observer(real_app, monkeypatch):
    seen = []
    hosted(real_app)
    pc.set_agent_observer(lambda event, **data: seen.append((event, data.get("text"))))
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: reply("answer", "because"))
    assert real_app.cli._call_model("say something") == "answer"
    assert seen == [("reasoning", "because")]

    def broken(event, **data):
        raise RuntimeError("an observer must never break a model call")

    pc.set_agent_observer(broken)
    assert real_app.cli._call_model("again") == "answer"
    assert pc._reasoning_of(types.SimpleNamespace(choices=[types.SimpleNamespace(message={"reasoning": " r "})])) == "r"
    assert pc._reasoning_of(object()) == ""


def test_tool_calls_are_recognised_by_name():
    answer = "NOTE: not a tool\nGREP: loss\n  VIEW: train.py:1-9\nTERMINAL: pytest -q\nDEPGRAPH:\nVERDICT: ok"
    assert pulse_code.tool_calls_in(answer) == ["GREP: loss", "VIEW: train.py:1-9", "TERMINAL: pytest -q", "DEPGRAPH"]


def test_outside_the_app_tool_results_print_as_before(capsys):
    pulse_code._show_tools("GREP: x", "3 matches")
    assert "[tool results]\n3 matches" in capsys.readouterr().out


# =========================================================================================
# Runs: the picker, the run view, going back
# =========================================================================================

def make_run(tmp_path, name="train", loss=0.5, step=10, finished=False, readings=6):
    project = tmp_path / f"{name}_project"
    project.mkdir(exist_ok=True)
    script = project / f"{name}.py"
    script.write_text(TRAIN)
    directory = str(project / ".pulse_stream" / name)
    monitor = Monitor(directory=directory, session_id=name, script_path=str(script),
                      interval=0.0, tensor_interval=0.0)
    for i in range(readings):
        monitor.observe_locals({"loss": loss + (readings - i) * 0.1, "step": step - readings + i + 1})
    monitor.snapshot_state({"finished": finished})
    if finished:
        monitor.close()
    else:
        monitor.writer.close()
    return con._describe_session(directory)


def test_opening_a_run_splits_the_screen_and_aims_the_agent_at_its_project(real_app, tmp_path):
    session = make_run(tmp_path)
    home = real_app.home_root
    real_app._open_run(session)
    view = real_app.view
    assert view.area == "DEBUG" and "train.py" in view.context
    side = "\n".join(plain(line) for line in view.side)
    assert "train.py" in side and "loss" in side and "step 10" in side and "Findings" in side
    assert real_app.cli._project_root == os.path.dirname(session["script"])
    assert os.path.realpath(os.getcwd()) == os.path.realpath(os.path.dirname(session["script"]))
    assert "LIVE TRAINING RUN" in real_app.cli._system_prompt_override
    assert any(c == "/findings" for c, _ in view.commands)
    frame = [plain(line) for line in tui.compose(view, 110, 30)[0]]
    assert all("│" in line for line in frame[2:])

    real_app._close_run()
    assert view.area == "" and view.side is None and real_app.console is None
    assert real_app.cli._project_root == home and os.path.realpath(os.getcwd()) == os.path.realpath(home)
    assert real_app.cli._system_prompt_override == pulse_code.CODE_SYSTEM_PROMPT


def test_a_question_about_an_open_run_carries_its_evidence(real_app, tmp_path, monkeypatch):
    real_app._open_run(make_run(tmp_path, loss=0.25))
    calls = script_model(monkeypatch, [("STEP 1 -- PLAN", reply("It is fine.\nNO_CHANGES"))])
    run_line(real_app, "is it healthy?")
    assert wait_for(lambda: not real_app.view.busy, 20)
    sent = "\n".join(str(m["content"]) for m in calls[0]["messages"])
    assert "EVIDENCE FROM THE RUN (train.py" in sent and "loss" in sent and "Request: is it healthy?" in sent
    assert "LIVE TRAINING RUN" in calls[0]["messages"][0]["content"]


def test_run_commands_print_into_the_transcript(real_app, tmp_path):
    real_app._open_run(make_run(tmp_path))
    for command, expected in (("/vars", "readings"), ("/curve loss", "first"), ("/findings", ""),
                              ("/audits off", "Scheduled audits are off"), ("/source", "lr = 0.3")):
        run_line(real_app, command)
        assert wait_for(lambda: not real_app.view.busy, 10)
        assert expected in text_of(real_app)
    run_line(real_app, "/back")
    assert wait_for(lambda: real_app.background and not real_app.view.busy, 10)
    assert real_app.console is not None                       # still watched, just not shown
    run_line(real_app, "/close")
    assert wait_for(lambda: real_app.console is None, 10)


def test_esc_puts_the_run_in_the_background_and_brings_it_back(real_app, tmp_path):
    real_app._open_run(make_run(tmp_path))
    press(real_app, "esc")
    assert wait_for(lambda: real_app.background and not real_app.view.busy, 10)
    assert real_app.view.area == "" and real_app.view.side is None and real_app.console is not None
    press(real_app, "esc")
    assert wait_for(lambda: not real_app.background and not real_app.view.busy, 10)
    assert real_app.view.area == "DEBUG" and real_app.view.side is not None


def test_the_picker_lists_runs_live_first_and_opens_the_one_chosen(real_app, tmp_path, monkeypatch):
    live = dict(make_run(tmp_path, "train"), status="live")
    done = make_run(tmp_path, "evalrun", finished=True)
    monkeypatch.setattr(con, "discover", lambda *a, **k: [done, live])
    monkeypatch.setattr(con, "unmonitored_python_processes", lambda: [])
    run_line(real_app, "/monitor")
    assert wait_for(lambda: real_app.view.options is not None, 10)
    labels = [label for label, _detail, _tag in real_app.view.options]
    assert labels == ["train.py", "evalrun.py"] and real_app.view.options[0][2] == "live"
    press(real_app, "enter")
    assert wait_for(lambda: real_app.console is not None and not real_app.view.busy, 10)
    assert real_app.session["session_id"] == "train"


def test_the_picker_says_what_to_do_when_there_is_nothing_to_watch(real_app, monkeypatch):
    monkeypatch.setattr(con, "discover", lambda *a, **k: [])
    monkeypatch.setattr(con, "unmonitored_python_processes", lambda: [])
    run_line(real_app, "/runs")
    assert wait_for(lambda: not real_app.view.busy, 10)
    assert "/run train.py" in text_of(real_app)


def test_monitor_by_name_opens_that_run_directly(real_app, tmp_path, monkeypatch):
    session = dict(make_run(tmp_path, "train"), status="live")
    monkeypatch.setattr(con, "discover", lambda *a, **k: [session])
    monkeypatch.setattr(con, "unmonitored_python_processes", lambda: [])
    run_line(real_app, "/monitor train.py")
    assert wait_for(lambda: real_app.console is not None and not real_app.view.busy, 10)
    assert real_app.view.options is None


def test_side_brief_is_two_lines_with_the_essentials(tmp_path, real_app):
    real_app._open_run(make_run(tmp_path, loss=0.25))
    brief = [plain(line) for line in real_app.view.side_brief]
    assert len(brief) == 2 and "train.py" in brief[0] and "step 10" in brief[1] and "loss" in brief[1]


# =========================================================================================
# /run and /restart
# =========================================================================================

REAL_POPEN = appmod.subprocess.Popen


def fake_launches(monkeypatch, process_for):
    """Replace only the launch of a run (`python -m pulse run ...`); everything else that
    starts a process (git, the tools) is left alone."""
    started = []

    def popen(argv, *args, **kw):
        if isinstance(argv, (list, tuple)) and len(argv) > 3 and argv[1] == "-c" and argv[3] == "run":
            started.append((list(argv), kw))
            return process_for()
        return REAL_POPEN(argv, *args, **kw)

    monkeypatch.setattr(appmod.subprocess, "Popen", popen)
    return started


class FakeProcess:
    pid = 4242

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.code = 0
        return 0


def test_run_starts_the_script_under_pulse_and_opens_it(real_app, tmp_path, monkeypatch):
    session = dict(make_run(tmp_path, "train"), status="live", pid=4242)
    script = os.path.join(real_app.home_root, "train.py")
    session["script"] = script
    started = fake_launches(monkeypatch, FakeProcess)
    monkeypatch.setattr(con, "discover", lambda *a, **k: [session])
    run_line(real_app, "/run train.py --epochs 3")
    assert wait_for(lambda: real_app.console is not None and not real_app.view.busy, 15)
    argv, kw = started[0]
    assert argv[1] == "-c" and "runpy.run_module('pulse'" in argv[2]
    assert argv[3:6] == ["run", "--stream", "--again"] and argv[6:] == [script, "--epochs", "3"]
    assert kw["cwd"] == real_app.home_root
    if os.name == "nt":
        assert kw["creationflags"] & appmod.subprocess.CREATE_NO_WINDOW
    else:
        assert kw["start_new_session"] is True
    assert kw["env"]["PULSE_NONINTERACTIVE"] == "1"
    info = real_app.launched["train"]
    assert info["argv"] == [script, "--epochs", "3"] and os.path.dirname(info["log"]).endswith("app-runs")

    # /restart stops it and starts the same command again
    controls = []
    real_app.console.send_control = lambda action, **f: controls.append(action)
    run_line(real_app, "/restart")
    assert wait_for(lambda: len(started) == 2 and not real_app.view.busy, 15), text_of(real_app)
    time.sleep(0.2)
    assert len(started) == 2
    assert controls == ["stop"] and started[1][0] == started[0][0]


def test_run_with_a_missing_script_or_a_run_that_never_reports(real_app, monkeypatch):
    run_line(real_app, "/run nope.py")
    assert wait_for(lambda: not real_app.view.busy, 10)
    assert "No such script" in text_of(real_app)

    process = FakeProcess()
    process.code = 1                                             # it died at once
    fake_launches(monkeypatch, lambda: process)
    monkeypatch.setattr(con, "discover", lambda *a, **k: [])
    run_line(real_app, "/run train.py")
    assert wait_for(lambda: not real_app.view.busy, 15)
    assert "exited with code 1" in text_of(real_app) and real_app.console is None


def test_restart_leaves_a_live_run_started_elsewhere_alone(real_app, tmp_path):
    real_app._open_run(make_run(tmp_path))
    real_app._status = "live"
    run_line(real_app, "/restart")
    assert wait_for(lambda: not real_app.view.busy, 10)
    assert "started outside Pulse's app and is still running" in text_of(real_app)


# =========================================================================================
# Where the app is used, and where it is not
# =========================================================================================

def test_the_app_needs_a_real_terminal_and_can_be_switched_off(monkeypatch):
    assert appmod.usable() is False                              # pytest: no terminal
    monkeypatch.setattr(ui, "enabled", lambda: True)
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((120, 40)))
    assert appmod.usable() is True
    monkeypatch.setenv("PULSE_CLASSIC", "1")
    assert appmod.usable() is False
    monkeypatch.delenv("PULSE_CLASSIC")
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((30, 8)))
    assert appmod.usable() is False                              # too small to draw in


def test_bare_pulse_and_pulse_code_open_the_app_on_a_terminal(monkeypatch):
    opened = []
    monkeypatch.setattr(appmod, "usable", lambda: True)
    monkeypatch.setattr(appmod, "run", lambda paths=(), yes=False, root=None: opened.append((list(paths), yes)) or 0)
    assert entry.main([]) == 0
    assert entry.main(["code", "-y"]) == 0
    assert opened == [([], False), ([], True)]


def test_one_shot_and_no_terminal_keep_the_plain_paths(monkeypatch):
    monkeypatch.setattr(appmod, "run", lambda *a, **k: pytest.fail("the app must not open"))
    one_shot = []
    monkeypatch.setattr(pulse_code, "run", lambda paths, prompt=None, yes=False, root=None: one_shot.append(prompt) or 0)
    monkeypatch.setattr(appmod, "usable", lambda: True)
    assert entry.main(["code", "-p", "add a flag"]) == 0         # -p: one request and out
    monkeypatch.setattr(appmod, "usable", lambda: False)
    assert entry.main(["code"]) == 0
    assert one_shot == ["add a flag", None]
    seen = []
    monkeypatch.setattr(con, "main", lambda argv=None: seen.append(argv) or 0)
    assert entry.main([]) == 0 and seen == [[]]                  # no terminal: the run console


def test_watching_a_run_opens_the_app_on_it(monkeypatch, tmp_path):
    session = make_run(tmp_path)
    opened = []
    monkeypatch.setattr(appmod, "usable", lambda: True)
    monkeypatch.setattr(appmod, "open_run", lambda s, model="", monitor=None: opened.append((s["session_id"], model)) or 0)
    from pulse.pulse_brain import build_litellm_agent
    assert con.run_console(session, [session], agent=build_litellm_agent("anthropic/x-y")) == 0
    assert con.run_console(session, [session]) == 0
    assert opened == [("train", "anthropic/x-y"), ("train", "")]


# =========================================================================================
# Starting up must stay quick
# =========================================================================================

def test_scanning_a_home_directory_never_asks_git_and_stays_bounded(tmp_path, monkeypatch):
    """A home directory is quite often a git repository with everything untracked; listing
    it with git walked the whole tree (seconds). It is not a project: walk a little, stop."""
    home = tmp_path / "home"
    for i in range(30):
        (home / f"dir{i}").mkdir(parents=True)
        (home / f"dir{i}" / "a.py").write_text("x = 1\n")
    (home / ".git").mkdir()
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)
    monkeypatch.setattr(pulse_code.subprocess, "run", lambda *a, **k: pytest.fail("git must not list a home directory"))
    files = pulse_code.scan_project(str(home), limit=10)
    assert len(files) == 10 and all(f.endswith("a.py") for f in files)


def test_a_project_scan_stops_once_it_has_enough(tmp_path, monkeypatch):
    for i in range(50):
        (tmp_path / f"d{i:02d}").mkdir()
        (tmp_path / f"d{i:02d}" / "m.py").write_text("")
    monkeypatch.setattr(pulse_code, "_SCAN_MAX_DIRS", 5)
    monkeypatch.setattr(pulse_code.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=b""))
    assert len(pulse_code.scan_project(str(tmp_path), limit=100)) <= 5


# =========================================================================================
# A watched run is logged to the dashboard
# =========================================================================================

class FakeCloud:
    def __init__(self, monkeypatch):
        self.created, self.patches = [], []
        monkeypatch.setattr(cloud, "create_debug_session", self.create)
        monkeypatch.setattr(cloud, "patch_debug_session", self.patch)
        monkeypatch.setattr(cloud, "collect_environment_info", lambda: {"python": "3.12"})
        monkeypatch.setattr(cloud, "current_git_commit_sha", lambda cwd=None: "abc123")

    def create(self, project_id, user_id, git_commit_sha=None):
        self.created.append((project_id, user_id, git_commit_sha))
        return "row-1"

    def patch(self, session_id, fields, timeout=5):
        decoded = {k: ([cloud.decode_entry(e) for e in v] if isinstance(v, list) else v) for k, v in fields.items()}
        self.patches.append((session_id, decoded))

    def field(self, name):
        for _sid, body in reversed(self.patches):
            if name in body:
                return body[name]
        return None


def settle(log):
    worker = log._worker
    if worker is not None:
        worker.join(5)


def finding(check, variable, severity, message):
    return types.SimpleNamespace(check=check, variable=variable, severity=severity, message=message)


def test_a_watched_run_gets_its_own_dashboard_row(monkeypatch):
    fake = FakeCloud(monkeypatch)
    cli = types.SimpleNamespace(user_id="u1", project_id="p1")
    session = {"session_id": "run1", "script": "/p/train.py", "pid": 7, "started": time.time() - 100,
               "last_seen": time.time()}
    log = appmod.RunLog(cli, session, "/p")
    log.start()
    settle(log)
    assert fake.created == [("p1", "u1", "abc123")]
    first = fake.field("telemetry")[0]
    assert first["script"] == "/p/train.py" and first["python"] == "3.12" and first["mode"] == "stream"

    state = {"step": 40, "histories": {"loss": [1.0, 0.5], "lr": [0.1]},
             "findings": [finding("plateau", "loss", "warning", "loss has barely moved")],
             "tensors": {"weights": {"shape": [32, 16], "dtype": "float32"}}, "gaps": 2}
    log.tick(state, "live", [], session)
    settle(log)
    incidents = fake.field("incidents")
    assert incidents[-1]["kind"] == "finding" and incidents[-1]["severity"] == "warning" and incidents[-1]["step"] == 40
    snapshot = fake.field("telemetry")[-1]
    assert snapshot["step"] == 40 and snapshot["loss"] == 0.5 and snapshot["lr"] == 0.1
    assert snapshot["status"] == "live" and snapshot["gaps"] == 2
    assert snapshot["findings"][0]["severity"] == "warning"
    assert snapshot["tensors"]["weights"]["shape"] == [32, 16]
    assert 95 <= fake.field("uptime_seconds") <= 105

    log.tick(state, "live", [], session)              # the same finding again: no new incident
    settle(log)
    assert sum(1 for i in fake.field("incidents") if i["kind"] == "finding") == 1

    crash = {"event": "crash", "exception": "ZeroDivisionError: x", "traceback": "Traceback...\nZeroDivisionError: x"}
    log.tick(state, "crashed", [crash], session)
    settle(log)
    assert fake.field("error_tracebacks") == ["Traceback...\nZeroDivisionError: x"]
    kinds = [i["kind"] for i in fake.field("incidents")]
    assert "crash" in kinds and "run_crashed" in kinds

    log.agent_turn("why?", "because", fix_applied={"explanation": "lower lr", "files": ["train.py"]})
    settle(log)
    assert fake.field("agent_logs")[-1]["question"] == "why?"
    assert fake.field("incidents")[-1]["kind"] == "fix_applied"
    log.close()
    assert all(sid == "row-1" for sid, _body in fake.patches)


def test_without_a_signed_in_account_nothing_is_sent(monkeypatch):
    fake = FakeCloud(monkeypatch)
    log = appmod.RunLog(types.SimpleNamespace(user_id=None, project_id=None), {"session_id": "r"}, "/p")
    log.start()
    log.tick({"step": 1, "histories": {}, "findings": []}, "live", [], {})
    log.agent_turn("q", "a")
    log.close()
    assert fake.created == [] and fake.patches == []


def test_an_offline_moment_keeps_the_entries_for_the_next_send(monkeypatch):
    fake = FakeCloud(monkeypatch)
    calls = {"n": 0}

    def flaky(session_id, fields, timeout=5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cloud.SupabaseError("offline")
        fake.patch(session_id, fields)

    monkeypatch.setattr(cloud, "patch_debug_session", flaky)
    log = appmod.RunLog(types.SimpleNamespace(user_id="u1", project_id=None), {"session_id": "r", "script": "/p/t.py"}, "/p")
    log.start()
    settle(log)
    assert fake.patches == []
    log.incident("stopped", "x")
    settle(log)
    assert fake.field("telemetry") and fake.field("incidents")[-1]["kind"] == "stopped"


def test_the_app_logs_the_open_run_and_its_agent_turns(real_app, tmp_path, monkeypatch):
    fake = FakeCloud(monkeypatch)
    real_app.cli.user_id, real_app.cli.project_id = "u1", "p1"
    original = []
    real_app.cli._sync_agent_turn = lambda q, a, traceback_signature=None, fix_applied=None: original.append(q)
    real_app._open_run(make_run(tmp_path))
    assert real_app.runlog is not None
    settle(real_app.runlog)
    assert fake.created and fake.field("telemetry")[0]["pulse_session"] == "train"
    real_app.cli._sync_agent_turn("is it healthy?", "yes")
    settle(real_app.runlog)
    assert original == ["is it healthy?"]                 # the app's own session still gets it
    assert fake.field("agent_logs")[-1]["question"] == "is it healthy?"    # and so does the run's row
    log = real_app.runlog
    real_app._close_run(quiet=True)
    assert real_app.runlog is None and log._closed
    real_app.cli._sync_agent_turn("later", "x")
    assert original == ["is it healthy?", "later"] and fake.field("agent_logs")[-1]["question"] == "is it healthy?"


def test_no_dashboard_row_when_not_signed_in(real_app, tmp_path, monkeypatch):
    fake = FakeCloud(monkeypatch)
    real_app.cli.user_id = None
    real_app._open_run(make_run(tmp_path))
    assert real_app.runlog is None and fake.created == []
    real_app._close_run(quiet=True)



# =========================================================================================
# Clicking: folded entries open and shut with the mouse
# =========================================================================================

def test_mouse_reports_become_clicks_and_wheel_keys():
    keys, rest = tui.parse_keys("\033[<0;12;5M\033[<0;12;5m\033[<64;3;3M\033[<65;3;3M")
    assert keys == ["press:11:4", "release:11:4", "wheel:up:2:2", "wheel:down:2:2"]
    assert tui.parse_keys("\033[<0;12")[1] == "\033[<0;12"            # still arriving


def test_a_click_on_a_folded_entry_opens_it_and_a_click_shuts_it(app):
    tool = tui.Entry("tool", calls=["GREP: lr"], output="a\nb\nc")
    app.view.entries = [tui.Entry("user", "q"), tool, tui.Entry("text", "answer")]
    frame, _r, _c = tui.compose(app.view, 90, 24)
    row = next(r for r, e in app.view.row_entries.items() if e is tool)
    assert "3 lines" in PLAIN.sub("", frame[row])
    app.on_key(f"click:5:{row}")
    frame, _r, _c = tui.compose(app.view, 90, 24)
    shown = [PLAIN.sub("", line).strip() for line in frame]
    assert any(line.endswith("⎿ a") for line in shown) and tool.open is True
    app.on_key(f"click:5:{row}")
    assert tool.open is False and "3 lines" in "\n".join(PLAIN.sub("", l) for l in tui.compose(app.view, 90, 24)[0])
    app.on_key("ctrl+o")                                   # Ctrl+O speaks for every entry again
    assert tool.open is None and app.view.expanded


def test_the_wheel_scrolls_the_transcript(app):
    app.view.entries = [tui.Entry("text", f"line {i}") for i in range(80)]
    app.on_key("wheel:up")
    assert app.view.scroll == 3
    app.on_key("wheel:down")
    assert app.view.scroll == 0


def test_a_long_pasted_request_folds_to_its_first_lines():
    entry = tui.Entry("user", "\n".join(f"line {i}" for i in range(20)))
    lines = [PLAIN.sub("", line) for line in tui.entry_lines(entry, 60, False)]
    assert len(lines) == tui.USER_FOLD_LINES and "more lines" in lines[-1] and entry.foldable()
    assert len(tui.entry_lines(entry, 60, True)) == 20


def test_a_shut_tool_call_stays_on_one_line():
    entry = tui.Entry("tool", calls=["TERMINAL: " + "x" * 200], output="")
    assert len(tui.entry_lines(entry, 40, False)) == 1
    assert len(tui.entry_lines(entry, 40, True)) > 1


def test_the_agents_words_and_its_thinking_look_different(monkeypatch):
    monkeypatch.setattr(ui, "color_enabled", lambda: True)
    say = [PLAIN.sub("", l) for l in tui.entry_lines(tui.Entry("say", "The lr is 0.3."), 40, False)]
    think = tui.entry_lines(tui.Entry("thinking", "maybe lr"), 40, True)
    assert say == ["The lr is 0.3."]                                   # the agent's words: plain
    assert PLAIN.sub("", think[0]).startswith("Thought") and "\033[3m" in think[0]   # thinking: orange italic
    assert "\033[38;5;208m" in think[0]
    tool = tui.entry_lines(tui.Entry("tool", calls=["GREP: lr"], output=""), 40, False)
    assert tool[0].startswith("\033[38;5;248m\033[1m")               # a command: grey, bold
    note = tui.entry_lines(tui.Entry("text", "[Pulse Code] applied"), 40, False)
    assert "\033[2m" in note[0]                                        # Pulse's own output: quiet


def test_an_opened_entry_stays_where_it_was_clicked(app):
    tool = tui.Entry("tool", calls=["VIEW: train.py:1-40"], output="\n".join(f"line {i}" for i in range(30)))
    app.view.entries = [tui.Entry("user", "q"), tool] + [tui.Entry("text", f"after {i}") for i in range(6)]
    frame, _r, _c = tui.compose(app.view, 90, 24)
    row = next(r for r, e in app.view.row_entries.items() if e is tool)
    app.on_key(f"click:5:{row}")
    frame, _r, _c = tui.compose(app.view, 90, 24)
    assert app.view.row_entries.get(row) is tool                    # still on the same row
    assert "Read train.py:1-40" in PLAIN.sub("", frame[row])


def test_a_click_in_the_split_view_keeps_the_entry_on_its_row(app):
    """The run pane takes columns; the re-scroll must wrap to the same width the frame did."""
    tool = tui.Entry("tool", calls=["GREP: lr", "VIEW: train.py:1-12"], output="\n".join(f"line {i}" for i in range(25)))
    app.view.side = [f"side {i}" for i in range(30)]
    app.view.entries = [tui.Entry("user", "why is the loss flat"), tui.Entry("say", "I need to see how lr is used."), tool,
                        tui.Entry("say", "x" * 66 + " " + "y" * 20), tui.Entry("text", "w" * 67)]
    frame, _r, _c = tui.compose(app.view, 110, 22)
    row = next(r for r, e in app.view.row_entries.items() if e is tool)
    app.on_key(f"click:50:{row}")
    frame, _r, _c = tui.compose(app.view, 110, 22)
    assert app.view.row_entries.get(row) is tool
    assert "Searched for lr" in PLAIN.sub("", frame[row])


# ---------------------------------------------------------------- a change to the files

DIFF = ("\033[1mmodified: train.py\033[0m\n\033[2m@@ -1,3 +1,3 @@\033[0m\n import x\n\033[31m-lr = 0.3\033[0m\n"
        "\033[32m+lr = 0.05\033[0m\n rng = 1\n")


def test_a_change_folds_to_one_line_and_opens_to_its_diff(app):
    app.edit(["EDIT: train.py"], DIFF)
    entry = app.view.entries[-1]
    assert entry.kind == "edit" and app._edit_pending is entry
    assert [plain(l) for l in tui.entry_lines(entry, 80, False)] == ["Edited train.py  · +1 −1"]
    app.edit_status("approved by deepseek-chat", detail="deepseek/deepseek-chat: does what was asked")
    app.edit_status("applied", detail="/undo to revert (commit abc123)", final=True)
    assert app._edit_pending is None
    assert [plain(l) for l in tui.entry_lines(entry, 80, False)] == ["Edited train.py  · +1 −1  · approved by deepseek-chat · applied"]
    opened = [plain(l) for l in tui.entry_lines(entry, 80, True)]
    assert opened[0].startswith("Edited train.py")
    assert "  modified: train.py" in opened and "  -lr = 0.3" in opened and "  +lr = 0.05" in opened
    assert opened[-2:] == ["  deepseek/deepseek-chat: does what was asked", "  /undo to revert (commit abc123)"]
    assert entry.foldable()


def test_a_status_with_no_change_on_the_table_is_a_note(app):
    app.edit_status("applied", final=True)
    assert app.view.entries[-1].kind == "note"


def test_show_change_and_change_status_print_on_a_plain_terminal(capsys):
    ui.set_host(None)
    ui.show_change(["EDIT: train.py"], "modified: train.py\n+lr = 0.05")
    ui.change_status("applied", color=None, terminal_text="✓ 1 file changed")
    out = capsys.readouterr().out
    assert "+lr = 0.05" in out and "✓ 1 file changed" in out and "applied\n" not in out


def test_the_native_loop_puts_an_edits_fate_on_its_diff_line():
    from pulse import pulse_code_agent as agent

    class Host:
        def __init__(self):
            self._edit_pending = object()
            self.statuses, self.tools = [], []

        def edit_status(self, text, detail=None, final=False):
            self.statuses.append((text, final))

        def tool(self, calls, output):
            self.tools.append(calls)

    host = Host()
    ui.set_host(host)
    try:
        agent._show("edit_file", {"path": "train.py"}, "train.py  (+1 -1 lines)")
        agent._show("edit_file", {"path": "train.py"}, "NOT APPLIED -- the user declined this change")
        host._edit_pending = None
        agent._show("edit_file", {"path": "train.py"}, "NOT APPLIED -- the change would break the file")
        agent._show("grep", {"pattern": "lr"}, "3 matches")
    finally:
        ui.set_host(None)
    assert host.statuses == [("applied", True), ("not applied", True)]
    assert host.tools == [["EDIT_FILE: train.py"], ["GREP: lr"]]


def test_pipeline_colours_are_dropped_but_an_error_stays_red(monkeypatch):
    monkeypatch.setattr(ui, "color_enabled", lambda: True)
    green = tui.Entry("text", "\033[32m[Pulse] all good\033[0m")
    red = tui.Entry("text", "\033[31m[Pulse] ⚠ lint FAILED\033[0m")
    g_line, = tui.entry_lines(green, 80, False)
    r_line, = tui.entry_lines(red, 80, False)
    assert "\033[32m" not in g_line and g_line.startswith("\033[2m")
    assert "\033[31m" in r_line and "\033[2m" not in r_line


def test_only_a_live_tag_is_orange_in_a_picker(app, monkeypatch):
    monkeypatch.setattr(ui, "color_enabled", lambda: True)
    app.view.options = [("train.py", "step 10", "live"), ("train.py", "step 9", "ended 2h ago"),
                        ("old.py", "", "not under Pulse")]
    app.view.option_ids = [0, 1, 2]
    frame, _r, _c = tui.compose(app.view, 90, 20)
    rows = [ln for ln in frame if "train.py" in PLAIN.sub("", ln) or "old.py" in PLAIN.sub("", ln)]
    assert "\033[38;5;208m  live" in rows[0]
    assert "\033[2m  ended 2h ago" in rows[1] and "\033[38;5;208m  ended" not in rows[1]
    assert "\033[2m  not under Pulse" in rows[2]


def test_machinery_lines_stack_and_paragraphs_breathe(app):
    app.view.entries = [tui.Entry("user", "q"), tui.Entry("thinking", "a b"), tui.Entry("say", "Plan"),
                        tui.Entry("edit", calls=["EDIT: t.py"], output="+x"), tui.Entry("tool", calls=["TERMINAL: ls"], output="a"),
                        tui.Entry("note", "Verified"), tui.Entry("thinking", "c d"), tui.Entry("say", "Done")]
    lines = [plain(l) for l in tui.transcript_lines(app.view, 80)]
    assert lines == ["❯ q", "", "Thought (2 words)", "", "Plan", "Edited t.py  · +1 −0", "Ran ls  · 1 line", "Verified",
                     "", "Thought (2 words)", "", "Done"]          # a thought opens a new step: air before it


# ---------------------------------------------------------------- auto_track() opens the app

class FakeMonitor:
    def __init__(self, tmp_path):
        self.session_id = "s1"
        self.directory = str(tmp_path / "stream")
        self.script_path = str(tmp_path / "train.py")


def _loop_until_done(app_):
    """Stands in for App.loop: the screen is up, output is captured, until done."""
    app_.install()
    app_.up.set()
    try:
        while not app_.done:
            time.sleep(0.01)
    finally:
        app_.uninstall()


def test_auto_track_opens_the_app_beside_the_script(tmp_path, monkeypatch, capsys):
    import atexit
    from pulse import pulse_monitor
    registered = []
    monkeypatch.setattr(appmod, "_register_end_hook", lambda fn: registered.append(fn))
    monkeypatch.setattr(appmod, "usable", lambda: True)
    monkeypatch.setattr(appmod, "_new_cli", lambda root, focus, review=True, chdir=True: FakeCli())
    monkeypatch.setattr(appmod, "_pick_agent_quietly", lambda cli, model="": None)
    monkeypatch.setattr(appmod.App, "loop", _loop_until_done)
    monkeypatch.setattr(appmod.App, "_open_run", lambda self, session, monitor=None: self.note("opened"))
    monkeypatch.setattr(appmod, "signal", types.SimpleNamespace(signal=lambda *a: None, SIGINT=2, default_int_handler=None,
                                                                getsignal=lambda s: None))
    detached = []
    monkeypatch.setattr(pulse_monitor, "detach", lambda: detached.append(True))
    cwd = os.getcwd()
    monitor = FakeMonitor(tmp_path)

    app_ = appmod.watch_in_process(monitor)
    assert app_ is not None and app_.hosting is monitor
    assert os.getcwd() == cwd                                  # the script's working directory is its own
    assert app_.up.is_set() and ui.host() is app_              # the script goes on with its output captured
    print("step 0  loss 1.0")                                  # what the script prints goes to the run pane
    assert wait_for(lambda: "step 0  loss 1.0" in app_.run_output)
    assert "step 0" not in text_of(app_)
    assert "train.py is running here" in text_of(app_)

    # the script ends: the run is marked finished, the app stays until the person leaves
    at_exit, = registered
    leaver = threading.Thread(target=lambda: (wait_for(lambda: "has finished" in text_of(app_)), setattr(app_, "done", True)))
    leaver.start()
    at_exit()
    leaver.join()
    assert detached == [True] and app_.done and ui.host() is None
    assert "Left the app" not in capsys.readouterr().out        # the script was over, not left


def test_leaving_the_app_gives_the_terminal_back_and_the_run_goes_on(tmp_path, monkeypatch, capsys):
    import atexit
    monkeypatch.setattr(appmod, "_register_end_hook", lambda fn: None)
    monkeypatch.setattr(appmod, "usable", lambda: True)
    monkeypatch.setattr(appmod, "_new_cli", lambda root, focus, review=True, chdir=True: FakeCli())
    monkeypatch.setattr(appmod, "_pick_agent_quietly", lambda cli, model="": None)
    monkeypatch.setattr(appmod.App, "loop", _loop_until_done)
    monkeypatch.setattr(appmod.App, "_open_run", lambda self, session, monitor=None: None)
    monkeypatch.setattr(appmod, "signal", types.SimpleNamespace(signal=lambda *a: None, SIGINT=2, default_int_handler=None,
                                                                getsignal=lambda s: None))
    app_ = appmod.watch_in_process(FakeMonitor(tmp_path))
    app_.done = True                                           # Ctrl+C twice
    assert wait_for(lambda: ui.host() is None)
    time.sleep(0.05)
    assert "Left the app; train.py goes on" in capsys.readouterr().out


def test_without_a_terminal_auto_track_stays_with_the_line_by_line_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "usable", lambda: False)
    assert appmod.watch_in_process(FakeMonitor(tmp_path)) is None
    from pulse import pulse as core
    assert core._default_mode() == "cli"
    monkeypatch.setattr(appmod, "usable", lambda: True)
    assert core._default_mode() == "app"
    monkeypatch.setenv("PULSE_MODE", "app")
    assert core._app_mode_requested(None)
    monkeypatch.delenv("PULSE_MODE")
    assert core._app_mode_requested("app") and not core._app_mode_requested("cli")


def test_app_mode_falls_back_to_cli_when_the_app_cannot_open(monkeypatch):
    from pulse import pulse as core
    calls = []
    monkeypatch.setattr(core, "_start_app", lambda frame, interval: calls.append("app") or None)
    monkeypatch.setattr(core, "_distributed_info", lambda: (0, 0, 1, "k"))
    monkeypatch.setattr(core, "_determine_mode", lambda mode: calls.append(("mode", mode)) or (_ for _ in ()).throw(RuntimeError("stop here")))
    with pytest.raises(RuntimeError, match="stop here"):
        core._auto_track_session(sys._getframe(), None, 1.0, None, None, "app")
    assert calls == ["app", ("mode", "cli")]


def test_the_ctrl_c_hint_says_the_training_goes_on(app):
    app.hosting = object()
    app.on_key("ctrl+c")
    assert "training goes on" in app.view.status and "/stop" in app.view.status
    app.on_key("ctrl+c")
    assert app.done


def test_what_the_agent_says_beside_its_tools_is_kept_with_them(app):
    """"I need to see how lr is used" is for the agent more than the person: it goes with
    the calls, shown when the line is opened, after the output."""
    from pulse import pulse_code
    ui.set_host(app)
    pulse_code._show_tools("I need to see how lr is used.\nGREP: lr\nVIEW: train.py:1-12", "3 matches")
    entry = app.view.entries[-1]
    assert entry.kind == "tool" and entry.text == "I need to see how lr is used."
    assert [e.kind for e in app.view.entries] == ["tool"]                      # no separate paragraph
    folded = [plain(l) for l in tui.entry_lines(entry, 80, False)]
    assert folded == ["Searched for lr · Read train.py:1-12  · 1 line"]
    opened = [plain(l) for l in tui.entry_lines(entry, 80, True)]
    assert opened[-1] == "  I need to see how lr is used." and "  ⎿ 3 matches" in opened


def test_the_native_loop_keeps_its_words_with_the_first_call(app):
    from pulse import pulse_code_agent as agent
    ui.set_host(app)
    state = types.SimpleNamespace(said="Let me look at the scheduler.")
    agent._show("grep", {"pattern": "lr"}, "3 matches", state)
    agent._show("read_file", {"path": "train.py"}, "...", state)
    first, second = app.view.entries[-2:]
    assert first.text == "Let me look at the scheduler." and second.text == "" and state.said == ""


# ---------------------------------------------------------------- the hosted run's output

def test_the_hosted_runs_output_goes_under_its_figures_not_into_the_conversation(app, monkeypatch):
    app.hosting = types.SimpleNamespace(session_id="s1")
    app.session = {"session_id": "s1", "directory": ""}
    app.console = types.SimpleNamespace(snapshot=lambda: {"step": 3, "histories": {}, "findings": [], "finished": False},
                                        brain=types.SimpleNamespace(events=[]), workdir="/x")
    app.feed("step 0  loss 1.0\nstep 10  loss 0.9\nstep 20")          # the script, on the main thread
    assert list(app.run_output) == ["step 0  loss 1.0", "step 10  loss 0.9"] and app._run_partial == "step 20"
    assert app.view.entries == []

    def pulse_speaks():
        app.feed("[Pulse] a finding\n")                               # Pulse's own threads: the conversation
    worker = threading.Thread(target=pulse_speaks, name="pulse-app-job")
    worker.start(); worker.join()
    assert [e.text for e in app.view.entries] == ["[Pulse] a finding"]

    monkeypatch.setattr(appmod, "side_lines", lambda *a, **k: ["train.py  ● live", "step 3", "", "Agent", "audits off"])
    app._height, app._total_w = 16, 120
    app._refresh_side(force=True)
    side = [plain(l) for l in app.view.side]
    assert side[5:] == ["", "Output", "step 0  loss 1.0", "step 10  loss 0.9", "step 20"]
    for i in range(30):
        app.feed(f"step {30 + i}\n")
    app._side_at = 0
    app._refresh_side(force=True)
    side = [plain(l) for l in app.view.side]
    assert side[6] == "Output" and len(side) == tui.body_height(16) and side[-1] == "step 59"   # all the pane has room for
    frame, _r, _c = tui.compose(app.view, 120, 16)
    assert "step 59" in plain(frame[-1])                               # the newest line is on screen



# ---------------------------------------------------------------- a crash

def _crashed_app(app, agent=True):
    app.session = {"session_id": "s1", "directory": "", "script": "/p/train.py"}
    app.console = types.SimpleNamespace(workdir="/p", brain=types.SimpleNamespace(events=[
        {"event": "crash", "exception": "ValueError: matmul mismatch", "traceback": "Traceback...\nValueError: matmul mismatch"}]))
    app.cli.agent_provider, app.cli.agent_key = ("DeepSeek", "sk") if agent else (None, None)
    started = []
    app._start = lambda work, *args: started.append(args)
    return started


def test_a_crash_starts_the_agent_on_it_once(app):
    started = _crashed_app(app)
    app._on_crash()
    app._on_crash()                                              # the next screen refresh: not again
    assert len(started) == 1 and "crashed with ValueError: matmul mismatch" in started[0][0]
    kinds = [(e.kind, plain(e.text)) for e in app.view.entries]
    assert kinds == [("error", "train.py crashed: ValueError: matmul mismatch."),
                     ("user", "Why did train.py crash? Fix it.")]


def test_a_crash_while_the_agent_is_busy_waits_for_it(app):
    started = _crashed_app(app)
    app.view.busy = True
    app._on_crash()
    assert started == [] and app._crash_pending == "train.py"
    app.view.busy = False
    app._run_job(lambda: None, ())                               # the job that was running ends
    assert len(started) == 1 and app._crash_pending is None


def test_a_crash_without_an_agent_says_how_to_get_one(app):
    started = _crashed_app(app, agent=False)
    app._on_crash()
    assert started == [] and "/agent" in text_of(app) and "ValueError" in text_of(app)


def test_an_audit_shows_its_words_not_its_json(app):
    text = 'Some prose.\n```json\n{"status": "ok", "risk": "low", "findings": [], "next_check_minutes": 15}\n```\nThe loss is flat near 0.5.'
    assert appmod.audit_words(text) == "Some prose. The loss is flat near 0.5."



def test_the_app_outlives_the_script_while_the_agent_can_still_work(tmp_path):
    """The hook that keeps the app open after the script ends runs early in shutdown: a
    model call from there (thread pools, a library imported late) must still work. A plain
    atexit hook failed with "can't register atexit after shutdown"."""
    import subprocess
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys, threading\n"
        "from pulse import pulse_app\n"
        "def end():\n"
        "    with pulse_app._shutdown_held():\n"
        "        def work():\n"
        "            import concurrent.futures as cf\n"
        "            with cf.ThreadPoolExecutor(1) as ex:\n"
        "                print('pool', ex.submit(lambda: 42).result(), flush=True)\n"
        "            threading._register_atexit(lambda: None)\n"
        "            print('register ok', flush=True)\n"
        "        t = threading.Thread(target=work, daemon=True); t.start(); t.join()\n"
        "pulse_app._register_end_hook(end)\n"
        "raise ValueError('the run crashed')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    r = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=120, env=env)
    assert "pool 42" in r.stdout and "register ok" in r.stdout, r.stdout + r.stderr
    assert "ValueError: the run crashed" in r.stderr


# ---------------------------------------------------------------- selecting and copying

class FakeScreen:
    def __init__(self):
        self.mouse, self.copied = True, []

    def set_mouse(self, on):
        self.mouse = on

    def copy(self, text):
        self.copied.append(text)


def test_mouse_off_gives_the_mouse_back_and_is_remembered(app, monkeypatch):
    from pulse import pulse_settings as settings
    app._screen = FakeScreen()
    app._command("/mouse off")
    assert app.mouse is False and app._screen.mouse is False and settings.get("mouse") == "off"
    assert "your terminal selects" in text_of(app)
    app._command("/mouse")                                   # toggles back
    assert app.mouse is True and app._screen.mouse is True and settings.get("mouse") == "on"


def test_the_screen_turns_mouse_reporting_on_and_off():
    written = []
    screen = tui.Screen.__new__(tui.Screen)
    screen._active, screen.write = True, written.append
    screen.set_mouse(False)
    screen.set_mouse(True)
    assert written == ["\033[?1006l\033[?1003l\033[?1000l", "\033[?1000h\033[?1003h\033[?1006h"]
    screen.copy("hi")
    assert written[-1] == "\033]52;c;aGk=\007"                   # OSC 52, base64


def test_copy_puts_the_agents_last_answer_on_the_clipboard(app, monkeypatch):
    monkeypatch.setattr(appmod, "_copy_with_tool", lambda text: False)
    app._screen = FakeScreen()
    app._command("/copy")
    assert "Nothing from the agent" in text_of(app) and app._screen.copied == []
    app.say("The loss is flat because of the + 0.5.")
    app.note("Checked the run.")
    app.say("Second answer.")
    app._command("/copy")
    app._command("/copy 2")
    assert app._screen.copied == ["Second answer.", "The loss is flat because of the + 0.5."]
    assert "Copied 14 characters" in text_of(app)


# ---------------------------------------------------------------- restarting after a fix

def _restart_app(app, monkeypatch, tmp_path, hosted, status):
    script = tmp_path / "train.py"
    script.write_text("print(1)\n")
    app.session = {"session_id": "s1", "directory": "", "script": str(script)}
    app.console = types.SimpleNamespace(workdir=str(tmp_path), brain=types.SimpleNamespace(events=[]),
                                        send_control=lambda *a, **k: sent.append(a))
    app.hosting = types.SimpleNamespace(session_id="s1") if hosted else None
    app._status = status
    sent, launches = [], []

    def launch(arg, replace=None):
        launches.append(replace)
        app.session = {"session_id": "s2", "script": str(script)}
    monkeypatch.setattr(app, "_launch", launch)
    return script, sent, launches


def test_restart_runs_a_crashed_auto_track_script_again(app, monkeypatch, tmp_path):
    script, sent, launches = _restart_app(app, monkeypatch, tmp_path, hosted=True, status="crashed")
    monkeypatch.setattr(sys, "argv", [str(script), "--epochs", "3"])
    answer = app._agent_restart_run()
    assert sent == [] and launches == [{"argv": [str(script), "--epochs", "3"], "cwd": os.getcwd()}]
    assert answer.startswith("Restarted with the current code") and "s2" in answer


def test_restart_stops_a_live_auto_track_script_first(app, monkeypatch, tmp_path):
    script, sent, launches = _restart_app(app, monkeypatch, tmp_path, hosted=True, status="live")
    monkeypatch.setattr(app, "_poll_status", lambda: setattr(app, "_status", "ended"))
    assert app._agent_restart_run().startswith("Restarted")
    assert len(sent) == 1 and launches and launches[0]["argv"][0] == str(script)


def test_restart_of_a_run_started_elsewhere(app, monkeypatch, tmp_path):
    script, sent, launches = _restart_app(app, monkeypatch, tmp_path, hosted=False, status="live")
    answer = app._agent_restart_run()
    assert "still running" in answer and launches == [] and sent == []      # not ours to stop and start
    app._status = "crashed"
    assert app._agent_restart_run().startswith("Restarted") and launches == [{"argv": [str(script)], "cwd": str(tmp_path)}]


def test_the_agent_stops_starting_on_crashes_after_three_in_a_row(app):
    started = _crashed_app(app)
    for i in range(5):
        app.session = dict(app.session, session_id=f"s{i}")
        app._on_crash()
    assert len(started) == 3 and "crashes in a row" in text_of(app)
    app.view.editor.set("try a smaller lr")
    app.on_key("enter")                                      # the person speaks: the count starts again
    assert app._auto_crash_turns == 0


def test_no_restart_reminder_after_the_agent_restarted_the_run_itself(app, monkeypatch):
    from pulse import pulse_code
    app.cli.agent_provider, app.cli.agent_key = "DeepSeek", "sk"
    app.session = {"session_id": "s1", "script": "/p/train.py"}
    app.console = types.SimpleNamespace(brain=types.SimpleNamespace(note_fix=lambda why: None))
    app._evidence = lambda: "evidence"

    def turn(cli, line, evidence=None):
        app.session = {"session_id": "s2", "script": "/p/train.py"}      # restart_run inside the turn
        return "applied"
    monkeypatch.setattr(pulse_code, "run_turn", turn)
    app._handle("fix it")
    assert "still executing the code" not in text_of(app)


# ---------------------------------------------------------------- hover

def test_movement_reports_become_hover_keys():
    keys, _rest = tui.parse_keys("\033[<35;12;5M\033[<32;12;5M")
    assert keys == ["hover:11:4", "drag:11:4"]               # with the button held it is a drag


def test_the_clickable_line_under_the_mouse_lights_up(app, monkeypatch):
    monkeypatch.setattr(ui, "color_enabled", lambda: True)
    tool = tui.Entry("tool", calls=["GREP: lr"], output="a\nb")
    think = tui.Entry("thinking", "maybe lr")
    app.view.entries = [tui.Entry("user", "q"), think, tool, tui.Entry("say", "answer")]
    tui.compose(app.view, 90, 20)
    row = next(r for r, e in app.view.row_entries.items() if e is tool)
    plain_line = tui.entry_lines(tool, 80, False)[0]
    assert "\033[38;5;248m" in plain_line                    # grey at rest
    app.dirty = False
    app.on_key(f"hover:10:{row}")
    assert tool.hover and app.dirty and "\033[38;5;255m" in tui.entry_lines(tool, 80, False)[0]
    app.dirty = False
    app.on_key(f"hover:20:{row}")                            # still on it: no redraw
    assert app.dirty is False
    trow = next(r for r, e in app.view.row_entries.items() if e is think)
    app.on_key(f"hover:10:{trow}")                           # onto the thought
    assert think.hover and not tool.hover
    assert "\033[38;5;214m" in tui.entry_lines(think, 80, False)[0]
    srow = next(r for r, e in app.view.row_entries.items() if e.kind == "say")
    app.on_key(f"hover:10:{srow}")                           # the agent's words are not clickable
    assert app._hovered is None and not think.hover


def test_moving_the_mouse_keeps_the_status_line(app):
    app.view.status = "Ctrl+C again to leave Pulse"
    app.on_key("hover:1:1")
    assert app.view.status == "Ctrl+C again to leave Pulse"


# ---------------------------------------------------------------- /home and /change

def test_home_frees_the_agent_and_keeps_the_run_watched(real_app, tmp_path, monkeypatch):
    from pulse import pulse_code
    real_app._open_run(make_run(tmp_path))
    run_project = real_app.cli._project_root
    run_line(real_app, "/home")
    assert wait_for(lambda: real_app.background and not real_app.view.busy, 10)
    assert real_app.console is not None and real_app.view.side is None          # still watched
    assert real_app.cli._project_root == real_app.home_root != run_project       # the agent is home
    assert "DEBUG" not in (real_app.cli._system_prompt_override or "") and "LIVE TRAINING RUN" not in (
        real_app.cli._system_prompt_override or "")
    seen = {}
    monkeypatch.setattr(pulse_code, "run_turn", lambda cli, line, evidence=None: seen.setdefault("evidence", evidence))
    run_line(real_app, "write a README for my other project")
    assert wait_for(lambda: "evidence" in seen and not real_app.view.busy, 10)
    assert seen["evidence"] is None                                            # no run evidence at home
    run_line(real_app, "/monitor")                                             # the run comes back
    assert wait_for(lambda: not real_app.background and not real_app.view.busy, 10)
    assert real_app.cli._project_root == run_project and real_app.view.area == "DEBUG"


def test_change_shows_another_run_and_keeps_watching_the_first(real_app, tmp_path, monkeypatch):
    first = dict(make_run(tmp_path, "train"), status="live")
    second = dict(make_run(tmp_path, "evalrun"), status="live")
    monkeypatch.setattr(con, "discover", lambda *a, **k: [first, second])
    monkeypatch.setattr(con, "unmonitored_python_processes", lambda: [])
    real_app._open_run(first)
    run_line(real_app, "/change evalrun")
    assert wait_for(lambda: not real_app.view.busy, 10)
    assert real_app.session["session_id"] == "evalrun" and list(real_app.parked) == ["train"]
    assert real_app.parked["train"]["console"]._thread.is_alive()            # still reading the first run
    assert "evalrun.py" in real_app.view.context
    run_line(real_app, "/change")                                              # the picker marks it watched
    assert wait_for(lambda: real_app.view.options is not None, 10)
    tags = {label: tag for label, _d, tag in real_app.view.options}
    assert tags == {"train.py": "watched", "evalrun.py": "open"}
    for _ in range([label for label, _d, _t in real_app.view.options].index("train.py")):
        press(real_app, "down")
    press(real_app, "enter")                                                   # back to the first
    assert wait_for(lambda: not real_app.view.busy and real_app.session["session_id"] == "train", 10)
    assert list(real_app.parked) == ["evalrun"] and "Back on train.py" in text_of(real_app)
    run_line(real_app, "/close all")
    assert wait_for(lambda: not real_app.view.busy, 10)
    assert real_app.console is None and real_app.parked == {}


def test_a_finding_from_a_run_off_screen_says_which_run(real_app, tmp_path, monkeypatch):
    first = make_run(tmp_path, "train")
    second = make_run(tmp_path, "evalrun")
    real_app._open_run(first)
    real_app._switch_to(second)
    parked = real_app.parked["train"]["console"]
    parked._announce([types.SimpleNamespace(message="loss went NaN", severity="critical")])
    real_app.console._announce([types.SimpleNamespace(message="grad norm high", severity="warning")])
    found = [e.text for e in real_app.view.entries if e.kind == "finding"]
    assert found[-2:] == ["train.py: loss went NaN", "grad norm high"]


def test_a_run_off_screen_that_crashes_while_home_gets_the_agent(real_app, tmp_path, monkeypatch):
    first = make_run(tmp_path, "train")
    second = make_run(tmp_path, "evalrun")
    real_app._open_run(first)
    real_app._switch_to(second)
    real_app._background_run()                                                 # home, both watched
    crashes = []
    monkeypatch.setattr(real_app, "_on_crash", lambda: crashes.append(real_app.session["session_id"]))
    monkeypatch.setattr(con, "_describe_session",
                        lambda d: {"status": "crashed" if d.endswith("train") else "live", "last_seen": 0})
    with real_app.lock:
        real_app._tick_parked(time.monotonic() + 10)
    assert crashes == ["train"] and real_app.background                       # the view stays home
    assert list(real_app.parked) == ["evalrun"]


def test_a_run_off_screen_that_crashes_while_another_is_shown_is_announced(real_app, tmp_path, monkeypatch):
    first = make_run(tmp_path, "train")
    second = make_run(tmp_path, "evalrun")
    real_app._open_run(first)
    real_app._switch_to(second)
    monkeypatch.setattr(con, "_describe_session",
                        lambda d: {"status": "crashed" if d.endswith("train") else "live", "last_seen": 0})
    with real_app.lock:
        real_app._tick_parked(time.monotonic() + 10)
    assert real_app.session["session_id"] == "evalrun"                        # the view is not taken
    assert "train.py (watched off screen) crashed. /change train.py" in text_of(real_app)


def test_the_top_bar_at_home_names_every_watched_run(real_app, tmp_path):
    real_app._open_run(make_run(tmp_path, "train"))
    real_app._switch_to(make_run(tmp_path, "evalrun"))
    real_app._background_run()
    context = real_app.view.context
    assert context.startswith(os.path.basename(real_app.home_root)) or "~" in context or real_app.home_root in context
    assert "evalrun.py" in context and "train.py" in context and "/change" in context
    assert "evalrun_project" not in context                                   # home's folder, not the run's


# ---------------------------------------------------------------- /config

def test_config_lists_and_changes_settings_and_applies_them_now(app, monkeypatch):
    from pulse import pulse_settings as settings
    app._screen = FakeScreen()
    app.install()                                            # what /config prints goes to the transcript
    try:
        app._command("/config")
        app._flush_partial()
    finally:
        app.uninstall()
    ui.set_host(app)
    shown = text_of(app)
    assert "mouse" in shown and "remember_keys" in shown and "/config reset" in shown
    app._command("/config mouse off")
    assert app.mouse is False and app._screen.mouse is False and settings.get("mouse") == "off"
    app._command("/config review off")
    assert app.cli.review is False and settings.get("review") == "off"
    app._command("/config mouse sideways")
    assert "mouse is on or off" in text_of(app)
    app._command("/settings audits")
    assert "audits = on" in text_of(app)
    picked = []
    app.cli._pick_agent_and_remember = lambda initial=False: picked.append(initial) or True
    app.cli.agent_provider = "DeepSeek"
    app._command("/config agent")
    assert picked == [False] and "remembered for the next start" in text_of(app)


def test_the_app_starts_with_the_remembered_mouse_and_audits(tmp_path, monkeypatch):
    from pulse import pulse_settings as settings
    monkeypatch.setattr(settings, "_home", lambda: tmp_path)
    settings.set("mouse", "off")
    settings.set("audits", "off")
    cli_ = FakeCli()
    app_ = appmod.App(cli_, str(tmp_path))
    assert app_.mouse is False and app_.audits is False


def test_the_dashboard_makes_its_window_before_its_tk_variables():
    """A Tk variable made before the window raised "Too early to create variable: no default
    root window" and the dashboard (mode="ui") never opened. Checked on the source: the test
    machines have no display."""
    import inspect
    from pulse import pulse as core
    source = inspect.getsource(core.Dashboard.__init__)
    window = source.index("tk.Tk()")
    for var in ("tk.BooleanVar(", "tk.StringVar(", "tk.IntVar(", "tk.DoubleVar("):
        if var in source:
            assert source.index(var) > window, f"{var} before the window"


def test_auto_mode_opens_the_app_in_a_terminal(monkeypatch):
    from pulse import pulse as core
    seen = []
    monkeypatch.setattr(core, "_default_mode", lambda: "app")
    monkeypatch.setattr(core, "_in_multiprocessing_bootstrap", lambda frame: False, raising=False)
    monkeypatch.setattr(core, "_auto_track_session",
                        lambda frame, fn, interval, code, root, mode: seen.append(mode) or core._NO_SESSION)
    core.auto_track(mode="auto")
    monkeypatch.setattr(core, "_default_mode", lambda: "cli")              # no terminal for the app
    core.auto_track(mode="auto")
    core.auto_track(mode="ui")                                              # the window, when asked for
    assert seen == ["app", "auto", "ui"]



# ---------------------------------------------------------------- selecting, copying, scrolling

def _frame_app(app):
    app.view.entries = [tui.Entry("user", "why is the loss flat"),
                        tui.Entry("say", "The loss is flat because of the + 0.5 on line 12.")]
    app._total_w, app._height = 90, 20
    frame, _r, _c = tui.compose(app.view, 90, 20)
    app._frame = frame
    return frame


def test_a_drag_selects_text_and_copies_it(app, monkeypatch):
    copied = []
    monkeypatch.setattr(appmod, "_copy_with_tool", lambda text: copied.append(("tool", text)))
    app._screen = FakeScreen()
    frame = _frame_app(app)
    row = next(i for i, line in enumerate(frame) if "The loss is flat" in plain(line))
    col = plain(frame[row]).index("The loss")
    app.on_key(f"press:{col}:{row}")
    app.on_key(f"drag:{col + 7}:{row}")
    app.on_key(f"release:{col + 7}:{row}")
    assert app._screen.copied == ["The loss"]
    assert wait_for(lambda: copied == [("tool", "The loss")])
    assert "Copied 8 characters" in app.view.status
    lit = tui.highlight(frame, app._sel)
    assert "\033[7mThe loss\033[27m" in lit[row]
    app.on_key("a")                                          # typing: the selection goes
    assert app._sel is None


def test_a_selection_over_several_lines_stays_in_its_pane():
    lines = ["  left pane one   | right one", "  left pane two   | right two"]
    sel = {"anchor": (2, 0), "head": (14, 1), "cols": (0, 16)}
    assert tui.selected_text(lines, sel) == "left pane one\n  left pane two"


def test_a_press_and_release_in_place_is_still_a_click(app):
    tool = tui.Entry("tool", calls=["GREP: lr"], output="a\nb")
    app.view.entries = [tui.Entry("user", "q"), tool]
    tui.compose(app.view, 90, 20)
    row = next(r for r, e in app.view.row_entries.items() if e is tool)
    app.on_key(f"press:5:{row}")
    app.on_key(f"release:5:{row}")
    assert tool.open is True and app._sel is None


def test_up_brings_back_what_was_typed_and_the_wheel_scrolls(app):
    app.view.editor.history = ["first question", "second question"]
    app.on_key("up")
    assert app.view.editor.text == "second question" and app.view.scroll == 0
    app.view.editor.take(remember=False)
    app.view.entries = [tui.Entry("text", f"line {i}") for i in range(80)]
    app.on_key("wheel:up:30:5")
    assert app.view.scroll == 3


def test_the_wheel_over_the_run_pane_scrolls_the_pane(app):
    app.view.side = [f"value {i}" for i in range(60)]
    tui.compose(app.view, 120, 20)
    assert app.view.side_cols > 0
    app.on_key("wheel:down:5:8")
    frame, _r, _c = tui.compose(app.view, 120, 20)
    assert app.view.side_scroll == 3 and plain(frame[2]).startswith(" value 3")
    assert app.view.scroll == 0                                  # the conversation did not move


def test_the_run_pane_lists_every_value():
    from pulse import pulse_console as con
    state = {"step": 5, "histories": {f"v{i}": [1.0, 2.0] for i in range(14)}, "findings": [], "tensors": {},
             "gaps": 0, "finished": False}
    console = types.SimpleNamespace(workdir="/x", brain=types.SimpleNamespace(agent=None, schedule=None))
    lines = [plain(l) for l in appmod.side_lines(state, {"script": "/x/t.py"}, "live", console, None, 40)]
    assert sum(1 for l in lines if l.startswith("v")) == 14 and not any("more (/vars)" in l for l in lines)


# ---------------------------------------------------------------- a serious finding

def _finding(check="nonfinite", variable="loss", severity="critical", message="loss is NaN"):
    return types.SimpleNamespace(check=check, variable=variable, severity=severity, message=message)


def test_a_nan_gets_the_agent_at_once(app):
    started = _crashed_app(app)
    app._status = "live"
    app._on_findings([_finding()])
    app._on_findings([_finding()])                                     # announced again: not twice
    assert len(started) == 1 and "loss is NaN" in started[0][0] and "restart" in started[0][0]
    assert any(e.kind == "user" and "What is wrong with loss" in e.text for e in app.view.entries)


def test_a_warning_or_a_pause_does_not_start_the_agent(app):
    started = _crashed_app(app)
    app._status = "live"
    app._on_findings([_finding(check="plateau", severity="warning", message="loss stopped improving")])
    app._on_findings([_finding(check="throughput_stopped", message="no steps for 2 min")])
    assert started == []


def test_a_finding_while_the_agent_is_busy_waits(app):
    started = _crashed_app(app)
    app._status = "live"
    app.view.busy = True
    app._on_findings([_finding()])
    assert started == [] and app._problem_pending
    app.view.busy = False
    app._run_job(lambda: None, ())
    assert len(started) == 1 and app._problem_pending is None


def test_the_open_runs_console_hands_findings_to_the_agent(real_app, tmp_path, monkeypatch):
    real_app._open_run(make_run(tmp_path))
    real_app._status = "live"
    seen = []
    monkeypatch.setattr(real_app, "_on_findings", lambda findings: seen.extend(findings))
    real_app.console.quiet = True                                       # /quiet: shown or not, the agent hears
    real_app.console._announce([_finding()])
    assert [f.message for f in seen] == ["loss is NaN"]


def test_findings_that_come_with_the_one_being_handled_are_not_a_new_turn(app):
    started = _crashed_app(app)
    app._status = "live"
    app._on_findings([_finding()])
    app.view.busy = True                                             # the agent is on it
    app._on_findings([_finding(variable="grad_norm", message="grad_norm is NaN")])
    app._on_findings([_finding(variable="val_loss", message="val_loss is infinite")])
    assert app._problem_pending is None
    app.view.busy = False
    app._run_job(lambda: None, ())
    assert len(started) == 1                                         # one turn for the one problem


# ---------------------------------------------------------------- the other "seen but not acted on" gaps

def test_a_problem_found_by_the_scheduled_check_gets_the_agent(real_app, tmp_path, monkeypatch):
    real_app._open_run(make_run(tmp_path))
    real_app._status = "live"
    seen = []
    monkeypatch.setattr(real_app, "_on_findings", lambda findings: seen.extend(findings))
    real_app.console._report_audit({"status": "problem", "step": 40, "findings": ["lr far too high"],
                                    "text": '{"status": "problem"}\nThe lr of 3.0 makes every step overshoot.'})
    assert len(seen) == 1 and seen[0].check == "audit" and "lr far too high" in seen[0].message
    assert "found a problem" in text_of(real_app)


def test_a_stall_is_said_once_and_so_is_moving_again(app):
    app.session = {"session_id": "s1", "script": "/p/train.py"}
    app._note_stall("stalled")
    app._note_stall("stalled")
    app._note_stall("live")
    texts = [e.text for e in app.view.entries]
    assert sum("stopped making steps" in t for t in texts) == 1 and any("moving again" in t for t in texts)


def test_a_stall_after_pause_is_not_called_stuck(app):
    app.session = {"session_id": "s1", "script": "/p/train.py"}
    app._paused_here = True
    app._note_stall("stalled")
    assert not any("stopped making steps" in e.text for e in app.view.entries)


def test_a_serious_finding_off_screen_is_taken_on_when_home(real_app, tmp_path, monkeypatch):
    real_app._open_run(make_run(tmp_path, "train"))
    real_app._switch_to(make_run(tmp_path, "evalrun"))
    real_app._background_run()
    seen = []
    monkeypatch.setattr(real_app, "_on_findings", lambda findings: seen.append(real_app.session["session_id"]))
    real_app.parked["train"]["console"]._announce([_finding()])
    assert seen == ["train"] and real_app.background                 # the view stays home


def test_a_serious_finding_off_screen_with_another_run_shown_says_where(real_app, tmp_path):
    real_app._open_run(make_run(tmp_path, "train"))
    real_app._switch_to(make_run(tmp_path, "evalrun"))
    real_app.parked["train"]["console"]._announce([_finding()])
    real_app.parked["train"]["console"]._announce([_finding()])
    assert real_app.session["session_id"] == "evalrun"
    assert sum("needs a look" in e.text for e in real_app.view.entries) == 1 and "/change train.py" in text_of(real_app)


def test_an_audit_with_no_readable_verdict_is_still_shown(real_app, tmp_path):
    real_app._open_run(make_run(tmp_path))
    real_app.console._report_audit({"status": None, "text": "decay_steps = epochs: lr is 0 from step 13."})
    assert "no verdict could be read" in text_of(real_app) and "decay_steps" in text_of(real_app)


def test_an_audit_problem_rated_low_risk_does_not_start_the_agent(real_app, tmp_path, monkeypatch):
    real_app._open_run(make_run(tmp_path))
    real_app._status = "live"
    seen = []
    monkeypatch.setattr(real_app, "_on_findings", lambda findings: seen.extend(findings))
    real_app.console._report_audit({"status": "problem", "risk": "low", "step": 9, "findings": ["step 399 of 400"],
                                    "text": "The run stopped at step 399."})
    assert seen == [] and "found a problem" in text_of(real_app)
