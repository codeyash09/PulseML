"""The modernized terminal UI: rounded cards, the login form, the agent dropdown with key status,
the composer box with the agent chip inside it, and the popup that asks for a key."""
import json
import os
from pathlib import Path
import pty
import re
import select
import signal
import sys
import time

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)

from pulse import pulse_app as appmod
from pulse import pulse_cli as pc
from pulse import pulse_supabase as cloud
from pulse import pulse_tui as tui
from pulse import pulse_ui as ui

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def plain(text):
    return ANSI.sub("", text)


def run_pty(code, keys=(), rows=40, cols=100, settle=0.7):
    """Run `code` on a real terminal, type `keys` (str, delay) into it, return the text it drew."""
    import fcntl
    import struct
    import termios
    pid, fd = pty.fork()
    if pid == 0:
        env = dict(os.environ, TERM="xterm-256color", PYTHONPATH=SRC, COLUMNS=str(cols), LINES=str(rows))
        env.pop("NO_COLOR", None)
        os.execvpe(sys.executable, [sys.executable, "-c", code], env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    out = []

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    return
                if not data:
                    return
                out.append(data)

    pump(settle)
    for key, delay in keys:
        os.write(fd, key.encode())
        pump(delay)
    pump(0.3)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
    return b"".join(out).decode("utf-8", "replace")


# ---- the card ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [8, 20, 30, 31, 64, 120])
def test_every_line_of_a_card_is_exactly_as_wide_as_asked(width):
    lines = ui.box(["hello", None, "a longer line that has to be clipped to fit"], width, title="Card", footer="esc")
    assert {ui._visible_len(line) for line in lines} == {width}
    assert plain(lines[0]).startswith("╭") and plain(lines[0]).endswith("╮")
    assert plain(lines[-1]).startswith("╰") and plain(lines[-1]).endswith("╯")


def test_styled_content_does_not_break_the_card():
    lines = ui.box([ui._s("bold red", "bold", "red"), ui._hilite(ui._fit("selected", 20))], 30)
    assert {ui._visible_len(line) for line in lines} == {30}


def test_a_card_falls_back_to_ascii_where_the_terminal_cannot_draw_it(monkeypatch):
    monkeypatch.setattr(ui, "_UNICODE", False)
    lines = ui.box(["x"], 12, title="t")
    assert plain(lines[0]).startswith("+") and plain(lines[1]).startswith("|") and "╭" not in "".join(lines)


# ---- the login form, on a real terminal ---------------------------------------------------

FORM = '''
import json
from pulse import pulse_ui as ui
email = dict(name="email", label="Email", placeholder="name@example.com",
             check=lambda t, v: None if "@" in t else "Enter a valid email address")
pw = dict(name="password", label="Password", secret=True,
          check=lambda t, v: None if len(t) >= 8 else "Password must be at least 8 characters")
confirm = dict(name="confirm", label="Confirm password", secret=True,
               check=lambda t, v: None if t == v.get("password") else "Passwords don't match")
login_pw = dict(name="password", label="Password", secret=True)
res = ui.form("", "Debug training runs", tabs=["Sign up", "Log in"],
              fields=[[email, pw, confirm], [email, login_pw]], button=["Create account", "Log in"])
print("RESULT " + json.dumps(res))
'''


def result_of(output):
    found = re.findall(r"RESULT (.*)", plain(output))
    return json.loads(found[-1]) if found else None


def test_the_login_form_returns_what_was_typed():
    out = run_pty(FORM, [("a@b.co\t", .2), ("longenough1\t", .2), ("longenough1\t", .2), ("\r", .5)])
    assert result_of(out) == [0, {"email": "a@b.co", "password": "longenough1", "confirm": "longenough1"}]


def test_the_form_draws_a_rounded_card_with_tabs_and_masked_secrets():
    out = plain(run_pty(FORM, [("a@b.co\t", .2), ("hunter22", .3)]))
    assert "╭" in out and "╯" in out and "Sign up" in out and "Log in" in out and "Create account" in out
    assert "hunter22" not in out and "••••••••" in out


def test_the_form_refuses_to_send_until_the_fields_are_valid():
    out = run_pty(FORM, [("\t\t\t", .2), ("\r", .5)])
    assert result_of(out) is None and "Required" in plain(out)
    out = run_pty(FORM, [("nope\t", .2), ("longenough1\t", .2), ("different\t", .2), ("\r", .5)])
    assert result_of(out) is None and "Passwords don't match" in plain(out)


def test_escape_leaves_the_form_without_a_result():
    out = run_pty(FORM, [("a@b.co", .2), ("\x1b", .6)])
    assert result_of(out) is None and "RESULT null" in plain(out)


def test_an_email_typed_under_one_tab_is_still_there_under_the_next():
    keys = [("a@b.co", .2), ("\x1b[A", .2), ("\x1b[C", .3), ("\r", .2), ("\t", .2), ("pw", .2), ("\t", .2), ("\r", .5)]
    assert result_of(run_pty(FORM, keys)) == [1, {"email": "a@b.co", "password": "pw"}]


def test_the_form_steps_aside_when_there_is_no_terminal():
    with pytest.raises(ui.Unavailable):
        ui.form("", None, fields=[dict(name="x", label="X")])        # pytest's stdin is not a terminal


# ---- the card is the front of the real sign-in --------------------------------------------

@pytest.fixture
def signin(monkeypatch, tmp_path):
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "credentials.json")
    monkeypatch.setattr(cloud, "load_cached_credentials", lambda: None)
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    cli = pc.PulseCLI.__new__(pc.PulseCLI)
    cli.config, cli.non_interactive = {}, False
    cli.user_id = cli.email = cli.team_id = cli.project_id = None
    monkeypatch.setattr(pc.PulseCLI, "_config_value", lambda self, *a, **k: k.get("default", ""), raising=False)
    monkeypatch.setattr(pc.PulseCLI, "_config_text", lambda self, *a, **k: "", raising=False)
    return cli


def test_the_cards_values_log_you_in_without_asking_again(signin, monkeypatch):
    monkeypatch.setattr(pc, "_ui_auth_card", lambda: ("l", {"email": "me@pulse.ai", "password": "correct-horse"}))
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: pytest.fail("asked a question the card already answered"))
    seen = []
    monkeypatch.setattr(cloud, "log_in", lambda email, password: seen.append((email, password)) or {"id": "u1"})
    monkeypatch.setattr(cloud, "attach_session_token", lambda uid: "tok")
    monkeypatch.setattr(cloud, "save_cached_credentials", lambda *a, **k: None)
    signin._auth_flow()
    assert seen == [("me@pulse.ai", "correct-horse")] and signin.user_id == "u1" and signin.email == "me@pulse.ai"


def test_without_a_terminal_the_card_is_skipped_and_the_old_prompts_stand():
    assert pc._ui_auth_card() is None


def test_the_card_maps_its_tabs_to_the_flows_letters(monkeypatch):
    monkeypatch.setattr(pc._ui, "enabled", lambda: True)
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    for tab, letter in ((0, "s"), (1, "l"), (2, "r")):
        monkeypatch.setattr(pc._ui, "form", lambda *a, _t=tab, **k: (_t, {"email": "x@y.zz"}))
        assert pc._ui_auth_card() == (letter, {"email": "x@y.zz"})
    monkeypatch.setattr(pc._ui, "form", lambda *a, **k: None)             # Esc: back to the plain menu
    assert pc._ui_auth_card() is None


# ---- the agent list says which agents have a key --------------------------------------------

def provider(**flags):
    return next((n for n, i in pc.PROVIDERS.items() if all(bool(i.get(k)) == v for k, v in flags.items())), None)


@pytest.fixture
def no_keys(monkeypatch):
    for info in pc.PROVIDERS.values():
        if info.get("env_key"):
            monkeypatch.delenv(info["env_key"], raising=False)
    monkeypatch.delenv(pc._openrouter.ENV_KEY, raising=False)
    monkeypatch.setattr(pc._settings, "saved_key", lambda env_var: "")
    monkeypatch.setattr(pc._openrouter, "saved_key", lambda: None)


def test_a_key_in_the_environment_makes_an_agent_ready(no_keys, monkeypatch):
    name = next(n for n, i in pc.PROVIDERS.items() if i.get("env_key") and not i.get("local") and not i.get("custom")
                and not i.get("openrouter") and i["env_key"] != pc._openrouter.ENV_KEY)
    assert pc.agent_key_state(name)[0] == "missing" and "add key" in pc.agent_key_state(name)[1]
    monkeypatch.setenv(PROVIDERS_ENV := pc.PROVIDERS[name]["env_key"], "sk-x")
    assert pc.agent_key_state(name) == ("ready", pc.agent_key_state(name)[1]) and "key set" in pc.agent_key_state(name)[1]


def test_a_key_pulse_was_allowed_to_keep_counts_too(no_keys, monkeypatch):
    name = next(n for n, i in pc.PROVIDERS.items() if i.get("env_key") and not i.get("local")
                and not i.get("custom") and i["env_key"] != pc._openrouter.ENV_KEY)
    monkeypatch.setattr(pc._settings, "saved_key", lambda env_var: "sk-saved" if env_var == pc.PROVIDERS[name]["env_key"] else "")
    assert pc.agent_key_state(name)[0] == "ready"


def test_local_custom_and_openrouter_agents_are_not_told_to_add_a_key(no_keys):
    for flags, state in (({"local": True}, "local"), ({"custom": True}, "custom")):
        name = provider(**flags)
        if name:
            assert pc.agent_key_state(name)[0] == state
    router = next((n for n, i in pc.PROVIDERS.items() if i.get("openrouter")), None)
    if router:
        assert pc.agent_key_state(router)[0] == "signin" and "sign in" in pc.agent_key_state(router)[1]


def test_the_dropdown_shows_each_agents_status_and_draws_the_eye_to_those_that_need_a_key(no_keys, monkeypatch):
    names = list(pc.PROVIDERS)
    ready = next(n for n in names if pc.PROVIDERS[n].get("env_key") and not pc.PROVIDERS[n].get("local")
                 and not pc.PROVIDERS[n].get("custom") and not pc.PROVIDERS[n].get("openrouter")
                 and pc.PROVIDERS[n]["env_key"] != pc._openrouter.ENV_KEY)
    monkeypatch.setenv(pc.PROVIDERS[ready]["env_key"], "sk-x")
    monkeypatch.setattr(pc._ui, "enabled", lambda: True)
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    monkeypatch.setattr(pc._ui, "header", lambda *a, **k: None)
    captured = {}

    def choose(options, **kwargs):
        captured["options"], captured["kwargs"] = options, kwargs
        return names.index(ready)

    monkeypatch.setattr(pc._ui, "choose", choose)
    assert pc._ui_pick_agent(names, None, None) == ready
    by_name = {o.label: o for o in captured["options"]}
    assert "key set" in by_name[ready].badge and by_name[ready].tone is None
    missing = next(o for o in captured["options"] if "add key" in (o.badge or ""))
    assert missing.tone == "attn"
    assert captured["kwargs"]["compact"] is True and captured["kwargs"]["searchable"] is True


# ---- the list itself, on a real terminal -----------------------------------------------------

def test_a_badge_stays_visible_however_long_the_row_is():
    code = '''
from pulse import pulse_ui as ui
i = ui.choose([ui.Option("A very long agent name that would push everything else off the edge of the card",
                         "an-extremely/long-model-identifier-string-that-goes-on", badge="○ add key", tone="attn"),
               ui.Option("Short", badge="● key set")], compact=True)
print("PICKED", i)
'''
    out = plain(run_pty(code, [("\r", .5)], cols=80))
    assert "add key" in out and "PICKED 0" in out and "╭" in out


# ---- the main screen -------------------------------------------------------------------------

def view_for(kind):
    v = tui.View()
    v.context, v.commands = "~/proj", [("/agent", "switch agent"), ("/help", "commands")]
    v.agent_name, v.agent_state = "Anthropic (Claude Sonnet 5)", "ready"
    v.entries.append(tui.Entry("user", "why did my loss stall?"))
    v.entries.append(tui.Entry("say", "The scheduler steps every batch."))
    if kind == "typing":
        v.editor.set("add flash attention")
    elif kind == "dropdown":
        v.question = "Select an agent"
        v.options = [("Anthropic (Claude Sonnet 5)", "claude-sonnet-5", "current"), ("OpenAI (GPT-5.5)", "gpt-5.5", ""),
                     ("Ollama (local)", "", "")]
        v.option_extras = [("● key set", ""), ("○ add key", "attn"), ("local", "")]
        v.option_ids, v.option_index = [0, 1, 2], 1
    elif kind == "popup":
        v.question, v.secret = "API key", True
        v.popup_title, v.popup_note = "OpenAI (GPT-5.5) API key", "Used only to call the model."
        v.editor.set("sk-secret-value")
    elif kind == "question":
        v.question = "Which file?"
        v.editor.set("train.py")
    return v


SIZES = [(20, 8), (30, 10), (40, 12), (60, 24), (84, 12), (100, 30), (140, 50), (80, 9), (25, 40)]


@pytest.mark.parametrize("width, height", SIZES)
@pytest.mark.parametrize("kind", ["idle", "typing", "dropdown", "popup", "question"])
def test_every_screen_fits_every_terminal_and_the_cursor_is_where_the_typing_is(kind, width, height):
    view = view_for(kind)
    frame, row, col = tui.compose(view, width, height)
    assert len(frame) == height - 1
    assert all(tui.visible_len(line) <= width for line in frame), "a line is wider than the screen"
    assert 0 <= row < height and 0 <= col < width
    line = plain(frame[row])
    if kind in ("idle", "typing", "question") or (kind == "popup" and "Private" not in "".join(map(plain, frame))
                                                  and "API key" not in "".join(map(plain, frame))):
        assert "❯" in line
    if kind == "dropdown":
        assert "Search" in line
    assert "sk-secret-value" not in "".join(map(plain, frame))      # a secret is never drawn


def test_the_composer_is_a_rounded_box_with_the_agent_chip_under_the_text():
    frame = [plain(line) for line in tui.compose(view_for("typing"), 100, 30)[0]]
    top = next(i for i, line in enumerate(frame) if "╭" in line)
    assert "add flash attention" in frame[top + 1] and "❯" in frame[top + 1]
    chip = next(line for line in frame[top + 2:] if "Anthropic (Claude Sonnet 5)" in line)
    assert "key set" in chip and "send" in chip and chip.strip().startswith("│")
    assert any("╰" in line for line in frame[top:])


def test_the_chip_says_when_there_is_no_agent_or_no_key():
    v = view_for("idle")
    v.agent_name, v.agent_state = "", ""
    assert "Choose an agent" in "".join(plain(l) for l in tui.compose(v, 100, 30)[0])
    v.agent_name, v.agent_state = "OpenAI (GPT-5.5)", "missing"
    assert "no key" in "".join(plain(l) for l in tui.compose(v, 100, 30)[0])


def test_the_chip_is_a_click_target_exactly_where_it_is_drawn():
    v = view_for("idle")
    frame, _, _ = tui.compose(v, 100, 30)
    row, x0, x1 = v.chip_hit
    drawn = plain(frame[row])
    assert "Anthropic (Claude Sonnet 5)" in drawn[x0:x1] and "key set" not in drawn[x0:x1]
    for kind in ("dropdown", "popup"):
        other = view_for(kind)
        tui.compose(other, 100, 30)
        assert other.chip_hit is None


def test_the_dropdown_lists_every_agent_with_its_status_at_the_right_edge():
    frame = [plain(line) for line in tui.compose(view_for("dropdown"), 100, 30)[0]]
    rows = {name: next(line for line in frame if name in line) for name in ("Claude Sonnet 5", "GPT-5.5", "Ollama")}
    assert rows["Claude Sonnet 5"].rstrip().rstrip("│").rstrip().endswith("key set")
    assert rows["GPT-5.5"].rstrip().rstrip("│").rstrip().endswith("add key") and "›" in rows["GPT-5.5"]
    assert rows["Ollama"].rstrip().rstrip("│").rstrip().endswith("local")
    assert any("Select an agent" in line for line in frame)


def test_a_key_is_asked_for_in_a_popup_with_the_provider_in_its_title_and_the_key_masked():
    v = view_for("popup")
    frame = [plain(line) for line in tui.compose(v, 100, 30)[0]]
    text = "\n".join(frame)
    assert "OpenAI (GPT-5.5) API key" in text and "Used only to call the model." in text
    assert "•" * 15 in text and "Enter confirm" in text
    top = next(i for i, line in enumerate(frame) if "OpenAI (GPT-5.5) API key" in line)
    left = frame[top].index("╭")
    assert 10 < left < 40                                              # centred, not pinned to the edge


def test_the_cursor_sits_in_the_popups_field():
    v = view_for("popup")
    frame, row, col = tui.compose(v, 100, 30)
    assert "❯" in plain(frame[row]) and "•" in plain(frame[row])[col - 2:] or col > 10


def test_on_a_screen_too_short_for_the_box_the_composer_shrinks_instead_of_vanishing():
    for height, expect_box in ((30, True), (12, True), (9, False)):
        frame = [plain(line) for line in tui.compose(view_for("typing"), 60, height)[0]]
        text = "\n".join(frame)
        assert "add flash attention" in text
        assert ("╭" in text and "╰" in text) == expect_box


# ---- the app --------------------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text("lr = 0.1\n")
    cli = appmod._new_cli(str(project), [str(project / "train.py")])
    cli.agent_provider = next(n for n, info in pc.PROVIDERS.items() if info.get("model") and info.get("env_key"))
    cli.agent_key = "sk-test"
    cli._sync_agent_turn = lambda *a, **k: None
    cli._native_off = True
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: pytest.fail("no model call expected"))
    app_ = appmod.App(cli, str(project))
    yield app_
    app_.uninstall()


def test_the_chip_names_the_agent_in_use_and_whether_it_can_be_used(app):
    app._refresh_header()
    assert app.view.agent_name == app.cli.agent_provider
    assert app.view.agent_state == "ready"             # a key was given for this session
    app.cli.agent_provider = None
    app._refresh_header()
    assert app.view.agent_name == "" and app.view.agent_state == ""


def test_ctrl_g_and_a_click_on_the_chip_both_open_the_agent_list(app, monkeypatch):
    opened = []
    monkeypatch.setattr(app, "_pick_agent", lambda: opened.append(1))
    app.on_key("ctrl+g")
    assert opened == [1]
    tui.compose(app.view, 100, 30)
    row, x0, x1 = app.view.chip_hit
    app._click(x0 + 1, row)
    assert opened == [1, 1]
    app._click(x1 + 5, row)                            # beside the chip: not a click on it
    app._click(x0, row + 1)
    assert opened == [1, 1]


def test_the_agent_list_is_not_opened_over_a_running_job_or_another_question(app, monkeypatch):
    started = []
    monkeypatch.setattr(app, "_start", lambda *a: started.append(a))
    app.view.busy = True
    app._pick_agent()
    assert started == [] and "Ctrl+C" in app.view.status
    app.view.busy, app.view.question = False, "a question"
    app._pick_agent()
    assert started == []
    app.view.question = ""
    app._pick_agent()
    assert len(started) == 1


def test_a_question_carries_its_popup_title_and_note_to_the_view(app):
    class Host(appmod.App):
        pass
    question = appmod._Question("API key", secret=True, title="Anthropic API key", note="Kept on this machine.")
    assert (question.title, question.note, question.secret) == ("Anthropic API key", "Kept on this machine.", True)


def test_ask_hands_its_title_and_note_to_the_host_that_draws_the_popup():
    seen = {}

    class Host:
        def ask(self, label, secret=False, placeholder=None, **extra):
            seen.update(label=label, secret=secret, **extra)
            return "answer"

    ui.set_host(Host())
    try:
        assert ui.ask("API key", secret=True, title="OpenAI API key", note="Never shown again.") == "answer"
    finally:
        ui.set_host(None)
    assert seen["title"] == "OpenAI API key" and seen["note"] == "Never shown again." and seen["secret"] is True


def test_ctrl_g_is_a_key_the_terminal_reader_knows_and_ctrl_p_is_left_to_the_editor():
    assert tui._CTRL_KEYS["\x07"] == "ctrl+g"
    assert "\x10" not in tui._CTRL_KEYS               # VS Code takes Ctrl+P for quick open


# ---- yes / no as buttons -----------------------------------------------------------------------

CONFIRM = '''
from pulse import pulse_ui as ui
print("ANSWER", ui.confirm("Create the workspace?", default=DEFAULT))
'''


def answered(keys, default=True):
    out = plain(run_pty(CONFIRM.replace("DEFAULT", str(default)), keys, rows=14, settle=0.5))
    found = re.findall(r"ANSWER (True|False)", out)
    return (found[-1] == "True") if found else None, out


@pytest.mark.parametrize("default, keys, expected", [
    (True, [("\r", .4)], True),                       # Enter takes the default button
    (False, [("\r", .4)], False),
    (False, [("y", .4)], True),                       # y / n answer at once
    (True, [("n", .4)], False),
    (True, [("\x1b[C", .2), ("\r", .4)], False),      # the arrows move between the buttons
    (False, [("\x1b[D", .2), ("\r", .4)], True),
    (True, [("\t", .2), ("\r", .4)], False),
    (True, [("\x1b", .6)], False),                    # Esc declines
])
def test_the_yes_no_buttons_answer_what_was_chosen(default, keys, expected):
    assert answered(keys, default)[0] is expected


def test_the_yes_no_question_is_drawn_as_two_rounded_buttons():
    _, out = answered([("\x1b[C", .3)])
    assert out.count("╭") >= 2 and "Yes" in out and "No" in out and "Create the workspace?" in out


def test_inside_the_app_a_confirmation_is_still_asked_as_text():
    class Host:
        def ask(self, label, **kw):
            return "no"
    ui.set_host(Host())
    try:
        assert ui.confirm("Sure?", default=True) is False
    finally:
        ui.set_host(None)


# ---- the agent list really opens (through the app as the UI host, not a stub) ---------------------

def wait_for(condition, seconds=8.0):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("key", ["tab", "ctrl+g"])
def test_the_key_really_opens_the_agent_list_with_every_agents_status(app, key):
    app.cli.non_interactive = False
    app.install()                                   # what the running app does: it becomes the UI host
    try:
        app.on_key(key)
        assert wait_for(lambda: app.view.options is not None), "the agent list never opened"
        assert app.view.question == "Select an agent" and len(app.view.options) > 5
        assert len(app.view.option_extras) == len(app.view.options)
        assert all(badge for badge, _ in app.view.option_extras)       # every agent says whether it has a key
        app.on_key("esc")                           # cancel: the list closes and the app is usable again
        assert wait_for(lambda: app.view.options is None and not app.view.busy)
    finally:
        app.uninstall()


def test_tab_still_completes_a_command_and_does_nothing_in_the_middle_of_a_message(app, monkeypatch):
    opened = []
    monkeypatch.setattr(app, "_pick_agent", lambda: opened.append(1))
    app.view.editor.set("/ag")
    app.on_key("tab")
    assert app.view.editor.text.startswith("/agent") and opened == []
    app.view.editor.set("why did it stall")
    app.on_key("tab")
    assert opened == []
    app.view.editor.set("")
    app.on_key("tab")
    assert opened == [1]


def test_the_footer_names_the_key_that_works_right_now():
    empty, typing = view_for("idle"), view_for("typing")
    footer = lambda v: plain(tui.compose(v, 100, 30)[0][-1])
    assert "Tab agent" in footer(empty) and "Ctrl+O details" in footer(empty)
    assert "Ctrl+O agent" in footer(typing) and "Ctrl+G" not in footer(typing)
    assert "PgUp/PgDn scroll" in footer(empty) and "drag" not in footer(empty)
    assert "Copy" in "\n".join(plain(line) for line in tui.compose(empty, 100, 30)[0])