"""The 2026-10-11 audit, prompts/keys/CLI (P) and agent core (A): one focused check per fix.
No model is called and nothing leaves the machine."""
import json
import math
import os
import re
import stat
import sys
import types

import pytest

import pulse.pulse_cli as pc
from pulse import pulse as core
from pulse import pulse_app as appmod
from pulse import pulse_code
from pulse import pulse_code_agent as native
from pulse import pulse_console as con
from pulse import pulse_openrouter as orr
from pulse import pulse_settings as settings
from pulse import pulse_supabase as cloud
from pulse import pulse_terminal as term
from pulse import pulse_ui as ui
from pulse.pulse_cli import PulseCLI

KEY = "sk-or-v1-" + "a1b2c3d4" * 8
PLAIN = re.compile(r"\033\[[0-9;]*m")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    saved_env, saved_providers, cwd = dict(os.environ), dict(pc.PROVIDERS), os.getcwd()
    for name in (orr.ENV_KEY, "PULSE_PROVIDER", "PULSE_CONFIG", "PULSE_APPROVER", "PULSE_OPENROUTER_SAVED_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "pulsehome" / "credentials.json")
    monkeypatch.setattr(cloud, "save_cached_profile", lambda **k: None)
    monkeypatch.setattr(orr, "_request_json", lambda *a, **k: pytest.fail("a test reached OpenRouter"))
    yield
    ui.set_host(None)
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(saved_env)
    pc.PROVIDERS.clear()
    pc.PROVIDERS.update(saved_providers)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "train.py").write_text("lr = 0.3\n")
    return root


@pytest.fixture
def cli(project, monkeypatch):
    c = appmod._new_cli(str(project), [str(project / "train.py")])
    c.agent_provider = next(n for n, info in pc.PROVIDERS.items() if info.get("model") and info.get("env_key"))
    c.agent_key = "sk-test"
    c._sync_agent_turn = lambda *a, **k: None
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: pytest.fail("no model call expected here"))
    c._approver_settings = lambda: None                    # a person answers, not a model
    return c


# ---- P1, P2, P3: the flagged-command and apply-change questions ------------------------

def test_choose_takes_a_title_and_hands_it_to_the_app(monkeypatch):
    seen = {}

    class Host:
        def choose(self, options, initial=0, hotkeys=None, title=""):
            seen["title"] = title
            return 0

    ui.set_host(Host())
    assert ui.choose([ui.Option("Run")], title="Run flagged command? rm x") == 0
    assert seen["title"] == "Run flagged command? rm x"
    ui.set_host(None)
    monkeypatch.setattr(ui, "enabled", lambda: False)
    with pytest.raises(ui.Unavailable):                  # not a TypeError any more
        ui.choose([ui.Option("Run")], title="Q")


def test_flagged_command_without_the_app_falls_back_to_the_text_question(cli, monkeypatch):
    monkeypatch.setattr(ui, "enabled", lambda: False)
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: "y")
    assert cli._confirm_terminal_command("rm x", {"deletes_files": True}) is True
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: "a")      # one letter no longer means always
    assert cli._confirm_terminal_command("rm x", {"deletes_files": True}) is False
    assert not getattr(cli, "_always_allow_terminal_commands", False)


def test_always_has_no_single_key_shortcut(cli, monkeypatch):
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    asked = []

    def choose(options, **kwargs):
        asked.append((options, kwargs))
        return 0

    monkeypatch.setattr(pc._ui, "choose", choose)
    cli._confirm_terminal_command("rm x", {"deletes_files": True})
    cli.review = True
    cli._review_change_with_approver = lambda *a: None
    monkeypatch.setattr(pulse_code, "render_diff", lambda *a: "diff")
    monkeypatch.setattr(ui, "show_change", lambda *a: None)
    native._confirm(types.SimpleNamespace(cli=cli, request="r", root=str(cli._project_root)),
                    "train.py", "a", "b", False)
    for options, kwargs in asked:
        always = [o for o in options if o.label.startswith("Always")]
        assert always and always[0].key is None and not kwargs.get("hotkeys")
        assert kwargs.get("title")


def test_ctrl_c_at_the_apply_question_means_not_applied(cli, monkeypatch):
    cli.review = True
    cli._review_change_with_approver = lambda *a: None
    monkeypatch.setattr(pulse_code, "render_diff", lambda *a: "diff")
    monkeypatch.setattr(ui, "show_change", lambda *a: None)

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(ui, "choose", interrupted)
    state = types.SimpleNamespace(cli=cli, request="r", root=str(cli._project_root))
    assert native._confirm(state, "train.py", "a", "b", False) is False


def test_always_allowed_commands_are_still_shown_and_logged(cli, monkeypatch, capsys):
    logged = []
    monkeypatch.setattr(pc, "_agent_log_event", lambda title, body="": logged.append((title, body)))
    cli._always_allow_terminal_commands = True
    assert cli._confirm_terminal_command("rm -rf build", {"deletes_files": True}) is True
    assert "rm -rf build" in PLAIN.sub("", capsys.readouterr().out)
    assert logged and "always-allow" in logged[0][0] and "rm -rf build" in logged[0][1]


# ---- P4..P7, P11, P12: keys ---------------------------------------------------------------

def test_logout_removes_the_key_from_both_stores():
    assert settings.save_key(orr.ENV_KEY, KEY) and orr.save_key(KEY)
    assert orr.forget_key() is True
    assert orr.saved_key() is None and settings.saved_key(orr.ENV_KEY) == ""
    settings.save_key(orr.ENV_KEY, KEY)                  # only in keys.json still counts as signed in
    assert orr.forget_key() is True and settings.load_keys_into_environment() == []


def test_a_custom_agent_does_not_inherit_an_earlier_save_the_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdf"))
    c.non_interactive = False
    c._remember_agent_key = True                          # left over from an earlier pick
    custom = next(n for n, info in pc.PROVIDERS.items() if info.get("custom"))
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    monkeypatch.setattr(pc._ui, "enabled", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: custom)
    answers = iter(["ollama_chat/llama3.1", ""])
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: next(answers))
    assert c._select_agent_provider_and_key(initial=True)
    assert c._remember_agent_key is False


def test_openrouter_login_declined_says_the_key_is_not_usable(monkeypatch, capsys):
    monkeypatch.setattr(orr, "sign_in", lambda **k: KEY)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert orr.main(["login"]) == 0
    out = capsys.readouterr().out
    assert "not saved" in out and "only for this session" not in out and "Sign in again" in out


def test_openrouter_login_with_saving_off_says_so_before_signing_in(monkeypatch, capsys):
    settings.set("remember_keys", "off")
    monkeypatch.setattr(orr, "sign_in", lambda **k: pytest.fail("signed in although nothing could keep the key"))
    assert orr.main(["login"]) == 1
    out = capsys.readouterr().out
    assert "remember_keys" in out and "writable" not in out


def test_key_screen_text_matches_whether_keys_are_saved(monkeypatch):
    notes = []
    monkeypatch.setattr(pc._ui, "enabled", lambda: True)
    monkeypatch.setattr(pc._ui, "header", lambda *a, **k: None)
    monkeypatch.setattr(pc._ui, "kv", lambda *a, **k: None)
    monkeypatch.setattr(pc._ui, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(pc._ui, "note", notes.append)
    pc._ui_key_screen("OpenRouter (x)", openrouter=True)
    text = " ".join(notes)
    assert "never written to disk" not in text and "only if you consent" in text
    notes.clear()
    settings.set("remember_keys", "off")
    pc._ui_key_screen("OpenRouter (x)", openrouter=True)
    text = " ".join(notes)
    assert "asks whether to save" not in text and "does not keep it" in text


# ---- P9: /experiment outside the plain CLI loop ------------------------------------------

def test_experiment_command_reaches_the_experiment_tool(capsys):
    seen = []
    fake = types.SimpleNamespace(
        _safe_tool=lambda name, arg, fn, *args: fn(*args),
        _run_experiment=lambda arg: seen.append(arg) or f"EXPERIMENT started for {arg}")
    assert pulse_code._handle_command(fake, "/experiment experiments/spec.json") is True
    assert seen == ["experiments/spec.json"]
    out = capsys.readouterr().out
    assert "EXPERIMENT started" in out and "Unknown command" not in out


# ---- P10: the Tk panel's sign-in flag ----------------------------------------------------

def test_dashboard_sign_in_flag_is_cleared_on_any_failure(monkeypatch):
    fake_tk = types.SimpleNamespace(Frame=type("Frame", (object,), {}), TclError=Exception)
    monkeypatch.setattr(core, "tk", fake_tk)
    monkeypatch.setattr(core, "HAS_TK", True)
    monkeypatch.setattr(core, "_CHAT_PANEL_CLS", None)
    cls = core._chat_panel_class()
    panel = cls.__new__(cls)
    panel.session_keys, panel.appended = {}, []
    panel.after = lambda ms, fn=None: fn() if fn else None
    panel._append = lambda who, text: panel.appended.append(text)

    def broken(**k):
        raise RuntimeError("port in use")

    monkeypatch.setattr(orr, "sign_in", broken)
    try:
        panel._openrouter_sign_in("OpenRouter").join(10)
    finally:
        core._CHAT_PANEL_CLS = None
    assert panel._openrouter_signing_in is False and "did not finish" in "\n".join(panel.appended)


# ---- P13: NaN in an incident ---------------------------------------------------------------

def test_an_incident_with_nan_is_valid_json(cli, monkeypatch):
    cli.debug_session_id = "s1"
    cli._incidents = []
    monkeypatch.setattr(cli, "_maybe_flush_cloud", lambda force=False: None)
    report = {"metrics": [1.0, float("nan"), float("inf")], "best": {"loss": -math.inf}, "ok": 2}
    cli._log_incident("experiment", "diverged", experiment=report)
    text = json.dumps(cli._incidents[-1], allow_nan=False)
    assert '"nan"' in text and '"-inf"' in text and '"ok": 2' in text
    assert math.isnan(report["metrics"][1])              # the caller's report is left alone


# ---- P14: stalled has its own colour -------------------------------------------------------

def test_stalled_and_live_are_painted_differently(monkeypatch):
    monkeypatch.setattr(con, "_COLOR", True)
    assert con.amber("x") != con.orange("x")


# ---- P16: the version is the package's -----------------------------------------------------

def test_version_comes_from_the_package_metadata(monkeypatch):
    import importlib.metadata
    import pulse
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.8.7" if name == "pulseml" else "")
    monkeypatch.delitem(pulse.__dict__, "__version__", raising=False)
    assert pulse.__version__ == "9.8.7"
    monkeypatch.delitem(pulse.__dict__, "__version__", raising=False)


# ---- A1, A2: nudge budgets -----------------------------------------------------------------

def _call(name, arguments, cid="c1"):
    return types.SimpleNamespace(id=cid, function=types.SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _reply(text, calls=()):
    message = types.SimpleNamespace(content=text, tool_calls=list(calls) or None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
                                 usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))


def test_an_open_outline_is_nudged_a_bounded_number_of_times(cli, monkeypatch):
    outline = [{"id": "a", "parent_id": "", "content": "Do it", "status": "in_progress"}]
    sent = []

    def completion(**kw):
        sent.append(kw["messages"][-1]["content"])
        return _reply("", [_call("todo_write", {"todos": outline})]) if len(sent) == 1 else _reply("Done.")

    monkeypatch.setattr(native.litellm, "completion", completion)
    assert native.run_native_turn(cli, "do the thing") == "answered"
    outline_nudges = [m for m in sent if isinstance(m, str) and "unfinished work" in m]
    assert len(outline_nudges) == native._MAX_OUTLINE_NUDGES
    assert len(sent) < 20


def test_outline_nudges_do_not_use_up_the_verify_nudge(cli, monkeypatch):
    state = native._State(cli, "r")
    state.todos = [{"id": "a", "parent_id": None, "content": "Do it", "status": "in_progress"}]
    state.dirty = True
    state.outline_nudges = native._MAX_OUTLINE_NUDGES       # its own budget is spent
    reason = native._nudge(state, "Done.", outline=False)
    assert reason and "smoke_test" in reason and state.nudges == 0


# ---- A3: a probe's exprs/expect are judged too ---------------------------------------------

def test_probe_expressions_are_classified_with_the_code(cli, monkeypatch):
    asked = []
    monkeypatch.setattr(cli, "_confirm_terminal_command",
                        lambda command, flags, purpose=None: asked.append(command) or False)
    state = native._State(cli, "r")
    report, why = native._run_probe(state, "x = 1", [], {"os.remove('train.py') or x": "()"}, 5, "probe")
    assert report is None and "NOT RUN" in why and "os.remove" in asked[0]
    asked.clear()
    report, why = native._run_probe(state, "", ["__import__('os').unlink('train.py')"], {}, 5, "probe")
    assert report is None and asked


# ---- A4: "always apply" is about code changes only ---------------------------------------

def test_always_apply_leaves_shell_confirmation_on(cli, monkeypatch):
    cli.review = True
    cli._always_apply_changes = True
    assert cli._terminal_needs_confirmation("rm -rf build") is not None
    pulse_code._handle_command(cli, "/review on")
    assert cli._always_apply_changes is False


# ---- A5, A10: rolling back --------------------------------------------------------------------

def _commit(cli, path, before, after):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(after)
    return cli._record_fix_commit({path: (before, after)}, "test change")


def test_rollback_last_touches_only_the_last_commit_and_not_newer_edits(cli, project):
    a, b = str(project / "a.py"), str(project / "b.py")
    for path in (a, b):
        open(path, "w").write("x = 0\n")
    _commit(cli, a, "x = 0\n", "x = 1\n")
    _commit(cli, b, "x = 0\n", "x = 2\n")
    open(a, "w").write("x = 'mine'\n")                    # the person edits a.py afterwards
    result = cli._run_rollback("last")
    assert "reverted" in result and open(b).read() == "x = 0\n" and open(a).read() == "x = 'mine'\n"

    _commit(cli, b, "x = 0\n", "x = 3\n")
    open(b, "w").write("x = 'mine too'\n")
    result = cli._run_rollback("last")
    assert "not done" in result and open(b).read() == "x = 'mine too'\n"


def test_revert_and_undo_keep_crlf_line_endings(cli, project):
    path = str(project / "w.py")
    with open(path, "w", newline="") as f:
        f.write("x = 1\r\ny = 2\r\n")
    # The log holds LF (as Pulse Code records it) while the disk copy is CRLF.
    cli._record_fix_commit({path: ("x = 0\ny = 2\n", "x = 1\ny = 2\n")}, "Pulse Code: change")
    assert pulse_code.undo(cli, quiet=True)
    assert open(path, newline="").read() == "x = 0\r\ny = 2\r\n"
    cli._run_rollback("last")                             # undoes the undo
    assert open(path, newline="").read() == "x = 1\r\ny = 2\r\n"


# ---- A6: replace_all with overlapping matches ------------------------------------------------

def test_replace_all_does_not_splice_overlapping_matches(cli, project):
    path = str(project / "s.py")
    open(path, "w").write("x = '==='\n")
    cli.review = False
    state = native._State(cli, "r")
    state.seen[path] = state.read(path)
    native._t_edit_file(state, {"path": "s.py", "old_string": "==", "new_string": "!", "replace_all": True})
    assert open(path).read() == "x = '!='\n"


# ---- A11: the todo schema ------------------------------------------------------------------

def test_parent_id_is_a_plain_string_and_empty_means_top_level(cli):
    todo = next(t for t in native.TOOLS if t["function"]["name"] == "todo_write")
    item = todo["function"]["parameters"]["properties"]["todos"]["items"]
    assert item["properties"]["parent_id"] == {"type": "string"}
    state = native._State(cli, "r")
    native._t_todo_write(state, {"todos": [{"id": "a", "parent_id": "", "content": "A", "status": "in_progress"}]})
    assert state.todos and state.todos[0]["parent_id"] is None


# ---- A13: a message is not an action ---------------------------------------------------------

def test_message_user_does_not_count_as_doing_something(cli):
    state = native._State(cli, "r")
    native._execute(state, {"id": "m", "name": "message_user", "arguments": json.dumps({"message": "I'll fix it"})})
    assert state.calls_made == 0
    assert "without doing it" in native._nudge(state, "I'll start by editing train.py.")


# ---- A14, A15: protected paths, atomic writes ------------------------------------------------

def test_protected_dirs_by_any_case_or_through_a_symlink(cli, project):
    (project / ".git").mkdir()
    os.symlink(str(project / ".git"), str(project / "gitlink"))
    state = native._State(cli, "r")
    for name in (".git/config", ".GIT/config", "gitlink/config", ".Pulse_History/x"):
        path, why = state.resolve(name)
        assert path is None and "protected" in why, name


def test_writes_replace_the_file_whole_and_keep_its_mode_and_symlink(cli, project):
    real = project / "real.py"
    real.write_text("old\n")
    os.chmod(real, 0o755)
    os.symlink(str(real), str(project / "link.py"))
    native._write_atomic(str(project / "link.py"), "new\n")
    assert os.path.islink(project / "link.py") and real.read_text() == "new\n"
    assert stat.S_IMODE(os.stat(real).st_mode) == 0o755
    assert not [n for n in os.listdir(project) if n.endswith(".pulse-tmp")]


# ---- A16: piping into an interpreter -------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["cat setup.sh | sh", "cat x.py | python3", "cat run.sh | bash -s",
                                 "git show HEAD:x.sh | bash"])
def test_piping_code_into_an_interpreter_is_flagged(cmd):
    assert term.classify_command(cmd)["reaches_outside_workspace"], cmd


@pytest.mark.parametrize("cmd", ["python x.py 2>&1 | tail", "cat log | grep loss", "bash run.sh",
                                 "ls | python3 -c \"import sys; print(len(sys.stdin.read()))\""])
def test_ordinary_pipes_are_not_flagged(cmd):
    assert not any(term.classify_command(cmd).values()), cmd
