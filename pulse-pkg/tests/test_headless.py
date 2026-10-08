"""Headless mode (pulse_headless): Pulse debugs a run by itself, in the background."""
import json
import os
import subprocess
import sys
import time
import types

import pytest

from pulse import cli as cli_mod
from pulse import pulse_app as appmod
from pulse import pulse_headless as headless
from pulse import pulse_supabase as cloud
from pulse import pulse_tui as tui


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "pulsehome" / "credentials.json")
    return tmp_path / "pulsehome"


class FakeCli:
    agent_provider = agent_key = agent_model_string = agent_api_base = None
    focus = []
    _project_root = None


def test_nobody_is_asked_anything(tmp_path):
    log = open(tmp_path / "log", "w")
    app = headless.HeadlessApp(FakeCli(), str(tmp_path), log)
    with pytest.raises(EOFError):
        app._ask(types.SimpleNamespace(label="Apply this change?"))
    log.close()
    assert "nobody to answer: no" in (tmp_path / "log").read_text()


def test_what_the_app_would_show_goes_to_the_log(tmp_path):
    log = open(tmp_path / "log", "w")
    app = headless.HeadlessApp(FakeCli(), str(tmp_path), log)
    app._add(tui.Entry("tool", calls=["TERMINAL: pytest -q"], output="3 passed"))
    app._add(tui.Entry("say", "Fixed the swapped matmul."))
    app._add(tui.Entry("finding", "loss is NaN", severity="critical"))
    app.feed("\x1b[32m[Pulse] colour and\x1b[K\x1b[0m clear codes\n")
    log.close()
    text = (tmp_path / "log").read_text()
    assert "Ran pytest -q" in text and "3 passed" in text and "agent: Fixed the swapped matmul." in text
    assert "⚠ CRITICAL loss is NaN" in text and "\x1b" not in text and "colour and clear codes" in text


def test_pulse_run_headless_starts_the_supervisor_and_returns(tmp_path, monkeypatch):
    script = tmp_path / "train.py"
    script.write_text("print(1)\n")
    started, asked = [], []
    monkeypatch.setattr(headless, "_spawn", lambda args, cwd, log: started.append((args, cwd)) or 4242)
    monkeypatch.setattr(headless, "choose_destination", lambda workspace, project, interactive=True: asked.append(
        (workspace, project)) or {"team_id": "t1", "project_id": "p1", "label": "workspace Lab, project MNIST"})
    assert cli_mod.main(["run", "--headless", "--to", "Lab/MNIST", "--cwd", str(tmp_path),
                         "train.py", "--epochs", "3"]) == 0
    (args, cwd), = started
    assert asked == [("Lab", "MNIST")]
    assert args[0] == "supervise" and args[-3:] == [str(script), "--epochs", "3"] and cwd == str(tmp_path)
    assert args[args.index("--team-id") + 1] == "t1" and args[args.index("--project-id") + 1] == "p1"


def test_to_needs_both_and_goes_with_headless(capsys):
    assert cli_mod.main(["run", "--to", "Lab/MNIST", "train.py"]) == 1
    assert "goes with --headless" in capsys.readouterr().out
    for half in ("Lab", "Lab/", "/MNIST"):
        assert cli_mod.main(["run", "--headless", "--to", half, "train.py"]) == 1
        assert "both of them" in capsys.readouterr().out
    assert headless.split_destination(" Lab / GPT small ") == ("Lab", "GPT small")


def test_list_and_stop(home, monkeypatch, capsys):
    (home / "headless").mkdir(parents=True)
    (home / "headless" / "1.json").write_text(json.dumps({"pid": os.getpid(), "script": "/p/train.py",
                                                          "log": "/l", "started": time.time()}))
    (home / "headless" / "2.json").write_text(json.dumps({"pid": 999999999, "script": "/p/gone.py", "log": "/l"}))
    assert headless.main([]) == 0
    out = capsys.readouterr().out
    assert "train.py" in out and "gone.py" not in out and not (home / "headless" / "2.json").exists()
    killed = []
    monkeypatch.setattr(headless.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    assert headless.main(["stop", "train.py"]) == 0 and killed[-1] == (os.getpid(), headless.signal.SIGTERM)


def test_auto_track_headless_starts_a_debugger_beside_the_run(monkeypatch):
    from pulse import pulse as core
    started = []
    monkeypatch.setattr(core, "_start_stream_monitor", lambda frame, interval, quiet=False: "monitor")
    monkeypatch.setattr(headless, "attach_in_background", lambda monitor: started.append(monitor))
    assert core._auto_track_session(sys._getframe(), None, 1.0, None, None, "headless") == "monitor"
    assert started == ["monitor"]


def test_a_headless_run_without_an_agent_is_watched_and_the_supervisor_ends(tmp_path, home):
    """For real: `pulse run --headless` on a short script, no model -- it starts, watches,
    and the supervisor exits after the run is over."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text(
        "import time\nloss = 1.0\nfor step in range(30):\n    loss *= 0.9\n    time.sleep(0.05)\nprint('DONE', loss)\n")
    src = os.path.dirname(os.path.dirname(headless.__file__))
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k not in ("PULSE_MODEL", "PULSE_PROVIDER")}
    env.update(PYTHONPATH=src, PULSE_HOME=str(home), PULSE_CACHE_DIR=str(home), PULSE_HEADLESS_GRACE="2")
    r = subprocess.run([sys.executable, "-c", "import sys; from pulse import cli; sys.exit(cli.main(sys.argv[1:]))",
                        "run", "--headless", "train.py"], cwd=str(project), env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and "Pulse is debugging it in the background" in r.stdout, r.stdout + r.stderr
    log = next((home / "headless").glob("*.log"))
    deadline = time.time() + 120
    while time.time() < deadline and "debugger stopped" not in log.read_text():
        time.sleep(1)
    text = log.read_text()
    assert "No agent is set up" in text and "Started under Pulse" in text, text
    assert "nothing is left to do" in text and "debugger stopped" in text, text
    assert not list((home / "headless").glob("*.json"))             # unregistered when it ends


def test_tensor_statistics_say_when_they_were_taken():
    from pulse import pulse_brain as brain
    pack = {"step": 199, "session": {}, "tensors": {"w": {"shape": [8, 1], "dtype": "float64"}},
            "tensor_stats": {"w": {"min": -0.1, "max": 0.2, "mean": 0.05, "nan": 0, "inf": 0, "taken_at_step": 29}}}
    assert "(values at step 29, the run is now at 199)" in brain.Brain.render_evidence(pack)


def test_run_status_asked_again_at_once_waits_for_progress(tmp_path, monkeypatch):
    app = appmod.App(FakeCli(), str(tmp_path))
    brain = types.SimpleNamespace(step=10)
    app.console = types.SimpleNamespace(brain=brain)
    app._status = "live"
    app._evidence = lambda: f"step {brain.step}"
    app._poll_status = lambda: setattr(brain, "step", brain.step + 30)
    assert app._agent_run_status() == "step 10"
    t = time.monotonic()
    assert app._agent_run_status() == "step 70" and time.monotonic() - t < 5     # waited for 50 steps



# ---------------------------------------------------------------- where the run goes online

TEAMS = [{"team_id": "t1", "join_code": "AAA", "name": "Lab", "admin_ids": []},
         {"team_id": "t2", "join_code": "BBB", "name": "Home", "admin_ids": []}]
PROJECTS = {"t1": [{"project_id": "p1", "name": "MNIST"}, {"project_id": "p2", "name": "GPT small"}],
            "t2": [{"project_id": "p3", "name": "Toys"}]}


@pytest.fixture
def signed_in(monkeypatch):
    from pulse import pulse_cli
    monkeypatch.setattr(cloud, "load_cached_credentials", lambda: {"user_id": "u1"})
    monkeypatch.setattr(cloud, "find_teams_for_user", lambda uid: TEAMS)
    monkeypatch.setattr(cloud, "find_projects_for_team", lambda team, uid: PROJECTS[team["team_id"]])
    monkeypatch.setattr(cloud, "save_cached_credentials", lambda *a, **k: None)
    monkeypatch.setattr(pulse_cli.PulseCLI, "_auth_flow", lambda self: setattr(self, "user_id", "u1"))
    monkeypatch.setattr(pulse_cli, "_ui_workspace_menu", lambda *a, **k: pytest.fail("the workspace picker was shown"))
    monkeypatch.setattr(pulse_cli, "_ui_project_menu", lambda *a, **k: pytest.fail("the project picker was shown"))


def test_flags_choose_the_workspace_and_project_and_are_remembered(signed_in):
    from pulse import pulse_settings as settings
    where = headless.choose_destination("lab", "gpt", interactive=False)
    assert where["team_id"] == "t1" and where["project_id"] == "p2" and "GPT small" in where["label"]
    assert settings.remembered_id("workspace") == "t1" and settings.remembered_id("project") == "p2"
    again = headless.choose_destination(interactive=False)            # next time: no flags, no question
    assert again["team_id"] == "t1" and again["project_id"] == "p2"


def test_a_flag_that_matches_nothing_says_what_there_is(signed_in):
    with pytest.raises(headless.DestinationError, match="Lab.*Home"):
        headless.choose_destination("nope", interactive=False)


def test_not_signed_in_and_nobody_to_ask_means_this_machine_only(monkeypatch):
    monkeypatch.setattr(cloud, "load_cached_credentials", lambda: None)
    assert headless.choose_destination(interactive=False) == {}


def test_the_headless_debugger_has_no_limit_on_automatic_fixes(tmp_path):
    log = open(tmp_path / "log", "w")
    app = headless.HeadlessApp(FakeCli(), str(tmp_path), log)
    assert app.auto_fix_limit is None and appmod.App(FakeCli(), str(tmp_path)).auto_fix_limit == 3



def test_with_nothing_remembered_the_pickers_ask_once(signed_in, monkeypatch):
    from pulse import pulse_cli
    from pulse import pulse_settings as settings
    monkeypatch.setattr(pulse_cli, "_ui_workspace_menu", lambda existing, cached, describe: "2")
    monkeypatch.setattr(pulse_cli, "_ui_project_menu", lambda projects, cached, describe, **k: "1")
    monkeypatch.setattr(pulse_cli, "_prompt_text", lambda *a, **k: "")     # "GitHub repo URL (optional)": skipped
    monkeypatch.setattr("builtins.input", lambda *a: "")
    where = headless.choose_destination(interactive=True)
    assert where["team_id"] == "t2" and where["project_id"] == "p3" and "Toys" in where["label"]
    assert settings.remembered_id("workspace") == "t2" and settings.remembered_id("project") == "p3"
