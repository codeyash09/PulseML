"""Sweep: cli.py (the `pulse` entry point) and install-sudo's safety check.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_* are regression coverage for behaviour verified to work.
"""
import ast
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import cli  # noqa: E402
from pulse import pulse_cli  # noqa: E402
from pulse import pulse_console  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolate everything run_script mutates: argv, path, __main__, cwd, env, hooks."""
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(pulse_cli, "_RESTART_ARGV_HOOK", getattr(pulse_cli, "_RESTART_ARGV_HOOK", None), raising=False)
    monkeypatch.setattr(cli, "_RUN_MODE", cli._RUN_MODE)
    monkeypatch.delenv(pulse_cli._RESTART_CHILD_ENV, raising=False)
    monkeypatch.delenv("PULSE_MODE", raising=False)
    monkeypatch.delenv("PULSE_AGENT_LOG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "already_running", lambda path: [])
    # never start real tracking: run the script as it is
    monkeypatch.setattr(cli.PulseASTInjector, "visit", lambda self, tree: tree)
    return tmp_path


def _injected_index(source):
    tree = ast.parse(source)
    cli.PulseASTInjector().visit(tree)
    for i, stmt in enumerate(tree.body):
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "auto_track"):
            return i
    return None


# ---------------------------------------------------------------------------------------
# bugs
# ---------------------------------------------------------------------------------------

def test_bug_launch_option_hint_drops_the_cwd_value(capsys):
    """`pulse --cwd proj train.py` is answered with a suggested command. `offered` keeps
    only the option token '--cwd' (not its value) and the 'script' is the first
    non-dash argument -- 'proj' -- so it suggests `pulse run --cwd proj` (no script)
    and `pulse proj`."""
    rc = cli.main(["--cwd", "proj", "train.py"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "pulse run --cwd proj train.py" in out, out


def test_bug_auto_track_injected_after_code_when_imports_are_scattered():
    """Notebook-exported scripts import in every cell. auto_track() goes 'after the last
    top-level import', i.e. after the training code of the earlier cells has already
    run untracked (and a crash there never reaches Pulse). It must precede the first
    statement that executes user code."""
    src = ("import numpy as np\n"
           "X = np.zeros(3)\n"
           "for epoch in range(3):\n"
           "    X = X + 1\n"
           "from sklearn.metrics import f1_score\n"
           "print(f1_score)\n")
    assert _injected_index(src) == 1


def test_bug_sys_argv0_is_not_as_typed(sandbox):
    """_set_process_view documents that sys.argv[0] 'stays as typed, also as under
    python', but run_script passes the abspath, so `pulse run train.py` gives the script
    sys.argv[0] == '/abs/.../train.py' where `python train.py` gives 'train.py'."""
    (sandbox / "train.py").write_text(
        "import sys\nopen('argv0.txt', 'w').write(sys.argv[0])\n")
    rc = cli.run_script("train.py", [])
    assert rc == 0
    assert (sandbox / "argv0.txt").read_text() == "train.py"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_bug_unreadable_script_crashes_with_traceback(sandbox):
    """A script that exists but cannot be read (permissions) raises PermissionError out
    of run_script -- only decode errors are caught -- so `pulse run` dies with a Python
    traceback instead of an error message and exit status 1."""
    script = sandbox / "train.py"
    script.write_text("print(1)\n")
    script.chmod(0)
    try:
        assert cli.run_script("train.py", []) == 1
    finally:
        script.chmod(0o644)


def test_bug_install_sudo_refuses_every_symlinked_pulse(tmp_path):
    """world_writable_by_others() lstat()s the path, and a symlink's own mode is always
    0777 on Linux. pipx (and many venv setups) put `pulse` in ~/.local/bin as a SYMLINK,
    so `pulse install-sudo` always refuses with 'writable by anybody' -- while the file it
    points to, which root would actually exec, is never checked at all."""
    real_dir = tmp_path / "venv" / "bin"
    real_dir.mkdir(parents=True)
    real = real_dir / "pulse"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    for d in (tmp_path, tmp_path / "venv", real_dir):
        d.chmod(0o755)
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    link_dir.chmod(0o755)
    link = link_dir / "pulse"
    os.symlink(real, link)
    assert pulse_console.world_writable_by_others(str(link)) is None


def test_bug_install_sudo_does_not_check_symlink_target(tmp_path):
    """The converse: the symlink's target lives in a world-writable directory, which is
    exactly the case the check exists for, but only the link's own path is walked."""
    if pulse_console.world_writable_by_others(str(tmp_path)) is not None:
        pytest.skip("tmp_path itself sits under a world-writable dir")
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_dir.chmod(0o777)
    real = open_dir / "pulse"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    safe = tmp_path / "safe"
    safe.mkdir()
    safe.chmod(0o755)
    link = safe / "pulse"
    os.symlink(real, link)
    assert pulse_console.world_writable_by_others(str(link)) is not None
    # and the symlink bits alone must not be what triggers it:
    assert pulse_console.world_writable_by_others(str(link)) != str(link)


# ---------------------------------------------------------------------------------------
# ok
# ---------------------------------------------------------------------------------------

def test_ok_parse_run_args():
    assert cli._parse_run_args(["train.py", "--lr", "3"]) == (False, None, False, "train.py", ["--lr", "3"])
    assert cli._parse_run_args(["--stream", "--again", "--cwd", "d", "t.py", "--stream"]) == (
        True, "d", True, "t.py", ["--stream"])
    assert cli._parse_run_args(["--cwd=d", "--", "-weird.py"]) == (False, "d", False, "-weird.py", [])
    for bad in ([], ["--cwd"], ["--bogus", "t.py"], ["--agent-log=", "t.py"]):
        with pytest.raises(cli._UsageError):
            cli._parse_run_args(bad)


def test_ok_agent_log_sets_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PULSE_AGENT_LOG", raising=False)
    cli._parse_run_args(["--agent-log", "t.py"])
    assert os.environ["PULSE_AGENT_LOG"] == str(tmp_path / "pulse_agent.log")
    cli._parse_run_args(["--agent-log=logs/a.log", "t.py"])
    assert os.environ["PULSE_AGENT_LOG"] == str(tmp_path / "logs" / "a.log")


def test_ok_parse_code_args():
    assert cli._parse_code_args(["-p", "do it", "-y", "a.py", "--cwd", "d"]) == ("do it", True, "d", ["a.py"])
    assert cli._parse_code_args(["--prompt=x", "--", "-y"]) == ("x", False, None, ["-y"])
    with pytest.raises(cli._UsageError):
        cli._parse_code_args(["-p"])
    with pytest.raises(cli._UsageError):
        cli._parse_code_args(["--nope"])


def test_ok_restart_argv_comes_back_under_pulse_run(monkeypatch):
    monkeypatch.setattr(cli, "_RUN_MODE", "stream")
    argv = cli._restart_argv("/usr/bin/python3", "/p/train.py", ["--lr", "1"])
    assert argv[0] == "/usr/bin/python3" and argv[1] == "-c"
    assert "['run', '--stream']" in argv[2]
    assert argv[3:] == ["/p/train.py", "--lr", "1"]


def test_ok_injector_prefers_main_guard_and_keeps_line_numbers():
    src = ("import os\n"
           "def main():\n    pass\n"
           "if __name__ == '__main__':\n    main()\n")
    tree = ast.parse(src)
    cli.PulseASTInjector(mode="stream").visit(tree)
    guard = tree.body[2]
    call = guard.body[0].value
    assert call.func.attr == "auto_track"
    assert call.keywords[0].value.value == "stream"
    assert guard.body[0].lineno == 5
    assert _injected_index("'''doc'''\nx = 1\n") == 1
    assert _injected_index("x = 1\n") == 0


def test_ok_already_uses_pulse():
    assert cli._already_uses_pulse(ast.parse("from pulse import auto_track\n"))
    assert cli._already_uses_pulse(ast.parse("import pulse.pulse_cli as p\n"))
    assert not cli._already_uses_pulse(ast.parse("import pulses\nimport pulse_utils\n"))


def test_ok_exit_code():
    assert cli._exit_code(SystemExit()) == 0
    assert cli._exit_code(SystemExit(3)) == 3
    assert cli._exit_code(SystemExit("msg")) == 1


def test_ok_run_script_runs_like_python(sandbox):
    (sandbox / "helper.py").write_text("VALUE = 7\n")
    (sandbox / "train.py").write_text(
        "import sys, helper\n"
        "open('out.txt', 'w').write(f'{__name__} {helper.VALUE} {sys.argv[1:]}')\n"
        "sys.exit(4)\n")
    assert cli.run_script("train.py", ["--x", "1"]) == 4
    assert (sandbox / "out.txt").read_text() == "__main__ 7 ['--x', '1']"


def test_ok_run_script_missing_and_bad_cwd(sandbox, capsys):
    assert cli.run_script("nope.py", []) == 1
    (sandbox / "t.py").write_text("x = 1\n")
    assert cli.run_script("t.py", [], cwd=str(sandbox / "missing")) == 1


def test_ok_restarted_child_with_syntax_error_does_not_repair(sandbox, monkeypatch):
    monkeypatch.setenv(pulse_cli._RESTART_CHILD_ENV, "1")
    (sandbox / "t.py").write_text("def f(:\n")
    called = []
    monkeypatch.setattr(cli, "_repair_and_restart", lambda *a: called.append(a) or 0)
    assert cli.run_script("t.py", []) == 1
    assert called == []


def test_ok_main_help_and_usage(capsys):
    assert cli.main(["--help"]) == 0
    assert cli.main(["run", "--help"]) == 0
    assert cli.main(["run"]) == 1
    assert cli.main(["frobnicate"]) == 1
    assert "pulse run" in capsys.readouterr().out


def test_ok_world_writable_detects_open_dir(tmp_path):
    d = tmp_path / "open"
    d.mkdir()
    d.chmod(0o777)
    f = d / "pulse"
    f.write_text("x")
    assert pulse_console.world_writable_by_others(str(f)) == str(d)
    d.chmod(0o1777)                      # sticky: others cannot replace our file
    assert pulse_console.world_writable_by_others(str(f)) is None
