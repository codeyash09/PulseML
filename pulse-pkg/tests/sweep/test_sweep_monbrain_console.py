"""Sweep: the console (`pulse`, `pulse watch`, `pulse sessions`) and `pulse attach`.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_*  are regression coverage for behaviour verified to work.
"""
import builtins
import contextlib
import io
import os
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_attach as attach          # noqa: E402
from pulse import pulse_console as console        # noqa: E402
from pulse import pulse_stream as stream          # noqa: E402
from pulse.pulse_monitor import Monitor           # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_HOME", str(tmp_path / "pulse_home"))
    monkeypatch.delenv("PULSE_STREAM_DIR", raising=False)
    monkeypatch.delenv("PULSE_MODEL", raising=False)
    yield


def _spool(directory, session_id="s", steps=3, pid=None, started=None):
    monitor = Monitor(directory=str(directory), session_id=session_id, interval=0.0)
    for i in range(steps):
        monitor.observe_locals({"loss": 1.0 / (i + 1)}, step=i + 1)
    monitor.snapshot_state()
    monitor.writer.flush(2.0)
    if pid is not None or started is not None:
        info = stream.StreamReader(str(directory)).session()
        if pid is not None:
            info["pid"] = pid
        if started is not None:
            info["started"] = started
        monitor.writer.write_session(info)
    return monitor


def _feed(monkeypatch, lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr(builtins, "input", fake_input)


# =====================================================================================
# finding runs / status
# =====================================================================================

def test_bug_a_crashed_run_is_listed_as_finished(tmp_path):
    """The stream-mode excepthook records {'crashed': True} and then detach()->close() adds
    {'finished': True}; pulse_monitor.snapshot_state keeps both flags ('flags like crashed
    stick' -- so a CUDA OOM is not 'listed as having finished normally'). But
    _describe_session tests `finished` before `crashed`, so every crashed run is shown as
    'finished' anyway (and in dim, not red).
    Correct: crashed wins."""
    monitor = _spool(tmp_path / "run")
    monitor.snapshot_state({"crashed": True})
    monitor.close()
    info = console._describe_session(str(tmp_path / "run"))
    assert info["status"] == "crashed"


def test_bug_pid_alive_says_dead_for_another_users_process():
    """_pid_alive uses os.kill(pid, 0) and treats every OSError as 'not alive'. EPERM means
    the process exists but belongs to someone else (a run started with sudo, by another
    user on a shared box, or pid 1). Such a live run is listed as 'ended'. The same mistake
    is in AttachedMonitor._alive (see the attach test).
    Correct: PermissionError -> alive."""
    if os.geteuid() == 0:
        pytest.skip("root can signal pid 1")
    assert console._pid_alive(1), "pid 1 (always running) reported dead"


def test_bug_a_recycled_pid_makes_a_dead_run_look_stalled_and_get_auto_attached(tmp_path):
    """Status is 'alive(pid) and not finished' -> 'stalled'. A run killed without writing
    `finished` (SIGKILL, OOM killer, pre-emption) whose pid has since been reused by an
    unrelated process is shown as 'stalled' forever, and pick_session treats live/stalled
    runs as 'the obvious one', so `pulse` auto-attaches to a corpse. The module already has
    _process_started(); it is not used here.
    Correct: a process that started after the session did is not the session's process."""
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.3)
        d = tmp_path / "run"
        _spool(d, pid=other.pid, started=time.time() - 3600)      # run began an hour ago
        old = time.time() - 600
        os.utime(str(d / "events.jsonl"), (old, old))              # quiet for 10 minutes
        info = console._describe_session(str(d))
        assert info["status"] == "ended", f"a dead run with a recycled pid is {info['status']!r}"
    finally:
        other.kill()
        other.wait()


def test_bug_an_out_of_range_index_silently_attaches_to_some_other_run():
    """`pulse watch 4` with two runs listed: 4 is not a valid index, so matching falls back
    to substring-matching '4' against session ids -- which are timestamps full of digits --
    and attaches to whichever run happens to contain a 4.
    Correct: an all-digit argument that is not a valid index (and not an exact id) matches
    nothing, so the user is asked which one."""
    sessions = [
        {"session_id": "20260918-2210-ab12cd", "status": "live", "script": "/x/train.py"},
        {"session_id": "20260918-1804-ef34ab", "status": "ended", "script": "/x/ft.py"},
    ]
    assert console.pick_session(sessions, "4") is None


def test_bug_registry_under_sudo_goes_to_roots_home(tmp_path, monkeypatch):
    """Under sudo (HOME=/root, which is sudo's default on Ubuntu and what a system-wide
    `pulse` on secure_path gets), registry_dir() uses root's home. spool_dir_for already
    sends an attached run's spool to the INVOKING user's home for exactly this reason, but
    its registry pointer lands in /root/.pulse/sessions (and _makedirs then chowns the new
    /root/.pulse to the user). The user's next plain `pulse` cannot see the run: the spool is
    under ~/.pulse/attached, which the cwd scan never looks at.
    Correct: registry_dir() prefers invoking_user_home() when running under sudo."""
    user_home = tmp_path / "home" / "alice"
    root_home = tmp_path / "root"
    user_home.mkdir(parents=True)
    root_home.mkdir()
    monkeypatch.delenv("PULSE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(root_home))
    monkeypatch.setattr(stream, "invoking_user_home", lambda: str(user_home))
    assert stream.registry_dir().startswith(str(user_home))


def test_ok_discover_finds_a_run_started_from_another_directory(tmp_path, monkeypatch):
    project = tmp_path / "proj a ü"
    project.mkdir()
    script = project / "train.py"
    script.write_text("pass\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    d = stream.session_dir_for(str(script), "sid-x")
    monitor = Monitor(script_path=str(script), directory=d, session_id="sid-x", interval=0.0)
    monitor.observe_locals({"loss": 0.5}, step=7)
    monitor.snapshot_state()
    monitor.writer.flush(2.0)
    monkeypatch.chdir(str(elsewhere))
    runs = console.discover()
    assert [r["session_id"] for r in runs] == ["sid-x"]
    assert runs[0]["status"] == "live" and runs[0]["step"] == 7
    monitor.close()


def test_bug_an_oversized_pid_in_any_spool_crashes_pulse(tmp_path, monkeypatch):
    """_pid_alive guards against None/0/negative/non-int pids, but os.kill raises
    OverflowError (not OSError/ValueError/TypeError) for a pid beyond C long. discover()
    reads session.json of every spool under the cwd, so one corrupt or hand-edited spool
    anywhere below where `pulse` is typed makes `pulse`, `pulse sessions` and /sessions
    crash with a traceback.
    Correct: an impossible pid is simply 'not alive'."""
    d = tmp_path / "proj" / ".pulse_stream" / "bad"
    writer = stream.StreamWriter(str(d), flush_seconds=0.01)
    writer.write_session({"session_id": "bad", "pid": 2 ** 80, "started": time.time()})
    writer.emit(stream.KIND_SCALARS, {"step": 1, "values": {"loss": 1.0}})
    writer.close()
    monkeypatch.chdir(str(tmp_path))
    runs = console.discover()
    assert [r["session_id"] for r in runs] == ["bad"]
    assert runs[0]["status"] == "ended"


def test_ok_pid_alive_rejects_nonsense():
    for bad in (None, 0, -1, "x", ""):
        assert console._pid_alive(bad) is False
    assert console._pid_alive(os.getpid()) is True


def test_ok_process_started_reads_the_real_start_time():
    if not os.path.isdir("/proc"):
        pytest.skip("linux only")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        started = console._process_started(child.pid, console._boot_time())
        assert started is not None and abs(started - time.time()) < 5
    finally:
        child.kill()
        child.wait()


def test_ok_sparkline_and_compact_survive_nonfinite():
    assert console.sparkline([float("nan"), float("inf"), 1.0, 2.0]) != ""
    assert console.sparkline([float("nan")]) == ""
    assert console.compact(float("nan")) == "NaN" and console.compact(float("-inf")) == "-inf"


# =====================================================================================
# the running console
# =====================================================================================

def _console_with_slow_agent(tmp_path, delay=1.5):
    _spool(tmp_path / "run")
    calls = []
    started = threading.Event()

    def agent(prompt):
        calls.append(prompt)
        started.set()
        time.sleep(delay)
        return 'fine\n{"status": "ok", "risk": "low", "next_check_minutes": 30, "reason": "x"}'

    c = console.Console({"directory": str(tmp_path / "run"), "session_id": "s", "script": None},
                        agent=agent)
    c.brain.schedule.next_at = 0.0                 # a scheduled audit is due
    return c, calls, started


def test_bug_a_scheduled_audit_freezes_every_console_view(tmp_path):
    """The pump runs brain.poll_once() -- including a due audit, i.e. a model call that can
    take minutes (timeout=300) -- while holding _brain_lock. Every view (/status, /vars,
    /curve, /findings, the status line) takes the same lock in snapshot(), so the prompt
    hangs for the whole model call, and no new frames are ingested meanwhile.
    Correct: the model call happens outside _brain_lock; views return immediately."""
    c, calls, started = _console_with_slow_agent(tmp_path)
    c.start()
    try:
        assert started.wait(5), "test setup: scheduled audit never started"
        t0 = time.monotonic()
        c.snapshot()
        blocked = time.monotonic() - t0
    finally:
        c.stop(join=True)
    assert blocked < 0.3, f"/status blocked {blocked:.2f}s behind the scheduled audit"


def test_bug_slash_audit_during_a_scheduled_audit_bills_a_second_model_call(tmp_path):
    """Brain.audit has a non-blocking lock so two audits at once do not both call the model,
    and Console.audit has a 'busy' branch for it. But Console.audit takes the RLock
    _brain_lock first, which the pump holds for its whole scheduled audit, so /audit simply
    waits, then runs a second full audit: two model calls billed, the 'busy' branch is dead.
    Correct: /audit while an audit is running makes no second model call."""
    c, calls, started = _console_with_slow_agent(tmp_path, delay=1.0)
    c.start()
    try:
        assert started.wait(5)
        with contextlib.redirect_stdout(io.StringIO()):
            c.audit()
    finally:
        c.stop(join=True)
    assert len(calls) == 1, f"{len(calls)} model calls for one audit request"


def test_bug_a_command_that_fails_kills_the_whole_console(tmp_path, monkeypatch):
    """run_console's command loop has no exception handling. /interval and /stop write to
    the run's control.jsonl; if the spool is not writable by the person at the console (a
    colleague's run in a shared project dir, a spool left by `sudo pulse`, a deleted run),
    PermissionError propagates out of run_console and the console exits with a traceback.
    The same applies to any view that raises.
    Correct: the error is reported and the prompt continues."""
    if os.geteuid() == 0:
        pytest.skip("root ignores permissions")
    d = tmp_path / "run"
    monitor = _spool(d)
    session = console._describe_session(str(d))
    _feed(monkeypatch, ["/interval 1", "/status", "/quit"])
    os.chmod(str(d), 0o555)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = console.run_console(session, [session])
    finally:
        os.chmod(str(d), 0o755)
        monitor.close()
    assert code == 0
    assert "step" in out.getvalue()


def test_ok_console_commands_with_odd_arguments(tmp_path, monkeypatch):
    d = tmp_path / "run"
    monitor = _spool(d)
    session = console._describe_session(str(d))
    _feed(monkeypatch, ["", "/curve", "/curve nosuch", "/curve LOSS", "/vars", "/findings",
                        "/code", "/cd", "/interval abc", "/interval nan", "/interval 1e9",
                        "/trace", "/trace ::", "/trace x:nofile.py:notaline", "/bogus",
                        "/sessions", "/attach 99", "/quiet", "/loud", "/help",
                        "a question with no model", "/QUIT"])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = console.run_console(session, [session])
    monitor.close()
    assert code == 0
    text = out.getvalue()
    assert "No such command: /bogus" in text and "takes a number" in text


def test_ok_main_parses_model_value_without_taking_it_as_the_run(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(console, "discover", lambda extra_roots=None: [
        {"session_id": "a", "status": "live", "script": "/x/a.py", "step": 1, "directory": "/x"},
        {"session_id": "b", "status": "live", "script": "/x/b.py", "step": 1, "directory": "/x"}])
    monkeypatch.setattr(console, "unmonitored_python_processes", lambda: [])
    monkeypatch.setattr(console, "run_console",
                        lambda session, sessions, agent=None, sensitivity=0.3: seen.setdefault("s", session) and 0)
    monkeypatch.setattr("pulse.pulse_brain.build_litellm_agent", lambda model: (lambda p: ""))
    with contextlib.redirect_stdout(io.StringIO()):
        console.main(["watch", "--model", "gpt-x", "2"])
    assert seen["s"]["session_id"] == "b"


# =====================================================================================
# pulse attach
# =====================================================================================

def test_bug_attached_monitor_declares_another_users_process_finished():
    """AttachedMonitor._alive: os.kill(pid, 0) raising EPERM is taken as 'exited'. Watching a
    root- or other-user-owned run through sudo'd py-spy (a non-root console with sudo, or
    ptrace_scope 0), the sampler loop writes 'finished: process exited' after the first
    interval and stops, while the run carries on.
    Correct: PermissionError means the process exists."""
    if os.geteuid() == 0:
        pytest.skip("root can signal pid 1")
    monitor = object.__new__(attach.AttachedMonitor)
    monitor.pid = 1
    assert monitor._alive()


def test_bug_attach_skips_user_scripts_in_a_directory_named_pulse():
    """Same '/pulse/' marker as the monitor: a script at ~/pulse/train.py is treated as
    Pulse's own code, so values_from() returns nothing and attach reports 'module globals,
    cannot be read' for a loop that is plainly inside a function.
    Correct: only the installed pulse package directory is skipped."""
    threads = [{"frames": [{"filename": "/home/alice/pulse/train.py", "name": "train",
                            "locals": [{"name": "loss", "repr": "0.25"}]}]}]
    assert attach.values_from(threads) == {"loss": 0.25}


def test_bug_console_interval_and_stop_are_silently_ignored_by_an_attached_run(tmp_path, monkeypatch):
    """`pulse attach --pid N` produces a spool the console treats like any other, and the
    console offers /interval and /stop ('asked the run to stop'). AttachedMonitor never
    reads control.jsonl, so both are silently dropped: /stop leaves the run training after
    the user confirmed 'Stop the training run?', /interval changes nothing.
    Correct: the attached sampler honours set_interval (and handles or explicitly refuses stop)."""
    frames = [{"frames": [{"filename": "/home/u/proj/train.py", "name": "train",
                           "locals": [{"name": "loss", "repr": "0.5"},
                                      {"name": "step", "repr": "3"}]}]}]
    monkeypatch.setattr(attach, "read_frames", lambda *a, **k: (frames, ""))
    monkeypatch.setattr(attach, "pyspy_path", lambda: "/usr/bin/py-spy")
    monkeypatch.setattr(attach, "needs_root", lambda: False)
    monitor = attach.AttachedMonitor(os.getpid(), directory=str(tmp_path / "att"), interval=0.2)
    monitor.start()
    try:
        reader = stream.StreamReader(monitor.directory)
        reader.send_control(stream.CONTROL_SET_INTERVAL, interval=0.5)
        reader.send_control(stream.CONTROL_STOP, reason="console")
        time.sleep(1.5)
        interval = monitor.interval
    finally:
        monitor.stop()
    events = [f.get("event") for f in stream.StreamReader(monitor.directory).poll()]
    assert interval == 0.5, "set_interval ignored by the attached sampler"
    assert any(e in ("stop_requested", "stop_failed", "stop_unsupported") for e in events), \
        "stop was neither acted on nor refused"


def test_ok_attached_run_reads_numbers_and_flags_nonfinite(tmp_path, monkeypatch):
    frames = [{"frames": [
        {"filename": "/usr/lib/python3.12/site-packages/torch/x.py", "locals": [{"name": "z", "repr": "9"}]},
        {"filename": "/home/u/proj/train.py", "name": "train",
         "locals": [{"name": "loss", "repr": "nan"}, {"name": "step", "repr": "12"},
                    {"name": "name", "repr": "'abc'"}]}]}]
    monkeypatch.setattr(attach, "read_frames", lambda *a, **k: (frames, ""))
    monkeypatch.setattr(attach, "pyspy_path", lambda: "/usr/bin/py-spy")
    monkeypatch.setattr(attach, "needs_root", lambda: False)
    monitor = attach.AttachedMonitor(os.getpid(), directory=str(tmp_path / "att"))
    values = monitor.sample_once()
    monitor.stop()
    assert set(values) == {"loss", "step"} and monitor._step == 12
    got = stream.StreamReader(monitor.directory).poll()
    assert any(f.get("event") == "nonfinite" and f.get("urgent") for f in got)
    assert monitor.script == "/home/u/proj/train.py"


def test_ok_attach_spool_under_sudo_goes_to_the_users_home(tmp_path, monkeypatch):
    monkeypatch.setattr(stream, "invoking_user_home", lambda: str(tmp_path / "alice"))
    assert attach.spool_dir_for("/etc/cron.d/evil.py", "sid").startswith(str(tmp_path / "alice"))


# =====================================================================================
# `pulse <script>` / install-sudo
# =====================================================================================

def test_bug_pulse_script_with_no_sessions_never_looks_for_the_running_process(monkeypatch):
    """`pulse train.py` (and `sudo pulse train.py`, which watch_by_name's docstring calls
    'the whole command') is supposed to find the running train.py and watch it. main()
    returns early with 'Nothing Pulse is watching yet' whenever there are no Pulse sessions
    at all -- i.e. on every machine where Pulse has never streamed a run, the exact case of
    a run started with plain `python train.py` -- so watch_by_name is never reached.
    Correct: a named script is looked up among running processes even with no sessions."""
    called = []
    monkeypatch.setattr(console, "discover", lambda extra_roots=None: [])
    monkeypatch.setattr(console, "unmonitored_python_processes", lambda: [
        {"pid": 4242, "script": "/home/u/proj/train.py", "cmdline": "python train.py",
         "started": time.time() - 600}])
    monkeypatch.setattr(console, "watch_by_name",
                        lambda wanted, model="": called.append(wanted) or 0)
    with contextlib.redirect_stdout(io.StringIO()):
        console.main(["train.py"])
    assert called == ["train.py"], "the running train.py was never looked for"


def test_bug_install_sudo_refuses_every_symlinked_pulse_as_world_writable(tmp_path):
    """world_writable_by_others() lstat()s the path and treats mode & 0o002 as 'anyone can
    replace this'. A symlink's own mode is always 0o777 on Linux and means nothing, so for a
    pipx install (~/.local/bin/pulse is a symlink into the pipx venv) -- or any symlinked
    entry point -- `pulse install-sudo` refuses with 'is writable by anybody'. It also never
    checks the symlink's TARGET chain, which is what root would actually execute.
    Correct: symlinks are resolved; only real world-writable, non-sticky files/dirs count."""
    real_dir = tmp_path / "venv" / "bin"
    real_dir.mkdir(parents=True)
    os.chmod(str(tmp_path), 0o755)
    os.chmod(str(tmp_path / "venv"), 0o755)
    os.chmod(str(real_dir), 0o755)
    target = real_dir / "pulse"
    target.write_text("#!/bin/sh\n")
    os.chmod(str(target), 0o755)
    local_bin = tmp_path / "local" / "bin"
    local_bin.mkdir(parents=True)
    os.chmod(str(tmp_path / "local"), 0o755)
    os.chmod(str(local_bin), 0o755)
    link = local_bin / "pulse"
    os.symlink(str(target), str(link))
    assert console.world_writable_by_others(str(link)) is None


def test_ok_world_writable_detects_a_writable_directory(tmp_path):
    d = tmp_path / "open"
    d.mkdir()
    os.chmod(str(d), 0o777)
    f = d / "pulse"
    f.write_text("x")
    os.chmod(str(f), 0o755)
    assert console.world_writable_by_others(str(f)) == str(d)


def test_ok_launcher_quotes_a_path_with_spaces():
    text = console.launcher_text("/home/a b/.local/bin/pulse")
    assert "PULSE='/home/a b/.local/bin/pulse'" in text
