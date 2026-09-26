"""Sweep 2, area `monitor`: the brain (pulse_brain) and the console (pulse_console).

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_*  are regression coverage for behaviour verified to work.
"""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("PULSE_SRC") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_stream as stream          # noqa: E402
from pulse import pulse_brain                     # noqa: E402
from pulse import pulse_console                   # noqa: E402
from pulse.pulse_brain import Brain               # noqa: E402
from pulse.pulse_monitor import Monitor           # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_HOME", str(tmp_path / "pulse_home"))
    monkeypatch.delenv("PULSE_STREAM_DIR", raising=False)
    yield


def _spool(directory, steps=5, finish=True, script=None):
    monitor = Monitor(script_path=script, directory=str(directory), interval=0.0)
    for step in range(steps):
        monitor.observe("loss", 1.0 / (step + 1), step=step)
    monitor.snapshot_state()
    if finish:
        monitor.close()
    else:
        monitor.writer.flush(2.0)
    return monitor


# =====================================================================================
# Brain: an audit answer without a usable next_check keeps the audit due
# =====================================================================================

@pytest.mark.parametrize("answer", [
    "The run looks healthy; nothing to add.",                         # no JSON at all
    'Fine.\n{"status": "ok", "risk": "low", "findings": []}',         # JSON, no interval
    'Fine.\n{"status": "ok", "next_check_minutes": "later"}',          # unparseable interval
])
def test_bug_an_audit_without_a_next_check_is_billed_again_on_every_poll(tmp_path, answer):
    """_audit() re-arms the schedule only through apply_decision(), which returns early
    when the reply has no numeric next_check_minutes. A reply without it -- no JSON (cut
    off at max_tokens, the model forgot), JSON without that key, or 'later' -- leaves
    next_at in the past, so poll_once() finds the audit due again on the very next poll
    (0.5 s later in the console pump and Brain.run) and pays for another full-evidence
    model call, back to back, for as long as the model keeps answering that way.
    (tests/test_brain.py only checks the interval is unchanged, not that the next look
    was pushed out.) Correct: one audit, then the next one is at least the minimum
    interval away."""
    _spool(tmp_path / "s", finish=False)
    calls = []
    brain = Brain(str(tmp_path / "s"), agent=lambda prompt: calls.append(prompt) or answer,
                  poll_interval=0.0)
    brain.schedule.next_at = time.time() - 1          # the first audit is due
    for _ in range(5):
        brain.poll_once()
    assert len(calls) == 1, f"{len(calls)} model calls in 5 polls"
    assert brain.schedule.seconds_remaining() >= pulse_brain.MIN_INTERVAL_SECONDS - 1


def test_ok_a_well_formed_audit_reschedules(tmp_path):
    _spool(tmp_path / "s", finish=False)
    calls = []
    answer = 'ok\n{"status": "ok", "risk": "low", "findings": [], "next_check_minutes": 30, "reason": "x"}'
    brain = Brain(str(tmp_path / "s"), agent=lambda p: calls.append(p) or answer)
    brain.schedule.next_at = time.time() - 1
    for _ in range(3):
        brain.poll_once()
    assert len(calls) == 1
    assert 1790 <= brain.schedule.seconds_remaining() <= 1800


# =====================================================================================
# Console: the pump keeps auditing a run that is over
# =====================================================================================

def test_bug_console_pump_keeps_paying_for_audits_of_a_finished_run(tmp_path, monkeypatch):
    """Brain.run() stops once the run is finished (or its process is gone -- the first
    sweep's fix for 'billing audits on a killed run forever'). The console does not use
    run(): its pump calls brain.poll_once() every 0.5 s for as long as the console is open,
    and poll_once audits whenever the schedule is due, finished or not. `pulse` on
    yesterday's run with --model, left open in a tmux pane, pays for a full-evidence audit
    of a dead run every interval, indefinitely (and immediately on attach, since the
    persisted next_at is long past).
    Correct: once the run is finished the pump audits it at most once.
    (MIN_INTERVAL_SECONDS is shrunk so 'every interval' fits in a test.)"""
    _spool(tmp_path / "s", finish=True)
    monkeypatch.setattr(pulse_brain, "MIN_INTERVAL_SECONDS", 0.2)
    calls = []
    answer = '{"status": "ok", "risk": "low", "findings": [], "next_check_minutes": 0.001}'
    session = pulse_console._describe_session(str(tmp_path / "s"))
    assert session["status"] == "finished"
    console = pulse_console.Console(session, agent=lambda p: calls.append(p) or answer)
    monkeypatch.setattr(console, "_print", lambda text: None)
    console.brain.poll_once()
    assert console.brain.finished
    console.brain.schedule.next_at = time.time() - 1
    console.start()
    time.sleep(3.0)
    console.stop(join=True)
    assert len(calls) <= 1, f"{len(calls)} paid audits of a finished run in 3 s"


# =====================================================================================
# session_process_alive: wall-clock steps
# =====================================================================================

def test_bug_a_wall_clock_step_makes_a_live_run_look_dead_and_ends_the_brain(tmp_path, monkeypatch):
    """session_process_alive compares the process start time, computed as /proc/stat btime
    + starttime ticks, with the session's `started` (time.time() when the monitor began).
    btime is not a constant: the kernel derives it as (wall clock - uptime), so when the
    wall clock is stepped forward after the run started -- NTP's first sync on a VM or a
    board without an RTC, WSL2 resyncing after the host slept, a container host's clock
    fix -- every process's computed start time moves forward by the same step. With a
    step over 5 s the run's own, still-running process 'started after its session' and is
    declared dead: `pulse` lists the live run as 'ended', and Brain.run() ends
    ('process_gone') after 2 s of quiet while the run trains on unwatched.
    Correct: a step of the wall clock does not make the session's own process look dead
    (compare a clock-independent identity, e.g. the starttime ticks recorded in session.json)."""
    boot = pulse_console._boot_time()
    if boot is None:
        pytest.skip("needs /proc/stat")
    me = os.getpid()
    began = pulse_console._process_started(me, boot)
    started = time.time()                        # the monitor starts now, in this process
    assert began is not None and began <= started + 1
    assert pulse_console.session_process_alive(me, started)
    # NTP steps the clock forward by one minute: /proc/stat btime now reads a minute later.
    monkeypatch.setattr(pulse_console, "_boot_time", lambda: boot + 60.0)
    assert pulse_console.session_process_alive(me, started), \
        "this very process is reported dead after a 60 s clock step"


def test_bug_brain_run_ends_on_a_live_quiet_run_after_a_clock_step(tmp_path, monkeypatch):
    """The user-visible end of the bug above: a live run that goes quiet for a few seconds
    (a long eval, a checkpoint save, /pause) after a clock step is abandoned by the brain.
    Correct: Brain.run() keeps watching a run whose process is alive."""
    if pulse_console._boot_time() is None:
        pytest.skip("needs /proc/stat")
    monitor = _spool(tmp_path / "s", finish=False)          # pid = this live process
    boot = pulse_console._boot_time()
    monkeypatch.setattr(pulse_console, "_boot_time", lambda: boot + 60.0)
    brain = Brain(str(tmp_path / "s"), poll_interval=0.1)
    deadline = time.time() + 4.0
    brain.run(stop=lambda: time.time() > deadline)
    monitor.close()
    assert not any(e.get("event") == "process_gone" for e in brain.events), \
        "the brain gave up on a live run"


def test_ok_brain_run_ends_when_the_process_is_really_gone(tmp_path):
    """Regression: a SIGKILLed run (no finished, no BYE) ends the brain after the quiet."""
    directory = tmp_path / "s"
    code = ("import sys, os, time; sys.path.insert(0, %r)\n"
            "from pulse.pulse_monitor import Monitor\n"
            "m = Monitor(directory=%r, interval=0.0)\n"
            "m.observe('loss', 1.0, step=1); m.writer.flush(2.0)\n"
            "os.kill(os.getpid(), 9)\n") % (SRC, str(directory))
    subprocess.run([sys.executable, "-c", code], timeout=60)
    brain = Brain(str(directory), poll_interval=0.1)
    started = time.time()
    brain.run(stop=lambda: time.time() - started > 15)
    assert brain.finished and any(e.get("event") == "process_gone" for e in brain.events)


# =====================================================================================
# Console: choosing which run `pulse <script>` means
# =====================================================================================

def _session(sid, script, status, started):
    return {"session_id": sid, "directory": "/tmp/" + sid, "script": script,
            "status": status, "started": started, "step": 10, "pid": 1}


def test_bug_pulse_train_py_attaches_to_pretrain_py(tmp_path):
    """matching_sessions accepts `wanted in basename(script)`, a substring test, next to the
    exact-name test. `pulse train.py` therefore also matches pretrain.py (and my_train.py,
    train.py.bak ...). With a live pretrain.py run and yesterday's train.py run, the match
    list holds a live run, so main() does not look for the running train.py process, and
    pick_session returns the one live match -- the console attaches to pretrain.py, which
    the user did not ask for.
    Correct: an exact script-name match wins; pretrain.py is not a match for train.py."""
    sessions = [_session("20260926-1000-aaaaaa", "/home/u/proj/pretrain.py", "live", 200.0),
                _session("20260925-0900-bbbbbb", "/home/u/proj/train.py", "ended", 100.0)]
    matched = pulse_console.matching_sessions(sessions, "train.py")
    chosen = pulse_console.pick_session(sessions, "train.py")
    assert all(os.path.basename(s["script"]) == "train.py" for s in matched), \
        [s["script"] for s in matched]
    assert chosen is None or os.path.basename(chosen["script"]) == "train.py", chosen["script"]


def test_ok_out_of_range_index_matches_nothing():
    sessions = [_session("20260926-1044-aaaaaa", "/x/a.py", "live", 1.0)]
    assert pulse_console.matching_sessions(sessions, "4") == []
    assert pulse_console.pick_session(sessions, "1") is sessions[0]


# --- real processes on this machine --------------------------------------------------

_SLEEPER = "import time\nloss = 1.0\ntime.sleep(90)\n"


@pytest.fixture(scope="module")
def running_scripts(tmp_path_factory):
    """Two plain `python <script>` runs, old enough (>5 s) for the console to list."""
    base = tmp_path_factory.mktemp("procs")
    procs = {}
    for name in ("train.py", "pulse_rate_net.py"):
        path = base / name
        path.write_text(_SLEEPER)
        procs[name] = (subprocess.Popen([sys.executable, str(path)], cwd=str(base)), str(path))
    time.sleep(6.0)
    yield procs
    for proc, _ in procs.values():
        proc.kill()
        proc.wait()


def test_bug_a_users_script_named_pulse_something_is_invisible(running_scripts):
    """_is_pulse_itself() still treats every file whose name STARTS with 'pulse' as one of
    Pulse's own processes. The first sweep narrowed it from 'pulse anywhere in the path',
    but a user's pulse_rate_net.py / pulse_classifier.py (ECG, radar, laser-pulse models)
    is hidden from unmonitored_python_processes(), so `pulse pulse_rate_net.py` says no
    process is running it and `pulse` never mentions it. Pulse never runs its own
    modules as `python pulse_x.py` (it uses `-m pulse...` or the `pulse` entry point).
    Correct: only Pulse's actual files (or its package dir) count as Pulse itself."""
    proc, path = running_scripts["pulse_rate_net.py"]
    listed = {p["pid"] for p in pulse_console.unmonitored_python_processes()}
    assert proc.pid in listed, "the user's running pulse_rate_net.py is not listed"


def test_bug_an_abandoned_attach_spool_hijacks_pulse_script(running_scripts, tmp_path, monkeypatch):
    """`sudo pulse train.py` watches a plain run from outside and writes a spool whose pid
    is the TARGET's. If that console dies without its finally (ssh drops -> SIGHUP, the
    terminal is closed, kill), state.json never gets 'finished' and the registry pointer
    stays. The target keeps training, so _describe_session calls the abandoned spool
    'stalled' forever. From then on `pulse train.py` finds a 'live' match, never looks for
    the process, and attaches to the frozen spool; the process itself is also dropped
    from the unmonitored list because its pid counts as monitored. Re-attaching needs
    `pulse attach --pid`, which nothing suggests.
    Correct: a spool whose sampler is gone is not a live run; `pulse train.py` goes on to
    watch the running process (watch_by_name)."""
    proc, script = running_scripts["train.py"]
    session_id = "20260926-0100-pid%d" % proc.pid
    directory = tmp_path / "attached" / session_id
    directory.mkdir(parents=True)
    info = {"session_id": session_id, "script": script, "pid": proc.pid,
            "started": time.time() - 3, "attached": True, "sampler": "py-spy"}
    (directory / "session.json").write_text(json.dumps(info))
    (directory / "events.jsonl").write_text(json.dumps({"seq": 1, "kind": "hello"}) + "\n")
    (directory / "state.json").write_text(json.dumps({"step": 7, "scalars": {"loss": 0.5},
                                                      "attached": True}))
    old = time.time() - 120
    os.utime(str(directory / "events.jsonl"), (old, old))
    stream.register_session(session_id, str(directory), info)

    called = {}
    monkeypatch.setattr(pulse_console, "watch_by_name",
                        lambda wanted, model="": called.setdefault("watch", wanted) and 0)
    monkeypatch.setattr(pulse_console, "run_console",
                        lambda session, sessions, agent=None, sensitivity=0.3:
                        called.setdefault("console", session["session_id"]) and 0)
    monkeypatch.chdir(str(tmp_path))
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        pulse_console.main(["train.py"])
    assert called.get("console") != session_id, \
        "`pulse train.py` attached to an abandoned spool nobody is writing"
    assert called.get("watch") == "train.py"
