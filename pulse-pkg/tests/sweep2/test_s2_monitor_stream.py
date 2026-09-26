"""Sweep 2, area `monitor`: the wire (pulse_stream), the in-process monitor
(pulse_monitor), the py-spy attach sampler (pulse_attach) and stream mode's /stop.

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
FAKE_LLM = os.path.join(os.path.dirname(HERE), "sweep", "e2e_fake_llm")

from pulse import pulse_stream as stream          # noqa: E402
from pulse import pulse_monitor                   # noqa: E402
from pulse import pulse_attach                    # noqa: E402
from pulse import pulse_console                   # noqa: E402
from pulse.pulse_monitor import Monitor           # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_HOME", str(tmp_path / "pulse_home"))
    monkeypatch.delenv("PULSE_STREAM_DIR", raising=False)
    yield


def _read_all_frames(directory):
    frames = []
    for name in ("events.jsonl.prev", "events.jsonl"):
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            for raw in handle:
                try:
                    frames.append(json.loads(raw))
                except ValueError:
                    pass
    return frames


# =====================================================================================
# StreamWriter: rotation failure
# =====================================================================================

def test_bug_a_failed_rotation_kills_the_writer_thread_for_the_rest_of_the_run(tmp_path, monkeypatch):
    """_maybe_rotate closes the handle BEFORE os.replace. If the rename fails -- on Windows
    the brain holding events.jsonl open for its poll is a sharing violation (WinError 32);
    anywhere, EMFILE on the reopen -- the error is reported but self._handle stays the
    CLOSED file. The next _flush writes to it: ValueError('I/O operation on closed file'),
    which only `except OSError` guards, so it escapes _run and the writer thread dies.
    From then on nothing reaches the spool for the rest of the run: the brain sees a
    silent run that is still alive (never 'finished', never 'process gone').
    Correct: a failed rotation keeps (or reopens) a usable handle and the writer lives on."""
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=0.02, max_bytes=1500)
    real_replace = os.replace
    failed = []

    def replace(src, dst):
        if str(dst).endswith(".prev") and not failed:
            failed.append(dst)
            raise PermissionError(13, "The process cannot access the file because it is "
                                      "being used by another process")
        return real_replace(src, dst)

    monkeypatch.setattr(stream.os, "replace", replace)
    for step in range(40):                       # well past max_bytes: rotation is attempted
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 1.0 / (step + 1)}})
        writer.flush(2.0)
    assert failed, "the test never exercised a rotation"
    for step in range(40, 45):
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 0.01}})
    writer.flush(2.0)
    alive = writer._thread.is_alive()
    writer.close()
    steps = {f.get("step") for f in _read_all_frames(str(tmp_path)) if f.get("kind") == "scalars"}
    assert alive, "the writer thread died after one failed rotation"
    assert {40, 41, 42, 43, 44} <= steps, "frames after the failed rotation never reached disk"


def test_ok_rotation_and_drain_keep_every_frame(tmp_path):
    """Regression: with a working rename, one rotation between polls loses nothing."""
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=5.0, max_bytes=2000)
    reader = stream.StreamReader(str(tmp_path))
    seen = []
    for step in range(60):
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 1.0}})
        assert writer.flush(2.0)
        if step % 7 == 0:
            seen += [f["step"] for f in reader.poll() if f.get("kind") == "scalars"]
    seen += [f["step"] for f in reader.poll() if f.get("kind") == "scalars"]
    writer.close()
    assert reader.rotations >= 1
    assert len(seen) + reader.gaps == 60
    assert seen == sorted(seen)


# =====================================================================================
# StreamWriter/StreamReader: sequence numbers out of order -> false drops
# =====================================================================================

def test_bug_two_emitting_threads_produce_false_dropped_frame_counts(tmp_path, monkeypatch):
    """emit() takes the sequence number under _seq_lock, releases it, then builds the item
    (time.time()) and puts it on the queue. Two threads emit in stream mode -- the sampler
    thread (scalars) and the training thread (monitor.event / observe, the crash hook) --
    so a thread switch in that window puts seq N+1 on the queue before seq N. The reader
    counts holes as `seq > last + 1`, so the reordering is booked as a lost frame, and the
    brain tells the agent 'N sample(s) were dropped under load; the curves have holes'
    and /status says 'dropped under load' about a stream that lost nothing.
    Correct: no frame was dropped, so reader.gaps == 0.
    (The thread switch is forced deterministically with a gate inside time.time.)"""
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=5.0)
    reader = stream.StreamReader(str(tmp_path))
    writer.emit(stream.KIND_HELLO, {})
    assert writer.flush(2.0)
    reader.poll()

    real_time = time.time
    gate = threading.Event()
    entered = threading.Event()

    def gated_time():
        if threading.current_thread().name == "sampler-like":
            entered.set()
            gate.wait(5.0)
        return real_time()

    monkeypatch.setattr(stream.time, "time", gated_time)
    slow = threading.Thread(target=lambda: writer.emit(stream.KIND_SCALARS,
                                                        {"step": 1, "values": {"loss": 1.0}}),
                            name="sampler-like")
    slow.start()
    assert entered.wait(5.0)
    writer.emit(stream.KIND_EVENT, {"event": "phase", "step": 1})   # training thread
    gate.set()
    slow.join(5.0)
    monkeypatch.setattr(stream.time, "time", real_time)
    assert writer.flush(2.0)
    frames = reader.poll()
    writer.close()
    assert len([f for f in frames if f.get("kind") in ("scalars", "event")]) == 2
    assert reader.gaps == 0, f"{reader.gaps} frame(s) reported lost; none were"


def test_ok_real_drops_are_counted_once(tmp_path):
    """Regression: frames evicted by backpressure are counted exactly once."""
    writer = stream.StreamWriter(str(tmp_path), max_frames=8, flush_seconds=5.0)
    reader = stream.StreamReader(str(tmp_path))
    hold = threading.Event()
    real_encode = writer._encode

    # Keep the writer thread busy so the queue overflows.
    def slow_encode(item):
        hold.wait(5.0)
        return real_encode(item)

    writer._encode = slow_encode
    for step in range(40):
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 1.0}})
    hold.set()
    time.sleep(0.5)                              # flush() refuses outright on a full queue
    assert writer.flush(3.0)
    frames = reader.poll()
    writer.close()
    got = [f["step"] for f in frames if f.get("kind") == "scalars"]
    assert len(got) + reader.gaps == 40


# =====================================================================================
# StreamWriter: queued snapshots when the writer is backed up
# =====================================================================================

def test_bug_a_newer_queued_snapshot_is_discarded_in_favour_of_an_older_one(tmp_path):
    """queue_state() stamps each snapshot with the current _state_generation, and the writer
    thread writes a queued one only if its stamp still equals the generation. But writing a
    queued snapshot goes through write_state(), which bumps the generation too. So with two
    snapshots queued (the writer is more than one snapshot interval behind -- a slow or
    networked disk, exactly when queue_state exists), the OLDER one is written, bumps the
    generation, and the NEWER one is thrown away as 'stale'. state.json -- what a late
    brain and the `pulse` session list read -- goes back in time.
    Correct: when two snapshots are queued, state.json ends with the newer one."""
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=0.05)
    with writer._state_lock:                     # the writer thread is stuck on a slow disk
        assert writer.queue_state({"step": 100, "which": "older"})
        assert writer.queue_state({"step": 200, "which": "newer"})
        time.sleep(0.2)
    assert writer.flush(3.0)
    writer.close()
    with open(os.path.join(str(tmp_path), "state.json"), encoding="utf-8") as handle:
        state = json.load(handle)
    assert state.get("which") == "newer", f"state.json holds the older snapshot: {state}"


def test_ok_a_direct_write_is_not_overwritten_by_an_older_queued_snapshot(tmp_path):
    """Regression for the first sweep's fix: close()'s direct `finished` write wins."""
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=0.05)
    with writer._state_lock:
        assert writer.queue_state({"step": 1})
        writer.write_state({"step": 2, "finished": True})
    assert writer.flush(3.0)
    writer.close()
    with open(os.path.join(str(tmp_path), "state.json"), encoding="utf-8") as handle:
        assert json.load(handle).get("finished") is True


# =====================================================================================
# StreamWriter.poll_control: the copy of the text-mode reader nobody fixed
# =====================================================================================

def test_bug_control_poll_raises_on_a_half_written_multibyte_character(tmp_path):
    """The first sweep moved StreamReader to binary reads because a line cut in the middle
    of a multi-byte character raised UnicodeDecodeError. poll_control -- the same tail
    logic in the other direction -- still reads in text mode and catches only OSError. A
    control line being appended (json.dumps(..., ensure_ascii=False): a /track of a
    variable called `损失`, or any non-ASCII reason) that is caught mid-character raises
    straight out of Monitor.observe_locals -- into the training loop when observe_locals is
    called directly, or as a skipped sample + sampler_error in attach mode.
    Correct: a partial line is left for the next poll; nothing raises."""
    directory = str(tmp_path)
    monitor = Monitor(script_path=None, directory=directory, interval=0.0)
    try:
        line = json.dumps({"action": "track", "names": ["损失"]}, ensure_ascii=False) + "\n"
        data = line.encode("utf-8")
        cut = data.index("损".encode("utf-8")) + 1          # inside the character
        with open(os.path.join(directory, "control.jsonl"), "wb") as handle:
            handle.write(data[:cut])
        monitor._last_control_poll = 0.0
        monitor.observe_locals({"loss": 0.5, "step": 1})      # must not raise
        with open(os.path.join(directory, "control.jsonl"), "ab") as handle:
            handle.write(data[cut:])
        monitor._last_control_poll = 0.0
        monitor.observe_locals({"loss": 0.4, "step": 2})
        assert "损失" in monitor._requested
    finally:
        monitor.close()


# =====================================================================================
# AttachedMonitor.stop() racing an in-flight py-spy sample
# =====================================================================================

def test_bug_attached_stop_during_a_slow_sample_loses_the_finished_flag(tmp_path, monkeypatch):
    """stop() joins the sampler for 2 s, then writes state {'finished': True} and closes the
    writer. A py-spy dump may take up to _SAMPLE_TIMEOUT (20 s) -- a big process, a stopped
    one, sudo being slow. When the in-flight sample then returns, _run's
    `if self.samples % 5 == 0: self.snapshot()` rewrites state.json WITHOUT 'finished'
    (write_state does not check that the writer is closed). The target process is still
    running, so the session list shows this abandoned spool as 'stalled'/'live' and
    `pulse` auto-attaches to a stream nobody is writing.
    Correct: after stop() returns and the sampler thread ends, state.json says finished."""
    release = threading.Event()
    entered = threading.Event()

    def slow_read_frames(pid, pyspy=None, use_sudo=None, non_interactive=True):
        entered.set()
        release.wait(10.0)
        return ([{"frames": [{"filename": "/home/u/train.py", "locals": [
            {"name": "loss", "repr": "0.5"}, {"name": "step", "repr": "5"}]}]}], "")

    monkeypatch.setattr(pulse_attach, "read_frames", slow_read_frames)
    monkeypatch.setattr(pulse_attach, "MIN_INTERVAL", 0.01)
    monitor = pulse_attach.AttachedMonitor(os.getpid(), script="/home/u/train.py",
                                           interval=0.01, directory=str(tmp_path / "spool"))
    monitor.samples = 4                          # the in-flight one is the 5th
    monitor.start()
    assert entered.wait(5.0)
    monitor.stop()                               # join times out after 2 s
    release.set()
    monitor._thread.join(5.0)
    assert not monitor._thread.is_alive()
    with open(os.path.join(monitor.directory, "state.json"), encoding="utf-8") as handle:
        state = json.load(handle)
    assert state.get("finished") is True, f"state after stop(): {state}"


def test_ok_attached_step_zero_and_nonfinite(tmp_path, monkeypatch):
    """Regression: step 0.0 is a step, and a NaN reading raises an urgent event."""
    monkeypatch.setattr(pulse_attach, "read_frames", lambda *a, **k: ([{"frames": [
        {"filename": "/home/u/train.py", "locals": [
            {"name": "loss", "repr": "nan"}, {"name": "step", "repr": "0"}]}]}], ""))
    monitor = pulse_attach.AttachedMonitor(os.getpid(), script="/home/u/train.py",
                                           directory=str(tmp_path / "spool"))
    values = monitor.sample_once()
    monitor.writer.flush(2.0)
    frames = _read_all_frames(monitor.directory)
    monitor.discard()
    assert values["step"] == 0.0 and monitor._step == 0
    assert any(f.get("event") == "nonfinite" and f.get("urgent") for f in frames)


# =====================================================================================
# Monitor: pause / resume / interval
# =====================================================================================

def test_ok_pause_then_resume_through_the_control_file(tmp_path):
    """Regression: a paused monitor still reads control, so /resume gets through."""
    monitor = Monitor(directory=str(tmp_path), interval=0.0)
    reader = stream.StreamReader(str(tmp_path))
    try:
        reader.send_control(stream.CONTROL_PAUSE, reason="x")
        monitor._last_control_poll = 0.0
        monitor.observe_locals({"loss": 1.0, "step": 1})
        assert monitor.paused
        reader.send_control(stream.CONTROL_RESUME)
        monitor._last_control_poll = 0.0
        monitor.observe_locals({"loss": 0.9, "step": 2})
        assert not monitor.paused
        reader.send_control(stream.CONTROL_SET_INTERVAL, interval=1e9)
        monitor._last_control_poll = 0.0
        monitor.observe_locals({"loss": 0.8, "step": 3})
        assert monitor.interval == pulse_monitor.MAX_INTERVAL
    finally:
        monitor.close()


# =====================================================================================
# Stream mode /stop, end to end in a real training process
# =====================================================================================

_STOP_SCRIPT = r'''
import os, sys, time
sys.path.insert(0, os.environ["PULSE_SRC"])
from pulse import pulse as P

def train():
    loss = 1.0
    for step in range(1000000):
        loss = loss * 0.9999 + 0.001
        time.sleep(0.01)

monitor = P._start_stream_monitor(sys._getframe(), 0.4)
sys.stdout.write("DIR=" + monitor.directory + "\n"); sys.stdout.flush()
train()
'''


def test_bug_stream_mode_stop_is_recorded_as_a_crash(tmp_path):
    """The console's /stop makes the monitor raise KeyboardInterrupt in the training thread
    ('what Ctrl-C does'). Stream mode's excepthook (pulse.py _start_stream_monitor) then
    records it exactly like a real failure: an urgent 'crash' event and a sticky
    state {'crashed': True}. The file-based hook right above it skips KeyboardInterrupt;
    this one does not. So a run the user deliberately stopped -- or Ctrl-C'd -- is listed
    in red as 'crashed' by `pulse`, and the agent is handed a KeyboardInterrupt 'crash'
    as evidence.
    Correct: a stop/Ctrl-C is not a crash -- state has no 'crashed', status is not 'crashed'."""
    script = tmp_path / "train.py"
    script.write_text(_STOP_SCRIPT)
    env = dict(os.environ, PULSE_SRC=SRC, PULSE_HOME=str(tmp_path / "home"),
               PULSE_STREAM_DIR=str(tmp_path / "spool"), CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([FAKE_LLM, SRC]), MPLBACKEND="Agg")
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    proc = subprocess.Popen([sys.executable, str(script)], cwd=str(tmp_path), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        directory = None
        deadline = time.time() + 120
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("DIR="):
                directory = line[4:].strip()
                break
        assert directory, proc.stderr.read()
        reader = stream.StreamReader(directory)
        deadline = time.time() + 30
        while time.time() < deadline and not any(
                f.get("kind") == "scalars" for f in reader.poll()):
            time.sleep(0.2)
        reader.send_control(stream.CONTROL_STOP, reason="asked from the Pulse console")
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    frames = _read_all_frames(directory)
    assert any(f.get("event") == "stop_requested" for f in frames), "stop never arrived"
    state = stream.StreamReader(directory).state()
    status = pulse_console._describe_session(directory)["status"]
    assert not state.get("crashed") and status != "crashed", \
        f"a deliberate /stop is listed as {status!r}; crash events: " \
        f"{[f.get('exception') for f in frames if f.get('event') == 'crash']}"


# =====================================================================================
# Found by reading in sweep 2; tests added with the fix (they fail on b6abffe)
# =====================================================================================

def test_bug_one_bad_control_message_does_not_lose_the_rest_of_its_batch(tmp_path):
    """A malformed message (names that is a number, a line that is not an object) used
    to raise out of _handle_control and take every later message of the batch with it."""
    directory = str(tmp_path)
    monitor = Monitor(script_path=None, directory=directory, interval=0.0)
    try:
        with open(os.path.join(directory, "control.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"action": "track", "names": 5}) + "\n")
            handle.write("[1, 2]\n")
            handle.write(json.dumps({"action": "track", "names": "grad_norm"}) + "\n")
            handle.write(json.dumps({"action": "pause", "reason": "x"}) + "\n")
        monitor._last_control_poll = 0.0
        monitor.observe_locals({"loss": 0.5, "step": 1})
        assert monitor.paused, "the /pause after a bad message was lost"
        assert monitor._requested == {"grad_norm"}
    finally:
        monitor.close()


@pytest.mark.parametrize("path, library", [
    ("/usr/lib/python3.11/threading.py", True),
    ("/home/u/.venv/lib/python3.12/site-packages/torch/nn/module.py", True),
    ("/usr/lib/python3/dist-packages/numpy/core.py", True),
    ("/opt/conda/lib/python3.10/json/__init__.py", True),
    ("C:\\Python311\\Lib\\threading.py", True),
    ("C:\\Users\\u\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\queue.py", True),
    ("C:\\Users\\u\\proj\\.venv\\Lib\\site-packages\\torch\\x.py", True),
    ("<frozen importlib._bootstrap>", True),
    ("/home/u/lib/python-experiments/train.py", False),
    ("/home/u/lib/python/train.py", False),
    ("/home/u/code/lib/train.py", False),
    ("C:\\Users\\u\\lib\\train.py", False),
    ("/home/u/python3.11/train.py", False),
])
def test_bug_the_library_path_filter(path, library):
    assert pulse_monitor.is_library_path(path) is library
    frame = {"filename": path}
    assert pulse_attach._is_user_frame(frame) is (not library)


def test_bug_reattaching_does_not_revive_the_old_sampler(tmp_path):
    """detach() gives up joining after 1 s; the old sampler, still inside a slow sample,
    then saw attach() clear the shared stop event and ran on beside the new one."""
    first = pulse_monitor.attach(directory=str(tmp_path / "a"), interval=0.05)
    old_stop = pulse_monitor._STOP
    pulse_monitor.detach()
    second = pulse_monitor.attach(directory=str(tmp_path / "b"), interval=0.05)
    try:
        assert second is not first
        assert old_stop.is_set(), "attach() cleared the stop signal of the old sampler"
        assert pulse_monitor._STOP is not old_stop and not pulse_monitor._STOP.is_set()
    finally:
        pulse_monitor.detach()


def test_bug_flush_waits_for_room_in_a_full_queue(tmp_path):
    """flush() on a momentarily full queue waits (within its timeout) instead of failing."""
    writer = stream.StreamWriter(str(tmp_path), max_frames=8, flush_seconds=5.0)
    hold = threading.Event()
    busy = threading.Event()
    real_encode = writer._encode

    def slow_encode(item):
        busy.set()
        hold.wait(5.0)
        return real_encode(item)

    writer._encode = slow_encode
    writer.emit(stream.KIND_SCALARS, {"step": 0, "values": {"loss": 1.0}})
    assert busy.wait(5.0)                        # the writer thread is stuck on a slow disk
    for step in range(1, 20):                    # ... while the queue fills up
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 1.0}})
    assert writer._queue.full()
    threading.Timer(0.3, hold.set).start()
    try:
        assert writer.flush(5.0), "flush gave up on a full queue"
    finally:
        hold.set()
        writer.close()
