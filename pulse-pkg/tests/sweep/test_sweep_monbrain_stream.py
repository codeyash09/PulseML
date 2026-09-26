"""Sweep: the monitor/brain wire (pulse_stream), the in-process monitor (pulse_monitor)
and the brain (pulse_brain).

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_*  are regression coverage for behaviour verified to work.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_stream as stream          # noqa: E402
from pulse import pulse_monitor                   # noqa: E402
from pulse.pulse_monitor import Monitor           # noqa: E402
from pulse import pulse_brain                     # noqa: E402
from pulse.pulse_brain import Brain               # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # Never touch the real ~/.pulse registry.
    monkeypatch.setenv("PULSE_HOME", str(tmp_path / "pulse_home"))
    monkeypatch.delenv("PULSE_STREAM_DIR", raising=False)
    yield


def _steps(frames):
    return [f["step"] for f in frames if f.get("kind") == stream.KIND_SCALARS]


# =====================================================================================
# pulse_stream
# =====================================================================================

def test_bug_reader_loses_frames_across_a_rotation(tmp_path):
    """The writer rotates events.jsonl -> events.jsonl.prev. The reader only ever reads
    events.jsonl: whatever was appended to the old file after the reader's last poll (and
    before the rotation) is never read, and if the new file has already grown past the old
    offset the reader resumes mid-file in the NEW file, skipping its head as well.
    Correct: one rotation between two polls must not lose frames (drain the tail of .prev,
    or detect rotation by inode rather than by size)."""
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=5.0, max_bytes=4096)
    reader = stream.StreamReader(str(tmp_path))
    seen = []
    step = 0
    for _ in range(5):
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 1.0}})
        step += 1
        assert writer.flush(2.0)
    seen += _steps(reader.poll())
    prev = str(tmp_path / "events.jsonl.prev")
    # Keep writing, one flush per frame, until exactly one rotation has happened...
    while not os.path.exists(prev):
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 1.0, "pad": "x" * 60}})
        step += 1
        assert writer.flush(2.0)
    # ...and a little more into the fresh file.
    for _ in range(3):
        writer.emit(stream.KIND_SCALARS, {"step": step, "values": {"loss": 1.0}})
        step += 1
        assert writer.flush(2.0)
    seen += _steps(reader.poll())
    writer.close()
    missing = sorted(set(range(step)) - set(seen))
    assert not missing, f"{len(missing)} of {step} frames lost across one rotation: {missing[:10]}..."


def test_bug_reader_crashes_on_a_partial_multibyte_utf8_line(tmp_path):
    """The writer encodes with ensure_ascii=False, so non-ASCII variable names, paths and
    messages reach the file as multi-byte UTF-8, and a big batch is written in several
    syscalls. A reader that polls between them can see a final line cut in the middle of a
    character. StreamReader opens the file in text mode, so decoding that tail raises
    UnicodeDecodeError -- a ValueError, not the OSError it catches -- out of poll(), after the
    offset was already advanced past the complete lines before it: those frames are lost.
    Correct: complete lines are returned, the partial one is left for next time."""
    path = tmp_path / "events.jsonl"
    whole = json.dumps({"seq": 1, "t": 0, "kind": "scalars", "step": 1,
                        "values": {"pérte": 0.5}}, ensure_ascii=False).encode("utf-8") + b"\n"
    partial = json.dumps({"seq": 2, "t": 0, "kind": "scalars", "step": 2,
                          "values": {"pérte": 0.4}}, ensure_ascii=False).encode("utf-8")
    cut = partial.index("é".encode("utf-8")) + 1          # in the middle of the é
    path.write_bytes(whole + partial[:cut])
    reader = stream.StreamReader(str(tmp_path))
    frames = reader.poll()                                 # raises UnicodeDecodeError today
    assert _steps(frames) == [1]
    with open(path, "ab") as handle:
        handle.write(partial[cut:] + b"\n")
    assert _steps(reader.poll()) == [2]


def test_bug_crlf_lines_make_the_reader_offset_drift_and_duplicate_frames(tmp_path):
    """On Windows the writer's text-mode handle writes every '\\n' as '\\r\\n' (the module
    doc promises a Windows box 'behaves the same'). The reader reads in text mode with
    universal newlines, so each line comes back one byte shorter than it is on disk and the
    byte offset falls one byte further behind per line. After a few dozen lines the next
    poll re-reads whole frames it has already returned.
    Correct: offsets are counted in real bytes (read in binary, or newline=''), so a poll
    after N frames returns only what is new."""
    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"seq": i, "t": 0, "kind": "scalars", "step": i, "values": {"loss": 1.0}})
             for i in range(1, 101)]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    reader = stream.StreamReader(str(tmp_path))
    assert _steps(reader.poll()) == list(range(1, 101))
    with open(path, "ab") as handle:
        handle.write((json.dumps({"seq": 101, "t": 0, "kind": "scalars", "step": 101,
                                  "values": {"loss": 1.0}}) + "\r\n").encode())
    assert _steps(reader.poll()) == [101]


def test_bug_dropped_frames_are_counted_twice(tmp_path):
    """When the writer drops a frame it both leaves a hole in the seq numbers AND appends a
    KIND_DROP frame saying how many it dropped. A reader that has already seen a frame
    before the hole adds both to `gaps`, so every dropped frame is reported twice ('N
    samples were dropped' in the audit prompt and the console is double the truth).
    Correct: gaps equals the number of frames that are really missing."""
    writer = stream.StreamWriter(str(tmp_path), max_frames=4, flush_seconds=5.0)
    reader = stream.StreamReader(str(tmp_path))
    writer.emit(stream.KIND_SCALARS, {"step": 0, "values": {"loss": 1.0}})
    assert writer.flush(2.0)
    reader.poll()                                   # the brain is live and has seen seq 1
    for i in range(1, 400):
        writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0}})
    writer.flush_seconds = 0.01
    time.sleep(0.5)                                 # let the writer drain: nothing evicted at close
    writer.close()
    frames = reader.poll()
    present = 1 + len({f["seq"] for f in frames if f.get("kind") == stream.KIND_SCALARS})
    missing = 400 - present
    assert missing > 0, "test setup: nothing was dropped"
    assert reader.gaps == missing, f"reader reports {reader.gaps} dropped, really {missing}"


def test_bug_close_evicts_a_queued_frame_without_recording_the_drop(tmp_path):
    """StreamWriter.close(): when the queue is full, it makes room for the shutdown sentinel
    with a bare get_nowait() -- the evicted frame (typically the last values before the run
    ended) vanishes without being added to _dropped, so no KIND_DROP record covers it.
    Correct: every frame that never reaches the file is accounted for in a drop record."""
    writer = stream.StreamWriter(str(tmp_path), max_frames=4, flush_seconds=5.0)
    for i in range(400):
        writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0}})
    writer.flush_seconds = 0.01
    writer.close()
    frames = stream.StreamReader(str(tmp_path)).poll()
    seqs = [f["seq"] for f in frames if f.get("seq", -1) > 0]
    missing = max(seqs) - len(seqs)
    recorded = sum(int(f.get("dropped") or 0) for f in frames if f.get("kind") == stream.KIND_DROP)
    assert recorded == missing, f"{missing} frames missing, drop records account for {recorded}"


def test_bug_rotation_reopens_without_nofollow_or_handing_back(tmp_path, monkeypatch):
    """_maybe_rotate reopens events.jsonl with a plain open(): no O_NOFOLLOW and no
    _hand_back. Under `sudo pulse attach` the spool is written as root, so after the first
    64 MB rotation events.jsonl is root-owned (the user can no longer read/delete their own
    run) and a symlink planted at events.jsonl in the rotation window is followed as root.
    Correct: rotation reopens through _open_append like the initial open."""
    handed = []
    monkeypatch.setattr(stream, "_hand_back", lambda path, fd=None: handed.append(path))
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=5.0, max_bytes=512)
    handed.clear()
    while not os.path.exists(str(tmp_path / "events.jsonl.prev")):
        writer.emit(stream.KIND_SCALARS, {"step": 1, "values": {"pad": "x" * 100}})
        assert writer.flush(2.0)
    writer.close()
    assert writer.events_path in handed, "the rotated events.jsonl was never handed back"


def test_ok_frames_round_trip_with_unicode_names_across_polls(tmp_path):
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=0.01)
    reader = stream.StreamReader(str(tmp_path))
    got = []
    for i in range(50):
        writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"损失": 1.0 / (i + 1), "λ": 0.1}})
        if i % 7 == 0:
            writer.flush(2.0)
            got += _steps(reader.poll())
    writer.close()
    got += _steps(reader.poll())
    assert got == list(range(50))
    assert reader.gaps == 0


def test_ok_control_channel_handles_partial_and_unicode_lines(tmp_path):
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=0.01)
    reader = stream.StreamReader(str(tmp_path))
    reader.send_control(stream.CONTROL_TRACK, names=["权重"])
    with open(reader.control_path, "a", encoding="utf-8") as handle:
        handle.write('{"action": "untrack", "names": ["x"]')      # no newline yet
    assert [m["action"] for m in writer.poll_control()] == ["track"]
    with open(reader.control_path, "a", encoding="utf-8") as handle:
        handle.write("}\n")
    assert [m["action"] for m in writer.poll_control()] == ["untrack"]
    assert writer.poll_control() == []
    writer.close()


def test_ok_session_dir_and_registry_with_spaces_and_unicode(tmp_path, monkeypatch):
    home = tmp_path / "hé me" / "pulse home"
    monkeypatch.setenv("PULSE_HOME", str(home))
    script = tmp_path / "my project ü" / "train me.py"
    script.parent.mkdir()
    script.write_text("pass\n")
    d = stream.session_dir_for(str(script), "sid1")
    assert d == os.path.join(str(script.parent), ".pulse_stream", "sid1")
    assert stream.register_session("sid1", str(tmp_path), {"started": 1.0})
    assert [e["session_id"] for e in stream.registered_sessions()] == ["sid1"]
    stream.unregister_session("sid1")
    assert stream.registered_sessions() == []


def test_ok_emit_never_blocks_with_a_stalled_writer(tmp_path):
    writer = stream.StreamWriter(str(tmp_path), max_frames=16, flush_seconds=0.01)
    gate = threading.Event()
    original = writer._flush
    writer._flush = lambda pending: (gate.wait(5), original(pending))
    t0 = time.perf_counter()
    for i in range(5000):
        writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0}})
    elapsed = time.perf_counter() - t0
    gate.set()
    writer.close()
    assert elapsed < 1.0


# =====================================================================================
# pulse_monitor
# =====================================================================================

def test_bug_tensorflow_scalar_loss_is_invisible_to_the_monitor(tmp_path):
    """_scalar_of reads a 1-element value through `.item()`. TensorFlow EagerTensors and
    Keras 3 Variables have no .item(), so in a TF custom training loop the loss (an eager
    0-d tensor) is never sampled: no history, no NaN tripwire, nothing for the brain.
    Correct: a 0-d / 1-element host tensor without .item (TF: .numpy()) is still read."""
    tf = pytest.importorskip("tensorflow")
    monitor = Monitor(directory=str(tmp_path), session_id="tf", interval=0.0, tensor_interval=0.0)
    loss = tf.constant(0.75)
    bad = tf.constant(float("nan"))
    monitor.observe_locals({"loss": loss, "bad_loss": bad}, step=3)
    monitor.close()
    frames = stream.StreamReader(str(tmp_path)).poll()
    values = {}
    for f in frames:
        if f.get("kind") == stream.KIND_SCALARS:
            values.update(f["values"])
    assert values.get("loss") == pytest.approx(0.75), f"TF loss never sampled; saw {values}"
    assert any(f.get("event") == "nonfinite" for f in frames), "NaN TF loss raised no tripwire"


def test_bug_a_project_directory_named_pulse_hides_all_user_frames():
    """_is_user_frame skips any filename containing '/pulse/' so as to skip Pulse itself,
    but that also matches the USER's code when it lives in e.g. ~/pulse/train.py or
    /data/pulse/exp1/train.py. Every frame of such a script is treated as library code and
    the stream-mode sampler reports nothing at all. (pulse_console._is_pulse_itself was
    already fixed for exactly this reason; the monitor and pulse_attach were not.)
    Correct: skip only frames inside the installed pulse package directory."""
    class Code:
        co_filename = "/home/alice/pulse/train.py"

    class Frame:
        f_code = Code()

    assert pulse_monitor._is_user_frame(Frame()), "user's own script treated as Pulse internals"


def test_bug_a_huge_python_int_in_scope_kills_every_sample(tmp_path):
    """_scalar_of does float(value) on any Python int. An int beyond float range (a seed, a
    hash, a big-number product in scope) raises OverflowError, which aborts the whole
    observe_locals call: in stream mode every sample fails ('sampler_error') for as long as
    that local is alive, so the loss next to it is never recorded.
    Correct: an unrepresentable int is skipped (or read as inf) and the other values sampled."""
    monitor = Monitor(directory=str(tmp_path), session_id="big", interval=0.0, tensor_interval=0.0)
    try:
        monitor.observe_locals({"loss": 0.5, "seed_product": 10 ** 400}, step=1)
    finally:
        monitor.close()
    frames = stream.StreamReader(str(tmp_path)).poll()
    values = {}
    for f in frames:
        if f.get("kind") == stream.KIND_SCALARS:
            values.update(f["values"])
    assert values.get("loss") == 0.5


def test_bug_interval_control_cannot_make_stream_mode_sample_faster(tmp_path):
    """In stream mode (auto_track(mode='stream') -> pulse_monitor.attach) the pace is set by
    the sampler thread's own `interval`, captured once at attach(); the Monitor itself is
    created with interval=0. CONTROL_SET_INTERVAL (the console's '/interval <sec>', documented
    as 'sample faster or slower') only changes Monitor.interval, which gates AFTER the
    sampler's wait, so asking for faster sampling has no effect at all.
    Correct: /interval 0.05 on a run attached at 0.5 s actually samples ~every 0.05 s."""
    stop = threading.Event()
    ready = threading.Event()
    ident = {}

    def training():
        ident["id"] = threading.get_ident()
        ready.set()
        loss = 1.0
        while not stop.is_set():
            loss = loss * 0.999
            time.sleep(0.001)

    t = threading.Thread(target=training, daemon=True)
    t.start()
    ready.wait(2)
    monitor = pulse_monitor.attach(directory=str(tmp_path), session_id="rate", interval=0.5,
                                   depth=1, thread_id=ident["id"])
    try:
        stream.StreamReader(str(tmp_path)).send_control(stream.CONTROL_SET_INTERVAL, interval=0.05)
        time.sleep(2.0)                  # let the control message be picked up (<= ~1.5 s)
        reader = stream.StreamReader(str(tmp_path))
        reader.poll()
        monitor.writer.flush(2.0)
        reader.poll()
        time.sleep(2.0)
        monitor.writer.flush(2.0)
        n = len(_steps(reader.poll()))
    finally:
        pulse_monitor.detach()
        stop.set()
        t.join(2)
    # at 0.05 s that is ~40 samples in 2 s; at the original 0.5 s it is ~4.
    assert n >= 15, f"only {n} samples in 2 s after asking for 0.05 s sampling"


def test_bug_observe_locals_does_file_io_on_the_calling_thread(tmp_path):
    """The module promises 'Nothing here can block for longer than it takes to append to a
    queue', yet observe_locals itself calls snapshot_state() -> _atomic_write_json (mkstemp,
    write, rename) every 5 s, and poll_control() (stat + open + read) every 1 s, on whatever
    thread called it -- for an instrumented loop, the training thread. On a slow or networked
    script directory (the spool sits beside the script by default) that stalls training.
    Correct: state snapshots are written by the writer thread, not the caller."""
    monitor = Monitor(directory=str(tmp_path), session_id="io", interval=0.0, tensor_interval=0.0)
    real = monitor.writer.write_state

    def slow_disk(state):
        time.sleep(0.5)
        real(state)

    monitor.writer.write_state = slow_disk
    try:
        monitor._last_snapshot = -1e9          # a snapshot is due on this sample
        t0 = time.perf_counter()
        monitor.observe_locals({"loss": 0.5}, step=1)
        elapsed = time.perf_counter() - t0
    finally:
        monitor.writer.write_state = real
        monitor.close()
    assert elapsed < 0.1, f"observe_locals blocked the caller for {elapsed:.2f}s on disk IO"


def test_ok_crash_then_close_keeps_crashed_flag(tmp_path):
    monitor = Monitor(directory=str(tmp_path), session_id="c", interval=0.0)
    monitor.observe_locals({"loss": 1.0}, step=1)
    monitor.snapshot_state({"crashed": True})
    monitor.close()
    state = stream.StreamReader(str(tmp_path)).state()
    assert state.get("crashed") is True and state.get("finished") is True


def test_ok_skipped_samples_cost_almost_nothing(tmp_path):
    monitor = Monitor(directory=str(tmp_path), session_id="cost", interval=3600.0)
    local_vars = {f"v{i}": float(i) for i in range(50)}
    monitor.observe_locals(local_vars, step=0)
    t0 = time.perf_counter()
    for i in range(20000):
        monitor.observe_locals(local_vars, step=i)
    per_call = (time.perf_counter() - t0) / 20000
    monitor.close()
    assert per_call < 20e-6, f"{per_call * 1e6:.1f} us per skipped sample"


def test_ok_stream_mode_sampler_overhead_on_a_pure_python_loop(tmp_path):
    """Measure what the sampler thread costs a GIL-bound training loop (the split's point)."""
    def loop(n=5_000_000):
        total = 0.0
        loss = 1.0
        for step in range(n):
            loss = loss * 0.99999 + 1e-6
            total += loss
        return total

    loop(50000)
    t0 = time.perf_counter()
    loop()
    base = time.perf_counter() - t0
    monitor = pulse_monitor.attach(directory=str(tmp_path), session_id="ovh", interval=0.05, depth=3)
    try:
        t0 = time.perf_counter()
        loop()
        watched = time.perf_counter() - t0
    finally:
        pulse_monitor.detach()
    overhead = watched / base - 1.0
    print(f"\nsampler overhead at 20 Hz on a pure-Python loop: {overhead * 100:.1f}% "
          f"({base:.3f}s -> {watched:.3f}s)")
    assert overhead < 0.25


# =====================================================================================
# pulse_brain
# =====================================================================================

def test_bug_all_nan_tensor_probe_crashes_the_audit(tmp_path):
    """backend.statistics() returns min/max/mean = None when a tensor has no finite values
    (an all-NaN weight or gradient -- exactly the failure Pulse exists for). The monitor
    streams that as-is; render_evidence then formats it with '%.4g' % None -> TypeError.
    Brain.audit() raises before it reaches schedule.set, so the audit stays due and the
    console's pump prints 'monitor read failed: TypeError' every 0.5 s forever, and
    Console.ask() dies the same way.
    Correct: missing stats render as n/a and the audit runs."""
    monitor = Monitor(directory=str(tmp_path), session_id="nan", interval=0.0, tensor_interval=0.0)
    stream.StreamReader(str(tmp_path)).send_control(stream.CONTROL_SNAPSHOT, names=["weights"])
    weights = np.full((4, 4), np.nan, dtype=np.float32)
    monitor.observe_locals({"loss": 1.0, "weights": weights}, step=1)
    monitor.observe_locals({"loss": 0.9, "weights": weights}, step=2)
    monitor.close()
    prompts = []
    brain = Brain(str(tmp_path), agent=lambda p: prompts.append(p) or
                  '{"status": "problem", "next_check_minutes": 5}')
    brain.ingest(brain.reader.poll())
    assert brain.tensor_stats.get("weights"), "test setup: probe never arrived"
    record = brain.audit()
    assert record.get("status") == "problem"
    assert prompts and "weights" in prompts[0]


def test_bug_brain_main_throws_away_the_persisted_schedule(tmp_path, monkeypatch):
    """Schedule is persisted to schedule.json precisely so 'a brain that is restarted ...
    picks up the cadence the agent asked for instead of resetting to the default'. But
    pulse_brain.main() unconditionally calls brain.schedule.set(args.interval, 'startup')
    with --interval defaulting to 900 s, so every restart of `python -m pulse.brain`
    overwrites the agent's chosen cadence (e.g. 2 min right after a fix) with 15 min.
    Correct: without an explicit --interval, main keeps the loaded schedule."""
    d = tmp_path / "sess"
    writer = stream.StreamWriter(str(d), flush_seconds=0.01)
    writer.write_session({"session_id": "s", "pid": os.getpid(), "started": time.time()})
    writer.close()
    sched = pulse_brain.Schedule(str(d / "schedule.json"))
    sched.set(120, "code changed: fixed lr", "high")
    monkeypatch.setattr(sys, "stdout", open(os.devnull, "w"))
    pulse_brain.main([str(d), "--once"])
    reloaded = pulse_brain.Schedule(str(d / "schedule.json"))
    assert reloaded.interval == 120, f"restart reset the agent's 120 s cadence to {reloaded.interval}"


def test_bug_brain_run_never_ends_for_a_killed_training_process(tmp_path):
    """Brain.run() loops `while not self.finished`, and `finished` is only set by a
    'finished' event or a BYE frame. A training process killed by SIGKILL / the OOM killer /
    a cluster pre-emption writes neither, so `python -m pulse.brain` watches a dead run
    forever -- and keeps paying for an audit every interval.
    Correct: the brain notices the session's pid is gone (and the stream is quiet) and ends."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=0.01)
    writer.write_session({"session_id": "dead", "pid": proc.pid, "started": time.time() - 100})
    writer.emit(stream.KIND_SCALARS, {"step": 1, "values": {"loss": 1.0}})
    writer.flush(2.0)
    writer._closed = True                          # simulate SIGKILL: no BYE, no finished
    brain = Brain(str(tmp_path), poll_interval=0.05)
    stop = threading.Event()
    t = threading.Thread(target=brain.run, kwargs={"stop": stop.is_set}, daemon=True)
    t.start()
    t.join(5.0)
    ended = not t.is_alive()
    stop.set()
    t.join(2.0)
    assert ended, "Brain.run() still watching a process that no longer exists"


def test_ok_parse_decision_handles_nested_objects_and_prose():
    reply = ('Looks fine. {"note": {"x": 1}}\n'
             '{"status": "watch", "risk": "medium", "findings": [{"a": 1}], '
             '"next_check_minutes": 7, "reason": "lr"}')
    decision = pulse_brain.parse_decision(reply)
    assert decision["status"] == "watch" and decision["next_check_minutes"] == 7


def test_ok_audit_prompt_bounds_match_the_schedule_clamp():
    text = pulse_brain.AUDIT_PROMPT.format(evidence="E", min_minutes=1.0, max_minutes=60.0)
    assert "between 1 and 60" in text
    s = pulse_brain.Schedule()
    s.apply_decision({"next_check_minutes": 0.1})
    assert s.interval == pulse_brain.MIN_INTERVAL_SECONDS
    s.apply_decision({"next_check_minutes": 1e9})
    assert s.interval == pulse_brain.MAX_INTERVAL_SECONDS


def test_ok_brain_run_ends_on_a_clean_finish(tmp_path):
    monitor = Monitor(directory=str(tmp_path), session_id="fin", interval=0.0)
    for i in range(5):
        monitor.observe_locals({"loss": 1.0 / (i + 1)}, step=i)
    monitor.close()
    brain = Brain(str(tmp_path), poll_interval=0.01)
    t = threading.Thread(target=brain.run, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive() and brain.finished
    assert brain.histories["loss"][-1] == pytest.approx(0.2)


def test_ok_nan_history_renders_in_evidence(tmp_path):
    brain = Brain(str(tmp_path))
    brain.ingest([{"kind": "scalars", "step": i, "values": {"loss": v}}
                  for i, v in enumerate([1.0, 0.5, float("nan"), float("inf")] * 60)])
    text = Brain.render_evidence(brain.evidence())
    assert "loss" in text


def test_bug_crash_traceback_in_the_stream_never_reaches_the_agent(tmp_path):
    """The stream-mode excepthook (pulse.py _stream_excepthook) writes the exception and
    traceback into a 'crash' event 'so the evidence is there whenever [the brain] next
    looks', and nonfinite events carry the variable name. render_evidence -- the only thing
    the audit prompt and Console.ask see -- prints events as bare 'event@step', so the agent
    is told 'crash@812' with no exception, and 'nonfinite@40' without saying which value.
    Correct: the crash's exception text (and the nonfinite variable's name) are in the prompt."""
    brain = Brain(str(tmp_path))
    brain.ingest([
        {"kind": "scalars", "step": 40, "values": {"loss": 1.0}},
        {"kind": "event", "event": "nonfinite", "name": "grad_norm", "value": "nan",
         "step": 40, "urgent": True},
        {"kind": "event", "event": "crash", "urgent": True, "step": 41,
         "exception": "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
         "traceback": "Traceback ...\n  File \"train.py\", line 88, in <module>\n"},
    ])
    text = Brain.render_evidence(brain.evidence())
    assert "CUDA out of memory" in text, "crash exception not shown to the agent"
    assert "grad_norm" in text, "which value went non-finite is not shown to the agent"


def test_bug_a_value_with_one_reading_is_silently_left_out_of_the_evidence(tmp_path):
    """render_evidence promises 'compact, but nothing silently dropped', but evidence() skips
    every history with fewer than 2 points. A metric recorded once so far -- the first
    val_loss / val_accuracy after an epoch, a final test score -- is invisible to the audit
    and to Console.ask, even though /vars shows it.
    Correct: single-reading values are listed (with their one value)."""
    brain = Brain(str(tmp_path))
    brain.ingest([{"kind": "scalars", "step": s, "values": {"loss": 1.0 / s}} for s in range(1, 20)]
                 + [{"kind": "scalars", "step": 20, "values": {"val_accuracy": 0.12}}])
    text = Brain.render_evidence(brain.evidence())
    assert "val_accuracy" in text


# =====================================================================================
# ledger items confirmed by reading (r:1), covered after the fix
# =====================================================================================

def test_ok_pause_stops_sampling_until_resume(tmp_path):
    monitor = Monitor(directory=str(tmp_path), session_id="p", interval=0.0, tensor_interval=0.0)
    reader = stream.StreamReader(str(tmp_path))
    reader.send_control(stream.CONTROL_PAUSE, reason="test")
    monitor._last_control_poll = -1e9
    monitor.observe_locals({"loss": 1.0}, step=1)
    reader.send_control(stream.CONTROL_RESUME)
    monitor._last_control_poll = -1e9
    monitor.observe_locals({"loss": 0.5}, step=2)
    monitor.close()
    assert _steps(stream.StreamReader(str(tmp_path)).poll()) == [2]


def test_ok_stop_interrupts_the_attached_thread_not_the_main_one(tmp_path):
    outcome = {}
    ready = threading.Event()

    def training():
        ready.set()
        try:
            while True:
                time.sleep(0.01)
        except KeyboardInterrupt:
            outcome["interrupted"] = True

    t = threading.Thread(target=training, daemon=True)
    t.start()
    ready.wait(2)
    monitor = pulse_monitor.attach(directory=str(tmp_path), session_id="stop", interval=0.05,
                                   depth=1, thread_id=t.ident)
    try:
        stream.StreamReader(str(tmp_path)).send_control(stream.CONTROL_STOP, reason="test")
        t.join(5)
    finally:
        pulse_monitor.detach()
    assert outcome.get("interrupted"), "the watched thread was never interrupted"
    assert monitor.stop_requested


def test_ok_writer_with_zero_flush_seconds_does_not_spin(tmp_path):
    writer = stream.StreamWriter(str(tmp_path), flush_seconds=0.0)
    cpu0 = time.process_time()
    time.sleep(1.0)
    used = time.process_time() - cpu0
    writer.emit(stream.KIND_SCALARS, {"step": 1, "values": {"loss": 1.0}})
    writer.close()
    assert _steps(stream.StreamReader(str(tmp_path)).poll()) == [1]
    assert used < 0.5, f"idle writer used {used:.2f}s of CPU in 1s"


def test_ok_brain_started_before_session_json_learns_the_start_time(tmp_path):
    brain = Brain(str(tmp_path))                      # no session.json yet
    assert not brain.session.get("started")
    monitor = Monitor(directory=str(tmp_path), session_id="late", interval=0.0)
    monitor.close()
    brain.ingest(brain.reader.poll())
    assert brain.session.get("started") == monitor._started


def test_ok_problem_audit_is_escalated(tmp_path):
    escalated = []
    brain = Brain(str(tmp_path), agent=lambda p: '{"status": "problem", "findings": ["lr"], '
                                                 '"next_check_minutes": 5}',
                  escalate=lambda findings, pack: escalated.append(pack))
    brain.ingest([{"kind": "scalars", "step": 1, "values": {"loss": 1.0}}])
    brain.audit()
    assert escalated and escalated[0]["audit"]["status"] == "problem"
