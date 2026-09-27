"""Sweep 2 -- pulse_terminal.TerminalExecutor (rewritten runner) and render()."""
import os
import subprocess
import time
import tracemalloc
import uuid

import pytest

from pulse import pulse_terminal as T


def _pids_with(marker):
    out = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if int(p) != os.getpid()]


def _kill(pids):
    for p in pids:
        try:
            os.kill(p, 9)
        except OSError:
            pass


def test_bug_interrupt_while_waiting_orphans_the_command(tmp_path, monkeypatch):
    """The command now runs in its own session (start_new_session=True), so a Ctrl-C at
    Pulse's terminal no longer reaches it. run() only kills the group on TimeoutExpired;
    a KeyboardInterrupt (or any other exception) raised while communicate() waits
    propagates and leaves the whole group running with nobody to reap it -- e.g. a
    `python train.py` the agent started keeps the GPU forever. Correct: kill the process
    group in a finally/except BaseException around communicate(), then re-raise."""
    marker = "pulse_s2_" + uuid.uuid4().hex
    real_popen = subprocess.Popen

    class Interrupt(BaseException):
        """Stands in for KeyboardInterrupt (which pytest itself intercepts)."""

    class InterruptingPopen(real_popen):
        def communicate(self, *a, **k):
            time.sleep(0.3)
            raise Interrupt

    monkeypatch.setattr(T.subprocess, "Popen", InterruptingPopen)
    ex = T.TerminalExecutor(default_cwd=str(tmp_path))
    try:
        with pytest.raises(Interrupt):
            ex.run(T.TerminalRequest(f"python3 -c 'import time; time.sleep(30)' {marker}", timeout=60))
        monkeypatch.undo()                  # pgrep below uses the real Popen
        time.sleep(0.3)
        assert _pids_with(marker) == [], "the interrupted command is still running"
    finally:
        monkeypatch.undo()
        _kill(_pids_with(marker))


def test_bug_whole_output_is_read_into_memory_before_truncation(tmp_path):
    """Output goes to temp files, then `out_f.read().decode()` loads ALL of it (bytes, then
    a str copy) before _truncate_stream keeps 8,000 characters. `cat` of a multi-GB
    dataset/checkpoint or a chatty process running for the 600 s maximum fills /tmp and
    then Pulse's (= the training process's) memory. Correct: seek and read only the
    head (HEAD_CHARS) and tail (TAIL_CHARS) windows, plus the total size for the note."""
    ex = T.TerminalExecutor(default_cwd=str(tmp_path))
    tracemalloc.start()
    try:
        result = ex.run(T.TerminalRequest("head -c 40000000 /dev/zero | tr '\\0' a", timeout=60))
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.exit_code == 0 and result.stdout_truncated
    assert peak < 10_000_000, f"peak Python allocation {peak/1e6:.0f} MB for 8 KB of kept output"


def test_ok_timeout_kills_the_group_and_background_does_not_block(tmp_path):
    marker = "pulse_s2_" + uuid.uuid4().hex
    ex = T.TerminalExecutor(default_cwd=str(tmp_path))
    try:
        r = ex.run(T.TerminalRequest(f"python3 -c 'import time; time.sleep(30)' {marker}; echo x", timeout=1))
        assert r.timed_out and r.exit_code is None
        time.sleep(0.3)
        assert _pids_with(marker) == []
        t0 = time.monotonic()
        r2 = ex.run(T.TerminalRequest(f"python3 -c 'import time; time.sleep(5)' {marker} & echo started",
                                      timeout=4))
        assert not r2.timed_out and "started" in r2.stdout and time.monotonic() - t0 < 2.5
    finally:
        _kill(_pids_with(marker))


def test_ok_stdin_is_devnull_and_bash_is_used(tmp_path):
    ex = T.TerminalExecutor(default_cwd=str(tmp_path))
    r = ex.run(T.TerminalRequest("read x; echo \"rc=$? [$x]\"; echo ${BASH_VERSION:+bash}", timeout=5))
    assert r.stdout.split() == ["rc=1", "[]", "bash"]
    r = ex.run(T.TerminalRequest("printf 'caf\\351\\n'"))
    assert r.exit_code == 0 and "caf" in r.stdout


def test_ok_render_scrubs_keys_from_output_and_command(tmp_path, monkeypatch):
    key = "sk-or-v1-" + "b" * 40
    monkeypatch.setenv("SOME_API_KEY", "custom-secret-value-123")
    ex = T.TerminalExecutor(default_cwd=str(tmp_path))
    r = ex.run(T.TerminalRequest(f"echo {key}; echo $SOME_API_KEY"))
    text = r.render()
    assert key not in text and "custom-secret-value-123" not in text


def test_ok_parse_inline_timeout():
    assert T.parse_inline_timeout("echo a | tr a b --timeout=30") == ("echo a | tr a b", 30.0)
    assert T.parse_inline_timeout("echo 'x --timeout=5'") == ("echo 'x --timeout=5'", None)
    assert T.parse_inline_timeout("ls --timeout=inf") == ("ls", None)


def test_ok_capture_file_is_capped_and_keeps_head_and_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_CAPTURE_BYTES", 200_000)
    monkeypatch.setattr(T, "_CAPTURE_POLL_SECONDS", 0.02)
    ex = T.TerminalExecutor(default_cwd=str(tmp_path))
    r = ex.run(T.TerminalRequest(
        "echo FIRSTLINE; for i in $(seq 1 60); do head -c 100000 /dev/zero | tr '\\0' a; "
        "sleep 0.01; done; echo; echo LASTLINE", timeout=60))
    assert r.exit_code == 0 and r.stdout_truncated
    assert r.stdout.startswith("FIRSTLINE") and r.stdout.rstrip().endswith("LASTLINE")
    assert "\0" not in r.stdout
