"""
The wire between the two halves of Pulse.

Pulse used to be one process: the tracker, the detectors and the AI agent all ran on
the training thread, so a run stopped dead while the model thought. Measured on a 20k
step loop, `PulseCLI.update()` held the training thread for 84% of the wall clock, and
almost all of that was two blocking model calls.

So Pulse is split in two:

    training process                     brain process
    ---------------                      -------------
    pulse_monitor.Monitor      --->      pulse_brain.Brain
      reads values, writes frames        detection, the agent, fixes
      never blocks, never thinks         thinks as long as it likes

This module is the pipe between them, and nothing else. It is deliberately built on
append-only files rather than sockets or shared memory:

  * A file needs no connection, no handshake and no port. Colab, a container, an SSH
    session and a Windows box all behave the same.
  * The brain can attach late, crash, or be restarted, and still read everything from
    the beginning. A socket would have dropped whatever was sent while it was away.
  * The training process never blocks on a reader that is not there. Writing to a
    socket nobody is draining eventually blocks or raises; appending to a file does not.
  * After the run, the spool is a complete record of what happened, which is exactly
    what the brain's wake-up audit wants to re-read.

The stream is lossy on purpose. If the writer thread falls behind, frames are dropped
rather than queued without bound, and the drop is recorded so the brain knows its view
has a hole in it. Training speed always wins over completeness.

Layout of a session directory:

    <dir>/session.json     written once: script, pid, argv, start time, format version
    <dir>/events.jsonl     append-only frames, one JSON object per line
    <dir>/state.json       latest snapshot, atomically replaced, for a late attach
    <dir>/control.jsonl    the other direction: brain -> monitor requests
"""
from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

FORMAT_VERSION = 1

# Frame kinds. Keep these stable: the brain matches on them.
KIND_HELLO = "hello"        # session metadata, first frame
KIND_SCALARS = "scalars"    # {"step": int, "values": {name: float|None}}
KIND_TENSOR = "tensor"      # {"name": str, "shape": [...], "dtype": str, "device": str, "stats": {...}}
KIND_EVENT = "event"        # {"event": str, ...} - tripwires, crashes, lint findings, phase changes
KIND_DROP = "drop"          # {"dropped": int} - frames lost to backpressure
KIND_BYE = "bye"            # clean shutdown

# Control messages (brain -> monitor).
CONTROL_PAUSE = "pause"
CONTROL_RESUME = "resume"
CONTROL_SNAPSHOT = "snapshot"       # asks for a full state dump now
CONTROL_SET_INTERVAL = "set_interval"
CONTROL_TRACK = "track"
CONTROL_UNTRACK = "untrack"
CONTROL_STOP = "stop"               # stop training (the monitor owns the process, so it acts)

_DEFAULT_QUEUE_FRAMES = 4096
_DEFAULT_FLUSH_SECONDS = 0.25
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def registry_dir() -> str:
    """Where every run on this machine announces itself.

    The spool lives beside the training script, which is right for keeping a run's
    record with the code it describes, and useless for finding it: `pulse` typed in
    some other directory has no way to guess where that script was. So a monitor also
    drops a pointer here, and the console can list every run on the machine without
    being told where to look.
    """
    base = os.environ.get("PULSE_HOME", "").strip()
    if not base:
        home = os.path.expanduser("~")
        if home == "~" or not os.path.isabs(home):
            # No HOME and no passwd entry: the usual shape of a container run started
            # with --user 1234:1234. expanduser hands back the literal "~", and joining
            # it produced a RELATIVE path, so Pulse created a directory actually named
            # "~" inside the training job's working directory -- often a mounted volume.
            home = tempfile.gettempdir()
        base = os.path.join(home, ".pulse")
    return os.path.join(base, "sessions")


def register_session(session_id: str, directory: str, info: Dict[str, Any]) -> Optional[str]:
    """Announce a run. Best effort: a read-only or missing home must not stop training."""
    try:
        os.makedirs(registry_dir(), exist_ok=True)
        path = os.path.join(registry_dir(), f"{session_id}.json")
        _atomic_write_json(path, dict(info, session_id=session_id,
                                      directory=os.path.abspath(directory)))
        return path
    except OSError:
        return None


def unregister_session(session_id: str) -> None:
    try:
        os.unlink(os.path.join(registry_dir(), f"{session_id}.json"))
    except OSError:
        pass


def registered_sessions() -> List[Dict[str, Any]]:
    """Every run this machine knows about, newest first, with dead pointers dropped."""
    directory = registry_dir()
    try:
        names = os.listdir(directory)
    except OSError:
        return []           # no registry yet, unreadable, or removed while we looked
    out = []
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
                entry = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(entry, dict) or not entry.get("directory"):
            continue
        if not os.path.isdir(entry["directory"]):
            continue        # the spool was deleted; the pointer is stale
        out.append(entry)
    return sorted(out, key=lambda e: e.get("started") or 0, reverse=True)


def session_dir_for(script_path: Optional[str], session_id: str) -> str:
    """Where a run's spool lives.

    Beside the script by default, so a brain started in the same directory finds it
    without being told, and so the spool travels with the code it describes.
    PULSE_STREAM_DIR overrides that for read-only or networked script directories.
    """
    override = os.environ.get("PULSE_STREAM_DIR", "").strip()
    if override:
        return os.path.join(os.path.abspath(override), session_id)
    if script_path:
        base = os.path.dirname(os.path.abspath(script_path))
    else:
        base = os.getcwd()
    return os.path.join(base, ".pulse_stream", session_id)


def _atomic_write_json(path: str, payload: Any) -> None:
    """Replace path with payload. A reader either sees the old file or the new one."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class StreamWriter:
    """The training-process end. Every public method is safe to call from the hot path.

    `emit` does three things: build a small tuple, put it on a bounded queue, return.
    No JSON encoding, no file IO, no locks held across IO. A daemon thread does the
    encoding and the writing. When the queue is full the oldest frame is dropped, so a
    slow disk slows Pulse down, never the training loop.
    """

    def __init__(
        self,
        directory: str,
        *,
        max_frames: int = _DEFAULT_QUEUE_FRAMES,
        flush_seconds: float = _DEFAULT_FLUSH_SECONDS,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        self.directory = os.path.abspath(directory)
        self.events_path = os.path.join(self.directory, "events.jsonl")
        self.state_path = os.path.join(self.directory, "state.json")
        self.session_path = os.path.join(self.directory, "session.json")
        self.control_path = os.path.join(self.directory, "control.jsonl")
        self.max_bytes = max_bytes
        self.flush_seconds = flush_seconds
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_frames)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._dropped = 0
        self._on_error = on_error
        self._closed = False
        self._control_offset = 0

        os.makedirs(self.directory, exist_ok=True)
        self._handle = open(self.events_path, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="pulse-stream-writer", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- producing

    def emit(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        """Queue one frame. Returns False if it was dropped. Never raises, never blocks."""
        if self._closed:
            return False
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        item = (seq, time.time(), kind, payload or {})
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            # Drop the oldest rather than the newest: the brain cares far more about
            # what is happening now than about a frame from a second ago.
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self._dropped += 1
                return False

    def write_session(self, info: Dict[str, Any]) -> None:
        _atomic_write_json(self.session_path, dict(info, format_version=FORMAT_VERSION))

    def write_state(self, state: Dict[str, Any]) -> None:
        """Replace the snapshot a late-attaching brain reads before tailing the log."""
        try:
            _atomic_write_json(self.state_path, state)
        except OSError as exc:
            self._report(exc)

    # ---------------------------------------------------------------- control channel

    def poll_control(self) -> List[Dict[str, Any]]:
        """Brain -> monitor messages that arrived since the last call.

        Reads only whole lines and remembers the byte offset, so a message being
        written while we read is picked up next time instead of being seen half-built.
        """
        try:
            size = os.path.getsize(self.control_path)
        except OSError:
            return []
        if size <= self._control_offset:
            if size < self._control_offset:     # truncated: start over
                self._control_offset = 0
            else:
                return []
        messages: List[Dict[str, Any]] = []
        try:
            with open(self.control_path, "r", encoding="utf-8") as handle:
                handle.seek(self._control_offset)
                for line in handle:
                    if not line.endswith("\n"):
                        break                   # partial write: leave the offset before it
                    self._control_offset += len(line.encode("utf-8"))
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        messages.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError as exc:
            self._report(exc)
        return messages

    # ---------------------------------------------------------------- writer thread

    def _run(self) -> None:
        pending: List[str] = []
        last_flush = time.monotonic()
        while True:
            timeout = max(0.0, self.flush_seconds - (time.monotonic() - last_flush))
            try:
                item = self._queue.get(timeout=timeout or self.flush_seconds)
            except queue.Empty:
                item = None
            else:
                if item is None:                # shutdown sentinel
                    self._flush(pending)
                    try:
                        self._handle.close()
                    except OSError:
                        pass
                    return
                pending.append(self._encode(item))
            if pending and (time.monotonic() - last_flush >= self.flush_seconds or len(pending) >= 256):
                self._flush(pending)
                last_flush = time.monotonic()

    def _encode(self, item: tuple) -> str:
        seq, timestamp, kind, payload = item
        frame = {"seq": seq, "t": round(timestamp, 4), "kind": kind}
        frame.update(payload)
        try:
            return json.dumps(frame, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # A payload we cannot encode must not kill the writer thread.
            return json.dumps({"seq": seq, "t": round(timestamp, 4), "kind": KIND_EVENT,
                               "event": "encode_failed", "original_kind": kind})

    def _flush(self, pending: List[str]) -> None:
        if not pending:
            return
        dropped, self._dropped = self._dropped, 0
        if dropped:
            pending.append(json.dumps({"seq": -1, "t": round(time.time(), 4),
                                       "kind": KIND_DROP, "dropped": dropped}))
        try:
            self._handle.write("\n".join(pending) + "\n")
            self._handle.flush()
        except OSError as exc:
            self._report(exc)
        finally:
            pending.clear()
        self._maybe_rotate()

    def _maybe_rotate(self) -> None:
        """Keep the spool bounded. The reader notices the truncation and re-attaches."""
        try:
            if self._handle.tell() < self.max_bytes:
                return
            self._handle.close()
            os.replace(self.events_path, self.events_path + ".prev")
            self._handle = open(self.events_path, "a", encoding="utf-8")
        except OSError as exc:
            self._report(exc)

    def _report(self, exc: BaseException) -> None:
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:
                pass

    # ---------------------------------------------------------------- shutdown

    def close(self, timeout: float = 2.0) -> None:
        """Flush what is queued and stop. Bounded: shutdown must not hang a training run."""
        if self._closed:
            return
        self.emit(KIND_BYE, {})
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        self._thread.join(timeout=timeout)


class StreamReader:
    """The brain end. Tails events.jsonl and notices when it has missed something.

    Holds no lock on the writer and never blocks it. `poll()` returns whatever whole
    lines have appeared since the last call; a partially written final line is left
    for next time.
    """

    def __init__(self, directory: str) -> None:
        self.directory = os.path.abspath(directory)
        self.events_path = os.path.join(self.directory, "events.jsonl")
        self.state_path = os.path.join(self.directory, "state.json")
        self.session_path = os.path.join(self.directory, "session.json")
        self.control_path = os.path.join(self.directory, "control.jsonl")
        self._offset = 0
        self._last_seq = 0
        self.gaps = 0               # frames the writer told us it dropped
        self.rotations = 0
        self._lock = threading.Lock()

    def session(self) -> Dict[str, Any]:
        return self._read_json(self.session_path)

    def state(self) -> Dict[str, Any]:
        return self._read_json(self.state_path)

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            return {}

    def poll(self) -> List[Dict[str, Any]]:
        """Every complete frame written since the last poll.

        Serialised: the console reads on a background thread while the main thread can
        ask for the same reader, and two threads sharing one offset each read the whole
        file and then each advance it, so every frame lands in the history twice and the
        doubled offset then looks like a rotation.
        """
        with self._lock:
            return self._poll()

    def _poll(self) -> List[Dict[str, Any]]:
        try:
            size = os.path.getsize(self.events_path)
        except OSError:
            return []
        if size < self._offset:
            # The writer rotated (or the file was replaced): re-read from the top.
            self._offset = 0
            self.rotations += 1
        if size == self._offset:
            return []
        frames: List[Dict[str, Any]] = []
        try:
            with open(self.events_path, "r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                for line in handle:
                    if not line.endswith("\n"):
                        break
                    self._offset += len(line.encode("utf-8"))
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(frame, dict):
                        continue
                    if frame.get("kind") == KIND_DROP:
                        self.gaps += int(frame.get("dropped") or 0)
                    else:
                        seq = frame.get("seq")
                        if isinstance(seq, int) and seq > 0:
                            if self._last_seq and seq > self._last_seq + 1:
                                self.gaps += seq - self._last_seq - 1
                            self._last_seq = max(self._last_seq, seq)
                    frames.append(frame)
        except OSError:
            return frames
        return frames

    def follow(self, interval: float = 0.25, stop: Optional[Callable[[], bool]] = None) -> Iterator[Dict[str, Any]]:
        """Block yielding frames as they arrive, until `stop()` says otherwise."""
        while True:
            if stop is not None and stop():
                return
            frames = self.poll()
            if frames:
                for frame in frames:
                    yield frame
            else:
                time.sleep(interval)

    def send_control(self, action: str, **fields: Any) -> None:
        """Ask the monitor for something. Appends a line; the monitor polls for it."""
        message = dict(fields, action=action, t=time.time())
        os.makedirs(self.directory, exist_ok=True)
        with open(self.control_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
            handle.flush()


def list_sessions(root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every session spool under root (default: ./.pulse_stream), newest first."""
    base = os.path.abspath(root or os.environ.get("PULSE_STREAM_DIR") or
                           os.path.join(os.getcwd(), ".pulse_stream"))
    if not os.path.isdir(base):
        return []
    sessions = []
    for name in os.listdir(base):
        directory = os.path.join(base, name)
        if not os.path.isdir(directory):
            continue
        info = StreamReader(directory).session()
        info.setdefault("session_id", name)
        info["directory"] = directory
        try:
            info["mtime"] = os.path.getmtime(os.path.join(directory, "events.jsonl"))
        except OSError:
            info["mtime"] = 0.0
        sessions.append(info)
    return sorted(sessions, key=lambda item: item.get("mtime", 0.0), reverse=True)
