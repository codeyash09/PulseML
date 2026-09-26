"""
Watching a run that was not started under Pulse.

Normally Pulse is inside the training process: `pulse run` puts it there, or the script
calls auto_track() itself. Neither helps for the run that is already going -- the one you
started an hour ago with `python train.py` and now want to look at.

A process cannot be made to report on itself after the fact. The only way in is from
outside: read the other process's memory and pick the training loop's variables out of
its stack. That is what py-spy does, and it is what this module drives.

It costs something, and the cost should be stated plainly rather than discovered:

  * **It needs root.** Reading another process's memory is ptrace, and Linux allows that
    only for a parent process or root unless `/proc/sys/kernel/yama/ptrace_scope` is 0.
    So every sample runs under sudo.
  * **It needs py-spy**, which is a separate program: `pip install py-spy`.
  * **It only sees variables inside functions.** py-spy reads frame locals, and a loop
    written at the top level of a script keeps its variables in the module globals, which
    py-spy does not read. A loop inside `def train():` reports everything; the same loop
    at module level reports nothing, and this module says so rather than showing an empty
    run.
  * **It samples**, a few times a second at most, because each sample is a separate
    process. A spike between two samples is not seen. A run started under Pulse is read
    from the inside and does not have that problem.

What it produces is an ordinary Pulse stream, so the console, the detectors and the brain
treat an attached run exactly like any other.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import pulse_stream as stream
from .pulse_monitor import is_library_path

DEFAULT_INTERVAL = 1.0
MIN_INTERVAL = 0.2
MAX_INTERVAL = 60.0
_SAMPLE_TIMEOUT = 20.0

# Pulse's own package. Not "any directory called pulse": that skipped ~/pulse/train.py,
# and attach reported a loop plainly inside a function as unreadable module globals. The
# other process may run Pulse from a different place, so a file of Pulse's own name in a
# directory called pulse counts too.
_PULSE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    _PULSE_FILES = frozenset(n for n in os.listdir(_PULSE_DIR) if n.endswith(".py"))
except OSError:
    _PULSE_FILES = frozenset()


def pyspy_path() -> Optional[str]:
    """Where py-spy is, including the user site that pip installs into but PATH may miss."""
    found = shutil.which("py-spy")
    if found:
        return found
    candidate = os.path.join(os.path.expanduser("~"), ".local", "bin", "py-spy")
    return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None


def ptrace_scope() -> Optional[int]:
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope", "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def needs_root() -> bool:
    """True when reading another process needs privileges we do not already have."""
    if os.name != "posix":
        return True
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return False
    scope = ptrace_scope()
    return scope is None or scope != 0


def sudo_is_passwordless() -> bool:
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True,
                              timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _command(pid: int, pyspy: str, use_sudo: bool, non_interactive: bool) -> List[str]:
    command = [pyspy, "dump", "--pid", str(pid), "--locals", "--json"]
    if use_sudo:
        # -n only when a password would otherwise be asked for in a place nobody is
        # watching; the first, interactive attempt is allowed to prompt.
        return ["sudo"] + (["-n"] if non_interactive else []) + command
    return command


def read_frames(pid: int, pyspy: Optional[str] = None, use_sudo: Optional[bool] = None,
                non_interactive: bool = True) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """One py-spy dump of `pid`. Returns (threads, error); exactly one is meaningful."""
    pyspy = pyspy or pyspy_path()
    if not pyspy:
        return None, "py-spy is not installed (pip install py-spy)"
    if use_sudo is None:
        use_sudo = needs_root()
    try:
        result = subprocess.run(_command(pid, pyspy, use_sudo, non_interactive),
                                capture_output=True, text=True, timeout=_SAMPLE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "py-spy did not answer in time"
    except OSError as exc:
        return None, f"could not run py-spy: {exc}"
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip().splitlines()
        detail = message[-1] if message else f"exit status {result.returncode}"
        if "password" in detail.lower() or "sudo:" in detail.lower():
            detail = "sudo refused without a password"
        return None, detail
    try:
        loaded = json.loads(result.stdout)
    except ValueError:
        return None, "py-spy returned something that is not JSON"
    return (loaded if isinstance(loaded, list) else []), ""


def _is_pulse_file(filename: str) -> bool:
    directory, base = os.path.split(filename)
    if directory == _PULSE_DIR:
        return True
    return os.path.basename(directory) == "pulse" and base in _PULSE_FILES


def _is_user_frame(frame: Dict[str, Any]) -> bool:
    filename = frame.get("filename") or ""
    # Not this interpreter's paths alone: the target may run another Python, so the
    # judgement is made from the path's own components.
    return (bool(filename) and not is_library_path(filename)
            and not _is_pulse_file(filename))


def _as_number(text: Any) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def values_from(threads: List[Dict[str, Any]]) -> Dict[str, float]:
    """Numbers the training loop is holding, innermost frame winning a name clash."""
    values: Dict[str, float] = {}
    for thread in threads or []:
        # py-spy lists innermost first; walk outwards so the innermost wins.
        for frame in reversed([f for f in thread.get("frames") or [] if _is_user_frame(f)]):
            for local in frame.get("locals") or []:
                number = _as_number(local.get("repr"))
                if number is not None:
                    values[str(local.get("name"))] = number
    return values


def script_of(threads: List[Dict[str, Any]]) -> Optional[str]:
    """The outermost user file in the stack: the script being run."""
    for thread in threads or []:
        user = [f for f in thread.get("frames") or [] if _is_user_frame(f)]
        if user:
            return user[-1].get("filename")
    return None


def describe_readiness(pid: int) -> Dict[str, Any]:
    """Can this process be read, and if not, exactly what is missing."""
    report: Dict[str, Any] = {
        "pid": pid,
        "pyspy": pyspy_path(),
        "needs_root": needs_root(),
        "ptrace_scope": ptrace_scope(),
        "passwordless_sudo": None,
        "ok": False,
        "reason": "",
        "locals_visible": None,
    }
    if not report["pyspy"]:
        report["reason"] = "py-spy is not installed"
        return report
    if report["needs_root"]:
        report["passwordless_sudo"] = sudo_is_passwordless()
    threads, error = read_frames(pid, report["pyspy"], non_interactive=True)
    if threads is None:
        report["reason"] = error
        return report
    report["ok"] = True
    report["locals_visible"] = bool(values_from(threads))
    if not report["locals_visible"]:
        report["reason"] = ("py-spy can read the process, but the training loop's variables "
                            "are module globals, which it cannot see. A loop inside a "
                            "function reports its values.")
    return report


def spool_dir_for(script: Optional[str], session_id: str) -> str:
    """Where an attached run's spool goes.

    Beside the script normally, so it sits with the code it describes and the console
    finds it without being told.

    Not under sudo. There the script path came out of the target process's memory, and
    we are root: a process that says its file is `/etc/cron.d/x` would have root create
    directories and files there, and hand them to whoever asked. Anybody who can start a
    Python process can say that. So when root-because-of-sudo, the spool goes under the
    invoking user's own home -- a path the kernel told us, not one the target did.
    """
    home = stream.invoking_user_home()
    if home is not None:
        return os.path.join(home, ".pulse", "attached", session_id)
    base = os.path.dirname(os.path.abspath(script)) if script else os.getcwd()
    return os.path.join(base, ".pulse_stream", session_id)


class AttachedMonitor:
    """Samples another process and writes an ordinary Pulse stream for it."""

    def __init__(self, pid: int, *, script: Optional[str] = None,
                 interval: float = DEFAULT_INTERVAL, directory: Optional[str] = None,
                 session_id: Optional[str] = None) -> None:
        self.pid = int(pid)
        self.interval = max(MIN_INTERVAL, float(interval))
        self.pyspy = pyspy_path()
        self.use_sudo = needs_root()
        # Find out which script this is BEFORE choosing where the spool goes: named after
        # the fact, the stream lands in whatever directory `pulse` happened to be run
        # from and the console has nothing to call the run but "?".
        if script is None:
            threads, _ = read_frames(self.pid, self.pyspy, self.use_sudo)
            script = script_of(threads) if threads else None
        self.script = script
        self.session_id = session_id or (time.strftime("%Y%m%d-%H%M%S") + f"-pid{self.pid}")
        self.directory = directory or spool_dir_for(script, self.session_id)
        self.writer = stream.StreamWriter(self.directory)
        self.samples = 0
        self.errors = 0
        self.last_error = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last: Dict[str, float] = {}
        self._step = 0
        # Once `finished` is written it stays: a sample still in flight when stop() gave
        # up waiting for it rewrote the state without it, and the abandoned spool of a
        # still-running target then looked like a live run.
        self._finished = False
        self._state_lock = threading.Lock()

        # The target's identity (start ticks, pid namespace) and the SAMPLER's: the pid
        # says whether the target runs, but only this process writes the spool, and a
        # spool whose sampler died without cleaning up is nobody's live run.
        sampler = stream.process_start_ticks(os.getpid())
        info = {"session_id": self.session_id, "script": self.script, "pid": self.pid,
                "started": time.time(), "attached": True, "sampler": "py-spy",
                "sampler_pid": os.getpid(),
                **({"sampler_start_ticks": sampler} if sampler is not None else {}),
                **stream.process_identity(self.pid)}
        self._info = info
        self._registered = False
        self.writer.write_session(info)
        # With the start time: a brain that read the hello before session.json had
        # nothing else to go on, and showed '0.0s elapsed' for the life of the run.
        self.writer.emit(stream.KIND_HELLO, dict(info))

    # ------------------------------------------------------------------ sampling

    def sample_once(self) -> Dict[str, float]:
        threads, error = read_frames(self.pid, self.pyspy, self.use_sudo, non_interactive=True)
        if threads is None:
            self.errors += 1
            self.last_error = error
            self.writer.emit(stream.KIND_EVENT, {"event": "sample_failed", "error": error})
            return {}
        if self.script is None:
            self.script = script_of(threads)
        values = values_from(threads)
        if not values:
            return {}
        if not self._registered:
            # Announce it only once it has actually produced numbers. An attach that
            # reads nothing -- a loop at module level, a process that exits straight
            # away -- should not leave a permanent empty run in the machine's list.
            stream.register_session(self.session_id, self.directory, self._info)
            self._registered = True
        self.samples += 1
        # The first name that is present, not the first that is truthy: step 0.0 is a step.
        step = next((values[name] for name in ("step", "global_step", "i") if name in values), None)
        self._step = int(step) if isinstance(step, float) and step.is_integer() else self._step + 1
        self.writer.emit(stream.KIND_SCALARS, {"step": self._step, "values": values})
        for name, value in values.items():
            if not math.isfinite(value):
                self.writer.emit(stream.KIND_EVENT, {
                    "event": "nonfinite", "name": name, "urgent": True,
                    "value": "nan" if value != value else ("inf" if value > 0 else "-inf"),
                    "step": self._step})
        self._last = values
        return values

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._handle_control()
            except Exception as exc:                 # a bad message must not stop sampling
                self.last_error = f"{type(exc).__name__}: {exc}"
            if not self._alive():
                self.writer.emit(stream.KIND_EVENT, {"event": "finished", "reason": "process exited"})
                self.snapshot({"finished": True})
                return
            try:
                self.sample_once()
            except Exception as exc:                 # sampling must never take the console down
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            if self._stop.is_set():
                return                      # stop() has written the final state
            if self.samples and self.samples % 5 == 0:
                self.snapshot()

    def _handle_control(self) -> None:
        """The console's /interval and /stop. Never read before, so both were silently
        dropped while the console said it had asked the run to stop."""
        for message in self.writer.poll_control():
            action = message.get("action")
            if action == stream.CONTROL_SET_INTERVAL:
                try:
                    wanted = float(message.get("interval", self.interval))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(wanted):
                    self.interval = min(MAX_INTERVAL, max(MIN_INTERVAL, wanted))
            elif action == stream.CONTROL_STOP:
                self._request_stop(message.get("reason") or "")

    def _request_stop(self, reason: str) -> None:
        """What Ctrl-C in its terminal would do: SIGINT, so the run unwinds normally."""
        if os.name != "posix":
            # os.kill on Windows is TerminateProcess, not Ctrl-C: refuse rather than kill.
            self.writer.emit(stream.KIND_EVENT, {"event": "stop_unsupported", "reason": reason,
                                                 "error": "cannot interrupt another process here"})
            return
        if self.pid == os.getpid():
            self.writer.emit(stream.KIND_EVENT, {"event": "stop_failed", "reason": reason,
                                                 "error": "that is this process"})
            return
        try:
            os.kill(self.pid, signal.SIGINT)
        except OSError as exc:
            self.writer.emit(stream.KIND_EVENT, {"event": "stop_failed", "reason": reason,
                                                 "error": f"{type(exc).__name__}: {exc}"})
            return
        self.writer.emit(stream.KIND_EVENT, {"event": "stop_requested", "reason": reason})
        self.snapshot({"stop_requested": True})

    def _alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
            return True
        except PermissionError:
            return True             # EPERM: it exists, it belongs to someone else
        except (OSError, OverflowError):
            return False

    def snapshot(self, extra: Optional[Dict[str, Any]] = None) -> None:
        with self._state_lock:
            extra = dict(extra or {})
            if extra.get("finished"):
                self._finished = True
            elif self._finished:
                extra["finished"] = True
            self.writer.write_state({
                "session_id": self.session_id, "script": self.script, "step": self._step,
                "updated": time.time(), "scalars": dict(self._last), "attached": True,
                "samples": self.samples, "sample_errors": self.errors,
                "interval": self.interval,
                **extra})

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pulse-attach", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.snapshot({"finished": True})
        self.writer.close()
        stream.unregister_session(self.session_id)

    def discard(self) -> None:
        """Give up before anything was recorded, leaving nothing behind."""
        self._stop.set()
        self.writer.close()
        stream.unregister_session(self.session_id)
        try:
            for name in os.listdir(self.directory):
                os.unlink(os.path.join(self.directory, name))
            os.rmdir(self.directory)
        except OSError:
            pass
