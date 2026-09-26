"""
The half of Pulse that lives inside the training process.

Its entire job is to notice values and hand them to the brain. It does not decide
whether a run is healthy, it does not call a model, it does not print, it does not
draw, and it does not keep history. All of that is the brain's, in another process,
where taking a second to think costs the training run nothing.

What that buys, measured on a 20k-step loop: the old single-process Pulse held the
training thread for 84% of the wall clock, almost all of it inside blocking model
calls made from `update()`. Nothing here can block for longer than it takes to append
to a queue.

Three rules keep it that way:

1. **Never convert data to answer a question about its shape.** The old
   `is_trackable()` called `to_numpy()` -- a device-to-host copy and a CUDA sync -- on
   every local in scope just to decide whether the value was interesting. This module
   reads `.shape` and `.dtype` attributes and nothing else.

2. **Only small things cross the device boundary, and only on a schedule.** A 0-d loss
   tensor is worth a 4-byte sync a few times a second; a weight matrix is not worth one
   at all unless the brain specifically asks.

3. **Hold no references.** The old tracker merged every local into a dict that was
   never cleared, pinning tensors, models and optimizers -- and their GPU memory --
   for the life of the run. This one keeps names, shapes and floats.
"""
from __future__ import annotations

import _thread
import atexit
import math
import os
import sys
import threading
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import pulse_stream as stream

# Values that are worth a device sync to read, in elements.
_SMALL_ELEMENT_LIMIT = 4

# Bounds on the sampling interval the brain may ask for.
MIN_INTERVAL = 0.01
MAX_INTERVAL = 60.0

# Names that are never interesting, whatever they hold.
_BORING_NAMES = frozenset({
    "self", "cls", "_", "__", "__builtins__", "__name__", "__file__", "__doc__",
    "__package__", "__loader__", "__spec__", "__all__", "__version__", "__annotations__",
})

_BORING_PREFIXES = ("__pulse", "_pulse")


def _is_boring(name: str) -> bool:
    return name in _BORING_NAMES or name.startswith(_BORING_PREFIXES)


# Reading a value that lives on an accelerator is a device-to-host copy and a stream
# synchronization, however small the value. Sampling the training loop's locals does not
# do that unless the run asked for it (PULSE_GPU_READS=1): a 4 Hz sampler calling .item() on
# every scalar tensor in scope is a sync point the training run never agreed to.
_GPU_READS = os.environ.get("PULSE_GPU_READS", "").strip().lower() in ("1", "true", "yes", "on")
_ACCEL_TOKENS = ("cuda", "mps", "xpu", "rocm", "hip", "gpu", "tpu")
_MIRROR_SUFFIXES = ("_cpu", "_host", "_np", "_numpy", "_cpu_copy", "_host_copy")


def _on_accelerator(value: Any) -> bool:
    """True when `value` lives on a non-CPU device. Reads attributes only -- never data.
    (Same test as pulse_cli._pulse_is_accelerator_value, kept local so this module stays
    free of the CLI's dependencies.)"""
    try:
        device = getattr(value, "device", None)
        if device is None:
            return False
        kind = getattr(device, "type", None)
        if isinstance(kind, str) and kind.lower() not in ("cpu", ""):
            return True
        text = str(device).lower()
        if text.startswith("cpu"):
            return False
        if any(token in text for token in _ACCEL_TOKENS):
            return True
        if "/device:" in text and "/device:cpu:" not in text:       # TensorFlow placement
            return True
        device_id = getattr(device, "id", None)                      # CuPy
        return device_id is not None and int(device_id) >= 0
    except Exception:
        return False


def _cpu_mirror(name: Optional[str], local_vars: Optional[Dict[str, Any]]) -> Any:
    """An explicitly maintained host copy of `name` (loss_cpu, loss_np, ...), if the training
    code keeps one. Pulse reads that instead of the device value."""
    if not name or not local_vars:
        return None
    base = name.rsplit(".", 1)[-1]
    for candidate in dict.fromkeys(b + suffix for b in (name, base) for suffix in _MIRROR_SUFFIXES):
        mirror = local_vars.get(candidate)
        if mirror is not None and not _on_accelerator(mirror):
            return mirror
    return None


def _scalar_of(value: Any, local_vars: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> Optional[float]:
    """The float in `value`, if reading it is cheap. None otherwise.

    Cheap means: a Python number, or an array-like holding a handful of elements, on the
    host. A big tensor returns None even though a float could be computed from it -- that
    computation belongs in the brain, off the training thread. So does anything on an
    accelerator, unless the run opted in: pass `local_vars` (the implicit sampling path) and
    such a value is read only through a CPU mirror the training code maintains, or with
    PULSE_GPU_READS=1. A direct call without `local_vars` (Monitor.observe) is the user
    asking for that value explicitly and is left alone.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            result = float(value)
        except OverflowError:
            return None         # an int beyond float range (a seed, a hash): not a metric
        return result if not isinstance(value, complex) else None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        size = 1
        for dim in shape:
            size *= int(dim)
    except (TypeError, ValueError):
        return None
    if size > _SMALL_ELEMENT_LIMIT:
        return None
    item = getattr(value, "item", None)
    if item is None:
        # TensorFlow eager tensors and Keras variables have no .item(); their .numpy()
        # is the same host read. Without this a TF custom loop's loss was never sampled.
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            item = lambda: to_numpy().item()      # noqa: E731
    if item is None or size != 1:
        return None
    if local_vars is not None and not _GPU_READS and _on_accelerator(value):
        mirror = _cpu_mirror(name, local_vars)
        return _scalar_of(mirror) if mirror is not None else None
    try:
        return float(item())
    except (TypeError, ValueError, RuntimeError, AttributeError, OverflowError):
        return None


# Loop counters, in the order they are trusted. Using the training loop's own counter
# rather than "how many times the sampler has run" is what lets the brain line a finding
# up with a step number the user recognises.
_STEP_NAMES = ("global_step", "step", "iteration", "iter", "batch_idx", "epoch", "i")


def _infer_step(local_vars: Dict[str, Any], fallback: int) -> int:
    for name in _STEP_NAMES:
        value = local_vars.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return fallback


def _describe(value: Any) -> Optional[Dict[str, Any]]:
    """Shape/dtype/device metadata, read from attributes only. No data is touched."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    meta: Dict[str, Any] = {"shape": dims}
    if dtype is not None:
        meta["dtype"] = str(dtype)
    if device is not None:
        meta["device"] = str(device)
    elements = 1
    for dim in dims:
        elements *= dim
    meta["elements"] = elements
    return meta


def _names(message: Dict[str, Any]) -> List[str]:
    """The `names` of a control message as a list. A bare string is one name, not its
    letters; anything else that is not a list is no names at all."""
    names = message.get("names")
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


class Monitor:
    """Collects values in the training process and streams them out.

    `observe_locals` is the hot entry point. It is written to do as little as possible
    when it is not this sample's turn: one subtraction and a comparison.
    """

    def __init__(
        self,
        *,
        script_path: Optional[str] = None,
        session_id: Optional[str] = None,
        directory: Optional[str] = None,
        interval: float = 0.25,
        tensor_interval: float = 5.0,
        on_pause: Optional[Any] = None,
    ) -> None:
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.script_path = script_path
        self.directory = directory or stream.session_dir_for(script_path, self.session_id)
        self.interval = float(interval)
        # Set by attach(): the sampler thread's own pace. When there is one, /interval
        # changes that -- `interval` above is 0 there, so changing it did nothing.
        self.sampler_interval: Optional[float] = None
        # The thread a stop request should interrupt; None means the main thread.
        self.training_thread: Optional[int] = None
        self.tensor_interval = float(tensor_interval)
        self.writer = stream.StreamWriter(self.directory)
        self.step = 0

        self._last_sample = 0.0
        self._last_tensor_sample = 0.0
        self._last_control_poll = 0.0
        self._last_snapshot = 0.0
        self.snapshot_interval = 5.0
        self._sticky: Dict[str, Any] = {}
        self._last_values: Dict[str, float] = {}
        self._known_tensors: Dict[str, Dict[str, Any]] = {}
        self._requested: set = set()        # names the brain asked to see in full
        self._pending_probe: List[str] = []
        self._paused = False
        self._on_pause = on_pause
        self._stop_requested = False
        self._samples = 0
        self._observe_seconds = 0.0
        self._started = time.time()
        # The process's start ticks and pid namespace: what tells a reader this process
        # apart from a later one with the same pid, without trusting the wall clock.
        identity = stream.process_identity()

        self.writer.write_session({
            "session_id": self.session_id,
            "script": os.path.abspath(script_path) if script_path else None,
            "pid": os.getpid(),
            "started": self._started,
            "interval": self.interval,
            **identity,
        })
        self.writer.emit(stream.KIND_HELLO, {
            "session_id": self.session_id,
            "script": os.path.abspath(script_path) if script_path else None,
            "pid": os.getpid(),
            # A brain that started before session.json existed learns the start time
            # only from here; without it the run read "0.0s elapsed" forever.
            "started": self._started,
            **identity,
        })
        # Tell the machine this run exists, so `pulse` typed anywhere can find it.
        stream.register_session(self.session_id, self.directory, {
            "script": os.path.abspath(script_path) if script_path else None,
            "pid": os.getpid(),
            "started": self._started,
            "cwd": os.getcwd(),
            **identity,
        })

    # ------------------------------------------------------------------ hot path

    def observe_locals(self, local_vars: Dict[str, Any], step: Optional[int] = None) -> None:
        """Sample the caller's variables, if this sample is due.

        Everything before the `return` is what the training loop pays on the samples
        it skips, which is the overwhelming majority of them.
        """
        now = time.monotonic()

        # Before the interval gate, always: a long interval must never make the monitor
        # unreachable. This used to sit after the early return, so `/interval 3600`
        # locked the brain out for an hour with no way to take it back.
        if now - self._last_control_poll >= 1.0:
            self._last_control_poll = now
            self._handle_control()

        if self._paused or now - self._last_sample < self.interval:
            return
        self._last_sample = now
        started = time.perf_counter()
        if step is not None:
            self.step = int(step)
        else:
            self.step = _infer_step(local_vars, self.step + 1)

        scalars: Dict[str, Any] = {}
        tripwires: List[Tuple[str, float]] = []
        tensor_due = (now - self._last_tensor_sample) >= self.tensor_interval

        for name, value in list(local_vars.items()):
            if _is_boring(name):
                continue
            number = _scalar_of(value, local_vars, name)
            if number is not None:
                # Every sample is sent, including one identical to the last. Suppressing
                # repeats looks like free bandwidth and is actually how a frozen run
                # becomes invisible: "this value has not moved in six readings" is the
                # signal, and it cannot be reconstructed from a stream that only carries
                # changes. A float a few times a second is not worth the blind spot.
                scalars[name] = number
                self._last_values[name] = number
                if not math.isfinite(number):
                    tripwires.append((name, number))
                continue
            if tensor_due or name in self._requested:
                meta = _describe(value)
                if meta is not None:
                    known = self._known_tensors.get(name)
                    if known != meta:
                        self._known_tensors[name] = meta
                        self.writer.emit(stream.KIND_TENSOR, dict(meta, name=name, step=self.step))

        if tensor_due:
            self._last_tensor_sample = now
        if scalars:
            self.writer.emit(stream.KIND_SCALARS, {"step": self.step, "values": scalars})
        for name, number in tripwires:
            # A non-finite loss is the one thing worth interrupting for: the brain may
            # be asleep, and every further step is wasted compute.
            self.writer.emit(stream.KIND_EVENT, {
                "event": "nonfinite", "name": name,
                "value": "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf"),
                "step": self.step, "urgent": True,
            })

        if self._pending_probe:
            self._answer_probes(local_vars)

        # After sampling, so the snapshot describes this sample rather than the state
        # before it: taken first, the very first snapshot said step 0 with no values,
        # and a live run read as "no steps" until the next one was due.
        # Written by the writer thread: this is the training thread, and a snapshot is a
        # temp file, a write and a rename, on a disk that may be slow or networked.
        if now - self._last_snapshot >= self.snapshot_interval:
            self._last_snapshot = now
            self.snapshot_state(background=True)

        self._samples += 1
        self._observe_seconds += time.perf_counter() - started

    def observe(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Record one named value explicitly, bypassing the sampling schedule.

        This is what an instrumented training loop should call: it costs a float
        conversion and a queue append, and unlike the locals scan it cannot miss a
        value that only existed between two samples.
        """
        if step is not None:
            self.step = int(step)
        number = _scalar_of(value)
        if number is None:
            meta = _describe(value)
            if meta is not None:
                self.writer.emit(stream.KIND_TENSOR, dict(meta, name=name, step=self.step))
            return
        self._last_values[name] = number
        self.writer.emit(stream.KIND_SCALARS, {"step": self.step, "values": {name: number}})
        if not math.isfinite(number):
            self.writer.emit(stream.KIND_EVENT, {
                "event": "nonfinite", "name": name,
                "value": "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf"),
                "step": self.step, "urgent": True,
            })

    def event(self, name: str, **fields: Any) -> None:
        """Report something that is not a number: a crash, a phase change, a lint finding."""
        self.writer.emit(stream.KIND_EVENT, dict(fields, event=name, step=self.step))

    # ------------------------------------------------------------------ brain requests

    def _answer_probes(self, local_vars: Dict[str, Any]) -> None:
        """Compute real statistics for the variables the brain asked about.

        This is the only place the monitor touches bulk data, it happens because the
        brain explicitly asked, and the reference is dropped before we return.
        """
        names, self._pending_probe = self._pending_probe, []
        for name in names:
            value = local_vars.get(name)
            if value is None:
                self.writer.emit(stream.KIND_EVENT, {"event": "probe_missing", "name": name})
                continue
            try:
                from . import pulse_backend as backend
                stats = backend.statistics(value)
            except Exception as exc:                      # a probe must never kill training
                self.writer.emit(stream.KIND_EVENT, {
                    "event": "probe_failed", "name": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            self.writer.emit(stream.KIND_TENSOR, {
                "name": name, "step": self.step, "stats": {
                    key: (float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else val)
                    for key, val in dict(stats).items()},
                **(_describe(value) or {}),
            })

    def _handle_control(self) -> None:
        # One message at a time, each on its own. The batch was read in one go (and the
        # file offset moved past all of it), so one malformed message -- `names` that is
        # a number -- raising out of this loop threw away every message after it,
        # including a /stop.
        for message in self.writer.poll_control():
            try:
                self._handle_message(message)
            except Exception as exc:
                try:
                    self.event("control_failed", action=str(message.get("action")),
                               error=f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass

    def _handle_message(self, message: Dict[str, Any]) -> None:
        action = message.get("action")
        if action == stream.CONTROL_SNAPSHOT:
            self._pending_probe.extend(str(n) for n in _names(message))
        elif action == stream.CONTROL_TRACK:
            for name in _names(message):
                self._requested.add(str(name))
        elif action == stream.CONTROL_UNTRACK:
            for name in _names(message):
                self._requested.discard(str(name))
        elif action == stream.CONTROL_SET_INTERVAL:
            try:
                wanted = float(message.get("interval", self.interval))
            except (TypeError, ValueError):
                return
            if not math.isfinite(wanted):
                return
            # Bounded on both sides. Unbounded, one `/interval 1e9` (or inf) put the
            # monitor beyond reach of the next control message for the rest of the run.
            wanted = min(MAX_INTERVAL, max(MIN_INTERVAL, wanted))
            if self.sampler_interval is not None:
                self.sampler_interval = wanted
            else:
                self.interval = wanted
        elif action == stream.CONTROL_PAUSE:
            self._paused = True
            self.event("paused", reason=message.get("reason") or "")
            if callable(self._on_pause):
                self._on_pause(message)
        elif action == stream.CONTROL_RESUME:
            self._paused = False
            self.event("resumed")
        elif action == stream.CONTROL_STOP:
            self._stop_requested = True
            self.event("stop_requested", reason=message.get("reason") or "")
            self.snapshot_state({"stop_requested": True})
            # Actually stop it. This used to set a flag that only the sampler thread
            # read, and reading it made the sampler exit -- so "stop" left training
            # running at full speed and destroyed Pulse's ability to watch it, while
            # telling the user it had worked. interrupt_main raises KeyboardInterrupt
            # in the training thread, which is what Ctrl-C does: the loop unwinds,
            # finally blocks run, and the excepthook still records the ending.
            try:
                self._interrupt_training()
            except Exception as exc:
                self.event("stop_failed", error=f"{type(exc).__name__}: {exc}")

    def _interrupt_training(self) -> None:
        """KeyboardInterrupt in the training thread -- which is not always the main one.

        attach(thread_id=...) watches some other thread; interrupt_main() then stopped the
        main thread and left the training loop running.
        """
        target = self.training_thread
        main = threading.main_thread().ident
        if target is None or target == main:
            _thread.interrupt_main()
            return
        import ctypes

        hit = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(target), ctypes.py_object(KeyboardInterrupt))
        if hit > 1:                                   # never happens; undo it if it does
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(target), None)
        if hit != 1:
            raise RuntimeError(f"training thread {target} is gone")

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------ bookkeeping

    def snapshot_state(self, extra: Optional[Dict[str, Any]] = None, background: bool = False) -> None:
        """Replace state.json so a brain attaching late starts from something real.

        Flags like `crashed` stick: the crash hook records one and then closes the
        monitor, and close() writes another snapshot. Rebuilding the dict from scratch
        each time meant `finished` landed on top of `crashed` two lines later, and every
        run that died of a CUDA OOM was listed as having finished normally.
        """
        self._sticky.update(extra or {})
        write = self.writer.queue_state if background else self.writer.write_state
        write({
            "session_id": self.session_id,
            "script": os.path.abspath(self.script_path) if self.script_path else None,
            "step": self.step,
            "updated": time.time(),
            "scalars": dict(self._last_values),
            "tensors": dict(self._known_tensors),
            "cost": self.cost(),
            # The pace the run is sampled at now: the console's "live" window has to be
            # longer than the gap between two samples, or `/interval 45` flips a healthy
            # run between live and stalled.
            "interval": self.sampler_interval if self.sampler_interval is not None else self.interval,
            **self._sticky,
        })

    def cost(self) -> Dict[str, Any]:
        """What the monitor has spent on the training thread. Pulse should own up to this."""
        elapsed = max(1e-9, time.time() - self._started)
        return {
            "samples": self._samples,
            "seconds_on_training_thread": round(self._observe_seconds, 4),
            "percent_of_run": round(100.0 * self._observe_seconds / elapsed, 3),
            "per_sample_ms": round(1000.0 * self._observe_seconds / max(1, self._samples), 4),
        }

    def close(self) -> None:
        self.snapshot_state({"finished": True})
        self.event("finished", cost=self.cost())
        self.writer.close()
        # The spool stays for later reading; the machine-wide pointer does not, or every
        # run ever started is opened and parsed by every `pulse` from then on.
        stream.unregister_session(self.session_id)


# ---------------------------------------------------------------------------------------
# Attaching to a running script
# ---------------------------------------------------------------------------------------

_ACTIVE: Optional["Monitor"] = None
_SAMPLER: Optional[threading.Thread] = None
# The stop signal of the CURRENT sampler. Each attach() makes a new one: a single shared
# event that attach() cleared again revived an old sampler that had not yet noticed the
# detach (it was inside a slow sample when join() gave up), and two samplers then ran.
_STOP = threading.Event()

# Pseudo-files that are never the user's code.
_SKIP_PATH_MARKERS = ("<frozen", "<string>")
_LIBRARY_DIRS = frozenset({"site-packages", "dist-packages"})
_PYTHON_LIB_DIR = re.compile(r"python\d+(\.\d+)*[a-z]?$")


def _own_stdlib_dirs() -> Tuple[str, ...]:
    dirs = set()
    try:
        import sysconfig

        paths = sysconfig.get_paths()
        for key in ("stdlib", "platstdlib"):
            if paths.get(key):
                dirs.add(os.path.normcase(os.path.abspath(paths[key])).rstrip("\\/") + os.sep)
    except Exception:
        pass
    return tuple(sorted(dirs))


_STDLIB_DIRS = _own_stdlib_dirs()


def is_library_path(filename: str) -> bool:
    """A file of an installed package or of the standard library, not the user's own.

    Replaces a substring test for "/lib/python", which was too broad on Linux (a user's
    ~/lib/python-experiments/train.py was skipped as library code) and never matched on
    Windows, where the stdlib is <prefix>\\Lib\\ -- so the sampler read locals out of
    threading.py and friends as if they were the training loop's. Judged by path
    components, so it also works for a path out of another interpreter (the attach path).
    """
    if not filename:
        return True
    if any(marker in filename for marker in _SKIP_PATH_MARKERS):
        return True
    normalised = os.path.normcase(filename)
    if _STDLIB_DIRS and normalised.startswith(_STDLIB_DIRS):
        return True
    parts = [p.lower() for p in filename.replace("\\", "/").split("/")][:-1]   # directories only
    if _LIBRARY_DIRS.intersection(parts):
        return True
    for parent, part in zip(parts, parts[1:]):
        # POSIX:   <prefix>/lib/python3.11/...   (also lib64)
        if parent in ("lib", "lib64") and _PYTHON_LIB_DIR.match(part):
            return True
        # Windows: C:\\Python311\\Lib\\...,  ...\\Programs\\Python\\Python311\\Lib\\...
        if part == "lib" and parent.startswith("python"):
            return True
    return False

# Pulse's own package directory. Not any directory called "pulse": that marker also
# matched ~/pulse/train.py, and a project living there streamed nothing at all.
_PULSE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _is_user_frame(frame: Any) -> bool:
    """True for a frame in the user's own code, rather than in Pulse, a library or the stdlib."""
    filename = getattr(getattr(frame, "f_code", None), "co_filename", "") or ""
    if filename.startswith(_PULSE_DIR):
        return False
    return not is_library_path(filename)


def _collect_locals(frame: Any, depth: int) -> Dict[str, Any]:
    """Locals of the innermost `depth` user frames, innermost winning on a name clash."""
    chain = []
    while frame is not None and len(chain) < depth:
        if _is_user_frame(frame):
            chain.append(frame)
        frame = frame.f_back
    merged: Dict[str, Any] = {}
    for candidate in reversed(chain):          # outermost first, so the innermost wins
        try:
            merged.update(candidate.f_locals)
        except Exception:
            continue
    return merged


def _sample_loop(monitor: "Monitor", thread_id: int, depth: int, interval: float,
                 stop: Optional[threading.Event] = None) -> None:
    """Read the training thread's variables from outside it.

    Pulse used to install sys.settrace and a frame-local trace function, which runs
    Python on every line of the training loop for the life of the run. Sampling from
    another thread costs the training loop nothing at all: it never runs Pulse's code,
    it is never called back into, and the only interaction is this thread briefly
    holding the GIL a few times a second.
    """
    stop = stop if stop is not None else _STOP
    while not stop.wait(monitor.sampler_interval or interval):
        frames = sys._current_frames()
        frame = frames.get(thread_id)
        if frame is None:
            continue
        try:
            monitor.observe_locals(_collect_locals(frame, depth))
        except Exception as exc:                 # sampling must never kill a run
            try:
                monitor.event("sampler_error", error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass


def attach(
    *,
    script_path: Optional[str] = None,
    interval: float = 0.25,
    depth: int = 3,
    tensor_interval: float = 5.0,
    session_id: Optional[str] = None,
    directory: Optional[str] = None,
    thread_id: Optional[int] = None,
) -> "Monitor":
    """Start streaming this thread's training variables to a brain process.

    Returns the Monitor. Sampling happens on a background thread, so the training loop
    is not instrumented at all -- nothing is installed into it and nothing calls into it.
    """
    global _ACTIVE, _SAMPLER, _STOP
    if _ACTIVE is not None:
        return _ACTIVE
    monitor = Monitor(script_path=script_path, session_id=session_id, directory=directory,
                      interval=0.0,            # the sampler thread sets the pace
                      tensor_interval=tensor_interval)
    monitor.sampler_interval = max(0.01, interval)
    monitor.training_thread = thread_id or threading.get_ident()
    _STOP = threading.Event()        # never clear() the old one: its sampler may still run
    _ACTIVE = monitor
    _SAMPLER = threading.Thread(
        target=_sample_loop,
        args=(monitor, thread_id or threading.get_ident(), depth, max(0.01, interval), _STOP),
        name="pulse-monitor-sampler", daemon=True)
    _SAMPLER.start()
    atexit.register(detach)
    return monitor


def detach() -> None:
    global _ACTIVE, _SAMPLER
    monitor, _ACTIVE = _ACTIVE, None
    _STOP.set()
    if _SAMPLER is not None:
        _SAMPLER.join(timeout=1.0)
    _SAMPLER = None
    if monitor is not None:
        try:
            monitor.close()
        except Exception:
            pass


def active() -> Optional["Monitor"]:
    return _ACTIVE