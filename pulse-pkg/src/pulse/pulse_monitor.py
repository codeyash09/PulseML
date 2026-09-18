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

import math
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import pulse_stream as stream

# Values that are worth a device sync to read, in elements.
_SMALL_ELEMENT_LIMIT = 4

# Names that are never interesting, whatever they hold.
_BORING_NAMES = frozenset({
    "self", "cls", "_", "__", "__builtins__", "__name__", "__file__", "__doc__",
    "__package__", "__loader__", "__spec__", "__all__", "__version__", "__annotations__",
})

_BORING_PREFIXES = ("__pulse", "_pulse")


def _is_boring(name: str) -> bool:
    return name in _BORING_NAMES or name.startswith(_BORING_PREFIXES)


def _scalar_of(value: Any) -> Optional[float]:
    """The float in `value`, if reading it is cheap. None otherwise.

    Cheap means: a Python number, or an array-like holding a handful of elements.
    A big tensor returns None even though a float could be computed from it -- that
    computation belongs in the brain, off the training thread.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
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
    if item is None or size != 1:
        return None
    try:
        return float(item())
    except (TypeError, ValueError, RuntimeError):
        return None


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
        self.tensor_interval = float(tensor_interval)
        self.writer = stream.StreamWriter(self.directory)
        self.step = 0

        self._last_sample = 0.0
        self._last_tensor_sample = 0.0
        self._last_control_poll = 0.0
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

        self.writer.write_session({
            "session_id": self.session_id,
            "script": os.path.abspath(script_path) if script_path else None,
            "pid": os.getpid(),
            "started": self._started,
            "interval": self.interval,
        })
        self.writer.emit(stream.KIND_HELLO, {
            "session_id": self.session_id,
            "script": os.path.abspath(script_path) if script_path else None,
            "pid": os.getpid(),
        })

    # ------------------------------------------------------------------ hot path

    def observe_locals(self, local_vars: Dict[str, Any], step: Optional[int] = None) -> None:
        """Sample the caller's variables, if this sample is due.

        Everything before the `return` is what the training loop pays on the samples
        it skips, which is the overwhelming majority of them.
        """
        now = time.monotonic()
        if now - self._last_sample < self.interval:
            return
        self._last_sample = now
        started = time.perf_counter()
        if step is not None:
            self.step = int(step)
        else:
            self.step += 1

        scalars: Dict[str, Any] = {}
        tripwires: List[Tuple[str, float]] = []
        tensor_due = (now - self._last_tensor_sample) >= self.tensor_interval

        for name, value in list(local_vars.items()):
            if _is_boring(name):
                continue
            number = _scalar_of(value)
            if number is not None:
                previous = self._last_values.get(name)
                if previous is None or previous != number:
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

        if now - self._last_control_poll >= 1.0:
            self._last_control_poll = now
            self._handle_control()

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
        for message in self.writer.poll_control():
            action = message.get("action")
            if action == stream.CONTROL_SNAPSHOT:
                names = message.get("names") or []
                self._pending_probe.extend(str(n) for n in names)
            elif action == stream.CONTROL_TRACK:
                for name in message.get("names") or []:
                    self._requested.add(str(name))
            elif action == stream.CONTROL_UNTRACK:
                for name in message.get("names") or []:
                    self._requested.discard(str(name))
            elif action == stream.CONTROL_SET_INTERVAL:
                try:
                    self.interval = max(0.0, float(message.get("interval", self.interval)))
                except (TypeError, ValueError):
                    pass
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

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------ bookkeeping

    def snapshot_state(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Replace state.json so a brain attaching late starts from something real."""
        self.writer.write_state({
            "session_id": self.session_id,
            "script": os.path.abspath(self.script_path) if self.script_path else None,
            "step": self.step,
            "updated": time.time(),
            "scalars": dict(self._last_values),
            "tensors": dict(self._known_tensors),
            "cost": self.cost(),
            **(extra or {}),
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
