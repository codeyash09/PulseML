"""
The half of Pulse that thinks, in a process of its own.

It reads the monitor's stream, keeps the history, runs the detectors, and decides when
to spend a model call. Because it is not in the training process, none of that costs
the run anything: the old design did all of it on the training thread and spent 84% of
one measured run inside blocking model calls.

Two behaviours here are the point of the split.

**The agent chooses when it is next needed.** Every time the brain finishes talking to
the model -- an audit, an escalation, a fix -- the last thing it asks for is when to
come back and why. A run that just had a fix applied, or that is sitting on an unresolved
warning, gets looked at again in minutes. A run that has been descending smoothly for an
hour gets left alone for an hour. Previously only the periodic check-in negotiated its
own interval, and a fix never touched it, so the most dangerous moment in a run -- the
minutes right after code changed underneath it -- was watched no more closely than any
other.

**Waking up means re-reading everything, not glancing at the last value.** The audit is
handed the whole run: every history downsampled in a way that preserves spikes, every
finding, what the tensors look like, what has been fixed, what the monitor dropped. It
is asked specifically for what the deterministic checks cannot see -- a metric that is
technically still moving but far slower than it should, a value that is plausible but
wrong, a pattern that only shows up across two curves. The old check-in asked a version
of this question but was handed a point-in-time snapshot with no history and no code,
so it could not answer it.
"""
from __future__ import annotations

import json
import math
import re
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import pulse_detect as detect
from . import pulse_stream as stream

# Bounds on how far out the agent may schedule its own next look. Below the floor it is
# burning money on a run that has not produced new evidence yet; above the ceiling an
# unattended run can go badly wrong for an hour without anyone noticing.
MIN_INTERVAL_SECONDS = 60.0
MAX_INTERVAL_SECONDS = 3600.0
DEFAULT_INTERVAL_SECONDS = 900.0

# How closely to watch immediately after the code changed under the run.
POST_FIX_INTERVAL_SECONDS = 120.0

HISTORY_CAP = 5000
AUDIT_POINTS = 100


class Schedule:
    """When the brain next looks properly, and why.

    Persisted beside the stream so a brain that is restarted -- or a second one started
    after a fix restarted the training process -- picks up the cadence the agent asked
    for instead of resetting to the default.
    """

    def __init__(self, path: Optional[str] = None, interval: float = DEFAULT_INTERVAL_SECONDS) -> None:
        self.path = path
        self.interval = self._clamp(interval)
        self.reason = "first look"
        self.risk = "unknown"
        self.next_at = time.time() + self.interval
        self.history: List[Dict[str, Any]] = []
        self.loaded = False             # True when a persisted schedule was picked up
        self._load()

    @staticmethod
    def _clamp(seconds: float) -> float:
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return DEFAULT_INTERVAL_SECONDS
        if not math.isfinite(value):
            return DEFAULT_INTERVAL_SECONDS
        return min(MAX_INTERVAL_SECONDS, max(MIN_INTERVAL_SECONDS, value))

    def due(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) >= self.next_at

    def seconds_remaining(self, now: Optional[float] = None) -> float:
        return max(0.0, self.next_at - (now or time.time()))

    def set(self, seconds: float, reason: str = "", risk: str = "unknown") -> None:
        self.interval = self._clamp(seconds)
        self.reason = reason or self.reason
        self.risk = risk or self.risk
        self.next_at = time.time() + self.interval
        self.history.append({"t": time.time(), "interval": self.interval,
                             "reason": self.reason, "risk": self.risk})
        del self.history[:-50]
        self._save()

    def bring_forward(self, seconds: float, reason: str) -> None:
        """Pull the next look closer, never push it out. Used when something happens."""
        target = time.time() + self._clamp(seconds)
        if target < self.next_at:
            self.next_at = target
            self.reason = reason
            self._save()

    def apply_decision(self, decision: Dict[str, Any]) -> None:
        minutes = decision.get("next_check_minutes")
        if minutes is None:
            minutes = decision.get("next_check")
        try:
            seconds = float(minutes) * 60.0
        except (TypeError, ValueError, OverflowError):
            seconds = None
        if seconds is None or not math.isfinite(seconds):
            # No usable interval -- a reply cut off at max_tokens, no JSON, or 'later'.
            # Re-arm with the current one anyway: returning here left next_at in the
            # past, and a full-evidence audit was paid for again on every poll.
            self.set(self.interval, "the last audit did not choose a next check",
                     str(decision.get("risk") or self.risk))
            return
        self.set(seconds, str(decision.get("reason") or ""), str(decision.get("risk") or "unknown"))

    def to_dict(self) -> Dict[str, Any]:
        return {"interval": self.interval, "next_at": self.next_at, "reason": self.reason,
                "risk": self.risk, "history": self.history[-10:]}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            stream._atomic_write_json(self.path, self.to_dict())
        except OSError:
            pass

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            return
        if not isinstance(saved, dict):
            return
        # Every field checked on its own: a schedule file with one bad value in it (a
        # hand edit, a string next_at, a history that is not a list) crashed the brain
        # at startup, and with it the console.
        self.interval = self._clamp(saved.get("interval", self.interval))
        self.reason = str(saved.get("reason") or self.reason)
        self.risk = str(saved.get("risk") or self.risk)
        history = saved.get("history")
        self.history = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
        self.loaded = True
        try:
            saved_next = float(saved.get("next_at") or 0.0)
        except (TypeError, ValueError, OverflowError):
            saved_next = 0.0
        if not math.isfinite(saved_next):
            saved_next = 0.0
        # A brain that was down while training continued should look promptly, but not
        # immediately-and-repeatedly if it is being restarted in a loop.
        self.next_at = min(saved_next or time.time(), time.time() + self.interval)


def downsample(values: Sequence[float], points: int = AUDIT_POINTS) -> List[Dict[str, float]]:
    """Compress a history to `points` buckets, keeping each bucket's extremes.

    A mean-only downsample hides exactly what the audit is looking for: a spike that
    lasted two steps disappears into the average of its neighbours. Each bucket keeps
    its min, mean and max, so a spike survives compression as a bucket whose max is far
    from its mean.
    """
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers:
        return []
    if len(numbers) <= points:
        return [{"i": i, "min": v, "mean": v, "max": v} for i, v in enumerate(numbers)]
    size = len(numbers) / float(points)
    buckets = []
    for i in range(points):
        chunk = numbers[int(i * size):int((i + 1) * size)] or [numbers[min(int(i * size), len(numbers) - 1)]]
        buckets.append({"i": i, "min": min(chunk), "mean": sum(chunk) / len(chunk), "max": max(chunk)})
    return buckets


AUDIT_PROMPT = """\
You are auditing a training run that Pulse has been watching. Pulse's deterministic \
checks have already run over this data and reported what they found (below). Your job \
is the part they cannot do.

Look at the actual numbers. Do not assume the checks would have caught anything worth \
catching: they compare each curve against fixed thresholds, one variable at a time. \
They cannot tell that a loss which is still technically decreasing is decreasing far \
slower than this architecture should, that two metrics are inconsistent with each \
other, that a value is plausible but wrong for this model, or that the run is healthy \
in every way except the one that matters.

{evidence}

Answer in two parts.

1. A short assessment. If something is wrong that the checks did not report, say what \
and give the evidence for it. If the run looks fine, say so in one line -- do not \
invent a concern to look useful.

2. A JSON object on its own line, and nothing after it:
{{"status": "ok" | "watch" | "problem",
  "risk": "low" | "medium" | "high",
  "findings": ["short description", ...],
  "next_check_minutes": <number between {min_minutes:g} and {max_minutes:g}>,
  "reason": "why that interval"}}

Choose next_check_minutes from what you just read, not from habit. Code that changed \
under a running job, a metric heading the wrong way, anything you are unsure about: \
look again soon. A run that has been descending smoothly for a long time with nothing \
outstanding: leave it alone and say so.
"""


class Brain:
    """Reads one session's stream, keeps its history, and decides when to think about it.

    The agent is injected rather than constructed: the brain is about what to ask and
    when, and tests need to drive it without a model. Any callable taking a prompt and
    returning text will do.
    """

    def __init__(
        self,
        directory: str,
        *,
        agent: Optional[Callable[[str], str]] = None,
        sensitivity: float = 0.3,
        poll_interval: float = 0.5,
        on_finding: Optional[Callable[[List[detect.Finding]], None]] = None,
        escalate: Optional[Callable[[List[detect.Finding], Dict[str, Any]], None]] = None,
    ) -> None:
        self.directory = os.path.abspath(directory)
        self.reader = stream.StreamReader(self.directory)
        self.engine = detect.DetectionEngine(sensitivity=sensitivity)
        self.schedule = Schedule(os.path.join(self.directory, "schedule.json"))
        self.agent = agent
        self.poll_interval = poll_interval
        self.on_finding = on_finding
        self.escalate = escalate

        self.histories: Dict[str, List[float]] = {}
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.tensor_stats: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.step = 0
        self.session: Dict[str, Any] = self.reader.session()
        self.finished = False
        self.audits: List[Dict[str, Any]] = []
        self.fixes: List[Dict[str, Any]] = []
        self.last_status = "unknown"
        # The console polls on a background thread while the person can type /audit on
        # the main one, so two audits can start at once: two model calls billed, both
        # writing the schedule, and the later answer silently winning.
        self._audit_lock = threading.Lock()
        # Guards the history and findings. Held for ingesting and for gathering evidence,
        # never across a model call: the console's views take it too, and an audit that
        # held it froze every one of them for as long as the model took to answer.
        self.state_lock = threading.RLock()
        self._last_activity = time.monotonic()
        # Set once the run is over and it has had its closing audit: after that, a due
        # schedule is not a reason to pay for another look at numbers that stopped moving.
        self._final_audit_done = False

    # ------------------------------------------------------------------ ingest

    def ingest(self, frames: Sequence[Dict[str, Any]]) -> List[detect.Finding]:
        """Fold new frames into the history and run the detectors over the result."""
        with self.state_lock:
            urgent, raised = self._fold(frames)
        if urgent:
            # Something the monitor itself flagged as not-worth-waiting-for.
            self.schedule.bring_forward(MIN_INTERVAL_SECONDS,
                                        f"monitor reported {urgent[0].get('event')}")
        if raised:
            if any(f.severity == detect.CRITICAL for f in raised):
                self.schedule.bring_forward(MIN_INTERVAL_SECONDS, "a critical check fired")
            if self.on_finding is not None:
                self.on_finding(raised)
            if self.escalate is not None:
                self.escalate(raised, self.evidence())
        return raised

    def _fold(self, frames: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[detect.Finding]]:
        urgent: List[Dict[str, Any]] = []
        for frame in frames:
            kind = frame.get("kind")
            if kind == stream.KIND_SCALARS:
                self.step = int(frame.get("step") or self.step)
                for name, value in (frame.get("values") or {}).items():
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    history = self.histories.setdefault(name, [])
                    history.append(float(value))
                    if len(history) > HISTORY_CAP:
                        del history[:-HISTORY_CAP]
            elif kind == stream.KIND_TENSOR:
                name = frame.get("name")
                if not name:
                    continue
                meta = {k: v for k, v in frame.items()
                        if k in ("shape", "dtype", "device", "elements", "step")}
                self.tensors[name] = meta
                if frame.get("stats"):
                    self.tensor_stats[name] = dict(frame["stats"])
            elif kind == stream.KIND_EVENT:
                self.events.append(frame)
                del self.events[:-200]
                if frame.get("urgent"):
                    urgent.append(frame)
                if frame.get("event") == "finished":
                    self.finished = True
            elif kind == stream.KIND_HELLO:
                self.session.update({k: v for k, v in frame.items() if k != "kind"})
            elif kind == stream.KIND_BYE:
                self.finished = True

        result = self.engine.update(self.histories, step=self.step, tensor_stats=self.tensor_stats)
        return urgent, result["raised"]

    # ------------------------------------------------------------------ evidence

    def evidence(self, include_code: bool = False) -> Dict[str, Any]:
        """Everything known about this run, small enough to put in a prompt."""
        with self.state_lock:
            return self._evidence(include_code)

    def _evidence(self, include_code: bool) -> Dict[str, Any]:
        findings = [f.to_dict() for f in self.engine.current()]
        curves = {}
        for name, history in self.histories.items():
            # One reading is still a reading: the first val_loss after an epoch, a final
            # test score. Leaving it out hid it from the audit while /vars showed it.
            if not history:
                continue
            curves[name] = {
                "points": len(history),
                "first": history[0],
                "last": history[-1],
                "min": min(history),
                "max": max(history),
                "curve": downsample(history),
            }
        pack: Dict[str, Any] = {
            "session": {k: self.session.get(k) for k in ("session_id", "script", "pid")},
            "step": self.step,
            "elapsed_seconds": round(time.time() - float(self.session.get("started") or time.time()), 1),
            "scalars": curves,
            "tensors": {k: dict(v) for k, v in self.tensors.items()},
            "tensor_stats": {k: dict(v) for k, v in self.tensor_stats.items()},
            "findings": findings,
            "recent_events": self.events[-20:],
            "fixes_applied": self.fixes[-5:],
            "previous_audits": [{"t": a.get("t"), "status": a.get("status"), "risk": a.get("risk"),
                                 "findings": a.get("findings")} for a in self.audits[-3:]],
            "stream_gaps": self.reader.gaps,
        }
        if include_code:
            pack["code"] = self._read_script()
        return pack

    def _read_script(self) -> Optional[str]:
        path = self.session.get("script")
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            return None
        return "\n".join(f"{i + 1:>4} | {line}" for i, line in enumerate(lines))

    @staticmethod
    def render_evidence(pack: Dict[str, Any]) -> str:
        """The evidence as text for a prompt: compact, but nothing silently dropped."""
        out: List[str] = []
        out.append(f"Run: step {pack.get('step')}, {pack.get('elapsed_seconds')}s elapsed, "
                   f"script {(pack.get('session') or {}).get('script')}")
        if pack.get("stream_gaps"):
            out.append(f"NOTE: {pack['stream_gaps']} sample(s) were dropped under load; "
                       f"the curves below have holes.")
        findings = pack.get("findings") or []
        out.append("\nWhat Pulse's deterministic checks currently report:")
        if findings:
            for f in findings:
                out.append(f"  [{f['severity']}] {f['check']} on {f['variable']}: {f['message']}"
                           f"  (seen {f['count']}x, confidence {f['confidence']})")
        else:
            out.append("  nothing")
        out.append("\nTracked values (each curve downsampled to buckets of min/mean/max, "
                   "so brief spikes survive the compression):")
        for name, info in (pack.get("scalars") or {}).items():
            out.append(f"  {name}: {info['points']} readings, first {info['first']:.6g}, "
                       f"last {info['last']:.6g}, min {info['min']:.6g}, max {info['max']:.6g}")
            curve = info.get("curve") or []
            rendered = ", ".join(
                (f"{b['mean']:.4g}" if abs(b["max"] - b["min"]) <= abs(b["mean"]) * 1e-6
                 else f"{b['mean']:.4g}[{b['min']:.4g}..{b['max']:.4g}]")
                for b in curve)
            out.append(f"    {rendered}")
        tensors = pack.get("tensors") or {}
        if tensors:
            out.append("\nTensors seen:")
            for name, meta in tensors.items():
                stats = (pack.get("tensor_stats") or {}).get(name)
                shape = "x".join(str(d) for d in (meta.get("shape") or []))
                line = f"  {name}: shape {shape} {meta.get('dtype', '')} on {meta.get('device', 'cpu')}"
                if stats:
                    # An all-NaN tensor has no min/max/mean (None): exactly the failure
                    # Pulse exists for, and '%.4g' % None crashed every audit over it.
                    line += (f"  min {_num(stats.get('min'))} max {_num(stats.get('max'))} "
                             f"mean {_num(stats.get('mean'))} nan {stats.get('nan', 0)} "
                             f"inf {stats.get('inf', 0)}")
                out.append(line)
        events = pack.get("recent_events") or []
        if events:
            out.append("\nRecent events:")
            for e in events[-10:]:
                out.extend(_describe_event(e))
        fixes = pack.get("fixes_applied") or []
        if fixes:
            out.append("\nFixes already applied this run:")
            for fix in fixes:
                out.append(f"  step {fix.get('step')}: {fix.get('summary')}")
        audits = pack.get("previous_audits") or []
        if audits:
            out.append("\nWhat previous audits concluded: " + "; ".join(
                f"{a.get('status')} (risk {a.get('risk')})" for a in audits))
        if pack.get("code"):
            heading = "\nTraining code:"
            if fixes:
                # A fix edits the file; the process keeps running the code it started with (the
                # run only takes a fix up when it is restarted). Read without this, the curves
                # were blamed on code that never produced them.
                first = min((f.get("step") or 0) for f in fixes)
                heading = (f"\nTraining code AS IT IS NOW on disk -- it was changed during this run (from "
                           f"step {first}, see the fixes above). This process keeps running the code it "
                           "started with until it is restarted: the curves were produced by the code "
                           "before those changes, not by what is shown here.")
            out.append(heading + "\n" + pack["code"])
        return "\n".join(out)

    # ------------------------------------------------------------------ thinking

    def audit(self, include_code: bool = True) -> Dict[str, Any]:
        """The wake-up pass: re-read everything and look for what the checks cannot see."""
        if not self._audit_lock.acquire(blocking=False):
            return {"status": "busy", "reason": "an audit is already running"}
        try:
            return self._audit(include_code)
        finally:
            self._audit_lock.release()

    def _audit(self, include_code: bool) -> Dict[str, Any]:
        pack = self.evidence(include_code=include_code)
        prompt = AUDIT_PROMPT.format(evidence=self.render_evidence(pack),
                                     min_minutes=MIN_INTERVAL_SECONDS / 60.0,
                                     max_minutes=MAX_INTERVAL_SECONDS / 60.0)
        if self.agent is None:
            # No model available: keep the loop honest rather than pretending to audit.
            self.schedule.set(self.schedule.interval, "no agent configured", "unknown")
            return {"status": "skipped", "reason": "no agent configured"}
        try:
            answer = self.agent(prompt)
        except Exception as exc:                       # a failed audit must not stop monitoring
            self.schedule.set(max(MIN_INTERVAL_SECONDS, self.schedule.interval / 2),
                              f"audit failed: {type(exc).__name__}", "unknown")
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        decision = parse_decision(answer)
        with self.state_lock:
            record = dict(decision, t=time.time(), step=self.step, text=answer)
            self.audits.append(record)
            del self.audits[:-20]
            self.last_status = str(decision.get("status") or "unknown")
        self.schedule.apply_decision(decision)
        if self.last_status == "problem" and self.escalate is not None:
            # The audit found something the checks did not: that is a problem to act on,
            # not just a line of text. It used to go nowhere.
            with self.state_lock:
                current = list(self.engine.current())
            try:
                self.escalate(current, dict(pack, audit=record))
            except Exception:
                pass
        return record

    def note_fix(self, summary: str, **fields: Any) -> None:
        """Record that the code changed under the run, and look again soon.

        This is the gap the split is meant to close: the interval used to be negotiated
        only by the periodic check-in, and applying a fix never touched it, so the
        riskiest minutes in a run were watched on the same lazy cadence as any other.
        """
        self.fixes.append(dict(fields, summary=summary, step=self.step, t=time.time()))
        del self.fixes[:-20]
        self.schedule.bring_forward(POST_FIX_INTERVAL_SECONDS, f"code changed: {summary}")

    # ------------------------------------------------------------------ loop

    def poll_once(self) -> Dict[str, Any]:
        """One turn of the loop: ingest, detect, and audit if one is due.

        A finished run -- or one whose process is gone -- gets one audit at most. The
        console calls this every 0.5 s for as long as it is open, not run(), so the check
        has to be here: `pulse --model` left open on yesterday's run paid for an audit of
        it every interval.
        """
        frames = self.reader.poll()
        if frames:
            self._last_activity = time.monotonic()
        raised = self.ingest(frames) if frames else []
        audited = None
        if self.schedule.due() and not self._final_audit_done:
            if not self.finished and not frames:
                self._check_process_gone()
            was_finished = self.finished
            audited = self.audit()
            if was_finished and (audited or {}).get("status") != "busy":
                self._final_audit_done = True
        return {"frames": len(frames), "raised": raised, "audit": audited}

    def run(self, stop: Optional[Callable[[], bool]] = None) -> None:
        while not self.finished:
            if stop is not None and stop():
                return
            if not self.poll_once()["frames"] and self._check_process_gone():
                break
            time.sleep(self.poll_interval)
        self.poll_once()        # drain whatever arrived with the closing frames

    def _check_process_gone(self) -> bool:
        """Mark the run over if its process died without a word. True if it did."""
        if self.finished or not self._training_process_gone():
            return False
        # Killed without a word (SIGKILL, the OOM killer, pre-emption): no 'finished'
        # and no BYE will ever come, and audits would be billed forever.
        self.events.append({"kind": stream.KIND_EVENT, "event": "process_gone",
                            "step": self.step})
        self.finished = True
        return True

    def _training_process_gone(self, quiet_seconds: float = 2.0) -> bool:
        """The run's process no longer exists, and nothing has arrived for a while."""
        if time.monotonic() - self._last_activity < quiet_seconds:
            return False
        if not self.session.get("pid"):
            self.session.update({k: v for k, v in self.reader.session().items()
                                 if k not in self.session})
        pid = self.session.get("pid")
        if not pid:
            return False
        from .pulse_console import session_liveness

        # False only when this machine can tell: a pid from a container's own namespace
        # names some other process here, and says nothing either way.
        return session_liveness(self.session) is False


def _num(value: Any) -> str:
    """A statistic for a prompt; a missing one (an all-NaN tensor has none) as n/a."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4g}"
    return "n/a"


def _describe_event(event: Dict[str, Any]) -> List[str]:
    """One event for a prompt, with what it says rather than just its name.

    'crash@41' told the agent a crash happened and nothing about it; the exception and the
    end of the traceback are the evidence. The same for which value went non-finite.
    """
    line = f"  {event.get('event')}@{event.get('step')}"
    details = []
    if event.get("name") is not None:
        details.append(str(event["name"]) + (f" = {event['value']}" if "value" in event else ""))
    for key in ("exception", "error", "reason", "message"):
        if event.get(key):
            details.append(str(event[key])[:500])
    if details:
        line += ": " + "; ".join(details)
    out = [line]
    traceback = event.get("traceback")
    if traceback:
        tail = str(traceback).rstrip().splitlines()[-8:]
        out.extend("      " + text for text in tail)
    return out


def parse_decision(answer: str) -> Dict[str, Any]:
    """Pull the decision JSON out of a model reply.

    Tolerant on purpose: the alternative in the old code was matching the reply against
    a set of acceptable status strings, so a model that said "looks fine to me" instead
    of "ok" was read as a problem and escalated.
    """
    if not answer:
        return {}
    text = answer.strip()
    start = text.rfind("{")
    while start != -1:
        end = text.find("}", start)
        while end != -1:
            chunk = text[start:end + 1]
            try:
                loaded = json.loads(chunk)
            except ValueError:
                end = text.find("}", end + 1)
                continue
            if isinstance(loaded, dict) and ("next_check_minutes" in loaded or "status" in loaded):
                return loaded
            end = text.find("}", end + 1)
        start = text.rfind("{", 0, start)
    return _labelled_decision(text)


_LABEL_RE = {
    "status": re.compile(r"(?im)^[\s*_#>-]*status[\s*_]*[:=-][\s*_`\"']*(ok|watch|problem)\b"),
    "risk": re.compile(r"(?im)^[\s*_#>-]*risk[\s*_]*[:=-][\s*_`\"']*(low|medium|high)\b"),
    "next_check_minutes": re.compile(
        r"(?im)^[\s*_#>-]*next[\s_]*check(?:[\s_]*minutes)?[\s*_]*[:=-][\s*_`\"']*(\d+(?:\.\d+)?)"),
}


def _labelled_decision(text: str) -> Dict[str, Any]:
    """The decision written as labelled lines instead of JSON ("**Status:** problem", "Risk:
    high", "Next check: 1 minute"). A model that answered the audit that way had its whole
    diagnosis dropped -- read as no verdict at all, and so never shown."""
    found: Dict[str, Any] = {}
    for key, pattern in _LABEL_RE.items():
        m = pattern.search(text)
        if m:
            found[key] = float(m.group(1)) if key == "next_check_minutes" else m.group(1).lower()
    if "status" not in found:
        return {}
    found.setdefault("findings", [])
    return found


# Room for the answer AND the thinking before it: a reasoning model (DeepSeek V4, o-series,
# Claude with thinking) counts its reasoning against max_tokens, and at 4,000 it spent all of
# it thinking about an audit's evidence and returned an empty answer -- every scheduled check
# paid for and silently recorded as "no verdict". The same budget the agent's own calls get.
AGENT_MAX_TOKENS = 32000


class EmptyAnswer(RuntimeError):
    """The model returned no answer text (it ran out of tokens while thinking, or refused)."""


# A careful audit of a long run can think for 20,000 tokens: 5-10 minutes on a fast model.
# At 300 s one in six was cut off -- still billed, and recorded as a failed audit.
AGENT_TIMEOUT_SECONDS = 900.0


def build_litellm_agent(model: str, api_key: Optional[str] = None, api_base: Optional[str] = None,
                        max_tokens: int = AGENT_MAX_TOKENS,
                        timeout: float = AGENT_TIMEOUT_SECONDS) -> Callable[[str], str]:
    """An agent callable backed by litellm, for running the brain standalone. An empty answer
    raises EmptyAnswer rather than returning "": a caller that parses "" sees no verdict and
    cannot tell that apart from a model that looked and had nothing to say."""
    if not api_key:
        # An openrouter/ model with no key given: the environment's, else the key from
        # `pulse openrouter` (litellm itself only knows about the environment).
        from . import pulse_openrouter
        api_key = pulse_openrouter.key_for(model)

    def ask(prompt: str) -> str:
        import litellm
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
            **({"api_key": api_key} if api_key else {}),
            **({"api_base": api_base} if api_base else {}),
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            finish = getattr(response.choices[0], "finish_reason", None)
            raise EmptyAnswer("the model gave no answer" + (
                f" (it used its whole {max_tokens:,}-token budget, most likely thinking)" if finish == "length"
                else f" (finish reason: {finish})" if finish else ""))
        return text
    ask.model = model                # what this agent thinks with, for whoever is handed it
    return ask


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pulse.brain",
        description="Watch a Pulse monitoring stream from outside the training process.")
    parser.add_argument("directory", nargs="?", help="session directory (default: the newest one)")
    parser.add_argument("--model", default=os.environ.get("PULSE_BRAIN_MODEL", ""),
                        help="litellm model id for the audit pass; omit to run detection only")
    parser.add_argument("--sensitivity", type=float, default=0.3)
    parser.add_argument("--interval", type=float, default=None,
                        help="seconds until the first audit (the agent chooses after that; "
                             f"default: the saved schedule, else {DEFAULT_INTERVAL_SECONDS:g})")
    parser.add_argument("--once", action="store_true", help="one pass over what exists, then exit")
    args = parser.parse_args(argv)

    directory = args.directory
    if not directory:
        sessions = stream.list_sessions()
        if not sessions:
            print("No Pulse sessions found. Start a run with auto_track(stream=True).")
            return 2
        directory = sessions[0]["directory"]
        print(f"Watching the newest session: {directory}")

    agent = build_litellm_agent(args.model) if args.model else None
    brain = Brain(directory, agent=agent, sensitivity=args.sensitivity)
    # Only when asked, or when there is nothing saved: the schedule is persisted precisely
    # so a restarted brain keeps the cadence the agent chose (2 minutes after a fix, say).
    if args.interval is not None:
        brain.schedule.set(args.interval, "startup", brain.schedule.risk)
    elif not brain.schedule.loaded:
        brain.schedule.set(DEFAULT_INTERVAL_SECONDS, "startup", brain.schedule.risk)

    def report(findings: List[detect.Finding]) -> None:
        for finding in findings:
            print(f"[{finding.severity}] {finding.message}")

    brain.on_finding = report
    if args.once:
        result = brain.poll_once()
        print(json.dumps({"frames": result["frames"],
                          "findings": [f.to_dict() for f in brain.engine.current()],
                          "audit": (result["audit"] or {}).get("status")}, indent=2, default=str))
        return 0
    try:
        brain.run()
    except KeyboardInterrupt:
        pass
    print(f"Run finished. {len(brain.engine.current())} open finding(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
