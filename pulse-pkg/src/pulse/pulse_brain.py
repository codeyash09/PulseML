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
        except (TypeError, ValueError):
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
        self.interval = self._clamp(saved.get("interval", self.interval))
        self.reason = str(saved.get("reason") or self.reason)
        self.risk = str(saved.get("risk") or self.risk)
        self.history = list(saved.get("history") or [])
        # A brain that was down while training continued should look promptly, but not
        # immediately-and-repeatedly if it is being restarted in a loop.
        self.next_at = min(float(saved.get("next_at") or 0.0) or time.time(),
                           time.time() + self.interval)


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

    # ------------------------------------------------------------------ ingest

    def ingest(self, frames: Sequence[Dict[str, Any]]) -> List[detect.Finding]:
        """Fold new frames into the history and run the detectors over the result."""
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
        raised = result["raised"]
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

    # ------------------------------------------------------------------ evidence

    def evidence(self, include_code: bool = False) -> Dict[str, Any]:
        """Everything known about this run, small enough to put in a prompt."""
        findings = [f.to_dict() for f in self.engine.current()]
        curves = {}
        for name, history in self.histories.items():
            if len(history) < 2:
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
            "tensors": self.tensors,
            "tensor_stats": self.tensor_stats,
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
                    line += ("  min %.4g max %.4g mean %.4g nan %s inf %s" %
                             (stats.get("min", float("nan")), stats.get("max", float("nan")),
                              stats.get("mean", float("nan")), stats.get("nan", 0), stats.get("inf", 0)))
                out.append(line)
        events = pack.get("recent_events") or []
        if events:
            out.append("\nRecent events: " + ", ".join(
                f"{e.get('event')}@{e.get('step')}" for e in events[-10:]))
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
            out.append("\nTraining code:\n" + pack["code"])
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
        record = dict(decision, t=time.time(), step=self.step, text=answer)
        self.audits.append(record)
        del self.audits[:-20]
        self.last_status = str(decision.get("status") or "unknown")
        self.schedule.apply_decision(decision)
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
        """One turn of the loop: ingest, detect, and audit if one is due."""
        frames = self.reader.poll()
        raised = self.ingest(frames) if frames else []
        audited = None
        if self.schedule.due():
            audited = self.audit()
        return {"frames": len(frames), "raised": raised, "audit": audited}

    def run(self, stop: Optional[Callable[[], bool]] = None) -> None:
        while not self.finished:
            if stop is not None and stop():
                return
            self.poll_once()
            time.sleep(self.poll_interval)
        self.poll_once()        # drain whatever arrived with the closing frames


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
    return {}


def build_litellm_agent(model: str, api_key: Optional[str] = None, api_base: Optional[str] = None,
                        max_tokens: int = 4000, timeout: float = 300.0) -> Callable[[str], str]:
    """An agent callable backed by litellm, for running the brain standalone."""
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
        return (response.choices[0].message.content or "").strip()
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
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS,
                        help="seconds until the first audit (the agent chooses after that)")
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
    brain.schedule.set(args.interval, "startup", brain.schedule.risk)

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
