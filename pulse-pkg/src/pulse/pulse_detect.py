"""
Deterministic detection, as data rather than prose.

This replaces a detector that returned `"; ".join(reasons)` -- a single string that
was also its own deduplication key. That had three consequences worth naming, because
each one is fixed here:

* **A number in the message made it a different problem.** The key was the whole
  formatted string, so "spiked to 4.12" and "spiked to 4.13" were unrelated events and
  both escalated. A finding now keys on (check, variable), and its numbers live in
  fields rather than in the key.

* **A problem that did not change was reported once, ever.** The same string never
  re-fired, even after a fix was applied and the run went bad again in the same way.
  Findings here clear when the condition goes away and can fire again afterwards.

* **One noisy sample was enough.** Every check fired on the first evaluation that
  crossed a threshold. Checks now have to hold for `confirmations` consecutive
  evaluations before anything is raised, except the ones where waiting is itself
  dangerous, like a NaN.

It also stops asking only about the variables discovered in the source. Under Keras
the interesting values -- `val_loss`, `accuracy` -- arrive as callback log keys and
never appear as script locals, so most checks simply never ran on a Keras run. The
engine here is given histories, whatever their origin.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

_SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}

# Checks that skip the confirmation wait, because waiting is itself the damage: once a
# value is NaN every subsequent step is wasted compute, and no second opinion is going
# to make it finite again. Everything else has to hold for `confirmations` rounds --
# a loss spike can be one bad batch, and firing on it was a reliable false alarm.
_IMMEDIATE_CHECKS = frozenset({"nonfinite", "tensor_nonfinite"})

LOSS_NAME_HINTS = ("loss", "cost", "nll", "cross_entropy", "crossentropy", "objective", "err")
METRIC_NAME_HINTS = ("acc", "accuracy", "f1", "auc", "precision", "recall", "iou", "dice",
                     "bleu", "rouge", "map", "mrr", "r2", "score")
NORM_NAME_HINTS = ("grad_norm", "gradnorm", "grad_scale", "weight_norm", "param_norm", "update_norm")
LR_NAME_HINTS = ("lr", "learning_rate", "learningrate", "step_size")
VAL_PREFIXES = ("val", "valid", "validation", "test", "eval", "dev")


def _lower(name: str) -> str:
    return (name or "").lower()


def looks_like_loss(name: str) -> bool:
    low = _lower(name)
    return any(hint in low for hint in LOSS_NAME_HINTS)


def looks_like_metric(name: str) -> bool:
    low = _lower(name)
    return any(hint in low for hint in METRIC_NAME_HINTS) and not looks_like_loss(name)


def looks_like_norm(name: str) -> bool:
    low = _lower(name).replace("-", "_")
    return any(hint in low for hint in NORM_NAME_HINTS)


def looks_like_lr(name: str) -> bool:
    low = _lower(name).replace("-", "_")
    return low in LR_NAME_HINTS or any(low.endswith("_" + hint) or low.startswith(hint + "_")
                                       for hint in LR_NAME_HINTS)


def is_validation(name: str) -> bool:
    low = _lower(name)
    return any(low.startswith(prefix) or ("_" + prefix) in low for prefix in VAL_PREFIXES)


def thresholds(sensitivity: float) -> Dict[str, float]:
    """Turn the single 0..1 dial into the numbers the checks compare against.

    Same shape as the thresholds the old detector derived, with two differences: the
    learning-rate jump ratio and the never-learned ratio used to be hardcoded, so no
    amount of turning the dial affected them. They are on the dial now.
    """
    s = min(1.0, max(0.0, float(sensitivity)))
    return {
        "explosion_multiplier": 8.0 - 5.5 * s,          # 8x at 0, 2.5x at 1
        "plateau_range_frac": 10 ** (-5 + 3 * s),
        "oscillation_flip_threshold": round(15 - 9 * s),
        "oscillation_delta_frac": 0.10 - 0.08 * s,
        "stagnation_frac": 0.003 + 0.077 * s,
        "stagnation_window": max(10, round(500 - 350 * s)),
        "lr_jump_ratio": 20.0 - 12.0 * s,               # was hardcoded 10x
        "never_learned_ratio": 0.995 - 0.03 * s,        # was hardcoded 0.98
        "divergence_val_frac": max(0.05, 0.003 + 0.077 * s),
        "divergence_train_frac": 0.02,
        "perfect_metric": 0.999,
    }


class Finding:
    """One thing a check believes about one variable, with the evidence attached."""

    __slots__ = ("check", "variable", "severity", "message", "step", "values",
                 "confidence", "first_seen", "last_seen", "count")

    def __init__(self, check: str, variable: str, severity: str, message: str,
                 step: Optional[int] = None, values: Optional[Dict[str, Any]] = None,
                 confidence: float = 0.5) -> None:
        self.check = check
        self.variable = variable
        self.severity = severity
        self.message = message
        self.step = step
        self.values = values or {}
        self.confidence = confidence
        now = time.time()
        self.first_seen = now
        self.last_seen = now
        self.count = 1

    @property
    def key(self) -> Tuple[str, str]:
        return (self.check, self.variable)

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "variable": self.variable, "severity": self.severity,
                "message": self.message, "step": self.step, "values": self.values,
                "confidence": round(self.confidence, 3), "first_seen": self.first_seen,
                "last_seen": self.last_seen, "count": self.count}

    def __repr__(self) -> str:
        return f"<Finding {self.severity} {self.check}:{self.variable} {self.message!r}>"


def _finite(history: Sequence[Any]) -> List[float]:
    return [float(v) for v in history if isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v))]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _confidence(observed: float, threshold: float, points: int) -> float:
    """How much past the line it is, tempered by how much data we have."""
    if threshold <= 0:
        margin = 1.0 if observed > 0 else 0.0
    else:
        margin = min(1.0, max(0.0, (observed - threshold) / max(threshold, 1e-12)))
    evidence = min(1.0, points / 30.0)
    return round(0.4 + 0.4 * margin + 0.2 * evidence, 3)


class DetectionEngine:
    """Runs the checks and decides when a belief is worth raising.

    Feed it `{name: [values...]}` whenever new data arrives. It returns the findings
    that *changed state* this round -- newly raised or newly cleared -- so a caller can
    act on those rather than re-reading a list of everything currently wrong.
    """

    def __init__(self, sensitivity: float = 0.3, confirmations: int = 2, rearm_after: int = 3) -> None:
        self.sensitivity = sensitivity
        self.confirmations = max(1, int(confirmations))
        self.rearm_after = max(1, int(rearm_after))
        self.active: Dict[Tuple[str, str], Finding] = {}
        self._streak: Dict[Tuple[str, str], int] = {}
        self._clear_streak: Dict[Tuple[str, str], int] = {}
        self._baselines: Dict[str, float] = {}        # agent-supplied "this is normal" anchors

    # ------------------------------------------------------------------ public

    def set_baseline(self, variable: str, value: float) -> None:
        """Tell the engine what normal looks like for a variable (the agent can know this)."""
        self._baselines[variable] = float(value)

    def update(self, histories: Dict[str, Sequence[Any]], *, step: Optional[int] = None,
               tensor_stats: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, List[Finding]]:
        """Evaluate every check against every history. Returns {"raised": [...], "cleared": [...]}."""
        t = thresholds(self.sensitivity)
        fired: Dict[Tuple[str, str], Finding] = {}

        for name, history in (histories or {}).items():
            if not history:
                continue
            for finding in self._check_variable(name, list(history), t, step):
                fired[finding.key] = finding
        for name, stats in (tensor_stats or {}).items():
            finding = self._check_tensor(name, stats, step)
            if finding is not None:
                fired[finding.key] = finding
        for finding in self._check_pairs(histories or {}, t, step):
            fired[finding.key] = finding

        return self._settle(fired)

    def current(self) -> List[Finding]:
        """Everything currently believed, worst first."""
        return sorted(self.active.values(),
                      key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), -f.confidence))

    # ------------------------------------------------------------------ state machine

    def _settle(self, fired: Dict[Tuple[str, str], Finding]) -> Dict[str, List[Finding]]:
        raised: List[Finding] = []
        cleared: List[Finding] = []

        for key, finding in fired.items():
            self._clear_streak.pop(key, None)
            needed = 1 if finding.check in _IMMEDIATE_CHECKS else self.confirmations
            streak = self._streak.get(key, 0) + 1
            self._streak[key] = streak
            if key in self.active:
                existing = self.active[key]
                existing.last_seen = time.time()
                existing.count += 1
                existing.message = finding.message
                existing.values = finding.values
                existing.step = finding.step
                existing.confidence = max(existing.confidence, finding.confidence)
                continue
            if streak >= needed:
                self.active[key] = finding
                raised.append(finding)

        for key in list(self.active):
            if key in fired:
                continue
            missed = self._clear_streak.get(key, 0) + 1
            self._clear_streak[key] = missed
            self._streak.pop(key, None)
            if missed >= self.rearm_after:
                cleared.append(self.active.pop(key))
                self._clear_streak.pop(key, None)
        for key in list(self._streak):
            if key not in fired:
                self._streak.pop(key, None)

        return {"raised": raised, "cleared": cleared}

    # ------------------------------------------------------------------ checks

    def _check_variable(self, name: str, history: List[Any], t: Dict[str, float],
                        step: Optional[int]) -> Iterable[Finding]:
        out: List[Finding] = []
        raw_last = history[-1] if history else None
        if isinstance(raw_last, (int, float)) and not isinstance(raw_last, bool) and not math.isfinite(float(raw_last)):
            kind = "NaN" if math.isnan(float(raw_last)) else "infinite"
            out.append(Finding("nonfinite", name, CRITICAL,
                               f"{name} is {kind}; every step from here is wasted compute",
                               step, {"value": str(raw_last), "points": len(history)}, 1.0))
            return out              # nothing else about this variable is meaningful now

        values = _finite(history)
        if len(values) < 3:
            return out
        latest = values[-1]
        is_loss = looks_like_loss(name)
        is_metric = looks_like_metric(name)

        # Frozen: the same bits over and over. A loss that does not move at all is
        # usually a detached graph or an optimizer that never steps.
        if (is_loss or is_metric) and len(values) >= 6 and len(set(values[-6:])) == 1:
            out.append(Finding("frozen", name, WARNING,
                               f"{name} has been exactly {latest:g} for {min(len(values), 6)} readings",
                               step, {"value": latest}, 0.85))

        if looks_like_norm(name) and len(values) >= 5:
            baseline = _mean(values[-50:-1]) if len(values) > 1 else latest
            if baseline > 0:
                if latest > baseline * t["explosion_multiplier"]:
                    out.append(Finding("norm_explosion", name, CRITICAL,
                                       f"{name} jumped to {latest:g}, {latest / baseline:.1f}x its recent average",
                                       step, {"latest": latest, "baseline": baseline},
                                       _confidence(latest / baseline, t["explosion_multiplier"], len(values))))
                elif latest * t["explosion_multiplier"] < baseline:
                    out.append(Finding("norm_collapse", name, WARNING,
                                       f"{name} collapsed to {latest:g} from a recent average of {baseline:g}",
                                       step, {"latest": latest, "baseline": baseline}, 0.7))

        if looks_like_lr(name) and len(values) >= 2:
            previous = values[-2]
            if previous > 0 and latest > 0:
                ratio = latest / previous
                if ratio >= t["lr_jump_ratio"] or ratio <= 1.0 / t["lr_jump_ratio"]:
                    out.append(Finding("lr_jump", name, WARNING,
                                       f"{name} changed by {ratio:.3g}x in one step ({previous:g} -> {latest:g})",
                                       step, {"from": previous, "to": latest, "ratio": ratio}, 0.75))

        if is_loss:
            out.extend(self._check_loss(name, values, t, step))
        elif is_metric:
            out.extend(self._check_metric(name, values, t, step))
        return out

    def _check_loss(self, name: str, values: List[float], t: Dict[str, float],
                    step: Optional[int]) -> Iterable[Finding]:
        out: List[Finding] = []
        latest = values[-1]
        window = values[-20:]

        if len(values) >= 5:
            baseline = min(values[-50:-1])
            anchor = self._baselines.get(name)
            if anchor is not None:
                baseline = min(baseline, anchor)
            if baseline > 0 and latest > baseline * t["explosion_multiplier"]:
                out.append(Finding("loss_spike", name, CRITICAL,
                                   f"{name} spiked to {latest:g}, {latest / baseline:.1f}x its recent floor of {baseline:g}",
                                   step, {"latest": latest, "baseline": baseline},
                                   _confidence(latest / baseline, t["explosion_multiplier"], len(values))))

        if len(values) >= 8:
            scale = _mean([abs(v) for v in window]) or 1.0
            if (max(window) - min(window)) <= scale * t["plateau_range_frac"]:
                out.append(Finding("plateau", name, WARNING,
                                   f"{name} has barely moved over its last {len(window)} readings "
                                   f"(range {max(window) - min(window):.3g} around {scale:.3g})",
                                   step, {"range": max(window) - min(window), "scale": scale}, 0.7))

            deltas = [b - a for a, b in zip(window, window[1:])]
            flips = sum(1 for a, b in zip(deltas, deltas[1:]) if a * b < 0)
            mean_step = _mean([abs(d) for d in deltas])
            scale = _mean([abs(v) for v in window]) or 1.0
            if flips >= t["oscillation_flip_threshold"] and mean_step > scale * t["oscillation_delta_frac"]:
                out.append(Finding("oscillation", name, WARNING,
                                   f"{name} is bouncing: {flips} direction changes in {len(deltas)} steps, "
                                   f"average move {mean_step:.3g}",
                                   step, {"flips": flips, "mean_step": mean_step},
                                   _confidence(flips, t["oscillation_flip_threshold"], len(values))))

        window_size = int(min(t["stagnation_window"], max(8, len(values))))
        if len(values) >= max(8, window_size // 4):
            recent = values[-window_size:]
            quarter = max(2, len(recent) // 4)
            early, late = _mean(recent[:quarter]), _mean(recent[-quarter:])
            if abs(early) > 1e-12:
                change = abs(late - early) / abs(early)
                if change < t["stagnation_frac"]:
                    out.append(Finding("stagnation", name, WARNING,
                                       f"{name} moved {change * 100:.2f}% across its last {len(recent)} readings "
                                       f"({early:.4g} -> {late:.4g})",
                                       step, {"early": early, "late": late, "change": change}, 0.65))

        # "Never learned": the end is no better than the beginning. Every other check
        # looks for training going wrong; this one is for training that never started.
        if len(values) >= 10:
            span = max(2, len(values) // 5)
            start, end = _mean(values[:span]), _mean(values[-span:])
            if start > 0 and end >= start * t["never_learned_ratio"]:
                out.append(Finding("never_learned", name, CRITICAL,
                                   f"{name} is no better than when it started ({start:.4g} -> {end:.4g} "
                                   f"over {len(values)} readings): this run is not learning",
                                   step, {"start": start, "end": end, "points": len(values)}, 0.9))

        if is_validation(name) and len(values) >= 5:
            last5 = values[-5:]
            worsening = sum(1 for a, b in zip(last5, last5[1:]) if b > a)
            if worsening >= 4 and last5[-1] > last5[0] > 0:
                out.append(Finding("val_regression", name, WARNING,
                                   f"{name} has worsened in {worsening} of its last 4 steps "
                                   f"({last5[0]:.4g} -> {last5[-1]:.4g})",
                                   step, {"from": last5[0], "to": last5[-1]}, 0.8))
            elif len(values) >= 8:
                half = len(values) // 2
                early, late = _mean(values[:half]), _mean(values[half:])
                if early > 0 and late > early * (1 + t["stagnation_frac"]):
                    out.append(Finding("val_drift", name, WARNING,
                                       f"{name} is drifting up: {early:.4g} early, {late:.4g} late",
                                       step, {"early": early, "late": late}, 0.65))
        return out

    def _check_metric(self, name: str, values: List[float], t: Dict[str, float],
                      step: Optional[int]) -> Iterable[Finding]:
        out: List[Finding] = []
        # Too good, too fast. The old version of this check compared len(history) to
        # exactly 2, so it could only ever fire in the single instant the history had
        # two points, and was dead for the rest of the run.
        if 2 <= len(values) <= 6 and max(values) >= t["perfect_metric"]:
            out.append(Finding("suspiciously_perfect", name, WARNING,
                               f"{name} reached {max(values):.4g} within {len(values)} readings, "
                               f"which usually means the labels are reachable from the inputs",
                               step, {"value": max(values), "points": len(values)}, 0.7))
        if len(values) >= 10:
            quarter = max(2, len(values) // 4)
            early, late = _mean(values[:quarter]), _mean(values[-quarter:])
            if abs(early) > 1e-12 and abs(late - early) / abs(early) < t["stagnation_frac"]:
                out.append(Finding("metric_stagnation", name, INFO,
                                   f"{name} has not moved ({early:.4g} -> {late:.4g} over {len(values)} readings)",
                                   step, {"early": early, "late": late}, 0.6))
        return out

    def _check_pairs(self, histories: Dict[str, Sequence[Any]], t: Dict[str, float],
                     step: Optional[int]) -> Iterable[Finding]:
        """Checks that need two histories at once: train loss against validation loss."""
        out: List[Finding] = []
        losses = {name: _finite(list(hist)) for name, hist in histories.items() if looks_like_loss(name)}
        train = next((name for name in losses if not is_validation(name)), None)
        val = next((name for name in losses if is_validation(name)), None)
        if not train or not val:
            return out
        train_hist, val_hist = losses[train], losses[val]
        if len(train_hist) < 5 or len(val_hist) < 5:
            return out
        n = min(len(train_hist), len(val_hist), 8)
        train_window, val_window = train_hist[-n:], val_hist[-n:]
        train_change = train_window[-1] - train_window[0]
        val_change = val_window[-1] - val_window[0]
        if val_change <= 0 or val_window[0] <= 0 or train_window[0] <= 0:
            return out
        val_relative = val_change / abs(val_window[0])
        train_relative = train_change / abs(train_window[0])
        if val_relative >= t["divergence_val_frac"] and train_relative <= t["divergence_train_frac"]:
            out.append(Finding("overfitting", val, WARNING,
                               f"{val} rose {val_relative * 100:.1f}% while {train} moved "
                               f"{train_relative * 100:.1f}%: the model is memorising",
                               step, {"val_change": val_relative, "train_change": train_relative,
                                      "train_variable": train}, 0.8))
        return out

    def _check_tensor(self, name: str, stats: Dict[str, Any], step: Optional[int]) -> Optional[Finding]:
        try:
            nan = float(stats.get("nan") or stats.get("nan_count") or 0)
            inf = float(stats.get("inf") or stats.get("inf_count") or 0)
        except (TypeError, ValueError):
            return None
        if nan or inf:
            return Finding("tensor_nonfinite", name, CRITICAL,
                           f"{name} contains {int(nan)} NaN and {int(inf)} infinite values",
                           step, {"nan": nan, "inf": inf}, 1.0)
        return None


def summarise(findings: Sequence[Finding]) -> str:
    """The one-line form, for a log line or a prompt header."""
    if not findings:
        return ""
    ordered = sorted(findings, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), -f.confidence))
    return "; ".join(f.message for f in ordered)
