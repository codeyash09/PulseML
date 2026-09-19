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
_IMMEDIATE_CHECKS = frozenset({"nonfinite", "tensor_nonfinite", "lr_jump"})

LOSS_NAME_HINTS = ("loss", "cost", "nll", "cross_entropy", "crossentropy", "objective", "err")
METRIC_NAME_HINTS = ("acc", "accuracy", "f1", "auc", "precision", "recall", "iou", "dice",
                     "bleu", "rouge", "map", "mrr", "r2", "score")
NORM_NAME_HINTS = ("grad_norm", "gradnorm", "grad_scale", "weight_norm", "param_norm", "update_norm")
LR_NAME_HINTS = ("lr", "learning_rate", "learningrate", "step_size")
# Objectives that are not progress measures. In adversarial and reinforcement learning
# the loss is a moving target by construction: a generator loss rises precisely when
# the discriminator gets better, and a policy loss follows whatever the current
# advantage estimate happens to be. "It went up" and "it stopped going down" are
# meaningless for these, and the checks that say so are turned off for them -- while
# NaN, a frozen value and a hundredfold spike stay on, because those are real anywhere.
# What carries the signal in RL is the reward, and that is checked like any other score.
MOVING_TARGET_HINTS = ("policy_loss", "value_loss", "actor_loss", "critic_loss", "q_loss",
                       "td_error", "entropy_loss", "g_loss", "d_loss", "gen_loss",
                       "disc_loss", "generator_loss", "discriminator_loss", "adversarial")
# How long a step takes is not a loss, a score, a norm or a learning rate, so nothing
# looked at it -- and a run whose step time climbs all the way through is leaking, and
# usually ends as an out-of-memory kill several hours in.
TIME_NAME_HINTS = ("step_time", "batch_time", "iter_time", "epoch_time", "elapsed_per",
                   "sec_per_step", "secs_per_step", "ms_per_batch", "ms_per_step", "time_per")
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


def looks_like_moving_target(name: str) -> bool:
    """Is this an objective whose direction carries no information?"""
    low = _lower(name).replace("-", "_")
    return any(hint in low for hint in MOVING_TARGET_HINTS)


def looks_like_step_time(name: str) -> bool:
    low = _lower(name).replace("-", "_")
    return any(hint in low for hint in TIME_NAME_HINTS)


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
        "oscillation_correlation": -0.60 + 0.25 * s,    # how systematic the alternation must be
        "stagnation_frac": 0.003 + 0.077 * s,
        "stagnation_window": max(10, round(500 - 350 * s)),
        "lr_jump_ratio": 20.0 - 12.0 * s,               # was hardcoded 10x
        "never_learned_ratio": 0.995 - 0.03 * s,        # was hardcoded 0.98
        "divergence_val_frac": max(0.05, 0.003 + 0.077 * s),
        "divergence_train_frac": 0.02,
        "perfect_metric": 0.999,
        "vanishing_norm": 1e-8,
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


def _as_float(value: Any) -> Optional[float]:
    """The number this reading stands for, or None if it is not a number.

    Testing `isinstance(value, (int, float))` is the obvious way to write this and it
    is wrong for the values training loops actually produce: numpy's float32 is not a
    Python float (float64 is, by inheritance, which is what hides the bug), and neither
    is a 0-dim array or a torch scalar. A run reporting float32 -- the default dtype in
    Keras and common in PyTorch -- had every reading discarded, so nothing was ever
    checked, including whether the loss had gone NaN.

    Strings stay excluded on purpose: float("nan") succeeds, and a metric that arrived
    as text is a formatting accident, not a reading to draw conclusions from.
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        number = float(value)
    except Exception:
        # Deliberately everything: __float__ belongs to the caller's object, it can
        # raise whatever it likes, and none of it is worth taking a training run down for.
        return None
    return number


def _finite(history: Sequence[Any]) -> List[float]:
    numbers = (_as_float(v) for v in history)
    return [v for v in numbers if v is not None and math.isfinite(v)]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scaled(values: Sequence[float]) -> Tuple[List[float], float]:
    """Divide out the magnitude so that squaring cannot overflow.

    A loss of 1e300 is absurd but reachable -- one bad step of an exploding run gets
    there -- and squaring it raises OverflowError, which would propagate out of the
    detector and kill the training process it is supposed to be watching.
    """
    biggest = max((abs(v) for v in values), default=0.0)
    if biggest > 1e150:
        return [v / biggest for v in values], biggest
    return list(values), 1.0


def _stdev(values: Sequence[float]) -> float:
    """How much this variable moves around on its own, with nothing wrong."""
    if len(values) < 2:
        return 0.0
    scaled, scale = _scaled(values)
    mean = _mean(scaled)
    return (sum((v - mean) ** 2 for v in scaled) / (len(scaled) - 1)) ** 0.5 * scale


def _trend(values: Sequence[float]) -> tuple:
    """Slope per reading, its significance, and how much of the movement it explains.

    Comparing the first half of a window to the second half asks two readings' worth of
    questions of the data; a fitted slope uses all of them, which is what tells a real
    climb apart from noise that happened to land high at the end.

    The third number matters as much as the second. A significant slope only says the
    line is not flat -- fit a line to a sine wave and you get one. The fraction of the
    variance the line accounts for is what says the run is actually going one way, and
    a policy loss swinging around zero fails it.
    """
    n = len(values)
    if n < 4:
        return 0.0, 0.0, 0.0
    values, scale = _scaled(values)
    mean_x = (n - 1) / 2.0
    mean_y = _mean(values)
    sxx = sum((i - mean_x) ** 2 for i in range(n))
    sxy = sum((i - mean_x) * (values[i] - mean_y) for i in range(n))
    if sxx <= 0:
        return 0.0, 0.0, 0.0
    slope = sxy / sxx
    residuals = [values[i] - (mean_y + slope * (i - mean_x)) for i in range(n)]
    total = sum((v - mean_y) ** 2 for v in values)
    unexplained = sum(r ** 2 for r in residuals)
    explained = 1.0 - (unexplained / total) if total > 0 else 0.0
    scatter = (unexplained / (n - 2)) ** 0.5
    standard_error = scatter / (sxx ** 0.5)
    # The t-statistic and the explained fraction are scale-free; the slope is reported
    # in the caller's own units.
    if standard_error <= 0:
        return slope * scale, (99.0 if slope else 0.0), (1.0 if slope else 0.0)
    return slope * scale, slope / standard_error, explained


def _lag1(values: Sequence[float]) -> float:
    """Correlation between each reading and the next one, in [-1, 1].

    Near -1 the series alternates systematically, which is what a loss bouncing under
    too high a learning rate does. Near 0 it is noise, which also changes direction
    constantly but means nothing by it.
    """
    if len(values) < 3:
        return 0.0
    scaled, _ = _scaled(values)
    centre = _mean(scaled)
    variance = sum((v - centre) ** 2 for v in scaled)
    if variance <= 0:
        return 0.0
    return sum((scaled[i] - centre) * (scaled[i + 1] - centre)
               for i in range(len(scaled) - 1)) / variance


def _lowest_sustained(values: Sequence[float], span: int) -> float:
    """The lowest level the variable held for `span` readings running.

    Rolled rather than recomputed per window: this runs inside the training process on
    every update, and on a step-level run the history reaches tens of thousands of
    readings, where the naive form costs a second of the run's own time per call.
    """
    if span <= 0 or len(values) < span:
        return _mean(values)
    total = sum(values[:span])
    best = total
    for i in range(span, len(values)):
        total += values[i] - values[i - span]
        if total < best:
            best = total
    return best / span


def _quantile(values: Sequence[float], q: float) -> float:
    """A low quantile is a floor that one lucky epoch cannot drag down."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


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

    def __init__(self, sensitivity: float = 0.3, confirmations: int = 2, rearm_after: int = 3,
                 overrides: Optional[Dict[str, float]] = None) -> None:
        self.sensitivity = sensitivity
        # Pinning one signal without moving the whole dial: /sensitivity spike 5 sets
        # the explosion multiplier and leaves everything else deriving from the dial.
        self.overrides: Dict[str, float] = dict(overrides or {})
        self.confirmations = max(1, int(confirmations))
        self.rearm_after = max(1, int(rearm_after))
        self.active: Dict[Tuple[str, str], Finding] = {}
        self._streak: Dict[Tuple[str, str], int] = {}
        self._clear_streak: Dict[Tuple[str, str], int] = {}
        self._baselines: Dict[str, float] = {}        # agent-supplied "this is normal" anchors
        self._already_good = False                    # is a success metric already high?

    # ------------------------------------------------------------------ public

    def set_baseline(self, variable: str, value: float) -> None:
        """Tell the engine what normal looks like for a variable (the agent can know this)."""
        self._baselines[variable] = float(value)

    def update(self, histories: Dict[str, Sequence[Any]], *, step: Optional[int] = None,
               tensor_stats: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, List[Finding]]:
        """Evaluate every check against every history. Returns {"raised": [...], "cleared": [...]}."""
        t = thresholds(self.sensitivity)
        t.update({name: value for name, value in self.overrides.items() if value is not None})
        fired: Dict[Tuple[str, str], Finding] = {}
        self._already_good = self._success_metric_is_high(histories or {})

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

    @staticmethod
    def _success_metric_is_high(histories: Dict[str, Sequence[Any]]) -> bool:
        """Is the model already doing well by its own scoreboard?

        A loss that stops falling means two different things depending on the answer.
        At 55% accuracy it has stalled; at 97% it has converged, and fine-tuning runs
        live there on purpose. Only scores that live on a 0-1 scale count as an answer.
        """
        for name, history in histories.items():
            if not looks_like_metric(name):
                continue
            values = _finite(list(history))
            if values and max(values) <= 1.0 and _mean(values[-5:]) >= 0.9:
                return True
        return False

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
        raw_last = _as_float(history[-1]) if history else None
        if raw_last is not None and not math.isfinite(raw_last):
            kind = "NaN" if math.isnan(raw_last) else "infinite"
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
        # The same readings coming round again, bit for bit. Two floats from real
        # arithmetic do not repeat a whole sequence by chance, so this means the same
        # data is going through the same weights: an exhausted iterator being re-used,
        # or per-epoch state being reset. Cheap to check and impossible to fake.
        if (is_loss or is_metric) and len(values) >= 12:
            for period in range(2, min(13, len(values) // 3 + 1)):
                tail = values[-period:]
                if len(set(tail)) == 1:
                    break            # that is "frozen", and it is reported as frozen
                if tail == values[-2 * period:-period] == values[-3 * period:-2 * period]:
                    out.append(Finding("repeating", name, WARNING,
                                       f"{name} is repeating the same {period} readings exactly "
                                       f"({', '.join(f'{v:g}' for v in tail)}): the same data is "
                                       f"going through the model every time",
                                       step, {"period": period, "cycle": tail}, 0.9))
                    break

        # A metric on a small evaluation set is quantised -- 197 right out of 200 is
        # exactly 0.985 every time -- so a good model's score repeats bit-for-bit for
        # epochs on end. That is the metric having converged, not the run having died,
        # and it is only "frozen" when the model is not already doing well.
        frozen_matters = is_loss or not getattr(self, "_already_good", False)
        if frozen_matters and (is_loss or is_metric) and len(values) >= 6 and len(set(values[-6:])) == 1:
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
                else:
                    # A gradient norm that shrinks as a model converges is what success
                    # looks like, so "smaller than it was" is not a finding: comparing
                    # against a 50-reading average flagged every healthy run. What is
                    # pathological is a norm that has effectively reached zero, or one
                    # that falls off a cliff between two consecutive readings.
                    previous = values[-2]
                    vanished = latest < t["vanishing_norm"]
                    fell_off = previous > 0 and latest * t["explosion_multiplier"] < previous
                    if vanished or fell_off:
                        detail = (f"{name} has effectively reached zero ({latest:g})" if vanished
                                  else f"{name} fell from {previous:g} to {latest:g} in one reading")
                        out.append(Finding("norm_collapse", name, WARNING,
                                           f"{detail}: gradients this small stop the model learning",
                                           step, {"latest": latest, "previous": previous,
                                                  "baseline": baseline, "vanished": vanished}, 0.7))

            # A norm does not have to spike to be wrong. Weights growing steadily all
            # run -- no weight decay, or an instability building -- never trip a
            # multiple-of-recent-average test, because the average climbs with them.
            if len(values) >= 15:
                window = values[-max(15, len(values) // 2):]
                slope, slope_t, explained = _trend(window)
                growth = (window[-1] - window[0]) / (abs(window[0]) or 1.0)
                if slope > 0 and slope_t > 4.0 and explained > 0.7 and growth > 0.5:
                    out.append(Finding("norm_growth", name, WARNING,
                                       f"{name} has grown steadily all run: {window[0]:g} to "
                                       f"{window[-1]:g} over its last {len(window)} readings",
                                       step, {"from": window[0], "to": window[-1], "slope": slope},
                                       _confidence(slope_t, 4.0, len(values))))

        if looks_like_lr(name) and len(values) >= 2:
            previous = values[-2]
            if previous > 0 and latest > 0:
                ratio = latest / previous
                if ratio >= t["lr_jump_ratio"] or ratio <= 1.0 / t["lr_jump_ratio"]:
                    out.append(Finding("lr_jump", name, WARNING,
                                       f"{name} changed by {ratio:.3g}x in one step ({previous:g} -> {latest:g})",
                                       step, {"from": previous, "to": latest, "ratio": ratio}, 0.75))

        if looks_like_step_time(name) and len(values) >= 15:
            window = values[-max(15, len(values) // 2):]
            slope, slope_t, explained = _trend(window)
            growth = (window[-1] - window[0]) / (abs(window[0]) or 1.0)
            if slope > 0 and slope_t > 6.0 and explained > 0.8 and growth > 0.5:
                out.append(Finding("slowing_down", name, WARNING,
                                   f"{name} has grown {100 * growth:.0f}% over its last "
                                   f"{len(window)} readings ({window[0]:g} to {window[-1]:g}): "
                                   f"something is accumulating, and the run gets slower every step",
                                   step, {"from": window[0], "to": window[-1], "growth": growth},
                                   _confidence(slope_t, 6.0, len(values))))

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
        # For a generator or a policy loss, everything below this point -- the plateau,
        # the bouncing, the climb, the "no better than when it started" -- is normal
        # behaviour rather than evidence, so only the spike check runs.
        directionless = looks_like_moving_target(name)

        if len(values) >= 5:
            # The floor is the low end of the recent readings, not the single lowest one:
            # on a noisy run the lowest epoch is an outlier, and measuring the spike
            # against an outlier is how a noisy healthy run gets called a divergence.
            recent = values[-50:-1]
            baseline = _quantile(recent, 0.1)
            anchor = self._baselines.get(name)
            if anchor is not None:
                baseline = min(baseline, anchor)
            noise = _stdev(recent)
            big_for_this_run = latest > _mean(recent) + 4.0 * noise
            if baseline > 0 and latest > baseline * t["explosion_multiplier"] and big_for_this_run:
                out.append(Finding("loss_spike", name, CRITICAL,
                                   f"{name} spiked to {latest:g}, {latest / baseline:.1f}x its recent floor of {baseline:g}",
                                   step, {"latest": latest, "baseline": baseline},
                                   _confidence(latest / baseline, t["explosion_multiplier"], len(values))))

        if len(values) >= 8 and not directionless:
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
            # Bouncing only matters if it is going nowhere. Judge that over a horizon
            # that grows with the run -- 20 readings is a blink of a 600-step run, and
            # any descending curve looks flat if you only look at the last 20 of them.
            horizon = values[-max(len(window), len(values) // 3):]
            half = max(2, len(horizon) // 2)
            net_progress = (_mean(horizon[:half]) - _mean(horizon[-half:])) / (abs(_mean(horizon[:half])) or 1.0)
            # Counting direction changes alone finds white noise: iid noise flips about
            # two readings in three, which clears any flip threshold. A loss that is
            # really oscillating alternates *systematically* -- up, down, up, down at a
            # steady size -- and that shows up as a strongly negative lag-1 correlation,
            # where noise sits near zero.
            if (flips >= t["oscillation_flip_threshold"] and mean_step > scale * t["oscillation_delta_frac"]
                    and _lag1(window) <= t["oscillation_correlation"]
                    and net_progress < t["stagnation_frac"]):
                out.append(Finding("oscillation", name, WARNING,
                                   f"{name} is bouncing: {flips} direction changes in {len(deltas)} steps, "
                                   f"average move {mean_step:.3g}",
                                   step, {"flips": flips, "mean_step": mean_step},
                                   _confidence(flips, t["oscillation_flip_threshold"], len(values))))

        window_size = max(10, min(int(t["stagnation_window"]), len(values) // 3))
        if len(values) >= 12 and len(values) >= window_size and not directionless:
            recent = values[-window_size:]
            scale = _mean([abs(v) for v in recent]) or 1.0
            spread = (max(recent) - min(recent)) / scale
            quarter = max(2, len(recent) // 4)
            early, late = _mean(recent[:quarter]), _mean(recent[-quarter:])
            improving = (early - late) / abs(early) if abs(early) > 1e-12 else 0.0
            # How far the run got before it stalled: a run that came down 99% and
            # then sat still has converged; one that came down 30% and stopped has
            # stalled, and those must not read the same.
            opening = _mean(values[:max(2, len(values) // 5)])
            progress = (opening - late) / abs(opening) if abs(opening) > 1e-12 else 0.0
            stalled = spread < t["stagnation_frac"] and improving < t["stagnation_frac"]
            if stalled and progress < 0.9 and not getattr(self, "_already_good", False):
                out.append(Finding("stagnation", name, WARNING,
                                   f"{name} stopped improving at {late:.4g} and has barely moved for "
                                   f"{len(recent)} readings, after coming down only {progress * 100:.0f}% "
                                   f"from {opening:.4g}",
                                   step, {"early": early, "late": late, "spread": spread,
                                          "progress": progress}, 0.65))

        # Divergence: the loss is climbing. Nothing else here catches a run that is
        # steadily getting worse without ever spiking -- it does not spike, it is not
        # frozen, and it is not stagnant: it is training, in the wrong direction. The
        # rise has to be sustained and to clear the curve's own noise, so that a warmup
        # or a schedule restart lifting the loss for a few epochs does not read as one.
        if len(values) >= 15 and not directionless:
            span = max(5, len(values) // 4)
            early, late = _mean(values[:span]), _mean(values[-span:])
            rise = late - early
            slope, slope_t, explained = _trend(values[-max(15, len(values) // 2):])
            # Over less than one period a sine is a straight line, so a significant
            # slope alone flags any slowly oscillating quantity -- an RL policy loss
            # measures 6 standard errors and explains 75% of its own variance while
            # going nowhere. Real divergences measured 16 to 59, explaining over 90%.
            if (abs(early) > 0 and rise > abs(early) * t["divergence_train_frac"]
                    and slope > 0 and slope_t > 6.0 and explained > 0.8):
                severity = CRITICAL if late > early * 2 else WARNING
                out.append(Finding("divergence", name, severity,
                                   f"{name} is climbing, not falling: {early:.4g} over its first {span} "
                                   f"readings, {late:.4g} over its last {span}",
                                   step, {"early": early, "late": late, "rise": rise, "slope": slope},
                                   _confidence(slope_t, 4.0, len(values))))

        # "Never learned": the end is no better than the beginning. Every other check
        # looks for training going wrong; this one is for training that never started.
        if len(values) >= 10 and not directionless:
            span = max(2, len(values) // 5)
            start, end = _mean(values[:span]), _mean(values[-span:])
            # The best the run ever *held*, not the single luckiest reading: one noisy
            # epoch dipping below the opening is not the run having learned something,
            # and taking it for one made this check a coin flip on any noisy curve.
            best = _lowest_sustained(values, span)
            # Expressed as a distance rather than a ratio, because a ratio changes
            # direction with the sign: an ELBO or a log-likelihood sitting flat at -3.2
            # forever passed `end >= start * 0.98` trivially, so this check was blind to
            # every objective that reports a negative number.
            worth_noticing = abs(start) * (1.0 - t["never_learned_ratio"])
            never_improved = (start - best) < worth_noticing
            # "This run is not learning" is the harshest thing here, so only say it when
            # the readings are steady enough to have shown an improvement had there been
            # one. On a curve that swings by more than it has moved -- the first few
            # steps of a step-level run, a GAN, a small validation split -- no conclusion
            # is available yet, and the honest answer is to keep watching.
            se = _stdev(values[:span] + values[-span:]) * (2.0 / span) ** 0.5
            precise_enough = 3.0 * se < start * 0.10
            improved_measurably = (start - end) > 3.0 * se
            if (abs(start) > 0 and precise_enough and not improved_measurably
                    and (start - end) < worth_noticing and never_improved):
                out.append(Finding("never_learned", name, CRITICAL,
                                   f"{name} is no better than when it started ({start:.4g} -> {end:.4g} "
                                   f"over {len(values)} readings): this run is not learning",
                                   step, {"start": start, "end": end, "points": len(values)}, 0.9))

        if is_validation(name) and len(values) >= 5:
            last5 = values[-5:]
            worsening = sum(1 for a, b in zip(last5, last5[1:]) if b > a)
            # Four rises in a row happens by chance once every sixteen epochs on a noisy
            # curve, so the size of the rise has to mean something too.
            noise = _stdev(values[-20:])
            if worsening >= 4 and last5[-1] > last5[0] > 0 and (last5[-1] - last5[0]) > 2.0 * noise:
                out.append(Finding("val_regression", name, WARNING,
                                   f"{name} has worsened in {worsening} of its last 4 steps "
                                   f"({last5[0]:.4g} -> {last5[-1]:.4g})",
                                   step, {"from": last5[0], "to": last5[-1]}, 0.8))
            elif len(values) >= 8:
                half = len(values) // 2
                early, late = _mean(values[:half]), _mean(values[half:])
                # A small validation split bounces. Two means differing by less than the
                # noise in those means is not drift, it is the split being small: hold the
                # rise to three standard errors before calling it a trend.
                noise = _stdev(values) * (2.0 / half) ** 0.5 if half else 0.0
                if early > 0 and late > early * (1 + t["stagnation_frac"]) and (late - early) > 3.0 * noise:
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
        # A leak does not have to be obvious in the first two epochs: one that takes ten
        # to saturate looks like fast learning on the way up and is only visible once
        # the score sits at the ceiling and stays there. On validation, that is not a
        # model that has learned the task, it is a model that can see the answer.
        elif (is_validation(name) and len(values) >= 8
                and min(values[-8:]) >= t["perfect_metric"]):
            out.append(Finding("suspiciously_perfect", name, WARNING,
                               f"{name} has been a perfect {values[-1]:.4g} for its last 8 readings: "
                               f"a validation score that stops at the ceiling usually means the "
                               f"labels are reachable from the inputs",
                               step, {"value": values[-1], "points": len(values)}, 0.7))
        if len(values) >= 10:
            quarter = max(2, len(values) // 4)
            early, late = _mean(values[:quarter]), _mean(values[-quarter:])
            if abs(early) > 1e-12 and abs(late - early) / abs(early) < t["stagnation_frac"]:
                out.append(Finding("metric_stagnation", name, INFO,
                                   f"{name} has not moved ({early:.4g} -> {late:.4g} over {len(values)} readings)",
                                   step, {"early": early, "late": late}, 0.6))
        return out

    def _check_metric_against_loss(self, histories: Dict[str, Sequence[Any]],
                                   losses: Dict[str, List[float]], train: Optional[str],
                                   t: Dict[str, float], step: Optional[int]) -> Iterable[Finding]:
        """The loss is coming down. Is the thing you actually care about moving?

        These can come apart, and when they do the loss is the one that lies: a model
        fitting shuffled labels drives its training loss down for fifty epochs while
        accuracy sits at chance, and every check that watches the loss alone sees a
        healthy run. Reported on its own because the fix is never in the optimiser --
        it is in the labels, the metric, or what the two are computed over.
        """
        out: List[Finding] = []
        if not train or getattr(self, "_already_good", False):
            return out
        loss_values = losses[train]
        if len(loss_values) < 12:
            return out
        span = max(3, len(loss_values) // 4)
        loss_early, loss_late = _mean(loss_values[:span]), _mean(loss_values[-span:])
        if loss_early <= 0 or (loss_early - loss_late) / loss_early < 0.2:
            return out          # the loss has not really moved either; other checks own that

        for name, history in histories.items():
            if not looks_like_metric(name):
                continue
            values = _finite(list(history))
            if len(values) < 12:
                continue
            metric_span = max(3, len(values) // 4)
            early, late = _mean(values[:metric_span]), _mean(values[-metric_span:])
            scale = max(abs(early), _stdev(values), 1e-12)
            if abs(late - early) / scale < t["stagnation_frac"]:
                out.append(Finding("metric_not_improving", name, WARNING,
                                   f"{train} has come down {100 * (loss_early - loss_late) / loss_early:.0f}% "
                                   f"({loss_early:.4g} to {loss_late:.4g}) while {name} has not moved "
                                   f"({early:.4g} to {late:.4g}): the loss is improving on something "
                                   f"{name} does not measure",
                                   step, {"loss_early": loss_early, "loss_late": loss_late,
                                          "metric_early": early, "metric_late": late}, 0.8))
        return out

    def _check_pairs(self, histories: Dict[str, Sequence[Any]], t: Dict[str, float],
                     step: Optional[int]) -> Iterable[Finding]:
        """Checks that need two histories at once: a loss against validation, or a score."""
        out: List[Finding] = []
        losses = {name: _finite(list(hist)) for name, hist in histories.items() if looks_like_loss(name)}
        train = next((name for name in losses if not is_validation(name)), None)
        val = next((name for name in losses if is_validation(name)), None)

        out.extend(self._check_metric_against_loss(histories, losses, train, t, step))

        if not train or not val:
            return out
        train_hist, val_hist = losses[train], losses[val]
        if len(train_hist) < 5 or len(val_hist) < 5:
            return out
        # Compare halves rather than endpoints, and require the rise to clear the
        # validation curve's own noise. Endpoint-to-endpoint on a small validation
        # split is mostly noise: it called a healthy run "memorising" after six
        # epochs because one endpoint happened to sit high.
        n = min(len(train_hist), len(val_hist), 12)
        if n < 8:
            return out
        train_window, val_window = train_hist[-n:], val_hist[-n:]
        half = max(2, n // 2)
        val_early, val_late = _mean(val_window[:half]), _mean(val_window[-half:])
        train_early, train_late = _mean(train_window[:half]), _mean(train_window[-half:])
        if val_early <= 0 or train_early <= 0:
            return out
        val_noise = (sum((v - _mean(val_window)) ** 2 for v in val_window) / len(val_window)) ** 0.5
        if (val_late - val_early) <= 1.5 * val_noise:
            return out
        val_relative = (val_late - val_early) / abs(val_early)
        train_relative = (train_late - train_early) / abs(train_early)
        if val_relative >= t["divergence_val_frac"] and train_relative <= t["divergence_train_frac"]:
            out.append(Finding("overfitting", val, WARNING,
                               f"{val} rose {val_relative * 100:.1f}% (beyond its own noise) while {train} moved "
                               f"{train_relative * 100:.1f}%: the model is memorising",
                               step, {"val_change": val_relative, "train_change": train_relative,
                                      "train_variable": train}, 0.8))
        return out

    def _check_tensor(self, name: str, stats: Dict[str, Any], step: Optional[int]) -> Optional[Finding]:
        if not isinstance(stats, dict):
            return None
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
