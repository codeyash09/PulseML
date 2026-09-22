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
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

_SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}

# How much history a check ever looks at. Detection is about what the run is doing now,
# and every check here works on a recent window or on early-versus-late halves of one.
# Unbounded, a run of 50k steps made the detector re-read 50k values per variable per
# update forever, which is cost paid on the training thread for evidence no check uses.
# Deep enough that "early" still means early on any realistic run.
ANALYSIS_WINDOW = 1500

# Checks that skip the confirmation wait, because waiting is itself the damage: once a
# value is NaN every subsequent step is wasted compute, and no second opinion is going
# to make it finite again. Everything else has to hold for `confirmations` rounds --
# a loss spike can be one bad batch, and firing on it was a reliable false alarm.
# Checks that report a *transient*: the thing they describe is over by the next reading,
# so requiring two consecutive confirmations means they can never fire at all. A
# benchmark of single-epoch spikes caught 0/8 at the default two confirmations and 4/8 at
# one, for exactly this reason -- a spike is gone before it can be confirmed.
_IMMEDIATE_CHECKS = frozenset({"nonfinite", "tensor_nonfinite", "lr_jump", "loss_spike",
                               "tensor_dtype_change", "tensor_device_change"})

LOSS_NAME_HINTS = ("loss", "cost", "nll", "cross_entropy", "crossentropy", "objective", "err",
                   "kl", "kld", "divergence", "ppl", "perplexity", "mse", "mae", "rmse")
METRIC_NAME_HINTS = ("acc", "accuracy", "f1", "auc", "precision", "recall", "iou", "dice",
                     "bleu", "rouge", "map", "mrr", "r2", "score")
NORM_NAME_HINTS = ("grad_norm", "gradnorm", "grad_scale", "weight_norm", "param_norm",
                   "update_norm", "momentum_norm", "_norm", "logit_max", "logit_norm",
                   "activation_norm", "emb_norm")
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

# ---------------------------------------------------------------------------------------
# What a variable *is*, beyond loss-or-metric.
#
# Every check used to key off two questions: does this look like a loss, does it look
# like a score. That covers the run's own arithmetic and nothing else, which is why a
# benchmark of 95 failure families found 35 of them invisible: a GPU at 2% utilisation,
# a router that collapsed onto one expert, a checkpoint counter that stopped moving, a
# rank training its own copy of the model. None of those show up in the loss, and all of
# them are logged by the runs they happen to.
#
# A role says which direction is bad and what "bad" means for that kind of number, so
# one check can serve every variable that shares a role.
# ---------------------------------------------------------------------------------------

# Higher is better: falling is the fault. Extends the metric hints with the names RL and
# generation runs use for the same idea.
SCORE_NAME_HINTS = ("reward", "return", "win_rate", "success_rate", "pass_rate", "bleu",
                    "rouge", "meteor", "ndcg", "map@", "mrr", "score", "hit_rate")

# Collapse towards zero is the fault: a distribution that stopped being a distribution.
ENTROPY_NAME_HINTS = ("entropy", "perplexity_of_router", "router_prob_std", "diversity")

# Should sit near zero. These are the "how much of the work is being wasted" numbers.
WASTE_NAME_HINTS = ("oov", "unk_rate", "padding_frac", "pad_frac", "dropped", "truncat",
                    "error_rate", "err_rate", "clip_fraction", "clipped_frac", "skip_rate",
                    "overflow", "collision", "retry", "timeout_rate", "miss_rate",
                    "invalid_frac", "reject")

# Should stay high. Falling means the hardware is idle or the pipeline is starved.
UTILISATION_NAME_HINTS = ("util", "occupancy", "efficiency", "mfu", "hfu", "sm_active")

# Work per unit time. Falling is the fault.
THROUGHPUT_NAME_HINTS = ("per_sec", "per_second", "_persec", "throughput", "samples_s",
                         "tokens_s", "it_s", "ips", "qps", "fps", "steps_s")

# Growth is the fault: something is accumulating.
MEMORY_NAME_HINTS = ("mem_", "memory", "vram", "rss", "allocated", "reserved", "heap")

# Monotonic counters. Standing still is the fault.
COUNTER_NAME_HINTS = ("_written", "_saved", "_count", "num_saved", "checkpoints",
                      "files_written", "_committed")

# Hardware health.
HARDWARE_TEMP_HINTS = ("gpu_temp", "temp_c", "_temp_", "temperature_c", "tempc")
CLOCK_NAME_HINTS = ("clock", "_mhz", "sm_clock", "freq_mhz")

# A learned softmax temperature, which is not a thermometer. Collapse is the fault.
SOFT_TEMPERATURE_HINTS = ("temperature", "logit_scale", "tau")

# Names whose value has a known meaning, used to spot a run sitting at chance level.
CLASS_COUNT_HINTS = ("num_classes", "n_classes", "vocab_size", "num_labels", "n_labels")


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


def _has(name: str, hints: Tuple[str, ...]) -> bool:
    lowered = _lower(name)
    return any(hint in lowered for hint in hints)


def looks_like_score(name: str) -> bool:
    """Higher is better. A fall is the fault, which is the opposite of a loss."""
    return looks_like_metric(name) or _has(name, SCORE_NAME_HINTS)


def looks_like_entropy(name: str) -> bool:
    return _has(name, ENTROPY_NAME_HINTS)


def looks_like_waste(name: str) -> bool:
    """A fraction of the work being thrown away: padding, OOV, dropped tokens."""
    return _has(name, WASTE_NAME_HINTS)


def looks_like_utilisation(name: str) -> bool:
    return _has(name, UTILISATION_NAME_HINTS)


def looks_like_throughput(name: str) -> bool:
    return _has(name, THROUGHPUT_NAME_HINTS)


def looks_like_memory(name: str) -> bool:
    return _has(name, MEMORY_NAME_HINTS)


def looks_like_counter(name: str) -> bool:
    return _has(name, COUNTER_NAME_HINTS)


def looks_like_hardware_temp(name: str) -> bool:
    return _has(name, HARDWARE_TEMP_HINTS)


def looks_like_clock(name: str) -> bool:
    return _has(name, CLOCK_NAME_HINTS)


def looks_like_soft_temperature(name: str) -> bool:
    """A learned softmax temperature, not a thermometer: `temperature`, `logit_scale`."""
    return _has(name, SOFT_TEMPERATURE_HINTS) and not looks_like_hardware_temp(name)


def looks_like_class_count(name: str) -> bool:
    return _has(name, CLASS_COUNT_HINTS)


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
        # --- roles beyond loss-and-score -------------------------------------------
        "score_drop_frac": 0.10 - 0.075 * s,        # fall from peak that counts as a regression
        "entropy_collapse_frac": 0.10 + 0.15 * s,   # fraction of its own early value
        "waste_fraction": 0.50 - 0.35 * s,          # padding/OOV/dropped share that is too much
        "utilisation_floor": 0.50 + 0.30 * s,       # fraction of its own early value
        "throughput_drop_frac": 0.50 + 0.30 * s,    # fraction of its own early value
        "memory_growth_frac": 1.50 - 1.10 * s,      # relative growth over the run
        "counter_stall_readings": round(20 - 12 * s),
        "group_spread_frac": 0.30 - 0.22 * s,       # disagreement across ranks/shards
        "pair_drift_ratio": 5.0 - 3.5 * s,          # two series that tracked each other
        "identical_tolerance": 1e-9,
        "relation_tolerance": 0.10 - 0.07 * s,      # perplexity vs exp(loss), etc.
        "component_share_floor": 0.15 - 0.08 * s,   # share of the objective a term has left
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
    """The finite numbers in a history, in order.

    The common case by far is a list that is already Python floats, and for that the
    general path -- four isinstance checks and a float() per reading -- is most of what
    the detector costs on a long run. `type(v) is float` is an exact identity test and
    cheap, so the fast path takes it and anything else falls through to _as_float,
    which still handles numpy scalars, torch scalars and 0-dim arrays.
    """
    out: List[float] = []
    isfinite = math.isfinite
    for value in history:
        if type(value) is float:
            if isfinite(value):
                out.append(value)
            continue
        number = _as_float(value)
        if number is not None and isfinite(number):
            out.append(number)
    return out


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
        self._run_is_moving = True                    # is anything still changing?
        self._lr_restarts = False                     # a warm restart, not a regression
        self._tensor_seen: Dict[str, Dict[str, Any]] = {}   # last dtype/device per tensor

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
        histories = histories or {}
        # Convert once. Every check used to call _finite on the whole history for
        # itself, so a run logging 100 variables over 20k steps re-parsed four million
        # values on every update -- most of the detector's cost, and it lands on the
        # training thread in cli mode.
        finite = {name: _finite(list(history)[-ANALYSIS_WINDOW:])
                  for name, history in histories.items() if history}
        self._already_good = self._success_metric_is_high(finite)
        self._run_is_moving = self._something_is_still_changing(finite)
        self._lr_restarts = self._learning_rate_restarts(finite)

        for name, history in histories.items():
            if not history:
                continue
            for finding in self._check_variable(name, list(history)[-ANALYSIS_WINDOW:], t,
                                                step, finite.get(name) or []):
                fired[finding.key] = finding
        for name, stats in (tensor_stats or {}).items():
            finding = self._check_tensor(name, stats, step)
            if finding is not None:
                fired[finding.key] = finding
        for finding in self._check_pairs(finite, t, step):
            fired[finding.key] = finding
        for finding in self._check_relations(finite, histories, t, step):
            fired[finding.key] = finding

        return self._settle(fired)

    @staticmethod
    def _learning_rate_restarts(finite: Dict[str, List[float]]) -> bool:
        """Does this run's learning-rate schedule go back up?

        A warm restart raises the LR and the loss follows it up, every cycle, on
        purpose. Without this, the loss going back above its best reads as a run that
        has lost its progress -- true of a resume that dropped the optimizer state,
        false of SGDR, and identical in the loss alone.

        Asked of the whole schedule rather than the last few readings: the loss stays
        elevated for most of a cycle, which is far longer than the restart itself, so a
        short lookback has already forgotten the restart by the time the loss is at its
        highest.
        """
        for name, values in finite.items():
            if not looks_like_lr(name) or len(values) < 3:
                continue
            for before, after in zip(values, values[1:]):
                if before > 0 and after > before * 2.0:
                    return True
        return False

    @staticmethod
    def _something_is_still_changing(histories: Dict[str, Sequence[Any]]) -> bool:
        """Is the run producing new numbers, or has everything gone still?

        A counter that stops incrementing matters only while the rest of the run is
        moving. On a finished run every counter has stopped, and saying so is noise.
        """
        for name, values in histories.items():
            if not (looks_like_loss(name) or looks_like_metric(name)):
                continue
            tail = list(values)[-6:]
            if len(tail) >= 3 and len(set(tail)) > 1:
                return True
        return False

    @staticmethod
    def _success_metric_is_high(histories: Dict[str, Sequence[Any]]) -> bool:
        """Is the model already doing well by its own scoreboard?

        A loss that stops falling means two different things depending on the answer.
        At 55% accuracy it has stalled; at 97% it has converged, and fine-tuning runs
        live there on purpose. Only scores that live on a 0-1 scale count as an answer.
        """
        for name, values in histories.items():
            if not looks_like_metric(name):
                continue
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
                        step: Optional[int], values: List[float]) -> Iterable[Finding]:
        out: List[Finding] = []
        raw_last = _as_float(history[-1]) if history else None
        if raw_last is not None and not math.isfinite(raw_last):
            kind = "NaN" if math.isnan(raw_last) else "infinite"
            out.append(Finding("nonfinite", name, CRITICAL,
                               f"{name} is {kind}; every step from here is wasted compute",
                               step, {"value": str(raw_last), "points": len(history)}, 1.0))
            return out              # nothing else about this variable is meaningful now

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
            if baseline <= 0 and max(values) <= t["vanishing_norm"]:
                # Zero from the very first reading, so there is no baseline to fall
                # from. Everything below is written as "it used to be bigger", which
                # meant a norm that was never anything but zero -- optimizer state
                # being wiped every step -- fell through every branch and was silent.
                out.append(Finding("norm_collapse", name, WARNING,
                                   f"{name} has been {latest:g} for the whole run: this is not a "
                                   f"norm that collapsed, it is one that was never anything else",
                                   step, {"latest": latest, "from_start": True}, 0.8))
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
                    # A warm restart returns the LR to a height it has already been at:
                    # SGDR decays to near zero and jumps back to its peak, over and
                    # over, and calling that a fault every cycle is how a detector gets
                    # switched off. A schedule that has been here before is a schedule;
                    # a jump to a value the run has never used is a bug.
                    seen_before = any(abs(v - latest) <= 0.1 * latest for v in values[:-2])
                    if not seen_before:
                        out.append(Finding("lr_jump", name, WARNING,
                                           f"{name} changed by {ratio:.3g}x in one step "
                                           f"({previous:g} -> {latest:g})",
                                           step, {"from": previous, "to": latest,
                                                  "ratio": ratio}, 0.75))

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

        out.extend(self._check_role(name, values, t, step))

        if is_loss:
            out.extend(self._check_loss(name, values, t, step))
        elif is_metric:
            out.extend(self._check_metric(name, values, t, step))
        return out

    # ------------------------------------------------------------------ roles

    def _check_role(self, name: str, values: List[float], t: Dict[str, float],
                    step: Optional[int]) -> Iterable[Finding]:
        """Checks that follow from what kind of number this is.

        One check per role rather than per variable: whatever a run calls its GPU
        utilisation, a utilisation that has halved means the same thing.
        """
        out: List[Finding] = []
        if len(values) < 8:
            return out
        latest = values[-1]
        span = max(3, len(values) // 4)
        early, late = _mean(values[:span]), _mean(values[-span:])
        recent = values[-span:]
        is_loss = looks_like_loss(name)
        is_score = looks_like_score(name) and not is_loss

        # A score that peaked and came back down. The existing val_regression tests
        # `b > a`, which is right for a loss and backwards for an accuracy: a validation
        # accuracy falling from 0.94 to 0.34 raised nothing at all.
        if is_score:
            peak = max(values)
            trough = _mean(values[-max(2, span // 2):])
            noise = _stdev(values[-20:])
            fallen = peak - trough
            if (peak > 0 and fallen > max(t["score_drop_frac"] * abs(peak), 3.0 * noise)
                    and values.index(peak) < len(values) - 2):
                severity = CRITICAL if fallen > 0.25 * abs(peak) else WARNING
                out.append(Finding("score_regression", name, severity,
                                   f"{name} peaked at {peak:.4g} and has fallen to {trough:.4g} "
                                   f"({100 * fallen / abs(peak):.0f}% of its best): higher is better "
                                   f"for this one, so it is getting worse",
                                   step, {"peak": peak, "now": trough, "fallen": fallen},
                                   _confidence(fallen / max(noise, 1e-12), 3.0, len(values))))

        # A distribution that stopped being one: routing onto a single expert, attention
        # onto a single token, a softmax that saturated.
        if looks_like_entropy(name) and early > 0:
            if late < early * t["entropy_collapse_frac"]:
                out.append(Finding("entropy_collapse", name, WARNING,
                                   f"{name} has collapsed from {early:.4g} to {late:.4g}: whatever "
                                   f"this distributes over, it is now going to one place",
                                   step, {"early": early, "late": late}, 0.8))

        # A learned temperature going to zero takes the loss with it, and the loss looks
        # wonderful on the way down.
        if looks_like_soft_temperature(name) and early > 0 and late < early * 0.2:
            out.append(Finding("temperature_collapse", name, WARNING,
                               f"{name} collapsed from {early:.4g} to {late:.4g}: the softmax it "
                               f"scales is becoming a hard argmax, which makes the loss look better "
                               f"than the model is",
                               step, {"early": early, "late": late}, 0.75))

        # The share of the work being thrown away. Both the climb and the level matter:
        # a padding fraction that has been a steady 50% all run was never "rising", and
        # half the compute has still gone nowhere.
        if looks_like_waste(name):
            share = _mean(recent)
            if share > t["waste_fraction"] and share > early * 1.5:
                out.append(Finding("waste_rising", name, WARNING,
                                   f"{name} has risen to {share:.3g} (from {early:.3g}): that share "
                                   f"of every batch is not training the model",
                                   step, {"early": early, "now": share}, 0.75))
            elif share > t["waste_fraction"] and len(values) >= 12:
                out.append(Finding("waste_high", name, WARNING,
                                   f"{name} has been {share:.3g} for the whole run: that share of "
                                   f"every batch is being thrown away, steadily",
                                   step, {"share": share}, 0.7))
            elif share >= 0.98 and _stdev(recent) < 1e-9:
                # Pinned at its ceiling every step -- a clip that is always active is not
                # protecting the run, it is setting the step size.
                out.append(Finding("waste_pinned", name, WARNING,
                                   f"{name} is pinned at {share:.3g} on every reading: it is not "
                                   f"an exception any more, it is the normal path",
                                   step, {"value": share}, 0.8))

        # Hardware that is idle, and a pipeline that has stopped feeding it.
        if looks_like_utilisation(name) and early > 0 and late < early * t["utilisation_floor"]:
            out.append(Finding("utilisation_drop", name, WARNING,
                               f"{name} fell from {early:.3g} to {late:.3g}: the accelerator is "
                               f"waiting for something, and the run is paying for it either way",
                               step, {"early": early, "late": late}, 0.75))

        if looks_like_throughput(name) and early > 0:
            if late <= 0:
                out.append(Finding("throughput_stopped", name, CRITICAL,
                                   f"{name} has reached zero: the run is not making progress",
                                   step, {"early": early}, 0.95))
            elif late < early * t["throughput_drop_frac"]:
                out.append(Finding("throughput_drop", name, WARNING,
                                   f"{name} fell from {early:.4g} to {late:.4g} "
                                   f"({100 * (1 - late / early):.0f}% slower than it started)",
                                   step, {"early": early, "late": late}, 0.75))

        # Something accumulating. Distinct from step_time's slowing_down: memory can
        # climb for a long time before it shows up as latency, and then it is an OOM.
        if looks_like_memory(name) and len(values) >= 15 and early > 0:
            window = values[-max(15, len(values) // 2):]
            slope, slope_t, explained = _trend(window)
            growth = (late - early) / abs(early)
            if slope > 0 and slope_t > 4.0 and explained > 0.7 and growth > t["memory_growth_frac"]:
                out.append(Finding("memory_growth", name, WARNING,
                                   f"{name} has grown {100 * growth:.0f}% ({early:.4g} to {late:.4g}) "
                                   f"and is still climbing: this run ends in an OOM",
                                   step, {"early": early, "late": late, "growth": growth},
                                   _confidence(slope_t, 4.0, len(values))))

        # A counter that stopped counting. Forty hours of training and nothing on disk.
        if looks_like_counter(name):
            stall = int(t["counter_stall_readings"])
            if len(values) >= stall and len(set(values[-stall:])) == 1 and self._run_is_moving:
                out.append(Finding("counter_stalled", name, WARNING,
                                   f"{name} has not moved from {latest:g} in {stall} readings while "
                                   f"the run kept going: whatever it counts has stopped happening",
                                   step, {"value": latest, "readings": stall}, 0.8))

        if looks_like_hardware_temp(name) and late > 85.0 and late > early:
            out.append(Finding("thermal_risk", name, WARNING,
                               f"{name} has reached {late:.1f} and is still climbing: cards throttle "
                               f"before they fail, so this costs throughput first",
                               step, {"early": early, "late": late}, 0.7))

        if looks_like_clock(name) and early > 0 and late < early * 0.9:
            out.append(Finding("clock_throttled", name, WARNING,
                               f"{name} dropped from {early:.4g} to {late:.4g}: the hardware is "
                               f"running slower than it can",
                               step, {"early": early, "late": late}, 0.7))

        # A ratio that is supposed to stay near one, climbing.
        if _has(name, ("_ratio", "imbalance", "skew")) and early > 0 and late > early * 3.0:
            out.append(Finding("imbalance_growing", name, WARNING,
                               f"{name} has grown from {early:.4g} to {late:.4g}: whatever this "
                               f"compares, one side is running away from the other",
                               step, {"early": early, "late": late}, 0.7))

        # The number is being rounded somewhere it should not be -- an fp16 accumulator,
        # a metric averaged in the wrong dtype. Tested as a *grid*: every value a
        # multiple of one step. Counting distinct values instead only caught the
        # coarsest cases, because a fine grid over a wide range still produces a
        # different value almost every reading while quantising away everything below
        # the step.
        if is_loss and len(values) >= 20:
            # Losses only. A score computed as k-correct-out-of-N lands on a grid by
            # arithmetic, not by anything going wrong.
            finding = self._check_quantisation(name, values, step)
            if finding is not None:
                out.append(finding)

        # The run is worse than its own best and has stayed there. Whatever caused it --
        # a resume that lost the optimizer state, a schedule that restarted, a batch
        # that blew the weights out -- progress that was already made has been given
        # back, and a threshold on "how fast is it getting worse" does not see that.
        if (is_loss and len(values) >= 12 and not looks_like_moving_target(name)
                and not self._lr_restarts):
            window = max(3, len(values) // 5)
            history = values[:-window] or values
            best = _lowest_sustained(history, min(len(history), max(3, window // 2)))
            # The median, not the mean: one bad batch pulls an average above the bar on
            # its own, and a single spike is `loss_spike`, which already reports it.
            recent_window = sorted(values[-window:])
            now = recent_window[len(recent_window) // 2]
            # Noise as step-to-step movement, not as the spread of the levels. On a
            # loss that is coming down steeply the spread is the descent itself, so a
            # standard deviation reads as huge noise and nothing ever clears it. And
            # measured before the event, since a spike inflates the very number used to
            # decide whether the spike is worth mentioning.
            tail = history[-20:]
            deltas = [abs(b - a) for a, b in zip(tail, tail[1:])]
            noise = _mean(deltas) if deltas else 0.0
            # Still elevated *and* not recovering. A warm restart and a curriculum step
            # both jump the loss and then dive below where they were, and neither is a
            # run that lost its progress. That also means a resume which dropped the
            # optimizer state is not reported here while it is recovering: in the loss
            # alone the two are the same shape, and a false alarm on two schedules
            # everybody uses costs more than a miss on one that recovers by itself.
            tail_slope = _trend(values[-window:])[0] if window >= 3 else 0.0
            recovering = tail_slope < -0.02 * abs(now or 1.0)
            # Every reading in the window has to be up there. A window straddling a
            # curriculum boundary or a restart is half old curve and half new one, and
            # its average sits above the old best without the run ever having settled
            # at a worse level -- which is where this fired on two healthy schedules.
            elevated = all(v > best * 1.25 for v in values[-window:])
            if (best > 0 and now > best * 1.25 and (now - best) > 3.0 * noise
                    and elevated and not recovering):
                out.append(Finding("progress_lost", name, WARNING,
                                   f"{name} reached {best:.4g} earlier in the run and is now "
                                   f"{now:.4g}: the run has given back progress it had already "
                                   f"made, which no rate-of-change check will show",
                                   step, {"best": best, "now": now}, 0.8))

        # Periodic, but not step-to-step. The oscillation check looks for alternation
        # between consecutive readings; data sorted by class and never shuffled gives a
        # clean cycle with the period of an epoch, and sails straight past it.
        if (is_loss or is_score) and len(values) >= 24 and not looks_like_moving_target(name):
            # A generator or policy loss cycles because of what it is: the thing it is
            # measured against is moving too.
            finding = self._check_periodicity(name, values, step)
            if finding is not None:
                out.append(finding)
        return out

    @staticmethod
    def _check_quantisation(name: str, values: List[float], step: Optional[int]) -> Optional[Finding]:
        """Every reading a multiple of one step: the number is on a grid.

        Real arithmetic does not land on a grid. When a loss does, something has rounded
        it -- an fp16 accumulator, a metric summed in a dtype that cannot hold it -- and
        everything finer than the step has been thrown away, including the improvements
        the run is trying to make.
        """
        distinct = sorted(set(values))
        if len(distinct) < 8:
            return None                       # too few to tell a grid from a short run
        # A monotone series is a line, and every line lands on a grid: the differences
        # between consecutive readings are all equal by construction. Quantisation is
        # only visible when there is something for the grid to round *away*.
        rises = sum(1 for a, b in zip(values, values[1:]) if b > a)
        falls = sum(1 for a, b in zip(values, values[1:]) if b < a)
        if min(rises, falls) < max(3, len(values) // 20):
            return None
        gaps = [b - a for a, b in zip(distinct, distinct[1:]) if b - a > 0]
        if len(gaps) < 6:
            return None
        # The grid step is the greatest common divisor of the gaps, not the smallest
        # gap: two readings a grid apart may simply never occur, and then the smallest
        # gap observed is a multiple of the real step and every other gap looks off it.
        step_size = gaps[0]
        for gap in gaps[1:]:
            a, b = step_size, gap
            for _ in range(48):
                if b < a:
                    a, b = b, a
                if a <= 0:
                    break
                remainder = b - a * math.floor(b / a)
                if remainder <= a * 1e-6:
                    break
                b = remainder
            step_size = a
            if step_size <= 0:
                return None
        spread = distinct[-1] - distinct[0]
        if step_size <= 0 or spread <= 0 or step_size < spread * 1e-9:
            return None
        # Every gap an integer multiple of the smallest one, within a tolerance well
        # below the step itself.
        tolerance = step_size * 0.02
        for gap in gaps:
            if abs(gap / step_size - round(gap / step_size)) * step_size > tolerance:
                return None
        if spread / step_size > len(values) * 40:
            return None                       # so fine it is just float resolution
        return Finding("quantised", name, WARNING,
                       f"{name} only ever lands on multiples of {step_size:g}: it is being "
                       f"rounded to a grid, so every improvement smaller than that is being "
                       f"thrown away before you see it",
                       step, {"step": step_size, "levels": len(distinct)}, 0.75)

    @staticmethod
    def _check_periodicity(name: str, values: List[float], step: Optional[int]) -> Optional[Finding]:
        """A cycle at a lag longer than one reading, found by autocorrelation.

        On the residuals after removing the trend, not on the values. A loss that is
        simply going down correlates with itself at every lag -- run against the raw
        series this fired on every healthy curve in the suite, which is the classic way
        to get autocorrelation wrong.
        """
        window = values[-48:]
        n = len(window)
        # High-pass rather than a straight line: subtracting a fitted line from an
        # exponential decay leaves a smooth bow, and a smooth bow autocorrelates at
        # short lags just as a cycle does. A centred moving average removes the shape
        # of the curve whatever that shape is, and leaves only what wiggles.
        half = 3
        residuals = []
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            residuals.append(window[i] - _mean(window[lo:hi]))

        denominator = sum(r * r for r in residuals)
        if denominator <= 0:
            return None
        scale = (denominator / n) ** 0.5
        if scale <= abs(_mean(window)) * 1e-6:
            return None                      # residuals are numerical dust

        def correlation(lag: int) -> float:
            pairs = list(zip(residuals, residuals[lag:]))
            if len(pairs) < 8:
                return 0.0
            return (sum(a * b for a, b in pairs) / len(pairs)) / (denominator / n)

        if correlation(2) > 0.45:
            return None      # step-to-step alternation: that is `oscillation`, not this

        # At most a quarter of the window, so the cycle is seen at least four times. A
        # lag of a third gives three repetitions, and three is not enough to tell a
        # cycle from a noisy curve that happened to line up once.
        scores = {lag: correlation(lag) for lag in range(3, max(4, n // 4) + 1)}
        if not scores:
            return None
        best_lag = max(scores, key=lambda lag: scores[lag])
        best_score = scores[best_lag]

        # The peak has to stand out from the other lags. Testing a dozen lags on a noisy
        # curve will always find one that correlates well; what a real cycle has and
        # noise does not is a single sharp peak rather than a field of similar values.
        others = [s for lag, s in scores.items() if abs(lag - best_lag) > 1]
        if others:
            spread = _stdev(others)
            if spread > 0 and best_score < _mean(others) + 2.5 * spread:
                return None
        # A real cycle comes back in phase at its period, is out of phase at half of it,
        # and comes back again at twice it. Noise that happens to correlate once does
        # none of those, and testing a dozen lags on a noisy curve will always turn up
        # one that correlates by chance -- requiring all three is what separates
        # unshuffled data from an ordinary noisy run.
        anti = correlation(best_lag // 2) if best_lag >= 6 else -1.0
        repeat = correlation(best_lag * 2) if best_lag * 2 <= n - 8 else best_score
        if best_lag and best_score > 0.55 and anti < -0.15 and repeat > 0.25:
            return Finding("periodic", name, WARNING,
                           f"{name} repeats on a cycle of {best_lag} readings "
                           f"(autocorrelation {best_score:.2f} after removing the trend): "
                           f"something in the input order is coming round again, so the "
                           f"batches are not being shuffled",
                           step, {"lag": best_lag, "correlation": best_score}, 0.7)
        return None

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
            values = list(history)
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
        losses = {name: list(hist) for name, hist in histories.items() if looks_like_loss(name)}
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

    # ------------------------------------------------------------------ relations

    # Numbered names that are copies of one quantity across parallel workers. Only these
    # are compared against each other: `metric_0 ... metric_39` are forty *different*
    # numbers that happen to be numbered, and treating their spread as disagreement
    # flagged a perfectly healthy run that logged a lot of metrics.
    SHARD_TOKENS = ("rank", "shard", "worker", "gpu", "node", "replica", "device",
                    "proc", "expert", "partition", "run", "seed", "fold")

    @classmethod
    def _group_key(cls, name: str) -> Optional[str]:
        """`loss_rank3` -> `loss_rank`, `expert_2_frac` -> `expert_frac`.

        Runs that shard something log it once per shard under a numbered name. Those
        series are supposed to be the same number, so their *disagreement* is a signal
        no single one of them carries.
        """
        lowered = _lower(name)
        if not any(token in lowered for token in cls.SHARD_TOKENS):
            return None
        stripped = re.sub(r"(?:^|(?<=[_\-/]))(\d+)(?=$|[_\-/])", "", lowered)
        stripped = re.sub(r"\d+$", "", stripped)
        stripped = re.sub(r"[_\-/]{2,}", "_", stripped).strip("_-/")
        return stripped if stripped and stripped != lowered else None

    def _check_relations(self, finite: Dict[str, List[float]],
                         histories: Dict[str, Sequence[Any]], t: Dict[str, float],
                         step: Optional[int]) -> Iterable[Finding]:
        """Checks that need more than one variable to mean anything.

        This is the gap a 95-family benchmark found: the checks were univariate almost
        without exception, and most of what goes wrong in a real run is a relationship
        coming apart. A rank training its own copy of the model, an EMA drifting from
        the weights it shadows, a reward climbing while the episodes it scores collapse
        -- every individual curve in those is unremarkable.
        """
        out: List[Finding] = []
        series = {name: values for name, values in finite.items() if len(values) >= 8}
        if len(series) < 2:
            return out

        # --- shards that are supposed to agree ------------------------------------
        groups: Dict[str, List[str]] = {}
        for name in series:
            key = self._group_key(name)
            if key:
                groups.setdefault(key, []).append(name)
        for key, members in groups.items():
            if len(members) < 2:
                continue
            latest = {name: series[name][-1] for name in members}
            values = list(latest.values())
            centre = _mean(values)
            spread = max(values) - min(values)
            if abs(centre) < 1e-12:
                continue
            head = max(3, min(len(series[m]) for m in members) // 4)
            first = [_mean(series[m][:head]) for m in members]
            early_spread = max(first) - min(first)
            early_centre = _mean(first)

            # They agreed and no longer do: the thing that was keeping them in step --
            # an all-reduce, a shared sampler -- has stopped doing it.
            drifted = (early_spread <= abs(early_centre or 1.0) * 0.05
                       and spread > abs(centre) * t["group_spread_frac"])
            if drifted:
                worst = max(latest, key=lambda n: abs(latest[n] - centre))
                out.append(Finding("group_diverged", key, CRITICAL,
                                   f"{len(members)} series under {key} started together and are now "
                                   f"{spread:.4g} apart ({min(values):.4g} to {max(values):.4g}), "
                                   f"furthest is {worst}: they are no longer the same number",
                                   step, {"members": sorted(members), "spread": spread,
                                          "early_spread": early_spread, "worst": worst},
                                   _confidence(spread / abs(centre), t["group_spread_frac"],
                                               len(members))))
                continue

            # One of them has always been the odd one out: a straggler holding every
            # other worker at the barrier, or an expert taking all the tokens.
            if len(members) >= 3 and spread > abs(centre) * t["group_spread_frac"]:
                worst = max(latest, key=lambda n: abs(latest[n] - centre))
                rest = [v for n, v in latest.items() if n != worst]
                rest_mean = _mean(rest)
                if abs(rest_mean) > 1e-12 and abs(latest[worst] - rest_mean) > abs(rest_mean) * 0.5:
                    out.append(Finding("group_outlier", key, WARNING,
                                       f"{worst} is {latest[worst]:.4g} while the other "
                                       f"{len(rest)} under {key} average {rest_mean:.4g}: one "
                                       f"worker out of step holds all of them up",
                                       step, {"members": sorted(members), "worst": worst,
                                              "value": latest[worst], "others": rest_mean}, 0.8))

        # --- two series that used to track each other -----------------------------
        for a, b in self._twins(series):
            first, second = series[a], series[b]
            n = min(len(first), len(second))
            if n < 10:
                continue
            head = max(3, n // 4)
            early_gap = abs(_mean(first[:head]) - _mean(second[:head]))
            late_gap = abs(_mean(first[-head:]) - _mean(second[-head:]))
            scale = max(abs(_mean(first[-head:])), abs(_mean(second[-head:])), 1e-12)
            if late_gap <= scale * 0.15:
                continue
            if late_gap > max(early_gap * t["pair_drift_ratio"], scale * 0.25):
                out.append(Finding("pair_drift", f"{a} vs {b}", WARNING,
                                   f"{a} and {b} started {early_gap:.4g} apart and are now "
                                   f"{late_gap:.4g} apart: they track each other when things are "
                                   f"working, and they have come apart",
                                   step, {"pair": [a, b], "early_gap": early_gap,
                                          "late_gap": late_gap}, 0.7))

        # --- two series that are suspiciously the same ----------------------------
        names = sorted(series)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if not (is_validation(a) != is_validation(b)):
                    continue            # only train-vs-val is interesting here
                first, second = series[a], series[b]
                n = min(len(first), len(second))
                if n < 10:
                    continue
                if all(abs(x - y) <= t["identical_tolerance"]
                       for x, y in zip(first[-n:], second[-n:])):
                    out.append(Finding("identical_series", f"{a} vs {b}", WARNING,
                                       f"{a} and {b} are identical to within {t['identical_tolerance']:g} "
                                       f"over {n} readings: validation is being computed on the "
                                       f"training data",
                                       step, {"pair": [a, b], "readings": n}, 0.9))

        out.extend(self._check_known_relations(series, t, step))
        out.extend(self._check_objective_gamed(series, t, step))
        out.extend(self._check_widening_gap(series, t, step))
        out.extend(self._check_chance_level(series, histories, t, step))
        out.extend(self._check_component_collapse(series, t, step))
        out.extend(self._check_eval_noise(series, t, step))
        out.extend(self._check_hyperparameters(series, step))
        return out

    @staticmethod
    def _check_component_collapse(series: Dict[str, List[float]], t: Dict[str, float],
                                  step: Optional[int]) -> Iterable[Finding]:
        """One term of a composite loss going to zero while the rest carries the run.

        A KL term that collapses is the VAE ignoring its latent; an auxiliary term that
        vanishes is a regularizer that stopped regularizing. The total keeps falling, so
        every check that watches the total sees a run that is going well -- and it is,
        at the wrong objective.
        """
        out: List[Finding] = []
        losses = {n: v for n, v in series.items()
                  if looks_like_loss(n) and not is_validation(n) and len(v) >= 12}
        # The total is not a component of itself: comparing `loss` against `kl_loss` and
        # `recon_loss` would have it dominate every share.
        components = {n: v for n, v in losses.items()
                      if _lower(n) not in ("loss", "total_loss", "total", "train_loss")}
        if len(components) < 2:
            return out

        head = max(3, min(len(v) for v in components.values()) // 4)
        early_total = sum(abs(_mean(v[:head])) for v in components.values())
        late_total = sum(abs(_mean(v[-head:])) for v in components.values())
        if early_total <= 0 or late_total <= 0:
            return out

        for name, values in components.items():
            early, late = abs(_mean(values[:head])), abs(_mean(values[-head:]))
            early_share, late_share = early / early_total, late / late_total
            # Share of the objective rather than absolute size: a term that was most of
            # the loss and is now a rounding error has stopped being optimised, whatever
            # its units happen to be. Testing absolute smallness instead missed a KL term
            # that fell from 62% of the objective to 4% because 0.04 is not a small
            # number in isolation.
            if (early_share > 0.20 and late_share < t["component_share_floor"]
                    and late < early * 0.2):
                out.append(Finding("component_collapse", name, WARNING,
                                   f"{name} was {100 * early_share:.0f}% of the objective and is now "
                                   f"{100 * late_share:.0f}% ({early:.4g} to {late:.4g}): this term "
                                   f"has stopped contributing, so the total is improving on what is "
                                   f"left of it",
                                   step, {"early_share": early_share, "late_share": late_share,
                                          "early": early, "late": late}, 0.75))
        return out

    @staticmethod
    def _check_eval_noise(series: Dict[str, List[float]], t: Dict[str, float],
                          step: Optional[int]) -> Iterable[Finding]:
        """Validation bouncing far more than training does.

        Two causes, both worth saying out loud: the eval set is too small to measure
        anything, or the model was never put in eval mode and dropout is still on. Either
        way every early-stopping decision taken from that number is a coin flip, and the
        curve itself looks unremarkable.
        """
        out: List[Finding] = []
        train = next((n for n in series if looks_like_loss(n) and not is_validation(n)), None)
        val = next((n for n in series if looks_like_loss(n) and is_validation(n)), None)
        if not train or not val:
            return out
        t_hist, v_hist = series[train], series[val]
        # Enough readings to have seen how much this number normally moves. Judged on
        # twelve it called an ordinary noisy validation split unusable.
        if len(t_hist) < 20 or len(v_hist) < 20:
            return out

        def jitter(values: List[float]) -> float:
            """Step-to-step movement, which is noise rather than progress."""
            steps = [abs(b - a) for a, b in zip(values, values[1:])]
            return _mean(steps[-20:]) if steps else 0.0

        train_jitter, val_jitter = jitter(t_hist), jitter(v_hist)
        if train_jitter <= 0 or val_jitter <= 0:
            return out
        # Validation noisier than training is backwards. Validation is usually computed
        # over more data than a single training batch, so it should be the steadier of
        # the two; when it is not, either the set is too small to measure anything or
        # the model is still in training mode and dropout is being sampled every time.
        # Floored against the scale so a perfectly flat training curve -- where the
        # ratio is unbounded -- does not make every run look unstable.
        # Bounce has to dominate the value, not merely exceed training's. Validation on
        # a small split is legitimately noisier than training, and saying so every time
        # is how a detector gets switched off; what is worth reporting is a number whose
        # movement between readings is a large fraction of the number itself.
        scale = abs(_mean(v_hist[-10:])) or 1.0
        floor = max(train_jitter, 0.002 * scale)
        if val_jitter > floor * 6.0 and val_jitter > 0.60 * scale:
            out.append(Finding("eval_unstable", val, WARNING,
                               f"{val} moves {val_jitter:.4g} between readings while {train} moves "
                               f"{train_jitter:.4g} -- {val_jitter / train_jitter:.0f}x as much. "
                               f"Either the eval set is too small to measure this, or the model is "
                               f"not in eval mode; either way this number cannot be compared to "
                               f"itself",
                               step, {"val_jitter": val_jitter, "train_jitter": train_jitter}, 0.75))
        return out

    # Hyperparameters that are logged as metrics and have a range outside which they are
    # simply wrong. Kept deliberately small: only values where being outside the range is
    # a mistake rather than a choice.
    SANE_RANGES = {
        "adam_eps": (1e-12, 1e-4, "Adam's epsilon swamps the second moment above ~1e-4, "
                                  "which silently shrinks every step"),
        "eps": (1e-12, 1e-4, "an optimizer epsilon this large swamps the second moment"),
        "bn_momentum": (1e-4, 1.0, "BatchNorm momentum of zero means the running statistics "
                                   "never update, so eval uses the values it was initialised with"),
        "dropout": (0.0, 0.9, "a dropout rate at or above 0.9 removes almost the whole layer"),
        "weight_decay": (0.0, 1.0, "weight decay above 1.0 will pull the weights to zero faster "
                                   "than the loss can put them back"),
        "clip_norm": (1e-3, 1e6, "a clip norm this small makes the clip, not the gradient, "
                                 "decide the step"),
        "temperature": (1e-4, 100.0, "a softmax temperature outside this range is either a hard "
                                     "argmax or a uniform distribution"),
    }

    @classmethod
    def _check_hyperparameters(cls, series: Dict[str, List[float]],
                               step: Optional[int]) -> Iterable[Finding]:
        """Config values that are logged, constant, and outside the range they work in.

        Not a curve shape at all: the run does exactly what it was told, and what it was
        told is wrong. These are invisible to every other check by construction, because
        nothing about the loss looks unusual -- it just converges somewhere worse.
        """
        out: List[Finding] = []
        for name, values in series.items():
            # Whole words, not substrings. `eps` is inside `steps`, and matching it there
            # told a healthy run that its step count was an optimizer epsilon outside
            # its working range. Padding both sides with the separator makes the test a
            # whole-token one and still matches multi-word keys like `weight_decay`.
            padded = "_" + re.sub(r"[^a-z0-9]+", "_", _lower(name)).strip("_") + "_"
            for key, (low, high, why) in cls.SANE_RANGES.items():
                if f"_{key}_" not in padded:
                    continue
                if looks_like_hardware_temp(name):
                    break                     # a thermometer, not a softmax temperature
                latest = values[-1]
                if len(set(values)) > 3:
                    break                     # it is being scheduled, not configured
                if low <= latest <= high:
                    break
                out.append(Finding("hyperparameter_out_of_range", name, WARNING,
                                   f"{name} is {latest:g}, outside the {low:g}..{high:g} range it "
                                   f"works in: {why}",
                                   step, {"value": latest, "low": low, "high": high}, 0.8))
                break
        return out

    @staticmethod
    def _twins(series: Dict[str, List[float]]) -> List[Tuple[str, str]]:
        """Pairs that name the same quantity measured two ways.

        `loss` and `ema_loss`, `text_emb_norm` and `image_emb_norm`: one is a prefixed
        or suffixed variant of the other, or the two share a stem and differ by one
        qualifier. Only these are compared, because comparing every pair of series in a
        run would find a coincidence every time.
        """
        pairs: List[Tuple[str, str]] = []
        names = sorted(series)
        qualifiers = ("ema_", "shadow_", "avg_", "smoothed_", "teacher_", "student_",
                      "text_", "image_", "audio_", "video_", "online_", "target_")
        for i, a in enumerate(names):
            low_a = _lower(a)
            for b in names[i + 1:]:
                low_b = _lower(b)
                if is_validation(a) != is_validation(b):
                    continue                     # train vs val is overfitting, handled above
                stem_a = low_a
                stem_b = low_b
                for q in qualifiers:
                    stem_a = stem_a.replace(q, "")
                    stem_b = stem_b.replace(q, "")
                if stem_a == stem_b and low_a != low_b:
                    pairs.append((a, b))
        return pairs

    @staticmethod
    def _check_known_relations(series: Dict[str, List[float]], t: Dict[str, float],
                               step: Optional[int]) -> Iterable[Finding]:
        """Numbers with a defined relationship to each other, checked against it.

        Perplexity is exp(loss). When the two disagree one of them is wrong, and both
        are on the dashboard being used to make decisions.
        """
        out: List[Finding] = []
        loss = next((v for n, v in series.items()
                     if looks_like_loss(n) and not is_validation(n)), None)
        ppl_name = next((n for n in series if "perplexity" in _lower(n) or _lower(n) == "ppl"), None)
        if loss is None or ppl_name is None:
            return out
        ppl = series[ppl_name]
        n = min(len(loss), len(ppl))
        if n < 8:
            return out
        errors = []
        for a, b in zip(loss[-n:], ppl[-n:]):
            if not (-50 < a < 50) or b <= 0:
                continue
            expected = math.exp(a)
            errors.append(abs(b - expected) / max(expected, 1e-12))
        if errors and _mean(errors) > max(t["relation_tolerance"], 0.05):
            out.append(Finding("relationship_broken", ppl_name, WARNING,
                               f"{ppl_name} is not exp(loss): off by {100 * _mean(errors):.0f}% on "
                               f"average over {len(errors)} readings, so one of the two is being "
                               f"computed wrong",
                               step, {"mean_error": _mean(errors), "readings": len(errors)}, 0.8))
        return out

    @staticmethod
    def _check_objective_gamed(series: Dict[str, List[float]], t: Dict[str, float],
                               step: Optional[int]) -> Iterable[Finding]:
        """A score going up while the thing it is supposed to summarise falls apart.

        Reward climbing as episode length collapses is the canonical shape: the policy
        found a way to score without doing the task, and the metric everybody watches
        says it is working.
        """
        out: List[Finding] = []
        companions = ("episode_length", "ep_len", "episode_len", "steps_per_episode",
                      "solve_rate", "task_success", "coverage", "diversity")
        scores = {n: v for n, v in series.items()
                  if looks_like_score(n) and not looks_like_loss(n) and len(v) >= 12}
        others = {n: v for n, v in series.items() if _has(n, companions)}
        for score_name, score in scores.items():
            span = max(3, len(score) // 4)
            score_gain = _mean(score[-span:]) - _mean(score[:span])
            if score_gain <= 0 or abs(_mean(score[:span])) < 1e-12:
                continue
            if score_gain / max(abs(_mean(score[:span])), 1e-12) < 0.15:
                continue
            for other_name, other in others.items():
                n = min(len(score), len(other))
                if n < 12:
                    continue
                o_span = max(3, n // 4)
                early, late = _mean(other[:o_span]), _mean(other[-o_span:])
                if early <= 0 or late > early * 0.6:
                    continue
                out.append(Finding("objective_gamed", score_name, WARNING,
                                   f"{score_name} is rising while {other_name} collapsed from "
                                   f"{early:.4g} to {late:.4g}: the score is going up without the "
                                   f"task being done",
                                   step, {"score": score_name, "companion": other_name,
                                          "early": early, "late": late}, 0.8))
        return out

    @staticmethod
    def _check_widening_gap(series: Dict[str, List[float]], t: Dict[str, float],
                            step: Optional[int]) -> Iterable[Finding]:
        """Train pulling away from validation while validation stands still.

        The overfitting check needs validation to *rise*. The commoner shape is that it
        simply stops improving while training keeps going, and no amount of val
        regression ever happens -- the gap is the whole signal.
        """
        out: List[Finding] = []
        train = next((n for n in series if looks_like_loss(n) and not is_validation(n)), None)
        val = next((n for n in series if looks_like_loss(n) and is_validation(n)), None)
        if not train or not val:
            return out
        t_hist, v_hist = series[train], series[val]
        n = min(len(t_hist), len(v_hist))
        if n < 14:
            return out
        head = max(4, n // 3)
        early_gap = _mean(v_hist[-n:][:head]) - _mean(t_hist[-n:][:head])
        late_gap = _mean(v_hist[-head:]) - _mean(t_hist[-head:])
        val_early, val_late = _mean(v_hist[-n:][:head]), _mean(v_hist[-head:])
        train_early, train_late = _mean(t_hist[-n:][:head]), _mean(t_hist[-head:])
        if train_early <= 0 or val_early <= 0:
            return out
        train_gain = (train_early - train_late) / abs(train_early)
        val_gain = (val_early - val_late) / abs(val_early)
        val_noise = _stdev(v_hist[-n:])
        # Training improved materially, validation did not, and the gap grew by more
        # than validation's own noise.
        if (train_gain > 0.15 and val_gain < 0.02
                and late_gap > early_gap + max(2.0 * val_noise, 0.05 * abs(val_early))
                and late_gap > 0):
            out.append(Finding("widening_gap", val, WARNING,
                               f"{train} improved {100 * train_gain:.0f}% while {val} moved "
                               f"{100 * val_gain:.0f}%: the gap between them went from "
                               f"{early_gap:.4g} to {late_gap:.4g}, which is memorising without "
                               f"{val} ever having to get worse",
                               step, {"train": train, "early_gap": early_gap,
                                      "late_gap": late_gap, "train_gain": train_gain}, 0.75))
        return out

    @staticmethod
    def _check_chance_level(series: Dict[str, List[float]], histories: Dict[str, Sequence[Any]],
                            t: Dict[str, float], step: Optional[int]) -> Iterable[Finding]:
        """A loss sitting at ln(number of classes) is a model predicting the prior.

        Shuffled labels, a tokenizer that does not match the embedding table, a target
        column of noise: the loss converges beautifully to exactly chance and every
        shape-based check sees a healthy curve that has converged.
        """
        out: List[Finding] = []
        count = None
        for name, history in histories.items():
            if looks_like_class_count(name):
                values = _finite(list(history)[-8:])
                if values and values[-1] >= 2:
                    count = values[-1]
                    break
        if count is None:
            return out
        chance = math.log(count)
        for name, values in series.items():
            if not looks_like_loss(name) or len(values) < 12:
                continue
            recent = values[-max(6, len(values) // 4):]
            if _stdev(recent) > 0.05 * max(chance, 1e-9):
                continue                      # still moving; not settled on anything
            if abs(_mean(recent) - chance) <= 0.02 * chance:
                out.append(Finding("chance_level", name, CRITICAL,
                                   f"{name} has settled at {_mean(recent):.4g}, which is ln({count:g}) "
                                   f"= {chance:.4g}: the model is predicting the class prior and has "
                                   f"learned nothing from the inputs",
                                   step, {"loss": _mean(recent), "chance": chance,
                                          "classes": count}, 0.9))
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
        return self._check_tensor_shape(name, stats, step)

    def _check_tensor_shape(self, name: str, stats: Dict[str, Any],
                            step: Optional[int]) -> Optional[Finding]:
        """Everything about a tensor other than whether it is finite.

        This used to stop at the NaN and inf counts, so a weight whose standard
        deviation had climbed to 1e12 -- every element finite, the model long past
        saving -- reported nothing at all. Likewise a parameter that quietly moved to
        the CPU, which is correct and fifty times slower.
        """
        previous = self._tensor_seen.get(name)
        current = {"dtype": stats.get("dtype"), "device": stats.get("device")}
        self._tensor_seen[name] = current

        std = _as_float(stats.get("std"))
        mean = _as_float(stats.get("mean"))
        if std is not None and math.isfinite(std) and std > 1e6:
            return Finding("tensor_norm_explosion", name, CRITICAL,
                           f"{name} has a standard deviation of {std:g}: the values are finite and "
                           f"far outside anything a trained weight holds",
                           step, {"std": std, "mean": mean}, 0.9)
        if (std is not None and mean is not None and std == 0.0 and mean == 0.0
                and stats.get("shape")):
            return Finding("tensor_all_zeros", name, WARNING,
                           f"{name} is entirely zero: a layer that was never initialised, or one "
                           f"whose gradients cannot reach it",
                           step, {"shape": list(stats.get("shape") or [])}, 0.85)

        if previous:
            if previous.get("dtype") and current["dtype"] and previous["dtype"] != current["dtype"]:
                return Finding("tensor_dtype_change", name, WARNING,
                               f"{name} changed dtype from {previous['dtype']} to {current['dtype']} "
                               f"mid-run: precision is being lost somewhere it was not intended",
                               step, {"from": previous["dtype"], "to": current["dtype"]}, 0.8)
            if previous.get("device") and current["device"] and previous["device"] != current["device"]:
                return Finding("tensor_device_change", name, WARNING,
                               f"{name} moved from {previous['device']} to {current['device']}: "
                               f"every step now copies it across the bus",
                               step, {"from": previous["device"], "to": current["device"]}, 0.85)
        elif current["device"] and "cpu" in str(current["device"]).lower():
            others = [v.get("device") for k, v in self._tensor_seen.items() if k != name]
            if any(d and "cpu" not in str(d).lower() for d in others):
                return Finding("tensor_device_change", name, WARNING,
                               f"{name} is on {current['device']} while the rest of the model is on "
                               f"an accelerator: every step copies it across the bus",
                               step, {"device": current["device"]}, 0.8)
        return None


def summarise(findings: Sequence[Finding]) -> str:
    """The one-line form, for a log line or a prompt header."""
    if not findings:
        return ""
    ordered = sorted(findings, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), -f.confidence))
    return "; ".join(f.message for f in ordered)
