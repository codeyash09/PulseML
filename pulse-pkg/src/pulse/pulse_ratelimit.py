"""
One policy for "the AI provider told us to slow down".

Every place Pulse calls a model (the debugger's pipeline, the GUI's chat, Pulse Code's agent
loop) used to carry its own copy of "retry 3 times after 1s, 2s, 4s". A rate limit lasts
seconds to minutes, so three quick retries all land inside the same window and fail together;
what happened next depended on which call in the pipeline happened to be the one that got the
429. This module is the single answer instead:

  * a rate limit is waited out, not retried a fixed number of times. When the provider gives a
    timer (a Retry-After header, or "retry in 34s" in the message) Pulse waits that long, in
    full, whatever the budget below says -- the provider knows when its limit lifts and Pulse
    does not. Without a timer it backs off exponentially with jitter;
  * Pulse's own guessing is bounded by a budget per request (PULSE_RATE_LIMIT_WAIT seconds,
    default 10 minutes; 0 means never wait). Provider timers are bounded only by a generous
    ceiling (an hour of them in total), so a provider that keeps saying "retry in 30s" forever,
    or asks for an hours-long reset, ends up parked and retried later instead of blocking;
  * a 429 that plainly means "no credit" (and carries no retry hint) is not waited on at all --
    it is reported as what it is. Quota-style limits that clear (per-minute, per-day) are
    waited out like any other rate limit;
  * the wait is announced once, not once per second, and can be cut short.

Deliberately stdlib-only: it is imported by modules that must stay cheap to import.
"""
import os
import random
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional

DEFAULT_MAX_WAIT_SECONDS = 600.0     # total time one request will spend waiting out rate limits
BASE_WAIT_SECONDS = 5.0              # first wait when the provider gives no hint
MAX_STEP_SECONDS = 60.0              # longest single wait chosen by us (a provider hint may exceed it)
MAX_HINT_TOTAL_SECONDS = 3600.0      # most one request will spend waiting on timers the PROVIDER gave
_SLICE_SECONDS = 0.5                 # granularity at which a wait notices it was cancelled

_ENV_MAX_WAIT = "PULSE_RATE_LIMIT_WAIT"

# Wording that only ever means "this account has no credit": waiting cannot change it.
# Deliberately narrow. "Quota" is NOT here: providers use it for limits that clear on their own
# (Gemini's per-minute free tier says "You exceeded your current quota, please check your plan and
# billing details" and, in the same message, "retry in 34s"), and so do per-day and per-minute
# token quotas. When in doubt a limit is waited out -- the wait is bounded, and a request that is
# still limited afterwards is parked and retried later, so being wrong costs minutes, not the run.
_HARD_LIMIT_RE = re.compile(
    r"insufficient[_ ]quota|out of credits?|insufficient credits?|credit balance (?:is )?too low"
    r"|payment required|add (?:more )?credits",
    re.IGNORECASE)

_RATE_LIMIT_TEXT_RE = re.compile(
    r"rate[ _-]?limit|too many requests|resource[_ ]exhausted|(?:error|status|code|http)\D{0,12}\b429\b",
    re.IGNORECASE)

# "Please retry in 34.5s", "retryDelay": "34s", "try again in 20 seconds", "retry after 1500ms"
_RETRY_HINT_RE = re.compile(
    r"(?:retry(?:[ _]?delay)?|try again)\W{0,6}(?:in|after)?\W{0,6}([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|secs?|seconds?|m|mins?|minutes?)\b",
    re.IGNORECASE)


def max_wait_seconds() -> float:
    """The per-request waiting budget: PULSE_RATE_LIMIT_WAIT, else the default. Never negative."""
    raw = os.environ.get(_ENV_MAX_WAIT)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_WAIT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MAX_WAIT_SECONDS
    if value != value:                      # NaN
        return DEFAULT_MAX_WAIT_SECONDS
    return max(0.0, value)


def _status_code(exc: BaseException) -> Optional[int]:
    for owner in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status", "http_status"):
            value = getattr(owner, attr, None)
            if isinstance(value, int):
                return value
    return None


def is_rate_limited(exc: BaseException, rate_limit_class: Optional[type] = None) -> bool:
    """True for a provider's "slow down" -- whichever SDK or proxy raised it.

    `rate_limit_class` is litellm.RateLimitError when the caller has it; the rest is by
    status code, class name and message, so a provider litellm maps differently (OpenRouter's
    wrapped 429s, a local gateway) is still recognised."""
    if rate_limit_class is not None and isinstance(exc, rate_limit_class):
        return True
    if _status_code(exc) == 429:
        return True
    if "ratelimit" in type(exc).__name__.lower():
        return True
    return bool(_RATE_LIMIT_TEXT_RE.search(str(exc) or ""))


def is_hard_limit(exc: BaseException) -> bool:
    """A rate-limit-shaped error that waiting cannot fix: the account has no credit.

    Never true when the provider says when to retry (a Retry-After header, "retry in 34s"):
    it is telling us the limit clears, whatever it calls it."""
    if retry_after_seconds(exc) is not None:
        return False
    return bool(_HARD_LIMIT_RE.search(str(exc) or ""))


def _header(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    try:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.title())
        return None if value is None else str(value)
    except Exception:
        return None


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """How long the provider itself said to wait, or None when it did not say."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    ms = _header(headers, "retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    value = _header(headers, "retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:                            # an HTTP-date
                return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except Exception:
                pass
    attr = getattr(exc, "retry_after", None)
    if isinstance(attr, (int, float)) and attr >= 0:
        return float(attr)
    match = _RETRY_HINT_RE.search(str(exc) or "")
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("ms") or unit.startswith("milli"):
            return amount / 1000.0
        if unit.startswith("m") and not unit.startswith("ms") and not unit.startswith("milli"):
            return amount * 60.0
        return amount
    return None


class RateLimitWaiter:
    """The waiting state of ONE request: how long it has waited, and how long to wait next.

    One waiter per model call, so a request that is rate limited five times in a row waits
    5s, 10s, 20s... (or whatever the provider asks) against a single budget, instead of five
    unrelated "attempt 1 of 3"s."""

    def __init__(self, budget: Optional[float] = None):
        self.budget = max_wait_seconds() if budget is None else max(0.0, float(budget))
        self.waited = 0.0
        self.hits = 0
        self.last_hint: Optional[float] = None       # the provider's timer behind the latest delay

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.waited)

    def next_delay(self, exc: BaseException) -> Optional[float]:
        """Seconds to wait before retrying after `exc`, or None when this request should stop
        waiting.

        A timer the provider gave is honoured in full, even past the budget: the budget limits
        how long Pulse guesses, not how long it takes the provider to lift its limit. It stops
        only when waiting was switched off (budget 0) or the provider's timers have already
        added up to MAX_HINT_TOTAL_SECONDS."""
        self.hits += 1
        hint = retry_after_seconds(exc)
        self.last_hint = hint
        if hint is not None:
            if self.budget <= 0:
                return None                                         # waiting was switched off
            delay = hint + 0.5 + random.uniform(0.0, 0.5)           # just past the provider's own mark
            if self.waited + delay > max(self.budget, MAX_HINT_TOTAL_SECONDS):
                return None
            return delay
        delay = min(MAX_STEP_SECONDS, BASE_WAIT_SECONDS * (2 ** (self.hits - 1)))
        delay *= random.uniform(1.0, 1.25)                          # de-synchronise parallel callers
        return delay if delay <= self.remaining else None

    def wait(self, delay: float, should_stop: Optional[Callable[[], bool]] = None) -> bool:
        """Sleep `delay` seconds in short slices. False if `should_stop()` ended the wait early.

        Time is accounted by the slices requested, not measured, so the budget is exact however
        the sleeps are scheduled (and a test can replace time.sleep)."""
        left = float(delay)
        while left > 0:
            if should_stop is not None and should_stop():
                return False
            step = min(_SLICE_SECONDS, left)
            time.sleep(step)
            left -= step
            self.waited += step
        return True


def describe_wait(delay: float, waiter: RateLimitWaiter) -> str:
    """The one line printed when a wait starts -- says how long, and why."""
    if waiter.last_hint is not None:
        return (f"rate limited by the provider -- it asked to retry in {waiter.last_hint:.0f}s; "
                f"waiting {delay:.0f}s")
    left = int(round(waiter.remaining - delay))
    more = f"; will keep trying for up to {left}s more" if left > 0 else "; this is the last try"
    return f"rate limited by the provider -- waiting {delay:.0f}s before retrying{more}"