"""
One policy for "the AI provider told us to slow down".

Every place Pulse calls a model (the debugger's pipeline, the GUI's chat, Pulse Code's agent
loop) used to carry its own copy of "retry 3 times after 1s, 2s, 4s". A rate limit lasts
seconds to minutes, so three quick retries all land inside the same window and fail together;
what happened next depended on which call in the pipeline happened to be the one that got the
429. This module is the single answer instead:

  * a rate limit is waited out, not retried a fixed number of times: the provider's own
    Retry-After (header, or "retry in 34s" in the message) when it gives one, otherwise an
    exponential wait with jitter;
  * waiting is bounded by a total budget per request (PULSE_RATE_LIMIT_WAIT seconds, default
    10 minutes; 0 means fail straight away), so nothing waits forever;
  * a 429 that can never clear by waiting (out of credit, billing, exhausted quota) is not
    waited on at all -- it is reported as what it is;
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
_SLICE_SECONDS = 0.5                 # granularity at which a wait notices it was cancelled

_ENV_MAX_WAIT = "PULSE_RATE_LIMIT_WAIT"

# A 429 with one of these in it is not "too many requests right now": the account itself
# is out of something, and no amount of waiting changes that.
_HARD_LIMIT_RE = re.compile(
    r"insufficient[_ ]quota|exceeded your current quota|out of credits?|insufficient credits?"
    r"|credit balance|billing|payment required|add (?:more )?credits|quota exceeded.{0,40}plan",
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
    """A rate-limit-shaped error that waiting cannot fix (no credit, quota gone for the plan)."""
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

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.waited)

    def next_delay(self, exc: BaseException) -> Optional[float]:
        """Seconds to wait before retrying after `exc`, or None when this request should stop
        waiting (the budget is spent, or the provider asked for longer than what is left)."""
        self.hits += 1
        hint = retry_after_seconds(exc)
        if hint is not None:
            delay = hint + 0.5 + random.uniform(0.0, 1.0)          # just past the provider's own mark
        else:
            delay = min(MAX_STEP_SECONDS, BASE_WAIT_SECONDS * (2 ** (self.hits - 1)))
            delay *= random.uniform(1.0, 1.25)                     # de-synchronise parallel callers
        if delay > self.remaining:
            return None
        return delay

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
    """The one line printed when a wait starts -- says how long, and for how long it will keep trying."""
    left = int(round(waiter.remaining - delay))
    more = f"; will keep trying for up to {left}s more" if left > 0 else "; this is the last try"
    return f"rate limited by the provider -- waiting {delay:.0f}s before retrying{more}"