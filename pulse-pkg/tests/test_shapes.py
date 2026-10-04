"""One policy for "the provider said slow down" (pulse_ratelimit), and how every model call
uses it: a rate limit is waited out against a budget -- honouring the provider's own
Retry-After -- instead of being retried three times a few seconds apart, and when the budget
is spent the failure is the same one everywhere (transient + rate_limited)."""
import types

import pytest

import pulse.pulse_cli as pc
from pulse import pulse_ratelimit as rl
from pulse.pulse_cli import PulseCLI


class Boom(Exception):
    """A provider error with whatever attributes a test wants."""

    def __init__(self, message="boom", status=None, headers=None):
        super().__init__(message)
        if status is not None:
            self.status_code = status
        if headers is not None:
            self.response = types.SimpleNamespace(headers=headers)


@pytest.fixture(autouse=True)
def no_budget_override(monkeypatch):
    monkeypatch.delenv("PULSE_RATE_LIMIT_WAIT", raising=False)


# ---- recognising a rate limit ----------------------------------------------------------

@pytest.mark.parametrize("exc", [
    Boom("x", status=429),
    Boom("Error code: 429 - {'error': 'slow down'}"),
    Boom("Rate limit reached for requests"),
    Boom("RESOURCE_EXHAUSTED: quota"),
    Boom("Too Many Requests"),
])
def test_rate_limit_is_recognised(exc):
    assert rl.is_rate_limited(exc)


def test_rate_limit_recognised_by_class_name_and_status_on_the_response():
    class RateLimitError(Exception):
        pass
    assert rl.is_rate_limited(RateLimitError("whatever"))
    e = Exception("opaque")
    e.response = types.SimpleNamespace(status_code=429, headers={})
    assert rl.is_rate_limited(e)


@pytest.mark.parametrize("exc", [
    ValueError("bad model"),
    Boom("You requested 429 tokens but the limit is 100"),     # a number, not a status
    Boom("server error", status=503),
])
def test_other_errors_are_not_rate_limits(exc):
    assert not rl.is_rate_limited(exc)


@pytest.mark.parametrize("text", [
    "You exceeded your current quota, please check your plan and billing details.",
    "insufficient_quota",
    "Your credit balance is too low to access the API",
    "Insufficient credits. Add more credits.",
])
def test_out_of_credit_is_a_hard_limit_that_waiting_cannot_fix(text):
    assert rl.is_hard_limit(Boom(text, status=429))


def test_an_ordinary_429_is_not_a_hard_limit():
    assert not rl.is_hard_limit(Boom("Rate limit reached for requests, retry in 20s", status=429))


# ---- how long the provider says to wait ------------------------------------------------

def test_retry_after_header_seconds_and_milliseconds():
    assert rl.retry_after_seconds(Boom("x", 429, {"retry-after": "12"})) == 12.0
    assert rl.retry_after_seconds(Boom("x", 429, {"retry-after-ms": "1500"})) == 1.5
    assert rl.retry_after_seconds(Boom("x", 429, {"Retry-After": "7"})) == 7.0


def test_retry_after_http_date(monkeypatch):
    monkeypatch.setattr(rl.time, "time", lambda: 1_000_000_000.0)          # Sun, 09 Sep 2001 01:46:40 GMT
    got = rl.retry_after_seconds(Boom("x", 429, {"retry-after": "Sun, 09 Sep 2001 01:47:10 GMT"}))
    assert got == pytest.approx(30.0, abs=1.0)
    past = rl.retry_after_seconds(Boom("x", 429, {"retry-after": "Sun, 09 Sep 2001 01:40:00 GMT"}))
    assert past == 0.0                                                     # a date already gone: no wait


@pytest.mark.parametrize("message,seconds", [
    ("Please retry in 34.5s.", 34.5),
    ('{"retryDelay": "34s"}', 34.0),
    ("try again in 20 seconds", 20.0),
    ("retry after 1500ms", 1.5),
    ("Rate limited. Try again in 2 minutes", 120.0),
])
def test_retry_hint_in_the_message(message, seconds):
    assert rl.retry_after_seconds(Boom(message, 429)) == pytest.approx(seconds)


def test_no_hint_is_none():
    assert rl.retry_after_seconds(Boom("rate limited", 429)) is None


# ---- the budget -------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, rl.DEFAULT_MAX_WAIT_SECONDS), ("", rl.DEFAULT_MAX_WAIT_SECONDS), ("abc", rl.DEFAULT_MAX_WAIT_SECONDS),
    ("nan", rl.DEFAULT_MAX_WAIT_SECONDS), ("30", 30.0), ("0", 0.0), ("-5", 0.0),
])
def test_budget_from_the_environment(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("PULSE_RATE_LIMIT_WAIT", value)
    assert rl.max_wait_seconds() == expected


def test_waits_grow_and_the_budget_is_not_exceeded():
    w = rl.RateLimitWaiter(budget=100)
    delays = []
    while True:
        d = w.next_delay(Boom("rate limit", 429))
        if d is None:
            break
        delays.append(d)
        w.waited += d
    assert len(delays) >= 4
    assert delays[0] < delays[1] < delays[2]                       # exponential
    assert 5.0 <= delays[0] <= 5.0 * 1.25
    assert sum(delays) <= 100


def test_a_wait_is_never_longer_than_the_step_cap_unless_the_provider_asks():
    w = rl.RateLimitWaiter(budget=10_000)
    w.hits = 20                                                    # many hits in: 5 * 2**20 would be huge
    assert w.next_delay(Boom("rate limit", 429)) <= rl.MAX_STEP_SECONDS * 1.25


def test_the_providers_retry_after_is_honoured_and_is_not_undercut():
    w = rl.RateLimitWaiter(budget=100)
    d = w.next_delay(Boom("x", 429, {"retry-after": "30"}))
    assert 30.0 <= d <= 32.0


def test_a_retry_after_longer_than_what_is_left_stops_waiting_at_once():
    w = rl.RateLimitWaiter(budget=20)
    assert w.next_delay(Boom("x", 429, {"retry-after": "300"})) is None


def test_zero_budget_never_waits():
    assert rl.RateLimitWaiter(budget=0).next_delay(Boom("rate limit", 429)) is None


def test_wait_sleeps_in_slices_and_accounts_for_them(monkeypatch):
    slept = []
    monkeypatch.setattr(rl.time, "sleep", slept.append)
    w = rl.RateLimitWaiter(budget=100)
    assert w.wait(2.2) is True
    assert sum(slept) == pytest.approx(2.2)
    assert max(slept) <= 0.5 + 1e-9
    assert w.waited == pytest.approx(2.2)


def test_wait_can_be_cut_short(monkeypatch):
    slept = []
    monkeypatch.setattr(rl.time, "sleep", slept.append)
    w = rl.RateLimitWaiter(budget=100)
    stop = iter([False, False, True])
    assert w.wait(60, should_stop=lambda: next(stop)) is False
    assert len(slept) == 2


# ---- _call_model -------------------------------------------------------------------------

@pytest.fixture
def cli(tmp_path, monkeypatch):
    c = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdf"))
    c.agent_provider, c.agent_key = "openai", "sk-test"
    c.agent_model_string, c.agent_api_base = "openai/test-model", None
    c._ensure_retry_ticker = lambda: None
    monkeypatch.setattr(pc, "_clamp_output_tokens", lambda m, r: r)
    return c


@pytest.fixture
def clock(monkeypatch):
    """Replace sleeping everywhere a model call can sleep, and total what was slept."""
    slept = []
    monkeypatch.setattr(rl.time, "sleep", slept.append)
    monkeypatch.setattr(pc.time, "sleep", slept.append)
    return slept


def _ok(text="answer"):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))], usage=None)


def _script(monkeypatch, outcomes):
    """litellm.completion that raises/returns the next outcome each call; returns the call counter."""
    calls = []

    def fake(**kw):
        calls.append(kw)
        outcome = outcomes(len(calls)) if callable(outcomes) else outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(pc.litellm, "completion", fake)
    return calls


def test_a_rate_limit_that_clears_is_waited_out_and_the_answer_returned(cli, monkeypatch, clock):
    calls = _script(monkeypatch, [Boom("Rate limit reached", 429), Boom("Rate limit reached", 429), _ok("fixed")])
    assert cli._call_model("hi") == "fixed"
    assert len(calls) == 3
    assert sum(clock) >= 5.0 + 10.0                                # 5s then 10s, not 1s then 2s


def test_a_rate_limit_is_not_capped_at_three_attempts(cli, monkeypatch, clock):
    """The old behaviour: three tries 1s and 2s apart, all inside one rate-limit window."""
    n_limited = 6
    calls = _script(monkeypatch, lambda n: Boom("Rate limit reached", 429) if n <= n_limited else _ok("late"))
    assert cli._call_model("hi") == "late"
    assert len(calls) == n_limited + 1


def test_the_provider_retry_after_sets_the_wait(cli, monkeypatch, clock):
    _script(monkeypatch, [Boom("slow down", 429, {"retry-after": "20"}), _ok()])
    cli._call_model("hi")
    assert 20.0 <= sum(clock) <= 22.5


def test_a_rate_limit_that_never_clears_fails_the_same_way_every_time(cli, monkeypatch, clock):
    monkeypatch.setenv("PULSE_RATE_LIMIT_WAIT", "60")
    calls = _script(monkeypatch, [Boom("Rate limit reached", 429)])
    with pytest.raises(pc.AgentRequestFailed) as info:
        cli._call_model("hi")
    assert info.value.rate_limited is True and info.value.transient is True
    assert "rate limited" in str(info.value)
    assert sum(clock) <= 60 + 1e-6                                 # bounded by the budget
    assert len(calls) > 3


def test_zero_budget_fails_a_rate_limit_straight_away(cli, monkeypatch, clock):
    monkeypatch.setenv("PULSE_RATE_LIMIT_WAIT", "0")
    calls = _script(monkeypatch, [Boom("Rate limit reached", 429)])
    with pytest.raises(pc.AgentRequestFailed) as info:
        cli._call_model("hi")
    assert info.value.rate_limited and len(calls) == 1 and clock == []


def test_out_of_credit_is_reported_at_once_and_not_waited_on(cli, monkeypatch, clock):
    calls = _script(monkeypatch, [Boom("You exceeded your current quota, check your plan and billing", 429)])
    with pytest.raises(pc.AgentRequestFailed) as info:
        cli._call_model("hi")
    assert info.value.transient is False and info.value.rate_limited is False
    assert "quota or credit" in str(info.value)
    assert len(calls) == 1 and clock == []


def test_while_training_is_paused_on_the_call_the_wait_is_short(cli, monkeypatch, clock):
    """A mid-run escalation has training paused on this very call: ten idle GPU-minutes are worse
    than parking the request and retrying in the background."""
    monkeypatch.setenv("PULSE_RATE_LIMIT_WAIT", "600")
    cli._pending_agent_start_ts = 1.0
    _script(monkeypatch, [Boom("Rate limit reached", 429)])
    with pytest.raises(pc.AgentRequestFailed) as info:
        cli._call_model("hi")
    assert info.value.rate_limited
    assert sum(clock) <= pc._STALLED_RATE_LIMIT_WAIT_SECONDS + 1e-6


def test_ctrl_c_during_the_wait_stops_without_queueing_a_retry(cli, monkeypatch, clock):
    cli._stop_requested = True
    _script(monkeypatch, [Boom("Rate limit reached", 429)])
    with pytest.raises(pc.AgentRequestFailed) as info:
        cli._call_model("hi")
    assert info.value.transient is False


def test_other_transient_errors_keep_their_three_quick_attempts(cli, monkeypatch, clock):
    down = pc.litellm.ServiceUnavailableError(message="down", llm_provider="openai", model="m")
    calls = _script(monkeypatch, [down])
    with pytest.raises(pc.AgentRequestFailed) as info:
        cli._call_model("hi")
    assert len(calls) == 3 and clock == [1, 2]
    assert info.value.rate_limited is False


def test_a_real_litellm_rate_limit_error_is_recognised(cli, monkeypatch, clock):
    err = pc.litellm.RateLimitError(message="Rate limit", llm_provider="openai", model="m")
    calls = _script(monkeypatch, [err, _ok("ok")])
    assert cli._call_model("hi") == "ok"
    assert len(calls) == 2


def test_rate_limited_failure_is_always_transient():
    assert pc.AgentRequestFailed("x", transient=False, rate_limited=True).transient is True
    assert pc.AgentRequestFailed("x").rate_limited is False