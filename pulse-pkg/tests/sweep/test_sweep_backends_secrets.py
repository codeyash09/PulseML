"""Sweep: where API keys / credentials can leak -- the --agent-log file and TERMINAL output.

No model is called: the agent-log helpers are exercised directly.
"""
import os

import pytest

from pulse import pulse_cli as cli
from pulse import pulse_terminal as term

FAKE_OPENROUTER_KEY = "sk-or-v1-" + "0123456789abcdef" * 4
FAKE_ANTHROPIC_KEY = "sk-ant-api03-" + "Z" * 60


@pytest.fixture
def agent_log(tmp_path, monkeypatch):
    path = tmp_path / "agent.log"
    monkeypatch.setenv("PULSE_AGENT_LOG", str(path))
    monkeypatch.setattr(cli, "_agent_log_resolved", {})
    monkeypatch.setattr(cli, "_agent_log_seen", {})
    return path


# ============================================================================ bugs

def test_bug_agent_log_writes_env_api_key_in_clear(agent_log, monkeypatch):
    """--agent-log records every prompt verbatim. Prompts include TERMINAL output and
    project files; `env`, `cat .env`, `pip config list` or a traceback can put the
    user's key in there. pulse_supabase.scrub_secrets exists for exactly this ('...or
    gets written to a local log the user might paste/share') but _agent_log_write never
    calls it, so the key lands in pulse_agent.log in clear text. Correct: pass `text`
    through cloud.scrub_secrets in _agent_log_write."""
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER_KEY)
    cli._agent_log_call("check-in", [
        {"role": "system", "content": "You are Pulse."},
        {"role": "user", "content": f"TERMINAL EXECUTION RESULT\nstdout:\nOPENROUTER_API_KEY={FAKE_OPENROUTER_KEY}\n"},
    ], "openrouter/deepseek/deepseek-v4-flash")
    text = agent_log.read_text()
    assert "TERMINAL EXECUTION RESULT" in text
    assert FAKE_OPENROUTER_KEY not in text


def test_bug_agent_log_writes_hardcoded_key_from_user_code(agent_log):
    """Same leak for a key hard-coded in the tracked script (the whole script is in the
    prompt): `client = Anthropic(api_key="sk-ant-...")` is copied into the agent log."""
    code = f'client = anthropic.Anthropic(api_key="{FAKE_ANTHROPIC_KEY}")\n'
    cli._agent_log_call("investigate", [{"role": "user", "content": code}], "m")
    cli._agent_log_event("ESCALATED", code)
    assert FAKE_ANTHROPIC_KEY not in agent_log.read_text()


def test_bug_terminal_result_exposes_provider_key_to_model(monkeypatch, tmp_path):
    """TERMINAL subprocesses inherit the whole environment, including the provider key
    Pulse itself put in os.environ (os.environ[env_var] = key). Whatever a command
    prints -- `env`, `printenv`, a debug dump -- goes back verbatim into the model's
    context (and from there into the agent log and cloud agent_logs). Keeping TERMINAL
    unrestricted is fine; the rendered result should be scrubbed. Correct: run
    TerminalResult.render() output (stdout/stderr) through cloud.scrub_secrets."""
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER_KEY)
    ex = term.TerminalExecutor(default_cwd=str(tmp_path))
    res = ex.run(term.TerminalRequest(command="printenv OPENROUTER_API_KEY"))
    assert res.exit_code == 0
    assert FAKE_OPENROUTER_KEY not in res.render()


# ============================================================================ ok

def test_ok_agent_log_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PULSE_AGENT_LOG", raising=False)
    monkeypatch.setattr(cli, "_agent_log_resolved", {})
    assert cli._agent_log_path() is None
    assert cli._agent_log_call("x", [{"role": "user", "content": "hi"}], "m") == 0


def test_ok_agent_log_dedups_repeated_messages(agent_log):
    msgs = [{"role": "system", "content": "S" * 50}, {"role": "user", "content": "Q"}]
    cli._agent_log_call("a", msgs, "m")
    cli._agent_log_call("b", msgs, "m")
    text = agent_log.read_text()
    assert text.count("S" * 50) == 1
    assert "same as before" in text


def test_ok_saved_profile_never_contains_key(tmp_path, monkeypatch):
    from pulse import pulse_supabase as cloud
    monkeypatch.setattr(cloud, "PROFILE_PATH", tmp_path / "profile.json")
    cloud.save_cached_profile(agent_provider="OpenRouter", agent_env_key="OPENROUTER_API_KEY")
    assert FAKE_OPENROUTER_KEY not in (tmp_path / "profile.json").read_text()
