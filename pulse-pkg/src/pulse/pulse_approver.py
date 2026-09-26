"""Auto mode: a second model answers the y/N for flagged shell commands.

The agent's TERMINAL commands that pulse_terminal.classify_command flags (they delete files,
rewrite git state, reach outside the project, start a background process, overwrite a file
via redirect) normally wait for a person to type y. In an unattended run nobody does: the
run either hangs on the prompt or the command is declined, including the ones the fix
actually needed. Auto mode hands that one question to a model chosen at startup instead.

Nothing about WHICH commands are flagged changes -- only who answers. The approver is told
it stands in for an absent user, sees the command, why it was flagged, the project folder
and what the agent was trying to do, and answers APPROVE or DENY with a reason. The reason
goes back to the agent either way. If the approver can't be reached or answers something
unreadable, the caller falls back to asking the person (or declining, when there is none).

Choosing it: `pulse run --approver MODEL`, PULSE_APPROVER=MODEL, "approver" in
pulse_config.json (a model string, or {"model", "api_key", "api_base"}), or the question in
interactive setup. MODEL is a litellm model id, e.g. openrouter/anthropic/claude-sonnet-5.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional

APPROVER_ENV = "PULSE_APPROVER"
APPROVER_KEY_ENV = "PULSE_APPROVER_API_KEY"
APPROVER_BASE_ENV = "PULSE_APPROVER_API_BASE"
APPROVER_TIMEOUT_SECONDS = float(os.environ.get("PULSE_APPROVER_TIMEOUT", "90") or 90)

SYSTEM_PROMPT = (
    "You approve or deny shell commands for an automated ML debugging agent (Pulse) while a "
    "training run is going and the user is not there to ask. The command was flagged by a "
    "rule-based classifier as needing the user's OK. You stand in for the user: say yes to "
    "what they would say yes to, and protect what they would want protected.\n\n"
    "APPROVE when the command is plausibly needed for what the agent is doing and anything it "
    "changes is inside the project folder and either regenerable (caches, logs, checkpoints and "
    "outputs the run itself writes, __pycache__, temporary files) or recoverable (the project is "
    "under git and the change can be undone). Installing a missing package into the run's own "
    "environment is fine when the traceback or the task calls for it.\n"
    "DENY when the command could destroy or overwrite work that cannot be regenerated (the user's "
    "source, datasets, notebooks, results outside the run's own outputs), rewrites or discards git "
    "history or uncommitted work, touches files outside the project folder, reads or sends "
    "credentials or private data, uploads project data anywhere, uses sudo, kills processes the "
    "run didn't start, or leaves a process running after the run. When you are unsure, DENY -- "
    "the agent is told why and can find another way.\n\n"
    "Answer with exactly one line: `APPROVE: <short reason>` or `DENY: <short reason>`."
)


@dataclass
class ApproverSettings:
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class Decision:
    approved: bool
    reason: str


class ApproverUnavailable(Exception):
    """The approver could not give a usable answer; the caller falls back to asking."""


def settings_from(config_value=None, agent_model: Optional[str] = None,
                  agent_key: Optional[str] = None) -> Optional[ApproverSettings]:
    """Who approves, or None when auto mode is off. The environment (set by --approver, and
    carried into a fix-triggered restart) wins over the config file. With no key given, a
    model from the agent's own provider reuses the agent's key; otherwise litellm reads the
    provider's usual environment variable (OPENROUTER_API_KEY, ...)."""
    model = os.environ.get(APPROVER_ENV, "").strip()
    api_key = os.environ.get(APPROVER_KEY_ENV, "").strip() or None
    api_base = os.environ.get(APPROVER_BASE_ENV, "").strip() or None
    if not model and config_value:
        if isinstance(config_value, dict):
            model = str(config_value.get("model") or "").strip()
            api_key = api_key or (str(config_value.get("api_key") or "").strip() or None)
            api_base = api_base or (str(config_value.get("api_base") or "").strip() or None)
        else:
            model = str(config_value).strip()
    if not model or model.lower() in ("off", "none", "no", "false", "0"):
        return None
    if not api_key and agent_key and agent_key != "local" and agent_model:
        if model.split("/", 1)[0] == agent_model.split("/", 1)[0]:
            api_key = agent_key
    return ApproverSettings(model=model, api_key=api_key, api_base=api_base)


def build_prompt(command: str, why: str, cwd: str, purpose: str = "") -> str:
    parts = [
        f"Command: {command}",
        f"Flagged because it {why}.",
        f"Project folder (the command runs here): {cwd}",
    ]
    if purpose:
        parts.append(f"What the agent is working on:\n{purpose.strip()[:2000]}")
    parts.append("Your answer, one line: APPROVE: <reason> or DENY: <reason>")
    text = "\n\n".join(parts)
    try:
        from pulse import pulse_supabase
        text = pulse_supabase.scrub_secrets(text)
    except Exception:
        pass
    return text


_VERDICT_RE = re.compile(r"^[\s>*_`#-]*(APPROVE|APPROVED|DENY|DENIED)\b[\s*_`]*[:\-—–]?\s*(.*)$",
                         re.IGNORECASE | re.MULTILINE)


def parse(answer: Optional[str]) -> Decision:
    """The first line that starts with APPROVE or DENY (markdown around it tolerated)."""
    m = _VERDICT_RE.search(answer or "")
    if not m:
        raise ApproverUnavailable(f"unreadable answer: {(answer or '').strip()[:120]!r}")
    word = m.group(1).upper()
    reason = m.group(2).strip().strip("*_` ") or "(no reason given)"
    return Decision(approved=word.startswith("APPROVE"), reason=reason)


def ask(settings: ApproverSettings, command: str, why: str, cwd: str, purpose: str = "",
        completion=None) -> Decision:
    """One decision. Raises ApproverUnavailable on any failure, so the caller can fall back."""
    if completion is None:
        import litellm
        completion = litellm.completion
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(command, why, cwd, purpose)}]
    last_error = "no answer"
    for _attempt in range(2):          # an empty reply from a reasoning model gets one retry
        try:
            response = completion(model=settings.model, messages=messages, max_tokens=4000,
                                  timeout=APPROVER_TIMEOUT_SECONDS, api_key=settings.api_key,
                                  api_base=settings.api_base)
            content = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise ApproverUnavailable(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        if content:
            return parse(content)
        last_error = "empty answer"
    raise ApproverUnavailable(last_error)
