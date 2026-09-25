"""
pulse_terminal -- shared agent terminal execution layer.

Both Pulse Code (`pulse_code.py`) and the debugging agent (`pulse_cli.py`'s
`PulseCLI`) let the model ask for a real shell command via a `TERMINAL:`
directive, the same way it already asks for `GREP:`/`VIEW:`/`REPL:` and the
rest of the extended toolset (see `_NEW_DIRECTIVE_RES` / `_apply_new_directives`
in pulse_cli.py). This module is the one place that actually runs those
commands, so the CLI and the debugging UI share one implementation instead of
each shelling out on its own.

Design:

  * `TerminalRequest` / `TerminalResult` -- plain dataclasses describing a
    command and what happened when it ran. Nothing here leaks a raw
    `subprocess.Popen`/`CompletedProcess` object to callers -- the model
    (and the rest of Pulse) only ever sees these structured results.
  * `TerminalExecutor` -- runs a `TerminalRequest` with a hard timeout,
    captures stdout/stderr/exit code/duration, truncates oversized output
    (never truncates it away entirely -- head+tail is kept, with a clear
    note), and keeps a bounded, inspectable history of every command it
    has run (`.history`) for `/terminal-log`-style introspection and for
    the debugging tooling described in the terminal-access spec.
  * `classify_command` -- a best-effort, conservative classifier for
    commands that can delete files, rewrite git state, reach outside the
    workspace, or start something persistent. It does not block anything;
    callers (the `TERMINAL:` directive handlers in pulse_cli.py) decide
    what to do with the classification -- normally: ask for confirmation
    the same way an applied code-fix already does, unless the session has
    already been told to skip confirmations (`-y` / `/review off`).

Nothing here silently expands what a command *can* do: it is exactly the
command the model wrote, run through the OS shell, in the workspace's
working directory unless the model gave a different one. The safety layer
is about *warning and gating*, never about rewriting the command.
"""
from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------------------
# Limits -- same spirit as the hard caps already used for GREP/VIEW output (see
# pulse_cli.py's _GREP_MAX_MATCHES / _VIEW_MAX_LINES): keep the model's context bounded without throwing away
# the diagnostics that make the tool useful in the first place.
# ---------------------------------------------------------------------------------------
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_TIMEOUT_SECONDS = 600.0
MAX_OUTPUT_CHARS = 8_000          # per stream (stdout, stderr), after head/tail truncation
HEAD_CHARS = 2_000                # how much of the *start* of a truncated stream to keep
TAIL_CHARS = MAX_OUTPUT_CHARS - HEAD_CHARS
MAX_HISTORY_ENTRIES = 200         # bounded ring buffer -- this is a debugging aid, not a log file


def _truncate_stream(text: str, limit: int = MAX_OUTPUT_CHARS) -> "tuple[str, bool]":
    """Head+tail truncation for one output stream. Returns (text, was_truncated).

    Dumping the first N characters of a 100,000-line test run is nearly
    always useless (it is almost never the interesting part -- the failure
    is usually at the end), and dumping only the last N characters can hide
    an early fatal error that explains everything after it. Keeping both
    ends, with a clearly labelled gap, is the closest a fixed character
    budget gets to "give the agent what it actually needs".
    """
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:] if TAIL_CHARS > 0 else ""
    omitted = len(text) - len(head) - len(tail)
    gap = f"\n... [truncated: {omitted:,} characters omitted -- re-run with a narrower " \
          f"command (e.g. pipe to head/tail/grep) to see a different slice] ...\n"
    return head + gap + tail, True


# ---------------------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass
class TerminalRequest:
    """What the agent asked to run. Deliberately thin -- the command itself is free-form
    shell text the model composes; Pulse does not parse it into a fixed vocabulary."""
    command: str
    working_directory: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    env: Optional[Dict[str, str]] = None
    # Interactive input isn't wired to a live TTY here (the model has no way to watch a
    # prompt and answer it), but a caller can still feed fixed stdin -- e.g. answering a
    # single known y/n prompt -- without pretending the process is fully interactive.
    stdin: Optional[str] = None


@dataclasses.dataclass
class TerminalResult:
    """Structured outcome of a TerminalRequest -- this, not a raw CompletedProcess, is what
    gets serialized back into the model's context."""
    command: str
    working_directory: str
    stdout: str
    stderr: str
    exit_code: Optional[int]      # None if the process never produced one (timeout / launch failure)
    duration: float
    timed_out: bool
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    launch_error: Optional[str] = None   # set if the command could not even be started
    request_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: float = dataclasses.field(default_factory=time.time)
    is_verification: bool = False

    @property
    def ok(self) -> bool:
        return self.launch_error is None and not self.timed_out and self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def render(self) -> str:
        """The exact block fed back into the model's context -- clearly labelled and
        distinct from ordinary conversational text (see the terminal-access spec's
        'AGENT CONTEXT' requirement)."""
        lines = [
            "TERMINAL EXECUTION RESULT",
            f"Command: {self.command}",
            f"Working directory: {self.working_directory}",
        ]
        if self.launch_error:
            lines.append(f"Could not start: {self.launch_error}")
            return "\n".join(lines)
        lines.append(f"Exit code: {self.exit_code if self.exit_code is not None else '(none -- see below)'}")
        lines.append(f"Timed out: {'true' if self.timed_out else 'false'}")
        lines.append(f"Duration: {self.duration:.2f}s")
        stdout = self.stdout or "(empty)"
        stderr = self.stderr or "(empty)"
        lines.append(f"STDOUT{' (truncated)' if self.stdout_truncated else ''}:\n{stdout}")
        lines.append(f"STDERR{' (truncated)' if self.stderr_truncated else ''}:\n{stderr}")
        if self.timed_out:
            lines.append(
                f"NOTE: the process did not finish within the {self.duration:.0f}s limit and was "
                "terminated. Output above is whatever it had produced up to that point. Treat this "
                "as \"did not complete\", not as success or failure of the command itself."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Safety classification -- advisory, not enforcement. Callers decide what a
# classification means (normally: ask for confirmation).
# ---------------------------------------------------------------------------------------

# Conservative, deliberately over-inclusive keyword/pattern checks. False positives (an
# ordinary command getting flagged) just mean one extra confirmation prompt; false
# negatives are the thing worth avoiding.
_DESTRUCTIVE_FILE_TOKENS = (
    "rm -rf", "rm -r", "rm -f", " rm ", "rmdir", "shred ", "del /s", "del /q",
)
_DESTRUCTIVE_GIT_TOKENS = (
    "git reset --hard", "git checkout -- ", "git checkout --force", "git clean -f",
    "git clean -x", "git push --force", "git push -f", "git rebase", "git filter-branch",
    "git stash drop", "git branch -D",
)
_OUTSIDE_WORKSPACE_TOKENS = (
    "sudo ", "chmod -r /", "chown -r /", " /etc/", " ~/.ssh", "curl ", "wget ",
    "pip install", "pip uninstall", "npm install -g", "npm uninstall -g",
)
_BACKGROUND_PERSISTENT_TOKENS = (
    " & ", "nohup ", "disown", "systemctl ", "docker run -d", "setsid ",
)
_OVERWRITE_REDIRECT_TOKENS = (">", ">>")


def classify_command(command: str) -> Dict[str, bool]:
    """Best-effort classification of what a shell command *could* do, used to decide
    whether to ask for confirmation before running it. Never used to silently block or
    rewrite the command -- only to gate it behind a y/n the same way an applied code-fix
    already is."""
    lowered = f" {command.strip().lower()} "
    flags = {
        "deletes_files": any(tok in lowered for tok in _DESTRUCTIVE_FILE_TOKENS),
        "modifies_git_state": any(tok in lowered for tok in _DESTRUCTIVE_GIT_TOKENS),
        "reaches_outside_workspace": any(tok in lowered for tok in _OUTSIDE_WORKSPACE_TOKENS),
        "launches_persistent_process": any(tok in lowered for tok in _BACKGROUND_PERSISTENT_TOKENS),
        "overwrites_via_redirect": any(tok in command for tok in _OVERWRITE_REDIRECT_TOKENS),
    }
    return flags


def is_destructive(command: str) -> bool:
    return any(classify_command(command).values())


def describe_classification(flags: Dict[str, bool]) -> str:
    labels = {
        "deletes_files": "deletes files",
        "modifies_git_state": "rewrites git history/state",
        "reaches_outside_workspace": "reaches outside the project (network, sudo, or system paths)",
        "launches_persistent_process": "starts a background/persistent process",
        "overwrites_via_redirect": "overwrites a file via a shell redirect",
    }
    hit = [labels[k] for k, v in flags.items() if v]
    return ", ".join(hit)


# ---------------------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------------------

class TerminalExecutor:
    """Runs TerminalRequests as real subprocesses and returns TerminalResults. One instance
    is shared per agent session (Pulse Code session or debugging PulseCLI instance) so its
    working-directory default and history are consistent across a whole conversation.
    """

    def __init__(self, default_cwd: Optional[str] = None):
        self.default_cwd = os.path.abspath(default_cwd) if default_cwd else os.getcwd()
        # Bounded ring buffer of every command actually run this session -- the traceable
        # record the terminal-access spec asks for (command, timestamp, exit code,
        # duration, stdout/stderr, verification flag/pass-fail). Not printed to the normal
        # UI; available to callers that want to show /terminal-log or sync it to the cloud
        # session the way agent Q&A turns already are.
        self.history: List[TerminalResult] = []

    def set_cwd(self, path: str) -> None:
        self.default_cwd = os.path.abspath(path)

    def run(self, request: TerminalRequest, *, is_verification: bool = False) -> TerminalResult:
        cwd = os.path.abspath(request.working_directory) if request.working_directory else self.default_cwd
        timeout = min(max(float(request.timeout or DEFAULT_TIMEOUT_SECONDS), 1.0), MAX_TIMEOUT_SECONDS)

        env = dict(os.environ)
        if request.env:
            env.update(request.env)

        if not os.path.isdir(cwd):
            result = TerminalResult(
                command=request.command, working_directory=cwd, stdout="", stderr="",
                exit_code=None, duration=0.0, timed_out=False,
                launch_error=f"working directory does not exist: {cwd}",
            )
            self._record(result)
            return result

        started = time.monotonic()
        try:
            proc = subprocess.run(
                request.command,
                shell=True,
                cwd=cwd,
                env=env,
                input=request.stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable=None if os.name == "nt" else (os.environ.get("SHELL") or "/bin/bash"),
            )
            duration = time.monotonic() - started
            stdout, stdout_trunc = _truncate_stream(proc.stdout or "")
            stderr, stderr_trunc = _truncate_stream(proc.stderr or "")
            result = TerminalResult(
                command=request.command, working_directory=cwd, stdout=stdout, stderr=stderr,
                exit_code=proc.returncode, duration=duration, timed_out=False,
                stdout_truncated=stdout_trunc, stderr_truncated=stderr_trunc,
                is_verification=is_verification,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace") if exc.stdout else ""
            err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace") if exc.stderr else ""
            stdout, stdout_trunc = _truncate_stream(out)
            stderr, stderr_trunc = _truncate_stream(err)
            result = TerminalResult(
                command=request.command, working_directory=cwd, stdout=stdout, stderr=stderr,
                exit_code=None, duration=duration, timed_out=True,
                stdout_truncated=stdout_trunc, stderr_truncated=stderr_trunc,
                is_verification=is_verification,
            )
        except (OSError, ValueError) as exc:
            duration = time.monotonic() - started
            result = TerminalResult(
                command=request.command, working_directory=cwd, stdout="", stderr="",
                exit_code=None, duration=duration, timed_out=False,
                launch_error=str(exc), is_verification=is_verification,
            )
        self._record(result)
        return result

    def _record(self, result: TerminalResult) -> None:
        self.history.append(result)
        if len(self.history) > MAX_HISTORY_ENTRIES:
            del self.history[: len(self.history) - MAX_HISTORY_ENTRIES]

    # -- introspection --------------------------------------------------------------
    def recent(self, n: int = 10) -> List[TerminalResult]:
        return self.history[-n:]

    def summary_line(self, result: TerminalResult) -> str:
        """One line for the normal (non-verbose) UI -- see the spec's 'USER EXPERIENCE'
        section: understandable without being noisy."""
        if result.launch_error:
            return f"[terminal] {result.command}  ->  could not start: {result.launch_error}"
        if result.timed_out:
            return f"[terminal] {result.command}  ->  timed out after {result.duration:.0f}s"
        status = f"exit {result.exit_code}"
        return f"[terminal] {result.command}  ->  {status}  ({result.duration:.2f}s)"


def parse_inline_timeout(arg: str) -> "tuple[str, Optional[float]]":
    """Allow the model to optionally suffix a directive with ` --timeout=<seconds>` (a
    convenience, not a required part of the protocol -- omitting it just uses the
    default). Returns (command, timeout_or_None)."""
    try:
        tokens = shlex.split(arg, posix=(os.name != "nt"))
    except ValueError:
        return arg, None
    timeout = None
    kept: List[str] = []
    for tok in tokens:
        if tok.startswith("--timeout="):
            try:
                timeout = float(tok.split("=", 1)[1])
            except ValueError:
                pass
            continue
        kept.append(tok)
    if timeout is None:
        return arg, None
    try:
        rebuilt = shlex.join(kept) if hasattr(shlex, "join") else " ".join(kept)
    except Exception:
        rebuilt = arg
    return rebuilt, timeout