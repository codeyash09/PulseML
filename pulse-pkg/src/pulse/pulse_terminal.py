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
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
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

# Conservative, deliberately over-inclusive checks. False positives (an ordinary command
# getting flagged) mean one extra confirmation prompt -- and a declined command when there
# is no TTY -- so they matter too; false negatives are the thing most worth avoiding. The
# command is split into shell tokens (quotes respected, operators separated) and each
# simple command is judged by the basename of the program it runs, so `/bin/rm x`,
# `ls;rm x` and `xargs rm` are caught while a quoted `"x > 0"` is not a redirect.

# Shell operators, longest first so `2>&1` lexes as `2` `>&` `1` and `&&` is not `&` `&`.
_SHELL_OPERATORS = (
    "&>>", "<<<", "&>", ">>", ">&", ">|", "<<", "<&", "<>", "&&", "||", "|&", ";;",
    "|", "&", ";", "(", ")", "<", ">", "\n", "`",
)
_SEPARATOR_OPERATORS = {"&&", "||", "|", "|&", "&", ";", ";;", "(", ")", "\n", "`"}
_REDIRECT_OUT_OPERATORS = {">", ">>", ">|", "&>", "&>>", ">&"}
_HARMLESS_REDIRECT_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"}

# Programs that run the command given in their arguments (`sudo rm x`, `xargs rm`).
_WRAPPER_COMMANDS = {
    "sudo", "doas", "env", "nohup", "time", "nice", "ionice", "exec", "command", "builtin",
    "stdbuf", "xargs", "timeout", "setsid", "chronic", "unbuffer",
}
_DELETE_COMMANDS = {"rm", "rmdir", "shred", "unlink", "truncate", "srm", "wipe"}
_NETWORK_COMMANDS = {
    "sudo", "su", "doas", "curl", "wget", "ssh", "scp", "sftp", "rsync", "ftp", "nc",
    "ncat", "netcat", "telnet", "aria2c",
}
_PACKAGE_MANAGERS = {
    "conda", "mamba", "micromamba", "apt", "apt-get", "yum", "dnf", "brew", "pacman",
    "apk", "zypper", "gem", "cargo",
}
_PACKAGE_SUBCOMMANDS = {"install", "uninstall", "remove", "update", "upgrade", "create", "add"}
_SENSITIVE_PATH_MARKERS = (
    "~/.ssh", "~/.aws", "~/.gnupg", "~/.kube", "~/.netrc", "~/.docker", "~/.pypirc",
    "~/.git-credentials", "~/.config/gcloud", "$home/.ssh", "$home/.aws", "${home}/.ssh",
    "${home}/.aws", "/.ssh/", "/.aws/",
)
_BACKGROUND_COMMANDS = {"nohup", "disown", "setsid", "systemctl", "daemonize", "crontab", "at"}
_PYTHON_DELETE_RE = re.compile(
    r"shutil\s*\.\s*rmtree|\bos\s*\.\s*(remove|unlink|rmdir|removedirs)\b|\.unlink\s*\(|"
    r"\.rmdir\s*\(|send2trash"
)


def _shell_tokens(command: str) -> "List[tuple[str, bool]]":
    """Split shell text into (token, is_operator) pairs. Quotes and backslashes are
    honoured (a quoted `>` is part of a word), `#` comments are dropped. Never raises:
    an unbalanced quote just ends the last word at the end of the text."""
    tokens: "List[tuple[str, bool]]" = []
    word: List[str] = []
    in_word = False
    i, n = 0, len(command)

    def flush():
        nonlocal word, in_word
        if in_word:
            tokens.append(("".join(word), False))
        word, in_word = [], False

    while i < n:
        ch = command[i]
        if ch == "\\" and i + 1 < n:
            if command[i + 1] != "\n":
                word.append(command[i + 1])
                in_word = True
            i += 2
            continue
        if ch == "'":
            j = command.find("'", i + 1)
            j = n if j < 0 else j
            word.append(command[i + 1:j])
            in_word = True
            i = j + 1
            continue
        if ch == '"':
            j = i + 1
            buf: List[str] = []
            while j < n and command[j] != '"':
                if command[j] == "\\" and j + 1 < n and command[j + 1] in '"\\$`':
                    buf.append(command[j + 1])
                    j += 2
                    continue
                buf.append(command[j])
                j += 1
            word.append("".join(buf))
            in_word = True
            i = j + 1
            continue
        if ch == "#" and not in_word:
            j = command.find("\n", i)
            i = n if j < 0 else j
            continue
        if ch in " \t\r":
            flush()
            i += 1
            continue
        op = next((o for o in _SHELL_OPERATORS if command.startswith(o, i)), None)
        if op is not None:
            flush()
            tokens.append((op, True))
            i += len(op)
            continue
        word.append(ch)
        in_word = True
        i += 1
    flush()
    return tokens


def _short_flags(args: List[str]) -> str:
    """All single-dash short-option letters in args (`-df` -> 'df')."""
    return "".join(a[1:] for a in args if len(a) > 1 and a[0] == "-" and a[1] != "-")


def _strip_wrappers(argv: List[str], flags: Dict[str, bool]) -> List[str]:
    """Drop `VAR=value` prefixes and wrapper programs (`sudo`, `env`, `xargs -0`, ...)
    so argv[0] is the program that really runs. Wrappers still count on their own."""
    argv = list(argv)
    while argv:
        head = argv[0]
        if "=" in head and not head.startswith("=") and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", head):
            argv.pop(0)
            continue
        base = os.path.basename(head).lower()
        if base not in _WRAPPER_COMMANDS:
            break
        _judge_program(base, argv[1:], flags)
        argv.pop(0)
        if base == "timeout":
            while argv and argv[0].startswith("-"):
                argv.pop(0)
            if argv:
                argv.pop(0)             # the duration
        while argv and (argv[0].startswith("-") or (base == "env" and "=" in argv[0])):
            argv.pop(0)
    return argv


def _judge_git(args: List[str], flags: Dict[str, bool]) -> None:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            i += 1
        i += 1
    if i >= len(args):
        return
    sub, rest = args[i], args[i + 1:]
    short = _short_flags(rest)
    destructive = False
    if sub == "reset":
        destructive = "--hard" in rest or "--merge" in rest or "--keep" in rest
    elif sub == "checkout":
        destructive = ("--" in rest or "." in rest or "--force" in rest or "f" in short
                       or "--ours" in rest or "--theirs" in rest)
    elif sub == "restore":
        destructive = not ("--staged" in rest or "S" in short) or "--worktree" in rest or "W" in short
    elif sub == "clean":
        destructive = "--force" in rest or "f" in short or "x" in short
    elif sub == "push":
        destructive = (any(a.startswith("--force") for a in rest) or "f" in short
                       or "--delete" in rest or "--mirror" in rest)
    elif sub in ("rebase", "filter-branch", "filter-repo"):
        destructive = True
    elif sub == "stash":
        destructive = bool(rest) and rest[0] in ("drop", "clear")
    elif sub == "branch":
        destructive = "D" in short or ("--delete" in rest and "--force" in rest) or (
            ("d" in short or "--delete" in rest) and ("f" in short or "--force" in rest))
    if destructive:
        flags["modifies_git_state"] = True


def _judge_program(base: str, args: List[str], flags: Dict[str, bool]) -> None:
    """Classify one program invocation (basename lower-cased, args as typed)."""
    if base in _DELETE_COMMANDS:
        flags["deletes_files"] = True
    elif base == "find" and ("-delete" in args or any(
            a in ("-exec", "-execdir", "-ok", "-okdir") and j + 1 < len(args)
            and os.path.basename(args[j + 1]).lower() in _DELETE_COMMANDS
            for j, a in enumerate(args))):
        flags["deletes_files"] = True
    elif base == "git":
        _judge_git(args, flags)
    elif base == "tee":
        flags["overwrites_via_redirect"] = True
    elif base == "dd" and any(a.startswith("of=") and a[3:] not in _HARMLESS_REDIRECT_TARGETS for a in args):
        flags["overwrites_via_redirect"] = True
    elif base in ("bash", "sh", "zsh", "dash", "ksh", "eval") and args:
        script = None
        if base == "eval":
            script = " ".join(args)
        elif "-c" in args and args.index("-c") + 1 < len(args):
            script = args[args.index("-c") + 1]
        if script:
            for k, v in classify_command(script).items():
                if v:
                    flags[k] = True

    if base in _NETWORK_COMMANDS:
        flags["reaches_outside_workspace"] = True
    pip_args = None
    if re.match(r"^pip[0-9.]*$", base) or base in ("pipx", "uv"):
        pip_args = args[1:] if base == "uv" and args[:1] == ["pip"] else args
    elif re.match(r"^python[0-9.]*$", base) and args[:2] == ["-m", "pip"]:
        pip_args = args[2:]
    if pip_args is not None and any(a in ("install", "uninstall", "download", "add") for a in pip_args[:1]):
        flags["reaches_outside_workspace"] = True
    if base in _PACKAGE_MANAGERS and args and any(a in _PACKAGE_SUBCOMMANDS for a in args[:2]):
        flags["reaches_outside_workspace"] = True
    if base in ("npm", "pnpm", "yarn") and ("-g" in args or "--global" in args or "global" in args):
        flags["reaches_outside_workspace"] = True
    if base in ("chmod", "chown", "chgrp") and ("R" in _short_flags(args) or "r" in _short_flags(args)
                                                  or "--recursive" in args) \
            and any(a.startswith("/") for a in args):
        flags["reaches_outside_workspace"] = True

    if base in _BACKGROUND_COMMANDS:
        flags["launches_persistent_process"] = True
    elif base == "docker" and args[:1] == ["run"] and ("d" in _short_flags(args) or "--detach" in args):
        flags["launches_persistent_process"] = True
    elif base == "screen" and any(a.startswith("-d") or a.startswith("-D") for a in args):
        flags["launches_persistent_process"] = True
    elif base == "tmux" and "d" in _short_flags(args):
        flags["launches_persistent_process"] = True


def classify_command(command: str) -> Dict[str, bool]:
    """Best-effort classification of what a shell command *could* do, used to decide
    whether to ask for confirmation before running it. Never used to silently block or
    rewrite the command -- only to gate it behind a y/n the same way an applied code-fix
    already is."""
    flags = {
        "deletes_files": False,
        "modifies_git_state": False,
        "reaches_outside_workspace": False,
        "launches_persistent_process": False,
        "overwrites_via_redirect": False,
    }
    text = command or ""
    lowered = text.lower()
    if _PYTHON_DELETE_RE.search(text) or "del /s" in lowered or "del /q" in lowered:
        flags["deletes_files"] = True

    tokens = _shell_tokens(text)
    simple: List[List[str]] = [[]]
    idx = 0
    while idx < len(tokens):
        tok, is_op = tokens[idx]
        if not is_op:
            if any(m in tok.lower() for m in _SENSITIVE_PATH_MARKERS) or tok == "/etc" \
                    or tok.startswith("/etc/"):
                flags["reaches_outside_workspace"] = True
            simple[-1].append(tok)
        elif tok == "&":
            flags["launches_persistent_process"] = True
            simple.append([])
        elif tok in _SEPARATOR_OPERATORS:
            simple.append([])
        elif tok in _REDIRECT_OUT_OPERATORS or tok == "<>":
            target = tokens[idx + 1][0] if idx + 1 < len(tokens) and not tokens[idx + 1][1] else None
            if tok == ">&" and target is not None and (target.isdigit() or target == "-"):
                pass                                    # fd duplication, e.g. 2>&1
            elif target is None or target not in _HARMLESS_REDIRECT_TARGETS:
                flags["overwrites_via_redirect"] = True
            idx += 2 if target is not None else 1
            continue
        else:
            # Input redirects / here-docs: skip their operand so it isn't read as a command.
            if idx + 1 < len(tokens) and not tokens[idx + 1][1]:
                idx += 1
        idx += 1

    for argv in simple:
        argv = _strip_wrappers(argv, flags)
        if argv:
            _judge_program(os.path.basename(argv[0]).lower(), argv[1:], flags)
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
        try:
            timeout = float(request.timeout or DEFAULT_TIMEOUT_SECONDS)
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT_SECONDS
        if not math.isfinite(timeout):
            timeout = DEFAULT_TIMEOUT_SECONDS
        timeout = min(max(timeout, 1.0), MAX_TIMEOUT_SECONDS)

        env = dict(os.environ)
        # `python` in a command means the interpreter this run uses, as it would in the user's
        # activated environment. A script started as /path/to/venv/bin/python train.py, with
        # the venv never activated, otherwise sent `python3 -c "import numpy"` to the system
        # interpreter -- and every check the agent ran failed on the import.
        interpreter_dir = os.path.dirname(os.path.abspath(sys.executable)) if sys.executable else ""
        if interpreter_dir:
            path = env.get("PATH", "")
            if path.split(os.pathsep)[0] != interpreter_dir:
                env["PATH"] = interpreter_dir + (os.pathsep + path if path else "")
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
        # Output goes to temporary files, not pipes: a command that backgrounds a child
        # (`python server.py &`) lets the shell exit at once, and the child would otherwise
        # hold the pipe open until the timeout. The command runs in its own process group
        # so a timeout kills everything it started, not just the shell. Always a POSIX
        # shell: the agent writes bash syntax whatever the user's login shell is.
        posix = os.name != "nt"
        shell = None
        if posix:
            shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        try:
            with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
                timed_out = False
                try:
                    proc = subprocess.Popen(
                        request.command,
                        shell=True,
                        cwd=cwd,
                        env=env,
                        stdin=subprocess.PIPE if request.stdin is not None else subprocess.DEVNULL,
                        stdout=out_f,
                        stderr=err_f,
                        executable=shell,
                        start_new_session=posix,
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
                try:
                    proc.communicate(
                        input=request.stdin.encode("utf-8", "replace") if request.stdin is not None else None,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._kill_group(proc)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                duration = time.monotonic() - started
                out_f.seek(0)
                err_f.seek(0)
                out = out_f.read().decode("utf-8", "replace")
                err = err_f.read().decode("utf-8", "replace")
        except OSError as exc:
            duration = time.monotonic() - started
            result = TerminalResult(
                command=request.command, working_directory=cwd, stdout="", stderr="",
                exit_code=None, duration=duration, timed_out=False,
                launch_error=str(exc), is_verification=is_verification,
            )
            self._record(result)
            return result
        stdout, stdout_trunc = _truncate_stream(out)
        stderr, stderr_trunc = _truncate_stream(err)
        result = TerminalResult(
            command=request.command, working_directory=cwd, stdout=stdout, stderr=stderr,
            exit_code=None if timed_out else proc.returncode, duration=duration,
            timed_out=timed_out, stdout_truncated=stdout_trunc, stderr_truncated=stderr_trunc,
            is_verification=is_verification,
        )
        self._record(result)
        return result

    @staticmethod
    def _kill_group(proc: "subprocess.Popen") -> None:
        """Kill the timed-out command and every process it started (its process group)."""
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                return
            except OSError:
                pass
        try:
            proc.kill()
        except OSError:
            pass

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
    default). Returns (command, timeout_or_None). Only a trailing suffix is taken, and
    the rest of the command text is returned exactly as written (pipes, redirects and
    quoting intact; a `--timeout=` belonging to the command itself stays with it)."""
    m = re.match(r"^(.*\S)[ \t]+--timeout=(\S*)\s*$", arg, re.S)
    if not m:
        return arg, None
    try:
        tokens = shlex.split(arg, posix=(os.name != "nt"))
    except ValueError:
        return arg, None
    if not tokens or tokens[-1] != "--timeout=" + m.group(2):
        return arg, None                  # the suffix is inside a quoted string
    try:
        timeout = float(m.group(2))
    except ValueError:
        return arg, None
    if not math.isfinite(timeout) or timeout <= 0:
        return m.group(1), None           # nonsense value: drop it, use the default
    return m.group(1), timeout
