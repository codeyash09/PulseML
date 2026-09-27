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
import threading
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
# Captured output kept on disk per stream while a command runs. Past this the middle is
# dropped (only HEAD_CHARS/TAIL_CHARS are ever shown), so a `cat` of a multi-GB checkpoint
# or 600 s of chatty output can't fill /tmp.
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
_CAPTURE_POLL_SECONDS = 0.25


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


def _pread(f, n: int, offset: int) -> bytes:
    if n <= 0:
        return b""
    if hasattr(os, "pread"):
        return os.pread(f.fileno(), n, offset)
    f.seek(offset)
    return f.read(n)


class _CaptureFile:
    """One output stream's temp file, capped at MAX_CAPTURE_BYTES while the command runs.
    When more than that is on disk, the head (once) and the current tail are saved and the
    file is truncated to zero; the command's later writes land past a hole (sparse), so the
    file's size still counts every byte written and what follows the cut is the newest
    output."""

    def __init__(self, f):
        self.f = f
        self.head: Optional[bytes] = None       # saved at the first cut
        self.tail_at_cut = b""                  # the last bytes before the latest cut
        self.cut_at = 0                         # file offset of the latest cut

    def enforce_cap(self) -> None:
        try:
            size = os.fstat(self.f.fileno()).st_size
            if size - self.cut_at <= MAX_CAPTURE_BYTES:
                return
            if self.head is None:
                self.head = _pread(self.f, HEAD_CHARS * 4, 0)
            self.tail_at_cut = _pread(self.f, TAIL_CHARS * 4, max(self.cut_at, size - TAIL_CHARS * 4))
            self.cut_at = size
            os.ftruncate(self.f.fileno(), 0)
        except (OSError, ValueError):
            pass

    def read(self) -> "tuple[str, bool]":
        """(text, truncated): the whole stream when small, else only its head and tail
        windows -- never the whole file in memory."""
        f = self.f
        size = os.fstat(f.fileno()).st_size
        if self.head is None and size <= MAX_OUTPUT_CHARS * 4:
            f.seek(0)
            return _truncate_stream(f.read().decode("utf-8", "replace"))
        window = TAIL_CHARS * 4
        if self.head is None:
            head_b = _pread(f, HEAD_CHARS * 4, 0)
            tail_b = _pread(f, window, max(0, size - window))
            total = size
        else:
            head_b = self.head
            start = max(self.cut_at, size - window)
            tail_b = (self.tail_at_cut + _pread(f, size - start, start))[-window:]
            total = max(size, self.cut_at)
        head = head_b.decode("utf-8", "replace")[:HEAD_CHARS]
        tail = tail_b.decode("utf-8", "replace")[-TAIL_CHARS:] if TAIL_CHARS > 0 else ""
        omitted = max(0, total - len(head.encode("utf-8")) - len(tail.encode("utf-8")))
        gap = f"\n... [truncated: about {omitted:,} bytes omitted -- re-run with a narrower " \
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
            return _scrub("\n".join(lines))
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
        return _scrub("\n".join(lines))


def _scrub(text: str) -> str:
    """Redact API keys/tokens from what goes back to the model (and from there to logs):
    the command itself stays unrestricted, only its reported output is scrubbed."""
    try:
        from pulse.pulse_supabase import scrub_secrets
        return scrub_secrets(text)
    except Exception:
        return text


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
    "stdbuf", "xargs", "timeout", "setsid", "chronic", "unbuffer", "busybox",
}
# Wrapper options whose VALUE is the next word (`nice -n 10 rm`: '10' is not the program).
_WRAPPER_OPTIONS_WITH_VALUE = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "-T", "--user", "--group",
             "--host", "--prompt", "--close-from", "--chdir", "--role", "--type", "--other-user",
             "--command-timeout"},
    "doas": {"-u", "-C"},
    "env": {"-u", "-C", "-S", "--unset", "--chdir", "--split-string"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "-P", "-u", "--class", "--classdata", "--pid", "--pgid", "--uid"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "xargs": {"-n", "-I", "-L", "-P", "-d", "-s", "-a", "-E", "--max-args", "--max-lines",
              "--max-procs", "--delimiter", "--max-chars", "--arg-file", "--eof", "--replace"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "time": {"-f", "-o", "--format", "--output"},
    "exec": {"-a"},
}
# Shell reserved words that can stand before the command they introduce
# (`for f in x; do rm "$f"; done`, `if ...; then rm x; fi`, `{ rm x; }`, `! rm x`).
_SHELL_RESERVED_WORDS = {
    "do", "then", "else", "elif", "if", "while", "until", "{", "}", "!", "done", "fi",
    "time", "coproc",
}
_SHELL_PROGRAMS = {"bash", "sh", "zsh", "dash", "ksh", "mksh", "ash", "fish"}
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
# systemctl subcommands that only look (everything else starts/stops/enables something).
_SYSTEMCTL_READ_ONLY = {
    "status", "show", "cat", "help", "list-units", "list-unit-files", "list-timers",
    "list-sockets", "list-jobs", "list-dependencies", "list-machines", "is-active",
    "is-enabled", "is-failed", "is-system-running", "get-default", "show-environment",
}
# Killing processes / stopping the machine reaches past the project (the user's own
# training run, everything the user owns).
_PROCESS_KILL_COMMANDS = {"kill", "pkill", "killall", "skill", "shutdown", "reboot", "poweroff",
                          "halt"}
_PYTHON_DELETE_RE = re.compile(
    r"shutil\s*\.\s*rmtree|\bos\s*\.\s*(remove|unlink|rmdir|removedirs)\b|\.unlink\s*\(|"
    r"\.rmdir\s*\(|send2trash"
)
# The same for the other interpreters' one-liners (perl -e, node -e, ruby -e).
_SCRIPT_DELETE_RE = re.compile(
    r"\bunlink(Sync)?\b|\brm(dir)?Sync\b|\brmtree\b|\bremove_tree\b|\bFile\s*\.\s*(delete|unlink)\b|"
    r"\bDir\s*\.\s*(delete|rmdir|unlink)\b|\bFileUtils\s*\.\s*(rm|rm_r|rm_rf|rm_f|remove\w*|rmdir|rmtree)\b|"
    r"\bfs(\.promises)?\s*\.\s*(rm|rmdir)\s*\("
)
# A call that hands a string (or an argv list) to a shell/exec: os.system('rm -rf x'),
# subprocess.run(['rm', ...]), system("rm x") in perl/ruby, execSync('rm x') in node.
_SHELL_OUT_CALL_RE = re.compile(
    r"\b(?:os\s*\.\s*(?:system|popen|exec\w*|spawn\w*)|subprocess\s*\.\s*\w+|"
    r"system|exec|execSync|execFileSync|spawn|spawnSync|execFile|qx|Popen|run|call|check_call|"
    r"check_output)\s*\("
)
# Interpreters and the options that take one-line code (`python -c CODE`, `perl -e CODE`).
_INTERPRETER_CODE_OPTIONS = {
    "python": "c", "perl": "eE", "ruby": "e", "node": "ep", "nodejs": "ep", "deno": "",
}


def _substitution_end(text: str, start: int) -> int:
    """Index just past the `)` closing the `$(` at text[start] (end of text if none)."""
    depth, j, n = 0, start + 1, len(text)
    while j < n:
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "'" and depth > 1:
            k = text.find("'", j + 1)
            j = n if k < 0 else k + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return n


def _shell_tokens(command: str, substitutions: "Optional[List[str]]" = None) -> "List[tuple[str, bool]]":
    """Split shell text into (token, is_operator) pairs. Quotes and backslashes are
    honoured (a quoted `>` is part of a word), `#` comments are dropped. Never raises:
    an unbalanced quote just ends the last word at the end of the text. The text of every
    `$(...)` / backtick substitution inside double quotes -- which still RUNS -- is
    appended to `substitutions` when a list is given."""
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
                if command.startswith("$(", j):
                    end = _substitution_end(command, j + 1)
                    if substitutions is not None:
                        substitutions.append(command[j + 2:end - 1] if command[end - 1:end] == ")"
                                             else command[j + 2:end])
                    buf.append(command[j:end])
                    j = end
                    continue
                if command[j] == "`":
                    k = command.find("`", j + 1)
                    k = n if k < 0 else k
                    if substitutions is not None:
                        substitutions.append(command[j + 1:k])
                    buf.append(command[j:k + 1])
                    j = k + 1
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
        if head in _SHELL_RESERVED_WORDS:
            argv.pop(0)
            continue
        if "=" in head and not head.startswith("=") and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", head):
            argv.pop(0)
            continue
        base = os.path.basename(head).lower()
        if base not in _WRAPPER_COMMANDS:
            break
        _judge_program(base, argv[1:], flags)
        argv.pop(0)
        takes_value = _WRAPPER_OPTIONS_WITH_VALUE.get(base, set())
        while argv and (argv[0].startswith("-") or (base == "env" and "=" in argv[0])):
            opt = argv.pop(0)
            if opt == "--":
                break
            if base == "env" and opt in ("-S", "--split-string") and argv:
                _merge_flags(flags, classify_command(argv[0]))      # env -S 'rm -rf x'
            elif base == "env" and opt.startswith("--split-string="):
                _merge_flags(flags, classify_command(opt.split("=", 1)[1]))
            if opt in takes_value and argv:
                argv.pop(0)
        if base == "timeout" and argv:
            argv.pop(0)                 # the duration
    return argv


def _merge_flags(flags: Dict[str, bool], other: Dict[str, bool]) -> None:
    for k, v in other.items():
        if v:
            flags[k] = True


def _judge_argv(argv: List[str], flags: Dict[str, bool]) -> None:
    """Judge one simple command: wrappers and reserved words off, then its program."""
    argv = _strip_wrappers(argv, flags)
    if argv:
        _judge_program(os.path.basename(argv[0]).lower(), argv[1:], flags)


def _code_argument(base: str, args: List[str]) -> "Optional[str]":
    """The one-line code of an interpreter invocation (`python -c CODE`, `perl -ne CODE`,
    `node --eval CODE`), or None."""
    family = re.sub(r"[0-9.]+$", "", base)
    letters = _INTERPRETER_CODE_OPTIONS.get(family)
    if letters is None:
        return None
    skip_value = False
    for j, a in enumerate(args):
        if skip_value:
            skip_value = False
            continue
        if family in ("node", "nodejs") and a in ("--eval", "--print", "-e", "-p"):
            return args[j + 1] if j + 1 < len(args) else None
        if family in ("node", "nodejs") and (a.startswith("--eval=") or a.startswith("--print=")):
            return a.split("=", 1)[1]
        if family == "python" and a in ("-W", "-X", "--check-hash-based-pycs"):
            skip_value = True
            continue
        if len(a) > 1 and a[0] == "-" and a[1] != "-" and letters:
            cluster = a[1:]
            if family != "python" and not cluster.isalpha():
                continue                        # an attached value: -MFile::Path, -pi.bak
            pos = next((k for k, c in enumerate(cluster) if c in letters), None)
            if pos is None:
                continue
            if family == "python" and cluster[pos + 1:]:
                return cluster[pos + 1:]        # `-cCODE`
            return args[j + 1] if j + 1 < len(args) else None
        if not a.startswith("-"):
            return None                 # a script file: its code isn't on the command line
    return None


def _reads_code_from_stdin(base: str, args: List[str]) -> bool:
    """`python`, `python -`, `perl` with no script: the program text comes from stdin
    (a pipe or a redirect in the same command line)."""
    family = re.sub(r"[0-9.]+$", "", base)
    if family not in _INTERPRETER_CODE_OPTIONS:
        return False
    if _code_argument(base, args) is not None:
        return False
    positional = [a for a in args if not a.startswith("-") or a == "-"]
    if family == "python" and "-m" in args:
        return False
    return not positional or positional[0] == "-"


def _string_literals(code: str, start: int) -> "tuple[List[str], int]":
    """The quoted string literals inside the call whose `(` is at code[start], and where
    the call ends. A small scanner, not a parser: good enough for a one-liner."""
    lits: List[str] = []
    depth, j, n = 0, start, len(code)
    while j < n:
        ch = code[j]
        if ch in "'\"`":
            k = j + 1
            buf: List[str] = []
            while k < n and code[k] != ch:
                if code[k] == "\\" and k + 1 < n:
                    buf.append(code[k + 1])
                    k += 2
                    continue
                buf.append(code[k])
                k += 1
            lits.append("".join(buf))
            j = k + 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth <= 0:
                return lits, j + 1
        j += 1
    return lits, n


def _judge_code(code: str, flags: Dict[str, bool]) -> None:
    """An interpreter's one-line code: its own delete calls, and every shell command it
    hands to os.system / subprocess / system / execSync (the literal text or argv list)."""
    if _PYTHON_DELETE_RE.search(code) or _SCRIPT_DELETE_RE.search(code):
        flags["deletes_files"] = True
    for m in _SHELL_OUT_CALL_RE.finditer(code):
        lits, _end = _string_literals(code, m.end() - 1)
        if lits:
            _merge_flags(flags, classify_command(" ".join(lits)))


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
        # `+main` force-pushes that ref, `:old-branch` deletes the remote branch.
        refspecs = [a for a in rest if not a.startswith("-")][1:]
        destructive = (any(a.startswith("--force") for a in rest) or "f" in short or "d" in short
                       or "--delete" in rest or "--mirror" in rest or "--prune" in rest
                       or any(r.startswith("+") or r.startswith(":") for r in refspecs))
    elif sub == "commit":
        destructive = "--amend" in rest
    elif sub == "switch":
        destructive = ("--discard-changes" in rest or "--force" in rest or "f" in short
                       or "C" in short or "--force-create" in rest)
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
    elif base == "find":
        if "-delete" in args:
            flags["deletes_files"] = True
        for j, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                inner: List[str] = []
                for b in args[j + 1:]:
                    if b in (";", "+"):
                        break
                    inner.append(b)
                _judge_argv(inner, flags)
    elif base in ("mv", "cp", "ln") and "/dev/null" in args:
        # mv f /dev/null deletes f; cp /dev/null f and ln -sf /dev/null f empty/replace f.
        flags["deletes_files" if base == "mv" else "overwrites_via_redirect"] = True
    elif base == "git":
        _judge_git(args, flags)
    elif base == "tee":
        flags["overwrites_via_redirect"] = True
    elif base == "dd" and any(a.startswith("of=") and a[3:] not in _HARMLESS_REDIRECT_TARGETS for a in args):
        flags["overwrites_via_redirect"] = True
    elif (base in _SHELL_PROGRAMS or base == "eval") and args:
        script = None
        if base == "eval":
            script = " ".join(args)
        else:
            # `-c`, or any short-option cluster with c in it (`-lc`, `-ec`, `-xc`): the
            # script is the next word that isn't an option.
            for j, a in enumerate(args):
                if len(a) > 1 and a[0] == "-" and a[1] != "-" and "c" in a[1:]:
                    script = next((b for b in args[j + 1:] if not b.startswith("-")), None)
                    break
        if script:
            _merge_flags(flags, classify_command(script))
    else:
        code = _code_argument(base, args)
        if code:
            _judge_code(code, flags)

    if base in _NETWORK_COMMANDS:
        flags["reaches_outside_workspace"] = True
    if base in _PROCESS_KILL_COMMANDS and not (
            base == "kill" and any(a in ("-0", "-l", "-L", "--list") for a in args[:1])):
        flags["reaches_outside_workspace"] = True       # `kill -0 PID` / `kill -l` only look
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

    if base == "systemctl":
        sub = next((a for a in args if not a.startswith("-")), "list-units")
        if sub not in _SYSTEMCTL_READ_ONLY:
            flags["launches_persistent_process"] = True
    elif base == "crontab":
        # `crontab -l` only lists; -e/-r/-i, a file, or stdin (no args) installs or removes.
        rest = list(args)
        if "-u" in rest:                                # `-u USER` picks whose table
            k = rest.index("-u")
            del rest[k:k + 2]
        if rest != ["-l"]:
            flags["launches_persistent_process"] = True
    elif base in _BACKGROUND_COMMANDS:
        flags["launches_persistent_process"] = True
    elif base == "docker" and args[:1] == ["run"] and ("d" in _short_flags(args) or "--detach" in args):
        flags["launches_persistent_process"] = True
    elif base == "screen" and any(a.startswith("-d") or a.startswith("-D") for a in args):
        flags["launches_persistent_process"] = True
    elif base == "tmux" and "d" in _short_flags(args):
        flags["launches_persistent_process"] = True


def _resolve_substituted_program(words: List[str]) -> "Optional[str]":
    """`$(which rm)` / `$(command -v rm)` / `$(type -P rm)` -> 'rm'; None when the
    substitution computes the program some other way."""
    words = [w for w in words if w]
    if len(words) >= 2 and words[0] in ("which", "whereis", "realpath", "readlink"):
        return next((w for w in words[1:] if not w.startswith("-")), None)
    if len(words) >= 3 and words[0] in ("command", "type") and words[1].startswith("-"):
        return next((w for w in words[2:] if not w.startswith("-")), None)
    return None


# Stand-in argv[0] for a program word that is computed (`$(...) args`): can't be judged.
_COMPUTED_PROGRAM = "\0computed-program"


def has_heredoc(command: str) -> bool:
    """Does the (one-line) command use a heredoc (`<<WORD`, `<<-'EOF'`) outside quotes?
    A `<<` inside a quoted argument (python3 -c "print(1 << n)") is a bit shift, and
    `$((1 << 3))` has no delimiter word."""
    tokens = _shell_tokens(command or "")
    depth = 0
    arithmetic: List[int] = []          # paren depths where a `((` / `$((` started
    for i, (tok, is_op) in enumerate(tokens):
        if not is_op:
            continue
        if tok == "(":
            depth += 1
            if i + 1 < len(tokens) and tokens[i + 1] == ("(", True) and not arithmetic:
                arithmetic.append(depth)
        elif tok == ")":
            if arithmetic and arithmetic[-1] == depth:
                arithmetic.pop()
            depth -= 1
        elif tok == "<<" and not arithmetic and i + 1 < len(tokens) and not tokens[i + 1][1]:
            if re.match(r"^-?[A-Za-z_]\w*$", tokens[i + 1][0]):
                return True
    return False


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
    if "del /s" in lowered or "del /q" in lowered:
        flags["deletes_files"] = True

    substitutions: List[str] = []
    tokens = _shell_tokens(text, substitutions)
    for sub in substitutions:                   # "$(rm -rf x)" inside double quotes still runs
        _merge_flags(flags, classify_command(sub))

    def at_program_position() -> bool:
        return all(w in _SHELL_RESERVED_WORDS or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w)
                   for w in simple[-1])

    def matching_close(open_idx: int, open_tok: str) -> int:
        depth = 0
        for k in range(open_idx, len(tokens)):
            t, op = tokens[k]
            if not op:
                continue
            if open_tok == "`":
                if t == "`" and k > open_idx:
                    return k
            elif t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
                if depth == 0:
                    return k
        return len(tokens)

    simple: List[List[str]] = [[]]
    computed_close: Dict[int, str] = {}     # token index closing a computed program -> argv[0]
    backticks = 0
    idx = 0
    while idx < len(tokens):
        tok, is_op = tokens[idx]
        if idx in computed_close:
            simple.append([computed_close.pop(idx)])
            if tok == "`":
                backticks += 1
            idx += 1
            continue
        # The program word itself is a substitution: `$(which rm) -rf x`, `` `which rm` x ``.
        open_idx = None
        if (not is_op and tok == "$" and idx + 1 < len(tokens) and tokens[idx + 1] == ("(", True)
                and at_program_position()):
            open_idx = idx + 1
        elif is_op and tok == "`" and backticks % 2 == 0 and at_program_position():
            open_idx = idx
        if open_idx is not None:
            close = matching_close(open_idx, tokens[open_idx][0])
            inner = [t for t, op in tokens[open_idx + 1:close] if not op]
            computed_close[close] = _resolve_substituted_program(inner) or _COMPUTED_PROGRAM
            if tok == "$":
                idx += 1                        # the "(" below starts the inner command
                continue
        if is_op and tok == "`":
            backticks += 1
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

    stdin_interpreter = False
    for argv in simple:
        argv = _strip_wrappers(argv, flags)
        if not argv:
            continue
        if argv[0] == _COMPUTED_PROGRAM:
            # The program is computed at run time and can't be judged: ask.
            flags["reaches_outside_workspace"] = True
            continue
        base = os.path.basename(argv[0]).lower()
        _judge_program(base, argv[1:], flags)
        stdin_interpreter = stdin_interpreter or _reads_code_from_stdin(base, argv[1:])
    if stdin_interpreter and (_PYTHON_DELETE_RE.search(text) or _SCRIPT_DELETE_RE.search(text)):
        # `echo "import os; os.remove('x')" | python`: the code is somewhere in this line.
        flags["deletes_files"] = True
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
                captures = (_CaptureFile(out_f), _CaptureFile(err_f))
                done = threading.Event()

                def cap_output():
                    while not done.wait(_CAPTURE_POLL_SECONDS):
                        for c in captures:
                            c.enforce_cap()

                capper = threading.Thread(target=cap_output, daemon=True, name="pulse-terminal-cap")
                capper.start()
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
                except BaseException:
                    # Ctrl-C (or anything else) while waiting: the command runs in its own
                    # session, so the terminal's SIGINT never reached it -- don't leave it
                    # (and everything it started) running with nobody to reap it.
                    self._kill_group(proc)
                    try:
                        proc.wait(timeout=5)
                    except BaseException:
                        pass
                    raise
                finally:
                    done.set()
                    capper.join()
                duration = time.monotonic() - started
                out, stdout_trunc = captures[0].read()
                err, stderr_trunc = captures[1].read()
        except OSError as exc:
            duration = time.monotonic() - started
            result = TerminalResult(
                command=request.command, working_directory=cwd, stdout="", stderr="",
                exit_code=None, duration=duration, timed_out=False,
                launch_error=str(exc), is_verification=is_verification,
            )
            self._record(result)
            return result
        result = TerminalResult(
            command=request.command, working_directory=cwd, stdout=out, stderr=err,
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
