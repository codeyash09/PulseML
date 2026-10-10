"""
The Pulse app: one screen for everything Pulse does in a terminal.

    pulse                 setup, then the agent. /monitor picks a run on this machine.
    pulse code [paths]    the same app, with those files in focus
    pulse watch <run>     the same app, opened on that run

It starts with the setup Pulse always had (account, workspace, agent model, key) and then
stays on one full-screen layout:

  * HOME is the coding agent: type a request and it plans, reads the project with its tools
    and proposes a diff. Slash commands do the rest -- /monitor (or /runs) lists every run
    on this machine to pick from, /run starts a script under Pulse, /agent switches model.
  * DEBUG is what the screen becomes when a run is open: the run on the left (status,
    step, every value with its curve, the detectors' findings), the agent on the right.
    Questions go to the agent together with the run's evidence; it can read the code,
    run commands and propose a fix. Its reasoning and every tool call are in the
    transcript, folded to a line each -- Ctrl+O unfolds them.

Nothing underneath is reimplemented. The agent is Pulse Code's (`pulse_code.run_turn`),
the run is watched by the console's brain (`pulse_console.Console`), and setup is
`PulseCLI.interactive_setup`. This module is the *host* those run inside: while the app is
up, what they print lands in the transcript, what they ask is answered in the input line,
and a stage they start is the spinner (see `pulse_ui.set_host`). So a fix, a diff, a y/N,
an /agent switch or an OpenRouter sign-in all work here exactly as they do in a plain
terminal, without knowing where they are.

One thread draws and reads keys; everything that can block -- a model call, a command, a
question waiting for its answer -- runs on a job thread.
"""
from __future__ import annotations

import builtins
import collections
import contextlib
import ctypes
import getpass
import io
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import pulse_tui as tui
from . import pulse_ui as _ui
from . import pulse_settings as _settings
from . import pulse_detect as detect

def _debug_prompt(restart: str, status: str) -> str:
    return (
        "\n\nYOU ARE ATTACHED TO A LIVE TRAINING RUN\n"
        "The person opened a run in the Pulse app and is looking at it with you. Each request comes "
        "with EVIDENCE FROM THE RUN: its step, every tracked value's curve, tensors, the detectors' "
        "findings and recent events. Use it and quote the numbers. Diagnose before you conclude: read "
        "the code with the tools, check what you suspect. A question about the run is a question -- "
        "answer it in words, with no code change. Change code only when asked to fix something, or "
        "when the person agrees to a fix you proposed. (A request Pulse makes by itself when the run "
        "crashes or its checks find something serious asks you to fix it: that is the person's "
        "standing instruction.) The run keeps executing the code it started with. After an edit, "
        f"decide whether to run it again with {restart}: do when the fix only takes effect in a "
        "fresh run or you need to see it work (after a crash, almost always); don't when the run is "
        "healthy and the change can wait for its next start -- and say which you chose. "
        f"{status} gives the run's latest numbers whenever you need them again.\n"
    )


# the text pipeline's directives, and the native tool loop's tool names (each loop is told
# about the tools it really has)
DEBUG_PROMPT = _debug_prompt("a RESTART: line", "RUNSTATUS:")
DEBUG_PROMPT_NATIVE = _debug_prompt("restart_run", "run_status")

HOME_COMMANDS: List[Tuple[str, str]] = [
    ("/monitor", "pick a run on this machine and open it beside the agent"),
    ("/change", "the same, keeping every run already watched"),
    ("/runs", "the same list: every run on this machine"),
    ("/run", "start a script under Pulse and watch it: /run train.py --epochs 3"),
    ("/agent", "switch AI provider/model (or sign in with OpenRouter)"),
    ("/files", "what the agent sees in full, and how much it can search"),
    ("/add", "put files or folders in focus"),
    ("/drop", "take a file out of focus"),
    ("/review", "show a diff and ask before applying: /review on|off"),
    ("/undo", "undo the latest change"),
    ("/log", "the change history"),
    ("/cloud", "sign-in, workspace and sync status"),
    ("/config", "your settings, remembered between starts: /config mouse off, /config agent"),
    ("/copy", "copy the agent's last answer to the clipboard (/copy 2: the one before)"),
    ("/mouse", "clicks open folded lines; /mouse off gives the mouse back to select text"),
    ("/help", "every command"),
    ("/exit", "leave Pulse"),
]
DEBUG_COMMANDS: List[Tuple[str, str]] = [
    ("/findings", "what the detectors currently believe, worst first"),
    ("/curve", "one value's history: /curve val_loss"),
    ("/vars", "every value being tracked"),
    ("/audit", "have the agent audit the whole run now"),
    ("/audits", "scheduled audits by the agent: /audits on|off"),
    ("/trace", "what feeds a variable and what it feeds: /trace loss"),
    ("/source", "show the training script"),
    ("/pause", "pause the run"),
    ("/resume", "resume a paused run"),
    ("/stop", "stop the training run"),
    ("/restart", "run the script again with the current code (stopping it first if it is still going)"),
    ("/output", "the last lines the run printed (runs started with /run)"),
    ("/interval", "sample faster or slower: /interval 0.5"),
    ("/quiet", "stop announcing findings (/loud resumes)"),
    ("/home", "back to the agent for anything else; the run stays watched (findings, crashes)"),
    ("/change", "look at another run; the one you leave stays watched"),
    ("/close", "stop watching the open run"),
]

_STATUS_STYLE = {"live": "accent", "stalled": "bold", "crashed": "red"}
# what makes up the run the app is looking at; a parked run keeps them in App.parked
_WATCH_FIELDS = ("console", "session", "_monitor", "runlog", "_status", "_rate", "_rate_at",
                 "_stall_said", "_paused_here")
_STATUS_PATIENCE_SECONDS = 10.0   # run_status asked again sooner than this waits for progress
_AUTO_CRASH_TURNS = 3       # crashes in a row the agent starts on by itself, before the person is asked
_RUN_LOG_DIR = "app-runs"

# How a run is launched: `python -m pulse` puts the working directory first on sys.path, so
# a stray pulse.py there (a home folder is a likely place) shadows the package and the
# launch dies at once. This boot drops the working directory from the path, then runs the
# package the way -m would.
_BOOT = ("import os, runpy, sys; cwd = os.getcwd(); "
         "sys.path[:] = [p for p in sys.path if p not in ('', '.', cwd, os.path.abspath(cwd))]; "
         "sys.argv[0] = 'pulse'; runpy.run_module('pulse', run_name='__main__', alter_sys=True)")


def launch_argv(script_argv: List[str]) -> List[str]:
    """The command that starts `script_argv` under Pulse, in stream mode, from any folder."""
    return [sys.executable, "-c", _BOOT, "run", "--stream", "--again", *script_argv]


class _Question:
    """Something a hosted pipeline asked; the job thread waits on it."""

    def __init__(self, label: str, secret: bool = False, options: Optional[List[Any]] = None,
                 hotkeys: Optional[Dict[str, Any]] = None) -> None:
        self.label = label
        self.secret = secret
        self.options = options
        self.hotkeys = {str(k).lower(): v for k, v in (hotkeys or {}).items()}
        self.answer: Any = ""
        self.cancelled = False
        self.done = threading.Event()


class CommandQueue:
    """Prompts sent from the web dashboard to one Debug_Sessions row (the Commands table).

    Every couple of seconds the row's pending commands are claimed. One that cannot be done
    from afar is answered at once with why (`refuse` returns the reason, or None); the rest
    are handed to `run`, one at a time, and what it returns -- or why it failed -- goes back
    to the dashboard. `row_id` is read each time: a row is created in the background."""

    POLL_EVERY = 2.0

    def __init__(self, cli: Any, row_id: Callable[[], Optional[str]], run: Callable[[str], str],
                 refuse: Optional[Callable[[str], Optional[str]]] = None,
                 on_error: Optional[Callable[[str], None]] = None) -> None:
        from . import pulse_supabase as cloud
        self.cli = cli
        self._cloud = cloud
        self._row_id = row_id
        self._run = run
        self._refuse = refuse
        self._on_error = on_error
        self._stop = threading.Event()
        self._poller: Optional[threading.Thread] = None
        self._active: Optional[threading.Thread] = None
        self._backlog: "collections.deque[Dict[str, Any]]" = collections.deque()
        self._lock = threading.Lock()
        self._error_said = False

    def start(self) -> None:
        if self._poller is None:
            self._poller = threading.Thread(target=self._listen, daemon=True, name="pulse-commands")
            self._poller.start()

    def close(self) -> None:
        """Stop listening. Commands claimed but not started are failed with the reason, so the
        dashboard does not show them as running for ever."""
        self._stop.set()
        with self._lock:
            left, self._backlog = list(self._backlog), collections.deque()
        for command in left:
            self._finish(command, "failed", "Pulse stopped watching this run before it got to this command.")
        poller = self._poller
        if poller is not None and poller is not threading.current_thread():
            poller.join(timeout=1.0)

    def _listen(self) -> None:
        while not self._stop.wait(self.POLL_EVERY):
            row = self._row_id()
            if not row:
                continue
            try:
                claimed = self._cloud.claim_commands(row, getattr(self.cli, "user_id", "") or "")
            except Exception as exc:
                self._report(f"Could not read the dashboard's commands: {exc}")
                continue
            for command in claimed:
                text = str(command.get("command") or "").strip()
                if text.startswith((self._cloud.QUESTION_PREFIX.strip(), self._cloud.LIVE_PREFIX)):
                    continue                     # a row this runner opened (a question, the live feed)
                if not text or len(text) > 8000:
                    self._finish(command, "failed", "A command is 1 to 8000 characters.")
                    continue
                try:
                    refused = self._refuse(text) if self._refuse else None
                except Exception as exc:
                    refused = f"{type(exc).__name__}: {exc}"
                if refused is not None:
                    self._finish(command, "failed", refused)
                elif self._stop.is_set():
                    self._finish(command, "failed", "Pulse stopped watching this run before it got to this command.")
                else:
                    with self._lock:
                        self._backlog.append(command)
            self._next()

    def _next(self) -> None:
        with self._lock:
            if self._stop.is_set() or not self._backlog:
                return
            if self._active is not None and self._active.is_alive():
                return
            command = self._backlog.popleft()
            self._active = threading.Thread(target=self._execute, args=(command,), daemon=True,
                                            name="pulse-command")
            self._active.start()

    def _execute(self, command: Dict[str, Any]) -> None:
        text = str(command.get("command") or "").strip()
        try:
            result = self._run(text)
            self._finish(command, "completed", result or "Done.")
        except Exception as exc:
            self._finish(command, "failed", str(exc) or type(exc).__name__)
        finally:
            self._next()

    def _finish(self, command: Dict[str, Any], status: str, result: str) -> None:
        try:
            self._cloud.finish_command(str(command.get("id") or ""), status, result)
        except Exception as exc:
            self._report(f"Could not send a command's result to the dashboard: {exc}")

    def _report(self, message: str) -> None:
        if self._error_said or self._on_error is None:
            return
        self._error_said = True
        try:
            self._on_error(message)
        except Exception:
            pass



class DashboardQuestion:
    """A question asked here, put on the run's dashboard page too: whoever answers first --
    at the machine or on the dashboard -- answers it. `on_answer` gets the dashboard's answer
    (the option's "#index", or the text typed); `close` says how it ended, if it was here."""

    POLL_EVERY = 1.5

    def __init__(self, row_id: str, project_id: str, user_id: str, question: Dict[str, Any],
                 on_answer: Callable[[str], None]) -> None:
        from . import pulse_supabase as cloud
        self._cloud = cloud
        self._ids = (row_id, project_id, user_id)
        self._question = question
        self._on_answer = on_answer
        self.id: Optional[str] = None
        self.answered_there = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pulse-dashboard-question")

    def start(self) -> "DashboardQuestion":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            self.id = self._cloud.open_question(*self._ids, self._question)
        except Exception:
            return
        while self.id and not self._stop.wait(self.POLL_EVERY):
            try:
                row = self._cloud.read_command(self.id)
            except Exception:
                continue
            status = (row or {}).get("status")
            if status == "completed" and not self._stop.is_set():
                self.answered_there = True
                self._on_answer(str((row or {}).get("result") or ""))
                return
            if status in ("completed", "failed", None):
                return

    def close(self, status: str, result: str) -> None:
        """The question is over here: its row says how (unless the dashboard answered it).
        In the background -- the row may still be being opened."""
        self._stop.set()
        if self.answered_there:
            return

        def finish() -> None:
            self._thread.join(15.0)
            if self.id and not self.answered_there:
                try:
                    self._cloud.finish_command(self.id, status, result)
                except Exception:
                    pass

        threading.Thread(target=finish, daemon=True, name="pulse-dashboard-question-close").start()


class LiveFeed:
    """What the app is doing right now, for the dashboard: a Commands row of its own
    ("pulse:live", left processing) whose result is rewritten as it changes -- the agent's
    reasoning, tool calls and words as they stream, what is running, the run's step and
    latest values. At most about once a second, and every 15 s at least, so the page knows
    the app is still there. It follows the row in hand (the open run's, else the app's own)."""

    EVERY = 0.8
    HEARTBEAT = 15.0
    RETRY = 30.0

    def __init__(self, app: "App") -> None:
        from . import pulse_supabase as cloud
        self._cloud = cloud
        self.app = app
        self._target: Optional[Tuple[str, str, str]] = None
        self.id: Optional[str] = None
        self._last = ""
        self._sent_at = 0.0
        self._retry_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pulse-live-feed")

    def start(self) -> "LiveFeed":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._end()

    def _loop(self) -> None:
        while not self._stop.wait(self.EVERY):
            try:
                self.tick()
            except Exception:
                self._retry_at = time.monotonic() + self.RETRY

    def tick(self) -> None:
        now = time.monotonic()
        if now < self._retry_at:
            return
        target = self.app._live_target()
        if target != self._target:
            self._end()
            self._target = target
        if target is None:
            return
        if self.id is None:
            self.id = self._cloud.open_runner_row(*target, self._cloud.LIVE_PREFIX)
            self._last = ""
            if self.id is None:
                self._retry_at = now + self.RETRY
                return
        state = self.app._live_state()
        body = json.dumps(state, sort_keys=True, default=str)
        if body == self._last and now - self._sent_at < self.HEARTBEAT:
            return
        self._cloud.update_runner_row(self.id, json.dumps(dict(state, t=time.time()), default=str))
        self._last, self._sent_at = body, now

    def _end(self) -> None:
        """The row is done with: the page stops showing it as live."""
        row, self.id = self.id, None
        if row:
            try:
                self._cloud.finish_command(row, "completed", json.dumps({"ended": True, "t": time.time()}))
            except Exception:
                pass


class RunLog:
    """One Debug_Sessions row for a run the app watches, so it shows on the web dashboard
    like a run started with `pulse run`: a first telemetry entry about the environment,
    metric snapshots, the detectors' findings and status changes as incidents, crashes as
    error tracebacks, the agent's turns about the run, and uptime. Everything is sent in
    the background and never blocks the screen; without a signed-in account there is
    nothing to send and this is inert."""

    TELEMETRY_EVERY = 60.0
    FLUSH_EVERY = 60.0

    def __init__(self, cli: Any, session: Dict[str, Any], workdir: str,
                 on_command: Optional[Callable[[str], str]] = None,
                 on_response: Optional[Callable[[str], Optional[str]]] = None,
                 on_error: Optional[Callable[[str], None]] = None) -> None:
        from . import pulse_supabase as cloud

        self.cli = cli
        self._cloud = cloud
        self.session = session
        self.workdir = workdir
        self.id: Optional[str] = None
        self.enabled = bool(getattr(cli, "user_id", None))
        self._lock = threading.Lock()
        self._fields: Dict[str, Any] = {"agent_logs": [], "error_tracebacks": [], "telemetry": [],
                                        "incidents": [], "uptime_seconds": 0}
        self._dirty: set = set()
        self._seen_findings: set = set()
        self._seen_events = 0
        self._status: Optional[str] = None
        self._last_telemetry = 0.0
        self._last_flush = time.monotonic()
        self._worker: Optional[threading.Thread] = None
        self._closed = False
        # prompts sent to this run from the web dashboard
        self.commands: Optional[CommandQueue] = (
            CommandQueue(cli, lambda: self.id, on_command, refuse=on_response, on_error=on_error)
            if on_command is not None else None)

    # ------------------------------------------------------------------ what gets recorded

    def start(self) -> None:
        if not self.enabled:
            return
        from . import pulse_supabase as cloud
        first = {"t": time.time(), "mode": "stream", "script": self.session.get("script"),
                 "pulse_session": self.session.get("session_id"), "pid": self.session.get("pid")}
        try:
            first.update(cloud.collect_environment_info() or {})
        except Exception:
            pass
        with self._lock:
            self._fields["telemetry"].append(first)
            self._dirty.add("telemetry")
        self._spawn(create=True)
        if self.commands is not None:
            self.commands.start()

    def tick(self, state: Dict[str, Any], status: str, events: List[Dict[str, Any]], session: Dict[str, Any]) -> None:
        """Called a few times a minute with the brain's current state."""
        if not self.enabled or self._closed:
            return
        now = time.monotonic()
        force = False
        with self._lock:
            if status != self._status:
                if self._status is not None:
                    self._incident(f"run_{status}", f"The run is {status}.", status=status)
                    force = True
                self._status = status
            for finding in state.get("findings") or []:
                key = (getattr(finding, "check", ""), getattr(finding, "variable", ""))
                if key in self._seen_findings:
                    continue
                self._seen_findings.add(key)
                severity = str(getattr(finding, "severity", "")).lower()
                if severity in ("warning", "critical", "error"):
                    self._incident("finding", getattr(finding, "message", str(finding)), severity=severity,
                                   check=key[0], variable=key[1], step=state.get("step"))
                    force = True
            for event in events[self._seen_events:]:
                if event.get("event") == "crash":
                    trace = event.get("traceback") or event.get("exception") or "crash"
                    self._fields["error_tracebacks"].append(str(trace)[-8000:])
                    self._dirty.add("error_tracebacks")
                    self._incident("crash", str(event.get("exception") or "the run crashed"))
                    force = True
                elif event.get("event") in ("stop_requested", "stopped"):
                    self._incident("stopped", str(event.get("reason") or "the run was stopped"))
                    force = True
            self._seen_events = len(events)
            if force or now - self._last_telemetry >= self.TELEMETRY_EVERY or not self._last_telemetry:
                self._last_telemetry = now
                snapshot: Dict[str, Any] = {
                    "t": time.time(), "step": state.get("step") or 0, "status": status,
                    "gaps": state.get("gaps") or 0, "finished": bool(state.get("finished")),
                    "findings": [
                        {"severity": getattr(item, "severity", "info"),
                         "message": getattr(item, "message", str(item)),
                         "check": getattr(item, "check", ""),
                         "variable": getattr(item, "variable", ""),
                         "step": state.get("step")}
                        for item in (state.get("findings") or [])
                    ],
                    "tensors": state.get("tensors") or {},
                }
                for name, history in (state.get("histories") or {}).items():
                    if history:
                        snapshot[name] = history[-1]
                self._fields["telemetry"].append(snapshot)
                self._dirty.add("telemetry")
            started = session.get("started")
            if started:
                try:
                    end = time.time() if status in ("live", "stalled") else float(session.get("last_seen") or time.time())
                    uptime = int(max(0.0, end - float(started)))
                    if uptime != self._fields["uptime_seconds"]:
                        self._fields["uptime_seconds"] = uptime
                        self._dirty.add("uptime_seconds")
                except (TypeError, ValueError):
                    pass
        if force or now - self._last_flush >= self.FLUSH_EVERY:
            self._spawn()

    def agent_turn(self, question: str, answer: str, fix_applied: Optional[Dict[str, Any]] = None,
                   who: Optional[str] = None) -> None:
        """`who` asked: "you" (typed here), "dashboard" (sent from the web), "pulse" (a turn
        Pulse started itself -- a crash, a finding, an audit)."""
        if not self.enabled or self._closed:
            return
        with self._lock:
            entry: Dict[str, Any] = {"t": time.time(), "question": question, "answer": answer}
            if who:
                entry["who"] = who
            if fix_applied:
                entry["fix_applied"] = fix_applied
                self._incident("fix_applied", str(fix_applied.get("explanation") or "code edited"),
                               files=fix_applied.get("files"))
            self._fields["agent_logs"].append(entry)
            self._dirty.add("agent_logs")
        self._spawn()

    def incident(self, kind: str, summary: str, **extra: Any) -> None:
        if not self.enabled or self._closed:
            return
        with self._lock:
            self._incident(kind, summary, **extra)
        self._spawn()

    def _incident(self, kind: str, summary: str, **extra: Any) -> None:
        entry: Dict[str, Any] = {"t": time.time(), "kind": kind, "summary": summary}
        entry.update({k: v for k, v in extra.items() if v is not None})
        self._fields["incidents"].append(entry)
        self._dirty.add("incidents")

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        if self.commands is not None:
            self.commands.close()
        self._spawn(wait=5.0)

    # ------------------------------------------------------------------ sending

    def _spawn(self, create: bool = False, wait: float = 0.0) -> None:
        worker = self._worker
        if worker is not None and worker.is_alive():
            if wait:
                worker.join(wait)
            return
        self._worker = threading.Thread(target=self._send, args=(create,), daemon=True, name="pulse-run-log")
        self._worker.start()
        if wait:
            self._worker.join(wait)

    def _send(self, create: bool) -> None:
        cloud = self._cloud
        dirty: set = set()
        try:
            if create and self.id is None:
                sha = None
                try:
                    sha = cloud.current_git_commit_sha(self.workdir)
                except Exception:
                    pass
                self.id = cloud.create_debug_session(getattr(self.cli, "project_id", None),
                                                     getattr(self.cli, "user_id", None), git_commit_sha=sha)
            if not self.id:
                return
            with self._lock:
                dirty, self._dirty = self._dirty, set()
                body: Dict[str, Any] = {}
                for name in dirty:
                    value = self._fields[name]
                    body[name] = ([cloud.encode_entry(e) for e in value] if isinstance(value, list) else value)
            if body:
                cloud.patch_debug_session(self.id, body)
            self._last_flush = time.monotonic()
        except Exception:                        # offline, a 5xx: kept, sent next time
            with self._lock:
                self._dirty |= dirty


class _Capture(io.TextIOBase):
    """sys.stdout / sys.stderr while the app is up: what is written goes to the transcript."""

    encoding = "utf-8"

    def __init__(self, app: "App") -> None:
        super().__init__()
        self._app = app

    def write(self, text: str) -> int:               # type: ignore[override]
        if text:
            self._app.feed(str(text))
        return len(text or "")

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def writable(self) -> bool:
        return True


class App:
    def __init__(self, cli: Any, root: str) -> None:
        from . import pulse_cli
        self.cli = cli
        self.home_root = root
        self.home_focus = list(getattr(cli, "focus", []) or [])
        self.view = tui.View()
        self.lock = threading.RLock()
        self.done = False
        self.exit_code = 0
        self.dirty = True
        self._partial = ""
        self._last_blank = True
        self._pane_w = 80
        self._edit_pending: Optional[tui.Entry] = None   # the change whose fate is still being decided
        self.hosting: Any = None                  # the in-process run's monitor, under auto_track()
        self._screen: Any = None                  # the screen, while the loop runs
        self._hovered: Optional[tui.Entry] = None # the entry under the mouse
        self._sel: Optional[Dict[str, Any]] = None # a selection being dragged or just made
        self._frame: List[str] = []               # the frame last drawn (what a selection copies)
        try:
            from . import pulse_supabase as cloud
            if _settings.is_set("mouse"):
                self.mouse = _settings.on("mouse")
            else:                                  # what /mouse saved before there were settings
                self.mouse = cloud.load_cached_profile().get("app_mouse", "on") != "off"
        except Exception:
            self.mouse = True
        self.run_output: "collections.deque[str]" = collections.deque(maxlen=400)   # what that run printed
        self._crashes_seen: set = set()            # sessions whose crash the agent was started on
        self._crash_pending: Optional[str] = None  # a crash that came while the agent was busy
        self._problem_pending: Optional[List[Any]] = None   # serious findings that came while it was busy
        self._problems_seen: set = set()           # (session, check, variable) the agent was started on
        self._auto_turn_session: Optional[str] = None   # the run an automatic turn is about, while it runs
        self._stall_said = False                   # the open run's stall was announced
        self._paused_here = False                  # /pause sent from here: a stall is expected
        self._auto_crash_turns = 0                 # crash turns started without the person typing
        self.auto_fix_limit: Optional[int] = _AUTO_CRASH_TURNS   # None: no limit (headless)
        self._run_partial = ""
        self.up = threading.Event()               # set once the screen is up and output is captured
        self._height = 30
        self._question: Optional[_Question] = None
        self._question_lock = threading.Lock()
        self._draft = ""
        self._hushed: Dict[int, List[str]] = {}      # thread id -> what it printed while hushed
        self._live_thinking: Optional[tui.Entry] = None
        self._live_answer: Optional[tui.Entry] = None
        self._think_started = 0.0
        self._streamed_reasoning = False
        self._job: Optional[threading.Thread] = None
        self._remote_pending: "collections.deque[Dict[str, Any]]" = collections.deque()
        self._remote_active: Optional[Dict[str, Any]] = None
        self._home_commands: Optional[CommandQueue] = None   # the dashboard's prompts to the app's own row
        self._live: Optional[LiveFeed] = None                # what the app is doing, streamed to the dashboard
        self._job_from = 0
        self._last_ctrl_c = 0.0
        self._ui_thread = threading.current_thread()
        # the run that is open (DEBUG), if any
        self.console: Any = None
        self.session: Optional[Dict[str, Any]] = None
        self.background = False                      # the run is watched, its pane is not shown
        self.runlog: Optional[RunLog] = None         # the run's row on the dashboard
        self._orig_sync: Any = None
        # runs still watched but not the open one: session id -> what _WATCH_FIELDS hold for it
        self.parked: Dict[str, Dict[str, Any]] = {}
        self._parked_at = 0.0
        self._monitor: Any = None                    # a py-spy sampler this app started
        self.audits = _settings.on("audits")
        self.launched: Dict[str, Dict[str, Any]] = {}    # session id -> what /run started
        self._rate: Optional[float] = None
        self._rate_at: Optional[Tuple[float, int]] = None
        self._side_at = 0.0
        self._status = "live"
        self._pc = pulse_cli
        self._saved: Dict[str, Any] = {}
        self._refresh_header()

    # ================================================================== the transcript

    def _add(self, entry: tui.Entry) -> None:
        with self.lock:
            self._flush_partial()
            self.view.entries.append(entry)
            self._last_blank = False
            self.dirty = True

    def note(self, text: str) -> None:
        self._add(tui.Entry("note", text))

    def trace_output(self, text: str) -> None:
        """/trace's tree (pulse_console hands it here): its own entry, drawn as a tree."""
        self._add(tui.Entry("trace", text))

    def error(self, text: str) -> None:
        self._add(tui.Entry("error", text))

    def _flush_partial(self) -> None:
        if self._partial:
            line, self._partial, self.view.partial = self._partial, "", ""
            self._commit(line)

    def _commit(self, line: str) -> None:
        line = line.rstrip()
        if not tui._SGR_RE.sub("", line).strip():
            if self._last_blank:
                return                               # runs of blank lines collapse to one
            self._last_blank = True
            line = ""
        else:
            self._last_blank = False
        entries = self.view.entries
        if entries and entries[-1].kind == "text":
            entries[-1].text += "\n" + line
            entries[-1].touch()
        elif line:
            entries.append(tui.Entry("text", line))

    def feed(self, text: str) -> None:
        """Captured output. Lines are committed as they complete; a line still being
        written (no newline yet) is shown live. A carriage return restarts the line, which
        is how spinners and progress lines redraw themselves."""
        with self.lock:
            held = self._hushed.get(threading.get_ident())
            if held is not None:
                held.append(text)
                return
            if self.hosting is not None and not threading.current_thread().name.startswith("pulse-"):
                # the run's own output (the script's threads, under auto_track()): it goes
                # under the run's figures in the pane, not into the conversation
                pieces = (self._run_partial + tui.clean(text)).split("\n")
                for line in pieces[:-1]:
                    line = tui._SGR_RE.sub("", line.rsplit("\r", 1)[-1]).rstrip()
                    if line.strip():
                        self.run_output.append(line)
                rest = pieces[-1]
                self._run_partial = rest.rsplit("\r", 1)[-1] if "\r" in rest else rest
                self.dirty = True
                return
            pieces = (self._partial + tui.clean(text)).split("\n")
            for line in pieces[:-1]:
                self._commit(line.rsplit("\r", 1)[-1])
            rest = pieces[-1]
            self._partial = rest.rsplit("\r", 1)[-1] if "\r" in rest else rest
            self.view.partial = self._partial
            self.dirty = True

    # ================================================================== the host (see pulse_ui)

    def width(self) -> int:
        return self._pane_w

    @contextlib.contextmanager
    def hush(self):
        """While tools run, their own progress lines ("-> /grep lr") are not shown: the
        transcript gets one entry for the calls instead (see tool()). If a tool stops to
        ask something, what it printed first is shown after all -- it is the context of
        the question (the command a y/N is about)."""
        ident = threading.get_ident()
        with self.lock:
            outer = self._hushed.get(ident)
            held = self._hushed[ident] = []
        try:
            yield held
        finally:
            with self.lock:
                if outer is None:
                    self._hushed.pop(ident, None)
                else:
                    self._hushed[ident] = outer

    def _ask(self, question: _Question) -> Any:
        if threading.current_thread() is self._ui_thread:
            raise _ui.Unavailable("a question cannot be asked from the drawing thread")
        with self._question_lock:
            return self._ask_one(question)

    def _ask_one(self, question: _Question) -> Any:
        """Present one question at a time: all hosted prompts share the same input field."""
        if self._edit_pending is not None:
            # a question while a change is on the table ("Apply this change?") is about that
            # change: open its diff so the person sees what they are answering
            with self.lock:
                self._edit_pending.open = True
                self._edit_pending.touch()
        with self.lock:
            held = self._hushed.get(threading.get_ident())
            if held:
                text, held[:] = "".join(held), []
                self._hushed.pop(threading.get_ident())
                try:
                    self.feed(text)
                finally:
                    self._hushed[threading.get_ident()] = held
            self._flush_partial()
            self._question = question
            view = self.view
            self._draft = view.editor.take(remember=False)
            view.question, view.secret, view.scroll = question.label, question.secret, 0
            if question.options is not None:
                view.options = [(str(o.label), str(o.detail or ""), str(o.tag or "")) for o in question.options]
                view.option_ids = list(range(len(view.options)))
            self.dirty = True
        posted = self._post_question(question)       # on the dashboard too: the first answer wins
        question.done.wait()
        with self.lock:
            view = self.view
            view.question, view.secret, view.options, view.option_ids = "", False, None, []
            view.editor.take(remember=False)
            view.editor.set(self._draft)
            self._question = None
            self.dirty = True
        self._settle_question(question, posted)
        if question.cancelled:
            raise KeyboardInterrupt
        return question.answer

    def _live_target(self) -> Optional[Tuple[str, str, str]]:
        """(row, project, user) of the dashboard row in hand: the open run's, else the app's own."""
        cli = self.cli
        row = (getattr(self.runlog, "id", None) if self.runlog is not None else None) \
            or getattr(cli, "debug_session_id", None)
        user, project = getattr(cli, "user_id", None), getattr(cli, "project_id", None)
        return (str(row), str(project), str(user)) if row and user and project else None

    @staticmethod
    def _entry_words(entry: tui.Entry, limit: int = 1500) -> str:
        if entry.kind in ("tool", "edit"):
            text = "\n".join(entry.calls) + (("\n" + entry.text) if entry.text else "")
        else:
            text = entry.text
        text = tui._SGR_RE.sub("", str(text or "")).strip()
        # the thinking streams at its end; anything else is read from its start
        return ("…" + text[-limit:] if len(text) > limit else text) if entry.kind == "thinking" else \
            (text[:limit] + "…" if len(text) > limit else text)

    def _live_state(self) -> Dict[str, Any]:
        """What the dashboard's live view shows: what the job in hand has shown so far (its
        reasoning streaming in), what is running, and the run's step and latest values."""
        with self.lock:
            view = self.view
            busy = bool(view.busy)
            activity: List[Dict[str, str]] = []
            if busy:
                for entry in view.entries[self._job_from:]:
                    words = self._entry_words(entry)
                    if words:
                        item = {"kind": entry.kind, "text": words}
                        if entry.kind == "trace":
                            item["ansi"] = tui.clean(entry.text)[:1500]
                        if getattr(entry, "live", False):
                            item["live"] = "1"
                        activity.append(item)
                partial = tui._SGR_RE.sub("", view.partial or "").strip()
                if partial:
                    activity.append({"kind": "text", "text": partial[-800:], "live": "1"})
            state: Dict[str, Any] = {
                "busy": busy, "activity": activity[-30:],
                "doing": tui._SGR_RE.sub("", str(view.live or "")).strip() if busy else "",
                "asking": self._question is not None,
            }
            console, status = self.console, self._status
        if console is not None:
            run: Dict[str, Any] = {"status": status}
            try:
                with console._brain_lock:
                    run["step"] = console.brain.step
                    values = {}
                    for name, history in list(console.brain.histories.items())[:16]:
                        last = history[-1] if history else None
                        if isinstance(last, (int, float)) and not isinstance(last, bool) and math.isfinite(last):
                            values[name] = last
                    run["values"] = values
            except Exception:
                pass
            state["run"] = run
        return state

    def _start_live_feed(self) -> None:
        if self._live is None and getattr(self.cli, "user_id", None):
            self._live = LiveFeed(self).start()

    def _post_question(self, question: _Question) -> Optional[DashboardQuestion]:
        """Put `question` on the dashboard page of the run in hand (or of the app's own row):
        an answer there answers it here. Not a secret (a key, a password), never."""
        target = self._live_target()
        if getattr(question, "secret", False) or target is None:
            return None
        row, project, user = target
        options = getattr(question, "options", None)

        def answered(text: str) -> None:
            with self.lock:
                if question.done.is_set():
                    return
                if options is not None:
                    match = re.match(r"#(\d+)", text.strip())
                    index = int(match.group(1)) if match else -1
                    if not 0 <= index < len(options):
                        return
                    question.answer = index
                else:
                    question.answer = text
                question.from_dashboard = True     # type: ignore[attr-defined]
                question.done.set()

        label = tui._SGR_RE.sub("", str(question.label or "")).strip()
        payload: Dict[str, Any] = {
            "label": label[:3000] or "Pulse is asking you to choose:",
            "options": [tui._SGR_RE.sub("", str(o.label))[:300] for o in options] if options else None}
        with self.lock:                              # what was said just before it: the command, why
            recent = [self._entry_words(e, 600) for e in self.view.entries[-8:]
                      if e.kind not in ("thinking", "user")]
        context = "\n".join(w for w in recent if w)[-2500:]
        if context:
            payload["context"] = context
        edit = self._edit_pending
        if edit is not None and edit.output:             # "Apply this change?": the change itself
            diff = tui._SGR_RE.sub("", str(edit.output))
            payload["detail"] = diff[:6000] + ("\n…" if len(diff) > 6000 else "")
        try:
            return DashboardQuestion(str(row), str(project), str(user), payload, answered).start()
        except Exception:
            return None

    def _settle_question(self, question: _Question, posted: Optional[DashboardQuestion]) -> None:
        if getattr(question, "from_dashboard", False):
            self.note("Answered on the dashboard.")
            return
        if posted is None:
            return
        if question.cancelled:
            posted.close("failed", "No longer asked: cancelled at the machine.")
            return
        answer = question.answer
        if question.options is not None:
            shown = str(question.options[answer].label) if isinstance(answer, int) and 0 <= answer < len(
                question.options) else "nothing chosen"
        else:
            shown = str(answer or "") or "(Enter)"
        posted.close("completed", f"Answered at the machine: {tui._SGR_RE.sub('', shown)}")

    def ask(self, label: str, secret: bool = False, placeholder: Optional[str] = None, **_ignored: Any) -> str:
        question = _Question(label + (f"  [{placeholder}]" if placeholder else ""), secret=secret)
        answer = self._ask(question)
        shown = (_ui._g("mask") * min(len(answer), 12)) if secret and answer else (answer or "↵")
        self.note(f"{label}  {shown}")
        return answer

    def choose(self, options: List[Any], initial: int = 0, hotkeys: Optional[Dict[str, Any]] = None,
               title: str = "", **_ignored: Any) -> Any:
        question = _Question(title, options=list(options), hotkeys=hotkeys)
        with self.lock:
            self.view.option_index = max(0, min(initial, len(options) - 1))
        answer = self._ask(question)
        if isinstance(answer, int) and 0 <= answer < len(options):
            self.note(f"{_ui._g('sel')} {options[answer].label}")
        return answer

    def stage_start(self, label: str) -> None:
        with self.lock:
            self._flush_partial()
            self.view.live, self.view.live_since, self.dirty = label, time.monotonic(), True

    def stage_end(self, label: str, ok: bool, seconds: float) -> None:
        # Nothing is added when a stage fails: whatever ran it reports why, in its own words.
        with self.lock:
            self.view.live, self.dirty = "", True

    def say(self, text: str) -> None:
        """The agent speaking to the person mid-turn (see pulse_ui.message)."""
        self._add(tui.Entry("say", text))

    # ------------------------------------------------------------------ what the agent may do with runs

    def run_actions(self) -> Dict[str, Callable[..., str]]:
        """Run control for the agent (its start_run / stop_run / restart_run / run_status tools,
        and the RUN:/STOP:/RESTART:/RUNSTATUS: directives). Each returns text for the model."""
        return {"start_run": self._agent_start_run, "stop_run": self._agent_stop_run,
                "restart_run": self._agent_restart_run, "run_status": self._agent_run_status}

    def _agent_start_run(self, script: str = "", args: str = "") -> str:
        script = str(script or "").strip()
        if not script:
            return "start_run needs the script to run, e.g. train.py (and its arguments, if any)."
        before = (self.session or {}).get("session_id")
        self._launch(f"{script} {str(args or '').strip()}".strip())
        session = self.session or {}
        if self.console is None or session.get("session_id") in (None, before):
            return (f"The run did not start, or Pulse saw no training step from it; the lines above "
                    f"say what happened. (Run it with run_command only to see an error quickly: a "
                    f"training run started that way is killed at the command timeout.)")
        return (f"Started {os.path.basename(session.get('script') or script)} under Pulse (session "
                f"{session.get('session_id')}). It is now open beside you: run_status gives its latest "
                f"numbers and findings, and the person sees it on their screen. It keeps running after "
                f"this turn ends.")

    def _agent_stop_run(self) -> str:
        if self.console is None:
            return "No run is open. Nothing was stopped."
        from . import pulse_stream as stream
        name = os.path.basename((self.session or {}).get("script") or "the run")
        if not _ui.confirm(f"The agent wants to stop {name}. Stop it?", default=False):
            return "The person did not allow stopping the run."
        self.console.send_control(stream.CONTROL_STOP, reason="asked by the agent")
        return f"Asked {name} to stop."

    def _agent_restart_run(self) -> str:
        session = self.session or {}
        if self.console is None:
            return "No run is open. start_run starts one."
        before = session.get("session_id")
        refused = self._restart()
        if refused:
            return refused
        after = (self.session or {}).get("session_id")
        if after and after != before:
            return (f"Restarted with the current code: the new run (session {after}) is open beside you. "
                    "run_status shows how it is doing -- check it before you call the fix done.")
        return "The restart did not produce a new run; the lines above say what happened (the log's last lines)."

    def _agent_run_status(self) -> str:
        if self.console is None:
            return "No run is open. /monitor (the person) or start_run (you) opens one."
        # Asked again within seconds -- an agent waiting for a restarted run to get somewhere
        # called this every second, a model call each time. A repeat waits for progress first.
        now = time.monotonic()
        last = getattr(self, "_status_asked_at", None)
        if last is not None and now - last < _STATUS_PATIENCE_SECONDS:
            step = int(getattr(self.console.brain, "step", 0) or 0)
            deadline = now + _STATUS_PATIENCE_SECONDS
            while time.monotonic() < deadline and self._status in ("live", "stalled") \
                    and int(getattr(self.console.brain, "step", 0) or 0) < step + 50:
                time.sleep(0.5)
                with self.lock:
                    self._poll_status()
        self._status_asked_at = time.monotonic()
        return self._evidence()

    def tool(self, calls: List[str], output: str, said: str = "") -> None:
        """Tools the agent ran, as one folded line. `said` is what it said as it reached for
        them ("I need to see how lr is used"): kept with the calls, shown when opened."""
        output = tui.clean(output).replace("\r", "")
        self._add(tui.Entry("tool", text=(said or "").strip(), calls=calls or ["tools"], output=output))

    def edit(self, labels: List[str], diff: str) -> None:
        """A change the agent wants to make: one folded line ("Edited train.py · +1 −1"),
        the diff under it when opened. What happens to it next (reviewed, applied, declined)
        goes on the same line through edit_status()."""
        entry = tui.Entry("edit", calls=labels or ["EDIT"], output=tui._SGR_RE.sub("", diff).replace("\r", ""))
        self._add(entry)
        self._edit_pending = entry

    def edit_status(self, text: str, detail: Optional[str] = None, final: bool = False) -> None:
        """`text` joins the change's line; `detail` (the reviewer's reason, the /undo hint)
        shows under the diff when the entry is opened."""
        entry = self._edit_pending
        if entry is None:
            self.note(text if not detail else f"{text} -- {detail}")
            return
        with self.lock:
            head, _, rest = entry.text.partition("\n")
            head = f"{head} · {text}" if head else text
            entry.text = head + (("\n" + rest) if rest else "") + (f"\n{detail}" if detail else "")
            if final:
                entry.open = None            # decided: back to one line (a click reopens it)
            entry.touch()
            self.dirty = True
        if final:
            self._edit_pending = None

    def _observe(self, event: str, **data: Any) -> None:
        """What the model is doing, as it does it (see pulse_cli._complete): its reasoning
        grows in a live Thinking entry, its answer in a live line that goes away once the
        pipeline prints the real thing."""
        with self.lock:
            if event == "stream_start":
                self._live_thinking = self._live_answer = None
                self._streamed_reasoning = False
            elif event == "reasoning_delta" and data.get("text"):
                self._flush_partial()
                if self._live_thinking is None:
                    self._live_thinking = tui.Entry("thinking", "", live=True)
                    self._think_started = time.monotonic()
                    self.view.entries.append(self._live_thinking)
                self._live_thinking.text += data["text"]
                self._live_thinking.touch()
                self._streamed_reasoning = True
            elif event == "content_delta" and data.get("text"):
                self._flush_partial()
                if self._live_answer is None:
                    self._live_answer = tui.Entry("text", "", live=True)
                    self.view.entries.append(self._live_answer)
                self._live_answer.text += data["text"]
                self._live_answer.touch()
            elif event == "stream_end":
                if self._live_thinking is not None:
                    self._live_thinking.live = False
                    self._live_thinking.seconds = max(0.1, time.monotonic() - self._think_started)
                    self._live_thinking.touch()
                if self._live_answer is not None and self._live_answer in self.view.entries:
                    self.view.entries.remove(self._live_answer)   # the pipeline prints the answer
                self._live_thinking = self._live_answer = None
            elif event == "reasoning" and data.get("text") and not self._streamed_reasoning:
                self.view.entries.append(tui.Entry("thinking", data["text"]))
            else:
                return
            self._last_blank = False
            self.dirty = True

    def _input(self, prompt: Any = "") -> str:
        """builtins.input while the app is up: the prompt is the question."""
        label = tui._SGR_RE.sub("", tui.clean(str(prompt))).replace("\r", "").strip().rstrip(">").strip()
        return self.ask(label or "Input")

    def _getpass(self, prompt: Any = "Password: ", stream: Any = None) -> str:
        label = tui.clean(str(prompt)).strip().rstrip(">:").strip()
        return self.ask(label or "Password", secret=True)

    def install(self) -> None:
        self._saved = {"stdout": sys.stdout, "stderr": sys.stderr, "input": builtins.input,
                       "getpass": getpass.getpass, "sigint": signal.getsignal(signal.SIGINT)}
        capture = _Capture(self)
        sys.stdout = sys.stderr = capture            # type: ignore[assignment]
        builtins.input = self._input
        getpass.getpass = self._getpass              # type: ignore[assignment]
        _ui.set_host(self)
        self._pc.set_agent_observer(self._observe)

    def uninstall(self) -> None:
        self._pc.set_agent_observer(None)
        _ui.set_host(None)
        saved = self._saved
        if saved:
            sys.stdout, sys.stderr = saved["stdout"], saved["stderr"]
            builtins.input = saved["input"]
            getpass.getpass = saved["getpass"]
        self._saved = {}

    # ================================================================== jobs

    def _start(self, work: Callable[..., None], *args: Any) -> None:
        with self.lock:
            self.view.busy, self.view.status, self.dirty = True, "", True
            self._job_from = len(self.view.entries)  # what this job shows starts here (the live feed)
        self._job = threading.Thread(target=self._run_job, args=(work, args), daemon=True, name="pulse-app-job")
        self._job.start()

    def _run_job(self, work: Callable[..., None], args: Tuple[Any, ...]) -> None:
        try:
            work(*args)
        except KeyboardInterrupt:
            self.note("Cancelled.")
        except EOFError:
            self.note("Cancelled.")
        except Exception as exc:                     # one failing command must not end the app
            self.error(f"That failed: {type(exc).__name__}: {exc}")
            if self._remote_active is not None:
                self._remote_active["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self._flush_partial()
                self.view.busy, self.view.live, self.view.status = False, "", ""
                self._job = None
                self._auto_turn_session = None
                self._refresh_header()
                self.dirty = True
                pending, self._crash_pending = self._crash_pending, None
                problems, self._problem_pending = self._problem_pending, None
                remote, self._remote_active = self._remote_active, None
                if remote is not None:
                    remote["result"] = self._remote_result(remote)
                    remote["done"].set()
                if pending and not self.done:
                    self._start_crash_turn(pending, self._crash_summary())
                elif problems and not self.done and self._status in ("live", "stalled"):
                    self._start_problem_turn(problems)

    # ================================================================== the web dashboard's prompts

    _REMOTE_REFUSED = {
        ("/exit", "/quit", "/q"): "Only the person at the machine can close Pulse.",
        ("/mouse", "/copy"): "That is about the terminal Pulse runs in; from the dashboard it does nothing.",
        ("/config", "/settings", "/agent"): "Settings and the agent's model are changed at the machine.",
        ("/monitor", "/runs", "/watch", "/sessions", "/attach", "/change"):
            "On the dashboard, open the run you want and send the prompt there.",
    }

    def _remote_refusal(self, line: str) -> Optional[str]:
        """Why a prompt from the dashboard cannot be done from afar (None: it can)."""
        word, _, rest = line.strip().partition(" ")
        word = word.lower()
        for words, why in self._REMOTE_REFUSED.items():
            if word in words:
                return f"{word}: {why}"
        if word == "/run" and not rest.strip():
            return "/run needs the script from the dashboard: /run train.py --epochs 3"
        return None

    def _start_home_commands(self) -> None:
        """The dashboard's prompts to the row setup opened for this app (each watched run has
        its own). The app answers them in turn with the person's own; the CLI's listener, which
        would answer them on a thread of its own, behind the app's back, is stopped."""
        cli = self.cli
        if self._home_commands is not None or not getattr(cli, "user_id", None) \
                or not getattr(cli, "debug_session_id", None):
            return
        stop = getattr(cli, "_stop_remote_command_listener", None)
        if callable(stop):
            stop()
        self._home_commands = CommandQueue(cli, lambda: getattr(cli, "debug_session_id", None),
                                           lambda text: self._run_remote_command(text, None),
                                           refuse=self._remote_refusal, on_error=self.error)
        self._home_commands.start()

    def _run_remote_command(self, line: str, session_id: Optional[str] = None) -> str:
        """A prompt from the dashboard, done as if typed here, after whatever is in hand. Sent
        to a run (`session_id`), it is about that run, which is brought on screen first if it
        is watched off screen. Returns what it showed, for the dashboard. Called on the
        command queue's thread; blocks until done."""
        remote: Dict[str, Any] = {"line": line, "session": session_id, "done": threading.Event()}
        with self.lock:
            self._remote_pending.append(remote)
            self.dirty = True
        while not remote["done"].wait(0.25):
            if self.done:
                raise RuntimeError("Pulse closed before it got to this command.")
        if remote.get("failed"):
            raise RuntimeError(remote["result"])
        return str(remote.get("result") or "Done.")

    def _start_pending_remote_command(self) -> None:
        with self.lock:
            if self._job is not None or self._question is not None or not self._remote_pending or self.done:
                return
            remote = self._remote_pending.popleft()
            self._remote_active = remote
        self._start(self._remote_job, remote)

    def _remote_job(self, remote: Dict[str, Any]) -> None:
        target = remote.get("session")
        if target is not None and target != (self.session or {}).get("session_id"):
            state = self.parked.get(target)
            if state is None:
                raise RuntimeError("Pulse is no longer watching that run.")
            self._switch_to(state["session"])
        with self.lock:
            self._flush_partial()
            self.view.entries.append(tui.Entry("note", "From the dashboard:"))
            mark = tui.Entry("user", remote["line"])
            self.view.entries.append(mark)
            self.view.scroll = 0
            self.dirty = True
        remote["mark"] = mark
        self._handle(remote["line"], about_run=target is not None)

    def _remote_result(self, remote: Dict[str, Any]) -> str:
        """What a dashboard prompt showed here, as plain text: the transcript after it, without
        the model's reasoning. With the lock held."""
        if remote.get("error"):
            remote["failed"] = True
            return str(remote["error"])
        entries, mark = self.view.entries, remote.get("mark")
        start = next((i for i in range(len(entries) - 1, -1, -1) if entries[i] is mark), None)
        lines: List[str] = []
        for entry in entries[start + 1:] if start is not None else []:
            if entry.kind in ("user", "thinking"):
                continue
            if entry.kind in ("tool", "edit"):
                lines.extend(entry.calls)
                if entry.kind == "edit" and entry.text:
                    lines.append(entry.text)
            elif entry.text:
                lines.append(entry.text)
        text = tui._SGR_RE.sub("", "\n".join(lines)).strip()
        if len(text) > 20000:
            text = text[:20000].rstrip() + "\n…"
        return text or "Done (nothing to show)."

    def _cancel_job(self) -> None:
        job = self._job
        if job is None or job.ident is None:
            return
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(job.ident), ctypes.py_object(KeyboardInterrupt))

    # ================================================================== keys

    def on_key(self, key: str) -> None:
        view = self.view
        with self.lock:
            question = self._question
            if key.startswith("hover:"):
                x, row = (int(v) for v in key.split(":")[1:])
                entry = view.row_entries.get(row) if x >= view.transcript_x - 1 else None
                entry = entry if entry is not None and entry.foldable() else None
                current = self._hovered
                if entry is not current:
                    if current is not None:
                        current.hover = False
                        current.touch()
                    if entry is not None:
                        entry.hover = True
                        entry.touch()
                    self._hovered = entry
                    self.dirty = True                # only a change needs a redraw
                return
            self.dirty = True
            if self._sel is not None and not self._sel.get("dragging"):
                self._sel = None                     # a key after a selection: it goes
            if key == "ctrl+c":
                if question is not None:
                    question.cancelled = True
                    question.done.set()
                elif view.busy:
                    view.status = "cancelling -- waiting for the current step to return"
                    self._cancel_job()
                elif view.editor.text:
                    view.editor.take(remember=False)
                elif time.monotonic() - self._last_ctrl_c < 2.0:
                    self.done = True
                else:
                    self._last_ctrl_c = time.monotonic()
                    view.status = ("Ctrl+C again leaves Pulse -- the training goes on without it · /stop stops the training"
                                   if self.hosting is not None else "Ctrl+C again to leave Pulse")
                return
            view.status = ""
            if key == "ctrl+o":
                view.expanded = not view.expanded
                for entry in view.entries:           # Ctrl+O speaks for every entry again
                    if entry.open is not None:
                        entry.open = None
                        entry.touch()
                return
            if key in ("pgup", "pgdn"):
                page = max(3, self._height // 2)
                view.scroll = max(0, view.scroll + (page if key == "pgup" else -page))
                return
            if key.startswith("wheel:"):
                parts = key.split(":")
                up = parts[1] == "up"
                x = int(parts[2]) if len(parts) > 2 else 10 ** 6
                if view.side is not None and 0 <= view.side_cols and x <= view.side_cols:
                    view.side_scroll = max(0, view.side_scroll + (-3 if up else 3))   # the run pane
                else:
                    view.scroll = max(0, view.scroll + (3 if up else -3))
                return
            if key.startswith(("press:", "drag:", "release:")):
                self._mouse_button(key)
                return
            if key.startswith("click:"):
                self._click(*(int(v) for v in key.split(":")[1:]))
                return
            if key == "end" and view.scroll:
                view.scroll = 0
                return
            if key == "ctrl+l":
                return
            if question is not None and question.options is not None:
                self._key_in_list(key, question)
                return
            if question is not None:
                if key == "enter":
                    question.answer = view.editor.take(remember=False)
                    question.done.set()
                elif key == "esc":
                    question.cancelled = True
                    question.done.set()
                else:
                    view.editor.handle(key, history=False)
                return
            if key == "enter":
                if view.busy:
                    view.status = "still working -- Ctrl+C cancels it"
                    return
                line = view.editor.take().strip()
                if line:
                    view.scroll = 0
                    self._flush_partial()
                    view.entries.append(tui.Entry("user", line))
                    self._last_blank = False
                    self._auto_crash_turns = 0
                    self._start(self._handle, line)
            elif key == "tab":
                hints = tui._hints(view)
                if hints:
                    view.editor.set(hints[0][0] + " ")
            elif key == "esc":
                if view.editor.text:
                    view.editor.take(remember=False)           # Esc clears what was typed
                elif self.console is not None and not view.busy:
                    self._start(self._foreground_run if self.background else self._background_run)
                elif self.console is not None:
                    view.status = "finish or cancel (Ctrl+C) what is running first"
            elif key == "ctrl+d":
                if not view.editor.text and not view.busy:
                    self.done = True
            else:
                view.editor.handle(key)

    # ------------------------------------------------------------------ the mouse: clicks and selections

    def _click(self, x: int, row: int) -> None:
        entry = self.view.row_entries.get(row)
        if entry is not None and entry.foldable():
            entry.toggle(self.view.expanded)
            tui.keep_in_place(self.view, entry, row)

    def _mouse_button(self, key: str) -> None:
        """A press starts a selection where it lands; a drag stretches it (within the pane it
        started in); the release copies it -- or, with no drag, is a click."""
        kind, x, y = key.split(":")[0], *(int(v) for v in key.split(":")[1:])
        view = self.view
        if kind == "press":
            side = view.side is not None and 0 <= view.side_cols and x <= view.side_cols
            cols = (0, view.side_cols) if side else (
                view.transcript_x if view.side_cols >= 0 else 0, max(self._total_w - 1, 0))
            self._sel = {"anchor": (x, y), "head": (x, y), "cols": cols, "dragging": True, "moved": False}
            return
        sel = self._sel
        if sel is None:
            if kind == "release":
                self._click(x, y)
            return
        x = max(sel["cols"][0], min(x, sel["cols"][1]))
        if kind == "drag":
            sel["head"] = (x, y)
            sel["moved"] = sel["moved"] or (x, y) != sel["anchor"]
            return
        sel["head"], sel["dragging"] = (x, y), False      # release
        if not sel["moved"]:
            self._sel = None
            self._click(*sel["anchor"])
            return
        text = tui.selected_text(self._frame, sel)
        if not text:
            self._sel = None
            return
        sent = []
        if self._screen is not None:
            self._screen.copy(text)
            sent.append("terminal")
        threading.Thread(target=_copy_with_tool, args=(text,), daemon=True).start()
        lines = text.count("\n") + 1
        view.status = (f"Copied {len(text):,} characters" + (f" ({lines} lines)" if lines > 1 else "")
                       + " -- paste with Ctrl+Shift+V (or your terminal's paste)")

    def _key_in_list(self, key: str, question: _Question) -> None:
        view = self.view
        options = question.options or []
        if key in ("up", "down"):
            if view.option_ids:
                view.option_index = (view.option_index + (1 if key == "down" else -1)) % len(view.option_ids)
            return
        if key == "enter":
            if view.option_ids:
                question.answer = view.option_ids[min(view.option_index, len(view.option_ids) - 1)]
            else:
                question.answer = None
            question.done.set()
            return
        if key == "esc":
            question.answer = None
            question.done.set()
            return
        if len(key) == 1 and not view.editor.text:
            # a row's own shortcut, or one of the caller's extra keys, before it is a search
            letter = key.lower()
            for index, option in enumerate(options):
                if getattr(option, "key", None) and str(option.key).lower() == letter:
                    question.answer = index
                    question.done.set()
                    return
            if letter in question.hotkeys:
                question.answer = question.hotkeys[letter]
                question.done.set()
                return
        if view.editor.handle(key, history=False):
            needle = view.editor.text.strip().lower()
            view.option_ids = [i for i, o in enumerate(options)
                               if not needle or needle in f"{o.label} {o.detail or ''} {o.tag or ''}".lower()]
            view.option_index = 0

    # ================================================================== what was typed

    def _handle(self, line: str, about_run: bool = False) -> None:
        if line.startswith("/") or line == "?":
            self._command(line)
            return
        cli = self.cli
        if not cli.agent_provider or not cli.agent_key:
            self.note("No agent is set up yet. /agent picks a model and a key "
                      "(or signs you in to OpenRouter, which needs no key).")
            return
        from . import pulse_code
        evidence = self._evidence() if self.console is not None and (not self.background or about_run) else None
        before = (self.session or {}).get("session_id")
        outcome = pulse_code.run_turn(cli, line, evidence=evidence)
        restarted = (self.session or {}).get("session_id") != before     # the agent ran it again itself
        if outcome == "applied" and self.console is not None and not restarted:
            try:
                self.console.brain.note_fix("code edited from the Pulse app")
            except Exception:
                pass
            launched = self.session is not None and self.session.get("session_id") in self.launched
            over = self._status not in ("live", "stalled")
            hosted = self.hosting is not None and (self.session or {}).get("session_id") == getattr(
                self.hosting, "session_id", None)
            if over:
                self.note("/restart runs it again with the change." if launched or hosted
                          else "Start it again to try the change.")
            else:
                self.note("The run is still executing the code it started with. "
                          + ("/restart stops it and starts it again with the change."
                             if launched else "Stop it (/stop) and start it again to apply the change."))

    def _command(self, line: str) -> None:
        from . import pulse_code
        word, _, rest = line.partition(" ")
        word, rest = word.lower(), rest.strip()
        if word in ("/exit", "/quit", "/q"):
            self.done = True
        elif word in ("/help", "/h", "?"):
            self._help()
        elif word in ("/monitor", "/runs", "/watch", "/sessions", "/attach"):
            self._pick_run(rest)
        elif word == "/change":
            self._change(rest)
        elif word == "/run":
            self._launch(rest)
        elif word == "/mouse":
            self._set_mouse(rest)
        elif word in ("/config", "/settings"):
            self._config(rest)
        elif word == "/copy":
            self._copy(rest)
        elif self.console is not None and self._run_command(word, rest):
            pass
        elif word == "/close":
            self.note("No run is open.")
        elif word in ("/back", "/home"):
            self.note("You are home already. " + ("/change opens a watched run." if self.parked
                                                   else "/monitor opens a run beside the agent."))
        elif not pulse_code._handle_command(self.cli, line):
            self.done = True

    # ------------------------------------------------------------------ the mouse, the clipboard

    def _set_mouse(self, arg: str) -> None:
        arg = arg.lower()
        on = (not self.mouse) if arg not in ("on", "off") else arg == "on"
        self.mouse = on
        screen = self._screen
        if screen is not None:
            screen.set_mouse(on)
        _settings.set("mouse", "on" if on else "off")
        self.note("Mouse on: drag over any text to copy it, a click opens a folded line, the wheel scrolls "
                  "(over the run pane, the pane)." if on else
                  "Mouse off: your terminal selects and scrolls as usual. Ctrl+O opens folded lines, PgUp/PgDn "
                  "scroll. /mouse on gives the app the mouse back.")

    def _config(self, arg: str) -> None:
        """/config: the remembered settings. /config <name> <value> sets one (and applies it
        now); /config agent|workspace|project opens its picker; /config reset forgets all."""
        words = arg.split()
        if not words:
            print()
            print(_ui._s("Settings", "bold") + _ui._s(f"  {_settings.path()}", "dim"))
            for name, shown, what in _settings.rows():
                print("  " + _ui._s(name.ljust(14), "accent") + shown)
                print("  " + " " * 14 + _ui._s(what, "dim"))
            print(_ui._s("/config <name> <value> changes one (/config mouse off) · /config agent, workspace or "
                         "project opens its picker · /config reset forgets them all", "dim"))
            return
        name, value = words[0].lower(), " ".join(words[1:]).strip()
        cli = self.cli
        if name == "reset":
            if _ui.confirm("Forget every setting and saved key? The next start asks again.", default=False):
                _settings.reset()
                self.note("Settings and saved keys forgotten. The next start asks again.")
            return
        problem = _settings.check(name, value or "x")
        if problem and not (name in _settings.KNOWN and not value):
            self.note(f"/config: {problem}.")
            return
        if name == "agent":
            if cli._pick_agent_and_remember(initial=False):
                self.note(f"Agent: {cli.agent_provider} -- remembered for the next start.")
                self._refresh_header()
            return
        if name in ("workspace", "project"):
            if not getattr(cli, "user_id", None):
                self.note("Not signed in to Pulse Cloud, so there is no workspace to pick. /cloud shows the status.")
                return
            _settings.unset("project")
            if name == "workspace":
                _settings.unset("workspace")
                cli._select_workspace_and_project()
            else:
                cli._project_flow(None)
                if cli.project_id:
                    _settings.set("project", {"id": cli.project_id, "name": cli.project_name or cli.project_id})
            self.note(f"{name.capitalize()} remembered for the next start.")
            return
        if not value:
            shown = dict((n, v) for n, v, _w in _settings.rows())[name]
            choices = _settings.KNOWN[name][1]
            self.note(f"{name} = {shown}" + (f"   (/config {name} {'|'.join(choices)})" if choices else ""))
            return
        value = value.lower() if _settings.KNOWN[name][1] else value
        _settings.set(name, value)
        if name == "mouse":
            self._set_mouse(value)
            return
        if name == "audits":
            self._set_audits(value)
            return
        if name == "review":
            cli.review = value == "on"
        if name == "remember_keys" and value == "off":
            _settings.forget_keys()
            self.note("Saved keys deleted; pasted keys are not kept from now on.")
            return
        self.note(f"{name} = {value} -- remembered for the next start.")

    def _copy(self, arg: str) -> None:
        """The agent's last answer (or the n-th from the end) to the clipboard."""
        words = [e.text for e in self.view.entries if e.kind == "say" and e.text.strip()]
        try:
            back = max(1, int(arg)) if arg else 1
        except ValueError:
            self.note("usage: /copy  (the agent's last answer), /copy 2 (the one before)")
            return
        if len(words) < back:
            self.note("Nothing from the agent to copy yet." if not words
                      else f"There are only {len(words)} answers to copy from.")
            return
        text = words[-back]
        sent = []
        if self._screen is not None:
            self._screen.copy(text)
            sent.append("the terminal")
        if _copy_with_tool(text):
            sent.append("the system clipboard")
        first = text.strip().splitlines()[0]
        self.note(f"Copied {len(text):,} characters (\u201c{first[:50]}{'…' if len(first) > 50 else ''}\u201d)"
                  + (f" to {' and '.join(sent)}." if sent else "."))

    def _help(self) -> None:
        def section(title: str, rows: List[Tuple[str, str]]) -> None:
            print(_ui._s(title, "bold"))
            for command, what in rows:
                print("  " + _ui._s(command.ljust(12), "accent") + _ui._s(what, "dim"))
            print()

        print()
        if self.console is not None:
            section("This run", DEBUG_COMMANDS)
        section("Pulse", HOME_COMMANDS)
        print(_ui._s("Anything else is for the agent" + (": a question about this run, or a fix to make."
                                                        if self.console is not None else
                                                        ": a feature, a fix, a question about the code."), "dim"))
        print(_ui._s("Ctrl+O unfolds the agent's thinking and tool output · PgUp/PgDn scroll · "
                     "Ctrl+C cancels what is running, twice leaves.", "dim"))

    # ================================================================== runs on this machine

    def _change(self, wanted: str = "") -> None:
        """/change [run]: look at another run; every run already watched stays watched."""
        if not wanted and not self.parked and self.console is None:
            self._pick_run("")
            return
        self._pick_run(wanted, keep=True)

    def _pick_run(self, wanted: str = "", keep: bool = False) -> None:
        from . import pulse_console as con
        sessions = con.discover()
        idle = con.unmonitored_python_processes()
        current = (self.session or {}).get("session_id")
        if not wanted and not keep and self.background and current:
            self._foreground_run()                   # /monitor alone: the run that is already watched
            return
        if wanted:
            chosen = con.pick_session(sessions, wanted)
            if chosen is not None:
                self._switch_to(chosen)
                return
            outside = [p for p in idle if os.path.basename(p["script"]) == os.path.basename(wanted)
                       or str(p["pid"]) == wanted]
            if len(outside) == 1:
                self._attach_outside(outside[0])
                return
        order = {"live": 0, "stalled": 1, "crashed": 2}
        sessions = sorted(sessions, key=lambda s: (order.get(s["status"], 3), -(s.get("last_seen") or 0)))
        going = [s for s in sessions if s["status"] in ("live", "stalled")]
        over = [s for s in sessions if s["status"] not in ("live", "stalled")][:12]
        options: List[Any] = []
        targets: List[Tuple[str, Any]] = []
        for session in going + over:
            script = os.path.basename(session.get("script") or "?")
            bits = [f"step {session['step']:,}" if session.get("step") else "no steps yet"]
            if session.get("loss") is not None:
                bits.append(f"loss {con.compact(session['loss'])}")
            when = session["status"] if session["status"] in ("live", "stalled") else \
                f"{session['status']} {con.ago(session.get('last_seen'))}"
            bits.append(os.path.dirname(session.get("script") or "") or session["session_id"])
            if session.get("session_id") == current:
                when = "watched" if self.background else "open"
            elif session.get("session_id") in self.parked:
                when = "watched"
            options.append(_ui.Option(script, " · ".join(bits), tag=when))
            targets.append(("session", session))
        for process in idle[:8]:
            options.append(_ui.Option(os.path.basename(process["script"]),
                                      f"pid {process['pid']} · {process.get('cmdline') or 'started outside Pulse'}",
                                      tag="not under Pulse"))
            targets.append(("process", process))
        if not options:
            self.note("No runs on this machine yet. /run train.py starts one under Pulse; a script "
                      "that calls auto_track(mode=\"stream\"), or `pulse run --stream train.py` in "
                      "another terminal, shows up here too.")
            return
        picked = self.choose(options, title=f"Runs on this machine ({len(going)} live)" if going
                             else "Runs on this machine (none live)")
        if not isinstance(picked, int):
            return
        kind, target = targets[picked]
        if kind == "session":
            self._switch_to(target)
        else:
            self._park()
            self._attach_outside(target)

    def _attach_outside(self, process: Dict[str, Any]) -> None:
        """A run that was not started under Pulse: sample it from outside (py-spy)."""
        from . import pulse_attach as attach
        from . import pulse_console as con
        pid = int(process["pid"])
        report = attach.describe_readiness(pid)
        if not report["ok"]:
            self.error(f"Cannot read pid {pid}: {report['reason']}")
            if not report.get("pyspy"):
                self.note("Install it:  pip install py-spy")
            elif report.get("needs_root") and hasattr(os, "geteuid") and os.geteuid() != 0:
                self.note("Reading another process needs root on this machine. Leave Pulse and run:  "
                          f"sudo pulse attach --pid {pid}")
            return
        with _ui.Stage(f"Reading pid {pid}"):
            monitor = attach.AttachedMonitor(pid)
            first = monitor.sample_once()
            monitor.snapshot()
        if not first:
            self.note(f"pid {pid} can be read, but no numbers came back: its loop is probably at the top "
                      "level of the script, where values cannot be read from outside. A run started "
                      "under Pulse always can be (/run).")
            monitor.discard()
            return
        monitor.writer.flush()
        monitor.start()
        session = con._describe_session(monitor.directory) or {
            "session_id": monitor.session_id, "directory": monitor.directory,
            "script": monitor.script, "status": "live", "step": 0}
        self._open_run(session, monitor=monitor)

    def _brain_agent(self) -> Optional[Callable[[str], str]]:
        """The model the open run's brain audits with: the app's agent, or None."""
        cli = self.cli
        if not self.audits or not cli.agent_provider or not cli.agent_key:
            return None
        from .pulse_brain import build_litellm_agent
        model = cli.agent_model_string or self._pc.PROVIDERS.get(cli.agent_provider, {}).get("model")
        if not model:
            return None
        key = cli.agent_key if cli.agent_key != "local" else None
        return build_litellm_agent(model, api_key=key, api_base=getattr(cli, "agent_api_base", None))

    def _open_run(self, session: Dict[str, Any], monitor: Any = None, replace: bool = False) -> None:
        """Show `session` beside the agent. The run in view until now stays watched (parked),
        unless `replace` (a restart: the old process is gone, its new one takes its place)."""
        if session.get("session_id") in self.parked:
            self._switch_to(session)
            return
        if replace:
            self._close_run(quiet=True)
        else:
            self._park()
        console = _make_console(self, session, self._brain_agent())
        console.brain.poll_once()
        console.start()
        with self.lock:
            self.console, self.session, self._monitor = console, session, monitor
            self._rate = self._rate_at = None
            self._stall_said = self._paused_here = False
            self._status = session.get("status") or "live"
        script = session.get("script") or ""
        workdir = console.workdir
        if os.path.isdir(workdir):
            self._point(workdir, [script] if script and os.path.isfile(script) else [])
        from . import pulse_code
        self.cli._system_prompt_override = pulse_code.CODE_SYSTEM_PROMPT + DEBUG_PROMPT
        self.cli._native_prompt_suffix = DEBUG_PROMPT_NATIVE
        self._start_runlog(session, workdir)
        self._refresh_header()
        name = os.path.basename(script) or session.get("session_id") or "the run"
        over = "" if self._status in ("live", "stalled") else f" ({self._status})"
        self.note(f"Opened {name}{over}. Ask about the run or tell the agent what to fix; /help lists "
                  "the commands, Esc closes the run.")
        if console.brain.agent is not None:
            self.note("The agent audits this run on its own schedule (/audits off stops that).")
        with self.lock:
            self._refresh_side(force=True)

    def _start_runlog(self, session: Dict[str, Any], workdir: str) -> None:
        """The run's own row on the dashboard; the agent's turns about it go there too."""
        self._stop_runlog()
        run_id = session.get("session_id")
        log = RunLog(self.cli, session, workdir,
                     on_command=lambda text: self._run_remote_command(text, run_id),
                     on_response=self._remote_refusal, on_error=self.error)
        if not log.enabled:
            return
        self.runlog = log
        log.start()
        self._wrap_sync(log)

    def _wrap_sync(self, log: "RunLog") -> None:
        """The agent's turns go on the open run's dashboard row too."""
        cli = self.cli
        self._orig_sync = original = cli._sync_agent_turn

        def sync(question: str, answer: str, traceback_signature: Optional[str] = None,
                 fix_applied: Optional[Dict[str, Any]] = None) -> None:
            who = "dashboard" if self._remote_active is not None else (
                "pulse" if self._auto_turn_session is not None else "you")
            try:
                original(question, answer, traceback_signature=traceback_signature, fix_applied=fix_applied)
            finally:
                log.agent_turn(question, answer, fix_applied=fix_applied, who=who)

        cli._sync_agent_turn = sync

    def _unwrap_sync(self) -> None:
        if self._orig_sync is not None:
            self.cli._sync_agent_turn = self._orig_sync
            self._orig_sync = None

    def _stop_runlog(self) -> None:
        log, self.runlog = self.runlog, None
        if self._orig_sync is not None:
            self.cli._sync_agent_turn = self._orig_sync
            self._orig_sync = None
        if log is not None:
            log.close()

    def _close_all(self) -> None:
        """Stop watching every run (they keep running): leaving the app, /close all."""
        self._close_run(quiet=True)
        while self.parked:
            self._unpark_quietly(next(iter(self.parked)), background=True)
            self._close_run(quiet=True)

    def _close_run(self, quiet: bool = False) -> None:
        """Stop watching the open run (it keeps running) and aim the agent at home again."""
        console = self.console
        if console is None:
            return
        self._stop_runlog()
        console.stop(join=True)
        monitor = self._monitor
        with self.lock:
            self.console = self.session = self._monitor = None
            self.view.side, self.view.side_brief = None, []
            self.background = False
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
        self._point(self.home_root, self.home_focus)
        from . import pulse_code
        self.cli._system_prompt_override = pulse_code.CODE_SYSTEM_PROMPT
        self.cli._native_prompt_suffix = ""
        self._refresh_header()
        if not quiet:
            self.note("Stopped watching the run. It keeps going; /monitor opens it again.")

    # ------------------------------------------------------------------ a crash

    def _crash_summary(self) -> str:
        """The exception the run died with ("ValueError: matmul: ..."), from its crash event."""
        console = self.console
        events = list(getattr(getattr(console, "brain", None), "events", []) or [])
        for event in reversed(events):
            if event.get("event") == "crash":
                text = str(event.get("exception") or "").strip()
                if not text and event.get("traceback"):
                    text = str(event["traceback"]).rstrip().splitlines()[-1]
                return text
        return ""

    def _on_crash(self) -> None:
        """The open run crashed: the agent looks at it at once -- the traceback and the code
        are its evidence, a fix is a change it proposes like any other. Once per run. Called
        with the lock held, from the screen loop."""
        session_id = (self.session or {}).get("session_id")
        if session_id in self._crashes_seen:
            return
        self._crashes_seen.add(session_id)
        script = os.path.basename((self.session or {}).get("script") or "the run")
        what = self._crash_summary()
        self._auto_crash_turns += 1
        if self.auto_fix_limit is not None and self._auto_crash_turns > self.auto_fix_limit:
            # crash, fix, restart, crash again...: the person decides what next
            self.error(f"{script} crashed again" + (f": {what}" if what else "") + ".")
            self.note(f"That is {self._auto_crash_turns} crashes in a row; the agent is not starting on this one "
                      "by itself. Tell it what to try next.")
            return
        cli = self.cli
        if not cli.agent_provider or not cli.agent_key:
            self.error(f"{script} crashed" + (f": {what}" if what else "") + ".")
            self.note("No agent is set up to look at it: /agent picks a model, then ask what went wrong.")
            return
        if self._job is not None or self.view.busy:
            self._crash_pending = script     # the agent is busy: it looks once it is done
            self.error(f"{script} crashed" + (f": {what}" if what else "")
                       + ". The agent looks at it when it has finished what it is doing.")
            return
        self._start_crash_turn(script, what)

    def _start_crash_turn(self, script: str, what: str) -> None:
        self.error(f"{script} crashed" + (f": {what}" if what else "") + ".")
        request = (f"{script} just crashed" + (f" with {what}" if what else "") + ". The traceback is in the "
                   "evidence. Find the cause in the code, explain it in a sentence or two, and fix it. Then "
                   "decide: run it again with the fix (restart_run, or a RESTART: line) to check it works, or "
                   "leave it stopped if a run now would not tell anything (say why) -- and say which you did.")
        self.view.entries.append(tui.Entry("user", f"Why did {script} crash? Fix it."))
        self._last_blank = False
        self.dirty = True
        self._auto_turn_session = (self.session or {}).get("session_id")
        self._start(self._crash_turn, request)

    # ------------------------------------------------------------------ a serious finding

    def _on_findings(self, findings: List[Any]) -> None:
        """The open run's checks raised something serious -- a NaN or Inf, an exploding norm, a
        loss that blew up or never learned: the agent takes it on at once, as for a crash
        (a warning is only announced). Once per problem; with the lock held."""
        session_id = (self.session or {}).get("session_id")
        serious = [f for f in findings if detect.acts_on(f)
                   and (session_id, getattr(f, "check", ""), getattr(f, "variable", "")) not in self._problems_seen]
        if not serious or self._status not in ("live", "stalled"):
            return
        if self._auto_turn_session == session_id:
            # the agent is already on this run's problem: the new findings are in its evidence
            # (a NaN loss brings a NaN gradient and an infinite val_loss with it), not a new turn
            for finding in serious:
                self._problems_seen.add((session_id, getattr(finding, "check", ""), getattr(finding, "variable", "")))
            return
        for finding in serious:
            self._problems_seen.add((session_id, getattr(finding, "check", ""), getattr(finding, "variable", "")))
        cli = self.cli
        if not cli.agent_provider or not cli.agent_key:
            self.note("No agent is set up to look at that: /agent picks a model, then ask what went wrong.")
            return
        self._auto_crash_turns += 1
        if self.auto_fix_limit is not None and self._auto_crash_turns > self.auto_fix_limit:
            self.note(f"The agent has started on {self.auto_fix_limit} problems in a row by itself; it leaves this one "
                      "to you. Ask about it here.")
            return
        if self._job is not None or self.view.busy:
            self._problem_pending = (self._problem_pending or []) + serious
            self.note("The agent looks at that when it has finished what it is doing.")
            return
        self._start_problem_turn(serious)

    def _note_stall(self, status: str) -> None:
        """A run that stops making steps while its process lives -- stuck (a deadlock, a hung
        data loader, a wait on a dead rank) or paused -- is said once, and so is it moving
        again. Not an agent turn by itself: a pause looks the same. With the lock held."""
        was = self._stall_said
        if status == "stalled" and not was:
            self._stall_said = True
            if not self._paused_here:
                name = os.path.basename((self.session or {}).get("script") or "") or "The run"
                self.error(f"{name} has stopped making steps but is still running: stuck (a deadlock, a "
                           "hung data loader) or paused. Ask the agent to look at it, or /stop.")
        elif status == "live" and was:
            self._stall_said = False
            if not self._paused_here:
                self.note(f"{os.path.basename((self.session or {}).get('script') or 'The run')} is moving again.")

    def _on_parked_findings(self, console: Any, findings: List[Any]) -> None:
        """A run watched off screen raised something serious. At home (no other run on
        screen) the agent takes it on, as for a crash there; with another run on screen the
        person is told where to look."""
        serious = [f for f in findings if detect.acts_on(f)]
        session_id = next((sid for sid, st in self.parked.items() if st["console"] is console), None)
        if not serious or session_id is None:
            return
        if self.console is None or self.background:
            self._park()
            self._unpark_quietly(session_id, background=True)
            self._on_findings(serious)
        else:
            name = os.path.basename((console.session or {}).get("script") or "") or "a watched run"
            key = (session_id, "parked-note")
            if key not in self._problems_seen:
                self._problems_seen.add(key)
                self.error(f"{name} (watched off screen) needs a look: {serious[0].message}. "
                           f"/change {name} opens it.")

    def _start_problem_turn(self, findings: List[Any]) -> None:
        script = os.path.basename((self.session or {}).get("script") or "the run")
        found = "; ".join(str(getattr(f, "message", f)) for f in findings[:3])
        variable = getattr(findings[0], "variable", "") or "it"
        request = (f"{script} is still running, and Pulse's checks found something serious: {found}. The "
                   "evidence has the run's values. Find the cause in the code, explain it in a sentence or two, "
                   "and fix it. Then decide what to do with the run: restart it with the fix (restart_run, or "
                   "a RESTART: line) when the run as it is cannot recover (a NaN in the weights never goes "
                   "away), stop it (stop_run) if it is only burning compute, or leave it running if the "
                   "problem is harmless -- and say which you did.")
        self.view.entries.append(tui.Entry("user", f"What is wrong with {variable} in {script}? Fix it."))
        self._last_blank = False
        self.dirty = True
        self._auto_turn_session = (self.session or {}).get("session_id")
        self._start(self._crash_turn, request)

    def _crash_turn(self, request: str) -> None:
        """A crash is about the run even from home: the agent works on the run's project for
        this turn, with its evidence, and goes back home after."""
        home = self.background
        if home:
            self._aim_at_run()
        try:
            self._handle(request, about_run=True)
        finally:
            if home and self.background:
                self._aim_home()

    def _background_run(self) -> None:
        """/home (and /back, Esc): back to the agent for anything at all -- it works on the home
        project, without the run's evidence -- while the run stays watched: its findings are
        still announced, a crash still gets the agent, and /monitor (or Esc) brings it back."""
        if self.console is None:
            self.note("You are home already. /monitor opens a run beside the agent.")
            return
        with self.lock:
            self.background = True
            self.view.side, self.view.side_brief = None, []
        self._aim_home()
        self._refresh_header()
        watched = [os.path.basename((self.session or {}).get("script") or "the run")] + [
            os.path.basename(state["session"].get("script") or "a run") for state in self.parked.values()]
        self.note(f"Home. Still watched: {', '.join(watched)} -- findings and crashes show up here; "
                  "/monitor brings the run back, /change picks another, /close stops watching.")

    def _foreground_run(self) -> None:
        if self.console is None or not self.background:
            return
        with self.lock:
            self.background = False
        self._aim_at_run()
        self._refresh_header()
        self._refresh_side(force=True)

    def _aim_home(self) -> None:
        from . import pulse_code
        self._point(self.home_root, self.home_focus)
        self.cli._system_prompt_override = pulse_code.CODE_SYSTEM_PROMPT
        self.cli._native_prompt_suffix = ""

    def _aim_at_run(self) -> None:
        from . import pulse_code
        console, session = self.console, self.session or {}
        if console is None:
            return
        script = session.get("script") or ""
        if os.path.isdir(console.workdir):
            self._point(console.workdir, [script] if script and os.path.isfile(script) else [])
        self.cli._system_prompt_override = pulse_code.CODE_SYSTEM_PROMPT + DEBUG_PROMPT
        self.cli._native_prompt_suffix = DEBUG_PROMPT_NATIVE

    def _park(self) -> None:
        """Keep watching the open run, off screen: its console goes on reading the run (its
        findings are announced with its name), its dashboard row goes on, and /change or
        /monitor brings it back."""
        session_id = (self.session or {}).get("session_id")
        if self.console is None or not session_id:
            return
        self._unwrap_sync()
        with self.lock:
            self.parked[session_id] = {field: getattr(self, field) for field in _WATCH_FIELDS}
            self.console = self.session = self._monitor = self.runlog = None
            self.view.side, self.view.side_brief = None, []
            self.background = False
        self._aim_home()

    def _unpark(self, session_id: str) -> None:
        state = self.parked.pop(session_id)
        with self.lock:
            for field, value in state.items():
                setattr(self, field, value)
            self.background = False
        if self.runlog is not None:
            self._wrap_sync(self.runlog)
        self._aim_at_run()
        self._refresh_header()
        name = os.path.basename((self.session or {}).get("script") or "the run")
        self.note(f"Back on {name} ({self._status}).")
        with self.lock:
            self._refresh_side(force=True)

    def _switch_to(self, session: Dict[str, Any]) -> None:
        """Show `session`, keeping every run already watched (the one left included)."""
        session_id = session.get("session_id")
        if session_id == (self.session or {}).get("session_id"):
            self._foreground_run()
            return
        self._park()
        if session_id in self.parked:
            self._unpark(session_id)
        else:
            self._open_run(session)

    def _tick_parked(self, now: float) -> None:
        """The runs watched off screen: their dashboard rows, and what became of them (a crash
        is said at once; at home, the agent takes it on). With the lock held."""
        if not self.parked or now - self._parked_at < 2.0:
            return
        self._parked_at = now
        from . import pulse_console as con
        for session_id, state in list(self.parked.items()):
            console, session = state["console"], state["session"]
            if state["runlog"] is not None:
                try:
                    state["runlog"].tick(console.snapshot(), state["_status"], list(console.brain.events), session)
                except Exception:
                    pass
            described = con._describe_session(session.get("directory", ""))
            if not described or described["status"] == state["_status"]:
                continue
            state["_status"] = described["status"]
            name = os.path.basename(session.get("script") or "a run")
            if described["status"] == "crashed" and session_id not in self._crashes_seen:
                if self.console is None or self.background:
                    # nobody is looking at another run: this one becomes the run in hand, so the
                    # agent can take the crash on (the view stays home)
                    home = self.console is None or self.background
                    self._park()
                    self._unpark_quietly(session_id, background=home)
                    self._on_crash()
                else:
                    self.error(f"{name} (watched off screen) crashed. /change {name} opens it; the agent "
                               "looks at it then.")
            elif described["status"] not in ("live", "stalled"):
                self.note(f"{name} (watched off screen) has {described['status']}.")

    def _unpark_quietly(self, session_id: str, background: bool) -> None:
        state = self.parked.pop(session_id)
        for field, value in state.items():
            setattr(self, field, value)
        self.background = background
        if self.runlog is not None:
            self._wrap_sync(self.runlog)
        if not background:
            self._aim_at_run()
        self._refresh_header()

    def _point(self, root: str, focus: List[str]) -> None:
        """Aim the code agent at a project: its files, its working directory, its history."""
        from . import pulse_code
        cli = self.cli
        review = getattr(cli, "review", True)
        try:
            os.chdir(root)
        except OSError:
            return
        cli.setup_code(root, [f for f in focus if os.path.isfile(f)], pulse_code.scan_project(root))
        cli.review = review
        cli.set_code_text(cli.code_text, script_path=cli.script_path)
        cli._project_root = root
        cli._terminal_executor = None                # its working directory was the old project
        cli.agent_history = []

    def _evidence(self) -> str:
        brain = self.console.brain
        script = os.path.basename((self.session or {}).get("script") or "?")
        return (f"EVIDENCE FROM THE RUN ({script}, {self._status})\n"
                + brain.render_evidence(brain.evidence(include_code=False)))

    # ------------------------------------------------------------------ commands on the open run

    def _run_command(self, word: str, rest: str) -> bool:
        from . import pulse_console as con
        from . import pulse_stream as stream
        console = self.console
        if word in ("/back", "/home"):
            self._background_run()
        elif word == "/close":
            if rest.lower() == "all":
                self._close_all()
                self.note("Stopped watching every run. They keep going; /monitor opens one again.")
            else:
                self._close_run()
                if self.parked:
                    names = ", ".join(os.path.basename(s["session"].get("script") or "a run")
                                      for s in self.parked.values())
                    self.note(f"Still watched: {names}. /change opens one; /close all stops them all.")
        elif word == "/findings":
            console.show_findings()
        elif word == "/status":
            console.show_status()
        elif word == "/curve":
            console.show_curve(rest or "loss")
        elif word == "/vars":
            console.show_vars()
        elif word == "/source":
            console.show_code()
        elif word == "/trace":
            console.show_trace(rest)
        elif word == "/audit":
            if console.brain.agent is None:
                console.brain.agent = console.agent = self._brain_agent_forced()
            console.audit()
        elif word == "/audits":
            self._set_audits(rest)
        elif word == "/cd":
            print(f"\n  {console.workdir}\n")
        elif word == "/interval":
            try:
                seconds = float(rest)
            except ValueError:
                seconds = float("nan")
            if not math.isfinite(seconds) or not (con.MIN_SAMPLE_INTERVAL <= seconds <= con.MAX_SAMPLE_INTERVAL):
                self.note(f"/interval takes {con.MIN_SAMPLE_INTERVAL:g} to {con.MAX_SAMPLE_INTERVAL:g} seconds, "
                          "e.g. /interval 0.5")
            else:
                console.send_control(stream.CONTROL_SET_INTERVAL, interval=seconds)
        elif word == "/pause":
            self._paused_here = True
            console.send_control(stream.CONTROL_PAUSE)
        elif word == "/resume":
            self._paused_here = False
            console.send_control(stream.CONTROL_RESUME)
        elif word == "/stop":
            if _ui.confirm("Stop the training run?", default=False):
                console.send_control(stream.CONTROL_STOP, reason="asked from the Pulse app")
        elif word == "/restart":
            self._restart()
        elif word == "/output":
            self._output()
        elif word == "/quiet":
            console.quiet = True
            self.note("Findings will not be announced here now (the run pane still shows them).")
        elif word == "/loud":
            console.quiet = False
            self.note("Findings are announced as they happen.")
        else:
            return False
        return True

    def _brain_agent_forced(self) -> Optional[Callable[[str], str]]:
        audits, self.audits = self.audits, True
        try:
            return self._brain_agent()
        finally:
            self.audits = audits

    def _set_audits(self, arg: str) -> None:
        arg = arg.lower()
        if arg in ("on", "off"):
            self.audits = arg == "on"
            _settings.set("audits", arg)
            if self.console is not None:
                agent = self._brain_agent()
                self.console.brain.agent = self.console.agent = agent
        state = "on" if (self.console is not None and self.console.brain.agent is not None) else "off"
        self.note(f"Scheduled audits are {state}."
                  + ("" if self.cli.agent_provider else " (No agent is set up: /agent.)"))

    # ------------------------------------------------------------------ starting a run

    def _launch(self, arg: str, replace: Optional[Dict[str, Any]] = None) -> None:
        """/run <script> [args]: start it under Pulse (stream mode) in the background, its
        output in a log file, and open it."""
        from . import pulse_console as con
        from . import pulse_supabase as cloud
        if replace is not None:
            argv, cwd = list(replace["argv"]), replace["cwd"]
        else:
            try:
                argv = shlex.split(arg)
            except ValueError as exc:
                self.note(f"/run could not read that: {exc}")
                return
            if not argv:
                self.note("usage: /run <script.py> [its arguments]")
                return
            cwd = self.home_root
            argv[0] = os.path.abspath(os.path.join(cwd, os.path.expanduser(argv[0])))
        script = argv[0]
        if not os.path.isfile(script):
            self.error(f"No such script: {script}")
            return
        log_dir = cloud.CACHE_PATH.parent / _RUN_LOG_DIR
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.error(f"Could not create {log_dir}: {exc}")
            return
        log_path = str(log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.path.basename(script)}.log")
        # Ensure the real pulse package is found (not shadowed by a root pulse.py file)
        pulse_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PULSE_NONINTERACTIVE="1", PYTHONUNBUFFERED="1",
                   PYTHONPATH=f"{pulse_pkg_root}:{os.environ.get('PYTHONPATH', '')}")
        started = time.time()
        with _ui.Stage(f"Starting {os.path.basename(script)}"):
            log = open(log_path, "ab", buffering=0)
            try:
                detach: Dict[str, Any] = {"start_new_session": True}
                if os.name == "nt":            # its own console group: it outlives the app
                    detach = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
                process = subprocess.Popen(
                    launch_argv(argv),
                    cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, **detach)
            finally:
                log.close()
            session = None
            deadline = time.monotonic() + 45.0
            while session is None:
                over = process.poll() is not None      # then this look is the last one
                for candidate in con.discover():
                    same = candidate.get("script") and os.path.realpath(candidate["script"]) == os.path.realpath(script)
                    if same and (candidate.get("pid") == process.pid
                                 or float(candidate.get("started") or 0) >= started - 1.0):
                        session = candidate
                        break
                if session is not None or over or time.monotonic() >= deadline:
                    break
                time.sleep(0.25)
        if session is None:
            code = process.poll()
            self.error(f"{os.path.basename(script)} "
                       + (f"exited with code {code} before Pulse saw a training step." if code is not None
                          else "has not reported to Pulse yet (no session after 45 s)."))
            self._print_tail(log_path, 25)
            return
        self.launched[session["session_id"]] = {"process": process, "argv": argv, "cwd": cwd, "log": log_path}
        self._open_run(session, replace=replace is not None)
        self.note(f"Started under Pulse. Its output goes to {log_path} (/output shows the end of it). "
                  "It keeps running if you leave Pulse.")

    def _restart(self) -> Optional[str]:
        """Stop the open run if it is still going, and start it again under Pulse with the
        current code. A run started here (/run) restarts with its own command; the script that
        opened this app (auto_track) with its arguments; a run started elsewhere only once it
        is over (its arguments are not known here). Returns why it refused, or None."""
        from . import pulse_stream as stream
        session = self.session or {}
        info = self.launched.get(session.get("session_id", ""))
        hosted = self.hosting is not None and session.get("session_id") == getattr(self.hosting, "session_id", None)
        over = self._status not in ("live", "stalled")
        script = session.get("script") or ""
        if info is None:
            if not script or not os.path.isfile(script):
                why = "Pulse does not know which script this run is, so it cannot start it again."
                self.note(why)
                return why
            if hosted:
                if not over:
                    # the script runs in this very process: stop it (its end leaves the app open)
                    self.console.send_control(stream.CONTROL_STOP, reason="restart asked from the Pulse app")
                    with _ui.Stage(f"Stopping {os.path.basename(script)}"):
                        deadline = time.monotonic() + 30.0
                        while self._status in ("live", "stalled") and time.monotonic() < deadline:
                            time.sleep(0.25)
                            self._poll_status()
                    if self._status in ("live", "stalled"):
                        why = f"{os.path.basename(script)} did not stop within 30 s, so it was not started again."
                        self.error(why)
                        return why
                args = sys.argv[1:] if sys.argv and os.path.realpath(sys.argv[0]) == os.path.realpath(script) else []
                self._launch("", replace={"argv": [script, *args], "cwd": os.getcwd()})
                return None
            if not over:
                why = (f"{os.path.basename(script)} was started outside Pulse's app and is still running: stop it "
                       "(/stop) and start it again the way it was started, or wait until it ends and /restart.")
                self.note(why)
                return why
            self.note(f"Starting {os.path.basename(script)} again under Pulse (without the arguments it was "
                      "first started with: Pulse does not know them).")
            self._launch("", replace={"argv": [script], "cwd": os.path.dirname(script) or self.home_root})
            return None
        process = info["process"]
        if self.runlog is not None:
            self.runlog.incident("restart", "The run was stopped to start again with the current code.")
        if process.poll() is None:
            self.console.send_control(stream.CONTROL_STOP, reason="restart asked from the Pulse app")
            with _ui.Stage("Stopping the run"):
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
        self._launch("", replace=info)
        return None

    def _poll_status(self) -> None:
        from . import pulse_console as con
        described = con._describe_session((self.session or {}).get("directory", ""))
        if described:
            self._status = described["status"]

    def _output(self) -> None:
        session = self.session or {}
        info = self.launched.get(session.get("session_id", ""))
        if info is None:
            self.note("/output shows what a run started here with /run printed. This run's output is in "
                      "the terminal it was started from.")
            return
        self._print_tail(info["log"], 40)

    def _print_tail(self, path: str, count: int) -> None:
        for line in _tail(path, count):
            print(_ui._s("  " + line, "dim"))

    # ================================================================== the run pane

    def _refresh_header(self) -> None:
        view = self.view
        cli = self.cli
        home = os.path.expanduser("~")

        def short(path: str) -> str:
            return "~" + path[len(home):] if path.startswith(home + os.sep) or path == home else path

        if self.console is not None and self.session is not None and not self.background:
            view.area = "DEBUG"
            view.context = f"{os.path.basename(self.session.get('script') or '?')}  ·  {short(self.console.workdir)}"
            view.commands = DEBUG_COMMANDS + HOME_COMMANDS
        elif (self.console is not None and self.session is not None) or self.parked:
            view.area = ""
            watched = ([(self.session or {}, self._status)] if self.console is not None else []) + [
                (state["session"], state["_status"]) for state in self.parked.values()]
            names = ", ".join(f"{os.path.basename(session.get('script') or '?')}"
                              + ("" if status in ("live", "stalled") else f" ({status})")
                              for session, status in watched)
            view.context = f"{short(self.home_root)}   watching {names} -- /change"
            view.commands = (DEBUG_COMMANDS if self.console is not None else []) + HOME_COMMANDS
        else:
            view.area = ""
            view.context = short(getattr(cli, "_project_root", None) or self.home_root)
            view.commands = HOME_COMMANDS
        view.agent = f"agent: {cli.agent_provider}" if cli.agent_provider else "no agent -- /agent"
        self.dirty = True

    def _refresh_side(self, force: bool = False) -> None:
        """Rebuild the run pane from the brain's current state (cheap; a few times a second)."""
        self._tick_parked(time.monotonic())
        console = self.console
        if console is None:
            return
        now = time.monotonic()
        if not force and now - self._side_at < 0.5:
            return
        if self.runlog is not None and (force or now - self._log_at >= 3.0):
            self._log_at = now
            try:
                self.runlog.tick(console.snapshot(), self._status, list(console.brain.events), self.session or {})
            except Exception:
                pass
        if self.background:
            if now - self._status_at > 2.0:
                from . import pulse_console as con
                self._status_at = now
                described = con._describe_session((self.session or {}).get("directory", ""))
                if described and described["status"] != self._status:
                    self._status = described["status"]
                    self._refresh_header()
                    self._note_stall(self._status)
                    if self._status == "crashed":
                        self._on_crash()
                    elif self._status not in ("live", "stalled"):
                        self.note(f"{os.path.basename((self.session or {}).get('script') or 'the run')} "
                                  f"has {self._status}.")
            self._side_at = now
            return
        from . import pulse_console as con
        self._side_at = now
        state = console.snapshot()
        session = self.session or {}
        if force or now - self._status_at > 2.0:      # a few file reads: not on every frame
            self._status_at = now
            described = con._describe_session(session.get("directory", ""))
            if described:
                self._status = described["status"]
                session["last_seen"] = described.get("last_seen")
                if self._status == "crashed":
                    self._on_crash()
                self._note_stall(described["status"])
        step = int(state["step"] or 0)
        if self._rate_at is not None and now > self._rate_at[0]:
            delta = (step - self._rate_at[1]) / (now - self._rate_at[0])
            if delta >= 0:
                self._rate = delta if self._rate is None else 0.7 * self._rate + 0.3 * delta
        self._rate_at = (now, step)
        width = tui.side_width(self._total_w) - 1 if self._total_w >= tui.SPLIT_MIN_WIDTH else self._total_w - 2
        side = side_lines(state, session, self._status, console, self._rate, width)
        # the run's output under its figures, in whatever room the pane has left
        # (at least a few lines, below the rest when the pane is longer than the screen:
        # the wheel over the pane reaches them)
        room = tui.body_height(self._height) - len(side) - 2
        if room < 3:
            room = 8
        if room >= 1:
            launched = self.launched.get(session.get("session_id", ""))
            if self.hosting is not None and session.get("session_id") == getattr(self.hosting, "session_id", None):
                tail = list(self.run_output)[-room:]
                if self._run_partial.strip() and len(tail) < room:
                    tail.append(self._run_partial)
            elif launched is not None:
                # the run's own lines; Pulse's notes about where it streams are not output
                tail = [line for line in _tail(launched["log"], room + 4)
                        if not line.startswith("[Pulse] ")][-room:]
            else:
                tail = []
            if tail:
                side.append("")
                side.append(_ui._s("Output", "bold"))
                side.extend(_ui._s(_ui._clip(row, width), "dim") for row in tail)
        self.view.side = side
        self.view.side_brief = side_brief(state, session, self._status, self._rate, self._total_w - 2)
        self.dirty = True

    _total_w = 100
    _status_at = 0.0
    _log_at = 0.0

    # ================================================================== the loop

    def loop(self) -> int:
        tui.Screen.mouse = self.mouse
        with tui.Screen() as screen, tui.KeyReader() as keys:
            self._screen = screen
            self.install()
            self.up.set()
            self._start_home_commands()
            self._start_live_feed()
            try:
                size = (0, 0)
                last_frame = 0.0
                while not self.done:
                    try:
                        now_size = screen.size()
                        with self.lock:
                            if now_size != size:
                                size, self.dirty = now_size, True
                                self._total_w, self._height = size
                            self._refresh_side(force=now_size != size and self.console is not None)
                            now = time.monotonic()
                            animating = bool(self.view.live) and now - last_frame >= 0.1
                            if self.dirty or animating:
                                self.view.frame += 1 if animating else 0
                                lines, row, col = tui.compose(self.view, *size)
                                self._frame = lines
                                if self._sel is not None and self._sel["head"] != self._sel["anchor"]:
                                    lines = tui.highlight(lines, self._sel)
                                split = self.view.side is not None and size[0] >= tui.SPLIT_MIN_WIDTH
                                self._pane_w = tui.pane_width(size[0], split)
                                self.dirty, last_frame = False, now
                            else:
                                lines = None
                        if lines is not None:
                            screen.draw(lines, row, col)
                        key = keys.read(0.05 if self.view.busy or self.console is not None else 0.25)
                    except KeyboardInterrupt:
                        key = "ctrl+c"
                    if key:
                        self.on_key(key)
                    self._start_pending_remote_command()
            finally:
                job = self._job
                if job is not None and job.is_alive():
                    question = self._question
                    if question is not None:
                        question.cancelled = True
                        question.done.set()
                    self._cancel_job()
                    job.join(timeout=2.0)
                console = self.console
                if console is not None:
                    try:
                        self._stop_runlog()
                        console.stop()
                    except Exception:
                        pass
                for state in list(self.parked.values()):     # the runs watched off screen
                    for stop in (lambda: state["runlog"] and state["runlog"].close(),
                                 lambda: state["console"].stop(),
                                 lambda: state["_monitor"] and state["_monitor"].stop()):
                        try:
                            stop()
                        except Exception:
                            pass
                self.parked.clear()
                if self._home_commands is not None:
                    self._home_commands.close()
                if self._live is not None:
                    self._live.close()
                if self._monitor is not None:
                    try:
                        self._monitor.stop()
                    except Exception:
                        pass
                self.uninstall()
                self._screen = None
        return self.exit_code


def _make_console(app: App, session: Dict[str, Any], agent: Optional[Callable[[str], str]]) -> Any:
    from . import pulse_console as con

    class AppConsole(con.Console):
        """The console's brain and views; what it prints goes to the app's transcript."""

        def _escalate(self, findings: List[Any], pack: Dict[str, Any]) -> None:
            pass                                 # the app's agent takes problems on (App._on_findings)

        def _check_crash(self) -> None:
            pass                                 # and crashes (App._on_crash)

        def _print(self, text: str) -> None:
            app.feed(text + "\n")

        def _announce(self, findings: List[Any]) -> None:
            if self is app.console:              # serious ones get the agent, /quiet or not
                with app.lock:
                    app._on_findings(findings)
            else:
                with app.lock:
                    app._on_parked_findings(self, findings)
            if self.quiet:
                return
            name = ""
            if self is not app.console:          # a run watched off screen: say which one
                name = os.path.basename((self.session or {}).get("script") or "") or "a watched run"
            for finding in findings:
                app._add(tui.Entry("finding", (f"{name}: " if name else "") + finding.message,
                                   severity=finding.severity))

        def _report_audit(self, record: Optional[Dict[str, Any]]) -> None:
            """A scheduled look at the run: one quiet line in the model's words. The decision
            JSON in its answer is for Pulse (when to look next), not for the person."""
            if not record or record.get("status") in ("skipped", "busy"):
                return
            if record.get("status") == "error":
                app.note(f"The scheduled check of the run failed: {record.get('error')}")
                return
            if record.get("status") is None:
                # an answer with no readable verdict: its words may still be the diagnosis
                text = audit_words(record.get("text") or "")
                if text:
                    app.note("Checked the run (no verdict could be read from the answer): " + text[:1200])
                return
            text = audit_words(record.get("text") or "")
            if self is not app.console:          # a run watched off screen: say which one
                name = os.path.basename((self.session or {}).get("script") or "") or "a watched run"
                text = f"{name}: {text}" if text else f"{name}."
            if record.get("status") == "problem":
                found = "; ".join(str(f) for f in (record.get("findings") or [])[:3])
                app.error("The agent's check found a problem" + (f": {found}" if found else "") + ".")
                if text:
                    app.say(text)
                # not just a line: the agent takes it on -- unless the check itself rates it
                # low risk ("problem, risk low" -- e.g. a cosmetic last-step gap -- is a note)
                if self is app.console and str(record.get("risk") or "").lower() != "low":
                    with app.lock:
                        app._on_findings([types.SimpleNamespace(
                            check="audit", variable=f"audit@{record.get('step')}", severity="critical",
                            message=(found or audit_words(record.get("text") or "") or "the scheduled check "
                                     "found a problem")[:600])])
                return
            app.note("Checked the run" + (f": {text}" if text else "."))

    return AppConsole(session, agent=agent)


_DECISION_RE = re.compile(r"\{[^{}]*\"(?:next_check_minutes|status)\"[^{}]*\}", re.S)


def audit_words(text: str) -> str:
    """An audit answer without its decision JSON (and the code fence around it, if any)."""
    text = _DECISION_RE.sub("", str(text))
    text = re.sub(r"```(?:json)?\s*```", "", text)
    return " ".join(text.split()).strip()


def _tail(path: str, count: int) -> List[str]:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [tui._SGR_RE.sub("", tui.clean(line.rsplit("\r", 1)[-1])) for line in text.splitlines()]
    return [line for line in lines if line.strip()][-count:]


def side_brief(state: Dict[str, Any], session: Dict[str, Any], status: str,
               rate: Optional[float], width: int) -> List[str]:
    """The run in two lines, for a terminal too narrow for the pane."""
    from . import pulse_console as con
    from . import pulse_detect as detect
    s, g = _ui._s, _ui._g
    name = os.path.basename(session.get("script") or "?")
    badge = f"{g('now')} {status}"
    gap = max(1, width - tui.visible_len(name) - len(badge))
    bits = [f"step {state['step']:,}" if state["step"] else "no steps yet"]
    if rate and status == "live":
        bits.append(f"{rate:.1f}/s" if rate >= 1 else f"{rate * 60:.1f}/min")
    for key, history in state["histories"].items():
        if history and detect.looks_like_loss(key):
            bits.append(f"{key} {con.compact(history[-1])}")
            break
    findings = state["findings"]
    if findings:
        worst = tui._SEVERITY_STYLE.get(str(findings[0].severity).lower(), "accent")
        bits.append(s(f"{len(findings)} finding{'s' if len(findings) != 1 else ''} (/findings)", worst))
    elif state["step"]:
        bits.append(s("healthy", "dim"))
    return [s(_ui._clip(name, width - len(badge) - 1), "bold") + " " * gap + s(badge, _STATUS_STYLE.get(status, "dim")),
            _ui._clip(f" {g('dot')} ".join(bits), width)]


def side_lines(state: Dict[str, Any], session: Dict[str, Any], status: str, console: Any,
               rate: Optional[float], width: int, launched: Optional[Dict[str, Any]] = None) -> List[str]:
    """The run pane: what the run is, where it is, every value with its curve, what the
    detectors believe. Pure: everything it shows is in its arguments. (The run's output
    goes under this, sized to the screen -- see App._refresh_side.)"""
    from . import pulse_console as con
    from . import pulse_detect as detect
    s, g = _ui._s, _ui._g
    width = max(24, width)
    name = os.path.basename(session.get("script") or "?")
    badge = f"{g('now')} {status}"
    gap = max(1, width - tui.visible_len(name) - len(badge))
    lines = [s(_ui._clip(name, width - len(badge) - 1), "bold") + " " * gap
             + s(badge, _STATUS_STYLE.get(status, "dim"))]
    home = os.path.expanduser("~")
    where = console.workdir
    lines.append(s(_ui._clip("~" + where[len(home):] if where.startswith(home) else where, width), "dim"))
    lines.append("")
    bits = [f"step {state['step']:,}" if state["step"] else "no steps yet"]
    if rate and status == "live":
        bits.append(f"{rate:.1f}/s" if rate >= 1 else f"{rate * 60:.1f}/min")
    started = session.get("started")
    if started:
        try:
            bits.append(_ui.elapsed_text(max(0.0, (session.get("last_seen") or time.time()) - float(started))
                                         if status not in ("live", "stalled") else time.time() - float(started)))
        except (TypeError, ValueError):
            pass
    lines.append(f" {g('dot')} ".join(bits))
    if state["gaps"]:
        lines.append(s(f"{state['gaps']} sample(s) dropped under load", "dim"))
    lines.append("")

    # a loop counter is the step, already shown above -- not a curve worth a row
    counters = {"step", "global_step", "iteration", "iter", "it", "i", "batch_idx"}
    histories = {k: v for k, v in state["histories"].items()
                 if v and not (k in counters and v[-1] == state["step"])}
    names = sorted(histories, key=lambda n: (not detect.looks_like_loss(n), n))
    label_w = min(14, max((len(n) for n in names), default=4))
    spark_w = max(6, width - label_w - 11)
    for name_ in names:                     # every one: the pane scrolls (the wheel over it)
        history = histories[name_]
        value = con.compact(history[-1])
        curve = con.sparkline(history, spark_w) if len(history) > 1 else ""
        lines.append(_ui._clip(name_, label_w).ljust(label_w) + " " + value.rjust(9) + " " + s(curve, "accent"))
    if not names:
        lines.append(s("no values yet", "dim"))
    lines.append("")

    findings = state["findings"]
    lines.append(s("Findings", "bold") + (s(f"  {len(findings)}", "dim") if findings else ""))
    if not findings:
        lines.append(s(g("ok") + " nothing the checks can see", "dim") if state["step"]
                     else s("waiting for the first steps", "dim"))
    for finding in findings[:5]:
        style = tui._SEVERITY_STYLE.get(str(finding.severity).lower(), "accent")
        mark = ("▲ " if _ui._unicode() else "! ")
        pieces = tui.wrap(finding.message, width - 2)[:3]
        lines.append(s(mark, style, "bold") + pieces[0])
        lines.extend("  " + piece for piece in pieces[1:])
    if len(findings) > 5:
        lines.append(s(f"+{len(findings) - 5} more (/findings)", "dim"))
    lines.append("")

    tensors = state["tensors"]
    if tensors:
        lines.append(s("Tensors", "bold"))
        for tensor, meta in sorted(tensors.items())[:5]:
            shape = "x".join(str(d) for d in (meta.get("shape") or []))
            lines.append(_ui._clip(f"{tensor}  " + s(f"{shape} {meta.get('dtype', '')}", "dim"), width))
        lines.append("")

    brain = console.brain
    lines.append(s("Agent", "bold"))
    if state["finished"] or status not in ("live", "stalled"):
        lines.append(s("the run is over", "dim"))
    elif brain.agent is not None:
        minutes = int(brain.schedule.seconds_remaining() / 60)
        lines.append(s(f"next audit in {minutes} min" if minutes else "audit due", "dim"))
    else:
        lines.append(s("audits off (/audits on)", "dim"))
    return lines


# ---------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------

def usable() -> bool:
    """Can the full-screen app run here? A real terminal, tall and wide enough, and not
    opted out (PULSE_CLASSIC=1 keeps the line-by-line screens)."""
    if os.environ.get("PULSE_CLASSIC", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if os.name == "nt" and not tui.windows_console_ready():
        return False            # a console that cannot show escape sequences
    if _ui.host() is not None or not _ui.enabled():
        return False
    try:
        size = os.get_terminal_size(sys.__stdout__.fileno())
    except (OSError, ValueError, AttributeError):
        return False
    return size.columns >= 40 and size.lines >= 12


def _new_cli(root: str, focus: List[str], review: bool = True, chdir: bool = True) -> Any:
    from . import pulse_code
    cli = pulse_code._CodeAgentCLI()
    cli.setup_code(root, focus, pulse_code.scan_project(root))
    cli.review = review
    if chdir:
        os.chdir(root)
    cli.set_code_text(cli.code_text, script_path=cli.script_path)
    cli._project_root = root
    return cli


def _welcome(app: App) -> None:
    from . import pulse_console as con
    cli = app.cli
    app.note("Pulse is ready. Describe what you want built or fixed, or open a run to debug it.")
    if app.mouse:
        app.note("Drag over any text to copy it; a click opens a folded line; the wheel scrolls; "
                 "↑ brings back what you typed.")
    if not cli.agent_provider:
        app.note("No agent yet: /agent picks a model (OpenRouter models need no key -- you can sign in).")
    try:
        live = [s for s in con.discover() if s["status"] in ("live", "stalled")]
    except Exception:
        live = []
    if live:
        names = ", ".join(sorted({os.path.basename(s.get("script") or "?") for s in live})[:4])
        app.note(f"{len(live)} run{'s' if len(live) != 1 else ''} live on this machine ({names}). "
                 "/monitor opens one beside the agent.")
    else:
        app.note("/monitor lists the runs on this machine · /run train.py starts one · /help for the rest.")


def _goodbye(app: App) -> None:
    going = [info for info in app.launched.values() if info["process"].poll() is None]
    for info in going:
        print(f"[Pulse] {os.path.basename(info['argv'][0])} is still running (pid {info['process'].pid}); "
              f"its output is in {info['log']}. `pulse` opens it again.")


def run(paths: Any = (), yes: bool = False, root: Optional[str] = None) -> int:
    """`pulse` and `pulse code [paths]`: setup, then the app at HOME."""
    from . import pulse_code
    from .pulse_cli import cprint, _RED, _YELLOW
    root = os.path.abspath(root or os.getcwd())
    focus, problems = pulse_code.expand_paths(root, list(paths))
    for problem in problems:
        cprint(f"[Pulse] {problem}", color=_YELLOW)
    if paths and not focus:
        cprint("[Pulse] None of those paths could be used.", color=_RED)
        return 1
    outside = [f for f in focus if not pulse_code._inside(root, f)]
    if outside:
        cprint(f"[Pulse] {os.path.relpath(outside[0], root)} is outside the project root {root}; "
               "run `pulse` from a directory that contains it.", color=_RED)
        return 1
    cli = _new_cli(root, focus, review=not yes and _settings.on("review"))
    cli.print_banner()
    try:
        cli.interactive_setup()                      # the setup Pulse always had, unchanged
    except (EOFError, KeyboardInterrupt):
        print("\n[Pulse] Setup cancelled.")
        return 1
    signal.signal(signal.SIGINT, signal.default_int_handler)   # the debugger's handler means "pause the run"
    cli.reload()
    app = App(cli, root)
    _welcome(app)
    try:
        return app.loop()
    finally:
        try:
            cli._flush_cloud_now()
        except Exception:
            pass
        _goodbye(app)


def watch_in_process(monitor: Any, model: str = "") -> Optional[App]:
    """auto_track() on a terminal: the app opens on this very run, beside the training.
    The script keeps the main thread and runs exactly as it would have; the app runs on a
    thread of its own, the script's output goes to the transcript, and the run's state,
    the detectors and the agent are the DEBUG screen as for any other run. Leaving the
    app (Ctrl+C twice, /quit) gives the terminal back and the training goes on; when the
    script ends, the app stays until the person leaves it, so the end of the run and
    the agent's verdict are not lost with the process. Returns the app, or None when the
    terminal cannot show it (the caller falls back to the line-by-line mode)."""
    from . import pulse_console as con
    from . import pulse_monitor
    if not usable() or threading.current_thread() is not threading.main_thread():
        return None
    root = os.getcwd()
    cli = _new_cli(root, [], chdir=False, review=_settings.on("review"))
    _pick_agent_quietly(cli, model)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    app = App(cli, root)
    app.hosting = monitor
    script = os.path.basename(getattr(monitor, "script_path", None) or "the script")
    session = {"session_id": monitor.session_id, "directory": monitor.directory,
               "script": getattr(monitor, "script_path", None), "status": "live", "step": 0}
    over = threading.Event()               # the script has ended

    def show() -> None:
        try:
            app.install()
            try:
                app._open_run(session)
                app.note(f"{script} is running here, under Pulse. Ask about it any time; Ctrl+C twice "
                         "leaves Pulse and the training goes on."
                         + (" Drag over any text to copy it; the wheel over the run's figures "
                            "scrolls them." if app.mouse else ""))
            finally:
                app.uninstall()
            threading.Thread(target=sign_in, name="pulse-app-signin", daemon=True).start()
            app.loop()
        except BaseException:              # the app must never take the training down with it
            try:
                app.uninstall()
            except Exception:
                pass
        finally:
            app.done = True
            app.up.set()
            if not over.is_set():
                print(f"[Pulse] Left the app; {script} goes on. `pulse` then /monitor opens it again.")

    def sign_in() -> None:
        # Signed in before on this machine: the run goes on the dashboard as it did in
        # the line-by-line mode, without a question. Never signed in: nothing is asked.
        from . import pulse_supabase as cloud
        try:
            if not cloud.load_cached_credentials():
                return
            cli.non_interactive = True
            try:
                with app.hush():
                    cli._auth_flow()
            finally:
                cli.non_interactive = False
            # the workspace and project from the settings, as the line-by-line setup uses them
            workspace, project = _settings.remembered_id("workspace"), _settings.remembered_id("project")
            if workspace:
                cli.team_id = workspace
                cli.project_id = project if project else None
            console = app.console
            if cli.user_id and console is not None and app.runlog is None and not app.done:
                app._start_runlog(session, console.workdir)
        except Exception:
            pass

    thread = threading.Thread(target=show, name="pulse-app", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0     # the script goes on once the screen captures its output
    while not app.up.is_set() and thread.is_alive() and not app.done and time.monotonic() < deadline:
        time.sleep(0.02)

    def at_exit() -> None:
        # The script is over (its end, or a crash already shown in the transcript). The
        # run is marked finished now, so the screen says so, and the app stays until the
        # person leaves it.
        over.set()
        if app.done or not thread.is_alive():
            return
        try:
            pulse_monitor.detach()
        except Exception:
            pass
        error = getattr(sys, "last_value", None)
        if error is not None and not isinstance(error, KeyboardInterrupt):
            app.note(f"{script} stopped with an error. Ctrl+C twice (or /quit) leaves Pulse.")
        else:
            app.note(f"{script} has finished. Ctrl+C twice (or /quit) leaves Pulse.")
        with _shutdown_held():
            thread.join()

    _register_end_hook(at_exit)
    return app


def _copy_with_tool(text: str) -> bool:
    """The system clipboard through the platform's own tool, when there is one (the terminal
    route, OSC 52, is not supported everywhere)."""
    import shutil
    if sys.platform == "darwin":
        tools = [["pbcopy"]]
    elif os.name == "nt":
        tools = [["clip"]]
    else:
        tools = ([["wl-copy"]] if os.environ.get("WAYLAND_DISPLAY") else []) + (
            [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]] if os.environ.get("DISPLAY") else [])
    for argv in tools:
        if shutil.which(argv[0]) is None:
            continue
        try:
            subprocess.run(argv, input=text.encode("utf-16-le" if argv[0] == "clip" else "utf-8"),
                           timeout=5, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _register_end_hook(hook: Callable[[], None]) -> None:
    """Run `hook` when the script is over -- early in the interpreter's shutdown, while the
    agent can still do its work. A plain atexit hook runs too late for that: by then Python
    refuses new thread pools ("cannot schedule new futures after interpreter shutdown"), and
    a library imported then fails with "can't register atexit after shutdown", so a model
    call made after the script ended (the agent looking at a crash) failed. The hooks
    threading runs before it joins threads come first; concurrent.futures registers its own
    there on import, so it is imported before ours is added (they run in reverse order: ours
    first, the pools still usable)."""
    try:
        import concurrent.futures.thread  # noqa: F401
        register = threading._register_atexit          # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        import atexit
        atexit.register(hook)
        return
    register(hook)


@contextlib.contextmanager
def _shutdown_held():
    """While the app outlives the script: a library imported now may still register its own
    exit hook (threading refuses once its shutdown has begun)."""
    was = getattr(threading, "_SHUTTING_DOWN", None)
    if was:
        threading._SHUTTING_DOWN = False                # type: ignore[attr-defined]
    try:
        yield
    finally:
        if was:
            threading._SHUTTING_DOWN = was              # type: ignore[attr-defined]


def _pick_agent_quietly(cli: Any, model: str = "") -> None:
    """The agent without a question: `model` (--model / PULSE_MODEL) if given, else the one in
    the settings (with its saved key), else the one used last time if its key is in the
    environment, else any provider whose key is. Nothing found leaves the agent unset
    (/agent sets one from inside)."""
    from . import pulse_supabase as cloud
    if not model and _settings.apply_agent(cli):
        return
    _settings.load_keys_into_environment()        # a saved key serves the fallbacks below too
    wanted = [model] if model else []
    try:
        last = str(cloud.load_cached_profile().get("agent_provider") or "").strip()
    except Exception:
        last = ""
    if last and last not in wanted:
        wanted.append(last)
    wanted.append("")                      # "": whichever provider has a key
    previous = os.environ.get("PULSE_PROVIDER")
    cli.non_interactive = True
    try:
        for want in wanted:
            if want:
                os.environ["PULSE_PROVIDER"] = want
            else:
                os.environ.pop("PULSE_PROVIDER", None)
            try:
                with contextlib.redirect_stdout(io.StringIO()):     # its "[Pulse] agent set to" lines
                    if cli._select_agent_provider_and_key(initial=True):
                        return
            except Exception:
                pass
    finally:
        cli.non_interactive = False
        if previous is None:
            os.environ.pop("PULSE_PROVIDER", None)
        else:
            os.environ["PULSE_PROVIDER"] = previous


def open_run(session: Dict[str, Any], model: str = "", monitor: Any = None) -> int:
    """`pulse watch`, `pulse <script>`, `pulse attach`: the app, opened on that run. No
    account setup here -- the point is to be looking at the run at once; `--model` (or
    PULSE_MODEL) sets the agent, and /agent sets or changes it from inside."""
    root = os.getcwd()
    cli = _new_cli(root, [], review=_settings.on("review"))
    _pick_agent_quietly(cli, model)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    app = App(cli, root)
    app.install()
    try:
        app._open_run(session, monitor=monitor)
    finally:
        app.uninstall()
    try:
        return app.loop()
    finally:
        _goodbye(app)
