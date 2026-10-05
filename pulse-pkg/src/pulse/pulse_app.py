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
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import pulse_tui as tui
from . import pulse_ui as _ui

DEBUG_PROMPT = (
    "\n\nYOU ARE ATTACHED TO A LIVE TRAINING RUN\n"
    "The person opened a run in the Pulse app and is looking at it with you. Each request comes "
    "with EVIDENCE FROM THE RUN: its step, every tracked value's curve, tensors, the detectors' "
    "findings and recent events. Use it and quote the numbers. Diagnose before you conclude: read "
    "the code with the tools, check what you suspect. A question about the run is a question -- "
    "answer it in words, with no code change. Change code only when asked to fix something, or "
    "when the person agrees to a fix you proposed. The run keeps executing the code it started "
    "with. After an edit, decide whether to run it again with restart_run (RESTART: in the text "
    "protocol): do when the fix only takes effect in a fresh run or you need to see it work (after "
    "a crash, almost always); don't when the run is healthy and the change can wait for its next "
    "start -- and say which you chose. run_status "
    "(RUNSTATUS:) gives the run's latest numbers whenever you need them again.\n"
)

HOME_COMMANDS: List[Tuple[str, str]] = [
    ("/monitor", "pick a run on this machine and open it beside the agent"),
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
    ("/back", "return to the agent; the run stays open in the background"),
    ("/close", "stop watching the open run"),
]

_STATUS_STYLE = {"live": "accent", "stalled": "bold", "crashed": "red"}
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


class RunLog:
    """One Debug_Sessions row for a run the app watches, so it shows on the web dashboard
    like a run started with `pulse run`: a first telemetry entry about the environment,
    metric snapshots, the detectors' findings and status changes as incidents, crashes as
    error tracebacks, the agent's turns about the run, and uptime. Everything is sent in
    the background and never blocks the screen; without a signed-in account there is
    nothing to send and this is inert."""

    TELEMETRY_EVERY = 60.0
    FLUSH_EVERY = 60.0

    def __init__(self, cli: Any, session: Dict[str, Any], workdir: str) -> None:
        self.cli = cli
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
            if now - self._last_telemetry >= self.TELEMETRY_EVERY or not self._last_telemetry:
                self._last_telemetry = now
                snapshot: Dict[str, Any] = {"t": time.time(), "step": state.get("step") or 0}
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

    def agent_turn(self, question: str, answer: str, fix_applied: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._closed:
            return
        with self._lock:
            entry: Dict[str, Any] = {"t": time.time(), "question": question, "answer": answer}
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
        from . import pulse_supabase as cloud
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
        try:
            from . import pulse_supabase as cloud
            self.mouse = cloud.load_cached_profile().get("app_mouse", "on") != "off"
        except Exception:
            self.mouse = True
        self.run_output: "collections.deque[str]" = collections.deque(maxlen=400)   # what that run printed
        self._crashes_seen: set = set()            # sessions whose crash the agent was started on
        self._crash_pending: Optional[str] = None  # a crash that came while the agent was busy
        self._auto_crash_turns = 0                 # crash turns started without the person typing
        self._run_partial = ""
        self.up = threading.Event()               # set once the screen is up and output is captured
        self._height = 30
        self._question: Optional[_Question] = None
        self._draft = ""
        self._hushed: Dict[int, List[str]] = {}      # thread id -> what it printed while hushed
        self._live_thinking: Optional[tui.Entry] = None
        self._live_answer: Optional[tui.Entry] = None
        self._think_started = 0.0
        self._streamed_reasoning = False
        self._job: Optional[threading.Thread] = None
        self._last_ctrl_c = 0.0
        self._ui_thread = threading.current_thread()
        # the run that is open (DEBUG), if any
        self.console: Any = None
        self.session: Optional[Dict[str, Any]] = None
        self.background = False                      # the run is watched, its pane is not shown
        self.runlog: Optional[RunLog] = None         # the run's row on the dashboard
        self._orig_sync: Any = None
        self._monitor: Any = None                    # a py-spy sampler this app started
        self.audits = True
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
        if self._edit_pending is not None:
            # a question while a change is on the table ("Apply this change?") is about that
            # change: open its diff so the person sees what they are answering
            with self.lock:
                self._edit_pending.open = True
                self._edit_pending.touch()
        if threading.current_thread() is self._ui_thread:
            raise _ui.Unavailable("a question cannot be asked from the drawing thread")
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
        question.done.wait()
        with self.lock:
            view = self.view
            view.question, view.secret, view.options, view.option_ids = "", False, None, []
            view.editor.take(remember=False)
            view.editor.set(self._draft)
            self._question = None
            self.dirty = True
        if question.cancelled:
            raise KeyboardInterrupt
        return question.answer

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
        finally:
            with self.lock:
                self._flush_partial()
                self.view.busy, self.view.live, self.view.status = False, "", ""
                self._job = None
                self._refresh_header()
                self.dirty = True
                pending, self._crash_pending = self._crash_pending, None
                if pending and not self.done:
                    self._start_crash_turn(pending, self._crash_summary())

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
            if key in ("wheel:up", "wheel:down"):
                view.scroll = max(0, view.scroll + (3 if key == "wheel:up" else -3))
                return
            if key.startswith("click:"):
                _x, row = key.split(":")[1:]
                entry = view.row_entries.get(int(row))
                if entry is not None and entry.foldable():
                    entry.toggle(view.expanded)
                    tui.keep_in_place(view, entry, int(row))
                return
            if key in ("up", "down") and not view.editor.text and question is None:
                # an empty input line: the arrows (and a mouse wheel, which most terminals turn
                # into arrows on this screen) scroll the transcript
                view.scroll = max(0, view.scroll + (3 if key == "up" else -3))
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

    def _handle(self, line: str) -> None:
        if line.startswith("/") or line == "?":
            self._command(line)
            return
        cli = self.cli
        if not cli.agent_provider or not cli.agent_key:
            self.note("No agent is set up yet. /agent picks a model and a key "
                      "(or signs you in to OpenRouter, which needs no key).")
            return
        from . import pulse_code
        evidence = self._evidence() if self.console is not None else None
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
        elif word == "/run":
            self._launch(rest)
        elif word == "/mouse":
            self._set_mouse(rest)
        elif word == "/copy":
            self._copy(rest)
        elif self.console is not None and self._run_command(word, rest):
            pass
        elif word == "/close":
            self.note("No run is open.")
        elif word in ("/back", "/home"):
            self.note("No run is open. /monitor picks one.")
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
        from . import pulse_supabase as cloud
        cloud.save_cached_profile(app_mouse="on" if on else "off")
        self.note("Mouse on: a click opens a folded line, the wheel scrolls (Shift+drag selects text in most "
                  "terminals)." if on else
                  "Mouse off: drag to select text as usual. Ctrl+O opens folded lines, the arrows scroll. "
                  "/mouse on gives clicks back.")

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

    def _pick_run(self, wanted: str = "") -> None:
        from . import pulse_console as con
        sessions = con.discover()
        idle = con.unmonitored_python_processes()
        current = (self.session or {}).get("session_id")
        if not wanted and self.background and current:
            self._foreground_run()                   # /monitor alone: the run that is already watched
            return
        if wanted:
            chosen = con.pick_session(sessions, wanted)
            if chosen is not None:
                if chosen.get("session_id") == current:
                    self._foreground_run()
                else:
                    self._open_run(chosen)
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
        if kind == "session" and target.get("session_id") == current:
            self._foreground_run()
        elif kind == "session":
            self._open_run(target)
        else:
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

    def _open_run(self, session: Dict[str, Any], monitor: Any = None) -> None:
        self._close_run(quiet=True)
        console = _make_console(self, session, self._brain_agent())
        console.brain.poll_once()
        console.start()
        with self.lock:
            self.console, self.session, self._monitor = console, session, monitor
            self._rate = self._rate_at = None
            self._status = session.get("status") or "live"
        script = session.get("script") or ""
        workdir = console.workdir
        if os.path.isdir(workdir):
            self._point(workdir, [script] if script and os.path.isfile(script) else [])
        from . import pulse_code
        self.cli._system_prompt_override = pulse_code.CODE_SYSTEM_PROMPT + DEBUG_PROMPT
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
        log = RunLog(self.cli, session, workdir)
        if not log.enabled:
            return
        self.runlog = log
        log.start()
        cli = self.cli
        self._orig_sync = cli._sync_agent_turn

        def sync(question: str, answer: str, traceback_signature: Optional[str] = None,
                 fix_applied: Optional[Dict[str, Any]] = None) -> None:
            try:
                self._orig_sync(question, answer, traceback_signature=traceback_signature, fix_applied=fix_applied)
            finally:
                log.agent_turn(question, answer, fix_applied=fix_applied)

        cli._sync_agent_turn = sync

    def _stop_runlog(self) -> None:
        log, self.runlog = self.runlog, None
        if self._orig_sync is not None:
            self.cli._sync_agent_turn = self._orig_sync
            self._orig_sync = None
        if log is not None:
            log.close()

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
        if self._auto_crash_turns > _AUTO_CRASH_TURNS:
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
        self._start(self._handle, request)

    def _background_run(self) -> None:
        """Back to the agent at home, with the run still watched: its pane goes, findings are
        still announced, and /monitor (or Esc) brings it back. The agent stays pointed at the
        run's project, so a fix for it can still be asked for from home."""
        if self.console is None:
            return
        with self.lock:
            self.background = True
            self.view.side, self.view.side_brief = None, []
        self._refresh_header()
        name = os.path.basename((self.session or {}).get("script") or "the run")
        self.note(f"{name} is still watched in the background: findings show up here, /monitor "
                  "brings it back, /close stops watching it.")

    def _foreground_run(self) -> None:
        if self.console is None or not self.background:
            return
        with self.lock:
            self.background = False
        self._refresh_header()
        self._refresh_side(force=True)

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
            self._close_run()
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
            console.send_control(stream.CONTROL_PAUSE)
        elif word == "/resume":
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
                    detach = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                              | getattr(subprocess, "DETACHED_PROCESS", 0)}
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
        self._open_run(session)
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
        elif self.console is not None and self.session is not None:
            view.area = ""
            view.context = (f"{short(self.console.workdir)}   watching "
                            f"{os.path.basename(self.session.get('script') or '?')} ({self._status}) -- /monitor")
            view.commands = DEBUG_COMMANDS + HOME_COMMANDS
        else:
            view.area = ""
            view.context = short(getattr(cli, "_project_root", None) or self.home_root)
            view.commands = HOME_COMMANDS
        view.agent = f"agent: {cli.agent_provider}" if cli.agent_provider else "no agent -- /agent"
        self.dirty = True

    def _refresh_side(self, force: bool = False) -> None:
        """Rebuild the run pane from the brain's current state (cheap; a few times a second)."""
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
        step = int(state["step"] or 0)
        if self._rate_at is not None and now > self._rate_at[0]:
            delta = (step - self._rate_at[1]) / (now - self._rate_at[0])
            if delta >= 0:
                self._rate = delta if self._rate is None else 0.7 * self._rate + 0.3 * delta
        self._rate_at = (now, step)
        width = tui.side_width(self._total_w) - 1 if self._total_w >= tui.SPLIT_MIN_WIDTH else self._total_w - 2
        side = side_lines(state, session, self._status, console, self._rate, width)
        # the run's output under its figures, in whatever room the pane has left
        room = tui.body_height(self._height) - len(side) - 2
        if room >= 2:
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

        def _print(self, text: str) -> None:
            app.feed(text + "\n")

        def _announce(self, findings: List[Any]) -> None:
            if self.quiet:
                return
            for finding in findings:
                app._add(tui.Entry("finding", finding.message, severity=finding.severity))

        def _report_audit(self, record: Optional[Dict[str, Any]]) -> None:
            """A scheduled look at the run: one quiet line in the model's words. The decision
            JSON in its answer is for Pulse (when to look next), not for the person."""
            if not record or record.get("status") in (None, "skipped", "busy"):
                return
            if record.get("status") == "error":
                app.note(f"The scheduled check of the run failed: {record.get('error')}")
                return
            text = audit_words(record.get("text") or "")
            if record.get("status") == "problem":
                found = "; ".join(str(f) for f in (record.get("findings") or [])[:3])
                app.error("The agent's check found a problem" + (f": {found}" if found else "")
                          + ". Ask about it here, or /findings.")
                if text:
                    app.say(text)
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
    for name_ in names[:10]:
        history = histories[name_]
        value = con.compact(history[-1])
        curve = con.sparkline(history, spark_w) if len(history) > 1 else ""
        lines.append(_ui._clip(name_, label_w).ljust(label_w) + " " + value.rjust(9) + " " + s(curve, "accent"))
    if len(names) > 10:
        lines.append(s(f"+{len(names) - 10} more (/vars)", "dim"))
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
        app.note("A click opens a folded line. To select text hold Shift while you drag, or /mouse off; "
                 "/copy copies the agent's last answer.")
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
    cli = _new_cli(root, focus, review=not yes)
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
    cli = _new_cli(root, [], chdir=False)
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
                         + (" To select text hold Shift while you drag, or /mouse off; /copy copies the "
                            "agent's last answer." if app.mouse else ""))
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
    """The agent without a question: `model` (--model / PULSE_MODEL) if given, else the one
    used last time if its key is still in the environment, else any provider whose key is.
    Nothing found leaves the agent unset (/agent sets one from inside)."""
    from . import pulse_supabase as cloud
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
    cli = _new_cli(root, [])
    if model:
        previous = os.environ.get("PULSE_PROVIDER")
        os.environ["PULSE_PROVIDER"] = model
        cli.non_interactive = True
        try:
            cli._select_agent_provider_and_key(initial=True)
        except Exception:
            pass
        finally:
            cli.non_interactive = False
            if previous is None:
                os.environ.pop("PULSE_PROVIDER", None)
            else:
                os.environ["PULSE_PROVIDER"] = previous
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
