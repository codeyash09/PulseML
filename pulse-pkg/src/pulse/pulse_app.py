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
import contextlib
import ctypes
import getpass
import io
import math
import os
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
    "with: after an edit, say that the run has to be restarted (/restart) for it to take effect.\n"
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
    ("/restart", "stop and start the run again (runs started with /run)"),
    ("/output", "the last lines the run printed (runs started with /run)"),
    ("/interval", "sample faster or slower: /interval 0.5"),
    ("/quiet", "stop announcing findings (/loud resumes)"),
    ("/back", "close the run and return to the code agent"),
]

_STATUS_STYLE = {"live": "green", "stalled": "amber", "crashed": "red"}
_RUN_LOG_DIR = "app-runs"


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
        self._height = 30
        self._question: Optional[_Question] = None
        self._draft = ""
        self._hushed: Dict[int, List[str]] = {}      # thread id -> what it printed while hushed
        self._job: Optional[threading.Thread] = None
        self._last_ctrl_c = 0.0
        self._ui_thread = threading.current_thread()
        # the run that is open (DEBUG), if any
        self.console: Any = None
        self.session: Optional[Dict[str, Any]] = None
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

    def tool(self, calls: List[str], output: str) -> None:
        output = tui.clean(output).replace("\r", "")
        self._add(tui.Entry("tool", calls=calls or ["tools"], output=output))

    def _observe(self, event: str, **data: Any) -> None:
        if event == "reasoning" and data.get("text"):
            self._add(tui.Entry("thinking", data["text"]))

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

    def _cancel_job(self) -> None:
        job = self._job
        if job is None or job.ident is None:
            return
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(job.ident), ctypes.py_object(KeyboardInterrupt))

    # ================================================================== keys

    def on_key(self, key: str) -> None:
        view = self.view
        with self.lock:
            self.dirty = True
            question = self._question
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
                    view.status = "Ctrl+C again to leave Pulse"
                return
            view.status = ""
            if key == "ctrl+o":
                view.expanded = not view.expanded
                return
            if key in ("pgup", "pgdn"):
                page = max(3, self._height // 2)
                view.scroll = max(0, view.scroll + (page if key == "pgup" else -page))
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
                    self._start(self._handle, line)
            elif key == "tab":
                hints = tui._hints(view)
                if hints:
                    view.editor.set(hints[0][0] + " ")
            elif key == "esc":
                if self.console is not None and not view.busy and not view.editor.text:
                    self._start(self._close_run)
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
        outcome = pulse_code.run_turn(cli, line, evidence=evidence)
        if outcome == "applied" and self.console is not None:
            try:
                self.console.brain.note_fix("code edited from the Pulse app")
            except Exception:
                pass
            launched = self.session is not None and self.session.get("session_id") in self.launched
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
        elif self.console is not None and self._run_command(word, rest):
            pass
        elif word in ("/back", "/home"):
            self.note("No run is open. /monitor picks one.")
        elif not pulse_code._handle_command(self.cli, line):
            self.done = True

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
        if wanted:
            chosen = con.pick_session(sessions, wanted)
            if chosen is not None:
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
        self._refresh_header()
        name = os.path.basename(script) or session.get("session_id") or "the run"
        over = "" if self._status in ("live", "stalled") else f" ({self._status})"
        self.note(f"Opened {name}{over}. Ask about the run or tell the agent what to fix; /help lists "
                  "the commands, Esc closes the run.")
        if console.brain.agent is not None:
            self.note("The agent audits this run on its own schedule (/audits off stops that).")
        with self.lock:
            self._refresh_side(force=True)

    def _close_run(self, quiet: bool = False) -> None:
        console = self.console
        if console is None:
            return
        console.stop(join=True)
        monitor = self._monitor
        with self.lock:
            self.console = self.session = self._monitor = None
            self.view.side, self.view.side_brief = None, []
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
        self._point(self.home_root, self.home_focus)
        self._refresh_header()
        if not quiet:
            self.note("Closed the run. It keeps going; /monitor opens it again.")

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
        env = dict(os.environ, PULSE_NONINTERACTIVE="1", PYTHONUNBUFFERED="1")
        started = time.time()
        with _ui.Stage(f"Starting {os.path.basename(script)}"):
            log = open(log_path, "ab", buffering=0)
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "pulse", "run", "--stream", "--again", *argv],
                    cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True)
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

    def _restart(self) -> None:
        from . import pulse_stream as stream
        session = self.session or {}
        info = self.launched.get(session.get("session_id", ""))
        if info is None:
            self.note("/restart works for runs started here with /run. This one was started elsewhere: "
                      "stop it (/stop) and start it again the way it was started.")
            return
        process = info["process"]
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

        if self.console is not None and self.session is not None:
            view.area = "DEBUG"
            view.context = f"{os.path.basename(self.session.get('script') or '?')}  ·  {short(self.console.workdir)}"
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
        step = int(state["step"] or 0)
        if self._rate_at is not None and now > self._rate_at[0]:
            delta = (step - self._rate_at[1]) / (now - self._rate_at[0])
            if delta >= 0:
                self._rate = delta if self._rate is None else 0.7 * self._rate + 0.3 * delta
        self._rate_at = (now, step)
        width = tui.side_width(self._total_w) - 1 if self._total_w >= tui.SPLIT_MIN_WIDTH else self._total_w - 2
        self.view.side = side_lines(state, session, self._status, console, self._rate, width,
                                    self.launched.get(session.get("session_id", "")))
        self.view.side_brief = side_brief(state, session, self._status, self._rate, self._total_w - 2)
        self.dirty = True

    _total_w = 100
    _status_at = 0.0

    # ================================================================== the loop

    def loop(self) -> int:
        with tui.Screen() as screen, tui.KeyReader() as keys:
            self.install()
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
                                self._pane_w = size[0] - (tui.side_width(size[0]) + 3 if split else 1) - 1
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
                        console.stop()
                    except Exception:
                        pass
                if self._monitor is not None:
                    try:
                        self._monitor.stop()
                    except Exception:
                        pass
                self.uninstall()
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

    return AppConsole(session, agent=agent)


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
        worst = tui._SEVERITY_STYLE.get(str(findings[0].severity).lower(), "amber")
        bits.append(s(f"{len(findings)} finding{'s' if len(findings) != 1 else ''} (/findings)", worst))
    elif state["step"]:
        bits.append(s("healthy", "green"))
    return [s(_ui._clip(name, width - len(badge) - 1), "bold") + " " * gap + s(badge, _STATUS_STYLE.get(status, "dim")),
            _ui._clip(f" {g('dot')} ".join(bits), width)]


def side_lines(state: Dict[str, Any], session: Dict[str, Any], status: str, console: Any,
               rate: Optional[float], width: int, launched: Optional[Dict[str, Any]] = None) -> List[str]:
    """The run pane: what the run is, where it is, every value with its curve, what the
    detectors believe. Pure: everything it shows is in its arguments."""
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
        lines.append(s(g("ok") + " nothing the checks can see", "green") if state["step"]
                     else s("waiting for the first steps", "dim"))
    for finding in findings[:5]:
        style = tui._SEVERITY_STYLE.get(str(finding.severity).lower(), "amber")
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
    if launched is not None:
        tail = _tail(launched["log"], 4)
        if tail:
            lines.append("")
            lines.append(s("Output", "bold"))
            lines.extend(s(_ui._clip(row, width), "dim") for row in tail)
    return lines


# ---------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------

def usable() -> bool:
    """Can the full-screen app run here? A real terminal, tall and wide enough, and not
    opted out (PULSE_CLASSIC=1 keeps the line-by-line screens)."""
    if os.environ.get("PULSE_CLASSIC", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if os.name == "nt" and os.environ.get("PULSE_APP", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False            # not tried on a Windows console yet: PULSE_APP=1 opts in
    if _ui.host() is not None or not _ui.enabled():
        return False
    try:
        size = os.get_terminal_size(sys.__stdout__.fileno())
    except (OSError, ValueError, AttributeError):
        return False
    return size.columns >= 40 and size.lines >= 12


def _new_cli(root: str, focus: List[str], review: bool = True) -> Any:
    from . import pulse_code
    cli = pulse_code._CodeAgentCLI()
    cli.setup_code(root, focus, pulse_code.scan_project(root))
    cli.review = review
    os.chdir(root)
    cli.set_code_text(cli.code_text, script_path=cli.script_path)
    cli._project_root = root
    return cli


def _welcome(app: App) -> None:
    from . import pulse_console as con
    cli = app.cli
    app.note("Pulse is ready. Describe what you want built or fixed, or open a run to debug it.")
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
