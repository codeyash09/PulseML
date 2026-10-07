"""Headless mode: Pulse debugs a run by itself, in the background, with no screen.

    pulse run --headless train.py --epochs 3      # start the run and walk away
    auto_track(mode="headless")                   # or PULSE_MODE=headless: same, from the script
    pulse headless                                # what is being debugged in the background
    pulse headless stop [pid|script]              # stop debugging (the training goes on)

A supervisor process runs detached from the terminal. It is the Pulse app without its
screen: it launches the training (or attaches to a script that called auto_track), reads
its stream, runs the checks and the scheduled audits, and when the run crashes, a check
finds something serious or an audit says "problem", it starts the agent on it -- which
reads the code, fixes it and decides whether to restart the run. Nobody is asked anything:
every command and every code change goes through the reviewer model (auto mode), and when
there is no reviewer, or it cannot answer, the answer is no. After three automatic fixes in
a row it stops acting and only watches, until the run has been healthy for a while.

Everything it does goes to a log file (~/.pulse/headless/) and, when this machine is signed
in to Pulse Cloud, to the run's row on the dashboard. `pulse`, then /monitor, opens the run
in the app at any time.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import pulse_app
from . import pulse_tui as tui

DASHBOARD_URL = "https://pulsedashb.netlify.app/"

# Seconds a finished run is kept after the last thing happened (a fix being written, the
# closing audit), before the supervisor stops.
_GRACE_SECONDS = float(os.environ.get("PULSE_HEADLESS_GRACE", "20"))
# terminal control sequences (colours, "clear to end of line"): not for a log file
_CONTROL_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
# After this long healthy, the run gets its three automatic fixes again.
_HEALTHY_RESET_SECONDS = 600.0

_BOOT = ("import os, runpy, sys; cwd = os.getcwd(); "
         "sys.path[:] = [p for p in sys.path if p not in ('', '.', cwd, os.path.abspath(cwd))]; "
         "runpy.run_module('pulse.pulse_headless', run_name='__main__', alter_sys=True)")


def _home() -> Path:
    from . import pulse_supabase as cloud
    return Path(cloud.CACHE_PATH).parent / "headless"


# ---------------------------------------------------------------------------------- starting

def _spawn(args: List[str], cwd: str, log_path: str) -> int:
    """The supervisor, detached: it outlives the terminal and is never sent its signals."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "ab", buffering=0)
    try:
        detach: Dict[str, Any] = {"start_new_session": True}
        if os.name == "nt":
            detach = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                      | getattr(subprocess, "DETACHED_PROCESS", 0)}
        process = subprocess.Popen([sys.executable, "-c", _BOOT, *args], cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ), **detach)
    finally:
        log.close()
    return process.pid


def _log_path(script: str) -> str:
    return str(_home() / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.path.basename(script)}.log")


def start(script: str, script_args: List[str], cwd: Optional[str] = None) -> int:
    """`pulse run --headless script args`: launch the supervisor in the background and return."""
    cwd = os.path.abspath(cwd or os.getcwd())
    path = os.path.abspath(os.path.join(cwd, os.path.expanduser(script)))
    if not os.path.isfile(path):
        print(f"pulse run --headless: no such script: {path}")
        return 1
    log_path = _log_path(path)
    pid = _spawn(["supervise", "--log", log_path, "--", path, *script_args], cwd, log_path)
    print(f"[Pulse] Debugging {os.path.basename(path)} in the background (supervisor pid {pid}).")
    print(f"[Pulse] What it does: {log_path}")
    print(f"[Pulse] Online: {DASHBOARD_URL} (when this machine is signed in) · in the app: `pulse`, then /monitor")
    print("[Pulse] `pulse headless` lists what is being debugged; `pulse headless stop` stops it (the run goes on).")
    return 0


def attach_in_background(monitor: Any) -> Optional[int]:
    """auto_track(mode="headless"): this process trains; a detached supervisor watches its stream."""
    script = getattr(monitor, "script_path", None) or "run"
    log_path = _log_path(script)
    try:
        pid = _spawn(["supervise", "--log", log_path, "--attach", monitor.directory], os.getcwd(), log_path)
    except OSError as exc:
        print(f"[Pulse] Could not start the background debugger ({exc}); the run is still monitored.")
        return None
    print(f"[Pulse] Debugging this run in the background (pid {pid}); log: {log_path} · online: {DASHBOARD_URL}")
    return pid


# ---------------------------------------------------------------------------------- the supervisor

class HeadlessApp(pulse_app.App):
    """The app with no screen and nobody to ask: what it shows goes to the log."""

    def __init__(self, cli: Any, root: str, log: Any) -> None:
        super().__init__(cli, root)
        self._log = log
        self._log_lock = threading.Lock()
        self.audits = True
        self.mouse = False

    # -------------------------------------------------------------- the log

    def write(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._log_lock:
            for line in str(text).rstrip("\n").split("\n"):
                line = _CONTROL_RE.sub("", line)
                if line.strip():
                    self._log.write(f"{stamp}  {line}\n")
            self._log.flush()

    def feed(self, text: str) -> None:
        if self._hushed.get(threading.get_ident()) is None and str(text).strip():
            self.write(str(text).replace("\r", "\n"))
        super().feed(text)

    def _add(self, entry: tui.Entry) -> None:
        self.write(_describe(entry))
        super()._add(entry)

    def edit_status(self, text: str, detail: Optional[str] = None, final: bool = False) -> None:
        self.write(f"  change: {text}" + (f" ({detail})" if detail else ""))
        super().edit_status(text, detail=detail, final=final)

    def _start_crash_turn(self, script: str, what: str) -> None:
        self.write(f"== the agent is looking at a crash of {script}" + (f": {what}" if what else ""))
        super()._start_crash_turn(script, what)

    def _start_problem_turn(self, findings: List[Any]) -> None:
        self.write("== the agent is looking at: " + "; ".join(str(getattr(f, "message", f)) for f in findings[:3]))
        super()._start_problem_turn(findings)

    # -------------------------------------------------------------- nobody to ask

    def _ask(self, question: Any) -> Any:
        # "Apply this change?" when the reviewer could not answer, a y/N for a command...: with
        # nobody here the answer is no -- the pipelines read EOF as "no terminal to confirm on"
        self.write(f"  (asked {question.label!r} with nobody to answer: no)")
        raise EOFError

    def _agent_stop_run(self) -> str:
        # The agent stopping a run that only burns compute is part of debugging it unattended.
        if self.console is None:
            return "No run is open. Nothing was stopped."
        from . import pulse_stream as stream
        name = os.path.basename((self.session or {}).get("script") or "the run")
        self.console.send_control(stream.CONTROL_STOP, reason="stopped by the agent (headless)")
        self.write(f"== the agent stopped {name}")
        return f"Asked {name} to stop."


def _describe(entry: tui.Entry) -> str:
    kind = entry.kind
    if kind == "user":
        return f"▶ {entry.text}"
    if kind == "say":
        return f"agent: {entry.text}"
    if kind == "thinking":
        return f"  (thought: {len(entry.text.split())} words)"
    if kind == "tool":
        calls = " · ".join(tui.describe_call(c) for c in entry.calls)
        output = entry.output.strip()
        if len(output) > 3000:
            output = output[:3000] + f"\n... ({len(entry.output) - 3000:,} more characters)"
        return f"  {calls}" + (("\n    " + output.replace("\n", "\n    ")) if output else "")
    if kind == "edit":
        return f"  change to {', '.join(c.split(':', 1)[-1].strip() for c in entry.calls)}:\n    " + \
            entry.output.strip().replace("\n", "\n    ")
    if kind == "finding":
        return f"⚠ {entry.severity.upper()} {entry.text}"
    if kind == "error":
        return f"✕ {entry.text}"
    return entry.text


def _sign_in(cli: Any, app: HeadlessApp) -> None:
    """The cached Pulse Cloud login, without a question: the run then has a dashboard row."""
    from . import pulse_settings as settings
    from . import pulse_supabase as cloud
    try:
        if not cloud.load_cached_credentials():
            app.write("Not signed in to Pulse Cloud on this machine: the run is debugged and logged here only "
                      "(`pulse` once signs in).")
            return
        with app.hush():
            cli._auth_flow()
        workspace, project = settings.remembered_id("workspace"), settings.remembered_id("project")
        if workspace:
            cli.team_id = workspace
            cli.project_id = project or None
        if cli.user_id:
            app.write(f"Signed in to Pulse Cloud: the run goes on the dashboard ({DASHBOARD_URL}).")
    except Exception as exc:
        app.write(f"Pulse Cloud sign-in failed ({type(exc).__name__}: {exc}); logging here only.")


def _register(log_path: str, script: str) -> Path:
    path = _home() / f"{os.getpid()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "script": script, "log": log_path, "started": time.time()}))
    return path


def supervise(argv: List[str]) -> int:
    """The background process itself (started by start() / attach_in_background())."""
    log_path, attach, script_argv = None, None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--log":
            log_path, i = argv[i + 1], i + 2
        elif argv[i] == "--attach":
            attach, i = argv[i + 1], i + 2
        elif argv[i] == "--":
            script_argv, i = argv[i + 1:], len(argv)
        else:
            i += 1
    log = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else sys.stdout
    cwd = os.getcwd()
    cli = pulse_app._new_cli(cwd, [script_argv[0]] if script_argv else [], chdir=False)
    cli.review = True               # every change goes through the reviewer (or is declined)
    cli.non_interactive = True      # nobody to ask: what needs a person is declined
    app = HeadlessApp(cli, cwd, log)
    script = script_argv[0] if script_argv else attach or "?"
    registration = _register(log_path or "", script)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_a: stopping.set())
    app.write(f"Pulse headless debugger started for {script} (pid {os.getpid()}).")
    pulse_app._pick_agent_quietly(cli, os.environ.get("PULSE_MODEL", "").strip())
    if cli.agent_provider and cli.agent_key:
        app.write(f"Agent: {cli.agent_provider}. It acts on crashes, serious findings and problems the "
                  "scheduled audit finds; commands and changes are approved by the reviewer model, or declined.")
    else:
        app.write("No agent is set up (`pulse` sets one, or /config agent): the run is watched and checked, "
                  "but nothing will be fixed.")
    _sign_in(cli, app)
    app.install()
    try:
        if attach:
            from . import pulse_console as con
            session = con._describe_session(attach)
            if session is None:
                app.write(f"No Pulse run at {attach}.")
                return 1
            app._open_run(session)
        else:
            app._launch(" ".join(shlex.quote(a) for a in script_argv))
        if app.console is None:
            app.write("The run did not start; see above.")
            return 1
        return _watch(app, stopping)
    finally:
        try:
            app._close_all()
        except Exception:
            pass
        app.uninstall()
        try:
            registration.unlink()
        except OSError:
            pass
        app.write("Pulse headless debugger stopped.")


def _watch(app: HeadlessApp, stopping: threading.Event) -> int:
    idle_since: Optional[float] = None
    quiet_since = time.monotonic()
    closing_audit_done: Optional[str] = None
    while not stopping.is_set():
        with app.lock:
            app._refresh_side(force=True)
        busy = app._job is not None or app.view.busy or bool(app._crash_pending or app._problem_pending)
        live = app._status in ("live", "stalled")
        now = time.monotonic()
        if busy or not live:
            quiet_since = now
        elif app._auto_crash_turns and now - quiet_since > _HEALTHY_RESET_SECONDS:
            app._auto_crash_turns = 0        # healthy for a while: automatic fixes are allowed again
            app.write("The run has been healthy for 10 minutes: automatic fixes are allowed again.")
        if live or busy:
            idle_since = None
        else:
            session_id = (app.session or {}).get("session_id")
            if closing_audit_done != session_id and app.console is not None:
                # the run is over and nothing is pending: one closing look, then stop
                closing_audit_done = session_id
                brain = app.console.brain
                if brain.agent is not None:
                    app.write(f"{os.path.basename((app.session or {}).get('script') or 'The run')} is over "
                              f"({app._status}); a closing audit.")
                    app.console._report_audit(brain.audit())
                idle_since = time.monotonic()
                continue
            if idle_since is None:
                idle_since = now
            elif now - idle_since > _GRACE_SECONDS:
                app.write(f"The run is over ({app._status}) and nothing is left to do.")
                return 0
        stopping.wait(1.0)
    app.write("Stopped by request (the training, if still going, goes on).")
    return 0


# ---------------------------------------------------------------------------------- `pulse headless`

def _running() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(_home().glob("*.json")):
        try:
            info = json.loads(path.read_text())
            os.kill(int(info["pid"]), 0)
        except (OSError, ValueError, KeyError):
            try:
                path.unlink()
            except OSError:
                pass
            continue
        out.append(info)
    return out


def main(argv: List[str]) -> int:
    """`pulse headless` (list) / `pulse headless stop [pid|script]`."""
    running = _running()
    if argv and argv[0] == "stop":
        wanted = argv[1] if len(argv) > 1 else None
        targets = [r for r in running if wanted is None or str(r["pid"]) == wanted
                   or os.path.basename(r["script"]) == os.path.basename(wanted)]
        if not targets:
            print("Nothing matching is being debugged in the background.")
            return 1
        if wanted is None and len(targets) > 1:
            print("Several are running; say which: pulse headless stop <pid|script>")
            return 1
        for target in targets:
            os.kill(int(target["pid"]), signal.SIGTERM)
            print(f"Stopped debugging {os.path.basename(target['script'])} (pid {target['pid']}); the run goes on.")
        return 0
    if not running:
        print("Nothing is being debugged in the background. `pulse run --headless train.py` starts one.")
        return 0
    for r in running:
        minutes = int((time.time() - r.get("started", time.time())) / 60)
        print(f"{r['pid']:>7}  {os.path.basename(r['script'])}  for {minutes} min  log: {r['log']}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["supervise"]:
        sys.exit(supervise(sys.argv[2:]))
    sys.exit(main(sys.argv[1:]))
