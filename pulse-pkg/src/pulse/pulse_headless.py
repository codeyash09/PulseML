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
there is no reviewer, or it cannot answer, the answer is no. There is no limit on how many
times it fixes and restarts.

Where the run goes online is settled before it leaves the terminal (choose_destination):
signed in to Pulse Cloud, a workspace and a project -- asked once, remembered in the settings,
or given on the command line (--to WORKSPACE/PROJECT, both together). Everything the supervisor does goes to
a log file (~/.pulse/headless/) and to the run's row on the dashboard. `pulse`, then /monitor,
opens the run in the app at any time.
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
# How long a question waits for an answer on the dashboard before it is declined.
_ANSWER_SECONDS = float(os.environ.get("PULSE_HEADLESS_ANSWER_SECONDS", "900"))
# terminal control sequences (colours, "clear to end of line"): not for a log file
_CONTROL_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")

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


def split_destination(value: str):
    """"Lab/GPT small" -> ("Lab", "GPT small"). Both halves are needed: a workspace without a
    project is a setup left half done."""
    workspace, sep, project = str(value or "").partition("/")
    if not sep or not workspace.strip() or not project.strip():
        raise ValueError(f"--to takes WORKSPACE/PROJECT, both of them (e.g. --to \"Lab/GPT small\"), not {value!r}")
    return workspace.strip(), project.strip()


class DestinationError(Exception):
    """A --workspace / --project that matches nothing (the message lists what there is)."""


def _match(items: List[Dict[str, Any]], wanted: str, keys: List[str], describe) -> Dict[str, Any]:
    wanted_l = wanted.strip().lower()
    exact = [i for i in items if any(str(i.get(k) or "").strip().lower() == wanted_l for k in keys)]
    if len(exact) == 1:
        return exact[0]
    partial = [i for i in items if wanted_l in describe(i).lower()]
    if len(partial) == 1:
        return partial[0]
    names = "; ".join(describe(i) for i in items) or "none"
    raise DestinationError(("several match" if (exact or partial) else "nothing matches")
                           + f" {wanted!r} -- there are: {names}")


def choose_destination(workspace: Optional[str] = None, project: Optional[str] = None,
                       interactive: bool = True) -> Dict[str, Any]:
    """Where the run goes online, settled while there is still a terminal: signed in to Pulse
    Cloud (asked, if not), the workspace and the project. Given on the command line, they are
    used and remembered; otherwise the remembered ones are used; otherwise the pickers ask --
    once, and the answer is remembered (pulse config workspace/project changes it). Returns
    {"team_id", "project_id", "label"}, or {} for "this machine only"."""
    from . import pulse_settings as settings
    from . import pulse_supabase as cloud
    cli = pulse_app._new_cli(os.getcwd(), [], chdir=False)
    cli.non_interactive = not interactive
    if not cloud.load_cached_credentials():
        if not interactive:
            return {}
        from . import pulse_ui as ui
        try:
            wanted = ui.confirm("Sign in to Pulse Cloud, so this run shows up on your dashboard?", default=True)
        except ui.Unavailable:
            wanted = input("Sign in to Pulse Cloud, so this run shows up on your dashboard? (Y/n) > ").strip().lower() \
                in ("", "y", "yes")
        if not wanted:
            return {}
    try:
        cli._auth_flow()
    except cloud.SupabaseError as exc:
        print(f"[Pulse] Could not sign in ({exc}); the run will be logged on this machine only.")
        return {}
    if not cli.user_id:
        return {}
    try:
        return _destination(cli, workspace, project, interactive)
    except cloud.SupabaseError as exc:
        # a cloud that refuses or is unreachable must not stop the run from starting
        print(f"[Pulse] Pulse Cloud did not answer ({exc}); the run will be logged on this machine only.")
        return {}


def _destination(cli: Any, workspace: Optional[str], project: Optional[str], interactive: bool) -> Dict[str, Any]:
    from . import pulse_settings as settings
    from . import pulse_supabase as cloud
    describe_team = lambda t: cli._describe_workspace(t, cli.user_id)   # noqa: E731
    if workspace:
        team = _match(cloud.find_teams_for_user(cli.user_id), workspace, ["team_id", "join_code", "name"],
                      describe_team)
        settings.set("workspace", {"id": team["team_id"], "name": describe_team(team)})
        if not project:
            settings.unset("project")       # the old project belonged to the old workspace
    if project:
        team_id = settings.remembered_id("workspace")
        if not team_id:
            raise DestinationError("a project needs its workspace: --to WORKSPACE/PROJECT")
        team = next((t for t in cloud.find_teams_for_user(cli.user_id) if t["team_id"] == team_id), None)
        if team is None:
            raise DestinationError("the remembered workspace is gone: --to WORKSPACE/PROJECT")
        ref = {"team_id": team["team_id"], "admin_ids": list(team.get("admin_ids") or [])}
        found = _match(cloud.find_projects_for_team(ref, cli.user_id), project, ["project_id", "name"],
                       cli._describe_project)
        settings.set("project", {"id": found["project_id"], "name": (found.get("name") or "").strip()
                                 or found["project_id"]})
    # the remembered workspace and project are used without a question; a missing one is
    # asked for (the pickers), when there is someone to ask, and remembered
    teams = cloud.find_teams_for_user(cli.user_id)
    team = next((t for t in teams if t["team_id"] == settings.remembered_id("workspace")), None)
    if team is None:
        if not interactive:
            return {}
        cli._select_workspace_and_project()            # pickers; remembers what is chosen
        team = next((t for t in teams if t["team_id"] == cli.team_id), None) if cli.team_id else None
        if team is None:
            return {}
    ref = {"team_id": team["team_id"], "admin_ids": list(team.get("admin_ids") or [])}
    projects = cloud.find_projects_for_team(ref, cli.user_id)
    chosen = next((p for p in projects if p["project_id"] == (cli.project_id or settings.remembered_id("project"))), None)
    if chosen is None and interactive:
        cli.team_id, cli.team_join_code = team["team_id"], team.get("join_code")
        cli.team_admin_ids = list(team.get("admin_ids") or [])
        cli._project_flow(None)
        chosen = next((p for p in projects if p["project_id"] == cli.project_id), None) if cli.project_id else None
        if chosen is None and cli.project_id:          # just created: not in the list read above
            chosen = {"project_id": cli.project_id, "name": cli.project_name}
    if chosen is not None:
        settings.set("project", {"id": chosen["project_id"],
                                 "name": (chosen.get("name") or "").strip() or chosen["project_id"]})
    team_name = (team.get("name") or "").strip() or f"join code {team.get('join_code')}"
    label = f"workspace {team_name}" + (
        f", project {(chosen.get('name') or '').strip() or chosen['project_id']}" if chosen else "")
    return {"team_id": team["team_id"], "project_id": chosen["project_id"] if chosen else None, "label": label}


def start(script: str, script_args: List[str], cwd: Optional[str] = None,
          workspace: Optional[str] = None, project: Optional[str] = None) -> int:
    """`pulse run --headless script args`: settle where it goes online, launch the supervisor
    in the background, and return."""
    cwd = os.path.abspath(cwd or os.getcwd())
    path = os.path.abspath(os.path.join(cwd, os.path.expanduser(script)))
    if not os.path.isfile(path):
        print(f"pulse run --headless: no such script: {path}")
        return 1
    try:
        where = choose_destination(workspace, project, interactive=sys.stdin.isatty())
    except DestinationError as exc:
        print(f"pulse run --headless: {exc}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n[Pulse] Cancelled; nothing was started.")
        return 1
    log_path = _log_path(path)
    pid = _spawn(["supervise", "--log", log_path, *_where_args(where), "--", path, *script_args], cwd, log_path)
    _say_started(os.path.basename(path), pid, log_path, where)
    return 0


def _where_args(where: Dict[str, Any]) -> List[str]:
    out = []
    if where.get("team_id"):
        out += ["--team-id", where["team_id"]]
    if where.get("project_id"):
        out += ["--project-id", where["project_id"]]
    return out


def _say_started(name: str, pid: int, log_path: str, where: Dict[str, Any]) -> None:
    print(f"[Pulse] {name} is running, and Pulse is debugging it in the background (pid {pid}).")
    print("[Pulse] You can close this terminal. If it crashes or goes wrong, the agent fixes the code and restarts it.")
    if where:
        print(f"[Pulse] Online: {where['label']} -- {DASHBOARD_URL}")
        print("[Pulse]   (remembered for next time; --to WORKSPACE/PROJECT, or `pulse config`, changes it)")
    else:
        print("[Pulse] Not signed in to Pulse Cloud: this machine only (`pulse` once signs in).")
    print(f"[Pulse] Everything it does: {log_path}")
    print("[Pulse] Look at it: `pulse`, then /monitor · list: `pulse headless` · stop debugging: `pulse headless stop`")


def attach_in_background(monitor: Any) -> Optional[int]:
    """auto_track(mode="headless"): this process trains; a detached supervisor watches its stream."""
    script = getattr(monitor, "script_path", None) or "run"
    log_path = _log_path(script)
    try:
        wanted = os.environ.get("PULSE_TO", "").strip()
        workspace, project = split_destination(wanted) if wanted else (None, None)
        where = choose_destination(workspace, project, interactive=sys.stdin.isatty())
    except (ValueError, DestinationError, EOFError, KeyboardInterrupt) as exc:
        print(f"[Pulse] {exc or 'No workspace chosen'}: this run is logged on this machine only.")
        where = {}
    try:
        pid = _spawn(["supervise", "--log", log_path, *_where_args(where), "--attach", monitor.directory],
                     os.getcwd(), log_path)
    except OSError as exc:
        print(f"[Pulse] Could not start the background debugger ({exc}); the run is still monitored.")
        return None
    _say_started(os.path.basename(script), pid, log_path, where)
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
        self.auto_fix_limit = None          # it fixes and restarts as often as the run needs

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
        # nobody here it goes on the run's dashboard page and waits there for a while; with no
        # answer the answer is no -- the pipelines read EOF as "no terminal to confirm on"
        posted = self._post_question(question)
        if posted is not None:
            minutes = max(1, round(_ANSWER_SECONDS / 60))
            self.write(f"  (asked {question.label!r}: waiting up to {minutes} min for an answer on the dashboard)")
            if question.done.wait(_ANSWER_SECONDS):
                self.write(f"  (answered on the dashboard: {question.answer!r})")
                return question.answer
            posted.close("failed", f"Nobody answered within {minutes} min, so the answer was no.")
            self.write(f"  (no answer in {minutes} min: no)")
            raise EOFError
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


def _sign_in(cli: Any, app: HeadlessApp, team_id: Optional[str] = None, project_id: Optional[str] = None) -> None:
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
        workspace = team_id or settings.remembered_id("workspace")
        project = project_id or (settings.remembered_id("project") if not team_id else None)
        if workspace:
            cli.team_id = workspace
            cli.project_id = project or None
        if cli.user_id:
            app.write(f"Signed in to Pulse Cloud: the run goes on the dashboard ({DASHBOARD_URL}), "
                      f"workspace {cli.team_id or '(none)'}, project {cli.project_id or '(none)'}.")
    except Exception as exc:
        app.write(f"Pulse Cloud sign-in failed ({type(exc).__name__}: {exc}); logging here only.")


def _register(log_path: str, script: str) -> Path:
    path = _home() / f"{os.getpid()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "script": script, "log": log_path, "started": time.time()}))
    return path


def supervise(argv: List[str]) -> int:
    """The background process itself (started by start() / attach_in_background())."""
    log_path, attach, script_argv, team_id, project_id = None, None, [], None, None
    i = 0
    while i < len(argv):
        if argv[i] == "--log":
            log_path, i = argv[i + 1], i + 2
        elif argv[i] == "--team-id":
            team_id, i = argv[i + 1], i + 2
        elif argv[i] == "--project-id":
            project_id, i = argv[i + 1], i + 2
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
    _sign_in(cli, app, team_id, project_id)
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
        app.done = True                              # a dashboard prompt still waiting is failed, not left hanging
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
    closing_audit_done: Optional[str] = None
    while not stopping.is_set():
        with app.lock:
            app._refresh_side(force=True)
        app._start_pending_remote_command()          # prompts sent from the dashboard
        busy = app._job is not None or app.view.busy or bool(app._crash_pending or app._problem_pending
                                                             or app._remote_pending)
        live = app._status in ("live", "stalled")
        now = time.monotonic()
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

def _alive(pid: int) -> bool:
    """Is that process still there? (On Windows os.kill(pid, 0) is not a check: signal 0 is
    CTRL_C_EVENT there, so it would interrupt the process it asks about.)"""
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32                  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _running() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(_home().glob("*.json")):
        try:
            info = json.loads(path.read_text())
            if not _alive(int(info["pid"])):
                raise OSError("gone")
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
