"""
Pulse as a terminal you sit in.

Everything else in Pulse starts from the training script: you edit it, or you launch it
through `pulse run`. That is the wrong way round when the run is already going -- on a
cluster, in a tmux pane you have lost, or started by somebody else an hour ago. This is
the other direction: type `pulse`, and it finds the runs on the machine, attaches to the
one you meant, and gives you a prompt.

    $ pulse
    Pulse - 1 run on this machine

      * 20260918-2210  train.py        step 4,120   loss 0.0431   live
        20260918-1804  finetune.py     step 9,900   loss 0.0022   finished 2h ago

    Attached to train.py (~/experiments)
    2 findings . next audit in 12 min

    pulse >

The prompt is where the brain lives: detection keeps running in the background while
you type, findings print as they happen, and anything that is not a /command goes to the
agent with the whole run as evidence. Because a session records the script it came from,
the console anchors itself to that script's directory, so a fix edits the file the run is
actually executing rather than whatever happens to be under your shell's cwd.

Nothing here replaces the old ways. `auto_track()` still works, `pulse run script.py`
still works, and `python -m pulse.brain <dir>` still works for a non-interactive watcher.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import pulse_detect as detect
from . import pulse_stream as stream

LIVE_WINDOW_SECONDS = 30.0        # no new frames for this long and a run is not "live"


# ---------------------------------------------------------------------------------------
# Terminal helpers. No curses: a Pulse console has to survive SSH, Colab and a dumb pipe.
# ---------------------------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(text: str) -> str:
    return _paint(text, "2")


def bold(text: str) -> str:
    return _paint(text, "1")


def red(text: str) -> str:
    return _paint(text, "91")


def yellow(text: str) -> str:
    return _paint(text, "93")


def green(text: str) -> str:
    return _paint(text, "92")


def severity_colour(severity: str) -> Callable[[str], str]:
    return {detect.CRITICAL: red, detect.WARNING: yellow}.get(severity, dim)


def ago(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "unknown"
    seconds = max(0.0, time.time() - float(timestamp))
    for limit, unit, size in ((60, "s", 1), (3600, "m", 60), (86400, "h", 3600)):
        if seconds < limit:
            return f"{int(seconds / size)}{unit} ago"
    return f"{int(seconds / 86400)}d ago"


def compact(value: Optional[float]) -> str:
    if value is None:
        return "-"
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e6):
        return f"{value:.3e}"
    return f"{value:,.4g}"


# ---------------------------------------------------------------------------------------
# Finding the runs
# ---------------------------------------------------------------------------------------

def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)        # signal 0: existence check, changes nothing
        return True
    except (OSError, ValueError, TypeError):
        return False


def _scan_for_spools(root: str, max_depth: int = 4) -> List[str]:
    """Session directories under `root`, without walking a whole checkout.

    Depth-limited and skips the directories that make a naive walk slow (.git,
    node_modules, virtualenvs), because this runs before the prompt appears.
    """
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache",
            "site-packages", ".tox", "build", "dist", ".pulse_history"}
    found: List[str] = []
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for current, dirnames, _files in os.walk(root):
        if current.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".pulse_stream")]
        candidate = os.path.join(current, ".pulse_stream")
        if os.path.isdir(candidate):
            for name in sorted(os.listdir(candidate)):
                session = os.path.join(candidate, name)
                if os.path.isfile(os.path.join(session, "events.jsonl")):
                    found.append(session)
    return found


def discover(extra_roots: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Every run this machine knows about, newest first.

    Two sources, because neither is complete on its own: the registry finds runs started
    anywhere, and a scan of the working directory finds runs whose registry entry never
    got written (a read-only home, a different user, an older Pulse).
    """
    seen: Dict[str, Dict[str, Any]] = {}

    for entry in stream.registered_sessions():
        info = _describe_session(entry["directory"])
        if info:
            info.update({k: v for k, v in entry.items() if k not in info or info[k] is None})
            seen[info["session_id"]] = info

    for root in [os.getcwd()] + list(extra_roots or []):
        for directory in _scan_for_spools(root):
            info = _describe_session(directory)
            if info and info["session_id"] not in seen:
                seen[info["session_id"]] = info

    # Ordered by when each run STARTED, not by which wrote most recently. Two live runs
    # take turns being the most recent writer, so a list sorted by activity reorders
    # between the moment you read a number and the moment you type it, and `pulse watch 2`
    # attaches to whichever one happened to flush last. Start time does not move.
    return sorted(seen.values(),
                  key=lambda s: (s.get("started") or 0, s.get("session_id") or ""),
                  reverse=True)


def _describe_session(directory: str) -> Optional[Dict[str, Any]]:
    """What can be known about a run without reading its whole stream."""
    reader = stream.StreamReader(directory)
    session = reader.session()
    state = reader.state()
    events_path = os.path.join(directory, "events.jsonl")
    try:
        last_seen = os.path.getmtime(events_path)
    except OSError:
        return None
    session_id = session.get("session_id") or os.path.basename(directory)
    pid = session.get("pid")
    alive = _pid_alive(pid)
    fresh = (time.time() - last_seen) < LIVE_WINDOW_SECONDS
    if state.get("finished"):
        status = "finished"
    elif state.get("crashed"):
        status = "crashed"
    elif alive and fresh:
        status = "live"
    elif alive:
        status = "stalled"          # the process is there; the stream stopped moving
    else:
        status = "ended"
    scalars = state.get("scalars") or {}
    loss = next((v for k, v in scalars.items() if detect.looks_like_loss(k)), None)
    return {
        "session_id": session_id,
        "directory": directory,
        "script": session.get("script"),
        "pid": pid,
        "started": session.get("started"),
        "last_seen": last_seen,
        "status": status,
        "step": state.get("step"),
        "loss": loss,
        "scalars": scalars,
    }


def unmonitored_python_processes() -> List[Dict[str, Any]]:
    """Python processes running a script that Pulse is not watching.

    Pulse cannot attach to a process that was not started with a monitor -- there is no
    supported way to inject one into a running interpreter -- so these are listed to be
    told about, with the command to restart them under Pulse, not to be attached to.
    """
    if not os.path.isdir("/proc"):
        return []                   # Linux only; elsewhere the registry is all we have
    monitored = {int(s["pid"]) for s in discover() if s.get("pid")}
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid() or pid in monitored:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                parts = [p.decode("utf-8", "replace") for p in handle.read().split(b"\0") if p]
            started = os.path.getmtime(f"/proc/{pid}")
        except OSError:
            continue
        if not parts or "python" not in os.path.basename(parts[0]).lower():
            continue
        script = next((p for p in parts[1:] if p.endswith(".py")), None)
        if not script or "pulse" in script:
            continue
        if time.time() - started < 5:
            continue                # too new to judge; probably a helper
        out.append({"pid": pid, "script": script, "cmdline": " ".join(parts)[:120],
                    "started": started})
    return sorted(out, key=lambda p: p["started"], reverse=True)


# ---------------------------------------------------------------------------------------
# The console
# ---------------------------------------------------------------------------------------

HELP = """\
  /status            where the run is now
  /findings          what the detectors currently believe, worst first
  /curve <name>      one value's history, as a sparkline
  /vars              every value being tracked
  /audit             run the full wake-up audit now (uses the agent)
  /code              show the training script
  /sessions          list runs on this machine again
  /attach <n|id>     watch a different run
  /cd                print the directory this run's file lives in
  /interval <sec>    ask the monitor to sample faster or slower
  /stop              ask the training process to stop
  /quiet, /loud      stop or resume printing findings as they happen
  /help, /quit

  Anything else is a question for the agent about this run.
"""

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 48) -> str:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    if not numbers:
        return ""
    if len(numbers) > width:
        size = len(numbers) / float(width)
        numbers = [numbers[min(int(i * size), len(numbers) - 1)] for i in range(width)]
    low, high = min(numbers), max(numbers)
    if high - low < 1e-12:
        return SPARK[0] * len(numbers)
    span = high - low
    return "".join(
        SPARK[min(len(SPARK) - 1, int((value - low) / span * (len(SPARK) - 1)))]
        for value in numbers)


class Console:
    """The interactive session: a brain on a background thread and a prompt on this one."""

    def __init__(self, session: Dict[str, Any], agent: Optional[Callable[[str], str]] = None,
                 sensitivity: float = 0.3) -> None:
        from .pulse_brain import Brain

        self.session = session
        self.brain = Brain(session["directory"], agent=agent, sensitivity=sensitivity)
        self.agent = agent
        self.quiet = False
        self.running = True
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.brain.on_finding = self._announce

    # -------------------------------------------------------------- background

    def _announce(self, findings: List[detect.Finding]) -> None:
        if self.quiet:
            return
        for finding in findings:
            paint = severity_colour(finding.severity)
            self._print(f"\n{paint('  ' + finding.severity.upper())} {finding.message}")

    def _print(self, text: str) -> None:
        """Print without mangling the line the person is typing."""
        with self._lock:
            sys.stdout.write("\r\033[K" if _COLOR else "\r")
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def _pump(self) -> None:
        while self.running:
            try:
                self.brain.poll_once()
            except Exception as exc:
                self._print(dim(f"  (monitor read failed: {type(exc).__name__}: {exc})"))
            time.sleep(0.5)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, name="pulse-console-brain", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False

    # -------------------------------------------------------------- views

    def status_line(self) -> str:
        brain = self.brain
        findings = brain.engine.current()
        bits = [f"step {brain.step:,}" if brain.step else "no steps yet"]
        for name, history in brain.histories.items():
            if detect.looks_like_loss(name) and history:
                bits.append(f"{name} {compact(history[-1])}")
                break
        if findings:
            worst = findings[0]
            bits.append(severity_colour(worst.severity)(f"{len(findings)} finding(s)"))
        else:
            bits.append(green("healthy"))
        if brain.agent is not None:
            bits.append(f"next audit in {int(brain.schedule.seconds_remaining() / 60)} min")
        if brain.finished:
            bits.append(dim("run finished"))
        return " · ".join(bits)

    @property
    def workdir(self) -> str:
        """Where the run's code lives -- what a fix has to edit -- not where its spool is."""
        script = self.session.get("script")
        if script and os.path.dirname(script):
            return os.path.dirname(script)
        return self.session.get("cwd") or self.session["directory"]

    def show_status(self) -> None:
        brain = self.brain
        print(f"\n  {bold(os.path.basename(self.session.get('script') or '?'))}   "
              f"{dim(self.workdir)}")
        print(f"  {self.status_line()}")
        if brain.reader.gaps:
            print(dim(f"  {brain.reader.gaps} sample(s) dropped under load"))
        tracked = {k: v for k, v in brain.histories.items() if len(v) > 1}
        if tracked:
            print()
            for name, history in list(tracked.items())[:8]:
                print(f"  {name:<18} {compact(history[-1]):>12}  {dim(sparkline(history, 32))}")
        print()

    def show_findings(self) -> None:
        findings = self.brain.engine.current()
        if not findings:
            print("\n  " + green("Nothing wrong that the checks can see.") + "\n")
            return
        print()
        for finding in findings:
            paint = severity_colour(finding.severity)
            label = f"{finding.severity.upper():<8}"
            print(f"  {paint(label)} {finding.message}")
            print(dim(f"           {finding.check} on {finding.variable}, seen {finding.count}x, "
                      f"confidence {finding.confidence}"))
        print()

    def show_curve(self, name: str) -> None:
        history = self.brain.histories.get(name)
        if not history:
            matches = [k for k in self.brain.histories if name.lower() in k.lower()]
            if len(matches) == 1:
                name, history = matches[0], self.brain.histories[matches[0]]
            else:
                print(f"\n  No value called {name!r}. Try /vars.\n")
                return
        print(f"\n  {bold(name)}  {len(history)} readings")
        print(f"  first {compact(history[0])}   last {compact(history[-1])}   "
              f"min {compact(min(history))}   max {compact(max(history))}")
        print(f"  {sparkline(history, min(shutil.get_terminal_size((80, 24)).columns - 6, 70))}\n")

    def show_vars(self) -> None:
        brain = self.brain
        print()
        for name, history in sorted(brain.histories.items()):
            print(f"  {name:<22} {len(history):>6} readings   last {compact(history[-1])}")
        for name, meta in sorted(brain.tensors.items()):
            shape = "x".join(str(d) for d in (meta.get("shape") or []))
            print(dim(f"  {name:<22} tensor {shape} {meta.get('dtype', '')} "
                      f"on {meta.get('device', 'cpu')}"))
        print()

    def show_code(self) -> None:
        path = self.session.get("script")
        if not path or not os.path.isfile(path):
            print(f"\n  The script is not readable from here: {path}\n")
            return
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
        print()
        for number, line in enumerate(lines[:200], 1):
            print(f"  {dim(f'{number:>4}')} {line}")
        if len(lines) > 200:
            print(dim(f"  ... {len(lines) - 200} more lines"))
        print()

    # -------------------------------------------------------------- actions

    def ask(self, question: str) -> None:
        if self.agent is None:
            print("\n  No model configured. Start with --model, or set PULSE_MODEL.\n")
            return
        pack = self.brain.evidence(include_code=True)
        prompt = (
            "You are Pulse, watching a training run for someone who is sitting at a "
            "terminal looking at it with you. Answer their question from the evidence "
            "below. Be specific about the numbers. If the evidence does not settle it, "
            "say what you would need.\n\n"
            f"{self.brain.render_evidence(pack)}\n\nTheir question: {question}\n")
        print(dim("\n  thinking...\n"))
        try:
            answer = self.agent(prompt)
        except Exception as exc:
            print(red(f"  The model could not be reached: {type(exc).__name__}: {exc}\n"))
            return
        print("  " + (answer or "(no answer)").replace("\n", "\n  ") + "\n")

    def audit(self) -> None:
        if self.agent is None:
            print("\n  No model configured, so there is nothing to audit with.\n")
            return
        print(dim("\n  auditing the whole run...\n"))
        record = self.brain.audit()
        if record.get("status") in ("error", "skipped"):
            print(red(f"  {record.get('error') or record.get('reason')}\n"))
            return
        print("  " + (record.get("text") or "").replace("\n", "\n  "))
        print(dim(f"\n  next audit in {int(self.brain.schedule.seconds_remaining() / 60)} min "
                  f"({self.brain.schedule.reason})\n"))

    def send_control(self, action: str, **fields: Any) -> None:
        self.brain.reader.send_control(action, **fields)
        print(dim(f"\n  asked the run to {action}\n"))


def render_session_list(sessions: List[Dict[str, Any]], attached: Optional[str] = None) -> None:
    status_paint = {"live": green, "stalled": yellow, "crashed": red}
    for index, session in enumerate(sessions, 1):
        marker = "*" if session["session_id"] == attached else " "
        paint = status_paint.get(session["status"], dim)
        script = os.path.basename(session.get("script") or "?")
        step = f"step {session['step']:,}" if session.get("step") else "no steps"
        loss = f"loss {compact(session.get('loss'))}" if session.get("loss") is not None else ""
        when = session["status"] if session["status"] in ("live", "stalled") else \
            f"{session['status']} {ago(session.get('last_seen'))}"
        print(f"  {marker} {index}. {session['session_id']:<22} {script:<22} "
              f"{step:<14} {loss:<18} {paint(when)}")


def pick_session(sessions: List[Dict[str, Any]], wanted: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve what the user asked for: an index, a session id, a script name, or nothing."""
    if wanted:
        if wanted.isdigit() and 1 <= int(wanted) <= len(sessions):
            return sessions[int(wanted) - 1]
        for session in sessions:
            if session["session_id"] == wanted:
                return session
        matches = [s for s in sessions
                   if wanted in (s.get("session_id") or "")
                   or wanted in os.path.basename(s.get("script") or "")]
        if len(matches) == 1:
            return matches[0]
        return None
    live = [s for s in sessions if s["status"] in ("live", "stalled")]
    if len(live) == 1:
        return live[0]              # the obvious one: attach without asking
    if not live and len(sessions) == 1:
        return sessions[0]
    return None


def run_console(session: Dict[str, Any], sessions: List[Dict[str, Any]],
                agent: Optional[Callable[[str], str]] = None,
                sensitivity: float = 0.3) -> int:
    """The prompt. Returns an exit code."""
    try:
        import readline  # noqa: F401   line editing and history, when the platform has it
    except ImportError:
        pass

    console = Console(session, agent=agent, sensitivity=sensitivity)
    console.start()
    console.brain.poll_once()        # so the first status line has something in it

    script = session.get("script") or "?"
    directory = console.workdir
    print(f"\nAttached to {bold(os.path.basename(script))}  {dim(directory)}")
    print(f"{console.status_line()}")
    print(dim("/help for commands, or just ask a question about the run.\n"))

    while True:
        try:
            line = input(f"{bold('pulse')} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if not line.startswith("/"):
            console.ask(line)
            continue

        command, _, argument = line[1:].partition(" ")
        command, argument = command.lower(), argument.strip()

        if command in ("quit", "exit", "q"):
            break
        elif command == "help":
            print("\n" + HELP)
        elif command == "status":
            console.show_status()
        elif command == "findings":
            console.show_findings()
        elif command == "curve":
            console.show_curve(argument or "loss")
        elif command == "vars":
            console.show_vars()
        elif command == "code":
            console.show_code()
        elif command == "audit":
            console.audit()
        elif command == "cd":
            print(f"\n  {directory}\n")
        elif command == "sessions":
            sessions = discover()
            print()
            render_session_list(sessions, attached=session["session_id"])
            print()
        elif command == "attach":
            sessions = discover()
            chosen = pick_session(sessions, argument)
            if chosen is None:
                print("\n  Which one? Give a number or an id:\n")
                render_session_list(sessions, attached=session["session_id"])
                print()
                continue
            console.stop()
            return run_console(chosen, sessions, agent=agent, sensitivity=sensitivity)
        elif command == "interval":
            try:
                console.send_control(stream.CONTROL_SET_INTERVAL, interval=float(argument))
            except (TypeError, ValueError):
                print("\n  /interval takes a number of seconds, e.g. /interval 0.5\n")
        elif command == "stop":
            console.send_control(stream.CONTROL_STOP, reason="asked from the Pulse console")
        elif command == "quiet":
            console.quiet = True
            print(dim("\n  findings will not interrupt you now\n"))
        elif command == "loud":
            console.quiet = False
            print(dim("\n  findings will print as they happen\n"))
        else:
            print(f"\n  No such command: /{command}. /help lists them.\n")

    console.stop()
    print(dim("bye"))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """`pulse`, `pulse watch [what]`, `pulse sessions`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv and not argv[0].startswith("-") else ""
    if command in ("watch", "attach", "console", "sessions"):
        argv = argv[1:]
    wanted = next((a for a in argv if not a.startswith("-")), None)

    model = ""
    for flag in ("--model", "-m"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv):
                model = argv[index + 1]
                if wanted == model:
                    wanted = None
    model = model or os.environ.get("PULSE_MODEL", "")

    sessions = discover()
    if command == "sessions":
        if not sessions:
            print("No Pulse runs on this machine yet.")
            return 1
        print()
        render_session_list(sessions)
        print()
        return 0

    live = [s for s in sessions if s["status"] in ("live", "stalled")]
    print(f"\n{bold('Pulse')} - {len(sessions)} run(s) on this machine"
          f"{f', {len(live)} live' if live else ''}\n")
    if sessions:
        render_session_list(sessions)

    if not sessions:
        print("  Nothing to watch yet. Start a run with either:\n")
        print("    pulse run train.py            (no changes to your script)")
        print("    auto_track(mode=\"stream\")     (from inside it)\n")
        idle = unmonitored_python_processes()
        if idle:
            print("  These Python processes are running but not monitored, and Pulse")
            print("  cannot attach to a process that did not start with it:\n")
            for process in idle[:5]:
                print(f"    pid {process['pid']:<8} {os.path.basename(process['script'])}"
                      f"   {dim('started ' + ago(process['started']))}")
            print(dim("\n  Restart one under `pulse run` to watch it.\n"))
        return 1

    session = pick_session(sessions, wanted)
    if session is None:
        if wanted:
            print(f"\n  Nothing matches {wanted!r}.\n")
        else:
            print("\n  Which one? `pulse watch <number>`\n")
        return 1

    agent = None
    if model:
        from .pulse_brain import build_litellm_agent
        agent = build_litellm_agent(model)
    return run_console(session, sessions, agent=agent)


if __name__ == "__main__":
    raise SystemExit(main())
