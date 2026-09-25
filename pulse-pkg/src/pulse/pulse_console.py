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

import math
import os
import shlex
import shutil
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import pulse_detect as detect
from . import pulse_stream as stream
from . import pulse_trace as trace_engine

LIVE_WINDOW_SECONDS = 30.0        # no new frames for this long and a run is not "live"
MIN_SAMPLE_INTERVAL = 0.01
MAX_SAMPLE_INTERVAL = 60.0


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
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if value != value else ("inf" if value > 0 else "-inf")
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e6):
        return f"{value:.3e}"
    return f"{value:,.4g}"


# ---------------------------------------------------------------------------------------
# Finding the runs
# ---------------------------------------------------------------------------------------

_STILL_ACTIVE = 259                     # Windows: GetExitCodeProcess for a running process
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _pid_alive(pid: Optional[int]) -> bool:
    """Is that process still there? Asks; never touches it.

    `os.kill(pid, 0)` is the usual POSIX way to ask, and on Windows it is not a
    question at all: any signal other than CTRL_C_EVENT or CTRL_BREAK_EVENT goes to
    TerminateProcess, so the "probe" kills the process. Listing your runs must not
    stop them, so Windows gets a real query instead.
    """
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        # Not a process id. On POSIX, os.kill(-1, ...) addresses every process the
        # caller may signal and os.kill(0, ...) the whole process group, so a corrupt
        # or negative entry must be rejected before it reaches a syscall, not after.
        return False

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == _STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False            # no answer is not a reason to touch the process

    try:
        os.kill(pid, 0)             # POSIX signal 0: existence check, changes nothing
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
        try:
            names = sorted(os.listdir(candidate))
        except OSError:
            continue        # not a directory, unreadable, or deleted since os.walk saw it
        for name in names:
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
    def is_empty_and_over(info: Dict[str, Any]) -> bool:
        """A run that ended without ever reporting a number is not worth listing."""
        return (info.get("status") in ("ended", "finished", "crashed")
                and not info.get("step") and not info.get("scalars"))

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
    return sorted((s for s in seen.values() if not is_empty_and_over(s)),
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


def _is_pulse_itself(script: str) -> bool:
    """Pulse's own processes, which are not runs to offer somebody.

    Matching "pulse" anywhere in the path also excluded the user's own scripts: a
    project in ~/pulse-experiments, or a checkout of this repo, made every script in it
    invisible to `pulse <script>`. Only the file's own name and the package directory
    it sits in say whether it is Pulse.
    """
    base = os.path.basename(script)
    parent = os.path.basename(os.path.dirname(script))
    return base.startswith("pulse") or parent == "pulse"


def _boot_time() -> Optional[float]:
    """Epoch seconds at which this machine booted, from /proc/stat."""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _process_started(pid: int, boot: Optional[float]) -> Optional[float]:
    """When a process actually began, in epoch seconds.

    Not the mtime of /proc/<pid>: that is the last time the directory changed, which on
    some kernels is continually refreshed and reads as "just now" for a process that has
    been running for hours. Pulse used that to skip processes too new to judge, so on
    those kernels every run looked one second old and was skipped -- `sudo pulse
    train.py` found nothing while train.py was plainly running, intermittently, because
    it depended on when the directory was last touched.

    Field 22 of /proc/<pid>/stat is the start time in clock ticks since boot. The comm
    field before it is parenthesised and may itself contain spaces and brackets, so the
    fields are taken from after its last ')'.
    """
    if boot is None:
        return None
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            raw = handle.read()
        fields = raw[raw.rindex(")") + 2:].split()
        return boot + float(fields[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def unmonitored_python_processes() -> List[Dict[str, Any]]:
    """Python processes running a script that Pulse is not watching.

    Pulse cannot attach to a process that was not started with a monitor -- there is no
    supported way to inject one into a running interpreter -- so these are listed to be
    told about, with the command to restart them under Pulse, not to be attached to.
    """
    if not os.path.isdir("/proc"):
        return []                   # Linux only; elsewhere the registry is all we have
    monitored = {int(s["pid"]) for s in discover() if s.get("pid")}
    boot = _boot_time()
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
        except OSError:
            continue
        if not parts or "python" not in os.path.basename(parts[0]).lower():
            continue
        script = next((p for p in parts[1:] if p.endswith(".py")), None)
        if not script or _is_pulse_itself(script):
            continue
        started = _process_started(pid, boot)
        if started is None:
            # No reliable start time. Listing a run that might be a helper is a far
            # smaller problem than hiding the one the person is asking about.
            started = time.time()
        elif time.time() - started < 5:
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
  /trace <var>       the variable's whole influence graph: what feeds it, and what it feeds
  /sessions          list runs on this machine again
  /attach <n|id>     watch a different run
  /cd                print the directory this run's file lives in
  /interval <sec>    ask the monitor to sample faster or slower
  /stop              stop the training run (like Ctrl-C in its terminal)
  /quiet, /loud      stop or resume printing findings as they happen
  /help, /quit

  Anything else is a question for the agent about this run.
"""

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 48) -> str:
    """A curve as one line of block characters. Non-finite readings are dropped.

    A loss going NaN is the failure this whole tool exists to catch, and it used to
    take the console down with it: inf makes the range inf, (v-low)/inf is nan, and
    int(nan) raises, so /status and /curve died at the worst possible moment.
    """
    numbers = [float(v) for v in values
               if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
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
        self._lock = threading.Lock()          # serialises writes to the terminal
        self._brain_lock = threading.RLock()   # the pump writes what the views read
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
                with self._brain_lock:
                    result = self.brain.poll_once()
            except Exception as exc:
                self._print(dim(f"  (monitor read failed: {type(exc).__name__}: {exc})"))
            else:
                self._report_audit(result.get("audit"))
            time.sleep(0.5)

    def _report_audit(self, record: Optional[Dict[str, Any]]) -> None:
        """Show a scheduled audit. It is paid for; it should not vanish.

        poll_once runs the audit when one is due -- a real model call with the whole run
        in the prompt -- and the pump used to throw the answer away, so with --model set
        the console spent a call every fifteen minutes and showed nothing.
        """
        if not record or record.get("status") in (None, "skipped", "busy"):
            return
        if record.get("status") == "error":
            self._print(red(f"\n  audit failed: {record.get('error')}"))
            return
        text = (record.get("text") or "").strip()
        if text:
            self._print("\n" + bold("  audit") + "\n  " + text.replace("\n", "\n  "))

    def snapshot(self) -> Dict[str, Any]:
        """Copies of everything a view reads, taken while the pump cannot be writing.

        The pump adds history keys and raises findings on its own thread; the views
        iterated those same dicts, so a new variable appearing -- a val_loss at the
        first eval -- while someone typed /status raised RuntimeError: dictionary
        changed size during iteration, out of the command loop and out of the program.
        """
        with self._brain_lock:
            return {
                "step": self.brain.step,
                "histories": {k: list(v) for k, v in self.brain.histories.items()},
                "tensors": dict(self.brain.tensors),
                "findings": list(self.brain.engine.current()),
                "finished": self.brain.finished,
                "gaps": self.brain.reader.gaps,
            }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, name="pulse-console-brain", daemon=True)
        self._thread.start()

    def stop(self, join: bool = False) -> None:
        self.running = False
        if join and self._thread is not None:
            self._thread.join(timeout=2.0)

    # -------------------------------------------------------------- views

    def status_line(self) -> str:
        state = self.snapshot()
        brain = self.brain
        findings = state["findings"]
        bits = [f"step {state['step']:,}" if state["step"] else "no steps yet"]
        for name, history in state["histories"].items():
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
        if state["finished"]:
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
        state = self.snapshot()
        print(f"\n  {bold(os.path.basename(self.session.get('script') or '?'))}   "
              f"{dim(self.workdir)}")
        print(f"  {self.status_line()}")
        if state["gaps"]:
            print(dim(f"  {state['gaps']} sample(s) dropped under load"))
        tracked = {k: v for k, v in state["histories"].items() if len(v) > 1}
        if tracked:
            print()
            for name, history in list(tracked.items())[:8]:
                print(f"  {name:<18} {compact(history[-1]):>12}  {dim(sparkline(history, 32))}")
        print()

    def show_findings(self) -> None:
        findings = self.snapshot()["findings"]
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
        histories = self.snapshot()["histories"]
        history = histories.get(name)
        if not history:
            matches = [k for k in histories if name.lower() in k.lower()]
            if len(matches) == 1:
                name, history = matches[0], histories[matches[0]]
            else:
                print(f"\n  No value called {name!r}. Try /vars.\n")
                return
        finite = [v for v in history if math.isfinite(v)]
        print(f"\n  {bold(name)}  {len(history)} readings"
              + (dim(f"  ({len(history) - len(finite)} not finite)") if len(finite) != len(history) else ""))
        print(f"  first {compact(history[0])}   last {compact(history[-1])}   "
              f"min {compact(min(finite) if finite else None)}   "
              f"max {compact(max(finite) if finite else None)}")
        print(f"  {sparkline(history, min(shutil.get_terminal_size((80, 24)).columns - 6, 70))}\n")

    def show_vars(self) -> None:
        state = self.snapshot()
        print()
        for name, history in sorted(state["histories"].items()):
            print(f"  {name:<22} {len(history):>6} readings   last {compact(history[-1])}")
        for name, meta in sorted(state["tensors"].items()):
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

    def show_trace(self, arg: str) -> None:
        """/trace <var>[:<file>[:<line>]] -- the variable's whole connected influence
        path: everything that feeds it (across function/file boundaries, following
        parameters back to call sites and returns forward into callers), and everything
        it feeds in turn, from the run's own project files on disk."""
        if not arg.strip():
            print("\n  usage: /trace <variable>  (or /trace <variable>:<file>:<line>, self.<attr> works too)\n")
            return
        script = self.session.get("script")
        files = trace_engine.project_files(self.workdir, entry=script)
        print()
        print(trace_engine.trace(files, arg, color=_COLOR))
        print()

    # -------------------------------------------------------------- actions

    def ask(self, question: str) -> None:
        if self.agent is None:
            print("\n  No model configured. Start with --model, or set PULSE_MODEL.\n")
            return
        with self._brain_lock:
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
        with self._brain_lock:
            record = self.brain.audit()
        if record.get("status") == "busy":
            print(dim("  an audit is already running; its answer will print here\n"))
            return
        if record.get("status") in ("error", "skipped"):
            print(red(f"  {record.get('error') or record.get('reason')}\n"))
            return
        print("  " + (record.get("text") or "").replace("\n", "\n  "))
        print(dim(f"\n  next audit in {int(self.brain.schedule.seconds_remaining() / 60)} min "
                  f"({self.brain.schedule.reason})\n"))

    def send_control(self, action: str, **fields: Any) -> None:
        self.brain.reader.send_control(action, **fields)
        print(dim(f"\n  asked the run to {action}\n"))


def report_unmonitored(processes: List[Dict[str, Any]], limit: int = 5) -> None:
    """List runs Pulse can see but is not watching, and say what it would take.

    These used to be mentioned only when there were no Pulse runs at all, so somebody
    with one finished run and one just started outside Pulse was told about the finished
    one and never about the one they cared about.
    """
    if not processes:
        return
    from . import pulse_attach as attach

    print(dim(f"\n  {len(processes)} Python run(s) not watched by Pulse:"))
    for process in processes[:limit]:
        print(dim(f"    pid {process['pid']:<8} {os.path.basename(process['script'])}"
                  f"   started {ago(process['started'])}"))
    if len(processes) > limit:
        print(dim(f"    ... and {len(processes) - limit} more"))

    first = processes[0]["pid"]
    if not attach.pyspy_path():
        print("\n  Pulse can watch one of these from outside, by reading its memory, but")
        print("  that needs py-spy:   pip install py-spy")
        print(f"  then:                sudo pulse attach --pid {first}\n")
        return
    scope = attach.ptrace_scope()
    if attach.needs_root():
        print("\n  To watch one of these Pulse has to read another process's memory, and")
        print(f"  Linux only allows that for a parent process or root"
              f"{f' (ptrace_scope is {scope})' if scope is not None else ''}.")
        print(f"  So it needs sudo:    sudo pulse attach --pid {first}\n")
    else:
        print(f"\n  Watch one:           pulse attach --pid {first}\n")


# Where sudo looks. It replaces PATH with its own secure_path, so a `pulse` in
# ~/.local/bin -- where pip puts it -- is invisible to it.
SECURE_PATH = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")


def secure_path_pulse() -> Optional[str]:
    """The `pulse` sudo would find, if there is one."""
    for directory in SECURE_PATH:
        candidate = os.path.join(directory, "pulse")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def pulse_on_secure_path() -> bool:
    """Is there a `pulse` in one of the directories sudo will actually search?"""
    return secure_path_pulse() is not None


def world_writable_by_others(path: str) -> Optional[str]:
    """Is `path` one that somebody other than its owner and root could replace?

    The launcher makes root exec this file. That is fine when only its owner can write
    it -- they are the person running sudo. It is not fine if the file, or a directory
    on the way to it, is writable by others: then anyone who can write there chooses
    what root runs. Returns the offending path, or None.
    """
    current = os.path.abspath(path)
    seen = set()
    while current and current not in seen:
        seen.add(current)
        try:
            info = os.lstat(current)
        except OSError:
            return None
        if info.st_mode & 0o002 and not (info.st_mode & 0o1000):
            return current          # world-writable and not sticky
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


LAUNCHER_PATH = "/usr/local/bin/pulse"

# A `pulse` sudo can find, that still runs the install this command was run from. Shell,
# not Python: it has to work before anything of Pulse's is importable by root.
#
# The path to the real pulse is baked in as an absolute path rather than looked up. A
# PATH lookup would find this launcher itself -- /usr/local/bin is on every PATH -- and
# exec itself until the process table gives out.
LAUNCHER_TEMPLATE = """#!/bin/sh
# Pulse, runnable under sudo. Written by `pulse install-sudo`.
#
# `sudo pulse` fails on its own twice over: sudo replaces PATH with secure_path, which
# does not contain the ~/.local/bin that pip installs the command into, and even by full
# path root's Python does not look in the invoking user's site-packages, so it fails on
# numpy. Carrying the real user's HOME across leaves both findable.
PULSE={pulse}

# Started empty deliberately: without this the test below reads whatever REAL_HOME the
# caller happened to export, and this file is on every user's PATH, not just sudo's.
REAL_HOME=
if [ -n "$SUDO_USER" ]; then
    # getent on Linux; dscl on macOS, which has no getent. head: a host with both files
    # and LDAP can answer twice.
    if command -v getent >/dev/null 2>&1; then
        REAL_HOME=$(getent passwd "$SUDO_USER" 2>/dev/null | head -n 1 | cut -d: -f6)
    elif command -v dscl >/dev/null 2>&1; then
        REAL_HOME=$(dscl . -read "/Users/$SUDO_USER" NFSHomeDirectory 2>/dev/null \\
                    | sed 's/^NFSHomeDirectory: //')
    fi
    if [ -z "$REAL_HOME" ]; then
        echo "pulse: cannot find the home directory of '$SUDO_USER'." >&2
        echo "pulse: without it Pulse runs with root's, and will not find your install." >&2
        exit 1
    fi
else
    REAL_HOME="$HOME"
fi

if [ ! -x "$PULSE" ]; then
    echo "pulse: $PULSE is gone (reinstalled elsewhere?)." >&2
    echo "pulse: re-run 'pulse install-sudo' as yourself to point this at it." >&2
    exit 127
fi
exec env HOME="$REAL_HOME" "$PULSE" "$@"
"""


def launcher_text(pulse_path: str) -> str:
    """The launcher script, pointing at a particular pulse."""
    return LAUNCHER_TEMPLATE.format(pulse=shlex.quote(pulse_path))


def _this_pulse() -> Optional[str]:
    """The absolute path of the `pulse` command that is running now.

    Not an assumed ~/.local/bin: Pulse may have been installed with pipx, into a venv, or
    system-wide, and the launcher has to point at the one the person actually uses.
    """
    entry = sys.argv[0] if sys.argv else ""
    if entry and os.path.basename(entry) == "pulse":
        resolved = os.path.abspath(entry)
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
    found = shutil.which("pulse")
    if found:
        return os.path.abspath(found)
    candidate = os.path.join(os.path.expanduser("~"), ".local", "bin", "pulse")
    return candidate if os.path.isfile(candidate) else None


def install_sudo_launcher(target: str = LAUNCHER_PATH) -> int:
    """Put a `pulse` where sudo looks, so `sudo pulse ...` works.

    Run as yourself, not under sudo: it asks sudo only to place one small file. That is
    the way round it has to be -- if `sudo pulse` worked there would be nothing to fix --
    and it means the launcher can record who the real user is.
    """
    import subprocess
    import tempfile

    if os.name != "posix":
        print("\n  This is for Linux and macOS; on Windows there is no sudo to fix.\n")
        return 1

    user_pulse = _this_pulse()
    if not user_pulse:
        print("\n  Cannot find the `pulse` command to point at.")
        print("  Install Pulse first:   pip install --user pulseml\n")
        return 1
    if os.path.abspath(user_pulse) == os.path.abspath(target):
        # Pointing the launcher at itself would exec itself forever.
        print(f"\n  The only pulse found is {target} itself, which is the launcher.")
        print("  Install Pulse properly first:   pip install --user pulseml\n")
        return 1

    # The long form works and needs no launcher, so say it here: after the launcher is
    # in place sudo_hint() shortens to `sudo pulse`, which would be circular advice in
    # the refusal below.
    long_form = " ".join(shlex.quote(part) for part in _sudo_env_command([]))

    exposed = world_writable_by_others(user_pulse)
    if exposed:
        # The launcher would make root exec this. If anyone can write it, anyone
        # chooses what root runs.
        print(f"\n  Refusing to install: {exposed} is writable by anybody.")
        print(f"  A launcher would make root run {user_pulse}, so whoever can write")
        print("  there would decide what root runs. Fix the permissions first.\n")
        return 1

    existing_elsewhere = secure_path_pulse()
    if existing_elsewhere and os.path.abspath(existing_elsewhere) != os.path.abspath(target):
        # /usr/local/bin comes before /usr/bin on every PATH, so installing here would
        # shadow a system-wide pulse for every user on the machine.
        print(f"\n  There is already a pulse sudo can find: {existing_elsewhere}")
        print("  `sudo pulse` should work as it is. Installing a launcher would shadow")
        print("  that one for everybody on this machine, so this is leaving it alone.\n")
        return 0

    launcher = launcher_text(user_pulse)
    if os.path.lexists(target):
        try:
            with open(target, encoding="utf-8", errors="replace") as handle:
                existing = handle.read()
        except OSError as exc:
            # A directory, or root-owned and unreadable: either way not ours to replace.
            print(f"\n  {target} is in the way and cannot be read ({exc.strerror}).")
            print(f"  Use:   {long_form}\n")
            return 1
        if existing == launcher:
            print(f"\n  Already installed at {target} -- `sudo pulse` works.\n")
            return 0
        if existing.startswith("#!/bin/sh\n# Pulse, runnable under sudo."):
            print(f"\n  Updating the launcher at {target} to point at {user_pulse}.\n")
        else:
            print(f"\n  {target} exists and is not this launcher; leaving it alone.")
            print(f"  Use:   {long_form}\n")
            return 1
    else:
        print(f"\n  Writing a launcher to {target} so `sudo pulse` works.")
        print(f"  It runs {user_pulse}, with your HOME, as root.\n")

    with tempfile.NamedTemporaryFile("w", suffix="-pulse", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(launcher)
        staged = handle.name
    os.chmod(staged, 0o755)
    try:
        command = ["sudo", "install", "-m", "0755", staged, target]
        print(dim("      " + " ".join(shlex.quote(part) for part in command) + "\n"))
        result = subprocess.run(command)
    except OSError as exc:
        print(f"  Could not run sudo: {exc}\n")
        return 1
    finally:
        try:
            os.unlink(staged)
        except OSError:
            pass

    if result.returncode != 0:
        print("\n  sudo did not install it. Without it, the command that works is:")
        print(f"      {sudo_hint([])}\n")
        return 1
    print("  Done. `sudo pulse train.py` now works.\n")
    return 0


def sudo_relaunch_command(args: Optional[List[str]] = None) -> List[str]:
    """A sudo command that runs `pulse <args>` and actually works.

    `sudo pulse ...` does not, on a pip install: sudo's secure_path does not include the
    ~/.local/bin the command lives in, so the shell reports "pulse: command not found".
    Giving the full path does not work either -- root's Python does not look in the
    user's site-packages, so it fails on numpy instead. Carrying PATH and HOME across is
    what leaves both findable.

    `args` is what should follow `pulse`. It defaults to this process's own arguments,
    which is right when the answer is "run again with sudo", and wrong whenever the
    caller means something more specific -- so callers that know say so.
    """
    args = list(sys.argv[1:] if args is None else args)
    # A `pulse` sudo can find -- a system-wide install, or a launcher in /usr/local/bin
    # -- makes the short form correct, and anything longer is noise.
    if pulse_on_secure_path():
        return ["sudo", "pulse"] + args
    return _sudo_env_command(args)


def _sudo_env_command(args: List[str]) -> List[str]:
    """The long form: works with no launcher, by carrying PATH and HOME across."""
    home = os.path.expanduser("~")
    entry = (sys.argv[0] if sys.argv else "") or "pulse"
    return (["sudo", "env", f"PATH={os.environ.get('PATH', '')}", f"HOME={home}"]
            + ([sys.executable, "-m", "pulse"] if entry.endswith(".py") else [entry])
            + list(args))


def sudo_hint(args: Optional[List[str]] = None) -> str:
    return " ".join(shlex.quote(part) for part in sudo_relaunch_command(args))


def _can_ask() -> bool:
    """Is there a person at both ends to ask a question of?

    stdin alone is not enough. Under `redirect_stdout` -- which is how the tests drive
    these paths -- the prompt is written into a buffer nobody sees while input() blocks
    on a terminal that was never told anything was wanted. That hangs a test run with
    no output at all, and answering it would exec sudo over the test runner.
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def offer_sudo(args: Optional[List[str]] = None) -> bool:
    """Show the command, and run it here if there is somebody to ask."""
    command = sudo_relaunch_command(args)
    print(f"      {sudo_hint(args)}\n")
    if not pulse_on_secure_path():
        # The long form is what works today; the short one is one command away.
        print(dim("  (`pulse install-sudo` makes plain `sudo pulse ...` work instead)\n"))
    if not _can_ask():
        return False
    if not confirm("Run that now?"):
        return False
    try:
        os.execvp(command[0], command)          # replaces this process
    except OSError as exc:
        print(f"\n  Could not start sudo: {exc}\n")
    return False


def watch_by_name(wanted: str, model: str = "") -> int:
    """No Pulse run of this name -- so find the process itself and watch that.

    Looking the pid up by hand is work the tool can do: there is exactly one process
    running that file nearly every time. `sudo pulse train.py` is the whole command.
    """
    from . import pulse_attach as attach

    name = os.path.basename(wanted)
    running = [p for p in unmonitored_python_processes()
               if os.path.basename(p["script"]) == name]

    if not running:
        print(f"\n  No Pulse run for {name!r}, and no process running it either.\n")
        print(f"  Start it with:   pulse run --stream {name}\n")
        return 1

    if len(running) > 1:
        pids = ", ".join(str(p["pid"]) for p in running)
        print(f"\n  {len(running)} processes are running {name!r} (pid {pids}).")
        print("  Which one?\n")
        for process in running:
            print(f"    sudo pulse attach --pid {process['pid']}"
                  f"   {dim('started ' + ago(process['started']))}")
        print()
        return 1

    pid = running[0]["pid"]
    report = attach.describe_readiness(pid)

    if not report["pyspy"]:
        print(f"\n  {name} is running (pid {pid}), started outside Pulse.\n")
        print("  Pulse can watch it from outside by reading its memory, which needs py-spy:\n")
        print("      pip install py-spy\n")
        print(f"  Or restart it under Pulse:   pulse run --stream {name}\n")
        return 1

    if report["needs_root"]:
        scope = report.get("ptrace_scope")
        print(f"\n  {name} is running (pid {pid}), started outside Pulse.\n")
        print("  Watching it means reading another process's memory. That is ptrace, and")
        print(f"  Linux allows it only for a parent process or root"
              f"{f' (ptrace_scope is {scope})' if scope is not None else ''}. So:\n")
        if report.get("passwordless_sudo") is False:
            print(dim("  (sudo will ask for your password)"))
        offer_sudo([wanted])
        print(f"  Or restart it under Pulse:   pulse run --stream {name}\n")
        return 1

    if not report["ok"]:
        print(f"\n  {name} is running (pid {pid}), but Pulse cannot read it:")
        print(f"  {report['reason']}\n")
        print(f"  Restart it under Pulse instead:   pulse run --stream {name}\n")
        return 1

    return attach_to_pid(pid, model=model)


def attach_to_pid(pid: int, model: str = "") -> int:
    """Watch a process Pulse did not start, by sampling it from outside."""
    from . import pulse_attach as attach

    report = attach.describe_readiness(pid)
    if not report["ok"]:
        print(f"\n  Cannot read pid {pid}: {report['reason']}\n")
        if not report["pyspy"]:
            print("  Install it:   pip install py-spy\n")
        elif report["needs_root"] and os.geteuid() != 0:
            scope = report.get("ptrace_scope")
            print("  Reading another process's memory is ptrace, which Linux allows only")
            print(f"  for a parent process or root"
                  f"{f' (ptrace_scope is {scope})' if scope is not None else ''}. Try:\n")
            offer_sudo(["attach", "--pid", str(pid)])
        return 1

    monitor = attach.AttachedMonitor(pid)
    first = monitor.sample_once()
    monitor.snapshot()          # so the console has a step and a value to show at once
    if not first:
        print(f"\n  pid {pid} can be read, but no numbers came back.")
        print("  Its loop is probably at the top level of the script, where the values are")
        print("  module globals; those cannot be read from outside. A loop inside a")
        print("  function can be, and a run started under Pulse always can.\n")
        monitor.discard()
        return 1
    # The sample above is still in the writer's queue; without this the console's first
    # poll finds an empty spool and the header says "no steps yet" about a run it has
    # just read a step from.
    monitor.writer.flush()
    monitor.start()
    print(f"\n  Watching pid {pid} from outside"
          f"{' (' + os.path.basename(monitor.script) + ')' if monitor.script else ''}: "
          f"{', '.join(sorted(first))}")
    print(dim(f"  Sampled {1 / monitor.interval:g} time(s) a second through py-spy; a spike "
              f"between samples is not seen.\n"))

    session = _describe_session(monitor.directory) or {
        "session_id": monitor.session_id, "directory": monitor.directory,
        "script": monitor.script, "status": "live", "step": 0}
    agent = None
    if model:
        from .pulse_brain import build_litellm_agent
        agent = build_litellm_agent(model)
    try:
        return run_console(session, [session], agent=agent)
    finally:
        monitor.stop()


def confirm(question: str) -> bool:
    """Ask before doing something to someone's training run."""
    try:
        return input(f"  {question} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


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


def matching_sessions(sessions: List[Dict[str, Any]], wanted: str) -> List[Dict[str, Any]]:
    """Every run that answers to `wanted`: an index, an id, or a script name."""
    if wanted.isdigit() and 1 <= int(wanted) <= len(sessions):
        return [sessions[int(wanted) - 1]]
    exact = [s for s in sessions if s.get("session_id") == wanted]
    if exact:
        return exact
    name = os.path.basename(wanted)
    return [s for s in sessions
            if wanted in (s.get("session_id") or "")
            or name == os.path.basename(s.get("script") or "")
            or wanted in os.path.basename(s.get("script") or "")]


def pick_session(sessions: List[Dict[str, Any]], wanted: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve what the user asked for: an index, a session id, a script name, or nothing."""
    if wanted:
        matches = matching_sessions(sessions, wanted)
        if len(matches) == 1:
            return matches[0]
        live = [s for s in matches if s["status"] in ("live", "stalled")]
        if len(live) == 1:
            return live[0]          # several runs of one script, only one still going
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
    console.brain.poll_once()        # so the first status line has something in it
    console.start()                  # only now: two threads on one reader read it twice

    script = session.get("script") or "?"
    directory = console.workdir
    # Say so when the run is over. Without this the header read exactly like a live
    # run and the status line said "healthy" about numbers that stopped moving
    # yesterday -- the detectors are right, there is nothing wrong with the data, it
    # is just finished.
    status = session.get("status")
    over = "" if status in ("live", "stalled", None) else f"  {yellow(status)}"
    if over and session.get("last_seen"):
        over += dim(f" {ago(session['last_seen'])}")
    print(f"\nAttached to {bold(os.path.basename(script))}  {dim(directory)}{over}")
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
        elif command == "trace":
            console.show_trace(argument)
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
            console.stop(join=True)      # its pump must not print into the next session
            session = chosen
            console = Console(session, agent=agent, sensitivity=sensitivity)
            console.brain.poll_once()
            console.start()
            directory = console.workdir
            print(f"\nAttached to {bold(os.path.basename(session.get('script') or '?'))}  "
                  f"{dim(directory)}")
            print(f"{console.status_line()}\n")
        elif command == "interval":
            try:
                seconds = float(argument)
            except (TypeError, ValueError):
                print("\n  /interval takes a number of seconds, e.g. /interval 0.5\n")
                continue
            if not math.isfinite(seconds) or not (MIN_SAMPLE_INTERVAL <= seconds <= MAX_SAMPLE_INTERVAL):
                print(f"\n  /interval takes {MIN_SAMPLE_INTERVAL:g} to {MAX_SAMPLE_INTERVAL:g} "
                      f"seconds. Longer than that and the run would stop answering.\n")
                continue
            console.send_control(stream.CONTROL_SET_INTERVAL, interval=seconds)
        elif command == "stop":
            if not confirm("Stop the training run?"):
                continue
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
    if command == "install-sudo":
        return install_sudo_launcher()
    if command in ("watch", "attach", "console", "sessions"):
        argv = argv[1:]
    # Walk the arguments once, so the value of --model is never mistaken for the run
    # being asked for: `pulse watch --model gpt-5 2` has to attach to run 2.
    model, wanted, skip = "", None, False
    for index, argument in enumerate(argv):
        if skip:
            skip = False
            continue
        if argument in ("--model", "-m"):
            model = argv[index + 1] if index + 1 < len(argv) else ""
            skip = True
        elif argument.startswith("--model="):
            model = argument.split("=", 1)[1]
        elif not argument.startswith("-") and wanted is None:
            wanted = argument
    model = model or os.environ.get("PULSE_MODEL", "")

    if "--pid" in argv:
        index = argv.index("--pid")
        try:
            pid = int(argv[index + 1])
        except (IndexError, ValueError):
            print("\n  --pid needs a process id, e.g. pulse attach --pid 12345\n")
            return 1
        return attach_to_pid(pid, model=model)

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

    idle = unmonitored_python_processes()
    if not sessions:
        print("  Nothing Pulse is watching yet. Start a run with either:\n")
        print("    pulse run --stream train.py   (no changes to your script)")
        print("    auto_track(mode=\"stream\")     (from inside it)")
        report_unmonitored(idle)
        return 1
    # Even with runs of our own: a run started outside Pulse is the one people are
    # looking for when they say it "doesn't see" their run. Skipped when they named a
    # script, because the answer about THAT script is more useful than a list of others.
    if not wanted:
        report_unmonitored(idle)

    if wanted:
        # A run of this script going NOW beats any number of finished sessions for it.
        # Pulse keeps sessions after a run ends, so they pile up: by the second day,
        # `pulse train.py` was offering a choice between yesterday's corpses while the
        # process that is actually training went unwatched. This has to consider every
        # match, not just the one pick_session would have chosen -- several dead
        # sessions are still all dead, and that was the case it missed.
        matched = matching_sessions(sessions, wanted)
        if not any(s["status"] in ("live", "stalled") for s in matched):
            name = os.path.basename(wanted)
            if any(os.path.basename(p["script"]) == name for p in idle):
                return watch_by_name(wanted, model=model)

    session = pick_session(sessions, wanted)
    if session is None:
        matches = matching_sessions(sessions, wanted) if wanted else [
            s for s in sessions if s["status"] in ("live", "stalled")]
        if len(matches) > 1:
            # Live ones first: if any are still going, an ended run is not what "watch
            # my run" means.
            live = [s for s in matches if s["status"] in ("live", "stalled")]
            shown = live or matches
            print(f"\n  {len(shown)} runs match{' ' + repr(wanted) if wanted else ''}. Which one?\n")
            render_session_list(shown)
            # By id, not by number: the numbers above are positions in THIS list, while
            # `pulse watch 2` counts through every run on the machine, so on a box with
            # other runs going the number would pick a different one.
            print("\n  pulse watch " + (shown[0].get("session_id") or "<id>") + "\n")
        elif wanted:
            return watch_by_name(wanted, model=model)
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
