"""Sweep 2 of src/pulse/pulse.py (area: core) -- auto_track guards, CLI/GUI
tracers, GUI worker bridge, GUI restart, REPLAY, DOCLOOKUP, fix splicing,
rank status.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_*  are regression coverage for behaviour verified to work.

Tracer tests run in a subprocess so sys.settrace, excepthooks and module
globals never leak into the pytest process.
"""
import json
import os
import queue as std_queue
import subprocess
import sys
import tempfile
import textwrap
import time
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_src():
    d = HERE
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "src", "pulse")):
            return os.path.join(d, "src")
        d = os.path.dirname(d)
    raise RuntimeError("could not locate src/pulse")


SRC = _find_src()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("PULSE_LOGGING", "0")

import pulse.pulse as core  # noqa: E402


def _env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["PULSE_LOGGING"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update(extra)
    return env


def run_py(code, timeout=120, files=None, env_extra=None):
    """Run `code` as script_under_test.py in a fresh temp dir (plus any `files`)."""
    with tempfile.TemporaryDirectory() as td:
        for name, text in (files or {}).items():
            with open(os.path.join(td, name), "w") as f:
                f.write(textwrap.dedent(text))
        path = os.path.join(td, "script_under_test.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(code))
        return subprocess.run([sys.executable, path], capture_output=True, text=True,
                              timeout=timeout, env=_env(**(env_extra or {})), cwd=td)


def _line(r, tag):
    lines = [l for l in r.stdout.splitlines() if l.startswith(tag)]
    assert lines, f"no {tag} line\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-3000:]}"
    return lines[-1]


# Stand-in for PulseCLI so _start_cli_tracker / auto_track(mode="cli") run headless.
FAKE_CLI = r'''
import sys, os, time, threading
sys.path.insert(0, SRC)
os.environ["PULSE_LOGGING"] = "0"
import pulse.pulse as core
import pulse.pulse_cli as pc

class FakeCLI:
    instances = []
    def __init__(self, discovered=None):
        self.discovered = dict(discovered or {})
        self.tracked_vars = []
        self.var_states = {}
        self.watch_locals = {}
        self.auto_mode = False
        self.agent_provider = None
        self.auto_intervene = True
        self.updates = []
        self.crashes = 0
        self._last_call_failed_transiently = False
        FakeCLI.instances.append(self)
    def set_code_text(self, *a, **k): pass
    def print_banner(self): pass
    def interactive_setup(self): pass
    def _default_state_for(self, name, val): return "lotrack"
    def update(self):
        self.updates.append((time.time(), self.watch_locals.get("step")))
        mark = os.environ.get("UPDATE_MARK")
        if mark:
            with open(mark, "a") as fh:
                fh.write(f"{os.getpid()}\n")
    def capture_crash_state(self, frames): pass
    def handle_crash(self, tb):
        self.crashes += 1
        return "sig"
    def offer_known_fix(self, sig): return True

pc.PulseCLI = FakeCLI
core._install_keras_pulse_hook = lambda cli: None
'''

# Stand-ins so auto_track(mode="ui") runs its GUI tracer without Tk, a
# dashboard process or a renderer process. Every mp.Queue() the GUI path makes
# is a ScriptQ: control_queue is ScriptQ.instances[0], response_queue [1].
GUI_PRELUDE = r'''
import sys, os, time, collections
sys.path.insert(0, SRC)
os.environ["PULSE_LOGGING"] = "0"
import numpy as np
import pulse.pulse as core

T0 = time.time()
logged = []
class FakeBG:
    def __init__(self, *a, **k): self.queue = None
    def log_matrix(self, var, m, config_override=None):
        logged.append((var, time.time()))
        return True
    def shutdown(self): pass
class FakeProc:
    def __init__(self, *a, **k): pass
    def start(self): pass
    def terminate(self): pass
class ScriptQ:
    instances = []
    def __init__(self, *a, **k):
        self.items = collections.deque()
        ScriptQ.instances.append(self)
    def get_nowait(self):
        if self.items:
            return self.items.popleft()
        raise Exception("empty")
    def get(self, timeout=None):
        if self.items:
            return self.items.popleft()
        time.sleep(timeout or 0)
        raise Exception("empty")
    def put(self, x, *a, **k): self.items.append(x)
class Dlg:
    def run(self): return ("FakeProv", "k")
core.HeatmapCreatorBG = FakeBG
core.mp.Process = FakeProc
core.mp.Queue = ScriptQ
core.AgentSetupDialog = Dlg
core.PROVIDERS["FakeProv"] = {"model": "x/y", "env_key": None}
core._install_pulse_excepthook = lambda sid: None
'''


def _indent(block):
    return textwrap.indent(block, "    ").strip()


# ---------------------------------------------------------------------------
# Fake tkinter so the GUI ChatPanel's non-Tk logic can be exercised headless.
# ---------------------------------------------------------------------------
@pytest.fixture
def chat_cls(monkeypatch):
    fake_tk = types.SimpleNamespace(Frame=type("Frame", (object,), {}), TclError=Exception)
    monkeypatch.setattr(core, "tk", fake_tk)
    monkeypatch.setattr(core, "HAS_TK", True)
    monkeypatch.setattr(core, "_CHAT_PANEL_CLS", None)
    cls = core._chat_panel_class()
    yield cls
    core._CHAT_PANEL_CLS = None


def make_panel(cls, script_path=None, code=None, extra_files=None, manifest=None):
    p = cls.__new__(cls)
    p.get_manifest_fn = lambda: manifest or {}
    p.get_code_fn = lambda: code
    p.script_path = script_path
    p.get_extra_files_fn = lambda: dict(extra_files or {})
    p.on_code_change = None
    p.promote_fn = None
    p.exec_fn = None
    p.restart_fn = None
    p._label_for_path, p._path_for_label = {}, {}
    p.history = []
    p._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "cost_usd": 0.0}
    p._changelog_baseline = None
    p._last_commit_id = None
    p._fix_applied_this_turn = False
    p._last_call_failed_transiently = False
    p._pending_agent_retry = None
    p.dist_info = (0, 0, 1, None)
    p.appended = []
    p.after = lambda ms, fn=None: None
    p._append = lambda who, text: p.appended.append((who, text))
    p._set_stage = lambda label: None
    return p


# ===========================================================================
# BUGS
# ===========================================================================

# --- GUI tracer: the module-level-auto_track fix was only applied to the CLI --

_GUI_MAIN_LOOP = f"""
SRC = {SRC!r}
{GUI_PRELUDE}
core.auto_track(mode="ui", throttle_interval=0.05)     # module level, loop in main()

def main():
    ctrl, resp = ScriptQ.instances[0], ScriptQ.instances[1]
    w = np.zeros((2, 2))
    sent_at = answered_at = None
    t0 = time.time()
    while time.time() - t0 < 2.0:
        w = w + 1                     # library calls only: no user function per step
        time.sleep(0.005)
        if sent_at is None and time.time() - t0 > 0.5:
            ctrl.put(("EXEC", 7, "REPL", "1 + 1"))
            sent_at = time.time()
        if answered_at is None and resp.items:
            answered_at = time.time()
    return sent_at, answered_at

sent_at, answered_at = main()
sys.settrace(None)
late = [t for v, t in logged if v == "w" and t - T0 > 0.6]
print("LATE_W", len(late))
print("EXEC_ANSWERED_IN_LOOP", answered_at is not None)
"""


@pytest.fixture(scope="module")
def gui_main_loop_run():
    return run_py(_GUI_MAIN_LOOP)


def test_bug_gui_module_level_auto_track_stops_observing_loop_in_main(gui_main_loop_run):
    """The first sweep fixed 'auto_track() at module level, loop in main()' for the
    CLI tracer only (_retrace_running_frames). The GUI persistent_tracer has the
    same bug: main()'s frame is traced only while the first window is open; on its
    first line event after that window closes it gets f_trace_lines = False and it
    never re-enters via a 'call', so no later window sees the loop's variables --
    the dashboard freezes after ~0.1 s of training. Correct: the loop's locals keep
    being logged in later windows (as the CLI now does)."""
    late = int(_line(gui_main_loop_run, "LATE_W").split()[1])
    assert late >= 2, f"'w' logged {late} times after the first window"


def test_bug_gui_control_queue_starves_when_loop_in_main_calls_only_library_code(gui_main_loop_run):
    """Same layout. persistent_tracer now polls control_queue only from events on
    user-code frames; main()'s line events are switched off after the first window,
    and a step made of library calls (torch/numpy, nn.Sequential) produces no other
    user-frame event. So EXEC requests (REPL/SHAPETRACE/GRADCHECK/REPLAY), PAUSE and
    the RESTART that applies an agent fix are never picked up while training runs:
    the chat's tool calls time out after 20 s and a fix is never restarted.
    Correct: a control message is handled within the loop."""
    assert _line(gui_main_loop_run, "EXEC_ANSWERED_IN_LOOP") == "EXEC_ANSWERED_IN_LOOP True"


def test_bug_gui_replay_snapshots_taken_in_main_cannot_be_replayed():
    """REPLAY snapshots are now taken on the training thread from caller_frame *or
    a frame it calls* ('auto_track() at module level -- in a function it calls
    (main())'), keyed by main()'s local names. But EXEC requests (REPLAY included)
    run against caller_frame -- the module frame -- whose namespace has neither
    `model` nor `train_step`, and main()'s line events stop after the first window
    so hardly any snapshot is taken. REPLAY can never work in the very layout the
    snapshot code was extended for. Correct: REPLAY replays from main()'s state."""
    code = f"""
    SRC = {SRC!r}
    {_indent(GUI_PRELUDE)}
    import torch
    core._REPLAY_MIN_INTERVAL = 0.05
    core.auto_track(mode="ui", throttle_interval=0.05)

    def main():
        ctrl, resp = ScriptQ.instances[0], ScriptQ.instances[1]
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        x, y = torch.randn(16, 4), torch.randn(16, 1)

        def train_step():
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
            return loss.item()

        answer = None
        sent = False
        t0 = time.time()
        while time.time() - t0 < 4.0:
            train_step()
            time.sleep(0.01)
            if not sent and time.time() - t0 > 1.0:
                ctrl.put(("EXEC", 1, "REPLAY", "1"))
                sent = True
            if resp.items:
                answer = resp.items.popleft()[1]
                break
        return answer

    answer = main()
    sys.settrace(None)
    print("SNAPS", len(core._REPLAY_CHECKPOINTS))
    print("ANSWER", repr(answer))
    """
    r = run_py(code)
    answer = _line(r, "ANSWER")
    assert "replayed" in answer, answer + " | " + _line(r, "SNAPS")


def test_bug_gui_restart_reports_a_failed_chain_once_and_still_tries_later_fixes():
    """GUI RESTART handler: restart_state['attempts'] is never reset. After the 5th
    failed restart it gives up on the chain -- but every LATER RESTART (the agent's
    next fix, for this or another problem) used to find attempts == 5, never launch
    the fixed script, and give up again, so a brand-new fix was never even tried.
    Correct: one report per failed chain; a later fix is at least tried."""
    code = f"""
    SRC = {SRC!r}
    {_indent(GUI_PRELUDE)}
    import io
    launches, rollbacks = [], []
    class FakePopen:
        def __init__(self, argv, **kw):
            launches.append(argv)
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("Traceback: boom\\n")
        def wait(self): return 1
    core.subprocess.Popen = FakePopen
    core._report_failed_fix_chain_gui = lambda path, cid: rollbacks.append(cid)
    core._load_fix_history = lambda p: [{{"id": "c1"}}]
    core._atomic_write_json = lambda *a, **k: None

    def main():
        w = np.zeros((2, 2))
        core.auto_track(mode="ui", throttle_interval=0.05)
        ctrl = ScriptQ.instances[0]
        for i in range(7):
            ctrl.put(("RESTART", "FakeProv"))
            t0 = time.time()
            while time.time() - t0 < 0.3:
                w = w + 1
                time.sleep(0.005)
        sys.settrace(None)
        print("LAUNCHES", len(launches))
        print("ROLLBACKS", len(rollbacks))
    main()
    """
    r = run_py(code)
    launches = int(_line(r, "LAUNCHES").split()[1])
    rollbacks = int(_line(r, "ROLLBACKS").split()[1])
    assert launches >= 5, (launches, r.stdout[-2000:])
    assert rollbacks <= 1, f"reported {rollbacks} times; later fixes launched {launches - 5} times"


# --- auto_track guards ---------------------------------------------------------

def test_bug_auto_track_after_shutdown_is_silently_ignored():
    """pulse.shutdown() (public, exported next to auto_track) now stops CLI tracing
    for good, and _AUTO_TRACK_STARTED is never cleared. A later auto_track() -- the
    next fold / next model in the same process, or a notebook re-run after
    shutdown() -- is a silent no-op (only a debug-log line says 'already running',
    which is false), and even if the flag were cleared _CLI_TRACING_STOPPED would
    keep the new ticker from ever opening a window. Correct: after shutdown(), a new
    auto_track() starts a working session."""
    code = f"""
    SRC = {SRC!r}
    {_indent(FAKE_CLI)}

    def run(seconds):
        step = 0
        t0 = time.time()
        while time.time() - t0 < seconds:
            step += 1
            time.sleep(0.002)

    def main():
        core.auto_track(mode="cli", throttle_interval=0.1)
        run(0.4)
        core.shutdown()
        core.auto_track(mode="cli", throttle_interval=0.1)
        t_second = time.time()
        run(1.2)
        sys.settrace(None)
        late = [t for t, s in FakeCLI.instances[-1].updates if t - t_second > 0.4]
        print("SESSIONS", len(FakeCLI.instances))
        print("LATE", len(late))
    main()
    """
    r = run_py(code)
    assert _line(r, "SESSIONS") == "SESSIONS 2", r.stdout[-1500:]
    assert int(_line(r, "LATE").split()[1]) >= 2


def test_bug_auto_track_skipped_in_script_run_by_multiprocessing_launcher():
    """The spawn guard skips any module-level auto_track() when mp.parent_process()
    is set. A sweep/launcher that runs each trial's script in a multiprocessing
    worker (runpy.run_path(path, run_name='__main__') inside Pool/Process) is a
    module-level call of a real __main__ -- not the spawn re-import -- and Pulse is
    silently skipped for every trial. Correct: only the __mp_main__ re-import is
    skipped; a script run as __main__ gets Pulse."""
    train = '''
    import pulse.pulse as core
    core._determine_mode = lambda m: (_ for _ in ()).throw(RuntimeError("reached setup"))
    if __name__ == "__main__":
        try:
            core.auto_track()
            print("RESULT skipped", flush=True)
        except RuntimeError as exc:
            print("RESULT", exc, flush=True)
    '''
    code = """
    import multiprocessing as mp, os, runpy, sys
    def trial(path):
        runpy.run_path(path, run_name="__main__")
    if __name__ == "__main__":
        here = os.path.dirname(os.path.abspath(__file__))
        ctx = mp.get_context("fork")
        p = ctx.Process(target=trial, args=(os.path.join(here, "train_trial.py"),))
        p.start(); p.join(60)
    """
    r = run_py(code, files={"train_trial.py": train})
    assert _line(r, "RESULT") == "RESULT reached setup"


def test_bug_auto_track_in_helper_module_runs_again_in_every_spawn_worker(tmp_path):
    """auto_track() at module level of a helper module the script imports
    (e.g. `import pulse_setup`) runs again in every 'spawn'/'forkserver' child (DataLoader
    workers on macOS/Windows, torch.multiprocessing.spawn): the child re-imports the
    main module, which imports the helper, before parent_process() is set, and the
    helper's __name__ is not '__mp_main__' -- so a whole Pulse session (banner,
    setup prompt, billed start-of-run call) starts in every worker. Correct: the
    call is skipped in the child as it is for the main module."""
    mark = tmp_path / "reached.txt"
    helper = '''
    import os
    import pulse.pulse as core
    def _reach(m):
        with open(os.environ["MARK"], "a") as fh:
            fh.write(f"{os.getpid()}\\n")
        raise RuntimeError("reached setup")
    core._determine_mode = _reach
    try:
        core.auto_track()
    except RuntimeError:
        pass
    '''
    code = """
    import multiprocessing as mp
    import pulse_setup_helper
    def work():
        pass
    if __name__ == "__main__":
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=work) for _ in range(2)]
        for p in ps: p.start()
        for p in ps: p.join(60)
        print("DONE")
    """
    r = run_py(code, files={"pulse_setup_helper.py": helper}, env_extra={"MARK": str(mark)})
    assert "DONE" in r.stdout, r.stdout + r.stderr[-2000:]
    pids = mark.read_text().split()
    assert len(pids) == 1, f"Pulse setup reached in {len(pids)} processes (parent + workers)"


# --- CLI tracer ------------------------------------------------------------------

def test_bug_cli_tracer_keeps_running_in_forked_dataloader_workers(tmp_path):
    """The CLI window starts open and stays open for ~throttle_interval after
    auto_track(); a DataLoader created right after (the usual layout) forks its
    workers from the traced training thread in that window. The child inherits the
    installed cli_tracer and window['open'] == True, but not the ticker thread, so
    the window never closes there: every line of the user's Dataset code in every
    worker is line-traced and the worker runs PulseCLI.update() (prints, check-ins,
    auto-intervention) as if it were the training process. Nothing disarms Pulse
    after fork (no os.register_at_fork). Correct: forked children run untraced and
    never call the debugger's update()."""
    pytest.importorskip("torch")
    mark = tmp_path / "updates.txt"
    code = f"""
    SRC = {SRC!r}
    {_indent(FAKE_CLI)}
    import torch
    from torch.utils.data import Dataset, DataLoader

    class DS(Dataset):
        def __len__(self):
            return 32
        def __getitem__(self, i):
            x = torch.full((2,), float(i))
            t0 = time.time()
            while time.time() - t0 < 0.06:
                y = x * 2
            return x

    def main():
        core._start_cli_tracker(sys._getframe(), None, 1.0, {{}}, {{}})
        loader = DataLoader(DS(), batch_size=4, num_workers=2, multiprocessing_context="fork")
        for batch in loader:
            pass
        sys.settrace(None)
        print("PARENT", os.getpid())
    main()
    """
    r = run_py(code, env_extra={"UPDATE_MARK": str(mark)}, timeout=180)
    parent = int(_line(r, "PARENT").split()[1])
    pids = [int(p) for p in mark.read_text().split()] if mark.exists() else []
    foreign = sorted(set(p for p in pids if p != parent))
    assert not foreign, f"cli.update() ran in forked worker(s) {foreign} ({len(pids)} updates total)"


# --- GUI renderer bridge -----------------------------------------------------------

def test_bug_heatmap_bg_eviction_reorders_control_messages():
    """log_matrix on a full queue evicts the OLDEST item; when that is a control
    message it is put back at the END of the queue. With two pending control
    messages for the same variable (promote then demote from the right-click menu,
    or a PROMOTE directive racing a user click) the first one is re-queued after
    the second and the worker applies them in the wrong order: the variable ends in
    the state the user left. Correct: control messages keep their order."""
    bg = core.HeatmapCreatorBG.__new__(core.HeatmapCreatorBG)
    bg.queue = std_queue.Queue(maxsize=2)
    bg.queue.put(("STATE", "w", "track"))
    bg.queue.put(("STATE", "w", "lotrack"))       # the user's latest choice
    bg.log_matrix("h", np.zeros((2, 2), dtype=np.float32))
    items = []
    while not bg.queue.empty():
        items.append(bg.queue.get_nowait())
    states = [it[2] for it in items if isinstance(it, tuple) and it[0] == "STATE"]
    assert states == ["track", "lotrack"], items


# --- import-time config ------------------------------------------------------------

def test_bug_bad_replay_max_mb_env_breaks_import_pulse():
    """_REPLAY_MAX_BYTES = int(float(os.environ.get('PULSE_REPLAY_MAX_MB', '256')...))
    runs at import time with no error handling: PULSE_REPLAY_MAX_MB=512MB (or '1g',
    '') makes `import pulse` raise ValueError, i.e. the user's training script dies
    on its import line. Correct: an unparsable value falls back to the default."""
    r = subprocess.run([sys.executable, "-c", "import pulse.pulse as c; print('OK', c._REPLAY_MAX_BYTES)"],
                       capture_output=True, text=True, timeout=120,
                       env=_env(PULSE_REPLAY_MAX_MB="512MB"))
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


# --- GUI agent helpers -------------------------------------------------------------

def test_bug_doclookup_fails_for_every_already_imported_library(chat_cls):
    """Sweep-1 added `import importlib.util` inside `if root not in sys.modules:` in
    ChatPanel._run_doclookup. That makes `importlib` a LOCAL name of the whole
    function, so whenever the root IS already imported -- torch, numpy, json: the
    normal case in the dashboard process -- `importlib.import_module(root)` raises
    UnboundLocalError, caught as "could not import 'torch' ... it may not be
    installed". DOCLOOKUP is broken for every installed library the process has
    loaded. (Sweep-1's test_ok check `"json.dumps" in _run_doclookup("json.dumps")`
    passes on the error message, which echoes the argument.) Correct: the real
    signature/docstring is returned."""
    import json as _json  # noqa: F401  (json is in sys.modules)
    p = make_panel(chat_cls)
    out = p._run_doclookup("json.dumps")
    assert "could not import" not in out, out
    assert "skipkeys" in out, out


def test_bug_write_code_fix_reindents_lines_already_at_file_indent(chat_cls, tmp_path):
    """_splice_fix re-bases every line of `new` from the indentation of the agent's
    FIRST old line to the file's. When the agent quotes a multi-line block with only
    the first line's indentation dropped (the snippet copied from mid-line -- an
    exact, unique match), lines 2+ of `new` already carry the file's indentation
    and get it added a second time: the loop body is pushed 4 columns right, the
    following statement no longer matches any level, the lint gate rejects the fix
    and nothing is written. Correct: the fix is applied and the file still works."""
    src = ("def train():\n"
           "    total = 0\n"
           "    for step in range(3):\n"
           "        total += step\n"
           "        total += 1\n"
           "    return total\n")
    path = tmp_path / "train.py"
    path.write_text(src)
    p = make_panel(chat_cls, script_path=str(path), code=src)
    p._build_file_labels({})
    fix = {"old": ["for step in range(3):\n        total += step"],
           "new": ["for step in range(4):\n        total += step"],
           "files": [None], "explanation": "one more step"}
    text, _applied, _orig, _skipped = p._write_code_fix(fix)
    new_src = path.read_text()
    ns = {}
    exec(compile(new_src, str(path), "exec"), ns)
    assert ns["train"]() == 10, text


def test_bug_log_matrix_blocks_training_thread_and_loses_control_message():
    """log_matrix() runs on the training thread and promises 'never wait for the
    renderer'. When the evicted item is a control message it is put back with a
    BLOCKING put(timeout=1.0). The Dashboard process shares this queue and its
    config_queue.put() of a promote/axis change is usually blocked on the full
    queue, so it takes the slot the eviction freed: the put-back then stalls the
    training step for a full second and, still full, the control message is
    dropped silently. Correct: log_matrix never blocks training and the pending
    control message is not lost (drop the frame instead)."""
    class RacyQ(std_queue.Queue):
        def get_nowait(self):
            item = super().get_nowait()
            if item[0] == "CONFIG":
                # the Dashboard's blocked put() grabs the freed slot first
                super().put_nowait(("STATE", "other", "track"))
            return item
    bg = core.HeatmapCreatorBG.__new__(core.HeatmapCreatorBG)
    bg.queue = RacyQ(maxsize=2)
    bg.queue.put(("CONFIG", "w", [0, 1]))
    bg.queue.put(("h", np.zeros((2, 2), dtype=np.float32), None))
    t0 = time.time()
    bg.log_matrix("h", np.ones((2, 2), dtype=np.float32))
    took = time.time() - t0
    items = []
    while not bg.queue.empty():
        items.append(std_queue.Queue.get_nowait(bg.queue))
    assert took < 0.25, f"training thread blocked {took:.2f}s in log_matrix"
    assert ("CONFIG", "w", [0, 1]) in items, items


def test_bug_worker_records_negative_inf_scalar_as_positive_inf(tmp_path, monkeypatch):
    """_worker_main now maps a non-finite scalar back from the stats flags: nan ->
    NaN, any inf -> float('inf'). statistics() only counts infs, so a scalar that
    went to -inf (a log-likelihood / reward diverging downwards) is stored, charted
    and shown to the agent as +inf -- the opposite direction. Correct: -inf."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    q = std_queue.Queue()
    for v in (-1.0, -2.0, float("-inf")):
        q.put(("log_likelihood", np.float64(v), None))
    q.put(None)
    core._worker_main(q, {}, "s2ninf", {})
    with open(os.path.join(str(tmp_path), "pulse_cache", "s2ninf", "manifest.json")) as f:
        man = json.load(f)
    assert man["log_likelihood"]["latest_value"] == float("-inf"), man["log_likelihood"]["latest_value"]


def test_bug_auto_track_retry_after_failed_first_call_is_ignored():
    """_AUTO_TRACK_STARTED is set before any setup runs. If the first call fails --
    auto_track(mode="ui") over SSH / in a headless notebook raises TclError from the
    Tk setup dialog, or the dialog is cancelled -- the obvious retry
    auto_track(mode="cli") is a silent no-op: no banner, no tracer, no crash hook,
    and nothing on the console says why. Correct: a call that never started a
    session doesn't block the next one."""
    code = f"""
    SRC = {SRC!r}
    {_indent(FAKE_CLI)}
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)

    def main():
        w = 1.0
        try:
            core.auto_track(mode="ui")
            print("FIRST returned")
        except Exception as exc:
            print("FIRST raised", type(exc).__name__)
        core.auto_track(mode="cli", throttle_interval=0.1)
        sys.settrace(None)
        print("SESSIONS", len(FakeCLI.instances))
    main()
    """
    r = run_py(code)
    assert _line(r, "FIRST").startswith("FIRST raised"), r.stdout[-800:]
    assert _line(r, "SESSIONS") == "SESSIONS 1"


def test_bug_plain_ml_questions_trigger_code_edit_and_restart():
    """_wants_implementation() decides whether ChatPanel._ask goes straight to the
    implement pass, which writes the fix to disk (no confirmation) and restarts
    training. The sweep-1 whole-word regex still fires on ordinary ML vocabulary:
    'fixed' (fixed seed / loss fixed at 2.30), 'patch' (ViT patch size), 'apply'
    (does weight decay apply to biases?). Asking such a question with Send Code on
    edits the user's script and restarts the run. Correct: only requests to change
    the code count."""
    questions = [
        "Why is my loss fixed at 2.30 from the first epoch?",
        "Is a patch size of 16 too large for 32x32 images?",
        "Does weight decay apply to the biases here?",
    ]
    fired = [q for q in questions if core._wants_implementation(q)]
    assert not fired, fired


# --- rank status -------------------------------------------------------------------

def test_bug_gui_gpustatus_shows_ranks_of_previous_job():
    """_current_job_rank_status learns this job's world size from the status file
    whose pid is os.getpid(). The GUI's GPUSTATUS/RANKDIVERGE run in the Dashboard
    process, whose pid is never a rank's pid, so the world-size filter is skipped
    there and only the 'older than process start - 60 s' filter remains: relaunch a
    2-GPU run within a minute of a 4-GPU one (e.g. after a crash) and the chat
    reports '4/2 rank(s) reporting' with two ghost ranks. Correct: 2/2."""
    key = "s2-core-" + os.urandom(4).hex()
    d = core._rank_status_dir(key)
    now = time.time()
    other_pid = os.getpid() + 100000      # the training processes, not this one
    try:
        for r in (2, 3):                   # previous 4-rank job, stopped seconds ago
            with open(os.path.join(d, f"rank_{r}.json"), "w") as f:
                json.dump({"rank": r, "world_size": 4, "pid": other_pid + r, "hostname": "h",
                           "updated": now - 5, "gpus": []}, f)
        for r in (0, 1):                   # the current 2-rank job
            with open(os.path.join(d, f"rank_{r}.json"), "w") as f:
                json.dump({"rank": r, "world_size": 2, "pid": other_pid + 10 + r, "hostname": "h",
                           "updated": now, "gpus": []}, f)
        report = core._format_multi_rank_gpu_status(key, 0)
        assert report.startswith("GPUSTATUS: 2/2 rank(s) reporting"), report.splitlines()[0]
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# --- found by reading (ledger r:1), tests added with the fix -----------------------

def test_bug_gui_doclookup_imports_unimported_submodule_of_project_package(chat_cls, tmp_path, monkeypatch):
    """The GUI DOCLOOKUP only vetted the ROOT name, and only when it wasn't imported yet:
    for an already-imported project package, `DOCLOOKUP: pkg.train_loop.main` imported
    (= ran) pkg/train_loop.py in the attribute walk. Correct: refused, nothing runs."""
    pkg = tmp_path / "s2core_projpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    marker = tmp_path / "ran.txt"
    (pkg / "train_loop.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\ndef main():\n    return 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import s2core_projpkg  # noqa: F401  (the script imported its own package)
    try:
        out = make_panel(chat_cls)._run_doclookup("s2core_projpkg.train_loop.main")
        assert not marker.exists(), out
        assert "s2core_projpkg.train_loop" not in sys.modules
    finally:
        for name in [m for m in sys.modules if m.startswith("s2core_projpkg")]:
            sys.modules.pop(name, None)


def test_bug_gui_doclookup_runs_project_file_in_namespace_package(chat_cls, tmp_path, monkeypatch):
    """A project folder without __init__.py is a namespace package: find_spec() gives
    no file, so the GUI check let it through and the walk imported its modules."""
    ns = tmp_path / "s2core_nsproj"
    ns.mkdir()
    marker = tmp_path / "ran.txt"
    (ns / "job.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\ndef run():\n    return 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        out = make_panel(chat_cls)._run_doclookup("s2core_nsproj.job.run")
        assert not marker.exists(), out
        assert "project" in out, out
    finally:
        for name in [m for m in sys.modules if m.startswith("s2core_nsproj")]:
            sys.modules.pop(name, None)


def test_bug_gui_poll_stops_for_good_after_one_failed_tick():
    """Dashboard._poll rescheduled itself only as its last statement. The worker prunes
    old PNGs, so an image can vanish between the exists() check and Image.open(): the
    exception escaped, the reschedule never ran and the dashboard froze for good.
    Correct: the next tick is scheduled whatever this one did."""
    d = core.Dashboard.__new__(core.Dashboard)
    scheduled = []
    d.root = types.SimpleNamespace(after=lambda ms, fn: scheduled.append(fn))

    def _boom():
        raise FileNotFoundError("heatmap_123.png was pruned")
    d._poll_once = _boom
    d._poll()
    assert len(scheduled) == 1 and scheduled[0] == d._poll


def test_bug_replay_counts_snapshots_and_serializes_on_the_training_thread(monkeypatch):
    """REPLAY: n picked the n-th newest SNAPSHOT (taken >= 10 s apart), not n steps back,
    and every snapshot's torch.save (up to 256 MB) ran on the training thread. Correct:
    REPLAY replays n steps from the latest snapshot; the UI tracer's snapshots are only
    copied on the training thread (consistently) and serialized elsewhere."""
    torch = pytest.importorskip("torch")
    import threading
    old = list(core._REPLAY_CHECKPOINTS)
    core._REPLAY_CHECKPOINTS[:] = []
    save_threads = []
    real_save = torch.save

    def _save(*a, **k):
        save_threads.append(threading.get_ident())
        return real_save(*a, **k)
    monkeypatch.setattr(torch, "save", _save)
    try:
        model = torch.nn.Linear(2, 1)
        seen = []

        def train_step():
            seen.append(float(model.weight[0, 0]))
            return 0.0
        frame = types.SimpleNamespace(f_globals={}, f_locals={"model": model, "train_step": train_step})

        def snapshot(value, count):
            with torch.no_grad():
                model.weight.fill_(value)
            core._replay_maybe_checkpoint(frame, count, background=True)
            with torch.no_grad():
                model.weight.fill_(-99.0)     # training moves on while it is being saved
            deadline = time.time() + 10
            while len(core._REPLAY_CHECKPOINTS) < count + 1 and time.time() < deadline:
                time.sleep(0.01)
        snapshot(1.0, 0)
        snapshot(2.0, 1)
        assert len(core._REPLAY_CHECKPOINTS) == 2
        assert save_threads and threading.get_ident() not in save_threads
        out = core._exec_replay(frame, "1")
        assert "replayed 1 step" in out, out
        assert seen == [2.0], f"replayed from a snapshot with weight {seen}, not the latest (2.0)"
    finally:
        core._REPLAY_CHECKPOINTS[:] = old


def test_bug_pulse_mode_env_is_ignored_by_a_plain_auto_track(monkeypatch):
    """auto_track()'s default mode="cli" beat PULSE_MODE=ui, so the environment variable
    only ever worked for 'stream'. Correct: PULSE_MODE decides when the call doesn't; an
    explicit mode still wins."""
    seen = []

    def _fake(mode):
        seen.append(mode)
        raise RuntimeError("stop before any setup")
    monkeypatch.setattr(core, "_determine_mode", _fake)
    monkeypatch.setenv("PULSE_MODE", "ui")
    with pytest.raises(RuntimeError):
        core.auto_track()
    with pytest.raises(RuntimeError):
        core.auto_track(mode="cli")
    assert seen == ["ui", "cli"]
    assert core._AUTO_TRACK_STARTED is False


def test_bug_gui_copies_miss_the_clis_fixes(chat_cls):
    """The dashboard's copies of CLI helpers never got the CLI's fixes: _lint_check had no
    `original` (a name undefined before the fix blocked every fix), _parse_code_fix
    dropped "resume" and any fix with prose around it, _scalar_history kept None
    readings, DIFFSTATS subtracted NaN, and PASS 4/5 read the string "false" as True."""
    pytest.importorskip("pyflakes")
    p = make_panel(chat_cls)
    original = "x = display(1)\ny = 1\n"
    ok, msgs = p._lint_check("x = display(1)\ny = 2\n", "train.py", original=original)
    assert ok, msgs
    ok, _ = p._lint_check("x = display(1)\ny = undefined_new\n", "train.py", original=original)
    assert not ok
    fix = chat_cls._parse_code_fix('Here is the fix:\n{"old": ["a = 1"], "new": ["a = 2"], "resume": false}')
    assert fix is not None and fix["old"] == ["a = 1"] and fix["resume"] is False
    p.get_manifest_fn = lambda: {"loss": {"history": [[0, 1.0], [1, None], [2, float("nan")]]}}
    name, hist = p._scalar_history("loss")
    assert [s for s, _ in hist] == [0, 2]
    assert "non-finite" in p._run_diffstats("loss 0 1")
    assert core._as_bool("false") is False and core._as_bool("true") is True and core._as_bool(True)


# ===========================================================================
# OK -- verified behaviour
# ===========================================================================

def test_ok_auto_track_module_level_call_skipped_in_spawn_child(tmp_path):
    """A module-level auto_track() in the main script is skipped when a 'spawn'
    child re-imports it as __mp_main__."""
    mark = tmp_path / "reached.txt"
    code = """
    import multiprocessing as mp, os
    import pulse.pulse as core
    def _reach(m):
        with open(os.environ["MARK"], "a") as fh:
            fh.write(f"{os.getpid()}\\n")
        raise RuntimeError("reached setup")
    core._determine_mode = _reach
    try:
        core.auto_track()
    except RuntimeError:
        pass
    def work():
        pass
    if __name__ == "__main__":
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=work)
        p.start(); p.join(60)
        print("DONE")
    """
    r = run_py(code, env_extra={"MARK": str(mark)})
    assert "DONE" in r.stdout, r.stderr[-2000:]
    assert len(mark.read_text().split()) == 1


def test_ok_second_auto_track_is_noop():
    """A second auto_track() in the same process (no shutdown) starts nothing."""
    code = f"""
    SRC = {SRC!r}
    {_indent(FAKE_CLI)}
    def main():
        core.auto_track(mode="cli", throttle_interval=0.1)
        core.auto_track(mode="cli", throttle_interval=0.1)
        sys.settrace(None)
        print("SESSIONS", len(FakeCLI.instances), "HOOKS_OK", sys.excepthook is core._CLI_EXCEPTHOOK_STATE["hook"])
    main()
    """
    r = run_py(code)
    assert _line(r, "SESSIONS") == "SESSIONS 1 HOOKS_OK True"


def test_ok_cli_tracer_module_level_auto_track_keeps_observing_main_loop():
    """CLI: auto_track() at module level + loop in main(): later windows see the loop."""
    code = f"""
    SRC = {SRC!r}
    {_indent(FAKE_CLI)}
    core._start_cli_tracker(sys._getframe(), None, 0.1, {{}}, {{}})

    def main():
        step = 0
        t0 = time.time()
        while time.time() - t0 < 1.5:
            step += 1
            x = step * 2
        return step

    main()
    sys.settrace(None)
    cli = FakeCLI.instances[0]
    t_first = cli.updates[0][0] if cli.updates else None
    late = [s for t, s in cli.updates if t_first is not None and t - t_first > 0.5 and s is not None]
    print("LATE", len(late))
    """
    r = run_py(code)
    assert int(_line(r, "LATE").split()[1]) >= 2


def test_ok_cli_shutdown_stops_ticker_without_deadlock():
    """shutdown() takes the arm lock the ticker holds while arming; it returns
    promptly and tracing stays off afterwards."""
    code = f"""
    SRC = {SRC!r}
    {_indent(FAKE_CLI)}
    def main():
        core._start_cli_tracker(sys._getframe(), None, 0.01, {{}}, {{}})
        t0 = time.time()
        while time.time() - t0 < 0.5:
            step = 1
        t1 = time.time()
        for _ in range(20):
            core.shutdown()
        took = time.time() - t1
        time.sleep(0.2)
        print("RESULT", sys.gettrace() is None, took < 1.0)
    main()
    """
    r = run_py(code)
    assert _line(r, "RESULT") == "RESULT True True"


def test_ok_heatmap_bg_control_message_survives_full_queue():
    bg = core.HeatmapCreatorBG.__new__(core.HeatmapCreatorBG)
    bg.queue = std_queue.Queue(maxsize=2)
    arr = np.zeros((2, 2), dtype=np.float32)
    bg.log_matrix("a", arr)
    bg.log_matrix("b", arr)
    bg.update_config("a", [0, 1])
    items = []
    while not bg.queue.empty():
        items.append(bg.queue.get_nowait())
    assert ("CONFIG", "a", [0, 1]) in items


def test_ok_worker_nan_scalar_recorded_as_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    q = std_queue.Queue()
    for v in (1.0, float("nan"), float("nan")):
        q.put(("loss", np.float64(v), None))
    q.put(None)
    core._worker_main(q, {}, "s2nan", {})
    with open(os.path.join(str(tmp_path), "pulse_cache", "s2nan", "manifest.json")) as f:
        man = json.load(f)
    hist = man["loss"]["history"]
    assert len(hist) == 2 and hist[-1][1] != hist[-1][1]      # NaN, deduplicated


def test_ok_device_is_host_variants():
    assert core._device_is_host("cpu")
    assert core._device_is_host("/job:localhost/replica:0/task:0/device:CPU:0")
    assert core._device_is_host("TFRT_CPU_0")
    assert not core._device_is_host("cuda:0")
    assert not core._device_is_host("/job:localhost/replica:0/task:0/device:GPU:0")
    assert not core._device_is_host("mps")


def test_ok_discover_project_files_relative_and_submodule(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    (proj / "s2pkg" / "sub").mkdir(parents=True)
    (proj / "s2pkg" / "__init__.py").write_text("")
    (proj / "s2pkg" / "sub" / "__init__.py").write_text("")
    (proj / "s2pkg" / "sub" / "deep.py").write_text("X = 1\n")
    (proj / "s2pkg" / "model.py").write_text("from .sub import deep\n")
    (proj / "train.py").write_text("from s2pkg.model import Net\n")
    (tmp_path / "proj_old").mkdir()
    (tmp_path / "proj_old" / "leak.py").write_text("")
    monkeypatch.syspath_prepend(str(proj))
    monkeypatch.syspath_prepend(str(tmp_path / "proj_old"))
    found = core._discover_project_files(str(proj / "train.py"), os.path.normcase(str(proj)))
    names = sorted(os.path.relpath(f, proj) for f in found)
    assert os.path.join("s2pkg", "model.py") in names
    assert os.path.join("s2pkg", "sub", "deep.py") in names
    assert not any("proj_old" in n for n in names)


def test_ok_wants_implementation_whole_words():
    assert core._wants_implementation("please fix the lr")
    assert core._wants_implementation("Can you apply it?")
    assert not core._wants_implementation("what does the prefix mean")
    assert not core._wants_implementation("explain the application")


def test_ok_replay_budget_and_identity_dedupe(monkeypatch):
    torch = pytest.importorskip("torch")
    old = list(core._REPLAY_CHECKPOINTS)
    core._REPLAY_CHECKPOINTS[:] = []
    try:
        m = torch.nn.Linear(8, 8)
        frame = types.SimpleNamespace(f_globals={}, f_locals={"model": m, "alias": m, "cls": torch.nn.Linear})
        core._replay_maybe_checkpoint(frame, 0)
        assert len(core._REPLAY_CHECKPOINTS) == 1
        assert list(core._REPLAY_CHECKPOINTS[0][1]) == ["model"]
        monkeypatch.setattr(core, "_REPLAY_MAX_BYTES", 10)
        core._REPLAY_CHECKPOINTS[:] = []
        core._replay_maybe_checkpoint(frame, 1)
        assert core._REPLAY_CHECKPOINTS == []
    finally:
        core._REPLAY_CHECKPOINTS[:] = old