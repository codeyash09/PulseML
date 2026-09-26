"""Sweep of src/pulse/pulse.py (auto_track, tracers, worker, GUI chat-panel logic,
exec directives, discovery, distributed rank status).

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_*  are regression coverage for behaviour verified to work.

Heavy / tracer tests run in a subprocess so sys.settrace, excepthooks and
module globals never leak into the pytest process.
"""
import json
import math
import os
import queue as std_queue
import subprocess
import sys
import tempfile
import textwrap
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


def run_py(code, timeout=90, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["PULSE_LOGGING"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "script_under_test.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(code))
        return subprocess.run([sys.executable, path], capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=cwd or td)


# Shared stand-in for PulseCLI so _start_cli_tracker can run without a terminal.
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
    def capture_crash_state(self, frames): pass
    def handle_crash(self, tb):
        self.crashes += 1
        return "sig"
    def offer_known_fix(self, sig): return True

pc.PulseCLI = FakeCLI
core._install_keras_pulse_hook = lambda cli: None
'''


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

def test_bug_tf_cpu_tensor_dropped_by_cpu_only_mode():
    """_is_accelerator_value treats any device string other than exactly 'cpu' as an
    accelerator. TensorFlow's CPU device string is '/job:localhost/replica:0/task:0/
    device:CPU:0', so under the default PULSE_CPU_ONLY=1 every TF CPU tensor is
    rejected by HeatmapCreatorBG.log_matrix and the UI tracer: a TF/Keras run on a
    laptop shows nothing. Correct: a tensor on a CPU device is CPU-resident."""
    tf = pytest.importorskip("tensorflow")
    t = tf.constant([[1.0, 2.0]])
    assert "CPU" in t.device
    assert core._cpu_resident(t) is True


def test_bug_heatmap_bg_drops_state_and_config_messages_when_queue_full():
    """HeatmapCreatorBG.queue has maxsize=2 and the tracer keeps it full with
    log_matrix() items. update_state()/update_config() use put_nowait and swallow
    queue.Full, so a promote (right-click / agent PROMOTE) or an axis change is
    silently lost whenever training is logging. log_matrix's drop-oldest can also
    evict a pending STATE/CONFIG. Correct: control messages must not be dropped."""
    bg = core.HeatmapCreatorBG.__new__(core.HeatmapCreatorBG)
    bg.queue = std_queue.Queue(maxsize=2)
    arr = np.zeros((2, 2), dtype=np.float32)
    bg.log_matrix("w", arr)
    bg.log_matrix("b", arr)
    bg.update_state("w", "track")
    items = []
    while not bg.queue.empty():
        items.append(bg.queue.get_nowait())
    assert ("STATE", "w", "track") in items


def test_bug_heatmap_bg_shutdown_sentinel_dropped_when_queue_full():
    """shutdown() uses put_nowait(None); when the (maxsize=2) queue is full the
    sentinel is swallowed, the worker never exits and shutdown blocks 5s in join()
    then leaves the process running. Correct: the stop sentinel is always delivered."""
    bg = core.HeatmapCreatorBG.__new__(core.HeatmapCreatorBG)
    bg.queue = std_queue.Queue(maxsize=2)
    bg.process = types.SimpleNamespace(join=lambda timeout=None: None)
    arr = np.zeros((2, 2), dtype=np.float32)
    bg.log_matrix("w", arr)
    bg.log_matrix("b", arr)
    bg.shutdown()
    items = []
    while not bg.queue.empty():
        items.append(bg.queue.get_nowait())
    assert None in items


def test_bug_worker_leaks_one_png_per_update(tmp_path, monkeypatch):
    """_worker_main writes every heatmap/line-chart frame to a NEW file
    '<var>_<time_ns>.png' in the session cache and never deletes the previous one.
    A run tracking 10 variables at 1 Hz leaves ~36k PNGs/hour in /tmp. Correct:
    superseded frames are removed (bounded number of files per variable)."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    q = std_queue.Queue()
    for i in range(6):
        q.put(("w", np.random.rand(4, 4).astype(np.float32), None))
        q.put(("loss", np.float64(1.0 / (i + 1)), None))
    q.put(None)
    core._worker_main(q, {}, "sess", {"w": "track"})
    pngs = [f for f in os.listdir(os.path.join(str(tmp_path), "pulse_cache", "sess")) if f.endswith(".png")]
    per_var = {v: sum(1 for f in pngs if f.startswith(v + "_")) for v in ("w", "loss")}
    assert per_var["w"] <= 2 and per_var["loss"] <= 2, per_var


def test_bug_gui_nan_loss_recorded_as_none_and_never_detected(tmp_path, monkeypatch):
    """_worker_main takes a scalar's value from statistics()['mean'], which is the mean
    of the FINITE elements only -- for a NaN/inf scalar it is None. So in GUI mode a
    NaN loss is stored as latest_value=None / history None ('unreadable'), the tile
    shows 'NoneType', and Dashboard._check_for_trouble (which only reads nan/inf flags
    for non-scalars) never sees a NaN: the 'loss is NaN' critical auto-intervention
    cannot fire. Correct: the NaN reaches the manifest and the detector flags it."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    q = std_queue.Queue()
    for v in (1.0, 0.9, 0.8, float("nan")):
        q.put(("loss", np.float64(v), None))
    q.put(None)
    core._worker_main(q, {}, "s3", {})
    with open(os.path.join(str(tmp_path), "pulse_cache", "s3", "manifest.json")) as f:
        man = json.load(f)
    dash = core.Dashboard.__new__(core.Dashboard)
    dash.explosion_multiplier = None
    problem = dash._check_for_trouble(man)
    assert man["loss"]["latest_value"] is not None
    assert problem and "nan" in problem.lower(), problem


def test_bug_ui_restart_crashes_training_when_main_has_no_file():
    """UI-mode RESTART handler: script_path = os.path.abspath(getattr(__main__,
    '__file__', None)). In a notebook / `python -c` / embedded interpreter __main__
    has no __file__, abspath(None) raises TypeError INSIDE the trace function, and
    that exception is raised into the user's training loop at whatever line it was
    on -- applying an agent fix kills the run. Correct: Pulse reports it cannot
    restart and training keeps running."""
    code = f"""
import sys, os, time
sys.path.insert(0, {SRC!r})
os.environ["PULSE_LOGGING"] = "0"
import numpy as np
import pulse.pulse as core

class FakeBG:
    def __init__(self, *a, **k): self.queue = None
    def log_matrix(self, *a, **k): return True
class FakeProc:
    def __init__(self, *a, **k): pass
    def start(self): pass
class OnceQ:
    sent = False
    def __init__(self, *a, **k): pass
    def get_nowait(self):
        if not OnceQ.sent:
            OnceQ.sent = True
            return ("RESTART", "FakeProv")
        raise Exception("empty")
    def get(self, timeout=None): raise Exception("empty")
    def put(self, x): pass
class Dlg:
    def run(self): return ("FakeProv", "k")
core.HeatmapCreatorBG = FakeBG
core.mp.Process = FakeProc
core.mp.Queue = OnceQ
core.AgentSetupDialog = Dlg
core.PROVIDERS["FakeProv"] = {{"model": "x/y", "env_key": None}}
core._install_pulse_excepthook = lambda sid: None

def main():
    w = np.zeros((2, 2))
    core.auto_track(mode="ui", throttle_interval=0.05)
    try:
        for step in range(50):
            w = w + 1
            time.sleep(0.002)
        print("TRAINING_FINISHED")
    except Exception as exc:
        print("TRAINING_KILLED", type(exc).__name__, exc)
    finally:
        sys.settrace(None)
main()
"""
    env = dict(os.environ, PYTHONPATH=SRC, PULSE_LOGGING="0", CUDA_VISIBLE_DEVICES="")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=90, env=env, cwd=td)
    out = r.stdout
    assert "TRAINING_FINISHED" in out, out[-500:] + r.stderr[-500:]


def test_bug_safe_eval_math_is_not_sandboxed():
    """_safe_eval_math (the agent's CALC: directive) claims 'no builtins, so this is
    safe to eval() directly', but attribute access is unrestricted: the classic
    ().__class__.__mro__[1].__subclasses__() walk reaches every loaded class (and
    from there os/subprocess). Agent output is untrusted (it echoes training data /
    code comments). Correct: dunder attribute access is refused (e.g. parse with ast
    and allow only numbers, operators and math names)."""
    res = core._safe_eval_math("().__class__.__mro__[1].__subclasses__()")
    assert isinstance(res, str) and "error" in res.lower(), type(res)


def test_bug_safe_eval_math_unbounded_power_hangs():
    """CALC: 9**9**9 (a 370-million-digit integer) runs for minutes and eats memory
    in the Dashboard's agent thread -- nothing bounds exponent size. Correct: CALC
    returns promptly with an error for absurd results."""
    code = f"""
    import sys, os
    sys.path.insert(0, {SRC!r})
    import pulse.pulse as core
    print("RESULT", str(core._safe_eval_math("9**9**9"))[:40])
    """
    try:
        r = run_py(code, timeout=20)
        ok = "RESULT" in r.stdout
    except subprocess.TimeoutExpired:
        ok = False
    assert ok, "CALC: 9**9**9 did not return within 20s"


def test_bug_rank_status_reports_ranks_from_previous_job():
    """Rank status files live in <tmp>/pulse_cache/ranks/<session_key>, never cleaned
    and never tied to the current job. torchrun's default TORCHELASTIC_RUN_ID is
    'none', so every non-rdzv torchrun job on the machine shares one directory:
    after a 4-GPU job, a 2-GPU job's GPUSTATUS reports '4/2 rank(s) reporting' and
    the deadlock detector warns about ranks 2 and 3 forever. Correct: only ranks of
    the current job (rank < world_size, this launch) are reported."""
    key = "sweep-test-" + os.urandom(4).hex()
    d = core._rank_status_dir(key)
    try:
        for r in (2, 3):  # left over from an earlier 4-rank job
            with open(os.path.join(d, f"rank_{r}.json"), "w") as f:
                json.dump({"rank": r, "world_size": 4, "pid": 1, "hostname": "h",
                           "updated": 0.0, "gpus": []}, f)
        for r in (0, 1):
            core._write_rank_status(key, r, r, 2)
        report = core._format_multi_rank_gpu_status(key, 0)
        assert report.startswith("GPUSTATUS: 2/2 rank(s) reporting"), report.splitlines()[0]
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_bug_distributed_session_key_not_filesystem_safe(monkeypatch):
    """Without PULSE_SESSION_ID/TORCHELASTIC_RUN_ID (mpirun, srun, torch.distributed.
    launch) the key is 'MASTER_ADDR:MASTER_PORT' and is used verbatim as a directory
    name. ':' is illegal in Windows paths, so os.makedirs raises inside the rank
    ticker thread (outside its try) and multi-rank status silently never works
    there. Correct: the key is sanitised into a portable directory name."""
    for k in ("PULSE_SESSION_ID", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    _, _, ws, key = core._distributed_info()
    assert ws == 2
    assert not any(c in key for c in ':<>"|?*\\/'), key


def test_bug_ui_tracer_never_logs_cpu_mirror():
    """UI-mode persistent_tracer finds an accelerator value's CPU mirror (h -> h_cpu)
    and stores it in observed_val, but the next check tests `val` (the accelerator
    value) again -- `if PULSE_CPU_ONLY and not _cpu_resident(val): continue` -- so
    the documented mirror path can never log anything. Correct: 'h' is logged using
    its CPU mirror."""
    code = f"""
    SRC = {SRC!r}
    import sys, os, time
    sys.path.insert(0, SRC)
    import numpy as np
    import pulse.pulse as core

    logged = []
    class FakeBG:
        def __init__(self, *a, **k): self.queue = None
        def log_matrix(self, var, m, config_override=None):
            logged.append((var, type(m).__name__)); return True
        def shutdown(self): pass
    class FakeProc:
        def __init__(self, *a, **k): pass
        def start(self): pass
    class FakeQ:
        def __init__(self, *a, **k): pass
        def get_nowait(self): raise Exception("empty")
        def get(self, timeout=None): raise Exception("empty")
        def put(self, x): pass
    class Dlg:
        def run(self): return ("FakeProv", "k")
    core.HeatmapCreatorBG = FakeBG
    core.mp.Process = FakeProc
    core.mp.Queue = FakeQ
    core.AgentSetupDialog = Dlg
    core.PROVIDERS["FakeProv"] = {{"model": "x/y", "env_key": None}}
    core._install_pulse_excepthook = lambda sid: None

    class GpuArray(np.ndarray):   # stands in for a CuPy/CUDA array
        device = "cuda:0"

    def main():
        core.auto_track(mode="ui", throttle_interval=0.05)
        h = np.zeros((4, 4), dtype=np.float32).view(GpuArray)
        h_cpu = np.zeros((4, 4), dtype=np.float32)
        t0 = time.time()
        while time.time() - t0 < 1.5:
            h_cpu = h_cpu + 1
            time.sleep(0.005)
        sys.settrace(None)
    main()
    print("LOGGED", sorted(set(v for v, _ in logged)))
    """
    r = run_py(code)
    line = [l for l in r.stdout.splitlines() if l.startswith("LOGGED")]
    assert line, r.stdout + r.stderr
    assert "h_cpu" in line[0], line[0]           # sanity: tracer ran and logged
    assert "'h'" in line[0], line[0]


def test_bug_ui_tracer_polls_control_queue_on_every_library_call():
    """UI-mode persistent_tracer stays installed for the whole run and, for EVERY
    'call' event (including every library function call a training step makes),
    first drains control_queue with get_nowait() -- an mp.Queue lock + poll()
    syscall -- before it even checks whether the frame is library code. The CLI
    tracer's comments measure this kind of per-call cost as enough to starve the
    GPU. Correct: library calls return before touching the queue."""
    code = f"""
    SRC = {SRC!r}
    import sys, os, time, json
    sys.path.insert(0, SRC)
    import numpy as np
    import pulse.pulse as core

    class FakeBG:
        def __init__(self, *a, **k): self.queue = None
        def log_matrix(self, *a, **k): return True
    class FakeProc:
        def __init__(self, *a, **k): pass
        def start(self): pass
    class CountQ:
        n = 0
        def __init__(self, *a, **k): pass
        def get_nowait(self):
            CountQ.n += 1
            raise Exception("empty")
        def get(self, timeout=None): raise Exception("empty")
        def put(self, x): pass
    class Dlg:
        def run(self): return ("FakeProv", "k")
    core.HeatmapCreatorBG = FakeBG
    core.mp.Process = FakeProc
    core.mp.Queue = CountQ
    core.AgentSetupDialog = Dlg
    core.PROVIDERS["FakeProv"] = {{"model": "x/y", "env_key": None}}
    core._install_pulse_excepthook = lambda sid: None

    def main():
        w = np.zeros((2, 2))
        core.auto_track(mode="ui", throttle_interval=5.0)
        before = CountQ.n
        for _ in range(2000):
            json.dumps({{"a": 1}})      # library (stdlib) calls
        after = CountQ.n
        sys.settrace(None)
        print("POLLS", after - before)
    main()
    """
    r = run_py(code)
    line = [l for l in r.stdout.splitlines() if l.startswith("POLLS")]
    assert line, r.stdout + r.stderr
    polls = int(line[0].split()[1])
    assert polls < 200, f"{polls} control-queue polls for 2000 library calls"


def test_bug_module_level_auto_track_loses_loop_in_main():
    """Common layout: auto_track() at module top, training loop inside main().
    caller_frame is the module frame, so main()'s frame is only traced because it
    was entered during the first window. When that window closes, the first line
    event in main() hits `frame.f_trace_lines = False` and main never gets line
    events again (it never re-enters via a 'call'). Every later window sees
    nothing of the loop. Correct: the loop's variables keep being observed in
    later windows."""
    code = f"""
    SRC = {SRC!r}
    {textwrap.indent(FAKE_CLI, '    ').strip()}
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
    print("LATE", len(late), "TOTAL", len(cli.updates))
    """
    r = run_py(code)
    line = [l for l in r.stdout.splitlines() if l.startswith("LATE")]
    assert line, r.stdout + r.stderr
    late = int(line[0].split()[1])
    assert late >= 2, line[0]


def test_bug_public_shutdown_does_not_stop_cli_tracing():
    """pulse.shutdown() (the only public teardown API) only stops the GUI worker.
    In CLI mode -- the default -- the global tracer and its ticker keep running
    (the ticker re-arms tracing on every window) after shutdown(). Correct:
    shutdown() disarms the CLI tracer for good."""
    code = f"""
    SRC = {SRC!r}
    {textwrap.indent(FAKE_CLI, '    ').strip()}
    core._start_cli_tracker(sys._getframe(), None, 0.05, {{}}, {{}})
    core.shutdown()
    time.sleep(0.3)
    print("TRACE", sys.gettrace() is not None, "STOPPED", core._CLI_TRACING_STOPPED)
    """
    r = run_py(code)
    line = [l for l in r.stdout.splitlines() if l.startswith("TRACE")]
    assert line, r.stdout + r.stderr
    assert line[0] == "TRACE False STOPPED True", line[0]


def test_bug_cli_tracer_left_armed_at_normal_exit():
    """Only the crash hook disarms the CLI tracer. When the script finishes normally
    cli_tracer is still installed into interpreter shutdown (the root cause of the
    "'NoneType' object has no attribute 'get_ident'" reports: it runs on __del__
    calls after module globals are torn down). Correct: Pulse disarms at exit
    (atexit), so user atexit handlers / finalizers run untraced."""
    code = f"""
    SRC = {SRC!r}
    import atexit, sys
    atexit.register(lambda: print("ATEXIT_TRACE", sys.gettrace() is not None, flush=True))
    {textwrap.indent(FAKE_CLI, '    ').strip()}
    core._start_cli_tracker(sys._getframe(), None, 100.0, {{}}, {{}})
    x = 1
    """
    r = run_py(code)
    line = [l for l in r.stdout.splitlines() if l.startswith("ATEXIT_TRACE")]
    assert line, r.stdout + r.stderr
    assert line[0] == "ATEXIT_TRACE False", line[0]


def test_bug_cli_tracer_teardown_guard_misses_underscore_globals():
    """The shutdown guard in cli_tracer only checks `threading is None`. CPython's
    module teardown (_PyModule_ClearDict) sets names starting with '_' to None
    FIRST and only then the rest, running __del__ code in between -- so the tracer
    can be entered with `threading` intact but `_is_library_frame` already None,
    raising TypeError from inside the trace function. Simulated here by clearing
    that one global. Correct: the tracer returns quietly."""
    code = f"""
    SRC = {SRC!r}
    {textwrap.indent(FAKE_CLI, '    ').strip()}
    core._start_cli_tracker(sys._getframe(), None, 100.0, {{}}, {{}})
    tracer = sys.gettrace()
    sys.settrace(None)
    core._is_library_frame = None      # phase 1 of module-dict clearing
    try:
        tracer(sys._getframe(), "line", None)
        print("RESULT ok")
    except Exception as exc:
        print("RESULT raised", type(exc).__name__)
    """
    r = run_py(code)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert line, r.stdout + r.stderr
    assert line[0] == "RESULT ok", line[0]


def test_bug_repeated_auto_track_handles_one_crash_many_times(monkeypatch):
    """Each auto_track() (CLI) call chains a new excepthook onto the previous one.
    Calling it more than once in a process (per fold / per config in a sweep, or a
    notebook cell re-run) means one crash is diagnosed once per call -- N agent
    pipelines, N prompts. Correct: one crash is handled once."""
    class FakeCLI:
        def __init__(self):
            self.crashes = 0
            self.agent_provider = None
            self.auto_intervene = True
        def capture_crash_state(self, frames): pass
        def handle_crash(self, tb):
            self.crashes += 1
            return "sig"
        def offer_known_fix(self, sig): return True
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(core, "_CLI_TRACING_STOPPED", False)
    trace_before = sys.gettrace()
    a, b = FakeCLI(), FakeCLI()
    core._install_cli_excepthook(a)
    core._install_cli_excepthook(b)
    try:
        raise ValueError("boom")
    except ValueError as e:
        sys.excepthook(type(e), e, e.__traceback__)
    sys.settrace(trace_before)
    assert a.crashes + b.crashes == 1, (a.crashes, b.crashes)


def test_bug_shapetrace_leaves_model_in_eval_mode_when_forward_raises():
    """_exec_shapetrace calls model.eval() then the forward; model.train(was_training)
    is inside the same try, so when the forward raises (very common: SHAPETRACE picks
    whatever tensor is in scope, e.g. the labels) the model is left in eval mode and
    training silently continues with dropout off / BatchNorm frozen. Correct: the
    training flag is restored regardless."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Dropout(0.5))
    model.train()
    y = torch.zeros(3, dtype=torch.long)  # labels: wrong shape/dtype for the model
    frame = types.SimpleNamespace(f_globals={}, f_locals={"model": model, "y": y})
    out = core._exec_shapetrace(frame, "model")
    assert "raised" in out
    assert model.training is True


def test_bug_gradcheck_leaves_parameter_perturbed_when_loss_fn_raises():
    """_exec_gradcheck sets flat[idx] = orig + eps, then calls loss_fn(). If loss_fn
    raises, the per-iteration restore never runs; the except's 'restore'
    (flat[idxs[0]] = flat[idxs[0]]) is a no-op and, being outside no_grad, itself
    raises on a leaf parameter. The live model keeps a corrupted weight. Correct:
    the parameter is restored exactly."""
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(3, 1)
    lin(torch.ones(1, 3)).sum().backward()
    before = lin.weight.detach().clone()
    calls = {"n": 0}

    def loss_fn():
        calls["n"] += 1
        raise RuntimeError("loss_fn needs a batch")

    frame = types.SimpleNamespace(f_globals={}, f_locals={"lin": lin, "loss_fn": loss_fn})
    core._handle_exec_request(frame, "GRADCHECK", "weight")
    assert calls["n"] == 1
    assert torch.equal(lin.weight.detach(), before), (lin.weight.detach() - before)


def test_bug_write_code_fix_breaks_indentation_for_unindented_snippet(chat_cls, tmp_path):
    """_write_code_fix replaces `old` with _banner_wrap_fix(old, new) at the exact
    match position. Agents usually quote a line without its leading indentation;
    the match then starts mid-line, so the banner's first line inherits the
    indentation but the 'new' code lands at column 0. For an indented line this is
    an IndentationError (the fix is rejected by the lint gate) or, at the end of a
    block, silently moves the statement out of its loop/function. Correct: the
    replacement keeps the original indentation and the fix is applied."""
    src = "def train():\n    total = 0\n    for i in range(3):\n        total += i\n    return total\n"
    path = tmp_path / "train.py"
    path.write_text(src)
    p = make_panel(chat_cls, script_path=str(path), code=src)
    p._build_file_labels({})
    fix = {"old": ["total += i"], "new": ["total += i * 2"], "files": [None], "explanation": "x"}
    text, applied, _orig, skipped = p._write_code_fix(fix)
    new_src = path.read_text()
    ns = {}
    exec(compile(new_src, str(path), "exec"), ns)
    assert ns["train"]() == 6, new_src




def test_bug_gui_file_labels_cannot_resolve_first_shared_basename(chat_cls, tmp_path):
    """GUI _build_file_labels gives the first of several same-named files the bare
    basename and later ones 'parent/base'. A request for the first one by its path
    ('config/base.py') resolves to nothing; and three files sharing basename AND
    parent name get the same label, so a fix is written to the wrong file. The CLI
    was already fixed for this (tests/test_file_labels.py); the GUI copy was not.
    Correct: every file has its own label and 'config/base.py' resolves to it."""
    files = {}
    for rel in ("config/base.py", "optim/base.py", "a/src/utils.py", "b/src/utils.py", "c/src/utils.py"):
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(f"# {rel}\n")
        files[str(fp)] = f"# {rel}\n"
    script = tmp_path / "train.py"
    script.write_text("x = 1\n")
    p = make_panel(chat_cls, script_path=str(script), code="x = 1\n", extra_files=files)
    p._build_file_labels(files)
    assert len(set(p._label_for_path.values())) == len(p._label_for_path), p._label_for_path
    assert p._resolve_fix_path("config/base.py") == str(tmp_path / "config/base.py")


def test_bug_histogram_directive_crashes_on_nan_history(chat_cls):
    """Scalar histories legitimately contain NaN (a diverged loss -- the exact case
    being debugged). _run_histogram does int((v - lo) / width) -> ValueError on NaN,
    and _apply_new_directives has no per-directive guard, so the exception escapes
    _ask (which only catches AgentRequestFailed) and kills the whole agent pipeline
    thread. Correct: HISTOGRAM reports (skipping/flagging non-finite points)."""
    manifest = {"loss": {"kind": "scalar", "history": [[1, 1.0], [2, 2.0], [3, float("nan")], [4, 3.0]]}}
    p = make_panel(chat_cls, manifest=manifest)
    out = p._apply_new_directives({"histogram": ["loss"]})
    assert "HISTOGRAM 'loss'" in out


def test_bug_corr_pairs_unaligned_points(chat_cls):
    """Scalar histories are deduplicated per variable (a point is stored only when
    the value changes) and tagged with a shared step. _run_corr ignores the steps
    and correlates the last n stored values of each series, pairing values from
    different steps -- here it reports r=+1.0 for series that are negatively
    correlated step-by-step. Correct: align on step (forward-filling repeats)."""
    x = [[1, 6.0], [2, 5.0], [3, 4.0], [4, 1.0], [5, 2.0], [6, 3.0]]
    y = [[1, 0.0], [2, 5.0], [3, 10.0]]  # then flat at 10.0 for steps 4..6
    manifest = {"a": {"kind": "scalar", "history": x}, "b": {"kind": "scalar", "history": y}}
    p = make_panel(chat_cls, manifest=manifest)
    out = p._run_corr("a b")
    r = float(out.rsplit("r =", 1)[1])
    assert r < 0, out


def test_bug_call_model_resends_images_and_context_every_pass(chat_cls, monkeypatch):
    """_call_model's docstring: images go 'on the first call only, so they aren't
    re-uploaded on every stage'. But _ask appends the user turn WITH the images (and
    the full code context) to self.history before pass 1, and every _call_model
    sends history[-10:] -- so every pass (and every later question) re-uploads all
    heatmap images, and pass 1 carries the full context twice. Correct: images are
    sent once."""
    import litellm
    sent = []

    class Resp:
        def __init__(self, text):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
            self.usage = None

    def fake_completion(model, messages, **kw):
        sent.append(messages)
        return Resp("- line 3")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **k: 0.0)
    monkeypatch.setitem(core.PROVIDERS, "FakeProv", {"model": "fake/model", "env_key": None})
    p = make_panel(chat_cls, script_path=None, code=None)
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    p._image_payloads = lambda: [img]
    p._ask("why is the loss flat?", False, "FakeProv")

    def n_images(messages):
        n = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                n += sum(1 for part in c if isinstance(part, dict) and part.get("type") == "image_url")
        return n

    assert len(sent) >= 2
    assert [n_images(m) for m in sent[1:]] == [0] * (len(sent) - 1), [n_images(m) for m in sent]


def test_bug_verify_unparsable_reported_as_passed(chat_cls):
    """_verify_fix_with_retries returns passed=True when the verifier's reply cannot be
    parsed, so the transcript says 'passed: (verification response was unparsable;
    proceeding anyway)'. Its own docstring says a failed verification applies the
    fix as UNVERIFIED best effort. Correct: passed is False for an unparsable
    verdict."""
    p = make_panel(chat_cls)
    p._call_model = lambda *a, **k: "Looks fine to me!"
    fix = {"old": ["a"], "new": ["b"], "files": [None], "explanation": ""}
    _fix, passed, reason = p._verify_fix_with_retries("m", fix, "diag")
    assert passed is False, reason


def test_bug_implement_keyword_substring_match(chat_cls):
    """_ask decides whether to WRITE code with a bare substring test against
    _IMPLEMENT_KEYWORDS ('fix', 'apply', 'edit', ...). 'prefix', 'suffix', 'credit',
    'applying' all match, so a pure question edits the user's files. Correct: a
    question like 'what does the prefix argument do?' does not trigger a code edit."""
    q = "what does the prefix argument do?"
    wants = any(kw in q.lower() for kw in core._IMPLEMENT_KEYWORDS)
    # mirror of the exact expression in ChatPanel._ask (include_code=True)
    assert wants is False


def test_bug_discover_project_files_misses_package_submodules(tmp_path, monkeypatch):
    """_discover_project_files keeps only the top-level name of every import
    (`from mypkg.model import Net` -> 'mypkg') and ignores relative imports, so the
    file that actually defines the model (mypkg/model.py) is never found -- only
    mypkg/__init__.py. Its variables are not candidates and its code is never shown
    to the agent. Correct: mypkg/model.py is discovered."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "model.py").write_text("def forward(x):\n    scores = x\n    return scores\n")
    entry = tmp_path / "train.py"
    entry.write_text("from mypkg.model import forward\nimport mypkg.model\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    found = [os.path.normcase(os.path.abspath(f)) for f in core._discover_project_files(str(entry))]
    assert os.path.normcase(str(pkg / "model.py")) in found, found


def test_bug_project_root_prefix_match_leaks_sibling_dirs(tmp_path, monkeypatch):
    """project_root is compared with a bare str.startswith, so root '/x/proj' also
    admits '/x/proj_old/...' (and in the tracers, a sibling checkout's frames).
    Correct: containment is checked on a path-component boundary."""
    proj = tmp_path / "proj"
    sib = tmp_path / "proj_old"
    proj.mkdir()
    sib.mkdir()
    (sib / "legacy_helpers.py").write_text("y = 1\n")
    entry = proj / "train.py"
    entry.write_text("import legacy_helpers\n")
    monkeypatch.syspath_prepend(str(sib))
    root = os.path.normcase(os.path.abspath(str(proj)))
    found = core._discover_project_files(str(entry), root)
    assert found == [], found


def test_bug_static_discovery_misses_starred_walrus_async_targets(tmp_path):
    """discover_static_names_from_file collects Name/Tuple/List targets only: starred
    unpacking (`head, *rest = ...`), walrus (`(n := ...)`), `async for`/`async
    with` targets are never offered as candidates. Correct: all assignment targets
    are discovered."""
    f = tmp_path / "m.py"
    f.write_text(textwrap.dedent("""
        head, *rest_items = [1, 2, 3]
        if (grad_norm := 3.0) > 1:
            pass
        async def run(loader):
            async for async_batch in loader:
                pass
            async with loader as async_ctx:
                pass
    """))
    names = core.discover_static_names_from_file(str(f))
    missing = {"rest_items", "grad_norm", "async_batch", "async_ctx"} - names
    assert not missing, missing


def test_bug_determine_mode_is_case_sensitive(monkeypatch):
    """_stream_mode_requested lower-cases the mode, _determine_mode does not:
    auto_track(mode='UI') or mode='CLI' is ignored and falls through to
    auto-detection (on a headless box 'UI' silently becomes cli; with a display
    'CLI' opens the GUI). Correct: modes are case-insensitive like 'stream'."""
    monkeypatch.delenv("PULSE_MODE", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert core._determine_mode("UI") == "ui"


def test_bug_auto_track_docstring_default_mode_mismatch():
    """auto_track's docstring says mode 'auto' is the default ("auto" (default --
    detects a real display...)), but the signature default is mode='cli', so a
    desktop user following the docstring never gets the dashboard. Correct: the
    documented default and the real default agree."""
    import inspect
    default = inspect.signature(core.auto_track).parameters["mode"].default
    doc = core.auto_track.__doc__
    documented = "auto" if '"auto" (default' in doc else default
    assert default == documented


# ===========================================================================
# OK -- regression coverage for behaviour verified to work
# ===========================================================================

def test_ok_distributed_info_defaults(monkeypatch):
    for k in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
              "SLURM_LOCALID", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        monkeypatch.delenv(k, raising=False)
    assert core._distributed_info() == (0, 0, 1, None)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("PULSE_SESSION_ID", "abc")
    assert core._distributed_info() == (3, 3, 4, "abc")


def test_ok_values_equal_nan_and_none():
    assert core._values_equal(float("nan"), float("nan"))
    assert core._values_equal(None, None)
    assert not core._values_equal(None, 0.0)
    assert not core._values_equal(1.0, 2.0)


def test_ok_downsample_bounds_and_filters():
    pts = [(i, float(i)) for i in range(1000)]
    out = core._downsample_for_display(pts)
    assert len(out) <= core.TARGET_SCALAR_DISPLAY_POINTS + 1
    out2 = core._downsample_for_display([(1, 1.0), (2, None), (3, float("nan")), (4, 2.0)])
    assert out2 == [(1, 1.0), (4, 2.0)]


def test_ok_reduce_to_2d_iterate_axis():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    red = core._reduce_to_2d(arr, [3, 1, 1])  # iterate axis0 at index 1
    assert red.shape == (3, 4)
    assert np.array_equal(red, arr[1])
    assert core._reduce_to_2d(np.float32(2.0), None).shape == (1, 1)


def test_ok_is_library_frame():
    import json as _json
    assert core._is_library_frame(_json.__file__)
    assert core._is_library_frame("<frozen importlib._bootstrap>")
    assert core._is_library_frame("/x/site-packages/torch/nn/modules/module.py")
    assert not core._is_library_frame(os.path.join(tempfile.gettempdir(), "proj", "train.py"))


def test_ok_determine_mode_env(monkeypatch):
    monkeypatch.setenv("PULSE_MODE", "cli")
    assert core._determine_mode("auto") == "cli"
    assert core._determine_mode("ui") == "ui"


def test_ok_stream_mode_requested(monkeypatch):
    monkeypatch.delenv("PULSE_MODE", raising=False)
    assert core._stream_mode_requested(" Stream ")
    assert not core._stream_mode_requested("cli")
    monkeypatch.setenv("PULSE_MODE", "stream")
    assert core._stream_mode_requested("cli")


def test_ok_discover_project_files_follows_local_imports(tmp_path, monkeypatch):
    (tmp_path / "model_sw.py").write_text("import utils_sw\nW = 1\n")
    (tmp_path / "utils_sw.py").write_text("U = 2\n")
    entry = tmp_path / "train.py"
    entry.write_text("import model_sw\nimport json\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    found = sorted(os.path.basename(f) for f in core._discover_project_files(str(entry)))
    assert found == ["model_sw.py", "utils_sw.py"]


def test_ok_discover_candidates_merges_static_and_runtime(tmp_path):
    f = tmp_path / "s.py"
    f.write_text("def g():\n    scores = 1\nweights = 2\n")
    code = compile("import numpy as np\nlive_w = np.zeros((2, 3))\nimport sys\nfr = sys._getframe()\n", str(f), "exec")
    ns = {}
    exec(code, ns)
    frame = ns["fr"]
    disc = core._discover_candidates(frame, None)
    assert "scores" in disc and disc["scores"] is None
    assert disc.get("live_w") == (2, 3)


def test_ok_atomic_write_json_roundtrip(tmp_path):
    p = str(tmp_path / "sub" / "m.json")
    core._atomic_write_json(p, {"a": 1})
    with open(p) as f:
        assert json.load(f) == {"a": 1}
    assert not os.path.exists(p + ".tmp")


def test_ok_worker_scalar_history_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    q = std_queue.Queue()
    for v in (1.0, 1.0, 0.5, float("nan")):
        q.put(("loss", np.float64(v), None))
    q.put(("w", np.ones((3, 3), dtype=np.float32), None))
    q.put(("none_var", None, None))
    q.put(None)
    core._worker_main(q, {}, "s2", {"w": "lotrack"})
    with open(os.path.join(str(tmp_path), "pulse_cache", "s2", "manifest.json")) as f:
        man = json.load(f)
    hist = man["loss"]["history"]
    assert [h[1] for h in hist][:2] == [1.0, 0.5]
    assert man["w"]["state"] == "lotrack" and "image" not in man["w"]
    assert man["none_var"]["error"] == "NoneType"


def test_ok_find_fuzzy_snippet_span_unique_only():
    content = "def f():\n    a = 1\n\n    b = 2\n    return a + b\n"
    span = core._find_fuzzy_snippet_span(content, "a = 1\nb = 2")
    assert span is not None and content[span[0]:span[1]] == "    a = 1\n\n    b = 2\n"
    assert core._find_fuzzy_snippet_span("x = 1\nx = 1\n", "x = 1") is None


def test_ok_fix_history_record_and_revert(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("lr = 1.0\n")
    cid = core._record_fix_history_commit(str(script), {str(script): ("lr = 1.0\n", "lr = 0.1\n")}, "lower lr")
    script.write_text("lr = 0.1\n")
    entries = core._load_fix_history(str(script))
    assert entries[-1]["id"] == cid
    restored, failed, new = core._perform_fix_history_revert(str(script), entries, -1, "start")
    assert restored == [str(script)] and not failed and new
    assert script.read_text() == "lr = 1.0\n"


def test_ok_extract_directives():
    text = "Diag.\nCALC: 2*3\nPROMOTE: w1, w2\nGREP: foo\nVIEW: model.py:1-5\nDEFOF: Net\nGPUSTATUS:\n"
    cleaned, calc, prom, grep, view = core._extract_directives(text)
    assert calc == ["2*3"] and prom == ["w1", "w2"] and grep == ["foo"] and view == ["model.py:1-5"]
    cleaned2, reqs = core._extract_new_directives(cleaned)
    assert reqs["defof"] == ["Net"] and "gpustatus" in reqs
    assert cleaned2 == "Diag."
    assert core._safe_eval_math("sqrt(16) + 1") == 5.0


def test_ok_parse_code_fix_shapes(chat_cls):
    P = chat_cls._parse_code_fix
    assert P('```json\n{"old": ["a"], "new": ["b"], "explanation": "e"}\n```')["files"] == [None]
    flat = P('{"old": ["x = 1", "y = 2"], "new": ["x = 3"], "files": "m.py"}')
    assert flat["old"] == ["x = 1\ny = 2"] and flat["files"] == ["m.py"]
    assert P('{"old": ["a"], "new": ["b"], "files": ["x", "y"]}') is None
    assert P("not json") is None


def test_ok_write_code_fix_exact_indented_snippet(chat_cls, tmp_path):
    src = "def f():\n    lr = 1.0\n    return lr\n"
    path = tmp_path / "train.py"
    path.write_text(src)
    p = make_panel(chat_cls, script_path=str(path), code=src)
    p._build_file_labels({})
    fix = {"old": ["    lr = 1.0"], "new": ["    lr = 0.1"], "files": [None], "explanation": "lower"}
    _t, applied, originals, skipped = p._write_code_fix(fix)
    assert applied and not skipped and originals[str(path)] == src
    ns = {}
    exec(path.read_text(), ns)
    assert ns["f"]() == 0.1


def test_ok_view_and_grep(chat_cls, tmp_path):
    src = "a = 1\nb = 2\nc = 3\n"
    path = tmp_path / "train.py"
    path.write_text(src)
    p = make_panel(chat_cls, script_path=str(path), code=src)
    assert "2 | b = 2" in p._run_view("2-2")
    assert "train.py, line 3" in p._run_grep("c =")


def test_ok_format_multi_rank_status_current_job():
    key = "sweep-ok-" + os.urandom(4).hex()
    d = core._rank_status_dir(key)
    try:
        core._write_rank_status(key, 0, 0, 2, extra={"scalars": {"loss": 0.5}})
        core._write_rank_status(key, 1, 1, 2, extra={"error": "Traceback\nValueError: x"})
        out = core._format_multi_rank_gpu_status(key, 0)
        assert out.startswith("GPUSTATUS: 2/2 rank(s) reporting")
        assert "loss=0.5" in out and "crashed" in out
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_ok_layerstats_and_replay_roundtrip():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 4)

    def train_step():
        opt.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        return loss.item()

    train_step()
    frame = types.SimpleNamespace(f_globals={}, f_locals={"model": model, "opt": opt, "train_step": train_step})
    assert "LAYERSTATS" in core._exec_layerstats(frame, "model")
    old = list(core._REPLAY_CHECKPOINTS)
    core._REPLAY_CHECKPOINTS.clear()
    try:
        core._replay_maybe_checkpoint(frame, 0)
        live = model.weight.detach().clone()
        out = core._exec_replay(frame, "2")
        assert "replayed 2 step(s)" in out
        assert torch.equal(model.weight.detach(), live)
    finally:
        core._REPLAY_CHECKPOINTS[:] = old


def test_ok_auto_track_inside_main_keeps_observing_loop():
    """Control for test_bug_module_level_auto_track_loses_loop_in_main: when the
    tracker is started from the frame that runs the loop, later windows do see it."""
    code = f"""
    SRC = {SRC!r}
    {textwrap.indent(FAKE_CLI, '    ').strip()}

    def main():
        core._start_cli_tracker(sys._getframe(), None, 0.1, {{}}, {{}})
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
    print("LATE", len(late), "TOTAL", len(cli.updates))
    """
    r = run_py(code)
    line = [l for l in r.stdout.splitlines() if l.startswith("LATE")]
    assert line, r.stdout + r.stderr
    assert int(line[0].split()[1]) >= 2, line[0]
