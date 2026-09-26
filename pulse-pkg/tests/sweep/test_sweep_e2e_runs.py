"""End-to-end sweep: real training scripts run under Pulse the way a user runs them.

Every test starts a real subprocess -- `python train.py` with `auto_track()` in it, or
`python -m pulse run [--stream] train.py` -- with Pulse's agent pointed at a FAKE model
(tests/sweep/e2e_fake_llm/sitecustomize.py patches litellm.completion; nothing goes to the
network), HOME in a temp dir, non-interactive, auto-fix on. See e2e_harness.run.

test_bug_* assert the CORRECT behaviour and fail on the current code; test_ok_* cover
behaviour verified to work. Each run takes ~10-20 s (importing Pulse alone is ~6 s), so run
them on a machine with a few cores to spare.

Run:  PYTHONPATH=src CUDA_VISIBLE_DEVICES= python -m pytest -q tests/sweep/test_sweep_e2e_runs.py
"""
import glob
import os
import re
import signal
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e_harness as h          # noqa: E402

AT = "from pulse import auto_track\nauto_track()\n"
NO_TRAINING = "without training a single step"


def np_loop(steps, sleep, nan_after=None):
    """A real (tiny) gradient-descent loop in plain NumPy; `loss` is a Python float."""
    lines = [
        "import time",
        "import numpy as np",
        "rng = np.random.default_rng(0)",
        "X = rng.normal(size=(256, 8)); y = X @ rng.normal(size=(8, 1))",
        "w = np.zeros((8, 1))",
        "t0 = time.perf_counter()",
        f"for step in range({steps}):",
        "    err = X @ w - y",
        "    loss = float((err ** 2).mean())",
        "    w -= 0.05 * (X.T @ err) * 2 / 256",
    ]
    if nan_after is not None:
        lines += [f"    if step > {nan_after}:", "        loss = float('nan')"]
    lines += [
        f"    time.sleep({sleep})",
        'print("TRAIN_SECONDS", time.perf_counter() - t0)',
        'print("TRAINED", loss, flush=True)',
    ]
    return "\n".join(lines) + "\n"


TORCH_PRE = """\
import time
import torch, torch.nn as nn
torch.manual_seed(0)
torch.set_num_threads(2)
X = torch.randn(512, 10); w_true = torch.randn(10, 1); Y = X @ w_true + 0.01 * torch.randn(512, 1)
"""


def escalated_at_exit(r):
    return NO_TRAINING in r.stdout or bool(r.calls_about(NO_TRAINING))


def pulse_log(r):
    p = os.path.join(r.workdir, "pulse.log")
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def flagged(r):
    """Did Pulse raise a training problem during the run (not the end-of-run review)?"""
    return "Auto-intervention" in r.stdout and not escalated_at_exit(r)


# =====================================================================================
# BUGS
# =====================================================================================

def _several(n, **kw):
    """Run the same script n times concurrently (the hang below is a race, so one clean
    exit proves little)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(lambda _: h.run(**kw), range(n)))


ARGPARSE_SCRIPT = """
import argparse
from pulse import auto_track
auto_track()
p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=0.1)
args = p.parse_args()
print("LR", args.lr)
"""


def test_bug_short_script_hangs_forever_at_exit():
    """A script that finishes soon after auto_track() -- `python train.py --help` on an
    argparse script, an early sys.exit, a config check -- can hang forever at exit
    (reproduced repeatedly; a race, so several copies are run).

    auto_track() starts the start-of-run model call on a daemon thread
    (_BackgroundModelCall, pulse_cli.py ~1479 -> _call_model -> _clamp_output_tokens ->
    litellm.get_model_info / litellm.completion, which lazily import litellm submodules).
    When the script ends first, interpreter shutdown freezes that daemon thread while it
    holds an import lock; litellm's AsyncHTTPHandler.__del__ then imports on the main thread
    and waits on that lock forever. Stacks at exit (dumped with an atexit hook):
      daemon thread: pulse_cli.py:11293 start-of-run lambda -> _call_model:6724 ->
        _clamp_output_tokens:1658 -> litellm get_model_info -> get_llm_provider ->
        `from litellm.llms.openai_like.dynamic_config import ...` (in importlib)
      main thread: litellm/llms/custom_httpx/http_handler.py:1097 __del__ -> :1073
        _dispose_wrapped_aiohttp_session -> importlib _get_module_lock  (blocked)
    Correct: the process exits promptly with the script's exit code -- e.g. wait
    (bounded) for the background call before interpreter teardown, or warm litellm's
    imports on the main thread before starting it.
    """
    runs = _several(6, source=ARGPARSE_SCRIPT, args=["--help"], timeout=60, installed=True)
    hung = [r for r in runs if r.timed_out]
    assert not hung, f"{len(hung)}/6 runs of `python train.py --help` never exited:\n" + hung[0].show(1500)
    assert all(r.returncode == 0 and "--lr" in r.stdout for r in runs)


def test_bug_short_script_hangs_forever_at_exit_under_pulse_run():
    """Same hang through `pulse run`: a script with nothing to do after setup (it just
    prints its arguments) never exits. Measured on the PC: 5/6 runs hung with Pulse run from
    a checkout (PYTHONPATH=src / `pip install -e .`, where the tracer also traces Pulse's own
    frames and slows the background thread), 0/6 with the site-packages layout -- a race.
    Correct: exits 0 within seconds of printing."""
    runs = _several(6, source="import sys, json\nprint('ARGV', json.dumps(sys.argv[1:]))\n",
                    mode="run", args=["--epochs", "3", "a b"], timeout=60, installed=False)
    hung = [r for r in runs if r.timed_out]
    assert not hung, f"{len(hung)}/6 `pulse run` runs never exited:\n" + hung[0].show(1500)
    assert all('ARGV ["--epochs", "3", "a b"]' in r.stdout and r.returncode == 0 for r in runs)


def test_bug_nan_loss_is_never_detected_in_cli_mode():
    """The loss turns NaN at step ~150 and stays NaN for ~5 s; Pulse (default cli mode)
    never flags it -- it prints "'loss' could not be read this step: TypeError: float()
    argument must be ... not 'NoneType'" and records the point as None.

    pulse_cli.py PulseCLI.update (~line 11957): `scalar_val = float(stats.get("mean"))`, but
    statistics() reports a NaN/inf scalar as mean=None with nan=1 / inf=1, so the float()
    raises, the reading is stored as None ("unreadable") and the nonfinite detector never
    sees a NaN. Correct: a NaN/inf scalar is recorded as NaN/inf and escalated as a
    critical problem. (Same for a torch tensor loss -- see the torch variant.)
    """
    src = AT + np_loop(600, 0.01, nan_after=150)
    r = h.run(src, installed=True)
    assert r.returncode == 0, r.show()
    assert "TRAINED nan" in r.stdout
    assert flagged(r), "a loss that went NaN for 450 steps was never flagged:\n" + r.show(3000)


def test_bug_nan_torch_loss_is_never_detected_in_cli_mode():
    """PyTorch version of the NaN blind spot: loss.backward() on a NaN loss for ~4 s of
    training and no finding. Correct: flagged as non-finite."""
    pytest.importorskip("torch")
    src = TORCH_PRE + AT + """
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for step in range(600):
    opt.zero_grad()
    loss = ((model(X) - Y) ** 2).mean()
    if step > 150:
        loss = loss * float("nan")
    loss.backward(); opt.step()
    time.sleep(0.01)
print("FINAL", loss.item())
"""
    r = h.run(src, installed=True)
    assert r.returncode == 0, r.show()
    assert "FINAL nan" in r.stdout
    assert flagged(r), "a NaN torch loss was never flagged:\n" + r.show(3000)


def test_ok_healthy_fast_numpy_run_is_not_escalated_at_exit():
    """Regression for the removed end-of-run review, which escalated a run that trained
    perfectly well.

    300 real gradient steps that finish in well under a second leave PulseCLI.step at 0
    (update() is throttled to once per throttle_interval, and the step counter only moves
    inside update()), so the review decided "The script finished without training a single
    step ... an error may have been caught" and ran the full diagnose-and-fix pipeline
    (6 more model calls, 'Fix the bug' prompts) on a correct script. The review is gone:
    nothing escalates at exit.
    """
    r = h.run(AT + np_loop(300, 0), installed=True)
    assert r.returncode == 0, r.show()
    assert not escalated_at_exit(r), "healthy run escalated as 'trained nothing':\n" + r.show(3000)


def test_ok_healthy_fast_run_under_pulse_run_is_not_escalated_at_exit():
    """Same removed false positive through `pulse run` (no auto_track in the file): the
    healthy run got the 'crashed invisibly' escalation and 6 extra agent calls."""
    r = h.run(np_loop(300, 0), mode="run", installed=True)
    assert r.returncode == 0, r.show()
    assert not escalated_at_exit(r), r.show(3000)
    assert len(r.llm_calls) <= 1, f"{len(r.llm_calls)} agent calls for a healthy run"


def test_ok_evaluation_only_script_is_not_escalated_at_exit():
    """An evaluation/inference script (model.eval(), no_grad, computes a test loss) is a
    legitimate script that trains nothing by design; the removed end-of-run review treated
    it as a failed training run and asked the agent to find and fix 'the bug'. Not
    escalated (with auto-fix on, a confident fake 'fix' would be written into the file)."""
    pytest.importorskip("torch")
    src = TORCH_PRE + AT + """
model = nn.Linear(10, 1)
model.eval()
total = 0.0
with torch.no_grad():
    for i in range(0, 512, 64):
        loss = ((model(X[i:i+64]) - Y[i:i+64]) ** 2).mean()
        total += loss.item()
print("EVAL", total / 8)
"""
    r = h.run(src, installed=True)
    assert r.returncode == 0, r.show()
    assert not escalated_at_exit(r), r.show(3000)


def test_ok_sklearn_script_that_reports_a_loss_is_not_escalated_at_exit():
    """A scikit-learn fit that reports its test loss (`loss = mean_squared_error(...)`) was
    escalated at exit by the removed review as 'trained nothing, maybe a swallowed error'.
    No escalation -- a non-iterative fit is still training."""
    pytest.importorskip("sklearn")
    src = """
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
""" + AT + """
rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 5)); y = X @ rng.normal(size=5)
model = Ridge(alpha=1.0).fit(X[:800], y[:800])
loss = mean_squared_error(y[800:], model.predict(X[800:]))
print("TEST LOSS", loss)
"""
    r = h.run(src, mode="run", installed=True)
    assert r.returncode == 0, r.show()
    assert not escalated_at_exit(r), r.show(3000)


def test_ok_healthy_torch_dataloader_training_is_not_escalated_at_exit():
    """8 epochs of a normal DataLoader loop (loss 2.57 -> 0.015, ~1 s of training) ended with
    the removed review's 'finished without training a single step' escalation and 6 extra
    agent calls. Not escalated."""
    pytest.importorskip("torch")
    src = TORCH_PRE + AT + """
from torch.utils.data import TensorDataset, DataLoader
dl = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(8):
    for xb, yb in dl:
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
    print("epoch", epoch, "loss", loss.item())
"""
    r = h.run(src, installed=True)
    assert r.returncode == 0, r.show()
    assert "epoch 7 loss" in r.stdout
    assert not escalated_at_exit(r), r.show(3000)


def test_ok_exit_work_is_not_traced_when_pulse_is_not_in_site_packages():
    """Regression for the removed end-of-run review. Run from a checkout / editable install
    (PYTHONPATH=src, `pip install -e .`), Pulse's own line tracer was still installed when
    the atexit review ran, and traced Pulse's own code: the review's locals ('self', 'call',
    'deadline') became tracked variables and cli.update() ran inside the review ("END-OF-RUN
    REVIEW failed: RuntimeError: dictionary changed size during iteration"). The review is
    gone, and Pulse's remaining exit work removes the tracer first."""
    r = h.run(AT + np_loop(300, 0), installed=False)
    assert r.returncode == 0, r.show()
    log = pulse_log(r)
    assert "END-OF-RUN REVIEW failed" not in log, re.findall(r".*END-OF-RUN REVIEW failed.*", log)
    tracked = re.findall(r"tracked=\[([^\]]*)\]", log)
    assert not any("'deadline'" in t for t in tracked), "Pulse tracked its own local 'deadline'"


def test_bug_importing_pulse_imports_tensorflow_and_torch():
    """`from pulse import auto_track` in a NumPy-only script imports TensorFlow AND PyTorch
    (pulse_backend.py:58-90 imports every optional backend eagerly at module import).
    pulse_cli.py's _KerasImportWatcher / _build_keras_tracker_class docstrings say this was
    removed on purpose ("merely importing Pulse pay[s] for a full TensorFlow import --
    seconds, and GPU memory on some builds"). Measured: `import pulse` 5.8 s, of which the
    backends ~2.6 s; a 3.8 s training script takes 12 s under Pulse. Correct: frameworks the
    script did not import are not imported."""
    code = ("import sys, time; t=time.time(); import numpy; from pulse import auto_track; "
            "print('TF', 'tensorflow' in sys.modules, 'TORCH', 'torch' in sys.modules, "
            "'SECONDS', round(time.time()-t, 2))")
    env = dict(os.environ, PYTHONPATH=h.SRC, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=120)
    line = [l for l in out.stdout.splitlines() if l.startswith("TF ")]
    assert line, out.stderr[-2000:]
    assert "TF False" in line[0], "importing Pulse imported tensorflow: " + line[0]
    assert "TORCH False" in line[0], "importing Pulse imported torch: " + line[0]


def test_bug_pulse_run_injects_auto_track_after_the_training_loop():
    """`pulse run` inserts auto_track() "after the last top-level import" (cli.py
    _find_import_insert_idx). A script that imports something late -- `import json` /
    `import matplotlib.pyplot as plt` to save results after training -- gets auto_track()
    AFTER its training loop: nothing is tracked, and the exit review then escalates the run
    as 'trained nothing'. Correct: insert after the leading block of imports (before the
    first non-import statement), never after executable code."""
    src = "import numpy as np\n" + np_loop(300, 0.01) + "import json\nprint(json.dumps({'done': 1}))\n"
    r = h.run(src, mode="run", installed=True)
    assert r.returncode == 0, r.show()
    m = re.search(r"AUTO_TRACK start .*caller=\S+train\.py:(\d+)", pulse_log(r))
    assert m, pulse_log(r)[-2000:]
    loop_line = next(i for i, l in enumerate(src.splitlines(), 1) if l.startswith("for step"))
    assert int(m.group(1)) < loop_line, (
        f"auto_track() was injected at line {m.group(1)}, after the training loop (line {loop_line})")
    assert not escalated_at_exit(r)


def test_bug_stream_mode_sees_nothing_in_a_directory_named_pulse():
    """--stream records no variables at all when the project lives under a directory named
    `pulse` (e.g. ~/code/pulse/train.py): pulse_monitor._SKIP_PATH_MARKERS contains
    os.sep+'pulse'+os.sep, meant to skip Pulse's own frames, so every frame of the user's
    script is treated as library code (pulse_monitor.py ~486 _is_user_frame). The same
    script under work/proj streams 16 loss readings. Correct: only Pulse's own package
    directory is skipped."""
    from pulse.pulse_brain import Brain
    r = h.run(np_loop(400, 0.01), mode="stream", subdir="pulse/proj", installed=True)
    assert r.returncode == 0, r.show()
    sessions = sorted(glob.glob(os.path.join(r.workdir, "pulse", "proj", ".pulse_stream", "*")))
    assert sessions, r.show()
    brain = Brain(sessions[-1])
    brain.poll_once()
    assert len(brain.histories.get("loss", [])) > 3, (
        f"no loss readings streamed: {dict((k, len(v)) for k, v in brain.histories.items())}")


def test_bug_spawned_worker_processes_each_start_their_own_pulse():
    """auto_track() at the top of the script, as the README shows, plus a 'spawn' pool
    (the default start method on macOS/Windows, and for DataLoader workers there): every
    worker re-imports the main module and starts a full Pulse CLI of its own -- banner,
    agent setup, its own start-of-run model call (billed), its own tracer. Correct:
    auto_track() is a no-op in a multiprocessing child (multiprocessing.parent_process()
    is not None)."""
    r = h.run("""
        import multiprocessing as mp
        from pulse import auto_track
        auto_track()
        def sq(x):
            return x * x
        if __name__ == "__main__":
            with mp.get_context("spawn").Pool(2) as p:
                print("POOL", sum(p.map(sq, range(10))))
        """, timeout=200, installed=True)
    assert r.returncode == 0, r.show()
    assert "POOL 285" in r.stdout
    assert r.stdout.count("Pulse is ready") == 1, f"{r.stdout.count('Pulse is ready')} Pulse sessions started"
    pids = {c["pid"] for c in r.llm_calls}
    assert len(pids) <= 1, f"model calls made from {len(pids)} processes"


def test_bug_ctrl_c_is_swallowed_outside_the_training_loop():
    """Ctrl+C while the script is outside the training loop -- here a sleep after training;
    in real life a long evaluation, a download, a checkpoint write -- is lost completely.
    PulseCLI._sigint_handler (pulse_cli.py ~3285) only latches "pause after the current
    training step" and returns, so the interrupted call resumes (PEP 475); no step follows,
    so the latch is never acted on and the script runs to the end and exits 0 as if it was
    never interrupted (measured: SIGINT at 25 s, 'AFTER SLEEP' printed at 97 s, rc 0).
    Without Pulse the same script stops at 25 s with KeyboardInterrupt (rc -2). Correct:
    Ctrl+C interrupts the program when there is no training step to pause at (and in
    non-interactive mode, where there is nobody to answer the pause prompt, always)."""
    src = AT + np_loop(100, 0.01) + "time.sleep(90)\nprint('AFTER SLEEP')\n"
    r = h.run(src, send_signal_after=(signal.SIGINT, 25), timeout=150, installed=True)
    assert "TRAINED" in r.stdout, r.show()
    assert "AFTER SLEEP" not in r.stdout, "Ctrl+C was ignored; the script ran on to the end"
    assert r.returncode != 0, "an interrupted run reported success (exit 0)"
    assert r.wall < 45, f"took {r.wall:.0f}s to react to Ctrl+C"


def test_ok_ctrl_c_without_pulse_interrupts_immediately():
    """Control for the test above: the harness delivers SIGINT and plain Python stops."""
    src = np_loop(100, 0.01) + "time.sleep(90)\nprint('AFTER SLEEP')\n"
    r = h.run(src, mode="plain", send_signal_after=(signal.SIGINT, 25), timeout=150, installed=True)
    assert "AFTER SLEEP" not in r.stdout and r.returncode != 0 and r.wall < 45, r.show()


def test_ok_ctrl_c_during_the_training_loop_stops_the_run():
    r = h.run(AT + np_loop(6000, 0.02), send_signal_after=(signal.SIGINT, 25), timeout=150,
              installed=True)
    assert r.returncode != 0 and r.wall < 45, r.show()
    assert "TRAINED" not in r.stdout


# =====================================================================================
# Verified working (coverage)
# =====================================================================================

def test_ok_uncaught_exception_keeps_exit_code_and_traceback():
    r = h.run(AT + np_loop(100, 0.01) + "raise ValueError('boom')\n", installed=True)
    assert r.returncode == 1
    assert "ValueError: boom" in r.stderr
    assert not r.modified


@pytest.mark.parametrize("mode", ["direct", "run", "stream"])
def test_ok_sys_exit_code_is_preserved(mode):
    src = (AT if mode == "direct" else "") + np_loop(200, 0.005) + "import sys\nsys.exit(3)\n"
    r = h.run(src, mode=mode, installed=True)
    assert not r.timed_out, r.show()
    assert r.returncode == 3, r.show()
    assert "TRAINED" in r.stdout


def test_ok_stream_mode_crash_keeps_exit_code_and_traceback():
    r = h.run(np_loop(100, 0.01) + "raise ValueError('boom')\n", mode="stream", installed=True)
    assert r.returncode == 1
    assert "ValueError: boom" in r.stderr


def test_ok_caught_and_printed_error_makes_no_model_call_at_exit():
    """Regression for the removed end-of-run review. It escalated a script that catches and
    prints its training error and exits 0 -- from atexit, after concurrent.futures had shut
    down, so every model call it made failed with "cannot schedule new futures after
    interpreter shutdown" after the request was sent and billed. The review is gone: no
    escalation, and no model call at exit that could fail that way."""
    pytest.importorskip("torch")
    src = TORCH_PRE + AT + """
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
try:
    for step in range(100):
        opt.zero_grad()
        loss = ((model(X[:, :5]) - Y) ** 2).mean()
        loss.backward(); opt.step()
except Exception as exc:
    print("training failed:", exc)
"""
    r = h.run(src, installed=True)
    assert r.returncode == 0
    assert "training failed:" in r.stdout, r.show()
    assert not escalated_at_exit(r), "escalated at exit:\n" + r.show()
    assert "cannot schedule new futures" not in r.stdout, r.stdout[-1500:]


def test_ok_caught_and_printed_error_is_not_handed_to_the_agent_at_exit():
    """With the model call short-circuited (FAKE_LLM_SHORTCUT=1, no litellm executor), the
    removed review used to hand the caught-and-printed error to the agent. Nothing is."""
    pytest.importorskip("torch")
    src = TORCH_PRE + AT + """
model = nn.Linear(10, 1)
try:
    for step in range(100):
        loss = ((model(X[:, :5]) - Y) ** 2).mean()
except Exception as exc:
    print("training failed:", exc)
"""
    r = h.run(src, installed=True, env={"FAKE_LLM_SHORTCUT": "1"})
    assert r.returncode == 0
    assert not escalated_at_exit(r), r.show()
    assert not r.calls_about("mat1 and mat2 shapes cannot be multiplied"), r.show()


def test_ok_healthy_longer_numpy_run_is_left_alone():
    r = h.run(AT + np_loop(300, 0.01), installed=True)
    assert r.returncode == 0
    assert not escalated_at_exit(r) and not flagged(r), r.show()
    assert not r.modified
    assert [f for f in r.files if not f.startswith("home")] == ["pulse.log", "train.py"]


@pytest.mark.parametrize("mode", ["direct", "run"])
def test_ok_torch_training_in_a_function_is_left_alone(mode):
    pytest.importorskip("torch")
    src = TORCH_PRE + """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 32); self.b = nn.Linear(32, 1)
    def forward(self, x):
        return self.b(torch.relu(self.a(x)))

def train():
    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=0.02)
    for step in range(600):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    print("FINAL", loss.item())

if __name__ == "__main__":
""" + ("    from pulse import auto_track\n    auto_track()\n" if mode == "direct" else "") + "    train()\n"
    r = h.run(src, mode=mode, installed=True)
    assert r.returncode == 0, r.show()
    assert "FINAL" in r.stdout
    assert not flagged(r) and not r.modified, r.show()


def test_ok_multiprocessing_pool_and_dataloader_workers_work():
    pytest.importorskip("torch")
    src = TORCH_PRE + """
import multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
def sq(x):
    return x * x
if __name__ == "__main__":
    from pulse import auto_track
    auto_track()
    with mp.Pool(3) as p:
        print("POOL", sum(p.map(sq, range(100))))
    dl = DataLoader(TensorDataset(X, Y), batch_size=64, num_workers=2)
    model = nn.Linear(10, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for epoch in range(20):
        for xb, yb in dl:
            opt.zero_grad(); loss = ((model(xb) - yb) ** 2).mean(); loss.backward(); opt.step()
    print("FINAL", loss.item())
"""
    r = h.run(src, installed=True)
    assert r.returncode == 0, r.show()
    assert "POOL 328350" in r.stdout and "FINAL" in r.stdout
    assert "Traceback" not in r.stderr


def test_ok_script_args_cwd_change_and_spaces_in_path():
    src = """
import os, sys, json, time
os.makedirs("out", exist_ok=True)
os.chdir("out")
loss = 1.0
for step in range(200):
    loss *= 0.99
    time.sleep(0.01)
with open("result.json", "w") as f:
    json.dump({"loss": loss}, f)
print("ARGV", json.dumps(sys.argv[1:]), "CWD", os.path.basename(os.getcwd()))
"""
    r = h.run(src, mode="run", subdir="my project", args=["--lr", "0.1", "two words"], installed=True)
    assert r.returncode == 0, r.show()
    assert 'ARGV ["--lr", "0.1", "two words"] CWD out' in r.stdout
    assert os.path.exists(os.path.join(r.workdir, "out", "result.json"))


def test_ok_python_dash_m_module():
    r = h.run("print('unused')\n", module="pkg.train", extra_files={
        "pkg/__init__.py": "",
        "pkg/train.py": AT + np_loop(200, 0.01)}, installed=True)
    assert r.returncode == 0, r.show()
    assert "TRAINED" in r.stdout


KPRE = """
import numpy as np
import keras
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
rng = np.random.default_rng(0)
X = rng.normal(size=(512, 8)).astype("float32")
y = (X @ rng.normal(size=(8, 1))).astype("float32")
def make():
    m = keras.Sequential([keras.Input((8,)), keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    return m
"""

KERAS_CASES = {
    "validation_data": "h = make().fit(X[:400], y[:400], validation_data=(X[400:], y[400:]), epochs=8, batch_size=32, verbose=2)",
    "validation_split": "h = make().fit(X, y, validation_split=0.2, epochs=8, batch_size=32, verbose=0)",
    "tf_data": "h = make().fit(tf.data.Dataset.from_tensor_slices((X, y)).batch(32), epochs=8, verbose=2)",
    "two_fits": "m = make(); m.fit(X, y, epochs=4, verbose=0); h = m.fit(X, y, epochs=4, verbose=0)",
    "user_callbacks": ("cbs = [keras.callbacks.EarlyStopping(monitor='loss', patience=50)]\n"
                       "h = make().fit(X, y, epochs=8, verbose=0, callbacks=cbs)\n"
                       "assert len(cbs) == 1, cbs"),
    "subclassed_unbuilt": ("class M(keras.Model):\n"
                           "    def __init__(self):\n"
                           "        super().__init__(); self.d1 = keras.layers.Dense(16, activation='relu'); self.d2 = keras.layers.Dense(1)\n"
                           "    def call(self, x):\n"
                           "        return self.d2(self.d1(x))\n"
                           "m = M(); m.compile(optimizer='adam', loss='mse')\n"
                           "h = m.fit(X, y, epochs=8, batch_size=32, verbose=0)"),
}


@pytest.mark.parametrize("case", sorted(KERAS_CASES))
def test_ok_keras_fit_variants_run_clean(case):
    pytest.importorskip("keras")
    src = KPRE + AT + KERAS_CASES[case] + "\nprint('FINAL', h.history['loss'][-1])\n"
    r = h.run(src, installed=True)
    assert r.returncode == 0, r.show()
    assert "FINAL" in r.stdout
    assert not flagged(r) and not escalated_at_exit(r) and not r.modified, r.show()


def test_ok_training_loop_speed_under_pulse():
    """The training loop itself (timed inside the script, so Pulse's ~6 s import is
    excluded) is not slowed down by more than 50% by cli-mode tracing."""
    src_plain = np_loop(4000, 0.0005)
    base = h.run(src_plain, mode="plain", installed=True)
    tracked = h.run(AT + src_plain, installed=True)

    def secs(r):
        return float(re.search(r"TRAIN_SECONDS ([0-9.]+)", r.stdout).group(1))
    assert base.returncode == 0 and tracked.returncode == 0, tracked.show()
    assert secs(tracked) < secs(base) * 1.5 + 0.5, (secs(base), secs(tracked))


def test_bug_clear_screen_escapes_are_written_into_piped_output():
    """With stdout redirected to a file or pipe (`python train.py > train.log`, nohup, a
    cluster job's log), the live dashboard still writes the clear-screen sequence
    ESC[2J ESC[H before every redraw (pulse_cli.py ~12172, unconditional
    sys.stdout.write). safe_print was fixed for exactly this (tests/test_headless.py
    QuietOutputTest: "Importing Pulse must not put escape codes in the program's own
    output"), but this path was missed. Correct: no terminal control sequences when stdout
    is not a TTY."""
    r = h.run(AT + np_loop(300, 0.01), installed=True)
    assert r.returncode == 0, r.show()
    assert "\x1b[2J" not in r.stdout, f"{r.stdout.count(chr(27) + '[2J')} clear-screen codes in piped stdout"


def test_bug_input_gets_an_extra_newline_on_stdout():
    """Importing Pulse replaces builtins.input for the whole process (pulse_cli.py
    safe_input, ~570): every input() call first writes "\\n" to stdout (plus escape codes
    on a TTY). A script that reads its stdin with input() -- a filter, a script answering
    prompts from a pipe -- gets a blank line in its own stdout per call; its output is no
    longer what it printed. Even the 'light' --stream mode does this, because importing
    Pulse imports pulse_cli. Correct: input() behaves as without Pulse for the user's code
    (Pulse's own prompts can add their spacing themselves)."""
    src = "import sys\n" + "a = input()\nb = input()\nprint('GOT', a + b)\n"
    plain = h.run(src, mode="plain", stdin="x\ny\n", installed=True)
    streamed = h.run(src, mode="stream", stdin="x\ny\n", installed=True)
    assert plain.stdout == "GOT xy\n", plain.show()
    own = [l for l in streamed.stdout.splitlines() if not l.startswith("[Pulse]")]
    assert own == ["GOT xy"], f"script output under Pulse: {own!r}"


def test_bug_auto_track_called_twice_starts_a_second_pulse():
    """A train() helper with auto_track() inside, called for two models (or a script whose
    imported module also calls it): the second call starts a whole second Pulse session --
    second banner and setup, second start-of-run model call, a second tracer and a second
    exit review. Correct: auto_track() is idempotent within a process (a later call can
    re-point tracking at the new frame, but must not start another session)."""
    src = """
import time
import numpy as np
from pulse import auto_track
def train(lr):
    auto_track()
    w = 0.0
    for step in range(150):
        loss = (w - 3.0) ** 2
        w -= lr * 2 * (w - 3.0)
        time.sleep(0.01)
    print("TRAINED", lr, round(loss, 6), flush=True)
train(0.1)
train(0.05)
"""
    r = h.run(src, installed=True)
    assert r.returncode == 0 and not r.timed_out, r.show()
    assert "TRAINED 0.1" in r.stdout and "TRAINED 0.05" in r.stdout
    assert r.stdout.count("Pulse is ready") <= 1, "a second Pulse session was started"
