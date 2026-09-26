"""Sweep 2, area e2e: the CURRENT Pulse run end to end, the way a user runs it.

Every test starts real subprocesses -- `python train.py` with auto_track() in it, or
`python -m pulse run [...] train.py` -- through tests/sweep/e2e_harness.py, with the agent
pointed at the FAKE model (tests/sweep/e2e_fake_llm/sitecustomize.py; nothing reaches the
network), HOME in a temp dir, non-interactive, auto-fix on.

test_bug_* assert the CORRECT behaviour and fail on the current code; test_ok_* pin behaviour
verified to work. Scenarios run one at a time (each 10-60 s); run this file on its own:

  PYTHONPATH=src PULSE_SRC=src CUDA_VISIBLE_DEVICES= nice -n 19 \
      python -m pytest -q -p no:cacheprovider tests/sweep2/test_s2_e2e_runs.py
"""
import json
import os
import re
import resource
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP1 = os.path.join(os.path.dirname(HERE), "sweep")
sys.path.insert(0, SWEEP1)
import e2e_harness as h          # noqa: E402

AT = "from pulse import auto_track\nauto_track()\n"


def np_loop(steps, sleep):
    return textwrap.dedent(f"""\
        import time
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.normal(size=(256, 8)); y = X @ rng.normal(size=(8, 1))
        w = np.zeros((8, 1))
        for step in range({steps}):
            err = X @ w - y
            loss = float((err ** 2).mean())
            w -= 0.05 * (X.T @ err) * 2 / 256
            time.sleep({sleep})
        print("TRAINED", loss, flush=True)
        """)


def flagged(r):
    return "Auto-intervention" in r.stdout


def fix_rules(fix, locate="- train.py: the suspect line", extra=()):
    """Fake-model answers that walk the crash pipeline to `fix` (see test_sweep_e2e_fix_restart)."""
    return list(extra) + [
        ["PASS 6", json.dumps({"resolved": True, "reason": "the re-run finished and printed its result"})],
        ["PASS 4", json.dumps({"passes": True, "reason": "fixes it"})],
        ["PASS 3", json.dumps(fix)],
        ["PASS 2", "Root cause: see the suspect line. Fix: change it."],
        ["PASS 1", locate],
    ]


# The first sweep's crash-and-fix script: 1.0 / scale with scale = 0.0 after a short loop.
CRASHING = """\
import time
from pulse import auto_track
auto_track()
scale = 0.0
loss = 1.0
for step in range(60):
    loss = loss * 0.9
    time.sleep(0.02)
print("RESULT", 1.0 / scale, flush=True)
print("saved model to model.pt", flush=True)
"""
SCALE_FIX = {"old": ["scale = 0.0"], "new": ["scale = 2.0"], "explanation": "scale was zero"}


def _with_popen(patch):
    """Run h.run with subprocess.Popen wrapped by `patch(real_popen, argv, **kw)`."""
    def runner(*a, **kw):
        real = h.subprocess.Popen
        h.subprocess.Popen = lambda argv, **pkw: patch(real, argv, **pkw)
        try:
            return h.run(*a, **kw)
        finally:
            h.subprocess.Popen = real
    return runner


def run_with_terminal_stdin(*a, **kw):
    """h.run, but the child's stdin is a terminal nobody types into -- a job left running in
    tmux/screen or an ssh session, or a Pulse-restarted child inheriting the user's terminal."""
    import pty
    master, slave = pty.openpty()

    def patch(real, argv, **pkw):
        if pkw.get("stdin") == subprocess.DEVNULL:
            pkw["stdin"] = slave
        return real(argv, **pkw)
    try:
        return _with_popen(patch)(*a, **kw)
    finally:
        os.close(slave)
        os.close(master)


def run_with_terminal_stdout(*a, **kw):
    """h.run with stdout on a pseudo-terminal (a person watching the run). Returns
    (RunResult, [(arrival_time, bytes), ...])."""
    import pty
    master, slave = pty.openpty()
    chunks = []

    def reader():
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:
                return
            if not data:
                return
            chunks.append((time.time(), data))

    def patch(real, argv, **pkw):
        if pkw.get("stdout") == subprocess.PIPE:
            pkw["stdout"] = slave
        p = real(argv, **pkw)
        os.close(slave)
        return p
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    r = _with_popen(patch)(*a, **kw)
    t.join(5)
    try:
        os.close(master)
    except OSError:
        pass
    return r, chunks


# =====================================================================================
# BUGS
# =====================================================================================

FORK_POOL_SEARCH = """
import time
import multiprocessing as mp
from pulse import auto_track
auto_track()

def train_one(lr):
    # a tiny hyper-parameter search: each worker trains one config
    w = 0.0
    for step in range(300):
        loss = (w - 3.0) ** 2
        w -= min(lr, 0.5) * 2 * (w - 3.0)
        if lr > 1:
            loss = float("nan")          # this config diverges
        time.sleep(0.01)
    return lr, loss

if __name__ == "__main__":
    with mp.get_context("fork").Pool(2) as p:
        print("RESULTS", p.map(train_one, [0.1, 5.0]), flush=True)
"""


def test_bug_fork_pool_worker_runs_pulse_fixes_the_script_and_hangs_the_pool():
    """auto_track() at the top (README style) and a hyper-parameter search over a fork Pool
    (Linux's default start method). The spawn fix (auto_track is a no-op in a 'spawn' child)
    does not cover fork: a forked worker inherits the live PulseCLI and its line tracer, so
    the worker whose config diverges runs Pulse's whole auto-intervention on its own --
    9 billed model calls from the worker's pid -- APPLIES A FIX to train.py, re-runs the
    ENTIRE script from inside the pool worker (a second full search, its own 'Pulse is
    ready'), then sys.exit()s the worker. Pool.map in the real parent never gets that
    task's result and waits forever: measured, the run hung until killed at 300 s.
    Without the fix rules the worker still made 9 model calls (a pipeline on a 'no-op').

    Correct: Pulse is inert in a forked child (os.register_at_fork(after_in_child=...):
    drop the tracer, detach the instance); only the parent's Pulse acts, and the pool
    returns. (Whether the parent then flags the NaN it cannot see is a separate matter.)"""
    fix = {"old": ["[0.1, 5.0]"], "new": ["[0.1, 0.2]"], "explanation": "lr 5.0 diverges"}
    r = h.run(FORK_POOL_SEARCH, llm_rules=fix_rules(fix, "- train.py line 20: the lr list"),
              timeout=150, installed=True)
    pids = {c["pid"] for c in r.llm_calls}
    assert not r.timed_out, "the run hung (Pool.map never returned):\n" + r.show(2500)
    assert r.returncode == 0, r.show()
    assert "RESULTS [(0.1," in r.stdout
    assert len(pids) == 1, f"model calls from {len(pids)} processes (a fork worker ran Pulse)"
    assert not r.modified, "a pool worker rewrote train.py"


KERAS_TWO_FIT = """
import numpy as np
import keras
import tensorflow as tf
from pulse import auto_track
auto_track()
tf.config.threading.set_intra_op_parallelism_threads(2)
keras.utils.set_random_seed(0)
FT_BUG = True
rng = np.random.default_rng(0)
X = rng.normal(size=(256, 8)).astype("float32")
y = (X @ rng.normal(size=(8, 1))).astype("float32")
m = keras.Sequential([keras.Input((8,)), keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
m.compile(optimizer=keras.optimizers.SGD(1e-2), loss="mse")
class Probe(keras.callbacks.Callback):
    def __init__(self, stage):
        super().__init__(); self.stage = stage
    def on_epoch_begin(self, epoch, logs=None):
        print(self.stage, "EPOCH_BEGIN", epoch, flush=True)
    def on_train_batch_end(self, batch, logs=None):
        if self.stage == "finetune" and FT_BUG and batch == 2:
            raise RuntimeError("fine-tune data pipeline broke")
h1 = m.fit(X, y, epochs=6, batch_size=32, verbose=0, callbacks=[Probe("pretrain")])
print("PRETRAIN_DONE", len(h1.history["loss"]), flush=True)
h2 = m.fit(X, y, epochs=5, batch_size=32, verbose=0, callbacks=[Probe("finetune")])
print("FINETUNE_EPOCHS_RUN", len(h2.history["loss"]), flush=True)
"""


def test_bug_keras_resume_into_a_second_fit_skips_most_of_its_epochs():
    """Pretrain fit (6 epochs) then fine-tune fit (5 epochs) on the same model; the
    fine-tune crashes at batch 2 of its FIRST epoch. Pulse checkpoints, the scripted fix
    lands, the script restarts, and the resume goes into the right fit() (fit index 2) --
    but with the PRETRAIN's epoch: _keras_epoch is only written in on_epoch_end and never
    reset when a new fit() starts, so the checkpoint says epoch 5 (pretrain's last) and
    _resume_from_fix_checkpoint (pulse_cli.py ~410) sets initial_epoch = min(5+1, 5-1) = 4.
    Measured: "continuing at epoch 5", 'finetune EPOCH_BEGIN 4', FINETUNE_EPOCHS_RUN 1 --
    the fine-tune silently trains 1 of its 5 epochs and the run reports success.

    Correct: a checkpoint taken before the crashing fit() finished any epoch resumes that
    fit at epoch 0 (reset _keras_epoch/_keras_model at each fit() start, e.g. in pulse_fit
    or the tracker's on_train_begin)."""
    pytest.importorskip("keras")
    fix = {"old": ["FT_BUG = True"], "new": ["FT_BUG = False"], "explanation": "fine-tune pipeline bug"}
    r = h.run(KERAS_TWO_FIT, llm_rules=fix_rules(fix), timeout=400, installed=True)
    assert not r.timed_out and r.returncode == 0, r.show()
    assert "FT_BUG = False" in r.script_after, "precondition: fix applied\n" + r.show(4000)
    assert "FINETUNE_EPOCHS_RUN 5" in r.stdout, (
        "the resumed fine-tune did not run its 5 epochs:\n"
        + "\n".join(l for l in r.stdout.splitlines() if "EPOCH" in l or "Resuming" in l or "RUN" in l))


def test_bug_unattended_run_waits_forever_when_the_approver_cannot_answer():
    """Auto mode (`pulse run --approver MODEL`) exists so "a long run never stops to ask".
    When the approver can't answer (down, rate-limited, unreadable reply),
    _confirm_terminal_command (pulse_cli.py ~8130-8147) falls back to input("Run it anyway?
    (y/N)") WITHOUT checking non_interactive -- pulse_approver's own docstring says it falls
    back "to asking the person (or declining, when there is none)". In a non-interactive run
    (PULSE_NONINTERACTIVE=1 / PULSE_CI, and every Pulse-restarted child, which inherits the
    user's terminal as stdin) whose stdin is a terminal nobody is at, the crash pipeline
    blocks on that prompt forever. Measured: stuck at "Run it anyway? (y/N) >" until killed
    at 120 s. (With stdin closed it gets EOF and declines -- see the test_ok below.)

    Correct: in non-interactive mode a failed approver means DECLINE, immediately."""
    rules = fix_rules(SCALE_FIX, "TERMINAL: rm -f stale.cache",
                      extra=[["Your answer, one line", "hmm, I am not sure about this one"]])
    r = run_with_terminal_stdin(CRASHING, mode="run", pulse_opts=["--approver", "openrouter/fake-approver"],
                                llm_rules=rules, extra_files={"stale.cache": "x\n"}, timeout=120,
                                installed=True)
    assert "could not answer" in r.stdout, r.show()
    assert not r.timed_out, "the unattended run is stuck on a y/N prompt nobody will answer:\n" + r.stdout[-1500:]


USER_SIGINT = """
import signal, time
stop = {"flag": False}
def on_int(sig, frame):
    print("HANDLER: will save and stop", flush=True)
    stop["flag"] = True
signal.signal(signal.SIGINT, on_int)
from pulse import auto_track
auto_track()
loss = 1.0
for step in range(4000):
    loss *= 0.999
    time.sleep(0.01)
    if stop["flag"]:
        print("SAVED checkpoint at step", step, flush=True)
        break
print("EXIT OK", flush=True)
"""


def test_bug_the_scripts_own_ctrl_c_handler_never_runs():
    """A common pattern: the script installs its own SIGINT handler to save a checkpoint
    and stop cleanly on Ctrl+C, then calls auto_track(). PulseCLI.__init__ replaces the
    handler (pulse_cli.py ~3494-3498) and _sigint_handler never calls it: non-interactive,
    the first Ctrl+C restores it and raises KeyboardInterrupt (~3698-3700); interactive,
    the first pauses and the second raises. The user's handler NEVER runs -- no 'SAVED
    checkpoint', rc=-2 with a KeyboardInterrupt traceback. Without Pulse: HANDLER, SAVED,
    EXIT OK, rc 0 (test_ok_... control below).

    Correct: when Pulse is not going to pause (non-interactive, or no step follows), a
    callable original handler is called (self.original_sigint(sig, frame)) instead of
    raising KeyboardInterrupt."""
    r = h.run(USER_SIGINT, send_signal_after=(signal.SIGINT, 20), timeout=120, installed=True)
    assert not r.timed_out, r.show()
    assert "HANDLER: will save and stop" in r.stdout, "the script's own Ctrl+C handler never ran:\n" + r.show(2000)
    assert "SAVED checkpoint" in r.stdout and r.returncode == 0


def test_ok_the_scripts_ctrl_c_handler_runs_without_pulse():
    r = h.run(USER_SIGINT.replace(AT, ""), mode="plain", send_signal_after=(signal.SIGINT, 5),
              timeout=120, installed=True)
    assert "HANDLER" in r.stdout and "SAVED checkpoint" in r.stdout and r.returncode == 0, r.show()


SLOW_RERUN = """\
import time
from pulse import auto_track
auto_track()
scale = 0.0
loss = 1.0
for step in range(60):
    loss = loss * 0.9
    time.sleep(0.02)
r = 1.0 / scale
for epoch in range(8):
    time.sleep(0.5)
    print("EPOCH", epoch, "AT", repr(time.time()))
print("RESULT", r, "AT", repr(time.time()))
"""


def test_bug_fixed_rerun_output_is_block_buffered_on_a_terminal():
    """The first sweep's fix streams the re-run's output live (_run_restart_child pumps the
    child's pipes to the terminal). But the child now writes into a PIPE, not the user's
    terminal, so its print()s are block-buffered: on a real terminal (no PYTHONUNBUFFERED),
    'EPOCH 0' printed at t reaches the screen only when the child exits or fills 8 KB.
    Measured: EPOCH 0..7 printed 0.5 s apart all arrived at once, EPOCH 0 3.5 s late; for
    a real run that prints an epoch line every few minutes the re-run is silent for hours
    again. (Progress bars and colours also switch to their non-terminal behaviour.)

    Correct: re-run output appears as it is printed -- give the child PYTHONUNBUFFERED=1
    in _restart_env, or run it on a pty when stdout is a terminal."""
    r, chunks = run_with_terminal_stdout(SLOW_RERUN, llm_rules=fix_rules(SCALE_FIX), timeout=200,
                                         env={"PYTHONUNBUFFERED": ""}, installed=True)
    assert not r.timed_out and r.returncode == 0, r.show()
    lags = {}
    for arrived, data in chunks:
        for m in re.finditer(rb"EPOCH (\d+) AT ([0-9.]+)", data):
            lags[int(m.group(1))] = arrived - float(m.group(2))
    assert 0 in lags, f"the re-run's EPOCH lines never reached the terminal: {chunks[-3:]}"
    assert lags[0] < 1.5, f"EPOCH 0 reached the terminal {lags[0]:.1f}s after it was printed (lags {lags})"


SPAWN_DATALOADER = """
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
PULSE_IMPORT
X = torch.randn(256, 10); Y = torch.randn(256, 1)
if __name__ == "__main__":
    PULSE_START
    dl = DataLoader(TensorDataset(X, Y), batch_size=64, num_workers=2, multiprocessing_context="spawn")
    model = torch.nn.Linear(10, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    t0 = time.perf_counter()
    for epoch in range(4):
        for xb, yb in dl:
            opt.zero_grad(); loss = ((model(xb) - yb) ** 2).mean(); loss.backward(); opt.step()
    print("TRAIN_SECONDS", round(time.perf_counter() - t0, 2), flush=True)
"""


def test_bug_spawn_dataloader_workers_pay_pulses_import_every_epoch():
    """README style -- `from pulse import auto_track` with the imports, auto_track() in the
    main block -- plus a DataLoader with workers under 'spawn' (the default on macOS and
    Windows; persistent_workers=False, the default, starts new workers every epoch). Each
    worker re-imports the main module, so each imports Pulse: `import pulse` is still
    ~3.3 s and pulls in litellm (2.7 s), matplotlib, openai, aiohttp, PIL (pulse_cli.py:57,
    pulse.py:84-114) even though the worker never uses any of it. Measured on 4 epochs:
    6.6 s without Pulse, 25.5 s with it, 6.5 s with the import moved inside the main block
    -- the whole 4x slowdown is the import in the workers.

    Correct: `from pulse import auto_track` is cheap -- litellm/matplotlib/pulse_cli are
    imported lazily, when auto_track() actually starts a session."""
    pytest.importorskip("torch")
    plain = SPAWN_DATALOADER.replace("PULSE_IMPORT\n", "").replace("    PULSE_START\n", "")
    readme = SPAWN_DATALOADER.replace("PULSE_IMPORT", "from pulse import auto_track").replace(
        "PULSE_START", "auto_track()")
    base = h.run(plain, mode="plain", timeout=300, installed=True)
    tracked = h.run(readme, timeout=300, installed=True)

    def secs(r):
        m = re.search(r"TRAIN_SECONDS ([0-9.]+)", r.stdout)
        assert m, r.show()
        return float(m.group(1))
    assert secs(tracked) < secs(base) * 1.5 + 1.0, (
        f"4 epochs: {secs(base)}s without Pulse, {secs(tracked)}s with `from pulse import auto_track` at the top")


HELPER_SPAWN = """
import multiprocessing as mp
import setup_pulse          # the project's own setup module; it calls auto_track()
def sq(x):
    return x * x
if __name__ == "__main__":
    with mp.get_context("spawn").Pool(2) as p:
        print("POOL", sum(p.map(sq, range(10))), flush=True)
"""


def test_bug_auto_track_in_a_helper_module_starts_pulse_in_every_spawn_worker():
    """The first sweep's fix makes a module-level auto_track() a no-op in a spawn child
    only when the CALLER's module is __mp_main__ (pulse.py ~7098-7102;
    multiprocessing.parent_process() is still None while the child re-imports main). A
    project whose setup module calls auto_track() at import (`import setup_pulse` in
    train.py) is re-imported in every worker under its own name, so each worker starts a
    full Pulse: measured 3 x 'Pulse is ready' and start-of-run model calls from 3 pids.

    Correct: one session -- detect the spawn bootstrap itself (e.g.
    getattr(multiprocessing.current_process(), '_inheriting', False), set while the child
    prepares), not the caller's module name."""
    r = h.run(HELPER_SPAWN, extra_files={"setup_pulse.py": AT}, timeout=200, installed=True)
    assert r.returncode == 0 and "POOL 285" in r.stdout, r.show()
    assert r.stdout.count("Pulse is ready") == 1, f"{r.stdout.count('Pulse is ready')} Pulse sessions started"
    assert len({c["pid"] for c in r.llm_calls}) <= 1


BIG_RERUN = """\
import os, sys, time
from pulse import auto_track
auto_track()
scale = 0.0
loss = 1.0
for step in range(60):
    loss = loss * 0.9
    time.sleep(0.02)
r = 1.0 / scale
n = int(os.environ.get("BIG_LINES", "0"))
line = "epoch log line " + "x" * 84 + "\\n"
for i in range(n):
    sys.stdout.write(line)
if os.environ.get("BINARY_OUT"):
    sys.stdout.flush()
    sys.stdout.buffer.write(bytes(range(256)) * 4)
    sys.stdout.buffer.flush()
print("RESULT", r, flush=True)
"""

_MAXRSS_PROBE = """
import json, os, resource, sys
sys.path.insert(0, {sweep!r})
import e2e_harness as h
r = h.run({src!r}, llm_rules={rules!r}, env={{"BIG_LINES": {n!r}}}, timeout=600, installed=True)
print(json.dumps({{"rc": r.returncode, "timed_out": r.timed_out, "out": len(r.stdout),
                  "ok": "RESULT 0.5" in r.stdout,
                  "maxrss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}}))
"""


def _rerun_maxrss(lines):
    code = _MAXRSS_PROBE.format(sweep=SWEEP1, src=BIG_RERUN, rules=fix_rules(SCALE_FIX), n=str(lines))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    last = [l for l in out.stdout.splitlines() if l.startswith("{")]
    assert last, out.stderr[-2000:]
    return json.loads(last[-1])


def test_bug_restart_keeps_the_whole_rerun_output_in_memory():
    """_run_restart_child (pulse_cli.py ~6809-6842) appends every chunk of the re-run's
    stdout/stderr to a list for the whole run, then joins it into one string -- though
    only the last _CONFIRM_OUTPUT_CHARS (4000) are ever used (post-fix check / the agent).
    A re-run is often the rest of a long job: measured, a re-run that prints 200 MB of log
    raised the Pulse parent's peak RSS from 450 MB to 1420 MB (~5 bytes per byte printed);
    days of logging exhausts RAM.

    Correct: keep a bounded tail (e.g. a deque of the last ~64 KB) per stream."""
    small = _rerun_maxrss(0)
    big = _rerun_maxrss(2_000_000)        # 2M lines x 100 B = 200 MB
    assert small["ok"] and big["ok"] and big["out"] > 200_000_000, (small, big)
    grew = big["maxrss_mb"] - small["maxrss_mb"]
    assert grew < 150, f"Pulse's peak RSS grew by {grew:.0f} MB for 200 MB of re-run output ({small}, {big})"


def test_bug_binary_output_of_the_fixed_rerun_is_mangled():
    """A re-run that writes bytes to stdout (a tool piping raw output, `train.py > out.bin`)
    has them decoded as UTF-8 with errors='replace' and re-encoded by the pump
    (_run_restart_child, pulse_cli.py ~6813-6823): every byte >= 0x80 becomes U+FFFD
    (EF BF BD). Without Pulse -- and in the original run -- the bytes pass through intact.

    Correct: pump bytes to sys.stdout.buffer / the raw fd, decode only the kept copy."""
    def patch(real, argv, **pkw):
        pkw["text"] = False
        return real(argv, **pkw)
    r = _with_popen(patch)(BIG_RERUN, llm_rules=fix_rules(SCALE_FIX), env={"BINARY_OUT": "1"},
                           timeout=300, installed=True)
    assert r.returncode == 0 and b"RESULT 0.5" in r.stdout, r.stdout[-2000:]
    replaced = r.stdout.count("\ufffd".encode("utf-8"))
    assert bytes(range(256)) * 4 in r.stdout, (
        f"the re-run's binary output was altered ({replaced} U+FFFD replacement chars)")


def test_bug_hugging_face_token_in_the_script_lands_in_the_agent_log():
    """`--agent-log` promises "never let an API key reach it" (_agent_log_write,
    pulse_cli.py ~516-518) and the first sweep fixed this by running scrub_secrets over
    every write -- but its patterns (pulse_supabase.py ~91-105) only know sk-/sk-or-/AIza/
    sb_/gsk_/xox/ghp_ tokens and *api_key=/password=* assignments. The most common secret
    in an ML script, a Hugging Face token (`HF_TOKEN = "hf_..."`, the value passed to
    login()/from_pretrained(token=...)), is written verbatim, several times (every prompt
    carries the code). Correct: hf_ tokens (and *_TOKEN / *_SECRET assignments in code)
    are redacted."""
    token = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
    src = CRASHING.replace("scale = 0.0", f'HF_TOKEN = "{token}"\nscale = 0.0')
    r = h.run(src, llm_rules=fix_rules(SCALE_FIX), env={"PULSE_AGENT_LOG": "1"}, timeout=300,
              installed=True)
    path = os.path.join(r.workdir, "pulse_agent.log")
    assert os.path.exists(path), r.files
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert token not in text, f"the HF token appears {text.count(token)} times in pulse_agent.log"


def test_bug_checkin_prompt_misstates_how_many_steps_since_the_last_one():
    """Each periodic check-in tells the model "[Automatic check-in, {steps} training step(s)
    after the last one ...]" with {steps} = checkin_interval_steps (the agent's NEXTCHECK),
    not the steps that actually passed (pulse_cli.py ~9460). A check-in only starts once the
    previous one's answer has been applied (it is polled in update(), about once a second),
    so with NEXTCHECK 20 on a ~50 step/s loop consecutive check-ins were at steps 25, 128,
    231, 335 -- ~103 steps apart, told "20". The model judges trends per step from this.

    Correct: report self.step - (the previous check-in's step)."""
    r = h.run(AT + np_loop(400, 0.02), llm_reply="VERDICT: ok\nNEXTCHECK: 20\nCHECKNOTE: watch the loss",
              timeout=300, installed=True)
    assert r.returncode == 0, r.show()
    starts = [int(s) for s in re.findall(r"check-in \(step (\d+),", r.stdout)]
    claimed = [int(n) for c in r.llm_calls
               for n in re.findall(r"\[Automatic check-in, (\d+) training step", c["last_user"])]
    assert len(starts) >= 3 and len(claimed) >= 3, r.show()
    actual = [b - a for a, b in zip(starts, starts[1:])]
    assert all(abs(a - c) <= 5 for a, c in zip(actual, claimed[1:])), (
        f"check-ins at steps {starts} (gaps {actual}); the prompts said {claimed[1:]} steps since the last")


LONG_RERUN = """\
import os, time
from pulse import auto_track
auto_track()
scale = 0.0
loss = 1.0
for step in range(60):
    loss = loss * 0.9
    time.sleep(0.02)
r = 1.0 / scale
for epoch in range(60):
    time.sleep(0.5)
    with open("heartbeat.txt", "a") as f:
        f.write("%d\\n" % epoch)
    print("EPOCH", epoch, flush=True)
print("RESULT", r, flush=True)
"""


def test_bug_ctrl_c_during_the_fixed_rerun_shows_the_old_crash_and_exits_1():
    """Ctrl+C (SIGINT to the terminal's process group) while the fixed script is re-running:
    the KeyboardInterrupt escapes subprocess.run inside the crash hook, so Python prints
    "Error in sys.excepthook:" + Pulse's internal frames + "Original exception was:" + the
    ZeroDivisionError that has ALREADY BEEN FIXED, and exits 1. The user interrupted a
    healthy run and is told it crashed with the old bug. Correct: an interrupted re-run
    ends like any Ctrl+C -- exit 130/-2, no stale traceback."""
    box = {}

    def patch(real, argv, **pkw):
        p = real(argv, **pkw)
        box["p"] = p
        return p

    def interrupter():
        t0 = time.time()
        while time.time() - t0 < 200:
            p, wd = box.get("p"), box.get("wd")
            hb = os.path.join(wd, "heartbeat.txt") if wd else None
            if p and hb and os.path.exists(hb):
                with open(hb) as fh:
                    if len(fh.read().splitlines()) >= 3:
                        os.killpg(p.pid, signal.SIGINT)
                        return
            time.sleep(0.2)
    import tempfile
    box["wd"] = tempfile.mkdtemp(prefix="pulse-e2e-")
    threading.Thread(target=interrupter, daemon=True).start()
    r = _with_popen(patch)(LONG_RERUN, llm_rules=fix_rules(SCALE_FIX), timeout=150,
                           installed=True, workdir=box["wd"])
    assert not r.timed_out and "EPOCH 2" in r.stdout, r.show()
    assert "RESULT" not in r.stdout, "precondition: the re-run was interrupted"
    assert "Error in sys.excepthook" not in r.stderr, r.stderr[-2500:]
    assert r.returncode in (130, -2), f"exit code {r.returncode} for a Ctrl+C"


# =====================================================================================
# Verified working
# =====================================================================================

KERAS_CRASH = """
import numpy as np
import keras
import tensorflow as tf
from pulse import auto_track
auto_track()
tf.config.threading.set_intra_op_parallelism_threads(2)
keras.utils.set_random_seed(0)
BAD_EPOCH = 3
rng = np.random.default_rng(0)
X = rng.normal(size=(256, 8)).astype("float32")
y = (X @ rng.normal(size=(8, 1))).astype("float32")
m = keras.Sequential([keras.Input((8,)), keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
m.compile(optimizer=keras.optimizers.SGD(1e-2), loss="mse")
class Probe(keras.callbacks.Callback):
    def wsum(self):
        return float(sum(np.abs(w).sum() for w in self.model.get_weights()))
    def on_train_begin(self, logs=None):
        print("BEGIN_WSUM %.5f" % self.wsum(), flush=True)
    def on_epoch_begin(self, epoch, logs=None):
        print("EPOCH_BEGIN", epoch, flush=True)
    def on_epoch_end(self, epoch, logs=None):
        print("EPOCH_END %d WSUM %.5f" % (epoch, self.wsum()), flush=True)
        if epoch == BAD_EPOCH:
            raise ValueError("bad epoch %d" % epoch)
h = m.fit(X, y, epochs=8, batch_size=32, verbose=0, callbacks=[Probe()])
print("DONE epochs_run", len(h.history["loss"]), flush=True)
"""
KERAS_FIX = {"old": ["BAD_EPOCH = 3"], "new": ["BAD_EPOCH = -1"], "explanation": "the probe raised at epoch 3"}


def test_ok_keras_crash_fix_resume_continues_with_the_crash_weights():
    """Crash in epoch 3 -> checkpoint -> scripted fix -> restart: the re-run starts from the
    weights the crashed run had (BEGIN_WSUM == the crash-time WSUM), at epoch 3 (the one that
    had not completed from Pulse's point of view), and trains to epoch 7."""
    pytest.importorskip("keras")
    r = h.run(KERAS_CRASH, llm_rules=fix_rules(KERAS_FIX), timeout=400, installed=True)
    assert r.returncode == 0 and not r.timed_out, r.show()
    assert "BAD_EPOCH = -1" in r.script_after
    crash_wsum = re.search(r"EPOCH_END 3 WSUM ([0-9.]+)", r.stdout).group(1)
    begins = re.findall(r"BEGIN_WSUM ([0-9.]+)", r.stdout)
    assert len(begins) == 2 and begins[1] == crash_wsum, (begins, crash_wsum)
    rerun = r.stdout.split("Resuming from the checkpoint", 1)[1]
    assert re.findall(r"EPOCH_BEGIN (\d+)", rerun) == ["3", "4", "5", "6", "7"], rerun[:1500]
    assert "DONE epochs_run 5" in rerun


def test_ok_keras_fix_with_resume_false_starts_fresh():
    pytest.importorskip("keras")
    r = h.run(KERAS_CRASH, llm_rules=fix_rules(dict(KERAS_FIX, resume=False)), timeout=400, installed=True)
    assert r.returncode == 0 and not r.timed_out, r.show()
    begins = re.findall(r"BEGIN_WSUM ([0-9.]+)", r.stdout)
    assert len(begins) == 2 and begins[0] == begins[1], begins
    assert "Resuming from the checkpoint" not in r.stdout
    assert "DONE epochs_run 8" in r.stdout


APPROVER_RULE = "Your answer, one line"


def _auto_mode(reply, **kw):
    rules = fix_rules(SCALE_FIX, "TERMINAL: rm -f stale.cache", extra=[[APPROVER_RULE, reply]])
    return h.run(CRASHING, mode="run", pulse_opts=["--approver", "openrouter/fake-approver"],
                 llm_rules=rules, extra_files={"stale.cache": "old cache\n"}, timeout=300,
                 installed=True, **kw)


def test_ok_auto_mode_approve_runs_the_flagged_command():
    r = _auto_mode("APPROVE: it is a regenerable cache file")
    assert r.returncode == 0 and not r.timed_out, r.show()
    assert "fake-approver APPROVED" in r.stdout
    assert not os.path.exists(os.path.join(r.workdir, "stale.cache"))
    assert "scale = 2.0" in r.script_after and "RESULT 0.5" in r.stdout
    assert any(c["model"] == "openrouter/fake-approver" for c in r.llm_calls)


def test_ok_auto_mode_deny_refuses_and_the_reason_reaches_the_agent():
    r = _auto_mode("DENY: deleting files is not needed to fix a division")
    assert r.returncode == 0 and not r.timed_out, r.show()
    assert os.path.exists(os.path.join(r.workdir, "stale.cache"))
    agent_calls = [c for c in r.llm_calls if c["model"] != "openrouter/fake-approver"]
    assert any("denied it: deleting files is not needed" in c["all"] for c in agent_calls)
    assert "scale = 2.0" in r.script_after


def test_ok_auto_mode_approver_down_declines_when_stdin_is_closed():
    r = _auto_mode("hmm, I am not sure about this one")
    assert not r.timed_out and r.returncode == 0, r.show()
    assert "could not answer" in r.stdout
    assert os.path.exists(os.path.join(r.workdir, "stale.cache"))


def test_ok_agent_log_is_readable_and_has_no_provider_key():
    key = "sk-or-v1-" + "a1b2c3d4" * 8
    r = h.run(CRASHING, llm_rules=fix_rules(SCALE_FIX), timeout=300, installed=True,
              env={"OPENROUTER_API_KEY": key, "PULSE_AGENT_LOG": "1"})
    assert r.returncode == 0, r.show()
    with open(os.path.join(r.workdir, "pulse_agent.log"), encoding="utf-8") as fh:
        text = fh.read()
    assert key not in text
    assert "CALL #1" in text and "PASS 3" in text and "ZeroDivisionError" in text


def test_ok_periodic_checkin_fires_and_says_ok():
    r = h.run(AT + np_loop(400, 0.02), llm_reply="VERDICT: ok\nNEXTCHECK: 20\nCHECKNOTE: watch the loss",
              timeout=300, installed=True)
    assert r.returncode == 0 and "TRAINED" in r.stdout, r.show()
    assert r.stdout.count("Check-in verdict: ok") >= 2, r.show()
    assert not flagged(r) and not r.modified


def test_ok_short_script_exits_promptly_every_time():
    """The first sweep's exit hang, re-checked sequentially (one run at a time) 8 times."""
    src = AT + "import argparse\np = argparse.ArgumentParser()\np.add_argument('--lr')\np.parse_args()\n"
    for i in range(8):
        r = h.run(src, args=["--help"], timeout=60, installed=bool(i % 2))
        assert not r.timed_out and r.returncode == 0 and "--lr" in r.stdout, f"run {i}:\n" + r.show()


def test_ok_sys_exit_code_of_the_fixed_rerun_is_the_exit_code():
    src = CRASHING + "import sys\nsys.exit(3)\n"
    r = h.run(src, llm_rules=fix_rules(SCALE_FIX), timeout=300, installed=True)
    assert not r.timed_out and "RESULT 0.5" in r.stdout, r.show()
    assert r.returncode == 3


DDP = """
import time
import torch
import torch.multiprocessing as tmp
from pulse import auto_track

def worker(rank, world):
    auto_track()
    w = torch.zeros(4, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    for step in range(150):
        opt.zero_grad()
        loss = ((w - 1.0) ** 2).sum()
        loss.backward(); opt.step()
        time.sleep(0.01)
    print("RANK", rank, "DONE", round(loss.item(), 6), flush=True)

if __name__ == "__main__":
    tmp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("ALL DONE", flush=True)
"""


def test_ok_ddp_style_spawn_workers_that_call_auto_track_get_pulse():
    pytest.importorskip("torch")
    r = h.run(DDP, timeout=300, installed=True)
    assert r.returncode == 0 and "ALL DONE" in r.stdout, r.show()
    assert "RANK 0 DONE" in r.stdout and "RANK 1 DONE" in r.stdout
    assert r.stdout.count("Pulse is ready") == 2
    assert len({c["pid"] for c in r.llm_calls}) == 2


TQDM_RICH = """
import time
from tqdm import tqdm
from rich.progress import track
from rich.console import Console
from pulse import auto_track
auto_track()
loss = 1.0
for step in tqdm(range(200), desc="train"):
    loss *= 0.99
    time.sleep(0.01)
for step in track(range(100), description="eval"):
    loss *= 0.999
    time.sleep(0.01)
Console().print("[bold]FINAL[/bold]", round(loss, 6))
print("TRAINED", round(loss, 6), flush=True)
"""


def test_ok_tqdm_and_rich_progress_output_is_intact():
    pytest.importorskip("tqdm")
    pytest.importorskip("rich")
    r = h.run(TQDM_RICH, timeout=200, installed=True)
    base = h.run(TQDM_RICH.replace(AT, ""), mode="plain", timeout=200, installed=True)
    assert r.returncode == 0 and base.returncode == 0, r.show()
    assert "FINAL 0.121224" in r.stdout and "TRAINED 0.121224" in r.stdout
    assert "200/200" in r.stderr and not flagged(r) and not r.modified
    assert "Traceback" not in r.stderr


def test_ok_sklearn_partial_fit_loop_is_left_alone():
    pytest.importorskip("sklearn")
    src = """
import time
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
""" + AT + """
rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 5)); y = X @ rng.normal(size=5)
model = SGDRegressor(random_state=0)
for epoch in range(150):
    model.partial_fit(X, y)
    loss = mean_squared_error(y, model.predict(X))
    time.sleep(0.01)
print("FINAL", round(loss, 6))
"""
    r = h.run(src, timeout=200, installed=True)
    assert r.returncode == 0 and "FINAL" in r.stdout, r.show()
    assert not flagged(r) and not r.modified
