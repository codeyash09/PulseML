"""End to end: a real training script in stream mode, read by a brain in another process.

These run actual subprocesses and take a few seconds. They are the tests that would
have caught auto_track() crashing on a machine without TensorFlow.
"""
import glob
import json
import os
import shutil
import site
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from pulse.pulse_brain import Brain            # noqa: E402

SCRIPT = """\
import os, time
from pulse import auto_track
import numpy as np

STEPS = int(os.environ.get("STEPS", "600"))
N, D = 256, 64
rng = np.random.default_rng(0)
X = rng.normal(size=(N, D)).astype(np.float32)
w_true = rng.normal(size=(D, 1)).astype(np.float32)
y = X @ w_true
w = np.zeros((D, 1), dtype=np.float32)
lr = float(os.environ["LR"])

auto_track(mode="stream", throttle_interval=0.08)

for step in range(STEPS):
    pred = X @ w
    err = pred - y
    loss = float((err * err).mean())
    grad = (X.T @ err) * (2.0 / N)
    grad_norm = float(np.linalg.norm(grad))
    w -= lr * grad
    time.sleep(0.004)
print("done", flush=True)
"""


def run_training(workdir, lr, steps=600, timeout=180):
    script = os.path.join(workdir, "train.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(SCRIPT)
    # HOME is redirected so the run cannot touch the real ~/.pulse state --
    # but on a machine where dependencies live in the user's site-packages
    # (pip install --user) that also hides them from the subprocess, so the
    # parent's user site directory is put back on PYTHONPATH explicitly.
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join(
                   p for p in (SRC, site.getusersitepackages(), os.environ.get("PYTHONPATH", "")) if p),
               PULSE_LOGGING="0", LR=str(lr), STEPS=str(steps),
               HOME=os.path.join(workdir, "home"))
    os.makedirs(env["HOME"], exist_ok=True)
    result = subprocess.run([sys.executable, script], cwd=workdir, env=env,
                            capture_output=True, text=True, timeout=timeout)
    sessions = glob.glob(os.path.join(workdir, ".pulse_stream", "*"))
    return result, (sorted(sessions)[-1] if sessions else None)


def read(directory, rounds=3):
    brain = Brain(directory)
    brain.poll_once()
    for _ in range(rounds):
        brain.ingest([])        # extra evaluations, for the confirmation gate
    return brain


class StreamModeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-e2e-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_a_healthy_run_streams_and_raises_nothing(self):
        result, directory = run_training(self.tmp, lr=0.01)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIsNotNone(directory, "no session directory was created")
        brain = read(directory)
        self.assertGreater(len(brain.histories.get("loss", [])), 5, "no loss readings streamed")
        self.assertTrue(brain.finished, "the monitor did not close the stream cleanly")
        self.assertEqual([f.message for f in brain.engine.current()], [],
                         "false positives on a healthy run")
        losses = brain.histories["loss"]
        self.assertLess(losses[-1], losses[0], "the run did not actually learn")

    def test_a_diverging_run_is_caught(self):
        result, directory = run_training(self.tmp, lr=5.0, steps=400)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        brain = read(directory)
        checks = {f.check for f in brain.engine.current()}
        self.assertIn("nonfinite", checks, f"divergence to NaN was not detected: {checks}")
        self.assertEqual(brain.engine.current()[0].severity, "critical")

    def test_a_run_that_never_learns_is_caught(self):
        # The fault that motivated the "never learned" check: nothing crashes, nothing
        # spikes, the loss simply sits where it started.
        result, directory = run_training(self.tmp, lr=0.0, steps=600)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        brain = read(directory)
        checks = {f.check for f in brain.engine.current()}
        self.assertTrue({"never_learned", "frozen"} & checks,
                        f"a completely flat run produced no finding: {checks}")

    def test_the_training_process_does_no_pulse_work_on_its_own_thread(self):
        result, directory = run_training(self.tmp, lr=0.01, steps=400)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        with open(os.path.join(directory, "state.json"), encoding="utf-8") as handle:
            state = json.load(handle)
        cost = state.get("cost") or {}
        self.assertGreater(cost.get("samples", 0), 3)
        # The sampling runs on its own thread; what it costs must still be small enough
        # to disclose honestly.
        self.assertLess(cost.get("percent_of_run", 100.0), 5.0, f"monitor cost too high: {cost}")

    def test_crash_reaches_the_stream(self):
        script = os.path.join(self.tmp, "boom.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent("""\
                from pulse import auto_track
                auto_track(mode="stream")
                loss = 1.0
                raise ValueError("training exploded")
            """))
        env = dict(os.environ,
                   PYTHONPATH=os.pathsep.join(
                       p for p in (SRC, site.getusersitepackages(), os.environ.get("PYTHONPATH", "")) if p),
                   PULSE_LOGGING="0", HOME=os.path.join(self.tmp, "home2"))
        os.makedirs(env["HOME"], exist_ok=True)
        subprocess.run([sys.executable, script], cwd=self.tmp, env=env,
                       capture_output=True, text=True, timeout=120)
        directory = sorted(glob.glob(os.path.join(self.tmp, ".pulse_stream", "*")))[-1]
        brain = read(directory)
        crashes = [e for e in brain.events if e.get("event") == "crash"]
        self.assertEqual(len(crashes), 1, f"the crash never reached the stream: {brain.events}")
        self.assertIn("training exploded", crashes[0]["traceback"])

    def test_auto_track_works_without_tensorflow(self):
        # The regression that made Pulse unusable for NumPy, PyTorch and JAX users:
        # auto_track() imported tensorflow unconditionally.
        result, directory = run_training(self.tmp, lr=0.01, steps=100)
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertNotIn("No module named 'tensorflow'", result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
