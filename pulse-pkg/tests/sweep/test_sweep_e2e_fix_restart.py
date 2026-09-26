"""End-to-end: a crash that Pulse's agent fixes, followed by Pulse's restart of the script.

The fake model (tests/sweep/e2e_fake_llm) answers the diagnose pipeline with a scripted,
correct one-line fix, so the whole path runs for real: crash hook -> agent passes -> the
fix written to train.py -> restart (subprocess) -> post-fix confirmation -> exit.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e_harness as h          # noqa: E402

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

FIX = {"old": ["scale = 0.0"], "new": ["scale = 2.0"],
       "explanation": "scale was zero, which divides by zero at the end of the run"}

RULES = [
    ["PASS 6", json.dumps({"resolved": True, "reason": "the re-run printed its result"})],
    ["PASS 4", json.dumps({"passes": True, "reason": "fixes the division by zero"})],
    ["PASS 3", json.dumps(FIX)],
    ["PASS 2", "Root cause: scale is 0.0 so 1.0 / scale raises ZeroDivisionError. Fix: set scale = 2.0."],
    ["PASS 1", "The problem is at train.py line 4: scale = 0.0 used as a divisor on line 9."],
]


def _run(**kw):
    return h.run(CRASHING, llm_rules=RULES, timeout=240, installed=True, **kw)


def test_ok_crash_is_fixed_in_the_file_and_the_script_restarted():
    r = _run()
    assert not r.timed_out, r.show()
    assert "scale = 2.0" in r.script_after, "the fix was not written:\n" + r.show(4000)
    assert r.returncode == 0, r.show(4000)
    assert os.path.isdir(os.path.join(r.workdir, ".pulse_history")), r.files


def test_bug_output_of_the_fixed_rerun_is_never_shown():
    """After Pulse fixes a crash it re-runs the script with subprocess.run(capture_output=
    True) (pulse_cli.py _restart_process ~6199) and, when the post-fix check says the fix
    worked, calls sys.exit(0) without ever writing the captured stdout/stderr
    (`if resolved: sys.exit(0)`, ~6213). Everything the fixed run printed -- its training
    log, its final metrics, "saved model to ..." -- is thrown away, and nothing at all is
    shown while it runs (for a real run: hours of silence). Correct: the re-run's output
    reaches the user (streamed live, or at the very least written out on success too)."""
    r = _run()
    assert "scale = 2.0" in r.script_after, "precondition: the fix was applied\n" + r.show(4000)
    assert "RESULT 0.5" in r.stdout, "the fixed run's output never reached the user:\n" + r.stdout[-3000:]
    assert "saved model to model.pt" in r.stdout
