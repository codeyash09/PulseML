"""The whole suite gets its own Pulse folder: credentials, profile, settings and saved keys
written by a test (an interactive setup driven with fake answers remembers its choices)
must never reach the real ~/.pulse, nor leak into the next test's start."""
import os
import tempfile

if not os.environ.get("PULSE_CACHE_DIR"):
    os.environ["PULSE_CACHE_DIR"] = tempfile.mkdtemp(prefix="pulse-tests-home-")
