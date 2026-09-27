"""The check-in worker's y/N for a flagged command must not read stdin while the training
thread is at its own prompt -- both reading at once splits the user's answer."""
import threading
import time

import pulse.pulse_cli as pc
from pulse.pulse_cli import PulseCLI


def test_bug_checkin_confirm_prompt_waits_for_the_training_threads_prompt(monkeypatch):
    for name in ("PULSE_APPROVER", "PULSE_APPROVER_API_KEY", "PULSE_APPROVER_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    cli = PulseCLI.__new__(PulseCLI)
    cli.config = {}
    cli.agent_provider = None
    cli.agent_key = None
    cli.agent_model_string = None
    events = []

    def fake_prompt(*a, **k):
        events.append(("worker-prompt", time.monotonic()))
        return "n"

    monkeypatch.setattr(pc, "_prompt_text", fake_prompt)
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)

    released = []
    with pc._PROMPT_LOCK:                       # the training thread is at its prompt
        t = threading.Thread(target=cli._confirm_terminal_command,
                             args=("rm -rf out", {"deletes_files": True}))
        t.start()
        time.sleep(0.5)
        assert events == [], "the worker prompted while the training thread's prompt was open"
        released.append(time.monotonic())
    t.join(5)
    assert events and events[0][1] >= released[0]


def test_ok_prompt_lock_is_reentrant_on_one_thread():
    with pc._PROMPT_LOCK:
        with pc._PROMPT_LOCK:
            pass
