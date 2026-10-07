"""The 2026-10-07 review: the scheduled check, prompts that did not fit their context, and
the one rule for what Pulse acts on by itself. No model is called."""
import types

import pytest

from pulse import pulse_approver as approver
from pulse import pulse_brain as brain
from pulse import pulse_code_agent as native
from pulse import pulse_detect as detect
from pulse import pulse_ui as ui


# ---------------------------------------------------------------- the scheduled check

def _response(text, finish="stop"):
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason=finish)
    return types.SimpleNamespace(choices=[choice])


def test_an_empty_answer_is_an_error_not_a_silent_verdict(monkeypatch):
    import litellm
    seen = {}
    monkeypatch.setattr(litellm, "completion", lambda **kw: seen.update(kw) or _response("", "length"))
    ask = brain.build_litellm_agent("openrouter/deepseek/deepseek-v4-flash", api_key="k")
    with pytest.raises(brain.EmptyAnswer, match="token budget"):
        ask("audit")
    assert seen["max_tokens"] >= 32000 and seen["timeout"] >= 900     # room to think, time to finish


def test_a_verdict_written_as_labelled_lines_is_read():
    text = ("**Problem**\n`decay_steps = epochs` sets the decay to 12 steps.\n\n"
            "**Status:** problem  \n**Risk:** high -- wasted run\n**Next check:** 1 minute (or stop)")
    assert brain.parse_decision(text) == {"status": "problem", "risk": "high", "next_check_minutes": 1.0,
                                          "findings": []}
    assert brain.parse_decision('{"status": "ok", "next_check_minutes": 5}')["status"] == "ok"
    assert brain.parse_decision("The status of the run is fine.") == {}


def test_the_audit_says_the_code_shown_is_newer_than_the_code_running():
    pack = {"step": 50, "session": {}, "fixes_applied": [{"step": 40, "summary": "lr 0.3 -> 0.05"}],
            "code": "   1 | lr = 0.05"}
    text = brain.Brain.render_evidence(pack)
    assert "AS IT IS NOW" in text and "keeps running the code it started with" in text


# ---------------------------------------------------------------- the reviewer

def test_a_cut_diff_says_so():
    long = "+x = 1\n" * 4000
    prompt = approver.build_change_prompt("fix it", "explanation", long, "/p")
    assert "diff cut here" in prompt and "DENY" in prompt
    assert "Pulse's on their behalf" in prompt


# ---------------------------------------------------------------- the native tool loop

def test_run_tools_are_offered_only_inside_the_app():
    ui.set_host(None)
    names = {t["function"]["name"] for t in native._tools()}
    assert not names & native._RUN_TOOLS
    prompt = native._system_prompt(types.SimpleNamespace())
    assert "start_run starts a script" not in prompt and "pulse run --stream" in prompt

    class Host:
        def run_actions(self):
            return {}
    ui.set_host(Host())
    try:
        assert native._RUN_TOOLS <= {t["function"]["name"] for t in native._tools()}
        assert "start_run starts a script" in native._system_prompt(types.SimpleNamespace())
    finally:
        ui.set_host(None)


def test_the_run_instructions_reach_the_tool_loop():
    from pulse import pulse_app
    cli = types.SimpleNamespace(_native_prompt_suffix=pulse_app.DEBUG_PROMPT_NATIVE)
    prompt = native._system_prompt(cli)
    assert "LIVE TRAINING RUN" in prompt and "restart_run" in prompt and "RESTART:" not in prompt
    assert "RESTART:" in pulse_app.DEBUG_PROMPT


def test_a_restart_counts_as_checking_the_change():
    state = types.SimpleNamespace(dirty=True)
    assert native._ran_the_change(state, "Restarted with the current code: ...").startswith("Restarted")
    assert state.dirty is False
    state.dirty = True
    native._ran_the_change(state, "No run is open.")
    assert state.dirty is True


# ---------------------------------------------------------------- one rule for acting

def test_the_shared_rule_for_what_pulse_acts_on():
    def f(check, severity):
        return detect.Finding(check, "loss", severity, "m")
    assert detect.acts_on(f("nonfinite", detect.CRITICAL))
    assert not detect.acts_on(f("plateau", detect.WARNING))
    assert not detect.acts_on(f("throughput_stopped", detect.CRITICAL))


# ---------------------------------------------------------------- what the evidence says

def test_the_last_value_a_script_computes_before_exiting_is_read(tmp_path):
    """The sampler looks a few times a second; a final val_loss computed just before the
    script exits used to be missing from every run (and the closing audit said so)."""
    import os
    import subprocess
    import sys
    from pulse import pulse_stream as stream
    script = tmp_path / "train.py"
    script.write_text(
        "import time\n"
        "from pulse import auto_track\n"
        "auto_track(mode='stream', throttle_interval=0.08)\n"
        "val_loss = 1.0\n"
        "loss = 2.0\n"
        "for i in range(25):\n"
        "    loss = loss * 0.9\n"
        "    time.sleep(0.04)\n"
        "val_loss = 0.123\n")
    src = os.path.dirname(os.path.dirname(stream.__file__))
    env = dict(os.environ, PYTHONPATH=src, PULSE_HOME=str(tmp_path / "home"), PULSE_CACHE_DIR=str(tmp_path / "home"))
    subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), env=env, timeout=120, check=True,
                   capture_output=True)
    directory = next(root for root, _d, files in os.walk(tmp_path / ".pulse_stream") if "session.json" in files)
    seen = []
    for frame in stream.StreamReader(directory).poll():
        if frame.get("kind") == stream.KIND_SCALARS and "val_loss" in (frame.get("values") or {}):
            seen.append(frame["values"]["val_loss"])
    assert seen and seen[-1] == 0.123, seen


def test_a_finished_runs_evidence_gives_its_length_not_the_time_since_it_started():
    pack = {"step": 300, "elapsed_seconds": 31.2, "finished": True, "session": {"script": "t.py"}}
    first = brain.Brain.render_evidence(pack).splitlines()[0]
    assert "31.2s long, finished" in first
    pack["finished"] = False
    assert "elapsed, still running" in brain.Brain.render_evidence(pack).splitlines()[0]


def test_a_low_risk_audit_problem_is_shown_not_acted_on():
    from pulse import pulse_console
    console = pulse_console.Console.__new__(pulse_console.Console)
    console._diagnosed, diagnosed = set(), []
    console._diagnose = diagnosed.append
    console._escalate([], {"audit": {"step": 3, "t": 1.0, "risk": "low", "text": "stopped at step 399 of 400"}})
    assert diagnosed == []
    console._escalate([], {"audit": {"step": 4, "t": 2.0, "risk": "high", "text": "lr is 0 from step 13"}})
    assert len(diagnosed) == 1
