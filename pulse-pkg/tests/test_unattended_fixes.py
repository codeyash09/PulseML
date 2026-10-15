"""What a 277-run benchmark of unattended (headless) debugging showed going wrong, each fixed:

* a crash fixed but never run again (the turn ended on "/restart runs it again", with nobody
  there to type it) -- headless now restarts it; and a script dying before its first step got
  no agent at all -- now it gets one;
* tool directives written as `<VIEW: x>` or as DeepSeek's own tool-call markup were read as
  prose, and the turn spent its retries asking again;
* audits thought through their whole token budget and answered nothing (64% of them);
* a heavy-tailed per-batch loss raised a CRITICAL "spike" every few seconds;
* a run losing most of its time to a sleep or a wait was invisible to every check;
* the first audit came 15 minutes in, and warnings never brought one sooner.

No model is called."""
import importlib.util
import sys
import threading
import time
import types

import pytest

from pulse import pulse_app as appmod
from pulse import pulse_approver as approver
from pulse import pulse_brain as brain
from pulse import pulse_cli
from pulse import pulse_detect as detect
from pulse import pulse_headless as headless
from pulse import pulse_monitor as monitor
from pulse import pulse_stream as stream
from pulse import pulse_supabase as cloud
from pulse.pulse_cli import PulseCLI


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "pulsehome" / "credentials.json")
    return tmp_path / "pulsehome"


class FakeCli:
    agent_provider = agent_key = agent_model_string = agent_api_base = None
    focus = []
    _project_root = None


class AgentCli(FakeCli):
    agent_provider, agent_key = "openrouter", "k"


# ---------------------------------------------------------------- directives as models write them

@pytest.mark.parametrize("written, read", [
    ("<VIEW: config.json:1-15>", "VIEW: config.json:1-15"),
    ("<RUNSTATUS>", "RUNSTATUS:"),
    ("<RUNSTATUS:>", "RUNSTATUS:"),
    ("<RESTART>", "RESTART:"),
    ("<TERMINAL: cat config.json>", "TERMINAL: cat config.json"),
    ("<VIEW>train.py:1-9</VIEW>", "VIEW: train.py:1-9"),
    ("RUNSTATUS", "RUNSTATUS:"),
    ("<｜DSML｜view: train.py:35-71>", "VIEW: train.py:35-71"),
    ("<RUNSTATUS: ", "RUNSTATUS:"),
])
def test_wrapped_directives_are_read(written, read):
    assert PulseCLI.normalize_directives(f"Let me look.\n{written}\n").splitlines()[1] == read


def test_deepseek_tool_call_markup_is_read():
    text = ('I will read the config.\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="VIEW">\n'
            '<｜DSML｜parameter name="file" string="true">config.json</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="range" string="true">1-11</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n</｜DSML｜tool_calls>')
    _clean, *_rest, views = PulseCLI._extract_directives(text)
    assert views == ["config.json:1-11"]
    # the broken form it also produced: the value runs into the closing tag
    broken = ('<｜DSML｜invoke name="TERMINAL">\n<｜DSML｜parameter name="command" value="pwd</｜DSML｜parameter>\n'
              '</｜DSML｜invoke>')
    assert PulseCLI._extract_new_directives(broken)[1] == {"terminal": ["pwd"]}


def test_only_pulse_directives_are_rewritten():
    for text in ("<details>", "<br>", "<END>", "a < b and c > d", "x<y>z", "<commit_begin>", "RESTARTING now",
                 "<STOP", "<RESTART:"):          # left open, a run-changing one is not guessed at
        assert PulseCLI.normalize_directives(text) == text
    # a command over several lines cannot be one directive line: left alone, not mangled
    multi = '<｜DSML｜invoke name="TERMINAL">\n<｜DSML｜parameter name="command">python -c "\nimport os\n"' \
            '</｜DSML｜parameter>\n</｜DSML｜invoke>'
    assert "TERMINAL:" not in PulseCLI.normalize_directives(multi)


def test_a_wrapped_restart_reaches_run_control(monkeypatch):
    from pulse import pulse_code
    from pulse import pulse_ui as ui
    asked = []
    host = types.SimpleNamespace(run_actions=lambda: {"restart_run": lambda: asked.append("restart") or "Restarted",
                                                      "run_status": lambda: "status"})
    monkeypatch.setattr(ui, "host", lambda: host)
    assert pulse_code._service_run_directives("Fixed it.\n<RESTART>\n") == ["Restarted"] and asked == ["restart"]


# ---------------------------------------------------------------- thinking within the budget

def _response(text, finish="stop"):
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason=finish)
    return types.SimpleNamespace(choices=[choice])


def test_an_audit_caps_its_thinking_and_retries_once_with_less(monkeypatch):
    import litellm
    calls = []
    answers = iter([_response("", "length"), _response('ok\n{"status": "ok"}')])
    monkeypatch.setattr(litellm, "completion", lambda **kw: calls.append(kw) or next(answers))
    ask = brain.build_litellm_agent("openrouter/deepseek/deepseek-v4-flash", api_key="k")
    assert ask("audit").startswith("ok")
    caps = [kw["extra_body"]["reasoning"]["max_tokens"] for kw in calls]
    assert caps == [brain.AUDIT_REASONING_TOKENS, brain.AUDIT_RETRY_REASONING_TOKENS]
    assert caps[0] * 2 < calls[0]["max_tokens"]          # it thinks up to ~2x the cap: room is left


def test_a_model_without_the_setting_is_called_as_before(monkeypatch):
    import litellm
    calls = []
    monkeypatch.setattr(litellm, "completion", lambda **kw: calls.append(kw) or _response("", "length"))
    with pytest.raises(brain.EmptyAnswer, match="token budget"):
        brain.build_litellm_agent("anthropic/claude-x", api_key="k")("audit")
    assert len(calls) == 1 and "extra_body" not in calls[0]


def test_agent_calls_cap_thinking_tighter_after_an_empty_answer():
    loose = pulse_cli._reasoning_cap("openrouter/deepseek/deepseek-v4-flash", 32000)
    tight = pulse_cli._reasoning_cap("openrouter/deepseek/deepseek-v4-flash", 32000, tight=True)
    assert 2 * loose["extra_body"]["reasoning"]["max_tokens"] < 32000
    assert tight["extra_body"]["reasoning"]["max_tokens"] < loose["extra_body"]["reasoning"]["max_tokens"]
    assert pulse_cli._reasoning_cap("openai/gpt-x", 32000) == {}


# ---------------------------------------------------------------- the loss "spike" that is only noise

def _engine():
    return detect.DetectionEngine(sensitivity=0.3)


def _spikes(engine, values):
    found = []
    history = []
    for v in values:
        history.append(v)
        result = engine.update({"loss": list(history)}, step=len(history))
        found += [f for f in result["raised"] if f.check == "loss_spike"]
    return found


@pytest.mark.parametrize("gap", [15, 35, 60])
def test_a_heavy_tailed_batch_loss_is_not_a_spike_each_time(gap):
    # converged near zero, with a hard batch every `gap` readings that comes straight back --
    # the benchmark run: 40 CRITICAL "spikes" in ten minutes at 4 readings a second
    import random
    rng = random.Random(0)
    values = [1.16 * 0.9 ** i for i in range(30)]
    for i in range(400):
        values.append(rng.uniform(0.3, 0.9) if i % gap == gap - 1 else 0.003 + 0.004 * rng.random())
    assert len(_spikes(_engine(), values)) <= 2


def test_a_real_divergence_after_a_calm_run_is_still_caught():
    values = [1.0 * 0.85 ** i for i in range(40)] + [0.02] * 60 + [0.02 * 3 ** k for k in range(1, 8)]
    assert _spikes(_engine(), values)


def test_a_jump_far_beyond_the_usual_tail_is_still_caught():
    values = [1.0 * 0.8 ** i for i in range(20)]
    for i in range(120):
        values.append(0.4 if i % 9 == 4 else 0.004)
    values += [25.0, 30.0, 40.0]                     # a learning rate that jumped 200x
    assert any(f.values["latest"] >= 25.0 for f in _spikes(_engine(), values))


# ---------------------------------------------------------------- where the time goes

def _user_module(tmp_path, source):
    path = tmp_path / "loop_code.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("loop_code_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, str(path)


def test_the_monitor_samples_which_line_the_training_thread_is_at(tmp_path):
    module, path = _user_module(tmp_path, "import time\n\ndef loop(stop):\n    while not stop.is_set():\n"
                                          "        time.sleep(0.01)  # the slow line\n")
    stop = threading.Event()
    worker = threading.Thread(target=module.loop, args=(stop,), daemon=True)
    worker.start()
    mon = monitor.Monitor(script_path=path, directory=str(tmp_path / "s"))
    try:
        for _ in range(40):
            mon.note_where(sys._current_frames()[worker.ident])
            time.sleep(0.005)
        mon._send_where(time.monotonic())
    finally:
        stop.set()
        mon.close()
    events = [f for f in stream.StreamReader(str(tmp_path / "s")).poll() if f.get("event") == "where"]
    top = events[-1]["lines"][0]
    assert top["file"] == path and top["line"] == 5 and "time.sleep(0.01)" in top["code"]
    assert top["count"] >= 30 and events[-1]["samples"] == 40


def _where(file, line, code, count, samples, seconds=30.0, inside=""):
    return {"kind": stream.KIND_EVENT, "event": "where", "step": 10, "samples": samples, "seconds": seconds,
            "lines": [{"file": file, "line": line, "code": code, "count": count, "inside": inside}]}


def _brain(tmp_path):
    (tmp_path / "session").mkdir(parents=True, exist_ok=True)
    return brain.Brain(str(tmp_path / "session"))


def test_a_run_waiting_most_of_its_time_is_a_warning_with_the_line(tmp_path):
    b = _brain(tmp_path)
    first = b._fold([_where("/p/train.py", 72, "time.sleep(1.0)  # throttle the reader", 100, 120)])[1]
    assert not first                                             # 30 s is not enough to say
    raised = b._fold([_where("/p/train.py", 72, "time.sleep(1.0)  # throttle the reader", 100, 120)])[1]
    (finding,) = [f for f in raised if f.check == "time_sink"]
    assert finding.severity == detect.WARNING and finding.variable == "train.py:72"
    assert "83%" in finding.message and "time.sleep(1.0)" in finding.message
    assert not [f for f in b._fold([_where("/p/train.py", 72, "time.sleep(1.0)", 100, 120)])[1]
                if f.check == "time_sink"]                       # said once
    text = brain.Brain.render_evidence(b.evidence())
    assert "Where the training thread spent its time" in text and "train.py:72" in text and "(waiting)" in text
    assert "time_sink on train.py:72" in text


def test_a_wait_that_stops_holding_the_run_is_no_longer_reported(tmp_path):
    b = _brain(tmp_path)
    for _ in range(2):
        b._fold([_where("/p/train.py", 72, "time.sleep(1.0)", 100, 120)])
    assert "time_sink" in str(b.evidence()["findings"])
    for _ in range(4):
        b._fold([_where("/p/train.py", 50, "loss.backward()", 110, 120)])
    assert "time_sink" not in str(b.evidence()["findings"])


def test_a_wait_inside_a_library_counts_and_computing_does_not(tmp_path):
    b = _brain(tmp_path)
    for _ in range(3):
        raised = b._fold([_where("/p/train.py", 40, "for xb, yb in loader:", 90, 120,
                                 inside="queues.py:get")])[1]
    (sink,) = [f for f in b.evidence()["findings"] if f["check"] == "time_sink"]
    # input-bound is worth knowing, not a bug: said so, and not as a warning (as one, it had
    # the agent deleting data-loading calls)
    assert sink["severity"] == detect.INFO and "I/O or a queue" in sink["message"]
    assert "not a bug" not in sink["message"]          # an observation, not a verdict (seat reports)
    computing = _brain(tmp_path / "other")
    for _ in range(3):
        raised = computing._fold([_where("/p/train.py", 50, "loss.backward()", 110, 120)])[1]
        assert not [f for f in raised if f.check == "time_sink"]


# ---------------------------------------------------------------- when the audit looks

def test_the_first_audit_comes_within_five_minutes(tmp_path):
    schedule = brain.Schedule(str(tmp_path / "schedule.json"))
    assert schedule.seconds_remaining() <= brain.FIRST_LOOK_SECONDS + 1
    assert schedule.interval == brain.DEFAULT_INTERVAL_SECONDS          # later looks keep their default


def test_a_warning_brings_the_audit_forward_once(tmp_path):
    b = _brain(tmp_path)
    b.schedule.set(1800, "quiet", "low")
    warning = detect.Finding("plateau", "loss", detect.WARNING, "loss stopped improving")
    b._fold = lambda frames: ([], [warning])
    b.ingest([{}])
    assert b.schedule.seconds_remaining() <= brain.WARNING_LOOK_SECONDS + 1
    b.schedule.set(1800, "the audit happened", "low")
    b.ingest([{}])                                                       # the same warning again
    assert b.schedule.seconds_remaining() > 1700


# ---------------------------------------------------------------- unattended: a fix is run

def _headless(tmp_path, cli=None):
    log = open(tmp_path / "log", "w")
    return headless.HeadlessApp(cli or FakeCli(), str(tmp_path), log), log


def test_headless_restarts_a_crashed_run_whose_fix_no_run_uses(tmp_path):
    app, log = _headless(tmp_path)
    restarted = []
    app._restart = lambda: restarted.append(True)
    app.session = {"script": "/p/train.py", "session_id": "s1"}
    app._fix_not_run(over=True, launched=True, hosted=False)
    log.close()
    assert restarted and "starts it again with the change" in (tmp_path / "log").read_text()


def test_headless_leaves_a_live_run_to_the_agent_and_says_so_in_the_request(tmp_path):
    app, log = _headless(tmp_path)
    app._restart = lambda: pytest.fail("a live run was restarted behind the agent's back")
    app._fix_not_run(over=False, launched=True, hosted=False)
    log.close()
    assert "Nobody is at the machine" in app._unattended_note and "restart_run" in app._unattended_note
    assert appmod.App._unattended_note == ""                          # the person at the screen decides


def test_the_app_still_tells_the_person(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    said = []
    app.note = said.append
    app._fix_not_run(over=True, launched=True, hosted=False)
    assert said == ["/restart runs it again with the change."]


def test_a_failed_restart_tells_the_agent_what_the_script_printed(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    out = tmp_path / "run.log"
    out.write_text("Traceback (most recent call last):\n  File \"train.py\", line 3\nNameError: name 'F' is not defined\n")

    def restart():
        app._start_failure = {"script": "/p/train.py", "code": 1, "log": str(out)}

    app.console, app.session = object(), {"session_id": "s1"}
    app._restart = restart
    text = app._agent_restart_run()
    assert "exited with code 1 before its first training step" in text and "NameError: name 'F'" in text


# ---------------------------------------------------------------- unattended: a script that never starts

def _startup(tmp_path, outcomes, monkeypatch, starts_after=None):
    """A HeadlessApp whose script keeps dying before its first step; the agent's turns end in
    `outcomes`; the launch after `starts_after` turns opens a run."""
    app, log = _headless(tmp_path, AgentCli())
    out = tmp_path / "run.log"
    out.write_text("Traceback (most recent call last):\nKeyError: 'lr'\n")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.json").write_text('{"lr": 0.1,}')
    failure = {"script": str(project / "train.py"), "code": 1, "log": str(out), "argv": ["train.py"],
               "cwd": str(project)}
    app._start_failure = dict(failure)
    seen = {"turns": [], "launches": 0}
    turns = iter(outcomes)

    def start(work, request):
        seen["turns"].append(request)
        app._last_outcome = next(turns)
        if app._last_outcome == "edited with a command":
            (project / "config.json").write_text('{"lr": 0.1}')
            app._last_outcome = "declined"
        app._job = None

    def launch(arg, replace=None):
        seen["launches"] += 1
        if starts_after is not None and seen["launches"] >= starts_after:
            app.console, app.session, app._start_failure = object(), {"session_id": "new"}, None
        else:
            app._start_failure = dict(failure)

    monkeypatch.setattr(app, "_start", start)
    monkeypatch.setattr(app, "_launch", launch)
    monkeypatch.setattr(app, "_still_starting", lambda: [])
    return app, log, seen


def test_a_script_dying_before_its_first_step_gets_the_agent(tmp_path, monkeypatch):
    app, log, seen = _startup(tmp_path, ["applied"], monkeypatch, starts_after=1)
    assert headless._fix_startup(app, threading.Event()) is True
    log.close()
    assert len(seen["turns"]) == 1 and "KeyError: 'lr'" in seen["turns"][0] and seen["launches"] == 1
    assert "before its first training step" in (tmp_path / "log").read_text()


def test_a_restart_after_a_fix_that_dies_before_its_first_step_gets_the_agent(tmp_path, monkeypatch):
    # the old (crashed) run is still the open one: what matters is that the last start failed
    app, log, seen = _startup(tmp_path, ["applied"], monkeypatch, starts_after=1)
    app.console, app.session = object(), {"session_id": "old"}
    assert headless._fix_startup(app, threading.Event()) is True
    log.close()
    assert len(seen["turns"]) == 1 and seen["launches"] == 1 and app._start_failure is None


def test_a_failure_given_up_on_is_not_taken_up_again(tmp_path, monkeypatch):
    app, log, seen = _startup(tmp_path, ["answered"], monkeypatch)
    assert headless._fix_startup(app, threading.Event()) is False
    assert app._start_failure is None and headless._fix_startup(app, threading.Event()) is False
    log.close()
    assert len(seen["turns"]) == 1


def test_a_start_the_agent_asked_for_that_went_nowhere_is_not_success(tmp_path, monkeypatch):
    # the verification run: the agent's own start_run failed ("No such script: .../python"),
    # which cleared the failure but opened nothing -- Pulse must still start the script itself
    app, log, seen = _startup(tmp_path, ["applied"], monkeypatch, starts_after=1)
    real_start = app._start

    def start(work, request):
        real_start(work, request)
        app._start_failure = None                   # what _launch's "No such script" left behind

    monkeypatch.setattr(app, "_start", start)
    assert headless._fix_startup(app, threading.Event()) is True
    log.close()
    assert seen["launches"] == 1


@pytest.mark.parametrize("script, args, expected", [
    ("python train.py", "", ("train.py", "")),
    ("python", "train.py --epochs 3", ("train.py", "--epochs 3")),
    ("python3 -u train.py", "", ("train.py", "")),
    ("python -X dev train.py", "", ("train.py", "")),
    ("/usr/bin/python3.12", "train.py", ("train.py", "")),
    ("train.py", "--lr 0.1", ("train.py", "--lr 0.1")),
    ("python -m mypkg.train", "", ("python -m mypkg.train", "")),
])
def test_start_run_takes_the_command_as_a_model_writes_it(script, args, expected):
    assert appmod._without_interpreter(script, args) == expected


def test_no_change_means_no_relaunch(tmp_path, monkeypatch):
    app, log, seen = _startup(tmp_path, ["answered"], monkeypatch)
    assert headless._fix_startup(app, threading.Event()) is False
    log.close()
    assert seen["launches"] == 0 and "changed nothing" in (tmp_path / "log").read_text()


def test_a_fix_made_with_a_command_is_started_even_if_the_edit_was_declined(tmp_path, monkeypatch):
    # the agent fixed config.json with a shell command; its formal edit then came out empty and
    # the reviewer declined it: the file is fixed all the same, so the script is started again
    app, log, seen = _startup(tmp_path, ["edited with a command"], monkeypatch, starts_after=1)
    assert headless._fix_startup(app, threading.Event()) is True
    log.close()
    assert seen["launches"] == 1


def test_project_files_see_an_edit_and_not_pulse_bookkeeping(tmp_path):
    (tmp_path / "train.py").write_text("x = 1\n")
    before = headless._project_files(str(tmp_path))
    (tmp_path / ".pulse_fixlog.json").write_text("[]")
    (tmp_path / ".pulse_stream").mkdir()
    (tmp_path / ".pulse_stream" / "events").write_text("{}")
    assert headless._project_files(str(tmp_path)) == before
    (tmp_path / "train.py").write_text("x = 22\n")
    assert headless._project_files(str(tmp_path)) != before


def test_an_edit_the_files_already_have_is_not_sent_for_review():
    from pulse import pulse_code
    cli = types.SimpleNamespace(_review_change_with_approver=lambda *a: pytest.fail("reviewed a no-op"))
    outcome, _ = pulse_code._confirm_and_apply(cli, "fix", "plan", {}, {"/p/config.json": ("{}", "{}", False)})
    assert outcome == "answered"


def test_tries_are_bounded(tmp_path, monkeypatch):
    app, log, seen = _startup(tmp_path, ["applied"] * 20, monkeypatch)
    assert headless._fix_startup(app, threading.Event()) is False
    log.close()
    assert len(seen["turns"]) == headless._STARTUP_FIXES and "still exits" in (tmp_path / "log").read_text()


def test_without_an_agent_nothing_is_tried(tmp_path, monkeypatch):
    app, log = _headless(tmp_path)
    app._start_failure = {"script": "/p/train.py", "code": 1, "log": "/nonexistent", "argv": ["x"], "cwd": "."}
    monkeypatch.setattr(app, "_start", lambda *a: pytest.fail("an agent turn without an agent"))
    assert headless._fix_startup(app, threading.Event()) is False
    log.close()


# ---------------------------------------------------------------- the reviewer

def test_the_reviewer_is_told_an_easier_task_is_not_a_fix():
    prompt = approver.CHANGE_SYSTEM_PROMPT
    assert "makes the problem easier" in prompt and "how the result is measured" in prompt


# ---------------------------------------------------------------- "not learning", said too soon

def _never_learned(engine, values):
    history, out = [], []
    for v in values:
        history.append(v)
        out += [f for f in engine.update({"loss": list(history)}, step=len(history))["raised"]
                if f.check == "never_learned"]
    return out


def test_not_learning_seconds_in_is_a_warning_and_becomes_critical_with_evidence():
    # a binary classifier's batch loss sitting at chance: 17 readings is 17 steps (the benchmark
    # run where a CRITICAL at that point had the agent "fixing" a healthy run)
    flat = [0.693 + 0.004 * ((i * 7) % 5 - 2) for i in range(60)]
    early = _never_learned(_engine(), flat[:20])
    assert early and all(f.severity == detect.WARNING for f in early)
    later = _never_learned(_engine(), flat)
    assert [f.severity for f in later] == [detect.WARNING, detect.CRITICAL]      # escalated, once
    assert not detect.acts_on(early[0]) and detect.acts_on(later[-1])


# ---------------------------------------------------------------- the prompts, read from the model's seat

def test_a_flag_is_handed_over_as_a_claim_with_its_source(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    started = []
    app._start = lambda work, request: started.append(request)
    app.session = {"script": "/p/train.py", "session_id": "s1"}
    app._start_problem_turn([types.SimpleNamespace(check="audit", variable="audit@9", severity="critical",
                                                   message="train.py:51 shuffles y_val")])
    app._start_problem_turn([types.SimpleNamespace(check="never_learned", variable="loss", severity="critical",
                                                   message="loss is no better than when it started")])
    audit, rule = started
    assert "scheduled audit (a model" in audit and "checks (fixed rules" in rule
    for request in started:
        assert "claim to check, not a fact" in request and "change nothing" in request
        assert "found something serious" not in request


def test_headless_tells_the_agent_nobody_is_watching_and_the_app_does_not():
    assert "nobody is watching" in appmod.UNATTENDED_DEBUG_PROMPT
    assert "looking at it with you" not in appmod.UNATTENDED_DEBUG_PROMPT
    assert "looking at it with you" in appmod.DEBUG_PROMPT
    assert headless.HeadlessApp._debug_prompts == (appmod.UNATTENDED_DEBUG_PROMPT,
                                                    appmod.UNATTENDED_DEBUG_PROMPT_NATIVE)
    for prompt in (appmod.DEBUG_PROMPT, appmod.UNATTENDED_DEBUG_PROMPT):
        assert "a flag can be wrong" in prompt
    # both spellings of the run tools, whichever loop the agent is in
    assert "RESTART:" in headless.HeadlessApp._unattended_note and "restart_run" in headless.HeadlessApp._unattended_note


def test_diffs_reach_a_model_without_terminal_colours():
    prompt = approver.build_change_prompt("fix it", "drop the shuffle",
                                          "\x1b[1mmodified: train.py\x1b[0m\n\x1b[31m-    y = y[p]\x1b[0m", "/p")
    assert "\x1b" not in prompt and "[31m" not in prompt and "-    y = y[p]" in prompt


def test_the_progress_check_counts_a_restart_still_to_do():
    from pulse import pulse_code
    text = pulse_code._CONTINUE.format(request="x")
    assert "RESTART:" in text and "RUNSTATUS:" in text


def test_loop_bookkeeping_is_not_drawn_as_curves(tmp_path):
    b = _brain(tmp_path)
    total = []
    for _ in range(3):
        total += [1.7 * i for i in range(12)]
    b.histories = {"loss": [1.3 * 0.9 ** i for i in range(36)], "total": total, "started": [1.79e9],
                   "epoch": [float(i) for i in range(10)]}
    text = brain.Brain.render_evidence(b.evidence())
    assert "loss: 36 readings" in text
    assert "total: " not in text and "started: " not in text and "epoch: " not in text
    assert "Not drawn -- the loop's own bookkeeping" in text and "started (a timestamp)" in text


# ---------------------------------------------------------------- what ten players in the seat reported

def test_not_learning_measures_from_the_real_start():
    # converged within its first 30 readings of 300, then flat: it learned
    import random
    rng = random.Random(0)
    values = [0.674 - (0.674 - 0.338) * min(1.0, i / 30) + 0.004 * rng.random() for i in range(300)]
    assert not _never_learned(_engine(), values)


def test_one_hard_batch_is_a_warning_and_a_loss_that_stays_up_is_critical():
    calm = [1.0 * 0.85 ** i for i in range(40)] + [0.02] * 40
    one = _spikes(_engine(), calm + [0.6, 0.02, 0.02])
    assert one and all(f.severity == detect.WARNING for f in one)
    held = _spikes(_engine(), calm + [0.6, 0.7, 0.8])
    assert held[-1].severity == detect.CRITICAL


def test_a_cycle_in_the_loss_is_a_hypothesis_not_a_verdict():
    import math
    values = [1.0 - 0.002 * i + 0.05 * math.sin(2 * math.pi * i / 6) for i in range(120)]
    f = detect.DetectionEngine._check_periodicity("loss", values, 120)
    if f is not None:
        assert f.severity == detect.INFO and "not being shuffled" not in f.message and "can" in f.message


def test_a_saturated_metric_is_not_stagnant():
    engine = _engine()
    history = []
    found = []
    for v in [0.95, 0.99, 0.995] + [0.9995] * 20:
        history.append(v)
        found += [f for f in engine.update({"val_acc": list(history)}, step=len(history))["raised"]
                  if f.check == "metric_stagnation"]
    assert not found


def test_a_flattened_epoch_loss_stays_a_curve():
    import random
    rng = random.Random(3)
    plateau = [0.50, 0.40, 0.36] + [0.338 + 0.002 * rng.random() for _ in range(25)]
    assert brain._bookkeeping(plateau) is None


def test_the_audit_sees_the_config_the_script_reads(tmp_path):
    (tmp_path / "config.json").write_text('{"epochs": 40, "lr": 0.003}\n')
    (tmp_path / "train.py").write_text('import json\nCONFIG = json.load(open("config.json"))\n')
    b = _brain(tmp_path)
    b.session["script"] = str(tmp_path / "train.py")
    code = b._read_script()
    assert "config.json (read by the script)" in code and '"epochs": 40' in code


def test_pulse_s_own_probe_events_and_recheck_counts_are_not_evidence(tmp_path):
    b = _brain(tmp_path)
    b.events = [{"event": "probe_missing", "name": "x"}] * 5
    b._own_findings[("time_sink", "train.py:9")] = detect.Finding("time_sink", "train.py:9", detect.INFO, "m")
    b._own_findings[("time_sink", "train.py:9")].count = 3063
    text = brain.Brain.render_evidence(b.evidence())
    assert "probe_missing" not in text and "3063" not in text


def test_a_critical_that_keeps_coming_back_pulls_the_audit_once(tmp_path):
    b = _brain(tmp_path)
    critical = detect.Finding("never_learned", "loss", detect.CRITICAL, "x")
    b._fold = lambda frames: ([], [critical])
    b.schedule.set(3600, "chosen by the audit", "low")
    b.ingest([{}])
    assert b.schedule.seconds_remaining() <= brain.MIN_INTERVAL_SECONDS + 1
    b.schedule.set(3600, "chosen by the audit", "low")
    b.ingest([{}])
    assert b.schedule.seconds_remaining() > 3500


def test_no_scheduled_audit_while_the_agent_is_at_work(tmp_path):
    b = _brain(tmp_path)
    asked = []
    b.audit = lambda *a, **k: asked.append(1) or {"status": "ok"}
    b.schedule.next_at = time.time() - 1
    b.hold_audits = lambda: True
    b.poll_once()
    assert not asked
    b.hold_audits = lambda: False
    b.poll_once()
    assert asked


def test_a_flag_that_is_false_does_not_forbid_fixing_what_is_really_wrong(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    started = []
    app._start = lambda work, request: started.append(request)
    app.session = {"script": "/p/train.py", "session_id": "s1"}
    app._start_problem_turn([types.SimpleNamespace(check="loss_spike", variable="loss", severity="critical",
                                                   message="loss spiked")])
    assert "something else that is really wrong" in started[0] and "whichever you have" in started[0]


def test_directives_run_in_the_order_written_and_wait_waits(monkeypatch):
    from pulse import pulse_code
    from pulse import pulse_ui as ui
    order = []
    host = types.SimpleNamespace(run_actions=lambda: {"run_status": lambda: order.append("status") or "s"})
    monkeypatch.setattr(ui, "host", lambda: host)
    cli = types.SimpleNamespace(_service_tool_requests=lambda answer: order.append("terminal") or "t")
    slept = []
    monkeypatch.setattr(pulse_code.time, "sleep", lambda s: slept.append(s))
    pulse_code._service(cli, "TERMINAL: sleep 30\nRUNSTATUS:\n")
    assert order == ["terminal", "status"]
    order.clear()
    pulse_code._service(cli, "RUNSTATUS:\nTERMINAL: ls\n")
    assert order == ["status", "terminal"]
    notes = pulse_code._service_run_directives("WAIT: 120\nRUNSTATUS:\n")
    assert slept == [120.0] and notes[0] == "Waited 120 s."
    pulse_code._service_run_directives("WAIT: 99999\n")
    assert slept[-1] == pulse_code._WAIT_MAX_SECONDS


def test_fixing_is_not_building():
    from pulse import pulse_code
    assert "not a request to build" in pulse_code.CODE_SYSTEM_PROMPT


def test_a_running_total_of_a_falling_loss_is_still_bookkeeping():
    total = []
    for epoch in range(4):
        total += [(1.7 / (epoch + 1)) * i for i in range(12)]
    assert brain._bookkeeping(total) is not None


def test_the_audit_sees_the_project_modules_the_script_imports(tmp_path):
    (tmp_path / "data.py").write_text("def load_splits():\n    return 1\n")
    (tmp_path / "train.py").write_text("import json\nfrom data import load_splits\nimport shards\n")
    b = _brain(tmp_path)
    b.session["script"] = str(tmp_path / "train.py")
    code = b._read_script()
    assert "data.py (imported by the script)" in code and "def load_splits" in code
    assert "shards.py" not in code                     # not in the project: not read


def test_the_closing_audit_knows_the_run_is_over_and_hears_what_cleared(tmp_path):
    b = _brain(tmp_path)
    prompts = []
    b.agent = lambda prompt: prompts.append(prompt) or 'ok\n{"status": "ok", "next_check_minutes": 60}'
    b._raised_since_audit = ["[info] periodic on loss: loss repeats on a cycle of 5 readings"]
    b.audits = [{"status": "watch", "risk": "low", "findings": ["shards.fetch holds 99% of the time"]}]
    b.finished = True
    b.audit()
    text = prompts[-1]
    assert "This run has ended" in text and "no longer active" in text and "cycle of 5 readings" in text
    assert "watch (risk low) -- shards.fetch holds 99% of the time" in text
    assert "healthy in every way except the one that matters" not in text
    b.audit()
    assert "cycle of 5 readings" not in prompts[-1]    # told once


def test_a_restarted_run_is_told_what_changed_before_it_and_not_that_its_code_is_newer():
    pack = {"step": 5, "session": {}, "code": "   1 | lr = 0.05",
            "fixes_applied": [{"step": 40, "summary": "removed the y_val shuffle", "before_restart": True}],
            "previous_audits": [{"status": "problem, on the run before the restart", "risk": "high",
                                 "findings": ["train.py:51 shuffles y_val"]}]}
    text = brain.Brain.render_evidence(pack)
    assert "started again after these changes" in text and "removed the y_val shuffle" in text
    assert "AS IT IS NOW" not in text                      # it runs this code
    assert "problem, on the run before the restart (risk high) -- train.py:51 shuffles y_val" in text


# ---------------------------------------------------------------- a reviewer's denial is answered

def _pipeline(monkeypatch, outcomes):
    from pulse import pulse_code
    cli = types.SimpleNamespace(compact_history=lambda: None, _last_applied_fix=None,
                                _sync_agent_turn=lambda *a, **k: None)
    asked = []

    def step(cli_, request, evidence=None):
        asked.append(request)
        outcome, denial = outcomes.pop(0)
        cli_._reviewer_declined = denial
        return outcome, ""
    monkeypatch.setattr(pulse_code, "_run_step", step)
    monkeypatch.setattr(pulse_code, "_next_step", lambda *a: None)
    return pulse_code._run_turn_pipeline(cli, "fix the crash"), asked


def test_a_change_the_reviewer_denied_goes_back_to_the_agent_with_the_reason(monkeypatch):
    outcome, asked = _pipeline(monkeypatch, [("declined", "the reviewer declined it: the model is on GPU"),
                                             ("applied", "")])
    assert outcome == "applied" and len(asked) == 2
    assert "the model is on GPU" in asked[1] and "can be wrong" in asked[1]


def test_a_person_declining_ends_the_turn(monkeypatch):
    outcome, asked = _pipeline(monkeypatch, [("declined", "")])
    assert outcome == "declined" and len(asked) == 1


def test_denials_are_answered_a_bounded_number_of_times(monkeypatch):
    outcome, asked = _pipeline(monkeypatch, [("declined", "no")] * 10)
    assert outcome == "declined" and len(asked) == 4


# ---------------------------------------------------------------- a verdict taken back; a run that ended wrong

def test_no_changes_taken_back_by_a_plan_is_not_the_verdict():
    from pulse import pulse_code
    taken_back = ("The lr is negative.\nNO_CHANGES\nWait -- the request asks for a fix, so I do need to change "
                  "the config.\n**Plan**\n- config.json line 9: \"lr\": -0.003 -> 0.003.\n" + "Then restart it. " * 30)
    assert not pulse_code._says_no_changes(taken_back)
    assert pulse_code._says_no_changes("The run is healthy: val_acc 0.91 at epoch 12.\nNO_CHANGES")
    assert pulse_code._says_no_changes("Nothing to change.\nNO_CHANGES\nThe curve matches the clean one.")
    assert not pulse_code._says_no_changes("Plan: change lr.")


def test_a_closing_audit_problem_is_a_claim_for_the_agent_unless_low_risk():
    assert headless._closing_problem({"status": "problem", "risk": "high", "findings": ["lr = -0.003 in config.json"]}) \
        == "lr = -0.003 in config.json"
    assert headless._closing_problem({"status": "problem", "risk": "low", "findings": ["x"]}) == ""
    assert headless._closing_problem({"status": "ok"}) == "" and headless._closing_problem(None) == ""


def test_a_problem_with_a_run_that_ended_says_it_ended(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    started = []
    app._start = lambda work, request: started.append(request)
    app.session, app._status = {"script": "/p/train.py", "session_id": "s1"}, "crashed"
    app._start_problem_turn([types.SimpleNamespace(check="audit", variable="audit@0", severity="critical",
                                                   message="lr = -0.003")], over=True)
    assert started[0].startswith("train.py has ended (crashed).") and "still running" not in started[0]
    assert "only shows in a new run" in started[0] and "claim to check" in started[0]


def test_headless_hands_over_a_problem_with_a_run_that_ended(tmp_path):
    # was: HeadlessApp's own _start_problem_turn did not take `over`, and the supervisor died
    # with a TypeError the moment a closing audit found a problem
    app, log = _headless(tmp_path, AgentCli())
    started = []
    app._start = lambda work, request: started.append(request)
    app.session, app._status = {"script": "/p/train.py", "session_id": "s1"}, "finished"
    app._start_problem_turn([types.SimpleNamespace(check="audit", variable="audit@0", severity="critical",
                                                   message="epochs=0: no training step ran")], over=True)
    log.close()
    assert started and started[0].startswith("train.py has ended (finished).")
    assert "epochs=0" in (tmp_path / "log").read_text()
