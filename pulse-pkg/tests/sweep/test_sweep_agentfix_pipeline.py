"""Bug sweep: the agent's investigate-and-fix pipeline in pulse_cli.py.

test_bug_* assert the CORRECT behaviour (they fail on the current code);
test_ok_* are regression protection for behaviour verified to work.
No real LLM calls: _call_model / litellm.completion are replaced.
"""
import json
import math
import os
import subprocess
import sys
import textwrap
import types

import pytest

os.environ.setdefault("PULSE_LOGGING", "0")

import pulse.pulse_cli as pc  # noqa: E402
from pulse.pulse_cli import PulseCLI  # noqa: E402
from pulse import pulse_terminal  # noqa: E402


def _no_network_completion(**kw):
    raise RuntimeError("real litellm.completion call attempted in a test")


# Never let any code path (including background retry threads) reach a provider.
pc.litellm.completion = _no_network_completion


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_cli(tmp_path, code="x = 1\n", extra=None, replies=None):
    """A real PulseCLI pointed at a script in tmp_path. If `replies` is given,
    _call_model answers from it (a list, or a callable(instruction) -> str)."""
    cli = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdf"))
    script = tmp_path / "train.py"
    script.write_text(code)
    cli.script_path = str(script)
    cli.code_text = code
    cli.extra_files = {}
    for rel, text in (extra or {}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        cli.extra_files[str(p)] = text
    cli._project_root = str(tmp_path)
    cli._repo_cwd = str(tmp_path)
    cli.agent_provider = "openai"
    cli.agent_key = "sk-test"
    cli.agent_model_string = "openai/test-model"
    cli.agent_api_base = None
    cli._suppress_auto_restart = True
    cli.sweep_enabled = False
    cli.review = False   # never prompt for TERMINAL confirmation in tests
    cli._queue_agent_retry = lambda *a, **k: None
    cli._ensure_retry_ticker = lambda: None
    # PulseCLI.__init__ registers an atexit end-of-run review that escalates to the
    # agent when no step was trained; never let a test instance do that.
    cli._end_review_done = True
    cli._build_file_labels()
    if replies is not None:
        cli.calls = []
        queue = list(replies) if isinstance(replies, list) else None

        def call_model(instruction, max_tokens=0, **kw):
            cli.calls.append(instruction)
            if queue is not None:
                return queue.pop(0)
            return replies(instruction)

        cli._call_model = call_model
    return cli


def stub_context(cli, marker="CONTEXT-MARKER"):
    cli._build_agent_context = lambda include_code=False: f"{marker}\n{cli.code_text}"


# ==========================================================================
# BUGS
# ==========================================================================

# ---- code-fix JSON parsing ------------------------------------------------

def test_bug_parse_code_fix_drops_resume_field(tmp_path):
    """SYSTEM_PROMPT tells the model to send "resume": false when the weights
    carry the bug (init/architecture/data fixes), and _apply_code_fix honours
    fix["resume"] -- but _parse_code_fix rebuilds the dict from old/new/files/
    explanation only, so "resume" never survives parsing and every fix resumes
    from the (poisoned) checkpoint. Correct: the parsed fix keeps resume, and
    applying it sets _resume_after_fix = False."""
    code = "w_init = 'zeros'\n"
    cli = make_cli(tmp_path, code)
    cli._resume_after_fix = True
    answer = json.dumps({"old": ["w_init = 'zeros'"], "new": ["w_init = 'he_normal'"],
                         "explanation": "zero init", "resume": False})
    fix = PulseCLI._parse_code_fix(answer)
    assert fix is not None
    cli._apply_code_fix(fix)
    assert cli._resume_after_fix is False


def test_bug_parse_code_fix_rejects_json_after_prose():
    """A reply like 'Here is the fix:\\n{...}' is rejected (text must start with
    '{' and end with '}'), unlike _parse_json_obj which finds the object. The
    pass-3 loop then nags the model with _PASS3_NO_TOOLS_NOTE and can end with
    no fix applied. Correct: extract the JSON object from surrounding prose."""
    answer = 'Here is the fix:\n{"old": ["a = 1"], "new": ["a = 2"], "explanation": "x"}\nDone.'
    fix = PulseCLI._parse_code_fix(answer)
    assert fix is not None and fix["new"] == ["a = 2"]


def test_bug_parse_code_fix_rejects_string_files_field():
    """"files": "model.py" (a single string for a single change) makes the
    standard-shape branch return None, discarding a valid fix -- while the
    fallback branch already accepts a string. Correct: treat a string as a
    one-element list."""
    answer = json.dumps({"old": ["a = 1"], "new": ["a = 2"], "files": "model.py",
                         "explanation": "x"})
    fix = PulseCLI._parse_code_fix(answer)
    assert fix is not None and fix["files"] == ["model.py"]


def test_bug_parse_json_obj_gives_up_on_braces_in_prose():
    """_parse_json_obj takes only the FIRST balanced {...}; a verdict preceded
    by prose that quotes code with braces (a dict literal, an f-string) yields
    None. Correct: try later spans until one parses."""
    text = 'I checked `cfg = {lr: 1}` first.\n{"passes": false, "reason": "wrong axis"}'
    assert PulseCLI._parse_json_obj(text) == {"passes": False, "reason": "wrong axis"}


# ---- PASS 4 verification ------------------------------------------------

def test_bug_verify_treats_unparsable_failing_verdict_as_pass(tmp_path):
    """The verifier said passes=false, but its reply had prose with braces
    first, so _parse_json_obj returned None and _verify_fix_with_retries
    returned passed=True ('unparsable; proceeding anyway') -- a fix the
    verifier rejected gets applied. Correct: the failing verdict is seen,
    and after the agent drops it the fix is not passed."""
    replies = [
        'I checked `cfg = {lr: 1}` first.\n{"passes": false, "reason": "wrong axis"}',
        '{"decision": "drop", "reason": "the fix is wrong"}',
    ]
    cli = make_cli(tmp_path, replies=replies)
    fix = {"old": ["a"], "new": ["b"], "files": [None], "explanation": ""}
    _fix, passed, _reason = cli._verify_fix_with_retries(fix, "diag")
    assert passed is False


def test_bug_verify_string_false_counts_as_pass(tmp_path):
    """verdict "passes": "false" (a quoted boolean, which models do emit) is
    read with bool(...), and bool("false") is True -- a failed check passes.
    Correct: interpret string booleans."""
    replies = [
        '{"passes": "false", "reason": "divides by the wrong axis"}',
        '{"decision": "drop", "reason": "wrong"}',
    ]
    cli = make_cli(tmp_path, replies=replies)
    fix = {"old": ["a"], "new": ["b"], "files": [None], "explanation": ""}
    _fix, passed, _reason = cli._verify_fix_with_retries(fix, "diag")
    assert passed is False




def test_bug_confirm_fix_string_false_counts_as_resolved(tmp_path):
    """PASS 6 (_confirm_fix_did_its_job) uses bool(verdict["resolved"]), so
    "resolved": "false" reports the fix as having done its job and the
    escalation to the full pipeline never happens."""
    cli = make_cli(tmp_path, replies=['{"resolved": "false", "reason": "loss still NaN"}'])
    cli._last_applied_fix = {"old": ["a"], "new": ["b"], "files": [None], "explanation": ""}
    cli._last_problem_description = "loss NaN"
    result = types.SimpleNamespace(stdout="loss nan\n", stderr="")
    resolved, _reason = cli._confirm_fix_did_its_job(result)
    assert resolved is False


# ---- the pipeline ---------------------------------------------------------

def test_bug_history_window_drops_code_and_question(tmp_path, monkeypatch):
    """_call_model sends only agent_history[-10:]. The question + context (the
    code, the traceback) is appended once at the start; each PASS 1/2 tool
    round appends two more messages, so after five rounds the model is asked
    to diagnose/fix without ever seeing the code or the question. Correct:
    the original question/context message is in every call of the pipeline."""
    seen = []

    def fake_completion(**kw):
        msgs = kw["messages"]
        seen.append(msgs)
        instr = msgs[-1]["content"]
        n_pass1 = sum(1 for m in seen if m[-1]["content"].startswith("PASS 1"))
        n_pass2 = sum(1 for m in seen if "PASS 2" in m[-1]["content"])
        if instr.startswith("PASS 1"):
            text = "GREP: loss" if n_pass1 <= 3 else "- train.py:1"
        elif "PASS 2" in instr:
            text = "GREP: lr" if n_pass2 <= 3 else "Diagnosis: lr too high."
        else:
            text = "Lower lr."
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))], usage=None)

    monkeypatch.setattr(pc.litellm, "completion", fake_completion)
    monkeypatch.setattr(pc, "_clamp_output_tokens", lambda m, r: r)
    cli = make_cli(tmp_path, "lr = 10.0\nloss = 1.0\n")
    stub_context(cli)
    cli._ask_agent_impl("why is the loss NaN?", include_code=True)
    assert seen
    missing = [i for i, msgs in enumerate(seen)
               if not any("CONTEXT-MARKER" in (m.get("content") or "") for m in msgs)]
    assert not missing, f"calls {missing} of {len(seen)} were sent without the question/code"


def test_bug_pass2_tool_exception_crashes_pipeline(tmp_path):
    """PASS 2 calls _apply_new_directives directly (no try/except, unlike
    _service_tool_requests). HISTOGRAM on a history that holds NaN raises
    ValueError, which escapes ask_agent -- on an auto-intervention that is an
    exception thrown into the user's training loop. Correct: a failing tool
    becomes an error note, the pipeline finishes."""
    replies = ["- train.py:1", "HISTOGRAM: loss", "Diagnosis: diverged.", "Lower lr."]
    cli = make_cli(tmp_path, "loss = 1.0\n", replies=replies)
    stub_context(cli)
    cli.scalar_histories = {"loss": [1.0, 0.9, float("nan")]}
    out = cli._ask_agent_impl("why is the loss NaN?", include_code=True)
    assert isinstance(out, str)


def test_bug_question_substring_triggers_code_edit(tmp_path):
    """wants_implementation is a substring test over _IMPLEMENT_KEYWORDS, so a
    plain question that merely contains 'prefix' / 'application' / 'credit'
    is treated as a request to edit the user's code (pass 3 goes to the
    code-fix JSON path and writes to disk). Correct: whole-word match."""
    def reply(instr):
        if instr.startswith("PASS 1"):
            return "- train.py:1"
        if "PASS 2" in instr:
            return "Diagnosis: it prepends learned tokens."
        return "It prepends learned tokens."
    cli = make_cli(tmp_path, "x = 1\n", replies=reply)
    stub_context(cli)
    cli._ask_agent_impl("What does the prefix tuning layer do here?", include_code=True)
    assert not any("DEVELOP & IMPLEMENT" in c for c in cli.calls)


def test_bug_pass3_has_no_way_to_decline_a_change(tmp_path):
    """Every auto-intervention question contains 'fix', so PASS 3 is the
    IMPLEMENT pass. If the analysis finds nothing wrong and the model says so,
    the answer isn't code-fix JSON, and the loop re-prompts it up to three
    more times with _PASS3_NO_TOOLS_NOTE ('respond with ONLY the code-fix
    JSON ... fixing the bug') -- the pipeline pushes the model to change a
    healthy program (seen in the benchmark as rewritten datasets / removed
    class weights). Correct: a 'no change needed' answer ends pass 3."""
    def reply(instr):
        if instr.startswith("PASS 1"):
            return "- train.py:1"
        if "PASS 2" in instr:
            return "Diagnosis: nothing is wrong; the run is healthy and converging."
        return "No code change is needed: the program is correct."
    cli = make_cli(tmp_path, "x = 1\n", replies=reply)
    stub_context(cli)
    cli._ask_agent_impl("Pulse just auto-paused training because it detected a problem: "
                        "plateau\nPlease diagnose the root cause and, if you can, fix it.",
                        include_code=True)
    implement_calls = [c for c in cli.calls if "DEVELOP & IMPLEMENT" in c]
    assert len(implement_calls) == 1, f"pass 3 re-asked {len(implement_calls)} times for a fix"


def test_bug_service_tool_requests_ignores_sensitivity_and_gputrack(tmp_path):
    """_service_tool_requests (used by PASS 1, 3 and 4) only applies the old
    directives when calc/promote/grep/view are present; a reply with just
    SENSITIVITY: or GPUTRACK: is silently dropped. Correct: apply them like
    PASS 2 does."""
    cli = make_cli(tmp_path, "x = 1\n")
    applied = []
    cli._cmd_sensitivity = lambda arg, quiet=False: applied.append(arg) or f"sensitivity {arg}"
    note = cli._service_tool_requests("SENSITIVITY: tight")
    assert applied == ["tight"] and note


# ---- directive extraction / tools ---------------------------------------

def test_bug_terminal_heredoc_is_cut_to_first_line():
    """TERMINAL: python3 - <<'EOF' / script / EOF captures only the first line;
    the body is dropped and bash runs python3 with an empty stdin, exit 0, no
    output -- the model is told its check 'succeeded'. The main SYSTEM_PROMPT
    (unlike the check-in prompt) never says TERMINAL is one line only.
    Correct: capture the heredoc body (or refuse it / document it)."""
    text = "TERMINAL: python3 - <<'EOF'\nimport sys\nprint('checked')\nEOF\n"
    _cleaned, req = PulseCLI._extract_new_directives(text)
    cmd = req["terminal"][0]
    assert "print('checked')" in cmd or "one line" in pc.SYSTEM_PROMPT.lower()


def test_bug_terminal_timeout_suffix_quotes_shell_operators():
    """The advertised ' --timeout=<s>' suffix makes parse_inline_timeout
    re-join shlex tokens with shlex.join, which quotes '|', '&&', '>' and
    globs -- 'grep x f | head --timeout=30' becomes "grep x f '|' head", a
    broken command. Correct: strip only the suffix, keep the command text."""
    cmd, timeout = pulse_terminal.parse_inline_timeout("grep -n loss train.py | head -5 --timeout=30")
    assert timeout == 30.0
    assert cmd == "grep -n loss train.py | head -5"


def test_bug_bare_flag_directive_swallows_next_line():
    """Flag-style directives use r'^\\s*NAME:\\s*(.*)$'; the \\s* after the colon
    also matches the newline, so a bare 'SHAPETRACE:' / 'LAYERSTATS:' /
    'DEPGRAPH:' takes the NEXT line of prose as its argument (SHAPETRACE then
    looks for a variable called 'The shapes look off.'), and that prose line
    is deleted from the printed/stored analysis. Correct: [ \\t]* after the
    colon; the argument is empty and the next line survives."""
    cleaned, req = PulseCLI._extract_new_directives("SHAPETRACE:\nThe shapes look off.")
    assert req.get("shapetrace") == [""]
    assert "The shapes look off." in cleaned


def test_bug_calc_sandbox_escape():
    """_safe_eval_math claims 'no builtins, no attribute access beyond that,
    so this is safe to eval() directly', but empty __builtins__ does not stop
    dunder attribute walks: ().__class__.__mro__[1].__subclasses__() reaches
    every loaded class (subprocess.Popen, os._wrap_close, ...). CALC: lines
    come from model output, which can be steered by code comments/data.
    Correct: AST-whitelist numbers/operators/math calls; reject attributes."""
    result = pc._safe_eval_math("().__class__.__mro__[1].__subclasses__()")
    assert isinstance(result, str) and "error" in result


def test_bug_calc_huge_power_hangs():
    """CALC: 9**9**9**9 never returns (big-int pow holds the GIL), freezing
    the agent and, in-process, training. Correct: bound exponent size."""
    code = ("import os; os.environ['PULSE_LOGGING']='0'\n"
            "from pulse.pulse_cli import _safe_eval_math\n"
            "print(_safe_eval_math('9**9**9**9'))\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    try:
        subprocess.run([sys.executable, "-c", code], env=env, timeout=20,
                       capture_output=True)
    except subprocess.TimeoutExpired:
        pytest.fail("CALC: 9**9**9**9 did not return within 20 s")


def test_bug_histogram_crashes_on_nan(tmp_path):
    """HISTOGRAM on a history containing NaN raises ValueError (int(nan)) --
    exactly when the loss has diverged and the model asks for it."""
    cli = make_cli(tmp_path)
    cli.scalar_histories = {"loss": [1.0, 0.5, float("nan")]}
    out = cli._run_histogram("loss")
    assert isinstance(out, str)


def test_bug_corr_crashes_on_none_in_history(tmp_path):
    """update() appends None to a scalar history when the variable is
    unreadable; CORR then does arithmetic on None and raises TypeError."""
    cli = make_cli(tmp_path)
    cli.scalar_histories = {"loss": [1.0, None, 0.8, 0.7], "lr": [0.1, 0.1, 0.2, 0.3]}
    out = cli._run_corr("loss lr")
    assert isinstance(out, str)


def test_bug_outlier_blind_to_nan(tmp_path):
    """OUTLIER with a NaN in the history computes mean=std=nan and reports
    'no points with |z| > 3' -- the diverged point is exactly the outlier.
    Correct: flag non-finite points (or say they exist)."""
    cli = make_cli(tmp_path)
    cli.scalar_histories = {"loss": [1.0, 0.9, 0.8, 0.7, float("nan")]}
    out = cli._run_outlier("loss")
    assert "no points" not in out


def test_bug_changelog_first_call_is_always_empty(tmp_path):
    """CHANGELOG's docstring says the baseline 'starts at this session's code',
    but the baseline is only taken at the FIRST CHANGELOG call, so after a
    fix changes the code the first CHANGELOG reports 'no prior checkpoint'
    and shows nothing. Correct: baseline = code at session start."""
    cli = make_cli(tmp_path, "lr = 10.0\n")
    cli.code_text = "lr = 0.01\n"      # what _apply_code_fix does after writing a fix
    out = cli._run_changelog()
    assert "lr = 0.01" in out


def test_bug_defof_misses_annotated_assignment(tmp_path):
    """DEFOF only looks at FunctionDef/ClassDef/plain Assign, so the
    definition of `LR: float = 10.0` (or `self.lr = ...`) is 'not found'."""
    cli = make_cli(tmp_path, "LR: float = 10.0\nprint(LR)\n")
    out = cli._run_defof("LR")
    assert "no definition found" not in out


# ---- applying fixes --------------------------------------------------------

def test_bug_fuzzy_match_eats_trailing_newline(tmp_path):
    """When a snippet only matches whitespace-insensitively, the replaced span
    runs through the last line's '\\n', but `new` has no trailing newline, so
    the next line is glued onto the fix ('    b = 3    return a'). The lint
    gate then rejects it and the fix silently never lands. Correct: keep the
    line structure (and re-indent `new` to the matched block)."""
    code = "def f():\n    a = 1\n    b = 2\n    return a\n\nprint(f())\n"
    cli = make_cli(tmp_path, code)
    fix = {"old": ["a = 1\nb = 2"], "new": ["    a = 1\n    b = 3"], "files": [None],
           "explanation": "x"}
    cli._apply_code_fix(fix)
    assert (tmp_path / "train.py").read_text() == code.replace("b = 2", "b = 3")




def test_bug_lint_gate_blocks_fix_on_preexisting_undefined_name(tmp_path):
    """The lint gate blocks any file whose FULL content has a pyflakes
    'undefined name', including one that was already there (a notebook export
    calling display(), a name injected via globals()). Every fix to such a
    file is refused though the fix itself is fine. Correct: block only on
    messages the fix introduced."""
    pytest.importorskip("pyflakes")
    code = "lr = 10.0\ndisplay(lr)\n"
    cli = make_cli(tmp_path, code)
    fix = {"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "x"}
    cli._apply_code_fix(fix)
    assert (tmp_path / "train.py").read_text().startswith("lr = 0.01")


def test_bug_snippet_matches_inside_a_longer_token(tmp_path):
    """content.count(old) is a raw substring count: old 'lr = 0.1' matches
    inside 'lr = 0.15', and the replacement silently yields 'lr = 0.015'.
    Correct: a snippet must match at line boundaries (else it's a miss and
    the re-quote retry runs)."""
    cli = make_cli(tmp_path, "lr = 0.15\n")
    fix = {"old": ["lr = 0.1"], "new": ["lr = 0.01"], "files": [None], "explanation": "x"}
    cli._apply_code_fix(fix)
    assert (tmp_path / "train.py").read_text() != "lr = 0.015\n"


def test_bug_corrected_snippet_retry_loses_target_file(tmp_path):
    """_request_corrected_snippets asks for corrected old/new for snippets that
    missed in, e.g., model.py; if the reply omits 'files' (the prompt only
    shows the file content), the corrected fix defaults to the MAIN script.
    Correct: default each corrected entry to the file its miss came from."""
    cli = make_cli(tmp_path, "x = 1\n", extra={"model.py": "h = 3\n"},
                   replies=['{"old": ["h = 3"], "new": ["h = 4"], "explanation": "x"}'])
    model_path = str(tmp_path / "model.py")
    fix = cli._request_corrected_snippets(
        {"old": ["h=3"], "new": ["h = 4"], "files": ["model.py"], "explanation": "x"},
        [("h=3", model_path, "no exact match found in the file")])
    assert fix is not None
    assert cli._resolve_fix_path(fix["files"][0]) == model_path


def test_bug_revert_of_created_file_leaves_empty_file(tmp_path):
    """A fix that CREATES a file records before=''. /revert then writes ''
    back instead of deleting the file, leaving an empty module (an empty
    helper.py still shadows / breaks imports). Correct: remove it."""
    cli = make_cli(tmp_path, "x = 1\n")
    fix = {"old": [], "new": [], "files": [], "explanation": "add helper",
           "create": [{"path": "helper.py", "content": "HELP = 1\n"}]}
    cli._apply_code_fix(fix)
    assert (tmp_path / "helper.py").exists()
    cli._cmd_revert("")
    assert not (tmp_path / "helper.py").exists()


def test_bug_lint_failure_reported_as_snippet_mismatch(tmp_path):
    """When every snippet matched but the result failed the lint gate, the
    report still says 'none of the proposed snippets matched cleanly', and
    _last_apply_skipped is [] -- so the one corrective retry
    (_request_corrected_snippets) never runs and the lint error is never
    shown to the model; the pipeline ends with the bug unfixed. Correct:
    report the lint failure as such (and give the model a chance to fix it)."""
    cli = make_cli(tmp_path, "x = 1\n")
    out = cli._apply_code_fix({"old": ["x = 1"], "new": ["x = (1"], "files": [None], "explanation": "x"})
    assert "none of the proposed snippets matched" not in out


def test_bug_rollback_error_advertises_nonexistent_log_directive(tmp_path):
    """ROLLBACK with an unknown id tells the model to 'send LOG' to see the
    commits, but there is no LOG: directive (_NEW_DIRECTIVE_RES has none), so
    the model's LOG: line is ignored and it can never list commit ids."""
    cli = make_cli(tmp_path, "lr = 10.0\n")
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "a"})
    out = cli._run_rollback("zzzz")
    _c, req = PulseCLI._extract_new_directives("LOG:")
    assert "LOG" not in out or req


# ---- MLLINT and its auto-fix ------------------------------------------------

HEALTHY_TORCH = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    model = nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    def train_step(x, y):
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        acc = (torch.softmax(logits, dim=1).argmax(1) == y).float().mean()
        return loss.item(), acc.item()
""")


def scan(tmp_path, code):
    cli = make_cli(tmp_path, code)
    return pc._mllint_scan(cli._iter_ast_trees())


def test_bug_mllint_double_softmax_false_positive(tmp_path):
    """Any softmax call anywhere + cross_entropy anywhere => 'double softmax',
    even when softmax is only used for accuracy on the side. At start of run
    that triggers a forced agent fix of a healthy program."""
    findings = scan(tmp_path, HEALTHY_TORCH)
    assert not [f for f in findings if "double softmax" in f[2]]


def test_bug_mllint_eval_name_heuristic_fires_on_keras(tmp_path):
    """Any function named *eval*/*valid*/*test* that calls anything and has no
    no_grad()/.eval() is flagged -- including Keras/numpy code where neither
    exists (model.evaluate)."""
    code = "def evaluate_model(model, x, y):\n    return model.evaluate(x, y, verbose=0)\n"
    assert scan(tmp_path, code) == []


def test_bug_mllint_seed_sweep_flagged_as_reseed_in_loop(tmp_path):
    """`for seed in seeds: np.random.seed(seed)` is a multi-seed experiment,
    not a per-step reseed, but it's flagged (and then force-'fixed')."""
    code = ("import numpy as np\n"
            "def run():\n    return np.random.rand()\n"
            "for seed in [0, 1, 2]:\n    np.random.seed(seed)\n    run()\n")
    assert scan(tmp_path, code) == []


def test_bug_mllint_numpy_loss_accumulation_flagged(tmp_path):
    """`total_loss += batch_loss` with a numpy float is flagged as keeping the
    autodiff graph alive, in a script with no autodiff framework at all."""
    code = ("import numpy as np\nw = np.zeros(3)\ntotal_loss = 0.0\n"
            "for xb, yb in [(np.ones((2, 3)), np.ones(2))]:\n"
            "    batch_loss = np.mean((xb @ w - yb) ** 2)\n"
            "    total_loss += batch_loss\n")
    assert scan(tmp_path, code) == []


def test_bug_start_of_run_forces_fix_for_self_described_false_positive(tmp_path):
    """_prime_at_start sends EVERY MLLINT finding to ask_agent with 'diagnose
    AND fix this now ... write a code fix, apply it', including findings whose
    own text says 'this is a naming heuristic, so it may be a false positive'
    and although SYSTEM_PROMPT calls MLLINT 'heuristic -- worth
    double-checking'. Correct: heuristic findings are investigated, not
    ordered fixed."""
    code = "def evaluate_model(model, x, y):\n    return model.evaluate(x, y, verbose=0)\n"
    cli = make_cli(tmp_path, code)
    asked = []
    cli.ask_agent = lambda q, include_code=False, **kw: asked.append(q) or ""
    cli._start_start_prime = lambda deferred: None
    cli._poll_start_prime = lambda: None
    cli._start_primed = False
    cli._prime_at_start()
    assert not [q for q in asked if "fix this now" in q or "apply it" in q]


def _two_model_script():
    return textwrap.dedent("""\
        clf = reg = None
        clf.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        reg.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    """)


def test_bug_mllint_auto_fix_rewrites_every_compile(tmp_path):
    """The regression+accuracy auto-fix is a file-wide regex over every
    'metrics=[...]', so the classifier's correct metrics=['accuracy'] is
    also changed to 'mae'. Correct: patch only the flagged compile() call."""
    code = _two_model_script()
    cli = make_cli(tmp_path, code)
    findings = pc._mllint_scan(cli._iter_ast_trees())
    assert findings
    cli._mllint_auto_fix(findings)
    first = (tmp_path / "train.py").read_text().splitlines()[1]
    assert 'metrics=["accuracy"]' in first


def test_bug_mllint_auto_fix_not_revertible(tmp_path):
    """The deterministic start-of-run fix writes the user's file but records it
    only in .pulse_fixlog.json (PASTFIX), not in .pulse_history -- /log does
    not show it and /revert cannot undo it. Correct: record a commit."""
    code = 'reg = None\nreg.compile(loss="mse", optimizer="adam", metrics=["accuracy"])\n'
    cli = make_cli(tmp_path, code)
    assert cli._mllint_auto_fix(pc._mllint_scan(cli._iter_ast_trees()))
    assert cli._load_fix_log(), "auto-fix left no .pulse_history commit"


def test_bug_mllint_auto_fix_leaves_code_text_stale(tmp_path):
    """After the auto-fix rewrites the file, self.code_text still holds the old
    code, so the agent that is asked right after to 'verify the deterministic
    patch' is shown the unpatched code (and its snippets then fail to match
    the file on disk)."""
    code = 'reg = None\nreg.compile(loss="mse", optimizer="adam", metrics=["accuracy"])\n'
    cli = make_cli(tmp_path, code)
    assert cli._mllint_auto_fix(pc._mllint_scan(cli._iter_ast_trees()))
    assert cli.code_text == (tmp_path / "train.py").read_text()


# ==========================================================================
# OK -- verified behaviour, regression protection
# ==========================================================================

def test_ok_exact_fix_applies_and_logs_commit(tmp_path):
    cli = make_cli(tmp_path, "lr = 10.0\nepochs = 3\n")
    fix = {"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "lower lr"}
    out = cli._apply_code_fix(fix)
    assert (tmp_path / "train.py").read_text() == "lr = 0.01\nepochs = 3\n"
    assert cli.code_text == "lr = 0.01\nepochs = 3\n"
    assert cli._fix_applied_this_turn is True
    entries = cli._load_fix_log()
    assert len(entries) == 1 and entries[0]["files"][0]["before"] == "lr = 10.0\nepochs = 3\n"
    assert "Logged as commit" in out


def test_ok_revert_restores_original_and_logs_revert(tmp_path):
    cli = make_cli(tmp_path, "lr = 10.0\n")
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "x"})
    cli._cmd_revert("")
    assert (tmp_path / "train.py").read_text() == "lr = 10.0\n"
    entries = cli._load_fix_log()
    assert [e["kind"] for e in entries] == ["fix", "revert"]
    assert cli.code_text == "lr = 10.0\n"


def test_ok_multi_file_fix_uses_labels(tmp_path):
    cli = make_cli(tmp_path, "import model\n", extra={"pkg/model.py": "hidden = 1\n"})
    fix = {"old": ["hidden = 1"], "new": ["hidden = 64"], "files": ["pkg/model.py"], "explanation": "x"}
    cli._apply_code_fix(fix)
    assert (tmp_path / "pkg" / "model.py").read_text() == "hidden = 64\n"
    assert (tmp_path / "train.py").read_text() == "import model\n"


def test_ok_ambiguous_snippet_is_skipped(tmp_path):
    cli = make_cli(tmp_path, "x = 1\nx = 1\n")
    cli._last_crash_location = None
    cli._apply_code_fix({"old": ["x = 1"], "new": ["x = 2"], "files": [None], "explanation": "x"})
    assert (tmp_path / "train.py").read_text() == "x = 1\nx = 1\n"
    assert "ambiguous" in cli._last_apply_skipped[0][2]


def test_ok_lint_gate_blocks_syntax_error(tmp_path):
    cli = make_cli(tmp_path, "x = 1\n")
    cli._apply_code_fix({"old": ["x = 1"], "new": ["x = (1"], "files": [None], "explanation": "x"})
    assert (tmp_path / "train.py").read_text() == "x = 1\n"
    assert cli._last_apply_lint_failed


def test_ok_create_file_outside_root_refused(tmp_path):
    cli = make_cli(tmp_path, "x = 1\n")
    skipped = []
    assert cli._create_file_for_fix({"path": "../evil.py", "content": "x=1"}, skipped) is None
    assert "outside the project root" in skipped[0][2]


def test_ok_parse_code_fix_shapes():
    std = PulseCLI._parse_code_fix('```json\n{"old": ["a"], "new": ["b"], "explanation": "e"}\n```')
    assert std == {"old": ["a"], "new": ["b"], "files": [None], "explanation": "e"}
    flat = PulseCLI._parse_code_fix(json.dumps({"old": ["a", "b"], "new": ["a"], "explanation": ""}))
    assert flat["old"] == ["a\nb"] and flat["new"] == ["a"]
    env = PulseCLI._parse_code_fix(json.dumps({"json": {"old": ["a"], "new": ["b"]}}))
    assert env["new"] == ["b"]
    assert PulseCLI._parse_code_fix('{"old": [], "new": []}') is None


def test_ok_extract_directives():
    text = "Reasoning\nCALC: 2*3\nGREP: lr\nVIEW: model.py:1-5\nPROMOTE: a, b\nGPUTRACK: none\n"
    cleaned, calc, promote, gput, gpuun, sens, ns, grep, view = PulseCLI._extract_directives(text)
    assert cleaned == "Reasoning"
    assert calc == ["2*3"] and grep == ["lr"] and view == ["model.py:1-5"]
    assert promote == ["a", "b"] and gput == []
    _c, req = PulseCLI._extract_new_directives("TRACE: loss\nCORR: a b\nDEPGRAPH:")
    assert req == {"depgraph": [""], "trace": ["loss"], "corr": ["a b"]}


def test_ok_calc_arithmetic():
    assert pc._safe_eval_math("math.log(10)") == pytest.approx(math.log(10))
    assert pc._safe_eval_math("sqrt(16) + 1") == 5.0
    assert "calc error" in pc._safe_eval_math("open('x')")


def test_ok_grep_and_view(tmp_path):
    cli = make_cli(tmp_path, "a = 1\nlearning_rate = 3\nb = 2\n")
    g = cli._run_grep("learning_rate")
    assert "line 2" in g and "1 match" in g
    v = cli._run_view("2-3")
    assert "   2 | learning_rate = 3" in v and "   3 | b = 2" in v
    assert "could not parse" in cli._run_view("nonsense")
    assert "no matches" in cli._run_grep("zzz_not_here")


def test_ok_stats_tools(tmp_path):
    cli = make_cli(tmp_path)
    cli.scalar_histories = {"loss": [4.0, 3.0, 2.0, 1.0], "acc": [0.1, 0.2, 0.3, 0.4],
                            "flat": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0]}
    assert "r = -1.0000" in cli._run_corr("loss acc")
    assert "delta = -3" in cli._run_diffstats("loss 0 -1")
    assert "index out of range" in cli._run_diffstats("loss 0 99")
    assert "1 outlier" in cli._run_outlier("flat")
    assert "4 points" in cli._run_histogram("loss")


def test_ok_defof_and_callers(tmp_path):
    cli = make_cli(tmp_path, "def build():\n    return 1\n\nm = build()\nbuild()\n")
    assert "function build" in cli._run_defof("build")
    assert "2 call site(s)" in cli._run_callers("build")


def test_ok_pipeline_diagnosis_only_path_writes_nothing(tmp_path):
    replies = ["- train.py:1", "Diagnosis: lr too high.\nCALC: 10*0.1", "Lower lr to 0.01."]
    cli = make_cli(tmp_path, "lr = 10.0\n", replies=replies)
    stub_context(cli)
    out = cli._ask_agent_impl("why is the loss exploding?", include_code=True)
    assert "Lower lr" in out
    assert (tmp_path / "train.py").read_text() == "lr = 10.0\n"
    assert cli._fix_applied_this_turn is False


def test_ok_pipeline_applies_verified_fix(tmp_path):
    fix = json.dumps({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "explanation": "lower"})
    replies = ["- train.py:1", "Diagnosis: lr too high.", fix,
               '{"passes": true, "reason": "ok"}']
    cli = make_cli(tmp_path, "lr = 10.0\n", replies=replies)
    stub_context(cli)
    cli._ask_agent_impl("please fix the exploding loss", include_code=True)
    assert (tmp_path / "train.py").read_text() == "lr = 0.01\n"
    assert cli._fix_applied_this_turn is True


def test_ok_pipeline_dropped_fix_not_applied(tmp_path):
    fix = json.dumps({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "explanation": "lower"})
    replies = ["- train.py:1", "Diagnosis: lr too high.", fix,
               '{"passes": false, "reason": "wrong"}', '{"decision": "drop", "reason": "no"}']
    cli = make_cli(tmp_path, "lr = 10.0\n", replies=replies)
    stub_context(cli)
    out = cli._ask_agent_impl("please fix the exploding loss", include_code=True)
    assert (tmp_path / "train.py").read_text() == "lr = 10.0\n"
    assert "Fix not applied" in out


def test_ok_call_model_retries_empty_then_succeeds(tmp_path, monkeypatch):
    replies = ["", "real answer"]

    def fake_completion(**kw):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=replies.pop(0)))],
            usage=None)

    monkeypatch.setattr(pc.litellm, "completion", fake_completion)
    monkeypatch.setattr(pc, "_clamp_output_tokens", lambda m, r: r)
    monkeypatch.setattr(pc.time, "sleep", lambda s: None)
    cli = make_cli(tmp_path)
    assert cli._call_model("hi") == "real answer"


def test_ok_call_model_nonretryable_raises_agent_request_failed(tmp_path, monkeypatch):
    def boom(**kw):
        raise ValueError("bad model")

    monkeypatch.setattr(pc.litellm, "completion", boom)
    monkeypatch.setattr(pc, "_clamp_output_tokens", lambda m, r: r)
    cli = make_cli(tmp_path)
    with pytest.raises(pc.AgentRequestFailed):
        cli._call_model("hi")


def test_ok_auto_rollback_restores_pre_chain_state(tmp_path):
    cli = make_cli(tmp_path, "lr = 10.0\n")
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "a"})
    first = cli._last_commit_id
    cli._apply_code_fix({"old": ["lr = 0.01"], "new": ["lr = 0.02"], "files": [None], "explanation": "b"})
    assert cli._auto_rollback_after_failed_restarts(first) is True
    assert (tmp_path / "train.py").read_text() == "lr = 10.0\n"


def test_ok_rollback_directive(tmp_path):
    cli = make_cli(tmp_path, "lr = 10.0\n")
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "a"})
    out = cli._run_rollback("last")
    assert "1 file(s) restored" in out
    assert (tmp_path / "train.py").read_text() == "lr = 10.0\n"


def test_ok_pastfix_finds_local_fix(tmp_path):
    cli = make_cli(tmp_path, "lr = 10.0\n")
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None],
                         "explanation": "lower learning rate"})
    assert "1 local" in cli._run_pastfix("learning rate")


def test_ok_mllint_true_positives(tmp_path):
    code = textwrap.dedent("""\
        import torch
        opt = torch.optim.SGD([], lr=50.0)
        layer = Dropout(rate=1.0)
        model.compile(loss="mse", metrics=["accuracy"])
    """)
    msgs = " ".join(m for _l, _n, m in scan(tmp_path, code))
    assert "learning rate above 1.0" in msgs
    assert "dropout rate of 1.0" in msgs
    assert "regression loss" in msgs
