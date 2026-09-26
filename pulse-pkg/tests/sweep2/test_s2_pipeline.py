"""Bug sweep 2: the agent's investigate-and-fix pipeline in pulse_cli.py.

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

import pytest

os.environ.setdefault("PULSE_LOGGING", "0")

import pulse.pulse_cli as pc  # noqa: E402
from pulse.pulse_cli import PulseCLI  # noqa: E402


def _no_network_completion(**kw):
    raise RuntimeError("real litellm.completion call attempted in a test")


pc.litellm.completion = _no_network_completion


def make_cli(tmp_path, code="x = 1\n", extra=None, replies=None):
    """A real PulseCLI pointed at a script in tmp_path (same helper as sweep 1)."""
    cli = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdf"))
    script = tmp_path / "train.py"
    if isinstance(code, bytes):
        script.write_bytes(code)
        code = code.decode("utf-8").replace("\r\n", "\n")
    else:
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
    cli.review = False
    cli._queue_agent_retry = lambda *a, **k: None
    cli._ensure_retry_ticker = lambda: None
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
# CALC sandbox (_safe_eval_math)
# ==========================================================================

def test_bug_calc_round_negative_ndigits_hangs():
    """round() is whitelisted, and int.__round__(n, ndigits<0) computes
    10 ** -ndigits internally. The pow-size estimate only guards the `**`
    operator and the size cap only runs on the RESULT (which is 0), so
    CALC: round(1, -10**9) builds a billion-digit int while holding the GIL
    -- the training thread / check-in worker freezes for hours (-10**7 already
    takes ~6 s, -10**8 minutes). Correct: refuse a huge ndigits (error at once)."""
    code = ("import os, time; os.environ['PULSE_LOGGING']='0'\n"
            "from pulse.pulse_cli import _safe_eval_math\n"
            "t = time.time(); r = _safe_eval_math('round(1, -10**8)')\n"
            "print(repr(r), round(time.time() - t, 2))\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    try:
        out = subprocess.run([sys.executable, "-c", code], env=env, timeout=45,
                             capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        pytest.fail("CALC: round(1, -10**8) did not return within 45 s")
    assert "error" in out.stdout, out.stdout + out.stderr


def test_bug_calc_prod_multiplies_lists():
    """The BinOp path refuses arithmetic on non-numbers ('[0] * 10**9'), but
    math.prod does the same multiplication inside C: prod([[0], 10**9])
    evaluates [0] * 10**9 -- an 8 GB list (MemoryError or swap on the
    training machine), and smaller N returns a huge list that is str()'d
    into the prompt. Correct: CALC returns an error for non-numeric results."""
    result = pc._safe_eval_math("prod([[0], 3])")
    assert isinstance(result, str) and "error" in result, result


def test_ok_calc_blocks_known_escapes():
    for expr in ("().__class__", "9**9**9**9", "factorial(10**6)", "[0] * 10**9",
                 "(lambda: 1)()", "[x for x in (1,)]", "__import__('os')",
                 "math.__dict__", "comb(10**6, 5)", "x := 3"):
        r = pc._safe_eval_math(expr)
        assert isinstance(r, str) and "error" in r, (expr, r)
    assert pc._safe_eval_math("math.sqrt(16) + 2**10 / 4") == 260.0
    assert pc._safe_eval_math("-" * 1900 + "1").startswith("(calc error")


# ==========================================================================
# PASS 3 "no change needed"
# ==========================================================================

def test_bug_no_change_prose_regex_swallows_a_real_fix():
    """_parse_no_change treats any directive-free, brace-free reply matching
    'no (code) change(s) (is/are) needed/required' as a decline. A reply that
    DOES describe a fix but not as JSON ('No code changes are required beyond
    lowering lr ...', '... no changes needed elsewhere') ends PASS 3 as 'No
    code change needed' instead of being re-asked for the JSON -- the bug the
    model found is never fixed. Correct: only a reply that actually declines
    counts (None here)."""
    for text in (
        "No code changes are required beyond lowering the learning rate: change "
        "`lr = 10.0` to `lr = 0.01` on line 1.",
        "Change `lr = 10.0` to `lr = 0.01` on line 1; no changes needed elsewhere.",
        "Nothing to fix in the model itself -- the bug is lr = 10.0 on line 1, which must become 0.01.",
    ):
        assert PulseCLI._parse_no_change(text) is None, text


def test_bug_no_change_prose_false_positive_ends_pipeline_unfixed(tmp_path):
    """End to end: PASS 3's first reply describes the fix in prose containing
    'No code changes are required beyond ...'. It is taken as a decline, so the
    pipeline never asks for the JSON and nothing is written. Correct: re-ask
    (as for any non-JSON reply) and apply the fix the model then sends."""
    def reply(instr):
        if instr.startswith("PASS 1"):
            return "- train.py:1"
        if "PASS 2" in instr:
            return "Diagnosis: lr = 10.0 is far too high and diverges."
        if "DEVELOP & IMPLEMENT" in instr:
            n = sum(1 for c in cli.calls if "DEVELOP & IMPLEMENT" in c)
            if n == 1:
                return ("No code changes are required beyond lowering the learning rate: "
                        "change `lr = 10.0` to `lr = 0.01` on line 1.")
            return json.dumps({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "explanation": "lower lr"})
        if "PASS 4" in instr:
            return '{"passes": true, "reason": "ok"}'
        return "?"
    cli = make_cli(tmp_path, "lr = 10.0\n", replies=reply)
    stub_context(cli)
    cli._ask_agent_impl("the loss is NaN, please fix it", include_code=True)
    assert (tmp_path / "train.py").read_text() == "lr = 0.01\n"


def test_ok_no_change_json_and_plain_decline():
    assert PulseCLI._parse_no_change('{"no_change": true, "reason": "healthy"}') == "healthy"
    assert PulseCLI._parse_no_change('{"no_change": "false", "reason": "x"}') is None
    assert PulseCLI._parse_no_change("No code change is needed: the run is healthy.")
    assert PulseCLI._parse_no_change("No change needed.\nVIEW: train.py:1-5") is None


# ==========================================================================
# Applying fixes
# ==========================================================================

def test_bug_crlf_file_converted_to_lf_by_fix(tmp_path):
    """_apply_code_fix reads with open(path, 'r') -- universal newlines -- so
    '\\r\\n' is already '\\n' when it checks `crlf = "\\r\\n" in original_content`;
    crlf is never True and the file is written back (newline='') with LF
    everywhere: a one-line fix rewrites every line ending of a Windows file
    (whole-file git diff; on Windows every fixed file becomes LF). The
    changelog's 'before' is LF too, so /revert cannot restore it. Correct:
    only the fixed line changes; CRLF endings are kept."""
    cli = make_cli(tmp_path, b"lr = 10.0\r\nepochs = 3\r\n")
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "x"})
    assert (tmp_path / "train.py").read_bytes() == b"lr = 0.01\r\nepochs = 3\r\n"


def test_bug_revert_converts_crlf_file_to_lf(tmp_path):
    """/revert reads and writes in text mode ('r'/'w'), and the log's 'before'
    was captured through universal newlines, so reverting a fix to a CRLF file
    does not restore the original bytes. Correct: /revert restores the file
    exactly as it was before the fix."""
    original = b"lr = 10.0\r\nepochs = 3\r\n"
    cli = make_cli(tmp_path, original)
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "x"})
    cli._cmd_revert("")
    assert (tmp_path / "train.py").read_bytes() == original


def test_bug_crash_occurrence_ignores_token_boundaries(tmp_path):
    """The token-boundary fix covers the count, but when the snippet is
    ambiguous (2+ real hits) _occurrence_at_crash falls back to a raw
    content.find(), which also matches INSIDE a longer token. If the crash
    line holds such a mid-token hit, that one is 'the' occurrence: old
    'a = 1' is replaced inside 'b = a = 10', giving 'b = a = 20'. Correct: a
    mid-token hit is never picked (skip as ambiguous)."""
    code = "a = 1\na = 1\nb = a = 10\n"
    cli = make_cli(tmp_path, code)
    cli._last_crash_location = (os.path.abspath(str(tmp_path / "train.py")), 3)
    cli._apply_code_fix({"old": ["a = 1"], "new": ["a = 2"], "files": [None], "explanation": "x"})
    assert "b = a = 10" in (tmp_path / "train.py").read_text()


def test_bug_multi_file_fix_half_applied_when_one_file_fails_lint(tmp_path):
    """A two-file fix where one file's edit fails the lint gate: the other file
    is written, first_pass_landed is True, and the lint-correction retry is
    only offered when NOTHING landed (`elif lint_failed and not
    first_pass_landed`). The run restarts with half a fix on disk (e.g. a
    renamed function in one file, the old name still called in the other),
    and the model is never shown the lint error. Correct: give the model the
    lint-correction retry for the failed file too (as the skipped-snippet
    retry already does after a partial landing)."""
    def reply(instr):
        if instr.startswith("PASS 1"):
            return "- train.py:1, model.py:1"
        if "PASS 2" in instr:
            return "Diagnosis: both constants are wrong."
        if "DEVELOP & IMPLEMENT" in instr:
            return json.dumps({"old": ["x = 1", "h = 3"], "new": ["x = 2", "h = (4"],
                               "files": ["train.py", "model.py"], "explanation": "fix both"})
        if "PASS 4" in instr:
            return '{"passes": true, "reason": "ok"}'
        if "CORRECTION" in instr:
            return json.dumps({"old": ["h = 3"], "new": ["h = 4"], "files": ["model.py"],
                               "explanation": "fixed paren"})
        return "?"
    cli = make_cli(tmp_path, "x = 1\n", extra={"model.py": "h = 3\n"}, replies=reply)
    stub_context(cli)
    cli._ask_agent_impl("please fix the crash", include_code=True)
    assert (tmp_path / "model.py").read_text() == "h = 4\n", \
        "model.py's half of the fix was dropped without a lint-correction retry"


def test_ok_fuzzy_match_reindents_and_keeps_next_line(tmp_path):
    code = "def f():\n    a = 1\n    b = 2\n    return a\n"
    cli = make_cli(tmp_path, code)
    cli._apply_code_fix({"old": ["a = 1\nb = 2"], "new": ["a = 1\nb = 3"], "files": [None],
                         "explanation": "x"})
    assert (tmp_path / "train.py").read_text() == code.replace("b = 2", "b = 3")


def test_ok_token_boundary_refuses_mid_token_match(tmp_path):
    cli = make_cli(tmp_path, "lr = 0.15\n")
    cli._apply_code_fix({"old": ["lr = 0.1"], "new": ["lr = 0.01"], "files": [None], "explanation": "x"})
    assert (tmp_path / "train.py").read_text() == "lr = 0.15\n"


def test_ok_lint_gate_allows_fix_with_preexisting_undefined_name(tmp_path):
    pytest.importorskip("pyflakes")
    cli = make_cli(tmp_path, "lr = 10.0\ndisplay(lr)\n")
    cli._apply_code_fix({"old": ["lr = 10.0"], "new": ["lr = 0.01"], "files": [None], "explanation": "x"})
    assert (tmp_path / "train.py").read_text().startswith("lr = 0.01")


def test_ok_lint_retry_when_nothing_landed(tmp_path):
    replies = ["- train.py:1", "Diagnosis: x wrong.",
               json.dumps({"old": ["x = 1"], "new": ["x = (2"], "explanation": "e"}),
               '{"passes": true, "reason": "ok"}',
               json.dumps({"old": ["x = 1"], "new": ["x = 2"], "explanation": "e"})]
    cli = make_cli(tmp_path, "x = 1\n", replies=replies)
    stub_context(cli)
    cli._ask_agent_impl("please fix x", include_code=True)
    assert (tmp_path / "train.py").read_text() == "x = 2\n"


def test_ok_revert_ambiguous_and_created_files(tmp_path):
    cli = make_cli(tmp_path, "x = 1\n")
    cli._apply_code_fix({"old": [], "new": [], "files": [], "explanation": "add",
                         "create": [{"path": "helper.py", "content": "H = 1\n"}]})
    cli._cmd_revert("")
    assert not (tmp_path / "helper.py").exists()
    cli._cmd_revert("")
    assert (tmp_path / "helper.py").read_text() == "H = 1\n"
    entries = cli._load_fix_log()
    idx, why = PulseCLI._match_commit_id(entries + [dict(entries[0], id=entries[0]["id"][:1] + "zzzzzzz")],
                                         entries[0]["id"][:1])
    assert idx is None and "ambiguous" in why


# ==========================================================================
# MLLINT auto-fix
# ==========================================================================

def test_bug_mllint_auto_fix_lint_gate_ignores_original(tmp_path):
    """_apply_code_fix's lint gate was fixed to block only problems a fix
    INTRODUCES (lint_check(..., original=...)), but _mllint_auto_fix still
    calls _lint_check(new_src, path) without the original: in a notebook
    export that calls display() (undefined for pyflakes) the one-token
    metrics fix is refused although it adds nothing. Correct: pass the
    original, apply the patch."""
    pytest.importorskip("pyflakes")
    code = ('reg = None\ndisplay(reg)\n'
            'reg.compile(loss="mse", optimizer="adam", metrics=["accuracy"])\n')
    cli = make_cli(tmp_path, code)
    findings = pc._mllint_scan(cli._iter_ast_trees())
    assert findings
    assert cli._mllint_auto_fix(findings)
    assert 'metrics=["mae"]' in (tmp_path / "train.py").read_text()


def test_bug_mllint_auto_fix_converts_crlf_to_lf(tmp_path):
    """_mllint_auto_fix reads with 'r' and writes with 'w': every CRLF line
    ending in the file becomes LF for a one-token metrics patch. Correct: keep
    the file's line endings."""
    code = b'reg = None\r\nreg.compile(loss="mse", optimizer="adam", metrics=["accuracy"])\r\n'
    cli = make_cli(tmp_path, code)
    assert cli._mllint_auto_fix(pc._mllint_scan(cli._iter_ast_trees()))
    assert (tmp_path / "train.py").read_bytes() == code.replace(b'"accuracy"', b'"mae"')


def test_bug_mllint_eval_heuristic_matches_substrings(tmp_path):
    """The torch 'evaluation function without no_grad()/eval()' rule tests
    `k in name.lower()` for eval/valid/test, so `load_latest` ('test'),
    `build_retrieval_index` ('eval') and `invalidate_cache` ('valid') are
    flagged as evaluation functions -- and every start-of-run finding costs a
    full agent pipeline. Correct: match name parts (like
    _MLLINT_TEST_NAME_RE), not substrings."""
    code = textwrap.dedent("""\
        import torch
        def load_latest(path):
            return torch.load(path)
        def build_retrieval_index(x):
            return torch.stack(x)
    """)
    cli = make_cli(tmp_path, code)
    findings = pc._mllint_scan(cli._iter_ast_trees())
    assert not [f for f in findings if "evaluation/validation function" in f[2]], findings


def test_ok_mllint_auto_fix_patches_only_flagged_call(tmp_path):
    code = ('clf = reg = None\n'
            'clf.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])\n'
            'reg.compile(loss="mse", optimizer="adam", metrics=["accuracy"])\n')
    cli = make_cli(tmp_path, code)
    assert cli._mllint_auto_fix(pc._mllint_scan(cli._iter_ast_trees()))
    lines = (tmp_path / "train.py").read_text().splitlines()
    assert 'metrics=["accuracy"]' in lines[1] and 'metrics=["mae"]' in lines[2]
    assert cli._load_fix_log() and cli.code_text == (tmp_path / "train.py").read_text()


# ==========================================================================
# Tools
# ==========================================================================

def test_bug_outlier_and_histogram_never_see_nan(tmp_path):
    """Two fixes collided: OUTLIER/HISTOGRAM were fixed to REPORT NaN/inf
    (_split_finite + 'non-finite point(s)'), but _scalar_history was later
    changed to drop every non-finite value before returning, so neither tool
    ever sees one. A diverged loss [1.0, 0.9, 0.8, 0.7, nan] gets 'no finite
    points with |z| > 3' with no mention of the NaN (sweep 1's
    test_bug_outlier_blind_to_nan passes only because the text now says 'no
    finite points', not 'no points'), and an all-NaN history is 'no numeric
    history available'. Correct: the NaN is reported."""
    cli = make_cli(tmp_path)
    cli.scalar_histories = {"loss": [1.0, 0.9, 0.8, 0.7, float("nan")]}
    out = cli._run_outlier("loss")
    assert "non-finite" in out or "nan" in out.lower(), out
    hist = cli._run_histogram("loss")
    assert "non-finite" in hist or "nan" in hist.lower(), hist
    cli.scalar_histories = {"loss": [float("nan"), float("inf")]}
    out = cli._run_outlier("loss")
    assert "non-finite" in out or "nan" in out.lower(), out


def _cleanup_modules(prefix):
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


def test_bug_doclookup_runs_project_file_in_namespace_package(tmp_path):
    """The new project-module refusal checks only the ROOT name, and
    _is_project_module needs spec.origin to be a file. A project directory
    without __init__.py (experiments/, scripts/) is a namespace package with
    origin None, so it is 'not a project module': DOCLOOKUP:
    experiments.side.run imports -- and so RUNS -- experiments/side.py inside
    the training process. Correct: refuse it like any project file."""
    (tmp_path / "experiments").mkdir()
    marker = tmp_path / "ran.txt"
    (tmp_path / "experiments" / "side.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\ndef run():\n    pass\n")
    cli = make_cli(tmp_path, "x = 1\n")
    sys.path.insert(0, str(tmp_path))
    try:
        out = cli._run_doclookup("experiments.side.run")
    finally:
        sys.path.remove(str(tmp_path))
        _cleanup_modules("experiments")
    assert not marker.exists(), out


def test_bug_doclookup_runs_unimported_submodule_of_project_package(tmp_path):
    """When the project's own package is already imported (the training script
    did `from mypkg import models`), `root not in sys.modules` is False, so
    the project-module check is skipped entirely, and DOCLOOKUP:
    mypkg.train_loop.main imports the not-yet-imported submodule, running its
    top level (another training script). Correct: refuse project modules
    whether or not the package is already loaded."""
    pkg = tmp_path / "mypkg_s2"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    marker = tmp_path / "ran2.txt"
    (pkg / "train_loop.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\ndef main():\n    pass\n")
    cli = make_cli(tmp_path, "import mypkg_s2\n")
    sys.path.insert(0, str(tmp_path))
    try:
        __import__("mypkg_s2")
        out = cli._run_doclookup("mypkg_s2.train_loop.main")
    finally:
        sys.path.remove(str(tmp_path))
        _cleanup_modules("mypkg_s2")
    assert not marker.exists(), out


def test_ok_doclookup_library_and_script_refusal(tmp_path):
    cli = make_cli(tmp_path, "x = 1\n")
    assert "sqrt" in cli._run_doclookup("math.sqrt")
    assert "own files" in cli._run_doclookup("train.main")


def test_bug_terminal_heredoc_check_refuses_bit_shift(tmp_path):
    """The heredoc refusal regex `(?:^|\\s)<<-?\\s*['"]?[A-Za-z_]` also matches a
    left shift followed by a name, so TERMINAL: python3 -c "n = 3; print(1 << n)"
    is refused as a heredoc and the model is told to rewrite a command that
    was already one line. Correct: it runs and prints 8."""
    cli = make_cli(tmp_path)
    out = cli._run_terminal('python3 -c "n = 3; print(1 << n)"')
    assert "heredoc" not in out and "8" in out, out


def test_ok_terminal_heredoc_without_space_refused(tmp_path):
    cli = make_cli(tmp_path)
    assert "heredoc" in cli._run_terminal("python3<<EOF")


def test_ok_terminal_heredoc_refused(tmp_path):
    cli = make_cli(tmp_path)
    assert "heredoc" in cli._run_terminal("python3 - <<'EOF'")


def test_ok_stats_tools_basic(tmp_path):
    cli = make_cli(tmp_path)
    cli.scalar_histories = {"loss": [1.0, None, 0.8, 0.7, 0.6], "lr": [0.1, 0.1, 0.2, 0.3, 0.4]}
    assert "r =" in cli._run_corr("loss lr")
    assert "delta" in cli._run_diffstats("loss 0 -1")


# ==========================================================================
# History window
# ==========================================================================

def test_ok_call_model_window_keeps_question(tmp_path, monkeypatch):
    import types
    seen = []

    def fake_completion(**kw):
        seen.append(kw["messages"])
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))], usage=None)

    monkeypatch.setattr(pc.litellm, "completion", fake_completion)
    monkeypatch.setattr(pc, "_clamp_output_tokens", lambda m, r: r)
    cli = make_cli(tmp_path)
    q = {"role": "user", "content": "QUESTION-MARKER"}
    cli._turn_question_msg = q
    cli.agent_history = [q] + [{"role": "user", "content": f"m{i}"} for i in range(30)]
    cli._call_model("instr")
    assert any(m.get("content") == "QUESTION-MARKER" for m in seen[-1])
