"""Sweep 2, area `static` -- prompts vs the parsers that read their replies.

Every test_bug_* asserts the CORRECT behaviour, so it fails on the code as found.
test_ok_* are generic structural guards (every JSON key / answer line a prompt asks for is
read by the code that consumes that reply) plus regressions for behaviour verified to work.

Run:  PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/sweep2/test_s2_static_prompts.py
"""
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(_ROOT, "src")
PKG = os.path.join(SRC, "pulse")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("PULSE_ASYNC_MODEL_CALLS", "0")
os.environ.setdefault("PULSE_LOGGING", "0")

from pulse import pulse_cli, pulse_approver, pulse_brain, pulse_code, pulse_terminal  # noqa: E402
from pulse.pulse_cli import PulseCLI  # noqa: E402


def _cli(tmp_path):
    cli = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdfs"))
    cli.sensitivity = 0.3
    return cli


def _scripted(cli, answers):
    """Replace the model with a fixed list of replies; record the prompts it was sent."""
    sent = []
    queue = list(answers)

    def fake(instruction, *args, **kwargs):
        sent.append(instruction)
        if not queue:
            raise AssertionError(f"unexpected extra model call: {instruction[:200]}")
        return queue.pop(0)

    cli._call_model = fake
    return sent


# =====================================================================================
# 1. A fix that asked for a fresh start ("resume": false) loses that request whenever the
#    fix is revised or corrected -- the restarted run resumes from the damaged weights.
# =====================================================================================

_FIX_FRESH = {"old": ["w = torch.zeros(10, 10)"], "new": ["w = torch.randn(10, 10) * 0.01"],
              "files": [None], "explanation": "zero init kills symmetry", "resume": False}


def test_bug_pass4_revision_drops_resume_false(tmp_path):
    """PASS 4b's "revise" shape (_PASS4_RECHECK_TMPL) lists old/new/files/explanation but not
    resume, and _verify_fix_with_retries replaces the fix with the revision wholesale. A fix
    that asked for a fresh start (init bug, weights already NaN) is then applied WITHOUT
    resume:false, so the restart resumes from the checkpoint that carries the bug.
    Correct: a revision that does not mention resume keeps the original fix's resume."""
    cli = _cli(tmp_path)
    _scripted(cli, [
        '{"passes": false, "reason": "too broad"}',
        '{"decision": "revise", "old": ["w = torch.zeros(10, 10)"], '
        '"new": ["w = torch.randn(10, 10) * 0.02"], "files": [null], "explanation": "narrower"}',
        '{"passes": true, "reason": "ok"}',
    ])
    fix, passed, _reason = cli._verify_fix_with_retries(dict(_FIX_FRESH), "diagnosis")
    assert passed
    assert fix.get("resume") is False, f"resume:false was dropped by the revision: {fix}"


def test_bug_lint_correction_drops_resume_false(tmp_path):
    """When the first fix fails the lint gate nothing lands, and the pipeline applies the
    corrected fix from _request_lint_corrected_fix instead -- whose prompt asks only for
    old/new/files/explanation. resume:false from the original is lost, so _apply_code_fix
    never sets _resume_after_fix=False. Correct: the correction inherits resume."""
    cli = _cli(tmp_path)
    _scripted(cli, ['{"old": ["w = torch.zeros(10, 10)"], "new": ["w = torch.randn(10, 10) * 0.01"], '
                    '"files": [null], "explanation": "fixed the indent"}'])
    corrected = cli._request_lint_corrected_fix(dict(_FIX_FRESH), {str(tmp_path / "train.py"): ["E999 bad indent"]})
    assert corrected is not None
    assert corrected.get("resume") is False, f"resume:false was dropped by the lint correction: {corrected}"


def test_bug_snippet_correction_drops_resume_false(tmp_path):
    """Same loss through _request_corrected_snippets (old snippet did not match, the model
    re-quotes it): the corrected fix is applied on its own when the first pass matched
    nothing, without the original resume:false."""
    script = tmp_path / "train.py"
    script.write_text("import torch\nw = torch.zeros(10, 10)\n")
    cli = _cli(tmp_path)
    cli.script_path = str(script)
    cli._resolve_fix_path = lambda label: str(script)
    _scripted(cli, ['{"old": ["w = torch.zeros(10, 10)"], "new": ["w = torch.randn(10, 10) * 0.01"], '
                    '"files": [null], "explanation": "re-quoted"}'])
    fix = dict(_FIX_FRESH, old=["w = torch.zeros(10,10)"])
    corrected = cli._request_corrected_snippets(fix, [(fix["old"][0], None, "no exact match in train.py")])
    assert corrected is not None
    assert corrected.get("resume") is False, f"resume:false was dropped by the snippet correction: {corrected}"


# =====================================================================================
# 2. Auto-mode approver: an answer whose first line merely starts with "Approve" is read
#    as APPROVE, even when the actual verdict line below says DENY (fail-open).
# =====================================================================================

@pytest.mark.parametrize("answer", [
    "Approve or deny? This deletes the user's dataset.\nDENY: destroys data that cannot be regenerated",
    "APPROVE/DENY decision below.\nDENY: rm -rf data/ is outside the run's outputs",
], ids=["approve-or-deny-question", "approve-slash-deny"])
def test_bug_approver_reads_a_preamble_starting_with_approve_as_approval(answer):
    """pulse_approver._VERDICT_RE accepts APPROVE followed by any text ('or deny?', '/DENY'),
    and parse() takes the FIRST such line. A reasoning preamble therefore approves a command
    the model explicitly DENIED on the next line -- a destructive command runs unattended.
    Correct: the verdict word must be followed by ':' / a dash / end of line; a preamble
    like this is not a verdict (and when unsure, the result must not be an approval)."""
    try:
        decision = pulse_approver.parse(answer)
    except pulse_approver.ApproverUnavailable:
        return                                    # falling back to asking is also safe
    assert decision.approved is False, f"approved a command the model denied: {decision}"


def test_ok_approver_parses_plain_verdicts():
    assert pulse_approver.parse("APPROVE: only touches __pycache__").approved is True
    assert pulse_approver.parse("DENY: deletes the dataset").approved is False
    assert pulse_approver.parse("**DENY** - rewrites git history").approved is False
    assert pulse_approver.parse("Thinking...\nAPPROVE: regenerable cache").reason == "regenerable cache"
    with pytest.raises(pulse_approver.ApproverUnavailable):
        pulse_approver.parse("I am not sure.")


# =====================================================================================
# 3. Prompt text that contradicts the code
# =====================================================================================

def test_bug_start_prime_prompt_counts_four_things_but_asks_five():
    """_START_PRIME_PROMPT announces 'Four things:' and then numbers five (the fifth, the
    CHECKNOTE for the first check-in, was added later) and asks for 'these five lines'.
    A model that trusts the count can drop item 5. Correct: the count matches the list."""
    prompt = PulseCLI._START_PRIME_PROMPT
    items = re.findall(r"(?m)(?:^|\n)(\d)\. ", prompt)
    words = {"Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6}
    m = re.search(r"\b(Two|Three|Four|Five|Six) things", prompt)
    assert m, "prompt no longer announces a count"
    assert words[m.group(1)] == len(items), f"announces {m.group(1)} things, numbers {len(items)}"


_CONFIRM_KEYWORDS = {
    "deletes_files": ("delet",),
    "modifies_git_state": ("git",),
    "reaches_outside_workspace": ("outside the project",),
    "launches_persistent_process": ("background",),
    "overwrites_via_redirect": ("redirect",),
}


@pytest.mark.parametrize("name,prompt", [
    ("pulse_cli.SYSTEM_PROMPT", pulse_cli.SYSTEM_PROMPT),
    ("PulseCLI._CHECKIN_SYSTEM_PROMPT", PulseCLI._CHECKIN_SYSTEM_PROMPT),
    ("pulse_code.CODE_SYSTEM_PROMPT", pulse_code.CODE_SYSTEM_PROMPT),
], ids=["debugger", "checkin", "pulse_code"])
def test_bug_every_terminal_prompt_names_every_confirmation_category(name, prompt):
    """Each prompt that offers TERMINAL tells the model which commands will pause for
    confirmation 'for those specific cases, not for ordinary commands'. The debugger's
    SYSTEM_PROMPT lists four -- it omits 'overwrites a file via a shell redirect', which
    classify_command also gates -- so the model is surprised by a pause/denial on
    `python x.py > out.txt` and misreads it. Correct: every category classify_command
    can raise is named in every such prompt."""
    flags = pulse_terminal.classify_command("true")
    assert set(flags) == set(_CONFIRM_KEYWORDS), "classify_command's categories changed -- update this map"
    lower = prompt.lower()
    missing = [cat for cat, words in _CONFIRM_KEYWORDS.items() if not any(w in lower for w in words)]
    assert not missing, f"{name} never mentions: {missing}"


# =====================================================================================
# 4. Generic guards: every field a prompt asks for is read by its consumer
# =====================================================================================

def _keys_in(template):
    return set(re.findall(r'\{\{?"([a-z_]+)"\s*:', template))


def _read_in_source(key, *sources):
    pat = re.compile(r"""(?:\.get\(\s*|\[\s*|in\s+)["']""" + re.escape(key) + r"""["']""")
    return any(pat.search(s) for s in sources)


def _src(mod):
    with open(mod.__file__, encoding="utf-8") as handle:
        return handle.read()


def test_ok_every_json_key_a_pipeline_prompt_asks_for_is_read():
    """Each key a template shows in its JSON reply shape ({"passes": ..}, {"decision": ..},
    {"resolved": ..}, {"no_change": ..}, the AUDIT decision...) is read somewhere with
    .get("key") / ["key"] / "key" in -- a key nobody reads is an instruction the code
    silently ignores."""
    cli_src = _src(pulse_cli)
    templates = {
        name: getattr(pulse_cli, name) for name in dir(pulse_cli)
        if name.startswith("_PASS") and isinstance(getattr(pulse_cli, name), str)
    }
    assert {"_PASS4_VERIFY_TMPL", "_PASS4_RECHECK_TMPL", "_PASS6_CONFIRM_TMPL"} <= set(templates)
    problems = []
    for name, template in templates.items():
        for key in _keys_in(template):
            if not _read_in_source(key, cli_src):
                problems.append(f"pulse_cli.{name}: {key}")
    for key in _keys_in(pulse_brain.AUDIT_PROMPT):
        if not _read_in_source(key, _src(pulse_brain)):
            problems.append(f"pulse_brain.AUDIT_PROMPT: {key}")
    code_src = _src(pulse_code)
    for name in ("_IMPLEMENT", "_VERIFY", "_VERIFY_TERMINAL"):
        for key in _keys_in(getattr(pulse_code, name)):
            if not _read_in_source(key, code_src, cli_src):
                problems.append(f"pulse_code.{name}: {key}")
    assert not problems, "asked for but never read:\n" + "\n".join(problems)


def test_ok_every_code_fix_field_in_the_system_prompt_survives_parsing():
    """SYSTEM_PROMPT's CODE FIXES section lists the fields of the fix object; each one must
    come out of _parse_code_fix (a field it drops is one the applier never sees)."""
    section = pulse_cli.SYSTEM_PROMPT.split("must have exactly these fields:", 1)[1].split("Rules for old/new", 1)[0]
    fields = set(re.findall(r"(?m)^  ([a-z_]+): ", section))
    assert fields == {"old", "new", "files", "explanation", "resume"}, fields
    import json
    payload = {"old": ["a = 1"], "new": ["a = 2"], "files": ["model.py"], "explanation": "e", "resume": False}
    parsed = PulseCLI._parse_code_fix(json.dumps(payload))
    assert parsed is not None
    for f in fields:
        assert f in parsed and parsed[f] == payload[f], f


def test_ok_checkin_answer_lines_all_have_parsers():
    """The check-in's answer format (VERDICT/PROBLEM/NEXTCHECK/CHECKNOTE) is all read."""
    prompt = PulseCLI._CHECKIN_SYSTEM_PROMPT
    tail = prompt.split("answer with ONLY these lines", 1)[1]
    fields = set(re.findall(r"(?m)^([A-Z_]+): ", tail))
    assert fields == {"VERDICT", "PROBLEM", "NEXTCHECK", "CHECKNOTE"}
    answer = ("VERDICT: problem\nPROBLEM: scaler fitted on test data at train.py:12\n"
              "NEXTCHECK: 1,000\nCHECKNOTE: re-check val_loss")
    assert PulseCLI._checkin_verdict(answer) == (True, "scaler fitted on test data at train.py:12")
    assert PulseCLI._parse_nextcheck_steps(answer) == 1000
    assert PulseCLI._CHECKNOTE_RE.search(answer).group(1).strip() == "re-check val_loss"
    assert PulseCLI._checkin_verdict("VERDICT: **ok** -- all finite")[0] is False


def test_ok_no_change_reply_is_recognised_and_not_a_fix():
    ans = '{"no_change": true, "reason": "the alarm was noise"}'
    assert PulseCLI._parse_code_fix(ans) is None
    assert PulseCLI._parse_no_change(ans) == "the alarm was noise"
    assert PulseCLI._parse_no_change('{"old": ["a"], "new": ["b"]}') is None


def test_ok_pass4_keep_path_keeps_the_original_fix(tmp_path):
    cli = _cli(tmp_path)
    _scripted(cli, ['{"passes": false, "reason": "unsure"}', '{"decision": "keep", "reason": "it is right"}'])
    fix, passed, reason = cli._verify_fix_with_retries(dict(_FIX_FRESH), "diagnosis")
    assert passed and fix["resume"] is False and "kept" in reason


def test_ok_audit_decision_parsed_from_prose():
    answer = ('The run looks fine.\n{"status": "ok", "risk": "low", "findings": [], '
              '"next_check_minutes": 30, "reason": "smooth"}')
    d = pulse_brain.parse_decision(answer)
    assert d["status"] == "ok" and d["next_check_minutes"] == 30
