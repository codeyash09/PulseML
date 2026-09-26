"""Sweep 2, area `static`: static/structural review of the whole codebase, weighted to what
the first sweep's fixes and the hand-merged fix branches changed.

Found with pyflakes/ruff/pylint/vulture/bandit, AST scans, and a scan of every function
that exists both in pulse.py (the GUI) and pulse_cli.py (the CLI) for fixes applied to one
copy only. Each finding was confirmed by reading and is proven here.

test_bug_* assert the CORRECT behaviour, so they fail on the code as found.
test_ok_*  are regression coverage / generic structural guards that pass today.

Run:  PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/sweep2/test_s2_static_bugs.py
"""
import ast
import glob
import json
import os
import sys
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_src():
    d = HERE
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "src", "pulse")):
            return os.path.join(d, "src")
        d = os.path.dirname(d)
    raise RuntimeError("could not locate src/pulse")


SRC = _find_src()
PKG = os.path.join(SRC, "pulse")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("PULSE_ASYNC_MODEL_CALLS", "0")
os.environ.setdefault("PULSE_LOGGING", "0")

import pulse.pulse as core  # noqa: E402
from pulse import pulse_cli  # noqa: E402
from pulse.pulse_cli import PulseCLI  # noqa: E402

MODULE_FILES = sorted(p for p in glob.glob(os.path.join(PKG, "*.py"))
                      if not os.path.basename(p).startswith("test_"))


def _cli(tmp_path, **watch):
    cli = PulseCLI(watch_locals=dict(watch), pdf_dir=str(tmp_path / "pdfs"))
    cli.epoch_scalar_histories = {}
    cli.batch_scalar_histories = {}
    return cli


@pytest.fixture
def chat_cls(monkeypatch):
    """The GUI chat panel class without a display (same trick as tests/sweep)."""
    fake_tk = types.SimpleNamespace(Frame=type("Frame", (object,), {}), TclError=Exception)
    monkeypatch.setattr(core, "tk", fake_tk)
    monkeypatch.setattr(core, "HAS_TK", True)
    monkeypatch.setattr(core, "_CHAT_PANEL_CLS", None)
    cls = core._chat_panel_class()
    yield cls
    core._CHAT_PANEL_CLS = None


# =====================================================================================
# 1. GUI DOCLOOKUP: a conditional `import importlib.util` makes `importlib` a local
#    of the whole method -> UnboundLocalError for every already-imported library.
# =====================================================================================

def test_bug_gui_doclookup_crashes_for_an_already_imported_library(chat_cls):
    """pulse.py ChatPanel._run_doclookup (added by the first sweep's DOCLOOKUP fix) does
    `import importlib.util` inside `if root not in sys.modules:`. That makes `importlib` a
    LOCAL name for the whole method, so when the library IS already imported -- torch,
    numpy, json: the normal case for a live training run -- `importlib.import_module(root)`
    raises UnboundLocalError and DOCLOOKUP never works. Correct: the signature/docstring."""
    panel = chat_cls.__new__(chat_cls)
    assert "json" in sys.modules
    out = panel._run_doclookup("json.dumps")
    assert "DOCLOOKUP 'json.dumps': json.dumps(" in out


def test_ok_cli_doclookup_works_for_an_already_imported_library(tmp_path):
    cli = _cli(tmp_path)
    out = cli._run_doclookup("json.dumps")
    assert out.startswith("DOCLOOKUP 'json.dumps': json.dumps(")


def _conditional_local_import_shadows():
    """(file, function, line, name) for every function-local `import X` that sits under an
    `if` while X is also read outside that `if` in the same function: Python makes X local
    to the whole function, so the read outside the branch is an UnboundLocalError whenever
    the branch was not taken."""
    problems = []

    def own(node, path=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            yield child, path
            yield from own(child, path + (child,))

    for path in MODULE_FILES:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nodes = list(own(fn))
            for imp, ipath in nodes:
                if not isinstance(imp, (ast.Import, ast.ImportFrom)):
                    continue
                ifs = [a for a in ipath if isinstance(a, ast.If)]
                if not ifs:
                    continue
                inside = {id(x) for x in ast.walk(ifs[0])}
                for alias in imp.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    unconditional = any(
                        isinstance(x, (ast.Import, ast.ImportFrom))
                        and not any(isinstance(q, (ast.If, ast.Try)) for q in xp)
                        and any((b.asname or b.name).split(".")[0] == name for b in x.names)
                        for x, xp in nodes)
                    if unconditional:
                        continue
                    outside = [x.lineno for x, _ in nodes
                               if isinstance(x, ast.Name) and x.id == name
                               and isinstance(x.ctx, ast.Load) and id(x) not in inside]
                    if outside:
                        problems.append(f"{os.path.basename(path)}:{imp.lineno} {fn.name}: "
                                        f"`import {name}` under an if, read at line(s) {outside}")
    return problems


def test_bug_no_conditional_local_import_shadows_a_name_read_outside_its_branch():
    """Generic guard for the DOCLOOKUP bug above: a local import under an `if` of a name
    that is also used outside that branch (usually because the module already imports it
    at top level) is an UnboundLocalError on the path where the branch is skipped.
    Currently fails only for pulse.py ChatPanel._run_doclookup."""
    problems = _conditional_local_import_shadows()
    # cli.py main() imports console_main in two separate if-branches, each used only there
    # after its own import on that branch's path -- genuinely fine.
    problems = [p for p in problems if "console_main" not in p and "hardexamples" not in p]
    assert not problems, "\n".join(problems)


# =====================================================================================
# 2-5. First-sweep fixes applied to the GUI copy (pulse.py) only: the CLI copies
#      (pulse_cli.py -- the `pulse run` path) still have the same bugs.
# =====================================================================================

def test_bug_cli_gradcheck_leaves_weight_perturbed_when_loss_fn_raises(tmp_path):
    """Ledger: 'GRADCHECK leaves a weight perturbed if the loss function raises' (high) was
    fixed in pulse.py _exec_gradcheck only. pulse_cli.PulseCLI._run_exec_gradcheck still
    does `flat[idx] = orig + eps; loss_plus = float(loss_fn())` with the restore after it,
    so a loss_fn that raises (very common: it needs a batch) leaves the LIVE model's weight
    shifted by +1e-3 for the rest of training. Correct: the weight is restored exactly."""
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(3, 1)
    lin(torch.ones(1, 3)).sum().backward()
    before = lin.weight.detach().clone()

    def loss_fn():
        raise RuntimeError("loss_fn needs a batch")

    cli = _cli(tmp_path, lin=lin, loss_fn=loss_fn)
    out = cli._run_exec_gradcheck("weight")
    assert "raised" in out
    assert torch.equal(lin.weight.detach(), before), (lin.weight.detach() - before)


def test_bug_cli_shapetrace_leaves_model_in_eval_mode_when_forward_raises(tmp_path):
    """Ledger: 'SHAPETRACE leaves the model in eval mode if its forward pass fails' (high)
    was fixed in pulse.py only. pulse_cli.PulseCLI._run_exec_shapetrace still calls
    model.train(was_training) inside the try, after the forward: when the forward raises
    (it picks whatever tracked tensor it finds -- the labels, say) the live model stays in
    eval mode -- dropout off, BatchNorm frozen -- for the rest of the run."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Dropout(0.5))
    model.train()
    y = torch.zeros(3, dtype=torch.long)       # wrong shape/dtype for the model
    cli = _cli(tmp_path, model=model, y=y)
    out = cli._run_exec_shapetrace("model")
    assert "raised" in out
    assert model.training is True


def test_ok_cli_shapetrace_restores_train_mode_after_a_good_forward(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU())
    model.train()
    cli = _cli(tmp_path, model=model, x=torch.zeros(2, 8))
    out = cli._run_exec_shapetrace("model")
    assert "SHAPETRACE (forward pass" in out and model.training is True


def test_bug_cli_replay_snapshot_ignores_the_memory_budget(tmp_path, monkeypatch):
    """Ledger: 'REPLAY keeps 20 full model copies in RAM' (high) was fixed in pulse.py's
    _replay_maybe_checkpoint only (byte budget PULSE_REPLAY_MAX_MB, one snapshot per
    object, CPU-only mode leaves accelerator state alone). pulse_cli's own
    _replay_maybe_checkpoint -- run before every agent question -- still torch.saves every
    state_dict it sees with no budget, the same model once per name it is bound to, and
    keeps 20 of them. Correct: a model larger than the budget is not snapshotted, and one
    object is saved once."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(core, "_REPLAY_MAX_BYTES", 1024 * 1024)     # 1 MB budget
    big = torch.nn.Linear(1024, 512)                                  # ~2 MB of weights
    cli = _cli(tmp_path, model=big, net=big)
    cli._replay_maybe_checkpoint()
    stored = sum(len(b) for _, snap in cli._replay_checkpoints for b in snap.values())
    assert stored <= core._REPLAY_MAX_BYTES, f"{stored / 1e6:.1f} MB kept for a 1 MB budget"


def test_bug_cli_rank_status_includes_ranks_from_a_previous_job(tmp_path, monkeypatch):
    """Ledger: 'Multi-GPU status reports ranks from previous jobs' (medium) was fixed in
    pulse.py (_current_job_rank_status) only. The CLI's GPUSTATUS/RANKDIVERGE read
    pulse_cli._read_all_rank_status, which still returns every rank_*.json in the shared
    directory -- a finished 4-rank job's files show up in a 2-rank job ('4/2 rank(s)
    reporting'), and RANKDIVERGE compares this run's loss against a dead job's."""
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    key = "job"
    d = pulse_cli._rank_status_dir(key)
    now = time.time()
    old = now - 3 * 24 * 3600
    for rank in range(4):        # yesterday's 4-rank job
        with open(os.path.join(d, f"rank_{rank}.json"), "w") as f:
            json.dump({"rank": rank, "world_size": 4, "pid": 10_000 + rank, "updated": old}, f)
    for rank in range(2):        # this 2-rank job overwrites ranks 0 and 1
        pid = os.getpid() if rank == 0 else 20_001
        with open(os.path.join(d, f"rank_{rank}.json"), "w") as f:
            json.dump({"rank": rank, "world_size": 2, "pid": pid, "updated": now}, f)
    ranks = sorted(s["rank"] for s in pulse_cli._read_all_rank_status(key))
    assert ranks == [0, 1], ranks


# =====================================================================================
# 6. The reverse: a pulse_cli fix the GUI copy never got.
# =====================================================================================

def test_bug_gui_fuzzy_snippet_match_glues_the_next_line_into_the_fix():
    """Ledger: 'Whitespace-tolerant snippet match glues the next line into the edit' (high)
    was fixed in pulse_cli._find_fuzzy_snippet_span only. pulse.py's own copy (used by the
    GUI's fix writer, pulse.py ~5828) still ends the span AFTER the last line's newline, and
    the replacement has none, so the next line is glued onto the fix ('b = 3    return a')."""
    content = "def f():\n    a = 1\n    b = 2\n    return a\n"
    old = "a = 1\nb = 2"                       # the agent quoted it without indentation
    span = core._find_fuzzy_snippet_span(content, old)
    assert span is not None
    start, end = span
    patched = content[:start] + "    a = 1\n    b = 3" + content[end:]
    assert patched == "def f():\n    a = 1\n    b = 3\n    return a\n", patched


def test_ok_cli_fuzzy_snippet_match_keeps_the_next_line():
    content = "def f():\n    a = 1\n    b = 2\n    return a\n"
    start, end = pulse_cli._find_fuzzy_snippet_span(content, "a = 1\nb = 2")
    patched = content[:start] + "    a = 1\n    b = 3" + content[end:]
    assert patched == "def f():\n    a = 1\n    b = 3\n    return a\n"


# =====================================================================================
# Generic structural guards (pass today)
# =====================================================================================

def test_ok_no_module_level_constant_or_function_bound_twice():
    """Two branches merged by hand can each add the same constant/function: the second
    silently wins. Every module/class-level name (assignments, defs, classes) is bound
    once, apart from imports of submodules (import a.b / import a.c) and property
    setters."""
    problems = []

    def scan(body, where, fname):
        seen = {}
        for node in body:
            names = []
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
                names = [node.target.id]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                if any(d.endswith((".setter", ".deleter")) for d in decorators):
                    continue
                names = [node.name]
                if isinstance(node, ast.ClassDef):
                    scan(node.body, f"{where}.{node.name}", fname)
            elif isinstance(node, ast.Import):
                names = [a.asname for a in node.names if a.asname]      # `import a.b` rebinds `a` harmlessly
            elif isinstance(node, ast.ImportFrom):
                names = [a.asname or a.name for a in node.names]
            for name in names:
                if name in seen:
                    problems.append(f"{fname}:{node.lineno} {where}.{name} (first bound at line {seen[name]})")
                seen[name] = node.lineno

    for path in MODULE_FILES:
        with open(path, encoding="utf-8") as handle:
            scan(ast.parse(handle.read()).body, "<module>", os.path.basename(path))
    # pulse.py imports `multiprocessing as mp` twice (harmless duplicate import line).
    problems = [p for p in problems if not p.startswith("pulse.py") or ".mp " not in p]
    assert not problems, "\n".join(problems)


def test_ok_every_agent_log_event_has_a_literal_or_fstring_title():
    """The agent log is read by people asking 'why did Pulse do that?': every event title is
    a string written in the code (never a model- or script-controlled value alone, which
    could forge a '>>> [..] FIX APPLIED' line)."""
    problems = []
    for path in MODULE_FILES:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_agent_log_event" and node.args):
                title = node.args[0]
                # A literal, or an f-string / concatenation that starts with literal text.
                head = title
                while isinstance(head, ast.BinOp):
                    head = head.left
                if isinstance(head, ast.JoinedStr) and head.values:
                    head = head.values[0]
                ok = isinstance(head, ast.Constant) and str(head.value).strip() != ""
                if not ok:
                    problems.append(f"{os.path.basename(path)}:{node.lineno} {ast.unparse(title)}")
    assert not problems, "\n".join(problems)


# =====================================================================================
# 7. A subclass override that bypasses a first-sweep fix: Pulse Code's own
#    _build_file_labels never takes the CHANGELOG baseline.
# =====================================================================================

def test_bug_pulse_code_changelog_first_call_hides_the_sessions_edits(tmp_path, monkeypatch):
    """Ledger: "CHANGELOG's first call is always empty" was fixed by taking the baseline in
    PulseCLI._build_file_labels / set_code_text (_snapshot_changelog_baseline). Pulse Code's
    _CodeAgentCLI overrides _build_file_labels (pulse_code.py:275) without that call, and
    reload() sets code_text directly, so in `pulse code` no baseline exists until the
    first CHANGELOG -- which then takes the ALREADY-EDITED files as its baseline and reports
    nothing changed. CHANGELOG is one of the seven tools Pulse Code offers. Correct: the
    first CHANGELOG shows what changed since the session loaded the files."""
    from pulse import pulse_code
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "model.py"
    src.write_text("lr = 0.1\n")
    cli = pulse_code._CodeAgentCLI(pdf_dir=str(tmp_path / "pdfs"))
    cli.setup_code(str(tmp_path), [str(src)], [])
    src.write_text("lr = 0.001\n")            # an edit made during the session
    cli.reload()
    out = cli._run_changelog()
    assert "0.001" in out, out
