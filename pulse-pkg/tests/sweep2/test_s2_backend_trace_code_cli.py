"""Sweep 2: pulse_trace.py (TRACE), pulse_code.py (`pulse code`) and cli.py.

No model is ever called: the pulse_code tests drive simulate()/_create_file_for_fix
directly with a stand-in CLI object.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_* are regression coverage for behaviour verified to work.
"""
import ast
import os
import sys
import textwrap
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_trace as PT  # noqa: E402


def F(label, src):
    return (label, "/proj/" + label, textwrap.dedent(src).lstrip("\n"))


def up_parents(graph, key=None):
    node = graph.nodes[key or graph.root]
    return [p for s in node.sites for p in s.parents]


# ======================================================================== TRACE bugs

def test_bug_line_hint_at_module_level_picks_a_function_local():
    """`TRACE: loss:train.py:6` where line 6 is module-level code (`print(loss)`). No
    function encloses the line, so find_origin falls back to "the assignment closest
    above the line" across ALL scopes -- and picks `loss` local to evaluate() (line 4)
    over the module's own `loss` (line 2), which is the one line 6 actually reads."""
    files = [F("train.py", """
        import torch
        loss = criterion(out, y)
        def evaluate(m):
            loss = m.eval_loss()
            return loss
        print(loss)
    """)]
    graph = PT.build(files, "loss", "train.py", 6)
    assert graph.root == ("train.py", PT.MODULE_SCOPE, "loss"), graph.root


def test_bug_positional_argument_after_varargs_is_mapped_to_keyword_only_param():
    """def step(model, *extra, lr=0.1): params are [model, extra, lr]. A call
    step(net, g1, g2) puts g1 AND g2 into *extra, but _forward_into_calls maps positional
    slot 2 to params[2] == 'lr' (and _argument_for maps lr back to g2): TRACE claims the
    gradient flows into the learning rate. Positional args past the positional params
    belong to *args; keyword-only params can only be passed by keyword."""
    files = [F("t.py", """
        def step(model, *extra, lr=0.1):
            model.update(lr)

        def main(net, g1, g2):
            step(net, g1, g2)
    """)]
    down = PT.build(files, "g2")
    targets = [e.target for e in down.nodes[down.root].out if e.target]
    assert ("t.py", "step", "lr") not in targets, targets
    up = PT.build(files, "lr")
    assert ("t.py", "main", "g2") not in up_parents(up), up_parents(up)


def test_bug_super_init_of_another_class_is_taken_as_a_constructor_call():
    """callers_of() matches by simple name, so every `super().__init__(...)` in the
    project is a "call site" of every class's __init__: tracing self.lr in Opt reports
    it as coming from Net.__init__'s `super().__init__(hidden)` (an unrelated class),
    while the real construction `Opt(0.1)` is never found."""
    files = [F("m.py", """
        class Opt:
            def __init__(self, lr):
                self.lr = lr

        class Net(Base):
            def __init__(self, hidden):
                super().__init__(hidden)
                self.h = hidden

        def main():
            opt = Opt(0.1)
    """)]
    graph = PT.build(files, "self.lr")
    lr_param = ("m.py", "Opt.__init__", "lr")
    assert lr_param in graph.nodes
    assert ("m.py", "Net.__init__", "hidden") not in up_parents(graph, lr_param)


def test_bug_assignments_inside_match_case_are_invisible():
    """_own_statements/_nested_definitions descend into body/orelse/finalbody/handlers but
    not into `match` cases (ast.Match keeps them in .cases), so a variable set in a
    `case` block is "never assigned" and TRACE returns nothing for it."""
    files = [F("train.py", """
        match args.sched:
            case "cosine":
                lr = 0.1
            case _:
                lr = 0.01
        opt = SGD(params, lr=lr)
    """)]
    graph = PT.build(files, "lr")
    assert graph is not None
    assert len(graph.nodes[graph.root].sites) == 2


def test_bug_except_star_block_is_credited_to_the_try_line():
    """(Found by reading.) read_parts() treats ast.Try as a compound statement but not
    ast.TryStar (`except*`), so a whole try/except* block was walked as one simple
    statement: every read inside it was credited to the `try:` line."""
    if not hasattr(ast, "TryStar"):
        pytest.skip("except* needs Python 3.11+")
    files = [F("t.py", """
        lr = 0.1
        try:
            opt = SGD(lr)
        except* ValueError:
            pass
    """)]
    graph = PT.build(files, "lr")
    lines = sorted(e.lineno for e in graph.nodes[graph.root].out)
    assert 2 not in lines and 3 in lines, lines


# ======================================================================== TRACE ok

def test_ok_trace_scoping():
    files = [F("train.py", """
        LR = 1e-3
        def make():
            scale = 2
            def inner(x):
                nonlocal scale
                scale = scale * x
                return scale
            return inner
        def train(batches):
            global LR
            LR = LR * 0.5
            vals = [b * 2 for b in batches]
            if (n := len(vals)) > 0:
                total = n
            return total
        class Base:
            def __init__(self):
                self.w = 1
        class Child(Base):
            def forward(self, x):
                return self.w * x
    """)]
    g = PT.build(files, "LR", "train.py", 11)
    assert g.root == ("train.py", PT.MODULE_SCOPE, "LR")
    g = PT.build(files, "scale", "train.py", 6)
    assert g.root == ("train.py", "make", "scale")
    g = PT.build(files, "total")
    assert ("train.py", "train", "n") in up_parents(g)
    g = PT.build(files, "self.w")
    assert g.root == ("train.py", "<class Base>", "self.w")
    assert any(e.lineno == 21 for e in g.nodes[g.root].out)
    out = PT.trace(files, "b")
    assert "never assigned" in out


# ======================================================================== pulse code

def _fake_code_cli(root, files):
    from pulse.pulse_cli import PulseCLI
    texts, labels, paths = {}, {}, {}
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        texts[path], labels[path], paths[rel] = text, rel, path
    cli = types.SimpleNamespace(_project_root=root, texts=texts, _label_for_path=labels,
                                _path_for_label=paths, focus=list(texts), known=list(texts))
    cli._lint_check = lambda content, path, original=None: PulseCLI._lint_check(None, content, path, original)
    return cli


def _fix(old, new, files, create=None):
    return {"old": old, "new": new, "files": files, "create": create or [], "explanation": "x"}


def test_bug_code_preview_refuses_edit_the_applier_would_make(tmp_path):
    """simulate() (the preview + all-or-nothing gate) still counts occurrences with
    str.count, while _apply_code_fix matches on token boundaries. `lr = 0.1` in a file
    that also has `base_lr = 0.1` is ONE match for the applier but "matched 2 times
    (ambiguous)" for the preview, so Pulse Code burns its revise rounds and then refuses
    a perfectly good edit."""
    from pulse import pulse_code as PC
    cli = _fake_code_cli(str(tmp_path), {"train.py": "base_lr = 0.1\nlr = 0.1\nprint(lr, base_lr)\n"})
    changes, problems = PC.simulate(cli, _fix(["lr = 0.1"], ["lr = 0.2"], ["train.py"]))
    assert not problems, problems


def test_bug_code_preview_shows_a_corrupting_diff_the_applier_would_not_write(tmp_path):
    """The other side of the same mismatch: `lr = 0.1` inside `lr = 0.15` is one str.count
    match, so the preview diff shows `lr = 0.25` (a corrupted literal) and asks the user
    to approve it; the token-boundary applier then finds nothing and the turn fails.
    Correct: preview uses the applier's matching, so no such diff is ever shown."""
    from pulse import pulse_code as PC
    cli = _fake_code_cli(str(tmp_path), {"train.py": "lr = 0.15\n"})
    changes, problems = PC.simulate(cli, _fix(["lr = 0.1"], ["lr = 0.2"], ["train.py"]))
    after = [a for (_b, a, _n) in changes.values()]
    assert "lr = 0.25\n" not in after, after


def test_bug_code_preview_blocks_files_with_preexisting_undefined_names(tmp_path):
    """The applier's lint gate only blocks problems a fix INTRODUCES (first-sweep fix:
    a notebook export's display() used to block every fix to that file). simulate()
    calls _lint_check without `original`, so in Pulse Code that file still can't be
    edited at all: every change is refused over the pre-existing undefined name."""
    pytest.importorskip("pyflakes")
    from pulse import pulse_code as PC
    src = "import pandas as pd\ndf = pd.DataFrame()\ndisplay(df)\nlr = 0.1\n"
    cli = _fake_code_cli(str(tmp_path), {"train.py": src})
    changes, problems = PC.simulate(cli, _fix(["lr = 0.1"], ["lr = 0.2"], ["train.py"]))
    assert not problems, problems


def test_bug_create_file_for_fix_still_compares_abspaths(tmp_path):
    """The symlink fix went into simulate() (realpath) but not into the function that
    actually writes new files: _create_file_for_fix still checks commonpath of abspath()s,
    so a create through an in-project symlink (data -> /mnt/...) writes outside the
    project whenever it is reached without the preview (or the link appears after it)."""
    from pulse.pulse_cli import PulseCLI
    root, outside = tmp_path / "proj", tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    os.symlink(outside, root / "data")
    cli = PulseCLI.__new__(PulseCLI)
    cli._project_root = str(root)
    cli._repo_cwd = str(root)
    cli.extra_files = {}
    cli._last_apply_lint_failed = []
    skipped = []
    cli._create_file_for_fix({"path": "data/evil.py", "content": "x = 1\n"}, skipped)
    assert not (outside / "evil.py").exists()


def test_ok_code_simulate_create_and_refusals(tmp_path):
    from pulse import pulse_code as PC
    cli = _fake_code_cli(str(tmp_path), {"m.py": "def f():\n    return 1\n"})
    changes, problems = PC.simulate(cli, _fix(["return 1"], ["return 2"], ["m.py"],
                                              create=[{"path": "pkg/new.py", "content": "y = 1"}]))
    assert not problems
    assert any(is_new for (_b, _a, is_new) in changes.values())
    for bad in ("../x.py", ".git/hooks/pre-commit", "/etc/x.py"):
        _c, p = PC.simulate(cli, _fix([], [], [], create=[{"path": bad, "content": "a = 1\n"}]))
        assert p, bad


# ======================================================================== cli.py

def test_bug_launch_option_hint_drops_the_approver_model(capsys):
    """`pulse --approver MODEL train.py` is answered with a suggested command. The
    first-sweep fix carries --cwd's value along but not --approver's (added later, same
    shape): the model id is taken as the script and the hint says
    `pulse run --approver openrouter/x` -- no script, a command that fails."""
    from pulse import cli
    rc = cli.main(["--approver", "openrouter/anthropic/claude-sonnet-5", "train.py"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "pulse run --approver openrouter/anthropic/claude-sonnet-5 train.py" in out, out


def test_bug_restart_makes_argv0_absolute(monkeypatch, tmp_path):
    """(Found by reading.) `pulse run train.py` sets sys.argv[0] to 'train.py' as typed,
    but the fix-triggered restart passed the absolute path, so after the first fix the
    script saw an absolute sys.argv[0]. Correct: the restart passes it as typed."""
    from pulse import cli
    script = tmp_path / "train.py"
    script.write_text("x = 1\n")
    monkeypatch.setattr(cli, "_RUN_ARGV0", ("train.py", str(tmp_path)))
    argv = cli._restart_argv(sys.executable, str(script), ["--lr", "1"])
    assert argv[-3:] == ["train.py", "--lr", "1"], argv
    monkeypatch.setattr(cli, "_RUN_ARGV0", ("other.py", str(tmp_path)))
    assert cli._restart_argv(sys.executable, str(script), [])[-1] == str(script)


def _injected_index(source):
    from pulse import cli
    tree = ast.parse(source)
    cli.PulseASTInjector().visit(tree)
    for i, stmt in enumerate(tree.body):
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "auto_track"):
            return i
    return None


def test_ok_auto_track_insertion_and_options(monkeypatch, tmp_path):
    from pulse import cli
    src = ('"""doc"""\nfrom __future__ import annotations\nimport os\n'
           "try:\n    import wandb\nexcept ImportError:\n    wandb = None\n"
           "X = 1\nimport json\n")
    assert _injected_index(src) == 3
    assert _injected_index("import os\nif __name__ == '__main__':\n    main()\n") is None
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PULSE_APPROVER", raising=False)
    monkeypatch.delenv("PULSE_AGENT_LOG", raising=False)
    stream, cwd, again, script, rest = cli._parse_run_args(
        ["--approver=m/x", "--agent-log=logs/a.log", "--cwd", "d", "train.py", "--lr", "1"])
    assert (script, rest, cwd) == ("train.py", ["--lr", "1"], "d")
    assert os.environ["PULSE_APPROVER"] == "m/x"
    assert os.environ["PULSE_AGENT_LOG"] == str(tmp_path / "logs" / "a.log")
    with pytest.raises(cli._UsageError):
        cli._parse_run_args(["--approver"])
