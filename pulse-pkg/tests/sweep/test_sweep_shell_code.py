"""Sweep: pulse_code -- the `pulse code` coding agent (file handling, simulate, paths).

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_* are regression coverage for behaviour verified to work.
"""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_code as PC  # noqa: E402
from pulse.pulse_cli import PulseCLI  # noqa: E402


def fake_cli(root, files):
    """Just enough of _CodeAgentCLI for simulate/normalise_targets/render_diff."""
    texts, labels, paths = {}, {}, {}
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        texts[path] = text
        labels[path] = rel
        paths[rel] = path
    cli = types.SimpleNamespace(
        _project_root=root, texts=texts, _label_for_path=labels, _path_for_label=paths,
        focus=list(texts), known=list(texts),
    )
    cli._lint_check = lambda content, path, original=None: PulseCLI._lint_check(None, content, path, original)
    return cli


def fix(old, new, files, create=None):
    return {"old": old, "new": new, "files": files, "create": create or [], "explanation": "x"}


# ---------------------------------------------------------------------------------------
# bugs
# ---------------------------------------------------------------------------------------

def test_bug_create_through_symlink_escapes_the_project(tmp_path):
    """`create` is refused outside the project by comparing abspath()s, which does not
    resolve symlinks: a path through a symlink inside the project (e.g. data -> /mnt/...,
    a very common ML layout) writes a brand-new file outside the project root."""
    root = tmp_path / "proj"
    outside = tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    os.symlink(outside, root / "data")
    cli = fake_cli(str(root), {"train.py": "x = 1\n"})
    changes, problems = PC.simulate(cli, fix([], [], [], create=[{"path": "data/evil.py", "content": "y = 2\n"}]))
    assert not changes, changes
    assert problems and "outside" in problems[0][2]


def test_bug_crlf_file_multiline_edit_is_refused(tmp_path):
    """Windows line endings: the model sees lines without '\\r', so a 2-line `old` never
    matches exactly and falls through to the fuzzy locator, whose span includes the last
    line's newline -- the replacement glues the next line onto it and the lint gate then
    rejects the change. CRLF files are effectively uneditable."""
    content = "a = 1\r\nb = 2\r\nc = 3\r\n"
    cli = fake_cli(str(tmp_path), {"m.py": content})
    changes, problems = PC.simulate(cli, fix(["a = 1\nb = 2"], ["a = 10\nb = 2"], ["m.py"]))
    assert problems == []
    after = changes[str(tmp_path / "m.py")][1]
    assert after.replace("\r\n", "\n") == "a = 10\nb = 2\nc = 3\n"


def test_bug_fuzzy_match_glues_the_following_line_on(tmp_path):
    """Same root cause on LF files: whenever the exact match misses by whitespace, the
    fuzzy span swallows the trailing newline of the matched block (and _apply_code_fix
    uses the same locator), so `return x` + next line become `return xprint(f())`."""
    content = "def f():\n    x = 1\n    return x\nprint(f())\n"
    cli = fake_cli(str(tmp_path), {"m.py": content})
    changes, problems = PC.simulate(
        cli, fix(["    x = 1\n    return  x"], ["    x = 2\n    return x"], ["m.py"]))
    assert problems == []
    assert changes[str(tmp_path / "m.py")][1] == "def f():\n    x = 2\n    return x\nprint(f())\n"


def test_bug_env_example_extension_can_never_match():
    """_TEXT_EXTS lists '.env.example', but it is compared with os.path.splitext(), which
    returns '.example' -- this entry can never match, so .env.example files are never
    indexed."""
    assert PC._looks_like_source(".env.example")


def test_bug_double_star_glob_is_not_recursive(tmp_path):
    """`/add src/**/*.py` ('globs work') uses glob.glob without recursive=True, so '**'
    behaves like '*': files directly in src/ and deeper than one level are silently
    missed."""
    (tmp_path / "src" / "pkg" / "sub").mkdir(parents=True)
    for rel in ("src/top.py", "src/pkg/mid.py", "src/pkg/sub/deep.py"):
        (tmp_path / rel).write_text("x = 1\n")
    files, problems = PC.expand_paths(str(tmp_path), ["src/**/*.py"])
    rels = sorted(os.path.relpath(f, tmp_path) for f in files)
    assert rels == ["src/pkg/mid.py", "src/pkg/sub/deep.py", "src/top.py"], rels


def test_bug_code_prompt_omits_redirect_confirmation():
    """The code agent's prompt says only deletes / git rewrites / outside-project /
    background commands pause for confirmation, 'not ordinary read/run/test commands' --
    but any '>' (e.g. `pytest 2>&1 | tail`) is gated too, and declined under -p without a
    TTY. The debugger's own prompt (pulse_cli ~8350) does list 'overwrite a file via
    redirect'; this one contradicts the code."""
    assert "redirect" in PC.CODE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------------------
# ok
# ---------------------------------------------------------------------------------------

def test_ok_simulate_exact_edit_and_create(tmp_path):
    cli = fake_cli(str(tmp_path), {"m.py": "def f():\n    return 1\n"})
    changes, problems = PC.simulate(cli, fix(["return 1"], ["return 2"], ["m.py"],
                                              create=[{"path": "pkg/new.py", "content": "z = 3"}]))
    assert problems == []
    assert changes[str(tmp_path / "m.py")][1] == "def f():\n    return 2\n"
    before, after, is_new = changes[str(tmp_path / "pkg" / "new.py")]
    assert (before, after, is_new) == ("", "z = 3\n", True)


@pytest.mark.parametrize("path", ["../escape.py", "/etc/evil.py", ".git/hooks/pre-commit",
                                  "sub/../../escape.py", ".pulse_history/x.py"])
def test_ok_simulate_refuses_create_outside_or_protected(tmp_path, path):
    root = tmp_path / "proj"
    root.mkdir()
    cli = fake_cli(str(root), {"m.py": "x = 1\n"})
    changes, problems = PC.simulate(cli, fix([], [], [], create=[{"path": path, "content": "a = 1\n"}]))
    assert not changes and problems


def test_ok_simulate_refuses_existing_file_ambiguous_and_lint(tmp_path):
    cli = fake_cli(str(tmp_path), {"m.py": "a = 1\na = 1\n"})
    _c, p = PC.simulate(cli, fix([], [], [], create=[{"path": "m.py", "content": "x"}]))
    assert "already exists" in p[0][2]
    _c, p = PC.simulate(cli, fix(["a = 1"], ["a = 2"], ["m.py"]))
    assert "ambiguous" in p[0][2]
    _c, p = PC.simulate(cli, fix(["a = 1\na = 1"], ["a = undefined_thing"], ["m.py"]))
    assert p and p[0][0] == "(lint)"


def test_ok_normalise_targets(tmp_path):
    cli = fake_cli(str(tmp_path), {"src/model.py": "x = 1\n"})
    out, problems = PC.normalise_targets(cli, fix(["x = 1"], ["x = 2"], ["model.py"]))
    assert out["files"] == ["src/model.py"] and problems == []
    out, problems = PC.normalise_targets(cli, fix(["x = 1"], ["x = 2"], [""]))
    assert out["files"] == ["src/model.py"]
    out, problems = PC.normalise_targets(cli, fix(["x"], ["y"], ["/etc/passwd"]))
    assert problems


def test_ok_parse_change_variants():
    cli = types.SimpleNamespace(_parse_code_fix=PulseCLI._parse_code_fix)
    body = {"old": ["a"], "new": ["b"], "files": ["m.py"], "create": [], "explanation": "e"}
    got = PC.parse_change(cli, "```json\n" + json.dumps(body) + "\n```")
    assert got["old"] == ["a"] and got["new"] == ["b"]
    only_create = {"old": [], "new": [], "files": [], "create": [{"path": "n.py", "content": "x=1\n"}]}
    assert PC.parse_change(cli, json.dumps(only_create))["create"][0]["path"] == "n.py"
    assert PC.parse_change(cli, json.dumps({"json": only_create}))["create"]
    assert PC.parse_change(cli, "not json") is None
    assert PC.parse_change(cli, json.dumps({"create": "bad"})) is None


def test_ok_read_text_and_scan(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    (tmp_path / "c.py").write_bytes(b"\x00abc")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "n.js").write_text("x")
    (tmp_path / "Makefile").write_text("all:\n")
    assert PC.read_text(str(tmp_path / "a.py")) == "x = 1\n"
    assert PC.read_text(str(tmp_path / "c.py")) is None
    found = sorted(os.path.basename(f) for f in PC.scan_project(str(tmp_path)))
    assert found == ["Makefile", "a.py", "c.py"]      # content is filtered later, by read_text


def test_ok_expand_paths_problems(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    files, problems = PC.expand_paths(str(tmp_path), ["a.py", "missing.py", "*.nothing"])
    assert files == [str(tmp_path / "a.py")]
    assert len(problems) == 2


def test_ok_render_diff_marks_new_and_modified(tmp_path):
    cli = fake_cli(str(tmp_path), {"m.py": "a = 1\n"})
    changes = {str(tmp_path / "m.py"): ("a = 1\n", "a = 2\n", False),
               str(tmp_path / "n.py"): ("", "b = 1\n", True)}
    text = PC.render_diff(cli, changes)
    assert "modified: m.py" in text and "new file: n.py" in text
    assert "-a = 1" in text and "+a = 2" in text and "+b = 1" in text


def test_ok_inside():
    assert PC._inside("/a/b", "/a/b/c.py")
    assert not PC._inside("/a/b", "/a/bc/d.py")
    assert not PC._inside("/a/b", "/a/b/../x.py")
