"""`pulse code`'s check_shape and smoke_test tools: real subprocess probes in a real (tiny) project,
with a numpy "model" standing in for torch. The stages -- compile check, shape-traced probe, the
project's own tests -- stop at the first failure, and the agent is nudged to run them after edits."""
import os

import pytest

import pulse.pulse_cli as pc
from pulse import pulse_app as appmod
from pulse import pulse_code_agent as native
from pulse import pulse_ui as ui
from pulse import pulse_supabase as cloud

MODEL = '''import numpy as np


class TinyNet:
    def __init__(self, in_features, hidden, classes):
        self.w1 = np.random.randn(in_features, hidden).astype("float32")
        self.w2 = np.random.randn(hidden, classes).astype("float32")

    def __call__(self, x):
        return np.maximum(x @ self.w1, 0) @ self.w2
'''

GOOD_PROBE = ("from model import TinyNet\nimport numpy as np\nm = TinyNet(784, 256, 10)\n"
              "x = np.zeros((4, 784), dtype='float32')\nlogits = m(x)\n")
BAD_PROBE = GOOD_PROBE.replace("784, 256", "256, 64")          # first layer built for 256, fed 784


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    saved_env, cwd = dict(os.environ), os.getcwd()
    for name in ("OPENROUTER_API_KEY", "PULSE_PROVIDER", "PULSE_CONFIG", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "pulsehome" / "credentials.json")
    monkeypatch.setattr(cloud, "save_cached_profile", lambda **k: None)
    yield
    ui.set_host(None)
    pc.set_agent_observer(None)
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(saved_env)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "model.py").write_text(MODEL)
    (root / "train.py").write_text("print('train')\n")
    (root / "tests" / "test_model.py").write_text(
        "import numpy as np\nfrom model import TinyNet\n\n\n"
        "def test_forward():\n    assert TinyNet(4, 8, 3)(np.zeros((2, 4), dtype='float32')).shape == (2, 3)\n")
    return root


@pytest.fixture
def state(project):
    cli = appmod._new_cli(str(project), [str(project / "train.py")])
    return native._State(cli, "test request")


def _changed(state, project, name, text):
    """Pretend the agent just wrote `name` with `text` (the way _commit records it)."""
    path = str(project / name)
    before = ""
    (project / name).write_text(text)
    state.changes[path] = (before, text, True)
    state.dirty = True
    return path


# ---- registered like every other tool -----------------------------------------------------

def test_tools_are_registered_with_matching_schema_and_handler():
    names = [t["function"]["name"] for t in native.TOOLS]
    for tool in ("check_shape", "smoke_test"):
        assert names.count(tool) == 1
        assert tool in native._HANDLERS
        assert tool in native._LOUD            # they may need to ask the user, so they are never hushed


def test_the_prompt_tells_the_agent_to_use_them():
    assert "smoke_test" in native.SYSTEM_PROMPT and "check_shape" in native.SYSTEM_PROMPT
    assert "probe" in native.SYSTEM_PROMPT


# ---- check_shape --------------------------------------------------------------------------

def test_check_shape_reports_real_shapes_and_holds_expectations(state):
    out = native._t_check_shape(state, {"code": GOOD_PROBE, "exprs": ["x", "logits", "m.w1"],
                                        "expect": {"x": "(B, 784)", "logits": "(B, 10)"}})
    assert out.startswith("CHECK_SHAPE: ok")
    assert "logits: numpy.ndarray float32 (4, 10)" in out
    assert "m.w1: numpy.ndarray float32 (784, 256)" in out
    assert "OK   logits (4, 10) vs (B, 10)" in out


def test_check_shape_runs_in_the_project_so_its_modules_import(state):
    out = native._t_check_shape(state, {"code": "import model", "exprs": ["model.TinyNet(3, 2, 1).w1"]})
    assert "(3, 2)" in out


def test_check_shape_explains_a_failure_and_shows_what_existed(state):
    out = native._t_check_shape(state, {"code": BAD_PROBE})
    assert out.startswith("CHECK_SHAPE: FAIL")
    assert "PROBE FAILED: ValueError" in out
    assert "A layer built for 256 inputs is being fed 784" in out
    assert "model.py" in out and "<probe>" in out               # both frames, with their source lines
    assert "x: numpy.ndarray float32 (4, 784)" in out           # state at the point of failure


def test_check_shape_flags_an_expectation_that_does_not_hold(state):
    out = native._t_check_shape(state, {"code": GOOD_PROBE, "expect": {"logits": "(B, 5)"}})
    assert out.startswith("CHECK_SHAPE: FAIL") and "dimension 1: expected 5, got 10" in out


def test_check_shape_symbols_must_agree_across_expressions(state):
    code = "import numpy as np\nx = np.zeros((8, 3))\ny = np.zeros((4,))\n"
    out = native._t_check_shape(state, {"code": code, "expect": {"x": "(B, 3)", "y": "(B,)"}})
    assert "FAIL" in out and "B is 8 (from x)" in out


def test_check_shape_accepts_a_single_expression_string(state):
    out = native._t_check_shape(state, {"code": GOOD_PROBE, "exprs": "logits"})
    assert "logits: numpy.ndarray float32 (4, 10)" in out


def test_check_shape_shows_what_the_snippet_printed(state):
    out = native._t_check_shape(state, {"code": "print('hello from the probe')\nimport numpy as np\nz = np.zeros(2)"})
    assert "hello from the probe" in out


def test_check_shape_rejects_an_unreadable_spec_without_running_anything(state):
    out = native._t_check_shape(state, {"code": "x = 1", "expect": {"x": "(B, 3x)"}})
    assert out.startswith("Not run") and "3x" in out


def test_check_shape_with_nothing_to_do_says_so(state):
    assert "Nothing to check" in native._t_check_shape(state, {})


def test_check_shape_kills_a_probe_that_runs_too_long(state):
    out = native._t_check_shape(state, {"code": "import time\ntime.sleep(60)", "timeout": 5})
    assert "killed after 5s" in out and "tiny input" in out


def test_check_shape_reports_a_probe_process_that_dies_before_reporting(state):
    out = native._t_check_shape(state, {"code": "import os\nos._exit(7)"})
    assert "died before it could report" in out and "exit 7" in out


def test_a_destructive_probe_pauses_for_the_user_and_does_not_run_when_declined(state, project, monkeypatch):
    victim = project / "keep_me.txt"
    victim.write_text("data")
    asked = []
    monkeypatch.setattr(state.cli, "_confirm_terminal_command",
                        lambda display, flags, purpose=None: asked.append(display) or False)
    out = native._t_check_shape(state, {"code": f"import os\nos.remove({str(victim)!r})"})
    assert asked, "a snippet that deletes a file was run without asking"
    assert out.startswith("NOT RUN")
    assert victim.exists()


def test_a_harmless_probe_does_not_interrupt_the_user(state, monkeypatch):
    monkeypatch.setattr(state.cli, "_confirm_terminal_command",
                        lambda *a, **k: pytest.fail("asked about a harmless probe"))
    assert native._t_check_shape(state, {"code": GOOD_PROBE}).startswith("CHECK_SHAPE: ok")


# ---- smoke_test ---------------------------------------------------------------------------

def test_smoke_test_passes_all_three_stages_and_clears_dirty(state, project):
    _changed(state, project, "extra.py", "VALUE = 1\n")
    out = native._t_smoke_test(state, {"probe": GOOD_PROBE, "expect": {"logits": "(B, 10)"}})
    assert out.startswith("SMOKE TEST: PASS")
    assert "[1/3] compile   ok (1 changed Python file)" in out
    assert "[2/3] probe     ok (1 shape expectation held)" in out
    assert "[3/3] suite     ok (pytest: 1 passed" in out
    assert state.smoke_ok is True and state.dirty is False


def test_smoke_test_compile_failure_stops_everything_after_it(state, project):
    _changed(state, project, "broken.py", "def f(:\n    pass\n")
    out = native._t_smoke_test(state, {"probe": GOOD_PROBE})
    assert out.startswith("SMOKE TEST: FAIL")
    assert "[1/3] compile   FAILED" in out and "broken.py" in out
    assert "[2/3] probe     not run" in out and "[3/3] suite     not run" in out
    assert state.smoke_ok is False and state.dirty is True


def test_smoke_test_catches_an_undefined_name_the_change_introduced(state, project):
    _changed(state, project, "oops.py", "def f():\n    return undefined_thing\n")
    out = native._t_smoke_test(state, {})
    assert "[1/3] compile   FAILED" in out and "undefined_thing" in out


def test_smoke_test_probe_shape_bug_is_named_and_the_suite_does_not_run(state, project):
    out = native._t_smoke_test(state, {"probe": BAD_PROBE})
    assert out.startswith("SMOKE TEST: FAIL")
    assert "[2/3] probe     FAILED" in out and "A layer built for 256 inputs is being fed 784" in out
    assert "[3/3] suite     not run (an earlier stage failed)" in out
    assert state.smoke_ok is False


def test_smoke_test_failed_expectation_fails_the_probe_stage(state):
    out = native._t_smoke_test(state, {"probe": GOOD_PROBE, "expect": {"logits": "(B, 5)"}})
    assert "[2/3] probe     FAILED" in out and "expected 5, got 10" in out


def test_smoke_test_without_a_probe_says_so_instead_of_pretending(state):
    out = native._t_smoke_test(state, {})
    assert "[2/3] probe     skipped (none given" in out


def test_smoke_test_reports_a_failing_project_test_with_its_output(state, project):
    (project / "tests" / "test_model.py").write_text(
        "def test_forward():\n    assert 1 + 1 == 3, 'arithmetic is broken'\n")
    state.dirty = True
    out = native._t_smoke_test(state, {})
    assert out.startswith("SMOKE TEST: FAIL")
    assert "[3/3] suite     FAILED (pytest" in out and "arithmetic is broken" in out
    assert state.dirty is True and state.smoke_ok is False


def test_smoke_test_can_skip_the_suite(state, project):
    (project / "tests" / "test_model.py").write_text("def test_forward():\n    assert False\n")
    out = native._t_smoke_test(state, {"suite": False})
    assert out.startswith("SMOKE TEST: PASS") and "suite     skipped (suite=false)" in out


def test_smoke_test_with_no_tests_in_the_project_skips_that_stage(state, project):
    (project / "tests" / "test_model.py").unlink()
    (project / "tests").rmdir()
    out = native._t_smoke_test(state, {})
    assert out.startswith("SMOKE TEST: PASS") and "no tests found in the project" in out


def test_smoke_test_pytest_that_collects_nothing_is_a_skip_not_a_failure(state, project):
    (project / "tests" / "test_model.py").unlink()
    (project / "pytest.ini").write_text("[pytest]\n")
    out = native._t_smoke_test(state, {})
    assert out.startswith("SMOKE TEST: PASS") and "pytest collected no tests" in out


def test_smoke_test_falls_back_to_unittest_when_pytest_is_missing(state, project, monkeypatch):
    (project / "tests" / "test_model.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n")
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda name: None)
    command, label = native._project_test_command(state)
    assert label == "unittest" and "unittest discover" in command


def test_smoke_test_rejects_an_unreadable_spec_before_running_anything(state):
    assert native._t_smoke_test(state, {"probe": "x = 1", "expect": {"x": "(B, 3x)"}}).startswith("Not run")


def test_a_destructive_probe_in_a_smoke_test_is_gated_like_run_command(state, project, monkeypatch):
    victim = project / "keep_me.txt"
    victim.write_text("data")
    monkeypatch.setattr(state.cli, "_confirm_terminal_command", lambda *a, **k: False)
    out = native._t_smoke_test(state, {"probe": f"import os\nos.remove({str(victim)!r})", "suite": False})
    assert "[2/3] probe     FAILED" in out and "NOT RUN" in out
    assert victim.exists()


def test_project_test_command_ignores_virtualenvs_and_hidden_folders(state, project):
    (project / "tests" / "test_model.py").unlink()
    (project / "tests").rmdir()
    for hidden in (".venv/lib", "node_modules/x", ".git/hooks"):
        (project / hidden).mkdir(parents=True)
        (project / hidden / "test_something.py").write_text("def test_x():\n    pass\n")
    command, why = native._project_test_command(state)
    assert command is None and "no tests found" in why


# ---- the nudge ----------------------------------------------------------------------------

def test_after_edits_the_agent_is_pointed_at_smoke_test(state):
    state.dirty = True
    note = native._nudge(state)
    assert "smoke_test" in note and "probe" in note


def test_after_a_failed_smoke_test_the_nudge_says_to_fix_and_rerun(state):
    state.dirty, state.smoke_ok = True, False
    note = native._nudge(state)
    assert "failed" in note and "smoke_test again" in note


def test_a_passing_smoke_test_lets_the_agent_finish(state, project):
    _changed(state, project, "extra.py", "VALUE = 1\n")
    native._t_smoke_test(state, {})
    assert native._nudge(state) is None


# ---- how the result is shown --------------------------------------------------------------

def test_the_call_is_summarised_for_the_person():
    assert native._brief("smoke_test", {"probe": "x = 1"}) == "smoke_test  compile, probe, suite"
    assert native._brief("smoke_test", {}) == "smoke_test  compile, suite"
    assert native._brief("check_shape", {"code": "import numpy\nx = 1", "exprs": ["x", "y"]}) == "check_shape  x, y"
    assert native._brief("check_shape", {"code": "import numpy\nx = 1"}) == "check_shape  import numpy"