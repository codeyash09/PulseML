"""Pulse Code is held to the user's request in code, not only in its prompt:

  * the outline must be detailed (concrete leaves, verification, a backward task per forward task)
  * an edit that swaps the user's stack (numpy/cupy -> torch) is refused
  * stubs and dummy code are flagged at once and keep the turn open
  * edits need an outline on a multi-part request, and stop when one leaf has swallowed many edits
"""
import json
import os
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pulse import pulse_app as appmod
from pulse import pulse_cli as pc
from pulse import pulse_code_agent as native
from pulse import pulse_code_rules as rules
from pulse import pulse_supabase as cloud
from pulse import pulse_ui as ui


def item(id, content, parent=None, status="pending"):
    return {"id": id, "parent_id": parent, "content": content, "status": status}


def outline(*leaves):
    """One phase holding the given leaf texts (the first in progress)."""
    items = [item("p", "Build the model phase")]
    for index, text in enumerate(leaves):
        items.append(item(f"l{index}", text, "p", "in_progress" if index == 0 else "pending"))
    return items


# ---- the stack ----------------------------------------------------------------------------

NUMPY_CUPY = {"numpy", "cupy"}


def test_importing_torch_into_a_numpy_cupy_project_is_a_swap():
    why = rules.find_stack_swap("model.py", "import numpy as np\n", "import numpy as np\nimport torch\n",
                                NUMPY_CUPY, "add flash attention and bpe")
    assert "torch" in why and "cupy, numpy" in why and "message_user" in why


@pytest.mark.parametrize("request_text", ["port the model to torch", "use PyTorch for this", "add pytorch"])
def test_a_framework_the_user_asked_for_is_not_a_swap(request_text):
    assert rules.find_stack_swap("m.py", "", "import torch\n", NUMPY_CUPY, request_text) == ""


def test_a_framework_the_project_already_uses_is_not_a_swap():
    assert rules.find_stack_swap("m.py", "", "import torch\n", {"torch", "numpy"}, "add attention") == ""


def test_numpy_is_never_a_swap_and_neither_is_a_fresh_project():
    assert rules.find_stack_swap("m.py", "", "import numpy as np\n", {"torch"}, "x") == ""
    assert rules.find_stack_swap("m.py", "", "import torch\n", set(), "x") == ""      # nothing to swap away from


def test_a_new_autograd_framework_replacing_another_is_a_swap():
    assert rules.find_stack_swap("m.py", "", "import tensorflow as tf\n", {"torch"}, "add dropout")


def test_the_swap_is_found_in_every_import_form_and_only_in_python():
    for line in ("from torch import nn", "import torch.nn.functional as F", "from torch.nn import Linear"):
        assert rules.find_stack_swap("m.py", "", line + "\n", NUMPY_CUPY, "x"), line
    assert rules.find_stack_swap("notes.md", "", "import torch\n", NUMPY_CUPY, "x") == ""
    assert rules.find_stack_swap("m.py", "", "# import torch would be easier\nx = 1\n", NUMPY_CUPY, "x") == ""


def test_the_stack_is_read_from_real_imports_and_described_for_the_agent():
    stack = rules.detect_stack(["import numpy as np\nimport cupy as cp\n", "x = 'import torch'\n", "from jax import grad"])
    assert stack == {"numpy", "cupy", "jax"}
    assert "NO autograd" in rules.describe_stack(NUMPY_CUPY)
    assert "autograd: torch" in rules.describe_stack({"torch", "numpy"})
    assert rules.autograd_in_play(NUMPY_CUPY, "") is False
    assert rules.autograd_in_play(NUMPY_CUPY, "write it in pytorch") is True


# ---- stubs --------------------------------------------------------------------------------

def test_empty_function_bodies_are_stubs():
    src = ("def attention(q, k, v):\n    pass\n\n"
           "def backward(dout):\n    raise NotImplementedError\n\n"
           "def norm(x):\n    \"\"\"Layer norm.\"\"\"\n    ...\n\n"
           "class Block:\n    def forward(self, x):\n        pass\n")
    found = rules.find_stubs("model.py", "", src)
    names = " ".join(found)
    assert all(n in names for n in ("attention()", "backward()", "norm()", "Block.forward()"))


def test_real_code_abstract_methods_and_exceptions_are_not_stubs():
    src = ("from abc import ABC, abstractmethod\n\n"
           "class Layer(ABC):\n    @abstractmethod\n    def forward(self, x):\n        pass\n\n"
           "class Error(Exception):\n    pass\n\n"
           "def real(x):\n    return x * 2\n")
    assert rules.find_stubs("model.py", "", src) == []


def test_placeholder_words_in_added_lines_are_stubs_but_old_ones_are_not():
    before = "# TODO: old note\nx = 1\n"
    after = before + "data = np.random.randn(8, 4)  # dummy data for now\n# simplified version of attention\n"
    found = rules.find_stubs("train.py", before, after)
    assert len(found) == 2 and not any("old note" in f for f in found)


def test_a_stub_that_was_already_there_is_not_the_agents_to_answer_for():
    src = "def hook():\n    pass\n"
    assert rules.find_stubs("m.py", src, src + "x = 1\n") == []


def test_tests_may_say_dummy_and_mock():
    assert rules.find_stubs("tests/test_model.py", "", "dummy_input = mock_batch()  # dummy data\n") == []


# ---- the outline --------------------------------------------------------------------------

GOOD = ["Add merge loop to train_bpe in tokenizer/bpe.py", "Wire the tokenizer into data.py load_batch",
        "Run smoke_test with a probe on a tiny batch"]


def test_a_detailed_outline_is_accepted_and_a_tiny_one_is_exempt():
    assert rules.check_outline(outline(*GOOD), "x") == ""
    assert rules.check_outline([item("a", "Fix lr", status="in_progress"), item("b", "Done", status="pending")]) == ""


def test_vague_leaves_are_rejected_with_how_to_fix_them():
    why = rules.check_outline(outline("Implement attention", "Add tokenizer", GOOD[2]), "x")
    assert "too vague" in why and "'Implement attention'" in why and "file and function" in why


def test_an_outline_must_verify_something():
    why = rules.check_outline(outline(GOOD[0], GOOD[1], "Document the tokenizer module API surface"), "x")
    assert "verifies anything" in why


def test_forward_tasks_without_backward_tasks_are_rejected_when_there_is_no_autograd():
    leaves = ["Write attention forward in model.py", "Write MLP forward in model.py",
              "Write layernorm forward in model.py", "Run smoke_test with a probe"]
    why = rules.check_outline(outline(*leaves), "add flash attention", autograd=False)
    assert "3 forward-pass task(s) but only 0 backward-pass task(s)" in why and "no autograd" in why


def test_a_forward_only_outline_is_rejected_even_with_autograd():
    why = rules.check_outline(outline("Write attention forward in model.py", "Write the training loop in train.py",
                                      "Run smoke_test with a probe"), "x", autograd=True)
    assert "no backward pass" in why


def test_mirrored_forward_and_backward_tasks_need_a_gradient_check():
    leaves = ["Write attention forward in model.py", "Write attention backward dQ dK dV in model.py",
              "Write MLP forward in model.py", "Write MLP backward dW db in model.py", "Run smoke_test with a probe"]
    why = rules.check_outline(outline(*leaves), "x", autograd=False)
    assert "forward-pass task" not in why and "finite" in why
    assert rules.check_outline(outline(*leaves, "Gradient check attention and MLP with finite differences"),
                               "x", autograd=False) == ""


def test_a_merged_forward_plus_backward_leaf_covers_itself():
    leaves = ["Write attention forward and backward in model.py", "Gradient check it numerically in tests",
              "Run smoke_test with a probe"]
    assert rules.check_outline(outline(*leaves), "x", autograd=False) == ""


def test_with_autograd_one_backward_leaf_is_enough_and_inference_only_is_exempt():
    leaves = ["Write attention forward in model.py", "Call loss.backward and the optimizer step in train.py",
              "Run smoke_test with a probe"]
    assert rules.check_outline(outline(*leaves), "x", autograd=True) == ""
    only = outline("Write attention forward in model.py", "Write the sampler in generate.py", "Run smoke_test")
    assert rules.check_outline(only, "inference only: add the kv cache", autograd=False) == ""


def test_debugging_an_outline_that_mentions_loss_is_not_a_training_outline():
    leaves = ["Read the loss curve in run_status", "Find where lr is set in train.py", "Run smoke_test after the fix"]
    assert rules.check_outline(outline(*leaves), "why did the loss stall?", autograd=False) == ""


# ---- planning and the size of a leaf --------------------------------------------------------

def test_which_requests_need_an_outline():
    assert not rules.is_multi_step("lower the lr")
    assert rules.is_multi_step("add flash attention and bpe and stuff to the transformer")
    assert rules.is_multi_step("make the model, then train it, also log the loss")


def test_the_plan_gate():
    assert rules.plan_gate(False, "lower the lr", 0, 0) == ""                            # a one-line fix needs no outline
    assert "no task outline" in rules.plan_gate(False, "lower the lr", 1, 0)             # a second edit does
    assert "no task outline" in rules.plan_gate(False, "add flash attention and bpe and a tokenizer", 0, 0)
    assert rules.plan_gate(True, "x", 5, 5) == ""
    assert "too broad" in rules.plan_gate(True, "x", 9, rules.MAX_WRITES_PER_LEAF)
    assert rules.broad_leaf_warning(rules.WARN_WRITES_PER_LEAF - 1) == ""
    assert "split" in rules.broad_leaf_warning(rules.WARN_WRITES_PER_LEAF)


def test_the_brief_changes_with_the_stack():
    hand = rules.planning_brief(NUMPY_CUPY, "add attention")
    auto = rules.planning_brief({"torch"}, "add attention")
    assert "for EVERY forward-pass task" in hand and "finite-difference" in hand and "cupy, numpy" in hand
    assert "loss.backward()" in auto and "finite-difference" not in auto
    assert "Requirements" in hand and "production code" in hand


# ---- through the real tools -----------------------------------------------------------------

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
def cli(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "model.py").write_text("import numpy as np\nimport cupy as cp\n\n\ndef forward(x):\n    return np.tanh(x)\n")
    (root / "train.py").write_text("lr = 0.3\n")
    c = appmod._new_cli(str(root), [str(root / "model.py"), str(root / "train.py")])
    c.agent_provider = next(n for n, info in pc.PROVIDERS.items() if info.get("model") and info.get("env_key"))
    c.agent_key = "sk-test"
    c._sync_agent_turn = lambda *a, **k: None
    c.review = False
    monkeypatch.setattr(pc.litellm, "completion", lambda **k: pytest.fail("no model call expected here"))
    return c


def run(state, name, **args):
    return native._execute(state, {"name": name, "arguments": json.dumps(args)})


def read(state, name):
    return run(state, "read_file", path=name)


def test_the_agent_cannot_switch_a_numpy_cupy_project_to_torch(cli):
    state = native._State(cli, "add flash attention to the transformer")
    read(state, "model.py")
    out = run(state, "edit_file", path="model.py", old_string="import cupy as cp\n",
              new_string="import cupy as cp\nimport torch\n")
    assert out.startswith("NOT APPLIED") and "torch" in out and "cupy, numpy" in out
    assert "import torch" not in Path(cli._project_root, "model.py").read_text()
    # writing the missing piece with the stack in use is fine
    ok = run(state, "edit_file", path="model.py", old_string="    return np.tanh(x)\n",
             new_string="    return np.tanh(x) * 1.0\n")
    assert "NOT APPLIED" not in ok


def test_a_new_file_importing_torch_is_refused_too(cli):
    state = native._State(cli, "add a tokenizer")
    out = run(state, "write_file", path="attn.py", content="import torch\n\n\ndef attend(q):\n    return q\n")
    assert out.startswith("NOT APPLIED") and "torch" in out
    assert not Path(cli._project_root, "attn.py").exists()


def test_torch_is_allowed_when_the_user_asked_for_it(cli):
    state = native._State(cli, "port this to pytorch")
    read(state, "model.py")
    out = run(state, "edit_file", path="model.py", old_string="import cupy as cp\n",
              new_string="import cupy as cp\nimport torch\n")
    assert "NOT APPLIED" not in out


def test_an_earlier_request_in_the_session_that_asked_for_torch_counts(cli):
    cli.native_history.append({"role": "user", "content": "context...\n\nRequest: move the model to torch"})
    state = native._State(cli, "now add dropout")
    read(state, "model.py")
    out = run(state, "edit_file", path="model.py", old_string="import cupy as cp\n",
              new_string="import cupy as cp\nimport torch\n")
    assert "NOT APPLIED" not in out


def test_a_stub_is_flagged_in_the_edit_result_and_keeps_the_turn_open(cli):
    state = native._State(cli, "lower the lr")
    state.todos = [item("t", "Lower the lr in train.py", status="in_progress")]
    out = run(state, "write_file", path="attn.py", content="def attend(q, k, v):\n    pass\n")
    assert "PLACEHOLDER CODE" in out and "attend()" in out
    reason = native._stub_nudge(state)
    assert reason and "placeholder code" in reason and "attend()" in reason
    read(state, "attn.py")
    run(state, "write_file", path="attn.py", content="def attend(q, k, v):\n    return q @ k.T @ v\n")
    assert native._stub_nudge(state) is None


def test_edits_need_an_outline_on_a_multi_part_request(cli):
    state = native._State(cli, "add flash attention and bpe and a full training loop to the transformer")
    out = run(state, "write_file", path="bpe.py", content="def train_bpe(text):\n    return text\n")
    assert out.startswith("NOT APPLIED -- no task outline yet")
    assert not Path(cli._project_root, "bpe.py").exists()


def test_a_one_line_request_may_edit_without_an_outline_once(cli):
    state = native._State(cli, "lower the lr")
    read(state, "train.py")
    assert "NOT APPLIED" not in run(state, "edit_file", path="train.py", old_string="0.3", new_string="0.1")
    assert "no task outline" in run(state, "edit_file", path="train.py", old_string="0.1", new_string="0.05")


def test_edits_stop_when_one_leaf_has_swallowed_many_of_them(cli):
    state = native._State(cli, "lower the lr")
    plan = [item("t", "Tune the learning rate in train.py", status="in_progress")]
    assert native._t_todo_write(state, {"todos": plan}).startswith("Task outline updated")
    read(state, "train.py")
    results = []
    for i in range(rules.MAX_WRITES_PER_LEAF + 1):
        results.append(run(state, "edit_file", path="train.py", old_string=f"lr = {0.3 if i == 0 else i - 1 + 0.5}",
                           new_string=f"lr = {i + 0.5}"))
    assert any("split" in r and "edits under one outline state" in r for r in results[:rules.MAX_WRITES_PER_LEAF])
    assert results[-1].startswith("NOT APPLIED -- 6 edits since you last updated the task outline")
    # an identical outline does not reset it; a real update does
    assert native._t_todo_write(state, {"todos": plan}).startswith("Outline unchanged")
    plan2 = [item("t", "Tune the learning rate in train.py", status="completed"),
             item("v", "Run smoke_test on the new lr", status="in_progress")]
    assert native._t_todo_write(state, {"todos": plan2}).startswith("Task outline updated")
    assert state.writes_since_outline == 0


def test_todo_write_rejects_a_forward_only_outline_for_a_numpy_project(cli):
    state = native._State(cli, "add flash attention and bpe to the transformer")
    bad = [item("a", "Attention work"),
           item("a1", "Write attention forward in model.py", "a", "in_progress"),
           item("a2", "Write MLP forward in model.py", "a"),
           item("v", "Run smoke_test with a probe", "a")]
    out = native._t_todo_write(state, {"todos": bad})
    assert out.startswith("Outline not accepted") and "2 forward-pass task(s) but only 0 backward-pass" in out
    assert state.todos == []
    good = bad[:3] + [item("a3", "Write attention backward dQ dK dV in model.py", "a"),
                      item("a4", "Write MLP backward dW db in model.py", "a"),
                      item("a5", "Gradient check attention and MLP with finite differences", "a"),
                      bad[3]]
    assert native._t_todo_write(state, {"todos": good}).startswith("Task outline updated")


def test_a_torch_project_is_not_forced_to_hand_write_backward_passes(cli):
    Path(cli._project_root, "net.py").write_text("import torch\n")
    cli.add_files([str(Path(cli._project_root, "net.py"))], focus=False)
    state = native._State(cli, "add attention to the net")
    plan = [item("a", "Attention work"),
            item("a1", "Write attention forward in net.py", "a", "in_progress"),
            item("a2", "Call loss.backward and the optimizer step in train.py", "a"),
            item("v", "Run smoke_test with a probe", "a")]
    assert native._t_todo_write(state, {"todos": plan}).startswith("Task outline updated")


def test_the_turn_sends_the_planning_brief_beside_the_request(cli, monkeypatch):
    sent = []

    def completion(**kwargs):
        sent.append(json.loads(json.dumps(kwargs["messages"])))     # as sent: the turn rewrites the list afterwards
        message = types.SimpleNamespace(content="It is in train.py.", tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
                                     usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    monkeypatch.setattr(native.litellm, "completion", completion)
    native.run_native_turn(cli, "where is lr set?")
    first = [m for m in sent[0] if m["role"] == "user"][-1]["content"]
    assert "Planning brief for this request" in first and "cupy, numpy" in first
    assert first.rstrip().endswith("Request: where is lr set?")
    assert "Planning brief" not in json.dumps(cli.native_history)       # not kept in the history afterwards


def test_the_prompt_tells_the_agent_what_it_is_held_to():
    prompt = native.SYSTEM_PROMPT
    for phrase in ("Planning: decompose until the whole job is written down", "Requirements",
                   "Follow the request exactly, including how", "Never swap in a different framework",
                   "Evidence before patches", "three independent observations",
                   "review_evidence", "disconfirming observation",
                   "plausible alternative", "use the `experiment` tool",
                   "Production code, not placeholders", "no `pass` or `...` bodies",
                   "every forward pass has a backward pass", "finite differences"):
        assert phrase in prompt, phrase


def test_the_text_pipeline_prompt_carries_the_same_rules():
    from pulse import pulse_code
    prompt = pulse_code.CODE_SYSTEM_PROMPT
    for phrase in ("Never switch because another framework is easier", "No stubs",
                   "three independent kinds of primary evidence", "disconfirming evidence",
                   "EXPERIMENT:", "Every forward pass has a backward pass",
                   "finite-difference gradient check"):
        assert phrase in prompt, phrase


def test_experiment_directive_is_available_to_the_text_coding_agent():
    from pulse import pulse_code
    cleaned, requests = pulse_code._CodeAgentCLI._extract_new_directives(
        "Need a controlled comparison.\nEXPERIMENT: experiments/fix.json"
    )
    assert "EXPERIMENT:" not in cleaned
    assert requests["experiment"] == ["experiments/fix.json"]