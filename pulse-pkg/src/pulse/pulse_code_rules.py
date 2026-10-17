"""Rules Pulse Code enforces in code, because the prompt alone did not hold.

Each one answers a way the agent was seen to go wrong:

  * it planned shallowly, or not at all           -> check_outline, plan_gate
  * it swapped the user's stack for an easier one -> find_stack_swap, detect_stack
  * it left stubs and dummy code                  -> find_stubs
  * it wrote a forward pass and no backward pass  -> check_outline (backward mirror)

Everything here is a pure function of text (no model, no terminal, no project state), so each rule
can be tested on its own. pulse_code_agent decides where to call them.
"""
from __future__ import annotations

import ast
import difflib
import os
import re
from typing import Dict, Iterable, List, Optional, Set

# --------------------------------------------------------------------------------------
# The stack a project is built on
# --------------------------------------------------------------------------------------

# import name -> the names a user would use for it in a request
_FRAMEWORKS = {
    "torch": ("torch", "pytorch"),
    "tensorflow": ("tensorflow", "tf"),
    "jax": ("jax", "flax"),
    "keras": ("keras",),
    "mxnet": ("mxnet",),
    "paddle": ("paddle", "paddlepaddle"),
    "tinygrad": ("tinygrad",),
    "cupy": ("cupy",),
    "numpy": ("numpy",),
}
# these give gradients for free; numpy / cupy do not
AUTOGRAD = frozenset({"torch", "tensorflow", "jax", "keras", "mxnet", "paddle", "tinygrad"})
# The rule is symmetric: whatever the project is built on, a DIFFERENT framework arriving is a swap,
# whichever way round (torch -> jax, jax -> torch, cupy -> torch, ...). The only exception is numpy as a
# newcomer: a project on any framework uses numpy for data handling, so importing it is ordinary.
_SWAP_TARGETS = frozenset(_FRAMEWORKS) - {"numpy"}


def imports_of(text: str) -> Set[str]:
    """The framework names (keys of _FRAMEWORKS) that `text` imports."""
    found: Set[str] = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        for match in re.finditer(r"(?m)^\s*(?:import|from)\s+([A-Za-z_][\w]*)", text):
            if match.group(1) in _FRAMEWORKS:
                found.add(match.group(1))
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root in _FRAMEWORKS:
                found.add(root)
    return found


def detect_stack(texts: Iterable[str]) -> Set[str]:
    """Every framework any of the project's files imports."""
    stack: Set[str] = set()
    for text in texts:
        stack |= imports_of(text)
    return stack


def has_autograd(stack: Iterable[str]) -> bool:
    return bool(set(stack) & AUTOGRAD)


def autograd_in_play(stack: Iterable[str], request_text: str = "") -> bool:
    """Gradients come for free: the project imports an autograd framework, or the request names one."""
    return has_autograd(stack) or any(_asked_for(request_text, name) for name in AUTOGRAD)


def describe_stack(stack: Iterable[str]) -> str:
    stack = sorted(set(stack))
    if not stack:
        return "no array/deep-learning framework detected yet: use what the request names"
    autograd = [name for name in stack if name in AUTOGRAD]
    if autograd:
        return f"{', '.join(stack)} (autograd: {', '.join(autograd)})"
    return f"{', '.join(stack)} (NO autograd: every backward pass is hand-written)"


_NEGATION = re.compile(r"(?i)\b(?:don'?t use|do not use|never use|stop using|stay off|instead of|rather than|"
                       r"don'?t|do not|never|without|avoid|remove|drop|ditch|not|no)\b")
_AFFIRM = re.compile(r"(?i)\b(?:use|using|switch to|port(?:ed)? to|move to|go with|convert(?:ed)? to|"
                     r"write (?:it )?in|build with|but)\b")


def _asked_for(request_text: str, framework: str) -> bool:
    """True when the request names `framework` and means it: "port it to jax", "use pytorch".

    A negated mention ("don't use torch", "no torch", "instead of torch") is the opposite of asking
    for it. Negation covers only what it directly governs: it stops at a comma or full stop, and a
    later "use X" in the same clause is affirmative ("instead of torch use jax" asks for jax,
    "don't use jax, use torch" asks for torch)."""
    low = (request_text or "").lower()
    for alias in _FRAMEWORKS[framework]:
        for match in re.finditer(rf"(?<![\w]){re.escape(alias)}(?![\w])", low):
            clause = re.split(r"[,;:.!?\n]", low[max(0, match.start() - 40):match.start()])[-1]
            negations = list(_NEGATION.finditer(clause))
            if not negations or _AFFIRM.search(clause[negations[-1].end():]):
                return True
    return False


def find_stack_swap(path: str, before: str, after: str, project_stack: Set[str], request_text: str) -> str:
    """"" when the edit is fine, else why it is refused.

    An edit is a swap when it introduces an import of a framework the project does not use
    anywhere, into a project that is built on a different one, and the user has not explicitly asked
    for it. Which framework the project is built on is read from its own files: nothing is assumed."""
    if not str(path).endswith(".py"):
        return ""
    new = (imports_of(after) - imports_of(before)) & _SWAP_TARGETS
    new -= project_stack
    new = {name for name in new if not _asked_for(request_text, name)}
    if not new:
        return ""
    built_on = sorted(project_stack & (_SWAP_TARGETS | {"numpy"}))
    if not built_on:
        return ""                    # nothing to swap away from: a fresh project may choose
    chosen = ", ".join(sorted(new))
    return (f"this edit imports {chosen}, but this project is built on {', '.join(built_on)} and the user has not "
            f"asked to change that. The framework the project uses is a requirement, not a default to improve on: "
            f"do not switch to something easier. Implement it with {', '.join(built_on)} (write the missing piece "
            "yourself: a kernel, a backward pass, a tokenizer...). If it is truly impossible with the current "
            f"framework, stop and tell the user exactly what blocks it with message_user, and ask. Only if the user "
            f"explicitly tells you to use {chosen} may you bring it in.")


# --------------------------------------------------------------------------------------
# Stubs and dummy code
# --------------------------------------------------------------------------------------

_STUB_WORDS = re.compile(
    r"(?i)\b(?:TODO|FIXME|XXX|placeholder|dummy|stubbed?|fake (?:data|implementation)|"
    r"mock(?:ed)? (?:data|implementation)|toy (?:example|version|implementation)|"
    r"simplified (?:version|implementation|for)|for (?:now|brevity|simplicity)|"
    r"in (?:a|the) real (?:implementation|system|app|world)|left as an exercise|"
    r"not (?:yet )?implemented|implement(?:ed)? later|real implementation would|"
    r"replace (?:this )?with (?:real|actual))\b")
_ABSTRACT_DECORATORS = {"abstractmethod", "overload", "abstractproperty"}
_ABSTRACT_BASES = {"ABC", "Protocol"}


def _is_test_file(path: str) -> bool:
    parts = str(path).replace("\\", "/").lower().split("/")
    name = parts[-1]
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in parts or "test" in parts


def _decorator_names(node) -> Set[str]:
    names = set()
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        names.add(target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", ""))
    return names


def _empty_body_kind(node) -> str:
    """How a function body is empty: "pass", "...", "raise NotImplementedError", or ""."""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]                                  # a docstring is not an implementation
    if len(body) != 1:
        return ""
    only = body[0]
    if isinstance(only, ast.Pass):
        return "pass"
    if isinstance(only, ast.Expr) and isinstance(getattr(only, "value", None), ast.Constant) \
            and only.value.value is Ellipsis:
        return "..."
    if isinstance(only, ast.Raise) and only.exc is not None:
        call = only.exc.func if isinstance(only.exc, ast.Call) else only.exc
        if getattr(call, "id", getattr(call, "attr", "")) == "NotImplementedError":
            return "raise NotImplementedError"
    return ""


def _stub_functions(text: str) -> Dict[str, tuple]:
    """qualified name -> (line, kind) for each function whose body is empty and is not abstract."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    found: Dict[str, tuple] = {}

    def visit(node, prefix, abstract_class):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                bases = {getattr(b, "id", getattr(b, "attr", "")) for b in child.bases}
                visit(child, f"{prefix}{child.name}.", bool(bases & _ABSTRACT_BASES))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = _empty_body_kind(child)
                if kind and not abstract_class and not (_decorator_names(child) & _ABSTRACT_DECORATORS):
                    found[f"{prefix}{child.name}"] = (child.lineno, kind)
                visit(child, f"{prefix}{child.name}.", False)
            else:
                visit(child, prefix, abstract_class)

    visit(tree, "", False)
    return found


def find_stubs(path: str, before: str, after: str, limit: int = 8) -> List[str]:
    """Placeholder code this change ADDED (pre-existing stubs are not the agent's to answer for)."""
    if _is_test_file(path):
        return []                      # tests legitimately say "dummy" and "mock"
    out: List[str] = []
    if str(path).endswith(".py"):
        old = _stub_functions(before or "")
        for name, (line, kind) in _stub_functions(after).items():
            if name not in old:
                out.append(f"{path}:{line} {name}() has an empty body ({kind})")
    before_lines = (before or "").splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("insert", "replace"):
            continue
        for offset in range(j1, j2):
            text = after_lines[offset]
            hit = _STUB_WORDS.search(text)
            if hit:
                out.append(f"{path}:{offset + 1} says \"{hit.group(0)}\": {text.strip()[:90]}")
    return out[:limit]


# --------------------------------------------------------------------------------------
# The task outline
# --------------------------------------------------------------------------------------

_VERIFY = re.compile(r"(?i)\b(?:verif\w*|tests?|check\w*|probe|smoke|assert\w*|validat\w*|run|"
                     r"overfit|gradcheck|benchmark)\b")
_FORWARD = re.compile(r"(?i)\bforward(?:[- ]pass)?\b")
_BACKWARD = re.compile(r"(?i)back-?prop\w*|\bbackward\b|\bgradients?\b|\bgrads?\b|autograd|\bvjp\b|chain rule")
_TRAINING = re.compile(r"(?i)\btrain(?:ing)?[- ](?:loop|step|script)\b")
_GRADCHECK = re.compile(r"(?i)grad(?:ient)?[- ]?check|finite[- ]difference|numerical(?:ly)? (?:check|gradient)")
_INFERENCE_ONLY = re.compile(r"(?i)inference[- ]only|eval(?:uation)?[- ]only|no training|without training")

MIN_LEAF_WORDS = 3
# Edits allowed under one outline state before the agent must update the outline (mark the finished
# leaf done / split a leaf that turned out to be broad). A leaf is "one edit or one command".
WARN_WRITES_PER_LEAF = 3
MAX_WRITES_PER_LEAF = 6


def check_outline(items: List[dict], request_text: str = "", autograd: bool = True) -> str:
    """"" when the outline is detailed enough to work from, else what to fix.

    Runs on outlines of three or more items: a one-line fix needs one or two leaves, not a plan.
    `items` are {id, parent_id, content, status} dicts that already passed structural validation."""
    if len(items) < 3:
        return ""
    parents = {item["parent_id"] for item in items if item.get("parent_id")}
    leaves = [item for item in items if item["id"] not in parents]
    problems: List[str] = []

    vague = [leaf["content"] for leaf in leaves
             if len(leaf["content"].split()) < MIN_LEAF_WORDS and not _VERIFY.search(leaf["content"])]
    if vague:
        shown = "; ".join(repr(v) for v in vague[:4])
        problems.append(f"leaf task(s) too vague to act on: {shown}. A leaf is one edit or one command: say what "
                        "changes and where (file and function/class), e.g. 'Add merge loop to train_bpe in "
                        "tokenizer/bpe.py'.")

    if not any(_VERIFY.search(leaf["content"]) for leaf in leaves):
        problems.append("no leaf verifies anything. Add leaves that check the work: run smoke_test with a probe, "
                        "run the tests, check shapes and gradients.")

    if not _INFERENCE_ONLY.search(request_text or ""):
        problems += _backward_problems(items, leaves, autograd)

    if not problems:
        return ""
    return ("Outline not accepted -- fix these and call todo_write again:\n- " + "\n- ".join(problems))


def _backward_problems(items: List[dict], leaves: List[dict], autograd: bool) -> List[str]:
    text = " ".join(item["content"] for item in items)
    if not (_FORWARD.search(text) or _TRAINING.search(text)):
        return []
    has_backward = bool(_BACKWARD.search(text))
    if autograd:
        # gradients come for free; the outline still has to say where they are used
        return [] if has_backward else [
            "the outline has forward-pass / training work but no backward pass. Add a leaf for loss.backward() "
            "and the optimizer step, and one that checks every parameter receives a gradient."]
    forward_only = [leaf for leaf in leaves
                    if _FORWARD.search(leaf["content"]) and not _BACKWARD.search(leaf["content"])]
    backward_only = [leaf for leaf in leaves
                     if _BACKWARD.search(leaf["content"]) and not _FORWARD.search(leaf["content"])]
    problems: List[str] = []
    if len(forward_only) > len(backward_only):
        missing = [leaf["content"] for leaf in forward_only][len(backward_only):]
        problems.append(
            f"{len(forward_only)} forward-pass task(s) but only {len(backward_only)} backward-pass task(s), and this "
            "project has no autograd. Mirror each forward task with its own backward task (same component, same "
            f"detail), or merge them as 'X forward + backward'. Still unmatched: {'; '.join(repr(m) for m in missing[:4])}.")
    elif not has_backward:
        problems.append("the outline has training work but no backward pass, and this project has no autograd. "
                        "Add the backward tasks (and an optimizer update).")
    if has_backward and not any(_GRADCHECK.search(item["content"]) for item in items):
        problems.append("no leaf checks the hand-written gradients. Add one that compares them with finite "
                        "differences on a tiny model (float64).")
    return problems


# --------------------------------------------------------------------------------------
# When planning is required, and when a leaf has grown too big
# --------------------------------------------------------------------------------------

def is_multi_step(request_text: str) -> bool:
    """A request long or listy enough that it needs an outline before the first edit."""
    text = (request_text or "").strip()
    if len(text.split()) >= 14:
        return True
    separators = len(re.findall(r",|;|\n|\band\b|\bthen\b|\balso\b", text, flags=re.I))
    return separators >= 2


def plan_gate(has_outline: bool, request_text: str, writes_total: int, writes_since_outline: int) -> str:
    """"" when this edit may go ahead, else the reason it may not."""
    if not has_outline:
        if writes_total >= 1 or is_multi_step(request_text):
            return ("NOT APPLIED -- no task outline yet. Call todo_write first: a 'Requirements' phase (one leaf "
                    "per thing the request names), then a phase per component broken down to single edits and "
                    "commands, including verification. A one-line change needs only one or two leaves.")
        return ""
    if writes_since_outline >= MAX_WRITES_PER_LEAF:
        return (f"NOT APPLIED -- {writes_since_outline} edits since you last updated the task outline, so the "
                "current leaf is too broad. Call todo_write: mark what is finished completed, split this leaf into "
                "smaller leaves (one edit or command each), and activate the next one. Then continue.")
    return ""


def broad_leaf_warning(writes_since_outline: int) -> str:
    if writes_since_outline < WARN_WRITES_PER_LEAF:
        return ""
    return (f"\n(That is {writes_since_outline} edits under one outline state. A leaf is one edit or one command: "
            "mark what is done and split what remains with todo_write; edits will be refused at "
            f"{MAX_WRITES_PER_LEAF}.)")


def outline_signature(items: List[dict]) -> tuple:
    return tuple((i["id"], i.get("parent_id"), i["content"], i["status"]) for i in items)


# --------------------------------------------------------------------------------------
# What the agent is told, right above the request, every turn
# --------------------------------------------------------------------------------------

def planning_brief(stack: Iterable[str], request_text: str = "") -> str:
    """A short, specific brief placed beside the request (where it is read), not only in the
    system prompt (where it was ignored)."""
    stack = set(stack)
    lines = [
        "Planning brief for this request:",
        f"- Stack in this project: {describe_stack(stack)}. Stay on it, whatever it is. Do not bring in another "
        "framework, library or algorithm because it is easier or already has the feature; change stack only if "
        "the user explicitly tells you to. The edit tool refuses a new framework the user did not ask for.",
        "- Call todo_write before your first edit. Its first phase is \"Requirements\": one leaf per thing the "
        "request names (every library, technique, feature, file), in the user's words. Then one phase per "
        "component, broken down until each leaf is a single edit or command naming a file and function.",
    ]
    if autograd_in_play(stack, request_text):
        lines.append("- Outline the backward pass too: a leaf for loss.backward() / the optimizer step, and a leaf "
                     "that checks every parameter receives a non-zero gradient.")
    else:
        lines.append("- There is no autograd here: for EVERY forward-pass task in the outline add its own backward-pass "
                     "task right after it (cache the activations the backward needs; compute dX and each parameter "
                     "gradient), then a finite-difference gradient-check leaf and an overfit-one-tiny-batch leaf. "
                     "An outline with forward tasks and no matching backward tasks is rejected.")
    lines += [
        "- Write complete production code: no stubs, placeholders, dummy or random stand-in data, or \"simplified\" "
        "versions. A training script is a full loop (data, batching, forward, loss, backward, clipping, optimizer, "
        "schedule, eval, logging, checkpoints).",
        "- You are done only when every requirement is implemented AND checked with smoke_test (a probe that runs "
        "forward and backward). Your summary lists each requirement and how you verified it.",
    ]
    return "\n".join(lines)