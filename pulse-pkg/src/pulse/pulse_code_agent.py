"""
Pulse Code's native agent loop.

The model is given real tools (function calling) and simply keeps going: it reads, edits, runs
commands, sees the results, and carries on until it answers without asking for another tool.
There is no plan -> JSON -> verify pipeline to fall out of; a failed edit or a failing test is
just a tool result the model reacts to.

What the other terminal agents do, this does too: read_file / edit_file / write_file / grep /
run_command / a todo list, automatic context compaction, a per-turn /undo.

What Pulse adds on top, because it can parse the code instead of treating it as text:

  * outline            -- the symbols of a file with line spans, without reading it
  * find_definition / find_references / dep_graph / trace_variable -- AST answers, not grep
  * replace_symbol     -- replace a whole function/class by name (decorators and indentation
                          handled), so large rewrites need no fragile exact-text snippet
  * impact report      -- after any edit that changes a function's signature, the call sites that
                          may now be wrong are listed in the same tool result, unprompted
  * lint gate          -- every Python write is syntax-checked (and pyflakes-checked) first

`run_native_turn` is the entry point; `supported` says whether the configured model can do
function calling at all (pulse_code falls back to its older text pipeline when it cannot).
"""
import ast
import difflib
import fnmatch
import json
import importlib.util
import subprocess
import tempfile
import os
import re
import shlex
import shutil
import sys
import textwrap
import time

from . import pulse_ui as _ui
from . import pulse_ratelimit as _ratelimit
from . import pulse_shapes as _shapes
from .pulse_cli import (
    AgentRequestFailed,
    PROVIDERS,
    _AGENT_MAX_TOKENS,
    _AGENT_TIMEOUT_SECONDS,
    _BLUE,
    _GREEN,
    _RED,
    _Spinner,
    _YELLOW,
    _clamp_output_tokens,
    _complete,
    _find_fuzzy_snippet_span,
    _prompt_text,
    _reindent_like,
    _token_boundary_occurrences,
    cprint,
    litellm,
)

# ---------------------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------------------

_MAX_ITERATIONS = 400            # model calls in one request; a safety net, not a plan
_READ_DEFAULT_LINES = 600        # read_file without a range
_READ_MAX_LINES = 1500
_RESULT_MAX_CHARS = 24_000       # one tool result sent back to the model
_LIST_MAX = 300
_REPEAT_LIMIT = 3                # identical consecutive tool calls before the model is told it is looping
_MAX_NUDGES = 2                  # retries for non-outline completion nudges

_COMPACT_AT_CHARS = 360_000      # conversation size (chars, ~90k tokens) that triggers compaction
_KEEP_RECENT = 14                # messages kept verbatim by compaction
_ELIDE_AFTER = 16                # tool results older than this many messages are shortened
_ELIDED_CHARS = 700
_SUMMARY_INPUT_CHARS = 140_000

_PROTECTED_PARTS = (".git", ".pulse_history")

# ---------------------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Pulse Code, an autonomous coding agent working in the user's terminal, inside their real project. You complete software engineering tasks end to end: you investigate, change code, run it, and keep going until the task is actually done.

# How you work
- Work autonomously. Do not stop to ask permission or to announce a plan and wait. If the request is clear enough to act on, act. Ask a question only when you are genuinely blocked on something only the user can decide, and say exactly what.
- Keep going until the whole request is implemented AND checked. One edit is rarely the end: wire the pieces together, update callers, then run the project's tests / linter / the code itself and fix what the output shows. Never claim something works unless you ran it and saw it work; report real exit codes and output.
- When the user asks to run, launch, train, or try the completed script/change, start it with start_run when that tool is available. This is Pulse's monitored `/run`: provide the project script and its arguments, then check run_status when useful. Do not substitute run_command for a training run; use run_command only for short checks and tests.
- For anything with more than ~2 steps, todo_write is required before coding. Build a recursive outline: phases can contain subtasks, and any subtask can itself contain smaller subtasks at any depth. Make the leaves concrete actions, include verification, and keep exactly one unfinished leaf in_progress. After completing a leaf, call todo_write to mark it completed and activate the next leaf before moving on. If active work turns out to need decomposition, add children under it and continue at the new leaf level. Do not give a final answer while any leaf remains unfinished; the outline gate will keep the turn open.
- Read before you write. Never guess at signatures, imports, names or file layout.
- Make the change the request needs, matching the surrounding style. Don't reformat, rename or "improve" unrelated code. If you notice a separate bug, mention it in your final answer instead of fixing it.
- Prefer small, targeted edits (edit_file) or replace_symbol for rewriting a whole function/class. Use write_file for new files. Make independent tool calls in the same turn when you can.
- If a tool fails, read the error, adjust, and try again; a failed edit or command is information, not the end of the task. Don't repeat the identical call hoping for a different result.
- Talk to the user while you work with message_user: what you found, what you are about to do, a heads-up about something you noticed. It does not end your turn and it gets no reply, so put it in the same reply as the tool calls you are making. Plain text next to tool calls is not shown to the user by every provider; message_user always is.
- If the request is a question, answer it from the code; make no edits.
- Do what the request says, not the nearest thing you know how to do: "run it" means start_run, "why is X" means an answer with the real numbers, "add Y" means edits. A reply that describes what you are going to do, without doing it, is not an answer: do it, in that same reply, with the tools.

# Use the code structure, not just text search
This is Pulse's advantage. For Python code prefer the AST tools over grep:
- outline: the classes/functions of a file with line ranges. Use it before reading a big file, then read_file just the range you need.
- find_definition / find_references: exact jump-to-definition and call sites.
- dep_graph: which project files import which.
- trace_variable: everything that feeds a variable and everything it feeds, across functions and files. Use it to find where a bad value came from.
- replace_symbol: replace a function/class by name.
- After an edit that changes a function's signature, the tool result lists the call sites that may now be wrong. Fix them.

# Running things
run_command runs a real command in the project directory and returns the real stdout/stderr/exit code. Use it for tests, linters, builds, git, and small repro scripts. Commands that delete files, rewrite git history, reach outside the project, start background processes or overwrite files via redirects pause for the user's confirmation; ordinary read/run/test commands do not. A command is killed at its timeout (two minutes unless you pass one), so run_command is NOT how a training run is started.

{runs}# Checking your work (do this like a careful engineer, not as an afterthought)
- smoke_test is the check to run after you change code. It runs, in order: a compile/undefined-name check of every file you changed, a tiny probe (if you give one), and the project's own test suite. It stops at the first stage that fails and tells you which. Run it before you finish, and again after fixing what it found.
- A probe is a few lines of Python that build the thing you changed on tiny inputs and run it once: construct the model, make a small synthetic batch (batch size 2-4), run forward, the loss and backward. Torch layers are traced, so a shape mismatch is reported as the exact layer and the shape it received, not a traceback into library code. Pass `expect` to pin down the shapes you rely on, e.g. {"logits": "(B, 10)", "y": "(B,)"}: B must be the same size everywhere. Write a probe whenever your change touches a model, a data pipeline, a loss, or anything with a shape. Never run the full training script as a probe.
- check_shape answers "what is the shape of X?" with the real value instead of your guess. Use it BEFORE you write code that depends on a shape (a Linear's in_features after a conv stack, what a DataLoader yields, what a function returns), and when a shape error leaves you unsure which side is wrong. It runs a snippet in the project and reports shape, dtype and device of the expressions you name.
- Probes run on CPU-sized inputs. Do not load a full dataset or train in a probe. If building the model needs data you cannot fake, construct tensors of the shape the code expects.

# Finishing
When the work is done and verified, reply with a short summary: what you changed, how you checked it, and anything the user should know. No tool call in a reply means you are finished, so only reply without one when you are.
"""

# Run control exists only inside the Pulse app, which owns the runs on screen. Outside it the
# tools are not offered at all (offered, every call came back "only available inside the app").
_RUNS_SECTION = """# Training runs
You can run the user's project yourself with start_run -- it is the same monitored launch as `/run <script> [args]`. When the user asks to run, start, launch, train, or try the completed change, choose the actual project script and its required arguments and call start_run; do not merely tell the user to run it. The script runs in the background under Pulse, which watches its steps, metrics and findings. Use run_status to check the run after launch or when reporting progress. Use run_command only for short bounded checks and tests, never to launch a training run.
restart_run stops the watched script and starts it again with current code after a fix; stop_run asks the user before stopping it.

"""
_NO_RUNS_SECTION = """# Training runs
A training run is not started from here: run_command kills anything still going at its timeout. When the user asks to run or train, tell them the command (`pulse run --stream <script>`, or `pulse` to open the Pulse app, where you can start and watch runs), and use run_command only for short, bounded checks.

"""
_RUN_TOOLS = {"start_run", "run_status", "restart_run", "stop_run"}


def _with_runs():
    host = _ui.host()
    return host is not None and hasattr(host, "run_actions")


def _system_prompt(cli):
    base = SYSTEM_PROMPT.replace("{runs}", _RUNS_SECTION if _with_runs() else _NO_RUNS_SECTION)
    return base + (getattr(cli, "_native_prompt_suffix", "") or "")


def _tools():
    return TOOLS if _with_runs() else [t for t in TOOLS if t["function"]["name"] not in _RUN_TOOLS]

# ---------------------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------------------


def _fn(name, description, properties, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required)}}}


_S = {"type": "string"}
_I = {"type": "integer"}

TOOLS = [
    _fn("read_file",
        "Read a file with line numbers. Without a range it reads the first 600 lines; pass start_line/end_line "
        "for a region (use outline first to find it).",
        {"path": _S, "start_line": _I, "end_line": _I}, ["path"]),
    _fn("list_files",
        "List project files (paths with line counts), optionally filtered by a glob such as 'src/**/*.py'.",
        {"pattern": _S}),
    _fn("grep",
        "Case-insensitive regex search across project files; returns matching lines with context. "
        "For Python symbols prefer find_definition / find_references.",
        {"pattern": _S}, ["pattern"]),
    _fn("outline",
        "The symbols in a file (classes, functions, methods) with line ranges and signatures -- the map of a file "
        "without reading it.",
        {"path": _S}, ["path"]),
    _fn("find_definition",
        "AST jump-to-definition: where a function/class/variable is defined, across all project files.",
        {"symbol": _S}, ["symbol"]),
    _fn("find_references",
        "AST: every call site of a function/class across all project files.",
        {"symbol": _S}, ["symbol"]),
    _fn("dep_graph", "The import graph between project files.", {}),
    _fn("trace_variable",
        "AST data-flow: everything that feeds a variable and everything it feeds, across function and file "
        "boundaries. Target format: name[:file[:line]]; self.<attr> works.",
        {"target": _S}, ["target"]),
    _fn("doc_lookup",
        "Real signature/docstring of an installed library function, e.g. 'torch.nn.functional.cross_entropy'.",
        {"name": _S}, ["name"]),
    _fn("edit_file",
        "Replace exact text in an existing file. old_string must match the file exactly (without line-number "
        "prefixes) and be unique -- include neighbouring lines to make it so -- unless replace_all is true. "
        "The file must have been read first.",
        {"path": _S, "old_string": _S, "new_string": _S,
         "replace_all": {"type": "boolean"}}, ["path", "old_string", "new_string"]),
    _fn("replace_symbol",
        "Replace an entire Python function, method or class by name (e.g. 'Trainer.step' or 'load_config') with "
        "new_source: the full new definition, including decorators. Indentation is handled for you. Prefer this "
        "to edit_file when rewriting most of a definition.",
        {"path": _S, "symbol": _S, "new_source": _S}, ["path", "symbol", "new_source"]),
    _fn("write_file",
        "Create a new file (or fully overwrite one you have already read) with the given content.",
        {"path": _S, "content": _S}, ["path", "content"]),
    _fn("run_command",
        "Run a shell command in the project directory and get back real stdout, stderr, exit code and duration. "
        "Optional timeout in seconds.",
        {"command": _S, "timeout": _I}, ["command"]),
    _fn("check_shape",
        "Run a short Python snippet in the project and report the real shape, dtype and device of values -- so you "
        "know a shape instead of guessing it. `code` runs first (imports, build a model, make a tiny batch); "
        "`exprs` are the expressions to describe afterwards (e.g. [\"model(x)\", \"batch[0]\"]; default: every array, "
        "module, list and dict the code left behind); `expect` maps an expression to a spec such as \"(B, 10)\" -- "
        "numbers are exact, a name like B must be the same size everywhere, * is any size, ... is any number of dims. "
        "Torch layers are traced, so a shape error names the layer that received the wrong shape. Variables are "
        "described even if the code failed part-way.",
        {"code": _S, "exprs": {"type": "array", "items": _S}, "expect": {"type": "object", "additionalProperties": _S},
         "timeout": _I}),
    _fn("smoke_test",
        "Check the changes you made, cheaply, in three stages that stop at the first failure: (1) compile + "
        "undefined-name check of every file you changed; (2) `probe`, a few lines of Python that build the changed "
        "thing on a tiny input and run it once (model forward/loss/backward on a batch of 2-4) -- torch layers are "
        "traced so a shape mismatch names the layer and the shape it received; `expect` pins shapes, e.g. "
        "{\"logits\": \"(B, 10)\"}; (3) the project's own test suite (pytest or unittest, found automatically; "
        "pass suite=false to skip). Run this after editing, before you finish.",
        {"probe": _S, "expect": {"type": "object", "additionalProperties": _S}, "suite": {"type": "boolean"},
         "timeout": _I}),
    _fn("start_run",
        "Start the user's project script under Pulse, the same monitored launch as `/run <script> [args]`. "
        "Call this when the user asks you to run, launch, train, or try the completed change: choose the actual "
        "script path from the project and include its needed command-line arguments. Pulse runs it in the "
        "background, watches its steps/metrics/findings, and shows it to the user. Then use run_status to "
        "inspect progress when useful. Do not use run_command for a training run; reserve that for short checks.",
        {"script": _S, "args": _S}, ["script"]),
    _fn("run_status",
        "The watched run right now: step, every tracked value's curve, the detectors' findings, recent events.",
        {}),
    _fn("restart_run",
        "Run the watched script again with the code as it is now -- after you changed it. A run still going is "
        "stopped first; one that crashed or ended is simply started again. Use it when the fix only takes effect "
        "in a fresh run or you need to see that it works; leave the run alone when it is healthy and the change "
        "can wait, and say so.", {}),
    _fn("stop_run", "Stop the watched training run. The user is asked to confirm.", {}),
    _fn("message_user",
        "Show the user a message right now, while you keep working: progress, a finding, a heads-up. It is "
        "not a question -- it gets no reply and does not end your turn. Use it in the same reply as your "
        "other tool calls.",
        {"message": _S}, ["message"]),
    _fn("todo_write",
        "Required for requests with more than about two steps: create and maintain the task outline before coding. "
        "Replace the full outline each time. Each item is {id, parent_id, content, status}; parent_id is null "
        "for a top-level phase or the id of any parent task. The hierarchy is recursive: phases contain subtasks, "
        "and subtasks may contain their own subtasks at any depth. Break broad work into concrete leaf actions, "
        "including verification. Statuses are pending | in_progress | completed. Parent statuses are derived from "
        "their children. Keep exactly one unfinished leaf in_progress. After completing a leaf, update the outline "
        "before moving on; do not finish while any leaf is unfinished.",
        {"todos": {"type": "array", "items": {"type": "object", "properties": {
            "id": _S, "parent_id": {"type": ["string", "null"]}, "content": _S,
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
            "required": ["id", "content", "status"]}}}, ["todos"]),
]

# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------


def supported(cli):
    """True unless litellm positively says this model cannot do function calling."""
    model = cli.agent_model_string or PROVIDERS[cli.agent_provider]["model"]
    try:
        return bool(litellm.supports_function_calling(model=model))
    except Exception:
        return True            # unknown to litellm: try it; a provider error falls back gracefully


def _clip(text, limit=_RESULT_MAX_CHARS):
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.7), int(limit * 0.25)
    return f"{text[:head]}\n...[{len(text) - head - tail:,} characters omitted]...\n{text[-tail:]}"


def _numbered(lines, start):
    return "\n".join(f"{n:>5} | {line}" for n, line in enumerate(lines, start))


def _rel(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


class _State:
    """Everything one request accumulates."""

    def __init__(self, cli, request):
        self.cli = cli
        self.request = request
        self.crlf = {}               # path -> the file on disk uses CRLF line endings
        self.seen = {}               # path -> text as last read or written by the agent
        self.changes = {}            # path -> [text before the turn, latest text, created?]
        self.todos = []
        self.dirty = False           # files changed since the last command that succeeded
        self.smoke_ok = None         # result of the last smoke_test (None: never run)
        self.said = ""               # what the model said beside its tool calls, until shown
        self.nudges = 0
        self.last_call = None
        self.repeats = 0
        self.edits_declined = 0
        self.calls_made = 0          # tool calls executed in this request

    @property
    def root(self):
        return self.cli._project_root

    def resolve(self, path):
        """(absolute path, None) or (None, reason)."""
        if not isinstance(path, str) or not path.strip():
            return None, "path is empty"
        path = os.path.expanduser(path.strip())
        absolute = os.path.abspath(path if os.path.isabs(path) else os.path.join(self.root, path))
        from . import pulse_code as _pc
        if not _pc._inside(self.root, absolute):
            return None, f"'{path}' is outside the project root {self.root}"
        parts = os.path.relpath(absolute, self.root).split(os.sep)
        if any(p in _PROTECTED_PARTS for p in parts):
            return None, f"'{path}' is in a protected directory"
        return absolute, None

    def read(self, absolute):
        """The file as the model sees it: LF line endings. A CRLF file is remembered as such and
        written back as CRLF (see _commit); /undo's own comparisons are LF too."""
        from . import pulse_code as _pc
        text = _pc.read_text(absolute)
        if text is not None:
            self.crlf[absolute] = "\r\n" in text
            text = text.replace("\r\n", "\n")
        return text


# ---------------------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------------------


def _signature(node):
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    ret = ""
    if node.returns is not None:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return f"{prefix} {node.name}({args}){ret}"


def _walk_defs(tree):
    """(qualified name, node, depth) for every class/function, in source order."""
    out = []

    def visit(body, prefix, depth):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{node.name}"
                out.append((name, node, depth))
                visit(node.body, name + ".", depth + 1)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for field in ("body", "orelse", "finalbody", "handlers"):
                    inner = getattr(node, field, None)
                    if not inner:
                        continue
                    for item in inner:
                        visit(item.body if isinstance(item, ast.ExceptHandler) else [item], prefix, depth)

    visit(tree.body, "", 0)
    return out


def _def_span(node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, node.end_lineno


def _outline(path, text):
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return f"(does not parse: {exc.msg} at line {exc.lineno})\n" + _regex_outline(text)
        rows = []
        for name, node, depth in _walk_defs(tree):
            start, end = _def_span(node)
            sig = f"class {node.name}" + (f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else "") \
                if isinstance(node, ast.ClassDef) else _signature(node)
            rows.append(f"{'  ' * depth}L{start}-{end}  {sig}")
        doc = ast.get_docstring(tree)
        head = f"{len(text.splitlines())} lines" + (f" -- {doc.splitlines()[0][:100]}" if doc else "")
        return head + "\n" + ("\n".join(rows) if rows else "(no classes or functions)")
    return f"{len(text.splitlines())} lines\n" + _regex_outline(text)


def _regex_outline(text):
    rx = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|fn|func|interface|struct|impl|type)\s+\w+.*$")
    rows = [f"L{i}  {line.strip()[:120]}" for i, line in enumerate(text.splitlines(), 1) if rx.match(line)]
    return "\n".join(rows[:200]) if rows else "(no recognisable definitions -- use read_file)"


def _signatures(text):
    """{qualified name: signature text} for every function in Python source ({} if it will not parse)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    return {name: _signature(node) for name, node, _d in _walk_defs(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _impact(cli, before, after):
    """Signature changes between two versions of a file and the call sites that may now be wrong."""
    old, new = _signatures(before), _signatures(after)
    changed = [(n, old[n], new.get(n)) for n in old if new.get(n) != old[n]]
    if not changed:
        return ""
    lines = []
    for name, was, now in changed[:5]:
        simple = name.rsplit(".", 1)[-1]
        if now is None:
            # renamed or removed: only worth flagging if nothing with that simple name survives here
            if any(n.rsplit(".", 1)[-1] == simple for n in new):
                continue
            lines.append(f"- `{name}` no longer exists in this file.")
        else:
            lines.append(f"- signature changed: `{was}` -> `{now}`")
        callers = cli._run_callers(simple)
        if "no call sites" not in callers:
            lines.append(textwrap.indent(_clip(callers, 3000), "  "))
    if not lines:
        return ""
    return "\n\nAST impact -- check that these callers still match the new signature:\n" + "\n".join(lines)


def _find_symbol(tree, symbol):
    """(node, None) or (None, reason) for 'name', 'Class.method', 'Class.method.inner'."""
    defs = _walk_defs(tree)
    exact = [(n, node) for n, node, _d in defs if n == symbol]
    if len(exact) == 1:
        return exact[0][1], None
    if len(exact) > 1:
        return None, f"'{symbol}' is defined {len(exact)} times (lines " + \
            ", ".join(str(node.lineno) for _n, node in exact) + "); use edit_file for this one"
    tail = [(n, node) for n, node, _d in defs if n.endswith("." + symbol) or n.rsplit(".", 1)[-1] == symbol]
    if len(tail) == 1:
        return tail[0][1], None
    if len(tail) > 1:
        return None, f"'{symbol}' is ambiguous: " + ", ".join(f"{n} (line {node.lineno})" for n, node in tail[:8])
    names = ", ".join(n for n, _node, _d in defs[:40])
    return None, f"no function/class named '{symbol}'. Defined here: {names or '(nothing)'}"


# ---------------------------------------------------------------------------------------
# Writing files (lint gate, diff, confirmation, change recording)
# ---------------------------------------------------------------------------------------


def _closest_region(content, old):
    """Where `old` most nearly is in `content` -- shown to the model when its snippet matched nowhere."""
    first = next((l.strip() for l in old.splitlines() if l.strip()), "")
    lines = content.splitlines()
    if not first or not lines:
        return ""
    stripped = [l.strip() for l in lines]
    best = difflib.get_close_matches(first, stripped, n=1, cutoff=0.5)
    if not best:
        return ""
    at = stripped.index(best[0])
    lo, hi = max(0, at - 2), min(len(lines), at + max(4, len(old.splitlines()) + 1))
    return "Closest text in the file:\n" + _numbered(lines[lo:hi], lo + 1)


def _confirm(state, path, before, after, created):
    """Show the diff and, in review mode, ask. Returns True to go ahead."""
    from . import pulse_code as _pc
    cli = state.cli
    diff = _pc.render_diff(cli, {path: (before, after, created)})
    _ui.show_change([f"{'CREATE' if created else 'EDIT'}: {_rel(state.root, path)}"], diff)
    if not cli.review:
        return True
    reviewed = cli._review_change_with_approver(state.request, f"{'create' if created else 'edit'} {_rel(state.root, path)}",
                                                diff)
    if reviewed is not None:
        return reviewed
    try:
        options = [
            _ui.Option("Apply this change", key="y"),
            _ui.Option("Always apply code changes this session", key="a"),
            _ui.Option("Reject this change", key="n"),
        ]
        choice = _ui.choose(options, title="Apply this change?", hotkeys={"a": "always"})
        if choice == "always" or choice == 1:
            cli.review = False
            cprint("[Pulse Code] Review is now OFF for this session (/review on to re-enable).", color=_YELLOW)
            return True
        if isinstance(choice, int):
            return choice == 0
        return False
    except _ui.Unavailable:
        pass
    try:
        answer = _prompt_text("Apply this change? (Y/n/a=always this session) > ",
                              label="Apply this change?  (Y/n/a=always this session)").strip().lower()
    except (EOFError, KeyboardInterrupt):
        cprint("[Pulse Code] No approval received -- change not applied.", color=_YELLOW)
        return False
    if answer in ("a", "all", "always"):
        cli.review = False
        cprint("[Pulse Code] Review is now OFF for this session (/review on to re-enable).", color=_YELLOW)
        return True
    return answer in ("", "y", "yes")


def _commit(state, path, new_text, label, before=None, created=False):
    """Write `new_text` to `path` and record it for this turn's /undo commit."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    on_disk = new_text.replace("\n", "\r\n") if state.crlf.get(path) else new_text
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(on_disk)
    prior = state.changes.get(path)
    if prior is None:
        state.changes[path] = [before if before is not None else "", new_text, created]
    else:
        prior[1] = new_text
    state.seen[path] = new_text
    state.dirty = True
    cli = state.cli
    if created:
        cli.add_files([path], focus=False)
    else:
        cli.reload()
    return f"{label}"


def _write_checked(state, path, before, after, label, created=False):
    """Lint gate -> confirm -> write -> impact. Returns the tool result text."""
    cli = state.cli
    ok, messages = cli._lint_check(after, path, original=None if created else before)
    if not ok:
        return ("NOT APPLIED -- the change would break the file:\n  " + "\n  ".join(messages[:12]) +
                "\nFix the problem in your edit and try again.")
    if after == before:
        return "NOT APPLIED -- the new text is identical to the old text; nothing to change."
    if not _confirm(state, path, before, after, created):
        state.edits_declined += 1
        denial = getattr(cli, "_last_change_denial", "") or "the user declined this change"
        return (f"NOT APPLIED -- {denial}. Do not retry the same edit. Either take a different approach, "
                "or stop and say what you were trying to do and ask how to proceed.")
    _commit(state, path, after, label, before=before, created=created)
    n_add = sum(1 for l in difflib.ndiff(before.splitlines(), after.splitlines()) if l.startswith("+ "))
    n_del = sum(1 for l in difflib.ndiff(before.splitlines(), after.splitlines()) if l.startswith("- "))
    note = f"{label}  (+{n_add} -{n_del} lines)"
    if path.endswith(".py") and not created:
        note += _impact(cli, before, after)
    return note


# ---------------------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------------------


def _t_read_file(state, a):
    path, err = state.resolve(a.get("path"))
    if err:
        return err
    if os.path.isdir(path):
        return f"'{a.get('path')}' is a directory; use list_files."
    text = state.read(path)
    if text is None:
        return f"'{a.get('path')}' does not exist, is binary, or is too large to read."
    state.seen[path] = text
    lines = text.splitlines()
    start = max(1, int(a.get("start_line") or 1))
    end = int(a.get("end_line") or (start + _READ_DEFAULT_LINES - 1))
    end = min(end, len(lines), start + _READ_MAX_LINES - 1)
    if start > len(lines):
        return f"{_rel(state.root, path)} only has {len(lines)} lines."
    body = _numbered(lines[start - 1:end], start)
    more = f"\n... {len(lines) - end} more lines; call read_file with start_line={end + 1}" if end < len(lines) else ""
    return f"{_rel(state.root, path)} (lines {start}-{end} of {len(lines)})\n{body}{more}"


def _t_list_files(state, a):
    from . import pulse_code as _pc
    pattern = (a.get("pattern") or "").strip()
    files = _pc.scan_project(state.root, limit=5000)
    rows = []
    for f in files:
        rel = _rel(state.root, f)
        if pattern and not (fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(os.path.basename(rel), pattern)):
            continue
        try:
            with open(f, "rb") as h:
                n = sum(1 for _ in h)
        except OSError:
            n = 0
        rows.append(f"{rel}  ({n} lines)")
    if not rows:
        return "No matching files."
    extra = f"\n... and {len(rows) - _LIST_MAX} more; narrow with a pattern" if len(rows) > _LIST_MAX else ""
    return "\n".join(rows[:_LIST_MAX]) + extra


def _t_grep(state, a):
    return state.cli._run_grep(str(a.get("pattern", "")))


def _t_outline(state, a):
    path, err = state.resolve(a.get("path"))
    if err:
        return err
    text = state.read(path)
    if text is None:
        return f"'{a.get('path')}' does not exist, is binary, or is too large."
    return f"{_rel(state.root, path)}: " + _outline(path, text)


def _t_find_definition(state, a):
    return state.cli._run_defof(str(a.get("symbol", "")))


def _t_find_references(state, a):
    return state.cli._run_callers(str(a.get("symbol", "")))


def _t_dep_graph(state, a):
    return state.cli._run_depgraph()


def _t_trace_variable(state, a):
    return state.cli._run_trace(str(a.get("target", "")))


def _t_doc_lookup(state, a):
    return state.cli._run_doclookup(str(a.get("name", "")))


def _t_edit_file(state, a):
    path, err = state.resolve(a.get("path"))
    if err:
        return err
    old, new = a.get("old_string"), a.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return "old_string and new_string are required strings."
    if not old:
        return "old_string is empty. To add text, use a nearby existing line as old_string and repeat it in new_string."
    content = state.read(path)
    if content is None:
        return f"'{a.get('path')}' does not exist (use write_file to create it), is binary, or is too large."
    if path not in state.seen:
        return f"Read '{a.get('path')}' with read_file before editing it."
    if state.seen[path] != content:
        state.seen[path] = content
        return (f"'{a.get('path')}' has changed on disk since you last read it (the user may have edited it). "
                "Read it again, then redo the edit.")
    label = f"edited {_rel(state.root, path)}"
    hits = _token_boundary_occurrences(content, old)
    if len(hits) == 1:
        after = content[:hits[0]] + new + content[hits[0] + len(old):]
    elif len(hits) > 1:
        if not a.get("replace_all"):
            where = ", ".join(str(content.count("\n", 0, h) + 1) for h in hits[:8])
            return (f"old_string matches {len(hits)} places (lines {where}). Add surrounding lines to make it "
                    "unique, or pass replace_all=true to change every one.")
        after = content
        for h in reversed(hits):
            after = after[:h] + new + after[h + len(old):]
        label = f"edited {_rel(state.root, path)} ({len(hits)} replacements)"
    else:
        span = _find_fuzzy_snippet_span(content, old)
        if span is None:
            return ("old_string was not found in the file. It must match exactly, including indentation.\n"
                    + _closest_region(content, old) + "\nRead the region again and retry.")
        after = content[:span[0]] + _reindent_like(new, old, content[span[0]:span[1]]) + content[span[1]:]
    return _write_checked(state, path, content, after, label)


def _t_replace_symbol(state, a):
    path, err = state.resolve(a.get("path"))
    if err:
        return err
    if not path.endswith(".py"):
        return "replace_symbol only works on Python files; use edit_file."
    new_source = a.get("new_source")
    if not isinstance(new_source, str) or not new_source.strip():
        return "new_source is required."
    content = state.read(path)
    if content is None:
        return f"'{a.get('path')}' does not exist, is binary, or is too large."
    if path not in state.seen:
        return f"Read '{a.get('path')}' with read_file before editing it."
    if state.seen[path] != content:
        state.seen[path] = content
        return f"'{a.get('path')}' has changed on disk since you last read it. Read it again, then redo the edit."
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return f"The file does not currently parse ({exc.msg}, line {exc.lineno}); fix it with edit_file first."
    node, why = _find_symbol(tree, str(a.get("symbol", "")))
    if node is None:
        return why
    start, end = _def_span(node)
    lines = content.splitlines(keepends=True)
    indent = " " * node.col_offset
    body = textwrap.dedent(new_source).strip("\n")
    body = textwrap.indent(body, indent) + "\n"
    after = "".join(lines[:start - 1]) + body + "".join(lines[end:])
    return _write_checked(state, path, content, after, f"replaced {a.get('symbol')} in {_rel(state.root, path)}")


def _t_write_file(state, a):
    path, err = state.resolve(a.get("path"))
    if err:
        return err
    content = a.get("content")
    if not isinstance(content, str):
        return "content is required."
    if content and not content.endswith("\n"):
        content += "\n"
    if os.path.isdir(path):
        return f"'{a.get('path')}' is a directory."
    if os.path.exists(path):
        before = state.read(path)
        if before is None:
            return f"'{a.get('path')}' exists but cannot be read as text; not overwriting it."
        if path not in state.seen:
            return f"'{a.get('path')}' already exists. Read it first if you mean to overwrite it (or use edit_file)."
        if state.seen[path] != before:
            state.seen[path] = before
            return f"'{a.get('path')}' has changed on disk since you last read it. Read it again first."
        return _write_checked(state, path, before, content, f"overwrote {_rel(state.root, path)}")
    return _write_checked(state, path, "", content, f"created {_rel(state.root, path)}", created=True)


def _t_run_command(state, a):
    command = str(a.get("command", "")).strip()
    if not command:
        return "command is empty."
    try:
        timeout = int(a.get("timeout") or 0)
    except (TypeError, ValueError):
        timeout = 0
    arg = f"{command} --timeout={timeout}" if timeout > 0 else command   # the executor's inline-timeout syntax
    result = state.cli._run_terminal(arg)
    if re.search(r"(?m)^Exit code: 0$", result):
        state.dirty = False                  # something ran cleanly since the last edit
    if re.search(r"timed out|timeout", result, re.I) and re.search(r"\bpython[0-9.]*\b.*\.py\b", command):
        result += ("\n\nThis looks like a training script, and it was killed at the timeout. A run is "
                   "started with start_run, which keeps it running and lets Pulse watch it; use "
                   "run_command only for short checks.")
    return result


# ---------------------------------------------------------------------------------------
# Probes and smoke tests
# ---------------------------------------------------------------------------------------

_PROBE_TIMEOUT = 90                  # seconds a probe snippet may run
_SUITE_TIMEOUT = 300                 # seconds the project's test suite may run
_SKIP_DIRS = {".git", ".pulse_history", "node_modules", "__pycache__", ".venv", "venv", "env", "site-packages",
              ".tox", "build", "dist"}


def _quote(arg):
    return subprocess.list2cmdline([arg]) if os.name == "nt" else shlex.quote(arg)


def _run_probe(state, code, exprs, expect, timeout, purpose):
    """Run `code` through pulse_shapes in the project directory. Returns (report dict, None) or
    (None, why it produced none). The snippet is judged like a `python -c` command: one that deletes
    files or reaches outside the project pauses for the user exactly as run_command does."""
    cli = state.cli
    pseudo = f"python -c {_quote(code)}" if code.strip() else ""
    flags = cli._terminal_needs_confirmation(pseudo) if pseudo else None
    if flags and not cli._confirm_terminal_command(pseudo, flags, purpose):
        return None, f"NOT RUN -- {cli._terminal_denial_reason()}. Try a probe that does not do that."
    from . import pulse_terminal as _terminal
    workdir = tempfile.mkdtemp(prefix="pulse_probe_")
    try:
        request_path = os.path.join(workdir, "request.json")
        report_path = os.path.join(workdir, "report.json")
        with open(request_path, "w", encoding="utf-8") as handle:
            json.dump({"code": code, "exprs": exprs, "expect": expect, "report_path": report_path}, handle)
        command = f"{_quote(sys.executable)} {_quote(_shapes.__file__)} {_quote(request_path)}"
        executor = cli._get_terminal_executor()
        result = executor.run(_terminal.TerminalRequest(command=command, timeout=float(timeout)))
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            report["_output"] = ((result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")).strip()
            return report, None
        if result.timed_out:
            return None, (f"the probe was killed after {timeout}s. A probe must build the thing on a tiny input and "
                          "run it once, not train or load a dataset.")
        tail = _clip(((result.stderr or "") + "\n" + (result.stdout or "")).strip(), 2500)
        return None, f"the probe process died before it could report (exit {result.exit_code}):\n{tail}"
    except (OSError, ValueError) as exc:
        return None, f"could not run the probe: {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _report_failed(report):
    return bool(report.get("error")) or any(not c.get("ok") for c in report.get("expect") or [])


def _printed_output(report, limit=1200):
    out = report.get("_output") or ""
    return "\n  the snippet printed:\n    " + _clip(out, limit).replace("\n", "\n    ") if out else ""


def _as_exprs(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def _as_expect(value):
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def _bad_specs(expect):
    for expr, spec in expect.items():
        try:
            _shapes.parse_spec(spec)
        except ValueError as exc:
            return f"{expr}: {exc}"
    return None


def _t_check_shape(state, a):
    code = a.get("code") if isinstance(a.get("code"), str) else ""
    exprs, expect = _as_exprs(a.get("exprs")), _as_expect(a.get("expect"))
    if not code.strip() and not exprs:
        return "Nothing to check: give `code` to run and/or `exprs` to describe."
    problem = _bad_specs(expect)
    if problem:
        return f"Not run -- a spec is not valid: {problem}"
    try:
        timeout = max(5, min(int(a.get("timeout") or _PROBE_TIMEOUT), 600))
    except (TypeError, ValueError):
        timeout = _PROBE_TIMEOUT
    report, why = _run_probe(state, code, exprs, expect, timeout, "check the shape of values in the project")
    if report is None:
        return why
    verdict = "FAIL" if _report_failed(report) else "ok"
    return f"CHECK_SHAPE: {verdict}\n" + _shapes.render_report(report) + _printed_output(report)


def _project_test_command(state):
    """(command, label) for the project's own test suite, or (None, why not)."""
    root = state.root
    has_config = False
    for name in ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "conftest.py"):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        if name in ("pytest.ini", "conftest.py"):
            has_config = True
        else:
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
                has_config = has_config or bool(re.search(r"\[(?:tool\.pytest|tool:pytest|pytest)", text))
            except OSError:
                pass
    test_files = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        if os.path.relpath(current, root).count(os.sep) >= 3:
            dirs[:] = []
        test_files += sum(1 for f in files if f.endswith(".py") and (f.startswith("test_") or f.endswith("_test.py")))
        if test_files >= 1 and has_config:
            break
    if not test_files and not has_config:
        return None, "no tests found in the project"
    py = _quote(sys.executable)
    if importlib.util.find_spec("pytest") is not None:
        return f"{py} -m pytest -q --maxfail=10 -p no:cacheprovider", "pytest"
    if test_files:
        return f"{py} -m unittest discover -q", "unittest"
    return None, "the project has pytest settings but pytest is not installed in this Python"


def _t_smoke_test(state, a):
    cli = state.cli
    probe = a.get("probe") if isinstance(a.get("probe"), str) else ""
    expect = _as_expect(a.get("expect"))
    run_suite = a.get("suite") is not False
    problem = _bad_specs(expect)
    if problem:
        return f"Not run -- a spec is not valid: {problem}"
    try:
        timeout = max(5, min(int(a.get("timeout") or _SUITE_TIMEOUT), 1800))
    except (TypeError, ValueError):
        timeout = _SUITE_TIMEOUT
    lines, failed = [], False

    # 1. compile / undefined names in what this request changed
    py_changes = {p: v for p, v in state.changes.items() if p.endswith(".py") and os.path.exists(p)}
    broken = []
    for path, (before, _after, created) in py_changes.items():
        text = state.read(path)
        if text is None:
            continue
        ok, messages = cli._lint_check(text, path, original=None if created else before)
        if not ok:
            broken.extend(f"{_rel(state.root, path)}: {m}" for m in messages[:6])
    if broken:
        failed = True
        lines.append("[1/3] compile   FAILED")
        lines.extend("      " + m for m in broken[:12])
    else:
        lines.append(f"[1/3] compile   ok ({len(py_changes)} changed Python file{'s' if len(py_changes) != 1 else ''})")

    # 2. the probe
    if failed:
        lines.append("[2/3] probe     not run (fix the compile errors first)")
    elif not probe.strip() and not expect:
        lines.append("[2/3] probe     skipped (none given -- for a change to a model, data, loss or anything with a "
                     "shape, pass `probe`)")
    else:
        report, why = _run_probe(state, probe, [], expect, _PROBE_TIMEOUT, "smoke-test the change with a tiny probe")
        if report is None:
            failed = True
            lines += ["[2/3] probe     FAILED", "      " + why.replace("\n", "\n      ")]
        elif _report_failed(report):
            failed = True
            lines.append("[2/3] probe     FAILED")
            lines.append("      " + (_shapes.render_report(report) + _printed_output(report)).replace("\n", "\n      "))
        else:
            calls = (report.get("trace") or {}).get("module_calls")
            extra = f", {calls} layer calls traced" if calls else ""
            checks = len(report.get("expect") or [])
            lines.append(f"[2/3] probe     ok ({checks} shape expectation{'s' if checks != 1 else ''} held{extra})")

    # 3. the project's tests
    if failed:
        lines.append("[3/3] suite     not run (an earlier stage failed)")
    elif not run_suite:
        lines.append("[3/3] suite     skipped (suite=false)")
    else:
        command, label = _project_test_command(state)
        if command is None:
            lines.append(f"[3/3] suite     skipped ({label})")
        else:
            result = cli._run_terminal(f"{command} --timeout={timeout}", purpose="run the project's test suite")
            exit_code = re.search(r"(?m)^Exit code: (\S+)$", result)
            code = exit_code.group(1) if exit_code else "?"
            if code == "0":
                summary = next((l for l in reversed(result.splitlines()) if re.search(r"\b(passed|ok|OK)\b", l)), "")
                lines.append(f"[3/3] suite     ok ({label}{': ' + summary.strip() if summary else ''})")
            elif code == "5" and label == "pytest":
                lines.append("[3/3] suite     skipped (pytest collected no tests)")
            else:
                failed = True
                lines.append(f"[3/3] suite     FAILED ({label}, exit {code})")
                lines.append("      " + _clip(result, 6000).replace("\n", "\n      "))
    state.smoke_ok = not failed
    if not failed:
        state.dirty = False                      # the changes have now been checked
    head = "SMOKE TEST: FAIL -- fix this and run smoke_test again." if failed else "SMOKE TEST: PASS"
    return head + "\n" + "\n".join(lines)


_STATUS_MARK = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}


def _t_todo_write(state, a):
    todos = a.get("todos")
    if not isinstance(todos, list):
        return "todos must be an outline of {id, parent_id, content, status} items."
    if not todos:
        if any(todo["status"] != "completed" for todo in state.todos):
            return "cannot clear an unfinished task outline; finish it or replace it with the remaining work."
        state.todos = []
        state.cli._todos = []
        return "Task outline cleared."
    if len(todos) > _LIST_MAX:
        return f"task outline is too large ({len(todos)} items); keep it to {_LIST_MAX} focused items."
    clean, by_id = [], {}
    for index, item in enumerate(todos, 1):
        if not isinstance(item, dict):
            return f"todo item {index} must be an object."
        task_id = item.get("id")
        content = item.get("content")
        status = item.get("status")
        parent_id = item.get("parent_id")
        if not isinstance(task_id, str) or not task_id.strip():
            return f"todo item {index} needs a non-empty id."
        task_id = task_id.strip()
        if task_id in by_id:
            return f"todo id {task_id!r} is duplicated."
        if not isinstance(content, str) or not content.strip():
            return f"todo {task_id!r} needs non-empty content."
        if not isinstance(status, str) or status not in _STATUS_MARK:
            return f"todo {task_id!r} status must be pending, in_progress, or completed."
        if parent_id is not None and (not isinstance(parent_id, str) or not parent_id.strip()):
            return f"todo {task_id!r} parent_id must be null or a non-empty todo id."
        task = {
            "id": task_id, "parent_id": parent_id.strip() if isinstance(parent_id, str) else None,
            "content": content.strip(), "status": status,
        }
        clean.append(task)
        by_id[task_id] = task

    children = {task_id: [] for task_id in by_id}
    roots = []
    for task in clean:
        parent_id = task["parent_id"]
        if parent_id is None:
            roots.append(task["id"])
            continue
        if parent_id not in by_id:
            return f"todo {task['id']!r} refers to missing parent {parent_id!r}."
        if parent_id == task["id"]:
            return f"todo {task['id']!r} cannot be its own parent."
        children[parent_id].append(task["id"])

    visiting, visited = set(), set()

    def derive_status(task_id):
        if task_id in visiting:
            raise ValueError(f"todo outline contains a parent cycle at {task_id!r}.")
        if task_id in visited:
            return by_id[task_id]["status"]
        visiting.add(task_id)
        descendants = children[task_id]
        if descendants:
            statuses = [derive_status(child_id) for child_id in descendants]
            status = ("completed" if all(value == "completed" for value in statuses)
                      else "in_progress" if any(value == "in_progress" for value in statuses)
                      else "pending")
            by_id[task_id]["status"] = status
        else:
            status = by_id[task_id]["status"]
        visiting.remove(task_id)
        visited.add(task_id)
        return status

    try:
        for root_id in roots:
            derive_status(root_id)
    except ValueError as exc:
        return str(exc)
    if len(visited) != len(clean):
        return "todo outline contains a cycle or a disconnected parent chain."
    if len(clean) >= 3 and not any(children.values()):
        return "group a multi-step task into top-level phases with concrete child subtasks."
    active_leaves = [task for task in clean
                     if not children[task["id"]] and task["status"] == "in_progress"]
    unfinished_leaves = [task for task in clean
                         if not children[task["id"]] and task["status"] != "completed"]
    if len(active_leaves) > 1:
        return "keep exactly one unfinished leaf subtask in_progress."
    if unfinished_leaves and not active_leaves:
        return "mark exactly one unfinished leaf subtask in_progress before continuing."

    def render(task_id, depth=0):
        task = by_id[task_id]
        lines = [f"{'  ' * depth}{_STATUS_MARK[task['status']]} {task['content']}"]
        for child_id in children[task_id]:
            lines.extend(render(child_id, depth + 1))
        return lines

    clean = [by_id[task["id"]] for task in clean]
    state.todos = clean
    state.cli._todos = clean
    return "Task outline updated:\n" + "\n".join(
        line for root_id in roots for line in render(root_id)
    )


def _run_action(name, **args):
    """Run control lives in the Pulse app (it owns the runs on screen); elsewhere the tool says so."""
    host = _ui.host()
    actions = host.run_actions() if host is not None and hasattr(host, "run_actions") else {}
    action = actions.get(name)
    if action is None:
        return (f"{name} is only available inside the Pulse app (`pulse`), which watches runs. Here, "
                "tell the user to start the run themselves (`pulse run --stream <script>`), or use "
                "run_command for a short, bounded check.")
    return action(**args)


def _ran_the_change(state, result):
    """A run started with the changed code is the check of the change: the "you changed files
    but have not checked" nudge no longer applies (run_status shows how it does)."""
    if str(result).startswith(("Started", "Restarted")):
        state.dirty = False
    return result


def _t_start_run(state, a):
    return _ran_the_change(state, _run_action("start_run", script=str(a.get("script") or ""),
                                               args=str(a.get("args") or "")))


def _t_run_status(state, a):
    return _run_action("run_status")


def _t_restart_run(state, a):
    return _ran_the_change(state, _run_action("restart_run"))


def _t_stop_run(state, a):
    return _run_action("stop_run")


def _t_message_user(state, args):
    text = str(args.get("message") or "").strip()
    if not text:
        return "Nothing shown: the message was empty."
    _ui.message(text)
    return "Shown to the user."


_HANDLERS = {
    "message_user": _t_message_user,
    "start_run": _t_start_run, "run_status": _t_run_status, "restart_run": _t_restart_run,
    "stop_run": _t_stop_run,
    "read_file": _t_read_file, "list_files": _t_list_files, "grep": _t_grep, "outline": _t_outline,
    "find_definition": _t_find_definition, "find_references": _t_find_references,
    "dep_graph": _t_dep_graph, "trace_variable": _t_trace_variable, "doc_lookup": _t_doc_lookup,
    "edit_file": _t_edit_file, "replace_symbol": _t_replace_symbol, "write_file": _t_write_file,
    "run_command": _t_run_command, "todo_write": _t_todo_write,
    "check_shape": _t_check_shape, "smoke_test": _t_smoke_test,
}
_WRITERS = {"edit_file", "replace_symbol", "write_file"}
# their output is for the person (or they may have to ask the person before they run)
_LOUD = {"run_command", "message_user", "start_run", "restart_run", "stop_run", "check_shape", "smoke_test"}


def _brief(name, args):
    """One-line description of a call for the transcript."""
    key = {"read_file": "path", "outline": "path", "edit_file": "path", "replace_symbol": "path",
           "write_file": "path", "grep": "pattern", "find_definition": "symbol", "find_references": "symbol",
           "trace_variable": "target", "doc_lookup": "name", "run_command": "command",
           "list_files": "pattern", "start_run": "script", "check_shape": "exprs"}.get(name)
    detail = str(args.get(key, "")) if key else ""
    if name == "check_shape" and not args.get("exprs"):
        detail = (str(args.get("code") or "").strip().splitlines() or [""])[0]
    if name == "smoke_test":
        detail = "compile, probe, suite" if args.get("probe") else "compile, suite"
    if isinstance(args.get(key), list):
        detail = ", ".join(str(v) for v in args[key])
    if name == "start_run" and args.get("args"):
        detail += f" {args['args']}"
    if name == "read_file" and args.get("start_line"):
        detail += f":{args.get('start_line')}-{args.get('end_line') or ''}"
    if name == "replace_symbol":
        detail += f" :: {args.get('symbol', '')}"
    if name == "todo_write":
        detail = f"{len(args.get('todos') or [])} item(s)"
    return f"{name}  {detail}".rstrip()[:160]


def _show(name, args, result, state=None):
    """What the agent just did, in the Pulse app's transcript or as a compact terminal line."""
    host = _ui.host()
    if host is not None and hasattr(host, "tool"):
        said = getattr(state, "said", "") if state is not None else ""
        if state is not None:
            state.said = ""
        if name in _WRITERS and getattr(host, "_edit_pending", None) is not None:
            # the diff is already in the transcript (show_change): say what became of it
            host.edit_status("not applied" if result.startswith("NOT APPLIED") else "applied",
                             detail=said or None, final=True)
            return
        try:
            host.tool([f"{name.upper()}: {_brief(name, args)[len(name):].strip()}"], result, said=said)
        except TypeError:                   # an older host without `said`
            if said:
                _ui.answer(said)
            host.tool([f"{name.upper()}: {_brief(name, args)[len(name):].strip()}"], result)
        return
    cprint(f"● {_brief(name, args)}", color=_BLUE)
    if name in ("edit_file", "replace_symbol", "write_file", "todo_write"):
        first = result.splitlines()[0] if result else ""
        if name == "todo_write":
            print(textwrap.indent(result.split("\n", 1)[1] if "\n" in result else "", "    "))
        elif first and not result.startswith("NOT APPLIED"):
            print(f"    {first}")
        else:
            cprint("    " + _clip(result, 400).replace("\n", "\n    "), color=_YELLOW)
    elif name == "run_command":
        tail = result.splitlines()[-12:]
        print(textwrap.indent("\n".join(tail), "    "))
    elif name in ("smoke_test", "check_shape"):
        shown = result.splitlines()[:24]
        failing = result.startswith(("SMOKE TEST: FAIL", "CHECK_SHAPE: FAIL"))
        text = textwrap.indent("\n".join(shown) + ("\n    ..." if len(result.splitlines()) > 24 else ""), "    ")
        if name == "smoke_test":
            cprint(text, color=_YELLOW if failing else _GREEN)
        else:
            print(text)
    else:
        first = (result.splitlines() or [""])[0]
        print(f"    {first[:140]}")


def _execute(state, call):
    """Run one tool call. Always returns a string; never raises."""
    name = call["name"]
    raw = call["arguments"]
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool '{name}'. Available: {', '.join(_HANDLERS)}."
    try:
        args = json.loads(raw) if raw and raw.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("arguments are not an object")
    except (ValueError, TypeError):
        return (f"The arguments to {name} were not valid JSON -- your reply was probably cut off because it was "
                "too long. Retry with a smaller call (edit_file with a short snippet, or write the file in parts).")
    signature = (name, raw)
    state.repeats = state.repeats + 1 if signature == state.last_call else 1
    state.last_call = signature
    state.calls_made += 1
    host = _ui.host()
    try:
        if host is not None and hasattr(host, "hush") and name not in _WRITERS and name not in _LOUD:
            with host.hush():
                result = handler(state, args)
        else:
            result = handler(state, args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        result = f"{name} failed: {type(exc).__name__}: {exc}"
    result = _clip(str(result))
    if state.repeats >= _REPEAT_LIMIT:
        result += (f"\n\n(You have made this exact call {state.repeats} times in a row with the same result. "
                   "Change your approach.)")
    if name != "message_user":          # the message itself is what the user sees
        _show(name, args, result, state)
    return result


# ---------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------


def _project_context(cli, evidence=None):
    from . import pulse_code as _pc
    root = cli._project_root
    lines = [f"Project root: {root}"]
    notes_name, notes = cli.project_notes()
    if notes:
        lines += [f"\nProject notes ({notes_name}) -- follow these:", notes]
    index = []
    focus = set(cli.focus)
    for path in cli.known[:200]:
        index.append(f"{'*' if path in focus else ' '} {_rel(root, path)}")
    lines.append(f"\nProject files ({len(cli.known)}; * = the user pointed you at these):")
    lines += index
    if len(cli.known) > 200:
        lines.append(f"  ... and {len(cli.known) - 200} more (list_files / grep)")
    if evidence:
        lines += ["", str(evidence)]
    return "\n".join(lines)


def _messages_chars(messages):
    total = 0
    for m in messages:
        total += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            total += len(tc["function"]["arguments"])
    return total


def _elide_old_results(messages):
    """Shorten bulky tool results that are no longer recent -- the cheap first line of defence."""
    cutoff = len(messages) - _ELIDE_AFTER
    for i, m in enumerate(messages[:max(cutoff, 0)]):
        if m.get("role") == "tool" and len(m.get("content") or "") > _ELIDED_CHARS:
            m["content"] = m["content"][:_ELIDED_CHARS] + "\n...[older output shortened]"


def _render_for_summary(messages):
    parts = []
    for m in messages:
        role = m.get("role")
        text = str(m.get("content") or "")
        calls = "".join(f"\n  -> {tc['function']['name']}({tc['function']['arguments'][:300]})"
                        for tc in m.get("tool_calls") or [])
        parts.append(f"[{role}] {text[:2500]}{calls}")
    return "\n\n".join(parts)


def compact(cli, force=False):
    """Fold old conversation into a summary. Returns True if it ran."""
    from . import pulse_code as _pc
    messages = cli.native_history
    _elide_old_results(messages)
    if len(messages) <= _KEEP_RECENT + 2 or (not force and _messages_chars(messages) < _COMPACT_AT_CHARS):
        return False
    # Cut where the kept part starts cleanly: a user turn or an assistant message, never a stray tool result.
    cut = len(messages) - _KEEP_RECENT
    while cut > 1 and not (messages[cut].get("role") == "assistant"
                           or (messages[cut].get("role") == "user")):
        cut -= 1
    if cut <= 1:
        return False
    old, keep = messages[:cut], messages[cut:]
    transcript = _render_for_summary(old)
    if len(transcript) > _SUMMARY_INPUT_CHARS:
        transcript = "...(earlier part omitted)...\n" + transcript[-_SUMMARY_INPUT_CHARS:]
    try:
        with _Spinner("Compacting context"):
            summary = cli._call_model(f"{_pc._SUMMARIZE}\n\n---\n{transcript}", max_tokens=_AGENT_MAX_TOKENS,
                                      history=[], system=_system_prompt(cli))
    except AgentRequestFailed:
        return False
    cli.native_history[:] = [{"role": "user", "content": f"Summary of the earlier conversation and work:\n{summary.strip()}"}] + keep
    return True


def _repair(messages):
    """After an interrupt: every tool call in the last assistant message needs a result."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            answered = {x.get("tool_call_id") for x in messages[i + 1:] if x.get("role") == "tool"}
            for tc in m["tool_calls"]:
                if tc["id"] not in answered:
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"],
                                     "content": "(interrupted before this ran)"})
            return
        if m.get("role") == "user":
            return


# ---------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------


class ToolsUnsupported(Exception):
    """The provider rejected function calling on the first request."""


class _Reply:
    def __init__(self, text, calls, finish, raw_message):
        self.text, self.calls, self.finish, self.raw_message = text, calls, finish, raw_message


def _chat(cli, messages):
    """One completion with tools. Retries transient errors; raises AgentRequestFailed otherwise."""
    model = cli.agent_model_string or PROVIDERS[cli.agent_provider]["model"]
    max_tokens = _clamp_output_tokens(model, _AGENT_MAX_TOKENS)
    # the app adds what a live run changes (DEBUG_PROMPT_NATIVE) while one is open
    payload = [{"role": "system", "content": _system_prompt(cli)}] + messages
    last_exc = None
    waiter = _ratelimit.RateLimitWaiter()      # a rate limit is waited out, not counted as an attempt
    attempt = 0
    while attempt < 4:
        attempt += 1
        try:
            response = _complete(
                model=model, messages=payload, tools=_tools(), tool_choice="auto", max_tokens=max_tokens,
                timeout=_AGENT_TIMEOUT_SECONDS, api_base=cli.agent_api_base,
                api_key=(cli.agent_key if cli.agent_key and cli.agent_key != "local" else None))
            cli._record_usage(response)
            choice = response.choices[0]
            msg = choice.message
            calls = []
            for tc in getattr(msg, "tool_calls", None) or []:
                calls.append({"id": tc.id or f"call_{time.monotonic_ns()}", "name": tc.function.name,
                              "arguments": tc.function.arguments or ""})
            text = (msg.content or "").strip()
            if not text and not calls:
                raise RuntimeError("the provider returned an empty response")
            return _Reply(text, calls, getattr(choice, "finish_reason", None), msg)
        except AgentRequestFailed:
            raise
        except Exception as exc:
            last_exc = exc
            if cli._is_rate_limited(exc):
                cli._wait_out_rate_limit(exc, waiter, "pulse code")      # raises when it should give up
                attempt -= 1
                continue
            retryable = cli._is_retryable_model_error(exc) or isinstance(exc, RuntimeError)
            if attempt < 4 and retryable:
                backoff = 2 ** (attempt - 1)
                cprint(f"[Pulse] ⚠ Agent request hit a transient error (attempt {attempt}/4), retrying in "
                       f"{backoff}s: {cli._classify_model_error(exc)}", color=_RED)
                time.sleep(backoff)
                continue
            transient = retryable
            raise AgentRequestFailed(cli._classify_model_error(exc), transient=transient) from exc
    raise AgentRequestFailed(cli._classify_model_error(last_exc) if last_exc else "unknown error")


def _assistant_message(reply):
    message = {"role": "assistant", "content": reply.text or None}
    if reply.calls:
        message["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                 for c in reply.calls]
    for extra in ("reasoning_content", "thinking_blocks"):
        value = getattr(reply.raw_message, extra, None)
        if value:
            message[extra] = value
    return message


# ---------------------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------------------


# phrasings of "I am about to do X" -- not advice ("you can", "I would suggest", "the next
# step would be"), which is a legitimate way to end an answer
_INTENT_RE = re.compile(
    r"(?i)\b(?:let me|let's|i(?:'ll| will| am going to| need to)|next,? i|first,? i|"
    r"i'?m going to|now i|i will now|i(?:'ll| will) (?:start|begin|now))\b(?! know)")


def _nudge(state, reply_text=""):
    """A reason the agent should not stop yet, or None."""
    open_items = [t for t in state.todos if t["status"] != "completed"]
    if open_items:
        depth = {}
        by_id = {todo["id"]: todo for todo in state.todos}

        def task_depth(task_id):
            if task_id not in depth:
                parent_id = by_id[task_id].get("parent_id")
                depth[task_id] = task_depth(parent_id) + 1 if parent_id else 0
            return depth[task_id]

        for todo in state.todos:
            task_depth(todo["id"])
        active = next((todo for todo in open_items
                       if not any(child.get("parent_id") == todo["id"] for child in state.todos)
                       and todo["status"] == "in_progress"), None)
        active_text = (f"The active leaf is {active['content']!r}. Finish it, mark it completed, "
                       "and activate the next unfinished leaf with todo_write. "
                       if active else "")
        return ("Your task outline still has unfinished work. " + active_text
                + "Continue the outline before giving a final answer. If the active item is too broad, "
                "replace it with concrete child subtasks at any depth using todo_write.\n"
                + "\n".join(
                    f"{'  ' * depth[t['id']]}- {t['content']} ({t['status']})"
                    for t in open_items
                ))
    if (reply_text and not state.calls_made and not state.changes
            and _INTENT_RE.search(reply_text.strip()[-300:]) and len(reply_text) < 1500):
        # It ended by announcing what it would do and stopped (no tool was called). Doing is
        # the job. Only the end of the reply counts: an answer that mentions "let me know" or
        # suggests a next step in its middle is still an answer.
        return ("Your reply ended by saying what you would do, without doing it. If it needs doing, do it "
                "now, in this reply, with the tools (read, grep, edit, run_command, start_run...). If that "
                "reply was your complete answer, repeat it on its own.")
    if state.dirty:
        if state.smoke_ok is False:
            return ("Your last smoke_test failed and you have not fixed it. Read what it reported. If your change "
                    "caused it, fix the cause and run smoke_test again; if the failure is unrelated to your change "
                    "(a test that was already failing), say so in your final answer instead of fixing it.")
        return ("You changed files but have not checked the result. Run smoke_test now (with a `probe` that builds "
                "what you changed on a tiny input if it touches a model, data, loss or any shape), and fix any "
                "failure. If nothing can be run here, say so plainly in your final answer.")
    return None


def _record_turn(state):
    """One /undo commit for everything this request changed."""
    cli = state.cli
    changed = {p: (b, a) for p, (b, a, _c) in state.changes.items() if b != a}
    if not changed:
        return None
    created = {p for p, (_b, _a, c) in state.changes.items() if c}
    commit = cli._record_fix_commit(changed, f"Pulse Code: {state.request[:80]}", created=created)
    cli._last_commit_id = commit
    cli._last_applied_fix = {
        "old": [], "new": [], "files": [_rel(state.root, p) for p in changed],
        "explanation": f"Pulse Code: {state.request[:120]}"}
    return commit


def run_native_turn(cli, request, evidence=None):
    """Work `request` to completion. Returns "applied", "answered" or "failed"."""
    messages = cli.native_history
    compact(cli)
    state = _State(cli, request)
    cli._todos = []
    cli.reload()
    marker = {"role": "user", "content": f"{_project_context(cli, evidence)}\n\nRequest: {request}"}
    messages.append(marker)
    outcome, final_text, unsupported = "failed", "", False
    try:
        for _i in range(_MAX_ITERATIONS):
            if _i and _i % 8 == 0:
                compact(cli)
            try:
                with _Spinner("Working"):
                    reply = _chat(cli, messages)
            except AgentRequestFailed as exc:
                detail = str(exc.__cause__ or exc)
                if _i == 0 and not exc.rate_limited and re.search(r"\b(tools?|function[ _]call\w*)\b", detail, re.I):
                    messages.remove(marker)          # nothing happened; let the caller use the text pipeline
                    unsupported = True
                    break
                cprint(f"[Pulse Code] ⚠ The agent request failed: {exc}", color=_RED)
                final_text = f"(stopped: the agent request failed: {exc})"
                break
            messages.append(_assistant_message(reply))
            if reply.text:
                if reply.calls and _ui.host() is not None and hasattr(_ui.host(), "tool"):
                    state.said = reply.text          # goes with the first call's line (see _show)
                else:
                    _ui.answer(reply.text)
            if not reply.calls:
                if reply.finish == "length":
                    messages.append({"role": "user", "content": "Your reply was cut off. Continue where you left off."})
                    continue
                has_unfinished_tasks = any(todo["status"] != "completed" for todo in state.todos)
                reason = (_nudge(state, reply.text)
                          if has_unfinished_tasks or state.nudges < _MAX_NUDGES else None)
                if reason:
                    state.nudges += 1
                    cprint("[Pulse Code] Not finished yet -- continuing.", color=_YELLOW)
                    messages.append({"role": "user", "content": reason})
                    continue
                final_text = reply.text
                outcome = "applied" if state.changes else "answered"
                break
            for call in reply.calls:
                result = _execute(state, call)
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})
            if state.edits_declined >= 3:
                cprint("[Pulse Code] Stopping: you declined several changes in a row.", color=_YELLOW)
                final_text = "(stopped after repeated declined changes)"
                outcome = "declined"
                break
        else:
            cprint(f"[Pulse Code] Stopped after {_MAX_ITERATIONS} model calls; say 'continue' to keep going.",
                   color=_YELLOW)
            final_text = "(stopped at the step limit)"
    finally:
        _repair(messages)
        marker["content"] = f"Request: {request}"
        commit = None if unsupported else _record_turn(state)
        if commit:
            n = sum(1 for b, a, _c in state.changes.values() if b != a)
            cprint(f"\n✓ {n} file{'s' if n != 1 else ''} changed  ·  /undo to revert (commit {commit})", color=_GREEN)
        if not unsupported:
            try:
                cli._sync_agent_turn(request, final_text or "(no answer)", traceback_signature=None,
                                     fix_applied=cli._last_applied_fix if commit else None)
            except Exception:
                pass
    if unsupported:
        raise ToolsUnsupported()
    if state.changes and outcome == "failed":
        outcome = "applied"
    return outcome