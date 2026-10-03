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
import os
import re
import textwrap
import time

from . import pulse_ui as _ui
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
_MAX_NUDGES = 2                  # times a stop is overruled because work looks unfinished

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
- For anything with more than ~2 steps, keep a todo list with todo_write and update it as you go. Do not finish with unfinished todos.
- Read before you write. Never guess at signatures, imports, names or file layout.
- Make the change the request needs, matching the surrounding style. Don't reformat, rename or "improve" unrelated code. If you notice a separate bug, mention it in your final answer instead of fixing it.
- Prefer small, targeted edits (edit_file) or replace_symbol for rewriting a whole function/class. Use write_file for new files. Make independent tool calls in the same turn when you can.
- If a tool fails, read the error, adjust, and try again; a failed edit or command is information, not the end of the task. Don't repeat the identical call hoping for a different result.
- If the request is a question, answer it from the code; make no edits.

# Use the code structure, not just text search
This is Pulse's advantage. For Python code prefer the AST tools over grep:
- outline: the classes/functions of a file with line ranges. Use it before reading a big file, then read_file just the range you need.
- find_definition / find_references: exact jump-to-definition and call sites.
- dep_graph: which project files import which.
- trace_variable: everything that feeds a variable and everything it feeds, across functions and files. Use it to find where a bad value came from.
- replace_symbol: replace a function/class by name.
- After an edit that changes a function's signature, the tool result lists the call sites that may now be wrong. Fix them.

# Running things
run_command runs a real command in the project directory and returns the real stdout/stderr/exit code. Use it for tests, linters, builds, git, and small repro scripts. Commands that delete files, rewrite git history, reach outside the project, start background processes or overwrite files via redirects pause for the user's confirmation; ordinary read/run/test commands do not.

# Finishing
When the work is done and verified, reply with a short summary: what you changed, how you checked it, and anything the user should know. No tool call in a reply means you are finished, so only reply without one when you are.
"""

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
    _fn("todo_write",
        "Replace your todo list. Each item: {content, status} with status pending | in_progress | completed. "
        "Keep exactly one item in_progress.",
        {"todos": {"type": "array", "items": {"type": "object", "properties": {
            "content": _S, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
            "required": ["content", "status"]}}}, ["todos"]),
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
        self.nudges = 0
        self.last_call = None
        self.repeats = 0
        self.edits_declined = 0

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
    print("\n" + _pc.render_diff(cli, {path: (before, after, created)}))
    if not cli.review:
        return True
    try:
        answer = _prompt_text("Apply this change? (Y/n/a=apply all from now on) > ",
                              label="Apply this change?  (Y/n/a=always)").strip().lower()
    except EOFError:
        cprint("[Pulse Code] No terminal to confirm on -- not applied (use -y to apply without asking).",
               color=_YELLOW)
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
        return ("The user declined this change. Do not retry the same edit. Either take a different approach, "
                "or stop and say what you were trying to do and ask how they would like to proceed.")
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
        return f"Read '{a.get('path')}' (or its outline) with read_file before editing it."
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
    return result


_STATUS_MARK = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}


def _t_todo_write(state, a):
    todos = a.get("todos")
    if not isinstance(todos, list):
        return "todos must be a list of {content, status}."
    clean = []
    for t in todos:
        if isinstance(t, dict) and isinstance(t.get("content"), str):
            status = t.get("status") if t.get("status") in _STATUS_MARK else "pending"
            clean.append({"content": t["content"], "status": status})
    state.todos = clean
    state.cli._todos = clean
    return "Todo list updated:\n" + "\n".join(f"{_STATUS_MARK[t['status']]} {t['content']}" for t in clean)


_HANDLERS = {
    "read_file": _t_read_file, "list_files": _t_list_files, "grep": _t_grep, "outline": _t_outline,
    "find_definition": _t_find_definition, "find_references": _t_find_references,
    "dep_graph": _t_dep_graph, "trace_variable": _t_trace_variable, "doc_lookup": _t_doc_lookup,
    "edit_file": _t_edit_file, "replace_symbol": _t_replace_symbol, "write_file": _t_write_file,
    "run_command": _t_run_command, "todo_write": _t_todo_write,
}
_WRITERS = {"edit_file", "replace_symbol", "write_file"}


def _brief(name, args):
    """One-line description of a call for the transcript."""
    key = {"read_file": "path", "outline": "path", "edit_file": "path", "replace_symbol": "path",
           "write_file": "path", "grep": "pattern", "find_definition": "symbol", "find_references": "symbol",
           "trace_variable": "target", "doc_lookup": "name", "run_command": "command",
           "list_files": "pattern"}.get(name)
    detail = str(args.get(key, "")) if key else ""
    if name == "read_file" and args.get("start_line"):
        detail += f":{args.get('start_line')}-{args.get('end_line') or ''}"
    if name == "replace_symbol":
        detail += f" :: {args.get('symbol', '')}"
    if name == "todo_write":
        detail = f"{len(args.get('todos') or [])} item(s)"
    return f"{name}  {detail}".rstrip()[:160]


def _show(name, args, result):
    """What the agent just did, in the Pulse app's transcript or as a compact terminal line."""
    host = _ui.host()
    if host is not None and hasattr(host, "tool"):
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
    host = _ui.host()
    try:
        if host is not None and hasattr(host, "hush") and name not in _WRITERS and name != "run_command":
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
    _show(name, args, result)
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
                                      history=[], system=SYSTEM_PROMPT)
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
    payload = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    last_exc = None
    for attempt in range(1, 5):
        try:
            response = litellm.completion(
                model=model, messages=payload, tools=TOOLS, tool_choice="auto", max_tokens=max_tokens,
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


def _nudge(state):
    """A reason the agent should not stop yet, or None."""
    open_items = [t for t in state.todos if t["status"] != "completed"]
    if open_items:
        return ("You stopped, but your todo list still has unfinished items:\n"
                + "\n".join(f"- {t['content']} ({t['status']})" for t in open_items)
                + "\nContinue with them, or update the list if they are no longer needed.")
    if state.dirty:
        return ("You changed files but have not run anything to check the result. Run the project's tests/linter "
                "or a quick script that exercises the change and fix any failure. If nothing can be run here, "
                "say so plainly in your final answer.")
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
                if _i == 0 and re.search(r"\b(tools?|function[ _]call\w*)\b", detail, re.I):
                    messages.remove(marker)          # nothing happened; let the caller use the text pipeline
                    unsupported = True
                    break
                cprint(f"[Pulse Code] ⚠ The agent request failed: {exc}", color=_RED)
                final_text = f"(stopped: the agent request failed: {exc})"
                break
            messages.append(_assistant_message(reply))
            if reply.text:
                print(f"\n{reply.text}\n" if not reply.calls else f"\n{reply.text}")
            if not reply.calls:
                if reply.finish == "length":
                    messages.append({"role": "user", "content": "Your reply was cut off. Continue where you left off."})
                    continue
                reason = _nudge(state) if state.nudges < _MAX_NUDGES else None
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
