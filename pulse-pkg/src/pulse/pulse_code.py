"""
Pulse Code -- a general coding agent in the terminal.

    pulse code                       open the agent in this directory
    pulse code src/model.py utils/   ...pointed at some files or folders
    pulse code -p "add a --resume flag to train.py"      one request, then exit

Setup is the debugger's, untouched: `PulseCLI.interactive_setup()` -- sign in or log in,
workspace, code version, agent model, API key -- so an account, a workspace and a chosen
agent carry straight over between `pulse run` and `pulse code`.

What happens after that is built from the debugger's own tooling rather than a second
implementation of it:

  * the model is called through `PulseCLI._call_model` (retry/backoff, token clamp, usage);
  * it can read the project with the same directive tools the debugger's agent has --
    GREP / VIEW / DEFOF / CALLERS / DEPGRAPH / DOCLOOKUP / CHANGELOG -- serviced by
    `_service_tool_requests`;
  * edits are the debugger's code-fix format applied by `_apply_code_fix`: exact snippets
    matched once, a fuzzy fallback for near misses, the automatic syntax/lint gate on every
    file before it is written, and a commit in `.pulse_history` (plus `_request_corrected_snippets`
    when a snippet doesn't match);
  * every turn is synced to the cloud session by `_sync_agent_turn`.

What is different, on purpose, because this is a coding session and not a training run:

  * its own system prompt and passes -- the debugger's are "fix the bug and only the bug";
  * new files, which the fix format could not express (`create`);
  * nothing restarts, no training probe runs, and your git tree is never stashed;
  * the agent may only edit files it has loaded from inside the project, and its toolset is the
    debugger's read-only tools plus `TERMINAL:` (a real subprocess in the project directory,
    serviced by the same shared executor the debugger's `TERMINAL:` uses) -- not `REPL:`/`DRYRUN:`
    (an eval against a live *training* process, which doesn't exist in a coding session);
  * after a change is applied, the agent gets one more turn to verify it for real -- compile/syntax
    -check, run a targeted test, reproduce the original problem -- via `TERMINAL:`, and can propose a
    follow-up fix if that verification fails (`_run_verification_pass`, bounded like everything else
    here);
  * changes are shown as a diff and confirmed first (`/review off` or `-y` to skip);
  * a request is all-or-nothing: if any part fails to match or lint, none of it is kept;
  * `/undo` reverts only the files still exactly as the agent left them, because the
    debugger's `/revert` restores every touched file to its logged state, which would overwrite
    edits you have made by hand since.
"""
import difflib
import fnmatch
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

from . import pulse_ui as _ui
from .pulse_cli import (
    AgentRequestFailed,
    PulseCLI,
    _AGENT_MAX_TOKENS,
    _BLUE,
    _GREEN,
    _RED,
    _Spinner,
    _YELLOW,
    _find_fuzzy_snippet_span,
    _flush_stdin,
    _prompt_text,
    cprint,
)

# ---------------------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------------------

_MAX_FILE_BYTES = 200_000            # per file; bigger files are skipped, not truncated
_MAX_PROJECT_FILES = 400             # known to the agent's tools
_CONTEXT_BUDGET_CHARS = 240_000      # focus-file text sent on each pass (~60k tokens)
_INDEX_MAX_LINES = 160               # entries in the project index shown to the model
_MAX_TOOL_ROUNDS = 3
_MAX_REVISE_ROUNDS = 2
_DIFF_MAX_LINES = 240
_MAX_VERIFY_TOOL_ROUNDS = 5     # TERMINAL/GREP/VIEW rounds within one verification pass
_MAX_VERIFY_FIX_CYCLES = 2      # additional change->verify loops if verification fails

_TEXT_EXTS = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini",
    ".sh", ".bash", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".sql", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".cu", ".cuh", ".java", ".go", ".rs", ".rb", ".lua", ".r", ".env.example",
}
_TEXT_NAMES = {"Makefile", "Dockerfile", "Procfile", "README", "LICENSE", ".gitignore"}
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", ".idea", ".vscode", ".pulse_history",
    ".pulse_stream", "site-packages", ".eggs",
}
_PROTECTED_PARTS = (".git", ".pulse_history")

# Read-only tools, plus TERMINAL -- a coding session's one execution primitive. The
# debugger's toolset also has REPL:/DRYRUN: (an eval / a call in the live training
# process), REPLAY:, ROLLBACK: and training-run statistics; none of those belong here --
# there is no live training process to eval against, and this session has its own
# `/undo` for reverting edits. TERMINAL runs a real subprocess in the project directory,
# which is exactly what a coding session needs for tests/linters/repro scripts and is
# handled by the same shared executor + confirmation policy as the debugging agent's
# TERMINAL: (see PulseCLI._run_terminal / pulse_terminal.py).
_ALLOWED_TOOLS = {"defof", "callers", "depgraph", "trace", "doclookup", "changelog", "terminal"}

# ---------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------

CODE_SYSTEM_PROMPT = (
    "You are Pulse Code, a coding agent working in the user's terminal, inside their real "
    "project. You add features, fix bugs, refactor and explain code in the files the user points "
    "you at.\n\n"
    "HOW YOU WORK\n"
    "- You are shown an index of the project and the full, line-numbered text of the files in "
    "focus. Anything else you can read with the tools below. Never guess at code you have not "
    "been shown -- signatures, imports, names, file layout.\n"
    "- Make the smallest change that fully does what was asked. Match the surrounding style, "
    "naming and structure. Do not reformat, rename, restructure or 'improve' code the request "
    "does not need touched. Do not add a dependency the project does not already use unless asked. "
    "If you notice a second, unrelated bug while you're in a file, do not fix it -- mention it in "
    "your explanation as worth a separate look, and leave that code untouched.\n"
    "- If the request is a question, or needs no change, just answer it.\n"
    "- You have a real terminal (TERMINAL:, below). Never say you ran or tested something unless you "
    "actually did and are reporting the real exit code/output you got back -- not what you expect it "
    "to say.\n\n"
    "TOOLS -- put the directive alone on its own line; Pulse runs it and replies with the "
    "results, then you continue. Use file labels exactly as they appear in the headers.\n"
    "  GREP: <pattern>            search every project file (word, phrase or regex), with context\n"
    "  VIEW: <file>:<start>-<end>  an exact line range from a file, e.g. VIEW: src/model.py:40-75\n"
    "  DEFOF: <symbol>            jump to where a function/class is defined\n"
    "  CALLERS: <symbol>          every call site of a function/class\n"
    "  DEPGRAPH:                  the import graph between project files\n"
    "  TRACE: <var>[:<file>[:<line>]]  the variable's whole connected influence path: everything "
    "that feeds it (across function/file boundaries) and everything it feeds in turn, from the "
    "real assignment chain, not a guess. self.<attr> works too.\n"
    "  DOCLOOKUP: <library>.<symbol>  the real signature/docstring of an installed library function\n"
    "  CHANGELOG:                 what has changed in the project files since the session started\n"
    "  TERMINAL: <shell command>  run a REAL command in the project directory; get back the actual "
    "stdout, stderr, exit code and duration -- e.g. 'TERMINAL: pytest tests/test_model.py', "
    "'TERMINAL: python -m py_compile src/model.py', 'TERMINAL: git status', 'TERMINAL: python "
    "repro.py'. This is a general terminal, not a fixed set of commands -- compose whatever command "
    "actually answers your question. Prefer it over guessing: to see whether a file parses, to "
    "reproduce a bug before proposing a fix, to run a project's existing tests/linters, or to check "
    "git state. A command that deletes files, rewrites git history, reaches outside the project, or "
    "starts a background process pauses for the user's confirmation first -- expect that only for "
    "those cases, not for ordinary read/run/test commands. Large output comes back truncated (head "
    "and tail kept); narrow the command if you need a different slice.\n"
)

_PLAN = (
    "STEP 1 -- PLAN. Read the request and the code. If you need to see more code first, use the "
    "tools (one directive per line) and stop; you will get the results. Otherwise: if the request "
    "is a question or needs no code change, answer it fully and end with a line containing only "
    "NO_CHANGES. If it does need changes, give a short plan -- which files and regions change, "
    "what is added, and anything risky or ambiguous. Scope the plan to exactly what was asked; note "
    "anything else you noticed as a separate observation, not as part of this plan. Do not write "
    "the code yet."
)

_IMPLEMENT = (
    "STEP 2 -- IMPLEMENT the plan.\n"
    "Fix the request and only the request: no unrelated formatting, no renaming, no reordering "
    "imports, no 'while I am here' cleanups, no speculative refactors -- even where you can see "
    "something you would write differently. If you notice a second, unrelated problem, do not fix "
    "it in this change; mention it in the explanation field as worth a separate look. Every line "
    "you touch beyond what the request needs is a line that can break something that already works.\n"
    "Respond with ONLY one JSON object -- no prose, no markdown fences:\n"
    '{{"old": [...], "new": [...], "files": [...], "create": [...], "explanation": "one sentence"}}\n'
    "- old[i] is an exact, verbatim snippet from the file's shown text (WITHOUT the line-number "
    "prefixes) that occurs exactly once in that file -- include enough neighbouring lines (2-4) to "
    "be unique. new[i] is what replaces it. files[i] is the exact header label of the file that "
    "old[i] is in.\n"
    "- To INSERT code, use a nearby existing line as old and repeat it in new with your addition.\n"
    '- create: NEW files only -- a list of {{"path": "relative/to/project/root.py", "content": '
    '"full file text"}}. Never for a file that already exists. Use [] when there is nothing to create.\n'
    "- If you only create files, old, new and files are [].\n"
    "- Keep every edit as small as the request allows. Preserve indentation exactly.\n"
    "- If you still need to see code, use a tool directive instead of guessing.\n\n"
    "The plan:\n{plan}"
)

_NO_TOOLS_NOTE = (
    "That was neither tool directives nor the JSON edit object. Respond with ONLY the JSON object "
    "described in step 2 (or a tool directive if you need to see more code)."
)

_VERIFY = (
    "STEP 3 -- VERIFY. The request was:\n{request}\n\nThe change you are about to make:\n{changes}\n\n"
    "Check it against the code you were shown: does it fully do what was asked? Any syntax errors, "
    "undefined names, missing imports, wrong signatures, or missing wiring between files? Check "
    "SCOPE too: does every changed line trace directly to the request, or does the diff also carry "
    "formatting changes, renames, reordered imports, or a fix to something unrelated that crept in "
    "alongside it? Respond with ONLY "
    '{{"passes": true or false, "reason": "one sentence"}}.'
)

_REVISE = (
    "Your change has a problem:\n{problem}\n\n"
    "Revise it. Respond with ONLY the corrected, complete JSON object from step 2 -- every change, "
    "not just the fixed one -- no prose, no fences."
)

_VERIFY_TERMINAL = (
    "STEP 4 -- VERIFY FOR REAL. This change was just applied:\n{explanation}\n\n{diff}\n\n"
    "Use TERMINAL: (and GREP:/VIEW: if useful) to actually check it -- don't just reason about "
    "whether it should work. Pick whatever's appropriate: compile/syntax-check the changed file(s), "
    "run a targeted existing test, or write and run a tiny repro script for the specific behavior "
    "that was asked for. Put one or more directives on their own lines and stop; you'll get the real "
    "output back and can run more before deciding. When you're done checking, respond with ONLY "
    '{{"verified": true or false, "reason": "one sentence, citing what you actually saw (exit code / '
    'output), not what you expect"}}.'
)

_VERIFY_NO_TOOLS_NOTE = (
    "That was neither a tool directive nor the JSON verdict. Either run a TERMINAL:/GREP:/VIEW: "
    'directive, or respond with ONLY {"verified": true or false, "reason": "..."}.'
)




# ---------------------------------------------------------------------------------------
# The session object: PulseCLI, with a coding session's differences
# ---------------------------------------------------------------------------------------

class _CodeAgentCLI(PulseCLI):
    """PulseCLI -- so setup, the model call, the tools, the applier and the history are the
    debugger's -- minus the behaviours that only make sense for a training run."""

    # -- what a training run does that a coding session must not ------------------------
    def _git_autostash(self):
        # Stashing the working tree before an edit would stash the very files being worked on
        # (a coding session's tree is dirty by definition) and hide the user's own changes.
        return None

    def _verify_fix_empirically(self, fix, diagnosis):
        return fix, None, "no training run to measure"

    def _restart_process(self):
        return None

    def _fix_log_dir(self):
        # History lives at the project root, not beside whichever file happened to be first.
        root = getattr(self, "_project_root", None)
        if not root:
            return super()._fix_log_dir()
        directory = os.path.join(root, ".pulse_history")
        try:
            os.makedirs(os.path.join(directory, "diffs"), exist_ok=True)
        except OSError:
            return None
        return directory

    def _fixlog_path(self):
        return os.path.join(getattr(self, "_project_root", None) or ".", ".pulse_fixlog.json")

    # -- tools: read-only ones only ------------------------------------------------------
    @classmethod
    def _extract_new_directives(cls, text):
        cleaned, requests = super()._extract_new_directives(text)
        return cleaned, {k: v for k, v in requests.items() if k in _ALLOWED_TOOLS}

    def _apply_directives(self, calc_exprs, promote_names, gputrack_names=None, gpuuntrack_names=None,
                          sensitivity_args=None, normal_start_args=None, grep_patterns=None,
                          view_requests=None):
        # Only GREP and VIEW; CALC / PROMOTE / GPUTRACK / SENSITIVITY are about a training run.
        return super()._apply_directives([], [], [], [], [], [], grep_patterns, view_requests)

    # -- labels are project-relative paths: unique by construction -----------------------
    def _build_file_labels(self):
        root = self._project_root
        labels, paths = {}, {}
        for path in [self.script_path] + list(self.extra_files):
            if not path or path in labels:
                continue
            label = os.path.relpath(path, root).replace(os.sep, "/")
            labels[path] = label
            paths[label] = path
        self._label_for_path = labels
        self._path_for_label = paths

    # -- files ---------------------------------------------------------------------------
    def setup_code(self, root, focus, known):
        self._code_mode = True
        self._project_root = root
        self._system_prompt_override = CODE_SYSTEM_PROMPT
        self._suppress_auto_restart = True
        self.review = True
        self.focus = list(focus)
        self.known = list(dict.fromkeys(list(focus) + list(known)))
        self.texts = {}
        self._last_focus_note = ""
        self.reload()

    def reload(self):
        """Re-read every known file from disk (the user edits alongside the agent) and
        refresh what the debugger's file plumbing sees."""
        texts = {}
        for path in self.known:
            text = read_text(path)
            if text is not None:
                texts[path] = text
        self.texts = texts
        self.known = [p for p in self.known if p in texts]
        self.focus = [p for p in self.focus if p in texts]
        main = self.focus[0] if self.focus else None
        self.script_path = main
        self.code_text = texts.get(main) if main else None
        self.extra_files = {p: t for p, t in texts.items() if p != main}
        self._build_file_labels()

    def add_files(self, paths, focus=True):
        for path in paths:
            if path not in self.known:
                self.known.append(path)
            if focus and path not in self.focus:
                self.focus.append(path)
        self.reload()

    def drop_focus(self, path):
        if path in self.focus:
            self.focus.remove(path)
        self.reload()


# ---------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------

def read_text(path):
    """A text file's contents, or None for anything unreadable, binary or over the size cap."""
    try:
        if os.path.getsize(path) > _MAX_FILE_BYTES:
            return None
        with open(path, "rb") as handle:
            raw = handle.read()
        if b"\x00" in raw[:4096]:
            return None
        return raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _looks_like_source(name):
    base = os.path.basename(name)
    return base in _TEXT_NAMES or os.path.splitext(base)[1].lower() in _TEXT_EXTS


def scan_project(root, limit=_MAX_PROJECT_FILES):
    """The project's text files: `git ls-files` (which honours .gitignore) when this is a git
    repository, else a walk that skips the usual build/cache/venv directories."""
    files = []
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root,
                             capture_output=True, timeout=10)
        if out.returncode == 0 and out.stdout:
            for rel in out.stdout.decode("utf-8", "replace").split("\0"):
                if rel and _looks_like_source(rel) and not any(p in _SKIP_DIRS for p in rel.split("/")[:-1]):
                    files.append(os.path.join(root, rel))
    except Exception:
        files = []
    if not files:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                if _looks_like_source(name):
                    files.append(os.path.join(dirpath, name))
    files = [f for f in dict.fromkeys(files) if os.path.isfile(f)]
    return files[:limit]


def expand_paths(root, args):
    """Files and folders a person pointed at -> absolute file paths. Returns (files, problems)."""
    files, problems = [], []
    for arg in args:
        pattern = os.path.expanduser(arg)
        if not os.path.isabs(pattern):
            pattern = os.path.join(root, pattern)
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            problems.append(f"no match for '{arg}'")
        for match in matches:
            match = os.path.abspath(match)
            if os.path.isdir(match):
                found = scan_project(match)
                if not found:
                    problems.append(f"no source files under '{arg}'")
                files.extend(found)
            elif os.path.isfile(match):
                if read_text(match) is None:
                    problems.append(f"'{arg}' is binary, unreadable or over {_MAX_FILE_BYTES // 1000} KB -- skipped")
                else:
                    files.append(match)
            else:
                problems.append(f"'{arg}' not found")
    return list(dict.fromkeys(files)), problems


def _inside(root, path):
    try:
        return os.path.commonpath([root, os.path.abspath(path)]) == root
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------

def build_context(cli):
    """The project index plus the full line-numbered text of the files in focus, in the
    header format the debugger's file labels and edit format already use."""
    cli._build_file_labels()
    root = cli._project_root
    lines = [f"Project root: {root}", ""]
    focus_set = set(cli.focus)
    index = []
    for path in cli.known:
        label = cli._label_for_path.get(path, os.path.relpath(path, root))
        index.append(f"{'*' if path in focus_set else ' '} {label}  ({len(cli.texts.get(path, '').splitlines())} lines)")
    shown = index[:_INDEX_MAX_LINES]
    lines.append(f"Project files ({len(index)}; * = in focus, shown in full below):")
    lines.extend(shown)
    if len(index) > len(shown):
        lines.append(f"  ... and {len(index) - len(shown)} more (use GREP / DEFOF to find things)")

    if not cli.focus:
        lines.append("\nNo file is in focus. Read what you need with the tools (GREP / VIEW / DEFOF).")
        return "\n".join(lines)

    budget = _CONTEXT_BUDGET_CHARS
    for path in cli.focus:
        text = cli.texts.get(path, "")
        label = cli._label_for_path.get(path, os.path.basename(path))
        numbered = "\n".join(f"{i + 1:>4} | {line}" for i, line in enumerate(text.splitlines()))
        if len(numbered) > budget:
            lines.append(f"\n=== {label} === (too large to include in full; read it with VIEW / GREP)")
            continue
        budget -= len(numbered)
        lines.append(f"\n=== {label} (line-numbered) ===\n```\n{numbered}\n```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Parsing the model's edit
# ---------------------------------------------------------------------------------------

def parse_change(cli, answer):
    """The model's JSON -> {"old","new","files","create","explanation"}, or None.

    The edit half is parsed by the debugger's own `_parse_code_fix`; `create` (new files) is
    the addition. A response that only creates files is valid here."""
    text = (answer or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not any(k in payload for k in ("old", "create")):
        for value in payload.values():                       # an envelope: {"json": {...}}
            if isinstance(value, dict) and any(k in value for k in ("old", "create")):
                payload = value
                break
    create = payload.get("create") or []
    if not isinstance(create, list) or not all(isinstance(c, dict) for c in create):
        return None
    create = [c for c in create if isinstance(c.get("path"), str) and isinstance(c.get("content"), str)]
    explanation = payload.get("explanation") if isinstance(payload.get("explanation"), str) else ""
    old, new = payload.get("old") or [], payload.get("new") or []
    if old or new:
        edit = cli._parse_code_fix(json.dumps({"old": old, "new": new, "files": payload.get("files"),
                                               "explanation": explanation}))
        if edit is None:
            return None
        return {**edit, "create": create}
    if not create:
        return None
    return {"old": [], "new": [], "files": [], "create": create, "explanation": explanation}


def normalise_targets(cli, fix):
    """Point every edit at a file this session has loaded, by its exact label. Anything the
    model names that is not one of those is refused (returns the problems)."""
    problems = []
    focus_labels = [cli._label_for_path[p] for p in cli.focus if p in cli._label_for_path]
    files = []
    for old, label in zip(fix["old"], fix["files"]):
        label = (label or "").strip()
        path = cli._path_for_label.get(label)
        if path is None and label:
            hits = [p for l, p in cli._path_for_label.items() if l.endswith("/" + label) or l == label]
            path = hits[0] if len(hits) == 1 else None
        if path is None and not label and len(focus_labels) == 1:
            path = cli._path_for_label[focus_labels[0]]
        if path is None or path not in cli.texts:
            problems.append((old, label or "(unspecified file)", "not a file loaded in this session"))
            files.append(label)
            continue
        files.append(cli._label_for_path[path])
    return {**fix, "files": files}, problems


def simulate(cli, fix):
    """Apply `fix` in memory exactly as `_apply_code_fix` will (exact match once, fuzzy
    fallback), lint every result, and report what would not go through.

    Returns (changes, problems): changes = {path: (before, after, is_new)}; problems is a list
    of (snippet, label, reason) in the same shape `_apply_code_fix` reports skipped snippets."""
    after, problems = {}, []
    for old, new, label in zip(fix["old"], fix["new"], fix["files"]):
        path = cli._path_for_label.get(label)
        if path is None or path not in cli.texts:
            problems.append((old, label, "not a file loaded in this session"))
            continue
        content = after.get(path, cli.texts[path])
        count = content.count(old)
        if count == 1:
            content = content.replace(old, new, 1)
        elif count == 0:
            span = _find_fuzzy_snippet_span(content, old)
            if span is None:
                problems.append((old, label, "no exact match found in the file"))
                continue
            content = content[:span[0]] + new + content[span[1]:]
        else:
            problems.append((old, label, f"matched {count} times (ambiguous), skipped for safety"))
            continue
        after[path] = content

    changes = {}
    for path, content in after.items():
        ok, messages = cli._lint_check(content, path)
        if not ok:
            problems.append(("(lint)", cli._label_for_path.get(path, path), "; ".join(messages)))
            continue
        changes[path] = (cli.texts[path], content, False)

    for spec in fix.get("create") or []:
        target = os.path.abspath(os.path.join(cli._project_root, spec["path"]))
        shown = spec["path"]
        if not _inside(cli._project_root, target) or any(p in _PROTECTED_PARTS for p in os.path.relpath(target, cli._project_root).split(os.sep)):
            problems.append((f"create {shown}", shown, "outside the project or in a protected directory -- refused"))
        elif os.path.exists(target):
            problems.append((f"create {shown}", shown, "already exists -- edit it with old/new snippets instead"))
        else:
            content = spec["content"] if (not spec["content"] or spec["content"].endswith("\n")) else spec["content"] + "\n"
            ok, messages = cli._lint_check(content, target)
            if not ok:
                problems.append((f"create {shown}", shown, "; ".join(messages)))
            else:
                changes[target] = ("", content, True)
    return changes, problems


# ---------------------------------------------------------------------------------------
# Showing a change
# ---------------------------------------------------------------------------------------

def _paint(line, kind):
    if not _ui.color_enabled():
        return line
    code = {"add": "\033[32m", "del": "\033[31m", "hunk": "\033[2m", "file": "\033[1m"}[kind]
    return f"{code}{line}\033[0m"


def render_diff(cli, changes):
    out = []
    for path, (before, after, is_new) in changes.items():
        label = cli._label_for_path.get(path) or os.path.relpath(path, cli._project_root).replace(os.sep, "/")
        out.append(_paint(f"{'new file' if is_new else 'modified'}: {label}", "file"))
        diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=2))[2:]
        for line in diff:
            kind = "add" if line.startswith("+") else "del" if line.startswith("-") else "hunk" if line.startswith("@@") else None
            out.append(_paint(line, kind) if kind else line)
        out.append("")
    if len(out) > _DIFF_MAX_LINES:
        extra = len(out) - _DIFF_MAX_LINES
        out = out[:_DIFF_MAX_LINES] + [f"... {extra} more diff line(s) not shown"]
    return "\n".join(out)


def describe_changes(cli, fix, changes):
    lines = []
    for old, new, label in zip(fix["old"], fix["new"], fix["files"]):
        lines.append(f"[{label}] replace:\n{old}\nwith:\n{new}")
    for spec in fix.get("create") or []:
        lines.append(f"[new file {spec['path']}]\n{spec['content']}")
    return "\n\n".join(lines)[:12000]


# ---------------------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------------------

_NO_CHANGES_RE = re.compile(r"(?m)^\s*NO_CHANGES\s*$")


def _tool_rounds(cli, answer, instruction):
    """If `answer` asked for tools, run them and re-ask -- a bounded number of times.
    Returns the final answer (which may still contain nothing further to service)."""
    for _round in range(_MAX_TOOL_ROUNDS):
        notes = cli._service_tool_requests(answer)
        if not notes:
            return answer
        print(f"\n[tool results]\n{notes}\n")
        cli.agent_history.append({"role": "assistant", "content": answer})
        cli.agent_history.append({"role": "user", "content": notes})
        with _Spinner("Reading"):
            answer = cli._call_model(instruction, max_tokens=_AGENT_MAX_TOKENS)
    return answer


def _plain_text(answer):
    """The model's answer with tool directive lines removed, for showing to a person."""
    cleaned, _ = PulseCLI._extract_new_directives(answer)
    cleaned = re.sub(r"(?m)^\s*(GREP|VIEW|DEFOF|CALLERS|DEPGRAPH|DOCLOOKUP|CHANGELOG):.*$", "", cleaned)
    return _NO_CHANGES_RE.sub("", cleaned).strip()


def run_turn(cli, request):
    """One request, start to finish. Returns "applied", "answered", "declined" or "failed"."""
    cli.reload()
    cli._last_applied_fix = None
    cli._fix_applied_this_turn = False
    context = build_context(cli)
    marker = {"role": "user", "content": f"{context}\n\nRequest: {request}"}
    cli.agent_history.append(marker)
    outcome, summary = "failed", ""
    try:
        outcome, summary = _run_turn_passes(cli, request)
    except AgentRequestFailed as exc:
        cprint(f"[Pulse Code] ⚠ The agent request failed: {exc}", color=_RED)
    except Exception as exc:              # a bug in one turn must not end the session
        cprint(f"[Pulse Code] ⚠ Unexpected error ({type(exc).__name__}: {exc}); nothing further was changed.", color=_RED)
    finally:
        # The bulky file context is only needed while the passes run; keeping it in history would
        # resend it -- once per turn -- for the rest of the session.
        marker["content"] = f"Request: {request}"
        if summary:
            cli.agent_history.append({"role": "assistant", "content": summary})
    try:
        cli._sync_agent_turn(request, summary or "(no answer)", traceback_signature=None,
                             fix_applied=cli._last_applied_fix)
    except Exception:
        pass
    return outcome


def _run_turn_passes(cli, request):
    with _Spinner("Planning"):
        answer = cli._call_model(_PLAN, max_tokens=_AGENT_MAX_TOKENS)
    answer = _tool_rounds(cli, answer, _PLAN)
    plan = _plain_text(answer)
    if _NO_CHANGES_RE.search(answer) or not plan:
        print(f"\n{plan}\n" if plan else "\n(no answer)\n")
        return "answered", plan
    print(f"\n{plan}\n")

    implement = _IMPLEMENT.format(plan=plan)
    with _Spinner("Implementing"):
        raw = cli._call_model(implement, max_tokens=_AGENT_MAX_TOKENS)
    fix = parse_change(cli, raw)
    for _round in range(_MAX_TOOL_ROUNDS):
        if fix is not None:
            break
        notes = cli._service_tool_requests(raw)
        cli.agent_history.append({"role": "assistant", "content": raw})
        if notes:
            print(f"\n[tool results]\n{notes}\n")
            cli.agent_history.append({"role": "user", "content": notes})
        else:
            cli.agent_history.append({"role": "user", "content": _NO_TOOLS_NOTE})
        with _Spinner("Implementing"):
            raw = cli._call_model(implement, max_tokens=_AGENT_MAX_TOKENS)
        fix = parse_change(cli, raw)
    if fix is None:
        cprint("[Pulse Code] ⚠ Could not turn that into a concrete change; nothing was edited.", color=_RED)
        text = _plain_text(raw)
        if text:
            print(f"\n{text}\n")
        return "failed", plan

    changes, fix = _settle(cli, request, plan, fix, implement)
    if fix is None:
        return "failed", plan

    with _Spinner("Verifying"):
        verdict = cli._parse_json_obj(cli._call_model(
            _VERIFY.format(request=request, changes=describe_changes(cli, fix, changes)),
            max_tokens=_AGENT_MAX_TOKENS))
    if verdict is not None and verdict.get("passes") is False:
        reason = str(verdict.get("reason", "")).strip() or "the check did not pass"
        revised = _revise(cli, implement, f"Verification found: {reason}")
        if revised is not None:
            changes2, fix2 = _settle(cli, request, plan, revised, implement)
            if fix2 is not None:
                changes, fix = changes2, fix2

    outcome, summary = _confirm_and_apply(cli, request, plan, fix, changes)
    if outcome != "applied":
        return outcome, summary

    for _cycle in range(_MAX_VERIFY_FIX_CYCLES):
        verified, verify_note = _run_verification_pass(cli, fix, changes)
        summary = f"{summary}\n\n{verify_note}" if verify_note else summary
        if verified is not False:
            # True, or unresolved after using up its tool rounds -- either way there is
            # nothing further to automatically retry; unresolved is reported, not silently
            # treated as success.
            break
        cprint(f"[Pulse Code] Verification failed -- attempting a fix "
               f"({_cycle + 1}/{_MAX_VERIFY_FIX_CYCLES})", color=_YELLOW)
        revised = _revise(cli, _IMPLEMENT.format(plan=plan), f"Verification found: {verify_note or 'verification failed'}")
        if revised is None:
            cprint("[Pulse Code] ⚠ Could not produce a fix for the verification failure; stopping here.", color=_RED)
            break
        new_changes, new_fix = _settle(cli, request, plan, revised, _IMPLEMENT.format(plan=plan))
        if new_fix is None:
            break
        outcome, apply_summary = _confirm_and_apply(cli, request, plan, new_fix, new_changes)
        summary = f"{summary}\n\n{apply_summary}"
        if outcome != "applied":
            return outcome, summary
        fix, changes = new_fix, new_changes

    return outcome, summary


def _run_verification_pass(cli, fix, changes):
    """STEP 4. After a change is applied, give the agent one more turn -- bounded by
    _MAX_VERIFY_TOOL_ROUNDS -- to actually check it with TERMINAL:/GREP:/VIEW: instead of
    just asserting it works. Returns (verified, note): verified is True/False/None
    (None = the model never produced a clear verdict after using its tool rounds -- treated
    as "not proven to have failed", but reported to the user either way, not silently
    swallowed)."""
    explanation = fix.get("explanation") or "(no explanation given)"
    diff = render_diff(cli, changes)
    prompt = _VERIFY_TERMINAL.format(explanation=explanation, diff=diff)
    cli._in_verification_pass = True   # tags any TERMINAL: run in this pass -- see PulseCLI._run_terminal
    try:
        with _Spinner("Verifying"):
            answer = cli._call_model(prompt, max_tokens=_AGENT_MAX_TOKENS)

        for _round in range(_MAX_VERIFY_TOOL_ROUNDS):
            verdict = cli._parse_json_obj(answer)
            if verdict is not None and "verified" in verdict:
                passed = bool(verdict.get("verified"))
                reason = str(verdict.get("reason", "")).strip()
                icon = "✓" if passed else "✗"
                color = _GREEN if passed else _RED
                cprint(f"[4] Verification {icon}{(': ' + reason) if reason else ''}", color=color)
                return passed, reason
            notes = cli._service_tool_requests(answer)
            if not notes:
                # Neither a verdict nor a tool request -- nudge once more rather than looping
                # forever on an unparsable reply.
                cli.agent_history.append({"role": "assistant", "content": answer})
                cli.agent_history.append({"role": "user", "content": _VERIFY_NO_TOOLS_NOTE})
                with _Spinner("Verifying"):
                    answer = cli._call_model(prompt, max_tokens=_AGENT_MAX_TOKENS)
                continue
            print(f"\n[verify tool results]\n{notes}\n")
            cli.agent_history.append({"role": "assistant", "content": answer})
            cli.agent_history.append({"role": "user", "content": notes})
            with _Spinner("Verifying"):
                answer = cli._call_model(prompt, max_tokens=_AGENT_MAX_TOKENS)

        cprint("[4] Verification: no clear pass/fail after checking -- treating the change as applied "
               "but unverified.", color=_YELLOW)
        return None, "Verification inconclusive after available checks."
    finally:
        cli._in_verification_pass = False


def _revise(cli, implement, problem):
    with _Spinner("Revising"):
        raw = cli._call_model(_REVISE.format(problem=problem), max_tokens=_AGENT_MAX_TOKENS)
    return parse_change(cli, raw)


def _settle(cli, request, plan, fix, implement):
    """Get `fix` to the point where every part of it will go through, or give up. Returns
    (changes, fix) or (None, None) after saying why. All-or-nothing starts here: a change
    with any unresolved part is never applied."""
    for attempt in range(_MAX_REVISE_ROUNDS + 1):
        fix, target_problems = normalise_targets(cli, fix)
        changes, problems = simulate(cli, fix)
        problems = target_problems + [p for p in problems if p not in target_problems]
        if not problems:
            return changes, fix
        if attempt == _MAX_REVISE_ROUNDS:
            break
        cprint("[Pulse Code] Part of that change would not apply cleanly -- asking for a correction "
               f"({attempt + 1}/{_MAX_REVISE_ROUNDS})", color=_YELLOW)
        no_match = [p for p in problems if "no exact match" in p[2]]
        others = [p for p in problems if "no exact match" not in p[2]]
        revised = None
        if no_match and not others:
            corrected = cli._request_corrected_snippets(fix, no_match)
            if corrected:
                matched = [(o, n, f) for o, n, f in zip(fix["old"], fix["new"], fix["files"])
                           if all(o != p[0] or f != p[1] for p in no_match)]
                # The model is asked only for the snippets that failed, but may resend the whole
                # change. An edit that already matched must not be applied a second time.
                seen = {(o, f) for o, _n, f in matched}
                merged = matched + [(o, n, f) for o, n, f in zip(corrected["old"], corrected["new"], corrected["files"])
                                    if (o, f) not in seen]
                revised = {**fix, "old": [m[0] for m in merged], "new": [m[1] for m in merged],
                           "files": [m[2] for m in merged]}
        if revised is None:
            text = "\n".join(f"- [{label}] {reason}: {str(snippet).splitlines()[0][:100] if str(snippet).strip() else ''}"
                             for snippet, label, reason in problems)
            revised = _revise(cli, implement, text)
        if revised is None:
            break
        fix = revised
    cprint("[Pulse Code] ⚠ Could not produce a change that applies cleanly, so nothing was edited:", color=_RED)
    for snippet, label, reason in problems:
        first = str(snippet).splitlines()[0][:90] if str(snippet).strip() else ""
        print(f"  - [{label}] {reason}{': ' + first if first else ''}")
    return None, None


def _confirm_and_apply(cli, request, plan, fix, changes):
    if not changes:
        cprint("[Pulse Code] The change turned out to be empty; nothing was edited.", color=_YELLOW)
        return "answered", plan
    print("\n" + render_diff(cli, changes))
    n_files = len(changes)
    if cli.review:
        try:
            answer = _prompt_text(
                f"Apply these changes to {n_files} file{'s' if n_files != 1 else ''}? (Y/n) > ",
                label=f"Apply these changes to {n_files} file{'s' if n_files != 1 else ''}?  (Y/n)").strip().lower()
        except EOFError:
            answer = "n"
            cprint("[Pulse Code] No terminal to confirm on -- not applied (use -y to apply without asking).", color=_YELLOW)
        if answer not in ("", "y", "yes"):
            cprint("[Pulse Code] Not applied.", color=_YELLOW)
            return "declined", f"{plan}\n\n(The user declined the proposed change.)"

    if not fix.get("explanation"):
        fix["explanation"] = f"Pulse Code: {request[:80]}"
    cli._last_apply_skipped = []
    cli._apply_code_fix(fix)
    applied_ok = (cli._fix_applied_this_turn and not cli._last_apply_skipped
                  and not getattr(cli, "_last_apply_lint_failed", []))
    if not applied_ok:
        # Something differed between the preview and the write (a file changed underneath us).
        # Never leave half a feature behind.
        if cli._fix_applied_this_turn and getattr(cli, "_last_commit_id", None):
            cprint("[Pulse Code] ⚠ Not every part was applied -- undoing the ones that were.", color=_RED)
            undo(cli, f"{cli._last_commit_id} force", quiet=True)
        else:
            cprint("[Pulse Code] ⚠ Nothing was applied.", color=_RED)
        cli.reload()
        return "failed", plan

    created = [os.path.abspath(os.path.join(cli._project_root, s["path"])) for s in fix.get("create") or []]
    cli.add_files([p for p in created if os.path.isfile(p)], focus=False)
    cli.reload()
    commit = getattr(cli, "_last_commit_id", None)
    cprint(f"\n✓ {n_files} file{'s' if n_files != 1 else ''} changed" + (f"  ·  /undo to revert (commit {commit})" if commit else ""),
           color=_GREEN)
    return "applied", f"{plan}\n\nApplied: {fix.get('explanation', '')}"


# ---------------------------------------------------------------------------------------
# /undo
# ---------------------------------------------------------------------------------------

def undo(cli, arg="", quiet=False):
    """Revert one Pulse Code commit -- the latest, or the id given -- touching only files that
    are still exactly as that commit left them. If any file has been edited since, nothing is
    undone unless `force` is given. Returns True if it reverted anything."""
    words = arg.split()
    force = "force" in words
    ids = [w for w in words if w != "force"]
    entries = cli._load_fix_log()
    undone = set()
    for entry in entries:
        match = re.match(r"Undid commit (\w+)", entry.get("explanation", "")) if entry.get("kind") == "revert" else None
        if match:
            undone.add(match.group(1))
    candidates = [e for e in entries if e.get("kind") == "fix" and e.get("id") not in undone]
    target = None
    if ids:
        target = next((e for e in candidates if e.get("id", "").startswith(ids[0])), None)
    elif candidates:
        target = candidates[-1]
    if target is None:
        if not quiet:
            cprint("[Pulse Code] Nothing to undo." if not ids else f"[Pulse Code] No un-undone commit matches '{ids[0]}'. /log lists them.")
        return False

    plan, conflicts = [], []
    for entry in target.get("files", []):
        path = entry["path"]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                current = handle.read()
        except OSError:
            current = None
        if entry.get("created"):
            if current is None:
                continue
            if current != entry["after"] and not force:
                conflicts.append(path)
                continue
            plan.append((path, current, None))
        else:
            if current == entry["before"]:
                continue
            if current != entry["after"] and not force:
                conflicts.append(path)
                continue
            plan.append((path, current, entry["before"]))
    if conflicts:
        cprint(f"[Pulse Code] Not undone -- changed since the agent edited them, so undoing would overwrite your work:", color=_YELLOW)
        for path in conflicts:
            print(f"    - {os.path.relpath(path, cli._project_root)}")
        print("  /undo force restores them anyway, or edit them back by hand.")
        return False

    done = []
    revert_files = {}
    for path, current, restored in plan:
        try:
            if restored is None:
                os.remove(path)
            else:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(restored)
        except OSError as exc:
            cprint(f"[Pulse Code] ⚠ Could not restore {os.path.basename(path)}: {exc}", color=_RED)
            continue
        revert_files[path] = (current or "", restored or "")
        done.append(path)
    if revert_files:
        cli._record_fix_commit(revert_files, f"Undid commit {target.get('id')}", kind="revert")
    for path in done:
        if path in cli.known and not os.path.exists(path):
            cli.known.remove(path)
    cli.reload()
    if not quiet:
        if done:
            cprint(f"[Pulse Code] ✓ Undid commit {target.get('id')} -- {len(done)} file(s):", color=_BLUE)
            for path in done:
                print(f"    - {os.path.relpath(path, cli._project_root)}")
        else:
            cprint("[Pulse Code] Already undone -- the files match the state before that commit.")
    return bool(done)


# ---------------------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------------------

_HELP = [
    ("Ask", [
        ("<request>", "just type it: 'add a --resume flag to train.py', 'why does loader.py drop the last batch?'"),
        ("/review on|off", "show a diff and ask before applying (default on)"),
        ("/undo [id]", "undo the latest change (or a given one); leaves files you've edited since alone"),
    ]),
    ("Files", [
        ("/files", "what is in focus (shown to the agent in full) and how many files it can search"),
        ("/add <path…>", "put files or folders in focus (globs work)"),
        ("/drop <path>", "take a file out of focus"),
    ]),
    ("Agent & account", [
        ("/agent", "switch AI provider/model"),
        ("/log", "the change history (.pulse_history)"),
        ("/cloud, /cloud flush", "sign-in, workspace and sync status"),
        ("/commit, /repo [url]", "the git commit / repo this session is associated with"),
        ("/password, /recover, /logout", "account"),
        ("/exit", "leave"),
    ]),
]


def _print_help():
    print()
    for title, rows in _HELP:
        print(f"  {title}")
        for cmd, desc in rows:
            print(f"    {cmd:<32} {desc}")
        print()


def _print_files(cli):
    print(f"\n  Project  {cli._project_root}")
    print(f"  Focus    {len(cli.focus)} file(s) shown to the agent in full")
    for path in cli.focus:
        print(f"    * {cli._label_for_path.get(path, path)}  ({len(cli.texts.get(path, '').splitlines())} lines)")
    others = len(cli.known) - len(cli.focus)
    print(f"  Also searchable: {others} more file(s) (GREP / VIEW / DEFOF)\n")


def _resolve_arg(cli, arg):
    files, problems = expand_paths(cli._project_root, [arg])
    for problem in problems:
        cprint(f"[Pulse Code] {problem}", color=_YELLOW)
    return [f for f in files if _inside(cli._project_root, f)]


_FORWARDED = {
    "/password": "_cmd_password", "/recover": "_cmd_recover", "/deleteaccount": "_cmd_deleteaccount",
    "/admin": "_cmd_admin", "/webhook": "_cmd_webhook", "/telemetry": "_cmd_telemetry",
    "/repo": "_cmd_repo", "/commit": "_cmd_commit",
}


def _handle_command(cli, line):
    """Slash commands. Anything that is the debugger's (account, cloud, agent, history) calls
    the very same method. Returns False to leave the session."""
    lowered = line.lower()
    word, _, rest = line.partition(" ")
    word = word.lower()
    if lowered in ("/exit", "/quit", "/q"):
        return False
    if lowered in ("/help", "/h", "?"):
        _print_help()
    elif word == "/files":
        _print_files(cli)
    elif word == "/add":
        found = _resolve_arg(cli, rest.strip()) if rest.strip() else []
        if found:
            cli.add_files(found)
            cprint(f"[Pulse Code] ✓ {len(found)} file(s) in focus.")
        elif not rest.strip():
            print("  usage: /add <path or folder or glob>")
    elif word == "/drop":
        target = _resolve_arg(cli, rest.strip()) if rest.strip() else []
        for path in target:
            cli.drop_focus(path)
        if target:
            cprint(f"[Pulse Code] ✓ dropped {len(target)} file(s) from focus (still searchable).")
    elif word == "/review":
        arg = rest.strip().lower()
        cli.review = True if arg in ("on", "true", "1") else False if arg in ("off", "false", "0") else not cli.review
        cprint(f"[Pulse Code] Review before applying is {'ON' if cli.review else 'OFF'}.")
    elif word in ("/undo", "/revert"):
        undo(cli, rest)
    elif word == "/log":
        cli._cmd_log("")
    elif lowered == "/agent":
        cli._select_agent_provider_and_key(initial=False)
    elif lowered == "/cloud":
        cli._print_cloud_status()
    elif lowered == "/cloud flush":
        cli._maybe_flush_cloud(force=True)
        cprint("[Pulse] Cloud sync flushed.")
    elif lowered == "/logout":
        cli._cmd_logout("")
    elif word in _FORWARDED:
        getattr(cli, _FORWARDED[word])(rest.strip())
    else:
        cprint(f"[Pulse Code] Unknown command {word}. /help lists them.", color=_YELLOW)
    return True


def _read_request(prompt):
    """One request: a line, or several joined by a trailing backslash."""
    parts = []
    while True:
        line = input(prompt if not parts else "  … ")
        if line.rstrip().endswith("\\"):
            parts.append(line.rstrip()[:-1])
            continue
        parts.append(line)
        return "\n".join(parts).strip()


def session(cli):
    prompt = _ui._s("pulse code", "bold", "accent") + _ui._s(" ❯ " if _ui._unicode() else " > ", "dim")
    while True:
        try:
            line = _read_request(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/") or line == "?":
            try:
                if not _handle_command(cli, line):
                    return 0
            except KeyboardInterrupt:
                print()
            continue
        if not cli.agent_provider or not cli.agent_key:
            cprint("[Pulse Code] No agent is set up. /agent picks a model and key.", color=_YELLOW)
            continue
        try:
            run_turn(cli, line)
        except KeyboardInterrupt:
            cprint("\n[Pulse Code] Cancelled.", color=_YELLOW)


def run(paths=(), prompt=None, yes=False, root=None):
    """`pulse code`. Returns the process exit status."""
    root = os.path.abspath(root or os.getcwd())
    focus, problems = expand_paths(root, list(paths))
    for problem in problems:
        cprint(f"[Pulse Code] {problem}", color=_YELLOW)
    if paths and not focus:
        cprint("[Pulse Code] None of those paths could be used.", color=_RED)
        return 1
    outside = [f for f in focus if not _inside(root, f)]
    if outside:
        cprint(f"[Pulse Code] {os.path.relpath(outside[0], root)} is outside the project root {root}; "
               "run `pulse code` from a directory that contains it.", color=_RED)
        return 1
    known = scan_project(root)

    cli = _CodeAgentCLI()
    cli.setup_code(root, focus, known)
    cli.review = not yes
    os.chdir(root)
    cli.set_code_text(cli.code_text, script_path=cli.script_path)
    cli._project_root = root

    cli.print_banner()
    try:
        cli.interactive_setup()                       # the debugger's setup, unchanged
    except (EOFError, KeyboardInterrupt):
        print("\n[Pulse Code] Setup cancelled.")
        return 1
    signal.signal(signal.SIGINT, signal.default_int_handler)   # the debugger's handler means "pause the run"
    cli.reload()

    _ui.note(f"{root}  ·  {len(cli.known)} files searchable  ·  {len(cli.focus)} in focus")
    try:
        if prompt is not None:
            if not cli.agent_provider or not cli.agent_key:
                cprint("[Pulse Code] No agent is set up, so there is nothing to answer with.", color=_RED)
                return 1
            outcome = run_turn(cli, prompt)
            return {"applied": 0, "answered": 0, "declined": 2, "failed": 1}.get(outcome, 1)
        return session(cli)
    finally:
        try:
            cli._flush_cloud_now()
        except Exception:
            pass
