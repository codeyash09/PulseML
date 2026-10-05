"""
The Pulse app's screen: a transcript, an input line and, while debugging, the run beside it.

    PULSE / DEBUG   train.py                                   agent: DeepSeek V4 Flash
    ──────────────────────────────────────────────────────────────────────────────────
     train.py                 ● live │ ❯ why did the loss stop moving?
     ~/experiments                   │
     step 4,120 · 3.1/s              │ ✻ Thinking  (ctrl+o to expand)
                                     │ ● GREP: lr_scheduler
     loss      0.0431 ▇▆▅▄▃▂▁        │ ● VIEW: train.py:40-80
     val_loss  0.0512 ▇▆▅▅▄▃▃        │   ⎿ 58 lines  (ctrl+o to expand)
                                     │ The scheduler steps every batch, so by step 900 ...
     Findings                        │ ──────────────────────────────────────────────
     ▲ WARNING loss plateau          │ ❯ ▏
                                     │ Enter send · Ctrl+O details · PgUp/PgDn scroll

Everything in this module is presentation, and almost all of it is pure: `compose()` turns
a `View` into the exact lines of one frame, `Editor` turns keys into text, `wrap()` folds
ANSI-coloured text. That is what the tests exercise. The two impure pieces are small:
`Screen` (alternate screen, diffed redraw) and `KeyReader` (cbreak mode, key names).

stdlib only, like pulse_ui, and the same visual language: one accent (Pulse orange), dim
for what is secondary, green / amber / red for success / warning / failure.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from . import pulse_ui as _ui

__all__ = ["Entry", "View", "Editor", "Screen", "KeyReader", "compose", "wrap", "clean",
           "entry_lines", "visible_len", "pad"]

RESET = "\033[0m"
_SGR_RE = re.compile(r"\033\[[0-9;]*m")
# Every other escape sequence a captured program may have written: cursor movement, erase,
# mode switches (CSI), window titles (OSC), and two-byte escapes. They are meaningless
# inside a pane -- and dangerous: they would move the real cursor out of it.
_OTHER_ESC_RE = re.compile(r"\033(?:\[[0-9;?<=>]*[ -/]*[@-ln-~]|\][^\a\033]*(?:\a|\033\\)?|[@-Z\\^_]|$)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]")


def s(text: str, *styles: str) -> str:
    return _ui._s(text, *styles)


def g(name: str) -> str:
    return _ui._g(name)


def visible_len(text: str) -> int:
    return _ui._visible_len(text)


def clean(text: str) -> str:
    """Captured output made safe for a pane: colours kept, every other escape and control
    character dropped, tabs expanded. Newlines and carriage returns are the caller's."""
    text = _OTHER_ESC_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return text.replace("\t", "    ")


def _tokens(text: str) -> List[str]:
    """Colour sequences and single characters, in order."""
    out: List[str] = []
    pos = 0
    for match in _SGR_RE.finditer(text):
        out.extend(text[pos:match.start()])
        out.append(match.group(0))
        pos = match.end()
    out.extend(text[pos:])
    return out


def _is_sgr(token: str) -> bool:
    return len(token) > 1 and token.startswith("\033[")


def _is_reset(token: str) -> bool:
    return token in ("\033[0m", "\033[m")


def wrap(text: str, width: int, indent: str = "") -> List[str]:
    """Fold one line to `width` columns at word boundaries. Colours are carried over a
    fold (and closed at the end of every piece), so a wrapped coloured line never bleeds
    into the pane beside it. `indent` prefixes continuation lines."""
    width = max(1, width)
    if "\033" not in text and visible_len(text) <= width:
        return [text]
    lines: List[str] = []
    active: List[str] = []          # colours in force at the cursor
    line: List[str] = []            # tokens of the line being built
    col = 0
    break_at: Optional[int] = None  # index in `line` of the last space
    break_col = 0
    indent_w = visible_len(indent)

    def flush(tokens: List[str]) -> None:
        body = "".join(tokens).rstrip(" ")
        lines.append(body + (RESET if "\033[" in body and not body.endswith(RESET) else ""))

    def start(carry: List[str]) -> Tuple[List[str], int]:
        """A continuation line: the indent, the colours in force, then `carry`."""
        fresh = ([indent] if indent else []) + list(active_before_carry) + carry
        return fresh, indent_w + sum(_ui._char_width(t) for t in carry if not _is_sgr(t))

    for token in _tokens(text):
        if _is_sgr(token):
            line.append(token)
            if _is_reset(token):
                active = []
            else:
                active.append(token)
            continue
        w = _ui._char_width(token)
        if col + w > width and col > 0:
            if break_at is not None and token != " ":
                carry = line[break_at + 1:]
                # the colours in force where the carried text begins
                active_before_carry = []
                for t in line[:break_at + 1]:
                    if _is_sgr(t):
                        active_before_carry = [] if _is_reset(t) else active_before_carry + [t]
                flush(line[:break_at])
                line, col = start(carry)
            else:
                active_before_carry = list(active)
                flush(line)
                line, col = start([])
                if token == " ":
                    break_at = None
                    continue
            break_at = None
        if token == " ":
            break_at, break_col = len(line), col
        line.append(token)
        col += w
    flush(line)
    return lines


def cut(text: str, width: int) -> str:
    """At most `width` columns, no ellipsis: what a frame row can be written as."""
    if visible_len(text) <= width:
        return text
    out: List[str] = []
    cols = 0
    for token in _tokens(text):
        if _is_sgr(token):
            out.append(token)
            continue
        w = _ui._char_width(token)
        if cols + w > width:
            break
        out.append(token)
        cols += w
    return "".join(out)


def pad(text: str, width: int) -> str:
    """Exactly `width` columns: clipped or space-filled, colours closed."""
    clipped = _ui._clip(text, width)
    gap = width - visible_len(clipped)
    closed = clipped + (RESET if "\033[" in clipped and not clipped.endswith(RESET) else "")
    return closed + (" " * gap if gap > 0 else "")


# ---------------------------------------------------------------------------------------
# The transcript
# ---------------------------------------------------------------------------------------

class Entry:
    """One thing in the transcript.

    kind      what it is                         text / calls / output
    --------  ---------------------------------  ------------------------------------
    user      what the person typed              text
    text      output of the agent or a command   text (may be several lines, coloured)
    thinking  the model's reasoning              text -- collapsed to one line by default
    tool      tools the agent ran                calls (one line each), output (collapsed)
    edit      a change to the project's files    calls (EDIT:/CREATE: label), output (the
                                                 diff, collapsed), text (its status)
    finding   a detector finding                 text, severity
    say       what the agent says while working   text (MESSAGE: / message_user)
    note      a quiet line from Pulse itself     text
    error     something failed                   text
    """
    __slots__ = ("kind", "text", "calls", "output", "severity", "live", "open", "seconds", "_cache")

    def __init__(self, kind: str, text: str = "", calls: Optional[List[str]] = None,
                 output: str = "", severity: str = "", live: bool = False) -> None:
        self.kind = kind
        self.text = text
        self.calls = list(calls or [])
        self.output = output
        self.severity = severity
        self.live = live                # still being streamed in: shown as it grows
        self.open: Optional[bool] = None   # clicked open (True) or shut (False); None = as the screen says
        self.seconds: Optional[float] = None   # how long the thinking took, when it streamed
        self._cache: Dict[Tuple[int, bool], List[str]] = {}

    def touch(self) -> None:
        self._cache = {}

    def foldable(self) -> bool:
        return self.kind in ("thinking", "tool", "edit") or (self.kind == "user" and self.text.count("\n") >= USER_FOLD_LINES)

    def toggle(self, expanded: bool = False) -> None:
        """A click: open what is shut, shut what is open (`expanded` is the screen-wide state)."""
        self.open = not self.is_open(expanded)
        self.touch()

    def is_open(self, expanded: bool) -> bool:
        return expanded if self.open is None else self.open


# One colour: Pulse's orange is for what is live, selected or worth a look; red is for what
# is wrong; everything else is a shade of grey. Nothing green or yellow. Three things are
# told apart by a box behind them: a command the agent ran (orange box), the model's
# thinking (grey box), and Pulse's own output, which is plain grey text.
_SEVERITY_STYLE = {"critical": "red", "error": "red", "warning": "accent", "info": "dim"}
_RED_SGR_RE = re.compile(r"\033\[(?:[0-9;]*;)?(?:31|91)(?:;[0-9;]*)?m")
_QUIET_KINDS = {"thinking", "tool", "edit", "note"}
_QUIET_TAG_RE = re.compile(r"\bago\b|^(?:not |ended|crashed|finished|stopped|done)")
EXPAND_HINT = "click or ctrl+o"
USER_FOLD_LINES = 6                 # a pasted request longer than this folds to its first lines


def _count(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


_CALL_WORDS = {
    "GREP": "Searched for {}", "VIEW": "Read {}", "READ_FILE": "Read {}", "OUTLINE": "Outlined {}",
    "LIST_FILES": "Listed the files {}", "FIND_DEFINITION": "Found where {} is defined", "DEFOF": "Found where {} is defined",
    "FIND_REFERENCES": "Found the uses of {}", "CALLERS": "Found the callers of {}", "DEP_GRAPH": "Mapped the imports",
    "DEPGRAPH": "Mapped the imports", "TRACE_VARIABLE": "Traced {}", "TRACE": "Traced {}",
    "DOC_LOOKUP": "Looked up {}", "DOCLOOKUP": "Looked up {}", "CHANGELOG": "Checked what changed",
    "TERMINAL": "Ran {}", "RUN_COMMAND": "Ran {}", "EDIT_FILE": "Edited {}", "EDIT": "Edited {}",
    "CREATE": "Created {}",
    "REPLACE_SYMBOL": "Rewrote {}", "WRITE_FILE": "Wrote {}", "TODO_WRITE": "Updated the plan ({})",
    "START_RUN": "Started {} under Pulse", "RUN": "Started {} under Pulse", "RUN_STATUS": "Checked the run",
    "RUNSTATUS": "Checked the run", "RESTART_RUN": "Restarted the run", "RESTART": "Restarted the run",
    "STOP_RUN": "Stopped the run", "STOP": "Stopped the run", "CALC": "Calculated {}", "CORR": "Correlated {}",
    "OUTLIER": "Looked for outliers in {}", "HISTOGRAM": "Plotted {}", "DIFFSTATS": "Compared {}",
    "MLLINT": "Ran the ML lint", "GPUSTATUS": "Checked the GPUs", "SHAPETRACE": "Traced the shapes",
    "GRADCHECK": "Checked the gradients of {}", "REPLAY": "Replayed {} steps", "REPL": "Evaluated {}",
    "DRYRUN": "Dry-ran {}", "PASTFIX": "Looked for past fixes ({})", "ROLLBACK": "Rolled back {}",
    "LAYERSTATS": "Checked the layers", "HARDEXAMPLES": "Looked at the hard examples",
    "AMPSTATUS": "Checked mixed precision", "SEEDCHECK": "Checked the seeds", "RANKDIVERGE": "Compared the ranks",
    "RUNCOMPARE": "Compared the runs", "COST": "Checked the cost",
    "CHECK_SHAPE": "Checked the shapes", "SMOKE_TEST": "Smoke-tested the change",
}


def describe_call(call: str) -> str:
    """A tool call as a short sentence: "Ran pytest -q", "Read train.py:1-40"."""
    name, sep, arg = call.partition(":")
    name, arg = name.strip().upper(), arg.strip()
    words = _CALL_WORDS.get(name)
    if words is None:
        return f"{name.lower().replace('_', ' ').capitalize()}{' ' + arg if arg else ''}"
    if "{}" in words:
        return words.format(arg) if arg else words.replace(" {}", "").replace("{}", "").strip()
    return words


def diff_counts(diff: str) -> Tuple[int, int]:
    """Lines added and removed in a unified diff (its ---/+++ header not counted)."""
    added = removed = 0
    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def box(text: str, kind: str) -> str:
    """`text` in its box: orange for a command ("tool"), grey for thinking ("think")."""
    return s(f" {text} ", "box_tool") if kind == "tool" else s(f" {text} ", "box_think", "italic")


def _edit_head(entry: Entry) -> str:
    said = [describe_call(c) for c in entry.calls]
    head = " · ".join(said[:3]) + (f" +{len(said) - 3}" if len(said) > 3 else "")
    added, removed = diff_counts(entry.output)
    counts = f"  · +{added} \u2212{removed}" if _ui._unicode() else f"  · +{added} -{removed}"
    status = entry.text.split("\n", 1)[0]
    return box(head, "tool") + s(counts + (f"  · {status}" if status else ""), "dim")


def entry_lines(entry: Entry, width: int, expanded: bool) -> List[str]:
    """The entry as finished screen lines, each at most `width` columns. `expanded` is the
    screen-wide Ctrl+O state; an entry that was clicked keeps its own."""
    expanded = entry.is_open(expanded)
    key = (width, expanded)
    cached = entry._cache.get(key)
    if cached is not None:
        return cached
    out: List[str] = []
    kind = entry.kind
    if kind == "user":
        mark = s(g("cursor") + " ", "accent", "bold")
        parts = entry.text.split("\n")
        folded = len(parts) > USER_FOLD_LINES and not expanded
        shown = parts[:USER_FOLD_LINES - 1] if folded else parts
        for i, part in enumerate(shown):
            for j, piece in enumerate(wrap(part, width - 2)):
                out.append((mark if i == 0 and j == 0 else "  ") + s(piece, "bold"))
        if folded:
            out.append("  " + s(f"… {len(parts) - len(shown)} more lines  ({EXPAND_HINT})", "dim"))
    elif kind == "say":
        # the agent speaking to the person: plain text, nothing in front of it
        for part in entry.text.strip("\n").split("\n"):
            out.extend(wrap(part, width))
    elif kind == "thinking":
        body = [ln for ln in entry.text.strip().split("\n")]
        if entry.live and not expanded:
            # streaming in: "Thinking…", then the last two lines as they arrive
            out.append(box("Thinking…", "think"))
            tail: List[str] = []
            for part in body:
                tail.extend(wrap(part, width - 2))
            out.extend("  " + s(piece, "dim", "italic") for piece in tail[-2:])
        elif not expanded:
            took = f"Thought for {_ui.elapsed_text(entry.seconds)}" if entry.seconds else \
                f"Thought ({_count(len(entry.text.split()), 'word')})"
            out.append(box(took, "think"))
        else:
            out.append(box("Thought" + (f" for {_ui.elapsed_text(entry.seconds)}" if entry.seconds else ""), "think"))
            for part in body:
                for piece in wrap(part, width - 2):
                    out.append("  " + s(piece, "dim", "italic"))
    elif kind == "tool":
        result = entry.output.strip("\n")
        rows = result.split("\n") if result else []
        if not expanded:
            # one line: what was done, and how much came back -- click for the rest
            said = [describe_call(c) for c in entry.calls]
            line = " · ".join(said[:3]) + (f" +{len(said) - 3}" if len(said) > 3 else "")
            tail = s(f"  · {_count(len(rows), 'line')}", "dim") if rows else ""
            out.append(_ui._clip(box(line, "tool") + tail, width))
        else:
            for call in entry.calls:
                name, sep, arg = call.partition(":")
                label = box(describe_call(call), "tool") + (s("   " + name.strip() + ": " + arg.strip(), "dim")
                                                          if sep and arg.strip() else "")
                out.extend(wrap(label, width, indent="  "))
            elbow = "⎿ " if _ui._unicode() else "L "
            first = True
            for row in rows:
                for piece in wrap(row, width - 4):
                    out.append("  " + s(elbow if first else "  ", "dim") + s(piece, "dim"))
                    first = False
    elif kind == "edit":
        # a change to the files: one line saying what changed, the diff under it when open
        out.append(_ui._clip(_edit_head(entry), width))
        if expanded:
            for row in entry.output.strip("\n").split("\n"):
                if row.startswith("+++") or row.startswith("---"):
                    continue
                style = "bold" if row.startswith(("modified: ", "new file: ")) else \
                    "accent" if row.startswith("+") else "dim"
                out.extend("  " + s(piece, style) for piece in wrap(row, width - 2))
            for detail in entry.text.split("\n")[1:]:        # the reviewer's reason, the /undo hint
                out.extend("  " + s(piece, "dim") for piece in wrap(detail, width - 2))
    elif kind == "finding":
        style = _SEVERITY_STYLE.get(entry.severity.lower(), "accent")
        mark = ("▲ " if _ui._unicode() else "! ") + (entry.severity.upper() + " " if entry.severity else "")
        pieces = wrap(entry.text, max(1, width - visible_len(mark)), indent="")
        out.append(s(mark, style, "bold") + pieces[0])
        out.extend(" " * visible_len(mark) + p for p in pieces[1:])
    elif kind == "note":
        for part in entry.text.split("\n"):
            out.extend(s(piece, "dim") for piece in wrap(part, width))
    elif kind == "error":
        for part in entry.text.split("\n"):
            pieces = wrap(part, width - 2)
            out.append(s(g("fail") + " ", "red") + s(pieces[0], "red"))
            out.extend("  " + s(p, "red") for p in pieces[1:])
    else:
        # Pulse's own output (a note from a pipeline, a command's output): quieter than the
        # agent's words, in grey; a line that came out red (an error) stays red. The colours
        # the pipelines paint for a plain terminal are dropped here.
        for part in entry.text.strip("\n").split("\n"):
            bare = _SGR_RE.sub("", part)
            lead = min(len(bare) - len(bare.lstrip(" ")), width // 2)
            style = "red" if _RED_SGR_RE.search(part) else "dim"
            out.extend(s(piece, style) for piece in wrap(bare, width, indent=" " * lead))
    entry._cache = {key: out}       # one frame size at a time is all that is ever asked for
    return out


# ---------------------------------------------------------------------------------------
# The input line
# ---------------------------------------------------------------------------------------

class Editor:
    """A one-line editor with history. `handle(key)` returns True when it used the key."""

    def __init__(self) -> None:
        self.text = ""
        self.pos = 0
        self.history: List[str] = []
        self._at: Optional[int] = None      # position while walking the history
        self._draft = ""

    def set(self, text: str) -> None:
        self.text = text
        self.pos = len(text)

    def take(self, remember: bool = True) -> str:
        """The text, and an empty line for the next thing."""
        text = self.text
        if remember and text.strip() and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self.text, self.pos, self._at, self._draft = "", 0, None, ""
        return text

    def insert(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
        text = _CONTROL_RE.sub("", text).replace("\t", " ")
        self.text = self.text[:self.pos] + text + self.text[self.pos:]
        self.pos += len(text)

    def handle(self, key: str, history: bool = True) -> bool:
        if key.startswith("paste:"):
            self.insert(key[6:])
        elif len(key) == 1 and key >= " ":
            self.insert(key)
        elif key == "backspace":
            if self.pos:
                self.text = self.text[:self.pos - 1] + self.text[self.pos:]
                self.pos -= 1
        elif key == "delete":
            self.text = self.text[:self.pos] + self.text[self.pos + 1:]
        elif key == "left":
            self.pos = max(0, self.pos - 1)
        elif key == "right":
            self.pos = min(len(self.text), self.pos + 1)
        elif key in ("home", "ctrl+a"):
            self.pos = 0
        elif key in ("end", "ctrl+e"):
            self.pos = len(self.text)
        elif key == "ctrl+u":
            self.text, self.pos = self.text[self.pos:], 0
        elif key == "ctrl+k":
            self.text = self.text[:self.pos]
        elif key == "ctrl+w":
            head = self.text[:self.pos].rstrip(" ")
            cut = head.rfind(" ") + 1
            self.text, self.pos = self.text[:cut] + self.text[self.pos:], cut
        elif key == "up" and history and self.history:
            if self._at is None:
                self._draft, self._at = self.text, len(self.history)
            self._at = max(0, self._at - 1)
            self.set(self.history[self._at])
        elif key == "down" and history and self._at is not None:
            self._at += 1
            if self._at >= len(self.history):
                self._at = None
                self.set(self._draft)
            else:
                self.set(self.history[self._at])
        else:
            return False
        return True


# ---------------------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------------------

class View:
    """Everything a frame shows. The app mutates this; compose() only reads it."""

    def __init__(self) -> None:
        self.area = ""                          # PULSE / <AREA> in the top bar; "" at home
        self.context = ""                       # project folder, or the run
        self.agent = ""                         # the model in use, or how to set one
        self.entries: List[Entry] = []
        self.expanded = False                   # ctrl+o: thinking and tool output in full
        self.scroll = 0                         # lines up from the newest
        self.only_last = True                   # unscrolled: just the latest exchange, not the whole log
        self.live = ""                          # what is running right now ("Planning")
        self.live_since = 0.0
        self.partial = ""                       # an unfinished output line (no newline yet)
        self.side: Optional[List[str]] = None   # the run pane's lines; None = no split
        self.side_brief: List[str] = []         # the same run in two lines, for a narrow terminal
        self.editor = Editor()
        self.busy = False
        # a question from a pipeline running inside the app
        self.question = ""                      # shown above the field
        self.secret = False
        self.options: Optional[List[Tuple[str, str, str]]] = None   # (label, detail, tag)
        self.option_index = 0
        self.option_ids: List[int] = []         # indexes into the caller's list, after filtering
        self.status = ""                        # one transient line in the footer
        self.commands: List[Tuple[str, str]] = []   # (command, what it does), for hints
        self.frame = 0
        self.row_entries: Dict[int, Entry] = {}    # screen row -> the entry drawn there (for clicks)
        self.transcript_x = 1                      # column the transcript starts at
        self.first_row = 2                         # screen row of the transcript's first line
        self.room = 0                              # transcript rows on screen
        self.pane_w = 0                            # columns the transcript was wrapped to


SPLIT_MIN_WIDTH = 84
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def side_width(width: int) -> int:
    return min(46, max(32, width * 36 // 100))


def pane_width(width: int, split: bool) -> int:
    """The columns the transcript gets: after the run pane, a space, the bar and a space
    when split, else after the left margin; one column of right margin either way."""
    return width - (side_width(width) + 4 if split else 1) - 1


def transcript_lines(view: View, width: int, only_last: bool = False) -> List[str]:
    return transcript_rows(view, width, only_last)[0]


def transcript_rows(view: View, width: int, only_last: bool = False) -> Tuple[List[str], List[Optional[Entry]]]:
    """The transcript as screen lines, and for each line the entry it belongs to (None for
    the air between entries and the live line). With `only_last`, just the latest exchange:
    from the last thing the person typed onward (everything before it is reached by
    scrolling)."""
    entries = view.entries
    if only_last:
        for index in range(len(entries) - 1, -1, -1):
            if entries[index].kind == "user":
                entries = entries[index:]
                break
    lines: List[str] = []
    owners: List[Optional[Entry]] = []
    previous = None
    for entry in entries:
        block = entry_lines(entry, width, view.expanded)
        if not block:
            continue
        # air before a paragraph, around what the person typed, and before a thought -- a
        # thought starts a step of the agent's work (think, say, do); the one-line machinery
        # under it (tool calls, edits, notes) stacks without any
        if previous is not None and (entry.kind == "user" or previous == "user" or entry.kind not in _QUIET_KINDS
                                     or entry.kind == "thinking"):
            lines.append("")
            owners.append(None)
        lines.extend(block)
        owners.extend([entry] * len(block))
        previous = entry.kind
    if view.partial:
        extra = wrap(view.partial, width)
        lines.extend(extra)
        owners.extend([None] * len(extra))
    if view.live:
        spin = _SPIN[view.frame % len(_SPIN)] if _ui._unicode() else "/-\\|"[view.frame % 4]
        took = _ui.elapsed_text(max(0.0, time.monotonic() - view.live_since)) if view.live_since else ""
        if lines and lines[-1] != "":
            lines.append("")
            owners.append(None)
        lines.append(s(spin + " ", "accent") + s(view.live + "…", "bold")
                     + s(f"  {took} · ctrl+c to cancel" if took else "", "dim"))
        owners.append(None)
    return lines, owners


def keep_in_place(view: View, entry: Entry, at_row: int) -> None:
    """After an entry was opened or shut by a click, scroll so its first line is still on
    the screen row it was clicked on -- the view does not jump to the bottom. Works from
    the layout of the last compose (the same width, so the same wrapping)."""
    width = view.pane_w or 80
    for only_last in ((True, False) if view.only_last else (False,)):
        lines, owners = transcript_rows(view, width, only_last=only_last)
        if entry not in owners:
            continue
        index = owners.index(entry)
        start = index - (at_row - view.first_row)
        scroll = max(0, len(lines) - (start + view.room))
        if scroll == 0 and only_last:
            view.scroll = 0
            return
        if not only_last:
            view.scroll = scroll
            return
    view.scroll = max(0, view.scroll)


def _hints(view: View) -> List[Tuple[str, str]]:
    text = view.editor.text
    if view.busy or view.question or view.options is not None or not text.startswith("/") or " " in text:
        return []
    return [(c, d) for c, d in view.commands if c.startswith(text.lower())][:6]


def _field(prompt: str, text: str, pos: int, width: int, secret: bool) -> Tuple[str, int]:
    """The input line and the column the cursor sits in. Long text scrolls sideways."""
    shown = (g("mask") * len(text)) if secret else text
    room = max(4, width - visible_len(prompt) - 1)
    start = 0
    if pos > room - 1:
        start = pos - (room - 1)
    window = shown[start:start + room]
    more_left = s("…", "dim") if start > 0 else ""
    if more_left:
        window = window[1:]
    line = prompt + more_left + window
    return line, visible_len(prompt) + (pos - start)


def compose(view: View, width: int, height: int) -> Tuple[List[str], int, int]:
    """One frame: `height` lines of at most `width` columns, and where the cursor goes
    (row, column, zero-based)."""
    # The last row of the terminal is left alone: on some terminals writing into it (or
    # into the bottom-right cell) scrolls the whole screen up a line, and the input line
    # was the one that vanished. The frame is drawn one row shorter than the screen.
    width, height = max(20, width), max(8, height) - 1
    rule = s(g("rule") * width, "dim")
    title = s("PULSE", "bold", "accent") + (s(" / ", "dim") + s(view.area.upper(), "bold") if view.area else "")
    if view.context:
        title += s("   " + view.context, "dim")
    agent = s(view.agent, "dim") if view.agent else ""
    gap = width - visible_len(title) - visible_len(agent) - 2
    top = " " + (title + " " * gap + agent if gap >= 1 else _ui._clip(title, width - 2))
    frame: List[str] = [pad(top, width), rule]
    body_h = height - 2

    split = view.side is not None and width >= SPLIT_MIN_WIDTH
    strip: List[str] = []
    if view.side is not None and not split:
        strip = [pad(" " + ln, width) for ln in (view.side_brief or view.side[:3])] + [rule]
    left_w = side_width(width) if split else 0
    # the right column starts after the pane (left_w + 1), a space, the bar and a space
    right_x = left_w + 4 if split else 1
    right_w = pane_width(width, split)

    # ---- the input block, built bottom-up
    block: List[str] = []
    cursor_col = 0
    if view.options is not None:
        if view.question:
            block.extend(s(piece, "bold") for piece in wrap(view.question, right_w)[:3])
        shown = view.option_ids
        rows = min(8, max(3, body_h // 3))
        first = max(0, min(view.option_index - rows // 2, len(shown) - rows))
        if first > 0:
            block.append(s(f"  {g('up')} {first} more", "dim"))
        for row, option_id in enumerate(shown[first:first + rows], first):
            label, detail, tag = view.options[option_id]
            chosen = row == view.option_index
            line = (s(g("sel") + " ", "accent", "bold") + s(label, "bold") if chosen else "  " + label)
            if tag:
                line += s(f"  {tag}", "dim" if _QUIET_TAG_RE.search(tag) else "accent")
            if detail:
                line += s(f"  {detail}", "dim")
            block.append(_ui._clip(line, right_w))
        if not shown:
            block.append(s("  no match", "dim"))
        elif first + rows < len(shown):
            block.append(s(f"  {g('down')} {len(shown) - first - rows} more", "dim"))
        field, cursor_col = _field(s("Search ", "dim") + s(g("cursor") + " ", "accent"), view.editor.text,
                                   view.editor.pos, right_w, False)
        block.append(field)
        footer = f"{g('up')}{g('down')} move · Enter select · type to filter · Esc cancel"
    else:
        if view.question:
            block.extend(s(piece, "bold") for piece in wrap(view.question, right_w)[:3])
        for command, what in _hints(view)[:4]:
            block.append(_ui._clip(s(command.ljust(14), "accent") + s(what, "dim"), right_w))
        field, cursor_col = _field(s(g("cursor") + " ", "accent", "bold"), view.editor.text, view.editor.pos,
                                   right_w, view.secret)
        block.append(field)
        if view.question:
            footer = "Enter answer · Esc cancel"
        elif view.busy:
            footer = "working · Ctrl+C cancel · click or Ctrl+O to expand · ↑↓ scroll"
        else:
            footer = "Enter send · / commands · click or Ctrl+O to expand · ↑↓ scroll" + (
                " · Esc back to the agent" if view.side is not None else "")
    if view.status:
        footer = view.status
    if view.scroll:
        footer = f"{g('up')} scrolled back {view.scroll} lines · End returns · " + footer
    # the field sits between two rules, the key hints under the lower one
    block.append(s(g("rule") * right_w, "dim"))
    block.append(s(_ui._clip(footer, right_w), "dim"))
    block = block[-max(3, body_h - len(strip) - 2):]
    field_row_in_block = len(block) - 3

    # ---- the transcript above it
    room = body_h - len(strip) - len(block) - 1          # -1: the rule over the input
    lines, owners = transcript_rows(view, right_w, only_last=view.only_last and view.scroll == 0)
    if view.scroll:
        lines, owners = transcript_rows(view, right_w)    # scrolled: the whole conversation
    limit = max(0, len(lines) - room)
    view.scroll = max(0, min(view.scroll, limit))
    end = len(lines) - view.scroll
    start = max(0, end - room)
    visible = lines[start:end]
    shown_owners = owners[start:end]
    top_pad = room - len(visible) if len(lines) >= room else 0
    visible = [""] * top_pad + visible if len(lines) >= room else visible + [""] * (room - len(visible))
    first_row = 2 + len(strip)
    view.first_row, view.room, view.pane_w = first_row, room, right_w
    view.row_entries = {first_row + top_pad + i: owner for i, owner in enumerate(shown_owners) if owner is not None}
    right = visible + [s(g("rule") * right_w, "dim")] + block

    # ---- put the columns together
    frame.extend(strip)
    if split:
        side = [(" " + ln) for ln in (view.side or [])]
        bar = s(g("bar"), "dim")
        for i in range(body_h):
            left = pad(side[i] if i < len(side) else "", left_w + 1)
            frame.append(left + " " + bar + " " + pad(right[i] if i < len(right) else "", right_w))
    else:
        for i in range(body_h - len(strip)):
            frame.append(" " + pad(right[i] if i < len(right) else "", right_w))
    frame = frame[:height]
    view.transcript_x = right_x
    cursor_row = 2 + len(strip) + room + 1 + field_row_in_block
    return frame, min(cursor_row, height - 1), right_x + cursor_col


# ---------------------------------------------------------------------------------------
# The terminal
# ---------------------------------------------------------------------------------------

class _WindowsConsole:
    """What a Windows console needs before it can show this: escape sequences switched on
    for output, UTF-8 for the glyphs, and input handed over raw (no line editing, no echo,
    and Ctrl+C as a key rather than a signal). Everything is put back on exit. Harmless
    anywhere else: every call is a no-op when there is no such console."""

    _VT_OUTPUT = 0x0004
    _LINE_INPUT, _ECHO_INPUT, _PROCESSED_INPUT = 0x0002, 0x0004, 0x0001
    _STD_INPUT, _STD_OUTPUT = -10, -11

    def __init__(self) -> None:
        self._saved: List[Tuple[Any, int]] = []
        self._codepage: Optional[int] = None
        self._k32 = None
        if os.name == "nt":
            try:
                import ctypes
                self._k32 = ctypes.windll.kernel32            # type: ignore[attr-defined]
            except Exception:
                self._k32 = None

    def _mode(self, which: int) -> Tuple[Any, Optional[int]]:
        import ctypes
        handle = self._k32.GetStdHandle(which)
        mode = ctypes.c_uint()
        if not self._k32.GetConsoleMode(handle, ctypes.byref(mode)):
            return handle, None
        return handle, mode.value

    def enter(self, output: bool = True, keys: bool = True) -> bool:
        """True when the console is ready (or there is nothing to do)."""
        if self._k32 is None:
            return os.name != "nt"
        try:
            if output:
                handle, mode = self._mode(self._STD_OUTPUT)
                if mode is None or not self._k32.SetConsoleMode(handle, mode | self._VT_OUTPUT):
                    return False
                self._saved.append((handle, mode))
                self._codepage = self._k32.GetConsoleOutputCP()
                self._k32.SetConsoleOutputCP(65001)
            if keys:
                handle, mode = self._mode(self._STD_INPUT)
                if mode is not None:
                    raw = mode & ~(self._LINE_INPUT | self._ECHO_INPUT | self._PROCESSED_INPUT)
                    if self._k32.SetConsoleMode(handle, raw):
                        self._saved.append((handle, mode))
            return True
        except Exception:
            self.exit()
            return False

    def exit(self) -> None:
        if self._k32 is None:
            return
        for handle, mode in reversed(self._saved):
            try:
                self._k32.SetConsoleMode(handle, mode)
            except Exception:
                pass
        self._saved = []
        if self._codepage:
            try:
                self._k32.SetConsoleOutputCP(self._codepage)
            except Exception:
                pass
            self._codepage = None


def windows_console_ready() -> bool:
    """Can a Windows console show the app? Tried for real (and put back) -- a console that
    refuses escape sequences, such as a very old conhost, says no."""
    console = _WindowsConsole()
    ok = console.enter(output=True, keys=False)
    console.exit()
    return ok


class Screen:
    """The alternate screen, redrawn by rewriting only the rows that changed. Writes go
    straight to the terminal's file descriptor: sys.stdout belongs to the app while it
    runs (it is where the hosted pipelines' output is captured)."""

    def __init__(self, fd: Optional[int] = None) -> None:
        self.fd = fd if fd is not None else sys.__stdout__.fileno()
        self._shown: List[str] = []
        self._size = (0, 0)
        self._active = False
        self._console = _WindowsConsole()

    def write(self, data: str) -> None:
        raw = data.encode("utf-8", "replace")
        while raw:
            try:
                sent = os.write(self.fd, raw)
            except InterruptedError:
                continue
            raw = raw[sent:]

    def size(self) -> Tuple[int, int]:
        try:
            size = os.get_terminal_size(self.fd)
        except OSError:
            size = shutil.get_terminal_size((100, 30))
        return size.columns, size.lines

    def __enter__(self) -> "Screen":
        if not self._console.enter(output=True, keys=False):
            raise _ui.Unavailable("this console cannot show escape sequences")
        # alternate screen, cursor home, bracketed paste on, mouse clicks and wheel reported
        # (SGR encoding), and the cursor as an orange bar (Pulse's accent) -- terminals that
        # cannot recolour it keep their own
        self.write("\033[?1049h\033[H\033[2J\033[?2004h\033[?1000h\033[?1006h\033]12;#ff8700\007\033[5 q")
        self._active = True
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._active:
            self.write("\033[?1006l\033[?1000l\033[?2004l\033[0m\033[?25h\033]112\007\033[0 q\033[?1049l")
            self._active = False
        self._console.exit()

    def draw(self, lines: List[str], cursor_row: int, cursor_col: int) -> None:
        size = self.size()
        out: List[str] = ["\033[?25l"]
        if size != self._size:
            self._size, self._shown = size, []
            out.append("\033[2J")
        for row, line in enumerate(lines):
            if row >= len(self._shown) or self._shown[row] != line:
                # never the last column: a terminal may wrap there and scroll everything
                out.append(f"\033[{row + 1};1H{cut(line.rstrip(' '), size[0] - 1)}\033[0m\033[K")
        self._shown = list(lines)
        out.append(f"\033[{cursor_row + 1};{cursor_col + 1}H\033[?25h")
        self.write("".join(out))


_CSI_KEYS = {
    "A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
    "1~": "home", "4~": "end", "7~": "home", "8~": "end", "3~": "delete",
    "5~": "pgup", "6~": "pgdn", "Z": "shift+tab",
    "1;2A": "pgup", "1;2B": "pgdn",           # shift+up / shift+down scroll too
}
_CTRL_KEYS = {
    "\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace", "\t": "tab",
    "\x01": "ctrl+a", "\x05": "ctrl+e", "\x0b": "ctrl+k", "\x0c": "ctrl+l", "\x0f": "ctrl+o",
    "\x15": "ctrl+u", "\x17": "ctrl+w", "\x04": "ctrl+d", "\x03": "ctrl+c", "\x12": "ctrl+r",
}
_PASTE_START, _PASTE_END = "\033[200~", "\033[201~"
_MOUSE_RE = re.compile(r"\033\[<(\d+);(\d+);(\d+)([Mm])")


def _mouse_key(button: int, x: int, y: int, release: bool) -> Optional[str]:
    """SGR mouse report -> a key name: a click (on release, so a drag selects text), or the
    wheel. x and y are 1-based from the terminal."""
    if button & 64:
        return "wheel:up" if button & 3 == 0 else "wheel:down"
    if release and button & 3 == 0:
        return f"click:{x - 1}:{y - 1}"
    return None


def parse_keys(data: str) -> Tuple[List[str], str]:
    """Key names out of raw terminal input, and what is left over (an escape sequence that
    has not finished arriving)."""
    keys: List[str] = []
    i = 0
    while i < len(data):
        ch = data[i]
        if data.startswith(_PASTE_START, i):
            end = data.find(_PASTE_END, i)
            if end < 0:
                return keys, data[i:]
            keys.append("paste:" + data[i + len(_PASTE_START):end])
            i = end + len(_PASTE_END)
            continue
        if ch == "\033":
            if i + 1 >= len(data):
                return keys, data[i:]            # alone so far: Esc, or the start of a sequence
            nxt = data[i + 1]
            mouse = _MOUSE_RE.match(data, i) if nxt == "[" else None
            if mouse:
                name = _mouse_key(int(mouse.group(1)), int(mouse.group(2)), int(mouse.group(3)), mouse.group(4) == "m")
                if name:
                    keys.append(name)
                i = mouse.end()
                continue
            if nxt == "[" and data.startswith("\033[<", i):
                return keys, data[i:]            # a mouse report still arriving
            if nxt in "[O":
                j = i + 2
                while j < len(data) and not (data[j].isalpha() or data[j] == "~"):
                    j += 1
                if j >= len(data):
                    return keys, data[i:]
                name = _CSI_KEYS.get(data[i + 2:j + 1])
                if name:
                    keys.append(name)
                i = j + 1
                continue
            keys.append("esc")
            i += 1
            continue
        name = _CTRL_KEYS.get(ch)
        if name:
            keys.append(name)
        elif ch >= " ":
            keys.append(ch)
        i += 1
    return keys, ""


class KeyReader:
    """Keys from the terminal, by name. Ctrl+C arrives as the key "ctrl+c" -- the app
    decides what it means (cancel the running turn, or leave) -- and so do Ctrl+O and the
    other keys a terminal would otherwise keep for itself."""

    def __init__(self, fd: Optional[int] = None) -> None:
        self._fd = fd
        self._saved: Any = None
        self._pending = ""
        self._queue: List[str] = []
        self._posix = os.name != "nt"

    def __enter__(self) -> "KeyReader":
        if not self._posix:
            self._console = _WindowsConsole()
            self._console.enter(output=False, keys=True)
            return self
        try:
            import termios
            if self._fd is None:
                self._fd = sys.__stdin__.fileno()
            self._saved = termios.tcgetattr(self._fd)
            mode = termios.tcgetattr(self._fd)
            mode[0] &= ~(termios.IXON | termios.ICRNL | termios.INLCR)
            mode[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG | termios.IEXTEN)
            mode[6][termios.VMIN] = 1
            mode[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSADRAIN, mode)
        except Exception as exc:
            raise _ui.Unavailable(str(exc))
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self._posix:
            console = getattr(self, "_console", None)
            if console is not None:
                console.exit()
            return
        if self._saved is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None

    def read(self, timeout: float = 0.1) -> Optional[str]:
        """The next key, or None if nothing was pressed within `timeout` seconds."""
        if self._queue:
            return self._queue.pop(0)
        if not self._posix:
            return self._read_windows(timeout)
        import select
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            if self._pending == "\033":          # nothing followed it: it was the Esc key
                self._pending = ""
                return "esc"
            return None
        chunk = os.read(self._fd, 4096)
        if not chunk:
            raise EOFError
        keys, self._pending = parse_keys(self._pending + chunk.decode("utf-8", "replace"))
        if self._pending == "\033" and not keys:
            more, _, _ = select.select([self._fd], [], [], 0.03)
            if not more:
                self._pending = ""
                return "esc"
            return self.read(0)
        self._queue.extend(keys)
        return self._queue.pop(0) if self._queue else None

    def _read_windows(self, timeout: float) -> Optional[str]:
        import msvcrt
        deadline = time.monotonic() + timeout
        while not msvcrt.kbhit():
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end",
                    "S": "delete", "I": "pgup", "Q": "pgdn"}.get(msvcrt.getwch())
        if ch == "\x1b":
            return "esc"
        if ch in ("\r", "\n") and msvcrt.kbhit():
            # more is already waiting behind this newline: a paste, not the Enter key
            return "paste:\n"
        return _CTRL_KEYS.get(ch) or (ch if ch >= " " else None)
