"""
Terminal UI primitives for Pulse: one visual language for every screen.

    PULSE / WORKSPACE
    ────────────────────────────────────────────────────────────
    Select a workspace
    Your runs, history, and debugging context live here.

Deliberately small and stdlib-only (termios / msvcrt). It is a *presentation* layer:
nothing here talks to Supabase, stores credentials, picks a model or decides anything
about a run. Callers hand it the data they already have and get back the same value the
old `input()` prompt would have returned, so the logic around every prompt is untouched.

Everything degrades to "not available" rather than to a broken screen. `enabled()` is
False -- and callers then run their original print()/input() code, unchanged -- when:

  * stdin or stdout is not an interactive terminal (pipes, CI, log files, notebooks),
  * TERM=dumb, or
  * PULSE_PLAIN=1 is set (an explicit opt-out).

Any failure while drawing or reading keys also raises `Unavailable`, which callers treat
the same way, so a terminal that turns out not to support this can never strand a user at
a prompt that does not work. Ctrl+C and Ctrl+D behave exactly like `input()`:
KeyboardInterrupt / EOFError.
"""
import os
import re
import shutil
import sys
import threading
import time
import unicodedata

__all__ = [
    "Unavailable", "enabled", "color_enabled", "header", "rule", "ok", "warn", "fail",
    "note", "kv", "subhead", "message", "answer", "show_change", "change_status", "ask", "confirm", "choose", "Option", "Stage", "ready_block",
    "box", "form", "elapsed_text", "commit_looks_valid", "key_hint_for", "set_host", "host",
]


class Unavailable(Exception):
    """The interactive UI can't be used here; fall back to the plain prompt."""


# --------------------------------------------------------------------------------------
# Capability detection
# --------------------------------------------------------------------------------------

def _isatty(stream):
    try:
        return bool(stream.isatty())
    except Exception:
        return False


# The Pulse app (pulse_app) runs the same setup and agent code inside its own full-screen
# layout. While it does, it is the host: questions are answered in its input line, a
# running stage is its spinner, and everything written to stdout lands in its transcript.
# The callers of this module do not change -- they still call ask() / choose() / Stage.
_HOST = None


def set_host(host):
    """Install (or, with None, remove) the object that draws for this module."""
    global _HOST
    _HOST = host


def host():
    return _HOST


def enabled():
    """True only on a real interactive terminal, and never when explicitly opted out."""
    if _HOST is not None:
        return True
    if os.environ.get("PULSE_PLAIN", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return _isatty(sys.stdin) and _isatty(sys.stdout)


def color_enabled():
    # no-color.org: only a non-empty NO_COLOR disables colour; a dumb terminal can't show it.
    if os.environ.get("NO_COLOR") or os.environ.get("TERM", "").lower() == "dumb":
        return False
    if _HOST is not None:
        return True
    return _isatty(sys.stdout)


def _can_encode(text):
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE = None


def _unicode():
    global _UNICODE
    if _UNICODE is None:
        _UNICODE = _can_encode("✓●○⚠✕→│─❯›·↑↓•╭╮╰╯├┤▾◆")
    return _UNICODE


def _g(name):
    """Glyphs, with an ASCII set for terminals whose encoding can't show them."""
    fancy = {"ok": "✓", "now": "●", "todo": "○", "warn": "⚠", "fail": "✕", "arrow": "→",
             "bar": "│", "rule": "─", "cursor": "❯", "sel": "›", "dot": "·", "up": "↑",
             "down": "↓", "mask": "•",
             "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│", "ml": "├", "mr": "┤",
             "chev": "▾", "logo": "◆"}
    plain = {"ok": "+", "now": "*", "todo": "o", "warn": "!", "fail": "x", "arrow": "->",
             "bar": "|", "rule": "-", "cursor": ">", "sel": ">", "dot": "-", "up": "^",
             "down": "v", "mask": "*",
             "tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|", "ml": "+", "mr": "+",
             "chev": "v", "logo": "*"}
    return (fancy if _unicode() else plain)[name]


# --------------------------------------------------------------------------------------
# Styling. Pulse orange marks active and warning states; quiet success and secondary detail
# stay dim, while failures remain red.
# --------------------------------------------------------------------------------------

_RESET = "\033[0m"
_CODES = {
    "accent": "\033[38;5;208m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "green": "\033[2m",
    "amber": "\033[38;5;208m",
    "red": "\033[31m",
    "sel_bg": "\033[48;5;236m",   # the highlighted row of a list
    "pill": "\033[48;5;208m\033[38;5;232m\033[1m",   # the active tab / the focused button: dark text on orange
    "grey": "\033[38;5;248m",      # a command in the app's transcript: grey, in bold
    "glow": "\033[38;5;255m",      # the same, under the mouse: it lights up
    "glow_accent": "\033[38;5;214m",  # a thought under the mouse: a brighter orange
}


def _s(text, *styles):
    if not color_enabled() or not styles:
        return text
    return "".join(_CODES[s] for s in styles) + text + _RESET


_ANSI_RE = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


def _char_width(ch):
    """Terminal columns one character takes: 2 for wide (CJK, most emoji), 0 for
    combining marks, else 1."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _visible_len(text):
    """Display width in terminal columns (ANSI-aware), not the character count."""
    plain = _ANSI_RE.sub("", text)
    if plain.isascii():
        return len(plain)
    return sum(_char_width(ch) for ch in plain)


def _width():
    if _HOST is not None:
        try:
            return max(20, min(int(_HOST.width()), 100))
        except Exception:
            return 80
    try:
        # Never wider than the real terminal: a line clipped to more columns
        # than it has wraps and breaks the in-place redraw.
        return max(1, min(shutil.get_terminal_size((80, 24)).columns, 100))
    except Exception:
        return 80


def _height():
    try:
        return shutil.get_terminal_size((80, 24)).lines
    except Exception:
        return 24


def _clip(text, width):
    """Truncate to `width` visible columns (ANSI-aware) so a line never wraps -- a wrapped
    line would break the cursor arithmetic that redraws a screen in place."""
    if _visible_len(text) <= width:
        return text
    out, visible, i = [], 0, 0
    while i < len(text) and visible < width - 1:
        m = _ANSI_RE.match(text, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        w = _char_width(text[i])
        if visible + w > width - 1:
            break
        out.append(text[i])
        visible += w
        i += 1
    return "".join(out) + "…" * (1 if _unicode() else 0) + (_RESET if color_enabled() else "")


_out_lock = threading.RLock()


def _write(text):
    """Write straight to stdout: bypasses the print() wrapper Pulse installs, whose
    line-clearing prefix would corrupt an in-place redraw."""
    with _out_lock:
        sys.stdout.write(text)
        sys.stdout.flush()


def _emit(line=""):
    _write(line + "\n")


# --------------------------------------------------------------------------------------
# Cards: rounded boxes, so a terminal screen reads like a window
# --------------------------------------------------------------------------------------

def _fit(text, width):
    """`text` clipped or padded with spaces to exactly `width` visible columns."""
    text = _clip(text, width)
    return text + " " * max(0, width - _visible_len(text))


def _hilite(line):
    """A row painted with the selection background, which survives the resets inside it."""
    if not color_enabled():
        return line
    bg = _CODES["sel_bg"]
    return bg + line.replace(_RESET, _RESET + bg) + _RESET


def box(lines, width, *, title="", footer="", tone="dim", pad=1):
    """A rounded card `width` columns wide around `lines` (already styled). The border takes the
    colour of `tone` ("dim", or "accent" for the thing that has focus); `title` sits in the top
    edge and `footer` in the bottom one. A line that is None draws a divider."""
    width = max(8, width)
    inner = width - 2 - 2 * pad
    edge = lambda text: _s(text, tone) if tone else text
    top = edge(_g("tl") + _g("h"))
    room = width - 3                      # what is left of an edge after its two corners and one dash
    if title:
        label = " " + _clip(title, room - 2) + " "
        top += _s(label, "bold") + edge(_g("h") * max(0, room - _visible_len(label)))
    else:
        top += edge(_g("h") * room)
    top += edge(_g("tr"))
    out = [top]
    for line in lines:
        if line is None:
            out.append(edge(_g("ml") + _g("h") * (width - 2) + _g("mr")))
            continue
        out.append(edge(_g("v")) + " " * pad + _fit(line, inner) + " " * pad + edge(_g("v")))
    bottom = edge(_g("bl") + _g("h"))
    if footer:
        label = " " + _clip(footer, room - 2) + " "
        bottom += _s(label, "dim") + edge(_g("h") * max(0, room - _visible_len(label)))
    else:
        bottom += edge(_g("h") * room)
    out.append(bottom + edge(_g("br")))
    return out


def _card_width(limit=72):
    return max(24, min(_width() - 2, limit))


def _field_box(body, width, focused=False, label=None, tone=None):
    """A text field drawn as its own small rounded box (three lines, four with a label)."""
    lines = box([body], width, tone=tone or ("accent" if focused else "dim"))
    return ([_s(label, "bold" if focused else "dim")] if label else []) + lines


def _input_body(shown, cursor, room, placeholder=None, final=False, focused=True):
    """The inside of a text field: a prompt glyph, the text scrolled so the cursor stays in view,
    and a caret at the cursor. `room` is the columns it may take."""
    prefix = _s(_g("cursor"), "accent" if focused else "dim") + " "
    room = max(4, room - 2)
    if not shown:
        hint = _s(_clip(placeholder, room), "dim") if placeholder and not final else ""
        return prefix + (_caret(final or not focused) + hint if hint else _caret(final or not focused))
    start = max(0, cursor - (room - 1))
    window = shown[start:start + room - 1]
    here = cursor - start
    left = _s("…", "dim") if start > 0 else ""
    if left:
        window, here = window[1:], here - 1
    return prefix + left + window[:here] + _caret(final or not focused) + window[here:]


def _caret(final=False):
    return "" if final else _s("▏" if _unicode() else "_", "accent")


# --------------------------------------------------------------------------------------
# Static pieces
# --------------------------------------------------------------------------------------

def rule(width=None):
    return _s(_g("rule") * (width or min(_width(), 60)), "dim")


def header(state, subtitle=None, explain=None):
    """A banner card: PULSE / STATE, then an optional question and explanation."""
    import textwrap
    width = _card_width()
    inner = width - 4
    body = [_s(_g("logo"), "accent") + " " + _s("PULSE", "bold", "accent") + _s("  /  ", "dim") + _s(state.upper(), "bold")]
    if subtitle or explain:
        body.append("")
    if subtitle:
        body += [_s(piece, "bold") for piece in textwrap.wrap(subtitle, inner) or [""]]
    if explain:
        body += [_s(piece, "dim") for piece in textwrap.wrap(explain, inner)]
    _emit()
    for line in box(body, width):
        _emit(line)
    _emit()


def subhead(text):
    """A bold line for a step inside the current screen (no second PULSE / ... header)."""
    _emit(_s(text, "bold"))


def ok(label, value=None):
    """A completed step: collapsed to one line once a screen is done."""
    line = _s(_g("ok"), "green") + " " + (_s(label.ljust(16), "dim") if value is not None else label)
    _emit(line + (value if value is not None else ""))


def warn(text):
    _emit(_s(_g("warn"), "amber") + " " + text)


def fail(text):
    _emit(_s(_g("fail"), "red") + " " + text)


def note(text):
    _emit(_s(text, "dim"))


def message(text):
    """Something the agent says to the person while it keeps working (the MESSAGE: directive,
    the message_user tool). Inside the Pulse app it is an entry of its own; on a plain
    terminal it is printed with a bar down its left side, so it reads as the agent's voice
    and not as tool output."""
    text = str(text or "").strip()
    if not text:
        return
    host = _HOST
    if host is not None and hasattr(host, "say"):
        host.say(text)
        return
    bar = _s(_g("bar") + " ", "accent")
    with _out_lock:
        sys.stdout.write("\n" + "\n".join(bar + line for line in text.splitlines()) + "\n\n")
        sys.stdout.flush()


def show_change(labels, diff):
    """A change the agent wants to make to the files, before it is reviewed or applied.
    `labels` name the files ("EDIT: train.py", "CREATE: tests/test_x.py"); `diff` is the
    coloured unified diff for a terminal. Inside the Pulse app it is one folded line that
    opens to the diff; on a plain terminal the diff is printed."""
    host = _HOST
    if host is not None and hasattr(host, "edit"):
        host.edit(list(labels), diff)
        return
    with _out_lock:
        sys.stdout.write("\n" + diff + "\n")
        sys.stdout.flush()


def change_status(text, detail=None, color=None, final=False, terminal_text=None):
    """What became of the change shown by show_change(): "approved by X", "applied", "not
    applied". In the app `text` joins the change's line and `detail` (the reviewer's reason,
    the /undo hint) shows under the diff when it is opened; `final` = the last word on it.
    On a terminal `terminal_text` (or `text`) is printed in `color`."""
    host = _HOST
    if host is not None and hasattr(host, "edit_status"):
        host.edit_status(text, detail=detail, final=final)
        return
    line = terminal_text if terminal_text is not None else text
    if color is not None:
        line = f"{color}{line}{_RESET}" if color_enabled() else line
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def answer(text):
    """The agent's own words to the person -- an answer, a plan, a summary. Inside the Pulse
    app it is shown as the agent speaking (like message()); on a plain terminal it is
    printed as it always was."""
    text = str(text or "")
    if not text.strip():
        return
    host = _HOST
    if host is not None and hasattr(host, "say"):
        host.say(text.strip())
        return
    with _out_lock:
        sys.stdout.write(f"\n{text.strip()}\n\n")
        sys.stdout.flush()


def kv(label, value, indent=2):
    _emit(" " * indent + _s(label.ljust(12), "dim") + str(value))


def elapsed_text(seconds):
    # Choose the unit from the value as it will be shown: 0.9996 s is "1.0s", not
    # "1000ms", and 59.96 s is "1m 00s", not "60.0s".
    if round(seconds * 1000) < 1000:
        return f"{seconds * 1000:.0f}ms"
    if round(seconds, 1) < 60:
        return f"{seconds:.1f}s"
    seconds = round(seconds)
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


# --------------------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------------------

class _Keys:
    """Read one key at a time in cbreak mode -- not raw mode, so Ctrl+C still raises
    KeyboardInterrupt through the normal signal path exactly as it does at input()."""

    def __init__(self):
        self._posix = os.name != "nt"
        self._fd = None
        self._saved = None

    def __enter__(self):
        if self._posix:
            try:
                import termios
                import tty
                self._fd = sys.stdin.fileno()
                self._saved = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except Exception as exc:
                raise Unavailable(str(exc))
        return self

    def __exit__(self, *exc):
        if self._posix and self._saved is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass

    def _read_byte_posix(self, timeout=None):
        import select
        if timeout is not None:
            ready, _, _ = select.select([self._fd], [], [], timeout)
            if not ready:
                return None
        data = os.read(self._fd, 1)
        if not data:
            raise EOFError
        return data

    def _read_char_posix(self, timeout=None):
        first = self._read_byte_posix(timeout)
        if first is None:
            return None
        lead = first[0]
        need = 0 if lead < 0x80 else 1 if lead >> 5 == 0b110 else 2 if lead >> 4 == 0b1110 else 3
        buf = first
        for _ in range(need):
            nxt = self._read_byte_posix(0.05)
            if nxt is None:
                break
            buf += nxt
        return buf.decode("utf-8", errors="replace")

    def read(self):
        """One of: 'up','down','left','right','home','end','delete','enter','esc',
        'backspace','tab', or a single printable character. Ctrl+C -> KeyboardInterrupt,
        Ctrl+D -> EOFError."""
        if not self._posix:
            return self._read_windows()
        ch = self._read_char_posix()
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("\x7f", "\x08"):
            return "backspace"
        if ch == "\t":
            return "tab"
        if ch == "\x04":
            raise EOFError
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1b":
            nxt = self._read_char_posix(0.05)
            if nxt is None:
                return "esc"
            if nxt in ("[", "O"):
                seq = ""
                while True:
                    c = self._read_char_posix(0.05)
                    if c is None:
                        break
                    seq += c
                    if c.isalpha() or c == "~":
                        break
                table = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
                         "1~": "home", "4~": "end", "7~": "home", "8~": "end", "3~": "delete"}
                return table.get(seq, "")
            return ""
        if ch and ch >= " ":
            return ch
        return ""

    def _read_windows(self):
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x08":
            return "backspace"
        if ch == "\t":
            return "tab"
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x04" or ch == "\x1a":
            raise EOFError
        if ch == "\x1b":
            return "esc"
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            return {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end",
                    "S": "delete"}.get(code, "")
        return ch if ch >= " " else ""


def _hide_cursor():
    _write("\033[?25l")


def _show_cursor():
    _write("\033[?25h")


def _erase_lines(count):
    """Move to the top of a block `count` lines tall (cursor is just below it) and clear it."""
    if count > 0:
        _write(f"\033[{count}A\033[J")


# --------------------------------------------------------------------------------------
# Text input
# --------------------------------------------------------------------------------------

def ask(label, *, secret=False, placeholder=None, validate=None, footer="Enter Continue",
        title=None, note=None):
    """A single-line field. Returns the text typed, exactly as `input()` / `getpass()`
    would: `.strip()` is left to the caller, an empty entry returns "", Ctrl+C raises
    KeyboardInterrupt and Ctrl+D raises EOFError. `secret` echoes bullets, never the text.

    `validate(text)` may return a short status string to show under the field as you type
    (purely informational -- it never blocks submitting). `title` and `note` are for the app's
    popup (its heading and a quiet line under the field); a plain terminal shows `note` under it.
    """
    if _HOST is not None:
        return _HOST.ask(_ANSI_RE.sub("", label), secret=secret, placeholder=placeholder,
                         title=title or "", note=note or "")
    if not enabled():
        raise Unavailable("not an interactive terminal")

    buf = []
    cursor = 0
    lines_drawn = [0]
    width = _width()

    def render(final=False):
        _erase_lines(lines_drawn[0])
        text = "".join(buf)
        shown = (_g("mask") * len(text)) if secret else text
        box_w = _card_width()
        body = _input_body(shown, cursor, box_w - 4, placeholder, final)
        out = [_s(_clip(label, box_w), "dim")] + box([body], box_w, tone="dim" if final else "accent")
        if note and not final:
            out.append(_s(_clip(note, box_w), "dim"))
        if validate is not None and not final:
            try:
                status = validate(text) if text else None
            except Exception:
                status = None
            out.append((_s(status, "dim") if status else ""))
        if not final:
            out.append(_s(footer, "dim"))
        clipped = [_clip(line, width) for line in out]
        _write("\n".join(clipped) + "\n")
        lines_drawn[0] = len(clipped)

    try:
        with _Keys() as keys:
            render()
            while True:
                key = keys.read()
                if key == "enter":
                    break
                if key == "backspace":
                    if cursor > 0:
                        buf.pop(cursor - 1)
                        cursor -= 1
                elif key == "delete":
                    if cursor < len(buf):
                        buf.pop(cursor)
                elif key == "left":
                    cursor = max(0, cursor - 1)
                elif key == "right":
                    cursor = min(len(buf), cursor + 1)
                elif key == "home":
                    cursor = 0
                elif key == "end":
                    cursor = len(buf)
                elif len(key) == 1 and key >= " ":
                    buf.insert(cursor, key)
                    cursor += 1
                render()
    except (KeyboardInterrupt, EOFError):
        _erase_lines(lines_drawn[0])
        raise
    except Unavailable:
        raise
    except Exception as exc:                 # a terminal we can't drive: let the caller fall back
        raise Unavailable(str(exc))

    _erase_lines(lines_drawn[0])
    return "".join(buf)


def confirm(question, *, default=True, detail=None):
    """Yes/No as two buttons, the default one selected. Returns True/False.

    Enter takes the selected button, Y / N answer at once, the arrow keys and Tab move between
    them, and Esc declines. (Inside the app the question is asked in its own popup instead.)"""
    if _HOST is not None:
        hint = "Y/n" if default else "y/N"
        answer = ask(f"{question}  ({hint})").strip().lower()
        return default if not answer else answer in ("y", "yes")
    if not enabled():
        raise Unavailable("not an interactive terminal")
    chosen = [bool(default)]
    drawn = [0]
    width = _width()

    def button(text, selected):
        label = _s(text, "bold", "accent") if selected else _s(text, "dim")
        return box([label], 12, tone="accent" if selected else "dim")

    def render():
        _erase_lines(drawn[0])
        yes, no = button("  Yes  ", chosen[0]), button("  No  ", not chosen[0])
        out = [_s(_clip(question, width), "bold")]
        if detail:
            out.append(_s(_clip(detail, width), "dim"))
        out += [a + "  " + b for a, b in zip(yes, no)]
        out.append(_s("\u2190 \u2192 Choose    Y / N    Enter Confirm" if _unicode() else "Left/Right Choose   Y/N   Enter Confirm", "dim"))
        clipped = [_clip(line, width) for line in out]
        _write("\n".join(clipped) + "\n")
        drawn[0] = len(clipped)

    try:
        with _Keys() as keys:
            _hide_cursor()
            try:
                render()
                while True:
                    key = keys.read()
                    if key == "enter":
                        break
                    if key in ("y", "Y"):
                        chosen[0] = True
                        break
                    if key in ("n", "N", "esc"):
                        chosen[0] = False
                        break
                    if key in ("left", "right", "tab", "up", "down"):
                        chosen[0] = not chosen[0]
                        render()
            finally:
                _show_cursor()
    except (KeyboardInterrupt, EOFError):
        _erase_lines(drawn[0])
        raise
    except Exception as exc:
        raise Unavailable(str(exc))
    _erase_lines(drawn[0])
    return chosen[0]


# --------------------------------------------------------------------------------------
# Choosing from a list
# --------------------------------------------------------------------------------------

class Option:
    """One row of a list. `key` is an optional single-letter shortcut."""

    def __init__(self, label, detail=None, tag=None, key=None, value=None, badge=None, tone=None):
        self.label = label
        self.detail = detail
        self.tag = tag
        self.key = key
        self.value = value
        self.badge = badge          # a status shown at the row's right edge ("● key set")
        self.tone = tone            # "attn": draws the eye (orange); anything else is quiet


def choose(options, *, initial=0, searchable=False, max_rows=None, footer=None,
           search_label="Search", allow_cancel=True, empty_text="No matches", hotkeys=None, title=None,
           compact=False):
    """Pick one option. Returns its index in `options`, or None if cancelled (Esc, or Enter
    on a filtered-empty list). `title`, when given, is the question shown above the list.

    Arrow keys move; Enter selects; a row's `key` shortcut selects it directly (unless
    `searchable`, where letters go to the search box). `hotkeys` maps extra letters that are
    not rows to a token: pressing one returns that token (a str) instead of an index.
    Ctrl+C raises KeyboardInterrupt.
    """
    if _HOST is not None:
        return _HOST.choose(options, initial=initial, hotkeys=hotkeys, title=title or "") if options else None
    if not enabled():
        raise Unavailable("not an interactive terminal")
    if not options:
        return None

    width = _width()
    query = []
    pos = [max(0, min(initial, len(options) - 1))]     # index into `visible`
    top = [0]
    lines_drawn = [0]

    def visible():
        text = "".join(query).strip().lower()
        if not text:
            return list(range(len(options)))
        terms = text.split()
        out = []
        for i, opt in enumerate(options):
            hay = " ".join(str(x) for x in (opt.label, opt.detail or "", opt.tag or "")).lower()
            if all(t in hay for t in terms):
                out.append(i)
        return out

    per_item = 2 if (any(o.detail for o in options) and not compact) else 1

    def rows_available():
        chrome = 7 + (2 if searchable else 0) + (1 if title else 0)
        budget = max(per_item, _height() - chrome - 4)
        rows = max(3, budget // per_item)
        return min(rows, max_rows or rows, len(options))

    def render(final_index=None):
        _erase_lines(lines_drawn[0])
        vis = visible()
        box_w = _card_width()
        inner = box_w - 4
        body = []
        if searchable:
            typed = "".join(query)
            body.append(_s(search_label, "dim") + "  " + _s(_g("cursor"), "accent") + " " + typed
                        + ("" if final_index is not None else _caret()))
            body.append(None)
        rows = rows_available()
        if vis:
            pos[0] = max(0, min(pos[0], len(vis) - 1))
            if pos[0] < top[0]:
                top[0] = pos[0]
            if pos[0] >= top[0] + rows:
                top[0] = pos[0] - rows + 1
            top[0] = max(0, min(top[0], max(0, len(vis) - rows)))
            if top[0] > 0:
                body.append(_s(f"  {_g('up')} {top[0]} more", "dim"))
            for slot in range(top[0], min(len(vis), top[0] + rows)):
                opt = options[vis[slot]]
                current = slot == pos[0]
                marker = _s(_g("sel"), "accent", "bold") if current else " "
                label = _s(opt.label, "bold") if current else opt.label
                tag = ("  " + _s(opt.tag, "dim")) if opt.tag else ""
                inline = ("  " + _s(opt.detail, "dim")) if (compact and opt.detail) else ""
                left = f"{marker} {label}{tag}{inline}"
                badge = _s(opt.badge, "accent" if opt.tone == "attn" else "dim") if getattr(opt, "badge", None) else ""
                if badge:
                    left = _clip(left, max(8, inner - _visible_len(badge) - 2))
                    line = left + " " * max(2, inner - _visible_len(left) - _visible_len(badge)) + badge
                else:
                    line = left
                lines = [line]
                if per_item == 2:
                    lines.append("  " + (_s(opt.detail, "dim") if opt.detail else ""))
                for text in lines:
                    body.append(_hilite(_fit(text, inner)) if current else text)
            hidden_below = len(vis) - (top[0] + rows)
            if hidden_below > 0:
                body.append(_s(f"  {_g('down')} {hidden_below} more", "dim"))
        else:
            body.append(_s(empty_text, "dim"))
        out = [_s(title, "bold")] if title else []
        out += box(body, box_w, tone="accent")
        keys_hint = [f"{_g('up')}{_g('down')} Navigate", "Enter Select"]
        if searchable:
            keys_hint.append("Type to search")
        if allow_cancel:
            keys_hint.append("Esc Skip" if not query else "Esc Clear")
        out.append(_s("    ".join(keys_hint) if footer is None else footer, "dim"))
        clipped = [_clip(line, width) for line in out]
        _write("\n".join(clipped) + "\n")
        lines_drawn[0] = len(clipped)

    result = None
    try:
        with _Keys() as keys:
            _hide_cursor()
            try:
                render()
                while True:
                    key = keys.read()
                    vis = visible()
                    if key == "enter":
                        result = vis[pos[0]] if vis else None
                        break
                    if key == "up":
                        pos[0] = (pos[0] - 1) % max(1, len(vis))
                    elif key == "down":
                        pos[0] = (pos[0] + 1) % max(1, len(vis))
                    elif key == "home":
                        pos[0] = 0
                    elif key == "end":
                        pos[0] = max(0, len(vis) - 1)
                    elif key == "esc":
                        if query:
                            query.clear()
                            pos[0], top[0] = 0, 0
                        elif allow_cancel:
                            result = None
                            break
                    elif key == "backspace":
                        if query:
                            query.pop()
                            pos[0], top[0] = 0, 0
                    elif len(key) == 1 and key >= " ":
                        if not searchable and hotkeys and key.lower() in hotkeys:
                            result = hotkeys[key.lower()]
                            break
                        shortcut = None if searchable else next(
                            (i for i, o in enumerate(options) if o.key and o.key.lower() == key.lower()), None)
                        if shortcut is not None:
                            result = shortcut
                            break
                        if searchable:
                            query.append(key)
                            pos[0], top[0] = 0, 0
                    render()
            finally:
                _show_cursor()
    except (KeyboardInterrupt, EOFError):
        _erase_lines(lines_drawn[0])
        raise
    except Unavailable:
        raise
    except Exception as exc:
        raise Unavailable(str(exc))

    _erase_lines(lines_drawn[0])
    lines_drawn[0] = 0
    return result


# --------------------------------------------------------------------------------------
# A form: one card, several fields (sign up / log in)
# --------------------------------------------------------------------------------------

def form(title, subtitle=None, *, tabs=None, tab=0, fields, button="Continue", footer=None):
    """Several fields on one card, with optional tabs along the top. Returns (tab, {name: text}) or
    None when the person pressed Esc. Ctrl+C / Ctrl+D behave as at input().

    `fields` is a list of field dicts, or (with `tabs`) one such list per tab. A field is
    {"name", "label", "secret", "placeholder", "hint", "check"}: `hint(text)` is a quiet line shown
    as you type, `check(text, values)` returns an error to show under the field (and stop the form
    from being sent) or None. Values with the same name carry across tabs, so an email typed under
    "Sign up" is still there under "Log in". `button` may be one label per tab.

    Tab / Down / Enter go to the next field, Up to the previous (and, from the first field, to the
    tabs), Left / Right switch tabs while they are focused."""
    if _HOST is not None:
        raise Unavailable("the app asks its own questions")
    if not enabled():
        raise Unavailable("not an interactive terminal")
    tabs = list(tabs or [])
    per_tab = [list(f) for f in fields] if tabs and fields and isinstance(fields[0], (list, tuple)) else None
    flat = [] if per_tab else list(fields)

    def current_fields(index):
        return per_tab[index] if per_tab else flat

    def label_for(index):
        return button[index] if isinstance(button, (list, tuple)) else button

    most = max((len(f) for f in (per_tab or [flat])), default=0)
    if _height() < 15 + most * 5:
        raise Unavailable("terminal too short for the form")

    values, cursors, errors = {}, {}, {}
    state = {"tab": max(0, min(tab, max(0, len(tabs) - 1))), "focus": 0}
    drawn = [0]
    width = _width()

    def render(final=False):
        _erase_lines(drawn[0])
        card_w = _card_width()
        inner = card_w - 4
        index = state["tab"]
        active = current_fields(index)
        focus = state["focus"]
        body = ["", _s(_g("logo"), "accent") + " " + _s("PULSE", "bold", "accent")
                + (_s("   " + subtitle, "dim") if subtitle else ""), ""]
        if tabs:
            pills = []
            for i, name in enumerate(tabs):
                if i == index:
                    pills.append(_s(f"  {name}  ", "bold", "accent"))      # the active tab: orange text, no fill
                else:
                    pills.append(_s(f"  {name}  ", "dim"))
            body += ["  ".join(pills), ""]
        for position, spec in enumerate(active):
            name = spec["name"]
            text = values.get(name, "")
            shown = (_g("mask") * len(text)) if spec.get("secret") else text
            focused = focus == position
            field = _input_body(shown, cursors.get(name, len(text)), inner - 4, spec.get("placeholder"),
                                final=final, focused=focused)
            note = errors.get(name)
            body += _field_box(field, inner, focused=focused, label=spec.get("label"), tone="red" if note else None)
            status = ""                 # always one row, so nothing moves when a message appears
            if note:
                status = _s(_g("fail"), "red") + " " + _s(note, "red")
            elif focused and spec.get("hint") and text:
                try:
                    hint = spec["hint"](text)
                except Exception:
                    hint = None
                status = _s(hint, "dim") if hint else ""
            body.append(status)
        label = label_for(index)
        pressed = focus == len(active)
        text_label = f"{label}  {_g('arrow')}" if pressed else label
        body += box([_s(text_label, "bold", "accent") if pressed else _s(text_label, "bold")],
                    min(inner, len(label) + 10), tone="accent" if pressed else "dim")
        body.append("")
        hint = footer or (f"Tab Next    {_g('up')} Back    " + ("\u2190\u2192 Switch tab    " if tabs and _unicode() else "")
                          + "Enter Continue    Esc Back")
        out = box(body, card_w, tone="accent") + [_s(hint, "dim")]
        clipped = [_clip(line, width) for line in out]
        _write("\n".join(clipped) + "\n")
        drawn[0] = len(clipped)

    def submit():
        index = state["tab"]
        active = current_fields(index)
        errors.clear()
        for position, spec in enumerate(active):
            text = values.get(spec["name"], "")
            problem = None
            if not text.strip():
                problem = "Required"
            elif spec.get("check"):
                try:
                    problem = spec["check"](text, values)
                except Exception:
                    problem = None
            if problem:
                errors[spec["name"]] = problem
        if errors:
            state["focus"] = next(p for p, spec in enumerate(active) if spec["name"] in errors)
            return False
        return True

    try:
        with _Keys() as keys:
            _hide_cursor()
            try:
                render()
                while True:
                    key = keys.read()
                    index = state["tab"]
                    active = current_fields(index)
                    focus = state["focus"]
                    spec = active[focus] if 0 <= focus < len(active) else None
                    if key == "esc":
                        _erase_lines(drawn[0])
                        drawn[0] = 0
                        return None
                    if key == "enter":
                        if focus == -1 or (spec is not None and focus < len(active) - 1):
                            state["focus"] = focus + 1
                        elif submit():
                            break
                    elif key in ("tab", "down"):
                        state["focus"] = (focus + 1) if focus < len(active) else (0 if key == "tab" else focus)
                    elif key == "up":
                        state["focus"] = max(-1 if tabs else 0, focus - 1)
                    elif key in ("left", "right") and focus == -1:
                        state["tab"] = (index + (1 if key == "right" else -1)) % len(tabs)
                        errors.clear()
                    elif spec is not None:
                        name = spec["name"]
                        text = values.get(name, "")
                        at = cursors.get(name, len(text))
                        errors.pop(name, None)
                        if key == "backspace" and at > 0:
                            text, at = text[:at - 1] + text[at:], at - 1
                        elif key == "delete":
                            text = text[:at] + text[at + 1:]
                        elif key == "left":
                            at = max(0, at - 1)
                        elif key == "right":
                            at = min(len(text), at + 1)
                        elif key == "home":
                            at = 0
                        elif key == "end":
                            at = len(text)
                        elif len(key) == 1 and key >= " ":
                            text, at = text[:at] + key + text[at:], at + 1
                        values[name], cursors[name] = text, at
                    render()
            finally:
                _show_cursor()
    except (KeyboardInterrupt, EOFError):
        _erase_lines(drawn[0])
        raise
    except Unavailable:
        raise
    except Exception as exc:
        raise Unavailable(str(exc))
    _erase_lines(drawn[0])
    return state["tab"], {spec["name"]: values.get(spec["name"], "") for spec in current_fields(state["tab"])}


# --------------------------------------------------------------------------------------
# Progress of something that is really happening
# --------------------------------------------------------------------------------------

class Stage:
    """`● label` while the work runs, `✓ label  (4.2s)` once it finishes.

    Wraps real work only -- it is used as a context manager around a call that is actually
    in progress, so what it shows is what Pulse is doing, never a scripted sequence. If the
    body raises, the line ends as `✕ label` and the exception propagates untouched.
    """

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = None
        self._started = 0.0
        self._drawn = False

    def _line(self, glyph, tail=""):
        return f"  {glyph} {self.label}{tail}"

    def _spin(self):
        on = True
        while not self._stop.is_set():
            glyph = _s(_g("now"), "accent") if on else _s(_g("now"), "dim")
            with _out_lock:
                sys.stdout.write("\r\033[K" + _clip(self._line(glyph, _s("…" if _unicode() else "...", "dim")), _width()))
                sys.stdout.flush()
            on = not on
            self._stop.wait(0.45)

    def __enter__(self):
        self._started = time.monotonic()
        self._hosted = _HOST
        if self._hosted is not None:
            self._hosted.stage_start(self.label)
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(self, "_hosted", None) is not None:
            self._hosted.stage_end(self.label, exc_type is None, time.monotonic() - self._started)
            self._hosted = None
            return False
        self._stop.set()
        if self._thread:
            self._thread.join()
        took = elapsed_text(time.monotonic() - self._started)
        if exc_type is None:
            glyph, tail = _s(_g("ok"), "green"), _s(f"  {took}", "dim")
        else:
            glyph, tail = _s(_g("fail"), "red"), _s(f"  {took}", "dim")
        with _out_lock:
            sys.stdout.write("\r\033[K" + _clip(self._line(glyph, tail), _width()) + "\n")
            sys.stdout.flush()
        return False


# --------------------------------------------------------------------------------------
# Small helpers callers use to describe real values
# --------------------------------------------------------------------------------------

_HEX_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def commit_looks_valid(text):
    """Informational only: 7-40 hex characters, the shape of a git commit id."""
    return bool(_HEX_SHA.match((text or "").strip()))


# Well-known key prefixes. Used only to say "this looks like a <provider> key" -- never to
# reject one, and never for a provider whose keys have no stable prefix.
_KEY_PREFIXES = (
    ("sk-ant-", "Anthropic"),
    ("sk-or-", "OpenRouter"),
    ("AIza", "Google AI Studio"),
    ("sk-", "OpenAI"),
)


def key_hint_for(text, env_key=None):
    """"Looks like an Anthropic key" -- said only when the key's own prefix agrees with the
    provider the person chose. Never a rejection: silence is the answer to anything else."""
    text = (text or "").strip()
    if len(text) < 12:
        return None
    expected = {
        "ANTHROPIC_API_KEY": "Anthropic", "OPENAI_API_KEY": "OpenAI",
        "GEMINI_API_KEY": "Google AI Studio", "OPENROUTER_API_KEY": "OpenRouter",
    }.get(env_key or "")
    if not expected:
        return None
    for prefix, vendor in _KEY_PREFIXES:          # ordered most-specific first
        if text.startswith(prefix):
            if vendor == expected:
                return f"{_g('ok')} Looks like {'an' if vendor[0] in 'AEIOU' else 'a'} {vendor} key"
            return None
    return None


def ready_block(rows, closing=None):
    """`rows` is [(label, value_or_None, done_bool)]. Printed as one compact block."""
    header("READY")
    for label, value, done in rows:
        glyph = _s(_g("ok"), "green") if done else _s(_g("todo"), "dim")
        text = label.ljust(16)
        _emit(f"{glyph} {_s(text, 'dim') if value else text}{value or ''}")
    if closing:
        _emit()
        _emit(closing)