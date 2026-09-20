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

__all__ = [
    "Unavailable", "enabled", "color_enabled", "header", "rule", "ok", "warn", "fail",
    "note", "kv", "subhead", "ask", "confirm", "choose", "Option", "Stage", "ready_block",
    "elapsed_text", "commit_looks_valid", "key_hint_for",
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


def enabled():
    """True only on a real interactive terminal, and never when explicitly opted out."""
    if os.environ.get("PULSE_PLAIN", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return _isatty(sys.stdin) and _isatty(sys.stdout)


def color_enabled():
    return _isatty(sys.stdout) and os.environ.get("NO_COLOR") is None


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
        _UNICODE = _can_encode("✓●○⚠✕→│─❯›·↑↓•")
    return _UNICODE


def _g(name):
    """Glyphs, with an ASCII set for terminals whose encoding can't show them."""
    fancy = {"ok": "✓", "now": "●", "todo": "○", "warn": "⚠", "fail": "✕", "arrow": "→",
             "bar": "│", "rule": "─", "cursor": "❯", "sel": "›", "dot": "·", "up": "↑",
             "down": "↓", "mask": "•"}
    plain = {"ok": "+", "now": "*", "todo": "o", "warn": "!", "fail": "x", "arrow": "->",
             "bar": "|", "rule": "-", "cursor": ">", "sel": ">", "dot": "-", "up": "^",
             "down": "v", "mask": "*"}
    return (fancy if _unicode() else plain)[name]


# --------------------------------------------------------------------------------------
# Styling. One accent (Pulse orange, the same 256-colour orange the rest of the CLI uses);
# green / amber / red are reserved for success / warning / failure.
# --------------------------------------------------------------------------------------

_RESET = "\033[0m"
_CODES = {
    "accent": "\033[38;5;208m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "amber": "\033[33m",
    "red": "\033[31m",
}


def _s(text, *styles):
    if not color_enabled() or not styles:
        return text
    return "".join(_CODES[s] for s in styles) + text + _RESET


_ANSI_RE = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


def _visible_len(text):
    return len(_ANSI_RE.sub("", text))


def _width():
    try:
        return max(40, min(shutil.get_terminal_size((80, 24)).columns, 100))
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
        out.append(text[i])
        visible += 1
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
# Static pieces
# --------------------------------------------------------------------------------------

def rule(width=None):
    return _s(_g("rule") * (width or min(_width(), 60)), "dim")


def header(state, subtitle=None, explain=None):
    """PULSE / STATE, a rule, then an optional one-line question and explanation."""
    _emit()
    _emit(_s("PULSE", "bold", "accent") + _s(" / ", "dim") + _s(state.upper(), "bold"))
    _emit(rule())
    if subtitle:
        _emit(_s(subtitle, "bold"))
    if explain:
        _emit(_s(explain, "dim"))
    if subtitle or explain:
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


def kv(label, value, indent=2):
    _emit(" " * indent + _s(label.ljust(12), "dim") + str(value))


def elapsed_text(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
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

def ask(label, *, secret=False, placeholder=None, validate=None, footer="Enter Continue"):
    """A single-line field. Returns the text typed, exactly as `input()` / `getpass()`
    would: `.strip()` is left to the caller, an empty entry returns "", Ctrl+C raises
    KeyboardInterrupt and Ctrl+D raises EOFError. `secret` echoes bullets, never the text.

    `validate(text)` may return a short status string to show under the field as you type
    (purely informational -- it never blocks submitting).
    """
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
        field_prefix = _s(_g("cursor"), "accent") + " "
        if shown:
            body = shown
        elif placeholder and not final:
            body = _s(placeholder, "dim")
        else:
            body = ""
        out = [_s(label, "dim"), field_prefix + body + ("" if final else _s("▏" if _unicode() else "_", "accent"))]
        if validate is not None and not final:
            try:
                status = validate(text) if text else None
            except Exception:
                status = None
            out.append((_s(status, "dim") if status else ""))
        if not final:
            out.append("")
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
    """Yes/No with a default on Enter. Returns True/False."""
    if not enabled():
        raise Unavailable("not an interactive terminal")
    hint = "Y/n" if default else "y/N"
    label = f"{question}  " + _s(f"({hint})", "dim")
    result = ask(label, footer="Enter Continue")
    answer = result.strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


# --------------------------------------------------------------------------------------
# Choosing from a list
# --------------------------------------------------------------------------------------

class Option:
    """One row of a list. `key` is an optional single-letter shortcut."""

    def __init__(self, label, detail=None, tag=None, key=None, value=None):
        self.label = label
        self.detail = detail
        self.tag = tag
        self.key = key
        self.value = value


def choose(options, *, initial=0, searchable=False, max_rows=None, footer=None,
           search_label="Search", allow_cancel=True, empty_text="No matches", hotkeys=None):
    """Pick one option. Returns its index in `options`, or None if cancelled (Esc, or Enter
    on a filtered-empty list).

    Arrow keys move; Enter selects; a row's `key` shortcut selects it directly (unless
    `searchable`, where letters go to the search box). `hotkeys` maps extra letters that are
    not rows to a token: pressing one returns that token (a str) instead of an index.
    Ctrl+C raises KeyboardInterrupt.
    """
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

    per_item = 2 if any(o.detail for o in options) else 1

    def rows_available():
        chrome = 7 + (2 if searchable else 0)
        budget = max(per_item, _height() - chrome - 4)
        rows = max(3, budget // per_item)
        return min(rows, max_rows or rows, len(options))

    def render(final_index=None):
        _erase_lines(lines_drawn[0])
        vis = visible()
        out = []
        if searchable:
            typed = "".join(query)
            out.append(_s(search_label, "dim"))
            out.append(_s(_g("cursor"), "accent") + " " + typed + ("" if final_index is not None else _s("▏" if _unicode() else "_", "accent")))
            out.append(rule())
        rows = rows_available()
        if vis:
            pos[0] = max(0, min(pos[0], len(vis) - 1))
            if pos[0] < top[0]:
                top[0] = pos[0]
            if pos[0] >= top[0] + rows:
                top[0] = pos[0] - rows + 1
            top[0] = max(0, min(top[0], max(0, len(vis) - rows)))
            if top[0] > 0:
                out.append(_s(f"  {_g('up')} {top[0]} more", "dim"))
            for slot in range(top[0], min(len(vis), top[0] + rows)):
                opt = options[vis[slot]]
                current = slot == pos[0]
                marker = _s(_g("sel"), "accent", "bold") if current else " "
                label = _s(opt.label, "bold") if current else opt.label
                tag = ("  " + _s(opt.tag, "dim")) if opt.tag else ""
                out.append(f" {marker} {label}{tag}")
                if per_item == 2:
                    out.append("   " + (_s(opt.detail, "dim") if opt.detail else ""))
            hidden_below = len(vis) - (top[0] + rows)
            if hidden_below > 0:
                out.append(_s(f"  {_g('down')} {hidden_below} more", "dim"))
        else:
            out.append(_s("  " + empty_text, "dim"))
        out.append("")
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
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
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