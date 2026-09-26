"""Sweep: pulse_ui.py -- capability detection, glyph fallback, clipping, ask()/choose()."""
import io
import sys
import types
import unicodedata

import pytest

from pulse import pulse_ui as ui


class _TTY(io.StringIO):
    def __init__(self, encoding="utf-8"):
        super().__init__()
        self._enc = encoding

    def isatty(self):
        return True

    @property
    def encoding(self):
        return self._enc


class _FakeKeys:
    """Stands in for ui._Keys: yields a scripted key sequence."""
    script = []

    def __enter__(self):
        self._it = iter(list(type(self).script))
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return next(self._it)


@pytest.fixture
def tty(monkeypatch):
    # pytest's capture re-binds sys.stdout between setup and call, so give pulse_ui its
    # own `sys` namespace instead of patching the real one.
    out = _TTY()
    monkeypatch.setattr(ui, "sys", types.SimpleNamespace(stdout=out, stdin=_TTY()))
    monkeypatch.delenv("PULSE_PLAIN", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(ui, "_UNICODE", None)
    return out


def _cols(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in ui._ANSI_RE.sub("", s))


# ============================================================================ bugs

def test_bug_clip_counts_wide_chars_as_one_column(tty, monkeypatch):
    """_clip() exists so a line 'never wraps -- a wrapped line would break the cursor
    arithmetic that redraws a screen in place'. It counts characters, not terminal
    columns, so CJK / emoji (workspace names, model names, labels) take 2 columns each
    and a 'clipped' line still wraps, corrupting every in-place redraw of choose()/ask().
    Correct: measure with unicodedata.east_asian_width (or wcwidth)."""
    monkeypatch.setattr(ui, "color_enabled", lambda: False)
    clipped = ui._clip("团队" * 40, 40)
    assert _cols(clipped) <= 40, _cols(clipped)


def test_bug_color_enabled_ignores_term_dumb(tty):
    """enabled() honours TERM=dumb, but color_enabled() doesn't: on a dumb terminal
    (Emacs shell, some CI consoles) that is still a tty, ok()/warn()/note() emit raw
    ANSI escape codes. Correct: color_enabled() should also return False for TERM=dumb."""
    import os
    os.environ["TERM"] = "dumb"   # restored by the fixture's monkeypatch.setenv
    assert ui.color_enabled() is False


def test_bug_empty_no_color_disables_color(tty, monkeypatch):
    """no-color.org: colour is disabled only when NO_COLOR is present *and not empty*.
    pulse_ui (and pulse_cli._COLOR_ENABLED) test `is None`, so NO_COLOR='' turns colour
    off, while pulse_console uses `not os.environ.get('NO_COLOR')` -- the two disagree
    inside one CLI. Correct: `not os.environ.get("NO_COLOR")` everywhere."""
    monkeypatch.setenv("NO_COLOR", "")
    assert ui.color_enabled() is True


def test_bug_elapsed_text_shows_60_seconds():
    """elapsed_text(59.96) renders '60.0s' instead of '1m 00s' (rounding after the
    <60 check). Correct: round first, then choose the unit."""
    assert ui.elapsed_text(59.96) == "1m 00s"


# ============================================================================ ok

def test_ok_enabled_detection(tty, monkeypatch):
    assert ui.enabled()
    monkeypatch.setenv("PULSE_PLAIN", "1")
    assert not ui.enabled()
    monkeypatch.delenv("PULSE_PLAIN")
    monkeypatch.setenv("TERM", "dumb")
    assert not ui.enabled()
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setattr(ui.sys, "stdin", io.StringIO())
    assert not ui.enabled()
    with pytest.raises(ui.Unavailable):
        ui.ask("x")
    with pytest.raises(ui.Unavailable):
        ui.choose([ui.Option("a")])


def test_ok_no_color_and_non_tty(tty, monkeypatch):
    assert ui.color_enabled()
    monkeypatch.setenv("NO_COLOR", "1")
    assert not ui.color_enabled()
    assert ui._s("x", "bold") == "x"


def test_ok_ascii_fallback_on_non_utf8_console(monkeypatch):
    monkeypatch.setattr(ui, "sys", types.SimpleNamespace(stdout=_TTY(encoding="cp1252"), stdin=_TTY()))
    monkeypatch.setattr(ui, "_UNICODE", None)
    assert ui._g("ok") == "+"
    assert ui._g("rule") == "-"
    monkeypatch.setattr(ui.sys, "stdout", _TTY(encoding="ascii"))
    monkeypatch.setattr(ui, "_UNICODE", None)
    ui.warn("hello")          # must not raise UnicodeEncodeError
    assert "hello" in ui.sys.stdout.getvalue() and "\u26a0" not in ui.sys.stdout.getvalue()


def test_ok_clip_ascii_and_ansi(tty, monkeypatch):
    monkeypatch.setattr(ui, "color_enabled", lambda: False)
    assert ui._clip("short", 40) == "short"
    c = ui._clip("\033[1m" + "x" * 100 + "\033[0m", 20)
    assert ui._visible_len(c) <= 20


def test_ok_ask_editing_and_secret(tty, monkeypatch):
    monkeypatch.setattr(ui, "_Keys", _FakeKeys)
    _FakeKeys.script = ["a", "c", "left", "b", "end", "d", "backspace", "home", "delete", "enter"]
    assert ui.ask("Label") == "bc"
    tty.truncate(0)
    tty.seek(0)
    _FakeKeys.script = list("hunter22") + ["enter"]
    assert ui.ask("Password", secret=True) == "hunter22"
    assert "hunter22" not in tty.getvalue()


def test_ok_ask_ctrl_c_propagates(tty, monkeypatch):
    class K(_FakeKeys):
        def read(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(ui, "_Keys", K)
    with pytest.raises(KeyboardInterrupt):
        ui.ask("x")


def test_ok_ask_validate_exception_is_ignored(tty, monkeypatch):
    monkeypatch.setattr(ui, "_Keys", _FakeKeys)
    _FakeKeys.script = ["x", "enter"]

    def bad(_t):
        raise RuntimeError("boom")

    assert ui.ask("L", validate=bad) == "x"


def test_ok_confirm_default(tty, monkeypatch):
    monkeypatch.setattr(ui, "_Keys", _FakeKeys)
    _FakeKeys.script = ["enter"]
    assert ui.confirm("Go?", default=False) is False
    _FakeKeys.script = ["y", "enter"]
    assert ui.confirm("Go?", default=False) is True


def test_ok_choose_navigation_shortcuts_hotkeys_search(tty, monkeypatch):
    monkeypatch.setattr(ui, "_Keys", _FakeKeys)
    opts = [ui.Option("Sign up", key="s"), ui.Option("Log in", key="l"), ui.Option("Recover", key="r")]
    _FakeKeys.script = ["down", "down", "down", "enter"]   # wraps around
    assert ui.choose(opts) == 0
    _FakeKeys.script = ["up", "enter"]
    assert ui.choose(opts) == 2
    _FakeKeys.script = ["L"]
    assert ui.choose(opts) == 1
    _FakeKeys.script = ["esc"]
    assert ui.choose(opts) is None
    _FakeKeys.script = ["d"]
    assert ui.choose(opts, hotkeys={"d": "d"}) == "d"
    _FakeKeys.script = list("reco") + ["enter"]
    assert ui.choose(opts, searchable=True) == 2
    _FakeKeys.script = list("zzz") + ["enter"]
    assert ui.choose(opts, searchable=True) is None
    assert ui.choose([]) is None


def test_ok_key_hint_for():
    assert "OpenRouter" in ui.key_hint_for("sk-or-v1-" + "a" * 40, "OPENROUTER_API_KEY")
    assert ui.key_hint_for("sk-or-v1-" + "a" * 40, "OPENAI_API_KEY") is None
    assert "Anthropic" in ui.key_hint_for("sk-ant-api03-" + "a" * 30, "ANTHROPIC_API_KEY")
    assert ui.key_hint_for("short", "ANTHROPIC_API_KEY") is None
    assert ui.commit_looks_valid(" abcdef1 ") and not ui.commit_looks_valid("xyz")


def test_ok_stage_marks_failure_and_propagates(tty, monkeypatch):
    monkeypatch.setattr(ui, "color_enabled", lambda: False)
    monkeypatch.setattr(ui, "_UNICODE", False)
    with pytest.raises(ValueError):
        with ui.Stage("Loading"):
            raise ValueError("x")
    assert "x Loading" in tty.getvalue()
