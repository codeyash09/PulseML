"""Sweep 2: pulse_ui.py and pulse_pdf.py.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_* are regression coverage for behaviour verified to work.
"""
import os
import sys
from collections import namedtuple

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_ui as UI  # noqa: E402

_Size = namedtuple("_Size", "columns lines")


# ---------------------------------------------------------------------------------------
# bugs
# ---------------------------------------------------------------------------------------

def test_bug_narrow_terminal_lines_are_clipped_to_40_columns(monkeypatch):
    """_width() is clamped to at least 40 columns, so in a narrower terminal (a tmux or
    IDE split pane, 30 columns) every line of ask()/choose() is clipped to 40, wraps, and
    the in-place redraw (which erases exactly `lines_drawn` rows) leaves stale rows
    behind on every keypress -- the bug the display-width clipping fix was about.
    Correct: never clip wider than the real terminal."""
    monkeypatch.setattr(UI.shutil, "get_terminal_size", lambda fallback=(80, 24): _Size(30, 24))
    assert UI._width() <= 30
    line = "x" * 80
    assert UI._visible_len(UI._clip(line, UI._width())) <= 30


def test_bug_complex_tensor_heatmap_drops_imaginary_part(tmp_path, monkeypatch):
    """statistics() now measures complex values by magnitude, but the PDF heatmap still
    does `np.abs(arr2d.astype(np.float64))`: the cast throws the imaginary part away
    first, so a purely imaginary tensor renders as all-zero (1e-12) and the image
    contradicts the stats printed above it. Correct: np.abs on the complex values."""
    pytest.importorskip("fpdf")
    from pulse import pulse_pdf as P
    seen = []
    real_lognorm = P.LogNorm

    def spy(vmin=None, vmax=None, **k):
        seen.append((vmin, vmax))
        return real_lognorm(vmin=vmin, vmax=vmax, **k)

    monkeypatch.setattr(P, "LogNorm", spy)
    z = (1j * np.full((4, 4), 3.0)).astype(np.complex64)
    P.generate_heatmap_pdf("phase", z, 1, output_dir=str(tmp_path))
    vmin, vmax = seen[-1]
    assert vmax == pytest.approx(3.0, rel=1e-3), (vmin, vmax)


def test_bug_distinct_variable_names_share_one_pdf_folder(tmp_path):
    """_safe_dir_name maps every non-word character to '_', so the tracked names
    'train/loss' and 'train_loss' (or 'w[0]' and 'w_0_') land in the same folder and
    each step's PDF of one silently overwrites the other's. Correct: an injective
    mapping (e.g. escape, or append a short hash when the name was changed)."""
    pytest.importorskip("fpdf")
    from pulse import pulse_pdf as P
    a = P.generate_heatmap_pdf("train/loss", np.ones((2, 2)), 1, output_dir=str(tmp_path))
    b = P.generate_heatmap_pdf("train_loss", np.zeros((2, 2)), 1, output_dir=str(tmp_path))
    assert os.path.abspath(a) != os.path.abspath(b)


# ---------------------------------------------------------------------------------------
# verified working
# ---------------------------------------------------------------------------------------

def test_ok_clip_measures_display_width():
    text = "\033[1m" + "损失" * 30 + "\033[0m"
    clipped = UI._clip(text, 40)
    assert UI._visible_len(clipped) <= 40
    assert UI._visible_len("é") == 1
    assert UI._clip("short", 40) == "short"


@pytest.mark.parametrize("secs,shown", [
    (0.0004, "0ms"), (0.9994, "999ms"), (0.9996, "1.0s"), (59.94, "59.9s"), (59.96, "1m 00s"),
    (61.4, "1m 01s"), (3599.6, "60m 00s"),
])
def test_ok_elapsed_text(secs, shown):
    assert UI.elapsed_text(secs) == shown


def test_ok_colour_rules(monkeypatch):
    class TTY:
        def isatty(self):
            return True

    monkeypatch.setattr(UI.sys, "stdout", TTY())
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("NO_COLOR", "")
    assert UI.color_enabled()
    monkeypatch.setenv("NO_COLOR", "1")
    assert not UI.color_enabled()
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert not UI.color_enabled()


def test_ok_pdf_latin1_and_sanitised_paths(tmp_path):
    pytest.importorskip("fpdf")
    from pulse import pulse_pdf as P
    out = P.generate_heatmap_pdf("../θ λ", np.arange(6, dtype=np.float32).reshape(2, 3), 2.0,
                                 output_dir=str(tmp_path))
    assert os.path.realpath(out).startswith(os.path.realpath(str(tmp_path)) + os.sep)
    assert out.endswith("step000002.pdf") and os.path.getsize(out) > 0
    out2 = P.generate_heatmap_pdf("x", np.array([np.nan, np.inf, 1.0]), 1.5, output_dir=str(tmp_path))
    assert os.path.basename(out2) == "step1.5.pdf"
