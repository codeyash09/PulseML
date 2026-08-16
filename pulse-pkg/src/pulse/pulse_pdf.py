"""
pulse_pdf.py
============

Labeled PDF snapshot generation for Pulse CLI mode.

CLI mode never shows heatmaps on screen (see pulse_cli.py) -- matrices and
tensors are only "tagged" as text. If the user opts in during
`interactive_setup()`, this module is used instead to save an actual
labeled heatmap of the tensor to disk each step, one PDF per variable per
step, organized as:

    <output_dir>/<variable_name>/step000001.pdf
    <output_dir>/<variable_name>/step000002.pdf
    ...

Each PDF contains a title/header, the tensor's summary stats (shape,
dtype, backend, device, min/max/mean/std, nan/inf counts), and a
log-scale heatmap image of the tensor reduced to 2D.

This module is intentionally standalone -- it does NOT import from pulse.py
(which pulls in tkinter and other GUI-only dependencies that have no
business loading in a headless CLI/Colab/SSH session). It only depends on
pulse_backend.py, matplotlib (Agg backend, so no display is required), and
fpdf.
"""
from __future__ import annotations

import os
import time
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from fpdf import FPDF

from pulse.pulse_backend import to_numpy, tensor_kind, statistics, detect_backend


# ----------------------------------------------------------------------
# theme -- kept in sync with pulse.py's palette but duplicated locally so
# this module has zero dependency on pulse.py / tkinter.
# ----------------------------------------------------------------------

_BG = "#0a0a0c"
_BORDER = "#26262e"
_TEXT_DIM = "#98989f"

_COLORS = [
    (0.0, "#0b3d3a"),
    (0.25, "#14b8a6"),
    (0.5, "#0a0a0a"),
    (0.75, "#ffb020"),
    (1.0, "#ff5a1f"),
]
_CMAP = LinearSegmentedColormap.from_list("Heat", _COLORS)

_FIG_SIZE = (6.0, 4.2)
_FIG_DPI = 130


# ----------------------------------------------------------------------
# reduction: collapse any-ndim tensor down to a 2D array suitable for a
# heatmap, without the interactive axis picker available in GUI mode.
# Mirrors the *default* config in pulse.py (_default_config): first two
# axes shown, everything past that fixed at index 0.
# ----------------------------------------------------------------------

def _reduce_to_2d(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    idx = [slice(None), slice(None)] + [0] * (arr.ndim - 2)
    return arr[tuple(idx)]


def _render_heatmap_png(arr2d, path, var_name, step):
    safe = np.abs(arr2d.astype(np.float64)) + 1e-12

    # LogNorm needs finite, positive vmin/vmax. NaN/Inf entries are already
    # reported separately via backend.statistics() on the original tensor --
    # here they'd otherwise crash the norm (vmax=inf) or render as blank
    # gaps (nan), so clamp them to the finite range for display purposes
    # only, before computing bounds.
    finite = safe[np.isfinite(safe)]
    if finite.size:
        vmin, vmax = finite.min(), finite.max()
    else:
        vmin, vmax = 1e-12, 1.0
    safe = np.nan_to_num(safe, nan=vmin, posinf=vmax, neginf=vmin)

    fig, ax = plt.subplots(figsize=_FIG_SIZE, dpi=_FIG_DPI, facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.imshow(safe, cmap=_CMAP, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(_BORDER)
    ax.set_title(f"{var_name}  ·  step {step}", color=_TEXT_DIM, fontsize=10)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------

def generate_heatmap_pdf(var_name, value, step, output_dir="Pulse_Output"):
    """Render `value` (any Pulse-trackable array/tensor, any backend) as a
    labeled heatmap and save it as a one-page PDF at:

        <output_dir>/<var_name>/step<NNNNNN>.pdf

    Returns the path to the saved PDF.
    """
    var_dir = os.path.join(output_dir, var_name)
    os.makedirs(var_dir, exist_ok=True)

    arr = to_numpy(value)
    kind = tensor_kind(value)
    stats = statistics(value)
    arr2d = _reduce_to_2d(arr)

    # Render the heatmap to a temp PNG, then embed it in the PDF -- fpdf
    # can only place raster images from a file path, not directly from an
    # in-memory numpy array.
    fd, tmp_png = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        _render_heatmap_png(arr2d, tmp_png, var_name, step)

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()

        pdf.set_fill_color(10, 10, 12)  # matches _BG
        pdf.rect(0, 0, pdf.w, pdf.h, style="F")

        pdf.set_text_color(245, 245, 247)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Pulse", ln=1)

        pdf.set_text_color(152, 152, 159)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"{var_name}  -  step {step}", ln=1)
        pdf.cell(0, 6, time.strftime("%Y-%m-%d %H:%M:%S"), ln=1)
        pdf.ln(4)

        pdf.set_text_color(245, 245, 247)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Summary", ln=1)

        pdf.set_font("Courier", "", 10)
        pdf.set_text_color(200, 200, 205)
        rows = [
            ("backend", stats.get("backend")),
            ("device", stats.get("device")),
            ("kind", kind),
            ("shape", stats.get("shape")),
            ("dtype", stats.get("dtype")),
            ("min", stats.get("min")),
            ("max", stats.get("max")),
            ("mean", stats.get("mean")),
            ("std", stats.get("std")),
            ("nan", stats.get("nan")),
            ("inf", stats.get("inf")),
        ]
        for label, val in rows:
            formatted = f"{val:.6g}" if isinstance(val, float) else str(val)
            pdf.cell(0, 6, f"{label:<8} {formatted}", ln=1)

        if stats.get("nan") or stats.get("inf"):
            pdf.set_text_color(255, 77, 77)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "WARNING: NaN/Inf values present in this tensor", ln=1)

        pdf.ln(4)
        img_w = pdf.w - 20
        pdf.image(tmp_png, x=10, w=img_w)

        out_path = os.path.join(var_dir, f"step{step:06d}.pdf")
        pdf.output(out_path)
    finally:
        try:
            os.remove(tmp_png)
        except OSError:
            pass

    return out_path
