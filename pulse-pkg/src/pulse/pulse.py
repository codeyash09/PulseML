"""
Pulse — a live ML training debugger, GUI or CLI, any backend.

    from pulse import auto_track
    auto_track(train_step)   # pass your training function for shape discovery

GUI mode: opens a matrix picker with live shapes, then a live dashboard --
heatmap grid on the left (click a tile to enlarge, right-click to reconfigure
axes), AI chat on the right that's briefed on its role and can optionally see
your training code when you check "Send Code". Scalars (loss, accuracy, lr --
anything shape ()) render as a live line chart instead of a heatmap.
Loss-like scalars (named loss/cost/nll/cross_entropy/objective/err) are
auto-detected and pre-selected in the picker so you don't have to hunt for
them every run.

CLI mode: for Colab, SSH, or anywhere headless. No heatmaps are ever shown --
matrices/tensors are just "tagged" (stats printed each step); scalars get a
live ASCII chart. Optionally saves labeled PDF snapshots per variable per
step if you opt in at setup.

Backends: NumPy, PyTorch, TensorFlow, CuPy, JAX -- detected automatically via
pulse_backend.py. Pulse never checks for torch/tf/etc directly; it only ever
talks to that module.

Performance: Pulse is built to stay light. Every tensor it touches is
converted to a host-side NumPy array (never a GPU op), heatmap/line-chart
Matplotlib figures are created once per variable and reused (`set_data`)
rather than rebuilt from scratch every step, render sizes match the actual
on-screen thumbnail so nothing is drawn larger than needed, and stats are
computed without forcing unnecessary float64 copies of large tensors.

Robustness: individual tracked variables that turn out to be None, NaN-only,
or otherwise unreadable at a given step are reported as such in the
dashboard/manifest (and to the AI agent) instead of raising -- one bad
variable never takes down the background worker or the whole session.

Install:
    pip install numpy matplotlib pillow litellm --break-system-packages
    # tkinter ships with most Python installs; on Debian/Ubuntu:
    #   sudo apt install python3-tk
    # for CLI-mode PDF snapshots:
    pip install fpdf --break-system-packages   # not required if you never opt in

Enable the AI chat panel: set the relevant provider's API key as an env var
(e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY) -- or just leave it unset and Pulse
will prompt you for one the first time you send a message.
"""
import io
import os
import re
import sys
import json
import time
import uuid
import tempfile
import threading
import traceback
import sysconfig
import multiprocessing as mp
import ast
import base64
import math
import __main__
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import tkinter as tk
from tkinter import ttk, Toplevel, scrolledtext, simpledialog, messagebox
from PIL import Image, ImageTk

from pulse.pulse_backend import (
    to_numpy, shape_of, tensor_kind, is_trackable, describe_tensor,
    statistics as backend_statistics, scalar_value,
)

import multiprocessing as mp

import litellm


# ============================================================================
# theme: shared dark, "Pulse"-branded styling for every window in the app
# ============================================================================

BG = "#0a0a0c"          # app background
PANEL = "#111114"        # header / side panel background
CARD = "#17171c"         # tile / card background
CARD_HOVER = "#1e1e25"
BORDER = "#26262e"
TEXT = "#f5f5f7"
TEXT_DIM = "#98989f"
TEXT_FAINT = "#5c5c64"
ORANGE = "#ff5a1f"
AMBER = "#ffb020"
TEAL = "#14b8a6"

FIND_MATCH = "#4a3f14"     
FIND_MATCH_CUR = "#7a5f10"
RED = "#ff4d4d"

FONT_UI = ("Segoe UI", 10)
FONT_UI_BOLD = ("Segoe UI", 10, "bold")
FONT_HEAD = ("Segoe UI", 14, "bold")
FONT_SUBHEAD = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 9)
FONT_MONO_BOLD = ("Consolas", 9, "bold")


def apply_dark_theme(root):
    """Configure a shared dark ttk theme. Call once per Tk() root."""
    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=TEXT, font=FONT_UI, bordercolor=BORDER)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT, font=FONT_UI)
    style.configure("Dim.TLabel", background=BG, foreground=TEXT_DIM, font=FONT_UI)
    style.configure("Faint.TLabel", background=BG, foreground=TEXT_FAINT, font=FONT_MONO)
    style.configure("Head.TLabel", background=BG, foreground=TEXT, font=FONT_HEAD)
    style.configure("SectionHead.TLabel", background=BG, foreground=TEXT_DIM, font=("Segoe UI", 9, "bold"))

    style.configure("TLabelframe", background=BG, bordercolor=BORDER, relief="flat")
    style.configure("TLabelframe.Label", background=BG, foreground=TEXT_DIM, font=("Segoe UI", 9, "bold"))

    style.configure("TButton", background=CARD, foreground=TEXT, bordercolor=BORDER,
                     focusthickness=0, padding=8, font=FONT_UI, relief="flat")
    style.map("TButton", background=[("active", CARD_HOVER)], bordercolor=[("active", TEXT_FAINT)])

    style.configure("Accent.TButton", background=ORANGE, foreground="#0a0a0a",
                     bordercolor=ORANGE, padding=9, font=FONT_UI_BOLD, relief="flat")
    style.map("Accent.TButton", background=[("active", AMBER)], bordercolor=[("active", AMBER)])

    style.configure("TCheckbutton", background=BG, foreground=TEXT, font=FONT_UI, focuscolor=BG)
    style.map("TCheckbutton", foreground=[("active", ORANGE)])

    style.configure("TEntry", fieldbackground=CARD, foreground=TEXT, bordercolor=BORDER,
                     insertcolor=TEXT, padding=7, relief="flat")
    style.map("TEntry", bordercolor=[("focus", ORANGE)])

    style.configure("TPanedwindow", background=BG)
    style.configure("Sash", sashthickness=6, gripcount=0)

    style.configure("Vertical.TScrollbar", background=CARD, troughcolor=BG,
                     bordercolor=BG, arrowcolor=TEXT_DIM, relief="flat")
    style.map("Vertical.TScrollbar", background=[("active", CARD_HOVER)])

    return style


def _header_bar(root, subtitle):
    """A slim branded header: '● Pulse   subtitle', with a hairline underneath."""
    header = tk.Frame(root, bg=PANEL)
    header.pack(fill=tk.X, side=tk.TOP)
    inner = tk.Frame(header, bg=PANEL)
    inner.pack(fill=tk.X, padx=18, pady=14)

    dot = tk.Canvas(inner, width=9, height=9, bg=PANEL, highlightthickness=0)
    dot.create_oval(1, 1, 8, 8, fill=ORANGE, outline="")
    dot.pack(side=tk.LEFT, padx=(0, 9))

    tk.Label(inner, text="Pulse", bg=PANEL, fg=TEXT, font=FONT_HEAD).pack(side=tk.LEFT)
    tk.Label(inner, text=f"   {subtitle}", bg=PANEL, fg=TEXT_DIM, font=FONT_SUBHEAD).pack(side=tk.LEFT)

    hairline = tk.Frame(root, bg=BORDER, height=1)
    hairline.pack(fill=tk.X, side=tk.TOP)
    return header


# ============================================================================
# loss detection: name-based heuristic used to auto-select the loss/metric
# scalar in both the GUI picker and the CLI setup prompt, instead of making
# the user hunt for it among every other tracked scalar every run.
# ============================================================================

LOSS_NAME_HINTS = ("loss", "cost", "nll", "cross_entropy", "crossentropy", "objective", "err")


def _looks_like_loss(name):
    n = (name or "").lower()
    return any(hint in n for hint in LOSS_NAME_HINTS)


def _default_var_state(name, shape):
    """Loss-like scalars, and any scalar in general, default to full 'track'
    (cheap regardless). Everything else (matrices/tensors, or a shape we
    don't know yet) defaults to lightweight 'lotrack' -- this is what keeps
    tracking "everything" affordable by default. See Dashboard for how a
    variable gets promoted back to full tracking later, manually or via the
    agent's PROMOTE directive.
    """
    if _looks_like_loss(name):
        return "track"
    if shape == ():
        return "track"
    return "lotrack"


import math as _math_module


def _safe_eval_math(expr: str):
    """Evaluate a plain arithmetic/math expression deterministically -- LLMs
    are unreliable at exact arithmetic, so the agent can hand off anything
    like update magnitudes or ratios here instead of eyeballing it. Only
    numbers, operators, and `math` module names are reachable; no builtins,
    so this is safe to eval() directly.
    """
    allowed_names = {k: v for k, v in vars(_math_module).items() if not k.startswith("_")}
    try:
        return eval(expr, {"__builtins__": {}}, allowed_names)  # noqa: S307 -- restricted namespace above
    except Exception as exc:
        return f"(calc error: {exc})"


def _values_equal(a, b) -> bool:
    """NaN-safe / None-safe equality, used when deciding whether a scalar's
    value actually changed since the last recorded point. `nan != nan` is
    always True in plain Python, so without this a NaN (or None) scalar that
    repeats every step would get treated as "changed" every single step."""
    if a is None or b is None:
        return a is b
    try:
        if a != a and b != b:  # both NaN
            return True
    except TypeError:
        pass
    return a == b


# ============================================================================
# core: background heatmap/linechart-rendering worker process + stats
# ============================================================================

_COLORS = [
    (0.0, "#0b3d3a"),
    (0.25, "#14b8a6"),
    (0.5, "#0a0a0a"),
    (0.75, "#ffb020"),
    (1.0, "#ff5a1f"),
]
CMAP = LinearSegmentedColormap.from_list("Heat", _COLORS)

# Target number of points ever actually drawn on a scalar's line chart. The
# raw (step, value) history kept in `scalar_histories` is NEVER truncated --
# every point the user ever logged stays in memory for the life of the
# session -- this only bounds how many of those points get averaged down
# into a single rendered frame, since a 2.2x2.2in/80dpi thumbnail can't
# usefully show more than a few hundred points anyway.
TARGET_SCALAR_DISPLAY_POINTS = 120

# Render figures at roughly the size they'll actually be displayed at
# (dashboard thumbnails are ~170x170) instead of rendering large and then
# downscaling twice -- this alone meaningfully cuts per-frame CPU cost.
FIG_SIZE = (2.2, 2.2)
FIG_DPI = 80


def session_dir(session_id):
    d = os.path.join(tempfile.gettempdir(), "pulse_cache", session_id)
    os.makedirs(d, exist_ok=True)
    return d


def _default_config(ndim):
    if ndim >= 2:
        return [1, 1] + [0] * (ndim - 2)
    return [1] * ndim


def _axis_val_label(v):
    """Human-readable tag for an axis config value -- see _reduce_to_2d for
    the full encoding. 0 = fixed at index 0, 1 = kept/shown, 2+N = iterate
    mode currently parked on index N (driven by the axis slideshow)."""
    if v == 1:
        return "show"
    if v == 0:
        return "fix@0"
    if v >= 2:
        return f"iter@{v - 2}"
    return str(v)


def _reduce_to_2d(matrix, config):
    """config values per axis:
        0     = fix at index 0
        1     = keep this axis (visible in the 2D heatmap)
        2 + N = "iterate" mode, fixed at index N -- this is what powers the
                axis slideshow (Next/Previous bump N up/down and re-render
                at that exact index, e.g. flipping through attention heads
                one at a time instead of flattening them together).
    """
    arr = to_numpy(matrix)
    cfg = [int(v) for v in (config or _default_config(arr.ndim))]
    while len(cfg) < arr.ndim:
        cfg.append(0)
    cfg = cfg[:arr.ndim]

    idx = []
    for axis, v in enumerate(cfg):
        if v == 1:
            idx.append(slice(None))
        elif v >= 2:
            axis_len = arr.shape[axis]
            iterate_index = v - 2
            idx.append(min(iterate_index, axis_len - 1) if axis_len else 0)
        else:
            idx.append(0)
    reduced = arr[tuple(idx)]

    if reduced.ndim == 0:
        reduced = reduced.reshape(1, 1)
    elif reduced.ndim == 1:
        reduced = reduced.reshape(1, -1)
    elif reduced.ndim > 2:
        # Keep the first visible axis as rows and flatten the remaining visible
        # axes into columns so the heatmap always redraws as a true 2D image.
        reduced = reduced.reshape(reduced.shape[0], -1)
    return reduced


def _downsample_for_display(points, target=TARGET_SCALAR_DISPLAY_POINTS):
    """Bucket-average a (step, value) point list down to ~`target` points for
    rendering. This never mutates the original history.

    `points` may contain None values (unreadable that step) -- those are
    dropped before averaging so a single None doesn't turn an entire bucket
    into None/NaN; a bucket with no readable points at all is skipped.
    """
    n = len(points)
    cleaned = points if n <= target else points

    def _bucketize(pts):
        out = []
        bucket = max(1, math.ceil(len(pts) / target)) if len(pts) > target else 1
        for i in range(0, len(pts), bucket):
            chunk = pts[i:i + bucket]
            readable = [p for p in chunk if p[1] is not None and math.isfinite(p[1])]
            if not readable:
                continue
            avg_value = sum(p[1] for p in readable) / len(readable)
            avg_step = int(round(sum(p[0] for p in readable) / len(readable)))
            out.append((avg_step, avg_value))
        return out

    if n <= target:
        return _bucketize(list(points)) if any(p[1] is None or not math.isfinite(p[1]) for p in points) else list(points)
    return _bucketize(list(points))


def _atomic_write_json(path, payload, retries=12, delay=0.05):
    temp_path = path + ".tmp"
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    last_error = None
    for attempt in range(retries):
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(temp_path, path)
            return
        except (PermissionError, OSError) as exc:
            last_error = exc
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise

    if last_error is not None:
        raise last_error


def _save_heatmap(arr2d, path, var, fig_cache):
    """Render (or update) a variable's heatmap PNG.

    A Figure/Axes/AxesImage triple is created once per variable and cached
    in `fig_cache`; subsequent calls just push new data into the existing
    image via `set_data` instead of rebuilding the whole figure (axes,
    spines, colorbar, layout pass) from scratch every step. This is the
    single biggest CPU saving in Pulse -- Figure construction and layout is
    far more expensive than updating an existing artist's data.

    NaN/inf-safe: `LogNorm(vmin=nan, vmax=nan)` raises, so if a tensor has
    gone entirely NaN/inf this falls back to a flat placeholder range
    instead of crashing the worker process.
    """
    safe = np.abs(arr2d.astype(np.float64)) + 1e-12
    finite_mask = np.isfinite(safe)

    if finite_mask.any():
        vmin = float(np.min(safe[finite_mask]))
        vmax = float(np.max(safe[finite_mask]))
        if vmin == vmax:
            vmax = vmin + 1e-12
        # Replace non-finite cells with vmin so imshow has something valid
        # to draw everywhere; the log/scale is still driven by real data.
        safe = np.where(finite_mask, safe, vmin)
    else:
        safe = np.full_like(safe, 1e-12)
        vmin, vmax = 1e-12, 1.0

    entry = fig_cache.get(var)

    if entry is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI, facecolor=BG)
        ax.set_facecolor(BG)
        im = ax.imshow(safe, cmap=CMAP, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        fig.tight_layout(pad=0.4)
        fig_cache[var] = (fig, ax, im)
    else:
        fig, ax, im = entry
        im.set_data(safe)
        try:
            im.set_norm(LogNorm(vmin=vmin, vmax=vmax))
        except Exception:
            pass  # degenerate (all-equal) arrays -- keep the previous norm

    tmp = path + ".tmp.png"
    fig.savefig(tmp, facecolor=BG)
    os.replace(tmp, path)


def _save_linechart(points, path, var_name, fig_cache):
    """Renders a scalar's (step, value) point history -- already downsampled
    (and NaN/None-filtered) for display by the caller -- as a step chart
    (flat until the value actually changes, then jumps), which is the GUI
    equivalent of the CLI's ASCII chart. Same reuse-the-figure strategy as
    `_save_heatmap`.

    If nothing readable is left after filtering, draws an empty chart with
    a small "no readable data" label rather than failing.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    entry = fig_cache.get(var_name)

    if entry is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI, facecolor=BG)
        ax.set_facecolor(BG)
        line, = ax.plot(xs, ys, color=ORANGE, linewidth=1.4, drawstyle="steps-post")
        scatter = ax.scatter([], [], color=AMBER, s=14, zorder=3)
        ax.set_title(var_name, color=TEXT_DIM, fontsize=8)
        ax.tick_params(colors=TEXT_DIM, labelsize=6)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(color=BORDER, linewidth=0.4, alpha=0.5)
        if not points:
            ax.text(0.5, 0.5, "no readable data", color=TEXT_FAINT, fontsize=7,
                     ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout(pad=0.4)
        fig_cache[var_name] = (fig, ax, line, scatter)
    else:
        fig, ax, line, scatter = entry
        line.set_data(xs, ys)
        if points:
            ax.relim()
            ax.autoscale_view()

    if points:
        scatter.set_offsets([[xs[-1], ys[-1]]])
    else:
        scatter.set_offsets(np.empty((0, 2)))

    tmp = path + ".tmp.png"
    fig.savefig(tmp, facecolor=BG)
    os.replace(tmp, path)


def _worker_main(queue, display_configs, session_id, var_states=None):
    cache = session_dir(session_id)
    manifest_path = os.path.join(cache, "manifest.json")
    manifest = {}
    last_numpy_arrays = {}
    # var -> "track" (full stats + heatmap) or "lotrack" (stats only, no
    # heatmap ever generated -- this is what keeps tracking "everything"
    # affordable by default). Started from auto_track()'s initial picker
    # result and updated live via ("STATE", var, new_state) queue messages
    # sent by the Dashboard (right-click menu, or the agent's PROMOTE
    # directive).
    var_states = dict(var_states or {})

    # var -> full, NEVER-truncated list of (step, value) tuples. A new point
    # is only appended when the value actually differs from the last one
    # recorded, so a loss that's flat for 500 steps costs one point, not 500
    # -- the step chart drawstyle in _save_linechart fills in the flat
    # segments visually without needing a point at every step. `value` may
    # be None (variable was None or unreadable that step).
    scalar_histories = {}
    # Step numbering is now SHARED across every scalar rather than each
    # variable keeping its own independent counter: it only advances when
    # the loss/metric scalar (the first variable whose name looks like a
    # loss) actually changes value, so every chart's x-axis stays in sync
    # with real training progress instead of every scalar assignment. If no
    # loss-like variable is ever seen, this falls back to advancing whenever
    # ANY scalar changes -- the old per-call behavior, just on a shared
    # counter instead of per-variable ones.
    step_state = {"global_step": 0, "loss_var": None, "last_loss_value": None}
    # var -> bounded recent-values deque, mirrored into each scalar's
    # manifest entry as "recent" so the Dashboard (a separate process) can
    # do spike/divergence detection for auto-intervention without needing
    # the full never-truncated history shipped over.
    import collections as _collections
    recent_values = _collections.defaultdict(lambda: _collections.deque(maxlen=20))

    # Persistent Matplotlib figures, reused across steps instead of being
    # rebuilt every call -- see _save_heatmap / _save_linechart.
    heatmap_figs = {}
    linechart_figs = {}

    while True:
        item = queue.get()
        if item is None:
            break

        if isinstance(item, tuple) and item[0] == "STATE":
            _, var, new_state = item
            var_states[var] = new_state
            continue

        if isinstance(item, tuple) and item[0] == "CONFIG":
            _, var, new_config = item
            display_configs[var] = list(new_config)

            if var in last_numpy_arrays:
                try:
                    arr = last_numpy_arrays[var]
                    config = list(new_config) if new_config else _default_config(arr.ndim)
                    arr2d = _reduce_to_2d(arr, config)

                    version_tag = time.time_ns()
                    img_path = os.path.join(cache, f"{var}_{version_tag}.png")
                    _save_heatmap(arr2d, img_path, var, heatmap_figs)

                    stats = manifest.get(var, {})
                    stats["image"] = img_path
                    stats["updated"] = time.time()
                    manifest[var] = stats

                    _atomic_write_json(manifest_path, manifest)
                except Exception as e:
                    print(f"[PULSE WORKER CONFIG ERROR] {e}")
            continue

        var, matrix, config_override = item

        # A variable can legitimately be None (not yet assigned, or an
        # optional value that's currently unset). Report it plainly instead
        # of letting it reach to_numpy()/tensor_kind() and raise.
        if matrix is None:
            manifest[var] = {
                "kind": "scalar",
                "backend": "NoneType",
                "latest_value": None,
                "nan": 0,
                "inf": 0,
                "error": "NoneType",
                "updated": time.time(),
            }
            _atomic_write_json(manifest_path, manifest)
            continue

        try:
            kind = tensor_kind(matrix)
            if kind == "scalar":
                # Loss, accuracy, lr, or any other shape-() value: track a
                # rolling (step, value) history -- deduplicated so flat runs
                # don't cost a point per step -- and render it as a step
                # chart rather than a 1x1 "heatmap", which would be useless.
                try:
                    stats = backend_statistics(matrix)
                except Exception as exc:
                    stats = {"kind": "scalar", "backend": "unknown", "nan": 0, "inf": 0,
                              "error": f"{type(exc).__name__}: {exc}"}

                try:
                    # Derive `value` from the SAME statistics() call that produced
                    # the nan/inf flags above, instead of a second, independent
                    # scalar_value() conversion of the live tensor. Two separate
                    # conversions of a live (possibly GPU/async) tensor can
                    # disagree -- e.g. the second read racing an in-flight op --
                    # which was the cause of normal, finite loss values getting
                    # flagged as NaN/inf on the dashboard/chart. For a true
                    # scalar, mean over its single element is just that element,
                    # so this stays perfectly consistent with stats['nan']/
                    # stats['inf'].
                    raw_value = stats.get("mean")
                    value = float(raw_value) if raw_value is not None else None
                except Exception:
                    value = None

                # Advance the SHARED step counter only when the loss-like
                # scalar changes (once one has been identified); otherwise
                # (no loss-like var seen yet this session) fall back to
                # advancing on any scalar's change, same spirit as before.
                is_loss = _looks_like_loss(var)
                if is_loss and step_state["loss_var"] is None:
                    step_state["loss_var"] = var

                if step_state["loss_var"] == var:
                    if not _values_equal(step_state["last_loss_value"], value):
                        step_state["global_step"] += 1
                    step_state["last_loss_value"] = value
                elif step_state["loss_var"] is None:
                    prev_hist = scalar_histories.get(var, [])
                    prev_val = prev_hist[-1][1] if prev_hist else None
                    if not prev_hist or not _values_equal(prev_val, value):
                        step_state["global_step"] += 1

                # Per-variable dedup is unchanged: only append a new point
                # for THIS var if its own value actually differs from its
                # own last recorded value -- just tagged with the shared
                # step number instead of a private counter.
                hist = scalar_histories.setdefault(var, [])
                last_value = hist[-1][1] if hist else None
                if not hist or not _values_equal(last_value, value):
                    hist.append((step_state["global_step"], value))
                    recent_values[var].append(value)

                # For rendering, use the raw history (no synthetic extension),
                # filtering out None/NaN/inf points so the chart never has to
                # do math on them.
                display_points = _downsample_for_display(hist)
                version_tag = time.time_ns()
                img_path = os.path.join(cache, f"{var}_{version_tag}.png")
                _save_linechart(display_points, img_path, var, linechart_figs)

                stats["image"] = img_path
                stats["updated"] = time.time()
                stats["latest_value"] = value
                stats["recent"] = list(recent_values[var])
                manifest[var] = stats

            else:
                var_state = var_states.get(var, "lotrack")

                if var_state == "lotrack":
                    # Lightweight tracking: compute cheap stats only, never
                    # touch the heatmap pipeline (no to_numpy/reduce/imshow/
                    # savefig) -- this is what keeps tracking "everything"
                    # affordable by default. No "image" key at all, so the
                    # Dashboard knows to render this as a stats-only row.
                    stats = backend_statistics(matrix)
                    stats.pop("image", None)
                    stats["updated"] = time.time()
                    stats["state"] = "lotrack"
                    manifest[var] = stats
                else:
                    arr = to_numpy(matrix)
                    last_numpy_arrays[var] = arr

                    config = config_override or display_configs.get(var) or _default_config(arr.ndim)
                    arr2d = _reduce_to_2d(arr, config)

                    version_tag = time.time_ns()
                    img_path = os.path.join(cache, f"{var}_{version_tag}.png")
                    _save_heatmap(arr2d, img_path, var, heatmap_figs)

                    # pass the ORIGINAL tensor (not the numpy conversion) so
                    # backend/device detection stays accurate (torch/tf/etc),
                    # not flattened down to "NumPy" just because we converted
                    # it internally for slicing.
                    stats = backend_statistics(matrix)
                    stats["image"] = img_path
                    stats["updated"] = time.time()
                    stats["state"] = "track"
                    manifest[var] = stats
        except Exception as e:
            manifest[var] = {"error": str(e), "updated": time.time()}

        _atomic_write_json(manifest_path, manifest)


class HeatmapCreatorBG:
    def __init__(self, display_configs, session_id, var_states=None):
        self.session_id = session_id
        self.queue = mp.Queue()
        self.process = mp.Process(
            target=_worker_main,
            args=(self.queue, display_configs, session_id, var_states or {}),
            daemon=True,
        )
        self.process.start()

    def log_matrix(self, var, matrix, config_override=None):
        self.queue.put((var, matrix, config_override))

    def update_state(self, var, new_state):
        self.queue.put(("STATE", var, new_state))

    def update_config(self, var, new_config):
        self.queue.put(("CONFIG", var, new_config))

    def shutdown(self):
        self.queue.put(None)
        self.process.join(timeout=5)


# ============================================================================
# config_ui: initial matrix discovery + selection/axis picker
# ============================================================================

def discover_static_names_from_file(filepath):
    """Parse a single file's source with ast to find every assignment
    target anywhere in it -- including inside nested functions that haven't
    run yet (e.g. a `scores` matrix inside an `attention()` only called
    from within the training loop). Pure source parsing, doesn't execute
    anything, so it's safe to run before training starts. Works on any
    file, not just the entry script -- see _discover_project_files."""
    discovered = set()

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for t in node.targets:
                self.collect(t)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            self.collect(node.target)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            self.collect(node.target)
            self.generic_visit(node)

        def visit_For(self, node):
            self.collect(node.target)
            self.generic_visit(node)

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars:
                    self.collect(item.optional_vars)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            if node.name:
                discovered.add(node.name)
            self.generic_visit(node)

        def collect(self, target):
            if isinstance(target, ast.Name):
                discovered.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    self.collect(elt)

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf8") as f:
                tree = ast.parse(f.read(), filepath)
            Visitor().visit(tree)
        except Exception:
            pass

    noise = {"i", "j", "k", "_", "self", "cls", "e", "args", "kwargs"}
    return discovered - noise


def discover_static_names(caller_frame):
    """Back-compat wrapper: static discovery for just the caller's own file."""
    return discover_static_names_from_file(caller_frame.f_code.co_filename)


def _discover_project_files(entry_path, root=None, max_files=25):
    """Find other local (non-stdlib, non-site-packages) .py files this
    script imports, directly or transitively, so a modularized project's
    variables (e.g. defined inside a model.py/utils.py this script imports)
    can be offered as trackable candidates -- and included as code context
    for the agent -- even before that code has actually run once.

    Purely static: reads each file's `import`/`from ... import` statements
    and resolves them via importlib.util.find_spec (never executes
    anything). Capped by max_files so a huge codebase doesn't turn setup
    into a slow crawl, and stays within `root` if one was given, same as
    the runtime tracer already does.
    """
    if not entry_path or not os.path.exists(entry_path):
        return []

    import importlib.util

    entry_norm = os.path.normcase(os.path.abspath(entry_path))
    seen = {entry_norm}
    to_scan = [entry_path]
    found = []

    while to_scan and len(found) < max_files:
        path = to_scan.pop(0)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                tree = ast.parse(f.read(), path)
        except Exception:
            continue

        module_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    module_names.add(node.module.split(".")[0])

        for name in sorted(module_names):
            if not name or len(found) >= max_files:
                break
            try:
                spec = importlib.util.find_spec(name)
            except Exception:
                spec = None
            if spec is None or not spec.origin or not spec.origin.endswith(".py"):
                continue

            resolved = os.path.normcase(os.path.abspath(spec.origin))
            if resolved in seen:
                continue
            seen.add(resolved)

            if _is_library_frame(spec.origin):
                continue
            if root and not resolved.startswith(root):
                continue

            found.append(spec.origin)
            to_scan.append(spec.origin)

    return found


class AgentSetupDialog:
    """Minimal UI setup: pick a provider, enter its API key, done. Matches
    the CLI's zero-config default -- everything else (which variables to
    track, autofix, code-in-context) already has a sensible default and
    doesn't need a prompt. References the module-level PROVIDERS dict
    defined further down in this file (fine -- this only runs at
    auto_track() call time, well after the whole module has loaded).
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pulse")
        self.root.geometry("420x260")
        apply_dark_theme(self.root)
        _header_bar(self.root, "Set up your AI agent")

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        ttk.Label(body, text="PROVIDER", style="Faint.TLabel").pack(anchor="w")
        self.provider_var = tk.StringVar(value=list(PROVIDERS.keys())[0])
        dropdown = tk.OptionMenu(body, self.provider_var, *PROVIDERS.keys())
        dropdown.config(bg=CARD, fg=TEXT, activebackground=CARD_HOVER, activeforeground=TEXT,
                         relief="flat", highlightthickness=0, font=FONT_UI)
        dropdown["menu"].config(bg=CARD, fg=TEXT, activebackground=ORANGE, activeforeground="#0a0a0a")
        dropdown.pack(fill=tk.X, pady=(4, 14))

        ttk.Label(body, text="API KEY", style="Faint.TLabel").pack(anchor="w")
        self.key_var = tk.StringVar()
        entry = ttk.Entry(body, textvariable=self.key_var, show="*")
        entry.pack(fill=tk.X, pady=(4, 4))

        self.existing_note = ttk.Label(body, text="", style="Dim.TLabel", wraplength=370)
        self.existing_note.pack(anchor="w", pady=(0, 14))

        def _check_existing(*_):
            env_var = PROVIDERS[self.provider_var.get()]["env_key"]
            if os.environ.get(env_var):
                self.existing_note.config(text=f"{env_var} is already set -- leave blank to use it.")
            else:
                self.existing_note.config(text="")

        self.provider_var.trace_add("write", _check_existing)
        _check_existing()

        self.result = None

        def submit():
            provider = self.provider_var.get()
            key = self.key_var.get().strip()
            env_var = PROVIDERS[provider]["env_key"]
            if not key:
                key = os.environ.get(env_var, "").strip()
            if not key:
                messagebox.showerror("API key required", f"Enter an API key for {provider}, or set {env_var} first.")
                return
            self.result = (provider, key)
            self.root.destroy()

        entry.bind("<Return>", lambda e: submit())
        ttk.Button(body, text="Start  \u2192", style="Accent.TButton", command=submit).pack(fill=tk.X, pady=(4, 0))

    def run(self):
        self.root.mainloop()
        return self.result


class MatrixConfigUI:
    def __init__(self, matrices, on_config_change=None):
        self.root = tk.Tk()
        self.root.title("Pulse — Select Matrices")
        self.root.geometry("800x580")
        apply_dark_theme(self.root)
        _header_bar(self.root, "Select matrices to track")

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        self.discovered = matrices
        self.final_configs = {}
        self.matrix_vars = {}
        self.on_config_change = on_config_change  # callback(name, new_config) -- used by the Dashboard later
        self.auto_mode = tk.BooleanVar(value=True)
        self.result = None

        top = ttk.Frame(body)
        top.pack(fill=tk.X, pady=(0, 12))
        ttk.Checkbutton(
            top,
            text="Track everything automatically (recommended -- includes anything that appears later)",
            variable=self.auto_mode,
        ).pack(side=tk.LEFT)
        ttk.Button(top, text="Select All", command=self._select_all).pack(side=tk.RIGHT)

        left_frame = ttk.LabelFrame(
            body,
            text="  Variables  ·  everything is tracked by default -- you can add/remove any time, including from the dashboard later  ",
            padding=12,
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.search_var = tk.StringVar()
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(search_frame, text="SEARCH", style="Faint.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        search_entry.bind("<KeyRelease>", self._filter_matrix_list)

        canvas = tk.Canvas(left_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.list_frame = ttk.Frame(canvas)
        self.list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Loss-like scalars (loss/cost/nll/cross_entropy/objective/err) are
        # floated to the top and pre-checked, so the thing you almost always
        # want tracked doesn't require hunting through the full variable
        # list every run.
        self._matrix_checkbuttons = {}
        ordered_names = sorted(self.discovered.items(), key=lambda kv: (not _looks_like_loss(kv[0]), kv[0]))
        preselected = []
        for name, shape in ordered_names:
            self.final_configs[name] = _default_config(len(shape)) if shape else []
            is_loss = _looks_like_loss(name)
            var = tk.BooleanVar(value=is_loss)
            self.matrix_vars[name] = var
            if shape == ():
                shape_str = "   scalar (loss/metric \u2192 line chart)"
            elif shape:
                shape_str = f"   {tuple(shape)}"
            else:
                shape_str = "   (not run yet)"
            star = "   \u2605 loss?" if is_loss else ""
            cb = ttk.Checkbutton(
                self.list_frame,
                text=f"{name}{shape_str}{star}",
                variable=var,
            )
            self._matrix_checkbuttons[name] = cb
            cb.pack(anchor="w", pady=4)
            if is_loss:
                preselected.append(name)

        self._filter_matrix_list()

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=16)
        ttk.Button(btn_frame, text="Save & Run  →", style="Accent.TButton", command=self._submit).pack(fill=tk.X)

    def _filter_matrix_list(self, *_):
        query = (self.search_var.get() or "").strip().lower()
        for name in sorted(self._matrix_checkbuttons):
            cb = self._matrix_checkbuttons[name]
            visible = not query or query in name.lower()
            if visible:
                if not cb.winfo_ismapped():
                    cb.pack(anchor="w", pady=4)
            else:
                cb.pack_forget()

    def _select_all(self):
        for name, var in self.matrix_vars.items():
            var.set(True)


    def _submit(self):
        if self.auto_mode.get():
            self.result = {"auto": True, "vars": list(self.discovered.keys())}
        else:
            selected = {n: c for n, c in self.final_configs.items() if self.matrix_vars[n].get()}
            self.result = {"auto": False, "vars": selected}
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


# ============================================================================
# chat: AI assistant panel wired to live matrix stats, briefed on its role
# ============================================================================
SYSTEM_PROMPT = (
    "You are Pulse, an AI analyst embedded in a live ML training debugger. Your job is to find "
    "the root cause of instability in the user's training run, not to give generic ML advice.\n\n"
    "INPUTS EACH TURN:\n"
    "- Live stats per tracked matrix/tensor: backend, shape, min, max, mean, std, nan/inf counts.\n"
    "- Scalars (loss, accuracy, lr) as a running history with their latest value.\n"
    "- Heatmap images (log-scale, dark background) when available — look for banding, dead rows/"
    "columns, saturation, or regions breaking from the surrounding pattern.\n"
    "- Line-numbered training code, only on turns where 'Send Code' is checked.\n\n"
    "RESPONSE FORMAT (always, in this order):\n"
    "1. **Diagnosis** — one sentence, the specific root cause.\n"
    "2. **Reasoning** — grounded in the actual numbers/image you were given, expressed with real "
    "math. E.g. if gradients have std=142.7, show the update magnitude: $\\Delta w = \\eta \\cdot "
    "\\nabla L \\approx 0.01 \\times 142.7$, and explain why that blows up the weights. If it's a "
    "log(0) or division issue, write the actual expression that hits the singularity. Reference "
    "code by line number (e.g. `line 42`) when code was sent.\n"
    "3. **Fix** — a concrete change, not a generic suggestion.\n\n"
    "Ground every claim in the specific numbers or image you were actually given — 'gradients has "
    "std=142.7 and 340 inf values' beats 'you may have exploding gradients.' If code is available, "
    "point to the exact line; if it isn't, say what you'd need and that checking 'Send Code' would "
    "help. If nothing looks abnormal, say so rather than inventing a problem.\n\n"
    "Some variables may show latest_value=None, backend=NoneType, or an 'error' field instead of "
    "normal stats -- that means the variable is currently None or was unreadable that step, not "
    "that it's missing. Treat that as real signal (e.g. an optional loss term never getting set, "
    "or a value that already went NaN/inf and is now failing to convert) rather than ignoring it.\n\n"
    "Be concise. No preamble before the diagnosis.\n\n"

    "TOOLS (use inline, each on its own line within your Reasoning, only when actually useful):\n"
    "  CALC: <python arithmetic expression>\n"
    "    You are not reliable at exact arithmetic. Anything like an update magnitude, a ratio, "
    "or a comparison between two numbers you were given -- hand it off here instead of computing "
    "it yourself. Pulse evaluates it deterministically (only numbers, operators, and `math` module "
    "functions are available) and gives you the exact result. You may include more than one.\n"
    "  PROMOTE: <comma-separated variable names>\n"
    "    Some tracked matrices/tensors are in lightweight 'lotrack' mode (intermittent sampling, "
    "stats only, no heatmap) -- each variable's stats include its state. If one of them looks like "
    "it needs a closer look (heatmap + full stats), name it here and Pulse will switch it to full "
    "tracking. Only promote variables that are currently lotrack.\n\n"

    "CODE FIXES:\n"
    "If, and only if, the user explicitly asks you to fix, edit, patch, or change the code (not just "
    "diagnose it) AND training code has been sent this turn, respond with ONLY a single JSON object "
    "and nothing else -- no prose before or after it, no markdown code fences, no Diagnosis/Reasoning/"
    "Fix sections. The JSON object must have exactly these fields:\n"
    "  old: a list of code snippets to find, each copied EXACTLY from the line-numbered code shown "
    "to you, including original indentation and whitespace, but WITHOUT the line-number prefix "
    "('  12 | ') itself.\n"
    "  new: a list of the same length as old, where new[i] is the full replacement for old[i].\n"
    "  files: OPTIONAL, a list of the same length as old, where files[i] is the exact file header "
    "(e.g. \"model.py\") that old[i]/new[i] belongs to, if more than one file was sent this turn. "
    "Omit this field entirely (or use null/\"\" for an entry) to default to the main script.\n"
    "  explanation: a short, concise text description of what changed and why.\n"
    "Rules for old/new:\n"
    "  - Each snippet in old must appear VERBATIM and exactly ONCE in its target file. Include "
    "enough surrounding lines (not just the single changed line) so the match is unambiguous.\n"
    "  - Each new[i] is the complete replacement block for old[i] -- to add a line, copy old[i] and "
    "append the new line(s) to it; to remove a line, copy old[i] and omit it.\n"
    "  - Never use placeholders like '...' or '# unchanged' inside old or new; both must be literal, "
    "complete code.\n"
    "  - If the actual bug lives in another file that was sent this turn (e.g. a modularized "
    "project's model.py), fix it there via files[i] rather than working around it in the main "
    "script.\n"
    "  - If code wasn't sent this turn, or the user hasn't asked for a fix, do not emit this JSON "
    "format -- answer normally per RESPONSE FORMAT above, and if a fix was requested without code, "
    "say that checking 'Send Code' is needed first."
)

# Provider/model choices for the chat panel dropdown. Kept to models that
# are actually current and reachable via litellm as of mid-2026 -- pick
# whichever one you already have an API key for; Pulse will prompt for a
# key the first time you send a message on a provider that doesn't have one
# set as an env var yet.
PROVIDERS = {
    "Anthropic (Claude Sonnet 5)": {
        "model": "anthropic/claude-sonnet-5",
        "env_key": "ANTHROPIC_API_KEY",
        "prompt_title": "Anthropic API Key Required",
        "prompt_msg": "Please enter your Anthropic API Key (sk-ant-...):"
    },
    "Anthropic (Claude Opus 4.8)": {
        "model": "anthropic/claude-opus-4-8",
        "env_key": "ANTHROPIC_API_KEY",
        "prompt_title": "Anthropic API Key Required",
        "prompt_msg": "Please enter your Anthropic API Key (sk-ant-...):"
    },
    "Anthropic (Claude Haiku 4.5)": {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "env_key": "ANTHROPIC_API_KEY",
        "prompt_title": "Anthropic API Key Required",
        "prompt_msg": "Please enter your Anthropic API Key (sk-ant-...):"
    },
    "OpenAI (GPT-5.5)": {
        "model": "openai/gpt-5.5",
        "env_key": "OPENAI_API_KEY",
        "prompt_title": "OpenAI API Key Required",
        "prompt_msg": "Please enter your OpenAI API Key (sk-...):"
    },
    "OpenAI (GPT-5.4)": {
        "model": "openai/gpt-5.4",
        "env_key": "OPENAI_API_KEY",
        "prompt_title": "OpenAI API Key Required",
        "prompt_msg": "Please enter your OpenAI API Key (sk-...):"
    },
    "OpenAI (GPT-5.3 Codex)": {
        "model": "openai/gpt-5.3-codex",
        "env_key": "OPENAI_API_KEY",
        "prompt_title": "OpenAI API Key Required",
        "prompt_msg": "Please enter your OpenAI API Key (sk-...):"
    },
    "Google AI Studio (Gemini 3.1 Pro)": {
        "model": "gemini/gemini-3.1-pro-preview",
        "env_key": "GEMINI_API_KEY",
        "prompt_title": "Google AI Studio API Key Required",
        "prompt_msg": "Please enter your Google AI Studio API Key:"
    },
    "Google AI Studio (Gemini 3.6 Flash)": {
        "model": "gemini/gemini-3.6-flash",
        "env_key": "GEMINI_API_KEY",
        "prompt_title": "Google AI Studio API Key Required",
        "prompt_msg": "Please enter your Google AI Studio API Key:"
    },
    "Google AI Studio (Gemini 3.5 Flash-Lite)": {
        "model": "gemini/gemini-3.5-flash-lite",
        "env_key": "GEMINI_API_KEY",
        "prompt_title": "Google AI Studio API Key Required",
        "prompt_msg": "Please enter your Google AI Studio API Key:"
    },
    "DeepSeek": {
        "model": "deepseek/deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "prompt_title": "DeepSeek API Key Required",
        "prompt_msg": "Please enter your DeepSeek API Key:"
    },
    "Mistral": {
        "model": "mistral/mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
        "prompt_title": "Mistral API Key Required",
        "prompt_msg": "Please enter your Mistral API Key:"
    },
    "OpenRouter (Llama 3.3 70B, free)": {
        "model": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "env_key": "OPENROUTER_API_KEY",
        "prompt_title": "OpenRouter API Key Required",
        "prompt_msg": "Please enter your OpenRouter API Key (sk-or-...):"
    },
    "OpenRouter (GPT-OSS 120B, free)": {
        "model": "openrouter/openai/gpt-oss-120b:free",
        "env_key": "OPENROUTER_API_KEY",
        "prompt_title": "OpenRouter API Key Required",
        "prompt_msg": "Please enter your OpenRouter API Key (sk-or-...):"
    },
}

class _SpinnerLabel:
    """Tiny /-\\| spinner driven by Tk's `after()` loop, used in the chat
    panel to show which agent stage (Suggesting/Developing/Implementing
    fix) is currently in flight without blocking the UI thread.
    """

    _FRAMES = "/-\\|"

    def __init__(self, widget, label_fn):
        self.widget = widget
        self.label_fn = label_fn  # () -> text to show, or None to stop
        self._frame_idx = 0
        self._job = None

    def _tick(self):
        text = self.label_fn()
        if text is None:
            self.stop()
            return
        frame = self._FRAMES[self._frame_idx % len(self._FRAMES)]
        self._frame_idx += 1
        try:
            self.widget.configure(text=f"{text}... {frame}")
        except tk.TclError:
            return
        self._job = self.widget.after(120, self._tick)

    def start(self):
        self.stop()
        self._frame_idx = 0
        self._tick()

    def stop(self):
        if self._job is not None:
            try:
                self.widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        try:
            self.widget.configure(text="")
        except tk.TclError:
            pass


# Adaptive multi-pass agent pipeline (mirrors PulseCLI.ask_agent in
# pulse_cli.py). Instead of a fixed 3-call sequence, how many passes
# actually run adapts to what's asked and what comes back:
#   Pass 1 -- LOCATE:   read everything, identify the region(s) of the error.
#   Pass 2 -- ANALYZE:  a focused second read of just those regions, propose
#                        a fix (diagnosis + reasoning).
#   Pass 3 -- DEVELOP:  develop and implement the fix (code-fix JSON, if a
#                        code change was actually asked for).
#   Pass 4 -- VERIFY:   check the math/logic of the fix. Pass -> hand it to
#                        the user. Fail -> revise and re-check (bounded).
#   Pass 5 -- SWEEP:    re-read the whole thing again for OTHER, unrelated
#                        errors; if any turn up, ask the user whether to fix
#                        those too.
#   Pass 6:             if the user says yes, recurse through the same
#                        format (passes 1-5) for the newly-found issue(s).
# Each call posts to the transcript as soon as it's ready, with the spinner
# showing which pass is currently running.
_PASS1_LOCATE = (
    "PASS 1 -- LOCATE: Read through everything you were given (stats, image, code, history) and "
    "identify the specific region(s) where the problem likely originates -- file/line numbers, "
    "variable names, or code sections. Respond with ONLY a short bullet list of the suspect "
    "location(s). No diagnosis, no fix yet."
)
_PASS2_ANALYZE_TMPL = (
    "Suspect region(s) from your first read:\n{regions}\n\n"
    "PASS 2 -- ANALYZE: Take a focused second look at just those regions. Give the Diagnosis (one "
    "sentence, the specific root cause) and the Reasoning behind it (grounded in the actual "
    "numbers/image/code you were given, with real math, referencing line numbers). Do not "
    "implement the fix yet."
)
_PASS3_FIX_TEXT = (
    "PASS 3 -- DEVELOP: Give the Fix: a concrete, concise change (not generic advice), in 1-3 "
    "sentences."
)
_PASS3_IMPLEMENT = (
    "PASS 3 -- DEVELOP & IMPLEMENT: The user wants this fix applied to their code. Respond with "
    "ONLY the code-fix JSON object described in your instructions (old/new/explanation) -- no "
    "prose, no markdown fences."
)
_PASS4_VERIFY_TMPL = (
    "The fix you are about to apply:\n{fix_desc}\n\n"
    "PASS 4 -- VERIFY: Carefully check the math/logic of this fix against the numbers and code you "
    'were given. Respond with ONLY a JSON object of the form {{"passes": true or false, "reason": '
    '"one sentence"}}. passes=true only if the fix is logically/numerically correct and actually '
    "addresses the diagnosed root cause."
)
_PASS4_REVISE_TMPL = (
    "Your proposed fix did not pass verification: {reason}\n\n"
    "Revise it. Respond with ONLY the corrected code-fix JSON object (old/new/explanation) -- no "
    "prose, no markdown fences."
)
_PASS5_SWEEP = (
    "PASS 5 -- FULL RE-READ: Re-read the ENTIRE code/context again -- not just the region you just "
    "fixed -- and check for any OTHER, unrelated bugs or issues. Respond with ONLY a JSON object of "
    'the form {"other_errors_found": true or false, "summary": "short description, or empty string '
    'if none"}.'
)
_IMPLEMENT_KEYWORDS = ("fix", "edit", "patch", "change the code", "apply", "implement")
_MAX_VERIFY_ATTEMPTS = 3

_CALC_RE = re.compile(r"^\s*CALC:\s*(.+)$", re.MULTILINE)
_PROMOTE_RE = re.compile(r"^\s*PROMOTE:\s*(.+)$", re.MULTILINE)


def _extract_directives(text):
    """Pull CALC:/PROMOTE: lines out of an agent response, returning
    (cleaned_text, calc_exprs, promote_names). Cleaned text has those lines
    stripped so they don't clutter what's shown in the transcript.
    """
    calc_exprs = [m.strip() for m in _CALC_RE.findall(text) if m.strip()]
    promote_names = []
    for m in _PROMOTE_RE.findall(text):
        promote_names.extend(n.strip() for n in m.split(",") if n.strip())

    cleaned = _CALC_RE.sub("", text)
    cleaned = _PROMOTE_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, calc_exprs, promote_names


class ChatPanel(tk.Frame):
    def __init__(self, parent, get_manifest_fn, get_code_fn=None, script_path=None, on_code_change=None, promote_fn=None, get_extra_files_fn=None, initial_provider=None, restart_fn=None):
        super().__init__(parent, bg=PANEL)
        self.get_manifest_fn = get_manifest_fn
        self.get_code_fn = get_code_fn
        self.script_path = script_path
        self.on_code_change = on_code_change
        self.promote_fn = promote_fn  # (var_name) -> None, switches a lotrack var to full tracking
        self.get_extra_files_fn = get_extra_files_fn  # () -> {path: text} for other local project files
        self.restart_fn = restart_fn  # (provider_name) -> None, restarts the whole training process
        self._label_for_path = {}
        self._path_for_label = {}
        self.history = []
        self.session_keys = {}
        # Set whenever a code fix is actually written to disk during the
        # current top-level _ask() call -- checked once at the end of that
        # call to decide whether to trigger a restart (see _ask/PASS 3-6).
        self._fix_applied_this_turn = False

        head = tk.Frame(self, bg=PANEL)
        head.pack(fill=tk.X, padx=14, pady=(14, 6))

        self.dot = tk.Canvas(head, width=8, height=8, bg=PANEL, highlightthickness=0)
        self.dot_id = self.dot.create_oval(1, 1, 7, 7, fill=TEAL, outline="")
        self.dot.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(head, text="PULSE AI ANALYST", bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        self.provider_var = tk.StringVar(value=initial_provider or list(PROVIDERS.keys())[0])
        self.provider_dropdown = tk.OptionMenu(head, self.provider_var, *PROVIDERS.keys(), command=self._on_provider_change)
        self.provider_dropdown.config(bg=CARD, fg=TEXT, activebackground=CARD_HOVER, activeforeground=TEXT,
                                       relief="flat", highlightthickness=0, font=("Segoe UI", 9))
        self.provider_dropdown["menu"].config(bg=CARD, fg=TEXT, activebackground=ORANGE, activeforeground="#0a0a0a")
        self.provider_dropdown.pack(side=tk.RIGHT)

        self.status_label = tk.Label(head, text="", bg=PANEL, fg=AMBER, font=FONT_MONO)
        self.status_label.pack(side=tk.RIGHT, padx=(0, 10))
        self._spinner_text = {"value": None}
        self._spinner = _SpinnerLabel(self.status_label, lambda: self._spinner_text["value"])

        self.transcript = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, state="disabled", height=24,
            bg=CARD, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, padx=10, pady=10,
            font=FONT_UI, highlightthickness=1, highlightbackground=BORDER,
        )
        self.transcript.tag_configure("who_user", foreground=TEAL, font=FONT_UI_BOLD)
        self.transcript.tag_configure("who_ai", foreground=ORANGE, font=FONT_UI_BOLD)
        self.transcript.tag_configure("bold", font=FONT_UI_BOLD)
        self.transcript.tag_configure("heading", font=("Segoe UI", 11, "bold"), foreground=AMBER)
        self.transcript.tag_configure("inline_code", font=FONT_MONO, background=CARD_HOVER, foreground=TEAL)
        self.transcript.tag_configure("code_block", font=FONT_MONO, background=CARD_HOVER, lmargin1=10, lmargin2=10)
        self.transcript.tag_configure("math", font=("Cambria Math", 10), foreground=AMBER)
        self.transcript.tag_configure("find_match", background=FIND_MATCH)
        self.transcript.tag_configure("find_current", background=FIND_MATCH_CUR)
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        self._find_matches = []
        self._find_idx = -1
        self._find_frame = None
        self.transcript.bind("<Control-f>", lambda e: self._open_find_bar())
        self.bind_all("<Control-f>", lambda e: self._open_find_bar())

        entry_frame = tk.Frame(self, bg=PANEL)
        entry_frame.pack(fill=tk.X, padx=14, pady=(0, 14))

        send_btn = tk.Button(
            entry_frame, text="Send", command=self.send,
            bg=ORANGE, fg="#0a0a0a", activebackground=AMBER, activeforeground="#0a0a0a",
            relief="flat", font=FONT_UI_BOLD, padx=14, bd=0, cursor="hand2",
        )
        send_btn.pack(side=tk.RIGHT)

        self.send_code_var = tk.BooleanVar(value=True)
        code_check = tk.Checkbutton(
            entry_frame, text="Send Code", variable=self.send_code_var,
            bg=PANEL, fg=TEXT_DIM, selectcolor=CARD,
            activebackground=PANEL, activeforeground=TEXT,
            font=FONT_MONO, bd=0, highlightthickness=0, cursor="hand2",
        )
        code_check.pack(side=tk.RIGHT, padx=8)

        self.entry = tk.Entry(
            entry_frame, bg=CARD, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=FONT_UI, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ORANGE,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self.send())

        self._update_status_indicator()

    def _open_find_bar(self):
        if getattr(self, "_find_frame", None) is not None:
            self._find_entry.focus_set()
            return
        self._find_frame = tk.Frame(self, bg=PANEL)
        self._find_frame.pack(fill=tk.X, padx=14, pady=(0, 6))
        self._find_var = tk.StringVar()
        self._find_entry = tk.Entry(self._find_frame, textvariable=self._find_var, bg=CARD, fg=TEXT,
                                     insertbackground=TEXT, relief="flat", font=FONT_UI)
        self._find_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self._find_entry.bind("<Return>", lambda e: self._find_next())
        self._find_entry.bind("<KeyRelease>", lambda e: self._run_find())
        tk.Button(self._find_frame, text="↓", command=self._find_next, bg=CARD, fg=TEXT,
                  relief="flat", bd=0).pack(side=tk.LEFT, padx=2)
        tk.Button(self._find_frame, text="✕", command=self._close_find_bar, bg=CARD, fg=TEXT,
                  relief="flat", bd=0).pack(side=tk.LEFT, padx=2)
        self._find_entry.focus_set()

    def _close_find_bar(self):
        self.transcript.tag_remove("find_match", "1.0", tk.END)
        self.transcript.tag_remove("find_current", "1.0", tk.END)
        self._find_frame.destroy()
        self._find_frame = None

    def _run_find(self):
        query = self._find_var.get()
        self.transcript.tag_remove("find_match", "1.0", tk.END)
        self.transcript.tag_remove("find_current", "1.0", tk.END)
        self._find_matches = []
        self._find_idx = -1
        if not query:
            return
        start = "1.0"
        while True:
            pos = self.transcript.search(query, start, stopindex=tk.END, nocase=True)
            if not pos:
                break
            end = f"{pos}+{len(query)}c"
            self.transcript.tag_add("find_match", pos, end)
            self._find_matches.append((pos, end))
            start = end
        if self._find_matches:
            self._find_idx = 0
            self._goto_current_match()

    def _goto_current_match(self):
        self.transcript.tag_remove("find_current", "1.0", tk.END)
        if not self._find_matches:
            return
        pos, end = self._find_matches[self._find_idx]
        self.transcript.tag_add("find_current", pos, end)
        self.transcript.see(pos)

    def _find_next(self):
        if not self._find_matches:
            self._run_find()
            return
        self._find_idx = (self._find_idx + 1) % len(self._find_matches)
        self._goto_current_match()

    def _has_active_key(self, provider_name):
        
        env_var = PROVIDERS[provider_name]["env_key"]
        if env_var is None:
            return True
        return bool(os.environ.get(env_var) or self.session_keys.get(provider_name))

    def _update_status_indicator(self):
        has_key = self._has_active_key(self.provider_var.get())
        self.dot.itemconfig(self.dot_id, fill=TEAL if has_key else AMBER)

    def _on_provider_change(self, _selection):
        self._update_status_indicator()

    def _set_stage(self, label):
        """label=None stops the spinner and clears the status text; a string
        starts/updates it (e.g. 'Suggesting fix', 'Developing fix',
        'Implementing fix')."""
        self._spinner_text["value"] = label
        if label is None:
            self._spinner.stop()
        else:
            self._spinner.start()

    def _append(self, who, text):
        self.transcript.configure(state="normal")
        tag = "who_user" if who.startswith("You") else "who_ai"
        self.transcript.insert(tk.END, f"{who}\n", tag)
        self._insert_formatted(text)
        self.transcript.insert(tk.END, "\n\n")
        self.transcript.configure(state="disabled")
        self.transcript.see(tk.END)

    def _insert_formatted(self, text):
        """Minimal markdown-ish renderer: ```code blocks```, `inline code`,
        **bold**, and $math$/$$math$$ spans (rendered in a distinct color/
        font rather than literally, since Tk can't do real LaTeX)."""
        t = self.transcript
        lines = text.split("\n")
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                t.insert(tk.END, line + "\n", "code_block")
                continue
            if line.startswith("#"):
                stripped = line.lstrip("#").strip()
                t.insert(tk.END, stripped + "\n", "heading")
                continue

            # inline: **bold**, `code`, $math$
            pos = 0
            pattern = re.compile(r"(\*\*.+?\*\*|`.+?`|\$\$.+?\$\$|\$.+?\$)")
            for m in pattern.finditer(line):
                if m.start() > pos:
                    t.insert(tk.END, line[pos:m.start()])
                chunk = m.group(0)
                if chunk.startswith("**"):
                    t.insert(tk.END, chunk[2:-2], "bold")
                elif chunk.startswith("`"):
                    t.insert(tk.END, chunk[1:-1], "inline_code")
                elif chunk.startswith("$$"):
                    t.insert(tk.END, chunk[2:-2], "math")
                else:
                    t.insert(tk.END, chunk[1:-1], "math")
                pos = m.end()
            t.insert(tk.END, line[pos:] + "\n")

    def send(self):
        question = self.entry.get().strip()
        if not question:
            return

        current_provider = self.provider_var.get()
        prov_info = PROVIDERS[current_provider]
        env_var = prov_info["env_key"]

        if not self._has_active_key(current_provider):
            api_key = simpledialog.askstring(
                prov_info["prompt_title"], prov_info["prompt_msg"], parent=self, show='*'
            )
            if not api_key or not api_key.strip():
                return
            clean_key = api_key.strip()
            self.session_keys[current_provider] = clean_key
            os.environ[env_var] = clean_key
            self._update_status_indicator()
            self._append("Pulse", f"API key saved for {current_provider}. Send your message again.")
            return

        include_code = self.send_code_var.get()
        self.entry.delete(0, tk.END)
        self._append("You  (+ code)" if include_code else "You", question)

        threading.Thread(target=self._ask, args=(question, include_code, current_provider), daemon=True).start()

    def _build_file_labels(self, extra_files):
        """Give every file a short, unique display label (usually just its
        basename) used both in the code shown to the agent and later to
        resolve which real file a proposed fix's "file" field refers to.
        """
        label_for_path, path_for_label, used = {}, {}, set()

        def add(path):
            if not path or path in label_for_path:
                return
            base = os.path.basename(path)
            label = base
            if label in used:
                parent = os.path.basename(os.path.dirname(path))
                label = f"{parent}/{base}"
            used.add(label)
            label_for_path[path] = label
            path_for_label[label] = path

        add(self.script_path)
        for p in extra_files:
            add(p)

        self._label_for_path = label_for_path
        self._path_for_label = path_for_label

    def _build_context(self, include_code=False):
        manifest = self.get_manifest_fn() or {}
        context = "Current tracked matrix/scalar stats:\n"
        for name, s in manifest.items():
            if "error" in s:
                context += f"- {name}: NoneType/error reading matrix ({s['error']})\n"
                continue
            if s.get("kind") == "scalar":
                latest = s.get("latest_value")
                value_str = "None" if latest is None else str(latest)
                context += (
                    f"- {name}: scalar, backend={s.get('backend')}, latest_value={value_str}, "
                    f"nan={s.get('nan')} inf={s.get('inf')}\n"
                )
            else:
                state = s.get("state", "track")
                context += (
                    f"- {name}: shape={s.get('shape')} backend={s.get('backend')} kind={s.get('kind')} "
                    f"min={s.get('min')} max={s.get('max')} mean={s.get('mean')} std={s.get('std')} "
                    f"nan={s.get('nan')} inf={s.get('inf')} state={state}\n"
                )
        if include_code and self.get_code_fn:
            code = self.get_code_fn()
            extra_files = (self.get_extra_files_fn() or {}) if self.get_extra_files_fn else {}
            self._build_file_labels(extra_files)
            entry_label = self._label_for_path.get(self.script_path, "main_script.py")

            if code:
                numbered = "\n".join(f"{i+1:>4} | {line}" for i, line in enumerate(code.splitlines()))
                context += f"\n=== {entry_label} (main script, line-numbered) ===\n```\n{numbered}\n```\n"
            else:
                context += "\n(User checked 'Send Code' but no code text is available.)\n"

            if extra_files:
                context += (
                    "\nThis project is modularized -- other local files it imports are included "
                    "below, each line-numbered under its own header. When proposing a code fix, "
                    "set each fix's \"file\" to the exact header shown here (e.g. \"model.py\") so "
                    f"Pulse edits the right file. Omit \"file\" to default to {entry_label}.\n"
                )
                for path, text in extra_files.items():
                    label = self._label_for_path.get(path, os.path.basename(path))
                    numbered = "\n".join(f"{i+1:>4} | {line}" for i, line in enumerate(text.splitlines()))
                    context += f"\n=== {label} ===\n```\n{numbered}\n```\n"
        return context

    def _image_payloads(self):
        manifest = self.get_manifest_fn() or {}
        payloads = []
        for name, s in sorted(manifest.items()):
            img_path = s.get("image")
            if not img_path or not os.path.exists(img_path):
                continue
            try:
                with Image.open(img_path) as img:
                    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG", optimize=True)
                    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                payloads.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                })
            except Exception:
                pass
        return payloads

    def _call_model(self, model_name, instruction, image_payloads=None, max_tokens=2000):
        """One lightweight completion call: system prompt + recent history +
        a one-off stage instruction (+ images on the first call only, so
        they aren't re-uploaded on every stage). Does not touch
        self.history -- the caller decides what gets persisted once the
        whole pipeline finishes.
        """
        content = [{"type": "text", "text": instruction}]
        if image_payloads:
            content.extend(image_payloads)
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.history[-10:]
            + [{"role": "user", "content": content}]
        )
        try:
            response = litellm.completion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                timeout=120.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            return f"(request failed: {e})"

    def _apply_directives(self, calc_exprs, promote_names):
        """Deterministically compute any CALC: expressions and figure out
        which PROMOTE: names match a currently-known variable. Pure
        computation only -- no Tk calls -- so this is safe to run on the
        background request thread. Returns (note_text, calc_lines,
        promoted_names) for the caller to display/apply on the main thread.
        """
        calc_lines = ""
        if calc_exprs:
            computed = [(expr, _safe_eval_math(expr)) for expr in calc_exprs]
            calc_lines = "\n".join(f"  {expr} = {result}" for expr, result in computed)

        promoted = []
        if promote_names and self.promote_fn:
            manifest = self.get_manifest_fn() or {}
            known = list(manifest.keys())
            for raw_name in promote_names:
                name = raw_name.strip()
                if not name:
                    continue
                if name in known:
                    target = name
                else:
                    matches = [k for k in known if name.lower() in k.lower()]
                    target = matches[0] if len(matches) == 1 else None
                if target:
                    promoted.append(target)

        notes = []
        if calc_lines:
            notes.append(f"Pulse computed these deterministically -- use these exact values:\n{calc_lines}")
        if promoted:
            notes.append(f"Promoted to full tracking: {', '.join(promoted)}.")

        return "\n\n".join(notes), calc_lines, promoted

    def _ask_yes_no_blocking(self, title, message):
        """Show a yes/no dialog from the background request thread and
        block until answered, by scheduling the actual messagebox call on
        the Tk main thread (via after) and waiting on an Event."""
        result = {}
        event = threading.Event()

        def _show():
            try:
                result["value"] = messagebox.askyesno(title, message, parent=self)
            finally:
                event.set()

        self.after(0, _show)
        event.wait()
        return bool(result.get("value"))

    @staticmethod
    def _parse_json_obj(answer):
        """Generic defensive JSON-object parser for the verify (pass 4) and
        sweep (pass 5) responses -- same tolerance for stray code fences as
        _parse_code_fix, but without requiring any particular fields."""
        text = (answer or "").strip()
        if not text:
            return None
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
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _describe_fix(fix):
        parts = []
        for i, (old, new) in enumerate(zip(fix["old"], fix["new"])):
            parts.append(f"--- change {i + 1} ---\nOLD:\n{old}\nNEW:\n{new}")
        if fix.get("explanation"):
            parts.append(f"Explanation: {fix['explanation']}")
        return "\n\n".join(parts)

    def _verify_fix_with_retries(self, model_name, fix):
        """PASS 4: check the fix's math/logic before it's handed to the
        user. If it fails, ask the agent to revise and re-check, up to
        _MAX_VERIFY_ATTEMPTS times. Returns (fix, passed, reason)."""
        reason = ""
        for attempt in range(_MAX_VERIFY_ATTEMPTS):
            fix_desc = self._describe_fix(fix)
            self.after(0, lambda: self._set_stage("Checking the fix"))
            verify_answer = self._call_model(
                model_name, _PASS4_VERIFY_TMPL.format(fix_desc=fix_desc), max_tokens=200
            )
            verdict = self._parse_json_obj(verify_answer)
            if verdict is None:
                return fix, True, "(verification response was unparsable; proceeding anyway)"
            passes = bool(verdict.get("passes"))
            reason = str(verdict.get("reason", "")).strip()
            if passes:
                return fix, True, reason
            if attempt == _MAX_VERIFY_ATTEMPTS - 1:
                break
            self.after(0, lambda: self._set_stage("Revising fix"))
            revised_answer = self._call_model(
                model_name, _PASS4_REVISE_TMPL.format(reason=reason), max_tokens=4000
            )
            revised = self._parse_code_fix(revised_answer)
            if revised is None:
                break
            fix = revised
        return fix, False, (reason or "(verification did not clearly pass after retries)")

    def _run_sweep_and_maybe_recurse(self, model_name, include_code, provider_name, _depth):
        """PASS 5: re-read everything for OTHER, unrelated errors. If any
        turn up, ask the user whether to fix those too (PASS 6 recurses
        through the same 1-5 format for the new issue)."""
        self.after(0, lambda: self._set_stage("Reading for other errors"))
        sweep_answer = self._call_model(model_name, _PASS5_SWEEP, max_tokens=300)
        self.after(0, lambda: self._set_stage(None))
        sweep = self._parse_json_obj(sweep_answer)
        found = bool(sweep.get("other_errors_found")) if sweep else False
        summary = str(sweep.get("summary", "")).strip() if sweep else ""

        if not found or not summary:
            return

        self.after(0, lambda: self._append("Pulse (5 · Full re-read)", f"Found other possible issue(s):\n\n{summary}"))
        want_fix = self._ask_yes_no_blocking("Fix other errors?", f"{summary}\n\nFix these too?")
        if not want_fix:
            return

        self.after(0, lambda: self._append("Pulse", "(6) Following the established format for the additional issue(s)..."))
        self._ask(f"Please also fix this: {summary}", include_code, provider_name, _depth=_depth + 1)

    def _ask(self, question, include_code, provider_name, _depth=0):
        """Runs the question through an adaptive multi-pass pipeline
        instead of a fixed number of calls -- how many passes actually run
        depends on whether a code fix was asked for, whether it verifies
        cleanly, and whether a final sweep turns up anything else:

          1. LOCATE  -- read everything, identify the region(s) of the error.
          2. ANALYZE -- focused second read of those regions; diagnosis + reasoning.
          3. DEVELOP -- develop and implement the fix (code-fix JSON, if requested).
          4. VERIFY  -- check the fix's math/logic; revise and re-check on failure.
          5. SWEEP   -- re-read everything for OTHER errors; ask the user y/n.
          6.         -- if yes, recurse through 1-5 for the new issue(s).

        Each pass posts to the transcript as soon as it's ready, with the
        header spinner showing which pass is running. Only the top-level
        call (_depth == 0) restarts the training process afterward, once,
        if any fix landed anywhere in the (possibly recursive) chain.
        """
        if _depth == 0:
            self._fix_applied_this_turn = False

        model_name = PROVIDERS[provider_name]["model"]
        context = self._build_context(include_code=include_code)
        base_text = f"{context}\nQuestion: {question}"
        images = self._image_payloads()

        self.history.append({"role": "user", "content": [{"type": "text", "text": base_text}] + images})

        wants_implementation = include_code and any(
            kw in question.lower() for kw in _IMPLEMENT_KEYWORDS
        )

        # Pass 1: locate the region(s) of the error.
        self.after(0, lambda: self._set_stage("Reading for region of error"))
        regions = self._call_model(model_name, f"{base_text}\n\n{_PASS1_LOCATE}", images, max_tokens=200)
        self.after(0, lambda: self._append("Pulse (1 · Region of error)", regions))

        # Pass 2: focused second read + diagnosis/reasoning.
        self.after(0, lambda: self._set_stage("Analyzing"))
        raw_analysis = self._call_model(model_name, _PASS2_ANALYZE_TMPL.format(regions=regions), max_tokens=700)
        analysis, calc_exprs, promote_names = _extract_directives(raw_analysis)
        self.after(0, lambda: self._append("Pulse (2 · Diagnosis & reasoning)", analysis))

        directive_note = ""
        if calc_exprs or promote_names:
            directive_note, calc_lines, promoted = self._apply_directives(calc_exprs, promote_names)
            if calc_lines:
                self.after(0, lambda: self._append("Pulse (verified calculations)", calc_lines))
            for target in promoted:
                self.after(0, lambda t=target: self.promote_fn(t))
            if promoted:
                promoted_str = ", ".join(promoted)
                self.after(0, lambda s=promoted_str: self._append("Pulse", f"⚙ Promoted to full tracking (agent request): {s}"))
            if directive_note:
                self.history.append({"role": "user", "content": [{"type": "text", "text": directive_note}]})

        full_answer = f"{regions}\n\n{analysis}"

        if not wants_implementation:
            self.after(0, lambda: self._set_stage("Developing fix"))
            fix_text = self._call_model(model_name, _PASS3_FIX_TEXT, max_tokens=300)
            self.after(0, lambda: self._set_stage(None))
            full_answer += f"\n\n{fix_text}"
            self.after(0, lambda: self._append("Pulse (3 · Fix)", fix_text))
            self.history.append({"role": "assistant", "content": full_answer})
            return

        # Pass 3: develop and implement the fix.
        self.after(0, lambda: self._set_stage("Developing & implementing fix"))
        fix_answer = self._call_model(model_name, _PASS3_IMPLEMENT, max_tokens=4000)
        fix = self._parse_code_fix(fix_answer)
        if fix is None:
            self.after(0, lambda: self._set_stage(None))
            full_answer += f"\n\n{fix_answer}"
            self.after(0, lambda: self._append("Pulse (3 · Fix)", fix_answer))
            self.history.append({"role": "assistant", "content": full_answer})
            return

        # Pass 4: verify the fix's math/logic before handing it to the
        # user; revise and re-check on failure (bounded retries).
        fix, verify_ok, verify_reason = self._verify_fix_with_retries(model_name, fix)
        status = "passed" if verify_ok else "did not clearly pass -- applying best effort"
        self.after(0, lambda: self._append("Pulse (4 · Verification)", f"{status}: {verify_reason}"))

        self.after(0, lambda: self._set_stage(None))
        self.history.append({"role": "assistant", "content": full_answer})

        write_lines, applied_by_path, _originals = self._write_code_fix(fix)
        self.after(0, lambda: self._append("Pulse", write_lines))
        if applied_by_path:
            self._fix_applied_this_turn = True

        # Pass 5 (+ 6): only worth a full re-read if a fix actually landed.
        if applied_by_path and _depth < 3:
            self._run_sweep_and_maybe_recurse(model_name, include_code, provider_name, _depth)

        if _depth == 0 and self._fix_applied_this_turn and self.restart_fn:
            self.after(0, lambda: self._append("Pulse", "⚙ Restarting the training loop to pick up the fix..."))
            self.restart_fn(provider_name)

    @staticmethod
    def _parse_code_fix(answer):
        """If `answer` is a well-formed code-fix JSON payload, return it, else None."""
        text = (answer or "").strip()
        if not text:
            return None

        # Agents sometimes wrap JSON in ```json ... ``` fences despite being told not to.
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

        old, new, explanation = payload.get("old"), payload.get("new"), payload.get("explanation")
        files = payload.get("files")
        if not isinstance(old, list) or not isinstance(new, list):
            return None
        if not old or len(old) != len(new):
            return None
        if not all(isinstance(x, str) for x in old) or not all(isinstance(x, str) for x in new):
            return None
        if files is not None:
            if not isinstance(files, list) or len(files) != len(old):
                return None
            if not all(f is None or isinstance(f, str) for f in files):
                return None
        else:
            files = [None] * len(old)

        return {
            "old": old, "new": new, "files": files,
            "explanation": explanation if isinstance(explanation, str) else "",
        }

    def _resolve_fix_path(self, file_label):
        """Map a fix entry's optional "file" label back to a real path on
        disk, defaulting to the main script when unset. Falls back to
        substring matching (case-insensitive) since the agent may not
        reproduce a header exactly."""
        if not file_label or not file_label.strip():
            return self.script_path
        label = file_label.strip()
        if label in self._path_for_label:
            return self._path_for_label[label]
        matches = [p for lbl, p in self._path_for_label.items() if label.lower() in lbl.lower()]
        if len(matches) == 1:
            return matches[0]
        return None

    def _write_code_fix(self, fix):
        """Apply an agent-proposed code fix to the file(s) it targets,
        writing directly to disk. Pure I/O -- no Tk calls -- so this is
        safe to run on the background request thread (see _ask). Each fix
        entry may target a different file (see "files" in the code-fix
        schema, for a modularized project) -- edits are grouped by resolved
        file path so each file is read/written once regardless of how many
        snippets in it changed.

        Each old[i] must appear exactly once in its target file's current
        contents; snippets that don't match cleanly, or whose file can't be
        resolved, are skipped and reported rather than guessed at.

        Returns (display_text, applied_by_path, originals): applied_by_path
        maps touched file path -> [(old, new), ...] (empty if nothing
        landed); originals maps path -> its content before this edit, kept
        around in case a manual revert is ever needed.
        """
        by_path = {}
        unresolved = []
        for old, new, label in zip(fix["old"], fix["new"], fix["files"]):
            path = self._resolve_fix_path(label)
            if not path:
                unresolved.append((old, label))
                continue
            by_path.setdefault(path, []).append((old, new))

        if not by_path and not unresolved:
            return "(Proposed a code fix with nothing to apply.)", {}, {}

        lines = []
        if fix["explanation"]:
            lines.append(f"**Explanation:** {fix['explanation']}")

        originals = {}
        applied_by_path = {}
        skipped = []

        for path, pairs in by_path.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    original_content = f.read()
            except OSError as exc:
                for old, _new in pairs:
                    skipped.append((old, path, f"couldn't read file: {exc}"))
                continue

            content = original_content
            applied = []
            for old, new in pairs:
                count = content.count(old)
                if count == 1:
                    content = content.replace(old, new, 1)
                    applied.append((old, new))
                elif count == 0:
                    skipped.append((old, path, "no exact match found in the file"))
                else:
                    skipped.append((old, path, f"matched {count} times (ambiguous), skipped for safety"))

            if not applied:
                continue

            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as exc:
                lines.append(f"⚠ Failed to write changes to '{path}': {exc}")
                continue

            originals[path] = original_content
            applied_by_path[path] = applied
            if path == self.script_path and self.on_code_change:
                self.on_code_change(content)

        for old, label in unresolved:
            skipped.append((old, label or "(unspecified file)", "couldn't determine which file this targets"))

        if not applied_by_path:
            lines.append("\n⚠ No changes were applied -- none of the proposed snippets matched cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: `{old.splitlines()[0][:80]}...`")
            return "\n".join(lines), {}, {}

        for path, applied in applied_by_path.items():
            lines.append(f"\n✓ Applied {len(applied)} change(s) to `{os.path.basename(path)}`:")
            for old, new in applied:
                lines.append(f"  - replaced `{old.splitlines()[0][:80]}...` with `{new.splitlines()[0][:80]}...`")
        if skipped:
            lines.append(f"\n⚠ Skipped {len(skipped)} proposed change(s) that didn't match cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: `{old.splitlines()[0][:80]}...`")

        return "\n".join(lines), applied_by_path, originals

    def report_training_trouble(self, problem):
        """Called by the Dashboard when auto-intervention detects training
        going bad (a value went NaN/inf, or a loss-like scalar spiked) and
        has already paused the user's training loop via the control queue.
        Surfaces what was detected and, if an AI provider is already
        configured, automatically asks the agent to diagnose -- and if it
        can, fix -- it. Unlike report_script_error, the process is still
        alive and paused (not crashed), so if a fix gets applied the user
        can just hit "Resume Training" once they're ready.
        """
        self._append(
            "Pulse",
            f"⚠ Auto-paused training -- this looks like it's going bad:\n\n{problem}",
        )

        current_provider = self.provider_var.get()
        if not self._has_active_key(current_provider):
            self._append(
                "Pulse",
                "Pick a provider and enter an API key above, then ask me about this and "
                "I'll take a look.",
            )
            return

        question = (
            f"Pulse just auto-paused training because it detected a problem: {problem}\n"
            "Please diagnose the root cause and, if you can, fix it."
        )
        self._append("You (training auto-paused)", "(auto-reported by Pulse)")
        threading.Thread(target=self._ask, args=(question, True, current_provider), daemon=True).start()

    def report_script_error(self, tb_text):
        """Called by the Dashboard when it finds a crash file: the user's
        script raised an uncaught exception somewhere (an init error, or
        anything that happened after auto_track() was called) and the
        process has already exited. Surfaces the traceback and, if an AI
        provider is already configured, automatically asks the agent to
        diagnose -- and if it can, fix -- it, the same way a normal
        question would. The fix still can't un-crash the process that
        already exited, but it means the *next* run has a shot at working.
        """
        self._append(
            "Pulse",
            "⚠ Your script crashed with an uncaught exception (the process has already exited). "
            f"Here's the traceback:\n\n```\n{tb_text}\n```",
        )

        current_provider = self.provider_var.get()
        if not self._has_active_key(current_provider):
            self._append(
                "Pulse",
                "Pick a provider and enter an API key above, then ask me about this error and "
                "I'll take a look.",
            )
            return

        question = (
            f"My script just crashed with this uncaught exception:\n{tb_text}\n"
            "Please diagnose the root cause and, if you can, fix it."
        )
        self._append("You (script crashed)", "(auto-reported by Pulse)")
        threading.Thread(target=self._ask, args=(question, True, current_provider), daemon=True).start()


# ============================================================================
# dashboard: live heatmap/linechart grid (left) + chat (right)
# ============================================================================

class Dashboard:
    # Slower than "instant" on purpose -- loss/tensor stats don't need to
    # update faster than a human can read them, and every poll re-parses
    # manifest.json and re-opens changed PNGs, so a lighter cadence here
    # directly reduces steady-state CPU use.
    REFRESH_MS = 600
    THUMB_SIZE = (170, 170)
    COLS = 3

    def __init__(self, session_id, display_configs, code_text=None, config_queue=None, control_queue=None, known_names=None, script_path=None, var_states=None, extra_files=None, initial_provider=None):
        self.session_id = session_id
        self.cache = session_dir(session_id)
        self.manifest_path = os.path.join(self.cache, "manifest.json")
        self.crash_path = os.path.join(self.cache, "crash.json")
        self._crash_reported = False
        self.display_configs = display_configs
        self.config_queue = config_queue
        self.control_queue = control_queue
        self.code_text = code_text
        self.script_path = script_path
        self.extra_files = dict(extra_files or {})
        self.known_names = set(known_names or [])
        self.var_states = dict(var_states or {})
        self.initial_provider = initial_provider
        self._tiles = {}
        self._manifest = {}

        # Auto-intervention: watch tracked values for signs training is
        # going bad (a scalar going non-finite, or a loss-like scalar
        # spiking well above its recent range) and, if so, pause training
        # (via the same control_queue used for ADD_VAR) and automatically
        # ask the agent to diagnose -- and if it can, fix -- it.
        self.auto_intervene = tk.BooleanVar(value=True)
        self.explosion_multiplier = 5.0
        self.is_paused = False
        self._last_intervention_signature = None

        self.root = tk.Tk()
        self.root.title("Pulse — Live Dashboard")
        self.root.geometry("1280x800")
        apply_dark_theme(self.root)
        _header_bar(self.root, "Live Dashboard")

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(paned, bg=BG)
        right = tk.Frame(paned, bg=PANEL, width=360)
        paned.add(left, weight=3)
        paned.add(right, weight=1)

        left_head = tk.Frame(left, bg=BG)
        left_head.pack(fill=tk.X, padx=18, pady=(16, 4))
        tk.Label(left_head, text="TRACKED MATRICES", bg=BG, fg=TEXT_DIM,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(left_head, text="  \"lo\" tiles are lightweight (stats only) -- right-click to track fully, change axes, or remove", bg=BG, fg=TEXT_FAINT,
                 font=FONT_MONO).pack(side=tk.LEFT)
        tk.Checkbutton(
            left_head, text="Auto-fix", variable=self.auto_intervene,
            bg=BG, fg=TEXT_DIM, selectcolor=CARD, activebackground=BG, activeforeground=TEXT,
            font=FONT_MONO, bd=0, highlightthickness=0, cursor="hand2",
        ).pack(side=tk.RIGHT)

        self.pause_banner = tk.Frame(left, bg=RED)
        pause_inner = tk.Frame(self.pause_banner, bg=RED)
        pause_inner.pack(fill=tk.X, padx=18, pady=8)
        self.pause_label = tk.Label(pause_inner, text="", bg=RED, fg="#0a0a0a", font=FONT_UI_BOLD, anchor="w")
        self.pause_label.pack(side=tk.LEFT)
        tk.Button(
            pause_inner, text="▶ Resume Training", command=self._resume_training,
            bg="#0a0a0a", fg=TEXT, activebackground=CARD_HOVER, activeforeground=TEXT,
            relief="flat", font=FONT_UI_BOLD, padx=10, pady=4, bd=0, cursor="hand2",
        ).pack(side=tk.RIGHT)
        # Not packed yet -- _poll() packs/unpacks it as is_paused changes.

        self.canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.grid_frame = tk.Frame(self.canvas, bg=BG)
        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=18, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._tile_outer_w = self.THUMB_SIZE[0] + 44

        self.chat_panel = ChatPanel(
            right,
            get_manifest_fn=lambda: self._manifest,
            get_code_fn=lambda: self.code_text,
            script_path=self.script_path,
            on_code_change=self._on_code_change,
            promote_fn=self._promote_to_track,
            get_extra_files_fn=lambda: self.extra_files,
            initial_provider=self.initial_provider,
            restart_fn=self._restart_training,
        )
        self.chat_panel.pack(fill=tk.BOTH, expand=True)

        self.hidden_tiles = set()
        self.pending_tiles = set() 
        add_bar = tk.Frame(left, bg=BG)
        add_bar.pack(fill=tk.X, padx=18, pady=(0, 10))
        add_btn = tk.Button(
            add_bar, text="+  Add Matrix", command=self._open_add_heatmap_popup,
            bg=CARD, fg=TEXT, activebackground=CARD_HOVER, activeforeground=TEXT,
            relief="flat", font=FONT_UI_BOLD, padx=14, pady=7, bd=0, cursor="hand2",
            highlightthickness=1, highlightbackground=BORDER,
        )
        add_btn.pack(anchor="w")

        self._poll()

    def _on_code_change(self, new_code_text):
        """Called by the chat panel after it applies or reverts a code fix on
        disk, so subsequent turns (and 'Send Code') use the up-to-date text."""
        self.code_text = new_code_text

    def _load_manifest(self):
        for _ in range(10):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
            except (PermissionError, OSError):
                time.sleep(0.05)
        return {}

    def _poll(self):
        if not self._crash_reported and os.path.exists(self.crash_path):
            self._crash_reported = True
            try:
                with open(self.crash_path, "r", encoding="utf-8") as f:
                    crash = json.load(f)
                self.chat_panel.report_script_error(crash.get("traceback", "(no traceback captured)"))
            except Exception:
                pass

        manifest = self._load_manifest()
        self._manifest = manifest

        if self.auto_intervene.get() and not self.is_paused:
            problem = self._check_for_trouble(manifest)
            if problem and problem != self._last_intervention_signature:
                self._last_intervention_signature = problem
                self._trigger_auto_intervention(problem)

        for name, stats in sorted(manifest.items()):
            if name in self.hidden_tiles:
                continue
            if "error" in stats:
                self.pending_tiles.discard(name)
                continue

            img_path = stats.get("image")
            is_scalar = stats.get("kind") == "scalar"

            if not img_path and not is_scalar:
                # lotrack: stats-only, no heatmap was ever generated for this
                # variable -- render a lightweight text tile instead.
                self.pending_tiles.discard(name)
                self._render_lotrack_tile(name, stats)
                continue

            if not img_path or not os.path.exists(img_path):
                continue

            tile = self._tiles.get(name)
            current_img_path = img_path

            if tile is not None and tile.get("kind") == "lotrack":
                # Just got promoted and now has a real image -- tear down the
                # old text-only tile so it can be rebuilt as an image tile.
                try:
                    tile["frame"].destroy()
                except Exception:
                    pass
                tile = None
                self._tiles.pop(name, None)

            if tile is not None and tile.get("img_path") == current_img_path:
                continue

            img = Image.open(current_img_path)
            thumb = img.copy()
            thumb.thumbnail(self.THUMB_SIZE)
            photo = ImageTk.PhotoImage(thumb)

            nan = stats.get("nan", 0) or 0
            inf = stats.get("inf", 0) or 0
            is_flagged = bool(nan or inf)
            border_color = RED if is_flagged else BORDER

            if tile is None:
                self.pending_tiles.discard(name) 
                frame = tk.Frame(self.grid_frame, bg=CARD, highlightbackground=border_color,
                                  highlightthickness=1, bd=0)
                inner = tk.Frame(frame, bg=CARD)
                inner.pack(padx=10, pady=10)
                label = tk.Label(inner, image=photo, bg=CARD, cursor="hand2")
                label.image = photo
                label.pack()
                caption = tk.Label(inner, text=name, bg=CARD, fg=TEXT, font=FONT_UI_BOLD, anchor="w")
                caption.pack(fill=tk.X, pady=(8, 0))
                sub = tk.Label(inner, text="", bg=CARD, fg=TEXT_DIM, font=FONT_MONO, anchor="w")
                sub.pack(fill=tk.X)

                def on_click(event, n=name):
                    tile_obj = self._tiles.get(n)
                    if tile_obj is not None:
                        self._enlarge(n, tile_obj["img_path"])

                label.bind("<Button-1>", on_click)
                label.bind("<Button-3>", self._make_tile_context_menu(name))

                tile = {"frame": frame, "label": label, "sub": sub, "img_path": current_img_path, "kind": "track"}
                self._tiles[name] = tile
            else:
                tile["frame"].configure(highlightbackground=border_color)
                tile["label"].configure(image=photo)
                tile["label"].image = photo
                tile["img_path"] = current_img_path
                tile["kind"] = "track"
                tile["label"].unbind("<Button-1>")
                tile["label"].bind("<Button-1>", lambda e, n=name: self._enlarge(n, self._tiles[n]["img_path"]))
                tile["label"].unbind("<Button-3>")
                tile["label"].bind("<Button-3>", self._make_tile_context_menu(name))

            if is_scalar:
                latest = stats.get("latest_value")
                if latest is None:
                    latest_str = "NoneType"
                elif isinstance(latest, (int, float)) and math.isfinite(latest):
                    latest_str = f"{latest:.4f}"
                else:
                    latest_str = "NaN/inf"
                tile["sub"].configure(text=f"value={latest_str}", fg=TEXT_DIM)
            else:
                flag = "  \u26a0 flagged" if is_flagged else "  nominal"
                flag_color = RED if is_flagged else TEXT_DIM
                tile["sub"].configure(text=f"nan={nan}  inf={inf}{flag}", fg=flag_color)

        for name in list(self._tiles.keys()):
            if name not in manifest or name in self.hidden_tiles:
                tile = self._tiles.pop(name, None)
                if tile is not None:
                    try:
                        tile["frame"].destroy()
                    except Exception:
                        pass

        self._relayout()
        self.root.after(self.REFRESH_MS, self._poll)

    def _check_for_trouble(self, manifest):
        """Look at the current manifest for signs training is going bad.
        Returns a short human-readable description, or None. Mirrors the
        CLI's PulseCLI._check_for_trouble: non-finite scalar values are an
        unambiguous trigger; a loss-like scalar spiking to
        explosion_multiplier-x its own recent minimum is a softer one.
        Also flags any matrix/tensor (track or lotrack) whose latest stats
        show nan/inf.
        """
        reasons = []
        for name, stats in manifest.items():
            if "error" in stats:
                continue
            if stats.get("kind") == "scalar":
                latest = stats.get("latest_value")
                if latest is not None and isinstance(latest, (int, float)) and not math.isfinite(latest):
                    reasons.append(f"'{name}' just went non-finite (NaN/inf): {latest}")
                    continue
                if _looks_like_loss(name) and latest is not None:
                    recent = [v for v in (stats.get("recent") or []) if v is not None and math.isfinite(v)]
                    if len(recent) >= 5:
                        baseline = min(recent[:-1])
                        if baseline > 0 and latest > baseline * self.explosion_multiplier:
                            reasons.append(
                                f"'{name}' spiked to {latest:.4g}, "
                                f"{latest / baseline:.1f}x its recent minimum ({baseline:.4g})"
                            )
            else:
                nan = stats.get("nan", 0) or 0
                inf = stats.get("inf", 0) or 0
                if nan or inf:
                    reasons.append(f"'{name}' has nan={nan} inf={inf}")
        return "; ".join(reasons) if reasons else None

    def _trigger_auto_intervention(self, problem):
        """Pause the user's training loop (via control_queue -- see
        persistent_tracer's PAUSE/RESUME handling in pulse.py) and
        automatically hand the problem to the agent to diagnose and, if it
        can, fix -- without waiting for the user to notice and ask.
        """
        self.is_paused = True
        if self.control_queue is not None:
            try:
                self.control_queue.put(("PAUSE", None))
            except Exception:
                pass
        self.pause_label.configure(
            text=f"⚠ Auto-paused: {problem}"
        )
        if not self.pause_banner.winfo_ismapped():
            self.pause_banner.pack(fill=tk.X, side=tk.TOP, before=self.canvas)
        self.chat_panel.report_training_trouble(problem)

    def _resume_training(self):
        self.is_paused = False
        self._last_intervention_signature = None
        if self.control_queue is not None:
            try:
                self.control_queue.put(("RESUME", None))
            except Exception:
                pass
        if self.pause_banner.winfo_ismapped():
            self.pause_banner.pack_forget()

    def _restart_training(self, provider_name):
        """Called by the chat panel right after a code fix has been
        applied to disk, so the training loop actually runs the fixed
        code -- the process that owns the loop is still executing the old
        code in memory otherwise. This process (the Dashboard) doesn't own
        the loop itself, so it just signals the trainer process via
        control_queue; see auto_track()'s persistent_tracer for the
        RESTART handler, which tears everything down and os.execv's a
        fresh process, threading the active provider through so the next
        run's setup auto-fills the agent and API key instead of asking
        again.
        """
        if self.control_queue is not None:
            try:
                self.control_queue.put(("RESTART", provider_name))
            except Exception:
                pass

    def _fmt_num(self, v):
        try:
            return f"{v:.4f}"
        except (TypeError, ValueError):
            return "n/a"

    def _render_lotrack_tile(self, name, stats):
        """Stats-only tile for a 'lotrack' variable: no heatmap was ever
        generated for it, so there's nothing to render but the numbers.
        Right-click still offers "Track fully" to promote it.
        """
        mean_v = stats.get("mean")
        nan = stats.get("nan", 0) or 0
        inf = stats.get("inf", 0) or 0
        is_flagged = bool(nan or inf)
        border_color = RED if is_flagged else BORDER
        text = f"mean={self._fmt_num(mean_v)}\nnan={nan}  inf={inf}"

        tile = self._tiles.get(name)
        if tile is None or tile.get("kind") != "lotrack":
            if tile is not None:
                try:
                    tile["frame"].destroy()
                except Exception:
                    pass
            frame = tk.Frame(self.grid_frame, bg=CARD, highlightbackground=border_color,
                              highlightthickness=1, bd=0, width=self.THUMB_SIZE[0], height=self.THUMB_SIZE[1])
            frame.pack_propagate(False)
            inner = tk.Frame(frame, bg=CARD)
            inner.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            tk.Label(inner, text="lo", bg=CARD, fg=TEXT_FAINT, font=FONT_MONO_BOLD, anchor="w").pack(anchor="w")
            caption = tk.Label(inner, text=name, bg=CARD, fg=TEXT, font=FONT_UI_BOLD, anchor="w", wraplength=self.THUMB_SIZE[0] - 20)
            caption.pack(fill=tk.X, pady=(6, 4), anchor="w")
            sub = tk.Label(inner, text=text, bg=CARD, fg=(RED if is_flagged else TEXT_DIM),
                            font=FONT_MONO, anchor="w", justify="left")
            sub.pack(fill=tk.X, anchor="w")

            frame.bind("<Button-3>", self._make_tile_context_menu(name))
            inner.bind("<Button-3>", self._make_tile_context_menu(name))
            for w in inner.winfo_children():
                w.bind("<Button-3>", self._make_tile_context_menu(name))

            self._tiles[name] = {"frame": frame, "sub": sub, "img_path": None, "kind": "lotrack"}
        else:
            tile["frame"].configure(highlightbackground=border_color)
            tile["sub"].configure(text=text, fg=(RED if is_flagged else TEXT_DIM))

    def _relayout(self):
        panel_width = self.canvas.winfo_width()
        cols = max(1, panel_width // self._tile_outer_w) if panel_width > 1 else self.COLS
        self.COLS = cols
        for i, name in enumerate(sorted(self._tiles.keys())):
            self._tiles[name]["frame"].grid(row=i // cols, column=i % cols, padx=7, pady=7)

    def _on_canvas_resize(self, event):
        new_cols = max(1, event.width // self._tile_outer_w)
        if new_cols != self.COLS:
            self.COLS = new_cols
            self._relayout()

    def _promote_to_track(self, name):
        """Switch a lotrack variable to full tracking -- called from the
        right-click menu, or by the agent via a PROMOTE: directive."""
        self.var_states[name] = "track"
        if self.config_queue is not None:
            try:
                self.config_queue.put(("STATE", name, "track"))
            except Exception:
                pass
        self.root.after(0, self._poll)

    def _demote_to_lotrack(self, name):
        self.var_states[name] = "lotrack"
        if self.config_queue is not None:
            try:
                self.config_queue.put(("STATE", name, "lotrack"))
            except Exception:
                pass
        tile = self._tiles.pop(name, None)
        if tile is not None:
            try:
                tile["frame"].destroy()
            except Exception:
                pass
        self.root.after(0, self._poll)

    def _make_tile_context_menu(self, name):
        def handler(event):
            stats = self._manifest.get(name, {})
            is_scalar = stats.get("kind") == "scalar"
            state = self.var_states.get(name) or ("track" if is_scalar else "lotrack")
            menu = tk.Menu(self.root, tearoff=0, bg=CARD, fg=TEXT, activebackground=ORANGE,
                            activeforeground="#0a0a0a", bd=0, relief="flat")
            if not is_scalar:
                if state == "lotrack":
                    menu.add_command(label="Track fully (stats + heatmap)", command=lambda: self._promote_to_track(name))
                else:
                    menu.add_command(label="Change Axes", command=lambda: self._open_axis_picker_popup(name))
                    menu.add_command(label="Set to lo-track (lightweight)", command=lambda: self._demote_to_lotrack(name))
                menu.add_separator()
            menu.add_command(label="Delete", command=lambda: self._delete_tile(name))
            menu.post(event.x_root, event.y_root)
        return handler

    def _delete_tile(self, name):
        self.hidden_tiles.add(name)
        tile = self._tiles.pop(name, None)
        if tile is not None:
            try:
                tile["frame"].destroy()
            except Exception:
                pass

    def _open_add_heatmap_popup(self):
        manifest = self._manifest or {}
        known = set(self.known_names) | set(manifest.keys())
        available = [
            name for name in sorted(known)
            if name not in self._tiles and name not in self.pending_tiles and "error" not in manifest.get(name, {})        
        ]

        top = Toplevel(self.root)
        top.title("Add Matrix")
        top.geometry("300x260")
        top.configure(bg=BG)
        top.grab_set()

        frame = tk.Frame(top, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(frame, text="Choose a matrix to add", bg=BG, fg=TEXT, font=FONT_UI_BOLD,
                 wraplength=260, anchor="w").pack(anchor="w", pady=(0, 10))

        search_var = tk.StringVar()
        search_entry = tk.Entry(frame, textvariable=search_var, bg=CARD, fg=TEXT,
                                 insertbackground=TEXT, relief="flat", font=FONT_UI,
                                 highlightthickness=1, highlightbackground=BORDER, highlightcolor=ORANGE)
        search_entry.pack(fill=tk.X, pady=(0, 10), ipady=5)
        search_entry.bind("<KeyRelease>", lambda event: self._filter_add_menu(frame, available, search_var, top))

        self._add_buttons = {}
        if not available:
            tk.Label(frame, text="No additional matrices are available to add yet.",
                     bg=BG, fg=TEXT_FAINT, wraplength=260, justify="left", font=FONT_UI).pack(anchor="w")
            tk.Button(frame, text="Close", command=top.destroy, bg=CARD, fg=TEXT,
                      relief="flat", font=FONT_UI, bd=0, pady=6).pack(fill=tk.X, pady=(14, 0))
            return

        self._filter_add_menu(frame, sorted(available), search_var, top)

    def _filter_add_menu(self, frame, available, search_var, popup):
        query = (search_var.get() or "").strip().lower()
        for name in list(self._add_buttons):
            if self._add_buttons[name].winfo_exists():
                self._add_buttons[name].destroy()
            del self._add_buttons[name]

        for name in sorted(available):
            if query and query not in name.lower():
                continue
            label = name
            if name in self.hidden_tiles:
                label = f"{name}  (hidden)"
            btn = tk.Button(
                frame, text=label, command=lambda n=name: self._add_visible_heatmap(n, popup),
                bg=CARD, fg=TEXT, activebackground=ORANGE, activeforeground="#0a0a0a",
                relief="flat", font=FONT_UI, bd=0, anchor="w", padx=10, pady=6, cursor="hand2",
            )
            self._add_buttons[name] = btn
            btn.pack(fill=tk.X, pady=2)

    def _add_visible_heatmap(self, name, popup):
        self.hidden_tiles.discard(name)
        self.known_names.add(name)
        self.pending_tiles.add(name)
        default_state = _default_var_state(name, None)
        self.var_states[name] = default_state
        if self.control_queue is not None:
            try:
                self.control_queue.put(("ADD_VAR", name))
            except Exception:
                pass
        if self.config_queue is not None:
            try:
                self.config_queue.put(("STATE", name, default_state))
                self.config_queue.put(("CONFIG", name, _default_config(2)))
            except Exception:
                pass
        if popup is not None:
            popup.destroy()
        self.root.after(0, self._poll)

    def _enlarge(self, name, img_path):
        top = Toplevel(self.root)
        top.title(name)
        top.configure(bg=BG)
        head = tk.Frame(top, bg=BG)
        head.pack(fill=tk.X, padx=14, pady=(12, 4))
        tk.Label(head, text=name, bg=BG, fg=TEXT, font=FONT_HEAD).pack(anchor="w")

        stats = self._manifest.get(name, {})
        shape = stats.get("shape")
        if shape and stats.get("kind") != "scalar":
            btn_bar = tk.Frame(top, bg=BG)
            btn_bar.pack(fill=tk.X, padx=14, pady=(0, 8))
            cfg = list(self.display_configs.get(name) or _default_config(len(shape)))
            
            # Find the currently iterating axis, or default to the first dimension with size > 1
            iter_axis = next((i for i, v in enumerate(cfg) if v >= 2), None)
            if iter_axis is None:
                iter_axis = next((i for i, dim in enumerate(shape) if dim > 1), 0)

            ttk.Button(
                btn_bar,
                text="Open Slideshow",
                command=lambda: self._open_iterate_slideshow(name, iter_axis, shape, list(cfg), None),
            ).pack(anchor="w")

        img = Image.open(img_path)
        img.thumbnail((780, 780))
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(top, image=photo, bg=BG)
        label.image = photo
        label.pack(padx=14, pady=14)

    def _open_iterate_slideshow(self, name, axis_idx, shape, temp_cfg, axis_button):
        """Opens when an axis button is double-clicked to enter "iterate"
        mode. Lets you click Next/Previous to step through that axis one
        index at a time -- e.g. an attention tensor shaped (heads, seq, seq)
        with axis 0 iterating: Next shows head 1, head 2, ... and Previous
        walks back down -- while every other axis stays exactly as
        configured. The heatmap re-renders live via the config queue as you
        step, it isn't gated behind "Apply Changes"."""
        axis_len = shape[axis_idx]
        state = {"idx": max(0, (temp_cfg[axis_idx] - 2)) if temp_cfg[axis_idx] >= 2 else 0}

        top = Toplevel(self.root)
        top.title(f"{name} — iterate axis {axis_idx}")
        top.configure(bg=BG)
        top.geometry("520x580")

        head = tk.Frame(top, bg=BG)
        head.pack(fill=tk.X, padx=14, pady=(14, 4))
        title_lbl = tk.Label(head, text="", bg=BG, fg=TEXT, font=FONT_UI_BOLD)
        title_lbl.pack(anchor="w")
        tk.Label(head, text="Next / Previous step through this axis one index at a time.",
                 bg=BG, fg=TEXT_FAINT, font=FONT_MONO).pack(anchor="w", pady=(2, 0))

        img_label = tk.Label(top, bg=BG)
        img_label.pack(padx=14, pady=10)

        nav = tk.Frame(top, bg=BG)
        nav.pack(pady=(0, 14))

        def push_config():
            temp_cfg[axis_idx] = 2 + state["idx"]
            if axis_button is not None:
                axis_button.config(text=f"Ax {axis_idx}\n({axis_len})\n{_axis_val_label(temp_cfg[axis_idx])}")
            self.display_configs[name] = list(temp_cfg)
            if self.config_queue is not None:
                self.config_queue.put(("CONFIG", name, list(temp_cfg)))

        def refresh_image(retry=0):
            if not top.winfo_exists():
                return
            stats_now = self._manifest.get(name, {})
            img_path = stats_now.get("image")
            title_lbl.config(text=f"{name}  ·  axis {axis_idx} = index {state['idx']} / {axis_len - 1}")
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                img.thumbnail((460, 460))
                photo = ImageTk.PhotoImage(img)
                img_label.configure(image=photo)
                img_label.image = photo
            elif retry < 10:
                top.after(80, lambda: refresh_image(retry + 1))

        def go(delta):
            state["idx"] = max(0, min(axis_len - 1, state["idx"] + delta))
            push_config()
            top.after(120, refresh_image)

        prev_btn = ttk.Button(nav, text="← Previous", command=lambda: go(-1))
        prev_btn.pack(side=tk.LEFT, padx=6)
        next_btn = ttk.Button(nav, text="Next →", command=lambda: go(1))
        next_btn.pack(side=tk.LEFT, padx=6)
        top.bind("<Left>", lambda e: go(-1))
        top.bind("<Right>", lambda e: go(1))

        push_config()
        top.after(120, refresh_image)

    def _open_axis_picker_popup(self, name):
        stats = self._manifest.get(name, {})
        shape = stats.get("shape")
        if not shape:
            return

        top = Toplevel(self.root)
        top.title(f"Configure Axes — {name}")
        top.geometry("420x270")
        top.configure(bg=BG)

        current_cfg = self.display_configs.get(name)
        if not current_cfg or len(current_cfg) != len(shape):
            current_cfg = _default_config(len(shape))
            self.display_configs[name] = current_cfg

        temp_cfg = list(current_cfg)

        tk.Label(top, text=f"{name}  {tuple(shape)}", bg=BG, fg=TEXT, font=FONT_UI_BOLD).pack(pady=(16, 4))
        tk.Label(top, text="click = show  ·  double-click = iterate (opens slideshow)  ·  right-click = reset",
                 bg=BG, fg=TEXT_FAINT, font=FONT_MONO).pack(pady=(0, 14))

        btn_frame = tk.Frame(top, bg=BG)
        btn_frame.pack(pady=6)

        for axis_idx, dim_size in enumerate(shape):
            val = temp_cfg[axis_idx]
            btn = ttk.Button(btn_frame, text=f"Ax {axis_idx}\n({dim_size})\n{_axis_val_label(val)}", width=8)
            btn.pack(side=tk.LEFT, padx=4)

            def make_handler(ax_i, button_widget):
                def set_val(v):
                    temp_cfg[ax_i] = v
                    button_widget.config(text=f"Ax {ax_i}\n({shape[ax_i]})\n{_axis_val_label(v)}")

                def start_iterate():
                    set_val(2)
                    self._open_iterate_slideshow(name, ax_i, shape, temp_cfg, button_widget)

                button_widget.bind("<Button-1>", lambda e: set_val(1))
                # Removed double-click binding as requested:
                # button_widget.bind("<Double-Button-1>", lambda e: start_iterate())
                button_widget.bind("<Button-3>", lambda e: set_val(0))

            make_handler(axis_idx, btn)

        def save_config():
            self.display_configs[name] = list(temp_cfg)
            if self.config_queue is not None:
                self.config_queue.put(("CONFIG", name, list(temp_cfg)))
            self.root.after(0, self._poll)
            top.destroy()

        ttk.Button(top, text="Apply Changes", style="Accent.TButton", command=save_config).pack(pady=20, padx=20, fill=tk.X)

    def run(self):
        self.root.mainloop()


def _run_dashboard(session_id, display_configs, code_text=None, config_queue=None, control_queue=None, known_names=None, script_path=None, var_states=None, extra_files=None, initial_provider=None):
    Dashboard(session_id, display_configs, code_text=code_text, config_queue=config_queue, control_queue=control_queue, known_names=known_names, script_path=script_path, var_states=var_states, extra_files=extra_files, initial_provider=initial_provider).run()


# ============================================================================
# tracker: auto_track() -- shared discovery/tracing, then dispatches to
# either the GUI (multiprocess dashboard) or CLI (synchronous, in-process)
# ============================================================================

_debugger_bg = None
_session_id = None
_STDLIB_DIR = os.path.normcase(os.path.abspath(sysconfig.get_paths()["stdlib"]))


def _is_library_frame(filename):
    norm = os.path.normcase(os.path.abspath(filename))
    if "site-packages" in norm or "dist-packages" in norm:
        return True
    if norm.startswith(_STDLIB_DIR):
        return True
    if filename.startswith("<"):
        return True
    return False


def _determine_mode(requested_mode):
    """ui or cli. Explicit arg > PULSE_MODE env var > auto-detect (headless
    Linux with no DISPLAY/WAYLAND_DISPLAY -- e.g. Colab, SSH -- gets cli)."""
    if requested_mode in ("ui", "cli"):
        return requested_mode

    env_mode = os.environ.get("PULSE_MODE", "").lower()
    if env_mode in ("ui", "cli"):
        return env_mode

    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return "cli"

    try:
        import tkinter as _tk
        _probe = _tk.Tk()
        _probe.destroy()
    except Exception:
        return "cli"

    return "ui"


def _discover_candidates(caller_frame, root):
    """Merge static (ast, whole-source, not-yet-run) names with runtime
    (is_trackable-filtered, currently-in-scope) shapes -- so both a matrix
    only reachable inside a nested function AND whatever's already sitting
    in scope right now end up as candidates.

    Static discovery isn't limited to the entry script: for a modularized
    project, variables assigned inside a function defined in another local
    file (e.g. model.py's forward()) show up here too, before that function
    has ever run -- see _discover_project_files. Runtime tracing already
    followed into other files naturally (sys.settrace is global, not
    per-file), so this closes the one place that was entry-script-only.
    """
    entry_path = caller_frame.f_code.co_filename
    static_names = discover_static_names_from_file(entry_path)

    for other_path in _discover_project_files(entry_path, root):
        static_names |= discover_static_names_from_file(other_path)

    runtime = {}
    for name, val in caller_frame.f_locals.items():
        if name.startswith("__"):
            continue
        if is_trackable(val):
            runtime[name] = shape_of(val)

    discovered = {name: None for name in static_names}
    discovered.update(runtime)
    return discovered


def _install_pulse_excepthook(session_id):
    """Install a sys.excepthook that persists any uncaught exception's
    traceback to this session's cache dir before letting the default hook
    print it and the process exit normally. The Dashboard polls for this
    file (see Dashboard._poll) and, once found, automatically asks the
    agent to diagnose (and, if it can, fix) it -- covering errors that
    happen anywhere after auto_track() was called, not just ones caught
    during the initial dry run.
    """
    previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            try:
                tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                _atomic_write_json(
                    os.path.join(session_dir(session_id), "crash.json"),
                    {"traceback": tb_text, "time": time.time()},
                )
            except Exception:
                pass
        previous_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def auto_track(train_fn=None, throttle_interval=1.0, code_text=None, project_root=None, mode="cli"):
    """
    Call this once before your training loop, optionally passing your
    training function for a one-off dry run that discovers shapes:
        auto_track(train_step)

    mode: "ui" (dashboard + chat), "cli" (headless, Colab/SSH-friendly),
    or "auto" (default -- detects a real display and falls back to cli).
    """
    global _debugger_bg, _session_id
    mp.freeze_support()

    active_mode = _determine_mode(mode)
    caller_frame = sys._getframe(1)
    root = os.path.normcase(os.path.abspath(project_root)) if project_root else None

    # The caller's own source file -- this is what the CLI agent's code-fix
    # feature writes back to, so it's resolved regardless of whether code_text
    # ends up being auto-read from it or passed in explicitly.
    entry_path = caller_frame.f_code.co_filename
    if entry_path.startswith("<") or not os.path.exists(entry_path):
        entry_path = None

    if code_text is None and entry_path:
        try:
            with open(entry_path, "r", encoding="utf-8", errors="ignore") as f:
                code_text = f.read()
        except OSError:
            code_text = None

    # For a modularized project, gather the source of other local files this
    # script imports too (model.py, utils.py, etc.) -- capped so a huge repo
    # doesn't blow up the agent's context -- so "Send Code"/`/code` and any
    # proposed code fix can actually see and reference functions that live
    # outside the entry script. Keyed by the path shown to the agent.
    extra_files = {}
    if entry_path:
        total_chars = len(code_text or "")
        for other_path in _discover_project_files(entry_path, root):
            if total_chars > 200_000 or len(extra_files) >= 12:
                break
            try:
                with open(other_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            extra_files[other_path] = text
            total_chars += len(text)

    discovered = _discover_candidates(caller_frame, root)

    runtime_shapes = {k: v for k, v in discovered.items() if v is not None}

    def shape_tracer(frame, event, arg):
        if event != "line":
            return shape_tracer
        filename = frame.f_code.co_filename
        if _is_library_frame(filename):
            return None
        if root and not os.path.normcase(os.path.abspath(filename)).startswith(root):
            return None

        for name, val in frame.f_locals.items():
            if name in discovered and name not in runtime_shapes and is_trackable(val):
                runtime_shapes[name] = shape_of(val)
        return shape_tracer

    # If the dry run itself raises (e.g. a genuine init/setup bug in the
    # user's code, not just "hasn't reached the loop yet"), don't silently
    # swallow it -- capture it so the agent can be handed the traceback and
    # a shot at fixing it once it's set up, instead of Pulse just quietly
    # doing nothing and the user never finding out why.
    startup_error = None
    if train_fn is not None:
        sys.settrace(shape_tracer)
        caller_frame.f_trace = shape_tracer
        try:
            train_fn()
        except Exception:
            startup_error = traceback.format_exc()
        finally:
            sys.settrace(None)

    if not discovered:
        print("[PULSE] No trackable variables found (nothing in scope, and nothing parseable in the source).")
        return

    if active_mode == "cli":
        _start_cli_tracker(
            caller_frame, root, throttle_interval, discovered, runtime_shapes,
            code_text, entry_path, extra_files=extra_files, startup_error=startup_error,
        )
        return

    # ---- UI mode ----
    # If this process was just restarted after a code fix (see the
    # "RESTART" handling in persistent_tracer below), PULSE_AUTO_PROVIDER
    # carries the provider that was active before the restart -- its API
    # key is already sitting in os.environ (set right before the restart
    # happened), so both get auto-filled here instead of popping the setup
    # dialog and asking the user all over again.
    auto_provider = os.environ.pop("PULSE_AUTO_PROVIDER", None)
    result = None
    if auto_provider and auto_provider in PROVIDERS:
        auto_key = os.environ.get(PROVIDERS[auto_provider]["env_key"], "").strip()
        if auto_key:
            result = (auto_provider, auto_key)
            print(f"[PULSE] Resumed with agent {auto_provider} (auto-filled after restart).")
    if result is None:
        dialog = AgentSetupDialog()
        result = dialog.run()
    if not result:
        print("[PULSE] Setup cancelled.")
        return
    initial_provider, api_key = result
    os.environ[PROVIDERS[initial_provider]["env_key"]] = api_key

    # Zero-config, same as the CLI: track everything by default.
    tracked_vars = set(discovered.keys())
    display_configs = {}
    auto_mode = True

    if not tracked_vars:
        print("[PULSE] No trackable variables found.")
        return

    var_states = {name: _default_var_state(name, runtime_shapes.get(name)) for name in tracked_vars}

    _session_id = str(uuid.uuid4())[:8]
    _debugger_bg = HeatmapCreatorBG(display_configs, _session_id, var_states)
    shared_config_queue = _debugger_bg.queue
    control_queue = mp.Queue()

    dash_process = mp.Process(
        target=_run_dashboard,
        args=(_session_id, display_configs, code_text, shared_config_queue, control_queue, sorted(discovered), entry_path, var_states, extra_files, initial_provider),
        daemon=True,
    )
    dash_process.start()

    # If the dry run above already crashed, hand that off to the Dashboard
    # immediately via the same crash file the excepthook below uses -- no
    # need to wait for a live exception once the agent is ready.
    if startup_error:
        try:
            _atomic_write_json(
                os.path.join(session_dir(_session_id), "crash.json"),
                {"traceback": startup_error, "time": time.time()},
            )
        except Exception:
            pass

    # Catch any uncaught exception that crashes the rest of the user's
    # script (init errors, bugs that only show up once the loop starts,
    # etc.) and hand the traceback to the Dashboard via the same crash file
    # mechanism, so the agent can diagnose -- and potentially fix -- it even
    # though the process still has to exit afterward (Python can't resume
    # past an unhandled exception; a fix just means the next run works).
    _install_pulse_excepthook(_session_id)

    last_logged = {}
    paused = {"value": False}

    def persistent_tracer(frame, event, arg):
        while True:
            try:
                msg = control_queue.get_nowait()
            except Exception:
                break
            if isinstance(msg, tuple) and msg:
                if msg[0] == "ADD_VAR":
                    tracked_vars.add(msg[1])
                    last_logged.pop(msg[1], None)
                elif msg[0] == "PAUSE":
                    paused["value"] = True
                elif msg[0] == "RESUME":
                    paused["value"] = False
                elif msg[0] == "RESTART":
                    # The chat panel just applied a code fix to disk --
                    # restart the whole process so the training loop
                    # actually runs the fixed code (this process is still
                    # executing the old code in memory otherwise). The
                    # active provider is threaded through PULSE_AUTO_PROVIDER
                    # so the next run's setup auto-fills the agent and API
                    # key (already set in os.environ) instead of asking
                    # again -- see the UI-mode setup above.
                    provider_name = msg[1] if len(msg) > 1 else None
                    if provider_name:
                        os.environ["PULSE_AUTO_PROVIDER"] = provider_name
                    try:
                        _debugger_bg.shutdown()
                    except Exception:
                        pass
                    try:
                        dash_process.terminate()
                    except Exception:
                        pass
                    sys.settrace(None)
                    print("\n[PULSE] Fix applied -- restarting the training loop to pick it up...\n")
                    sys.stdout.flush()
# Replace the final os.execv line with this:
                    script_path = getattr(sys.modules.get('__main__'), '__file__', None)
                    
                    script_path = os.path.abspath(script_path)
                    

                    # Spawn the new process safely using subprocess (which handles spaces on Windows)
                    subprocess.Popen([sys.executable, script_path] + sys.argv[1:])
                    sys.exit(0)
        # bad (NaN/inf, a loss spike) and asked training to hold here until
        # the user hits "Resume Training" -- or a fix gets applied and they
        # resume manually. Blocks this exact line from executing further,
        # which is as close to "pausing training" as Pulse can get without
        # owning the training loop itself.
        while paused["value"]:
            try:
                msg = control_queue.get(timeout=0.2)
            except Exception:
                msg = None
            if isinstance(msg, tuple) and msg:
                if msg[0] == "RESUME":
                    paused["value"] = False
                elif msg[0] == "ADD_VAR":
                    tracked_vars.add(msg[1])
                    last_logged.pop(msg[1], None)

        filename = frame.f_code.co_filename
        if _is_library_frame(filename):
            return None
        if root and not os.path.normcase(os.path.abspath(filename)).startswith(root):
            return None

        if event == "line":
            now = time.time()
            for name, val in frame.f_locals.items():
                if name.startswith("__"):
                    continue
                if auto_mode:
                    if not is_trackable(val):
                        continue
                    tracked_vars.add(name)
                elif name not in tracked_vars:
                    continue
                elif not is_trackable(val):
                    continue

                if now - last_logged.get(name, 0) > throttle_interval:
                    _debugger_bg.log_matrix(name, val)
                    last_logged[name] = now

        return persistent_tracer

    sys.settrace(persistent_tracer)
    caller_frame.f_trace = persistent_tracer


def _install_cli_excepthook(cli):
    """Catch any uncaught exception that crashes the rest of the user's
    script (anywhere after CLI tracing starts -- an init error, or a bug
    that only shows up once the loop runs) and offer to have the agent
    diagnose -- and potentially fix -- it right there, before the process
    actually exits. Python can't resume execution past an unhandled
    exception, but a fix means the *next* run has a shot at working.
    """
    previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        previous_hook(exc_type, exc_value, exc_tb)
        if issubclass(exc_type, KeyboardInterrupt):
            return
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print("\n[Pulse] Your script just crashed with the exception above.")
        try:
            if not cli.agent_provider:
                resp = input(
                    "[Pulse] Set up an AI agent now so Pulse can try to diagnose/fix it? (y/n) > "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    return
                if not cli._select_agent_provider_and_key(initial=True):
                    return
            else:
                resp = input(
                    "[Pulse] Ask the agent to diagnose (and try to fix) this? (y/n) > "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    return
        except (EOFError, KeyboardInterrupt):
            return

        question = (
            f"My script just crashed with this uncaught exception:\n{tb_text}\n"
            "Please diagnose the root cause and, if you can, fix it."
        )
        cli.ask_agent(question, include_code=True)

    sys.excepthook = _hook


def _start_cli_tracker(
    caller_frame, root, throttle_interval, discovered, runtime_shapes,
    code_text=None, script_path=None, extra_files=None, startup_error=None,
):
    """CLI mode: synchronous, in-process -- no subprocess, no multiprocessing
    Queue, no pickling tensors across a process boundary (which can be
    genuinely broken for CUDA tensors anyway). Just prints as training runs
    and, per your setup choice, saves labeled PDF snapshots. The CLI also
    provides the same AI agent/provider selection flow as the GUI.

    `discovered`/`runtime_shapes` are the exact same static-AST + in-scope
    merge that feeds the GUI's `MatrixConfigUI` picker (see
    `_discover_candidates`). Previously this function ignored both and only
    offered whatever happened to be a local variable at the first traced
    line -- so anything assigned later in the loop, or only reachable
    inside a nested function that hadn't run yet, never showed up in the
    CLI's setup menu even though the GUI would have listed it (as
    "not run yet"). Threading them through here brings CLI discovery in
    line with the GUI.

    `script_path` is the caller's own source file -- passed through so the
    CLI agent's code-fix feature (see PulseCLI._apply_code_fix) knows which
    file on disk to write proposed fixes to by default. `extra_files` is
    {path: text} for other local project files this script imports (a
    modularized project's model.py/utils.py/etc.), so the agent can see and
    propose fixes to code that isn't in the entry script at all.
    `startup_error` is a formatted traceback if the dry run passed to
    auto_track() raised -- surfaced immediately so the user can ask the
    agent to fix a bug that was blocking training from even starting.
    """
    from .pulse_cli import PulseCLI

    cli = PulseCLI(discovered={name: runtime_shapes.get(name) for name in discovered})
    cli.set_code_text(code_text, script_path=script_path)
    cli.extra_files = dict(extra_files or {})
    cli.pending_startup_error = startup_error
    cli.print_banner()
    _install_cli_excepthook(cli)

    if startup_error:
        print("[Pulse] The function passed to auto_track() raised an exception during its dry run:")
        print(startup_error)
        print("[Pulse] Set up an AI agent below and Pulse will offer to diagnose/fix it before continuing.\n")

    setup_done = {"value": False}
    last_logged = {"t": 0.0}

    def cli_tracer(frame, event, arg):
        filename = frame.f_code.co_filename
        if _is_library_frame(filename):
            return None
        if root and not os.path.normcase(os.path.abspath(filename)).startswith(root):
            return None

        if event != "line":
            return cli_tracer

        local_vars = frame.f_locals

        # Fill in shapes for statically-discovered names as soon as they
        # actually resolve to a value -- same as the GUI's shape_tracer
        # dry-run, just kept live instead of front-loaded into a single
        # pre-pass. If a variable was defaulted to 'lotrack' before its
        # shape was known and it turns out to actually be a scalar, upgrade
        # it to full 'track' -- scalars are cheap regardless.
        for name, val in local_vars.items():
            if name in cli.discovered and cli.discovered.get(name) is None and is_trackable(val):
                cli.discovered[name] = shape_of(val)
                if (
                    name in cli.tracked_vars
                    and cli.var_states.get(name) == "lotrack"
                    and shape_of(val) == ()
                ):
                    cli.var_states[name] = "track"

        if not setup_done["value"]:
            has_resolved_shape = any(shape is not None for shape in cli.discovered.values())
            has_trackable_local = any(
                not n.startswith("__") and is_trackable(v) for n, v in local_vars.items()
            )
            if has_resolved_shape or has_trackable_local:
                cli.watch_locals = local_vars
                cli.interactive_setup()
                setup_done["value"] = True
            return cli_tracer

        cli.watch_locals = local_vars

        # Auto mode (the CLI's equivalent of the GUI's "track every matrix
        # automatically" toggle): keep picking up newly-trackable locals as
        # the loop runs, instead of being limited to what was chosen once
        # at setup time.
        if cli.auto_mode:
            for name, val in local_vars.items():
                if name.startswith("__"):
                    continue
                if is_trackable(val) and name not in cli.tracked_vars:
                    cli.tracked_vars.append(name)
                    cli.var_states[name] = cli._default_state_for(name, val)

        now = time.time()
        if now - last_logged["t"] > throttle_interval:
            cli.update()
            last_logged["t"] = now

        return cli_tracer

    sys.settrace(cli_tracer)
    caller_frame.f_trace = cli_tracer
    # No routine "tracing started" print -- minimal UI stays silent unless
    # something's actually wrong (a read error, a crash, auto-intervention).


def shutdown():
    global _debugger_bg
    if _debugger_bg:
        _debugger_bg.shutdown()
        _debugger_bg = None
