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
import queue as _queue
import traceback
import sysconfig
import multiprocessing as mp
import ast
import base64
import math
import copy
import difflib
import hashlib
import itertools
import inspect
import importlib
import __main__
import subprocess
import shutil
from  pulse.pulse_cli import (
    _install_keras_pulse_hook, _RESTART_CHILD_ENV, _RESTART_DEPTH_ENV,
    _AGENT_MAX_TOKENS, _AGENT_TIMEOUT_SECONDS, _clamp_output_tokens,
    _PASS4B_PROBE_TMPL, _PASS4B_REVISE_WITH_EVIDENCE_TMPL, _PROBE_HARNESS_SRC,
    _loss_probe_verdict, _scrape_stdout_losses, _MAX_EMPIRICAL_ATTEMPTS,
    _PROBE_SOFT_TIME_BUDGET_SECONDS, _PROBE_HARD_TIMEOUT_SECONDS,
)

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
from pulse import pulse_detect as _pulse_detect

import multiprocessing as mp

import litellm




# ---------------------------------------------------------------------------
# TEMPORARY DEBUG INSTRUMENTATION
# Enable with PULSE_LOGGING=1 (default ON in this logging build).
# Writes a low-volume trace to ./pulse.log without dumping tensor data.
# ---------------------------------------------------------------------------
_PULSE_LOGGING = os.environ.get("PULSE_LOGGING", "1").strip().lower() not in ("0", "off", "false", "no")
_PULSE_LOG_FILE = os.path.abspath(os.environ.get("PULSE_LOG_FILE", "pulse.log"))


def _pulse_log(message: str, *, console: bool = False) -> None:
    if not _PULSE_LOGGING:
        return
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    try:
        with open(_PULSE_LOG_FILE, "a", encoding="utf-8") as _f:
            _f.write(line + "\n")
    except Exception:
        pass
   


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

METRIC_NAME_HINTS = (
    "acc", "accuracy", "precision", "recall", "f1", "auc", "iou", "dice",
    "score", "bleu", "rouge", "map", "psnr", "ssim",
)


def _looks_like_loss(name):
    n = (name or "").lower()
    return any(hint in n for hint in LOSS_NAME_HINTS)


def _looks_like_metric(name):
    """Accuracy/precision/recall/etc-style variables -- see
    PulseCLI._check_for_trouble's copy of this same helper for the full
    rationale. Needed here too now that the GUI's own _check_for_trouble
    checks for suspiciously-perfect-early accuracy (data leakage)."""
    n = (name or "").lower()
    return any(hint in n for hint in METRIC_NAME_HINTS)


def _looks_like_grad_or_weight_norm(name):
    """A tracked scalar that's some kind of gradient/parameter norm --
    see PulseCLI._check_for_trouble's copy of this same helper."""
    n = (name or "").lower()
    return any(
        hint in n for hint in (
            "grad_norm", "gradient_norm", "grad norm", "gradnorm",
            "weight_norm", "param_norm", "parameter_norm", "weightnorm",
        )
    )


def _looks_like_learning_rate(name):
    """A tracked scalar that's the current learning rate -- see
    PulseCLI._check_for_trouble's copy of this same helper."""
    n = (name or "").lower()
    return n in ("lr", "learning_rate", "learningrate") or n.endswith(("_lr", "_learning_rate"))


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


def _flush_stdin() -> None:
    """Discard any input sitting unread in the terminal's input buffer.

    Without this, keystrokes typed while output was streaming (a restart
    banner, a traceback, training-step logs) sit in the OS-level tty
    buffer and get delivered as soon as the next input() starts reading --
    landing in the middle of that prompt's line instead of being ignored.
    Called right before every input() so each prompt starts from a clean,
    empty line. Mirrors the same helper in pulse_cli.py.
    """
    try:
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        else:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass  # not a real terminal (piped input, some IDEs, etc.) -- nothing to flush


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
# Cap on how many (step, value) points of a scalar's full history are kept
# in the manifest JSON for the CORR/OUTLIER/DIFFSTATS/HISTOGRAM chat tools
# -- bounded so a long run's manifest file doesn't grow without limit.
_SCALAR_HISTORY_CAP = 2000

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
                # Full (step, value) history, bounded so the manifest JSON
                # doesn't grow unbounded over a long run -- this is what
                # backs the CORR/OUTLIER/DIFFSTATS/HISTOGRAM chat
                # directives (see ChatPanel._scalar_history), which need
                # real step-aligned data rather than just the last few
                # values in `recent`.
                stats["history"] = hist[-_SCALAR_HISTORY_CAP:]
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


# ============================================================================
# CPU-ONLY OBSERVATION MODE
# ============================================================================
# Pulse must never call .cpu(), .numpy(), .item(), .detach().cpu(), or backend
# statistics on an accelerator tensor. There is no way for a CPU process to
# read arbitrary GPU VRAM without a device-to-host transfer; that transfer is
# itself a GPU/driver operation. In strict mode we therefore observe only CPU
# resident values (including NumPy arrays and CPU tensors). If a training
# program wants a GPU tensor visualized, it must expose an already-created CPU
# mirror. Pulse will consume that mirror without touching the accelerator.
PULSE_CPU_ONLY = os.environ.get("PULSE_CPU_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_accelerator_value(value):
    """Best-effort device check without importing or touching any ML backend."""
    if value is None:
        return False
    try:
        dev = getattr(value, "device", None)
        if dev is not None and str(dev).lower() not in {"cpu", "none", ""}:
            return True
    except Exception:
        pass
    # CuPy exposes .device; TensorFlow/JAX/PyTorch expose device metadata in
    # different forms. Avoid calling backend methods here: metadata only.
    try:
        dev = getattr(getattr(value, "array", None), "device", None)
        if dev is not None and str(dev).lower() not in {"cpu", "none", ""}:
            return True
    except Exception:
        pass
    return False


def _cpu_resident(value):
    """True only when Pulse can safely hand `value` to CPU-side code."""
    if value is None:
        return True
    if _is_accelerator_value(value):
        return False
    if isinstance(value, np.ndarray):
        return True
    # Python scalars/lists/dicts are already host-side.
    if isinstance(value, (int, float, complex, bool, str, bytes, list, tuple, dict)):
        return True
    # PyTorch CPU tensors and similar host tensors generally expose a CPU
    # device. Do not call .cpu() or .numpy() here.
    try:
        dev = getattr(value, "device", None)
        if dev is not None:
            return str(dev).lower() == "cpu"
    except Exception:
        pass
    # Unknown objects are NOT assumed safe: strict mode drops them.
    return False


class AgentRequestFailed(Exception):
    """Raised by _call_model once its in-call retries are exhausted, with
    a short, classified, user-facing reason (see _classify_model_error)
    -- never returned as a plain string, which would risk getting parsed
    downstream as if it were a real diagnosis/fix. Caught at exactly one
    place, ChatPanel._ask's top-level try/except, which stops the
    pipeline cleanly and queues the request for a later background retry
    -- see _queue_agent_retry."""
    pass


def _mirror_name_candidates(name):
    """Names commonly used by training code for an already-created CPU copy."""
    return (
        f"{name}_cpu", f"{name}_host", f"{name}_np", f"cpu_{name}", f"host_{name}",
    )


# ============================================================================
# Multi-GPU support: (a) real per-device status (memory/utilization) via
# nvidia-smi + torch.cuda, framework-agnostic and independent of
# PULSE_CPU_ONLY (querying device *metadata* -- not tensor values -- isn't
# the accelerator read that strict mode is about); (b) distributed-launch
# awareness (torchrun/torch.distributed.launch/OpenMPI/Slurm) so a
# multi-process, one-rank-per-GPU job gets exactly one Pulse UI (rank 0)
# instead of one per process, while every rank's GPU status/crash state is
# still visible to that one UI via small per-rank status files.
# ============================================================================
def _distributed_info():
    """Best-effort detection of a multi-process, one-rank-per-GPU launch,
    purely from environment variables -- no torch import required. Returns
    (rank, local_rank, world_size, session_key). session_key is a value
    shared by every rank of the same job (so they agree on where to write
    their status files) -- None when not running distributed. Falls back
    to (0, 0, 1, None) for an ordinary single-process run, so every call
    site can treat world_size == 1 as "not distributed" without a separate
    check.
    """
    def _int_env(*names, default=None):
        for n in names:
            v = os.environ.get(n)
            if v is not None:
                try:
                    return int(v)
                except ValueError:
                    continue
        return default

    rank = _int_env("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", default=0)
    local_rank = _int_env("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", default=rank)
    world_size = _int_env("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS", default=1)

    session_key = None
    if world_size > 1:
        session_key = (
            os.environ.get("PULSE_SESSION_ID")
            or os.environ.get("TORCHELASTIC_RUN_ID")
            or f"{os.environ.get('MASTER_ADDR', 'local')}:{os.environ.get('MASTER_PORT', '0')}"
        )
    return rank, local_rank, world_size, session_key


def _gpu_status_snapshot():
    """Real per-device GPU status visible to THIS process: name/total
    memory/utilization (via `nvidia-smi`, if present on PATH) merged with
    this process's own allocated/reserved memory (via torch.cuda, if
    importable) -- deterministic numbers instead of a model guessing about
    'the GPU'. Framework-agnostic: works even when the training code
    doesn't use torch, as long as nvidia-smi is available. Returns a list
    of per-device dicts (possibly empty if nothing detectable), sorted by
    device index.
    """
    devices = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 5:
                    continue
                idx, name, mem_total, mem_used, util = parts
                try:
                    devices[int(idx)] = {
                        "index": int(idx), "name": name,
                        "mem_total_mb": float(mem_total), "mem_used_mb": float(mem_used),
                        "util_pct": float(util),
                    }
                except ValueError:
                    continue
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                entry = devices.setdefault(i, {"index": i, "name": torch.cuda.get_device_name(i)})
                try:
                    entry["this_process_alloc_mb"] = torch.cuda.memory_allocated(i) / (1024 ** 2)
                    entry["this_process_reserved_mb"] = torch.cuda.memory_reserved(i) / (1024 ** 2)
                    if "mem_total_mb" not in entry:
                        entry["mem_total_mb"] = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
                except Exception:
                    continue
    except ImportError:
        pass

    return [devices[i] for i in sorted(devices)]


def _format_gpu_status(devices, label=None):
    suffix = f" ({label})" if label else ""
    if not devices:
        return f"GPUSTATUS{suffix}: no CUDA devices detected (no nvidia-smi on PATH and/or torch.cuda unavailable)."
    lines = []
    for d in devices:
        parts = [f"cuda:{d['index']}"]
        if d.get("name"):
            parts.append(d["name"])
        if "mem_used_mb" in d and "mem_total_mb" in d:
            parts.append(f"{d['mem_used_mb']:.0f}/{d['mem_total_mb']:.0f} MB")
        elif "mem_total_mb" in d:
            parts.append(f"{d['mem_total_mb']:.0f} MB total")
        if "util_pct" in d:
            parts.append(f"{d['util_pct']:.0f}% util")
        if "this_process_alloc_mb" in d:
            parts.append(f"this process: {d['this_process_alloc_mb']:.0f} MB allocated")
        lines.append("  " + ", ".join(parts))
    return f"GPUSTATUS{suffix}: {len(devices)} device(s)\n" + "\n".join(lines)


def _rank_status_dir(session_key):
    d = os.path.join(tempfile.gettempdir(), "pulse_cache", "ranks", session_key)
    os.makedirs(d, exist_ok=True)
    return d


def _write_rank_status(session_key, rank, local_rank, world_size, extra=None):
    """Persist this rank's GPU status (and, via `extra`, a crash traceback
    if it has one) to a small per-rank JSON file so rank 0's Pulse UI can
    aggregate a multi-GPU/multi-rank view without any direct IPC between
    ranks -- just a shared directory on a shared filesystem, the same
    trust model the rest of Pulse already uses for crash-file handoff."""
    path = os.path.join(_rank_status_dir(session_key), f"rank_{rank}.json")
    try:
        import socket
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown-host"
    payload = {
        "rank": rank, "local_rank": local_rank, "world_size": world_size,
        "pid": os.getpid(), "hostname": hostname, "updated": time.time(),
        "gpus": _gpu_status_snapshot(),
    }
    if extra:
        payload.update(extra)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass


def _read_all_rank_status(session_key):
    """All ranks' latest status payloads (see _write_rank_status), sorted
    by rank. A rank that hasn't written yet (or whose file vanished) is
    just absent rather than raising."""
    out = []
    try:
        names = os.listdir(_rank_status_dir(session_key))
    except OSError:
        return out
    for name in sorted(names):
        if not (name.startswith("rank_") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(_rank_status_dir(session_key), name), "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda p: p.get("rank", 0))
    return out


def _format_multi_rank_gpu_status(session_key, this_rank):
    statuses = _read_all_rank_status(session_key)
    if not statuses:
        return f"GPUSTATUS (rank {this_rank}, distributed): no rank status files found yet -- other ranks may not have started reporting."
    now = time.time()
    lines = []
    for s in statuses:
        age = now - s.get("updated", 0)
        staleness = "" if age < 15 else f"  [STALE, last update {age:.0f}s ago]"
        tag = f"rank {s.get('rank')}" + (" (this process)" if s.get("rank") == this_rank else "")
        lines.append(f"{tag} on {s.get('hostname', '?')} (pid {s.get('pid', '?')}){staleness}:")
        devices = s.get("gpus") or []
        if not devices:
            lines.append("    no GPU status reported")
        else:
            for d in devices:
                parts = [f"cuda:{d.get('index')}"]
                if d.get("name"):
                    parts.append(d["name"])
                if "mem_used_mb" in d and "mem_total_mb" in d:
                    parts.append(f"{d['mem_used_mb']:.0f}/{d['mem_total_mb']:.0f} MB")
                if "util_pct" in d:
                    parts.append(f"{d['util_pct']:.0f}% util")
                if "this_process_alloc_mb" in d:
                    parts.append(f"this process: {d['this_process_alloc_mb']:.0f} MB allocated")
                lines.append("    " + ", ".join(parts))
        scalars = s.get("scalars") or {}
        if scalars:
            shown = ", ".join(f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in list(scalars.items())[:6])
            lines.append(f"    scalars: {shown}")
        if s.get("error"):
            lines.append(f"    ⚠ this rank crashed: {_format_exc_short(s['error'], max_lines=6)}")
    world_size = statuses[0].get("world_size") if statuses else "?"
    return f"GPUSTATUS: {len(statuses)}/{world_size} rank(s) reporting\n" + "\n".join(lines)


class HeatmapCreatorBG:
    """CPU-only bridge between the training process and the renderer.

    In strict CPU mode this object NEVER converts, synchronizes, copies, or
    queries accelerator tensors. The training process can only submit values
    that are already resident in host memory. This makes Pulse observational:
    it cannot enqueue a CUDA copy, synchronize a CUDA stream, or execute GPU
    statistics on behalf of the debugger.
    """
    def __init__(self, display_configs, session_id, var_states=None):
        self.session_id = session_id
        self.queue = mp.Queue(maxsize=2)
        self.process = mp.Process(
            target=_worker_main,
            args=(self.queue, display_configs, session_id, var_states or {}),
            daemon=True,
        )
        self.process.start()

    def log_matrix(self, var, matrix, config_override=None):
        # Strictly reject accelerator values. No .cpu(), .numpy(), .item(),
        # backend statistics, or other device operation occurs here.
        if PULSE_CPU_ONLY and not _cpu_resident(matrix):
            return False
        item = (var, matrix, config_override)
        try:
            self.queue.put_nowait(item)
            return True
        except Exception:
            # Visualization is lossy by design. Never wait for the renderer.
            try:
                self.queue.get_nowait()
            except Exception:
                pass
            try:
                self.queue.put_nowait(item)
                return True
            except Exception:
                return False

    def log_cpu_snapshot(self, var, cpu_value, config_override=None):
        """Explicit API for an already-created CPU mirror.

        The caller owns creation of the mirror. Pulse never creates it and
        therefore never touches the accelerator.
        """
        if not _cpu_resident(cpu_value):
            return False
        return self.log_matrix(var, cpu_value, config_override)

    def update_state(self, var, new_state):
        try:
            self.queue.put_nowait(("STATE", var, new_state))
        except Exception:
            pass

    def update_config(self, var, new_config):
        try:
            self.queue.put_nowait(("CONFIG", var, new_config))
        except Exception:
            pass

    def shutdown(self):
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
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
        self.root.geometry("420x400")
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
        self.existing_note.pack(anchor="w", pady=(0, 10))

        # Only relevant when PROVIDER is set to the "Custom" or "OpenRouter
        # (any model)" sentinel -- left empty/disabled otherwise, and ignored
        # by submit() for every other provider.
        ttk.Label(body, text="MODEL STRING (Custom / OpenRouter any model)", style="Faint.TLabel").pack(anchor="w")
        self.custom_model_var = tk.StringVar()
        custom_model_entry = ttk.Entry(body, textvariable=self.custom_model_var)
        custom_model_entry.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(body, text="e.g. openai/gpt-6-astra, ollama_chat/llama3.1 (OpenRouter: deepseek/deepseek-v4-flash)", style="Faint.TLabel").pack(anchor="w", pady=(0, 10))

        ttk.Label(body, text="ENV VAR FOR KEY (Custom only, optional)", style="Faint.TLabel").pack(anchor="w")
        self.custom_env_var = tk.StringVar()
        ttk.Entry(body, textvariable=self.custom_env_var).pack(fill=tk.X, pady=(4, 4))

        def _check_existing(*_):
            info = PROVIDERS[self.provider_var.get()]
            if info.get("custom"):
                self.existing_note.config(text="")
                return
            env_var = info["env_key"]
            if os.environ.get(env_var):
                self.existing_note.config(text=f"{env_var} is already set -- leave blank to use it.")
            else:
                self.existing_note.config(text="")

        self.provider_var.trace_add("write", _check_existing)
        _check_existing()

        self.result = None

        def submit():
            provider = self.provider_var.get()

            if PROVIDERS[provider].get("custom"):
                model_string = self.custom_model_var.get().strip()
                if not model_string or "/" not in model_string:
                    messagebox.showerror("Model string required", "Enter a full model string, e.g. openai/gpt-6-astra")
                    return
                env_var = self.custom_env_var.get().strip() or None
                key = self.key_var.get().strip() or (os.environ.get(env_var, "").strip() if env_var else "local")
                if env_var and not key:
                    messagebox.showerror("API key required", f"Enter an API key, or set {env_var} first.")
                    return
                label = f"Custom: {model_string}"
                PROVIDERS[label] = {"model": model_string, "env_key": env_var}
                if env_var and key and key != "local":
                    os.environ[env_var] = key
                self.result = (label, key or "local")
                self.root.destroy()
                return

            if PROVIDERS[provider].get("openrouter"):
                model_string = self.custom_model_var.get().strip()
                if not model_string or "/" not in model_string:
                    messagebox.showerror("Model required", "Enter an OpenRouter model, e.g. deepseek/deepseek-v4-flash")
                    return
                provider = register_openrouter_model(model_string)

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
    "tracking. Only promote variables that are currently lotrack.\n"
    "  GREP: <pattern>\n"
    "    You don't need the whole file to check one detail. Search every tracked project file for a "
    "pattern (a plain word/phrase, or a regex) and get back each match with a line of context on "
    "either side -- e.g. 'GREP: learning_rate' or 'GREP: def forward'. Works even if 'Send Code' "
    "isn't checked this turn. Use this to locate something before deciding whether you need the "
    "full surrounding block via VIEW. Capped to a modest number of matches; narrow the pattern if "
    "truncated.\n"
    "  VIEW: <file>:<start>-<end>\n"
    "    Pull an exact line range from a specific file (the file label is whatever header you were "
    "shown, e.g. 'model.py'; omit '<file>:' to default to the main script), e.g. 'VIEW: "
    "model.py:40-75' or 'VIEW: 120-160'. Use this after a GREP hit to see the exact surrounding code "
    "before proposing a fix. Capped to a few hundred lines per call.\n"
    "  Put CALC:/PROMOTE:/GREP:/VIEW: lines anywhere in your Reasoning, not in the Diagnosis or Fix. "
    "GREP/VIEW results come back as a new message before your next turn -- if what you get back "
    "changes your diagnosis, say so.\n\n"

    "MORE TOOLS -- the same idea taken further: anything Pulse can just compute or look up for you "
    "beats you reasoning your way to a guess. Use these the same way as above (own line, anywhere in "
    "Reasoning); each returns a result as a new message before your next turn.\n"
    "  Execution (the single biggest lever -- no amount of reasoning substitutes for actually running "
    "code):\n"
    "    REPL: <expr> -- evaluate an expression against the LIVE training process's current "
    "globals/locals right now, instead of reasoning from a context dump that may already be stale.\n"
    "    DRYRUN: <function_or_method_call> -- actually execute a specific call (e.g. "
    "'DRYRUN: model.forward(x)') in the live process and get back the real output, exception, and "
    "traceback. Turns 'I think this will raise a shape error' into an actual answer.\n"
    "    SHAPETRACE: [optional model variable name] -- run one real forward pass with a live tensor "
    "already in scope and dump every submodule's input/output shape. Eliminates shape-mismatch "
    "guessing entirely. PyTorch only.\n"
    "    GRADCHECK: <param_name> -- numerical finite-difference gradient check on a specific "
    "parameter, deterministic pass/fail against its real .grad. Needs a zero-arg `loss_fn` callable "
    "in scope that recomputes the current loss -- if none exists, Pulse will tell you to define one.\n"
    "    REPLAY: <n_steps> -- replay the last n_steps from an isolated checkpoint (with a zero-arg "
    "`train_step` callable you define) and diff the resulting loss curve, then restores live state -- "
    "a real empirical check instead of restarting the whole process and hoping.\n"
    "    These run in the actual training process; if none is available this session Pulse will say so "
    "rather than fabricating a result.\n"
    "  Code intelligence beyond GREP/VIEW (structure, not text search):\n"
    "    DEFOF: <symbol> -- AST-based jump-to-definition across every tracked file.\n"
    "    CALLERS: <symbol> -- every call site of a function/class.\n"
    "    DEPGRAPH: -- the import graph between tracked local files.\n"
    "  Statistics over a variable's whole recorded history (scalars only -- loss/accuracy/lr/custom "
    "logged values), not eyeballing a chart:\n"
    "    CORR: <var1> <var2> -- real correlation coefficient between two histories.\n"
    "    OUTLIER: <var> -- deterministic z-score anomaly detection, flags exact points.\n"
    "    DIFFSTATS: <var> <index_a> <index_b> -- exact delta between two recorded points.\n"
    "    HISTOGRAM: <var> -- actual bucketed distribution counts.\n"
    "  Grounding (lookup beats memorized/stale recall):\n"
    "    DOCLOOKUP: <library>.<symbol> -- real signature/docstring for an installed library function.\n"
    "    CHANGELOG: -- diff of what's actually changed in tracked files since the last checkpoint.\n"
    "    PASTFIX: <symbol_or_region> -- search this project's own log of prior fixes for the same area.\n"
    "  Multi-GPU:\n"
    "    GPUSTATUS: -- real per-device memory/utilization for every visible GPU. If this run is one "
    "rank of a multi-process, one-rank-per-GPU launch (torchrun/torch.distributed.launch/etc.), this "
    "aggregates every rank's GPU status instead of just the one this chat is attached to (which is "
    "always rank 0 -- other ranks run headless).\n"
    "  ML health/anti-patterns (heuristic -- worth double-checking, not guaranteed):\n"
    "    MLLINT: -- a handful of AST-detectable ML anti-patterns: metric/loss mismatches in either "
    "direction (e.g. 'accuracy' against a regression loss, or 'mae' against a classification loss), "
    "double-softmax/double-sigmoid into a loss that already applies one, plain Softmax feeding NLLLoss "
    "(which expects log-probabilities), missing zero_grad(), backward()+zero_grad() with no optimizer "
    "step() ever taken, suspiciously high literal learning rates, eval-looking functions missing "
    "no_grad()/.eval(). Cheap to run any time; worth running proactively after a fix, not just when "
    "something looks wrong.\n"
    "    LAYERSTATS: [optional model var] -- per-layer gradient-norm/weight-norm ratio for a live "
    "torch model, right after backward(). Flags likely dead (near-zero ratio) or exploding (large "
    "ratio) layers individually, instead of only seeing an aggregate gradient norm.\n"
    "    HARDEXAMPLES: [optional N, default 10] -- per-example loss for the current batch, surfacing "
    "the highest-loss samples. Needs a zero-arg `per_example_losses_fn` callable in scope (a "
    "reduction='none' loss call). Often the fastest way to spot label noise.\n"
    "    AMPSTATUS: [optional GradScaler var] -- current mixed-precision scale factor/skip state, for "
    "spotting fp16 numerical instability.\n"
    "  Reliability/reproducibility:\n"
    "    SEEDCHECK: -- RNG fingerprint (Python/NumPy/PyTorch) vs the last recorded fingerprint for this "
    "project. A mismatch isn't automatically wrong (more steps, an intentional reseed, a new shuffle "
    "order all cause it too) but worth confirming when two runs were meant to be identical.\n"
    "    RANKDIVERGE: <var> -- under a multi-rank launch, compares this rank's latest value of a "
    "tracked scalar against every other rank's. A rank whose value has drifted from the rest signals "
    "something rank-specific gone wrong (unsynced BatchNorm, a corrupted shard, stale weights).\n"
    "    RUNCOMPARE: -- this run's current scalar values vs the most recent previous run recorded for "
    "this project, so a regression against last time is visible without anyone remembering the numbers.\n"
    "    COST: -- running token/cost usage for this chat session's agent calls so far.\n\n"

    "CODE FIXES:\n"
    "If, and only if, the user explicitly asks you to fix, edit, patch, or change the code (not just "
    "diagnose it) AND training code has been sent this turn, respond with ONLY a single JSON object "
    "and nothing else -- no prose before or after it, no markdown code fences, no Diagnosis/Reasoning/"
    "Fix sections.\n"
    "PREFER THE SMALLEST FIX THAT ADDRESSES THE ROOT CAUSE: a single changed line, a changed argument, "
    "a swapped function call, or a few adjacent lines is almost always the right size for a bug fix. "
    "Do not rewrite a function, restructure a class, reformat unrelated code, or 'clean up' anything "
    "you weren't asked to touch, even if you notice something else that could be improved -- mention "
    "that separately in explanation/PASS 5 instead of folding it into this fix. Prefer one small old/"
    "new pair over one large one; only use several old/new pairs, or a large replacement block, when "
    "the root cause genuinely cannot be fixed with a smaller change (e.g. a bug that requires touching "
    "several call sites, or a block where the broken logic is inherently multi-line and can't be "
    "isolated to less). When in doubt, fix less, not more.\n"
    "The JSON object must have exactly these fields:\n"
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
    "enough surrounding lines (not just the single changed line) so the match is unambiguous -- but "
    "no more than that; a snippet padded with unrelated unchanged lines just to 'be safe' is still a "
    "larger fix than necessary and makes the change harder to review.\n"
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
    "OpenRouter (DeepSeek V4 Flash)": {
        "model": "openrouter/deepseek/deepseek-v4-flash",
        "env_key": "OPENROUTER_API_KEY",
        "prompt_title": "OpenRouter API Key Required",
        "prompt_msg": "Please enter your OpenRouter API Key (sk-or-...):"
    },
    # Sentinel entry: picking this prompts for any OpenRouter model slug,
    # then registers a real PROVIDERS entry for it via
    # register_openrouter_model -- see AgentSetupDialog.submit() and
    # ChatPanel._on_provider_change().
    "OpenRouter (any model)": {
        "openrouter": True,
        "env_key": "OPENROUTER_API_KEY",
        "prompt_title": "OpenRouter API Key Required",
        "prompt_msg": "Please enter your OpenRouter API Key (sk-or-...):"
    },
    # Sentinel entry: picking this prompts for a raw litellm model string
    # (e.g. "openai/gpt-6-astra", "ollama_chat/llama3.1") plus an optional
    # env var for its key, then dynamically registers a real PROVIDERS
    # entry for it -- see AgentSetupDialog.submit() and
    # ChatPanel._on_provider_change(), the two places this is handled.
    "Custom (enter provider/model manually)": {"custom": True},
}


def register_openrouter_model(slug):
    """Register an OpenRouter model slug (e.g. "deepseek/deepseek-v4-flash",
    with or without a leading "openrouter/") as a PROVIDERS entry and return
    its label. Stored like a "Custom: ..." entry, so the restart hand-off
    (PULSE_AUTO_CUSTOM_MODEL) re-registers it with no extra handling."""
    slug = slug.strip()
    if slug.lower().startswith("openrouter/"):
        slug = slug.split("/", 1)[1]
    label = f"OpenRouter: {slug}"
    PROVIDERS[label] = {
        "model": f"openrouter/{slug}",
        "env_key": "OPENROUTER_API_KEY",
        "prompt_title": "OpenRouter API Key Required",
        "prompt_msg": "Please enter your OpenRouter API Key (sk-or-...):"
    }
    return label

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
    "variable names, or code sections. Aim for the smallest region that could plausibly contain the "
    "root cause (often a single line or a few adjacent lines), not a whole function or file, unless "
    "the evidence genuinely doesn't narrow further than that. Respond with ONLY a short bullet list "
    "of the suspect location(s). No diagnosis, no fix yet."
)
_PASS2_ANALYZE_TMPL = (
    "Suspect region(s) from your first read:\n{regions}\n\n"
    "PASS 2 -- ANALYZE: Take a focused second look at just those regions. Give the Diagnosis (one "
    "sentence, the specific root cause) and the Reasoning behind it (grounded in the actual "
    "numbers/image/code you were given, with real math, referencing line numbers). Do not "
    "implement the fix yet."
)
_PASS3_FIX_TEXT_TMPL = (
    "Your analysis so far:\n{diagnosis}\n\n"
    "PASS 3 -- DEVELOP: Give the Fix: a concrete, concise change (not generic advice), in 1-3 "
    "sentences. Describe the smallest change that fixes the root cause -- a changed value, argument, "
    "or line -- not a broader rewrite."
)
_PASS3_IMPLEMENT_TMPL = (
    "Your analysis so far:\n{diagnosis}\n\n"
    "PASS 3 -- DEVELOP & IMPLEMENT: Implement the fix for the root cause diagnosed above. The user wants this fix applied to their code. Default to the "
    "smallest possible change -- a single line or a few adjacent lines -- and only widen the edit if "
    "the root cause genuinely can't be fixed that narrowly. Do not refactor, restructure, or rewrite "
    "code beyond what's needed to fix the diagnosed root cause. Respond with ONLY the code-fix JSON "
    "object described in your instructions (old/new/explanation) -- no prose, no markdown fences."
)
_PASS4_VERIFY_TMPL = (
    "Your analysis:\n{diagnosis}\n\n"
    "The fix you are about to apply:\n{fix_desc}\n\n"
    "PASS 4 -- VERIFY: Carefully check the math/logic of this fix against the numbers and code you "
    'were given. Also check its SCOPE: does it change only what\'s needed to fix the diagnosed root '
    "cause, or does it also rewrite/restructure/reformat code that didn't need to change? Respond "
    'with ONLY a JSON object of the form {{"passes": true or false, "reason": "one sentence"}}. '
    "passes=true only if the fix is logically/numerically correct, actually addresses the diagnosed "
    "root cause, AND is no larger than necessary to do so."
)
_PASS4_REVISE_TMPL = (
    "Your analysis:\n{diagnosis}\n\n"
    "The fix you proposed:\n{fix_desc}\n\n"
    "Your proposed fix did not pass verification: {reason}\n\n"
    "Revise it -- if the issue was scope (too large a change), narrow it down to the smallest edit "
    "that still fixes the root cause. Respond with ONLY the corrected code-fix JSON object (old/new/"
    "explanation) -- no prose, no markdown fences."
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
_GREP_RE = re.compile(r"^\s*GREP:\s*(.+)$", re.MULTILINE)
_VIEW_RE = re.compile(r"^\s*VIEW:\s*(.+)$", re.MULTILINE)


def _extract_directives(text):
    """Pull CALC:/PROMOTE:/GREP:/VIEW: lines out of an agent response,
    returning (cleaned_text, calc_exprs, promote_names, grep_patterns,
    view_requests). Cleaned text has those lines stripped so they don't
    clutter what's shown in the transcript.
    """
    calc_exprs = [m.strip() for m in _CALC_RE.findall(text) if m.strip()]
    promote_names = []
    for m in _PROMOTE_RE.findall(text):
        promote_names.extend(n.strip() for n in m.split(",") if n.strip())
    grep_patterns = [m.strip() for m in _GREP_RE.findall(text) if m.strip()]
    view_requests = [m.strip() for m in _VIEW_RE.findall(text) if m.strip()]

    cleaned = _CALC_RE.sub("", text)
    cleaned = _PROMOTE_RE.sub("", cleaned)
    cleaned = _GREP_RE.sub("", cleaned)
    cleaned = _VIEW_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, calc_exprs, promote_names, grep_patterns, view_requests


# ----------------------------------------------------------------------
# Extended toolset: execution, code-intelligence, statistics, and
# grounding directives. Kept in a separate dict-based extractor/applier
# (_extract_new_directives/_apply_new_directives on ChatPanel) rather than
# folded into _extract_directives/_apply_directives above, so none of that
# function's existing positional-tuple call sites need to change shape.
# ----------------------------------------------------------------------
_NEW_DIRECTIVE_RES = {
    "defof": re.compile(r"^\s*DEFOF:\s*(.+)$", re.MULTILINE),
    "callers": re.compile(r"^\s*CALLERS:\s*(.+)$", re.MULTILINE),
    "depgraph": re.compile(r"^\s*DEPGRAPH:\s*(.*)$", re.MULTILINE),
    "corr": re.compile(r"^\s*CORR:\s*(.+)$", re.MULTILINE),
    "outlier": re.compile(r"^\s*OUTLIER:\s*(.+)$", re.MULTILINE),
    "diffstats": re.compile(r"^\s*DIFFSTATS:\s*(.+)$", re.MULTILINE),
    "histogram": re.compile(r"^\s*HISTOGRAM:\s*(.+)$", re.MULTILINE),
    "doclookup": re.compile(r"^\s*DOCLOOKUP:\s*(.+)$", re.MULTILINE),
    "changelog": re.compile(r"^\s*CHANGELOG:\s*(.*)$", re.MULTILINE),
    "pastfix": re.compile(r"^\s*PASTFIX:\s*(.+)$", re.MULTILINE),
    "dryrun": re.compile(r"^\s*DRYRUN:\s*(.+)$", re.MULTILINE),
    "repl": re.compile(r"^\s*REPL:\s*(.+)$", re.MULTILINE),
    "replay": re.compile(r"^\s*REPLAY:\s*(.+)$", re.MULTILINE),
    "gradcheck": re.compile(r"^\s*GRADCHECK:\s*(.+)$", re.MULTILINE),
    "shapetrace": re.compile(r"^\s*SHAPETRACE:\s*(.*)$", re.MULTILINE),
    "gpustatus": re.compile(r"^\s*GPUSTATUS:\s*(.*)$", re.MULTILINE),
    "rollback": re.compile(r"^\s*ROLLBACK:\s*(.+)$", re.MULTILINE),
    "mllint": re.compile(r"^\s*MLLINT:\s*(.*)$", re.MULTILINE),
    "layerstats": re.compile(r"^\s*LAYERSTATS:\s*(.*)$", re.MULTILINE),
    "hardexamples": re.compile(r"^\s*HARDEXAMPLES:\s*(.*)$", re.MULTILINE),
    "ampstatus": re.compile(r"^\s*AMPSTATUS:\s*(.*)$", re.MULTILINE),
    "seedcheck": re.compile(r"^\s*SEEDCHECK:\s*(.*)$", re.MULTILINE),
    "rankdiverge": re.compile(r"^\s*RANKDIVERGE:\s*(.+)$", re.MULTILINE),
    "runcompare": re.compile(r"^\s*RUNCOMPARE:\s*(.*)$", re.MULTILINE),
    "cost": re.compile(r"^\s*COST:\s*(.*)$", re.MULTILINE),
}
# Directives that are meaningful with no argument at all (DEPGRAPH,
# CHANGELOG, bare SHAPETRACE) -- a bare match still counts as a request.
_FLAG_STYLE_DIRECTIVES = {"depgraph", "changelog", "shapetrace", "gpustatus", "mllint", "layerstats", "hardexamples", "ampstatus", "seedcheck", "runcompare", "cost"}


def _extract_new_directives(text):
    """Pull the extended toolset's directive lines out of an agent
    response. Returns (cleaned_text, requests) where requests is
    {directive_name: [arg, ...]} for every directive that actually
    appeared -- a flag-style directive with no argument still shows up
    as [''] so callers can just check truthiness/`in`.
    """
    requests = {}
    cleaned = text
    for key, rx in _NEW_DIRECTIVE_RES.items():
        raw = [m.strip() for m in rx.findall(text)]
        if key in _FLAG_STYLE_DIRECTIVES:
            if raw:
                requests[key] = raw
        else:
            raw = [m for m in raw if m]
            if raw:
                requests[key] = raw
        cleaned = rx.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, requests


def _format_exc_short(exc_text, max_lines=25):
    """Trim a traceback.format_exc() string down to something worth
    putting in a chat transcript / feeding back into an agent's context --
    keep the first few lines (exception chain header) and the tail (the
    actual raising frame + message), since that's almost always what
    matters and a full traceback of a deep training loop can be huge.
    """
    lines = (exc_text or "").strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:3] + ["    ... (truncated) ..."] + lines[-max_lines:]
    return "\n".join(lines)


def _exec_namespace(frame):
    """Build an eval/exec namespace out of a live frame -- globals first,
    then locals on top (locals shadow globals, same precedence Python
    itself uses when resolving a bare name inside that frame). This is a
    NAMESPACE-level snapshot, not full OS-process isolation: pickling a
    live CUDA/GPU tensor across a real subprocess boundary is unreliable
    (see the CPU-only-observation notes elsewhere in this file), so
    REPL/DRYRUN/etc. run directly against copies of the live objects
    instead. Mutating an object reached this way (e.g. calling a method
    with side effects) does affect the real, live training process --
    directives here are for real, in-place introspection, not a
    consequence-free sandbox.
    """
    ns = dict(frame.f_globals)
    ns.update(frame.f_locals)
    return ns


def _exec_eval(expr, ns):
    return eval(expr, {"__builtins__": __builtins__, "math": math}, ns)


def _describe_exec_value(value):
    desc = repr(value)
    if len(desc) > 800:
        desc = desc[:800] + "... (truncated)"
    extra = ""
    try:
        if is_trackable(value):
            extra = f"  shape={shape_of(value)} kind={tensor_kind(value)}"
    except Exception:
        pass
    return f"{desc}{extra}"


def _exec_repl(frame, expr):
    """REPL: <expr> -- evaluate a sandboxed expression against the live
    process's CURRENT globals/locals, instead of reasoning from a context
    dump that may already be stale."""
    ns = _exec_namespace(frame)
    try:
        value = _exec_eval(expr, ns)
    except Exception:
        return f"REPL '{expr}': raised\n{_format_exc_short(traceback.format_exc())}"
    return f"REPL '{expr}' = {_describe_exec_value(value)}"


def _exec_dryrun(frame, call_expr):
    """DRYRUN: <function>(<args>) -- execute a specific function/method
    call against the live process's current namespace and return the real
    output, exception, and traceback. Runs on a bounded-timeout thread so
    a hang doesn't block the training process's control-queue drain
    forever; if it's still running after the timeout, it keeps running in
    the background rather than being (unsafely) killed mid-call."""
    ns = _exec_namespace(frame)
    result = {}

    def _run():
        try:
            result["value"] = _exec_eval(call_expr, ns)
        except Exception:
            result["error"] = traceback.format_exc()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    if t.is_alive():
        return f"DRYRUN '{call_expr}': still running after 10s -- not waiting further (it keeps running in the background)."
    if "error" in result:
        return f"DRYRUN '{call_expr}': raised\n{_format_exc_short(result['error'])}"
    return f"DRYRUN '{call_expr}' -> {_describe_exec_value(result.get('value'))}"


def _exec_shapetrace(frame, arg):
    """SHAPETRACE: [optional model var name] -- run one forward pass
    against a live torch.nn.Module (named explicitly, or the first one
    found in scope) using a synthetic input built from a live tensor
    already in scope, and dump every submodule's input/output shape via
    forward hooks. Best-effort/PyTorch-only; eliminates shape-mismatch
    guessing for the common case of a single top-level model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return "SHAPETRACE: requires PyTorch to be importable in the training process; none found."

    ns = _exec_namespace(frame)
    model = None
    arg = (arg or "").strip()
    if arg and arg in ns and isinstance(ns[arg], nn.Module):
        model = ns[arg]
    if model is None:
        for v in ns.values():
            if isinstance(v, nn.Module):
                model = v
                break
    if model is None:
        return "SHAPETRACE: no torch.nn.Module found in scope to trace."

    sample = None
    for name in ("x", "inputs", "input", "batch", "images", "data"):
        if name in ns and torch.is_tensor(ns[name]):
            sample = ns[name]
            break
    if sample is None:
        for v in ns.values():
            if torch.is_tensor(v):
                sample = v
                break
    if sample is None:
        return "SHAPETRACE: no tensor found in scope to use as a synthetic input."

    records = []
    hooks = []

    def _make_hook(name):
        def _fn(module, inp, out):
            in_shapes = [tuple(t.shape) for t in inp if torch.is_tensor(t)]
            out_list = out if isinstance(out, (tuple, list)) else [out]
            out_shapes = [tuple(o.shape) for o in out_list if torch.is_tensor(o)]
            records.append(f"  {name} ({type(module).__name__}): in={in_shapes} out={out_shapes}")
        return _fn

    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(sample.clone())
        model.train(was_training)
    except Exception:
        return f"SHAPETRACE: forward pass raised\n{_format_exc_short(traceback.format_exc())}"
    finally:
        for h in hooks:
            h.remove()

    if not records:
        return "SHAPETRACE: forward pass ran but no submodule shapes were captured."
    header = f"SHAPETRACE (forward pass, synthetic input shape {tuple(sample.shape)}):\n"
    return header + "\n".join(records)


def _exec_gradcheck(frame, param_name):
    """GRADCHECK: <param> -- numerical finite-difference gradient check on
    a specific parameter. Deterministic pass/fail against the parameter's
    already-computed .grad, catching wrong backward-pass implementations
    without anyone reasoning through calculus. Requires a zero-argument
    callable named `loss_fn` in scope that recomputes and returns the
    current scalar loss purely from live model state -- there's no
    general way to know how to recompute a loss otherwise."""
    try:
        import torch
    except ImportError:
        return "GRADCHECK: requires PyTorch to be importable in the training process; none found."

    ns = _exec_namespace(frame)
    param_name = param_name.strip()
    param = ns.get(param_name)
    if param is None or not torch.is_tensor(param):
        param = None
        for v in ns.values():
            named_parameters = getattr(v, "named_parameters", None)
            if not callable(named_parameters):
                continue
            try:
                for pname, p in v.named_parameters():
                    if pname == param_name or pname.endswith("." + param_name):
                        param = p
                        break
            except Exception:
                continue
            if param is not None:
                break
    if param is None or not hasattr(param, "grad"):
        return (
            f"GRADCHECK '{param_name}': couldn't find a tensor parameter with a .grad by that name in "
            "scope (checked locals/globals directly and any model's named_parameters())."
        )
    if param.grad is None:
        return f"GRADCHECK '{param_name}': has no .grad yet -- call backward() at least once first."

    loss_fn = ns.get("loss_fn")
    if not callable(loss_fn):
        return (
            f"GRADCHECK '{param_name}': needs a zero-argument callable named `loss_fn` in scope that "
            "recomputes and returns the current scalar loss from live model state (e.g. "
            "`loss_fn = lambda: criterion(model(x), y)`) -- define one and resend GRADCHECK."
        )

    eps = 1e-3
    flat = param.data.view(-1)
    grad_flat = param.grad.view(-1)
    n_check = min(5, flat.numel())
    if n_check == 0:
        return f"GRADCHECK '{param_name}': parameter is empty."
    idxs = sorted({int(i) for i in torch.linspace(0, flat.numel() - 1, n_check).tolist()})
    lines = []
    max_rel_err = 0.0
    try:
        with torch.no_grad():
            for idx in idxs:
                orig = flat[idx].item()
                flat[idx] = orig + eps
                loss_plus = float(loss_fn())
                flat[idx] = orig - eps
                loss_minus = float(loss_fn())
                flat[idx] = orig
                numeric = (loss_plus - loss_minus) / (2 * eps)
                analytic = grad_flat[idx].item()
                denom = max(abs(numeric), abs(analytic), 1e-8)
                rel_err = abs(numeric - analytic) / denom
                max_rel_err = max(max_rel_err, rel_err)
                lines.append(f"  idx {idx}: analytic={analytic:.6g} numeric={numeric:.6g} rel_err={rel_err:.2e}")
    except Exception:
        flat[idxs[0]] = flat[idxs[0]]  # best-effort restore already happened per-iteration above
        return f"GRADCHECK '{param_name}': loss_fn() raised while probing\n{_format_exc_short(traceback.format_exc())}"

    verdict = "PASS" if max_rel_err < 1e-2 else "FAIL"
    header = f"GRADCHECK '{param_name}': {verdict} (max rel_err={max_rel_err:.2e} over {n_check} sampled entries)\n"
    return header + "\n".join(lines)


# Rolling ring buffer of (step, {name: serialized state_dict bytes})
# snapshots for REPLAY -- module-level and process-local, since the
# training process that owns the live model/optimizer is the only place
# this is ever populated or consumed (see _replay_maybe_checkpoint, called
# periodically from persistent_tracer's ticker, and _exec_replay below).
_REPLAY_CHECKPOINTS = []
_REPLAY_MAX_CHECKPOINTS = 20


def _replay_maybe_checkpoint(frame, step):
    """Snapshot every in-scope object exposing state_dict() (model,
    optimizer, ...) so REPLAY has something to roll back to. Cheap
    best-effort: skipped entirely if torch isn't importable, and any
    individual object that fails to serialize is just left out rather
    than aborting the whole snapshot."""
    try:
        import torch
    except ImportError:
        return
    ns = _exec_namespace(frame)
    snap = {}
    for name, v in ns.items():
        state_dict_fn = getattr(v, "state_dict", None)
        if not callable(state_dict_fn):
            continue
        try:
            buf = io.BytesIO()
            torch.save(v.state_dict(), buf)
            snap[name] = buf.getvalue()
        except Exception:
            continue
    if snap:
        _REPLAY_CHECKPOINTS.append((step, snap))
        del _REPLAY_CHECKPOINTS[:-_REPLAY_MAX_CHECKPOINTS]


def _exec_replay(frame, arg):
    """REPLAY: <n_steps> -- from the last checkpoint at least n_steps
    back, restore a copy of tracked state and replay n_steps via a
    zero-argument `train_step` callable the user defines in scope,
    reporting the resulting loss curve, then restore live state back to
    where it actually was. A much more conclusive check than restarting
    the whole process and seeing if it crashes -- correctness becomes an
    empirical fact instead of another LLM call asserting it."""
    try:
        n_steps = int(str(arg).strip())
    except ValueError:
        return f"REPLAY '{arg}': expected an integer number of steps, e.g. 'REPLAY: 5'."

    if not _REPLAY_CHECKPOINTS:
        return (
            "REPLAY: no checkpoints captured yet -- Pulse periodically snapshots any in-scope object "
            "with a state_dict() (model/optimizer); wait for at least one snapshot after training starts."
        )

    if n_steps >= len(_REPLAY_CHECKPOINTS):
        step, snap = _REPLAY_CHECKPOINTS[0]
    else:
        step, snap = _REPLAY_CHECKPOINTS[-1 - n_steps]

    try:
        import torch
    except ImportError:
        return "REPLAY: requires PyTorch to be importable in the training process; none found."

    ns = _exec_namespace(frame)
    train_step = ns.get("train_step")
    if not callable(train_step):
        return (
            "REPLAY: needs a zero-argument callable named `train_step` in scope that performs one "
            "optimizer step on the live model and returns the step's loss -- define one and resend REPLAY."
        )

    restored, originals = [], {}
    for name, blob in snap.items():
        obj = ns.get(name)
        if obj is None or not hasattr(obj, "load_state_dict"):
            continue
        try:
            originals[name] = copy.deepcopy(obj.state_dict())
            obj.load_state_dict(torch.load(io.BytesIO(blob)))
            restored.append(name)
        except Exception:
            continue
    if not restored:
        return "REPLAY: had a checkpoint but couldn't restore it into any live object (state_dict shape mismatch?)."

    def _restore_live():
        for name in restored:
            try:
                ns[name].load_state_dict(originals[name])
            except Exception:
                pass

    losses = []
    try:
        for _ in range(max(1, n_steps)):
            losses.append(float(train_step()))
    except Exception:
        _restore_live()
        return f"REPLAY: patched replay raised\n{_format_exc_short(traceback.format_exc())}"
    _restore_live()

    curve = ", ".join(f"{l:.6g}" for l in losses)
    return (
        f"REPLAY: replayed {len(losses)} step(s) from a checkpoint at step {step}, on an isolated copy of "
        f"tracked state, then restored the live process back to its actual current values. Resulting loss "
        f"curve: [{curve}]"
    )


def _handle_exec_request(frame, kind, arg):
    """Dispatch a single EXEC-style directive (REPL/DRYRUN/SHAPETRACE/
    GRADCHECK/REPLAY/LAYERSTATS/HARDEXAMPLES) against the live training
    process's frame. Runs IN the training process (see persistent_tracer's
    control_queue drain loop) -- never let an unexpected exception here
    propagate and take training down with it."""
    try:
        if kind == "REPL":
            return _exec_repl(frame, arg)
        if kind == "DRYRUN":
            return _exec_dryrun(frame, arg)
        if kind == "SHAPETRACE":
            return _exec_shapetrace(frame, arg)
        if kind == "GRADCHECK":
            return _exec_gradcheck(frame, arg)
        if kind == "REPLAY":
            return _exec_replay(frame, arg)
        if kind == "LAYERSTATS":
            return _exec_layerstats(frame, arg)
        if kind == "HARDEXAMPLES":
            return _exec_hardexamples(frame, arg)
        if kind == "AMPSTATUS":
            return _exec_ampstatus(frame, arg)
        if kind == "SEEDCHECK":
            return _exec_seedcheck(frame, arg)
        return f"{kind}: unknown execution directive."
    except Exception:
        return f"{kind} '{arg}': Pulse's own handler raised while trying to run this:\n{_format_exc_short(traceback.format_exc())}"


def _exec_find_model(ns, arg=""):
    """Shared model-finding heuristic used by SHAPETRACE/LAYERSTATS: an
    explicitly-named variable if given and it checks out, else the first
    torch.nn.Module found in the namespace."""
    import torch.nn as nn
    arg = (arg or "").strip()
    if arg and arg in ns and isinstance(ns[arg], nn.Module):
        return ns[arg]
    for v in ns.values():
        if isinstance(v, nn.Module):
            return v
    return None


def _exec_layerstats(frame, arg):
    """LAYERSTATS: [optional model variable name] -- per-layer gradient
    norm, weight norm, and their ratio for every parameter of a live
    torch.nn.Module, right after backward() has populated .grad. The
    ratio is Karpathy's classic training-health sanity check (a healthy
    ratio is usually roughly 1e-4 to 1e-2 per step for SGD-like updates);
    a layer near 0 signals a vanishing-gradient dead layer, a layer near
    or above 1 signals a likely-exploding one. This is a raw grad/weight
    norm ratio, not literally "how much the optimizer will move the
    weight" -- for adaptive optimizers (Adam etc.) the real per-step
    update also depends on that optimizer's own internal moment
    estimates, which this can't see, so treat the numbers as a relative
    signal across layers, not an exact update size. PyTorch only."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return "LAYERSTATS: requires PyTorch to be importable in the training process; none found."

    ns = _exec_namespace(frame)
    model = _exec_find_model(ns, arg)
    if model is None:
        return "LAYERSTATS: no torch.nn.Module found in scope."

    rows = []
    any_grad = False
    for name, param in model.named_parameters():
        if param.grad is None:
            rows.append((name, tuple(param.shape), None, None, None))
            continue
        any_grad = True
        with torch.no_grad():
            w_norm = float(param.data.norm().item())
            g_norm = float(param.grad.norm().item())
        ratio = g_norm / (w_norm + 1e-12)
        rows.append((name, tuple(param.shape), w_norm, g_norm, ratio))

    if not any_grad:
        return "LAYERSTATS: found the model but no parameter has a .grad yet -- call backward() at least once first."

    scored = [r for r in rows if r[4] is not None]
    scored.sort(key=lambda r: r[4])
    lines = []
    for name, shape, w_norm, g_norm, ratio in scored:
        flag = ""
        if ratio < 1e-6:
            flag = "  <- near-zero, possible dead/vanishing layer"
        elif ratio > 1.0:
            flag = "  <- large, possible exploding layer"
        lines.append(f"  {name} {shape}: weight_norm={w_norm:.4g} grad_norm={g_norm:.4g} ratio={ratio:.2e}{flag}")
    no_grad_names = [name for name, _s, w, g, r in rows if r is None]
    footer = f"\n  ({len(no_grad_names)} parameter(s) with no .grad, not shown: {', '.join(no_grad_names[:5])}{'...' if len(no_grad_names) > 5 else ''})" if no_grad_names else ""
    return f"LAYERSTATS ({len(scored)} parameter(s) with gradients, sorted low->high ratio):\n" + "\n".join(lines) + footer


def _exec_hardexamples(frame, arg):
    """HARDEXAMPLES: [optional N, default 10] -- per-example loss for the
    current batch, surfacing the N highest-loss samples. Often the
    fastest way to spot label noise or a preprocessing bug: a handful of
    examples with wildly higher loss than everything else is a strong
    signal, much stronger than an aggregate loss number. Needs a
    zero-argument callable named `per_example_losses_fn` in scope that
    recomputes and returns a per-sample loss array/tensor (e.g. a
    criterion called with reduction='none') for the current batch -- no
    general way to know how to get per-sample losses otherwise. If a
    tracked `sample_ids` (or `indices`) array of the same length is also
    in scope, its values are used to label examples instead of raw
    positions."""
    ns = _exec_namespace(frame)
    fn = ns.get("per_example_losses_fn")
    if not callable(fn):
        return (
            "HARDEXAMPLES: needs a zero-argument callable named `per_example_losses_fn` in scope that "
            "returns a per-sample loss array for the current batch (e.g. "
            "`per_example_losses_fn = lambda: F.cross_entropy(model(x), y, reduction='none')`) -- "
            "define one and resend HARDEXAMPLES."
        )
    try:
        losses = fn()
    except Exception:
        return f"HARDEXAMPLES: per_example_losses_fn() raised\n{_format_exc_short(traceback.format_exc())}"

    try:
        import torch
        if torch.is_tensor(losses):
            losses = losses.detach().cpu().numpy()
    except ImportError:
        pass
    try:
        losses = np.asarray(losses, dtype=float).reshape(-1)
    except Exception:
        return f"HARDEXAMPLES: per_example_losses_fn() returned something that isn't a flat numeric array ({type(losses)})."

    n = 10
    arg = (arg or "").strip()
    if arg:
        try:
            n = max(1, int(arg))
        except ValueError:
            return f"HARDEXAMPLES '{arg}': expected an integer N, e.g. 'HARDEXAMPLES: 20'."

    ids = ns.get("sample_ids", ns.get("indices"))
    id_arr = None
    if ids is not None:
        try:
            import torch
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().numpy()
        except ImportError:
            pass
        try:
            id_arr = np.asarray(ids).reshape(-1)
            if len(id_arr) != len(losses):
                id_arr = None
        except Exception:
            id_arr = None

    order = np.argsort(-losses)[:n]
    lines = []
    for rank, idx in enumerate(order, 1):
        label = f"id={id_arr[idx]}" if id_arr is not None else f"batch-position {idx}"
        lines.append(f"  #{rank}: {label}, loss={losses[idx]:.6g}")
    mean, std = float(losses.mean()), float(losses.std())
    return (
        f"HARDEXAMPLES: top {len(order)} of {len(losses)} example(s) by loss "
        f"(batch mean={mean:.4g}, std={std:.4g}):\n" + "\n".join(lines)
    )


def _exec_ampstatus(frame, arg):
    """AMPSTATUS: [optional GradScaler variable name] -- current mixed-
    precision (torch.cuda.amp.GradScaler) state: scale factor and, where
    obtainable, the running count of skipped steps due to inf/NaN
    gradients. A scale that's collapsed toward 1.0 (from a much larger
    starting value like 65536) or a high skip count both point at fp16
    numerical instability -- exactly the kind of thing that's otherwise
    invisible unless you go looking for it."""
    ns = _exec_namespace(frame)
    arg = (arg or "").strip()
    scaler = ns.get(arg) if arg else None
    if scaler is None:
        for v in ns.values():
            if type(v).__name__ == "GradScaler":
                scaler = v
                break
    if scaler is None:
        return "AMPSTATUS: no GradScaler found in scope (this training run may not be using AMP)."
    try:
        scale = float(scaler.get_scale())
    except Exception:
        scale = None
    growth_tracker = getattr(scaler, "_get_growth_tracker", None)
    skips = None
    try:
        state = scaler.state_dict()
        skips = state.get("_growth_tracker")
    except Exception:
        pass
    parts = []
    if scale is not None:
        parts.append(f"current scale factor: {scale:.6g}")
    if skips is not None:
        parts.append(f"growth tracker (consecutive non-skipped steps at current scale): {skips}")
    if not parts:
        return "AMPSTATUS: found a GradScaler but couldn't read its internal state (API may have changed)."
    return "AMPSTATUS:\n  " + "\n  ".join(parts)


def _run_history_path(script_path):
    base = os.path.dirname(script_path) if script_path else "."
    return os.path.join(base, ".pulse_run_history.json")


def _load_run_history(script_path):
    try:
        with open(_run_history_path(script_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _seed_fingerprint():
    """Best-effort, deterministic fingerprint of every RNG this process
    can introspect. NumPy/stdlib `random` don't expose the ORIGINAL seed
    once reseeded, only their large internal generator state -- still
    useful as a "did anything about RNG state change" fingerprint even
    though it's not a human-readable seed number. PyTorch's
    initial_seed() IS the real recoverable seed value, when available."""
    info = {}
    try:
        import random as _random
        info["python_random_state_hash"] = hashlib.sha256(repr(_random.getstate()).encode()).hexdigest()[:16]
    except Exception:
        pass
    try:
        state = np.random.get_state()
        info["numpy_state_hash"] = hashlib.sha256(repr(state).encode()).hexdigest()[:16]
    except Exception:
        pass
    try:
        import torch
        info["torch_initial_seed"] = torch.initial_seed()
        if torch.cuda.is_available():
            info["torch_cuda_initial_seed"] = torch.cuda.initial_seed()
    except Exception:
        pass
    info["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")
    return info


def _seed_history_path(script_path):
    base = os.path.dirname(script_path) if script_path else "."
    return os.path.join(base, ".pulse_seed_history.json")


def _exec_seedcheck(frame, arg):
    """SEEDCHECK: -- a reproducibility fingerprint of every RNG this
    process can see, compared against the fingerprint from the last time
    SEEDCHECK ran on this project. A mismatch isn't necessarily wrong --
    intentional reseeding, a new data shuffle order, or simply more steps
    having run before checking will all show up as a difference too --
    but an UNEXPECTED difference (e.g. between two runs meant to be
    identical for a REPLAY/A-B comparison) is worth flagging instead of
    silently assuming determinism held.
    """
    current = _seed_fingerprint()
    path = _seed_history_path(frame.f_code.co_filename)
    previous = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            previous = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current, f)
    except OSError:
        pass

    lines = [f"  {k}: {v}" for k, v in current.items() if v is not None]
    header = "SEEDCHECK: current RNG fingerprint\n" + "\n".join(lines)
    if previous is None:
        return header + "\n\n(no prior fingerprint recorded for this project -- this is now the baseline for future SEEDCHECK calls.)"
    diffs = [k for k in current if current.get(k) != previous.get(k)]
    if not diffs:
        return header + "\n\nMatches the last recorded fingerprint for this project exactly."
    diff_lines = [f"  {k}: was {previous.get(k)!r}, now {current.get(k)!r}" for k in diffs]
    return (
        header + "\n\nDIFFERS from the last recorded fingerprint in:\n" + "\n".join(diff_lines) +
        "\n(this alone doesn't mean anything is wrong -- more steps run, an intentional reseed, or a "
        "different data shuffle order would all cause this too -- but worth confirming it's expected.)"
    )


# ----------------------------------------------------------------------
# ML anti-pattern static checks (MLLINT) -- see the matching copy in
# pulse_cli.py. A small, high-confidence set of
# AST-detectable patterns worded as things "worth double-checking," not
# certainties. Organized by scope: general/cross-backend patterns that
# apply regardless of which of numpy/torch/tf-keras/jax/cupy a script
# uses, then patterns specific to one backend's idioms.
# ----------------------------------------------------------------------
_MLLINT_REGRESSION_LOSSES = {
    "mse", "mae", "mean_squared_error", "mean_absolute_error", "msle",
    "mean_squared_logarithmic_error", "huber", "huber_loss", "logcosh",
}
_MLLINT_ACCURACY_METRICS = {
    "accuracy", "acc", "categorical_accuracy", "binary_accuracy",
    "sparse_categorical_accuracy", "top_k_categorical_accuracy",
}
_MLLINT_CLASSIFICATION_LOSSES = {
    "categorical_crossentropy", "sparse_categorical_crossentropy",
    "binary_crossentropy", "crossentropy", "hinge", "categorical_hinge",
    "kld", "kullback_leibler_divergence",
}
_MLLINT_REGRESSION_METRICS = {
    "mae", "mse", "mean_absolute_error", "mean_squared_error", "rmse",
    "r2", "r2_score", "msle", "mean_squared_logarithmic_error",
}
# Optimizer constructors across torch, keras/tf, and optax (jax) -- kept
# as one set since the "suspicious lr" check is otherwise identical for
# all three. optax's own constructors are plain lowercase functions
# (optax.adam(...)), unlike torch/keras classes, hence the mixed case.
_MLLINT_OPTIMIZER_NAMES = {
    "Adam", "SGD", "RMSprop", "Adagrad", "AdamW", "Adadelta", "NAdam",
    "Nadam", "RAdam", "Adamax", "ASGD", "Rprop", "LBFGS", "SparseAdam",
    "Ftrl",
    # optax (jax)
    "adam", "sgd", "adamw", "rmsprop", "adagrad", "adabelief", "lamb",
    "novograd", "yogi", "radam", "lars", "fromage", "adamax", "noisy_sgd",
    "sm3",
}
# Initializer kwargs that set a layer's *weight* matrix (as opposed to its
# bias, where zeros is the normal/harmless default). Zero-initializing any
# of these breaks symmetry: every unit in the layer starts identical, sees
# an identical gradient, and (for a layer fed by a single upstream tensor)
# can stay identical for a very long time. This is a genuinely silent bug:
# a network with a zeroed weight matrix deeper than the input layer can
# still limp forward via asymmetry introduced elsewhere (a nonzero bias,
# an adjacent randomly-initialized layer feeding it a nonzero backprop
# signal), so the loss curve often still trends down -- just far slower
# and far worse than intended -- which is exactly the shape that the
# runtime "loss never improved at all" detector cannot see.
_MLLINT_WEIGHT_INITIALIZER_PARAMS = {
    "kernel_initializer", "depthwise_initializer", "pointwise_initializer",
    "recurrent_kernel_initializer", "embeddings_initializer",
}
_MLLINT_ZERO_INITIALIZER_NAMES = {"zeros", "zero", "Zeros", "zeros_initializer"}
# Layer types whose entire purpose is to randomly zero out units -- a rate
# of 1.0 (or a torch `p=1.0`) zeroes *every* unit on *every* forward pass
# during training, so no signal at all passes through the layer. Loss can
# still visibly decrease (whatever the layers before/after are able to do
# on their own), so this is another "trains to completion looking mostly
# normal" bug rather than a crash or an obviously flat loss curve.
_MLLINT_DROPOUT_LAYER_NAMES = {
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d",
    "SpatialDropout1D", "SpatialDropout2D", "SpatialDropout3D",
    "AlphaDropout", "GaussianDropout",
}
# A PRNG reseed call across every backend Pulse supports: np.random.seed,
# random.seed, torch.manual_seed, tf.random.set_seed, cp.random.seed all
# end in one of these names (JAX has no global seed to reseed -- keys are
# explicit values, which is exactly why it gets its own key-reuse check
# instead, below).
_MLLINT_SEED_FUNC_NAMES = {"seed", "manual_seed", "set_seed", "set_random_seed"}
# sklearn-style fit calls -- used for the "fit on test data" leak check.
_MLLINT_FIT_CALL_NAMES = {"fit", "fit_transform"}
# Keras/TF loss classes that take a `from_logits=` kwarg. Paired with an
# already-activated final layer, this is the Keras equivalent of the
# torch double-softmax/double-sigmoid checks below.
_MLLINT_LOGITS_LOSS_CLASSES = {
    "CategoricalCrossentropy", "SparseCategoricalCrossentropy", "BinaryCrossentropy",
}
_MLLINT_TEST_NAME_RE = re.compile(r"(?:^|_)test(?:$|_)", re.IGNORECASE)
_MLLINT_TRAIN_NAME_RE = re.compile(r"(?:^|_)train(?:$|_)", re.IGNORECASE)
_MLLINT_LOSSY_ACCUM_TARGET_RE = re.compile(r"(?:^|_)(loss|total|running|epoch)(?:$|_)", re.IGNORECASE)
_MLLINT_LOSS_NAME_RE = re.compile(r"(?:^|_)loss(?:$|_)", re.IGNORECASE)
_MLLINT_JAX_KEY_NAME_RE = re.compile(r"(?:^|_)(key|rng)(?:$|_)", re.IGNORECASE)


def _mllint_const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _mllint_list_of_str(node):
    if isinstance(node, (ast.List, ast.Tuple)):
        return [s for s in (_mllint_const_str(e) for e in node.elts) if s is not None]
    return []


def _mllint_fname(node):
    """The bare (unqualified) function name of a Call node, e.g. 'Zeros'
    for both `Zeros()` and `initializers.Zeros()`."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _mllint_is_zero_initializer(node):
    """True if `node` (an AST expression) is recognizably a zero
    initializer: the string 'zeros'/'zero', a bare Zeros()/zeros_initializer()
    call (e.g. `initializers.Zeros()`, `tf.zeros_initializer()`), or a
    Constant-style initializer whose value is literally 0."""
    s = _mllint_const_str(node)
    if s is not None:
        return s.lower() in {"zeros", "zero"}
    if isinstance(node, ast.Call):
        fname = _mllint_fname(node)
        if fname in _MLLINT_ZERO_INITIALIZER_NAMES:
            return True
        if fname == "Constant":
            val_node = node.keywords[0].value if node.keywords and node.keywords[0].arg in (None, "value") else (
                node.args[0] if node.args else None)
            if isinstance(val_node, ast.Constant) and isinstance(val_node.value, (int, float)):
                return val_node.value == 0
    return False


def _mllint_numeric_const(node):
    """Return the numeric value of `node` if it's a numeric literal,
    handling unary minus too (`-0.01` parses as UnaryOp(USub, Constant(0.01)),
    not as a negative Constant, so a plain isinstance(node, ast.Constant)
    check silently misses every negative literal)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _mllint_numeric_const(node.operand)
        if inner is not None:
            return -inner
    return None


def _mllint_numeric_kwarg_or_arg(node, kwarg_names, arg_index=0):
    """Return the numeric value passed either as one of `kwarg_names` or
    as the positional arg at `arg_index`, else None."""
    for kw in node.keywords:
        if kw.arg in kwarg_names:
            val = _mllint_numeric_const(kw.value)
            if val is not None:
                return val
    if len(node.args) > arg_index:
        return _mllint_numeric_const(node.args[arg_index])
    return None


def _mllint_is_grad_result(call_node):
    """True for a direct `grad(...)`/`value_and_grad(...)` call, or for
    JAX's common curried form `jax.grad(f)(params, x, y)`, where the outer
    call's own `.func` is itself a call to grad/value_and_grad."""
    if _mllint_fname(call_node) in ("grad", "value_and_grad"):
        return True
    if isinstance(call_node.func, ast.Call) and _mllint_fname(call_node.func) in ("grad", "value_and_grad"):
        return True
    return False


def _mllint_assign_target_names(assign_node):
    """Flat list of plain Name targets of an Assign, unpacking simple
    tuple/list targets (`a, b = ...`) one level deep."""
    names = []
    for tgt in assign_node.targets:
        if isinstance(tgt, ast.Name):
            names.append(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for elt in tgt.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
    return names


def _mllint_find_calls_in_loops(tree, names):
    """Depth-first walk returning every Call node whose bare function name
    is in `names` and that sits lexically inside the body of a For/While/
    AsyncFor loop somewhere above it (a loop the call node itself might
    introduce doesn't count)."""
    found = []

    def visit(node, in_loop):
        child_in_loop = in_loop or isinstance(node, (ast.For, ast.While, ast.AsyncFor))
        if isinstance(node, ast.Call) and in_loop:
            fname = _mllint_fname(node)
            if fname in names:
                found.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, child_in_loop)

    visit(tree, False)
    return found


def _mllint_scan(trees) -> list[tuple]:
    """trees: iterable of (label, path, text, tree). Returns a list of
    (label, lineno, message) findings."""
    findings = []
    softmax_loc = crossentropy_loc = sigmoid_loc = bce_logits_loc = None
    softmax_module_loc = nllloss_loc = None
    trees = list(trees)

    for label, _path, _text, tree in trees:
        # Per-file accumulators used by the cross-check passes below.
        activation_calls = []      # [(lineno, 'softmax'|'sigmoid'|...)]
        compile_infos = []         # [{'lineno', 'loss_str', 'loss_call_fname', 'from_logits', 'metrics'}]
        to_categorical_loc = None
        one_hot_encoder_loc = None
        label_encoder_loc = None
        one_hot_torch_loc = None
        ce_or_nll_loss_loc = None

        # --------------------------------------------------------------
        # Pass 1: every Call node, single walk.
        # --------------------------------------------------------------
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = _mllint_fname(node)
            if fname is None:
                continue

            # ---- Keras/TF: loss/metric mismatches at model.compile() ----
            if fname == "compile":
                loss_val, metrics_val = None, None
                loss_call_fname, from_logits = None, False
                for kw in node.keywords:
                    if kw.arg == "loss":
                        loss_val = _mllint_const_str(kw.value)
                        if loss_val is None and isinstance(kw.value, ast.Call):
                            loss_call_fname = _mllint_fname(kw.value)
                            from_logits = any(
                                k.arg == "from_logits" and isinstance(k.value, ast.Constant) and k.value.value is True
                                for k in kw.value.keywords
                            )
                    elif kw.arg == "metrics":
                        metrics_val = _mllint_list_of_str(kw.value)
                compile_infos.append({
                    "lineno": node.lineno, "loss_str": loss_val,
                    "loss_call_fname": loss_call_fname, "from_logits": from_logits,
                    "metrics": metrics_val,
                })

                if loss_val and metrics_val and loss_val.lower() in _MLLINT_REGRESSION_LOSSES:
                    if any(m.lower() in _MLLINT_ACCURACY_METRICS for m in metrics_val):
                        findings.append((label, node.lineno,
                            f"model.compile(loss='{loss_val}', metrics={metrics_val}): '{loss_val}' is a "
                            "regression loss, but an accuracy-style metric was also requested. Accuracy is "
                            "an exact-match classification metric and is typically meaningless (often "
                            "silently stuck at 0.0 for the whole run) against a continuous target -- worth "
                            "double-checking mae/rmse/r2 or similar is what's actually meant to be watched."))

                if loss_val and metrics_val and loss_val.lower() in _MLLINT_CLASSIFICATION_LOSSES:
                    bad_metrics = [m for m in metrics_val if m.lower() in _MLLINT_REGRESSION_METRICS]
                    if bad_metrics:
                        findings.append((label, node.lineno,
                            f"model.compile(loss='{loss_val}', metrics={metrics_val}): '{loss_val}' is a "
                            f"classification loss, but {bad_metrics} is a regression-only metric. mae/mse/r2 "
                            "compare continuous values and are typically meaningless against class labels -- "
                            "worth double-checking accuracy or similar is what's actually meant to be watched."))

            if fname in ("Softmax", "LogSoftmax", "softmax", "log_softmax") and softmax_loc is None:
                softmax_loc = (label, node.lineno)
            if fname in ("CrossEntropyLoss", "cross_entropy") and crossentropy_loc is None:
                crossentropy_loc = (label, node.lineno)
            if fname in ("Sigmoid", "sigmoid") and sigmoid_loc is None:
                sigmoid_loc = (label, node.lineno)
            if fname == "BCEWithLogitsLoss" and bce_logits_loc is None:
                bce_logits_loc = (label, node.lineno)
            if fname == "Softmax" and softmax_module_loc is None:
                softmax_module_loc = (label, node.lineno)
            if fname == "NLLLoss" and nllloss_loc is None:
                nllloss_loc = (label, node.lineno)
            if fname in ("CrossEntropyLoss", "NLLLoss") and ce_or_nll_loss_loc is None:
                ce_or_nll_loss_loc = node.lineno
            if fname == "to_categorical" and to_categorical_loc is None:
                to_categorical_loc = node.lineno
            if fname == "OneHotEncoder" and one_hot_encoder_loc is None:
                one_hot_encoder_loc = node.lineno
            if fname == "LabelEncoder" and label_encoder_loc is None:
                label_encoder_loc = node.lineno
            if fname == "one_hot" and one_hot_torch_loc is None:
                one_hot_torch_loc = node.lineno

            # Collect every string-literal `activation=` kwarg anywhere in
            # the file; the cross-check pass below treats the latest one
            # (by line number) before compile() as "the final layer's
            # activation" -- a coarse but effective proxy for these
            # typically-linear Sequential-style scripts.
            for kw in node.keywords:
                if kw.arg == "activation":
                    act_val = _mllint_const_str(kw.value)
                    if act_val:
                        activation_calls.append((node.lineno, act_val))

            # ---- GENERAL: zero-initialized weight matrix (any backend
            # exposing a `*_initializer=` kwarg -- Keras/TF layers) ----
            for kw in node.keywords:
                if kw.arg in _MLLINT_WEIGHT_INITIALIZER_PARAMS and _mllint_is_zero_initializer(kw.value):
                    findings.append((label, node.lineno,
                        f"{fname}(..., {kw.arg}='zeros'): zero-initializing a layer's weights means every "
                        "unit starts identical and (absent asymmetry introduced elsewhere) receives an "
                        "identical gradient, so the layer can fail to break symmetry for a very long time. "
                        "Unlike a totally dead network, a run like this can still show the loss slowly "
                        "improving via other layers, so it won't necessarily look flat -- worth "
                        "double-checking a non-zero default (e.g. 'glorot_uniform'/'he_normal') was meant "
                        "here instead."))

            # ---- PyTorch equivalent: nn.init.zeros_(module.weight) ----
            if fname == "zeros_" and node.args:
                arg_src = ast.dump(node.args[0]).lower()
                if "bias" not in arg_src:
                    findings.append((label, node.lineno,
                        "nn.init.zeros_(...) called here on what looks like a weight tensor (not a bias) -- "
                        "zero-initializing a layer's weight matrix breaks symmetry the same way a Keras "
                        "kernel_initializer='zeros' does: every unit starts identical and can receive an "
                        "identical gradient for a long time. Zero-initializing a *bias* is normal and "
                        "harmless; zero-initializing a *weight* usually isn't -- worth double-checking this "
                        "wasn't meant to target .bias instead of .weight, or use a proper weight init (e.g. "
                        "nn.init.xavier_uniform_/kaiming_normal_) instead."))

            # ---- GENERAL: dropout rate of 1.0 (Keras `rate=`/torch `p=`) ----
            if fname in _MLLINT_DROPOUT_LAYER_NAMES:
                rate = _mllint_numeric_kwarg_or_arg(node, {"rate", "p"})
                if rate is not None and rate >= 1.0:
                    findings.append((label, node.lineno,
                        f"{fname}(rate={rate}): a dropout rate of 1.0 zeroes out every unit on every "
                        "forward pass during training, so no signal passes through this layer at all. "
                        "The loss can still visibly decrease using whatever the surrounding layers can do "
                        "on their own, so this won't necessarily look like a stalled run -- worth "
                        "double-checking this wasn't meant to be a much smaller value (e.g. 0.1-0.5)."))

            # ---- GENERAL: suspicious learning rate, any backend ----
            # Checks both `lr=` (torch, older keras) and `learning_rate=`
            # (current keras/tf, optax) -- the old version only checked
            # `lr`, silently missing every keras/tf/optax script.
            if fname in _MLLINT_OPTIMIZER_NAMES:
                lr_node = None
                for kw in node.keywords:
                    if kw.arg in ("lr", "learning_rate"):
                        lr_node = kw.value
                if lr_node is None and node.args:
                    lr_node = node.args[-1] if len(node.args) >= 2 else None
                lr_val = _mllint_numeric_const(lr_node) if lr_node is not None else None
                if lr_val is not None:
                    if lr_val > 1.0:
                        findings.append((label, node.lineno,
                            f"{fname}(..., lr={lr_val}): a learning rate above 1.0 is unusually high "
                            "for almost any optimizer/architecture combination and often causes immediate "
                            "divergence -- worth double-checking this wasn't meant to be a smaller value "
                            "(e.g. missing an extra leading zero or an accidental e2/e-2 typo)."))
                    elif lr_val <= 0:
                        findings.append((label, node.lineno,
                            f"{fname}(..., lr={lr_val}): a learning rate of {lr_val} means the "
                            "optimizer will never move the weights at all (zero) or will move them in the "
                            "wrong direction on every single step (negative). The run will train to "
                            "completion looking superficially normal -- no crash, a loss value every epoch "
                            "-- while either never learning anything or actively getting worse. Worth "
                            "double-checking this wasn't meant to be a small positive value."))

            # ---- GENERAL: data leakage -- fitting a transformer on
            # something that looks like held-out data ----
            if fname in _MLLINT_FIT_CALL_NAMES and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Name) and _MLLINT_TEST_NAME_RE.search(first_arg.id):
                    findings.append((label, node.lineno,
                        f".{fname}({first_arg.id}, ...): fitting on a variable named '{first_arg.id}' -- "
                        "scalers/encoders/vectorizers (and models) should only ever be *fit* on training "
                        "data and *applied* (.transform()/.predict()) to test data; fitting on test data "
                        "leaks its statistics into preprocessing and inflates apparent performance without "
                        "any error or crash. Worth double-checking this wasn't meant to be the training "
                        "split, or a .transform() call instead of .fit()/.fit_transform()."))

        # --------------------------------------------------------------
        # Pass 2: cross-checks that need the whole-file collections above.
        # --------------------------------------------------------------
        final_activation = None
        if activation_calls:
            first_compile_line = min((c["lineno"] for c in compile_infos), default=None)
            pool = (
                [a for a in activation_calls if first_compile_line is None or a[0] < first_compile_line]
                or activation_calls
            )
            final_activation = max(pool, key=lambda a: a[0])

        for cinfo in compile_infos:
            loss_str = (cinfo["loss_str"] or "").lower()

            if final_activation:
                _fa_line, fa_val = final_activation
                fa_val_l = fa_val.lower()
                if fa_val_l == "sigmoid" and loss_str in ("categorical_crossentropy", "sparse_categorical_crossentropy"):
                    findings.append((label, cinfo["lineno"],
                        f"model.compile(loss='{cinfo['loss_str']}', ...): the final layer's activation is "
                        "'sigmoid' (a single-probability, binary output), but the loss is a multi-class "
                        "crossentropy that expects a probability distribution over multiple classes -- worth "
                        "double-checking the final layer should have `activation='softmax'` with enough "
                        "units for the number of classes, or the loss should be 'binary_crossentropy'."))
                if fa_val_l == "softmax" and loss_str == "binary_crossentropy":
                    findings.append((label, cinfo["lineno"],
                        f"model.compile(loss='{cinfo['loss_str']}', ...): the final layer's activation is "
                        "'softmax' (a multi-class probability distribution), but 'binary_crossentropy' "
                        "expects a single independent probability per output unit -- worth double-checking "
                        "the final layer should have `activation='sigmoid'`, or the loss should be a "
                        "categorical crossentropy variant."))

            # Keras/TF "double softmax"/"double sigmoid" via an
            # already-activated final layer feeding a from_logits=True loss.
            if cinfo.get("loss_call_fname") in _MLLINT_LOGITS_LOSS_CLASSES and cinfo.get("from_logits") and final_activation:
                _fa_line, fa_val = final_activation
                fa_val_l = fa_val.lower()
                mismatched = (
                    (cinfo["loss_call_fname"] in ("CategoricalCrossentropy", "SparseCategoricalCrossentropy") and fa_val_l == "softmax")
                    or (cinfo["loss_call_fname"] == "BinaryCrossentropy" and fa_val_l == "sigmoid")
                )
                if mismatched:
                    findings.append((label, cinfo["lineno"],
                        f"{cinfo['loss_call_fname']}(from_logits=True) is paired with a final layer whose "
                        f"activation is already '{fa_val}' -- from_logits=True tells the loss to apply "
                        f"{'softmax' if fa_val_l == 'softmax' else 'sigmoid'} internally, so an already-"
                        f"activated output is a 'double {fa_val_l}': it typically flattens gradients and "
                        "hurts convergence without ever erroring. Worth double-checking either the final "
                        "layer's activation should be removed (feed raw logits), or from_logits should be "
                        "False here."))

            # to_categorical() (one-hot) + sparse_categorical_crossentropy
            # (wants integer labels) -- a common, silent shape mismatch.
            if to_categorical_loc is not None and loss_str == "sparse_categorical_crossentropy":
                findings.append((label, cinfo["lineno"],
                    "model.compile(loss='sparse_categorical_crossentropy', ...) here, combined with a "
                    f"to_categorical(...) call at line {to_categorical_loc} -- sparse_categorical_crossentropy "
                    "expects integer class labels, but to_categorical(...) produces one-hot vectors. Worth "
                    "double-checking either the loss should be plain 'categorical_crossentropy', or the "
                    "labels shouldn't be one-hot encoded."))

            # LabelEncoder (integer labels) + a one-hot-expecting loss,
            # with no to_categorical/OneHotEncoder anywhere to fix it up.
            if (
                label_encoder_loc is not None
                and to_categorical_loc is None
                and one_hot_encoder_loc is None
                and loss_str in ("categorical_crossentropy", "hinge", "categorical_hinge", "kld", "kullback_leibler_divergence")
            ):
                findings.append((label, cinfo["lineno"],
                    f"model.compile(loss='{cinfo['loss_str']}', ...) here expects one-hot encoded targets, "
                    f"but the only label preparation found in this file is LabelEncoder (line {label_encoder_loc}), "
                    "which produces integer labels, not one-hot vectors -- worth double-checking either the "
                    "loss should be the 'sparse_' variant, or the labels should be one-hot encoded (e.g. via "
                    "to_categorical()/OneHotEncoder)."))

        # PyTorch: one_hot(...) feeding CrossEntropyLoss/NLLLoss, both of
        # which want raw integer class indices, not a one-hot vector.
        if one_hot_torch_loc is not None and ce_or_nll_loss_loc is not None:
            findings.append((label, one_hot_torch_loc,
                f"a one_hot(...) call here, combined with CrossEntropyLoss/NLLLoss at line "
                f"{ce_or_nll_loss_loc} -- both of these losses expect raw integer class indices as the "
                "target, not a one-hot vector, so feeding them one-hot-encoded labels is usually a shape "
                "mismatch (or, when it doesn't error, computes the wrong loss). Worth double-checking the "
                "raw class-index tensor is what's actually passed in, not its one_hot(...) encoding."))

        # --------------------------------------------------------------
        # Pass 3: per-function checks (torch training-loop hygiene, TF
        # GradientTape hygiene, JAX grad/PRNG-key hygiene).
        # --------------------------------------------------------------
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # ---- PyTorch: backward()/zero_grad()/step() presence ----
            has_backward, has_zero_grad, has_step, backward_line = False, False, False, None
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    if inner.func.attr == "backward":
                        has_backward = True
                        backward_line = backward_line or inner.lineno
                    if inner.func.attr == "zero_grad":
                        has_zero_grad = True
                    if inner.func.attr == "step":
                        has_step = True
            if has_backward and not has_zero_grad:
                findings.append((label, backward_line,
                    f"function '{node.name}' calls .backward() but no .zero_grad() appears anywhere in "
                    "it -- worth double-checking gradients are being cleared each step somewhere else in "
                    "the call chain, since silently accumulating them across steps is a common, subtle bug."))
            if has_backward and has_zero_grad and not has_step:
                findings.append((label, backward_line,
                    f"function '{node.name}' calls .backward() and .zero_grad() but no .step() appears "
                    "anywhere in it -- worth double-checking an optimizer step is actually being taken "
                    "somewhere else in the call chain, since without it gradients are computed and "
                    "cleared but the weights never update (loss can look like it's training while the "
                    "model silently never learns anything)."))

            # ---- TF: GradientTape().gradient(...) never applied ----
            has_tape = any(
                isinstance(inner, ast.withitem) and isinstance(inner.context_expr, ast.Call)
                and _mllint_fname(inner.context_expr) == "GradientTape"
                for inner in ast.walk(node)
            )
            tape_gradient_line = None
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "gradient":
                    tape_gradient_line = tape_gradient_line or inner.lineno
            has_apply_gradients = any(
                isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "apply_gradients"
                for inner in ast.walk(node)
            )
            if has_tape and tape_gradient_line and not has_apply_gradients:
                findings.append((label, tape_gradient_line,
                    f"function '{node.name}' opens a tf.GradientTape() and calls .gradient(...) on it, but "
                    "no .apply_gradients(...) call appears anywhere in it -- worth double-checking the "
                    "optimizer is actually applying these gradients somewhere else in the call chain, since "
                    "without it the gradients are computed but the weights never update (loss can look like "
                    "it's training while the model silently never learns anything)."))

            # ---- JAX: jax.grad()/value_and_grad() result never used ----
            grad_var_names = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Call):
                    if _mllint_is_grad_result(inner.value):
                        grad_var_names.update(_mllint_assign_target_names(inner))
            if grad_var_names:
                used_names = {
                    inner.id for inner in ast.walk(node)
                    if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)
                }
                unused = grad_var_names - used_names
                if unused:
                    findings.append((label, node.lineno,
                        f"function '{node.name}' computes gradients via jax.grad/value_and_grad into "
                        f"{sorted(unused)}, but that value is never used anywhere else in the function (e.g. "
                        "passed to optax.apply_updates or a manual tree_map parameter update) and isn't "
                        "returned either -- worth double-checking the gradients are actually applied to the "
                        "parameters somewhere, since jax.grad alone computes a gradient but never updates "
                        "anything."))

            # ---- JAX: same PRNG key reused across multiple draws
            # without an intervening jax.random.split() ----
            key_uses = {}       # name -> [linenos]
            key_reassigned = set()
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr != "split" and inner.args
                    and "random" in ast.dump(inner.func)
                ):
                    first_arg = inner.args[0]
                    if isinstance(first_arg, ast.Name) and _MLLINT_JAX_KEY_NAME_RE.search(first_arg.id):
                        key_uses.setdefault(first_arg.id, []).append(inner.lineno)
                if isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Call) and _mllint_fname(inner.value) == "split":
                    key_reassigned.update(_mllint_assign_target_names(inner))
            for key_name, linenos in key_uses.items():
                if len(linenos) >= 2 and key_name not in key_reassigned:
                    findings.append((label, linenos[0],
                        f"function '{node.name}' passes the same PRNG key '{key_name}' to {len(linenos)} "
                        f"separate jax.random.* calls without ever calling jax.random.split({key_name}) to "
                        "derive a fresh subkey -- reusing one key for multiple draws produces correlated (in "
                        "some cases identical) 'random' outputs, a common, silent JAX bug. Worth "
                        "double-checking a fresh subkey is derived per use."))

            # ---- JAX: @jit-decorated function calling print() ----
            is_jitted = any(
                _mllint_fname(dec) == "jit" if isinstance(dec, ast.Call) else (
                    (isinstance(dec, ast.Name) and dec.id == "jit")
                    or (isinstance(dec, ast.Attribute) and dec.attr == "jit")
                )
                for dec in node.decorator_list
            )
            if is_jitted:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "print":
                        findings.append((label, inner.lineno,
                            f"function '{node.name}' is decorated with @jit but calls print(...) inside it "
                            "-- under jax.jit tracing, the function body runs (and prints) once to build the "
                            "trace, not on every actual call, which is a common source of \"my print "
                            "statement only fired once / isn't showing up\" confusion. Worth double-checking "
                            "jax.debug.print(...) is used instead if per-call output is actually wanted."))
                        break

        # --------------------------------------------------------------
        # Pass 4: per-file, non-function-scoped checks.
        # --------------------------------------------------------------

        # ---- GENERAL: PRNG reseeded on every iteration of a loop ----
        for call_node in _mllint_find_calls_in_loops(tree, _MLLINT_SEED_FUNC_NAMES):
            findings.append((label, call_node.lineno,
                f"a {_mllint_fname(call_node)}(...) call sits inside a loop here -- reseeding a PRNG on "
                "every iteration forces every epoch/step to draw the exact same sequence of 'random' "
                "numbers (identical shuffles, identical dropout masks, identical augmentation/noise), "
                "silently defeating randomness for the rest of the run. Worth double-checking the seed is "
                "meant to be set once, before the loop, not inside it."))

        # ---- GENERAL: shuffle=False on something that looks like the
        # training set (DataLoader, tf.data.Dataset, etc.) ----
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            shuffle_kw = next((kw for kw in node.value.keywords if kw.arg == "shuffle"), None)
            if not (shuffle_kw and isinstance(shuffle_kw.value, ast.Constant) and shuffle_kw.value.value is False):
                continue
            for tgt_name in _mllint_assign_target_names(node):
                if _MLLINT_TRAIN_NAME_RE.search(tgt_name):
                    call_fname = _mllint_fname(node.value) or "call"
                    findings.append((label, node.lineno,
                        f"{tgt_name} = {call_fname}(..., shuffle=False): building a training data "
                        f"loader/dataset named '{tgt_name}' with shuffling explicitly turned off -- "
                        "training on data in a fixed order (especially if it's sorted or grouped by label) "
                        "can bias gradient updates within each epoch and hurt convergence. Worth "
                        "double-checking this wasn't meant to be shuffle=True."))

        # ---- GENERAL: loss accumulated into a running total without
        # detaching it from the autodiff graph first (torch/tf tensors) ----
        for node in ast.walk(tree):
            if not (isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add)):
                continue
            target = node.target
            if not (isinstance(target, ast.Name) and _MLLINT_LOSSY_ACCUM_TARGET_RE.search(target.id)):
                continue
            calls_in_rhs = {
                (inner.func.attr if isinstance(inner.func, ast.Attribute) else getattr(inner.func, "id", None))
                for inner in ast.walk(node.value) if isinstance(inner, ast.Call)
            }
            if calls_in_rhs & {"item", "detach", "cpu", "numpy", "float", "block_until_ready"}:
                continue
            names_in_rhs = {inner.id for inner in ast.walk(node.value) if isinstance(inner, ast.Name)}
            if any(_MLLINT_LOSS_NAME_RE.search(n) for n in names_in_rhs):
                findings.append((label, node.lineno,
                    f"{target.id} += ...: accumulating a loss-like value into '{target.id}' without calling "
                    ".item()/.detach()/.cpu()/float() on it first -- if the right-hand side is still a "
                    "tensor attached to the autodiff graph (common in torch/tf), this keeps every step's "
                    "entire computation graph alive for the rest of the loop, which typically shows up as "
                    "steadily growing memory use (and eventually an OOM) rather than a wrong result. Worth "
                    "double-checking a plain Python number is what's actually being accumulated here."))

        # ---- PyTorch: backward() called inside a `with ...no_grad():`
        # block, where no gradients were tracked in the first place ----
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            is_no_grad = any(
                isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "no_grad"
                for item in node.items
            )
            if not is_no_grad:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "backward":
                    findings.append((label, inner.lineno,
                        "a .backward() call appears inside a `with torch.no_grad():` block -- tensors "
                        "created (or operated on) inside no_grad() don't track gradients, so calling "
                        ".backward() here either raises an error immediately or, if it doesn't, isn't "
                        "backpropagating through anything meaningful. Worth double-checking this call isn't "
                        "meant to sit outside the no_grad block."))

        # ---- PyTorch: a model (an nn.Module subclass instance) that's
        # never moved to a device, in a file that otherwise moves things
        # to a device -- a common source of "works until it doesn't" ----
        module_class_names = {
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef)
            and any((b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", None)) == "Module" for b in n.bases)
        }
        if module_class_names:
            instance_vars = {}  # var name -> assign lineno
            for n in ast.walk(tree):
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                    ctor = n.value.func.id if isinstance(n.value.func, ast.Name) else None
                    if ctor in module_class_names:
                        for tgt_name in _mllint_assign_target_names(n):
                            instance_vars.setdefault(tgt_name, n.lineno)
            if instance_vars:
                moved, any_device_call = set(), False
                for n in ast.walk(tree):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("to", "cuda"):
                        any_device_call = True
                        base = n.func.value
                        if isinstance(base, ast.Name) and base.id in instance_vars:
                            moved.add(base.id)
                unmoved = set(instance_vars) - moved
                if unmoved and any_device_call:
                    for var_name in sorted(unmoved):
                        findings.append((label, instance_vars[var_name],
                            f"'{var_name}' (an instance of an nn.Module subclass) never has .to(...)/.cuda() "
                            "called on it, but .to(...)/.cuda() is called elsewhere in this file on "
                            "something else -- worth double-checking the model itself is being moved onto "
                            "the same device as its data; a device mismatch usually raises immediately, but "
                            "if the mismatched path only runs during eval/inference this can be silent."))

        # ---- Evaluation-looking function without no_grad()/eval() ----
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            lname = node.name.lower()
            if not any(k in lname for k in ("eval", "valid", "test")):
                continue
            has_no_grad = any(
                isinstance(inner, ast.withitem) and isinstance(inner.context_expr, ast.Call)
                and isinstance(inner.context_expr.func, ast.Attribute) and inner.context_expr.func.attr == "no_grad"
                for inner in ast.walk(node)
            )
            has_eval_call = any(
                isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "eval"
                for inner in ast.walk(node)
            )
            calls_something = any(isinstance(inner, ast.Call) for inner in ast.walk(node))
            if calls_something and not has_no_grad and not has_eval_call:
                findings.append((label, node.lineno,
                    f"function '{node.name}' looks like an evaluation/validation function by name but has "
                    "no `with torch.no_grad():` block and no `.eval()` call anywhere in it -- worth "
                    "double-checking dropout/batchnorm are in eval mode and gradients aren't being tracked "
                    "unnecessarily during evaluation (this is a naming heuristic, so it may be a false "
                    "positive for a function that isn't actually doing torch evaluation)."))

    # --------------------------------------------------------------
    # Cross-file checks (a model class and its loss are often defined in
    # different files).
    # --------------------------------------------------------------
    if softmax_loc and crossentropy_loc:
        findings.append((softmax_loc[0], softmax_loc[1],
            f"a Softmax/LogSoftmax call here, combined with CrossEntropyLoss/cross_entropy at "
            f"{crossentropy_loc[0]}:{crossentropy_loc[1]} -- CrossEntropyLoss already applies "
            "log_softmax internally, so feeding it already-softmaxed logits (a 'double softmax') "
            "typically flattens gradients and hurts convergence. Worth double-checking the softmax "
            "layer is only used for inference-time probabilities, not fed into this loss."))
    if sigmoid_loc and bce_logits_loc:
        findings.append((sigmoid_loc[0], sigmoid_loc[1],
            f"a Sigmoid call here, combined with BCEWithLogitsLoss at "
            f"{bce_logits_loc[0]}:{bce_logits_loc[1]} -- BCEWithLogitsLoss already applies sigmoid "
            "internally for numerical stability, so a 'double sigmoid' here is the likely equivalent "
            "of the CrossEntropyLoss case above."))
    if softmax_module_loc and nllloss_loc:
        findings.append((softmax_module_loc[0], softmax_module_loc[1],
            f"a plain Softmax call here, combined with NLLLoss at "
            f"{nllloss_loc[0]}:{nllloss_loc[1]} -- NLLLoss expects log-probabilities as input, but a "
            "plain Softmax produces raw probabilities (not log-probabilities), so this silently trains "
            "with the wrong gradient scale instead of erroring. Worth double-checking this should be "
            "LogSoftmax instead, or the loss should be CrossEntropyLoss on raw logits."))

    return findings


_COMMENT_EXT_MAP = {
    ".js": "//", ".ts": "//", ".jsx": "//", ".tsx": "//", ".java": "//",
    ".c": "//", ".cpp": "//", ".h": "//", ".hpp": "//", ".cs": "//",
    ".go": "//", ".rs": "//", ".swift": "//", ".kt": "//",
}


def _comment_char_for(path):
    return _COMMENT_EXT_MAP.get(os.path.splitext(path)[1].lower(), "#")


def _normalize_ws_for_match(s):
    """Collapse each line's leading/trailing whitespace and drop blank
    lines, for a lenient equality check used only to LOCATE a snippet
    that doesn't match verbatim -- never used to decide what to write.
    A weaker model is far more likely to reproduce a snippet with a
    slightly different indent depth or an extra/missing blank line than
    to get the actual tokens wrong, so this recovers those cases instead
    of silently skipping a fix that's semantically exact."""
    lines = [ln.strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _find_fuzzy_snippet_span(content, old):
    """If `old` doesn't appear verbatim in `content` but a whitespace-
    normalized version of it appears exactly once as a contiguous run of
    lines, return (start, end) character offsets of that exact run in
    `content` (so the ORIGINAL text spanning those offsets -- not `old`
    itself -- can be replaced, preserving whatever indentation/blank
    lines the file actually has). Returns None if there's no unambiguous
    match. This is a fallback path only; an exact match is always tried
    first and preferred."""
    target = _normalize_ws_for_match(old)
    if not target:
        return None

    content_lines = content.splitlines(keepends=True)
    # Precompute normalized, non-blank (line_index, stripped_text) pairs
    # so we can slide a window over just the "meaningful" lines while
    # still tracking real offsets for the eventual span.
    meaningful = [(i, ln.strip()) for i, ln in enumerate(content_lines) if ln.strip()]
    target_lines = target.split("\n")
    n = len(target_lines)
    if n == 0 or len(meaningful) < n:
        return None

    matches = []
    for start in range(len(meaningful) - n + 1):
        window = [meaningful[start + k][1] for k in range(n)]
        if window == target_lines:
            first_line_idx = meaningful[start][0]
            last_line_idx = meaningful[start + n - 1][0]
            matches.append((first_line_idx, last_line_idx))

    if len(matches) != 1:
        return None

    first_line_idx, last_line_idx = matches[0]
    start_offset = sum(len(l) for l in content_lines[:first_line_idx])
    end_offset = sum(len(l) for l in content_lines[:last_line_idx + 1])
    return start_offset, end_offset


def _banner_wrap_fix(old, new, path):
    """Wrap a code-fix replacement so the OLD code stays visible, commented
    out, directly above the NEW (live) code -- instead of silently
    swapping one for the other with no trace in the file itself. Written
    directly into the file content that gets saved to disk, so it shows
    up the next time the file is opened, not just in Pulse's own chat
    transcript.
    """
    c = _comment_char_for(path)
    old_commented = "\n".join(f"{c} {line}" if line.strip() else c for line in old.splitlines())
    return f"{c} =====Pulse Change====\n{c} Old\n{old_commented}\n{c} New\n{new}"


# ============================================================================
# Persistent, cross-process fix history + revert -- module-level (not a
# ChatPanel method) because the automatic-rollback-on-repeated-restart-
# failure case (see persistent_tracer's RESTART handler) needs to call
# this from the TRAINING process, not the Dashboard process that owns
# ChatPanel; they only share a filesystem, so this is plain file I/O
# using the same on-disk format pulse_cli.py's .pulse_history uses, kept
# independent (no cross-import) since either tool may run standalone.
# ============================================================================
def _fix_history_dir(script_path):
    base = os.path.dirname(script_path) if script_path else "."
    d = os.path.join(base, ".pulse_history")
    os.makedirs(os.path.join(d, "diffs"), exist_ok=True)
    return d


def _fix_history_path(script_path):
    return os.path.join(_fix_history_dir(script_path), "changelog.jsonl")


def _load_fix_history(script_path):
    path = _fix_history_path(script_path)
    entries = []
    if not os.path.exists(path):
        return entries
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return entries


def _record_fix_history_commit(script_path, files, explanation, kind="fix"):
    """files: {path: (before, after)}. Returns the new commit's short id,
    or None if there was nothing to record."""
    if not files:
        return None
    commit_id = hashlib_sha1_short(f"{time.time()}|{explanation}|{sorted(files)}")
    file_entries = []
    diff_parts = []
    for fpath, (before, after) in files.items():
        label = os.path.basename(fpath)
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{label}", tofile=f"b/{label}",
        ))
        diff_parts.append(diff or f"(no textual change to {label})\n")
        file_entries.append({"path": fpath, "before": before, "after": after, "diff": diff})
    entry = {"id": commit_id, "kind": kind, "t": time.time(), "explanation": explanation, "files": file_entries}
    try:
        with open(_fix_history_path(script_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        return None
    try:
        with open(os.path.join(_fix_history_dir(script_path), "diffs", f"{commit_id}.diff"), "w", encoding="utf-8") as f:
            f.write("\n".join(diff_parts))
    except OSError:
        pass
    return commit_id


def hashlib_sha1_short(s):
    import hashlib
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:8]


def _fix_history_state_asof(entries, target_idx):
    state = {}
    for i, e in enumerate(entries):
        if i > target_idx:
            break
        for fc in e.get("files", []):
            state[fc["path"]] = fc["after"]
    return state


def _fix_history_original(entries, path):
    for e in entries:
        for fc in e.get("files", []):
            if fc["path"] == path:
                return fc["before"]
    return None


def _perform_fix_history_revert(script_path, entries, target_idx, target_label, on_code_change=None, extra_files=None):
    """Restore every file the changelog has ever touched to its state as
    of `target_idx` (-1 == "before the very first commit"). Pure file
    I/O -- safe to call from either the Dashboard process (ChatPanel's
    ROLLBACK: directive) or the training process (automatic rollback
    after repeated restart failures). `on_code_change`/`extra_files` are
    optional in-memory updates for a caller that has a live copy of the
    script/extra-file text to keep in sync (ChatPanel does; the training
    process's persistent_tracer doesn't need to, since it's about to
    restart anyway). Returns (restored_paths, failed_paths, new_commit_id).
    """
    all_paths = sorted({fc["path"] for e in entries for fc in e.get("files", [])})
    asof = _fix_history_state_asof(entries, target_idx)
    restored, failed = [], []
    revert_files = {}
    for fpath in all_paths:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                current = f.read()
        except OSError:
            current = None
        content = asof.get(fpath)
        if content is None:
            content = _fix_history_original(entries, fpath)
        if content is None or content == current:
            continue
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            failed.append((fpath, str(exc)))
            continue
        if current is not None:
            revert_files[fpath] = (current, content)
        restored.append(fpath)
        if fpath == script_path and on_code_change:
            on_code_change(content)
        elif extra_files is not None and fpath in extra_files:
            extra_files[fpath] = content
    new_commit = None
    if restored:
        new_commit = _record_fix_history_commit(script_path, revert_files, f"Reverted to state {target_label}", kind="revert")
    return restored, failed, new_commit


def _auto_rollback_after_failed_restarts_gui(script_path, chain_start_commit_id):
    """GUI-side counterpart of pulse_cli.py's
    PulseCLI._auto_rollback_after_failed_restarts -- called from
    persistent_tracer's RESTART handler once MAX_RESTART_ATTEMPTS is
    exhausted, in the training process itself. Prints its own status
    (there's no ChatPanel to hand a return string to here -- this
    process's stdout is the only channel), and the training process is
    continuing afterward either way, so a rolled-back script only takes
    effect the NEXT time it's actually restarted (manually, or by a
    future successful auto-fix).
    """
    if not chain_start_commit_id:
        print("[PULSE] ⚠ Could not automatically roll back (no recorded pre-fix commit for this chain) -- inspect .pulse_history/changelog.jsonl manually.")
        return False
    entries = _load_fix_history(script_path)
    chain_start_idx = next((i for i, e in enumerate(entries) if e.get("id") == chain_start_commit_id), None)
    if chain_start_idx is None:
        print(f"[PULSE] ⚠ Could not find commit {chain_start_commit_id} in .pulse_history to roll back from.")
        return False
    target_idx = chain_start_idx - 1
    target_label = f"before commit {chain_start_commit_id} (auto-rollback after repeated restart failures)"
    restored, failed, new_commit = _perform_fix_history_revert(script_path, entries, target_idx, target_label)
    if restored:
        print(
            f"[PULSE] 🛟 Automatic rollback: the fix chain starting at commit {chain_start_commit_id} kept "
            f"crashing after every retry, so Pulse restored {len(restored)} file(s) to their state right "
            "before that fix, instead of leaving known-broken code on disk for an unattended run."
        )
        for p in restored:
            print(f"    - {os.path.basename(p)}")
        if new_commit:
            print(f"[PULSE] 📝 Logged as commit {new_commit} in .pulse_history/ -- inspect/undo it there if this wasn't wanted.")
    if failed:
        print(f"[PULSE] ⚠ Automatic rollback failed to restore {len(failed)} file(s):")
        for p, err in failed:
            print(f"    - {os.path.basename(p)}: {err}")
    return bool(restored)


class ChatPanel(tk.Frame):
    def __init__(self, parent, get_manifest_fn, get_code_fn=None, script_path=None, on_code_change=None, promote_fn=None, get_extra_files_fn=None, initial_provider=None, restart_fn=None, exec_fn=None):
        super().__init__(parent, bg=PANEL)
        self.get_manifest_fn = get_manifest_fn
        self.get_code_fn = get_code_fn
        self.script_path = script_path
        self.on_code_change = on_code_change
        self.promote_fn = promote_fn  # (var_name) -> None, switches a lotrack var to full tracking
        self.get_extra_files_fn = get_extra_files_fn  # () -> {path: text} for other local project files
        self.restart_fn = restart_fn  # (provider_name) -> None, restarts the whole training process
        self.exec_fn = exec_fn  # (kind, arg) -> result str; runs an EXEC-style directive in the live training process
        # Lazily-initialized {path: text} snapshot of "the last checkpoint
        # that didn't crash" for CHANGELOG -- see _run_changelog.
        self._changelog_baseline = None
        # (rank, local_rank, world_size, session_key) -- set by Dashboard
        # right after construction when running under a multi-GPU/
        # multi-process launch; defaults to "not distributed".
        self.dist_info = (0, 0, 1, None)
        # Per-session id (fresh every time this ChatPanel is constructed,
        # i.e. every process start) used to identify "this run" in the
        # local run-history file -- see _run_runcompare.
        self._run_id = str(uuid.uuid4())[:8]
        # Running token/cost accounting across every agent call this
        # session -- see _record_usage / _run_cost.
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "cost_usd": 0.0}
        # Background retry queue for agent calls that fail only due to a
        # transient error (rate limit, timeout, provider blip) after
        # exhausting _call_model's own in-call retries -- see
        # _queue_agent_retry / _ensure_retry_ticker. Training itself is
        # never blocked on any of this.
        self._last_call_failed_transiently = False
        self._pending_agent_retry = None
        self._retry_ticker_started = False
        # id of the most recently recorded .pulse_history commit -- lets
        # Dashboard._restart_training know exactly what to roll back to
        # if the fix that triggered a restart keeps crashing (see
        # _auto_rollback_after_failed_restarts_gui).
        self._last_commit_id = None
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

    def _on_provider_change(self, selection):
        if PROVIDERS.get(selection, {}).get("custom"):
            model_string = simpledialog.askstring(
                "Custom Model", "Model string (e.g. openai/gpt-6-astra):", parent=self
            )
            if not model_string or "/" not in model_string:
                # Cancelled or invalid -- fall back to whatever provider was
                # active before this selection instead of leaving the
                # sentinel entry selected with nothing behind it.
                self.provider_var.set(list(PROVIDERS.keys())[0])
                self._update_status_indicator()
                return
            env_var = simpledialog.askstring(
                "Env Var", "Env var name for the API key (blank = none needed):", parent=self
            ) or None
            label = f"Custom: {model_string}"
            PROVIDERS[label] = {"model": model_string, "env_key": env_var}
            menu = self.provider_dropdown["menu"]
            menu.add_command(label=label, command=tk._setit(self.provider_var, label, self._on_provider_change))
            self.provider_var.set(label)
        elif PROVIDERS.get(selection, {}).get("openrouter"):
            model_string = simpledialog.askstring(
                "OpenRouter Model", "OpenRouter model (e.g. deepseek/deepseek-v4-flash):", parent=self
            )
            if not model_string or "/" not in model_string:
                self.provider_var.set(list(PROVIDERS.keys())[0])
                self._update_status_indicator()
                return
            label = register_openrouter_model(model_string)
            menu = self.provider_dropdown["menu"]
            menu.add_command(label=label, command=tk._setit(self.provider_var, label, self._on_provider_change))
            self.provider_var.set(label)
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
        env_var = prov_info.get("env_key")

        if not self._has_active_key(current_provider):
            api_key = simpledialog.askstring(
                prov_info.get("prompt_title", "API Key Required"),
                prov_info.get("prompt_msg", f"Enter the API key for {env_var}:"),
                parent=self, show='*'
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

        threading.Thread(target=self._ask_and_maybe_retry, args=(question, include_code, current_provider), daemon=True).start()

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

    _RETRYABLE_ERROR_ATTRS = (
        "RateLimitError", "Timeout", "APIConnectionError",
        "ServiceUnavailableError", "InternalServerError",
    )

    def _is_retryable_model_error(self, exc):
        for attr in self._RETRYABLE_ERROR_ATTRS:
            cls = getattr(litellm, attr, None)
            if cls is not None and isinstance(exc, cls):
                return True
        return False

    def _classify_model_error(self, exc):
        """Turn whatever litellm/the provider's SDK raised into one short,
        stable, user-facing reason instead of a raw exception repr."""
        for attr, reason in (
            ("AuthenticationError", "authentication failed -- check the API key for this provider"),
            ("PermissionDeniedError", "the API key doesn't have permission for this model"),
            ("RateLimitError", "rate limited by the provider -- try again shortly"),
            ("ContextWindowExceededError", "the conversation/code context is too large for this model's context window"),
            ("BadRequestError", "the provider rejected the request (often an invalid/unavailable model name)"),
            ("NotFoundError", "model not found -- check the model name is correct and available on this provider/account"),
            ("Timeout", "the request timed out"),
            ("APIConnectionError", "couldn't connect to the provider (network issue, or a local server that isn't running)"),
            ("ServiceUnavailableError", "the provider's service is temporarily unavailable"),
            ("InternalServerError", "the provider's own service hit an internal error"),
        ):
            cls = getattr(litellm, attr, None)
            if cls is not None and isinstance(exc, cls):
                return reason
        msg = str(exc)
        return msg[:200] + ("…" if len(msg) > 200 else "")

    def _call_model(self, model_name, instruction, image_payloads=None, max_tokens=_AGENT_MAX_TOKENS):
        """One lightweight completion call: system prompt + recent history +
        a one-off stage instruction (+ images on the first call only, so
        they aren't re-uploaded on every stage). Does not touch
        self.history -- the caller decides what gets persisted once the
        whole pipeline finishes.

        Transient failures (rate limits, timeouts, connection blips, a
        provider's own 500) are retried up to 3 times with a short
        backoff before giving up; anything else (bad API key, invalid
        model name, context too long) fails immediately since retrying
        won't change the outcome. On exhaustion, raises AgentRequestFailed
        rather than returning an error string dressed up as a real answer
        -- see _ask's try/except, the one chokepoint for the whole
        multi-pass pipeline.
        """
        content = [{"type": "text", "text": instruction}]
        if image_payloads:
            content.extend(image_payloads)
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.history[-10:]
            + [{"role": "user", "content": content}]
        )
        max_tokens = _clamp_output_tokens(model_name, max_tokens)
        max_attempts = 3
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=_AGENT_TIMEOUT_SECONDS,
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    raise AgentRequestFailed("the provider returned an empty response")
                self._record_usage(response)
                return text
            except AgentRequestFailed:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts and self._is_retryable_model_error(exc):
                    backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    reason = self._classify_model_error(exc)
                    self.after(0, lambda a=attempt, m=max_attempts, b=backoff, r=reason: self._append(
                        "Pulse", f"⚠ Agent request hit a transient error (attempt {a}/{m}), retrying in {b}s: {r}"))
                    time.sleep(backoff)
                    continue
                raise AgentRequestFailed(self._classify_model_error(exc)) from exc
        raise AgentRequestFailed(self._classify_model_error(last_exc) if last_exc else "unknown error")

    def _record_usage(self, response):
        """Best-effort token/cost accounting for the COST directive --
        never let a usage-accounting hiccup affect the actual chat
        response. Cost is litellm's own estimate where it has pricing
        data for the model; otherwise only token counts are available."""
        try:
            usage = getattr(response, "usage", None)
            if usage:
                self._token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                self._token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                self._token_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
            self._token_usage["calls"] += 1
            try:
                self._token_usage["cost_usd"] += float(litellm.completion_cost(completion_response=response) or 0.0)
            except Exception:
                pass
        except Exception:
            pass

    _GREP_MAX_MATCHES = 40
    _GREP_CONTEXT_LINES = 1
    _VIEW_MAX_LINES = 400
    _VIEW_ARG_RE = re.compile(r"^(?:([^:]+):)?\s*(\d+)\s*-\s*(\d+)\s*$")

    def _iter_searchable_files(self):
        """(label, path, text) for the entry script plus every other local
        project file Pulse knows about, regardless of whether 'Send Code'
        is currently checked -- GREP/VIEW let the agent pull in only the
        slice it actually needs instead of requiring the whole file(s) to
        already be in context.
        """
        extra_files = (self.get_extra_files_fn() or {}) if self.get_extra_files_fn else {}
        self._build_file_labels(extra_files)
        code = self.get_code_fn() if self.get_code_fn else None
        if self.script_path and code is not None:
            label = self._label_for_path.get(self.script_path, os.path.basename(self.script_path))
            yield label, self.script_path, code
        for path, text in extra_files.items():
            label = self._label_for_path.get(path, os.path.basename(path))
            yield label, path, text

    def _run_grep(self, pattern):
        """Case-insensitive regex (or literal, if not valid regex) search
        across every known project file, returning matches as
        'label, line N' with a line of context on each side, capped at
        _GREP_MAX_MATCHES total."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)

        blocks = []
        total = 0
        for label, _path, text in self._iter_searchable_files():
            if total >= self._GREP_MAX_MATCHES:
                break
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if total >= self._GREP_MAX_MATCHES:
                    break
                if not rx.search(line):
                    continue
                lo = max(0, i - self._GREP_CONTEXT_LINES)
                hi = min(len(lines), i + self._GREP_CONTEXT_LINES + 1)
                snippet = "\n".join(f"    {n + 1:>4} | {lines[n]}" for n in range(lo, hi))
                blocks.append(f"  {label}, line {i + 1}:\n{snippet}")
                total += 1

        if not blocks:
            return f"GREP '{pattern}': no matches in any tracked file."
        truncated = " (truncated -- narrow the pattern for more)" if total >= self._GREP_MAX_MATCHES else ""
        return f"GREP '{pattern}': {total} match(es){truncated}\n" + "\n".join(blocks)

    def _run_view(self, arg):
        """Exact line range from a known project file -- '<file>:<start>-
        <end>' (file label optional, defaults to the main script), e.g.
        'model.py:40-75' or '120-160'. Capped at _VIEW_MAX_LINES."""
        m = self._VIEW_ARG_RE.match(arg.strip())
        if not m:
            return f"VIEW '{arg}': could not parse -- expected '<file>:<start>-<end>' or '<start>-<end>'."
        file_label, start_str, end_str = m.groups()
        start, end = int(start_str), int(end_str)
        if end < start:
            start, end = end, start

        extra_files = (self.get_extra_files_fn() or {}) if self.get_extra_files_fn else {}
        # Build/refresh the label<->path map before resolving -- VIEW can
        # be sent on its own, without a preceding GREP or 'Send Code' this
        # turn, so _path_for_label may otherwise still be empty or stale.
        self._build_file_labels(extra_files)
        path = self._resolve_fix_path(file_label) if file_label else self.script_path
        if not path:
            return f"VIEW '{arg}': could not resolve file '{file_label}'."

        code = self.get_code_fn() if self.get_code_fn else None
        text = code if path == self.script_path else extra_files.get(path)
        if text is None:
            return f"VIEW '{arg}': no source text available for '{file_label or os.path.basename(path)}'."

        lines = text.splitlines()
        end = min(end, len(lines))
        start = max(1, start)
        if end - start + 1 > self._VIEW_MAX_LINES:
            end = start + self._VIEW_MAX_LINES - 1
        if start > len(lines):
            return f"VIEW '{arg}': file only has {len(lines)} lines."

        label = self._label_for_path.get(path, os.path.basename(path))
        numbered = "\n".join(f"    {n:>4} | {lines[n - 1]}" for n in range(start, end + 1))
        return f"VIEW {label}:{start}-{end}\n{numbered}"

    # ------------------------------------------------------------------
    # Code intelligence (AST-based, pure/local -- no live-process access
    # needed): DEFOF/CALLERS/DEPGRAPH.
    # ------------------------------------------------------------------
    def _iter_ast_trees(self):
        """(label, path, text, tree) for every known project file that
        parses as valid Python -- a file that doesn't parse (mid-edit
        syntax error, or a non-Python file that slipped into extra_files)
        is skipped rather than raising."""
        for label, path, text in self._iter_searchable_files():
            try:
                tree = ast.parse(text, filename=path)
            except (SyntaxError, ValueError):
                continue
            yield label, path, text, tree

    def _run_defof(self, symbol):
        """DEFOF: <symbol> -- AST-based jump-to-definition across every
        tracked file, not text search."""
        symbol = symbol.strip()
        hits = []
        for label, _path, text, tree in self._iter_ast_trees():
            lines = text.splitlines()
            for node in ast.walk(tree):
                kind, name = None, None
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                    name = node.name
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == symbol:
                            name, kind = symbol, "assignment"
                            break
                if name is None:
                    continue
                start = node.lineno
                end = min(start + 4, len(lines))
                snippet = "\n".join(f"    {n:>4} | {lines[n - 1]}" for n in range(start, end + 1) if n <= len(lines))
                hits.append(f"  {label}:{start} ({kind} {symbol})\n{snippet}")
        if not hits:
            return f"DEFOF '{symbol}': no definition found in any tracked file."
        return f"DEFOF '{symbol}': {len(hits)} definition(s)\n" + "\n\n".join(hits)

    def _run_callers(self, symbol):
        """CALLERS: <symbol> -- every call site of a function/class, so a
        model doesn't have to guess whether a signature change breaks
        something elsewhere."""
        symbol = symbol.strip()
        hits = []
        for label, _path, text, tree in self._iter_ast_trees():
            lines = text.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called_name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
                if called_name != symbol:
                    continue
                lineno = node.lineno
                line = lines[lineno - 1].strip() if lineno <= len(lines) else ""
                hits.append(f"  {label}, line {lineno}: {line}")
        if not hits:
            return f"CALLERS '{symbol}': no call sites found in any tracked file."
        capped = hits[:60]
        suffix = f" (truncated, showing first {len(capped)})" if len(hits) > len(capped) else ""
        return f"CALLERS '{symbol}': {len(hits)} call site(s){suffix}\n" + "\n".join(capped)

    def _run_depgraph(self):
        """DEPGRAPH: -- the actual import graph between tracked project
        files, so the model knows which file owns a piece of logic
        instead of guessing from filenames."""
        trees = list(self._iter_ast_trees())
        label_by_modname = {os.path.splitext(os.path.basename(path))[0]: label for label, path, _t, _tr in trees}
        edges = []
        for label, _path, _text, tree in trees:
            deps = set()
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module.split(".")[0]]
                for m in mods:
                    other = label_by_modname.get(m)
                    if other and other != label:
                        deps.add(other)
            if deps:
                edges.append(f"  {label} -> {', '.join(sorted(deps))}")
        if not edges:
            return "DEPGRAPH: no import relationships found between tracked local files."
        return "DEPGRAPH (import graph between tracked local files):\n" + "\n".join(edges)

    # ------------------------------------------------------------------
    # Statistical tooling beyond CALC: pattern-detection across a
    # variable's full recorded history, exactly the kind of thing code is
    # better at than eyeballing a chart. Scalars (loss/accuracy/lr/custom
    # logged values) get a persisted (step, value) history in the
    # manifest -- see the "history" field written in _worker_main.
    # ------------------------------------------------------------------
    def _scalar_history(self, name):
        """Resolve `name` against the manifest (exact match, else a
        unique case-insensitive substring match, same convention as
        PROMOTE) and return (resolved_name, [(step, value), ...] or None).
        """
        manifest = self.get_manifest_fn() or {}
        stats = manifest.get(name)
        if stats is None:
            matches = [k for k in manifest if name.lower() in k.lower()]
            if len(matches) == 1:
                name, stats = matches[0], manifest[matches[0]]
        if stats is None:
            return name, None
        hist = stats.get("history")
        if hist is None:
            recent = stats.get("recent")
            hist = list(enumerate(recent)) if recent else None
        return name, hist

    def _run_corr(self, arg):
        """CORR: <var1> <var2> -- real correlation coefficient between two
        tracked scalar histories, instead of eyeballing whether two charts
        move together."""
        parts = arg.split()
        if len(parts) != 2:
            return f"CORR '{arg}': expected two variable names, e.g. 'CORR: loss grad_norm'."
        name1, hist1 = self._scalar_history(parts[0])
        name2, hist2 = self._scalar_history(parts[1])
        if not hist1 or not hist2:
            missing = parts[0] if not hist1 else parts[1]
            return f"CORR '{arg}': no numeric history for '{missing}' -- only scalar-tracked variables (loss, accuracy, lr, ...) have one."
        v1 = [v for _, v in hist1 if v is not None]
        v2 = [v for _, v in hist2 if v is not None]
        n = min(len(v1), len(v2))
        if n < 3:
            return f"CORR '{name1}' vs '{name2}': not enough overlapping data points yet ({n})."
        v1, v2 = v1[-n:], v2[-n:]
        mean1, mean2 = sum(v1) / n, sum(v2) / n
        cov = sum((a - mean1) * (b - mean2) for a, b in zip(v1, v2))
        var1 = sum((a - mean1) ** 2 for a in v1)
        var2 = sum((b - mean2) ** 2 for b in v2)
        if var1 == 0 or var2 == 0:
            return f"CORR '{name1}' vs '{name2}': one series is constant over these {n} points -- correlation undefined."
        r = cov / math.sqrt(var1 * var2)
        return f"CORR '{name1}' vs '{name2}' (last {n} points): r = {r:.4f}"

    def _run_outlier(self, var):
        """OUTLIER: <var> -- deterministic z-score anomaly detection over a
        variable's history, flags exact points instead of 'eyeball the
        chart.'"""
        name, hist = self._scalar_history(var)
        if not hist:
            return f"OUTLIER '{var}': no numeric history available -- only scalar-tracked variables have one."
        pts = [(i, v) for i, v in hist if v is not None]
        if len(pts) < 4:
            return f"OUTLIER '{name}': not enough data points yet ({len(pts)})."
        values = [v for _, v in pts]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        if std == 0:
            return f"OUTLIER '{name}': series is constant -- no outliers."
        flagged = [(i, v, (v - mean) / std) for i, v in pts if abs((v - mean) / std) > 3]
        if not flagged:
            return f"OUTLIER '{name}': no points with |z| > 3 over {len(pts)} points (mean={mean:.4g}, std={std:.4g})."
        lines = "\n".join(f"  history idx {i}: {v:.6g} (z={z:.2f})" for i, v, z in flagged[-20:])
        return f"OUTLIER '{name}': {len(flagged)} outlier point(s) (mean={mean:.4g}, std={std:.4g})\n{lines}"

    _DIFFSTATS_ARG_RE = re.compile(r"^\s*(\S+)\s+(-?\d+)\s+(-?\d+)\s*$")

    def _run_diffstats(self, arg):
        """DIFFSTATS: <var> <index_a> <index_b> -- exact delta in a
        variable's recorded history between two points (indices count
        into that variable's own deduplicated history, negative counts
        from the end -- like a Python list index), instead of mentally
        subtracting two numbers off two different messages."""
        m = self._DIFFSTATS_ARG_RE.match(arg.strip())
        if not m:
            return f"DIFFSTATS '{arg}': expected '<var> <index_a> <index_b>', e.g. 'DIFFSTATS: loss 10 40'."
        var, a_str, b_str = m.groups()
        name, hist = self._scalar_history(var)
        if not hist:
            return f"DIFFSTATS '{var}': no numeric history available."
        values = [v for _, v in hist]
        try:
            va, vb = values[int(a_str)], values[int(b_str)]
        except IndexError:
            return f"DIFFSTATS '{name}': index out of range (history has {len(values)} point(s))."
        if va is None or vb is None:
            return f"DIFFSTATS '{name}': one of those points is None/unreadable."
        delta = vb - va
        pct = f" ({delta / va * 100:+.2f}%)" if va else ""
        return f"DIFFSTATS '{name}' [{a_str}] -> [{b_str}]: {va:.6g} -> {vb:.6g}, delta = {delta:+.6g}{pct}"

    def _run_histogram(self, var, buckets=10):
        """HISTOGRAM: <var> -- actual bucketed distribution counts over a
        variable's history, not a description of what a heatmap image
        'looks like it shows.'"""
        name, hist = self._scalar_history(var)
        if not hist:
            return f"HISTOGRAM '{var}': no numeric history available."
        values = [v for _, v in hist if v is not None]
        if len(values) < 2:
            return f"HISTOGRAM '{name}': not enough data points yet ({len(values)})."
        lo, hi = min(values), max(values)
        if lo == hi:
            return f"HISTOGRAM '{name}': all {len(values)} point(s) equal {lo:.6g}."
        width = (hi - lo) / buckets
        counts = [0] * buckets
        for v in values:
            counts[min(buckets - 1, int((v - lo) / width))] += 1
        peak = max(counts) or 1
        lines = []
        for i, c in enumerate(counts):
            b_lo, b_hi = lo + i * width, lo + (i + 1) * width
            bar = "#" * max(1, int(40 * c / peak)) if c else ""
            lines.append(f"  [{b_lo:>10.4g}, {b_hi:>10.4g}): {c:>5}  {bar}")
        return f"HISTOGRAM '{name}' ({len(values)} points, range [{lo:.4g}, {hi:.4g}]):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Grounding tooling: replacing recall with lookup.
    # ------------------------------------------------------------------
    def _run_doclookup(self, arg):
        """DOCLOOKUP: <library>.<symbol> -- fetch the actual current
        signature/docstring for an already-installed library function,
        instead of trusting memorized API shape that may be stale for the
        installed version."""
        arg = arg.strip()
        parts = arg.split(".")
        if len(parts) < 2:
            return f"DOCLOOKUP '{arg}': expected '<library>.<symbol>', e.g. 'DOCLOOKUP: torch.nn.functional.cross_entropy'."
        root = parts[0]
        try:
            obj = importlib.import_module(root)
        except Exception as exc:
            return f"DOCLOOKUP '{arg}': could not import '{root}' ({type(exc).__name__}: {exc}) -- it may not be installed in this environment."
        resolved = [root]
        for attr in parts[1:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                try:
                    obj = importlib.import_module(".".join(resolved + [attr]))
                except Exception:
                    return f"DOCLOOKUP '{arg}': '{'.'.join(resolved)}' has no attribute '{attr}'."
            resolved.append(attr)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = ""
        doc = inspect.getdoc(obj) or "(no docstring)"
        if len(doc) > 1200:
            doc = doc[:1200] + "\n... (truncated)"
        return f"DOCLOOKUP '{arg}': {'.'.join(resolved)}{sig}\n{doc}"

    def _run_changelog(self):
        """CHANGELOG: -- diff of exactly what's changed in tracked files
        since the last checkpoint, so attention narrows to what's actually
        different rather than re-reading a whole file. The baseline is
        the code as of this chat session's start (i.e. the last state
        training was actually running under) and is advanced to "now"
        every time CHANGELOG runs, so a second call only shows what's
        changed since the first."""
        extra_files = (self.get_extra_files_fn() or {}) if self.get_extra_files_fn else {}
        self._build_file_labels(extra_files)
        code = self.get_code_fn() if self.get_code_fn else None
        current = dict(extra_files)
        if self.script_path and code is not None:
            current[self.script_path] = code

        if self._changelog_baseline is None:
            self._changelog_baseline = current
            return "CHANGELOG: no prior checkpoint yet -- this turn's code is now the baseline for future CHANGELOG calls."

        diffs = []
        for path, new_text in current.items():
            old_text = self._changelog_baseline.get(path, "")
            if old_text == new_text:
                continue
            label = self._label_for_path.get(path, os.path.basename(path))
            diff = "\n".join(difflib.unified_diff(
                old_text.splitlines(), new_text.splitlines(),
                fromfile=f"{label} (last checkpoint)", tofile=f"{label} (now)", lineterm="", n=2,
            ))
            if diff:
                diffs.append(diff)

        self._changelog_baseline = current
        if not diffs:
            return "CHANGELOG: no changes since the last checkpoint."
        body = "\n\n".join(diffs)
        if len(body) > 4000:
            body = body[:4000] + "\n... (truncated)"
        return f"CHANGELOG (since last checkpoint):\n{body}"

    def _fixlog_path(self):
        base = os.path.dirname(self.script_path) if self.script_path else "."
        return os.path.join(base, ".pulse_fixlog.json")

    def _load_fixlog(self):
        try:
            with open(self._fixlog_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def _append_fixlog(self, path, old, new, explanation):
        """Persist every applied code fix (old/new/explanation) to a
        project-level fix-log so PASTFIX can search it later -- this
        project's own history of prior fixes, so established local
        patterns get reused instead of rediscovered from scratch."""
        log = self._load_fixlog()
        log.append({"time": time.time(), "path": path, "old": old, "new": new, "explanation": explanation})
        log = log[-200:]
        try:
            with open(self._fixlog_path(), "w", encoding="utf-8") as f:
                json.dump(log, f)
        except OSError:
            pass

    def _run_pastfix(self, query):
        """PASTFIX: <symbol_or_region> -- search this project's own
        persistent fix-log for prior fixes touching the same
        function/region."""
        query_l = query.strip().lower()
        hits = []
        for entry in reversed(self._load_fixlog()):
            haystack = " ".join([
                str(entry.get("explanation", "")), str(entry.get("old", "")),
                str(entry.get("new", "")), str(entry.get("path", "")),
            ]).lower()
            if query_l in haystack:
                hits.append(entry)
            if len(hits) >= 5:
                break
        if not hits:
            return f"PASTFIX '{query}': no prior fix in this project's fix-log mentions that."
        lines = []
        for e in hits:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("time", 0)))
            lines.append(f"  [{ts}] {os.path.basename(str(e.get('path', '?')))}: {e.get('explanation', '(no explanation)')}")
        return f"PASTFIX '{query}': {len(hits)} prior fix(es)\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Automatic safety/verification gate -- NOT model-invoked, runs on
    # every proposed fix regardless of what directives the model used.
    # ------------------------------------------------------------------
    def _lint_check(self, content, path):
        """Syntax/AST validation, then a real static-analysis pass
        (pyflakes) if it's importable, on the FULL proposed file content
        before it's ever written to disk -- catches undefined names, bad
        syntax, and unused imports deterministically instead of trusting
        the agent's own read of its diff. Returns (ok, messages)."""
        if os.path.splitext(path)[1] != ".py":
            return True, []
        try:
            compile(content, path, "exec")
        except SyntaxError as exc:
            return False, [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

        try:
            import pyflakes.api as _pyflakes_api
            import pyflakes.reporter as _pyflakes_reporter
        except ImportError:
            return True, []

        out, err = io.StringIO(), io.StringIO()
        _pyflakes_api.check(content, path, _pyflakes_reporter.Reporter(out, err))
        messages = [l for l in (out.getvalue() + err.getvalue()).splitlines() if l.strip()]
        # Only undefined-name-class errors block the write -- style-only
        # warnings (unused imports/locals) are reported but don't block,
        # since those are common in intentionally-scaffolded fixes.
        blocking = [m for m in messages if "undefined name" in m.lower() or "syntaxerror" in m.lower()]
        return (len(blocking) == 0), messages

    def _run_gpustatus(self):
        """GPUSTATUS: -- real per-device GPU memory/utilization instead of
        the model guessing about 'the GPU'. Under a multi-GPU/multi-rank
        launch (see _distributed_info), aggregates every rank's status
        (published periodically to a shared status directory) into one
        report instead of just this process's own device(s)."""
        rank, _local_rank, world_size, session_key = self.dist_info
        if world_size > 1 and session_key:
            return _format_multi_rank_gpu_status(session_key, rank)
        return _format_gpu_status(_gpu_status_snapshot())

    def _run_rollback(self, arg):
        """ROLLBACK: <commit_id, or 'last'> -- deterministically revert
        via the same .pulse_history mechanism the automatic
        restart-failure rollback uses, instead of the agent trying to
        manually reconstruct old code from memory of the conversation."""
        entries = _load_fix_history(self.script_path)
        if not entries:
            return "ROLLBACK: no recorded Pulse code changes to revert."
        arg = arg.strip()
        if not arg or arg.lower() == "last":
            target_idx = len(entries) - 2
            target_label = f"before commit {entries[-1].get('id', '?')} (most recent)"
        else:
            target_idx = None
            target_label = ""
            for i, e in enumerate(entries):
                eid = e.get("id", "")
                if eid == arg or eid.startswith(arg):
                    target_idx = i
                    target_label = f"commit {eid}"
                    break
            if target_idx is None:
                return f"ROLLBACK '{arg}': no commit matches that id -- send CHANGELOG or check .pulse_history/changelog.jsonl."
        extra_files = (self.get_extra_files_fn() or {}) if self.get_extra_files_fn else {}
        restored, failed, new_commit = _perform_fix_history_revert(
            self.script_path, entries, target_idx, target_label,
            on_code_change=self.on_code_change, extra_files=extra_files,
        )
        if new_commit:
            self._last_commit_id = new_commit
        if not restored and not failed:
            return f"ROLLBACK '{arg}': workspace already matches that state -- nothing to revert."
        parts = []
        if restored:
            parts.append(
                f"ROLLBACK: reverted to state {target_label} -- {len(restored)} file(s) restored: "
                + ", ".join(os.path.basename(p) for p in restored)
                + (f" (logged as commit {new_commit})" if new_commit else "")
                + " -- restart training to pick it up."
            )
        if failed:
            parts.append("ROLLBACK: failed to restore " + ", ".join(f"{os.path.basename(p)} ({err})" for p, err in failed))
        return "\n".join(parts)

    def _run_mllint(self):
        """MLLINT: -- a small set of high-confidence, AST-detectable ML
        anti-patterns (metric/loss mismatches, double-softmax/sigmoid,
        missing zero_grad, suspiciously high LR, eval functions missing
        no_grad/.eval()) across every tracked file. These are heuristics
        about ML *semantics*, worded as things worth double-checking, not
        certainties the way the syntax LINT gate's findings are."""
        findings = _mllint_scan(self._iter_ast_trees())
        if not findings:
            return "MLLINT: no anti-patterns found in this heuristic check across the tracked files (this doesn't guarantee correctness, just that none of Pulse's known patterns matched)."
        lines = [f"  {label}:{lineno}: {msg}" for label, lineno, msg in findings]
        return f"MLLINT: {len(findings)} finding(s) worth double-checking\n" + "\n\n".join(lines)

    def _run_rankdiverge(self, var):
        """RANKDIVERGE: <var> -- compares this rank's latest value of a
        tracked scalar against every other rank's latest value (published
        via the same rank-status files GPUSTATUS reads). In correctly
        synced distributed training, per-rank loss/metric values should
        track closely after gradient sync; a rank whose value has drifted
        noticeably from the rest is a strong signal of something rank-
        specific gone wrong (unsynced BatchNorm stats, a corrupted data
        shard, a rank stuck on stale weights, ...). Only meaningful under
        a multi-rank launch."""
        rank, _local_rank, world_size, session_key = self.dist_info
        if world_size <= 1 or not session_key:
            return "RANKDIVERGE: this process isn't part of a multi-rank launch (world_size == 1)."
        statuses = _read_all_rank_status(session_key)
        if not statuses:
            return "RANKDIVERGE: no rank status files found yet."
        values = {}
        for s in statuses:
            scalars = s.get("scalars") or {}
            if var in scalars and scalars[var] is not None:
                values[s.get("rank")] = scalars[var]
        if len(values) < 2:
            return f"RANKDIVERGE '{var}': fewer than 2 ranks currently report this variable ({len(values)} found) -- nothing to compare yet."
        mean = sum(values.values()) / len(values)
        lines = [f"  rank {r}{' (this process)' if r == rank else ''}: {v:.6g}  (delta from mean: {v - mean:+.4g})" for r, v in sorted(values.items())]
        spread = max(values.values()) - min(values.values())
        verdict = "large spread across ranks -- worth investigating" if spread > abs(mean) * 0.1 + 1e-9 else "spread looks tight"
        return f"RANKDIVERGE '{var}' across {len(values)}/{world_size} rank(s), mean={mean:.6g}, spread={spread:.4g} ({verdict}):\n" + "\n".join(lines)

    def _run_runcompare(self):
        """RUNCOMPARE: -- compares this run's current scalar values
        against the most recent PREVIOUS run recorded for this project (a
        "run" is one process lifetime), so a regression against last
        time's numbers is visible without anyone having to remember them.
        Local, per-project file -- see PulseCLI's version for the
        additional workspace/team-shared comparison when logged in."""
        manifest = self.get_manifest_fn() or {}
        current_scalars = {
            name: stats.get("latest_value") for name, stats in manifest.items()
            if isinstance(stats, dict) and stats.get("latest_value") is not None
        }
        if not current_scalars:
            return "RUNCOMPARE: no scalar values recorded yet this run."

        history = _load_run_history(self.script_path)
        previous = next((r for r in reversed(history) if r.get("run_id") != self._run_id), None)

        this_entry = next((r for r in history if r.get("run_id") == self._run_id), None)
        if this_entry is None:
            this_entry = {"run_id": self._run_id, "started": time.time()}
            history.append(this_entry)
        this_entry["updated"] = time.time()
        this_entry["scalars"] = current_scalars
        history = history[-10:]
        try:
            with open(_run_history_path(self.script_path), "w", encoding="utf-8") as f:
                json.dump(history, f)
        except OSError:
            pass

        if previous is None:
            return "RUNCOMPARE: no prior run recorded for this project yet -- this run is now the baseline for future RUNCOMPARE calls."

        prev_scalars = previous.get("scalars", {})
        lines = []
        for name, cur_val in current_scalars.items():
            prev_val = prev_scalars.get(name)
            if prev_val is None or cur_val is None:
                continue
            delta = cur_val - prev_val
            pct = f" ({delta / prev_val * 100:+.2f}%)" if prev_val else ""
            lname = name.lower()
            got_worse = (("loss" in lname or "err" in lname) and delta > 0) or ("acc" in lname and delta < 0)
            flag = "  <- worse than last run" if got_worse else ""
            lines.append(f"  {name}: {prev_val:.6g} -> {cur_val:.6g}  (delta {delta:+.6g}{pct}){flag}")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(previous.get("updated", previous.get("started", 0))))
        if not lines:
            return f"RUNCOMPARE: found a previous run (last updated {when}) but no overlapping scalar names to compare."
        return f"RUNCOMPARE vs previous run (last updated {when}):\n" + "\n".join(lines)

    def _run_cost(self):
        """COST: -- running token/cost usage for this chat session's
        agent calls, accumulated across every pipeline pass (LOCATE/
        ANALYZE/DEVELOP/VERIFY/SWEEP), not just the current question."""
        u = self._token_usage
        if u["calls"] == 0:
            return "COST: no agent calls recorded yet this session."
        lines = [
            f"  agent calls: {u['calls']}",
            f"  prompt tokens: {u['prompt_tokens']:,}",
            f"  completion tokens: {u['completion_tokens']:,}",
            f"  total tokens: {u['total_tokens']:,}",
        ]
        if u["cost_usd"] > 0:
            lines.append(f"  estimated cost: ${u['cost_usd']:.4f}")
        else:
            lines.append("  estimated cost: unavailable (no pricing data for this model/provider)")
        return "COST (this session so far):\n" + "\n".join(lines)

    def _apply_new_directives(self, requests):
        """Deterministically service the extended toolset (see
        _extract_new_directives) the same way _apply_directives handles
        CALC/PROMOTE/GREP/VIEW -- pure computation/file-reading/subprocess
        round-trip only, safe to run on the background request thread."""
        notes = []
        for symbol in requests.get("defof", []):
            notes.append(self._run_defof(symbol))
        for symbol in requests.get("callers", []):
            notes.append(self._run_callers(symbol))
        if "depgraph" in requests:
            notes.append(self._run_depgraph())
        for arg in requests.get("corr", []):
            notes.append(self._run_corr(arg))
        for var in requests.get("outlier", []):
            notes.append(self._run_outlier(var))
        for arg in requests.get("diffstats", []):
            notes.append(self._run_diffstats(arg))
        for var in requests.get("histogram", []):
            notes.append(self._run_histogram(var))
        for arg in requests.get("doclookup", []):
            notes.append(self._run_doclookup(arg))
        if "changelog" in requests:
            notes.append(self._run_changelog())
        if "gpustatus" in requests:
            notes.append(self._run_gpustatus())
        if "mllint" in requests:
            notes.append(self._run_mllint())
        for var in requests.get("rankdiverge", []):
            notes.append(self._run_rankdiverge(var))
        if "runcompare" in requests:
            notes.append(self._run_runcompare())
        if "cost" in requests:
            notes.append(self._run_cost())
        for arg in requests.get("rollback", []):
            notes.append(self._run_rollback(arg))
        for query in requests.get("pastfix", []):
            notes.append(self._run_pastfix(query))

        exec_kinds = (("dryrun", "DRYRUN"), ("repl", "REPL"), ("replay", "REPLAY"),
                      ("gradcheck", "GRADCHECK"), ("shapetrace", "SHAPETRACE"),
                      ("layerstats", "LAYERSTATS"), ("hardexamples", "HARDEXAMPLES"),
                      ("ampstatus", "AMPSTATUS"), ("seedcheck", "SEEDCHECK"))
        for req_key, kind in exec_kinds:
            if req_key not in requests:
                continue
            if not self.exec_fn:
                notes.append(f"{kind}: execution tooling isn't available in this session (no live training process bridge).")
                continue
            for arg in requests[req_key]:
                notes.append(self.exec_fn(kind, arg))

        return "\n\n".join(n for n in notes if n)

    def _apply_directives(self, calc_exprs, promote_names, grep_patterns=None, view_requests=None):
        """Deterministically compute any CALC: expressions, figure out
        which PROMOTE: names match a currently-known variable, and run any
        GREP:/VIEW: requests. Pure computation/file-reading only -- no Tk
        calls -- so this is safe to run on the background request thread.
        Returns (note_text, calc_lines, promoted_names) for the caller to
        display/apply on the main thread; GREP/VIEW results are folded
        into note_text (and so into the agent's own history) rather than
        surfaced as separate return values, since nothing on the main
        thread needs to act on them the way it does for promotions.
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
        if grep_patterns:
            for pattern in grep_patterns:
                notes.append(self._run_grep(pattern))
        if view_requests:
            for arg in view_requests:
                notes.append(self._run_view(arg))

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
        _parse_code_fix, but without requiring any particular fields.

        Models are told to respond with ONLY the JSON object, but often
        don't -- a stray "Sure, here you go:" before it, a trailing
        sentence after it, or a code fence that isn't stripped cleanly is
        enough to make a naive "whole string must be exactly `{...}`"
        check fail, which is why verification used to report "unparsable"
        on almost every call. Instead, find the first balanced {...} span
        anywhere in the text (respecting strings, so braces inside a
        quoted "reason" don't throw off the count) and parse just that.
        """
        text = (answer or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return None

        try:
            payload = json.loads(text[start:end + 1])
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

    def _verify_fix_with_retries(self, model_name, fix, diagnosis):
        """PASS 4: check the fix's math/logic before it's handed to the
        user. If it fails, ask the agent to revise and re-check, up to
        _MAX_VERIFY_ATTEMPTS times. Returns (fix, passed, reason). A failed
        verify/revise request applies the fix as unverified best effort
        instead of discarding it (see pulse_cli's copy)."""
        reason = ""
        for attempt in range(_MAX_VERIFY_ATTEMPTS):
            fix_desc = self._describe_fix(fix)
            self.after(0, lambda: self._set_stage("Checking the fix"))
            try:
                verify_answer = self._call_model(
                    model_name, _PASS4_VERIFY_TMPL.format(diagnosis=diagnosis, fix_desc=fix_desc),
                    max_tokens=_AGENT_MAX_TOKENS,
                )
            except AgentRequestFailed as exc:
                return fix, False, f"(verification request failed: {exc})"
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
            try:
                revised_answer = self._call_model(
                    model_name, _PASS4_REVISE_TMPL.format(diagnosis=diagnosis, fix_desc=fix_desc, reason=reason),
                    max_tokens=_AGENT_MAX_TOKENS,
                )
            except AgentRequestFailed:
                break
            revised = self._parse_code_fix(revised_answer)
            if revised is None:
                break
            fix = revised
        return fix, False, (reason or "(verification did not clearly pass after retries)")

    _PROBE_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

    def _build_fast_probe_source(self, model_name, path, content):
        """Ask the agent for a fast-but-faithful copy of `content` (same
        model/data/loss, just iteration-capped) -- see pulse_cli's copy
        for the full rationale. Returns None if nothing parseable came
        back, so the empirical check is skipped rather than run on
        garbage."""
        try:
            answer = self._call_model(
                model_name,
                _PASS4B_PROBE_TMPL.format(filename=os.path.basename(path), content=content),
                max_tokens=_AGENT_MAX_TOKENS,
            )
        except AgentRequestFailed:
            return None
        m = self._PROBE_CODEBLOCK_RE.search(answer or "")
        probe_src = (m.group(1) if m else (answer or "")).strip()
        if not probe_src:
            return None
        try:
            ast.parse(probe_src)
        except SyntaxError:
            return None
        return probe_src

    def _run_loss_probe(self, target_path, probe_src):
        """Write `probe_src` next to the real script, run it as a short,
        hard-capped subprocess, and return whatever loss readings it
        reported. Always cleans up every file it created, including
        anything the run itself wrote to disk. Same mechanics as
        PulseCLI._run_loss_probe -- kept as a separate copy here since the
        GUI has no _resolve_restart_interpreter of its own and instead
        just uses sys.executable."""
        directory = os.path.dirname(os.path.abspath(target_path)) or "."
        token = uuid.uuid4().hex[:10]
        probe_path = os.path.join(directory, f".__pulse_probe_{token}.py")
        harness_name = f"_pulse_probe_harness_{token}"
        harness_path = os.path.join(directory, f"{harness_name}.py")
        metrics_path = os.path.join(directory, f".__pulse_probe_{token}.json")

        try:
            before_entries = set(os.listdir(directory))
        except OSError:
            before_entries = set()

        result = {"loss": [], "note": "", "stdout_tail": "", "stderr_tail": ""}
        try:
            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(_PROBE_HARNESS_SRC)
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write(f"import {harness_name}  # noqa -- Pulse empirical-verify harness, deleted after this probe\n")
                f.write(probe_src)

            depth = 0
            try:
                depth = int(os.environ.get(_RESTART_DEPTH_ENV, "0"))
            except ValueError:
                depth = 0
            env = dict(
                os.environ,
                **{
                    _RESTART_CHILD_ENV: "1",
                    _RESTART_DEPTH_ENV: str(depth + 1),
                    "PULSE_AUTO_RESTART": "1",
                    "PULSE_PROBE_METRICS_PATH": metrics_path,
                    "PULSE_PROBE_TIME_BUDGET": str(_PROBE_SOFT_TIME_BUDGET_SECONDS),
                },
            )
            try:
                proc = subprocess.run(
                    [sys.executable, probe_path], cwd=directory, env=env,
                    capture_output=True, text=True, timeout=_PROBE_HARD_TIMEOUT_SECONDS,
                )
                result["stdout_tail"] = (proc.stdout or "")[-4000:]
                result["stderr_tail"] = (proc.stderr or "")[-4000:]
                if proc.returncode != 0:
                    result["note"] = f"probe process exited with code {proc.returncode}"
            except subprocess.TimeoutExpired as exc:
                result["note"] = f"probe run hit the hard {_PROBE_HARD_TIMEOUT_SECONDS:.0f}s cap and was stopped"
                result["stdout_tail"] = (exc.stdout or "")[-4000:]
                result["stderr_tail"] = (exc.stderr or "")[-4000:]
            except Exception as exc:
                result["note"] = f"probe run failed to launch: {exc}"

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and isinstance(data.get("loss"), list):
                        result["loss"] = [v for v in data["loss"] if isinstance(v, (int, float))]
                except Exception:
                    pass

            if not result["loss"]:
                result["loss"] = _scrape_stdout_losses(result["stdout_tail"])
        finally:
            for p in (probe_path, harness_path, metrics_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            try:
                after_entries = set(os.listdir(directory))
            except OSError:
                after_entries = set()
            for name in after_entries - before_entries:
                full = os.path.join(directory, name)
                try:
                    if os.path.isdir(full):
                        shutil.rmtree(full, ignore_errors=True)
                    else:
                        os.remove(full)
                except OSError:
                    pass
        return result

    def _verify_fix_empirically(self, model_name, fix, diagnosis, original_content):
        """PASS 4.5 -- MEASURE: same idea as PulseCLI._verify_fix_empirically
        -- actually run a fast, hard-capped slice of the real training loop
        and check the loss for real, instead of trusting Pass 4's
        self-report. On failure, reverts this attempt, asks the agent to
        revise using the measured curve, and retries (bounded). If every
        attempt fails/is inconclusive, the LAST attempted fix is left
        applied (same best-effort philosophy as Pass 4) but flagged.

        `original_content` is this fix's pre-edit content for script_path
        (from _write_code_fix's `originals` return), since the GUI doesn't
        keep a standing _pending_revert_backups the way the CLI does.

        Returns (fix, ok, detail): ok is True/False/None (None = skipped,
        nothing to measure or a probe couldn't be built/run at all).
        """
        target_path = self.script_path
        if not target_path or original_content is None:
            return fix, None, "fix didn't touch the main tracked script -- nothing to run standalone"

        evidence = ""
        for attempt in range(_MAX_EMPIRICAL_ATTEMPTS):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    current_content = f.read()
            except OSError as exc:
                return fix, None, f"couldn't re-read {os.path.basename(target_path)}: {exc}"

            probe_src = self._build_fast_probe_source(model_name, target_path, current_content)
            if probe_src is None:
                return fix, None, "couldn't build a runnable fast probe of the fix"

            self.after(0, lambda: self._set_stage(f"Running fast probe (hard cap {_PROBE_HARD_TIMEOUT_SECONDS:.0f}s)"))
            probe_result = self._run_loss_probe(target_path, probe_src)

            verdict, detail = _loss_probe_verdict(probe_result["loss"])
            note = probe_result.get("note") or ""
            evidence = detail + (f" [{note}]" if note else "")

            if verdict == "pass":
                return fix, True, evidence
            if verdict == "inconclusive":
                return fix, None, evidence

            if attempt == _MAX_EMPIRICAL_ATTEMPTS - 1:
                break
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
            except OSError:
                break
            fix_desc = self._describe_fix(fix)
            try:
                revised_answer = self._call_model(
                    model_name,
                    _PASS4B_REVISE_WITH_EVIDENCE_TMPL.format(
                        diagnosis=diagnosis, fix_desc=fix_desc, evidence=evidence,
                    ),
                    max_tokens=_AGENT_MAX_TOKENS,
                )
            except AgentRequestFailed:
                break
            revised = self._parse_code_fix(revised_answer)
            if revised is None:
                break
            _write_lines2, applied_by_path2, _orig2, _skipped2 = self._write_code_fix(revised)
            if not applied_by_path2:
                break
            fix = revised
        return fix, False, evidence or "loss did not improve after retries"

    def _run_sweep_and_maybe_recurse(self, model_name, include_code, provider_name, _depth):
        """PASS 5: re-read everything for OTHER, unrelated errors. If any
        turn up, ask the user whether to fix those too (PASS 6 recurses
        through the same 1-5 format for the new issue)."""
        self.after(0, lambda: self._set_stage("Reading for other errors"))
        sweep_answer = self._call_model(model_name, _PASS5_SWEEP, max_tokens=_AGENT_MAX_TOKENS)
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
        try:
            if _depth == 0:
                self._fix_applied_this_turn = False
                self._last_call_failed_transiently = False
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
            regions = self._call_model(model_name, f"{base_text}\n\n{_PASS1_LOCATE}", images, max_tokens=_AGENT_MAX_TOKENS)
            self.after(0, lambda: self._append("Pulse (1 · Region of error)", regions))

            # Pass 2: focused second read + diagnosis/reasoning.
            self.after(0, lambda: self._set_stage("Analyzing"))
            raw_analysis = self._call_model(model_name, _PASS2_ANALYZE_TMPL.format(regions=regions), max_tokens=_AGENT_MAX_TOKENS)
            analysis, calc_exprs, promote_names, grep_patterns, view_requests = _extract_directives(raw_analysis)
            analysis, new_requests = _extract_new_directives(analysis)
            self.after(0, lambda: self._append("Pulse (2 · Diagnosis & reasoning)", analysis))

            directive_note = ""
            if calc_exprs or promote_names or grep_patterns or view_requests:
                directive_note, calc_lines, promoted = self._apply_directives(
                    calc_exprs, promote_names, grep_patterns, view_requests
                )
                if calc_lines:
                    self.after(0, lambda: self._append("Pulse (verified calculations)", calc_lines))
                for target in promoted:
                    self.after(0, lambda t=target: self.promote_fn(t))
                if promoted:
                    promoted_str = ", ".join(promoted)
                    self.after(0, lambda s=promoted_str: self._append("Pulse", f"⚙ Promoted to full tracking (agent request): {s}"))
                if directive_note:
                    self.history.append({"role": "user", "content": [{"type": "text", "text": directive_note}]})

            if new_requests:
                new_note = self._apply_new_directives(new_requests)
                if new_note:
                    self.after(0, lambda n=new_note: self._append("Pulse (tool results)", n))
                    self.history.append({"role": "user", "content": [{"type": "text", "text": new_note}]})

            full_answer = f"{regions}\n\n{analysis}"

            if not wants_implementation:
                self.after(0, lambda: self._set_stage("Developing fix"))
                fix_text = self._call_model(
                    model_name, _PASS3_FIX_TEXT_TMPL.format(diagnosis=full_answer), max_tokens=_AGENT_MAX_TOKENS
                )
                self.after(0, lambda: self._set_stage(None))
                full_answer += f"\n\n{fix_text}"
                self.after(0, lambda: self._append("Pulse (3 · Fix)", fix_text))
                self.history.append({"role": "assistant", "content": full_answer})
                return

            # Pass 3: develop and implement the fix, with passes 1-2's
            # findings handed over (otherwise it re-derives them from scratch).
            self.after(0, lambda: self._set_stage("Developing & implementing fix"))
            fix_answer = self._call_model(
                model_name, _PASS3_IMPLEMENT_TMPL.format(diagnosis=full_answer), max_tokens=_AGENT_MAX_TOKENS
            )
            fix = self._parse_code_fix(fix_answer)
            if fix is None:
                self.after(0, lambda: self._set_stage(None))
                full_answer += f"\n\n{fix_answer}"
                self.after(0, lambda: self._append("Pulse (3 · Fix)", fix_answer))
                self.history.append({"role": "assistant", "content": full_answer})
                return

            # Pass 4: verify the fix's math/logic before handing it to the
            # user; revise and re-check on failure (bounded retries).
            fix, verify_ok, verify_reason = self._verify_fix_with_retries(model_name, fix, full_answer)
            status = "passed" if verify_ok else "did not clearly pass -- applying best effort"
            self.after(0, lambda: self._append("Pulse (4 · Verification)", f"{status}: {verify_reason}"))

            self.after(0, lambda: self._set_stage(None))
            self.history.append({"role": "assistant", "content": full_answer})

            write_lines, applied_by_path, _originals, skipped = self._write_code_fix(fix)

            # If any snippet failed to match (verbatim or fuzzy), give the
            # model ONE chance to re-quote the exact text before reporting
            # a miss -- a weaker model is much more likely to slightly
            # misquote a snippet than to have picked the wrong fix
            # entirely, and this is the cheapest place to recover that
            # without re-running the whole diagnosis.
            if skipped and self._path_for_label:
                retry_fix = self._request_corrected_snippets(model_name, fix, skipped)
                if retry_fix is not None:
                    retry_write_lines, retry_applied, _retry_orig, retry_skipped = self._write_code_fix(retry_fix)
                    if retry_applied:
                        # Merge: keep whatever landed on the first pass,
                        # add whatever landed on the retry (concatenating
                        # per-file change lists rather than overwriting,
                        # in case both passes touched the same file).
                        write_lines = write_lines + "\n\n(retry after correcting a snippet mismatch)\n" + retry_write_lines
                        for p, changes in retry_applied.items():
                            applied_by_path[p] = applied_by_path.get(p, []) + changes
                        skipped = retry_skipped

            self.after(0, lambda: self._append("Pulse", write_lines))
            if applied_by_path:
                self._fix_applied_this_turn = True

            # Pass 4.5: don't just trust Pass 4's self-report -- actually run
            # a fast, hard-capped slice of the real training loop and check
            # the loss for real. Only meaningful once a fix landed on disk.
            if applied_by_path:
                script_original = _originals.get(self.script_path) if self.script_path else None
                try:
                    fix, empirical_ok, empirical_detail = self._verify_fix_empirically(
                        model_name, fix, full_answer, script_original,
                    )
                except Exception as exc:
                    empirical_ok, empirical_detail = None, f"probe step raised an unexpected error: {exc}"
                if empirical_ok is True:
                    empirical_note = f"PASSED -- {empirical_detail}"
                elif empirical_ok is False:
                    empirical_note = f"DID NOT PASS -- {empirical_detail} -- fix left applied as best effort."
                else:
                    empirical_note = f"skipped -- {empirical_detail}"
                self.after(0, lambda: self._set_stage(None))
                self.after(0, lambda: self._append("Pulse (4.5 · Empirical check)", empirical_note))

            # Pass 5 (+ 6): only worth a full re-read if a fix actually landed.
            # The fix is already on disk, so a failed sweep request must not
            # stop the restart that tests it.
            if applied_by_path and _depth < 3:
                try:
                    self._run_sweep_and_maybe_recurse(model_name, include_code, provider_name, _depth)
                except AgentRequestFailed as exc:
                    msg = f"⚠ Full re-read skipped (agent request failed: {exc}) -- continuing with the applied fix."
                    self.after(0, lambda m=msg: self._append("Pulse", m))

            if _depth == 0 and self._fix_applied_this_turn and self.restart_fn:
                self.after(0, lambda: self._append("Pulse", "⚙ Restarting the training loop to pick up the fix..."))
                self.restart_fn(provider_name)
        except AgentRequestFailed as exc:
            # Whichever pass hit this, stop the pipeline right here rather
            # than let a raw failure get treated as a real diagnosis/fix
            # downstream. Queuing for a background retry (rather than just
            # printing and dropping it) happens in the thin wrapper that
            # spawns _ask -- see _ask_and_maybe_retry.
            msg = f"⚠ AI agent request failed: {exc}"
            self.after(0, lambda m=msg: self._append("Pulse", m))
            self.history.append({"role": "assistant", "content": msg})
            if _depth == 0 and self._fix_applied_this_turn and self.restart_fn:
                # A fix already landed before this later request failed --
                # still restart so it actually gets run and tested.
                self.after(0, lambda: self._append("Pulse", "⚙ Restarting the training loop to pick up the fix..."))
                self.restart_fn(provider_name)
            elif _depth == 0:
                self._last_call_failed_transiently = True

    def _ask_and_maybe_retry(self, question, include_code, provider_name, _depth=0):
        """Thin wrapper around _ask -- every _ask spawn site calls this
        instead, so a transient failure gets queued for a background
        retry (see _queue_agent_retry) rather than just printed and
        dropped. Recursive sweep-pass calls (_depth > 0) are invoked
        directly by _ask itself via _run_sweep_and_maybe_recurse, not
        through here, so only top-level asks are ever queued."""
        self._ask(question, include_code, provider_name, _depth)
        if _depth == 0:
            if self._last_call_failed_transiently:
                self._queue_agent_retry(question, include_code, provider_name)
            elif self._pending_agent_retry and self._pending_agent_retry.get("question") == question:
                self._pending_agent_retry = None

    def _queue_agent_retry(self, question, include_code, provider_name):
        """Called when a top-level ask failed only because of a transient
        agent-side error after exhausting _call_model's own in-call
        retries. Kept and retried again later on a backoff instead of
        lost for good, while training keeps going untouched in the
        meantime."""
        existing = self._pending_agent_retry
        backoff = 60.0
        if existing and existing.get("question") == question:
            backoff = min(existing.get("backoff", 60.0) * 2, 600.0)
        self._pending_agent_retry = {
            "question": question, "include_code": include_code, "provider_name": provider_name,
            "next_attempt": time.time() + backoff, "backoff": backoff,
        }
        msg = (f"⚠ Will retry this request again in ~{backoff:.0f}s (agent request hit a rate limit/"
               "transient error) -- training continues unaffected in the meantime.")
        self.after(0, lambda m=msg: self._append("Pulse", m))
        self._ensure_retry_ticker()

    def _ensure_retry_ticker(self):
        """Starts the one background thread that periodically retries
        whatever's in self._pending_agent_retry, if anything. Lazily
        started on first need, idempotent."""
        if self._retry_ticker_started:
            return
        self._retry_ticker_started = True

        def _loop():
            while True:
                time.sleep(15.0)
                pending = self._pending_agent_retry
                if pending and time.time() >= pending["next_attempt"]:
                    msg = "⚙ Retrying an earlier request that hit a rate limit/transient error..."
                    self.after(0, lambda m=msg: self._append("Pulse", m))
                    try:
                        self._ask_and_maybe_retry(pending["question"], pending["include_code"], pending["provider_name"])
                    except Exception:
                        pass

        threading.Thread(target=_loop, daemon=True).start()

    @staticmethod
    def _parse_code_fix(answer):
        """If `answer` is a well-formed code-fix JSON payload, return it, else None.

        Two shapes are accepted:
        - The intended one: "old"/"new" are each a list with one entry per
          change, where each entry is that change's full snippet (possibly
          multi-line via embedded "\\n"), zipped index-by-index with an
          optional "files" list of the same length.
        - One some models fall into anyway: "old"/"new" are each a flat
          list of individual source *lines* for a single whole-block
          change, rather than one entry per change -- since the fix
          usually adds or removes lines, "old" and "new" end up different
          lengths and can't be zipped pairwise. There's no "files" list in
          this shape either. Detected by the length mismatch and handled
          by joining each list into one snippet and treating it as a
          single change.
        """
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
        explanation = explanation if isinstance(explanation, str) else ""

        def _valid_str_list(lst):
            return isinstance(lst, list) and bool(lst) and all(isinstance(x, str) for x in lst)

        if not _valid_str_list(old) or not _valid_str_list(new):
            return None

        if len(old) == len(new):
            # Standard shape: one change per (old[i], new[i]) pair.
            if files is not None:
                if not isinstance(files, list) or len(files) != len(old):
                    return None
                if not all(f is None or isinstance(f, str) for f in files):
                    return None
            else:
                files = [None] * len(old)

            return {"old": old, "new": new, "files": files, "explanation": explanation}

        # Fallback shape: "old"/"new" are flat line arrays for a single
        # change -- join them back into one snippet each.
        joined_old = "\n".join(old)
        joined_new = "\n".join(new)
        if not joined_old.strip() or joined_old == joined_new:
            return None

        file_label = None
        if isinstance(files, str):
            file_label = files
        elif isinstance(files, list) and files and isinstance(files[0], str):
            file_label = files[0]

        return {
            "old": [joined_old], "new": [joined_new], "files": [file_label],
            "explanation": explanation,
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
        # Already a real path: _write_code_fix records resolved paths (not
        # labels) in `skipped`, which _request_corrected_snippets resolves
        # again -- without this the re-quote retry finds no file to show.
        if os.path.isfile(label):
            return os.path.abspath(label)
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

        Returns (display_text, applied_by_path, originals, skipped):
        applied_by_path maps touched file path -> [(old, new), ...] (empty
        if nothing landed); originals maps path -> its content before this
        edit, kept around in case a manual revert is ever needed; skipped
        is a list of (old, file_label_or_path, reason) for snippets that
        couldn't be matched/applied, so the caller can decide whether to
        ask the agent for a corrected snippet and retry.
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
            return "(Proposed a code fix with nothing to apply.)", {}, {}, []

        lines = []
        if fix["explanation"]:
            lines.append(f"**Explanation:** {fix['explanation']}")

        originals = {}
        after_by_path = {}
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
                    content = content.replace(old, _banner_wrap_fix(old, new, path), 1)
                    applied.append((old, new))
                    continue
                if count == 0:
                    # Fallback: the agent may have reproduced the right
                    # lines with slightly different indentation or an
                    # extra/missing blank line -- a weaker model does this
                    # far more often than it gets the actual tokens wrong.
                    # Only take this path when a whitespace-normalized
                    # match is unambiguous (exactly one span in the file);
                    # otherwise fall through to reporting it as unmatched
                    # rather than guessing.
                    span = _find_fuzzy_snippet_span(content, old)
                    if span is not None:
                        start, end = span
                        actual_old = content[start:end]
                        content = content[:start] + _banner_wrap_fix(actual_old, new, path) + content[end:]
                        applied.append((actual_old, new))
                    else:
                        skipped.append((old, path, "no exact match found in the file"))
                else:
                    skipped.append((old, path, f"matched {count} times (ambiguous), skipped for safety"))

            if not applied:
                continue

            # Automatic gate -- NOT model-invoked, runs regardless of what
            # directives the model used. Syntax/AST validation (always)
            # plus a real lint pass (pyflakes, if importable) on the FULL
            # proposed file content, before anything touches disk.
            lint_ok, lint_messages = self._lint_check(content, path)
            if not lint_ok:
                lines.append(
                    f"⚠ Fix for '{path}' failed the automatic syntax/lint gate and was NOT written:\n"
                    + "\n".join(lint_messages)
                )
                continue

            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as exc:
                lines.append(f"⚠ Failed to write changes to '{path}': {exc}")
                continue

            originals[path] = original_content
            applied_by_path[path] = applied
            after_by_path[path] = content
            for old, new in applied:
                self._append_fixlog(path, old, new, fix.get("explanation", ""))
            if path == self.script_path and self.on_code_change:
                self.on_code_change(content)

        for old, label in unresolved:
            skipped.append((old, label or "(unspecified file)", "couldn't determine which file this targets"))

        if not applied_by_path:
            lines.append("\n⚠ No changes were applied -- none of the proposed snippets matched cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: `{old.splitlines()[0][:80]}...`")
            return "\n".join(lines), {}, {}, skipped

        for path, applied in applied_by_path.items():
            lines.append(f"\n✓ Applied {len(applied)} change(s) to `{os.path.basename(path)}`:")
            for old, new in applied:
                safe_old = old.splitlines()[0][:80] if old.splitlines() else "(empty)"
                safe_new = new.splitlines()[0][:80] if new.splitlines() else "(empty)"
                lines.append(f"  - replaced `{safe_old}...` with `{safe_new}...`")
        if skipped:
            lines.append(f"\n⚠ Skipped {len(skipped)} proposed change(s) that didn't match cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: `{old.splitlines()[0][:80]}...`")

        commit_files = {p: (originals[p], after_by_path[p]) for p in applied_by_path}
        commit_id = _record_fix_history_commit(self.script_path, commit_files, fix.get("explanation") or "(no explanation given)")
        if commit_id:
            self._last_commit_id = commit_id
            lines.append(f"\n📝 Logged as commit {commit_id} in .pulse_history/ -- send `ROLLBACK: {commit_id}` (or ROLLBACK: last) to undo.")

        return "\n".join(lines), applied_by_path, originals, skipped

    def _request_corrected_snippets(self, model_name, fix, skipped):
        """One bounded retry for snippets that failed to match (verbatim
        or fuzzy) in _write_code_fix. Shows the model the actual current
        content of each affected file plus exactly which of its own "old"
        snippets didn't match and why, and asks ONLY for corrected old/new
        pairs for those -- not a full re-diagnosis. Returns a fix dict
        covering just the corrected entries, or None if nothing could be
        recovered (unparsable response, or no snippets left worth
        retrying)."""
        if not skipped:
            return None

        # Ambiguous-match skips ("matched N times") aren't a quoting
        # mistake -- retrying with the same or a re-guessed snippet is
        # just as likely to be ambiguous again, so only retry the
        # genuinely-not-found cases.
        retryable = [(old, label) for old, label, reason in skipped if "no exact match" in reason]
        if not retryable:
            return None

        file_blocks = []
        seen_paths = set()
        for old, label in retryable:
            path = self._resolve_fix_path(label)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    file_blocks.append(f"--- current content of {os.path.basename(path)} ---\n{f.read()}")
            except OSError:
                continue

        if not file_blocks:
            return None

        mismatch_lines = "\n".join(
            f"- targeting '{label or self.script_path}': your snippet did not match anywhere in that "
            f"file's current content:\n{old}"
            for old, label in retryable
        )
        prompt = (
            "CORRECTION: some of the exact snippets you proposed did not match the file's actual "
            "current content (below), so nothing was changed for them. This is almost always a "
            "quoting mistake (wrong indentation, a missing/extra blank line, or slightly different "
            "text) rather than a wrong diagnosis -- do NOT change what the fix does, just re-quote "
            "the 'old' text EXACTLY as it appears in the file content shown below.\n\n"
            + "\n\n".join(file_blocks)
            + f"\n\nSnippets that didn't match:\n{mismatch_lines}\n\n"
            "Respond with ONLY a corrected code-fix JSON object (old/new/files/explanation) covering "
            "just these snippets -- no prose, no markdown fences."
        )
        answer = self._call_model(model_name, prompt, max_tokens=_AGENT_MAX_TOKENS)
        return self._parse_code_fix(answer)

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
        threading.Thread(target=self._ask_and_maybe_retry, args=(question, True, current_provider), daemon=True).start()

    _TB_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')

    def _last_user_frame(self, tb_text):
        """Return (file, line) of the deepest traceback frame that belongs
        to the tracked script, so a crash report can lead with 'line N'
        instead of making the user hunt through the whole traceback for
        it. Falls back to the deepest frame overall if none match."""
        matches = self._TB_FRAME_RE.findall(tb_text)
        if not matches:
            return None
        if self.script_path:
            own = [m for m in matches if os.path.abspath(m[0]) == os.path.abspath(self.script_path)]
            if own:
                return own[-1]
        return matches[-1]

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
        frame_info = self._last_user_frame(tb_text)
        header = f"⚠ Error at {os.path.basename(frame_info[0])}, line {frame_info[1]}\n\n" if frame_info else ""
        self._append(
            "Pulse",
            f"{header}⚠ Your script crashed with an uncaught exception (the process has already exited). "
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
        threading.Thread(target=self._ask_and_maybe_retry, args=(question, True, current_provider), daemon=True).start()

    def report_restart_failure(self, failure):
        """Called by the Dashboard when a fix-triggered restart's
        replacement process crashed immediately (nonzero exit) instead of
        picking up training. Unlike the old behavior of blindly retrying
        the exact same broken code (or giving up and leaving the run on
        stale in-memory code), this hands the failure's stdout/stderr
        straight back to the agent so it can fix the CURRENT code rather
        than starting the diagnosis over from scratch. If a new fix lands,
        `_ask` will trigger another restart via `restart_fn` the same way
        as any other fix -- persistent_tracer's RESTART handler (which is
        still alive, since a failed attempt no longer tears the process
        down) picks that up and tries again.
        """
        returncode = failure.get("returncode")
        stdout = failure.get("stdout", "")
        stderr = failure.get("stderr", "")
        self._append(
            "Pulse",
            f"⚠ The restarted training process crashed immediately (exit code {returncode}) instead of "
            f"picking up the fix. Feeding this back to the agent to fix the current code:\n\n"
            f"```\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n```",
        )

        current_provider = self.provider_var.get()
        if not self._has_active_key(current_provider):
            self._append(
                "Pulse",
                "Pick a provider and enter an API key above, then ask me to fix this and I'll take a look.",
            )
            return

        question = (
            f"The fix you just applied caused the restarted training process to crash immediately with "
            f"exit code {returncode}. Its output:\n\nSTDOUT:\n{stdout or '(empty)'}\n\n"
            f"STDERR:\n{stderr or '(empty)'}\n\n"
            "This is the same bug context as before -- fix the CURRENT code (shown below) directly. Do "
            "not restart the diagnosis from scratch, and do not reintroduce whatever change just failed."
        )
        self._append("You (restart failed)", "(auto-reported by Pulse)")
        threading.Thread(target=self._ask_and_maybe_retry, args=(question, True, current_provider), daemon=True).start()


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

    def __init__(self, session_id, display_configs, code_text=None, config_queue=None, control_queue=None, known_names=None, script_path=None, var_states=None, extra_files=None, initial_provider=None, response_queue=None, dist_info=None):
        self.session_id = session_id
        self.cache = session_dir(session_id)
        self.manifest_path = os.path.join(self.cache, "manifest.json")
        self.crash_path = os.path.join(self.cache, "crash.json")
        self._crash_reported = False
        self.restart_failure_path = os.path.join(self.cache, "restart_failure.json")
        self.display_configs = display_configs
        self.config_queue = config_queue
        self.control_queue = control_queue
        self.response_queue = response_queue
        self._exec_req_counter = itertools.count()
        # (rank, local_rank, world_size, session_key) -- see
        # _distributed_info in auto_track(). world_size == 1 (the default)
        # means an ordinary single-process/single-GPU run.
        self.dist_info = dist_info or (0, 0, 1, None)
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
            exec_fn=self._exec_request,
        )
        self.chat_panel.dist_info = self.dist_info
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

        if os.path.exists(self.restart_failure_path):
            # Consumed (removed) immediately after reading rather than a
            # boolean "already reported" flag like crash.json -- a restart
            # can legitimately fail more than once per run (agent's second
            # fix attempt also crashes), and each occurrence should still
            # reach the agent.
            try:
                with open(self.restart_failure_path, "r", encoding="utf-8") as f:
                    failure = json.load(f)
                os.remove(self.restart_failure_path)
                self.chat_panel.report_restart_failure(failure)
            except Exception:
                pass

        manifest = self._load_manifest()
        self._manifest = manifest

        if self.auto_intervene.get() and not self.is_paused:
            problem = self._check_for_trouble(manifest)
            _pulse_log(
                f"GUI AUTO_GATE enabled={self.auto_intervene.get()} paused={self.is_paused} "
                f"manifest_count={len(manifest)} problem={problem!r}",
                console=True,
            )
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
        CLI's PulseCLI._check_for_trouble, adapted to what's actually
        available here -- a per-variable `stats` dict with `latest_value`
        and a short `recent` window, not PulseCLI's full unbounded
        per-step history -- so these checks are necessarily coarser than
        their CLI counterparts, but catch the same broad classes of
        problem:
        - non-finite scalar values (unambiguous trigger)
        - a loss-like scalar spiking to explosion_multiplier-x its own
          recent minimum
        - a loss/metric frozen bit-for-bit across the whole `recent`
          window (not just slow-moving -- something not actually running)
        - a grad/weight-norm scalar spiking (exploding) or collapsing
          toward zero (vanishing) relative to its own recent average
        - a learning-rate scalar jumping >=10x between its last two
          observations (scheduler misconfiguration)
        - a metric hitting (near-)perfect accuracy within its first
          couple of observations (classic data-leakage shape)
        Also flags any matrix/tensor (track or lotrack) whose latest
        stats show nan/inf.
        """
        histories = {}
        tensor_stats = {}
        for name, stats in manifest.items():
            if not isinstance(stats, dict) or "error" in stats:
                continue
            if stats.get("kind") != "scalar":
                nan = stats.get("nan", 0) or 0
                inf = stats.get("inf", 0) or 0
                if nan or inf:
                    tensor_stats[name] = {"nan": nan, "inf": inf}
                continue
            recent = list(stats.get("recent") or [])
            latest = stats.get("latest_value")
            # `recent` is mirrored into the manifest a beat before latest_value is
            # written, so the newest reading can be missing from it.
            if latest is not None and (not recent or recent[-1] != latest):
                recent.append(latest)
            if recent:
                histories[name] = recent

        engine = getattr(self, "_detector", None)
        if engine is None:
            engine = _pulse_detect.DetectionEngine(sensitivity=0.3)
            self._detector = engine
        engine.overrides = ({"explosion_multiplier": self.explosion_multiplier}
                            if getattr(self, "explosion_multiplier", None) else {})
        try:
            raised = engine.update(histories, tensor_stats=tensor_stats or None)["raised"]
        except Exception as exc:
            _pulse_log(f"GUI DETECTOR ERROR {type(exc).__name__}: {exc}")
            return None
        actionable = [f for f in raised
                      if f.severity in (_pulse_detect.CRITICAL, _pulse_detect.WARNING)]
        return "; ".join(f.message for f in actionable) if actionable else None

    def _trigger_auto_intervention(self, problem):
        """Pause the user's training loop and hand the problem to the agent.

        The Dashboard owns the Tk UI, while the training loop lives in this
        process. Therefore the pause has to be explicitly sent through
        control_queue. Previously this method only called
        report_training_trouble(), even though its docstring claimed the loop
        had already been paused. That made GUI auto-intervention advisory
        rather than an actual intervention.
        """
        self.is_paused = True

        # Tell the training process to stop at its next trace boundary BEFORE
        # starting the agent thread. That way the agent sees a genuinely
        # paused run, even if the model request takes a while.
        if self.control_queue is not None:
            try:
                self.control_queue.put(("PAUSE", None))
            except Exception:
                pass

        if not self.pause_banner.winfo_ismapped():
            self.pause_label.configure(text=f"Auto-paused: {problem}")
            self.pause_banner.pack(fill=tk.X, side=tk.TOP, before=self.canvas)

        # Auto-diagnostics must never block the Tk main thread.
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

    def _exec_request(self, kind, arg):
        """Bridge for the chat panel's execution directives (REPL/DRYRUN/
        SHAPETRACE/GRADCHECK/REPLAY) -- called from ChatPanel's background
        request thread (see _apply_new_directives), never from the Tk main
        thread, so blocking here is safe. Sends an EXEC message to the
        training process via control_queue (drained in persistent_tracer,
        which actually runs it against the live frame -- see
        _handle_exec_request) and blocks on response_queue for the
        matching request id, with a bounded timeout so a stuck training
        process can't hang the chat panel forever.
        """
        if self.control_queue is None or self.response_queue is None:
            return f"{kind}: execution tooling isn't available (no live training process bridge in this run)."
        req_id = next(self._exec_req_counter)
        try:
            self.control_queue.put(("EXEC", req_id, kind, arg))
        except Exception as exc:
            return f"{kind} '{arg}': couldn't reach the training process ({exc})."

        deadline = time.time() + 20.0
        pending = []
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            try:
                resp_id, result = self.response_queue.get(timeout=remaining)
            except Exception:
                break
            if resp_id == req_id:
                for leftover in pending:
                    try:
                        self.response_queue.put(leftover)
                    except Exception:
                        pass
                return result
            pending.append((resp_id, result))
        for leftover in pending:
            try:
                self.response_queue.put(leftover)
            except Exception:
                pass
        return f"{kind} '{arg}': timed out waiting for the training process to respond."

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


def _run_dashboard(session_id, display_configs, code_text=None, config_queue=None, control_queue=None, known_names=None, script_path=None, var_states=None, extra_files=None, initial_provider=None, response_queue=None, dist_info=None):
    Dashboard(session_id, display_configs, code_text=code_text, config_queue=config_queue, control_queue=control_queue, known_names=known_names, script_path=script_path, var_states=var_states, extra_files=extra_files, initial_provider=initial_provider, response_queue=response_queue, dist_info=dist_info).run()


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


def _stream_mode_requested(mode):
    return str(mode or "").strip().lower() == "stream" or \
        os.environ.get("PULSE_MODE", "").strip().lower() == "stream"


def _start_stream_monitor(caller_frame, throttle_interval):
    """Attach the light monitor and leave the thinking to a separate brain process.

    Nothing is installed into the training loop: no sys.settrace, no frame-local trace
    function, no callback on the user's code at all. A background thread reads the
    training thread's variables a few times a second and streams them out, and a brain
    process picks them up from there. Detection, the agent and any fix happen over
    there, where taking a minute to think costs the run nothing.
    """
    from . import pulse_monitor

    script_path = None
    try:
        script_path = os.path.abspath(caller_frame.f_code.co_filename)
    except (AttributeError, OSError):
        pass
    try:
        interval = float(throttle_interval)
    except (TypeError, ValueError):
        interval = 1.0
    interval = min(2.0, max(0.05, interval / 4.0))

    monitor = pulse_monitor.attach(script_path=script_path, interval=interval,
                                   thread_id=threading.get_ident())

    previous_excepthook = sys.excepthook

    def _stream_excepthook(exc_type, exc_value, exc_tb):
        # The brain may be mid-sleep when the run dies. Put the traceback in the stream
        # first, so the evidence is there whenever it next looks.
        try:
            monitor.event("crash", urgent=True,
                          exception=f"{getattr(exc_type, '__name__', exc_type)}: {exc_value}",
                          traceback="".join(traceback.format_exception(exc_type, exc_value, exc_tb))[-8000:])
            monitor.snapshot_state({"crashed": True})
            pulse_monitor.detach()
        except Exception:
            pass
        return previous_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _stream_excepthook

    print(f"[Pulse] Monitoring this run into {monitor.directory}")
    print(f"[Pulse] Watch it with:  python -m pulse.brain {monitor.directory} --model <model>")
    return monitor


def auto_track(train_fn=None, throttle_interval=1.0, code_text=None, project_root=None, mode="cli"):
    """
    Call this once before your training loop, optionally passing your
    training function for a one-off dry run that discovers shapes:
        auto_track(train_step)

    mode: "ui" (dashboard + chat), "cli" (headless, Colab/SSH-friendly),
    "stream" (monitor only: stream to a separate brain process, the lightest
    option and the one that never blocks training), or "auto" (default --
    detects a real display and falls back to cli).
    """
    global _debugger_bg, _session_id
    mp.freeze_support()

    if _stream_mode_requested(mode):
        return _start_stream_monitor(sys._getframe(1), throttle_interval)

    # Multi-GPU / multi-process launch detection (torchrun,
    # torch.distributed.launch, OpenMPI, Slurm) -- world_size > 1 means
    # this is one of several ranks, typically one per GPU. Only rank 0
    # gets the interactive dashboard/chat; every rank (including rank 0)
    # periodically publishes its own GPU status so rank 0's GPUSTATUS
    # directive can show the whole job, not just its own device(s).
    rank, local_rank, world_size, session_key = _distributed_info()
    is_distributed = world_size > 1
    is_primary_rank = rank == 0
    if is_distributed:
        _previously_stale_ranks = set()

        def _rank_status_ticker():
            while True:
                scalars = {}
                # Best-effort scalar enrichment: only available once
                # _session_id (GUI mode) is assigned and manifest.json
                # exists -- CLI mode enriches its own rank status
                # separately (see PulseCLI's periodic checkpoint), since
                # it has no manifest.json/worker process to read from here.
                if _session_id:
                    try:
                        with open(os.path.join(session_dir(_session_id), "manifest.json"), "r", encoding="utf-8") as f:
                            manifest_now = json.load(f)
                        scalars = {
                            name: stats.get("latest_value") for name, stats in manifest_now.items()
                            if isinstance(stats, dict) and "latest_value" in stats
                        }
                    except (OSError, json.JSONDecodeError):
                        pass
                _write_rank_status(session_key, rank, local_rank, world_size, extra={"scalars": scalars} if scalars else None)

                # Proactive hang/deadlock detection (rank 0 only, to avoid
                # every rank printing the same alert): a rank going stale
                # while others keep advancing is the signature of the most
                # common real-world DDP failure mode, a collective-op
                # deadlock -- surfaced here instead of waiting for the
                # person to notice a frozen terminal on their own.
                if is_primary_rank:
                    try:
                        statuses = _read_all_rank_status(session_key)
                        now = time.time()
                        stale_now = {s.get("rank") for s in statuses if now - s.get("updated", 0) > 15}
                        newly_stale = stale_now - _previously_stale_ranks
                        if newly_stale:
                            print(f"[PULSE] ⚠ rank(s) {sorted(newly_stale)} of {world_size} have stopped "
                                  f"reporting (no update in >15s) while other ranks are still active -- this "
                                  f"is the usual signature of a collective-op deadlock/hang in distributed "
                                  f"training. Send GPUSTATUS in chat for full per-rank detail.")
                        _previously_stale_ranks.clear()
                        _previously_stale_ranks.update(stale_now)
                    except Exception:
                        pass
                time.sleep(5.0)
        threading.Thread(target=_rank_status_ticker, daemon=True).start()

    active_mode = _determine_mode(mode)
    caller_frame = sys._getframe(1)
    _pulse_log(
        f"AUTO_TRACK start requested_mode={mode!r} active_mode={active_mode!r} "
        f"caller={caller_frame.f_code.co_filename}:{caller_frame.f_lineno} "
        f"caller_func={caller_frame.f_code.co_name!r} throttle={throttle_interval}",
        console=True,
    )
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
    _pulse_log(
        f"DISCOVERY static/runtime candidates count={len(discovered)} "
        f"names={sorted(discovered)!r} shapes={dict((k, v) for k, v in discovered.items() if v is not None)!r}",        console=True,
    )

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
        # CLI mode still starts: its crash hook and agent don't need any
        # tracked variables, and a script whose first lines already fail
        # (e.g. using a name that was never defined) is exactly the case
        # that needs them. The UI dashboard has nothing to show, so it stops.
        if active_mode != "cli":
            return

    if is_distributed and not is_primary_rank:
        # Multiple ranks racing for the same GUI window or the same
        # terminal's stdin (CLI mode) would be actively harmful, not just
        # redundant -- so only rank 0 gets the interactive experience.
        # This rank still: (a) keeps publishing its GPU status via the
        # ticker started above, so rank 0's GPUSTATUS shows it, and (b)
        # installs an excepthook so a crash here is recorded into its
        # status file instead of silently vanishing into a terminal no
        # one is watching.
        print(f"[PULSE] rank {rank}/{world_size} (local_rank {local_rank}): running headless -- "
              f"only rank 0 opens the Pulse dashboard/chat. This rank's GPU status is still "
              f"visible to rank 0 via the GPUSTATUS chat directive.")
        if startup_error:
            _write_rank_status(session_key, rank, local_rank, world_size, extra={"error": startup_error})

        def _rank_excepthook(exc_type, exc_value, exc_tb):
            formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            _write_rank_status(session_key, rank, local_rank, world_size, extra={"error": formatted})
            sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _rank_excepthook
        return

    if active_mode == "cli":
        _start_cli_tracker(
            caller_frame, root, throttle_interval, discovered, runtime_shapes,
            code_text, entry_path, extra_files=extra_files, startup_error=startup_error,
            dist_info=(rank, local_rank, world_size, session_key),
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
    auto_custom_model = os.environ.pop("PULSE_AUTO_CUSTOM_MODEL", None)
    auto_custom_env_key = os.environ.pop("PULSE_AUTO_CUSTOM_ENV_KEY", None)
    # A dynamically-registered "Custom: ..." provider only ever lived in
    # the PROVIDERS dict of the process that created it -- a fresh process
    # starts with just the static entries, so re-register it here from
    # what the pre-restart process carried over (see persistent_tracer's
    # RESTART handler).
    if auto_provider and auto_provider not in PROVIDERS and auto_custom_model:
        PROVIDERS[auto_provider] = {"model": auto_custom_model, "env_key": auto_custom_env_key or None}
    result = None
    if auto_provider and auto_provider in PROVIDERS:
        env_key = PROVIDERS[auto_provider].get("env_key")
        auto_key = os.environ.get(env_key, "").strip() if env_key else "local"
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
    env_key = PROVIDERS[initial_provider].get("env_key")
    if env_key:
        os.environ[env_key] = api_key

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
    # Response side of the EXEC bridge (REPL/DRYRUN/SHAPETRACE/GRADCHECK/
    # REPLAY) -- the training process (this one) puts results here after
    # draining an ("EXEC", req_id, kind, arg) message off control_queue;
    # see persistent_tracer below and Dashboard._exec_request.
    response_queue = mp.Queue()

    dash_process = mp.Process(
        target=_run_dashboard,
        args=(_session_id, display_configs, code_text, shared_config_queue, control_queue, sorted(discovered), entry_path, var_states, extra_files, initial_provider, response_queue, (rank, local_rank, world_size, session_key)),
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
    last_scan = {"t": 0.0}
    paused = {"value": False}

    # ------------------------------------------------------------------
    # Windowed line tracing -- same fix as the CLI tracer (see
    # _start_cli_tracker/cli_tracer for the full writeup of why this is
    # needed: sys.settrace('line') armed for the whole run disables
    # CPython's specializing interpreter and pays a callback on every
    # bytecode line, which starves async GPU kernel dispatch).
    #
    # caller_frame (the user's training loop) is deliberately NOT windowed:
    # its f_trace was set directly rather than opted into via a 'call'
    # event, so it keeps receiving line events every step regardless of the
    # window. That's what the Pause/Resume/Restart control-queue draining
    # below relies on to stay responsive -- and it's cheap to always trace,
    # since the loop's own statements are few compared to everything a
    # forward/backward pass calls into. What IS windowed is (a) whether a
    # *nested* call (attention/layernorm/MLP/etc. helpers -- the actual
    # expensive, many-line frames) opts into line tracing at all, and (b)
    # whether the locals-scan-and-log work runs on any given line event,
    # including caller_frame's own.
    _CAPTURE_SPAN = min(0.05, throttle_interval / 4 if throttle_interval > 0 else 0.05)
    window = {"open": True}  # start open so initial discovery isn't starved
    _replay_checkpoint_counter = itertools.count()

    def _ticker():
        while True:
            time.sleep(throttle_interval)
            window["open"] = True
            # REPLAY: <n_steps> needs something to roll back to -- piggyback
            # the (already-throttled) snapshot cadence to also stash a
            # state_dict checkpoint of anything in scope that has one.
            try:
                _replay_maybe_checkpoint(caller_frame, next(_replay_checkpoint_counter))
            except Exception:
                pass
            time.sleep(_CAPTURE_SPAN)
            window["open"] = False

    threading.Thread(target=_ticker, daemon=True).start()

    # Total restart-launch attempts across the whole "restart storm" for
    # this process, shared across every RESTART message persistent_tracer
    # handles (a fresh agent re-fix can trigger another RESTART message
    # without this process ever having exited) -- caps the total work at
    # MAX_RESTART_ATTEMPTS regardless of how many separate RESTART
    # messages that takes, instead of resetting the counter (and so the
    # cap) on every individual message.
    restart_state = {"attempts": 0}

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
                elif msg[0] == "EXEC":
                    # ("EXEC", req_id, kind, arg) from ChatPanel's execution
                    # directives (REPL/DRYRUN/SHAPETRACE/GRADCHECK/REPLAY)
                    # -- run it right here against the real live frame, and
                    # hand the result back over response_queue.
                    _req_id, _kind, _arg = msg[1], msg[2], msg[3]
                    try:
                        _result = _handle_exec_request(caller_frame, _kind, _arg)
                    except Exception:
                        _result = f"{_kind} '{_arg}': Pulse's own dispatcher raised\n{_format_exc_short(traceback.format_exc())}"
                    if response_queue is not None:
                        try:
                            response_queue.put((_req_id, _result))
                        except Exception:
                            pass
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
                    # again -- see the UI-mode setup above. A dynamically
                    # registered "Custom: ..." provider doesn't exist in a
                    # fresh process's (static) PROVIDERS dict, so its model
                    # string/env var are carried over too via
                    # PULSE_AUTO_CUSTOM_MODEL/PULSE_AUTO_CUSTOM_ENV_KEY --
                    # see auto_track()'s resume path.
                    provider_name = msg[1] if len(msg) > 1 else None
                    if provider_name:
                        os.environ["PULSE_AUTO_PROVIDER"] = provider_name
                        info = PROVIDERS.get(provider_name)
                        if info and "model" in info:
                            os.environ["PULSE_AUTO_CUSTOM_MODEL"] = info["model"]
                            if info.get("env_key"):
                                os.environ["PULSE_AUTO_CUSTOM_ENV_KEY"] = info["env_key"]

                    print("\n[PULSE] Fix applied -- restarting the training loop to pick it up...\n")
                    sys.stdout.flush()
                    script_path = getattr(sys.modules.get('__main__'), '__file__', None)
                    script_path = os.path.abspath(script_path)

                    sys.stdout.write("\033c\033[0m")  # Reset ANSI and clear screen
                    sys.stdout.flush()
                    if os.name == 'nt':
                        subprocess.run('cls', shell=True)
                    else:
                        subprocess.run('stty sane', shell=True)
                        subprocess.run('clear', shell=True)

                    # Retry the restart itself instead of falling back to
                    # "keep running the old, already-in-memory process" on
                    # a bad exit code. IMPORTANT: unlike the old version of
                    # this handler, the dashboard/tracer are NOT torn down
                    # here up front -- they're only torn down right before
                    # a SUCCESSFUL relaunch's sys.exit(0), below. This is
                    # what lets a failed restart hand the failure back to
                    # the still-alive agent (via restart_failure.json,
                    # picked up by Dashboard._poll -> report_restart_failure)
                    # to fix the CURRENT code, instead of the old behavior
                    # of blindly retrying the exact same broken code, or
                    # giving up with the dashboard already killed.
                    MAX_RESTART_ATTEMPTS = 5
                    RETRY_BACKOFF_SECONDS = 3

                    restart_argv = [sys.executable, script_path] + sys.argv[1:]
                    attempt = restart_state.get("attempts", 0)
                    restarted_ok = False
                    restart_result = None

                    while attempt < MAX_RESTART_ATTEMPTS:
                        attempt += 1
                        try:
                            restart_result = subprocess.run(
                                restart_argv,
                                capture_output=True,
                                text=True,
                            )
                        except Exception as exc:
                            print(f"[PULSE] Restart attempt {attempt}/{MAX_RESTART_ATTEMPTS} failed to launch ({exc}).")
                            if attempt < MAX_RESTART_ATTEMPTS:
                                time.sleep(min(RETRY_BACKOFF_SECONDS * attempt, 30))
                            continue

                        if restart_result.returncode == 0:
                            restarted_ok = True
                            break

                        # A nonzero exit almost always means the agent's fix
                        # didn't fully take, or introduced a new bug -- hand
                        # the failure straight back to the (still-running)
                        # agent instead of blindly relaunching the same
                        # broken code or giving up and reverting to scratch.
                        print(
                            f"\n[PULSE] Replacement training process exited with code "
                            f"{restart_result.returncode} (attempt {attempt}/{MAX_RESTART_ATTEMPTS})."
                        )
                        if restart_result.stderr:
                            print("[PULSE] STDERR from failed run:\n", restart_result.stderr)
                        if restart_result.stdout:
                            print("[PULSE] STDOUT from failed run:\n", restart_result.stdout)
                        break  # hand off to the agent below rather than looping blindly

                    restart_state["attempts"] = attempt

                    if restarted_ok:
                        # Only now -- once a replacement process has actually
                        # taken over successfully -- tear this one down.
                        try:
                            _debugger_bg.shutdown()
                        except Exception:
                            pass
                        try:
                            dash_process.terminate()
                        except Exception:
                            pass
                        sys.settrace(None)
                        sys.exit(0)

                    if attempt >= MAX_RESTART_ATTEMPTS:
                        print(f"[PULSE] ⚠ Giving up after {MAX_RESTART_ATTEMPTS} restart attempts -- continuing the current process (dashboard stays up).")
                    elif restart_result is not None:
                        try:
                            _atomic_write_json(
                                os.path.join(session_dir(_session_id), "restart_failure.json"),
                                {
                                    "returncode": restart_result.returncode,
                                    "stdout": restart_result.stdout or "",
                                    "stderr": restart_result.stderr or "",
                                    "attempt": attempt,
                                    "time": time.time(),
                                },
                            )
                            print("[PULSE] Handed the failure to the AI agent to fix the current code -- see the dashboard chat.")
                        except Exception as exc:
                            print(f"[PULSE] ⚠ Could not write restart_failure.json ({exc}) -- continuing the current process.")
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
                elif msg[0] == "EXEC":
                    _req_id, _kind, _arg = msg[1], msg[2], msg[3]
                    try:
                        _result = _handle_exec_request(caller_frame, _kind, _arg)
                    except Exception:
                        _result = f"{_kind} '{_arg}': Pulse's own dispatcher raised\n{_format_exc_short(traceback.format_exc())}"
                    if response_queue is not None:
                        try:
                            response_queue.put((_req_id, _result))
                        except Exception:
                            pass

        filename = frame.f_code.co_filename
        if _is_library_frame(filename):
            return None
        if root and not os.path.normcase(os.path.abspath(filename)).startswith(root):
            return None

        is_caller = frame is caller_frame

        if event == "call":
            # Only let a nested call (the expensive, many-line frames --
            # attention/layernorm/MLP/backward helpers) opt into line
            # tracing while a snapshot window is open. caller_frame never
            # reaches this branch since its f_trace was set directly below,
            # not via a 'call' event.
            return persistent_tracer if window["open"] else None

        if event != "line":
            return persistent_tracer

        if not window["open"]:
            if not is_caller:
                # This nested frame's window closed while it was still
                # running -- stop asking for further line events from it.
                # A future window will re-trace it fresh via a new 'call'.
                try:
                    frame.f_trace_lines = False
                except Exception:
                    pass
            # caller_frame keeps getting line events even with the window
            # closed (cheap, and needed for Pause/Restart responsiveness --
            # see the control-queue drain above), but skips the expensive
            # locals scan below until the next window.
            return persistent_tracer

        now = time.time()
        # Gate the whole locals scan (is_trackable on every local) behind
        # throttle_interval, same fix as the CLI tracer -- this was
        # previously running unthrottled on every line event and starving
        # GPU dispatch on large models. Halved here since it's now also
        # bounded by the window itself.
        if now - last_scan["t"] <= throttle_interval * 0.5:
            return persistent_tracer
        last_scan["t"] = now
        locals_now = frame.f_locals
        for name, val in locals_now.items():
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

            # If the live value is on an accelerator, Pulse does not touch
            # it. If the user's code already maintains a CPU mirror, use
            # that mirror instead. Creating the mirror remains the user's
            # responsibility and can therefore be scheduled however they
            # choose (or omitted entirely).
            observed_val = val
            if PULSE_CPU_ONLY and not _cpu_resident(val):
                observed_val = None
                for mirror_name in _mirror_name_candidates(name):
                    candidate = locals_now.get(mirror_name)
                    if candidate is not None and _cpu_resident(candidate):
                        observed_val = candidate
                        break
                if observed_val is None:
                    continue

            if now - last_logged.get(name, 0) > throttle_interval:
                # CPU-only observation: do not even enqueue accelerator
                # objects. An mp.Queue cannot be used as a CUDA siphon.
                # Pulse will observe an already-created CPU mirror instead.
                if PULSE_CPU_ONLY and not _cpu_resident(val):
                    continue
                if _debugger_bg.log_matrix(name, observed_val):
                    last_logged[name] = now

        return persistent_tracer

    sys.settrace(persistent_tracer)
    caller_frame.f_trace = persistent_tracer
    caller_frame.f_trace_lines = True


_CRASH_RETRY_DELAYS = (5, 20, 60)


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
        # Disarm the global line/call tracer immediately. Once the script
        # has crashed, nothing below should still be traced -- and if we
        # leave `sys.settrace` pointed at `cli_tracer`, it stays the
        # process-wide trace function into interpreter shutdown, where it
        # gets invoked on unrelated __del__ calls (e.g. litellm's
        # AsyncHTTPHandler) after module globals have already been cleared
        # to None. That produces a second, uncatchable "Exception ignored
        # in ..." failure from *inside* the tracer, separate from (and
        # after) whatever we report here.
        sys.settrace(None)
        try:
            frame = exc_tb.tb_frame if exc_tb else None
            while frame is not None:
                frame.f_trace = None
                frame = frame.f_back
        except Exception:
            pass

        previous_hook(exc_type, exc_value, exc_tb)
        if issubclass(exc_type, KeyboardInterrupt):
            return
        if os.environ.get(_RESTART_CHILD_ENV) == "1":
            # This run was launched by a fix-triggered restart: the parent
            # Pulse process is waiting on it, captures this output, and
            # feeds it back to the agent within its own bounded retry loop.
            # Starting a second fix/restart chain here is what nested into
            # hundreds of agent calls.
            print("\n[Pulse] Restarted run crashed -- handing the traceback back to the Pulse process that restarted it.")
            return
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print("\n[Pulse] Your script just crashed with the exception above.")
        user_frames = []
        tb = exc_tb
        while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            if not filename.startswith("<") and not _is_library_frame(filename):
                user_frames.append(tb.tb_frame)
            tb = tb.tb_next
        cli.capture_crash_state(user_frames)
        # handle_crash logs the traceback (same as the old direct
        # log_traceback call did) AND does the signature/dedup bookkeeping
        # that offer_known_fix below depends on -- without it, a bug
        # Pulse already fixed once would go through the full ask-the-agent
        # pipeline again from scratch every time it recurs, instead of
        # just reapplying the known fix.
        sig = cli.handle_crash(tb_text)
        if cli.offer_known_fix(sig):
            return  # already reapplied (and, if auto_intervene, already restarted)

        try:
            if not cli.agent_provider:
                if cli.auto_intervene:
                    # Auto-fix can't configure an agent on its own -- there's
                    # no API key to use -- so this is the one crash-handling
                    # case that genuinely can't proceed unattended. Say so
                    # and move on rather than blocking on a prompt that (in
                    # an unattended/non-interactive run) will never be
                    # answered.
                    print("[Pulse] Auto-fix is on, but no AI agent is configured -- run /agent to set one up.")
                    return
                _flush_stdin()
                resp = input(
                    "[Pulse] Set up an AI agent now so Pulse can try to diagnose/fix it? (y/n) > "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    return
                if not cli._select_agent_provider_and_key(initial=True):
                    return
            elif not cli.auto_intervene:
                _flush_stdin()
                resp = input(
                    "[Pulse] Ask the agent to diagnose (and try to fix) this? (y/n) > "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    return
            # else: auto_intervene is on and an agent is configured --
            # proceed straight to asking it, no prompt (see the module's
            # "never block on a y/n when autofix is on" convention).
        except (EOFError, KeyboardInterrupt):
            return

        question = (
            f"My script just crashed with this uncaught exception:\n{tb_text}\n"
            "Please diagnose the root cause and, if you can, fix it."
        )
        cli.ask_agent(question, include_code=True)

        # A transient agent failure normally queues a retry on a background
        # ticker while training carries on -- but the script has crashed and
        # the process exits as soon as this hook returns, taking that daemon
        # thread with it. Retry here, in the foreground, instead.
        for delay in _CRASH_RETRY_DELAYS:
            if not cli._last_call_failed_transiently:
                break
            print(f"[Pulse] Agent request failed while handling the crash -- retrying in {delay}s before exiting...")
            time.sleep(delay)
            cli.ask_agent(question, include_code=True)
        if cli._last_call_failed_transiently:
            print("[Pulse] ⚠ Agent still unavailable after retries -- exiting without a fix.")

    sys.excepthook = _hook


def _start_cli_tracker(
    caller_frame, root, throttle_interval, discovered, runtime_shapes,
    code_text=None, script_path=None, extra_files=None, startup_error=None,
    dist_info=None,
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
    cli.dist_info = dist_info or (0, 0, 1, None)
    cli.print_banner()
    # Agent/provider setup used to be deferred until cli_tracer saw its
    # first 'line' event with a trackable/resolved local in scope -- fine
    # for a training loop, but it meant any crash before that point (e.g.
    # failing during data loading, before the loop even starts) left
    # cli.agent_provider unset, so the crash hook below had no agent to
    # hand the traceback to and could only print "no AI agent is
    # configured". discover_variables() already falls back to the
    # statically-discovered `cli.discovered` names (populated from AST
    # parsing at auto_track()-call time, before any user code runs) when
    # there are no live locals yet, so setup doesn't actually need to wait
    # for a live frame -- run it eagerly, before the excepthook can matter.
    cli.interactive_setup()

    # Internally attach Pulse to Keras Model.fit().
    # Users only need auto_track().
    _install_keras_pulse_hook(cli)

    if startup_error:
        print("[Pulse] The function passed to auto_track() raised an exception during its dry run:")
        print(startup_error)
        if cli.agent_provider:
            print("[Pulse] Pulse will offer to diagnose/fix it (via the agent set up above) before continuing.\n")
        else:
            print("[Pulse] No agent configured (see summary above) -- run /agent to set one up before continuing.\n")

    _install_cli_excepthook(cli)

    last_logged = {"t": 0.0}

    # ------------------------------------------------------------------
    # Windowed line tracing.
    #
    # Previously `cli_tracer` was armed for 'line' events on every frame,
    # for the entire run, and stayed armed forever (it always returned
    # `cli_tracer`, never `None`, from a line event). That alone -- before
    # any of the work inside the callback -- is expensive: CPython disables
    # its specializing/adaptive interpreter for any code object under
    # active line tracing, and pays a full Python callback dispatch on
    # every bytecode line boundary. On a hand-written training loop (many
    # small Python-level ops per step -- layernorm, attention, MLP, backward,
    # each its own function/lines) this is a 10-50x slowdown of the Python
    # side. Since cupy/GPU kernel launches are async, a Python side that
    # slow can't issue launches fast enough to keep the GPU fed: VRAM stays
    # full (weights/activations are already resident) but utilization
    # collapses to single digits. Throttling the *work done inside* the
    # callback (as before) doesn't fix this -- the callback still fires,
    # and the interpreter deopt still applies, on every single line
    # regardless.
    #
    # Fix: don't leave line tracing armed continuously. Only open a
    # tracing "window" right when a snapshot is actually due (every
    # `throttle_interval` seconds), keep it open just long enough to catch
    # one step's worth of nested calls, then explicitly disable further
    # line/call events via `frame.f_trace_lines` / returning None until the
    # next window. Between windows, the training loop runs at native,
    # untraced speed.
    _CAPTURE_SPAN = min(0.05, throttle_interval / 4 if throttle_interval > 0 else 0.05)
    window = {"open": True, "closes_at": 0.0}  # start open so the first snapshot can run immediately

    def _close_window():
        window["open"] = False
        try:
            caller_frame.f_trace_lines = False
        except Exception:
            pass

    def _ticker():
        while True:
            time.sleep(throttle_interval)
            window["open"] = True
            window["closes_at"] = time.time() + _CAPTURE_SPAN
            try:
                caller_frame.f_trace_lines = True
            except Exception:
                pass
            time.sleep(_CAPTURE_SPAN)
            if window["open"]:
                _close_window()

    # Setup now runs eagerly, before tracing even starts (see above), so
    # the ticker can start right away too -- no need to wait for the
    # tracer to see a live frame first.
    threading.Thread(target=_ticker, daemon=True).start()

    def cli_tracer(frame, event, arg):
        filename = frame.f_code.co_filename
        if _is_library_frame(filename):
            return None
        if root and not os.path.normcase(os.path.abspath(filename)).startswith(root):
            return None

        if event == "call":
            # Don't opt a nested call into line tracing at all unless a
            # snapshot window is currently open -- this is what keeps
            # forward()/backward() helper calls at full speed between
            # snapshots instead of paying the trace tax on every line
            # inside them every single step.
            return cli_tracer if window["open"] else None

        if event != "line":
            return cli_tracer

        if not window["open"]:
            # Belt-and-braces: make sure this frame stops asking for line
            # events even if it slipped in right as the window closed.
            try:
                frame.f_trace_lines = False
            except Exception:
                pass
            return cli_tracer

        local_vars = frame.f_locals

        # DEBUG: prove whether execution enters the user's model.fit() call
        # and whether tracing resumes only after the whole Keras fit returns.
        try:
            _src_line = linecache.getline(filename, frame.f_lineno).strip()
            if _src_line and ("model.fit" in _src_line or "auto_track" in _src_line):
                _pulse_log(
                    f"TRACE user-code {frame.f_code.co_name} {os.path.basename(filename)}:"
                    f"{frame.f_lineno}: {_src_line!r} | locals={sorted(k for k in local_vars if not k.startswith('__'))}",
                    console=True,
                )
        except Exception:
            pass

        # Fill in shapes for statically-discovered names as soon as they
        # actually resolve to a value -- same as the GUI's shape_tracer
        # dry-run, just kept live instead of front-loaded into a single
        # pre-pass. If a variable was defaulted to 'lotrack' before its
        # shape was known and it turns out to actually be a scalar, upgrade
        # it to full 'track' -- scalars are cheap regardless. This only
        # runs while a window is open now, not on every line forever.
        for name, val in local_vars.items():
            if name in cli.discovered and cli.discovered.get(name) is None and is_trackable(val):
                cli.discovered[name] = shape_of(val)
                if (
                    name in cli.tracked_vars
                    and cli.var_states.get(name) == "lotrack"
                    and shape_of(val) == ()
                ):
                    cli.var_states[name] = "track"

        # MERGE, don't replace: a snapshot window can land on any traced
        # frame -- the outer train.py loop, or a nested call like a Keras
        # callback's on_epoch_end() in a different file -- and different
        # frames define different names. Wholesale-replacing watch_locals
        # with whichever frame happened to fire the most recent 'line'
        # event meant a variable that's only ever assigned inside, say,
        # on_train_batch_end() would read back as missing (-> reported to
        # the agent as "NoneType, not run yet") the instant some other
        # frame's line event was the one caught next, even though its real
        # value hadn't changed at all -- it just wasn't a local of *that*
        # particular frame. Once a name has been observed anywhere, its
        # last known value should stick until it's actually reassigned.
        cli.watch_locals.update(local_vars)

        now = time.time()
        if now - last_logged["t"] > throttle_interval * 0.5:
            # Auto mode (the CLI's equivalent of the GUI's "track every
            # matrix automatically" toggle): keep picking up newly-trackable
            # locals as the loop runs, instead of being limited to what was
            # chosen once at setup time. Still cheap now: this whole branch
            # only runs during the brief open window, not on every line.
            if cli.auto_mode:
                for name, val in local_vars.items():
                    if name.startswith("__") or name in cli.tracked_vars:
                        continue
                    if is_trackable(val):
                        cli.tracked_vars.append(name)
                        cli.var_states[name] = cli._default_state_for(name, val)

            try:
                cli.update()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(
                    f"[Pulse] Warning: debugger update failed ({type(exc).__name__}: {exc}); "
                    "continuing training."
                )
            last_logged["t"] = now

        return cli_tracer

    sys.settrace(cli_tracer)
    caller_frame.f_trace = cli_tracer
    caller_frame.f_trace_lines = True
    # No routine "tracing started" print -- minimal UI stays silent unless
    # something's actually wrong (a read error, a crash, auto-intervention).


def shutdown():
    global _debugger_bg
    if _debugger_bg:
        _debugger_bg.shutdown()
        _debugger_bg = None
