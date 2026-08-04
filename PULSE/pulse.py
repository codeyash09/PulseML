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
import sys
import json
import time
import uuid
import tempfile
import threading
import sysconfig
import multiprocessing as mp
import ast
import base64
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import tkinter as tk
from tkinter import ttk, Toplevel, scrolledtext, simpledialog
from PIL import Image, ImageTk

from PULSE.pulse_backend import (
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
    rendering. This never mutates the original history."""
    n = len(points)
    if n <= target:
        return list(points)

    bucket = math.ceil(n / target)
    out = []
    for i in range(0, n, bucket):
        chunk = points[i:i + bucket]
        avg_value = sum(p[1] for p in chunk) / len(chunk)
        # use mean step index for better x placement
        avg_step = int(round(sum(p[0] for p in chunk) / len(chunk)))
        out.append((avg_step, avg_value))
    return out


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
    """
    safe = np.abs(arr2d.astype(np.float64)) + 1e-12
    entry = fig_cache.get(var)

    if entry is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI, facecolor=BG)
        ax.set_facecolor(BG)
        im = ax.imshow(safe, cmap=CMAP, norm=LogNorm(vmin=safe.min(), vmax=safe.max()), aspect="auto")
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
            im.set_norm(LogNorm(vmin=safe.min(), vmax=safe.max()))
        except Exception:
            pass  # degenerate (all-equal) arrays -- keep the previous norm

    tmp = path + ".tmp.png"
    fig.savefig(tmp, facecolor=BG)
    os.replace(tmp, path)


def _save_linechart(points, path, var_name, fig_cache):
    """Renders a scalar's (step, value) point history -- already downsampled
    for display by the caller -- as a step chart (flat until the value
    actually changes, then jumps), which is the GUI equivalent of the CLI's
    ASCII chart. Same reuse-the-figure strategy as `_save_heatmap`."""
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
        fig.tight_layout(pad=0.4)
        fig_cache[var_name] = (fig, ax, line, scatter)
    else:
        fig, ax, line, scatter = entry
        line.set_data(xs, ys)
        ax.relim()
        ax.autoscale_view()

    if points:
        scatter.set_offsets([[xs[-1], ys[-1]]])

    tmp = path + ".tmp.png"
    fig.savefig(tmp, facecolor=BG)
    os.replace(tmp, path)


def _worker_main(queue, display_configs, session_id):
    cache = session_dir(session_id)
    manifest_path = os.path.join(cache, "manifest.json")
    manifest = {}
    last_numpy_arrays = {}

    # var -> full, NEVER-truncated list of (step, value) tuples. A new point
    # is only appended when the value actually differs from the last one
    # recorded, so a loss that's flat for 500 steps costs one point, not 500
    # -- the step chart drawstyle in _save_linechart fills in the flat
    # segments visually without needing a point at every step.
    scalar_histories = {}
    # var -> latest step index seen (whether or not it produced a new point),
    # used only to extend the rendered line up to "now".
    scalar_step_counters = {}

    # Persistent Matplotlib figures, reused across steps instead of being
    # rebuilt every call -- see _save_heatmap / _save_linechart.
    heatmap_figs = {}
    linechart_figs = {}

    while True:
        item = queue.get()
        if item is None:
            break

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

        try:
            kind = tensor_kind(matrix)
            if kind == "scalar":
                # Loss, accuracy, lr, or any other shape-() value: track a
                # rolling (step, value) history -- deduplicated so flat runs
                # don't cost a point per step -- and render it as a step
                # chart rather than a 1x1 "heatmap", which would be useless.
                value = scalar_value(matrix)

                # Only advance the step counter when the scalar value actually changes.
                # This makes the x-axis move only on real changes.
                last_step = scalar_step_counters.get(var, 0)
                hist = scalar_histories.setdefault(var, [])
                last_value = hist[-1][1] if hist else None

                if hist and last_value == value:
                    # value unchanged: do not increment step counter, do not append a new point
                    step_count = last_step
                else:
                    # value changed (or first value): advance step and append
                    step_count = last_step + 1
                    scalar_step_counters[var] = step_count
                    hist.append((step_count, value))

                # For rendering, use the raw history (no synthetic extension).
                display_points = _downsample_for_display(hist)
                version_tag = time.time_ns()
                img_path = os.path.join(cache, f"{var}_{version_tag}.png")
                _save_linechart(display_points, img_path, var, linechart_figs)

                stats = backend_statistics(matrix)
                stats["image"] = img_path
                stats["updated"] = time.time()
                stats["latest_value"] = value
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
                manifest[var] = stats
        except Exception as e:
            manifest[var] = {"error": str(e), "updated": time.time()}

        _atomic_write_json(manifest_path, manifest)


class HeatmapCreatorBG:
    def __init__(self, display_configs, session_id):
        self.session_id = session_id
        self.queue = mp.Queue()
        self.process = mp.Process(target=_worker_main, args=(self.queue, display_configs, session_id), daemon=True)
        self.process.start()

    def log_matrix(self, var, matrix, config_override=None):
        self.queue.put((var, matrix, config_override))

    def update_config(self, var, new_config):
        self.queue.put(("CONFIG", var, new_config))

    def shutdown(self):
        self.queue.put(None)
        self.process.join(timeout=5)


# ============================================================================
# config_ui: initial matrix discovery + selection/axis picker
# ============================================================================

def discover_static_names(caller_frame):
    """Parse the caller's source with ast to find every assignment target
    anywhere in the file -- including inside nested functions that haven't
    run yet (e.g. a `scores` matrix inside an `attention()` only called
    from within the training loop). Pure source parsing, doesn't execute
    anything, so it's safe to run before training starts."""
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

    filename = caller_frame.f_code.co_filename
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf8") as f:
                tree = ast.parse(f.read(), filename)
            Visitor().visit(tree)
        except Exception:
            pass

    noise = {"i", "j", "k", "_", "self", "cls", "e", "args", "kwargs"}
    return discovered - noise


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
        self.axis_buttons = {}
        self.on_config_change = on_config_change  # callback(name, new_config)
        self.auto_mode = tk.BooleanVar(value=False)
        self.result = None

        top = ttk.Frame(body)
        top.pack(fill=tk.X, pady=(0, 12))
        ttk.Checkbutton(
            top,
            text="Track every matrix automatically (incl. ones that appear later)",
            variable=self.auto_mode,
        ).pack(side=tk.LEFT)
        ttk.Button(top, text="Select All", command=self._select_all).pack(side=tk.RIGHT)

        left_frame = ttk.LabelFrame(body, text="  1 · SELECT MATRICES  ", padding=12)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.right_frame = ttk.LabelFrame(
            body,
            text="  2 · CONFIGURE AXES  ·  click = show  ·  double-click = iterate  ·  right-click = reset  ",
            padding=12,
        )
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

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
            self.final_configs[name] = [0] * len(shape) if shape else []
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
                command=lambda n=name: self._toggle_matrix_view(n),
            )
            self._matrix_checkbuttons[name] = cb
            cb.pack(anchor="w", pady=4)
            if is_loss:
                preselected.append(name)

        self._filter_matrix_list()
        for name in preselected:
            self._toggle_matrix_view(name)

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
            if not var.get():
                var.set(True)
                self._toggle_matrix_view(name)


    def _open_axis_slideshow(self, name, axis_idx):
        # Ensure the axis is in iterate mode; if not, initialize it to iterate@0
        cfg = self.final_configs.setdefault(name, [0] * len(self.discovered[name]))
        if cfg[axis_idx] < 2:
            cfg[axis_idx] = 2  # iterate@0

        # helper to get current iterate index
        def get_iter_index():
            return cfg[axis_idx] - 2

        popup = Toplevel(self.root)
        popup.title(f"{name} — Axis {axis_idx} slideshow")
        popup.configure(bg=BG)
        popup.geometry("420x320")

        # image placeholder (you can replace with actual image loading from cache)
        img_label = ttk.Label(popup, text="(preview)", style="Dim.TLabel")
        img_label.pack(pady=(12, 6))

        idx_label = ttk.Label(popup, text=f"Index: {get_iter_index()}", font=FONT_UI_BOLD)
        idx_label.pack()

        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=12)

        def apply_and_notify():
            # update button text in the main axis UI
            self._set_axis_val(name, axis_idx, cfg[axis_idx])
            if callable(self.on_config_change):
                # send a copy of the config
                self.on_config_change(name, list(cfg))

        def prev_idx():
            if get_iter_index() > 0:
                cfg[axis_idx] -= 1
                idx_label.config(text=f"Index: {get_iter_index()}")
                apply_and_notify()

        def next_idx():
            # clamp to axis length - 1
            axis_len = self.discovered[name][axis_idx]
            if get_iter_index() < max(0, axis_len - 1):
                cfg[axis_idx] += 1
                idx_label.config(text=f"Index: {get_iter_index()}")
                apply_and_notify()

        ttk.Button(btn_frame, text="Previous", command=prev_idx).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="Next", command=next_idx).pack(side=tk.LEFT, padx=8)
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=(8, 12))


    def _toggle_matrix_view(self, name):
        if self.matrix_vars[name].get():
            shape = self.discovered[name]
            if not shape:  # empty tuple (scalar) or None (not seen yet) -- nothing to configure
                return
            row_frame = ttk.Frame(self.right_frame, name=name.lower().replace(".", "_"))
            row_frame.pack(fill=tk.X, pady=7, anchor="w")
            ttk.Label(row_frame, text=f"{name}", font=FONT_UI_BOLD).pack(side=tk.LEFT, padx=(0, 8))
            self.axis_buttons[name] = []
            for axis_idx, dim_size in enumerate(shape):
                btn = ttk.Button(row_frame, text=f"Ax {axis_idx}\n({dim_size})\n{_axis_val_label(0)}", width=8)
                btn.pack(side=tk.LEFT, padx=3)
                btn.bind("<Button-1>", lambda e, n=name, ax=axis_idx: self._set_axis_val(n, ax, 1))
                btn.bind("<Button-3>", lambda e, n=name, ax=axis_idx: self._set_axis_val(n, ax, 0))

                self.axis_buttons[name].append(btn)
        else:
            try:
                self.right_frame.nametowidget(name.lower().replace(".", "_")).destroy()
                del self.axis_buttons[name]
            except Exception:
                pass

    def _set_axis_val(self, name, axis_idx, val):
        self.final_configs[name][axis_idx] = val
        dim_size = self.discovered[name][axis_idx]
        self.axis_buttons[name][axis_idx].config(text=f"Ax {axis_idx}\n({dim_size})\n{_axis_val_label(val)}")

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
    "You are Pulse, an AI analyst embedded directly inside a live ML training debugger. "
    "Your job is to help the user find and fix the root cause of instability in their training "
    "run -- not to give generic machine learning advice.\n\n"
    "WHAT YOU'RE GIVEN EACH TURN:\n"
    "- Live statistics for every matrix/tensor the user is tracking: backend (PyTorch/TensorFlow/"
    "NumPy/CuPy/JAX), shape, min, max, mean, std, and nan/inf counts.\n"
    "- Scalars (loss, accuracy, learning rate -- anything shape ()) are tracked as a running "
    "history and charted as a line graph; their stats include the latest value.\n"
    "- Heatmap images of matrix/tensor variables when available (log-scale color, dark "
    "background). Look for banding, dead rows/columns, saturation at the extremes, or regions "
    "that break from the surrounding pattern -- these are usually the tell.\n"
    "- The user's training code, but only on turns where they've checked 'Send Code' -- if it "
    "isn't present, don't assume what the code looks like.\n\n"
    "HOW TO REASON:\n"
    "- Ground every claim in the specific numbers or image you were actually given. Name the "
    "matrix and the exact pattern -- 'gradients has std=142.7 and 340 inf values, consistent with "
    "an exploding gradient' beats 'you may have exploding gradients.'\n"
    "- Common root causes worth checking against the data: NaN/Inf propagating from an earlier "
    "layer, exploding or vanishing gradients, dead ReLU units, bad weight initialization, "
    "learning rate too high, mismatched loss scaling, division by zero or log(0) in a custom "
    "loss, unnormalized inputs, mixed-precision underflow.\n"
    "- If code is available, point to the specific line or operation likely responsible. If it "
    "isn't, say what you'd need to see, and note that checking 'Send Code' would let you look.\n"
    "- If nothing in the current data looks abnormal, say so plainly rather than inventing a "
    "problem to sound useful.\n\n"
    "REMEMBER: WHENEVER THE USER ASKS FOR A DIAGNOSIS, GIVE IT IN ONE SENTENCE, THEN EXPLAIN YOUR REASONING AND THEN PROVIDE A SOLUTION. ALWAYS USE MATHEMATICAL REASONING TO DETERMINE THE ISSUE. IE IF THE REASON IS NUMERICAL INSTABILITY, USE MATH TO SHOW HOW AN ERROR IN THEIR CODE IS CAUSING IT, TRANSLATE EVERYTHING TO MATH\n"
    "STYLE: Be concise and direct. Lead with the diagnosis or the most likely cause, not preamble."
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


class ChatPanel(tk.Frame):
    def __init__(self, parent, get_manifest_fn, get_code_fn=None):
        super().__init__(parent, bg=PANEL)
        self.get_manifest_fn = get_manifest_fn
        self.get_code_fn = get_code_fn
        self.history = []
        self.session_keys = {}

        head = tk.Frame(self, bg=PANEL)
        head.pack(fill=tk.X, padx=14, pady=(14, 6))

        self.dot = tk.Canvas(head, width=8, height=8, bg=PANEL, highlightthickness=0)
        self.dot_id = self.dot.create_oval(1, 1, 7, 7, fill=TEAL, outline="")
        self.dot.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(head, text="PULSE AI ANALYST", bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        self.provider_var = tk.StringVar(value=list(PROVIDERS.keys())[0])
        self.provider_dropdown = tk.OptionMenu(head, self.provider_var, *PROVIDERS.keys(), command=self._on_provider_change)
        self.provider_dropdown.config(bg=CARD, fg=TEXT, activebackground=CARD_HOVER, activeforeground=TEXT,
                                       relief="flat", highlightthickness=0, font=("Segoe UI", 9))
        self.provider_dropdown["menu"].config(bg=CARD, fg=TEXT, activebackground=ORANGE, activeforeground="#0a0a0a")
        self.provider_dropdown.pack(side=tk.RIGHT)

        self.transcript = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, state="disabled", height=24,
            bg=CARD, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, padx=10, pady=10,
            font=FONT_UI, highlightthickness=1, highlightbackground=BORDER,
        )
        self.transcript.tag_configure("who_user", foreground=TEAL, font=FONT_UI_BOLD)
        self.transcript.tag_configure("who_ai", foreground=ORANGE, font=FONT_UI_BOLD)
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        entry_frame = tk.Frame(self, bg=PANEL)
        entry_frame.pack(fill=tk.X, padx=14, pady=(0, 14))

        send_btn = tk.Button(
            entry_frame, text="Send", command=self.send,
            bg=ORANGE, fg="#0a0a0a", activebackground=AMBER, activeforeground="#0a0a0a",
            relief="flat", font=FONT_UI_BOLD, padx=14, bd=0, cursor="hand2",
        )
        send_btn.pack(side=tk.RIGHT)

        self.send_code_var = tk.BooleanVar(value=False)
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

    def _append(self, who, text):
        self.transcript.configure(state="normal")
        tag = "who_user" if who.startswith("You") else "who_ai"
        self.transcript.insert(tk.END, f"{who}\n", tag)
        self.transcript.insert(tk.END, f"{text}\n\n")
        self.transcript.configure(state="disabled")
        self.transcript.see(tk.END)

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

    def _build_context(self, include_code=False):
        manifest = self.get_manifest_fn() or {}
        context = "Current tracked matrix/scalar stats:\n"
        for name, s in manifest.items():
            if "error" in s:
                context += f"- {name}: error reading matrix ({s['error']})\n"
                continue
            if s.get("kind") == "scalar":
                context += (
                    f"- {name}: scalar, backend={s.get('backend')}, latest_value={s.get('latest_value')}, "
                    f"nan={s.get('nan')} inf={s.get('inf')}\n"
                )
            else:
                context += (
                    f"- {name}: shape={s.get('shape')} backend={s.get('backend')} kind={s.get('kind')} "
                    f"min={s.get('min')} max={s.get('max')} mean={s.get('mean')} std={s.get('std')} "
                    f"nan={s.get('nan')} inf={s.get('inf')}\n"
                )
        if include_code and self.get_code_fn:
            code = self.get_code_fn()
            if code:
                context += f"\nTraining code (attached by user this turn):\n```python\n{code}\n```\n"
            else:
                context += "\n(User checked 'Send Code' but no code text is available.)\n"
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

    def _ask(self, question, include_code, provider_name):
        model_name = PROVIDERS[provider_name]["model"]
        context = self._build_context(include_code=include_code)
        user_content = [{"type": "text", "text": f"{context}\nQuestion: {question}"}]
        user_content.extend(self._image_payloads())

        self.history.append({"role": "user", "content": user_content})

        try:
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-10:],
                max_tokens=800,
                timeout=30.0,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"(request failed: {e})"

        self.history.append({"role": "assistant", "content": answer})
        self.after(0, lambda: self._append("Pulse", answer))


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

    def __init__(self, session_id, display_configs, code_text=None, config_queue=None, control_queue=None, known_names=None):
        self.session_id = session_id
        self.cache = session_dir(session_id)
        self.manifest_path = os.path.join(self.cache, "manifest.json")
        self.display_configs = display_configs
        self.config_queue = config_queue
        self.control_queue = control_queue
        self.code_text = code_text
        self.known_names = set(known_names or [])
        self._tiles = {}
        self._manifest = {}

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
        tk.Label(left_head, text="  right-click a tile to reconfigure axes", bg=BG, fg=TEXT_FAINT,
                 font=FONT_MONO).pack(side=tk.LEFT)

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

        self.chat_panel = ChatPanel(right, get_manifest_fn=lambda: self._manifest, get_code_fn=lambda: self.code_text)
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
        manifest = self._load_manifest()
        self._manifest = manifest

        for name, stats in sorted(manifest.items()):
            if name in self.hidden_tiles:
                continue
            if "error" in stats:
                self.pending_tiles.discard(name)
                continue
            img_path = stats.get("image")
            if not img_path or not os.path.exists(img_path):
                continue

            tile = self._tiles.get(name)
            current_img_path = img_path

            if tile is not None and tile.get("img_path") == current_img_path:
                continue

            img = Image.open(current_img_path)
            thumb = img.copy()
            thumb.thumbnail(self.THUMB_SIZE)
            photo = ImageTk.PhotoImage(thumb)

            is_scalar = stats.get("kind") == "scalar"
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

                tile = {"frame": frame, "label": label, "sub": sub, "img_path": current_img_path}
                self._tiles[name] = tile
            else:
                tile["frame"].configure(highlightbackground=border_color)
                tile["label"].configure(image=photo)
                tile["label"].image = photo
                tile["img_path"] = current_img_path
                tile["label"].unbind("<Button-1>")
                tile["label"].bind("<Button-1>", lambda e, n=name: self._enlarge(n, self._tiles[n]["img_path"]))
                tile["label"].unbind("<Button-3>")
                tile["label"].bind("<Button-3>", self._make_tile_context_menu(name))

            if is_scalar:
                latest = stats.get("latest_value")
                latest_str = f"{latest:.4f}" if isinstance(latest, (int, float)) else "n/a"
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

    def _make_tile_context_menu(self, name):
        def handler(event):
            stats = self._manifest.get(name, {})
            menu = tk.Menu(self.root, tearoff=0, bg=CARD, fg=TEXT, activebackground=ORANGE,
                            activeforeground="#0a0a0a", bd=0, relief="flat")
            if stats.get("kind") != "scalar":
                menu.add_command(label="Change Axes", command=lambda: self._open_axis_picker_popup(name))
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
        if self.control_queue is not None:
            try:
                self.control_queue.put(("ADD_VAR", name))
            except Exception:
                pass
        if self.config_queue is not None:
            try:
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


def _run_dashboard(session_id, display_configs, code_text=None, config_queue=None, control_queue=None, known_names=None):
    Dashboard(session_id, display_configs, code_text=code_text, config_queue=config_queue, control_queue=control_queue, known_names=known_names).run()


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
    in scope right now end up as candidates."""
    static_names = discover_static_names(caller_frame)

    runtime = {}
    for name, val in caller_frame.f_locals.items():
        if name.startswith("__"):
            continue
        if is_trackable(val):
            runtime[name] = shape_of(val)

    discovered = {name: None for name in static_names}
    discovered.update(runtime)
    return discovered


def auto_track(train_fn=None, throttle_interval=1.0, code_text=None, project_root=None, mode="auto"):
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

    if code_text is None:
        entry_path = caller_frame.f_code.co_filename
        if os.path.exists(entry_path) and not entry_path.startswith("<"):
            try:
                with open(entry_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_text = f.read()
            except OSError:
                code_text = None

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

    if train_fn is not None:
        sys.settrace(shape_tracer)
        caller_frame.f_trace = shape_tracer
        try:
            train_fn()
        except Exception:
            pass
        finally:
            sys.settrace(None)

    if not discovered:
        print("[PULSE] No trackable variables found (nothing in scope, and nothing parseable in the source).")
        return

    if active_mode == "cli":
        _start_cli_tracker(caller_frame, root, throttle_interval, discovered, runtime_shapes)
        return

    # ---- UI mode ----
    picker_data = {var: runtime_shapes.get(var) for var in discovered}
    ui = MatrixConfigUI(picker_data)
    result = ui.run()
    if not result:
        print("[PULSE] Setup cancelled.")
        return

    auto_mode = result["auto"]
    if auto_mode:
        tracked_vars = set(result["vars"])
        display_configs = {}
    else:
        tracked_vars = set(result["vars"].keys())
        display_configs = {k: v for k, v in result["vars"].items() if v}

    if not tracked_vars:
        print("[PULSE] No matrices selected.")
        return

    print(f"[PULSE] Mode: UI | Tracking: {sorted(tracked_vars)}" + (" (+ auto-discovering new ones)" if auto_mode else ""))

    _session_id = str(uuid.uuid4())[:8]
    _debugger_bg = HeatmapCreatorBG(display_configs, _session_id)
    shared_config_queue = _debugger_bg.queue
    control_queue = mp.Queue()

    dash_process = mp.Process(
        target=_run_dashboard,
        args=(_session_id, display_configs, code_text, shared_config_queue, control_queue, sorted(discovered)),
        daemon=True,
    )
    dash_process.start()

    last_logged = {}

    def persistent_tracer(frame, event, arg):
        while True:
            try:
                msg = control_queue.get_nowait()
            except Exception:
                break
            if isinstance(msg, tuple) and msg and msg[0] == "ADD_VAR":
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


def _start_cli_tracker(caller_frame, root, throttle_interval, discovered, runtime_shapes):
    """CLI mode: synchronous, in-process -- no subprocess, no multiprocessing
    Queue, no pickling tensors across a process boundary (which can be
    genuinely broken for CUDA tensors anyway). Just prints as training runs
    and, per your setup choice, saves labeled PDF snapshots."""
    from .pulse_cli import PulseCLI

    cli = PulseCLI()
    cli.print_banner()

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

        if not setup_done["value"]:
            trackable = {n: v for n, v in local_vars.items() if not n.startswith("__") and is_trackable(v)}
            if trackable:
                cli.watch_locals = local_vars
                cli.interactive_setup()
                setup_done["value"] = True
            return cli_tracer

        now = time.time()
        if now - last_logged["t"] > throttle_interval:
            cli.watch_locals = local_vars
            cli.update()
            last_logged["t"] = now

        return cli_tracer

    sys.settrace(cli_tracer)
    caller_frame.f_trace = cli_tracer
    print("[PULSE] Mode: CLI | tracing started -- run your training loop now.")


def shutdown():
    global _debugger_bg
    if _debugger_bg:
        _debugger_bg.shutdown()
        _debugger_bg = None
