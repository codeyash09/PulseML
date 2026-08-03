"""
pulse_cli.py
============

Command-Line Edition of Pulse for Google Colab, SSH, and headless servers.

No GUI, no heatmap images displayed inline -- matrices/tensors are just
"tagged" (their stats printed each step). Scalars (loss, accuracy, lr,
anything shape ()) get a live ASCII history chart instead, since that's
cheap to render as text. If you opt in during setup, matrix/tensor
snapshots are additionally saved as labeled PDFs (one folder per variable,
one file per step) via pulse_pdf.generate_heatmap_pdf, for cases where you
want the actual heatmap later even though nothing's shown on screen now.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

from pulse_backend import (
    available_backends,
    describe_tensor,
    detect_backend,
    is_trackable,
    statistics,
    to_numpy,
)
from pulse_pdf import generate_heatmap_pdf


class PulseCLI:
    def __init__(self, watch_locals: Optional[Dict[str, Any]] = None, pdf_dir: str = "Pulse_Output"):
        self.watch_locals = watch_locals or {}
        self.tracked_vars: List[str] = []
        self.scalar_histories: Dict[str, List[float]] = {}
        self.step = 0
        self.generate_pdfs = False
        self.pdf_dir = pdf_dir

    def print_banner(self) -> None:
        backends = available_backends()
        print("\n" + "=" * 50)
        print("                 PULSE CLI EDITION                 ")
        print("=" * 50)
        print("Detected backend(s):")
        for name, active in backends.items():
            status = "✓" if active else "✗"
            print(f"  {status} {name}")
        print("-" * 50 + "\n")

    def discover_variables(self) -> Dict[str, Any]:
        """Finds trackable variables (any backend, any numeric shape) from the provided scope."""
        trackable = {}
        for name, val in self.watch_locals.items():
            if name.startswith("_"):
                continue
            if is_trackable(val):
                trackable[name] = val
        return trackable

    def interactive_setup(self) -> None:
        """Interactive terminal menu to select variables to track, then ask
        once whether matrix/tensor snapshots should also be saved as PDFs."""
        variables = self.discover_variables()
        if not variables:
            print("[Pulse CLI] No trackable variables found in scope.")
            return

        print("Detected variables:")
        var_list = list(variables.keys())
        for idx, name in enumerate(var_list, 1):
            info = describe_tensor(variables[name])
            print(f"  {idx}) {name} [{info.backend} {info.kind} {info.shape}]")

        print("\nEnter variable numbers to track (e.g., 1,2,5 or 'all', or press Enter to skip):")
        choice = input("> ").strip()

        if not choice:
            return

        if choice.lower() == "all":
            self.tracked_vars = var_list
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
                self.tracked_vars = [var_list[i - 1] for i in indices if 0 < i <= len(var_list)]
            except Exception:
                print("[Pulse CLI] Invalid selection input. Tracking nothing.")
                return

        print(f"\n✓ Tracking variables: {', '.join(self.tracked_vars)}")

        # Only worth asking if at least one tracked var is an actual
        # matrix/tensor -- pure scalars (loss, lr, etc.) already get a
        # free ASCII chart and have no heatmap to save anyway.
        has_matrix_like = any(
            describe_tensor(variables[name]).kind in ("vector", "matrix", "tensor")
            for name in self.tracked_vars
            if name in variables
        )
        if has_matrix_like:
            print(
                "\nNo heatmaps are shown in this CLI view -- matrices/tensors are just "
                "tagged (their stats printed each step)."
            )
            resp = input(f"Save labeled PDF snapshots to '{self.pdf_dir}/<variable>/' each step? (y/n): ").strip().lower()
            self.generate_pdfs = resp in ("y", "yes")
            if self.generate_pdfs:
                print(f"[Pulse CLI] PDF snapshots enabled -> {self.pdf_dir}/<variable>/stepNNNNNN.pdf\n")
            else:
                print("[Pulse CLI] PDF snapshots disabled -- only stats will be printed.\n")
        else:
            print()

    def update(self, step: Optional[int] = None, generate_pdfs: Optional[bool] = None) -> None:
        """Called at every training step/checkpoint.

        generate_pdfs: override the interactive_setup() choice for this
        call only. Leave as None to use whatever was chosen at setup.
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1

        want_pdfs = self.generate_pdfs if generate_pdfs is None else generate_pdfs

        print(f"\n--- Pulse Step {self.step} ---")

        for var_name in self.tracked_vars:
            if var_name not in self.watch_locals:
                continue

            val = self.watch_locals[var_name]
            if not is_trackable(val):
                continue
            stats = statistics(val)

            if stats["kind"] == "scalar":
                scalar_val = float(to_numpy(val).reshape(-1)[0])
                self.scalar_histories.setdefault(var_name, []).append(scalar_val)

                print(f"  • {var_name}: {scalar_val:.4f}")
                self._print_ascii_chart(var_name, self.scalar_histories[var_name])
            else:
                flag = ""
                if stats["nan"] or stats["inf"]:
                    flag = f"  ⚠ nan={stats['nan']} inf={stats['inf']}"
                print(
                    f"  • Tagging '{var_name}' [{stats['backend']} {stats['kind']} {stats['shape']}] "
                    f"| mean={stats['mean']:.4f} min={stats['min']:.4f} max={stats['max']:.4f}{flag}"
                )

                if want_pdfs:
                    pdf_path = generate_heatmap_pdf(var_name, val, self.step, output_dir=self.pdf_dir)
                    print(f"    ↳ saved snapshot: {pdf_path}")

    def _print_ascii_chart(self, name: str, history: List[float], height: int = 5, width: int = 30) -> None:
        """Renders a simple ASCII history chart in the terminal -- this is
        the CLI's equivalent of the GUI's loss/scalar line graph."""
        if not history:
            return

        data = history[-width:]
        min_val = min(data)
        max_val = max(data)
        val_range = max_val - min_val if max_val != min_val else 1.0

        print(f"    [{name} history over last {len(data)} steps]")
        blocks = "  ▂▃▄▅▆▇█"

        for h in range(height - 1, -1, -1):
            threshold = min_val + (h / (height - 1)) * val_range if height > 1 else max_val
            line = f"    {threshold:7.2f} ┤ "
            for val in data:
                normalized = (val - min_val) / val_range if val_range > 0 else 0.5
                block_idx = int(normalized * (len(blocks) - 1))
                block_idx = max(0, min(block_idx, len(blocks) - 1))

                row_frac = h / (height - 1) if height > 1 else 0.5
                if normalized >= row_frac - (1.0 / (height * 2)):
                    line += blocks[block_idx]
                else:
                    line += " "
            print(line)
        print(f"             └" + "─" * len(data))