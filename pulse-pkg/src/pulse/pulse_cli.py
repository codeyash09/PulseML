"""
pulse_cli.py
============
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import signal
import getpass
import itertools
from typing import Any, Dict, List, Optional

from pulse.pulse_backend import (
    available_backends,
    describe_tensor,
    detect_backend,
    is_trackable,
    statistics,
    to_numpy,
)
from pulse.pulse_pdf import generate_heatmap_pdf
import litellm


# Name-based heuristic for pre-flagging the loss/metric scalar -- same list
# and same purpose as pulse.py's LOSS_NAME_HINTS/_looks_like_loss, kept as a
# local copy so this module has no dependency on pulse.py (which pulls in
# tkinter and isn't safe to import in a headless CLI/Colab/SSH session).
LOSS_NAME_HINTS = ("loss", "cost", "nll", "cross_entropy", "crossentropy", "objective", "err")


def _looks_like_loss(name: str) -> bool:
    n = (name or "").lower()
    return any(hint in n for hint in LOSS_NAME_HINTS)

# ----------------------------------------------------------------------------
# CLI AI agent
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Pulse, an AI analyst embedded in a live ML training debugger. "
    "Find the root cause of instability in the user's training run, not generic ML advice.\n\n"
    "INPUTS:\n"
    "- Current tracked matrix/tensor/scalar stats.\n"
    "- Static variable names and shapes discovered from the user's source.\n"
    "- Training code when available.\n\n"
    "RESPONSE FORMAT:\n"
    "1. Diagnosis — one sentence, the specific root cause.\n"
    "2. Reasoning — grounded in the actual numbers and code, with real math.\n"
    "3. Fix — a concrete change.\n\n"
    "Be concise. Do not invent problems or data that were not provided.\n\n"

    "CODE FIXES:\n"
    "If, and only if, the user explicitly asks you to fix, edit, patch, or change the code "
    "(not just diagnose it), respond with ONLY a single JSON object and nothing else -- no prose "
    "before or after it, no markdown code fences. The JSON object must have exactly these fields:\n"
    "  old: a list of code snippets to find, each copied EXACTLY from the line-numbered training "
    "code shown to you, including original indentation and whitespace, but WITHOUT the line-number "
    "prefix ('  12 | ') itself.\n"
    "  new: a list of the same length as old, where new[i] is the full replacement for old[i].\n"
    "  explanation: a short, concise text description of what changed and why.\n"
    "Rules for old/new:\n"
    "  - Each snippet in old must appear VERBATIM and exactly ONCE in the current file. Include "
    "enough surrounding lines (not just the single changed line) so the match is unambiguous.\n"
    "  - Each new[i] is the complete replacement block for old[i] -- to add a line, copy old[i] and "
    "append the new line(s) to it; to remove a line, copy old[i] and omit it.\n"
    "  - Never use placeholders like '...' or '# unchanged' inside old or new; both must be literal, "
    "complete code.\n"
    "  - If you were not shown the code, or the user has not asked for a fix, do not emit this JSON "
    "format -- answer normally per RESPONSE FORMAT above."
)

PROVIDERS = {
    "Anthropic (Claude Sonnet 5)": {
        "model": "anthropic/claude-sonnet-5",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "Anthropic (Claude Opus 4.8)": {
        "model": "anthropic/claude-opus-4-8",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "Anthropic (Claude Haiku 4.5)": {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "OpenAI (GPT-5.5)": {
        "model": "openai/gpt-5.5",
        "env_key": "OPENAI_API_KEY",
    },
    "OpenAI (GPT-5.4)": {
        "model": "openai/gpt-5.4",
        "env_key": "OPENAI_API_KEY",
    },
    "OpenAI (GPT-5.3 Codex)": {
        "model": "openai/gpt-5.3-codex",
        "env_key": "OPENAI_API_KEY",
    },
    "Google AI Studio (Gemini 3.1 Pro)": {
        "model": "gemini/gemini-3.1-pro-preview",
        "env_key": "GEMINI_API_KEY",
    },
    "Google AI Studio (Gemini 3.6 Flash)": {
        "model": "gemini/gemini-3.6-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "Google AI Studio (Gemini 3.5 Flash-Lite)": {
        "model": "gemini/gemini-3.5-flash-lite",
        "env_key": "GEMINI_API_KEY",
    },
    "DeepSeek": {
        "model": "deepseek/deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "Mistral": {
        "model": "mistral/mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
    },
    "OpenRouter (Llama 3.3 70B, free)": {
        "model": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "env_key": "OPENROUTER_API_KEY",
    },
    "OpenRouter (GPT-OSS 120B, free)": {
        "model": "openrouter/openai/gpt-oss-120b:free",
        "env_key": "OPENROUTER_API_KEY",
    },
}


class PulseCLI:
    def __init__(
        self,
        watch_locals: Optional[Dict[str, Any]] = None,
        pdf_dir: str = "Pulse_Output",
        discovered: Optional[Dict[str, Optional[tuple]]] = None,
    ):
        self.watch_locals = watch_locals or {}
        self.discovered: Dict[str, Optional[tuple]] = dict(discovered or {})
        self.tracked_vars: List[str] = []
        self.var_configs: Dict[str, str] = {}  # Axis layout mapping e.g. {"A": "0211"}
        self.auto_mode: bool = False
        self.scalar_histories: Dict[str, List[float]] = {}
        self.step = 0
        self.generate_pdfs = False
        self.pdf_dir = pdf_dir
        self.agent_provider: Optional[str] = None
        self.agent_key: Optional[str] = None
        self.agent_history: List[Dict[str, Any]] = []
        self.code_text: Optional[str] = None
        self.script_path: Optional[str] = None

        # Backup of the file's contents from immediately before the most
        # recent agent-applied code fix, so that fix can be reverted.
        self._pending_revert_path: Optional[str] = None
        self._pending_revert_backup: Optional[str] = None

        # Matrix/tensor probing is intentionally decoupled from the training loop.
        # statistics() on GPU arrays can force a device->host synchronization, so
        # NEVER run it for tagged matrices on every training step.
        self._matrix_cache: Dict[str, Dict[str, Any]] = {}
        self._matrix_cached_vars: set[str] = set()
        self._last_matrix_probe: float = 0.0
        self.matrix_probe_interval: float = 1.0

        # Interactive Mode & Interrupt Handling
        self.continuous = False
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)


    def _sigint_handler(self, sig, frame):
        """Intercept Ctrl+C during continuous execution to drop into the debugger."""
        if self.continuous:
            self.continuous = False
            print("\n[Pulse] Intercepted Ctrl+C. Pausing at next step...")
        else:
            # If already paused and user hits Ctrl+C again, restore original behavior and exit
            if self.original_sigint:
                signal.signal(signal.SIGINT, self.original_sigint)
            raise KeyboardInterrupt

    def print_banner(self) -> None:
        backends = available_backends()
        print("\n" + "=" * 50)
        print("                PULSE CLI EDITION                ")
        print("=" * 50)
        print("Detected backend(s):")
        for name, active in backends.items():
            status = "✓" if active else "✗"
            print(f"  {status} {name}")
        print("-" * 50 + "\n")

    def set_code_text(self, code_text: Optional[str], script_path: Optional[str] = None) -> None:
        self.code_text = code_text
        if script_path is not None:
            self.script_path = script_path

    def discover_variables(self) -> Dict[str, Any]:
        trackable: Dict[str, Any] = {}
        for name, val in self.watch_locals.items():
            if name.startswith("__"):
                continue
            if is_trackable(val):
                trackable[name] = val

        for name in self.discovered:
            if name not in trackable:
                trackable[name] = None

        return trackable

    def _cmd_add(self, var_name: str) -> None:
        """Add a variable to tracking.

        If `var_name` isn't an exact match for a currently-discovered
        variable, fall back to the same partial-name matching used during
        interactive setup: show the matches with numbers and let the user
        pick one.
        """
        var_name = var_name.strip()
        if not var_name:
            print("Usage: /add <variable name or partial name>")
            return

        variables = self.discover_variables()

        if var_name not in variables:
            var_list = sorted(variables.keys(), key=lambda n: (not _looks_like_loss(n), n))
            terms = [t.strip().lower() for t in var_name.split(",") if t.strip()]
            matches = [name for name in var_list if any(term in name.lower() for term in terms)]

            if not matches:
                print(f"[Pulse CLI] No variables match '{var_name}'.")
                return

            if len(matches) > 1:
                print("\nMatching variables:")
                for idx, name in enumerate(matches, 1):
                    val = variables.get(name)
                    star = "  ★ loss?" if _looks_like_loss(name) else ""
                    if val is None:
                        info = "[not run yet]"
                    else:
                        try:
                            d = describe_tensor(val)
                            info = f"[{d.backend} {d.kind} {d.shape}]"
                        except Exception:
                            info = "[trackable]"
                    print(f"  {idx}) {name} {info}{star}")
                choice = input("Select a number to add (Enter to cancel) > ").strip()
                if not choice:
                    print("[Pulse CLI] Add cancelled.")
                    return
                if not choice.isdigit() or not (0 < int(choice) <= len(matches)):
                    print("[Pulse CLI] Invalid selection.")
                    return
                var_name = matches[int(choice) - 1]
            else:
                var_name = matches[0]

        if var_name not in self.tracked_vars:
            self.tracked_vars.append(var_name)
            self._matrix_cached_vars.discard(var_name)
            print(f"✓ Added '{var_name}' to tracking.")
        else:
            print(f"'{var_name}' is already tracked.")
        self._cmd_edit(var_name)

    def _cmd_edit(self, var_name: str) -> None:
        if var_name not in self.tracked_vars:
            print(f"'{var_name}' is not tracked. Use /add first.")
            return

        shape = None
        if var_name in self.watch_locals and is_trackable(self.watch_locals[var_name]):
            try:
                shape = describe_tensor(self.watch_locals[var_name]).shape
            except Exception:
                pass
        if shape is None and var_name in self.discovered and self.discovered[var_name]:
            shape = self.discovered[var_name]

        if not shape or not isinstance(shape, tuple):
            print(f"Cannot configure axes for '{var_name}' yet (shape unknown). It will be tracked globally.")
            return

        shape_str = " ".join(str(s) for s in shape)
        print(f"Shape: {shape_str}")
        config = input("Axes map (0=Fix at 0, 1=Show/Keep, 2=Iterate) > ").strip()
        if config:
            self.var_configs[var_name] = config
            print(f"✓ Saved config '{config}' for '{var_name}'.")

    def _cmd_delete_pdfs(self, var_name: str) -> None:
        """Delete saved heatmap PDF snapshots for a variable, or all of them.

        Snapshots are saved under `self.pdf_dir/<safe_variable_name>/`, using
        the same name-sanitization as `update()` uses when writing them.
        """
        var_name = var_name.strip()
        if not var_name:
            print("Usage: /delete <variable name> or /delete all")
            return

        if var_name.lower() == "all":
            if not os.path.isdir(self.pdf_dir):
                print(f"[Pulse CLI] No PDF output directory found at '{self.pdf_dir}'.")
                return
            confirm = input(
                f"Delete ALL heatmap snapshots under '{self.pdf_dir}'? This cannot be undone. (y/n) > "
            ).strip().lower()
            if confirm not in ("y", "yes"):
                print("[Pulse CLI] Delete cancelled.")
                return
            try:
                shutil.rmtree(self.pdf_dir)
                print(f"✓ Deleted all heatmap snapshots under '{self.pdf_dir}'.")
            except Exception as exc:
                print(f"[Pulse CLI] ⚠ Failed to delete '{self.pdf_dir}': {exc}")
            return

        safe_name = var_name.replace("[", "_").replace("]", "").replace(",", "_")
        target_dir = os.path.join(self.pdf_dir, safe_name)
        if not os.path.isdir(target_dir):
            print(f"[Pulse CLI] No heatmap snapshots found for '{var_name}' (looked in '{target_dir}').")
            return

        confirm = input(
            f"Delete all heatmap snapshots for '{var_name}' in '{target_dir}'? (y/n) > "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            print("[Pulse CLI] Delete cancelled.")
            return
        try:
            shutil.rmtree(target_dir)
            print(f"✓ Deleted heatmap snapshots for '{var_name}'.")
        except Exception as exc:
            print(f"[Pulse CLI] ⚠ Failed to delete '{target_dir}': {exc}")

    def interactive_setup(self) -> None:
        """Interactive CLI setup."""
        variables = self.discover_variables()
        if not variables:
            print("[Pulse CLI] No trackable variables found in scope.")
            return

        var_list = sorted(variables.keys(), key=lambda n: (not _looks_like_loss(n), n))

        def show_matches(matches: List[str]) -> None:
            print("\nMatching variables:")
            for idx, name in enumerate(matches, 1):
                val = variables[name]
                star = "  ★ loss?" if _looks_like_loss(name) else ""
                if val is None:
                    info = "[not run yet]"
                else:
                    try:
                        d = describe_tensor(val)
                        info = f"[{d.backend} {d.kind} {d.shape}]"
                    except Exception:
                        info = "[trackable]"
                print(f"  {idx}) {name} {info}{star}")

        print("Detected variables:")
        show_matches(var_list)
        print(
            "\nSelect variables:"
            "\n  • numbers: 1,2,5"
            "\n  • partial name + Enter: matri   (finds matrix, matrices, matrix_...)"
            "\n  • exact/partial names: matrix,matrix2"
            "\n  • all = track everything currently known"
            "\n  • auto = track everything, including variables appearing later"
            "\n  • blank = skip tracking"
        )

        while True:
            choice = input("\nPulse variables > ").strip()
            if not choice:
                return

            lower = choice.lower()
            if lower.startswith("/add "):
                self._cmd_add(choice[5:].strip())
                continue
            if lower.startswith("/edit "):
                self._cmd_edit(choice[6:].strip())
                continue
            if lower.startswith("/delete "):
                self._cmd_delete_pdfs(choice[8:].strip())
                continue

            if lower == "auto":
                self.auto_mode = True
                self.tracked_vars = [name for name in var_list if variables[name] is not None]
                started = ", ".join(self.tracked_vars) if self.tracked_vars else "(none yet)"
                print(f"\n✓ Auto-tracking every trackable variable (already in scope: {started})")
                break

            if lower == "all":
                self.tracked_vars = var_list[:]
                print(f"\n✓ Tracking variables: {', '.join(self.tracked_vars)}")
                break

            if all(part.strip().isdigit() for part in choice.split(",")):
                indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
                selected = [var_list[i - 1] for i in indices if 0 < i <= len(var_list)]
                if selected:
                    self.tracked_vars = list(dict.fromkeys(selected))
                    print(f"\n✓ Tracking variables: {', '.join(self.tracked_vars)}")
                    break
                print("[Pulse CLI] No valid variable numbers.")
                continue

            terms = [x.strip().lower() for x in choice.split(",") if x.strip()]
            matches = [
                name for name in var_list
                if any(term in name.lower() for term in terms)
            ]

            if not matches:
                print(f"[Pulse CLI] No variables match '{choice}'. Try another partial name.")
                continue

            show_matches(matches)
            if len(matches) == 1:
                confirm = input(f"Track '{matches[0]}'? (y/n) > ").strip().lower()
                if confirm in ("y", "yes"):
                    self.tracked_vars = [matches[0]]
                    print(f"\n✓ Tracking variables: {matches[0]}")
                    break
                continue

            print(
                "\nEnter the matching numbers to track, another partial name to narrow it, "
                "or 'all' to track every match."
            )
            subchoice = input(f"Match selection [{choice}] > ").strip()

            if subchoice.lower() == "all":
                self.tracked_vars = matches
                print(f"\n✓ Tracking variables: {', '.join(self.tracked_vars)}")
                break

            if all(part.strip().isdigit() for part in subchoice.split(",")):
                nums = [int(x.strip()) for x in subchoice.split(",") if x.strip()]
                selected = [matches[i - 1] for i in nums if 0 < i <= len(matches)]
                if selected:
                    self.tracked_vars = list(dict.fromkeys(selected))
                    print(f"\n✓ Tracking variables: {', '.join(self.tracked_vars)}")
                    break

            narrowed = [
                name for name in matches
                if subchoice.lower() in name.lower()
            ]
            if narrowed:
                self.tracked_vars = narrowed
                print(f"\n✓ Tracking variables: {', '.join(self.tracked_vars)}")
                break

            print("[Pulse CLI] Invalid selection.")

        has_matrix_like = any(
            variables.get(name) is not None
            and describe_tensor(variables[name]).kind in ("vector", "matrix", "tensor")
            for name in self.tracked_vars
        )
        if has_matrix_like:
            print(
                "\nNo heatmaps are shown in this CLI view -- matrices/tensors are just "
                "tagged (their stats printed each step)."
            )
            resp = input(
                f"Save labeled PDF snapshots to '{self.pdf_dir}/<variable>/' each step? (y/n): "
            ).strip().lower()
            self.generate_pdfs = resp in ("y", "yes")
            if self.generate_pdfs:
                print(f"[Pulse CLI] PDF snapshots enabled -> {self.pdf_dir}/<variable>/stepNNNNNN.pdf\n")
            else:
                print("[Pulse CLI] PDF snapshots disabled -- only stats will be printed.\n")
        else:
            print()

        self._agent_setup()

    def _select_agent_provider_and_key(self, initial: bool = False) -> bool:
        """Prompt the user to pick an AI provider and API key.

        When `initial` is True (first-time setup), pressing Enter skips agent
        setup entirely and leaves the agent disabled. When False (switching
        mid-run), pressing Enter cancels the switch and keeps whatever
        agent/key is already active.

        Returns True if the active agent/key changed as a result of this call.
        """
        names = list(PROVIDERS.keys())

        print("\nSelect an agent/provider:")
        for i, name in enumerate(names, 1):
            marker = "  (current)" if name == self.agent_provider else ""
            print(f"  {i}) {name}{marker}")

        prompt = (
            "\nAgent number (or Enter to skip) > "
            if initial
            else "\nAgent number (or Enter to cancel) > "
        )

        while True:
            raw = input(prompt).strip()
            if not raw:
                if initial:
                    print("[Pulse CLI] AI agent disabled for this run.")
                else:
                    print("[Pulse CLI] Agent switch cancelled.")
                return False
            if raw.isdigit() and 1 <= int(raw) <= len(names):
                chosen = names[int(raw) - 1]
                break
            matches = [n for n in names if raw.lower() in n.lower()]
            if len(matches) == 1:
                chosen = matches[0]
                break
            print("[Pulse CLI] Pick a valid agent number or provider name.")

        info = PROVIDERS[chosen]
        env_var = info["env_key"]
        existing = os.environ.get(env_var, "").strip()

        print(f"\n✓ Agent selected: {chosen}")
        if existing:
            use_existing = input(
                f"An {env_var} is already set. Use it? (Y/n) > "
            ).strip().lower()
            if use_existing in ("", "y", "yes"):
                key = existing
            else:
                key = getpass.getpass("API key > ").strip()
        else:
            key = getpass.getpass("API key > ").strip()

        if not key:
            print("[Pulse CLI] No API key entered. Agent unchanged.")
            if initial:
                self.agent_provider = None
            return False

        self.agent_provider = chosen
        self.agent_key = key
        os.environ[env_var] = key
        # Switching providers mid-conversation would send one provider's turns
        # to another; start a fresh history so context isn't mixed across models.
        self.agent_history = []

        if initial:
            print(f"✓ API key accepted for {self.agent_provider}.")
        else:
            print(f"✓ Switched to {self.agent_provider}. Conversation history reset for the new agent.")
        return True

    def _agent_setup(self) -> None:
        """Choose the AI provider first, then obtain its API key, then enter Q&A."""
        print("\n" + "=" * 60)
        print("                    PULSE AI AGENT")
        print("=" * 60)

        if not self._select_agent_provider_and_key(initial=True):
            return

        print(
            "\nAgent ready. Ask questions now. Commands:\n"
            "  /vars     show discovered variables\n"
            "  /tracked  show tracked variables\n"
            "  /add      add a new variable to track (e.g., /add my_matrix)\n"
            "  /edit     edit axis configuration (e.g., /edit my_matrix)\n"
            "  /delete   delete saved heatmap PDFs (e.g., /delete my_matrix, /delete all)\n"
            "  /agent    switch AI provider/API key\n"
            "  /code     include the training code in the next question\n"
            "  /exit     finish setup and start training\n"
        )

        while True:
            try:
                question = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question.lower() in ("/exit", "exit", "quit", "q"):
                break

            # Interactive Setup Commands
            if question.lower().startswith("/add "):
                self._cmd_add(question[5:].strip())
                continue
            if question.lower().startswith("/edit "):
                self._cmd_edit(question[6:].strip())
                continue
            if question.lower().startswith("/delete "):
                self._cmd_delete_pdfs(question[8:].strip())
                continue
            if question.lower() == "/agent":
                self._select_agent_provider_and_key(initial=False)
                continue
            if question.lower() == "/vars":
                self._print_variable_summary()
                continue
            if question.lower() == "/tracked":
                print("Tracked:", ", ".join(self.tracked_vars) if self.tracked_vars else "(none)")
                continue

            include_code = False
            if question.lower() == "/code":
                include_code = True
                question = input("Code question > ").strip()
                if not question:
                    continue

            print("Pulse > ", end="", flush=True)
            answer = self.ask_agent(question, include_code=include_code)
            print(answer)

        print("\n✓ AI setup complete. Starting training.")
        print("  Press 'Ctrl+C' at any time while training runs to pause and drop into the agent prompt.")

    def _print_variable_summary(self) -> None:
        variables = self.discover_variables()
        print("\nDiscovered variables:")
        for name in sorted(variables):
            val = variables[name]
            if val is None:
                print(f"  • {name}: [not run yet]")
            else:
                try:
                    d = describe_tensor(val)
                    print(f"  • {name}: {d.backend} {d.kind} {d.shape}")
                except Exception:
                    print(f"  • {name}: trackable")

    def _yield_slices(self, var_name: str, val: Any):
        """
        Dynamically slices axes according to layout string (e.g. '0211'):
          '0': fixed index 0
          '1': show dimension
          '2': iterate through values
        """
        if not is_trackable(val):
            return
        try:
            shape = describe_tensor(val).shape
        except Exception:
            yield var_name, val
            return

        config = self.var_configs.get(var_name, "")
        if not config or not isinstance(shape, tuple):
            yield var_name, val
            return

        iter_dims = []
        for i, c in enumerate(config):
            if i >= len(shape):
                break
            if c == '2':
                iter_dims.append((i, range(shape[i])))

        if not iter_dims:
            slices = []
            for i, c in enumerate(config):
                if i >= len(shape):
                    break
                if c == '0':
                    slices.append(0)
                else:
                    slices.append(slice(None))
            while len(slices) < len(shape):
                slices.append(slice(None))
            try:
                yield var_name, val[tuple(slices)]
            except Exception:
                yield var_name, val
            return

        iter_indices = [x[1] for x in iter_dims]
        for combo in itertools.product(*iter_indices):
            slices = []
            combo_idx = 0
            for i, c in enumerate(config):
                if i >= len(shape):
                    break
                if c == '0':
                    slices.append(0)
                elif c == '2':
                    slices.append(combo[combo_idx])
                    combo_idx += 1
                else:
                    slices.append(slice(None))
            while len(slices) < len(shape):
                slices.append(slice(None))

            suffix = []
            combo_idx = 0
            for i, c in enumerate(config):
                if i >= len(shape):
                    break
                if c == '2':
                    suffix.append(str(combo[combo_idx]))
                    combo_idx += 1

            sub_name = f"{var_name}[{','.join(suffix)}]"
            try:
                yield sub_name, val[tuple(slices)]
            except Exception:
                pass

    def _build_agent_context(self, include_code: bool = False) -> str:
        variables = self.discover_variables()
        lines = ["Current Pulse variable state:"]

        for name in sorted(variables):
            val = variables[name]
            if val is None:
                lines.append(f"- {name}: not run yet")
                continue
            try:
                for sub_name, s_val in self._yield_slices(name, val):
                    s = statistics(s_val)
                    if s.get("kind") == "scalar":
                        lines.append(
                            f"- {sub_name}: scalar backend={s.get('backend')} "
                            f"value={float(to_numpy(s_val).reshape(-1)[0])} "
                            f"nan={s.get('nan')} inf={s.get('inf')}"
                        )
                    else:
                        lines.append(
                            f"- {sub_name}: shape={s.get('shape')} backend={s.get('backend')} "
                            f"kind={s.get('kind')} min={s.get('min')} max={s.get('max')} "
                            f"mean={s.get('mean')} std={s.get('std')} "
                            f"nan={s.get('nan')} inf={s.get('inf')}"
                        )
            except Exception as exc:
                lines.append(f"- {name}: unable to read stats ({exc})")

        cfg_strs = [f"{v}({self.var_configs[v]})" if v in self.var_configs else v for v in self.tracked_vars]
        lines.append(
            "\nTracked variables: "
            + (", ".join(cfg_strs) if cfg_strs else "(none)")
        )

        if include_code and self.code_text:
            numbered = "\n".join(
                f"{i+1:>4} | {line}" for i, line in enumerate(self.code_text.splitlines())
            )
            lines.append(f"\nTraining code (line-numbered):\n```\n{numbered}\n```")

        return "\n".join(lines)

    def ask_agent(self, question: str, include_code: bool = False) -> str:
        if not self.agent_provider or not self.agent_key:
            return "(AI agent is not enabled. Run setup again or set the API key.)"

        context = self._build_agent_context(include_code=include_code)
        user_content = f"{context}\n\nQuestion: {question}"
        self.agent_history.append({"role": "user", "content": user_content})

        try:
            response = litellm.completion(
                model=PROVIDERS[self.agent_provider]["model"],
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.agent_history[-10:],
                max_tokens=20000,
                timeout=240.0,
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            answer = f"(request failed: {exc})"

        self.agent_history.append({"role": "assistant", "content": answer})

        fix = self._parse_code_fix(answer)
        if fix is not None:
            return self._apply_code_fix(fix)
        return answer

    @staticmethod
    def _parse_code_fix(answer: str) -> Optional[Dict[str, Any]]:
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
        if not isinstance(old, list) or not isinstance(new, list):
            return None
        if not old or len(old) != len(new):
            return None
        if not all(isinstance(x, str) for x in old) or not all(isinstance(x, str) for x in new):
            return None

        return {"old": old, "new": new, "explanation": explanation if isinstance(explanation, str) else ""}

    def _apply_code_fix(self, fix: Dict[str, Any]) -> str:
        """Apply an agent-proposed code fix to `self.script_path` on disk.

        Each old[i] must appear exactly once in the current file contents;
        snippets that don't match cleanly are skipped and reported rather
        than guessed at. On success, a single-level backup is kept so the
        user can immediately revert back to the pre-fix file.
        """
        if not self.script_path:
            return (
                "[Pulse CLI] The agent proposed a code fix, but no script file path is known, "
                "so it can't be written to disk.\n\nExplanation: " + (fix["explanation"] or "(none given)")
            )

        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                original_content = f.read()
        except OSError as exc:
            return f"[Pulse CLI] ⚠ Could not read '{self.script_path}' to apply the fix: {exc}"

        content = original_content
        applied, skipped = [], []
        for old, new in zip(fix["old"], fix["new"]):
            count = content.count(old)
            if count == 1:
                content = content.replace(old, new, 1)
                applied.append((old, new))
            elif count == 0:
                skipped.append((old, "no exact match found in the file"))
            else:
                skipped.append((old, f"matched {count} times (ambiguous), skipped for safety"))

        lines = ["[Pulse CLI] Code fix"]
        if fix["explanation"]:
            lines.append(f"Explanation: {fix['explanation']}")

        if not applied:
            lines.append("\n⚠ No changes were applied -- none of the proposed snippets matched the file cleanly:")
            for old, reason in skipped:
                lines.append(f"  - {reason}: {old.splitlines()[0][:80]}...")
            return "\n".join(lines)

        try:
            with open(self.script_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            return f"[Pulse CLI] ⚠ Failed to write changes to '{self.script_path}': {exc}"

        self.code_text = content
        self._pending_revert_path = self.script_path
        self._pending_revert_backup = original_content

        lines.append(f"\n✓ Applied {len(applied)} change(s) to '{self.script_path}':")
        for old, new in applied:
            lines.append(f"  - replaced:\n      {old.splitlines()[0][:80]}...\n    with:\n      {new.splitlines()[0][:80]}...")
        if skipped:
            lines.append(f"\n⚠ Skipped {len(skipped)} proposed change(s) that didn't match cleanly:")
            for old, reason in skipped:
                lines.append(f"  - {reason}: {old.splitlines()[0][:80]}...")

        print("\n".join(lines))

        revert = input("\nRevert this change and restore the file to how it was before? (y/n) > ").strip().lower()
        if revert in ("y", "yes"):
            try:
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                self.code_text = original_content
                self._pending_revert_path = None
                self._pending_revert_backup = None
                return f"✓ Reverted '{self.script_path}' to its state before this fix."
            except OSError as exc:
                return f"[Pulse CLI] ⚠ Failed to revert '{self.script_path}': {exc}"

        return "✓ Change kept."

    def update(self, step: Optional[int] = None, generate_pdfs: Optional[bool] = None) -> None:
        """Called at every training step/checkpoint.

        Scalars are cheap and are sampled every update. Matrix/tensor statistics
        are expensive (especially on GPU), so they are probed at most once per
        ``matrix_probe_interval`` seconds. Between probes, Pulse does not even
        slice or call statistics() on matrices; it only uses the cached result.
        """

        if step is not None:
            self.step = step
        else:
            self.step += 1

        want_pdfs = self.generate_pdfs if generate_pdfs is None else generate_pdfs
        now = time.monotonic()

        # First call probes immediately. After that, matrix inspection is capped
        # by wall-clock time rather than training-step count.
        probe_matrices = (
            not self._matrix_cache
            or (now - self._last_matrix_probe) >= self.matrix_probe_interval
            or any(v not in self._matrix_cached_vars for v in self.tracked_vars)
        )

        scalar_lines: List[tuple[str, float]] = []
        matrix_lines: List[tuple[str, Dict[str, Any], Any, bool]] = []
        any_scalar_changed = False

        # Fast path: discover/process scalars every step. Do NOT call
        # _yield_slices() for matrices unless this is an actual matrix probe.
        for var_name in self.tracked_vars:
            if var_name not in self.watch_locals:
                continue

            orig_val = self.watch_locals[var_name]
            if not is_trackable(orig_val):
                continue

            try:
                kind = describe_tensor(orig_val).kind
            except Exception:
                kind = None

            if kind == "scalar":
                stats = statistics(orig_val)
                scalar_val = float(to_numpy(orig_val).reshape(-1)[0])
                hist = self.scalar_histories.setdefault(var_name, [])
                changed = not hist or hist[-1] != scalar_val
                if changed:
                    hist.append(scalar_val)
                    any_scalar_changed = True
                scalar_lines.append((var_name, scalar_val))
                continue

            # Matrix/tensor path. No slicing, no statistics(), and no GPU->CPU
            # copy at all unless the wall-clock probe is due.
            if not probe_matrices:
                continue

            for sub_name, val in self._yield_slices(var_name, orig_val):
                try:
                    stats = statistics(val)
                except Exception:
                    continue

                if stats.get("kind") == "scalar":
                    scalar_val = float(to_numpy(val).reshape(-1)[0])
                    hist = self.scalar_histories.setdefault(sub_name, [])
                    changed = not hist or hist[-1] != scalar_val
                    if changed:
                        hist.append(scalar_val)
                        any_scalar_changed = True
                    scalar_lines.append((sub_name, scalar_val))
                else:
                    self._matrix_cache[sub_name] = {
                        "base_name": var_name,
                        "stats": stats,
                    }
                    matrix_lines.append((sub_name, stats, val, True))

            self._matrix_cached_vars.add(var_name)

        if probe_matrices:
            self._last_matrix_probe = now

        # Don't redraw the entire terminal if absolutely nothing visible changed.
        # A matrix probe always redraws because its snapshot may have changed.
        should_redraw = probe_matrices or any_scalar_changed or not self.scalar_histories

        if should_redraw:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            print(f"--- Pulse Live Debugger | Step {self.step} ---")

            for sub_name, scalar_val in scalar_lines:
                hist = self.scalar_histories.get(sub_name, [])
                print(f"  • {sub_name}: {scalar_val:.6g}")
                self._print_ascii_chart(sub_name, hist)

            if probe_matrices:
                # Print matrix statistics ONLY when they were freshly measured.
                # Cached matrices are deliberately silent between probes.
                for sub_name, stats, val, fresh in matrix_lines:
                    flag = ""
                    if stats.get("nan") or stats.get("inf"):
                        flag = f"  ⚠ nan={stats.get('nan')} inf={stats.get('inf')}"
                    print(
                        f"  • Tagging '{sub_name}' "
                        f"[{stats.get('backend')} {stats.get('kind')} {stats.get('shape')}] "
                        f"| mean={stats.get('mean'):.4f} min={stats.get('min'):.4f} "
                        f"max={stats.get('max'):.4f}{flag}"
                    )

                    if want_pdfs and val is not None:
                        safe_name = sub_name.replace("[", "_").replace("]", "").replace(",", "_")
                        try:
                            pdf_path = generate_heatmap_pdf(
                                safe_name, val, self.step, output_dir=self.pdf_dir
                            )
                            print(f"    ↳ saved snapshot: {pdf_path}")
                        except Exception as exc:
                            print(f"    ↳ ⚠ failed to save PDF snapshot for '{sub_name}': {exc}")

                # If the cache is already populated, show a tiny status line rather
                # than reprinting every cached matrix on every training iteration.
                if self._matrix_cache:
                    print(
                        f"  [matrices cached: {len(self._matrix_cache)} | "
                        f"next probe ≤ {self.matrix_probe_interval:g}s]"
                    )

        if self.continuous:
            return

        # Interactive Training Loop Prompt
        while True:
            try:
                cmd = input(
                    "\nPulse [Enter=step, /c=continuous, /add <var>, /edit <var>, "
                    "/delete <var>, /agent, or ask AI] > "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting Pulse...")
                if self.original_sigint and callable(self.original_sigint):
                    signal.signal(signal.SIGINT, self.original_sigint)
                raise KeyboardInterrupt

            if not cmd:
                break

            if cmd.lower() in ("/c", "/continue", "c"):
                self.continuous = True
                break

            if cmd.lower().startswith("/add "):
                self._cmd_add(cmd[5:].strip())
                continue

            if cmd.lower().startswith("/edit "):
                self._cmd_edit(cmd[6:].strip())
                continue

            if cmd.lower().startswith("/delete "):
                self._cmd_delete_pdfs(cmd[8:].strip())
                continue

            if cmd.lower() == "/agent":
                self._select_agent_provider_and_key(initial=False)
                continue

            if cmd.lower() == "/vars":
                self._print_variable_summary()
                continue

            if cmd.lower() == "/tracked":
                cfg_strs = [
                    f"{v}({self.var_configs[v]})" if v in self.var_configs else v
                    for v in self.tracked_vars
                ]
                print("Tracked:", ", ".join(cfg_strs) if cfg_strs else "(none)")
                continue

            print("Pulse AI > ", end="", flush=True)
            answer = self.ask_agent(cmd, include_code=False)
            print(answer)

    def _print_ascii_chart(
        self,
        name: str,
        history: List[float],
        height: int = 8,
        width: int = 64,
    ) -> None:
        """Render a compact line-style loss/metric graph.

        The X axis advances only when the scalar value changes, so repeated
        training-loop calls do not create fake horizontal steps.
        """
        if not history:
            return

        data = history[-width:]
        if len(data) == 1:
            print(f"    {data[0]:>10.5g} ┤ ●")
            print("              └─ step 1")
            return

        lo = min(data)
        hi = max(data)

        # Give flat/near-flat curves a useful visible range.
        if hi == lo:
            pad = max(abs(hi) * 0.02, 1e-6)
            lo -= pad
            hi += pad
        else:
            pad = (hi - lo) * 0.08
            lo -= pad
            hi += pad

        rows = height
        cols = len(data)
        grid = [[" "] * cols for _ in range(rows)]

        def y_for(v: float) -> int:
            norm = (v - lo) / (hi - lo)
            return max(0, min(rows - 1, int(round((1.0 - norm) * (rows - 1)))))

        ys = [y_for(v) for v in data]

        # Plot points and simple line segments.
        for i, y in enumerate(ys):
            grid[y][i] = "●"
            if i == 0:
                continue
            py = ys[i - 1]
            x = i - 1
            if py == y:
                grid[y][x] = "─"
            else:
                ch = "╱" if y < py else "╲"
                grid[y][x] = ch
                # Fill vertical movement without trying to interpolate a fake
                # curve; this keeps the terminal graph readable.
                step_dir = 1 if y > py else -1
                rr = py + step_dir
                while rr != y:
                    if grid[rr][x] == " ":
                        grid[rr][x] = "│"
                    rr += step_dir

        print(f"    [{name} | {len(history)} points | showing last {cols}]")

        for r in range(rows):
            value = hi - (hi - lo) * (r / (rows - 1))
            print(f"    {value:>10.5g} ┤ " + "".join(grid[r]))

        print("              └" + "─" * cols)
        print(f"               {max(1, len(history) - cols + 1):<{max(1, cols // 2)}}"
              f"{len(history)}")