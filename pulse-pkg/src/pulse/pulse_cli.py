"""
pulse_cli.py
============
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import shutil
import signal
import getpass
import itertools
import subprocess
import threading
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


def _values_equal(a, b) -> bool:
    """NaN-safe / None-safe equality for scalar history dedup.

    `nan != nan` is always True in Python, so without this a NaN (or None)
    scalar that repeats every step would get appended to history -- and
    re-rendered -- every single step instead of being deduped like any
    other repeated value.
    """
    if a is None or b is None:
        return a is b
    try:
        if a != a and b != b:  # both NaN
            return True
    except TypeError:
        pass
    return a == b

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
    "Some tracked values may show up as 'NoneType' (the variable currently holds None, or "
    "hasn't run yet) or flagged as NaN/inf-only in the chart. Treat those as data too -- a "
    "variable that is NoneType or all-NaN at a point in training is itself often the root "
    "cause (e.g. an optional metric never getting set, or a loss that has already collapsed "
    "to NaN before this step) rather than a gap to ignore.\n\n"

    "TOOLS (use inline, each on its own line, only when actually useful -- omit both if not needed):\n"
    "  CALC: <python arithmetic expression>\n"
    "    You are not reliable at exact arithmetic. Anything like an update magnitude, a ratio, "
    "or a comparison between two numbers you were given -- hand it off here instead of computing "
    "it yourself. Pulse evaluates it deterministically (only numbers, operators, and `math` "
    "module functions are available) and gives you the exact result. You may include more than "
    "one CALC: line.\n"
    "  PROMOTE: <comma-separated variable names>\n"
    "    Some tracked matrices/tensors are in lightweight 'lotrack' mode (intermittent sampling, "
    "stats only, no PDFs) -- see each variable's [track]/[lotrack] tag in the state you were "
    "given. If one of them looks like it needs a closer look, name it here and Pulse will switch "
    "it to full tracking. Only promote variables that are currently [lotrack]; don't bother for "
    "ones already [track].\n"
    "  Put CALC:/PROMOTE: lines anywhere in your Reasoning, not in the Diagnosis or Fix.\n\n"

    "CODE FIXES:\n"
    "If, and only if, the user explicitly asks you to fix, edit, patch, or change the code "
    "(not just diagnose it), respond with ONLY a single JSON object and nothing else -- no prose "
    "before or after it, no markdown code fences. The JSON object must have exactly these fields:\n"
    "  old: a list of code snippets to find, each copied EXACTLY from the line-numbered code "
    "shown to you, including original indentation and whitespace, but WITHOUT the line-number "
    "prefix ('  12 | ') itself.\n"
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


import math as _math_module


def _safe_eval_math(expr: str):
    """Evaluate a plain arithmetic/math expression deterministically -- LLMs
    are unreliable at exact arithmetic, so the agent can hand off anything
    like update magnitudes or ratios here instead of eyeballing it. Only
    numbers, operators, and `math` module names are reachable; no builtins,
    no attribute access beyond that, so this is safe to eval() directly.
    """
    allowed_names = {k: v for k, v in vars(_math_module).items() if not k.startswith("_")}
    allowed_names["math"] = _math_module
    try:
        return eval(expr, {"__builtins__": {}}, allowed_names)  # noqa: S307 -- restricted namespace above
    except Exception as exc:
        return f"(calc error: {exc})"


class _Spinner:
    """Minimal terminal spinner (/-\\|) shown while an agent stage is running.

    Used as a context manager: `with _Spinner("Diagnosing"): ...`. Prints
    nothing else -- callers are responsible for printing the stage's result
    once the spinner stops.
    """

    _FRAMES = "/-\\|"

    def __init__(self, label: str):
        self.label = label
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        for frame in itertools.cycle(self._FRAMES):
            if self._stop_evt.is_set():
                break
            sys.stdout.write(f"\r{self.label}... {frame}")
            sys.stdout.flush()
            time.sleep(0.12)
        sys.stdout.write("\r" + " " * (len(self.label) + 6) + "\r")
        sys.stdout.flush()

    def __enter__(self) -> "_Spinner":
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join()


# Adaptive multi-pass agent pipeline (see PulseCLI.ask_agent). Instead of a
# fixed 3-call sequence, the number of passes actually run adapts to what's
# being asked and what comes back:
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
# Each call is kept small/cheap (small max_tokens) and streams to the
# terminal as soon as it's ready, same spirit as the old fixed pipeline.
_PASS1_LOCATE = (
    "PASS 1 -- LOCATE: Read through everything you were given (stats, code, history) and identify "
    "the specific region(s) where the problem likely originates -- file/line numbers, variable "
    "names, or code sections. Respond with ONLY a short bullet list of the suspect location(s). "
    "No diagnosis, no fix yet."
)
_PASS2_ANALYZE_TMPL = (
    "Suspect region(s) from your first read:\n{regions}\n\n"
    "PASS 2 -- ANALYZE: Take a focused second look at just those regions. Give the Diagnosis (one "
    "sentence, the specific root cause) and the Reasoning behind it (grounded in the actual "
    "numbers/code you were given, with real math, referencing line numbers). Do not implement the "
    "fix yet."
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
        # Per-variable tracking depth: "track" (full stats + PDFs if enabled,
        # probed every matrix_probe_interval) or "lotrack" (intermittent,
        # stats-only, never generates a PDF -- the default for matrices/
        # tensors so tracking "everything" stays cheap). Scalars are always
        # effectively full-track regardless of what's stored here. A
        # variable not present in tracked_vars is simply not tracked at all.
        self.var_states: Dict[str, str] = {}
        self.auto_mode: bool = False
        self.scalar_histories: Dict[str, List[float]] = {}
        self.step = 0
        # Global step only advances when the loss/metric scalar (the first
        # tracked var that looks like a loss) actually changes value -- see
        # update(). None until a loss-like var is seen at least once.
        self._last_loss_value: Optional[float] = None
        self.generate_pdfs = False
        self.pdf_dir = pdf_dir
        self.agent_provider: Optional[str] = None
        self.agent_key: Optional[str] = None
        self.agent_history: List[Dict[str, Any]] = []
        self.code_text: Optional[str] = None
        self.script_path: Optional[str] = None
        # {path: text} for other local project files this script imports
        # (a modularized project's model.py/utils.py/etc.) -- set by
        # pulse.py's auto_track() so /code and "fix it" can see and edit
        # code that isn't in the entry script at all.
        self.extra_files: Dict[str, str] = {}
        self._label_for_path: Dict[str, str] = {}
        self._path_for_label: Dict[str, str] = {}
        # A formatted traceback if the dry run passed to auto_track() raised
        # -- surfaced once the agent is set up so a bug blocking training
        # from even starting can still get diagnosed/fixed.
        self.pending_startup_error: Optional[str] = None

        # Backups of every file touched by the most recent agent-applied
        # code fix (path -> original content). Kept around in case a
        # revert is ever needed manually; the normal flow now auto-restarts
        # after a fix instead of asking to revert (see _restart_process).
        self._pending_revert_backups: Dict[str, str] = {}
        # Set by _apply_code_fix whenever a fix is actually written to
        # disk during the current top-level ask_agent() call. Checked once,
        # at the end of that call, to decide whether to restart the process.
        self._fix_applied_this_turn: bool = False

        # Matrix/tensor probing is intentionally decoupled from the training loop.
        # statistics() on GPU arrays can force a device->host synchronization, so
        # NEVER run it for tagged matrices on every training step. "track" and
        # "lotrack" variables are probed on separate, independent cadences.
        self._matrix_cache: Dict[str, Dict[str, Any]] = {}
        self._matrix_cached_vars: set[str] = set()
        self._last_matrix_probe: float = 0.0
        self.matrix_probe_interval: float = 1.0
        self._last_lotrack_probe: float = 0.0
        self.lotrack_probe_interval: float = 5.0

        # Auto-intervention: watch tracked values for signs training is
        # going bad (a scalar going non-finite, or a loss-like scalar
        # spiking well above its recent range) and, if so, automatically
        # pause (even out of continuous mode) and ask the agent to diagnose
        # -- and if it can, fix -- it, without waiting for the user to
        # notice and ask manually. On by default; toggle with /autofix.
        self.auto_intervene: bool = True
        self.explosion_multiplier: float = 5.0
        self._last_intervention_signature: Optional[str] = None
        # /code is on by default -- every manually-asked question includes
        # the training code (and any cross-file context) unless turned off.
        self.include_code_default: bool = True

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
        pass  # minimal UI: no banner -- setup only asks for a provider/API key below

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

    def _default_state_for(self, name: str, val: Any) -> str:
        """Loss-like scalars are always cheap to compute so they default to
        full 'track'. Everything else (matrices/tensors, or anything whose
        shape isn't known yet) defaults to lightweight 'lotrack' -- this is
        what keeps "track everything" affordable. Once a not-yet-resolved
        variable turns out to actually be a scalar, callers upgrade it to
        'track' automatically (see the runtime tracer in pulse.py).
        """
        if _looks_like_loss(name):
            return "track"
        if val is not None:
            try:
                if describe_tensor(val).kind == "scalar":
                    return "track"
            except Exception:
                pass
        return "lotrack"

    def _state_of(self, name: str) -> str:
        return self.var_states.get(name, "track")

    def _set_var_state(self, var_name: str, new_state: str, quiet: bool = False) -> Optional[str]:
        """Shared implementation behind /track and /lotrack (and the agent's
        PROMOTE directive, which calls this with quiet=True). Returns the
        canonical variable name that was changed, or None if nothing changed
        (not tracked, ambiguous match while quiet, or user cancelled).
        """
        var_name = var_name.strip()
        if not var_name:
            if not quiet:
                print(f"Usage: /{new_state} <variable name>  (or /{new_state} all)")
            return None

        if var_name.lower() == "all":
            changed = [v for v in self.tracked_vars if self.var_states.get(v, "track") != new_state]
            for v in changed:
                self.var_states[v] = new_state
                self._matrix_cached_vars.discard(v)
            if not quiet:
                print(f"✓ Set {len(changed)} variable(s) to '{new_state}'.")
            return "all" if changed else None

        target = var_name
        if target not in self.tracked_vars:
            matches = [n for n in self.tracked_vars if var_name.lower() in n.lower()]
            if not matches:
                if not quiet:
                    print(f"[Pulse CLI] '{var_name}' is not currently tracked.")
                return None
            if len(matches) > 1:
                if quiet:
                    # Ambiguous and unattended (agent-triggered) -- don't guess.
                    return None
                print("\nMatching tracked variables:")
                for idx, name in enumerate(matches, 1):
                    print(f"  {idx}) {name}")
                choice = input("Select a number (Enter to cancel) > ").strip()
                if not choice or not choice.isdigit() or not (0 < int(choice) <= len(matches)):
                    print("[Pulse CLI] Cancelled.")
                    return None
                target = matches[int(choice) - 1]
            else:
                target = matches[0]

        self.var_states[target] = new_state
        self._matrix_cached_vars.discard(target)  # force a fresh probe under the new cadence
        if not quiet:
            print(f"✓ '{target}' is now '{new_state}'.")
        return target

    def _cmd_track(self, var_name: str, quiet: bool = False) -> Optional[str]:
        return self._set_var_state(var_name, "track", quiet=quiet)

    def _cmd_lotrack(self, var_name: str, quiet: bool = False) -> Optional[str]:
        return self._set_var_state(var_name, "lotrack", quiet=quiet)

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
            self.var_states[var_name] = self._default_state_for(var_name, variables.get(var_name))
            state_note = "  ★ loss? -> full track" if _looks_like_loss(var_name) else f"  ({self.var_states[var_name]})"
            print(f"✓ Added '{var_name}' to tracking.{state_note}")
        else:
            print(f"'{var_name}' is already tracked.")

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

    def _cmd_delete(self, var_name: str) -> None:
        """Remove a variable from tracking, or clear all tracked variables."""
        var_name = var_name.strip()
        if not var_name:
            print("Usage: /delete <variable name> or /delete all")
            return

        if var_name.lower() == "all":
            if not self.tracked_vars:
                print("[Pulse CLI] No variables are currently tracked.")
                return

            removed = list(self.tracked_vars)
            self.tracked_vars.clear()
            self.var_configs.clear()
            self.var_states.clear()
            self.scalar_histories.clear()
            self._matrix_cache.clear()
            self._matrix_cached_vars.clear()
            print(f"✓ Removed all variables from tracking: {', '.join(removed)}")
            return

        # Exact name first, then partial-name matching.
        target = var_name
        if target not in self.tracked_vars:
            matches = [
                name for name in self.tracked_vars
                if var_name.lower() in name.lower()
            ]

            if not matches:
                print(f"[Pulse CLI] '{var_name}' is not currently tracked.")
                return

            if len(matches) > 1:
                print("\nMatching tracked variables:")
                for idx, name in enumerate(matches, 1):
                    print(f"  {idx}) {name}")
                choice = input("Select a number to delete (Enter to cancel) > ").strip()
                if not choice:
                    print("[Pulse CLI] Delete cancelled.")
                    return
                if not choice.isdigit() or not (0 < int(choice) <= len(matches)):
                    print("[Pulse CLI] Invalid selection.")
                    return
                target = matches[int(choice) - 1]
            else:
                target = matches[0]

        self.tracked_vars.remove(target)
        self.var_configs.pop(target, None)
        self.var_states.pop(target, None)
        self.scalar_histories.pop(target, None)
        self._matrix_cached_vars.discard(target)

        # Remove cached sliced entries belonging to this base variable.
        for cache_name in list(self._matrix_cache):
            if (
                cache_name == target
                or cache_name.startswith(f"{target}[")
            ):
                del self._matrix_cache[cache_name]

        print(f"✓ Removed '{target}' from tracking.")

    def _cmd_delete_pdfs(self, var_name: str) -> None:
        """Delete saved heatmap PDF snapshots for a variable, or all of them."""
        var_name = var_name.strip()
        if not var_name:
            print("Usage: /deletepdf <variable name> or /deletepdf all")
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
        """Zero-config, zero-noise CLI setup: silently track everything
        Pulse can see, skip straight to picking an AI provider/API key (the
        only thing this actually needs to ask), and let training run
        without printing anything else unless something goes wrong. Power
        users can still narrow things down with /add, /delete, /track,
        /lotrack, /autofix once training is running (Ctrl+C to pause).
        """
        variables = self.discover_variables()
        if not variables:
            print("[Pulse CLI] No trackable variables found in scope.")
            return

        var_list = sorted(variables.keys(), key=lambda n: (not _looks_like_loss(n), n))

        self.auto_mode = True
        self.tracked_vars = list(var_list)
        self.var_states = {name: self._default_state_for(name, variables[name]) for name in var_list}
        self.generate_pdfs = False  # opt-in only, via /track + a future setting -- no prompt by default

        self._agent_setup()

        # Run silently by default -- no per-step prompt, no dashboard
        # printing -- unless the user explicitly interrupts (Ctrl+C) or
        # something worth flagging happens (a read error, or the
        # auto-intervention check in update() finding real trouble).
        self.continuous = True

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
        """The one thing CLI setup actually asks for: which AI provider,
        and its API key. Everything else (variable tracking, autofix,
        /code, PDF snapshots) already has a sensible default and needs no
        prompt. Training starts silently right after this.

        If this process was just restarted after a code fix (see
        _restart_process), PULSE_AUTO_PROVIDER carries the provider that
        was active before the restart -- its API key is already sitting in
        os.environ (set right before the restart happened), so both get
        auto-filled here instead of prompting the user all over again.
        """
        auto_provider = os.environ.pop("PULSE_AUTO_PROVIDER", None)
        if auto_provider and auto_provider in PROVIDERS:
            env_var = PROVIDERS[auto_provider]["env_key"]
            key = os.environ.get(env_var, "").strip()
            if key:
                self.agent_provider = auto_provider
                self.agent_key = key
                self.agent_history = []
                print(f"[Pulse] Resumed with agent {auto_provider} (auto-filled after restart).")

        if not self.agent_provider and not self._select_agent_provider_and_key(initial=True):
            return

        if self.pending_startup_error:
            print("[Pulse] Your dry run raised an exception:")
            print(self.pending_startup_error)
            question = (
                f"My script just crashed with this uncaught exception:\n{self.pending_startup_error}\n"
                "Please diagnose the root cause and, if you can, fix it."
            )
            self.ask_agent(question, include_code=True)
            self.pending_startup_error = None

    def _restart_process(self) -> None:
        """Restart the whole process so the training loop actually runs
        the fixed code -- an already-running process keeps executing the
        old code that's still sitting in memory otherwise. Called once,
        automatically, right after a code fix from the agent has been
        applied to disk (see ask_agent's PASS 3/4).

        The active provider is threaded through PULSE_AUTO_PROVIDER so the
        next run's setup auto-fills the agent and API key (already set in
        os.environ) instead of prompting the user again -- see
        _agent_setup.
        """
        if self.agent_provider:
            os.environ["PULSE_AUTO_PROVIDER"] = self.agent_provider
        print("\n[Pulse] Fix applied -- restarting the training loop to pick it up...\n")
        sys.stdout.flush()
        script_path = getattr(sys.modules.get('__main__'), '__file__', None)
                            
        script_path = os.path.abspath(script_path)
                            
        
        # Spawn the new process safely using subprocess (which handles spaces on Windows)
        subprocess.Popen([sys.executable, script_path] + sys.argv[1:])
        sys.exit(0)

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

    def _build_file_labels(self) -> None:
        """Give every file (entry script + extra project files) a short,
        unique display label -- usually just its basename -- used both in
        the code shown to the agent and later to resolve which real file a
        proposed fix's "file" field refers to.
        """
        label_for_path: Dict[str, str] = {}
        path_for_label: Dict[str, str] = {}
        used: set = set()

        def add(path: Optional[str]) -> None:
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
        for p in self.extra_files:
            add(p)

        self._label_for_path = label_for_path
        self._path_for_label = path_for_label

    def _resolve_fix_path(self, file_label: Optional[str]) -> Optional[str]:
        """Map a fix entry's optional "file" label back to a real path on
        disk, defaulting to the entry script when unset. Falls back to
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

    def _build_agent_context(self, include_code: bool = False) -> str:
        variables = self.discover_variables()
        lines = ["Current Pulse variable state:"]

        for name in sorted(variables):
            val = variables[name]
            if val is None:
                lines.append(f"- {name}: NoneType (not run yet, or currently None)")
                continue
            try:
                for sub_name, s_val in self._yield_slices(name, val):
                    if s_val is None:
                        lines.append(f"- {sub_name}: NoneType")
                        continue
                    try:
                        s = statistics(s_val)
                    except Exception as exc:
                        lines.append(f"- {sub_name}: unable to read stats ({exc})")
                        continue
                    if s.get("kind") == "scalar":
                        try:
                            # Reuse `s` (the statistics() result) instead of a second,
                            # independent to_numpy() conversion -- see update() for why.
                            scalar_val = float(s.get("mean"))
                            value_str = str(scalar_val)
                        except Exception:
                            value_str = "NoneType"
                        lines.append(
                            f"- {sub_name}: scalar backend={s.get('backend')} "
                            f"value={value_str} "
                            f"nan={s.get('nan')} inf={s.get('inf')}"
                        )
                    else:
                        state_tag = f" [{self._state_of(name)}]" if name in self.tracked_vars else ""
                        lines.append(
                            f"- {sub_name}: shape={s.get('shape')} backend={s.get('backend')} "
                            f"kind={s.get('kind')} min={s.get('min')} max={s.get('max')} "
                            f"mean={s.get('mean')} std={s.get('std')} "
                            f"nan={s.get('nan')} inf={s.get('inf')}{state_tag}"
                        )
            except Exception as exc:
                lines.append(f"- {name}: unable to read stats ({exc})")

        cfg_strs = [
            f"{v}({self.var_configs[v]})[{self._state_of(v)}]" if v in self.var_configs else f"{v}[{self._state_of(v)}]"
            for v in self.tracked_vars
        ]
        lines.append(
            "\nTracked variables: "
            + (", ".join(cfg_strs) if cfg_strs else "(none)")
            + "\n('track' = full stats every probe, PDFs if enabled. 'lotrack' = intermittent, "
            "stats-only, no PDFs ever -- most matrices/tensors default here. Use a PROMOTE: line "
            "in your Reasoning to switch a lo-tracked variable to full tracking if you need a "
            "closer look at it.)"
        )

        if include_code and self.code_text:
            self._build_file_labels()
            entry_label = self._label_for_path.get(self.script_path, "main_script.py")
            numbered = "\n".join(
                f"{i+1:>4} | {line}" for i, line in enumerate(self.code_text.splitlines())
            )
            lines.append(f"\n=== {entry_label} (main script, line-numbered) ===\n```\n{numbered}\n```")

            if self.extra_files:
                lines.append(
                    "\nThis project is modularized -- other local files it imports are included "
                    "below, each line-numbered under its own header. When proposing a code fix, "
                    "set each fix's \"file\" to the exact header shown here (e.g. \"model.py\") so "
                    f"Pulse edits the right file. Omit \"file\" to default to {entry_label}."
                )
                for path, text in self.extra_files.items():
                    label = self._label_for_path.get(path, os.path.basename(path))
                    numbered = "\n".join(
                        f"{i+1:>4} | {line}" for i, line in enumerate(text.splitlines())
                    )
                    lines.append(f"\n=== {label} ===\n```\n{numbered}\n```")

        return "\n".join(lines)

    def _call_model(self, instruction: str, max_tokens: int = 2000) -> str:
        """One lightweight completion call: system prompt + recent history +
        a one-off stage instruction. Does not touch self.agent_history --
        callers decide what (if anything) gets persisted once the whole
        pipeline is done, so intermediate stage instructions don't bloat the
        conversation the next question is built on.
        """
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.agent_history[-10:]
            + [{"role": "user", "content": instruction}]
        )
        try:
            response = litellm.completion(
                model=PROVIDERS[self.agent_provider]["model"],
                messages=messages,
                max_tokens=max_tokens,
                timeout=120.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"(request failed: {exc})"

    _CALC_RE = re.compile(r"^\s*CALC:\s*(.+)$", re.MULTILINE)
    _PROMOTE_RE = re.compile(r"^\s*PROMOTE:\s*(.+)$", re.MULTILINE)

    @classmethod
    def _extract_directives(cls, text: str):
        """Pull CALC:/PROMOTE: lines out of an agent response, returning
        (cleaned_text, calc_exprs, promote_names). Cleaned text has those
        lines stripped so they don't clutter what's printed/stored.
        """
        calc_exprs = [m.strip() for m in cls._CALC_RE.findall(text) if m.strip()]
        promote_names = []
        for m in cls._PROMOTE_RE.findall(text):
            promote_names.extend(n.strip() for n in m.split(",") if n.strip())

        cleaned = cls._CALC_RE.sub("", text)
        cleaned = cls._PROMOTE_RE.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, calc_exprs, promote_names

    def _apply_directives(self, calc_exprs: List[str], promote_names: List[str]) -> str:
        """Deterministically compute any CALC: expressions and apply any
        PROMOTE: requests, returning a short human-readable summary to
        print and to feed back into the agent's own history (so it sees
        the verified numbers on the next turn instead of trusting its own
        arithmetic).
        """
        notes = []

        if calc_exprs:
            computed = [(expr, _safe_eval_math(expr)) for expr in calc_exprs]
            calc_lines = "\n".join(f"  {expr} = {result}" for expr, result in computed)
            print(f"  🧮 Verified calculations:\n{calc_lines}")
            notes.append(f"Pulse computed these deterministically -- use these exact values:\n{calc_lines}")

        if promote_names:
            promoted = []
            for name in promote_names:
                result = self._cmd_track(name, quiet=True)
                if result:
                    promoted.append(result if result != "all" else "all tracked variables")
            if promoted:
                print(f"  ⚙ Promoted to full tracking (agent request): {', '.join(promoted)}")
                notes.append(f"Promoted to full tracking: {', '.join(promoted)}.")

        return "\n\n".join(notes)

    @staticmethod
    def _parse_json_obj(answer: str) -> Optional[Dict[str, Any]]:
        """Generic defensive JSON-object parser, used for the verify (pass
        4) and sweep (pass 5) responses -- same tolerance for stray code
        fences as _parse_code_fix, but without requiring any particular
        fields."""
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
    def _describe_fix(fix: Dict[str, Any]) -> str:
        parts = []
        for i, (old, new) in enumerate(zip(fix["old"], fix["new"])):
            parts.append(f"--- change {i + 1} ---\nOLD:\n{old}\nNEW:\n{new}")
        if fix.get("explanation"):
            parts.append(f"Explanation: {fix['explanation']}")
        return "\n\n".join(parts)

    def _verify_fix_with_retries(self, fix: Dict[str, Any]):
        """PASS 4: check the fix's math/logic before it's handed to the
        user. If it fails, ask the agent to revise and re-check, up to
        _MAX_VERIFY_ATTEMPTS times. Returns (fix, passed, reason).
        """
        reason = ""
        for attempt in range(_MAX_VERIFY_ATTEMPTS):
            fix_desc = self._describe_fix(fix)
            with _Spinner("Checking the fix"):
                verify_answer = self._call_model(
                    _PASS4_VERIFY_TMPL.format(fix_desc=fix_desc), max_tokens=200
                )
            verdict = self._parse_json_obj(verify_answer)
            if verdict is None:
                # Unparsable verdict -- don't block the user on a formatting
                # slip; hand off the fix as-is with a note.
                return fix, True, "(verification response was unparsable; proceeding anyway)"
            passes = bool(verdict.get("passes"))
            reason = str(verdict.get("reason", "")).strip()
            if passes:
                return fix, True, reason
            if attempt == _MAX_VERIFY_ATTEMPTS - 1:
                break
            with _Spinner("Revising fix"):
                revised_answer = self._call_model(
                    _PASS4_REVISE_TMPL.format(reason=reason), max_tokens=4000
                )
            revised = self._parse_code_fix(revised_answer)
            if revised is None:
                break
            fix = revised
        return fix, False, (reason or "(verification did not clearly pass after retries)")

    def _run_sweep_and_maybe_recurse(self, include_code: bool, _depth: int) -> None:
        """PASS 5: re-read everything for OTHER, unrelated errors. If any
        turn up, ask the user whether to fix those too (PASS 6 recurses
        through the same 1-5 format for the new issue)."""
        with _Spinner("Reading for other errors"):
            sweep_answer = self._call_model(_PASS5_SWEEP, max_tokens=300)
        sweep = self._parse_json_obj(sweep_answer)
        found = bool(sweep.get("other_errors_found")) if sweep else False
        summary = str(sweep.get("summary", "")).strip() if sweep else ""

        if not found or not summary:
            return

        print(f"\n[5] Full re-read found other possible issue(s):\n{summary}\n")
        try:
            resp = input("Fix other errors? (y/n) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if resp not in ("y", "yes"):
            return

        print("\n[6] Following the established format for the additional issue(s)...")
        self.ask_agent(
            f"Please also fix this: {summary}", include_code=include_code, _depth=_depth + 1
        )

    def ask_agent(self, question: str, include_code: bool = False, _depth: int = 0) -> str:
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

        Each pass prints as soon as it's ready, with a small spinner shown
        while it's in flight. Only the top-level call (_depth == 0)
        restarts the process afterward, once, if any fix was applied
        anywhere in the (possibly recursive) chain.
        """
        if _depth == 0:
            self._fix_applied_this_turn = False

        if not self.agent_provider or not self.agent_key:
            return "(AI agent is not enabled. Run setup again or set the API key.)"

        context = self._build_agent_context(include_code=include_code)
        user_content = f"{context}\n\nQuestion: {question}"
        self.agent_history.append({"role": "user", "content": user_content})

        wants_implementation = include_code and any(
            kw in question.lower() for kw in _IMPLEMENT_KEYWORDS
        )

        # Pass 1: locate the region(s) of the error.
        with _Spinner("Reading for region of error"):
            regions = self._call_model(_PASS1_LOCATE, max_tokens=200)
        print(f"\n[1] Region of error\n{regions}\n")

        # Pass 2: focused second read + diagnosis/reasoning. May include
        # CALC:/PROMOTE: directives, stripped out and executed
        # deterministically rather than trusted from the model.
        with _Spinner("Analyzing"):
            raw_analysis = self._call_model(
                _PASS2_ANALYZE_TMPL.format(regions=regions), max_tokens=700
            )
        analysis, calc_exprs, promote_names = self._extract_directives(raw_analysis)
        print(f"[2] Diagnosis & reasoning\n{analysis}\n")
        directive_note = self._apply_directives(calc_exprs, promote_names)
        if directive_note:
            self.agent_history.append({"role": "user", "content": directive_note})

        full_answer = f"{regions}\n\n{analysis}"

        if not wants_implementation:
            # No code change requested -- pass 3 is just the concrete fix
            # in text; nothing to verify or sweep.
            with _Spinner("Developing fix"):
                fix_text = self._call_model(_PASS3_FIX_TEXT, max_tokens=300)
            print(f"[3] Fix\n{fix_text}\n")
            full_answer += f"\n\n{fix_text}"
            self.agent_history.append({"role": "assistant", "content": full_answer})
            return full_answer

        # Pass 3: develop and implement the fix.
        with _Spinner("Developing & implementing fix"):
            fix_answer = self._call_model(_PASS3_IMPLEMENT, max_tokens=4000)
        fix = self._parse_code_fix(fix_answer)
        if fix is None:
            print(f"[3] Fix\n{fix_answer}\n")
            full_answer += f"\n\n{fix_answer}"
            self.agent_history.append({"role": "assistant", "content": full_answer})
            return full_answer

        # Pass 4: verify the fix's math/logic before handing it to the
        # user; revise and re-check on failure (bounded retries).
        fix, verify_ok, verify_reason = self._verify_fix_with_retries(fix)
        status = "passed" if verify_ok else "did not clearly pass -- applying best effort"
        print(f"[4] Verification {status}: {verify_reason}\n")

        self.agent_history.append({"role": "assistant", "content": full_answer})
        apply_result = self._apply_code_fix(fix)
        result = f"{full_answer}\n\n{apply_result}"

        # Pass 5 (+ 6): only worth a full re-read if a fix actually landed.
        if self._fix_applied_this_turn and _depth < 3:
            self._run_sweep_and_maybe_recurse(include_code, _depth)

        if _depth == 0 and self._fix_applied_this_turn:
            self._restart_process()  # does not return

        return result

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

    def _apply_code_fix(self, fix: Dict[str, Any]) -> str:
        """Apply an agent-proposed code fix to the file(s) it targets.

        Each fix entry may target a different file (see "files" in the
        code-fix schema, for a modularized project) -- edits are grouped by
        resolved file path so each file is read/written once regardless of
        how many snippets in it changed. Each old[i] must appear exactly
        once in its target file's current contents; snippets that don't
        match cleanly, or whose file can't be resolved, are skipped and
        reported rather than guessed at. On success, backups of every
        touched file are kept so the user can revert them all together.
        """
        self._build_file_labels()

        by_path: Dict[str, List[tuple]] = {}
        unresolved = []
        for old, new, label in zip(fix["old"], fix["new"], fix["files"]):
            path = self._resolve_fix_path(label)
            if not path:
                unresolved.append((old, label))
                continue
            by_path.setdefault(path, []).append((old, new))

        if not by_path and not unresolved:
            return "[Pulse CLI] Proposed a code fix with nothing to apply."

        lines = ["[Pulse CLI] Code fix"]
        if fix["explanation"]:
            lines.append(f"Explanation: {fix['explanation']}")

        originals: Dict[str, str] = {}
        applied_by_path: Dict[str, List[tuple]] = {}
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
            if path == self.script_path:
                self.code_text = content
            elif path in self.extra_files:
                self.extra_files[path] = content

        for old, label in unresolved:
            skipped.append((old, label or "(unspecified file)", "couldn't determine which file this targets"))

        if not applied_by_path:
            lines.append("\n⚠ No changes were applied -- none of the proposed snippets matched cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: {old.splitlines()[0][:80]}...")
            return "\n".join(lines)

        for path, applied in applied_by_path.items():
            lines.append(f"\n✓ Applied {len(applied)} change(s) to '{os.path.basename(path)}':")
            for old, new in applied:
                lines.append(f"  - replaced:\n      {old.splitlines()[0][:80]}...\n    with:\n      {new.splitlines()[0][:80]}...")
        if skipped:
            lines.append(f"\n⚠ Skipped {len(skipped)} proposed change(s) that didn't match cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: {old.splitlines()[0][:80]}...")

        print("\n".join(lines))

        self._pending_revert_backups = dict(originals)  # path -> original content
        self._fix_applied_this_turn = True  # tells ask_agent's top-level call to restart afterward

        return "\n".join(lines)

    def _cmd_code(self, arg: str) -> None:
        """Toggle whether questions include the training code (and any
        cross-file project context) by default. On by default."""
        arg = arg.strip().lower()
        if arg in ("on", "true", "1", "enable", "enabled"):
            self.include_code_default = True
            print("✓ Code is now included with every question by default.")
        elif arg in ("off", "false", "0", "disable", "disabled"):
            self.include_code_default = False
            print("✓ Code will no longer be included by default.")
        else:
            self.include_code_default = not self.include_code_default
            state = "ON" if self.include_code_default else "OFF"
            print(f"✓ Sending code by default is now {state}.")

    def _cmd_autofix(self, arg: str) -> None:
        """Toggle auto-intervention: Pulse watching tracked values for signs
        of trouble and automatically pausing + asking the agent to diagnose
        (and try to fix) it, without waiting to be asked."""
        arg = arg.strip().lower()
        if arg in ("on", "true", "1", "enable", "enabled"):
            self.auto_intervene = True
            print("✓ Auto-intervention is ON -- Pulse will pause and ask the agent if training looks like it's going bad.")
        elif arg in ("off", "false", "0", "disable", "disabled"):
            self.auto_intervene = False
            print("✓ Auto-intervention is OFF -- Pulse will only diagnose issues when you ask.")
        else:
            state = "ON" if self.auto_intervene else "OFF"
            print(f"Auto-intervention is currently {state}. Usage: /autofix on|off")

    def _check_for_trouble(self) -> Optional[str]:
        """Look at the current tracked values for signs training is going
        bad. Returns a short human-readable description of what's wrong, or
        None if nothing looks off. Two triggers, deliberately conservative
        to avoid false alarms:
          - any tracked scalar's latest value is non-finite (NaN/inf) --
            unambiguous, always worth stopping for.
          - a loss-like scalar has spiked to `explosion_multiplier`x (or
            more) its own recent minimum -- a strong divergence signal
            without needing a hardcoded absolute threshold.
        Also flags any cached matrix/tensor whose most recent stats show
        nan/inf, whether it's fully tracked or lotracked.
        """
        reasons = []

        for var_name in self.tracked_vars:
            hist = self.scalar_histories.get(var_name)
            if not hist:
                continue
            latest = hist[-1]
            if latest is not None and isinstance(latest, (int, float)) and not math.isfinite(latest):
                reasons.append(f"'{var_name}' just went non-finite (NaN/inf): {latest}")
                continue
            if _looks_like_loss(var_name) and latest is not None:
                finite_recent = [v for v in hist[-20:] if v is not None and math.isfinite(v)]
                if len(finite_recent) >= 5:
                    baseline = min(finite_recent[:-1])
                    if baseline > 0 and latest > baseline * self.explosion_multiplier:
                        reasons.append(
                            f"'{var_name}' spiked to {latest:.4g}, "
                            f"{latest / baseline:.1f}x its recent minimum ({baseline:.4g})"
                        )

        for sub_name, entry in self._matrix_cache.items():
            stats = entry.get("stats", {})
            if stats.get("nan") or stats.get("inf"):
                reasons.append(f"'{sub_name}' has nan={stats.get('nan')} inf={stats.get('inf')}")

        return "; ".join(reasons) if reasons else None

    def update(self, step: Optional[int] = None, generate_pdfs: Optional[bool] = None) -> None:
        """Called at every training step/checkpoint.

        The global step counter only advances when the loss/metric scalar
        (the first tracked variable that looks like a loss) actually
        changes value -- calling update() every micro-iteration of a loop
        that only updates loss occasionally no longer inflates the step
        count. If no loss-like variable is tracked, the step counter just
        falls back to incrementing on every call, same as before.

        Matrix/tensor statistics are expensive (especially on GPU), so they
        are probed on a schedule -- but the schedule now depends on each
        variable's state: fully-'track'ed variables use matrix_probe_interval
        (and, if enabled, get PDF snapshots); 'lotrack' variables use the
        much slower lotrack_probe_interval, never get PDFs, and are printed
        as a single condensed stats line instead of the full tagging line.
        Between probes, Pulse does not even slice or call statistics() on a
        variable; it only uses the cached result.

        Individual variables that are currently None, NaN-only, or otherwise
        unreadable are reported as such (rather than raising) so one bad
        variable never takes down the whole debugger mid-training.
        """
        loss_var = next((v for v in self.tracked_vars if _looks_like_loss(v)), None)
        new_loss_value: Optional[float] = None
        if loss_var is not None and loss_var in self.watch_locals:
            raw = self.watch_locals[loss_var]
            if raw is not None and is_trackable(raw):
                try:
                    if describe_tensor(raw).kind == "scalar":
                        new_loss_value = float(statistics(raw).get("mean"))
                except Exception:
                    new_loss_value = None

        if step is not None:
            self.step = step
        elif loss_var is None:
            # No loss-like variable tracked -- nothing to gate on, so fall
            # back to the old "advance every call" behavior.
            self.step += 1
        else:
            if not _values_equal(self._last_loss_value, new_loss_value):
                self.step += 1
            self._last_loss_value = new_loss_value

        want_pdfs = self.generate_pdfs if generate_pdfs is None else generate_pdfs
        now = time.monotonic()

        # First call probes immediately. After that, each state has its own
        # independent wall-clock cadence.
        probe_track = (
            not self._matrix_cache
            or (now - self._last_matrix_probe) >= self.matrix_probe_interval
            or any(v not in self._matrix_cached_vars for v in self.tracked_vars if self._state_of(v) == "track")
        )
        probe_lotrack = (
            (now - self._last_lotrack_probe) >= self.lotrack_probe_interval
            or any(v not in self._matrix_cached_vars for v in self.tracked_vars if self._state_of(v) == "lotrack")
        )
        probe_matrices = probe_track or probe_lotrack

        scalar_lines: List[tuple[str, Optional[float]]] = []
        # (sub_name, stats, val, state)
        matrix_lines: List[tuple[str, Dict[str, Any], Any, str]] = []
        any_scalar_changed = False

        # Fast path: discover/process scalars every step. Do NOT call
        # _yield_slices() for matrices unless a probe for that variable's
        # state is actually due this call.
        for var_name in self.tracked_vars:
            if var_name not in self.watch_locals:
                continue

            orig_val = self.watch_locals[var_name]
            var_state = self._state_of(var_name)
            due_this_var = probe_track if var_state == "track" else probe_lotrack

            if orig_val is None:
                hist = self.scalar_histories.setdefault(var_name, [])
                if not hist or not _values_equal(hist[-1], None):
                    hist.append(None)
                    any_scalar_changed = True
                scalar_lines.append((var_name, None))
                continue

            if not is_trackable(orig_val):
                continue

            try:
                kind = describe_tensor(orig_val).kind
            except Exception:
                kind = None

            if kind == "scalar":
                try:
                    stats = statistics(orig_val)
                    # Derive scalar_val from the SAME statistics() call that produced
                    # the nan/inf flags, rather than converting the tensor a second,
                    # independent time via to_numpy(). Two separate conversions of a
                    # live (possibly GPU/async) tensor can disagree -- e.g. the second
                    # read racing an in-flight op -- which was causing normal, finite
                    # loss values to get flagged as NaN/inf. For a true scalar, mean
                    # over its single element is just that element, so this stays
                    # perfectly consistent with stats['nan']/stats['inf'].
                    scalar_val = float(stats.get("mean"))
                except Exception as exc:
                    hist = self.scalar_histories.setdefault(var_name, [])
                    if not hist or not _values_equal(hist[-1], None):
                        hist.append(None)
                        any_scalar_changed = True
                    scalar_lines.append((var_name, None))
                    print(f"  ⚠ '{var_name}' could not be read this step: {type(exc).__name__}: {exc}")
                    continue
                hist = self.scalar_histories.setdefault(var_name, [])
                changed = not hist or not _values_equal(hist[-1], scalar_val)
                if changed:
                    hist.append(scalar_val)
                    any_scalar_changed = True
                scalar_lines.append((var_name, scalar_val))
                continue

            # Matrix/tensor path. No slicing, no statistics(), and no GPU->CPU
            # copy at all unless this variable's own probe cadence is due.
            if not due_this_var:
                continue

            for sub_name, val in self._yield_slices(var_name, orig_val):
                if val is None:
                    hist = self.scalar_histories.setdefault(sub_name, [])
                    if not hist or not _values_equal(hist[-1], None):
                        hist.append(None)
                        any_scalar_changed = True
                    scalar_lines.append((sub_name, None))
                    continue

                try:
                    stats = statistics(val)
                except Exception as exc:
                    print(f"  ⚠ '{sub_name}' could not be read this step: {type(exc).__name__}: {exc}")
                    continue

                if stats.get("kind") == "scalar":
                    try:
                        # Same fix as the top-level scalar path above: reuse the
                        # already-computed `stats` rather than re-converting `val`
                        # independently, to avoid spurious NaN/inf false positives.
                        scalar_val = float(stats.get("mean"))
                    except Exception:
                        hist = self.scalar_histories.setdefault(sub_name, [])
                        if not hist or not _values_equal(hist[-1], None):
                            hist.append(None)
                            any_scalar_changed = True
                        scalar_lines.append((sub_name, None))
                        continue
                    hist = self.scalar_histories.setdefault(sub_name, [])
                    changed = not hist or not _values_equal(hist[-1], scalar_val)
                    if changed:
                        hist.append(scalar_val)
                        any_scalar_changed = True
                    scalar_lines.append((sub_name, scalar_val))
                else:
                    self._matrix_cache[sub_name] = {
                        "base_name": var_name,
                        "stats": stats,
                    }
                    matrix_lines.append((sub_name, stats, val, var_state))

            self._matrix_cached_vars.add(var_name)

        if probe_track:
            self._last_matrix_probe = now
        if probe_lotrack:
            self._last_lotrack_probe = now

        # Quiet by design: a scalar (loss, accuracy, whatever) is only ever
        # printed when it's actually WRONG -- unreadable (None) or non-finite
        # (NaN/inf). A healthy loss ticking along normally never shows up
        # here; auto-intervention (below) is what's watching it, silently.
        def _is_wrong(v):
            return v is None or (isinstance(v, (int, float)) and not math.isfinite(v))

        wrong_scalars = [(n, v) for n, v in scalar_lines if _is_wrong(v)]

        # lotrack matrices/tensors NEVER print, under any circumstances --
        # they're still probed and cached (so auto-intervention still sees
        # nan/inf on them), just never surfaced in the terminal. Only
        # 'track' variables get a tagging line, and only when freshly
        # measured this probe.
        track_matrix_lines = (
            [t for t in matrix_lines if t[3] != "lotrack"] if probe_matrices else []
        )

        should_redraw = bool(wrong_scalars or track_matrix_lines)

        if should_redraw:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            print(f"--- Pulse Live Debugger | Step {self.step} ---")

            for sub_name, scalar_val in wrong_scalars:
                hist = self.scalar_histories.get(sub_name, [])
                if scalar_val is None:
                    print(f"  • {sub_name}: NoneType  ⚠ (unreadable this step)")
                else:
                    print(f"  • {sub_name}: {scalar_val:.6g}  ⚠")
                self._print_ascii_chart(sub_name, hist)

            if track_matrix_lines:
                def _fmt(v):
                    try:
                        return f"{v:.4f}"
                    except (TypeError, ValueError):
                        return "n/a"

                for sub_name, stats, val, var_state in track_matrix_lines:
                    flag = ""
                    if stats.get("nan") or stats.get("inf"):
                        flag = f"  ⚠ nan={stats.get('nan')} inf={stats.get('inf')}"
                    mean_v, min_v, max_v = stats.get("mean"), stats.get("min"), stats.get("max")

                    print(
                        f"  • Tagging '{sub_name}' "
                        f"[{stats.get('backend')} {stats.get('kind')} {stats.get('shape')}] "
                        f"| mean={_fmt(mean_v)} min={_fmt(min_v)} "
                        f"max={_fmt(max_v)}{flag}"
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

                if self._matrix_cache:
                    print(
                        f"  [matrices cached: {len(self._matrix_cache)} | "
                        f"next full probe ≤ {self.matrix_probe_interval:g}s | "
                        f"next lotrack probe ≤ {self.lotrack_probe_interval:g}s]"
                    )

        if self.auto_intervene:
            problem = self._check_for_trouble()
            if problem and problem != self._last_intervention_signature:
                self._last_intervention_signature = problem
                self.continuous = False  # force a stop even mid-continuous-run
                print("\n" + "=" * 60)
                print("[Pulse] ⚠ Auto-intervention: training looks like it's going bad. Pausing.")
                print(f"[Pulse] Detected: {problem}")
                print("=" * 60)
                if self.agent_provider:
                    question = (
                        f"Pulse just auto-paused training because it detected a problem: {problem}\n"
                        "Please diagnose the root cause and, if you can, fix it."
                    )
                    print("Pulse:")
                    self.ask_agent(question, include_code=bool(self.code_text))
                else:
                    print("[Pulse] No AI agent is configured yet -- run /agent to set one up, then ask about this.")
                print(
                    "[Pulse] Training is paused here (Enter to single-step, /c to resume continuous "
                    "running as-is). Note: if a fix was just applied to a file on disk, this already-"
                    "running process is still executing the old code in memory -- you'll need to stop "
                    "and re-run the script to pick it up.\n"
                )

        if self.continuous:
            return

        # Interactive Training Loop Prompt
        while True:
            try:
                cmd = input(
                    "\nPulse [Enter=step, /c=continuous, /add <var>, /track <var>, "
                    "/lotrack <var>, /delete <var>, /deletepdf <var>, /autofix on|off, "
                    "/agent, /code, or ask AI] > "
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

            if cmd.lower().startswith("/track "):
                self._cmd_track(cmd[7:].strip())
                continue
            if cmd.lower().startswith("/lotrack "):
                self._cmd_lotrack(cmd[9:].strip())
                continue
            if cmd.lower().startswith("/autofix"):
                self._cmd_autofix(cmd[8:].strip())
                continue

            if cmd.lower().startswith("/deletepdf "):
                self._cmd_delete_pdfs(cmd[11:].strip())
                continue
            if cmd.lower().startswith("/delete "):
                self._cmd_delete(cmd[8:].strip())
                continue

            if cmd.lower() == "/agent":
                self._select_agent_provider_and_key(initial=False)
                continue

            if cmd.lower() == "/vars":
                self._print_variable_summary()
                continue

            if cmd.lower() == "/tracked":
                cfg_strs = [
                    f"{v}({self.var_configs[v]})[{self._state_of(v)}]" if v in self.var_configs
                    else f"{v}[{self._state_of(v)}]"
                    for v in self.tracked_vars
                ]
                print("Tracked:", ", ".join(cfg_strs) if cfg_strs else "(none)")
                continue

            if cmd.lower().startswith("/code"):
                self._cmd_code(cmd[5:].strip())
                continue

            print("Pulse AI:")
            self.ask_agent(cmd, include_code=self.include_code_default)

    def _print_ascii_chart(
        self,
        name: str,
        history: List[Optional[float]],
        height: int = 8,
        width: int = 64,
    ) -> None:
        """Render a compact line-style loss/metric graph.

        The X axis advances only when the scalar value changes, so repeated
        training-loop calls do not create fake horizontal steps.

        `history` may contain None (variable was unreadable/NoneType that
        step) or NaN/inf floats. Those points are never fed into the min/max/
        round math -- they're drawn as a distinct '!' marker instead -- since
        `round(float('nan'))` raises and would otherwise crash every call.
        """
        if not history:
            return

        data = history[-width:]
        finite = [v for v in data if isinstance(v, (int, float)) and math.isfinite(v)]

        if not finite:
            print(f"    [{name} | {len(history)} points] ⚠ no readable values (None/NaN/inf) -- nothing to chart")
            return

        if len(data) == 1:
            v = data[0]
            if isinstance(v, (int, float)) and math.isfinite(v):
                print(f"    {v:>10.5g} ┤ ●")
            else:
                print(f"    {'None/NaN':>10} ┤ !")
            print("              └─ step 1")
            return

        lo = min(finite)
        hi = max(finite)

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

        def _readable(v):
            return isinstance(v, (int, float)) and math.isfinite(v)

        ys = [y_for(v) if _readable(v) else None for v in data]
        any_broken = False

        # Plot points and simple line segments. Unreadable points (None/NaN/
        # inf) get a '!' marker on the bottom row and never anchor a line
        # segment, so a single bad step doesn't distort the whole chart.
        for i, y in enumerate(ys):
            if y is None:
                any_broken = True
                grid[rows - 1][i] = "!"
                continue
            grid[y][i] = "●"
            if i == 0 or ys[i - 1] is None:
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

        flag = "  ⚠ '!' = None/NaN/inf this step" if any_broken else ""
        print(f"    [{name} | {len(history)} points | showing last {cols}]{flag}")

        for r in range(rows):
            value = hi - (hi - lo) * (r / (rows - 1))
            print(f"    {value:>10.5g} ┤ " + "".join(grid[r]))

        print("              └" + "─" * cols)
        print(f"               {max(1, len(history) - cols + 1):<{max(1, cols // 2)}}"
              f"{len(history)}")
