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
import hashlib
import difflib
import itertools
import subprocess
import threading
import atexit
from datetime import datetime
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
from pulse import pulse_supabase as cloud
try:
    import litellm
    # Quiets litellm's own verbose provider-debug logging (request/response
    # dumps) that would otherwise interleave with Pulse's own output on
    # every single agent call -- this is a CLI tool, not a debug console
    # for litellm itself. PULSE_LITELLM_DEBUG=1 re-enables it.
    if os.environ.get("PULSE_LITELLM_DEBUG", "").strip() not in ("1", "true", "yes"):
        litellm.suppress_debug_info = True
except ImportError:
    # A fresh `pip install` of Pulse without its one non-stdlib runtime
    # dependency is the single most likely first-run failure a brand new
    # user hits -- and a bare ModuleNotFoundError traceback here, before
    # any of Pulse's own error handling exists yet to catch it, is a bad
    # first impression. Fail with one clear, actionable line instead.
    print(
        "\n[Pulse] Missing dependency: litellm (used to talk to AI providers).\n"
        "         Install it with:  pip install litellm\n",
        file=sys.stderr,
    )
    sys.exit(1)

import builtins

_io_lock = threading.Lock()
_original_print = builtins.print
_original_input = builtins.input



def safe_print(*args, **kwargs):
    with _io_lock:
        # Move cursor to column 0 and clear line before printing background text
        sys.stdout.write("\r\033[K")
        _original_print(*args, **kwargs)

def safe_input(prompt=""):
    # Clear formatting and force prompt to a clean new line
    sys.stdout.write("\033[0m\n\r\033[K")
    sys.stdout.flush()
    
    # Hold the lock while waiting for user input
    with _io_lock:
        return _original_input(prompt)

# Global overrides
builtins.print = safe_print
builtins.input = safe_input
# Name-based heuristic for pre-flagging the loss/metric scalar -- same list
# and same purpose as pulse.py's LOSS_NAME_HINTS/_looks_like_loss, kept as a
# local copy so this module has no dependency on pulse.py (which pulls in
# tkinter and isn't safe to import in a headless CLI/Colab/SSH session).


def _pulse_is_accelerator_value(value: Any) -> bool:
    """Return True when *value* lives on a non-CPU accelerator.

    Pulse CLI is deliberately CPU-only: this check must never call a device
    synchronization, .cpu(), .numpy(), .item(), or any other operation that
    reads accelerator memory. It only inspects already-available metadata.
    """
    if value is None:
        return False
    try:
        device = getattr(value, "device", None)
        if device is not None:
            dtype = getattr(device, "type", None)
            if isinstance(dtype, str) and dtype.lower() not in ("cpu", ""):
                return True
            ds = str(device).lower()
            if ds and not ds.startswith("cpu") and any(x in ds for x in ("cuda", "mps", "xpu", "rocm", "hip", "gpu", "tpu")):
                return True
    except Exception:
        pass
    try:
        d = str(getattr(value, "device", "")).lower()
        if d and any(x in d for x in ("cuda", "mps", "xpu", "rocm", "hip", "gpu", "tpu")):
            return True
    except Exception:
        pass
    try:
        # TensorFlow exposes placement through .device without requiring a
        # tensor read.
        d = str(getattr(value, "device", "")).lower()
        if "/device:cpu:" not in d and "/device:" in d:
            return True
    except Exception:
        pass
    try:
        # CuPy exposes device metadata without synchronizing the array.
        dev = getattr(value, "device", None)
        did = getattr(dev, "id", None)
        if did is not None and int(did) >= 0:
            return True
    except Exception:
        pass
    return False


def _pulse_cpu_mirror_candidates(name: str):
    """Names commonly used by training code for an explicitly maintained CPU mirror."""
    bases = [name, name.rsplit('.', 1)[-1]]
    suffixes = ("_cpu", "_host", "_np", "_numpy", "_cpu_copy", "_host_copy")
    seen = set()
    for base in bases:
        for suffix in suffixes:
            candidate = base + suffix
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


LOSS_NAME_HINTS = ("loss", "cost", "nll", "cross_entropy", "crossentropy", "objective", "err")


def _looks_like_loss(name: str) -> bool:
    n = (name or "").lower()
    return any(hint in n for hint in LOSS_NAME_HINTS)


def _enable_windows_ansi() -> None:
    """Turn on ANSI escape processing in classic Windows consoles (Win10+
    supports it, but it has to be switched on explicitly). No-op, and safe
    to call, everywhere else."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


_enable_windows_ansi()

# ---- minimal terminal styling ------------------------------------------
# Kept deliberately small: "Pulse" is always orange, error/warning text is
# always red, and code patches/diffs are always blue. Nothing else in the
# CLI is colored.
_COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_ORANGE = "\033[38;5;208m"
_RED = "\033[91m"
_BLUE = "\033[94m"
_RESET = "\033[0m"
_YELLOW = "\033[93m"
_PULSE_RE = re.compile(r"Pulse(?:\s(?:CLI|AI))?")


# ----------------------------------------------------------------------------
# Terms of Service -- the actual license Pulse ships under (mirrors the
# LICENSE file in the Pulse GitHub repo). Shown in full at sign-up (see
# _prompt_tos_acceptance) and its acceptance is what cloud.record_tos_
# acceptance timestamps server-side. PULSE_TOS_URL, if set, is shown
# alongside this as a link to the canonical hosted copy -- update that env
# var (and this text, if the license is ever revised) together so the two
# never drift apart.
# ----------------------------------------------------------------------------
PULSE_LICENSE_TEXT = """\
PULSE PROPRIETARY SOFTWARE LICENSE

Copyright (c) 2026 Yash Patel. All Rights Reserved.

This software and associated files (the "Software") are the proprietary
property of the copyright holder. The Software is licensed, not sold.

1. GRANT OF LICENSE
   Subject to the terms of this license, the copyright holder grants you a
   limited, non-exclusive, non-transferable, revocable license to install
   and run the Software for your own internal use.

2. RESTRICTIONS
   Except as expressly permitted above, you may NOT, and may not permit
   others to:
   (a) copy, reproduce, distribute, sublicense, rent, lease, or resell the
       Software;
   (b) modify, adapt, translate, or create derivative works based on the
       Software;
   (c) reverse engineer, decompile, disassemble, or otherwise attempt to
       derive the source code of the Software, except to the extent this
       restriction is prohibited by applicable law;
   (d) remove, obscure, or alter any proprietary notices on the Software;
   (e) use the Software to build a competing product or service.

3. FUTURE PAID TERMS
   The copyright holder reserves the right to change pricing, introduce
   paid tiers or license keys, limit functionality, or discontinue free
   distribution of future versions at any time. Continued use of any
   future version may be subject to additional terms, including payment.

4. NO WARRANTY
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
   NONINFRINGEMENT.

5. LIMITATION OF LIABILITY
   IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM,
   DAMAGES, OR OTHER LIABILITY ARISING FROM, OUT OF, OR IN CONNECTION
   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

6. TERMINATION
   This license is effective until terminated. It will terminate
   automatically without notice if you fail to comply with any of its
   terms. Upon termination, you must cease all use of the Software and
   destroy all copies in your possession.

For licensing inquiries, contact: codeyash09@gmail.com
"""


def _highlight_pulse(text: str, base_color: Optional[str] = None) -> str:
    """Color every occurrence of 'Pulse' (and 'Pulse CLI'/'Pulse AI') orange,
    resuming `base_color` afterward so nesting inside a red/blue line works."""
    if not _COLOR_ENABLED:
        return text
    resume = base_color or ""
    return _PULSE_RE.sub(lambda m: f"{_ORANGE}{m.group(0)}{_RESET}{resume}", text)


def cprint(text: str = "", color: Optional[str] = None) -> None:
    """print() that always highlights 'Pulse' in orange and, if `color` is
    given (_RED for errors/warnings, _BLUE for code patches), paints the
    rest of the line in that color."""
    text = str(text)
    if not _COLOR_ENABLED:
        print(text)
        return
    if color:
        print(f"{color}{_highlight_pulse(text, base_color=color)}{_RESET}")
    else:
        print(_highlight_pulse(text))


def _flush_stdin() -> None:
    """Discard any input sitting unread in the terminal's input buffer.

    Without this, keystrokes typed while output was streaming (spinner
    frames, restart banners, training-step logs) sit in the OS-level tty
    buffer and get delivered as soon as the next input() starts reading --
    landing in the middle of that prompt's line instead of being ignored,
    which is what makes a y/n prompt look "stuck" or uneditable. Called
    right before every input() so each prompt always starts from a clean,
    empty line.
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
    "  GPUTRACK: <comma-separated variable names>\n"
    "    If you genuinely need to watch a GPU-resident variable more closely to diagnose this "
    "(not just out of curiosity), name it here. WARNING: GPU tracking forces a device-to-host "
    "sync on every probe, which slows down training -- only do this when it's actually necessary, "
    "and it will still only be probed on the slow GPU cadence, never every step.\n"
    "  GPUUNTRACK: <comma-separated variable names>\n"
    "    Stop GPU-tracking a variable once you no longer need the closer look. Use this as soon "
    "as a GPU-tracked variable has told you what you needed.\n"
    "  SENSITIVITY: <0.0-1.0, a preset (loose/medium/tight), or 'spike|plateau|oscillation <value|auto>'>\n"
    "    Pulse's auto-intervention (spike/plateau/oscillation detection) runs off a single "
    "sensitivity dial, defaulted loose because normal training noise oscillates some of the time. "
    "If you can see this run's loss curve is unusually noisy (or unusually smooth) for its stage "
    "of training, adjust the dial so future auto-intervention checks match reality instead of "
    "either missing real trouble or crying wolf on normal noise. Only send this when you have an "
    "actual reason from the data, not by default.\n"
    "  NORMAL_START: <comma-separated var=value pairs, or 'none'>\n"
    "    Spike detection normally needs several real data points before it has a baseline to "
    "compare against, which means a genuine explosion in the first few steps of training can go "
    "undetected. If you can estimate a loss-like tracked variable's expected value at the very "
    "start of training from the code alone (e.g. a randomly-initialized N-class classifier's "
    "cross-entropy loss starts near ln(N); a policy's initial reward is often near a known "
    "random-policy baseline), send it here so Pulse can catch a real explosion from step one "
    "instead of waiting for history to accumulate. Only estimate what you can actually justify "
    "from the code -- omit a variable (or send 'none') rather than guess.\n"
    "  Put CALC:/PROMOTE:/GPUTRACK:/GPUUNTRACK:/SENSITIVITY:/NORMAL_START: lines anywhere in your Reasoning, not in the "
    "Diagnosis or Fix.\n\n"
    "PERIODIC GPU CHECK-IN:\n"
    "Every 15 minutes after Pulse starts, if you have an AI provider configured, Pulse will send "
    "you an automated check-in message asking whether you still want any variables GPU-tracked. "
    "When you get that message, reply with ONLY these two lines and nothing else:\n"
    "  GPUTRACK: <comma-separated names to track, or 'none'>\n"
    "  GPUUNTRACK: <comma-separated names to stop tracking, or 'none'>\n\n"

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

class AgentRequestFailed(Exception):
    """Raised by _call_model when a completion request fails for any
    reason (auth, rate limit, timeout, connection, malformed response --
    see _classify_model_error), carrying a short, clean, user-facing
    reason string as its message. Never lets a raw exception repr (which
    for some providers is a multi-line JSON blob) leak into the pipeline
    disguised as a real model answer -- see _ask_agent_impl's try/except,
    the one chokepoint that catches this for the whole multi-pass
    pipeline, and the GPU check-in's own try/except for the other call
    site outside that pipeline."""


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
    # --- Local / self-hosted -- no API key, nothing leaves the machine.
    # For teams whose training code/data/logs can't go to a third-party
    # LLM API at all. See _select_agent_provider_and_key's "local" branch
    # for the extra (api_base, model name) prompts these need instead of
    # a key, and _call_model for how they're actually invoked.
    "Ollama (local, private)": {
        "local": True,
        "default_api_base": "http://localhost:11434",
        "model_hint": "e.g. llama3.1, qwen2.5-coder, deepseek-r1 -- whatever you've pulled with `ollama pull`",
        "litellm_prefix": "ollama_chat",
    },
    "Local / self-hosted (OpenAI-compatible: LM Studio, vLLM, TGI...)": {
        "local": True,
        "default_api_base": "http://localhost:1234/v1",
        "model_hint": "the model name your local server is serving it as",
        "litellm_prefix": "openai",
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
        # Only set for local/self-hosted providers (see PROVIDERS' "local"
        # entries) -- the litellm model string ("ollama_chat/llama3.1",
        # "openai/my-model") and the local server's URL. None for a
        # normal cloud provider, where PROVIDERS[...]["model"]/api_base's
        # default (litellm talking straight to the vendor) are used
        # instead. Set together in _select_agent_provider_and_key,
        # consumed together in _call_model.
        self.agent_model_string: Optional[str] = None
        self.agent_api_base: Optional[str] = None
        self.agent_history: List[Dict[str, Any]] = []
        self.code_text: Optional[str] = None
        self.script_path: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.config_path: Optional[str] = None
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
        self._cpu_only_warned: set[str] = set()
        self._last_matrix_probe: float = 0.0
        self.matrix_probe_interval: float = 1.0
        self._last_lotrack_probe: float = 0.0
        self.lotrack_probe_interval: float = 5.0

        # GPU-variable tracking: the agent can flag a variable as worth
        # watching on the GPU (mirrors the manual /gputrack command) by
        # emitting a GPUTRACK:/GPUUNTRACK: directive -- see
        # _extract_directives/_apply_directives. GPU stat probes force a
        # device-to-host sync, which costs real training throughput, so
        # this is opt-in, always folded into the slow gpu_probe_interval
        # cadence (never the fast per-step one), and reviewed on a
        # schedule: every gpu_checkin_interval seconds after start, Pulse
        # proactively asks the agent whether it still wants each
        # GPU-tracked variable (or wants to add one), in a specific,
        # directive-only reply format -- see _maybe_gpu_checkin.
        self.gpu_tracked_vars: set[str] = set()
        self._last_gpu_probe: float = 0.0
        self.gpu_probe_interval: float = 600.0  # 10 minutes
        self._last_gpu_checkin: float = time.monotonic()
        self.gpu_checkin_interval: float = 900.0  # 15 minutes, first one 15 min after start

        # Auto-intervention: watch tracked values for signs training is
        # going bad (a scalar going non-finite, or a loss-like scalar
        # spiking well above its recent range) and, if so, automatically
        # pause (even out of continuous mode) and ask the agent to diagnose
        # -- and if it can, fix -- it, without waiting for the user to
        # notice and ask manually. On by default; toggle with /autofix.
        self.auto_intervene: bool = True
        # Sensitivity: one dial (0.0 = loosest, 1.0 = tightest) that drives
        # spike/plateau/oscillation detection in _check_for_trouble, plus
        # a few advanced per-signal overrides for anyone who wants finer
        # control than the single dial gives. Default is intentionally on
        # the loose side (not 0.5) because real training curves oscillate
        # some amount of the time as a matter of course -- a tight default
        # would auto-intervene constantly on normal noise. Adjustable by
        # the user (/sensitivity) and by the agent (a SENSITIVITY:
        # directive in its Reasoning, same mechanism as GPUTRACK: --
        # see _extract_directives/_apply_directives) since the agent is
        # often in a better position than a fixed default to judge
        # whether a given run's noise level is normal for its loss shape.
        self.sensitivity: float = 0.3
        # Advanced overrides: None means "derive from self.sensitivity"
        # (see _sensitivity_thresholds); set any of these directly (via
        # /sensitivity spike|plateau|oscillation <value>) to pin just that
        # one signal instead of moving the whole dial.
        self.explosion_multiplier: Optional[float] = None       # loss spike: latest > baseline * N
        self.plateau_range_frac: Optional[float] = None         # plateau: window range < frac * |latest|
        self.oscillation_flip_threshold: Optional[int] = None   # oscillation: sign flips required (of 18)
        self.oscillation_delta_frac: Optional[float] = None     # oscillation: avg |delta| > frac * scale
        self.stagnation_frac: Optional[float] = None            # stagnation: long-window improvement < frac * |early mean|
        # Agent-estimated (from reading the code, via a NORMAL_START:
        # directive -- see _apply_directives) expected starting value for
        # a loss-like tracked variable. Explosion detection normally needs
        # several real finite data points before it has anything to
        # compare `latest` against, which means a genuine blow-up in the
        # first few steps of training can slip through undetected -- this
        # gives it a code-derived anchor to compare against from step one,
        # before real history exists. See _prime_at_start, which asks the
        # agent for this automatically, once, before the first step.
        self._normal_start_baselines: Dict[str, float] = {}
        self._start_primed: bool = False   # _prime_at_start runs at most once per process
        self._last_intervention_signature: Optional[str] = None
        # /code is on by default -- every manually-asked question includes
        # the training code (and any cross-file context) unless turned off.
        self.include_code_default: bool = True

        # Interactive Mode & Interrupt Handling
        self.continuous = False
        self.original_sigint = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, self._sigint_handler)
        except ValueError:
            # signal.signal only works from the main thread on every
            # platform (raises ValueError elsewhere) -- some training
            # scripts construct their Pulse tracker from a worker thread
            # (e.g. a DataLoader/launcher wrapper). Ctrl+C just won't get
            # Pulse's custom handling in that case; that's strictly worse
            # UX, never a crash, so this is worth falling back on rather
            # than taking the whole training process down over it.
            self.original_sigint = None

        # Non-interactive / CI mode: set via PULSE_CI=1 or PULSE_NONINTERACTIVE=1
        # (or --non-interactive on the launcher, see auto_track). Every
        # place that would otherwise block on input()/getpass() for a
        # human (auth prompts, agent/key selection, "prompt for missing
        # value", /revert confirmations) checks this first and falls back
        # to an env-var-driven default or skips entirely instead of
        # hanging a CI job waiting on a TTY that isn't there. Auto-fix
        # itself still runs -- it never needed a human either -- this only
        # changes the one-time setup/confirmation prompts.
        #
        # PULSE_AUTO_RESTART folds in here too: _restart_process sets it
        # right before a fix-triggered restart, alongside PULSE_AUTO_
        # PROVIDER/API_BASE/MODEL. The whole point of that restart is
        # resuming an unattended run (it's 3am, nobody's at the
        # keyboard) -- so the resumed process needs to behave exactly
        # like non-interactive mode for its ENTIRE lifetime, not just
        # through setup: the workspace picker (_team_flow always
        # re-prompts by design -- see its docstring), the GPU/git-commit
        # manual-entry fallback (_prompt_for_missing), and any later
        # /agent switch all need to auto-pick/skip rather than block,
        # or training silently sits at an input() prompt until morning.
        # Popped (not just read) so a user-initiated re-run of the script
        # later doesn't inherit it by accident.
        resumed_after_restart = os.environ.pop("PULSE_AUTO_RESTART", "").strip() == "1"
        self._resumed_after_restart = resumed_after_restart  # kept as an attribute so _team_flow/_cloud_setup can print a "resumed" message instead of the generic non-interactive one
        # A normal run is interactive. Only an explicit auto-restart resume
        # may select unattended mode; stdin/CI environment heuristics are
        # intentionally not used because VS Code terminals can report a
        # redirected stdin while still accepting typed input.
        self.non_interactive: bool = resumed_after_restart
        if resumed_after_restart:
            cprint("[Pulse] Resumed after an auto-fix restart -- running unattended (non-interactive) for this session.")
        if self.non_interactive:
            # A CI job (or an unattended post-restart resume) has no one
            # at a keyboard to answer "Enter=step" -- run straight
            # through like /c would, instead of blocking forever on the
            # very first input() call.
            self.continuous = True
            # Auto-intervention (self.auto_intervene, set True above) is
            # what makes an unattended run actually self-heal instead of
            # just sitting there once something goes wrong -- it's
            # already on by default for every mode, but reaffirmed here
            # explicitly (rather than just relying on the shared default)
            # because leaving it off would be strictly worse in
            # non-interactive mode specifically: there's no one around to
            # notice a stuck run and type /autofix on themselves.
            self.auto_intervene = True
            cprint("[Pulse] Non-interactive mode: auto-fix is ON (use /autofix off if you don't want it, or PULSE_CI/PULSE_NONINTERACTIVE unset for a normal run).")

        # Telemetry opt-out (Phase 3): environment-info collection/sync can
        # be disabled entirely via PULSE_TELEMETRY=off, or toggled live
        # with /telemetry on|off. Checked at every collection site, not
        # just once at startup.
        self.telemetry_enabled: bool = cloud.telemetry_enabled()

        # Incident alerting (Slack-compatible webhook -- see /webhook and
        # _post_incident_webhook). PULSE_WEBHOOK_URL is the env-var
        # default; /webhook set overrides it for this session only.
        self._incident_webhook_url: Optional[str] = os.environ.get("PULSE_WEBHOOK_URL", "").strip() or None

        # --- Cloud (Supabase): account, team, live Debug_Sessions logging ---
        self.user_id: Optional[str] = None
        self.email: Optional[str] = None
        self.team_id: Optional[str] = None
        self.team_join_code: Optional[str] = None
        self.team_admin_ids: List[str] = []  # admin_ids of the currently-selected workspace
        self.debug_session_id: Optional[str] = None
        self._cloud_sync = cloud.BackgroundSync(on_error=self._on_cloud_error)
        self._cloud_warned = False
        # Local mirrors of the three array columns on Debug_Sessions. Kept
        # in full locally (uncompressed) since that's what the rest of
        # Pulse's logic (history context, known-fix lookup, /cloud status)
        # reads; only compressed (see pulse_supabase.encode_entry) right
        # before a flush actually goes over the wire, since PostgREST has
        # no "append to array column" verb and each flush necessarily
        # resends the whole array.
        self._agent_logs: List[Dict[str, Any]] = []
        self._error_tracebacks: List[str] = []
        self._telemetry: List[Dict[str, Any]] = []

        # Uptime/downtime + incidents: backs the (separate) web dashboard's
        # "what's actually happening in this run" view -- uptime is wall-
        # clock time spent in the user's own training code between Pulse
        # update() calls, downtime is wall-clock time Pulse itself spent
        # blocking on agentic work (diagnosing/fixing an auto-detected
        # problem), and incidents is a discrete log of exactly what
        # happened and when (crash, auto-intervention, fix applied,
        # revert, restart failure, etc.) -- see update()/_log_incident.
        self._uptime_seconds: float = 0.0
        self._downtime_seconds: float = 0.0
        self._incidents: List[Dict[str, Any]] = []
        self._last_update_end_ts: Optional[float] = None
        # Set right before an auto-intervention's ask_agent() call, read
        # by _finalize_agent_downtime -- see that method's docstring for
        # why this can't just be a local variable in update().
        self._pending_agent_start_ts: Optional[float] = None
        self._pending_agent_problem: Optional[str] = None

        # Cloud sync is batched, not per-event: fields touched since the
        # last flush are marked dirty here, and sent together as ONE
        # PATCH (not one per field) either when the flush timer elapses,
        # when a git commit is detected, or immediately for rare/important
        # events (a crash, an agent turn). This is what keeps Debug_Sessions
        # writes infrequent enough for a small compute tier to keep up with,
        # instead of a request every few seconds all training run long.
        self._cloud_dirty_fields: set = set()
        self.cloud_flush_interval: float = 600.0  # 10 minutes, for the noisy stuff (telemetry)
        self._last_cloud_flush: float = 0.0
        self._last_telemetry_sample: float = 0.0
        self.telemetry_sample_interval: float = 5.0  # local sampling only -- not a network call
        # uptime_seconds/downtime_seconds are two small integers, not
        # "noisy" like telemetry -- there's no real cost to syncing them
        # far more often, and a short run (well under 10 minutes) would
        # otherwise never see them synced at all, which is what made them
        # look like they "weren't logging". Checked on the same 5-second
        # cadence as telemetry sampling (see _maybe_collect_telemetry),
        # so this is the only place uptime/downtime freshness actually
        # depends on -- not on telemetry being enabled.
        self.uptime_flush_interval: float = 30.0
        self._last_uptime_flush: float = 0.0

        # Repo/commit tracking -- resolved to the tracked script's own
        # directory (not necessarily the process cwd) once script_path is
        # known, so `git` commands hit the right repo. See _cloud_setup.
        # Captured ONCE per logical run (an auto-fix restart carries the
        # same value forward rather than re-detecting -- see
        # _maybe_sync_commit_sha's docstring for why re-detecting is
        # actively wrong here) and only ever updated afterward via the
        # explicit /commit command.
        self._repo_cwd: Optional[str] = None
        self._last_synced_commit_sha: Optional[str] = None

        # Cross-session memory: previous Debug_Sessions for this team (or
        # user, if no team) are summarized into the agent's context, and
        # any fix that was actually applied is indexed by a signature of
        # the traceback it fixed -- so a recurring bug can be re-fixed
        # directly instead of re-diagnosed from scratch every time.
        self._history_context: str = ""
        self._known_fixes: Dict[str, Dict[str, Any]] = {}  # signature -> fix dict
        self._last_applied_fix: Optional[Dict[str, Any]] = None
        # Dedup bookkeeping so one recurring bug doesn't get treated as N
        # separate incidents within a single run.
        self._traceback_signatures_seen: Dict[str, int] = {}
        self._resolved_signatures: set = set()   # a fix was applied for this signature this run
        self._declined_signatures: set = set()   # user said no to help for this signature this run


    def _sigint_handler(self, sig, frame):
        """Intercept Ctrl+C during continuous execution to drop into the debugger."""
        if self.continuous:
            self.continuous = False
            cprint("\n[Pulse] Intercepted Ctrl+C. Pausing at next step...")
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
        self._load_config()

    @staticmethod
    def _config_key(key: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")

    def _load_config(self) -> None:
        """Load optional pulse_config.json beside the training script.
        Its presence selects unattended setup; without it, the original
        interactive sign-in, workspace, and provider UI remains active."""
        configured_path = os.environ.get("PULSE_CONFIG", "").strip()
        candidates = [configured_path] if configured_path else []
        if self.script_path:
            directory = os.path.dirname(os.path.abspath(self.script_path))
            candidates.extend((os.path.join(directory, "pulse_config.json"), os.path.join(directory, "pulse_config")))
        candidates.extend((os.path.join(os.getcwd(), "pulse_config.json"), os.path.join(os.getcwd(), "pulse_config")))
        path = next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), None)
        if not path:
            return

        self.non_interactive = True
        self.continuous = True
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                raw = json.load(config_file)
            if not isinstance(raw, dict):
                raise ValueError("top level must be a JSON object")
            self.config = {self._config_key(key): value for key, value in raw.items()}
            self.config_path = path
        except (OSError, ValueError, TypeError) as exc:
            cprint(f"[Pulse] ⚠ Could not read config '{path}': {exc}. Using defaults.", color=_RED)
            return

        self.auto_intervene = self._config_bool("autofix", "auto_fix", default=True)
        self.telemetry_enabled = self._config_bool("telemetry", default=cloud.telemetry_enabled())
        if self._config_has("sensitivity"):
            self._cmd_sensitivity(str(self._config_value("sensitivity")), quiet=True)
        if self._config_has("code", "include_code"):
            self.include_code_default = self._config_bool("code", "include_code", default=True)
        if self._config_has("pdfs", "generate_pdfs"):
            self.generate_pdfs = self._config_bool("pdfs", "generate_pdfs", default=False)

    def _config_has(self, *names: str) -> bool:
        return any(self._config_key(name) in self.config for name in names)

    def _config_value(self, *names: str, default: Any = None) -> Any:
        for name in names:
            key = self._config_key(name)
            if key in self.config:
                return self.config[key]
        return default

    def _config_bool(self, *names: str, default: bool = False) -> bool:
        value = self._config_value(*names, default=default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in ("1", "true", "yes", "on", "enable", "enabled")

    def _config_text(self, *names: str) -> str:
        value = self._config_value(*names, default="")
        return str(value).strip() if value is not None else ""

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
    def _cmd_gputrack(self, var_name: str, quiet: bool = False) -> Optional[str]:
        """Flag a variable as GPU-resident and worth a closer look -- via
        the manual /gputrack command, or via the agent's GPUTRACK:
        directive (quiet=True). GPU stat probes force a device-to-host
        sync, which is real overhead on a training loop, so this only
        ever folds the variable into the slow gpu_probe_interval cadence
        (never the fast per-step one) and the agent gets asked every
        gpu_checkin_interval whether it's still needed -- see
        _maybe_gpu_checkin. Returns the variable name on success, else
        None (mirrors _set_var_state's contract so it composes the same
        way in _apply_directives).
        """
        var_name = var_name.strip()
        if not var_name:
            if not quiet:
                print("Usage: /gputrack <exact_variable_name>")
            return None

        if var_name not in self.watch_locals and var_name not in self.tracked_vars and var_name not in self.discovered:
            if not quiet:
                cprint(f"[Pulse CLI] '{var_name}' is not a known variable.")
            return None

        already_tracked = var_name in self.gpu_tracked_vars
        self.gpu_tracked_vars.add(var_name)
        if var_name not in self.tracked_vars:
            self.tracked_vars.append(var_name)
        self.var_states[var_name] = "lotrack"

        if not quiet:
            print(f"✓ '{var_name}' flagged for GPU tracking. It will probe every {self.gpu_probe_interval / 60:g} min.")
        if not already_tracked:
            cprint(
                f"[Pulse CLI] ⚠ GPU tracking forces a device-to-host sync for '{var_name}' on every "
                "probe -- this will slow training down. Untrack it (/gpuuntrack, or a GPUUNTRACK: "
                "reply) once it's no longer needed.",
                color=_YELLOW,
            )

        if self.team_id:
            try:
                # Note: Requires `update_team_saved_vars` to be defined in `pulse_supabase.py`
                # e.g., supabase.table("Teams").update({"saved_vars": list(...)}).eq("team_id", team_id)
                cloud.update_team_saved_vars(self.team_id, list(self.gpu_tracked_vars))
                if not quiet:
                    print(f"✓ Synced '{var_name}' to team saved_vars.")
            except Exception as exc:
                cprint(f"[Pulse CLI] ⚠ Could not sync to team table: {exc}", color=_RED)

        return var_name

    def _cmd_gpuuntrack(self, var_name: str, quiet: bool = False) -> Optional[str]:
        """Stop GPU-tracking a variable -- via the manual /gpuuntrack
        command, or via the agent's GPUUNTRACK: directive (quiet=True).
        Leaves the variable's normal track/lotrack state untouched; this
        only removes it from the slow GPU probe cadence. `var_name` may
        be 'all' to clear every GPU-tracked variable at once.
        """
        var_name = var_name.strip()
        if not var_name:
            if not quiet:
                print("Usage: /gpuuntrack <exact_variable_name>  (or /gpuuntrack all)")
            return None

        if var_name.lower() == "all":
            n = len(self.gpu_tracked_vars)
            self.gpu_tracked_vars.clear()
            if not quiet:
                print(f"✓ Stopped GPU tracking on {n} variable(s).")
            if self.team_id and n:
                try:
                    cloud.update_team_saved_vars(self.team_id, [])
                except Exception as exc:
                    cprint(f"[Pulse CLI] ⚠ Could not sync to team table: {exc}", color=_RED)
            return "all" if n else None

        if var_name not in self.gpu_tracked_vars:
            if not quiet:
                cprint(f"[Pulse CLI] '{var_name}' is not currently GPU-tracked.")
            return None

        self.gpu_tracked_vars.discard(var_name)
        if not quiet:
            print(f"✓ '{var_name}' is no longer GPU-tracked.")
        if self.team_id:
            try:
                cloud.update_team_saved_vars(self.team_id, list(self.gpu_tracked_vars))
            except Exception as exc:
                cprint(f"[Pulse CLI] ⚠ Could not sync to team table: {exc}", color=_RED)
        return var_name

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
                    cprint(f"[Pulse CLI] '{var_name}' is not currently tracked.")
                return None
            if len(matches) > 1:
                if quiet:
                    # Ambiguous and unattended (agent-triggered) -- don't guess.
                    return None
                print("\nMatching tracked variables:")
                for idx, name in enumerate(matches, 1):
                    print(f"  {idx}) {name}")
                _flush_stdin()
                choice = input("Select a number (Enter to cancel) > ").strip()
                if not choice or not choice.isdigit() or not (0 < int(choice) <= len(matches)):
                    cprint("[Pulse CLI] Cancelled.")
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
                cprint(f"[Pulse CLI] No variables match '{var_name}'.")
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
                _flush_stdin()
                choice = input("Select a number to add (Enter to cancel) > ").strip()
                if not choice:
                    cprint("[Pulse CLI] Add cancelled.")
                    return
                if not choice.isdigit() or not (0 < int(choice) <= len(matches)):
                    cprint("[Pulse CLI] Invalid selection.")
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
        _flush_stdin()
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
                cprint("[Pulse CLI] No variables are currently tracked.")
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
                cprint(f"[Pulse CLI] '{var_name}' is not currently tracked.")
                return

            if len(matches) > 1:
                print("\nMatching tracked variables:")
                for idx, name in enumerate(matches, 1):
                    print(f"  {idx}) {name}")
                _flush_stdin()
                choice = input("Select a number to delete (Enter to cancel) > ").strip()
                if not choice:
                    cprint("[Pulse CLI] Delete cancelled.")
                    return
                if not choice.isdigit() or not (0 < int(choice) <= len(matches)):
                    cprint("[Pulse CLI] Invalid selection.")
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
                cprint(f"[Pulse CLI] No PDF output directory found at '{self.pdf_dir}'.")
                return
            _flush_stdin()
            confirm = input(
                f"Delete ALL heatmap snapshots under '{self.pdf_dir}'? This cannot be undone. (y/n) > "
            ).strip().lower()
            if confirm not in ("y", "yes"):
                cprint("[Pulse CLI] Delete cancelled.")
                return
            try:
                shutil.rmtree(self.pdf_dir)
                print(f"✓ Deleted all heatmap snapshots under '{self.pdf_dir}'.")
            except Exception as exc:
                cprint(f"[Pulse CLI] ⚠ Failed to delete '{self.pdf_dir}': {exc}", color=_RED)
            return

        safe_name = var_name.replace("[", "_").replace("]", "").replace(",", "_")
        target_dir = os.path.join(self.pdf_dir, safe_name)
        if not os.path.isdir(target_dir):
            cprint(f"[Pulse CLI] No heatmap snapshots found for '{var_name}' (looked in '{target_dir}').")
            return

        _flush_stdin()
        confirm = input(
            f"Delete all heatmap snapshots for '{var_name}' in '{target_dir}'? (y/n) > "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            cprint("[Pulse CLI] Delete cancelled.")
            return
        try:
            shutil.rmtree(target_dir)
            print(f"✓ Deleted heatmap snapshots for '{var_name}'.")
        except Exception as exc:
            cprint(f"[Pulse CLI] ⚠ Failed to delete '{target_dir}': {exc}", color=_RED)

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
            cprint("[Pulse CLI] No trackable variables found in scope.")
            return

        var_list = sorted(variables.keys(), key=lambda n: (not _looks_like_loss(n), n))

        self.auto_mode = True
        self.tracked_vars = list(var_list)
        self.var_states = {name: self._default_state_for(name, variables[name]) for name in var_list}
        self.generate_pdfs = False  # opt-in only, via /track + a future setting -- no prompt by default

        self._cloud_setup()
        self._agent_setup()
        self._print_ready_summary()

        # Run silently by default -- no per-step prompt, no dashboard
        # printing -- unless the user explicitly interrupts (Ctrl+C) or
        # something worth flagging happens (a read error, or the
        # auto-intervention check in update() finding real trouble).
        self.continuous = True

    def _print_ready_summary(self) -> None:
        """One clean, glanceable block at the end of setup instead of
        scattering the "are we actually ready" answer across a dozen
        separate messages printed during auth/team/agent setup above --
        the last thing printed before training goes quiet, so it's also
        the thing still visible on screen while everything's running.
        """
        cprint("\n" + "─" * 60)
        cprint("  Pulse is ready")
        if self.user_id:
            cprint(f"  Signed in as {self.email}" + (f"  ·  workspace {self.team_join_code}" if self.team_join_code else "  ·  no workspace"))
        else:
            cprint("  Running locally -- not signed in to Pulse Cloud (/cloud for details)")
        cprint(f"  Agent: {self.agent_provider or 'not configured (/agent to set one up)'}")
        cprint(f"  Auto-fix: {'ON' if self.auto_intervene else 'OFF'}  ·  Sensitivity: {self.sensitivity:.2f}  ·  Tracking {len(self.tracked_vars)} variable{'s' if len(self.tracked_vars) != 1 else ''}")
        if self.tracked_vars:
            cprint(f"  Tracked: {', '.join(self.tracked_vars[:6])}" + (f"  (+{len(self.tracked_vars) - 6} more, see /vars)" if len(self.tracked_vars) > 6 else ""))
        cprint("  Type /help any time for commands, or just ask a question about your run.")
        cprint("─" * 60 + "\n")

    # ------------------------------------------------------------------
    # Cloud: account (Users), team (Teams/join_code), live session sync
    # (Debug_Sessions) against Supabase.
    # ------------------------------------------------------------------

    def _on_cloud_error(self, message: str) -> None:
        # Fires at most once per session (BackgroundSync only reports the
        # first failure) -- cloud sync degrading never blocks training.
        # NOTE this does NOT stop Pulse from continuing to try syncing in
        # the background for the rest of the run (self.debug_session_id
        # is untouched) -- only the repeated warning is suppressed, so a
        # transient network blip doesn't spam the console. If sync is
        # failing on every attempt (e.g. a schema mismatch -- see the
        # uptime_seconds/downtime_seconds bigint-vs-float gotcha fixed in
        # _build_cloud_patch_body, a good example of the kind of error
        # that shows up here), it'll keep failing silently rather than
        # loop-spamming this message.
        cprint(
            f"[Pulse] ⚠ Cloud sync hit an error, continuing training regardless: {message}\n"
            "         (Pulse keeps retrying sync in the background; this warning only shows once per run.)",
            color=_RED,
        )

    def _prompt_for_missing(self, label: str, context: str, default: Optional[str] = None) -> Optional[str]:
        """Anything Pulse can't auto-detect locally (a git commit sha, a
        GPU name, ...) gets asked for once, right here, instead of
        silently falling back to 'unknown'/None. Enter alone accepts
        `default` if one was given (usually the last value cached in
        ~/.pulse/profile.json -- see cloud.load_cached_profile), else
        skips it -- this is never a hard requirement, and never raises if
        stdin isn't interactive. In non-interactive/CI mode, never blocks
        on input at all: returns `default` immediately."""
        if self.non_interactive:
            return default
        try:
            _flush_stdin()
            suffix = f" [{default}]" if default else ""
            val = input(f"[Pulse] {context}. Enter {label} manually (Enter to skip){suffix} > ").strip()
            return val or default
        except (EOFError, KeyboardInterrupt):
            return default

    def _cloud_setup(self) -> None:
        """First-run: sign up or log in (cached to ~/.pulse/credentials.json
        after that, so this is a one-time prompt per machine), then join or
        create a team, then open a Debug_Sessions row for this run. Any
        failure here (offline, bad credentials, etc.) is reported and the
        CLI continues in local-only mode -- cloud sync is never required to
        use Pulse.
        """
        # Resolve the repo directory from the tracked script's own path
        # (set via set_code_text before interactive_setup runs), not the
        # process's cwd -- so `git` commands hit the right repo even when
        # Pulse is launched from elsewhere.
        self._repo_cwd = (
            os.path.dirname(os.path.abspath(self.script_path)) if self.script_path else os.getcwd()
        )

        try:
            self._auth_flow()
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Could not sign in to Pulse cloud, continuing locally: {exc}", color=_RED)
            return

        if not self.user_id:
            return  # user chose to skip auth entirely

        try:
            self._team_flow()
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Team setup failed, continuing without a team: {exc}", color=_RED)

        try:
            # PULSE_AUTO_SESSION_ID/COMMIT_SHA/UPTIME/DOWNTIME: set by
            # _restart_process right before a fix-triggered restart
            # (alongside PULSE_AUTO_PROVIDER/etc). Checked FIRST, before
            # any live git detection or prompting -- this is the "ask at
            # the start, not after a restart" behavior: a resumed process
            # reuses the EXACT sha the original run captured, full stop,
            # rather than re-running `git rev-parse HEAD` and potentially
            # picking up a commit/push that happened on this machine for
            # entirely unrelated reasons while training was running (see
            # _maybe_sync_commit_sha's docstring -- the sha describes
            # what code produced this run, not "whatever HEAD is right
            # now"). Skipping detection/prompting entirely here (not just
            # preferring the carried value) is what actually prevents
            # that: re-detecting-then-overriding would still be wrong the
            # moment detection succeeds and finds something different.
            auto_session_id = os.environ.pop("PULSE_AUTO_SESSION_ID", "").strip()
            auto_sha = os.environ.pop("PULSE_AUTO_COMMIT_SHA", "").strip()

            if auto_session_id:
                sha = auto_sha or "unknown"
            else:
                # Fresh start only (no restart in progress) -- this is the
                # one place a commit sha ever gets asked for or detected.
                sha = self._config_text("sha", "commit", "git_commit_sha") or cloud.current_git_commit_sha(self._repo_cwd)
                if not sha:
                    cached_profile = cloud.load_cached_profile()
                    sha = self._prompt_for_missing(
                        "git commit sha", "Could not detect a git commit sha here (no repo, or git isn't installed)",
                        default=cached_profile.get("last_git_commit_sha"),
                    )
                    if sha:
                        # Cache it regardless of source (typed manually here,
                        # or a default that came from the profile cache) so
                        # the NEXT fresh run has this to fall back to too --
                        # previously only a live `git` detection got cached,
                        # so a manually-entered sha was asked for again next
                        # time even though the user had already supplied it.
                        cloud.save_cached_profile(last_git_commit_sha=sha)
                else:
                    cloud.save_cached_profile(last_git_commit_sha=sha)
                # Debug_Sessions.git_commit_sha is NOT NULL -- every path
                # above can still legitimately end in sha being None (no
                # git, prompt skipped, non-interactive with nothing cached
                # yet), and this is the one place ALL of those paths funnel
                # through before ever reaching create_debug_session/a PATCH
                # body, so this is the right (and only necessary) fallback.
                sha = sha or "unknown"

            # Without carrying debug_session_id/uptime/downtime the same
            # way, every restart would open a BRAND NEW Debug_Sessions row
            # and both counters would silently reset to 0 -- fragmenting
            # one continuous run into disconnected sessions on the
            # dashboard, and making uptime/downtime look like they
            # "aren't logging" right after any auto-fix. Reusing the same
            # row and resuming the in-memory counters where they left off
            # keeps one training run as one session end to end.
            if auto_session_id:
                self.debug_session_id = auto_session_id
                try:
                    self._uptime_seconds = float(os.environ.pop("PULSE_AUTO_UPTIME", "0") or 0)
                except ValueError:
                    self._uptime_seconds = 0.0
                try:
                    self._downtime_seconds = float(os.environ.pop("PULSE_AUTO_DOWNTIME", "0") or 0)
                except ValueError:
                    self._downtime_seconds = 0.0
                self._last_synced_commit_sha = sha
                # uptime/downtime marked dirty as a belt-and-suspenders
                # re-send: the pre-restart flush (_flush_cloud_now in
                # _restart_process) is best-effort and could have failed
                # (offline blip, etc.) even though these env vars still
                # carry the correct in-memory values -- re-sending them
                # now self-heals that rather than trusting the server
                # already has what we think it has. git_commit_sha is
                # deliberately NOT marked dirty here -- it hasn't changed
                # (see above), so there's nothing to re-send.
                self._cloud_dirty_fields.update({"uptime_seconds", "downtime_seconds"})
                # Preload this session's already-synced history so THIS
                # process's flushes extend it instead of quietly replacing
                # it. self._agent_logs/_error_tracebacks/_telemetry/
                # _incidents all start empty in __init__ -- with nothing
                # more done, the first flush from here would PATCH the
                # whole array column (see _build_cloud_patch_body) down to
                # just whatever this process adds, erasing every entry
                # logged before the restart even though it's still sitting
                # in local memory of a process that's gone. Best-effort:
                # if this fetch fails (offline blip, etc.) we still carry
                # on -- worst case is the pre-restart history is briefly
                # at risk on the next flush rather than the run refusing
                # to continue.
                try:
                    existing = cloud.fetch_debug_session(self.debug_session_id)
                except cloud.SupabaseError:
                    existing = None
                if existing:
                    self._agent_logs = cloud.decode_entries(existing.get("agent_logs"))
                    self._error_tracebacks = cloud.decode_entries(existing.get("error_tracebacks"))
                    self._telemetry = cloud.decode_entries(existing.get("telemetry"))
                    self._incidents = cloud.decode_entries(existing.get("incidents"))
                    cprint(
                        f"[Pulse] Loaded {len(self._agent_logs)} agent log(s), {len(self._incidents)} "
                        f"incident(s) from before the restart -- history preserved."
                    )
                else:
                    cprint(
                        "[Pulse] ⚠ Could not load this session's history before the restart -- "
                        "new entries will be appended locally, but the next sync may not include "
                        "everything logged before the restart.",
                        color=_YELLOW,
                    )
                cprint(f"[Pulse] Resumed git commit {sha[:10]}… -- carried over, not re-detected (see /commit to update it manually).")
                cprint(f"[Pulse] Resumed debug session (id={self.debug_session_id[:8]}…) after restart -- uptime/downtime counters carried over.")
            else:
                self.debug_session_id = cloud.create_debug_session(self.team_id, self.user_id, git_commit_sha=sha)
                self._last_synced_commit_sha = sha
                if self.debug_session_id:
                    shown_sha = sha[:10] if sha else "unknown"
                    cprint(f"[Pulse] Debug session started (id={self.debug_session_id}, commit={shown_sha}).")
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Could not start a cloud debug session, continuing locally: {exc}", color=_RED)
            return

        if self.debug_session_id:
            atexit.register(self._flush_cloud_now)
            self._last_cloud_flush = time.monotonic()
            if self._cloud_dirty_fields:
                # Covers the resumed-after-restart case above (git_commit_
                # sha/uptime_seconds/downtime_seconds) when telemetry is
                # off below and wouldn't otherwise force a prompt flush --
                # push whatever's already dirty now rather than waiting on
                # the 10-minute periodic timer.
                self._maybe_flush_cloud(force=True)
            if not self.telemetry_enabled:
                cprint("[Pulse] Telemetry is off (PULSE_TELEMETRY=off) -- skipping environment info collection.")
            else:
                env_info = cloud.collect_environment_info()
                cached_profile = cloud.load_cached_profile()
                if not env_info.get("gpu_name"):
                    manual_gpu = self._prompt_for_missing(
                        "GPU name (e.g. 'NVIDIA A100-SXM4-80GB')", "Could not auto-detect a GPU",
                        default=cached_profile.get("last_gpu_name"),
                    )
                    if manual_gpu:
                        env_info["gpu_name"] = manual_gpu
                        count_str = self._prompt_for_missing(
                            "GPU count", f"Got it -- {manual_gpu}",
                            default=str(cached_profile.get("last_gpu_count") or "") or None,
                        )
                        try:
                            env_info["gpu_count"] = int(count_str) if count_str else 1
                        except ValueError:
                            env_info["gpu_count"] = 1
                        if not env_info.get("cuda_version"):
                            env_info["cuda_version"] = self._prompt_for_missing(
                                "CUDA/ROCm version", "Optional",
                                default=cached_profile.get("last_cuda_version"),
                            )
                else:
                    # Auto-detected cleanly this time -- remember it so a future
                    # run where detection fails (e.g. nvidia-smi transiently
                    # unavailable) still has a sensible manual-prompt default.
                    cloud.save_cached_profile(
                        last_gpu_name=env_info.get("gpu_name"),
                        last_gpu_count=env_info.get("gpu_count"),
                        last_cuda_version=env_info.get("cuda_version"),
                    )
                self._telemetry.append(env_info)  # first telemetry[] entry -- see collect_environment_info
                self._cloud_dirty_fields.add("telemetry")
                self._maybe_flush_cloud(force=True)  # one-off, worth sending promptly
                gpu_desc = (
                    f"{env_info['gpu_count']}x {env_info['gpu_name']}"
                    if env_info.get("gpu_name") else "no GPU detected"
                )
                cprint(
                    f"[Pulse] Environment: Python {env_info['python_version']} | {gpu_desc} | "
                    f"CUDA {env_info.get('cuda_version') or 'n/a'} | "
                    f"{env_info['framework_versions'] or '(no known ML framework detected)'}"
                )

        self._load_history_context()

    def _auth_flow(self) -> None:
        configured_auth = str(self._config_value("auth", "account_action", default="login")).strip().lower()

        configured_email = self._config_text("email", "pulse_email")
        configured_password = self._config_value("password", "pulse_password", default="")
        cached = None if configured_email and configured_password else cloud.load_cached_credentials()
        if cached:
            # Re-establish a real Supabase Auth session from the cached
            # refresh_token BEFORE any of the profiles/Teams calls below --
            # otherwise those calls run as the anon key and come back
            # empty under RLS no matter how valid the cached login is.
            # Best-effort: an older cache with no refresh_token yet, or an
            # expired one, just leaves us on the anon key as before.
            cloud.refresh_session(cached.get("refresh_token"))
            verified = cloud.verify_session_token(cached["user_id"], cached.get("session_token"))
            # verified is True (token matches -- trust it), False (token
            # present but WRONG -- someone/something tampered with or
            # forged this credentials.json; do not trust it, force a
            # fresh login), or None (deployment doesn't support session
            # tokens, or this cache predates them -- fall back to the
            # old TTL-only trust model rather than locking the user out).
            if verified is False:
                cprint("[Pulse] ⚠ Cached login failed verification -- signing in fresh.", color=_RED)
                cloud.clear_cached_credentials()
            else:
                user = cloud.fetch_user(cached["user_id"])
                if user:
                    self.user_id = user["id"]
                    self.email = user["email"]
                    self.team_id = cached.get("team_id")
                    cprint(f"[Pulse] Signed in as {self.email}.")
                    return
                # Cached user no longer exists server-side -- clear cache and fall through
                cloud.clear_cached_credentials()

        if self.non_interactive:
            # CI mode never blocks on a TTY prompt. PULSE_email/
            # PULSE_PASSWORD are how a pipeline authenticates. If neither
            # is set, Pulse just runs in local (no-cloud) mode instead of
            # hanging the job.
            env_user = configured_email or os.environ.get("PULSE_email", "").strip()
            env_pass = configured_password or os.environ.get("PULSE_PASSWORD", "")
            if env_user and env_pass:
                try:
                    if configured_auth in ("signup", "sign_up", "register", "create"):
                        if not self._config_bool("accept_tos", "tos_accepted", default=False):
                            raise cloud.SupabaseError("signup requires 'Accept ToS': true in pulse_config.json")
                        user = cloud.sign_up(env_user, env_pass)
                        cloud.record_tos_acceptance(user["id"])
                        cprint(f"[Pulse] ✓ Account created and signed in as {env_user} (non-interactive).")
                    else:
                        user = cloud.log_in(env_user, env_pass)
                        cprint(f"[Pulse] ✓ Signed in as {env_user} (non-interactive).")
                    self.user_id = user["id"]
                    self.email = env_user
                    token = cloud.attach_session_token(self.user_id)
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id, session_token=token)
                except cloud.SupabaseEmailConfirmationRequired:
                    # Account created, but this deployment requires email
                    # confirmation before a session can be issued -- an
                    # unattended job can't click that link, so there's
                    # nothing more to do here this run. Not a "login
                    # failed" -- just fall through to local (no-cloud) mode
                    # for this run and let a human confirm and re-run.
                    cprint(
                        f"[Pulse] Confirm your account: check {env_user}'s email for a confirmation link, "
                        "then log in. Continuing this run in local (no-cloud) mode.",
                    )
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Non-interactive login failed ({exc}). Continuing in local (no-cloud) mode.", color=_RED)
            else:
                cprint(
                    "[Pulse] Non-interactive mode and no cached login -- set PULSE_email/PULSE_PASSWORD "
                    "to enable cloud sync in CI, or ignore this to run local-only.",
                )
            return

        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            _flush_stdin()
            cprint("\n--- Pulse Cloud Authentication ---")
            resp = input(
                "[Pulse] (s)ign up [default] / (l)og in / (r)ecover account > "
            ).strip().lower() or "s"

            if resp in ("s", "signup", "sign up"):
                _flush_stdin()
                # Retry only the field that actually failed (email or
                # password) instead of bouncing back to the top-level
                # "sign up / log in / recover" menu on every typo -- that
                # used to mean a single password mismatch cost you a full
                # re-type of the email address too.
                while True:
                    email = input("email (e.g. name@example.com, min. 8 characters) > ").strip()
                    if not email:
                        cprint("[Pulse] email cannot be blank.", color=_RED)
                        continue
                    if not cloud.is_valid_email(email):
                        cprint("[Pulse] enter a valid email address (min. 8 characters), e.g. name@example.com.", color=_RED)
                        continue
                    break
                while True:
                    password = getpass.getpass("Password (min. 8 characters) > ")
                    if len(password) < 8:
                        cprint("[Pulse] Password must be at least 8 characters.", color=_RED)
                        continue
                    confirm_password = getpass.getpass("Confirm password > ")
                    if password != confirm_password:
                        # A typo here with no confirmation step means signing up
                        # with a password the user doesn't actually know -- and
                        # with no email on file yet at this point, there'd be no
                        # way back in. Re-prompt for password only, not email.
                        cprint("[Pulse] Passwords didn't match. Let's try again.", color=_RED)
                        continue
                    break
                accepted = self._prompt_tos_acceptance()
                if not accepted:
                    cprint("[Pulse] Sign up cancelled -- acceptance is required to create an account.")
                    continue

                try:
                    user = cloud.sign_up(email, password)
                    cprint(f"[Pulse] ✓ Account created. Signed in as {email}.")
                    self.user_id = user["id"]
                    self.email = email
                    cloud.record_tos_acceptance(self.user_id)
                    token = cloud.attach_session_token(self.user_id)
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id, session_token=token)
                    self._show_recovery_code(cloud.attach_recovery_code(self.user_id))
                    return
                except cloud.SupabaseEmailConfirmationRequired:
                    # The account WAS created -- this isn't a failure. The
                    # OLD behavior here just printed a message and fell
                    # back to the top-level menu, which meant: leave the
                    # CLI, go confirm in email, come back, remember to
                    # pick "log in" instead of "sign up", and retype the
                    # email+password you just typed 10 seconds ago. That's
                    # exactly the kind of drop-off point that loses a
                    # just-converted cold-email lead. Instead, stay in
                    # this flow and retry with the SAME credentials
                    # already in memory -- the only thing left to do is
                    # click the link and hit Enter.
                    cprint("[Pulse] Almost there -- check your email for a confirmation link and click it.")
                    while True:
                        _flush_stdin()
                        resp2 = input(
                            "Press Enter once confirmed to continue (or type 'skip' to do this later) > "
                        ).strip().lower()
                        if resp2 == "skip":
                            cprint("[Pulse] No problem -- run Pulse again and choose (l)og in once you've confirmed.")
                            break
                        try:
                            user = cloud.log_in(email, password)
                        except cloud.SupabaseError as exc2:
                            cprint(f"[Pulse] Not confirmed yet ({exc2}). Check your email and try again.", color=_YELLOW)
                            continue
                        cprint(f"[Pulse] ✓ Confirmed! Signed in as {email}.")
                        self.user_id = user["id"]
                        self.email = email
                        token = cloud.attach_session_token(self.user_id)
                        cloud.save_cached_credentials(self.user_id, self.email, self.team_id, session_token=token)
                        return
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Sign up failed: {exc}", color=_RED)

            elif resp in ("l", "login", "log in"):
                _flush_stdin()
                email = input("email > ").strip()
                if not email:
                    cprint("[Pulse] email cannot be blank.", color=_RED)
                    continue
                password = getpass.getpass("Password > ")
                if not password:
                    cprint("[Pulse] Password cannot be blank.", color=_RED)
                    continue

                try:
                    user = cloud.log_in(email, password)
                    cprint(f"[Pulse] ✓ Signed in as {email}.")
                    self.user_id = user["id"]
                    self.email = email
                    token = cloud.attach_session_token(self.user_id)
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id, session_token=token)
                    return
                except cloud.SupabaseError as exc:
                    attempts += 1
                    cprint(f"[Pulse] ⚠ Login failed ({attempts}/{max_attempts}): {exc}", color=_RED)

            elif resp in ("r", "recover", "recover account"):
                _flush_stdin()
                cprint(
                    "[Pulse] Account recovery uses the one-time recovery code shown when you signed up "
                    "(or the last time you recovered/changed your password)."
                )
                email = input("email > ").strip()
                recovery_code = input("Recovery code (e.g. 7F3K-9QRT-2LXP) > ").strip()
                if not email or not recovery_code:
                    cprint("[Pulse] Both email and recovery code are required.", color=_RED)
                    continue
                new_password = getpass.getpass("New password (min. 8 characters) > ")
                if len(new_password) < 8:
                    cprint("[Pulse] Password must be at least 8 characters.", color=_RED)
                    continue
                confirm_new_password = getpass.getpass("Confirm new password > ")
                if new_password != confirm_new_password:
                    cprint("[Pulse] Passwords didn't match. Let's try again.", color=_RED)
                    continue
                try:
                    new_recovery_code = cloud.recover_account(email, recovery_code, new_password)
                    cprint("[Pulse] ✓ Password reset. Signing you in...")
                    user = cloud.log_in(email, new_password)
                    self.user_id = user["id"]
                    self.email = email
                    token = cloud.attach_session_token(self.user_id)
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id, session_token=token)
                    self._show_recovery_code(new_recovery_code)
                    return
                except cloud.SupabaseError as exc:
                    attempts += 1
                    cprint(f"[Pulse] ⚠ Recovery failed ({attempts}/{max_attempts}): {exc}", color=_RED)

            else:
                cprint("[Pulse] Invalid option. Please choose (s)ign up, (l)og in, or (r)ecover account.")

        # Reached after 3 failed login attempts
        cprint(
            "\n[Pulse] ⚠ Maximum login attempts (3) exceeded. Switching to local (no-cloud) mode.",
            color=_RED
        )
        self.user_id = None
        self.email = None

    def _prompt_tos_acceptance(self) -> bool:
        """Show a short summary of the Pulse license up front (full text
        is PULSE_LICENSE_TEXT, mirrored from the LICENSE file in the
        Pulse GitHub repo, and is always one keystroke away via 'full')
        and require typing 'yes' to accept before an account is created.

        The full ~50-line license used to be printed unconditionally
        before every sign-up -- legally fine, but it's exactly the kind
        of wall of text that makes someone who just clicked through from
        a cold email bail before ever seeing the product. A short,
        scannable summary gets the same acceptance (still gated on
        actually typing 'yes', still timestamped server-side by
        cloud.record_tos_acceptance right after this returns True) without
        that drop-off risk, while anyone who wants the full text still
        gets it by typing 'full'.

        PULSE_PRIVACY_URL, if set, is shown alongside it for a separate
        privacy policy (this license covers software use, not data
        handling) -- point that env var at your actual Privacy Policy
        before relying on this for anything.
        """
        privacy_url = os.environ.get("PULSE_PRIVACY_URL", "(set PULSE_PRIVACY_URL to link your Privacy Policy here)")
        tos_url = os.environ.get("PULSE_TOS_URL")
        print(
            "\nQuick terms before you create an account:\n"
            "  • Pulse is proprietary software, licensed (not sold) for your own use.\n"
            "  • No redistributing, reselling, or reverse-engineering it, and no using it to build a competing product.\n"
            "  • Provided as-is, no warranty -- pricing/paid tiers may be introduced for future versions.\n"
        )
        if tos_url:
            print(f"Full license: {tos_url}")
        print(f"Privacy Policy: {privacy_url}\n")
        _flush_stdin()
        resp = input("Type 'yes' to accept (or 'full' to read the complete license first) > ").strip().lower()
        if resp == "full":
            print(f"\n{PULSE_LICENSE_TEXT}")
            _flush_stdin()
            resp = input("Type 'yes' to accept and create your account > ").strip().lower()
        return resp == "yes"

    def _show_recovery_code(self, code: Optional[str]) -> None:
        if not code:
            return  # this deployment doesn't have the recovery_code_hash column yet -- see attach_recovery_code
        cprint("\n[Pulse] ⚠ Save this recovery code somewhere safe (a password manager) -- it's the only way")
        cprint("         back into your account if you forget your password, and it's shown ONLY this once:")
        print(f"\n    {code}\n")
    
    @property
    def is_team_admin(self) -> bool:
        return bool(self.user_id) and self.user_id in self.team_admin_ids

    @staticmethod
    def _describe_workspace(team: Dict[str, Any], user_id: Optional[str]) -> str:
        """Human-readable label for a workspace menu entry -- Teams has no
        dedicated 'name' column, so this falls back to the repo (if set)
        or a generic label built from the join code, plus an admin tag.
        """
        repo = team.get("repo")
        if repo and repo != "unknown":
            label = repo.rstrip("/").rsplit("/", 1)[-1]
            if label.endswith(".git"):
                label = label[:-4]
        elif len(team.get("members") or []) <= 1:
            label = "Personal workspace"
        else:
            label = f"Team {team.get('join_code', '?')}"
        role = " (admin)" if user_id and user_id in (team.get("admin_ids") or []) else ""
        return f"{label}{role}  --  join code {team.get('join_code', '?')}"
    
    def _prompt_and_set_repo(self) -> None:
        """Prompt the user for a repository URL if the workspace repo is missing."""
        if not self.team_id or self.non_interactive:
            return
        
        detected = cloud.git_remote_url(self._repo_cwd)
        _flush_stdin()
        prompt = (
            f"GitHub repo URL (Enter to use detected: {detected}) > "
            if detected else
            "GitHub repo URL (optional, Enter to skip) > "
        )
        repo_input = input(prompt).strip()
        repo_url = repo_input or detected
        
        if repo_url:
            try:
                cloud.update_team_repo(self.team_id, repo_url)
                cprint(f"[Pulse] ✓ Updated workspace repository to: {repo_url}")
            except cloud.SupabaseError as exc:
                cprint(f"[Pulse] ⚠ Could not update workspace repository: {exc}", color=_RED)
    def _team_flow(self) -> None:
        """Pick (or create/join) the workspace for this run. Always shows
        the picker -- even when credentials were resumed from a cached
        login and even when the user only belongs to one workspace --
        rather than silently reusing whichever team happened to be cached
        or came back first, since people can belong to several workspaces
        (a personal one, one per model/project, etc.) and the one they
        want can change run to run. The previously-used workspace (from
        cached credentials, if any) is offered as the Enter-key default
        for convenience, not auto-selected.
        """
        if not self.user_id:
            return

        cached_team_id = self.team_id
        self.team_id = None
        self.team_join_code = None
        self.team_admin_ids = []

        existing = cloud.find_teams_for_user(self.user_id)

        if self.non_interactive:
            # Auto-pick without prompting: the previously-used workspace if
            # still available, else the user's only workspace, else stay
            # local-only (no team) rather than block on a picker.
            workspace = self._config_value("workspace", "team", "team_id", default=None)
            workspace_action = self._config_text("workspace_action", "team_action").lower()
            workspace_options = workspace if isinstance(workspace, dict) else {}
            if workspace_options:
                workspace_action = str(workspace_options.get("action", workspace_action)).strip().lower()
                workspace = workspace_options.get("name") or workspace_options.get("join_code") or workspace_options.get("team_id")
            if workspace_action in ("create", "new", "signup", "sign_up"):
                repo = self._config_text("repo", "repository")
                if not repo and workspace_options:
                    repo = str(workspace_options.get("repo") or "").strip()
                try:
                    team = cloud.create_team(self.user_id, repo=repo or None, cwd=self._repo_cwd)
                    self.team_id = team["team_id"]
                    self.team_join_code = team.get("join_code")
                    self.team_admin_ids = list(team.get("admin_ids") or [])
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
                    cprint(f"[Pulse] Non-interactive mode -- created workspace (join code: {self.team_join_code}).")
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not create workspace: {exc}. Continuing without a team.", color=_RED)
                return
            pick = None
            if workspace is not None:
                if isinstance(workspace, int) or (isinstance(workspace, str) and workspace.strip().isdigit()):
                    index = int(workspace) - 1
                    if 0 <= index < len(existing):
                        pick = existing[index]
                else:
                    wanted = str(workspace).strip().lower()
                    matches = [team for team in existing if wanted in self._describe_workspace(team, self.user_id).lower() or wanted == str(team.get("team_id", "")).lower() or wanted == str(team.get("join_code", "")).lower()]
                    if len(matches) == 1:
                        pick = matches[0]
            if not pick:
                pick = next((t for t in existing if t["team_id"] == cached_team_id), None)
            if not pick and len(existing) == 1:
                pick = existing[0]
            if not pick and workspace_action in ("join", "join_code") and isinstance(workspace, str):
                try:
                    pick = cloud.join_team(workspace.strip().upper(), self.user_id)
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not join workspace: {exc}. Continuing without a team.", color=_RED)
            if pick:
                self.team_id = pick["team_id"]
                self.team_join_code = pick.get("join_code")
                self.team_admin_ids = list(pick.get("admin_ids") or [])
                if self._resumed_after_restart:
                    cprint(f"[Pulse] Resumed workspace (join code: {self.team_join_code}) -- auto-filled after restart.")
                else:
                    cprint(f"[Pulse] Non-interactive mode -- using workspace (join code: {self.team_join_code}).")
                cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
            else:
                cprint("[Pulse] Non-interactive mode and no unambiguous workspace to pick -- continuing without a team.")
            return

        # A brand-new sign-up (or anyone who currently belongs to zero
        # workspaces) has nothing to actually pick between here -- the
        # old behavior forced everyone through this menu regardless, with
        # no Enter-key default, meaning a just-signed-up trial user's very
        # next required action was "type c, then optionally type a GitHub
        # repo URL" before they could do anything else. Skip straight to
        # a silently auto-created personal workspace instead; joining a
        # teammate's workspace with a code is one command away (/repo can
        # set a repo on it later, too), but creating your own shouldn't
        # need a prompt at all when there's nothing else it could be.
        if not existing:
            try:
                team = cloud.create_team(self.user_id, repo=cloud.git_remote_url(self._repo_cwd), cwd=self._repo_cwd)
                self.team_id = team["team_id"]
                self.team_join_code = team.get("join_code")
                self.team_admin_ids = list(team.get("admin_ids") or [])
                cprint(f"[Pulse] ✓ Workspace ready (join code: {self.team_join_code}) -- share this to invite teammates, or /repo to set a repo.")
                cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
            except cloud.SupabaseError as exc:
                cprint(f"[Pulse] ⚠ Could not create a workspace ({exc}) -- continuing without a team. Try /repo or restart to pick one.", color=_RED)
            return

        while True:
            _flush_stdin()
            cprint("\n--- Pulse Workspace ---")
            default_choice = None
            for i, team in enumerate(existing, start=1):
                is_default = team["team_id"] == cached_team_id
                if is_default:
                    default_choice = str(i)
                marker = "  [last used]" if is_default else ""
                print(f"  {i}) {self._describe_workspace(team, self.user_id)}{marker}")
            print("  j) Join a new Workspace")
            print("  c) Create a new Workspace")

            suffix = f" (Enter = {default_choice})" if default_choice else ""
            resp = input(f"[Pulse] Select a workspace{suffix} > ").strip().lower()
            if not resp and default_choice:
                resp = default_choice

            if resp.isdigit() and existing and 1 <= int(resp) <= len(existing):
                team = existing[int(resp) - 1]
                self.team_id = team["team_id"]
                self.team_join_code = team.get("join_code")
                self.team_admin_ids = list(team.get("admin_ids") or [])
                cprint(f"[Pulse] Using workspace (join code: {self.team_join_code}).")
                if not team.get("repo") or team.get("repo") == "unknown":
                    self._prompt_and_set_repo()
                cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
                return

            if resp in ("c", "create"):
                detected = cloud.git_remote_url(self._repo_cwd)
                _flush_stdin()
                prompt = (
                    f"GitHub repo URL (Enter to use detected: {detected}) > "
                    if detected else
                    "GitHub repo URL (optional, Enter to skip) > "
                )
                repo_input = input(prompt).strip()
                try:
                    team = cloud.create_team(self.user_id, repo=repo_input or detected)
                    self.team_id = team["team_id"]
                    self.team_join_code = team["join_code"]
                    self.team_admin_ids = list(team.get("admin_ids") or [])
                    cprint(f"[Pulse] ✓ Workspace created. Share this join code with teammates: {self.team_join_code}")
                    cprint(f"[Pulse]   Repo: {team.get('repo')}")
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
                    return
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not create workspace: {exc}", color=_RED)

            elif resp in ("j", "join"):
                _flush_stdin()
                code = input("Join code > ").strip().upper()
                if not code:
                    cprint("[Pulse] Join code cannot be blank.", color=_RED)
                    continue
                try:
                    team = cloud.join_team(code, self.user_id)
                    self.team_id = team["team_id"]
                    self.team_join_code = team.get("join_code")
                    self.team_admin_ids = list(team.get("admin_ids") or [])
                    cprint(f"[Pulse] ✓ Joined workspace {self.team_join_code}.")
                    if not team.get("repo") or team.get("repo") == "unknown":
                        self._prompt_and_set_repo()
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
                    return
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not join workspace: {exc}", color=_RED)

            else:
                cprint("[Pulse] Invalid option.")

    def _cmd_admin(self, arg: str) -> None:
        """/admin add <email> | /admin remove <email> | /admin list
        -- manage the current workspace's admins. Admins must already be
        team members (enforced server-side in add_team_admin); only
        existing admins (or the workspace owner) may grant/revoke admin
        rights.
        """
        if not self.team_id:
            cprint("[Pulse] No workspace selected.")
            return

        parts = arg.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"

        if sub == "list":
            if not self.team_admin_ids:
                cprint("[Pulse] No admins set for this workspace yet.")
                return
            cprint("[Pulse] Admins:")
            for uid in self.team_admin_ids:
                user = cloud.fetch_user(uid)
                name = user["email"] if user else uid
                print(f"    - {name}")
            return

        if sub not in ("add", "remove") or len(parts) < 2:
            cprint("Usage: /admin add <email> | /admin remove <email> | /admin list")
            return

        if not self.is_team_admin:
            cprint("[Pulse] ⚠ Only an existing workspace admin can add or remove admins.", color=_RED)
            return

        email = parts[1].strip()
        user = cloud.find_user_by_email(email)
        if not user:
            cprint(f"[Pulse] No user found with email '{email}'.", color=_RED)
            return

        try:
            if sub == "add":
                team = cloud.add_team_admin(self.team_id, user["id"])
                self.team_admin_ids = list(team.get("admin_ids") or [])
                cprint(f"[Pulse] ✓ '{email}' is now an admin of this workspace.")
            else:
                team = cloud.remove_team_admin(self.team_id, user["id"])
                self.team_admin_ids = list(team.get("admin_ids") or [])
                cprint(f"[Pulse] ✓ '{email}' is no longer an admin of this workspace.")
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ {exc}", color=_RED)

    def _cmd_password(self, arg: str) -> None:
        """/password -- change your account password. Requires your
        current password even though you're already signed in (a
        walked-away or leaked cached session shouldn't be enough to take
        over the account) and revokes every other cached login/device
        when it succeeds -- if that's not what you want (e.g. you have
        several machines logged in and don't suspect anything's wrong),
        just sign back in on each afterward; this is a deliberate
        "kick everyone else off" safety behavior, not a bug."""
        if not self.user_id:
            cprint("[Pulse] Sign in to Pulse Cloud first.")
            return
        if self.non_interactive:
            cprint("[Pulse] /password requires an interactive session.")
            return
        current = getpass.getpass("Current password > ")
        new = getpass.getpass("New password (min. 8 characters) > ")
        confirm = getpass.getpass("Confirm new password > ")
        if new != confirm:
            cprint("[Pulse] New passwords didn't match. Nothing changed.", color=_RED)
            return
        try:
            cloud.change_password(self.user_id, current, new)
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Could not change password: {exc}", color=_RED)
            return
        cprint("[Pulse] ✓ Password changed. Every other signed-in device/CI token using PULSE_email/PULSE_PASSWORD will need to sign in again.")
        # Re-attach a fresh session token for THIS run (change_password just
        # revoked the old one along with everyone else's).
        token = cloud.attach_session_token(self.user_id)
        cloud.save_cached_credentials(self.user_id, self.email, self.team_id, session_token=token)
        self._show_recovery_code(cloud.attach_recovery_code(self.user_id))

    def _cmd_recover(self, arg: str) -> None:
        """Reset a forgotten password with the one-time recovery code."""
        if self.non_interactive:
            cprint("[Pulse] /recover requires an interactive session.")
            return
        _flush_stdin()
        email = input("email > ").strip()
        recovery_code = input("Recovery code (e.g. 7F3K-9QRT-2LXP) > ").strip()
        new_password = getpass.getpass("New password (min. 8 characters) > ")
        confirm_password = getpass.getpass("Confirm new password > ")
        if new_password != confirm_password:
            cprint("[Pulse] Passwords didn't match. Nothing changed.", color=_RED)
            return
        try:
            new_recovery_code = cloud.recover_account(email, recovery_code, new_password)
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Password recovery failed: {exc}", color=_RED)
            return
        cprint("[Pulse] ✓ Password reset. Existing cached sessions were revoked.")
        self._show_recovery_code(new_recovery_code)

    def _cmd_deleteaccount(self, arg: str) -> None:
        """/deleteaccount -- permanently delete your Pulse account:
        removes you from every workspace's member/admin lists, revokes
        all API tokens and cached sessions, and deletes your Users row.
        Does not delete your teams' training history (see
        cloud.delete_account's docstring for why) -- if a workspace has
        no other members left after this, an admin should also delete
        the workspace itself if it should stop existing.
        """
        if not self.user_id:
            cprint("[Pulse] Sign in to Pulse Cloud first.")
            return
        if self.non_interactive:
            cprint("[Pulse] /deleteaccount requires an interactive session (this should never happen unattended).")
            return
        cprint(
            f"[Pulse] ⚠ This permanently deletes the account '{self.email}' -- it cannot be undone. "
            "Your teams' training history is kept (see /deleteaccount's help), but you'll lose access to it."
        )
        _flush_stdin()
        confirm = input(f"Type the email '{self.email}' to confirm > ").strip()
        if confirm != self.email:
            cprint("[Pulse] Confirmation didn't match. Nothing deleted.")
            return
        password = getpass.getpass("Password > ")
        try:
            cloud.delete_account(self.user_id, password)
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Could not delete account: {exc}", color=_RED)
            return
        cloud.clear_cached_credentials()
        cprint("[Pulse] ✓ Account deleted. Continuing this run in local (no-cloud) mode.")
        self.user_id = None
        self.email = None
        self.team_id = None
        self.debug_session_id = None

    def _cmd_logout(self, arg: str) -> None:
        """/logout -- sign out of the account currently in use for this
        run, then immediately offers to sign back in (as the same account
        or a different one) without having to restart Pulse. Revokes the
        server-side session token (so the cached credentials.json on this
        machine can't silently log back in on its own) and clears the
        local cache. If a new account signs in, this also re-runs the
        workspace picker and opens a fresh cloud debug session under that
        identity -- the old session is flushed and left as-is on the
        dashboard rather than mixing two identities' data into one row.
        """
        if not self.user_id:
            cprint("[Pulse] Not signed in to Pulse Cloud -- nothing to log out of.")
            return
        if self.non_interactive:
            cprint("[Pulse] /logout requires an interactive session.")
            return

        old_email = self.email

        # Flush anything still pending for the CURRENT session/identity
        # before switching -- otherwise a dirty field written after
        # logout could still get attributed to the old debug session.
        self._flush_cloud_now()

        cloud.revoke_session_token(self.user_id)  # best-effort; never raises
        cloud.clear_cached_credentials()
        self.user_id = None
        self.email = None
        self.team_id = None
        self.team_admin_ids = []
        self.debug_session_id = None
        cprint(f"[Pulse] ✓ Logged out of {old_email}.")

        try:
            self._auth_flow()
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Could not sign in to Pulse cloud, continuing locally: {exc}", color=_RED)
            return

        if not self.user_id:
            cprint("[Pulse] Continuing this run in local (no-cloud) mode.")
            return

        try:
            self._team_flow()
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Team setup failed, continuing without a team: {exc}", color=_RED)

        try:
            sha = self._last_synced_commit_sha or cloud.current_git_commit_sha(self._repo_cwd) or "unknown"
            self.debug_session_id = cloud.create_debug_session(self.team_id, self.user_id, git_commit_sha=sha)
            self._last_synced_commit_sha = sha
            if self.debug_session_id:
                cprint(f"[Pulse] Debug session started (id={self.debug_session_id}, commit={sha[:10] if sha != 'unknown' else 'unknown'}).")
                atexit.register(self._flush_cloud_now)
                self._last_cloud_flush = time.monotonic()
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Could not start a cloud debug session, continuing locally: {exc}", color=_RED)

    def _cmd_webhook(self, arg: str) -> None:
        """/webhook set <url> | /webhook test | /webhook off | /webhook
        -- a Slack-incoming-webhook-compatible URL that gets a message
        posted to it whenever _log_incident records something a human
        would want to know about right away (crash, auto-intervention,
        restart failure), so a team doesn't have to keep the dashboard
        open to notice a run went bad. Session-local only for now (an
        env var / this command, not yet persisted server-side per team --
        see the TODO on _team_webhook_url for the natural next step)."""
        arg = arg.strip()
        sub = arg.split(None, 1)[0].lower() if arg else ""

        if sub == "set" and len(arg.split(None, 1)) > 1:
            url = arg.split(None, 1)[1].strip()
            if not url.startswith("http"):
                cprint("[Pulse] That doesn't look like a URL. Usage: /webhook set <url>")
                return
            self._incident_webhook_url = url
            cprint("[Pulse] ✓ Webhook set for this session. Incidents will be posted there. Try /webhook test.")
            return

        if sub == "off":
            self._incident_webhook_url = None
            cprint("[Pulse] ✓ Webhook disabled for this session.")
            return

        if sub == "test":
            if not self._incident_webhook_url:
                cprint("[Pulse] No webhook set. Usage: /webhook set <url>")
                return
            ok = self._post_incident_webhook("test", "This is a test alert from Pulse -- if you can see this, it's working.")
            cprint("[Pulse] ✓ Test alert sent." if ok else "[Pulse] ⚠ Test alert failed to send -- check the URL.")
            return

        state = self._incident_webhook_url or "(not set -- also checks PULSE_WEBHOOK_URL)"
        cprint(f"[Pulse] Webhook: {state}\nUsage: /webhook set <url> | /webhook test | /webhook off")

    def _post_incident_webhook(self, kind: str, summary: str) -> bool:
        """Best-effort POST in Slack's incoming-webhook body shape
        ({"text": ...}) -- also what Discord/Mattermost/most alerting
        tools that accept "a webhook URL" expect, so this works without
        per-service branching. Never raises, never blocks the training
        loop on a slow/dead webhook endpoint (short timeout, swallowed
        errors) -- an alert that fails to send should never be the thing
        that hangs a training run."""
        url = self._incident_webhook_url or os.environ.get("PULSE_WEBHOOK_URL", "").strip()
        if not url:
            return False
        script_name = os.path.basename(self.script_path) if self.script_path else "a training run"
        text = f":rotating_light: *Pulse alert* -- `{script_name}` [{kind}]\n{summary}"
        try:
            import urllib.request as _urlreq
            body = json.dumps({"text": text}).encode("utf-8")
            req = _urlreq.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
            with _urlreq.urlopen(req, timeout=5) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False


        """Ask for a GitHub URL for the current team's `repo` column,
        offering the git-detected remote (if any) as the default."""
        if not self.team_id:
            return
        detected = cloud.git_remote_url(self._repo_cwd)
        _flush_stdin()
        prompt = (
            f"GitHub repo URL for this team (Enter to use detected: {detected}) > "
            if detected else
            "GitHub repo URL for this team (optional, Enter to skip) > "
        )
        repo_input = input(prompt).strip()
        repo = repo_input or detected
        if repo:
            try:
                cloud.update_team_repo(self.team_id, repo)
                cprint(f"[Pulse]   Repo: {repo}")
            except cloud.SupabaseError as exc:
                cprint(f"[Pulse] ⚠ Could not save repo URL: {exc}", color=_RED)

    def _cmd_repo(self, arg: str) -> None:
        """/repo [url] -- show or set the current team's GitHub repo URL."""
        if not self.team_id:
            cprint("[Pulse] No team is active -- join or create one first (restart Pulse to do so).")
            return
        if not arg:
            self._prompt_and_set_repo()
            return
        try:
            cloud.update_team_repo(self.team_id, arg)
            cprint(f"[Pulse] ✓ Repo set to: {arg}")
        except cloud.SupabaseError as exc:
            cprint(f"[Pulse] ⚠ Could not save repo URL: {exc}", color=_RED)

    def log_traceback(self, tb_text: str) -> None:
        """Called whenever an uncaught exception crashes the user's script
        (see pulse.py's _install_cli_excepthook) so the crash is captured
        in Debug_Sessions.error_tracebacks even if the user declines AI
        help for it. Crashes are rare and high-value, so this flushes
        promptly rather than waiting for the batch timer."""
        if not self.debug_session_id:
            return
        self._error_tracebacks.append(tb_text)
        self._cloud_dirty_fields.add("error_tracebacks")
        self._maybe_flush_cloud(force=True)

    def _sync_agent_turn(
        self,
        question: str,
        answer: str,
        traceback_signature: Optional[str] = None,
        fix_applied: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.debug_session_id:
            return
        entry = {"t": time.time(), "question": question, "answer": answer}
        # Structured fix info (not just the prose answer) so a future
        # session can find and re-apply this exact fix -- see
        # _load_history_context / offer_known_fix.
        if traceback_signature:
            entry["traceback_signature"] = traceback_signature
        if fix_applied:
            entry["fix_applied"] = fix_applied
        self._agent_logs.append(entry)
        self._cloud_dirty_fields.add("agent_logs")
        # Agent turns are low-frequency and high-value (unlike telemetry
        # ticks) -- flush promptly instead of waiting up to 10 minutes.
        self._maybe_flush_cloud(force=True)

    def _finalize_agent_downtime(self) -> None:
        """Idempotent: turns the currently-pending agent timer (set right
        before calling self.ask_agent() in update()'s auto-intervene
        block) into an accumulated downtime total plus one logged
        incident, exactly once.

        This has to be called from TWO places: right after ask_agent()
        returns normally in update() (the common case -- diagnosis with
        no fix, or a fix that couldn't restart), AND at the very top of
        _restart_process() (because a *successful* restart replaces this
        whole process via a synchronous subprocess.run + sys.exit without
        ever returning control back up to update() -- that's the only
        guaranteed chokepoint before the process can vanish). The
        pending-timer guard (cleared to None once used) means whichever
        of the two call sites runs first does the actual work and the
        other becomes a harmless no-op.
        """
        if self._pending_agent_start_ts is None:
            return
        downtime = time.monotonic() - self._pending_agent_start_ts
        problem = self._pending_agent_problem or "(unknown)"
        self._pending_agent_start_ts = None
        self._pending_agent_problem = None

        self._downtime_seconds += downtime
        self._cloud_dirty_fields.add("downtime_seconds")
        self._log_incident(
            "auto_intervention", problem,
            downtime_seconds=round(downtime, 2),
            fix_applied=bool(self._fix_applied_this_turn),
        )

    _ALERT_WORTHY_KINDS = {"crash", "restart_failed", "auto_intervention"}

    def _log_incident(self, kind: str, summary: str, **extra: Any) -> None:
        """Record one discrete, dashboard-visible event: what happened,
        when, and any structured detail (extra kwargs). Incidents are the
        thing an admin/member/owner looking at the web dashboard actually
        wants to see -- crashes, auto-interventions, fixes applied,
        reverts, restart failures -- as opposed to the continuous
        uptime/downtime counters, which say how much but not what.
        Low-frequency and high-value like agent_logs/error_tracebacks, so
        this force-flushes immediately rather than waiting for the batch
        timer.

        The webhook alert below fires independently of Pulse Cloud sync
        (even with no debug session / not logged in at all) -- it's a
        direct CLI-to-webhook POST, nothing routes through Pulse's own
        servers, so it works the same for a team running fully local for
        confidentiality (see _warn_if_cloud_sync_defeats_local_privacy)
        as for one with cloud sync on.
        """
        if kind in self._ALERT_WORTHY_KINDS:
            self._post_incident_webhook(kind, summary)

        if not self.debug_session_id:
            return
        entry: Dict[str, Any] = {"t": time.time(), "kind": kind, "summary": summary}
        entry.update(extra)
        self._incidents.append(entry)
        self._cloud_dirty_fields.add("incidents")
        self._maybe_flush_cloud(force=True)

    def _build_cloud_patch_body(self) -> Dict[str, Any]:
        """Compress and package every dirty field into ONE PATCH body,
        clearing the dirty set. Each array entry is individually
        compressed (see pulse_supabase.encode_entry) -- the array itself
        still has to be resent in full each flush (PostgREST has no array
        "append" verb), but batching flushes infrequently plus compressing
        each entry keeps that resend cheap instead of the dominant cost.
        """
        body: Dict[str, Any] = {}
        if "agent_logs" in self._cloud_dirty_fields:
            body["agent_logs"] = [cloud.encode_entry(e) for e in self._agent_logs]
        if "error_tracebacks" in self._cloud_dirty_fields:
            body["error_tracebacks"] = [cloud.encode_entry(e) for e in self._error_tracebacks]
        if "telemetry" in self._cloud_dirty_fields:
            body["telemetry"] = [cloud.encode_entry(e) for e in self._telemetry]
        if "incidents" in self._cloud_dirty_fields:
            body["incidents"] = [cloud.encode_entry(e) for e in self._incidents]
        if "uptime_seconds" in self._cloud_dirty_fields:
            # Debug_Sessions.uptime_seconds is a bigint column -- sending a
            # float (e.g. round(x, 2) -> 15.19) fails with a "not a
            # bigInt"-style Postgres error and, per the disable-on-error
            # handling in _maybe_flush_cloud, was disabling cloud sync
            # for the rest of the run over what's really just a rounding
            # choice. int() (truncating, not rounding, to match downtime
            # below and avoid ever reporting a hair more uptime than
            # actually elapsed) fixes it at the source.
            body["uptime_seconds"] = int(self._uptime_seconds)
        if "downtime_seconds" in self._cloud_dirty_fields:
            body["downtime_seconds"] = int(self._downtime_seconds)
        if "git_commit_sha" in self._cloud_dirty_fields:
            body["git_commit_sha"] = self._last_synced_commit_sha
        self._cloud_dirty_fields.clear()
        return body

    def _maybe_flush_cloud(self, force: bool = False) -> None:
        """The only place that actually enqueues a Debug_Sessions PATCH.
        Batches everything dirty into one request, sent either because
        `force` was requested (a crash, an agent turn, a new commit) or
        because cloud_flush_interval has elapsed since the last flush
        (the steady 10-minute drip for telemetry)."""
        if not self.debug_session_id or not self._cloud_dirty_fields:
            return
        now = time.monotonic()
        if not force and (now - self._last_cloud_flush) < self.cloud_flush_interval:
            return
        self._last_cloud_flush = now
        self._cloud_sync.submit(self.debug_session_id, self._build_cloud_patch_body())

    def _flush_cloud_now(self) -> None:
        """Synchronous, best-effort final flush for moments the background
        thread might not get to run before the process dies: right before
        a fix-triggered restart (_restart_process), and registered with
        `atexit` for normal/Ctrl-C shutdown."""
        if not self.debug_session_id:
            return
        body = self._build_cloud_patch_body() if self._cloud_dirty_fields else {}
        # Queue this behind every earlier snapshot instead of making a direct
        # PATCH. A direct PATCH can complete before an older background PATCH
        # and let that stale array overwrite newer issues/fixes.
        done = self._cloud_sync.submit(self.debug_session_id, body, wait=True)
        if done is not None:
            done.wait(timeout=5.0)

    def _maybe_collect_telemetry(
        self,
        scalar_lines: "List[tuple[str, Optional[float]]]",
        matrix_lines: "List[tuple[str, Dict[str, Any], Any, str]]",
    ) -> None:
        if not self.debug_session_id:
            return
        now = time.monotonic()
        if now - self._last_telemetry_sample < self.telemetry_sample_interval:
            return
        self._last_telemetry_sample = now

        snapshot: Dict[str, Any] = {"t": time.time(), "step": self.step}
        for name, val in scalar_lines:
            snapshot[name] = val
        for name, stats, _val, _state in matrix_lines:
            snapshot[name] = {
                "mean": stats.get("mean"), "min": stats.get("min"), "max": stats.get("max"),
                "nan": stats.get("nan"), "inf": stats.get("inf"),
            }

        self._telemetry.append(snapshot)
        self._cloud_dirty_fields.add("telemetry")
        self._maybe_flush_cloud()  # respects the 10-minute batch timer -- this is the noisy one

    def _maybe_sync_commit_sha(self) -> None:
        """DEPRECATED as an automatic per-step check -- kept only as the
        implementation behind /commit (a manual, explicit "yes, actually
        update it now" action). It used to run on a timer during every
        training run and silently overwrite Debug_Sessions.git_commit_sha
        the moment `git rev-parse HEAD` changed -- which sounds right
        until you notice HEAD can change for reasons that have nothing to
        do with the code this process is actually running: a commit or
        push from another terminal/session, a CI job, a pre-commit hook,
        anyone else on the machine. The sha recorded for a run should
        describe what code produced it, not "whatever HEAD happens to be
        right now" -- so this no longer runs automatically. See
        _cloud_setup for where a session's sha is captured once (and
        _restart_process/PULSE_AUTO_COMMIT_SHA for why an auto-fix
        restart carries that same original value forward instead of
        re-detecting it -- a Pulse-applied fix never touches git itself,
        so there's no real commit for a restart to legitimately pick up
        anyway).
        """
        sha = cloud.current_git_commit_sha(self._repo_cwd)
        if sha and sha != self._last_synced_commit_sha:
            self._last_synced_commit_sha = sha
            self._cloud_dirty_fields.add("git_commit_sha")
            self._maybe_flush_cloud(force=True)
            return True
        return False

    def _cmd_commit(self, arg: str) -> None:
        """/commit -- explicitly refresh the debug session's recorded git
        commit to the repo's current HEAD. Manual and on-demand only
        (see _maybe_sync_commit_sha's docstring for why this isn't
        automatic) -- use this right after you actually commit your own
        changes mid-run, if you want the dashboard to reflect it."""
        if not self.debug_session_id:
            cprint("[Pulse] Not signed in to Pulse Cloud -- nothing to update.")
            return
        if self._maybe_sync_commit_sha():
            cprint(f"[Pulse] ✓ Updated to current HEAD: {self._last_synced_commit_sha[:10]}…")
        else:
            cprint("[Pulse] No change detected (or no git repo here).")

    def _cmd_help(self, arg: str) -> None:
        """/help -- lists every command, grouped by what it's for. The
        main prompt only shows the three most-used actions (Enter, /c,
        and this) to stay readable -- everything else lives here instead
        of being crammed into one unreadable hint line."""
        sections = [
            ("Basics", [
                ("Enter", "advance one training step"),
                ("/c", "switch to continuous (don't stop at every step)"),
                ("<question>", "just type a question -- no slash needed -- to ask the agent about the run"),
                ("/help", "this list"),
            ]),
            ("Tracking variables", [
                ("/add <var>", "track a new variable (asks what kind)"),
                ("/track <var>", "promote a variable to full tracking"),
                ("/lotrack <var>", "track less often (lighter-weight)"),
                ("/gputrack <var> / /gpuuntrack <var>", "track a GPU-resident tensor directly (slower, more precise)"),
                ("/delete <var> / /deletepdf <var>", "stop tracking / delete saved heatmap PDFs"),
                ("/vars / /tracked", "list all seen variables / currently tracked ones"),
            ]),
            ("Auto-fix & sensitivity", [
                ("/autofix on|off", f"toggle auto-intervention (currently {'ON' if self.auto_intervene else 'OFF'})"),
                ("/sensitivity [value]", f"how eagerly spikes/plateaus/oscillation trigger it (currently {self.sensitivity:.2f} -- run with no argument for details)"),
                ("/revert [id]", "undo a fix Pulse applied (/log to see fix history first)"),
                ("/commit", "manually refresh the recorded git commit to current HEAD"),
            ]),
            ("Agent & account", [
                ("/agent", "switch AI provider/model (cloud or local -- Ollama/LM Studio/etc.)"),
                ("/code", "toggle whether questions include your training code"),
                ("/password", "change your Pulse account password"),
                    ("/recover", "reset a forgotten password with a recovery code"),
                ("/logout", "sign out, then sign back in as the same or a different account"),
            ]),
            ("Cloud & team", [
                ("/cloud", "show sign-in, workspace, and sync status at a glance"),
                ("/cloud flush", "force an immediate sync instead of waiting for the batch timer"),
                ("/telemetry on|off", f"environment-info collection (currently {'ON' if self.telemetry_enabled else 'OFF'})"),
                ("/webhook set|test|off", "get a Slack-style alert on crashes/auto-interventions"),
                ("/admin add|remove|list", "manage workspace admins (admins only)"),
                ("/repo [url]", "set/show the repo this run is associated with"),
                ("/log", "show recorded fixes/incidents for this session"),
            ]),
        ]
        print()
        for title, rows in sections:
            print(f"  {title}")
            for cmd, desc in rows:
                print(f"    {cmd:<38} {desc}")
            print()
        print("  Anything not a slash command is sent to the AI agent as a question.\n")

    def _print_cloud_status(self) -> None:
        if not self.user_id:
            print("[Pulse] Not signed in -- sessions are local only. Restart Pulse to sign in.")
            return
        print(f"[Pulse] Signed in as {self.email} (user_id={self.user_id})")
        if self.team_id:
            admin_tag = "  [admin]" if self.is_team_admin else ""
            print(f"  Team: {self.team_id}  (join code: {self.team_join_code}){admin_tag}")
        else:
            print("  Team: none")
        if self.debug_session_id:
            print(
                f"  Debug session: {self.debug_session_id}  "
                f"[{len(self._agent_logs)} agent turns, {len(self._error_tracebacks)} tracebacks, "
                f"{len(self._telemetry)} telemetry snapshots recorded]"
            )
            print(
                f"  Uptime: {self._uptime_seconds / 60:.1f} min   "
                f"Downtime (agentic work): {self._downtime_seconds / 60:.1f} min   "
                f"Incidents: {len(self._incidents)}"
            )
            print(f"  Commit: {self._last_synced_commit_sha or 'unknown'}")
            print(f"  Known fixes loaded from history: {len(self._known_fixes)}")
            if self._cloud_dirty_fields:
                since = time.monotonic() - self._last_cloud_flush
                print(
                    f"  Pending sync: {sorted(self._cloud_dirty_fields)} "
                    f"(next batch flush in ~{max(0, self.cloud_flush_interval - since):.0f}s, "
                    f"or immediately on a crash/commit/agent turn)"
                )
            else:
                print("  Pending sync: none -- everything is flushed")
        else:
            print("  Debug session: none (cloud sync unavailable this run)")

        agent_desc = self.agent_provider or "(none configured -- /agent to set one up)"
        print(
            f"\n  Agent: {agent_desc}   Auto-fix: {'ON' if self.auto_intervene else 'OFF'}   "
            f"Sensitivity: {self.sensitivity:.2f} (see /sensitivity for details)"
        )
        print(
            f"  Telemetry: {'ON' if self.telemetry_enabled else 'OFF'}   "
            f"Webhook: {'set' if (self._incident_webhook_url or os.environ.get('PULSE_WEBHOOK_URL')) else 'not set'}   "
            f"Mode: {'non-interactive' if self.non_interactive else 'interactive'}"
        )

    def _load_history_context(self) -> None:
        """Pull previous Debug_Sessions for this team (or this user, if no
        team) and (1) build a short summary injected into the agent's
        context so it has memory of past crashes/fixes on this repo, and
        (2) index every fix that was actually applied by the signature of
        the traceback it fixed, so a recurring bug can be re-fixed
        directly (see _handle_crash) instead of re-diagnosed from
        scratch every time.
        """
        try:
            sessions = cloud.fetch_recent_sessions(self.team_id, self.user_id)
        except cloud.SupabaseError:
            return

        sessions = [s for s in sessions if s.get("id") != self.debug_session_id]
        if not sessions:
            return

        # Every entry was compressed on the way in (see
        # pulse_supabase.encode_entry) -- decode before anything below
        # reads into them. decode_entry passes legacy plain entries
        # through untouched, so older rows stay readable too.
        for s in sessions:
            s["agent_logs"] = cloud.decode_entries(s.get("agent_logs"))
            s["telemetry"] = cloud.decode_entries(s.get("telemetry"))
            s["error_tracebacks"] = cloud.decode_entries(s.get("error_tracebacks"))

        def _last_activity(s: Dict[str, Any]) -> float:
            # Prefer the real created_at column (now present); fall back to
            # timestamps Pulse itself stamps into agent_logs/telemetry for
            # older rows or if the fetch fell back to an unordered query.
            created = s.get("created_at")
            if created:
                try:
                    return datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                except Exception:
                    pass
            times = [
                e["t"] for e in (s.get("agent_logs") or []) if isinstance(e, dict) and "t" in e
            ] + [
                e["t"] for e in (s.get("telemetry") or []) if isinstance(e, dict) and "t" in e
            ]
            return max(times) if times else 0.0

        # fetch_recent_sessions already orders server-side by created_at
        # when that column is available; this re-sort is just a safety net
        # (and the only ordering available at all in the fallback path).
        sessions.sort(key=_last_activity, reverse=True)

        for s in sessions:
            for entry in (s.get("agent_logs") or []):
                if not isinstance(entry, dict):
                    continue
                sig = entry.get("traceback_signature")
                fix = entry.get("fix_applied")
                if sig and fix and sig not in self._known_fixes:
                    self._known_fixes[sig] = fix  # newest session wins (sessions is newest-first)

        lines = []
        for s in sessions[:5]:
            sha = (s.get("git_commit_sha") or "unknown")[:10]
            tbs = s.get("error_tracebacks") or []
            fixed = sum(
                1 for e in (s.get("agent_logs") or [])
                if isinstance(e, dict) and e.get("fix_applied")
            )
            if not tbs and not fixed:
                continue
            lines.append(f"- commit {sha}: {len(tbs)} crash(es) logged, {fixed} fix(es) applied")

        if lines:
            self._history_context = (
                "Known history from previous Pulse debugging sessions on this repo/team "
                "(most recent first):\n" + "\n".join(lines) + "\n"
                "If the current issue matches one of these, say so explicitly instead of "
                "re-deriving the same diagnosis from scratch."
            )
            cprint(f"[Pulse] Loaded context from {len(sessions)} previous session(s) "
                   f"({len(self._known_fixes)} known fix(es)).")

    @staticmethod
    def _traceback_signature(tb_text: str) -> str:
        """A stable-ish fingerprint for 'is this the same bug' -- the
        exception type/message plus the last couple of frames, so the
        same bug re-raising every training step (or across separate runs)
        is recognized as one incident instead of N.
        """
        lines = [l for l in tb_text.strip().splitlines() if l.strip()]
        tail = "\n".join(lines[-4:])
        return hashlib.sha1(tail.encode("utf-8")).hexdigest()[:12]

    def handle_crash(self, tb_text: str) -> Optional[str]:
        """Called from pulse.py's excepthook for every uncaught exception.
        Always logs the traceback to Debug_Sessions.error_tracebacks, then
        does the dedup bookkeeping so the *same* bug recurring doesn't get
        treated -- or re-asked about -- as a brand new incident each time.
        Returns the traceback's signature for the caller to use when
        deciding what to do next (offer a known fix, ask the agent, or
        just note it's the same one as before).
        """
        self.log_traceback(tb_text)
        sig = self._traceback_signature(tb_text)
        self._traceback_signatures_seen[sig] = self._traceback_signatures_seen.get(sig, 0) + 1
        self._log_incident(
            "crash", tb_text.strip().splitlines()[-1] if tb_text.strip() else "(empty traceback)",
            signature=sig, occurrence=self._traceback_signatures_seen[sig],
        )
        return sig

    def offer_known_fix(self, signature: str) -> bool:
        """If a fix for this exact bug (by signature) was already applied
        -- in this run or a previous cloud session -- offer to re-apply it
        directly instead of re-running the full diagnose pipeline. Returns
        True if a fix was offered and applied (caller should skip asking
        the agent from scratch)."""
        fix = self._known_fixes.get(signature)
        if not fix:
            return False
        seen = self._traceback_signatures_seen.get(signature, 1)
        if signature in self._resolved_signatures and seen > 1:
            cprint(
                "[Pulse] ⚠ This is the same bug Pulse already fixed once this run -- "
                "the earlier fix may not have taken effect (or this process is still "
                "running the old code in memory).",
                color=_RED,
            )
        explanation = fix.get("explanation") or "(no explanation recorded)"
        cprint(f"[Pulse] This looks like a bug Pulse has already fixed before: {explanation}")
        if self.auto_intervene:
            cprint("[Pulse] Auto-fix is on -- re-applying it automatically.")
            resp = "y"
        else:
            try:
                _flush_stdin()
                resp = input("Re-apply that fix now? (y/n) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
        if resp not in ("y", "yes"):
            return False

        self._apply_code_fix(fix)
        self._last_applied_fix = fix
        self._resolved_signatures.add(signature)
        self._sync_agent_turn(
            question="(known-fix reapplication, no agent call)",
            answer=f"Re-applied previously recorded fix: {explanation}",
            traceback_signature=signature,
            fix_applied=fix,
        )
        if self._fix_applied_this_turn:
            self._restart_process()  # does not return
        return True

    def _select_agent_provider_and_key(self, initial: bool = False) -> bool:
        """Prompt the user to pick an AI provider and API key (or, for a
        local/self-hosted provider, an API base URL + model name instead
        of a key -- see the "local" branch below).

        When `initial` is True (first-time setup), pressing Enter skips agent
        setup entirely and leaves the agent disabled. When False (switching
        mid-run), pressing Enter cancels the switch and keeps whatever
        agent/key is already active.

        Returns True if the active agent/key changed as a result of this call.

        Non-interactive/CI mode never reaches an input() prompt here: for a
        cloud provider, set PULSE_PROVIDER to a provider name (e.g.
        "Claude") plus that provider's own API key env var (e.g.
        ANTHROPIC_API_KEY -- see PROVIDERS[...]["env_key"]); for a local
        provider, set PULSE_PROVIDER to it and PULSE_LOCAL_MODEL (+
        optionally PULSE_LOCAL_API_BASE). If what's needed is missing,
        the agent is simply left disabled for the run rather than
        blocking on a prompt that will never be answered.
        """
        names = list(PROVIDERS.keys())

        if self.non_interactive:
            if not initial:
                cprint("[Pulse] Non-interactive mode -- use PULSE_PROVIDER (+ key or PULSE_LOCAL_MODEL) to switch agents.")
                return False
            configured_agent = self._config_value("agent", "provider", default=None)
            want = str(configured_agent).strip() if configured_agent is not None else os.environ.get("PULSE_PROVIDER", "").strip()
            if want.isdigit() and 1 <= int(want) <= len(names):
                candidates = [names[int(want) - 1]]
            elif want:
                candidates = [want]
            else:
                candidates = names
            for name in candidates:
                match = next((n for n in names if n.lower() == name.lower()), None)
                if not match:
                    partial = [n for n in names if name.lower() in n.lower()]
                    match = partial[0] if len(partial) == 1 else None
                if not match:
                    continue
                info = PROVIDERS[match]
                if info.get("local"):
                    model_name = self._config_text("local_model", "model") or os.environ.get("PULSE_LOCAL_MODEL", "").strip()
                    if not model_name:
                        continue  # can't use a local provider without knowing what model it's serving
                    api_base = self._config_text("local_api_base", "api_base") or os.environ.get("PULSE_LOCAL_API_BASE", "").strip() or info["default_api_base"]
                    self._set_local_agent(match, api_base, model_name)
                    cprint(f"[Pulse] Non-interactive mode -- agent set to {match} ({model_name} @ {api_base}).")
                    return True
                env_var = info["env_key"]
                key = self._config_text("api_key", "key") or os.environ.get(env_var, "").strip()
                if key:
                    self.agent_provider = match
                    self.agent_key = key
                    self.agent_model_string = None
                    self.agent_api_base = None
                    self.agent_history = []
                    cprint(f"[Pulse] Non-interactive mode -- agent set to {match} (via {env_var}).")
                    os.environ[env_var] = key
                    cloud.save_cached_profile(agent_provider=match, agent_env_key=env_var)
                    return True
            cprint(
                "[Pulse] Non-interactive mode and no usable provider found in the environment -- "
                "AI agent disabled for this run. Set PULSE_PROVIDER and either its matching *_API_KEY, "
                "or PULSE_LOCAL_MODEL for a local provider, to enable it."
            )
            self.agent_provider = None
            return False

        print("\nSelect an agent/provider:")
        cached_provider = cloud.load_cached_profile().get("agent_provider") if not self.agent_provider else None
        for i, name in enumerate(names, 1):
            local_tag = "  [local -- no data leaves this machine]" if PROVIDERS[name].get("local") else ""
            marker = "  (current)" if name == self.agent_provider else ("  (last used)" if name == cached_provider else "")
            print(f"  {i}) {name}{local_tag}{marker}")

        prompt = (
            "\nAgent number (or Enter to skip) > "
            if initial
            else "\nAgent number (or Enter to cancel) > "
        )

        while True:
            _flush_stdin()
            raw = input(prompt).strip()
            if not raw:
                if initial:
                    cprint("[Pulse CLI] AI agent disabled for this run.")
                else:
                    cprint("[Pulse CLI] Agent switch cancelled.")
                return False
            if raw.isdigit() and 1 <= int(raw) <= len(names):
                chosen = names[int(raw) - 1]
                break
            matches = [n for n in names if raw.lower() in n.lower()]
            if len(matches) == 1:
                chosen = matches[0]
                break
            cprint("[Pulse CLI] Pick a valid agent number or provider name.")

        info = PROVIDERS[chosen]

        if info.get("local"):
            print(f"\n✓ Agent selected: {chosen} -- runs on your own infrastructure, no API key needed.")
            _flush_stdin()
            cached_profile = cloud.load_cached_profile()
            default_base = cached_profile.get("agent_api_base") or info["default_api_base"]
            api_base = input(f"Server URL [{default_base}] > ").strip() or default_base
            default_model = cached_profile.get("agent_model_name") or ""
            model_prompt = f"Model name ({info['model_hint']})"
            model_prompt += f" [{default_model}] > " if default_model else " > "
            model_name = input(model_prompt).strip() or default_model
            if not model_name:
                cprint("[Pulse CLI] No model name entered. Agent unchanged.")
                if initial:
                    self.agent_provider = None
                return False
            self._set_local_agent(chosen, api_base, model_name)
            if initial:
                print(f"✓ Connected to {chosen} at {api_base}.")
            else:
                print(f"✓ Switched to {chosen}. Conversation history reset for the new agent.")
            self._warn_if_cloud_sync_defeats_local_privacy()
            return True

        env_var = info["env_key"]
        existing = os.environ.get(env_var, "").strip()

        print(f"\n✓ Agent selected: {chosen}")
        if existing:
            _flush_stdin()
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
            cprint("[Pulse CLI] No API key entered. Agent unchanged.")
            if initial:
                self.agent_provider = None
            return False

        self.agent_provider = chosen
        self.agent_key = key
        self.agent_model_string = None
        self.agent_api_base = None
        os.environ[env_var] = key
        # Switching providers mid-conversation would send one provider's turns
        # to another; start a fresh history so context isn't mixed across models.
        self.agent_history = []
        # Remember the provider (never the key itself -- see save_cached_profile's
        # docstring) so next run's prompt can highlight it as the likely pick.
        cloud.save_cached_profile(agent_provider=chosen, agent_env_key=env_var)

        if initial:
            print(f"✓ API key accepted for {self.agent_provider}.")
        else:
            print(f"✓ Switched to {self.agent_provider}. Conversation history reset for the new agent.")
        return True

    def _set_local_agent(self, provider_name: str, api_base: str, model_name: str) -> None:
        """Wire up a local/self-hosted provider: compose the litellm model
        string, no API key required (self.agent_key gets a non-secret
        sentinel purely so the existing "is an agent configured" guards
        -- `if not self.agent_provider or not self.agent_key` -- still
        pass; _call_model never sends it anywhere, see its api_key= line).
        """
        info = PROVIDERS[provider_name]
        self.agent_provider = provider_name
        self.agent_key = "local"
        self.agent_api_base = api_base
        self.agent_model_string = f"{info['litellm_prefix']}/{model_name}"
        self.agent_history = []
        cloud.save_cached_profile(
            agent_provider=provider_name, agent_env_key=None,
            agent_api_base=api_base, agent_model_name=model_name,
        )

    def _warn_if_cloud_sync_defeats_local_privacy(self) -> None:
        """The whole point of picking a local provider is usually that
        code/data can't leave the machine -- but Pulse Cloud sync (if
        the user is logged in) still uploads agent Q&A and tracebacks to
        Supabase, which very likely contains exactly that code/data. A
        local LLM alone doesn't give you that guarantee back, so say so
        plainly rather than let someone assume it does."""
        if self.debug_session_id and self.user_id:
            cprint(
                "[Pulse] ⚠ Note: you're signed in to Pulse Cloud, which still syncs agent Q&A and "
                "tracebacks (likely containing your code) to Pulse's servers for the team dashboard, "
                "independent of which LLM answers the questions. For fully local operation, run "
                "without signing in (or set PULSE_NONINTERACTIVE=1 with no PULSE_email/PASSWORD).",
                color=_YELLOW,
            )

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
        auto_api_base = os.environ.pop("PULSE_AUTO_API_BASE", None)
        auto_model = os.environ.pop("PULSE_AUTO_MODEL", None)
        if auto_provider and auto_provider in PROVIDERS:
            info = PROVIDERS[auto_provider]
            if info.get("local"):
                if auto_api_base and auto_model:
                    self._set_local_agent(auto_provider, auto_api_base, auto_model)
                    cprint(f"[Pulse] Resumed with agent {auto_provider} (auto-filled after restart).")
            else:
                env_var = info["env_key"]
                key = os.environ.get(env_var, "").strip()
                if key:
                    self.agent_provider = auto_provider
                    self.agent_key = key
                    self.agent_model_string = None
                    self.agent_api_base = None
                    self.agent_history = []
                    cprint(f"[Pulse] Resumed with agent {auto_provider} (auto-filled after restart).")

        if not self.agent_provider and not self._select_agent_provider_and_key(initial=True):
            return

        if self.pending_startup_error:
            cprint("[Pulse] Your dry run raised an exception:", color=_RED)
            cprint(self.pending_startup_error, color=_RED)
            question = (
                f"My script just crashed with this uncaught exception:\n{self.pending_startup_error}\n"
                "Please diagnose the root cause and, if you can, fix it."
            )
            self.ask_agent(question, include_code=True)
            self.pending_startup_error = None

    def _resolve_restart_interpreter(self) -> Optional[str]:
        """Find a Python interpreter that actually exists on disk to
        restart with. sys.executable is normally fine, but can go stale
        (e.g. an env that was recreated/moved since this process started,
        or -- a known Windows quirk -- the Microsoft Store Python shim
        misreporting the interpreter path). No assumptions about venvs
        here since clients may not have one: just fall back to whatever
        'python'/'python3' resolves to on PATH.
        """
        if sys.executable and os.path.isfile(sys.executable):
            return sys.executable
        return shutil.which("python") or shutil.which("python3")

    def _restart_process(self) -> None:
        """Restart the whole process so the training loop actually runs
        the fixed code. Uses a synchronous subprocess.run (not Popen +
        exit, and not os.execv) deliberately: Popen spawns a *second*
        process on top of this one and exits this one out from under it,
        which -- especially on Windows -- leaves the terminal's stdin in
        a broken state for the new process (input stops being detected
        correctly). A synchronous subprocess.run keeps this process (and
        its already-attached stdin/stdout) alive as the parent for the
        whole child run, which is the one approach that has stayed
        reliable across macOS/Linux/Windows -- os.execv is POSIX-only and
        unavailable on Windows at all, and was tried and reverted here for
        that reason; do not switch back to it without re-verifying Windows
        stdin behavior.

        IMPORTANT: these runs are mostly unsupervised, so a restart that
        can't actually happen must never take the process down anyway.
        Every failure path below prints a warning and returns, leaving
        the current process running the old (already-on-disk-fixed, but
        still-in-memory-old) code rather than exiting -- the fix is still
        saved to disk and a real restart (manual, or Pulse's next
        successful auto-intervention) will still pick it up. This only
        calls sys.exit() once a replacement process has actually been
        spawned to take over.
        """
        # Finalize the pending agent-downtime timer (if any) and log the
        # incident now, before doing anything else below -- a successful
        # restart replaces this process via subprocess.run + sys.exit
        # without returning, so this is the last guaranteed chokepoint to
        # record what just happened. No-op if update() already finalized
        # it (e.g. this was
        # called from offer_known_fix's reapply path instead, which never
        # set a pending timer to begin with).
        self._finalize_agent_downtime()

        # self.script_path is the file that called auto_track() (see
        # pulse.py's auto_track -> _start_cli_tracker -> set_code_text),
        # i.e. the actual training script -- restart that, not whatever
        # sys.modules['__main__'] happens to report (fragile: unset or
        # wrong under some launchers/IDEs).
        script_path = self.script_path
        if not script_path or not os.path.isfile(script_path):
            cprint(
                f"[Pulse] ⚠ Could not find the training script to restart (looked for: "
                f"{script_path!r}). The fix is saved, but Pulse is staying up and continuing "
                "with the current run rather than killing an unsupervised job -- restart the "
                "script yourself whenever you can to pick it up.",
                color=_RED,
            )
            self._log_incident("restart_failed", f"Could not find training script to restart: {script_path!r}")
            return

        python_exe = self._resolve_restart_interpreter()
        if not python_exe:
            cprint(
                f"[Pulse] ⚠ Could not find a working Python interpreter to restart with "
                f"(sys.executable was {sys.executable!r}, which no longer exists, and no "
                "'python'/'python3' was found on PATH either). The fix is saved, but Pulse is "
                "staying up and continuing with the current run rather than killing an "
                "unsupervised job -- restart the script yourself whenever you can to pick it up.",
                color=_RED,
            )
            self._log_incident(
                "restart_failed",
                f"No working Python interpreter found (sys.executable={sys.executable!r} was stale)",
            )
            return
        if python_exe != sys.executable:
            cprint(
                f"[Pulse] ⚠ sys.executable ({sys.executable!r}) doesn't exist -- "
                f"restarting with {python_exe!r} instead.",
                color=_YELLOW,
            )

        # Mark the replacement process as an unattended auto-fix resume.
        # __init__ consumes this flag before workspace/provider setup so the
        # restarted run never stops for interactive input.
        os.environ["PULSE_AUTO_RESTART"] = "1"

        if self.agent_provider:
            os.environ["PULSE_AUTO_PROVIDER"] = self.agent_provider
            if self.agent_api_base:
                os.environ["PULSE_AUTO_API_BASE"] = self.agent_api_base
            if self.agent_model_string:
                # Strip the litellm prefix back off -- _set_local_agent adds
                # it back on the other side of the restart.
                os.environ["PULSE_AUTO_MODEL"] = self.agent_model_string.split("/", 1)[-1]

        if self.debug_session_id:
            # Carry the SAME Debug_Sessions row (and where its counters
            # currently stand) across the restart -- see _cloud_setup's
            # PULSE_AUTO_SESSION_ID branch. Without this the resumed
            # process opens a brand new row starting both counters at 0,
            # which is what made uptime/downtime look like they "weren't
            # logging": every auto-fix fragmented one continuous run into
            # a new, mostly-empty session.
            os.environ["PULSE_AUTO_SESSION_ID"] = self.debug_session_id
            os.environ["PULSE_AUTO_UPTIME"] = str(self._uptime_seconds)
            os.environ["PULSE_AUTO_DOWNTIME"] = str(self._downtime_seconds)
            # Carry the EXACT sha this run was already using too -- see
            # _cloud_setup's "ask at the start, not after a restart"
            # ordering. Not re-detecting it here (only forwarding
            # whatever's already recorded) is what keeps an unrelated
            # commit/push elsewhere on the machine from ever being able
            # to change what a resumed run is attributed to.
            os.environ["PULSE_AUTO_COMMIT_SHA"] = self._last_synced_commit_sha or ""
            # Explicitly dirty regardless of whether something else already
            # marked them -- we're about to snapshot+carry these exact
            # values, so the server-side row should reflect them too.
            self._cloud_dirty_fields.update({"uptime_seconds", "downtime_seconds"})

        # Best-effort final push of whatever's dirty (including the
        # up-to-the-moment uptime/downtime above) so the server-side row
        # reflects reality even if the child process never gets back here
        # (crashes immediately, etc.) -- the resumed process re-sends
        # these same values again on startup regardless (belt and
        # suspenders, see _cloud_setup), so this isn't the only copy.
        self._flush_cloud_now()
        cprint("\n[Pulse] Fix applied -- restarting the training loop to pick it up...\n")
        sys.stdout.flush()

        script_path = os.path.abspath(script_path)

        if os.name == 'nt':
            # Legacy cmd.exe (not Windows Terminal) doesn't interpret raw
            # ANSI escapes by default -- writing them there just prints
            # garbage characters instead of clearing anything. `cls`
            # below is the actually-reliable way to clear on Windows.
            try:
                subprocess.run(["cls"], shell=True)
            except Exception:
                pass  # best-effort screen clear -- never worth failing a restart over
        else:
            sys.stdout.write("\033c\033[0m")
            sys.stdout.flush()
            for clear_cmd in (["stty", "sane"], ["clear"]):
                try:
                    subprocess.run(clear_cmd, timeout=5)
                except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
                    pass  # minimal/containerized environments may not have these on PATH

        # Use the interpreter _resolve_restart_interpreter actually found
        # working (sys.executable itself if it still exists, otherwise
        # whatever fallback it located) -- NOT sys.executable directly,
        # which may be the stale path this whole resolution step exists
        # to route around.
        argv = [python_exe, script_path] + sys.argv[1:]

        # Retry the restart itself instead of ever falling back to "keep
        # running the old, already-in-memory process" on a bad exit code.
        # A nonzero exit here almost always means the replacement process
        # crashed immediately (e.g. the agent's fix didn't fully fix it,
        # or introduced a new bug) -- silently resuming the OLD code path
        # used to look like a safe fallback, but for an unsupervised run
        # it just means the SAME already-crashed-once code keeps limping
        # along, or the loop quietly stalls, with nobody watching to
        # notice. Retrying the launch instead gives a transient problem
        # (e.g. a port/file briefly locked by the process that's still
        # exiting, a flaky import) a real chance to clear, and gives the
        # agent's fix -- which is already saved to disk -- more chances to
        # actually take effect. Attempts are capped (not infinite) so a
        # deterministically-broken script can't spin forever burning
        # compute/cost unattended; MAX_RESTART_ATTEMPTS is the one knob to
        # raise if that cap is ever too low for a given job.
        MAX_RESTART_ATTEMPTS = 5
        RETRY_BACKOFF_SECONDS = 3  # multiplied by attempt number, capped below

        attempt = 0
        while True:
            attempt += 1
            try:
                # Synchronous run keeps stdin attached and handles spaces in paths correctly on Windows
                result = subprocess.run(argv)
            except Exception as exc:
                cprint(f"[Pulse] ⚠ Restart attempt {attempt}/{MAX_RESTART_ATTEMPTS} failed to launch ({exc}).", color=_RED)
                if attempt >= MAX_RESTART_ATTEMPTS:
                    cprint(
                        f"[Pulse] ⚠ Giving up after {MAX_RESTART_ATTEMPTS} restart attempts -- "
                        "continuing current run with the old in-memory code.",
                        color=_RED,
                    )
                    self._log_incident("restart_failed", f"Could not launch replacement process after {attempt} attempts: {exc}")
                    return
                time.sleep(min(RETRY_BACKOFF_SECONDS * attempt, 30))
                continue

            if result.returncode == 0:
                sys.exit(0)

            message = f"Replacement training process exited with code {result.returncode} (attempt {attempt}/{MAX_RESTART_ATTEMPTS})"
            if attempt >= MAX_RESTART_ATTEMPTS:
                cprint(
                    f"[Pulse] ⚠ {message}. Giving up after {MAX_RESTART_ATTEMPTS} attempts -- "
                    "continuing current run with the old in-memory code.",
                    color=_RED,
                )
                self._log_incident("restart_failed", message)
                return

            cprint(f"[Pulse] ⚠ {message}. Retrying restart...", color=_YELLOW)
            time.sleep(min(RETRY_BACKOFF_SECONDS * attempt, 30))

    def _print_variable_summary(self) -> None:
        variables = self.discover_variables()
        print("\nDiscovered variables:")
        for name in sorted(variables):
            val = self._cpu_observation(name, variables[name])
            if val is None and variables[name] is not None:
                continue
            if val is None:
                print(f"  • {name}: [not run yet]")
            else:
                try:
                    d = describe_tensor(val)
                    print(f"  • {name}: {d.backend} {d.kind} {d.shape}")
                except Exception:
                    print(f"  • {name}: trackable")

    def _cpu_observation(self, var_name: str, value: Any):
        """Return a CPU-resident observation without ever touching accelerator memory.

        If the observed object is on an accelerator, Pulse only looks for an
        explicitly maintained CPU mirror in watch_locals. It intentionally does
        NOT perform the tempting value.cpu()/numpy()/item() conversion here.
        That conversion belongs in the user's training code if they want Pulse
        to observe the value without Pulse owning the GPU transfer.
        """
        if value is None:
            return None
        if not _pulse_is_accelerator_value(value):
            return value

        for candidate in _pulse_cpu_mirror_candidates(var_name):
            mirror = self.watch_locals.get(candidate)
            if mirror is None or mirror is value:
                continue
            if _pulse_is_accelerator_value(mirror):
                continue
            try:
                if is_trackable(mirror):
                    return mirror
            except Exception:
                continue
        return None

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
            val = self._cpu_observation(name, variables[name])
            if val is None and variables[name] is not None:
                continue
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

        if self._history_context:
            lines.append(f"\n{self._history_context}")

        return "\n".join(lines)

    # Exception types worth retrying (the provider's own infrastructure
    # hiccupped, not something a retry can't fix like a bad API key or a
    # too-long context). Gemini's InternalServerError ("500 INTERNAL") is
    # exactly the kind of thing that shows up here and clears up on retry
    # a moment later -- checked the same getattr-based way as
    # _classify_model_error, for the same reason (never breaks if a
    # given litellm version doesn't define one of these).
    _RETRYABLE_ERROR_ATTRS = (
        "RateLimitError", "Timeout", "APIConnectionError",
        "ServiceUnavailableError", "InternalServerError",
    )

    def _is_retryable_model_error(self, exc: Exception) -> bool:
        for attr in self._RETRYABLE_ERROR_ATTRS:
            cls = getattr(litellm, attr, None)
            if cls is not None and isinstance(exc, cls):
                return True
        return False

    def _classify_model_error(self, exc: Exception) -> str:
        """Turn whatever litellm/the provider's SDK raised into one short,
        stable, user-facing reason instead of a raw exception repr (which
        for some providers is a multi-line JSON blob that's unreadable
        inline and, worse, risks getting fed right back into the pipeline
        as if it were a real answer -- see _call_model's caller). Checked
        via getattr rather than a direct `except litellm.XError` so this
        never breaks if a given litellm version doesn't define one of
        these classes."""
        for attr, reason in (
            ("AuthenticationError", "authentication failed -- check the API key for this provider (/agent to re-enter it)"),
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
        # Local providers (Ollama/LM Studio/vLLM) most often fail as a
        # plain connection error before litellm even gets to classify it
        # (nothing listening on that port) -- catch that case by message
        # shape too, not just exception type.
        msg = str(exc)
        if self.agent_api_base and ("Connection" in msg or "connect" in msg.lower()):
            return f"couldn't reach the local server at {self.agent_api_base} -- is it running?"
        # Unknown shape -- still short-circuit the pipeline cleanly, just
        # without a friendly label. Truncated: some SDKs raise with a
        # very long body (full request/response dump).
        return msg[:200] + ("…" if len(msg) > 200 else "")

    def _call_model(self, instruction: str, max_tokens: int = 2000) -> str:
        """One lightweight completion call: system prompt + recent history +
        a one-off stage instruction. Does not touch self.agent_history --
        callers decide what (if anything) gets persisted once the whole
        pipeline is done, so intermediate stage instructions don't bloat the
        conversation the next question is built on.

        Transient failures (rate limits, timeouts, connection blips, a
        provider's own 500 -- see _is_retryable_model_error) are retried
        up to 3 times with a short backoff before giving up; anything
        else (bad API key, invalid model name, context too long) fails
        immediately since retrying won't change the outcome.

        On failure, raises AgentRequestFailed (never returns an error
        string dressed up as a real answer) -- see _ask_agent_impl's
        try/except, the one chokepoint that catches it for the whole
        multi-pass pipeline, and the GPU check-in's own try/except for
        the one call site outside that pipeline.
        """
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.agent_history[-10:]
            + [{"role": "user", "content": instruction}]
        )
        max_attempts = 3
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = litellm.completion(
                    model=self.agent_model_string or PROVIDERS[self.agent_provider]["model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=120.0,
                    api_base=self.agent_api_base,  # only set for local/self-hosted providers
                    api_key=(self.agent_key if self.agent_key and self.agent_key != "local" else None),
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise AgentRequestFailed("the provider returned an empty response")
                return content
            except AgentRequestFailed:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts and self._is_retryable_model_error(exc):
                    backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    cprint(
                        f"[Pulse] ⚠ Agent request hit a transient error (attempt {attempt}/{max_attempts}), "
                        f"retrying in {backoff}s: {self._classify_model_error(exc)}",
                        color=_RED,
                    )
                    time.sleep(backoff)
                    continue
                raise AgentRequestFailed(self._classify_model_error(exc)) from exc
        # Unreachable in practice (the loop above always returns or raises),
        # but keeps type-checkers happy and fails safe if that ever changes.
        raise AgentRequestFailed(self._classify_model_error(last_exc) if last_exc else "unknown error")

    _CALC_RE = re.compile(r"^\s*CALC:\s*(.+)$", re.MULTILINE)
    _PROMOTE_RE = re.compile(r"^\s*PROMOTE:\s*(.+)$", re.MULTILINE)
    _GPUTRACK_RE = re.compile(r"^\s*GPUTRACK:\s*(.+)$", re.MULTILINE)
    _GPUUNTRACK_RE = re.compile(r"^\s*GPUUNTRACK:\s*(.+)$", re.MULTILINE)
    _SENSITIVITY_RE = re.compile(r"^\s*SENSITIVITY:\s*(.+)$", re.MULTILINE)
    _NORMAL_START_RE = re.compile(r"^\s*NORMAL_START:\s*(.+)$", re.MULTILINE)

    @classmethod
    def _extract_directives(cls, text: str):
        """Pull CALC:/PROMOTE:/GPUTRACK:/GPUUNTRACK:/SENSITIVITY:/
        NORMAL_START: lines out of an agent response, returning
        (cleaned_text, calc_exprs, promote_names, gputrack_names,
        gpuuntrack_names, sensitivity_args, normal_start_args). Cleaned
        text has those lines stripped so they don't clutter what's
        printed/stored. A bare 'none' value (as instructed for the
        periodic GPU check-in reply format) is dropped rather than
        treated as a variable name. sensitivity_args/normal_start_args
        are lists of raw argument strings (usually 0 or 1) -- applied via
        _cmd_sensitivity(..., quiet=True) / the NORMAL_START parsing in
        _apply_directives, same as a manual /sensitivity.
        """
        calc_exprs = [m.strip() for m in cls._CALC_RE.findall(text) if m.strip()]
        promote_names = []
        for m in cls._PROMOTE_RE.findall(text):
            promote_names.extend(n.strip() for n in m.split(",") if n.strip())
        gputrack_names = []
        for m in cls._GPUTRACK_RE.findall(text):
            gputrack_names.extend(
                n.strip() for n in m.split(",") if n.strip() and n.strip().lower() != "none"
            )
        gpuuntrack_names = []
        for m in cls._GPUUNTRACK_RE.findall(text):
            gpuuntrack_names.extend(
                n.strip() for n in m.split(",") if n.strip() and n.strip().lower() != "none"
            )
        sensitivity_args = [m.strip() for m in cls._SENSITIVITY_RE.findall(text) if m.strip()]
        normal_start_args = [m.strip() for m in cls._NORMAL_START_RE.findall(text) if m.strip()]

        cleaned = cls._CALC_RE.sub("", text)
        cleaned = cls._PROMOTE_RE.sub("", cleaned)
        cleaned = cls._GPUTRACK_RE.sub("", cleaned)
        cleaned = cls._GPUUNTRACK_RE.sub("", cleaned)
        cleaned = cls._SENSITIVITY_RE.sub("", cleaned)
        cleaned = cls._NORMAL_START_RE.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, calc_exprs, promote_names, gputrack_names, gpuuntrack_names, sensitivity_args, normal_start_args

    def _apply_directives(
        self,
        calc_exprs: List[str],
        promote_names: List[str],
        gputrack_names: Optional[List[str]] = None,
        gpuuntrack_names: Optional[List[str]] = None,
        sensitivity_args: Optional[List[str]] = None,
        normal_start_args: Optional[List[str]] = None,
    ) -> str:
        """Deterministically compute any CALC: expressions and apply any
        PROMOTE:/GPUTRACK:/GPUUNTRACK:/SENSITIVITY:/NORMAL_START: requests,
        returning a short human-readable summary to print and to feed
        back into the agent's own history (so it sees the verified
        numbers/state on the next turn instead of trusting its own
        arithmetic or memory).
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

        if gputrack_names:
            tracked = []
            for name in gputrack_names:
                result = self._cmd_gputrack(name, quiet=True)
                if result:
                    tracked.append(result)
            if tracked:
                print(f"  🎯 GPU-tracking (agent request): {', '.join(tracked)} -- this slows training.")
                notes.append(
                    f"Now GPU-tracking: {', '.join(tracked)}. This forces a device-to-host sync every "
                    "probe and costs training throughput -- send a GPUUNTRACK: line for it as soon as "
                    "you no longer need the closer look."
                )

        if gpuuntrack_names:
            untracked = []
            for name in gpuuntrack_names:
                result = self._cmd_gpuuntrack(name, quiet=True)
                if result:
                    untracked.append(result if result != "all" else "all GPU-tracked variables")
            if untracked:
                print(f"  ✋ Stopped GPU-tracking (agent request): {', '.join(untracked)}")
                notes.append(f"Stopped GPU-tracking: {', '.join(untracked)}.")

        if sensitivity_args:
            # Only the last SENSITIVITY: line (if the agent sent more than
            # one, which it shouldn't) is applied -- last-write-wins, same
            # as any other directive would if duplicated.
            result = self._cmd_sensitivity(sensitivity_args[-1], quiet=True)
            if result:
                print(f"  🎚 Sensitivity adjusted (agent request): {result}")
                notes.append(result)

        if normal_start_args:
            applied = []
            for raw in normal_start_args:
                if raw.strip().lower() == "none":
                    continue
                for pair in raw.split(","):
                    pair = pair.strip()
                    if not pair or "=" not in pair:
                        continue
                    name, _, val_str = pair.partition("=")
                    name = name.strip()
                    if not name:
                        continue
                    try:
                        val = float(val_str.strip())
                    except ValueError:
                        continue
                    self._normal_start_baselines[name] = val
                    applied.append(f"{name}={val:.4g}")
            if applied:
                print(f"  🌱 Seeded starting baseline (agent estimate, from code): {', '.join(applied)}")
                notes.append(
                    f"Seeded expected starting value(s) from the code: {', '.join(applied)}. "
                    "These act as a spike-detection baseline until real data accumulates, so a "
                    "genuine explosion in the first few steps can still be caught."
                )

        return "\n\n".join(notes)

    _GPU_CHECKIN_PROMPT = (
        "[Automated GPU check-in -- sent automatically every {mins:g} minutes since Pulse started] "
        "Currently GPU-tracked variable(s): {tracked}. GPU stat probes force a device-to-host sync "
        "on every probe, which slows down training, so nothing should stay GPU-tracked longer than "
        "you actually need it -- and if training currently looks like it needs a closer look at a "
        "GPU-resident variable you're NOT already tracking, this is also your chance to start.\n\n"
        "Reply with ONLY these two lines, in exactly this format, and nothing else -- no diagnosis, "
        "no prose:\n"
        "GPUTRACK: <comma-separated variable names to track, or 'none'>\n"
        "GPUUNTRACK: <comma-separated variable names to stop tracking, or 'none'>"
    )

    def _maybe_gpu_checkin(self) -> None:
        """Every gpu_checkin_interval seconds (15 min by default) after
        Pulse starts, proactively ask the agent whether it still wants to
        GPU-track anything -- rather than only ever reacting to a
        GPUTRACK:/GPUUNTRACK: line the agent happened to include while
        diagnosing something else. Requires the reply to use the same
        strict two-line format as the check-in prompt, so it's parsed the
        same deterministic way as any other directive.
        """
        if not self.agent_provider or not self.agent_key:
            return
        now = time.monotonic()
        if now - self._last_gpu_checkin < self.gpu_checkin_interval:
            return
        self._last_gpu_checkin = now

        tracked = ", ".join(sorted(self.gpu_tracked_vars)) if self.gpu_tracked_vars else "(none)"
        prompt = self._GPU_CHECKIN_PROMPT.format(
            mins=self.gpu_checkin_interval / 60.0, tracked=tracked
        )
        cprint(
            f"\n[Pulse] ⏱ {self.gpu_checkin_interval / 60:g}-min GPU check-in -- asking agent whether "
            "it still needs GPU tracking...",
            color=_YELLOW,
        )
        try:
            answer = self._call_model(prompt, max_tokens=150)
        except AgentRequestFailed as exc:
            # This is a periodic, low-stakes background check -- never
            # worth interrupting training over. Quietly skip this round
            # and try again at the next interval.
            cprint(f"[Pulse] ⚠ GPU check-in skipped (agent request failed: {exc})", color=_RED)
            return
        _, _calc, _promote, gputrack_names, gpuuntrack_names, _sens, _norm = self._extract_directives(answer)
        summary = self._apply_directives([], [], gputrack_names, gpuuntrack_names)
        if summary:
            self.agent_history.append({"role": "user", "content": prompt})
            self.agent_history.append({"role": "assistant", "content": answer})
            cprint(f"[Pulse] {summary}", color=_YELLOW)
        else:
            cprint("[Pulse] Agent made no GPU-tracking changes at this check-in.", color=_YELLOW)

    @staticmethod
    def _parse_json_obj(answer: str) -> Optional[Dict[str, Any]]:
        """Generic defensive JSON-object parser, used for the verify (pass
        4) and sweep (pass 5) responses -- same tolerance for stray code
        fences as _parse_code_fix, but without requiring any particular
        fields.

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
                    _PASS4_VERIFY_TMPL.format(fix_desc=fix_desc), max_tokens=350
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
        if self.auto_intervene:
            cprint("[Pulse] Auto-fix is on -- addressing it automatically.")
            resp = "y"
        else:
            try:
                _flush_stdin()
                resp = input("Fix other errors? (y/n) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
        if resp not in ("y", "yes"):
            return

        print("\n[6] Following the established format for the additional issue(s)...")
        self.ask_agent(
            f"Please also fix this: {summary}", include_code=include_code, _depth=_depth + 1
        )

    def ask_agent(
        self, question: str, include_code: bool = False, _depth: int = 0,
        traceback_signature: Optional[str] = None,
    ) -> str:
        """Thin wrapper around `_ask_agent_impl` that syncs the resulting
        Q&A turn to Debug_Sessions.agent_logs (cloud) once the top-level
        call completes. Only the top-level call (_depth == 0) syncs --
        recursive sweep passes are logged as part of that same turn's
        answer, not as separate entries.

        `traceback_signature`, when given (crash-triggered calls), is
        stamped onto the log entry along with any fix that actually got
        applied -- see _load_history_context / offer_known_fix, which use
        that to recognize and re-fix the same bug in a later session
        without a full re-diagnosis.
        """
        if _depth == 0:
            self._last_applied_fix = None
        answer = self._ask_agent_impl(question, include_code=include_code, _depth=_depth)
        if _depth == 0:
            self._sync_agent_turn(
                question, answer,
                traceback_signature=traceback_signature,
                fix_applied=self._last_applied_fix,
            )
            if traceback_signature and self._last_applied_fix:
                self._known_fixes[traceback_signature] = self._last_applied_fix
                self._resolved_signatures.add(traceback_signature)
        return answer

    def _ask_agent_impl(self, question: str, include_code: bool = False, _depth: int = 0) -> str:
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

        try:
            # Pass 1: locate the region(s) of the error.
            with _Spinner("Reading for region of error"):
                regions = self._call_model(_PASS1_LOCATE, max_tokens=200)
            print(f"\n[1] Region of error\n{regions}\n")

            # Pass 2: focused second read + diagnosis/reasoning. May include
            # CALC:/PROMOTE:/GPUTRACK:/GPUUNTRACK: directives, stripped out and
            # executed deterministically rather than trusted from the model.
            with _Spinner("Analyzing"):
                raw_analysis = self._call_model(
                    _PASS2_ANALYZE_TMPL.format(regions=regions), max_tokens=700
                )
            analysis, calc_exprs, promote_names, gputrack_names, gpuuntrack_names, sensitivity_args, normal_start_args = self._extract_directives(raw_analysis)
            print(f"[2] Diagnosis & reasoning\n{analysis}\n")
            directive_note = self._apply_directives(calc_exprs, promote_names, gputrack_names, gpuuntrack_names, sensitivity_args, normal_start_args)
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
        except AgentRequestFailed as exc:
            # Whichever pass hit this, stop the pipeline right here rather
            # than let the raw failure get treated as a real diagnosis/fix
            # downstream (parsed as JSON, offered to /revert, etc.). At
            # _depth > 0 (a recursive sweep pass) this becomes that
            # recursive call's return value, which the caller already
            # treats as plain text -- no special-casing needed there.
            msg = f"⚠ AI agent request failed: {exc}"
            print(f"\n{msg}\n")
            self.agent_history.append({"role": "assistant", "content": msg})
            return msg

        if _depth == 0 and self._fix_applied_this_turn:
            self._restart_process()  # does not return

        return result

    @staticmethod
    def _parse_code_fix(answer: str) -> Optional[Dict[str, Any]]:
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

    def _fix_log_dir(self) -> Optional[str]:
        """Directory Pulse's persistent change log lives in: a
        '.pulse_history' folder inside the user's actual project
        workspace (next to the tracked script), not a temp dir -- so it
        survives both a Pulse restart and a full reboot. Falls back to
        the resolved repo cwd if script_path isn't known yet.
        """
        base = None
        if self.script_path:
            base = os.path.dirname(os.path.abspath(self.script_path))
        elif self._repo_cwd:
            base = self._repo_cwd
        if not base:
            return None
        d = os.path.join(base, ".pulse_history")
        try:
            os.makedirs(d, exist_ok=True)
            os.makedirs(os.path.join(d, "diffs"), exist_ok=True)
        except OSError:
            return None
        return d

    def _fix_log_path(self) -> Optional[str]:
        d = self._fix_log_dir()
        return os.path.join(d, "changelog.jsonl") if d else None

    def _load_fix_log(self) -> List[Dict[str, Any]]:
        """Read every recorded commit back from disk, oldest first. Reads
        fresh every time (rather than caching in memory) specifically so
        this reflects commits from a *previous* Pulse process too -- the
        whole point of a workspace-persisted log is that /revert still
        works after a restart.
        """
        path = self._fix_log_path()
        entries: List[Dict[str, Any]] = []
        if not path or not os.path.exists(path):
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
                        continue  # tolerate a corrupted/partial trailing line
        except OSError:
            pass
        return entries

    def _record_fix_commit(
        self, files: Dict[str, tuple], explanation: str, kind: str = "fix"
    ) -> Optional[str]:
        """Append one 'commit' to the persistent, git-diff-style change
        log. `files` maps path -> (before, after) full file contents.
        Each commit stores a real unified diff (for human reading, and
        written out to its own .diff file too) AND the full before/after
        content of every file it touched -- the latter is what actually
        makes /revert able to reconstruct the workspace as it looked at
        ANY past commit, not just undo the single most recent change.
        Returns the new commit's short id, or None if there was nothing
        to record.
        """
        path = self._fix_log_path()
        if not path or not files:
            return None

        commit_id = hashlib.sha1(
            f"{time.time()}|{explanation}|{sorted(files)}".encode("utf-8", "replace")
        ).hexdigest()[:8]

        file_entries = []
        diff_parts = []
        for fpath, (before, after) in files.items():
            label = os.path.basename(fpath)
            diff = "".join(difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{label}", tofile=f"b/{label}",
            ))
            diff_parts.append(diff or f"(no textual change to {label})\n")
            file_entries.append({"path": fpath, "before": before, "after": after, "diff": diff})

        entry = {
            "id": commit_id,
            "kind": kind,  # "fix" (agent-applied) or "revert" (undo of a previous commit)
            "t": time.time(),
            "explanation": explanation,
            "files": file_entries,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            cprint(f"[Pulse CLI] ⚠ Could not write to change log: {exc}", color=_RED)
            return None

        diff_dir = os.path.join(self._fix_log_dir() or "", "diffs")
        try:
            with open(os.path.join(diff_dir, f"{commit_id}.diff"), "w", encoding="utf-8") as f:
                f.write("\n".join(diff_parts))
        except OSError:
            pass  # non-fatal -- the jsonl entry above already has the same diff text

        return commit_id

    def _fix_log_state_asof(self, entries: List[Dict[str, Any]], target_idx: int) -> Dict[str, str]:
        """{path: content} the workspace would have had right after the
        commit at `target_idx` landed -- built by replaying the log in
        order and keeping the latest 'after' seen per file, up to and
        including that commit. `target_idx == -1` means "before the very
        first commit" (an empty dict; see _fix_log_original for that
        case's fallback).
        """
        state: Dict[str, str] = {}
        for i, e in enumerate(entries):
            if i > target_idx:
                break
            for fc in e.get("files", []):
                state[fc["path"]] = fc["after"]
        return state

    @staticmethod
    def _fix_log_original(entries: List[Dict[str, Any]], path: str) -> Optional[str]:
        """The very first 'before' recorded for `path` anywhere in the
        log, i.e. its content before Pulse ever touched it."""
        for e in entries:
            for fc in e.get("files", []):
                if fc["path"] == path:
                    return fc["before"]
        return None

    def _cmd_log(self, arg: str) -> None:
        """/log -- list Pulse's persisted code-fix commits for this
        project. Survives restarts: reads straight from
        .pulse_history/changelog.jsonl in the script's own directory."""
        entries = self._load_fix_log()
        if not entries:
            cprint("[Pulse CLI] No recorded Pulse code changes yet for this project.")
            return
        print(f"\n{len(entries)} recorded change(s) (.pulse_history/changelog.jsonl):")
        for e in entries:
            try:
                ts = datetime.fromtimestamp(e.get("t", 0)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = "?"
            files = ", ".join(os.path.basename(fc["path"]) for fc in e.get("files", []))
            tag = "" if e.get("kind", "fix") == "fix" else "  [revert]"
            print(f"  {e.get('id', '?')}  {ts}  [{files}]{tag}")
            expl = e.get("explanation")
            if expl:
                print(f"           {expl}")
        print(
            "\nUse /revert <id> to restore the workspace to right after that commit, "
            "or /revert with no id to undo the most recent one."
        )

    def _cmd_revert(self, arg: str) -> None:
        """/revert [commit_id] -- restore the workspace to exactly how it
        looked right after a given Pulse commit (or, with no id given,
        undo the single most recent one). Reads the persistent
        .pulse_history log, so this works even if Pulse -- or the whole
        machine -- was restarted since the fix was applied, unlike the
        old in-memory-only _pending_revert_backups. The revert itself is
        logged as a new commit too, so it can be undone the same way.
        """
        entries = self._load_fix_log()
        if not entries:
            cprint("[Pulse CLI] No recorded Pulse code changes to revert.")
            return

        arg = arg.strip()
        if not arg:
            target_idx = len(entries) - 2  # state right before the last commit
            target_label = f"before commit {entries[-1].get('id', '?')} (most recent)"
        else:
            target_idx = None
            target_label = ""
            for i, e in enumerate(entries):
                eid = e.get("id", "")
                if eid == arg or (arg and eid.startswith(arg)):
                    target_idx = i
                    target_label = f"commit {eid}"
                    break
            if target_idx is None:
                cprint(f"[Pulse CLI] No commit matches '{arg}'. Use /log to see recorded commits.")
                return

        all_paths = sorted({fc["path"] for e in entries for fc in e.get("files", [])})
        asof = self._fix_log_state_asof(entries, target_idx)

        restored, failed = [], []
        revert_files: Dict[str, tuple] = {}
        for fpath in all_paths:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    current = f.read()
            except OSError:
                current = None  # file has since been deleted/moved -- still try to restore it

            content = asof.get(fpath)
            if content is None:
                content = self._fix_log_original(entries, fpath)
            if content is None or content == current:
                continue  # nothing recorded for this file, or it's already in that state

            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as exc:
                failed.append((fpath, str(exc)))
                continue

            if current is not None:
                revert_files[fpath] = (current, content)
            restored.append(fpath)
            if fpath == self.script_path:
                self.code_text = content
            elif fpath in self.extra_files:
                self.extra_files[fpath] = content

        if restored:
            cprint(f"[Pulse CLI] ✓ Reverted to state {target_label} -- {len(restored)} file(s) restored:", color=_BLUE)
            for p in restored:
                print(f"    - {os.path.basename(p)}")
            new_commit = self._record_fix_commit(
                revert_files, f"Reverted to state {target_label}", kind="revert"
            )
            if new_commit:
                cprint(f"[Pulse CLI] 📝 Logged as commit {new_commit}.")
                self._log_incident(
                    "revert", f"Reverted to state {target_label}",
                    commit_id=new_commit, files=[os.path.basename(p) for p in restored],
                )
            cprint(
                "[Pulse CLI] Note: if this script is already running, it's still executing the old "
                "code in memory -- stop and re-run it to pick up the reverted version.",
                color=_YELLOW,
            )
        if failed:
            cprint(f"[Pulse CLI] ⚠ Failed to revert {len(failed)} file(s):", color=_RED)
            for p, err in failed:
                print(f"    - {os.path.basename(p)}: {err}")
        if not restored and not failed:
            cprint("[Pulse CLI] Workspace already matches that state -- nothing to revert.")

    def _git_autostash(self) -> Optional[str]:
        """Best-effort git-level safety net, on top of (not instead of)
        the .pulse_history changelog/revert system above: right before
        Pulse writes an agent-proposed fix to disk, stash any of the
        user's OWN uncommitted changes in the repo first, so a fix never
        gets silently mixed into work the user hadn't committed yet, and
        `git stash pop` alone (independent of Pulse) is always enough to
        get back to exactly the pre-fix working tree. No-op (returns
        None) if this isn't a git repo, git isn't installed, git is
        already mid-operation (rebase/merge -- stashing there can make
        things worse), or there's nothing uncommitted to stash. Never
        raises and never blocks applying the fix -- this is a convenience
        net, not a requirement.
        """
        repo = self._repo_cwd or (os.path.dirname(os.path.abspath(self.script_path)) if self.script_path else None)
        if not repo or not os.path.isdir(os.path.join(repo, ".git")):
            return None
        try:
            status = subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            if status.returncode != 0 or not status.stdout.strip():
                return None  # not a repo (already checked) or nothing dirty to protect

            for marker in ("MERGE_HEAD", "rebase-merge", "rebase-apply"):
                if os.path.exists(os.path.join(repo, ".git", marker)):
                    return None  # mid-merge/rebase -- don't touch the working tree

            label = f"pulse-autostash-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            stash = subprocess.run(
                ["git", "-C", repo, "stash", "push", "--include-untracked", "-m", label],
                capture_output=True, text=True, timeout=10,
            )
            if stash.returncode != 0:
                return None
            return label
        except Exception:
            return None

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
        Also takes a best-effort git-level backup first -- see
        _git_autostash.
        """
        self._build_file_labels()
        autostash_label = self._git_autostash()

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
        after_by_path: Dict[str, str] = {}
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
            after_by_path[path] = content
            if path == self.script_path:
                self.code_text = content
            elif path in self.extra_files:
                self.extra_files[path] = content

        for old, label in unresolved:
            skipped.append((old, label or "(unspecified file)", "couldn't determine which file this targets"))

        if not applied_by_path:
            if autostash_label:
                # Nothing was actually applied -- pop the stash back immediately
                # rather than leaving the user's own changes sitting stashed
                # for no reason.
                try:
                    subprocess.run(
                        ["git", "-C", self._repo_cwd or os.path.dirname(os.path.abspath(self.script_path)), "stash", "pop"],
                        capture_output=True, text=True, timeout=10,
                    )
                except Exception:
                    pass
            lines.append("\n⚠ No changes were applied -- none of the proposed snippets matched cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: {old.splitlines()[0][:80]}...")
            return "\n".join(lines)

        for path, applied in applied_by_path.items():
            lines.append(f"\n✓ Applied {len(applied)} change(s) to '{os.path.basename(path)}':")
            for old, new in applied:
                safe_old = old.splitlines()[0][:80] if old.splitlines() else "(empty)"
                safe_new = new.splitlines()[0][:80] if new.splitlines() else "(empty)"
                lines.append(f"  - replaced:\n      {safe_old}...\n    with:\n      {safe_new}...")
        if skipped:
            lines.append(f"\n⚠ Skipped {len(skipped)} proposed change(s) that didn't match cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: {old.splitlines()[0][:80]}...")

        commit_files = {p: (originals[p], after_by_path[p]) for p in applied_by_path}
        commit_id = self._record_fix_commit(commit_files, fix.get("explanation") or "(no explanation given)")
        if commit_id:
            lines.append(f"\n📝 Logged as commit {commit_id} in .pulse_history/ -- /revert {commit_id} to undo, or /log to see history.")
            self._log_incident(
                "fix_applied", fix.get("explanation") or "(no explanation given)",
                commit_id=commit_id, files=[os.path.basename(p) for p in applied_by_path],
            )
        if autostash_label:
            lines.append(
                f"🛟 Your own uncommitted changes were stashed first (git stash: '{autostash_label}') -- "
                f"`git stash pop` restores them independent of Pulse's own /revert."
            )

        cprint("\n".join(lines), color=_BLUE)

        self._pending_revert_backups = dict(originals)  # path -> original content
        self._fix_applied_this_turn = True  # tells ask_agent's top-level call to restart afterward
        self._last_applied_fix = fix  # structured old/new/files/explanation, for cross-session reuse

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

    def _cmd_telemetry(self, arg: str) -> None:
        """/telemetry on|off -- controls whether Pulse collects/sends
        environment info (Python/GPU/CUDA/framework versions) as part of
        the cloud debug session. Independent of the rest of cloud sync
        (agent Q&A, error tracebacks still sync if a debug session is
        active) -- this only gates the environment-info collection path.
        Off by default only if PULSE_TELEMETRY=off was set before Pulse
        started; this command changes it for the rest of THIS run."""
        arg = arg.strip().lower()
        if arg in ("on", "true", "1", "enable", "enabled"):
            self.telemetry_enabled = True
            print("✓ Telemetry is ON -- environment info (Python/GPU/framework versions) will be collected and synced.")
        elif arg in ("off", "false", "0", "disable", "disabled"):
            self.telemetry_enabled = False
            print("✓ Telemetry is OFF for this run -- no environment info will be collected or sent.")
        else:
            state = "ON" if self.telemetry_enabled else "OFF"
            print(f"Telemetry is currently {state}. Usage: /telemetry on|off (or set PULSE_TELEMETRY=off before starting)")

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

    # Named presets for /sensitivity -- just friendly aliases for a
    # self.sensitivity value, since "0.3" means nothing to most users but
    # "loose"/"tight" does.
    _SENSITIVITY_PRESETS = {
        "loosest": 0.0, "quiet": 0.0,
        "loose": 0.2,
        "default": 0.3,
        "medium": 0.5, "normal": 0.5,
        "tight": 0.75,
        "tightest": 1.0, "twitchy": 1.0,
    }

    def _sensitivity_thresholds(self) -> Dict[str, float]:
        """Derive the four spike/plateau/oscillation thresholds from the
        single self.sensitivity dial (0.0=loosest .. 1.0=tightest),
        letting any of the four advanced overrides (self.explosion_
        multiplier / plateau_range_frac / oscillation_flip_threshold /
        oscillation_delta_frac) pin that one signal instead. Called fresh
        on every _check_for_trouble pass, so /sensitivity and a live
        SENSITIVITY: directive both take effect on the very next check --
        no restart needed."""
        s = max(0.0, min(1.0, self.sensitivity))
        return {
            # Loose (s=0): only a 8x jump counts as a spike. Tight (s=1): 2.5x does.
            "explosion_multiplier": (
                self.explosion_multiplier if self.explosion_multiplier is not None
                else 8.0 - 5.5 * s
            ),
            # Loose (s=0): only a near-dead-flat window (1e-5 of latest) counts as a
            # plateau. Tight (s=1): even a window with noticeable movement (1e-2) does.
            # (Old hardcoded behavior was a fixed 1e-4, roughly s≈0.55 on this curve,
            # so the s=0.3 default below is deliberately looser than the old fixed value.)
            "plateau_range_frac": (
                self.plateau_range_frac if self.plateau_range_frac is not None
                else 10 ** (-5 + 3 * s)
            ),
            # Loose: needs 15/18 directional reversals. Tight: only 6.
            # (A 20-point window yields 19 deltas and only 18 consecutive
            # delta-pairs to check for a sign flip -- see _check_for_trouble.)
            "oscillation_flip_threshold": (
                self.oscillation_flip_threshold if self.oscillation_flip_threshold is not None
                else round(15 - 9 * s)
            ),
            # Loose: reversals need to average 10% of |latest| to count.
            # Tight: 2% is enough.
            "oscillation_delta_frac": (
                self.oscillation_delta_frac if self.oscillation_delta_frac is not None
                else 0.10 - 0.08 * s
            ),
            # Stagnation: has a loss-like variable meaningfully improved
            # over a LONG window, once normal per-step noise is averaged
            # out? This is deliberately separate from "plateau" above --
            # plateau looks at the *range* of the last 20 steps, which a
            # loss with completely normal noise (bouncing around by, say,
            # 0.02 every step while never actually trending down) will
            # never look "flat enough" to trigger, even after a thousand
            # steps of zero real progress. Stagnation instead compares
            # the mean of the first quarter of a long window against the
            # mean of the last quarter, so per-step noise washes out and
            # only genuine lack of improvement is left.
            # Loose (s=0): needs 500 steps of history, and even a 0.3%
            # improvement over that window counts as "still improving".
            # Tight (s=1): needs only 150 steps, and demands a full 8%
            # improvement before it stops flagging.
            "stagnation_window": round(500 - 350 * s),
            "stagnation_frac": (
                self.stagnation_frac if self.stagnation_frac is not None
                else 0.003 + 0.077 * s
            ),
        }

    def _cmd_sensitivity(self, arg: str, quiet: bool = False) -> Optional[str]:
        """/sensitivity [0.0-1.0 | preset | spike|plateau|oscillation <value> | reset]

        Controls how eagerly _check_for_trouble flags a loss-like variable
        as spiking/plateaued/oscillating (see _sensitivity_thresholds).
        quiet=True is used when this is invoked via the agent's
        SENSITIVITY: directive instead of the manual /sensitivity command
        (mirrors _cmd_gputrack's quiet param) -- same effect, just no
        console prompt-usage text on a bad argument and a returned summary
        string instead of a printed one, so ask_agent can fold it into the
        directive-applied note it shows the user."""
        arg = arg.strip()
        if not arg:
            th = self._sensitivity_thresholds()
            msg = (
                f"Sensitivity is {self.sensitivity:.2f} (0=loosest, 1=tightest). Derived thresholds: "
                f"spike >{th['explosion_multiplier']:.1f}x baseline, "
                f"plateau range <{th['plateau_range_frac']:.1e} of latest, "
                f"oscillation >={int(th['oscillation_flip_threshold'])} reversals/18 "
                f"with avg swing >{th['oscillation_delta_frac']*100:.0f}% of latest, "
                f"stagnation <{th['stagnation_frac']*100:.1f}% improvement over {int(th['stagnation_window'])} steps. "
                "Usage: /sensitivity <0.0-1.0|loose|medium|tight> or "
                "/sensitivity <spike|plateau|oscillation|stagnation> <value|auto>"
            )
            if quiet:
                return msg
            print(msg)
            return None

        parts = arg.split(None, 1)
        sub = parts[0].lower()
        if sub in ("spike", "plateau", "oscillation", "stagnation") and len(parts) == 2:
            val_str = parts[1].strip().lower()
            field = {
                "spike": "explosion_multiplier",
                "plateau": "plateau_range_frac",
                "oscillation": "oscillation_flip_threshold",  # flips; delta_frac follows the dial
                "stagnation": "stagnation_frac",  # window length always follows the dial
            }[sub]
            if val_str == "auto":
                setattr(self, field, None)
                msg = f"✓ '{sub}' sensitivity now follows the overall dial ({self.sensitivity:.2f})."
            else:
                try:
                    val = float(val_str)
                    setattr(self, field, int(val) if field == "oscillation_flip_threshold" else val)
                    msg = f"✓ '{sub}' sensitivity pinned to {val_str}."
                except ValueError:
                    msg = f"Usage: /sensitivity {sub} <number|auto>"
            if quiet:
                return msg
            print(msg)
            return None

        if sub == "reset":
            self.explosion_multiplier = self.plateau_range_frac = None
            self.oscillation_flip_threshold = self.oscillation_delta_frac = None
            self.stagnation_frac = None
            msg = "✓ Cleared per-signal overrides -- all thresholds now follow the overall dial."
            if quiet:
                return msg
            print(msg)
            return None

        if sub in self._SENSITIVITY_PRESETS:
            self.sensitivity = self._SENSITIVITY_PRESETS[sub]
        else:
            try:
                self.sensitivity = max(0.0, min(1.0, float(sub)))
            except ValueError:
                msg = (
                    f"Usage: /sensitivity <0.0-1.0|{'|'.join(self._SENSITIVITY_PRESETS)}> or "
                    "/sensitivity <spike|plateau|oscillation|stagnation> <value|auto>"
                )
                if quiet:
                    return msg
                print(msg)
                return None

        msg = f"✓ Sensitivity set to {self.sensitivity:.2f}. Use /sensitivity with no argument to see the derived thresholds."
        if quiet:
            return msg
        print(msg)
        return None

    def _check_for_trouble(self) -> Optional[str]:
        reasons = []
        th = self._sensitivity_thresholds()

        for var_name in self.tracked_vars:
            hist = self.scalar_histories.get(var_name)
            if not hist:
                continue
            latest = hist[-1]
            if latest is not None and isinstance(latest, (int, float)) and not math.isfinite(latest):
                reasons.append(f"'{var_name}' just went non-finite (NaN/inf): {latest}")
                continue

            if _looks_like_loss(var_name) and latest is not None:
                finite_recent = [v for v in hist[-50:] if v is not None and math.isfinite(v)]

                # Explosion detection. Normally needs a few real finite
                # points to compute a "recent minimum" baseline -- which
                # meant a genuine explosion in the first few steps of
                # training could never be caught, since there just wasn't
                # enough history yet to compare against. If the agent has
                # seeded an expected starting value for this variable from
                # reading the code (a NORMAL_START: directive, sent
                # automatically once at the start of training -- see
                # _apply_directives / _prime_at_start), fold it into the
                # baseline pool so a step-1 blow-up has something real to
                # compare against too.
                seed = self._normal_start_baselines.get(var_name)
                baseline_pool = finite_recent[:-1] if len(finite_recent) >= 5 else []
                if seed is not None:
                    baseline_pool = baseline_pool + [seed]
                if baseline_pool:
                    baseline = min(baseline_pool)
                    if baseline > 0 and latest > baseline * th["explosion_multiplier"]:
                        basis = (
                            "its expected starting value (estimated from the code)"
                            if not finite_recent[:-1] else "its recent minimum"
                        )
                        reasons.append(
                            f"'{var_name}' spiked to {latest:.4g}, "
                            f"{latest / baseline:.1f}x {basis} ({baseline:.4g})"
                        )

                if len(finite_recent) >= 20:
                    recent_window = finite_recent[-20:]

                    deltas = [recent_window[i] - recent_window[i-1] for i in range(1, len(recent_window))]

                    # Scale the plateau/oscillation thresholds off the
                    # window's own typical magnitude (mean |value| across
                    # all 20 points), not off `latest` alone. `latest` is
                    # a single sample that can itself land at a momentary
                    # peak, trough, or near-zero crossing of the very
                    # curve being judged -- using it as the sole scale
                    # reference made both checks unreliable: too
                    # trigger-happy right as a curve crossed zero, too lax
                    # whenever `latest` happened to be sitting at a local
                    # extreme instead of a typical value.
                    scale = sum(abs(v) for v in recent_window) / len(recent_window)
                    if scale == 0:
                        scale = abs(latest)  # degenerate all-zero window; fall back rather than lose the check entirely

                    window_range = max(recent_window) - min(recent_window)
                    if window_range == 0:
                        # Identical value for 20 straight steps (dead
                        # gradient, lr=0, a frozen model) is the most
                        # extreme plateau there is, not an edge case to
                        # skip. The old `window_range > 0` guard here
                        # excluded exactly this, so a totally frozen loss
                        # -- arguably the easiest plateau to catch -- was
                        # the one case that could never be flagged.
                        reasons.append(f"'{var_name}' has completely frozen (identical value for the last 20 steps).")
                    elif window_range < (scale * th["plateau_range_frac"]):
                        reasons.append(f"'{var_name}' has plateaued (range across last 20 steps is {window_range:.2e}).")

                    sign_flips = sum(1 for i in range(1, len(deltas)) if (deltas[i] * deltas[i-1]) < 0)
                    avg_delta_mag = sum(abs(d) for d in deltas) / len(deltas)

                    if sign_flips >= th["oscillation_flip_threshold"] and avg_delta_mag > (scale * th["oscillation_delta_frac"]):
                        reasons.append(f"'{var_name}' is heavily oscillating ({sign_flips} directional reversals in the last 20 steps).")

                # Long-horizon stagnation check -- deliberately separate
                # from the plateau check above, and computed from its own
                # independently-sized slice of `hist` (not finite_recent,
                # which stays capped at 50 so it doesn't change the
                # explosion baseline's "recent minimum" into a
                # much-older, possibly stale minimum). Plateau looks at
                # the *range* of just the last 20 steps, so a loss
                # bouncing around by completely normal per-step noise
                # (e.g. +/-0.02 every step, never trending down) will
                # never look "flat enough" to trip it, no matter how many
                # hundreds of steps go by with zero real progress --
                # which is exactly what a loss stuck oscillating in a
                # narrow band around the same value for 1000+ steps looks
                # like. This instead compares the mean of the first
                # quarter of a long window against the mean of the last
                # quarter, so per-step noise averages out and only
                # genuine lack of improvement is left standing.
                window = int(th["stagnation_window"])
                long_window_raw = [v for v in hist[-window:] if v is not None and math.isfinite(v)]
                if len(long_window_raw) >= window:
                    quarter = max(1, window // 4)
                    early_mean = sum(long_window_raw[:quarter]) / quarter
                    late_mean = sum(long_window_raw[-quarter:]) / quarter
                    if early_mean != 0 and abs(early_mean - late_mean) < abs(early_mean) * th["stagnation_frac"]:
                        reasons.append(
                            f"'{var_name}' hasn't meaningfully improved over the last {window} steps "
                            f"(from {early_mean:.4g} to {late_mean:.4g}, despite normal step-to-step noise)."
                        )

        for sub_name, entry in self._matrix_cache.items():
            stats = entry.get("stats", {})
            if stats.get("nan") or stats.get("inf"):
                reasons.append(f"'{sub_name}' has nan={stats.get('nan')} inf={stats.get('inf')}")

        return "; ".join(reasons) if reasons else None
    _START_PRIME_PROMPT = (
        "[Automatic start-of-run check -- sent once, automatically, before the first training step, "
        "so this is your only chance to set these from the code alone, before any real data exists] "
        "Look at the training code and the tracked variables above. Three things:\n"
        "1. Judge how noisy this run's loss/metric curves are likely to be, given the model type, "
        "batch size, learning rate, and loss function, and set an appropriate sensitivity for "
        "spike/plateau/oscillation detection.\n"
        "2. For each loss-like tracked variable, estimate its expected value at the very start of "
        "training from the code alone if you can justify one (e.g. a randomly-initialized N-class "
        "classifier's cross-entropy loss starts near ln(N); a policy's initial reward is often near "
        "a known random-policy baseline). This is the ONLY way Pulse can catch a real explosion in "
        "the first few steps of training -- normally spike detection needs several real data points "
        "before it has anything to compare against, so a blow-up before then would otherwise go "
        "completely undetected.\n"
        "3. Identify any critical variables -- e.g. loss, or core tensors that live exclusively in "
        "GPU memory -- that are worth the closer, GPU-synced look from step one, to catch memory "
        "spikes or numerical instability early rather than after they've already caused visible "
        "damage.\n\n"
        "Reply with ONLY these three lines, in exactly this format, and nothing else -- no diagnosis, "
        "no prose:\n"
        "SENSITIVITY: <0.0-1.0, a preset (loose/medium/tight), or 'spike|plateau|oscillation <value|auto>'>\n"
        "NORMAL_START: <comma-separated var=value pairs for loss-like tracked variables you can justify, or 'none'>\n"
        "GPUTRACK: <comma-separated variable names to track closely from the start, or 'none'>"
    )

    def _prime_at_start(self) -> None:
        """Ask the agent, once, automatically, before the very first
        training step, to (a) set a sensitivity appropriate to this run's
        code, (b) estimate a starting-value baseline for loss-like tracked
        variables by reading the code alone (a NORMAL_START: directive --
        see _apply_directives), and (c) flag any variables worth GPU-level
        tracking from step one (a GPUTRACK: directive).

        All three of these used to only ever happen reactively: sensitivity
        only got tuned once the agent had already seen live data (e.g. a
        GPU check-in or a manual /ask), GPU-tracking only ever got turned
        on after something had already looked suspicious enough to ask
        about, and there was no mechanism at all for seeding a starting
        baseline -- which meant an explosion in the first few steps of
        training, before 5 real data points existed, could never be
        caught (see _check_for_trouble). Doing all three once up front,
        from the code alone, closes those gaps before training even
        starts.

        Best-effort and silent on failure -- this must never be the thing
        that makes a training run fail to start. Runs at most once per
        process (guarded by self._start_primed), and only if an agent
        provider/key is actually configured and the training code was
        made available via set_code_text.
        """
        if self._start_primed:
            return
        self._start_primed = True
        if not self.agent_provider or not self.agent_key or not self.code_text:
            return
        try:
            context = self._build_agent_context(include_code=True)
            answer = self._call_model(f"{context}\n\n{self._START_PRIME_PROMPT}", max_tokens=300)
        except AgentRequestFailed as exc:
            cprint(f"[Pulse] ⚠ Start-of-run sensitivity check skipped (agent request failed: {exc})", color=_YELLOW)
            return
        _, _calc, _promote, gputrack_names, _gpuu, sensitivity_args, normal_start_args = self._extract_directives(answer)
        summary = self._apply_directives([], [], gputrack_names, None, sensitivity_args, normal_start_args)
        if summary:
            cprint(f"[Pulse] Start-of-run check: {summary}", color=_YELLOW)

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

        self._prime_at_start()

        # Uptime: wall-clock time since the *previous* update() call handed
        # control back to the training loop, i.e. time actually spent in
        # the user's own training code (forward/backward/optimizer step).
        # Excludes any agent downtime and any interactive-prompt wait from
        # that previous call, since both are marked separately/excluded
        # below -- see _last_update_end_ts.
        _update_start_ts = time.monotonic()
        if self._last_update_end_ts is not None:
            gap = _update_start_ts - self._last_update_end_ts
            if gap > 0:
                self._uptime_seconds += gap
                self._cloud_dirty_fields.add("uptime_seconds")

        # Independent of the 10-minute telemetry batch timer (and of
        # whether telemetry is enabled at all) -- see uptime_flush_interval's
        # docstring in __init__ for why these two get their own, much
        # shorter cadence.
        if (
            self.debug_session_id
            and {"uptime_seconds", "downtime_seconds"} & self._cloud_dirty_fields
            and (_update_start_ts - self._last_uptime_flush) >= self.uptime_flush_interval
        ):
            self._last_uptime_flush = _update_start_ts
            self._maybe_flush_cloud(force=True)

        if loss_var is not None and loss_var in self.watch_locals:
            raw = self._cpu_observation(loss_var, self.watch_locals[loss_var])
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

        # 1. Check if the global GPU probe cadence is due
        # (Assuming self._last_gpu_probe is initialized to 0.0 in __init__)
        gpu_ready = (now - getattr(self, "_last_gpu_probe", 0.0)) >= getattr(self, "gpu_probe_interval", 1.0)

        # 2. Gate the matrix probes: they only trigger if BOTH their specific 
        # cadence (matrix_probe_interval/lotrack_probe_interval) AND the global 
        # gpu_ready gate permit it (or if it's the very first un-cached pass).
        probe_track = (
            not self._matrix_cache
            or (gpu_ready and (now - self._last_matrix_probe) >= self.matrix_probe_interval)
            or any(v not in self._matrix_cached_vars for v in self.tracked_vars if self._state_of(v) == "track")
        )
        
        probe_lotrack = (
            (gpu_ready and (now - self._last_lotrack_probe) >= self.lotrack_probe_interval)
            or any(v not in self._matrix_cached_vars for v in self.tracked_vars if self._state_of(v) == "lotrack")
        )
        
        probe_matrices = probe_track or probe_lotrack

        # 3. Update the global GPU probe timestamp if a matrix probe is actually firing
        if probe_matrices and self._matrix_cache:
            self._last_gpu_probe = now
            
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

            orig_val = self._cpu_observation(var_name, self.watch_locals[var_name])
            var_state = self._state_of(var_name)
            if orig_val is None and self.watch_locals[var_name] is not None:
                if _pulse_is_accelerator_value(self.watch_locals[var_name]):
                    if var_name not in self._cpu_only_warned:
                        cprint(f"  • '{var_name}' is accelerator-resident; Pulse will not touch it. Maintain a CPU mirror named '{var_name}_cpu' (or '{var_name}_np') to visualize it.", color=_YELLOW)
                        self._cpu_only_warned.add(var_name)
                    continue
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
                    cprint(f"  ⚠ '{var_name}' could not be read this step: {type(exc).__name__}: {exc}", color=_RED)
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
                    cprint(f"  ⚠ '{sub_name}' could not be read this step: {type(exc).__name__}: {exc}", color=_RED)
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
            cprint(f"--- Pulse Live Debugger | Step {self.step} ---")

            for sub_name, scalar_val in wrong_scalars:
                hist = self.scalar_histories.get(sub_name, [])
                if scalar_val is None:
                    cprint(f"  • {sub_name}: NoneType  ⚠ (unreadable this step)", color=_RED)
                else:
                    cprint(f"  • {sub_name}: {scalar_val:.6g}  ⚠", color=_RED)
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
                            cprint(f"    ↳ ⚠ failed to save PDF snapshot for '{sub_name}': {exc}", color=_RED)

                if self._matrix_cache:
                    print(
                        f"  [matrices cached: {len(self._matrix_cache)} | "
                        f"next full probe ≤ {self.matrix_probe_interval:g}s | "
                        f"next lotrack probe ≤ {self.lotrack_probe_interval:g}s]"
                    )

        self._maybe_collect_telemetry(scalar_lines, matrix_lines)
        self._maybe_gpu_checkin()

        if self.auto_intervene:
            problem = self._check_for_trouble()
            if problem and problem != self._last_intervention_signature:
                self._last_intervention_signature = problem
                self.continuous = True
                print("\n" + "=" * 60)
                cprint("[Pulse] ⚠ Auto-intervention: training looks like it's going bad. Diagnosing while preserving the loop.", color=_RED)
                cprint(f"[Pulse] Detected: {problem}", color=_RED)
                print("=" * 60)
                if self.agent_provider:
                    question = (
                        f"Pulse just auto-paused training because it detected a problem: {problem}\n"
                        "Please diagnose the root cause and, if you can, fix it."
                    )
                    cprint("Pulse:")
                    self._pending_agent_start_ts = time.monotonic()
                    self._pending_agent_problem = problem
                    self.ask_agent(question, include_code=bool(self.code_text))
                    # No-op if a fix inside ask_agent() already triggered a
                    # restart and _restart_process() finalized this first.
                    self._finalize_agent_downtime()
                else:
                    cprint("[Pulse] No AI agent is configured yet -- run /agent to set one up, then ask about this.")
                    self._log_incident("auto_intervention", problem, downtime_seconds=0.0, fix_applied=False)
                cprint("[Pulse] Continuing training automatically.")

        # Mark the automated portion of this update() call as finished
        # here -- before the interactive human-wait prompt below -- so
        # that wait time is counted as neither uptime nor agent downtime;
        # it's a paused-for-a-human period, not training or agent work.
        self._last_update_end_ts = time.monotonic()

        if self.continuous:
            return

        # Interactive Training Loop Prompt
        while True:
            try:
                _flush_stdin()
                cmd = input(
                    _highlight_pulse(
                        "\nPulse [Enter=step, /c=continuous, /help, or ask AI] > "
                    )
                ).strip()
            except (EOFError, KeyboardInterrupt):
                cprint("\nExiting Pulse...")
                if self.original_sigint and callable(self.original_sigint):
                    signal.signal(signal.SIGINT, self.original_sigint)
                raise KeyboardInterrupt

            if not cmd:
                break

            if cmd.lower() in ("/c", "/continue", "c"):
                self.continuous = True
                break

            if cmd.lower() in ("/help", "/h", "?", "/commands"):
                self._cmd_help(cmd)
                continue

            if cmd.lower().startswith("/add "):
                self._cmd_add(cmd[5:].strip())
                continue

            if cmd.lower().startswith("/track "):
                self._cmd_track(cmd[7:].strip())
                continue
            if cmd.lower().startswith("/lotrack "):
                self._cmd_lotrack(cmd[9:].strip())
                continue
            if cmd.lower().startswith("/gputrack "):
                self._cmd_gputrack(cmd[10:].strip())
                continue
            if cmd.lower().startswith("/gpuuntrack "):
                self._cmd_gpuuntrack(cmd[12:].strip())
                continue
            if cmd.lower().startswith("/autofix"):
                self._cmd_autofix(cmd[8:].strip())
                continue
            if cmd.lower().startswith("/sensitivity"):
                self._cmd_sensitivity(cmd[len("/sensitivity"):].strip())
                continue
            if cmd.lower().startswith("/telemetry"):
                self._cmd_telemetry(cmd[len("/telemetry"):].strip())
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

            if cmd.lower() == "/cloud":
                self._print_cloud_status()
                continue

            if cmd.lower() == "/cloud flush":
                self._maybe_flush_cloud(force=True)
                cprint("[Pulse] Cloud sync flushed.")
                continue

            if cmd.lower().startswith("/repo"):
                self._cmd_repo(cmd[5:].strip())
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

            if cmd.lower() == "/log":
                self._cmd_log("")
                continue

            if cmd.lower().startswith("/admin"):
                self._cmd_admin(cmd[6:].strip())
                continue
            if cmd.lower().startswith("/webhook"):
                self._cmd_webhook(cmd[8:].strip())
                continue
            if cmd.lower().startswith("/password"):
                self._cmd_password(cmd[9:].strip())
                continue
            if cmd.lower().startswith("/recover"):
                self._cmd_recover(cmd[8:].strip())
                continue
            if cmd.lower().startswith("/deleteaccount"):
                self._cmd_deleteaccount(cmd[14:].strip())
                continue
            if cmd.lower() == "/logout":
                self._cmd_logout("")
                continue
            if cmd.lower().startswith("/commit"):
                self._cmd_commit(cmd[7:].strip())
                continue

            if cmd.lower().startswith("/revert"):
                self._cmd_revert(cmd[7:].strip())
                continue

            cprint("Pulse AI:")
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
            cprint(f"    [{name} | {len(history)} points] ⚠ no readable values (None/NaN/inf) -- nothing to chart", color=_RED)
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
