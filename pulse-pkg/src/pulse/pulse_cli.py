"""
pulse_cli.py
============
"""

from __future__ import annotations

import os
import re
import sys
import io
import ast
import copy
import json
import time
import math
import uuid
import shutil
import signal
import getpass
import hashlib
import difflib
import inspect
import importlib
import importlib.util
import traceback
import itertools
import subprocess
import threading
import tempfile
import atexit
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pulse.pulse_backend import (
    available_backends,
    describe_tensor,
    detect_backend,
    is_trackable,
    shape_of,
    tensor_kind,
    statistics,
    to_numpy,
)
from pulse.pulse_pdf import generate_heatmap_pdf
from pulse import pulse_detect as _pulse_detect
from pulse import pulse_supabase as cloud
from pulse import pulse_ui as _ui
from pulse import pulse_terminal as _terminal
try:
    import litellm
    # Quiets litellm's own verbose provider-runtime logging unless explicitly enabled.
    if os.environ.get("PULSE_LITELLM_DEBUG", "").strip().lower() not in ("1", "true", "yes"):
        litellm.suppress_debug_info = True
except ImportError:
    print(
        "\n[Pulse] Missing dependency: litellm (used to talk to AI providers).\n"
        "         Install it with:  pip install litellm\n",
        file=sys.stderr,
    )
    sys.exit(1)




# ---------------------------------------------------------------------------
# TEMPORARY DEBUG INSTRUMENTATION
# Enable with PULSE_LOGGING=1 (default ON in this logging build).
# Writes a low-volume trace to ./pulse.log without dumping tensor data.
# ---------------------------------------------------------------------------
_PULSE_LOGGING = os.environ.get("PULSE_LOGGING", "1").strip().lower() not in ("0", "off", "false", "no")
_PULSE_LOG_FILE = os.path.abspath(os.environ.get("PULSE_LOG_FILE", "pulse.log"))
_PULSE_KERAS_TRACKER_CLS = None


def _build_keras_tracker_class(callback_base):
    """Define the Keras callback against whichever Callback base is actually loaded.

    This used to be a module-level `import tensorflow` guarded by try/except, with the
    class defined only when the import succeeded. That made merely importing Pulse pay
    for a full TensorFlow import -- seconds, and GPU memory on some builds -- for
    PyTorch, JAX and NumPy users who never touch Keras, while users without TensorFlow
    got no class at all. Built on demand instead, from a module that is already loaded.
    """
    global _PULSE_KERAS_TRACKER_CLS
    if _PULSE_KERAS_TRACKER_CLS is not None:
        return _PULSE_KERAS_TRACKER_CLS

    class PulseKerasTracker(callback_base):
        """
        Internal Pulse -> Keras bridge.

        Users do not need to add this callback themselves.
        Pulse injects it automatically into Model.fit().
        """

        def __init__(self, pulse_instance):
            super().__init__()
            self.pulse = pulse_instance

        def on_train_begin(self, logs=None):
            if not hasattr(self.pulse, "epoch_scalar_histories"):
                self.pulse.epoch_scalar_histories = {}

            if not hasattr(self.pulse, "batch_scalar_histories"):
                self.pulse.batch_scalar_histories = {}

        def on_train_batch_end(self, batch, logs=None):
            # Keep batch handling extremely cheap.
            try:
                self.pulse._record_keras_batch_logs(logs)
            except Exception as exc:
                if _PULSE_LOGGING:
                    try:
                        _pulse_log(
                            "KERAS BATCH LOG ERROR "
                            f"{type(exc).__name__}: {exc}"
                        )
                    except Exception:
                        pass

        def on_epoch_end(self, epoch, logs=None):
            try:
                self.pulse._record_keras_logs(
                    logs,
                    epoch=epoch,
                )
            except Exception as exc:
                if _PULSE_LOGGING:
                    try:
                        _pulse_log(
                            "KERAS EPOCH LOG ERROR "
                            f"{type(exc).__name__}: {exc}"
                        )
                    except Exception:
                        pass
                return

            # What a fix-time checkpoint needs: the model being trained and where it is.
            try:
                self.pulse._keras_model = self.model
                self.pulse._keras_epoch = epoch
                self.pulse._keras_epochs = (self.params or {}).get("epochs")
            except Exception:
                pass

            # The epoch boundary is the correct point to run diagnosis.
            # We do NOT run the expensive detector on every batch.
            try:
                if getattr(self.pulse, "auto_intervene", False):
                    self.pulse.update()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if _PULSE_LOGGING:
                    try:
                        _pulse_log(
                            "KERAS EPOCH UPDATE ERROR "
                            f"{type(exc).__name__}: {exc}"
                        )
                    except Exception:
                        pass

            if _PULSE_LOGGING:
                try:
                    histories = getattr(
                        self.pulse,
                        "epoch_scalar_histories",
                        {},
                    )

                    summary = {
                        k: v[-1]
                        for k, v in histories.items()
                        if v
                    }

                    _pulse_log(
                        "KERAS TRACKER "
                        f"epoch={epoch} "
                        f"metrics={summary!r}"
                    )
                except Exception:
                    pass

        def on_train_end(self, logs=None):
            if _PULSE_LOGGING:
                try:
                    histories = getattr(
                        self.pulse,
                        "epoch_scalar_histories",
                        {},
                    )

                    summary = {
                        k: len(v)
                        for k, v in histories.items()
                    }

                    _pulse_log(
                        "KERAS TRAIN END "
                        f"histories={summary!r}"
                    )
                except Exception:
                    pass

    _PULSE_KERAS_TRACKER_CLS = PulseKerasTracker
    return PulseKerasTracker


_PULSE_KERAS_HOOK_INSTALLED = False
_PULSE_ACTIVE_INSTANCE = None
_PULSE_KERAS_IMPORT_WATCHER = None


class _KerasImportWatcher:
    """Patches Keras the moment it is imported, without importing it ourselves.

    auto_track() used to `import tensorflow` outright. That raised ImportError on
    every machine without TensorFlow -- a NumPy, PyTorch or JAX user could not call
    auto_track() at all -- and on machines that do have it, it paid seconds of import
    time and hundreds of MB for a framework the script may never touch.

    Deferring is not enough on its own: the usual shape is `auto_track()` at the top
    of the script and `import tensorflow` below it, so a one-shot check at auto_track
    time would miss it. This sits in sys.meta_path, lets the normal machinery do the
    import, and wraps the loader so the hook goes in as soon as the module finishes
    executing.
    """

    _TARGETS = ("tensorflow", "keras")

    def __init__(self):
        self._resolving = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._TARGETS or self._resolving:
            return None
        self._resolving = True          # our own find_spec re-enters meta_path
        try:
            spec = importlib.util.find_spec(fullname)
        except (ImportError, AttributeError, ValueError):
            return None
        finally:
            self._resolving = False
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None or not hasattr(loader, "exec_module"):
            return None
        original_exec_module = loader.exec_module

        def exec_module(module, _original=original_exec_module):
            _original(module)
            _install_keras_fit_hook()   # the module is fully executed by now

        try:
            loader.exec_module = exec_module
        except (AttributeError, TypeError):
            return None                 # immutable loader: leave the import alone
        return spec


def _keras_callback_base():
    """keras.callbacks.Callback if Keras is ALREADY imported, else None."""
    tf_module = sys.modules.get("tensorflow")
    keras_module = getattr(tf_module, "keras", None) if tf_module is not None else None
    if keras_module is None:
        keras_module = sys.modules.get("keras")
    callbacks = getattr(keras_module, "callbacks", None)
    return getattr(callbacks, "Callback", None)


def _keras_model_class():
    """tf.keras.Model / keras.Model if either is ALREADY imported, else None.

    Only reads sys.modules: asking for a module we have not imported would trigger
    the very import this whole dance exists to avoid.
    """
    tf_module = sys.modules.get("tensorflow")
    keras_module = getattr(tf_module, "keras", None) if tf_module is not None else None
    if keras_module is None:
        keras_module = sys.modules.get("keras")
    return getattr(keras_module, "Model", None)


def _install_keras_fit_hook():
    """Wrap Model.fit so Pulse's callback rides along. Safe to call repeatedly."""
    global _PULSE_KERAS_HOOK_INSTALLED

    if _PULSE_KERAS_HOOK_INSTALLED or _PULSE_ACTIVE_INSTANCE is None:
        return
    model_cls = _keras_model_class()
    callback_base = _keras_callback_base()
    if model_cls is None or callback_base is None:
        return
    tracker_cls = _build_keras_tracker_class(callback_base)

    original_fit = model_cls.fit

    def pulse_fit(self, *args, **kwargs):
        _resume_from_fix_checkpoint(self, args, kwargs)
        callbacks = kwargs.get("callbacks")

        if callbacks is None:
            callbacks = []
        else:
            callbacks = list(callbacks)

        callbacks.append(
            tracker_cls(_PULSE_ACTIVE_INSTANCE)
        )

        kwargs["callbacks"] = callbacks

        return original_fit(self, *args, **kwargs)

    try:
        model_cls.fit = pulse_fit
    except (AttributeError, TypeError) as exc:      # exotic//frozen Keras build
        _pulse_log(f"KERAS FIT HOOK could not be installed: {exc}")
        return
    _PULSE_KERAS_HOOK_INSTALLED = True

    _pulse_log("KERAS FIT HOOK installed")


# Checkpoint-and-resume around a fix. When Pulse starts fixing a running Keras job it saves
# the model's weights (PulseCLI._save_fix_checkpoint); after the fix is written, the
# restarted script continues from that point instead of retraining from scratch:
# _resume_from_fix_checkpoint loads the weights into the first model.fit() and starts it
# at the next epoch. A fresh start is used instead when the fix asked for one ("resume":
# false -- e.g. an initializer, architecture, data or label fix, where the saved weights
# carry the bug), when the saved weights were not finite, or when they do not fit the fixed
# model. Keras only; other frameworks restart from the beginning as before.
_RESUME_ENV = "PULSE_RESUME_CHECKPOINT"


def _fit_epochs(args, kwargs) -> Optional[int]:
    """The `epochs` a model.fit() call was given, positionally or by keyword."""
    if "epochs" in kwargs:
        return kwargs["epochs"]
    # fit(x, y, batch_size, epochs, ...)
    return args[3] if len(args) > 3 else 1


def _resume_from_fix_checkpoint(model, args, kwargs) -> None:
    meta_path = os.environ.pop(_RESUME_ENV, "")      # once: the first fit() of the restarted run
    if not meta_path:
        return
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not getattr(model, "built", False):
            _agent_log_event("RESUME SKIPPED -- the model is not built before fit(), starting fresh")
            return
        before = [w.copy() for w in model.get_weights()]
        model.load_weights(meta["weights"], skip_mismatch=True)
        after = model.get_weights()
        loaded = sum(1 for a, b in zip(before, after) if a.shape == b.shape and not np.array_equal(a, b))
        if not all(np.all(np.isfinite(w)) for w in after):
            model.set_weights(before)
            _agent_log_event("RESUME ABANDONED -- loaded weights were not finite, starting fresh")
            return
        epochs = _fit_epochs(args, kwargs)
        start = int(meta.get("epoch", -1)) + 1
        if isinstance(epochs, int) and epochs > 0:
            start = max(0, min(start, epochs - 1))     # always train at least one epoch
        kwargs["initial_epoch"] = max(int(kwargs.get("initial_epoch", 0) or 0), start)
        msg = (f"[Pulse] Resuming from the checkpoint taken when the fix started: "
               f"{loaded}/{len(after)} weight tensors restored, continuing at epoch {kwargs['initial_epoch'] + 1}.")
        cprint(msg, color=_YELLOW)
        _agent_log_event("RESUMED FROM CHECKPOINT",
                         f"{loaded}/{len(after)} weight tensors restored; continuing at epoch "
                         f"{kwargs['initial_epoch'] + 1} of {epochs}")
    except Exception as exc:
        cprint(f"[Pulse] Could not resume from the fix checkpoint ({type(exc).__name__}: {exc}) -- training from the start.", color=_YELLOW)
        _agent_log_event("RESUME FAILED -- starting fresh", f"{type(exc).__name__}: {exc}")


def _install_keras_pulse_hook(pulse_instance):
    """Arrange for Pulse to see Keras training, whenever Keras shows up.

    Never imports TensorFlow or Keras. If one of them is already imported the hook
    goes in now; otherwise a sys.meta_path watcher installs it if and when the
    script imports one.
    """
    global _PULSE_ACTIVE_INSTANCE
    global _PULSE_KERAS_IMPORT_WATCHER

    _PULSE_ACTIVE_INSTANCE = pulse_instance

    if _PULSE_KERAS_HOOK_INSTALLED:
        return

    _install_keras_fit_hook()
    if _PULSE_KERAS_HOOK_INSTALLED or _PULSE_KERAS_IMPORT_WATCHER is not None:
        return
    watcher = _KerasImportWatcher()
    sys.meta_path.insert(0, watcher)
    _PULSE_KERAS_IMPORT_WATCHER = watcher
# Agent log: what Pulse's AI was shown, what it answered, and what Pulse did about it --
# every model call (system prompt, conversation, reply, timing, failures) plus the actions
# that follow (check-in verdicts, escalations, applied fixes, restarts). Plain
# text, readable as-is. pulse.log above is the internal trace; this one is for people
# asking "why did Pulse do that?".
#
# OFF unless asked for: it records every prompt in full, code included, so a long run
# writes megabytes. Turn it on with `pulse run --agent-log[=PATH] script.py`, or
# PULSE_AGENT_LOG=1 (-> ./pulse_agent.log) / PULSE_AGENT_LOG=<path>. Read at write time,
# not import time, so the flag works however Pulse was imported, and a fix-triggered
# restart (which inherits the environment) keeps logging to the same file.
_AGENT_LOG_ENV = "PULSE_AGENT_LOG"
_agent_log_lock = threading.Lock()
_agent_log_seen: Dict[str, int] = {}     # message hash -> the number it was logged under
_agent_log_call_no = 0
_agent_log_resolved: Dict[str, str] = {}  # setting -> absolute path, fixed at first write


def _agent_log_path() -> Optional[str]:
    setting = os.environ.get(_AGENT_LOG_ENV, "").strip()
    if setting.lower() in ("", "0", "off", "false", "no"):
        return None
    if setting not in _agent_log_resolved:
        target = "pulse_agent.log" if setting.lower() in ("1", "on", "true", "yes") else setting
        _agent_log_resolved[setting] = os.path.abspath(os.path.expanduser(target))
        # Pin it for anything this process starts (a restart may run from elsewhere).
        os.environ[_AGENT_LOG_ENV] = _agent_log_resolved[setting]
        _agent_log_resolved[_agent_log_resolved[setting]] = _agent_log_resolved[setting]
    return _agent_log_resolved[setting]


def _agent_log_write(text: str) -> None:
    path = _agent_log_path()
    if not path:
        return
    try:
        with _agent_log_lock, open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def _agent_log_event(title: str, body: str = "") -> None:
    """One thing Pulse did: a verdict, an escalation, a fix, a restart."""
    stamp = time.strftime("%H:%M:%S")
    text = f"\n>>> [{stamp}] {title}\n"
    if body:
        text += "".join(f"    {line}\n" for line in str(body).splitlines())
    _agent_log_write(text)


def _agent_log_call(purpose: str, messages: List[Dict[str, str]], model: str) -> int:
    """Log what a model call is being shown. Each distinct message is written out in full
    the first time and referred to by number after that -- the investigation passes resend
    the same conversation on every call, and the log should show that without repeating it."""
    global _agent_log_call_no
    if not _agent_log_path():
        return 0
    with _agent_log_lock:
        _agent_log_call_no += 1
        call_no = _agent_log_call_no
    parts = [f"\n{'=' * 78}\nCALL #{call_no} [{time.strftime('%H:%M:%S')}] {purpose}  ({model})\n"
             f"{'=' * 78}\nThe model sees {len(messages)} message(s):\n"]
    for msg in messages:
        content = str(msg.get("content", ""))
        key = hashlib.sha1((msg.get("role", "") + "\0" + content).encode("utf-8", "replace")).hexdigest()
        with _agent_log_lock:
            seen = _agent_log_seen.get(key)
            if seen is None:
                _agent_log_seen[key] = seen = len(_agent_log_seen) + 1
                first = True
            else:
                first = False
        if first:
            parts.append(f"--- {msg.get('role', '?')} [message M{seen}, {len(content):,} chars] ---\n{content}\n")
        else:
            parts.append(f"--- {msg.get('role', '?')} [message M{seen}, same as before, {len(content):,} chars] ---\n")
    _agent_log_write("".join(parts))
    return call_no


def _agent_log_reply(call_no: int, reply: Optional[str], seconds: float, usage=None,
                     error: Optional[str] = None) -> None:
    if not _agent_log_path():
        return
    tokens = ""
    if usage is not None:
        tokens = (f", {getattr(usage, 'prompt_tokens', '?')} tokens in / "
                  f"{getattr(usage, 'completion_tokens', '?')} out")
    if error is not None:
        _agent_log_write(f"--- CALL #{call_no} FAILED after {seconds:.1f}s: {error}\n")
    else:
        _agent_log_write(f"--- CALL #{call_no} reply ({seconds:.1f}s{tokens}) ---\n{reply}\n")


def _pulse_log(message: str, *, console: bool = False) -> None:
    if not _PULSE_LOGGING:
        return
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    try:
        with open(_PULSE_LOG_FILE, "a", encoding="utf-8") as _f:
            _f.write(line + "\n")
    except Exception:
        pass
    

import builtins

_io_lock = threading.Lock()
_original_print = builtins.print
_original_input = builtins.input



def _stdout_is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def safe_print(*args, **kwargs):
    with _io_lock:
        # Move cursor to column 0 and clear line before printing background text -- but only
        # when this print is going to a terminal. Into a pipe, a log file or a notebook cell
        # the escape is not a control sequence, just junk at the start of every line.
        if kwargs.get("file") in (None, sys.stdout) and _stdout_is_tty():
            sys.stdout.write("\r\033[K")
        _original_print(*args, **kwargs)

def safe_input(prompt=""):
    # Clear formatting and force prompt to a clean new line. Off a terminal keep the new line
    # (it is what keeps each prompt on its own line in a log) and drop only the escape codes.
    sys.stdout.write("\033[0m\n\r\033[K" if _stdout_is_tty() else "\n")
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


def _trackable_without_reading(value: Any) -> bool:
    """is_trackable(), except that a value living on an accelerator is judged from its
    metadata alone (it has a shape and a dtype) and never handed to the backend.

    The tracer asks this about every local in scope, on the training thread, each time a
    window opens. is_trackable() lives in the backend; if it ever converts or copies to
    decide ("is this interesting?") that is a device-to-host transfer and a synchronization
    per tensor per window -- stalls the training loop never asked for. Anything not on an
    accelerator goes to the backend exactly as before.
    """
    if _pulse_is_accelerator_value(value):
        return getattr(value, "shape", None) is not None and getattr(value, "dtype", None) is not None
    return is_trackable(value)


LOSS_NAME_HINTS = ("loss", "cost", "nll", "cross_entropy", "crossentropy", "objective", "err")

METRIC_NAME_HINTS = (
    "acc", "accuracy", "precision", "recall", "f1", "auc", "iou", "dice",
    "score", "bleu", "rouge", "map", "psnr", "ssim",
)


def _looks_like_loss(name: str) -> bool:
    n = (name or "").lower()
    return any(hint in n for hint in LOSS_NAME_HINTS)


def _looks_like_metric(name: str) -> bool:
    """Accuracy/precision/recall/F1/AUC/etc-style variables -- distinct
    from loss-like ones (see LOSS_NAME_HINTS) because a metric is
    normally expected to trend UP, not down, but the same "hasn't moved
    in a long time" stagnation shape is just as worth flagging either
    way: an accuracy stuck dead flat for hundreds of steps is at least as
    strong a signal as a loss that's stopped improving, and previously
    had NO automatic check at all since _check_for_trouble only ever
    looked at loss-like names."""
    n = (name or "").lower()
    return any(hint in n for hint in METRIC_NAME_HINTS)


def _looks_like_grad_or_weight_norm(name: str) -> bool:
    """A tracked scalar that's some kind of gradient/parameter norm
    (grad_norm, gradient_norm, weight_norm, param_norm, ...) rather than
    a loss or a metric -- these should be roughly stable, so a sudden
    multiplicative jump (exploding gradients) or collapse toward zero
    (vanishing gradients) is itself the anomaly, independent of what the
    loss curve happens to be doing at the same time."""
    n = (name or "").lower()
    return any(
        hint in n for hint in (
            "grad_norm", "gradient_norm", "grad norm", "gradnorm",
            "weight_norm", "param_norm", "parameter_norm", "weightnorm",
        )
    )


def _looks_like_learning_rate(name: str) -> bool:
    """A tracked scalar that's the current learning rate -- expected to
    change only smoothly/deliberately (a scheduler step, a warmup ramp),
    so a sudden large multiplicative jump between consecutive
    observations usually means the scheduler itself is misconfigured
    (wrong milestone/step count, wrong decay factor, stepped more than
    once per actual optimizer step) rather than intended behavior."""
    n = (name or "").lower()
    return n in ("lr", "learning_rate", "learningrate") or n.endswith(("_lr", "_learning_rate"))


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
_GREEN = "\033[92m"
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


# ---- terminal UI adapters (see pulse_ui.py) -----------------------------------------
# Presentation only. Each adapter hands back exactly what the plain prompt it stands in for
# would have returned, so the logic around every call site is untouched -- and each acts
# only on a real interactive terminal. Anywhere else (pipes, CI, notebooks, PULSE_PLAIN=1,
# or a terminal that turns out not to support it) the call site's original print()/input()
# code runs unchanged.

def _prompt_text(plain_prompt: str, *, label: Optional[str] = None, secret: bool = False,
                 placeholder: Optional[str] = None, validate=None) -> str:
    """input(plain_prompt) / getpass.getpass(plain_prompt), as a styled field on a terminal."""
    if _ui.enabled():
        try:
            _flush_stdin()
            return _ui.ask(label or plain_prompt.strip().rstrip(">").strip(),
                           secret=secret, placeholder=placeholder, validate=validate)
        except _ui.Unavailable:
            pass
    return getpass.getpass(plain_prompt) if secret else input(plain_prompt)


def _ui_screen(state: str, subtitle: Optional[str] = None, explain: Optional[str] = None) -> None:
    if _ui.enabled():
        _ui.header(state, subtitle, explain)


def _ui_step(text: str) -> None:
    """A sub-step inside the screen already on display (e.g. "Log in" under PULSE / ACCOUNT)."""
    if _ui.enabled():
        _ui.subhead(text)


def _ui_done(label: str, value: Optional[str] = None) -> None:
    """Collapse a finished field to a check line."""
    if _ui.enabled():
        _ui.ok(label, value)


def _say(plain: str, label: Optional[str] = None, value: Optional[str] = None,
         color: Optional[str] = None) -> None:
    """Report a finished step: a collapsed check line on a terminal UI, else the original line."""
    if label is not None and _ui.enabled():
        _ui.ok(label, value)
    else:
        cprint(plain, color=color)


def _password_progress(text: str) -> str:
    return f"{len(text)}/8 characters" if len(text) < 8 else ""


def _ui_account_menu() -> Optional[str]:
    """Sign up / log in / recover as an arrow-key menu -> "s", "l" or "r" (the letters the
    plain prompt takes; Enter still means sign up). None: use the plain prompt."""
    if not _ui.enabled():
        return None
    try:
        _flush_stdin()
        _ui.header("Account", "Welcome to Pulse.",
                   "Your runs, history and debugging context sync to your Pulse account.")
        idx = _ui.choose(
            [
                _ui.Option("Sign up", "Create a Pulse account", tag="default", key="s"),
                _ui.Option("Log in", "Use an existing account", key="l"),
                _ui.Option("Recover account", "Reset your password with your recovery code", key="r"),
            ],
            initial=0, allow_cancel=False,
            footer=f"{_ui._g('up')}{_ui._g('down')} Navigate    Enter Select    S / L / R Jump",
        )
    except _ui.Unavailable:
        return None
    return "slr"[idx]


def _ui_workspace_menu(existing: List[Dict[str, Any]], cached_team_id: Optional[str], describe) -> Optional[str]:
    """The workspace picker -> the same token the plain prompt takes: "1".."n" for a listed
    workspace, or "c" / "j" / "r" / "l" / "d". None: use the plain prompt."""
    if not _ui.enabled():
        return None
    try:
        _flush_stdin()
        _ui.header("Workspace", "Select a workspace",
                   "Your runs, history and debugging context live here.")
        options: List[Any] = []
        initial = 0
        for i, team in enumerate(existing):
            name, _sep, rest = describe(team).partition("  --  ")
            last_used = team.get("team_id") == cached_team_id
            if last_used:
                initial = i
            options.append(_ui.Option(name, rest or None, tag="last used" if last_used else None))
        options.append(_ui.Option("+ Create new workspace", "Start a new one", key="c"))
        options.append(_ui.Option("Join with a code", "Join a teammate's workspace", key="j"))
        manage = "    R Rename    L Leave    D Delete" if existing else ""
        picked = _ui.choose(
            options, initial=initial, allow_cancel=False,
            hotkeys={"r": "r", "l": "l", "d": "d"} if existing else None,
            footer=f"{_ui._g('up')}{_ui._g('down')} Select    Enter Continue    C New    J Join{manage}",
        )
    except _ui.Unavailable:
        return None
    if isinstance(picked, str):
        return picked
    return str(picked + 1) if picked < len(existing) else ("c" if picked == len(existing) else "j")


def _say_plain(label: str, value: str, plain: str) -> None:
    """Like _say, for the places that used a bare print() rather than cprint()."""
    if _ui.enabled():
        _ui.ok(label, value)
    else:
        print(plain)


def _ui_pick_agent(names: List[str], cached_provider: Optional[str], current: Optional[str]) -> Optional[str]:
    """A searchable list of the registered providers. Returns the chosen provider's exact
    name, "" if the person skipped (Esc -- what pressing Enter on a blank line meant), or
    None to use the plain numbered list. Everything shown comes from PROVIDERS: the name,
    its model id, and the local/custom/OpenRouter flags. Nothing is described that the
    registry does not already say."""
    if not _ui.enabled():
        return None
    try:
        _flush_stdin()
        _ui.header("Agent model", "Select the model that powers Pulse's analysis.")
        options: List[Any] = []
        initial = 0
        for i, name in enumerate(names):
            info = PROVIDERS[name]
            if info.get("custom"):
                detail = "type any provider/model string"
            elif info.get("openrouter"):
                detail = "type any OpenRouter model"
            elif info.get("local"):
                detail = "local -- no data leaves this machine"
            else:
                detail = info.get("model") or None
            tag = "current" if name == current else ("last used" if name == cached_provider else None)
            if tag and (name == current or initial == 0):
                initial = i
            options.append(_ui.Option(name, detail, tag=tag))
        picked = _ui.choose(options, initial=initial, searchable=True, max_rows=7)
    except _ui.Unavailable:
        return None
    return "" if picked is None else names[picked]


def _ui_key_screen(chosen: str) -> None:
    if not _ui.enabled():
        return
    _ui.header("API key", "Connect your agent model")
    _ui.kv("Model", chosen, indent=0)
    _ui.note("Used only to call the model you selected. Pulse never writes it to disk.")
    _ui._emit()


def _ui_code_version(sha: Optional[str], cwd: Optional[str]) -> None:
    """Show the commit this run is attached to. Detection already happened; this only says so."""
    if not _ui.enabled():
        return
    if not sha or sha == "unknown":
        _ui.note("○ Code version   unknown -- no git commit found (/commit sets one)")
        return
    subject = None
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%s", sha], cwd=cwd or None,
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0:
            subject = out.stdout.strip()[:60] or None
    except Exception:
        subject = None
    _ui.ok("Code version", sha[:10] + (f'  "{subject}"' if subject else ""))


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
    "  GREP: <pattern>\n"
    "    You don't need to ask for a whole file to check one detail. Search every tracked project "
    "file for a pattern (a plain word/phrase, or a regex) and get back each match with a line of "
    "context on either side -- e.g. 'GREP: learning_rate' or 'GREP: def forward'. Use this to "
    "locate where something is defined/used before deciding whether you need the full surrounding "
    "block via VIEW. Capped to a modest number of matches; narrow the pattern if it's truncated.\n"
    "  VIEW: <file>:<start>-<end>\n"
    "    Pull an exact line range from a specific file (the file label is whatever header you were "
    "shown, e.g. 'model.py'; omit '<file>:' to default to the main script), e.g. 'VIEW: "
    "model.py:40-75' or 'VIEW: 120-160'. Use this after a GREP hit (or a line number from a "
    "traceback/diagnosis) to see the exact surrounding code before proposing a fix, instead of "
    "relying on a possibly-stale full-file dump from earlier in the conversation. Capped to a few "
    "hundred lines per call -- issue more than one VIEW if you need a wider span.\n"
    "  Put CALC:/PROMOTE:/GPUTRACK:/GPUUNTRACK:/SENSITIVITY:/NORMAL_START:/GREP:/VIEW: lines "
    "anywhere in your Reasoning, not in the Diagnosis or Fix. GREP/VIEW results come back as a new "
    "message before your next turn -- if what you get back changes your diagnosis, say so.\n\n"

    "MORE TOOLS -- the same idea taken further: anything Pulse can just compute or look up for you "
    "beats you reasoning your way to a guess. Use these the same way as above (own line, anywhere in "
    "Reasoning); each returns a result as a new message before your next turn.\n"
    "  Execution (the single biggest lever -- no amount of reasoning substitutes for actually running "
    "code; these run against the current tracked-variable snapshot in the live process):\n"
    "    REPL: <expr> -- evaluate an expression against the LIVE tracked variables right now, "
    "instead of reasoning from a context dump that may already be stale.\n"
    "    DRYRUN: <function_or_method_call> -- actually execute a specific call (e.g. "
    "'DRYRUN: model.forward(x)') and get back the real output, exception, and traceback. Turns 'I "
    "think this will raise a shape error' into an actual answer.\n"
    "    SHAPETRACE: [optional model variable name] -- run one real forward pass with a live tracked "
    "tensor and dump every submodule's input/output shape. PyTorch only.\n"
    "    GRADCHECK: <param_name> -- numerical finite-difference gradient check on a tracked "
    "parameter, deterministic pass/fail against its real .grad. Needs a zero-arg `loss_fn` callable "
    "among tracked variables that recomputes the current loss -- if none exists, Pulse will say so.\n"
    "    REPLAY: <n_steps> -- replay the last n_steps from an isolated checkpoint (with a zero-arg "
    "`train_step` callable you define) and report the resulting loss curve, then restore live state.\n"
    "    TERMINAL: <shell command> -- run a REAL command in the project's working directory and get "
    "back its actual stdout, stderr, exit code and duration -- e.g. 'TERMINAL: pytest tests/test_model.py', "
    "'TERMINAL: python -m py_compile model.py', 'TERMINAL: git status', 'TERMINAL: grep -R \"loss_val\" .'. "
    "This is not one of the narrow tools above with a fixed shape -- you compose the command yourself, the "
    "way you would at a real shell. Use it to inspect files, search code, run linters/tests, check git "
    "state, reproduce a bug, or run a small diagnostic script. NEVER assume a command succeeded because "
    "you ran it or because a fix 'should' work -- always read the exit code and stdout/stderr you get back "
    "before saying something is fixed. After changing code, prefer verifying it for real: compile/syntax-"
    "check the file, run the specific failing test or a small repro script, and only report success once "
    "you've actually seen it pass. If a command you ran fails, investigate the real output (another GREP/ "
    "VIEW/TERMINAL) and try again rather than guessing at a second fix blind. A command that deletes "
    "files, rewrites git history, reaches outside the project, or starts a background process will ask "
    "the user to confirm before it runs -- expect that pause for those specific cases, not for ordinary "
    "commands like running a test or reading a file. Large output is truncated (head and tail kept, with "
    "a note); narrow the command (grep/head/tail, a single test) if you need something you can't see. Add "
    "' --timeout=<seconds>' at the end of the line to override the default timeout for a long-running "
    f"command (default {int(_terminal.DEFAULT_TIMEOUT_SECONDS)}s, capped at {int(_terminal.MAX_TIMEOUT_SECONDS)}s).\n"
    "  Code intelligence beyond GREP/VIEW (structure, not text search):\n"
    "    DEFOF: <symbol> -- AST-based jump-to-definition across every tracked file.\n"
    "    CALLERS: <symbol> -- every call site of a function/class.\n"
    "    DEPGRAPH: -- the import graph between tracked local files.\n"
    "  Statistics over a variable's whole recorded history (scalars only), not eyeballing a chart:\n"
    "    CORR: <var1> <var2> -- real correlation coefficient between two histories.\n"
    "    OUTLIER: <var> -- deterministic z-score anomaly detection, flags exact points.\n"
    "    DIFFSTATS: <var> <index_a> <index_b> -- exact delta between two recorded points.\n"
    "    HISTOGRAM: <var> -- actual bucketed distribution counts.\n"
    "  Grounding (lookup beats memorized/stale recall):\n"
    "    DOCLOOKUP: <library>.<symbol> -- real signature/docstring for an installed library function.\n"
    "    CHANGELOG: -- diff of what's actually changed in tracked files since the last checkpoint.\n"
    "    PASTFIX: <symbol_or_region> -- search prior fixes for the same area, both this project's own "
    "local log AND (when team history is loaded) every fix any teammate has applied on this repo/team.\n"
    "  Multi-GPU:\n"
    "    GPUSTATUS: -- real per-device memory/utilization for every visible GPU. If this run is one "
    "rank of a multi-process, one-rank-per-GPU launch (torchrun/etc.), this aggregates every rank's "
    "GPU status instead of just this one (other ranks run headless, so this chat is always rank 0).\n"
    "  Safety net:\n"
    "    ROLLBACK: <commit_id, or 'last'> -- deterministically revert the workspace to a prior state "
    "via Pulse's own persistent change log (see /log), instead of trying to manually reconstruct old "
    "code from memory. Note: if a fix you just applied causes repeated restart failures, Pulse already "
    "rolls back to the pre-fix state automatically -- you don't need to invoke this for that case, only "
    "for a deliberate revert mid-conversation.\n"
    "  ML health/anti-patterns (heuristic -- worth double-checking, not guaranteed):\n"
    "    MLLINT: -- a handful of AST-detectable ML anti-patterns: metric/loss mismatches in either "
    "direction (e.g. 'accuracy' against a regression loss, or 'mae' against a classification loss), "
    "double-softmax/double-sigmoid into a loss that already applies one, plain Softmax feeding NLLLoss "
    "(which expects log-probabilities), missing zero_grad(), backward()+zero_grad() with no optimizer "
    "step() ever taken, suspiciously high literal learning rates, eval-looking functions missing "
    "no_grad()/.eval(). This already runs automatically once at the very start of every run (see "
    "'Start-of-run ML anti-pattern check') and auto-triggers a diagnosis if it finds something -- "
    "calling it again yourself is for re-checking after a fix.\n"
    "    LAYERSTATS: [optional model var] -- per-layer gradient-norm/weight-norm ratio for a tracked "
    "torch model, right after backward(). Flags likely dead (near-zero ratio) or exploding (large "
    "ratio) layers individually, instead of only seeing an aggregate gradient norm.\n"
    "    HARDEXAMPLES: [optional N, default 10] -- per-example loss for the current batch, surfacing "
    "the highest-loss samples. Needs a zero-arg `per_example_losses_fn` callable among tracked "
    "variables (a reduction='none' loss call). Often the fastest way to spot label noise.\n"
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
    "PERIODIC CHECK-INS:\n"
    "While training runs, a separate check-in agent reviews it for bugs and instability. When a "
    "message here starts '[periodic check-in]', that is its finding, with its evidence: verify it "
    "against the code and numbers yourself rather than taking it on trust.\n\n"

    "CODE FIXES:\n"
    "If, and only if, the user explicitly asks you to fix, edit, patch, or change the code "
    "(not just diagnose it), respond with ONLY a single JSON object and nothing else -- no prose "
    "before or after it, no markdown code fences.\n"
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
    "  old: a list of code snippets to find, each copied EXACTLY from the line-numbered code "
    "shown to you, including original indentation and whitespace, but WITHOUT the line-number "
    "prefix ('  12 | ') itself.\n"
    "  new: a list of the same length as old, where new[i] is the full replacement for old[i].\n"
    "  files: OPTIONAL, a list of the same length as old, where files[i] is the exact file header "
    "(e.g. \"model.py\") that old[i]/new[i] belongs to, if more than one file was sent this turn. "
    "Omit this field entirely (or use null/\"\" for an entry) to default to the main script.\n"
    "  explanation: a short, concise text description of what changed and why.\n"
    "  resume: OPTIONAL, true or false. Training resumes from the weights it had when this fix "
    "started (true, the default) unless you set false. Set false when those weights carry the "
    "bug and must be retrained from scratch: the fix changes initialization, the architecture, "
    "the data, the labels or the preprocessing, or the weights are already damaged (NaN, "
    "exploded, collapsed to a constant output). Keep true for a learning rate, loss, optimizer, "
    "regularization or logging fix.\n"
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
    "OpenRouter (DeepSeek V4 Flash)": {
        "model": "openrouter/deepseek/deepseek-v4-flash",
        "env_key": "OPENROUTER_API_KEY",
    },
    # Sentinel entry: picking this prompts for any OpenRouter model slug
    # (one OPENROUTER_API_KEY reaches every model OpenRouter serves), then
    # registers a real PROVIDERS entry for it -- see
    # register_openrouter_model and _select_agent_provider_and_key's
    # "openrouter" branch.
    "OpenRouter (any model)": {
        "openrouter": True,
        "env_key": "OPENROUTER_API_KEY",
        "model_hint": "an OpenRouter model slug, e.g. deepseek/deepseek-v4-flash, anthropic/claude-sonnet-5",
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
    # Sentinel entry: picking this prompts for a raw litellm model string
    # (e.g. "openai/gpt-6-astra") plus an optional env var for its key,
    # then dynamically registers a real PROVIDERS entry for it -- see
    # _select_agent_provider_and_key's "custom" branch.
    "Custom (enter provider/model manually)": {"custom": True},
}


def register_openrouter_model(slug: str) -> str:
    """Register an OpenRouter model slug (e.g. "deepseek/deepseek-v4-flash",
    with or without a leading "openrouter/") as a PROVIDERS entry and return
    its label. Registered like a "Custom: ..." entry -- a plain model string
    plus env var -- so _call_model and the restart hand-off
    (PULSE_AUTO_CUSTOM_MODEL) need no OpenRouter-specific handling."""
    slug = slug.strip()
    if slug.lower().startswith("openrouter/"):
        slug = slug.split("/", 1)[1]
    label = f"OpenRouter: {slug}"
    PROVIDERS[label] = {"model": f"openrouter/{slug}", "env_key": "OPENROUTER_API_KEY"}
    return label


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
        self._stage = None
        if _ui.enabled():
            self._stage = _ui.Stage(self.label)
            self._stage.__enter__()
            return self
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if getattr(self, "_stage", None) is not None:
            self._stage.__exit__(*exc)
            self._stage = None
            return
        self._stop_evt.set()
        if self._thread:
            self._thread.join()


def _async_model_calls_enabled() -> bool:
    """Background model calls (check-ins, start-of-run priming) are on unless PULSE_ASYNC_MODEL_CALLS=0."""
    return os.environ.get("PULSE_ASYNC_MODEL_CALLS", "1").strip().lower() not in ("0", "off", "false", "no")


class _EmptyModelResponse(Exception):
    """A model call that came back with no text. Retried; AgentRequestFailed once retries run out."""


class _BackgroundModelCall:
    """One model call running on a worker thread, so the training thread never waits for it.

    Only the call itself runs here. Whatever the answer changes -- tracked variables,
    sensitivity, the console -- is applied later by the training thread, in update(), by the
    same code that used to apply it inline. The worker touches no Pulse state.
    """

    def __init__(self, fn, **tags):
        self.done = False
        self.result = None
        self.error = None
        for key, value in tags.items():
            setattr(self, key, value)
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True, name="pulse-model-call")
        self._thread.start()

    def _run(self, fn) -> None:
        try:
            self.result = fn()
        except BaseException as exc:          # re-raised on the training thread when applied
            self.error = exc
        finally:
            self.done = True


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
# Each call prints to the terminal as soon as it's ready, same spirit as the
# old fixed pipeline. Every pass gets the same generous _AGENT_MAX_TOKENS
# budget (see its comment) -- the prompts, not the cap, keep answers short.
_PASS1_LOCATE = (
    "PASS 1 -- LOCATE: Read through everything you were given (stats, code, history) and identify "
    "the specific region(s) where the problem likely originates -- file/line numbers, variable "
    "names, or code sections. If that isn't enough to narrow it down, investigate before guessing: "
    "GREP:/VIEW: to see more code, or TERMINAL: to check real state directly (git log/diff on a "
    "suspect file, grep for other call sites, a quick check of a config value) -- put the directive "
    "on its own line and stop; you'll get the result back and can keep looking before answering. "
    "Aim for the smallest region that could plausibly contain the root cause (often a single line or "
    "a few adjacent lines), not a whole function or file, unless the evidence genuinely doesn't "
    "narrow further than that. Once you're not just guessing, respond with ONLY a short bullet list "
    "of the suspect location(s). No diagnosis, no fix yet."
)
_PASS2_ANALYZE_TMPL = (
    "Suspect region(s) from your first read:\n{regions}\n\n"
    "PASS 2 -- ANALYZE: Take a focused second look at just those regions. If the numbers/code you "
    "have don't settle the root cause, check before diagnosing -- TERMINAL: to inspect real state "
    "(recent git diff on the suspect lines, a grep for every other place a suspect variable is set, "
    "a quick python -c check on a value you're unsure of), or GREP:/VIEW: for more code -- rather "
    "than reasoning from a guess. Put directives on their own lines and stop; you'll get the results "
    "and can diagnose once you've actually checked. Once you have, give the Diagnosis (one sentence, "
    "the specific root cause) and the Reasoning behind it (grounded in the actual numbers/code/tool "
    "results you have, with real math, referencing line numbers). Do not implement the fix yet."
)
_PASS3_FIX_TEXT_TMPL = (
    "Your analysis so far:\n{diagnosis}\n\n"
    "PASS 3 -- DEVELOP: Give the Fix: a concrete, concise change (not generic advice), in 1-3 "
    "sentences. Describe the smallest change that fixes the root cause -- a changed value, argument, "
    "or line -- not a broader rewrite."
)
_PASS3_IMPLEMENT_TMPL = (
    "Your analysis so far:\n{diagnosis}\n\n"
    "PASS 3 -- DEVELOP & IMPLEMENT: You are fixing a bug in someone's training run, nothing else. "
    "Fix the bug and only the bug: no optimisation, no refactoring, no renaming, no added "
    "callbacks or seeds, no style or formatting changes, no 'while I am here' improvements -- "
    "even where you can see something you would write differently. Every line you touch beyond "
    "the bug is a line that can break a run that is otherwise working. If you notice a second, "
    "unrelated problem while you're in here, do not fix it in this change -- mention it in the "
    "explanation field as something worth a separate look. "
    "Implement the fix for the root cause diagnosed above. The user wants this fix applied to their code. Default to the "
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
    "cause, or does it also rewrite/restructure/reformat code that didn't need to change -- every "
    "line in the diff should trace directly to the diagnosed root cause, with nothing swept in "
    "alongside it. If you can quickly confirm any of this for real instead of just reasoning about "
    "it -- TERMINAL: python -m py_compile on the changed file, GREP: for other call sites that would "
    "need the same change -- do that first; you'll get the result back before you have to answer. "
    'Once you have, respond with ONLY a JSON object of the form {{"passes": true or false, "reason": '
    '"one sentence"}}. passes=true only if the fix is logically/numerically correct, actually '
    "addresses the diagnosed root cause, AND is no larger than necessary to do so."
)
_PASS4_REVISE_TMPL = (
    "Your analysis:\n{diagnosis}\n\n"
    "The fix you proposed:\n{fix_desc}\n\n"
    "Your proposed fix did not pass verification: {reason}\n\n"
    "Revise it -- if the issue was scope (too large a change), narrow it down to the smallest edit "
    "that still fixes the root cause. Respond with ONLY the corrected code-fix JSON object (old/new/"
    "explanation) -- no prose, no markdown fences."
)
_PASS6_CONFIRM_TMPL = (
    "The fix below was applied and the program was then re-run from the start.\n\n"
    "The fix:\n{fix_desc}\n\n"
    "What the problem was:\n{problem}\n\n"
    "What the re-run printed (its last {limit} characters):\n{output}\n\n"
    "PASS 6 -- CONFIRM: answer one question and nothing else: did that fix do its job? "
    "Judge it from the output above. Do not look for other problems, do not propose "
    "changes, do not re-diagnose -- if something is still wrong, saying so here is enough "
    "and the full pipeline will run next.\n"
    'Respond with ONLY {{"resolved": true or false, "reason": "one sentence"}}.'
)
_CONFIRM_OUTPUT_CHARS = 4000

_PASS5_SWEEP = (
    "PASS 5 -- FULL RE-READ: Re-read the ENTIRE code/context again -- not just the region you just "
    "fixed -- and check for any OTHER, unrelated bugs or issues. Respond with ONLY a JSON object of "
    'the form {"other_errors_found": true or false, "summary": "short description, or empty string '
    'if none"}.'
)
_IMPLEMENT_KEYWORDS = ("fix", "edit", "patch", "change the code", "apply", "implement")
_MAX_VERIFY_ATTEMPTS = 3

# How many times the fix pass may ask to see more code before giving up.
_MAX_FIX_TOOL_ROUNDS = 3
# Same idea, one turn earlier: how many times PASS 1/2 may investigate (GREP/VIEW/TERMINAL/etc.)
# before they have to commit to a suspect region / diagnosis instead of looking forever.
_MAX_LOCATE_TOOL_ROUNDS = 3
_MAX_ANALYZE_TOOL_ROUNDS = 3
# How many times PASS 4 may check a directive (e.g. TERMINAL: py_compile) before it has to
# actually answer the passes/fails verdict.
_MAX_VERIFY_TOOL_ROUNDS = 3
_PASS3_NO_TOOLS_NOTE = (
    "You did not return the code-fix JSON. Everything you were given is above. If you need to "
    "see more code, ask for it with a directive line -- e.g. 'VIEW: <file>:<start>-<end>' or "
    "'GREP: <pattern>' -- and it will be answered. Otherwise respond with ONLY the code-fix "
    "JSON object (old/new/files/explanation), fixing the bug and nothing else."
)
_PASS4_RECHECK_TMPL = (
    "Your analysis:\n{diagnosis}\n\n"
    "The fix you proposed:\n{fix_desc}\n\n"
    "An automated check on this fix reported:\n{reason}\n\n"
    "PASS 4b -- RE-EXAMINE: That check is itself automated and can be wrong; it is one more "
    "piece of evidence, not a verdict, and it is not evidence that your fix is right either. "
    "Look again at the code and the evidence you were given and decide for yourself.\n"
    "Respond with ONLY one of:\n"
    '- {{"decision": "keep", "reason": "one sentence"}} if the fix should be applied as it is\n'
    '- {{"decision": "revise", "old": [...], "new": [...], "files": [...], "explanation": "..."}} '
    "with a corrected fix\n"
    '- {{"decision": "drop", "reason": "one sentence"}} if it should not be applied at all\n'
    "Fix only the bug; a revision must stay as small as the bug requires."
)

# Output-token budget for every agent call. Deliberately generous: reasoning
# models spend part of it on hidden reasoning before any visible text, and a
# tight per-pass cap (the old 200/300/700) left them returning an empty reply
# with finish_reason="length". The prompts themselves ask for short answers,
# so non-reasoning models don't get longer. Clamped per model in _call_model.
_AGENT_MAX_TOKENS = 32000
_AGENT_TIMEOUT_SECONDS = 600.0

# Restarted runs launched by _restart_process get this set: their crashes are
# reported back to the parent process (which feeds the output to the agent
# and owns the bounded retry loop) instead of starting a nested fix/restart
# chain of their own.
_RESTART_CHILD_ENV = "PULSE_RESTART_CHILD"
# How deep fix -> restart -> fix chains may nest. A restarted run that is not
# crashing (a training-quality fault, say) can still auto-intervene and
# restart again on its own, so without a cap the chain nests indefinitely.
_RESTART_DEPTH_ENV = "PULSE_RESTART_DEPTH"
_MAX_RESTART_DEPTH = 3
# Set by `pulse run` (cli.py). A script run that way has no auto_track() in the file on
# disk -- `pulse run` adds it in memory -- so re-running the file directly would restart it
# with nothing watching. When set, this is called as hook(python_exe, script_path,
# script_args) and returns the argv to restart with (`pulse run` again). None: the
# original behaviour, `python script.py`, for scripts that call auto_track() themselves.
_RESTART_ARGV_HOOK = None
_RESTART_FEEDBACK_MAX_CHARS = 20000


def _clamp_output_tokens(model: str, requested: int) -> int:
    """Cap `requested` at the model's known max output (e.g. deepseek-chat
    allows 8192); unknown models get the request unchanged."""
    try:
        limit = litellm.get_model_info(model).get("max_output_tokens")
    except Exception:
        return requested
    return min(requested, int(limit)) if limit else requested

_COMMENT_EXT_MAP = {
    ".js": "//", ".ts": "//", ".jsx": "//", ".tsx": "//", ".java": "//",
    ".c": "//", ".cpp": "//", ".h": "//", ".hpp": "//", ".cs": "//",
    ".go": "//", ".rs": "//", ".swift": "//", ".kt": "//",
}


def _comment_char_for(path: str) -> str:
    return _COMMENT_EXT_MAP.get(os.path.splitext(path)[1].lower(), "#")


def _normalize_ws_for_match(s: str) -> str:
    """Collapse each line's leading/trailing whitespace and drop blank
    lines -- see the matching, more heavily-commented copy in pulse.py.
    A lenient equality check used only to LOCATE a snippet that doesn't
    match verbatim, never to decide what gets written."""
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _find_fuzzy_snippet_span(content: str, old: str):
    """Whitespace-tolerant fallback snippet locator -- see the matching
    copy in pulse.py. Returns (start, end) char offsets of the single
    unambiguous match, or None."""
    target = _normalize_ws_for_match(old)
    if not target:
        return None

    content_lines = content.splitlines(keepends=True)
    meaningful = [(i, re.sub(r"\s+", " ", ln.strip()))
                  for i, ln in enumerate(content_lines) if ln.strip()]
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


def _banner_wrap_fix(old: str, new: str, path: str) -> str:
    """Wrap a code-fix replacement so the OLD code stays visible, commented
    out, directly above the NEW (live) code -- instead of silently
    swapping one for the other with no trace in the file itself. Written
    directly into the file content saved to disk, so it shows up the next
    time the file is opened, not just in Pulse's own console output.
    """
    return new


def _format_exc_short(exc_text: str, max_lines: int = 25) -> str:
    """Trim a traceback.format_exc() string to the exception-chain header
    plus the tail (the actual raising frame + message) -- almost always
    what matters, and a full traceback of a deep training loop can be
    huge."""
    lines = (exc_text or "").strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:3] + ["    ... (truncated) ..."] + lines[-max_lines:]
    return "\n".join(lines)


def _describe_exec_value(value: Any) -> str:
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


# ----------------------------------------------------------------------
# Multi-GPU status (see the matching, more heavily-commented copy in
# pulse.py -- the actual per-rank status FILES are written by that
# module's auto_track()/_rank_status_ticker regardless of whether this
# process ends up running in GUI or CLI mode; PulseCLI here only needs to
# read them back for the GPUSTATUS directive).
# ----------------------------------------------------------------------
def _gpu_status_snapshot() -> List[Dict[str, Any]]:
    devices: Dict[int, Dict[str, Any]] = {}
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


def _format_gpu_status(devices: List[Dict[str, Any]], label: Optional[str] = None) -> str:
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


def _rank_status_dir(session_key: str) -> str:
    d = os.path.join(tempfile.gettempdir(), "pulse_cache", "ranks", session_key)
    os.makedirs(d, exist_ok=True)
    return d


def _write_rank_status(session_key: str, rank: int, local_rank: int, world_size: int, extra: Optional[Dict[str, Any]] = None) -> None:
    """Enrich this rank's status file (already being written by pulse.py's
    auto_track()/_rank_status_ticker, which runs regardless of GUI/CLI
    mode) with CLI-only data -- currently, this rank's latest tracked
    scalar values, for RANKDIVERGE. Merges rather than overwrites the GPU
    fields that ticker already wrote, so both writers' data survives."""
    path = os.path.join(_rank_status_dir(session_key), f"rank_{rank}.json")
    existing: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    try:
        import socket
        hostname = socket.gethostname()
    except Exception:
        hostname = existing.get("hostname", "unknown-host")
    payload = dict(existing)
    payload.update({
        "rank": rank, "local_rank": local_rank, "world_size": world_size,
        "pid": os.getpid(), "hostname": hostname, "updated": time.time(),
    })
    if extra:
        payload.update(extra)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass


def _read_all_rank_status(session_key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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


def _format_multi_rank_gpu_status(session_key: str, this_rank: int) -> str:
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
        if s.get("error"):
            lines.append(f"    \u26a0 this rank crashed: {_format_exc_short(s['error'], max_lines=6)}")
    world_size = statuses[0].get("world_size") if statuses else "?"
    return f"GPUSTATUS: {len(statuses)}/{world_size} rank(s) reporting\n" + "\n".join(lines)


def _run_history_path(script_path: Optional[str]) -> str:
    base = os.path.dirname(script_path) if script_path else "."
    return os.path.join(base, ".pulse_run_history.json")


def _load_run_history(script_path: Optional[str]) -> List[Dict[str, Any]]:
    try:
        with open(_run_history_path(script_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _seed_fingerprint() -> Dict[str, Any]:
    """Best-effort, deterministic fingerprint of every RNG this process
    can introspect. NumPy/stdlib `random` don't expose the ORIGINAL seed
    once reseeded, only their large internal generator state -- still
    useful as a "did anything about RNG state change" fingerprint even
    though it's not a human-readable seed number. PyTorch's
    initial_seed() IS the real recoverable seed value, when available."""
    info: Dict[str, Any] = {}
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


def _seed_history_path(script_path: Optional[str]) -> str:
    base = os.path.dirname(script_path) if script_path else "."
    return os.path.join(base, ".pulse_seed_history.json")


# ----------------------------------------------------------------------
# ML anti-pattern static checks (MLLINT) -- see the matching, more
# heavily-commented copy in pulse.py. A small, high-confidence set of
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


def _mllint_resolve(node, env):
    """Resolve `node` to a literal Python value if possible: a direct
    literal, a negative literal (`-0.01` parses as UnaryOp(USub,
    Constant(0.01))), or -- the whole point of `env` -- a plain variable
    reference to a name that a lightweight, conservative constant-
    propagation pass (see `_mllint_literal_env`) could pin down to a
    single literal value elsewhere in the file. Without this, something
    as ordinary as

        DROPOUT_RATE = 1.0
        ...
        model.add(Dropout(DROPOUT_RATE))

    passes every check silently, because the Dropout(...) call site holds
    a bare Name, not the literal -- the bug is real but AST-invisible
    unless the Name is traced back to its assignment."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _mllint_resolve(node.operand, env)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner
        return None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _mllint_const_str(node, env=None):
    val = _mllint_resolve(node, env or {})
    return val if isinstance(val, str) else None


def _mllint_list_of_str(node, env=None):
    if isinstance(node, (ast.List, ast.Tuple)):
        return [s for s in (_mllint_const_str(e, env) for e in node.elts) if s is not None]
    return []


def _mllint_bool_const(node, env=None):
    val = _mllint_resolve(node, env or {})
    return val if isinstance(val, bool) else None


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


def _mllint_literal_env(tree):
    """Tiny, deliberately conservative constant-propagation table: name ->
    literal value, but ONLY for a name assigned to a literal exactly once
    (to the same literal) anywhere in the file. This is not real dataflow
    analysis -- it doesn't know about scope or assignment order -- so any
    name that's reassigned, reassigned to a *different* literal, or ever
    assigned something non-literal, is dropped entirely rather than risk
    resolving a use to a stale or wrong value. That's enough to catch the
    extremely common pattern of hyperparameters pulled out to a named
    constant once near the top of a script."""
    MULTI = object()
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            lit = _mllint_resolve(node.value, {})
            candidate = lit if lit is not None else MULTI
            if name not in seen:
                seen[name] = candidate
            elif seen[name] != candidate:
                seen[name] = MULTI
    return {k: v for k, v in seen.items() if v is not MULTI and v is not None}


def _mllint_call_env(tree):
    """Companion to `_mllint_literal_env` for the non-literal case that
    matters most here: an object built once and reused, e.g.

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        ...
        model.compile(loss=loss_fn, ...)

    Maps name -> the Call node that built it, but only when that name is
    assigned to a Call exactly once anywhere in the file (same
    conservative reasoning as `_mllint_literal_env`: any reassignment,
    even to another Call, drops the name rather than guessing)."""
    MULTI = object()
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            candidate = node.value if isinstance(node.value, ast.Call) else MULTI
            if name not in seen:
                seen[name] = candidate
            elif seen[name] is not candidate:
                seen[name] = MULTI
    return {k: v for k, v in seen.items() if v is not MULTI}


def _mllint_is_zero_initializer(node, env=None):
    """True if `node` (an AST expression, resolved through `env` if it's
    just a variable reference) is recognizably a zero initializer: the
    string 'zeros'/'zero', a bare Zeros()/zeros_initializer() call (e.g.
    `initializers.Zeros()`, `tf.zeros_initializer()`), or a Constant-style
    initializer whose value is literally 0."""
    s = _mllint_const_str(node, env)
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


def _mllint_numeric_const(node, env=None):
    """Return the numeric value of `node` if it's a numeric literal (or a
    variable resolvable to one via `env`), handling unary minus too."""
    val = _mllint_resolve(node, env or {})
    return val if isinstance(val, (int, float)) and not isinstance(val, bool) else None


def _mllint_numeric_kwarg_or_arg(node, kwarg_names, arg_index=0, env=None):
    """Return the numeric value passed either as one of `kwarg_names` or
    as the positional arg at `arg_index`, else None."""
    for kw in node.keywords:
        if kw.arg in kwarg_names:
            val = _mllint_numeric_const(kw.value, env)
            if val is not None:
                return val
    if len(node.args) > arg_index:
        return _mllint_numeric_const(node.args[arg_index], env)
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


def _mllint_scan(trees) -> List[tuple]:
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

        # Conservative variable resolution -- see _mllint_literal_env's
        # docstring. Threaded into every check below that previously only
        # recognized a literal sitting directly at the call site, so e.g.
        # `DROPOUT_RATE = 1.0` ... `Dropout(DROPOUT_RATE)` is caught the
        # same as `Dropout(1.0)` would have been.
        env = _mllint_literal_env(tree)
        call_env = _mllint_call_env(tree)

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
                        loss_val = _mllint_const_str(kw.value, env)
                        # `loss=` may be an already-built loss object,
                        # either inline (`loss=CategoricalCrossentropy(...)`)
                        # or via a variable assigned to one exactly once
                        # earlier in the file (`loss_fn = ...; loss=loss_fn`).
                        loss_call_node = kw.value if isinstance(kw.value, ast.Call) else (
                            call_env.get(kw.value.id) if isinstance(kw.value, ast.Name) else None)
                        if loss_val is None and loss_call_node is not None:
                            loss_call_fname = _mllint_fname(loss_call_node)
                            from_logits = any(
                                k.arg == "from_logits" and _mllint_bool_const(k.value, env) is True
                                for k in loss_call_node.keywords
                            )
                    elif kw.arg == "metrics":
                        metrics_val = _mllint_list_of_str(kw.value, env)
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

            # Collect every string-literal (or single-assignment-variable)
            # `activation=` kwarg anywhere in the file, plus a standalone
            # `Activation('softmax')` layer -- a distinct, common Keras
            # idiom (activation as its own layer rather than a kwarg) that
            # the kwarg-only version of this collection used to miss
            # entirely. The cross-check pass below treats the latest one
            # (by line number) before compile() as "the final layer's
            # activation" -- a coarse but effective proxy for these
            # typically-linear Sequential-style scripts.
            for kw in node.keywords:
                if kw.arg == "activation":
                    act_val = _mllint_const_str(kw.value, env)
                    if act_val:
                        activation_calls.append((node.lineno, act_val))
            if fname == "Activation" and node.args:
                act_val = _mllint_const_str(node.args[0], env)
                if act_val:
                    activation_calls.append((node.lineno, act_val))

            # ---- GENERAL: zero-initialized weight matrix (any backend
            # exposing a `*_initializer=` kwarg -- Keras/TF layers) ----
            for kw in node.keywords:
                if kw.arg in _MLLINT_WEIGHT_INITIALIZER_PARAMS and _mllint_is_zero_initializer(kw.value, env):
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
                rate = _mllint_numeric_kwarg_or_arg(node, {"rate", "p"}, env=env)
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
            # `lr`, silently missing every keras/tf/optax script. Also now
            # resolves through a named constant (`LR = 5.0; Adam(learning_rate=LR)`)
            # via `env`, instead of only recognizing a literal at the call site.
            if fname in _MLLINT_OPTIMIZER_NAMES:
                lr_node = None
                for kw in node.keywords:
                    if kw.arg in ("lr", "learning_rate"):
                        lr_node = kw.value
                if lr_node is None and node.args:
                    lr_node = node.args[-1] if len(node.args) >= 2 else None
                lr_val = _mllint_numeric_const(lr_node, env) if lr_node is not None else None
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
            if not (shuffle_kw and _mllint_bool_const(shuffle_kw.value, env) is False):
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

class PulseCLI:
    def _try_keras_history(var_name: str, watch_locals: Dict[str, Any]) -> Optional[float]:
        """When a tracked variable is None in the outer frame (typically because
        it's set inside a Keras callback from `logs.get(...)` and that metric
        doesn't exist in logs for this task), try to read the latest value from
        the Keras model's own `.history.history` dict, which IS accessible in
        the outer frame via the tracked `model` variable and is always
        up-to-date after each epoch. This handles the extremely common case of
        a user tracking `train_acc`, `val_acc`, `train_mape`, etc. that are
        assigned inside callbacks but whose real values live in model.history.

        Name mapping: strips 'train_' / 'val_' prefix, then tries both with and
        without 'val_' prefix in model.history. E.g.:
            train_acc   -> history['accuracy'][-1]  or history['acc'][-1]
            val_acc     -> history['val_accuracy'][-1]
            train_loss  -> history['loss'][-1]
            val_mape    -> history['val_mean_absolute_percentage_error'][-1] etc.
        """
        # Find any Keras model in scope
        model = None
        for candidate_name in ("model", "clf", "net", "network", "estimator"):
            candidate = watch_locals.get(candidate_name)
            if candidate is not None and hasattr(candidate, "history") and hasattr(candidate.history, "history"):
                model = candidate
                break
        if model is None:
            # Also scan for any object with a .history.history attribute
            for val in watch_locals.values():
                if val is not None and hasattr(val, "history") and hasattr(val.history, "history"):
                    model = val
                    break
        if model is None:
            return None

        hist = model.history.history
        if not hist:
            return None

        name_lower = var_name.lower()
        is_val = name_lower.startswith("val_")
        # Strip known prefixes to get the bare metric name
        bare = name_lower
        for prefix in ("train_", "val_", "tr_", "training_"):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break

        # Build candidate keys to try in model.history.history
        candidates = []
        if is_val:
            candidates += [f"val_{bare}", f"val_{bare.replace('_', '')}"]
            # Common Keras metric name expansions
            expansions = {
                "acc": ["val_accuracy", "val_acc"],
                "accuracy": ["val_accuracy", "val_acc"],
                "mse": ["val_mean_squared_error", "val_mse"],
                "mae": ["val_mean_absolute_error", "val_mae"],
                "mape": ["val_mean_absolute_percentage_error", "val_mape"],
                "rmse": ["val_root_mean_squared_error", "val_rmse"],
                "loss": ["val_loss"],
            }
            candidates += expansions.get(bare, [])
        else:
            candidates += [bare, bare.replace("_", "")]
            expansions = {
                "acc": ["accuracy", "acc"],
                "accuracy": ["accuracy", "acc"],
                "mse": ["mean_squared_error", "mse"],
                "mae": ["mean_absolute_error", "mae"],
                "mape": ["mean_absolute_percentage_error", "mape"],
                "rmse": ["root_mean_squared_error", "rmse"],
                "loss": ["loss"],
            }
            candidates += expansions.get(bare, [])

        for key in candidates:
            values = hist.get(key)
            if values:
                try:
                    v = float(values[-1])
                    if math.isfinite(v):
                        return v
                except (TypeError, ValueError):
                    continue
        return None



    def __init__(
        self,
        watch_locals: Optional[Dict[str, Any]] = None,
        pdf_dir: str = "Pulse_Output",
        discovered: Optional[Dict[str, Optional[tuple]]] = None,
    ):
        self.watch_locals = watch_locals or {}
        # Lazily-initialized {path: text} snapshot for CHANGELOG, and a
        # rolling ring buffer of state_dict checkpoints for REPLAY -- see
        # _run_changelog / _replay_maybe_checkpoint.
        self._changelog_baseline = None
        self._replay_checkpoints: List[tuple] = []
        self._replay_step_counter = itertools.count()
        # id of the most recently recorded .pulse_history commit -- lets
        # _restart_process() know exactly what to roll back to if the
        # fix that triggered a restart keeps crashing (see
        # _auto_rollback_after_failed_restarts).
        self._last_commit_id: Optional[str] = None
        # (rank, local_rank, world_size, session_key) -- set by
        # _start_cli_tracker when running under a multi-GPU/multi-process
        # launch; defaults to "not distributed".
        self.dist_info = (0, 0, 1, None)
        # Per-process id used to identify "this run" in the local
        # run-history file -- see _run_runcompare.
        self._run_id = str(uuid.uuid4())[:8]
        # Running token/cost accounting across every agent call this
        # session -- see _record_usage / _run_cost.
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "cost_usd": 0.0}
        # Decoded cross-session history (see _load_history_context) --
        # defaults empty so PASTFIX works whether or not cloud history
        # ever successfully loaded (no team, offline, fetch failed, ...).
        self._history_sessions: List[Dict[str, Any]] = []
        # Background retry queue for transient failures (agent rate
        # limits/timeouts, restart-launch hiccups) -- see
        # _queue_agent_retry/_queue_restart_retry and _ensure_retry_ticker.
        # Training/the run itself is never blocked on any of this; a
        # pending retry just gets picked back up automatically later.
        self._last_call_failed_transiently = False
        self._pending_agent_retry: Optional[Dict[str, Any]] = None
        self._pending_restart_retry: Optional[Dict[str, Any]] = None
        self._retry_ticker_started = False
        self._retry_ticker_lock = threading.RLock()
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
        # Set by _restart_process while it's feeding a failed restart's
        # crash output back to the agent for another fix pass -- prevents
        # that nested call from triggering its own recursive
        # _restart_process() (see _ask_agent_impl's trigger at the end);
        # the retry loop already in flight relaunches the re-patched
        # script directly instead.
        self._suppress_auto_restart: bool = False

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
        # schedule: every checkin_interval seconds after the last one,
        # Pulse proactively asks the agent whether it still wants each
        # GPU-tracked variable (or wants to add one) -- see
        # _maybe_periodic_checkin. That same check-in also asks the
        # agent to actually look at the current run and flag anything
        # that looks off (a STATUS: line, escalated the same way as
        # _check_for_trouble's own detections -- see
        # _escalate_training_problem), and lets the agent set its own
        # next interval via a NEXTCHECK: reply, so the cadence is
        # dynamic rather than a fixed 15 minutes: a run that looks fine
        # can go longer between check-ins, a borderline one can ask to
        # be checked again sooner. The very first interval is itself
        # negotiable the same way, via NEXTCHECK: in the one-time
        # _START_PRIME_PROMPT reply (see _prime_at_start) -- 900s below
        # is only the fallback if the agent doesn't set one.
        self.gpu_tracked_vars: set[str] = set()
        self._last_gpu_probe: float = 0.0
        self.gpu_probe_interval: float = 600.0  # 10 minutes
        self._last_checkin: float = time.monotonic()
        # Model calls made off the training thread (see _BackgroundModelCall).
        self._checkin_call: Optional[_BackgroundModelCall] = None
        # What the agent that scheduled the next check-in wants it to look at (CHECKNOTE:).
        self._checkin_note: str = ""
        self._start_prime_call: Optional[_BackgroundModelCall] = None
        self._start_prime_answered: bool = False
        self._start_prime_retried: bool = False
        self.checkin_interval: float = 900.0  # 15 minutes, first one 15 min after start unless negotiated sooner

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
        self._start_primed_with_agent: bool = False  # MLLINT+fix re-runs once when agent first becomes available
        self._last_intervention_signature: Optional[str] = None
        # /code is on by default -- every manually-asked question includes
        # the training code (and any cross-file context) unless turned off.
        self.include_code_default: bool = True

        # Interactive Mode & Interrupt Handling
        self.continuous = False
        # Separate interrupt latch. Do not use `continuous` itself as the
        # interrupt state: other code (notably auto-intervention and /c) can
        # legitimately change `continuous` while a SIGINT is being delivered.
        # The latch guarantees Ctrl+C survives until the next safe boundary.
        self._stop_requested = False
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
        self._last_apply_skipped: List[tuple] = []
        # (path, line) of the most recent crash -- see _occurrence_at_crash.
        self._last_crash_location: Optional[tuple] = None
        # What the agent was last asked about, quoted back by the post-fix check.
        self._last_problem_description: Optional[str] = None
        # Look for unrelated bugs after fixing the reported one? Off by
        # default -- see the pass 5 call site.
        self.sweep_enabled: bool = os.environ.get("PULSE_SWEEP", "").strip().lower() in ("1", "true", "yes", "on")
        # Dedup bookkeeping so one recurring bug doesn't get treated as N
        # separate incidents within a single run.
        self._traceback_signatures_seen: Dict[str, int] = {}
        self._resolved_signatures: set = set()   # a fix was applied for this signature this run
        self._declined_signatures: set = set()   # user said no to help for this signature this run


    def _sigint_handler(self, sig, frame):
        """Turn Ctrl+C into a request to pause at the next Pulse boundary.

        The signal handler itself must stay tiny. In particular, it must not
        call input(), touch queues, or raise KeyboardInterrupt on the first
        press: the user's training code may be between Python instructions
        (for example inside a long GPU/framework call). We latch the request
        and let update() consume it at a safe point.
        """
        if not self._stop_requested:
            self._stop_requested = True
            self.continuous = False
            cprint("\n[Pulse] Ctrl+C received. Pausing after the current training step...")
            return

        # A second Ctrl+C while already paused keeps the old hard-exit
        # behavior, which is useful if the user really wants to terminate.
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
        self.sweep_enabled = self._config_bool("sweep", "sweep_for_other_bugs", default=self.sweep_enabled)
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
            if _trackable_without_reading(val):
                trackable[name] = val

        for name in self.discovered:
            # A name Pulse has seen holding a non-trackable value (a list,
            # a model object, ...) did run -- don't report it as unassigned.
            if name not in trackable and name not in self.watch_locals:
                trackable[name] = None

        return trackable

    def capture_crash_state(self, user_frames) -> None:
        """Refresh watch_locals from the user-code frames of an uncaught
        exception (outermost first -- see pulse.py's crash hook). The
        periodic sampler only snapshots every throttle_interval, so a
        script that crashes quickly (or between windows) would otherwise
        show the agent stale or empty state -- e.g. every variable as "not
        run yet" when it was in fact assigned. Inner frames' names win.
        Also makes REPL/DRYRUN evaluate against the real crash-time objects."""
        for frame in user_frames:
            self.watch_locals.update(
                {k: v for k, v in frame.f_locals.items() if not k.startswith("__")}
            )
    def _cmd_gputrack(self, var_name: str, quiet: bool = False) -> Optional[str]:
        """Flag a variable as GPU-resident and worth a closer look -- via
        the manual /gputrack command, or via the agent's GPUTRACK:
        directive (quiet=True). GPU stat probes force a device-to-host
        sync, which is real overhead on a training loop, so this only
        ever folds the variable into the slow gpu_probe_interval cadence
        (never the fast per-step one) and the agent gets asked every
        checkin_interval whether it's still needed -- see
        _maybe_periodic_checkin. Returns the variable name on success, else
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
                if _pulse_is_accelerator_value(val):
                    # Decided from the shape alone: an accelerator value is never handed to
                    # the backend to describe.
                    elements = 1
                    for dim in getattr(val, "shape", ()) or ():
                        elements *= int(dim)
                    if getattr(val, "shape", None) is not None and elements == 1:
                        return "track"
                elif describe_tensor(val).kind == "scalar":
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
        if not variables and not getattr(self, "_code_mode", False):
            # Still set up the agent: crash diagnosis/fixing doesn't need
            # any tracked variables (see auto_track's no-variables branch).
            cprint("[Pulse CLI] No trackable variables found in scope -- crash diagnosis stays on.")

        var_list = sorted(variables.keys(), key=lambda n: (not _looks_like_loss(n), n))

        self.auto_mode = True
        self.tracked_vars = list(var_list)
        self.var_states = {name: self._default_state_for(name, variables[name]) for name in var_list}
        self.generate_pdfs = False  # opt-in only, via /track + a future setting -- no prompt by default

        self._cloud_setup()
        self._agent_setup()
        self._print_ready_summary()
        self._print_run_header()

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
        if _ui.enabled():
            self._print_ready_summary_ui()
            return
        code_mode = getattr(self, "_code_mode", False)
        cprint("\n" + "─" * 60)
        cprint("  Pulse Code is ready" if code_mode else "  Pulse is ready")
        if self.user_id:
            cprint(f"  Signed in as {self.email}" + (f"  ·  workspace {self.team_join_code}" if self.team_join_code else "  ·  no workspace"))
        else:
            cprint("  Running locally -- not signed in to Pulse Cloud (/cloud for details)")
        cprint(f"  Agent: {self.agent_provider or 'not configured (/agent to set one up)'}")
        if code_mode:
            cprint("  Describe what you want built. Type /help any time for commands.")
            cprint("─" * 60 + "\n")
            return
        cprint(f"  Auto-fix: {'ON' if self.auto_intervene else 'OFF'}  ·  Sensitivity: {self.sensitivity:.2f}  ·  Tracking {len(self.tracked_vars)} variable{'s' if len(self.tracked_vars) != 1 else ''}")
        if self.tracked_vars:
            cprint(f"  Tracked: {', '.join(self.tracked_vars[:6])}" + (f"  (+{len(self.tracked_vars) - 6} more, see /vars)" if len(self.tracked_vars) > 6 else ""))
        cprint("  Type /help any time for commands, or just ask a question about your run.")
        cprint("─" * 60 + "\n")

    def _print_ready_summary_ui(self) -> None:
        """PULSE / READY: the same facts as the plain summary, from the same attributes."""
        sha = getattr(self, "_last_synced_commit_sha", None)
        known_sha = bool(sha and sha != "unknown")
        rows = [
            ("Account", (self.email or "signed in") if self.user_id else "local -- not signed in (/cloud)", bool(self.user_id)),
            ("Workspace", (f"join code {self.team_join_code}" if self.team_join_code else "selected") if self.team_id else "none", bool(self.team_id)),
            ("Code version", sha[:10] if known_sha else "unknown", known_sha),
        ]
        if self.agent_provider:
            rows.append(("Agent model", self.agent_provider, True))
            local = bool(PROVIDERS.get(self.agent_provider, {}).get("local"))
            rows.append(("API key", "not needed -- local model" if local else "set for this session", True))
        else:
            rows.append(("Agent model", "not configured (/agent to set one up)", False))
        if getattr(self, "_code_mode", False):
            _ui.ready_block(rows, closing="Pulse Code is ready.")
            _ui.note("Describe what you want built. Type /help any time for commands.")
            return
        _ui.ready_block(rows, closing="Pulse is ready.")
        _ui.note(
            f"Auto-fix {'ON' if self.auto_intervene else 'OFF'}  ·  Sensitivity {self.sensitivity:.2f}  ·  "
            f"Tracking {len(self.tracked_vars)} variable{'s' if len(self.tracked_vars) != 1 else ''}"
        )
        _ui.note("Type /help any time for commands, or just ask a question about your run.")

    def _print_run_header(self) -> None:
        """PULSE / RUN: what is running, from data Pulse already has (the environment record it
        collected at sign-in, and what the script has imported). Static -- no live figures, so
        it never competes with the script's own output. Nothing is shown that is not known."""
        if not _ui.enabled() or self._resumed_from_restart() or getattr(self, "_code_mode", False):
            return
        import platform
        env = next((e for e in getattr(self, "_telemetry", []) if isinstance(e, dict) and e.get("python_version")), {})
        _ui.header("Run")
        if self.script_path:
            _ui.kv("Script", os.path.basename(self.script_path), indent=0)
        _ui.kv("PID", os.getpid(), indent=0)
        _ui.kv("Python", env.get("python_version") or platform.python_version(), indent=0)
        frameworks = env.get("framework_versions")
        if not frameworks:
            found = []
            for module_name, pretty in (("torch", "PyTorch"), ("tensorflow", "TensorFlow"), ("jax", "JAX")):
                module = sys.modules.get(module_name)
                version = getattr(module, "__version__", None) if module is not None else None
                if version:
                    found.append(f"{pretty} {version}")
            frameworks = ", ".join(found)
        if frameworks:
            _ui.kv("Framework", frameworks, indent=0)
        if env.get("gpu_name"):
            count = env.get("gpu_count")
            _ui.kv("GPU", (f"{count}x " if count and count != 1 else "") + str(env["gpu_name"]), indent=0)
        if env.get("cuda_version"):
            _ui.kv("CUDA", env["cuda_version"], indent=0)
        if self.tracked_vars:
            _ui.kv("Tracking", f"{len(self.tracked_vars)} variable{'s' if len(self.tracked_vars) != 1 else ''}", indent=0)
        _ui._emit()
        _ui.note("Pulse stays quiet while training runs. Ctrl+C pauses after the current step.")
        _ui._emit(_ui.rule())
        _ui._emit()

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
            if _ui.enabled() and "commit" in label:
                _ui_screen("Code version", "Which version of your code are you running?", f"{context}.")
            val = _prompt_text(
                f"[Pulse] {context}. Enter {label} manually (Enter to skip){suffix} > ",
                label=f"Enter {label} manually",
                placeholder=(f"Enter to use {default}" if default else "Enter to skip"),
                validate=((lambda t: f"{_ui._g('ok')} Valid commit format" if _ui.commit_looks_valid(t) else "expects 7-40 hex characters")
                          if "commit" in label else None),
            ).strip()
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
                _ui_code_version(sha, self._repo_cwd)

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
                if not _ui.enabled():      # on a terminal this is shown in PULSE / RUN instead
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
            resp = _ui_account_menu()
            if resp is None:
                cprint("\n--- Pulse Cloud Authentication ---")
                resp = input(
                    "[Pulse] (s)ign up [default] / (l)og in / (r)ecover account > "
                ).strip().lower() or "s"

            if resp in ("s", "signup", "sign up"):
                _flush_stdin()
                _ui_step("Sign up")
                # Retry only the field that actually failed (email or
                # password) instead of bouncing back to the top-level
                # "sign up / log in / recover" menu on every typo -- that
                # used to mean a single password mismatch cost you a full
                # re-type of the email address too.
                while True:
                    email = _prompt_text("email (e.g. name@example.com, min. 8 characters) > ", label="Email", placeholder="name@example.com").strip()
                    if not email:
                        cprint("[Pulse] email cannot be blank.", color=_RED)
                        continue
                    if not cloud.is_valid_email(email):
                        cprint("[Pulse] enter a valid email address (min. 8 characters), e.g. name@example.com.", color=_RED)
                        continue
                    _ui_done("Email", email)
                    break
                while True:
                    password = _prompt_text("Password (min. 8 characters) > ", label="Password", secret=True, placeholder="min. 8 characters", validate=_password_progress)
                    if len(password) < 8:
                        cprint("[Pulse] Password must be at least 8 characters.", color=_RED)
                        continue
                    confirm_password = _prompt_text("Confirm password > ", label="Confirm password", secret=True)
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
                        resp2 = _prompt_text(
                            "Press Enter once confirmed to continue (or type 'skip' to do this later) > ",
                            label="Press Enter once you've confirmed  (or type 'skip' to do this later)",
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
                _ui_step("Log in")
                email = _prompt_text("email > ", label="Email", placeholder="name@example.com").strip()
                if not email:
                    cprint("[Pulse] email cannot be blank.", color=_RED)
                    continue
                _ui_done("Email", email)
                password = _prompt_text("Password > ", label="Password", secret=True)
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
                _ui_step("Recover your account")
                email = _prompt_text("email > ", label="Email", placeholder="name@example.com").strip()
                recovery_code = _prompt_text("Recovery code (e.g. 7F3K-9QRT-2LXP) > ", label="Recovery code", placeholder="e.g. 7F3K-9QRT-2LXP").strip()
                if not email or not recovery_code:
                    cprint("[Pulse] Both email and recovery code are required.", color=_RED)
                    continue
                new_password = _prompt_text("New password (min. 8 characters) > ", label="New password", secret=True, placeholder="min. 8 characters", validate=_password_progress)
                if len(new_password) < 8:
                    cprint("[Pulse] Password must be at least 8 characters.", color=_RED)
                    continue
                confirm_new_password = _prompt_text("Confirm new password > ", label="Confirm new password", secret=True)
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
        _ui_screen("Terms")
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
        resp = _prompt_text("Type 'yes' to accept (or 'full' to read the complete license first) > ", label="Type 'yes' to accept, or 'full' to read the complete license first").strip().lower()
        if resp == "full":
            print(f"\n{PULSE_LICENSE_TEXT}")
            _flush_stdin()
            resp = _prompt_text("Type 'yes' to accept and create your account > ", label="Type 'yes' to accept and create your account").strip().lower()
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
        """Human-readable label for a workspace menu entry -- prefers an
        explicit name (see cloud.update_team_name) if the team has one,
        then falls back to the repo basename, then a generic label built
        from the join code, plus an admin tag.
        """
        custom_name = (team.get("name") or "").strip()
        repo = team.get("repo")
        if custom_name:
            label = custom_name
        elif repo and repo != "unknown":
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
        repo_input = _prompt_text(prompt, label="GitHub repository", placeholder=(f"Enter to use detected: {detected}" if detected else "optional -- Enter to skip")).strip()
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
                new_name = str(workspace_options.get("new_name") or "").strip() if workspace_options else ""
                try:
                    team = cloud.create_team(self.user_id, repo=repo or None, cwd=self._repo_cwd, name=new_name or None)
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
            existing = cloud.find_teams_for_user(self.user_id)  # re-fetch so rename/delete/leave are reflected immediately
            resp = _ui_workspace_menu(existing, cached_team_id, lambda t: self._describe_workspace(t, self.user_id))
            if resp is None:
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
                if existing:
                    print("  r) Rename a Workspace")
                    print("  l) Leave a Workspace")
                    print("  d) Delete a Workspace (admins only)")

                suffix = f" (Enter = {default_choice})" if default_choice else ""
                resp = input(f"[Pulse] Select a workspace{suffix} > ").strip().lower()
                if not resp and default_choice:
                    resp = default_choice

            if resp.isdigit() and existing and 1 <= int(resp) <= len(existing):
                team = existing[int(resp) - 1]
                self.team_id = team["team_id"]
                self.team_join_code = team.get("join_code")
                self.team_admin_ids = list(team.get("admin_ids") or [])
                _say(f"[Pulse] Using workspace (join code: {self.team_join_code}).",
                     "Workspace", self._describe_workspace(team, self.user_id).replace("  --  ", "  ·  "))
                if not team.get("repo") or team.get("repo") == "unknown":
                    self._prompt_and_set_repo()
                cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
                return

            if resp in ("c", "create"):
                _flush_stdin()
                _ui_screen("New workspace", "Name your workspace")
                name_input = _prompt_text("Workspace name (optional, Enter to skip) > ", label="Workspace name", placeholder="optional -- Enter to skip").strip()
                detected = cloud.git_remote_url(self._repo_cwd)
                _flush_stdin()
                prompt = (
                    f"GitHub repo URL (Enter to use detected: {detected}) > "
                    if detected else
                    "GitHub repo URL (optional, Enter to skip) > "
                )
                repo_input = _prompt_text(prompt, label="GitHub repository", placeholder=(f"Enter to use detected: {detected}" if detected else "optional -- Enter to skip")).strip()
                try:
                    team = cloud.create_team(self.user_id, repo=repo_input or detected, name=name_input or None)
                    self.team_id = team["team_id"]
                    self.team_join_code = team["join_code"]
                    self.team_admin_ids = list(team.get("admin_ids") or [])
                    label = (team.get("name") or "").strip() or "(no name set)"
                    cprint(f"[Pulse] ✓ Workspace '{label}' created. Share this join code with teammates: {self.team_join_code}")
                    if name_input and not team.get("name"):
                        cprint("[Pulse]   ⚠ Name couldn't be saved (this deployment's Teams table doesn't have a 'name' column yet) -- the workspace still works, just unnamed.", color=_YELLOW)
                    cprint(f"[Pulse]   Repo: {team.get('repo')}")
                    cloud.save_cached_credentials(self.user_id, self.email, self.team_id)
                    return
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not create workspace: {exc}", color=_RED)

            elif resp in ("j", "join"):
                _flush_stdin()
                _ui_screen("Join workspace", "Enter the join code")
                code = _prompt_text("Join code > ", label="Join code").strip().upper()
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

            elif resp in ("r", "rename") and existing:
                _flush_stdin()
                idx = self._pick_workspace_index(existing, "Rename which workspace")
                if idx is None:
                    continue
                target = existing[idx]
                _flush_stdin()
                new_name = input("New workspace name > ").strip()
                if not new_name:
                    cprint("[Pulse] Name cannot be blank.", color=_RED)
                    continue
                try:
                    saved = cloud.update_team_name(target["team_id"], new_name)
                    if saved:
                        cprint(f"[Pulse] ✓ Renamed to '{new_name[:80]}'.")
                    else:
                        cprint("[Pulse] ⚠ This deployment's Teams table doesn't have a 'name' column yet -- couldn't save the name.", color=_YELLOW)
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not rename workspace: {exc}", color=_RED)

            elif resp in ("l", "leave") and existing:
                _flush_stdin()
                idx = self._pick_workspace_index(existing, "Leave which workspace")
                if idx is None:
                    continue
                target = existing[idx]
                confirm = input(f"Leave '{self._describe_workspace(target, self.user_id)}'? (y/n) > ").strip().lower()
                if confirm not in ("y", "yes"):
                    continue
                try:
                    cloud.leave_team(target["team_id"], self.user_id)
                    cprint("[Pulse] ✓ Left the workspace.")
                    if self.team_id == target["team_id"]:
                        self.team_id = None
                        self.team_join_code = None
                        self.team_admin_ids = []
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not leave workspace: {exc}", color=_RED)

            elif resp in ("d", "delete") and existing:
                _flush_stdin()
                idx = self._pick_workspace_index(existing, "Delete which workspace")
                if idx is None:
                    continue
                target = existing[idx]
                if not cloud.is_team_admin(target, self.user_id):
                    cprint("[Pulse] ⚠ Only an admin of that workspace can delete it.", color=_RED)
                    continue
                member_count = len(target.get("members") or [])
                warn = "" if member_count <= 1 else f" -- this removes all {member_count} member(s) from it, not just you"
                confirm = input(f"Permanently delete '{self._describe_workspace(target, self.user_id)}'{warn}? Type DELETE to confirm > ").strip()
                if confirm != "DELETE":
                    cprint("[Pulse] Cancelled -- deletion requires typing DELETE exactly.")
                    continue
                try:
                    cloud.delete_team(target["team_id"], self.user_id)
                    cprint("[Pulse] ✓ Workspace deleted.")
                    if self.team_id == target["team_id"]:
                        self.team_id = None
                        self.team_join_code = None
                        self.team_admin_ids = []
                except cloud.SupabaseError as exc:
                    cprint(f"[Pulse] ⚠ Could not delete workspace: {exc}", color=_RED)

            else:
                cprint("[Pulse] Invalid option.")

    def _pick_workspace_index(self, existing: List[Dict[str, Any]], prompt_label: str) -> Optional[int]:
        """Shared sub-picker for the rename/leave/delete menu actions --
        lists the same workspaces again so the number always matches what
        the user just saw, and returns the chosen 0-based index or None if
        they cancelled/entered something invalid."""
        for i, team in enumerate(existing, start=1):
            print(f"  {i}) {self._describe_workspace(team, self.user_id)}")
        _flush_stdin()
        resp = input(f"{prompt_label} (number, or Enter to cancel) > ").strip()
        if not resp:
            return None
        if resp.isdigit() and 1 <= int(resp) <= len(existing):
            return int(resp) - 1
        cprint("[Pulse] Invalid selection.", color=_RED)
        return None

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
                ("/chart [var]", "ASCII loss/metric curve for a tracked scalar (defaults to the main loss)"),
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

        # Retained (not just used transiently above) so PASTFIX can search
        # every teammate's applied fixes across every session on this
        # team, not just this local machine's own fix-log -- workspace-
        # shared fix knowledge, for free, off infrastructure that already
        # existed for the crash-recovery "known fixes" index above.
        self._history_sessions = sessions

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

    _TB_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')

    def _last_user_frame(self, tb_text: str):
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
        frame_info = self._last_user_frame(tb_text)
        if frame_info:
            cprint(f"[Pulse] ⚠ Error at {os.path.basename(frame_info[0])}, line {frame_info[1]}", color=_RED)
            # Remembered so _apply_code_fix can disambiguate a snippet that
            # occurs more than once (duplicated cell/block) by preferring the
            # occurrence at the crash.
            try:
                self._last_crash_location = (os.path.abspath(frame_info[0]), int(frame_info[1]))
            except (TypeError, ValueError):
                self._last_crash_location = None
        self._log_incident(
            "crash", tb_text.strip().splitlines()[-1] if tb_text.strip() else "(empty traceback)",
            signature=sig, occurrence=self._traceback_signatures_seen[sig],
            file=frame_info[0] if frame_info else None,
            line=int(frame_info[1]) if frame_info else None,
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
        optionally PULSE_LOCAL_API_BASE); for any OpenRouter model, set
        PULSE_PROVIDER to "openrouter/<model slug>" (e.g.
        "openrouter/deepseek/deepseek-v4-flash") plus OPENROUTER_API_KEY.
        If what's needed is missing,
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
            # A raw "provider/model" string that doesn't match any named
            # entry is treated as a custom model directly -- mirrors the
            # interactive "Custom" branch below, just without the prompts.
            if want.lower().startswith("openrouter/") and not any(n.lower() == want.lower() for n in names):
                # OpenRouter always reads OPENROUTER_API_KEY, so there's no
                # env var to ask for -- and keeping the key in os.environ
                # under that name is what carries it across a fix restart.
                candidates = [register_openrouter_model(want)]
                names.append(candidates[0])
            elif want and "/" in want and not any(n.lower() == want.lower() for n in names):
                env_var = self._config_text("api_key_env", "env_key") or os.environ.get("PULSE_API_KEY_ENV", "").strip() or None
                key = self._config_text("api_key", "key") or (os.environ.get(env_var, "").strip() if env_var else "")
                label = f"Custom: {want}"
                PROVIDERS[label] = {"model": want, "env_key": env_var}
                self.agent_provider = label
                self.agent_key = key or "local"
                self.agent_model_string = None
                self.agent_api_base = None
                self.agent_history = []
                if env_var and key:
                    os.environ[env_var] = key
                cloud.save_cached_profile(agent_provider=label, agent_env_key=env_var)
                cprint(f"[Pulse] Non-interactive mode -- custom agent set to {want}.")
                return True
            elif want.isdigit() and 1 <= int(want) <= len(names):
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
                if info.get("custom"):
                    continue  # needs a model string -- see the raw "provider/model" branch above
                if info.get("openrouter"):
                    slug = self._config_text("model") or os.environ.get("PULSE_OPENROUTER_MODEL", "").strip()
                    if not slug:
                        continue  # can't use "any model" without knowing which one
                    match = register_openrouter_model(slug)
                    info = PROVIDERS[match]
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

        cached_provider = cloud.load_cached_profile().get("agent_provider") if not self.agent_provider else None
        ui_pick = _ui_pick_agent(names, cached_provider, self.agent_provider)
        if ui_pick is None:
            print("\nSelect an agent/provider:")
            for i, name in enumerate(names, 1):
                local_tag = "  [local -- no data leaves this machine]" if PROVIDERS[name].get("local") else ""
                if PROVIDERS[name].get("custom"):
                    local_tag = "  [type any provider/model string]"
                if PROVIDERS[name].get("openrouter"):
                    local_tag = "  [type any OpenRouter model]"
                marker = "  (current)" if name == self.agent_provider else ("  (last used)" if name == cached_provider else "")
                print(f"  {i}) {name}{local_tag}{marker}")

        prompt = (
            "\nAgent number (or Enter to skip) > "
            if initial
            else "\nAgent number (or Enter to cancel) > "
        )

        while True:
            if ui_pick is not None:
                raw = ui_pick          # already resolved by the selector: an exact name, or "" = skip
            else:
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
            if raw in names:           # an exact provider name (what the selector returns)
                chosen = raw
                break
            matches = [n for n in names if raw.lower() in n.lower()]
            if len(matches) == 1:
                chosen = matches[0]
                break
            cprint("[Pulse CLI] Pick a valid agent number or provider name.")

        info = PROVIDERS[chosen]

        if info.get("custom"):
            _flush_stdin()
            _ui_screen("Agent model", "Custom model")
            model_string = _prompt_text("Model string (e.g. openai/gpt-6-astra, ollama_chat/llama3.1) > ", label="Model string", placeholder="e.g. openai/gpt-6-astra, ollama_chat/llama3.1").strip()
            if not model_string:
                cprint("[Pulse CLI] No model string entered. Agent unchanged.")
                if initial:
                    self.agent_provider = None
                return False
            env_var = _prompt_text("Env var name for the API key (optional, Enter to skip) > ", label="Env var for the API key", placeholder="optional -- Enter to skip").strip() or None
            key = "local"
            if env_var:
                existing = os.environ.get(env_var, "").strip()
                if existing and _prompt_text(f"An {env_var} is already set. Use it? (Y/n) > ", label=f"{env_var} is already set. Use it?  (Y/n)").strip().lower() in ("", "y", "yes"):
                    key = existing
                else:
                    key = _prompt_text("API key > ", label="API key", secret=True).strip()
                if not key:
                    cprint("[Pulse CLI] No API key entered. Agent unchanged.")
                    if initial:
                        self.agent_provider = None
                    return False
                os.environ[env_var] = key
            label = f"Custom: {model_string}"
            PROVIDERS[label] = {"model": model_string, "env_key": env_var}
            self.agent_provider = label
            self.agent_key = key
            self.agent_model_string = None
            self.agent_api_base = None
            self.agent_history = []
            cloud.save_cached_profile(agent_provider=label, agent_env_key=env_var)
            _say_plain("Agent model", f"Custom: {model_string}", f"✓ Custom agent set: {model_string}")
            return True

        if info.get("local"):
            _say_plain("Agent model", f"{chosen}  ·  runs on your own infrastructure, no API key needed",
                       f"\n✓ Agent selected: {chosen} -- runs on your own infrastructure, no API key needed.")
            _flush_stdin()
            cached_profile = cloud.load_cached_profile()
            default_base = cached_profile.get("agent_api_base") or info["default_api_base"]
            api_base = _prompt_text(f"Server URL [{default_base}] > ", label="Server URL", placeholder=default_base).strip() or default_base
            default_model = cached_profile.get("agent_model_name") or ""
            model_prompt = f"Model name ({info['model_hint']})"
            model_prompt += f" [{default_model}] > " if default_model else " > "
            model_name = _prompt_text(model_prompt, label=f"Model name ({info['model_hint']})", placeholder=default_model or None).strip() or default_model
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

        if info.get("openrouter"):
            _flush_stdin()
            _ui_screen("Agent model", "OpenRouter model")
            slug = _prompt_text(f"Model ({info['model_hint']}) > ", label=f"Model ({info['model_hint']})").strip()
            if not slug:
                cprint("[Pulse CLI] No model entered. Agent unchanged.")
                if initial:
                    self.agent_provider = None
                return False
            # From here on it's an ordinary cloud provider: same key prompt,
            # same OPENROUTER_API_KEY, just under its registered label.
            chosen = register_openrouter_model(slug)
            info = PROVIDERS[chosen]

        env_var = info["env_key"]
        existing = os.environ.get(env_var, "").strip()

        _say_plain("Agent model", chosen, f"\n✓ Agent selected: {chosen}")
        _ui_key_screen(chosen)
        if existing:
            _flush_stdin()
            use_existing = _prompt_text(
                f"An {env_var} is already set. Use it? (Y/n) > ",
                label=f"{env_var} is already set. Use it?  (Y/n)",
            ).strip().lower()
            if use_existing in ("", "y", "yes"):
                key = existing
            else:
                key = _prompt_text("API key > ", label="API key", secret=True, validate=lambda t: _ui.key_hint_for(t, env_var)).strip()
        else:
            key = _prompt_text("API key > ", label="API key", secret=True, validate=lambda t: _ui.key_hint_for(t, env_var)).strip()

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
            # "accepted" here has always meant Pulse took the key, not that the provider
            # verified it -- on the terminal UI say what actually happened.
            _say_plain("API key", "set for this session", f"✓ API key accepted for {self.agent_provider}.")
        else:
            print(f"✓ Switched to {self.agent_provider}. Conversation history reset for the new agent.")

        # If there were queued auto-interventions or unfired MLLINT findings
        # from before the agent was configured, fire them now rather than
        # waiting for the next update() call (which could be an epoch away).
        self._prime_with_agent_if_needed()
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
        auto-filled here instead of prompting the user all over again. A
        dynamically-registered "Custom: ..." provider only ever lived in
        the PROVIDERS dict of the process that created it, so its model
        string/env var are carried separately via PULSE_AUTO_CUSTOM_MODEL/
        PULSE_AUTO_CUSTOM_ENV_KEY and re-registered here before lookup.
        """
        auto_provider = os.environ.pop("PULSE_AUTO_PROVIDER", None)
        auto_api_base = os.environ.pop("PULSE_AUTO_API_BASE", None)
        auto_model = os.environ.pop("PULSE_AUTO_MODEL", None)
        auto_custom_model = os.environ.pop("PULSE_AUTO_CUSTOM_MODEL", None)
        auto_custom_env_key = os.environ.pop("PULSE_AUTO_CUSTOM_ENV_KEY", None)
        if auto_provider and auto_provider not in PROVIDERS and auto_custom_model:
            PROVIDERS[auto_provider] = {"model": auto_custom_model, "env_key": auto_custom_env_key or None}
        if auto_provider and auto_provider in PROVIDERS:
            info = PROVIDERS[auto_provider]
            if info.get("local"):
                if auto_api_base and auto_model:
                    self._set_local_agent(auto_provider, auto_api_base, auto_model)
                    cprint(f"[Pulse] Resumed with agent {auto_provider} (auto-filled after restart).")
            else:
                env_var = info.get("env_key")
                key = os.environ.get(env_var, "").strip() if env_var else "local"
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
        # Captured now (before the retry loop below can apply further
        # fixes of its own) -- this is "the fix that triggered this
        # restart chain", i.e. what _auto_rollback_after_failed_restarts
        # rolls back to if every retry in this chain still crashes.
        chain_start_commit_id = self._last_commit_id
        _agent_log_event("RESTARTING the training script to run the fixed code")

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

        depth = 0
        try:
            depth = int(os.environ.get(_RESTART_DEPTH_ENV, "0"))
        except ValueError:
            depth = 0
        if depth >= _MAX_RESTART_DEPTH:
            cprint(
                f"[Pulse] ⚠ Already {depth} restarts deep -- not restarting again. The fix is saved "
                "to disk; run the script yourself to pick it up.",
                color=_YELLOW,
            )
            self._log_incident("restart_skipped", f"restart depth cap ({_MAX_RESTART_DEPTH}) reached")
            return

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
            # A dynamically-registered "Custom: ..." provider only exists in
            # THIS process's PROVIDERS dict -- carry its model string/env var
            # separately so _agent_setup can re-register it in the fresh
            # process before looking it up (see that method's docstring).
            # Harmless no-op for a normal named cloud provider, which already
            # has a static entry the fresh process can look up on its own.
            provider_info = PROVIDERS.get(self.agent_provider, {})
            if "model" in provider_info and not provider_info.get("local"):
                os.environ["PULSE_AUTO_CUSTOM_MODEL"] = provider_info["model"]
                if provider_info.get("env_key"):
                    os.environ["PULSE_AUTO_CUSTOM_ENV_KEY"] = provider_info["env_key"]

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
        elif _stdout_is_tty():
            # Only on a real terminal. Into a pipe or a log the reset is a raw ESC-c in the
            # output, and `stty` on a non-terminal stdin just prints an ioctl error.
            sys.stdout.write("\033c\033[0m")
            sys.stdout.flush()
            for clear_cmd in (["stty", "sane"], ["clear"]):
                if clear_cmd[0] == "stty" and not sys.stdin.isatty():
                    continue
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
        if _RESTART_ARGV_HOOK is not None:
            argv = _RESTART_ARGV_HOOK(python_exe, script_path, sys.argv[1:]) or argv
        # Only the child gets the marker (not this process's os.environ):
        # if every retry fails and this process keeps running the old code,
        # its own later crashes must still go through the agent as normal.
        child_env = self._restart_env(depth)

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
                # capture_output=True so a failure's stdout/stderr can be
                # fed back to the agent below -- printed after the fact
                # either way, so nothing is hidden, just no longer live.
                result = subprocess.run(argv, capture_output=True, text=True, env=child_env)
            except Exception as exc:
                cprint(f"[Pulse] ⚠ Restart attempt {attempt}/{MAX_RESTART_ATTEMPTS} failed to launch ({exc}).", color=_RED)
                if attempt >= MAX_RESTART_ATTEMPTS:
                    cprint(
                        f"[Pulse] ⚠ Giving up after {MAX_RESTART_ATTEMPTS} restart attempts -- "
                        "continuing current run with the old in-memory code.",
                        color=_RED,
                    )
                    self._log_incident("restart_failed", f"Could not launch replacement process after {attempt} attempts: {exc}")
                    self._queue_restart_retry()
                    return
                time.sleep(min(RETRY_BACKOFF_SECONDS * attempt, 30))
                continue

            if result.returncode == 0:
                resolved, reason = self._confirm_fix_did_its_job(result)
                if resolved:
                    sys.exit(0)
                sys.stdout.write(result.stdout or "")
                sys.stderr.write(result.stderr or "")
                message = (f"The re-run finished, but the fix does not appear to have done its job "
                           f"(attempt {attempt}/{MAX_RESTART_ATTEMPTS}): {reason}")
            else:
                sys.stdout.write(result.stdout or "")
                sys.stderr.write(result.stderr or "")
                message = f"Replacement training process exited with code {result.returncode} (attempt {attempt}/{MAX_RESTART_ATTEMPTS})"

            if attempt >= MAX_RESTART_ATTEMPTS:
                cprint(
                    f"[Pulse] ⚠ {message}. Giving up after {MAX_RESTART_ATTEMPTS} attempts -- "
                    "continuing current run with the old in-memory code.",
                    color=_RED,
                )
                self._log_incident("restart_failed", message)
                # "Automatic rollback offered rather than requiring the
                # model to reason its way to 'maybe I should revert'" --
                # every retry in this chain still crashed, so restore the
                # workspace to right before the fix that started it,
                # using the same .pulse_history mechanism /revert uses.
                # Nothing is destroyed: the failed chain stays fully
                # recoverable afterward via /log + /revert.
                self._auto_rollback_after_failed_restarts(chain_start_commit_id)
                return

            # A nonzero exit almost always means the agent's own fix didn't
            # fully take, or introduced a new bug -- hand the failure
            # straight back to it to patch the CURRENT (already-fixed-once)
            # code, instead of blindly relaunching the exact same broken
            # code or reverting and starting the diagnosis over from
            # scratch. _suppress_auto_restart keeps this nested call from
            # triggering its own recursive _restart_process() -- if it
            # applies a new fix, this loop's own next iteration relaunches
            # the (now re-patched) script_path directly.
            if self.agent_provider and self.agent_key:
                cprint(f"[Pulse] ⚠ {message}. Feeding the failure back to the agent to fix the CURRENT code (not from scratch)...", color=_YELLOW)
                try:
                    with open(script_path, "r", encoding="utf-8") as f:
                        self.code_text = f.read()
                except OSError:
                    pass
                def _tail(text: str) -> str:
                    # Progress bars and training logs can run to megabytes;
                    # the traceback that matters is at the end.
                    text = text or "(empty)"
                    if len(text) <= _RESTART_FEEDBACK_MAX_CHARS:
                        return text
                    return "[... earlier output truncated ...]\n" + text[-_RESTART_FEEDBACK_MAX_CHARS:]

                failure_question = (
                    f"After the fix you just applied, the restarted training process crashed "
                    f"with exit code {result.returncode}. Its output:\n\n"
                    f"STDOUT:\n{_tail(result.stdout)}\n\nSTDERR:\n{_tail(result.stderr)}\n\n"
                    "This is the same bug context as before -- fix the CURRENT code shown below "
                    "directly. Do not start the diagnosis over from scratch, and do not reintroduce "
                    "whatever change just failed."
                )
                self._suppress_auto_restart = True
                try:
                    self._ask_agent_impl(failure_question, include_code=True, _depth=0)
                finally:
                    self._suppress_auto_restart = False
            else:
                cprint(f"[Pulse] ⚠ {message}. Retrying restart...", color=_YELLOW)
                time.sleep(min(RETRY_BACKOFF_SECONDS * attempt, 30))

    def _confirm_fix_did_its_job(self, result) -> tuple:
        """One small question after a re-run that finished cleanly: did the fix
        work? Returns (resolved, reason).

        A re-run used to be taken as success on its exit code alone, while
        anything further was a fresh full investigation of an
        already-fixed program -- locate, diagnose, write, verify, all over
        again. This is the cheap middle step: one call that only judges the
        outcome. If it says no, the caller escalates to the full pipeline
        with the re-run's own output as evidence.

        Anything that goes wrong here (no agent, a failed request, an
        unparsable answer) counts as resolved: a check that cannot answer
        must not hold up a run that finished cleanly.
        """
        if not self.agent_provider or not self.agent_key:
            return True, ""
        fix = self._last_applied_fix
        fix_desc = self._describe_fix(fix) if fix else "(no fix recorded)"
        problem = (self._last_problem_description or "(not recorded)")[:1500]
        output = ((result.stdout or "") + "\n" + (result.stderr or ""))[-_CONFIRM_OUTPUT_CHARS:]
        try:
            with _Spinner("Checking the fix did its job"):
                answer = self._call_model(
                    _PASS6_CONFIRM_TMPL.format(fix_desc=fix_desc, problem=problem,
                                               limit=_CONFIRM_OUTPUT_CHARS, output=output),
                    max_tokens=_AGENT_MAX_TOKENS,
                )
        except AgentRequestFailed as exc:
            cprint(f"[Pulse] ⚠ Post-fix check skipped (agent request failed: {exc})", color=_YELLOW)
            return True, ""
        verdict = self._parse_json_obj(answer)
        if not verdict or "resolved" not in verdict:
            return True, ""
        reason = str(verdict.get("reason", "")).strip() or "(no reason given)"
        if bool(verdict["resolved"]):
            cprint(f"[Pulse] ✓ Re-run checked: the fix did its job -- {reason}", color=_YELLOW)
            return True, reason
        cprint(f"[Pulse] ⚠ Re-run checked: the fix did NOT do its job -- {reason}", color=_RED)
        return False, reason

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
        """Give every file (entry script + extra project files) a label that
        names it unambiguously: its path relative to the root the files share,
        the same scheme `pulse code` already uses.

        Labelling by basename looked tidier and was wrong on any project that
        has two files with the same name -- and frameworks are full of them
        (optim/base.py and config/base.py, one registry.py per package). The
        first file seen took the bare name, every later one got parent/name,
        and _resolve_fix_path's basename fallback then sent every request for
        one of the later ones to the first: the agent asked to VIEW
        optim/base.py and was shown config/base.py under a header reading
        "base.py", and a fix targeting optim/base.py was applied against
        config/base.py.
        """
        paths = [p for p in [self.script_path] + list(self.extra_files) if p]
        absolute = {p: os.path.abspath(p) for p in paths}
        try:
            root = os.path.commonpath(list(absolute.values())) if absolute else ""
        except ValueError:                      # different drives (Windows)
            root = ""
        if root in absolute.values():           # a single file: its own directory is the root
            root = os.path.dirname(root)

        label_for_path: Dict[str, str] = {}
        path_for_label: Dict[str, str] = {}
        for path in paths:
            if path in label_for_path:
                continue
            if root:
                label = os.path.relpath(absolute[path], root).replace(os.sep, "/")
            else:
                label = os.path.basename(path)
            if label in path_for_label:         # the same file under two spellings
                continue
            label_for_path[path] = label
            path_for_label[label] = path

        self._label_for_path = label_for_path
        self._path_for_label = path_for_label

    def _occurrence_at_crash(self, content: str, snippet: str, path: str) -> Optional[int]:
        """Offset of the occurrence of `snippet` containing the crash line in
        `path`, when exactly one occurrence does. Else None (stay safe)."""
        location = getattr(self, "_last_crash_location", None)
        if not location or os.path.abspath(path) != location[0]:
            return None
        crash_line = location[1]
        hits = []
        start = content.find(snippet)
        while start != -1:
            first_line = content.count("\n", 0, start) + 1
            last_line = first_line + snippet.count("\n")
            if first_line <= crash_line <= last_line:
                hits.append(start)
            start = content.find(snippet, start + 1)
        return hits[0] if len(hits) == 1 else None

    def _resolve_fix_path(self, file_label: Optional[str]) -> Optional[str]:
        """Map a fix entry's optional "file" label (or a VIEW's file) back to a
        real path on disk, defaulting to the entry script when unset.

        The agent does not always reproduce a header exactly -- it may write a
        longer path than the label ("src/optim/base.py" for "optim/base.py") or
        a shorter one -- so a near miss still resolves, but only while exactly
        one known file matches. An ambiguous name resolves to nothing rather
        than to whichever file happened to be seen first: showing or editing
        the wrong file is worse than saying the label could not be resolved.
        """
        if not file_label or not file_label.strip():
            return self.script_path
        label = file_label.strip().replace("\\", "/").lstrip("./").lower()
        if file_label.strip() in self._path_for_label:
            return self._path_for_label[file_label.strip()]
        # Already a real path: _apply_code_fix records resolved paths (not
        # labels) in `skipped`, and _request_corrected_snippets resolves
        # those again -- without this, every re-quote retry silently found
        # no file to show the model and gave up.
        if os.path.isfile(file_label.strip()):
            return os.path.abspath(file_label.strip())
        # One path is a tail of the other: "a/b/model.py" for label "b/model.py".
        matches = [p for lbl, p in self._path_for_label.items()
                   if lbl.lower() == label or lbl.lower().endswith("/" + label)
                   or label.endswith("/" + lbl.lower())]
        if len(matches) != 1:
            basename = os.path.basename(label)
            matches = [p for lbl, p in self._path_for_label.items()
                       if os.path.basename(lbl).lower() == basename]
        return matches[0] if len(matches) == 1 else None

    def _build_agent_context(self, include_code: bool = False) -> str:
        variables = self.discover_variables()
        lines = ["Current Pulse variable state:"]
        for name in sorted(variables):
            val = self._cpu_observation(name, variables[name])
            if val is None and variables[name] is not None:
                continue
            if val is None:
                lines.append(f"- {name}: no value observed (not assigned yet, currently None, or not sampled yet)")
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

    def _call_model(self, instruction: str, max_tokens: int = _AGENT_MAX_TOKENS, *,
                    system: Optional[str] = None,
                    history: Optional[List[Dict[str, str]]] = None,
                    purpose: Optional[str] = None) -> str:
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

        `system`/`history` replace the default system prompt and the shared
        conversation -- the periodic check-in runs its own conversation on a
        worker thread and must neither read nor grow agent_history.
        """
        messages = (
            [{"role": "system", "content": system or getattr(self, "_system_prompt_override", None) or SYSTEM_PROMPT}]
            + (self.agent_history[-10:] if history is None else list(history))
            + [{"role": "user", "content": instruction}]
        )
        model = self.agent_model_string or PROVIDERS[self.agent_provider]["model"]
        max_tokens = _clamp_output_tokens(model, max_tokens)
        max_attempts = 3
        last_exc: Optional[Exception] = None
        label = purpose or (instruction.strip().splitlines() or ["(empty)"])[0][:90]
        call_no = _agent_log_call(label, messages, model)
        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=_AGENT_TIMEOUT_SECONDS,
                    api_base=self.agent_api_base,  # only set for local/self-hosted providers
                    api_key=(self.agent_key if self.agent_key and self.agent_key != "local" else None),
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    # Seen with reasoning models: the whole answer went into reasoning and none
                    # came back. It is billed like any other call, and the same request usually
                    # succeeds on a second try, so it is retried like a transient error.
                    self._record_usage(response)
                    raise _EmptyModelResponse("the provider returned an empty response")
                self._record_usage(response)
                _agent_log_reply(call_no, content, time.monotonic() - started, getattr(response, "usage", None))
                return content
            except AgentRequestFailed:
                raise
            except Exception as exc:
                last_exc = exc
                _agent_log_reply(call_no, None, time.monotonic() - started,
                                 error=f"attempt {attempt}/{max_attempts}: {self._classify_model_error(exc)}")
                if attempt < max_attempts and (isinstance(exc, _EmptyModelResponse)
                                               or self._is_retryable_model_error(exc)):
                    backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    cprint(
                        f"[Pulse] ⚠ Agent request hit a transient error (attempt {attempt}/{max_attempts}), "
                        f"retrying in {backoff}s: {self._classify_model_error(exc)}",
                        color=_RED,
                    )
                    time.sleep(backoff)
                    continue
                _agent_log_event("AGENT REQUEST FAILED", f"{label}: {self._classify_model_error(exc)}")
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
    _GREP_RE = re.compile(r"^\s*GREP:\s*(.+)$", re.MULTILINE)
    _VIEW_RE = re.compile(r"^\s*VIEW:\s*(.+)$", re.MULTILINE)

    @classmethod
    def _extract_directives(cls, text: str):
        """Pull CALC:/PROMOTE:/GPUTRACK:/GPUUNTRACK:/SENSITIVITY:/
        NORMAL_START:/GREP:/VIEW: lines out of an agent response, returning
        (cleaned_text, calc_exprs, promote_names, gputrack_names,
        gpuuntrack_names, sensitivity_args, normal_start_args,
        grep_patterns, view_requests). Cleaned text has those lines
        stripped so they don't clutter what's printed/stored. A bare
        'none' value (as instructed for the periodic GPU check-in reply
        format) is dropped rather than treated as a variable name.
        sensitivity_args/normal_start_args/grep_patterns/view_requests are
        lists of raw argument strings -- applied via _apply_directives.
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
        grep_patterns = [m.strip() for m in cls._GREP_RE.findall(text) if m.strip()]
        view_requests = [m.strip() for m in cls._VIEW_RE.findall(text) if m.strip()]

        cleaned = cls._CALC_RE.sub("", text)
        cleaned = cls._PROMOTE_RE.sub("", cleaned)
        cleaned = cls._GPUTRACK_RE.sub("", cleaned)
        cleaned = cls._GPUUNTRACK_RE.sub("", cleaned)
        cleaned = cls._SENSITIVITY_RE.sub("", cleaned)
        cleaned = cls._NORMAL_START_RE.sub("", cleaned)
        cleaned = cls._GREP_RE.sub("", cleaned)
        cleaned = cls._VIEW_RE.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return (
            cleaned, calc_exprs, promote_names, gputrack_names, gpuuntrack_names,
            sensitivity_args, normal_start_args, grep_patterns, view_requests,
        )

    # Hard caps on GREP/VIEW output so a broad pattern or a huge range
    # request can't blow up the agent's context the same way dumping every
    # file in full would -- the whole point of these tools is to let the
    # agent pull a *narrow* slice instead of everything every turn.
    _GREP_MAX_MATCHES = 40
    _GREP_CONTEXT_LINES = 1
    _VIEW_MAX_LINES = 400

    def _iter_searchable_files(self):
        """(label, path, text) for every file the agent can GREP/VIEW --
        the entry script plus any local project files Pulse already knows
        about (see pulse.py's auto_track -> _discover_project_files).
        Reuses the same label scheme as code-fix "files" targeting, so a
        GREP/VIEW result's file label matches what a later fix's "file"
        field should say.
        """
        self._build_file_labels()
        if self.script_path and self.code_text is not None:
            label = self._label_for_path.get(self.script_path, os.path.basename(self.script_path))
            yield label, self.script_path, self.code_text
        for path, text in self.extra_files.items():
            label = self._label_for_path.get(path, os.path.basename(path))
            yield label, path, text

    def _run_grep(self, pattern: str) -> str:
        """Case-insensitive regex search across every known project file,
        returning matches as 'label:line: text' with a line of context on
        each side, capped at _GREP_MAX_MATCHES total. Falls back to a
        literal substring search if `pattern` isn't valid regex, so a
        plain word/phrase still works without the agent needing to escape
        anything.
        """
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
                snippet = "\n".join(
                    f"    {n + 1:>4} | {lines[n]}" for n in range(lo, hi)
                )
                blocks.append(f"  {label}, line {i + 1}:\n{snippet}")
                total += 1

        if not blocks:
            return f"GREP '{pattern}': no matches in any tracked file."
        truncated_note = " (truncated -- narrow the pattern for more)" if total >= self._GREP_MAX_MATCHES else ""
        return f"GREP '{pattern}': {total} match(es){truncated_note}\n" + "\n".join(blocks)

    _VIEW_ARG_RE = re.compile(r"^(?:([^:]+):)?\s*(\d+)\s*-\s*(\d+)\s*$")

    def _run_view(self, arg: str) -> str:
        """Exact line range from a known project file, in the same
        numbered format used for full-file context (`{n:>4} | {line}`), so
        the agent can zoom into one region instead of needing the whole
        file re-sent. `arg` is '<file>:<start>-<end>' (file label optional
        -- omitting it defaults to the main script), e.g. 'model.py:40-75'
        or '120-160'. Range is capped at _VIEW_MAX_LINES.
        """
        m = self._VIEW_ARG_RE.match(arg.strip())
        if not m:
            return f"VIEW '{arg}': could not parse -- expected '<file>:<start>-<end>' or '<start>-<end>'."
        file_label, start_str, end_str = m.groups()
        start, end = int(start_str), int(end_str)
        if end < start:
            start, end = end, start

        # Build/refresh the label<->path map before resolving -- VIEW can
        # be sent on its own, without a preceding GREP or 'Send Code' this
        # turn, so _path_for_label may otherwise still be empty or stale.
        self._build_file_labels()
        path = self._resolve_fix_path(file_label) if file_label else self.script_path
        if not path:
            return f"VIEW '{arg}': could not resolve file '{file_label}'."

        text = self.code_text if path == self.script_path else self.extra_files.get(path)
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
    # Extended toolset: execution, code-intelligence, statistics, and
    # grounding directives. A separate dict-based extractor/applier so
    # none of the CALC/PROMOTE/GPUTRACK/... positional-tuple call sites
    # above need to change shape.
    # ------------------------------------------------------------------
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
        "terminal": re.compile(r"^\s*TERMINAL:\s*(.+)$", re.MULTILINE),
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
    _FLAG_STYLE_DIRECTIVES = {"depgraph", "changelog", "shapetrace", "gpustatus", "mllint", "layerstats", "hardexamples", "ampstatus", "seedcheck", "runcompare", "cost"}

    @classmethod
    def _extract_new_directives(cls, text: str):
        """Pull the extended toolset's directive lines out of an agent
        response. Returns (cleaned_text, requests) where requests is
        {directive_name: [arg, ...]} for every directive that appeared --
        a flag-style directive with no argument (DEPGRAPH/CHANGELOG/bare
        SHAPETRACE) still shows up as [''] so callers can just check
        truthiness/`in`."""
        requests: Dict[str, List[str]] = {}
        cleaned = text
        for key, rx in cls._NEW_DIRECTIVE_RES.items():
            raw = [m.strip() for m in rx.findall(text)]
            if key in cls._FLAG_STYLE_DIRECTIVES:
                if raw:
                    requests[key] = raw
            else:
                raw = [m for m in raw if m]
                if raw:
                    requests[key] = raw
            cleaned = rx.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, requests

    def _iter_ast_trees(self):
        """(label, path, text, tree) for every known project file that
        parses as valid Python."""
        for label, path, text in self._iter_searchable_files():
            try:
                tree = ast.parse(text, filename=path)
            except (SyntaxError, ValueError):
                continue
            yield label, path, text, tree

    def _run_defof(self, symbol: str) -> str:
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

    def _run_callers(self, symbol: str) -> str:
        """CALLERS: <symbol> -- every call site of a function/class."""
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

    def _run_depgraph(self) -> str:
        """DEPGRAPH: -- the import graph between tracked local files."""
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

    def _scalar_history(self, name: str):
        """Resolve `name` against self.scalar_histories (exact, else a
        unique case-insensitive substring match) and return
        (resolved_name, [value, ...] or None). Unlike the GUI's manifest,
        this is a flat value list in logging order (deduplicated on
        change), not (step, value) pairs -- DIFFSTATS indices below count
        into that list."""
        hist = self.scalar_histories.get(name)
        if hist is None:
            matches = [k for k in self.scalar_histories if name.lower() in k.lower()]
            if len(matches) == 1:
                name, hist = matches[0], self.scalar_histories[matches[0]]
        return name, (list(hist) if hist else None)

    def _run_corr(self, arg: str) -> str:
        """CORR: <var1> <var2> -- real correlation coefficient between two
        tracked scalar histories."""
        parts = arg.split()
        if len(parts) != 2:
            return f"CORR '{arg}': expected two variable names, e.g. 'CORR: loss grad_norm'."
        name1, v1 = self._scalar_history(parts[0])
        name2, v2 = self._scalar_history(parts[1])
        if not v1 or not v2:
            missing = parts[0] if not v1 else parts[1]
            return f"CORR '{arg}': no numeric history for '{missing}' -- only scalar-tracked variables have one."
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

    def _run_outlier(self, var: str) -> str:
        """OUTLIER: <var> -- deterministic z-score anomaly detection."""
        name, values = self._scalar_history(var)
        if not values:
            return f"OUTLIER '{var}': no numeric history available."
        if len(values) < 4:
            return f"OUTLIER '{name}': not enough data points yet ({len(values)})."
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        if std == 0:
            return f"OUTLIER '{name}': series is constant -- no outliers."
        flagged = [(i, v, (v - mean) / std) for i, v in enumerate(values) if abs((v - mean) / std) > 3]
        if not flagged:
            return f"OUTLIER '{name}': no points with |z| > 3 over {len(values)} points (mean={mean:.4g}, std={std:.4g})."
        lines = "\n".join(f"  history idx {i}: {v:.6g} (z={z:.2f})" for i, v, z in flagged[-20:])
        return f"OUTLIER '{name}': {len(flagged)} outlier point(s) (mean={mean:.4g}, std={std:.4g})\n{lines}"

    _DIFFSTATS_ARG_RE = re.compile(r"^\s*(\S+)\s+(-?\d+)\s+(-?\d+)\s*$")

    def _run_diffstats(self, arg: str) -> str:
        """DIFFSTATS: <var> <index_a> <index_b> -- exact delta between two
        recorded points (indices count into that variable's own history,
        negative counts from the end)."""
        m = self._DIFFSTATS_ARG_RE.match(arg.strip())
        if not m:
            return f"DIFFSTATS '{arg}': expected '<var> <index_a> <index_b>', e.g. 'DIFFSTATS: loss 10 40'."
        var, a_str, b_str = m.groups()
        name, values = self._scalar_history(var)
        if not values:
            return f"DIFFSTATS '{var}': no numeric history available."
        try:
            va, vb = values[int(a_str)], values[int(b_str)]
        except IndexError:
            return f"DIFFSTATS '{name}': index out of range (history has {len(values)} point(s))."
        delta = vb - va
        pct = f" ({delta / va * 100:+.2f}%)" if va else ""
        return f"DIFFSTATS '{name}' [{a_str}] -> [{b_str}]: {va:.6g} -> {vb:.6g}, delta = {delta:+.6g}{pct}"

    def _run_histogram(self, var: str, buckets: int = 10) -> str:
        """HISTOGRAM: <var> -- actual bucketed distribution counts."""
        name, values = self._scalar_history(var)
        if not values:
            return f"HISTOGRAM '{var}': no numeric history available."
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

    def _run_doclookup(self, arg: str) -> str:
        """DOCLOOKUP: <library>.<symbol> -- real signature/docstring for an
        already-installed library function."""
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

    def _run_changelog(self) -> str:
        """CHANGELOG: -- diff of what's changed in tracked files since the
        last checkpoint. Baseline starts at this session's code and
        advances to 'now' every call, so a second CHANGELOG only shows
        what's changed since the first."""
        self._build_file_labels()
        current = dict(self.extra_files)
        if self.script_path and self.code_text is not None:
            current[self.script_path] = self.code_text

        baseline = getattr(self, "_changelog_baseline", None)
        if baseline is None:
            self._changelog_baseline = current
            return "CHANGELOG: no prior checkpoint yet -- this turn's code is now the baseline for future CHANGELOG calls."

        diffs = []
        for path, new_text in current.items():
            old_text = baseline.get(path, "")
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

    def _fixlog_path(self) -> str:
        base = os.path.dirname(self.script_path) if self.script_path else "."
        return os.path.join(base, ".pulse_fixlog.json")

    def _load_fixlog(self) -> List[Dict[str, Any]]:
        try:
            with open(self._fixlog_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def _append_fixlog(self, path: str, old: str, new: str, explanation: str) -> None:
        """Persist every applied code fix to a project-level fix-log so
        PASTFIX can search it later."""
        log = self._load_fixlog()
        log.append({"time": time.time(), "path": path, "old": old, "new": new, "explanation": explanation})
        log = log[-200:]
        try:
            with open(self._fixlog_path(), "w", encoding="utf-8") as f:
                json.dump(log, f)
        except OSError:
            pass

    def _run_pastfix(self, query: str) -> str:
        """PASTFIX: <symbol_or_region> -- search this project's own local
        fix-log AND, when signed in with team history loaded (see
        _load_history_context), every fix any teammate has applied across
        every session on this repo/team, for prior fixes touching the
        same function/region. Local and shared hits are labeled
        separately since a shared hit came from someone else's machine/
        session, not necessarily this one's current code state."""
        query_l = query.strip().lower()

        local_hits = []
        for entry in reversed(self._load_fixlog()):
            haystack = " ".join([
                str(entry.get("explanation", "")), str(entry.get("old", "")),
                str(entry.get("new", "")), str(entry.get("path", "")),
            ]).lower()
            if query_l in haystack:
                local_hits.append(entry)
            if len(local_hits) >= 5:
                break

        shared_hits = []
        for s in self._history_sessions:
            sha = (s.get("git_commit_sha") or "unknown")[:10]
            for entry in (s.get("agent_logs") or []):
                if not isinstance(entry, dict):
                    continue
                fix = entry.get("fix_applied")
                if not fix:
                    continue
                fix_files = fix.get("files") if isinstance(fix, dict) else None
                haystack = " ".join([
                    str(fix.get("explanation", "") if isinstance(fix, dict) else fix),
                    str(fix_files or ""),
                ]).lower()
                if query_l in haystack:
                    shared_hits.append((sha, entry.get("t"), fix))
                if len(shared_hits) >= 5:
                    break
            if len(shared_hits) >= 5:
                break

        if not local_hits and not shared_hits:
            return f"PASTFIX '{query}': no prior fix (local or team-shared) mentions that."

        lines = []
        for e in local_hits:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("time", 0)))
            lines.append(f"  [local, {ts}] {os.path.basename(str(e.get('path', '?')))}: {e.get('explanation', '(no explanation)')}")
        for sha, t, fix in shared_hits:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "unknown time"
            explanation = fix.get("explanation", "(no explanation)") if isinstance(fix, dict) else str(fix)
            lines.append(f"  [team-shared, commit {sha}, {ts}]: {explanation}")
        return f"PASTFIX '{query}': {len(local_hits) + len(shared_hits)} prior fix(es) ({len(local_hits)} local, {len(shared_hits)} team-shared)\n" + "\n".join(lines)

    def _lint_check(self, content: str, path: str):
        """Automatic, non-model-invoked gate: syntax/AST validation, then
        pyflakes (if importable), run on the FULL proposed file content
        before it's ever written to disk. Returns (ok, messages)."""
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
        blocking = [m for m in messages if "undefined name" in m.lower() or "syntaxerror" in m.lower()]
        return (len(blocking) == 0), messages

    # ------------------------------------------------------------------
    # Execution tooling. Pulse CLI runs the agent conversation in the SAME
    # process as training (unlike the GUI, which talks to a separate
    # process over queues), so these act directly on self.watch_locals --
    # a live, periodically-refreshed {name: value} snapshot of every
    # tracked variable. That's a real, but bounded, limitation: only
    # tracked-variable names are visible here, not arbitrary
    # locals/globals or a live frame object.
    # ------------------------------------------------------------------
    def _exec_namespace(self) -> Dict[str, Any]:
        return dict(self.watch_locals)

    def _run_exec_repl(self, expr: str) -> str:
        """REPL: <expr> -- evaluate against the current tracked-variable
        snapshot right now, instead of reasoning from a context dump that
        may already be stale."""
        ns = self._exec_namespace()
        try:
            value = eval(expr, {"__builtins__": __builtins__, "math": math}, ns)
        except Exception:
            return f"REPL '{expr}': raised\n{_format_exc_short(traceback.format_exc())}"
        return f"REPL '{expr}' = {_describe_exec_value(value)}"

    def _run_exec_dryrun(self, call_expr: str) -> str:
        """DRYRUN: <function>(<args>) -- execute a specific call against
        the current tracked-variable snapshot and return the real output,
        exception, and traceback."""
        ns = self._exec_namespace()
        result: Dict[str, Any] = {}

        def _run():
            try:
                result["value"] = eval(call_expr, {"__builtins__": __builtins__, "math": math}, ns)
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

    # ------------------------------------------------------------------
    # TERMINAL: -- shared agent terminal execution (pulse_terminal.py).
    # Same infrastructure backs Pulse Code's TERMINAL: handling
    # (pulse_code.py only widens/narrows the safety-confirmation policy,
    # never the executor itself).
    # ------------------------------------------------------------------
    def _get_terminal_executor(self) -> "_terminal.TerminalExecutor":
        executor = getattr(self, "_terminal_executor", None)
        if executor is None:
            root = os.path.abspath(getattr(self, "_project_root", None) or self._repo_cwd or os.getcwd())
            executor = _terminal.TerminalExecutor(default_cwd=root)
            self._terminal_executor = executor
        return executor

    def _terminal_needs_confirmation(self, command: str) -> Optional[Dict[str, bool]]:
        """None if the command can just run; otherwise the classification flags that
        triggered a confirmation prompt. `self.review` (set on Pulse Code sessions,
        toggled by /review or -y) explicitly OFF skips confirmation entirely -- the same
        opt-out an applied code-fix already respects. Sessions with no `review` attribute
        (the plain debugging agent) default to always confirming destructive commands,
        since there is no equivalent 'apply without asking' flag there yet."""
        if getattr(self, "review", True) is False:
            return None
        flags = _terminal.classify_command(command)
        return flags if any(flags.values()) else None

    def _confirm_terminal_command(self, command: str, flags: Dict[str, bool]) -> bool:
        why = _terminal.describe_classification(flags)
        cprint(f"[Pulse] ⚠ This command {why}: {command}", color=_YELLOW)
        try:
            _flush_stdin()
            resp = _prompt_text(f"Run it anyway? (y/N) > ", label="Run it anyway? (y/N)").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return resp in ("y", "yes")

    def _run_terminal(self, arg: str, *, is_verification: bool = False) -> str:
        """TERMINAL: <command> -- run a real shell command via the shared TerminalExecutor
        and hand back a structured result the model can actually reason about (never just
        a 'done' -- see pulse_terminal.TerminalResult.render). Destructive-looking commands
        (delete files, rewrite git state, reach outside the workspace, start a background
        process) pause for a y/n first, exactly like applying a code fix already does;
        everything else -- reading files, grepping, running tests/linters, git status,
        launching a script -- runs immediately."""
        command, inline_timeout = _terminal.parse_inline_timeout(arg.strip())
        if not command:
            return "TERMINAL: empty command -- nothing to run."
        flags = self._terminal_needs_confirmation(command)
        if flags and not self._confirm_terminal_command(command, flags):
            return f"TERMINAL '{command}': the user declined to run this command -- try a different " \
                   "approach, or ask a read-only tool (GREP/VIEW) instead if you were only trying to " \
                   "inspect something."
        executor = self._get_terminal_executor()
        request = _terminal.TerminalRequest(
            command=command,
            timeout=inline_timeout or _terminal.DEFAULT_TIMEOUT_SECONDS,
        )
        result = executor.run(request, is_verification=is_verification or getattr(self, "_in_verification_pass", False))
        cprint(executor.summary_line(result), color=(_GREEN if result.ok else _RED))
        return result.render()

    def _run_exec_shapetrace(self, arg: str) -> str:
        """SHAPETRACE: [optional model var name] -- forward pass with a
        live tensor already tracked, dumping every submodule's shape via
        forward hooks. PyTorch only; only sees tracked variables."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return "SHAPETRACE: requires PyTorch to be importable; none found."
        ns = self._exec_namespace()
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
            return "SHAPETRACE: no torch.nn.Module found among tracked variables."
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
            return "SHAPETRACE: no tensor found among tracked variables to use as a synthetic input."

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
            if name:
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
        return f"SHAPETRACE (forward pass, synthetic input shape {tuple(sample.shape)}):\n" + "\n".join(records)

    def _run_exec_gradcheck(self, param_name: str) -> str:
        """GRADCHECK: <param> -- numerical finite-difference gradient
        check on a tracked parameter, deterministic pass/fail against its
        .grad. Requires a zero-arg `loss_fn` among tracked variables that
        recomputes the current scalar loss."""
        try:
            import torch
        except ImportError:
            return "GRADCHECK: requires PyTorch to be importable; none found."
        ns = self._exec_namespace()
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
            return f"GRADCHECK '{param_name}': couldn't find a tracked tensor parameter with a .grad by that name."
        if param.grad is None:
            return f"GRADCHECK '{param_name}': has no .grad yet -- call backward() at least once first."
        loss_fn = ns.get("loss_fn")
        if not callable(loss_fn):
            return (
                f"GRADCHECK '{param_name}': needs a zero-argument callable named `loss_fn` among tracked "
                "variables that recomputes and returns the current scalar loss -- define one and resend GRADCHECK."
            )
        eps = 1e-3
        flat = param.data.view(-1)
        grad_flat = param.grad.view(-1)
        n_check = min(5, flat.numel())
        if n_check == 0:
            return f"GRADCHECK '{param_name}': parameter is empty."
        idxs = sorted({int(i) for i in torch.linspace(0, flat.numel() - 1, n_check).tolist()})
        lines, max_rel_err = [], 0.0
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
            return f"GRADCHECK '{param_name}': loss_fn() raised while probing\n{_format_exc_short(traceback.format_exc())}"
        verdict = "PASS" if max_rel_err < 1e-2 else "FAIL"
        header = f"GRADCHECK '{param_name}': {verdict} (max rel_err={max_rel_err:.2e} over {n_check} sampled entries)\n"
        return header + "\n".join(lines)

    def _replay_maybe_checkpoint(self) -> None:
        try:
            import torch
        except ImportError:
            return
        ns = self._exec_namespace()
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
            checkpoints = getattr(self, "_replay_checkpoints", None)
            if checkpoints is None:
                checkpoints = []
                self._replay_checkpoints = checkpoints
            step = next(getattr(self, "_replay_step_counter", itertools.count()))
            self._replay_step_counter = getattr(self, "_replay_step_counter", itertools.count(step + 1))
            checkpoints.append((step, snap))
            del checkpoints[:-20]

    def _run_exec_replay(self, arg: str) -> str:
        """REPLAY: <n_steps> -- from the last checkpoint at least n_steps
        back, restore a copy of tracked state and replay n_steps via a
        zero-argument `train_step` tracked callable, reporting the
        resulting loss curve, then restore live state."""
        try:
            n_steps = int(str(arg).strip())
        except ValueError:
            return f"REPLAY '{arg}': expected an integer number of steps, e.g. 'REPLAY: 5'."
        checkpoints = getattr(self, "_replay_checkpoints", None)
        if not checkpoints:
            return (
                "REPLAY: no checkpoints captured yet -- Pulse periodically snapshots any tracked variable "
                "with a state_dict() (model/optimizer); wait for at least one snapshot after training starts."
            )
        try:
            import torch
        except ImportError:
            return "REPLAY: requires PyTorch to be importable; none found."
        if n_steps >= len(checkpoints):
            step, snap = checkpoints[0]
        else:
            step, snap = checkpoints[-1 - n_steps]

        ns = self._exec_namespace()
        train_step = ns.get("train_step")
        if not callable(train_step):
            return (
                "REPLAY: needs a zero-argument callable named `train_step` among tracked variables that "
                "performs one optimizer step on the live model and returns the step's loss."
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
            f"REPLAY: replayed {len(losses)} step(s) from a checkpoint at step {step} on an isolated copy "
            f"of tracked state, then restored live values back. Resulting loss curve: [{curve}]"
        )

    def _run_gpustatus(self) -> str:
        """GPUSTATUS: -- real per-device GPU memory/utilization. Under a
        multi-GPU/multi-rank launch, aggregates every rank's status
        (published periodically by pulse.py's auto_track) into one report
        instead of just this process's own device(s)."""
        rank, _local_rank, world_size, session_key = self.dist_info
        if world_size > 1 and session_key:
            return _format_multi_rank_gpu_status(session_key, rank)
        return _format_gpu_status(_gpu_status_snapshot())

    def _run_rollback(self, arg: str) -> str:
        """ROLLBACK: <commit_id, or 'last'> -- deterministically revert
        the workspace via the same .pulse_history mechanism /revert uses,
        instead of the agent trying to manually reconstruct old code from
        memory of the conversation. 'last' undoes the most recent commit;
        otherwise `arg` is matched as a commit id (see /log)."""
        entries = self._load_fix_log()
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
                return f"ROLLBACK '{arg}': no commit matches that id -- send LOG (or /log interactively) to see recorded commits."
        restored, failed, new_commit = self._perform_revert(entries, target_idx, target_label)
        if not restored and not failed:
            return f"ROLLBACK '{arg}': workspace already matches that state -- nothing to revert."
        parts = []
        if restored:
            parts.append(f"ROLLBACK: reverted to state {target_label} -- {len(restored)} file(s) restored: " + ", ".join(os.path.basename(p) for p in restored) + (f" (logged as commit {new_commit})" if new_commit else ""))
        if failed:
            parts.append(f"ROLLBACK: failed to restore {len(failed)} file(s): " + ", ".join(f"{os.path.basename(p)} ({err})" for p, err in failed))
        return "\n".join(parts)

    def _run_mllint(self) -> str:
        """MLLINT: -- a small set of high-confidence, AST-detectable ML
        anti-patterns across every tracked file. Heuristics about ML
        *semantics*, worded as things worth double-checking, not
        certainties the way the syntax LINT gate's findings are."""
        findings = _mllint_scan(self._iter_ast_trees())
        if not findings:
            return "MLLINT: no anti-patterns found in this heuristic check across the tracked files (this doesn't guarantee correctness, just that none of Pulse's known patterns matched)."
        lines = [f"  {label}:{lineno}: {msg}" for label, lineno, msg in findings]
        return f"MLLINT: {len(findings)} finding(s) worth double-checking\n" + "\n\n".join(lines)

    def _run_exec_layerstats(self, arg: str) -> str:
        """LAYERSTATS: [optional model variable name] -- per-layer
        gradient-norm/weight-norm ratio for every parameter of a tracked
        torch.nn.Module, right after backward() has populated .grad."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return "LAYERSTATS: requires PyTorch to be importable; none found."
        ns = self._exec_namespace()
        arg = (arg or "").strip()
        model = ns.get(arg) if arg and isinstance(ns.get(arg), nn.Module) else None
        if model is None:
            model = next((v for v in ns.values() if isinstance(v, nn.Module)), None)
        if model is None:
            return "LAYERSTATS: no torch.nn.Module found among tracked variables."

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

    def _run_exec_hardexamples(self, arg: str) -> str:
        """HARDEXAMPLES: [optional N, default 10] -- per-example loss for
        the current batch via a zero-arg `per_example_losses_fn` tracked
        callable, surfacing the highest-loss samples."""
        ns = self._exec_namespace()
        fn = ns.get("per_example_losses_fn")
        if not callable(fn):
            return (
                "HARDEXAMPLES: needs a zero-argument callable named `per_example_losses_fn` among "
                "tracked variables that returns a per-sample loss array for the current batch (e.g. "
                "`per_example_losses_fn = lambda: F.cross_entropy(model(x), y, reduction='none')`)."
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
        for rank_i, idx in enumerate(order, 1):
            label = f"id={id_arr[idx]}" if id_arr is not None else f"batch-position {idx}"
            lines.append(f"  #{rank_i}: {label}, loss={losses[idx]:.6g}")
        mean, std = float(losses.mean()), float(losses.std())
        return (
            f"HARDEXAMPLES: top {len(order)} of {len(losses)} example(s) by loss "
            f"(batch mean={mean:.4g}, std={std:.4g}):\n" + "\n".join(lines)
        )

    def _run_exec_ampstatus(self, arg: str) -> str:
        """AMPSTATUS: [optional GradScaler variable name] -- current
        mixed-precision (torch.cuda.amp.GradScaler) scale/skip state."""
        ns = self._exec_namespace()
        arg = (arg or "").strip()
        scaler = ns.get(arg) if arg else None
        if scaler is None:
            scaler = next((v for v in ns.values() if type(v).__name__ == "GradScaler"), None)
        if scaler is None:
            return "AMPSTATUS: no GradScaler found among tracked variables (this run may not be using AMP)."
        try:
            scale = float(scaler.get_scale())
        except Exception:
            scale = None
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

    def _run_exec_seedcheck(self, arg: str) -> str:
        """SEEDCHECK: -- a reproducibility fingerprint of every RNG this
        process can see, compared against the fingerprint from the last
        time SEEDCHECK ran on this project."""
        current = _seed_fingerprint()
        path = _seed_history_path(self.script_path)
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

    def _run_rankdiverge(self, var: str) -> str:
        """RANKDIVERGE: <var> -- compares this rank's latest value of a
        tracked scalar against every other rank's latest value."""
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
            return f"RANKDIVERGE '{var}': fewer than 2 ranks currently report this variable ({len(values)} found)."
        mean = sum(values.values()) / len(values)
        lines = [f"  rank {r}{' (this process)' if r == rank else ''}: {v:.6g}  (delta from mean: {v - mean:+.4g})" for r, v in sorted(values.items())]
        spread = max(values.values()) - min(values.values())
        verdict = "large spread across ranks -- worth investigating" if spread > abs(mean) * 0.1 + 1e-9 else "spread looks tight"
        return f"RANKDIVERGE '{var}' across {len(values)}/{world_size} rank(s), mean={mean:.6g}, spread={spread:.4g} ({verdict}):\n" + "\n".join(lines)

    def _run_runcompare(self) -> str:
        """RUNCOMPARE: -- this run's current scalar values vs the most
        recent previous run recorded for this project. Local file, plus
        (when this project's team history is available -- see
        _load_history_context) a note about the most recent *shared*
        session's outcome too."""
        current_scalars = {name: (v[-1] if v else None) for name, v in self.scalar_histories.items() if v}
        current_scalars = {k: v for k, v in current_scalars.items() if v is not None}
        if not current_scalars:
            return "RUNCOMPARE: no scalar values recorded yet this run."

        history = _load_run_history(self.script_path)
        run_id = getattr(self, "_run_id", None)
        previous = next((r for r in reversed(history) if r.get("run_id") != run_id), None)

        this_entry = next((r for r in history if r.get("run_id") == run_id), None)
        if this_entry is None:
            this_entry = {"run_id": run_id, "started": time.time()}
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

    def _record_usage(self, response) -> None:
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

    def _ensure_retry_ticker(self) -> None:
        """Starts the one background thread that periodically retries
        whatever's in self._pending_agent_retry / self._pending_restart_retry,
        if anything. Lazily started on first need rather than always
        running, and idempotent -- safe to call every time something gets
        queued."""
        if self._retry_ticker_started:
            return
        self._retry_ticker_started = True

        def _loop():
            while True:
                time.sleep(15.0)
                # Blocking acquire (with a timeout) rather than skip-if-busy:
                # _retry_ticker_lock is an RLock also held for the duration
                # of every real ask_agent() call (see ask_agent), so this
                # just waits for whatever the user is doing right now to
                # finish rather than racing it -- and since ask_agent()
                # itself re-enters this same lock reentrantly, calling it
                # from inside this `with` block below is safe, not a
                # deadlock.
                if not self._retry_ticker_lock.acquire(timeout=120.0):
                    continue
                try:
                    pending = self._pending_agent_retry
                    if pending and time.time() >= pending["next_attempt"]:
                        cprint(
                            "\n[Pulse] retrying an earlier diagnosis/fix request that hit a rate limit/"
                            "transient error -- training was not interrupted while waiting.",
                            color=_YELLOW,
                        )
                        try:
                            self.ask_agent(pending["question"], pending["include_code"], traceback_signature=pending.get("traceback_signature"))
                        except Exception:
                            pass  # _ask_agent_impl already handles/re-queues its own failures
                    pending_restart = self._pending_restart_retry
                    if pending_restart and time.time() >= pending_restart["next_attempt"]:
                        cprint("\n[Pulse] retrying an earlier restart that failed to launch...", color=_YELLOW)
                        try:
                            self._restart_process()  # does not return on success
                        except Exception:
                            pass
                finally:
                    self._retry_ticker_lock.release()

        threading.Thread(target=_loop, daemon=True).start()

    def _queue_agent_retry(self, question: str, include_code: bool, traceback_signature: Optional[str]) -> None:
        """Called when a top-level ask_agent() call failed only because
        of a transient agent-side error (rate limit, timeout, provider
        blip -- see _is_retryable_model_error) after exhausting its
        immediate in-call retries. Rather than losing that diagnosis/fix
        attempt for good, it's kept and retried again later on a backoff,
        while the run itself keeps going untouched in the meantime."""
        existing = self._pending_agent_retry
        backoff = 60.0
        if existing and existing.get("question") == question:
            backoff = min(existing.get("backoff", 60.0) * 2, 600.0)
        self._pending_agent_retry = {
            "question": question, "include_code": include_code,
            "traceback_signature": traceback_signature,
            "next_attempt": time.time() + backoff, "backoff": backoff,
        }
        cprint(
            f"[Pulse] will retry this request again in ~{backoff:.0f}s (agent request hit a rate limit/"
            "transient error) -- the run continues unaffected in the meantime.",
            color=_YELLOW,
        )
        self._ensure_retry_ticker()

    def _queue_restart_retry(self) -> None:
        """Called when _restart_process gave up after MAX_RESTART_ATTEMPTS
        because it couldn't even LAUNCH the replacement process (an OS/
        environment-level failure, not the training script itself
        crashing -- that second case already triggers an auto-rollback
        instead, since it means the fix itself is the problem, and
        retrying a demonstrably-broken fix forever isn't safe). A launch
        failure has nothing to do with whether the fix is good, so it's
        legitimately worth trying again later."""
        existing = self._pending_restart_retry
        backoff = 60.0
        if existing:
            backoff = min(existing.get("backoff", 60.0) * 2, 600.0)
        self._pending_restart_retry = {"next_attempt": time.time() + backoff, "backoff": backoff}
        cprint(f"[Pulse] will retry launching the replacement process again in ~{backoff:.0f}s.", color=_YELLOW)
        self._ensure_retry_ticker()

    def _run_cost(self) -> str:
        """COST: -- running token/cost usage for this chat session's
        agent calls, accumulated across every pipeline pass."""
        u = getattr(self, "_token_usage", None)
        if not u or u["calls"] == 0:
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

    def _echo_directives(self, requests: Dict[str, List[str]]) -> None:
        """Print a visible '-> /NAME arg' line for every directive the
        agent actually invoked, before any of its results are shown --
        so it's never a black box whether Pulse genuinely ran a real
        check (GREP'd the file, executed LINT, called out to the live
        process, ...) versus the agent merely claiming it did. Covers
        both the original CALC/PROMOTE/GREP/VIEW/... directives (see the
        call in _apply_directives) and the extended toolset (see
        _apply_new_directives)."""
        for name, args in requests.items():
            if not args:
                cprint(f"  -> /{name}", color=_YELLOW)
                continue
            for arg in args:
                label = f"/{name} {arg}" if arg else f"/{name}"
                cprint(f"  -> {label}", color=_YELLOW)

    def _apply_new_directives(self, requests: Dict[str, List[str]]) -> str:
        """Deterministically service the extended toolset the same way
        _apply_directives handles CALC/PROMOTE/GREP/VIEW."""
        self._echo_directives(requests)
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
        for arg in requests.get("rollback", []):
            notes.append(self._run_rollback(arg))
        for query in requests.get("pastfix", []):
            notes.append(self._run_pastfix(query))
        for arg in requests.get("dryrun", []):
            notes.append(self._run_exec_dryrun(arg))
        for arg in requests.get("repl", []):
            notes.append(self._run_exec_repl(arg))
        for arg in requests.get("replay", []):
            notes.append(self._run_exec_replay(arg))
        for arg in requests.get("terminal", []):
            notes.append(self._run_terminal(arg))
        for arg in requests.get("gradcheck", []):
            notes.append(self._run_exec_gradcheck(arg))
        if "shapetrace" in requests:
            for arg in requests["shapetrace"]:
                notes.append(self._run_exec_shapetrace(arg))
        if "mllint" in requests:
            notes.append(self._run_mllint())
        for arg in requests.get("layerstats", []):
            notes.append(self._run_exec_layerstats(arg))
        for arg in requests.get("hardexamples", []):
            notes.append(self._run_exec_hardexamples(arg))
        for arg in requests.get("ampstatus", []):
            notes.append(self._run_exec_ampstatus(arg))
        if "seedcheck" in requests:
            for arg in requests["seedcheck"]:
                notes.append(self._run_exec_seedcheck(arg))
        for var in requests.get("rankdiverge", []):
            notes.append(self._run_rankdiverge(var))
        if "runcompare" in requests:
            notes.append(self._run_runcompare())
        if "cost" in requests:
            notes.append(self._run_cost())
        return "\n\n".join(n for n in notes if n)

    def _apply_directives(
        self,
        calc_exprs: List[str],
        promote_names: List[str],
        gputrack_names: Optional[List[str]] = None,
        gpuuntrack_names: Optional[List[str]] = None,
        sensitivity_args: Optional[List[str]] = None,
        normal_start_args: Optional[List[str]] = None,
        grep_patterns: Optional[List[str]] = None,
        view_requests: Optional[List[str]] = None,
    ) -> str:
        """Deterministically compute any CALC: expressions and apply any
        PROMOTE:/GPUTRACK:/GPUUNTRACK:/SENSITIVITY:/NORMAL_START:/GREP:/
        VIEW: requests, returning a short human-readable summary to print
        and to feed back into the agent's own history (so it sees the
        verified numbers/state/code on the next turn instead of trusting
        its own arithmetic, memory, or a stale full-file dump).
        """
        self._echo_directives({k: v for k, v in {
            "calc": calc_exprs, "promote": promote_names, "gputrack": gputrack_names,
            "gpuuntrack": gpuuntrack_names, "sensitivity": sensitivity_args,
            "normal_start": normal_start_args, "grep": grep_patterns, "view": view_requests,
        }.items() if v})
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

        if grep_patterns:
            for pattern in grep_patterns:
                result = self._run_grep(pattern)
                print(f"  🔎 {result.splitlines()[0]}")
                notes.append(result)

        if view_requests:
            for arg in view_requests:
                result = self._run_view(arg)
                print(f"  📄 {result.splitlines()[0]}")
                notes.append(result)

        return "\n\n".join(notes)

    # The periodic check-in is its own agent, not the investigation pipeline: its own system
    # prompt, its own conversation (never agent_history), and only the tools it can really run.
    # It runs on a worker thread while training carries on, so tools that execute against live
    # training objects (REPL, DRYRUN, REPLAY, GRADCHECK, SHAPETRACE...) are not offered -- the
    # training thread is using those objects at the same moment. TERMINAL covers experiments:
    # it runs a separate process, under the same confirmation rules as everywhere else.
    _CHECKIN_SYSTEM_PROMPT = (
        "You are Pulse's periodic check-in: an ML engineer looking over a training run that is in "
        "progress, while it keeps running. Your job is to decide whether this run will produce a "
        "model and numbers that can be trusted, and if not, to say exactly why. That is the whole "
        "scope: something that makes the trained model worse than it should be, or its reported "
        "metrics wrong or misleading. Saving, logging, file layout, warnings, speed and style are "
        "out of scope unless they change the model or the numbers.\n\n"
        "Look for both kinds of problem:\n"
        "- INSTABILITY: divergence, NaN/inf, loss spikes, oscillation, a metric stuck at a constant "
        "or at chance, dead units, vanishing or exploding values.\n"
        "- BUGS that let training look healthy while the result is wrong. These are the ones a "
        "loss curve will never show, and the reason you read the code:\n"
        "    data leakage (a scaler/encoder fitted before the train/test split; test data or the "
        "target reaching the inputs; a window that includes the value being predicted),\n"
        "    validation that is not held out (validation_data built from the training set),\n"
        "    features and labels misaligned (one shuffled or permuted without the other, labels "
        "computed over the wrong axis),\n"
        "    wrong output activation or loss for the task (relu/linear into cross-entropy, softmax "
        "into a from_logits loss, a regression loss on class labels),\n"
        "    data scaled twice or not at all, targets on a different scale from what the loss "
        "expects,\n"
        "    capacity or hyperparameters far outside sane (a 1-unit layer, learning rate >= 0.1 "
        "for Adam, dropout near 1, zero-initialised weights) -- judged by what they do to this "
        "run, as below,\n"
        "    errors swallowed by try/except, so the script 'finishes' without training.\n"
        "  Warning signs in the numbers: validation better than training, a train/validation gap "
        "that keeps growing, accuracy suspiciously close to 1.0, or no epochs at all.\n\n"
        "TOOLS -- put each on its own line. Results come back to you as the next message, and you "
        "can keep investigating over several rounds before answering:\n"
        "  GREP: <pattern>              search the project's files (regex or plain text)\n"
        "  VIEW: <file>:<start>-<end>   exact line range; omit '<file>:' for the main script\n"
        "  CALC: <expression>           exact arithmetic (numbers, operators, math.*)\n"
        "  CORR: <var1> <var2>          correlation between two recorded histories\n"
        "  OUTLIER: <var>               z-score outliers in a recorded history\n"
        "  DIFFSTATS: <var> <i> <j>     exact change between two recorded points\n"
        "  HISTOGRAM: <var>             distribution of a recorded history\n"
        "  MLLINT:                      Pulse's static ML anti-pattern scan of the code\n"
        "  TERMINAL: <shell command>    ONE line -- only the text after 'TERMINAL:' on that line is "
        "run, so a heredoc or a quoted script spread over several lines arrives cut off. For "
        "Python, join statements with ';' in one python3 -c \"...\" line. It runs in the project "
        "directory, where the files you were shown live: cat/grep/head the "
        "code and data files, run a short python snippet to check a shape, a range, whether "
        "labels line up with features, what a split contains. It runs as a separate process, so "
        "it cannot see the live variables of the running script -- reload what you need. "
        "Commands that delete files, rewrite git state, reach outside the project, start a "
        "background process or overwrite a file via redirect ask the user first and may be "
        "declined; plain reads and short scripts run straight away. Keep scripts short: a slow "
        "command holds up this check. Read the exit code and output before relying on a result.\n"
        "  GPUTRACK: / GPUUNTRACK: <names>  start/stop the costly GPU-synced look at a variable\n\n"
        "The script is part-way through. Code after the current point has not run yet, so a file, "
        "folder, variable or value that does not exist yet is not evidence of a bug -- check where "
        "in the script execution is before concluding anything from what is missing.\n\n"
        "A bug in the code is a problem even when the numbers look healthy. Leakage and validation "
        "that is not held out make the numbers look BETTER, not worse; healthy-looking metrics are "
        "exactly how those bugs hide. If you find a defect that makes the model or its reported "
        "numbers wrong, the verdict is 'problem', whatever the curves look like.\n\n"
        "Also check that the run performs about as well as this setup can. From the code and the "
        "data, work out what a working version should reach -- the noise level in generated data, "
        "how separable the classes are, the scale of the target -- and compare the recorded "
        "numbers on a scale that means something: error relative to the target's spread, accuracy "
        "relative to chance and to what the data allows. Not 'better than chance': 'about what "
        "this setup should get'. This is a check, not a hunt. If the numbers are in the range this "
        "setup should reach, or you cannot tell what that range is, that is fine and not a "
        "problem. Slow progress, a run still part-way through, a small shortfall, or a choice that "
        "is merely small, simple or unusual is not a problem by itself. A shortfall is a problem "
        "only when it is large, the remaining epochs cannot plausibly close it, and you can point "
        "to what in the code causes it.\n\n"
        "Investigate before judging. The snapshot shows values, not intent; the code shows intent. "
        "When the code and the numbers disagree, trust neither until you have checked -- that "
        "disagreement is usually the bug. Report a problem only with evidence you can point to "
        "(a line, a value, a command's output), and do not report style or harmless choices.\n\n"
        "When you are done investigating, answer with ONLY these lines, and no tool lines:\n"
        "VERDICT: ok | problem\n"
        "PROBLEM: <if problem: what is wrong, where (file:line), and the evidence. Omit if ok.>\n"
        "NEXTCHECK: <minutes until the next check-in, {min_mins:g}-{max_mins:g}>\n"
        "CHECKNOTE: <a note to the next check-in: what to look at then and why -- a suspicion "
        "still to confirm, a number to watch, a line to re-read. 'none' if nothing.>"
    )

    _PERIODIC_CHECKIN_PROMPT = (
        "[Automatic check-in, {mins:g} minutes after the last one. Training is still running.]\n\n"
        "Note left for this check-in by the previous one:\n{note}\n\n"
        "Recorded metric history (evenly sampled, oldest first):\n{history}\n\n"
        "{snapshot}\n\n"
        "Currently GPU-tracked: {tracked}.\n\n"
        "Is this run healthy and its result trustworthy? Investigate with the tools, then give "
        "your final answer in the VERDICT/PROBLEM/NEXTCHECK/CHECKNOTE format."
    )

    _CHECKIN_VERDICT_ONLY = (
        "Your last reply had no VERDICT line, and no more tool results will come back. Based on "
        "everything you have seen, reply now with ONLY the VERDICT/PROBLEM/NEXTCHECK/CHECKNOTE lines."
    )

    _CHECKIN_LAST_ROUND = (
        "That was your last tool round -- no more tool results will come back. Answer now, with "
        "ONLY the VERDICT/PROBLEM/NEXTCHECK/CHECKNOTE lines, based on everything you have seen."
    )

    # Rounds of tool use a check-in may take before it must answer. Each round is one model
    # call, so this caps a check-in's cost at MAX_ROUNDS + 1 calls.
    _CHECKIN_MAX_ROUNDS = int(os.environ.get("PULSE_CHECKIN_ROUNDS", "6") or 6)
    _CHECKIN_TOOL_NAMES = ("terminal", "corr", "outlier", "diffstats", "histogram", "mllint")

    # Bounds on the agent-negotiated NEXTCHECK: interval -- keeps the
    # cadence genuinely dynamic (see _maybe_periodic_checkin and the
    # NEXTCHECK: line added to _START_PRIME_PROMPT) without letting a
    # malformed or adversarial reply set it to something absurd (a
    # near-0 value hammering the API in a loop, or a near-infinite value
    # that never checks in again).
    _CHECKIN_MIN_INTERVAL = 120.0    # 2 minutes
    _CHECKIN_MAX_INTERVAL = 3600.0   # 60 minutes
    _CHECKIN_OK_STATUSES = {
        "ok", "ok.", "okay", "fine", "healthy", "looks good", "looks fine",
        "looks healthy", "nominal", "good", "none", "no issues", "no problems",
    }
    _STATUS_RE = re.compile(r"^\s*STATUS:\s*(.+)$", re.MULTILINE)
    _VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(.+)$", re.MULTILINE)
    _PROBLEM_RE = re.compile(r"^\s*PROBLEM:\s*(.+?)(?=^\s*(?:NEXTCHECK|CHECKNOTE|VERDICT|GPUTRACK|GPUUNTRACK):|\Z)",
                             re.MULTILINE | re.DOTALL)
    _CHECKNOTE_RE = re.compile(r"^\s*CHECKNOTE:\s*(.+?)(?=^\s*(?:NEXTCHECK|VERDICT|PROBLEM|GPUTRACK|GPUUNTRACK):|\Z)",
                               re.MULTILINE | re.DOTALL)
    _NEXTCHECK_RE = re.compile(r"^\s*NEXTCHECK:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)

    @classmethod
    def _parse_nextcheck_minutes(cls, text: Optional[str]) -> Optional[float]:
        """Pull a NEXTCHECK: <minutes> line out of an agent reply (the
        periodic check-in prompt below, or the one-time
        _START_PRIME_PROMPT), clamped to [_CHECKIN_MIN_INTERVAL,
        _CHECKIN_MAX_INTERVAL] so a malformed or extreme reply can't set
        an unreasonable cadence. Returns None if there's no parseable
        NEXTCHECK: line at all, in which case the caller should leave
        the existing interval alone rather than guess."""
        m = cls._NEXTCHECK_RE.search(text or "")
        if not m:
            return None
        try:
            minutes = float(m.group(1))
        except ValueError:
            return None
        if minutes <= 0 or not math.isfinite(minutes):
            return None
        lo, hi = cls._CHECKIN_MIN_INTERVAL / 60.0, cls._CHECKIN_MAX_INTERVAL / 60.0
        return max(lo, min(hi, minutes))

    def _restart_env(self, depth: int) -> Dict[str, str]:
        """The environment for the restarted, fixed script: marked as a restart, and told
        where to resume from when there is a usable checkpoint and the fix did not ask for a
        fresh start."""
        env = dict(os.environ, **{_RESTART_CHILD_ENV: "1", _RESTART_DEPTH_ENV: str(depth + 1)})
        env.pop(_RESUME_ENV, None)
        checkpoint = getattr(self, "_fix_checkpoint", None)
        if not checkpoint:
            return env
        try:
            with open(checkpoint, encoding="utf-8") as f:
                finite = json.load(f).get("finite", False)
        except Exception:
            finite = False
        if not getattr(self, "_resume_after_fix", True):
            _agent_log_event("RESTARTING FROM SCRATCH -- the fix asked for a fresh start (resume: false)")
        elif not finite:
            _agent_log_event("RESTARTING FROM SCRATCH -- the checkpointed weights were not finite")
        else:
            env[_RESUME_ENV] = checkpoint
        return env

    def _save_fix_checkpoint(self) -> None:
        """Save the Keras model's weights as they are right now -- the point a fix starts --
        so the restarted, fixed script can resume from here (see _resume_from_fix_checkpoint).
        Nothing to save for non-Keras runs or before the first epoch; failures only mean the
        restart trains from the beginning, as it always did."""
        self._fix_checkpoint = None
        self._resume_after_fix = True
        model = getattr(self, "_keras_model", None)
        if model is None or not self.script_path:
            return
        try:
            folder = os.path.join(os.path.dirname(os.path.abspath(self.script_path)), ".pulse_checkpoints")
            os.makedirs(folder, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            weights = os.path.join(folder, f"fix-{stamp}.weights.h5")
            model.save_weights(weights)
            finite = all(bool(np.all(np.isfinite(w))) for w in model.get_weights())
            meta = {"weights": weights, "epoch": getattr(self, "_keras_epoch", -1),
                    "epochs": getattr(self, "_keras_epochs", None), "finite": finite}
            meta_path = os.path.join(folder, f"fix-{stamp}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            self._fix_checkpoint = meta_path
            _agent_log_event(f"CHECKPOINT SAVED before fixing (after epoch {meta['epoch'] + 1})",
                             f"{weights}\nweights finite: {finite}")
        except Exception as exc:
            _agent_log_event("CHECKPOINT FAILED -- a restart will train from the beginning",
                             f"{type(exc).__name__}: {exc}")

    def _escalate_training_problem(self, problem: Optional[str]) -> None:
        """Shared escalation path for anything that decides training
        looks like it's going wrong -- either _check_for_trouble's
        deterministic, every-step signal-based detector, or the agent's
        own free-text STATUS: read from a periodic check-in (see
        _maybe_periodic_checkin). Either way: pause (even out of
        continuous mode), hand the problem to the agent to diagnose and,
        if possible, fix, then resume automatically. Deduplicated
        against the last-escalated problem so a standing issue that's
        already being worked doesn't re-trigger on every single
        step/check-in it's still present for.
        """
        if not problem or problem == self._last_intervention_signature:
            return
        self._last_intervention_signature = problem
        _agent_log_event("ESCALATED -- pausing to investigate and fix", problem)
        self._save_fix_checkpoint()
        self.continuous = True

        ui = _ui.enabled()
        if ui:
            _ui.header("Auto-intervention")
            _ui.warn("Anomaly detected")
            for detected_line in str(problem).splitlines() or [""]:
                _ui._emit("  " + detected_line)
            _ui._emit()
            _ui.note("Pulse is investigating the failure." if (self.agent_provider and self.agent_key)
                     else "No agent is configured, so this can't be investigated yet.")
        else:
            print("\n" + "=" * 60)
            cprint(
                "[Pulse] ⚠ Auto-intervention: training looks like it's going bad. "
                "Diagnosing while preserving the loop.",
                color=_RED,
            )
            cprint(f"[Pulse] Detected: {problem}", color=_RED)
            print("=" * 60)

        if self.agent_provider and self.agent_key:
            question = (
                "Pulse just auto-paused training because it detected a problem: "
                f"{problem}\nPlease diagnose the root cause and, if you can, fix it."
            )
            if ui:
                _ui.header("Investigation")   # the stages below are the agent pipeline's real ones
            else:
                cprint("Pulse:")
            self._pending_agent_start_ts = time.monotonic()
            self._pending_agent_problem = problem
            self.ask_agent(question, include_code=bool(self.code_text))
            self._finalize_agent_downtime()
            if ui:
                _ui.header("Auto-intervention")
                if getattr(self, "_fix_applied_this_turn", False):
                    _ui.ok("Pulse intervened", "a code fix was applied")
                else:
                    _ui.note("No code change was applied.")
        else:
            cprint(
                "[Pulse] No AI agent is configured yet -- problem queued. Run /agent to set one "
                "up and it will be diagnosed immediately.",
                color=_YELLOW,
            )
            self._log_incident("auto_intervention", problem, downtime_seconds=0.0, fix_applied=False)
            if not hasattr(self, "_queued_interventions"):
                self._queued_interventions = []
            if problem not in self._queued_interventions:
                self._queued_interventions.append(problem)

        if ui:
            _ui.ok("Training resumed")
            _ui._emit()
        else:
            cprint("[Pulse] Continuing training automatically.")

    def _maybe_periodic_checkin(self) -> None:
        """Every checkin_interval seconds (agent-chosen: NEXTCHECK: from the start-of-run
        prime, then from each check-in's own answer), run a check-in: an agent that reads the
        code, the metric history and the current snapshot, investigates over several tool
        rounds, and returns a VERDICT. A 'problem' verdict goes through the same
        _escalate_training_problem path as a deterministic detection. Whoever schedules a
        check-in also leaves it a CHECKNOTE -- what to look at and why -- which it is given.

        The conversation (model calls and tool use) runs on a worker thread, so training
        never waits on it; its answer is applied here, on the training thread, once it lands.
        """
        pending = self._checkin_call
        if pending is not None:
            if not pending.done:
                return                      # still investigating; the loop carries on meanwhile
            self._checkin_call = None
            answer, transcript = pending.result if pending.result else (None, [])
            self._finish_periodic_checkin(answer, pending.error, pending.prompt, transcript)
            return
        if not self.agent_provider or not self.agent_key:
            return
        now = time.monotonic()
        if now - self._last_checkin < self.checkin_interval:
            return
        self._last_checkin = now

        # Everything the check-in reads from live state is captured here, on the training thread.
        tracked = ", ".join(sorted(self.gpu_tracked_vars)) if self.gpu_tracked_vars else "(none)"
        try:
            snapshot = self._build_agent_context(include_code=bool(self.code_text))
        except Exception as exc:
            snapshot = f"(unable to build a variable snapshot: {exc})"
        try:
            history = self._checkin_history_summary()
        except Exception as exc:
            history = f"(unable to summarise metric history: {exc})"
        prompt = self._PERIODIC_CHECKIN_PROMPT.format(
            mins=self.checkin_interval / 60.0, tracked=tracked, snapshot=snapshot,
            history=history, note=(getattr(self, "_checkin_note", "") or "(none)"),
        )
        cprint(
            f"\n[Pulse] ⏱ {self.checkin_interval / 60:g}-min check-in -- agent is reviewing the run "
            "for bugs and instability (training continues)...",
            color=_YELLOW,
        )
        _agent_log_event(f"PERIODIC CHECK-IN started ({self.checkin_interval / 60:g} min after the last)")
        if _async_model_calls_enabled():
            self._checkin_call = _BackgroundModelCall(lambda: self._run_checkin(prompt), prompt=prompt)
            return
        try:
            (answer, transcript), error = self._run_checkin(prompt), None
        except AgentRequestFailed as exc:
            answer, transcript, error = None, [], exc
        self._finish_periodic_checkin(answer, error, prompt, transcript)

    def _checkin_history_summary(self, max_points: int = 12, max_series: int = 40) -> str:
        """Each recorded scalar history as a short evenly-sampled series. The snapshot alone
        is one instant; trends (validation parting from training, a metric pinned at a
        constant) need the history, and a check-in cannot ask the live process for it."""
        histories = self._detector_histories()
        lines = []
        for name in sorted(histories)[:max_series]:
            values = []
            for v in histories[name]:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    continue
            if not values:
                continue
            n = len(values)
            if n <= max_points:
                picked = values
            else:
                step = (n - 1) / (max_points - 1)
                picked = [values[round(i * step)] for i in range(max_points)]
            shown = ", ".join(f"{v:.4g}" for v in picked)
            lines.append(f"- {name} ({n} readings): {shown}")
        return "\n".join(lines) if lines else "(no scalar history recorded yet)"

    def _checkin_service_tools(self, answer: str) -> Tuple[str, List[str]]:
        """Run the tools a check-in asked for. Returns (results for the model, one-line
        summaries for the console). Only the check-in's toolset: nothing here touches live
        training objects, since this runs on a worker thread while training continues."""
        results: List[str] = []
        summary: List[str] = []
        for expr in (m.strip() for m in self._CALC_RE.findall(answer)):
            if expr:
                results.append(f"CALC {expr} = {_safe_eval_math(expr)}")
                summary.append(f"CALC {expr}")
        for pattern in (m.strip() for m in self._GREP_RE.findall(answer)):
            if pattern:
                results.append(self._run_grep(pattern))
                summary.append(f"GREP {pattern}")
        for arg in (m.strip() for m in self._VIEW_RE.findall(answer)):
            if arg:
                results.append(self._run_view(arg))
                summary.append(f"VIEW {arg}")
        _clean, requests = self._extract_new_directives(answer)
        runners = {
            "terminal": self._run_terminal, "corr": self._run_corr, "outlier": self._run_outlier,
            "diffstats": self._run_diffstats, "histogram": self._run_histogram,
            "mllint": lambda _arg: self._run_mllint(),
        }
        for name in self._CHECKIN_TOOL_NAMES:
            for arg in requests.get(name, []):
                try:
                    results.append(runners[name](arg))
                except Exception as exc:          # one broken tool must not end the check-in
                    results.append(f"{name.upper()} {arg}: failed ({type(exc).__name__}: {exc})")
                summary.append(f"{name.upper()} {arg}".strip())
        refused = sorted(set(requests) - set(self._CHECKIN_TOOL_NAMES))
        if refused:
            results.append(
                "Not available during a check-in (training is using the live objects right now): "
                + ", ".join(r.upper() for r in refused) + ". Use TERMINAL to run a separate check instead.")
            summary.append("refused " + ", ".join(r.upper() for r in refused))
        return "\n\n".join(r for r in results if r), summary

    def _run_checkin(self, prompt: str) -> Tuple[str, List[str]]:
        """The check-in conversation: up to _CHECKIN_MAX_ROUNDS rounds of tool use, then an
        answer. Its own system prompt and its own message list -- never agent_history.
        Returns (final answer, console transcript of the tools it used)."""
        system = self._CHECKIN_SYSTEM_PROMPT.format(
            min_mins=self._CHECKIN_MIN_INTERVAL / 60.0, max_mins=self._CHECKIN_MAX_INTERVAL / 60.0)
        history: List[Dict[str, str]] = []
        message = prompt
        transcript: List[str] = []
        answer = ""
        for round_no in range(self._CHECKIN_MAX_ROUNDS + 1):
            answer = self._call_model(message, max_tokens=_AGENT_MAX_TOKENS, system=system, history=history,
                                      purpose=f"periodic check-in, round {round_no + 1}")
            if self._checkin_verdict(answer)[0] is not None:
                break                               # answered -- any stray tool lines are ignored
            results, used = self._checkin_service_tools(answer)
            if not used:
                break                               # neither a verdict nor a tool: take it as is
            transcript.extend(f"round {round_no + 1}: {u}" for u in used)
            history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
            last = round_no + 1 >= self._CHECKIN_MAX_ROUNDS
            message = (results or "(the tools returned nothing)") + "\n\n" + (
                self._CHECKIN_LAST_ROUND if last else
                "Keep investigating with more tool lines, or give your final VERDICT answer now.")
        if self._checkin_verdict(answer)[0] is None:
            # Ran out of rounds mid-investigation, or answered in prose: one more call, for the
            # verdict alone, rather than throwing away everything it has looked at.
            history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
            answer = self._call_model(self._CHECKIN_VERDICT_ONLY, max_tokens=_AGENT_MAX_TOKENS,
                                      system=system, history=history,
                                      purpose="periodic check-in, asking for the missing verdict")
            transcript.append("verdict: asked again for a missing VERDICT line")
        return answer, transcript

    @classmethod
    def _checkin_verdict(cls, answer: Optional[str]) -> Tuple[Optional[bool], str]:
        """(is_problem, description). is_problem is None when the answer carries no verdict.

        VERDICT: ok|problem is the format; the old free-text STATUS: line is still read so an
        answer in that shape is not lost. Either way only the verdict's first word decides --
        'ok -- everything finite, scaler fitted on train' is ok. Matching the whole line against
        a list of exact phrases used to turn every explained 'ok' into an escalation."""
        text = answer or ""
        m = cls._VERDICT_RE.search(text)
        if m:
            first = re.split(r"[\s.,;:!\-—–(]+", m.group(1).strip().lower(), maxsplit=1)[0]
            problem_m = cls._PROBLEM_RE.search(text)
            problem = problem_m.group(1).strip() if problem_m else ""
            if first in ("ok", "okay", "healthy", "fine", "good", "none"):
                return False, ""
            return True, problem or m.group(1).strip()
        m = cls._STATUS_RE.search(text)
        if m:
            status = m.group(1).strip()
            first = re.split(r"[\s.,;:!\-—–(]+", status.lower(), maxsplit=1)[0]
            if first in ("ok", "okay", "healthy", "fine", "good", "none", "nominal"):
                return False, ""
            if status.lower().rstrip(".") in cls._CHECKIN_OK_STATUSES:
                return False, ""
            return True, status
        return None, ""

    def _finish_periodic_checkin(self, answer, error, prompt, transcript=None) -> None:
        """Everything a check-in does with the agent's answer. Runs on the training thread."""
        if error is not None:
            if not isinstance(error, AgentRequestFailed):
                raise error
            # A periodic, low-stakes background check -- never worth interrupting training
            # over. Skip this round and try again at the next interval.
            cprint(f"[Pulse] ⚠ Check-in skipped (agent request failed: {error})", color=_RED)
            _agent_log_event("CHECK-IN SKIPPED: agent request failed", str(error))
            return
        answer = answer or ""
        if transcript:
            cprint(f"[Pulse] Check-in investigated with {len(transcript)} tool call(s): "
                   + "; ".join(t.split(': ', 1)[1][:80] for t in transcript), color=_YELLOW)

        _, _calc, _promote, gputrack_names, gpuuntrack_names, _sens, _norm, _grep, _view = self._extract_directives(answer)
        summary = self._apply_directives([], [], gputrack_names, gpuuntrack_names)
        if summary:
            cprint(f"[Pulse] {summary}", color=_YELLOW)

        note_m = self._CHECKNOTE_RE.search(answer)
        note = note_m.group(1).strip() if note_m else ""
        self._checkin_note = "" if note.lower().rstrip(".") in ("", "none", "n/a") else note

        is_problem, problem = self._checkin_verdict(answer)
        _agent_log_event(
            "CHECK-IN VERDICT: " + {None: "none (inconclusive)", False: "ok", True: "problem"}[is_problem],
            "\n".join(x for x in (problem, f"tools used: {len(transcript or [])}",
                                    f"note for the next check-in: {self._checkin_note or 'none'}") if x))
        if is_problem is None:
            cprint("[Pulse] Check-in gave no verdict -- treating this one as inconclusive.", color=_YELLOW)
        elif not is_problem:
            cprint("[Pulse] Check-in verdict: ok", color=_YELLOW)
        else:
            cprint(f"[Pulse] Check-in verdict: problem -- {problem}", color=_YELLOW)
            # The investigation that follows starts from what the check-in found.
            self.agent_history.append({"role": "user", "content": prompt})
            self.agent_history.append({"role": "assistant", "content": answer})
            if self.auto_intervene:
                self._escalate_training_problem(f"[periodic check-in] {problem}")
            else:
                cprint(
                    "[Pulse] Auto-intervene is off -- not auto-pausing for this, but flagging "
                    "it for you to look at.",
                    color=_YELLOW,
                )
        if self._checkin_note:
            cprint(f"[Pulse] Note for the next check-in: {self._checkin_note}", color=_YELLOW)

        next_minutes = self._parse_nextcheck_minutes(answer)
        if next_minutes is not None:
            next_seconds = next_minutes * 60.0
            if abs(next_seconds - self.checkin_interval) > 1e-6:
                cprint(
                    f"[Pulse] Agent set its next check-in for {next_minutes:g} minutes from now "
                    f"(was {self.checkin_interval / 60:g}).",
                    color=_YELLOW,
                )
            self.checkin_interval = next_seconds

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

    def _verify_fix_with_retries(self, fix: Dict[str, Any], diagnosis: str):
        """PASS 4: check the fix's math/logic before it's handed to the
        user. If it fails, ask the agent to revise and re-check, up to
        _MAX_VERIFY_ATTEMPTS times. Returns (fix, passed, reason).

        A failed verify/revise *request* never discards the fix: the fix
        was already developed, so it goes ahead as unverified best effort
        (same as a verdict that doesn't clearly pass) rather than being
        dropped along with the diagnosis behind it.
        """
        reason = ""
        seen = set()
        for attempt in range(_MAX_VERIFY_ATTEMPTS):
            fix_desc = self._describe_fix(fix)
            signature = json.dumps([fix["old"], fix["new"]], sort_keys=True)
            if signature in seen:
                # Same fix, same evidence -- asking again just burns calls.
                return fix, True, (reason or "(re-examined and unchanged)")
            seen.add(signature)
            try:
                with _Spinner("Checking the fix"):
                    verify_answer = self._call_model(
                        _PASS4_VERIFY_TMPL.format(diagnosis=diagnosis, fix_desc=fix_desc),
                        max_tokens=_AGENT_MAX_TOKENS,
                    )
            except AgentRequestFailed as exc:
                return fix, False, f"(verification request failed: {exc})"
            verdict = self._parse_json_obj(verify_answer)
            if verdict is None:
                # Not the JSON verdict -- most likely a TERMINAL:/GREP:/VIEW: directive
                # checking something for real before answering. Service it and give the
                # model another chance to actually answer, bounded so a confused reply
                # can't stall verification forever.
                for _tool_round in range(_MAX_VERIFY_TOOL_ROUNDS):
                    note = self._service_tool_requests(verify_answer)
                    if not note:
                        break
                    print(f"[tool results]\n{note}\n")
                    self.agent_history.append({"role": "assistant", "content": verify_answer})
                    self.agent_history.append({"role": "user", "content": note})
                    try:
                        with _Spinner("Checking the fix"):
                            verify_answer = self._call_model(
                                _PASS4_VERIFY_TMPL.format(diagnosis=diagnosis, fix_desc=fix_desc),
                                max_tokens=_AGENT_MAX_TOKENS,
                            )
                    except AgentRequestFailed as exc:
                        return fix, False, f"(verification request failed: {exc})"
                    verdict = self._parse_json_obj(verify_answer)
                    if verdict is not None:
                        break
                if verdict is None:
                    # Still nothing usable after giving it a chance to check -- don't
                    # block the user on a formatting slip; hand off as best effort.
                    return fix, True, "(verification response was unparsable; proceeding anyway)"
            passes = bool(verdict.get("passes"))
            reason = str(verdict.get("reason", "")).strip()
            if passes:
                return fix, True, reason

            # Hand the check's finding back and let the model decide.
            try:
                with _Spinner("Re-examining the fix"):
                    recheck_answer = self._call_model(
                        _PASS4_RECHECK_TMPL.format(
                            diagnosis=diagnosis, fix_desc=fix_desc,
                            reason=reason or "(no reason given)"),
                        max_tokens=_AGENT_MAX_TOKENS,
                    )
            except AgentRequestFailed as exc:
                return fix, False, f"(re-examination request failed: {exc})"

            decision_obj = self._parse_json_obj(recheck_answer) or {}
            decision = str(decision_obj.get("decision", "")).strip().lower()
            note = str(decision_obj.get("reason", "") or decision_obj.get("explanation", "")).strip()
            if decision == "keep":
                return fix, True, f"kept by the agent after re-examination: {note or '(no reason given)'}"
            if decision == "drop":
                return fix, False, f"dropped by the agent after re-examination: {note or '(no reason given)'}"
            revised = self._parse_code_fix(recheck_answer)
            if revised is None:
                # No usable decision: treat the check as unconfirmed rather
                # than applying a fix nobody stood behind.
                return fix, False, (reason or "(the check did not pass and the agent did not respond usably)")
            fix = revised
        return fix, False, (reason or "(verification did not clearly pass after retries)")

    _PROBE_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

    def _service_tool_requests(self, answer: str) -> str:
        """Run any directives the model put in `answer` (GREP:/VIEW:/CALC:
        and the extended toolset) and return their combined output, or ""
        if it asked for nothing. Same servicing pass 2 gets -- shared so
        that asking for context works wherever the model does it."""
        notes = []
        try:
            (_clean, calc_exprs, promote_names, gputrack_names, gpuuntrack_names,
             sensitivity_args, normal_start_args, grep_patterns, view_requests) = self._extract_directives(answer)
            if calc_exprs or promote_names or grep_patterns or view_requests:
                note = self._apply_directives(
                    calc_exprs, promote_names, gputrack_names, gpuuntrack_names,
                    sensitivity_args, normal_start_args, grep_patterns, view_requests,
                )
                if note:
                    notes.append(note)
            _clean2, new_requests = self._extract_new_directives(answer)
            if new_requests:
                note = self._apply_new_directives(new_requests)
                if note:
                    notes.append(note)
        except Exception as exc:  # a malformed directive must not end the fix
            _pulse_log(f"TOOL REQUEST servicing failed: {exc!r}")
        return "\n\n".join(n for n in notes if n)

    def _run_sweep_and_maybe_recurse(self, include_code: bool, _depth: int) -> None:
        """PASS 5: re-read everything for OTHER, unrelated errors. If any
        turn up, ask the user whether to fix those too (PASS 6 recurses
        through the same 1-5 format for the new issue)."""
        with _Spinner("Reading for other errors"):
            sweep_answer = self._call_model(_PASS5_SWEEP, max_tokens=_AGENT_MAX_TOKENS)
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
        with self._retry_ticker_lock:
            if _depth == 0:
                self._last_applied_fix = None
                self._last_problem_description = question
            answer = self._ask_agent_impl(question, include_code=include_code, _depth=_depth)
            if _depth == 0:
                if self._last_call_failed_transiently:
                    self._queue_agent_retry(question, include_code, traceback_signature)
                elif self._pending_agent_retry and self._pending_agent_retry.get("question") == question:
                    # This exact question just succeeded (either a fresh ask,
                    # or the retry ticker below re-asking it) -- nothing left
                    # to retry.
                    self._pending_agent_retry = None
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
          5. SWEEP   -- re-read everything for OTHER errors; ask the user y/n.          6.         -- if yes, recurse through 1-5 for the new issue(s).

        Each pass prints as soon as it's ready, with a small spinner shown
        while it's in flight. Only the top-level call (_depth == 0)
        restarts the process afterward, once, if any fix was applied
        anywhere in the (possibly recursive) chain.
        """
        if _depth == 0:
            self._fix_applied_this_turn = False
            self._last_call_failed_transiently = False

        # Cheap best-effort snapshot for REPLAY: <n_steps> -- see
        # _replay_maybe_checkpoint. Runs once per top-level question
        # rather than continuously, since (unlike the GUI) there's no
        # separate always-on ticker thread here.
        if _depth == 0:
            try:
                self._replay_maybe_checkpoint()
            except Exception:
                pass

        # Same cadence: enrich this rank's status file with current
        # scalar values, for RANKDIVERGE, under a multi-rank launch.
        if _depth == 0:
            rank, local_rank, world_size, session_key = self.dist_info
            if world_size > 1 and session_key:
                try:
                    scalars = {name: v[-1] for name, v in self.scalar_histories.items() if v}
                    _write_rank_status(session_key, rank, local_rank, world_size, extra={"scalars": scalars} if scalars else None)
                except Exception:
                    pass

        if not self.agent_provider or not self.agent_key:
            return "(AI agent is not enabled. Run setup again or set the API key.)"

        context = self._build_agent_context(include_code=include_code)
        user_content = f"{context}\n\nQuestion: {question}"
        self.agent_history.append({"role": "user", "content": user_content})

        wants_implementation = include_code and any(
            kw in question.lower() for kw in _IMPLEMENT_KEYWORDS
        )

        try:
            # Pass 1: locate the region(s) of the error. May itself investigate first via
            # GREP:/VIEW:/TERMINAL:/etc. instead of guessing -- serviced and looped the same
            # way PASS 3's fix loop already is, bounded so it can't stall forever.
            with _Spinner("Reading for region of error"):
                regions = self._call_model(_PASS1_LOCATE, max_tokens=_AGENT_MAX_TOKENS)
            for _round in range(_MAX_LOCATE_TOOL_ROUNDS):
                note = self._service_tool_requests(regions)
                if not note:
                    break
                print(f"[tool results]\n{note}\n")
                self.agent_history.append({"role": "assistant", "content": regions})
                self.agent_history.append({"role": "user", "content": note})
                with _Spinner("Reading for region of error"):
                    regions = self._call_model(_PASS1_LOCATE, max_tokens=_AGENT_MAX_TOKENS)
            print(f"\n[1] Region of error\n{regions}\n")

            # Pass 2: focused second read + diagnosis/reasoning. Investigative directives
            # (GREP:/VIEW:/TERMINAL:/etc.) are looped the same way, so the diagnosis below is
            # written after seeing what they turned up, not before. CALC:/PROMOTE:/GPUTRACK:/
            # GPUUNTRACK: are instrumentation side-effects, not investigation -- applied
            # deterministically same-turn either way, no need to loop on those alone.
            raw_analysis = analysis = ""
            new_requests: Dict[str, List[str]] = {}
            for _round in range(_MAX_ANALYZE_TOOL_ROUNDS + 1):
                with _Spinner("Analyzing"):
                    raw_analysis = self._call_model(
                        _PASS2_ANALYZE_TMPL.format(regions=regions), max_tokens=_AGENT_MAX_TOKENS
                    )
                (
                    analysis, calc_exprs, promote_names, gputrack_names, gpuuntrack_names,
                    sensitivity_args, normal_start_args, grep_patterns, view_requests,
                ) = self._extract_directives(raw_analysis)
                analysis, new_requests = self._extract_new_directives(analysis)
                directive_note = self._apply_directives(
                    calc_exprs, promote_names, gputrack_names, gpuuntrack_names,
                    sensitivity_args, normal_start_args, grep_patterns, view_requests,
                )
                new_note = self._apply_new_directives(new_requests) if new_requests else ""
                if new_note:
                    print(f"[tool results]\n{new_note}\n")
                combined_note = "\n\n".join(n for n in (directive_note, new_note) if n)
                investigating = bool(grep_patterns or view_requests or new_requests)
                if investigating and _round < _MAX_ANALYZE_TOOL_ROUNDS:
                    # Still gathering evidence -- feed the results back and diagnose on a
                    # later round, once there's actually something to diagnose from.
                    self.agent_history.append({"role": "assistant", "content": raw_analysis})
                    if combined_note:
                        self.agent_history.append({"role": "user", "content": combined_note})
                    continue
                if combined_note:
                    self.agent_history.append({"role": "user", "content": combined_note})
                break
            print(f"[2] Diagnosis & reasoning\n{analysis}\n")

            full_answer = f"{regions}\n\n{analysis}"

            if not wants_implementation:
                # No code change requested -- pass 3 is just the concrete fix
                # in text; nothing to verify or sweep.
                with _Spinner("Developing fix"):
                    fix_text = self._call_model(
                        _PASS3_FIX_TEXT_TMPL.format(diagnosis=full_answer), max_tokens=_AGENT_MAX_TOKENS
                    )
                print(f"[3] Fix\n{fix_text}\n")
                full_answer += f"\n\n{fix_text}"
                self.agent_history.append({"role": "assistant", "content": full_answer})
                return full_answer

            # Pass 3: develop and implement the fix. Passes 1-2's findings
            # are handed over explicitly -- otherwise this call sees only the
            # original code + traceback and has to re-derive the diagnosis
            # from scratch (tool results from pass 2 are already in history).
            with _Spinner("Developing & implementing fix"):
                fix_answer = self._call_model(
                    _PASS3_IMPLEMENT_TMPL.format(diagnosis=full_answer), max_tokens=_AGENT_MAX_TOKENS
                )
            fix = self._parse_code_fix(fix_answer)

            # Wanting to see more code is not a failed fix. Pass 2 already
            # services GREP:/VIEW:/REPL: and friends; this pass used to
            # accept nothing but the finished JSON, so a model that asked
            # for context ("let me look at lines 295-305") ended the whole
            # pipeline with the bug undiagnosed. Answer what it asked for
            # and let it try again, a bounded number of times.
            for _round in range(_MAX_FIX_TOOL_ROUNDS):
                if fix is not None:
                    break
                print(f"[3] Fix (not final)\n{fix_answer}\n")
                self.agent_history.append({"role": "assistant", "content": fix_answer})
                note = self._service_tool_requests(fix_answer)
                if note:
                    print(f"[tool results]\n{note}\n")
                    self.agent_history.append({"role": "user", "content": note})
                else:
                    self.agent_history.append({"role": "user", "content": _PASS3_NO_TOOLS_NOTE})
                try:
                    with _Spinner("Developing & implementing fix"):
                        fix_answer = self._call_model(
                            _PASS3_IMPLEMENT_TMPL.format(diagnosis=full_answer),
                            max_tokens=_AGENT_MAX_TOKENS,
                        )
                except AgentRequestFailed:
                    break
                fix = self._parse_code_fix(fix_answer)

            if fix is None:
                print(f"[3] Fix\n{fix_answer}\n")
                full_answer += f"\n\n{fix_answer}"
                self.agent_history.append({"role": "assistant", "content": full_answer})
                return full_answer

            # Pass 4: verify the fix's math/logic before handing it to the
            # user; revise and re-check on failure (bounded retries).
            fix, verify_ok, verify_reason = self._verify_fix_with_retries(fix, full_answer)
            status = "passed" if verify_ok else "not confirmed -- fix NOT applied"
            print(f"[4] Verification {status}: {verify_reason}\n")
            if not verify_ok:
                # Nobody stood behind this edit: the check flagged it and the
                # agent, shown that finding, did not confirm it. Writing it
                # anyway is how a working program gets broken after the real
                # fix has already landed.
                result = f"{full_answer}\n\n(Fix not applied: {verify_reason})"
                self.agent_history.append({"role": "assistant", "content": full_answer})
                return result

            self.agent_history.append({"role": "assistant", "content": full_answer})
            apply_result = self._apply_code_fix(fix)
            first_pass_landed = self._fix_applied_this_turn

            # If any snippet failed to match (verbatim or fuzzy), give the
            # model ONE chance to re-quote it exactly before accepting the
            # miss -- see the matching logic in pulse.py's _ask.
            if self._last_apply_skipped:
                retry_fix = self._request_corrected_snippets(fix, self._last_apply_skipped)
                if retry_fix is not None:
                    self._fix_applied_this_turn = False
                    retry_result = self._apply_code_fix(retry_fix)
                    retry_landed = self._fix_applied_this_turn
                    self._fix_applied_this_turn = first_pass_landed or retry_landed
                    if retry_landed:
                        note = "additional" if first_pass_landed else "retry after correcting a snippet mismatch --"
                        apply_result = apply_result + f"\n\n({note} change applied)\n" + retry_result

            result = f"{full_answer}\n\n{apply_result}"

            # There is no probe step here any more (it was "Pass 4.5"). It re-ran a 75 s
            # slice of training after each fix and, when the loss had not visibly fallen in
            # that window, wrote the ORIGINAL code back and asked for a revision; when the
            # revision could not be parsed it stopped there, leaving the bug on disk while
            # reporting "fix left applied". The resumed run and the post-run confirmation
            # (_confirm_fix_did_its_job) judge the fix instead, on the whole run.

            # Pass 5 (+ 6): a re-read of the whole file looking for OTHER,
            # unrelated bugs, which then recurses into a fresh diagnose/fix
            # round of its own. Off unless asked for ("sweep": true in
            # pulse_config.json, or PULSE_SWEEP=1): on a benchmark of 36
            # fixed bugs it turned 36 pipelines into 83, and most of the
            # work happened after the reported bug was already fixed.
            if self.sweep_enabled and self._fix_applied_this_turn and _depth < 3:
                try:
                    self._run_sweep_and_maybe_recurse(include_code, _depth)
                except AgentRequestFailed as exc:
                    cprint(f"[Pulse] ⚠ Full re-read skipped (agent request failed: {exc}) -- continuing with the applied fix.", color=_YELLOW)
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
            if not (_depth == 0 and self._fix_applied_this_turn):
                if _depth == 0:
                    self._last_call_failed_transiently = True
                return msg
            # A fix already landed on disk before this later request failed
            # -- still restart so it actually gets run and tested.
            result = msg

        if _depth == 0 and self._fix_applied_this_turn and not self._suppress_auto_restart:
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

        if not ("old" in payload and "new" in payload):
            # Some models wrap the object they were asked for in an envelope
            # ({"pulse_analysis": {...}, "json": {"old": [...], ...}}). The
            # schema is right, it is just one level down -- take it rather
            # than throwing a usable fix away over packaging.
            for value in payload.values():
                if isinstance(value, dict) and "old" in value and "new" in value:
                    payload = value
                    break

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
        self, files: Dict[str, tuple], explanation: str, kind: str = "fix",
        created: Optional[set] = None,
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
            file_entry = {"path": fpath, "before": before, "after": after, "diff": diff}
            if created and fpath in created:
                file_entry["created"] = True      # `before` is "" because the file did not exist
            file_entries.append(file_entry)

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

    def _perform_revert(self, entries: List[Dict[str, Any]], target_idx: int, target_label: str):
        """Core of /revert, factored out so it can also be invoked
        automatically (see _auto_rollback_after_failed_restarts) without
        going through interactive arg-parsing. Restores every file the
        changelog has ever touched to its state as of `target_idx`
        (`-1` meaning "before the very first recorded commit"), logs the
        revert itself as a new commit, and returns (restored_paths,
        failed_paths, new_commit_id)."""
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

        new_commit = None
        if restored:
            new_commit = self._record_fix_commit(
                revert_files, f"Reverted to state {target_label}", kind="revert"
            )
            if new_commit:
                self._last_commit_id = new_commit
                self._log_incident(
                    "revert", f"Reverted to state {target_label}",
                    commit_id=new_commit, files=[os.path.basename(p) for p in restored],
                )
        return restored, failed, new_commit

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

        restored, failed, new_commit = self._perform_revert(entries, target_idx, target_label)

        if restored:
            cprint(f"[Pulse CLI] ✓ Reverted to state {target_label} -- {len(restored)} file(s) restored:", color=_BLUE)
            for p in restored:
                print(f"    - {os.path.basename(p)}")
            if new_commit:
                cprint(f"[Pulse CLI] 📝 Logged as commit {new_commit}.")
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

    def _auto_rollback_after_failed_restarts(self, chain_start_commit_id: Optional[str]) -> bool:
        """Automatic safety net for the auto-fix/auto-restart loop: called
        when a restarted process keeps crashing until MAX_RESTART_ATTEMPTS
        is exhausted (see _restart_process). Rather than leaving a
        known-broken fix sitting on disk indefinitely and just telling a
        possibly-nobody-is-watching unattended run to go run /revert
        itself, this automatically restores the workspace to its state
        right BEFORE the fix that started this failing chain -- using the
        exact same persistent .pulse_history mechanism /revert uses, so
        nothing is silently lost (the failed chain is still fully
        recoverable via /log + /revert afterward). Returns True if
        anything was actually restored.
        """
        entries = self._load_fix_log()
        if not entries or not chain_start_commit_id:
            cprint(
                "[Pulse CLI] ⚠ Could not automatically roll back (no recorded pre-fix state found) -- "
                "use /log and /revert <id> to inspect and restore manually.",
                color=_RED,
            )
            return False

        chain_start_idx = None
        for i, e in enumerate(entries):
            if e.get("id") == chain_start_commit_id:
                chain_start_idx = i
                break
        if chain_start_idx is None:
            cprint(
                f"[Pulse CLI] ⚠ Could not find commit {chain_start_commit_id} in the change log to roll "
                "back from -- use /log and /revert <id> to inspect and restore manually.",
                color=_RED,
            )
            return False

        target_idx = chain_start_idx - 1
        target_label = f"before commit {chain_start_commit_id} (auto-rollback after repeated restart failures)"
        restored, failed, new_commit = self._perform_revert(entries, target_idx, target_label)

        if restored:
            cprint(
                f"[Pulse CLI] 🛟 Automatic rollback: the fix chain starting at commit "
                f"{chain_start_commit_id} kept crashing after every retry, so Pulse restored "
                f"{len(restored)} file(s) to their state right before that fix -- instead of leaving "
                "known-broken code on disk for an unattended run.",
                color=_YELLOW,
            )
            for p in restored:
                print(f"    - {os.path.basename(p)}")
            if new_commit:
                cprint(f"[Pulse CLI] 📝 Logged as commit {new_commit}. Use /log + /revert to undo this rollback if it wasn't wanted.")
        if failed:
            cprint(f"[Pulse CLI] ⚠ Automatic rollback failed to restore {len(failed)} file(s):", color=_RED)
            for p, err in failed:
                print(f"    - {os.path.basename(p)}: {err}")
        return bool(restored)

    def _apply_code_fix(self, fix: Dict[str, Any]) -> str:
        """Apply an agent-proposed code fix to the file(s) it targets.

        Each fix entry may target a different file (see "files" in the
        code-fix schema, for a modularized project) -- edits are grouped by
        resolved file path so each file is read/written once regardless of
        how many snippets in it changed. Each old[i] must appear exactly
        once in its target file's current contents; snippets that don't
        match cleanly, or whose file can't be resolved, are skipped and
        reported rather than guessed at. On success, backups of every
        touched file are kept so the user can revert them all together
        (/revert, via .pulse_history).

        There is no git stash here any more. Stashing the working tree before
        each fix also stashed Pulse's own earlier fix from the same run, so a
        second fix was written onto the original code and the first one
        vanished into `git stash` -- a correct fix undone by the next attempt.
        """
        self._build_file_labels()

        by_path: Dict[str, List[tuple]] = {}
        unresolved = []
        self._last_apply_lint_failed = []
        for old, new, label in zip(fix["old"], fix["new"], fix["files"]):
            path = self._resolve_fix_path(label)
            if not path:
                unresolved.append((old, label))
                continue
            by_path.setdefault(path, []).append((old, new))

        if not by_path and not unresolved and not fix.get("create"):
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
                    content = content.replace(old, _banner_wrap_fix(old, new, path), 1)
                    applied.append((old, new))
                    continue
                if count == 0:
                    # Fallback: tolerate a snippet that's right except for
                    # indentation/blank-line differences -- see the
                    # matching comment in pulse.py's _write_code_fix. Only
                    # taken when it's unambiguous.
                    span = _find_fuzzy_snippet_span(content, old)
                    if span is not None:
                        start, end = span
                        actual_old = content[start:end]
                        content = content[:start] + _banner_wrap_fix(actual_old, new, path) + content[end:]
                        applied.append((actual_old, new))
                    else:
                        skipped.append((old, path, "no exact match found in the file"))
                else:
                    # Ambiguous -- but if the crash happened inside one of the
                    # occurrences (a duplicated cell/block is exactly how that
                    # happens), that one is the one the fix is about.
                    start = self._occurrence_at_crash(content, old, path)
                    if start is not None:
                        content = content[:start] + _banner_wrap_fix(old, new, path) + content[start + len(old):]
                        applied.append((old, new))
                        cprint(
                            f"     snippet occurs {count}x -- applied the one at the crash "
                            f"(line {self._last_crash_location[1]})", color=_YELLOW,
                        )
                    else:
                        skipped.append((old, path, f"matched {count} times (ambiguous), skipped for safety"))

            if not applied:
                continue

            # Automatic gate -- NOT model-invoked, runs regardless of what
            # directives the model used. Syntax/AST validation (always)
            # plus a real lint pass (pyflakes, if importable) on the FULL
            # proposed file content, before anything touches disk.
            cprint(f"  -> /lint {os.path.basename(path)}", color=_YELLOW)
            lint_ok, lint_messages = self._lint_check(content, path)
            if not lint_ok:
                self._last_apply_lint_failed.append(path)
                cprint(f"     lint FAILED -- fix will not be written:\n     " + "\n     ".join(lint_messages), color=_RED)
                lines.append(
                    f"⚠ Fix for '{path}' failed the automatic syntax/lint gate and was NOT written:\n"
                    + "\n".join(lint_messages)
                )
                continue
            cprint("     lint passed", color=_YELLOW)

            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as exc:
                lines.append(f"⚠ Failed to write changes to '{path}': {exc}")
                continue

            originals[path] = original_content
            applied_by_path[path] = applied
            after_by_path[path] = content
            _agent_log_event(
                f"CODE FIX WRITTEN to {os.path.basename(path)}",
                (fix.get("explanation", "") or "").strip() + "\n" + "".join(difflib.unified_diff(
                    original_content.splitlines(True), content.splitlines(True),
                    fromfile="before", tofile="after", n=1)))
            for old, new in applied:
                self._append_fixlog(path, old, new, fix.get("explanation", ""))
            if path == self.script_path:
                self.code_text = content
            elif path in self.extra_files:
                self.extra_files[path] = content

        # New files (Pulse Code). A fix from the debugger never has a "create" key.
        created_paths: set = set()
        for spec in fix.get("create") or []:
            created = self._create_file_for_fix(spec, skipped)
            if created:
                cpath, ctext = created
                originals[cpath] = ""
                after_by_path[cpath] = ctext
                applied_by_path[cpath] = [("", ctext)]
                created_paths.add(cpath)

        for old, label in unresolved:
            skipped.append((old, label or "(unspecified file)", "couldn't determine which file this targets"))

        if not applied_by_path:
            lines.append("\n⚠ No changes were applied -- none of the proposed snippets matched cleanly:")
            for old, where, reason in skipped:
                lines.append(f"  - [{os.path.basename(str(where))}] {reason}: {old.splitlines()[0][:80]}...")
            self._last_apply_skipped = skipped
            # Printed like the success path below -- no caller prints the
            # returned text, so otherwise a rejected fix fails silently.
            cprint("\n".join(lines), color=_RED)
            return "\n".join(lines)

        for path, applied in applied_by_path.items():
            if path in created_paths:
                lines.append(f"\n✓ Created '{os.path.basename(path)}' ({len(after_by_path[path].splitlines())} lines)")
                continue
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
        commit_id = self._record_fix_commit(commit_files, fix.get("explanation") or "(no explanation given)",
                                            created=created_paths)
        if commit_id:
            self._last_commit_id = commit_id
            lines.append(f"\n📝 Logged as commit {commit_id} in .pulse_history/ -- /revert {commit_id} to undo, or /log to see history.")
            self._log_incident(
                "fix_applied", fix.get("explanation") or "(no explanation given)",
                commit_id=commit_id, files=[os.path.basename(p) for p in applied_by_path],
            )

        cprint("\n".join(lines), color=_BLUE)

        self._pending_revert_backups = dict(originals)  # path -> original content
        self._fix_applied_this_turn = True  # tells ask_agent's top-level call to restart afterward
        if fix.get("resume") is False or str(fix.get("resume", "")).strip().lower() == "false":
            self._resume_after_fix = False
        self._last_applied_fix = fix  # structured old/new/files/explanation, for cross-session reuse
        self._last_apply_skipped = skipped

        return "\n".join(lines)

    def _create_file_for_fix(self, spec: Any, skipped: List[tuple]) -> Optional[tuple]:
        """Write one new file named by a fix's "create" list. Returns (path, content) or None
        (with the reason appended to `skipped`). Only Pulse Code uses this. A new file must
        be inside the project root, must not already exist (existing files are edited with
        old/new snippets, never replaced), and goes through the same lint gate as any edit."""
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str) or not isinstance(spec.get("content"), str):
            skipped.append(("(new file)", "(unspecified file)", "a new file needs a string path and content"))
            return None
        root = os.path.abspath(getattr(self, "_project_root", None) or self._repo_cwd or os.getcwd())
        target = os.path.abspath(os.path.join(root, spec["path"]))
        shown = spec["path"]
        if os.path.commonpath([root, target]) != root:
            skipped.append((f"create {shown}", shown, "outside the project root -- refused"))
            return None
        parts = os.path.relpath(target, root).split(os.sep)
        if any(part in (".git", ".pulse_history") for part in parts):
            skipped.append((f"create {shown}", shown, "inside a protected directory -- refused"))
            return None
        if os.path.exists(target):
            skipped.append((f"create {shown}", shown, "already exists -- edit it with old/new snippets instead"))
            return None
        content = spec["content"] if spec["content"].endswith("\n") or not spec["content"] else spec["content"] + "\n"
        cprint(f"  -> /lint {os.path.basename(target)}", color=_YELLOW)
        lint_ok, lint_messages = self._lint_check(content, target)
        if not lint_ok:
            self._last_apply_lint_failed.append(target)
            cprint(f"     lint FAILED -- file will not be written:\n     " + "\n     ".join(lint_messages), color=_RED)
            return None
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            skipped.append((f"create {shown}", shown, f"couldn't write file: {exc}"))
            return None
        cprint("     lint passed", color=_YELLOW)
        self.extra_files[target] = content
        return target, content

    def _request_corrected_snippets(self, fix: Dict[str, Any], skipped: List[tuple]) -> Optional[Dict[str, Any]]:
        """One bounded retry for snippets that failed to match (verbatim
        or fuzzy) in _apply_code_fix -- see the matching, more heavily
        commented copy in pulse.py's ChatPanel. Shows the model the
        actual current content of each affected file plus exactly which
        of its own snippets didn't match, and asks ONLY for corrected
        old/new pairs -- not a full re-diagnosis. Returns a fix dict for
        just the corrected entries, or None if nothing is recoverable."""
        if not skipped:
            return None

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
        answer = self._call_model(prompt, max_tokens=_AGENT_MAX_TOKENS)
        return self._parse_code_fix(answer)

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
            # But cap the window at 30% of actual history so the check
            # fires in time on short runs (e.g. 50 epochs × 9 batches =
            # 450 steps -- a 395-step window means it never triggers
            # until there's almost no data left to act on).
            "stagnation_window": min(
                round(500 - 350 * s),
                max(10, round(len(max(
                    (self.scalar_histories.get(v) or [] for v in self.tracked_vars),
                    key=lambda h: len(h), default=[]
                )) * 0.30))
            ),
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
    
    # Checks the engine runs that the legacy detector did not have a counterpart for,
    # and vice versa, are listed in the benchmark README; the switch below is what
    # decides which of the two a normal `pulse run` actually uses.
    _LEGACY_DETECTOR_ENV = "PULSE_LEGACY_DETECTOR"

    def _detector_histories(self) -> Dict[str, List[Any]]:
        """Every scalar history worth checking, whatever route it arrived by.

        Names come from three places and the union matters: a Keras metric exists only
        in epoch_scalar_histories, a sampled local only in scalar_histories, and the
        per-variable checks used to iterate tracked_vars alone -- so a metric that
        static discovery had not found was never checked at all, which is why sixteen
        Deep4ge runs detected nothing.
        """
        names = set(getattr(self, "epoch_scalar_histories", {}) or {})
        names.update(getattr(self, "scalar_histories", {}) or {})
        names.update(getattr(self, "tracked_vars", []) or [])
        histories: Dict[str, List[Any]] = {}
        for name in names:
            try:
                values = self._history_for_detector(name)
            except Exception:
                continue
            if values:
                histories[name] = list(values)
        return histories

    def _detection_engine(self):
        """The engine, kept across calls: confirmations and clearing are stateful."""
        engine = getattr(self, "_detector", None)
        overrides = {
            "explosion_multiplier": self.explosion_multiplier,
            "plateau_range_frac": self.plateau_range_frac,
            "oscillation_flip_threshold": self.oscillation_flip_threshold,
            "oscillation_delta_frac": self.oscillation_delta_frac,
            "stagnation_frac": self.stagnation_frac,
        }
        if engine is None:
            engine = _pulse_detect.DetectionEngine(sensitivity=self.sensitivity)
            self._detector = engine
        # /sensitivity takes effect on the next check, with no restart, as before.
        engine.sensitivity = self.sensitivity
        engine.overrides = {k: v for k, v in overrides.items() if v is not None}
        return engine

    def _check_for_trouble(self) -> Optional[str]:
        """Is this training run going wrong? Returns why, or None.

        The judgment itself lives in pulse_detect.DetectionEngine, which is the version
        that is tested and benchmarked. On the 37-run detection benchmark the two agree
        on every broken run and disagree sharply on healthy ones: the implementation
        below this one raised a false alarm on 7 of 15 healthy runs -- a deliberate
        learning-rate drop, a converged run sitting at its floor, a fine-tune improving
        slowly, a GAN, a noisy small validation split -- against 0 for the engine.

        Set PULSE_LEGACY_DETECTOR=1 to get the old one back.
        """
        if os.environ.get(self._LEGACY_DETECTOR_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
            return self._check_for_trouble_legacy()

        histories = self._detector_histories()
        if not histories:
            return None
        tensor_stats = {}
        for name, entry in (getattr(self, "_matrix_cache", {}) or {}).items():
            if isinstance(entry, dict) and isinstance(entry.get("stats"), dict):
                tensor_stats[name] = entry["stats"]
        step = max((len(values) for values in histories.values()), default=0)
        try:
            raised = self._detection_engine().update(
                histories, step=step, tensor_stats=tensor_stats or None)["raised"]
        except Exception as exc:
            # A detector that raises takes the training run with it. Say nothing.
            if _PULSE_LOGGING:
                try:
                    _pulse_log(f"DETECTOR ERROR {type(exc).__name__}: {exc}")
                except Exception:
                    pass
            return None
        # Info-level findings are observations, not problems: "accuracy has not moved"
        # on a fine-tune sitting at 97% is true and worth showing, and pausing the run
        # to ask an agent about it is not. Only actionable severities escalate.
        raised = [f for f in raised if f.severity in (_pulse_detect.CRITICAL, _pulse_detect.WARNING)]
        if not raised:
            return None
        order = {_pulse_detect.CRITICAL: 0, _pulse_detect.WARNING: 1}
        raised.sort(key=lambda f: (order.get(f.severity, 9), -f.confidence))
        result = "; ".join(f.message for f in raised)
        if _PULSE_LOGGING:
            try:
                _pulse_log(f"DETECTOR result={result!r} "
                           f"histories={ {k: len(v) for k, v in histories.items()} !r}")
            except Exception:
                pass
        return result

    def _check_for_trouble_legacy(self) -> Optional[str]:
        """
        Deterministic runtime training-health detector.

        The detector intentionally separates:
        1. Hard failures: NaN / inf
        2. Sudden numerical failures: loss spikes
        3. Validation regression: val_loss consistently worsening
        4. Train/validation divergence: training improves while validation worsens
        5. Loss stagnation / plateau
        6. Strong loss oscillation
        7. Metric stagnation
        8. Matrix/tensor NaN / inf

        Important:
        - Do NOT require 10+ epochs to detect validation regression.
        - Do NOT compare identical early/late windows.
        - Do NOT trigger on None values alone. None generally means the
            variable simply wasn't readable from the current scope.
        - Histories are sparse by design, so detection works with whatever
            scalar observations are actually available.
        """

        reasons = []
        th = self._sensitivity_thresholds()

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------

        def _finite(values):
            return [
                v for v in values
                if v is not None
                and isinstance(v, (int, float))
                and math.isfinite(v)
            ]

        def _history(name):
            # Epoch-level metrics first: under Keras, loss/accuracy arrive
            # once per epoch through the fit hook and land in
            # epoch_scalar_histories, while scalar_histories (step-level
            # sampling) can stay empty for the whole run. Reading only the
            # latter meant a Keras run had nothing to detect on at all.
            return _finite(self._history_for_detector(name))

        def _find_history(*names):
            """
            Find the first useful history among exact names and then
            fall back to case-insensitive matching.
            """
            for name in names:
                values = _history(name)
                if values:
                    return name, values

            wanted = {n.lower() for n in names}

            for actual_name in self.scalar_histories:
                if actual_name.lower() in wanted:
                    values = _history(actual_name)
                    if values:
                        return actual_name, values

            return None, []

        def _is_loss_name(name):
            return _looks_like_loss(name)

        def _is_validation_name(name):
            lower = name.lower()
            return any(
                token in lower
                for token in (
                    "val",
                    "valid",
                    "validation",
                    "test",
                    "eval",
                )
            )

        def _is_train_loss_name(name):
            lower = name.lower()
            return (
                _is_loss_name(name)
                and not _is_validation_name(name)
            )

        # ------------------------------------------------------------
        # 1. Inspect individual tracked scalar histories
        # ------------------------------------------------------------

        for var_name in self.tracked_vars:
            hist = self._history_for_detector(var_name)
            if not hist:
                continue

            latest_raw = hist[-1]

            # --------------------------------------------------------
            # Hard numerical failure
            # --------------------------------------------------------

            if (
                latest_raw is not None
                and isinstance(latest_raw, (int, float))
                and not math.isfinite(latest_raw)
            ):
                reasons.append(
                    f"'{var_name}' just went non-finite (NaN/inf): {latest_raw}"
                )
                continue

            # None means Pulse could not currently read the value.
            #
            # DO NOT treat this as a training failure. A huge number of
            # statically discovered variables are legitimately inaccessible
            # from the current runtime scope.
            if latest_raw is None:
                continue

            hist_values = _finite(hist)

            if not hist_values:
                continue

            latest = hist_values[-1]

            # --------------------------------------------------------
            # G. Exact repetition -- value frozen bit-for-bit
            #
            # Distinct from the plateau check below (which tolerates
            # small movement): if a loss/metric hasn't changed AT ALL
            # across several consecutive observations, that's a much
            # stronger signal of an actual bug -- a forward/backward
            # pass silently not running, an optimizer step not
            # executing, a metric callback reading a stale/cached value
            # -- rather than just slow learning. Scoped to loss/metric
            # names specifically (not every tracked scalar) since a
            # config value or a counter is *supposed* to sit still.
            # --------------------------------------------------------
            if (_is_loss_name(var_name) or _looks_like_metric(var_name)) and len(hist_values) >= 6:
                frozen_window = hist_values[-6:]
                if len(set(frozen_window)) == 1:
                    reasons.append(
                        f"'{var_name}' has been exactly frozen at {frozen_window[-1]:.6g} for the "
                        f"last {len(frozen_window)} observations -- not just slow-moving, but "
                        "bit-for-bit unchanged, which usually means something isn't actually "
                        "running each step (a forward/backward pass being skipped, an optimizer "
                        "step not executing, or a metric reading a stale/cached value) rather than "
                        "the model just learning slowly."
                    )

            # --------------------------------------------------------
            # H. Gradient/weight-norm explosion or collapse
            #
            # A grad/weight-norm scalar is expected to move smoothly; a
            # sudden multiplicative jump is exploding gradients, and a
            # sudden collapse toward zero is vanishing gradients -- the
            # latter has NO equivalent among the loss-specific checks
            # below, since a vanishing-gradient network can keep
            # producing a loss curve that still looks like it's (very
            # slowly) improving right up until it flatlines.
            # --------------------------------------------------------
            if _looks_like_grad_or_weight_norm(var_name) and len(hist_values) >= 5:
                recent = hist_values[-50:]
                previous = recent[:-1]
                if previous:
                    baseline = sum(previous) / len(previous)
                    if baseline > 1e-12:
                        if latest > baseline * th["explosion_multiplier"]:
                            reasons.append(
                                f"'{var_name}' spiked to {latest:.4g}, {latest / baseline:.1f}x its "
                                f"recent average ({baseline:.4g}) -- looks like exploding "
                                "gradients/weights."
                            )
                        elif latest < baseline / th["explosion_multiplier"]:
                            reasons.append(
                                f"'{var_name}' collapsed to {latest:.4g} from a recent average of "
                                f"{baseline:.4g} -- looks like vanishing gradients (the network may "
                                "have effectively stopped learning even if loss/metrics still look "
                                "okay for now)."
                            )

            # --------------------------------------------------------
            # I. Learning-rate discontinuity
            #
            # A tracked lr/learning_rate scalar is expected to change
            # only smoothly/deliberately (a scheduler step, a warmup
            # ramp) -- a sudden order-of-magnitude jump between two
            # consecutive observations usually means the scheduler
            # itself is misconfigured, not that a change was intended.
            # --------------------------------------------------------
            if _looks_like_learning_rate(var_name) and len(hist_values) >= 2:
                prev_lr = hist_values[-2]
                if prev_lr and prev_lr > 0 and latest > 0:
                    ratio = latest / prev_lr
                    if ratio >= 10.0 or ratio <= 0.1:
                        reasons.append(
                            f"'{var_name}' jumped from {prev_lr:.4g} to {latest:.4g} ({ratio:.3g}x) "
                            "between consecutive observations -- worth double-checking a "
                            "learning-rate scheduler isn't misconfigured (wrong step count/"
                            "milestone, wrong decay factor, or being stepped more than once per "
                            "actual optimizer step)."
                        )

            # --------------------------------------------------------
            # Loss-specific checks
            # --------------------------------------------------------

            if _is_loss_name(var_name):

                # ====================================================
                # A. Sudden loss explosion
                # ====================================================

                recent = hist_values[-50:]

                if len(recent) >= 2:
                    previous = recent[:-1]

                    if previous:
                        baseline = min(previous)

                        seed = self._normal_start_baselines.get(var_name)
                        if seed is not None and math.isfinite(seed):
                            baseline_candidates = [baseline, seed]
                            baseline = min(baseline_candidates)

                        if (
                            baseline > 0
                            and latest > baseline * th["explosion_multiplier"]
                        ):
                            reasons.append(
                                f"'{var_name}' spiked to {latest:.4g}, "
                                f"{latest / baseline:.1f}x its recent minimum "
                                f"({baseline:.4g})"
                            )

                # ====================================================
                # B. Short-term plateau
                # ====================================================

                if len(hist_values) >= 8:
                    plateau_window = hist_values[-20:]

                    scale = (
                        sum(abs(v) for v in plateau_window)
                        / len(plateau_window)
                    )

                    if scale == 0:
                        scale = max(abs(latest), 1e-12)

                    window_range = (
                        max(plateau_window)
                        - min(plateau_window)
                    )

                    plateau_threshold = (
                        scale * th["plateau_range_frac"]
                    )

                    if window_range <= plateau_threshold:
                        reasons.append(
                            f"'{var_name}' has plateaued: "
                            f"range over the last {len(plateau_window)} "
                            f"observations is {window_range:.3g}"
                        )

                # ====================================================
                # C. Oscillation
                # ====================================================

                if len(hist_values) >= 8:
                    oscillation_window = hist_values[-20:]

                    deltas = [
                        oscillation_window[i]
                        - oscillation_window[i - 1]
                        for i in range(1, len(oscillation_window))
                    ]

                    sign_flips = sum(
                        1
                        for i in range(1, len(deltas))
                        if deltas[i] != 0
                        and deltas[i - 1] != 0
                        and (
                            (deltas[i] > 0 and deltas[i - 1] < 0)
                            or
                            (deltas[i] < 0 and deltas[i - 1] > 0)
                        )
                    )

                    if deltas:
                        scale = (
                            sum(abs(v) for v in oscillation_window)
                            / len(oscillation_window)
                        )

                        avg_delta = (
                            sum(abs(d) for d in deltas)
                            / len(deltas)
                        )

                        if (
                            sign_flips
                            >= th["oscillation_flip_threshold"]
                            and scale > 0
                            and avg_delta
                            > scale * th["oscillation_delta_frac"]
                        ):
                            reasons.append(
                                f"'{var_name}' is heavily oscillating "
                                f"({sign_flips} directional reversals in the "
                                f"last {len(oscillation_window)} observations)"
                            )

                # ====================================================
                # D. Long-term stagnation
                # ====================================================

                stagnation_window = max(
                    8,
                    int(th["stagnation_window"])
                )

                if len(hist_values) >= stagnation_window:
                    long_window = hist_values[-stagnation_window:]

                    quarter = max(
                        2,
                        len(long_window) // 4
                    )

                    early = long_window[:quarter]
                    late = long_window[-quarter:]

                    early_mean = sum(early) / len(early)
                    late_mean = sum(late) / len(late)

                    if early_mean != 0:
                        relative_change = (
                            abs(late_mean - early_mean)
                            / abs(early_mean)
                        )

                        if relative_change < th["stagnation_frac"]:
                            reasons.append(
                                f"'{var_name}' hasn't meaningfully improved "
                                f"over the last {stagnation_window} observations "
                                f"({early_mean:.4g} → {late_mean:.4g})"
                            )

                # ====================================================
                # E. Validation-loss regression
                #
                # This is the important fix.
                #
                # val_loss is often sampled once per epoch, so requiring
                # 10 observations is unnecessarily slow. Five observations
                # are enough to detect a persistent monotonic regression.
                # ====================================================

                if _is_validation_name(var_name) and len(hist_values) >= 5:

                    trend = hist_values[-5:]

                    worsening_steps = sum(
                        trend[i] > trend[i - 1]
                        for i in range(1, len(trend))
                    )

                    first = trend[0]
                    last = trend[-1]

                    if (
                        worsening_steps >= 4
                        and first > 0
                        and last > first
                    ):
                        pct = (
                            (last - first)
                            / abs(first)
                            * 100.0
                        )

                        reasons.append(
                            f"'{var_name}' has consistently worsened "
                            f"({first:.4g} → {last:.4g}, +{pct:.1f}% "
                            f"over the last {len(trend)} observations)"
                        )

                # ====================================================
                # F. Validation regression even when noisy
                #
                # A monotonic sequence isn't required. Compare the first
                # and second half of a recent window.
                # ====================================================

                if _is_validation_name(var_name) and len(hist_values) >= 8:

                    trend_window = hist_values[-8:]

                    half = len(trend_window) // 2

                    early = trend_window[:half]
                    late = trend_window[half:]

                    early_mean = sum(early) / len(early)
                    late_mean = sum(late) / len(late)

                    if (
                        early_mean > 0
                        and late_mean > early_mean
                        * (1.0 + th["stagnation_frac"])
                    ):
                        pct = (
                            (late_mean - early_mean)
                            / abs(early_mean)
                            * 100.0
                        )

                        reasons.append(
                            f"'{var_name}' is trending worse: "
                            f"recent average increased from "
                            f"{early_mean:.4g} to {late_mean:.4g} "
                            f"(+{pct:.1f}%)"
                        )

            # --------------------------------------------------------
            # Metric checks
            # --------------------------------------------------------

            elif _looks_like_metric(var_name):

                metric_window = max(
                    8,
                    int(th["stagnation_window"])
                )

                if len(hist_values) >= metric_window:

                    values = hist_values[-metric_window:]

                    quarter = max(
                        2,
                        len(values) // 4
                    )

                    early = values[:quarter]
                    late = values[-quarter:]

                    early_mean = sum(early) / len(early)
                    late_mean = sum(late) / len(late)

                    denom = max(abs(early_mean), 1e-12)

                    if (
                        abs(late_mean - early_mean)
                        / denom
                        < th["stagnation_frac"]
                    ):
                        reasons.append(
                            f"'{var_name}' has barely moved over the last "
                            f"{metric_window} observations "
                            f"({early_mean:.4g} → {late_mean:.4g})"
                        )

        # ------------------------------------------------------------
        # 2. Cross-variable train/validation divergence
        #
        # This is stronger than looking at val_loss alone.
        # ------------------------------------------------------------

        train_loss_name, train_loss = _find_history(
            "train_loss",
            "loss",
        )

        val_loss_name, val_loss = _find_history(
            "val_loss",
            "validation_loss",
            "valid_loss",
            "test_loss",
        )

        if (
            train_loss
            and val_loss
            and len(train_loss) >= 5
            and len(val_loss) >= 5
        ):
            n = min(
                len(train_loss),
                len(val_loss),
                8,
            )

            train_recent = train_loss[-n:]
            val_recent = val_loss[-n:]

            train_first = train_recent[0]
            train_last = train_recent[-1]

            val_first = val_recent[0]
            val_last = val_recent[-1]

            train_change = train_last - train_first
            val_change = val_last - val_first

            train_relative = (
                train_change / abs(train_first)
                if train_first != 0
                else 0.0
            )

            val_relative = (
                val_change / abs(val_first)
                if val_first != 0
                else 0.0
            )

            # Validation clearly worsens while training loss is flat
            # or improving.
            if (
                val_change > 0
                and val_relative >= max(
                    0.05,
                    th["stagnation_frac"]
                )
                and train_relative <= 0.02
            ):
                reasons.append(
                    "train/validation loss divergence detected: "
                    f"{train_loss_name} changed "
                    f"{train_first:.4g} → {train_last:.4g}, while "
                    f"{val_loss_name} changed "
                    f"{val_first:.4g} → {val_last:.4g}. "
                    "Training loss is not translating into improving "
                    "validation performance."
                )

        # ------------------------------------------------------------
        # 2b. The run never learned anything
        #
        # Every check above looks for training going *wrong* -- a spike, a
        # regression, a plateau after progress. A run that is simply not
        # learning (loss flat from the first epoch to the last) trips none
        # of them, which is how a zeroed initializer, dropout(1.0) or a
        # far-too-large learning rate can train to the end unremarked.
        #
        # Only the whole-run shape is used: the loss at the end vs at the
        # start. Improvement of any size keeps this quiet, so a run that
        # simply converges to a poor result is NOT flagged (Pulse has no
        # way to know what result was achievable).
        # ------------------------------------------------------------

        loss_name, loss_hist = _find_history("loss", "train_loss")

        if loss_hist and len(loss_hist) >= 10:
            span = max(2, len(loss_hist) // 5)
            start_mean = sum(loss_hist[:span]) / span
            end_mean = sum(loss_hist[-span:]) / span

            if start_mean > 0 and end_mean >= start_mean * 0.98:
                accuracy_note = ""
                for acc_name in ("accuracy", "val_accuracy", "acc", "categorical_accuracy"):
                    acc_hist = _history(acc_name)
                    if len(acc_hist) >= 10 and max(acc_hist) - min(acc_hist) < 0.01:
                        accuracy_note = (
                            f", and '{acc_name}' never moved from {acc_hist[-1]:.4g}"
                        )
                        break

                reasons.append(
                    f"'{loss_name}' is no better at the end of the run than at the start "
                    f"({start_mean:.4g} → {end_mean:.4g} over {len(loss_hist)} epochs)"
                    f"{accuracy_note} -- the model does not appear to be learning"
                )

        # ------------------------------------------------------------
        # 2c. Suspiciously perfect performance, suspiciously early
        #
        # The mirror image of 2b above: that one flags too little
        # progress; this one flags implausibly complete progress,
        # implausibly fast. Hitting (near-)perfect accuracy within the
        # first couple of observations is a classic symptom of data
        # leakage -- the target leaking into the input features,
        # evaluating on training data, or a train/test split bug -- far
        # more often than it's genuinely instant learning.
        # ------------------------------------------------------------
        for acc_name in ("val_accuracy", "val_acc", "accuracy", "acc", "categorical_accuracy"):
            acc_hist = _history(acc_name)
            if len(acc_hist) >= 2 and len(acc_hist) <= 2 and max(acc_hist) >= 0.999:
                loss_note = ""
                if loss_hist and len(loss_hist) <= 2 and loss_hist[-1] < 0.01:
                    loss_note = f" (loss is also already down to {loss_hist[-1]:.4g})"
                reasons.append(
                    f"'{acc_name}' already hit {max(acc_hist):.4g} within the first "
                    f"{len(acc_hist)} observation(s){loss_note} -- (near-)perfect accuracy this "
                    "early is a classic symptom of data leakage (the target leaking into the "
                    "input features, evaluating on training data, or a train/test split bug) "
                    "rather than of genuinely fast learning. Worth double-checking the data "
                    "pipeline before trusting this result."
                )
                break

        # ------------------------------------------------------------
        # 3. Matrix/tensor numerical failures
        # ------------------------------------------------------------

        for sub_name, entry in self._matrix_cache.items():

            stats = entry.get("stats", {})

            nan_count = stats.get("nan", 0) or 0
            inf_count = stats.get("inf", 0) or 0

            if nan_count or inf_count:
                reasons.append(
                    f"'{sub_name}' has nan={nan_count} inf={inf_count}"
                )

        # ------------------------------------------------------------
        # 4. Deduplicate reasons
        #
        # The same underlying problem can be detected by several checks.
        # Keep the first occurrence so the agent receives a clean signal.
        # ------------------------------------------------------------

        deduped = []

        seen = set()

        for reason in reasons:
            normalized = reason.strip().lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            deduped.append(reason)

        result = "; ".join(deduped) if deduped else None

        # ------------------------------------------------------------
        # Debugging
        # ------------------------------------------------------------

        if _PULSE_LOGGING:
            try:
                history_lengths = {
                    k: len(v)
                    for k, v in self.scalar_histories.items()
                }

                _pulse_log(
                    f"DETECTOR result={result!r} "
                    f"tracked={self.tracked_vars!r} "
                    f"histories={history_lengths!r}",
                )
            except Exception:
                pass

        return result

    _START_PRIME_PROMPT = (
        "[Automatic start-of-run check -- sent once, automatically, before the first training step, "
        "so this is your only chance to set these from the code alone, before any real data exists] "
        "Look at the training code and the tracked variables above. Four things:\n"
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
        "damage.\n"
        "4. Pulse will periodically check back in while training runs: an agent reads the code, "
        "the metric history and the tracked values, investigates with tools (grep, view, a shell), "
        "and looks for instability AND for bugs that let training look healthy while the result is "
        "wrong (leakage, validation that is not held out, misaligned labels, the wrong output "
        "activation or loss, bad scaling, absurd hyperparameters). Decide how long until the FIRST "
        "check-in, based on how quickly problems would show in this specific run: a short, "
        "fast-iterating script warrants just a couple of minutes; a long, slow run much longer. "
        "Anywhere from 2 to 60 minutes.\n"
        "5. Leave that first check-in a note. You have the code now and it will be busy with a "
        "live run: say what in this code deserves a closer look once real numbers exist, and why "
        "-- a line that looks wrong, an assumption to verify, which metric would expose it. Be "
        "specific (file:line, variable names). 'none' only if the code gives you nothing to "
        "suspect.\n\n"
        "Reply with ONLY these five lines, in exactly this format, and nothing else -- no diagnosis, "
        "no prose:\n"
        "SENSITIVITY: <0.0-1.0, a preset (loose/medium/tight), or 'spike|plateau|oscillation <value|auto>'>\n"
        "NORMAL_START: <comma-separated var=value pairs for loss-like tracked variables you can justify, or 'none'>\n"
        "GPUTRACK: <comma-separated variable names to track closely from the start, or 'none'>\n"
        "NEXTCHECK: <minutes (2-60) until your first periodic check-in>\n"
        "CHECKNOTE: <your note to the first check-in, or 'none'>"
    )

    def _mllint_auto_fix(self, findings: List[tuple]) -> bool:
        """Attempt deterministic, agent-free AST fixes for the clearest
        anti-patterns -- no LLM needed for e.g. replacing 'accuracy' with
        'mae' in a regression compile() call. Returns True if any file was
        patched so the caller knows to record it in the fix log.

        Falls back to the agent pipeline (ask_agent with an explicit
        'fix it' instruction) for anything not covered here, or when the
        deterministic patch fails. Either way, this is a genuine auto-
        intervention that applies the fix -- not just a diagnosis prompt
        that asks the model to 'confirm and propose' something.
        """
        patched_any = False

        # Group findings by the type of pattern so we know which
        # deterministic paths are safe to apply.
        for label, lineno, msg in findings:
            # Find the actual file path this label corresponds to.
            path = None
            for lbl, fpath, _text, _tree in self._iter_ast_trees():
                if lbl == label:
                    path = fpath
                    break
            if path is None:
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
            except OSError:
                continue

            new_src = src
            patched_this = False

            # Pattern 1: regression loss + accuracy metric in
            # model.compile(). Replace accuracy-type metrics with 'mae'
            # which is appropriate for any regression loss.
            if "regression loss" in msg and "accuracy" in msg:
                import re
                # Match metrics=[...] or metrics='accuracy' inside
                # compile() -- replace every accuracy-type name with 'mae'
                def replace_metric(m):
                    inner = m.group(1)
                    for acc_name in sorted(_MLLINT_ACCURACY_METRICS, key=len, reverse=True):
                        inner = re.sub(
                            r"""(['"])""" + re.escape(acc_name) + r"""(['"])""",
                            r"\1mae\2", inner, flags=re.IGNORECASE
                        )
                    return f"metrics={inner}"

                new_src = re.sub(
                    r"metrics=(\[[^\]]*\]|'[^']*'|\"[^\"]*\")",
                    replace_metric, src
                )
                patched_this = new_src != src

            # Pattern 1b: classification loss + regression-only metric --
            # the inverse of Pattern 1. Replace regression-metric names
            # with 'accuracy', appropriate for any classification loss.
            elif "classification loss" in msg and "regression-only metric" in msg:
                import re
                def replace_metric_inverse(m):
                    inner = m.group(1)
                    for reg_name in sorted(_MLLINT_REGRESSION_METRICS, key=len, reverse=True):
                        inner = re.sub(
                            r"""(['"])""" + re.escape(reg_name) + r"""(['"])""",
                            r"\1accuracy\2", inner, flags=re.IGNORECASE
                        )
                    return f"metrics={inner}"

                new_src = re.sub(
                    r"metrics=(\[[^\]]*\]|'[^']*'|\"[^\"]*\")",
                    replace_metric_inverse, src
                )
                patched_this = new_src != src

            # Pattern 2: double-softmax -- remove Softmax() from the
            # model definition (the model's final layer) when it's
            # immediately paired with CrossEntropyLoss.
            elif "double softmax" in msg.lower() or "double-softmax" in msg.lower():
                import re
                # Remove Softmax() / nn.Softmax() as a standalone layer
                new_src = re.sub(
                    r"\bmodel\.add\(.*?[Ss]oftmax\s*\(.*?\)\s*\)\s*\n",
                    "", src
                )
                patched_this = new_src != src

            if not patched_this:
                continue

            # Run lint gate before writing -- same check as _apply_code_fix.
            lint_ok, lint_msgs = self._lint_check(new_src, path)
            if not lint_ok:
                cprint(
                    f"[Pulse] MLLINT auto-fix for '{label}' failed lint check "
                    f"(won't write): {'; '.join(lint_msgs)}", color=_YELLOW
                )
                continue

            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_src)
            except OSError as exc:
                cprint(f"[Pulse] MLLINT auto-fix couldn't write '{path}': {exc}", color=_YELLOW)
                continue

            self._append_fixlog(
                path, src, new_src,
                f"Automatic MLLINT fix (start-of-run): {msg[:200]}"
            )
            _agent_log_event(f"START-OF-RUN LINT FIX written to {os.path.basename(path)}", msg[:300])
            cprint(
                f"[Pulse] ✓ Auto-fixed '{label}' at line {lineno}: {msg[:120]}",
                color=_GREEN
            )
            patched_any = True

        return patched_any

    def _resumed_from_restart(self) -> bool:
        """This process was launched by a fix-triggered restart."""
        return bool(getattr(self, "_resumed_after_restart", False)
                    or os.environ.get(_RESTART_CHILD_ENV) == "1")

    def _report_mllint_clean(self) -> None:
        """The clean verdict, once. _prime_at_start and _prime_with_agent_if_needed both scan
        the same code (the second exists to start the agent fix pipeline once an agent is
        configured) and each used to announce "no issues" -- twice on every clean script."""
        if getattr(self, "_mllint_clean_reported", False):
            return
        self._mllint_clean_reported = True
        cprint("[Pulse] ✓ Start-of-run ML anti-pattern check: no issues found in the code.", color=_YELLOW)

    def _prime_at_start(self) -> None:
        """Runs once, automatically, before the very first training step.

        (0) Static MLLINT pass -- pure AST, no agent needed. For any
        finding that has a deterministic fix (regression+accuracy mismatch,
        double-softmax, ...) the fix is applied immediately to the file,
        the same as any other Pulse auto-intervention, before training has
        taken a single step. For findings without a deterministic path,
        the agent pipeline is triggered with an explicit 'diagnose AND fix'
        instruction -- not 'confirm and propose'. Either way, a finding
        always produces an actual fix attempt, not a suggestion.

        (1) If an agent is configured: asks it to set sensitivity, seed a
        starting-value baseline (NORMAL_START:) and flag GPU-track
        variables (GPUTRACK:) from code alone.
        """
        self._poll_start_prime()
        if self._start_primed:
            return
        self._start_primed = True

        mllint_findings = []
        if self.code_text:
            try:
                mllint_findings = _mllint_scan(self._iter_ast_trees())
            except Exception:
                mllint_findings = []
            if not mllint_findings:
                self._report_mllint_clean()

        if mllint_findings:
            lines = [f"  {label}:{lineno}: {msg}" for label, lineno, msg in mllint_findings]
            cprint(
                f"\n[Pulse] ⚠ Start-of-run ML anti-pattern check found "
                f"{len(mllint_findings)} problem(s) -- auto-fixing before training starts:\n"
                + "\n\n".join(lines) + "\n",
                color=_RED,
            )

            # Deterministic fix first (no agent, no LLM cost, zero latency).
            fixed_deterministically = self._mllint_auto_fix(mllint_findings)

            # Always trigger the full agent fix pipeline too -- for
            # patterns without a deterministic path the agent IS the fix;
            # for ones that were patched it verifies correctness and
            # catches anything the regex missed.
            if self.agent_provider and self.agent_key and self.code_text:
                finding_summary = "; ".join(
                    f"{label}:{lineno}: {msg}" for label, lineno, msg in mllint_findings[:3]
                )
                fix_note = (
                    " Pulse has already applied a deterministic mechanical fix to the file; "
                    "verify it is correct and fix anything remaining."
                    if fixed_deterministically else
                    " Pulse could not apply a deterministic fix -- diagnose AND fix this now."
                )
                cprint(
                    "[Pulse] Auto-intervention: agent fixing "
                    f"({'verifying deterministic patch' if fixed_deterministically else 'no deterministic fix available'})...",
                    color=_RED,
                )
                self.ask_agent(
                    f"Pulse's automatic start-of-run ML anti-pattern check found: {finding_summary}."
                    f"{fix_note} Use the full fix pipeline (diagnose, write a code fix, apply it).",
                    include_code=True,
                )
            elif not fixed_deterministically:
                cprint(
                    "[Pulse] ⚠ No agent configured -- cannot auto-fix. "
                    "Please fix it manually before continuing.",
                    color=_RED,
                )

        # Sensitivity / baseline / GPU-track priming (independent of MLLINT).
        if not self.agent_provider or not self.agent_key or not self.code_text:
            return
        if self._resumed_from_restart():
            # A restart re-runs the same script with the same settings; the
            # first run already answered this. Paying for it again on every
            # restart was 154 calls across a 36-run benchmark and changed
            # nothing.
            return
        self._start_start_prime(deferred=False)

    def _start_start_prime(self, deferred: bool) -> None:
        """Ask the agent for the start-of-run directives (sensitivity, baseline, GPU-track,
        first check-in). The snapshot is built here, on the training thread, from CPU-side
        values only; the wait for the answer happens on a worker thread, so training does not
        sit idle for the length of a model round trip. The answer is applied by
        _finish_start_prime, on the training thread, the next time update() runs after it lands."""
        try:
            context = self._build_agent_context(include_code=True)
        except AgentRequestFailed:
            return
        prompt = f"{context}\n\n{self._START_PRIME_PROMPT}"
        if _async_model_calls_enabled():
            self._start_prime_call = _BackgroundModelCall(
                lambda: self._call_model(prompt, max_tokens=_AGENT_MAX_TOKENS, purpose="start-of-run check"),
                deferred=deferred)
            return
        try:
            answer, error = self._call_model(prompt, max_tokens=_AGENT_MAX_TOKENS, purpose="start-of-run check"), None
        except AgentRequestFailed as exc:
            answer, error = None, exc
        self._finish_start_prime(answer, error, deferred)

    def _poll_start_prime(self) -> None:
        """Apply a start-of-run answer that finished while training was running."""
        call = self._start_prime_call
        if call is not None and call.done:
            self._start_prime_call = None
            self._finish_start_prime(call.result, call.error, getattr(call, "deferred", False))

    def _finish_start_prime(self, answer, error, deferred: bool) -> None:
        if error is not None:
            expected = isinstance(error, AgentRequestFailed)
            if not deferred:
                if expected:
                    cprint(f"[Pulse] ⚠ Start-of-run sensitivity check skipped (agent request failed: {error})", color=_YELLOW)
                # Same as before: whatever went wrong with the first attempt, the deferred one
                # still got its try (it never depended on the first having worked).
                if not self._start_prime_retried and self.code_text:
                    self._start_prime_retried = True
                    self._start_start_prime(deferred=True)
            if not expected:
                raise error          # a bug, not an outage: surface it as it always was
            return
        self._start_prime_answered = True
        _, _calc, _promote, gputrack_names, _gpuu, sensitivity_args, normal_start_args, _grep, _view = self._extract_directives(answer)
        summary = self._apply_directives([], [], gputrack_names, None, sensitivity_args, normal_start_args)
        if summary:
            label = "Start-of-run check (deferred, agent now available)" if deferred else "Start-of-run check"
            cprint(f"[Pulse] {label}: {summary}", color=_YELLOW)
        note_m = self._CHECKNOTE_RE.search(answer or "")
        note = note_m.group(1).strip() if note_m else ""
        if note and note.lower().rstrip(".") not in ("none", "n/a"):
            self._checkin_note = note
            cprint(f"[Pulse] Note for the first check-in: {note}", color=_YELLOW)
            _agent_log_event("NOTE FOR THE FIRST CHECK-IN", note)
        initial_minutes = self._parse_nextcheck_minutes(answer)
        if initial_minutes is not None:
            _agent_log_event(f"FIRST CHECK-IN SCHEDULED in {initial_minutes:g} min")
            self.checkin_interval = initial_minutes * 60.0
            self._last_checkin = time.monotonic()
            if deferred:
                cprint(
                    f"[Pulse] Agent scheduled its first check-in for {initial_minutes:g} "
                    "minutes from now.",
                    color=_YELLOW,
                )
            else:
                cprint(
                    f"[Pulse] Agent scheduled its first check-in for {initial_minutes:g} minutes "
                    "from now.",
                    color=_YELLOW,
                )

    def _prime_with_agent_if_needed(self) -> None:
        """Called every update() once an agent is confirmed available.
        Handles the case where the user configured the agent AFTER the
        first update() call (the normal interactive flow) -- by which
        time _prime_at_start had already run without an agent and couldn't
        fix anything. Runs exactly once, guarded by _start_primed_with_agent.
        """
        if self._start_primed_with_agent:
            return
        if not self.agent_provider or not self.agent_key:
            return
        self._start_primed_with_agent = True

        # Re-run MLLINT now that we have an agent. We already ran it in
        # _prime_at_start (without an agent), applied any deterministic
        # fixes, and printed the findings -- but couldn't kick off the
        # agent fix pipeline then. Do it now.
        mllint_findings = []
        if self.code_text:
            try:
                mllint_findings = _mllint_scan(self._iter_ast_trees())
            except Exception:
                mllint_findings = []
            if not mllint_findings:
                self._report_mllint_clean()

        if mllint_findings:
            fixed_deterministically = self._mllint_auto_fix(mllint_findings)
            finding_summary = "; ".join(
                f"{label}:{lineno}: {msg}" for label, lineno, msg in mllint_findings[:3]
            )
            fix_note = (
                " Pulse has already applied a deterministic mechanical fix; verify it and fix anything remaining."
                if fixed_deterministically else
                " Pulse could not apply a deterministic fix -- diagnose AND fix this now."
            )
            cprint(
                f"\n[Pulse] ⚠ Re-running ML anti-pattern fix now that agent is configured "
                f"({'verifying deterministic patch' if fixed_deterministically else 'agent fix required'})...",
                color=_RED,
            )
            self.ask_agent(
                f"Pulse's start-of-run ML anti-pattern check found: {finding_summary}."
                f"{fix_note} Use the full fix pipeline (diagnose, write a code fix, apply it).",
                include_code=True,
            )

        # Also run sensitivity/baseline priming now that we have an agent,
        # only if the first call didn't get to do it. (It used to run unconditionally, so a
        # normal fresh run -- agent already configured before the first step -- made the same
        # model call twice before step 1, and a restarted run made one that _prime_at_start
        # deliberately skips.)
        if (not mllint_findings and self.code_text
                and not self._start_prime_answered
                and self._start_prime_call is None
                and not self._resumed_from_restart()):
            self._start_start_prime(deferred=True)

        # Drain any problems that were detected before the agent was
        # configured -- these were queued rather than dropped so they
        # don't silently vanish just because the user hadn't run /agent
        # yet when they fired.
        queued = getattr(self, "_queued_interventions", [])
        if queued:
            self._queued_interventions = []
            for problem in queued:
                cprint(
                    f"\n[Pulse] ⚠ Diagnosing queued auto-intervention (detected earlier, "
                    f"agent now available): {problem}",
                    color=_RED,
                )
                self.ask_agent(
                    f"Pulse detected this problem during training (before the agent was configured): {problem}\n"
                    "Please diagnose the root cause and fix it.",
                    include_code=bool(self.code_text),
                )
    def _record_keras_logs(self, logs=None, epoch=None):
        """
        Capture metrics emitted by Keras callbacks.

        Keras does not expose loss/validation loss as normal Python locals in
        the user's training frame. Instead, they arrive through callback `logs`.

        Epoch metrics are kept separately from normal scalar histories so that
        the auto-intervention detector can reason about actual epoch-to-epoch
        training behavior rather than a mixture of batch and epoch values.
        """
        if not logs:
            return

        if not hasattr(self, "epoch_scalar_histories"):
            self.epoch_scalar_histories = {}

        recorded = {}

        for name, value in logs.items():
            if value is None:
                continue

            try:
                # TensorFlow / NumPy scalar -> Python float
                if hasattr(value, "numpy"):
                    value = value.numpy()

                value = float(value)
            except (TypeError, ValueError, OverflowError):
                continue

            # Preserve NaN / inf.
            # The detector needs to see these instead of silently losing them.
            hist = self.epoch_scalar_histories.setdefault(name, [])
            hist.append(value)

            if len(hist) > 2000:
                del hist[:-2000]

            recorded[name] = value

        # Keras calls the training loss simply "loss".
        # Pulse also understands "train_loss", so maintain an alias.
        if "loss" in logs:
            try:
                value = logs["loss"]

                if hasattr(value, "numpy"):
                    value = value.numpy()

                value = float(value)

                hist = self.epoch_scalar_histories.setdefault(
                    "train_loss", []
                )
                hist.append(value)

                if len(hist) > 2000:
                    del hist[:-2000]

                recorded["train_loss"] = value

            except (TypeError, ValueError, OverflowError):
                pass

        # Keep scalar_histories aware of the latest epoch values too.
        # This makes existing CLI commands such as /chart and /tracked
        # continue to work without forcing the detector to use mixed
        # batch/epoch histories.
        for name, value in recorded.items():
            hist = self.scalar_histories.setdefault(name, [])

            if not hist or not _values_equal(hist[-1], value):
                hist.append(value)

            if len(hist) > 2000:
                del hist[:-2000]

        if _PULSE_LOGGING:
            try:
                history_lengths = {
                    k: len(v)
                    for k, v in self.epoch_scalar_histories.items()
                }

                _pulse_log(
                    "KERAS EPOCH LOGS "
                    f"epoch={epoch!r} "
                    f"values={recorded!r} "
                    f"history_lengths={history_lengths!r}",
                )
            except Exception:
                pass


    def _record_keras_batch_logs(self, logs=None):
        """
        Lightweight batch-level Keras metric capture.

        This intentionally does NOT put batch metrics into
        epoch_scalar_histories. Batch loss and epoch loss have different
        semantics and must not be mixed for trend detection.
        """
        if not logs:
            return

        if not hasattr(self, "batch_scalar_histories"):
            self.batch_scalar_histories = {}

        for name, value in logs.items():
            if value is None:
                continue

            try:
                if hasattr(value, "numpy"):
                    value = value.numpy()

                value = float(value)
            except (TypeError, ValueError, OverflowError):
                continue

            hist = self.batch_scalar_histories.setdefault(name, [])
            hist.append(value)

            if len(hist) > 2000:
                del hist[:-2000]


    def _history_for_detector(self, name):
        """
        Return the correct history for auto-diagnosis.

        Loss/validation metrics prefer epoch-level Keras history.
        Other scalar variables continue using normal Pulse histories.
        """
        epoch_histories = getattr(self, "epoch_scalar_histories", {})

        if name in epoch_histories and epoch_histories[name]:
            return epoch_histories[name]

        if name == "train_loss":
            loss_hist = epoch_histories.get("loss")
            if loss_hist:
                return loss_hist

        return self.scalar_histories.get(name, [])


    def update(self, step: Optional[int] = None,
            generate_pdfs: Optional[bool] = None) -> None:
        """Called at every training step/checkpoint."""

        # ------------------------------------------------------------
        # SIGINT / CTRL-C
        # ------------------------------------------------------------
        if self._stop_requested:
            self.continuous = False

        if _PULSE_LOGGING:
            try:
                _pulse_log(
                    "UPDATE ENTER "
                    f"step_arg={step!r} "
                    f"continuous={self.continuous} "
                    f"auto_intervene={self.auto_intervene} "
                    f"tracked={self.tracked_vars!r} "
                    f"watch_locals={sorted(k for k in self.watch_locals if not k.startswith('__'))!r}"
                )

            except Exception as _exc:
                try:
                    _pulse_log(
                        f"UPDATE DEBUG ERROR: "
                        f"{type(_exc).__name__}: {_exc}"
                    )
                except Exception:
                    pass

        # ------------------------------------------------------------
        # FIND LOSS VARIABLE
        # ------------------------------------------------------------
        loss_var = next(
            (
                v for v in self.tracked_vars
                if _looks_like_loss(v)
            ),
            None,
        )

        new_loss_value: Optional[float] = None

        # ------------------------------------------------------------
        # STARTUP
        # ------------------------------------------------------------
        self._prime_at_start()
        self._prime_with_agent_if_needed()

        # ------------------------------------------------------------
        # UPTIME
        # ------------------------------------------------------------
        _update_start_ts = time.monotonic()

        if self._last_update_end_ts is not None:
            gap = _update_start_ts - self._last_update_end_ts

            if gap > 0:
                self._uptime_seconds += gap
                self._cloud_dirty_fields.add("uptime_seconds")

        if (
            self.debug_session_id
            and {
                "uptime_seconds",
                "downtime_seconds",
            } & self._cloud_dirty_fields
            and (
                _update_start_ts - self._last_uptime_flush
            ) >= self.uptime_flush_interval
        ):
            self._last_uptime_flush = _update_start_ts
            self._maybe_flush_cloud(force=True)

        # ------------------------------------------------------------
        # TRY TO GET LOSS FROM NORMAL PYTHON LOCALS
        # ------------------------------------------------------------
        if loss_var is not None and loss_var in self.watch_locals:
            raw = self._cpu_observation(
                loss_var,
                self.watch_locals[loss_var],
            )

            if raw is not None and is_trackable(raw):
                try:
                    if describe_tensor(raw).kind == "scalar":
                        stats = statistics(raw)
                        value = stats.get("mean")

                        if value is not None:
                            new_loss_value = float(value)

                except Exception:
                    new_loss_value = None

        # ------------------------------------------------------------
        # KERAS FALLBACK
        #
        # If loss isn't visible in Python locals, use the epoch history
        # populated by the Keras callback bridge.
        # ------------------------------------------------------------
        if new_loss_value is None:
            epoch_histories = getattr(
                self,
                "epoch_scalar_histories",
                {},
            )

            keras_loss = epoch_histories.get("loss")

            if keras_loss:
                try:
                    new_loss_value = float(keras_loss[-1])
                except (TypeError, ValueError):
                    new_loss_value = None

        # ------------------------------------------------------------
        # STEP COUNTER
        # ------------------------------------------------------------
        if step is not None:
            self.step = step

        elif loss_var is None and new_loss_value is None:
            self.step += 1

        elif not _values_equal(
            self._last_loss_value,
            new_loss_value,
        ):
            self.step += 1

        if new_loss_value is not None:
            self._last_loss_value = new_loss_value

        # ------------------------------------------------------------
        # PDF / GPU / MATRIX PROBES
        # ------------------------------------------------------------
        want_pdfs = (
            self.generate_pdfs
            if generate_pdfs is None
            else generate_pdfs
        )

        now = time.monotonic()

        gpu_ready = (
            now - getattr(
                self,
                "_last_gpu_probe",
                0.0,
            )
        ) >= getattr(
            self,
            "gpu_probe_interval",
            1.0,
        )

        probe_track = (
            not self._matrix_cache
            or (
                gpu_ready
                and (
                    now - self._last_matrix_probe
                ) >= self.matrix_probe_interval
            )
            or any(
                v not in self._matrix_cached_vars
                for v in self.tracked_vars
                if self._state_of(v) == "track"
            )
        )

        probe_lotrack = (
            (
                gpu_ready
                and (
                    now - self._last_lotrack_probe
                ) >= self.lotrack_probe_interval
            )
            or any(
                v not in self._matrix_cached_vars
                for v in self.tracked_vars
                if self._state_of(v) == "lotrack"
            )
        )

        probe_matrices = probe_track or probe_lotrack

        if probe_matrices and self._matrix_cache:
            self._last_gpu_probe = now

        scalar_lines: List[
            tuple[str, Optional[float]]
        ] = []

        matrix_lines: List[
            tuple[str, Dict[str, Any], Any, str]
        ] = []

        any_scalar_changed = False

        # ------------------------------------------------------------
        # NORMAL PULSE VARIABLE TRACKING
        # ------------------------------------------------------------
        for var_name in self.tracked_vars:

            if var_name not in self.watch_locals:
                continue

            raw_local = self.watch_locals[var_name]

            orig_val = self._cpu_observation(
                var_name,
                raw_local,
            )

            var_state = self._state_of(var_name)

            if (
                orig_val is None
                and raw_local is not None
                and _pulse_is_accelerator_value(raw_local)
            ):
                if var_name not in self._cpu_only_warned:
                    cprint(
                        f"  • '{var_name}' is accelerator-resident; "
                        f"Pulse will not touch it. Maintain a CPU mirror "
                        f"named '{var_name}_cpu' (or '{var_name}_np') "
                        f"to visualize it.",
                        color=_YELLOW,
                    )

                    self._cpu_only_warned.add(var_name)

                continue

            due_this_var = (
                probe_track
                if var_state == "track"
                else probe_lotrack
            )

            if _PULSE_LOGGING:
                try:
                    _pulse_log(
                        f"OBSERVE name={var_name!r} "
                        f"state={var_state!r} "
                        f"raw_type={type(orig_val).__name__} "
                        f"raw_none={orig_val is None} "
                        f"history_len="
                        f"{len(self.scalar_histories.get(var_name, []))}",
                    )
                except Exception:
                    pass

            # --------------------------------------------------------
            # KERAS HISTORY FALLBACK FOR INDIVIDUAL METRICS
            # --------------------------------------------------------
            if orig_val is None:
                keras_val = _try_keras_history(
                    var_name,
                    self.watch_locals,
                )

                if keras_val is not None:
                    orig_val = keras_val

            # --------------------------------------------------------
            # UNREADABLE VARIABLE
            # --------------------------------------------------------
            if orig_val is None:
                hist = self.scalar_histories.setdefault(
                    var_name,
                    [],
                )

                if (
                    not hist
                    or not _values_equal(hist[-1], None)
                ):
                    hist.append(None)
                    any_scalar_changed = True

                scalar_lines.append(
                    (var_name, None)
                )

                continue

            if not is_trackable(orig_val):
                continue

            try:
                kind = describe_tensor(orig_val).kind
            except Exception:
                kind = None

            # --------------------------------------------------------
            # SCALAR
            # --------------------------------------------------------
            if kind == "scalar":
                try:
                    stats = statistics(orig_val)
                    scalar_val = float(
                        stats.get("mean")
                    )

                except Exception as exc:
                    hist = self.scalar_histories.setdefault(
                        var_name,
                        [],
                    )

                    if (
                        not hist
                        or not _values_equal(
                            hist[-1],
                            None,
                        )
                    ):
                        hist.append(None)
                        any_scalar_changed = True

                    scalar_lines.append(
                        (var_name, None)
                    )

                    cprint(
                        f"  ⚠ '{var_name}' could not be read "
                        f"this step: "
                        f"{type(exc).__name__}: {exc}",
                        color=_RED,
                    )

                    continue

                hist = self.scalar_histories.setdefault(
                    var_name,
                    [],
                )

                changed = (
                    not hist
                    or not _values_equal(
                        hist[-1],
                        scalar_val,
                    )
                )

                if changed:
                    hist.append(scalar_val)
                    any_scalar_changed = True

                if len(hist) > 2000:
                    del hist[:-2000]

                scalar_lines.append(
                    (var_name, scalar_val)
                )

                continue

            # --------------------------------------------------------
            # MATRIX / TENSOR
            # --------------------------------------------------------
            if not due_this_var:
                continue

            for sub_name, val in self._yield_slices(
                var_name,
                orig_val,
            ):
                if val is None:
                    hist = self.scalar_histories.setdefault(
                        sub_name,
                        [],
                    )

                    if (
                        not hist
                        or not _values_equal(
                            hist[-1],
                            None,
                        )
                    ):
                        hist.append(None)
                        any_scalar_changed = True

                    scalar_lines.append(
                        (sub_name, None)
                    )

                    continue

                try:
                    stats = statistics(val)

                except Exception as exc:
                    cprint(
                        f"  ⚠ '{sub_name}' could not be read "
                        f"this step: "
                        f"{type(exc).__name__}: {exc}",
                        color=_RED,
                    )
                    continue

                if stats.get("kind") == "scalar":
                    try:
                        scalar_val = float(
                            stats.get("mean")
                        )

                    except Exception:
                        hist = self.scalar_histories.setdefault(
                            sub_name,
                            [],
                        )

                        if (
                            not hist
                            or not _values_equal(
                                hist[-1],
                                None,
                            )
                        ):
                            hist.append(None)
                            any_scalar_changed = True

                        scalar_lines.append(
                            (sub_name, None)
                        )

                        continue

                    hist = self.scalar_histories.setdefault(
                        sub_name,
                        [],
                    )

                    changed = (
                        not hist
                        or not _values_equal(
                            hist[-1],
                            scalar_val,
                        )
                    )

                    if changed:
                        hist.append(scalar_val)
                        any_scalar_changed = True

                    if len(hist) > 2000:
                        del hist[:-2000]

                    scalar_lines.append(
                        (sub_name, scalar_val)
                    )

                else:
                    self._matrix_cache[sub_name] = {
                        "base_name": var_name,
                        "stats": stats,
                    }

                    matrix_lines.append(
                        (
                            sub_name,
                            stats,
                            val,
                            var_state,
                        )
                    )

            self._matrix_cached_vars.add(var_name)

        # ------------------------------------------------------------
        # PROBE TIMESTAMPS
        # ------------------------------------------------------------
        if probe_track:
            self._last_matrix_probe = now

        if probe_lotrack:
            self._last_lotrack_probe = now

        # ------------------------------------------------------------
        # TERMINAL DISPLAY
        # ------------------------------------------------------------
        def _is_wrong(v):
            return (
                v is None
                or (
                    isinstance(v, (int, float))
                    and not math.isfinite(v)
                )
            )

        wrong_scalars = [
            (n, v)
            for n, v in scalar_lines
            if _is_wrong(v)
        ]

        track_matrix_lines = (
            [
                t for t in matrix_lines
                if t[3] != "lotrack"
            ]
            if probe_matrices
            else []
        )

        should_redraw = bool(
            wrong_scalars
            or track_matrix_lines
        )

        if should_redraw:
            sys.stdout.write(
                "\033[2J\033[H"
            )
            sys.stdout.flush()

            cprint(
                f"--- Pulse Live Debugger | Step {self.step} ---"
            )

            for sub_name, scalar_val in wrong_scalars:
                hist = self.scalar_histories.get(
                    sub_name,
                    [],
                )

                if scalar_val is None:
                    cprint(
                        f"  • {sub_name}: NoneType "
                        f"⚠ (unreadable this step)",
                        color=_RED,
                    )
                else:
                    cprint(
                        f"  • {sub_name}: "
                        f"{scalar_val:.6g} ⚠",
                        color=_RED,
                    )

                self._print_ascii_chart(
                    sub_name,
                    hist,
                )

            if track_matrix_lines:

                def _fmt(v):
                    try:
                        return f"{v:.4f}"
                    except (
                        TypeError,
                        ValueError,
                    ):
                        return "n/a"

                for (
                    sub_name,
                    stats,
                    val,
                    var_state,
                ) in track_matrix_lines:

                    flag = ""

                    if (
                        stats.get("nan")
                        or stats.get("inf")
                    ):
                        flag = (
                            f"  ⚠ nan={stats.get('nan')} "
                            f"inf={stats.get('inf')}"
                        )

                    mean_v = stats.get("mean")
                    min_v = stats.get("min")
                    max_v = stats.get("max")

                    print(
                        f"  • Tagging '{sub_name}' "
                        f"[{stats.get('backend')} "
                        f"{stats.get('kind')} "
                        f"{stats.get('shape')}] "
                        f"| mean={_fmt(mean_v)} "
                        f"min={_fmt(min_v)} "
                        f"max={_fmt(max_v)}"
                        f"{flag}"
                    )

                    if (
                        want_pdfs
                        and val is not None
                    ):
                        safe_name = (
                            sub_name
                            .replace("[", "_")
                            .replace("]", "")
                            .replace(",", "_")
                        )

                        try:
                            pdf_path = generate_heatmap_pdf(
                                safe_name,
                                val,
                                self.step,
                                output_dir=self.pdf_dir,
                            )

                            print(
                                f"    ↳ saved snapshot: "
                                f"{pdf_path}"
                            )

                        except Exception as exc:
                            cprint(
                                f"    ↳ ⚠ failed to save PDF "
                                f"snapshot for '{sub_name}': {exc}",
                                color=_RED,
                            )

                if self._matrix_cache:
                    print(
                        f"  [matrices cached: "
                        f"{len(self._matrix_cache)} | "
                        f"next full probe ≤ "
                        f"{self.matrix_probe_interval:g}s | "
                        f"next lotrack probe ≤ "
                        f"{self.lotrack_probe_interval:g}s]"
                    )

        # ------------------------------------------------------------
        # TELEMETRY / GPU
        # ------------------------------------------------------------
        self._maybe_collect_telemetry(
            scalar_lines,
            matrix_lines,
        )

        self._maybe_periodic_checkin()

        # ------------------------------------------------------------
        # AUTO-INTERVENTION
        #
        # This is completely independent of CTRL-C.
        # The Keras callback has already populated epoch histories,
        # so the detector can now actually see loss / val_loss.
        # ------------------------------------------------------------
        if self.auto_intervene:

            problem = self._check_for_trouble()

            if _PULSE_LOGGING:
                try:
                    detector_histories = {
                        k: len(v)
                        for k, v in getattr(
                            self,
                            "epoch_scalar_histories",
                            {},
                        ).items()
                    }

                    _pulse_log(
                        "DETECTOR "
                        f"result={problem!r} "
                        f"epoch_histories="
                        f"{detector_histories!r}",
                    )

                except Exception:
                    pass

            self._escalate_training_problem(problem)

        # ------------------------------------------------------------
        # UPDATE FINISHED
        # ------------------------------------------------------------
        self._last_update_end_ts = time.monotonic()

        if self._stop_requested:
            self.continuous = False
            self._stop_requested = False

        if self.continuous:
            return

        # ------------------------------------------------------------
        # INTERACTIVE PROMPT
        # ------------------------------------------------------------
        while True:
            try:
                _flush_stdin()

                cmd = input(
                    _highlight_pulse(
                        "\nPulse "
                        "[Enter=step, /c=continuous, "
                        "/help, or ask AI] > "
                    )
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):
                cprint("\nExiting Pulse...")

                if (
                    self.original_sigint
                    and callable(self.original_sigint)
                ):
                    signal.signal(
                        signal.SIGINT,
                        self.original_sigint,
                    )

                raise KeyboardInterrupt

            if not cmd:
                break

            cmd_lower = cmd.lower()

            if cmd_lower in (
                "/c",
                "/continue",
                "c",
            ):
                self.continuous = True
                break

            if cmd_lower in (
                "/help",
                "/h",
                "?",
                "/commands",
            ):
                self._cmd_help(cmd)
                continue

            if cmd_lower.startswith("/add "):
                self._cmd_add(cmd[5:].strip())
                continue

            if cmd_lower.startswith("/track "):
                self._cmd_track(cmd[7:].strip())
                continue

            if cmd_lower.startswith("/lotrack "):
                self._cmd_lotrack(cmd[9:].strip())
                continue

            if cmd_lower.startswith("/gputrack "):
                self._cmd_gputrack(cmd[10:].strip())
                continue

            if cmd_lower.startswith("/gpuuntrack "):
                self._cmd_gpuuntrack(cmd[12:].strip())
                continue

            if cmd_lower.startswith("/autofix"):
                self._cmd_autofix(
                    cmd[8:].strip()
                )
                continue

            if cmd_lower.startswith("/sensitivity"):
                self._cmd_sensitivity(
                    cmd[len("/sensitivity"):].strip()
                )
                continue

            if cmd_lower.startswith("/telemetry"):
                self._cmd_telemetry(
                    cmd[len("/telemetry"):].strip()
                )
                continue

            if cmd_lower.startswith("/deletepdf "):
                self._cmd_delete_pdfs(
                    cmd[11:].strip()
                )
                continue

            if cmd_lower.startswith("/delete "):
                self._cmd_delete(
                    cmd[8:].strip()
                )
                continue

            if cmd_lower == "/agent":
                self._select_agent_provider_and_key(
                    initial=False
                )
                continue

            if cmd_lower == "/cloud":
                self._print_cloud_status()
                continue

            if cmd_lower == "/cloud flush":
                self._maybe_flush_cloud(
                    force=True
                )
                cprint(
                    "[Pulse] Cloud sync flushed."
                )
                continue

            if cmd_lower.startswith("/repo"):
                self._cmd_repo(
                    cmd[5:].strip()
                )
                continue

            if cmd_lower == "/vars":
                self._print_variable_summary()
                continue

            if (
                cmd_lower == "/chart"
                or cmd_lower.startswith("/chart ")
            ):
                self._cmd_chart(
                    cmd[6:].strip()
                )
                continue

            if cmd_lower == "/tracked":
                cfg_strs = [
                    (
                        f"{v}({self.var_configs[v]})"
                        f"[{self._state_of(v)}]"
                    )
                    if v in self.var_configs
                    else (
                        f"{v}"
                        f"[{self._state_of(v)}]"
                    )
                    for v in self.tracked_vars
                ]

                print(
                    "Tracked:",
                    (
                        ", ".join(cfg_strs)
                        if cfg_strs
                        else "(none)"
                    ),
                )

                continue

            if cmd_lower.startswith("/code"):
                self._cmd_code(
                    cmd[5:].strip()
                )
                continue

            if cmd_lower == "/log":
                self._cmd_log("")
                continue

            if cmd_lower.startswith("/admin"):
                self._cmd_admin(
                    cmd[6:].strip()
                )
                continue

            if cmd_lower.startswith("/webhook"):
                self._cmd_webhook(
                    cmd[8:].strip()
                )
                continue

            if cmd_lower.startswith("/password"):
                self._cmd_password(
                    cmd[9:].strip()
                )
                continue

            if cmd_lower.startswith("/recover"):
                self._cmd_recover(
                    cmd[8:].strip()
                )
                continue

            if cmd_lower.startswith("/deleteaccount"):
                self._cmd_deleteaccount(
                    cmd[14:].strip()
                )
                continue

            if cmd_lower == "/logout":
                self._cmd_logout("")
                continue

            if cmd_lower.startswith("/commit"):
                self._cmd_commit(
                    cmd[7:].strip()
                )
                continue

            if cmd_lower.startswith("/revert"):
                self._cmd_revert(
                    cmd[7:].strip()
                )
                continue

            cprint("Pulse AI:")
            self.ask_agent(
                cmd,
                include_code=self.include_code_default,
            )


    # ================================================================
    # KERAS TRACKER / CALLBACK BRIDGE
    # ================================================================

   

    

    def _cmd_chart(self, arg: str) -> None:
        """/chart [var] -- render the ASCII loss/metric curve for any
        tracked scalar on demand, not just when it's already flagged
        unhealthy (_print_ascii_chart below is otherwise only invoked
        automatically for that case, as part of the anomaly redraw).
        Defaults to the most loss-like tracked scalar (see
        LOSS_NAME_HINTS) when no name is given, so plain '/chart' just
        works for the common case of "show me the loss curve"."""
        name = arg.strip()
        if not name:
            candidates = [n for n in self.scalar_histories if any(h in n.lower() for h in LOSS_NAME_HINTS)]
            name = candidates[0] if candidates else next(iter(self.scalar_histories), None)
            if name is None:
                cprint("[Pulse] no scalar history recorded yet.", color=_RED)
                return
        hist = self.scalar_histories.get(name)
        if hist is None:
            matches = [k for k in self.scalar_histories if name.lower() in k.lower()]
            if len(matches) == 1:
                name, hist = matches[0], self.scalar_histories[matches[0]]
        if not hist:
            known = ", ".join(sorted(self.scalar_histories)) or "(none yet)"
            cprint(f"[Pulse] no scalar history recorded yet for '{name}'. Known scalars: {known}", color=_RED)
            return
        cprint(f"--- {name} ({len(hist)} point(s)) ---")
        self._print_ascii_chart(name, hist)

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
