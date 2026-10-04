"""
Pulse — a live ML training debugger, GUI or CLI, any backend.

    from pulse import auto_track
    auto_track(train_step)

Importing the package is cheap on purpose: `from pulse import auto_track` sits at the top
of training scripts, so it runs again in every 'spawn' worker (DataLoader workers on
macOS/Windows, each epoch), which never use Pulse. The debugger itself (pulse.pulse) is
only imported when auto_track() actually starts a session.
"""
import sys

__version__ = "0.1.2"
__all__ = ["auto_track", "shutdown", "check_shape", "check_shapes"]


def _in_multiprocessing_bootstrap(caller=None):
    """True while a multiprocessing child is re-importing the parent's modules.

    A 'spawn'/'forkserver' child imports the main module again (as __mp_main__), and with it
    every module that one imports, before it runs its target. multiprocessing marks that
    phase with current_process()._inheriting. An auto_track() reached then -- at module
    level of the script or of a helper module it imports -- is not a request for a new
    session but the parent's, replayed. (parent_process() is no guide: it is set in a
    forked child too, where a launcher may runpy a real __main__ script that wants Pulse.)
    """
    mp = sys.modules.get("multiprocessing")
    if mp is None:
        return False          # nothing started this process through multiprocessing
    try:
        if getattr(mp.current_process(), "_inheriting", False):
            return True
    except Exception:
        pass
    try:
        return (caller is not None and caller.f_code.co_name == "<module>"
                and caller.f_globals.get("__name__") == "__mp_main__")
    except Exception:
        return False


def auto_track(*args, **kwargs):
    """Start Pulse for this process. See pulse.pulse.auto_track for the arguments."""
    # pulse.pulse.auto_track skips this wrapper's frame when it looks for its caller.
    if _in_multiprocessing_bootstrap(sys._getframe(1)):
        return None
    from .pulse import auto_track as _auto_track
    return _auto_track(*args, **kwargs)


def shutdown():
    """Stop Pulse in this process (a no-op if it was never started)."""
    core = sys.modules.get(__name__ + ".pulse")
    if core is None:
        return None
    return core.shutdown()


def check_shape(value, expected=None, name=None, **kwargs):
    """Describe a value's shape and, optionally, check it against an expectation.

        pulse.check_shape(logits, "(B, 10)")                      # a name like B is any size, but one size
        pulse.check_shape(x, "(B, 3, 224, 224)", raise_on_mismatch=True)

    Returns a report (truthy when the shape matches; print it for the details). See
    pulse.pulse_shapes.check_shape."""
    from .pulse_shapes import check_shape as _check_shape
    return _check_shape(value, expected, name, **kwargs)


def check_shapes(checks, **kwargs):
    """check_shapes({"x": (x, "(B, 784)"), "y": (y, "(B,)")}): every spec is checked against one
    shared symbol table, so each B must be the same size. See pulse.pulse_shapes.check_shapes."""
    from .pulse_shapes import check_shapes as _check_shapes
    return _check_shapes(checks, **kwargs)