"""
Pulse — a live ML training debugger, GUI or CLI, any backend.

    from pulse import auto_track
    auto_track(train_step)
"""
from .pulse import auto_track, shutdown

__version__ = "0.1.2"
__all__ = ["auto_track", "shutdown"]
