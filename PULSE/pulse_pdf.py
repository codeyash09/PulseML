"""
pulse_backend.py
================

Universal backend abstraction for Pulse.

Supported backends
------------------
- NumPy
- CuPy
- PyTorch
- TensorFlow
- JAX

Everything else is treated as a generic Python object.

Pulse should ONLY communicate with this module instead of checking
for torch/cupy/etc directly.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

# ----------------------------------------------------------------------
# Optional imports
# ----------------------------------------------------------------------

HAS_TORCH = False
HAS_TF = False
HAS_CUPY = False
HAS_JAX = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    tf = None

try:
    import cupy
    HAS_CUPY = True
except Exception:
    cupy = None

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    jax = None
    jnp = None


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------

@dataclass
class TensorInfo:
    backend: str
    kind: str
    shape: tuple
    ndim: int
    dtype: str
    device: Optional[str]
    object: Any


# ----------------------------------------------------------------------
# Backend detection
# ----------------------------------------------------------------------

def detect_backend(x):
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return "PyTorch"

    if HAS_TF and tf.is_tensor(x):
        return "TensorFlow"

    if HAS_CUPY and isinstance(x, cupy.ndarray):
        return "CuPy"

    if HAS_JAX:
        try:
            if isinstance(x, jax.Array):
                return "JAX"
        except Exception:
            pass

    if isinstance(x, np.ndarray):
        return "NumPy"

    return "Python"


# ----------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------

def device_of(x):
    backend = detect_backend(x)

    if backend == "PyTorch":
        return str(x.device)

    if backend == "TensorFlow":
        try:
            return x.device
        except Exception:
            return None

    if backend == "CuPy":
        try:
            return f"cuda:{x.device.id}"
        except Exception:
            return "cuda"

    if backend == "JAX":
        try:
            return str(x.device())
        except Exception:
            return None

    return "CPU"


# ----------------------------------------------------------------------
# Shape
# ----------------------------------------------------------------------

def shape_of(x):
    try:
        return tuple(x.shape)
    except Exception:
        return ()


def ndim_of(x):
    return len(shape_of(x))


# ----------------------------------------------------------------------
# Type classification
# ----------------------------------------------------------------------
def scalar_value(x):
    """
    Extract a Python float from any scalar tensor across all supported backends.
    Pulse expects this helper to exist.
    """
    arr = to_numpy(x)
    if arr.shape != () and arr.size != 1:
        raise ValueError(f"scalar_value expected a scalar, got shape {arr.shape}")
    return float(arr)

def tensor_kind(x):
    shape = shape_of(x)

    if shape == ():
        return "scalar"

    if len(shape) == 1:
        return "vector"

    if len(shape) == 2:
        return "matrix"

    return "tensor"


def is_scalar(x):
    return tensor_kind(x) == "scalar"


def is_vector(x):
    return tensor_kind(x) == "vector"


def is_matrix(x):
    return tensor_kind(x) == "matrix"


def is_tensor(x):
    return tensor_kind(x) == "tensor"


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------

def to_numpy(x):
    backend = detect_backend(x)

    if backend == "NumPy":
        return x

    if backend == "PyTorch":
        return x.detach().cpu().numpy()

    if backend == "TensorFlow":
        return x.numpy()

    if backend == "CuPy":
        return cupy.asnumpy(x)

    if backend == "JAX":
        return np.asarray(x)

    if isinstance(x, (int, float, complex, bool)):
        return np.asarray(x)

    try:
        return np.asarray(x)
    except Exception:
        raise TypeError(f"Cannot convert {type(x)} to numpy.")


# ----------------------------------------------------------------------
# Metadata (renamed from inspect to avoid stdlib conflict)
# ----------------------------------------------------------------------

def describe_tensor(x):
    backend = detect_backend(x)
    arr = to_numpy(x)

    return TensorInfo(
        backend=backend,
        kind=tensor_kind(x),
        shape=tuple(arr.shape),
        ndim=arr.ndim,
        dtype=str(arr.dtype),
        device=device_of(x),
        object=x,
    )


# ----------------------------------------------------------------------
# Human-readable labels
# ----------------------------------------------------------------------

def label(x):
    info = describe_tensor(x)
    return f"{info.backend} {info.kind} {info.shape}"


# ----------------------------------------------------------------------
# Supported backends
# ----------------------------------------------------------------------

def available_backends():
    return {
        "NumPy": True,
        "PyTorch": HAS_TORCH,
        "TensorFlow": HAS_TF,
        "CuPy": HAS_CUPY,
        "JAX": HAS_JAX,
    }


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def statistics(x):
    arr = to_numpy(x).astype(np.float64)
    finite = arr[np.isfinite(arr)]

    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "backend": detect_backend(x),
        "kind": tensor_kind(x),
        "device": device_of(x),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "nan": int(np.isnan(arr).sum()),
        "inf": int(np.isinf(arr).sum()),
    }


# ----------------------------------------------------------------------
# Variable filtering
# ----------------------------------------------------------------------

def is_trackable(obj):
    try:
        # If describe_tensor and to_numpy both succeed, we consider it trackable.
        describe_tensor(obj)
        return True
    except Exception:
        return False
