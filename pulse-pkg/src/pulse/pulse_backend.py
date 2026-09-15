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
- SciPy (sparse matrices)
- Pandas (DataFrame / Series)

Everything else is treated as a generic Python object.

Pulse should ONLY communicate with this module instead of checking
for torch/cupy/etc directly.

GPU note: every conversion path below lands on host (CPU) memory before
Pulse does anything else with the data (`.detach().cpu().numpy()` for
PyTorch, `.numpy()` for TensorFlow, `cupy.asnumpy()` for CuPy, etc). Pulse
never issues a CUDA kernel or otherwise touches the GPU beyond the
unavoidable device->host copy needed to read a tracked tensor's value.

scikit-learn note: sklearn doesn't define its own array type -- it reads
and writes plain numpy arrays, and (very commonly) scipy sparse matrices
and pandas DataFrames/Series (e.g. `TfidfVectorizer`/`OneHotEncoder`
output sparse matrices; most preprocessing steps accept/return
DataFrames). Those two are handled explicitly below so a variable holding
sklearn pipeline output is just as trackable as a torch/TF tensor. A
fitted estimator object itself (a `LogisticRegression`, a `Pipeline`, ...)
is intentionally NOT made "trackable" here, the same way an
`nn.Module` isn't -- it's the arrays it produces/consumes and its fitted
array attributes (`.coef_`, `.feature_importances_`, already plain numpy
arrays) that are the actual trackable values, not the estimator wrapper.
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
HAS_SCIPY_SPARSE = False
HAS_PANDAS = False

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

try:
    import scipy.sparse as sp_sparse
    HAS_SCIPY_SPARSE = True
except Exception:
    sp_sparse = None

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    pd = None


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

    # scipy sparse matrices (csr_matrix, csc_matrix, coo_matrix, ...) --
    # the standard output of sklearn's TfidfVectorizer/OneHotEncoder/etc.
    # Checked before the generic numpy/pandas branches since sp.issparse
    # is the canonical, version-stable way to recognize any of scipy's
    # several sparse matrix/array classes.
    if HAS_SCIPY_SPARSE and sp_sparse.issparse(x):
        return "SciPy Sparse"

    # pandas DataFrame/Series -- the standard input/output of most sklearn
    # preprocessing steps (ColumnTransformer, train_test_split, ...).
    if HAS_PANDAS and isinstance(x, (pd.DataFrame, pd.Series)):
        return "Pandas"

    # NOTE: catches BOTH real arrays (np.ndarray) and numpy *scalar*
    # types (np.float32, np.float64, np.int64, np.bool_, ...), whose
    # common base class is np.generic -- np.ndarray alone does not cover
    # them, even though hasattr(x, "shape") is True for both. This matters
    # in practice: np.float32/float64 is exactly what most ML frameworks'
    # logging paths hand back for a scalar metric (e.g. Keras callbacks'
    # `logs` dict, or anything that's already called `.item()`-adjacent
    # code upstream) -- without this, every one of those values used to
    # silently fall through to the generic "Python" label below.
    if isinstance(x, (np.ndarray, np.generic)):
        return "NumPy"

    if isinstance(x, (int, float, bool)):
        return "Python"

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

def has_shape(x):
    """True only for things that genuinely carry a shape/array-like
    interface -- arrays/tensors of any supported backend, or a bare
    Python int/float/bool (which counts as a 0-d scalar). Everything else
    (strings, dicts, arbitrary objects, loop counters that are secretly
    something weirder) returns False so discovery doesn't sweep them up."""
    if isinstance(x, (int, float, bool)) and not isinstance(x, complex):
        return True
    if hasattr(x, "shape"):
        return True
    return False


def shape_of(x):
    if isinstance(x, (int, float, bool)) and not isinstance(x, complex):
        return ()
    try:
        return tuple(x.shape)
    except Exception:
        return None


def ndim_of(x):
    shape = shape_of(x)
    return len(shape) if shape is not None else None


# ----------------------------------------------------------------------
# Type classification
# ----------------------------------------------------------------------

def tensor_kind(x):
    shape = shape_of(x)
    if shape is None:
        return None

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

def _generic_to_numpy(x):
    """Last-resort conversion, used both as the final fallback for
    unrecognized objects AND as the fallback when a backend-specific
    conversion below fails despite detect_backend() guessing that
    backend -- see the comment in to_numpy() for why that matters."""
    if isinstance(x, (int, float, complex, bool)):
        return np.asarray(x)
    return np.asarray(x)


def to_numpy(x):
    """Convert to a NumPy array. For the NumPy backend this returns the
    original array with no copy; for other backends this performs the
    minimum device->host copy required and nothing more.

    Each backend-specific path is now defensive: if the detected backend's
    own conversion method raises (e.g. `.numpy()` on a TensorFlow tensor
    that turns out to be a non-eager/symbolic tensor from some
    tf.function-traced or distribution-strategy code path -- rare, but
    real), we fall through to a generic np.asarray() attempt instead of
    propagating the exception. That matters because the caller two levels
    up is is_trackable(), which used to catch ANY exception here and
    quietly report the variable as untrackable -- which is what makes it
    show up to the agent as "NoneType, not run yet" even though it has a
    perfectly good value. A wrong backend guess should degrade to "maybe a
    slightly mislabeled backend string" wherever possible, never to
    "the value silently vanished."
    """
    backend = detect_backend(x)

    if backend == "NumPy":
        return x

    if backend == "PyTorch":
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return _generic_to_numpy(x)

    if backend == "TensorFlow":
        try:
            return x.numpy()
        except Exception:
            return _generic_to_numpy(x)

    if backend == "CuPy":
        try:
            return cupy.asnumpy(x)
        except Exception:
            return _generic_to_numpy(x)

    if backend == "JAX":
        try:
            return np.asarray(x)
        except Exception:
            return _generic_to_numpy(x)

    if backend == "SciPy Sparse":
        # .toarray() densifies -- for a genuinely huge sparse matrix (e.g.
        # a large TF-IDF matrix) this can be a real memory jump versus the
        # sparse representation, but Pulse needs real values to compute
        # min/max/mean/nan/inf on, the same trade-off it already makes for
        # every other backend (nothing here is downsampled either).
        try:
            return x.toarray()
        except Exception:
            return _generic_to_numpy(x)

    if backend == "Pandas":
        try:
            return x.to_numpy()
        except Exception:
            return _generic_to_numpy(x)

    try:
        return _generic_to_numpy(x)
    except Exception:
        raise TypeError(f"Cannot convert {type(x)} to numpy.")


def scalar_value(x) -> float:
    """Pull a plain Python float out of any backend's 0-d tensor/array,
    or a bare Python number. Used for loss/metric line charts."""
    arr = to_numpy(x)
    return float(np.asarray(arr).reshape(-1)[0])


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
        "SciPy Sparse": HAS_SCIPY_SPARSE,
        "Pandas": HAS_PANDAS,
    }


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def statistics(x):
    """Compute summary stats without forcing an unnecessary float64 copy.

    Float tensors (float16/32/64) are measured in their native dtype --
    np.isnan/np.isinf work fine on any float type, so there's no reason to
    duplicate a large float32 weight/gradient tensor into float64 just to
    read its min/max/mean. Only non-floating dtypes (int/bool), which can't
    represent NaN/Inf natively, get upcast -- and only for the isnan/isinf
    checks, which numpy requires a float dtype for.
    """
    arr = to_numpy(x)

    if np.issubdtype(arr.dtype, np.floating):
        work = arr
    else:
        work = arr.astype(np.float64)

    finite = work[np.isfinite(work)] if work.size else work

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
        "nan": int(np.isnan(work).sum()) if work.size else 0,
        "inf": int(np.isinf(work).sum()) if work.size else 0,
    }


# ----------------------------------------------------------------------
# Variable filtering
# ----------------------------------------------------------------------

def is_trackable(obj):
    """True only for arrays/tensors (any backend) or bare numbers with a
    genuinely numeric dtype -- not strings, dicts, or arbitrary objects
    that happen to survive a best-effort np.asarray() call."""
    if not has_shape(obj):
        return False
    try:
        info = describe_tensor(obj)
    except Exception:
        return False
    return np.issubdtype(np.dtype(info.dtype), np.number)