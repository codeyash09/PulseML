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
# Optional backends -- looked up lazily
# ----------------------------------------------------------------------
#
# None of the optional frameworks is imported here: importing Pulse must not
# pay for a TensorFlow/PyTorch/JAX import (seconds, and GPU memory on some
# builds) in a script that never uses them. An object can only be a
# torch.Tensor / tf.Tensor / jax.Array / DataFrame if the script already
# imported that framework, so each backend is looked up in sys.modules at the
# moment an object is inspected.

import importlib.util
import sys


def _loaded(name, attr):
    """The already-imported module `name`, or None. A module that is still
    half-way through its own import (no `attr` yet) counts as not loaded."""
    mod = sys.modules.get(name)
    if mod is None or not hasattr(mod, attr):
        return None
    return mod


def _torch():
    return _loaded("torch", "Tensor")


def _tf():
    return _loaded("tensorflow", "is_tensor")


def _cupy():
    return _loaded("cupy", "ndarray")


def _jax():
    return _loaded("jax", "Array")


def _sp_sparse():
    return _loaded("scipy.sparse", "issparse")


def _pd():
    return _loaded("pandas", "DataFrame")


def _installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


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
    torch = _torch()
    if torch is not None and isinstance(x, torch.Tensor):
        return "PyTorch"

    tf = _tf()
    if tf is not None:
        try:
            if tf.is_tensor(x):
                return "TensorFlow"
        except Exception:
            pass

    cupy = _cupy()
    if cupy is not None and isinstance(x, cupy.ndarray):
        return "CuPy"

    jax = _jax()
    if jax is not None:
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
    sp_sparse = _sp_sparse()
    if sp_sparse is not None and sp_sparse.issparse(x):
        return "SciPy Sparse"

    # pandas DataFrame/Series -- the standard input/output of most sklearn
    # preprocessing steps (ColumnTransformer, train_test_split, ...).
    pd = _pd()
    if pd is not None and isinstance(x, (pd.DataFrame, pd.Series)):
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
        # Current JAX: `.device` is a property (older: a method), and
        # `.devices()` is the set for sharded arrays.
        try:
            dev = getattr(x, "device", None)
            if callable(dev):
                dev = dev()
            if dev is None:
                dev = next(iter(x.devices()))
            return str(dev)
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
            return _torch_to_numpy(x)
        except Exception:
            return _generic_to_numpy(x)

    if backend == "TensorFlow":
        try:
            return _tf_to_numpy(x)
        except Exception:
            return _generic_to_numpy(x)

    if backend == "CuPy":
        try:
            return _cupy().asnumpy(x)
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


def _torch_to_numpy(x):
    """Host copy of a torch tensor. Sparse layouts are densified, and dtypes
    numpy has no type for (bfloat16, float8_*) are upcast to float32 -- on
    the host, after the copy, so no kernel runs on the accelerator."""
    torch = _torch()
    t = x.detach()
    if t.layout != torch.strided:
        t = t.to_dense()
    t = t.cpu()
    try:
        return t.numpy()
    except TypeError:
        if t.is_floating_point():
            return t.float().numpy()
        if t.is_complex():
            return t.to(torch.complex64).numpy()
        raise


def _tf_to_numpy(x):
    tf = _tf()
    if isinstance(x, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(x).numpy()
    ragged = getattr(tf, "RaggedTensor", None)
    if ragged is not None and isinstance(x, ragged):
        # The stored values; padding to a dense tensor would invent zeros.
        return x.flat_values.numpy()
    return x.numpy()


def scalar_value(x) -> float:
    """Pull a plain Python float out of any backend's 0-d tensor/array,
    or a bare Python number. Used for loss/metric line charts."""
    arr = np.asarray(to_numpy(x)).reshape(-1)
    if arr.size == 0:
        raise ValueError("scalar_value() of an empty array")
    return float(arr[0])


# ----------------------------------------------------------------------
# Metadata (renamed from inspect to avoid stdlib conflict)
# ----------------------------------------------------------------------

def _metadata_dtype(x, backend):
    """dtype name of an accelerator tensor read from its metadata, without
    copying its data to the host; None for backends that need a conversion."""
    if backend == "PyTorch":
        return str(x.dtype).replace("torch.", "")
    if backend == "TensorFlow":
        return x.dtype.name
    if backend in ("CuPy", "JAX"):
        return str(x.dtype)
    return None


def describe_tensor(x):
    backend = detect_backend(x)
    dtype = None
    try:
        dtype = _metadata_dtype(x, backend)
        shape = tuple(x.shape) if dtype is not None else None
    except Exception:
        dtype = None
    if dtype is not None and shape is not None and None not in shape:
        return TensorInfo(
            backend=backend,
            kind=tensor_kind(x),
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=device_of(x),
            object=x,
        )

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
    """Which backends are installed (checked without importing them)."""
    return {
        "NumPy": True,
        "PyTorch": _installed("torch"),
        "TensorFlow": _installed("tensorflow"),
        "CuPy": _installed("cupy"),
        "JAX": _installed("jax"),
        "SciPy Sparse": _installed("scipy"),
        "Pandas": _installed("pandas"),
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
    elif np.issubdtype(arr.dtype, np.complexfloating):
        # Magnitude: casting to float would silently drop the imaginary part.
        work = np.abs(arr)
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
        # Accumulate in float64: a float16/float32 sum or variance overflows
        # to inf on perfectly finite data.
        "mean": float(finite.mean(dtype=np.float64)) if finite.size else None,
        "std": float(finite.std(dtype=np.float64)) if finite.size else None,
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
    backend = detect_backend(obj)
    try:
        if backend == "PyTorch":
            dt = obj.dtype
            return dt != _torch().bool and not str(dt).startswith("torch.q")
        if backend == "TensorFlow":
            dt = obj.dtype
            return bool(dt.is_floating or dt.is_complex or dt.is_integer) and not dt.is_bool
        info = describe_tensor(obj)
    except Exception:
        return False
    return _numeric_dtype_name(info.dtype)


def _numeric_dtype_name(name):
    try:
        return bool(np.issubdtype(np.dtype(name), np.number))
    except Exception:
        # bfloat16 / float8_* (ml_dtypes) are numeric but not numpy types.
        return "float" in str(name)