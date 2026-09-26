"""Sweep: pulse_backend.py -- framework detection, conversion and statistics.

torch-specific tests are skipped when torch is not importable (run them on the PC venv).
"""
import warnings

import numpy as np
import pytest

from pulse import pulse_backend as B

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


# ----------------------------------------------------------------------------- bugs

@needs_torch
def test_bug_torch_bfloat16_tensor_not_trackable():
    """bf16 is the default dtype of modern mixed-precision / LLM training, but
    `to_numpy` calls `.numpy()` which torch refuses for bfloat16, and the generic
    `np.asarray` fallback fails the same way. So `is_trackable` says False and
    `statistics` raises TypeError: every bf16 weight/activation/loss is invisible to
    Pulse. Correct: upcast unsupported float dtypes (bf16, fp8) to float32 on the host
    and report stats (dtype label may stay 'bfloat16')."""
    x = torch.tensor([1.0, 2.0, float("nan")], dtype=torch.bfloat16)
    assert B.is_trackable(x)
    stats = B.statistics(x)
    assert stats["nan"] == 1
    assert stats["max"] == 2.0


@needs_torch
def test_bug_torch_bfloat16_scalar_loss_unreadable():
    """A bf16 0-d loss (autocast output) can't be turned into a float by
    scalar_value -> TypeError, so the loss chart / NaN detection never sees it."""
    loss = torch.tensor(0.5, dtype=torch.bfloat16)
    assert B.scalar_value(loss) == 0.5


@needs_torch
def test_bug_describe_tensor_copies_data_to_read_metadata():
    """describe_tensor/is_trackable/label only need shape/dtype/device, but they call
    to_numpy() first -- a full device->host copy (and CUDA sync) of the whole tensor on
    the training thread, contrary to the module's 'never touch accelerator memory beyond
    reading a tracked value' promise (pulse_cli calls describe_tensor(...).shape and
    is_trackable() directly in several places). A 'meta' tensor has metadata but no
    data, so it exposes this: describe_tensor raises. Correct: build TensorInfo from
    .shape/.dtype/.device without converting."""
    x = torch.empty(3, 4, device="meta")
    info = B.describe_tensor(x)
    assert info.shape == (3, 4)
    assert "float32" in info.dtype


@needs_torch
def test_bug_is_trackable_triggers_host_copy(monkeypatch):
    """Same bug, observed directly: is_trackable() on a torch tensor calls
    Tensor.cpu() (a D2H copy for a CUDA tensor) just to decide trackability."""
    calls = []
    orig = torch.Tensor.cpu

    def spy(self, *a, **k):
        calls.append(tuple(self.shape))
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "cpu", spy)
    assert B.is_trackable(torch.zeros(64, 64))
    assert calls == [], "is_trackable copied the tensor to host to read its dtype"


@needs_torch
def test_bug_torch_sparse_tensor_crashes_statistics():
    """A torch sparse COO tensor (embedding grads with sparse=True, graph adjacency)
    makes statistics() raise TypeError ('can't convert Sparse layout tensor to numpy');
    the scipy-sparse path densifies, the torch one doesn't. Correct: to_dense() first."""
    x = torch.eye(3).to_sparse()
    stats = B.statistics(x)
    assert stats["max"] == 1.0 and stats["min"] == 0.0


def test_bug_complex_statistics_silently_drop_imaginary_part():
    """complex arrays are 'trackable' (np.number) but statistics() does
    arr.astype(float64), silently discarding the imaginary part: [1+1j, 5j] is reported
    as min 0 / max 1 / mean 0.5, which is simply wrong (|z| is 1.41 and 5). Correct:
    compute on np.abs(arr) (magnitude) or report real/imag separately; never cast away."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = B.statistics(np.array([1 + 1j, 5j]))
    assert stats["max"] == pytest.approx(5.0)


def test_bug_float16_std_overflows_to_inf():
    """statistics() keeps float16 in its native dtype 'to avoid a copy', but the
    variance of values around +-300 exceeds float16's max (65504), so std comes back
    inf for a perfectly finite fp16 activation tensor -- a false 'inf' signal. Correct:
    reduce with dtype=np.float32/float64 (np.std(..., dtype=np.float64)) for low-precision
    floats."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = B.statistics(np.array([-300.0, 300.0], dtype=np.float16))
    assert np.isfinite(stats["std"])
    assert stats["std"] == pytest.approx(300.0)


def test_bug_float32_mean_overflows_to_inf():
    """Same class: float32 mean of large-but-finite values overflows during the sum
    (mean=inf while min=max=3e38). Accumulate in float64."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = B.statistics(np.array([3e38, 3e38], dtype=np.float32))
    assert np.isfinite(stats["mean"])


def test_bug_scalar_value_of_empty_array_raises_indexerror():
    """scalar_value() on an empty array raises a bare IndexError; callers treating it
    as 'pull the float out of a loss' get an unrelated crash. Correct: raise a clear
    ValueError (or return nan)."""
    with pytest.raises(ValueError):
        B.scalar_value(np.array([]))


def test_bug_jax_device_is_none_on_current_jax(monkeypatch):
    """Current JAX exposes `.device` as a property (not a method), so `x.device()` raised
    and device_of returned None."""
    class _Dev:
        def __str__(self):
            return "cuda:0"

    class _FakeJaxArray:
        device = _Dev()

    monkeypatch.setattr(B, "detect_backend", lambda x: "JAX")
    assert B.device_of(_FakeJaxArray()) == "cuda:0"


def test_bug_tf_sparse_and_ragged_crash_statistics():
    tf = pytest.importorskip("tensorflow")
    s = B.statistics(tf.sparse.from_dense(tf.eye(3)))
    assert s["max"] == 1.0 and s["min"] == 0.0
    r = B.statistics(tf.ragged.constant([[1.0, 2.0], [3.0]]))
    assert r["mean"] == 2.0


def test_ok_frameworks_detected_without_importing_them():
    import sys
    assert B.detect_backend(np.zeros(2)) == "NumPy"
    if "torch" in sys.modules:
        assert B.detect_backend(sys.modules["torch"].zeros(2)) == "PyTorch"


# ----------------------------------------------------------------------------- ok

def test_ok_numpy_passthrough_no_copy():
    a = np.arange(6.0)
    assert B.to_numpy(a) is a


def test_ok_numpy_scalar_types_detected_as_numpy():
    assert B.detect_backend(np.float32(1.5)) == "NumPy"
    assert B.is_trackable(np.float64(2.0))
    assert B.tensor_kind(np.float32(1.0)) == "scalar"


def test_ok_python_numbers_and_non_numbers():
    assert B.is_trackable(3) and B.is_trackable(2.5)
    assert not B.is_trackable("abc")
    assert not B.is_trackable([1, 2, 3])
    assert not B.is_trackable({"a": 1})
    assert not B.has_shape(1 + 2j)


def test_ok_statistics_counts_nan_inf_and_ignores_them_for_minmax():
    s = B.statistics(np.array([1.0, np.nan, np.inf, -2.0], dtype=np.float32))
    assert s["nan"] == 1 and s["inf"] == 1
    assert s["min"] == -2.0 and s["max"] == 1.0


def test_ok_statistics_all_nonfinite_and_empty():
    s = B.statistics(np.array([np.nan, np.inf]))
    assert s["min"] is None and s["nan"] == 1 and s["inf"] == 1
    e = B.statistics(np.zeros((0, 3)))
    assert e["min"] is None and e["nan"] == 0 and e["shape"] == (0, 3)


def test_ok_int_and_bool_statistics():
    s = B.statistics(np.array([1, 2, 3], dtype=np.int32))
    assert s["mean"] == 2.0 and s["nan"] == 0
    assert not B.is_trackable(np.array([True, False]))


def test_ok_tensor_kinds():
    assert B.tensor_kind(np.zeros(())) == "scalar"
    assert B.tensor_kind(np.zeros(3)) == "vector"
    assert B.tensor_kind(np.zeros((2, 3))) == "matrix"
    assert B.tensor_kind(np.zeros((2, 3, 4))) == "tensor"
    assert B.tensor_kind(object()) is None


def test_ok_scipy_sparse_is_densified():
    sp = pytest.importorskip("scipy.sparse")
    m = sp.csr_matrix(np.eye(3))
    assert B.detect_backend(m) == "SciPy Sparse"
    s = B.statistics(m)
    assert s["max"] == 1.0 and s["shape"] == (3, 3)


@needs_torch
def test_ok_torch_requires_grad_noncontiguous_and_0d():
    w = torch.ones(3, 4, requires_grad=True)
    assert B.statistics(w)["mean"] == 1.0
    t = torch.arange(12.0).reshape(3, 4).t()
    assert B.statistics(t)["shape"] == (4, 3)
    assert B.scalar_value(torch.tensor(2.5)) == 2.5
    assert B.device_of(w) == "cpu"
    assert B.detect_backend(w) == "PyTorch"


def test_ok_tensorflow_tensor_and_variable():
    tf = pytest.importorskip("tensorflow")
    v = tf.Variable([1.0, 2.0])
    assert B.detect_backend(v) == "TensorFlow"
    s = B.statistics(v)
    assert s["mean"] == 1.5
    assert B.is_trackable(tf.constant([1, 2]))
    assert not B.is_trackable(tf.constant(["a"]))
