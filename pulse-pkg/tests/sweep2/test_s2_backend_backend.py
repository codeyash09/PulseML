"""Sweep 2: pulse_backend.py -- lazy framework lookup, conversions, metadata, statistics.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_* are regression coverage for behaviour verified to work.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_backend as B  # noqa: E402


def _run_isolated(code, extra_path=None, tmp_path=None):
    """Run `code` in a fresh interpreter (no framework preloaded) with Pulse's src first."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    parts = ([extra_path] if extra_path else []) + [SRC]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                          text=True, timeout=240, env=env, cwd=str(tmp_path) if tmp_path else None)


# ---------------------------------------------------------------------------------------
# bugs
# ---------------------------------------------------------------------------------------

def test_bug_describe_tensor_densifies_scipy_sparse_matrix():
    """describe_tensor()/is_trackable() only read metadata for accelerator backends; for a
    scipy sparse matrix (_metadata_dtype returns None) they call to_numpy(), i.e.
    .toarray(), just to learn its shape and dtype. For a realistic TF-IDF / one-hot matrix
    (200k x 200k, a few non-zeros) that is a 160+ GB dense allocation; when it fails,
    the MemoryError is swallowed, np.asarray() wraps the matrix in a 0-d object array,
    and the variable is reported as shape () dtype object -- untrackable. Correct: shape
    and dtype come from the sparse matrix's own .shape/.dtype, no densifying."""
    sp = pytest.importorskip("scipy.sparse")
    m = sp.csr_matrix((np.ones(3, dtype=np.float32), (np.array([0, 5, 9]), np.array([1, 2, 3]))),
                      shape=(200_000, 200_000))
    calls = []

    class Spy(type(m)):
        def toarray(self, *a, **k):
            calls.append(1)
            raise MemoryError("Unable to allocate 149. GiB")

    spy = Spy(m)
    info = B.describe_tensor(spy)
    assert info.shape == (200_000, 200_000), info.shape
    assert info.dtype == "float32", info.dtype
    assert B.is_trackable(spy)
    assert calls == [], "metadata lookup densified the sparse matrix"


def test_bug_torch_conjugate_view_crashes_statistics():
    """`zc = z.conj()` on a complex tensor is a lazy view with the conjugate bit set;
    `.imag` of it has the negative bit set. `.numpy()` refuses both with RuntimeError
    (not TypeError, so _torch_to_numpy's fallback doesn't run), and the generic
    np.asarray() fallback calls .numpy() again -- statistics() raises. Correct: resolve
    the view (resolve_conj/resolve_neg) before the host copy."""
    torch = pytest.importorskip("torch")
    z = torch.tensor([3 + 4j, 1 - 1j], dtype=torch.complex64)
    stats = B.statistics(z.conj())
    assert stats["max"] == pytest.approx(5.0)
    neg = B.statistics(z.conj().imag)
    assert neg["min"] == pytest.approx(-4.0) and neg["max"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------
# verified working
# ---------------------------------------------------------------------------------------

def test_ok_framework_imported_after_pulse_is_detected(tmp_path):
    """Detection is a sys.modules lookup at call time: torch imported AFTER pulse (and
    only through a submodule import) is still recognised, and importing pulse imports
    neither torch nor tensorflow."""
    pytest.importorskip("torch")
    r = _run_isolated("""
        import sys
        from pulse import pulse_backend as B
        assert "torch" not in sys.modules and "tensorflow" not in sys.modules
        import torch.nn
        import torch
        t = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
        assert B.detect_backend(t) == "PyTorch", B.detect_backend(t)
        assert B.is_trackable(t)
        s = B.statistics(t)
        assert s["mean"] == 1.0 and s["shape"] == (2, 3), s
        print("OK")
    """, tmp_path=tmp_path)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_ok_user_module_named_like_a_framework_does_not_break_detection(tmp_path):
    """A user's own `torch.py` / `jax.py` beside the script (no Tensor / Array attribute)
    is treated as 'not that framework' instead of crashing detection."""
    (tmp_path / "torch.py").write_text("x = 1\n")
    (tmp_path / "jax.py").write_text("Array = None\n")
    r = _run_isolated("""
        import sys, numpy as np
        sys.path.insert(0, ".")
        import torch, jax
        from pulse import pulse_backend as B
        a = np.ones((2, 2), dtype=np.float32)
        assert B.detect_backend(a) == "NumPy"
        assert B.detect_backend(3.0) == "Python"
        assert B.statistics(a)["mean"] == 1.0
        print("OK")
    """, extra_path=str(tmp_path), tmp_path=tmp_path)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_ok_available_backends_does_not_import(tmp_path):
    r = _run_isolated("""
        import sys
        from pulse import pulse_backend as B
        info = B.available_backends()
        assert info["NumPy"] is True
        assert "torch" not in sys.modules and "tensorflow" not in sys.modules
        print("OK")
    """, tmp_path=tmp_path)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_ok_torch_dtypes_and_layouts():
    torch = pytest.importorskip("torch")
    assert B.statistics(torch.ones(3, dtype=torch.bfloat16))["mean"] == 1.0
    assert B.statistics(torch.ones(3).to(torch.float8_e4m3fn))["mean"] == 1.0
    sparse = torch.eye(3).to_sparse()
    assert B.statistics(sparse)["max"] == 1.0
    z = torch.tensor([3 + 4j], dtype=torch.complex64)
    assert B.statistics(z)["max"] == pytest.approx(5.0)
    big = torch.full((10,), 60000.0, dtype=torch.float16)
    s = B.statistics(big)
    assert np.isfinite(s["mean"]) and np.isfinite(s["std"])
    assert not B.is_trackable(torch.ones(2, dtype=torch.bool))


def test_ok_numpy_scalars_and_empty():
    assert B.detect_backend(np.float32(1.5)) == "NumPy"
    assert B.scalar_value(np.float32(1.5)) == 1.5
    with pytest.raises(ValueError):
        B.scalar_value(np.array([]))
    s = B.statistics(np.array([1.0, np.nan, np.inf], dtype=np.float32))
    assert s["nan"] == 1 and s["inf"] == 1 and s["max"] == 1.0
    assert not B.is_trackable("a string")
    assert not B.is_trackable({"a": 1})
