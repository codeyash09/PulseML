"""Sweep: pulse_pdf.py -- heatmap PDF snapshots."""
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("fpdf")
pytest.importorskip("matplotlib")

from pulse.pulse_pdf import generate_heatmap_pdf, _reduce_to_2d  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ============================================================================ bugs

def test_bug_non_latin1_variable_name_crashes_pdf(tmp_path):
    """Python identifiers may be Unicode, and ML code uses them (θ, α, λ, μ). The PDF is
    written with the core 'Helvetica' font, which only covers latin-1, so fpdf raises
    FPDFUnicodeEncodingException for `θ` and no snapshot is ever saved for that variable.
    Correct: sanitise text for the core font (e.g. encode('latin-1','replace')) or embed
    a Unicode TTF (DejaVu ships with matplotlib)."""
    path = generate_heatmap_pdf("θ", np.random.randn(4, 5), 1, output_dir=str(tmp_path))
    assert os.path.isfile(path)


def test_bug_variable_name_can_escape_output_dir(tmp_path):
    """var_name is joined straight into the path. The CLI only strips '[', ']' and ','
    from sub-names such as batch['key'] -- a dict key containing '/' or '..' (e.g.
    metrics['../../x'] or 'train/loss') writes outside/into nested dirs of output_dir,
    and an absolute-looking name discards output_dir entirely. Correct: replace os.sep,
    os.altsep and '..' in var_name (e.g. re.sub(r'[^\\w.-]', '_', name).strip('.'))."""
    out = tmp_path / "out"
    try:
        path = generate_heatmap_pdf("../escaped", np.ones((3, 3)), 1, output_dir=str(out))
    except Exception as exc:  # pragma: no cover - any crash is also wrong
        pytest.fail(f"crashed: {exc!r}")
    assert os.path.realpath(path).startswith(os.path.realpath(str(out)) + os.sep)
    assert not (tmp_path / "escaped").exists()


def test_bug_bfloat16_torch_tensor_pdf_fails(tmp_path):
    """Inherits the backend bf16 bug: a bf16 weight matrix can never get a snapshot
    (TypeError: Got unsupported ScalarType BFloat16)."""
    torch = pytest.importorskip("torch")
    path = generate_heatmap_pdf("w", torch.ones(3, 3, dtype=torch.bfloat16), 1, output_dir=str(tmp_path))
    assert os.path.isfile(path)


# ============================================================================ ok

def test_ok_basic_layout_and_path(tmp_path):
    path = generate_heatmap_pdf("weights", np.random.randn(8, 6).astype(np.float32), 42, output_dir=str(tmp_path))
    assert path == os.path.join(str(tmp_path), "weights", "step000042.pdf")
    with open(path, "rb") as f:
        assert f.read(5) == b"%PDF-"


@pytest.mark.parametrize("value", [
    np.zeros((4, 5)),                       # all zero -> vmin == vmax
    np.full((3, 3), np.nan),                # all nan
    np.array([[1.0, np.inf], [-np.inf, 2]]),
    np.float32(3.0),                        # 0-d
    np.arange(10.0),                        # 1-d
    np.random.randn(2, 3, 4, 5),            # n-d
    np.ones((3, 3), dtype=bool),
    np.zeros((0, 4)),                       # empty
])
def test_ok_edge_values_render(tmp_path, value):
    path = generate_heatmap_pdf("v", value, 1, output_dir=str(tmp_path))
    assert os.path.getsize(path) > 0


def test_ok_no_temp_png_left_behind(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    os.makedirs(tmp_path / "tmp")
    generate_heatmap_pdf("v", np.ones((2, 2)), 1, output_dir=str(tmp_path / "o"))
    assert os.listdir(tmp_path / "tmp") == []


def test_ok_reduce_to_2d():
    assert _reduce_to_2d(np.zeros(())).shape == (1, 1)
    assert _reduce_to_2d(np.zeros(5)).shape == (1, 5)
    assert _reduce_to_2d(np.zeros((2, 3, 4, 5))).shape == (2, 3)
