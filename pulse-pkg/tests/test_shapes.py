"""pulse_shapes: describing shapes, checking them against specs with shared symbols, explaining the
usual shape errors, tracing a torch forward pass to the layer that failed, and the probe runner."""
import json
import subprocess
import sys
import types

import numpy as np
import pytest

import pulse
from pulse import pulse_shapes as ps


# ---- describing ----------------------------------------------------------------------------

def test_describe_array_reports_shape_dtype_device():
    info = ps.describe(np.zeros((32, 784), dtype="float32"))
    assert info["kind"] == "array"
    assert info["shape"] == [32, 784]
    assert info["dtype"] == "float32"
    assert info["type"].endswith("ndarray")


def test_describe_scalar_and_zero_dim():
    assert ps.describe(3)["kind"] == "scalar"
    zero = ps.describe(np.float32(1.0))
    assert zero["kind"] == "array" and zero["shape"] == []


def test_describe_containers_are_bounded_and_nested():
    x = np.zeros((2, 3))
    info = ps.describe({"a": x, "b": [x] * 20, "c": "s"})
    assert info["kind"] == "dict" and set(info["items"]) == {"a", "b", "c"}
    assert info["items"]["b"]["len"] == 20
    assert len(info["items"]["b"]["items"]) == ps._MAX_ITEMS
    assert "assumed to match" in info["items"]["b"]["note"]


def test_describe_never_recurses_past_the_depth_limit():
    deep = [[[[[np.zeros(1)]]]]]
    text = "\n".join(ps.render("deep", ps.describe(deep)))
    assert "deep" in text                                   # and, above all, it returned


def test_describe_module_counts_parameters():
    class P:
        def __init__(self, shape):
            self.shape, self.dtype = shape, "torch.float32"

    class M:
        def forward(self, x):
            return x

        def named_parameters(self):
            return [("w", P((4, 3))), ("b", P((3,)))]

    info = ps.describe(M())
    assert info["kind"] == "module" and info["parameters"] == 15
    assert info["params"][0] == {"name": "w", "shape": [4, 3], "dtype": "float32"}


def test_tensorflow_style_shape_with_unknown_dim():
    class TS:
        def as_list(self):
            return [None, 10]

    class T:
        shape, dtype = TS(), "tf.float32"

    info = ps.describe(T())
    assert info["shape"] == [None, 10]
    assert "(?, 10)" in "\n".join(ps.render("t", info))


def test_render_one_dim_shape_has_trailing_comma():
    assert "(5,)" in ps.render("v", ps.describe(np.zeros(5)))[0]


# ---- specs ---------------------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("(B, 10)", [("sym", "B"), ("int", 10)]),
    ("[N, *, 3]", [("sym", "N"), ("any", None), ("int", 3)]),
    ("B, C, ...", [("sym", "B"), ("sym", "C"), ("rest", None)]),
    ("(10,)", [("int", 10)]),
    ("()", []),
    ((None, 10), [("any", None), ("int", 10)]),
])
def test_parse_spec(spec, expected):
    assert ps.parse_spec(spec) == expected


@pytest.mark.parametrize("bad", ["(B, 3x)", "(..., ...)", "(-1, 2)", "(B; 3)"])
def test_parse_spec_rejects_what_it_cannot_read(bad):
    with pytest.raises(ValueError):
        ps.parse_spec(bad)


def test_match_exact_and_symbolic():
    assert ps.check_shape(np.zeros((32, 10)), "(B, 10)").ok
    r = ps.check_shape(np.zeros((32, 7)), "(B, 10)")
    assert not r.ok and "dimension 1: expected 10, got 7" in r.problems[0]


def test_rank_mismatch_is_explained():
    r = ps.check_shape(np.zeros((32, 10)), "(B, C, H, W)")
    assert not r.ok and "expected 4 dimensions" in r.problems[0] and "has 2" in r.problems[0]


def test_ellipsis_and_wildcard():
    x = np.zeros((2, 3, 4, 5))
    assert ps.check_shape(x, "(B, ...)").ok
    assert ps.check_shape(x, "(B, ..., 5)").ok
    assert not ps.check_shape(x, "(B, ..., 6)").ok
    assert ps.check_shape(x, "(*, *, *, *)").ok
    assert not ps.check_shape(np.zeros((2,)), "(B, C, ...)").ok         # too few dims for the fixed part


def test_scalar_spec():
    assert ps.check_shape(np.float32(1), "()").ok
    assert not ps.check_shape(np.zeros(3), "()").ok


def test_unknown_dimensions_match_anything_and_bind_nothing():
    class TS:
        def as_list(self):
            return [None, 10]

    class T:
        shape, dtype = TS(), "tf.float32"

    env = {}
    assert ps.check_shape(T(), "(B, 10)", env=env).ok
    assert "B" not in env


def test_symbols_must_agree_across_checks():
    x, y = np.zeros((32, 784)), np.zeros((16,))
    reports = ps.check_shapes({"x": (x, "(B, 784)"), "y": (y, "(B,)")})
    assert reports[0].ok and not reports[1].ok
    assert "B is 32 (from x)" in reports[1].problems[0]
    same = ps.check_shapes({"x": (x, "(B, 784)"), "y": (np.zeros((32,)), "(B,)")})
    assert all(r.ok for r in same)


def test_raise_on_mismatch():
    with pytest.raises(ps.ShapeMismatch) as info:
        ps.check_shape(np.zeros((2, 3)), "(2, 4)", "logits", raise_on_mismatch=True)
    assert "logits" in str(info.value)
    with pytest.raises(AssertionError):                    # ShapeMismatch is an AssertionError
        ps.check_shapes({"a": (np.zeros(2), "(3,)")}, raise_on_mismatch=True)


def test_non_array_with_a_spec_says_so():
    r = ps.check_shape("hello", "(B,)")
    assert not r.ok and "no shape" in r.problems[0]


def test_report_is_truthy_and_renders():
    r = ps.check_shape(np.zeros((2, 3)), "(B, 3)", "x")
    assert bool(r) is True
    assert "matches (B, 3)" in str(r)
    assert "DOES NOT MATCH" in str(ps.check_shape(np.zeros((2, 3)), "(B, 4)", "x"))


def test_public_api_is_lazy_and_works():
    assert pulse.check_shape(np.zeros((2, 3)), "(B, 3)").ok
    assert all(r.ok for r in pulse.check_shapes({"a": (np.zeros((2, 3)), "(B, 3)")}))
    assert "check_shape" in pulse.__all__


# ---- explaining shape errors ---------------------------------------------------------------

@pytest.mark.parametrize("message,needle", [
    ("mat1 and mat2 shapes cannot be multiplied (32x784 and 256x10)", "256 input features received 784"),
    ("shapes (32,784) and (256,10) not aligned: 784 (dim 1) != 256 (dim 0)", "784 and 256"),
    ("matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature "
     "(n?,k),(k,m?)->(n?,m?) (size 256 is different from 784)", "256"),
    ("The size of tensor a (32) must match the size of tensor b (10) at non-singleton dimension 1", "dimension 1"),
    ("Expected input batch_size (32) to match target batch_size (16).", "batch 32"),
    ("expected input[4, 1, 28, 28] to have 3 channels, but got 1 channels", "in_channels"),
    ("shape '[-1, 7]' is invalid for input of size 30", "30 elements"),
    ("operands could not be broadcast together with shapes (3,4) (5,) ", "do not broadcast"),
    ("Incompatible shapes: [32,10] vs. [16]", "TensorFlow"),
])
def test_known_shape_errors_are_explained(message, needle):
    hint = ps.explain_shape_error(message)
    assert hint and needle in hint


def test_unknown_errors_are_not_explained():
    assert ps.explain_shape_error("KeyError: 'lr'") is None
    assert ps.explain_shape_error("") is None


# ---- tracing a torch forward pass (a stand-in implementing torch's global module hooks) -------

class FakeTensor:
    def __init__(self, *shape):
        self.shape, self.dtype = tuple(shape), "torch.float32"


def _install_fake_torch(monkeypatch):
    pre, post = [], []

    class Handle:
        def __init__(self, bucket, fn):
            self.bucket, self.fn = bucket, fn

        def remove(self):
            self.bucket.remove(self.fn)

    def register(bucket):
        def _register(fn):
            bucket.append(fn)
            return Handle(bucket, fn)
        return _register

    class Module:
        def __init__(self, *children):
            self._modules = {str(i): c for i, c in enumerate(children)}

        def extra_repr(self):
            return ""

        def __call__(self, x):
            for hook in list(pre):
                hook(self, (x,))
            out = self.forward(x)
            for hook in list(post):
                hook(self, (x,), out)
            return out

    class Linear(Module):
        def __init__(self, i, o):
            super().__init__()
            self.i, self.o = i, o

        def extra_repr(self):
            return f"in_features={self.i}, out_features={self.o}"

        def forward(self, x):
            if x.shape[-1] != self.i:
                raise RuntimeError(f"mat1 and mat2 shapes cannot be multiplied "
                                   f"({x.shape[0]}x{x.shape[1]} and {self.i}x{self.o})")
            return FakeTensor(*x.shape[:-1], self.o)

    class ReLU(Module):
        def forward(self, x):
            return x

    class Sequential(Module):
        def forward(self, x):
            for child in self._modules.values():
                x = child(x)
            return x

    mod = types.ModuleType("torch.nn.modules.module")
    mod.register_module_forward_pre_hook = register(pre)
    mod.register_module_forward_hook = register(post)
    modules = types.ModuleType("torch.nn.modules")
    modules.module = mod
    nn = types.ModuleType("torch.nn")
    nn.modules = modules
    torch = types.ModuleType("torch")
    torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch", torch)
    return types.SimpleNamespace(Linear=Linear, ReLU=ReLU, Sequential=Sequential, pre=pre, post=post)


def test_trace_records_each_layers_shapes(monkeypatch):
    t = _install_fake_torch(monkeypatch)
    net = t.Sequential(t.Linear(784, 256), t.ReLU(), t.Linear(256, 10))
    with ps.ShapeTrace() as trace:
        net(FakeTensor(4, 784))
    report = trace.report()
    assert report["available"] and report["module_calls"] == 4          # 3 layers + the Sequential
    assert "failed" not in report
    layers = [(e["module"], e["in"], e["out"]) for e in report["last_ok"]]       # leaves only
    assert layers[0] == ("Linear(in_features=784, out_features=256)", ["(4, 784)"], ["(4, 256)"])
    assert layers[-1][2] == ["(4, 10)"]


def test_trace_names_the_layer_that_failed_and_where_it_was_called_from(monkeypatch):
    t = _install_fake_torch(monkeypatch)
    net = t.Sequential(t.Linear(784, 128), t.ReLU(), t.Linear(256, 10))        # 128 -> 256: the bug
    with ps.ShapeTrace() as trace:
        with pytest.raises(RuntimeError):
            net(FakeTensor(4, 784))
    failed = trace.report()["failed"]
    assert failed["module"] == "Linear(in_features=256, out_features=10)"
    assert failed["in"] == ["(4, 128)"]
    assert failed["inside"] == ["Sequential"]
    text = "\n".join(ps.render_trace(trace.report()))
    assert "FAILED INSIDE: Linear(in_features=256, out_features=10) receiving (4, 128)" in text
    assert "called from: Sequential" in text


def test_trace_removes_its_hooks_even_after_an_error(monkeypatch):
    t = _install_fake_torch(monkeypatch)
    with ps.ShapeTrace():
        assert len(t.pre) == 1 and len(t.post) == 1
    assert t.pre == [] and t.post == []


def test_trace_without_torch_is_a_harmless_noop(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with ps.ShapeTrace() as trace:
        pass
    assert trace.available is False
    assert "no torch module trace" in "\n".join(ps.render_trace(trace.report()))


def test_trace_keeps_only_the_recent_calls(monkeypatch):
    t = _install_fake_torch(monkeypatch)
    layer = t.ReLU()
    with ps.ShapeTrace() as trace:
        for _ in range(ps._MAX_TRACE_EVENTS + 50):
            layer(FakeTensor(1, 1))
    assert len(trace.events) == ps._MAX_TRACE_EVENTS
    assert trace.calls == ps._MAX_TRACE_EVENTS + 50


# ---- the probe runner ----------------------------------------------------------------------

def test_run_probe_describes_what_the_code_left_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = ps.run_probe({"code": "import numpy as np\nx = np.zeros((4, 784))\ny = 3\nnot_shaped = len"})
    names = [v["name"] for v in report["vars"]]
    assert "x" in names and "not_shaped" not in names
    assert report["error"] is None


def test_run_probe_reports_variables_as_they_were_when_it_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = "import numpy as np\nx = np.zeros((32, 784))\nw = np.zeros((256, 10))\nout = x @ w\n"
    report = ps.run_probe({"code": code})
    assert report["error"]["type"] == "ValueError"
    assert "256" in report["error"]["hint"] and "784" in report["error"]["hint"]
    assert any("out = x @ w" in f for f in report["error"]["frames"])      # the probe's own line is shown
    assert {v["name"] for v in report["vars"]} >= {"x", "w"}
    assert "PROBE FAILED" in ps.render_report(report)


def test_run_probe_checks_expectations_with_shared_symbols(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = "import numpy as np\nx = np.zeros((32, 784))\ny = np.zeros((16,))\n"
    report = ps.run_probe({"code": code, "expect": {"x": "(B, 784)", "y": "(B,)", "missing": "(1,)"}})
    ok = {c["name"]: c["ok"] for c in report["expect"]}
    assert ok == {"x": True, "y": False, "missing": False}
    assert "could not evaluate" in next(c for c in report["expect"] if c["name"] == "missing")["problems"][0]
    assert "FAIL y" in ps.render_report(report)


def test_run_probe_ignores_a_failure_the_probe_caught_itself(tmp_path, monkeypatch):
    t = _install_fake_torch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "fake_lib", types.SimpleNamespace(t=t, FakeTensor=FakeTensor))
    code = ("from fake_lib import t, FakeTensor\n"
            "try:\n    t.Linear(3, 3)(FakeTensor(2, 5))\nexcept RuntimeError:\n    pass\n"
            "ok = t.Linear(5, 3)(FakeTensor(2, 5))\n")
    report = ps.run_probe({"code": code})
    assert report["error"] is None
    assert "failed" not in report["trace"]


def test_run_probe_survives_exit_and_syntax_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ps.run_probe({"code": "import sys\nsys.exit(3)"})["error"]["type"] == "SystemExit"
    assert ps.run_probe({"code": "import sys\nsys.exit(0)"})["error"] is None
    assert ps.run_probe({"code": "def f(:"})["error"]["type"] == "SyntaxError"


def test_run_probe_can_import_from_the_project_directory(tmp_path, monkeypatch):
    (tmp_path / "mymodel.py").write_text("import numpy as np\nW = np.zeros((7, 2))\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    report = ps.run_probe({"code": "import mymodel", "exprs": ["mymodel.W"]})
    assert report["vars"][0]["info"]["shape"] == [7, 2]


def test_the_runner_works_as_a_standalone_script(tmp_path):
    (tmp_path / "request.json").write_text(json.dumps({
        "code": "import numpy as np\nx = np.zeros((2, 3))", "expect": {"x": "(B, 3)"},
        "report_path": str(tmp_path / "report.json")}))
    done = subprocess.run([sys.executable, ps.__file__, str(tmp_path / "request.json")], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["expect"][0]["ok"] is True


def test_the_runner_does_not_let_its_own_folder_shadow_project_modules(tmp_path):
    """Run by path, sys.path[0] is pulse's package folder, which holds modules like pulse_ui.py;
    a project module with the same name must win."""
    (tmp_path / "pulse_ui.py").write_text("MARK = 'project'\n")
    (tmp_path / "request.json").write_text(json.dumps({
        "code": "import pulse_ui\nmark = pulse_ui.MARK", "exprs": ["mark"],
        "report_path": str(tmp_path / "report.json")}))
    # With the project already on PYTHONPATH (behind the script's own folder), only dropping that
    # folder from sys.path lets the project's module win.
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path)}
    subprocess.run([sys.executable, ps.__file__, str(tmp_path / "request.json")], cwd=tmp_path, env=env,
                   capture_output=True, text=True, timeout=60)
    report = json.loads((tmp_path / "report.json").read_text())
    assert "'project'" in report["vars"][0]["info"]["value"]


def test_render_report_when_there_is_nothing_to_say():
    assert "nothing with a shape" in ps.render_report({"error": None, "vars": [], "expect": [], "trace": None})