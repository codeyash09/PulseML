"""Sweep: pulse_trace -- the TRACE: variable influence graph.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_* are regression coverage for behaviour verified to work.
"""
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_trace as PT  # noqa: E402


def F(label, src):
    return (label, "/proj/" + label, textwrap.dedent(src).lstrip("\n"))


def names(graph, direction):
    attr = "depth_up" if direction == "up" else "depth_down"
    return {(n.key[1], n.name) for n in graph.nodes.values()
            if n.key != graph.root and getattr(n, attr) is not None}


def all_names(graph):
    return {n.name for n in graph.nodes.values()}


# ---------------------------------------------------------------------------------------
# bugs
# ---------------------------------------------------------------------------------------

def test_bug_module_global_downstream_misses_functions_that_read_it():
    """`LR = 1e-3` at module level, read inside train(). Upstream from inside train()
    lands on the module node (_home_key), but downstream from the module node only walks
    MODULE statements, so tracing LR says nothing in any function reads it."""
    files = [F("train.py", """
        LR = 1e-3

        def train(model):
            opt = make_opt(model.parameters(), LR)
            return opt
        """)]
    g = PT.build(files, "LR")
    assert g is not None
    assert "opt" in {n for _s, n in names(g, "down")}, PT.render(g)


def test_bug_line_hint_inside_function_with_parameter_picks_another_function():
    """TRACE: x:model.py:6 where line 6 is inside forward(self, x) and x is forward's
    parameter. find_origin prefers 'the closest assignment above the line' across ALL
    scopes, so it starts at helper()'s unrelated local x instead of forward's parameter."""
    files = [F("model.py", """
        def helper():
            x = 5
            return x

        def forward(x):
            y = x * 2
            return y
        """)]
    g = PT.build(files, "x", file_hint="model.py", line_hint=6)
    assert g is not None
    assert g.root[1] == "forward", g.origin


def test_bug_file_hint_matches_other_files_with_same_suffix():
    """`loss:train.py` also matches pretrain.py / my_train.py (path.endswith), and the
    first such file wins -- the trace lands in the wrong file."""
    files = [
        F("pretrain.py", """
            def main():
                loss = 1.0
                return loss
            """),
        F("train.py", """
            def main():
                loss = 2.0
                return loss
            """),
    ]
    g = PT.build(files, "loss", file_hint="train.py")
    assert g is not None
    assert g.root[0] == "train.py", g.origin


def test_bug_module_qualified_call_maps_arguments_to_wrong_parameters():
    """`utils.scale(raw, m)` calls a plain function; the '-1 for the implicit self'
    offset is applied to every Attribute call, so parameter `factor` is said to come from
    `raw` (and `raw` flows into `factor`). Multi-file projects call helpers this way."""
    files = [
        F("main.py", """
            import utils

            def run():
                raw = load()
                m = 3
                out = utils.scale(raw, m)
                return out
            """),
        F("utils.py", """
            def scale(x, factor):
                return x * factor
            """),
    ]
    g = PT.build(files, "factor", file_hint="utils.py")
    assert g is not None
    ups = {n for _s, n in names(g, "up")}
    assert "m" in ups and "raw" not in ups, PT.render(g)


def test_bug_comprehension_variable_resolves_to_unrelated_module_variable():
    """`[v * 2 for v in data]`: `v` is the comprehension's own variable, but _reads
    treats it as a free name and _home_key binds it to the module-level `v`, inventing a
    causal link from an unrelated global."""
    files = [F("prep.py", """
        v = expensive_global()

        def double(data):
            out = [v * 2 for v in data]
            return out
        """)]
    g = PT.build(files, "out")
    assert g is not None
    ups = names(g, "up")
    assert ("<module>", "v") not in ups, PT.render(g)


def test_bug_closure_reads_of_enclosing_variable_are_not_followed():
    """A nested function reading the enclosing function's local is part of its flow;
    downstream from `scale` in outer() never reaches the use in inner()."""
    files = [F("m.py", """
        def outer(data):
            scale = 0.5
            def inner(x):
                y = x * scale
                return y
            return [inner(d) for d in data]
        """)]
    g = PT.build(files, "scale")
    assert g is not None
    assert "y" in {n for _s, n in names(g, "down")}, PT.render(g)


def test_bug_global_statement_is_ignored():
    """`global step` makes `step += 1` inside train() write the MODULE variable, but the
    trace keys it as train()'s own local, so it is disconnected from the module `step`
    that log() reads."""
    files = [F("m.py", """
        step = 0

        def train():
            global step
            step += 1

        def log():
            msg = f"step {step}"
            return msg
        """)]
    g = PT.build(files, "msg")
    assert g is not None
    up_nodes = [n for n in g.nodes.values() if n.name == "step" and n.depth_up is not None]
    lines = {s.lineno for n in up_nodes for s in n.sites}
    assert 5 in lines, PT.render(g)       # the `step += 1` in train()


def test_bug_walrus_assignment_is_not_an_assignment():
    """`while (batch := next(it, None)) is not None:` assigns `batch`; TRACE says it is
    never assigned anywhere."""
    files = [F("loop.py", """
        def run(it):
            total = 0
            while (batch := next(it, None)) is not None:
                total = total + batch
            return total
        """)]
    out = PT.trace(files, "batch")
    assert "never assigned" not in out, out


def test_bug_nested_attribute_store_is_dropped():
    """_target_names documents 'writing through an object changes that object'
    (cfg.lr -> cfg), but a two-level store `cfg.optim.lr = lr` yields NO target, so the
    flow of `lr` into `cfg` is lost."""
    files = [F("cfgs.py", """
        def setup(cfg):
            lr = 0.1
            cfg.optim.lr = lr
            return cfg
        """)]
    g = PT.build(files, "lr")
    assert g is not None
    assert "cfg" in {n for _s, n in names(g, "down")}, PT.render(g)


def test_bug_huge_generated_expression_crashes_the_whole_trace():
    """A file mid-edit is skipped ('not a reason to fail the trace'), but a generated file
    with a very deep expression makes ast.parse raise RecursionError, which is not caught:
    every TRACE in that project crashes."""
    deep = "data = " + "+".join(["1"] * 200_000) + "\n"
    files = [
        F("train.py", """
            def main():
                loss = 1.0
                return loss
            """),
        ("gen.py", "/proj/gen.py", deep),
    ]
    out = PT.trace(files, "loss")
    assert out.startswith("TRACE")


def test_bug_inherited_self_attribute_is_a_separate_variable():
    """self.lr set in Base.__init__ and read in Child.step is one attribute of one object,
    but keys are per class, so the trace of Child's use never reaches its assignment."""
    files = [F("m.py", """
        class Base:
            def __init__(self, lr):
                self.lr = lr

        class Child(Base):
            def step(self, g):
                delta = self.lr * g
                return delta
        """)]
    g = PT.build(files, "delta")
    assert g is not None
    lr_nodes = [n for n in g.nodes.values() if n.name == "self.lr"]
    assert lr_nodes and any(s.kind != "external" for n in lr_nodes for s in n.sites), PT.render(g)


# ---------------------------------------------------------------------------------------
# ok
# ---------------------------------------------------------------------------------------

def test_ok_assignment_chain_and_multiple_sites():
    files = [F("t.py", """
        def train(model, x, y, flag):
            out = model(x)
            if flag:
                loss = crit(out, y)
            else:
                loss = crit2(out)
            loss.backward()
            total = 0
            total += loss
            return total
        """)]
    g = PT.build(files, "loss")
    root = g.nodes[g.root]
    assert len(root.sites) == 2
    ups = {n for _s, n in names(g, "up")}
    assert {"out", "y", "x"} <= ups
    downs = {n for _s, n in names(g, "down")}
    assert "total" in downs
    text = PT.render(g)
    assert "loss.backward()" in text


def test_ok_self_attribute_shared_across_methods():
    files = [F("net.py", """
        class Net:
            def __init__(self, lr):
                self.lr = lr
            def step(self, g):
                upd = self.lr * g
                return upd
        """)]
    g = PT.build(files, "self.lr")
    assert "upd" in {n for _s, n in names(g, "down")}
    assert g.root[1] == "<class Net>"


def test_ok_parameter_hops_to_call_site_and_return_hops_back():
    files = [
        F("a.py", """
            from b import build
            def main():
                width = 64
                net = build(width)
                return net
            """),
        F("b.py", """
            def build(w):
                layer = make(w)
                return layer
            """),
    ]
    g = PT.build(files, "w", file_hint="b.py")
    assert "width" in {n for _s, n in names(g, "up")}
    g2 = PT.build(files, "net")
    assert "layer" in {n for _s, n in names(g2, "up")}


def test_ok_tuple_unpacking_and_for_loop():
    files = [F("t.py", """
        def run(loader):
            for xb, yb in loader:
                a, b = split(xb)
                c = a + yb
            return c
        """)]
    g = PT.build(files, "c")
    ups = {n for _s, n in names(g, "up")}
    assert {"a", "yb", "xb", "loader"} <= ups


def test_ok_cycles_terminate():
    files = [F("c.py", """
        def f(a):
            b = a
            a = b + 1
            b = a * 2
            return b
        """)]
    out = PT.trace(files, "b")
    assert out.startswith("TRACE")


def test_ok_syntax_error_file_is_skipped_and_unicode_names_work():
    files = [
        ("bad.py", "/p/bad.py", "def (:\n"),
        F("u.py", """
            def f(x):
                café = x * 2
                résultat = café + 1
                return résultat
            """),
    ]
    g = PT.build(files, "résultat")
    assert g is not None and "café" in {n for _s, n in names(g, "up")}


def test_ok_parse_target():
    assert PT.parse_target("self.lr:model.py:44") == ("self.lr", "model.py", 44, None)
    assert PT.parse_target("loss") == ("loss", None, None, None)
    assert PT.parse_target("")[3]
    assert PT.parse_target("a.b.c")[3]
    assert PT.parse_target("1abc")[3]


def test_ok_unknown_symbol_and_no_files():
    files = [F("t.py", "x = 1\n")]
    assert "never assigned" in PT.trace(files, "zzz")
    assert "no Python files" in PT.trace([], "x")


def test_ok_node_bound_is_respected():
    body = "\n".join(f"    v{i} = v{i-1} + 1" for i in range(1, 400))
    files = [("big.py", "/p/big.py", "def f(v0):\n" + body + "\n    return v399\n")]
    g = PT.build(files, "v200", max_depth=1000)
    assert len(g.nodes) <= PT.MAX_NODES
    assert g.truncated


def test_ok_project_files_entry_first_and_skips(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "v.py").write_text("y = 1\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("z = 1\n")
    entry = tmp_path / "pkg" / "m.py"
    files = PT.project_files(str(tmp_path), entry=str(entry))
    labels = [f[0] for f in files]
    assert labels[0] == "pkg/m.py"
    assert "a.py" in labels and not any(l.startswith("venv") for l in labels)
    assert len(labels) == len(set(labels))
