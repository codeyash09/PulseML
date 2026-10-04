"""
Shapes: what a variable's shape actually is, and whether it is what the code expects.

Pulse Code used to edit training code without ever seeing a tensor, so a change that
quietly broke a shape (a Linear fed 784 features where it was built for 256, a label vector
of (N,) meeting logits of (N, 1)) was only found when the user ran the whole job. This
module is what lets it look:

  describe(obj)                  the real shape / dtype / device of any array-like, or of the
                                 things inside a list, tuple, dict or torch module;
  check_shape(x, "(B, 10)")      compare against an expectation. B is a symbol: it binds to
                                 the first size it meets and every later use must agree, so
                                 check_shapes({...}) can state "the batch dimension is the
                                 same everywhere" in one line;
  ShapeTrace                     records every torch module's input/output shapes while code
                                 runs, so a mismatch is reported as "this Linear, receiving
                                 this shape" instead of a traceback into torch internals;
  explain_shape_error(message)   the usual shape-error messages, turned into what to look at;
  main()                         the small probe runner `pulse code` uses: run as a script it
                                 executes a snippet in the user's project, then reports.

Deliberately stdlib-only, and torch/numpy/etc. are only ever touched through duck typing or
if already imported: this file is run as a bare script inside the user's environment, and
`import pulse` must stay cheap.
"""
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

_MAX_ITEMS = 8            # elements / parameters / keys described per container
_MAX_DEPTH = 3
_MAX_TRACE_EVENTS = 400   # module calls kept while tracing (the tail is what matters)


# ---------------------------------------------------------------------------------------
# Describing
# ---------------------------------------------------------------------------------------


def _backend(obj: Any) -> str:
    root = (type(obj).__module__ or "").split(".")[0]
    return {"jaxlib": "jax", "jax": "jax", "keras": "keras", "tf_keras": "keras"}.get(root, root)


def _dim(value: Any) -> Any:
    """One dimension as a plain int, None (unknown/dynamic) or a string for a symbolic one."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return str(value)


def _shape_of(obj: Any) -> Optional[Tuple[Any, ...]]:
    shape = getattr(obj, "shape", None)
    if shape is None or callable(shape):
        return None
    try:
        if hasattr(shape, "as_list"):                    # tf.TensorShape; unknown rank raises
            return tuple(_dim(d) for d in shape.as_list())
        return tuple(_dim(d) for d in shape)
    except Exception:
        return None


def _short_dtype(obj: Any) -> Optional[str]:
    dtype = getattr(obj, "dtype", None)
    if dtype is None:
        return None
    text = str(getattr(dtype, "name", None) or dtype)
    return re.sub(r"^(torch|tf|tensorflow|numpy|jnp)\.", "", text)


def _device_of(obj: Any) -> Optional[str]:
    try:
        device = getattr(obj, "device", None)
        if device is not None and not callable(device):
            return str(device)
        devices = getattr(obj, "devices", None)         # jax arrays
        if callable(devices):
            found = sorted(str(d) for d in devices())
            return found[0] if len(found) == 1 else ",".join(found) if found else None
    except Exception:
        pass
    return None


def describe(obj: Any, _depth: int = 0) -> Dict[str, Any]:
    """A JSON-safe description of `obj`: {'kind': 'array'|'module'|'list'|'dict'|'scalar'|'other', ...}.

    Never evaluates the data, only its metadata, so it is safe on a tensor on a busy GPU."""
    shape = _shape_of(obj)
    if shape is not None and (hasattr(obj, "dtype") or hasattr(obj, "columns")):
        info: Dict[str, Any] = {"kind": "array", "type": f"{_backend(obj)}.{type(obj).__name__}",
                                "shape": list(shape), "dtype": _short_dtype(obj)}
        device = _device_of(obj)
        if device:
            info["device"] = device
        if getattr(obj, "requires_grad", False) is True:
            info["requires_grad"] = True
        if hasattr(obj, "columns") and not hasattr(obj, "dtype"):           # a DataFrame
            try:
                info["columns"] = [str(c) for c in list(obj.columns)[:_MAX_ITEMS]]
            except Exception:
                pass
        return info
    if callable(getattr(obj, "named_parameters", None)) and callable(getattr(obj, "forward", None)):
        return _describe_module(obj)
    if isinstance(obj, (list, tuple)):
        return _describe_sequence(obj, _depth)
    if isinstance(obj, dict):
        items = list(obj.items())
        out: Dict[str, Any] = {"kind": "dict", "len": len(items), "items": {}}
        if _depth < _MAX_DEPTH:
            for key, value in items[:_MAX_ITEMS]:
                out["items"][str(key)] = describe(value, _depth + 1)
        return out
    if isinstance(obj, (bool, int, float, complex, str, bytes, type(None))):
        text = repr(obj)
        return {"kind": "scalar", "type": type(obj).__name__, "value": text if len(text) <= 60 else text[:57] + "..."}
    return {"kind": "other", "type": f"{_backend(obj)}.{type(obj).__name__}"}


def _describe_sequence(obj: Any, depth: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kind": "list", "type": type(obj).__name__, "len": len(obj), "items": []}
    if depth >= _MAX_DEPTH:
        return out
    described = [describe(v, depth + 1) for v in list(obj)[:_MAX_ITEMS]]
    out["items"] = described
    uniform = (len(obj) > _MAX_ITEMS and bool(described) and all(
        d.get("kind") == "array" and d.get("shape") == described[0].get("shape")
        and d.get("dtype") == described[0].get("dtype") for d in described))
    if uniform:
        out["note"] = "the rest are assumed to match the first items shown"
    return out


def _describe_module(model: Any) -> Dict[str, Any]:
    params: List[Dict[str, Any]] = []
    total = 0
    try:
        for name, p in model.named_parameters():
            shape = _shape_of(p)
            if shape is not None:
                n = 1
                for d in shape:
                    n *= d if isinstance(d, int) else 1
                total += n
                if len(params) < _MAX_ITEMS * 3:
                    params.append({"name": name, "shape": list(shape), "dtype": _short_dtype(p)})
    except Exception:
        pass
    return {"kind": "module", "type": f"{_backend(model)}.{type(model).__name__}",
            "parameters": total, "params": params}


def _fmt_shape(shape: Any) -> str:
    if shape is None:
        return "?"
    parts = ["?" if d is None else str(d) for d in shape]
    return "(" + ", ".join(parts) + (",)" if len(parts) == 1 else ")")


def render(name: str, info: Dict[str, Any], indent: str = "") -> List[str]:
    """`describe()` output as readable lines."""
    kind = info.get("kind")
    if kind == "array":
        bits = [info.get("type", "array"), info.get("dtype") or "", _fmt_shape(info.get("shape"))]
        line = f"{indent}{name}: " + " ".join(b for b in bits if b)
        if info.get("device"):
            line += f" on {info['device']}"
        if info.get("requires_grad"):
            line += " requires_grad"
        if info.get("columns"):
            line += f" columns={info['columns']}"
        return [line]
    if kind == "module":
        lines = [f"{indent}{name}: {info.get('type')} with {info.get('parameters', 0):,} parameters"]
        for p in info.get("params", []):
            lines.append(f"{indent}    {p['name']}: {_fmt_shape(p['shape'])}")
        return lines
    if kind in ("list", "dict"):
        lines = [f"{indent}{name}: {info.get('type', 'dict')} of {info.get('len', 0)}"]
        children = info.get("items")
        pairs = children.items() if isinstance(children, dict) else enumerate(children or [])
        for key, child in pairs:
            lines.extend(render(f"[{key!r}]" if kind == "dict" else f"[{key}]", child, indent + "    "))
        if info.get("note"):
            lines.append(f"{indent}    ({info['note']})")
        return lines
    if kind == "scalar":
        return [f"{indent}{name}: {info.get('type')} = {info.get('value')}"]
    return [f"{indent}{name}: {info.get('type', 'object')}"]


# ---------------------------------------------------------------------------------------
# Expectations: "(B, 10)", "[N, *, 3]", "B, C, ..."
# ---------------------------------------------------------------------------------------

Token = Tuple[str, Any]        # ('int', 10) | ('sym', 'B') | ('any', None) | ('rest', None)
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_spec(spec: Any) -> List[Token]:
    """'(B, 10)' / '[B, C, H, W]' / 'N, *, 3' / '(B, ...)' / [None, 10] -> tokens.

    An int must match exactly; a name is a symbol (binds on first use, must agree after);
    `*`, `?` and `_` match any one dimension; `...` matches any number of dimensions (at most
    one per spec); `()` is a scalar."""
    if isinstance(spec, (tuple, list)):
        parts = ["?" if d is None else str(d) for d in spec]
    else:
        text = str(spec).strip()
        if len(text) >= 2 and text[0] in "([" and text[-1] in ")]":
            text = text[1:-1]
        parts = [p.strip() for p in text.split(",")]
        if parts and parts[-1] == "":
            parts.pop()                                  # "(10,)" is a 1-D shape
        if len(parts) == 1 and parts[0] == "":
            parts = []
    tokens: List[Token] = []
    for part in parts:
        if part in ("*", "?", "_"):
            tokens.append(("any", None))
        elif part in ("...", "…"):
            if any(t[0] == "rest" for t in tokens):
                raise ValueError("a shape spec can contain '...' only once")
            tokens.append(("rest", None))
        elif re.fullmatch(r"-?[0-9]+", part):
            if int(part) < 0:
                raise ValueError(f"'{part}': a dimension cannot be negative")
            tokens.append(("int", int(part)))
        elif _SYMBOL_RE.match(part):
            tokens.append(("sym", part))
        else:
            raise ValueError(f"cannot read '{part}' in the shape spec {spec!r}: use a number, a name "
                             "like B, '*' for any size, or '...' for any number of dims")
    return tokens


def _spec_text(tokens: List[Token]) -> str:
    names = {"any": "*", "rest": "..."}
    parts = [str(v) if k in ("int", "sym") else names[k] for k, v in tokens]
    return "(" + ", ".join(parts) + (",)" if len(parts) == 1 and tokens[0][0] != "rest" else ")")


def match_shape(shape: Any, tokens: List[Token], env: Optional[Dict[str, Tuple[int, str]]] = None,
                who: str = "this") -> List[str]:
    """Problems with `shape` against `tokens` (empty list = it matches). `env` maps each symbol
    to (size, who bound it) and is updated, so one dict carried across calls enforces that a
    name means the same size everywhere."""
    env = {} if env is None else env
    if shape is None:
        return [f"{who} has no shape to check (it is not an array)"]
    shape = list(shape)
    rest_at = next((i for i, t in enumerate(tokens) if t[0] == "rest"), None)
    fixed = len(tokens) - (1 if rest_at is not None else 0)
    if rest_at is None and len(shape) != fixed:
        return [f"expected {fixed} dimension{'s' if fixed != 1 else ''} {_spec_text(tokens)} but {who} has "
                f"{len(shape)}: {_fmt_shape(shape)}"]
    if rest_at is not None and len(shape) < fixed:
        return [f"expected at least {fixed} dimension{'s' if fixed != 1 else ''} {_spec_text(tokens)} but "
                f"{who} has {len(shape)}: {_fmt_shape(shape)}"]
    pairs: List[Tuple[int, Token, Any]] = []
    if rest_at is None:
        pairs = [(i, tokens[i], shape[i]) for i in range(len(shape))]
    else:
        after = len(tokens) - rest_at - 1
        pairs = [(i, tokens[i], shape[i]) for i in range(rest_at)]
        pairs += [(len(shape) - after + j, tokens[rest_at + 1 + j], shape[len(shape) - after + j])
                  for j in range(after)]
    problems: List[str] = []
    for axis, (kind, want), got in pairs:
        if kind == "any" or got is None or not isinstance(got, int):
            continue                                      # unknown/dynamic sizes match anything
        if kind == "int" and got != want:
            problems.append(f"dimension {axis}: expected {want}, got {got}")
        elif kind == "sym":
            if want in env and env[want][0] != got:
                size, by = env[want]
                problems.append(f"dimension {axis}: {want} is {size} (from {by}) but {who} has {got}")
            else:
                env.setdefault(want, (got, who))
    return problems


class ShapeMismatch(AssertionError):
    """Raised by check_shape(..., raise_on_mismatch=True)."""


class ShapeReport:
    def __init__(self, name: str, info: Dict[str, Any], expected: Optional[str], problems: List[str]):
        self.name, self.info, self.expected, self.problems = name, info, expected, problems

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def shape(self) -> Optional[Tuple[Any, ...]]:
        shape = self.info.get("shape")
        return tuple(shape) if shape is not None else None

    def render(self) -> str:
        lines = render(self.name, self.info)
        if self.expected is not None:
            lines.append("  " + ("matches " + self.expected if self.ok else
                                 f"DOES NOT MATCH {self.expected}: " + "; ".join(self.problems)))
        return "\n".join(lines)

    __str__ = render

    def __bool__(self) -> bool:
        return self.ok


def check_shape(value: Any, expected: Any = None, name: Optional[str] = None, *,
                env: Optional[Dict[str, Tuple[int, str]]] = None,
                raise_on_mismatch: bool = False) -> ShapeReport:
    """Describe `value`'s shape and, if `expected` is given, check it.

        check_shape(logits, "(B, 10)")
        check_shape(x, "(B, 3, 224, 224)", raise_on_mismatch=True)

    Returns a ShapeReport (truthy when it matches). Pass the same `env` dict to several calls,
    or use check_shapes, to require a symbol like B to be the same size in all of them."""
    label = name or "value"
    info = describe(value)
    problems: List[str] = []
    spec_text = None
    if expected is not None:
        tokens = parse_spec(expected)
        spec_text = _spec_text(tokens)
        problems = match_shape(info.get("shape") if info.get("kind") == "array" else None, tokens, env, label)
    report = ShapeReport(label, info, spec_text, problems)
    if raise_on_mismatch and problems:
        raise ShapeMismatch(report.render())
    return report


def check_shapes(checks: Dict[str, Tuple[Any, Any]], *, raise_on_mismatch: bool = False) -> List[ShapeReport]:
    """check_shapes({'x': (x, '(B, 784)'), 'y': (y, '(B,)')}): every spec checked against one
    shared symbol table, so both B's must be the same size."""
    env: Dict[str, Tuple[int, str]] = {}
    reports = [check_shape(value, spec, name, env=env) for name, (value, spec) in checks.items()]
    bad = [r for r in reports if not r.ok]
    if raise_on_mismatch and bad:
        raise ShapeMismatch("\n".join(r.render() for r in reports))
    return reports


# ---------------------------------------------------------------------------------------
# Shape errors, explained
# ---------------------------------------------------------------------------------------

_HINTS = [
    (re.compile(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)"),
     lambda m: f"a Linear layer built for {m[3]} input features received {m[2]} (its input is "
               f"{m[1]} rows x {m[2]} features). Fix the layer's in_features, or the shape of what feeds it "
               "(a missing flatten, or a conv output of a different size, are the usual causes)."),
    (re.compile(r"shapes \(([\d, ]*)\) and \(([\d, ]*)\) not aligned: (\d+) \(dim (-?\d+)\) != (\d+) \(dim (-?\d+)\)"),
     lambda m: f"matrix product of {m[1]} and {m[2]}: the inner sizes must be equal but are {m[3]} and {m[5]}."),
    (re.compile(r"mismatch in its core dimension (\d+).*?\(size (\d+) is different from (\d+)\)"),
     lambda m: f"matrix product: the inner sizes must be equal but are {m[2]} (dimension {m[1]} of the second "
               f"operand) and {m[3]} (last dimension of the first). A layer built for {m[2]} inputs is being "
               f"fed {m[3]}."),
    (re.compile(r"operands could not be broadcast together with shapes ([\d, ()]+?) ([\d, ()]+?)(?:\s|$)"),
     lambda m: f"{m[1]} and {m[2]} do not broadcast: compare the sizes from the last dimension backwards; "
               "each pair must be equal or 1."),
    (re.compile(r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\) at non-singleton "
                r"dimension (-?\d+)"),
     lambda m: f"dimension {m[3]} is {m[1]} in one operand and {m[2]} in the other. If one of them should have "
               "been broadcast, it is probably missing a unsqueeze/keepdim."),
    (re.compile(r"Expected input batch_size \((\d+)\) to match target batch_size \((\d+)\)"),
     lambda m: f"the inputs have batch {m[1]} but the targets have {m[2]}: the model output was probably "
               "reshaped or flattened somewhere between the batch and the loss."),
    (re.compile(r"expected input\[([\d, ]+)\] to have (\d+) channels, but got (\d+) channels"),
     lambda m: f"a conv layer was built for {m[2]} input channels but its input has {m[3]} "
               f"(input shape [{m[1]}]): check in_channels, or whether the data is channels-last."),
    (re.compile(r"shape '\[([-\d, ]+)\]' is invalid for input of size (\d+)"),
     lambda m: f"cannot view/reshape {m[2]} elements as [{m[1]}]: the product of the sizes must equal {m[2]}."),
    (re.compile(r"(?:Incompatible shapes|Dimensions must be equal|Incompatible shape)[^\n]*"),
     lambda m: "TensorFlow shape mismatch: compare the two shapes it prints, dimension by dimension."),
]


def explain_shape_error(message: str) -> Optional[str]:
    """What a common shape-error message means for the code, or None if it is not one we know."""
    for pattern, make in _HINTS:
        match = pattern.search(message or "")
        if match:
            try:
                return make((match.group(0),) + match.groups())
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------------------
# Tracing a torch forward pass
# ---------------------------------------------------------------------------------------


def _shapes_in(obj: Any, limit: int = 4) -> List[str]:
    """Shapes of the tensors in a module's args/output (nested tuples/lists/dicts)."""
    found: List[str] = []

    def walk(o: Any, depth: int = 0) -> None:
        if len(found) >= limit or depth > 3:
            return
        shape = _shape_of(o)
        if shape is not None and hasattr(o, "dtype"):
            found.append(_fmt_shape(shape))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, depth + 1)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)

    walk(obj)
    return found


class ShapeTrace:
    """Context manager: while active, every torch.nn.Module call is recorded with its input and
    output shapes. If code raises inside a module's forward, `failed` names the module that was
    running (innermost first) -- the layer that received the wrong shape, not the torch
    internals the traceback ends in. A no-op, with `available` False, when torch is not imported
    or is too old for global module hooks."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.calls = 0
        self._stack: List[Dict[str, Any]] = []
        self._handles: List[Any] = []
        self.available = False

    def __enter__(self) -> "ShapeTrace":
        torch = sys.modules.get("torch")
        nn_module = getattr(getattr(torch, "nn", None), "modules", None)
        mod = getattr(nn_module, "module", None)
        pre, post = getattr(mod, "register_module_forward_pre_hook", None), getattr(mod, "register_module_forward_hook", None)
        if not (callable(pre) and callable(post)):
            return self
        try:
            self._handles = [pre(self._enter), post(self._exit)]
            self.available = True
        except Exception:
            self._handles = []
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    @staticmethod
    def _label(module: Any) -> str:
        name = type(module).__name__
        try:
            extra = module.extra_repr()
        except Exception:
            extra = ""
        return f"{name}({extra})" if extra else name

    def _enter(self, module: Any, args: Any) -> None:
        self._stack.append({"module": self._label(module), "in": _shapes_in(args),
                            "leaf": not getattr(module, "_modules", None)})

    def _exit(self, module: Any, args: Any, output: Any) -> None:
        entry = self._stack.pop() if self._stack else {"module": self._label(module), "in": _shapes_in(args),
                                                         "leaf": True}
        entry["out"] = _shapes_in(output)
        self.calls += 1
        self.events.append(entry)
        if len(self.events) > _MAX_TRACE_EVENTS:
            del self.events[: len(self.events) - _MAX_TRACE_EVENTS]

    def report(self, tail: int = 6) -> Dict[str, Any]:
        leaves = [e for e in self.events if e["leaf"]]
        out: Dict[str, Any] = {"available": self.available, "module_calls": self.calls,
                               "last_ok": leaves[-tail:]}
        if self._stack:
            out["failed"] = {"module": self._stack[-1]["module"], "in": self._stack[-1]["in"],
                             "inside": [s["module"] for s in self._stack[:-1]]}
        return out


def render_trace(trace: Dict[str, Any]) -> List[str]:
    if not trace.get("available"):
        return ["  (no torch module trace: torch was not imported by the probe)"]
    lines: List[str] = []
    if trace.get("last_ok"):
        lines.append(f"  layers that ran ({trace.get('module_calls', 0)} module calls; last {len(trace['last_ok'])}):")
        for e in trace["last_ok"]:
            lines.append(f"    {e['module']}: {', '.join(e['in']) or '-'} -> {', '.join(e.get('out', [])) or '-'}")
    failed = trace.get("failed")
    if failed:
        lines.append(f"  FAILED INSIDE: {failed['module']} receiving {', '.join(failed['in']) or '(no tensor input)'}")
        if failed.get("inside"):
            lines.append("    called from: " + " > ".join(failed["inside"]))
    return lines


# ---------------------------------------------------------------------------------------
# The probe runner (python pulse_shapes.py request.json)
# ---------------------------------------------------------------------------------------


def _shaped_names(ns: Dict[str, Any]) -> List[str]:
    names = []
    for key, value in ns.items():
        if key.startswith("_") or isinstance(value, type(sys)) or callable(value) and not hasattr(value, "shape"):
            continue
        if _shape_of(value) is not None or callable(getattr(value, "named_parameters", None)) \
                or isinstance(value, (list, tuple, dict)):
            names.append(key)
    return names


def _error_info(exc: BaseException, root: str, source: str = "") -> Dict[str, Any]:
    frames = traceback.extract_tb(exc.__traceback__)
    probe_lines = source.splitlines()
    mine = [f for f in frames if f.filename == "<probe>" or os.path.abspath(f.filename).startswith(root + os.sep)]
    shown = (mine or frames)[-6:]
    message = str(exc)
    info: Dict[str, Any] = {
        "type": type(exc).__name__, "message": message[:1500],
        "frames": [f"{os.path.relpath(f.filename, root) if f.filename != '<probe>' else '<probe>'}:{f.lineno} "
                   f"in {f.name}: " + ((probe_lines[f.lineno - 1] if f.filename == '<probe>' and 0 < f.lineno <= len(probe_lines)
                                       else f.line) or "").strip() for f in shown]}
    hint = explain_shape_error(message)
    if hint:
        info["hint"] = hint
    return info


def run_probe(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute request['code'] (in the current directory, with it importable), then describe
    request['exprs'] (default: every array/module/list/dict the code left behind) and check
    request['expect'] ({expr: spec}) with one shared symbol table. Variables are described even
    if the code failed part-way: what exists at the point of failure is the evidence."""
    root = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    ns: Dict[str, Any] = {"__name__": "__pulse_probe__"}
    report: Dict[str, Any] = {"error": None, "vars": [], "expect": [], "trace": None}
    tracer = ShapeTrace()
    with tracer:
        try:
            exec(compile(request.get("code") or "", "<probe>", "exec"), ns)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                report["error"] = {"type": "SystemExit", "message": f"the probe called exit({exc.code!r})", "frames": []}
        except BaseException as exc:                     # noqa: BLE001 -- the probe's failure is the result
            report["error"] = _error_info(exc, root, request.get("code") or "")
    report["trace"] = tracer.report()
    if report["error"] is None:
        report["trace"].pop("failed", None)      # an exception the probe itself caught is not a failure

    exprs = request.get("exprs") or _shaped_names(ns)
    for expr in exprs[:40]:
        try:
            value = eval(compile(str(expr), "<expr>", "eval"), ns)
            report["vars"].append({"name": str(expr), "info": describe(value)})
        except BaseException as exc:                     # noqa: BLE001
            report["vars"].append({"name": str(expr), "error": f"{type(exc).__name__}: {exc}"[:300]})

    env: Dict[str, Tuple[int, str]] = {}
    for expr, spec in (request.get("expect") or {}).items():
        try:
            value = eval(compile(str(expr), "<expr>", "eval"), ns)
            r = check_shape(value, spec, str(expr), env=env)
            report["expect"].append({"name": str(expr), "spec": r.expected, "ok": r.ok, "problems": r.problems,
                                     "shape": list(r.shape) if r.shape is not None else None})
        except ValueError as exc:
            report["expect"].append({"name": str(expr), "spec": str(spec), "ok": False, "problems": [str(exc)]})
        except BaseException as exc:                     # noqa: BLE001
            report["expect"].append({"name": str(expr), "spec": str(spec), "ok": False,
                                     "problems": [f"could not evaluate: {type(exc).__name__}: {exc}"[:300]]})
    return report


def render_report(report: Dict[str, Any]) -> str:
    """The probe's report as the text the agent reads."""
    lines: List[str] = []
    error = report.get("error")
    if error:
        lines.append(f"PROBE FAILED: {error['type']}: {error['message']}")
        for frame in error.get("frames", []):
            lines.append(f"    {frame}")
        if error.get("hint"):
            lines.append(f"  What it means: {error['hint']}")
        lines.extend(render_trace(report.get("trace") or {}) if (report.get("trace") or {}).get("failed") or
                     (report.get("trace") or {}).get("last_ok") else [])
    elif (report.get("trace") or {}).get("available") and report["trace"].get("module_calls"):
        lines.extend(render_trace(report["trace"]))
    variables = report.get("vars") or []
    if variables:
        lines.append("variables" + (" (as they were when it failed):" if error else ":"))
        for v in variables:
            if "error" in v:
                lines.append(f"  {v['name']}: could not be evaluated ({v['error']})")
            else:
                lines.extend(render(v["name"], v["info"], "  "))
    checks = report.get("expect") or []
    if checks:
        lines.append("expectations:")
        for c in checks:
            shape = _fmt_shape(c["shape"]) if c.get("shape") is not None else "?"
            lines.append(f"  {'OK  ' if c['ok'] else 'FAIL'} {c['name']} {shape} vs {c.get('spec')}"
                         + ("" if c["ok"] else ": " + "; ".join(c["problems"])))
    if not lines:
        lines.append("the probe ran; there was nothing with a shape to report")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: pulse_shapes.py REQUEST.json", file=sys.stderr)
        return 2
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]   # not our own folder first
    with open(argv[0], encoding="utf-8") as handle:
        request = json.load(handle)
    report = run_probe(request)
    target = request.get("report_path")
    text = json.dumps(report)
    if target:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())