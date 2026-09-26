"""
TRACE -- the variable influence graph: what feeds a variable, and what it feeds.

The question this answers is the one you actually ask at 2am: "this number is wrong --
where did it come from, and what else is already wrong because of it?" Answering it by
reading code means holding a dozen assignments in your head at once and missing the one
that happens in another file; answering it by asking the model means trusting a guess
about code it may only have seen half of. So Pulse computes it instead, from the real
AST of the real project files, and prints the whole connected path.

What it follows, and why those and not others:

  * assignment chains inside a scope -- `loss = criterion(out, y)` means `out` and `y`
    feed `loss`, and whatever fed them feeds it too. Every branch of an if/for/try is
    walked, and a variable assigned in more than one place keeps ALL of its assignment
    sites instead of Pulse silently picking one: "assigned here OR here" is usually the
    answer to why a value is sometimes wrong.
  * across functions, in both directions -- a parameter's real source is the argument at
    its call sites, and a `return` lands in whatever the caller assigned it to. A slice
    that stopped at the function boundary would stop exactly where most real ML bugs live
    (a learning rate threaded through three layers of setup, a tensor built in a
    dataloader and used in a loss).
  * `self.<attr>` as a first-class variable, shared across every method of its class,
    because in a `nn.Module` that is where most of the interesting state lives. An
    attribute of anything else resolves to the object itself (`cfg.lr` -> `cfg`), which
    is the thing worth tracing.
  * uses that produce no new variable (`loss.backward()`, `print(acc)`) are reported as
    what they are -- where the value is consumed -- rather than dropped for not fitting
    the graph.

What it does not do: this is a static, heuristic slice, not a runtime taint tracker. It
does not evaluate anything, does not know which branch actually ran, and does not resolve
values through a dict, a list, `getattr`, or a call into a library it cannot see. It is
meant to point straight at where a bad value came from, and it says so out loud (as
"external to the traced code") when the trail leaves the project rather than guessing.

Everything is bounded -- depth, sites per variable, parents/children per node, total
nodes -- so a pathological file produces a readable answer instead of a wall of text.

Used by:
  * the debugger's agent, as the TRACE: directive (PulseCLI._run_trace)
  * `/trace <var>` in the debugger prompt, in `pulse code`, and in `pulse watch`
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------------------
# Bounds. A trace is a debugging aid printed into a terminal, so "complete" matters less
# than "readable and finite": a 4000-line training script with a variable threaded through
# every function must not produce 4000 lines of output.
# ---------------------------------------------------------------------------------------
MAX_DEPTH = 6            # hops away from the traced variable, each direction
MAX_SITES = 4            # assignment sites kept for one variable (it may be assigned in many branches)
MAX_PARENTS = 6          # upstream variables followed from one assignment
MAX_CHILDREN = 6         # downstream variables followed from one variable
MAX_EFFECTS = 4          # non-assigning uses reported per variable
MAX_CALLERS = 3          # call sites followed when hopping out of a parameter
MAX_NODES = 150          # total variables in one graph
MAX_SNIPPET = 96         # characters of source shown per line

# Names not worth treating as an interesting upstream cause: builtins and constants that
# appear on the right-hand side of countless assignments without explaining anything.
IGNORED_NAMES: FrozenSet[str] = frozenset({
    "self", "cls", "True", "False", "None", "len", "int", "float", "str", "list", "dict",
    "tuple", "set", "range", "print", "super", "type", "isinstance", "min", "max", "sum",
    "abs", "round", "enumerate", "zip", "map", "filter", "open", "sorted", "reversed",
    "bool", "bytes", "getattr", "setattr", "hasattr", "format", "repr", "any", "all",
    "NotImplemented", "Ellipsis",
})

MODULE_SCOPE = "<module>"

Key = Tuple[str, str, str]   # (file label, scope key, variable name)


# ---------------------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------------------
@dataclass
class Site:
    """One place a variable gets its value. A variable assigned in two branches has two
    of these, and both are shown -- which one ran is exactly what a static slice cannot
    know, and pretending otherwise is how a trace points at the wrong line."""
    label: str
    scope: str
    lineno: Optional[int]
    snippet: str
    kind: str                                  # assign | aug | loop | with | param | external
    parents: List[Key] = field(default_factory=list)
    note: str = ""                             # e.g. "via build_model() -> return"


@dataclass
class Edge:
    """One place a variable is used, and the variable (if any) that use produces."""
    label: str
    lineno: int
    snippet: str
    target: Optional[Key]                      # None for a use that assigns nothing
    note: str = ""


@dataclass
class Node:
    key: Key
    name: str
    sites: List[Site] = field(default_factory=list)
    out: List[Edge] = field(default_factory=list)
    more_sites: int = 0                        # sites that existed but were not kept
    more_out: int = 0
    depth_up: Optional[int] = None             # hops from the traced variable, upstream
    depth_down: Optional[int] = None

    @property
    def scope(self) -> str:
        return self.key[1]


@dataclass
class Graph:
    symbol: str
    root: Key
    nodes: Dict[Key, Node]
    origin: str                                # "train.py:117 in train_one_epoch"
    truncated: bool = False                    # a bound was hit somewhere
    notes: List[str] = field(default_factory=list)

    def upstream_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.key != self.root and n.depth_up is not None)

    def downstream_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.key != self.root and n.depth_down is not None)

    def files(self) -> List[str]:
        return sorted({k[0] for k in self.nodes})


# ---------------------------------------------------------------------------------------
# The project index: scopes, calls and imports, computed once per trace
# ---------------------------------------------------------------------------------------
@dataclass
class Scope:
    """A function body, or a module's top level. Its *own* statements only -- a nested
    function's body belongs to that function's scope, not to this one, because its locals
    are a different set of variables that happen to share a file."""
    label: str
    path: str
    name: str                                  # qualified: "train", "Net.forward", "<module>"
    key_scope: str                             # what a Key uses (same as name for functions)
    func: Optional[ast.AST]                    # FunctionDef/AsyncFunctionDef, or None for a module
    class_name: Optional[str]
    params: List[str]
    stmts: List[ast.stmt]
    assigned: Set[str] = field(default_factory=set)
    parent: Optional["Scope"] = None           # the enclosing function/module (a class is not a scope)
    globals: Set[str] = field(default_factory=set)      # `global x`: x is the module's
    nonlocals: Set[str] = field(default_factory=set)    # `nonlocal x`: x is an enclosing function's

    @property
    def is_module(self) -> bool:
        return self.func is None


@dataclass
class CallSite:
    call: ast.Call
    callee: str                                # the simple name called: f, obj.f -> "f"
    scope: Scope                               # where the call happens
    stmt: ast.stmt                             # the statement containing it


class Index:
    """Everything a trace needs to look up, built once from (label, path, text) triples."""

    def __init__(self, files: Sequence[Tuple[str, str, str]]):
        self.lines: Dict[str, List[str]] = {}
        self.scopes: List[Scope] = []
        self.module_scope: Dict[str, Scope] = {}
        self.by_key: Dict[Tuple[str, str], List[Scope]] = {}
        self.funcs_by_name: Dict[str, List[Scope]] = {}
        self.classes: Dict[Tuple[str, str], List[Scope]] = {}
        self.import_aliases: Dict[str, Set[str]] = {}
        self.class_bases: Dict[Tuple[str, str], List[str]] = {}
        self.attr_owner: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
        self.calls: List[CallSite] = []
        self.parsed = 0

        for label, path, text in files:
            try:
                tree = ast.parse(text, filename=path or label)
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                # A file mid-edit -- or a generated one too deeply nested for the parser --
                # is not a reason to fail the trace.
                continue
            try:
                aliases = _import_aliases(tree)
                first = len(self.scopes)
                self._index_scope(label, path, tree, name=MODULE_SCOPE, class_name=None)
            except (RecursionError, MemoryError):
                del self.scopes[first:]
                continue
            self.parsed += 1
            self.lines[label] = text.splitlines()
            self.import_aliases[label] = aliases

        for scope in self.scopes:
            self.by_key.setdefault((scope.label, scope.key_scope), []).append(scope)
            if scope.is_module:
                self.module_scope[scope.label] = scope
            else:
                self.funcs_by_name.setdefault(_simple_name(scope.name), []).append(scope)
            if scope.class_name:
                self.classes.setdefault((scope.label, scope.class_name), []).append(scope)
        self._index_calls()

    # -- construction -------------------------------------------------------------------

    def _index_scope(self, label: str, path: str, node: ast.AST, name: str,
                     class_name: Optional[str], parent: Optional[Scope] = None) -> None:
        """Register `node` as a scope and recurse into the functions and classes it holds."""
        body = list(getattr(node, "body", []))
        stmts = _own_statements(body)
        params: List[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = _parameters(node)
        scope = Scope(
            label=label, path=path, name=name,
            key_scope=name,
            func=node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
            class_name=class_name, params=params, stmts=stmts, parent=parent,
        )
        for stmt in stmts:
            scope.assigned.update(targets_of(stmt))
            if isinstance(stmt, ast.Global):
                scope.globals.update(stmt.names)
            elif isinstance(stmt, ast.Nonlocal):
                scope.nonlocals.update(stmt.names)
        self.scopes.append(scope)

        # Nested definitions get their own scope. A class contributes no scope of its own
        # (its body is mostly defs); its methods carry its name so `self.x` can be resolved
        # across all of them.
        for child in _nested_definitions(body):
            if isinstance(child, ast.ClassDef):
                self.class_bases.setdefault((label, child.name), []).extend(
                    b.id if isinstance(b, ast.Name) else b.attr
                    for b in child.bases if isinstance(b, (ast.Name, ast.Attribute)))
                for grandchild in _nested_definitions(child.body):
                    if isinstance(grandchild, ast.ClassDef):
                        continue               # a class inside a class: rare, not worth the depth
                    self._index_scope(label, path, grandchild,
                                      name=f"{child.name}.{grandchild.name}",
                                      class_name=child.name, parent=scope)
            else:
                qualified = child.name if name == MODULE_SCOPE else f"{name}.{child.name}"
                self._index_scope(label, path, child, name=qualified, class_name=class_name,
                                  parent=scope)

    def _index_calls(self) -> None:
        for scope in self.scopes:
            for stmt in scope.stmts:
                for part in read_parts(stmt):
                    for node in ast.walk(part):
                        if not isinstance(node, ast.Call):
                            continue
                        callee = _called_name(node)
                        if callee:
                            self.calls.append(CallSite(node, callee, scope, stmt))

    # -- lookup -------------------------------------------------------------------------

    def scopes_for(self, key: Key) -> List[Scope]:
        """Every scope a key's variable can live in. One function, one module -- or, for
        `self.x`, every method of its class, since that is one variable however many
        methods touch it."""
        label, scope_key, _name = key
        return self.by_key.get((label, scope_key), [])

    def snippet(self, label: str, lineno: Optional[int]) -> str:
        lines = self.lines.get(label) or []
        if lineno is None or not (0 < lineno <= len(lines)):
            return ""
        text = lines[lineno - 1].strip()
        return text if len(text) <= MAX_SNIPPET else text[:MAX_SNIPPET - 1] + "…"

    def local_function(self, name: str) -> Optional[Scope]:
        """The project-local function `name` refers to, when there is exactly one -- the
        only case where hopping into a callee's body is unambiguous enough to be worth it."""
        hits = self.funcs_by_name.get(name) or []
        return hits[0] if len(hits) == 1 else None

    def callers_of(self, scope: Scope) -> List[CallSite]:
        if scope.is_module:
            return []
        simple = _simple_name(scope.name)
        if simple == "__init__" and scope.class_name:
            # A constructor is called by the class's name (`Opt(0.1)`), or from a subclass
            # through `super().__init__(...)` -- not by every `__init__` call anywhere.
            return [c for c in self.calls if c.scope is not scope
                    and (c.callee == scope.class_name or self._super_init_of(c, scope))]
        return [c for c in self.calls if c.callee == simple and c.scope is not scope]

    def _super_init_of(self, call_site: CallSite, init: Scope) -> bool:
        """`super().__init__(...)` in a class whose bases include init's class."""
        func = call_site.call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "__init__"
                and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"):
            return False
        caller_class = call_site.scope.class_name
        if not caller_class:
            return False
        return init.class_name in self.class_bases.get((call_site.scope.label, caller_class), [])


# ---------------------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------------------
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_TRY_NODES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())   # except* (3.11+)
_MATCH = getattr(ast, "Match", None)                                            # 3.10+


def _own_statements(body: Sequence[ast.stmt]) -> List[ast.stmt]:
    """Flatten a body into one statement list, descending through if/for/while/try/with
    (every branch) but NOT into a nested def or class -- those are separate scopes. Order
    is source order, which is close enough to execution order for straight-line and
    lightly-branching code and is not claimed to be exact for anything else."""
    flat: List[ast.stmt] = []
    for stmt in body:
        flat.append(stmt)
        if isinstance(stmt, _DEF_NODES):
            continue
        for attr in ("body", "orelse", "finalbody"):
            flat.extend(_own_statements(getattr(stmt, attr, None) or []))
        for handler in getattr(stmt, "handlers", None) or []:
            if isinstance(handler, ast.ExceptHandler):
                flat.extend(_own_statements(handler.body))
        for case in getattr(stmt, "cases", None) or []:     # match / case
            flat.extend(_own_statements(case.body))
    return flat


def _nested_definitions(body: Sequence[ast.stmt]) -> List[ast.stmt]:
    """Functions and classes defined anywhere in `body`, including inside an if/try (a
    conditional import shim or a `if TYPE_CHECKING` block still defines real functions)."""
    found: List[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, _DEF_NODES):
            found.append(stmt)
            continue
        for attr in ("body", "orelse", "finalbody"):
            found.extend(_nested_definitions(getattr(stmt, attr, None) or []))
        for handler in getattr(stmt, "handlers", None) or []:
            if isinstance(handler, ast.ExceptHandler):
                found.extend(_nested_definitions(handler.body))
        for case in getattr(stmt, "cases", None) or []:
            found.extend(_nested_definitions(case.body))
    return found


def _parameters(func: ast.AST) -> List[str]:
    args = func.args
    names = [a.arg for a in list(getattr(args, "posonlyargs", []) or []) + list(args.args)]
    if args.vararg:
        names.append(args.vararg.arg)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _positional_params(func: Optional[ast.AST]) -> Tuple[List[str], Optional[str]]:
    """The parameters a positional argument can fill, in order, and the *args name. Past
    those, a positional argument goes into *args; keyword-only parameters and **kwargs are
    never filled positionally."""
    if func is None:
        return [], None
    args = func.args
    positional = [a.arg for a in list(getattr(args, "posonlyargs", []) or []) + list(args.args)]
    return positional, (args.vararg.arg if args.vararg else None)


def _import_aliases(tree: ast.AST) -> Set[str]:
    """Every name this file binds by importing it. Those are modules and library symbols,
    not the user's variables, so following them upstream just leads out of the project."""
    aliases: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                aliases.add((alias.asname or alias.name).split(".")[0])
    return aliases


def _simple_name(qualified: str) -> str:
    return qualified.rsplit(".", 1)[-1]


def _called_name(call: ast.Call) -> Optional[str]:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _target_names(node: ast.expr) -> List[str]:
    """The variable a target expression writes to.

    `x` -> x. `a, b` -> a, b. `self.w` -> self.w (class state is a real variable worth
    following). `cfg.lr` -> cfg and `buf[i]` -> buf, because writing through an object
    changes that object, and the object is the thing the rest of the code shares."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: List[str] = []
        for element in node.elts:
            out.extend(_target_names(element))
        return out
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id in ("self", "cls"):
                return [f"{node.value.id}.{node.attr}"]
            return [node.value.id]
        # `cfg.optim.lr = x` writes through cfg.optim, i.e. into cfg.
        return _target_names(node.value) if isinstance(node.value, (ast.Attribute, ast.Subscript)) else []
    if isinstance(node, ast.Subscript):
        return _target_names(node.value)
    return []


def targets_of(stmt: ast.stmt) -> List[str]:
    """Every variable a statement assigns to, including a for-loop's own variable and a
    `with ... as` name."""
    targets: List[ast.expr] = []
    if isinstance(stmt, ast.Assign):
        targets = list(stmt.targets)
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        targets = [i.optional_vars for i in stmt.items if i.optional_vars is not None]
    elif _MATCH is not None and isinstance(stmt, _MATCH):
        # `case Point(x=px)` / `case [first, *rest]` / `case {"k": v, **extra}` bind names.
        for case in stmt.cases:
            for node in ast.walk(case.pattern):
                for attr in ("name", "rest"):
                    bound = getattr(node, attr, None)
                    if isinstance(bound, str):
                        targets.append(ast.Name(id=bound, ctx=ast.Store()))
    # `(batch := next(it))` assigns batch in the enclosing scope, wherever it appears.
    if not isinstance(stmt, _DEF_NODES):
        for part in read_parts(stmt):
            for node in ast.walk(part):
                if isinstance(node, ast.NamedExpr):
                    targets.append(node.target)
    names: List[str] = []
    for target in targets:
        for name in _target_names(target):
            if name not in names:
                names.append(name)
    return names


def read_parts(stmt: ast.stmt) -> List[ast.AST]:
    """The parts of a statement that read values.

    A compound statement contributes only its own header (a `for`'s iterable, an `if`'s
    test): its body is in the scope's statement list separately, and walking it here would
    credit every line of the body to the `for` line."""
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [i.context_expr for i in stmt.items]
    if isinstance(stmt, _TRY_NODES):
        return []                      # try / except*: every block is in the statement list
    if _MATCH is not None and isinstance(stmt, _MATCH):
        return [stmt.subject] + [c.guard for c in stmt.cases if c.guard is not None]
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        defaults = [d for d in stmt.args.defaults if d is not None]
        defaults += [d for d in stmt.args.kw_defaults if d is not None]
        return list(stmt.decorator_list) + defaults
    if isinstance(stmt, ast.ClassDef):
        return list(stmt.decorator_list) + list(stmt.bases)
    return [stmt]      # a simple statement: Store/Del names are excluded by their ctx


def value_parts(stmt: ast.stmt) -> List[ast.AST]:
    """Only the value side of an assignment -- what actually *produces* the new value, as
    opposed to every name the statement mentions. `buf[i] = x` reads `buf` and `i`, but
    what feeds the write is `x`."""
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        parts: List[ast.AST] = [stmt.value] if stmt.value is not None else []
        if isinstance(stmt, ast.AugAssign):
            parts.append(stmt.target)          # x += y depends on x's previous value too
        return parts
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [i.context_expr for i in stmt.items]
    return read_parts(stmt)


def _reads(parts: Sequence[ast.AST], scope: Scope, index: Index) -> List[str]:
    """Variable names read in `parts`, in source order, deduplicated.

    `self.x`/`cls.x` come back dotted (that attribute is the variable). An attribute of
    anything else comes back as the object (`cfg.lr` -> `cfg`). A name that is only ever
    the function being called, and is a known project function or an import, is dropped:
    `out = build(x)` is about `x`, not about `build`."""
    dotted: Dict[int, str] = {}
    consumed: Set[int] = set()
    called: Set[int] = set()
    for root in parts:
        for node in ast.walk(root):
            _mark_inner_bindings(node, consumed)
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")):
                # ctx must be Load: `self.w = x` does not *read* self.w, it writes it -- without
                # this check every plain attribute assignment showed up as a spurious use of
                # itself (self.w = x rendered as "self.w feeds self.w").
                dotted[id(node)] = f"{node.value.id}.{node.attr}"
                consumed.add(id(node.value))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(id(node.func))

    aliases = index.import_aliases.get(scope.label, frozenset())
    found: List[Tuple[int, int, str]] = []
    for root in parts:
        for node in ast.walk(root):
            name: Optional[str] = None
            if id(node) in dotted:
                name = dotted[id(node)]
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if id(node) in consumed:
                    continue
                name = node.id
                local = name in scope.assigned or name in scope.params
                if name.startswith("__") or name in IGNORED_NAMES:
                    continue
                if not local and (name in aliases or index.local_function(name) is not None):
                    continue                   # a function or an import, not a value
                if id(node) in called and not local:
                    continue
            if name is None:
                continue
            found.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), name))

    found.sort(key=lambda item: (item[0], item[1]))
    ordered: List[str] = []
    for _line, _col, name in found:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _mark_inner_bindings(node: ast.AST, consumed: Set[int]) -> None:
    """A comprehension's loop variable and a lambda's parameters are that expression's
    own names, not reads of a variable with the same name outside it: mark every Name
    node that refers to one of them as consumed."""
    inner: List[ast.AST] = []
    bound: Set[str] = set()
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        for i, gen in enumerate(node.generators):
            bound.update(n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name))
            inner.extend(gen.ifs)
            if i:
                inner.append(gen.iter)         # the first iterable is evaluated outside
        inner.extend([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt])
    elif isinstance(node, ast.Lambda):
        bound.update(_parameters(node))
        inner.append(node.body)
    if not bound:
        return
    for part in inner:
        for sub in ast.walk(part):
            if isinstance(sub, ast.Name) and sub.id in bound:
                consumed.add(id(sub))


def _site_kind(stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.AugAssign):
        return "aug"
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return "loop"
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return "with"
    return "assign"


# ---------------------------------------------------------------------------------------
# Building the graph
# ---------------------------------------------------------------------------------------
def _home_key(index: Index, scope: Scope, name: str) -> Key:
    """Which node a name belongs to when it is read in `scope`.

    `self.x` belongs to its class, not to the method that happens to mention it, so every
    method touching it lands on one node instead of one node per method. A name this scope
    neither assigns nor takes as a parameter, but the module does, belongs to the module --
    otherwise a global read from three functions would look like three variables."""
    if name.startswith(("self.", "cls.")) and scope.class_name:
        owner_label, owner_class = _attribute_owner(index, scope.label, scope.class_name, name)
        return (owner_label, f"<class {owner_class}>", name)
    module = index.module_scope.get(scope.label)
    if name in scope.globals and module is not None:
        return (scope.label, MODULE_SCOPE, name)
    if name not in scope.nonlocals and (name in scope.assigned or name in scope.params):
        return (scope.label, scope.key_scope, name)
    # A closure: a nested function reading (or `nonlocal`-writing) its enclosing function's variable.
    enclosing = scope.parent
    while enclosing is not None and not enclosing.is_module:
        if name in enclosing.globals:
            break
        if name not in enclosing.nonlocals and (name in enclosing.assigned or name in enclosing.params):
            return (enclosing.label, enclosing.key_scope, name)
        enclosing = enclosing.parent
    if module is not None and module is not scope and name in module.assigned:
        return (scope.label, MODULE_SCOPE, name)
    return (scope.label, scope.key_scope, name)


def _attribute_owner(index: Index, label: str, class_name: str, name: str) -> Tuple[str, str]:
    """The class whose methods assign `self.<attr>`: this class, or else the nearest base
    class that does -- an inherited attribute is one variable of one object."""
    cache_key = (label, class_name, name)
    if cache_key in index.attr_owner:
        return index.attr_owner[cache_key]
    index.attr_owner[cache_key] = (label, class_name)          # guards a cycle
    owner = (label, class_name)
    queue: List[Tuple[str, str]] = [(label, class_name)]
    seen: Set[Tuple[str, str]] = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if any(name in method.assigned for method in index.classes.get(current) or []):
            owner = current
            break
        for base in index.class_bases.get(current, []):
            if (current[0], base) in index.class_bases:
                queue.append((current[0], base))
            else:
                queue.extend(k for k in index.class_bases if k[1] == base)
    index.attr_owner[cache_key] = owner
    return owner


def _related_scopes(index: Index, key: Key) -> List[Scope]:
    """Every scope where the name refers to this key's variable: the key's own scopes, plus
    functions that read a module global, a closure's enclosing variable, a `global`
    declaration, or an attribute inherited from a base class."""
    label, _scope_key, name = key
    scopes = list(index.scopes_for(key))
    attribute = name.startswith(("self.", "cls."))
    for scope in index.scopes:
        if scope in scopes or (not attribute and scope.label != label):
            continue
        if attribute and not scope.class_name:
            continue
        if _home_key(index, scope, name) == key:
            scopes.append(scope)
    return scopes


def _class_scope_key(index: Index, label: str, class_name: str) -> str:
    key = f"<class {class_name}>"
    if (label, key) not in index.by_key:
        index.by_key[(label, key)] = list(index.classes.get((label, class_name)) or [])
    return key


def _register_class_scopes(index: Index) -> None:
    """`self.x` keys use a `<class C>` scope, which is not a scope the parser produced --
    point it at every method of C so lookups resolve."""
    for (label, class_name), methods in index.classes.items():
        index.by_key.setdefault((label, f"<class {class_name}>"), list(methods))


def find_origin(index: Index, symbol: str, file_hint: Optional[str] = None,
                line_hint: Optional[int] = None) -> Optional[Tuple[Scope, Optional[int]]]:
    """The scope whose body actually assigns `symbol` (or takes it as a parameter), and the
    line to measure "the assignment in effect here" from.

    Preference order: an explicit file/line hint, then the closest assignment above that
    line, then a function that assigns it, then a parameter, then module level. Ties go to
    the first file the caller handed over -- which is the entry script, so plain
    `TRACE: loss` lands in the training loop rather than in some utility module."""
    def _matches_file(scope: Scope) -> bool:
        if not file_hint:
            return True
        hint = file_hint.replace(os.sep, "/")
        path = (scope.path or "").replace(os.sep, "/")
        label = scope.label.replace(os.sep, "/")
        return (label == hint or path == hint or label.endswith("/" + hint)
                or path.endswith("/" + hint) or os.path.basename(path) == hint)

    assigns: List[Tuple[Scope, List[int]]] = []
    params: List[Scope] = []
    for scope in index.scopes:
        if not _matches_file(scope):
            continue
        lines = [s.lineno for s in scope.stmts if symbol in targets_of(s)]
        if lines:
            assigns.append((scope, sorted(lines)))
        elif symbol in scope.params:
            params.append(scope)

    if line_hint is not None:
        # The function the hinted line is in, if it (or a function around it) binds the name.
        enclosing = [s for s in index.scopes if not s.is_module and _matches_file(s)
                     and s.func.lineno <= line_hint <= (getattr(s.func, "end_lineno", None) or s.func.lineno)]
        if enclosing:
            scope: Optional[Scope] = max(enclosing, key=lambda s: s.func.lineno)
            while scope is not None and not scope.is_module:
                if symbol in scope.globals:
                    break
                if symbol not in scope.nonlocals and (symbol in scope.assigned or symbol in scope.params):
                    return scope, line_hint
                scope = scope.parent
        # No function around the line binds it, so the line reads the module's variable:
        # prefer a module-level assignment (closest above) over some function's local.
        module_assigns = [(s, lines) for s, lines in assigns if s.is_module]
        module_best: Optional[Tuple[int, Scope]] = None
        for scope, lines in module_assigns:
            earlier = [ln for ln in lines if ln <= line_hint]
            if earlier and (module_best is None or line_hint - earlier[-1] < module_best[0]):
                module_best = (line_hint - earlier[-1], scope)
        if module_best is not None:
            return module_best[1], line_hint
        # Otherwise the scope that assigns it closest above the hinted line is the one being asked about.
        best: Optional[Tuple[int, Scope, int]] = None
        for scope, lines in assigns:
            earlier = [ln for ln in lines if ln <= line_hint]
            if not earlier:
                continue
            distance = line_hint - earlier[-1]
            if best is None or distance < best[0]:
                best = (distance, scope, earlier[-1])
        if best is not None:
            return best[1], line_hint
        for scope in params:
            func = scope.func
            if func is not None and func.lineno <= line_hint <= (getattr(func, "end_lineno", None) or line_hint):
                return scope, line_hint

    functions = [(s, lines) for s, lines in assigns if not s.is_module]
    if functions:
        scope, lines = functions[0]
        return scope, lines[-1]
    if assigns:
        scope, lines = assigns[0]
        return scope, lines[-1]
    if params:
        return params[0], None
    return None


def build(files: Sequence[Tuple[str, str, str]], symbol: str,
          file_hint: Optional[str] = None, line_hint: Optional[int] = None,
          max_depth: int = MAX_DEPTH) -> Optional[Graph]:
    """Build the influence graph for `symbol`, or None when no such variable is written
    anywhere in `files`. `files` is (label, path, text); the label is what gets printed."""
    index = Index(files)
    _register_class_scopes(index)
    origin = find_origin(index, symbol, file_hint, line_hint)
    if origin is None:
        return None
    origin_scope, ref_line = origin

    root = _home_key(index, origin_scope, symbol)
    where = origin_scope.name if not origin_scope.is_module else "module scope"
    origin_text = f"{origin_scope.label}:{ref_line} in {where}" if ref_line else f"{origin_scope.label} in {where}"
    graph = Graph(symbol=symbol, root=root, nodes={}, origin=origin_text)
    graph.nodes[root] = Node(key=root, name=symbol, depth_up=0, depth_down=0)

    _expand_upstream(index, graph, root, ref_line, max_depth)
    _expand_downstream(index, graph, root, ref_line, max_depth)
    if len(graph.nodes) >= MAX_NODES:
        graph.truncated = True
        graph.notes.append(
            f"stopped at {MAX_NODES} variables -- trace a narrower variable, or pass a "
            "file:line to start somewhere more specific"
        )
    return graph


def _node(graph: Graph, key: Key) -> Optional[Node]:
    node = graph.nodes.get(key)
    if node is None:
        if len(graph.nodes) >= MAX_NODES:
            graph.truncated = True
            return None
        node = Node(key=key, name=key[2])
        graph.nodes[key] = node
    return node


def _pick_sites(defs: List[Tuple[Scope, ast.stmt]], ref_line: Optional[int]) -> Tuple[List[Tuple[Scope, ast.stmt]], int]:
    """Which assignment sites to keep when a variable has more than MAX_SITES of them: the
    ones nearest the point the trace is asking about, since those are the ones that most
    plausibly produced the value being looked at."""
    if len(defs) <= MAX_SITES:
        return defs, 0
    anchor = ref_line if ref_line is not None else defs[-1][1].lineno
    ordered = sorted(defs, key=lambda item: abs(item[1].lineno - anchor))
    kept = sorted(ordered[:MAX_SITES], key=lambda item: item[1].lineno)
    return kept, len(defs) - len(kept)


def _expand_upstream(index: Index, graph: Graph, root: Key, root_ref: Optional[int],
                     max_depth: int) -> None:
    """Walk backwards: what fed this, and what fed those."""
    frontier: List[Tuple[Key, Optional[int], int]] = [(root, root_ref, 0)]
    resolved: Set[Key] = set()
    while frontier:
        key, ref_line, depth = frontier.pop(0)
        if key in resolved:
            continue
        resolved.add(key)
        node = graph.nodes.get(key)
        if node is None:
            continue
        node.depth_up = depth if node.depth_up is None else min(node.depth_up, depth)

        label, _scope_key, name = key
        scopes = index.scopes_for(key)
        defs = [(s, stmt) for s in _related_scopes(index, key) for stmt in s.stmts
                if name in targets_of(stmt) and _home_key(index, s, name) == key]
        defs.sort(key=lambda item: item[1].lineno)

        if not defs:
            # Not assigned here: a parameter (whose real source is at the call sites), or
            # something the traced code never writes at all.
            param_scopes = [s for s in scopes if name in s.params]
            if param_scopes:
                node.sites.append(_parameter_site(index, graph, param_scopes[0], name, depth, max_depth, frontier))
            else:
                node.sites.append(Site(
                    label=label, scope=_scope_key or MODULE_SCOPE, lineno=None, snippet="",
                    kind="external",
                    note="external to the traced code (a global, an import, or set outside the project)",
                ))
            continue

        kept, dropped = _pick_sites(defs, ref_line)
        node.more_sites = dropped
        if dropped:
            graph.truncated = True
        for scope, stmt in kept:
            site = Site(
                label=scope.label, scope=scope.name, lineno=stmt.lineno,
                snippet=index.snippet(scope.label, stmt.lineno), kind=_site_kind(stmt),
            )
            if depth < max_depth:
                parents = _reads(value_parts(stmt), scope, index)
                for parent in parents[:MAX_PARENTS]:
                    parent_key = _home_key(index, scope, parent)
                    if parent_key == key:
                        continue               # x = x + 1: its own previous value, not a new parent
                    child_node = _node(graph, parent_key)
                    if child_node is None:
                        continue
                    site.parents.append(parent_key)
                    frontier.append((parent_key, stmt.lineno, depth + 1))
                if len(parents) > MAX_PARENTS:
                    graph.truncated = True
                _add_return_hop(index, graph, scope, stmt, site, depth, max_depth, frontier)
            node.sites.append(site)


def _add_return_hop(index: Index, graph: Graph, scope: Scope, stmt: ast.stmt, site: Site,
                    depth: int, max_depth: int, frontier: List[Tuple[Key, Optional[int], int]]) -> None:
    """`x = build_model(cfg)` -- follow into build_model and treat whatever it returns as
    feeding x, so the chain does not dead-end at the call. Only for a project-local
    function with exactly one definition: anything else is a guess."""
    for part in value_parts(stmt):
        for node in ast.walk(part):
            if not isinstance(node, ast.Call):
                continue
            callee = _called_name(node)
            if not callee:
                continue
            target = index.local_function(callee)
            if target is None or target is scope:
                continue
            returns = [s for s in target.stmts if isinstance(s, ast.Return) and s.value is not None]
            if not returns:
                continue
            site.note = f"via {callee}() in {target.label}:{getattr(target.func, 'lineno', '?')}"
            for ret in returns[:2]:
                for name in _reads([ret.value], target, index)[:MAX_PARENTS]:
                    parent_key = _home_key(index, target, name)
                    if parent_key in site.parents:
                        continue
                    if _node(graph, parent_key) is None:
                        continue
                    site.parents.append(parent_key)
                    frontier.append((parent_key, ret.lineno, depth + 1))
            return                             # one hop out per site is enough to stay readable


def _parameter_site(index: Index, graph: Graph, scope: Scope, name: str, depth: int,
                    max_depth: int, frontier: List[Tuple[Key, Optional[int], int]]) -> Site:
    """A parameter's value comes from its call sites, so that is where the trace continues
    -- the single most common place a slice used to stop one hop short of the real cause."""
    func = scope.func
    site = Site(
        label=scope.label, scope=scope.name,
        lineno=getattr(func, "lineno", None),
        snippet=index.snippet(scope.label, getattr(func, "lineno", None)),
        kind="param", note=f"parameter of {scope.name}",
    )
    if depth >= max_depth:
        return site
    position = scope.params.index(name) if name in scope.params else -1
    callers = index.callers_of(scope)
    if not callers:
        site.note += " -- no call site found in the traced code"
        return site
    for caller in callers[:MAX_CALLERS]:
        argument = _argument_for(caller.call, name, position, _takes_self(caller.call, scope), scope)
        if argument is None:
            continue
        for read in _reads([argument], caller.scope, index)[:MAX_PARENTS]:
            parent_key = _home_key(index, caller.scope, read)
            if parent_key in site.parents:
                continue
            if _node(graph, parent_key) is None:
                continue
            site.parents.append(parent_key)
            frontier.append((parent_key, caller.stmt.lineno, depth + 1))
    if len(callers) > MAX_CALLERS:
        graph.truncated = True
    sites = ", ".join(f"{c.scope.label}:{c.stmt.lineno}" for c in callers[:MAX_CALLERS])
    site.note += f" -- called from {sites}"
    return site


def _takes_self(call: ast.Call, target: Scope) -> bool:
    """obj.method(a) passes `self` implicitly; utils.helper(a) -- a plain function reached
    through a module -- does not."""
    if target.class_name is None or target.params[:1] not in (["self"], ["cls"]):
        return False
    if isinstance(call.func, ast.Attribute):
        return True
    # `Opt(0.1)` constructs the object and passes it to __init__ as self.
    return _simple_name(target.name) == "__init__" and _called_name(call) == target.class_name


def _argument_for(call: ast.Call, name: str, position: int,
                  implicit_self: bool = False, target: Optional[Scope] = None) -> Optional[ast.expr]:
    """The expression a call passes for parameter `name`. Keyword first (it is explicit),
    then the positional slot -- offset by one for a method call, whose `self` is implicit.
    With the callee's scope known, a keyword-only parameter is never taken from a
    positional slot, and *args gets every positional argument past the named ones."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    if position < 0:
        return None
    if target is not None and target.func is not None:
        positional, vararg = _positional_params(target.func)
        offset = 1 if implicit_self else 0
        if name in positional:
            position = positional.index(name)
        elif name == vararg:
            extra = [a for a in call.args[max(len(positional) - offset, 0):]
                     if not isinstance(a, ast.Starred)]
            return ast.Tuple(elts=extra, ctx=ast.Load()) if extra else None
        else:
            return None                        # keyword-only / **kwargs: never positional
    index = position
    if implicit_self:
        index -= 1                             # obj.method(a): `a` is parameter 1, not 0
    if 0 <= index < len(call.args):
        argument = call.args[index]
        return None if isinstance(argument, ast.Starred) else argument
    return None


def _expand_downstream(index: Index, graph: Graph, root: Key, root_ref: Optional[int],
                       max_depth: int) -> None:
    """Walk forwards: what this value flows into, and what those flow into. This is the
    half that answers "what else is already wrong because of it"."""
    frontier: List[Tuple[Key, int]] = [(root, 0)]
    resolved: Set[Key] = set()
    while frontier:
        key, depth = frontier.pop(0)
        if key in resolved:
            continue
        resolved.add(key)
        node = graph.nodes.get(key)
        if node is None:
            continue
        node.depth_down = depth if node.depth_down is None else min(node.depth_down, depth)
        if depth >= max_depth:
            continue

        label, _scope_key, name = key
        edges: List[Edge] = []
        effects = 0
        for scope in _related_scopes(index, key):
            for stmt in scope.stmts:
                if name not in _reads(read_parts(stmt), scope, index):
                    continue
                snippet = index.snippet(scope.label, stmt.lineno)
                written = [t for t in targets_of(stmt) if t != name]
                if written:
                    for target in written[:MAX_CHILDREN]:
                        target_key = _home_key(index, scope, target)
                        if target_key == key or _node(graph, target_key) is None:
                            continue
                        edges.append(Edge(scope.label, stmt.lineno, snippet, target_key))
                        frontier.append((target_key, depth + 1))
                else:
                    # A use that produces no variable -- loss.backward(), print(acc), a
                    # condition. Worth showing: it is where the value actually lands.
                    if effects < MAX_EFFECTS:
                        edges.append(Edge(scope.label, stmt.lineno, snippet, None))
                        effects += 1
                    else:
                        node.more_out += 1
                        graph.truncated = True
                _forward_into_calls(index, graph, scope, stmt, name, depth, edges, frontier)
                if isinstance(stmt, ast.Return):
                    _forward_through_return(index, graph, scope, stmt, depth, edges, frontier)

        seen: Set[Tuple[Optional[Key], int]] = set()
        for edge in edges:
            fingerprint = (edge.target, edge.lineno)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            if len(node.out) >= MAX_CHILDREN + MAX_EFFECTS:
                node.more_out += 1
                graph.truncated = True
                continue
            node.out.append(edge)


def _forward_into_calls(index: Index, graph: Graph, scope: Scope, stmt: ast.stmt, name: str,
                        depth: int, edges: List[Edge], frontier: List[Tuple[Key, int]]) -> None:
    """`train(model, loss)` -- the value keeps going as the callee's parameter, in the
    callee's own scope. Without this, a trace stops at every function boundary."""
    for part in read_parts(stmt):
        for node in ast.walk(part):
            if not isinstance(node, ast.Call):
                continue
            callee = _called_name(node)
            if not callee:
                continue
            target_scope = index.local_function(callee)
            if target_scope is None or target_scope is scope:
                continue
            for position, argument in enumerate(node.args):
                if name not in _reads([argument], scope, index):
                    continue
                slot = position
                if _takes_self(node, target_scope):
                    slot += 1                  # skip the implicit self
                positional, vararg = _positional_params(target_scope.func)
                if slot < len(positional):
                    param = positional[slot]
                elif vararg:
                    param = vararg             # past the named ones: it lands in *args
                else:
                    continue
                param_key = (target_scope.label, target_scope.key_scope, param)
                if _node(graph, param_key) is None:
                    continue
                edges.append(Edge(scope.label, stmt.lineno, index.snippet(scope.label, stmt.lineno),
                                  param_key, note=f"passed into {callee}()"))
                frontier.append((param_key, depth + 1))
            for keyword in node.keywords:
                if keyword.arg is None or name not in _reads([keyword.value], scope, index):
                    continue
                if keyword.arg not in target_scope.params:
                    continue
                param_key = (target_scope.label, target_scope.key_scope, keyword.arg)
                if _node(graph, param_key) is None:
                    continue
                edges.append(Edge(scope.label, stmt.lineno, index.snippet(scope.label, stmt.lineno),
                                  param_key, note=f"passed into {callee}()"))
                frontier.append((param_key, depth + 1))


def _forward_through_return(index: Index, graph: Graph, scope: Scope, stmt: ast.stmt,
                            depth: int, edges: List[Edge], frontier: List[Tuple[Key, int]]) -> None:
    """`return loss` -- the value continues as whatever each caller assigned the call to."""
    for caller in index.callers_of(scope)[:MAX_CALLERS]:
        for target in targets_of(caller.stmt)[:MAX_CHILDREN]:
            target_key = _home_key(index, caller.scope, target)
            if _node(graph, target_key) is None:
                continue
            edges.append(Edge(caller.scope.label, caller.stmt.lineno,
                              index.snippet(caller.scope.label, caller.stmt.lineno),
                              target_key, note=f"returned from {_simple_name(scope.name)}()"))
            frontier.append((target_key, depth + 1))


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"


class _Paint:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self.on and text else text

    def dim(self, text: str) -> str:
        return self._wrap(_DIM, text)

    def bold(self, text: str) -> str:
        return self._wrap(_BOLD, text)

    def var(self, text: str) -> str:
        return self._wrap(_CYAN, text)

    def hop(self, text: str) -> str:
        return self._wrap(_YELLOW, text)


def render(graph: Graph, color: bool = False, width: int = 100) -> str:
    """The whole trace as text: a header, the upstream graph, the downstream graph, and a
    flat inventory of every connected variable.

    Both shapes are printed on purpose. The graph shows the thing you cannot see by
    reading code -- whether this value has one long chain of causes or a wide fan-in of
    twelve -- and the flat list is what you scan for a name you recognize (and what the
    model reads, where a tree drawn in box characters is just noise)."""
    paint = _Paint(color)
    out: List[str] = []
    ups, downs = graph.upstream_count(), graph.downstream_count()
    total = len(graph.nodes)
    files = graph.files()

    out.append(f"TRACE  {paint.bold(graph.symbol)}    {paint.dim(graph.origin)}")
    summary = (f"{total} connected variable(s) across {len(files)} file(s) -- "
               f"{ups} upstream, {downs} downstream")
    out.append("  " + paint.dim(summary))
    if graph.truncated:
        out.append("  " + paint.dim("(bounded: some branches were cut -- see the notes at the end)"))
    out.append("")

    out.append(paint.bold(f"  what feeds {graph.symbol}") + paint.dim("  (each line's children are its causes)"))
    out.append(paint.dim("  " + "-" * min(width - 4, 72)))
    out.extend(_render_upstream(graph, paint))
    out.append("")

    out.append(paint.bold(f"  what {graph.symbol} feeds") + paint.dim("  (each line's children are its effects)"))
    out.append(paint.dim("  " + "-" * min(width - 4, 72)))
    out.extend(_render_downstream(graph, paint))
    out.append("")

    out.extend(_render_inventory(graph, paint))
    for note in graph.notes:
        out.append("  " + paint.dim(note))
    return "\n".join(out)


def _where(graph: Graph, node: Node, paint: _Paint) -> str:
    """The one-line description of a variable: where it is set, and to what."""
    if not node.sites:
        return paint.dim("(not resolved)")
    site = node.sites[0]
    if site.kind == "param" or site.kind == "external":
        return paint.dim(site.note or site.kind)
    place = f"{site.label}:{site.lineno}"
    text = f"{paint.dim(place)}  {site.snippet}"
    if len(node.sites) > 1 or node.more_sites:
        extra = len(node.sites) - 1 + node.more_sites
        text += paint.hop(f"  (+{extra} other assignment site(s))")
    if site.note:
        text += paint.dim(f"  [{site.note}]")
    return text


def _render_upstream(graph: Graph, paint: _Paint) -> List[str]:
    lines: List[str] = []
    root = graph.nodes[graph.root]
    lines.append(f"  {paint.var(root.name)}   {_where(graph, root, paint)}")
    drawn: Set[Key] = {graph.root}

    def walk(key: Key, prefix: str, last: bool, scope_shown: str) -> None:
        node = graph.nodes.get(key)
        connector = "`-- " if last else "|-- "
        if node is None:
            lines.append(f"  {prefix}{connector}{paint.dim(key[2])}")
            return
        hop = ""
        if node.scope != scope_shown:
            hop = paint.hop(f"  ^ in {node.scope}")
        label = f"{paint.var(node.name)}{hop}   {_where(graph, node, paint)}"
        if key in drawn:
            parents = [p for site in node.sites for p in site.parents]
            lines.append(f"  {prefix}{connector}{label}" +
                         (paint.dim("  (shown above)") if parents else ""))
            return
        drawn.add(key)
        lines.append(f"  {prefix}{connector}{label}")
        child_prefix = prefix + ("    " if last else "|   ")
        _walk_sites(node, child_prefix, walk, node.scope)

    def _walk_sites(node: Node, child_prefix: str, recurse, scope_shown: str) -> None:
        # One assignment site: its causes hang straight off the variable. Several: each
        # site gets its own line first, so "assigned here from a, or here from b" reads as
        # the two separate possibilities it is.
        sites = [s for s in node.sites if s.parents]
        if len(sites) == 1:
            parents = sites[0].parents
            for i, parent in enumerate(parents):
                recurse(parent, child_prefix, i == len(parents) - 1, scope_shown)
            return
        for j, site in enumerate(sites):
            last_site = j == len(sites) - 1
            connector = "`-- " if last_site else "|-- "
            head = paint.dim(f"{site.label}:{site.lineno}  {site.snippet}")
            lines.append(f"  {child_prefix}{connector}{head}")
            inner = child_prefix + ("    " if last_site else "|   ")
            for i, parent in enumerate(site.parents):
                recurse(parent, inner, i == len(site.parents) - 1, scope_shown)

    _walk_sites(root, "", walk, root.scope)
    if len(lines) == 1:
        site = root.sites[0] if root.sites else None
        reason = (site.note or "nothing in the traced code feeds it") if site else "nothing found"
        lines.append("  " + paint.dim(f"  (no upstream variables -- {reason})"))
    return lines


def _render_downstream(graph: Graph, paint: _Paint) -> List[str]:
    lines: List[str] = []
    root = graph.nodes[graph.root]
    lines.append(f"  {paint.var(root.name)}")
    drawn: Set[Key] = {graph.root}
    any_edge = False

    def walk(node: Node, prefix: str, scope_shown: str) -> None:
        nonlocal any_edge
        edges = node.out
        for i, edge in enumerate(edges):
            last = i == len(edges) - 1
            connector = "`-> " if last else "|-> "
            place = paint.dim(f"{edge.label}:{edge.lineno}")
            note = paint.hop(f"  {edge.note}") if edge.note else ""
            if edge.target is None:
                lines.append(f"  {prefix}{connector}{paint.dim('used:')} {edge.snippet}   {place}")
                any_edge = True
                continue
            child = graph.nodes.get(edge.target)
            name = edge.target[2]
            hop = ""
            if child is not None and child.scope != scope_shown:
                hop = paint.hop(f"  v in {child.scope}")
            lines.append(f"  {prefix}{connector}{paint.var(name)}{hop}{note}   {place}  {edge.snippet}")
            any_edge = True
            if child is None:
                continue
            if edge.target in drawn:
                if child.out:
                    lines[-1] += paint.dim("  (shown above)")
                continue
            drawn.add(edge.target)
            walk(child, prefix + ("    " if last else "|   "), child.scope)
        if node.more_out:
            lines.append(f"  {prefix}    {paint.dim(f'(+{node.more_out} more use(s) not shown)')}")

    walk(root, "", root.scope)
    if not any_edge:
        lines.append("  " + paint.dim("  (nothing in the traced code reads it -- it is a dead end here)"))
    return lines


def _render_inventory(graph: Graph, paint: _Paint) -> List[str]:
    """Every connected variable, one per line, nearest first. This is the part that gets
    scanned for a name -- and the part a model reads."""
    lines = [paint.bold("  all connected variables"), paint.dim("  " + "-" * 72)]
    def sort_key(node: Node):
        up = node.depth_up if node.depth_up is not None else 99
        down = node.depth_down if node.depth_down is not None else 99
        return (min(up, down), node.scope, node.name)

    for node in sorted(graph.nodes.values(), key=sort_key):
        marks = []
        if node.depth_up is not None and node.key != graph.root:
            marks.append(f"u{node.depth_up}")
        if node.depth_down is not None and node.key != graph.root:
            marks.append(f"d{node.depth_down}")
        mark = paint.dim(",".join(marks) or "-")
        scope = "" if node.scope == MODULE_SCOPE else f" in {node.scope}"
        lines.append(f"    {mark:<12} {paint.var(node.name)}{paint.dim(scope)}   {_where(graph, node, paint)}")
    return lines


# ---------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------
_SYMBOL_HINT = ("use  trace <variable>  or  trace <variable>:<file>:<line>  "
                "(self.<attr> works too)")


def parse_target(raw: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """Split `loss`, `loss:train.py`, `loss:train.py:117` or `self.lr:model.py:44` into
    (symbol, file, line, error)."""
    text = (raw or "").strip()
    if not text:
        return None, None, None, "no variable given -- " + _SYMBOL_HINT
    parts = text.split(":")
    symbol = parts[0].strip()
    file_hint = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else None
    line_hint: Optional[int] = None
    if len(parts) >= 3 and parts[2].strip().isdigit():
        line_hint = int(parts[2].strip())
    pieces = symbol.split(".")
    if not pieces or not all(p.isidentifier() for p in pieces) or len(pieces) > 2:
        return None, None, None, f"'{symbol}' does not look like a variable name -- " + _SYMBOL_HINT
    return symbol, file_hint, line_hint, None


def trace(files: Sequence[Tuple[str, str, str]], target: str, color: bool = False,
          max_depth: int = MAX_DEPTH) -> str:
    """Parse a target, build the graph, render it. The one call both the agent's TRACE:
    directive and the terminal's /trace need."""
    symbol, file_hint, line_hint, error = parse_target(target)
    if error:
        return f"TRACE: {error}"
    if not files:
        return ("TRACE: there are no Python files to trace through here yet -- Pulse needs the "
                "script (and any project files it imports) to be known first.")
    graph = build(files, symbol, file_hint, line_hint, max_depth=max_depth)
    if graph is None:
        where = f" in {file_hint}" if file_hint else " in any known file"
        return (f"TRACE '{symbol}': never assigned{where}, and not a parameter of any function "
                "there either. It may be an attribute of something other than self, a dict or "
                "list entry, or it may come from outside the project.")
    return render(graph, color=color)


def project_files(root: str, entry: Optional[str] = None, max_files: int = 200,
                  max_bytes: int = 2_000_000) -> List[Tuple[str, str, str]]:
    """(label, path, text) for the Python files under `root`, entry script first.

    For callers that don't already have Pulse's own file set (the `pulse watch` console
    reads a run's directory off disk). Skips the directories that are never the user's
    code, and stops at max_files so pointing this at a huge tree stays quick."""
    skip = {".git", ".hg", "__pycache__", ".venv", "venv", "env", "node_modules",
            ".mypy_cache", ".pytest_cache", ".tox", "build", "dist", ".idea", ".vscode"}
    found: List[Tuple[str, str, str]] = []
    seen: Set[str] = set()

    def add(path: str) -> None:
        real = os.path.abspath(path)
        if real in seen or len(found) >= max_files:
            return
        try:
            if os.path.getsize(real) > max_bytes:
                return
            with open(real, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            return
        seen.add(real)
        label = os.path.relpath(real, root).replace(os.sep, "/") if root else os.path.basename(real)
        found.append((label, real, text))

    if entry and os.path.isfile(entry):
        add(entry)
    if root and os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in sorted(dirnames) if d not in skip and not d.startswith(".")]
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    add(os.path.join(dirpath, filename))
            if len(found) >= max_files:
                break
    return found
