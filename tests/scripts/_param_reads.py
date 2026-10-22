"""The ADR-129 parameter read set of a stage (S5.9 "Declared parameters"): every pipeline parameter a stage's
callables read, so the gate can assert each one moves the stage token.

The analysis is interprocedural over the stage module, and it is an ALLOWLIST (S5.9 fail-closed, CCTP-IMPL-19):
it RECOGNISES a fixed set of safe shapes and fails the gate on every other use of ``ctx`` / ``ctx.params`` / an
alias of either. The recognised shapes:
- a literal-key read ``ctx.params["x"]`` / ``ctx.params.get("x")`` and an aliased ``p = ctx.params; p["x"]``;
- a call to a helper DEFINED IN THIS MODULE (followed with the right bindings), including a key-arg helper
  (``_staged_refs(ctx, "dataset")``) and a factory closure (``_rows_work("provider")``);
- a call to a ``scripts._stages`` / ``scripts._driver`` framework helper that takes ``ctx`` but reads no param
  (``sorted_units(ctx)``, ``unit_states(ctx, ...)``);
- a method seam on ``ctx`` or on a ctx-derived object (``ctx.pipeline.stage(...)``, ``st.ref_map(ctx)`` with
  ``st = ctx.pipeline.stage(...)``).
Anything else -- a whole-dict use, an imported/dynamic callee handed ``ctx`` or ``ctx.params`` (positionally, by
keyword, starred, or inside a literal), a dotted-module or module-alias base, or an unresolvable key -- fails.

A ``cache_dir`` read never keys the token: it is a cache LOCATION, the parameter analogue of the ``neutral``
``SILLY_KICKS_CORPUS_CACHE_DIR`` env (env_contract), so it is excluded. A dataset parameter (a stage's
``dataset_param``) IS in the set: it keys the token through the dataset IDENTITY, and the gate flips it there.
"""

from __future__ import annotations

import ast
import pathlib

#: Parameters a stage may read without keying the token: a cache location (output-invariant), mirroring the
#: ``neutral`` env class (env_contract). Excluded from the read set the gate checks.
_NEUTRAL_PARAMS = frozenset({"cache_dir"})

#: Reads the static analysis over-reports because they are DEAD for this stage's binding (the guard excludes
#: them), keyed and live on a SIBLING stage. Each entry cites the guard. ``(stage module, stage) -> {param}``.
_DEAD_READS: dict[tuple[str, str], frozenset[str]] = {
    # `_home_map` reads ctx.params["dataset"] only under `if ref.dataset_param == "dataset"` (`_load_core`),
    # and every probe ref sets dataset_param="probe_dataset" (`_probe_refs`): the read is dead for probe. It
    # is live and keyed on the `extract` stage (ref.dataset_param == "dataset"). The guard is a runtime ref
    # attribute, not a literal the static fold can resolve.
    ("scripts._xcross_stages", "probe"): frozenset({"dataset"}),
}

_CALLABLE_KWARGS = ("work", "load", "reduce", "units", "refs", "key", "counters", "declare", "reduce_order")


def _module_tree(module) -> ast.Module:
    return ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))


def _funcs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _stage_call(tree: ast.Module, name: str) -> ast.Call | None:
    """The ``Stage(...)``/``ItemStage(...)`` Call whose ``name=`` is ``name``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("Stage", "ItemStage"):
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant) and kw.value.value == name:
                    return node
    return None


def _returned_def(fn: ast.FunctionDef) -> ast.FunctionDef | None:
    """The inner function a factory returns (``def work(...): ...; return work``)."""
    returned: str | None = None
    for node in fn.body:
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
            returned = node.value.id
    for node in fn.body:
        if isinstance(node, ast.FunctionDef) and node.name == returned:
            return node
    return None


class _Reader:
    def __init__(self, funcs: dict[str, ast.FunctionDef], framework: frozenset[str] = frozenset()):
        self.funcs = funcs
        self.framework = framework  # names imported from scripts._stages/_driver that take ctx but read no param
        self.reads: set[str] = set()
        self.unresolved: set[str] = set()  # ctx / ctx.params uses the gate cannot prove safe (fail closed, S5.9)
        self._seen: set = set()

    def _is_ctx_params(self, expr, ctx_names, params_names) -> bool:
        if isinstance(expr, ast.Name) and expr.id in params_names:
            return True
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr == "params"
            and isinstance(expr.value, ast.Name)
            and expr.value.id in ctx_names
        )

    def _key(self, expr, keyvars) -> str | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.Name) and expr.id in keyvars:
            return keyvars[expr.id]
        return None

    def _root_name(self, expr):
        """The root ``Name`` id of an attribute/subscript/call chain (``scripts._cache.fn`` -> ``scripts``,
        ``ctx.pipeline.stage(...)`` -> ``ctx``), or None."""
        while isinstance(expr, (ast.Attribute, ast.Subscript)):
            expr = expr.value
        if isinstance(expr, ast.Call):
            return self._root_name(expr.func)
        return expr.id if isinstance(expr, ast.Name) else None

    def _folds_false(self, test, keyvars) -> bool:
        """``<keyvar> == "<const>"`` that is statically false, or an ``and`` with such a first operand -- so the
        guarded branch is dead for this binding (receiver ``_staged_refs``'s ``if param == "dataset"``)."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left = self._key(test.left, keyvars)
            right = test.comparators[0]
            if left is not None and isinstance(right, ast.Constant):
                return left != right.value
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            return any(self._folds_false(v, keyvars) for v in test.values)
        return False

    def _dead(self, body, keyvars) -> set[int]:
        """Node ids inside a dead ``if``/``A if test else B`` branch (statically false test)."""
        dead: set[int] = set()
        for node in body:
            for a in ast.walk(node):
                if isinstance(a, ast.If) and self._folds_false(a.test, keyvars):
                    for stmt in a.body:
                        dead.update(id(n) for n in ast.walk(stmt))
                elif isinstance(a, ast.IfExp) and self._folds_false(a.test, keyvars):
                    dead.update(id(n) for n in ast.walk(a.body))
                elif isinstance(a, ast.BoolOp) and isinstance(a.op, ast.And):
                    # `x == "dataset" and ctx.params["match_ids_json"]`: a false first operand short-circuits
                    # the rest, so later operands never evaluate (the read is dead for this binding).
                    for i, v in enumerate(a.values):
                        if self._folds_false(v, keyvars):
                            for later in a.values[i + 1 :]:
                                dead.update(id(n) for n in ast.walk(later))
                            break
        return dead

    def analyze(self, fn: ast.FunctionDef | ast.Lambda, *, ctx_names, params_names, ctx_obj_names, keyvars) -> None:
        sig = (
            id(fn),
            frozenset(ctx_names),
            frozenset(params_names),
            frozenset(ctx_obj_names),
            tuple(sorted(keyvars.items())),
        )
        if sig in self._seen:
            return
        self._seen.add(sig)
        ctx_names, params_names = set(ctx_names), set(params_names)
        ctx_obj_names, keyvars = set(ctx_obj_names), dict(keyvars)
        body = [fn.body] if isinstance(fn, ast.Lambda) else list(fn.body)
        # Local bindings first (in source order): ctx aliases (`c = ctx`), params aliases (`p = ctx.params`),
        # ctx-derived object seams (`st = ctx.pipeline.stage(...)`) and literal key vars (`name = "x"`). An
        # imported module bound to a local name (`m = helpmod`) is none of these, so `m.fn(ctx)` fails closed.
        for node in body:
            for a in ast.walk(node):
                if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name):
                    tgt, val = a.targets[0].id, a.value
                    if isinstance(val, ast.Name) and val.id in ctx_names:
                        ctx_names.add(tgt)
                    elif self._is_ctx_params(val, ctx_names, params_names):
                        params_names.add(tgt)
                    elif not isinstance(val, ast.Name) and self._root_name(val) in ctx_names:
                        ctx_obj_names.add(tgt)  # a ctx-derived object: its methods are a runner seam
                    elif isinstance(val, ast.Constant) and isinstance(val.value, str):
                        keyvars[tgt] = val.value
        dead = self._dead(body, keyvars)
        for node in body:
            for a in ast.walk(node):
                if id(a) not in dead:
                    self._visit(a, ctx_names, params_names, ctx_obj_names, keyvars)

    def _callee_class(self, f, ctx_names, params_names, ctx_obj_names) -> str:
        """Classify a call target for the ALLOWLIST gate (S5.9 fail-closed): ``follow`` (a helper in this module,
        analysed), ``framework`` (a scripts._stages/_driver helper that takes ctx but reads no param), ``seam`` (a
        method on ctx or on a ctx-derived object, ``st.ref_map(...)``), ``params_method`` (a method on the params
        dict) or ``unfollowable`` (everything else: an imported/builtin name, a dotted-module or module-alias
        base, a dynamic callee)."""
        if isinstance(f, ast.Name):
            if f.id in self.funcs:
                return "follow"
            if f.id in self.framework:
                return "framework"
            return "unfollowable"
        if isinstance(f, ast.Attribute):
            if self._is_ctx_params(f.value, ctx_names, params_names):
                return "params_method"
            root = self._root_name(f.value)
            if root in ctx_names or root in ctx_obj_names:
                return "seam"
            return "unfollowable"
        return "unfollowable"  # a lambda / subscript / call-result callee: dynamic

    def _ctx_escape(self, expr, ctx_names, params_names):
        """The label (``ctx`` / ``ctx.params``) when ``expr`` hands the WHOLE ctx or the WHOLE params dict to a
        callee -- directly, through a starred arg, or inside a list/tuple/set/dict literal. A specific value
        (``ctx.params["x"]``) is NOT an escape: its key is recorded as a read elsewhere."""
        if isinstance(expr, ast.Starred):
            return self._ctx_escape(expr.value, ctx_names, params_names)
        if self._is_ctx_params(expr, ctx_names, params_names):
            return "ctx.params"
        if isinstance(expr, ast.Name) and expr.id in ctx_names:
            return "ctx"
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            return next((lbl for e in expr.elts if (lbl := self._ctx_escape(e, ctx_names, params_names))), None)
        if isinstance(expr, ast.Dict):
            return next((lbl for v in expr.values if (lbl := self._ctx_escape(v, ctx_names, params_names))), None)
        return None

    def _visit(self, a, ctx_names, params_names, ctx_obj_names, keyvars) -> None:
        # ctx.params[...] / params_alias[...]: a read (literal key) or an unresolvable key (fail closed).
        if isinstance(a, ast.Subscript) and self._is_ctx_params(a.value, ctx_names, params_names):
            k = self._key(a.slice, keyvars)
            (self.reads.add(k) if k is not None else self.unresolved.add(ast.unparse(a.slice)))
        if not isinstance(a, ast.Call):
            if isinstance(a, ast.Dict):  # {**ctx.params}: a whole-dict spread (the gate cannot tell WHICH params)
                for key, val in zip(a.keys, a.values, strict=False):
                    if key is None and self._is_ctx_params(val, ctx_names, params_names):
                        self.unresolved.add("{**ctx.params}")
            return
        f = a.func
        cls = self._callee_class(f, ctx_names, params_names, ctx_obj_names)
        if cls == "params_method":
            assert isinstance(f, ast.Attribute)  # _callee_class returns "params_method" only for an Attribute
            if f.attr == "get" and a.args:  # ctx.params.get(key)
                k = self._key(a.args[0], keyvars)
                (self.reads.add(k) if k is not None else self.unresolved.add(ast.unparse(a.args[0])))
            else:  # items()/values()/keys()/any other whole-dict method: the gate cannot tell WHICH params
                self.unresolved.add(f"ctx.params.{f.attr}()")
            return
        if cls == "follow":
            assert isinstance(f, ast.Name)  # _callee_class returns "follow" only for a Name in self.funcs
            self._follow(self.funcs[f.id], a.args, ctx_names, params_names, ctx_obj_names, keyvars)
            return
        if cls in ("framework", "seam"):
            return  # a framework helper / a method seam on ctx or a ctx-derived object reads no param
        # unfollowable: the whole ctx or params dict reaching it by ANY argument (positional, starred, keyword,
        # or inside a literal) fails the gate (S5.9). A specific `ctx.params["x"]` value is not an escape.
        for arg in list(a.args) + [kw.value for kw in a.keywords]:
            lbl = self._ctx_escape(arg, ctx_names, params_names)
            if lbl:
                self.unresolved.add(f"{ast.unparse(f)}({lbl})")

    def _follow(self, fn: ast.FunctionDef, args, ctx_names, params_names, ctx_obj_names, keyvars) -> None:
        names = [p.arg for p in fn.args.args]
        new_ctx, new_params, new_obj, new_keyvars = set(), set(), set(), {}
        for pname, arg in zip(names, args, strict=False):
            if isinstance(arg, ast.Name) and arg.id in ctx_names:
                new_ctx.add(pname)
            elif self._is_ctx_params(arg, ctx_names, params_names):
                new_params.add(pname)
            elif isinstance(arg, ast.Name) and arg.id in ctx_obj_names:
                new_obj.add(pname)
            elif isinstance(arg, ast.Attribute) and self._root_name(arg) in ctx_names:
                new_obj.add(pname)  # a ctx-derived attribute object (ctx.pipeline) passed in stays a seam
            else:
                k = self._key(arg, keyvars)
                if k is not None:
                    new_keyvars[pname] = k
        inner = _returned_def(fn)  # a factory: analyze the function it returns, with the bound keyvars free
        if inner is not None:
            inner_ctx = {p.arg for p in inner.args.args if p.arg == "ctx"}
            inner_params = {p.arg for p in inner.args.args if p.arg in ("params", "p")}
            self.analyze(
                inner, ctx_names=inner_ctx, params_names=inner_params, ctx_obj_names=set(), keyvars=new_keyvars
            )
        else:
            self.analyze(fn, ctx_names=new_ctx, params_names=new_params, ctx_obj_names=new_obj, keyvars=new_keyvars)


#: Unresolvable ``ctx.params[<expr>]`` keys the gate accepts without failing closed, with the reason. Each
#: reads a DATASET parameter already covered by the stage's refs (keyed via the dataset identity), so the token
#: test on the resolved dataset params is the real guarantee. ``(stage module, stage) -> {source expr}``.
_UNRESOLVED_OK: dict[tuple[str, str], frozenset[str]] = {
    # `_load_core` loads `ctx.params[ref.dataset_param]` where ref.dataset_param is "dataset" | "probe_dataset"
    # (set by _extract_refs / _probe_refs); both are dataset params keyed via the dataset identity, and both
    # are already in the resolved read set through the refs' literal `staged_*_refs(..., "dataset"/...)`.
    ("scripts._xcross_stages", "extract"): frozenset({"ref.dataset_param"}),
    ("scripts._xcross_stages", "probe"): frozenset({"ref.dataset_param"}),
}


def _read_report(module, st) -> tuple[set[str], set[str]]:
    """``(resolved read set minus neutral/dead, unresolvable-key sources)`` for stage ``st``."""
    tree = _module_tree(module)
    funcs = _funcs(tree)
    call = _stage_call(tree, st.name)
    framework = frozenset(
        a.asname or a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in ("scripts._stages", "scripts._driver")
        for a in node.names
        if a.name != "*"
    )
    reader = _Reader(funcs, framework)
    if call is None:  # the gate could not locate the stage's Stage/ItemStage call -> it cannot prove the reads
        return set(), {f"no Stage/ItemStage(name={st.name!r}) call in {module.__name__}"}  # fail closed (S5.9)
    for kw in call.keywords:
        if kw.arg not in _CALLABLE_KWARGS:
            continue
        params_role = kw.arg == "declare"  # declare(params): its sole arg is a params dict, not ctx
        _seed(reader, funcs, kw.value, params_role=params_role)
    dead = _DEAD_READS.get((module.__name__, st.name), frozenset())
    reads = {p for p in reader.reads if p not in _NEUTRAL_PARAMS and p not in dead}
    return reads, reader.unresolved


def param_reads(module, st) -> set[str]:
    return _read_report(module, st)[0]


def unresolved_reads(module, st) -> set[str]:
    """``ctx`` / ``ctx.params`` uses the gate could NOT prove safe, minus the ``_UNRESOLVED_OK`` allowlist
    (ADR-129 fail-closed, S5.9: an unresolvable key or an unfollowable use that is not allow-listed fails the
    gate, naming the site)."""
    return _read_report(module, st)[1] - set(_UNRESOLVED_OK.get((module.__name__, st.name), frozenset()))


def _seed(reader: _Reader, funcs, expr, *, params_role: bool) -> None:
    if isinstance(expr, ast.Lambda):
        ctx_names = {p.arg for p in expr.args.args if p.arg == "ctx"}
        params_names = {p.arg for p in expr.args.args if p.arg in ("params", "p")} if params_role else set()
        reader.analyze(expr, ctx_names=ctx_names, params_names=params_names, ctx_obj_names=set(), keyvars={})
    elif isinstance(expr, ast.Name) and expr.id in funcs:
        fn = funcs[expr.id]
        ctx_names = {p.arg for p in fn.args.args if p.arg == "ctx"}
        params_names = {fn.args.args[0].arg} if params_role and fn.args.args else set()
        reader.analyze(fn, ctx_names=ctx_names, params_names=params_names, ctx_obj_names=set(), keyvars={})
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in funcs:
        # a factory call: bind its literal args and analyze the returned closure
        reader._follow(funcs[expr.func.id], expr.args, set(), set(), set(), {})


def second_value(pipe, name: str, value):
    """A second VALID value of parameter ``name`` -- typed by the pipeline ``Param`` so a flip never feeds a
    float param a path, etc. The gate flips to it and asserts the token moves."""
    param = next((p for p in pipe.params if p.name == name), None)
    choices = getattr(param, "choices", None) if param is not None else None
    if choices:
        return next(c for c in choices if c != value)
    typ = getattr(param, "type", None) if param is not None else None
    flag = getattr(param, "flag", False) if param is not None else False
    if flag or isinstance(value, bool):
        return not bool(value)
    if typ is float or isinstance(value, float):
        return value + 1.0 if isinstance(value, (int, float)) else 1.0
    if typ is int or isinstance(value, int):
        return value + 1 if isinstance(value, int) else 1
    if typ is pathlib.Path or isinstance(value, pathlib.Path):
        base = value if isinstance(value, pathlib.Path) else pathlib.Path("x")
        return base.parent / (base.name + "_x")
    return value + "_x" if isinstance(value, str) else "sentinel_x"
