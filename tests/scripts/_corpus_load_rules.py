"""AST rules for the corpus-driver load seam (spec section 5).

Four static rules over EVERY ``scripts/*.py`` (private included -- the ``_xtgk_comparability`` /
seam-module blind spot the old gate had):

* Rule A -- no stream loader (``load_matches`` / ``load_statsbomb_matches`` /
  ``load_open_data_matches``) called in a driver outside ``_STREAM_LOADER_EXEMPT``.
* Rule B -- every ``load_match(...)`` passes ``events_only=`` as a keyword.
* Rule C -- no un-sharded LOADING loop (iterates a load AND loads per item) outside
  ``_UNSHARDED_LOOP_EXEMPT``.
* Rule D -- a ``load_match(...)`` whose ``events_only=`` is not literal ``False`` only inside an
  ``_UNADMITTED_EVENTS_ONLY_ALLOWED`` function (closures within included).

The population is DERIVED from the loader modules by enumeration (spec section 5.1). The ledgers
``_RULE_A_PENDING`` / ``_RULE_C_PENDING`` record the exact violations at ``4ac26d0``; each migration
(Tasks 9-15) deletes its own rows, and Task 17 asserts both empty.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"

#: The loader modules whose public functions form the corpus-call set (spec section 5.1).
_LOADER_MODULES = ("_loader_pining", "_sb_open_data", "_loader_databricks")

#: A corpus function is named by one of these prefixes.
_CORPUS_PREFIXES = ("load_", "list_", "select_", "fetch_")

#: The stream loaders Rule A bans from drivers.
_STREAM_LOADERS = frozenset({"load_matches", "load_statsbomb_matches", "load_open_data_matches"})

#: Public loader functions that are NOT corpus functions, each with a reason (spec section 5.1).
#: Asserted EXACT both ways by the gate: a new corpus loader named by the convention joins the corpus
#: set automatically; any other new public function fails CI until classified here.
_LOADER_NON_CORPUS: dict[str, str] = {
    "build_statsbomb_match": "builds ONE SB360 match (a load_match helper), not a corpus lister/loader",
    "build_skillcorner_frames": "builds SkillCorner tracking frames for one match, not a corpus call",
    "match_visibility": "returns a public/restricted visibility label for one match id",
    "assert_statsbomb_open_data_mode": "a guard: refuses when SB credentials are set (open-data only)",
    "all_open_competitions": "returns the (competition, season) pairs, a constant catalogue, not a load",
    "shape_action_values": "reshapes a databricks cohort frame; consumes no corpus",
    "resolve_retention_model": "resolves a bundled model variant; consumes no corpus",
    "resolve_cache_dir": "resolves the raw-artifact cache root (arg/env/None); consumes no corpus",
    "pining_source": "(refs, load) tracking factory: composes list_match_refs + load_match; not a stream/loop",
    "open_data_source": "(refs, load) open-data factory: composes list_open_data_refs + load_open_data_match",
}

#: Rule A exemption, function-granular (spec section 5.3). ``(module, function)``.
_STREAM_LOADER_EXEMPT: dict[tuple[str, str], str] = {
    ("calibrate_tracking_defaults", "_load_fold"): (
        "the documented X item: the Optuna objective's input, held whole by design behind its own RAM fail-fast guard"
    ),
}

#: Rule C exemption -- the only legitimate homes of a loading loop (spec section 5.5). ``module.function``.
_UNSHARDED_LOOP_EXEMPT: dict[str, str] = {
    "_driver.for_each": "IS the sharded loop -- it calls its `load` parameter per item",
    "_loader_pining.load_matches": "the stream wrapper Rule A bans from drivers",
    "_loader_pining.load_statsbomb_matches": "the stream wrapper Rule A bans from drivers",
    "_sb_open_data.load_open_data_matches": "the stream wrapper Rule A bans from drivers",
    "calibrate_tracking_defaults._load_fold": "the documented X item (also in _STREAM_LOADER_EXEMPT)",
}

#: Rule D allowlist -- the only functions that may call load_match with events_only != literal False
#: (spec section 5.4a). Function-granular; closures defined within an entry are included.
_UNADMITTED_EVENTS_ONLY_ALLOWED: dict[str, str] = {
    "_events_admission.events_only_loader": "IS the admitted events-only loader",
    "build_skillcorner_s1_event_validity._events_pass": (
        "the Task-0 producer, which must measure every match's events without the admission gate its own output feeds"
    ),
}


@dataclasses.dataclass(frozen=True)
class Violation:
    """One rule violation: ``module.func`` at ``lineno``, with a human detail."""

    module: str
    func: str  # innermost enclosing function name, or "" at module scope
    lineno: int
    detail: str


@functools.cache
def all_script_trees() -> dict[str, ast.AST]:
    """Every ``scripts/*.py`` parsed once, PRIVATE INCLUDED (spec section 5.2)."""
    return {p.stem: ast.parse(p.read_text(encoding="utf-8")) for p in sorted(_SCRIPTS.glob("*.py"))}


def _call_name(call: ast.Call) -> str:
    return getattr(call.func, "id", "") or getattr(call.func, "attr", "")


def _func_stack(tree: ast.AST) -> dict[int, tuple[str, ...]]:
    """``id(node) -> tuple of enclosing FunctionDef names, outermost first``.

    A Lambda is NOT a FunctionDef, so a call inside a lambda inherits the lambda's enclosing
    FunctionDef -- which is why Rule D's ``events_only_loader`` closure and ``_events_pass``'s lambda
    resolve to their enclosing module-level function (spec section 5.4a).
    """
    out: dict[int, tuple[str, ...]] = {id(tree): ()}

    def rec(node: ast.AST, stack: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[id(child)] = stack
                rec(child, (*stack, child.name))
            else:
                out[id(child)] = stack
                rec(child, stack)

    rec(tree, ())
    return out


@functools.cache
def public_loader_functions() -> dict[str, set[str]]:
    """``module -> {public top-level function names}`` for the three loader modules (spec 5.1)."""
    trees = all_script_trees()
    out: dict[str, set[str]] = {}
    for mod in _LOADER_MODULES:
        tree = trees.get(mod)
        names: set[str] = set()
        if tree is not None:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    names.add(node.name)
        out[mod] = names
    return out


@functools.cache
def corpus_functions() -> frozenset[str]:
    """Public loader-module functions named by a corpus prefix (spec 5.1) -- the derived corpus-call set."""
    names: set[str] = set()
    for mod_names in public_loader_functions().values():
        names |= {n for n in mod_names if n.startswith(_CORPUS_PREFIXES)}
    return frozenset(names)


def _corpus_load_functions() -> frozenset[str]:
    """The corpus functions that LOAD (``load_`` prefix): what an iterable/body must hit for Rule C."""
    return frozenset(n for n in corpus_functions() if n.startswith("load_"))


# --- Rule A -------------------------------------------------------------------------------


def rule_a_tree(mod: str, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    stacks = _func_stack(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in _STREAM_LOADERS:
            stack = stacks.get(id(node), ())
            if any((mod, f) in _STREAM_LOADER_EXEMPT for f in stack):
                continue
            out.append(Violation(mod, stack[-1] if stack else "", node.lineno, f"calls {_call_name(node)}"))
    return out


def rule_a() -> list[Violation]:
    """Stream-loader calls outside ``_STREAM_LOADER_EXEMPT`` (spec section 5.3)."""
    return [v for mod, tree in all_script_trees().items() for v in rule_a_tree(mod, tree)]


# --- Rule B -------------------------------------------------------------------------------


def rule_b_tree(mod: str, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    stacks = _func_stack(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "load_match":
            if not any(kw.arg == "events_only" for kw in node.keywords):
                stack = stacks.get(id(node), ())
                out.append(Violation(mod, stack[-1] if stack else "", node.lineno, "load_match without events_only="))
    return out


def rule_b() -> list[Violation]:
    """``load_match(...)`` calls that omit the ``events_only=`` keyword (spec section 5.4)."""
    return [v for mod, tree in all_script_trees().items() for v in rule_b_tree(mod, tree)]


# --- Rule D -------------------------------------------------------------------------------


def _events_only_value(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "events_only":
            return kw.value
    return None


def rule_d_tree(mod: str, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    stacks = _func_stack(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "load_match"):
            continue
        value = _events_only_value(node)
        if value is None:
            continue  # Rule B's job; here we only police non-literal-False values
        if isinstance(value, ast.Constant) and value.value is False:
            continue
        stack = stacks.get(id(node), ())
        if any(f"{mod}.{f}" in _UNADMITTED_EVENTS_ONLY_ALLOWED for f in stack):
            continue
        out.append(Violation(mod, stack[-1] if stack else "", node.lineno, "unadmitted events_only= load_match"))
    return out


def rule_d() -> list[Violation]:
    """``load_match(events_only=<not literal False>)`` outside the allowlist (spec section 5.4a)."""
    return [v for mod, tree in all_script_trees().items() for v in rule_d_tree(mod, tree)]


# --- Rule C -------------------------------------------------------------------------------


def _enclosing_func_params(tree: ast.AST, node: ast.AST, stacks: dict[int, tuple[str, ...]]) -> set[str]:
    """Parameter names of the innermost FunctionDef enclosing ``node`` (spec section 5.5)."""
    stack = stacks.get(id(node), ())
    if not stack:
        return set()
    target = stack[-1]
    # Find the FunctionDef with that name whose stack matches (nearest enclosing).
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == target:
            fn_stack = stacks.get(id(fn), ())
            if fn_stack == stack[:-1]:
                a = fn.args
                return {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    return set()


def _is_load_call(node: ast.AST, params: set[str], load_fns: frozenset[str]) -> bool:
    """A Call to a corpus ``load_*`` function, or to a parameter whose name starts with ``load``."""
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node)
    return name in load_fns or (name in params and name.startswith("load"))


def _reaching_call(name: str, loop_lineno: int, fn_node: ast.AST) -> ast.Call | None:
    """The nearest preceding assignment ``name = <call>`` in the same function (flow-sensitive)."""
    best: tuple[int, ast.Call] | None = None
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == name and n.lineno < loop_lineno:
                    if best is None or n.lineno > best[0]:
                        best = (n.lineno, n.value)
    return best[1] if best else None


def _fn_node_for(tree: ast.AST, node: ast.AST, stacks: dict[int, tuple[str, ...]]) -> ast.AST | None:
    stack = stacks.get(id(node), ())
    if not stack:
        return None
    target = stack[-1]
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == target:
            if stacks.get(id(fn), ()) == stack[:-1]:
                return fn
    return None


def _iterates_load(iterable: ast.expr, params: set[str], load_fns: frozenset[str], loop: ast.AST, fn_node) -> bool:
    """(i) the iterable is a load call, a load-param call, or a Name reaching such a call (spec 5.5)."""
    if _is_load_call(iterable, params, load_fns):
        return True
    if isinstance(iterable, ast.Name) and fn_node is not None:
        reaching = _reaching_call(iterable.id, getattr(loop, "lineno", 1_000_000), fn_node)
        if reaching is not None and _is_load_call(reaching, params, load_fns):
            return True
    return False


def _body_loads(body_nodes: list[ast.AST], params: set[str], load_fns: frozenset[str]) -> bool:
    """(ii) the body performs a load per item (spec 5.5)."""
    for b in body_nodes:
        for n in ast.walk(b):
            if _is_load_call(n, params, load_fns):
                return True
    return False


def _map_filter_body_loads(call: ast.AST, params: set[str], load_fns: frozenset[str]) -> bool:
    """For ``map``/``filter`` the body is the function argument: a Name, or a lambda's body."""
    if not isinstance(call, ast.Call) or not call.args:
        return False
    fn = call.args[0]
    if isinstance(fn, ast.Name):
        return fn.id in load_fns or (fn.id in params and fn.id.startswith("load"))
    if isinstance(fn, ast.Lambda):
        return _body_loads([fn.body], params, load_fns)
    return False


def rule_c() -> list[Violation]:
    """Un-sharded LOADING loops outside ``_UNSHARDED_LOOP_EXEMPT`` (spec section 5.5).

    Keyed on the innermost enclosing function. A loop is a ``for``, a comprehension, a generator
    expression, or a ``map``/``filter`` call; it is a loading loop iff (i) it iterates a load OR
    (ii) it loads per item. The two are distinct un-sharded shapes -- ``for x in load_matches()``
    (streaming, i-only) and ``[load_open_data_match(r) for r in refs]`` (ii-only) -- and each has its
    own RED plant (spec section 6). A ``for_each(load_matches(...), ...)`` is NOT a loop: the loader
    is an ARGUMENT, so Rule A catches the call while Rule C leaves the (correctly sharded) driver be.
    """
    return [v for mod, tree in all_script_trees().items() for v in rule_c_tree(mod, tree)]


def rule_c_tree(mod: str, tree: ast.AST) -> list[Violation]:
    load_fns = _corpus_load_functions()
    out: list[Violation] = []
    seen: set[str] = set()
    stacks = _func_stack(tree)
    for node in ast.walk(tree):
        iterables: list[ast.expr] = []
        body_nodes: list[ast.AST] = []
        is_map_filter = False
        if isinstance(node, ast.For):
            iterables = [node.iter]
            body_nodes = [*node.body, *node.orelse]
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            iterables = [g.iter for g in node.generators]
            body_nodes = [node.elt, *(i for g in node.generators for i in g.ifs)]
        elif isinstance(node, ast.DictComp):
            iterables = [g.iter for g in node.generators]
            body_nodes = [node.key, node.value, *(i for g in node.generators for i in g.ifs)]
        elif isinstance(node, ast.Call) and _call_name(node) in ("map", "filter") and len(node.args) >= 2:
            iterables = list(node.args[1:])
            is_map_filter = True
        else:
            continue

        params = _enclosing_func_params(tree, node, stacks)
        fn_node = _fn_node_for(tree, node, stacks)
        iterates = any(_iterates_load(it, params, load_fns, node, fn_node) for it in iterables)
        loads = (
            _map_filter_body_loads(node, params, load_fns)
            if is_map_filter
            else _body_loads(body_nodes, params, load_fns)
        )
        if not (iterates or loads):
            continue
        stack = stacks.get(id(node), ())
        func = ".".join(stack)  # FULL enclosing path, so a nested `main._all_matches` is distinct
        qual = f"{mod}.{func}" if func else mod
        if qual in _UNSHARDED_LOOP_EXEMPT:
            continue
        if qual in seen:
            continue
        seen.add(qual)
        out.append(Violation(mod, func, node.lineno, "un-sharded loading loop"))
    return out


# --- Ledgers (populated in Task 8 from the live derivation; drained by Tasks 9-15) ---------

#: Modules with a Rule A violation at this tree (37 at 4ac26d0). Asserted EXACT both ways; each
#: migration (Tasks 9-15) deleted its own rows -- now FULLY DRAINED (every driver migrated).
_RULE_A_PENDING: frozenset[str] = frozenset(set())

#: ADR-056 third bucket: a corpus driver GENUINELY invisible to these rules. Asserted EMPTY -- the
#: rules are complete by enumeration over the derived population, so nothing should need it. The one
#: KNOWN residual (spec section 4.6) is a cross-function load indirection, which is pinned as an
#: uncaught PLANT rather than parked here, because it is a rule LIMIT, not an enrolled driver.
_UNDERIVABLE: dict[str, str] = {}

# NOTE: the key-exception ledger for `for_each(load=...)` drivers whose key is not `ref.key` lives in
# tests/scripts/test_corpus_driver_resilience.py::_KEY_EXCEPTIONS (it is a test-side scaffold, not an
# AST rule); see that file. build_sb360_coverage is one such: its f-string key stays byte-identical to
# the pre-migration `f"{comp}_{season}_{mid}"` so finished shard generations resume across the migration.

#: ``module.function`` with a Rule C violation at this tree (22 non-exempt at 4ac26d0; the 23rd,
#: ``calibrate_tracking_defaults._load_fold``, is in _UNSHARDED_LOOP_EXEMPT). Asserted EXACT both
#: ways -- now FULLY DRAINED (every loading loop migrated into a ``for_each`` pass).
_RULE_C_PENDING: frozenset[str] = frozenset(set())
