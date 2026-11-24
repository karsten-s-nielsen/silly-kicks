"""One AST walk over `scripts/*.py`, shared by every gate that derives a driver population.

ADR-052's corpus-driver gate and Cycle B's artifact-driver gate need the same scaffolding --
glob, skip private, parse -- and differ only in their predicate. Two independent walkers would
drift with nothing relating them, which is the defect class this cycle exists to remove. The
reconciliation is structural: single-source the UNIVERSE, let the predicates differ.
"""

from __future__ import annotations

import ast
import functools
import pathlib
from collections.abc import Iterable, Iterator, Mapping

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


@functools.cache
def iter_scripts() -> dict[str, ast.AST]:
    """Every non-private script in `scripts/`, parsed once per session, keyed by stem.

    Cached because both gates call this from inside PARAMETRISED test bodies, not just at
    collection. Measured: 97.6 ms to parse the 48 scripts, and ADR-052's population gate alone
    reaches it 23 times for its 22 drivers -- 2.2 s from one module, before the artifact-driver
    gate's own calls. `@functools.cache` is the established idiom here (the `spadlconfig`
    DataFrames use it).

    The cache assumes `scripts/` does not change mid-session, which holds: a test run never
    writes a driver. Callers must not mutate the returned dict -- it is shared.
    """
    return {
        p.stem: ast.parse(p.read_text(encoding="utf-8"))
        for p in sorted(SCRIPTS.glob("*.py"))
        if not p.name.startswith("_")
    }


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Identity of every leading Expr(Constant(str)) that IS a docstring."""
    out: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(n, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def string_literals(tree: ast.AST) -> set[str]:
    """Every string literal EXCEPT docstrings.

    Docstrings must be excluded or every content predicate fires on PROSE. Measured: three
    separate source-text scans were fooled during this cycle's review --
    `make_ghost_gk_golden` matched a `_weights` rule solely because its module docstring
    mentions `test_weights_bundle_golden.py`; `render_sb360_matrix` matched a `_provenance`
    scan through the sentence "No provenance guard, deliberately."; and
    `regenerate_gs_et_native_gk` matched a corpus-loader scan through a docstring saying it
    MIRRORS the loader. None of the three carried the literal in code.

    `tests/scripts/test_provenance_wiring.py` already learned this once -- its
    `_shells_out_to_rev_parse` is AST-matched on CALLS with the comment "they cannot tell a
    described defect from a committed one". This is that same lesson, in the shared seam so
    both populations inherit it.
    """
    skip = _docstring_ids(tree)
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip
    }


def called_names(tree: ast.AST) -> set[str]:
    """Every called function/method name -- `f(...)` and `x.f(...)` alike."""
    return {
        (getattr(n.func, "id", "") or getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)
    }


def coordination_corpus_drivers() -> tuple[str, ...]:
    """The TF-58 corpus drivers, DERIVED (review A-54; ADR-056): every non-private script that goes through the shared
    ``run_params_token`` seam -- the M-4 anchor every TF-58 corpus pass uses, and ONLY they (``corpus_source`` alone
    also matches ``train_receiver_model``). Single-sourced here so the worker-combine token guards and the pre-flight
    gate derive the SAME population from one predicate instead of each keeping its own hand-list."""
    return tuple(sorted(stem for stem, tree in iter_scripts().items() if "run_params_token" in called_names(tree)))


# --------------------------------------------------------------------------------------------------------------------
# Module walkers and the call graph (spec 2026-11-08 tracking-batch, R12): ONE parse of silly_kicks/ and scripts/ for
# every gate that derives a population from code -- the group_rows caller set (ADR-073) and the W9 link population.
# --------------------------------------------------------------------------------------------------------------------

REPO = SCRIPTS.parent


@functools.cache
def _parse_modules(root: pathlib.Path, base: str) -> tuple[tuple[str, ast.AST, str], ...]:
    """Every ``<root>/<base>/**/*.py`` as (dotted module name, tree, source). An ``__init__.py`` is keyed by its
    import name (``silly_kicks.tracking``). The source is read with universal newlines, so a CRLF checkout and an LF
    checkout give the same text."""
    out = []
    for py in sorted((root / base).rglob("*.py")):
        parts = py.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        src = py.read_text(encoding="utf-8")
        out.append((".".join(parts), ast.parse(src), src))
    return tuple(out)


def iter_package_modules(root: pathlib.Path = REPO) -> dict[str, ast.AST]:
    """Every module under ``<root>/silly_kicks``, parsed once per session, keyed by dotted module name."""
    return {name: tree for name, tree, _src in _parse_modules(root, "silly_kicks")}


def iter_script_modules(root: pathlib.Path = REPO) -> dict[str, ast.AST]:
    """Every module under ``<root>/scripts`` (private and nested included), keyed by dotted module name.

    ``iter_scripts`` is the non-private top-level driver universe; this is the whole tree (``scripts._driver``,
    ``scripts.hf_jobs.campaign``), which a caller-set gate needs."""
    return {name: tree for name, tree, _src in _parse_modules(root, "scripts")}


def module_sources(root: pathlib.Path = REPO, base: str = "silly_kicks") -> dict[str, str]:
    """The source text of every module that :func:`iter_package_modules` / :func:`iter_script_modules` parse."""
    return {name: src for name, _tree, src in _parse_modules(root, base)}


_DYNAMIC_IMPORTERS = frozenset({"import_kernels", "import_module"})
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)


def _blocks(st: ast.stmt) -> list[list[ast.stmt]]:
    """The nested statement lists of an if/for/while/with/try statement."""
    out = [getattr(st, "body", []), getattr(st, "orelse", []), getattr(st, "finalbody", [])]
    out.extend(h.body for h in getattr(st, "handlers", []))
    return out


def _flat_body(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements of a body, descending into if/for/while/with/try blocks but not into definitions."""
    for st in body:
        yield st
        if isinstance(st, _BLOCKS):
            for blk in _blocks(st):
                yield from _flat_body(blk)


class _Scope:
    __slots__ = ("defs", "imports", "is_class", "qual")

    def __init__(self, qual: str, *, is_class: bool) -> None:
        self.qual = qual
        self.is_class = is_class
        self.defs: dict[str, str] = {}
        self.imports: dict[str, tuple[str, str, str | None]] = {}


class CallGraph:
    """Static reference graph over a set of parsed modules.

    Nodes are ``module:qualname`` for functions, classes and methods (nested closures as
    ``module:factory.<locals>.inner``), and ``module:NAME`` for module-level assignments (globals). An edge
    ``a -> b`` means the definition of ``a`` (decorators included) names ``b``. The edges are:

    - a direct call or any other reference to a function or class. A function passed as a value (an xfn list, a
      dispatch table, a callback) is reachable too;
    - the factory rule: a reference inside a nested function is also an edge of every enclosing function, and a
      function has an edge to each function that it defines;
    - a class has an edge to each of its methods; ``self.m`` / ``cls.m`` inside a method resolves to the method;
    - a dynamic kernel import: ``import_kernels("<module>")`` or ``importlib.import_module("<module>")`` with a
      string literal is an edge to every definition of that module (A2-PLAN-48);
    - a global has edges to the functions and globals that its value names.

    Resolution follows ``import``, ``from ... import`` (relative too), and re-exports through ``__init__`` modules,
    function-local imports included. Local-variable shadowing is ignored, so the graph over-approximates. A name
    outside the parsed modules (pandas, builtins) is dropped.
    """

    def __init__(self, trees: Mapping[str, ast.AST], packages: frozenset[str], sources: Mapping[str, str]) -> None:
        self._trees = trees
        self._packages = packages
        self._sources = sources
        self._modscope: dict[str, _Scope] = {}
        self._scopes: dict[int, _Scope] = {}
        self._globals: dict[str, set[str]] = {}
        self._nodes: dict[str, tuple[str, list[ast.stmt]]] = {}
        self._out: dict[str, set[str]] = {}
        self._in: dict[str, set[str]] = {}
        for mod, tree in trees.items():
            self._index_module(mod, tree)
        for mod, tree in trees.items():
            self._link_body(mod, [self._modscope[mod]], getattr(tree, "body", []))

    # -- indexing ---------------------------------------------------------------------------------------------------

    def _abs_module(self, mod: str, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = mod if mod in self._packages else mod.rpartition(".")[0]
        for _ in range(node.level - 1):
            base = base.rpartition(".")[0]
        return f"{base}.{node.module}" if node.module else base

    def _record_import(self, scope: _Scope, mod: str, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    scope.imports[a.asname] = ("module", a.name, None)
                else:
                    top = a.name.partition(".")[0]
                    scope.imports[top] = ("module", top, None)
            return
        src = self._abs_module(mod, node)
        for a in node.names:
            if a.name != "*":
                scope.imports[a.asname or a.name] = ("from", src, a.name)

    def _index_scope(self, mod: str, scope: _Scope, body: list[ast.stmt], prefix: str) -> None:
        for st in _flat_body(body):
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                self._record_import(scope, mod, st)
            elif isinstance(st, _DEFS):
                qual = f"{prefix}{st.name}"
                scope.defs[st.name] = f"{mod}:{qual}"
                self._nodes[f"{mod}:{qual}"] = (mod, [st])
                inner = _Scope(f"{mod}:{qual}", is_class=isinstance(st, ast.ClassDef))
                self._scopes[id(st)] = inner
                self._index_scope(mod, inner, st.body, f"{qual}." if inner.is_class else f"{qual}.<locals>.")

    def _index_module(self, mod: str, tree: ast.AST) -> None:
        scope = _Scope(mod, is_class=False)
        self._modscope[mod] = scope
        body = getattr(tree, "body", [])
        self._index_scope(mod, scope, body, "")
        stmts: dict[str, list[ast.stmt]] = {}
        for st in _flat_body(body):
            for name in _assigned_names(st):
                if name not in scope.defs:
                    stmts.setdefault(name, []).append(st)
        self._globals[mod] = set(stmts)
        for name, sts in stmts.items():
            self._nodes[f"{mod}:{name}"] = (mod, sts)

    # -- resolution -------------------------------------------------------------------------------------------------

    def _is_module(self, name: str) -> bool:
        return name in self._trees or any(m.startswith(name + ".") for m in self._trees)

    def _resolve_in_module(self, mod: str, name: str, seen: frozenset[tuple[str, str]] = frozenset()) -> str | None:
        """``module:qual`` of ``name`` at the top level of ``mod``, or ``"module=<m>"`` for a module object."""
        if f"{mod}.{name}" in self._trees and (mod, name) not in seen:
            sc = self._modscope.get(mod)
            if sc is None or (name not in sc.defs and name not in self._globals[mod] and name not in sc.imports):
                return f"module={mod}.{name}"
        if (mod, name) in seen or mod not in self._modscope:
            return None
        seen = seen | {(mod, name)}
        sc = self._modscope[mod]
        if name in sc.defs:
            return sc.defs[name]
        if name in self._globals[mod]:
            return f"{mod}:{name}"
        if name in sc.imports:
            return self._resolve_import(sc.imports[name], seen)
        return None

    def _resolve_import(
        self, imp: tuple[str, str, str | None], seen: frozenset[tuple[str, str]] = frozenset()
    ) -> str | None:
        kind, src, orig = imp
        if kind == "module":
            return f"module={src}" if self._is_module(src) else None
        assert orig is not None
        if f"{src}.{orig}" in self._trees:
            return f"module={src}.{orig}"
        return self._resolve_in_module(src, orig, seen)

    def _resolve_name(self, mod: str, chain: list[_Scope], name: str) -> str | None:
        for sc in reversed(chain[1:]):
            if sc.is_class:
                continue  # class-body names are not visible inside methods
            if name in sc.defs:
                return sc.defs[name]
            if name in sc.imports:
                return self._resolve_import(sc.imports[name])
        return self._resolve_in_module(mod, name)

    def _resolve_expr(self, mod: str, chain: list[_Scope], node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self._resolve_name(mod, chain, node.id)
        if not isinstance(node, ast.Attribute):
            return None
        if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            cls = next((sc for sc in reversed(chain) if sc.is_class), None)
            return cls.defs.get(node.attr) if cls is not None else None
        base = self._resolve_expr(mod, chain, node.value)
        if base is None:
            return None
        if base.startswith("module="):
            m = base[len("module=") :]
            if f"{m}.{node.attr}" in self._trees:
                return f"module={m}.{node.attr}"
            return self._resolve_in_module(m, node.attr)
        node_def = self._nodes.get(base)
        if node_def is not None:
            sc = self._scopes.get(id(node_def[1][0]))
            if sc is not None and sc.is_class:
                return sc.defs.get(node.attr)
        return None

    # -- linking ----------------------------------------------------------------------------------------------------

    def _add(self, src: str, dst: str) -> None:
        if src != dst:
            self._out.setdefault(src, set()).add(dst)
            self._in.setdefault(dst, set()).add(src)

    def module_definitions(self, m: str) -> list[str]:
        """Every top-level definition (function or class) of module ``m``, sorted."""
        sc = self._modscope.get(m)
        return sorted(sc.defs.values()) if sc is not None else []

    def _link_def(
        self,
        mod: str,
        chain: list[_Scope],
        st: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        owners: list[str],
    ) -> None:
        inner = self._scopes[id(st)]
        mine = [*owners, inner.qual]
        for o in owners:
            self._add(o, inner.qual)
        for dec in st.decorator_list:
            self._refs(mod, chain, dec, mine)
        if isinstance(st, ast.ClassDef):
            for part in (*st.bases, *st.keywords):
                self._refs(mod, chain, part, mine)
            for q in inner.defs.values():
                self._add(inner.qual, q)
            self._link_body(mod, [*chain, inner], st.body, [inner.qual])
        else:
            self._refs(mod, chain, st.args, mine)
            if st.returns is not None:
                self._refs(mod, chain, st.returns, mine)
            self._link_body(mod, [*chain, inner], st.body, mine)

    def _link_body(self, mod: str, chain: list[_Scope], body: list[ast.stmt], owners: list[str] | None = None) -> None:
        in_class = chain[-1].is_class
        for st in body:
            if isinstance(st, _DEFS):
                # a method is not owned by its class (the class -> method edge is added once in _link_def)
                self._link_def(mod, chain, st, [] if in_class else owners or [])
            elif owners is None and isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                value = st.value
                if value is not None:
                    self._refs(mod, chain, value, [f"{mod}:{n}" for n in _assigned_names(st)])
            elif isinstance(st, _BLOCKS):
                for blk in _blocks(st):
                    self._link_body(mod, chain, blk, owners)
                if owners:
                    for part in _block_headers(st):
                        self._refs(mod, chain, part, owners)
            elif owners:
                self._refs(mod, chain, st, owners)

    def _refs(self, mod: str, chain: list[_Scope], node: ast.AST, owners: list[str]) -> None:
        """Every reference under ``node`` is an edge from each of ``owners``. A nested definition opens a scope."""
        if not owners:
            return
        if isinstance(node, _DEFS):
            self._link_def(mod, chain, node, owners)
            return
        if isinstance(node, ast.Call):
            fn = node.func
            fname = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            arg0 = node.args[0] if node.args else None
            if fname in _DYNAMIC_IMPORTERS and isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                for q in self.module_definitions(arg0.value):
                    for o in owners:
                        self._add(o, q)
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(getattr(node, "ctx", None), ast.Load):
            tgt = self._resolve_expr(mod, chain, node)
            if tgt is not None and not tgt.startswith("module="):
                for o in owners:
                    self._add(o, tgt)
            if isinstance(node, ast.Attribute):
                self._refs(mod, chain, node.value, owners)
            return
        for child in ast.iter_child_nodes(node):
            self._refs(mod, chain, child, owners)

    # -- queries ----------------------------------------------------------------------------------------------------

    def definition(self, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
        """The AST node of a definition node (``None`` for a global)."""
        node = self._nodes.get(qualname)
        if node is None or not isinstance(node[1][0], _DEFS):
            return None
        return node[1][0]  # type: ignore[return-value]

    def functions(self) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        """Every function and method node, keyed by qualname."""
        return {
            q: st[0]
            for q, (_m, st) in self._nodes.items()
            if isinstance(st[0], (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def has_module(self, name: str) -> bool:
        """True when ``name`` is one of the parsed modules."""
        return name in self._trees

    def nodes(self) -> frozenset[str]:
        """Every node (definitions and globals)."""
        return frozenset(self._nodes)

    def is_global(self, qualname: str) -> bool:
        """True for a module-level assignment node."""
        mod, _, name = qualname.partition(":")
        return name in self._globals.get(mod, ())

    def callees(self, qualname: str) -> set[str]:
        """Functions and classes that ``qualname`` references (calls, closures, dynamic kernel imports)."""
        return {q for q in self._out.get(qualname, ()) if not self.is_global(q)}

    def callers(self, qualname: str) -> set[str]:
        """Nodes that reference ``qualname``."""
        return set(self._in.get(qualname, ()))

    def referenced_globals(self, qualname: str) -> set[str]:
        """Module-level assignments that ``qualname`` names."""
        return {q for q in self._out.get(qualname, ()) if self.is_global(q)}

    def closure(self, seeds: Iterable[str]) -> frozenset[str]:
        """Every node reachable from ``seeds`` through callees and referenced globals (the seeds included)."""
        todo, seen = list(seeds), set()
        while todo:
            q = todo.pop()
            if q not in seen:
                seen.add(q)
                todo.extend(self._out.get(q, ()))
        return frozenset(seen)

    def source(self, qualname: str) -> str:
        """The exact source of a node: a definition with its decorators, or every module-level statement that
        assigns the global, in file order. Raises ``KeyError`` when the graph was built without sources."""
        mod, stmts = self._nodes[qualname]
        lines = self._sources[mod].splitlines()
        parts = []
        for st in stmts:
            start = min([st.lineno, *(d.lineno for d in getattr(st, "decorator_list", []))])
            parts.append("\n".join(lines[start - 1 : st.end_lineno]))
        return "\n".join(parts)

    def resolve(self, module: str, name: str) -> str | None:
        """The node that ``name`` denotes at the top level of ``module`` (following re-exports), or None."""
        r = self._resolve_in_module(module, name)
        return None if r is None or r.startswith("module=") else r


def _assigned_names(st: ast.stmt) -> list[str]:
    if isinstance(st, ast.Assign):
        targets: list[ast.expr] = list(st.targets)
    elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
        targets = [st.target]
    else:
        return []
    return [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]


def _block_headers(st: ast.stmt) -> list[ast.AST]:
    """The expressions of an if/for/while/with header (the test, the iterable, the context managers)."""
    out: list[ast.AST] = []
    for field in ("test", "iter"):
        if hasattr(st, field):
            out.append(getattr(st, field))
    out.extend(getattr(st, "items", []))
    return out


def call_graph(
    trees: Mapping[str, ast.AST],
    *,
    packages: Iterable[str] = (),
    sources: Mapping[str, str] | None = None,
) -> CallGraph:
    """The :class:`CallGraph` over ``trees`` (dotted module name -> tree).

    A module is a package when another key starts with ``<name>.``, or when it is in ``packages``. ``sources``
    (module -> text) enables :meth:`CallGraph.source`."""
    inferred = {m for m in trees if any(k.startswith(m + ".") for k in trees)}
    return CallGraph(trees, frozenset(inferred | set(packages)), sources or {})
