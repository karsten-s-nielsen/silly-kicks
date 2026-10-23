"""S5.9: every callable of a stage lives in ONE stage module; a reusable stage's upstreams are stage modules too;
a stage module's closure never contains its trainer."""

from __future__ import annotations

import importlib
import inspect

import pytest

from scripts import _stages as s

REUSABLE = {
    "scripts.train_ghost_gk": ("validate", "extract"),
    "scripts.train_ghost_outfield": ("extract",),
    "scripts.train_receiver_model": ("extract", "pool"),
    "scripts.train_win_probability": ("refs", "extract"),
    "scripts.train_xcross_attempt": ("extract", "probe"),
    "scripts.train_xshot_occurrence": ("extract",),
    "scripts.build_territory_ranking_census": ("refs", "fit_prepass", "prefit", "score"),
}
_FIELDS = ("work", "load", "reduce", "reduce_order", "units", "refs", "key", "counters", "declare", "dataset_files")


def _callables(st):
    for f in _FIELDS:
        fn = getattr(st, f, None)
        if fn is not None and callable(fn) and not inspect.ismethod(fn):
            yield f, fn


@pytest.mark.parametrize("module", sorted(REUSABLE))
def test_each_reusable_stage_is_declared_in_one_stage_module(module):
    pipe = importlib.import_module(module).PIPELINE
    for name in REUSABLE[module]:
        st = pipe.stage(name)
        homes = {f: fn.__module__ for f, fn in _callables(st)}
        assert len(set(homes.values())) == 1, (name, homes)
        home = next(iter(homes.values()))
        assert home != module and importlib.import_module(home).STAGES[name] is st


@pytest.mark.parametrize("module", sorted(REUSABLE))
def test_a_reusable_stage_has_only_reusable_upstreams(module):
    pipe = importlib.import_module(module).PIPELINE
    for name in REUSABLE[module]:
        assert set(pipe.stage(name).needs) <= set(REUSABLE[module]), name


# ---- the parameter gate (S5.9 "Declared parameters") -----------------------------------------------

#: Valid JSON content for a file-content param, so flipping it exercises `declare` instead of crashing on a
#: bogus path. competitions_json is a list of [competition, season] pairs; match_ids_json a per-provider map.
_FILE_CONTENT = {"competitions_json": "[[1, 2]]", "match_ids_json": '{"statsbomb": ["x"]}'}


def _toy_params(pipe) -> dict:
    from scripts._stages import build_parser
    from scripts._toy_corpora import TOY_PARAMS

    ns = build_parser(pipe).parse_args([*TOY_PARAMS[pipe.module], "--run-all", "--store", "s"])
    return {p.name: getattr(ns, p.name) for p in pipe.params}


def _identities(pipe, params) -> dict:
    """The dataset identities built from the parameter VALUES, as `_dataset_identity` keys them in production."""
    others = {
        st.dataset_param for st in pipe.stages if getattr(st, "uses_dataset", False) and st.dataset_param != "dataset"
    }
    return {
        "dataset_identity": {"name": params.get("dataset")},
        "dataset_identities": {n: {"name": params.get(n)} for n in sorted(others)},
    }


def _token(pipe, params, stage):
    return s.compute_tokens(pipe, params, "c", **_identities(pipe, params))[1][stage]


@pytest.mark.parametrize("module", sorted(REUSABLE))
def test_every_parameter_a_reusable_stage_reads_moves_its_token(module, tmp_path):
    from scripts.hf_jobs.closures import stage_module
    from tests.scripts._param_reads import param_reads, second_value, unresolved_reads

    pipe = importlib.import_module(module).PIPELINE
    base = _toy_params(pipe)
    for name in REUSABLE[module]:
        st = pipe.stage(name)
        sm = importlib.import_module(stage_module(st))
        # Fail closed (S5.9): an unresolvable ctx.params key that is not allow-listed fails the gate.
        assert not unresolved_reads(sm, st), f"{module}/{name}: unresolvable param key(s) {unresolved_reads(sm, st)}"
        reads = param_reads(sm, st)
        for p in reads:
            t0 = _token(pipe, base, name)
            moved = {**base, p: second_value(pipe, p, base[p])}
            try:
                t1 = _token(pipe, moved, name)
            except (OSError, ValueError, TypeError):
                # a file-content param (competitions_json / match_ids_json): a bogus path makes `declare` open
                # it. Flip to a REAL temp JSON file with VALID content -- the content keys the token through
                # `declare`, and the path string differs for a params=-keyed param.
                real = tmp_path / f"{p}.json"
                real.write_text(_FILE_CONTENT.get(p, "null"), encoding="utf-8")
                moved = {**base, p: str(real)}
                t1 = _token(pipe, moved, name)
            assert t0 != t1, f"{module}/{name} reads param {p!r} but its token ignores it"


def _synthetic_stage_module(tmp_path, load_body: str):
    import importlib.util

    src = tmp_path / "psyn.py"
    src.write_text(
        "from scripts._stages import ItemStage\n"
        "def _refs(ctx): return []\n"
        f"def _load(ref, ctx): return {load_body}\n"
        "def _work(item, ctx): return None\n"
        "S = ItemStage(name='x', refs=_refs, key=str, load=_load, work=_work)\n"
        "STAGES = {'x': S}\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("psyn", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_param_gate_fails_closed_on_an_unresolvable_key(tmp_path):
    """S5.9 negative gate: a ctx.params key the gate cannot reduce to a literal is reported (fail closed)."""
    from tests.scripts._param_reads import unresolved_reads

    mod = _synthetic_stage_module(tmp_path, "ctx.params[ref.dataset_param]")
    assert unresolved_reads(mod, mod.S) == {"ref.dataset_param"}


def test_the_param_gate_resolves_a_literal_key(tmp_path):
    """S5.9 positive gate: a literal ctx.params key resolves and leaves no unresolvable site."""
    from tests.scripts._param_reads import param_reads, unresolved_reads

    mod = _synthetic_stage_module(tmp_path, 'ctx.params["dataset"]')
    assert param_reads(mod, mod.S) == {"dataset"} and unresolved_reads(mod, mod.S) == set()


@pytest.mark.parametrize(
    "body",
    [
        "dict(ctx.params)",
        "ctx.params.items()",
        "ctx.params.values()",
        "{**ctx.params}",
        "_imported(ctx.params)",
        "json.dumps(ctx.params)",  # an ATTRIBUTE callee handed the whole params dict (CCTP-IMPL-19)
        "_imported(ctx)",  # a bare ctx handed to an unfollowable, non-framework NAME helper (CCTP-IMPL-19)
        "mod.helper(ctx)",  # a bare ctx handed to an imported-module attribute helper (CCTP-IMPL-19 r4)
        "(lambda p: p)(ctx.params)",  # a DYNAMIC (lambda) callee handed the params dict (CCTP-IMPL-19 r4)
        "scripts._cache.fn(ctx)",  # a DOTTED-MODULE base callee (CCTP-IMPL-19 r5)
        "ext_helper(obj=ctx)",  # ctx handed by KEYWORD argument (CCTP-IMPL-19 r5)
        "ext_helper(*[ctx])",  # ctx handed through a STARRED argument (CCTP-IMPL-19 r5)
        "ext_helper(**{'c': ctx})",  # ctx handed through a double-starred argument (CCTP-IMPL-19 r5)
        "helpmod.fn(p=ctx.params)",  # the params dict handed by keyword to an attribute callee (CCTP-IMPL-19 r5)
    ],
)
def test_the_param_gate_fails_closed_on_a_whole_dict_or_unfollowable_use(tmp_path, body):
    """S5.9 negative gate (ALLOWLIST): a whole-dict use, or ctx / ctx.params handed to an unfollowable callee by
    any argument shape (positional, keyword, starred, dotted/dynamic base), fails closed."""
    from tests.scripts._param_reads import unresolved_reads

    mod = _synthetic_stage_module(tmp_path, body)
    assert unresolved_reads(mod, mod.S), body


def _synthetic_stage_stmts(tmp_path, stmts: str):
    """A synthetic ItemStage whose ``_load(ref, ctx)`` body is ``stmts`` (multi-line), for the alias/seam shapes
    that a single return expression cannot express."""
    import importlib.util

    src = tmp_path / "psyn2.py"
    indented = "\n".join("    " + line for line in stmts.splitlines())
    src.write_text(
        "from scripts._stages import ItemStage\n"
        "def _refs(ctx): return []\n"
        f"def _load(ref, ctx):\n{indented}\n"
        "def _work(item, ctx): return None\n"
        "S = ItemStage(name='x', refs=_refs, key=str, load=_load, work=_work)\n"
        "STAGES = {'x': S}\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("psyn2", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("stmts", "reads", "fails"),
    [
        ("c = ctx\nreturn c.params['x']", {"x"}, False),  # a ctx ALIAS read is captured, not missed (r5)
        ("c = ctx\nreturn ext_helper(c)", set(), True),  # a ctx alias handed to an unfollowable helper fails (r5)
        ("m = helpmod\nreturn m.fn(ctx)", set(), True),  # a module bound to a LOCAL name is not a seam (r5)
        ("st = ctx.pipeline.stage('x')\nreturn st.ref_map(ctx)", set(), False),  # a ctx-derived object seam: OK
    ],
)
def test_the_param_gate_handles_ctx_aliases_and_the_object_seam(tmp_path, stmts, reads, fails):
    """S5.9 ALLOWLIST (CCTP-IMPL-19 r5): a ctx alias is tracked (its reads captured, its escapes flagged); a
    module bound to a local name is NOT a seam; a ctx-derived object method seam stays allowed."""
    from tests.scripts._param_reads import param_reads, unresolved_reads

    mod = _synthetic_stage_stmts(tmp_path, stmts)
    assert param_reads(mod, mod.S) == reads, stmts
    assert bool(unresolved_reads(mod, mod.S)) is fails, stmts


def test_the_param_gate_fails_closed_when_the_stage_call_is_not_found(tmp_path):
    """S5.9 negative gate (CCTP-IMPL-19): a stage whose Stage/ItemStage call the gate cannot locate cannot have
    its reads proven, so it fails closed instead of returning an empty, trusted read set."""
    import types

    from tests.scripts._param_reads import unresolved_reads

    mod = _synthetic_stage_module(tmp_path, '""')  # defines stage 'x' only
    assert unresolved_reads(mod, types.SimpleNamespace(name="not_a_real_stage"))


def test_no_reusable_stage_module_opens_a_file_at_import():
    """CCTP-IMPL-24: a stage module is imported while the worker builds the context -- BEFORE the audit hook is
    installed -- so a module-level file open would escape the open audit (the install-time check_modules
    backstops module LOADS, not top-level opens). No reusable stage module reads a file at import; this gate
    keeps the pre-hook window safe."""
    import ast
    import pathlib

    from scripts.hf_jobs.closures import stage_module

    homes = {
        stage_module(importlib.import_module(module).PIPELINE.stage(name))
        for module in REUSABLE
        for name in REUSABLE[module]
    }
    func_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def module_scope_calls(node):  # every Call reachable from module scope WITHOUT entering a function body
        for child in ast.iter_child_nodes(node):
            if isinstance(child, func_nodes):
                continue
            if isinstance(child, ast.Call):
                yield child
            yield from module_scope_calls(child)

    for home in sorted(homes):
        src = importlib.import_module(home).__file__
        assert src is not None
        tree = ast.parse(pathlib.Path(src).read_text(encoding="utf-8"))
        opens = [
            ast.unparse(c)
            for c in module_scope_calls(tree)
            if (isinstance(c.func, ast.Name) and c.func.id == "open")
            or (isinstance(c.func, ast.Attribute) and c.func.attr in ("open", "read_bytes", "read_text"))
        ]
        assert not opens, f"{home} opens a file at import (CCTP-IMPL-24): {opens}"
