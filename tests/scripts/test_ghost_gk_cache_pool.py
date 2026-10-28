"""Spec r4 D19: the `ghost_gk` `cache` stage reads its upstream units in a pool -- the same bytes as the serial read.

At `2474b73` `cache` read its 179 extract results one by one on one core (26.5-28.6 min per variant on HF). It now
reads every unit its upstream's reduce combined through `_stages._read_units` (one listing, `REDUCE_READ_WORKERS`
threads, each result verified once), in the same key order. Since spec r4 D18 (a) that upstream is the per-variant
`select` (the old per-variant extract; `test_ghost_gk_superset_select` proves the two equal). The oracle is the
`2474b73` `_cache_work` body, kept verbatim below except for the name of the stage it reads; the corpus is the
committed toy `ghost_gk` tree, run in-process through the real stage code.
"""

from __future__ import annotations

import json

import pytest

from scripts import _stages as S
from scripts import train_ghost_gk as T
from scripts._toy_corpora import TOY_BUILDERS, TOY_PARAMS
from tests.scripts._stage_harness import PROV, params_from_args

_MODULE = "scripts.train_ghost_gk"


def _serial_cache_reference(ctx, stage: str = "select"):
    """The `2474b73` `_cache_work`, verbatim but for the stage it reads (the oracle): the DONE units, one at a time."""
    from scripts._stages import read_result, sorted_units, unit_states

    st = ctx.pipeline.stage(stage)
    gen = ctx.generation(stage)
    done = [k for k, v in unit_states(ctx, st, sorted_units(ctx, st), verify=False).items() if v == "done"]
    parts = [f for f in (read_result(ctx.store, f"{gen}/{k}") for k in done) if len(f)]
    if not parts:
        raise RuntimeError("no training samples extracted")
    return T._combine_packs(parts)


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    """A store whose `validate`, `extract` and `select` stages ran and reduced in-process on the toy ghost_gk tree."""
    root = tmp_path_factory.mktemp("gk")
    data = TOY_BUILDERS[_MODULE](root / "data")
    ctx = S.make_context(
        T.PIPELINE,
        params_from_args(T.PIPELINE, list(TOY_PARAMS[_MODULE])),
        PROV,
        store=S.LocalStore(root / "s"),
        data_root=data,
    )
    for name in ("validate", "extract", "select"):
        st = ctx.pipeline.stage(name)
        refs = st.ref_map(ctx) if isinstance(st, S.ItemStage) else {}
        for k in S.sorted_units(ctx, st):
            S.run_unit(ctx, st, k, worker="w0", ref=refs.get(k))
        S.reduce_stage(ctx, name)
    return ctx


@pytest.mark.slow  # slow: the toy ghost_gk extract runs in-process, measured 12 s on the Windows dev box (ADR-023)
def test_the_cache_reads_the_extract_units_in_a_pool_and_matches_the_serial_read(extracted, monkeypatch):
    ctx = extracted
    want = S._encode(_serial_cache_reference(ctx))
    pooled: list[tuple[str, list[str]]] = []
    unlisted: list[str] = []
    read_units, read_result = S._read_units, S.read_result

    def spy_read_units(c, name, keys, **kw):
        pooled.append((name, list(keys)))
        return read_units(c, name, keys, **kw)

    def spy_read_result(store, base, *, names=None, timings=None):
        if names is None:  # a read without the generation's listing: one probe chain per key
            unlisted.append(base)
        return read_result(store, base, names=names, timings=timings)

    monkeypatch.setattr(S, "_read_units", spy_read_units)
    monkeypatch.setattr(S, "read_result", spy_read_result)
    got = T._cache_work("only", ctx)
    assert S._encode(got) == want  # the same bytes as the serial read
    summary = json.loads(ctx.store.read_bytes(f"{ctx.generation('select')}/_summary.json"))
    dropped = set(summary["excluded"]) | set(summary["failed_keys"])
    combined = [k for k in S.sorted_units(ctx, ctx.pipeline.stage("select")) if k not in dropped]
    assert pooled == [("select", combined)]  # every unit the reduce combined, in key order, pooled
    assert len(combined) > 1 and unlisted == []  # non-vacuous; no unit is read one key at a time
