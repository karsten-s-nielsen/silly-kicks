"""Spec r4 D18 (b), (d): xshot, xcross and ghost_outfield extract the FAITHFUL features once for both variants, and the
one consumer of the extract output projects `position_only` -- the old `position_only` extract output, column for
column. The xcross `probe` extracts once too: its cohort never depended on the feature set.

At `2474b73` each variant extracted the same matches again (`position_only` is faithful minus 1-4 velocity columns,
the audit's stored-result comparison). The oracles are the `2474b73` extraction functions kept verbatim
(`tests/scripts/_{xshot,xcross,ghost_outfield}_extract_reference.py`), run with the variant's feature set through the
real stage machinery on the committed toy trees. The consumer then sees byte-identical input, so everything it
derives -- the study inputs, the folds, the fits, the bundles -- is unchanged.
"""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from scripts import _stages as S
from scripts._toy_corpora import TOY_BUILDERS, TOY_PARAMS
from scripts.hf_jobs.registry import REGISTRY
from tests.scripts import _ghost_outfield_extract_reference as GO_REF
from tests.scripts import _xcross_extract_reference as XC_REF
from tests.scripts import _xshot_extract_reference as XS_REF
from tests.scripts._stage_harness import PROV, params_from_args

#: job -> (module, oracle, the faithful-only columns a position_only consumer drops, the consumer stage)
CASES = {
    "xshot_occurrence": ("scripts.train_xshot_occurrence", XS_REF, ("speed",), "prep"),
    "xcross_attempt": ("scripts.train_xcross_attempt", XC_REF, ("ball_speed",), "prep"),
    "ghost_outfield": (
        "scripts.train_ghost_outfield",
        GO_REF,
        ("ball_speed", "ball_vx", "ball_vy", "opp_forward_centroid_vx"),
        "cv",
    ),
}


def _ctx(job: str, variant: str, store, data_root, pipeline=None):
    """The variant's context on the toy tree: the campaign's own variant build args over the toy parameters."""
    module = CASES[job][0]
    pipe = pipeline or importlib.import_module(module).PIPELINE
    build = next(b for b in REGISTRY[job].builds if b.variant == variant)
    params = params_from_args(pipe, [*TOY_PARAMS[module], *build.run_args])
    return S.make_context(pipe, params, PROV, store=S.LocalStore(store), data_root=data_root)


def _with_work(pipeline, **works):
    """``pipeline`` with the named stages' work swapped for the oracle's (tokens unchanged)."""
    stages = tuple(dataclasses.replace(s, work=works[s.name]) if s.name in works else s for s in pipeline.stages)
    return dataclasses.replace(pipeline, stages=stages)


def _run(ctx, name: str):
    """Run every unit of ``name`` in-process through the real stage code, reduce it, and return its output."""
    st = ctx.pipeline.stage(name)
    refs = st.ref_map(ctx)
    for k in S.sorted_units(ctx, st):
        assert S.run_unit(ctx, st, k, worker="w0", ref=refs[k]) in ("done", "excluded"), k
    S.reduce_stage(ctx, name)
    return ctx.upstream(name)


@pytest.mark.parametrize("job", sorted(CASES))
def test_the_extract_generation_is_shared_by_both_feature_sets(job, tmp_path):
    tokens = {v: _ctx(job, v, tmp_path, None).tokens for v in ("default", "position_only")}
    assert tokens["default"]["extract"] == tokens["position_only"]["extract"]  # one extraction for both
    consumer = CASES[job][3]
    assert tokens["default"][consumer] != tokens["position_only"][consumer]  # the consumer keys the feature set
    if job == "xcross_attempt":
        assert tokens["default"]["probe"] == tokens["position_only"]["probe"]  # one probe for both (D18 (d))


@pytest.mark.slow  # slow: two extraction passes per variant over a toy tree, in-process (ADR-023)
@pytest.mark.parametrize("job", sorted(CASES))
def test_the_consumer_sees_the_old_extract_of_each_variant(job, tmp_path):
    module, ref, drops, _consumer = CASES[job]
    m = importlib.import_module(module)
    data = TOY_BUILDERS[module](tmp_path / "data")
    oracle = _with_work(m.PIPELINE, extract=ref._extract_work)
    shared = None
    olds = {}
    for variant in ("position_only", "default"):
        olds[variant] = old = _run(_ctx(job, variant, tmp_path / f"old-{variant}", data, oracle), "extract")
        new_ctx = _ctx(job, variant, tmp_path / "new", data)  # ONE store: the extract generation is shared
        if shared is None:
            shared = _run(new_ctx, "extract")
        got = m._extract_for(new_ctx)
        assert list(got.columns) == list(old.columns) and dict(got.dtypes) == dict(old.dtypes), variant
        assert got.equals(old), variant
        assert S._encode(got) == S._encode(old), variant
        assert len(got) > 0, variant
    # Non-vacuous, both sides: the shared extract carries the faithful-only columns, the old position_only did not.
    assert set(drops) <= set(shared.columns) and not set(drops) & set(olds["position_only"].columns)


@pytest.mark.slow  # slow: two probe passes over the toy probe tree, in-process (ADR-023)
def test_the_xcross_probe_cohort_does_not_depend_on_the_feature_set(tmp_path):
    """Spec r4 D18 (d): the probe keeps the matches whose extraction yields rows and records their game groups --
    neither depends on the feature set -- so one faithful probe serves both variants, byte for byte."""
    job = "xcross_attempt"
    m = importlib.import_module(CASES[job][0])
    data = TOY_BUILDERS[CASES[job][0]](tmp_path / "data")
    old = _run(
        _ctx(job, "position_only", tmp_path / "old", data, _with_work(m.PIPELINE, probe=XC_REF._probe_work)), "probe"
    )
    new = _run(_ctx(job, "position_only", tmp_path / "new", data), "probe")
    assert isinstance(new, S.FileSet) and new.files  # non-vacuous: a cohort was written
    assert S._encode(new) == S._encode(old)
