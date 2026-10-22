"""`download` access (spec R3): a job fetches exactly the files its units will read -- its own pending
units, a profile job only its sample, a plain dataset Stage its declared files -- and nothing else."""

from __future__ import annotations

import pytest

from scripts._driver import shard_select
from scripts._stages import (
    ItemStage,
    LocalStore,
    Param,
    Pipeline,
    Stage,
    make_context,
    profile_sample,
    reduce_stage,
    run_unit,
    sorted_units,
)
from scripts.hf_jobs import datasets as D


def _refs(ctx):
    from scripts._driver_sources import staged_flat_refs

    return staged_flat_refs(ctx.data_root, ctx.params["dataset"])


def _probe_refs(ctx):
    from scripts._driver_sources import staged_flat_refs

    return staged_flat_refs(ctx.data_root, ctx.params["probe"])


def _pipe() -> Pipeline:
    return Pipeline(
        name="fetchfix",
        module="tests.fixture",
        params=(Param("dataset", str), Param("probe", str)),
        stages=(
            Stage(
                name="check",
                units=lambda ctx: ["only"],
                work=lambda k, ctx: 1,
                reduce=lambda r, ctx: r["only"],
                uses_dataset=True,
                dataset_files=lambda ctx: ["home_teams.json", _refs(ctx)[0].files[0]],
            ),
            ItemStage(
                name="extract",
                refs=_refs,
                key=lambda r: r.stem,
                work=lambda item, ctx: None,
                needs=("check",),
                uses_dataset=True,
            ),
            ItemStage(
                name="held_out",
                refs=_probe_refs,
                key=lambda r: r.stem,
                work=lambda item, ctx: None,
                uses_dataset=True,
                dataset_param="probe",
            ),
        ),
    )


@pytest.fixture
def ctx(tmp_path):
    from scripts._toy_corpora import build_flat_corpus

    build_flat_corpus(tmp_path / "data" / "main", n_games=6, providers=("skillcorner",))
    build_flat_corpus(tmp_path / "data" / "side", n_games=2, providers=("skillcorner",))
    return make_context(
        _pipe(),
        {"dataset": "main", "probe": "side"},
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=tmp_path / "data",
    )


def _files(ctx, st_name, keys):
    refs = ctx.pipeline.stage(st_name).ref_map(ctx)
    return {f for k in keys for f in refs[k].files}


def test_an_item_stage_fetches_only_its_shards_pending_units_plus_the_metadata(ctx):
    keys = sorted_units(ctx, ctx.pipeline.stage("extract"))
    mine = shard_select(keys, 1, 3)
    run_unit(
        ctx, ctx.pipeline.stage("extract"), mine[0], worker="t", ref=ctx.pipeline.stage("extract").ref_map(ctx)[mine[0]]
    )
    got = D.fetch_patterns(ctx, "main", ("extract",), (1, 3), None)
    assert set(got) == _files(ctx, "extract", mine[1:]) | {"home_teams.json"}
    assert not set(got) & _files(ctx, "extract", [k for k in keys if k not in mine[1:]])  # non-vacuity: others exist


def test_a_profile_job_fetches_only_its_sample(ctx):
    keys = sorted_units(ctx, ctx.pipeline.stage("extract"))
    sample = profile_sample(keys, 2)
    assert len(sample) == 2 < len(keys)
    got = D.fetch_patterns(ctx, "main", ("extract",), (0, 1), 2)
    assert set(got) == _files(ctx, "extract", sample) | {"home_teams.json"}


def test_a_plain_dataset_stage_fetches_its_declared_files_until_reduced(ctx):
    first = ctx.pipeline.stage("extract").refs(ctx)[0].files[0]
    assert D.fetch_patterns(ctx, "main", ("check",), (0, 1), None) == sorted({"home_teams.json", first})
    st = ctx.pipeline.stage("check")
    run_unit(ctx, st, "only", worker="t")
    reduce_stage(ctx, "check")
    assert D.fetch_patterns(ctx, "main", ("check",), (0, 1), None) == []


def test_a_grouped_job_fetches_every_stage_it_runs(ctx):
    got = D.fetch_patterns(ctx, "main", ("check", "extract"), (0, 1), None)
    keys = sorted_units(ctx, ctx.pipeline.stage("extract"))
    assert set(got) == _files(ctx, "extract", keys) | {"home_teams.json"}


def test_a_stage_of_another_dataset_is_never_fetched_into_this_one(ctx):
    assert D.fetch_patterns(ctx, "main", ("held_out",), (0, 1), None) == []
    side = D.fetch_patterns(ctx, "side", ("held_out",), (0, 1), None)
    assert set(side) == _files(ctx, "held_out", sorted_units(ctx, ctx.pipeline.stage("held_out"))) | {"home_teams.json"}


def test_nothing_pending_fetches_nothing(ctx):
    st = ctx.pipeline.stage("extract")
    refs = st.ref_map(ctx)
    for k in sorted_units(ctx, st):
        run_unit(ctx, st, k, worker="t", ref=refs[k])
    assert D.fetch_patterns(ctx, "main", ("extract",), (0, 1), None) == []


def test_stage_files_refuses_an_undeclared_plain_stage(ctx):
    st = Stage(name="x", units=lambda c: ["only"], work=lambda k, c: 1, uses_dataset=True)
    with pytest.raises(ValueError, match="dataset_files"):
        D.stage_files(ctx, st, None)
