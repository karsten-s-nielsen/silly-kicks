"""Every Cycle 1 campaign instance runs its committed recipe: the args that carry corpus scope and search size
(HFU spec 14.2, Change 7a; launch-3 findings F6, F7, F8, F10)."""

from __future__ import annotations

import pathlib
import types

import pytest

from scripts.hf_jobs import campaign as C

CYCLE1 = pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml")
COMMIT = "abcdef1" + "0" * 33


def _instance(job: str, tier: str | None = None, variant: str = "default") -> C.Instance:
    insts = C.instances(C.load_campaign(CYCLE1), COMMIT)
    return next(i for i in insts if i.job == job and (tier is None or i.tier == tier) and i.variant == variant)


def _has(args, *seq: str) -> bool:
    args = list(args)
    return any(args[i : i + len(seq)] == list(seq) for i in range(len(args)))


def test_instance_recipe_args():
    assert "--all-competitions" in _instance("pass_completion").args
    xs = _instance("xsuccess").args
    assert "--all-competitions" in xs and _has(xs, "--hpo-trials", "30")
    for job in ("pass_completion", "xsuccess"):  # native pipelines (HFU spec 15.5a), no adapter
        assert _instance(job).module == f"scripts.train_{job}"
    assert "--all-competitions" in _instance("build_territory_ranking_census").args
    rec = _instance("receiver_model")
    assert _has(rec.args, "--pool-dataset", "tc3") and _has(rec.args, "--pool-provider", "gradientsports")
    assert "tc3" in {d.name for d in rec.datasets}
    for job in ("xshot_occurrence", "xcross_attempt"):
        for tier in ("public", "sc_extended"):
            for variant in ("default", "position_only"):
                assert _has(_instance(job, tier, variant).args, "--n-trials", "50"), (job, tier, variant)


def test_all_competitions_uses_the_manifest(monkeypatch):
    """The shared open-data `refs` parameters (xsuccess, pass_completion, win_probability) walk every release."""
    import scripts._open_data_stages as od
    import scripts._sb_open_data as sbod

    pairs = [(43, 106), (55, 43)]
    monkeypatch.setattr(sbod, "all_open_competitions", lambda: pairs)
    monkeypatch.setattr(sbod, "assert_statsbomb_open_data_mode", lambda: None)
    params = {"all_competitions": True, "competition_id": None, "season_id": None}
    assert od.competitions(params) == pairs


@pytest.mark.parametrize("other", [{"competition_id": 43}, {"season_id": 106}])
def test_all_competitions_is_exclusive(other):
    import scripts._open_data_stages as od

    with pytest.raises(ValueError, match="mutually exclusive"):
        od.competitions({"all_competitions": True, "competition_id": None, "season_id": None, **other})


def test_pool_refs_keep_pool_provider(monkeypatch):
    import scripts._driver_sources as src
    import scripts.train_receiver_model as R

    refs = [
        src.StagedRef(p, g, f"{p}__{g}", sp, (f"shards/g/{p}__{g}.parquet",))
        for p, sp in (("gradientsports", "gradientsports"), ("skillcorner", "skillcorner"), ("idsse", "sportec"))
        for g in ("1", "2")
    ]
    monkeypatch.setattr(src, "staged_flat_refs", lambda _root, _ds: refs)
    ctx = types.SimpleNamespace(
        data_root=None, params={"pool_dataset": "tc3", "pool_provider": "gradientsports", "match_ids_json": None}
    )
    assert [r.stem for r in R._staged_refs(ctx, "pool_dataset")] == ["gradientsports__1", "gradientsports__2"]
