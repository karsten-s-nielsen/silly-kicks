"""The §3.3 entanglement driver's corpus pass: shards, resume, partitioning, honest scope.

It used to walk ~81 matches serially, in memory, writing nothing until the end -- the exact shape
that lost an 8.7h power run to a single raise in the step afterwards.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.validate_xshot_causal as mod  # bare import: tests/scripts/ has NO __init__.py


def _opp(n: int, *, resolved: bool = True) -> pd.DataFrame:
    """A shard-shaped opportunity frame carrying the REGISTERED confounder set.

    Taken from `SHOT_ARM_CONFOUNDERS` + the config's `gk_block` rather than hand-listed, so a
    change to the registered design fails these tests loudly instead of leaving them passing
    against a column set the analysis no longer uses.
    """
    import numpy as np

    from silly_kicks.causal import SHOT_ARM_CONFOUNDERS, shot_arm_config

    rng = np.random.default_rng(0)
    cols = {c: rng.normal(size=n) for c in SHOT_ARM_CONFOUNDERS}
    cols.update({c: rng.normal(size=n) for c in shot_arm_config({}).gk_block})
    cols.update(
        {
            "carrier_resolved": [resolved] * n,
            "Z": ([0, 1] * ((n // 2) + 1))[:n],
            "Y": ([0, 0, 1] * ((n // 3) + 1))[:n],
            "game_id": [f"g{i % 3}" for i in range(n)],
        }
    )
    return pd.DataFrame(cols)


def _write_shard(out, provider: str, match_id: str, df: pd.DataFrame) -> None:
    d = out / "shards"
    d.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["provider"] = provider
    df["match_id"] = match_id
    df.to_parquet(d / f"{provider}__{match_id}.parquet", index=False)


def test_load_shards_concatenates_every_match(tmp_path):
    _write_shard(tmp_path, "gradientsports", "1", _opp(4))
    _write_shard(tmp_path, "gradientsports", "2", _opp(6))
    assert len(mod.load_shards(tmp_path)) == 10


def test_load_shards_of_an_empty_dir_is_an_empty_frame_not_a_crash(tmp_path):
    assert mod.load_shards(tmp_path).empty


def test_an_EMPTY_shard_is_distinct_from_an_ABSENT_one(tmp_path):
    """Absent means "not yet run"; present-and-empty means "run, produced nothing". Conflating them
    makes every resume recompute the barren matches forever."""
    _write_shard(tmp_path, "gradientsports", "1", _opp(0))
    assert (tmp_path / "shards" / "gradientsports__1.parquet").is_file()
    assert mod.load_shards(tmp_path).empty  # contributes no rows...
    # ...but the file exists, which is what the resume check reads.


def test_provider_survives_the_round_trip_rather_than_being_parsed_from_the_filename(tmp_path):
    """`coverage` is keyed on provider. Re-deriving it by splitting the filename on "__" would
    mis-split any provider containing that separator -- so it is stored as a column."""
    _write_shard(tmp_path, "grad__ients", "1", _opp(2))
    got = mod.load_shards(tmp_path)
    assert set(got["provider"]) == {"grad__ients"}


@pytest.fixture
def spy_analyze(monkeypatch):
    """Replace the 200-replicate placebo analysis with a spy.

    These tests assert SCOPE and PROVENANCE, not the entanglement math (which has its own tests).
    Running the real analysis cost 35s and proved nothing extra -- while the spy additionally
    captures the frame handed over, so the eligibility wiring is checked rather than assumed.
    """
    seen = {}

    def _fake(opp, *, seed=0, n_seeds=200):
        seen["opp"] = opp.copy()
        # Deliberately NOT "ok": `_render` branches on that and would demand the full metric key
        # set a stub cannot honestly supply. These tests assert the metrics dict, not the report.
        return {"status": "stubbed", "n_opportunities": len(opp)}

    monkeypatch.setattr(mod, "analyze", _fake)
    return seen


def _manifest(out, *, dirty: bool, n_matches: int = 1, n_opp: int = 4) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest_p0.json").write_text(
        json.dumps({"n_matches": n_matches, "n_opportunities": n_opp, "run_commit": "abc", "run_tree_dirty": dirty}),
        encoding="utf-8",
    )


def test_analysis_records_the_CORPUS_SCOPE_it_actually_saw(tmp_path, spy_analyze):
    """A partitioned run can legitimately analyse a subset. A metrics.json that does not say how
    much of the corpus it covered is the same defect as a manifest reporting one partition's
    totals as if they were the whole."""
    _write_shard(tmp_path, "gradientsports", "1", _opp(4))
    _manifest(tmp_path, dirty=False)
    m = mod.analyze_shards(
        tmp_path, ["gradientsports"], carrier_min=0.5, seed=0, provenance={"commit": "abc", "dirty": False}
    )
    assert m["corpus"] == {
        "n_matches": 1,
        "n_opportunities": 4,
        "n_partitions": 1,
        "n_shards": 1,
        "commit_consistent": True,
    }


def test_only_carrier_resolved_rows_of_ELIGIBLE_providers_reach_the_analysis(tmp_path, spy_analyze):
    """The eligibility filter is what the carrier-coverage gate exists to enforce; a shard-based
    rewrite that quietly passed everything through would still satisfy the scope assertions."""
    _write_shard(tmp_path, "gradientsports", "1", _opp(4, resolved=True))
    _write_shard(tmp_path, "skillcorner", "2", _opp(6, resolved=False))  # 0.0 coverage -> excluded
    _manifest(tmp_path, dirty=False)
    m = mod.analyze_shards(
        tmp_path,
        ["gradientsports", "skillcorner"],
        carrier_min=0.5,
        seed=0,
        provenance={"commit": "abc", "dirty": False},
    )
    assert m["coverage"]["gradientsports"]["included"] is True
    assert m["coverage"]["skillcorner"]["included"] is False
    assert set(spy_analyze["opp"]["provider"]) == {"gradientsports"}
    assert len(spy_analyze["opp"]) == 4


def test_a_DIRTY_shard_builder_taints_the_analysis_even_from_a_clean_tree(tmp_path, spy_analyze):
    """The analysis step can run clean over shards built dirty. Reporting only the analysis SHA
    would launder the input -- the same rule the power driver applies to its upstream tables."""
    _write_shard(tmp_path, "gradientsports", "1", _opp(4))
    _manifest(tmp_path, dirty=True)
    m = mod.analyze_shards(
        tmp_path, ["gradientsports"], carrier_min=0.5, seed=0, provenance={"commit": "abc", "dirty": False}
    )
    assert m["run_tree_dirty"] is True

    # The other side: an all-clean corpus must NOT be reported dirty, or the flag is decoration.
    other = tmp_path / "clean"
    _write_shard(other, "gradientsports", "1", _opp(4))
    _manifest(other, dirty=False)
    m2 = mod.analyze_shards(
        other, ["gradientsports"], carrier_min=0.5, seed=0, provenance={"commit": "abc", "dirty": False}
    )
    assert m2["run_tree_dirty"] is False


def test_build_only_returns_without_writing_a_corpus_looking_metrics(tmp_path, monkeypatch):
    """A partitioned worker analysing its OWN slice would emit a metrics.json indistinguishable
    from a full-corpus one. `--build-only` is what stops eight workers writing eight of them."""
    monkeypatch.setattr(mod, "build_shards", lambda *a, **k: {"n_matches": 0, "n_opportunities": 0})
    out = mod.run(tmp_path, ["gradientsports"], 0.6, 0, provenance={"commit": "abc", "dirty": False}, build_only=True)
    assert out["status"] == "shards_built"
    assert not (tmp_path / "metrics.json").exists()


def test_out_is_required_unless_listing_matches(monkeypatch):
    monkeypatch.setattr("sys.argv", ["validate_xshot_causal.py"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "--out is required" in str(e.value)
