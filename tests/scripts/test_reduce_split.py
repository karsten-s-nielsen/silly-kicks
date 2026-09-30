"""Task 6b guard: the DAS reduce split -- shards-only workers + ONE reduce == the serial artifact.

The reduce's population/exclusion counters come from the per-worker manifests (aggregated), NOT the
shards, so an excluded match (no output row) is counted correctly across MULTIPLE workers (DPL-PLAN-10).
"""

import json

import _loader_pining as lp

from scripts import validate_das_native_parity as D
from scripts._item_outcome import ItemExcluded
from tests.scripts.test_das_native_parity_driver import _CLEAN_PROV, _corpus


def _split_corpus():
    """The golden two-game corpus + a third ref whose loader raises ItemExcluded (SB360-like)."""
    refs, load, stub_ref = _corpus()
    excluded = lp.MatchRef("skillcorner", "999", {})

    def load2(ref):
        if ref.match_id == "999":
            raise ItemExcluded("velocity-less freeze-frame (structural)")
        return load(ref)

    return [*refs, excluded], load2, stub_ref


def test_das_reduce_only_equals_serial_with_excluded_and_two_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_numba_available", lambda: False)  # skip the numba compile in a unit test
    refs, load, stub_ref = _split_corpus()  # [g1, g2, excluded-999]

    # Parallel: worker A scores g1; worker B scores g2 AND hits the excluded ref -> two manifests.
    par = tmp_path / "par"
    shard_root = par / "shards"
    D.run_corpus(
        [refs[0]],
        load,
        par,
        prov=_CLEAN_PROV,
        shard_root=shard_root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="A",
    )
    D.run_corpus(
        refs[1:],
        load,
        par,
        prov=_CLEAN_PROV,
        shard_root=shard_root,
        direction_col="dir",
        reference_leg=stub_ref,
        shards_only=True,
        worker_tag="B",
    )
    assert not (par / "metrics.json").exists()  # workers write NO combined artifact
    D.reduce_parity_artifact(refs, shard_root, par, prov=_CLEAN_PROV)  # ONE reduce: all refs + all manifests

    # Serial reference over the identical corpus.
    ser = tmp_path / "ser"
    D.run_corpus(
        refs, load, ser, prov=_CLEAN_PROV, shard_root=ser / "shards", direction_col="dir", reference_leg=stub_ref
    )

    par_m = json.loads((par / "metrics.json").read_text(encoding="utf-8"))
    ser_m = json.loads((ser / "metrics.json").read_text(encoding="utf-8"))
    # The parity RESULT is deterministic; `timings_ms_per_frame` is wall-clock profiling measured at
    # map time and baked into the shards, so it differs between ANY two runs (serial vs serial too).
    # Byte-identity is asserted over the result, modulo the timing block.
    _drop_timings(par_m)
    _drop_timings(ser_m)
    assert par_m == ser_m  # identical parity result + population, summed identically across workers
    assert par_m["n_attempted"] == 3  # all three items reached (attempt is counted before exclusion)
    assert par_m["n_excluded"] == 1  # aggregated from the manifests, invisible to the shards
    assert par_m["providers"]["skillcorner"]["n_matches_scored"] == 2  # only g1 + g2 produced rows
    assert par_m["population"]["listed_per_provider"]["skillcorner"] == 3  # listed from the FULL refs


def _drop_timings(metrics: dict) -> None:
    for prov in metrics.get("providers", {}).values():
        prov.pop("timings_ms_per_frame", None)
