"""E14 (ADR-128): pre-drawing every replicate from the ONE stream, then fitting in any order, is byte-identical."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts._crossed_icc import bootstrap_icc_and_power


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n = 600
    d = rng.integers(0, 40, n)
    t = (d // 4 + rng.integers(0, 2, n)) % 12
    y = rng.normal(size=n) + 0.3 * (d % 5)
    return y, d.astype(str), t.astype(str)


def test_reversed_replicate_map_equals_serial():
    y, d, t = _toy()
    kw = {"n_boot": 60, "alpha": 0.05, "effect_size": 0.05, "rng_seed": 7}
    serial = bootstrap_icc_and_power(y, d, t, **kw)
    seen: list[str] = []

    def rmap(task):  # deliberately reversed completion; the function must combine by replicate index
        fn, items = task
        out = {}
        for k in sorted(items, reverse=True):
            out[k] = fn(*items[k])
            seen.append(k)
        return out

    assert bootstrap_icc_and_power(y, d, t, replicate_map=rmap, **kw) == serial
    assert len(seen) == 60 and seen != sorted(seen), "non-vacuity: every replicate ran through the map, out of order"
    assert serial["n_boot_effective"] > 30


@pytest.mark.slow  # slow: measured 5 s on the Windows dev box (ADR-023)
def test_the_census_bootstrap_on_an_inner_pool_equals_the_serial_bootstrap(tmp_path):
    """The census's own replicate map (`_replicate_map`: chunks of 50 replicates per pool task, the arrays
    persisted once per unit) on a REAL InnerPool at W=2: byte-identical Tier-2, and all 8 chunks went
    through the pool. (Completion order of equal-cost chunks is not asserted -- it would be flaky; the
    any-order combination is the reversed-map test above.)"""
    import build_territory_ranking_census as census

    from scripts.hf_jobs.executor import InnerPool

    rng = np.random.default_rng(3)
    n = 900
    d = rng.integers(0, 45, n)
    t = (d // 3 + rng.integers(0, 3, n)) % 15
    dtm = pd.DataFrame(
        {
            "game_id": rng.integers(0, 30, n).astype(str),
            "player_id": d.astype(str),
            "team_id": t.astype(str),
            census._METRIC_COL: rng.normal(size=n) + 0.3 * (d % 5),
            census._VOLUME_COL: rng.integers(1, 60, n),
        }
    )
    serial = census.tier2_icc(dtm)
    pool = InnerPool(2)
    try:
        parallel = census.tier2_icc(dtm, replicate_map=census._replicate_map(pool, tmp_path / "boot"))
        log = pool.drain_log()
    finally:
        pool.shutdown()
    assert parallel == serial
    assert sorted(log) == [f"{c:04d}" for c in range(8)], "non-vacuity: the 8 chunks ran through the pool"
    assert serial["n_boot_effective"] > 0, "non-vacuity: the toy design really bootstrapped"
