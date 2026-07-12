"""W5 (4.45.0): action-level keeper-discrimination ICC (R2 -- NOT degenerate-on-means)."""

import numpy as np

from scripts.xtgk_v2_keeper_discrimination import icc_one_way, keeper_spread


def _grouped(keeper_means, within_sd, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    vals, keys = [], []
    for k, mu in enumerate(keeper_means):
        vals.extend(rng.normal(mu, within_sd, n_per))
        keys.extend([f"K{k}"] * n_per)
    return np.array(vals), np.array(keys)


def test_icc_responds_to_within_keeper_spread_not_degenerate():
    # Same distinct keeper means; ICC must be HIGH when within-keeper spread is small, LOW when it's large.
    means = [0.0, 0.1, 0.2, 0.3, 0.4]
    v_tight, k = _grouped(means, within_sd=0.01)
    v_noisy, _ = _grouped(means, within_sd=0.50)
    icc_tight = icc_one_way(v_tight, k)
    icc_noisy = icc_one_way(v_noisy, k)
    assert icc_tight > 0.8  # keepers well separated relative to within-keeper noise
    assert icc_noisy < 0.2  # separation swamped by within-keeper noise
    assert icc_tight > icc_noisy  # ICC is NOT computed on collapsed means (would ignore within spread)


def test_keeper_spread_filters_and_ranks():
    v, k = _grouped([0.0, 0.2, 0.4], within_sd=0.02, n_per=30)
    # add a keeper below the min-N filter -> excluded
    v = np.concatenate([v, [0.9, 0.9]])
    k = np.concatenate([k, ["SMALL", "SMALL"]])
    s = keeper_spread(v, k, min_n=20)
    assert s["n_keepers"] == 3  # SMALL (n=2) filtered out
    assert np.isfinite(s["icc"])
    assert [r[0] for r in s["ranking"]][:1] == ["K2"]  # highest mean (0.4) ranked first
