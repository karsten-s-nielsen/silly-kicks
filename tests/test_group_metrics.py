"""Domain-free grouped statistics lifted from scripts/xtgk_v2_keeper_discrimination.py.

Behaviour-preservation is the point of the lift, so these mirror the xtgk suite's
assertions and add the cases the script's own tests did not carry.
"""

from __future__ import annotations

import numpy as np

from silly_kicks._group_metrics import DEFAULT_MIN_N, group_spread, icc_one_way


def _grouped(group_means, within_sd, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    vals, keys = [], []
    for k, mu in enumerate(group_means):
        vals.extend(rng.normal(mu, within_sd, n_per))
        keys.extend([f"K{k}"] * n_per)
    return np.array(vals), np.array(keys)


def test_icc_responds_to_within_group_spread_not_degenerate():
    means = [0.0, 0.1, 0.2, 0.3, 0.4]
    v_tight, k = _grouped(means, within_sd=0.01)
    v_noisy, _ = _grouped(means, within_sd=0.50)
    icc_tight = icc_one_way(v_tight, k)
    icc_noisy = icc_one_way(v_noisy, k)
    assert icc_tight > 0.8
    assert icc_noisy < 0.2
    assert icc_tight > icc_noisy  # NOT computed on collapsed means (would ignore within spread)


def test_icc_is_nan_with_fewer_than_two_groups():
    """A single group has no between-group term; returning 0.0 there would read as
    'measured, and flat' rather than 'not measurable'."""
    v, k = _grouped([0.0], within_sd=0.1)
    assert np.isnan(icc_one_way(v, k))


def test_icc_drops_singleton_groups_that_carry_no_within_term():
    """A group of one contributes between-group variance with no within-group counterpart,
    which inflates the ICC. The lifted body filters them; this pins that it still does."""
    v, k = _grouped([0.0, 0.5], within_sd=0.05, n_per=30)
    with_singleton = icc_one_way(np.concatenate([v, [99.0]]), np.concatenate([k, ["SOLO"]]))
    without = icc_one_way(v, k)
    assert np.isclose(with_singleton, without)


def test_group_spread_filters_and_ranks():
    v, k = _grouped([0.0, 0.2, 0.4], within_sd=0.02, n_per=30)
    v = np.concatenate([v, [0.9, 0.9]])
    k = np.concatenate([k, ["SMALL", "SMALL"]])
    s = group_spread(v, k, min_n=20)
    assert s["n_keepers"] == 3  # SMALL (n=2) filtered out
    assert np.isfinite(s["icc"])
    assert [r[0] for r in s["ranking"]][:1] == ["K2"]  # highest mean (0.4) ranked first


def test_group_spread_returns_the_declared_shape_when_everything_is_filtered_out():
    v, k = _grouped([0.0, 0.2], within_sd=0.02, n_per=3)
    s = group_spread(v, k, min_n=20)
    assert np.isnan(s["icc"]) and np.isnan(s["cv"])
    assert s["n_keepers"] == 0 and s["ranking"] == []


def test_default_min_n_is_the_single_source_of_the_registered_floor():
    """The script prints this number in its report header AND uses it as the filter. Two
    copies would silently disagree the moment either moved."""
    assert DEFAULT_MIN_N == 20

    from scripts.xtgk_v2_keeper_discrimination import _MIN_N

    assert _MIN_N is DEFAULT_MIN_N


def test_the_lift_left_no_duplicate_definition_behind():
    """Delete-and-depend: a copy left in scripts/ would drift from the library body."""
    import pathlib

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "xtgk_v2_keeper_discrimination.py"
    text = script.read_text(encoding="utf-8")
    assert "def icc_one_way" not in text
    assert "def keeper_spread" not in text
    assert "def group_spread" not in text
