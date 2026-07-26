"""TF-19 sign-off package: the ICC plasmode power curve (spec §5.3, F2).

`ICC_ANCHORS` shipped in PR-3 promising "a power curve is reported at all three" that no code could
produce, while spec §6.1 registers that curve as a PRECONDITION on the ICC gate. This is that curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks._group_metrics import _icc_fast, icc_one_way, icc_power_curve


@pytest.mark.parametrize("seed", range(6))
def test_fast_icc_matches_the_shipped_icc_exactly(seed):
    """`icc_power_curve` uses a numpy-only ICC because the permutation loop evaluates it thousands
    of times. `icc_one_way` is shipped and consumer-tested, so it is UNTOUCHED -- and the fast
    path's equivalence is gated here rather than assumed.

    Includes unbalanced groups AND a singleton (which `icc_one_way` DROPS rather than letting it
    inflate the estimate); a fixture of equal-sized groups would not discriminate that branch.
    """
    rng = np.random.default_rng(seed)
    sizes = [7, 3, 11, 2, 1, 5]  # unbalanced + one singleton
    labels = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    values = rng.normal(size=len(labels))
    codes = np.unique(labels, return_inverse=True)[1]
    assert _icc_fast(values, codes, len(sizes)) == pytest.approx(
        icc_one_way(values, labels), rel=1e-12, abs=1e-12, nan_ok=True
    )


def test_fast_icc_returns_nan_when_fewer_than_two_groups_survive():
    """ "Not measurable", never 0.0 -- which would read as "measured, and flat"."""
    values = np.array([0.1, 0.2])
    codes = np.array([0, 1])
    assert np.isnan(_icc_fast(values, codes, 2))  # both singletons


def _corpus(n_groups: int = 30, per_group: int = 40, seed: int = 0):
    """Keepers must SPAN matches, and matches must hold more than one keeper.

    MEASURED, do not "simplify" this fixture. With ``blocks`` 1:1 with ``groups`` (one keeper per
    match) the block-permutation null is a pure RELABELLING of an identical partition, and
    ``icc_one_way`` is label-invariant -- observed ICC 0.213228 and all five nulls 0.213228, so
    nothing is ever detectable. Spec §6.1's own floor language ("for a single-match keeper, keeper ==
    match") excludes exactly this shape.

    A 2-keepers-per-block fixture does NOT fix it either: the null stops equalling the observed value
    but stops VARYING (constant 0.162025), so a test goes green on a fixed comparison that proves
    nothing. Keepers spanning matches is what makes the null vary.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    groups = np.repeat([f"k{i}" for i in range(n_groups)], per_group)
    blocks = np.array([f"m{(i // 10) % (n // 20)}" for i in range(n)])
    values = rng.normal(0.0, 1.0, size=n)
    return values, groups, blocks


def test_power_is_high_at_a_large_injected_effect():
    values, groups, blocks = _corpus()
    out = icc_power_curve(values, groups, blocks, anchors=(0.30,), n_replicates=40, rng_seed=1)
    assert out["power"][0.30] >= 0.80


def test_power_collapses_to_alpha_at_zero_injected_effect():
    """The other side. A one-sided "power is high" assertion passes identically when the simulator
    silently produces nothing."""
    values, groups, blocks = _corpus()
    out = icc_power_curve(values, groups, blocks, anchors=(0.0,), n_replicates=100, rng_seed=1)
    assert out["power"][0.0] <= 0.15  # alpha=0.05 + Monte-Carlo slack


def test_injection_measurably_moves_the_data_non_vacuity():
    values, groups, blocks = _corpus()
    out = icc_power_curve(values, groups, blocks, anchors=(0.30,), n_replicates=10, rng_seed=1)
    assert out["mean_observed_icc"][0.30] > out["mean_observed_icc_at_zero"] + 0.05


def test_block_structure_inflates_the_NULL_which_is_what_plasmode_means():
    """ "Plasmode, not i.i.d." has teeth only if block structure changes the null. Assert on the null
    MEAN -- the only form of this claim that is stable.

    MEASURED over 20 seeds x 40 permutations; do not "improve" this into another statistic::

        null MEAN clustered > iid : 20/20 at anchor 0.02, 20/20 at anchor 0.30
        null P95  clustered > iid : 13/20 at anchor 0.02, 20/20 at anchor 0.30
        POWER     iid >= clustered: FAILS (0.000 vs 0.013 -- both on the noise floor)

    Block permutation reassigns whole CHUNKS, so the permuted grouping retains real clustering and
    the null sits higher; i.i.d. permutation fully randomises and collapses it toward zero. That is a
    structural property of the null, which is why it survives at both anchors, while power and p95
    comparisons are noise-dominated at the small ones.
    """
    values, groups, real_blocks = _corpus()
    iid_blocks = np.array([f"b{i}" for i in range(len(values))])
    clustered = icc_power_curve(values, groups, real_blocks, anchors=(0.02,), n_replicates=40, rng_seed=3)
    iid = icc_power_curve(values, groups, iid_blocks, anchors=(0.02,), n_replicates=40, rng_seed=3)
    assert clustered["mean_null_icc"][0.02] > iid["mean_null_icc"][0.02]
