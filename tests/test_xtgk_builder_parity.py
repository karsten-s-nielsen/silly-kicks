"""xtgk-local builders must be byte-identical to the stock xthreat builders when the move-set
is restricted to pass+dribble+cross (no goal-kicks/throw-ins). Property test over N cohorts."""

import numpy as np
import pytest

from silly_kicks.xtgk._moves import xtgk_action_prob, xtgk_transition_matrix
from silly_kicks.xthreat import GridSpec, kde_smoothed_transition_matrix, singh_transition_matrix
from silly_kicks.xthreat._grid import _action_prob
from silly_kicks.xthreat._params import KDEParams
from tests.xtgk.conftest import (
    CROSS,
    DRIBBLE,
    FAIL,
    PASS,
    SHOT,
    SUCCESS,
    _row,
    make_cohort,
)

GRID = GridSpec(n_zones_x=16, n_zones_y=12)


def _random_pass_only_cohort(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(60):
        t = rng.choice([PASS, DRIBBLE, CROSS])
        res = SUCCESS if rng.random() < 0.7 else FAIL
        rows.append(
            _row(
                i,
                int(t),
                int(res),
                float(rng.uniform(0, 105)),
                float(rng.uniform(0, 68)),
                float(rng.uniform(0, 105)),
                float(rng.uniform(0, 68)),
            )
        )
    rows.append(_row(60, SHOT, SUCCESS, 100, 34, 105, 34))
    return make_cohort(rows)


def _one_zone_mixed_success_cohort():
    """Many mixed-success passes piled into ONE source zone -> exercises the denominator=all /
    numerator=success row-normalization edge, not just the sparse average case."""
    rows = [_row(i, PASS, SUCCESS if i % 3 else FAIL, 10, 34, 60 + (i % 4), 34) for i in range(30)]
    rows.append(_row(30, SHOT, SUCCESS, 100, 34, 105, 34))
    return make_cohort(rows)


@pytest.mark.parametrize("seed", range(8))
def test_singh_parity(seed):
    a = _random_pass_only_cohort(seed)
    assert np.array_equal(xtgk_transition_matrix(a, GRID, method="singh_counts"), singh_transition_matrix(a, GRID))


@pytest.mark.parametrize("seed", range(8))
def test_kde_parity(seed):
    a = _random_pass_only_cohort(seed)
    assert np.allclose(
        xtgk_transition_matrix(a, GRID, method="kde_smoothed", params=KDEParams()),
        kde_smoothed_transition_matrix(a, GRID, KDEParams()),
        atol=1e-12,
    )


@pytest.mark.parametrize("seed", range(8))
def test_action_prob_parity(seed):
    a = _random_pass_only_cohort(seed)
    s0, m0 = xtgk_action_prob(a, 16, 12)
    s1, m1 = _action_prob(a, 16, 12)
    assert np.array_equal(s0, s1) and np.array_equal(m0, m1)


def test_singh_parity_normalization_edge():
    a = _one_zone_mixed_success_cohort()
    assert np.array_equal(xtgk_transition_matrix(a, GRID, method="singh_counts"), singh_transition_matrix(a, GRID))


def test_kde_parity_normalization_edge():
    a = _one_zone_mixed_success_cohort()
    assert np.allclose(
        xtgk_transition_matrix(a, GRID, method="kde_smoothed", params=KDEParams()),
        kde_smoothed_transition_matrix(a, GRID, KDEParams()),
        atol=1e-12,
    )


def test_action_prob_parity_normalization_edge():
    a = _one_zone_mixed_success_cohort()
    s0, m0 = xtgk_action_prob(a, 16, 12)
    s1, m1 = _action_prob(a, 16, 12)
    assert np.array_equal(s0, s1) and np.array_equal(m0, m1)
