import numpy as np
import pandas as pd
import pytest

from silly_kicks.xthreat import (
    GridSpec,
    compute_holdout_nll,
    compute_holdout_nll_per_group,
    holdout_split,
    singh_transition_matrix,
)
from tests._xthreat_helpers import _moves


def test_holdout_split_deterministic_and_disjoint():
    df = _moves(n_per_zone=10).assign(game_id=lambda d: d.action_id % 7)
    tr1, ho1 = holdout_split(df, holdout_fraction=0.3)
    _, ho2 = holdout_split(df, holdout_fraction=0.3)
    pd.testing.assert_frame_equal(ho1, ho2)  # deterministic
    assert set(tr1.game_id) & set(ho1.game_id) == set()  # game-level disjoint


def test_compute_holdout_nll_shape_guard():
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    bad = np.zeros((10, 10))
    with pytest.raises(ValueError):
        compute_holdout_nll(bad, _moves(), grid=g)


def test_compute_holdout_nll_synthetic_truth():
    # NLL is lower for the matrix that generated the data than for a uniform matrix.
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    df = _moves(n_per_zone=200)
    T = singh_transition_matrix(df, g)
    n = g.n_zones
    uniform = np.full((n, n), 1.0 / n)
    nll_fit = compute_holdout_nll(T, df, grid=g)
    nll_uniform = compute_holdout_nll(uniform, df, grid=g)
    assert nll_fit < nll_uniform


def test_per_group_returns_dict():
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    df = _moves(n_per_zone=20).assign(game_id=lambda d: d.action_id % 3)
    T = singh_transition_matrix(df, g)
    out = compute_holdout_nll_per_group(T, df, grid=g, group_col="game_id")
    assert isinstance(out, dict) and len(out) == 3
    assert all(isinstance(v, float) for v in out.values())
