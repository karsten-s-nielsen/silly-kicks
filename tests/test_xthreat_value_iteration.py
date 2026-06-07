import numpy as np
import pandas as pd

import silly_kicks.spadl.config as cfg
from silly_kicks.xthreat import ExpectedThreat, KDEParams
from silly_kicks.xthreat._value_iteration import value_iteration
from tests._xthreat_helpers import _moves


def _moves_with_shots(n_per_zone=100):
    df = _moves(n_per_zone=n_per_zone)
    shots = df.iloc[:20].copy()
    shots["type_id"] = cfg.actiontype_id["shot"]
    shots["result_id"] = cfg.result_id["success"]
    shots["action_id"] = range(10_000, 10_020)
    return pd.concat([df, shots], ignore_index=True)


def test_max_iter_none_matches_bounded_when_converged():
    rows, cols = 4, 6
    rng = np.random.default_rng(0)
    p_scoring = rng.random((rows, cols)) * 0.2
    p_shot = rng.random((rows, cols)) * 0.3
    p_move = 1 - p_shot
    T = rng.random((rows * cols, rows * cols))
    T = T / T.sum(axis=1, keepdims=True)
    xt_unbounded, _ = value_iteration(p_scoring, p_shot, p_move, T, eps=1e-7)
    xt_bounded, _ = value_iteration(p_scoring, p_shot, p_move, T, eps=1e-7, max_iter=10_000)
    np.testing.assert_array_equal(xt_unbounded, xt_bounded)


def test_max_iter_caps_nonconverging_loop():
    # Non-physical inputs (passed directly, not via the model, so p_shot+p_move need not be 1):
    # gs=0.5 constant injection + p_move=1 + row-stochastic T => operator spectral radius 1,
    # xT grows by 0.5 every iteration and NEVER converges. Without the cap this loops forever;
    # with max_iter=5 it returns after exactly 5 iterations.
    rows, cols = 2, 2
    p_scoring = np.full((rows, cols), 0.5)
    p_shot = np.ones((rows, cols))  # gs = p_scoring * p_shot = 0.5
    p_move = np.ones((rows, cols))
    T = np.full((rows * cols, rows * cols), 1.0 / (rows * cols))  # row-stochastic
    xt, heatmaps = value_iteration(p_scoring, p_shot, p_move, T, eps=1e-9, max_iter=5)
    assert len(heatmaps) == 6  # initial snapshot + exactly 5 capped iterations
    assert np.all(np.isfinite(xt))


def test_kde_dense_matrix_converges_quickly():
    # KDE produces a dense T; with shots present (non-zero scoring prob) the fixed point is
    # non-trivial, so this genuinely exercises dense-matrix value iteration. It must converge
    # in a sane number of iterations and yield a non-zero, finite surface.
    m = ExpectedThreat(l=6, w=4, method="kde_smoothed", params=KDEParams(bandwidth=2.0)).fit(
        _moves_with_shots(n_per_zone=100)
    )
    assert np.isfinite(m.xT).all()
    assert np.any(m.xT > 0)  # non-trivial fixed point (shots seed scoring prob)
    assert 1 < len(m.heatmaps) < 500  # dense but still a contraction (shot prob > 0 somewhere)
