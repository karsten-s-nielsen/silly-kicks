import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as cfg
from silly_kicks.xthreat import ExpectedThreat, KDEParams
from silly_kicks.xthreat._grid import _get_flat_indexes
from silly_kicks.xthreat._params import GridSpec
from silly_kicks.xthreat._transitions import (
    _zone_centres,
    kde_smoothed_transition_matrix,
    silverman_2d,
)
from tests._xthreat_helpers import _moves


def test_dispatch_actually_swaps_transition():
    # cheap guard: kde_smoothed must produce a different transition matrix than singh_counts.
    # (xT is compared in test_dispatch_swaps_xt_with_shots — the pass-only _moves corpus has no
    # shots, so scoring_prob is 0 and xT is degenerately all-zeros for both methods.)
    df = _moves(n_per_zone=80)
    singh = ExpectedThreat(l=6, w=4, method="singh_counts").fit(df)
    kde = ExpectedThreat(l=6, w=4, method="kde_smoothed").fit(df)
    assert not np.array_equal(singh.transition_matrix, kde.transition_matrix)  # type: ignore[arg-type]


def test_dispatch_swaps_xt_with_shots():
    # With shots present (non-zero scoring prob), the two methods yield different xT surfaces too.
    df = _moves(n_per_zone=80)
    shots = df.iloc[:10].copy()
    shots["type_id"] = cfg.actiontype_id["shot"]
    shots["result_id"] = cfg.result_id["success"]
    shots["action_id"] = range(10_000, 10_010)
    df = pd.concat([df, shots], ignore_index=True)
    singh = ExpectedThreat(l=6, w=4, method="singh_counts").fit(df)
    kde = ExpectedThreat(l=6, w=4, method="kde_smoothed").fit(df)
    assert np.any(singh.xT > 0)
    assert not np.array_equal(singh.xT, kde.xT)


def test_silverman_2d_formula():
    assert silverman_2d(64, 2.0) == pytest.approx(64 ** (-1 / 6) * 2.0)


def test_zone_centres_invert_flat_index():
    g = GridSpec(n_zones_x=4, n_zones_y=3)
    centres = _zone_centres(g)
    assert centres.shape == (12, 2)
    # the centre of each zone must map back to that flat index
    xs = pd.Series(centres[:, 0])
    ys = pd.Series(centres[:, 1])
    flat = _get_flat_indexes(xs, ys, g.n_zones_x, g.n_zones_y).to_numpy()
    np.testing.assert_array_equal(flat, np.arange(12))


@pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
def test_kde_rows_stochastic(bandwidth):
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    T = kde_smoothed_transition_matrix(_moves(), g, KDEParams(bandwidth=bandwidth, adaptive=True))
    assert T.shape == (24, 24)
    np.testing.assert_allclose(T.sum(axis=1), np.ones(24), atol=1e-9)


def test_kde_zero_event_row_uses_populated_mean():
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    T = kde_smoothed_transition_matrix(_moves(), g, KDEParams(bandwidth=1.0))
    # an unobserved source zone still sums to 1 (fallback), never all-zero
    assert np.all(np.isclose(T.sum(axis=1), 1.0))


def test_kde_concentrates_as_bandwidth_shrinks():
    # The bandwidth knob controls smoothing: a smaller bandwidth yields peakier (lower-entropy)
    # rows. (Point-evaluation-at-centres KDE does not reproduce Singh's argmax exactly near cell
    # boundaries, so we assert the smoothing behaviour, not argmax equality.)
    g = GridSpec(n_zones_x=6, n_zones_y=4)
    df = _moves(n_per_zone=200)

    def mean_row_entropy(mat):
        ents = []
        for row in mat:
            if row.sum() > 0:
                p = row[row > 0]
                ents.append(-np.sum(p * np.log(p)))
        return float(np.mean(ents))

    tight = kde_smoothed_transition_matrix(df, g, KDEParams(bandwidth=0.5, adaptive=False))
    wide = kde_smoothed_transition_matrix(df, g, KDEParams(bandwidth=5.0, adaptive=False))
    assert mean_row_entropy(tight) < mean_row_entropy(wide)
