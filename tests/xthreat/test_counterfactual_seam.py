import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat import ExpectedThreat, destination_profiles, values_at_points


def _toy_xt() -> ExpectedThreat:
    # Small analytic grid so we can hand-check. Fit on a tiny stream of successful moves PLUS a
    # batch of shots (some goals) — without shots ``_scoring_prob`` is zero everywhere and value
    # iteration converges to an all-zero surface, which ``require_fitted_xt`` (correctly) rejects
    # as unfitted, so a non-degenerate fit needs a scoring signal.
    rng = np.random.default_rng(0)
    n = 400
    sx = rng.uniform(0, 105, n)
    sy = rng.uniform(0, 68, n)
    ex = np.clip(sx + rng.uniform(0, 20, n), 0, 105)
    ey = np.clip(sy + rng.uniform(-10, 10, n), 0, 68)
    passes = pd.DataFrame(
        {
            "type_id": spadlconfig.actiontype_id["pass"],
            "result_id": spadlconfig.result_id["success"],
            "start_x": sx,
            "start_y": sy,
            "end_x": ex,
            "end_y": ey,
        }
    )
    ns = 80
    shx = rng.uniform(80, 105, ns)
    shy = rng.uniform(20, 48, ns)
    shot_result = np.where(rng.uniform(0, 1, ns) < 0.2, spadlconfig.result_id["success"], spadlconfig.result_id["fail"])
    shots = pd.DataFrame(
        {
            "type_id": spadlconfig.actiontype_id["shot"],
            "result_id": shot_result,
            "start_x": shx,
            "start_y": shy,
            "end_x": shx,
            "end_y": shy,
        }
    )
    actions = pd.concat([passes, shots], ignore_index=True)
    return ExpectedThreat(l=16, w=12).fit(actions)


def test_zone_values_match_values_at_points_at_centres():
    xt = _toy_xt()
    prof = destination_profiles(xt, np.array([30.0]), np.array([34.0]))
    # zone_values are xT at the zone centres — must equal values_at_points at those centres.
    expect = values_at_points(xt, prof.zone_centres[:, 0], prof.zone_centres[:, 1])
    np.testing.assert_allclose(prof.zone_values, expect, rtol=0, atol=1e-9)


def test_probabilities_are_the_origin_row_of_the_transition_matrix():
    xt = _toy_xt()
    from silly_kicks.xthreat._grid import _get_flat_indexes

    ox, oy = np.array([30.0]), np.array([34.0])
    prof = destination_profiles(xt, ox, oy)
    cell = int(_get_flat_indexes(pd.Series(ox), pd.Series(oy), xt.l, xt.w).to_numpy()[0])
    assert xt.transition_matrix is not None  # fitted by _toy_xt() above
    np.testing.assert_allclose(prof.probabilities[0], xt.transition_matrix[cell], rtol=0, atol=0)


def test_renormalized_distribution_is_family_agnostic_valid():
    xt = _toy_xt()
    prof = destination_profiles(xt, np.array([30.0]), np.array([34.0]))
    row = prof.probabilities[0]
    subset = row > 0
    q = row[subset] / row[subset].sum()
    assert abs(q.sum() - 1.0) < 1e-12  # a valid distribution after renormalization


def test_fails_closed_on_unfitted_and_str_and_none():
    # require_fitted_xt raises NotImplementedError (str), ValueError (None), NotFittedError (unfitted).
    for bad in [ExpectedThreat(), "singh_counts", None]:
        with pytest.raises((NotImplementedError, ValueError, NotFittedError)):
            destination_profiles(bad, np.array([1.0]), np.array([1.0]))
