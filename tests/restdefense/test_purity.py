"""Purity gate for restdefense (ADR-033): compute_/summarize_ never mutate caller inputs.

Two variants for the ``visible_area`` present/absent branch, and the with-``visible_area`` variant
snapshots the caller-supplied polygon arrays (the array-arg purity rule)."""

import numpy as np
import pandas as pd

from silly_kicks.restdefense._compute import compute_rest_defense, summarize_rest_defense
from tests.restdefense._fixtures import make_rest_defense_fixture

_WHOLE_PITCH = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])


def test_compute_is_pure_without_visible_area():
    actions, frames = make_rest_defense_fixture()
    a_before, f_before = actions.copy(), frames.copy()
    out, _ = compute_rest_defense(actions, frames)
    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)
    assert out is not actions and out is not frames


def test_compute_is_pure_with_visible_area_and_polygon_arrays():
    actions, frames = make_rest_defense_fixture()
    polygon = _WHOLE_PITCH.copy()
    poly_snapshot = polygon.copy()
    va = pd.DataFrame({"action_id": [0, 1, 2, 3], "polygon": [polygon] * 4})
    va_before = va.copy()
    a_before, f_before = actions.copy(), frames.copy()
    compute_rest_defense(actions, frames, visible_area=va)
    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)
    pd.testing.assert_frame_equal(va, va_before)
    np.testing.assert_array_equal(polygon, poly_snapshot)  # the caller's polygon ndarray is untouched


def test_summarize_is_pure():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)
    s_before = samples.copy()
    out = summarize_rest_defense(samples, by="match")
    pd.testing.assert_frame_equal(samples, s_before)
    assert out is not samples


def test_compute_is_pure_with_xt_and_field_weight():
    """ADR-033: the Layer-2 path (xt + danger_field_weight) must not mutate caller inputs -- it builds
    a keeper-removed frame via boolean indexing (a new object), never in place."""
    from silly_kicks.restdefense import RestDefenseParams
    from tests.restdefense._fixtures import make_fitted_xt

    actions, frames = make_rest_defense_fixture()
    a_before, f_before = actions.copy(), frames.copy()
    compute_rest_defense(actions, frames, xt=make_fitted_xt(), params=RestDefenseParams(danger_field_weight=True))
    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)
