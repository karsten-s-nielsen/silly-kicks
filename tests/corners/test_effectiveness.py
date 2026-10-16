"""TF-46 chance quality: per-corner shots/goals/xG + per-feature aggregate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.corners import (
    CORNER_CHANCE_COLUMNS,
    aggregate_chance_quality_by_feature,
    compute_corner_chance_quality,
)

_SHOT = 11
_PASS = 0
_SUCCESS = 1
_FAIL = 0


def _corners():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "corner_id": [0, 1],
            "period_id": [1, 1],
            "t_kick": [100.0, 200.0],
            "t_outcome": [106.0, 206.0],
        }
    )


def _actions():
    return pd.DataFrame(
        {
            "game_id": ["g1"] * 4,
            "period_id": [1, 1, 1, 1],
            "time_seconds": [101.0, 103.0, 150.0, 201.0],  # 2 in corner-0 window, 1 outside, 1 in corner-1
            "type_id": [_SHOT, _SHOT, _SHOT, _PASS],
            "result_id": [_SUCCESS, _FAIL, _FAIL, _SUCCESS],
            "xg": [0.5, 0.2, 0.9, np.nan],
        }
    )


def test_counts_and_xg_in_window():
    out = compute_corner_chance_quality(_corners(), _actions(), xg_column="xg")
    c0 = out[out["corner_id"] == 0].iloc[0]
    assert c0["n_shots"] == 2  # the shot at 150 is outside corner-0's [100,106] window
    assert c0["n_goals"] == 1
    assert c0["sum_xg"] == 0.7  # 0.5 + 0.2
    assert list(out.columns) == list(CORNER_CHANCE_COLUMNS)


def test_corner_with_no_shots_keeps_zero_row():
    out = compute_corner_chance_quality(_corners(), _actions(), xg_column="xg")
    c1 = out[out["corner_id"] == 1].iloc[0]  # only a pass in its window
    assert c1["n_shots"] == 0
    assert c1["n_goals"] == 0
    assert c1["sum_xg"] == 0.0


def test_missing_xg_column_raises():
    try:
        compute_corner_chance_quality(_corners(), _actions(), xg_column="post_shot_xg")
    except KeyError as e:
        assert "post_shot_xg" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_aggregate_by_feature_is_weighted_mean_not_ranking():
    quality = compute_corner_chance_quality(_corners(), _actions(), xg_column="xg")
    features = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "corner_id": [0, 1],
            "routine_feat_0": [1.0, 0.0],  # feature 0 loads entirely on corner 0 (xg 0.7)
            "routine_feat_1": [0.0, 1.0],  # feature 1 loads entirely on corner 1 (xg 0.0)
        }
    )
    out = aggregate_chance_quality_by_feature(features, quality)
    assert list(out["routine_feature"]) == ["routine_feat_0", "routine_feat_1"]
    f0 = out[out["routine_feature"] == "routine_feat_0"].iloc[0]
    f1 = out[out["routine_feature"] == "routine_feat_1"].iloc[0]
    assert f0["weighted_mean_sum_xg"] == 0.7  # only corner 0 weighted in
    assert f1["weighted_mean_sum_xg"] == 0.0  # only corner 1


def test_pure_does_not_mutate():
    corners, actions = _corners(), _actions()
    cb, ab = corners.copy(deep=True), actions.copy(deep=True)
    compute_corner_chance_quality(corners, actions, xg_column="xg")
    pd.testing.assert_frame_equal(corners, cb)
    pd.testing.assert_frame_equal(actions, ab)
