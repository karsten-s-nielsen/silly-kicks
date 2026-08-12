# tests/tracking/test_player_influence_aggregator.py
"""Tests for add_player_influence aggregator + per-Series helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions


@pytest.fixture
def xt_grid():
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


@pytest.fixture
def sportec_data(xt_grid):
    frames = load_provider_frames("sportec")
    actions = synthesize_actions(frames, n_actions=5)
    return actions, frames, xt_grid


_OUTPUT_COLS = [
    "actor_reachable_area_m2",
    "off_ball_xt_team",
    "off_ball_xt_opponent",
    "off_ball_xt_diff",
    "reachable_area_team",
    "reachable_area_opponent",
    "reachable_area_diff",
]


def test_add_player_influence_output_columns(sportec_data):
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    result = add_player_influence(
        actions,
        frames,
        xt,
    )
    for col in _OUTPUT_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_diff_identity(sportec_data):
    """_diff = _team - _opponent (exact equality)."""
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    result = add_player_influence(actions, frames, xt)

    valid = result["off_ball_xt_team"].notna()
    pd.testing.assert_series_equal(
        result.loc[valid, "off_ball_xt_diff"],
        (result.loc[valid, "off_ball_xt_team"] - result.loc[valid, "off_ball_xt_opponent"]).rename("off_ball_xt_diff"),
    )
    pd.testing.assert_series_equal(
        result.loc[valid, "reachable_area_diff"],
        (result.loc[valid, "reachable_area_team"] - result.loc[valid, "reachable_area_opponent"]).rename(
            "reachable_area_diff"
        ),
    )


def test_provenance_columns_added(sportec_data):
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    result = add_player_influence(actions, frames, xt)

    provenance = {"frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"}
    assert provenance.issubset(result.columns)


def test_provenance_skip_guard(sportec_data):
    """Calling add_player_influence twice doesn't create _x/_y suffixed columns."""
    from silly_kicks.tracking.features import add_player_influence

    actions, frames, xt = sportec_data
    result = add_player_influence(actions, frames, xt)
    result2 = add_player_influence(result, frames, xt)

    for col in ["frame_id", "time_offset_seconds"]:
        bad_x = f"{col}_x"
        bad_y = f"{col}_y"
        assert bad_x not in result2.columns, f"Found {bad_x} — skip guard failed"
        assert bad_y not in result2.columns, f"Found {bad_y} — skip guard failed"


# --- Per-Series helpers ---


@pytest.mark.parametrize(
    "helper_name",
    [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "reachable_area_team",
        "reachable_area_opponent",
    ],
)
def test_per_series_helper_returns_series(sportec_data, helper_name):
    from silly_kicks.tracking import features

    actions, frames, xt = sportec_data
    fn = getattr(features, helper_name)
    result = fn(actions, frames, xt)
    assert isinstance(result, pd.Series)
    assert len(result) == len(actions)


@pytest.mark.parametrize(
    "helper_name",
    [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "reachable_area_team",
        "reachable_area_opponent",
    ],
)
def test_per_series_helper_none_frames(helper_name, xt_grid):
    """frames=None -> all NaN (column-name probing tolerance)."""
    from silly_kicks.tracking import features

    actions = pd.DataFrame(
        {
            "action_id": [1, 2],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [1.0, 2.0],
            "team_id": [1, 2],
            "player_id": [10, 60],
        }
    )
    fn = getattr(features, helper_name)
    result = fn(actions, None, xt_grid)
    assert result.isna().all()


# --- VAEP xfns ---


def test_player_influence_xfns_column_names(xt_grid):
    """feature_column_names probing (empty frames) returns 21 columns."""
    from silly_kicks.tracking.features import player_influence_xfns

    xfns = player_influence_xfns(xt_grid)
    assert len(xfns) == 1

    # Simulate VAEP probing: 10-row dummy actions, no frames
    dummy = pd.DataFrame(
        {
            "action_id": range(10),
            "game_id": 1,
            "period_id": 1,
            "time_seconds": np.arange(10, dtype=float),
            "team_id": 1,
            "player_id": 10,
            "type_id": 0,
            "result_id": 0,
            "bodypart_id": 0,
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 55.0,
            "end_y": 34.0,
        }
    )
    states = [dummy, dummy.copy(), dummy.copy()]

    transformer = xfns[0]
    result = transformer(states, None)

    # 7 base columns x 3 slots = 21
    assert result.shape[1] == 21
    assert result.isna().all().all()

    # Verify column naming pattern
    for col_base in _OUTPUT_COLS:
        for slot in range(3):
            expected = f"{col_base}_a{slot}"
            assert expected in result.columns, f"Missing VAEP column: {expected}"
