"""Unit tests for silly_kicks.tracking.utils.link_actions_to_frames."""

import pandas as pd

from silly_kicks.tracking.utils import link_actions_to_frames


def _frame_row(period_id, frame_id, t):
    return {
        "game_id": 1,
        "period_id": period_id,
        "frame_id": frame_id,
        "time_seconds": t,
        "frame_rate": 25.0,
        "player_id": 7,
        "team_id": 100,
        "is_ball": False,
        "is_goalkeeper": False,
        "x": 50.0,
        "y": 34.0,
        "z": float("nan"),
        "speed": 5.0,
        "speed_source": "native",
        "ball_state": "alive",
        "team_attacking_direction": "ltr",
        "confidence": None,
        "visibility": None,
        "source_provider": "pff",
    }


def _action_row(action_id, period_id, t):
    return {
        "game_id": 1,
        "action_id": action_id,
        "period_id": period_id,
        "time_seconds": t,
        "team_id": 100,
        "player_id": 7,
        "type_id": 0,
        "result_id": 1,
        "bodypart_id": 0,
        "start_x": 50.0,
        "start_y": 34.0,
        "end_x": 60.0,
        "end_y": 34.0,
    }


def test_exact_time_match_link_quality_one():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0)])
    pointers, report = link_actions_to_frames(actions, frames)
    assert pointers.iloc[0]["time_offset_seconds"] == 0.0
    assert pointers.iloc[0]["link_quality_score"] == 1.0
    assert report.n_actions_linked == 1


def test_link_within_tolerance():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 1, 0.15)])
    pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.2)
    # offset convention: action_time - frame_time (positive => action is after the frame)
    assert abs(pointers.iloc[0]["time_offset_seconds"] - 0.11) < 1e-9
    assert report.n_actions_linked == 1


def test_unlinked_outside_tolerance():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 1, 0.15)])
    pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.1)
    assert pd.isna(pointers.iloc[0]["frame_id"])
    assert pd.isna(pointers.iloc[0]["time_offset_seconds"])
    assert pd.isna(pointers.iloc[0]["link_quality_score"])
    assert report.n_actions_unlinked == 1
    assert report.n_actions_linked == 0


def test_empty_actions_returns_empty_pointer():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0)])
    actions = pd.DataFrame(columns=["action_id", "period_id", "time_seconds", "team_id"])
    pointers, report = link_actions_to_frames(actions, frames)
    assert len(pointers) == 0
    assert report.n_actions_in == 0
    assert report.link_rate == 0.0


def test_action_with_no_frame_in_period():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 0.04)])
    actions = pd.DataFrame([_action_row(0, 2, 0.0)])
    pointers, report = link_actions_to_frames(actions, frames)
    assert pd.isna(pointers.iloc[0]["frame_id"])
    assert report.n_actions_unlinked == 1


def test_multi_candidate_chooses_closest():
    frames = pd.DataFrame(
        [
            _frame_row(1, 0, 0.10),
            _frame_row(1, 1, 0.20),
            _frame_row(1, 2, 0.30),
        ]
    )
    actions = pd.DataFrame([_action_row(0, 1, 0.18)])
    pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.2)
    assert pointers.iloc[0]["frame_id"] == 1
    assert report.n_actions_multi_candidate >= 1


def test_cross_period_does_not_link():
    frames = pd.DataFrame([_frame_row(2, 0, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0)])
    pointers, report = link_actions_to_frames(actions, frames)
    assert pd.isna(pointers.iloc[0]["frame_id"])
    assert report.n_actions_unlinked == 1


def test_default_tolerance_is_0_2_seconds():
    """Memory: tests-crossing-pipelines-need-default-stable-params.

    Pins the default to 0.2 s. Changing the default requires updating
    this test --- preventing silent default drift breaking downstream callers.
    """
    frames = pd.DataFrame([_frame_row(1, 0, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.19)])
    pointers_default, _ = link_actions_to_frames(actions, frames)
    assert not pd.isna(pointers_default.iloc[0]["frame_id"])

    actions2 = pd.DataFrame([_action_row(0, 1, 0.21)])
    pointers_default2, _ = link_actions_to_frames(actions2, frames)
    assert pd.isna(pointers_default2.iloc[0]["frame_id"])
