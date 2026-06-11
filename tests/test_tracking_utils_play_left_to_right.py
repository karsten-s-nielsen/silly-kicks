"""Unit tests for silly_kicks.tracking.utils.play_left_to_right (tracking variant)."""

import pandas as pd

from silly_kicks.tracking.utils import play_left_to_right


def _row(period_id, frame_id, player_id, team_id, x, y, *, is_ball=False, td: str | None = "rtl"):
    return {
        "game_id": 1,
        "period_id": period_id,
        "frame_id": frame_id,
        "time_seconds": frame_id / 25.0,
        "frame_rate": 25.0,
        "player_id": player_id,
        "team_id": team_id,
        "is_ball": is_ball,
        "is_goalkeeper": False,
        "x": x,
        "y": y,
        "z": float("nan"),
        "speed": 0.0,
        "speed_source": "native",
        "ball_state": "alive",
        "team_attacking_direction": td,
        "confidence": None,
        "visibility": None,
        "source_provider": "gradientsports",
    }


def test_ball_flipped_with_home_player_in_same_period():
    """Ball is flipped when a home-team player in the same period has 'rtl'."""
    frames = pd.DataFrame(
        [
            _row(1, 0, 7, 100, 30.0, 20.0, td="rtl"),  # home player, RTL
            _row(1, 0, None, None, 20.0, 34.0, is_ball=True, td=None),  # ball, converter-realistic
        ]
    )
    out = play_left_to_right(frames, home_team_id=100)
    # Period-level flip: both player and ball flipped
    assert out.iloc[1]["x"] == 85.0
    assert out.iloc[1]["y"] == 68.0 - 34.0


def test_player_rows_flip_consistently_with_ball():
    frames = pd.DataFrame(
        [
            _row(1, 0, 7, 100, 30.0, 20.0, td="rtl"),
            _row(1, 0, None, None, 40.0, 50.0, is_ball=True, td=None),
        ]
    )
    out = play_left_to_right(frames, home_team_id=100)
    # Period-level flip: home player "rtl" triggers flip of ALL rows
    assert out.iloc[0]["x"] == 75.0
    assert out.iloc[0]["y"] == 48.0
    assert out.iloc[1]["x"] == 65.0
    assert out.iloc[1]["y"] == 18.0


def test_ltr_frames_pass_through_unchanged():
    frames = pd.DataFrame([_row(1, 0, 7, 100, 30.0, 20.0, td="ltr")])
    out = play_left_to_right(frames, home_team_id=100)
    assert out.iloc[0]["x"] == 30.0
    assert out.iloc[0]["y"] == 20.0
    assert out.iloc[0]["team_attacking_direction"] == "ltr"


def test_team_attacking_direction_set_to_ltr_after_flip():
    frames = pd.DataFrame([_row(1, 0, 7, 100, 30.0, 20.0, td="rtl")])
    out = play_left_to_right(frames, home_team_id=100)
    assert out.iloc[0]["team_attacking_direction"] == "ltr"


def test_nan_xy_stays_nan():
    frames = pd.DataFrame([_row(1, 0, 7, 100, float("nan"), float("nan"), td="rtl")])
    out = play_left_to_right(frames, home_team_id=100)
    assert pd.isna(out.iloc[0]["x"])
    assert pd.isna(out.iloc[0]["y"])
