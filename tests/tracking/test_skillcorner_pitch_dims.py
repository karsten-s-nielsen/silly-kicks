"""Pitch-dimension normalisation (spec 3.4). The builder must scale, not offset -- and must
NOT clamp (that is the events transform's job; see tests/spadl/test_skillcorner_coords.py)."""

import pandas as pd
import pytest

from silly_kicks.tracking.skillcorner import convert_to_frames


def _bronze(pitch_length: float, pitch_width: float, *, x: float, y: float) -> pd.DataFrame:
    """One frame, one player, one ball, on a pitch of the given dimensions."""
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "period": 1,
                "frame": 1,
                "timestamp": 0.0,
                "player_id": "p1",
                "team_id": "A",
                "is_goalkeeper": False,
                "x": x,
                "y": y,
                "ball_x": 0.0,
                "ball_y": 0.0,
                "ball_z": 0.0,
                "is_visible": True,
                "frame_rate": 10.0,
                "pitch_length": pitch_length,
                "pitch_width": pitch_width,
            }
        ]
    )


def test_goal_line_lands_on_the_goal_line_for_a_short_pitch():
    # On a 101 m pitch the goal line is at raw x = 50.5. It must map to SPADL x = 105.
    frames, _ = convert_to_frames(
        _bronze(101.0, 67.0, x=50.5, y=0.0), home_team_id="A", output_convention="absolute_frame"
    )
    p = frames[~frames["is_ball"].astype(bool)].iloc[0]
    assert p["x"] == pytest.approx(105.0, abs=1e-9)
    assert p["y"] == pytest.approx(34.0, abs=1e-9)


def test_standard_pitch_is_unchanged():
    # 105 x 68: the new scale must be a NO-OP versus the old +52.5/+34 offset.
    frames, _ = convert_to_frames(
        _bronze(105.0, 68.0, x=10.0, y=-5.0), home_team_id="A", output_convention="absolute_frame"
    )
    p = frames[~frames["is_ball"].astype(bool)].iloc[0]
    assert p["x"] == pytest.approx(62.5, abs=1e-9)
    assert p["y"] == pytest.approx(29.0, abs=1e-9)


def test_off_pitch_positions_survive():
    """The clamp regression. A ball beyond the goal line keeps x > 105 -- goal vs save.

    KILL-LINE: restore the `+ 52.5` offset (or route through the clamping _transform_coords)
    and this test MUST fail. Verify that before moving on.
    """
    b = _bronze(105.0, 68.0, x=0.0, y=0.0)
    b.loc[0, "ball_x"] = 57.0  # 4.5 m beyond the goal line
    b.loc[0, "ball_y"] = 40.0  # 6 m past the touchline
    frames, _ = convert_to_frames(b, home_team_id="A", output_convention="absolute_frame")
    ball = frames[frames["is_ball"].astype(bool)].iloc[0]
    assert ball["x"] > 105.0
    assert ball["y"] > 68.0


def test_missing_pitch_dims_raise():
    """Fail-CLOSED (spec 3.4 / reviewer m1): a silent 105x68 default would reproduce the very
    defect being fixed, and a warning is invisible in a DGX batch log."""
    b = _bronze(105.0, 68.0, x=0.0, y=0.0).drop(columns=["pitch_length", "pitch_width"])
    with pytest.raises(ValueError, match="pitch_length"):
        convert_to_frames(b, home_team_id="A", output_convention="absolute_frame")


def test_assume_standard_pitch_is_the_explicit_opt_in():
    b = _bronze(105.0, 68.0, x=10.0, y=0.0).drop(columns=["pitch_length", "pitch_width"])
    frames, _ = convert_to_frames(b, home_team_id="A", output_convention="absolute_frame", assume_standard_pitch=True)
    p = frames[~frames["is_ball"].astype(bool)].iloc[0]
    assert p["x"] == pytest.approx(62.5, abs=1e-9)
