"""orient_frames_to_ltr_by_geometry --- promoted ADR-053 geometric frame-LTR net."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.direction import orient_frames_to_ltr_by_geometry


def _frame(period, team, player, x, y, is_gk=False, is_ball=False):
    return {
        "game_id": "g1",
        "period_id": period,
        "frame_id": 0,
        "time_seconds": 0.0,
        "frame_rate": 10.0,
        "player_id": player,
        "team_id": team,
        "is_ball": is_ball,
        "is_goalkeeper": is_gk,
        "x": x,
        "y": y,
        "z": np.nan,
        "speed": 1.0,
        "speed_source": "derived",
        "ball_state": None,
        "team_attacking_direction": None,
        "confidence": None,
        "visibility": None,
        "source_provider": "skillcorner",
        "is_goalkeeper_source": "native",
    }


def _two_period_match(home_gk_x_p1, home_gk_x_p2):
    """home GK + an away outfield marker per period; asymmetric/extreme positions."""
    rows = [
        _frame(1, "H", "hgk", home_gk_x_p1, 5.0, is_gk=True),
        _frame(1, "A", "afw", 100.0, 60.0),
        _frame(1, "A", "agk", 100.0, 34.0, is_gk=True),
        _frame(2, "H", "hgk", home_gk_x_p2, 5.0, is_gk=True),
        _frame(2, "A", "afw", 5.0, 60.0),
        _frame(2, "A", "agk", 5.0, 34.0, is_gk=True),
    ]
    return pd.DataFrame(rows)


def test_home_gk_on_attacking_half_period_is_flipped():
    # P1: home GK at x=100 (>52.5) => mis-oriented => flip. P2: home GK at x=5 => keep.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    p1_hgk = out[(out.period_id == 1) & (out.player_id == "hgk")].iloc[0]
    p2_hgk = out[(out.period_id == 2) & (out.player_id == "hgk")].iloc[0]
    # Both periods: home GK now at LOW x (home defends x=0 in the canonical LTR frame).
    assert p1_hgk.x == pytest.approx(5.0)  # 105 - 100
    assert p1_hgk.y == pytest.approx(63.0)  # 68 - 5
    assert p2_hgk.x == pytest.approx(5.0)  # unchanged


def test_labels_populated_ltr_for_home_rtl_for_away_after_orient():
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    home = out[(~out.is_ball) & (out.team_id == "H")]
    away = out[(~out.is_ball) & (out.team_id == "A")]
    assert (home.team_attacking_direction == "ltr").all()
    assert (away.team_attacking_direction == "rtl").all()


def test_idempotent_label_rederivation():
    # Re-orienting after clearing labels reproduces the same labels AND coords.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    once = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    twice = orient_frames_to_ltr_by_geometry(once.assign(team_attacking_direction=None), home_team_id="H")
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True), check_dtype=False)


def test_idempotent_pure_coordinates():
    # The property that matters: orienting an ALREADY-oriented frame (no label reset) is a
    # pure no-op --- home GK is already low-x so no period flips, labels stay put.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    once = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    twice = orient_frames_to_ltr_by_geometry(once, home_team_id="H")  # NO reset
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True), check_dtype=False)


def test_extra_time_periods_flip_independently():
    rows = [
        _frame(3, "H", "hgk", 100.0, 5.0, is_gk=True),
        _frame(3, "A", "agk", 100.0, 34.0, is_gk=True),
        _frame(4, "H", "hgk", 5.0, 5.0, is_gk=True),
        _frame(4, "A", "agk", 5.0, 34.0, is_gk=True),
    ]
    out = orient_frames_to_ltr_by_geometry(pd.DataFrame(rows), home_team_id="H")
    assert out[(out.period_id == 3) & (out.player_id == "hgk")].iloc[0].x == pytest.approx(5.0)
    assert out[(out.period_id == 4) & (out.player_id == "hgk")].iloc[0].x == pytest.approx(5.0)


def test_ball_rows_flipped_with_their_period():
    rows = [
        _frame(1, "H", "hgk", 100.0, 5.0, is_gk=True),
        _frame(1, "A", "agk", 100.0, 34.0, is_gk=True),
        _frame(1, None, None, 80.0, 10.0, is_ball=True),
    ]
    out = orient_frames_to_ltr_by_geometry(pd.DataFrame(rows), home_team_id="H")
    ball = out[out.is_ball].iloc[0]
    assert ball.x == pytest.approx(25.0)  # 105 - 80
    assert ball.y == pytest.approx(58.0)  # 68 - 10


def test_velocity_components_negated_on_flip():
    # NOTE: the default builders emit no vx/vy (only `speed`); vx/vy exist only under
    # `preprocess`. This test injects them to exercise the negate-on-flip path directly,
    # since the default-builder output wouldn't reach it.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    frames["vx"] = 2.0
    frames["vy"] = -3.0
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    p1 = out[(out.period_id == 1) & (out.player_id == "hgk")].iloc[0]
    p2 = out[(out.period_id == 2) & (out.player_id == "hgk")].iloc[0]
    assert p1.vx == pytest.approx(-2.0) and p1.vy == pytest.approx(3.0)  # flipped
    assert p2.vx == pytest.approx(2.0) and p2.vy == pytest.approx(-3.0)  # unchanged


def test_zero_home_match_raises():
    frames = _two_period_match(100.0, 5.0)
    with pytest.raises(ValueError, match="matched ZERO"):
        orient_frames_to_ltr_by_geometry(frames, home_team_id="NOPE")


def test_missing_required_column_raises():
    frames = _two_period_match(100.0, 5.0).drop(columns=["is_goalkeeper"])
    with pytest.raises(ValueError, match="required column"):
        orient_frames_to_ltr_by_geometry(frames, home_team_id="H")


def test_already_correct_frames_are_noop():
    # home GK already low-x in both periods => no flip, positions unchanged.
    frames = _two_period_match(home_gk_x_p1=5.0, home_gk_x_p2=5.0)
    out = orient_frames_to_ltr_by_geometry(frames, home_team_id="H")
    pd.testing.assert_frame_equal(
        out.drop(columns=["team_attacking_direction"]).reset_index(drop=True),
        frames.drop(columns=["team_attacking_direction"]).reset_index(drop=True),
        check_dtype=False,
    )
