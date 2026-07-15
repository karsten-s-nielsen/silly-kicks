"""tracking.skillcorner.convert_to_frames --- bronze->canonical frame builder."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import skillcorner as sk
from silly_kicks.tracking.schema import KLOPPY_TRACKING_FRAMES_COLUMNS


def _bronze_row(frame, period, ts, player, team, x, y, is_gk, ball_x, ball_y, ball_z, is_vis):
    return {
        "match_id": "m1",
        "period": period,
        "frame": frame,
        "timestamp": ts,
        "player_id": player,
        "team_id": team,
        "is_goalkeeper": is_gk,
        "x": x,
        "y": y,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_z": ball_z,
        "is_visible": is_vis,
        "frame_rate": 10,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    }


def _bronze(n_frames=4):
    """Two teams (home 31, away 42), one GK each; home GK on the LEFT in P1 (low centre-x)."""
    rows = []
    for f in range(n_frames):
        period = 1 if f < 2 else 2
        ts = f * 0.1 if period == 1 else 2700.0 + (f - 2) * 0.1
        # centre-origin meters: home GK near own goal (left, x~-50) in P1
        rows += [
            _bronze_row(f, period, ts, 311, 31, -50.0, 0.0, True, 5.0, 1.0, 2.0, True),
            _bronze_row(f, period, ts, 312, 31, -10.0, 5.0, False, 5.0, 1.0, 2.0, True),
            _bronze_row(f, period, ts, 421, 42, 50.0, 0.0, True, 5.0, 1.0, 2.0, False),
            _bronze_row(f, period, ts, 422, 42, 10.0, -5.0, False, 5.0, 1.0, 2.0, True),
        ]
    return pd.DataFrame(rows)


def test_rescale_centre_origin_to_spadl():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    # away outfield at centre-origin (10, -5) -> SPADL (62.5, 29.0)
    row = frames[(frames.player_id == "422") & (frames.frame_id == 0)].iloc[0]
    assert row.x == pytest.approx(62.5) and row.y == pytest.approx(29.0)


def test_ball_z_recovered_not_nan():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    ball = frames[frames.is_ball].iloc[0]
    assert ball.z == pytest.approx(2.0)  # bronze ball_z preserved, NOT NaN


def test_player_z_is_nan_and_visibility_mapped():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    p = frames[(~frames.is_ball) & (frames.player_id == "421")].iloc[0]
    assert np.isnan(p.z)
    assert bool(p.visibility) is False  # away GK had is_visible=False


def test_period_relative_clock():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    p2 = frames[frames.period_id == 2]["time_seconds"]
    assert p2.min() == pytest.approx(0.0)  # 2700 - 2700


def test_ids_are_object_strings():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert frames["team_id"].dropna().map(lambda v: isinstance(v, str)).all()
    assert frames["player_id"].dropna().map(lambda v: isinstance(v, str)).all()


def test_output_schema_matches_kloppy_variant():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert list(frames.columns) == list(KLOPPY_TRACKING_FRAMES_COLUMNS)


def test_ltr_orientation_applied_by_default():
    # Home GK starts left (low x) in P1 -> P1 keeps; P2 home defends right -> flips to low x.
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31")  # output_convention default "ltr"
    hgk = frames[(~frames.is_ball) & (frames.player_id == "311")]
    assert (hgk.x < 52.5).all()  # home GK low-x every period post-LTR
    assert (frames[(~frames.is_ball) & (frames.team_id == "31")].team_attacking_direction == "ltr").all()


def test_clock_constant_is_single_sourced():
    # Regression guard for duplicated-truth #3: the builder must import the SPADL constant.
    from silly_kicks.spadl import skillcorner as sk_spadl

    assert sk._PERIOD_START_SECONDS is sk_spadl._PERIOD_START_SECONDS


def test_native_skillcorner_gk_survives_derivation():
    # Orientation anchors on home-GK median x; SkillCorner has an AUTHORITATIVE native GK
    # (skillcorner_matches.position_acronym). derive_goalkeepers is Tier-1 roster-validated
    # (PR-S86, 20/20) and must keep the native pick --- a wrong overwrite silently mirrors
    # orientation for the one provider with ground-truth GK identity.
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    home_gk = set(frames[(~frames.is_ball) & (frames.team_id == "31") & frames.is_goalkeeper]["player_id"])
    assert home_gk == {"311"}  # the native SkillCorner GK (bronze is_goalkeeper=True), survived


def test_missing_input_column_raises():
    bad = _bronze().drop(columns=["ball_z"])
    with pytest.raises(ValueError, match="ball_z"):
        sk.convert_to_frames(bad, home_team_id="31")
