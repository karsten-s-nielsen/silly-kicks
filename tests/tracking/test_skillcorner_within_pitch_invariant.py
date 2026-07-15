"""SkillCorner S1 within-pitch invariant + S2 visibility-survives guard (CR 2026-06-30)."""

from __future__ import annotations

import warnings

import pandas as pd

from silly_kicks.tracking import skillcorner


def _bronze(extra_player_native_x=None, ball_native_x=0.0):
    """Minimal SkillCorner bronze (centre-origin) with one home + one away player + ball, period 1."""
    base = []
    for pid, tid, gk, x, y in [("p1", "31", True, -50.0, 0.0), ("p2", "40", False, 10.0, 5.0)]:
        base.append(
            dict(
                match_id="m",
                period=1,
                frame=1,
                timestamp=0.0,
                player_id=pid,
                team_id=tid,
                is_goalkeeper=gk,
                x=x,
                y=y,
                ball_x=ball_native_x,
                ball_y=0.0,
                ball_z=0.0,
                is_visible=True,
                frame_rate=10.0,
                pitch_length=105.0,
                pitch_width=68.0,
            )
        )
    if extra_player_native_x is not None:
        base.append(
            dict(
                match_id="m",
                period=1,
                frame=1,
                timestamp=0.0,
                player_id="p3",
                team_id="40",
                is_goalkeeper=False,
                x=extra_player_native_x,
                y=0.0,
                ball_x=ball_native_x,
                ball_y=0.0,
                ball_z=0.0,
                is_visible=True,
                frame_rate=10.0,
                pitch_length=105.0,
                pitch_width=68.0,
            )
        )
    return pd.DataFrame(base)


# --------------------------------------------------------------------------------------
# Task 8: within-pitch invariant (warn-and-count, never crash) -- LAYERED (ADR-024 amendment):
#   players = a thin S1 band just inside the pre-existing derive_goalkeepers catastrophic bound;
#   the ball = S1's SOLE off-pitch signal (derive_goalkeepers is player-only).
# --------------------------------------------------------------------------------------
def test_behind_goal_keeper_within_tolerance_does_not_warn():
    # native x=-60 -> SPADL x=-7.5 (behind goal line, legit) -> inside the S1 player band -> no warn.
    bronze = _bronze(extra_player_native_x=-60.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert not any("off-pitch" in str(x.message) for x in w)
    assert report.n_gross_off_pitch == 0


def test_mildly_off_pitch_player_warns_and_counts_but_does_not_crash():
    # native x=-65.5 -> SPADL x=-13: inside the S1 player band (< -12) but within derive_goalkeepers'
    # catastrophic bound (>= -15) -> warn + count, NO crash (the band sits just below the crash).
    bronze = _bronze(extra_player_native_x=-65.5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert any("off-pitch" in str(x.message) for x in w)
    assert report.n_gross_off_pitch >= 1
    assert len(frames) > 0  # did not crash; row retained (never clamped/dropped)


def test_off_pitch_ball_warns_and_counts_and_never_crashes():
    # The ball has NO existing guard (derive_goalkeepers is player-only) -> S1 is its sole signal.
    # ball native x=-90 -> SPADL -37.5 -> beyond the ball tolerance (15 m) -> warn + count, no crash.
    bronze = _bronze(ball_native_x=-90.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert any("off-pitch" in str(x.message) for x in w)
    assert report.n_gross_off_pitch >= 1
    assert len(frames) > 0


def test_clean_fixture_is_not_geometry_excluded():
    # The per-row S1 warn (n_gross_off_pitch) and the SYSTEMATIC rate-gate (geometry_excluded) must
    # stay consistent: a clean fixture trips NEITHER. Pins the new report fields onto the clean path
    # so the two guards cannot drift apart.
    bronze = _bronze()
    _frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert report.geometry_excluded is False
    assert report.player_off_pitch_rate == 0.0
    assert report.ball_off_pitch_rate == 0.0


# --------------------------------------------------------------------------------------
# Task 9: S2 visibility-survives guard
# --------------------------------------------------------------------------------------
def test_visibility_survives_convert_and_preprocess():
    from silly_kicks.tracking.preprocess import PreprocessConfig

    bronze = _bronze()
    frames, _ = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert "visibility" in frames.columns
    gk = frames[frames["player_id"] == "p1"]
    assert bool(gk["visibility"].iloc[0]) is True
    # preprocess (smoothing) must not drop/blank visibility
    frames_pp, _ = skillcorner.convert_to_frames(
        bronze, home_team_id="31", preprocess=PreprocessConfig(smoothing_method="ema")
    )
    assert "visibility" in frames_pp.columns
    assert frames_pp["visibility"].notna().any()
