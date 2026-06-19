"""Positive extra-time (P3/P4) orientation regression guard for the native adapters.

gradientsports/sportec.convert_to_frames take RAW physical centered coords (x_centered)
+ the start-left flags and flip per period internally. With home_team_start_left=True,
home_team_start_left_extratime=False the per-period home-attacks-right flags are
{1:T, 2:F, 3:F, 4:T} (home_attacks_right_per_period); in physical coords the home GK
(own goal behind the attack) is at x_centered=-47.5 in P1/P4 and +47.5 in P2/P3.
Correct orientation -> home GK lands at x=5 in ALL four periods, away GK at x=100.

Context: the live GS-ET flip (2026-06-13) was a consumer bug (wrong
home_team_start_left_extratime placeholder), NOT a silly_kicks bug. This locks the
positive ET path; the wrong-flag control reproduces the live reversal so the guard
discriminates. sportec team_id/player_id are object strings; gradientsports are
nullable Int64 -> id values are parametrized per provider.
"""

import pandas as pd
import pytest

from silly_kicks.tracking import gradientsports, sportec


def _raw_row(period, frame, pid, tid, isball, isgk, xc, yc):
    return {
        "game_id": 1,
        "period_id": period,
        "frame_id": frame,
        "time_seconds": frame / 25.0,
        "frame_rate": 25.0,
        "player_id": pid,
        "team_id": tid,
        "is_ball": isball,
        "is_goalkeeper": isgk,
        "x_centered": xc,
        "y_centered": yc,
        "z": float("nan"),
        "speed_native": float("nan"),
        "ball_state": "alive",
    }


def _raw_4period(home_id, away_id, home_gk, away_gk):
    """Physical raw frames: home attacks right in P1/P4, left in P2/P3."""
    home_gk_xc = {1: -47.5, 2: 47.5, 3: 47.5, 4: -47.5}
    away_gk_xc = {1: 47.5, 2: -47.5, 3: -47.5, 4: 47.5}
    rows = []
    for p in (1, 2, 3, 4):
        f = p * 100
        rows += [
            _raw_row(p, f, home_gk, home_id, False, True, home_gk_xc[p], 0.0),
            _raw_row(p, f, away_gk, away_id, False, True, away_gk_xc[p], 0.0),
            _raw_row(p, f, None, None, True, False, 0.0, 0.0),
        ]
    return pd.DataFrame(rows)


def _raw_4period_outfield(home_id, away_id, home_gk, away_gk):
    """Like _raw_4period but with three home + three away OUTFIELD players at asymmetric x."""
    home_gk_xc = {1: -47.5, 2: 47.5, 3: 47.5, 4: -47.5}
    away_gk_xc = {1: 47.5, 2: -47.5, 3: -47.5, 4: 47.5}
    # home outfielders sit ahead of their GK (own half -> middle): asymmetric offsets.
    home_of_xc = {1: [-30.0, -20.0, -5.0], 2: [30.0, 20.0, 5.0], 3: [30.0, 20.0, 5.0], 4: [-30.0, -20.0, -5.0]}
    away_of_xc = {1: [30.0, 20.0, 5.0], 2: [-30.0, -20.0, -5.0], 3: [-30.0, -20.0, -5.0], 4: [30.0, 20.0, 5.0]}
    rows = []
    for p in (1, 2, 3, 4):
        f = p * 100
        rows.append(_raw_row(p, f, home_gk, home_id, False, True, home_gk_xc[p], 0.0))
        rows.append(_raw_row(p, f, away_gk, away_id, False, True, away_gk_xc[p], 0.0))
        for i, xc in enumerate(home_of_xc[p]):
            rows.append(_raw_row(p, f, 10 + i, home_id, False, False, xc, 5.0 * i))
        for i, xc in enumerate(away_of_xc[p]):
            rows.append(_raw_row(p, f, 20 + i, away_id, False, False, xc, 5.0 * i))
        rows.append(_raw_row(p, f, None, None, True, False, 0.0, 0.0))
    return pd.DataFrame(rows)


_ADAPTERS = [
    pytest.param(sportec, "H", "A", "HOME-GK", "AWAY-GK", id="sportec"),
    pytest.param(gradientsports, 57, 99, 1, 2, id="gradientsports"),
]


@pytest.mark.parametrize("adapter, home_id, away_id, home_gk, away_gk", _ADAPTERS)
def test_positive_extra_time_orientation(adapter, home_id, away_id, home_gk, away_gk):
    """Correct ET flag -> home GK at x=5, away GK at x=100, in ALL FOUR periods."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw,
        home_team_id=home_id,
        home_team_start_left=True,
        home_team_start_left_extratime=False,
        output_convention="ltr",
    )
    for p in (1, 2, 3, 4):
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        ag = out[(out["period_id"] == p) & (out["player_id"] == away_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK x={hg['x']}"
        assert abs(ag["x"] - 100.0) < 0.01, f"{adapter.__name__} P{p} away GK x={ag['x']}"


@pytest.mark.parametrize("adapter, home_id, away_id, home_gk, away_gk", _ADAPTERS)
def test_wrong_extra_time_flag_self_corrects_via_geometry(adapter, home_id, away_id, home_gk, away_gk):
    """TF-23b: a WRONG ET flag is self-corrected by the geometric backstop ->
    home GK at x=5 (away GK x=100) in ALL FOUR periods (ltr convention)."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw,
        home_team_id=home_id,
        home_team_start_left=True,
        home_team_start_left_extratime=True,  # WRONG for this physical setup (P3/P4 reversed)
        output_convention="ltr",
    )
    for p in (1, 2, 3, 4):
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        ag = out[(out["period_id"] == p) & (out["player_id"] == away_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK x={hg['x']}"
        assert abs(ag["x"] - 100.0) < 0.01, f"{adapter.__name__} P{p} away GK x={ag['x']}"


@pytest.mark.parametrize("adapter, home_id, away_id, home_gk, away_gk", _ADAPTERS)
def test_wrong_extra_time_flag_self_corrects_absolute_frame(adapter, home_id, away_id, home_gk, away_gk):
    """The backstop runs before the convention branch, so absolute_frame is corrected too:
    home GK at low x=5 in all periods despite the wrong ET flag."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw,
        home_team_id=home_id,
        home_team_start_left=True,
        home_team_start_left_extratime=True,  # WRONG
        output_convention="absolute_frame",
    )
    for p in (1, 2, 3, 4):
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK abs x={hg['x']}"


def _outfield_backline_centroid_x(out, team_id, period):
    """Mean x of a team's three deepest (lowest-x) OUTFIELD players in a period.

    A multi-player, non-GK-dominated orientation-sensitive quantity. Used to confirm the
    feature layer (not just GK-x) is correctly oriented after self-correction.
    """
    sub = out[(out["period_id"] == period) & (out["team_id"] == team_id) & (~out["is_ball"])]
    sub = sub[sub["is_goalkeeper"] == False]  # noqa: E712 -- explicit non-GK
    return sub.nsmallest(3, "x")["x"].mean()


def test_wrong_et_flag_self_corrects_outfield_feature():
    """The geometry-derived back-line centroid (non-GK, multi-player) matches the correct-flag
    conversion on the ET periods after the backstop self-corrects a wrong ET flag (gradientsports
    fixture carries multiple outfielders per team)."""
    raw = _raw_4period_outfield(home_id=57, away_id=99, home_gk=1, away_gk=2)
    correct, _ = gradientsports.convert_to_frames(
        raw,
        home_team_id=57,
        home_team_start_left=True,
        home_team_start_left_extratime=False,
        output_convention="absolute_frame",
    )
    wrong, _ = gradientsports.convert_to_frames(
        raw,
        home_team_id=57,
        home_team_start_left=True,
        home_team_start_left_extratime=True,
        output_convention="absolute_frame",  # WRONG
    )
    for p in (3, 4):
        c = _outfield_backline_centroid_x(correct, 57, p)
        w = _outfield_backline_centroid_x(wrong, 57, p)
        assert abs(c - w) < 0.5, f"P{p} outfield centroid mismatch: correct={c} wrong={w}"
