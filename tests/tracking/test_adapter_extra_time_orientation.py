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
def test_wrong_extra_time_flag_reverses_p3_p4(adapter, home_id, away_id, home_gk, away_gk):
    """Control: a WRONG ET flag reverses P3/P4 (the live GS pattern) -> guard discriminates."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw,
        home_team_id=home_id,
        home_team_start_left=True,
        home_team_start_left_extratime=True,  # WRONG for this physical setup
        output_convention="ltr",
    )
    for p in (1, 2):  # regulation still correct
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK x={hg['x']}"
    for p in (3, 4):  # ET reversed by the wrong flag (the live bug signature)
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        assert abs(hg["x"] - 100.0) < 0.01, f"{adapter.__name__} P{p} home GK should reverse, x={hg['x']}"
