"""Tests for silly_kicks.tracking._gk_resolve.acting_gk_from_frames (mirror of defending, CR 2026-07-01).

Resolves the ACTING team's GK (== team predicate) with an identity fallback so a goal-kick whose keeper
is undetected at the linked event frame still resolves (the keeper's identity is roster-stable post-4.38.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._gk_resolve import acting_gk_from_frames, defending_gk_from_frames


def _row(pid, team, gk, x, *, frame_id, t, period=1, is_ball=False):
    return dict(
        game_id=1,
        period_id=period,
        frame_id=frame_id,
        time_seconds=t,
        frame_rate=25.0,
        player_id=pid,
        team_id=team,
        is_ball=is_ball,
        is_goalkeeper=gk,
        x=x,
        y=34.0,
        source_provider="sportec",
        team_attacking_direction="ltr",
    )


def _one_frame(*, include_home_gk=True):
    """Single frame at t=5: home GK 100, away GK 200, one outfield each, ball."""
    rows = [
        _row(np.nan, np.nan, False, 50.0, frame_id=10, t=5.0, is_ball=True),
        _row(200, 2, True, 100.0, frame_id=10, t=5.0),
        _row(101, 1, False, 40.0, frame_id=10, t=5.0),
        _row(201, 2, False, 60.0, frame_id=10, t=5.0),
    ]
    if include_home_gk:
        rows.append(_row(100, 1, True, 5.0, frame_id=10, t=5.0))
    return pd.DataFrame(rows)


def _actions(team_id=1, t=5.0, period=1, aid=1):
    return pd.DataFrame(
        {
            "action_id": [aid],
            "period_id": [period],
            "time_seconds": [t],
            "team_id": [team_id],
            "player_id": [101],
            "start_x": [40.0],
            "start_y": [30.0],
            "type_id": [0],
            "game_id": [1],
        }
    )


# --------------------------------------------------------------------------------------
# Mirror of defending -- the acting-team inversion
# --------------------------------------------------------------------------------------
def test_resolves_acting_gk_home():
    result = acting_gk_from_frames(_actions(team_id=1), _one_frame())
    assert result.iloc[0] == 100  # home acts -> home GK (not away 200)


def test_resolves_acting_gk_away():
    result = acting_gk_from_frames(_actions(team_id=2), _one_frame())
    assert result.iloc[0] == 200  # away acts -> away GK


def test_all_actions_each_own_team_gk():
    frames = _one_frame()
    actions = pd.DataFrame(
        {
            "action_id": [1, 2, 3],
            "period_id": [1, 1, 1],
            "time_seconds": [5.0, 5.0, 5.0],
            "team_id": [1, 1, 2],
            "player_id": [101, 101, 201],
            "start_x": [40.0, 40.0, 60.0],
            "start_y": [30.0, 30.0, 34.0],
            "type_id": [0, 5, 8],
            "game_id": [1, 1, 1],
        }
    )
    result = acting_gk_from_frames(actions, frames)
    assert result.iloc[0] == 100 and result.iloc[1] == 100 and result.iloc[2] == 200


def test_nan_when_no_acting_gk_identity_anywhere():
    # home has no GK identity in ANY frame -> NaN for home actions
    result = acting_gk_from_frames(_actions(team_id=1), _one_frame(include_home_gk=False))
    assert pd.isna(result.iloc[0])


def test_nan_when_team_id_nan():
    actions = _actions(team_id=1)
    actions["team_id"] = pd.array([pd.NA], dtype="Int64")
    result = acting_gk_from_frames(actions, _one_frame())
    assert pd.isna(result.iloc[0])


# --------------------------------------------------------------------------------------
# Identity fallback -- keeper undetected at the linked (goal-kick) frame
# --------------------------------------------------------------------------------------
def test_identity_fallback_when_gk_undetected_at_linked_frame():
    # home GK 100 detected at t=5 but ABSENT at t=20 (broadcast miss); a home action at t=20 links to
    # the t=20 frame (no home GK row) -> per-frame link NaN -> identity fallback -> 100 (not NaN).
    rows = [
        _row(np.nan, np.nan, False, 50.0, frame_id=10, t=5.0, is_ball=True),
        _row(100, 1, True, 5.0, frame_id=10, t=5.0),
        _row(101, 1, False, 40.0, frame_id=10, t=5.0),
        _row(200, 2, True, 100.0, frame_id=10, t=5.0),
        # t=20 frame: NO home GK row (undetected)
        _row(np.nan, np.nan, False, 60.0, frame_id=20, t=20.0, is_ball=True),
        _row(101, 1, False, 42.0, frame_id=20, t=20.0),
        _row(200, 2, True, 100.0, frame_id=20, t=20.0),
    ]
    frames = pd.DataFrame(rows)
    result = acting_gk_from_frames(_actions(team_id=1, t=20.0), frames)
    assert result.iloc[0] == 100  # resolved by identity fallback, not NaN


# --------------------------------------------------------------------------------------
# GK-sub -- two keeper identities -> time-appropriate one
# --------------------------------------------------------------------------------------
def _sub_frames():
    # home starter GK 100 at t=5; home sub GK 150 at t=60. Neither detected at the action frames below,
    # forcing the identity fallback to pick nearest-in-time.
    return pd.DataFrame(
        [
            _row(np.nan, np.nan, False, 50.0, frame_id=10, t=5.0, is_ball=True),
            _row(100, 1, True, 5.0, frame_id=10, t=5.0),
            _row(200, 2, True, 100.0, frame_id=10, t=5.0),
            _row(np.nan, np.nan, False, 50.0, frame_id=90, t=60.0, is_ball=True),
            _row(150, 1, True, 5.0, frame_id=90, t=60.0),
            _row(200, 2, True, 100.0, frame_id=90, t=60.0),
            # bare action frames (no home GK detected) at t=10 and t=55
            _row(101, 1, False, 40.0, frame_id=15, t=10.0),
            _row(101, 1, False, 40.0, frame_id=85, t=55.0),
        ]
    )


def test_gk_sub_returns_time_appropriate_keeper():
    frames = _sub_frames()
    early = acting_gk_from_frames(_actions(team_id=1, t=10.0, aid=1), frames)
    late = acting_gk_from_frames(_actions(team_id=1, t=55.0, aid=1), frames)
    assert early.iloc[0] == 100  # nearest-in-time to t=10 is the starter (t=5)
    assert late.iloc[0] == 150  # nearest-in-time to t=55 is the sub (t=60)


# --------------------------------------------------------------------------------------
# dtype
# --------------------------------------------------------------------------------------
def test_dtype_object_provider():
    frames = _one_frame()
    frames["player_id"] = frames["player_id"].astype(object)
    result = acting_gk_from_frames(_actions(team_id=1), frames)
    assert result.iloc[0] == 100


# --------------------------------------------------------------------------------------
# Regression: factoring the shared body leaves defending byte-identical
# --------------------------------------------------------------------------------------
def test_defending_unchanged_after_factor():
    frames = _one_frame()
    assert defending_gk_from_frames(_actions(team_id=1), frames).iloc[0] == 200  # home acts -> away GK
    assert defending_gk_from_frames(_actions(team_id=2), frames).iloc[0] == 100  # away acts -> home GK
