"""Task 3 (spec Part 2): space_creation opponent-perspective softening on velocity-less SB360.

With ``include_opponent_perspective=True`` a legitimate one-team SB360 FOV crop used to abort the whole
batch via ``_resolve_opponent_team_id``'s two-team raise. Marker-gated softening degrades the opponent
side to NaN (with a ``space_opponent_source`` provenance token) instead of raising, while full-tracking
frames keep the loud corrupt-frame raise and 0/3+-team frames raise in BOTH modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._space_creation import _resolve_opponent_team_id, compute_space_created
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE


def _frame(*, teams: list[int], marker: bool) -> pd.DataFrame:
    """One frame: two outfield players per team id in ``teams`` + a ball. ``marker`` stamps
    speed_source=unavailable on every row (the SB360 declared-velocity-less shape)."""
    rows = []
    x = 30.0
    for t in teams:
        for j in range(2):  # >=2 attacking players -> a non-trivial leave-one-out
            rows.append(
                dict(
                    player_id=100 * t + j, team_id=t, is_ball=False, is_goalkeeper=(j == 0), x=x, y=30.0 + 8 * j, z=0.0
                )
            )
            x += 6.0
    rows.append(dict(player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=52.5, y=34.0, z=0.0))
    df = pd.DataFrame(rows)
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 10
    df["time_seconds"] = 5.0
    df["ball_state"] = "alive"
    df["team_attacking_direction"] = "ltr"
    df["speed"] = np.nan
    df["speed_source"] = SPEED_SOURCE_UNAVAILABLE if marker else "derived"
    df["source_provider"] = "synthetic"  # add_space_creation's linking/context path reads it
    return df


def test_one_team_sb360_frame_softens_not_raises():
    frame = _frame(teams=[1], marker=True)  # one-team FOV + SB360 marker
    out = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
    assert out["space_created_m2"].notna().any()  # team side still computes (ADR-063 zero-velocity)
    assert out["space_denied_m2_opponent"].isna().all()  # opponent unresolvable -> NaN
    assert (out["space_opponent_source"] == "unresolved_one_team").all()


def test_one_team_full_tracking_frame_still_raises():
    frame = _frame(teams=[1], marker=False)  # one team, NO marker -> corrupt full-tracking
    with pytest.raises(ValueError):
        compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)


def test_two_team_frame_resolves_source_and_computes_opponent():
    frame = _frame(teams=[1, 2], marker=True)
    out = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
    assert (out["space_opponent_source"] == "resolved").all()
    assert out["space_denied_m2_opponent"].notna().any()  # non-vacuity: opponent side computes


def test_default_no_opponent_perspective_has_no_source_column():
    # space_opponent_source is emitted ONLY when include_opponent_perspective=True (conditional column).
    frame = _frame(teams=[1], marker=True)
    out = compute_space_created(frame, attacking_team_id=1)  # default include_opponent_perspective=False
    assert "space_opponent_source" not in out.columns
    assert "space_denied_m2_opponent" not in out.columns


def test_attacking_team_matches_neither_still_raises_both_modes():
    frame = _frame(teams=[1, 2], marker=True)
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(frame, attacking_team_id=999, on_unresolvable="nan")
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(frame, attacking_team_id=999, on_unresolvable="raise")


def test_zero_and_three_team_frames_raise_even_in_nan_mode():
    # 0-team (ball only) and 3-team frames are corrupt, not FOV crops -> raise in BOTH modes.
    # Only len==1 softens; this keeps the "unresolved_one_team" provenance label accurate.
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(_frame(teams=[], marker=True), attacking_team_id=1, on_unresolvable="nan")
    with pytest.raises(ValueError):
        _resolve_opponent_team_id(_frame(teams=[1, 2, 3], marker=True), attacking_team_id=1, on_unresolvable="nan")


def test_add_space_creation_softens_one_team_sb360_end_to_end():
    # #4 (whole-branch review): exercise the AGGREGATOR path, not just compute_space_created.
    # add_space_creation always uses opponent perspective, so a one-team SB360 FOV crop must soften
    # here too -- _compute_space_creation_for_action's OWN two-team guard is marker-gated, and the
    # string provenance column is threaded through its per-action assembly.
    from silly_kicks.tracking.features import add_space_creation

    frame = _frame(teams=[1], marker=True)
    actions = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [1],
            "time_seconds": [5.0],
            "team_id": [1],
            "player_id": [100],  # 100*1+0 -- a team-1 player present in the one-team frame
            "start_x": [30.0],
            "start_y": [30.0],
        }
    )
    # A tiny fitted xT avoids the escalated SyntheticEPVWarning (same precedent as the NaN-safety
    # gate's obso/pausa branch); it does not affect the opponent-resolution path under test.
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    out = add_space_creation(actions, frame, home_team_id=1, xt=xt)
    assert (out["space_opponent_source"] == "unresolved_one_team").all()
    assert out["space_denied_m2_opponent"].isna().all()
    assert out["space_created_m2"].notna().any()
