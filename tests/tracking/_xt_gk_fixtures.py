"""Shared fixtures for the SkillCorner keeper-origin distrust tests (CR 2026-06-30).

Production-realistic SkillCorner geometry: centre-origin already shifted to SPADL coords by the
converter (so these are post-`convert_to_frames` frames), ``source_provider="skillcorner"``, a
visible GK detection near/away from the goal, and an action whose NATIVE origin is the scattered
broadcast ball-event location (NOT the keeper).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.xthreat import ExpectedThreat

_GOALKICK = 22
_PASS = 0


def make_fitted_xt() -> ExpectedThreat:
    """A fitted xT whose value rises toward goal with enough curvature that distinct origin x land
    in distinct grid values (so origin-sensitivity is observable). Cube ramp like the GK-realistic
    grid in test_xt_gk.py."""
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16) ** 3, (12, 1))
    return xt


def make_skillcorner_case(
    *,
    native_origin_x: float,
    tracked_gk_x: float,
    defender_near: tuple[float, float] | None = None,
    type_id: int = _GOALKICK,
):
    """Return ``(actions, frames)`` for one SkillCorner GK distribution.

    ``native_origin_x`` is the (distrusted) broadcast ball-event origin on the SPADL action;
    ``tracked_gk_x`` is where the visible keeper actually is in the frame. ``type_id`` selects a
    goal-kick (default) or an open-play GK pass. ``defender_near`` optionally places an opponent at a
    fixed point (for the pressure coherence test)."""
    actions = pd.DataFrame(
        {
            "game_id": [9],
            "action_id": [0],
            "team_id": [1],
            "player_id": [10],
            "period_id": [1],
            "time_seconds": [10.0],
            "type_id": [type_id],
            "start_x": [native_origin_x],
            "start_y": [34.0],
            "end_x": [40.0],
            "end_y": [34.0],
        }
    )
    players = [
        (10, 1, True, tracked_gk_x, 34.0),  # acting-team GK, visible detection
        (11, 1, False, 30.0, 30.0),
        (12, 2, False, 45.0, 40.0),
        (20, 2, True, 100.0, 34.0),  # opponent GK (far end)
    ]
    if defender_near is not None:
        players.append((21, 2, False, defender_near[0], defender_near[1]))
    rows = []
    for pid, team, gk, x, y in players:
        rows.append(
            dict(
                game_id=9,
                period_id=1,
                frame_id=0,
                time_seconds=10.0,
                frame_rate=10.0,
                team_id=team,
                player_id=pid,
                is_goalkeeper=gk,
                is_ball=False,
                x=x,
                y=y,
                vx=0.0,
                vy=0.0,
                team_in_possession=1,
                source_provider="skillcorner",
                team_attacking_direction="ltr",
                visibility=True,
            )
        )
    rows.append(
        dict(
            game_id=9,
            period_id=1,
            frame_id=0,
            time_seconds=10.0,
            frame_rate=10.0,
            team_id=None,
            player_id=-1,
            is_goalkeeper=False,
            is_ball=True,
            x=native_origin_x,
            y=34.0,
            vx=0.0,
            vy=0.0,
            team_in_possession=1,
            source_provider="skillcorner",
            team_attacking_direction="ltr",
            visibility=None,
        )
    )
    return actions, pd.DataFrame(rows)
