"""Shared fixture for the space-creation opponent-mirror gate (ADR-041, plan Task 0/6).

Imported by BOTH ``scripts/gen_space_creation_mirror_golden.py`` (which captures the
pre-change golden) and ``tests/tracking/test_space_creation_mirror.py`` (which asserts
against it), so the golden and the assertion cannot drift.

DELIBERATELY STANDALONE — it does NOT reuse the liveness fixture
(``test_aggregator_column_liveness``), which a later task in this same PR extends with
sprinting runners. Sharing that fixture would silently invalidate this golden.

Geometry: one game, one period, team 5 (home, attacking, "ltr") vs team 6, two completed
passes. Player layout is fixed and asymmetric in y so the opponent-perspective mirror is
actually exercised (a y-symmetric layout would make the axis=1 -> axis=(0,1) upgrade
vacuously identical for reasons unrelated to the synthetic grids).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ``space_created_m2`` is the ACTOR's own leave-one-out OBSO differential. It is
# identically 0 unless the actor is a MARGINAL controller of a region with non-trivial
# OBSO weight. Two distinct ways to score 0 were measured while building this fixture,
# and BOTH would have made the golden vacuous (0 == 0 passes under any change):
#   * actor behind the ball  -> near-zero EPV region, nothing to gain;
#   * actor in wide-open space with the defence parked deep -> the attacking team already
#     holds ~100% control there, so removing one player leaves att/(att+def) ~ 1 unchanged.
# The layout below therefore keeps each actor AHEAD of the ball with the nearest opponent
# ~11-13 m beyond -- close enough to contest, far enough that the actor is the marginal
# controller. The generator refuses to write an all-zero golden as a standing guard.
#
# (t_action, ball_x, ball_y, actor_pid)
_WINDOWS = (
    (10.0, 55.0, 30.0, 13),
    (20.0, 70.0, 42.0, 14),
)

# player_id -> (x offset from the ball, absolute y). y is asymmetric so the
# opponent-perspective mirror is genuinely exercised.
_TEAM_A = {11: (-14.0, 18.0), 12: (-2.0, 30.0), 13: (7.0, 44.0), 14: (15.0, 55.0)}
_TEAM_B = {21: (6.0, 22.0), 22: (13.0, 33.0), 23: (20.0, 47.0), 24: (26.0, 58.0)}
_GKS = {1: (5, 6.0, 33.0), 2: (6, 99.0, 35.0)}

_OFFSETS = (-0.4, -0.2, 0.0)
_FRAME_RATE = 25.0


def _frow(pid, team, gk, x, y, t, *, is_ball=False, vx=0.0, vy=0.0) -> dict:
    return {
        "game_id": 1,
        "period_id": 1,
        "frame_id": round(t * _FRAME_RATE),
        "time_seconds": float(t),
        "frame_rate": _FRAME_RATE,
        "player_id": pid,
        "team_id": team,
        "is_ball": is_ball,
        "is_goalkeeper": gk,
        "x": float(min(max(x, 0.5), 104.5)),
        "y": float(min(max(y, 0.5), 67.5)),
        "z": 0.0,
        "speed": float(np.hypot(vx, vy)),
        "vx": float(vx),
        "vy": float(vy),
        "speed_source": "native",
        "ball_state": "alive",
        "team_attacking_direction": "ltr" if team == 5 else "rtl",
        "confidence": None,
        "visibility": None,
        "source_provider": "gradientsports",
        "is_goalkeeper_source": "native",
    }


def build_mirror_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(actions, frames)`` for the opponent-mirror golden.

    Deterministic: no RNG, no clock, no I/O.

    Examples
    --------
    Build the fixture::

        actions, frames = build_mirror_fixture()
    """
    actions = pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [0, 1],
            "period_id": [1, 1],
            "time_seconds": [10.0, 20.0],
            "team_id": pd.Series([5, 5], dtype="int64"),
            "player_id": pd.Series([13, 14], dtype="int64"),
            "start_x": [55.0, 70.0],
            "start_y": [30.0, 42.0],
            "end_x": [72.0, 88.0],
            "end_y": [40.0, 36.0],
            "type_id": [0, 0],  # pass, pass
            "type_name": ["pass", "pass"],
            "result_id": [1, 1],
            "result_name": ["success", "success"],
            "bodypart_id": [0, 0],
            "bodypart_name": ["foot", "foot"],
        }
    )

    rows: list[dict] = []
    for t_a, ball_x, ball_y, actor_pid in _WINDOWS:
        for off in _OFFSETS:
            t = round(t_a + off, 3)
            rows.append(_frow(None, None, False, ball_x, ball_y, t, is_ball=True))
            for pid, (dx, y) in _TEAM_A.items():
                vx = 2.4 if pid == actor_pid else 1.6
                rows.append(_frow(pid, 5, False, ball_x + dx, y, t, vx=vx, vy=0.2))
            for pid, (dx, y) in _TEAM_B.items():
                rows.append(_frow(pid, 6, False, ball_x + dx, y, t, vx=-1.3, vy=-0.15))
            for pid, (team, gx, gy) in _GKS.items():
                rows.append(_frow(pid, team, True, gx, gy, t))
    frames = pd.DataFrame(rows)
    return actions, frames
