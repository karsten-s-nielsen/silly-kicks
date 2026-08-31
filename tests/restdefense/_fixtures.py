"""Shared fixtures for the restdefense suite (TF-60, ADR-080).

``make_scaling_fixture`` (Task 4) drives the ADR-073 growth guard: it scales the LOOP/GROUP
dimension (the number of distinct frames == the number of samples), so a rescan-in-loop regression
would go quadratic while the ``group_rows`` implementation stays linear. ``make_rest_defense_fixture``
(the multi-domain functional fixture) is added in Task 5.
"""

from __future__ import annotations

import pandas as pd

# --- multi-domain functional fixture (Task 5+) -------------------------------------------------
# One game, one period, two teams (1=home defends x=0, 2=away defends x=105), a keeper + 5 outfield
# each. Frames are LTR-normalized (home attacks right). Four in-possession on-ball actions exercise
# every windows branch:
#   a0  home advanced pass  (t=10)  -> scored
#   a1  home advanced pass  (t=12)  -> scored, possession P0 terminal -> is_possession_loss
#   a2  away advanced pass  (t=20)  -> scored, possession P1 terminal -> is_possession_loss (orientation)
#   a3  home NON-advanced   (t=30)  -> dropped "not_committed_forward"
# Event clocks are jittered +0.04 s off the frame grid (liveness-gate convention) but stay within the
# 0.2 s link tolerance.

_GAME = 1
_PERIOD = 1
_SRC = "fixture"


def _player(fid, t, team_id, player_id, x, y, *, gk=False):
    return {
        "game_id": _GAME,
        "period_id": _PERIOD,
        "frame_id": fid,
        "time_seconds": t,
        "frame_rate": 25.0,
        "team_id": team_id,
        "player_id": player_id,
        "is_ball": False,
        "is_goalkeeper": gk,
        "x": float(x),
        "y": float(y),
        # vx/vy present but ZERO (TF-60 PR2): pitch control runs (frames are NOT velocity-declared-
        # absent, so `#5` reachable computes a real number), and zero velocity is point-reflection-
        # invariant so a0/a2 stay exact mirrors.
        "vx": 0.0,
        "vy": 0.0,
        "ball_state": "alive",
        "source_provider": _SRC,
    }


def _ball(fid, t, x, y=34.0):
    return {
        "game_id": _GAME,
        "period_id": _PERIOD,
        "frame_id": fid,
        "time_seconds": t,
        "frame_rate": 25.0,
        "team_id": pd.NA,
        "player_id": pd.NA,
        "is_ball": True,
        "is_goalkeeper": False,
        "x": float(x),
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "ball_state": "alive",
        "source_provider": _SRC,
    }


# Per-frame outfield layouts. Team 1 defends x=0, team 2 defends x=105. Positions are chosen so the
# back-4 (n_rearguard) and the behind-ball counts are hand-computable (see test_structure.py).
# Each team: GK + 5 outfield (a clean back-4 + one forward).
def _home_attacking_frame(fid, t, ball_x):
    """Home (team 1) in possession, attacking right; ball advanced. Team 2 defends near x=105
    with one forward left upfield as a counter threat."""
    players = [
        _player(fid, t, 1, 101, 5.0, 34.0, gk=True),  # team1 GK
        _player(fid, t, 1, 102, 18.0, 20.0),  # back4
        _player(fid, t, 1, 103, 22.0, 48.0),
        _player(fid, t, 1, 104, 26.0, 20.0),
        _player(fid, t, 1, 105, 30.0, 48.0),
        # forward at a WIDE y (10) so the whole-team width exceeds the back-4 lateral width -- this is
        # what lets rd_width (back-4 lateral, Option B) be distinguished from whole-team team_width.
        _player(fid, t, 1, 106, 65.0, 10.0),  # forward (support near the ball)
        _player(fid, t, 2, 201, 100.0, 34.0, gk=True),  # team2 GK
        _player(fid, t, 2, 202, 75.0, 20.0),  # back4
        _player(fid, t, 2, 203, 79.0, 48.0),
        _player(fid, t, 2, 204, 83.0, 20.0),
        _player(fid, t, 2, 205, 87.0, 48.0),
        _player(fid, t, 2, 206, 45.0, 10.0),  # forward left upfield (counter threat, x in [0, ball])
    ]
    return [*players, _ball(fid, t, ball_x)]


def _away_attacking_frame(fid, t, ball_x):
    """Away (team 2) in possession, attacking left (toward x=0); ball advanced (low x). Team 1
    defends near x=0 with one forward upfield as a counter threat."""
    players = [
        _player(fid, t, 2, 201, 100.0, 34.0, gk=True),  # team2 GK
        _player(fid, t, 2, 202, 75.0, 20.0),  # back4 (deep near own goal x=105)
        _player(fid, t, 2, 203, 79.0, 48.0),
        _player(fid, t, 2, 204, 83.0, 20.0),
        _player(fid, t, 2, 205, 87.0, 48.0),
        # Exact point-reflection of the home frame (x->105-x, y->68-y): forward y 10 -> 58, so a0 and
        # a2 stay mirrors AND the whole-team width still exceeds the back-4 lateral width.
        _player(fid, t, 2, 206, 40.0, 58.0),  # forward advanced toward x=0
        _player(fid, t, 1, 101, 5.0, 34.0, gk=True),  # team1 GK
        _player(fid, t, 1, 102, 18.0, 20.0),  # back4
        _player(fid, t, 1, 103, 22.0, 48.0),
        _player(fid, t, 1, 104, 26.0, 20.0),
        _player(fid, t, 1, 105, 30.0, 48.0),
        _player(fid, t, 1, 106, 60.0, 58.0),  # forward upfield (counter threat, x in [ball, 105])
    ]
    return [*players, _ball(fid, t, ball_x)]


def _home_attacking_frame_variant(fid, t, ball_x):
    """A DISTINCT home-possession layout (higher line, deeper keeper, different shape) so the
    per-sample float metrics VARY across the scored set -- the liveness gate needs non-constant
    float columns (Task 9). Frame 100 (a0) and frame 102 (a2) stay exact point-reflections so the
    orientation-symmetry assertion holds naturally; this variant only feeds a1."""
    players = [
        _player(fid, t, 1, 101, 9.0, 34.0, gk=True),  # team1 GK deeper than frame 100
        _player(fid, t, 1, 102, 26.0, 22.0),  # back4 pushed up (line ~30.5)
        _player(fid, t, 1, 103, 28.0, 46.0),
        _player(fid, t, 1, 104, 32.0, 22.0),
        _player(fid, t, 1, 105, 36.0, 46.0),
        _player(fid, t, 1, 106, 70.0, 10.0),  # forward at a wide y (distinguishes rd_width from team_width)
        _player(fid, t, 2, 201, 96.0, 34.0, gk=True),
        _player(fid, t, 2, 202, 70.0, 22.0),
        _player(fid, t, 2, 203, 74.0, 46.0),
        _player(fid, t, 2, 204, 78.0, 22.0),
        _player(fid, t, 2, 205, 82.0, 46.0),
        _player(fid, t, 2, 206, 50.0, 10.0),
    ]
    return [*players, _ball(fid, t, ball_x)]


def make_rest_defense_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(actions, frames)`` -- the shared multi-domain fixture (see the module docstring)."""
    frame_rows: list[dict] = []
    frame_rows += _home_attacking_frame(100, 10.0, ball_x=70.0)  # a0
    frame_rows += _home_attacking_frame_variant(101, 12.0, ball_x=75.0)  # a1 (loss); distinct layout
    frame_rows += _away_attacking_frame(102, 20.0, ball_x=35.0)  # a2 (loss); point-reflection of a0
    frame_rows += _home_attacking_frame(103, 30.0, ball_x=30.0)  # a3 (non-advanced)
    frames = pd.DataFrame(frame_rows)
    frames["team_id"] = frames["team_id"].astype("Int64")
    frames["player_id"] = frames["player_id"].astype("Int64")
    frames["is_ball"] = frames["is_ball"].astype(bool)
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype(bool)

    actions = pd.DataFrame(
        {
            "game_id": [_GAME] * 4,
            "period_id": [_PERIOD] * 4,
            "action_id": [0, 1, 2, 3],
            "team_id": pd.array([1, 1, 2, 1], dtype="Int64"),
            "player_id": pd.array([106, 106, 206, 106], dtype="Int64"),
            "type_id": [0, 0, 0, 0],  # pass
            "type_name": ["pass"] * 4,
            "result_id": [1, 0, 1, 1],
            "result_name": ["success", "fail", "success", "success"],
            "start_x": [70.0, 75.0, 35.0, 30.0],
            "start_y": [34.0, 34.0, 34.0, 34.0],
            "end_x": [80.0, 50.0, 25.0, 40.0],
            "end_y": [34.0, 34.0, 34.0, 34.0],
            "time_seconds": [10.04, 12.04, 20.04, 30.04],  # jittered off the frame grid
        }
    )
    return actions, frames


def make_fitted_xt():
    """A fitted ExpectedThreat (x-increasing grid), matching the tracking ``fitted_xt`` fixture."""
    import numpy as np

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def make_keeper_sensitive_fixture():
    """``(actions, frames, xt)`` where A (team 1, in possession, attacks right, defends x=0) keeps a
    keeper ALONE in the deep zone and B (team 2) has a counter-receiver broken in BEHIND A's
    rearguard -- so removing A's keeper measurably raises B's deep danger (the non-vacuity anchor).

    The layout is load-bearing (see ``tests/tracking/test_compute_threat_pc``): the keeper registers
    in the threat integral only when it is the NEAREST defender to cells inside a dangerous B
    receiver's Voronoi region, so no A outfielder sits between the keeper (x=4) and the back line
    (x~28), and B's striker (x=14) sits in that keeper-only deep zone. Zero velocities."""
    fid, t = 200, 10.0
    rows = [
        _player(fid, t, 1, 101, 4.0, 34.0, gk=True),  # A keeper, alone deep
        _player(fid, t, 1, 102, 26.0, 20.0),  # A back-4 (defensive_line_x ~ 28)
        _player(fid, t, 1, 103, 26.0, 48.0),
        _player(fid, t, 1, 104, 30.0, 20.0),
        _player(fid, t, 1, 105, 30.0, 48.0),
        _player(fid, t, 1, 106, 58.0, 30.0),  # A attacker near the ball
        _player(fid, t, 2, 201, 14.0, 34.0),  # B striker broken in behind (keeper-only zone)
        _player(fid, t, 2, 202, 48.0, 22.0),  # B rest, upfield
        _player(fid, t, 2, 203, 52.0, 46.0),
        _player(fid, t, 2, 204, 68.0, 30.0),
        _player(fid, t, 2, 205, 72.0, 20.0),
        _player(fid, t, 2, 206, 100.0, 34.0, gk=True),  # B keeper
        _ball(fid, t, 60.0),  # ball with A, committed forward
    ]
    frames = pd.DataFrame(rows)
    frames["team_id"] = frames["team_id"].astype("Int64")
    frames["player_id"] = frames["player_id"].astype("Int64")
    frames["is_ball"] = frames["is_ball"].astype(bool)
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype(bool)
    actions = pd.DataFrame(
        {
            "game_id": [_GAME],
            "period_id": [_PERIOD],
            "action_id": [0],
            "team_id": pd.array([1], dtype="Int64"),
            "player_id": pd.array([106], dtype="Int64"),
            "type_id": [0],
            "type_name": ["pass"],
            "result_id": [1],
            "result_name": ["success"],
            "start_x": [60.0],
            "start_y": [34.0],
            "end_x": [70.0],
            "end_y": [34.0],
            "time_seconds": [10.04],
        }
    )
    return actions, frames, make_fitted_xt()


def make_scaling_fixture(n_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(frames, samples)`` with ``n_frames`` distinct frames, one small sample each.

    Scaling the number of frames scales the number of samples (the loop-iteration dimension), which
    is what makes the growth guard discriminate a rescan: a full-table filter inside the per-sample
    loop is O(n_frames x rows) and goes quadratic, while ``group_rows`` + ``.get()`` stays linear.
    Each frame is deliberately small (2 players per team + a ball) so the WITHIN-group size is
    constant and only the group count grows.
    """
    rows = []
    for fid in range(n_frames):
        for team_id, base in ((1, 10.0), (2, 40.0)):
            for j in range(2):
                rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "team_id": team_id,
                        "player_id": team_id * 10 + j,
                        "x": base + j * 5.0,
                        "y": 34.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "team_id": pd.NA,
                "player_id": pd.NA,
                "x": 50.0,
                "y": 34.0,
                "is_ball": True,
                "is_goalkeeper": False,  # schema: is_goalkeeper is plain bool; ball row is False (not NA)
            }
        )
    frames = pd.DataFrame(rows)
    frames["team_id"] = frames["team_id"].astype("Int64")
    frames["player_id"] = frames["player_id"].astype("Int64")
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype(bool)
    samples = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": list(range(n_frames)),
            "team_id": 1,
            "ball_x": 52.5,
            "own_goal_x": 0.0,
        }
    )
    return frames, samples


def make_score_scaling_fixture(n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(keep, frames, opp_map)`` for the ``_score_samples`` growth guard.

    ``keep`` has one SCORED sample per distinct frame (the loop dimension = ``n_samples``), with the
    engine columns already merged, so ``_score_samples``'s single ``group_rows`` pass stays linear
    while a rescan-in-loop regression would go quadratic (ADR-073)."""
    frames, _base = make_scaling_fixture(n_samples)
    keep = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "team_id": pd.array([1] * n_samples, dtype="Int64"),
            "action_id": list(range(n_samples)),
            "possession_id": list(range(n_samples)),
            "frame_id": pd.array(list(range(n_samples)), dtype="Int64"),
            "ball_x": 52.5,
            "own_goal_x": 0.0,
            "attacked_goal_x": 105.0,
            "is_possession_loss": False,
            "gate_drop_reason": pd.array([pd.NA] * n_samples, dtype="object"),
            "defensive_line_x": 25.0,
            "compactness_x": 10.0,
            "lateral_width": 20.0,
            "team_length": 30.0,
        }
    )
    opp_map = {("1", "1"): 2, ("1", "2"): 1}
    return keep, frames, opp_map
