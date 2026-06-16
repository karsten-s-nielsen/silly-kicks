"""Tiny shared fixture + the enumerated aggregator surface for the id-dtype gate (ADR-019).

`AGGREGATORS` is the list under test (per-aggregator adapters with the uniform gate signature
`(actions, frames, home_team_id) -> DataFrame`). `REGISTERED_AGGREGATORS` is derived from the
public `add_*` exports so the meta-assertion (B3) catches an aggregator added but not wired in.
"""

import functools

import numpy as np
import pandas as pd

from silly_kicks.tracking import features as F


def make_actions() -> pd.DataFrame:
    """2 actions, team 5 in possession, numeric ids (int64 — the GS SPADL contract)."""
    return pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [0, 1],
            "period_id": [1, 1],
            "time_seconds": [10.0, 20.0],
            "team_id": pd.Series([5, 5], dtype="int64"),
            "player_id": pd.Series([10, 11], dtype="int64"),
            # defending GK = opponent (team 6) keeper = player 2; populated as int64 so the
            # asymmetric variant stringifies it (it is an entity id) and exercises the GK seam.
            "defending_gk_player_id": pd.Series([2, 2], dtype="int64"),
            "start_x": [50.0, 60.0],
            "start_y": [34.0, 30.0],
            "end_x": [60.0, 80.0],
            "end_y": [30.0, 40.0],
            "type_id": [0, 11],
            "result_id": [1, 1],
            "bodypart_id": [0, 0],
        }
    )


def make_frames() -> pd.DataFrame:
    """Non-degenerate fixture: 5 outfield + 1 GK per team (5, 6) + ball, sampled across a
    pre-window before each action time so defensive-line geometry and off-ball runs produce
    REAL (non-NaN), home_team_id/team-resolution-SENSITIVE feature values -- otherwise the
    gate false-greens on all-NaN output without exercising the seam. GS-style Int64 ids.

    Team 5 (home, defends x=0) attacks toward x=105: its outfielders advance +x across the
    window (off-ball runs toward the attacking goal). Team 6 defends x=105.
    """
    # base x positions at the START of each pre-window (team 5 lower, team 6 higher)
    team5 = {10: 50.0, 11: 45.0, 12: 30.0, 13: 25.0, 14: 20.0}  # outfield
    team6 = {20: 70.0, 21: 72.0, 22: 80.0, 23: 85.0, 24: 88.0}  # outfield
    gks = {1: (5, 4.0), 2: (6, 101.0)}  # pid -> (team, x)
    offsets = (-1.4, -1.05, -0.7, -0.35, 0.0)  # within pre_seconds=1.5

    rows = []
    for t_a in (10.0, 20.0):
        for off in offsets:
            t = round(t_a + off, 3)
            frac = (off + 1.4) / 1.4  # 0 at window start -> 1 at the action
            for pid, bx in team5.items():
                rows.append(_frow(pid, 5, False, bx + 6.0 * frac, 34.0 + pid % 3, t, t_a))
            for pid, bx in team6.items():
                rows.append(_frow(pid, 6, False, bx - 2.0 * frac, 30.0 + pid % 3, t, t_a))
            for pid, (team, x) in gks.items():
                rows.append(_frow(pid, team, True, x, 34.0, t, t_a))
            rows.append(_frow(pd.NA, pd.NA, False, 58.0 + 4.0 * frac, 32.0, t, t_a, is_ball=True))
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f


def _frow(pid, team, gk, x, y, t, t_a, *, is_ball=False):
    return dict(
        game_id=1,
        period_id=1,
        frame_id=round(t * 25),
        time_seconds=t,
        frame_rate=25.0,
        player_id=pid,
        team_id=team,
        is_ball=is_ball,
        is_goalkeeper=gk,
        x=float(x),
        y=float(y),
        z=0.0,
        speed=4.3 if not is_ball else 6.0,
        vx=4.3,
        vy=0.0,
        speed_source="native",
        ball_state="alive",
        team_attacking_direction="ltr",
        confidence=None,
        visibility=None,
        source_provider="gradientsports",
        is_goalkeeper_source="native",
    )


@functools.cache
def _xt():
    # Any VALID fitted xT suffices -- the gate asserts dtype-invariance with a shared artifact,
    # so this is NOT required to track tests/conftest.py::fitted_xt. Built LAZILY so an xT-API
    # problem breaks only the two influence tests, not collection of the whole gate.
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _named(fn, name):
    fn.__name__ = name
    return fn


def _a(fn, name):  # no home_team_id (teams resolved from id columns internally)
    return _named(lambda a, f, home_team_id: fn(a, f), name)


def _ah(fn, name, **extra):  # keyword-only home_team_id
    return _named(lambda a, f, home_team_id: fn(a, f, home_team_id=home_team_id, **extra), name)


def _axh(fn, name):  # positional xt + keyword home_team_id (influence/cover-shadow family)
    return _named(lambda a, f, home_team_id: fn(a, f, _xt(), home_team_id=home_team_id), name)


AGGREGATORS = [
    _a(F.add_action_context, "add_action_context"),
    _a(F.add_actor_pre_window, "add_actor_pre_window"),
    _a(F.add_das, "add_das"),
    _a(F.add_elastic_sync, "add_elastic_sync"),
    _a(F.add_gk_completion, "add_gk_completion"),
    _a(F.add_obso, "add_obso"),
    _a(F.add_pausa, "add_pausa"),
    _a(F.add_pitch_control, "add_pitch_control"),
    _named(lambda a, f, home_team_id: F.add_pre_shot_gk_angle(a, frames=f), "add_pre_shot_gk_angle"),
    _a(F.add_pre_shot_gk_position, "add_pre_shot_gk_position"),
    _a(F.add_pressure_on_actor, "add_pressure_on_actor"),
    _a(F.add_shot_goalmouth, "add_shot_goalmouth"),
    _a(F.add_space_creation, "add_space_creation"),
    _ah(F.add_defensive_line, "add_defensive_line", n=4),
    _ah(F.add_line_break, "add_line_break"),
    _ah(F.add_off_ball_context, "add_off_ball_context"),
    _ah(F.add_off_ball_runs, "add_off_ball_runs"),
    _ah(F.add_shape_graph, "add_shape_graph"),
    _ah(F.add_structural_pass, "add_structural_pass"),
    _ah(F.add_team_shape, "add_team_shape"),
    _ah(F.add_ghost_gk, "add_ghost_gk", kde_backend="cpu-numba"),
    _axh(F.add_cover_shadows, "add_cover_shadows"),
    _axh(F.add_gk_influence, "add_gk_influence"),
    _axh(F.add_player_influence, "add_player_influence"),
    _axh(F.add_xt_gk, "add_xt_gk"),
]

# Public add_* surface -- the meta-assertion (B3) checks AGGREGATORS covers all LINKED ones.
REGISTERED_AGGREGATORS = {name for name in dir(F) if name.startswith("add_") and callable(getattr(F, name))}

# Aggregators that legitimately compare NO ids. Each entry MUST carry a one-line "compares no
# ids" justification (N-d). The AST lint (Task 10) is the cross-check.
NON_LINKED_AGGREGATORS: dict[str, str] = {
    # "add_xxx": "reason it compares no action/frame/home_team ids",
}
