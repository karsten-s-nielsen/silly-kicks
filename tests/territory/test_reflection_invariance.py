"""ADR-028 frame reconciliation: the opponent-pass reflection is load-bearing + team-symmetric.

The hull is in the DEFENDER's action-LTR frame; the opponent's passes are in the opponent's frame (180
degrees apart). Membership must reflect the opponent pass end into the defender frame -- if it did not,
an opponent attacking-third pass (near x=105 in its own frame) would never register in the defender's
own-third hull (near x=0). These tests pin that the reflection is what makes the pass count, and that a
role-swapped (either-team-perspective) scene scores identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.territory import TerritoryParams, compute_territorial_dominance
from silly_kicks.territory._hull import build_trimmed_hull
from silly_kicks.xthreat import ExpectedThreat

_SUCCESS = spadlconfig.result_id["success"]
_TACKLE = spadlconfig.actiontype_id["tackle"]
_PASS = spadlconfig.actiontype_id["pass"]
_KEEP_ALL = TerritoryParams(trim_fraction=1.0)


def _toy_xt(value: float = 0.1) -> ExpectedThreat:
    xt = ExpectedThreat()
    xt.xT = np.full(np.asarray(xt.xT).shape, value, dtype=float)
    return xt


def _def(game, player, team, x, y):
    return {
        "game_id": game,
        "period_id": 1,
        "team_id": team,
        "player_id": player,
        "type_id": _TACKLE,
        "result_id": _SUCCESS,
        "start_x": x,
        "start_y": y,
        "end_x": x,
        "end_y": y,
        "time_seconds": 10.0,
    }


def _pass(game, team, sx, sy, ex, ey):
    return {
        "game_id": game,
        "period_id": 1,
        "team_id": team,
        "player_id": 99,
        "type_id": _PASS,
        "result_id": _SUCCESS,
        "start_x": sx,
        "start_y": sy,
        "end_x": ex,
        "end_y": ey,
        "time_seconds": 20.0,
    }


def _actions(rows):
    df = pd.DataFrame(rows)
    df["action_id"] = range(len(df))
    return df


def _hull_corners(game, player, team):
    return [
        _def(game, player, team, 5, 20),
        _def(game, player, team, 15, 20),
        _def(game, player, team, 15, 48),
        _def(game, player, team, 5, 48),
    ]


def test_reflection_is_load_bearing():
    # Directly: the raw (opponent-frame) pass end is OUTSIDE the defender hull; the reflected end is
    # INSIDE. So compute counts the pass ONLY because it reflects -- skipping the reflection scores 0.
    hull = build_trimmed_hull(np.array([[5.0, 20], [15, 20], [15, 48], [5, 48]]), trim_fraction=1.0)
    assert hull is not None
    assert not bool(hull.contains(np.array([95.0, 40.0])))  # raw opponent-frame end: OUTSIDE
    assert bool(hull.contains(np.array([105 - 95.0, 68 - 40.0])))  # reflected (10,28): INSIDE

    rows = [*_hull_corners(1, 1, 10), _pass(1, 20, 80, 40, 95, 40)]
    out, _ = compute_territorial_dominance(_actions(rows), xt=_toy_xt(0.1), params=_KEEP_ALL)
    assert out.iloc[0]["territory_xt_conceded"] == pytest.approx(0.1)  # counted, via the reflection


def test_team_symmetry_either_perspective():
    # Same physical structure with the roles SWAPPED (team 20 defends, team 10 attacks) scores identically
    # -- no home/away / team-identity bias (ADR-028 / ADR-051-D3).
    a = [*_hull_corners(1, 1, 10), _pass(1, 20, 80, 40, 95, 40)]  # team 10 defends
    b = [*_hull_corners(2, 2, 20), _pass(2, 10, 80, 40, 95, 40)]  # team 20 defends, mirror roles
    out_a, _ = compute_territorial_dominance(_actions(a), xt=_toy_xt(0.1), params=_KEEP_ALL)
    out_b, _ = compute_territorial_dominance(_actions(b), xt=_toy_xt(0.1), params=_KEEP_ALL)
    assert out_a.iloc[0]["territory_xt_conceded"] == pytest.approx(out_b.iloc[0]["territory_xt_conceded"])
    assert out_a.iloc[0]["territory_xt_conceded"] == pytest.approx(0.1)
