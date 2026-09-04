"""compute_territorial_dominance is PURE -- never mutates the caller's actions (ADR-033).

Two variants (per the ADR-033 contract for a conditional path): window=None (per-game atoms) and a
window set (pooled aggregation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.territory import TerritoryParams, compute_territorial_dominance
from silly_kicks.xthreat import ExpectedThreat

_SUCCESS = spadlconfig.result_id["success"]
_TACKLE = spadlconfig.actiontype_id["tackle"]
_PASS = spadlconfig.actiontype_id["pass"]
_KEEP_ALL = TerritoryParams(trim_fraction=1.0)


def _toy_xt(value: float = 0.1) -> ExpectedThreat:
    xt = ExpectedThreat()
    xt.xT = np.full(np.asarray(xt.xT).shape, value, dtype=float)
    return xt


def _scene():
    rows = [
        {
            "game_id": 1,
            "period_id": 1,
            "team_id": 10,
            "player_id": 1,
            "type_id": _TACKLE,
            "result_id": _SUCCESS,
            "start_x": x,
            "start_y": y,
            "end_x": x,
            "end_y": y,
            "time_seconds": 10.0,
        }
        for x, y in [(5, 20), (15, 20), (15, 48), (5, 48)]
    ] + [
        {
            "game_id": 1,
            "period_id": 1,
            "team_id": 20,
            "player_id": 99,
            "type_id": _PASS,
            "result_id": _SUCCESS,
            "start_x": 80,
            "start_y": 40,
            "end_x": 95,
            "end_y": 40,
            "time_seconds": 20.0,
        }
    ]
    df = pd.DataFrame(rows)
    df["action_id"] = range(len(df))
    return df


@pytest.mark.parametrize("window", [None, [1]])
def test_compute_does_not_mutate_actions(window):
    actions = _scene()
    snapshot = actions.copy(deep=True)
    out, _ = compute_territorial_dominance(actions, xt=_toy_xt(0.1), window=window, params=_KEEP_ALL)
    pd.testing.assert_frame_equal(actions, snapshot)  # input untouched
    assert out is not actions
