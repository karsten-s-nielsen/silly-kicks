"""Purity gate for shot_stopping (ADR-033 discipline): compute_shot_stopping never mutates inputs."""

from __future__ import annotations

import pandas as pd

from silly_kicks.shot_stopping import compute_shot_stopping
from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_FAIL = spadlconfig.result_id["fail"]


def _actions() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "game_id": 1,
                "period_id": 1,
                "team_id": 10,
                "type_id": _SHOT,
                "result_id": _FAIL,
                "psxg": 0.3,
                "defending_gk_player_id": 99,
                "defending_gk_team_id": 20,
                "shot_blocked": pd.NA,
            },
        ]
    )
    df["shot_blocked"] = df["shot_blocked"].astype("boolean")
    df["defending_gk_player_id"] = df["defending_gk_player_id"].astype("object")
    df["defending_gk_team_id"] = df["defending_gk_team_id"].astype("object")
    return df


def test_compute_is_pure():
    actions = _actions()
    before = actions.copy()
    out, _ = compute_shot_stopping(actions, psxg_column="psxg")
    pd.testing.assert_frame_equal(actions, before)
    assert out is not actions
