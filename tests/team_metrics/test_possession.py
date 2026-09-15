"""Possession-foundation layer (TF-52 Task 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.team_metrics import TeamKpiParams
from silly_kicks.team_metrics._possession import (
    add_possession_context,
    build_spells,
    possession_minutes,
)

_PASS = spadlconfig.actiontype_id["pass"]
_INT = spadlconfig.actiontype_id["interception"]
_SUCCESS = spadlconfig.result_id["success"]


def _fixture(team_ids=(10, 10, 10, 20, 20)):
    # game 1: team 10 holds t=0..8 (3 passes), team 20 wins it t=10..12 (interception, pass).
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1],
            "team_id": list(team_ids),
            "player_id": [1, 2, 3, 4, 5],
            "type_id": [_PASS, _PASS, _PASS, _INT, _PASS],
            "result_id": [_SUCCESS] * 5,
            "time_seconds": [0.0, 4.0, 8.0, 10.0, 12.0],
            "start_x": [20.0, 40.0, 60.0, 50.0, 55.0],
            "start_y": [34.0] * 5,
            "end_x": [40.0, 60.0, 80.0, 55.0, 60.0],
            "end_y": [34.0] * 5,
            "action_id": [0, 1, 2, 3, 4],
        }
    )


def test_context_is_pure_and_adds_columns():
    actions = _fixture()
    snapshot = actions.copy(deep=True)
    ctx = add_possession_context(actions, params=TeamKpiParams())
    pd.testing.assert_frame_equal(actions, snapshot)  # input unmutated
    assert "possession_id" in ctx.columns
    assert "team_in_possession" in ctx.columns
    assert ctx["possession_id"].nunique() == 2


def test_spells_and_minutes():
    ctx = add_possession_context(_fixture(), params=TeamKpiParams())
    spells = build_spells(ctx)
    assert len(spells) == 2

    p10 = spells[spells["team_id"] == 10].iloc[0]
    assert p10["duration_s"] == 8.0
    assert bool(p10["is_open_play"]) is True
    assert bool(p10["is_recovery"]) is False  # first possession of the period

    p20 = spells[spells["team_id"] == 20].iloc[0]
    assert p20["duration_s"] == 2.0
    assert bool(p20["is_recovery"]) is True  # team changed 10 -> 20

    mins = possession_minutes(spells)
    m10 = mins[mins["team_id"] == 10].iloc[0]
    assert m10["in_possession_min"] == 8.0 / 60
    assert m10["out_of_possession_min"] == 2.0 / 60
    m20 = mins[mins["team_id"] == 20].iloc[0]
    assert m20["in_possession_min"] == 2.0 / 60
    assert m20["out_of_possession_min"] == 8.0 / 60


def test_open_play_flag_on_set_piece_start():
    actions = _fixture()
    actions.loc[0, "type_id"] = spadlconfig.actiontype_id["goalkick"]  # first possession starts on a goalkick
    ctx = add_possession_context(actions, params=TeamKpiParams())
    spells = build_spells(ctx)
    first_spell = spells.sort_values(["period_id", "start_time"]).iloc[0]
    assert bool(first_spell["is_open_play"]) is False


def test_id_dtype_invariance_string_team_ids():
    # ADR-019: string team ids must yield the SAME durations/minutes as integer ids.
    spells_int = build_spells(add_possession_context(_fixture(), params=TeamKpiParams()))
    spells_str = build_spells(
        add_possession_context(_fixture(team_ids=("10", "10", "10", "20", "20")), params=TeamKpiParams())
    )
    np.testing.assert_array_equal(
        np.sort(spells_int["duration_s"].to_numpy()),
        np.sort(spells_str["duration_s"].to_numpy()),
    )
    mins_int = possession_minutes(spells_int).sort_values("in_possession_min")
    mins_str = possession_minutes(spells_str).sort_values("in_possession_min")
    np.testing.assert_allclose(mins_int["in_possession_min"].to_numpy(), mins_str["in_possession_min"].to_numpy())
    np.testing.assert_allclose(
        mins_int["out_of_possession_min"].to_numpy(), mins_str["out_of_possession_min"].to_numpy()
    )
