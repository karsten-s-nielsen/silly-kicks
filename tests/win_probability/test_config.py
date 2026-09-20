import dataclasses

import pytest

from silly_kicks.win_probability import (
    WIN_PROBABILITY_COLUMNS,
    WIN_PROBABILITY_KEYS,
    WinProbabilityParams,
)


def test_params_frozen_and_defaults():
    p = WinProbabilityParams.default()
    assert p.interval_minutes == 1
    assert p.regulation_minutes == 90
    assert p.lattice_pad == 10
    assert p.ece_max == 0.10 and p.slope_tol == 0.25
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.interval_minutes = 2  # type: ignore[misc]


def test_for_provider_is_empty_adr009():
    # ADR-009: no per-provider tuning ships in v1.
    assert WinProbabilityParams.for_provider("skillcorner") == WinProbabilityParams.default()


def test_columns_contract():
    assert WIN_PROBABILITY_KEYS == ("game_id", "action_id")
    for c in (
        "game_id",
        "action_id",
        "team_id",
        "period_id",
        "p_win",
        "p_draw",
        "p_loss",
        "win_prob_leverage",
        "win_prob_source",
    ):
        assert c in WIN_PROBABILITY_COLUMNS
    for c in ("p_win", "p_draw", "p_loss", "win_prob_leverage"):
        assert WIN_PROBABILITY_COLUMNS[c] == "float64"
