"""Task 7 gates: paired-leg DAS (SC-1 derived moved set; ADR-043-safe; non-vacuous)."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking._das_engine import compute_das, compute_das_paired
from silly_kicks.tracking._das_pack import pack_frames, pack_paired
from silly_kicks.tracking._das_params import DAS_PARAMS
from tests.tracking._das_helpers import ghost_pair_from_golden

_ENGINES = ["numpy"]
try:
    import numba  # noqa: F401

    _ENGINES.append("numba")
except ImportError:
    pass


@pytest.mark.parametrize("engine", _ENGINES)
def test_paired_equals_two_independent_calls_bitwise(engine):
    actual, ghost = ghost_pair_from_golden("S01", keeper_shift_m=3.0)
    pa, pc, moved = pack_paired(actual, ghost, attacking_direction_col="dir")
    leg_a, leg_c = compute_das_paired(pa, pc, moved, DAS_PARAMS, engine=engine)
    solo_a = compute_das(pack_frames(actual, attacking_direction_col="dir"), DAS_PARAMS, engine=engine)
    solo_c = compute_das(pack_frames(ghost, attacking_direction_col="dir"), DAS_PARAMS, engine=engine)
    for got, exp in ((leg_a, solo_a), (leg_c, solo_c)):
        assert np.array_equal(got.team_das, exp.team_das, equal_nan=True)
        assert np.array_equal(got.player_das, exp.player_das, equal_nan=True)


def test_moved_mask_marks_only_the_keeper():
    actual, ghost = ghost_pair_from_golden("S01", keeper_shift_m=3.0)
    _pa, _pc, moved = pack_paired(actual, ghost, attacking_direction_col="dir")
    assert int(moved.sum()) == 3  # one keeper row per frame, 3 frames


def test_moved_keeper_changes_das():
    actual, ghost = ghost_pair_from_golden("S01", keeper_shift_m=8.0)
    pa, pc, moved = pack_paired(actual, ghost, attacking_direction_col="dir")
    leg_a, leg_c = compute_das_paired(pa, pc, moved, DAS_PARAMS, engine="numpy")
    assert np.nanmax(np.abs(leg_a.team_das - leg_c.team_das)) > 0.0


@pytest.mark.parametrize(
    "mutation", ["ball_moved", "possession_changed", "carrier_changed", "row_order_changed", "player_set_changed"]
)
def test_leg_contract_violations_raise(mutation):
    actual, ghost = ghost_pair_from_golden("S01")
    if mutation == "ball_moved":
        ghost.loc[ghost["is_ball"], "x"] = ghost.loc[ghost["is_ball"], "x"].astype(float) + 1.0
    elif mutation == "possession_changed":
        ghost["team_in_possession"] = 2
    elif mutation == "carrier_changed":
        actual["ball_carrier_player_id"] = 100  # golden frames carry no carrier col; add to both
        ghost["ball_carrier_player_id"] = 999
    elif mutation == "row_order_changed":
        ghost = ghost.iloc[::-1].reset_index(drop=True)
    elif mutation == "player_set_changed":
        ghost = ghost.iloc[:-1].reset_index(drop=True)  # drop a row -> different length/structure
    with pytest.raises(ValueError):
        pack_paired(actual, ghost, attacking_direction_col="dir")
