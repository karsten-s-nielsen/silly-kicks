import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk._moves import (
    MOVE_TYPE_IDS,
    _is_turnover,
    extended_move_actions,
    xtgk_action_prob,
    xtgk_transition_matrix,
)
from silly_kicks.xthreat import GridSpec
from tests.xtgk.conftest import (
    FAIL,
    GOALKICK,
    PASS,
    SHOT,
    SUCCESS,
    THROW_IN,
    _row,
    make_cohort,
)

GRID = GridSpec(n_zones_x=16, n_zones_y=12)


def test_is_turnover_is_failed_move_only():
    df = pd.DataFrame({"type_id": [PASS, PASS, SHOT], "result_id": [SUCCESS, FAIL, FAIL]})
    out = _is_turnover(df)
    assert list(out) == [False, True, False]  # failed pass=turnover; failed shot is NOT a move-set turnover


def test_extended_move_set_includes_goalkick_and_throw_in_not_shots():
    rows = [
        _row(0, PASS, SUCCESS, 10, 34, 20, 34),
        _row(1, GOALKICK, SUCCESS, 5, 34, 50, 34),
        _row(2, THROW_IN, SUCCESS, 40, 0, 45, 10),
        _row(3, SHOT, SUCCESS, 100, 34, 105, 34),
    ]
    out = extended_move_actions(make_cohort(rows))
    assert set(out["type_id"]) == {PASS, GOALKICK, THROW_IN}
    assert spadlconfig.actiontype_id["goalkick"] in MOVE_TYPE_IDS


def test_singh_transition_includes_goalkick_rows():
    rows = [_row(0, GOALKICK, SUCCESS, 5, 34, 60, 34), _row(1, PASS, SUCCESS, 60, 34, 80, 34)]
    transition = xtgk_transition_matrix(make_cohort(rows), GRID, method="singh_counts")
    assert transition.sum() > 0  # goal-kick produced a transition row (excluded by classic xT)


def test_kde_path_uses_successful_moves_only():
    rows = [
        _row(0, GOALKICK, FAIL, 5, 34, 60, 34),  # failed -> no destination
        _row(1, GOALKICK, SUCCESS, 5, 34, 62, 34),  # success -> destination
    ]
    transition = xtgk_transition_matrix(make_cohort(rows), GRID, method="kde_smoothed")
    assert np.isfinite(transition).all()


def test_action_prob_returns_grid_shaped_probs():
    rows = [_row(i, PASS, SUCCESS, 10 + i, 34, 60, 40) for i in range(5)]
    rows.append(_row(5, SHOT, SUCCESS, 100, 34, 105, 34))
    p_shot, p_move = xtgk_action_prob(make_cohort(rows), 16, 12)
    assert p_shot.shape == (12, 16) and p_move.shape == (12, 16)
