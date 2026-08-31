"""compute_threat_pc field_weight hook (TF-60 PR2): additive, default byte-identical."""

import numpy as np

from silly_kicks.tracking import compute_threat_pc
from tests.tracking.test_compute_threat_pc import HOME_GOAL_MAP, _fitted_xt, _frame


def test_field_weight_none_is_byte_identical_to_default():
    frame = _frame()
    a = compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    b = compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP, field_weight=None)
    assert a == b  # exact


def test_uniform_half_weight_halves_the_threat():
    frame = _frame()
    base = compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    halved = compute_threat_pc(
        frame,
        attacking_team_id=2,
        xt=_fitted_xt(),
        goal_map=HOME_GOAL_MAP,
        field_weight=lambda gx, gy: np.full((len(gy), len(gx)), 0.5),
    )
    assert base > 0.0
    assert abs(halved - base * 0.5) < 1e-12


def test_zero_weight_zeroes_the_threat():
    frame = _frame()
    zeroed = compute_threat_pc(
        frame,
        attacking_team_id=2,
        xt=_fitted_xt(),
        goal_map=HOME_GOAL_MAP,
        field_weight=lambda gx, gy: np.zeros((len(gy), len(gx))),
    )
    assert zeroed == 0.0
