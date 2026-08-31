"""OBPV w_field closure (TF-60 PR2): oriented toward G_A, sigmoid x Gaussian."""

import numpy as np

from silly_kicks.restdefense import WFieldParams
from silly_kicks.restdefense._wfield import build_w_field

_GX = np.linspace(0.0, 105.0, 50)
_GY = np.linspace(0.0, 68.0, 32)


def test_shape_and_range():
    w = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    assert w.shape == (len(_GY), len(_GX))
    assert np.all((w > 0.0) & (w <= 1.0))


def test_high_near_defended_goal_low_far():
    w = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    mid_y = len(_GY) // 2
    assert w[mid_y, 0] > w[mid_y, -1]  # near G_A=0 weighted above the far end


def test_orientation_flips_with_goal_end():
    lo = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    hi = build_w_field(own_goal_x=105.0, params=WFieldParams())(_GX, _GY)
    mid_y = len(_GY) // 2
    assert lo[mid_y, 0] > lo[mid_y, -1]
    assert hi[mid_y, -1] > hi[mid_y, 0]  # mirror


def test_central_channel_weighted_above_wings():
    w = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    assert w[len(_GY) // 2, 0] > w[0, 0]  # centre-y > touchline-y at the same x
