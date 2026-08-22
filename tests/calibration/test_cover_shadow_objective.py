"""Task 11: the sigma/lambda discrimination re-tuning objective."""

from __future__ import annotations

import numpy as np

from silly_kicks.calibration._cover_shadow_objective import (
    CoverShadowDiscriminationObjective,
    PreparedPass,
    lane_pressure_shift_share,
)
from tests.tracking.test_cover_shadows import HOME_GOAL_MAP, _make_lane_control_frame


def _pass(defender_pos, is_fail: bool) -> PreparedPass:
    fr = _make_lane_control_frame(passer_pos=(50.0, 34.0), receiver_pos=(75.0, 34.0), defender_pos=defender_pos)
    return PreparedPass(fr, (50.0, 34.0), (75.0, 34.0), 2, HOME_GOAL_MAP, is_fail, not is_fail)


def _passes() -> list[PreparedPass]:
    # failed = defender on the pass line (blocked, high margin); completed = defender far off (open)
    on = [(62.5, 34.0), (60.0, 34.0), (65.0, 34.0)]
    off = [(62.5, 5.0), (60.0, 60.0), (65.0, 8.0)]
    return [_pass(p, True) for p in on] + [_pass(p, False) for p in off]


def test_margins_respond_to_sigma():
    obj = CoverShadowDiscriminationObjective(_passes())
    m1, *_ = obj._measure(0.20, 4.3)
    m2, *_ = obj._measure(0.60, 4.3)
    assert not np.allclose(m1, m2)  # sigma changes the blocking margins (non-vacuity)


def test_objective_discriminates_failed_from_completed():
    assert CoverShadowDiscriminationObjective(_passes()).score(0.20, 4.3) >= 0.8


def test_fp_constraint_rejects_overblocking():
    passes = [*_passes(), _pass((62.5, 34.0), is_fail=False)]  # blocked BUT completed = a false positive
    assert np.isnan(CoverShadowDiscriminationObjective(passes, incumbent_fp=0.0).score(0.20, 4.3))
    assert np.isfinite(CoverShadowDiscriminationObjective(passes, incumbent_fp=1.0).score(0.20, 4.3))


def test_lane_pressure_shift_share_bounds():
    assert lane_pressure_shift_share((0.2, 4.3), (0.2, 4.3), sigma_range=1.0, lambda_range=5.0) == 0.0
    v = lane_pressure_shift_share((0.2, 4.3), (0.7, 4.3), sigma_range=1.0, lambda_range=5.0)
    assert 0.0 < v <= 1.0


def test_lane_pressure_shift_share_clamps_over_wide_delta():
    """A caller passing too-small a range (understating span -> overstating the shift beyond 1.0) must not
    understate the bias below MAX_BIAS_SHARE by wrapping past 1.0 -- the result is CLAMPED to [0, 1]."""
    v = lane_pressure_shift_share((0.2, 4.3), (2.2, 14.3), sigma_range=1.0, lambda_range=5.0)
    assert v == 1.0  # ds=2, dl=2 -> hypot(2,2)/sqrt(2)=2.0, clamped to 1.0
