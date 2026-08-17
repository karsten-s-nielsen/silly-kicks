"""A2: structural SkillCorner rate-gates (all legs, not @e2e).

These assert the rate is computed, finite, and under a DELIBERATELY LOOSE ceiling, plus a MANDATORY
both-sides mutation (a fixture that breaches the ceiling is asserted to fail). The TIGHT corpus
baseline lives in the owner-run @e2e data-contract (A3), not here -- C2's population split.
"""

from __future__ import annotations

import numpy as np
import validate_skillcorner_keeper_origin as drv

from tests.scripts.conftest import _fixture_with_all_rows_offpitch

_LOOSE_OFFPITCH_CEILING = 0.25
_LOOSE_OOR_CEILING = 0.70  # generous: the synthetic fixture deliberately plants one broadcast artifact


def test_offpitch_rate_computed_finite_and_under_loose_ceiling(slim_skillcorner_match):
    frame = drv.measure_match(*slim_skillcorner_match)
    rate = drv.offpitch_rate(frame)
    assert np.isfinite(rate)
    assert 0.0 <= rate <= _LOOSE_OFFPITCH_CEILING


def test_offpitch_gate_fails_when_breached():
    # mandatory failing side: every row gross-off-pitch -> rate 1.0 > the loose ceiling
    frame = _fixture_with_all_rows_offpitch()
    assert drv.offpitch_rate(frame) > _LOOSE_OFFPITCH_CEILING


def test_out_of_region_rate_computed_finite_and_under_loose_ceiling(slim_skillcorner_match):
    frame = drv.measure_match(*slim_skillcorner_match)
    rate = drv.out_of_region_goalkick_rate(frame)
    assert np.isfinite(rate)
    assert 0.0 <= rate <= _LOOSE_OOR_CEILING


def test_out_of_region_gate_fails_when_breached():
    # mandatory failing side: every goal-kick's RESOLVED origin out of the own box -> rate 1.0
    import pandas as pd

    frame = pd.DataFrame({"is_goalkick": [True, True, True], "in_own_box": [False, False, False]})
    assert drv.out_of_region_goalkick_rate(frame) > _LOOSE_OOR_CEILING
