"""Physical invariant: GK angles in [-pi, pi].

``pre_shot_gk_angle_to_shot_trajectory`` and ``pre_shot_gk_angle_off_goal_line``
both return values in radians and must be bounded by [-pi, pi]. Parametrised
over the 4 supported tracking providers via the per-period-dense synthesizer.

Memory: feedback_invariant_testing (PR-S24 dedicated-files structural fix).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import (
    pre_shot_gk_angle_off_goal_line,
    pre_shot_gk_angle_to_shot_trajectory,
)
from tests.tracking._provider_inputs import (
    load_provider_frames,
    synthesize_actions_per_period_dense,
)


def _enriched_actions_for(provider: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = load_provider_frames(provider)
    actions = synthesize_actions_per_period_dense(frames)
    enriched = add_pre_shot_gk_context(actions)
    return enriched, frames


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_to_shot_trajectory_bounded(provider):
    enriched, frames = _enriched_actions_for(provider)
    s = pre_shot_gk_angle_to_shot_trajectory(enriched, frames)
    valid = s.dropna()
    if len(valid) == 0:
        pytest.skip(f"{provider}: no valid GK-angle rows in synthesized fixture")
    assert valid.min() >= -math.pi - 1e-9
    assert valid.max() <= math.pi + 1e-9


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_off_goal_line_bounded(provider):
    enriched, frames = _enriched_actions_for(provider)
    s = pre_shot_gk_angle_off_goal_line(enriched, frames)
    valid = s.dropna()
    if len(valid) == 0:
        pytest.skip(f"{provider}: no valid GK-angle rows in synthesized fixture")
    assert valid.min() >= -math.pi - 1e-9
    assert valid.max() <= math.pi + 1e-9
