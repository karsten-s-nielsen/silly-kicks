"""Per-period DOP-symmetry invariant for TF-12 GK angle features.

Closes the 3.0.0 / 3.0.1 blind-spot pattern: every numeric tracking-aware
feature must explicitly verify that LTR-normalized output is identical
across periods (P1 / P2 / ET) when the underlying physical situation is.

Memory: feedback_invariant_testing (fixture-density corollary -- the
invariant must be physically exercised, not vacuously skip-passing).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import (
    pre_shot_gk_angle_off_goal_line,
    pre_shot_gk_angle_to_shot_trajectory,
)

_SHOT_TYPE_ID = spadlconfig.actiontype_id["shot"]


def _make_mirrored_shot_actions_frames(provider: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two shot actions in different periods with identical post-LTR-normalization
    physical setup. After LTR normalization, both periods attack toward x=105;
    the angle output must be identical for both shots within float tolerance.
    """
    common = {
        "game_id": "G1",
        "type_id": _SHOT_TYPE_ID,
        "team_id": 1,
        "player_id": 10,
        "start_y": 34.0,
        "defending_gk_player_id": 99.0,
    }
    actions = pd.DataFrame(
        [
            {"action_id": 1, "period_id": 1, "start_x": 95.0, "time_seconds": 10.0, **common},
            {"action_id": 2, "period_id": 2, "start_x": 95.0, "time_seconds": 60.0, **common},
        ]
    )
    frame_rows = []
    for period, t in [(1, 10.0), (2, 60.0)]:
        for player_id, x, y, is_gk, team in [(10, 95.0, 34.0, False, 1), (99, 100.0, 36.0, True, 2)]:
            frame_rows.append(
                {
                    "game_id": "G1",
                    "period_id": period,
                    "frame_id": 100,
                    "time_seconds": t,
                    "frame_rate": 25.0,
                    "player_id": player_id,
                    "team_id": team,
                    "is_ball": False,
                    "is_goalkeeper": is_gk,
                    "x": x,
                    "y": y,
                    "z": np.nan,
                    "speed": 0.0,
                    "speed_source": "native",
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "source_provider": provider,
                }
            )
    frames = pd.DataFrame(frame_rows)
    return actions, frames


@pytest.mark.parametrize("provider", ["sportec", "metrica"])
def test_per_period_dop_symmetry_to_shot_trajectory(provider):
    actions, frames = _make_mirrored_shot_actions_frames(provider)
    s = pre_shot_gk_angle_to_shot_trajectory(actions, frames)
    assert math.isclose(float(s.iloc[0]), float(s.iloc[1]), abs_tol=1e-9), (
        f"{provider}: per-period DOP-symmetry violated for to_shot_trajectory: P1={s.iloc[0]} vs P2={s.iloc[1]}"
    )


@pytest.mark.parametrize("provider", ["sportec", "metrica"])
def test_per_period_dop_symmetry_off_goal_line(provider):
    actions, frames = _make_mirrored_shot_actions_frames(provider)
    s = pre_shot_gk_angle_off_goal_line(actions, frames)
    assert math.isclose(float(s.iloc[0]), float(s.iloc[1]), abs_tol=1e-9), (
        f"{provider}: per-period DOP-symmetry violated for off_goal_line: P1={s.iloc[0]} vs P2={s.iloc[1]}"
    )


@pytest.mark.parametrize("provider", ["sportec", "metrica"])
def test_density_gate_at_least_one_shot_per_period(provider):
    """Fixture-density gate: invariant must be physically exercised, not vacuously pass.

    Memory: feedback_invariant_testing (PR-S23 corollary).
    """
    actions, _ = _make_mirrored_shot_actions_frames(provider)
    for period in (1, 2):
        n_shots_in_period = int(((actions["period_id"] == period) & (actions["type_id"] == _SHOT_TYPE_ID)).sum())
        assert n_shots_in_period >= 1, f"{provider} P{period}: NO shot rows in fixture"
