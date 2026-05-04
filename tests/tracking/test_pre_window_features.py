"""TF-3 pre-window kernel: arc-length + displacement with NaN-bridge rule.

References (see NOTICE):
- Pure geometric formulation; NOT Bauer & Anzer 2021 covered-distance feature
  (which uses sprint-intensity filtering).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._kernels import _actor_pre_window_kernel
from tests.tracking._provider_inputs import ConstantVelocityActorScenario


def _build_pre_window_input(
    *,
    pre_seconds: float,
    actor_player_id: int,
    frame_xys: list[tuple[float, float, float]],  # (x, y, time_offset_from_action)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_time = 100.0
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [action_time],
            "player_id": [actor_player_id],
        }
    )
    rows = []
    for frame_id, (x, y, dt) in enumerate(frame_xys):
        rows.append(
            {
                "frame_id": frame_id,
                "period_id": 1,
                "time_seconds": action_time + dt,
                "player_id": actor_player_id,
                "is_ball": False,
                "x": x,
                "y": y,
            }
        )
        # Add a ball row per frame (so frames df is realistic shape)
        rows.append(
            {
                "frame_id": frame_id,
                "period_id": 1,
                "time_seconds": action_time + dt,
                "player_id": None,
                "is_ball": True,
                "x": x,
                "y": y,
            }
        )
    frames = pd.DataFrame(rows)
    _ = pre_seconds  # consumed by kernel callers, not by this builder
    return actions, frames


def test_pre_window_stationary_actor() -> None:
    """All frames at (10, 10) -> arc-length = 0, displacement = 0."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5,
        actor_player_id=10,
        frame_xys=[(10.0, 10.0, dt) for dt in [-0.4, -0.3, -0.2, -0.1, 0.0]],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert out.iloc[0]["actor_arc_length_pre_window"] == pytest.approx(0.0, abs=1e-9)
    assert out.iloc[0]["actor_displacement_pre_window"] == pytest.approx(0.0, abs=1e-9)


def test_pre_window_constant_velocity_5ms() -> None:
    """5 m/s along +x for 0.5s -> arc-length ~= 2.5 ~= displacement.

    Uses ConstantVelocityActorScenario named-dataclass per spec section 9.
    """
    scenario = ConstantVelocityActorScenario()
    actions, frames = _build_pre_window_input(
        pre_seconds=scenario.pre_seconds,
        actor_player_id=10,
        frame_xys=[
            (
                scenario.actor_start_pos[0] + scenario.velocity_ms[0] * (dt + scenario.pre_seconds),
                scenario.actor_start_pos[1],
                dt,
            )
            for dt in np.linspace(-scenario.pre_seconds, 0.0, scenario.n_frames)
        ],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=scenario.pre_seconds)
    assert out.iloc[0]["actor_arc_length_pre_window"] == pytest.approx(2.5, rel=1e-6)
    assert out.iloc[0]["actor_displacement_pre_window"] == pytest.approx(2.5, rel=1e-6)


def test_pre_window_circular_path() -> None:
    """Half-circle radius 2 in 0.5s -> arc-length ~= pi*2 ~= 6.28; displacement = 4 (diameter)."""
    n_frames = 13
    angles = np.linspace(0, math.pi, n_frames)
    xys = [(50.0 + 2.0 * math.cos(a), 34.0 + 2.0 * math.sin(a)) for a in angles]
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5,
        actor_player_id=10,
        frame_xys=[(x, y, dt) for (x, y), dt in zip(xys, np.linspace(-0.5, 0.0, n_frames), strict=False)],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    arc = out.iloc[0]["actor_arc_length_pre_window"]
    disp = out.iloc[0]["actor_displacement_pre_window"]
    # 12 secants of pi/12 each ~= 12 * 2 * sin(pi/24) ~= 6.255 (slightly less than full pi*2)
    assert 5.5 <= arc <= 6.5
    assert disp == pytest.approx(4.0, rel=1e-6)
    assert arc > disp  # circular path: arc > displacement always


def test_pre_window_bridge_rule_one_nan() -> None:
    """Frames at -0.4, -0.3, NaN, -0.1, 0.0 with positions (0,0)/(1,0)/NaN/(3,0)/(4,0):
    arc-length = 1 + 2 + 1 = 4 (bridges across NaN); displacement = 4 (first valid 0,0 to last valid 4,0)."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5,
        actor_player_id=10,
        frame_xys=[
            (0.0, 0.0, -0.4),
            (1.0, 0.0, -0.3),
            (float("nan"), float("nan"), -0.2),
            (3.0, 0.0, -0.1),
            (4.0, 0.0, 0.0),
        ],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert out.iloc[0]["actor_arc_length_pre_window"] == pytest.approx(4.0, rel=1e-6)
    assert out.iloc[0]["actor_displacement_pre_window"] == pytest.approx(4.0, rel=1e-6)


def test_pre_window_one_valid_frame_returns_nan() -> None:
    """Only 1 valid-position frame -> NaN for both."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5,
        actor_player_id=10,
        frame_xys=[
            (0.0, 0.0, -0.4),
            (float("nan"), float("nan"), -0.3),
            (float("nan"), float("nan"), -0.2),
        ],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert pd.isna(out.iloc[0]["actor_arc_length_pre_window"])
    assert pd.isna(out.iloc[0]["actor_displacement_pre_window"])


def test_pre_window_no_frames_returns_nan() -> None:
    """Action has no frames in window for actor -> NaN."""
    action_time = 100.0
    actions = pd.DataFrame({"action_id": [1], "period_id": [1], "time_seconds": [action_time], "player_id": [10]})
    frames = pd.DataFrame(
        {
            "frame_id": [0],
            "period_id": [1],
            "time_seconds": [action_time + 5.0],  # outside window
            "player_id": [10],
            "is_ball": [False],
            "x": [50.0],
            "y": [34.0],
        }
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert pd.isna(out.iloc[0]["actor_arc_length_pre_window"])
    assert pd.isna(out.iloc[0]["actor_displacement_pre_window"])


def test_pre_window_shuffled_input_rows() -> None:
    """Kernel MUST sort rows by time_seconds ASC; shuffled input gives same answer (per spec section 3.3)."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5,
        actor_player_id=10,
        frame_xys=[(50.0 + 5.0 * (dt + 0.5), 34.0, dt) for dt in np.linspace(-0.5, 0.0, 13)],
    )
    frames_shuffled = frames.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_sorted = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    out_shuffled = _actor_pre_window_kernel(actions, frames_shuffled, pre_seconds=0.5)
    pd.testing.assert_frame_equal(out_sorted, out_shuffled)
