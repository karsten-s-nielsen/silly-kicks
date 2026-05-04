"""Andrienko 2017 oval-pressure kernel: 4-point Theta pin tests + sum-aggregation.

References (see NOTICE):
- Andrienko et al. (2017), DMKD 31:1793-1839.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from silly_kicks.tracking._kernels import _pressure_andrienko
from silly_kicks.tracking.feature_framework import ActionFrameContext
from silly_kicks.tracking.pressure import AndrienkoParams


def _build_ctx(
    *,
    actor_xy: tuple[float, float],
    defenders_xy: list[tuple[float, float]],
) -> tuple[pd.Series, pd.Series, ActionFrameContext]:
    """Minimal ActionFrameContext for 1-action / N-defender pressure tests."""
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [actor_xy[0]],
            "start_y": [actor_xy[1]],
        }
    )
    defenders = pd.DataFrame(
        [
            {
                "action_id": 1,
                "team_id_action": "home",
                "team_id_frame": "away",
                "player_id_action": 10,
                "player_id_frame": 100 + i,
                "is_ball": False,
                "x": dx,
                "y": dy,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
            }
            for i, (dx, dy) in enumerate(defenders_xy)
        ]
    )
    pointers = pd.DataFrame(
        {
            "action_id": [1],
            "frame_id": pd.array([1000], dtype="Int64"),
            "time_offset_seconds": [0.0],
            "n_candidate_frames": [1],
            "link_quality_score": [1.0],
        }
    )
    actor_rows = pd.DataFrame(
        {
            "action_id": [1],
            "x": [actor_xy[0]],
            "y": [actor_xy[1]],
            "speed": [0.0],
        }
    )
    ctx = ActionFrameContext(
        actions=actions,
        pointers=pointers,
        actor_rows=actor_rows,
        opposite_rows_per_action=defenders,
        defending_gk_rows=pd.DataFrame(),
    )
    return actions["start_x"], actions["start_y"], ctx


def _expected_andrienko_pr(*, d: float, theta_deg: float, q: float = 1.75) -> float:
    """Hand-computed Andrienko per-defender pressure (verified vs Gemini PDF readout)."""
    cos_theta = math.cos(math.radians(theta_deg))
    z = (1.0 + cos_theta) / 2.0
    L = 3.0 + (9.0 - 3.0) * (z**3 + 0.3 * z) / 1.3
    if d >= L:
        return 0.0
    return (1.0 - d / L) ** q * 100.0


def test_andrienko_l_at_theta_0() -> None:
    """At Theta=0 (presser between target and threat), L = 9.00 m. Defender at d=9 boundary -> Pr=0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0 + 9.0, 34.0)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_l_at_theta_90() -> None:
    """At Theta=+/-90 (presser to side), L = 4.27 m. Defender at d=4.27 boundary -> Pr=0."""
    expected_L = 3.0 + 6.0 * (0.5**3 + 0.3 * 0.5) / 1.3  # = 4.2692...
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(50.0, 34.0 + expected_L)],
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_l_at_theta_180() -> None:
    """At Theta=180 (presser behind target, away from threat), L = 3.00 m. Defender at d=3 boundary -> Pr=0."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(50.0 - 3.0, 34.0)],
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_pr_inside_zone_theta_0() -> None:
    """Defender at d=4.5 (half of L=9) at Theta=0 -> Pr = (1-0.5)**1.75 * 100 ~= 29.730%."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(54.5, 34.0)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    expected = _expected_andrienko_pr(d=4.5, theta_deg=0.0)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)
    assert out.iloc[0] == pytest.approx(29.7301779, rel=1e-6)  # double-check vs hand-computed


def test_andrienko_pr_at_theta_45() -> None:
    """At Theta=45deg: L = 7.05; defender at d=L/2 = 3.525 -> Pr = (0.5)^1.75 * 100 ~= 29.730%."""
    L_45 = (
        3.0 + 6.0 * (((1 + math.cos(math.radians(45))) / 2) ** 3 + 0.3 * ((1 + math.cos(math.radians(45))) / 2)) / 1.3
    )
    d = L_45 / 2
    angle = math.radians(45)
    defender_x = 50.0 + d * math.cos(angle)
    defender_y = 34.0 + d * math.sin(angle)
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(defender_x, defender_y)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    expected = _expected_andrienko_pr(d=d, theta_deg=45.0)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_andrienko_sum_aggregation() -> None:
    """Three pressers each at d=L/2 at Theta=0 -> total Pr should exceed any single contribution."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(54.5, 34.0), (54.5, 35.0), (54.5, 33.0)],
        # Note: only the first is Theta=0; others are slightly off-axis.
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    # Loose check: total pressure > sum of any single contribution (lower bound).
    assert out.iloc[0] > 30.0


def test_andrienko_zero_outside_all_zones() -> None:
    """Defender at d=10m in front -> beyond L=9 in any direction -> Pr=0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(60.0, 34.0)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_zero_no_defenders_returns_zero_not_nan() -> None:
    """Linked frame with empty opposite_rows -> 0.0 (not NaN)."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_monotone_in_distance_at_fixed_angle() -> None:
    """Pressure decreases monotonically as defender->actor distance increases (Theta=0)."""
    distances = [1.0, 2.0, 3.0, 5.0, 7.0]
    actor_xy = (50.0, 34.0)
    pressures: list[float] = []
    for d in distances:
        ax, ay, ctx = _build_ctx(actor_xy=actor_xy, defenders_xy=[(50.0 + d, 34.0)])
        out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
        pressures.append(float(out.iloc[0]))
    for i in range(1, len(pressures)):
        assert pressures[i] <= pressures[i - 1]


def test_andrienko_axially_symmetric() -> None:
    """Defender at +y vs -y at same distance -> equal pressure."""
    ax_pos, ay_pos, ctx_pos = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 + 3.0)])
    ax_neg, ay_neg, ctx_neg = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 - 3.0)])
    p_pos = _pressure_andrienko(ax_pos, ay_pos, ctx_pos, params=AndrienkoParams()).iloc[0]
    p_neg = _pressure_andrienko(ax_neg, ay_neg, ctx_neg, params=AndrienkoParams()).iloc[0]
    assert p_pos == pytest.approx(p_neg, rel=1e-9)


def test_andrienko_non_negative() -> None:
    """Pressure is never negative for any geometry."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(60.0, 50.0), (40.0, 20.0), (50.0, 35.0), (49.0, 33.0)],
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] >= 0.0
