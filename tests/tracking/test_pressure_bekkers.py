"""Bekkers 2024 Pressing Intensity kernel (probabilistic time-to-intercept).

References (see NOTICE):
- Bekkers, J. (2025), arXiv:2501.04712.
- BSD-3-Clause: UnravelSports/unravelsports unravel/soccer/models/utils.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._kernels import (
    _bekkers_p_intercept,
    _bekkers_tti,
    _pressure_bekkers,
)
from silly_kicks.tracking.pressure import BekkersParams


def test_bekkers_tti_stationary_defender_zero_distance() -> None:
    """Defender at d=0, all velocities zero -> tti = 0 + tau_r + 0/v_max = tau_r = 0.7s."""
    p1 = np.array([[50.0, 34.0]])
    p2 = np.array([[50.0, 34.0]])
    v1 = np.array([[0.0, 0.0]])
    v2 = np.array([[0.0, 0.0]])
    tti = _bekkers_tti(p1=p1, p2=p2, v1=v1, v2=v2, reaction_time=0.7, max_object_speed=12.0)
    assert tti.shape == (1, 1)
    assert tti[0, 0] == pytest.approx(0.7, rel=1e-9)


def test_bekkers_tti_stationary_d10() -> None:
    """Defender at d=10, all v=0 -> tti = 0 + 0.7 + 10/12 = 0.7 + 0.8333... = 1.5333..."""
    p1 = np.array([[50.0, 34.0]])
    p2 = np.array([[60.0, 34.0]])
    v1 = np.array([[0.0, 0.0]])
    v2 = np.array([[0.0, 0.0]])
    tti = _bekkers_tti(p1=p1, p2=p2, v1=v1, v2=v2, reaction_time=0.7, max_object_speed=12.0)
    assert tti[0, 0] == pytest.approx(0.7 + 10.0 / 12.0, rel=1e-9)


def test_bekkers_p_intercept_at_threshold() -> None:
    """At tti = T_threshold, p = 0.5."""
    tti = np.array([[1.5]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    assert p[0, 0] == pytest.approx(0.5, rel=1e-9)


def test_bekkers_p_intercept_below_threshold() -> None:
    """tti=0.7 < T=1.5, sigma=0.45 -> p > 0.5 (close defender, high pressure)."""
    tti = np.array([[0.7]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    expected = 1.0 / (1.0 + math.exp(-math.pi / math.sqrt(3.0) / 0.45 * (1.5 - 0.7)))
    assert p[0, 0] == pytest.approx(expected, rel=1e-9)
    assert p[0, 0] > 0.5


def test_bekkers_p_intercept_above_threshold() -> None:
    """tti=3.0 > T=1.5 -> p < 0.5 (distant defender, low pressure)."""
    tti = np.array([[3.0]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    assert p[0, 0] < 0.5


def test_bekkers_p_intercept_clipping_avoids_overflow() -> None:
    """Extreme tti values shouldn't crash on np.exp overflow (per canonical clip)."""
    tti = np.array([[1e6]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    assert 0.0 <= p[0, 0] < 1.0  # very low but valid


def _build_ctx_with_velocities_and_ball(
    *,
    actor_xy: tuple[float, float],
    actor_vxvy: tuple[float, float] = (0.0, 0.0),
    defenders: list[tuple[float, float, float, float]],  # (x, y, vx, vy)
    ball_xyvxvy: tuple[float, float, float, float] | None = (50.0, 34.0, 0.0, 0.0),
):
    """Build ActionFrameContext + ball-rows-per-action mapping for Bekkers tests."""
    from silly_kicks.tracking.feature_framework import ActionFrameContext

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
    actor_rows = pd.DataFrame(
        {
            "action_id": [1],
            "x": [actor_xy[0]],
            "y": [actor_xy[1]],
            "vx": [actor_vxvy[0]],
            "vy": [actor_vxvy[1]],
            "speed": [math.hypot(*actor_vxvy)],
        }
    )
    defender_rows = pd.DataFrame(
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
                "vx": dvx,
                "vy": dvy,
                "speed": math.hypot(dvx, dvy),
            }
            for i, (dx, dy, dvx, dvy) in enumerate(defenders)
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
    ctx = ActionFrameContext(
        actions=actions,
        pointers=pointers,
        actor_rows=actor_rows,
        opposite_rows_per_action=defender_rows,
        defending_gk_rows=pd.DataFrame(),
    )
    if ball_xyvxvy is not None:
        ball_per_action = pd.DataFrame(
            [
                {
                    "action_id": 1,
                    "x": ball_xyvxvy[0],
                    "y": ball_xyvxvy[1],
                    "vx": ball_xyvxvy[2],
                    "vy": ball_xyvxvy[3],
                }
            ]
        )
    else:
        ball_per_action = pd.DataFrame(columns=["action_id", "x", "y", "vx", "vy"])
    return actions["start_x"], actions["start_y"], ctx, ball_per_action


def test_bekkers_kernel_stationary_defender_at_d_zero() -> None:
    """Defender at d=0 with v=0; actor v=0; ball=actor; speed_threshold=2.0 (filtered)."""
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(50.0, 34.0, 0.0, 0.0)],
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    # Defender speed=0 < threshold=2.0, so p_to_player=0; ball at d=0 also stationary, p=0
    # 1 - (1 - 0) = 0
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_bekkers_kernel_active_defender_at_d_zero() -> None:
    """Defender at d=0 with speed=3 m/s above threshold -> active-pressing path engaged.

    Loose-bound pin (range, not exact value). Per lakehouse v3 review item 6:
    exact-value pinning for Bekkers belongs in the golden-master parity test
    (bit-equivalent rtol=1e-9 vs UnravelSports canonical) and snapshot
    determinism gate (SHA-256 regression). This test's job is to verify that
    the active-pressing speed filter does NOT zero out a defender above
    threshold (regression guard against active-pressing logic bug); exact
    value drift is caught by the dedicated gates.
    """
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(50.0, 34.0, 3.0, 0.0)],
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    # Active-pressing filter does NOT fire (speed=3 > threshold=2.0), so p > 0.
    # Bounded in [0, 1] by aggregation construction.
    assert out.iloc[0] > 0.0
    assert 0.0 <= out.iloc[0] <= 1.0


def test_bekkers_kernel_two_defenders_aggregate() -> None:
    """Two defenders each above speed threshold -> aggregate in [0,1]."""
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(48.0, 34.0, 3.0, 0.0), (52.0, 34.0, -3.0, 0.0)],  # both moving toward actor
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    assert 0.0 <= out.iloc[0] <= 1.0


def test_bekkers_kernel_speed_threshold_filters() -> None:
    """Defender below speed threshold -> p=0 -> aggregate 0 (only this defender)."""
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(51.0, 34.0, 1.0, 0.0)],  # speed=1 < 2.0
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_bekkers_kernel_use_ball_carrier_max_strictly_higher_when_ball_closer() -> None:
    """When ball is closer to defender than actor, max(p_to_ball, p_to_player)
    must produce STRICTLY higher pressure than p_to_player alone.

    Lakehouse v3 review item 5: original test asserted >=, which is tautological
    (max(a,b) >= a always). Strict-greater assertion verifies the ball-comparison
    path actually fires and increases pressure when ball is closer.

    Construct: defender at (60, 34) v=(-3, 0) moving toward both actor AND ball
    along the -x axis. Actor at (50, 34) is 10 m from defender; ball at (52, 34)
    is only 8 m from defender. Defender's velocity is aligned with both targets
    (no angle penalty), so smaller spatial distance translates directly to lower
    tti and higher p_to_ball than p_to_actor.
    """
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(60.0, 34.0, -3.0, 0.0)],
        ball_xyvxvy=(52.0, 34.0, 0.0, 0.0),
    )
    out_with_max = _pressure_bekkers(
        ax,
        ay,
        ctx,
        params=BekkersParams(use_ball_carrier_max=True),
        ball_xy_v_per_action=ball,
    )
    out_no_max = _pressure_bekkers(
        ax,
        ay,
        ctx,
        params=BekkersParams(use_ball_carrier_max=False),
        ball_xy_v_per_action=ball,
    )
    assert out_with_max.iloc[0] > out_no_max.iloc[0] + 1e-6, (
        f"ball-carrier-max must produce STRICTLY higher pressure when ball is closer "
        f"than actor; got with_max={out_with_max.iloc[0]:.6f} vs no_max={out_no_max.iloc[0]:.6f}"
    )


@pytest.mark.parametrize("use_ball_carrier_max", [True, False])
def test_bekkers_kernel_zero_defenders(use_ball_carrier_max: bool) -> None:
    """Linked, no defenders -> 0.0 (NOT NaN). Lakehouse v3 review item 4:
    parametrized over both use_ball_carrier_max settings to catch regressions
    in the optional ball-comparison path.
    """
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[],
    )
    out = _pressure_bekkers(
        ax,
        ay,
        ctx,
        params=BekkersParams(use_ball_carrier_max=use_ball_carrier_max),
        ball_xy_v_per_action=ball,
    )
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_bekkers_no_ball_rows_anywhere_raises_value_error() -> None:
    """Hard-fail per spec section 4.5 + section 11 risk row (lakehouse v2 review item 9):
    use_ball_carrier_max=True with ZERO ball rows in entire frames -> ValueError.

    Without this test, a future refactor could silently revert the load-bearing
    decision against the silent UserWarning fallback.
    """
    from silly_kicks.tracking.features import pressure_on_actor

    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [50.0],
            "start_y": [34.0],
            "type_id": [0],
        }
    )
    # Frames with NO is_ball=True rows
    frames = pd.DataFrame(
        [
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "home",
                "player_id": 10,
                "is_ball": False,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "away",
                "player_id": 100,
                "is_ball": False,
                "x": 52.0,
                "y": 34.0,
                "vx": 3.0,
                "vy": 0.0,
                "speed": 3.0,
                "source_provider": "synthetic",
            },
        ]
    )
    with pytest.raises(ValueError, match="missing is_ball=True rows"):
        pressure_on_actor(actions, frames, method="bekkers_pi")


def test_bekkers_no_ball_rows_with_opt_out_succeeds() -> None:
    """Opt-out path: use_ball_carrier_max=False bypasses the hard-fail.

    Caller chose to compute pressure-on-player only; should succeed even
    without ball rows.
    """
    from silly_kicks.tracking.features import pressure_on_actor

    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [50.0],
            "start_y": [34.0],
            "type_id": [0],
        }
    )
    frames = pd.DataFrame(
        [
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "home",
                "player_id": 10,
                "is_ball": False,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "away",
                "player_id": 100,
                "is_ball": False,
                "x": 52.0,
                "y": 34.0,
                "vx": -3.0,
                "vy": 0.0,
                "speed": 3.0,
                "source_provider": "synthetic",
            },
        ]
    )
    # Opt out -- should NOT raise
    result = pressure_on_actor(
        actions,
        frames,
        method="bekkers_pi",
        params=BekkersParams(use_ball_carrier_max=False),
    )
    assert result.notna().any()


def test_bekkers_per_action_ball_row_absence_emits_nan() -> None:
    """Per-action NaN per spec section 4.3 / section 4.5 (lakehouse v3 review item 3).

    Some actions link to ball-present frames, others to ball-absent frames.
    Ball-absent actions should emit NaN; ball-present actions compute normally.
    Two actions, two distinct frames; only frame 1 has a ball row.
    """
    from silly_kicks.tracking.features import pressure_on_actor

    actions = pd.DataFrame(
        {
            "action_id": [1, 2],
            "period_id": [1, 1],
            "time_seconds": [0.0, 1.0],
            "team_id": ["home", "home"],
            "player_id": [10, 11],
            "start_x": [50.0, 50.0],
            "start_y": [34.0, 34.0],
            "type_id": [0, 0],
        }
    )
    frames = pd.DataFrame(
        [
            # Action 1 frame: ball PRESENT
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "home",
                "player_id": 10,
                "is_ball": False,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": None,
                "player_id": None,
                "is_ball": True,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 1,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": "away",
                "player_id": 100,
                "is_ball": False,
                "x": 52.0,
                "y": 34.0,
                "vx": -3.0,
                "vy": 0.0,
                "speed": 3.0,
                "source_provider": "synthetic",
            },
            # Action 2 frame: ball ABSENT
            {
                "frame_id": 2,
                "period_id": 1,
                "time_seconds": 1.0,
                "team_id": "home",
                "player_id": 11,
                "is_ball": False,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            },
            {
                "frame_id": 2,
                "period_id": 1,
                "time_seconds": 1.0,
                "team_id": "away",
                "player_id": 101,
                "is_ball": False,
                "x": 52.0,
                "y": 34.0,
                "vx": -3.0,
                "vy": 0.0,
                "speed": 3.0,
                "source_provider": "synthetic",
            },
        ]
    )
    # frames does have AT LEAST ONE ball row (action 1's), so the entire-frames
    # hard-fail does NOT trigger; per-action NaN handling kicks in for action 2.
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    assert pd.notna(result.iloc[0]), "action 1 should compute (ball row present at its frame)"
    assert pd.isna(result.iloc[1]), "action 2 should be NaN (no ball row at its frame)"
