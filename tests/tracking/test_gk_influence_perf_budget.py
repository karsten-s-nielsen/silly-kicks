"""Structural performance guard for compute_gk_influence (TF-15).

Replaces a flaky wall-clock budget (the 10ms Linux ceiling once measured 10.4ms on a slow
shared runner) with a deterministic call-count invariant: compute_gk_influence builds the
pitch-control surface ONCE (it then derives every zone/threat metric from that single
surface). A regression to per-zone surface construction is the real cost blow-up. See
tests/_perf_structural.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._perf_structural import call_counter
from tests.tracking._goal_map_helpers import goal_map_for

#: ADR-055 replaced ``home_team_id=1`` at this file's re-keyed call sites. Its frames carry
#: game 1 / period 1 with teams {1, 2} and each keeper at its own end, so this states exactly
#: what ``home_team_id=1`` meant and matches what ``resolve_defended_goals`` derives there.
HOME_GOAL_MAP = goal_map_for({1: 0.0, 2: 105.0})


def _make_22_player_frame():
    """Standard 22-player frame for benchmarking."""
    rows = []
    # Ball
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=0,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            vx=5.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Home team: GK + 10 outfield
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=1,
            team_id=1,
            is_ball=False,
            is_goalkeeper=True,
            x=3.0,
            y=34.0,
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    rng = np.random.default_rng(42)
    for i in range(10):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=10 + i,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(10, 60)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    # Away team: GK + 10 outfield
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=50,
            team_id=2,
            is_ball=False,
            is_goalkeeper=True,
            x=102.0,
            y=34.0,
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    for i in range(10):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=60 + i,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(45, 95)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fixture_22():
    from silly_kicks.xthreat import ExpectedThreat

    frame = _make_22_player_frame()
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return frame, xt


def test_gk_influence_builds_one_pitch_control_surface(fixture_22, monkeypatch):
    """compute_gk_influence builds exactly ONE pitch-control surface for the whole frame."""
    from silly_kicks.tracking import _gk_influence
    from silly_kicks.tracking.pitch_control import _cache

    frame, xt = fixture_22
    calls = call_counter(monkeypatch, _cache, "compute_pitch_control")

    result = _gk_influence.compute_gk_influence(
        frame, attacking_team_id=2, gk_player_id=1, xt=xt, goal_map=HOME_GOAL_MAP
    )

    assert result is not None
    assert calls["n"] == 1, (
        f"compute_gk_influence built {calls['n']} pitch-control surfaces (expected 1). "
        "Every zone/threat metric must derive from the single cached surface, not a per-zone recompute."
    )
