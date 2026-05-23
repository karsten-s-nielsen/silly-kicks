# tests/tracking/test_player_influence_perf_budget.py
"""Performance budget for compute_player_influence (TF-36 + TF-33).

Uses pytest-benchmark. Budget set from first CI observation + 1.5x headroom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Flat ceiling — no platform ternary per feedback_windows_ci_perf_budget.md.
# FIRST RUN: set from worst observed CI timing with 1.5x headroom.
_BUDGET = 0.100  # 100ms — generous initial budget, tighten after first CI run


def _make_22_player_frame():
    """Standard 22-player frame for benchmarking."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
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
    for gk_pid, gk_tid, gk_x in [(1, 1, 3.0), (50, 2, 102.0)]:
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=gk_pid,
                team_id=gk_tid,
                is_ball=False,
                is_goalkeeper=True,
                x=gk_x,
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


def test_compute_player_influence_perf_budget(benchmark, fixture_22):
    """compute_player_influence on 22-player frame within budget."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    frame, xt = fixture_22

    result = benchmark(
        compute_player_influence,
        frame,
        xt,
        attacking_team_id=1,
        home_team_id=1,
    )
    assert result is not None
    assert len(result) == 20  # 20 outfield players
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET, (
            f"compute_player_influence mean {benchmark.stats.stats.mean * 1000:.1f}ms > budget {_BUDGET * 1000:.0f}ms"
        )
