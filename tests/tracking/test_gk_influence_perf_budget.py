"""Performance budget for compute_gk_influence (TF-15).

Uses pytest-benchmark, matching test_pressure_perf_budget.py
and pitch_control/test_perf_budget.py patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# compute_gk_influence runs ~4ms locally; the budget is a generous headroom ceiling that
# still catches a >3x regression. The prior 10ms Linux ceiling was flaky on slow shared CI
# runners (one measured 10.4ms while local is ~4ms), so it is raised to 15ms to match Windows
# -- a wall-clock budget needs headroom for runner variance (it is not a precise SLA).
_BUDGET = 0.015


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


def test_compute_gk_influence_perf_budget(benchmark, fixture_22):
    """compute_gk_influence on 22-player frame within budget."""
    from silly_kicks.tracking._gk_influence import compute_gk_influence

    frame, xt = fixture_22

    result = benchmark(
        compute_gk_influence,
        frame,
        attacking_team_id=2,
        gk_player_id=1,
        xt=xt,
        home_team_id=1,
    )
    assert result is not None
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET, (
            f"compute_gk_influence mean {benchmark.stats.stats.mean * 1000:.1f}ms > budget {_BUDGET * 1000:.0f}ms"
        )
