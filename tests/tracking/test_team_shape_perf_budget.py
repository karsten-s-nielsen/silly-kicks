"""Performance budget for compute_team_shape (TF-31)."""

import numpy as np
import pandas as pd
import pytest

_BUDGET = 0.005


@pytest.fixture
def team_shape_frame():
    rows = []
    rng = np.random.default_rng(42)
    for i in range(10):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=i + 10,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(10, 90)),
                y=float(rng.uniform(5, 63)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
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
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    return pd.DataFrame(rows)


def test_team_shape_perf_budget(benchmark, team_shape_frame):
    from silly_kicks.tracking._team_shape import compute_team_shape

    result = benchmark(compute_team_shape, team_shape_frame, team_id=1)
    assert result is not None
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET, (
            f"compute_team_shape: {benchmark.stats.stats.mean * 1000:.1f}ms > {_BUDGET * 1000:.0f}ms"
        )
