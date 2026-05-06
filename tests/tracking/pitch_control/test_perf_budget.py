"""pytest-benchmark gates per spec section 10.6 performance budget.

Single-frame 22-player pitch control must complete within 50ms (Linux) / 75ms (Windows).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import (
    compute_pitch_control,
)

_BUDGET = 0.05 if sys.platform != "win32" else 0.075  # seconds


@pytest.fixture(scope="module")
def full_frame_22():
    """Realistic 22-player frame (11v11 + ball)."""
    rng = np.random.default_rng(777)
    rows = []
    for i in range(11):
        rows.append(
            {
                "player_id": 100 + i,
                "team_id": 1,
                "x": rng.uniform(5, 100),
                "y": rng.uniform(5, 63),
                "vx": rng.uniform(-3, 3),
                "vy": rng.uniform(-3, 3),
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    for i in range(11):
        rows.append(
            {
                "player_id": 200 + i,
                "team_id": 2,
                "x": rng.uniform(5, 100),
                "y": rng.uniform(5, 63),
                "vx": rng.uniform(-3, 3),
                "vy": rng.uniform(-3, 3),
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    rows.append(
        {
            "player_id": np.nan,
            "team_id": np.nan,
            "x": 52.5,
            "y": 34.0,
            "vx": 0,
            "vy": 0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


def test_spearman_single_frame_budget(benchmark, full_frame_22) -> None:
    result = benchmark(compute_pitch_control, full_frame_22, 1, method="spearman")
    assert result.surface.shape[0] > 0
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET


def test_fernandez_bornn_single_frame_budget(benchmark, full_frame_22) -> None:
    result = benchmark(compute_pitch_control, full_frame_22, 1, method="fernandez_bornn")
    assert result.surface.shape[0] > 0
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET


def test_voronoi_single_frame_budget(benchmark, full_frame_22) -> None:
    result = benchmark(compute_pitch_control, full_frame_22, 1, method="voronoi")
    assert result.surface.shape[0] > 0
    if benchmark.stats is not None:
        assert benchmark.stats.stats.mean < _BUDGET * 0.5  # Voronoi is trivially fast
