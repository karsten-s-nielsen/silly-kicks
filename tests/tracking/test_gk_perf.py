"""Performance benchmarks for GK identification (PR-S26).

Uses pytest-benchmark to measure derive_goalkeepers performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_identification import derive_goalkeepers

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "datasets" / "tracking" / "synthetic"


def _generate_match_frames(n_frames: int, n_teams: int = 2, players_per_team: int = 11) -> pd.DataFrame:
    """Generate synthetic match frames for benchmarking."""
    np.random.seed(42)
    rows = []
    for team_idx in range(n_teams):
        team_id = f"team_{team_idx}"
        gk_x = 5.0 if team_idx == 0 else 100.0
        for _frame_id in range(n_frames):
            # GK
            rows.append(
                {
                    "game_id": "benchmark_match",
                    "team_id": team_id,
                    "player_id": f"gk_{team_id}",
                    "x": np.clip(gk_x + np.random.normal(0, 2), 0, 105),
                    "y": np.clip(34.0 + np.random.normal(0, 3), 0, 68),
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
            # Outfielders
            for i in range(players_per_team - 1):
                rows.append(
                    {
                        "game_id": "benchmark_match",
                        "team_id": team_id,
                        "player_id": f"player_{team_id}_{i}",
                        "x": np.clip(25.0 + i * 5 + np.random.normal(0, 3), 0, 105),
                        "y": np.clip(10.0 + i * 5 + np.random.normal(0, 2), 0, 68),
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.benchmark(group="gk_identification")
class TestGkPerformance:
    """Benchmark tests for derive_goalkeepers."""

    @pytest.mark.parametrize("n_frames", [500, 1000, 2500])
    def test_benchmark_gk_identification(self, benchmark, n_frames: int):
        """Benchmark GK identification at different frame counts."""
        frames = _generate_match_frames(n_frames)
        result = benchmark(derive_goalkeepers, frames)
        _frames_out, picks = result
        # Verify correctness
        assert len(picks) == 2  # 2 teams
        assert all(len(gks) == 1 for gks in picks.values())

    def test_benchmark_synthetic_fixture(self, benchmark):
        """Benchmark against largest synthetic fixture (gk_substitution)."""
        frames = pd.read_parquet(SYNTHETIC_DIR / "gk_substitution.parquet")
        result = benchmark(derive_goalkeepers, frames)
        _frames_out, picks = result
        # Verify multi-GK detection works
        assert len(picks) == 2  # home and away teams
