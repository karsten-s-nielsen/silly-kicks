"""Performance benchmarks for regression detection.

Run with: pytest tests/test_benchmark.py --benchmark-only
"""

import pandas as pd

import silly_kicks.spadl.config as spadlcfg
import silly_kicks.spadl.utils as spu
import silly_kicks.vaep.features as fs


def _make_spadl_actions(n: int = 1000) -> pd.DataFrame:
    """Create a synthetic SPADL DataFrame for benchmarking."""
    return pd.DataFrame(
        {
            "game_id": [1] * n,
            "original_event_id": [str(i) for i in range(n)],
            "action_id": list(range(n)),
            "period_id": [1] * (n // 2) + [2] * (n - n // 2),
            "time_seconds": [float(i) for i in range(n)],
            "team_id": [100 if i % 3 != 0 else 200 for i in range(n)],
            "player_id": [i % 10 + 300 for i in range(n)],
            "start_x": [50.0 + (i % 50) for i in range(n)],
            "start_y": [34.0 + (i % 30) - 15 for i in range(n)],
            "end_x": [55.0 + (i % 50) for i in range(n)],
            "end_y": [34.0 + (i % 30) - 15 for i in range(n)],
            "type_id": [spadlcfg.actiontype_id["pass"]] * (n - n // 10) + [spadlcfg.actiontype_id["shot"]] * (n // 10),
            "result_id": [spadlcfg.result_id["success"]] * (n - n // 20) + [spadlcfg.result_id["fail"]] * (n // 20),
            "bodypart_id": [spadlcfg.bodypart_id["foot"]] * n,
        }
    )


def test_gamestates_benchmark(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Benchmark gamestates computation on 1000 actions."""
    actions = spu.add_names(_make_spadl_actions(1000))
    benchmark(fs.gamestates, actions, 3)


def test_feature_column_names_benchmark(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Benchmark feature column name generation."""
    from silly_kicks.vaep.base import xfns_default

    benchmark(fs.feature_column_names, xfns_default, 3)
