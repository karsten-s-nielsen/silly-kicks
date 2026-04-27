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


def test_add_possessions_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Benchmark ``add_possessions`` on a 1500-action match (typical SPADL match size).

    Design budget per spec: median < 50ms on local dev hardware. Hard CI bound is
    looser (200ms) to absorb shared-runner variance; the catch-quadratic-blowup
    safeguard is the sublinear scaling test below.
    """
    import time as _time

    actions = _make_spadl_actions(1500)
    spu.add_possessions(actions)  # warmup
    start = _time.perf_counter()
    result = spu.add_possessions(actions)
    elapsed = _time.perf_counter() - start
    assert "possession_id" in result.columns
    assert elapsed < 0.2, f"add_possessions(1500) took {elapsed * 1000:.2f}ms, hard CI budget 200ms"
    benchmark(spu.add_possessions, actions)


def test_add_possessions_sublinear_scaling_10k(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """``add_possessions`` on 10k actions must stay sublinear (<2s hard CI bound).

    The vectorised pandas/numpy implementation is O(n); a regression to a
    Python-level row loop would push this benchmark into multi-second territory.
    """
    import time as _time

    actions = _make_spadl_actions(10_000)
    spu.add_possessions(actions)  # warmup
    start = _time.perf_counter()
    result = spu.add_possessions(actions)
    elapsed = _time.perf_counter() - start
    assert "possession_id" in result.columns
    assert elapsed < 2.0, f"add_possessions(10k) took {elapsed * 1000:.2f}ms, hard CI budget 2000ms"
    benchmark(spu.add_possessions, actions)
