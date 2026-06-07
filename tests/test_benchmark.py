"""Performance benchmarks for regression detection.

Run with: pytest tests/test_benchmark.py --benchmark-only
"""

import pandas as pd
import pytest

import silly_kicks.atomic.spadl as atomicspadl
import silly_kicks.atomic.spadl.config as atomicspadlcfg
import silly_kicks.atomic.spadl.utils as atomicspu
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
    """Benchmark ``add_possessions`` on a 1500-action match (pure measurement).

    Regression protection is the deterministic ``test_throughput_converters_stay_vectorised``
    guard below — the old wall-clock ``elapsed < 200ms`` ceiling flaked on shared CI runners.
    """
    actions = _make_spadl_actions(1500)
    result = spu.add_possessions(actions)
    assert "possession_id" in result.columns
    benchmark(spu.add_possessions, actions)


def test_add_possessions_benchmark_10k(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Benchmark ``add_possessions`` on 10k actions (pure measurement at scale).

    The O(n)-vs-row-loop guarantee the old ``< 2s`` ceiling proxied is now enforced
    deterministically by ``test_throughput_converters_stay_vectorised``.
    """
    actions = _make_spadl_actions(10_000)
    result = spu.add_possessions(actions)
    assert "possession_id" in result.columns
    benchmark(spu.add_possessions, actions)


def _make_spadl_actions_with_gk(n: int = 1500) -> pd.DataFrame:
    """1500-action match with ~5% keeper actions and ~5% shots — realistic mix
    for benchmarking the GK helpers under near-production load.
    """
    actions = _make_spadl_actions(n)
    # Inject ~75 keeper actions (every 20th action) by team 100, player 999.
    keeper_idx = list(range(0, n, 20))
    for i in keeper_idx:
        actions.loc[i, "team_id"] = 100
        actions.loc[i, "player_id"] = 999
        actions.loc[i, "type_id"] = spadlcfg.actiontype_id["keeper_save"]
        actions.loc[i, "start_x"] = 5.0
        actions.loc[i, "start_y"] = 34.0
    # Inject ~75 shots (every 20th action offset by 10) by team 200, attacker 700.
    shot_idx = list(range(10, n, 20))
    for i in shot_idx:
        actions.loc[i, "team_id"] = 200
        actions.loc[i, "player_id"] = 700
        actions.loc[i, "type_id"] = spadlcfg.actiontype_id["shot"]
        actions.loc[i, "start_x"] = 95.0
    return actions


def test_add_gk_role_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """``add_gk_role`` on a 1500-action match (Q8 budget < 50ms median; CI hard 200ms)."""
    actions = _make_spadl_actions_with_gk(1500)
    result = spu.add_gk_role(actions)
    assert "gk_role" in result.columns
    benchmark(spu.add_gk_role, actions)


def test_add_gk_distribution_metrics_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """``add_gk_distribution_metrics`` (without xT grid) on 1500 actions (CI hard 200ms)."""
    actions = _make_spadl_actions_with_gk(1500)
    result = spu.add_gk_distribution_metrics(actions)
    assert "gk_pass_length_m" in result.columns
    benchmark(spu.add_gk_distribution_metrics, actions)


def test_add_pre_shot_gk_context_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """``add_pre_shot_gk_context`` on 1500 actions with ~75 shots (pure measurement).

    Per-shot lookback loops over a numpy shot-index array (~75 shots), not the full action
    frame — so the stays-vectorised guard (no DataFrame row iteration) still holds.
    """
    actions = _make_spadl_actions_with_gk(1500)
    result = spu.add_pre_shot_gk_context(actions)
    assert "gk_was_engaged" in result.columns
    benchmark(spu.add_pre_shot_gk_context, actions)


# ---------------------------------------------------------------------------
# Atomic-SPADL helper benchmarks (PR-S5, silly-kicks 1.5.0)
# ---------------------------------------------------------------------------


def _make_atomic_actions(n: int = 1500) -> pd.DataFrame:
    """Synthetic Atomic-SPADL frame for benchmarking.

    Built directly in atomic shape (x/y/dx/dy, no result_id) rather than via
    convert_to_atomic to keep the helper benchmark independent of conversion
    cost. Mirrors the structure of ``_make_spadl_actions``: alternating teams,
    period split halfway through, ~5% keeper actions, ~5% shots, scattered
    receival follow-ups so the distribution-success-detection path is hit.
    """
    actions = pd.DataFrame(
        {
            "game_id": [1] * n,
            "original_event_id": [str(i) for i in range(n)],
            "action_id": list(range(n)),
            "period_id": [1] * (n // 2) + [2] * (n - n // 2),
            "time_seconds": [float(i) for i in range(n)],
            "team_id": [100 if i % 3 != 0 else 200 for i in range(n)],
            "player_id": [i % 10 + 300 for i in range(n)],
            "x": [50.0 + (i % 50) for i in range(n)],
            "y": [34.0 + (i % 30) - 15 for i in range(n)],
            "dx": [5.0 + (i % 10) for i in range(n)],
            "dy": [(i % 5) - 2 for i in range(n)],
            "type_id": [atomicspadlcfg.actiontype_id["pass"]] * n,
            "bodypart_id": [atomicspadlcfg.bodypart_id["foot"]] * n,
        }
    )
    # Inject ~75 keeper actions at every 20th index by team 100, player 999.
    keeper_idx = list(range(0, n, 20))
    for i in keeper_idx:
        actions.loc[i, "team_id"] = 100
        actions.loc[i, "player_id"] = 999
        actions.loc[i, "type_id"] = atomicspadlcfg.actiontype_id["keeper_save"]
        actions.loc[i, "x"] = 5.0
        actions.loc[i, "y"] = 34.0
    # Inject ~75 shots at every 20th offset by 10 by team 200, attacker 700.
    shot_idx = list(range(10, n, 20))
    for i in shot_idx:
        actions.loc[i, "team_id"] = 200
        actions.loc[i, "player_id"] = 700
        actions.loc[i, "type_id"] = atomicspadlcfg.actiontype_id["shot"]
        actions.loc[i, "x"] = 95.0
    # Inject receival follow-ups every 5 actions to exercise pass-success detection
    # in add_gk_distribution_metrics.
    receival_idx = list(range(2, n, 5))
    for i in receival_idx:
        actions.loc[i, "type_id"] = atomicspadlcfg.actiontype_id["receival"]
    return actions


def test_atomic_convert_to_atomic_preserve_native_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """``convert_to_atomic(preserve_native=...)`` on a 1500-action SPADL match (pure measurement)."""
    actions = _make_spadl_actions(1500)
    actions["my_extra"] = 1
    result = atomicspadl.convert_to_atomic(actions, preserve_native=["my_extra"])
    assert "my_extra" in result.columns
    benchmark(atomicspadl.convert_to_atomic, actions, preserve_native=["my_extra"])


def test_atomic_add_possessions_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Atomic ``add_possessions`` on a 1500-action match (CI hard 200ms)."""
    actions = _make_atomic_actions(1500)
    result = atomicspu.add_possessions(actions)
    assert "possession_id" in result.columns
    benchmark(atomicspu.add_possessions, actions)


def test_atomic_add_gk_role_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Atomic ``add_gk_role`` on a 1500-action match (CI hard 200ms)."""
    actions = _make_atomic_actions(1500)
    result = atomicspu.add_gk_role(actions)
    assert "gk_role" in result.columns
    benchmark(atomicspu.add_gk_role, actions)


def test_atomic_add_gk_distribution_metrics_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Atomic ``add_gk_distribution_metrics`` (without xT grid) on 1500 actions (CI hard 200ms)."""
    actions = _make_atomic_actions(1500)
    result = atomicspu.add_gk_distribution_metrics(actions)
    assert "gk_pass_length_m" in result.columns
    benchmark(atomicspu.add_gk_distribution_metrics, actions)


def test_atomic_add_pre_shot_gk_context_benchmark_1500(benchmark: "BenchmarkFixture") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Atomic ``add_pre_shot_gk_context`` on 1500 actions (pure measurement)."""
    actions = _make_atomic_actions(1500)
    result = atomicspu.add_pre_shot_gk_context(actions)
    assert "gk_was_engaged" in result.columns
    benchmark(atomicspu.add_pre_shot_gk_context, actions)


# ---------------------------------------------------------------------------
# Structural throughput guard (replaces the per-test wall-clock `elapsed < X` ceilings,
# which flaked on shared CI runners). The blow-up those budgets actually guarded against is
# de-vectorisation — a converter regressing from the O(n) np.select path into a Python row
# loop. We assert that deterministically: each converter does ZERO pandas row-wise iteration
# (apply(axis=1) / iterrows / itertuples). See tests/_perf_structural.py.
# ---------------------------------------------------------------------------


def _converter_cases():
    """(label, callable) for every throughput converter the old wall-clock budgets covered."""

    return [
        ("spadl.add_possessions", lambda: spu.add_possessions(_make_spadl_actions(1500))),
        ("spadl.add_gk_role", lambda: spu.add_gk_role(_make_spadl_actions_with_gk(1500))),
        (
            "spadl.add_gk_distribution_metrics",
            lambda: spu.add_gk_distribution_metrics(_make_spadl_actions_with_gk(1500)),
        ),
        ("spadl.add_pre_shot_gk_context", lambda: spu.add_pre_shot_gk_context(_make_spadl_actions_with_gk(1500))),
        (
            "convert_to_atomic",
            lambda: atomicspadl.convert_to_atomic(
                _make_spadl_actions(1500).assign(my_extra=1), preserve_native=["my_extra"]
            ),
        ),
        ("atomic.add_possessions", lambda: atomicspu.add_possessions(_make_atomic_actions(1500))),
        ("atomic.add_gk_role", lambda: atomicspu.add_gk_role(_make_atomic_actions(1500))),
        (
            "atomic.add_gk_distribution_metrics",
            lambda: atomicspu.add_gk_distribution_metrics(_make_atomic_actions(1500)),
        ),
        ("atomic.add_pre_shot_gk_context", lambda: atomicspu.add_pre_shot_gk_context(_make_atomic_actions(1500))),
    ]


@pytest.mark.parametrize("label,run", _converter_cases(), ids=[c[0] for c in _converter_cases()])
def test_throughput_converters_stay_vectorised(label, run, monkeypatch) -> None:
    """Each throughput converter stays vectorised: zero pandas row-wise iteration on a 1500-action
    match. A regression to a Python row loop (apply(axis=1) / iterrows / itertuples) is the O(n)
    blow-up the old wall-clock budgets proxied — caught here deterministically, runner-independent."""
    from tests._perf_structural import row_iteration_counter

    calls = row_iteration_counter(monkeypatch)
    run()
    assert calls["n"] == 0, (
        f"{label} performed {calls['n']} pandas row-wise iteration(s) on 1500 actions "
        "(apply(axis=1)/iterrows/itertuples) — it de-vectorised. Restore the np.select / vectorised path."
    )
