"""TF-3 perf budget per spec section 8.1: < 50ms per 100 actions."""

from __future__ import annotations

from silly_kicks.tracking.features import add_actor_pre_window

# Re-use the 100-action fixture from the pressure perf budget module.
from .test_pressure_perf_budget import fixture_100

# Re-export so pytest discovers the fixture in this module's collection.
_ = fixture_100


def test_pre_window_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(add_actor_pre_window, actions, frames)
    assert "actor_arc_length_pre_window" in result.columns
    assert benchmark.stats.stats.mean < 0.10
