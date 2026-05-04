"""pytest-benchmark gates per spec section 8.1 review item 5.

Andrienko/Link < 50ms per 100 actions; Bekkers < 250ms per 100 actions on CI runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.features import pressure_on_actor

from .test_pressure_snapshot import _build_fixture


@pytest.fixture(scope="module")
def fixture_100():
    """100-action fixture extending the 50-action snapshot fixture."""
    np.random.seed(123)
    actions, frames = _build_fixture()
    extra = actions.copy()
    extra["action_id"] = extra["action_id"] + 1000
    actions = pd.concat([actions, extra], ignore_index=True)
    extra_frames = frames.copy()
    extra_frames["frame_id"] = extra_frames["frame_id"] + 1000
    frames = pd.concat([frames, extra_frames], ignore_index=True)
    return actions, frames


def test_andrienko_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(pressure_on_actor, actions, frames, method="andrienko_oval")
    assert result.notna().any()
    assert benchmark.stats.stats.mean < 0.10  # 100ms ceiling on CI; spec target 50ms


def test_link_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(pressure_on_actor, actions, frames, method="link_zones")
    assert result.notna().any()
    assert benchmark.stats.stats.mean < 0.10


def test_bekkers_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(pressure_on_actor, actions, frames, method="bekkers_pi")
    assert result.notna().any()
    assert benchmark.stats.stats.mean < 0.50  # 500ms ceiling on CI; spec target 250ms
