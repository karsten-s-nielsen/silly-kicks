"""Structural performance guard for pressure_on_actor (TF-2), all three methods.

Replaces flaky wall-clock budgets (120ms/500ms per 100 actions, runner-variance-prone) with a
deterministic invariant: every method links actions→frames ONCE for the whole batch (via
``_resolve_action_frame_context``) and then vectorises the pressure kernel over all actions.
A regression to per-action re-linking is the real O(actions) cost blow-up. See
tests/_perf_structural.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import features as _features
from tests._perf_structural import call_counter

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


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones", "bekkers_pi"])
def test_pressure_links_once_per_100_actions(method, fixture_100, monkeypatch) -> None:
    actions, frames = fixture_100
    # pressure_on_actor resolves the linked frame context via features._resolve_action_frame_context
    # (imported into the features namespace); patch that name.
    calls = call_counter(monkeypatch, _features, "_resolve_action_frame_context")

    result = _features.pressure_on_actor(actions, frames, method=method)

    assert result.notna().any()
    assert calls["n"] == 1, (
        f"pressure_on_actor(method={method!r}) resolved the frame context {calls['n']} times for "
        "100 actions (expected 1). Per-action re-linking is the O(actions) regression the budget proxied."
    )
