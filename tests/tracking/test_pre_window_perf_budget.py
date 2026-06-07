"""Structural performance guard for add_actor_pre_window (TF-3).

Replaces a flaky wall-clock budget (the 150ms/100-action ceiling, runner-variance-prone) with
a deterministic invariant: the aggregator links actions→frames ONCE for the whole batch, then
computes the pre-window arc-length/displacement vectorised. Per-action re-linking is the real
O(actions) cost blow-up. See tests/_perf_structural.py.
"""

from __future__ import annotations

from silly_kicks.tracking import features as _features
from tests._perf_structural import call_counter

# Re-use the 100-action fixture from the pressure perf budget module.
from .test_pressure_perf_budget import fixture_100

# Re-export so pytest discovers the fixture in this module's collection.
_ = fixture_100


def test_pre_window_links_once_per_100_actions(fixture_100, monkeypatch) -> None:
    actions, frames = fixture_100
    calls = call_counter(monkeypatch, _features, "link_actions_to_frames")

    result = _features.add_actor_pre_window(actions, frames)

    assert "actor_arc_length_pre_window" in result.columns
    assert calls["n"] == 1, (
        f"add_actor_pre_window linked {calls['n']} times for 100 actions (expected 1). "
        "Per-action re-linking is the O(actions) regression the wall-clock budget proxied."
    )
