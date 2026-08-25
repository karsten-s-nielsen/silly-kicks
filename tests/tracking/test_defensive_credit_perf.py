"""Structural perf guard for compute_defensive_credits (ADR-068): the per-frame lookup is built
ONCE per call, not re-scanned per action x per rule.

The group_rows change is byte-identical, so the PARITY half is the (unchanged, comprehensive)
correctness suite -- test_defensive_credit_{aggregate,rules,resolution,orchestration,...}.py; this
file adds only the STRUCTURAL half the ADR pairs with it."""

import pandas as pd

import silly_kicks.tracking.defensive_credit._orchestration as _orch
from silly_kicks.tracking.defensive_credit import compute_defensive_credits
from tests._perf_structural import call_counter
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action


def _shot_scene():
    actions = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=34.0)
    actions["shot_blocked"] = pd.array([False], dtype="boolean")
    actions["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    actions["shot_on_target_derived"] = pd.array([False], dtype="boolean")
    actions["xg"] = [0.2]
    frames = frame_with_defender(defender_x=96.0, defender_y=34.0)
    return actions, frames


def test_frame_lookup_built_once_per_call(monkeypatch, fitted_xt):
    calls = call_counter(monkeypatch, _orch, "group_rows")
    actions, frames = _shot_scene()
    compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    # ONE build per compute_defensive_credits call. Pre-ADR-068 the whole `frames` table was
    # boolean-filtered per action x per enabled rule; a regression back to that shape (or moving
    # the build inside the per-action loop) would push this above 1.
    assert calls["n"] == 1
