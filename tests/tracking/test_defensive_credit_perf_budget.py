import pandas as pd

from silly_kicks.tracking import features as _features
from tests._perf_structural import call_counter
from tests.tracking.test_pressure_perf_budget import fixture_100  # 100-action (actions, frames)

_ = fixture_100


def _prep(actions):
    """Supply the SPADL + block/xg columns add_defensive_credit needs that the pressure fixture omits."""
    actions = actions.copy()
    if "game_id" not in actions.columns:
        actions["game_id"] = "g"
    if "result_id" not in actions.columns:
        actions["result_id"] = 0
    if "end_x" not in actions.columns:
        actions["end_x"] = actions["start_x"]
    if "end_y" not in actions.columns:
        actions["end_y"] = actions["start_y"]
    actions["xg"] = 0.1
    actions["shot_blocked"] = pd.array([pd.NA] * len(actions), dtype="boolean")
    actions["cross_blocked"] = pd.array([pd.NA] * len(actions), dtype="boolean")
    return actions


def test_add_defensive_credit_links_once_per_100_actions(fixture_100, monkeypatch, fitted_xt):
    actions, frames = fixture_100
    actions = _prep(actions)
    actions["shot_on_target_derived"] = pd.array([pd.NA] * len(actions), dtype="boolean")  # skip TF-48 fallback

    calls = call_counter(monkeypatch, _features, "link_actions_to_frames")
    result = _features.add_defensive_credit(actions, frames, xg_column="xg", xt=fitted_xt)  # P-2: no home_team_id

    assert "defensive_credit_net" in result.columns
    assert calls["n"] == 1, (
        f"add_defensive_credit linked {calls['n']} times for 100 actions (expected 1). "
        "Per-action re-linking is the O(actions) regression the structural budget proxies."
    )


def test_add_defensive_credit_links_once_even_via_tf48_fallback(fixture_100, monkeypatch, fitted_xt):
    """The on-target TF-48 fallback path (NO shot_on_target_derived supplied) must ALSO reuse the
    single link -- the budget transitively depends on add_shot_goalmouth honoring links=. If TF-48
    ever re-links internally despite links=pointers, this fails loud (Task-15 self-check)."""
    actions, frames = fixture_100
    actions = _prep(actions)  # deliberately no shot_on_target_derived -> forces _ensure_on_target's TF-48 call

    calls = call_counter(monkeypatch, _features, "link_actions_to_frames")
    result = _features.add_defensive_credit(actions, frames, xg_column="xg", xt=fitted_xt)

    assert "defensive_credit_net" in result.columns
    assert calls["n"] == 1, (
        f"add_defensive_credit linked {calls['n']} times via the TF-48 on-target fallback (expected 1). "
        "add_shot_goalmouth must reuse the pointers passed via links=, not re-link internally."
    )
