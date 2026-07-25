import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import features as _features
from silly_kicks.tracking.defensive_credit import _line_break_signal
from tests._perf_structural import call_counter
from tests.tracking._defensive_credit_fixtures import one_action
from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_pressure_perf_budget import fixture_100  # 100-action (actions, frames)

_ = fixture_100

_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]


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


def _line_break_perf_fixture(n_actions: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """100 actions, half SUCCESSFUL PASSES threading a 3-line frame (so the Item-3 line-break
    precompute is genuinely exercised -- non-vacuity, P9), half dribbles, all linked to one frame."""
    frames = _make_frame_rows(
        home_outfield_xs=[8.0, 12.0, 16.0, 20.0, 24.0],
        home_outfield_ys=[10.0, 24.0, 34.0, 44.0, 58.0],
        away_outfield_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
        away_outfield_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
        frame_id=500,
        time_seconds=1.0,
    )
    rows = []
    for i in range(n_actions):
        if i % 2 == 0:  # candidate: a successful pass through the lines
            rows.append(
                one_action(
                    action_id=i,
                    type_name="pass",
                    result_name="success",
                    start_x=48.0,
                    start_y=34.0,
                    end_x=95.0,
                    end_y=34.0,
                    team_id=1,
                    player_id=200,
                    time_seconds=1.0,
                    game_id=1,
                )
            )
        else:  # non-candidate: a dribble (must NOT trigger Ward clustering)
            rows.append(
                one_action(
                    action_id=i,
                    type_name="dribble",
                    result_name="success",
                    start_x=40.0,
                    start_y=34.0,
                    end_x=45.0,
                    end_y=34.0,
                    team_id=1,
                    player_id=201,
                    time_seconds=1.0,
                    game_id=1,
                )
            )
    actions = pd.concat(rows, ignore_index=True)
    actions["shot_blocked"] = pd.array([pd.NA] * n_actions, dtype="boolean")
    actions["cross_blocked"] = pd.array([pd.NA] * n_actions, dtype="boolean")
    actions["shot_on_target_derived"] = pd.array([pd.NA] * n_actions, dtype="boolean")
    actions["xg"] = np.nan
    return actions, frames


def test_line_break_precompute_clusters_only_candidate_passes(monkeypatch, fitted_xt):
    """Item 3 (P9/P10): the line-break precompute links ONCE and runs Ward (`_straddle_core`) ONLY
    on candidate rows (successful passes), never per-action. Non-vacuous: the fixture contains real
    candidates reaching the line-break path (else the guard passes on an empty surface)."""
    actions, frames = _line_break_perf_fixture(100)
    n_candidates = int(((actions["type_id"] == _PASS) & (actions["result_id"] == _SUCCESS)).sum())
    assert n_candidates >= 10  # non-vacuity: the precompute genuinely reaches Ward

    link = call_counter(monkeypatch, _features, "link_actions_to_frames")
    straddle = call_counter(monkeypatch, _line_break_signal, "_straddle_core")
    result = _features.add_defensive_credit(actions, frames, xg_column="xg", xt=fitted_xt)

    assert "defensive_credit_net" in result.columns
    assert link["n"] == 1, f"add_defensive_credit linked {link['n']} times (expected 1)."
    assert 0 < straddle["n"] <= n_candidates, (
        f"_straddle_core ran {straddle['n']}x for {n_candidates} candidate passes among "
        f"{len(actions)} actions -- Ward must cluster ONLY candidate rows (P10), not every action."
    )
