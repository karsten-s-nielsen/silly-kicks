import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.defensive_credit import (
    DEFENSIVE_CREDIT_RULES,
    DefensiveCreditParams,
    compute_defensive_credits,
)
from silly_kicks.tracking.defensive_credit._params import (
    ANCHOR_TYPE_VALUES,
    RESOLUTION_VALUES,
    SIZING_VALUES,
)
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action

_LONG_COLS = [
    "game_id",
    "period_id",
    "action_id",
    "player_id",
    "team_id",
    "rule",
    "signed_value",
    "anchor_type",
    "frame_id",
    "sizing",
    "resolution",
]


def _shot_scene(fitted_xt):
    actions = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=34.0)
    actions["shot_blocked"] = pd.array([False], dtype="boolean")
    actions["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    # deterministic OFF-target so the orchestrator does not need the TF-48 frame fallback.
    actions["shot_on_target_derived"] = pd.array([False], dtype="boolean")
    actions["xg"] = [0.2]
    frames = frame_with_defender(defender_x=96.0, defender_y=34.0)
    return actions, frames


def test_long_form_schema_and_values(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert list(out.columns) == _LONG_COLS
    assert (out["rule"] == "pressure_on_missed_shot").all()
    assert out["signed_value"].iloc[0] == pytest.approx(0.2)
    assert set(out["sizing"]) <= set(SIZING_VALUES)
    assert set(out["anchor_type"]) <= set(ANCHOR_TYPE_VALUES)
    assert set(out["resolution"]) <= set(RESOLUTION_VALUES)


def test_pressured_saved_shot_is_negative_end_to_end(fitted_xt):
    # THE P-1 regression at the orchestrator level.
    actions, frames = _shot_scene(fitted_xt)
    actions["shot_on_target_derived"] = pd.array([True], dtype="boolean")  # a save = on-target
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert (out["rule"] == "failed_pressure_shot_on_target").all()
    assert out["signed_value"].iloc[0] == pytest.approx(-0.2)


def test_closed_vocabulary(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert set(out["rule"]) <= set(DEFENSIVE_CREDIT_RULES)


def test_rules_gating_disables_a_rule(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    params = DefensiveCreditParams(rules=frozenset(set(DEFENSIVE_CREDIT_RULES) - {"pressure_on_missed_shot"}))
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt, params=params)
    assert "pressure_on_missed_shot" not in set(out["rule"])
    out2 = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert "pressure_on_missed_shot" in set(out2["rule"])


def test_fired_but_unsizable_is_nan_row(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    actions["xg"] = [np.nan]
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert len(out) == 1 and np.isnan(out["signed_value"].iloc[0])


def test_no_defender_no_row(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    frames.loc[frames["player_id"] == 900, "x"] = 80.0  # move defender far
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert out.empty


def test_xg_column_missing_fails_loud(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    actions = actions.drop(columns=["xg"])
    with pytest.raises(ValueError, match="xg_column"):
        compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)


def test_mirror_invariance_home_vs_away(fitted_xt):
    """Same physical situation as a home and an away action -> identical action-LTR credit (ADR-028);
    asymmetric + extreme fixture so a y-symmetric one can't pass vacuously."""
    home_actions = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=20.0, team_id=10)
    home_actions["shot_blocked"] = pd.array([False], dtype="boolean")
    home_actions["xg"] = [0.2]
    home_actions["shot_on_target_derived"] = pd.array([False], dtype="boolean")
    home_frames = frame_with_defender(defender_x=96.0, defender_y=20.0, acting_team_id=10, home_team_id=10)
    home = compute_defensive_credits(home_actions, home_frames, xg_column="xg", xt=fitted_xt)

    # AWAY: acting team 20 != home(10). action-LTR (95,20) -> frame (10,48); defender frame (9,48).
    away_actions = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=20.0, team_id=20)
    away_actions["shot_blocked"] = pd.array([False], dtype="boolean")
    away_actions["xg"] = [0.2]
    away_actions["shot_on_target_derived"] = pd.array([False], dtype="boolean")
    away_frames = frame_with_defender(
        defender_x=9.0, defender_y=48.0, acting_team_id=20, home_team_id=10, defender_team_id=10
    )
    away = compute_defensive_credits(away_actions, away_frames, xg_column="xg", xt=fitted_xt)

    assert not home.empty and not away.empty
    assert home["signed_value"].iloc[0] == pytest.approx(away["signed_value"].iloc[0])
    assert home["rule"].iloc[0] == away["rule"].iloc[0]


def test_sizing_regression_dangerous_turnover_scores_higher(fitted_xt):
    """A turnover forced near the defending goal (high xT(origin)) >> the same turnover deep in own half."""

    def _pass_scene(sx):
        a = one_action(type_name="pass", result_name="fail", start_x=sx, start_y=34.0, team_id=10, player_id=5)
        a["shot_blocked"] = pd.array([pd.NA], dtype="boolean")
        a["xg"] = [np.nan]
        a["shot_on_target_derived"] = pd.array([pd.NA], dtype="boolean")
        f = frame_with_defender(defender_x=sx + 1.0, defender_y=34.0)
        return compute_defensive_credits(
            a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(rules=frozenset({"pressure_pass_fail"}))
        )

    near = _pass_scene(95.0)  # near attacked goal -> high xT(origin)
    deep = _pass_scene(10.0)  # own half -> low xT(origin)
    near_plus = near[near["signed_value"] > 0]["signed_value"].iloc[0]
    deep_plus = deep[deep["signed_value"] > 0]["signed_value"].iloc[0]
    assert near_plus > deep_plus
