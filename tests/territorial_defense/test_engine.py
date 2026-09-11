"""TF-54b engine: removal counterfactual + SPEC-02 FOV gate + PLAN-01 depletion guard + conservation."""

import numpy as np
import pandas as pd

from silly_kicks.territorial_defense._config import TerritorialDefenseParams
from silly_kicks.territorial_defense._engine import (
    action_ltr_goal_map,
    build_report,
    classify_arm_a_domain,
    local_completeness_ok,
    removal_leaves_enough_defenders,
    remove_player_row,
    select_arm_a_domain,
)
from silly_kicks.tracking import resolve_defended_goals

from ._fixtures import (
    ACTOR_PLAYER_ID,
    ATTACKING_TEAM_ID,
    DEFENDING_TEAM_ID,
    make_fitted_xt,
    make_rich_frame,
    make_thin_one_defender_frame,
)

# --- primitives --------------------------------------------------------------------------------


def test_remove_player_row_drops_exactly_one_and_is_pure():
    f = pd.DataFrame({"player_id": [0, 1, 2, 3], "is_ball": [False, False, False, True], "x": [1.0, 2.0, 3.0, 4.0]})
    cf = remove_player_row(f, player_pos=1)
    assert cf is not f and len(f) == 4  # pure, input intact
    assert len(cf) == 3 and 1 not in cf["player_id"].tolist()


def test_local_completeness_gate_two_sided():
    center = (52.5, 34.0)
    observed = np.array([[30.0, 10.0], [75.0, 10.0], [75.0, 58.0], [30.0, 58.0]])  # covers the 10 m disk
    cropped = np.array([[90.0, 60.0], [100.0, 60.0], [100.0, 68.0], [90.0, 68.0]])  # far corner
    assert local_completeness_ok(observed, center, radius_m=10.0, min_fraction=0.7) is True
    assert local_completeness_ok(cropped, center, radius_m=10.0, min_fraction=0.7) is False
    assert local_completeness_ok(None, center, radius_m=10.0, min_fraction=0.7) is False  # missing polygon


def test_removal_depletion_guard_two_sided_with_threat_evidence():
    from silly_kicks.tracking import compute_threat_pc, resolve_defended_goals

    thin = make_thin_one_defender_frame()  # D is the SOLE team-1 defender
    assert removal_leaves_enough_defenders(thin, defending_team_id=DEFENDING_TEAM_ID, min_after=1) is False
    rich = make_rich_frame()  # keeper + D + 4 others
    assert removal_leaves_enough_defenders(rich, defending_team_id=DEFENDING_TEAM_ID, min_after=1) is True

    # EVIDENCE the guard is not cosmetic: removing the sole defender inflates the attacking threat.
    # (goal_map from the rich frame -- the keeper-less thin frame cannot resolve its own geometry.)
    xt, gm = make_fitted_xt(), resolve_defended_goals(rich)
    dpos = int(thin.index[thin["player_id"] == ACTOR_PLAYER_ID][0])
    cf = remove_player_row(thin, player_pos=dpos)  # -> 0 defenders
    assert compute_threat_pc(cf, attacking_team_id=ATTACKING_TEAM_ID, xt=xt, goal_map=gm) > compute_threat_pc(
        thin, attacking_team_id=ATTACKING_TEAM_ID, xt=xt, goal_map=gm
    )


# --- domain selection --------------------------------------------------------------------------


def test_select_arm_a_domain_filters_defensive_types_and_derives_attacking_team():
    actions = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "action_id": [0, 1, 2],
            "team_id": pd.array([1, 1, 2], dtype="Int64"),
            "player_id": pd.array([102, 106, 201], dtype="Int64"),
            "type_id": [10, 0, 9],  # interception (in), pass (out), tackle (in)
        }
    )
    dom = select_arm_a_domain(actions)
    assert dom["action_id"].tolist() == [0, 2]  # only the defensive actions
    assert dom["frame_id"].tolist() == [0, 2]
    # defender team 1 -> attacking team 2, and vice versa
    row0 = dom[dom["action_id"] == 0].iloc[0]
    assert row0["defending_team_id"] == 1 and row0["attacking_team_id"] == 2
    row2 = dom[dom["action_id"] == 2].iloc[0]
    assert row2["defending_team_id"] == 2 and row2["attacking_team_id"] == 1


# --- conservation (ADR-042) --------------------------------------------------------------------


def _stamp(frame: pd.DataFrame, fid: int) -> pd.DataFrame:
    out = frame.copy()
    out["frame_id"] = fid
    return out


def _mixed_domain_and_frames():
    """4 in-domain candidates mapping to scored / removal_undersupported / no_actor / fov_cropped_local,
    plus a team-2 pass so the match has two teams (attacking team resolvable)."""
    rich0 = _stamp(make_rich_frame(), 0)  # -> scored (full-coverage polygon)
    thin1 = _stamp(make_thin_one_defender_frame(), 1)  # -> removal_undersupported
    noactor2 = _stamp(make_rich_frame(), 2)
    noactor2["is_actor"] = False  # -> no_actor
    crop3 = _stamp(make_rich_frame(), 3)  # -> fov_cropped_local (cropped polygon)
    frames = pd.concat([rich0, thin1, noactor2, crop3], ignore_index=True)

    actions = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1],
            "action_id": [0, 1, 2, 3, 4],
            "team_id": pd.array([1, 1, 1, 1, 2], dtype="Int64"),
            "player_id": pd.array([102, 102, 102, 102, 201], dtype="Int64"),
            "type_id": [10, 10, 10, 10, 0],  # 4 interceptions (in-domain) + 1 team-2 pass (out)
        }
    )
    full = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    crop = np.array([[95.0, 60.0], [104.0, 60.0], [104.0, 67.0], [95.0, 67.0]])  # far from D
    visible_area = pd.DataFrame({"action_id": [0, 3], "polygon": [full, crop]})
    return actions, frames, visible_area


def test_engine_conserves_frames_dropped_and_counted():
    actions, frames, visible_area = _mixed_domain_and_frames()
    dom = select_arm_a_domain(actions)
    # match-oriented fixture -> the per-match map (match_ltr) resolves; wrap it as the factory the
    # convention-agnostic classify now takes. Conservation is convention-agnostic.
    classified = classify_arm_a_domain(
        dom, frames, visible_area=visible_area, goal_map_for=lambda *a: resolve_defended_goals(frames)
    )
    report = build_report(classified["td_source"], params=TerritorialDefenseParams(), n_frames_in=len(dom))

    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    # non-vacuity: each of the four outcomes is present
    assert report.n_frames_scored == 1
    assert report.drop_reasons.get("removal_undersupported", 0) == 1  # PLAN-01
    assert report.drop_reasons.get("no_actor", 0) == 1
    assert report.drop_reasons.get("fov_cropped_local", 0) == 1  # SPEC-02


def test_action_ltr_goal_map_resolves_both_ends():
    # per-action-LTR convention: acting team attacks x=105 (defends 0); opponent defends 105.
    gm = action_ltr_goal_map(7, 1, acting_team_id=1, opponent_team_id=2)
    assert gm.attacked_goal(7, 1, 1, allow_guess=True) == 105.0  # acting attacks opponent's end
    assert gm.attacked_goal(7, 1, 2, allow_guess=True) == 0.0  # opponent attacks acting's end
    assert gm.attacked_goal(7, 1, "1", allow_guess=True) == 105.0  # dtype-agnostic keys (ADR-019/055)


def test_action_ltr_goal_map_is_a_goalmap():
    # Not an ADR-055 fork: same GoalMap type + accessor as resolve_defended_goals.
    from silly_kicks.tracking import GoalMap

    assert isinstance(action_ltr_goal_map(1, 1, acting_team_id=1, opponent_team_id=2), GoalMap)
