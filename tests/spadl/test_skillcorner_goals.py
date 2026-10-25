"""Task 2.4 — SkillCorner events-only goals reconciliation (spec section 6).

Synthetic actions + possessions + match.json truth, each repair path with uniqueness met and not met.
"""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlcfg
from silly_kicks.spadl._skillcorner_goals import _OWN_GOAL_END, _PENALTY_SPOT, reconcile_goals

_ACTION_COLS = [
    "game_id",
    "original_event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_id",
    "result_id",
    "result_source",
    "bodypart_id",
    "action_provenance",
    "restart_evidence",
    "type_source",
    "repair_kind",
]


def _action(eid, team, player, type_name="shot", result="fail", t=10.0):
    return {
        "game_id": "M1",
        "original_event_id": eid,
        "action_id": 0,
        "period_id": 1,
        "time_seconds": t,
        "team_id": team,
        "player_id": player,
        "start_x": 90.0,
        "start_y": 34.0,
        "end_x": 105.0,
        "end_y": 34.0,
        "type_id": spadlcfg.actiontype_id[type_name],
        "result_id": spadlcfg.result_id[result],
        "result_source": "native",
        "bodypart_id": spadlcfg.bodypart_id["foot"],
        "action_provenance": "native",
        "restart_evidence": pd.NA,
        "type_source": pd.NA,
        "repair_kind": pd.NA,
    }


def _actions(rows):
    df = pd.DataFrame(rows)
    for i in range(len(df)):
        df.loc[i, "action_id"] = i
    return df[_ACTION_COLS]


def _poss(eid, team, player, *, end_type="pass", gi_before="", gi_after="", hs=0, as_=0, t=10.0):
    # team_score / opponent_team_score are relative to the POSSESSING team.
    return {
        "event_id": eid,
        "period_id": 1,
        "time_seconds": t,
        "team_id": team,
        "player_id": player,
        "end_type": end_type,
        "start_type": "pass_reception",
        "game_interruption_before": gi_before,
        "game_interruption_after": gi_after,
        "team_score": hs if team == "H" else as_,
        "opponent_team_score": as_ if team == "H" else hs,
    }


def _meta(players, home_score=1, away_score=0):
    return {
        "id": "M1",
        "home_team": {"id": "H"},
        "away_team": {"id": "A"},
        "home_team_score": home_score,
        "away_team_score": away_score,
        "players": players,
    }


def test_normal_goal_sets_shot_success_and_consumes_budget():
    acts = _actions([_action("e1", "H", "h9", "shot", "fail")])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="shot", gi_after="goal_for", hs=1, as_=0),
            _poss("e2", "A", "a5", hs=1, as_=0, t=20.0),  # kick-off carrying the new score
        ]
    )
    out, rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    row = out[out["original_event_id"] == "e1"].iloc[0]
    assert row["result_id"] == spadlcfg.result_id["success"]
    assert row["repair_kind"] == "normal"
    assert rep.repaired.get("normal") == 1


def test_scored_tag_uses_gi_before():
    acts = _actions([_action("e1", "H", "h9", "shot", "fail")])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="shot", gi_before="goal_for", hs=1, as_=0),
            _poss("e2", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    out, rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    assert rep.repaired.get("scored_tag") == 1
    assert out[out["original_event_id"] == "e1"].iloc[0]["repair_kind"] == "scored_tag"


def test_own_goal_unsignalled_emits_bad_touch():
    # Home score goes up; no home goal row; no home deficit; unique away own-goal candidate a3.
    acts = _actions([_action("e1", "H", "h9", "pass", "success")])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", hs=0, as_=0),
            _poss("e2", "A", "a3", end_type="possession_loss", hs=1, as_=0, t=20.0),  # score now 1-0, no shot
        ]
    )
    meta = _meta(
        [
            {"id": "h9", "team_id": "H", "goal": 0, "own_goal": 0},
            {"id": "a3", "team_id": "A", "goal": 0, "own_goal": 1},
        ]
    )
    out, _rep = reconcile_goals(acts, pp, meta)
    og = out[out["repair_kind"] == "own_goal_unsignalled"]
    assert len(og) == 1
    r = og.iloc[0]
    assert r["team_id"] == "A" and r["player_id"] == "a3"
    assert r["result_id"] == spadlcfg.result_id["owngoal"]
    assert r["type_id"] == spadlcfg.actiontype_id["bad_touch"]
    assert (r["end_x"], r["end_y"]) == _OWN_GOAL_END
    assert pd.isna(r["start_x"])


def test_missing_goal_sets_last_shot_success():
    acts = _actions(
        [
            _action("e1", "H", "h9", "shot", "fail", t=10.0),
            _action("e2", "H", "h9", "shot", "fail", t=15.0),  # the later shot
        ]
    )
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="shot", hs=0, as_=0, t=10.0),
            _poss("e2", "H", "h9", end_type="shot", hs=1, as_=0, t=15.0),  # score up, no goal_for tag
            _poss("e3", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    out, rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    assert rep.repaired.get("missing_goal") == 1
    assert out[out["original_event_id"] == "e2"].iloc[0]["result_id"] == spadlcfg.result_id["success"]


def test_missing_goal_penalty_emits_shot_penalty_at_spot():
    acts = _actions([_action("e1", "H", "h9", "pass", "success", t=10.0)])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", gi_before="penalty_for", hs=0, as_=0, t=10.0),
            _poss("e2", "H", "h9", end_type="possession_loss", gi_before="penalty_for", hs=1, as_=0, t=15.0),
            _poss("e3", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    out, rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    assert rep.repaired.get("missing_goal") == 1
    pen = out[out["type_id"] == spadlcfg.actiontype_id["shot_penalty"]]
    assert len(pen) == 1
    assert (pen.iloc[0]["start_x"], pen.iloc[0]["start_y"]) == _PENALTY_SPOT


def test_coverage_gap_flagged():
    acts = _actions([_action("e1", "H", "h9", "pass", "success")])
    pp = pd.DataFrame([_poss("e1", "H", "h9", end_type="pass", hs=0, as_=0)])  # GI never reaches 1-0
    _out, rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    assert rep.coverage_gap is True


def test_reading_i_retypes_the_derived_restart_row():
    # A derived free kick immediately before the goal, no possession between -> retype that one row.
    acts = _actions(
        [
            _action("e1", "H", None, "freekick_crossed", "success", t=12.0),
            _action("e3", "A", "a5", "pass", "success", t=20.0),
        ]
    )
    acts.loc[0, "action_provenance"] = "derived"
    acts.loc[0, "restart_evidence"] = "tagged"
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", hs=0, as_=0, t=10.0),
            _poss("e2", "H", "h9", end_type="possession_loss", hs=1, as_=0, t=15.0),  # score up, no goal_for
            _poss("e3", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    out, rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    assert rep.repaired.get("reading_i") == 1
    r = out[out["original_event_id"] == "e1"].iloc[0]
    assert r["type_id"] == spadlcfg.actiontype_id["shot_freekick"]
    assert r["result_id"] == spadlcfg.result_id["success"]
    assert r["player_id"] == "h9"  # the reconciled scorer


def test_own_goal_surplus_emits_own_goal():
    # GI credits h9's shot, but h9 has 0 metadata goals (surplus) and a3 (away) has an own goal.
    acts = _actions([_action("e1", "H", "h9", "shot", "fail")])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="shot", gi_after="goal_for", hs=1, as_=0),
            _poss("e2", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    meta = _meta(
        [
            {"id": "h9", "team_id": "H", "goal": 0, "own_goal": 0},
            {"id": "a3", "team_id": "A", "goal": 0, "own_goal": 1},
        ]
    )
    out, rep = reconcile_goals(acts, pp, meta)
    assert rep.repaired.get("own_goal_surplus") == 1
    og = out[out["repair_kind"] == "own_goal_surplus"].iloc[0]
    assert og["team_id"] == "A" and og["player_id"] == "a3"
    assert og["result_id"] == spadlcfg.result_id["owngoal"]


def test_swap_reported_not_row_repaired():
    # GI credits h1's shot, but match.json says h2 scored (and h1 scored 0): a swap. Reported only.
    acts = _actions([_action("e1", "H", "h1", "shot", "fail")])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h1", end_type="shot", gi_after="goal_for", hs=1, as_=0),
            _poss("e2", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    meta = _meta(
        [
            {"id": "h1", "team_id": "H", "goal": 0, "own_goal": 0},
            {"id": "h2", "team_id": "H", "goal": 1, "own_goal": 0},
        ]
    )
    out, rep = reconcile_goals(acts, pp, meta)
    assert rep.repaired.get("swap") == 1
    # not row-repaired: no shot was flipped to success, no row appended.
    assert rep.repaired.get("missing_goal") is None
    assert (out["result_id"] == spadlcfg.result_id["success"]).sum() == 0


def test_own_goal_unsignalled_requires_a_unique_candidate():
    # Two away players each with own_goal 1 -> not unique -> NOT repaired, reported ambiguous.
    acts = _actions([_action("e1", "H", "h9", "pass", "success")])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", hs=0, as_=0),
            _poss("e2", "A", "a3", end_type="possession_loss", hs=1, as_=0, t=20.0),
        ]
    )
    meta = _meta(
        [
            {"id": "h9", "team_id": "H", "goal": 0, "own_goal": 0},
            {"id": "a3", "team_id": "A", "goal": 0, "own_goal": 1},
            {"id": "a4", "team_id": "A", "goal": 0, "own_goal": 1},
        ]
    )
    out, rep = reconcile_goals(acts, pp, meta)
    assert rep.repaired.get("own_goal_unsignalled") is None
    assert rep.ambiguous >= 1
    assert (out["result_id"] == spadlcfg.result_id["owngoal"]).sum() == 0


def test_two_deficit_candidates_is_ambiguous():
    acts = _actions([_action("e1", "H", "h9", "pass", "success")])
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", hs=0, as_=0),
            _poss("e2", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    meta = _meta(
        [
            {"id": "h1", "team_id": "H", "goal": 1, "own_goal": 0},
            {"id": "h2", "team_id": "H", "goal": 1, "own_goal": 0},
        ],
        home_score=2,
    )
    _out, rep = reconcile_goals(acts, pp, meta)
    assert rep.ambiguous >= 1 and rep.repaired.get("missing_goal") is None


def test_reading_i_own_goal_branch_keeps_restart_unretyped():
    # A derived corner immediately before an OWN-goal score change: own_goal_unsignalled emits the
    # own-goal row and the restart delivery is NOT retyped to a shot (spec section 6.6 own-goal branch).
    acts = _actions(
        [
            _action("e1", "H", None, "corner_crossed", "fail", t=12.0),
            _action("e3", "A", "a5", "pass", "success", t=20.0),
        ]
    )
    acts.loc[0, "action_provenance"] = "derived"
    acts.loc[0, "restart_evidence"] = "tagged"
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", hs=0, as_=0, t=10.0),
            _poss("e2", "A", "a3", end_type="possession_loss", hs=1, as_=0, t=15.0),
            _poss("e3", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    meta = _meta(
        [
            {"id": "h9", "team_id": "H", "goal": 0, "own_goal": 0},
            {"id": "a3", "team_id": "A", "goal": 0, "own_goal": 1},
        ]
    )
    out, rep = reconcile_goals(acts, pp, meta)
    assert rep.repaired.get("own_goal_unsignalled") == 1
    corner = out[out["original_event_id"] == "e1"].iloc[0]
    assert corner["type_id"] == spadlcfg.actiontype_id["corner_crossed"]  # NOT retyped to shot
    assert corner["result_id"] == spadlcfg.result_id["fail"]
    assert (out["result_id"] == spadlcfg.result_id["owngoal"]).sum() == 1


def test_c2_own_goal_row_survives_validate_spadl_and_add_names():
    # Spec section 10.3: a reconciled own-goal row (NaN start, bad_touch) survives schema validation and
    # enrichment without a drop or a raise. Exercised on real opendata (1899585 = own_goal_unsignalled).
    import json as _json
    from pathlib import Path

    from silly_kicks.spadl import skillcorner
    from silly_kicks.spadl.utils import add_names, validate_spadl

    root = Path(__file__).parents[1] / "datasets" / "skillcorner" / "opendata"
    meta = _json.loads((root / "1899585_match.json").read_text(encoding="utf-8"))
    actions, _ = skillcorner.convert_to_actions(pd.read_csv(root / "1899585_events_slim.csv", low_memory=False), meta)
    og = actions[actions["result_id"] == spadlcfg.result_id["owngoal"]]
    assert len(og) >= 1 and bool(og["start_x"].isna().all())  # NaN-start own-goal row present
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)  # SkillCorner string ids warn (expected); C2 asserts no RAISE
        validate_spadl(actions)  # raises on a schema breach
        named = add_names(actions)  # NaN-id-safe enrichment, no crash
    assert len(named) == len(actions)  # row not dropped


def test_c2_own_goal_row_survives_vaep_features_and_xt():
    # Spec section 10.3: the NaN-start bad_touch own-goal row survives VAEP feature computation and xT
    # fitting/scoring -- it is not a move action, so xT's move-action coord dropna never touches it and
    # nothing crashes. Exercised on real opendata (1899585).
    import json as _json
    import warnings as _w
    from pathlib import Path

    from silly_kicks.spadl import skillcorner
    from silly_kicks.vaep import features as vf
    from silly_kicks.xthreat import ExpectedThreat

    root = Path(__file__).parents[1] / "datasets" / "skillcorner" / "opendata"
    meta = _json.loads((root / "1899585_match.json").read_text(encoding="utf-8"))
    actions, _ = skillcorner.convert_to_actions(pd.read_csv(root / "1899585_events_slim.csv", low_memory=False), meta)
    assert (actions["result_id"] == spadlcfg.result_id["owngoal"]).sum() >= 1

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        gs = vf.gamestates(actions, 1)
        loc = vf.simple(vf.startlocation)(gs)  # reads the NaN start -> NaN feature, never a drop or crash
        atype = vf.simple(vf.actiontype)(gs)
        xt = ExpectedThreat(l=16, w=12).fit(actions)  # no raise on the NaN-start bad_touch row
    assert len(loc) == len(actions) and len(atype) == len(actions)  # own-goal row not dropped
    assert xt is not None


def test_anchor_post_goal_kickoff_and_fallback_both_reconcile():
    # G5: the goal reconciles whether the new score first appears on the post-goal kick-off possession
    # (anchor) or on the scoring possession itself (fallback).
    meta = _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}])
    anchor_pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="shot", gi_after="goal_for", hs=0, as_=0),  # score not yet updated
            _poss("e2", "A", "a5", hs=1, as_=0, t=20.0),  # post-goal kick-off carries the new score
        ]
    )
    _o, ra = reconcile_goals(_actions([_action("e1", "H", "h9", "shot", "fail")]), anchor_pp, meta)
    assert ra.repaired.get("normal") == 1 and ra.unresolved == 0

    fallback_pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="shot", gi_after="goal_for", hs=1, as_=0),  # score updates HERE
            _poss("e2", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    )
    _o2, rf = reconcile_goals(_actions([_action("e1", "H", "h9", "shot", "fail")]), fallback_pp, meta)
    assert rf.repaired.get("normal") == 1 and rf.unresolved == 0


def test_kickoff_anchor_shifts_to_unknown_start_conceding_possession():
    # Spec section 6.4: the anchor shifts from the raw score increment to the conceding team's post-goal
    # kick-off (first start_type=="unknown" conceding possession before the next change). Exercises the gate.
    from silly_kicks.spadl._skillcorner_goals import _kickoff_anchors

    changes = [(1, "H")]
    pp_team = ["H", "H", "A", "A"]
    start_type = ["pass_reception", "pass_reception", "unknown", "pass_reception"]
    assert _kickoff_anchors(changes, pp_team, start_type, "H", "A") == [(2, "H")]


def test_kickoff_anchor_falls_back_to_score_change_without_unknown_kickoff():
    from silly_kicks.spadl._skillcorner_goals import _kickoff_anchors

    out = _kickoff_anchors([(1, "H")], ["H", "H", "A"], ["pass_reception"] * 3, "H", "A")
    assert out == [(1, "H")]  # no unknown-start conceding possession -> fallback to the increment index


def test_penalty_does_not_rewrite_an_earlier_miss_in_episode():
    # Episode-bounding (B-05 synthetic): the taker's earlier MISS (before the penalty marker) is NOT
    # rewritten to the spot; the penalty is a derived shot_penalty anchored at the marker.
    acts = _actions(
        [
            _action("e1", "H", "h9", "shot", "fail", t=10.0),  # earlier miss by the taker
            _action("e2", "H", "h9", "pass", "success", t=30.0),
        ]
    )
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="shot", hs=0, as_=0, t=10.0),  # miss, before the penalty
            _poss("e2", "H", "h9", end_type="pass", gi_before="penalty_for", hs=0, as_=0, t=30.0),  # penalty marker
            _poss("e3", "H", "h9", end_type="possession_loss", gi_before="penalty_for", hs=1, as_=0, t=35.0),
            _poss("e4", "A", "a5", hs=1, as_=0, t=40.0),
        ]
    )
    out, rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    assert rep.repaired.get("missing_goal") == 1
    miss = out[out["original_event_id"] == "e1"].iloc[0]
    assert miss["type_id"] == spadlcfg.actiontype_id["shot"] and miss["result_id"] == spadlcfg.result_id["fail"]
    pen = out[out["type_id"] == spadlcfg.actiontype_id["shot_penalty"]]
    assert len(pen) == 1 and (pen.iloc[0]["start_x"], pen.iloc[0]["start_y"]) == _PENALTY_SPOT


@pytest.mark.parametrize("sources", [("native", "frame"), ("frame", "native")])
def test_own_goal_row_takes_the_period_source_of_its_possession(sources):
    # TF-46 Part 2 spec section 26.5: a row whose period comes from a GI possession carries that possession's
    # period_source; the twin (sources swapped) proves it copies the RIGHT possession, not a constant.
    acts = _actions([_action("e1", "H", "h9", "pass", "success")]).assign(period_source="native")
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", hs=0, as_=0),
            _poss("e2", "A", "a3", end_type="possession_loss", hs=1, as_=0, t=20.0),
        ]
    ).assign(period_source=list(sources))
    meta = _meta(
        [
            {"id": "h9", "team_id": "H", "goal": 0, "own_goal": 0},
            {"id": "a3", "team_id": "A", "goal": 0, "own_goal": 1},
        ]
    )
    out, _rep = reconcile_goals(acts, pp, meta)
    og = out[out["repair_kind"] == "own_goal_unsignalled"].iloc[0]
    assert og["period_source"] == dict(zip(pp["event_id"], pp["period_source"], strict=True))[og["original_event_id"]]


@pytest.mark.parametrize("sources", [("native", "frame", "native"), ("frame", "native", "frame")])
def test_derived_shot_row_takes_the_period_source_of_its_possession(sources):
    acts = _actions([_action("e1", "H", "h9", "pass", "success", t=10.0)]).assign(period_source="native")
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", gi_before="penalty_for", hs=0, as_=0, t=10.0),
            _poss("e2", "H", "h9", end_type="possession_loss", gi_before="penalty_for", hs=1, as_=0, t=15.0),
            _poss("e3", "A", "a5", hs=1, as_=0, t=20.0),
        ]
    ).assign(period_source=list(sources))
    out, _rep = reconcile_goals(acts, pp, _meta([{"id": "h9", "team_id": "H", "goal": 1, "own_goal": 0}]))
    pen = out[out["type_id"] == spadlcfg.actiontype_id["shot_penalty"]].iloc[0]
    assert pen["period_source"] == dict(zip(pp["event_id"], pp["period_source"], strict=True))[pen["original_event_id"]]


def test_goals_rows_default_to_native_without_a_period_source_column():
    # the possession table carries no period_source (a caller without the repair): the emitted row says "native"
    acts = _actions([_action("e1", "H", "h9", "pass", "success")]).assign(period_source="native")
    pp = pd.DataFrame(
        [
            _poss("e1", "H", "h9", end_type="pass", hs=0, as_=0),
            _poss("e2", "A", "a3", end_type="possession_loss", hs=1, as_=0, t=20.0),
        ]
    )
    meta = _meta(
        [
            {"id": "h9", "team_id": "H", "goal": 0, "own_goal": 0},
            {"id": "a3", "team_id": "A", "goal": 0, "own_goal": 1},
        ]
    )
    out, _rep = reconcile_goals(acts, pp, meta)
    assert out[out["repair_kind"] == "own_goal_unsignalled"].iloc[0]["period_source"] == "native"
