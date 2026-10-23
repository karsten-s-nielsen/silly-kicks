"""Task 2.7 refine_restart_types (spec section 11): native-never-rewritten, gated subtype rewrite, the B8
merge, penalty rows, one renumber, and reversibility (the *_events round trip). Both sides of the gate."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from silly_kicks.spadl import _refine_restart_types as rr
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import refine_restart_types
from silly_kicks.tracking._gk_resolve import GoalMap
from silly_kicks.tracking._restart_validation import DataProfile, GateEntry

_CS = spadlconfig.actiontype_id["corner_short"]
_CC = spadlconfig.actiontype_id["corner_crossed"]
_SHOT = spadlconfig.actiontype_id["shot"]
_SP = spadlconfig.actiontype_id["shot_penalty"]
_FKS = spadlconfig.actiontype_id["freekick_short"]
_P = DataProfile("skillcorner", "r1", "g1", "2025")
# H defends x = 0, so H attacks +x: the frame convention IS H's action frame (the section 11.2 rule's frame).
_GM = GoalMap(resolved={("g", "1", "H"): 0.0, ("g", "1", "A"): 105.0}, guessed={}, unresolved=frozenset())
# the mirror: H defends x = 105 and attacks -x.
_GM_MIRROR = GoalMap(resolved={("g", "1", "H"): 105.0, ("g", "1", "A"): 0.0}, guessed={}, unresolved=frozenset())
_GM_NONE = GoalMap(resolved={}, guessed={}, unresolved=frozenset())


def _pass(*a, **k):
    return GateEntry("transfer", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell")


def _action(
    action_id,
    type_id,
    *,
    type_source: object = "default",
    t=10.0,
    team="H",
    provenance="native",
    result="success",
    player: object = "h7",
):
    return {
        "game_id": "g",
        "action_id": action_id,
        "period_id": 1,
        "time_seconds": t,
        "team_id": team,
        "player_id": player,
        "type_id": type_id,
        "type_source": type_source,
        "result_id": spadlconfig.result_id[result],
        "result_source": "native",
        "action_provenance": provenance,
        "start_x": 1.0,
        "start_y": 1.0,
        "end_x": 2.0,
        "end_y": 2.0,
        "restart_evidence": pd.NA,
        "repair_kind": pd.NA,
    }


def _frames_ball_in_box(t0=10.0):
    # ball enters the penalty area (x >= L-16.5 = 88.5) shortly after the restart.
    rows = []
    for i in range(10):
        rows.append(
            {
                "game_id": "g",
                "period_id": 1,
                "frame_id": 1000 + i,
                "time_seconds": t0 + 0.1 * (i + 1),
                "is_ball": True,
                "x": 90.0,
                "y": 34.0,
            }
        )
    return pd.DataFrame(rows)


def test_native_row_is_never_rewritten(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_action(0, _CS, type_source="native")])
    out = refine_restart_types(a, _frames_ball_in_box(), provider="sportec", data_profile=_P, goal_map=_GM)
    assert out.loc[0, "type_id"] == _CS  # unchanged
    assert bool(out.loc[0, "type_tracking_disagrees"]) is True  # tracking says crossed


def test_subtype_rewrite_on_default_keeps_prior_in_events(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_action(0, _CS, type_source="default")])
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P, goal_map=_GM)
    assert out.loc[0, "type_id"] == _CC and out.loc[0, "type_source"] == "tracking"
    assert out.loc[0, "type_id_events"] == _CS and out.loc[0, "type_source_events"] == "default"


def test_c2_empty_gate_is_noop(monkeypatch):
    # data_profile=None -> no gate -> no rewrite (C2); type_tracking_disagrees still recorded.
    a = pd.DataFrame([_action(0, _CS, type_source="default")])
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=None, goal_map=_GM)
    assert out.loc[0, "type_id"] == _CS and pd.isna(out.loc[0, "type_id_events"])
    assert bool(out.loc[0, "type_tracking_disagrees"]) is True
    assert not out["refine_created"].any()


def test_subtype_round_trip(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_action(0, _CS, type_source="default")])
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P, goal_map=_GM)
    assert out.loc[0, "type_id"] == _CC  # non-vacuity: the rewrite happened before it is undone
    # unrefine the subtype rewrite: restore type_id / type_source from *_events.
    restored = out.copy()
    m = restored["type_id_events"].notna()
    restored.loc[m, "type_id"] = restored.loc[m, "type_id_events"].astype("int64")
    restored.loc[m, "type_source"] = restored.loc[m, "type_source_events"]
    assert restored.loc[0, "type_id"] == _CS and restored.loc[0, "type_source"] == "default"


def test_subtype_rule_reads_direction_from_goal_map_both_sides(monkeypatch):
    # Section 11.2 / ADR-055: the PA is the ATTACKED one. The ball at x = 90 is in the +x PA: crossed when H
    # attacks +x, short when H attacks -x (its attacked PA is then x <= 16.5) -- the same frames, mirrored map.
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_action(0, _FKS, type_source="default")])
    fwd = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P, goal_map=_GM)
    mir = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P, goal_map=_GM_MIRROR)
    assert fwd.loc[0, "type_id"] == spadlconfig.actiontype_id["freekick_crossed"]
    assert mir.loc[0, "type_id"] == _FKS and bool(mir.loc[0, "type_tracking_disagrees"]) is False
    near = _frames_ball_in_box().assign(x=15.0)  # the -x PA
    mir2 = refine_restart_types(a, near, provider="skillcorner", data_profile=_P, goal_map=_GM_MIRROR)
    assert mir2.loc[0, "type_id"] == spadlconfig.actiontype_id["freekick_crossed"]


def test_subtype_rule_fails_closed_without_direction_or_ball(monkeypatch):
    # An unresolved direction or an unobserved ball gives NO tracking verdict: no rewrite, disagreement <NA>.
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_action(0, _CS, type_source="default")])
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P, goal_map=_GM_NONE)
    assert out.loc[0, "type_id"] == _CS and pd.isna(out.loc[0, "type_tracking_disagrees"])
    late = _frames_ball_in_box(t0=30.0)  # no ball frame in [10, 16]
    out2 = refine_restart_types(a, late, provider="skillcorner", data_profile=_P, goal_map=_GM)
    assert out2.loc[0, "type_id"] == _CS and pd.isna(out2.loc[0, "type_tracking_disagrees"])


def _at(row, x, y):
    row = dict(row)
    row["start_x"], row["start_y"] = x, y
    return row


def test_b8_merge_captures_dropped_row(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame(
        [
            _action(0, _FKS, type_source="inferred", t=10.0, provenance="derived"),
            _at(_action(1, _SHOT, type_source="native", t=10.5, provenance="native"), 94.5, 34.0),
        ]
    )
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
    assert len(out) == 1  # the derived free-kick row merged into the shot
    row = out.iloc[0]
    assert row["type_id"] == _SP and row["type_source"] == "inferred"
    assert row["type_id_events"] == _SHOT  # the shot's prior type (round-trips)
    assert row["merged_from_derived_action_id"] == 0 and row["merged_derived_type_id"] == _FKS


def _tracking_evidence(row, *, taker="h7", ox=80.0, oy=30.0):
    """A derived free-kick row carrying the GATED taker + origin the section 11.4 tracking rule needs."""
    row = dict(row)
    row.update(
        taker_player_id=taker,
        taker_source="tracking",
        origin_evidence="transfer",
        enriched_start_x=ox,
        enriched_start_y=oy,
    )
    return row


def test_b8_merge_off_spot_without_tracking_evidence_does_not_merge(monkeypatch):
    # Spec 11.4: off the spot, the merge needs the TRACKING rule (gated taker == shooter, gated origin near the
    # shot start). Without it there is no merge -- never "any FK + shot" (B review).
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame(
        [
            _action(0, _FKS, type_source="inferred", t=10.0, provenance="derived"),
            _at(_action(1, _SHOT, type_source="native", t=10.5), 80.0, 30.0),
        ]
    )
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
    assert len(out) == 2 and _SHOT in set(out["type_id"])  # no merge (the FK row may be subtype-refined)
    assert out["merged_from_derived_action_id"].isna().all()


def test_b8_merge_tracking_form_keeps_the_shot_type(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame(
        [
            _tracking_evidence(_action(0, _FKS, type_source="inferred", t=10.0, provenance="derived")),
            _at(_action(1, _SHOT, type_source="native", t=10.5), 80.5, 30.0),
        ]
    )
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
    assert len(out) == 1 and out.iloc[0]["type_id"] == _SHOT  # C-34: the tracking form keeps the shot type
    assert out.iloc[0]["merged_from_derived_action_id"] == 0


def test_b8_merge_tracking_form_both_sides(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    for taker, shot_x, merges in (("h7", 82.0, True), ("h7", 82.01, False), ("h8", 80.0, False)):
        a = pd.DataFrame(
            [
                _tracking_evidence(_action(0, _FKS, type_source="inferred", t=10.0, provenance="derived"), taker=taker),
                _at(_action(1, _SHOT, type_source="native", t=10.5), shot_x, 30.0),
            ]
        )
        out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
        assert (len(out) == 1) is merges, (taker, shot_x)


def test_b8_merge_events_first_spot_boundary_both_sides(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    for dx, merged in ((2.0, True), (2.01, False)):
        a = pd.DataFrame(
            [
                _action(0, _FKS, type_source="inferred", t=10.0, provenance="derived"),
                _at(_action(1, _SHOT, type_source="native", t=10.5), 94.0 + dx, 34.0),
            ]
        )
        out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
        assert bool(len(out) == 1 and out.iloc[0]["type_id"] == _SP) is merged


def test_b8_merge_cells_are_per_form(monkeypatch):
    # Each form has its own gate cell: an events_first-only table merges the spot form, not the tracking form.
    def events_first_only(profile, rt, sig, cs, **k):
        return _pass() if (sig != "merge" or cs == "events_first") else None

    monkeypatch.setattr(rr, "gate", events_first_only)
    a = pd.DataFrame(
        [
            _tracking_evidence(_action(0, _FKS, type_source="inferred", t=10.0, provenance="derived")),
            _at(_action(1, _SHOT, type_source="native", t=10.5), 80.0, 30.0),
        ]
    )
    assert len(refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)) == 2


def test_b8_merge_round_trip_restores_both_rows(monkeypatch):
    # Spec 19: the round trip restores the shot (type_id_events) and the dropped derived row (merged_derived_*),
    # including its gi_* frame facts, original_event_id and repair_kind (B-19 / B-42).
    monkeypatch.setattr(rr, "gate", _pass)
    derived = _action(0, _FKS, type_source="inferred", t=10.0, provenance="derived")
    derived.update(
        original_event_id="ev-17",
        repair_kind="restart_derived",
        gi_marker_possession_frame_end=1200,
        gi_next_possession_frame_start=1260,
        gi_attacking_side="left_to_right",
    )
    a = pd.DataFrame([derived, _at(_action(1, _SHOT, type_source="native", t=10.5), 94.0, 34.0)])
    m = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P).iloc[0]
    assert (int(m["type_id_events"]), m["type_source_events"]) == (_SHOT, "native")
    assert int(m["merged_derived_type_id"]) == _FKS and float(m["merged_derived_time_seconds"]) == 10.0
    assert m["merged_derived_type_source"] == "inferred" and m["merged_derived_original_event_id"] == "ev-17"
    assert m["merged_derived_repair_kind"] == "restart_derived"
    assert int(m["merged_derived_gi_marker_possession_frame_end"]) == 1200
    assert int(m["merged_derived_gi_next_possession_frame_start"]) == 1260
    assert m["merged_derived_gi_attacking_side"] == "left_to_right"


def test_b8_merge_requires_the_restarting_team(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame(
        [
            _action(0, _FKS, type_source="inferred", t=10.0, provenance="derived", team="H"),
            _at(_action(1, _SHOT, type_source="native", t=10.5, team="A"), 94.0, 34.0),
        ]
    )
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
    assert len(out) == 2  # opponent shot: not the population, no merge


def _pen(release=20.0, outcome: object = "off_target", attribution: object = "missed", keeper="a1", team="H"):
    return pd.DataFrame(
        [
            {
                "game_id": "g",
                "period_id": 1,
                "release_time": release,
                "attacking_team_id": team,
                "taker_player_id": "h9",
                "keeper_player_id": keeper,
                "outcome": outcome,
                "attribution": attribution,
            }
        ]
    )


def _recon_goal(action_id=5, t=60.0):
    # The Part-1 reconciliation goal row: shot_penalty, success, the constant spot start.
    return _at(_action(action_id, _SP, type_source=pd.NA, t=t, provenance="derived"), 94.0, 34.0)


def _other_team_row():
    return _action(9, _CS, type_source="native", t=1.0, team="A")


def test_crosscheck_agree_adds_no_row_and_no_retype(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_recon_goal()])
    out = refine_restart_types(
        a,
        _frames_ball_in_box(),
        provider="skillcorner",
        penalties=_pen(outcome="goal", attribution="goal"),
        data_profile=_P,
    )
    assert len(out) == 1 and out.iloc[0]["type_id"] == _SP
    assert out.iloc[0]["penalty_crosscheck"] == "agree" and out.iloc[0]["penalty_outcome"] == "goal"
    assert not out["refine_created"].any()


def test_crosscheck_rebound_ships_missed_row_and_retypes_goal(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_recon_goal(), _other_team_row()])
    out = refine_restart_types(
        a,
        _frames_ball_in_box(),
        provider="skillcorner",
        penalties=_pen(outcome="saved", attribution="rebound"),
        data_profile=_P,
    )
    goal = out[out["type_id_events"] == _SP].iloc[0]
    assert goal["type_id"] == _SHOT and pd.isna(goal["start_x"]) and pd.isna(goal["start_y"])
    assert goal["start_x_events"] == 94.0 and goal["start_y_events"] == 34.0
    missed = out[(out["type_id"] == _SP) & out["refine_created"]].iloc[0]
    assert missed["penalty_crosscheck"] == "rebound" and missed["result_id"] == spadlconfig.result_id["fail"]
    # the rebound retype round-trips
    restored = goal.copy()
    restored["type_id"], restored["start_x"], restored["start_y"] = (
        int(goal["type_id_events"]),
        goal["start_x_events"],
        goal["start_y_events"],
    )
    assert (restored["type_id"], restored["start_x"], restored["start_y"]) == (_SP, 94.0, 34.0)


def test_crosscheck_later_goal_retypes_the_goal_row(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_recon_goal()])
    out = refine_restart_types(
        a,
        _frames_ball_in_box(),
        provider="skillcorner",
        penalties=_pen(outcome="off_target", attribution="later_goal"),
        data_profile=_P,
    )
    assert (out["type_id_events"] == _SP).sum() == 1
    assert out.loc[out["refine_created"], "penalty_crosscheck"].iloc[0] == "later_goal"


def test_crosscheck_disagree_when_unattributed(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_recon_goal()])
    out = refine_restart_types(
        a,
        _frames_ball_in_box(),
        provider="skillcorner",
        penalties=_pen(outcome=pd.NA, attribution=pd.NA),
        data_profile=_P,
    )
    assert len(out) == 1 and out.iloc[0]["type_id"] == _SP  # no row, no retype
    assert out.iloc[0]["penalty_crosscheck"] == "disagree"


def test_no_retype_without_the_attribution_gate(monkeypatch):
    def det_only(profile, rt, sig, cs, **k):
        return _pass() if cs == "detection" else None

    monkeypatch.setattr(rr, "gate", det_only)
    a = pd.DataFrame([_recon_goal()])
    out = refine_restart_types(
        a, _frames_ball_in_box(), provider="skillcorner", penalties=_pen(attribution="rebound"), data_profile=_P
    )
    assert out.iloc[0]["type_id"] == _SP and not out["refine_created"].any()


def test_saved_adds_keeper_save_and_split_gate_fail_gives_missed(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_other_team_row()])
    out = refine_restart_types(
        a, _frames_ball_in_box(), provider="skillcorner", penalties=_pen(outcome="saved"), data_profile=_P
    )
    ks = out[out["type_id"] == spadlconfig.actiontype_id["keeper_save"]]
    assert len(ks) == 1 and ks.iloc[0]["player_id"] == "a1" and ks.iloc[0]["team_id"] == "A"
    assert bool(ks.iloc[0]["refine_created"]) and ks.iloc[0]["result_source"] == "inferred"

    def no_split(profile, rt, sig, cs, **k):
        return None if cs == "split" else _pass()

    monkeypatch.setattr(rr, "gate", no_split)
    out2 = refine_restart_types(
        a, _frames_ball_in_box(), provider="skillcorner", penalties=_pen(outcome="saved"), data_profile=_P
    )
    assert (out2["type_id"] == spadlconfig.actiontype_id["keeper_save"]).sum() == 0
    assert out2.loc[out2["refine_created"], "penalty_outcome"].iloc[0] == "missed"


def test_existing_keeper_action_suppresses_derived_keeper_save(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    ks = _action(3, spadlconfig.actiontype_id["keeper_save"], type_source="native", t=20.3, team="A", player="a1")
    a = pd.DataFrame([ks])
    out = refine_restart_types(
        a, _frames_ball_in_box(), provider="skillcorner", penalties=_pen(outcome="saved"), data_profile=_P
    )
    assert (out["type_id"] == spadlconfig.actiontype_id["keeper_save"]).sum() == 1


def test_one_kick_one_row_annotates_existing_shot(monkeypatch):
    # C-56/B-52: an existing attacking-team shot at the release is annotated (outcome + provenance only), never a
    # second derived shot_penalty; its result_id / type_id stay unchanged.
    monkeypatch.setattr(rr, "gate", _pass)
    shot = _at(_action(4, _SHOT, type_source="native", t=20.1, result="fail"), 70.0, 10.0)  # noisy far location
    out = refine_restart_types(
        pd.DataFrame([shot]), _frames_ball_in_box(), provider="skillcorner", penalties=_pen(), data_profile=_P
    )
    assert len(out) == 1 and not out["refine_created"].any()
    assert out.iloc[0]["type_id"] == _SHOT and out.iloc[0]["result_id"] == spadlconfig.result_id["fail"]
    assert out.iloc[0]["penalty_outcome"] == "off_target" and out.iloc[0]["penalty_evidence"] == "transfer"


def test_native_provider_annotates_only_never_adds(monkeypatch):
    # Spec 11.3: on a native provider the detector never adds rows; it annotates the native penalty shot within
    # the tau_event ceiling of the release (B-52: result_id / type_id unchanged).
    monkeypatch.setattr(rr, "gate", _pass)
    native = _at(_action(2, _SP, type_source="native", t=20.2, result="fail"), 94.0, 34.0)
    out = refine_restart_types(
        pd.DataFrame([native]),
        _frames_ball_in_box(),
        provider="sportec",
        penalties=_pen(outcome="saved"),
        data_profile=_P,
    )
    assert len(out) == 1 and not out["refine_created"].any()
    assert out.iloc[0]["penalty_outcome"] == "saved" and out.iloc[0]["result_id"] == spadlconfig.result_id["fail"]
    # a release outside tau_event (0.40 s) -> no annotation
    far = _at(_action(2, _SP, type_source="native", t=20.5, result="fail"), 94.0, 34.0)
    out2 = refine_restart_types(
        pd.DataFrame([far]), _frames_ball_in_box(), provider="sportec", penalties=_pen(outcome="saved"), data_profile=_P
    )
    assert pd.isna(out2.iloc[0]["penalty_outcome"]) and len(out2) == 1


def test_penalty_outcome_closed_vocabulary(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    out = refine_restart_types(
        pd.DataFrame([_other_team_row()]),
        _frames_ball_in_box(),
        provider="skillcorner",
        penalties=_pen(),
        data_profile=_P,
    )
    assert set(out["penalty_outcome"].dropna()) <= {"goal", "saved", "off_target", "missed"}


def test_penalty_row_added_is_refine_created_and_excluded(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_action(0, _CS, type_source="default", t=5.0)])
    pens = pd.DataFrame(
        [
            {
                "game_id": "g",
                "period_id": 1,
                "release_time": 20.0,
                "attacking_team_id": "H",
                "taker_player_id": "h9",
                "outcome": "off_target",
            }
        ]
    )
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", penalties=pens, data_profile=_P)
    pen_rows = out[out["type_id"] == _SP]
    assert len(pen_rows) == 1
    assert bool(pen_rows.iloc[0]["refine_created"]) is True
    assert pen_rows.iloc[0]["result_source"] == "inferred" and pen_rows.iloc[0]["penalty_outcome"] == "off_target"


def test_idempotent(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame([_action(0, _CS, type_source="default")])
    out1 = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
    out2 = refine_restart_types(out1, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
    assert list(out2["type_id"]) == list(out1["type_id"])
    assert out2["refine_created"].sum() == out1["refine_created"].sum()


def test_b8_merge_tracking_form_computes_the_gated_origin_itself(monkeypatch):
    # The restart pipeline runs refine BEFORE add_restart_coordinates, so the merge computes the section 10.1 gated
    # origin itself (gated_origin): here the stationary_window tier from the resolver's action-frame rest median.
    import silly_kicks.tracking._restart_validation as rv

    monkeypatch.setattr(rr, "gate", _pass)
    monkeypatch.setattr(rv, "gate", _pass)
    fk = _action(0, _FKS, type_source="inferred", t=10.0, provenance="derived")
    fk.update(taker_player_id="h7", taker_source="tracking", kick_rest_x=80.0, kick_rest_y=30.0)
    a = pd.DataFrame([fk, _at(_action(1, _SHOT, type_source="native", t=10.5), 80.5, 30.0)])
    out = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P)
    assert len(out) == 1 and out.iloc[0]["type_id"] == _SHOT and out.iloc[0]["merged_from_derived_action_id"] == 0
    # the rest median 2.5 m off the shot start -> outside merge_r_m -> no merge
    fk.update(kick_rest_x=83.0)
    out2 = refine_restart_types(
        pd.DataFrame([fk, _at(_action(1, _SHOT, type_source="native", t=10.5), 80.5, 30.0)]),
        _frames_ball_in_box(),
        provider="skillcorner",
        data_profile=_P,
    )
    assert len(out2) == 2


# ---- Plan Task 2.7 Step 1 (B-14): the every-column round trip -- `unrefine` is the executable statement of the
# section 3 reversibility scope. ----

_KICK_VALUES = {
    "kick_bracket_lo": 29.0,
    "kick_bracket_hi": 31.0,
    "t_kick": 30.1,
    "kick_time_source": "tf79_contact",
    "taker_player_id": "h9",
}


def _with_kick_columns(row, values=None):
    """Every RESOLVE_KICKS_COLUMNS column present (NA unless given): refine runs after resolve_kicks."""
    from silly_kicks.tracking._restart_validation import RESOLVE_KICKS_COLUMNS

    row = dict(row)
    for c in RESOLVE_KICKS_COLUMNS:
        row[c] = (values or {}).get(c, pd.NA)
    return row


def _derived_fk(action_id, t):
    # The Part-1 derivation constants: player NaN, start NaN, action_provenance "derived", the restarting team.
    row = _action(action_id, _FKS, type_source="inferred", t=t, provenance="derived", player=pd.NA)
    row.update(start_x=np.nan, start_y=np.nan, restart_evidence="tagged", repair_kind="restart_derived")
    return _with_kick_columns(row, _KICK_VALUES)


def _mixed_fixture() -> pd.DataFrame:
    corner = _with_kick_columns(_action(0, _CS, type_source="default", t=10.0))  # subtype-rewritten (ball in box)
    shot = _with_kick_columns(_at(_action(2, _SHOT, type_source="native", t=30.5), 94.0, 34.0))  # merged (spot form)
    goal = _recon_goal(action_id=3, t=60.0)  # the Part-1 reconciliation goal: rebound-retyped + annotated
    goal.update(repair_kind="score_reconciliation")
    rows = [corner, _derived_fk(1, 30.0), shot, _with_kick_columns(goal)]
    from silly_kicks.tracking._restart_validation import RESTART_REFINEMENT_COLUMNS

    out = pd.DataFrame(rows)
    # built as object from the rows: under pandas 3 the inferred `str` column would turn the derived row's <NA> into NaN
    out["player_id"] = pd.Series([r["player_id"] for r in rows], dtype=object)
    for c, dtype in RESTART_REFINEMENT_COLUMNS.items():  # the resolver's declared dtypes (refine runs after it)
        if c in out.columns:
            col = pd.to_numeric(out[c], errors="coerce") if dtype == "float64" else out[c]
            out[c] = col.astype(cast(Any, dtype))  # a registry dtype NAME string (stubs type only dtype objects)
    return out


def unrefine(out: pd.DataFrame, original_columns) -> pd.DataFrame:
    """Undo refine_restart_types (spec section 3 / plan Task 2.7): drop refine-created rows; restore type / source /
    start from the *_events columns; split each merged row back into the shot and the derived row (the derived row
    from merged_derived_* + its kick columns + the Part-1 constants); restore the pre-renumber action ids."""
    from silly_kicks.tracking._restart_validation import RESOLVE_KICKS_COLUMNS

    back = out[~out["refine_created"].fillna(False).astype(bool)].copy()
    has_type = back["type_id_events"].notna()
    back.loc[has_type, "type_id"] = back.loc[has_type, "type_id_events"].astype("int64")
    back.loc[has_type, "type_source"] = back.loc[has_type, "type_source_events"]
    for c in ("start_x", "start_y"):
        has = back[f"{c}_events"].notna()
        back.loc[has, c] = back.loc[has, f"{c}_events"]
    derived_rows = []
    for i in back.index[back["merged_from_derived_action_id"].notna()]:
        m = back.loc[i]
        d: dict[str, object] = {c: pd.NA for c in original_columns}
        d.update(
            game_id=m["game_id"],
            period_id=m["period_id"],
            team_id=m["team_id"],
            action_id=int(m["merged_from_derived_action_id"]),
            time_seconds=float(m["merged_derived_time_seconds"]),
            type_id=int(m["merged_derived_type_id"]),
            type_source=m["merged_derived_type_source"],
            result_id=int(m["merged_derived_result_id"]),
            result_source=m["merged_derived_result_source"],
            restart_evidence=m["merged_derived_restart_evidence"],
            repair_kind=m["merged_derived_repair_kind"],
            action_provenance="derived",
            player_id=pd.NA,
            start_x=np.nan,
            start_y=np.nan,
            end_x=float(m["merged_derived_end_x"]),
            end_y=float(m["merged_derived_end_y"]),
        )
        for c in RESOLVE_KICKS_COLUMNS:
            if c in original_columns:
                d[c] = m[c]
                back.at[i, c] = pd.NA  # a shot row carries no kick columns of its own
        back.at[i, "restart_evidence"] = pd.NA
        back.at[i, "action_id"] = int(m["merged_from_shot_action_id"])
        derived_rows.append(d)
    plain = back["merged_from_derived_action_id"].isna()
    back.loc[plain, "action_id"] = -1  # resolved below from the chronological order
    back = pd.concat([back[list(original_columns)], pd.DataFrame(derived_rows)], ignore_index=True)
    back = back.sort_values(["period_id", "time_seconds"], kind="mergesort").reset_index(drop=True)
    known = set(back.loc[back["action_id"] != -1, "action_id"].astype(int))
    free = iter(sorted(set(range(len(back))) - known))
    back["action_id"] = [int(a) if a != -1 else next(free) for a in back["action_id"]]
    return back


def test_round_trip_reconstructs_every_column(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = _mixed_fixture()
    out = refine_restart_types(
        a,
        _frames_ball_in_box(),
        provider="skillcorner",
        penalties=_pen(release=59.0, outcome="saved", attribution="rebound"),
        data_profile=_P,
        goal_map=_GM,
    )
    # non-vacuity: every refine branch the round trip must undo actually fired
    assert out["refine_created"].any()  # the missed penalty row
    assert (out["type_id_events"] == _CS).any() and (out["type_id_events"] == _SP).any()  # subtype + rebound retype
    merged = out[out["merged_from_derived_action_id"].notna()].iloc[0]
    assert merged["restart_evidence"] == "tagged" and float(merged["t_kick"]) == 30.1  # section 11.4 / 11.5 carry
    assert merged["taker_player_id"] == "h9" and merged["type_source"] == "inferred"  # events-first form
    back = unrefine(out, list(a.columns))
    for c in a.columns:  # restore the input schema's dtypes (the concat of re-inserted rows widens them)
        col = pd.to_numeric(back[c], errors="coerce") if str(a[c].dtype) == "float64" else back[c]
        back[c] = col.astype(a[c].dtype)
    pd.testing.assert_frame_equal(
        back[list(a.columns)].reset_index(drop=True),
        a.sort_values(["period_id", "time_seconds"]).reset_index(drop=True),
    )


def test_b8_merge_tracking_form_sets_type_source_tracking(monkeypatch):
    monkeypatch.setattr(rr, "gate", _pass)
    a = pd.DataFrame(
        [
            _tracking_evidence(_action(0, _FKS, type_source="inferred", t=10.0, provenance="derived")),
            _at(_action(1, _SHOT, type_source="native", t=10.5), 80.5, 30.0),
        ]
    )
    m = refine_restart_types(a, _frames_ball_in_box(), provider="skillcorner", data_profile=_P).iloc[0]
    assert m["type_source"] == "tracking" and m["type_source_events"] == "native"  # C-34; the shot's prior source
