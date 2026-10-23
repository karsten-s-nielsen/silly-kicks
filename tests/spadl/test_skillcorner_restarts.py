"""Task 2.3 — SkillCorner marker-keyed derived restarts (spec section 5).

Synthetic possession frames, each rule from both sides and non-vacuous. ``derive_restarts`` consumes a
chronologically-sorted ``player_possession`` frame carrying SPADL-frame start coordinates and the GI
marker/tag columns, and emits one derived restart row per restart.
"""

import json
from pathlib import Path

import pandas as pd

from silly_kicks.spadl import config as spadlcfg
from silly_kicks.spadl._skillcorner_restarts import derive_restarts
from silly_kicks.spadl.skillcorner import convert_to_actions

_META = {"id": "M1", "home_team": {"id": "100"}}
_OPP = "200"

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "skillcorner"


def _pp(rows: list[dict]) -> pd.DataFrame:
    """Build a possession frame; each row dict overrides the per-possession defaults."""
    out = []
    for k, r in enumerate(rows):
        base = {
            "event_id": f"e{k}",
            "period_id": 1,
            "time_seconds": 10.0 * (k + 1),
            "team_id": "100",
            "start_type": "pass_reception",
            "game_interruption_before": "",
            "game_interruption_after": "",
            "start_x": 50.0,
            "start_y": 34.0,
        }
        base.update(r)
        out.append(base)
    return pd.DataFrame(out)


def _tid(name: str) -> int:
    return spadlcfg.actiontype_id[name]


def test_corner_for_emits_one_won_corner():
    pp = _pp(
        [
            {"team_id": "100", "game_interruption_after": "corner_for"},
            {"team_id": "100", "game_interruption_before": "corner_for", "start_x": 95.0, "start_y": 34.0},
        ]
    )
    out = derive_restarts(pp, _OPP, _META)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["type_id"] == _tid("corner_crossed")
    assert row["team_id"] == "100"
    assert row["result_id"] == spadlcfg.result_id["success"]
    assert row["restart_evidence"] == "tagged"
    assert row["type_source"] == "default"  # won corner
    assert row["action_provenance"] == "derived"
    assert pd.isna(row["player_id"]) and pd.isna(row["start_x"])
    assert row["end_x"] == 95.0 and row["end_y"] == 34.0  # same team -> no reflection


def test_corner_against_awards_restart_to_other_team():
    # A corner awarded to 100 reads as corner_against on 200's marker possession and corner_for on
    # 100's own receiving possession (each relative to its own team).
    pp = _pp(
        [
            {"team_id": "200", "game_interruption_after": "corner_against"},
            {"team_id": "100", "game_interruption_before": "corner_for", "start_x": 95.0, "start_y": 34.0},
        ]
    )
    out = derive_restarts(pp, _OPP, _META)
    assert len(out) == 1
    assert out.iloc[0]["team_id"] == "100"  # other(200) = home 100


def test_lost_corner_reflects_end_and_is_inferred():
    # corner_against on 100 -> restarting team 200; receiver is 100 (opponent first touch, lost).
    pp = _pp(
        [
            {"team_id": "100", "game_interruption_after": "corner_against"},
            {"team_id": "100", "game_interruption_before": "corner_against", "start_x": 95.0, "start_y": 20.0},
        ]
    )
    out = derive_restarts(pp, _OPP, _META)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["team_id"] == "200"
    assert row["type_source"] == "inferred"  # lost corner
    assert row["result_id"] == spadlcfg.result_id["fail"]
    # next team (100) differs from restarting (200) -> 180-degree point reflection.
    assert row["end_x"] == 105.0 - 95.0 and row["end_y"] == 68.0 - 20.0


def test_period_end_marker_and_penalty_marker_emit_none():
    # Marker on the last possession of period 1 (half-time carry-over) + a penalty marker.
    pp = _pp(
        [
            {"team_id": "100", "game_interruption_after": "corner_for"},
            {"team_id": "100", "period_id": 2, "game_interruption_before": "corner_for"},
        ]
    )
    assert len(derive_restarts(pp, _OPP, _META)) == 0

    pp_pen = _pp(
        [
            {"team_id": "100", "game_interruption_after": "penalty_for"},
            {"team_id": "100", "start_type": "unknown"},
        ]
    )
    assert len(derive_restarts(pp_pen, _OPP, _META)) == 0


def test_goal_tag_and_first_of_period_and_kickoff_emit_none():
    pp = _pp(
        [
            {"team_id": "100", "game_interruption_after": "goal_for"},  # goal, not a restart
            {"team_id": "200", "start_type": "unknown"},  # post-goal kick-off (no restart tag)
            {"team_id": "200", "game_interruption_before": "goal_for"},  # goal tag, not a restart
        ]
    )
    assert len(derive_restarts(pp, _OPP, _META)) == 0


def test_no_marker_tag_only_restart():
    # A restart tag with no explaining marker on the previous possession -> one tag_only row.
    pp = _pp(
        [
            {"team_id": "100", "start_type": "pass_reception"},  # no marker
            {"team_id": "100", "game_interruption_before": "throw_in_for", "start_x": 60.0, "start_y": 10.0},
            {"team_id": "100", "start_x": 62.0, "start_y": 12.0},
        ]
    )
    out = derive_restarts(pp, _OPP, _META)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["type_id"] == _tid("throw_in")
    assert row["restart_evidence"] == "tag_only"
    assert row["type_source"] == "native"
    assert row["bodypart_id"] == spadlcfg.bodypart_id["other"]  # throw-in


def test_chain_emits_first_marker_and_second_tag_only():
    # free_kick_for marker on poss0; poss1 is a DIFFERENT restart (corner for 200) -> chain.
    pp = _pp(
        [
            {"team_id": "100", "game_interruption_after": "free_kick_for"},
            {"team_id": "200", "game_interruption_before": "corner_for", "start_x": 95.0, "start_y": 34.0},
            {"team_id": "200", "start_x": 96.0, "start_y": 34.0},
        ]
    )
    out = derive_restarts(pp, _OPP, _META).reset_index(drop=True)
    assert len(out) == 2
    first = out[out["restart_evidence"].isin(["chain", "chain_ambiguous"])]
    assert len(first) == 1
    f = first.iloc[0]
    # second restart (corner) awarded to 200 != T(100) -> chain_ambiguous, fail; end NaN.
    assert f["restart_evidence"] == "chain_ambiguous"
    assert f["result_id"] == spadlcfg.result_id["fail"]
    assert pd.isna(f["end_x"])
    second = out[out["restart_evidence"] == "tag_only"]
    assert len(second) == 1
    assert second.iloc[0]["type_id"] == _tid("corner_crossed")
    assert second.iloc[0]["team_id"] == "200"


def test_freekick_box_rule_crossed_vs_short():
    crossed = _pp(
        [
            {"team_id": "100", "game_interruption_after": "free_kick_for"},
            {"team_id": "100", "game_interruption_before": "free_kick_for", "start_x": 95.0, "start_y": 34.0},
        ]
    )
    row = derive_restarts(crossed, _OPP, _META).iloc[0]
    assert row["type_id"] == _tid("freekick_crossed")
    assert row["type_source"] == "inferred"

    short = _pp(
        [
            {"team_id": "100", "game_interruption_after": "free_kick_for"},
            {"team_id": "100", "game_interruption_before": "free_kick_for", "start_x": 50.0, "start_y": 34.0},
        ]
    )
    assert derive_restarts(short, _OPP, _META).iloc[0]["type_id"] == _tid("freekick_short")


def test_result_source_native_vs_inferred():
    native = _pp(
        [
            {"team_id": "100", "game_interruption_after": "goal_kick_for"},
            {"team_id": "100", "game_interruption_before": "goal_kick_for", "start_type": "pass_reception"},
        ]
    )
    assert derive_restarts(native, _OPP, _META).iloc[0]["result_source"] == "native"

    inferred = _pp(
        [
            {"team_id": "100", "game_interruption_after": "goal_kick_for"},
            {"team_id": "100", "game_interruption_before": "goal_kick_for", "start_type": "unknown"},
        ]
    )
    assert derive_restarts(inferred, _OPP, _META).iloc[0]["result_source"] == "inferred"


def test_empty_frame_returns_columns():
    out = derive_restarts(_pp([]), _OPP, _META)
    assert len(out) == 0
    assert "restart_evidence" in out.columns and "type_source" in out.columns


def test_derived_time_is_strictly_inside_the_marker_bracket():
    # Marker possession at t=10, receiver at t=20 -> the derived restart time is strictly inside (10, 20),
    # i.e. after the marker and before its receiver (spec section 5.4 ordering invariant / G4).
    pp = _pp(
        [
            {"team_id": "100", "game_interruption_after": "corner_for", "time_seconds": 10.0},
            {"team_id": "100", "game_interruption_before": "corner_for", "time_seconds": 20.0, "start_x": 95.0},
        ]
    )
    t = float(derive_restarts(pp, _OPP, _META).iloc[0]["time_seconds"])
    assert 10.0 < t < 20.0


# ---------------------------------------------------------------------------
# Task 2.3 — GI facts on derived and native rows (spec section 17.2, CL-1 / CL-13 / B-11)
# ---------------------------------------------------------------------------


def test_derived_row_carries_gi_bracket_and_possession_facts():
    pp = _pp(
        [
            {
                "team_id": "100",
                "game_interruption_after": "corner_for",
                "event_id": "m0",
                "frame_start": 1000,
                "frame_end": 1050,
                "attacking_side": "left_to_right",
            },
            {
                "team_id": "100",
                "game_interruption_before": "corner_for",
                "start_x": 95.0,
                "start_y": 34.0,
                "event_id": "m1",
                "frame_start": 1080,
                "frame_end": 1120,
                "attacking_side": "left_to_right",
            },
        ]
    )
    row = derive_restarts(pp, _OPP, _META).iloc[0]
    assert row["gi_marker_possession_frame_end"] == 1050  # marker possession frame_end (bracket lo)
    assert row["gi_next_possession_frame_start"] == 1080  # next possession frame_start (bracket hi)
    assert row["gi_possession_frame_start"] == 1000
    assert row["gi_possession_frame_end"] == 1050
    assert row["gi_event_id"] == "m0"
    assert row["gi_attacking_side"] == "left_to_right"


def test_chain_marker_row_has_na_next_possession_frame():
    # poss0 corner marker is chained (poss1 carries a different restart tag) -> no possession lies between.
    pp = _pp(
        [
            {
                "team_id": "100",
                "game_interruption_after": "corner_for",
                "event_id": "m0",
                "frame_start": 1000,
                "frame_end": 1050,
                "attacking_side": "ltr",
            },
            {
                "team_id": "100",
                "game_interruption_before": "free_kick_for",
                "start_x": 60.0,
                "start_y": 34.0,
                "event_id": "m1",
                "frame_start": 1080,
                "frame_end": 1120,
                "attacking_side": "ltr",
            },
            {"team_id": "100", "start_x": 62.0, "start_y": 34.0, "event_id": "m2", "frame_start": 1200},
        ]
    )
    out = derive_restarts(pp, _OPP, _META)
    corner = out[out["type_id"] == _tid("corner_crossed")].iloc[0]
    assert pd.isna(corner["gi_next_possession_frame_start"])
    assert corner["gi_marker_possession_frame_end"] == 1050


def test_derived_rows_na_frame_cols_when_input_lacks_gi_columns():
    pp = _pp(
        [
            {"team_id": "100", "game_interruption_after": "corner_for", "event_id": "e0"},
            {"team_id": "100", "game_interruption_before": "corner_for", "start_x": 95.0, "start_y": 34.0},
        ]
    )
    row = derive_restarts(pp, _OPP, _META).iloc[0]
    for c in (
        "gi_marker_possession_frame_end",
        "gi_next_possession_frame_start",
        "gi_attacking_side",
        "gi_possession_frame_start",
        "gi_possession_frame_end",
    ):
        assert pd.isna(row[c])
    assert row["gi_event_id"] == "e0"  # the possession's own event_id is always present


def _load_derived_events() -> tuple[pd.DataFrame, dict]:
    events = pd.read_csv(_FIXTURE_DIR / "derived_actions.csv", low_memory=False)
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        meta = json.load(f)
    return events, meta


def test_native_rows_carry_possession_gi_facts():
    events, meta = _load_derived_events()
    events["frame_start"] = 2000
    events["frame_end"] = 2050
    events["attacking_side"] = "home_ltr"
    actions, _ = convert_to_actions(events, meta)
    native = actions[actions["action_provenance"] == "native"]
    assert len(native) > 0
    assert (native["gi_attacking_side"] == "home_ltr").all()
    assert (native["gi_possession_frame_start"] == 2000).all()
    assert (native["gi_possession_frame_end"] == 2050).all()
    assert native["gi_event_id"].notna().all()
    # the bracket ids are derived-restart-only -> <NA> on native rows
    assert native["gi_marker_possession_frame_end"].isna().all()
    assert native["gi_next_possession_frame_start"].isna().all()


def test_convert_is_backward_compatible_without_gi_columns():
    events, meta = _load_derived_events()
    actions, _ = convert_to_actions(events, meta)
    for c in (
        "gi_marker_possession_frame_end",
        "gi_next_possession_frame_start",
        "gi_attacking_side",
        "gi_possession_frame_start",
        "gi_possession_frame_end",
    ):
        assert actions[c].isna().all()
    native = actions[actions["action_provenance"] == "native"]
    assert native["gi_event_id"].notna().all()
