"""SkillCorner derived restart rows (R30, R35; action-time repair spec §4.2, §4.15 carrier, §4.18 key)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import restart_kind
from tests.spadl._sc_restart_scenes import convert, derived_restarts, pp

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id


def _only(rows: pd.DataFrame, kind: str) -> pd.Series:
    sel = rows[rows["type_id"].map(restart_kind) == kind]
    assert len(sel) == 1, (kind, len(sel))
    return sel.iloc[0]


def test_paired_corner_is_one_placeholder_row_on_the_receivers_side():
    a = convert(
        pp("p0", 10, 14, "A", "a1", end=(50.0, -30.0), gia="corner_against"),
        pp("p1", 30, 33, "H", "h1", start=(49.0, 30.0), end=(40.0, 0.0), gib="corner_for"),
    )
    r = derived_restarts(a)
    assert len(r) == 1
    c = r.iloc[0]
    assert c["type_id"] == _T["corner_short"]  # the reception lies within 15 m of the start, by the restart team
    assert c["team_id"] == "H"
    assert pd.isna(c["player_id"])
    assert c["time_source"] == "placeholder"
    assert c["time_seconds"] == pytest.approx(30.0 - 0.02)
    assert c["original_event_id"] == "p1"
    # the side of the previous possession's end: A's (102.5, 4) is H's (2.5, 64) -> the y = 68 touchline
    assert (c["start_x"], c["start_y"]) == (105.0, 68.0)
    assert (c["end_x"], c["end_y"]) == pytest.approx((101.5, 64.0))
    assert c["result_id"] == _R["success"]
    native = a[a["original_event_id"] == "p1"]
    native = native[native["action_provenance"] == "native"]
    assert native["type_id"].tolist() == [_T["pass"]]  # the first possession takes its own type


def test_after_only_corner_then_throw_in():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(50.0, 20.0), gia="corner_for"),
        pp("p1", 60, 62, "A", "a1", start=(10.0, -33.0), end=(15.0, -30.0), gib="throw_in_for"),
    )
    r = derived_restarts(a)
    corner, throw = _only(r, "corner"), _only(r, "throw_in")
    assert corner["time_source"] == "order_only"
    assert corner["time_seconds"] == pytest.approx(60.0 - 0.03)
    assert corner["type_id"] == _T["corner_crossed"]  # no first possession -> crossed
    assert corner["team_id"] == "H"
    assert corner["result_id"] == _R["fail"]
    assert (corner["start_x"], corner["start_y"]) == (105.0, 68.0)
    assert (corner["end_x"], corner["end_y"]) == (corner["start_x"], corner["start_y"])  # ends at its start
    assert throw["time_source"] == "placeholder"
    assert throw["time_seconds"] == pytest.approx(60.0 - 0.02)
    assert throw["team_id"] == "A"
    assert throw["bodypart_id"] == spadlconfig.bodypart_id["other"]
    # it follows an "after"-only restart, so the end of p0 gives it no place
    assert math.isnan(throw["start_x"]) and math.isnan(throw["start_y"])
    assert throw["result_id"] == _R["success"]
    # one possession gives its id to two derived rows of different kinds: the key (id, kind) is unique
    assert corner["original_event_id"] == throw["original_event_id"] == "p1"


def test_before_only_free_kick_from_where_play_stopped():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(0.0, 10.0)),
        pp("p1", 30, 33, "A", "a1", start=(5.0, 12.0), end=(20.0, 0.0), gib="free_kick_for"),
    )
    fk = _only(derived_restarts(a), "free_kick")
    assert fk["team_id"] == "A"
    assert (fk["start_x"], fk["start_y"]) == pytest.approx((52.5, 24.0))  # H's (52.5, 44) in A's frame
    assert fk["type_id"] == _T["freekick_crossed"]  # the reception is 22.6 m away
    assert fk["time_source"] == "placeholder"


def test_goal_kick_at_the_rule_point():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(50.0, 0.0), end_type="shot", gia="goal_kick_against"),
        pp("p1", 30, 33, "A", "a1", start=(-45.0, 5.0), end=(0.0, 0.0), gib="goal_kick_for"),
    )
    gk = _only(derived_restarts(a), "goal_kick")
    assert gk["type_id"] == _T["goalkick"]
    assert (gk["start_x"], gk["start_y"]) == (5.5, 34.0)
    assert gk["team_id"] == "A"


def test_throw_in_on_the_touchline_of_the_previous_end():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(20.0, 33.9), gia="throw_in_against"),
        pp("p1", 30, 33, "A", "a1", start=(-18.0, -30.0), end=(0.0, 0.0), gib="throw_in_for"),
    )
    ti = _only(derived_restarts(a), "throw_in")
    assert ti["team_id"] == "A"
    assert (ti["start_x"], ti["start_y"]) == pytest.approx((32.5, 0.0))


def test_period_end_restart_marker_is_not_counted():
    a = convert(pp("p0", 10, 14, "H", "h1", gia="corner_for"))
    assert len(derived_restarts(a)) == 0


def test_disagreeing_pair_takes_the_before_marker_team():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(0.0, 0.0), gia="free_kick_for"),
        pp("p1", 30, 33, "A", "a1", start=(0.0, 0.0), end=(10.0, 0.0), gib="free_kick_for"),
    )
    r = derived_restarts(a)
    assert len(r) == 1
    assert r.iloc[0]["team_id"] == "A"


@pytest.mark.parametrize(("marker", "result"), [("goal_against", "success"), ("goal_for", "fail")])
def test_after_only_restart_followed_by_a_goal(marker, result):
    # goal_against on A's kick-off possession: A conceded, so H (the corner team) scored
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(50.0, 20.0), gia="corner_for"),
        pp("p1", 80, 82, "A", "a1", start=(0.0, 0.0), end=(-5.0, 0.0), gib=marker),
    )
    corner = _only(derived_restarts(a), "corner")
    assert corner["time_source"] == "order_only"
    assert corner["result_id"] == _R[result]


def test_corner_with_no_known_side_measures_to_the_nearer_flag():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(30.0, 0.0), gia="free_kick_for"),
        pp("p1", 30, 33, "H", "h2", start=(49.0, -30.0), end=(40.0, 0.0), gib="corner_for"),
    )
    corner = _only(derived_restarts(a), "corner")
    assert corner["start_x"] == 105.0 and math.isnan(corner["start_y"])
    assert corner["type_id"] == _T["corner_short"]  # 5.3 m from the (105, 0) flag


def test_preserve_native_list_carries_the_running_score_on_native_rows():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(50.0, 20.0), gia="corner_for", team_score=1, opponent_team_score=0),
        pp("p1", 60, 62, "A", "a1", start=(10.0, -33.0), gib="throw_in_for", team_score=0, opponent_team_score=1),
        preserve_native=["team_score", "opponent_team_score"],
    )
    assert {"team_score", "opponent_team_score"} <= set(a.columns)
    native = a[a["action_provenance"] == "native"].set_index("original_event_id")
    assert native.loc["p0", "team_score"] == 1 and native.loc["p1", "opponent_team_score"] == 1
    assert a.loc[a["action_provenance"] == "derived", "team_score"].isna().all()


def test_default_output_has_no_running_score_columns():
    a = convert(pp("p0", 10, 14, "H", "h1"), pp("p1", 30, 33, "A", "a1"))
    assert "team_score" not in a.columns


def test_preserve_native_true_returns_the_schema_columns_and_does_not_raise():
    # R62: `original_event_id` is a schema column, so True adds no extra column (it raised the
    # `_finalize_output: extra_columns overlap with schema` ValueError before)
    from silly_kicks.spadl.skillcorner import SKILLCORNER_SPADL_COLUMNS

    a = convert(pp("p0", 10, 14, "H", "h1"), pp("p1", 30, 33, "A", "a1"), preserve_native=True)
    assert list(a.columns) == list(SKILLCORNER_SPADL_COLUMNS)
    assert a["original_event_id"].notna().any()


def test_preserve_native_list_with_inferred_defensive_and_keeper_rows():
    # Task 15 finding: every staged SkillCorner match has inferred defensive rows; with the R56 list the native rows
    # carry the running score and the derived rows must carry NaN there (not raise a KeyError on the concat)
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(50.0, 20.0), team_score=0, opponent_team_score=0),
        pp("p1", 20, 23, "A", "a1", start=(40.0, 30.0), start_type="interception", end=(80.0, 34.0),
           end_type="shot", team_score=0, opponent_team_score=0),
        pp("p2", 25, 28, "H", "g1", start=(-50.0, 0.0), start_type="recovery", team_score=0, opponent_team_score=0),
        preserve_native=["team_score", "opponent_team_score"],
    )  # fmt: skip
    derived = a[a["action_provenance"] == "derived"]
    assert len(derived) > 0  # non-vacuity: the scene makes inferred rows
    assert derived["team_score"].isna().all()
    assert a.loc[a["action_provenance"] == "native", "team_score"].notna().all()


@pytest.mark.parametrize("kind", ["corner", "throw_in"])
def test_previous_end_missing_gives_an_unknown_side(kind):
    # RTR-IMPL A03: spec 4.2 defines the unknown side as a NaN start_y; `NaN < 34` is False, so a missing end of `a`
    # put the corner on the y = 68 flag and the throw-in on the y = 68 touchline with no source
    a = convert(
        pp("p0", 10, 14, "A", "a1", end=(np.nan, np.nan), gia=f"{kind}_against"),
        pp("p1", 30, 33, "H", "h1", start=(49.0, 30.0), end=(40.0, 0.0), gib=f"{kind}_for"),
    )
    r = derived_restarts(a)
    assert len(r) == 1  # non-vacuity: the restart row exists
    assert pd.isna(r.iloc[0]["start_y"])


def test_metadata_without_away_team_takes_the_other_team_from_the_events():
    # RTR-IMPL A08: the away team fell back to "" when the metadata had no away_team, so a restart "against" the home
    # team got team "". The other team id of the events is the away team.
    from tests.spadl._sc_restart_scenes import META

    meta = {k: v for k, v in META.items() if k != "away_team"}
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(50.0, -30.0), gia="corner_for"),
        pp("p1", 30, 33, "A", "a1", start=(49.0, 30.0), end=(40.0, 0.0), gib="corner_against"),
        pp("p2", 50, 53, "H", "h2", gia="throw_in_against"),
        pp("p3", 70, 73, "A", "a2"),
        meta=meta,
    )
    teams = set(derived_restarts(a)["team_id"])
    assert "" not in teams
    assert teams <= {"H", "A"} and "A" in teams
