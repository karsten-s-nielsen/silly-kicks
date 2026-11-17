"""SkillCorner derived penalty rows (R58; action-time repair spec §4.17, §4.18 key)."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import restart_kind
from tests.spadl._sc_restart_scenes import META, convert, derived_restarts, pp

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id


def _penalty(actions: pd.DataFrame) -> pd.Series:
    r = derived_restarts(actions)
    sel = r[r["type_id"] == _T["shot_penalty"]]
    assert len(sel) == 1, len(sel)
    return sel.iloc[0]


def test_after_only_penalty_scored_by_the_running_score():
    a = convert(
        pp("p0", 10, 14, "H", "h1", end=(40.0, 5.0), gia="penalty_for", team_score=0, opponent_team_score=0),
        pp("p1", 90, 91, "A", "a1", gib="goal_against", team_score=0, opponent_team_score=1),
    )
    p = _penalty(a)
    assert p["team_id"] == "H"
    assert pd.isna(p["player_id"])
    assert (p["start_x"], p["start_y"]) == (94.0, 34.0)
    assert (p["end_x"], p["end_y"]) == (94.0, 34.0)
    assert p["time_source"] == "order_only"
    assert p["time_seconds"] == pytest.approx(90.0 - 0.03)
    assert p["result_id"] == _R["success"]
    assert p["original_event_id"] == "p1"
    assert p["bodypart_id"] == spadlconfig.bodypart_id["foot"]


def test_after_only_penalty_not_scored():
    a = convert(
        pp("p0", 10, 14, "H", "h1", gia="penalty_for", team_score=0, opponent_team_score=0),
        pp("p1", 90, 91, "A", "a1", gib="corner_against", team_score=0, opponent_team_score=0),
    )
    assert _penalty(a)["result_id"] == _R["fail"]


def test_paired_penalty_is_fail_and_the_rebound_row_is_the_goal():
    a = convert(
        pp("p0", 10, 14, "H", "h1", gia="penalty_for", team_score=0, opponent_team_score=0),
        pp(
            "p1",
            92,
            93,
            "H",
            "h2",
            end_type="shot",
            gib="penalty_for",
            gia="goal_for",
            team_score=1,
            opponent_team_score=0,
        ),
    )
    p = _penalty(a)
    assert p["time_source"] == "placeholder"
    assert p["result_id"] == _R["fail"]
    rebound = a[(a["original_event_id"] == "p1") & (a["action_provenance"] == "native")]
    assert rebound["type_id"].tolist() == [_T["shot"]]
    assert rebound["result_id"].tolist() == [_R["success"]]


def test_goal_marker_fallback_without_the_running_score():
    a = convert(
        pp("p0", 10, 14, "H", "h1", gia="penalty_for"),
        pp("p1", 90, 91, "A", "a1", gib="goal_against"),
    )
    assert _penalty(a)["result_id"] == _R["success"]


def test_before_only_penalty_is_counted_and_fail():
    a = convert(
        pp("p0", 10, 14, "H", "h1"),
        pp("p1", 30, 31, "A", "a1", gib="penalty_for"),
    )
    p = _penalty(a)
    assert p["team_id"] == "A"
    assert p["time_source"] == "placeholder"
    assert p["result_id"] == _R["fail"]


def test_period_end_penalty_is_counted():
    a = convert(
        pp("p0", 10, 14, "H", "h1", gia="penalty_for", team_score=0, opponent_team_score=0),
        pp("p1", 1, 2, "A", "a1", period=2, team_score=0, opponent_team_score=1),
    )
    p = _penalty(a)
    assert p["time_source"] == "order_only"
    assert p["period_id"] == 1
    assert p["time_seconds"] == pytest.approx(14.0 + 0.01)
    assert p["original_event_id"] == "p0"  # no b in its period: the key takes a
    assert p["result_id"] == _R["success"]  # the next possession is the first of the next period


@pytest.mark.parametrize(
    ("meta_scores", "result"), [({"home_team_score": 1, "away_team_score": 0}, "success"), ({}, "fail")]
)
def test_last_period_penalty_with_no_later_event(meta_scores, result):
    meta = {**META, **meta_scores}
    a = convert(
        pp("p0", 10, 14, "H", "h1", period=2, gia="penalty_for", team_score=0, opponent_team_score=0),
        meta=meta,
    )
    assert _penalty(a)["result_id"] == _R[result]


def test_shared_possession_gives_two_keys():
    a = convert(
        pp("p0", 10, 14, "A", "a1", end=(50.0, -30.0), gia="corner_against"),
        pp(
            "p1",
            30,
            33,
            "H",
            "h1",
            start=(49.0, 30.0),
            end=(45.0, 0.0),
            end_type="foul_suffered",
            gib="corner_for",
            gia="penalty_for",
        ),
        pp("p2", 120, 121, "A", "a2", gib="goal_against"),
    )
    r = derived_restarts(a)
    keys = sorted(zip(r["original_event_id"], r["type_id"].map(restart_kind), strict=True))
    assert keys == [("p1", "corner"), ("p2", "penalty")]
