"""Orchestrator compute_team_kpis (TF-52 Task 6): assembly, census, purity, order, companions."""

from __future__ import annotations

import pandas as pd

from silly_kicks.team_metrics import TEAM_KPI_COLUMNS, compute_team_kpis
from tests.team_metrics._helpers import INTERCEPTION, PASS, SHOT, make_actions


def _two_team_fixture(game_id=1):
    return make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 30.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 4.0, "start_x": 75.0},
            {"team_id": 10, "type_id": SHOT, "time_seconds": 5.0, "start_x": 95.0, "xg": 0.30},
            {"team_id": 20, "type_id": PASS, "time_seconds": 8.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 10.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 12.0, "start_x": 40.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 14.0, "start_x": 45.0},
        ],
        game_id=game_id,
    )


def test_two_rows_raw_ids_and_conservation():
    samples, report = compute_team_kpis(_two_team_fixture(), xg_column="xg")
    assert len(samples) == 2
    assert set(samples["team_id"]) == {10, 20}
    assert list(samples.columns) == list(TEAM_KPI_COLUMNS)
    assert report.n_matches_scored + report.n_matches_excluded_not_two_teams == report.n_matches_in
    assert report.n_shots_with_xg + report.n_shots_null_xg == report.n_shots
    assert report.n_matches_scored == 1 and report.n_matches_in == 1


def test_not_two_teams_excluded_and_counted():
    # game 1 valid (2 teams); game 2 has only one team -> excluded + counted; game 1 still scored.
    good = _two_team_fixture(game_id=1)
    solo = make_actions(
        [
            {"team_id": 30, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
            {"team_id": 30, "type_id": PASS, "time_seconds": 2.0, "start_x": 50.0},
        ],
        game_id=2,
    )
    samples, report = compute_team_kpis(pd.concat([good, solo], ignore_index=True))
    assert report.n_matches_in == 2
    assert report.n_matches_scored == 1
    assert report.n_matches_excluded_not_two_teams == 1
    assert set(samples["game_id"]) == {1}  # only the 2-team game is scored


def test_purity_input_unmutated():
    actions = _two_team_fixture()
    snapshot = actions.copy(deep=True)
    compute_team_kpis(actions, xg_column="xg")
    pd.testing.assert_frame_equal(actions, snapshot)


def test_order_insensitivity():
    actions = _two_team_fixture()
    permuted = actions.iloc[::-1].reset_index(drop=True)
    s1, _ = compute_team_kpis(actions, xg_column="xg")
    s2, _ = compute_team_kpis(permuted, xg_column="xg")
    s1 = s1.sort_values(["game_id", "team_id"]).reset_index(drop=True)
    s2 = s2.sort_values(["game_id", "team_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(s1, s2)


def test_no_xg_column_high_opp_nan_else_unchanged():
    fx = _two_team_fixture()
    with_xg, _ = compute_team_kpis(fx, xg_column="xg")
    no_xg, report = compute_team_kpis(fx, xg_column=None)
    assert no_xg.set_index("team_id").loc[10, "high_opportunity_shots"] is pd.NA
    assert no_xg.set_index("team_id").loc[10, "high_opportunity_shots_post_recovery"] is pd.NA
    # a non-xG KPI is identical with and without the xg column
    a = with_xg.sort_values("team_id").reset_index(drop=True)
    b = no_xg.sort_values("team_id").reset_index(drop=True)
    pd.testing.assert_series_equal(a["ppda"], b["ppda"])
    # census: no xG column -> no shots counted as with-xg
    assert report.n_shots_with_xg == 0


def test_within_ns_companion():
    # team 10 recovers at t=2; the transition-output block (final-third entries, box touches, shots)
    # is re-computed inside the 10s window. Each event has a within-window twin and an outside-window
    # twin, so every companion is exactly half its full-match value.
    fx = make_actions(
        [
            {"team_id": 20, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 2.0, "start_x": 40.0},  # recovery
            {"team_id": 10, "type_id": PASS, "time_seconds": 3.0, "start_x": 50.0, "end_x": 80.0},  # entry, in
            {"team_id": 10, "type_id": SHOT, "time_seconds": 4.0, "start_x": 75.0},  # shot (not box), in
            {"team_id": 10, "type_id": PASS, "time_seconds": 5.0, "start_x": 90.0},  # box, in
            {"team_id": 10, "type_id": PASS, "time_seconds": 20.0, "start_x": 90.0},  # box, out
            {"team_id": 10, "type_id": SHOT, "time_seconds": 21.0, "start_x": 75.0},  # shot, out
            {"team_id": 10, "type_id": PASS, "time_seconds": 22.0, "start_x": 50.0, "end_x": 80.0},  # entry, out
        ]
    )
    samples, _ = compute_team_kpis(fx)
    # 2-arg .loc[team, col] returns a scalar (typed Any) -- keeps the comparison out of pyright's
    # `Series[bool]` inference that a Series-row `r["col"] == n` triggers.
    r = samples.set_index("team_id")
    # full-match counts (both twins)
    assert r.loc[10, "final_third_entries"] == 2
    assert r.loc[10, "box_touches"] == 2
    assert r.loc[10, "shots"] == 2
    # each companion captures only the in-window twin -> half the full value (differs, per spec 4.5)
    assert r.loc[10, "final_third_entries_post_recovery"] == 1
    assert r.loc[10, "box_touches_post_recovery"] == 1
    assert r.loc[10, "shots_post_recovery"] == 1


def test_empty_actions():
    empty = make_actions([{"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0}]).iloc[:0]
    samples, report = compute_team_kpis(empty)
    assert len(samples) == 0
    assert list(samples.columns) == list(TEAM_KPI_COLUMNS)
    assert report.n_matches_in == 0
