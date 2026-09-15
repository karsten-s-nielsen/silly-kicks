"""TeamKpiReport conservation census (TF-52 Task 1). Mirrors tests/shot_stopping/test_report.py."""

from __future__ import annotations

from silly_kicks.team_metrics import TeamKpiParams, TeamKpiReport


def test_match_conservation():
    r = TeamKpiReport(
        TeamKpiParams(),
        n_matches_in=10,
        n_matches_scored=9,
        n_matches_excluded_not_two_teams=1,
        n_shots=0,
        n_shots_with_xg=0,
        n_shots_null_xg=0,
    )
    assert r.n_matches_scored + r.n_matches_excluded_not_two_teams == r.n_matches_in


def test_shot_census_conservation():
    r = TeamKpiReport(TeamKpiParams(), 2, 2, 0, n_shots=25, n_shots_with_xg=23, n_shots_null_xg=2)
    assert r.n_shots_with_xg + r.n_shots_null_xg == r.n_shots


def test_report_carries_params():
    p = TeamKpiParams.default()
    r = TeamKpiReport(p, 1, 1, 0, 0, 0, 0)
    assert r.params is p
