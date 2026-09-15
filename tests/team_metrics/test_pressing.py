"""Pressing family (TF-52 Task 3): PPDA, intensity, times, recoveries, counter-press arms."""

from __future__ import annotations

import numpy as np

from silly_kicks.team_metrics import CounterpressWindow, TeamKpiParams
from silly_kicks.team_metrics._possession import (
    add_possession_context,
    build_spells,
    possession_minutes,
)
from silly_kicks.team_metrics._pressing import compute_pressing_kpis
from tests.team_metrics._helpers import INTERCEPTION, PASS, TACKLE, make_actions


def _pressing(actions, params=None):
    params = params or TeamKpiParams()
    ctx = add_possession_context(actions, params=params)
    spells = build_spells(ctx)
    minutes = possession_minutes(spells)
    return compute_pressing_kpis(ctx, spells, minutes, params=params)


def _row(out, team_id):
    return out[out["team_id"] == team_id].iloc[0]


def test_ppda_known_value():
    # team 10: 2 tackles in the pressing zone; team 20: 8 passes that reflect into that zone.
    recs = [
        {"team_id": 10, "type_id": TACKLE, "time_seconds": 0.0, "start_x": 50.0},
        {"team_id": 10, "type_id": TACKLE, "time_seconds": 1.0, "start_x": 60.0},
    ] + [{"team_id": 20, "type_id": PASS, "time_seconds": 2.0 + i, "start_x": 30.0} for i in range(8)]
    out = _pressing(make_actions(recs))
    assert _row(out, 10)["ppda"] == 4.0  # 8 opp passes / 2 our tackles


def test_ppda_nan_then_finite_both_sides():
    base = [
        {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
    ] + [{"team_id": 20, "type_id": PASS, "time_seconds": 2.0 + i, "start_x": 30.0} for i in range(4)]
    out = _pressing(make_actions(base))
    assert np.isnan(_row(out, 10)["ppda"])  # 0 team-10 defensive actions in zone -> undefined

    mutated = [*base, {"team_id": 10, "type_id": TACKLE, "time_seconds": 20.0, "start_x": 50.0}]
    out2 = _pressing(make_actions(mutated))
    assert np.isfinite(_row(out2, 10)["ppda"])  # a tackle in zone -> finite


def _match_fixture():
    # 5 possessions in one period (see plan Task 3 / spec Section 4.1 worked expectations).
    return make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 30.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 4.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 6.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 8.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 9.0, "start_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 11.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 20.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 22.0, "start_x": 50.0},
            {"team_id": 10, "type_id": TACKLE, "time_seconds": 30.0, "start_x": 50.0},
        ]
    )


def test_recoveries_intensity_and_times():
    r = _row(_pressing(_match_fixture()), 10)
    assert r["recoveries"] == 2
    assert r["recoveries_within_ns_pct"] == 0.5  # gaps [5, 19]; only 5 <= 5s
    assert r["time_to_recovery_s"] == 12.0  # mean([5, 19])
    assert r["defensive_intensity"] == 30.0  # 2 def actions / (4s / 60)
    assert r["time_to_defensive_action_s"] == 12.0  # mean([5, 19])
    assert r["ppda"] == 2.0  # 4 opp passes in zone / 2 our def in zone


def test_defensive_intensity_nan_when_no_out_of_possession():
    # single team, single possession -> team has no out-of-possession time -> intensity NaN.
    out = _pressing(
        make_actions(
            [
                {"team_id": 10, "type_id": TACKLE, "time_seconds": 0.0, "start_x": 50.0},
                {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 50.0},
            ]
        )
    )
    assert np.isnan(_row(out, 10)["defensive_intensity"])


def _counterpress_fixture():
    # team10 loses at t=0; team20 completes ONE pass at t=2; team10 recovers at t=9 (gap 9s).
    return make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 2.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 9.0, "start_x": 50.0},
        ]
    )


def test_counterpress_arms_disagree():
    fx = _counterpress_fixture()
    # seconds arm (5s): gap 9s > 5 -> NOT a counter-press regain.
    sec = _pressing(fx, TeamKpiParams(counterpress_window=CounterpressWindow(seconds=5.0)))
    assert _row(sec, 10)["counterpress_regains"] == 0
    # passes arm (<=3): opponent completed only 1 pass before the regain -> IS a regain.
    pas = _pressing(fx, TeamKpiParams(counterpress_window=CounterpressWindow(passes=3)))
    assert _row(pas, 10)["counterpress_regains"] == 1
    assert _row(pas, 10)["counterpress_regain_pct"] == 1.0


def test_counterpress_nan_when_no_losses():
    # team10 never loses/regains (single possession) -> no recoveries -> pct NaN.
    out = _pressing(
        make_actions(
            [
                {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
                {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 50.0},
            ]
        )
    )
    assert _row(out, 10)["recoveries"] == 0
    assert np.isnan(_row(out, 10)["counterpress_regain_pct"])
