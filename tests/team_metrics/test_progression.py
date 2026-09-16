"""Progression family (TF-52 Task 4): field tilt, tempo, long-ball, heights, chain, breakout, retained."""

from __future__ import annotations

import numpy as np

from silly_kicks.team_metrics import TeamKpiParams
from silly_kicks.team_metrics._possession import (
    add_possession_context,
    build_spells,
    possession_minutes,
)
from silly_kicks.team_metrics._progression import compute_progression_kpis
from tests.team_metrics._helpers import GOALKICK, INTERCEPTION, PASS, SHOT, SHOT_PENALTY, make_actions


def _progression(actions, xg_column=None, params=None):
    params = params or TeamKpiParams()
    ctx = add_possession_context(actions, params=params)
    spells = build_spells(ctx)
    minutes = possession_minutes(spells)
    return compute_progression_kpis(ctx, spells, minutes, xg_column=xg_column, params=params)


def _row(out, team_id):
    return out[out["team_id"] == team_id].iloc[0]


def _main_fixture():
    return make_actions(
        [
            # team 10 possession A: own-half long pass, cross-half (left), final third, box, shot.
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 30.0, "end_x": 70.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 1.0, "start_x": 60.0, "start_y": 10.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 75.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 3.0, "start_x": 90.0},
            {"team_id": 10, "type_id": SHOT, "time_seconds": 4.0, "start_x": 95.0, "xg": 0.30},
            # team 20 possession B
            {"team_id": 20, "type_id": PASS, "time_seconds": 6.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 8.0, "start_x": 50.0},
            # team 10 possession C: recovery in own half, no progression
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 9.0, "start_x": 40.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 11.0, "start_x": 45.0},
            # team 20 possession D
            {"team_id": 20, "type_id": PASS, "time_seconds": 20.0, "start_x": 50.0},
        ]
    )


def test_progression_known_values():
    r = _row(_progression(_main_fixture(), xg_column="xg"), 10)
    assert r["field_tilt_pct"] == 1.0  # 3 own final-third touches / 3 total
    assert r["pass_tempo"] == 50.0  # 5 passes / (6s / 60)
    assert r["long_ball_pct"] == 0.5  # 1 long of 2 own-half passes
    assert r["defensive_action_height_m"] == 40.0  # the lone interception at x=40
    assert r["recovery_line_height_m"] == 40.0  # recovery possession starts at x=40
    assert r["turnover_line_height_m"] == 70.0  # mean last-action x of [95, 45]
    assert r["poss_to_final_third_pct"] == 0.5  # 1 of 2 possessions reached the final third
    assert r["final_third_to_box_pct"] == 1.0  # the one final-third possession reached the box
    assert r["box_to_shot_pct"] == 1.0  # the one box possession produced a shot
    assert r["box_touches"] == 2  # x=90 and x=95
    assert r["final_third_entries"] == 1  # the 30->70 pass crosses x=2*FL/3
    assert r["shots"] == 1  # the lone shot (xG-independent)
    assert r["high_opportunity_shots"] == 1  # the 0.30-xG shot
    assert r["breakout_left"] == 1 and r["breakout_left_pct"] == 1.0
    assert r["breakout_center"] == 0 and r["breakout_right"] == 0
    assert r["breakout_center_pct"] == 0.0 and r["breakout_right_pct"] == 0.0


def test_shots_and_final_third_entries_known_values():
    fx = make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0, "end_x": 80.0},  # entry
            {"team_id": 10, "type_id": PASS, "time_seconds": 1.0, "start_x": 75.0, "end_x": 90.0},  # inside -> no entry
            {"team_id": 10, "type_id": SHOT, "time_seconds": 2.0, "start_x": 95.0},  # shot
            {"team_id": 10, "type_id": SHOT_PENALTY, "time_seconds": 3.0, "start_x": 94.0},  # shot (penalty counts)
            {"team_id": 20, "type_id": PASS, "time_seconds": 5.0, "start_x": 50.0},  # other team: no entry / no shot
        ]
    )
    out = _progression(fx)
    r10, r20 = _row(out, 10), _row(out, 20)
    assert r10["final_third_entries"] == 1  # only the 50->80 pass crosses; the 75->90 starts inside
    assert r10["shots"] == 2  # open-play shot + penalty (raw, xG-independent)
    assert r20["final_third_entries"] == 0 and r20["shots"] == 0  # both are honest zero counts, not NaN


def test_high_opportunity_shots_both_sides():
    fx = _main_fixture()
    assert np.isnan(_row(_progression(fx, xg_column=None), 10)["high_opportunity_shots"])  # no xG -> NaN
    assert _row(_progression(fx, xg_column="xg"), 10)["high_opportunity_shots"] == 1  # with xG -> real count


def _retained_fixture():
    # team 10 open-play possessions of durations 3, 6, 8; team 20 splits them.
    return make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 3.0, "start_x": 50.0},  # dur 3
            {"team_id": 20, "type_id": PASS, "time_seconds": 5.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 7.0, "start_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 10.0, "start_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 13.0, "start_x": 50.0},  # dur 6
            {"team_id": 20, "type_id": PASS, "time_seconds": 15.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 17.0, "start_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 21.0, "start_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 25.0, "start_x": 50.0},  # dur 8
        ]
    )


def test_retained_after_ns_known_value():
    r = _row(_progression(_retained_fixture()), 10)
    assert r["possessions_retained_after_ns_pct"] == 2.0 / 3.0  # durations 6 and 8 >= 5s


def test_retained_after_ns_nan_both_sides():
    # team 10's only possession starts on a goalkick -> no open-play possession -> NaN.
    nan_fx = make_actions(
        [
            {"team_id": 10, "type_id": GOALKICK, "time_seconds": 0.0, "start_x": 5.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 20.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 10.0, "start_x": 50.0},
        ]
    )
    assert np.isnan(_row(_progression(nan_fx), 10)["possessions_retained_after_ns_pct"])

    finite_fx = make_actions(
        [
            {"team_id": 10, "type_id": GOALKICK, "time_seconds": 0.0, "start_x": 5.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 20.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 10.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 12.0, "start_x": 40.0},  # open-play poss
            {"team_id": 10, "type_id": PASS, "time_seconds": 20.0, "start_x": 45.0},  # dur 8 >= 5
        ]
    )
    assert np.isfinite(_row(_progression(finite_fx), 10)["possessions_retained_after_ns_pct"])


def test_field_tilt_nan_when_no_final_third_touches():
    # neither team enters the final third -> field tilt undefined.
    fx = make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 30.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 5.0, "start_x": 30.0},
        ]
    )
    assert np.isnan(_row(_progression(fx), 10)["field_tilt_pct"])


def test_long_ball_nan_when_no_own_half_passes():
    # team 10's only pass is in the opponent half -> long-ball % undefined.
    fx = make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 80.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 5.0, "start_x": 30.0},
        ]
    )
    assert np.isnan(_row(_progression(fx), 10)["long_ball_pct"])
