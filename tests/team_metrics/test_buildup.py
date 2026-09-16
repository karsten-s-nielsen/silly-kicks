"""Build-up family (TF-52 Task 5): taxonomy, post-regain security, switch-conditioned press."""

from __future__ import annotations

import numpy as np

from silly_kicks.team_metrics import TeamKpiParams
from silly_kicks.team_metrics._buildup import compute_buildup_kpis
from silly_kicks.team_metrics._possession import add_possession_context, build_spells
from tests.team_metrics._helpers import FAIL, GOALKICK, INTERCEPTION, PASS, SHOT, make_actions


def _buildup(actions, params=None):
    params = params or TeamKpiParams()
    ctx = add_possession_context(actions, params=params)
    spells = build_spells(ctx)
    return compute_buildup_kpis(ctx, spells, params=params)


def _row(out, team_id):
    return out[out["team_id"] == team_id].iloc[0]


def test_taxonomy_four_states():
    # BU1 final_quarter, BU2 next_phase, BU3 opp_won_own_half, BU4 stayed_phase_one.
    fx = make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 20.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 80.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 4.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 6.0, "start_x": 20.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 8.0, "start_x": 60.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 10.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 12.0, "start_x": 25.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 14.0, "start_x": 30.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 16.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 25.0, "start_x": 25.0},
        ]
    )
    r = _row(_buildup(fx), 10)
    assert r["buildup_final_quarter"] == 1
    assert r["buildup_next_phase"] == 1
    assert r["buildup_opp_won_own_half"] == 1
    assert r["buildup_stayed_phase_one"] == 1
    assert r["buildup_led_opp_shot"] == 0
    assert r["buildup_opp_int_own_half"] == 0
    assert r["buildup_success_pct"] == 0.5  # (final_quarter + next_phase) / 4


def test_taxonomy_shot_and_interception_states():
    fx = make_actions(
        [
            # BU_led_opp_shot: low build-up lost, opponent's next possession has a shot.
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 20.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 30.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 4.0, "start_x": 50.0},
            {"team_id": 20, "type_id": SHOT, "time_seconds": 6.0, "start_x": 90.0},
            # BU_opp_int_own_half: low build-up lost via an opponent interception in own half.
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 10.0, "start_x": 20.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 12.0, "start_x": 25.0},
            {"team_id": 20, "type_id": INTERCEPTION, "time_seconds": 14.0, "start_x": 50.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 16.0, "start_x": 50.0},
        ]
    )
    r = _row(_buildup(fx), 10)
    assert r["buildup_led_opp_shot"] == 1
    assert r["buildup_opp_int_own_half"] == 1


def test_buildup_success_nan_when_no_buildups():
    # team 10's only possession starts in the opponent half -> no build-up -> success NaN.
    fx = make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 60.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 5.0, "start_x": 50.0},
        ]
    )
    r = _row(_buildup(fx), 10)
    assert r["buildup_final_quarter"] == 0
    assert np.isnan(r["buildup_success_pct"])


def test_post_regain_security():
    fx = make_actions(
        [
            {"team_id": 20, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 2.0, "start_x": 40.0, "end_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 4.0, "start_x": 45.0, "result_id": FAIL},
            {"team_id": 10, "type_id": PASS, "time_seconds": 6.0, "start_x": 50.0},  # 2nd pass completed
        ]
    )
    r = _row(_buildup(fx), 10)
    assert r["post_regain_forward_first_pct"] == 1.0  # interception carried forward (40 -> 50)
    assert r["post_regain_failed_first_passes"] == 1  # the first pass failed
    assert r["post_regain_second_pass_pct"] == 1.0  # the second pass completed


def test_post_regain_nan_when_no_recoveries():
    # single-possession team -> no recoveries -> pcts NaN, failed count 0.
    fx = make_actions(
        [
            {"team_id": 10, "type_id": PASS, "time_seconds": 0.0, "start_x": 50.0},
            {"team_id": 10, "type_id": PASS, "time_seconds": 2.0, "start_x": 50.0},
        ]
    )
    r = _row(_buildup(fx), 10)
    assert np.isnan(r["post_regain_second_pass_pct"])
    assert np.isnan(r["post_regain_forward_first_pct"])
    assert r["post_regain_failed_first_passes"] == 0


def test_switch_conditioned_press():
    # team 20 plays a SHORT goal kick (goalkick then a short pass, no switch); team 10 regains.
    fx = make_actions(
        [
            {"team_id": 20, "type_id": GOALKICK, "time_seconds": 0.0, "start_x": 5.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 2.0, "start_x": 15.0},  # short (no lateral switch)
            {"team_id": 10, "type_id": INTERCEPTION, "time_seconds": 4.0, "start_x": 40.0},  # team 10 regains
        ]
    )
    r10 = _row(_buildup(fx), 10)
    assert r10["switch_press_n"] == 1  # one opponent short goal kick faced
    assert r10["switch_press_success_pct"] == 1.0  # switch prevented AND regained

    r20 = _row(_buildup(fx), 20)
    assert r20["switch_press_n"] == 0  # team 10 played no short goal kick
    assert np.isnan(r20["switch_press_success_pct"])
