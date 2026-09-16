"""TF-51 Item 4 -- atomic-SPADL mirror of the defensive-credit family.

Faithful (atom-stream) mirror: adapter re-lift + per-type next-atom result synthesis, atomic
compute/add/bravery entry points, preserve_native required-column raises, honest-NaN set-piece bravery,
and the faithful-representation limitation pins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.atomic.tracking.features as atf
from silly_kicks.atomic.spadl import config as ac
from silly_kicks.spadl import config as sc
from silly_kicks.tracking.defensive_credit import DefensiveCreditParams

A = atf._defensive_credit_atomic_adapter


def _atom(type_name, *, team=1, player=10, x=50.0, y=34.0, dx=5.0, dy=0.0, gid=1, pid=1, aid=0, t=0.0):
    return {
        "game_id": gid,
        "period_id": pid,
        "action_id": aid,
        "time_seconds": t,
        "team_id": team,
        "player_id": player,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "type_id": ac.actiontype_id[type_name],
        "bodypart_id": 0,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---- Task 2: adapter ----
def test_endpoints_synthesized():
    out = A(_frame([_atom("pass", x=10.0, y=20.0, dx=3.0, dy=-4.0)]), DefensiveCreditParams())
    assert out.loc[0, "start_x"] == 10.0 and out.loc[0, "start_y"] == 20.0
    assert out.loc[0, "end_x"] == 13.0 and out.loc[0, "end_y"] == 16.0


def test_domain_type_map_and_nonaction():
    out = A(_frame([_atom("pass", aid=0), _atom("receival", aid=1)]), DefensiveCreditParams())
    assert out.loc[0, "type_id"] == sc.actiontype_id["pass"]
    assert out.loc[1, "type_id"] == sc.actiontype_id["non_action"]  # atomic-only atom off-domain


def test_pass_success_iff_next_receival():
    completed = A(_frame([_atom("pass", aid=0), _atom("receival", aid=1)]), DefensiveCreditParams())
    assert completed.loc[0, "result_id"] == sc.result_id["success"]
    failed = A(_frame([_atom("pass", aid=0), _atom("interception", team=2, aid=1)]), DefensiveCreditParams())
    assert failed.loc[0, "result_id"] == sc.result_id["fail"]


def test_pass_success_to_keeper_reception():
    out = A(_frame([_atom("pass", team=1, aid=0), _atom("keeper_pick_up", team=1, aid=1)]), DefensiveCreditParams())
    assert out.loc[0, "result_id"] == sc.result_id["success"]


def test_take_on_success_iff_retained_same_team_not_lost():
    retained = A(_frame([_atom("take_on", team=1, aid=0), _atom("dribble", team=1, aid=1)]), DefensiveCreditParams())
    assert retained.loc[0, "result_id"] == sc.result_id["success"]
    lost = A(_frame([_atom("take_on", team=1, aid=0), _atom("interception", team=2, aid=1)]), DefensiveCreditParams())
    assert lost.loc[0, "result_id"] == sc.result_id["fail"]


def test_shot_success_iff_next_goal():
    scored = A(_frame([_atom("shot", aid=0), _atom("goal", aid=1)]), DefensiveCreditParams())
    assert scored.loc[0, "result_id"] == sc.result_id["success"]
    missed = A(_frame([_atom("shot", aid=0), _atom("out", aid=1)]), DefensiveCreditParams())
    assert missed.loc[0, "result_id"] == sc.result_id["fail"]


def test_bad_touch_maps_to_std_and_result_is_fail():
    # bad_touch is a domain type (rule_forced_bad_touch anchors on type only); its result is UNREAD,
    # synthesized fail even when a same-team atom follows (per the §3.3 table).
    out = A(_frame([_atom("bad_touch", team=1, aid=0), _atom("dribble", team=1, aid=1)]), DefensiveCreditParams())
    assert out.loc[0, "type_id"] == sc.actiontype_id["bad_touch"]
    assert out.loc[0, "result_id"] == sc.result_id["fail"]


def test_period_last_atom_is_fail():
    out = A(_frame([_atom("pass", aid=0)]), DefensiveCreditParams())
    assert out.loc[0, "result_id"] == sc.result_id["fail"]


def test_next_atom_in_different_period_does_not_complete():
    rows = [_atom("pass", aid=0, pid=1), _atom("receival", aid=1, pid=2)]
    out = A(_frame(rows), DefensiveCreditParams())
    assert out.loc[0, "result_id"] == sc.result_id["fail"]  # cross-period next atom is not a completion


def test_adapter_does_not_mutate_caller():
    frame = _frame([_atom("pass", aid=0), _atom("receival", aid=1)])
    before = frame.copy(deep=True)
    _ = A(frame, DefensiveCreditParams())
    pd.testing.assert_frame_equal(frame, before)


# ---- scene builders for the delegated compute/aggregate/bravery tests ----
def _pressured_failed_pass_atomic():
    """A failed pass by acting team 10 at (95,34) + an opponent interception (which both makes the pass
    synth-fail and IS the recovery), a frame with an opponent defender 1 m away, and the injected
    analytics columns threaded as preserve_native would."""
    from tests.tracking._defensive_credit_fixtures import frame_with_defender

    rows = [
        dict(
            game_id="g1",
            period_id=1,
            action_id=0,
            time_seconds=50.0,
            team_id=10,
            player_id=5,
            x=95.0,
            y=34.0,
            dx=5.0,
            dy=0.0,
            type_id=ac.actiontype_id["pass"],
            bodypart_id=0,
        ),
        dict(
            game_id="g1",
            period_id=1,
            action_id=1,
            time_seconds=50.1,
            team_id=20,
            player_id=900,
            x=100.0,
            y=34.0,
            dx=0.0,
            dy=0.0,
            type_id=ac.actiontype_id["interception"],
            bodypart_id=0,
        ),
    ]
    a = pd.DataFrame(rows)
    a["xg"] = [np.nan, np.nan]
    a["shot_blocked"] = pd.array([pd.NA, pd.NA], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA, pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA, pd.NA], dtype="boolean")
    f = frame_with_defender(game_id="g1", defender_x=96.0, defender_y=34.0)
    return a, f


# ---- Task 4: atomic compute_defensive_credits + required-column raise ----
def test_missing_xg_column_raises(fitted_xt):
    a, f = _pressured_failed_pass_atomic()
    with pytest.raises(ValueError, match="xg"):
        atf.compute_defensive_credits(a.drop(columns=["xg"]), f, xg_column="xg", xt=fitted_xt)


def test_long_form_credits_through_public_entry(fitted_xt):
    a, f = _pressured_failed_pass_atomic()
    long = atf.compute_defensive_credits(a, f, xg_column="xg", xt=fitted_xt)
    assert "pressure_pass_fail" in set(long["rule"])  # the scripted rule fired
    fired = long[long["rule"] == "pressure_pass_fail"]
    assert (fired["signed_value"].abs() > 0).all()


def _full_pitch_visible(actions):
    poly = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    return pd.DataFrame({"action_id": list(actions["action_id"]), "polygon": [poly] * len(actions)})


# ---- Task 5: atomic _aggregate_defensive_credit + add_defensive_credit ----
def test_aggregate_columns_and_no_synth_leak(fitted_xt):
    a, f = _pressured_failed_pass_atomic()
    before = a.copy(deep=True)
    out = atf.add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    for c in ["defensive_credit_net", "defensive_credit_plus", "defensive_credit_minus", "n_defensive_credits"]:
        assert c in out.columns
    # atomic columns intact; NO synthesized std columns leaked
    assert "start_x" not in out.columns and "result_id" not in out.columns
    assert {"x", "y", "dx", "dy"}.issubset(out.columns)
    # the defending (team 20) credit is net-positive on the pass action
    assert out.loc[out["action_id"] == 0, "defensive_credit_net"].iloc[0] > 0
    assert out.loc[out["action_id"] == 0, "n_defensive_credits"].iloc[0] >= 1
    # purity: caller unmutated
    pd.testing.assert_frame_equal(a, before)


def test_visible_area_companion_additive(fitted_xt):
    a, f = _pressured_failed_pass_atomic()
    polygons = _full_pitch_visible(a)
    base = atf.add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    withva = atf.add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt, visible_area=polygons)
    for c in ["defensive_credit_net", "defensive_credit_plus", "defensive_credit_minus", "n_defensive_credits"]:
        pd.testing.assert_series_equal(base[c], withva[c])  # primary columns byte-identical
    assert {"defensive_credit_observed_fraction", "defensive_credit_observed_source"}.issubset(withva.columns)


# ---- Task 6: atomic compute_bravery (honest-NaN set-piece) ----
def _atomic_bravery_scene():
    """Shots + open-play crosses (block status known) + a COLLAPSED set-piece (atomic `corner`), plus a
    team-20 action so the opponent resolves. shot + open-play cross survive atomic intact; the collapsed
    set-piece is unobservable."""
    rows = [
        dict(
            game_id="g1",
            period_id=1,
            action_id=0,
            time_seconds=1.0,
            team_id=10,
            player_id=5,
            x=95.0,
            y=34.0,
            dx=5.0,
            dy=0.0,
            type_id=ac.actiontype_id["shot"],
            bodypart_id=0,
        ),
        dict(
            game_id="g1",
            period_id=1,
            action_id=1,
            time_seconds=2.0,
            team_id=10,
            player_id=6,
            x=90.0,
            y=10.0,
            dx=10.0,
            dy=20.0,
            type_id=ac.actiontype_id["cross"],
            bodypart_id=0,
        ),
        dict(
            game_id="g1",
            period_id=1,
            action_id=2,
            time_seconds=3.0,
            team_id=10,
            player_id=7,
            x=100.0,
            y=0.0,
            dx=5.0,
            dy=34.0,
            type_id=ac.actiontype_id["corner"],
            bodypart_id=0,
        ),
        dict(
            game_id="g1",
            period_id=1,
            action_id=3,
            time_seconds=4.0,
            team_id=20,
            player_id=99,
            x=50.0,
            y=34.0,
            dx=0.0,
            dy=0.0,
            type_id=ac.actiontype_id["pass"],
            bodypart_id=0,
        ),
    ]
    a = pd.DataFrame(rows)
    a["shot_blocked"] = pd.array([True, pd.NA, pd.NA, pd.NA], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA, False, pd.NA, pd.NA], dtype="boolean")
    return a


def test_bravery_setpiece_is_honest_nan():
    b = atf.compute_bravery(_atomic_bravery_scene())
    assert b["bravery_set_piece_crosses"].isna().all()
    assert b["n_set_piece_crosses_faced"].isna().all()  # honest NA, NOT a fabricated 0 (ADR-027)
    assert b["bravery_shots"].notna().any()  # shots survive atomic intact
    assert b["bravery_pct_known_domain"].notna().any()  # headline (shots + open crosses) survives


def test_bravery_missing_cross_blocked_raises():
    a = _atomic_bravery_scene().drop(columns=["cross_blocked"])
    with pytest.raises(ValueError, match="cross_blocked"):
        atf.compute_bravery(a)


# ---- Task 8: faithful-representation limitation pins ----
def test_shot_freekick_not_a_resulting_shot_on_atomic(fitted_xt):
    """A successful take_on whose only following shot-like action is a shot_freekick (collapsed to
    atomic `freekick` by _simplify) is NOT detected as a resulting shot, so beaten_1v1 cannot fire --
    a documented faithful-representation limitation, pinned so a future change is deliberate."""
    from tests.tracking._defensive_credit_fixtures import frame_with_defender

    rows = [
        # successful take_on (next atom is same-team `freekick` -> retained -> success)
        dict(
            game_id="g1",
            period_id=1,
            action_id=0,
            time_seconds=50.0,
            team_id=10,
            player_id=5,
            x=80.0,
            y=34.0,
            dx=4.0,
            dy=0.0,
            type_id=ac.actiontype_id["take_on"],
            bodypart_id=0,
        ),
        # what was a shot_freekick -> atomic `freekick` (crossed/short/shot_freekick all collapse)
        dict(
            game_id="g1",
            period_id=1,
            action_id=1,
            time_seconds=50.5,
            team_id=10,
            player_id=5,
            x=85.0,
            y=34.0,
            dx=20.0,
            dy=0.0,
            type_id=ac.actiontype_id["freekick"],
            bodypart_id=0,
        ),
    ]
    a = pd.DataFrame(rows)
    a["xg"] = [0.4, 0.4]
    a["shot_blocked"] = pd.array([pd.NA, pd.NA], dtype="boolean")
    f = frame_with_defender(game_id="g1", defender_x=81.0, defender_y=34.0)  # defender by the take_on
    long = atf.compute_defensive_credits(a, f, xg_column="xg", xt=fitted_xt)
    assert "beaten_1v1" not in set(long["rule"])  # collapsed freekick is not a resulting shot


def test_recovery_fires_on_distance1_interception_atom(fitted_xt):
    """On atomic, a failed pass's opponent regain IS the inserted interception atom at distance 1 --
    atom-dense recovery, tighter than the standard-action scan."""
    a, f = _pressured_failed_pass_atomic()
    long = atf.compute_defensive_credits(a, f, xg_column="xg", xt=fitted_xt)
    assert "recovery_double_credit" in set(long["rule"])
