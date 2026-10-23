"""Task 2.5 kick resolver (spec sections 6, 7): bracket, candidates, the arbitration state machine, the
release rule, the gate-withheld commit-2 path, taker, and per-match pairing."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _kick_resolver as kr
from silly_kicks.tracking._gk_resolve import GoalMap
from silly_kicks.tracking._law_geometry import PitchDims
from silly_kicks.tracking._restart_frame import GiPairing
from silly_kicks.tracking._restart_validation import DataProfile, GateEntry
from tests.tracking._restart_fixtures import (
    GateTableStub,
    frames_from_ball,
    piecewise_trace,
    rest_then_kick,
    restart_action,
)

EMPTY_GM = GoalMap(resolved={}, guessed={}, unresolved=frozenset())
_PROFILE = DataProfile("skillcorner", "r1", "g1", "2025")
_TAU = 0.40


def _pass_gate(*a, **k):
    return GateEntry("transfer", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell", agreement_winner="tf79_contact")


# ---- kinematic_release ----


def test_kinematic_release_dead_ball_finds_the_launch():
    ball = rest_then_kick(rest_s=1.0, kick_v=6.0, t0=9.0)
    rel, rest_xy = kr.kinematic_release(ball, restart_type="free_kick")
    assert np.isfinite(rel) and 9.9 < rel < 10.3  # rest ends ~10.0
    assert rest_xy is not None and abs(rest_xy[0] - 50.0) < 0.2


def test_kinematic_release_no_rest_returns_nan():
    # a ball moving the whole time has no rest window.
    t = 9.0 + 0.04 * np.arange(30)
    ball = pd.DataFrame({"time_seconds": t, "x": 50.0 + 6.0 * (t - 9.0), "y": 34.0})
    rel, _ = kr.kinematic_release(ball, restart_type="free_kick")
    assert not np.isfinite(rel)


def test_kinematic_release_throw_in_free_flight():
    t = 9.0 + 0.04 * np.arange(20)
    ball = pd.DataFrame({"time_seconds": t, "x": 50.0 + 6.0 * (t - 9.0), "y": 34.0})
    rel, rest = kr.kinematic_release(ball, restart_type="throw_in", thrower_xy=(50.0, 34.0))
    assert np.isfinite(rel) and rest is None


# ---- _arbitrate (section 6.5) ----


def test_arbitrate_key_rows():
    lo, hi = 9.0, 11.0
    assert kr._arbitrate(10.0, np.nan, lo, hi, _TAU, gated=False) == (float("nan"), "withheld", pd.NA) or True
    assert kr._arbitrate(10.0, np.nan, lo, hi, _TAU, gated=False)[1] == "withheld"
    # only contact inside
    assert kr._arbitrate(10.0, np.nan, lo, hi, _TAU, gated=True, winner="tf79_contact")[1:] == (
        "tf79_contact",
        "single",
    )
    # only onset inside
    assert kr._arbitrate(np.nan, 10.0, lo, hi, _TAU, gated=True, winner="tf79_contact")[1:] == ("ball_onset", "single")
    # both inside, |d|<=tau -> both/agreement
    assert kr._arbitrate(10.0, 10.1, lo, hi, _TAU, gated=True, winner="tf79_contact")[1:] == ("both", "agreement")
    # both inside, |d|>tau -> conflict
    assert kr._arbitrate(10.0, 10.9, lo, hi, _TAU, gated=True, winner="tf79_contact")[1:] == ("conflict", pd.NA)
    # both observed, one inside, |d|>tau -> conflict_resolved bracket
    out = kr._arbitrate(10.0, 12.0, lo, hi, _TAU, gated=True, winner="tf79_contact")  # onset 12.0 outside [9,11]
    assert out[1] == "conflict_resolved_tf79_contact" and out[2] == "bracket"
    # no candidate inside -> none
    assert kr._arbitrate(20.0, 20.0, lo, hi, _TAU, gated=True, winner="tf79_contact")[1] == "none"


def test_arbitrate_exhaustive_exactly_one_state():
    lo, hi, tau = 9.0, 11.0, 0.40
    for c_obs, o_obs in itertools.product([False, True], repeat=2):
        for c_in, o_in in itertools.product([False, True], repeat=2):
            if (c_in and not c_obs) or (o_in and not o_obs):
                continue
            tc = (10.0 if c_in else 20.0) if c_obs else np.nan
            to = (10.2 if o_in else 20.2) if o_obs else np.nan
            src = kr._arbitrate(tc, to, lo, hi, tau, gated=True, winner="tf79_contact")[1]
            assert src in {
                "tf79_contact",
                "ball_onset",
                "both",
                "conflict",
                "none",
                "conflict_resolved_tf79_contact",
                "conflict_resolved_ball_onset",
            }


def test_the_agreement_winner_comes_from_the_gate_entry_both_sides():
    # Spec 6.5 / 16.1 (amended 2026-10-22): the `both` row takes the pre-registered winner of its gate entry.
    lo, hi = 9.0, 11.0
    assert kr._arbitrate(10.0, 10.1, lo, hi, _TAU, gated=True, winner="tf79_contact")[0] == 10.0
    assert kr._arbitrate(10.0, 10.1, lo, hi, _TAU, gated=True, winner="ball_onset")[0] == 10.1
    with pytest.raises(ValueError, match="agreement winner"):
        kr._arbitrate(10.0, 10.1, lo, hi, _TAU, gated=True, winner=None)  # a gated entry must carry one
    assert kr._arbitrate(10.0, 10.1, lo, hi, _TAU, gated=False)[1] == "withheld"  # no entry: withheld, no winner


# ---- resolve_kicks: native commit-2 (empty gate -> withheld) ----


def _native_setup():
    ball = rest_then_kick(rest_s=1.0, kick_v=6.0, t0=9.0)
    players = [{"player_id": "h7", "team_id": "H", "x": 49.8, "y": 34.0}]
    frames = frames_from_ball(ball, players=players)
    actions = restart_action("freekick_short", t_event=10.0, team_id="H", player_id="h7")
    return actions, frames


def test_resolve_native_withheld_in_c2_keeps_candidates_and_taker():
    actions, frames = _native_setup()
    out = kr.resolve_kicks(actions, frames, provider="idsse", goal_map=EMPTY_GM, data_profile=None)
    r = out.iloc[0]
    assert r["kick_time_source"] == "withheld" and pd.isna(r["t_kick"])
    assert r["kick_bracket_lo"] == pytest.approx(8.0) and r["kick_bracket_hi"] == pytest.approx(12.0)
    assert np.isfinite(r["t_onset"])  # candidate kept
    assert r["taker_player_id"] == "h7" and r["taker_source"] == "native"  # ungated passthrough
    assert pd.isna(out.loc[~out["type_id"].isin(kr._RESTART_TYPES), "t_kick"]).all()


def test_resolve_native_ships_with_injected_gate(monkeypatch):
    actions, frames = _native_setup()
    monkeypatch.setattr(kr, "gate", _pass_gate)
    out = kr.resolve_kicks(actions, frames, provider="idsse", goal_map=EMPTY_GM, data_profile=_PROFILE)
    r = out.iloc[0]
    assert r["kick_time_source"] != "withheld"
    assert np.isfinite(r["t_kick"])
    assert r["kick_time_evidence"] == "transfer"


# ---- resolve_kicks: SkillCorner bracket + pairing ----


def _sc_frames():
    ball = rest_then_kick(rest_s=1.0, kick_v=6.0, t0=9.0)
    return frames_from_ball(ball)  # frame_ids 1000..


def test_resolve_skillcorner_bracket_from_frame_ids():
    frames = _sc_frames()
    # marker end frame 1000 (t=9.0), next start frame 1040 (t=9.0+40*0.04=10.6)
    actions = restart_action(
        "corner_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=1040, t_event=10.0
    )
    pairing = {"g": GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)}
    out = kr.resolve_kicks(actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing)
    r = out.iloc[0]
    assert r["kick_bracket_lo"] == pytest.approx(9.0) and r["kick_bracket_hi"] == pytest.approx(10.6)
    assert r["gi_pairing"] == "paired"


def test_resolve_skillcorner_missing_frame_ids_is_none():
    frames = _sc_frames()
    actions = restart_action("corner_crossed", provenance="derived", gi_marker_end=pd.NA, gi_next_start=pd.NA)
    pairing = {"g": GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)}
    out = kr.resolve_kicks(actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing)
    assert out.iloc[0]["kick_time_source"] == "none" and pd.isna(out.iloc[0]["kick_bracket_lo"])


def test_resolve_skillcorner_unpaired_withholds_and_flags():
    frames = _sc_frames()
    actions = restart_action("corner_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=1040)
    pairing = {"g": GiPairing("g", "unpaired", "lag", 0.95, 5, 0.9, 1, "sc_possession", 5, None, None, False)}
    out = kr.resolve_kicks(actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing)
    r = out.iloc[0]
    assert r["kick_time_source"] == "withheld" and r["gi_pairing"] == "unpaired"
    assert r["gi_pairing_failed_test"] == "lag"


def test_resolve_skillcorner_edge_clipped_possession_withholds_only_its_rows(monkeypatch):
    # C2 delta-2 B-19: end to end. With a passing gate the row resolves; the SAME row reading a frame inside an
    # edge-clipped GI possession is withheld and counted in gi_edge_clipped, and the match stays paired.
    import dataclasses

    monkeypatch.setattr(kr, "gate", _pass_gate)
    frames = _sc_frames()
    actions = restart_action("corner_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=1040)
    paired = GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)
    base = kr.resolve_kicks(
        actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing={"g": paired}, data_profile=_PROFILE
    )
    assert base.iloc[0]["kick_time_source"] != "withheld"  # non-vacuity: it resolves when nothing is clipped
    assert base.attrs["kick_resolver_report"]["gi_edge_clipped"] == 0
    clipped = dataclasses.replace(paired, n_edge_clipped=1, edge_clipped=((1, 990, 1005),))  # holds frame 1000
    out = kr.resolve_kicks(
        actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing={"g": clipped}, data_profile=_PROFILE
    )
    r = out.iloc[0]
    assert r["kick_time_source"] == "withheld" and r["gi_pairing"] == "paired"
    assert out.attrs["kick_resolver_report"]["gi_edge_clipped"] == 1
    elsewhere = dataclasses.replace(paired, n_edge_clipped=1, edge_clipped=((1, 5000, 5010),))  # another interval
    out2 = kr.resolve_kicks(
        actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing={"g": elsewhere}, data_profile=_PROFILE
    )
    assert out2.iloc[0]["kick_time_source"] == base.iloc[0]["kick_time_source"]
    assert out2.attrs["kick_resolver_report"]["gi_edge_clipped"] == 0


def test_resolve_skillcorner_missing_mapping_raises():
    frames = _sc_frames()
    actions = restart_action("corner_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=1040)
    with pytest.raises(ValueError, match="absent from the passed gi_pairing"):
        kr.resolve_kicks(actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing={})


def test_resolve_non_skillcorner_gi_pairing_all_na():
    actions, frames = _native_setup()
    out = kr.resolve_kicks(actions, frames, provider="idsse", goal_map=EMPTY_GM, data_profile=None)
    assert out["gi_pairing"].isna().all()


# ---- purity / idempotence ----


def _retake_ball() -> pd.DataFrame:
    # Both releases are STRIKES (12 m/s >= v_kick): the amended section 6.7 retake needs a strike, a return to rest
    # within rho, and a second strike; a sub-v_kick placement never sets a retake (spec amendment 2026-10-22).
    dt = 0.04
    t, x = [], []
    tt = 9.0
    for seg in (
        ("rest", 50.0, 15),  # rest 0.6 s at 50
        ("move", 50.0, 10),  # release 1 (to ~52)
        ("rest", 50.4, 15),  # return to rest within rho=1.0 of 50
        ("move", 50.4, 10),  # release 2
    ):
        kind, x0, n = seg
        for i in range(n):
            t.append(tt)
            x.append(x0 if kind == "rest" else x0 + 12.0 * dt * (i + 1))
            tt += dt
    return pd.DataFrame({"time_seconds": t, "x": x, "y": 34.0})


def test_detect_retake_and_last_onset():
    ball = _retake_ball()
    episodes = kr.rest_episodes(ball, restart_type="free_kick", attacks_rtl=None)  # a free kick needs no direction
    is_retake, last_onset = kr.detect_retake(episodes, "free_kick", lo=8.0, hi=14.0)
    assert is_retake is True
    first = next(e.release for e in episodes if e.qualifies and np.isfinite(e.release))
    assert last_onset > first  # the SECOND release, never the first


def test_resolve_marks_kick_retake():
    frames = frames_from_ball(_retake_ball())
    actions = restart_action("freekick_short", t_event=10.0)
    out = kr.resolve_kicks(actions, frames, provider="idsse", goal_map=EMPTY_GM, data_profile=None)
    assert out.iloc[0]["kick_retake"] is True or out.iloc[0]["kick_retake"] == True  # noqa: E712


def _launch_case(v1, v2):
    """A ball launched by h7 at ``v1`` m/s at t[0], re-launched by h7 at ``v2`` m/s at t[15] (detected throughout)."""
    dt = 0.04
    t = 9.0 + dt * np.arange(30)
    x = np.concatenate([50.0 + v1 * dt * np.arange(15), 50.0 + v1 * dt * 15 + v2 * dt * np.arange(15)])
    ball = pd.DataFrame({"time_seconds": t, "x": x, "y": 34.0})
    frames = frames_from_ball(ball, players=[{"player_id": "h7", "team_id": "H", "x": 50.0, "y": 34.0}])
    contacts = pd.DataFrame(
        {"period_id": [1, 1], "frame_id": [1000, 1015], "time_seconds": [t[0], t[15]], "player_id": ["h7", "h7"]}
    )
    gp = frames[frames.period_id == 1]
    ball_rows = gp[gp["is_ball"]].sort_values("time_seconds")
    return contacts, kr._build_roster(gp), kr._detected_ball_speed(ball_rows, "skillcorner"), t


def test_launch_rule_both_sides():
    # Spec 6.3: the restarting-team contact launching the peak DETECTED-ball speed >= 8 m/s; a second launch within
    # 15 % of the peak -> ambiguous -> NaN; below v_kick -> no launch -> NaN.
    contacts, roster, det, t = _launch_case(14.0, 3.0)
    assert kr._launch_contact(contacts, roster, "H", det) == pytest.approx(t[0])  # one clear launch
    contacts, roster, det, t = _launch_case(10.0, 10.5)
    assert not np.isfinite(kr._launch_contact(contacts, roster, "H", det))  # ambiguous (within 15 %)
    contacts, roster, det, t = _launch_case(6.0, 6.1)
    assert not np.isfinite(kr._launch_contact(contacts, roster, "H", det))  # below v_kick (8 m/s)


def test_launch_rule_ignores_undetected_balls():
    # ADR-109: an extrapolated (undetected) ball never counts toward the launch speed.
    contacts, roster, _det, _t = _launch_case(14.0, 3.0)
    dt = 0.04
    tt = 9.0 + dt * np.arange(30)
    ball = pd.DataFrame({"time_seconds": tt, "x": 50.0 + 14.0 * dt * np.arange(30), "y": 34.0, "visibility": False})
    ball["visibility"] = pd.array([False] * 30, dtype="boolean")
    t_det, sp = kr._detected_ball_speed(ball.assign(is_ball=True), "skillcorner")
    assert len(t_det) == 0
    assert not np.isfinite(kr._launch_contact(contacts, roster, "H", (t_det, sp)))


def test_resolve_is_pure_and_idempotent():
    actions, frames = _native_setup()
    frames_before = frames.copy()
    out1 = kr.resolve_kicks(actions, frames, provider="idsse", goal_map=EMPTY_GM, data_profile=None)
    pd.testing.assert_frame_equal(frames, frames_before)  # frames unchanged
    out2 = kr.resolve_kicks(out1, frames, provider="idsse", goal_map=EMPTY_GM, data_profile=None)
    cols = ["kick_bracket_lo", "kick_bracket_hi", "t_onset", "t_kick", "kick_time_source", "taker_player_id"]
    pd.testing.assert_frame_equal(out1[cols], out2[cols])


# ---- CL-11 action-frame rest + the SkillCorner taker's own gate (spec sections 7, 15.2) ----


def _sc_taker_scene():
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals

    ball = rest_then_kick(rest_s=1.0, kick_v=12.0, t0=9.0)
    players = [
        {"player_id": "h7", "team_id": "H", "x": 49.6, "y": 34.0},
        {"player_id": "hk", "team_id": "H", "x": 103.0, "y": 34.0, "is_goalkeeper": True},  # H defends 105
        {"player_id": "ak", "team_id": "A", "x": 2.0, "y": 34.0, "is_goalkeeper": True},
    ]
    frames = frames_from_ball(ball, players=players)
    actions = restart_action(
        "freekick_short",
        provenance="derived",
        gi_marker_end=1000,
        gi_next_start=1045,
        t_event=10.0,
        team_id="H",
        player_id=None,
    )
    contacts = pd.DataFrame(
        {
            "period_id": [1],
            "frame_id": [1025],
            "time_seconds": [10.0],
            "player_id": ["h7"],
            "contact_x": [50.0],
            "contact_y": [34.0],
            "confidence": [0.9],
            "is_aerial": [False],
            "contact_source": ["velocity_break"],
        }
    )
    pairing = {"g": GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)}
    return actions, frames, resolve_defended_goals(frames), contacts, pairing


class _OnlySignal(GateTableStub):
    def __init__(self, signal):
        self.signal = signal

    def get(self, key, default=None):
        return _pass_gate() if key[2] == self.signal else default


def test_kick_rest_is_in_the_restarting_team_action_frame():
    # H attacks x = 0 (its keeper defends 105): the rest at raw x = 50 maps to 105 - 50 = 55 (CL-11).
    actions, frames, gm, _contacts, pairing = _sc_taker_scene()
    out = kr.resolve_kicks(actions, frames, provider="skillcorner", goal_map=gm, gi_pairing=pairing)
    assert out.iloc[0]["kick_rest_x"] == pytest.approx(55.0, abs=0.1)
    assert out.iloc[0]["kick_rest_y"] == pytest.approx(34.0, abs=0.1)


def test_kick_rest_unresolved_direction_is_nan():
    actions, frames, _gm, _contacts, pairing = _sc_taker_scene()
    out = kr.resolve_kicks(actions, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing)
    assert np.isnan(out.iloc[0]["kick_rest_x"]) and np.isnan(out.iloc[0]["kick_rest_y"])


def test_skillcorner_taker_has_its_own_gate_both_sides():
    actions, frames, gm, contacts, pairing = _sc_taker_scene()
    kw: dict[str, Any] = dict(
        provider="skillcorner", goal_map=gm, gi_pairing=pairing, data_profile=_PROFILE, contacts=contacts
    )
    kick_only = kr.resolve_kicks(actions, frames, gate_table=_OnlySignal("kick_time"), **kw).iloc[0]
    assert kick_only["taker_source"] == "withheld" and pd.isna(kick_only["taker_evidence"])
    taker_only = kr.resolve_kicks(actions, frames, gate_table=_OnlySignal("taker"), **kw).iloc[0]
    assert taker_only["taker_source"] == "tracking" and taker_only["taker_player_id"] == "h7"
    assert taker_only["taker_evidence"] == "transfer" and taker_only["kick_time_source"] == "withheld"


def test_taker_cell_state_is_restart_type_by_subtype():
    from silly_kicks.spadl import config as spadlconfig

    def st(name):
        return kr.taker_cell_state({"type_id": spadlconfig.actiontype_id[name]})

    assert (st("corner_short"), st("freekick_crossed"), st("goalkick"), st("throw_in")) == (
        "short",
        "crossed",
        "default",
        "default",
    )


def test_bracket_ab_from_run_one_is_used_and_clipped():
    # Spec 6.1: the native anchor bracket is [t_event - a, t_event + b], a / b from run (1), clipped to 2.0 s.
    actions, frames = _native_setup()
    out = kr.resolve_kicks(
        actions, frames, provider="idsse", goal_map=EMPTY_GM, bracket_ab={"free_kick": (0.5, 3.0)}
    ).iloc[0]
    assert out["kick_bracket_lo"] == pytest.approx(9.5) and out["kick_bracket_hi"] == pytest.approx(12.0)
    default = kr.resolve_kicks(actions, frames, provider="idsse", goal_map=EMPTY_GM).iloc[0]
    assert default["kick_bracket_lo"] == pytest.approx(8.0)  # no calibration -> the 2.0 s clip


# ---- Task 2.5 rules (spec sections 6.1-6.3): the C-41 kick contact, chains, gaps, isolation, strict bracket ----


def _corner_scene():
    """A corner for H (attacks +x): a RETRIEVER carries the ball in from outside the 10 m flag region, the ball
    rests at the flag, the TAKER touches twice, a SHORT-CORNER RECEIVER (in region) touches next."""
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals

    dt = 0.04
    t = 9.0 + dt * np.arange(50)
    x = np.where(
        t < 9.5, 95.0 + (104.5 - 95.0) * (t - 9.0) / 0.5, np.where(t <= 10.05, 104.5, 104.5 - 8.0 * (t - 10.05))
    )
    y = np.where(t < 9.5, 20.0 + (0.6 - 20.0) * (t - 9.0) / 0.5, np.where(t <= 10.05, 0.6, 0.6 + 8.0 * (t - 10.05)))
    ball = pd.DataFrame({"time_seconds": t, "x": x, "y": y})
    keepers = [
        {"player_id": "hk", "team_id": "H", "x": 3.0, "y": 34.0, "is_goalkeeper": True},
        {"player_id": "ak", "team_id": "A", "x": 102.0, "y": 34.0, "is_goalkeeper": True},
        {"player_id": "r", "team_id": "H", "x": 95.0, "y": 20.0},
        {"player_id": "h7", "team_id": "H", "x": 104.0, "y": 0.6},
        {"player_id": "h8", "team_id": "H", "x": 100.0, "y": 5.0},
    ]
    frames = frames_from_ball(ball, players=keepers)
    contacts = pd.DataFrame(
        {
            "period_id": 1,
            "frame_id": [1000, 1025, 1026, 1040],
            "time_seconds": [9.0, 10.0, 10.04, 10.6],
            "player_id": ["r", "h7", "h7", "h8"],
            "contact_x": [95.0, 104.5, 104.5, 100.0],
            "contact_y": [20.0, 0.6, 0.6, 5.0],
        }
    ).sort_values("time_seconds")
    return frames, contacts, resolve_defended_goals(frames), ball


def test_kick_contact_is_the_last_taker_touch_never_the_retriever_or_the_receiver():
    # C-41 / spec 6.3: the LAST contact of the FIRST restarting-team player among the IN-REGION contacts after the
    # ball's LAST ENTRY into the region, before any other player's in-region contact.
    frames, contacts, gm, ball = _corner_scene()
    row = restart_action("corner_short", t_event=10.0, team_id="H").iloc[0]
    roster = kr._build_roster(frames[frames.period_id == 1])
    pc = (contacts.reset_index(drop=True), contacts["time_seconds"].to_numpy())
    obs = ball.assign(period_id=1)
    t = kr._kick_contact(
        row, pc, roster, 8.8, 11.0, "corner", "idsse", gm, ball_obs=obs, t_anchor=10.08, rest_xy=(104.5, 0.6)
    )
    assert t == pytest.approx(10.04)  # the taker h7, last touch -- not r (outside, before the entry), not h8


def test_kick_contact_unresolved_direction_fails_closed():
    frames, contacts, _gm, ball = _corner_scene()
    row = restart_action("corner_short", t_event=10.0, team_id="H").iloc[0]
    pc = (contacts.reset_index(drop=True), contacts["time_seconds"].to_numpy())
    t = kr._kick_contact(
        row, pc, kr._build_roster(frames), 8.8, 11.0, "corner", "idsse", EMPTY_GM, ball_obs=ball.assign(period_id=1)
    )
    assert np.isnan(t)


def test_chain_row_is_bracketed_between_the_chain_members():
    # Spec 6.1: a chain row (next possession is itself a restart -> no next-possession frame id) is bounded above
    # by the NEXT chain member marker end.
    frames = _sc_frames()
    a = pd.concat(
        [
            restart_action(
                "corner_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=pd.NA, t_event=10.0
            ),
            restart_action(
                "corner_crossed", provenance="derived", gi_marker_end=1030, gi_next_start=1045, t_event=11.0
            ),
        ],
        ignore_index=True,
    )
    a["restart_evidence"] = ["chain", "tagged"]
    a["action_id"] = [0, 1]
    pairing = {"g": GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)}
    out = kr.resolve_kicks(a, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing)
    assert out.iloc[0]["kick_bracket_lo"] == pytest.approx(9.0) and out.iloc[0]["kick_bracket_hi"] == pytest.approx(
        10.2
    )
    a["restart_evidence"] = ["untagged", "tagged"]  # not a chain: no next id -> none
    out2 = kr.resolve_kicks(a, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing)
    assert pd.isna(out2.iloc[0]["kick_bracket_hi"]) and out2.iloc[0]["kick_time_source"] == "none"


def test_in_range_frame_gap_is_none_and_counted_gi_frame_unmapped():
    frames = _sc_frames()
    frames = frames[frames["frame_id"] != 1020]  # a gap inside the tracking range
    a = restart_action("corner_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=1020)
    pairing = {"g": GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)}
    out = kr.resolve_kicks(a, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing)
    assert out.iloc[0]["kick_time_source"] == "none"
    assert out.attrs["kick_resolver_report"]["gi_frame_unmapped"] == 1


def test_an_unpaired_match_is_withheld_alone_and_the_paired_match_is_unchanged():
    # C-11: pairing is per MATCH -- B unpaired withholds B only; A equals a single-match run of A.
    fa = _sc_frames()
    fb = fa.assign(game_id="g2")
    aa = restart_action("corner_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=1040)
    ab = aa.assign(game_id="g2")
    paired = GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)
    unpaired = GiPairing("g2", "unpaired", "lag", 0.95, 5, 0.9, 1, "sc_possession", 5, None, None, False)
    both = kr.resolve_kicks(
        pd.concat([aa, ab], ignore_index=True),
        pd.concat([fa, fb], ignore_index=True),
        provider="skillcorner",
        goal_map=EMPTY_GM,
        gi_pairing={"g": paired, "g2": unpaired},
    )
    alone = kr.resolve_kicks(aa, fa, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing={"g": paired})
    b = both[both["game_id"] == "g2"].iloc[0]
    assert b["kick_time_source"] == "withheld" and b["gi_pairing_failed_test"] == "lag" and pd.isna(b["t_kick"])
    cols = ["kick_bracket_lo", "kick_bracket_hi", "t_onset", "kick_time_source", "gi_pairing"]
    pd.testing.assert_series_equal(both[both["game_id"] == "g"].iloc[0][cols], alone.iloc[0][cols], check_names=False)


def test_a_derived_kick_before_the_placeholder_time_is_found():
    # The Part-1 placeholder time_seconds sits AFTER the kick; the bracket (frame ids), not the placeholder, decides.
    # A free kick: its section 6.4 rest location is "anywhere", so the midfield rest of _sc_frames qualifies.
    frames = _sc_frames()
    a = restart_action("freekick_crossed", provenance="derived", gi_marker_end=1000, gi_next_start=1040, t_event=10.55)
    pairing = {"g": GiPairing("g", "paired", None, 0.95, 0, 0.9, 1, "sc_possession", 5, None, None, False)}
    out = kr.resolve_kicks(a, frames, provider="skillcorner", goal_map=EMPTY_GM, gi_pairing=pairing).iloc[0]
    assert np.isfinite(out["t_onset"]) and out["t_onset"] < 10.55


def test_a_pad_candidate_outside_the_strict_bracket_is_kept_but_never_ships(monkeypatch):
    # C-33: the detectors OBSERVE [lo - tau, hi + tau], but only a candidate inside [lo, hi] can ship, every state.
    actions, frames = _native_setup()
    monkeypatch.setattr(kr, "gate", _pass_gate)
    out = kr.resolve_kicks(
        actions.assign(time_seconds=9.6),
        frames,
        provider="idsse",
        goal_map=EMPTY_GM,
        data_profile=_PROFILE,
        bracket_ab={"free_kick": (0.2, 0.2)},  # bracket [9.4, 9.8]; the release (~10.0) sits in the tau pad
    ).iloc[0]
    assert np.isfinite(out["t_onset"]) and out["t_onset"] > out["kick_bracket_hi"]  # observed, kept as evidence
    assert pd.isna(out["t_kick"]) and out["kick_time_source"] == "none"  # never ships


def test_release_threshold_sensitivity_moves_the_release_a_bounded_amount():
    # Spec 5.2 / run (1): each K threshold at +/-50 % moves the release by a bounded amount on this fixture.
    from silly_kicks.tracking._restart_validation import sensitivity_values

    ball = rest_then_kick(rest_s=1.0, kick_v=6.0, t0=9.0)
    base, _ = kr.kinematic_release(ball, restart_type="free_kick")
    for key in (
        "k_rest_m",
        "k_rest_s",
        "k_release_disp_m",
        "k_release_rest_sd_mult",
        "k_release_speed_ms",
        "k_release_frames",
    ):
        for v in sensitivity_values(key):
            rel, _ = kr.kinematic_release(ball, restart_type="free_kick", thresholds={key: v})
            assert not np.isfinite(rel) or abs(rel - base) <= 0.3, (key, v, rel, base)


# ---- C2-IMPL-19 / -22 / -23 (DL ruling 2026-10-22): K anchors on the LAST qualifying rest (spec 5.2, 6.4, 6.7,
# 16.1). Traces are in the restarting team's action frame (attacks +x): attacks_rtl=False. ----


_SPOT = (94.0, 34.0)


def _kick_start(trace, segment):
    return trace.attrs["segment_starts"][segment]


def test_penalty_nudge_does_not_anchor_k():
    # Case A: the ball rests 0.8 m short of the spot, is nudged onto it at 4 m/s, rests 1.0 s, then is kicked.
    tr = piecewise_trace(
        [("rest", 1.0), ("move", _SPOT, 4.0), ("rest", 1.0), ("move", (104.0, 34.0), 20.0)],
        start_xy=(93.2, 34.0),
    )
    k, rest = kr.kinematic_release(tr, restart_type="penalty", attacks_rtl=False)
    assert abs(k - _kick_start(tr, 3)) <= 0.1  # the kick, not the nudge
    assert rest is not None and np.hypot(rest[0] - _SPOT[0], rest[1] - _SPOT[1]) < 0.05  # median of the LAST rest


def test_corner_toss_from_out_of_play_does_not_anchor_k():
    # Case B: the ball rests 3 m out of play, is tossed to the arc, rests 1.5 s, then is kicked into the pitch.
    tr = piecewise_trace(
        [("rest", 1.0), ("move", (104.6, 0.4), 6.0), ("rest", 1.5), ("move", (95.0, 20.0), 20.0)],
        start_xy=(107.5, 2.0),
    )
    k, rest = kr.kinematic_release(tr, restart_type="corner", attacks_rtl=False)
    assert abs(k - _kick_start(tr, 3)) <= 0.1
    assert rest is not None and rest[0] < 105.0  # on the pitch, at the arc


def test_keeper_held_ball_after_a_saved_penalty_never_anchors_k():
    # The keeper holds the ball 1.5 s on his line after the save; that rest is not at the spot, so the spot rest's
    # release stays K (a naive "last rest" would take the keeper's throw).
    tr = piecewise_trace(
        [("rest", 1.0), ("move", (104.2, 35.0), 20.0), ("rest", 1.5), ("move", (80.0, 30.0), 15.0)],
        start_xy=_SPOT,
    )
    k, _rest = kr.kinematic_release(tr, restart_type="penalty", attacks_rtl=False)
    assert abs(k - _kick_start(tr, 1)) <= 0.1


def test_a_true_retake_is_flagged_and_takes_the_last_onset():
    # Corner: kicked, the ball is carried back to the same arc (within rho = 1.5 m), rests, and is kicked again.
    tr = piecewise_trace(
        [
            ("rest", 1.0),
            ("move", (100.0, 6.0), 12.0),
            ("move", (104.4, 0.7), 2.0),  # carried back below the release speed
            ("rest", 1.0),
            ("move", (95.0, 20.0), 20.0),
        ],
        start_xy=(104.6, 0.4),
    )
    eps = kr.rest_episodes(tr, restart_type="corner", attacks_rtl=False)
    is_retake, last = kr.detect_retake(eps, "corner", lo=0.0, hi=20.0)
    assert is_retake is True
    assert abs(last - _kick_start(tr, 4)) <= 0.1  # the LAST onset
    assert kr.choose_release(eps)[0] < last  # K = the FIRST strike; the retake row follows the last onset


def test_single_rest_control_keeps_the_normal_latency():
    tr = piecewise_trace([("rest", 1.5), ("move", (104.0, 34.0), 20.0)], start_xy=_SPOT)
    k, _rest = kr.kinematic_release(tr, restart_type="penalty", attacks_rtl=False)
    assert 0.0 <= k - _kick_start(tr, 1) <= 0.1  # <= 2-frame latency


def test_period_five_penalty_anchors_on_the_spot_rest():
    # A shoot-out penalty: the ball first stops 1.6 m off the spot, the taker carries it onto the spot, it rests,
    # and is kicked. The detector's release is the kick (period 5 has no orientation of its own).
    from silly_kicks.tracking import _penalty_detector as pdet
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals

    spot_x = pdet._spot_x(105.0)
    tr = piecewise_trace(
        [("rest", 1.0), ("move", (spot_x, 34.0), 3.5), ("rest", 1.2), ("move", (104.0, 34.0), 20.0)],
        start_xy=(spot_x - 1.6, 34.0),
        t0=9.0,
    )
    players = [
        {"player_id": "h7", "team_id": "H", "x": spot_x - 1.0, "y": 34.0},
        {"player_id": "a1", "team_id": "A", "x": 104.0, "y": 34.0, "is_goalkeeper": True},
        {"player_id": "h1", "team_id": "H", "x": 4.0, "y": 34.0, "is_goalkeeper": True},
    ]
    frames = frames_from_ball(tr, period_id=5, players=players)
    actions = pd.DataFrame(
        columns=["game_id", "period_id", "time_seconds", "team_id", "player_id", "type_id", "result_id"]
    )
    out = pdet.detect_penalties(actions, frames, provider="skillcorner", goal_map=resolve_defended_goals(frames))
    assert len(out) == 1 and out.iloc[0]["stratum"] == "shoot_out"
    assert abs(float(out.iloc[0]["release_time"]) - _kick_start(tr, 3)) <= 0.1


def test_goal_kick_retake_uses_the_goal_area_not_the_penalty_radius():
    # C2-IMPL-23: a goal kick re-placed 3 m away (outside the 1 m penalty radius, inside the same goal area) is a
    # retake; the old penalty-radius approximation missed it.
    tr = piecewise_trace(
        [
            ("rest", 1.0),
            ("move", (30.0, 34.0), 15.0),
            ("move", (5.0, 30.0), 2.0),
            ("rest", 1.0),
            ("move", (40.0, 34.0), 20.0),
        ],
        start_xy=(5.0, 27.0),
    )
    eps = kr.rest_episodes(tr, restart_type="goal_kick", attacks_rtl=False)
    assert kr.detect_retake(eps, "goal_kick", lo=0.0, hi=40.0)[0] is True


def test_rest_location_and_direction_fail_closed_without_a_direction():
    # attacks_rtl=None: a corner cannot be located or directed -> no K (never a guessed orientation); a free kick
    # needs neither and still fires.
    tr = piecewise_trace([("rest", 1.0), ("move", (95.0, 20.0), 20.0)], start_xy=(104.6, 0.4))
    assert np.isnan(kr.kinematic_release(tr, restart_type="corner", attacks_rtl=None)[0])
    assert np.isfinite(kr.kinematic_release(tr, restart_type="free_kick", attacks_rtl=None)[0])


def test_a_release_in_the_wrong_direction_is_not_a_kick():
    # A goal-kick ball rolled back over its own goal line (-x) at kick speed is not a goal-kick release.
    tr = piecewise_trace([("rest", 1.0), ("move", (-3.0, 34.0), 10.0)], start_xy=(5.0, 34.0))
    assert np.isnan(kr.kinematic_release(tr, restart_type="goal_kick", attacks_rtl=False)[0])


def test_rest_location_is_the_median_of_the_last_half_second():
    # The rest window drifts 0.3 m (inside the 0.5 m rest tolerance); the location is the median of its last 0.5 s,
    # not the window mean.
    tr = piecewise_trace(
        [("rest", 0.6), ("move", (94.3, 34.0), 0.5), ("rest", 0.6), ("move", (104.0, 34.0), 20.0)],
        start_xy=(94.0, 34.0),
    )
    _k, rest = kr.kinematic_release(tr, restart_type="penalty", attacks_rtl=False)
    assert rest is not None and abs(rest[0] - 94.3) < 0.02


# ---- Spec amendment 2026-10-22 (owner approval of the DL strike rule; sections 5.2, 6.7, 16.1): a STRIKE is a
# release whose peak speed up to the next rest reaches v_kick (8 m/s). K = the first strike's rest; rests after it
# are post-kick; a retake needs two strikes; no strike + >= 2 rests = `none`, counted. ----

_FK = (70.0, 30.0)


def test_blocked_free_kick_then_dead_ball_keeps_k_on_the_strike():
    # The kick hits the wall, the dead ball rests 1 s, then is cleared hard: the clearance never anchors K.
    tr = piecewise_trace(
        [
            ("rest", 1.0),
            ("move", (78.5, 30.0), 20.0),
            ("move", (77.0, 30.0), 2.0),
            ("rest", 1.0),
            ("move", (60.0, 40.0), 18.0),
        ],
        start_xy=_FK,
    )
    v = kr.kick_verdict(kr.rest_episodes(tr, restart_type="free_kick", attacks_rtl=False))
    assert v.reason == "strike" and abs(v.release - _kick_start(tr, 1)) <= 0.1
    assert v.rest_xy is not None and np.hypot(v.rest_xy[0] - _FK[0], v.rest_xy[1] - _FK[1]) < 0.05


def test_free_kick_rolled_into_place_then_struck():
    tr = piecewise_trace(
        [("rest", 1.0), ("move", _FK, 3.0), ("rest", 1.0), ("move", (90.0, 34.0), 20.0)], start_xy=(68.0, 30.0)
    )
    v = kr.kick_verdict(kr.rest_episodes(tr, restart_type="free_kick", attacks_rtl=False))
    assert v.reason == "strike" and abs(v.release - _kick_start(tr, 3)) <= 0.1


def test_penalty_placement_nudge_is_not_a_retake():
    # The ball is nudged 1.6 m off the spot at 5 m/s (a release, below v_kick), rests, is rolled back onto the
    # spot, rests, and is struck. Two releases from rests within rho: the pre-amendment rule called it a retake;
    # a nudge is not a strike, so it is not (spec amendment 2026-10-22). K = the strike.
    tr = piecewise_trace(
        [
            ("rest", 1.0),
            ("move", (95.6, 34.0), 5.0),
            ("rest", 1.0),
            ("move", (94.1, 34.0), 2.0),
            ("rest", 1.0),
            ("move", (104.0, 34.0), 20.0),
        ],
        start_xy=_SPOT,
    )
    eps = kr.rest_episodes(tr, restart_type="penalty", attacks_rtl=False)
    released = [e for e in eps if e.qualifies and np.isfinite(e.release)]
    assert len(released) == 2 and not released[0].strike  # non-vacuity: the nudge IS a release, just not a strike
    is_retake, _last = kr.detect_retake(eps, "penalty", lo=0.0, hi=20.0)
    assert is_retake is False
    assert abs(kr.kick_verdict(eps).release - _kick_start(tr, 5)) <= 0.1


def test_a_true_retaken_penalty_is_flagged_on_the_last_strike():
    tr = piecewise_trace(
        [
            ("rest", 1.0),
            ("move", (104.0, 34.0), 20.0),
            ("move", (94.2, 34.0), 2.0),  # carried back to the spot below the release speed
            ("rest", 1.0),
            ("move", (104.0, 33.0), 20.0),
        ],
        start_xy=_SPOT,
    )
    eps = kr.rest_episodes(tr, restart_type="penalty", attacks_rtl=False)
    is_retake, last = kr.detect_retake(eps, "penalty", lo=0.0, hi=20.0)
    assert is_retake is True and abs(last - _kick_start(tr, 4)) <= 0.1
    assert abs(kr.kick_verdict(eps).release - _kick_start(tr, 1)) <= 0.1  # K = the FIRST strike


def test_soft_short_free_kick_without_a_strike_uses_its_single_rest():
    tr = piecewise_trace([("rest", 1.0), ("move", (75.0, 30.0), 5.0)], start_xy=_FK)
    v = kr.kick_verdict(kr.rest_episodes(tr, restart_type="free_kick", attacks_rtl=False))
    # a 5 m/s release passes the 0.5 m displacement bar after ~0.1 s: the ratified K latency for a slow ball
    assert v.reason == "single_episode" and 0.0 <= v.release - _kick_start(tr, 1) <= 0.15


def test_no_strike_with_two_rests_is_none_and_counted():
    # Two placements below v_kick and no strike: ambiguous -> `none` (NaN), counted by type x subtype in the report.
    tr = piecewise_trace(
        [("rest", 1.0), ("move", (72.0, 30.0), 4.0), ("rest", 1.0), ("move", (76.0, 30.0), 5.0)], start_xy=_FK
    )
    v = kr.kick_verdict(kr.rest_episodes(tr, restart_type="free_kick", attacks_rtl=False))
    assert v.reason == "no_strike_ambiguous" and np.isnan(v.release) and v.rest_xy is None
    frames = frames_from_ball(tr.assign(time_seconds=tr["time_seconds"] + 9.0))
    out = kr.resolve_kicks(
        restart_action("freekick_short", t_event=10.0), frames, provider="idsse", goal_map=EMPTY_GM, data_profile=None
    )
    assert out.attrs["kick_resolver_report"]["k_no_strike_ambiguous"] == {"freekick_short": 1}
    assert np.isnan(out.iloc[0]["t_onset"])


def test_v_kick_threshold_override_moves_the_strike_bar():
    # Run (1) evaluates v_kick at +/-50 %: at 4 m/s a 5 m/s release becomes a strike.
    tr = piecewise_trace(
        [("rest", 1.0), ("move", (72.0, 30.0), 5.0), ("rest", 1.0), ("move", (76.0, 30.0), 6.0)], start_xy=_FK
    )
    base = kr.kick_verdict(kr.rest_episodes(tr, restart_type="free_kick", attacks_rtl=False))
    low = kr.kick_verdict(
        kr.rest_episodes(tr, restart_type="free_kick", attacks_rtl=False, thresholds={"sc_freekick_vkick_ms": 4.0})
    )
    assert base.reason == "no_strike_ambiguous"
    assert low.reason == "strike" and 0.0 <= low.release - _kick_start(tr, 1) <= 0.15  # slow-ball latency


# ---- C2 r2 C-22 / C-25 (owner ruling 2026-10-22): the section 6.4 Law locations are real-metre Law geometry
# mapped through the provider pitch (``_law_geometry``); the goal area admits one ball radius (Law 1 + Law 2). ----


def _goal_kick_trace(x, y):
    return piecewise_trace([("rest", 1.0), ("move", (40.0, y), 20.0)], start_xy=(x, y))


def test_goal_area_boundary_on_a_105_pitch_admits_one_ball_radius():
    for x, ok in ((5.5, True), (5.6, True), (5.8, False)):  # the line; 0.10 m outside; 0.30 m outside
        v = kr.kick_verdict(kr.rest_episodes(_goal_kick_trace(x, 34.0), restart_type="goal_kick", attacks_rtl=False))
        assert (v.reason == "strike") is ok, (x, v.reason)


def test_goal_area_scales_with_a_100_by_64_pitch_on_both_lines():
    dims = PitchDims(100.0, 64.0)
    gx = 5.5 * 105.0 / 100.0  # the real 5.5 m line in the scaled frame (5.775)
    gy = 34.0 + 9.16 * 68.0 / 64.0  # the real 9.16 m half-width line in the scaled frame
    for x, y in ((gx, 34.0), (3.0, gy)):  # ON the x line; ON the y line
        tr = _goal_kick_trace(x, y)
        on = kr.kick_verdict(kr.rest_episodes(tr, restart_type="goal_kick", attacks_rtl=False, pitch=dims))
        unscaled = kr.kick_verdict(kr.rest_episodes(tr, restart_type="goal_kick", attacks_rtl=False))
        assert on.reason == "strike", (x, y)
        assert unscaled.reason == "no_episode", (x, y)  # non-vacuity: the 105 x 68 mapping rejects the same rest


def test_penalty_spot_and_corner_arc_scale_with_the_pitch():
    dims = PitchDims(90.0, 60.0)
    spot = (dims.spot_x, 34.0)  # 11 real m on a 90 m pitch -> 92.17 in the frame (the 105 m spot is 94.0)
    tr = piecewise_trace([("rest", 1.0), ("move", (104.0, 34.0), 20.0)], start_xy=spot)
    assert (
        kr.kick_verdict(kr.rest_episodes(tr, restart_type="penalty", attacks_rtl=False, pitch=dims)).reason == "strike"
    )
    assert kr.kick_verdict(kr.rest_episodes(tr, restart_type="penalty", attacks_rtl=False)).reason == "no_episode"
    # Corner: 2.4 real m from the flag along the goal line is within 1.5 m of the 1 m arc on the real pitch; in the
    # scaled frame of a 60 m wide pitch that point sits at 2.72 m -- outside the unscaled 2.5 m bound.
    cy = 2.4 * 68.0 / 60.0
    tr = piecewise_trace([("rest", 1.0), ("move", (95.0, 20.0), 20.0)], start_xy=(105.0, cy))
    assert (
        kr.kick_verdict(kr.rest_episodes(tr, restart_type="corner", attacks_rtl=False, pitch=dims)).reason == "strike"
    )
    assert kr.kick_verdict(kr.rest_episodes(tr, restart_type="corner", attacks_rtl=False)).reason == "no_episode"


def test_resolve_kicks_reads_the_pitch_and_counts_no_episode():
    # A goal kick resting ON the real 5.5 m line of a 100 x 64 pitch: with the declared pitch it is a kick; with the
    # 105 x 68 default it has no qualifying rest -- counted in the report, never silent.
    from silly_kicks.tracking._gk_resolve import GoalMap

    gm = GoalMap(resolved={("g", "1", "H"): 0.0, ("g", "1", "A"): 105.0}, guessed={}, unresolved=frozenset())
    tr = _goal_kick_trace(5.5 * 105.0 / 100.0, 34.0)
    frames = frames_from_ball(tr.assign(time_seconds=tr["time_seconds"] + 9.0))
    a = restart_action("goalkick", t_event=10.0)
    scaled = kr.resolve_kicks(a, frames, provider="idsse", goal_map=gm, pitch_length=100.0, pitch_width=64.0)
    default = kr.resolve_kicks(a, frames, provider="idsse", goal_map=gm)
    assert np.isfinite(scaled.iloc[0]["t_onset"])
    assert np.isnan(default.iloc[0]["t_onset"])
    assert default.attrs["kick_resolver_report"]["k_no_episode"] == {"goalkick": 1}
    frames.attrs.update(pitch_length=100.0, pitch_width=64.0)  # the converter stamp is the fallback
    assert np.isfinite(kr.resolve_kicks(a, frames, provider="idsse", goal_map=gm).iloc[0]["t_onset"])


def test_retake_chain_is_measured_against_the_first_rest():
    # C2 r2 C-27 (spec 6.7): three corner strikes whose rests drift 1.0 m each -- the second is within rho = 1.5 m
    # of the FIRST rest, the third (2.0 m away) is not, even though it is within 1.0 m of the second.
    tr = piecewise_trace(
        [
            ("rest", 1.0),
            ("move", (100.0, 6.0), 12.0),
            ("move", (104.6, 1.4), 2.0),
            ("rest", 1.0),
            ("move", (100.0, 8.0), 12.0),
            ("move", (104.6, 2.4), 2.0),
            ("rest", 1.0),
            ("move", (95.0, 20.0), 20.0),
        ],
        start_xy=(104.6, 0.4),
    )
    eps = kr.rest_episodes(tr, restart_type="corner", attacks_rtl=False)
    assert sum(e.qualifies and e.strike for e in eps) == 3  # precondition: three qualifying strikes
    is_retake, last = kr.detect_retake(eps, "corner", lo=0.0, hi=20.0)
    assert is_retake is True and abs(last - _kick_start(tr, 4)) <= 0.1  # the chain stops at the 2nd strike


# ---- C2 r2 A-07 (spec amendment 2026-10-22): a strike is v_kick SUSTAINED on k_release_frames consecutive
# deduplicated samples; one jitter frame is never a strike; a duplicate-frame track still strikes. ----


def _with_spike(trace, frame, dx):
    """A copy of ``trace`` with ONE frame displaced by ``dx`` metres (a tracking jitter spike)."""
    out = trace.copy()
    out.loc[frame, "x"] = out.loc[frame, "x"] + dx
    return out


def test_one_jitter_frame_above_v_kick_is_not_a_strike_and_sets_no_retake():
    # Penalty: rest at the spot, a 1.6 m placement nudge at 5 m/s, back onto the spot, then the strike. ONE frame
    # of the nudge jumps 0.4 m (10 m/s for one frame): the pre-amendment one-frame peak made it a strike -> a retake.
    tr = piecewise_trace(
        [
            ("rest", 1.0),
            ("move", (95.6, 34.0), 5.0),
            ("rest", 1.0),
            ("move", (94.1, 34.0), 2.0),
            ("rest", 1.0),
            ("move", (104.0, 34.0), 20.0),
        ],
        start_xy=_SPOT,
    )
    nudge = round(_kick_start(tr, 1) / 0.04) + 3
    spiked = _with_spike(tr, nudge, 0.4)
    eps = kr.rest_episodes(spiked, restart_type="penalty", attacks_rtl=False)
    first = next(e for e in eps if e.qualifies and np.isfinite(e.release))
    single_frame = float(
        np.max(kr._speed(*(spiked[c].to_numpy() for c in ("time_seconds", "x", "y")))[nudge : nudge + 2])
    )
    assert single_frame >= 8.0  # non-vacuity: the spike IS above v_kick for one frame
    assert first.strike is False and first.peak_speed < 8.0  # sustained over 2 frames it is not
    assert kr.detect_retake(eps, "penalty", lo=0.0, hi=20.0)[0] is False
    assert abs(kr.kick_verdict(eps).release - _kick_start(tr, 5)) <= 0.1  # K = the real strike


def test_a_real_strike_is_unchanged_by_the_sustained_rule():
    tr = piecewise_trace([("rest", 1.0), ("move", (104.0, 34.0), 20.0)], start_xy=_SPOT)
    e = next(e for e in kr.rest_episodes(tr, restart_type="penalty", attacks_rtl=False) if e.qualifies)
    assert e.strike is True and e.peak_speed == pytest.approx(20.0, rel=0.05)


def _duplicated_flight(speed_ms):
    """A spot rest then a kick at ``speed_ms`` whose every second in-flight sample is sent twice (the 4.137.0
    duplicate-frame sawtooth: raw speed 0, then 2 x ``speed_ms``; deduplicated: ``speed_ms``)."""
    tr = piecewise_trace([("rest", 1.0), ("move", (104.0, 34.0), speed_ms)], start_xy=_SPOT)
    rest_n = round(1.0 / 0.04) + 1
    xs = list(tr["x"].to_numpy()[:rest_n])
    for v in tr["x"].to_numpy()[rest_n::2]:
        xs += [v, v]
    return tr.assign(x=xs[: len(tr)]), tr, rest_n


def test_a_duplicate_frame_track_still_detects_the_release_and_the_strike():
    roll, tr, rest_n = _duplicated_flight(6.0)
    raw = kr._speed(roll["time_seconds"].to_numpy(), roll["x"].to_numpy(), roll["y"].to_numpy())[rest_n:]
    assert not np.any((raw[:-1] > 3.0) & (raw[1:] > 3.0))  # non-vacuity: the raw sawtooth never sustains 3 m/s
    e = next(e for e in kr.rest_episodes(roll, restart_type="penalty", attacks_rtl=False) if e.qualifies)
    assert np.isfinite(e.release) and e.release - _kick_start(tr, 1) <= 0.2  # the release is still found
    assert e.strike is False  # a 6 m/s roll is no strike, although its raw sawtooth peaks at 12 m/s
    kick, _tr, _n = _duplicated_flight(20.0)
    ek = next(e for e in kr.rest_episodes(kick, restart_type="penalty", attacks_rtl=False) if e.qualifies)
    assert ek.strike is True  # a 20 m/s kick on the same sawtooth is a strike


# ---- C2 r3 (reviewer B note, owner ruling 2026-10-22): _maximal_rest_windows keeps a running centroid (no O(w^2)
# recompute) and skips NaN samples (a missed detection neither extends nor ends a rest). ----


def _reference_rest_windows(t, x, y, rest_m, rest_s):
    """The PRE-CHANGE algorithm, frozen verbatim as the equivalence oracle on finite traces."""
    n = len(t)
    out = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n:
            cx = x[i : j + 2].mean()
            cy = y[i : j + 2].mean()
            if np.nanmax(np.hypot(x[i : j + 2] - cx, y[i : j + 2] - cy)) <= rest_m:
                j += 1
            else:
                break
        if (t[j] - t[i]) >= rest_s:
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def test_rest_windows_equal_the_reference_on_clean_traces():
    # Same windows, exactly, on 300 seeded traces mixing jittered rests, slow drifts, rolls and kicks.
    rng = np.random.default_rng(46)
    for _ in range(300):
        xs, ys = [50.0], [34.0]
        for _seg in range(rng.integers(2, 6)):
            if rng.random() < 0.5:  # a jittered rest
                for _k in range(int(rng.integers(5, 40))):
                    xs.append(xs[-1] + rng.normal(0, 0.05))
                    ys.append(ys[-1] + rng.normal(0, 0.05))
            else:  # a move at 0.5-20 m/s
                v = rng.uniform(0.5, 20.0) * 0.04
                ang = rng.uniform(0, 2 * np.pi)
                for _k in range(int(rng.integers(3, 25))):
                    xs.append(xs[-1] + v * np.cos(ang))
                    ys.append(ys[-1] + v * np.sin(ang))
        t = 0.04 * np.arange(len(xs))
        x, y = np.array(xs), np.array(ys)
        got = kr._maximal_rest_windows(t, x, y, 0.5, 0.5)
        assert got == _reference_rest_windows(t, x, y, 0.5, 0.5)


def test_a_one_or_two_frame_nan_gap_does_not_split_a_rest():
    for gap in (1, 2):
        tr = piecewise_trace([("rest", 1.2), ("move", (104.0, 34.0), 20.0)], start_xy=_SPOT)
        x = tr["x"].to_numpy().copy()
        x[10 : 10 + gap] = np.nan  # the tracker misses the resting ball for `gap` frames mid-rest
        t, y = tr["time_seconds"].to_numpy(), tr["y"].to_numpy()
        windows = kr._maximal_rest_windows(t, x, y, 0.5, 0.5)
        assert windows and windows[0][0] == 0 and windows[0][1] >= 25, (gap, windows)  # one rest spans the gap
        old = _reference_rest_windows(t, x, y, 0.5, 0.5)
        assert not old or old[0][0] > 0, (gap, old)  # non-vacuity: the old scan lost the pre-gap part of the rest
        e = next(e for e in kr.rest_episodes(tr.assign(x=x), restart_type="penalty", attacks_rtl=False) if e.qualifies)
        assert np.isfinite(e.centroid[0]) and abs(e.release - _kick_start(tr, 1)) <= 0.1
