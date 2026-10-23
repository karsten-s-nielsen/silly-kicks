"""Task 2.6 missed-penalty detector (spec section 12): detection on the rest-window median, the pitch-scaled
spot, encroachment, the three gated fields, strata, and purity. Both sides of every geometry bar.

The candidate comes from the TRACKING (a ball rest at the pitch-scaled spot), not a window around a
``shot_penalty`` event: a missed penalty with no event row is still found, and an event row (when present)
supplies the result for the gated outcome. The outcome / attribution GI-sequence (rules (a)-(d), the
C-58/C-61 net-clearance operational test) is pinned on ``_outcome`` / ``_live_after_w`` below."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import _penalty_detector as pdet
from silly_kicks.tracking._gk_resolve import resolve_defended_goals
from silly_kicks.tracking._restart_validation import DataProfile, GateEntry

_SP = spadlconfig.actiontype_id["shot_penalty"]
_SPOT_X = pdet._spot_x(105.0)  # ~94.0 on a 105 m pitch
_PROFILE = DataProfile("skillcorner", "r1", "g1", "2025")


def _gate_all(*a, **k):
    return GateEntry("transfer", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell")


def _frames(*, keeper_x=104.0, keeper_y=34.0, encroachers=(), period=1, ball_xy=None, pitch=105.0):
    """A penalty scene in the action frame: ball at the spot, H attacks +x (goal at 105), A defends 105.
    ``encroachers`` = list of (x, y) extra A/H outfielders. ~1 s of rest at 25 fps then the release."""
    ball_xy = ball_xy or (_SPOT_X, 34.0)
    dt = 0.04
    rest_n = 30  # 1.2 s rest
    t = 9.0 + dt * np.arange(rest_n + 10)
    rows = []
    for i, ti in enumerate(t):
        bx = ball_xy[0] if i < rest_n else ball_xy[0] + 6.0 * dt * (i - rest_n + 1)
        rows.append(_row("g", period, 1000 + i, ti, None, None, True, False, bx, ball_xy[1]))
        rows.append(_row("g", period, 1000 + i, ti, "h7", "H", False, False, _SPOT_X - 1.0, 34.0))  # taker near ball
        rows.append(_row("g", period, 1000 + i, ti, "a1", "A", False, True, keeper_x, keeper_y))  # defending GK
        rows.append(_row("g", period, 1000 + i, ti, "h1", "H", False, True, 4.0, 34.0))  # H's own GK (defends 0)
        for j, (ex, ey) in enumerate(encroachers):
            rows.append(_row("g", period, 1000 + i, ti, f"e{j}", "A", False, False, ex, ey))
    return pd.DataFrame(rows)


def _row(g, p, f, t, pid, team, is_ball, is_gk, x, y):
    return {
        "game_id": g,
        "period_id": p,
        "frame_id": f,
        "time_seconds": t,
        "player_id": pid,
        "team_id": team,
        "is_ball": is_ball,
        "is_goalkeeper": is_gk,
        "x": x,
        "y": y,
        "visibility": True,
    }


def _goal_map(frames):
    return resolve_defended_goals(frames)


def _penalty_action(period=1, result="fail"):
    return pd.DataFrame(
        [
            {
                "game_id": "g",
                "period_id": period,
                "time_seconds": 10.2,
                "team_id": "H",
                "player_id": "h7",
                "type_id": _SP,
                "result_id": spadlconfig.result_id[result],
            }
        ]
    )


def test_detects_penalty_on_rest_median():
    frames = _frames()
    out = pdet.detect_penalties(_penalty_action(), frames, provider="skillcorner", goal_map=_goal_map(frames))
    assert len(out) == 1
    assert out.iloc[0]["attacking_team_id"] == "H" and out.iloc[0]["keeper_player_id"] == "a1"
    assert out.iloc[0]["stratum"] == "in_play"


def test_detects_tracking_only_missed_penalty_with_no_event_row():
    # C2-IMPL-02: the candidate comes from the TRACKING. A missed penalty that produced no goal and no
    # reconciliation leaves NO shot_penalty row, yet the ball-rest-at-the-spot scene is still detected.
    frames = _frames()
    out = pdet.detect_penalties(
        pd.DataFrame(columns=["game_id", "period_id", "time_seconds", "team_id", "player_id", "type_id", "result_id"]),
        frames,
        provider="skillcorner",
        goal_map=_goal_map(frames),
    )
    assert len(out) == 1
    assert out.iloc[0]["attacking_team_id"] == "H" and out.iloc[0]["keeper_player_id"] == "a1"


def test_event_row_supplies_result_but_does_not_source_the_candidate(monkeypatch):
    # With an event row present AND the gate injected, the event result reaches the outcome; the candidate is
    # still the tracking rest (the same scene detects with or without the row).
    frames = _frames()
    monkeypatch.setattr(pdet, "gate", _gate_all)
    out = pdet.detect_penalties(
        _penalty_action(result="success"),
        frames,
        provider="skillcorner",
        goal_map=_goal_map(frames),
        data_profile=_PROFILE,
    )
    assert len(out) == 1 and out.iloc[0]["outcome"] in ("goal", "saved", "off_target")


def test_keeper_distance_boundary():
    for kx, detect in [(103.0, True), (102.99, False)]:  # d_gk = 2.0 -> within 2 m of x=105
        frames = _frames(keeper_x=kx)
        out = pdet.detect_penalties(_penalty_action(), frames, provider="skillcorner", goal_map=_goal_map(frames))
        assert (len(out) == 1) is detect


def test_keeper_lateral_band_boundary():
    for ky, detect in [(34.0 + 4.66, True), (34.0 + 4.67, False)]:
        frames = _frames(keeper_y=ky)
        out = pdet.detect_penalties(_penalty_action(), frames, provider="skillcorner", goal_map=_goal_map(frames))
        assert (len(out) == 1) is detect


def test_encroachment_k_limit():
    # k = 2 encroachers inside the arc (dist to spot < 9.15 - 1.0 = 8.15) -> detect; a third -> not.
    two = [(_SPOT_X + 5.0, 34.0), (_SPOT_X - 5.0, 34.0)]
    three = [*two, (_SPOT_X, 34.0 + 5.0)]
    f2 = _frames(encroachers=two)
    f3 = _frames(encroachers=three)
    assert len(pdet.detect_penalties(_penalty_action(), f2, provider="skillcorner", goal_map=_goal_map(f2))) == 1
    assert len(pdet.detect_penalties(_penalty_action(), f3, provider="skillcorner", goal_map=_goal_map(f3))) == 0


def test_shootout_stratum():
    frames = _frames(period=5)
    out = pdet.detect_penalties(_penalty_action(period=5), frames, provider="skillcorner", goal_map=_goal_map(frames))
    assert len(out) == 1 and out.iloc[0]["stratum"] == "shoot_out"


def test_gate_withheld_in_c2_outcome_is_na():
    frames = _frames()
    out = pdet.detect_penalties(_penalty_action(), frames, provider="skillcorner", goal_map=_goal_map(frames))
    assert pd.isna(out.iloc[0]["outcome"]) and pd.isna(out.iloc[0]["attribution"])
    assert pd.isna(out.iloc[0]["penalty_evidence"])


def test_gate_injected_ships_outcome(monkeypatch):
    frames = _frames()
    monkeypatch.setattr(pdet, "gate", _gate_all)
    out = pdet.detect_penalties(
        _penalty_action(result="success"),
        frames,
        provider="skillcorner",
        goal_map=_goal_map(frames),
        data_profile=_PROFILE,
    )
    assert out.iloc[0]["outcome"] in ("goal", "saved", "off_target")
    assert out.iloc[0]["penalty_evidence"] == "transfer"


def test_pitch_scaled_spot_detects_on_100m_and_110m():
    for pitch in (100.0, 110.0):
        spot = pdet._spot_x(pitch)
        frames = _frames(ball_xy=(spot, 34.0))
        frames.attrs["pitch_length"] = pitch
        out = pdet.detect_penalties(_penalty_action(), frames, provider="skillcorner", goal_map=_goal_map(frames))
        assert len(out) == 1  # the spot scales with the pitch -> detection still fires


def _pen_row(result="success"):
    return pd.Series({"period_id": 1, "result_id": spadlconfig.result_id[result]})


def _bline(times, xs, ys=None):
    ys = ys if ys is not None else [34.0] * len(times)
    return pd.DataFrame({"time_seconds": times, "x": xs, "y": ys})


def test_outcome_goal_when_score_change_no_contact_no_live():
    rel = 10.0
    ball = _bline([rel, rel + 0.1, rel + 0.2], [94.0, 96.0, 98.0])  # crosses, no frames after W
    o, a = pdet._outcome(_pen_row("success"), None, ball, rel, "skillcorner", None, 8.0, "a1")
    assert (o, a) == ("goal", "goal")


def test_outcome_saved_when_keeper_contact_and_no_score():
    rel = 10.0
    ball = _bline([rel, rel + 0.1], [94.0, 95.0])
    contacts = pd.DataFrame({"period_id": [1], "time_seconds": [rel + 0.2], "player_id": ["a1"]})  # keeper save
    o, a = pdet._outcome(_pen_row("fail"), None, ball, rel, "skillcorner", contacts, 8.0, "a1")
    assert (o, a) == ("saved", "missed")


def test_outcome_rebound_goal_after_save_stays_saved():
    rel = 10.0
    ball = _bline([rel, rel + 0.1], [94.0, 95.0])
    contacts = pd.DataFrame(
        {"period_id": [1, 1], "time_seconds": [rel + 0.2, rel + 1.0], "player_id": ["a1", "h7"]}  # save + taker rebound
    )
    o, a = pdet._outcome(_pen_row("success"), None, ball, rel, "skillcorner", contacts, 8.0, "a1")
    assert (o, a) == ("saved", "rebound")  # taker goal after a save is a legal rebound; penalty stays saved


def test_live_after_w_parry_is_live_net_clearance_is_not():
    rel = 10.0
    t = [rel + i * 0.1 for i in range(60)]  # out to rel+5.9
    # parry: ball moves in-field at rel+4 (10 m/s) -> live
    xp = [90.0] * 40 + [90.0 + 1.0 * i for i in range(20)]
    assert pdet._live_after_w(_bline(t, xp), rel, 3.0, 8.0) is True
    # net clearance: the live segment starts beyond the goal line between the posts -> excluded
    xn = [94.0] * 40 + [106.0 + 0.5 * i for i in range(20)]
    assert pdet._live_after_w(_bline(t, xn), rel, 3.0, 8.0) is False


def test_is_pure():
    frames = _frames()
    before = frames.copy()
    pdet.detect_penalties(_penalty_action(), frames, provider="skillcorner", goal_map=_goal_map(frames))
    pd.testing.assert_frame_equal(frames, before)


def test_constants_override_moves_the_bar_for_the_run_twelve_sensitivity():
    # Run (12) evaluates each penalty constant at +/-50 %: halving d_gk (2.0 -> 1.0 m) rejects a keeper 2 m off.
    frames = _frames(keeper_x=103.0)
    gm = _goal_map(frames)
    assert len(pdet.detect_penalties(_penalty_action(), frames, provider="skillcorner", goal_map=gm)) == 1
    tight = pdet.detect_penalties(
        _penalty_action(), frames, provider="skillcorner", goal_map=gm, constants={"penalty_d_gk_m": 1.0}
    )
    assert len(tight) == 0


def _sc_penalty_bronze(pitch_length: float) -> pd.DataFrame:
    """A SkillCorner bronze penalty scene on a pitch of ``pitch_length`` (raw centre-origin metres): the ball rests
    on the true 11 m spot of the +x goal for 1.5 s, then is struck; A's taker beside it, B's keeper on the line."""
    half = pitch_length / 2.0
    spot = half - 11.0
    rows = []
    for f in range(1, 41):
        t = 0.1 * f
        bx = spot if f <= 15 else spot + 20.0 * (t - 1.5)
        for pid, team, gk, x, y in (
            ("a7", "A", False, spot - 0.5, 0.0),
            ("ak", "A", True, -half + 1.0, 0.0),
            ("bk", "B", True, half - 0.5, 0.0),
        ):
            rows.append(
                {
                    "match_id": "m1",
                    "period": 1,
                    "frame": f,
                    "timestamp": t,
                    "player_id": pid,
                    "team_id": team,
                    "is_goalkeeper": gk,
                    "x": x,
                    "y": y,
                    "ball_x": bx,
                    "ball_y": 0.0,
                    "ball_z": 0.1,
                    "is_visible": True,
                    "frame_rate": 10.0,
                    "pitch_length": pitch_length,
                    "pitch_width": 68.0,
                }
            )
    return pd.DataFrame(rows)


def test_pitch_length_reaches_the_detector_from_converter_output():
    # C2-IMPL-07: on CONVERTER output the original pitch length rides the frames (the converter stamp) and is a
    # declared keyword. A tight r (0.3 m) makes the spot position decisive: the 101 m pitch's scaled spot
    # (93.56 m) detects; forcing 105 m through the keyword (spot 94.0) misses by 0.44 m.
    from silly_kicks.tracking.skillcorner import convert_to_frames

    frames, _ = convert_to_frames(_sc_penalty_bronze(101.0), home_team_id="A", output_convention="absolute_frame")
    assert frames.attrs["pitch_length"] == 101.0
    gm = _goal_map(frames)
    empty = pd.DataFrame(
        columns=["game_id", "period_id", "time_seconds", "team_id", "player_id", "type_id", "result_id"]
    )
    tight = {"penalty_spot_r_m": 0.3}
    hit = pdet.detect_penalties(empty, frames, provider="skillcorner", goal_map=gm, constants=tight)
    assert len(hit) == 1
    explicit = pdet.detect_penalties(
        empty, frames.copy(), provider="skillcorner", goal_map=gm, constants=tight, pitch_length=101.0
    )
    assert len(explicit) == 1  # the declared keyword alone (a reshaped frame may have lost attrs)
    wrong = pdet.detect_penalties(
        empty, frames, provider="skillcorner", goal_map=gm, constants=tight, pitch_length=105.0
    )
    assert len(wrong) == 0
