"""TF-86 ``add_cross_deliveries``: the tracking cross attempt and its height (spec §6.3, ADR-133).

Synthetic scenes: team A (players 1 passer, 2 teammate) attacks left-to-right, team B (player 3) right-to-left,
25 Hz. The ball path is scripted per test. Most scenes carry empty precomputed ``elastic_*`` columns, so the
release is the action time (step 2 of §6.3) and ELASTIC is not run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._ball_contact import BallContactParams
from silly_kicks.tracking._cross_delivery import CROSS_HEIGHT_VALUES, CrossDeliveryParams, add_cross_deliveries

_ID = spadlconfig.actiontype_id
FPS = 25
P = CrossDeliveryParams()
AERIAL = BallContactParams().aerial_z_m


def _path(*legs, z=0.3, fps=FPS):
    """legs: (t0, t1, (x0, y0), (x1, y1)); a linear ball path per leg, sampled at ``fps``; held after the last."""
    pts = []
    for t0, t1, (x0, y0), (x1, y1) in legs:
        for t in np.arange(t0, t1, 1 / fps):
            f = (t - t0) / (t1 - t0)
            pts.append((round(t, 4), x0 + f * (x1 - x0), y0 + f * (y1 - y0), z))
    last = legs[-1]
    for t in np.arange(last[1], last[1] + 3.0, 1 / fps):
        pts.append((round(t, 4), last[3][0], last[3][1], z))
    return pts


def _frames(ball, players=None, *, period=1, mirror=False, z_source="measured", ltr_team: object = "A"):
    """ball: [(t, x, y, z)], players: {pid: (team, x, y)} constant positions; ``ltr_team`` attacks left to right."""
    players = players or {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", 60.0, 40.0)}
    rows = []
    for t, x, y, z in ball:
        fid = round(t * FPS)
        bx, by = (105 - x, 68 - y) if mirror else (x, y)
        rows.append(
            dict(
                period_id=period,
                frame_id=fid,
                time_seconds=t,
                frame_rate=float(FPS),
                player_id=pd.NA,
                team_id=pd.NA,
                is_ball=True,
                x=bx,
                y=by,
                z=z,
                z_source=z_source,
                team_attacking_direction=pd.NA,
            )
        )
        for pid, (team, px, py) in players.items():
            qx, qy = (105 - px, 68 - py) if mirror else (px, py)
            ltr = (team == ltr_team) != mirror
            rows.append(
                dict(
                    period_id=period,
                    frame_id=fid,
                    time_seconds=t,
                    frame_rate=float(FPS),
                    player_id=pid,
                    team_id=team,
                    is_ball=False,
                    x=qx,
                    y=qy,
                    z=np.nan,
                    z_source="absent",
                    team_attacking_direction="ltr" if ltr else "rtl",
                )
            )
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["game_id"] = 1
    return f


def _actions(rows):
    """rows: (type_name, t, x, y[, team, player])."""
    recs = []
    for i, r in enumerate(rows):
        name, t, x, y = r[:4]
        team = r[4] if len(r) > 4 else "A"
        player = r[5] if len(r) > 5 else 1
        recs.append(
            dict(
                game_id=1,
                period_id=1,
                action_id=i,
                time_seconds=float(t),
                team_id=team,
                player_id=player,
                type_id=_ID[name],
                start_x=x,
                start_y=y,
                end_x=x,
                end_y=y,
            )
        )
    a = pd.DataFrame(recs)
    a["player_id"] = a["player_id"].astype("Int64")
    a["elastic_frame_id"] = np.nan
    a["elastic_confidence"] = np.nan
    return a


def _attempt(actions, frames, **kw):
    out = add_cross_deliveries(actions, frames, params=CrossDeliveryParams(**kw) if kw else P)
    return out["cross_attempt"].tolist(), out["cross_height"].tolist()


INTO_BOX = _path((1.0, 2.0, (80.0, 5.0), (95.0, 34.0)))
STAYS_WIDE = _path((1.0, 2.0, (80.0, 5.0), (100.0, 3.0)))
#: (deflection point, where the ball goes next): an opponent block 4.3 m from the release, heading into the box,
#: and one 3.0 m away, heading towards the own half.
BLOCK_IN = ((82.0, 8.8), (70.0, 2.0))
BLOCK_AWAY = ((77.0, 5.5), (90.0, 2.0))


#: A far opponent contact (14.4 m from the release, still outside the box) on a ball heading into the box.
FAR_IN = ((88.0, 17.0), (70.0, 2.0))


def _deflected(deflect, after, *, team="B", t_hit=1.25, **kw):
    """Released at (80, 5) at t=1.0; player 3 (of ``team``) stands at ``deflect`` and deflects the ball at ``t_hit``."""
    path = _path((1.0, t_hit, (80.0, 5.0), deflect), (t_hit, t_hit + 0.5, deflect, after))
    return _frames(path, {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: (team, *deflect)}, **kw)


def test_path_stays_wide_is_not():
    att, _ = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(STAYS_WIDE))
    assert att == [False]


def test_row_class_table():
    rows = [
        ("pass", 1.0, 80.0, 5.0),  # cross zone, answerable -> True
        ("pass", 1.0, 40.0, 5.0),  # outside the zone -> False
        ("corner_short", 1.0, 80.0, 5.0),  # set piece, answerable -> True (no zone condition)
        ("shot", 1.0, 95.0, 34.0),
        ("shot_freekick", 1.0, 80.0, 30.0),
        ("shot_penalty", 1.0, 94.0, 34.0),
        ("throw_in", 1.0, 80.0, 0.0),
        ("goalkick", 1.0, 5.0, 34.0),
        ("dribble", 1.0, 80.0, 5.0),
    ]
    att, height = _attempt(_actions(rows), _frames(INTO_BOX))
    assert att[:3] == [True, False, True]
    assert all(pd.isna(v) for v in att[3:])
    assert all(pd.isna(h) for h in height[1:2] + height[3:])


def test_unanswerable_cross_zone_row_is_unknown():
    a = _actions([("pass", 30.0, 80.0, 5.0)])  # no ball frame at t=30
    att, height = _attempt(a, _frames(INTO_BOX))
    assert pd.isna(att[0]) and pd.isna(height[0])


def test_cut_out_is_unknown_at_any_distance():
    # Cross spec §6.3 case 2: no box entry, an opponent's first contact inside cut_out_window_s, the initial direction
    # into the box -> <NA> (cut out), whatever the contact's distance from the release (block_radius_m is gone).
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    a = _actions([("pass", 1.0, 80.0, 5.0)])
    for scene, t_hit in ((BLOCK_IN, 1.25), (FAR_IN, 1.6)):
        f = _deflected(*scene, t_hit=t_hit)
        d = _cross_delivery_diagnostics(a, f)
        assert bool(d["first_contact_is_opponent"].iloc[0])  # precondition: the opponent's contact is detected
        assert pd.isna(_attempt(a, f)[0][0])
        assert bool(d["cut_out"].iloc[0])
    with pytest.raises(TypeError):
        CrossDeliveryParams(block_radius_m=5.0)  # type: ignore[call-arg]


def test_cut_out_window_both_sides():
    # Review Focus 9: a contact exactly at cut_out_window_s is inside it (<=); just beyond it the row is False.
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    a, f = _actions([("pass", 1.0, 80.0, 5.0)]), _deflected(*BLOCK_IN)
    dt = float(_cross_delivery_diagnostics(a, f)["first_contact_dt"].iloc[0])
    assert 0.0 < dt < 1.0  # precondition
    assert pd.isna(_attempt(a, f, cut_out_window_s=dt)[0][0])
    assert _attempt(a, f, cut_out_window_s=dt - 1e-6)[0] == [False]


def test_cut_out_needs_the_box_direction():
    # The same opponent contact on a ball heading away from the box is not aimed at it: False, not cut out.
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    a = _actions([("pass", 1.0, 80.0, 5.0)])
    for mirror in (False, True):
        f = _deflected(*BLOCK_AWAY, mirror=mirror)
        assert _attempt(a, f)[0] == [False]
        assert not bool(_cross_delivery_diagnostics(a, f)["cut_out"].iloc[0])


def test_teammate_first_contact_is_not_a_cut_out():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    assert _attempt(a, _deflected(*BLOCK_IN, team="A"))[0] == [False]


def test_box_entry_wins_over_a_later_opponent_contact():
    # The ball enters the box before the opponent touches it: an attempt, whatever happens next.
    into = _path((1.0, 1.6, (80.0, 5.0), (92.0, 30.0)), (1.6, 2.1, (92.0, 30.0), (80.0, 50.0)))
    f = _frames(into, {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 50.0), 3: ("B", 92.0, 30.0)})
    assert _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), f)[0] == [True]


def test_cut_out_has_no_height():
    ball_z = [(t, x, y, 2.5) for t, x, y, _z in _path((1.0, 1.25, (80.0, 5.0), BLOCK_IN[0]), (1.25, 1.75, *BLOCK_IN))]
    f = _frames(ball_z, {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", *BLOCK_IN[0])})
    att, height = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), f)
    assert pd.isna(att[0]) and pd.isna(height[0])


def test_cut_out_orientation_invariance():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    straight = add_cross_deliveries(a, _deflected(*BLOCK_IN))
    mirrored = add_cross_deliveries(a, _deflected(*BLOCK_IN, mirror=True))
    pd.testing.assert_frame_equal(straight, mirrored)
    assert pd.isna(straight["cross_attempt"].iloc[0])


def test_cut_out_corner_falls_back_to_event():
    # Review Focus 9: a cut-out corner is <NA>; TF-46 then takes the SPADL type (corner_kind_source "event").
    from silly_kicks.corners import identify_corners

    a = _actions([("corner_crossed", 1.0, 80.0, 5.0)])
    out = add_cross_deliveries(a, _deflected(*BLOCK_IN))
    assert pd.isna(out["cross_attempt"].iloc[0])
    corners = identify_corners(out, provider="gradientsports")
    assert corners["corner_kind"].tolist() == ["direct"]
    assert corners["corner_kind_source"].tolist() == ["event"]


def test_cut_out_cross_makes_the_xcross_label_unknown():
    # Cross spec §6.5: the enricher's cut-out row is an unknown label (NaN), never 0 and never 1.
    from silly_kicks.tracking import _xcross_attempt as xc

    out = add_cross_deliveries(_actions([("pass", 1.0, 80.0, 5.0)]), _deflected(*BLOCK_IN))
    assert pd.isna(out["cross_attempt"].iloc[0])
    fidx = pd.DataFrame(
        {"game_id": [1], "period_id": [1], "frame_id": [1], "time_seconds": [0.5], "team_in_possession": ["A"]}
    )
    y = np.asarray(xc.build_xcross_labels(fidx, out, horizon_seconds=1.0), dtype=float)
    assert np.isnan(y).all()


def test_unresolved_direction_is_unknown():
    # Review B CD-SB-28 (owner ruling): an initial direction that cannot be resolved (fewer than 2 finite ball frames
    # in the direction window) is unknown. BLOCK_AWAY resolves to "away" (False); with one finite frame it is <NA>.
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    a = _actions([("pass", 1.0, 80.0, 5.0)])
    base = _path((1.0, 1.25, (80.0, 5.0), BLOCK_AWAY[0]), (1.25, 1.75, *BLOCK_AWAY))
    players = {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", *BLOCK_AWAY[0])}

    def blank(lo):
        return [(t, np.nan, np.nan, z) if lo <= t < 1.2 else (t, x, y, z) for t, x, y, z in base]

    # A 0.15 s direction window over [1.0, 1.15]: 1 / 2 finite frames (the 1.2 s frame, kept for the contact, is out).
    one, two = _frames(blank(1.01), players), _frames(blank(1.05), players)
    params = CrossDeliveryParams(min_ball_coverage=0.3, direction_window_s=0.15)
    for f in (one, two):
        assert bool(_cross_delivery_diagnostics(a, f, params)["first_contact_is_opponent"].iloc[0])
    assert pd.isna(_attempt(a, one, min_ball_coverage=0.3, direction_window_s=0.15)[0][0])
    assert _attempt(a, two, min_ball_coverage=0.3, direction_window_s=0.15)[0] == [False]


def test_cut_window_excludes_the_contact_frame_at_10hz():
    # Review A CD-IMPL-15 (cross spec 6.3, C39): at 10 Hz an opponent's contact on the frame at 1.1 s lies inside
    # the 0.2 s direction window; the cut window holds the frames STRICTLY before it -- only the release frame -- so
    # the direction is unresolved and the row is <NA>. Counting the contact's own frame would resolve it.
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    hit = (81.0, 6.5)
    ball = _path((1.0, 1.1, (80.0, 5.0), hit), (1.1, 1.6, hit, (70.0, 2.0)), fps=10)
    f = _frames(ball, {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", *hit)})
    f["frame_rate"] = 10.0
    f["frame_id"] = (f["time_seconds"] * 10).round().astype(int)
    params = CrossDeliveryParams(release_gap_s=0.05)
    d = _cross_delivery_diagnostics(_actions([("pass", 1.0, 80.0, 5.0)]), f, params)
    assert bool(d["first_contact_is_opponent"].iloc[0])  # precondition: the opponent's contact is the first one
    assert d["first_contact_dt"].iloc[0] == pytest.approx(0.1, abs=1e-6)
    assert pd.isna(d["attempt"].iloc[0])
    assert bool(d["direction_unresolved"].iloc[0]) and bool(d["cut_out"].iloc[0])


def test_direction_window_stops_at_first_contact():
    # Review A C37: the opponent touches the ball 0.1 s after the release and it rebounds away from the box. The
    # direction window (0.2 s) is cut at the contact, so the direction is the cross's own (into the box): <NA>.
    from silly_kicks.tracking._cross_delivery import _ray_hits_box

    hit = (81.2, 6.8)
    path = _path((1.0, 1.1, (80.0, 5.0), hit), (1.1, 1.5, hit, (66.0, 0.5)))
    f = _frames(path, {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", *hit)})
    # Non-vacuity: over the uncut window the ball's net movement points away from the box.
    w = [(x, y) for t, x, y, _z in path if 1.0 <= t <= 1.2 + 1e-9]
    assert not _ray_hits_box(80.0, 5.0, w[-1][0] - w[0][0], w[-1][1] - w[0][1])
    assert pd.isna(_attempt(_actions([("pass", 1.0, 80.0, 5.0)]), f, release_gap_s=0.04)[0][0])


def test_passer_excluded():
    # The event time is 0.3 s before the kick. The ball rolls to the passer along +x and the kick turns it ~63 deg
    # towards the box, so the detector resolves the passer's own touch AFTER release + release_gap_s. Excluding
    # the passer, the path runs on into the box; without the exclusion it would end at the passer's feet.
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    pre = _path((0.7, 1.0, (78.5, 5.0), (80.0, 5.0)), (1.0, 2.0, (80.0, 5.0), (95.0, 34.0)))
    a, f = _actions([("pass", 0.7, 80.0, 5.0)]), _frames(pre)
    d = _cross_delivery_diagnostics(a, f)
    assert d["passer_contact_dt"].iloc[0] > P.release_gap_s  # precondition: the passer's touch is detected
    assert _attempt(a, f)[0] == [True]


def test_path_capped_without_contact():
    # The ball keeps rolling wide past the cap; no contact; the path ends at the cap and never enters the box.
    long_wide = _path((1.0, 9.0, (80.0, 5.0), (104.0, 4.0)))
    att, _ = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(long_wide))
    assert att == [False]


def test_restart_ended_delivery_typed_from_path():
    rows = [
        ("pass", 1.0, 80.0, 5.0),
        ("goalkick", 4.0, 5.0, 34.0, "B", 3),
        ("corner_short", 1.0, 80.0, 5.0),
        ("throw_in", 4.0, 80.0, 0.0, "B", 3),
    ]
    att, _ = _attempt(_actions(rows), _frames(INTO_BOX))
    assert att[0] is True and att[2] is True


def _with_nan_ball(ball, frac):
    out = list(ball)
    win = [i for i, (t, *_rest) in enumerate(out) if 1.0 <= t <= 1.0 + P.path_cap_s]
    for i in win[: round(frac * len(win))]:
        t, _x, _y, z = out[i]
        out[i] = (t, np.nan, np.nan, z)
    return out


def test_coverage_both_sides():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    # Keep the release frame finite so the release match is defined; remove ball samples later in the path.
    ball = INTO_BOX
    below = [
        (
            t,
            (np.nan if 1.2 <= t < 1.2 + 0.3 * P.path_cap_s else x),
            y if not (1.2 <= t < 1.2 + 0.3 * P.path_cap_s) else np.nan,
            z,
        )
        for t, x, y, z in ball
    ]
    att_below, _ = _attempt(a, _frames(below), min_ball_coverage=0.8)
    assert pd.isna(att_below[0])
    att_ok, _ = _attempt(a, _frames(below), min_ball_coverage=0.5)
    assert att_ok == [True]


def test_coverage_counts_missing_ball_rows():
    # Coverage is judged against the frames the ball's frame rate implies, so DROPPED ball rows count as missing,
    # not only NaN ones (the same window as test_coverage_both_sides, removed instead of blanked).
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    gap = [p for p in INTO_BOX if not 1.2 <= p[0] < 1.2 + 0.3 * P.path_cap_s]
    assert pd.isna(_attempt(a, _frames(gap), min_ball_coverage=0.8)[0][0])
    assert _attempt(a, _frames(gap), min_ball_coverage=0.5)[0] == [True]


def test_single_freeze_frame_is_unknown():
    # An SB360-style freeze frame: one ball sample at the release and no trajectory. The path cannot be judged,
    # so the attempt is unknown -- never False (honest NaN, ADR-055).
    one = [(1.0, 80.0, 5.0, 0.3)]
    att, height = _attempt(_actions([("pass", 1.0, 80.0, 5.0), ("corner_short", 1.0, 80.0, 5.0)]), _frames(one))
    assert all(pd.isna(v) for v in att) and all(pd.isna(h) for h in height)


def test_release_alignment(monkeypatch):
    import silly_kicks.tracking._cross_delivery as cd

    def _boom(*a, **k):
        raise AssertionError("precomputed elastic columns must be read, not recomputed")

    monkeypatch.setattr(cd, "align_events_to_frames", _boom)
    a = _actions([("pass", 0.4, 80.0, 5.0)])  # the event time is 0.6 s early; the ball is elsewhere then
    early = _path((0.0, 1.0, (60.0, 30.0), (80.0, 5.0)), (1.0, 2.0, (80.0, 5.0), (95.0, 34.0)))
    att_unaligned, _ = _attempt(a, _frames(early))
    assert pd.isna(att_unaligned[0])  # misaligned release -> unknown
    a["elastic_frame_id"] = 25.0  # the true kick (t = 1.0)
    a["elastic_confidence"] = 0.9
    att_aligned, _ = _attempt(a, _frames(early))
    assert att_aligned == [True]
    a["elastic_confidence"] = 0.4  # below ElasticSyncParams.min_confidence -> not used
    assert pd.isna(_attempt(a, _frames(early))[0][0])


def _early_frames():
    """The ball sits at (60, 30) until t=1.0, then is kicked from (80, 5) into the box."""
    return _frames(_path((0.0, 1.0, (60.0, 30.0), (80.0, 5.0)), (1.0, 2.0, (80.0, 5.0), (95.0, 34.0))))


def _diag(actions, frames):
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    return _cross_delivery_diagnostics(actions, frames, CrossDeliveryParams())


@pytest.mark.parametrize(
    ("time_source", "expected_t", "source"), [("tracking", 1.0, "native"), ("event", 0.4, "elastic")]
)
def test_tracking_row_uses_native_time(time_source, expected_t, source):
    # Cross spec 6.3 step 0: a "tracking" row's time_seconds IS the release; a confident ELASTIC frame elsewhere
    # (frame 10 = t 0.4) is not consulted. The same row as "event" uses ELASTIC (both sides).
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    a["elastic_frame_id"] = 10.0
    a["elastic_confidence"] = 0.9
    a["time_source"] = time_source
    d = _diag(a, _early_frames())
    assert d["release_source"].iloc[0] == source
    assert d["release_t"].iloc[0] == pytest.approx(expected_t)


def test_missing_time_source_means_event():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    a["elastic_frame_id"] = 10.0
    a["elastic_confidence"] = 0.9
    assert "time_source" not in a.columns
    assert _diag(a, _early_frames())["release_source"].iloc[0] == "elastic"


def test_release_match_applies_to_native_rows():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    a["time_source"] = "tracking"
    m = P.release_match_m
    off = _path((1.0, 2.0, (80.0, 5.0 + m + 0.01), (95.0, 34.0)))
    on = _path((1.0, 2.0, (80.0, 5.0 + m - 0.01), (95.0, 34.0)))
    assert pd.isna(_attempt(a, _frames(off))[0][0])
    assert _attempt(a, _frames(on))[0] == [True]


def test_release_order_is_per_row():
    a = _actions([("pass", 1.0, 80.0, 5.0), ("pass", 1.0, 80.0, 5.0)])
    a["elastic_frame_id"] = 10.0
    a["elastic_confidence"] = 0.9
    a["time_source"] = ["tracking", "event"]
    d = _diag(a, _early_frames())
    assert d["release_source"].tolist() == ["native", "elastic"]
    assert d["release_t"].tolist() == pytest.approx([1.0, 0.4])


def test_elastic_skipped_when_every_applicable_row_is_native(monkeypatch):
    # Review A C31: a SkillCorner match always holds synthetic dribbles ("event"), which are never applicable. Only the
    # applicable rows' release is read, so ELASTIC must not run when all of THOSE are native -- both sides.
    import silly_kicks.tracking._cross_delivery as cd

    calls = []
    real = cd.align_events_to_frames
    monkeypatch.setattr(cd, "align_events_to_frames", lambda *a, **k: calls.append(1) or real(*a, **k))
    a = _actions([("pass", 1.0, 80.0, 5.0), ("dribble", 1.5, 85.0, 10.0)]).drop(
        columns=["elastic_frame_id", "elastic_confidence"]
    )
    a["time_source"] = ["tracking", "event"]
    d = _diag(a, _early_frames())
    assert calls == []
    assert d["release_source"].iloc[0] == "native"
    a["time_source"] = ["event", "event"]  # an applicable non-native row needs ELASTIC
    _diag(a, _early_frames())
    assert calls == [1]


def test_misaligned_release_is_unknown():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    m = P.release_match_m
    off = _path((1.0, 2.0, (80.0, 5.0 + m + 0.01), (95.0, 34.0)))
    on = _path((1.0, 2.0, (80.0, 5.0 + m - 0.01), (95.0, 34.0)))
    assert pd.isna(_attempt(a, _frames(off))[0][0])
    assert _attempt(a, _frames(on))[0] == [True]


def test_requires_action_id_for_the_elastic_release():
    # Without action_id (and without precomputed elastic columns) ELASTIC cannot be aligned; refuse rather than
    # silently fall back to the event time for every row.
    a = _actions([("pass", 1.0, 80.0, 5.0)]).drop(columns=["action_id", "elastic_frame_id", "elastic_confidence"])
    with pytest.raises(ValueError, match="action_id"):
        add_cross_deliveries(a, _frames(INTO_BOX))


def test_refuses_more_than_one_match():
    # Ball frames, contacts and ELASTIC times are keyed by period: two matches would mix silently.
    a = pd.concat([_actions([("pass", 1.0, 80.0, 5.0)]), _actions([("pass", 1.0, 80.0, 5.0)]).assign(game_id=2)])
    with pytest.raises(ValueError, match="one match"):
        add_cross_deliveries(a, _frames(INTO_BOX))
    f = pd.concat([_frames(INTO_BOX), _frames(INTO_BOX).assign(game_id=2)])
    with pytest.raises(ValueError, match="one match"):
        add_cross_deliveries(_actions([("pass", 1.0, 80.0, 5.0)]), f)


def test_unknown_release_time_is_never_decided():
    # A NaN event time (and no ELASTIC frame) has no release: the row must not be matched against the last ball
    # frame of the period.
    end_in_box = [*_path((1.0, 2.0, (80.0, 5.0), (95.0, 34.0))), (5.5, 80.0, 5.0, 0.3)]  # ball back at the start
    att, height = _attempt(_actions([("pass", np.nan, 80.0, 5.0)]), _frames(end_in_box))
    assert pd.isna(att[0]) and pd.isna(height[0])


def test_unknown_start_is_never_decided():
    # A delivery whose action start is unknown cannot pass the release match, so the path is unanswerable (<NA>)
    # even when the ball enters the box; an open-play pass with an unknown start is not known to be outside the zone.
    rows = [
        ("corner_crossed", 1.0, np.nan, 0.0),
        ("freekick_short", 1.0, 80.0, np.nan),
        ("pass", 1.0, np.nan, 5.0),
        ("pass", 1.0, 80.0, np.nan),
    ]
    att, height = _attempt(_actions(rows), _frames(INTO_BOX))
    assert all(pd.isna(v) for v in att) and all(pd.isna(h) for h in height)
    # Control: the same corner with a known start is decided.
    assert _attempt(_actions([("corner_crossed", 1.0, 80.0, 5.0)]), _frames(INTO_BOX))[0] == [True]


def test_release_takes_the_nearest_ball_frame():
    # 10 Hz ball (SkillCorner-like). The event time 0.92 s sits 0.02 s after a frame and 0.08 s before the next, so
    # the release is the frame at 0.9 s (within half a frame period), not unknown.
    ten_hz = _path((0.9, 1.9, (80.0, 5.0), (95.0, 34.0)), fps=10)
    a = _actions([("pass", 0.92, 80.0, 5.0)])
    assert _attempt(a, _frames(ten_hz))[0] == [True]
    # Other side: with the frames at 0.9 s and 1.0 s missing, no ball frame lies within half a period of 0.92 s.
    gap = [p for p in ten_hz if p[0] not in (0.9, 1.0)]
    assert pd.isna(_attempt(a, _frames(gap))[0][0])


@pytest.mark.parametrize(("peak", "expected"), [(AERIAL + 0.01, "aerial"), (AERIAL - 0.01, "ground")])
def test_height_aerial_and_ground(peak, expected):
    ball = [(t, x, y, (peak if 1.3 <= t <= 1.5 else 0.3)) for t, x, y, _z in INTO_BOX]
    _att, height = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(ball))
    assert height == [expected]


def test_height_estimated_is_unknown():
    ball = [(t, x, y, (2.5 if 1.3 <= t <= 1.5 else 0.3)) for t, x, y, _z in INTO_BOX]
    att, height = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(ball, z_source="estimated"))
    assert att == [True] and pd.isna(height[0])


def test_orientation_invariance():
    a = _actions([("pass", 1.0, 80.0, 5.0), ("pass", 1.0, 80.0, 5.0)])
    straight = add_cross_deliveries(a, _frames(INTO_BOX))
    mirrored = add_cross_deliveries(a, _frames(INTO_BOX, mirror=True))
    pd.testing.assert_frame_equal(straight, mirrored)


def test_period_without_ball_frames_is_unknown():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    a["period_id"] = 2
    f = pd.concat([_frames(INTO_BOX), _frames(INTO_BOX, period=2)])
    f = f[~((f["period_id"] == 2) & f["is_ball"])]
    att, height = _attempt(a, f)
    assert pd.isna(att[0]) and pd.isna(height[0])
    zf = _frames([(t, x, y, np.nan) for t, x, y, _z in INTO_BOX])
    att2, height2 = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), zf)
    assert att2 == [True] and pd.isna(height2[0])


def test_dtypes_and_domain():
    out = add_cross_deliveries(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(INTO_BOX))
    assert str(out["cross_attempt"].dtype) == "boolean"
    assert isinstance(out["cross_height"].dtype, pd.CategoricalDtype)
    assert tuple(out["cross_height"].cat.categories) == CROSS_HEIGHT_VALUES


def test_diagnostics_expose_calibration_quantities():
    # The corpus calibration (spec §7.3) reads raw per-row quantities from the SAME code path as the enricher.
    from silly_kicks.tracking._cross_delivery import _cross_delivery_diagnostics

    a = _actions([("pass", 1.0, 80.0, 5.0), ("pass", 1.0, 40.0, 5.0)])
    d = _cross_delivery_diagnostics(a, _frames(INTO_BOX))
    assert list(d.index) == list(a.index)
    row = d.iloc[0]
    assert bool(row["applicable"]) and not bool(d.iloc[1]["applicable"])
    assert row["release_source"] == "event" and row["release_dist_m"] == pytest.approx(0.0, abs=1e-6)
    assert row["coverage"] == pytest.approx(1.0, abs=0.02)  # the fixture holds the ball to one frame short of the cap
    assert bool(row["attempt"]) is True
    assert pd.isna(d.iloc[1]["release_dist_m"])  # not applicable -> no measurement
    pd.testing.assert_series_equal(
        add_cross_deliveries(a, _frames(INTO_BOX))["cross_attempt"].reset_index(drop=True),
        d["attempt"].reset_index(drop=True),
        check_names=False,
    )
