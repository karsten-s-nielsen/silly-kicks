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


def test_path_enters_box_is_attempt():
    att, _ = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(INTO_BOX))
    assert att == [True]


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


def test_blocked_cross_heading_into_box():
    # Released towards the box; the defender (team B) stands 4.3 m along the path and deflects it back out.
    d = (82.0, 8.8)
    blocked_in = _path((1.0, 1.25, (80.0, 5.0), d), (1.25, 1.75, d, (70.0, 2.0)))
    players = {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", d[0], d[1])}
    att, _ = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(blocked_in, players))
    assert att == [True]
    # Pointing away from the box (towards the own half), the same block is not an attempt.
    d2 = (77.0, 5.5)
    blocked_away = _path((1.0, 1.25, (80.0, 5.0), d2), (1.25, 1.75, d2, (90.0, 2.0)))
    players2 = {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", d2[0], d2[1])}
    att2, _ = _attempt(_actions([("pass", 1.0, 80.0, 5.0)]), _frames(blocked_away, players2))
    assert att2 == [False]


def test_block_radius_both_sides():
    d = (82.0, 8.8)  # 4.3 m from the release
    blocked_in = _path((1.0, 1.25, (80.0, 5.0), d), (1.25, 1.75, d, (70.0, 2.0)))
    players = {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", d[0], d[1])}
    dist = float(np.hypot(d[0] - 80.0, d[1] - 5.0))
    a, f = _actions([("pass", 1.0, 80.0, 5.0)]), _frames(blocked_in, players)
    assert _attempt(a, f, block_radius_m=dist + 0.5)[0] == [True]
    assert _attempt(a, f, block_radius_m=dist - 0.5)[0] == [False]  # decided by the path only


def test_block_radius_measured_from_release():
    # Spec 6.3: the block radius is measured from the RELEASE (the ball), not the event's start. The event start sits
    # 2.5 m from the ball (inside release_match_m), and the defender 4.3 m from the ball but 2.4 m from the start.
    d = (82.0, 8.8)
    blocked_in = _path((1.0, 1.25, (80.0, 5.0), d), (1.25, 1.75, d, (70.0, 2.0)))
    players = {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", d[0], d[1])}
    a, f = _actions([("pass", 1.0, 80.0, 7.5)]), _frames(blocked_in, players)
    assert _attempt(a, f, block_radius_m=3.5)[0] == [False]
    assert _attempt(a, f, block_radius_m=4.8)[0] == [True]


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


@pytest.mark.parametrize(
    ("deflect", "after", "expected"),
    [((82.0, 8.8), (70.0, 2.0), True), ((77.0, 5.5), (90.0, 2.0), False)],
    ids=["heading_in", "heading_away"],
)
def test_orientation_invariance_on_the_block_branch(deflect, after, expected):
    # The block test reads the initial ball DIRECTION, so a right-to-left attacking team must flip it too (ADR-028).
    path = _path((1.0, 1.25, (80.0, 5.0), deflect), (1.25, 1.75, deflect, after))
    players = {1: ("A", 80.0, 5.0), 2: ("A", 96.0, 34.0), 3: ("B", *deflect)}
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    assert _attempt(a, _frames(path, players))[0] == [expected]
    assert _attempt(a, _frames(path, players, mirror=True))[0] == [expected]


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


def test_pure():
    a = _actions([("pass", 1.0, 80.0, 5.0)])
    f = _frames(INTO_BOX)
    a0, f0 = a.copy(deep=True), f.copy(deep=True)
    add_cross_deliveries(a, f)
    pd.testing.assert_frame_equal(a, a0)
    pd.testing.assert_frame_equal(f, f0)


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
