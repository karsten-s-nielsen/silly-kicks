"""Tests for the TF-79 ball-contact detection primitive."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _ball_contact as bc

# --------------------------------------------------------------------------- #
# Detector v2 (detector spec sections 4-5): kinematic candidates, reach, params, actor skip.
# Retired v1 tests (each ledgered): the RDP kernel tests, the velocity-break kernel tests, the fallback attribution
# tests, the RDP-only merge test and the opt-in velocity floor test -- the code they pinned is gone.
# --------------------------------------------------------------------------- #
_V2 = dict(contact_radius_m=1.2, min_velocity_change_ms=3.0, kinematic_window_s=0.067, merge_window_s=0.12)


def _p(**kw) -> bc.BallContactParams:
    return bc.BallContactParams(**{**_V2, **kw})


def _scene(legs, players=None, *, fps=25.0, z: object = 0.1, z_source="measured"):
    """A ball on piecewise-linear legs ``(t0, t1, (x0, y0), (x1, y1))``, sampled at ``fps``; ``players`` maps a player
    id to ``(team, x, y)`` (static). ``z`` is a number or a callable of time. A ball at rest repeats its position
    exactly, as a provider that freezes the dead ball does (IDSSE)."""
    players = {7: ("T1", 60.0, 34.0)} if players is None else players
    t_end = legs[-1][1]
    rows = []
    for k in range(round(t_end * fps) + 1):
        t = k / fps
        for t0, t1, (x0, y0), (x1, y1) in legs:
            if t0 - 1e-9 <= t <= t1 + 1e-9:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                break
        zz = z(t) if callable(z) else z
        base = dict(game_id=1, period_id=1, frame_id=k, time_seconds=t, frame_rate=fps)
        rows.append(
            {
                **base,
                "player_id": pd.NA,
                "team_id": pd.NA,
                "is_ball": True,
                "x": x,
                "y": y,
                "z": zz,
                "z_source": z_source,
            }
        )
        for pid, (team, px, py) in players.items():
            rows.append(
                {
                    **base,
                    "player_id": pid,
                    "team_id": team,
                    "is_ball": False,
                    "x": px,
                    "y": py,
                    "z": 0.0,
                    "z_source": "absent",
                }
            )
    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype("Int64")
    return df


def _speed_change(d, *, knee=1.0, at=(60.0, 34.0)):
    """A ball along +x at 10 m/s that changes to 10 + d m/s at ``knee``, at the point ``at``."""
    x0 = at[0] - 10.0 * knee
    return [(0.0, knee, (x0, at[1]), at), (knee, knee + 1.0, at, (at[0] + 10.0 + d, at[1]))]


def _resolved(out):
    return out[out["contact_source"] == "velocity_break"]


def _near(out, t, tol=0.05):
    return out[(out["time_seconds"] - t).abs() <= tol]


def test_candidate_threshold_both_sides():
    above = bc.detect_ball_contacts(_scene(_speed_change(3.2)), params=_p())
    below = bc.detect_ball_contacts(_scene(_speed_change(2.8)), params=_p())
    assert len(_near(_resolved(above), 1.0)) == 1
    assert len(_near(below, 1.0, tol=0.2)) == 0


def test_constant_speed_has_no_candidate():
    out = bc.detect_ball_contacts(_scene([(0.0, 2.0, (40.0, 34.0), (60.0, 34.0))]), params=_p())
    assert len(out) == 0


def test_curved_flight_is_not_a_contact():
    # A ball at 20 m/s turning 1.5 degrees per frame: a curve in flight (the v1 RDP false positive), no touch.
    fps, v, turn = 25.0, 20.0, np.radians(1.5)
    pts, x, y, h = [], 40.0, 10.0, 0.0
    for k in range(51):
        pts.append((k / fps, x, y))
        x, y, h = x + v / fps * np.cos(h), y + v / fps * np.sin(h), h + turn
    legs = [(pts[i][0], pts[i + 1][0], pts[i][1:], pts[i + 1][1:]) for i in range(len(pts) - 1)]
    out = bc.detect_ball_contacts(_scene(legs, {7: ("T1", 50.0, 12.0)}), params=_p())
    assert len(out) == 0


def test_kick_from_rest_passes_the_speed_gate():
    kick = [(0.0, 1.0, (60.0, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (75.0, 34.0))]
    out = bc.detect_ball_contacts(_scene(kick), params=_p())
    assert len(_near(_resolved(out), 1.0)) == 1  # incoming 0 (an exactly static ball), outgoing 15 m/s
    slow = [(0.0, 1.0, (59.5, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (61.5, 34.0))]  # 0.5 -> 1.5 m/s
    out_slow = bc.detect_ball_contacts(_scene(slow), params=_p(min_velocity_change_ms=0.1))
    assert len(_near(out_slow, 1.0, tol=0.2)) == 0  # both speeds below min_candidate_speed_ms (2.0)


def test_kick_timed_within_one_frame():
    out = _resolved(bc.detect_ball_contacts(_scene(_speed_change(8.0, knee=1.2)), params=_p()))
    assert len(out) == 1 and abs(float(out["time_seconds"].iloc[0]) - 1.2) <= 1 / 25.0 + 1e-9


@pytest.mark.parametrize(("offset", "credited"), [(1.15, True), (1.25, False)], ids=["inside", "outside"])
def test_reach_radius_both_sides(offset, credited):
    out = bc.detect_ball_contacts(_scene(_speed_change(5.0), {7: ("T1", 60.0, 34.0 + offset)}), params=_p())
    hit = _near(out, 1.0)
    assert len(hit) == 1
    assert (hit["contact_source"].iloc[0] == "velocity_break") is credited


def test_ball_above_reach_is_unresolved():
    high = bc.detect_ball_contacts(_scene(_speed_change(5.0), z=3.0), params=_p())
    assert _near(high, 1.0)["contact_source"].tolist() == ["unresolved"]
    unknown_z = bc.detect_ball_contacts(_scene(_speed_change(5.0), z=np.nan), params=_p())
    assert _near(unknown_z, 1.0)["contact_source"].tolist() == ["velocity_break"]  # NaN z skips the height check


def test_no_fallback_beyond_reach():
    out = bc.detect_ball_contacts(_scene(_speed_change(5.0), {7: ("T1", 60.0, 37.0)}), params=_p())
    assert _near(out, 1.0)["contact_source"].tolist() == ["unresolved"]  # 3 m away: v1 credited it (5 m fallback)


def test_frame_without_players_is_unresolved():
    out = bc.detect_ball_contacts(_scene(_speed_change(5.0), {}), params=_p())
    assert _near(out, 1.0)["contact_source"].tolist() == ["unresolved"]


def test_ground_bounce_is_not_a_candidate():
    def z(t):  # falls to the ground at t = 1.0 and bounces back up; the horizontal velocity never changes
        return 2.0 * (t - 1.0) ** 2

    out = bc.detect_ball_contacts(_scene([(0.0, 2.0, (50.0, 34.0), (70.0, 34.0))], z=z), params=_p())
    assert len(out) == 0


def test_merge_window_both_sides():
    legs = [
        (0.0, 1.0, (50.0, 34.0), (60.0, 34.0)),
        (1.0, 1.2, (60.0, 34.0), (62.0, 34.0 + 2.0)),  # a turn at 1.0, another at 1.2
        (1.2, 2.2, (62.0, 36.0), (72.0, 36.0)),
    ]
    players = {7: ("T1", 60.0, 34.0), 9: ("T2", 62.0, 36.0)}
    wide = bc.detect_ball_contacts(_scene(legs, players), params=_p(merge_window_s=0.3))
    narrow = bc.detect_ball_contacts(_scene(legs, players), params=_p(merge_window_s=0.1))
    assert len(_near(wide, 1.1, tol=0.15)) == 1
    assert len(_near(narrow, 1.1, tol=0.15)) == 2


@pytest.mark.parametrize("fps", [29.97, 25.0, 10.0])
def test_frame_rate_floor(fps):
    # kinematic_window_s (0.067 s) is below the 25 and 10 Hz frame floors; the window floors at two frame intervals.
    # The ball rests exactly still before the kick: the rest run outlasts the window, so the dedup keeps it.
    kick = [(0.0, 1.0, (60.0, 34.0), (60.0, 34.0)), (1.0, 2.0, (60.0, 34.0), (75.0, 34.0))]
    out = bc.detect_ball_contacts(_scene(kick, fps=fps), params=_p())
    assert len(_near(_resolved(out), 1.0, tol=1.5 / fps)) == 1


@pytest.mark.parametrize(("fps", "off_m"), [(25.0, 0.1), (10.0, 0.2)])
def test_window_spans_two_frame_intervals(fps, off_m):
    # One sample off the line of a 10 m/s flight by off_m, across the flight: the change at that sample is
    # 2 * off_m / window. Over two frame intervals it is 2.5 / 2.0 m/s (below 3.0, no candidate); over one interval it
    # would be 5.0 / 4.0 m/s (a false candidate).
    legs = [(0.0, 2.0, (40.0, 34.0), (60.0, 34.0))]
    noisy = _scene(legs, fps=fps)
    k = round(1.0 * fps)
    noisy.loc[noisy["is_ball"] & (noisy["frame_id"] == k), "y"] += off_m
    assert len(bc.detect_ball_contacts(noisy, params=_p())) == 0
    sharper = bc.detect_ball_contacts(noisy, params=_p(min_velocity_change_ms=1.5))
    assert len(_near(sharper, 1.0, tol=0.01)) == 1  # precondition: the noisy sample is a peak of the change


def test_stale_repeat_in_flight_is_collapsed():
    # A 10 m/s flight whose provider repeats one position for a single frame (shorter than the window): a stale
    # repeat, collapsed by the dedup, so no candidate. Kept, it would read as a stop and a restart.
    legs = [(0.0, 2.0, (40.0, 34.0), (60.0, 34.0))]
    f = _scene(legs)
    ball = f["is_ball"]
    for col in ("x", "y"):
        f.loc[ball & (f["frame_id"] == 26), col] = f.loc[ball & (f["frame_id"] == 25), col].to_numpy()
    assert len(bc.detect_ball_contacts(f, params=_p())) == 0


def test_merge_floor_is_two_frame_intervals():
    # Two candidates 0.15 s apart: one cluster at 10 Hz (floor 0.2 s), two at 25 Hz (merge_window_s 0.12 s wins).
    mag, src, t = {0: 5.0, 1: 4.0}, {0: "velocity_break", 1: "velocity_break"}, {0: 1.0, 1: 1.15}
    assert len(bc._merge_clusters([0, 1], mag, src, t, _p(), frame_period=0.1)) == 1
    assert len(bc._merge_clusters([0, 1], mag, src, t, _p(), frame_period=0.04)) == 2


def test_change_threshold_is_inclusive():
    # min_velocity_change_ms is "at least": a peak exactly at the threshold is kept, one just below it is dropped.
    peaks = bc._candidate_peaks(_scene(_speed_change(5.0)), _p())
    dv = float(peaks["dv"].max())
    at = bc._finalize_contacts(peaks, _p(min_velocity_change_ms=dv), pid_dtype="Int64")
    above = bc._finalize_contacts(peaks, _p(min_velocity_change_ms=float(np.nextafter(dv, np.inf))), pid_dtype="Int64")
    assert len(at) == 1 and len(above) == 0


def test_velocity_change_is_the_admitting_change():
    out = _resolved(bc.detect_ball_contacts(_scene(_speed_change(3.2)), params=_p()))
    assert float(out["velocity_change_ms"].iloc[0]) == pytest.approx(3.2, abs=0.05)


def test_confidence_proximity_both_ends():
    at = _resolved(bc.detect_ball_contacts(_scene(_speed_change(5.0), {7: ("T1", 60.0, 34.0)}), params=_p()))
    edge = _resolved(bc.detect_ball_contacts(_scene(_speed_change(5.0), {7: ("T1", 60.0, 35.199)}), params=_p()))
    # proximity = 1 - dist / contact_radius_m, weight 0.20: 1.0 at the ball, about 0.0 at the edge of reach
    assert float(at["confidence"].iloc[0]) - float(edge["confidence"].iloc[0]) == pytest.approx(0.20, abs=1e-3)


@pytest.mark.parametrize("field", ["rdp_epsilon_m", "speed_drop_frac", "heading_turn_deg", "contact_exist_radius_m"])
def test_removed_params_raise_type_error(field):
    removed: dict[str, Any] = {field: 1.0}
    with pytest.raises(TypeError):
        bc.BallContactParams(**removed)


def test_positional_params_raise_type_error():
    with pytest.raises(TypeError):
        bc.BallContactParams(0.3)  # type: ignore[misc]  # a v1 positional call (rdp_epsilon_m first)


@pytest.mark.parametrize(
    ("field", "bad", "ok"),
    [
        ("kinematic_window_s", 0.0, 0.001),
        ("contact_radius_m", 0.0, 0.01),
        ("min_velocity_change_ms", -0.1, 0.0),
        ("merge_window_s", 0.0, 0.01),
    ],
)
def test_post_init_checks_both_sides(field, bad, ok):
    with pytest.raises(ValueError):
        bc.BallContactParams(**{field: bad})
    assert getattr(bc.BallContactParams(**{field: ok}), field) == ok


def test_v2_emits_only_velocity_break_and_unresolved():
    out = pd.concat(
        [
            bc.detect_ball_contacts(_synthetic_match_frames()),
            bc.detect_ball_contacts(_scene(_speed_change(5.0), {7: ("T1", 60.0, 37.0)}), params=_p()),
        ]
    )
    assert set(out["contact_source"].astype(object)) == {"velocity_break", "unresolved"}


def test_empty_and_single_frame_inputs():
    one = _scene([(0.0, 0.0, (60.0, 34.0), (60.0, 34.0))])
    assert len(bc.detect_ball_contacts(one, params=_p())) == 0
    no_ball = one[~one["is_ball"]]
    out = bc.detect_ball_contacts(no_ball, params=_p())
    assert len(out) == 0 and list(out.columns) == list(bc._CONTACT_COLUMNS)


def _contacts_actor():
    return pd.DataFrame(
        {
            "period_id": [1, 1, 1],
            "frame_id": [24, 26, 30],
            "time_seconds": [0.98, 1.02, 1.20],
            "player_id": pd.array([7, 7, 9], dtype="Int64"),
            "contact_x": [60.0, 60.2, 61.0],
            "contact_y": [34.0, 34.0, 34.0],
            "velocity_change_ms": [5.0, 5.0, 5.0],
            "confidence": [0.8, 0.8, 0.8],
            "is_aerial": pd.array([False, False, False], dtype="boolean"),
            "aerial_source": pd.Categorical(["measured"] * 3),
            "contact_source": pd.Categorical(["velocity_break"] * 3, categories=bc.CONTACT_SOURCE_VALUES),
        }
    )


@pytest.mark.parametrize("excluded", [7, "7", 7.0], ids=["int", "str", "float"])
def test_first_contact_in_window_skips_excluded_player(excluded):
    # Detector spec section 5 (owner ruling 6): the actor's own contact after t0 is skipped, dtype-safe.
    c = _contacts_actor()
    found = [
        bc.first_contact_in_window(c, 1, 1.0, 2.0),  # no exclusion: the actor at 1.02
        bc.first_contact_in_window(c, 1, 1.0, 2.0, exclude_player_id=excluded),
        bc.first_contact_in_window(c, 1, 0.9, 2.0, exclude_player_id=excluded),  # the actor at 0.98 too
    ]
    assert all(f is not None for f in found)
    assert [f["player_id"] for f in found if f is not None] == [7, 9, 9]


@pytest.mark.parametrize("dtype", ["Int64", "object-str"])
def test_add_ball_contacts_skips_the_actor(dtype):
    # The synthetic match: player A (7) touches at 0.4 s, player B (9) at 0.8 s. An action by A at 0.3 s gets B as its
    # first contact; an action by B gets A. The action's player_id dtype differs from the frames' (Int64).
    frames = _synthetic_match_frames()
    for actor, expected in ((_P_A, _P_B), (_P_B, _P_A)):
        pid = pd.array([actor], dtype="Int64") if dtype == "Int64" else [str(actor)]
        actions = pd.DataFrame(
            {"game_id": [1], "action_id": [0], "period_id": [1], "time_seconds": [0.3], "player_id": pid}
        )
        assert int(bc.add_ball_contacts(actions, frames)["first_contact_player_id"].iloc[0]) == expected


def test_merge_representative_is_max_magnitude_frame():
    cands = [1, 2]
    mag = {1: 0.5, 2: 1.3}
    src = {1: "velocity_break", 2: "velocity_break"}
    t = {1: 1.00, 2: 1.04}
    out = bc._merge_clusters(cands, mag, src, t, bc.BallContactParams())
    assert len(out) == 1 and out[0].frame_pos == 2


def test_merge_keeps_two_touches_beyond_window():
    cands = [1, 50]
    mag = {1: 1.0, 50: 1.0}
    src = {1: "velocity_break", 50: "velocity_break"}
    t = {1: 1.00, 50: 3.00}
    out = bc._merge_clusters(cands, mag, src, t, bc.BallContactParams())
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# Task 4: detect_ball_contacts (synthetic fixture)
# --------------------------------------------------------------------------- #
_P_A, _P_B = 7, 9
_AX, _AY = 60.0, 34.0
_BX, _BY = 62.0, 48.0


def _synthetic_match_frames(*, gap_period2: bool = True) -> pd.DataFrame:
    """Two periods, 25 fps, ball + two players. Planted contacts:
    - contact A at frame 10: a sharp heading turn at player A (ground, z=0.1) -> a v2 "velocity_break";
    - contact B at frame 20: ball dead-stops at player B (aerial, z=2.4)   -> source "velocity_break".
    Period 2 inserts a 3-frame NaN-ball detection gap (frames 12-14).
    """
    fps = 25.0
    dt = 1.0 / fps
    rows = []

    def add(period, fid, player_id, team_id, is_ball, x, y, z):
        rows.append(
            dict(
                game_id=1,
                period_id=period,
                frame_id=fid,
                time_seconds=fid * dt,
                frame_rate=fps,
                player_id=player_id,
                team_id=team_id,
                is_ball=is_ball,
                x=x,
                y=y,
                z=z,
                z_source="measured",  # ADR-125: optical-measured fixture -> boolean is_aerial
            )
        )

    for period in (1, 2):
        n = 30
        for fid in range(n):
            if fid <= 10:
                x = 50.0 + (_AX - 50.0) * (fid / 10.0)
                y = 34.0
                z = 0.1
            elif fid <= 20:
                f = (fid - 10) / 10.0
                x = _AX + (_BX - _AX) * f
                y = _AY + (_BY - _AY) * f
                z = 0.1 + (2.4 - 0.1) * f
            else:
                # ball decelerates to a slow CRAWL along its approach direction (realistic -- NOT
                # frozen to the segment end; a permanent terminal static run would be collapsed by
                # the duplicate-frame dedup). Same direction as the A->B leg -> a pure speed drop at
                # B (contact_source "velocity_break", no RDP corner), so source diversity is kept.
                _nrm = ((_BX - _AX) ** 2 + (_BY - _AY) ** 2) ** 0.5
                _ux, _uy = (_BX - _AX) / _nrm, (_BY - _AY) / _nrm
                x = _BX + 1.0 * (fid - 20) * dt * _ux
                y = _BY + 1.0 * (fid - 20) * dt * _uy
                z = 2.4
            if gap_period2 and period == 2 and 12 <= fid <= 14:
                x = np.nan
                y = np.nan
            add(period, fid, pd.NA, "T1", True, x, y, z)
            add(period, fid, _P_A, "T1", False, _AX, _AY, 0.0)
            add(period, fid, _P_B, "T2", False, _BX, _BY, 0.0)

    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("category")
    for c in ("x", "y", "z"):
        df[c] = df[c].astype("float32")
    return df


def test_detect_contacts_recovers_planted_touches():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    assert set(out.columns) == {
        "period_id",
        "frame_id",
        "time_seconds",
        "player_id",
        "contact_x",
        "contact_y",
        "velocity_change_ms",
        "confidence",
        "is_aerial",
        "aerial_source",
        "contact_source",
    }
    resolved = out[out["contact_source"] != "unresolved"]
    assert resolved["player_id"].notna().all()
    assert set(resolved["contact_source"].astype(object)) == {"velocity_break"}  # v2: one candidate method
    assert resolved["player_id"].nunique() >= 2  # non-vacuity: >1 player


def test_detect_contacts_is_aerial_boolean_and_marks_high_ball():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    assert out["is_aerial"].dtype.name == "boolean"
    assert bool(out["is_aerial"].fillna(False).any())  # contact B is aerial


def test_detect_contacts_no_contact_across_gap():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    p2 = out[out["period_id"] == 2]
    assert not ((p2["frame_id"] >= 12) & (p2["frame_id"] <= 14)).any()


def test_detect_contacts_confidence_bounded():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    c = out["confidence"].dropna()
    assert ((c >= 0) & (c <= 1)).all()


def test_detect_contacts_emits_velocity_change():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    resolved = out[out["contact_source"] != "unresolved"]
    assert "velocity_change_ms" in out.columns
    # the planted kick/stop impart a real velocity change (>> flight drag)
    assert (resolved["velocity_change_ms"].dropna() > 2.0).any()


def test_confidence_monotone_in_velocity_change():
    lo = bc._confidence(1.0, 1.0, 1.0, 1.0, dv_scale_ms=5.0)
    hi = bc._confidence(20.0, 1.0, 1.0, 1.0, dv_scale_ms=5.0)
    assert hi > lo and 0.0 <= lo <= hi <= 1.0


def test_detect_contacts_pure():
    frames = _synthetic_match_frames()
    before = frames.copy(deep=True)
    _ = bc.detect_ball_contacts(frames)
    pd.testing.assert_frame_equal(frames, before)


def test_detect_contacts_reflection_mirror_equal():
    frames = _synthetic_match_frames()
    base = bc.detect_ball_contacts(frames)
    mirrored = frames.copy()
    mirrored["x"] = (105.0 - mirrored["x"].astype("float64")).astype("float32")
    mout = bc.detect_ball_contacts(mirrored)
    np.testing.assert_allclose(
        sorted(mout["contact_x"].dropna().to_numpy()),
        sorted((105.0 - base["contact_x"].dropna()).to_numpy()),
        atol=1e-3,
    )


# --------------------------------------------------------------------------- #
# Task 5: add_ball_contacts + first_contact_in_window
# --------------------------------------------------------------------------- #
def _actions_and_frames():
    actions = pd.DataFrame({"game_id": [1], "action_id": [0], "period_id": [1], "time_seconds": [0.0]})
    return actions, _synthetic_match_frames(), _P_A  # expected first-contact player = A (frame 10)


def _actions_and_frames_no_contact():
    actions = pd.DataFrame({"game_id": [1], "action_id": [0], "period_id": [1], "time_seconds": [100.0]})
    return actions, _synthetic_match_frames()


def test_first_contact_in_window_picks_earliest():
    contacts = pd.DataFrame(
        {
            "period_id": [1, 1],
            "frame_id": [10, 20],
            "time_seconds": [1.0, 2.0],
            "player_id": pd.array([7, 9], dtype="Int64"),
            "contact_x": [30.0, 40.0],
            "contact_y": [20.0, 25.0],
            "confidence": [0.8, 0.9],
            "is_aerial": pd.array([False, True], dtype="boolean"),
            "contact_source": pd.Categorical(["both", "rdp"], categories=bc.CONTACT_SOURCE_VALUES),
        }
    )
    s = bc.first_contact_in_window(contacts, 1, 0.5, 1.5)
    assert s is not None and s["player_id"] == 7


def test_add_ball_contacts_attaches_first_contact_and_is_pure():
    actions, frames, expected_pid = _actions_and_frames()
    before = actions.copy(deep=True)
    out = bc.add_ball_contacts(actions, frames)
    assert {
        "first_contact_time",
        "first_contact_player_id",
        "first_contact_x",
        "first_contact_y",
        "first_contact_velocity_change_ms",
        "first_contact_confidence",
        "first_contact_is_aerial",
        "first_contact_source",
    } <= set(out.columns)
    assert out.loc[0, "first_contact_player_id"] == expected_pid
    pd.testing.assert_frame_equal(actions, before)  # purity


def test_add_ball_contacts_nan_when_no_contact():
    actions, frames = _actions_and_frames_no_contact()
    out = bc.add_ball_contacts(actions, frames)
    assert out["first_contact_player_id"].isna().all()


def test_add_ball_contacts_raises_without_time_base():
    actions = pd.DataFrame({"action_id": [0], "period_id": [1]})
    frames = _synthetic_match_frames()
    with pytest.raises(ValueError):
        bc.add_ball_contacts(actions, frames)


# --------------------------------------------------------------------------- #
# Task 6: reflection registration + contact_source categorical home
# --------------------------------------------------------------------------- #
def test_contact_source_vocab_closed():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    assert set(out["contact_source"].cat.categories) == set(bc.CONTACT_SOURCE_VALUES)


def test_contact_columns_registered_for_reflection():
    from silly_kicks.reflection import SPADL_REFLECTION_KINDS, TRACKING_REFLECTION_KINDS

    assert TRACKING_REFLECTION_KINDS["contact_x"] == "point_x"
    assert TRACKING_REFLECTION_KINDS["contact_y"] == "point_y"
    assert SPADL_REFLECTION_KINDS["first_contact_x"] == "point_x"
    assert SPADL_REFLECTION_KINDS["first_contact_y"] == "point_y"


# --------------------------------------------------------------------------- #
# Task 2 (TF-79 follow-up): time-based |dv| and detection-quality windows
# (dedup makes the per-period arrays sparse + non-uniform-dt, so fixed frame-count
#  windows span the wrong seconds; these must be derived from tt.)
# --------------------------------------------------------------------------- #
# dense cluster around the touch then a sparse tail (the post-dedup shape); 25 fps nominal
_TT_SPARSE = np.array([0.00, 0.04, 0.08, 0.12, 0.72, 1.32, 1.92])


def test_time_window_bounds_each_side_within_span():
    span = 0.12
    for gp in range(1, len(_TT_SPARSE) - 1):
        lo, hi = bc._time_window(_TT_SPARSE, gp, span)
        assert _TT_SPARSE[gp] - _TT_SPARSE[lo] <= span + 1e-9
        assert _TT_SPARSE[hi - 1] - _TT_SPARSE[gp] <= span + 1e-9


def test_old_frame_count_window_violates_the_seconds_bound():
    # RED-GREEN teeth (ADR-056): the RETIRED semantics were +/- round(span/frame_period) SAMPLES.
    # At 25 fps, span 0.12 s -> +/-3 samples. At gp=3 (t=0.12) that reaches sample 6 (t=1.92) ->
    # 1.80 s on the after side, >> 0.12 s. The new time window must NOT.
    span, frame_period = 0.12, 1.0 / 25.0
    k = max(1, round(span / frame_period))  # 3
    gp = 3
    i1_old = min(len(_TT_SPARSE) - 1, gp + k)
    assert _TT_SPARSE[i1_old] - _TT_SPARSE[gp] > span  # OLD span is wrong (1.80 s > 0.12 s)
    _lo, hi = bc._time_window(_TT_SPARSE, gp, span)
    assert _TT_SPARSE[hi - 1] - _TT_SPARSE[gp] <= span + 1e-9  # NEW is correct


def test_velocity_change_detects_turn_over_the_time_window():
    # +x 10 m/s before the touch, +y 10 m/s after: |dv| = hypot(10, 10) ~= 14.14 over +/-0.12 s
    tt = np.array([0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24])
    bx = np.where(tt <= 0.12, 10.0 * tt, 10.0 * 0.12)
    by = np.where(tt <= 0.12, 0.0, 10.0 * (tt - 0.12))
    dv, s_in, s_out, _turn = bc._velocity_change_series(bx, by, tt, 0.12)
    assert 13.0 < dv[3] < 15.0
    assert s_in[3] == pytest.approx(10.0) and s_out[3] == pytest.approx(10.0)
    assert np.isnan(dv[0]) and np.isnan(dv[-1])  # no sample on one side


def test_time_based_window_matches_old_on_dense_uniform_sampling():
    # regression: on uniform 25 fps the time window == the old +/-3-sample frame-count window
    tt = np.round(np.arange(0.0, 1.0, 0.04), 2)
    lo, hi = bc._time_window(tt, 10, 0.12)
    assert (tt[10] - tt[lo]) <= 0.12 + 1e-9 and (tt[hi - 1] - tt[10]) <= 0.12 + 1e-9
    assert hi - lo == 2 * 3 + 1


# --------------------------------------------------------------------------- #
# Task 3 (TF-79 follow-up): duplicate-frame dedup recovers GS-like under-detection
# --------------------------------------------------------------------------- #
_TRAP_PLAYER = 7  # the receiver of the trap touch at t ~= 0.6 s
_TRAP_T = 0.6


def _trap_frames(dup: bool) -> pd.DataFrame:
    """Ball 12 m/s +x then a sharp SPEED-DROP (trap) to a 1 m/s crawl at player 7 at t=0.6 s.

    The real path is sampled at 0.04 s. ``dup=False`` is the clean 25 fps sampling. ``dup=True``
    samples at 50 fps with every SECOND frame a held duplicate (the GS velocity sawtooth),
    TIME-ALIGNED so the trap still lands at t=0.6 s (dedup collapses the held dups back to the
    clean path).
    """
    dt_real = 0.04
    pts = []
    x, y = 20.0, 34.0
    for _ in range(15):  # approach 0.6 s at 12 m/s +x
        pts.append((x, y))
        x += 12.0 * dt_real
    tx, ty = x, y
    for _ in range(15):  # 1 m/s crawl after the trap
        pts.append((x, y))
        x += 1.0 * dt_real
    fps = 50.0 if dup else 25.0
    dt = 1.0 / fps
    rows = []

    def emit(fid, bx, by):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=fid * dt,
                frame_rate=fps,
                player_id=pd.NA,
                team_id="T1",
                is_ball=True,
                x=bx,
                y=by,
                z=0.1,
                z_source="measured",
            )
        )
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=fid * dt,
                frame_rate=fps,
                player_id=_TRAP_PLAYER,
                team_id="T1",
                is_ball=False,
                x=tx,
                y=ty,
                z=0.0,
                z_source="measured",
            )
        )
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=fid * dt,
                frame_rate=fps,
                player_id=9,
                team_id="T2",
                is_ball=False,
                x=5.0,
                y=5.0,
                z=0.0,
                z_source="measured",
            )
        )

    fid = 0
    for bx, by in pts:
        emit(fid, bx, by)
        fid += 1
        if dup:
            emit(fid, bx, by)  # held duplicate at +dt -> 50% consecutive-duplicate ball frames
            fid += 1
    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("category")
    for c in ("x", "y", "z"):
        df[c] = df[c].astype("float32")
    return df


def _strip_consecutive_dups(frames: pd.DataFrame) -> pd.DataFrame:
    ball = frames[frames["is_ball"]]
    players = frames[~frames["is_ball"]]
    parts = [players]
    for _pid, g in ball.groupby("period_id"):
        g = g.sort_values("frame_id")
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        m = np.ones(len(g), dtype=bool)
        m[1:] = np.hypot(np.diff(x), np.diff(y)) > 1e-6
        parts.append(g[m])
    return pd.concat(parts, ignore_index=True)


def _resolved_near(frames: pd.DataFrame, player: int, t: float, tol: float = 0.12) -> bool:
    c = bc.detect_ball_contacts(frames)
    r = c[(c["contact_source"].astype(object) != "unresolved") & (c["player_id"] == player)]
    return bool(((r["time_seconds"] - t).abs() <= tol).any())


def _resolved_keys(frames: pd.DataFrame) -> set:
    c = bc.detect_ball_contacts(frames)
    r = c[(c["contact_source"].astype(object) != "unresolved") & c["player_id"].notna()]
    return {
        (int(p), int(pl), round(float(tt), 2))
        for p, pl, tt in zip(r["period_id"], r["player_id"], r["time_seconds"], strict=True)
    }


def test_trap_touch_detected_without_dups():
    # floor (two-sided): the planted trap touch IS detectable on the clean sampling
    assert _resolved_near(_trap_frames(dup=False), _TRAP_PLAYER, _TRAP_T)


def test_trap_touch_recovered_through_duplicate_frame_sawtooth():
    # recovery: the same touch, buried under 50% consecutive-duplicate ball frames, is recovered
    assert _resolved_near(_trap_frames(dup=True), _TRAP_PLAYER, _TRAP_T)


def test_dedup_does_not_over_collapse_dense_control():
    # GSBC-PLAN-01: deduping must change ONLY the zero-displacement duplicates -> the resolved
    # contact set is IDENTICAL to its explicitly-stripped twin (an over-collapsing dedup fails).
    dense = _trap_frames(dup=True)
    assert _resolved_keys(dense) == _resolved_keys(_strip_consecutive_dups(dense))
    assert _resolved_near(dense, _TRAP_PLAYER, _TRAP_T)


# --------------------------------------------------------------------------- #
# Z2/Z3 (ADR-125): is_aerial honest-NaN unless z_source=="measured"; aerial_source echo
# --------------------------------------------------------------------------- #
def test_is_aerial_requires_measured_z_source():
    # measured -> boolean (synthetic contact B is aerial, z=2.4); estimated/absent -> all <NA>
    meas = bc.detect_ball_contacts(_synthetic_match_frames())  # tagged z_source="measured"
    assert bool(meas["is_aerial"].fillna(False).any())
    assert set(meas["aerial_source"].unique()) == {"measured"}
    for src in ("estimated", "absent"):
        out = bc.detect_ball_contacts(_synthetic_match_frames().assign(z_source=src))
        assert out["is_aerial"].isna().all()  # honest-NaN: z is not a trustworthy measurement
        assert set(out["aerial_source"].unique()) == {src}  # aerial_source echoes z_source


def test_is_aerial_fail_safe_absent_when_z_source_column_missing():
    # GSBC-PLAN-02: a legacy/synthetic caller with no z_source column -> fail-safe "absent" -> NaN,
    # never a false boolean.
    out = bc.detect_ball_contacts(_synthetic_match_frames().drop(columns=["z_source"]))
    assert out["is_aerial"].isna().all()
    assert set(out["aerial_source"].unique()) == {"absent"}
