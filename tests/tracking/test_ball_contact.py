"""Tests for the TF-79 ball-contact detection primitive."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _ball_contact as bc


# --------------------------------------------------------------------------- #
# Task 1: RDP kernel + params
# --------------------------------------------------------------------------- #
def test_rdp_keeps_true_vertices_drops_collinear():
    # An L-shaped path: straight run, one corner, straight run.
    xy = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], dtype=float)
    keep = bc._rdp(xy, epsilon_m=0.3)
    assert keep[0] and keep[-1]
    assert keep[2]  # the corner
    assert not keep[1] and not keep[3]  # collinear mids


def test_rdp_epsilon_monotone():
    xy = np.array([[0, 0], [1, 0.2], [2, 0], [3, 0.2], [4, 0]], dtype=float)
    coarse = bc._rdp(xy, epsilon_m=1.0).sum()
    fine = bc._rdp(xy, epsilon_m=0.05).sum()
    assert fine >= coarse


def test_ball_contact_params_rejects_bad_radius():
    with pytest.raises(ValueError):
        bc.BallContactParams(contact_radius_m=6.0, contact_exist_radius_m=5.0)


# --------------------------------------------------------------------------- #
# Task 2: velocity-break kernel
# --------------------------------------------------------------------------- #
def _line(n, dt, vx, vy, x0=0.0, y0=0.0):
    t = np.arange(n) * dt
    x = x0 + vx * t
    y = y0 + vy * t
    return np.column_stack([x, y]), t


def _join(xy1, t1, xy2, dt):
    t2 = t1[-1] + dt + np.arange(len(xy2)) * dt
    return np.vstack([xy1, xy2]), np.concatenate([t1, t2])


def test_velocity_break_fires_on_speed_drop():
    xy1, t1 = _line(6, 0.1, 10.0, 0.0)
    xy2, _ = _line(6, 0.1, 1.0, 0.0, x0=xy1[-1, 0])
    xy, t = _join(xy1, t1, xy2, 0.1)
    mask, _mag = bc._velocity_breaks(xy, t, bc.BallContactParams())
    assert mask.any()


def test_velocity_break_fires_on_heading_turn():
    xy1, t1 = _line(6, 0.1, 5.0, 0.0)
    xy2, _ = _line(6, 0.1, 0.0, 5.0, x0=xy1[-1, 0], y0=xy1[-1, 1])
    xy, t = _join(xy1, t1, xy2, 0.1)
    mask, _mag = bc._velocity_breaks(xy, t, bc.BallContactParams())
    assert mask.any()


def test_velocity_break_silent_on_smooth_line():
    xy, t = _line(20, 0.1, 6.0, 0.0)
    mask, _mag = bc._velocity_breaks(xy, t, bc.BallContactParams())
    assert not mask.any()


def test_velocity_break_frame_rate_invariant():
    for dt in (0.1, 0.04):
        xy1, t1 = _line(8, dt, 8.0, 0.0)
        xy2, _ = _line(8, dt, 0.5, 0.0, x0=xy1[-1, 0])
        xy, t = _join(xy1, t1, xy2, dt)
        mask, _m = bc._velocity_breaks(xy, t, bc.BallContactParams())
        assert mask.any()


# --------------------------------------------------------------------------- #
# Task 3: attribution + merge/representative
# --------------------------------------------------------------------------- #
def test_attribution_rejects_overhead_ball():
    players = pd.DataFrame({"player_id": [7], "x": [10.0], "y": [10.0]})
    pid, _dist = bc._attribute(0, np.array([10.0]), np.array([10.0]), np.array([4.0]), players, bc.BallContactParams())
    assert pd.isna(pid)


def test_attribution_nearest_playable():
    players = pd.DataFrame({"player_id": [7, 9], "x": [10.5, 20.0], "y": [10.0, 10.0]})
    pid, dist = bc._attribute(0, np.array([10.0]), np.array([10.0]), np.array([0.1]), players, bc.BallContactParams())
    assert pid == 7 and dist < 1.0


def test_attribution_no_player_in_range_is_unresolved():
    players = pd.DataFrame({"player_id": [7], "x": [50.0], "y": [50.0]})
    pid, _dist = bc._attribute(0, np.array([10.0]), np.array([10.0]), np.array([0.1]), players, bc.BallContactParams())
    assert pd.isna(pid)


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


def test_merge_rdp_only_cluster_keeps_earliest_vertex():
    cands = [3, 4]
    mag = {3: 0.0, 4: 0.0}
    src = {3: "rdp", 4: "rdp"}
    t = {3: 2.00, 4: 2.02}
    out = bc._merge_clusters(cands, mag, src, t, bc.BallContactParams())
    assert len(out) == 1 and out[0].frame_pos == 3 and out[0].source == "rdp"


# --------------------------------------------------------------------------- #
# Task 4: detect_ball_contacts (synthetic fixture)
# --------------------------------------------------------------------------- #
_P_A, _P_B = 7, 9
_AX, _AY = 60.0, 34.0
_BX, _BY = 62.0, 48.0


def _synthetic_match_frames(*, gap_period2: bool = True) -> pd.DataFrame:
    """Two periods, 25 fps, ball + two players. Planted contacts:
    - contact A at frame 10: a sharp heading turn at player A (ground, z=0.1) -> source "both";
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
                x, y, z = _BX, _BY, 2.4
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
        "contact_source",
    }
    resolved = out[out["contact_source"] != "unresolved"]
    assert resolved["player_id"].notna().all()
    assert out["contact_source"].nunique() >= 2  # non-vacuity: >1 source
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


def test_min_velocity_change_floor_is_opt_in():
    frames = _synthetic_match_frames()
    keep_all = bc.detect_ball_contacts(frames)  # default floor 0.0 -> no loss
    floored = bc.detect_ball_contacts(frames, params=bc.BallContactParams(min_velocity_change_ms=100.0))
    # a 100 m/s floor is physically unreachable -> drops candidates; default keeps strictly more
    assert len(keep_all) >= len(floored)
    assert (floored["contact_source"] != "unresolved").sum() <= (keep_all["contact_source"] != "unresolved").sum()


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
