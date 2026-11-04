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
def test_attribution_overhead_gate_is_conditional(bc_params=None):
    # TF-46 Arm 1 Part 2 (spec section 14.1): an overhead ball removes a contact ONLY when the z_overhead gate
    # is ACTIVE for the data profile AND z_source is "measured". Previously the gate was always-on; this is a
    # value change (its CHANGELOG entry lands at the C4 release commit, owner ruling 2026-10-22). _attribute
    # gained `zsrc` + the keyword-only `overhead_active`.
    players = pd.DataFrame({"player_id": [7], "x": [10.0], "y": [10.0]})
    bx, by, bz = np.array([10.0]), np.array([10.0]), np.array([4.0])  # z above the 2.6 ceiling
    p = bc.BallContactParams()
    assert pd.isna(bc._attribute(0, bx, by, bz, np.array(["measured"]), players, p, overhead_active=True)[0])
    assert bc._attribute(0, bx, by, bz, np.array(["measured"]), players, p, overhead_active=False)[0] == 7
    assert bc._attribute(0, bx, by, bz, np.array(["estimated"]), players, p, overhead_active=True)[0] == 7


def test_attribution_nearest_playable():
    players = pd.DataFrame({"player_id": [7, 9], "x": [10.5, 20.0], "y": [10.0, 10.0]})
    pid, dist = bc._attribute(
        0,
        np.array([10.0]),
        np.array([10.0]),
        np.array([0.1]),
        np.array(["measured"]),
        players,
        bc.BallContactParams(),
        overhead_active=True,
    )
    assert pid == 7 and dist < 1.0


def test_attribution_no_player_in_range_is_unresolved():
    players = pd.DataFrame({"player_id": [7], "x": [50.0], "y": [50.0]})
    pid, _dist = bc._attribute(
        0,
        np.array([10.0]),
        np.array([10.0]),
        np.array([0.1]),
        np.array(["measured"]),
        players,
        bc.BallContactParams(),
        overhead_active=True,
    )
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
        # TF-46 Arm 1 Part 2 attribution ambiguity (spec section 17.4)
        "attribution_margin",
        "attribution_margin_player",
        "attribution_other_visible",
        "attribution_other_unseen_reason",
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
    dv = bc._velocity_change_at(bx, by, tt, 3, 0.12)
    assert 13.0 < dv < 15.0


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
