"""The missed-penalty detector (TF-46 Arm 1 Part 2, spec section 12, signal ``penalty``).

``detect_penalties`` is PURE and mutates no row: it returns a detection table (``PENALTY_DETECTION_COLUMNS``),
one row per detected penalty, with the geometry evaluated on the MEDIAN position over the rest window (never
the release frame), the outcome / attribution per the data-profile goal-time semantics, the in-play /
shoot-out stratum, and the three gated fields (detection / attribution / split). The gate table is EMPTY in
commit 2 (CL-3), so ``outcome`` / ``attribution`` are ``<NA>`` (withheld) and ``penalty_evidence`` is unset.

Geometry runs in the action frame (section 4.3): the attacking team attacks +x, so the attacked goal is at
x = field_length and the spot is ~11 m before it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig

from ._gk_resolve import GoalMap
from ._kick_resolver import kinematic_release
from ._law_geometry import PitchDims, resolve_pitch_dims
from ._restart_frame import to_action_frame
from ._restart_validation import GATE_TABLE_VERSION, RATIFIED, DataProfile, gate


def _f(v) -> float:
    """pandas Scalar -> float; the untyped arg sidesteps the Scalar stub friction."""
    return float(v)


PENALTY_DETECTION_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "period_id": "int64",
    "rest_start": "float64",
    "release_time": "float64",
    "attacking_team_id": "object",
    "taker_player_id": "object",
    "keeper_player_id": "object",
    "outcome": "object",  # goal | saved | off_target | missed | <NA>
    "attribution": "object",  # goal | rebound | later_goal | missed | <NA>
    "stratum": "object",  # in_play | shoot_out
    "visible_share": "float64",
    "n_encroaching": "int64",
    "penalty_evidence": "object",
    "penalty_gate_version": "object",
}

_SHOT_PENALTY = spadlconfig.actiontype_id.get("shot_penalty")
_L = spadlconfig.field_length
_W = spadlconfig.field_width
_PA_DEPTH = 16.5
_ARC_R = RATIFIED["penalty_enc_disk_m"]  # 9.15


def _spot_x(pitch_length: float) -> float:
    """The penalty spot x in the action frame: 11 real metres before the attacked goal line (x=L), scaled
    from the provider pitch length to the 105 m frame (spec section 12). A constant 94.0 is NOT used. Delegates to
    the single Law-geometry helper (``_law_geometry.PitchDims``)."""
    return PitchDims(float(pitch_length)).spot_x


def _rest_median(ball_window: pd.DataFrame, rel: float, rest_s: float):
    """The ball's median (x, y) over the last ``rest_s`` of rest before the release ``rel`` (action frame
    coords already applied). ``None`` when no rest sample precedes the release."""
    pre = ball_window[(ball_window["time_seconds"] <= rel) & (ball_window["time_seconds"] >= rel - rest_s)]
    if pre.empty:
        return None
    return float(np.median(pre["x"])), float(np.median(pre["y"]))


def _encroach_count(
    players_rest: pd.DataFrame, spot, taker_id, keeper_id, e_enc: float, arc_r: float = _ARC_R
) -> tuple[int, float]:
    """Count outfield players (both teams) whose MEDIAN rest-window position is inside R by MORE than
    ``e_enc``, excluding the taker and the defending keeper. R = penalty area / the 9.15 m arc disk around the
    spot. Each player is counted ONCE (by its rest-window median). Unseen players are not counted; returns
    ``(count, visible_share)``."""
    from silly_kicks.id_compat import canonical_id, canonical_id_series

    from ._provider_visibility import detected_mask

    if players_rest.empty:
        return 0, float("nan")
    pr = players_rest.copy()
    pr["_cid"] = canonical_id_series(pr["player_id"]).to_numpy(dtype=object)
    pr["_seen"] = (
        detected_mask(pr["visibility"], provider="skillcorner") if "visibility" in pr else np.ones(len(pr), dtype=bool)
    )
    exclude = {
        canonical_id(taker_id) if taker_id is not None and not pd.isna(taker_id) else None,
        canonical_id(keeper_id) if keeper_id is not None and not pd.isna(keeper_id) else None,
    }
    half_w = spadlconfig.penalty_area_half_width
    count = 0
    n_seen = 0
    denom = 0
    for cid, g in pr.groupby("_cid", sort=False):
        if cid in exclude:
            continue
        denom += 1
        if not bool(g["_seen"].to_numpy(dtype=bool).any()):
            continue  # a player never seen in the rest window is not counted
        n_seen += 1
        mx, my = float(g["x"].median()), float(g["y"].median())
        in_disk = np.hypot(mx - spot[0], my - spot[1]) < (arc_r - e_enc)
        in_pa = (mx >= (_L - _PA_DEPTH) + e_enc) and (mx <= _L) and (abs(my - _W / 2.0) <= half_w - e_enc)
        if in_disk or in_pa:
            count += 1
    visible_share = (n_seen / denom) if denom else float("nan")
    return count, visible_share


def _action_frame_xy(frames_window: pd.DataFrame, rtl) -> pd.DataFrame:
    """Reflect a window of frames into the action frame (section 4.3) for the penalty's attacking team."""
    out = frames_window.copy()
    rtl_series = pd.Series([rtl] * len(out), index=out.index, dtype="boolean")
    ax, ay = to_action_frame(out["x"].to_numpy(dtype=np.float64), out["y"].to_numpy(dtype=np.float64), rtl_series)
    out["x"] = ax
    out["y"] = ay
    return out


def _spot_positions(pitch_length: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The two penalty spots in the ABSOLUTE frame as ``(spot_x, spot_y, attacked_goal_x)``: 11 real metres
    before each goal line, scaled from the provider pitch length to the 105 m frame (spec section 12)."""
    off = 11.0 * (_L / float(pitch_length))
    return (off, _W / 2.0, 0.0), (_L - off, _W / 2.0, _L)


def _rest_candidates(
    ball: pd.DataFrame, spots, r: float, s: float, k_rest: float = RATIFIED["k_rest_m"]
) -> list[tuple[float, float]]:
    """Scan ONE period's ball trace (absolute frame) for rest-at-a-penalty-spot segments, independent of any
    event row (spec section 12: the candidate comes from the tracking, so a missed penalty with no shot_penalty
    row is still found). A rest = frames staying within ``k_rest_m`` of a running centroid; it qualifies when it
    spans >= ``s`` and its median lies within ``r`` of a spot. Yields ``(release_time, attacked_goal_x)`` per
    candidate; the release is the first frame after the rest."""
    n = len(ball)
    if n < 2:
        return []
    t = ball["time_seconds"].to_numpy(dtype=np.float64)
    x = ball["x"].to_numpy(dtype=np.float64)
    y = ball["y"].to_numpy(dtype=np.float64)
    out: list[tuple[float, float]] = []
    i = 0
    while i < n:
        j = i
        cx, cy = float(x[i]), float(y[i])
        while j + 1 < n and np.hypot(x[j + 1] - cx, y[j + 1] - cy) <= k_rest:
            j += 1
            cx = float(np.median(x[i : j + 1]))
            cy = float(np.median(y[i : j + 1]))
        if float(t[j] - t[i]) >= s:
            mx = float(np.median(x[i : j + 1]))
            my = float(np.median(y[i : j + 1]))
            for sx, sy, goal_x in spots:
                if abs(mx - sx) <= r and abs(my - sy) <= r:
                    rel = float(t[j + 1]) if j + 1 < n else float(t[j])
                    out.append((rel, goal_x))
                    break
            i = j + 1
        else:
            i += 1
    return out


def _defending_keeper(players_rest: pd.DataFrame, d_gk: float, gk_y_tol: float):
    """The defending goalkeeper in the action frame: the GK whose MEDIAN rest-window x is nearest the attacked
    goal line (x = L). Returns ``(keeper_id, keeper_team, ok)`` -- ``ok`` is within ``d_gk`` of the line and
    laterally within the posts band. ``(None, None, False)`` when no GK is seen."""
    from silly_kicks.id_compat import canonical_id_series

    gks = players_rest[players_rest["is_goalkeeper"].to_numpy(dtype=bool)]
    if gks.empty:
        return None, None, False
    g = gks.copy()
    g["_cid"] = canonical_id_series(g["player_id"]).to_numpy(dtype=object)
    med = g.groupby("_cid", sort=False).agg(
        x=("x", "median"), y=("y", "median"), team=("team_id", "first"), pid=("player_id", "first")
    )
    row = med.iloc[int(np.nanargmax(med["x"].to_numpy(dtype=np.float64)))]
    ok = (_L - _f(row["x"])) <= d_gk and abs(_f(row["y"]) - _W / 2.0) <= gk_y_tol
    return row["pid"], row["team"], bool(ok)


def detect_penalties(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    provider: str,
    goal_map: GoalMap,
    contacts: pd.DataFrame | None = None,
    keepers=None,
    data_profile: DataProfile | None = None,
    gate_table: Mapping | None = None,
    constants: Mapping[str, float] | None = None,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
) -> pd.DataFrame:
    """Detect penalties from tracking ``frames`` (section 12), using ``actions`` only for the outcome / result.

    Pure; returns a detection table. The candidate is a ball-rest-at-the-pitch-scaled-spot in the tracking, NOT
    a window around a ``shot_penalty`` event: a missed penalty with no reconciliation row is still found, and the
    event row (when present) supplies the result for the gated outcome / attribution (empty gate -> ``<NA>`` in
    commit 2, CL-3).

    ``constants`` overrides individual ratified penalty constants (run (12) reports each at +/-50 %, spec
    section 16.1); default ``None`` = the ratified values.

    ``pitch_length`` is the provider's ORIGINAL pitch length (metres): the frames are scaled to the 105 m frame,
    so it fixes where the 11 m spot lands (spec section 12; the constant 94.0 is never used). Declared input
    first; else the SkillCorner converter's ``frames.attrs["pitch_length"]`` stamp (attrs do not survive every
    pandas op, so a caller that reshapes frames passes it explicitly); else 105.0 (a provider already on the 105 m
    frame).

    Examples
    --------
    Detect once per match, with the match's GoalMap, then hand the table to ``refine_restart_types``::

        frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
        goal_map = resolve_defended_goals(frames)
        pens = detect_penalties(actions, frames, provider="skillcorner", goal_map=goal_map, data_profile=profile)
        # one row per detected penalty: release_time, attacking_team_id, keeper_player_id, stratum, outcome, ...
    """
    c = {**RATIFIED, **(constants or {})}
    if frames.empty:
        return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in PENALTY_DETECTION_COLUMNS.items()})

    # The frames are already scaled to the 105 m action frame, so the ORIGINAL pitch length (which fixes where
    # the 11 m spot lands after scaling) must come from the frame attrs stamped by the converter.
    dims = resolve_pitch_dims(frames, pitch_length, pitch_width)
    pitch_length = dims.length
    spot_x = dims.spot_x
    spots_abs = _spot_positions(pitch_length)
    r = c["penalty_spot_r_m"]
    s = c["penalty_rest_s"]
    d_gk = c["penalty_d_gk_m"]
    gk_y_tol = c["penalty_gk_y_tol_m"]
    eval_s = c["penalty_position_eval_s"]
    e_enc = c["penalty_e_enc_m"]
    k_max = round(c["penalty_enc_count_k"])
    taker_near = c["penalty_taker_near_m"]
    w_rb = c["penalty_W_rb_s"]

    det_entry = (
        gate(data_profile, "penalty", "penalty", "detection", table=gate_table) if data_profile is not None else None
    )
    attr_entry = (
        gate(data_profile, "penalty", "penalty", "attribution", table=gate_table) if data_profile is not None else None
    )

    pens = (
        actions[actions["type_id"] == _SHOT_PENALTY]
        if _SHOT_PENALTY is not None and "type_id" in actions
        else actions.iloc[:0]
    )
    has_gid = "game_id" in frames.columns
    keys = frames[["game_id", "period_id"]].drop_duplicates() if has_gid else frames[["period_id"]].drop_duplicates()

    rows: list[dict] = []
    for _, key in keys.iterrows():
        period = int(key["period_id"])
        gid = key["game_id"] if has_gid else None
        gp = frames[(frames["game_id"] == gid) & (frames["period_id"] == period)] if has_gid else frames
        ball_abs = gp[gp["is_ball"].to_numpy(dtype=bool)].sort_values("time_seconds")
        for rel_abs, goal_x in _rest_candidates(ball_abs, spots_abs, r, s, c["k_rest_m"]):
            # Transform the window around the candidate into the action frame: the attacked goal lands at x = L.
            rtl = goal_x == 0.0
            win = _action_frame_xy(
                gp[(gp["time_seconds"] >= rel_abs - 5.0) & (gp["time_seconds"] <= rel_abs + w_rb)], rtl
            )
            ball = win[win["is_ball"].to_numpy(dtype=bool)].sort_values("time_seconds")
            # The window is already in the action frame (attacks +x): the section 6.4 spot rest + toward-goal
            # release apply directly; a keeper-held ball after the kick is not at the spot and never anchors K.
            rel, _rest_xy = (
                kinematic_release(ball, restart_type="penalty", attacks_rtl=False, pitch=dims)
                if len(ball)
                else (float("nan"), None)
            )
            if not np.isfinite(rel):
                rel = rel_abs
            med = _rest_median(ball, rel, eval_s)
            stratum = "shoot_out" if period >= 5 else "in_play"

            rest_frames = win[(win["time_seconds"] <= rel) & (win["time_seconds"] >= rel - eval_s)]
            players_rest = rest_frames[~rest_frames["is_ball"].to_numpy(dtype=bool)]
            keeper_id, keeper_team, gk_ok = _defending_keeper(players_rest, d_gk, gk_y_tol)

            # taker: the nearest NON-defending-team outfielder to the ball at the release; its team attacks.
            taker_id, team = _taker_at_release(win, rel, keeper_team, med if med else (spot_x, _W / 2.0), taker_near)

            ball_ok = (
                med is not None
                and abs(med[0] - spot_x) <= r
                and abs(med[1] - _W / 2.0) <= r
                and _rest_duration(ball, rel, c["k_rest_m"]) >= s
            )
            n_enc, vis_share = _encroach_count(
                players_rest, (spot_x, _W / 2.0), taker_id, keeper_id, e_enc, c["penalty_enc_disk_m"]
            )
            detected = bool(ball_ok and gk_ok and n_enc <= k_max)
            if not detected:
                continue

            # outcome / attribution (gated; <NA> in commit 2). The event row (when one exists near the release)
            # supplies the result; otherwise a synthetic row carries the period with an unknown result.
            pen = _event_for(pens, gid, period, rel, w_rb)
            outcome, attribution = (pd.NA, pd.NA)
            if det_entry is not None and attr_entry is not None:
                outcome, attribution = _outcome(pen, win, ball, rel, provider, contacts, w_rb, keeper_id, c)

            rows.append(
                {
                    "game_id": gid,
                    "period_id": period,
                    "rest_start": rel - eval_s,
                    "release_time": rel,
                    "attacking_team_id": team,
                    "taker_player_id": taker_id,
                    "keeper_player_id": keeper_id,
                    "outcome": outcome,
                    "attribution": attribution,
                    "stratum": stratum,
                    "visible_share": vis_share,
                    "n_encroaching": n_enc,
                    "penalty_evidence": (det_entry.tier if det_entry is not None else pd.NA),
                    "penalty_gate_version": GATE_TABLE_VERSION,
                }
            )
    if not rows:
        return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in PENALTY_DETECTION_COLUMNS.items()})
    out = pd.DataFrame(rows)
    for c, d in PENALTY_DETECTION_COLUMNS.items():
        if d in ("int64", "float64"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(d) if d == "float64" else out[c].astype("int64")
        else:
            out[c] = out[c].astype("object")
    return out[list(PENALTY_DETECTION_COLUMNS)]


def _event_for(pens: pd.DataFrame, gid, period: int, rel: float, w_rb: float) -> pd.Series:
    """The ``shot_penalty`` event row nearest the candidate release in the window ``[rel - 5, rel + W_rb]`` (the
    result for the gated outcome), or a synthetic row carrying only the period when no event row backs the
    candidate (a tracking-only missed penalty)."""

    if len(pens):
        c = pens[pens["period_id"].astype("int64") == period]
        if gid is not None and "game_id" in c:
            c = c[c["game_id"].map(lambda v: same_id(v, gid))]
        c = c[(c["time_seconds"] >= rel - 5.0) & (c["time_seconds"] <= rel + w_rb)]
        if len(c):
            return c.iloc[int(np.nanargmin((c["time_seconds"] - rel).abs().to_numpy(dtype=np.float64)))]
    return pd.Series({"period_id": period, "result_id": pd.NA, "time_seconds": rel})


def _rest_duration(ball: pd.DataFrame, rel: float, k_rest: float = RATIFIED["k_rest_m"]) -> float:
    """Seconds the ball rested (within k_rest_m of its pre-release centroid) before ``rel``."""
    pre = ball[ball["time_seconds"] <= rel]
    if len(pre) < 2:
        return 0.0
    tail = pre.tail(max(2, len(pre)))
    cx, cy = float(tail["x"].median()), float(tail["y"].median())
    near = tail[np.hypot(tail["x"] - cx, tail["y"] - cy) <= k_rest]
    if near.empty:
        return 0.0
    return float(near["time_seconds"].max() - near["time_seconds"].min())


def _taker_at_release(win: pd.DataFrame, rel: float, exclude_team, spot, near: float):
    """The taker = the outfielder of the NON-defending team nearest the ball/spot at the release frame, within
    ``near``. Returns ``(taker_id, attacking_team_id)`` -- the attacking team is read FROM the taker's own team
    row (the detector never takes the attacking team from an event). ``(None, None)`` when no such player is
    seen within ``near``."""

    at = win[(np.abs(win["time_seconds"] - rel) <= 0.05) & ~win["is_ball"].to_numpy(dtype=bool)]
    if exclude_team is not None:
        at = at[at["team_id"].map(lambda t: t is None or not same_id(t, exclude_team))]
    at = at[at["team_id"].notna()]
    if at.empty:
        return None, None
    d = np.hypot(at["x"].to_numpy(dtype=np.float64) - spot[0], at["y"].to_numpy(dtype=np.float64) - spot[1])
    if not np.isfinite(d).any() or float(np.nanmin(d)) > near:
        return None, None
    taker = at.iloc[int(np.nanargmin(d))]
    return taker["player_id"], taker["team_id"]


_GOAL_HALF_W = 3.66  # the goal mouth half-width (7.32 m posts), action frame centre y = W/2


def _outcome(pen, win, ball, rel, provider, contacts, w_rb, keeper_id, c: Mapping[str, float] = RATIFIED):
    """Outcome + attribution (section 12). ``goal`` requires a score change AND no non-goalkeeper in-field
    contact over [release, release + W_rb]; on SkillCorner it also requires the ball not be LIVE in
    [W, W_rb] (excluding a post-goal net clearance, C-58/C-61). Otherwise ``missed`` -- ``saved`` when a
    defending-keeper contact in flight is seen, else ``off_target`` -- with ``rebound`` / ``later_goal``
    attribution when the score changed but the goal was not the penalty's. ``goal`` after a keeper touch or
    woodwork stays the penalty's (the non-GK check excludes the keeper)."""
    w = c["penalty_W_s"]
    score_goal = int(pen.get("result_id", -1)) == spadlconfig.result_id.get("success", -99)
    nongk = _nongk_contact(contacts, pen, rel, w_rb, keeper_id)
    live = _live_after_w(ball, rel, w, w_rb, c["penalty_rb_live_speed_ms"])
    keeper_contact = _keeper_contact_in_flight(contacts, pen, rel, w, keeper_id)
    is_goal = score_goal and not nongk and (provider in ("idsse", "gradientsports") or not live)
    if is_goal:
        return "goal", "goal"
    if not score_goal:
        return ("saved" if keeper_contact else "off_target"), "missed"
    # score changed, but not the penalty's goal: a rebound (a non-GK in-field touch in W_rb) or a later goal.
    attr = "rebound" if nongk else "later_goal"
    return ("saved" if keeper_contact else "off_target"), attr


def _nongk_contact(contacts, pen, rel, w_rb, keeper_id) -> bool:
    """A non-goalkeeper ball contact over [release, release + W_rb] (GI rule (c); C-52)."""
    from silly_kicks.id_compat import canonical_id, canonical_id_series

    if contacts is None or len(contacts) == 0:
        return False
    c = contacts[
        (contacts["period_id"] == int(pen["period_id"]))
        & (contacts["time_seconds"] >= rel)
        & (contacts["time_seconds"] <= rel + w_rb)
    ]
    c = c[c["player_id"].notna()]
    if c.empty:
        return False
    kid = canonical_id(keeper_id) if keeper_id is not None and not pd.isna(keeper_id) else None
    return bool(any(cid != kid for cid in canonical_id_series(c["player_id"]).to_numpy(dtype=object)))


def _live_after_w(ball, rel, w, w_rb, live_speed: float = RATIFIED["penalty_rb_live_speed_ms"]) -> bool:
    """The ball is LIVE in [W, W_rb] after the release: detected, in the field of play, >= 3 m/s -- EXCLUDING
    a segment that starts after the ball entered the goal (a net clearance, C-58/C-61: the live frame or the
    position just before it lies beyond the goal line between the posts)."""
    seg = ball[(ball["time_seconds"] >= rel) & (ball["time_seconds"] <= rel + w_rb)].sort_values("time_seconds")
    if len(seg) < 2:
        return False
    t = seg["time_seconds"].to_numpy(dtype=np.float64)
    x = seg["x"].to_numpy(dtype=np.float64)
    y = seg["y"].to_numpy(dtype=np.float64)
    sp = np.zeros(len(t))
    dt = np.diff(t)
    sp[1:] = np.where(dt > 0, np.hypot(np.diff(x), np.diff(y)) / dt, 0.0)
    infield = (x >= 0) & (x <= _L) & (y >= 0) & (y <= _W)
    live = (sp >= live_speed) & infield & (t >= rel + w)
    for i in np.where(live)[0]:
        beyond_now = x[i] > _L and abs(y[i] - _W / 2.0) <= _GOAL_HALF_W
        beyond_prev = i > 0 and x[i - 1] > _L and abs(y[i - 1] - _W / 2.0) <= _GOAL_HALF_W
        if not (beyond_now or beyond_prev):
            return True  # a genuine live segment not starting from inside the goal
    return False


def _keeper_contact_in_flight(contacts, pen, rel, w, keeper_id) -> bool:
    """A defending-goalkeeper ball contact in flight over [release, release + W] (the ``saved`` signal)."""
    from silly_kicks.id_compat import canonical_id, canonical_id_series

    if contacts is None or len(contacts) == 0:
        return False
    c = contacts[
        (contacts["period_id"] == int(pen["period_id"]))
        & (contacts["time_seconds"] >= rel)
        & (contacts["time_seconds"] <= rel + w)
    ]
    c = c[c["player_id"].notna()]
    if c.empty:
        return False
    if keeper_id is None or pd.isna(keeper_id):
        return True  # any in-flight contact, keeper unresolved -> treat as a save (conservative)
    kid = canonical_id(keeper_id)
    return bool(any(cid == kid for cid in canonical_id_series(c["player_id"]).to_numpy(dtype=object)))
