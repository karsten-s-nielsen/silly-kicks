"""ELASTIC v2 event-tracking synchronization (Kim et al. 2026).

Aligns SPADL events to tracking frames as a GLOBAL, order-preserving sequence
alignment (an extended Needleman-Wunsch DP) between the event sequence -- enriched
with virtual termination events so each event's end (reception / out / goal) is found
jointly with its start -- and a sparse set of physically-plausible ball-touch candidate
frames. Clean-room reimplementation of the published method; no neural nets (scipy peak
detection + a numpy DP). See spec ``docs/superpowers/specs/2026-09-11-tf57-elastic-nw-sync-design.md``.

Two surfaces over one pure engine:
- ``align_events_to_frames`` / ``add_elastic_sync`` / ``elastic_sync_xfns`` -- the elastic
  columns (``elastic_frame_id`` + reception frame).
- ``link_actions_to_frames_elastic`` (in ``utils``) -- the canonical ``(pointers, LinkReport)``
  contract, an alternative to the guarded time-based ``link_actions_to_frames`` (which stays
  the default; this does NOT replace it).

See NOTICE for full bibliographic citations.

References
----------
Kim, H., Kim, J., & Kim, H. (2026). "ELASTIC: Event-Level Alignment of STreaming data
Including Coordinates." CIKM 2026. arXiv:2608.30227. (Extends the MLSA-2025 version,
arXiv:2508.09238, with the extended-Needleman-Wunsch alignment + virtual termination events.)
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id, canonical_id_series, same_id
from silly_kicks.spadl import config as spadlconfig

#: A physically-plausible ball-touch candidate frame: ``players`` is the tuple of canonical
#: player ids within ``touch_distance_m`` of the ball at that frame (the paper's set P_c).
Candidate = namedtuple("Candidate", "game_id period_id frame_id players")


@dataclass(frozen=True)
class ElasticSyncParams:
    """Parameters for the ELASTIC v2 (extended Needleman-Wunsch) sync.

    All values are the paper's intent-set constants (Kim et al. 2026,
    arXiv:2608.30227); ``for_provider`` is intentionally absent (ADR-009).

    Examples
    --------
    >>> params = ElasticSyncParams()
    >>> params.min_confidence
    0.5
    """

    frame_rate: int = 25
    #: Feasibility + PBD/KD/OD clip upper bound (m): a candidate (frame, player) is
    #: retained only where player-ball distance <= this.
    touch_distance_m: float = 3.0
    #: Feasibility upper bound on ball height (m); NO-OP where ``z`` is NaN.
    ball_height_max_m: float = 4.0
    #: s_BA clip upper bound (m/s^2).
    accel_clip_max: float = 30.0
    #: Slope window h for the pre/post player-ball-distance-slope features (s).
    slope_window_seconds: float = 0.2
    #: PBDS clip bound (m/s).
    slope_clip_mps: float = 7.0
    #: Down-match repeat penalty r (Eq. 14): cost of one candidate frame serving two
    #: consecutive events (one-touch actions).
    repeat_penalty: float = -0.1
    #: Event-gap penalty g_e (Eq. 14): cost of leaving an event unmatched.
    event_gap_penalty: float = 0.0
    #: Candidate-gap penalty g_c (Eq. 14): cost of leaving a candidate frame unused.
    candidate_gap_penalty: float = 0.0
    #: Post-processing rejection threshold: a matched score below this is unsynchronized.
    min_confidence: float = 0.5


def _col_f64(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract column as float64 numpy array."""
    return np.asarray(df[col].values, dtype=np.float64)


def _ball_kinematics(
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams,
) -> pd.DataFrame:
    """Per-(game_id, period_id, frame_id) ball kinematics + distance to nearest boundary.

    Velocity/speed/acceleration via per-period finite differences; ``boundary_dist`` = distance
    to the nearest pitch line (``min(x, L-x, y, W-y)``); ``ball_z`` passes through (NaN where the
    provider has no height). Backs both ``extract_ball_features`` and ``_detect_candidate_frames``.
    """
    out_cols = pd.Index(
        [
            "game_id",
            "period_id",
            "frame_id",
            "ball_x",
            "ball_y",
            "ball_z",
            "ball_speed",
            "ball_accel",
            "boundary_dist",
        ]
    )
    if frames.empty:
        return pd.DataFrame(columns=out_cols)

    ball_mask = frames["is_ball"] == True  # noqa: E712
    has_z = "z" in frames.columns
    sel = ["game_id", "period_id", "frame_id", "x", "y"] + (["z"] if has_z else [])
    b = frames.loc[ball_mask, sel].copy()
    b = b.dropna(subset=["x", "y"])
    b = b.drop_duplicates(subset=["game_id", "period_id", "frame_id"])
    b = b.sort_values(["game_id", "period_id", "frame_id"]).reset_index(drop=True)
    if b.empty:
        return pd.DataFrame(columns=out_cols)

    dt = 1.0 / params.frame_rate
    bx = _col_f64(b, "x")
    by = _col_f64(b, "y")
    game_ids = np.asarray(b["game_id"].values)
    period_ids = np.asarray(b["period_id"].values)

    vx = np.zeros_like(bx)
    vy = np.zeros_like(by)
    same_group = (game_ids[1:] == game_ids[:-1]) & (period_ids[1:] == period_ids[:-1])
    vx[1:] = np.where(same_group, (bx[1:] - bx[:-1]) / dt, 0.0)
    vy[1:] = np.where(same_group, (by[1:] - by[:-1]) / dt, 0.0)
    speed = np.sqrt(vx**2 + vy**2)
    # Acceleration = magnitude of the ball's velocity-VECTOR change, computed as the CENTRAL second
    # difference of position a[i] = (x[i+1] - 2*x[i] + x[i-1]) / dt**2 -- the paper's "acceleration
    # directly from position differences" (arXiv:2608.30227 Eq. 3). Two properties are load-bearing:
    #  (1) VECTOR (not speed-magnitude): a touch that redirects the ball at ~constant speed has |dv|
    #      large but |d|v|| ~ 0, so the speed-difference form peaked on decelerations/bounces and
    #      MISSED the direction-change touches (the dominant s_BA mis-peak).
    #  (2) CENTRAL (not a double backward difference): a backward a[i] = (v[i]-v[i-1])/dt is centred
    #      at i-1, so its peak lands one frame AFTER the true touch. On the CC-BY 3-match corpus the
    #      signed offset (pred-gt) was bimodal at 0 and +1; the central form collapses the +1 spike
    #      into exact-frame (pooled exact 0.335 -> 0.583, W2 0.835 -> 0.845). See ADR-093 / spec 16.
    ax = np.zeros_like(bx)
    ay = np.zeros_like(by)
    same3 = (
        (game_ids[2:] == game_ids[1:-1])
        & (game_ids[1:-1] == game_ids[:-2])
        & (period_ids[2:] == period_ids[1:-1])
        & (period_ids[1:-1] == period_ids[:-2])
    )
    ax[1:-1] = np.where(same3, (bx[2:] - 2.0 * bx[1:-1] + bx[:-2]) / dt**2, 0.0)
    ay[1:-1] = np.where(same3, (by[2:] - 2.0 * by[1:-1] + by[:-2]) / dt**2, 0.0)
    accel = np.sqrt(ax**2 + ay**2)

    length = float(spadlconfig.field_length)
    width = float(spadlconfig.field_width)
    boundary = np.minimum.reduce([bx, length - bx, by, width - by])

    result = b[["game_id", "period_id", "frame_id"]].copy()
    result["ball_x"] = bx
    result["ball_y"] = by
    result["ball_z"] = _col_f64(b, "z") if has_z else np.nan
    result["ball_speed"] = speed
    result["ball_accel"] = accel
    result["boundary_dist"] = boundary
    return result


def extract_ball_features(
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams | None = None,
) -> pd.DataFrame:
    """Extract ball speed + acceleration per (game_id, period_id, frame_id).

    Thin public projection of :func:`_ball_kinematics` (kept for backward compatibility).

    Returns
    -------
    pd.DataFrame
        Columns: ``game_id``, ``period_id``, ``frame_id``, ``ball_x``, ``ball_y``,
        ``ball_speed``, ``ball_accel``.

    Examples
    --------
    Extract per-frame ball kinematics from a match's frames::

        bf = extract_ball_features(frames)
        bf.columns.tolist()
        # ['game_id', 'period_id', 'frame_id', 'ball_x', 'ball_y', 'ball_speed', 'ball_accel']
    """
    if params is None:
        params = ElasticSyncParams()
    cols = pd.Index(["game_id", "period_id", "frame_id", "ball_x", "ball_y", "ball_speed", "ball_accel"])
    kin = _ball_kinematics(frames, params=params)
    if kin.empty:
        return pd.DataFrame(columns=cols)
    return kin[cols].copy()


def _detect_candidate_frames(
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams,
) -> dict[tuple, list[Candidate]]:
    """Sparse ball-touch candidate frames per ``(canonical game_id, period_id)``.

    A candidate is a local minimum of nearest-player-ball distance, a local minimum of
    ball-to-boundary distance, or a local maximum of ball acceleration. ``players`` = canonical
    ids within ``touch_distance_m`` of the ball at that frame, gated by ball height <=
    ``ball_height_max_m`` (no-op where ``z`` is NaN). Frames with no feasible player are dropped.
    """
    if frames.empty:
        return {}
    ball = _ball_kinematics(frames, params=params)
    if ball.empty:
        return {}

    ball_mask = frames["is_ball"] == True  # noqa: E712
    pl = frames.loc[~ball_mask, ["game_id", "period_id", "frame_id", "player_id", "x", "y"]].copy()
    pl = pl.dropna(subset=["x", "y", "player_id"])
    if pl.empty:
        return {}
    merged = pl.merge(
        ball[["game_id", "period_id", "frame_id", "ball_x", "ball_y", "ball_z"]],
        on=["game_id", "period_id", "frame_id"],
        how="inner",
    )
    if merged.empty:
        return {}
    dist = np.hypot(
        merged["x"].to_numpy(dtype=float) - merged["ball_x"].to_numpy(dtype=float),
        merged["y"].to_numpy(dtype=float) - merged["ball_y"].to_numpy(dtype=float),
    )
    merged["dist"] = dist
    zc = merged["ball_z"].to_numpy(dtype=float)
    z_ok = np.isnan(zc) | (zc <= params.ball_height_max_m)
    feas = merged.loc[(dist <= params.touch_distance_m) & z_ok].copy()
    feas_groups = group_rows(feas, ("game_id", "period_id", "frame_id"))
    mpd = merged.groupby(["game_id", "period_id", "frame_id"])["dist"].min().reset_index()
    mpd_lookup = {
        (canonical_id(g), int(p), int(f)): float(v)
        for g, p, f, v in zip(mpd["game_id"], mpd["period_id"], mpd["frame_id"], mpd["dist"], strict=True)
    }

    out: dict[tuple, list[Candidate]] = {}
    for (gid, pid), bgrp in ball.groupby(["game_id", "period_id"], sort=False):
        bgrp = bgrp.sort_values("frame_id")
        fids = bgrp["frame_id"].to_numpy()
        accel = bgrp["ball_accel"].to_numpy(dtype=float)
        boundary = bgrp["boundary_dist"].to_numpy(dtype=float)
        gcanon = canonical_id(gid)
        pnum = int(pid)  # type: ignore[arg-type]  # groupby key Hashable (pandas-stubs)
        nearest = np.array([mpd_lookup.get((gcanon, pnum, int(f)), np.inf) for f in fids], dtype=float)
        boundary_pos = set(argrelextrema(boundary, np.less)[0].tolist())
        cand_pos: set[int] = set(boundary_pos)
        cand_pos.update(argrelextrema(nearest, np.less)[0].tolist())
        cand_pos.update(find_peaks(accel)[0].tolist())
        cands: list[Candidate] = []
        for i in sorted(cand_pos):
            fid = int(fids[i])
            sub = feas_groups.get(gid, pid, fid)
            players = tuple(canonical_id_series(sub["player_id"]).tolist()) if len(sub) else ()
            # Player-touch candidates need a feasible player; boundary-local-min candidates are
            # kept regardless (they anchor the ball-out / goal virtual termination events, which
            # have no acting player). See _score's "out"/"goal" branch.
            if not players and i not in boundary_pos:
                continue
            cands.append(Candidate(gcanon, pnum, fid, players))
        if cands:
            out[(gcanon, pnum)] = cands
    return out


#: Per-(game, period) scoring lookups (built once by the assembler, threaded into ``_score``):
#: ``accel`` frame->ball acceleration; ``dist`` (frame, canon player)->player-ball distance;
#: ``frame_players`` frame->list of canon player ids present; ``player_team`` canon player->canon team;
#: ``boundary`` frame->ball distance to the nearest pitch line (used by the out/goal branch only).
_FrameLookups = namedtuple("_FrameLookups", "accel dist frame_players player_team boundary", defaults=(None,))


def _clip_linear(x: float, x0: float, x1: float) -> float:
    """Clipped linear map ``f(x; x0, x1)`` (spec 3.1 Eq. 2): 0 below x0, 1 above x1, linear between."""
    if x1 == x0:
        return 1.0 if x >= x1 else 0.0
    return float(min(1.0, max(0.0, (x - x0) / (x1 - x0))))


def _win_max_dist(lk: _FrameLookups, actor: str, lo: int, hi: int, fallback: float) -> float:
    """Max player-ball distance for ``actor`` over present frames in ``[lo, hi]`` (else ``fallback``)."""
    vals = [lk.dist[(fr, actor)] for fr in range(lo, hi + 1) if (fr, actor) in lk.dist]
    return max(vals) if vals else fallback


def _score(event, candidate: Candidate, lookups: _FrameLookups, *, params: ElasticSyncParams) -> float:
    """Pairwise event-candidate compatibility ``s(e, c) in [0, 1]`` (spec 3.1; scoring refined, ADR-093).

    Hard gate: 0.0 if the acting player is not among the candidate's feasible players ``P_c``.
    Category-dispatched WEIGHTED blend (``w_ba=0.6, w_pbd=1.6, w_kd=0.8, w_dyn=1.0``): proximity
    (s_PBD) is UP-weighted as the strongest true-touch localiser, while acceleration (s_BA) and
    kick-distance (s_KD) are DAMPENED because decoys (hard control touches, carries, bounces) inflate
    them. Outgoing/incoming carry a directional slope term (s_dyn) rewarding the ball DEPARTING
    (outgoing) or ARRIVING (incoming) across the touch -- separating the true touch from a
    stationary-carry decoy. The virtual ``out``/``goal`` termination events have no acting player and
    are scored purely by ball-to-boundary proximity.

    The weights + the directional term were derived by an OpenEvolve LLM code-search over this
    formulation on the WC2022 CC-BY 3-match corpus (ADR-093 / spec 16): vs the uniform-0.25 baseline
    they lift the white-box score-peak (0.728 -> 0.745) and W2 (0.856 -> 0.862) on ALL THREE folds
    without regressing the min-fold W2 or coverage. The blend of clipped-linear [0,1] terms with
    positive weights keeps ``s`` in [0, 1].
    """
    tc = int(candidate.frame_id)
    touch = params.touch_distance_m

    if event.category in ("out", "goal"):
        bd = (lookups.boundary or {}).get(tc)
        return 0.0 if bd is None else 1.0 - _clip_linear(bd, 0.0, touch)

    actor = canonical_id(event.player_id)
    if not isinstance(actor, str) or actor not in candidate.players:
        return 0.0
    h_frames = max(1, round(params.slope_window_seconds * params.frame_rate))
    h = params.slope_window_seconds

    s_ba = _clip_linear(lookups.accel.get(tc, 0.0), 0.0, params.accel_clip_max)
    d_tc = lookups.dist.get((tc, actor), touch)
    s_pbd = 1.0 - _clip_linear(d_tc, 0.0, touch)

    # Evolve-derived weights (ADR-093): up-weight proximity (the cleanest true-touch cue), dampen
    # accel + kick-distance (decoys inflate them). See the module/function attribution above.
    w_ba, w_pbd, w_kd, w_dyn = 0.6, 1.6, 0.8, 1.0
    w_sum = w_ba + w_pbd + w_kd + w_dyn

    if event.category == "outgoing":
        s_kd = _clip_linear(_win_max_dist(lookups, actor, tc, tc + h_frames, d_tc), 0.0, touch)
        v_pre = (d_tc - lookups.dist.get((tc - h_frames, actor), d_tc)) / h
        s_pbds = 1.0 - _clip_linear(v_pre, 0.0, params.slope_clip_mps)
        # directional: at the true outgoing touch the ball is close AND about to leave -> reward the
        # forward-window departure slope, separating it from a stationary-carry decoy.
        v_after = (lookups.dist.get((tc + h_frames, actor), d_tc) - d_tc) / h
        s_leave = _clip_linear(v_after, 0.0, params.slope_clip_mps)
        s_dyn = 0.5 * (s_pbds + s_leave)
        return (w_ba * s_ba + w_pbd * s_pbd + w_kd * s_kd + w_dyn * s_dyn) / w_sum
    if event.category == "incoming":
        s_kd = _clip_linear(_win_max_dist(lookups, actor, tc - h_frames, tc, d_tc), 0.0, touch)
        v_post = (lookups.dist.get((tc + h_frames, actor), d_tc) - d_tc) / h
        s_pbds = _clip_linear(v_post, -params.slope_clip_mps, 0.0)
        # directional: at the true incoming touch the ball is close AND was arriving -> reward the
        # backward-window approach slope, separating it from a following carry/control decoy.
        v_before = (d_tc - lookups.dist.get((tc - h_frames, actor), d_tc)) / h
        s_arrive = _clip_linear(-v_before, 0.0, params.slope_clip_mps)
        s_dyn = 0.5 * (s_pbds + s_arrive)
        return (w_ba * s_ba + w_pbd * s_pbd + w_kd * s_kd + w_dyn * s_dyn) / w_sum
    # minor (contested): symmetric kick distance + nearest-opponent proximity
    s_kd = _clip_linear(_win_max_dist(lookups, actor, tc - h_frames, tc + h_frames, d_tc), 0.0, touch)
    actor_team = lookups.player_team.get(actor)
    opp_dists = [
        lookups.dist.get((tc, q), touch)
        for q in lookups.frame_players.get(tc, ())
        if lookups.player_team.get(q) != actor_team
    ]
    min_opp = min(opp_dists) if opp_dists else touch
    s_od = 1.0 - _clip_linear(min_opp, 0.0, touch)
    return (w_ba * s_ba + w_pbd * s_pbd + w_kd * s_kd + w_dyn * s_od) / w_sum


#: An event in the alignment sequence. Virtual termination events carry ``action_id = pd.NA`` and
#: ``kind != "real"``. ``category`` is the scoring category ("outgoing"/"incoming"/"minor" for real
#: events + receptions, "out"/"goal" for the ball-boundary termination events).
Event = namedtuple("Event", "action_id player_id category kind expected_time")

#: SPADL type_name -> ELASTIC scoring category (spec section 7). Set-pieces use the "outgoing"
#: composition. ``dribble`` (synthetic carry) and ``non_action``/cards map to None (excluded).
_OUTGOING_TYPES = frozenset(
    {
        "pass",
        "cross",
        "shot",
        "shot_freekick",
        "clearance",
        "take_on",
        "keeper_punch",
        "throw_in",
        "freekick_crossed",
        "freekick_short",
        "corner_crossed",
        "corner_short",
        "goalkick",
        "shot_penalty",
    }
)
_INCOMING_TYPES = frozenset({"interception", "keeper_save", "keeper_claim", "keeper_pick_up"})
_MINOR_TYPES = frozenset({"foul", "tackle", "bad_touch"})
#: A following action of one of these types means the ball went out of play before it (spec section 7).
_OUT_TRIGGER_TYPES = frozenset({"throw_in", "goalkick", "corner_crossed", "corner_short"})
_SHOT_TYPES = frozenset({"shot", "shot_penalty", "shot_freekick"})


def _map_category(type_name) -> str | None:
    """SPADL ``type_name`` -> scoring category, or None if the event is not a ball touch (spec 7)."""
    if type_name in _OUTGOING_TYPES:
        return "outgoing"
    if type_name in _INCOMING_TYPES:
        return "incoming"
    if type_name in _MINOR_TYPES:
        return "minor"
    return None


def _virtual_between(cur, nxt) -> Event | None:
    """The virtual termination event to insert between consecutive real events, or None (spec 7)."""
    et = (float(cur.time_seconds) + float(nxt.time_seconds)) / 2.0
    if getattr(nxt, "type_name", None) in _OUT_TRIGGER_TYPES:
        return Event(pd.NA, cur.player_id, "out", "out", et)
    if getattr(cur, "type_name", None) in _SHOT_TYPES and getattr(cur, "result_name", None) == "success":
        return Event(pd.NA, cur.player_id, "goal", "goal", et)
    same_possession = getattr(cur, "possession_id", None) == getattr(nxt, "possession_id", None)
    if (
        same_possession
        and pd.notna(cur.player_id)
        and pd.notna(nxt.player_id)
        and not same_id(cur.player_id, nxt.player_id)
    ):
        # reception ("control"): scored as the NEXT actor receiving the ball (incoming).
        return Event(pd.NA, nxt.player_id, "incoming", "reception", et)
    return None


def _enrich_events(actions_period: pd.DataFrame, *, params: ElasticSyncParams | None = None) -> list[Event]:
    """Enrich a period's chronological, possession-tagged SPADL actions with virtual termination
    events (reception / out / goal), returning the alternating event sequence (spec 2.1 / 7).

    ``actions_period`` must carry ``action_id``, ``player_id``, ``type_name``, ``possession_id``,
    ``time_seconds`` (and ``result_name`` for the goal rule). Excluded types (``_map_category``
    None) are dropped. ``params`` is accepted for call-site uniformity (unused here).
    """
    del params  # accepted for a uniform engine call signature; enrichment has no thresholds
    reals = [r for r in actions_period.itertuples() if _map_category(getattr(r, "type_name", None)) is not None]
    events: list[Event] = []
    for k, r in enumerate(reals):
        events.append(
            Event(
                int(r.action_id),  # type: ignore[arg-type]  # itertuples attr typed Scalar; pandas-stubs limitation
                r.player_id,
                _map_category(r.type_name),
                "real",
                float(r.time_seconds),  # type: ignore[arg-type]  # itertuples attr typed Scalar; pandas-stubs limitation
            )
        )
        if k + 1 < len(reals):
            virt = _virtual_between(r, reals[k + 1])
            if virt is not None:
                events.append(virt)
    return events


def _needleman_wunsch(score_matrix: np.ndarray, *, params: ElasticSyncParams) -> list[int | None]:
    """Extended Needleman-Wunsch alignment (spec 3.1 Eq. 14) of ``m`` events to ``n`` candidates.

    ``score_matrix[i, j]`` is ``s(e_i, c_j) in [0, 1]``. Returns a length-``m`` list mapping each
    event to a candidate column index, or ``None`` where the event is gapped (unmatched). Moves:
    diag-match, candidate-gap (g_c), event-gap (g_e), and the down-match (a candidate serving two
    consecutive events at cost ``repeat_penalty``). Order-preserving by construction. Ties prefer
    gaps over matches, so a zero-information (s<=0) match is left unmatched rather than fabricated.
    """
    sm = np.asarray(score_matrix, dtype=float)
    m, n = sm.shape
    if m == 0:
        return []
    ge = params.event_gap_penalty
    gc = params.candidate_gap_penalty
    r = params.repeat_penalty
    neg = float("-inf")
    dp = np.full((m + 1, n + 1), neg, dtype=float)  # DP table F in Eq. 14
    # move codes: 0 none, 1 diag-match, 2 candidate-gap, 3 event-gap, 4 down-match
    bp = np.zeros((m + 1, n + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            best = neg
            move = 0
            # gaps first so a tie prefers NOT matching (the s<=0 case stays unmatched)
            if j > 0:
                v = dp[i, j - 1] + gc
                if v > best:
                    best, move = v, 2
            if i > 0:
                v = dp[i - 1, j] + ge
                if v > best:
                    best, move = v, 3
            if i > 0 and j > 0:
                v = dp[i - 1, j - 1] + sm[i - 1, j - 1]
                if v > best:
                    best, move = v, 1
                v = dp[i - 1, j] + sm[i - 1, j - 1] + r  # down-match: c_j serves e_{i-1} and e_i
                if v > best:
                    best, move = v, 4
            dp[i, j] = best
            bp[i, j] = move
    assign: list[int | None] = [None] * m
    i, j = m, n
    while i > 0 or j > 0:
        move = int(bp[i, j])
        if move == 1:  # diag-match: e_i <-> c_j
            assign[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == 2:  # candidate-gap
            j -= 1
        elif move == 3:  # event-gap
            i -= 1
        elif move == 4:  # down-match: e_i <-> c_j (j unchanged; c_j reused)
            assign[i - 1] = j - 1
            i -= 1
        elif i > 0:  # move == 0 only at an edge; step toward the origin
            i -= 1
        else:
            j -= 1
    return assign


def _fit_frame_time_relationship(
    frames: pd.DataFrame,
) -> dict[tuple, tuple[float, float]]:
    """Per-(game_id, period_id) linear fit ``frame_id ~= slope * time + intercept``.

    fps is constant, so ``frame_id`` is linear in ``time_seconds``. Deriving the
    fit from the frames' own ``(frame_id, time_seconds)`` pairs handles both
    0-based providers (Metrica/StatsBomb, where ``frame_id == time * rate``) and
    native-frame-numbered providers (IDSSE/Sportec, where ``frame_id`` is offset
    from 0 — e.g. period 1 from 10000). Groups lacking >=2 distinct usable
    ``time_seconds`` values are omitted; the caller falls back to
    ``time * frame_rate`` for those.

    Returns
    -------
    dict
        Maps ``(game_id, period_id)`` to ``(slope, intercept)``.
    """
    fits: dict[tuple, tuple[float, float]] = {}
    if "time_seconds" not in frames.columns:
        return fits

    for (gid, pid), grp in frames.groupby(["game_id", "period_id"]):
        pairs = grp[["frame_id", "time_seconds"]].dropna().drop_duplicates()
        if len(pairs) < 2:
            continue
        t = np.asarray(pairs["time_seconds"].values, dtype=np.float64)
        f = np.asarray(pairs["frame_id"].values, dtype=np.float64)
        if float(np.ptp(t)) < 1e-9:
            continue  # degenerate (no time spread) -> caller falls back
        slope, intercept = np.polyfit(t, f, 1)
        if abs(slope) < 1e-9:
            continue
        fits[(gid, pid)] = (float(slope), float(intercept))

    return fits


#: Candidate-window half-margin (s) around an episode's event time span; covers the provider
#: misalignment the sync must correct. Wide enough for the observed offsets, narrow enough to keep
#: each episode's DP small.
_WINDOW_MARGIN_SECONDS = 5.0

_ALIGN_COLUMNS = pd.Index(
    [
        "action_id",
        "elastic_frame_id",
        "elastic_confidence",
        "elastic_error_seconds",
        "elastic_receive_frame_id",
        "elastic_receive_confidence",
        "elastic_receive_error_seconds",
    ]
)


def _empty_alignment() -> pd.DataFrame:
    df = pd.DataFrame(columns=_ALIGN_COLUMNS)
    df["action_id"] = df["action_id"].astype("int64")
    for c in ("elastic_frame_id", "elastic_receive_frame_id"):
        df[c] = df[c].astype("Int64")
    for c in (
        "elastic_confidence",
        "elastic_error_seconds",
        "elastic_receive_confidence",
        "elastic_receive_error_seconds",
    ):
        df[c] = df[c].astype("float64")
    return df


def _build_frame_lookups(frames: pd.DataFrame, *, params: ElasticSyncParams) -> dict[tuple, _FrameLookups]:
    """Per-(canonical game_id, period_id) :class:`_FrameLookups` for ``_score`` (built once/match)."""
    ball = _ball_kinematics(frames, params=params)
    out: dict[tuple, _FrameLookups] = {}
    if ball.empty:
        return out
    ball_mask = frames["is_ball"] == True  # noqa: E712
    pl = frames.loc[~ball_mask, ["game_id", "period_id", "frame_id", "player_id", "team_id", "x", "y"]].copy()
    pl = pl.dropna(subset=["x", "y", "player_id"])
    merged = pl.merge(
        ball[["game_id", "period_id", "frame_id", "ball_x", "ball_y"]],
        on=["game_id", "period_id", "frame_id"],
        how="inner",
    )
    if not merged.empty:
        merged["dist"] = np.hypot(
            merged["x"].to_numpy(dtype=float) - merged["ball_x"].to_numpy(dtype=float),
            merged["y"].to_numpy(dtype=float) - merged["ball_y"].to_numpy(dtype=float),
        )
        merged["pcanon"] = canonical_id_series(merged["player_id"])
        merged["tcanon"] = canonical_id_series(merged["team_id"])
    merged_groups = group_rows(merged, ("game_id", "period_id")) if not merged.empty else None

    for (gid, pid), bgrp in ball.groupby(["game_id", "period_id"], sort=False):
        gcanon = canonical_id(gid)
        pnum = int(pid)  # type: ignore[arg-type]  # groupby key Hashable (pandas-stubs)
        accel = {int(f): float(a) for f, a in zip(bgrp["frame_id"], bgrp["ball_accel"], strict=True)}
        boundary = {int(f): float(b) for f, b in zip(bgrp["frame_id"], bgrp["boundary_dist"], strict=True)}
        dist: dict[tuple, float] = {}
        frame_players: dict[int, list] = {}
        player_team: dict = {}
        mg = merged_groups.get(gid, pid) if merged_groups is not None else None
        if mg is not None and len(mg):
            for f, pc, dd, tcn in zip(
                mg["frame_id"].to_numpy(),
                mg["pcanon"].to_numpy(),
                mg["dist"].to_numpy(),
                mg["tcanon"].to_numpy(),
                strict=True,
            ):
                fi = int(f)
                dist[(fi, pc)] = float(dd)
                frame_players.setdefault(fi, []).append(pc)
                player_team[pc] = tcn
        out[(gcanon, pnum)] = _FrameLookups(accel, dist, frame_players, player_team, boundary)
    return out


def _ensure_type_name(actions: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a ``type_name`` column (deriving from ``type_id`` if needed)."""
    if "type_name" in actions.columns:
        return actions
    if "type_id" in actions.columns:
        id2name = spadlconfig.actiontypes_df().set_index("type_id")["type_name"]
        out = actions.copy()
        out["type_name"] = out["type_id"].map(id2name)
        return out
    raise KeyError("align_events_to_frames requires a 'type_name' or 'type_id' column on actions")


def _partition_episodes(events: list[Event], real_possessions: list) -> list[list[Event]]:
    """Split the enriched sequence into per-possession episodes; a trailing out/goal virtual event
    stays with the episode it terminates (it is emitted before the next possession's first real event)."""
    episodes: list[list[Event]] = []
    cur: list[Event] = []
    cur_poss: object = object()  # sentinel distinct from any real possession_id
    ri = 0
    for ev in events:
        if ev.kind == "real":
            poss = real_possessions[ri]
            ri += 1
            if cur and poss != cur_poss:
                episodes.append(cur)
                cur = []
            cur_poss = poss
        cur.append(ev)
    if cur:
        episodes.append(cur)
    return episodes


def _aligned_error(frame: int, expected_time: float, fit, params: ElasticSyncParams) -> float:
    """|aligned frame time - the action's nominal time| (the provider misalignment the sync corrects)."""
    if fit is not None:
        slope, intercept = fit
        aligned = (frame - intercept) / slope if slope else float(frame) / params.frame_rate
    else:
        aligned = float(frame) / params.frame_rate
    return abs(aligned - expected_time)


def _inplay_segments(frames: pd.DataFrame) -> dict[tuple, np.ndarray]:
    """Per ``(canonical game_id, period_id)`` maximal ``ball_state=="alive"`` run time intervals.

    The paper's alignment episodes are *consecutive in-play frames* (arXiv:2608.30227), NOT
    ``spadl.add_possessions`` possessions. Returns a dict mapping ``(canonical game_id, period_id)``
    to a float ``(k, 2)`` array of ``[time_lo, time_hi]`` for each maximal run of consecutive alive
    frames (sorted by ``frame_id``). A group with no alive frame is omitted; missing ``ball_state``
    or an empty frame set yields ``{}``. ``ball_state`` is frame-level, so the value is taken once
    per ``frame_id``.
    """
    out: dict[tuple, np.ndarray] = {}
    if frames.empty or "ball_state" not in frames.columns:
        return out
    for (gid, pid), grp in frames.groupby(["game_id", "period_id"], sort=False):
        per_frame = grp.drop_duplicates("frame_id").sort_values("frame_id")
        alive = (per_frame["ball_state"] == "alive").to_numpy()
        if not alive.any():
            continue
        times = np.asarray(per_frame["time_seconds"].values, dtype=np.float64)
        pos = np.flatnonzero(alive)
        # split the alive positions into maximal runs of consecutive frames (a dead frame, which is
        # absent from ``pos``, leaves a gap > 1 between adjacent alive positions).
        splits = np.flatnonzero(np.diff(pos) > 1) + 1
        runs = np.split(pos, splits)
        out[(canonical_id(gid), int(pid))] = np.array(  # type: ignore[arg-type]  # groupby key Hashable
            [[times[r[0]], times[r[-1]]] for r in runs], dtype=np.float64
        )
    return out


def _assign_segments(times: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Nearest alive-segment index per time (spec §3.1 in-play episode assignment).

    ``segments`` is a sorted, non-overlapping ``(k, 2)`` array of ``[time_lo, time_hi]``. A time
    inside a segment maps to that segment; a time in a dead span / outside every segment maps to the
    nearest segment by boundary distance. Empty ``segments`` -> all zeros (one implicit episode).
    Nearest-by-time is monotone in a sorted ``times``, so episode ids are non-decreasing along the
    chronological action order the assembler scores on.
    """
    n = len(times)
    if len(segments) == 0:
        return np.zeros(n, dtype=np.int64)
    lo = segments[:, 0][None, :]
    hi = segments[:, 1][None, :]
    t = np.asarray(times, dtype=np.float64)[:, None]
    inside = (t >= lo) & (t <= hi)
    dist = np.where(inside, 0.0, np.minimum(np.abs(t - lo), np.abs(t - hi)))
    return np.argmin(dist, axis=1).astype(np.int64)


def _inplay_episode_ids(actions: pd.DataFrame, frames: pd.DataFrame, *, params: ElasticSyncParams) -> pd.Series:
    """Per-action episode id = the nearest ``ball_state=="alive"`` segment (spec §3.1 / §7).

    Replaces ``add_possessions`` as the episode/partition key: the paper's episodes are consecutive
    in-play frames, and the "control (reception)" virtual event fires for same-*episode*
    different-player actions, so this id feeds BOTH ``_partition_episodes`` and ``_virtual_between``.
    A ``(game, period)`` with no alive segment (or absent from ``frames``) maps all its actions to
    episode 0. The returned Series is index-aligned to ``actions``.
    """
    del params  # episodes come from ball_state directly; accepted for call-site uniformity
    segs = _inplay_segments(frames)
    ids = np.zeros(len(actions), dtype=np.int64)
    if segs and not actions.empty and "time_seconds" in actions.columns:
        # reset to a positional RangeIndex so group -> positions is robust to a non-unique caller
        # index; iterate the groupby directly (the file's idiom -- ``.indices`` types its key as a
        # bare Hashable, which does not unpack).
        a = actions.reset_index(drop=True)
        for (gid, pid), grp in a.groupby(["game_id", "period_id"], sort=False):
            s = segs.get((canonical_id(gid), int(pid)))  # type: ignore[arg-type]  # groupby key Hashable
            if s is None or len(s) == 0:
                continue
            pos = np.asarray(grp.index, dtype=np.intp)
            ids[pos] = _assign_segments(np.asarray(grp["time_seconds"].values, dtype=np.float64), s)
    return pd.Series(ids, index=actions.index, dtype="int64")


def align_events_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams | None = None,
    _candidates: dict[tuple, list] | None = None,
) -> pd.DataFrame:
    """ELASTIC v2 alignment: each action's start (and reception/termination) frame with confidence.

    Global, order-preserving extended-Needleman-Wunsch alignment (Kim et al. 2026) between the
    virtual-termination-enriched SPADL event sequence and sparse ball-touch candidate frames, run
    per possession-episode. Requires continuous tracking with player identity + ``team_id`` (for
    possession); SB360 freeze-frames are not supported.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with ``action_id``, ``game_id``, ``period_id``, ``time_seconds``,
        ``player_id``, ``team_id``, and ``type_id`` or ``type_name``.
    frames : pd.DataFrame
        Long-form tracking frames.
    params : ElasticSyncParams or None
        Algorithm parameters.

    Returns
    -------
    pd.DataFrame
        Columns: ``action_id``, ``elastic_frame_id``, ``elastic_confidence``,
        ``elastic_error_seconds``, ``elastic_receive_frame_id``, ``elastic_receive_confidence``,
        ``elastic_receive_error_seconds``. One row per action that matched a start or a reception;
        unmatched actions are absent (callers left-merge and NaN-fill).

    Examples
    --------
    Align a match's actions to frames (start + reception) via ELASTIC-NW::

        result = align_events_to_frames(actions, frames)
        result.columns.tolist()
        # ['action_id', 'elastic_frame_id', 'elastic_confidence', 'elastic_error_seconds',
        #  'elastic_receive_frame_id', 'elastic_receive_confidence', 'elastic_receive_error_seconds']
    """
    from silly_kicks.tracking.schema import POSITIONAL_ONLY
    from silly_kicks.tracking.utils import validate_velocity_regime

    if params is None:
        params = ElasticSyncParams()
    if actions.empty or frames.empty:
        return _empty_alignment()

    # ELASTIC-NW requires CONTINUOUS tracking: candidate detection needs a ball TRAJECTORY sampled
    # densely over time. Freeze-frame providers (StatsBomb-360) declare themselves
    # velocity-unavailable-BY-DESIGN (the POSITIONAL_ONLY regime) -- a single snapshot per event, no
    # trajectory -- so there is nothing to synchronise. Return an honest empty alignment rather than
    # fabricate one from disconnected snapshots (ADR-063 discipline: a DECLARED-unavailable regime
    # degrades honestly, it does not crash a mixed-provider pipeline). NOTE this gates on
    # velocity-unavailable-BY-DESIGN, not on velocity absence per se: a velocity-less but CONTINUOUS
    # provider is VELOCITY_MISSING (not POSITIONAL_ONLY) and DOES run here, because the engine derives
    # acceleration from positions and never reads vx/vy.
    # on_mismatch="ignore": READ the regime, never raise. A VELOCITY_MISSING frame set (continuous
    # positions, vx/vy not derived) is FINE for elastic (it uses positions, not velocity), so the
    # default raise-on-VELOCITY_MISSING must not fire here; only the declared POSITIONAL_ONLY
    # freeze-frame regime causes a refusal.
    if validate_velocity_regime(frames, on_mismatch="ignore").regime == POSITIONAL_ONLY:
        return _empty_alignment()

    # ``_candidates`` (private) lets a caller that already ran _detect_candidate_frames (e.g.
    # link_actions_to_frames_elastic, which needs them for n_candidate_frames) thread them in to avoid
    # a second full-frame detection pass -- byte-identical to detecting here (same call, same params).
    candidates = _candidates if _candidates is not None else _detect_candidate_frames(frames, params=params)
    if not candidates:
        return _empty_alignment()
    lookups = _build_frame_lookups(frames, params=params)
    fits = _fit_frame_time_relationship(frames)

    # Episodes = consecutive in-play (ball_state=="alive") segments (arXiv:2608.30227), assigned via
    # _inplay_episode_ids and carried on ``possession_id`` so both the episode partition
    # (_partition_episodes) and the same-episode reception rule (_virtual_between) key on it. The
    # copy guards the caller's DataFrame (ADR-033); _ensure_type_name may return the input unchanged.
    work = _ensure_type_name(actions).copy()
    work["possession_id"] = _inplay_episode_ids(work, frames, params=params)
    start: dict[int, tuple] = {}
    receive: dict[int, tuple] = {}
    margin = params.frame_rate * _WINDOW_MARGIN_SECONDS

    for (gid, pid), agrp in work.groupby(["game_id", "period_id"], sort=False):
        gp = (canonical_id(gid), int(pid))  # type: ignore[arg-type]  # groupby key Hashable (pandas-stubs)
        cand_list = candidates.get(gp)
        lk = lookups.get(gp)
        if not cand_list or lk is None:
            continue
        fit = fits.get((gid, pid))
        slope = fit[0] if fit is not None else float(params.frame_rate)
        intercept = fit[1] if fit is not None else 0.0

        agrp = agrp.sort_values(["time_seconds", "action_id"])
        events = _enrich_events(agrp, params=params)
        real_poss = [
            getattr(r, "possession_id", 0)
            for r in agrp.itertuples()
            if _map_category(getattr(r, "type_name", None)) is not None
        ]
        episodes = _partition_episodes(events, real_poss)
        cand_frames = np.array([c.frame_id for c in cand_list], dtype=np.int64)

        for ep in episodes:
            times = [ev.expected_time for ev in ep]
            f_lo = slope * min(times) + intercept - margin
            f_hi = slope * max(times) + intercept + margin
            lo = int(np.searchsorted(cand_frames, f_lo, side="left"))
            hi = int(np.searchsorted(cand_frames, f_hi, side="right"))
            ep_cands = cand_list[lo:hi]
            if not ep_cands:
                continue
            sm = np.array([[_score(ev, c, lk, params=params) for c in ep_cands] for ev in ep], dtype=float)
            if sm.size == 0:
                continue
            assign = _needleman_wunsch(sm, params=params)
            for k, ci in enumerate(assign):
                if ci is None:
                    continue
                conf = float(sm[k, ci])
                if conf < params.min_confidence:
                    continue
                ev = ep[k]
                frame = int(ep_cands[ci].frame_id)
                err = _aligned_error(frame, ev.expected_time, fit, params)
                if ev.kind == "real":
                    start[int(ev.action_id)] = (frame, round(conf, 4), round(err, 4))
                elif k > 0 and ep[k - 1].kind == "real":
                    receive[int(ep[k - 1].action_id)] = (frame, round(conf, 4), round(err, 4))

    if not start and not receive:
        return _empty_alignment()

    rows = []
    for aid in sorted(set(start) | set(receive)):
        sf, sc, se = start.get(aid, (pd.NA, np.nan, np.nan))
        rf, rc, re_ = receive.get(aid, (pd.NA, np.nan, np.nan))
        rows.append(
            {
                "action_id": aid,
                "elastic_frame_id": sf,
                "elastic_confidence": sc,
                "elastic_error_seconds": se,
                "elastic_receive_frame_id": rf,
                "elastic_receive_confidence": rc,
                "elastic_receive_error_seconds": re_,
            }
        )
    df = pd.DataFrame(rows, columns=_ALIGN_COLUMNS)
    df["action_id"] = df["action_id"].astype("int64")
    for c in ("elastic_frame_id", "elastic_receive_frame_id"):
        df[c] = df[c].astype("Int64")
    for c in (
        "elastic_confidence",
        "elastic_error_seconds",
        "elastic_receive_confidence",
        "elastic_receive_error_seconds",
    ):
        df[c] = df[c].astype("float64")
    return df
