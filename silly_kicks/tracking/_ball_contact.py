"""Ball-contact detection primitive (TF-79).

Candidate touches from a ball trajectory (RDP structural breaks UNION velocity
breaks), z-aware 3D proximity attribution to a player, confidence, aerial flag,
and provenance. Pure (pandas in, pandas out; no I/O, no global-state mutation).

This is the candidate-touch stage only: the Needleman-Wunsch event alignment and
inserted touch events stay in the deferred PathCRF item (that code is never
lifted; RDP here is the classical Ramer-Douglas-Peucker algorithm, implemented
fresh). See the design at
``docs/superpowers/specs/2026-10-14-tf79-ball-contact-detection-design.md`` and
``NOTICE`` for citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.tracking.schema import Z_SOURCE_VALUES


@dataclass(frozen=True)
class BallContactParams:
    """Fixed, pre-registered detection parameters (spec sec. 3).

    Defaults are derived from physics / geometry and the TF-48 contact priors;
    they are NEVER selected against the real accuracy corpus (spec sec. 14
    no-leakage rule), only against the synthetic-recovery fixture.
    """

    rdp_epsilon_m: float = 0.3
    """RDP tolerance: ~2-3x optical ball positional noise (above noise, below a real deflection)."""
    speed_drop_frac: float = 0.40
    """Velocity-break: fractional per-second speed drop."""
    heading_turn_deg: float = 35.0
    """Velocity-break: heading change, degrees."""
    contact_radius_m: float = 2.0
    """Attribution radius (TF-48 prior ``_CONTACT_RADIUS_M``)."""
    contact_exist_radius_m: float = 5.0
    """Looser existence bar (TF-48 prior ``_CONTACT_EXIST_RADIUS_M``)."""
    contact_max_z_m: float = 2.6
    """Playable ball height (TF-48 prior ``_CONTACT_MAX_Z_M``)."""
    aerial_z_m: float = 1.8
    """``is_aerial`` threshold (chest/head boundary); meaningful only where z is measured."""
    merge_window_s: float = 0.12
    """Dedup window; applied as ``max(merge_window_s, 2 * frame_period)``."""
    min_candidate_speed_ms: float = 2.0
    """Ignore near-dead-ball jitter below this incoming speed."""
    dv_scale_ms: float = 5.0
    """Soft scale for folding the contact's velocity-vector change |Δv| into ``confidence``
    (``1 - exp(-|Δv| / dv_scale_ms)``); ~a firm touch. NOT a hard cut."""
    min_velocity_change_ms: float = 0.0
    """OPT-IN hard floor on |Δv| (m/s). Default 0.0 = keep EVERY candidate (no data loss); a
    consumer that wants fewer candidates raises it or thresholds on ``velocity_change_ms``/``confidence``."""

    def __post_init__(self) -> None:
        if self.contact_radius_m > self.contact_exist_radius_m:
            raise ValueError("contact_radius_m must be <= contact_exist_radius_m")
        if not (0.0 < self.speed_drop_frac <= 1.0):
            raise ValueError("speed_drop_frac must be in (0, 1]")
        if self.heading_turn_deg <= 0.0:
            raise ValueError("heading_turn_deg must be > 0")


#: Shared immutable default; a frozen dataclass is safe as a module singleton (avoids a B008 call in
#: a parameter default).
_DEFAULT_PARAMS = BallContactParams()


def _rdp(xy: np.ndarray, epsilon_m: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification.

    Returns a boolean keep-mask over ``xy`` rows: endpoints always kept, and an
    interior point kept iff its perpendicular distance to the current segment
    exceeds ``epsilon_m``. Classical algorithm (Ramer 1972; Douglas & Peucker
    1973), implemented fresh -- no third-party code.

    Parameters
    ----------
    xy:
        ``(n, 2)`` float array of successive points.
    epsilon_m:
        Simplification tolerance in metres.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = True
    keep[-1] = True
    if n <= 2:
        return keep
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a = xy[i0]
        b = xy[i1]
        ab = b - a
        denom = float(np.hypot(ab[0], ab[1]))
        seg = xy[i0 + 1 : i1] - a
        if denom == 0.0:
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / denom
        k = int(np.argmax(dist))
        if dist[k] > epsilon_m:
            idx = i0 + 1 + k
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return keep


def _velocity_breaks(xy: np.ndarray, t: np.ndarray, params: BallContactParams) -> tuple[np.ndarray, np.ndarray]:
    """Kinematic break detection over a single ball segment.

    Returns ``(break_mask, break_magnitude)`` over the ``n`` input points.
    A point fires when the incoming per-second speed drops by at least
    ``speed_drop_frac`` OR the heading turns by at least ``heading_turn_deg``,
    provided the incoming speed is at least ``min_candidate_speed_ms``. Speed is
    computed per second (``dx/dt``), so thresholds are frame-rate independent.
    ``break_magnitude`` is ``max(drop / speed_drop_frac, turn / heading_turn_deg)``
    -- used later as the merged-cluster representative-frame selector and as a
    confidence input.
    """
    xy = np.asarray(xy, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = len(xy)
    mask = np.zeros(n, dtype=bool)
    mag = np.zeros(n, dtype=np.float64)
    if n < 3:
        return mask, mag
    dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(dt > 0, dt, np.nan)
        vel = np.diff(xy, axis=0) / dt[:, None]  # per-second velocity
    speed = np.hypot(vel[:, 0], vel[:, 1])
    for i in range(1, n - 1):
        s0 = speed[i - 1]
        s1 = speed[i]
        if not (np.isfinite(s0) and np.isfinite(s1)):
            continue
        if s0 < params.min_candidate_speed_ms:
            continue
        drop = (s0 - s1) / s0 if s0 > 0 else 0.0
        if s0 > 0 and s1 > 0:
            cos = float(np.clip(np.dot(vel[i - 1], vel[i]) / (s0 * s1), -1.0, 1.0))
            turn_deg = float(np.degrees(np.arccos(cos)))
        else:
            turn_deg = 0.0
        fired_drop = drop >= params.speed_drop_frac
        fired_turn = turn_deg >= params.heading_turn_deg
        if fired_drop or fired_turn:
            mask[i] = True
            mag[i] = max(drop / params.speed_drop_frac, turn_deg / params.heading_turn_deg)
    return mask, mag


@dataclass(frozen=True)
class _Candidate:
    """A merged candidate contact (internal)."""

    frame_pos: int  # position into the segment's ball-row arrays
    time: float
    source: str  # one of {"rdp", "velocity_break", "both"}
    magnitude: float


def _attribute(cand_pos, bx, by, bz, players, params: BallContactParams):
    """Attribute a candidate frame to the nearest playable player.

    ``bx``/``by``/``bz`` are the ball coordinate arrays for the segment and
    ``cand_pos`` indexes the candidate frame. ``players`` is a DataFrame with
    ``player_id``/``x``/``y`` columns for that frame. Returns
    ``(player_id, distance)`` for the nearest player within the attribution
    radius, else ``(pd.NA, nan)``. A ball above ``contact_max_z_m`` (and not NaN)
    is overhead -- not a contact (returns NA).
    """
    z = bz[cand_pos]
    playable = bool(np.isnan(z) or z <= params.contact_max_z_m)
    if not playable or players is None or len(players) == 0:
        return pd.NA, float("nan")
    px = players["x"].to_numpy(dtype=np.float64, copy=True)
    py = players["y"].to_numpy(dtype=np.float64, copy=True)
    d = np.hypot(px - float(bx[cand_pos]), py - float(by[cand_pos]))
    finite = np.isfinite(d)
    if not finite.any():
        return pd.NA, float("nan")
    near = finite & (d <= params.contact_radius_m)
    pool = near if near.any() else (finite & (d <= params.contact_exist_radius_m))
    if not pool.any():
        return pd.NA, float("nan")
    cand_idx = np.where(pool)[0]
    best = cand_idx[int(np.argmin(d[cand_idx]))]
    return players["player_id"].iloc[int(best)], float(d[best])


def _merge_clusters(cands, mag, source, t, params: BallContactParams, *, frame_period: float = 0.0) -> list[_Candidate]:
    """Collapse candidate frames that fall within the merge window.

    ``cands`` is a list of frame positions; ``mag``/``source``/``t`` map a frame
    position to its velocity-break magnitude, source token, and time.
    ``frame_period`` is the ball-track SAMPLING interval (seconds), NOT candidate
    spacing -- candidates are sparse events, so the window must come from the real
    sample rate. Candidates within ``max(merge_window_s, 2 * frame_period)`` of
    each other collapse to one contact whose representative frame is the
    maximum-magnitude frame (ties -> the earliest). A cluster with no
    velocity-break member (all magnitudes 0, i.e. an RDP-only cluster) keeps its
    earliest RDP vertex frame. The cluster source is ``"both"`` when it mixes rdp
    and velocity_break signals (or contains a frame already tagged ``both``), else
    the single source.
    """
    if not cands:
        return []
    ordered = sorted(cands, key=lambda p: t[p])
    threshold = max(params.merge_window_s, 2.0 * float(frame_period))

    out: list[_Candidate] = []
    cluster = [ordered[0]]
    for prev, cur in pairwise(ordered):
        if (t[cur] - t[prev]) <= threshold:
            cluster.append(cur)
        else:
            out.append(_representative(cluster, mag, source, t))
            cluster = [cur]
    out.append(_representative(cluster, mag, source, t))
    return out


def _representative(cluster, mag, source, t) -> _Candidate:
    mags = [mag[p] for p in cluster]
    if any(m > 0 for m in mags):
        # velocity-break present: max magnitude, ties -> earliest
        best = max(cluster, key=lambda p: (mag[p], -t[p]))
    else:
        # RDP-only cluster: earliest vertex
        best = min(cluster, key=lambda p: t[p])
    signals = {source[p] for p in cluster}
    if "both" in signals or {"rdp", "velocity_break"} <= signals:
        src = "both"
    else:
        src = next(iter(signals))
    return _Candidate(frame_pos=best, time=float(t[best]), source=src, magnitude=float(mag[best]))


CONTACT_SOURCE_VALUES: tuple[str, ...] = ("rdp", "velocity_break", "both", "unresolved")
"""Closed vocabulary for ``contact_source`` / ``first_contact_source`` (category dtype)."""

_CONTACT_COLUMNS: tuple[str, ...] = (
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
)


def _time_window(tt: np.ndarray, gp: int, span_s: float) -> tuple[int, int]:
    """``[lo, hi)`` index range of samples within +/- ``span_s`` seconds of ``tt[gp]``.

    Derived from ``tt`` by ``searchsorted``, NOT a fixed frame count: after duplicate-frame
    dedup the sampling is sparse and non-uniform, so a frame-count window would span the wrong
    number of seconds and silently corrupt |dv| / detection-quality. On a dense uniform sampling
    it reduces to the equivalent +/- round(span/frame_period) frames.
    """
    lo = int(np.searchsorted(tt, tt[gp] - span_s, side="left"))
    hi = int(np.searchsorted(tt, tt[gp] + span_s, side="right"))
    return lo, hi


def _velocity_change_at(bx, by, tt, gp: int, span_s: float) -> float:
    """|Δv| (m/s) of the ball's velocity VECTOR across ``gp``, over +/- ``span_s`` SECONDS.

    A touch changes the ball's velocity vector (accel, decel, OR turn) well above passive drag
    (<~1 m/s over 0.1 s); this is the single physical signal that separates a touch from flight
    noise. The window is time-based (``_time_window``), so it spans the intended seconds on the
    sparse post-dedup sampling. NaN if no sample lies within the window on a side, or a side has
    zero duration.
    """
    lo, hi = _time_window(tt, gp, span_s)
    i0, i1 = lo, hi - 1
    if gp - i0 < 1 or i1 - gp < 1:
        return float("nan")
    dt_b = tt[gp] - tt[i0]
    dt_a = tt[i1] - tt[gp]
    if dt_b <= 0 or dt_a <= 0:
        return float("nan")
    vbx, vby = (bx[gp] - bx[i0]) / dt_b, (by[gp] - by[i0]) / dt_b
    vax, vay = (bx[i1] - bx[gp]) / dt_a, (by[i1] - by[gp]) / dt_a
    return float(np.hypot(vax - vbx, vay - vby))


def _confidence(
    dv_ms: float, source_weight: float, proximity: float, detection_quality: float, *, dv_scale_ms: float
) -> float:
    """Monotone, bounded ``[0, 1]`` ranking score for a candidate contact.

    Increasing in the velocity-vector change ``|Δv|`` (via ``1 - exp(-|Δv| / dv_scale_ms)``, so a
    flight-curvature candidate lands near 0 and a real touch near 1), in source agreement
    (``both`` > a single signal), proximity, and detection quality. Weights sum to 1. This RANKS
    candidates (the consumer picks an operating point); it never drops one.
    """
    dv_norm = 1.0 - float(np.exp(-max(dv_ms, 0.0) / dv_scale_ms)) if dv_scale_ms > 0 else 0.0
    sw = min(max(source_weight, 0.0), 1.0)
    pr = min(max(proximity, 0.0), 1.0)
    dq = min(max(detection_quality, 0.0), 1.0)
    return float(min(1.0, 0.45 * dv_norm + 0.20 * sw + 0.20 * pr + 0.15 * dq))


def _empty_contacts(frames: pd.DataFrame) -> pd.DataFrame:
    pid_dtype = frames["player_id"].dtype if "player_id" in frames else "Int64"
    return pd.DataFrame(
        {
            "period_id": pd.Series([], dtype="int64"),
            "frame_id": pd.Series([], dtype="int64"),
            "time_seconds": pd.Series([], dtype="float64"),
            "player_id": pd.Series([], dtype=pid_dtype),
            "contact_x": pd.Series([], dtype="float64"),
            "contact_y": pd.Series([], dtype="float64"),
            "velocity_change_ms": pd.Series([], dtype="float64"),
            "confidence": pd.Series([], dtype="float64"),
            "is_aerial": pd.Series([], dtype="boolean"),
            "aerial_source": pd.Categorical([], categories=Z_SOURCE_VALUES),
            "contact_source": pd.Categorical([], categories=CONTACT_SOURCE_VALUES),
        }
    )


def _segments(finite: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[lo, hi)`` runs where ``finite`` is True (detection gaps split segments)."""
    segs: list[tuple[int, int]] = []
    n = len(finite)
    i = 0
    while i < n:
        if finite[i]:
            j = i
            while j < n and finite[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def detect_ball_contacts(frames: pd.DataFrame, *, params: BallContactParams = _DEFAULT_PARAMS) -> pd.DataFrame:
    """Detect candidate ball contacts from tracking frames (TF-79 primitive).

    Pure: does not mutate ``frames``. Returns one row per candidate contact with
    columns ``period_id, frame_id, time_seconds, player_id, contact_x, contact_y,
    confidence, is_aerial, contact_source``. "Unresolved" candidates (a motion
    break with no playable player in range, or an undetected ball) are KEPT as
    rows with ``player_id=<NA>``, ``confidence=NaN``, ``contact_source="unresolved"``
    (E7 -- never a guess). ``is_aerial`` is ``<NA>`` where ``z`` is NaN.
    """
    from silly_kicks._frame_index import group_rows
    from silly_kicks.id_compat import restore_id_dtype
    from silly_kicks.tracking.preprocess._dedup import trajectory_dedup_mask

    required = {"is_ball", "period_id", "frame_id", "time_seconds", "x", "y"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"detect_ball_contacts requires frame columns {sorted(missing)}")

    ball = frames.loc[frames["is_ball"].to_numpy(dtype=bool)]
    players = frames.loc[~frames["is_ball"].to_numpy(dtype=bool)]
    if len(ball) == 0:
        return _empty_contacts(frames)

    ball_by_period = group_rows(ball, "period_id")
    players_by_frame = group_rows(players, ("period_id", "frame_id"))

    rows: list[dict] = []
    for pid in ball_by_period.keys():
        seg_df = ball_by_period.get(pid).sort_values("frame_id")
        fid = seg_df["frame_id"].to_numpy(dtype=np.int64, copy=True)
        tt = seg_df["time_seconds"].to_numpy(dtype=np.float64, copy=True)
        bx = seg_df["x"].to_numpy(dtype=np.float64, copy=True)
        by = seg_df["y"].to_numpy(dtype=np.float64, copy=True)
        bz = seg_df["z"].to_numpy(dtype=np.float64, copy=True) if "z" in seg_df else np.full(len(seg_df), np.nan)
        # z_source (ADR-125): provenance of the ball z -- "measured" | "estimated" | "absent". FAIL-SAFE
        # default "absent" when the column is missing (a legacy/synthetic caller) -> honest-NaN is_aerial,
        # never a false boolean.
        zsrc = (
            seg_df["z_source"].astype(object).to_numpy()
            if "z_source" in seg_df
            else np.full(len(seg_df), "absent", dtype=object)
        )
        fr = seg_df["frame_rate"].to_numpy(dtype=np.float64, copy=True) if "frame_rate" in seg_df else np.array([])
        # Collapse consecutive-duplicate ball positions before any consecutive-frame differencing:
        # a duplicate frame gives velocity 0, so a run of duplicates makes a 0<->real velocity
        # sawtooth that masks real breaks (the GS under-detection; provider-agnostic, harmless where
        # there are no duplicates). Original frame_ids are preserved -> player attribution unchanged.
        keep = trajectory_dedup_mask(bx, by)
        fid, tt, bx, by, bz, zsrc = fid[keep], tt[keep], bx[keep], by[keep], bz[keep], zsrc[keep]
        finite_fr = fr[np.isfinite(fr)] if len(fr) else np.array([])
        if len(finite_fr) and finite_fr[0] > 0:
            frame_period = 1.0 / float(finite_fr[0])
        else:
            dt = np.diff(tt)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            frame_period = float(np.median(dt)) if len(dt) else 0.0

        finite = np.isfinite(bx) & np.isfinite(by)
        for lo, hi in _segments(finite):
            if hi - lo < 2:
                continue
            sxy = np.column_stack([bx[lo:hi], by[lo:hi]])
            st = tt[lo:hi]
            rdp_keep = _rdp(sxy, params.rdp_epsilon_m)
            rdp_keep[0] = rdp_keep[-1] = False  # endpoints are not contacts
            vmask, vmag = _velocity_breaks(sxy, st, params)

            cand_local = sorted(set(np.where(rdp_keep)[0].tolist()) | set(np.where(vmask)[0].tolist()))
            if not cand_local:
                continue
            mag = {}
            source = {}
            tmap = {}
            for p in cand_local:
                is_rdp = bool(rdp_keep[p])
                is_vel = bool(vmask[p])
                source[p] = "both" if (is_rdp and is_vel) else ("rdp" if is_rdp else "velocity_break")
                mag[p] = float(vmag[p]) if is_vel else 0.0
                tmap[p] = float(st[p])
            merged = _merge_clusters(cand_local, mag, source, tmap, params, frame_period=frame_period)

            for cand in merged:
                gp = lo + cand.frame_pos
                dv = _velocity_change_at(bx, by, tt, gp, 0.12)
                # OPT-IN hard floor (default 0.0 -> never drops; preserves every candidate).
                if np.isfinite(dv) and dv < params.min_velocity_change_ms:
                    continue
                players_here = players_by_frame.get(pid, int(fid[gp]))
                attr_id, dist = _attribute(gp, bx, by, bz, players_here, params)
                z_here = bz[gp]
                # ADR-125: is_aerial is honest-NaN unless z is a trustworthy MEASUREMENT (z_source
                # "measured"); "estimated" (SkillCorner) and "absent" (SB360) -> <NA>.
                z_src_here = str(zsrc[gp])
                if z_src_here != "measured" or np.isnan(z_here):
                    is_aerial = pd.NA
                else:
                    is_aerial = bool(z_here > params.aerial_z_m)
                a, b = _time_window(tt, gp, 0.2)
                detection_quality = float(finite[a:b].mean())
                if pd.isna(attr_id):
                    conf = float("nan")
                    src = "unresolved"
                else:
                    proximity = 1.0 - (dist / params.contact_exist_radius_m)
                    source_weight = 1.0 if cand.source == "both" else 0.5
                    dv_for_conf = dv if np.isfinite(dv) else 0.0
                    conf = _confidence(
                        dv_for_conf, source_weight, proximity, detection_quality, dv_scale_ms=params.dv_scale_ms
                    )
                    src = cand.source
                rows.append(
                    {
                        "period_id": int(pid),
                        "frame_id": int(fid[gp]),
                        "time_seconds": float(tt[gp]),
                        "player_id": attr_id,
                        "contact_x": float(bx[gp]),
                        "contact_y": float(by[gp]),
                        "velocity_change_ms": float(dv) if np.isfinite(dv) else float("nan"),
                        "confidence": conf,
                        "is_aerial": is_aerial,
                        "aerial_source": z_src_here,
                        "contact_source": src,
                    }
                )

    if not rows:
        return _empty_contacts(frames)

    out = pd.DataFrame(rows, columns=list(_CONTACT_COLUMNS))
    out = out.sort_values(["period_id", "frame_id"], kind="stable").reset_index(drop=True)
    out["player_id"] = restore_id_dtype(out["player_id"].to_numpy(dtype=object), frames["player_id"].dtype)
    out["is_aerial"] = out["is_aerial"].astype("boolean")
    out["aerial_source"] = pd.Categorical(out["aerial_source"], categories=Z_SOURCE_VALUES)
    out["contact_source"] = pd.Categorical(out["contact_source"], categories=CONTACT_SOURCE_VALUES)
    return out


_FIRST_CONTACT_COLUMNS: tuple[str, ...] = (
    "first_contact_time",
    "first_contact_player_id",
    "first_contact_x",
    "first_contact_y",
    "first_contact_velocity_change_ms",
    "first_contact_confidence",
    "first_contact_is_aerial",
    "first_contact_aerial_source",
    "first_contact_source",
)


def first_contact_in_window(contacts: pd.DataFrame, period_id: int, t0: float, t1: float) -> pd.Series | None:
    """The earliest RESOLVED contact in ``period_id`` with ``t0 <= time_seconds < t1``.

    Returns the contact row (a Series) or ``None`` when no resolved contact falls in
    the window. The low-level query the TF-46 corner logic calls directly.
    """
    if contacts is None or len(contacts) == 0:
        return None
    src = contacts["contact_source"].astype(object)
    mask = (
        (contacts["period_id"] == period_id)
        & (src != "unresolved")
        & contacts["player_id"].notna()
        & (contacts["time_seconds"] >= t0)
        & (contacts["time_seconds"] < t1)
    )
    sub = contacts.loc[mask]
    if len(sub) == 0:
        return None
    return sub.sort_values("time_seconds", kind="stable").iloc[0]


@nan_safe_enrichment
def add_ball_contacts(
    actions: pd.DataFrame, frames: pd.DataFrame, *, params: BallContactParams = _DEFAULT_PARAMS
) -> pd.DataFrame:
    """Additive action enricher: the first ball contact in each action's window.

    Pure: returns a copy of ``actions`` with the ``first_contact_*`` columns added
    (all NaN/``<NA>`` where no resolved contact falls in the window). The window is
    ``[action.time_seconds, next_action.time_seconds)`` within a period (the last
    action runs to ``+inf``). ``frames`` is positional-or-keyword; ``params`` is
    keyword-only (ADR-078). Fails loud if ``actions`` carries no time base.
    """
    from silly_kicks.id_compat import restore_id_dtype

    if "period_id" not in actions.columns:
        raise ValueError("add_ball_contacts requires 'period_id' in actions")
    if "time_seconds" in actions.columns:
        tcol = "time_seconds"
    elif "start_time" in actions.columns:
        tcol = "start_time"
    else:
        raise ValueError("add_ball_contacts requires a time base in actions: 'time_seconds' or 'start_time'")

    contacts = detect_ball_contacts(frames, params=params)
    out = actions.copy()
    n = len(out)
    ft = np.full(n, np.nan)
    fx = np.full(n, np.nan)
    fy = np.full(n, np.nan)
    fdv = np.full(n, np.nan)
    fconf = np.full(n, np.nan)
    fpid = np.empty(n, dtype=object)
    fpid[:] = pd.NA
    faer = np.empty(n, dtype=object)
    faer[:] = pd.NA
    faersrc = np.empty(n, dtype=object)
    faersrc[:] = pd.NA
    fsrc = np.empty(n, dtype=object)
    fsrc[:] = pd.NA

    if n and len(contacts):
        src = contacts["contact_source"].astype(object)
        res = contacts.loc[(src != "unresolved") & contacts["player_id"].notna()]
        tvals = out[tcol].to_numpy(dtype=np.float64)
        pvals = out["period_id"].to_numpy()
        for period in pd.unique(pvals):
            a_idx = np.where(pvals == period)[0]
            if len(a_idx) == 0:
                continue
            a_sorted = a_idx[np.argsort(tvals[a_idx], kind="stable")]
            a_t = tvals[a_sorted]
            a_end = np.append(a_t[1:], np.inf)
            cper = res.loc[res["period_id"] == period].sort_values("time_seconds", kind="stable")
            if len(cper) == 0:
                continue
            c_t = cper["time_seconds"].to_numpy(dtype=np.float64)
            for j, ai in enumerate(a_sorted):
                pos = np.searchsorted(c_t, a_t[j], side="left")
                if pos < len(c_t) and c_t[pos] < a_end[j]:
                    row = cper.iloc[pos]
                    ft[ai] = row["time_seconds"]
                    fx[ai] = row["contact_x"]
                    fy[ai] = row["contact_y"]
                    fdv[ai] = row["velocity_change_ms"]
                    fconf[ai] = row["confidence"]
                    fpid[ai] = row["player_id"]
                    faer[ai] = row["is_aerial"]
                    faersrc[ai] = row["aerial_source"]
                    fsrc[ai] = row["contact_source"]

    pid_dtype = contacts["player_id"].dtype if len(contacts) else "Int64"
    out["first_contact_time"] = ft
    out["first_contact_player_id"] = restore_id_dtype(fpid, pid_dtype)
    out["first_contact_x"] = fx
    out["first_contact_y"] = fy
    out["first_contact_velocity_change_ms"] = fdv
    out["first_contact_confidence"] = fconf
    out["first_contact_is_aerial"] = pd.array(faer, dtype="boolean")
    out["first_contact_aerial_source"] = pd.Categorical(faersrc, categories=Z_SOURCE_VALUES)
    out["first_contact_source"] = pd.Categorical(fsrc, categories=CONTACT_SOURCE_VALUES)
    return out
