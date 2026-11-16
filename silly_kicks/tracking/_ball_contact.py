"""Ball-contact detection primitive (TF-79, v2: ADR-134; detector and scorer specs of 2026-11-15).

A candidate touch is a real kinematic change in the ball: a local maximum of the change between the mean horizontal
ball velocity just before and just after a sample. The credit step is either the reach gate (``contact_scorer=None``:
the nearest player within ``contact_radius_m``, no fallback radius) or the contact scorer (the K nearest players
scored by :mod:`._contact_scorer` with their control features; after its first-touch decoding, the best one is
credited when its probability reaches ``min_contact_probability``; scorer spec r7).
A measured ball above ``contact_max_z_m`` is never credited; a candidate nobody is credited for is kept as an
``"unresolved"`` row (E7 -- never a guess). Pure (pandas in, pandas out; no global-state mutation; the scorer's
bundled weights are read once, lazily).

The detector is staged, so a sweep or a fit scores exactly the shipped code: :func:`_candidate_peaks` (the
threshold-free kinematic peaks with their kinematics and nearest players), :func:`_merged_peaks` (the change threshold
and the merge) and :func:`_finalize_contacts` (the credit and the output values). Designs:
``docs/superpowers/specs/2026-11-15-tf79-contact-detector-v2-design.md`` and ``...-tf79-contact-scorer-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.tracking.schema import Z_SOURCE_VALUES


@dataclass(frozen=True, kw_only=True)
class BallContactParams:
    """Detection parameters (detector spec section 4). Keyword-only: a v1 positional call raises ``TypeError``.

    ``contact_radius_m`` and ``min_velocity_change_ms`` are chosen on a development half of the GS matches and
    scored once on the held-out half (ADR-124 as amended by ADR-134); every other field is a physics prior.

    Examples
    --------
    A wider reach::

        params = BallContactParams(contact_radius_m=1.5)
        contacts = detect_ball_contacts(frames, params=params)
    """

    contact_radius_m: float = 1.2
    """Reach: the nearest player within this horizontal distance of the ball is credited; no fallback radius."""
    contact_max_z_m: float = 2.6
    """Playable ball height; a measured ball above it is out of reach (TF-48 prior)."""
    aerial_z_m: float = 1.8
    """``is_aerial`` threshold (chest/head boundary); meaningful only where z is measured."""
    min_velocity_change_ms: float = 3.0
    """Candidate threshold: the peak change between the mean velocity before and after the sample (m/s)."""
    min_candidate_speed_ms: float = 2.0
    """Ignore near-dead-ball jitter: the incoming or the outgoing speed must reach this."""
    kinematic_window_s: float = 0.067
    """Half-window of the mean velocities; applied as ``max(kinematic_window_s, 2 * frame_period)``."""
    merge_window_s: float = 0.12
    """Peaks closer than ``max(merge_window_s, 2 * frame_period)`` keep the larger one."""
    dv_scale_ms: float = 5.0
    """Soft scale for folding the velocity change into ``confidence`` (``1 - exp(-|dv| / dv_scale_ms)``)."""
    contact_scorer: str | None = None
    """The credit step: ``None`` is the reach gate; ``"default"`` (the bundled weights) or a directory path is the
    contact scorer. A string keeps the params hashable and free of I/O; the weights load on first use."""
    min_contact_probability: float = 0.5
    """With a scorer: the best option is credited when its probability is at least this."""

    def __post_init__(self) -> None:
        if not 0.0 < self.min_contact_probability < 1.0:
            raise ValueError("min_contact_probability must be in (0, 1)")
        if self.kinematic_window_s <= 0.0:
            raise ValueError("kinematic_window_s must be > 0")
        if self.contact_radius_m <= 0.0:
            raise ValueError("contact_radius_m must be > 0")
        if self.min_velocity_change_ms < 0.0:
            raise ValueError("min_velocity_change_ms must be >= 0")
        if self.merge_window_s <= 0.0:
            raise ValueError("merge_window_s must be > 0")


#: Shared immutable default; a frozen dataclass is safe as a module singleton (avoids a B008 call in
#: a parameter default).
_DEFAULT_PARAMS = BallContactParams()


@dataclass(frozen=True)
class _Candidate:
    """A merged candidate contact (internal)."""

    frame_pos: int  # position into the candidate arrays
    time: float
    source: str  # always "velocity_break" in v2
    magnitude: float


def _merge_clusters(cands, mag, source, t, params: BallContactParams, *, frame_period: float = 0.0) -> list[_Candidate]:
    """Collapse candidate positions that fall within the merge window, keeping the larger change.

    ``cands`` is a list of positions; ``mag``/``source``/``t`` map a position to its velocity change, source token and
    time. ``frame_period`` is the ball-track SAMPLING interval, not the candidate spacing. Candidates within
    ``max(merge_window_s, 2 * frame_period)`` of the previous one join its cluster; the representative is the
    largest change (ties -> the earliest).
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
    best = max(cluster, key=lambda p: (mag[p], -t[p]))
    return _Candidate(frame_pos=best, time=float(t[best]), source=source[best], magnitude=float(mag[best]))


CONTACT_SOURCE_VALUES: tuple[str, ...] = ("rdp", "velocity_break", "both", "unresolved")
"""Closed vocabulary for ``contact_source`` / ``first_contact_source`` (category dtype). v2 emits only
``"velocity_break"`` and ``"unresolved"``; ``"rdp"`` and ``"both"`` stay so the category set does not change."""

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


def _velocity_change_series(bx, by, tt, span_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(dv, speed_in, speed_out, turn)`` at every sample: |v_after - v_before| of the ball's velocity VECTOR, where
    v_before / v_after are the mean velocities over the ``span_s`` seconds before / after the sample (window edges in
    seconds via ``searchsorted``, with a 1e-9 s tolerance so a sample exactly ``span_s`` away is inside), the two mean
    speeds, and the angle between the two velocities (rad, in [0, pi]; 0 when either speed is below 1e-9). NaN where a
    side has no sample or zero duration. A touch changes the velocity vector (accel, decel or turn) well above
    passive drag, which is what separates it from flight."""
    n = len(tt)
    nan = np.full(n, np.nan)
    if n < 3:
        return nan, nan.copy(), nan.copy(), nan.copy()
    idx = np.arange(n)
    lo = np.searchsorted(tt, tt - span_s - 1e-9, side="left")
    hi = np.searchsorted(tt, tt + span_s + 1e-9, side="right") - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_b = tt - tt[lo]
        dt_a = tt[hi] - tt
        ok = (idx - lo >= 1) & (hi - idx >= 1) & (dt_b > 0) & (dt_a > 0)
        vbx, vby = (bx - bx[lo]) / dt_b, (by - by[lo]) / dt_b
        vax, vay = (bx[hi] - bx) / dt_a, (by[hi] - by) / dt_a
    dv = np.where(ok, np.hypot(vax - vbx, vay - vby), np.nan)
    s_in, s_out = np.hypot(vbx, vby), np.hypot(vax, vay)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.clip((vbx * vax + vby * vay) / (s_in * s_out), -1.0, 1.0)
    turn = np.where((s_in < 1e-9) | (s_out < 1e-9), 0.0, np.arccos(cos))
    return dv, np.where(ok, s_in, np.nan), np.where(ok, s_out, np.nan), np.where(ok, turn, np.nan)


def _confidence(
    dv_ms: float, source_weight: float, proximity: float, detection_quality: float, *, dv_scale_ms: float
) -> float:
    """Monotone, bounded ``[0, 1]`` ranking score for a candidate contact.

    Increasing in the velocity-vector change (via ``1 - exp(-|Δv| / dv_scale_ms)``), the source weight, the proximity
    and the detection quality. Weights sum to 1. This RANKS candidates; it never drops one.
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


def _nearest_players(players: pd.DataFrame | None, x: float, y: float, k: int) -> tuple[list, np.ndarray]:
    """The ``k`` nearest players with a finite position, nearest first, as ``(player_ids, horizontal distances)``."""
    if players is None or len(players) == 0:
        return [], np.array([])
    d = np.hypot(players["x"].to_numpy(dtype=np.float64) - x, players["y"].to_numpy(dtype=np.float64) - y)
    idx = np.flatnonzero(np.isfinite(d))
    order = idx[np.argsort(d[idx], kind="stable")][:k]
    return [players["player_id"].iloc[i] for i in order], d[order]


_PEAK_COLUMNS: tuple[str, ...] = (
    "period_id",
    "segment",
    "frame_period",
    "frame_id",
    "prior_frame_id",
    "time_seconds",
    "contact_x",
    "contact_y",
    "z",
    "z_source",
    "dv",
    "speed_in",
    "speed_out",
    "turn",
    "nearest_player_id",
    "nearest_dist",
    "detection_quality",
)


def _option_columns(k_options: int) -> tuple[str, ...]:
    """The per-peak option columns: the ``k_options`` nearest players (id, distance, approach) and the distance of
    the next player beyond them (for the last option's gap)."""
    cols: list[str] = []
    for r in range(k_options):
        cols += [f"opt{r}_player_id", f"opt{r}_dist", f"opt{r}_approach"]
    return (*cols, f"opt{k_options}_dist") if k_options else ()


def _ball_samples(
    seg_df: pd.DataFrame, params: BallContactParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """One period's ball rows (sorted by ``frame_id``) -> ``(frame_id, time, x, y, keep, frame_period, span)``.

    ``keep`` is the ADR-125 dedup: consecutive-duplicate ball positions collapse before differencing (the GS
    sample-and-hold sawtooth), and a run longer than the velocity window is a ball at rest, kept whole (ADR-134), so a
    kick from an exactly static ball has its "before" samples. ``span = max(kinematic_window_s, 2 * frame_period)``.
    The single source of the deduped samples for the peak search and the contact scorer's control features."""
    from silly_kicks.tracking.preprocess._dedup import trajectory_dedup_mask

    fid = seg_df["frame_id"].to_numpy(dtype=np.int64, copy=True)
    tt = seg_df["time_seconds"].to_numpy(dtype=np.float64, copy=True)
    bx = seg_df["x"].to_numpy(dtype=np.float64, copy=True)
    by = seg_df["y"].to_numpy(dtype=np.float64, copy=True)
    fr = seg_df["frame_rate"].to_numpy(dtype=np.float64, copy=True) if "frame_rate" in seg_df else np.array([])
    finite_fr = fr[np.isfinite(fr)] if len(fr) else np.array([])
    if len(finite_fr) and finite_fr[0] > 0:
        frame_period = 1.0 / float(finite_fr[0])
    else:
        dt = np.diff(tt[trajectory_dedup_mask(bx, by)])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        frame_period = float(np.median(dt)) if len(dt) else 0.0
    span = max(params.kinematic_window_s, 2.0 * frame_period)
    keep = trajectory_dedup_mask(bx, by, t=tt, rest_s=span)
    return fid, tt, bx, by, keep, frame_period, span


def _candidate_peaks(
    frames: pd.DataFrame, params: BallContactParams = _DEFAULT_PARAMS, *, k_options: int = 0
) -> pd.DataFrame:
    """The threshold-free kinematic peaks of the ball (detector spec section 4, items 1-2).

    One row per local maximum of the velocity change whose incoming or outgoing speed reaches
    ``min_candidate_speed_ms``, with its change ``dv``, the two mean speeds and the turn, its nearest player and
    distance, and its detection quality. With ``k_options`` > 0 (the contact scorer, scorer spec section 2) each row
    also carries the ``k_options`` nearest players with their distances and their approach (the distance now minus
    the distance at the last frame at or before ``t - window``; 0 when the player has no such frame). It reads
    ``kinematic_window_s`` and ``min_candidate_speed_ms`` only; the change threshold, the merge and the credit are
    :func:`_merged_peaks` and :func:`_finalize_contacts`.
    """
    from silly_kicks._frame_index import group_rows
    from silly_kicks.id_compat import canonical_id

    columns = [*_PEAK_COLUMNS, *_option_columns(k_options)]
    ball = frames.loc[frames["is_ball"].to_numpy(dtype=bool)]
    players = frames.loc[~frames["is_ball"].to_numpy(dtype=bool)]
    rows: list[dict] = []
    if len(ball) == 0:
        return pd.DataFrame(columns=columns)
    ball_by_period = group_rows(ball, "period_id")
    players_by_frame = group_rows(players, ("period_id", "frame_id"))
    for pid in ball_by_period.keys():
        seg_df = ball_by_period.get(pid).sort_values("frame_id")
        fid, tt, bx, by, keep, frame_period, span = _ball_samples(seg_df, params)
        bz = seg_df["z"].to_numpy(dtype=np.float64, copy=True) if "z" in seg_df else np.full(len(seg_df), np.nan)
        # z_source (ADR-125): FAIL-SAFE "absent" when the column is missing -> no height check, honest-NaN is_aerial.
        zsrc = (
            seg_df["z_source"].astype(object).to_numpy()
            if "z_source" in seg_df
            else np.full(len(seg_df), "absent", dtype=object)
        )
        fid_all, tt_all, bx_all, by_all = fid, tt, bx, by
        fid, tt, bx, by, bz, zsrc = fid[keep], tt[keep], bx[keep], by[keep], bz[keep], zsrc[keep]
        finite = np.isfinite(bx) & np.isfinite(by)
        for seg, (lo, hi) in enumerate(_segments(finite)):
            if hi - lo < 3:
                continue
            dv, s_in, s_out, turn = _velocity_change_series(bx[lo:hi], by[lo:hi], tt[lo:hi], span)
            d = np.where(np.isfinite(dv), dv, -np.inf)
            prev = np.r_[-np.inf, d[:-1]]
            nxt = np.r_[d[1:], -np.inf]
            fast = np.fmax(s_in, s_out) >= params.min_candidate_speed_ms
            for k in np.flatnonzero(np.isfinite(dv) & (d >= prev) & (d > nxt) & fast):
                gp = lo + int(k)
                here = players_by_frame.get(pid, int(fid[gp]))
                ids, dists = _nearest_players(here, bx[gp], by[gp], max(k_options + 1, 1))
                j = int(np.searchsorted(tt_all, tt[gp] - span + 1e-9, side="right")) - 1
                prior_fid = int(fid_all[j]) if j >= 0 else -1
                a, b = _time_window(tt, gp, 0.2)
                row = {
                    "period_id": int(pid),
                    "segment": seg,
                    "frame_period": frame_period,
                    "frame_id": int(fid[gp]),
                    "prior_frame_id": prior_fid,
                    "time_seconds": float(tt[gp]),
                    "contact_x": float(bx[gp]),
                    "contact_y": float(by[gp]),
                    "z": float(bz[gp]),
                    "z_source": str(zsrc[gp]),
                    "dv": float(dv[k]),
                    "speed_in": float(s_in[k]),
                    "speed_out": float(s_out[k]),
                    "turn": float(turn[k]),
                    "nearest_player_id": ids[0] if ids else pd.NA,
                    "nearest_dist": float(dists[0]) if ids else float("nan"),
                    "detection_quality": float(finite[a:b].mean()),
                }
                if k_options:
                    prior: dict[str, tuple[float, float]] = {}
                    pbx = pby = float("nan")
                    if j >= 0:
                        pbx, pby = float(bx_all[j]), float(by_all[j])
                        before = players_by_frame.get(pid, prior_fid)
                        if before is not None and len(before):
                            prior = {
                                str(canonical_id(p)): (float(px), float(py))
                                for p, px, py in zip(before["player_id"], before["x"], before["y"], strict=True)
                            }
                    for r in range(k_options):
                        has = r < len(ids)
                        approach = float("nan")
                        if has:
                            xy = prior.get(str(canonical_id(ids[r])))
                            back = float(np.hypot(xy[0] - pbx, xy[1] - pby)) if xy is not None else float("nan")
                            approach = float(dists[r]) - back if np.isfinite(back) else 0.0
                        row[f"opt{r}_player_id"] = ids[r] if has else pd.NA
                        row[f"opt{r}_dist"] = float(dists[r]) if has else float("nan")
                        row[f"opt{r}_approach"] = approach
                    row[f"opt{k_options}_dist"] = float(dists[k_options]) if len(ids) > k_options else float("nan")
                rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _merged_peaks(peaks: pd.DataFrame, params: BallContactParams) -> pd.DataFrame:
    """The change threshold (``dv >= min_velocity_change_ms``), then the merge within each segment (the larger change
    wins; detector spec section 4 item 2). The surviving peak rows, chronological, with a ``peak_id``."""
    p = peaks[peaks["dv"] >= params.min_velocity_change_ms]
    keep: list[pd.DataFrame] = []
    for (_period, _seg), g in p.groupby(["period_id", "segment"], sort=True):
        g = g.reset_index(drop=True)
        mag = dict(enumerate(g["dv"].to_numpy(dtype=np.float64)))
        tmap = dict(enumerate(g["time_seconds"].to_numpy(dtype=np.float64)))
        src = dict.fromkeys(range(len(g)), "velocity_break")
        merged = _merge_clusters(
            list(range(len(g))), mag, src, tmap, params, frame_period=float(g["frame_period"].iloc[0])
        )
        keep.append(g.iloc[[c.frame_pos for c in merged]])
    out = pd.concat(keep, ignore_index=True) if keep else p.iloc[0:0].reset_index(drop=True)
    out = out.sort_values(["period_id", "frame_id"], kind="stable").reset_index(drop=True)
    out.insert(0, "peak_id", np.arange(len(out), dtype=np.int64))
    return out


def _finalize_contacts(
    peaks: pd.DataFrame, params: BallContactParams, *, pid_dtype, scorer=None, frames: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The contacts from the candidate peaks: :func:`_merged_peaks`, then the credit and the output values. With
    ``scorer`` None, the reach gate (detector spec section 4 items 3-4); with a scorer, the scored options with their
    control features from ``frames`` and the first-touch decoding (scorer spec r7 section 2; ``peaks`` must carry the
    option columns). Returns the ``_CONTACT_COLUMNS`` schema."""
    from silly_kicks.id_compat import restore_id_dtype

    merged = _merged_peaks(peaks, params)
    best: dict[int, tuple[object, float, bool]] = {}
    if scorer is not None and len(merged):
        from silly_kicks.tracking._contact_scorer import control_features, decide_options, option_rows

        if frames is None:
            raise ValueError("the contact scorer needs the frames (its control features read them)")
        opts = control_features(option_rows(merged), frames, params)
        if len(opts):
            decided = decide_options(
                opts, scorer.predict_proba(opts), params, decode_window_s=float(scorer.decode_window_s)
            )
            for rec in decided.to_dict("records"):
                best[int(rec["peak_id"])] = (rec["player_id"], float(rec["prob"]), bool(rec["credited"]))
    rows: list[dict] = []
    records = cast("list[dict[str, Any]]", merged.to_dict("records"))
    for rec in records:
        r = SimpleNamespace(**rec)
        measured = r.z_source == "measured" and np.isfinite(r.z)
        playable = not (measured and r.z > params.contact_max_z_m)
        if scorer is None:
            dist = float(r.nearest_dist)
            credited = playable and pd.notna(r.nearest_player_id) and dist <= params.contact_radius_m
            player = r.nearest_player_id
            conf = (
                _confidence(
                    float(r.dv),
                    1.0,
                    1.0 - dist / params.contact_radius_m,
                    float(r.detection_quality),
                    dv_scale_ms=params.dv_scale_ms,
                )
                if credited
                else float("nan")
            )
        else:
            # The one credit gate (decoding, threshold, height) is decide_options, shared with the scorer driver.
            player, prob, credited = best.get(int(r.peak_id), (pd.NA, float("nan"), False))
            conf = prob if credited else float("nan")
        rows.append(
            {
                "period_id": int(r.period_id),
                "frame_id": int(r.frame_id),
                "time_seconds": float(r.time_seconds),
                "player_id": player if credited else pd.NA,
                "contact_x": float(r.contact_x),
                "contact_y": float(r.contact_y),
                "velocity_change_ms": float(r.dv),
                "confidence": conf,
                # ADR-125: is_aerial is honest-NaN unless z is a trustworthy MEASUREMENT.
                "is_aerial": bool(r.z > params.aerial_z_m) if measured else pd.NA,
                "aerial_source": r.z_source,
                "contact_source": "velocity_break" if credited else "unresolved",
            }
        )
    if not rows:
        return _empty_contacts(pd.DataFrame({"player_id": pd.Series([], dtype=pid_dtype)}))
    out = pd.DataFrame(rows, columns=list(_CONTACT_COLUMNS))
    out["player_id"] = restore_id_dtype(out["player_id"].to_numpy(dtype=object), pid_dtype)
    out["is_aerial"] = out["is_aerial"].astype("boolean")
    out["aerial_source"] = pd.Categorical(out["aerial_source"], categories=Z_SOURCE_VALUES)
    out["contact_source"] = pd.Categorical(out["contact_source"], categories=CONTACT_SOURCE_VALUES)
    return out


def detect_ball_contacts(frames: pd.DataFrame, *, params: BallContactParams = _DEFAULT_PARAMS) -> pd.DataFrame:
    """Detect candidate ball contacts from tracking frames (TF-79 primitive, v2).

    Pure: does not mutate ``frames``. Returns one row per contact with columns ``period_id, frame_id, time_seconds,
    player_id, contact_x, contact_y, velocity_change_ms, confidence, is_aerial, aerial_source, contact_source``. A
    kinematic peak nobody is credited for (or a measured ball above ``contact_max_z_m``) is KEPT as an
    ``"unresolved"`` row with ``player_id=<NA>`` and ``confidence=NaN`` (E7 -- never a guess). ``velocity_change_ms``
    is the peak's change. ``confidence`` is the reach gate's ranking score, or with ``contact_scorer`` the scorer's
    probability for the credited player. ``is_aerial`` is ``<NA>`` unless the ball z is measured.

    With ``contact_scorer``, a credit depends on the frames within about 0.7 s of its peak (the 0.4 s control
    windows and the scorer's first-touch decoding window), so a slice of a match can credit differently near the
    slice edges than the whole match. Pass whole matches (scorer spec r7 section 2.1).

    Examples
    --------
    Contacts of a tracked match, then the first one after an action at 12.3 s in period 1::

        contacts = detect_ball_contacts(frames)
        first = first_contact_in_window(contacts, 1, 12.3, 15.0, exclude_player_id=passer_id)
    """
    required = {"is_ball", "period_id", "frame_id", "time_seconds", "x", "y"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"detect_ball_contacts requires frame columns {sorted(missing)}")
    if not frames["is_ball"].to_numpy(dtype=bool).any():
        return _empty_contacts(frames)
    pid_dtype = frames["player_id"].dtype if "player_id" in frames else "Int64"
    if params.contact_scorer is None:
        return _finalize_contacts(_candidate_peaks(frames, params), params, pid_dtype=pid_dtype)
    from silly_kicks.tracking import _contact_scorer as cs

    scorer = cs.resolve_scorer(params.contact_scorer)
    scorer.check_params(params)
    peaks = _candidate_peaks(frames, params, k_options=cs.K_OPTIONS)
    return _finalize_contacts(peaks, params, pid_dtype=pid_dtype, scorer=scorer, frames=frames)


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


def first_contact_in_window(
    contacts: pd.DataFrame, period_id: int, t0: float, t1: float, *, exclude_player_id=None
) -> pd.Series | None:
    """The earliest RESOLVED contact in ``period_id`` with ``t0 <= time_seconds < t1``.

    ``exclude_player_id`` (keyword-only, detector spec section 5) skips the contacts credited to that player, for a
    caller that asks who touched the ball after an action: the actor's own release is not the next contact. The
    compare is dtype-safe (``id_compat``, ADR-019). Returns the contact row (a Series) or ``None``.
    """
    from silly_kicks.id_compat import ids_match

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
    if exclude_player_id is not None and not pd.isna(exclude_player_id):
        mask &= ~ids_match(contacts["player_id"], exclude_player_id).fillna(False).astype(bool)
    sub = contacts.loc[mask]
    if len(sub) == 0:
        return None
    return sub.sort_values("time_seconds", kind="stable").iloc[0]


@nan_safe_enrichment
def add_ball_contacts(
    actions: pd.DataFrame, frames: pd.DataFrame, *, params: BallContactParams = _DEFAULT_PARAMS
) -> pd.DataFrame:
    """Additive action enricher: the first ball contact by another player in each action's window.

    Pure: returns a copy of ``actions`` with the ``first_contact_*`` columns added (all NaN/``<NA>`` where no resolved
    contact falls in the window). The window is ``[action.time_seconds, next_action.time_seconds)`` within a period
    (the last action runs to ``+inf``). When ``actions`` carries ``player_id``, the action's own player is skipped
    (detector spec section 5; dtype-safe, ADR-019). ``frames`` is positional-or-keyword; ``params`` is keyword-only
    (ADR-078). Fails loud if ``actions`` carries no time base. With ``contact_scorer``, pass the whole match's frames:
    a credit reads the frames within about 0.7 s of its peak (see :func:`detect_ball_contacts`).

    Examples
    --------
    The next touch after each action::

        out = add_ball_contacts(actions, frames)
        out[["time_seconds", "first_contact_time", "first_contact_player_id"]]
    """
    from silly_kicks.id_compat import canonical_id, restore_id_dtype

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

    actor = (
        np.array([None if pd.isna(v) else canonical_id(v) for v in out["player_id"]], dtype=object)
        if "player_id" in out.columns
        else np.full(n, None, dtype=object)
    )
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
            c_pid = [canonical_id(v) for v in cper["player_id"]]
            for j, ai in enumerate(a_sorted):
                pos = int(np.searchsorted(c_t, a_t[j], side="left"))
                while pos < len(c_t) and c_t[pos] < a_end[j] and actor[ai] is not None and c_pid[pos] == actor[ai]:
                    pos += 1  # the action's own player is not its next contact
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
