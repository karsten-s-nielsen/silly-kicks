"""ELASTIC event-tracking synchronization (Kim et al. 2025).

Aligns event data with tracking frames using ball acceleration spikes
and player-ball proximity features.

Standalone refinement pass -- does NOT replace ``link_actions_to_frames``.
Produces alternative ``(action_id, frame_id, confidence)`` pointers that
callers can substitute.

See NOTICE for full bibliographic citations.

References
----------
Kim, H., Kim, J., & Kim, H. (2025). "ELASTIC: Event-Level Alignment of
STreaming data Including Coordinates." arXiv:2508.09238. ECML/PKDD MLSA
2025.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ElasticSyncParams:
    """Parameters for the ELASTIC event-tracking sync algorithm.

    Examples
    --------
    >>> params = ElasticSyncParams()
    >>> params.window_seconds
    1.0
    """

    window_seconds: float = 1.0
    accel_weight: float = 0.6
    proximity_weight: float = 0.4
    min_confidence: float = 0.1
    frame_rate: int = 25


def _col_f64(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract column as float64 numpy array."""
    return np.asarray(df[col].values, dtype=np.float64)


def extract_ball_features(
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams | None = None,
) -> pd.DataFrame:
    """Extract ball speed + acceleration per (game_id, period_id, frame_id).

    Filters ball rows from the tracking DataFrame, computes velocity via
    finite differences, and derives speed and acceleration.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS).
    params : ElasticSyncParams or None
        For frame_rate. None uses defaults.

    Returns
    -------
    pd.DataFrame
        Columns: ``game_id``, ``period_id``, ``frame_id``, ``ball_x``,
        ``ball_y``, ``ball_speed``, ``ball_accel``.

    Examples
    --------
    >>> bf = extract_ball_features(frames)
    >>> bf.columns.tolist()
    ['game_id', 'period_id', 'frame_id', 'ball_x', 'ball_y', 'ball_speed', 'ball_accel']
    """
    if params is None:
        params = ElasticSyncParams()

    out_cols = pd.Index(
        [
            "game_id",
            "period_id",
            "frame_id",
            "ball_x",
            "ball_y",
            "ball_speed",
            "ball_accel",
        ]
    )

    if frames.empty:
        return pd.DataFrame(columns=out_cols)

    # Extract ball rows
    ball_mask = frames["is_ball"] == True  # noqa: E712
    ball_df = frames.loc[ball_mask, ["game_id", "period_id", "frame_id", "x", "y"]].copy()
    ball_df = ball_df.rename(columns={"x": "ball_x", "y": "ball_y"})
    ball_df = ball_df.dropna(subset=["ball_x", "ball_y"])
    ball_df = ball_df.drop_duplicates(subset=["game_id", "period_id", "frame_id"])
    ball_df = ball_df.sort_values(["game_id", "period_id", "frame_id"]).reset_index(drop=True)

    if ball_df.empty:
        return pd.DataFrame(columns=out_cols)

    dt = 1.0 / params.frame_rate

    bx = _col_f64(ball_df, "ball_x")
    by = _col_f64(ball_df, "ball_y")
    game_ids = np.asarray(ball_df["game_id"].values)
    period_ids = np.asarray(ball_df["period_id"].values)

    vx = np.zeros_like(bx)
    vy = np.zeros_like(by)

    # Same (game_id, period_id) consecutive frames
    same_group = (game_ids[1:] == game_ids[:-1]) & (period_ids[1:] == period_ids[:-1])
    vx[1:] = np.where(same_group, (bx[1:] - bx[:-1]) / dt, 0.0)
    vy[1:] = np.where(same_group, (by[1:] - by[:-1]) / dt, 0.0)

    speed = np.sqrt(vx**2 + vy**2)
    accel = np.zeros_like(speed)
    accel[1:] = np.where(same_group, np.abs(speed[1:] - speed[:-1]) / dt, 0.0)

    result = ball_df[["game_id", "period_id", "frame_id"]].copy()
    result["ball_x"] = bx
    result["ball_y"] = by
    result["ball_speed"] = speed
    result["ball_accel"] = accel

    return result


def _build_player_ball_distance_lookup(
    frames: pd.DataFrame,
) -> dict[tuple, float]:
    """Pre-compute player-ball distances for all (game_id, period_id, frame_id, player_id).

    Extracts ball position per frame from ball rows, joins to player rows,
    and computes Euclidean distances.

    Returns dict mapping ``(game_id, period_id, frame_id, player_id)`` to distance.
    """
    if frames.empty:
        return {}

    # Get ball positions per frame
    ball_mask = frames["is_ball"] == True  # noqa: E712
    ball_df = frames.loc[ball_mask, ["game_id", "period_id", "frame_id", "x", "y"]].copy()
    ball_df = ball_df.rename(columns={"x": "ball_x", "y": "ball_y"})
    ball_df = ball_df.drop_duplicates(subset=["game_id", "period_id", "frame_id"])

    # Get player rows
    player_mask = ~ball_mask
    player_df = frames.loc[player_mask, ["game_id", "period_id", "frame_id", "player_id", "x", "y"]].copy()

    if player_df.empty or ball_df.empty:
        return {}

    # Merge ball position onto player rows
    merged = player_df.merge(
        ball_df[["game_id", "period_id", "frame_id", "ball_x", "ball_y"]],
        on=["game_id", "period_id", "frame_id"],
        how="left",
    )

    px = _col_f64(merged, "x")
    py = _col_f64(merged, "y")
    bx = _col_f64(merged, "ball_x")
    by = _col_f64(merged, "ball_y")

    dist = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
    ball_missing = np.isnan(bx) | np.isnan(by)
    dist[ball_missing] = np.inf

    lookup: dict[tuple, float] = {}
    for i in range(len(dist)):
        key = (
            merged.iloc[i]["game_id"],
            merged.iloc[i]["period_id"],
            int(merged.iloc[i]["frame_id"]),
            str(merged.iloc[i]["player_id"]),
        )
        lookup[key] = float(dist[i])

    return lookup


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


def align_events_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams | None = None,
) -> pd.DataFrame:
    """ELASTIC alignment: action_id -> frame_id with confidence.

    For each action, searches a time window in tracking data, scores
    candidate frames by ball acceleration and player-ball proximity,
    and returns the best match.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with ``action_id``, ``time_seconds``, ``period_id``,
        ``player_id``, ``game_id``.
    frames : pd.DataFrame
        Long-form tracking frames.
    params : ElasticSyncParams or None
        Algorithm parameters.

    Returns
    -------
    pd.DataFrame
        Columns: ``action_id``, ``elastic_frame_id``,
        ``elastic_confidence``, ``elastic_error_seconds``.

    Examples
    --------
    >>> result = align_events_to_frames(actions, frames)
    >>> result.columns.tolist()
    ['action_id', 'elastic_frame_id', 'elastic_confidence', 'elastic_error_seconds']
    """
    if params is None:
        params = ElasticSyncParams()

    result_cols = pd.Index(
        [
            "action_id",
            "elastic_frame_id",
            "elastic_confidence",
            "elastic_error_seconds",
        ]
    )

    if actions.empty or frames.empty:
        return pd.DataFrame(columns=result_cols)

    # Pre-compute ball features
    ball_features = extract_ball_features(frames, params=params)
    if ball_features.empty:
        return pd.DataFrame(columns=result_cols)

    # Build acceleration lookup: (game_id, period_id, frame_id) -> accel
    accel_lookup: dict[tuple, float] = {}
    for _, row in ball_features.iterrows():
        key = (row["game_id"], row["period_id"], int(row["frame_id"]))
        accel_lookup[key] = float(row["ball_accel"])

    # Build distance lookup
    distance_lookup = _build_player_ball_distance_lookup(frames)

    # Build per-(game_id, period_id) sorted frame arrays
    frames_by_group: dict[tuple, np.ndarray] = {}
    for (gid, pid), grp in ball_features.groupby(["game_id", "period_id"]):
        frames_by_group[(gid, pid)] = np.sort(np.asarray(grp["frame_id"].values, dtype=np.int64))

    # Per-(game_id, period_id) frame_id <-> time_seconds fit, so frame windows
    # and frame->time conversions work for native-numbered providers
    # (IDSSE/Sportec) as well as 0-based ones. Groups absent here fall back to
    # the time * frame_rate assumption below.
    frame_time_fits = _fit_frame_time_relationship(frames)

    window_frames = int(params.window_seconds * params.frame_rate)
    results: list[dict] = []

    for _, action_row in actions.iterrows():
        action_id = action_row["action_id"]
        action_time = float(action_row["time_seconds"])
        period_id = action_row["period_id"]
        game_id = action_row["game_id"]
        player_id = str(action_row["player_id"])

        fit = frame_time_fits.get((game_id, period_id))
        if fit is not None:
            slope, intercept = fit
        else:
            slope, intercept = float(params.frame_rate), 0.0
        nominal_frame = round(slope * action_time + intercept)

        period_frames = frames_by_group.get((game_id, period_id))
        if period_frames is None or len(period_frames) == 0:
            continue

        frame_min = nominal_frame - window_frames
        frame_max = nominal_frame + window_frames

        idx_lo = int(np.searchsorted(period_frames, frame_min, side="left"))
        idx_hi = int(np.searchsorted(period_frames, frame_max, side="right"))
        candidate_frames = period_frames[idx_lo:idx_hi]

        if len(candidate_frames) == 0:
            continue

        # Collect acceleration scores
        accels = np.array([accel_lookup.get((game_id, period_id, int(f)), 0.0) for f in candidate_frames])
        max_accel = float(np.max(accels)) if len(accels) > 0 else 1.0
        if max_accel < 1e-9:
            max_accel = 1.0

        best_frame = int(candidate_frames[0])
        best_score = -1.0

        for i, frame_val in enumerate(candidate_frames):
            frame_int = int(frame_val)

            accel_score = accels[i] / max_accel

            dist = distance_lookup.get(
                (game_id, period_id, frame_int, player_id),
                float("inf"),
            )
            proximity_score = 1.0 / (1.0 + dist) if dist < float("inf") else 0.0

            score = params.accel_weight * accel_score + params.proximity_weight * proximity_score

            if score > best_score:
                best_score = score
                best_frame = frame_int

        max_possible = params.accel_weight + params.proximity_weight
        confidence = min(1.0, max(0.0, best_score / max_possible)) if max_possible > 0 else 0.0

        if confidence < params.min_confidence:
            continue

        aligned_ts = (best_frame - intercept) / slope
        error_seconds = abs(aligned_ts - action_time)

        results.append(
            {
                "action_id": action_id,
                "elastic_frame_id": best_frame,
                "elastic_confidence": round(confidence, 4),
                "elastic_error_seconds": round(error_seconds, 4),
            }
        )

    if not results:
        return pd.DataFrame(columns=result_cols)

    return pd.DataFrame(results, columns=result_cols)
