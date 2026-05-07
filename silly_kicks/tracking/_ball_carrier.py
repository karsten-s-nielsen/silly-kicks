"""Per-frame ball-carrier inference (TF-5).

Heuristic: composite scoring of distance + velocity-toward-ball with
hysteresis to prevent flickering. Operates on tracking frames (long-form
TRACKING_FRAMES_COLUMNS shape). Returns one row per (game_id, period_id,
frame_id).

See spec: docs/superpowers/specs/2026-05-05-tf5-infer-ball-carrier-design.md
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def infer_ball_carrier(
    frames: pd.DataFrame,
    *,
    tolerance_m: float = 3.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """Per-frame ball-carrier inference with hysteresis.

    For each frame where ``ball_state`` is not ``"dead"`` (NaN/None treated
    as alive), identifies the player most likely carrying the ball via a
    composite score of distance and velocity-toward-ball, with an incumbent
    bonus (``gamma``) to prevent flickering.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_m : float, default 3.0
        Maximum ball-to-player distance for candidacy (meters).
        Carrier-attribution radius, not dribbling-contact threshold.
    beta : float, default 0.5
        Distance advantage per m/s of velocity toward ball (seconds).
    gamma : float, default 1.0
        Hysteresis bonus for incumbent carrier (meters). The incumbent's
        score is reduced by ``gamma``, so a new candidate must be
        ``gamma`` meters better in composite score to take over.
        ``gamma=0`` gives stateless per-frame behaviour.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, period_id, frame_id, ball_carrier_player_id,
        ball_carrier_distance_m, ball_carrier_team_id. One row per unique
        (game_id, period_id, frame_id). Fresh RangeIndex.

    Examples
    --------
    Infer ball carrier for a full match::

        from silly_kicks.tracking import infer_ball_carrier
        carriers = infer_ball_carrier(frames, tolerance_m=3.0, beta=0.5, gamma=1.0)

    References
    ----------
    Bauer & Anzer (2021). "Data-driven detection of counterpressing in
        professional football." Data Mining and Knowledge Discovery.
    Vidal-Codina et al. (2022). "Automatic Event Detection in Football
        Using Tracking Data." Sports Engineering.

    See NOTICE for full bibliographic citations.
    """
    result_cols = [
        "game_id",
        "period_id",
        "frame_id",
        "ball_carrier_player_id",
        "ball_carrier_distance_m",
        "ball_carrier_team_id",
    ]

    if len(frames) == 0:
        return pd.DataFrame(columns=result_cols)

    has_velocity = "vx" in frames.columns and "vy" in frames.columns
    if not has_velocity:
        warnings.warn(
            "vx/vy columns not found; falling back to distance-only carrier "
            "inference. Call derive_velocities() first for velocity-aware scoring.",
            UserWarning,
            stacklevel=2,
        )

    # Detect output dtypes from frames
    pid_dtype = frames["player_id"].dtype
    tid_dtype = frames["team_id"].dtype

    # Split ball and player rows
    ball_mask = frames["is_ball"] == True  # noqa: E712
    ball_rows = frames[ball_mask]
    player_rows = frames[~ball_mask & frames["x"].notna()]

    # Per-frame ball position: mean of non-NaN ball x/y
    ball_pos = (
        ball_rows.groupby(["game_id", "period_id", "frame_id"], dropna=False)
        .agg(bx=("x", "mean"), by=("y", "mean"), bs=("ball_state", "first"))
        .reset_index()
    )

    # Unique frames (from all rows, not just ball rows)
    unique_frames = (
        frames[["game_id", "period_id", "frame_id"]]
        .drop_duplicates()
        .sort_values(["game_id", "period_id", "frame_id"])
        .reset_index(drop=True)
    )

    # Merge ball position onto unique frames
    frame_ball = unique_frames.merge(ball_pos, on=["game_id", "period_id", "frame_id"], how="left")

    # Pre-build grouped dict for O(1) per-frame candidate lookup.
    # Avoids O(n*m) boolean-mask filtering inside the sequential loop.
    _empty_player_df = player_rows.iloc[:0]
    player_groups: dict[tuple, pd.DataFrame] = dict(iter(player_rows.groupby(["game_id", "period_id", "frame_id"])))

    # Process sequentially within each (game_id, period_id) group
    results: list[dict] = []
    groups = frame_ball.groupby(["game_id", "period_id"], dropna=False)

    for (_gid, _pid), group in groups:
        incumbent_pid = None
        group_sorted = group.sort_values("frame_id")

        for _, frow in group_sorted.iterrows():
            gid = frow["game_id"]
            pid_val = frow["period_id"]
            fid = frow["frame_id"]
            bx = frow["bx"]
            by = frow["by"]
            bs = frow["bs"]

            # Dead ball -> NaN, reset incumbent
            if bs == "dead":
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            # No ball position -> NaN, reset incumbent
            if pd.isna(bx) or pd.isna(by):
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            # O(1) candidate lookup via pre-built dict
            cands = player_groups.get((gid, pid_val, fid), _empty_player_df)

            if cands.empty:
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            # Compute distances
            cx = cands["x"].to_numpy(dtype=float)
            cy = cands["y"].to_numpy(dtype=float)
            dx = cx - float(bx)
            dy = cy - float(by)
            dists = np.sqrt(dx * dx + dy * dy)

            # Filter to tolerance
            within = dists <= tolerance_m
            if not within.any():
                incumbent_pid = None
                results.append(_nan_row(gid, pid_val, fid))
                continue

            cand_idx = np.flatnonzero(within)
            cand_dists = dists[cand_idx]
            cand_pids = cands["player_id"].to_numpy()[cand_idx]
            cand_tids = cands["team_id"].to_numpy()[cand_idx]

            # Compute scores
            scores = cand_dists.copy()

            if has_velocity:
                vx_vals = cands["vx"].to_numpy(dtype=float)[cand_idx]
                vy_vals = cands["vy"].to_numpy(dtype=float)[cand_idx]

                # Direction from player to ball
                dir_x = -dx[cand_idx]
                dir_y = -dy[cand_idx]
                dir_norm = np.sqrt(dir_x * dir_x + dir_y * dir_y)
                # Avoid division by zero
                safe_norm = np.where(dir_norm > 0, dir_norm, 1.0)
                unit_x = dir_x / safe_norm
                unit_y = dir_y / safe_norm

                # Velocity toward ball (dot product)
                v_toward = vx_vals * unit_x + vy_vals * unit_y
                # Clamp negative, handle NaN
                v_toward = np.where(np.isnan(v_toward), 0.0, np.maximum(v_toward, 0.0))

                scores = cand_dists - beta * v_toward

            # Apply hysteresis bonus to incumbent
            if incumbent_pid is not None and gamma > 0:
                inc_mask = cand_pids == incumbent_pid
                if inc_mask.any():
                    inc_idx = np.flatnonzero(inc_mask)
                    scores[inc_idx] -= gamma

            # Select best: lowest score, tiebreak by lowest player_id
            best_idx = _select_best(scores, cand_pids)
            winner_pid = cand_pids[best_idx]
            winner_dist = float(cand_dists[best_idx])
            winner_tid = cand_tids[best_idx]

            incumbent_pid = winner_pid
            results.append(
                {
                    "game_id": gid,
                    "period_id": pid_val,
                    "frame_id": fid,
                    "ball_carrier_player_id": winner_pid,
                    "ball_carrier_distance_m": winner_dist,
                    "ball_carrier_team_id": winner_tid,
                }
            )

    if not results:
        return pd.DataFrame(columns=result_cols)

    out = pd.DataFrame(results, columns=result_cols)

    # Preserve dtype for player_id and team_id
    if str(pid_dtype) == "Int64":
        out["ball_carrier_player_id"] = pd.to_numeric(out["ball_carrier_player_id"], errors="coerce").astype("Int64")
    if str(tid_dtype) == "Int64":
        out["ball_carrier_team_id"] = pd.to_numeric(out["ball_carrier_team_id"], errors="coerce").astype("Int64")

    return out


def _nan_row(game_id, period_id, frame_id) -> dict:  # type: ignore[type-arg]
    return {
        "game_id": game_id,
        "period_id": period_id,
        "frame_id": frame_id,
        "ball_carrier_player_id": np.nan,
        "ball_carrier_distance_m": np.nan,
        "ball_carrier_team_id": np.nan,
    }


def _select_best(scores: np.ndarray, pids: np.ndarray) -> int:
    """Index of lowest score; tiebreak by lowest player_id.

    Uses Python-level ``<`` for tiebreak comparison so both int and
    string player_ids (e.g. Sportec DFL-OBJ-*) work safely across
    numpy versions.
    """
    min_score = np.nanmin(scores)
    tied = np.flatnonzero(np.abs(scores - min_score) < 1e-12)
    if len(tied) == 1:
        return int(tied[0])
    # Tiebreak: lowest player_id via Python comparison (safe for
    # both int and object/string dtypes).
    best_idx = tied[0]
    best_pid = pids[tied[0]]
    for i in tied[1:]:
        if pids[i] < best_pid:
            best_idx = i
            best_pid = pids[i]
    return int(best_idx)


def derive_team_in_possession(
    frames: pd.DataFrame,
    carrier: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ball-carrier team into tracking frames as ``team_in_possession``.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    carrier : pd.DataFrame
        Output of :func:`infer_ball_carrier`: must contain ``game_id``,
        ``period_id``, ``frame_id``, ``ball_carrier_team_id``.

    Returns
    -------
    pd.DataFrame
        Copy of ``frames`` with an additional ``team_in_possession`` column.
        Frames with no carrier match get ``NaN``.

    Examples
    --------
    Typical pipeline --- infer carrier, then derive possession::

        from silly_kicks.tracking import infer_ball_carrier, derive_team_in_possession

        carrier = infer_ball_carrier(frames)
        frames_with_poss = derive_team_in_possession(frames, carrier)
    """
    merge_cols = ["game_id", "period_id", "frame_id"]
    carrier_slim = carrier[[*merge_cols, "ball_carrier_team_id"]].copy()
    carrier_slim = carrier_slim.rename(columns={"ball_carrier_team_id": "team_in_possession"})
    return frames.merge(carrier_slim, on=merge_cols, how="left")
