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

try:
    from ._ball_carrier_numba import _carrier_loop_numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _pre_index_frames(
    frames: pd.DataFrame,
) -> dict:
    """Convert long-form tracking frames to dense numpy arrays for kernel consumption.

    Returns a dict with keys:
        bx, by, ball_dead: (n_frames,) arrays
        px, py, pvx, pvy: (n_frames, max_players) arrays, NaN-padded
        player_slots: (n_frames, max_players) int64, -1 for empty
        n_valid: (n_frames,) int64
        seg_starts, seg_ends: (n_segments,) int64
        frame_meta: (n_frames, 3) — game_id, period_id, frame_id per frame
        slot_to_pid: list — inverse mapping (slot → player_id)
        slot_to_team_id: list — player_slot → team_id direct lookup (O(1) post-process)
        pid_to_slot: dict — forward mapping
        pid_dtype, tid_dtype: dtype — for output casting
        has_velocity: bool
    """
    has_velocity = "vx" in frames.columns and "vy" in frames.columns

    ball_mask = frames["is_ball"] == True  # noqa: E712
    ball_rows = frames[ball_mask]
    player_rows = frames[~ball_mask & frames["x"].notna()]

    # Player/team ID ↔ slot mappings (sorted for deterministic tiebreak)
    unique_pids = sorted(player_rows["player_id"].unique())
    pid_to_slot = {pid: i for i, pid in enumerate(unique_pids)}
    slot_to_pid = list(unique_pids)

    # Direct player_slot -> team_id lookup (O(1) in post-process).
    _pid_tid = (
        player_rows[["player_id", "team_id"]]
        .drop_duplicates(subset=["player_id"])
        .set_index("player_id")["team_id"]
        .to_dict()
    )
    slot_to_team_id = [_pid_tid.get(pid) for pid in unique_pids]

    # Ball position per frame
    ball_pos = (
        ball_rows.groupby(["game_id", "period_id", "frame_id"], dropna=False)
        .agg(bx=("x", "mean"), by=("y", "mean"), bs=("ball_state", "first"))
        .reset_index()
    )

    # Unique frames sorted for stable ordering
    unique_frames = (
        frames[["game_id", "period_id", "frame_id"]]
        .drop_duplicates()
        .sort_values(["game_id", "period_id", "frame_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    n_frames = len(unique_frames)

    frame_ball = unique_frames.merge(ball_pos, on=["game_id", "period_id", "frame_id"], how="left")

    # Build frame index for O(1) lookup: (game_id, period_id, frame_id) → row index
    frame_to_idx: dict[tuple, int] = {}
    for i, row in enumerate(unique_frames.itertuples(index=False)):
        frame_to_idx[(row.game_id, row.period_id, row.frame_id)] = i

    # Per-frame ball arrays
    bx_arr = frame_ball["bx"].to_numpy(dtype=np.float64)
    by_arr = frame_ball["by"].to_numpy(dtype=np.float64)
    bs_arr = frame_ball["bs"].to_numpy()
    ball_dead = np.array(
        [(bs == "dead") or np.isnan(bx_arr[i]) or np.isnan(by_arr[i]) for i, bs in enumerate(bs_arr)],
        dtype=np.bool_,
    )

    # Player groups
    player_groups = dict(iter(player_rows.groupby(["game_id", "period_id", "frame_id"])))
    max_players = max((len(g) for g in player_groups.values()), default=0)
    if max_players == 0:
        max_players = 1  # avoid zero-width arrays

    # Dense player arrays
    px = np.full((n_frames, max_players), np.nan)
    py = np.full((n_frames, max_players), np.nan)
    pvx = np.full((n_frames, max_players), np.nan)
    pvy = np.full((n_frames, max_players), np.nan)
    player_slot_arr = np.full((n_frames, max_players), -1, dtype=np.int64)
    n_valid = np.zeros(n_frames, dtype=np.int64)

    for key, group in player_groups.items():
        f_idx = frame_to_idx.get(key)
        if f_idx is None:
            continue
        n = min(len(group), max_players)
        n_valid[f_idx] = n
        px[f_idx, :n] = group["x"].to_numpy(dtype=np.float64)[:n]
        py[f_idx, :n] = group["y"].to_numpy(dtype=np.float64)[:n]
        if has_velocity:
            pvx[f_idx, :n] = group["vx"].to_numpy(dtype=np.float64)[:n]
            pvy[f_idx, :n] = group["vy"].to_numpy(dtype=np.float64)[:n]
        else:
            pvx[f_idx, :n] = 0.0
            pvy[f_idx, :n] = 0.0
        pids = group["player_id"].to_numpy()
        for j in range(n):
            player_slot_arr[f_idx, j] = pid_to_slot.get(pids[j], -1)

    # Segment boundaries: contiguous ranges per (game_id, period_id)
    seg_groups = unique_frames.groupby(["game_id", "period_id"], dropna=False, sort=True)
    seg_starts_list = []
    seg_ends_list = []
    for _, seg_idx in seg_groups.groups.items():
        idx_arr = np.asarray(sorted(seg_idx), dtype=np.int64)
        seg_starts_list.append(int(idx_arr[0]))
        seg_ends_list.append(int(idx_arr[-1]) + 1)
    seg_starts = np.array(seg_starts_list, dtype=np.int64)
    seg_ends = np.array(seg_ends_list, dtype=np.int64)

    # Frame metadata for post-processing
    frame_meta_gid = unique_frames["game_id"].to_numpy()
    frame_meta_pid = unique_frames["period_id"].to_numpy()
    frame_meta_fid = unique_frames["frame_id"].to_numpy()

    return dict(
        bx=bx_arr,
        by=by_arr,
        ball_dead=ball_dead,
        px=px,
        py=py,
        pvx=pvx,
        pvy=pvy,
        player_slots=player_slot_arr,
        n_valid=n_valid,
        seg_starts=seg_starts,
        seg_ends=seg_ends,
        frame_meta_gid=frame_meta_gid,
        frame_meta_pid=frame_meta_pid,
        frame_meta_fid=frame_meta_fid,
        slot_to_pid=slot_to_pid,
        slot_to_team_id=slot_to_team_id,
        pid_to_slot=pid_to_slot,
        pid_dtype=frames["player_id"].dtype,
        tid_dtype=frames["team_id"].dtype,
        has_velocity=has_velocity,
    )


def _carrier_loop_numpy(
    bx: np.ndarray,
    by: np.ndarray,
    ball_dead: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pvx: np.ndarray,
    pvy: np.ndarray,
    player_slots: np.ndarray,
    n_valid: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    tolerance_m: float,
    beta: float,
    gamma: float,
    has_velocity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Python fallback for carrier loop — identical logic to numba kernel."""
    n_frames = len(bx)
    winner_slot = np.full(n_frames, -1, dtype=np.int64)
    winner_dist = np.full(n_frames, np.nan)
    n_segments = len(seg_starts)

    for s in range(n_segments):
        incumbent = -1
        for f in range(seg_starts[s], seg_ends[s]):
            if ball_dead[f]:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            nv = n_valid[f]
            if nv == 0:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            # Compute distances for valid players
            dists = np.empty(nv)
            for i in range(nv):
                dx = px[f, i] - bx[f]
                dy = py[f, i] - by[f]
                dists[i] = np.sqrt(dx * dx + dy * dy)

            # Filter to tolerance
            within_mask = dists <= tolerance_m
            if not within_mask.any():
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            # Build candidate arrays
            cand_indices = np.flatnonzero(within_mask)
            cand_dists = dists[cand_indices]
            scores = cand_dists.copy()

            if has_velocity:
                for ci, i in enumerate(cand_indices):
                    dx = px[f, i] - bx[f]
                    dy = py[f, i] - by[f]
                    d = dists[i]
                    if d > 0:
                        ux = -dx / d
                        uy = -dy / d
                    else:
                        ux = 0.0
                        uy = 0.0
                    vx_val = pvx[f, i]
                    vy_val = pvy[f, i]
                    if np.isnan(vx_val) or np.isnan(vy_val):
                        v_toward = 0.0
                    else:
                        v_toward = vx_val * ux + vy_val * uy
                        if v_toward < 0:
                            v_toward = 0.0
                    scores[ci] = cand_dists[ci] - beta * v_toward

            # Hysteresis
            if incumbent >= 0 and gamma > 0:
                for ci, i in enumerate(cand_indices):
                    if player_slots[f, i] == incumbent:
                        scores[ci] -= gamma
                        break

            # Select best: lowest score, tiebreak by lowest slot
            min_score = scores[0]
            best_ci = 0
            for ci in range(1, len(scores)):
                if scores[ci] < min_score - 1e-12:
                    min_score = scores[ci]
                    best_ci = ci
                elif abs(scores[ci] - min_score) < 1e-12:
                    if player_slots[f, cand_indices[ci]] < player_slots[f, cand_indices[best_ci]]:
                        best_ci = ci
                        min_score = scores[ci]

            best_i = cand_indices[best_ci]
            winner_slot[f] = player_slots[f, best_i]
            winner_dist[f] = cand_dists[best_ci]
            incumbent = player_slots[f, best_i]

    return winner_slot, winner_dist


def _post_process(
    winner_slot: np.ndarray,
    winner_dist: np.ndarray,
    pre: dict,
) -> pd.DataFrame:
    """Map kernel output back to a DataFrame with player_id/team_id."""
    result_cols = [
        "game_id",
        "period_id",
        "frame_id",
        "ball_carrier_player_id",
        "ball_carrier_distance_m",
        "ball_carrier_team_id",
    ]
    n = len(winner_slot)
    slot_to_pid = pre["slot_to_pid"]
    slot_to_team_id = pre["slot_to_team_id"]
    pid_dtype = pre["pid_dtype"]
    tid_dtype = pre["tid_dtype"]

    carrier_pids = np.empty(n, dtype=object)
    carrier_tids = np.empty(n, dtype=object)
    for i in range(n):
        ws = winner_slot[i]
        if ws < 0:
            carrier_pids[i] = np.nan
            carrier_tids[i] = np.nan
        else:
            carrier_pids[i] = slot_to_pid[ws]
            tid = slot_to_team_id[ws]
            carrier_tids[i] = tid if tid is not None else np.nan

    out = pd.DataFrame(
        {
            "game_id": pre["frame_meta_gid"],
            "period_id": pre["frame_meta_pid"],
            "frame_id": pre["frame_meta_fid"],
            "ball_carrier_player_id": carrier_pids,
            "ball_carrier_distance_m": winner_dist,
            "ball_carrier_team_id": carrier_tids,
        },
        columns=result_cols,
    )

    if str(pid_dtype) == "Int64":
        out["ball_carrier_player_id"] = pd.to_numeric(out["ball_carrier_player_id"], errors="coerce").astype("Int64")
    if str(tid_dtype) == "Int64":
        out["ball_carrier_team_id"] = pd.to_numeric(out["ball_carrier_team_id"], errors="coerce").astype("Int64")

    return out


def infer_ball_carrier(
    frames: pd.DataFrame,
    *,
    tolerance_m: float = 3.0,
    beta: float = 0.0,
    gamma: float = 0.25,
    pre: dict | None = None,
) -> pd.DataFrame:
    """Per-frame ball-carrier inference with hysteresis.

    For each frame where ``ball_state`` is not ``"dead"`` (NaN/None treated
    as alive), identifies the player most likely carrying the ball via a
    composite score of distance and velocity-toward-ball, with an incumbent
    bonus (``gamma``) to prevent flickering.

    The ``beta`` and ``gamma`` defaults are **Optuna-calibrated (TF-24)** at the
    held ``tolerance_m=3.0`` against a 3-provider fold (SkillCorner + IDSSE/DFL +
    Gradient Sports), maximizing carrier accuracy; the Balanced and Gold-max folds
    agreed (``beta``≈0.0002/0.0009, ``gamma``≈0.221/0.259). ``beta=0`` (down from an
    engineering default of 0.5) means velocity-toward-ball weighting did not help
    carrier-actor accuracy; ``gamma≈0.25`` (down from 1.0) means near-stateless
    selection is best. ``tolerance_m`` is **left at 3.0**: the carrier-actor-action
    calibration objective is *under-determined* on the radius (its labels are
    on-ball moments only, with no loose-ball negatives, so it presses the radius to
    the upper search bound — not a value to apply). See the calibration manifest in
    the TF-24 run reports.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_m : float, default 3.0
        Maximum ball-to-player distance for candidacy (meters).
        Carrier-attribution radius, not dribbling-contact threshold. Held at the
        original engineering value — objective-under-determined (see above).
    beta : float, default 0.0
        Distance advantage per m/s of velocity toward ball (seconds).
        Optuna-calibrated (TF-24) to ~0 — velocity weighting did not help.
        ``beta=0`` makes selection purely distance-based.
    gamma : float, default 0.25
        Hysteresis bonus for incumbent carrier (meters). The incumbent's
        score is reduced by ``gamma``, so a new candidate must be
        ``gamma`` meters better in composite score to take over.
        ``gamma=0`` gives stateless per-frame behaviour. Optuna-calibrated
        (TF-24) to ~0.25 (near-stateless).
    pre : dict, optional
        Precomputed ``_pre_index_frames(frames)`` result. The pre-index step
        (long-form frames → dense per-frame numpy arrays) is a pure function of
        ``frames`` and independent of ``tolerance_m`` / ``beta`` / ``gamma``, but
        it dominates the cost. Callers that re-infer carriers on the *same*
        frames with *different* params (e.g. the TF-24 Optuna calibration sweep)
        can pass a cached ``pre`` to skip re-marshalling on every call — the
        result is bit-identical to recomputing it. Leave ``None`` (default) to
        compute it internally.

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
        carriers = infer_ball_carrier(frames, tolerance_m=3.0, beta=0.0, gamma=0.25)

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

    # Phase 1: pre-index (pure function of `frames`; a caller may supply a cached
    # `pre` to skip this dominant cost when re-inferring on the same frames with
    # different params — bit-identical to recomputing).
    if pre is None:
        pre = _pre_index_frames(frames)

    # Phase 2: kernel
    _kernel = _carrier_loop_numba if _HAS_NUMBA else _carrier_loop_numpy
    winner_slot, winner_dist = _kernel(
        bx=pre["bx"],
        by=pre["by"],
        ball_dead=pre["ball_dead"],
        px=pre["px"],
        py=pre["py"],
        pvx=pre["pvx"],
        pvy=pre["pvy"],
        player_slots=pre["player_slots"],
        n_valid=pre["n_valid"],
        seg_starts=pre["seg_starts"],
        seg_ends=pre["seg_ends"],
        tolerance_m=tolerance_m,
        beta=beta,
        gamma=gamma,
        has_velocity=has_velocity,
    )

    # Phase 3: post-process
    return _post_process(winner_slot, winner_dist, pre)


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
        Copy of ``frames`` with two added columns: ``team_in_possession`` and
        ``ball_carrier_player_id`` (possession's team + player facets — the latter
        forwarded to accessible-space as ``player_in_possession_col`` for correct
        offside masking in DAS). Frames with no carrier match get ``NaN``.

    Examples
    --------
    Typical pipeline --- infer carrier, then derive possession::

        from silly_kicks.tracking import infer_ball_carrier, derive_team_in_possession

        carrier = infer_ball_carrier(frames)
        frames_with_poss = derive_team_in_possession(frames, carrier)
    """
    merge_cols = ["game_id", "period_id", "frame_id"]
    carrier_slim = carrier[[*merge_cols, "ball_carrier_team_id", "ball_carrier_player_id"]].copy()
    carrier_slim = carrier_slim.rename(columns={"ball_carrier_team_id": "team_in_possession"})
    return frames.merge(carrier_slim, on=merge_cols, how="left")
