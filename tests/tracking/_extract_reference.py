"""The pre-D20 per-frame extraction code, VERBATIM -- the test oracles for spec r4 D20 (the numpy extract).

D20 rewrites the per-frame pandas object work of the extract hot paths (boolean masks, a DataFrame per frame,
per-frame id comparisons, ``Series.apply``) as array slices, bit-identical by construction. Each function below is
the `2474b73` body, copied unchanged, so the identity tests (``tests/tracking/test_extract_numpy_identity.py``)
compare the new code against what ran before. Never edit a body.

- ``_pre_index_frames``: ``silly_kicks/tracking/_ball_carrier.py`` at `2474b73`
  (git blob ``351cbcefbca3268e2c1e111b2b86b144b59cf032``).
- ``extract_ghost_gk_features``, ``_extract_all_ghost_gk_features``: ``silly_kicks/tracking/_ghost_gk.py`` at
  `2474b73` (git blob ``ad765b0f12c90449507570a777342f2f86339fa2``). The module-level names they call are
  imported from the same sources; the ghost-GK helpers among them are unchanged since `2474b73`.
- ``extract_xcross_features``, ``prepare_xcross_training_data``, ``compute_xcross_attempt``:
  ``silly_kicks/tracking/_xcross_attempt.py`` at `2474b73` (git blob ``0b4bcec2b0900478f51d6ad2484a3f8726f51e85``);
  the module-level names they call come from the same sources and are unchanged since `2474b73`.
- ``extract_xshot_features``, ``prepare_xshot_training_data``, ``compute_xshot_occurrence``:
  ``silly_kicks/tracking/_xshot_occurrence.py`` at `2474b73` (git blob ``2c1347c73350345c2788b50a90a20d45c4611e4f``);
  the module-level names they call come from the same sources and are unchanged since `2474b73`.

Each function keeps its `@overload` stubs, so the oracle types like the original.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from silly_kicks.id_compat import canonical_id, canonical_id_series, ids_match, same_id
from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
from silly_kicks.tracking._ghost_gk import (
    _FIELD_LENGTH,
    _GOAL_Y,
    _VELOCITY_WINDOW_S,
    GHOST_GK_FEATURE_NAMES,
    GHOST_GK_FEATURE_NAMES_POSITION_ONLY,
    GhostFeatureSet,
    _to_gr_vx,
    _to_gr_vy,
    _to_gr_x,
    _to_gr_y,
)
from silly_kicks.tracking._gk_resolve import resolve_defended_goals
from silly_kicks.tracking._velocity_availability import (
    velocity_availability_is_mixed as _velocity_availability_is_mixed,
)
from silly_kicks.tracking._velocity_availability import (
    velocity_unavailable_by_design as _velocity_unavailable_by_design,
)
from silly_kicks.tracking._xcross_attempt import (
    _ADVANCE_M,
    _DEFAULT_CARRIER_PARAMS,
    _DEFAULT_CROSS_TYPES,
    _GOAL_HALF_WIDTH_M,
    XCROSS_FEATURE_NAMES_FAITHFUL,
    XCROSS_FEATURE_NAMES_POSITION_ONLY,
    XCrossAttemptModel,
    XCrossFeatureSet,
    _dominant_region_area,
    _has_results,
    _in_wide_area,
    _nearest_dist,
    _polar,
    _resolve_xcross_model_for_frames,
    build_xcross_labels,
)
from silly_kicks.tracking._xshot_occurrence import (
    _DEFAULT_SHOT_TYPES,
    XSHOT_FEATURE_NAMES_FAITHFUL,
    XSHOT_FEATURE_NAMES_POSITION_ONLY,
    XShotFeatureSet,
    XShotOccurrenceModel,
    _ball_in_attacking_third,
    _nearest_k,
    _open_goal_fraction,
    _resolve_xshot_model_for_frames,
    build_xshot_labels,
)
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE


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


def extract_ghost_gk_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id: float | int | str,
    goal_x: float = 0.0,
    score_diff: float = 0.0,
    phase: int = 0,
    ball_carrier_team_id: float | int | str | None = None,
    prev_defensive_line_x: float | None = None,
    prev_defending_centroid_x: float | None = None,
    dt: float = _VELOCITY_WINDOW_S,
    feature_set: GhostFeatureSet = "faithful",
) -> pd.DataFrame:
    """Extract 26 ghost-GK features from frame data in goal-relative coords.

    Accepts a single frame (one game_id/period_id/frame_id group) and
    returns a single-row DataFrame. Called in a vectorized batch by
    compute_ghost_gk.

    Input frames MUST be in LTR-normalized convention (home attacks right
    in all periods). The goal_x parameter specifies which end the GK
    defends: 0.0 for the home GK, 105.0 for the away GK.

    Parameters
    ----------
    frame_data : pd.DataFrame
        All rows for one frame — players + ball.
    gk_team_id : float | int | str
        Team ID of the GK whose ghost position we predict. ``float`` is admitted
        deliberately, not as a convenience: a float-backed team id is what a merge or an
        NaN-carrying column leaves behind (``infer_ball_carrier`` emits boxed floats, and
        the zero-row serve path types ``gk_team_id`` float64), and ``ids_match`` resolves
        it via ``canonical_id`` -- ``canonical_id(366.0) == canonical_id("366")``. Typing
        it away would have made the ONE dtype this function must survive unrepresentable.
    goal_x : float
        x-coordinate of the defending goal (0.0 or 105.0).
    score_diff : float
        GK's team score minus opponent.
    phase : int
        0 = open_play, 1 = set_piece, 2 = goal_kick.
    ball_carrier_team_id : float | int | str | None
        Team currently in possession. Same id-dtype latitude as ``gk_team_id`` above --
        this is precisely the column ``infer_ball_carrier`` hands over as boxed floats.
    prev_defensive_line_x : float | None
        Previous frame's defensive line x (for velocity).
    prev_defending_centroid_x : float | None
        Previous frame's defending centroid x.
    dt : float
        Time delta for velocity computation.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 26 columns (GHOST_GK_FEATURE_NAMES).

    Examples
    --------
    Extract the 26-column feature row for a single frame::

        features = extract_ghost_gk_features(frame, gk_team_id=1, goal_x=0.0)
        features.shape  # (1, 26)
    """
    # --- Coordinate transform (ADR-089 both-axes; helpers are module-level, shared with serve) ---
    flip = goal_x > 50.0

    # --- Ball ---
    ball = frame_data[frame_data["is_ball"].astype(bool)]
    if len(ball) > 0:
        bx_raw = float(ball["x"].iloc[0])
        by_raw = float(ball["y"].iloc[0])
        bvx_raw = float(ball["vx"].iloc[0]) if "vx" in ball.columns else np.nan
        bvy_raw = float(ball["vy"].iloc[0]) if "vy" in ball.columns else np.nan
    else:
        bx_raw = by_raw = bvx_raw = bvy_raw = np.nan

    ball_x = _to_gr_x(bx_raw, flip)
    ball_y = _to_gr_y(by_raw, flip)
    ball_vx = _to_gr_vx(bvx_raw, flip)
    ball_vy = _to_gr_vy(bvy_raw, flip)
    ball_speed = float(np.sqrt(bvx_raw**2 + bvy_raw**2)) if not np.isnan(bvx_raw) else np.nan
    ball_dist = float(np.sqrt(ball_x**2 + (ball_y - _GOAL_Y) ** 2)) if not np.isnan(ball_x) else np.nan

    # --- Player splits ---
    players = frame_data[~frame_data["is_ball"].astype(bool)]
    # ADR-019: gk_team_id is a caller-supplied scalar and need NOT share the frames'
    # team_id dtype (GS emits nullable Int64; sportec/kloppy carry object strings). A raw
    # compare is silently all-False across dtypes -> an EMPTY defending split and a corrupt
    # feature row rather than an error.
    #
    # The attacking split is `~ids_match`, NOT `ids_differ`: ids_differ requires BOTH sides
    # present, so it would move an NA-team player out of `attacking` (where the original
    # `!=` put it) into neither split. `~ids_match` reproduces the original NA semantics
    # exactly -- for an NA player team_id AND for an NA gk_team_id.
    _is_gk_team = ids_match(players["team_id"], gk_team_id)
    _is_gk_flag = players["is_goalkeeper"].astype(bool)
    defending = players[_is_gk_team & ~_is_gk_flag]
    attacking = players[~_is_gk_team & ~_is_gk_flag]
    gk_rows = players[_is_gk_team & _is_gk_flag]

    # --- Defensive line ---
    if len(defending) > 0:
        def_xs = np.asarray(defending["x"].apply(lambda v: _to_gr_x(v, flip)).values)
        sorted_xs = np.sort(def_xs)
        n_back = min(4, len(sorted_xs))
        defensive_line_x = float(np.median(sorted_xs[:n_back]))
        deepest_defender_x = float(sorted_xs[0])
        defensive_line_width = float(defending["y"].max() - defending["y"].min())
    else:
        defensive_line_x = deepest_defender_x = defensive_line_width = np.nan

    if len(gk_rows) > 0 and not np.isnan(defensive_line_x):
        gk_x_gr = _to_gr_x(float(gk_rows["x"].iloc[0]), flip)
        defensive_line_depth = abs(defensive_line_x - gk_x_gr)
    else:
        defensive_line_depth = np.nan

    # --- Attacking players ---
    if len(attacking) > 0:
        atk_xs = np.asarray(attacking["x"].apply(lambda v: _to_gr_x(v, flip)).values)
        atk_ys = np.asarray(attacking["y"].apply(lambda v: _to_gr_y(v, flip)).values)
        nearest_atk_x = float(np.min(atk_xs))
        atk_cx = float(np.mean(atk_xs))
        atk_cy = float(np.mean(atk_ys))
        # atk_xs/atk_ys are goal-relative (both axes, ADR-089), which is what the helper expects.
        in_box = _geo.in_penalty_area_goal_relative_array(atk_xs, atk_ys)
        attackers_in_box = int(np.sum(in_box))
    else:
        nearest_atk_x = atk_cx = atk_cy = np.nan
        attackers_in_box = 0

    # --- Defenders behind ball ---
    if len(defending) > 0 and not np.isnan(ball_x):
        def_xs_arr = np.asarray(defending["x"].apply(lambda v: _to_gr_x(v, flip)).values)
        defenders_behind_ball = int(np.sum(def_xs_arr < ball_x))
    else:
        defenders_behind_ball = 0

    # --- Spatial geometry ---
    ball_to_goal_angle = float(np.arctan2(ball_y - _GOAL_Y, ball_x)) if not np.isnan(ball_x) else np.nan

    if len(attacking) > 0 and not np.isnan(ball_x):
        atk_xs_rel = np.asarray(attacking["x"].apply(lambda v: _to_gr_x(v, flip)).values)
        atk_ys_rel = np.asarray(attacking["y"].apply(lambda v: _to_gr_y(v, flip)).values)
        dists = np.sqrt((atk_xs_rel - ball_x) ** 2 + (atk_ys_rel - ball_y) ** 2)
        ball_to_nearest_atk = float(np.min(dists))
    else:
        ball_to_nearest_atk = np.nan

    if len(defending) >= 3:
        coords = np.column_stack(
            [
                np.asarray(defending["x"].apply(lambda v: _to_gr_x(v, flip)).values),
                np.asarray(defending["y"].apply(lambda v: _to_gr_y(v, flip)).values),
            ]
        )
        try:
            hull = ConvexHull(coords)
            compactness = float(hull.volume)
        except QhullError:
            compactness = np.nan
    else:
        compactness = np.nan

    ball_in_own_half = 1.0 if (not np.isnan(ball_x) and ball_x < _FIELD_LENGTH / 2) else 0.0
    # ADR-019 scalar-vs-scalar seam. `ball_carrier_team_id` may reach here from the public
    # `carrier=` cache kwarg with a dtype the frames do not share, in which case a raw
    # compare pins team_in_possession to 0 for every row. `same_id` also subsumes the
    # try/except this replaced: it returns False for None/pd.NA/NaN on either side rather
    # than raising (the fence's stated reason -- "pd.NA comparison raises TypeError" -- is
    # handled inside `_canonical`), so the None/NA/NaN behaviour is unchanged.
    team_in_poss = 1.0 if same_id(ball_carrier_team_id, gk_team_id) else 0.0
    period_clamped = min(int(frame_data["period_id"].iloc[0]), 2)
    time_s = float(frame_data["time_seconds"].iloc[0]) if "time_seconds" in frame_data.columns else 0.0

    # --- Velocity (actual dt from consecutive frames) ---
    if prev_defensive_line_x is not None and not np.isnan(defensive_line_x):
        def_line_speed = (defensive_line_x - prev_defensive_line_x) / dt
    else:
        def_line_speed = np.nan

    if prev_defending_centroid_x is not None and len(defending) > 0:
        def_cx = float(np.mean(np.asarray(defending["x"].apply(lambda v: _to_gr_x(v, flip)).values)))
        def_centroid_vx = (def_cx - prev_defending_centroid_x) / dt
    else:
        def_centroid_vx = np.nan

    row = [
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        ball_dist,
        defensive_line_x,
        defensive_line_depth,
        defensive_line_width,
        attackers_in_box,
        nearest_atk_x,
        atk_cx,
        atk_cy,
        defenders_behind_ball,
        deepest_defender_x,
        phase,
        team_in_poss,
        score_diff,
        time_s,
        period_clamped,
        ball_to_goal_angle,
        ball_to_nearest_atk,
        compactness,
        ball_in_own_half,
        ball_speed,
        def_line_speed,
        def_centroid_vx,
    ]
    # Feature-set-driven final assembly of ONE computation (DRY): position_only selects the same
    # values minus the 5 velocity features. On a lone freeze frame the dropped velocity features would
    # be NaN (no vx/vy, no predecessor for the cross-frame derivatives) -- but they are not selected,
    # so the 21-vector is finite (the single-frame obligation).
    _values = dict(zip(GHOST_GK_FEATURE_NAMES, row, strict=True))
    _names = GHOST_GK_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else GHOST_GK_FEATURE_NAMES
    return pd.DataFrame([[_values[c] for c in _names]], columns=_names)


def _extract_all_ghost_gk_features(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    carrier: pd.DataFrame | None = None,
    score_at_time: Callable[[Any, float], float] | None = None,
    phase_at_time: Callable[[Any, float], int] | None = None,
    subsample_fps: float | None = None,
    link_frame_ids: set[int] | None = None,
    feature_set: GhostFeatureSet = "faithful",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared batch helper: iterate frames, extract features for every GK.

    Both compute_ghost_gk (inference) and prepare_ghost_gk_training_data
    (training) call this function. Single source of truth for the
    frame-iteration + velocity-tracking + feature-extraction loop.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS, LTR-normalized, vx/vy present.
    home_team_id : str | int
        Home team (used as fallback; defending goal is inferred per-period
        from mean GK x position to handle LTR data with period flips).
    carrier : pd.DataFrame | None
        Per-frame ball_carrier_team_id (from derive_team_in_possession).
    score_at_time : callable | None
        (game_id, time_seconds) -> score_diff (home perspective).
    phase_at_time : callable | None
        (game_id, time_seconds) -> int (0=open, 1=set_piece, 2=goal_kick).
    subsample_fps : float | None
        Thin frames to target fps before extraction.
    link_frame_ids : set[int] | None, default None
        When provided, build a feature row ONLY for frames in this set, but still
        walk every frame to maintain the cross-period one-step velocity state, so
        each linked frame sees its true predecessor (byte-identical velocity to the
        unrestricted pass). The per-period defending-goal mean-x is computed over
        the full frames either way. When None, every frame is extracted. PR-S66 §5.

    Returns
    -------
    features : pd.DataFrame
        (n_samples, len(GHOST_GK_FEATURE_NAMES)).
    meta : pd.DataFrame
        (n_samples, 8): game_id, period_id, frame_id, gk_team_id,
        gk_x_gr, gk_y_gr, gk_player_id, gk_visibility. The last two carry
        the keeper's identity and per-frame detection flag off the SAME
        frames row the label came from (spec 4.3: keeper-grouped CV +
        detected-only targets).

    Examples
    --------
    Batch-extract features and metadata across all frames of a match::

        features, meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
        features.shape[1]  # 26
    """
    # --- Team ID normalization (§7) ---
    frame_team_dtype = frames["team_id"].dtype
    if frame_team_dtype is object:
        home_team_id = str(home_team_id)
    else:
        try:
            home_team_id = type(frames["team_id"].dropna().iloc[0])(home_team_id)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"home_team_id={home_team_id!r} cannot be coerced to frames['team_id'] dtype {frame_team_dtype}"
            ) from exc

    # --- Pre-index carrier for O(1) lookup ---
    carrier_idx: pd.Series | None = None
    if carrier is not None and "ball_carrier_team_id" in carrier.columns:
        carrier_idx = carrier.set_index(["game_id", "period_id", "frame_id"])["ball_carrier_team_id"]

    # --- Subsample ---
    work = frames
    if subsample_fps is not None and "frame_rate" in frames.columns:
        fr = frames["frame_rate"].iloc[0]
        if fr > 0 and subsample_fps > 0:
            step = max(1, round(fr / subsample_fps))
            # Keep every step-th unique frame_id per (game_id, period_id)
            unique_frames = (
                frames[["game_id", "period_id", "frame_id"]]
                .drop_duplicates()
                .sort_values(["game_id", "period_id", "frame_id"])
            )
            keep_mask = unique_frames.groupby(["game_id", "period_id"]).cumcount() % step == 0
            keep_keys = unique_frames[keep_mask.values]
            work = frames.merge(keep_keys, on=["game_id", "period_id", "frame_id"])

    # --- Defending goal per (game_id, period_id, team_id), from the pinned seam ---
    # The end comes from the FRAMES, never from team identity: identity-keying is correct
    # only while frames are home-attacks-right, which is a property of the caller's pipeline
    # rather than of this function's inputs. Built from `work` -- the post-subsample /
    # post-link-filter set -- because passing `frames` would change the map whenever
    # `subsample_fps` is set.
    _goal_map = resolve_defended_goals(work)

    # --- Group and iterate ---
    group_keys = ["game_id", "period_id", "frame_id"]
    grouped = list(work.groupby(group_keys, sort=True))

    feature_rows: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    prev_state: dict[tuple, tuple[float, float]] = {}
    prev_timestamps: dict[tuple, float] = {}

    for (gid, pid, fid), frame_data in grouped:
        gk_rows = frame_data[frame_data["is_goalkeeper"].astype(bool) & ~frame_data["is_ball"].astype(bool)]
        time_s = float(frame_data["time_seconds"].iloc[0]) if "time_seconds" in frame_data.columns else 0.0

        # PR-S66 §5: when link_frame_ids is set, build a feature row only for
        # linked frames, but ALWAYS walk every frame below to update the velocity
        # state so a linked frame's predecessor is its true (cross-period) neighbour
        # rather than the previous *linked* frame.
        # fid is the groupby key (int/np.int); membership in the int set matches
        # by value regardless of numpy-vs-python int, so no explicit cast needed.
        fid_linked = link_frame_ids is None or fid in link_frame_ids

        for _, gk_row in gk_rows.iterrows():
            gk_team = gk_row["team_id"]
            goal_x = _goal_map.get(gid, pid, gk_team, allow_guess=True)
            if goal_x is None:
                # No usable end -> no ghost. The former fallback was
                # `0.0 if same_id(gk_team, home_team_id) else 105.0`, a CONSTANT 105.0 for any
                # NA-team keeper -- wrong for half the possible cases, and silently, because the
                # feature vector is then computed in the mirrored goal-relative frame.
                continue
            flip = goal_x > 50.0

            # Cheap defensive-line-x + centroid in goal-relative coords, computed
            # for EVERY frame to drive the velocity state. These mirror exactly
            # extract_ghost_gk_features' defensive_line_x and the stored centroid
            # (median of the back-4 goal-relative x; mean goal-relative x) — see
            # TestExtractionRestriction golden which guards bit-identical velocity.
            # ADR-019: must use the SAME id-identity rule as extract_ghost_gk_features'
            # defending split -- the TestExtractionRestriction golden guards that the
            # velocity state computed here is bit-identical to the extractor's.
            defending = frame_data[
                ids_match(frame_data["team_id"], gk_team)
                & ~frame_data["is_goalkeeper"].astype(bool)
                & ~frame_data["is_ball"].astype(bool)
            ]
            if len(defending) > 0:
                _dxs = defending["x"].to_numpy(dtype="float64")
                _gr = (_FIELD_LENGTH - _dxs) if flip else _dxs
                _sorted_xs = np.sort(_gr)
                _n_back = min(4, len(_sorted_xs))
                dl_x = float(np.median(_sorted_xs[:_n_back]))
                def_cx = float(np.mean(_gr))
            else:
                dl_x = np.nan
                def_cx = np.nan

            # Velocity state (true predecessor regardless of restriction)
            state_key = (gid, gk_team)
            prev_dl_x, prev_dc_x = prev_state.get(state_key, (None, None))
            prev_ts = prev_timestamps.get(state_key)
            actual_dt = (time_s - prev_ts) if prev_ts is not None and time_s > prev_ts else _VELOCITY_WINDOW_S

            if fid_linked:
                # Score: callback returns home perspective, negate for away
                if score_at_time is not None:
                    sd = score_at_time(gid, time_s)
                    if not same_id(gk_team, home_team_id):
                        sd = -sd
                else:
                    sd = 0.0

                # Phase
                ph = phase_at_time(gid, time_s) if phase_at_time is not None else 0

                # Carrier
                carrier_team = None
                if carrier_idx is not None:
                    key = (gid, pid, fid)
                    if key in carrier_idx.index:
                        carrier_team = carrier_idx[key]  # type: ignore[call-overload]

                feat = extract_ghost_gk_features(
                    frame_data,
                    gk_team_id=gk_team,
                    goal_x=goal_x,
                    score_diff=sd,
                    phase=ph,
                    ball_carrier_team_id=carrier_team,
                    prev_defensive_line_x=prev_dl_x,
                    prev_defending_centroid_x=prev_dc_x,
                    dt=actual_dt,
                    feature_set=feature_set,
                )
                feature_rows.append(feat)

                # GK position in goal-relative coords for labels (ADR-089: both axes flip,
                # so the target matches the both-axes features and serve_ghost_gk_positions'
                # inverse).
                gk_x_raw = float(gk_row["x"])
                gk_y_raw = float(gk_row["y"])
                gk_x_gr = _to_gr_x(gk_x_raw, flip)
                gk_y_gr = _to_gr_y(gk_y_raw, flip)

                meta_rows.append(
                    {
                        "game_id": gid,
                        "period_id": pid,
                        "frame_id": fid,
                        "gk_team_id": gk_team,
                        "gk_x_gr": gk_x_gr,
                        "gk_y_gr": gk_y_gr,
                        "gk_player_id": gk_row["player_id"],
                        "gk_visibility": gk_row.get("visibility"),
                    }
                )

            # Update velocity state (every frame, linked or not)
            prev_state[state_key] = (dl_x, def_cx)
            prev_timestamps[state_key] = time_s

    if not feature_rows:
        # feature_set-appropriate empty columns, matching the non-empty concat below (a hardcoded
        # faithful 26-col empty frame would fail the position_only width-check in prepare).
        _empty_names = (
            GHOST_GK_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else GHOST_GK_FEATURE_NAMES
        )
        return (
            pd.DataFrame(columns=_empty_names),
            pd.DataFrame(
                columns=[
                    "game_id",
                    "period_id",
                    "frame_id",
                    "gk_team_id",
                    "gk_x_gr",
                    "gk_y_gr",
                    "gk_player_id",
                    "gk_visibility",
                ]
            ),
        )

    features = pd.concat(feature_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    return features, meta


@overload
def extract_xcross_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id,
    goal_x: float,
    carrier_player_id,
    feature_set: XCrossFeatureSet = ...,
    score_differential: float = ...,
    return_box_detail: Literal[False] = ...,
) -> pd.DataFrame: ...


@overload
def extract_xcross_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id,
    goal_x: float,
    carrier_player_id,
    feature_set: XCrossFeatureSet = ...,
    score_differential: float = ...,
    return_box_detail: Literal[True],
) -> tuple[pd.DataFrame, dict]: ...


def extract_xcross_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id,
    goal_x: float,
    carrier_player_id,
    feature_set: XCrossFeatureSet = "faithful",
    score_differential: float = np.nan,
    return_box_detail: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Extract the faithful xCross feature row for a single (game, period, frame) snapshot.

    ``return_box_detail=True`` (default False, so every existing caller and the serve path are
    byte-identical) additionally returns the behind-line box detail used by the keeper-box gr_x
    measurement -- computed from the SAME ``in_box``/``gr_x``/``team`` arrays that build
    ``box_off_def_ratio``, so the descriptive split cannot drift from the feature it explains.

    All spatial features are in goal-relative coordinates (attacked goal at gr_x = 0). The GK block
    is the contiguous tail (``XCROSS_GK_BLOCK``). ``score_differential`` is supplied by the caller
    (prepare/compute build it from match-context actions; NaN when unavailable). NaN-tolerant.
    """
    if feature_set == "extended":
        raise NotImplementedError(
            "xCrossAttempt feature_set='extended' is a deferred extension point; "
            "only 'faithful' and 'position_only' are implemented. See the design spec."
        )
    f = frame_data
    is_ball = f["is_ball"].to_numpy(dtype=bool)
    is_gk = f["is_goalkeeper"].to_numpy(dtype=bool)
    # CANONICAL id arrays (ADR-019): a nullable-Int64 id column (GradientSports) -> .to_numpy()
    # upcasts to float64, so a naive ``.astype(str)`` yields "366.0" and never matches the clean-int
    # carrier id "366" -> carrier_mask all-False -> every carrier-anchored feature (and the
    # carrier-gated GK block) silently NaN. canonical_id_series collapses 366/366.0/Int64(366)/"366"
    # -> "366"; the sentinel fill keeps NA (ball rows) out of the comparisons. Genuine-string
    # providers (kloppy) are unchanged, so the trained public model needs no retrain.
    na_fill = "\x00"  # sentinel: no canonical id equals it, so NA (ball) rows never match
    team = canonical_id_series(f["team_id"]).fillna(na_fill).to_numpy()
    pid = canonical_id_series(f["player_id"]).fillna(na_fill).to_numpy()
    gr_x = np.array([_geo.to_goal_relative_x(x, goal_x=goal_x) for x in f["x"].to_numpy(dtype="float64")])
    # y is GOAL-RELATIVE from here down (PR 5). Paired with gr_x this is the 180-degree point
    # reflection; before PR 5 y rode through untransformed, so `atan2(y - GOAL_Y, gr_x)` negated
    # every bearing between the two goal ends while every radial stayed byte-identical. The local
    # name is kept because every consumer below already reads `y`.
    y = np.array([_geo.to_goal_relative_y(v, goal_x=goal_x) for v in f["y"].to_numpy(dtype=float)])

    out: dict[str, float] = {name: np.nan for name in XCROSS_FEATURE_NAMES_FAITHFUL}
    box_detail: dict = {
        "n_off": 0,
        "n_def": 0,
        "n_off_behind": 0,
        "n_def_behind": 0,
        "box_off_def_ratio_clamped": np.nan,
        "behind_line_box_gr_x": np.array([], dtype=float),
    }

    # Ball (ball-anchored)
    if is_ball.any():
        bx, by = float(gr_x[is_ball][0]), float(y[is_ball][0])
        out["ball_r"], out["ball_theta"] = _polar(bx, by - _geo.GOAL_Y)
        # NAN-TOLERANT, as this function's docstring promises and its NaN pre-fill implements: an
        # absent vx/vy leaves `ball_speed` NaN instead of raising. It used to read the columns
        # unguarded, so a freeze-frame -- which legitimately carries neither, and says so via
        # `speed_source` -- died on `KeyError: 'vx'`. A zero-fill would be worse than either: a
        # fabricated stationary ball fed to a trained feature. The POSITIONAL ball features above
        # are deliberately computed first, since they do not depend on velocity.
        if "vx" in f.columns and "vy" in f.columns:
            bvx = float(f.loc[is_ball, "vx"].to_numpy()[0])
            bvy = float(f.loc[is_ball, "vy"].to_numpy()[0])
            out["ball_speed"] = math.hypot(bvx, bvy)

    # Carrier-anchored geometry. Match the carrier by CANONICAL id (pid is already canonical above),
    # so it works whether the frame player_id is native string (kloppy/sportec/skillcorner/metrica)
    # OR nullable Int64 (gradientsports) -- see the canonical-id comment above.
    carrier_mask = np.zeros(len(f), dtype=bool)
    has_carrier = carrier_player_id is not None and not (
        isinstance(carrier_player_id, float) and math.isnan(carrier_player_id)
    )
    carrier_key = canonical_id(carrier_player_id) if has_carrier else None
    if has_carrier:
        carrier_mask = (pid == carrier_key) & ~is_ball
    cx = cy = None
    if carrier_mask.any():
        cx, cy = float(gr_x[carrier_mask][0]), float(y[carrier_mask][0])
        carrier_team = team[carrier_mask][0]
        opp = [(gr_x[i], y[i]) for i in range(len(f)) if not is_ball[i] and team[i] != carrier_team]
        mate = [
            (gr_x[i], y[i])
            for i in range(len(f))
            if not is_ball[i] and team[i] == carrier_team and pid[i] != carrier_key
        ]
        all_xy = [(gr_x[i], y[i]) for i in range(len(f)) if not is_ball[i]]
        out["dist_nearest_def"] = _nearest_dist((cx, cy), opp)
        out["dist_nearest_teammate"] = _nearest_dist((cx, cy), mate)
        out["dist_endline"] = float(cx)  # C1: attacked goal at gr_x=0 -> distance to endline IS cx
        out["space_controlled"] = _dominant_region_area((cx, cy), all_xy)

    # GK block (NOVEL extension) -- defending GK = is_goalkeeper row on gk_team_id.
    # C1: posts live at the ATTACKED goal line gr_x = 0 (NOT PITCH_LENGTH). Gated on carrier (needs cy).
    gk_mask = is_gk & (team == canonical_id(gk_team_id))
    if gk_mask.any() and cx is not None and cy is not None:  # cy set iff the carrier resolved
        gkx, gky = float(gr_x[gk_mask][0]), float(y[gk_mask][0])
        out["gk_r"], out["gk_theta"] = _polar(gkx, gky - _geo.GOAL_Y)
        out["gk_lateral_offset"] = float(gky - _geo.GOAL_Y)
        # carrier flank sign; `or 1.0` only guards the exactly-central carrier (cy == GOAL_Y).
        side = np.sign(cy - _geo.GOAL_Y) or 1.0
        near_post_y = _geo.GOAL_Y + _GOAL_HALF_WIDTH_M * side  # post on the carrier's flank
        far_post_y = _geo.GOAL_Y - _GOAL_HALF_WIDTH_M * side
        out["gk_dist_near_post"] = math.hypot(gkx, gky - near_post_y)  # goal at gr_x=0
        out["gk_dist_far_post"] = math.hypot(gkx, gky - far_post_y)
        out["gk_carrier_side"] = float((gky - _geo.GOAL_Y) * side)

    # #6 off/def ratio in the attacked box. C1: attacked box is goal-relative gr_x <= 16.5
    # (attacked goal at gr_x=0), |y-34| <= 20.16.
    # The helper has no ball concept, so `& ~is_ball` composes HERE rather than inside it.
    in_box = _geo.in_penalty_area_goal_relative_array(gr_x, y) & ~is_ball
    if carrier_mask.any():
        carrier_team = team[carrier_mask][0]
        off_in_box = (team == carrier_team) & in_box & ~is_gk
        def_in_box = (team != carrier_team) & in_box & ~is_gk
        n_off = int(off_in_box.sum())
        n_def = int(def_in_box.sum())
        out["box_off_def_ratio"] = float(n_off / n_def) if n_def > 0 else float(n_off)
        if return_box_detail:
            # The clamp (gr_x >= 0) removes exactly the behind-line box players from both counts,
            # so the clamped ratio is recomputed here from the SAME masks -- no second extraction.
            off_behind = off_in_box & (gr_x < 0.0)
            def_behind = def_in_box & (gr_x < 0.0)
            n_off_b, n_def_b = int(off_behind.sum()), int(def_behind.sum())
            c_off, c_def = n_off - n_off_b, n_def - n_def_b
            box_detail = {
                "n_off": n_off,
                "n_def": n_def,
                "n_off_behind": n_off_b,
                "n_def_behind": n_def_b,
                "box_off_def_ratio_clamped": float(c_off / c_def) if c_def > 0 else float(c_off),
                "behind_line_box_gr_x": gr_x[off_behind | def_behind],
            }

    # #1 score differential (passed in; NaN at serve unless a score lookup supplied it)
    out["score_differential"] = (
        float(score_differential) if not (score_differential is None or math.isnan(score_differential)) else np.nan
    )

    # #8 ten-minute warning (final 10 min of a 45-min half). PA-M1: relies on time_seconds being
    # PER-PERIOD (resets to 0 each half) -- verified: the kloppy gateway sets
    # time_seconds = frame.timestamp.total_seconds() and kloppy timestamps are period-relative
    # (sportec native expects the same contract). ET periods (3/4/5) get 0 by design (the paper's
    # feature is "final 10 min of a half"). Locked by test_ten_minute_warning_period2_early_is_zero.
    t = float(f["time_seconds"].iloc[0])
    period = int(f["period_id"].iloc[0])
    out["ten_minute_warning"] = 1 if period in (1, 2) and t >= 35 * 60 else 0

    # Feature-set-driven final assembly of ONE computation (DRY): position_only selects `out` minus
    # `ball_speed`; a velocity-less frame leaves `ball_speed` NaN but it is simply not selected.
    _names = XCROSS_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XCROSS_FEATURE_NAMES_FAITHFUL
    features = pd.DataFrame([out], columns=_names)
    return (features, box_detail) if return_box_detail else features


@overload
def prepare_xcross_training_data(
    frames: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    home_team_id,
    feature_set: XCrossFeatureSet = ...,
    horizon_seconds: float = ...,
    wide_area_only: bool = ...,
    advance_m: float = ...,
    cross_types: tuple[str, ...] = ...,
    carrier_params: dict | None = ...,
    return_meta: Literal[False] = ...,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]: ...


@overload
def prepare_xcross_training_data(
    frames: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    home_team_id,
    feature_set: XCrossFeatureSet = ...,
    horizon_seconds: float = ...,
    wide_area_only: bool = ...,
    advance_m: float = ...,
    cross_types: tuple[str, ...] = ...,
    carrier_params: dict | None = ...,
    return_meta: Literal[True],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]: ...


def prepare_xcross_training_data(
    frames: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    home_team_id,
    feature_set: XCrossFeatureSet = "faithful",
    horizon_seconds: float = 1.0,
    wide_area_only: bool = True,
    advance_m: float = _ADVANCE_M,
    cross_types: tuple[str, ...] = _DEFAULT_CROSS_TYPES,
    carrier_params: dict | None = None,
    return_meta: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (features, labels, groups=game_id) for one match. Shared train/serve extractor =
    anti-skew guarantee. ``home_team_id`` is USED to sign score_differential (PA-H1).

    ``return_meta=True`` additionally returns a per-example meta frame (row-aligned with features,
    ghost-verbatim contract) carrying the behind-line box detail for the keeper-box gr_x decision:
    ``n_off``/``n_def``/``n_off_behind``/``n_def_behind``, ``box_off_def_ratio_clamped`` (the ratio
    with behind-line box players removed), and ``behind_line_box_gr_x``. Default False leaves the
    training path byte-identical."""
    # PA-H1: confounder #1 score_differential -- reuse ghost-GK's _build_score_lookup (local import
    # keeps `import _xcross_attempt` light + avoids pulling the ghost-GK model at module load).
    # The callback is _score(game_id, time) -> home_score - away_score (negate for away possessor).
    from silly_kicks.tracking._ghost_gk import _build_score_lookup

    cp: dict = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
    carrier = infer_ball_carrier(frames, **cp)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = resolve_defended_goals(frames)  # H1: precompute once, from FULL frames
    score_fn = _build_score_lookup(actions, home_team_id) if _has_results(actions) else None
    has_ball_state = "ball_state" in poss.columns  # M3: column-presence guard (mirror xS:655)

    feat_rows: list[pd.DataFrame] = []
    frame_index_rows: list[dict] = []
    meta_details: list[dict] = []
    coverage = {"in_domain": 0, "carrier_resolved": 0}  # L-2 carrier-coverage log
    for (gid, pid, fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], sort=False):
        if has_ball_state:
            ball_row = grp[grp["is_ball"]]
            # M3: judge ball_state on the ball row (row-level), not .all() over every player row
            if len(ball_row) and str(ball_row["ball_state"].iloc[0]) == "dead":
                continue
        in_poss = grp["team_in_possession"].dropna()
        if in_poss.empty:
            continue
        poss_team = in_poss.iloc[0]
        # Defending team = the OTHER team's non-ball players. Filter by is_ball + dropna so a real
        # frame whose team_id column carries pd.NA (e.g. the ball row, or an unresolved GS jersey)
        # does not (a) hit "t not in (poss_team, 'ball')" -> "boolean value of NA is ambiguous", or
        # (b) be mistaken for a defending team. Mirrors compute_xcross_attempt's dropna()+!= pattern.
        non_ball = grp[~grp["is_ball"].astype(bool)]
        defending = [t for t in non_ball["team_id"].dropna().unique() if t != poss_team]
        if not defending:
            continue
        goal_x = goal_map.get(gid, pid, defending[0], allow_guess=True)
        if goal_x is None:
            continue
        ball = grp[grp["is_ball"]]
        bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
        by = float(ball["y"].iloc[0]) if len(ball) else np.nan
        if wide_area_only and not _in_wide_area(bx, by, goal_x, advance_m):
            continue
        coverage["in_domain"] += 1
        carrier_pid_s = grp["ball_carrier_player_id"].dropna()
        carrier_pid = carrier_pid_s.iloc[0] if not carrier_pid_s.empty else None
        if carrier_pid is not None:
            coverage["carrier_resolved"] += 1
        sd = np.nan  # PA-H1: score differential from the POSSESSING team's perspective
        if score_fn is not None:
            raw = score_fn(gid, float(grp["time_seconds"].iloc[0]))  # home_score - away_score
            # ADR-019: a naive str()==str() renders a float-backed id as "5.0" against a
            # scalar "5", so the compare is ALWAYS False and every row's sign flips.
            sd = raw if same_id(poss_team, home_team_id) else -raw
        if return_meta:
            _feat, _detail = extract_xcross_features(
                grp,
                gk_team_id=defending[0],
                goal_x=goal_x,
                carrier_player_id=carrier_pid,
                feature_set=feature_set,
                score_differential=sd,
                return_box_detail=True,
            )
            feat_rows.append(_feat)
            meta_details.append(_detail)
        else:
            feat_rows.append(
                extract_xcross_features(
                    grp,
                    gk_team_id=defending[0],
                    goal_x=goal_x,
                    carrier_player_id=carrier_pid,
                    feature_set=feature_set,
                    score_differential=sd,
                )
            )
        frame_index_rows.append(
            dict(
                game_id=gid,
                period_id=pid,
                frame_id=fid,
                time_seconds=grp["time_seconds"].iloc[0],
                team_in_possession=poss_team,
            )
        )

    if coverage["in_domain"]:
        rate = coverage["carrier_resolved"] / coverage["in_domain"]
        if rate < 0.8:
            warnings.warn(
                f"xCross carrier-resolution coverage {rate:.0%} over the wide-area domain "
                f"({coverage['carrier_resolved']}/{coverage['in_domain']}); GK/confounder "
                f"features may be degraded for this corpus.",
                UserWarning,
                stacklevel=2,
            )

    if not feat_rows:
        empty = pd.DataFrame(
            columns=(
                XCROSS_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XCROSS_FEATURE_NAMES_FAITHFUL
            )
        )
        if return_meta:
            return empty, np.array([], dtype=int), np.array([], dtype=object), pd.DataFrame()
        return empty, np.array([], dtype=int), np.array([], dtype=object)

    features = pd.concat(feat_rows, ignore_index=True)
    frame_index = pd.DataFrame(frame_index_rows)  # carries team_in_possession (matches label helper)
    y = np.asarray(build_xcross_labels(frame_index, actions, horizon_seconds=horizon_seconds, cross_types=cross_types))
    groups = frame_index["game_id"].to_numpy()
    if return_meta:
        meta = frame_index.copy()
        for _k in ("n_off", "n_def", "n_off_behind", "n_def_behind", "box_off_def_ratio_clamped"):
            meta[_k] = [d[_k] for d in meta_details]
        meta["behind_line_box_gr_x"] = [d["behind_line_box_gr_x"] for d in meta_details]
        return features, y, groups, meta
    return features, y, groups


def compute_xcross_attempt(
    frames: pd.DataFrame,
    *,
    model: XCrossAttemptModel | str | None = None,
    home_team_id: int | str | None = None,
    actions: pd.DataFrame | None = None,
    pitch_control_cache=None,  # reserved for 'extended' (not used by 'faithful')
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Add an ``xcross_attempt`` column (P(cross within ~horizon)) per in-possession frame.

    STATE-anchored: scores the in-possession team's cross propensity at each frame. Possession +
    defended goal use the model's metadata ``carrier_params`` (R3). ``actions`` (optional, PA-H1)
    enables ``score_differential`` via the match-context score lookup; omitted -> NaN
    (XGBoost-tolerant). ``pitch_control_cache`` is reserved for the deferred 'extended' variant.
    """
    out = frames.copy()
    out["xcross_attempt"] = np.nan

    # VELOCITY-AVAILABILITY CONTRACT (ADR-054) + velocity-keyed auto-select, at the SHARED seam. All
    # three public entry points reach scoring through here (`add_xcross_attempt`,
    # `xcross_attempt_xfns`, a direct call).
    #
    # MIXED availability (some-but-not-all rows declare velocity unavailable) is a caller error: it
    # would resolve to the default velocity model and fabricate ball_speed=NaN on the marked rows (M3).
    if _velocity_availability_is_mixed(frames):
        raise ValueError(
            "compute_xcross_attempt: mixed velocity-availability -- some rows declare "
            f"speed_source={SPEED_SOURCE_UNAVAILABLE!r} and some do not. Pass a single-availability "
            "frame set (all freeze-frame, or all velocity-bearing)."
        )
    # UNDECLARED missing vx/vy (NOT the declared freeze-frame shape): the "forgot derive_velocities()"
    # case. Fail loud (ADR-054 prong, preserved).
    if (
        not _velocity_unavailable_by_design(frames)
        and len(frames)
        and ("vx" not in frames.columns or "vy" not in frames.columns)
    ):
        raise ValueError(
            "compute_xcross_attempt requires vx/vy on frames (call derive_velocities() first), or "
            f"declare speed_source {SPEED_SOURCE_UNAVAILABLE!r}. See the velocity-availability "
            "contract."
        )
    # Layer B: override -> "custom"; declared-unavailable -> position_only (or NaN if unbundled, NEVER
    # the default); else default. compute output stays byte-identical (the public
    # `xcross_attempt_variant` column is added by add_xcross_attempt, which re-resolves the key).
    m, _variant = _resolve_xcross_model_for_frames(frames, model)
    if m is None:
        return out  # declared-unavailable but no bundled position_only variant -> honest NaN

    # N-A (mirror xS): carrier inference + possession run on the FULL contiguous frames (cross-frame
    # hysteresis); restrict ONLY the per-frame extract + batched predict to link_frame_ids.
    carrier = infer_ball_carrier(frames, **m.carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = resolve_defended_goals(frames)  # H1: once, from FULL frames
    score_fn = None
    if actions is not None and _has_results(actions) and home_team_id is not None:
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        score_fn = _build_score_lookup(actions, home_team_id)

    feat_rows: list[pd.DataFrame] = []
    keys: list[tuple] = []  # (gid, pid, frame_id, team_in_possession)
    for (gid, pid, frame_id), grp in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
        if link_frame_ids is not None and int(str(frame_id)) not in link_frame_ids:
            continue
        tip = grp["team_in_possession"].iloc[0]
        if pd.isna(tip):
            continue
        outfield = grp[~grp["is_ball"].astype(bool)]
        teams = [t for t in outfield["team_id"].dropna().unique() if t != tip]
        if not teams:
            continue
        def_team = teams[0]
        goal_x = goal_map.get(gid, pid, def_team, allow_guess=True)
        if goal_x is None:
            continue
        carrier_pid_s = grp["ball_carrier_player_id"].dropna()
        carrier_pid = carrier_pid_s.iloc[0] if not carrier_pid_s.empty else None
        sd = np.nan
        if score_fn is not None:
            raw = score_fn(gid, float(grp["time_seconds"].iloc[0]))
            # ADR-019: see the sibling site above -- a naive str compare flips every sign.
            sd = raw if same_id(tip, home_team_id) else -raw
        feat_rows.append(
            extract_xcross_features(
                grp,
                gk_team_id=def_team,
                goal_x=goal_x,
                carrier_player_id=carrier_pid,
                score_differential=sd,
                feature_set=m.feature_set,
            )
        )
        keys.append((gid, pid, frame_id, tip))

    if not feat_rows:
        return out

    feature_matrix = pd.concat(feat_rows, ignore_index=True)
    probs = m.predict_proba(feature_matrix)
    key_df = pd.DataFrame(keys, columns=["game_id", "period_id", "frame_id", "team_id"])
    key_df["__p"] = probs
    # N-B (mirror xS): join on TEMPORARY canonical id keys -> never mutate out dtypes.
    # canonical_id_series (ADR-019) is the dtype-safe stringify (366/366.0/Int64(366) -> "366"),
    # avoiding the .astype(str) "366.0" artifact on any float-upcast id column.
    out["__gid"] = canonical_id_series(out["game_id"])
    out["__tid"] = canonical_id_series(out["team_id"])
    key_df["__gid"] = canonical_id_series(key_df["game_id"])
    key_df["__tid"] = canonical_id_series(key_df["team_id"])
    key_df = key_df.drop(columns=["game_id", "team_id"])
    out = out.merge(key_df, on=["__gid", "period_id", "frame_id", "__tid"], how="left")
    out["xcross_attempt"] = out["__p"]
    return out.drop(columns=["__p", "__gid", "__tid"])


def extract_xshot_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id: int | str,
    goal_x: float,
    feature_set: XShotFeatureSet = "faithful",
) -> pd.DataFrame:
    """Extract xS features from one frame (goal-relative). Returns a 1-row frame.

    The defending team is ``gk_team_id`` (the team whose goal is being attacked);
    ``goal_x`` is the absolute x of that defended goal (0.0 or 105.0).

    ``feature_set="extended"`` is not implemented in PR-S75 (raises
    NotImplementedError) --- only ``"faithful"`` (the paper's 27 features) ships
    here; see the TF-16 spec.

    Examples
    --------
    >>> # row = extract_xshot_features(frame, gk_team_id=1, goal_x=0.0)
    >>> # row.shape == (1, 27)

    See NOTICE for full bibliographic citations.
    """
    if feature_set == "extended":
        raise NotImplementedError(
            "xShotOccurrence feature_set='extended' is not implemented in this "
            "release; only 'faithful' (paper Appendix A) and 'position_only' are "
            "available. See the TF-16 weights/TF-19 follow-up."
        )

    # Goal-relative ONCE, for every consumer below: the 180-degree point reflection
    # (x, y) -> (105 - x, 68 - y) when the defended goal is at high x. Do NOT reintroduce a
    # per-site helper. `gx` was x-only while y was read at FOUR independent sites (ball, defenders,
    # attackers, keeper), so before PR 5 every bearing negated between the two goal ends while every
    # radial stayed byte-identical -- and a per-site fix leaves "no call site can be missed" an
    # assertion rather than a property. `assign` returns a NEW frame (ADR-033: never mutate the
    # caller's).
    fd = frame_data.assign(
        x=frame_data["x"].map(lambda v: _geo.to_goal_relative_x(float(v), goal_x=goal_x)),
        y=frame_data["y"].map(lambda v: _geo.to_goal_relative_y(float(v), goal_x=goal_x)),
    )

    is_ball = fd["is_ball"].astype(bool)
    is_gk = fd["is_goalkeeper"].astype(bool)
    ball = fd[is_ball]
    players = fd[~is_ball]
    players_is_gk = is_gk[~is_ball]

    if len(ball) > 0:
        bx = float(ball["x"].iloc[0])
        by = float(ball["y"].iloc[0])
        bvx = float(ball["vx"].iloc[0]) if "vx" in ball.columns else np.nan
        bvy = float(ball["vy"].iloc[0]) if "vy" in ball.columns else np.nan
        bz = float(ball["z"].iloc[0]) if "z" in ball.columns else np.nan
    else:
        bx = by = bvx = bvy = bz = np.nan

    r = math.hypot(bx, by - _geo.GOAL_Y) if not math.isnan(bx) else np.nan
    theta = math.atan2(by - _geo.GOAL_Y, bx) if not math.isnan(bx) else np.nan
    speed = math.hypot(bvx, bvy) if not math.isnan(bvx) else np.nan

    # ADR-019: dtype-safe team-id identity (Int64(2) vs "2" must match). One computed
    # mask, three consumers; ids_match resolves NA to False, so NaN-team rows land in
    # ``attacking`` via the plain complement -- byte-matching the legacy ``!=``.
    is_gk_team = ids_match(players["team_id"], gk_team_id)
    defending = players[is_gk_team & (~players_is_gk)]
    attacking = players[~is_gk_team]
    gk_rows = players[is_gk_team & players_is_gk]

    def_xy = (
        np.column_stack(
            [
                defending["x"].to_numpy(dtype=float),
                defending["y"].to_numpy(dtype=float),
            ]
        )
        if len(defending)
        else np.empty((0, 2))
    )
    atk_xy = (
        np.column_stack(
            [
                attacking["x"].to_numpy(dtype=float),
                attacking["y"].to_numpy(dtype=float),
            ]
        )
        if len(attacking)
        else np.empty((0, 2))
    )

    open_goal = _open_goal_fraction((bx, by), def_xy)

    if len(gk_rows) > 0:
        gkx = float(gk_rows["x"].iloc[0])
        gky = float(gk_rows["y"].iloc[0])
        gk_r = math.hypot(gkx, gky - _geo.GOAL_Y)
        gk_theta = math.atan2(gky - _geo.GOAL_Y, gkx)
    else:
        gk_r = gk_theta = np.nan

    ddist, dang = _nearest_k((bx, by), def_xy)
    odist, oang = _nearest_k((bx, by), atk_xy)

    values: dict[str, float] = {
        "r": r,
        "theta": theta,
        "z": bz,
        "speed": speed,
        "openGoal": open_goal,
        "GK_r": gk_r,
        "GK_theta": gk_theta,
    }
    for k in range(5):
        values[f"DefDist_{k}"] = ddist[k]
        values[f"DefAngle_{k}"] = dang[k]
        values[f"OffDist_{k}"] = odist[k]
        values[f"OffAngle_{k}"] = oang[k]

    # Feature-set-driven final assembly of ONE computation (DRY): position_only selects the same
    # `values` minus `speed`; a velocity-less frame leaves `speed` NaN but it is simply not selected.
    names = XSHOT_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XSHOT_FEATURE_NAMES_FAITHFUL
    return pd.DataFrame([[values[c] for c in names]], columns=names)


@overload
def prepare_xshot_training_data(
    frames: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    home_team_id: int | str,
    feature_set: XShotFeatureSet = ...,
    horizon_seconds: float = ...,
    attacking_third_only: bool = ...,
    shot_types: tuple[str, ...] | None = ...,
    carrier_params: dict | None = ...,
    return_keys: Literal[False] = ...,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]: ...


@overload
def prepare_xshot_training_data(
    frames: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    home_team_id: int | str,
    feature_set: XShotFeatureSet = ...,
    horizon_seconds: float = ...,
    attacking_third_only: bool = ...,
    shot_types: tuple[str, ...] | None = ...,
    carrier_params: dict | None = ...,
    return_keys: Literal[True],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]: ...


def prepare_xshot_training_data(
    frames: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    home_team_id: int | str,  # reserved for symmetry with add_*/compute_*
    feature_set: XShotFeatureSet = "faithful",
    horizon_seconds: float = 1.0,
    attacking_third_only: bool = True,
    shot_types: tuple[str, ...] | None = None,
    carrier_params: dict | None = None,
    return_keys: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build (features, labels, groups) for one match's frames -- the shared,
    public train/serve-parity entry point (spec §3.5).

    Applies the paper's data-curation domain filter (spec §4): keep only frames
    where (a) ``ball_state == "alive"``, (b) a possession team is resolvable via
    ``infer_ball_carrier`` -> ``derive_team_in_possession`` (carrier params from
    ``carrier_params``, default the library default), and (c) when
    ``attacking_third_only`` (default True) the ball is in the in-possession
    team's attacking third. Labels come from :func:`build_xshot_labels` over
    ``shots`` filtered to ``shot_types`` (default {shot, shot_penalty,
    shot_freekick}). ``groups`` is ``game_id`` for match-stratified GroupKFold.

    Always returns the **faithful** class distribution (no subsampling): this is the
    train/serve-parity entry point, so contaminating it with a class-balance change
    would silently distort any downstream CV's eval folds + base-rate baselines. For
    wall-clock/memory control on large corpora, apply :func:`subsample_negatives` to a
    **training split only** (never the held-out/eval fold). [PR-S80: moved out of here.]

    Parameters
    ----------
    frames : pd.DataFrame
        Canonical tracking frames for one match.
    shots : pd.DataFrame
        Shot actions; columns ``game_id``/``period_id``/``team_id``/
        ``time_seconds`` (+ optionally ``type_name`` for shot-type filtering).
    home_team_id : int | str
        Accepted for call-site symmetry with the ``add_*``/``compute_*`` surfaces.

    Returns
    -------
    (features, labels, groups)
        ``features`` is a (n, 27) DataFrame; ``labels`` int 0/1; ``groups`` the
        per-row ``game_id``. With ``return_keys=True`` a 4th element is appended: a
        ``(game_id, period_id, frame_id)`` DataFrame row-aligned to ``features`` (the
        selected-frame identity, for keyed comparison across two feature computations;
        purely additive -- the default 3-tuple path is byte-identical).

    Examples
    --------
    >>> # X, y, groups = prepare_xshot_training_data(frames, shots, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    if feature_set == "extended":
        raise NotImplementedError("feature_set='extended' is not implemented; use 'faithful' or 'position_only'.")
    cp: dict = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
    types = _DEFAULT_SHOT_TYPES if shot_types is None else tuple(shot_types)

    work = frames
    if "ball_state" in frames.columns:
        work = frames[frames["ball_state"].astype(str) == "alive"]
    if len(work) == 0:
        empty = pd.DataFrame(
            columns=(
                XSHOT_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XSHOT_FEATURE_NAMES_FAITHFUL
            )
        )
        if return_keys:
            return (
                empty,
                np.zeros(0, dtype=int),
                np.zeros(0),
                pd.DataFrame(columns=["game_id", "period_id", "frame_id"]),
            )
        return empty, np.zeros(0, dtype=int), np.zeros(0)

    carrier = infer_ball_carrier(work, **cp)
    poss = derive_team_in_possession(work, carrier)
    goal_map = resolve_defended_goals(work)

    feat_rows: list[pd.DataFrame] = []
    labels_idx: list[dict] = []
    key_rows: list[dict] = []  # ADDITIVE (return_keys): (game_id, period_id, frame_id) per selected row
    for (gid, pid, fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
        tip = grp["team_in_possession"].iloc[0]
        if pd.isna(tip):
            continue
        outfield = grp[~grp["is_ball"].astype(bool)]
        others = [t for t in outfield["team_id"].dropna().unique() if t != tip]
        if not others:
            continue
        def_team = others[0]
        goal_x = goal_map.get(gid, pid, def_team, allow_guess=True)
        if goal_x is None:
            continue
        if attacking_third_only:
            ball = grp[grp["is_ball"].astype(bool)]
            ball_x = float(ball["x"].iloc[0]) if len(ball) else float("nan")
            if not _ball_in_attacking_third(ball_x, goal_x):
                continue
        feat_rows.append(extract_xshot_features(grp, gk_team_id=def_team, goal_x=goal_x, feature_set=feature_set))
        labels_idx.append(
            {
                "game_id": gid,
                "period_id": pid,
                "time_seconds": float(grp["time_seconds"].iloc[0]),
                "team_in_possession": tip,
            }
        )
        if return_keys:
            key_rows.append({"game_id": gid, "period_id": pid, "frame_id": fid})

    # feature_set-appropriate column set -- "extended" is guarded out above, so this is
    # faithful/position_only. Single-sourced so the empty-path and the concat-select cannot drift
    # (a FAITHFUL select on a position_only frame raises KeyError: ['speed'] -- caught by the smoke).
    _names = XSHOT_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else XSHOT_FEATURE_NAMES_FAITHFUL

    if not feat_rows:
        empty = pd.DataFrame(columns=_names)
        if return_keys:
            return (
                empty,
                np.zeros(0, dtype=int),
                np.zeros(0),
                pd.DataFrame(columns=["game_id", "period_id", "frame_id"]),
            )
        return empty, np.zeros(0, dtype=int), np.zeros(0)

    features = pd.concat(feat_rows, ignore_index=True)[_names]
    fidx = pd.DataFrame(labels_idx)

    # ADR-126: the shot-type filter lives in build_xshot_labels (by type_id); pass the raw table.
    labels = build_xshot_labels(fidx, shots, horizon_seconds=horizon_seconds, shot_types=types).to_numpy()
    groups = fidx["game_id"].to_numpy()
    if return_keys:
        return features, labels, groups, pd.DataFrame(key_rows, columns=["game_id", "period_id", "frame_id"])
    return features, labels, groups


def compute_xshot_occurrence(
    frames: pd.DataFrame,
    *,
    model: XShotOccurrenceModel | str | None = None,
    home_team_id: int | str | None = None,  # unused (goal resolved GK-based); kept for call symmetry
    pitch_control_cache=None,  # reserved for 'extended' (not used by 'faithful')
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Add an ``xshot_occurrence`` column (P(shot within ~1 s)) per in-possession frame.

    Possession + defended goal are resolved with the carrier params stored in the
    model's metadata (R3), so serve-time selection matches training. Rows whose
    state is undefined get NaN. ``pitch_control_cache`` is accepted for forward
    compat with the (deferred) 'extended' variant but is valid only for canonical
    frames --- counterfactual callers must omit it. See NOTICE / spec §7.

    Examples
    --------
    >>> # out = compute_xshot_occurrence(frames, model=m, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    out = frames.copy()
    out["xshot_occurrence"] = np.nan

    # VELOCITY-AVAILABILITY CONTRACT (ADR-054) + velocity-keyed auto-select, at the SHARED seam. All
    # three public entry points reach scoring through here -- `add_xshot_occurrence`,
    # `xshot_occurrence_xfns` and a direct call.
    #
    # MIXED availability (some-but-not-all rows declare velocity unavailable) is a caller error: it
    # would otherwise resolve to the default velocity model and fabricate speed=NaN on the marked rows
    # (M3). Raise BEFORE the other prongs.
    if _velocity_availability_is_mixed(frames):
        raise ValueError(
            "compute_xshot_occurrence: mixed velocity-availability -- some rows declare "
            f"speed_source={SPEED_SOURCE_UNAVAILABLE!r} and some do not. Pass a single-availability "
            "frame set (all freeze-frame, or all velocity-bearing)."
        )
    # UNDECLARED missing vx/vy (NOT the declared freeze-frame shape): the "forgot derive_velocities()"
    # case. Fail loud (ADR-054 prong, preserved) -- declared-unavailable frames proceed to the
    # resolver below and get the position-only variant.
    if (
        not _velocity_unavailable_by_design(frames)
        and len(frames)
        and ("vx" not in frames.columns or "vy" not in frames.columns)
    ):
        raise ValueError(
            "compute_xshot_occurrence requires vx/vy on frames (call derive_velocities() first), or "
            f"declare speed_source {SPEED_SOURCE_UNAVAILABLE!r}. See the velocity-availability contract."
        )
    # Layer B: override -> "custom"; declared-unavailable -> position_only (or NaN if unbundled, NEVER
    # the default velocity model); else default. `compute_*` output stays byte-identical (no provenance
    # column here -- the byte-identical constraint for velocity-bearing direct callers); the public
    # `xshot_occurrence_variant` column is added by `add_xshot_occurrence`, which RE-RESOLVES the key
    # (deterministic; `from_variant` is cached), leaving the xfns/VAEP path numeric.
    m, _variant = _resolve_xshot_model_for_frames(frames, model)
    if m is None:
        return out  # declared-unavailable but no bundled position_only variant -> honest NaN

    # N-A: carrier inference + possession MUST run on the FULL contiguous frames.
    # infer_ball_carrier has a CROSS-FRAME dependency (gamma hysteresis carries the
    # incumbent carrier across consecutive frames within a (game, period) segment).
    # Running it on the scattered link_frame_ids subset would yield a different
    # carrier -> different team_in_possession, AND diverge from the trainer (which
    # runs it on full frames) -> train/serve skew (R3). So: carrier on full frames;
    # restrict ONLY the per-frame extract + batched predict (the expensive part).
    carrier = infer_ball_carrier(frames, **m.carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = resolve_defended_goals(frames)

    # Pass 1: build ONE feature row per target frame (extraction stays per-frame;
    # the XGBoost predict must NOT be --- P1). Restrict to link_frame_ids HERE.
    feat_rows: list[pd.DataFrame] = []
    keys: list[tuple] = []  # (gid, pid, frame_id, team_in_possession)
    n_groups = 0
    n_skipped_goal = 0  # N1: count frames dropped for a missing goal map entry
    for (gid, pid, frame_id), grp in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
        if link_frame_ids is not None and int(str(frame_id)) not in link_frame_ids:
            continue
        n_groups += 1
        tip = grp["team_in_possession"].iloc[0]
        if pd.isna(tip):
            continue
        # Defending team = a non-ball team that is NOT in possession. Exclude the
        # ball row (sentinel team_id) so it is never mistaken for a defending team.
        outfield = grp[~grp["is_ball"].astype(bool)]
        teams = [t for t in outfield["team_id"].dropna().unique() if t != tip]
        if not teams:
            continue
        def_team = teams[0]
        goal_x = goal_map.get(gid, pid, def_team, allow_guess=True)
        if goal_x is None:
            n_skipped_goal += 1
            continue
        feat_rows.append(extract_xshot_features(grp, gk_team_id=def_team, goal_x=goal_x, feature_set=m.feature_set))
        keys.append((gid, pid, frame_id, tip))

    # N1: surface coverage loss rather than dropping silently.
    if n_skipped_goal and n_groups and n_skipped_goal / n_groups > 0.05:
        warnings.warn(
            f"xshot_occurrence: {n_skipped_goal}/{n_groups} frame-groups skipped "
            f"(no defended-goal resolution); possible GK-identification gap.",
            stacklevel=2,
        )

    if not feat_rows:
        return out

    # Pass 2: ONE batched predict over the stacked matrix, scatter back.
    feature_matrix = pd.concat(feat_rows, ignore_index=True)
    probs = m.predict_proba(feature_matrix)
    key_df = pd.DataFrame(keys, columns=["game_id", "period_id", "frame_id", "team_id"])
    key_df["__p"] = probs
    # N-B: join on TEMPORARY string keys so we never mutate out["game_id"]/["team_id"]
    # dtypes (preserve the TRACKING_FRAMES schema --- "add one column, change nothing else").
    out["__gid"] = out["game_id"].astype(str)
    out["__tid"] = out["team_id"].astype(str)
    key_df["__gid"] = key_df["game_id"].astype(str)
    key_df["__tid"] = key_df["team_id"].astype(str)
    key_df = key_df.drop(columns=["game_id", "team_id"])
    out = out.merge(key_df, on=["__gid", "period_id", "frame_id", "__tid"], how="left")
    out["xshot_occurrence"] = out["__p"]
    return out.drop(columns=["__p", "__gid", "__tid"])
