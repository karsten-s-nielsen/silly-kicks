"""The oracle of the ball-carrier pre-index (HFU fit-units spec 4.3).

`pre_index_frames_oracle` is `silly_kicks.tracking._ball_carrier._pre_index_frames` as it was at `bb0c601c`
(4.140.0), copied VERBATIM: only the name differs. The rewritten function must equal it on every key
(`assert_pre_index_equal`), and must raise the same exception type where it raises. Do not edit the body.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def pre_index_frames_oracle(
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


def _is_null(v: Any) -> bool:
    """True for a scalar null (None, a float NaN, ``pd.NA``, ``pd.NaT``)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    try:
        return bool(v != v)
    except (TypeError, ValueError):
        return False


def _same(a: Any, b: Any) -> bool:
    """NaN-aware scalar equality: two nulls of the same type are equal; else the same type and ``==``."""
    if _is_null(a) or _is_null(b):
        return _is_null(a) and _is_null(b) and type(a) is type(b)
    return type(a) is type(b) and bool(a == b)


def _same_seq(a: Any, b: Any) -> bool:
    return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))


def assert_pre_index_equal(old: dict, new: dict) -> None:
    """The spec 4.3 equality rule over two `_pre_index_frames` results; raises ``AssertionError`` on a difference.

    Arrays: the same dtype and shape. A float array is equal BY BITS (so NaN equals NaN and -0.0 differs
    from 0.0: the rewrite only places values, it computes none). An object array is compared element by
    element with the scalar rule; every other array element by element. Lists and dicts (keys and values):
    equal in order with the same rule. Every other value (dtypes, the velocity flag): ``type`` and ``==``.
    """
    assert list(old) == list(new), f"keys differ: {list(old)} != {list(new)}"
    for key in old:
        a, b = old[key], new[key]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), f"{key}: one side is not an array"
            assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
            assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
            if a.dtype == object:
                assert _same_seq(a.ravel().tolist(), b.ravel().tolist()), f"{key}: object elements differ"
            elif a.dtype.kind in "fc":
                assert a.tobytes() == b.tobytes(), f"{key}: elements differ (compared by bits)"
            else:
                assert np.array_equal(a, b), f"{key}: elements differ"
        elif isinstance(a, dict):
            assert isinstance(b, dict), f"{key}: one side is not a dict"
            assert _same_seq(list(a), list(b)), f"{key}: dict keys differ"
            assert _same_seq(list(a.values()), list(b.values())), f"{key}: dict values differ"
        elif isinstance(a, list):
            assert isinstance(b, list), f"{key}: one side is not a list"
            assert _same_seq(a, b), f"{key}: list entries differ"
        else:
            assert type(a) is type(b) and a == b, f"{key}: {a!r} != {b!r}"
