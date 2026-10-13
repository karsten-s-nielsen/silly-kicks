"""Dense-grid reindex machinery for the smoother/velocity path (TF-65 §4.1).

A tracking frame is one row per DETECTED frame; a non-detection is a MISSING ROW, not a
NaN-position row. Row-order smoothing therefore treats the last pre-gap and first post-gap
detections as adjacent and fabricates a through-gap velocity. These helpers reindex each
``(game_id, period_id, is_ball, player_id)`` group to its contiguous ``frame_id`` range so a
non-detection becomes a NaN-position run the existing ``max_gap_seconds`` policy handles, then
``segments`` splits the series at runs longer than the cap so no filter window spans a big gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def densify_group(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reindex one group to its contiguous ``frame_id`` range.

    Duplicate ``frame_id`` within the group (GS emits them) are de-duplicated **keep-first**
    (TF-65 §4.1, owner-approved; matches ``_elastic_sync.py`` / ``features.py`` precedent). Missing
    frames become NaN-position rows.

    Returns ``(full_frame_ids, x_dense, y_dense, real_mask)`` where ``real_mask`` marks the rows that
    existed in the input (the rest are the inserted NaN-position rows).
    """
    g = g.drop_duplicates("frame_id")
    f = g["frame_id"].to_numpy()
    full = np.arange(int(f.min()), int(f.max()) + 1)
    gi = g.set_index("frame_id")
    x = gi["x"].reindex(full).to_numpy(dtype=float)
    y = gi["y"].reindex(full).to_numpy(dtype=float)
    real = np.isin(full, f)
    return full, x, y, real


def segments(isvalid: np.ndarray, max_gap_frames: int) -> list[tuple[int, int]]:
    """Dense-grid ``[start, end)`` ranges split at NaN runs longer than ``max_gap_frames``.

    Each returned segment starts and ends on a valid row; interior NaN runs are ``<= max_gap_frames``.
    """
    n = len(isvalid)
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        while i < n and not isvalid[i]:
            i += 1
        if i >= n:
            break
        start = i
        gap = 0
        last_valid = i
        while i < n:
            if isvalid[i]:
                last_valid = i
                gap = 0
            else:
                gap += 1
                if gap > max_gap_frames:
                    break
            i += 1
        out.append((start, last_valid + 1))
    return out


def bridge_small(values: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linear-bridge interior NaN runs ``<= max_gap_frames``; leave longer runs NaN.

    Uses row-index interpolation within the (already dense) grid. Runs touching an endpoint are
    left untouched (no anchor on one side).
    """
    out = values.copy()
    isnan = np.isnan(out)
    if not isnan.any():
        return out
    n = len(out)
    idx = np.arange(n)
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap_frames:
                out[i:j] = np.interp(idx[i:j], [i - 1, j], [out[i - 1], out[j]])
            i = j
        else:
            i += 1
    return out
