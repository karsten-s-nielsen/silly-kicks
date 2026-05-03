"""interpolate_frames -- linear NaN gap-fill up to max_gap_seconds.

Lakehouse review N3: only ``method="linear"`` is supported in PR-S24. Cubic
ships in TF-9-cubic when a concrete consumer asks (would use
``scipy.interpolate.CubicSpline``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._config_dataclass import PreprocessConfig

_GROUP_KEYS = ["period_id", "is_ball", "player_id"]


def _interp_per_group(
    times: np.ndarray, x: np.ndarray, y: np.ndarray, max_gap_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation only (N3 fix from review)."""
    nan_mask_x = np.isnan(x)
    nan_mask_y = np.isnan(y)
    if not (nan_mask_x.any() or nan_mask_y.any()):
        return x.copy(), y.copy()
    out_x, out_y = x.copy(), y.copy()

    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 2:
        return out_x, out_y

    nan_mask_combined = ~valid
    change = np.diff(nan_mask_combined.astype(np.int8), prepend=0, append=0)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    for s, e in zip(starts, ends, strict=False):
        if s == 0 or e >= len(x):
            continue  # gap touches an endpoint; no anchors on both sides
        gap_seconds = float(times[e] - times[s - 1])
        if gap_seconds > max_gap_seconds:
            continue
        x_left, x_right = x[s - 1], x[e]
        y_left, y_right = y[s - 1], y[e]
        if np.isnan(x_left) or np.isnan(x_right) or np.isnan(y_left) or np.isnan(y_right):
            continue
        # Linearly interpolate this run from the anchor times[s-1], times[e]
        t_run = times[s:e]
        denom = max(float(times[e] - times[s - 1]), 1e-9)
        frac = (t_run - times[s - 1]) / denom
        out_x[s:e] = x_left + frac * (x_right - x_left)
        out_y[s:e] = y_left + frac * (y_right - y_left)
    return out_x, out_y


def interpolate_frames(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Fill NaN positional gaps up to ``max_gap_seconds`` via linear interpolation.

    Gaps longer than ``max_gap_seconds`` remain NaN (no fabrication).
    Idempotent on already-filled frames.

    Only ``method="linear"`` is supported in PR-S24 (lakehouse review N3).
    Cubic interpolation will ship as TF-9-cubic when a concrete consumer asks,
    via ``scipy.interpolate.CubicSpline``.

    Examples
    --------
    >>> # See tests/test_interpolate_frames.py for runnable example.
    """
    cfg = config or PreprocessConfig.default()
    method_used = method or cfg.interpolation_method or "linear"
    if method_used != "linear":
        raise ValueError(
            f"interpolate_frames: unsupported method={method_used!r}. "
            "Only 'linear' is supported in PR-S24; cubic ships in TF-9-cubic when requested."
        )

    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    sorted_frames = frames.sort_values(sort_cols, kind="mergesort").reset_index()
    original_index = sorted_frames["index"].to_numpy()
    sorted_frames = sorted_frames.drop(columns="index")

    x = sorted_frames["x"].to_numpy(dtype=float, copy=True)
    y = sorted_frames["y"].to_numpy(dtype=float, copy=True)
    t = sorted_frames["time_seconds"].to_numpy(dtype=float)

    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        gx, gy = _interp_per_group(t[idx_arr], x[idx_arr], y[idx_arr], cfg.max_gap_seconds)
        x[idx_arr] = gx
        y[idx_arr] = gy

    sorted_frames["x"] = x
    sorted_frames["y"] = y
    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    return sorted_frames
