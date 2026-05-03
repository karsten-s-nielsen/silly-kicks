"""smooth_frames -- Savitzky-Golay or EMA smoothing of player/ball positions.

References
----------
Savitzky, A., & Golay, M. J. E. (1964). "Smoothing and Differentiation of Data
by Simplified Least Squares Procedures." Analytical Chemistry, 36(8), 1627-1639.

See NOTICE for full bibliographic citation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._config_dataclass import PreprocessConfig

_GROUP_KEYS = ["period_id", "is_ball", "player_id"]


def _provenance_tag(config: PreprocessConfig, method_used: str) -> str:
    return (
        f"method={method_used}|sg_window_s={config.sg_window_seconds}|"
        f"sg_poly={config.sg_poly_order}|ema_alpha={config.ema_alpha}"
    )


def _savgol_per_group(values: np.ndarray, window_frames: int, poly_order: int) -> np.ndarray:
    if len(values) < window_frames or window_frames < poly_order + 2:
        return values.copy()  # too short -- pass through
    # Use integer-index assignment via np.flatnonzero so pyright's stricter numpy stubs
    # accept the SetIndex argument (bool-mask NDArray[Any] is not assignable to SetIndex
    # in numpy 2.x stubs; integer-array indexing is always assignable).
    nan_idx = np.flatnonzero(np.isnan(values))
    valid_idx = np.flatnonzero(~np.isnan(values))
    out = values.copy()
    if len(valid_idx) == 0:
        return out
    if len(nan_idx) > 0:
        idx = np.arange(len(values))
        out[nan_idx] = np.interp(idx[nan_idx], idx[valid_idx], values[valid_idx])
    # np.asarray cast pins the savgol_filter return type so pyright sees a concrete
    # NDArray rather than the union it infers from scipy stubs.
    smoothed: np.ndarray = np.asarray(
        savgol_filter(out, window_length=window_frames, polyorder=poly_order), dtype=np.float64
    )
    if len(nan_idx) > 0:
        smoothed[nan_idx] = np.nan
    return smoothed


def _ema_per_group(values: np.ndarray, alpha: float) -> np.ndarray:
    nan_idx = np.flatnonzero(np.isnan(values))
    # Newer pandas Series.ewm(...).to_numpy() returns a read-only view on Python 3.11+;
    # explicit copy makes the result writeable for the NaN-restore step below.
    out = np.array(pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy(), copy=True)
    if len(nan_idx) > 0:
        out[nan_idx] = np.nan
    return out


def smooth_frames(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Smooth player/ball position columns; emit additive ``x_smoothed``/``y_smoothed``.

    Raw ``x``/``y`` columns are preserved unchanged. The chosen method + key
    parameters are recorded in a per-row ``_preprocessed_with`` column.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS.
    config : PreprocessConfig or None
        Smoothing config. Defaults to ``PreprocessConfig.default()``.
    method : {"savgol", "ema"} or None
        Override ``config.smoothing_method`` for this call.

    Returns
    -------
    pd.DataFrame
        Frames with additional ``x_smoothed``, ``y_smoothed``, ``_preprocessed_with``
        columns. Original ``x``/``y`` are bit-identical to the input.

    Idempotent: a re-call with the same config returns equal output (detected via
    the existing ``_preprocessed_with`` column).

    Examples
    --------
    >>> # See tests/test_smooth_frames.py for runnable example.
    """
    cfg = config or PreprocessConfig.default()
    method_used = method or cfg.smoothing_method or "savgol"
    tag = _provenance_tag(cfg, method_used)

    if "_preprocessed_with" in frames.columns and (frames["_preprocessed_with"] == tag).all():
        out = frames.copy()
        if "x_smoothed" in out.columns and "y_smoothed" in out.columns:
            return out

    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    sorted_frames = frames.sort_values(sort_cols, kind="mergesort").reset_index()
    original_index = sorted_frames["index"].to_numpy()
    sorted_frames = sorted_frames.drop(columns="index")

    hz = float(sorted_frames["frame_rate"].dropna().iloc[0]) if "frame_rate" in sorted_frames.columns else 25.0
    # SG requires odd window_length >= poly_order + 2.
    # `int(round(x)) | 1` forces odd, but `max(odd, even)` can still yield even
    # when poly_order + 2 (the lower bound) is even -- re-odd-ify after the max.
    window_frames = max(round(cfg.sg_window_seconds * hz) | 1, cfg.sg_poly_order + 2)
    if window_frames % 2 == 0:
        window_frames += 1

    x_smoothed = np.full(len(sorted_frames), np.nan)
    y_smoothed = np.full(len(sorted_frames), np.nan)

    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        x_vals = sorted_frames.loc[idx_arr, "x"].to_numpy(dtype=float)
        y_vals = sorted_frames.loc[idx_arr, "y"].to_numpy(dtype=float)
        if method_used == "savgol":
            x_smoothed[idx_arr] = _savgol_per_group(x_vals, window_frames, cfg.sg_poly_order)
            y_smoothed[idx_arr] = _savgol_per_group(y_vals, window_frames, cfg.sg_poly_order)
        elif method_used == "ema":
            x_smoothed[idx_arr] = _ema_per_group(x_vals, cfg.ema_alpha)
            y_smoothed[idx_arr] = _ema_per_group(y_vals, cfg.ema_alpha)
        else:
            raise ValueError(f"smooth_frames: unsupported method={method_used!r}")

    sorted_frames["x_smoothed"] = x_smoothed
    sorted_frames["y_smoothed"] = y_smoothed

    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    sorted_frames["_preprocessed_with"] = tag
    sorted_frames.attrs["preprocess"] = tag
    return sorted_frames
