"""derive_velocities -- vx/vy/speed columns via Savitzky-Golay derivative.

PR-S24 lakehouse review S4: requires smoothed positions on input. Raises
``ValueError`` if ``_preprocessed_with``/``x_smoothed``/``y_smoothed`` are
absent -- principle-of-least-surprise (no hidden schema mutation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._config_dataclass import PreprocessConfig

_GROUP_KEYS = ["period_id", "is_ball", "player_id"]


def derive_velocities(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """Add ``vx``, ``vy``, ``speed`` columns from smoothed positions.

    REQUIRES ``_preprocessed_with`` (and ``x_smoothed``/``y_smoothed``) on
    ``frames`` -- call :func:`smooth_frames` first. Lakehouse-review S4 fix:
    earlier drafts auto-invoked ``smooth_frames`` here, but that meant a
    caller asking for vx/vy/speed got back a DataFrame with FIVE new columns
    instead of the documented three. Loud raise is the principle-of-least-surprise
    choice.

    Output schema additions: ``vx``, ``vy``, ``speed`` (all float64, m/s).
    No other columns are added or mutated.

    Examples
    --------
    >>> # See tests/test_derive_velocities.py for runnable example.
    """
    cfg = config or PreprocessConfig.default()
    missing = [c for c in ("_preprocessed_with", "x_smoothed", "y_smoothed") if c not in frames.columns]
    if missing:
        raise ValueError(
            f"derive_velocities: frames missing required column(s) {missing}. "
            "Call silly_kicks.tracking.preprocess.smooth_frames(frames, ...) first."
        )

    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    sorted_frames = frames.sort_values(sort_cols, kind="mergesort").reset_index()
    original_index = sorted_frames["index"].to_numpy()
    sorted_frames = sorted_frames.drop(columns="index")

    hz = float(sorted_frames["frame_rate"].dropna().iloc[0]) if "frame_rate" in sorted_frames.columns else 25.0
    dt = 1.0 / hz
    window_frames = max(round(cfg.sg_window_seconds * hz) | 1, cfg.sg_poly_order + 2)
    if window_frames % 2 == 0:
        window_frames += 1

    # Smoothed columns guaranteed present by the up-front check above (S4 fix).
    x_src = sorted_frames["x_smoothed"].to_numpy(dtype=float)
    y_src = sorted_frames["y_smoothed"].to_numpy(dtype=float)
    vx = np.full(len(sorted_frames), np.nan)
    vy = np.full(len(sorted_frames), np.nan)

    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        x_vals = x_src[idx_arr]
        y_vals = y_src[idx_arr]
        if len(x_vals) < window_frames:
            x_safe = np.where(np.isnan(x_vals), 0.0, x_vals)
            y_safe = np.where(np.isnan(y_vals), 0.0, y_vals)
            vx_g = np.gradient(x_safe, dt)
            vy_g = np.gradient(y_safe, dt)
            mask = np.isnan(x_vals) | np.isnan(y_vals)
            vx_g[mask] = np.nan
            vy_g[mask] = np.nan
            vx[idx_arr] = vx_g
            vy[idx_arr] = vy_g
            continue

        # Use integer-index assignment via np.flatnonzero so pyright's stricter numpy stubs
        # accept the SetIndex argument (bool-mask NDArray[Any] is not assignable to SetIndex
        # in numpy 2.x stubs).
        nan_mask = np.isnan(x_vals) | np.isnan(y_vals)
        nan_idx = np.flatnonzero(nan_mask)
        if (~nan_mask).any():
            x_filled = np.where(
                np.isnan(x_vals),
                np.interp(
                    np.arange(len(x_vals)),
                    np.arange(len(x_vals))[~np.isnan(x_vals)],
                    x_vals[~np.isnan(x_vals)],
                )
                if (~np.isnan(x_vals)).any()
                else 0.0,
                x_vals,
            )
            y_filled = np.where(
                np.isnan(y_vals),
                np.interp(
                    np.arange(len(y_vals)),
                    np.arange(len(y_vals))[~np.isnan(y_vals)],
                    y_vals[~np.isnan(y_vals)],
                )
                if (~np.isnan(y_vals)).any()
                else 0.0,
                y_vals,
            )
        else:
            x_filled = x_vals
            y_filled = y_vals
        vx_g: np.ndarray = np.asarray(
            savgol_filter(x_filled, window_length=window_frames, polyorder=cfg.sg_poly_order, deriv=1, delta=dt),
            dtype=np.float64,
        )
        vy_g: np.ndarray = np.asarray(
            savgol_filter(y_filled, window_length=window_frames, polyorder=cfg.sg_poly_order, deriv=1, delta=dt),
            dtype=np.float64,
        )
        if len(nan_idx) > 0:
            vx_g[nan_idx] = np.nan
            vy_g[nan_idx] = np.nan
        vx[idx_arr] = vx_g
        vy[idx_arr] = vy_g

    sorted_frames["vx"] = vx
    sorted_frames["vy"] = vy
    sorted_frames["speed"] = np.sqrt(vx * vx + vy * vy)
    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    return sorted_frames
