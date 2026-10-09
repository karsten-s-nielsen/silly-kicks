"""smooth_frames -- Savitzky-Golay or EMA smoothing of player/ball positions.

References
----------
Savitzky, A., & Golay, M. J. E. (1964). "Smoothing and Differentiation of Data
by Simplified Least Squares Procedures." Analytical Chemistry, 36(8), 1627-1639.

See NOTICE for full bibliographic citation.

TF-65 §4.1b -- two gap-fill stages, complementary not overlapping: this function's internal dense-grid
reindex handles MISSING ROWS (non-detections); the standalone ``interpolate_frames`` fills NaN-position
runs BETWEEN existing rows. Both honour the same ``max_gap_seconds`` cap. Default-pipeline order is
``interpolate_frames`` -> ``smooth_frames``/``derive_velocities``; the reindex is idempotent w.r.t.
already-filled frames, so there is no double-fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._config_dataclass import PreprocessConfig
from ._densify import bridge_small, densify_group, segments

# game_id in the key so a two-game frame never smooths one entity's series across the game boundary (review A-31;
# the defect fixed in resample_frames this cycle). Single-game frames are byte-identical (game_id is constant).
_GROUP_KEYS = ["game_id", "period_id", "is_ball", "player_id"]


def _provenance_tag(config: PreprocessConfig, method_used: str) -> str:
    # butterworth gets its OWN tag so the savgol/ema tags stay byte-identical (Hyrum's law).
    if method_used == "butterworth":
        return f"method=butterworth|bw_cutoff_hz={config.butterworth_cutoff_hz}|bw_order={config.butterworth_order}"
    return (
        f"method={method_used}|sg_window_s={config.sg_window_seconds}|"
        f"sg_poly={config.sg_poly_order}|ema_alpha={config.ema_alpha}"
    )


def _butterworth_per_group(values: np.ndarray, hz: float, config: PreprocessConfig) -> np.ndarray:
    # Mirror _savgol_per_group's NaN handling: interior NaN bridged, filtered, then restored. A group shorter
    # than sosfiltfilt's pad passes through unchanged.
    from ._butterworth import butterworth_lowpass, butterworth_min_length

    if len(values) < butterworth_min_length(hz, config.butterworth_cutoff_hz, config.butterworth_order):
        return values.copy()
    nan_idx = np.flatnonzero(np.isnan(values))
    valid_idx = np.flatnonzero(~np.isnan(values))
    if len(valid_idx) == 0:
        return values.copy()
    out = values.copy()
    if len(nan_idx) > 0:
        idx = np.arange(len(values))
        out[nan_idx] = np.interp(idx[nan_idx], idx[valid_idx], values[valid_idx])
    smoothed = butterworth_lowpass(out, hz, config.butterworth_cutoff_hz, config.butterworth_order)
    if len(nan_idx) > 0:
        smoothed[nan_idx] = np.nan
    return smoothed


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


def _smooth_one(
    values: np.ndarray, method: str, window: int, poly: int, cfg: PreprocessConfig, hz: float
) -> np.ndarray:
    """Smooth a single contiguous (bridged) segment via the chosen method."""
    if method == "savgol":
        return _savgol_per_group(values, window, poly)
    if method == "ema":
        return _ema_per_group(values, cfg.ema_alpha)
    if method == "butterworth":
        return _butterworth_per_group(values, hz, cfg)
    raise ValueError(f"smooth_frames: unsupported method={method!r}")


def _smooth_positions_dense(
    frame_ids: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    method: str,
    window: int,
    poly: int,
    cfg: PreprocessConfig,
    hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth one group on its dense ``frame_id`` grid (TF-65 §4.1).

    Densify -> bridge small gaps -> smooth per segment (split at runs > max_gap) -> map back to the
    group's rows by ``frame_id`` (duplicate-frame-safe). Kalman handles gaps natively (whole-series).
    Byte-identical to the row-order path on a contiguous single-detection-run group.
    """
    g = pd.DataFrame(
        {"frame_id": np.asarray(frame_ids), "x": np.asarray(x_vals, float), "y": np.asarray(y_vals, float)}
    )
    full, xd, yd, _real = densify_group(g)
    f0 = int(full[0])
    dense_idx = np.asarray(frame_ids, dtype=int) - f0
    if method == "kalman":
        from ._kalman import kalman_ca

        dt = 1.0 / hz
        xs = kalman_ca(xd, dt, cfg).pos
        ys = kalman_ca(yd, dt, cfg).pos
        return xs[dense_idx], ys[dense_idx]
    maxg = round(cfg.max_gap_seconds * hz)
    valid = ~(np.isnan(xd) | np.isnan(yd))
    xs = np.full(len(xd), np.nan)
    ys = np.full(len(yd), np.nan)
    for s, e in segments(valid, maxg):
        xb = bridge_small(xd[s:e], maxg)
        yb = bridge_small(yd[s:e], maxg)
        sv = valid[s:e]
        xss = np.asarray(_smooth_one(xb, method, window, poly, cfg, hz), dtype=float).copy()
        yss = np.asarray(_smooth_one(yb, method, window, poly, cfg, hz), dtype=float).copy()
        xss[~sv] = np.nan
        yss[~sv] = np.nan
        xs[s:e] = xss
        ys[s:e] = yss
    return xs[dense_idx], ys[dense_idx]


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
    method : {"savgol", "ema", "butterworth", "kalman"} or None
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

    sort_cols = ["game_id", "period_id", "is_ball", "player_id", "frame_id"]  # game_id first (A-31)
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

    if method_used not in ("savgol", "ema", "butterworth", "kalman"):
        raise ValueError(f"smooth_frames: unsupported method={method_used!r}")

    # Each group is smoothed on its dense frame_id grid (TF-65 §4.1): a non-detection becomes a
    # NaN-position run the max_gap cap handles, and segments split at runs > max_gap so no window
    # spans a big gap. Contiguous single-detection-run groups are byte-identical to the old row path.
    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        fids = sorted_frames.loc[idx_arr, "frame_id"].to_numpy()
        x_vals = sorted_frames.loc[idx_arr, "x"].to_numpy(dtype=float)
        y_vals = sorted_frames.loc[idx_arr, "y"].to_numpy(dtype=float)
        xs, ys = _smooth_positions_dense(fids, x_vals, y_vals, method_used, window_frames, cfg.sg_poly_order, cfg, hz)
        x_smoothed[idx_arr] = xs
        y_smoothed[idx_arr] = ys

    # F1b (ADR-106): smoothed-position STORAGE is float32, matching the float32 coordinate columns.
    sorted_frames["x_smoothed"] = x_smoothed.astype(np.float32)
    sorted_frames["y_smoothed"] = y_smoothed.astype(np.float32)

    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    # ADR-103 F1a: `category` -- a match-constant string across every row (the single biggest object
    # column, ~210 MB on a GS half). Static/set-once (only re-stamped whole-column by an idempotent
    # re-run; reflect leaves it as an "invariant" kind), so `category` is safe here (unlike the dynamic
    # team_attacking_direction/speed_source/visibility). See feedback_category_dtype_only_for_static_columns.
    sorted_frames["_preprocessed_with"] = pd.Categorical([tag] * len(sorted_frames))
    sorted_frames.attrs["preprocess"] = tag
    return sorted_frames
