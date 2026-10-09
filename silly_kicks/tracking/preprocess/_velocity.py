"""derive_velocities -- vx/vy/speed columns via Savitzky-Golay derivative.

PR-S24 lakehouse review S4: requires smoothed positions on input. Raises
``ValueError`` if ``_preprocessed_with``/``x_smoothed``/``y_smoothed`` are
absent -- principle-of-least-surprise (no hidden schema mutation).

TF-65 §4.1b -- the internal dense-grid reindex here handles MISSING ROWS (non-detections); the
standalone ``interpolate_frames`` fills NaN-position runs between existing rows. Same ``max_gap_seconds``
cap, order ``interpolate_frames`` -> smooth/derive, reindex idempotent on filled frames -> no double-fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._config_dataclass import PreprocessConfig
from ._densify import bridge_small, segments
from ._guard import apply_plausibility_guard
from ._kalman import kalman_ca

# game_id in the key so a two-game frame never derives velocity across the game boundary (TF-65 §4.1a;
# mirrors the A-31 fix in _smoothing.py). Single-game frames are byte-identical (game_id is constant).
_GROUP_KEYS = ["game_id", "period_id", "is_ball", "player_id"]


def _parse_method(frames: pd.DataFrame, cfg: PreprocessConfig) -> str:
    """Which smoother produced x_smoothed (from the _preprocessed_with tag), else the config default."""
    if "_preprocessed_with" in frames.columns and len(frames):
        tag = str(frames["_preprocessed_with"].iloc[0])
        for part in tag.split("|"):
            if part.startswith("method="):
                return part.split("=", 1)[1]
    return cfg.smoothing_method or "savgol"


def _derive_group_dense(fids, xs_sm, ys_sm, raw_x, raw_y, method, window, poly, cfg, hz):
    """Per-group velocity/acceleration + per-frame uncertainty on the dense frame grid (TF-65).

    Velocity/acceleration come from the active smoother (savgol derivative, or finite differences of
    a butterworth/ema-smoothed series, or the coherent Kalman state); the uncertainty always comes
    from a constant-acceleration Kalman/RTS pass on the RAW positions. Edge-NaN is applied only at
    gap-adjacent segment boundaries, so a contiguous group's natural start/end is byte-identical to
    the row-order path. Returns arrays aligned to the input rows (duplicate-frame-safe keep-first).
    """
    fids = np.asarray(fids, dtype=int)
    f0, f1 = int(fids.min()), int(fids.max())
    n = f1 - f0 + 1
    _, first_idx = np.unique(fids, return_index=True)
    pos_unique = fids[first_idx] - f0

    def _dense(vals):
        d = np.full(n, np.nan)
        d[pos_unique] = np.asarray(vals, dtype=float)[first_idx]
        return d

    xd_sm, yd_sm = _dense(xs_sm), _dense(ys_sm)
    xd_r, yd_r = _dense(raw_x), _dense(raw_y)
    dt = 1.0 / hz
    maxg = round(cfg.max_gap_seconds * hz)

    # Always-run CA Kalman/RTS on the RAW positions: the uncertainty source (decoupled from the point
    # smoother), and the coherent point estimate when smoothing_method="kalman".
    kx = kalman_ca(xd_r, dt, cfg)
    ky = kalman_ca(yd_r, dt, cfg)
    pos_var = kx.pos_var + ky.pos_var
    vel_var = kx.vel_var + ky.vel_var
    acc_var = kx.acc_var + ky.acc_var

    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    if method == "kalman":
        vx, vy, ax, ay = kx.vel.copy(), ky.vel.copy(), kx.acc.copy(), ky.acc.copy()
    else:
        valid = ~(np.isnan(xd_sm) | np.isnan(yd_sm))
        r = window // 2
        for s, e in segments(valid, maxg):
            length = e - s
            xb = bridge_small(xd_sm[s:e], maxg)
            yb = bridge_small(yd_sm[s:e], maxg)
            sv = valid[s:e]
            if method == "savgol" and length >= window:
                vsx = savgol_filter(xb, window, poly, deriv=1, delta=dt)
                vsy = savgol_filter(yb, window, poly, deriv=1, delta=dt)
                asx = savgol_filter(xb, window, poly, deriv=2, delta=dt)
                asy = savgol_filter(yb, window, poly, deriv=2, delta=dt)
            else:  # butterworth / ema, or a short savgol segment: finite differences
                vsx = np.gradient(xb, dt) if length >= 2 else np.full(length, np.nan)
                vsy = np.gradient(yb, dt) if length >= 2 else np.full(length, np.nan)
                asx = np.gradient(np.gradient(xb, dt), dt) if length >= 3 else np.full(length, np.nan)
                asy = np.gradient(np.gradient(yb, dt), dt) if length >= 3 else np.full(length, np.nan)
            vsx, vsy = np.asarray(vsx, float).copy(), np.asarray(vsy, float).copy()
            asx, asy = np.asarray(asx, float).copy(), np.asarray(asy, float).copy()
            # Edge-NaN ONLY at gap-adjacent boundaries (s>0 / e<n); the group's natural start/end is kept.
            if r > 0 and s > 0:
                for seg in (vsx, vsy, asx, asy):
                    seg[: min(r, length)] = np.nan
            if r > 0 and e < n:
                for seg in (vsx, vsy, asx, asy):
                    seg[max(0, length - r) :] = np.nan
            for seg in (vsx, vsy, asx, asy):
                seg[~sv] = np.nan
            vx[s:e], vy[s:e], ax[s:e], ay[s:e] = vsx, vsy, asx, asy

    dense_idx = fids - f0
    return (
        vx[dense_idx],
        vy[dense_idx],
        ax[dense_idx],
        ay[dense_idx],
        pos_var[dense_idx],
        vel_var[dense_idx],
        acc_var[dense_idx],
    )


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

    Output schema additions (all float32 storage; TF-65): ``vx``, ``vy``, ``speed`` (m/s); ``accel_x``,
    ``accel_y``, ``accel`` (m/s^2); and the always-on constant-acceleration Kalman/RTS uncertainty columns
    ``pos_var``, ``vel_var``, ``accel_var`` (decoupled from the point smoother). Velocity/acceleration
    come from the smoother named in ``_preprocessed_with`` (savgol derivative; butterworth/ema finite
    differences; or the coherent Kalman state); a soft plausibility guard then NaNs implausible values.

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

    sort_cols = ["game_id", "period_id", "is_ball", "player_id", "frame_id"]  # game_id first (TF-65 §4.1a)
    sorted_frames = frames.sort_values(sort_cols, kind="mergesort").reset_index()
    original_index = sorted_frames["index"].to_numpy()
    sorted_frames = sorted_frames.drop(columns="index")

    hz = float(sorted_frames["frame_rate"].dropna().iloc[0]) if "frame_rate" in sorted_frames.columns else 25.0
    window_frames = max(round(cfg.sg_window_seconds * hz) | 1, cfg.sg_poly_order + 2)
    if window_frames % 2 == 0:
        window_frames += 1

    method_used = _parse_method(frames, cfg)
    x_sm = sorted_frames["x_smoothed"].to_numpy(dtype=float)
    y_sm = sorted_frames["y_smoothed"].to_numpy(dtype=float)
    raw_x = sorted_frames["x"].to_numpy(dtype=float)
    raw_y = sorted_frames["y"].to_numpy(dtype=float)
    fid_all = sorted_frames["frame_id"].to_numpy()
    n = len(sorted_frames)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    pos_var = np.full(n, np.nan)
    vel_var = np.full(n, np.nan)
    acc_var = np.full(n, np.nan)

    for _key, idx in sorted_frames.groupby(_GROUP_KEYS, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        gvx, gvy, gax, gay, gpv, gvv, gav = _derive_group_dense(
            fid_all[idx_arr],
            x_sm[idx_arr],
            y_sm[idx_arr],
            raw_x[idx_arr],
            raw_y[idx_arr],
            method_used,
            window_frames,
            cfg.sg_poly_order,
            cfg,
            hz,
        )
        vx[idx_arr], vy[idx_arr] = gvx, gvy
        ax[idx_arr], ay[idx_arr] = gax, gay
        pos_var[idx_arr], vel_var[idx_arr], acc_var[idx_arr] = gpv, gvv, gav

    # F1b (ADR-106): kinematic STORAGE is float32 (compute upcasts at kernel boundaries). NaN preserved.
    sorted_frames["vx"] = np.asarray(vx, dtype=np.float32)
    sorted_frames["vy"] = np.asarray(vy, dtype=np.float32)
    sorted_frames["speed"] = np.sqrt(vx * vx + vy * vy).astype(np.float32)
    # accel COMPONENT columns are accel_x/accel_y (NOT ax/ay -- _kernels.py uses ax/ay as triangle
    # anchor-coordinate columns in add_action_context's merge; a frame "ax"/"ay" collides there).
    sorted_frames["accel_x"] = np.asarray(ax, dtype=np.float32)
    sorted_frames["accel_y"] = np.asarray(ay, dtype=np.float32)
    sorted_frames["accel"] = np.sqrt(ax * ax + ay * ay).astype(np.float32)
    sorted_frames["pos_var"] = np.asarray(pos_var, dtype=np.float32)
    sorted_frames["vel_var"] = np.asarray(vel_var, dtype=np.float32)
    sorted_frames["accel_var"] = np.asarray(acc_var, dtype=np.float32)
    # Soft plausibility guard (TF-65 §4.3): NaN implausible velocity/accel + warn, never raise.
    sorted_frames = apply_plausibility_guard(sorted_frames, cfg)
    sorted_frames = sorted_frames.iloc[np.argsort(original_index)].reset_index(drop=True)
    return sorted_frames
