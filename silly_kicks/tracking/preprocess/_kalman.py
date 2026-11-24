"""Constant-acceleration Kalman + RTS smoother (TF-65).

State ``[position, velocity, acceleration]`` per coordinate; a forward Kalman pass followed by a
Rauch-Tung-Striebel backward smoother. A missing frame (NaN measurement) skips the update step
(predict only), so the position variance GROWS through an occlusion gap -- honest uncertainty
rather than a fabricated through-gap value. Used both as the ``smoothing_method="kalman"`` point
estimator and as the always-run per-frame uncertainty source (decoupled from the point smoother).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._config_dataclass import PreprocessConfig


@dataclass(frozen=True)
class KalmanOut:
    """Smoothed 1-D kinematics + the RTS state-variance diagonal (confidence)."""

    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    pos_var: np.ndarray
    vel_var: np.ndarray
    acc_var: np.ndarray


@dataclass(frozen=True)
class _KalmanCov:
    """The data-independent half of :func:`kalman_ca`: every covariance and gain of the forward pass and the RTS
    smoother. It depends only on the detection mask, ``dt`` and the two Kalman parameters, never on the values."""

    f_mat: np.ndarray  # (3, 3) the constant-acceleration transition
    gain: np.ndarray  # (n, 3) the Kalman gain on a detected frame (unused on a missed one)
    rts: np.ndarray  # (n - 1, 3, 3) the RTS gain c[k] = P_f[k] F^T inv(P_p[k+1])
    ps_s: np.ndarray  # (n, 3, 3) the smoothed state covariances


def _kalman_covariance(valid: np.ndarray, dt: float, cfg: PreprocessConfig) -> _KalmanCov:
    """The covariance recursion of :func:`kalman_ca` for one detection mask (spec 2026-11-08 tracking-batch W7.1).

    The same operations in the same order as the historical interleaved loop, so a filter that reuses the result
    is bit-identical to one that recomputes it.
    """
    n = len(valid)
    f_mat = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    h_vec = np.array([1.0, 0.0, 0.0])
    sj = cfg.kalman_jerk_std
    q_mat = (
        sj
        * sj
        * np.array(
            [
                [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
                [dt**3 / 6.0, dt**2 / 2.0, dt],
            ]
        )
    )
    r_scalar = cfg.kalman_meas_noise_m**2
    eye3 = np.eye(3)

    ps_f = np.zeros((n, 3, 3))
    pp_a = np.zeros((n, 3, 3))
    gains = np.zeros((n, 3))
    p = np.diag([1.0, 10.0, 10.0])
    for k in range(n):
        pp = f_mat @ p @ f_mat.T + q_mat
        pp_a[k] = pp
        if valid[k]:
            s = pp[0, 0] + r_scalar
            gain = pp[:, 0] / s
            gains[k] = gain
            p = (eye3 - np.outer(gain, h_vec)) @ pp
        else:
            p = pp
        ps_f[k] = p

    ps_s = ps_f.copy()
    rts = np.zeros((max(n - 1, 0), 3, 3))
    for k in range(n - 2, -1, -1):
        c_mat = ps_f[k] @ f_mat.T @ np.linalg.inv(pp_a[k + 1])
        rts[k] = c_mat
        ps_s[k] = ps_f[k] + c_mat @ (ps_s[k + 1] - pp_a[k + 1]) @ c_mat.T
    return _KalmanCov(f_mat=f_mat, gain=gains, rts=rts, ps_s=ps_s)


def kalman_ca(
    values: np.ndarray, dt: float, config: PreprocessConfig | None = None, *, cov_cache: dict | None = None
) -> KalmanOut:
    """Constant-acceleration Kalman + RTS smoother over a 1-D measurement series.

    ``values`` is one coordinate on a (dense) frame grid; NaN entries are treated as non-detections
    (predict-only). Returns position / velocity / acceleration and the position-variance trace.
    """
    cfg = config or PreprocessConfig.default()
    z = np.asarray(values, dtype=np.float64)
    n = len(z)
    nan = np.full(n, np.nan)
    valid = ~np.isnan(z)
    if int(valid.sum()) < 3:
        return KalmanOut(nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy())

    # W7.1: the covariance recursion depends only on (dt, the Kalman parameters, the mask); a caller that filters
    # many series passes one ``cov_cache`` per call, so series that share a mask share one recursion.
    if cov_cache is None:
        cov = _kalman_covariance(valid, dt, cfg)
    else:
        key = (dt, cfg.kalman_jerk_std, cfg.kalman_meas_noise_m, valid.tobytes())
        cov = cov_cache.get(key)
        if cov is None:
            cov = _kalman_covariance(valid, dt, cfg)
            cov_cache[key] = cov
    f_mat = cov.f_mat

    xs_f = np.zeros((n, 3))
    xp_a = np.zeros((n, 3))
    x = np.array([z[int(np.argmax(valid))], 0.0, 0.0])
    for k in range(n):
        xp = f_mat @ x
        xp_a[k] = xp
        x = xp + cov.gain[k] * (z[k] - xp[0]) if valid[k] else xp
        xs_f[k] = x

    xs_s = xs_f.copy()
    for k in range(n - 2, -1, -1):
        xs_s[k] = xs_f[k] + cov.rts[k] @ (xs_s[k + 1] - xp_a[k + 1])
    ps_s = cov.ps_s

    return KalmanOut(
        pos=xs_s[:, 0],
        vel=xs_s[:, 1],
        acc=xs_s[:, 2],
        pos_var=ps_s[:, 0, 0].copy(),  # copies: the covariance may be shared through cov_cache
        vel_var=ps_s[:, 1, 1].copy(),
        acc_var=ps_s[:, 2, 2].copy(),
    )
