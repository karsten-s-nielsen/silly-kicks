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


def kalman_ca(values: np.ndarray, dt: float, config: PreprocessConfig | None = None) -> KalmanOut:
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

    xs_f = np.zeros((n, 3))
    ps_f = np.zeros((n, 3, 3))
    xp_a = np.zeros((n, 3))
    pp_a = np.zeros((n, 3, 3))
    x = np.array([z[int(np.argmax(valid))], 0.0, 0.0])
    p = np.diag([1.0, 10.0, 10.0])
    for k in range(n):
        xp = f_mat @ x
        pp = f_mat @ p @ f_mat.T + q_mat
        xp_a[k], pp_a[k] = xp, pp
        if valid[k]:
            s = pp[0, 0] + r_scalar
            gain = pp[:, 0] / s
            x = xp + gain * (z[k] - xp[0])
            p = (eye3 - np.outer(gain, h_vec)) @ pp
        else:
            x, p = xp, pp
        xs_f[k], ps_f[k] = x, p

    xs_s = xs_f.copy()
    ps_s = ps_f.copy()
    for k in range(n - 2, -1, -1):
        c_mat = ps_f[k] @ f_mat.T @ np.linalg.inv(pp_a[k + 1])
        xs_s[k] = xs_f[k] + c_mat @ (xs_s[k + 1] - xp_a[k + 1])
        ps_s[k] = ps_f[k] + c_mat @ (ps_s[k + 1] - pp_a[k + 1]) @ c_mat.T

    return KalmanOut(
        pos=xs_s[:, 0],
        vel=xs_s[:, 1],
        acc=xs_s[:, 2],
        pos_var=ps_s[:, 0, 0],
        vel_var=ps_s[:, 1, 1],
        acc_var=ps_s[:, 2, 2],
    )
