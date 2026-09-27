"""Fused numba kernel for native DAS (ADR-107; the numba adapter to the ``_das_engine`` numpy port).

One streaming pass per (frame, angle): time-to-arrive -> interception rate -> possibility density ->
normalise -> team/player surface integral, with O(players x radial) working memory per frame. Matches
the numpy engine's formulas and player order, so it agrees with accessible-space 2.0.15 within
rtol=atol=1e-10 (bit-identity is NOT claimed -- libm exp vs numpy exp, sequential vs pairwise sums;
the ADR-076 KDE cpu-numba precedent). Serial and ``prange`` variants are byte-identical to each other
(frames are independent). Explicit float64 signatures reject a float32 array (the ADR-106 rule made
structural for DAS). Imported lazily so a bare ``import silly_kicks.tracking._das`` stays numba-free.
"""

from __future__ import annotations

import os

import numpy as np

from silly_kicks.tracking._das_engine import DasResult
from silly_kicks.tracking._das_pack import PackedFrames, Reason
from silly_kicks.tracking._das_params import PassSimParams, simulation_grids

try:
    import numba
    from numba import njit, prange

    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on the numpy-only legs
    AVAILABLE = False

_NUMBA_CACHE = os.environ.get("SILLY_KICKS_NUMBA_CACHE", "0") == "1" or bool(os.environ.get("NUMBA_CACHE_DIR"))

_SEMI = 7.32 / 2 + 0.06
_X_GOAL = 52.5
_D_INT = -0.52156283
_D_DIST = -0.14447723
_D_ANG = 0.40579492


if AVAILABLE:

    @njit(cache=_NUMBA_CACHE, inline="always")
    def _approx_sigmoid(x):
        return 0.5 * (x / (1.0 + abs(x)) + 1.0)

    @njit(cache=_NUMBA_CACHE)
    def _one_frame(
        lo,
        hi,
        px,
        py,
        pvx,
        pvy,
        att,
        passer,
        ball_x,
        ball_y,
        direction,
        cos_phi,
        sin_phi,
        v0,
        d,
        t_ball,
        dt0,
        rate_divisor,
        dr,
        d_area,
        b0,
        b1,
        player_velocity,
        inertial,
        tol,
        use_max,
        v_max,
        a_max,
        factor2_unused,
        normalize,
        respect_offside,
        exclude_passer,
        danger_weight,
        out_team_as,
        out_team_das,
        out_player_as,
        out_player_das,
        fi,
    ):
        m = hi - lo  # players in this frame
        nphi = cos_phi.shape[0]
        nv = v0.shape[0]
        nt = d.shape[0]
        inv_dw = 1.0 / danger_weight

        # Offside: mark offside attackers (their TTA becomes inf).
        off = np.zeros(m, dtype=np.bool_)
        if respect_offside and m >= 2:
            # second-largest norm_x among finite defenders
            max1 = -np.inf
            max2 = -np.inf
            for p in range(m):
                nx = px[lo + p] * direction
                is_def = (not att[lo + p]) and np.isfinite(nx)
                if is_def:
                    if nx > max1:
                        max2 = max1
                        max1 = nx
                    elif nx > max2:
                        max2 = nx
            ball_norm_x = ball_x * direction
            line = max2 if max2 > ball_norm_x else ball_norm_x
            for p in range(m):
                nx = px[lo + p] * direction
                if att[lo + p] and np.isfinite(nx) and nx > line and nx > 0.0 and (not passer[lo + p]):
                    off[p] = True

        player_poss = np.zeros((m, nt))
        ar = np.empty((m, nt))
        sum_att = np.empty(nt)
        sum_def = np.empty(nt)
        cum_att = np.empty(nt)
        cum_def = np.empty(nt)
        xt = np.empty(nt)
        yt = np.empty(nt)

        team_as = 0.0
        team_das = 0.0
        pas = np.zeros(m)
        pdas = np.zeros(m)

        for j in range(nphi):
            c = cos_phi[j]
            s = sin_phi[j]
            for t in range(nt):
                xt[t] = ball_x + c * d[t]
                yt[t] = ball_y + s * d[t]

            # TTA[p,t] for this angle.
            tta = np.empty((m, nt))
            for p in range(m):
                x = px[lo + p]
                y = py[lo + p]
                vx = pvx[lo + p]
                vy = pvy[lo + p]
                is_passer = passer[lo + p]
                if off[p]:
                    for t in range(nt):
                        tta[p, t] = np.inf
                    continue
                if use_max:
                    v_mag = np.sqrt(vx * vx + vy * vy)
                    remaining = v_mag + a_max * inertial
                    if remaining > v_max:
                        remaining = v_max
                else:
                    remaining = player_velocity
                x_mid = x + vx * inertial
                y_mid = y + vy * inertial
                for t in range(nt):
                    d0 = np.sqrt((x - xt[t]) ** 2 + (y - yt[t]) ** 2)
                    if d0 < tol:
                        val = np.hypot(xt[t] - x, yt[t] - y) / player_velocity
                    else:
                        val = np.sqrt((x_mid - xt[t]) ** 2 + (y_mid - yt[t]) ** 2) / remaining + inertial
                    if exclude_passer and is_passer:
                        val = np.inf
                    if val != val:  # NaN -> inf
                        val = np.inf
                    tta[p, t] = val

            # v0 loop: running max of the possibility density over ball speeds.
            for k in range(nv):
                for t in range(nt):
                    sa = 0.0
                    sd = 0.0
                    tb = t_ball[k, t]
                    for p in range(m):
                        v = b0 + b1 * (tta[p, t] - tb)
                        sig = _approx_sigmoid(v)
                        if sig != sig:  # NaN (overflow) -> 0
                            sig = 0.0
                        a = sig / rate_divisor[k]
                        ar[p, t] = a
                        if att[lo + p]:
                            sa += a
                        else:
                            sd += a
                    sum_att[t] = sa
                    sum_def[t] = sd
                # cumulative trapezoid along t
                cum_att[0] = 0.0
                cum_def[0] = 0.0
                ia = 0.0
                idf = 0.0
                for t in range(1, nt):
                    dtt = t_ball[k, t] - t_ball[k, t - 1]
                    ia += dtt * (sum_att[t] + sum_att[t - 1]) / 2.0
                    idf += dtt * (sum_def[t] + sum_def[t - 1]) / 2.0
                    cum_att[t] = np.exp(-ia)
                    cum_def[t] = np.exp(-idf)
                cum_att[0] = np.exp(0.0)
                cum_def[0] = np.exp(0.0)
                scale = dt0[k] / (d[1] - d[0])  # dt0[k]/radial_gridsize (d is equally spaced)
                for p in range(m):
                    is_att = att[lo + p]
                    for t in range(nt):
                        opp = cum_def[t] if is_att else cum_att[t]
                        val = opp * ar[p, t] * scale
                        if k == 0:
                            player_poss[p, t] = val
                        elif val > player_poss[p, t]:
                            player_poss[p, t] = val

            # normalise (per frame, per angle): divide by max over (p, t) of player_poss * dx
            if normalize:
                dx = d[1] - d[0]
                num_max = 0.0
                first = True
                for p in range(m):
                    for t in range(nt):
                        v = player_poss[p, t] * dx
                        if first or v > num_max:
                            num_max = v
                            first = False
                for p in range(m):
                    for t in range(nt):
                        player_poss[p, t] = player_poss[p, t] / num_max

            # danger + on-pitch weight for this angle, then integrate.
            for t in range(nt):
                w = dr[t] * d_area[j, t]
                on = (xt[t] >= -_X_GOAL) and (xt[t] <= _X_GOAL) and (yt[t] >= -34.0) and (yt[t] <= 34.0)
                if not on:
                    w = 0.0
                # danger
                x_n = xt[t] * direction
                y_n = yt[t] * direction
                yg = y_n
                if yg > _SEMI:
                    yg = _SEMI
                elif yg < -_SEMI:
                    yg = -_SEMI
                dist = np.sqrt((x_n - _X_GOAL) ** 2 + (y_n - yg) ** 2)
                u0 = _X_GOAL - x_n
                u1 = _SEMI - y_n
                v1 = -_SEMI - y_n
                nu = np.sqrt(u0 * u0 + u1 * u1)
                nvv = np.sqrt(u0 * u0 + v1 * v1)
                div = nu * nvv
                if div == 0.0:
                    div = np.inf
                dot = u0 * u0 + u1 * v1
                ang = abs(np.arccos(dot / div))
                logit = _D_INT + _D_DIST * dist + _D_ANG * ang
                danger = 1.0 / (1.0 + np.exp(-logit))
                dfac = danger**inv_dw

                # team density = max over attacking players (implicit 0), nan-skipping
                team_d = 0.0
                for p in range(m):
                    if att[lo + p]:
                        pv = player_poss[p, t]
                        if pv == pv and pv > team_d:  # skip NaN
                            team_d = pv
                team_as += team_d * w
                team_das += dfac * team_d * w
                for p in range(m):
                    pv = player_poss[p, t]
                    pas[p] += pv * w
                    pdas[p] += dfac * pv * w

        out_team_as[fi] = team_as
        out_team_das[fi] = team_das
        for p in range(m):
            out_player_as[lo + p] = pas[p]
            out_player_das[lo + p] = pdas[p]

    def _make_kernel(parallel):
        @njit(cache=_NUMBA_CACHE, parallel=parallel)
        def _kernel(
            ok,
            offsets,
            px,
            py,
            pvx,
            pvy,
            att,
            passer,
            ball_x,
            ball_y,
            direction,
            cos_phi,
            sin_phi,
            v0,
            d,
            t_ball,
            dt0,
            rate_divisor,
            dr,
            d_area,
            b0,
            b1,
            player_velocity,
            inertial,
            tol,
            use_max,
            v_max,
            a_max,
            factor2,
            normalize,
            respect_offside,
            exclude_passer,
            danger_weight,
            out_team_as,
            out_team_das,
            out_player_as,
            out_player_das,
        ):
            n = ok.shape[0]
            loop = prange(n) if parallel else range(n)
            for i in loop:
                fi = ok[i]
                _one_frame(
                    offsets[fi],
                    offsets[fi + 1],
                    px,
                    py,
                    pvx,
                    pvy,
                    att,
                    passer,
                    ball_x[fi],
                    ball_y[fi],
                    direction[fi],
                    cos_phi,
                    sin_phi,
                    v0,
                    d,
                    t_ball,
                    dt0,
                    rate_divisor,
                    dr,
                    d_area,
                    b0,
                    b1,
                    player_velocity,
                    inertial,
                    tol,
                    use_max,
                    v_max,
                    a_max,
                    factor2,
                    normalize,
                    respect_offside,
                    exclude_passer,
                    danger_weight,
                    out_team_as,
                    out_team_das,
                    out_player_as,
                    out_player_das,
                    fi,
                )

        return _kernel

    _KERNEL_SERIAL = _make_kernel(False)
    _KERNEL_PARALLEL = _make_kernel(True)


def _require_float64(*arrays):
    for a in arrays:
        if a.dtype != np.float64:
            raise TypeError(f"DAS numba kernel requires float64 arrays, got {a.dtype}")


def das_frames_serial(*args):
    """Direct serial-kernel entry (tests). Rejects non-float64 kinematic arrays (px/py/pvx/pvy)."""
    _require_float64(args[2], args[3], args[4], args[5])
    return _KERNEL_SERIAL(*args)


def compute_das_numba(packed: PackedFrames, params: PassSimParams, *, chunk_size=None, n_threads=None) -> DasResult:
    """Compute team + per-player AS/DAS via the numba kernel (serial default; prange when n_threads>1)."""
    if not AVAILABLE:  # pragma: no cover
        raise RuntimeError("numba is not available")
    grids = simulation_grids(params)
    n = packed.n_frames
    n_rows = len(packed.px)
    team_as = np.full(n, np.nan)
    team_das = np.full(n, np.nan)
    player_as = np.full(n_rows, np.nan)
    player_das = np.full(n_rows, np.nan)
    ok = np.flatnonzero(packed.reason == Reason.OK).astype(np.int64)
    if len(ok) == 0:
        return DasResult(team_as, team_das, player_as, player_das, packed.reason.copy())

    px = np.ascontiguousarray(packed.px, dtype=np.float64)
    py = np.ascontiguousarray(packed.py, dtype=np.float64)
    pvx = np.ascontiguousarray(packed.pvx, dtype=np.float64)
    pvy = np.ascontiguousarray(packed.pvy, dtype=np.float64)
    _require_float64(px, py, pvx, pvy)
    att = np.ascontiguousarray(packed.p_attacking, dtype=np.bool_)
    passer = np.ascontiguousarray(packed.p_is_passer, dtype=np.bool_)
    ball_x = np.ascontiguousarray(packed.ball_xy[:, 0], dtype=np.float64)
    ball_y = np.ascontiguousarray(packed.ball_xy[:, 1], dtype=np.float64)
    direction = np.ascontiguousarray(packed.direction, dtype=np.float64)
    offsets = np.ascontiguousarray(packed.offsets, dtype=np.int64)

    kernel = _KERNEL_SERIAL
    prev_threads = None
    if n_threads is not None and n_threads > 1:
        kernel = _KERNEL_PARALLEL
        prev_threads = numba.get_num_threads()
        numba.set_num_threads(int(n_threads))
    try:
        kernel(
            ok,
            offsets,
            px,
            py,
            pvx,
            pvy,
            att,
            passer,
            ball_x,
            ball_y,
            direction,
            grids.cos_phi,
            grids.sin_phi,
            grids.v0,
            grids.d,
            grids.t_ball,
            grids.dt0,
            grids.rate_divisor,
            grids.dr,
            grids.d_area,
            params.b0,
            params.b1,
            params.player_velocity,
            params.inertial_seconds,
            params.tol_distance,
            params.use_max,
            params.v_max,
            params.a_max,
            params.factor2,
            params.normalize,
            params.respect_offside,
            params.exclude_passer,
            params.danger_weight,
            team_as,
            team_das,
            player_as,
            player_das,
        )
    finally:
        if prev_threads is not None:
            numba.set_num_threads(prev_threads)

    return DasResult(team_as, team_das, player_as, player_das, packed.reason.copy())
