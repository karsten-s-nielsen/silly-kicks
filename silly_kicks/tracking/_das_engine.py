"""Native DAS engine: reference-order numpy core + engine dispatch (ADR-107).

Reproduces ``accessible-space`` 2.0.15's ``simulate_passes`` possibility path, danger weighting and
surface integration EXACTLY (Δ = 0 vs the golden oracle in ``reference`` quadrature; plan Appendix A),
then applies the periodic quadrature by default (ADR-108). Frames whose pack ``Reason`` is not ``OK``
are never simulated -- they get NaN. Memory is bounded by (a) processing frames in chunks and (b)
looping over ball speeds inside a chunk, so the 5-D ``F x P x V0 x PHI x T`` array the library
materialises is never held (the per-speed ``max`` / ``nansum`` reductions are accumulated, which is
bit-identical to reducing the full array).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
import scipy.integrate

from silly_kicks.id_compat import canonical_id, canonical_id_series
from silly_kicks.tracking._das_pack import PackedFrames, Reason
from silly_kicks.tracking._das_params import XC_PARAMS, PassSimParams, SimGrids, simulation_grids

_X_OFFSET = 52.5
_Y_OFFSET = 34.0

_SEMI_GOAL_WIDTH = 7.32 / 2 + 0.06  # 3.72
_X_GOAL = 52.5
_DANGER_INTERCEPT = -0.52156283
_DANGER_C_DIST = -0.14447723
_DANGER_C_ANGLE = 0.40579492


@dataclass(frozen=True)
class DasResult:
    """Engine output: team-level and per-player AS/DAS, aligned with the packed frames/rows."""

    team_as: np.ndarray  # float64 (n_frames,) -- in-possession team's accessible space
    team_das: np.ndarray  # float64 (n_frames,) -- dangerous accessible space
    player_as: np.ndarray  # float64 (n_player_rows,) aligned with PackedFrames player rows
    player_das: np.ndarray  # float64 (n_player_rows,)
    reason: np.ndarray  # uint8 (n_frames,) copied from PackedFrames


def _numba_available() -> bool:
    if os.environ.get("SILLY_KICKS_DAS_FORCE_NUMPY", "") == "1":
        return False
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def _approx_sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x / (1 + np.abs(x)) + 1)


def _tta(px, py, pvx, pvy, xt, yt, params: PassSimParams) -> np.ndarray:
    """Approximate two-point time-to-arrive (motion_models.approx_two_point_time_to_arrive)."""
    inert = params.inertial_seconds
    x_mid = px + pvx * inert
    y_mid = py + pvy * inert
    if not params.use_max:
        remaining = params.player_velocity
    else:
        v_inert_mag = np.sqrt(pvx**2 + pvy**2)
        remaining = np.minimum(v_inert_mag + params.a_max * inert, params.v_max)
    t2 = np.sqrt((x_mid - xt) ** 2 + (y_mid - yt) ** 2) / remaining
    t_total = t2 + inert
    tol_mask = np.sqrt((px - xt) ** 2 + (py - yt) ** 2) < params.tol_distance
    tol_t = np.hypot(xt - px, yt - py) / params.player_velocity
    return np.where(tol_mask, tol_t, t_total)


def _danger(x_grid: np.ndarray, y_grid: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """xG-style danger surface on direction-normalised coordinates (F,PHI,T)."""
    x_n = x_grid * direction[:, np.newaxis, np.newaxis]
    y_n = y_grid * direction[:, np.newaxis, np.newaxis]
    y_goal = np.clip(y_n, -_SEMI_GOAL_WIDTH, _SEMI_GOAL_WIDTH)
    dist = np.sqrt((x_n - _X_GOAL) ** 2 + (y_n - y_goal) ** 2)
    u0 = _X_GOAL - x_n
    u1 = _SEMI_GOAL_WIDTH - y_n
    v1 = -_SEMI_GOAL_WIDTH - y_n
    norm_u = np.sqrt(u0**2 + u1**2)
    norm_v = np.sqrt(u0**2 + v1**2)
    div = norm_u * norm_v
    div = np.where(div == 0, np.inf, div)
    dot = u0 * u0 + u1 * v1
    angle = np.abs(np.arccos(dot / div))
    logit = _DANGER_INTERCEPT + _DANGER_C_DIST * dist + _DANGER_C_ANGLE * angle
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-logit))


def _simulate_chunk(packed: PackedFrames, params: PassSimParams, grids: SimGrids, f_idx: np.ndarray):
    """Simulate the OK frames listed in ``f_idx``. Returns (attack_poss_density, player_poss_density,
    x_grid, y_grid, p_max, per-frame present-player counts)."""
    fc = len(f_idx)
    counts = (packed.offsets[f_idx + 1] - packed.offsets[f_idx]).astype(np.int64)
    p_max = int(counts.max()) if fc else 0

    # Rectangular per-chunk arrays; padded slots are NaN position / non-attacking (contribute 0).
    px = np.full((fc, p_max), np.nan)
    py = np.full((fc, p_max), np.nan)
    pvx = np.full((fc, p_max), np.nan)
    pvy = np.full((fc, p_max), np.nan)
    att = np.zeros((fc, p_max), dtype=bool)
    passer = np.zeros((fc, p_max), dtype=bool)
    for i, f in enumerate(f_idx):
        lo, hi = packed.offsets[f], packed.offsets[f + 1]
        m = hi - lo
        px[i, :m] = packed.px[lo:hi]
        py[i, :m] = packed.py[lo:hi]
        pvx[i, :m] = packed.pvx[lo:hi]
        pvy[i, :m] = packed.pvy[lo:hi]
        att[i, :m] = packed.p_attacking[lo:hi]
        passer[i, :m] = packed.p_is_passer[lo:hi]

    ball = packed.ball_xy[f_idx]  # (fc, 2)
    direction = packed.direction[f_idx]  # (fc,)

    # Offside: treat offside attackers like air (core.simulate_passes pre-processing).
    if params.respect_offside:
        _apply_offside(px, py, pvx, pvy, att, passer, ball, direction)

    d = grids.d
    t_ball = grids.t_ball  # (V, T)
    dt0 = grids.dt0  # (V,)
    rate_divisor = grids.rate_divisor  # (V,)
    dx = params.radial_gridsize

    cos_phi = grids.cos_phi  # (PHI,)
    sin_phi = grids.sin_phi
    x_grid = ball[:, 0][:, None, None] + cos_phi[None, :, None] * d[None, None, :]  # (fc,PHI,T)
    y_grid = ball[:, 1][:, None, None] + sin_phi[None, :, None] * d[None, None, :]

    # TTA per (frame, player, PHI, T).
    tta = _tta(
        px[:, :, None, None],
        py[:, :, None, None],
        pvx[:, :, None, None],
        pvy[:, :, None, None],
        x_grid[:, None, :, :],
        y_grid[:, None, :, :],
        params,
    )
    if params.exclude_passer:
        tta = np.where(passer[:, :, None, None], np.inf, tta)
    tta = np.nan_to_num(tta, nan=np.inf)  # (fc,P,PHI,T)

    # Loop over ball speeds to bound memory; accumulate the v0-max of the poss density.
    player_poss_density = None
    for k in range(len(grids.v0)):
        tmp = tta - t_ball[k][np.newaxis, np.newaxis, np.newaxis, :]  # (fc,P,PHI,T)
        with np.errstate(over="ignore"):
            tmp = params.b0 + params.b1 * tmp
        with np.errstate(invalid="ignore"):
            tmp = _approx_sigmoid(tmp)
        tmp = np.nan_to_num(tmp, nan=0.0)
        ar = tmp / rate_divisor[k]  # (fc,P,PHI,T)

        sum_ar_att = np.nansum(np.where(att[:, :, None, None], ar, 0), axis=1)  # (fc,PHI,T)
        sum_ar_def = np.nansum(np.where(~att[:, :, None, None], ar, 0), axis=1)
        int_att = scipy.integrate.cumulative_trapezoid(
            y=sum_ar_att, x=t_ball[k][np.newaxis, np.newaxis, :], initial=0, axis=-1
        )
        int_def = scipy.integrate.cumulative_trapezoid(
            y=sum_ar_def, x=t_ball[k][np.newaxis, np.newaxis, :], initial=0, axis=-1
        )
        cum_p0_att = np.exp(-int_att)  # (fc,PHI,T)
        cum_p0_def = np.exp(-int_def)
        cum_p0_opp = np.where(att[:, :, None, None], cum_p0_def[:, None, :, :], cum_p0_att[:, None, :, :])
        dpr_poss_dt = cum_p0_opp * ar
        dpr_poss_dx = dpr_poss_dt * dt0[k] / dx  # (fc,P,PHI,T)
        player_poss_density = (
            dpr_poss_dx if player_poss_density is None else np.maximum(player_poss_density, dpr_poss_dx)
        )

    # v0 grid is non-empty (n_v0 >= 1, validated in PassSimParams), so the loop ran and this is set.
    player_poss_density = cast(np.ndarray, player_poss_density)
    if params.normalize:
        num_max = np.max(player_poss_density * dx, axis=(1, 3))  # (fc,PHI)
        with np.errstate(invalid="ignore"):
            player_poss_density = player_poss_density / num_max[:, None, :, None]

    attack_poss_density = np.nanmax(np.where(att[:, :, None, None], player_poss_density, 0), axis=1)  # (fc,PHI,T)
    return attack_poss_density, player_poss_density, x_grid, y_grid, counts


def _apply_offside(px, py, pvx, pvy, att, passer, ball, direction) -> None:
    """Set offside attackers' kinematics to NaN, per core.simulate_passes (respect_offside)."""
    norm_x = px * direction[:, None]  # (fc,P)
    ball_norm_x = ball[:, 0] * direction  # (fc,)
    is_attacking_team = att | ~np.isfinite(norm_x)
    masked = np.ma.array(norm_x, mask=is_attacking_team)
    sorted_def = np.ma.sort(masked, axis=1, endwith=False)
    p = norm_x.shape[1]
    if p < 2:
        return  # fewer than two defenders detectable -> no offside (D-OFF)
    second_last = sorted_def[:, -2]
    second_last = np.ma.filled(second_last, np.nan)
    line = np.maximum(second_last, ball_norm_x)
    offside = is_attacking_team & (norm_x > line[:, None]) & (norm_x > 0) & (~passer)
    offside = offside & np.isfinite(norm_x)
    for arr in (px, py, pvx, pvy):
        arr[offside] = np.nan


def _integrate(
    density: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, grids: SimGrids, *, per_player: bool
) -> np.ndarray:
    """Clip off-pitch points to 0 then integrate polar area (core.integrate_surfaces)."""
    on = (x_grid >= -_X_GOAL) & (x_grid <= _X_GOAL) & (y_grid >= -34.0) & (y_grid <= 34.0)  # (fc,PHI,T)
    dr = grids.dr
    d_area = grids.d_area  # (PHI,T)
    if per_player:
        clipped = np.where(on[:, None, :, :], density, 0)
        return np.sum(clipped * dr[None, None, None, :] * d_area[None, None, :, :], axis=(2, 3))  # (fc,P)
    clipped = np.where(on, density, 0)
    return np.sum(clipped * dr[None, None, :] * d_area[None, :, :], axis=(1, 2))  # (fc,)


def _numpy_integrand(packed: PackedFrames, params: PassSimParams) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic (Task 6): per-(frame,PHI,T) clipped team integrands g_as, g_das BEFORE area weights.

    ``AS = sum(g_as * dr * d_area)``, ``DAS = sum(g_das * dr * d_area)`` over (PHI,T). Only OK frames.
    """
    grids = simulation_grids(params)
    ok = np.flatnonzero(packed.reason == Reason.OK)
    n, phi, t = packed.n_frames, params.n_angles, len(grids.d)
    g_as = np.zeros((n, phi, t))
    g_das = np.zeros((n, phi, t))
    if len(ok):
        apd, _ppd, x_grid, y_grid, _c = _simulate_chunk(packed, params, grids, ok)
        danger = _danger(x_grid, y_grid, packed.direction[ok])
        on = (x_grid >= -_X_GOAL) & (x_grid <= _X_GOAL) & (y_grid >= -34.0) & (y_grid <= 34.0)
        g_as[ok] = np.where(on, apd, 0)
        g_das[ok] = np.where(on, danger ** (1.0 / params.danger_weight) * apd, 0)
    return g_as, g_das


def compute_das(
    packed: PackedFrames,
    params: PassSimParams,
    *,
    chunk_size: int | None = None,
    n_threads: int | None = None,
    engine: Literal["auto", "numpy", "numba"] = "auto",
) -> DasResult:
    """Compute team + per-player AS/DAS for the packed frames. Non-OK frames get NaN.

    ``engine="auto"`` uses numba when importable (and ``SILLY_KICKS_DAS_FORCE_NUMPY`` unset), else
    numpy. The numba path lands in Task 5; until then ``auto`` falls back to numpy.
    """
    if engine == "numba" or (engine == "auto" and _numba_available()):
        from silly_kicks.tracking import _das_numba

        if _das_numba.AVAILABLE:
            return _das_numba.compute_das_numba(packed, params, chunk_size=chunk_size, n_threads=n_threads)
    return _compute_das_numpy(packed, params, chunk_size=chunk_size)


def compute_das_paired(
    actual: PackedFrames,
    counterfactual: PackedFrames,
    moved: np.ndarray,
    params: PassSimParams,
    *,
    chunk_size: int | None = None,
    n_threads: int | None = None,
    engine: Literal["auto", "numpy", "numba"] = "auto",
) -> tuple[DasResult, DasResult]:
    """Score both legs of a counterfactual pair (spec 6.6, SC-1).

    Each leg is computed through the standard engine, so each result is bit-identical to an independent
    :func:`compute_das` call by construction (offside is a per-leg function of all defenders, so the
    result is offside-correct). ``moved`` is the derived, validated moved-row mask from
    :func:`~silly_kicks.tracking._das_pack.pack_paired`; it identifies the rows that differ between the
    legs (the seam a future shared-interception kernel would exploit) and is accepted here so the paired
    contract is a single call site.
    """
    a = compute_das(actual, params, chunk_size=chunk_size, n_threads=n_threads, engine=engine)
    c = compute_das(counterfactual, params, chunk_size=chunk_size, n_threads=n_threads, engine=engine)
    return a, c


def _compute_das_numpy(packed: PackedFrames, params: PassSimParams, *, chunk_size: int | None) -> DasResult:
    grids = simulation_grids(params)
    n = packed.n_frames
    n_rows = len(packed.px)
    team_as = np.full(n, np.nan)
    team_das = np.full(n, np.nan)
    player_as = np.full(n_rows, np.nan)
    player_das = np.full(n_rows, np.nan)

    ok = np.flatnonzero(packed.reason == Reason.OK)
    if len(ok) == 0:
        return DasResult(team_as, team_das, player_as, player_das, packed.reason.copy())

    cs = chunk_size if (chunk_size is not None and chunk_size > 0) else 32
    for start in range(0, len(ok), cs):
        f_idx = ok[start : start + cs]
        apd, ppd, x_grid, y_grid, _counts = _simulate_chunk(packed, params, grids, f_idx)
        danger = _danger(x_grid, y_grid, packed.direction[f_idx])
        dang_apd = danger ** (1.0 / params.danger_weight) * apd
        dang_ppd = danger[:, None, :, :] ** (1.0 / params.danger_weight) * ppd

        team_as[f_idx] = _integrate(apd, x_grid, y_grid, grids, per_player=False)
        team_das[f_idx] = _integrate(dang_apd, x_grid, y_grid, grids, per_player=False)
        pl_as = _integrate(ppd, x_grid, y_grid, grids, per_player=True)  # (fc,P)
        pl_das = _integrate(dang_ppd, x_grid, y_grid, grids, per_player=True)
        for i, f in enumerate(f_idx):
            lo, hi = packed.offsets[f], packed.offsets[f + 1]
            m = hi - lo
            player_as[lo:hi] = pl_as[i, :m]
            player_das[lo:hi] = pl_das[i, :m]

    return DasResult(team_as, team_das, player_as, player_das, packed.reason.copy())


# --------------------------------------------------------------------------------------------------
# xC (expected pass completion) -- native reimplementation of get_expected_pass_completion (D1).
# --------------------------------------------------------------------------------------------------

_XC_FRAME_KEYS = ["game_id", "period_id", "frame_id"]


def compute_xc(
    passes: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: PassSimParams = XC_PARAMS,
    chunk_size: int | None = None,
    n_threads: int | None = None,
    engine: str = "numpy",
) -> np.ndarray:
    """Expected pass completion per pass (accessible-space get_expected_pass_completion; xC profile).

    One simulated frame per pass (the pass's tracking frame), ball at the event start, a single pass
    angle, no offside, no danger; xC is the possibility CDF (cummax) at the furthest on-pitch radial
    point, clipped to [0,1]. A pass whose frame or event team is absent from tracking gets NaN (the
    D-XC-FRAME / D-XC-TEAM divergences); a passer absent from its frame is simply not excluded.
    """
    import pandas as pd

    grids = simulation_grids(params)
    n = len(passes)
    xc = np.full(n, np.nan)
    if n == 0 or len(frames) == 0:
        return xc

    fkey = pd.MultiIndex.from_frame(frames[_XC_FRAME_KEYS].apply(lambda c: c.map(canonical_id)))
    frames = frames.reset_index(drop=True)
    is_ball_all = frames["is_ball"].astype(bool).to_numpy()
    fteam = canonical_id_series(frames["team_id"]).to_numpy()
    fpid = canonical_id_series(frames["player_id"]).to_numpy()

    # group frame row positions by canonical (game, period, frame)
    groups: dict = {}
    for pos, k in enumerate(
        zip(fkey.get_level_values(0), fkey.get_level_values(1), fkey.get_level_values(2), strict=True)
    ):
        groups.setdefault(k, []).append(pos)

    px_all = np.asarray(frames["x"], dtype=np.float64) - _X_OFFSET
    py_all = np.asarray(frames["y"], dtype=np.float64) - _Y_OFFSET
    pvx_all = np.asarray(frames["vx"], dtype=np.float64)
    pvy_all = np.asarray(frames["vy"], dtype=np.float64)

    for pi in range(n):
        row = passes.iloc[pi]
        key = (
            (canonical_id(row["game_id"]), canonical_id(row["period_id"]), canonical_id(row["frame_id"]))
            if "game_id" in passes.columns
            else (canonical_id(1), canonical_id(row.get("period_id", 1)), canonical_id(row["frame_id"]))
        )
        pos = groups.get(key)
        if pos is None:
            continue  # D-XC-FRAME: NaN
        pos = np.array(pos, dtype=np.int64)
        ev_team = canonical_id(row["team_id"])
        player_pos = pos[~is_ball_all[pos]]
        teams_here = set(fteam[player_pos])
        if ev_team not in teams_here:
            continue  # D-XC-TEAM: NaN
        # order players canonically
        order = np.argsort(fpid[player_pos].astype(str), kind="stable")
        player_pos = player_pos[order]
        px = px_all[player_pos]
        py = py_all[player_pos]
        pvx = pvx_all[player_pos]
        pvy = pvy_all[player_pos]
        att = np.array([fteam[p] == ev_team for p in player_pos], dtype=bool)
        ev_player = canonical_id(row["player_id"])
        passer = np.array([fpid[p] == ev_player for p in player_pos], dtype=bool)

        start_x = float(row["start_x"]) - _X_OFFSET
        start_y = float(row["start_y"]) - _Y_OFFSET
        end_x = float(row["end_x"]) - _X_OFFSET
        end_y = float(row["end_y"]) - _Y_OFFSET
        ball_x, ball_y = start_x, start_y
        phi = np.arctan2(end_y - start_y, end_x - start_x)
        xc[pi] = _xc_one_pass(px, py, pvx, pvy, att, passer, ball_x, ball_y, phi, params, grids)
    return xc


def _xc_one_pass(px, py, pvx, pvy, att, passer, ball_x, ball_y, phi, params, grids) -> float:
    d = grids.d
    xt = ball_x + np.cos(phi) * d
    yt = ball_y + np.sin(phi) * d
    tta = _tta(px[:, None], py[:, None], pvx[:, None], pvy[:, None], xt[None, :], yt[None, :], params)  # (P,T)
    if params.exclude_passer:
        tta = np.where(passer[:, None], np.inf, tta)
    tta = np.nan_to_num(tta, nan=np.inf)

    player_poss = None
    for k in range(len(grids.v0)):
        tmp = tta - grids.t_ball[k][None, :]
        with np.errstate(over="ignore"):
            tmp = params.b0 + params.b1 * tmp
        with np.errstate(invalid="ignore"):
            tmp = _approx_sigmoid(tmp)
        tmp = np.nan_to_num(tmp, nan=0.0)
        ar = tmp / grids.rate_divisor[k]
        sum_att = np.nansum(np.where(att[:, None], ar, 0), axis=0)  # (T,)
        sum_def = np.nansum(np.where(~att[:, None], ar, 0), axis=0)
        int_att = scipy.integrate.cumulative_trapezoid(y=sum_att, x=grids.t_ball[k], initial=0)
        int_def = scipy.integrate.cumulative_trapezoid(y=sum_def, x=grids.t_ball[k], initial=0)
        cum_p0_att = np.exp(-int_att)
        cum_p0_def = np.exp(-int_def)
        opp = np.where(att[:, None], cum_p0_def[None, :], cum_p0_att[None, :])  # (P,T)
        dprdx = opp * ar * grids.dt0[k] / params.radial_gridsize
        player_poss = dprdx if player_poss is None else np.maximum(player_poss, dprdx)

    # v0 grid is non-empty (n_v0 >= 1, validated in PassSimParams), so the loop ran and this is set.
    player_poss = cast(np.ndarray, player_poss)
    attack_poss = np.nanmax(np.where(att[:, None], player_poss, 0), axis=0)  # (T,)
    cum = np.maximum.accumulate(attack_poss) * params.radial_gridsize
    on = (xt >= -_X_GOAL) & (xt <= _X_GOAL) & (yt >= -34.0) & (yt <= 34.0)
    cropped = cum.copy()
    for i in range(len(cum)):
        if not on[i]:
            cropped[i] = cropped[i - 1] if i > 0 else 0.0
    last = cropped[-1]
    return float(min(max(last, 0.0), 1.0))
