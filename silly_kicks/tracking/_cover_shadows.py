"""Cover shadow features — lane control + blocking score (TF-30).

Implements Cascioli, Wang, Stradiotti, Van Roy, Robberechts, Wouters,
Jaspers & Davis 2025 (Hudl/DTAI, KU Leuven).

See docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._id_compat import ids_match, same_id
from .pitch_control import PitchControlCache, PitchControlParams, compute_pitch_control
from .pitch_control._surface import PitchControlSurface

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverShadowParams:
    """Tunable parameters for cover shadow computation.

    All defaults from Cascioli et al. 2025 except where noted.
    Calibration deferred to TF-24 (Optuna).

    Examples
    --------
    >>> from silly_kicks.tracking._cover_shadows import CoverShadowParams
    >>> p = CoverShadowParams()
    >>> round(p.k_drag, 5)
    0.01383
    """

    # Corridor parameterization
    n_sample_points: int = 30
    cone_width_factor: float = 0.2

    # Ball drag model (Spearman 2017)
    air_density: float = 1.22
    drag_coefficient: float = 0.25
    ball_cross_section: float = 0.038
    ball_mass: float = 0.42
    ball_initial_speed: float = 12.0

    # Player TTI
    reaction_time: float = 0.7
    max_acceleration: float = 7.0
    max_speed: float = 12.0
    block_radius: float = 0.7

    # Probability conversion
    sigma: float = 0.20
    lambda_ctrl: float = 4.3

    # Man-marking filter
    man_mark_radius: float = 3.0
    man_mark_behind_offset: float = 1.0

    @property
    def k_drag(self) -> float:
        """Drag coefficient k = (rho * C_D * A) / (2 * m)."""
        return (self.air_density * self.drag_coefficient * self.ball_cross_section) / (2 * self.ball_mass)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneControlResult:
    """Per-(passer, receiver) lane blocking result.

    Examples
    --------
    >>> from silly_kicks.tracking._cover_shadows import LaneControlResult
    >>> r = LaneControlResult(0.8, 0.7, 0.9, 0.1, 0.2, 0.05,
    ...                       True, True, False)
    """

    p_blocked_center: float
    p_blocked_left: float
    p_blocked_right: float
    p_received_center: float
    p_received_left: float
    p_received_right: float
    is_blocked_any: bool
    is_blocked_majority: bool
    is_blocked_all: bool


@dataclass(frozen=True)
class BlockingScoreResult:
    """Result from compute_blocking_score — includes threat breakdown.

    Returning all three values avoids a redundant third PC call in callers
    that need both blocking_score and blocked_threat_fraction.

    Examples
    --------
    >>> from silly_kicks.tracking._cover_shadows import BlockingScoreResult
    >>> r = BlockingScoreResult(0.5, 1.2, 1.7)
    >>> r.blocked_threat_fraction
    0.294...
    """

    blocking_score: float
    threat_original: float
    threat_unblocked: float

    @property
    def blocked_threat_fraction(self) -> float:
        """blocking_score / threat_unblocked, 0.0 if threat_unblocked <= 0."""
        if self.threat_unblocked <= 0:
            return 0.0
        return self.blocking_score / self.threat_unblocked


# ---------------------------------------------------------------------------
# Ball drag model (Spearman 2017)
# ---------------------------------------------------------------------------


def ball_drag_time(
    distance: np.ndarray,
    params: CoverShadowParams | None = None,
) -> np.ndarray:
    """Ball travel time with quadratic air drag.

    T_ball(d) = expm1(k_drag * d) / (v0 * k_drag)

    Parameters
    ----------
    distance : np.ndarray
        Pass distances in meters (any shape).
    params : CoverShadowParams | None
        Parameters. None uses defaults.

    Returns
    -------
    np.ndarray
        Travel time in seconds, same shape as ``distance``.

    Examples
    --------
    >>> from silly_kicks.tracking._cover_shadows import ball_drag_time
    >>> import numpy as np
    >>> t = ball_drag_time(np.array([10.0]))
    >>> t[0] > 10.0 / 12.0  # drag slows ball
    True
    """
    p = params or CoverShadowParams()
    d = np.asarray(distance, dtype=np.float64)
    return np.expm1(p.k_drag * d) / (p.ball_initial_speed * p.k_drag)


# ---------------------------------------------------------------------------
# 3-phase player TTI (Cascioli et al. 2025)
# ---------------------------------------------------------------------------


def player_tti(
    player_pos: np.ndarray,
    player_vel: np.ndarray,
    targets: np.ndarray,
    *,
    is_defender: bool,
    params: CoverShadowParams | None = None,
) -> np.ndarray:
    """3-phase player time-to-intercept: react + accelerate + cruise.

    Parameters
    ----------
    player_pos : np.ndarray, shape (n_players, 2)
        Player positions in meters.
    player_vel : np.ndarray, shape (n_players, 2)
        Player velocities in m/s.
    targets : np.ndarray, shape (n_points, 2)
        Target positions in meters.
    is_defender : bool
        If True, apply block_radius advantage.
    params : CoverShadowParams | None
        Parameters. None uses defaults.

    Returns
    -------
    np.ndarray, shape (n_players, n_points)
        Time-to-intercept in seconds.

    Examples
    --------
    >>> from silly_kicks.tracking._cover_shadows import player_tti
    >>> import numpy as np
    >>> pos = np.array([[0.0, 0.0]])
    >>> vel = np.array([[0.0, 0.0]])
    >>> tgt = np.array([[5.0, 0.0]])
    >>> tti = player_tti(pos, vel, tgt, is_defender=False)
    >>> tti.shape
    (1, 1)
    """
    p = params or CoverShadowParams()

    # Position after reaction phase
    r_react = (
        player_pos[:, np.newaxis, :] + player_vel[:, np.newaxis, :] * p.reaction_time
    )  # (n_players, 1, 2) broadcast to (n_players, n_points, 2)

    delta = targets[np.newaxis, :, :] - r_react  # (n_players, n_points, 2)
    d = np.linalg.norm(delta, axis=2)  # (n_players, n_points)

    # Block radius advantage for defenders
    if is_defender:
        d_eff = np.maximum(d - p.block_radius, 0.0)
    else:
        d_eff = d.copy()

    # Unit direction toward target (safe against zero-distance)
    safe_d = np.where(d > 1e-12, d, 1.0)
    e_hat = delta / safe_d[:, :, np.newaxis]

    # Velocity component toward target, clamped >= 0
    v0 = np.sum(player_vel[:, np.newaxis, :] * e_hat, axis=2)
    v0 = np.maximum(v0, 0.0)

    # Time to accelerate from v0 to max_speed
    t_accel_full = np.maximum((p.max_speed - v0) / p.max_acceleration, 0.0)
    d_accel_full = v0 * t_accel_full + 0.5 * p.max_acceleration * t_accel_full**2

    # Case 1: already cruising (v0 >= max_speed)
    tti_cruise = p.reaction_time + d_eff / p.max_speed

    # Case 2: acceleration only (d_eff <= d_accel_full)
    discriminant = v0**2 + 2 * p.max_acceleration * d_eff
    tti_accel_only = p.reaction_time + (-v0 + np.sqrt(np.maximum(discriminant, 0.0))) / p.max_acceleration

    # Case 3: accelerate then cruise
    tti_accel_cruise = p.reaction_time + t_accel_full + (d_eff - d_accel_full) / p.max_speed

    result = np.where(
        v0 >= p.max_speed,
        tti_cruise,
        np.where(d_eff <= d_accel_full, tti_accel_only, tti_accel_cruise),
    )

    # Zero effective distance -> just reaction time
    result = np.where(d_eff <= 1e-12, p.reaction_time, result)

    return result


# ---------------------------------------------------------------------------
# LTR validation (same pattern as _off_ball_runs._validate_ltr and
# _defensive_line inline guard — caller-specific error message)
# ---------------------------------------------------------------------------


def _validate_ltr(frames: pd.DataFrame, caller: str = "_cover_shadows") -> None:
    """Raise ValueError if frames are not period-normalized (home attacks LTR).

    After ``play_left_to_right``, home-team rows have ``"ltr"`` and away-team
    rows have ``"rtl"`` — both valid in the period-normalized frame. Rejects
    frames with unexpected direction values or frames with only ``"rtl"``
    (period normalization not applied).
    """
    if "team_attacking_direction" in frames.columns:
        directions = set(frames["team_attacking_direction"].dropna().unique())
        valid = {"ltr", "rtl"}
        unexpected = directions - valid
        if unexpected:
            raise ValueError(
                f"{caller}: frames have unexpected team_attacking_direction "
                f"values: {sorted(unexpected)}. Expected 'ltr'/'rtl' only."
            )
        if directions and "ltr" not in directions:
            raise ValueError(
                f"{caller}: frames must be period-normalized "
                "(play_left_to_right). Found only 'rtl' direction values — "
                "no home-team rows with 'ltr'."
            )


# ---------------------------------------------------------------------------
# Man-marking filter
# ---------------------------------------------------------------------------


def _classify_man_markers(
    defenders: pd.DataFrame,
    attackers: pd.DataFrame,
    *,
    goal_x_own: float,
    params: CoverShadowParams,
) -> set:
    """Return set of player_ids that are man-marking (excluded from lane analysis).

    A defender is man-marking if within man_mark_radius of the point
    man_mark_behind_offset meters behind any attacker toward the defender's
    own goal. Each defender can mark at most one attacker (mutual exclusion
    via greedy nearest-first assignment).
    """
    if defenders.empty or attackers.empty:
        return set()

    # Unit toward own goal
    if goal_x_own < 52.5:
        toward_own_goal = np.array([-1.0, 0.0])
    else:
        toward_own_goal = np.array([1.0, 0.0])

    att_pos = attackers[["x", "y"]].to_numpy()
    def_pos = defenders[["x", "y"]].to_numpy()
    def_ids = defenders["player_id"].to_numpy()

    # Compute behind-points for all attackers
    behind_points = att_pos + params.man_mark_behind_offset * toward_own_goal

    # Build all (defender_idx, attacker_idx, distance) candidates within radius
    candidates = []
    for ai, bp in enumerate(behind_points):
        dists = np.linalg.norm(def_pos - bp, axis=1)
        for di in np.where(dists < params.man_mark_radius)[0]:
            candidates.append((di, ai, dists[di]))

    # Greedy nearest-first 1:1 assignment
    candidates.sort(key=lambda c: c[2])
    assigned_defenders: set[int] = set()
    assigned_attackers: set[int] = set()
    man_markers: set = set()

    for di, ai, _dist in candidates:
        if di in assigned_defenders or ai in assigned_attackers:
            continue
        assigned_defenders.add(di)
        assigned_attackers.add(ai)
        man_markers.add(def_ids[di])

    return man_markers


# ---------------------------------------------------------------------------
# Lane control primitive
# ---------------------------------------------------------------------------


def _lane_int_probs(
    targets: np.ndarray,
    defender_pos: np.ndarray,
    defender_vel: np.ndarray,
    attacker_pos: np.ndarray,
    attacker_vel: np.ndarray,
    *,
    params: CoverShadowParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-lane clamp-independent precompute (PR-S65).

    Returns ``(p_int_def, p_int_att, t_ball, p_ctrl)`` where ``p_int_*`` are the
    per-(player, point) sigmoid interception probabilities and
    ``p_ctrl[k] = 1 - exp(-lambda * dt_k)`` for ``dt_k > 0`` else ``0.0``
    (``p_ctrl[0] = 0.0``). A ``0.0`` entry reproduces the original ``dt_k <= 0``
    skip exactly (zero contribution, prior unchanged), so survival callers may
    treat it as a no-op step. None of these quantities depend on the clamped
    accumulation — see INV-1 in the design doc.

    See docs/superpowers/specs/2026-05-28-cover-shadows-leave-one-out-decouple-design.md.
    """
    n_points = targets.shape[0]

    # Ball travel time to each sample point
    d_from_passer = np.linalg.norm(targets - targets[0:1], axis=1)
    t_ball = ball_drag_time(d_from_passer, params)

    # Defender / attacker TTI (per-player, independent of which other players are present)
    tti_def = player_tti(defender_pos, defender_vel, targets, is_defender=True, params=params)
    tti_att = player_tti(attacker_pos, attacker_vel, targets, is_defender=False, params=params)

    # Sigmoid width + per-player interception probability
    s = np.sqrt(3.0) * params.sigma / np.pi

    def _p_int(tti_matrix: np.ndarray) -> np.ndarray:
        dt = t_ball[np.newaxis, :] - tti_matrix  # positive = arrives before ball
        return 1.0 / (1.0 + np.exp(-dt / s))

    p_int_def = _p_int(tti_def)  # (n_defenders, n_points)
    p_int_att = _p_int(tti_att)  # (n_attackers, n_points)

    # Per-step control probability; 0.0 where dt_k <= 0 (reproduces the original skip).
    dt = np.empty(n_points)
    dt[0] = 0.0
    dt[1:] = t_ball[1:] - t_ball[:-1]
    p_ctrl = np.where(dt > 0, 1.0 - np.exp(-params.lambda_ctrl * dt), 0.0)

    return p_int_def, p_int_att, t_ball, p_ctrl


def _lane_received_survival(
    p_int_def: np.ndarray,
    p_int_att: np.ndarray,
    p_ctrl: np.ndarray,
) -> tuple[float, float]:
    """Sequential clamped survival scan for one lane (PR-S65).

    Verbatim arithmetic of the pre-refactor ``_compute_lane_probabilities`` inner
    loop (sequential ``+=`` over players, ``min(prior + total, 1.0)`` clamp), so it
    is bit-identical to the original. Returns ``(p_blocked, p_received)``.
    """
    n_points = p_ctrl.shape[0]
    n_def = p_int_def.shape[0]
    n_att = p_int_att.shape[0]
    p_blocked = 0.0
    p_received = 0.0
    p_anyone_prior = 0.0
    for k in range(1, n_points):
        pc = p_ctrl[k]
        if pc <= 0.0:
            continue
        total_contrib_k = 0.0
        for j in range(n_def):
            contrib = float(p_int_def[j, k]) * pc * (1.0 - p_anyone_prior)
            p_blocked += contrib
            total_contrib_k += contrib
        for j in range(n_att):
            contrib = float(p_int_att[j, k]) * pc * (1.0 - p_anyone_prior)
            p_received += contrib
            total_contrib_k += contrib
        p_anyone_prior = min(p_anyone_prior + total_contrib_k, 1.0)
    return p_blocked, p_received


def _compute_lane_probabilities(
    targets: np.ndarray,
    defender_pos: np.ndarray,
    defender_vel: np.ndarray,
    attacker_pos: np.ndarray,
    attacker_vel: np.ndarray,
    *,
    params: CoverShadowParams,
) -> tuple[float, float]:
    """Compute P(blocked) and P(received) for one lane (one set of targets).

    Composition of ``_lane_int_probs`` (clamp-independent precompute) and
    ``_lane_received_survival`` (sequential clamped scan). Bit-identical to the
    pre-refactor implementation. Returns (p_blocked, p_received).
    """
    p_int_def, p_int_att, _t_ball, p_ctrl = _lane_int_probs(
        targets, defender_pos, defender_vel, attacker_pos, attacker_vel, params=params
    )
    return _lane_received_survival(p_int_def, p_int_att, p_ctrl)


def _lane_received_batched(
    p_int_def: np.ndarray,
    p_int_att: np.ndarray,
    p_ctrl: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Baseline + per-blocker leave-one-out p_received for one lane, vectorized (PR-S65).

    Returns ``(p_blocked_full, p_received_full, p_received_loo)`` where
    ``p_received_loo[m]`` is p_received with lane-blocker row ``m`` excluded.

    INV-1: the clamped recurrence is RE-RUN per variant (variant 0 = full racer set,
    variant ``m+1`` = exclude blocker ``m``), each tracked by an independent ``prior``.
    Excluding a blocker adjusts only the per-step PLAYER sum (``full_def - def_col``),
    never a post-clamp subtraction. Differs from the sequential scan only by float
    reduction order (well under rtol 1e-10 for ~10-element sums).

    See docs/superpowers/specs/2026-05-28-cover-shadows-leave-one-out-decouple-design.md.
    """
    n_points = p_ctrl.shape[0]
    nb = p_int_def.shape[0]
    nv = nb + 1  # variant 0 = full; variant m+1 = exclude blocker m
    prior = np.zeros(nv)
    p_blocked = np.zeros(nv)
    p_received = np.zeros(nv)
    att_sum_all = p_int_att.sum(axis=0)  # (n_points,)
    for k in range(1, n_points):
        pc = p_ctrl[k]
        if pc <= 0.0:
            continue
        def_col = p_int_def[:, k]  # (nb,)
        full_def = def_col.sum()
        def_sum = np.empty(nv)
        def_sum[0] = full_def
        def_sum[1:] = full_def - def_col  # exclude each blocker (per-step masked sum)
        att_sum = att_sum_all[k]
        one_minus_prior = 1.0 - prior
        blk = def_sum * pc * one_minus_prior
        rec = att_sum * pc * one_minus_prior
        p_blocked += blk
        p_received += rec
        prior = np.minimum(prior + blk + rec, 1.0)
    return float(p_blocked[0]), float(p_received[0]), p_received[1:]


def lane_control(
    frame: pd.DataFrame,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    *,
    home_team_id: int | str,
    attacking_team_id: int | str,
    params: CoverShadowParams | None = None,
) -> LaneControlResult:
    """Per-(passer, receiver) lane blocking probability.

    Parameters
    ----------
    frame : pd.DataFrame
        Single tracking frame (LTR-normalized).
    passer_xy : tuple[float, float]
        Passer position (x, y) in meters.
    receiver_xy : tuple[float, float]
        Receiver position (x, y) in meters.
    home_team_id : int | str
        Home team identifier (defends x=0).
    attacking_team_id : int | str
        Attacking team identifier.
    params : CoverShadowParams | None
        Parameters. None uses defaults.

    Returns
    -------
    LaneControlResult

    Examples
    --------
    >>> from silly_kicks.tracking._cover_shadows import lane_control
    >>> # See tests/tracking/test_cover_shadows.py for runnable examples.

    References
    ----------
    Cascioli et al. (2025). "Quantifying Off-Ball Defensive Impact through Cover Shadows."
    """
    _validate_ltr(frame, caller="lane_control")
    p = params or CoverShadowParams()

    passer = np.array(passer_xy, dtype=np.float64)
    receiver = np.array(receiver_xy, dtype=np.float64)
    pass_vec = receiver - passer
    pass_dist = np.linalg.norm(pass_vec)
    if pass_dist < 1e-6:
        return LaneControlResult(0, 0, 0, 0, 0, 0, False, False, False)

    u = pass_vec / pass_dist
    u_perp = np.array([-u[1], u[0]])
    half_width = p.cone_width_factor * pass_dist / 2.0

    # Generate sample points along 3 lines
    t = np.linspace(0.0, 1.0, p.n_sample_points)
    center = passer[np.newaxis, :] + t[:, np.newaxis] * pass_vec[np.newaxis, :]
    left = center + t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]
    right = center - t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]

    # Identify players
    players = frame[~frame["is_ball"].astype(bool)].copy()
    defenders_all = players[~ids_match(players["team_id"], attacking_team_id)].copy()
    attackers_all = players[ids_match(players["team_id"], attacking_team_id)].copy()

    # Exclude GK from lane-blocking defenders
    defenders_outfield = defenders_all[~defenders_all["is_goalkeeper"].astype(bool)]

    # Man-marking filter
    if same_id(attacking_team_id, home_team_id):
        goal_x_own = 105.0  # defenders' own goal
    else:
        goal_x_own = 0.0
    man_markers = _classify_man_markers(
        defenders_outfield,
        attackers_all,
        goal_x_own=goal_x_own,
        params=p,
    )
    lane_blockers = defenders_outfield[~defenders_outfield["player_id"].isin(man_markers)]

    if lane_blockers.empty:
        return LaneControlResult(0, 0, 0, 0, 0, 0, False, False, False)

    # Build position/velocity arrays
    def_pos = lane_blockers[["x", "y"]].to_numpy(dtype=np.float64)
    def_vel = lane_blockers[["vx", "vy"]].to_numpy(dtype=np.float64)

    # Attackers: exclude passer (already at start), include receiver + others
    att_pos = attackers_all[["x", "y"]].to_numpy(dtype=np.float64)
    att_vel = attackers_all[["vx", "vy"]].to_numpy(dtype=np.float64)

    # Compute probabilities for each lane
    results = []
    for lane_targets in (center, left, right):
        pb, pr = _compute_lane_probabilities(
            lane_targets,
            def_pos,
            def_vel,
            att_pos,
            att_vel,
            params=p,
        )
        results.append((pb, pr))

    p_bc, p_rc = results[0]
    p_bl, p_rl = results[1]
    p_br, p_rr = results[2]

    blocked_flags = [p_bc > p_rc, p_bl > p_rl, p_br > p_rr]
    n_blocked = sum(blocked_flags)

    return LaneControlResult(
        p_blocked_center=p_bc,
        p_blocked_left=p_bl,
        p_blocked_right=p_br,
        p_received_center=p_rc,
        p_received_left=p_rl,
        p_received_right=p_rr,
        is_blocked_any=n_blocked >= 1,
        is_blocked_majority=n_blocked >= 2,
        is_blocked_all=n_blocked == 3,
    )


# ---------------------------------------------------------------------------
# Grid-based Voronoi threat model
# ---------------------------------------------------------------------------


def _voronoi_threat(
    surface: PitchControlSurface,
    xt: ExpectedThreat,
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    home_team_id: int | str,
) -> tuple[float, dict]:
    """Compute threat via Voronoi-partitioned grid sum.

    Returns (total_threat, per_receiver_threat_dict).
    per_receiver_threat_dict maps player_id -> threat for dangerous receivers.
    """
    from scipy.spatial.distance import cdist

    players = frame[~frame["is_ball"].astype(bool)].copy()
    attackers = players[ids_match(players["team_id"], attacking_team_id)].copy()

    # Exclude GK from receiver set
    attackers_outfield = attackers[~attackers["is_goalkeeper"].astype(bool)]

    if attackers_outfield.empty:
        return 0.0, {}

    # Ball position
    ball_rows = frame[frame["is_ball"].astype(bool)]
    if ball_rows.empty or pd.isna(ball_rows.iloc[0]["x"]):
        return 0.0, {}
    ball_x = float(ball_rows.iloc[0]["x"])

    # Dangerous receivers: ahead of ball toward defending goal
    # After play_left_to_right, home team attacks toward high x.
    attacking_toward_high_x = same_id(attacking_team_id, home_team_id)
    if attacking_toward_high_x:
        dangerous = attackers_outfield[attackers_outfield["x"] > ball_x]
    else:
        dangerous = attackers_outfield[attackers_outfield["x"] < ball_x]

    if dangerous.empty:
        return 0.0, {}

    # Build grid coordinates from surface
    x_coords = surface.grid_x
    y_coords = surface.grid_y
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])  # (n_cells, 2)

    # Physically-oriented (ascending-y) threat grid -- ADR-041: the raw xt.interpolator()
    # output preserves xT's INVERTED row storage (row 0 = TOP of the pitch), which silently
    # y-mirrored this product against the ascending-y pitch-control surface. Lazy import:
    # a module-level xthreat import closes a real cycle (see _player_influence.py).
    from silly_kicks.xthreat import physical_grid

    xt_vals = physical_grid(xt, x_coords, y_coords, require_fitted=False)  # (ny, nx)
    if attacking_toward_high_x:
        threat_grid = xt_vals * surface.surface
    else:
        # BOTH axes: ADR-028's relation is a 180-degree point reflection (x->105-x AND
        # y->68-y), and an x-only mirror is exact only for a y-symmetric grid.
        threat_grid = xt_vals[::-1, ::-1] * surface.surface

    # Voronoi partition over ALL outfield attackers (not just dangerous)
    all_att_pos = attackers_outfield[["x", "y"]].to_numpy(dtype=np.float64)
    all_att_ids = attackers_outfield["player_id"].to_numpy()
    dists = cdist(grid_points, all_att_pos)  # (n_cells, n_all_attackers)
    nearest = np.argmin(dists, axis=1)  # (n_cells,)

    # Sum threat only for dangerous receivers
    dangerous_ids = set(dangerous["player_id"].tolist())
    per_receiver: dict = {}
    for i, pid in enumerate(all_att_ids):
        if pid not in dangerous_ids:
            continue
        mask = nearest == i
        threat_sum = float(threat_grid.ravel()[mask].sum())
        per_receiver[pid] = threat_sum

    total = sum(per_receiver.values())
    return total, per_receiver


# ---------------------------------------------------------------------------
# Blocking score primitive
# ---------------------------------------------------------------------------


def compute_blocking_score(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    defenders_to_remove: list[int | str] | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
    pitch_control_cache: PitchControlCache | None = None,
) -> BlockingScoreResult:
    """Blocking score: counterfactual threat reduction from defender removal.

    blocking_score = threat(frame_without_defenders) - threat(frame)

    Parameters
    ----------
    frame : pd.DataFrame
        Single tracking frame (LTR-normalized).
    attacking_team_id : int | str
        Attacking team identifier.
    xt : ExpectedThreat
        Fitted xT model for threat weighting.
    home_team_id : int | str
        Home team identifier (defends x=0).
    defenders_to_remove : list | None
        Explicit list of defender player_ids to remove. None auto-identifies
        lane-blockers (non-man-marking defenders).
    method : str
        Pitch control method.
    params : PitchControlParams | None
        Pitch control parameters.

    Returns
    -------
    BlockingScoreResult
        Contains blocking_score, threat_original, threat_unblocked.
        Avoids redundant PC call for callers needing blocked_threat_fraction.

    Examples
    --------
    >>> from silly_kicks.tracking._cover_shadows import compute_blocking_score
    >>> # See tests/tracking/test_cover_shadows.py for runnable examples.

    References
    ----------
    Cascioli et al. (2025).
    """
    _validate_ltr(frame, caller="compute_blocking_score")

    # Original (canonical-frame) threat — routed through the shared cache.
    # The counterfactual surface below is computed on a *modified* frame and
    # must NOT use the cache (different content at the same frame_id).
    cache = pitch_control_cache if pitch_control_cache is not None else PitchControlCache()
    surface_orig = cache.surface(
        frame,
        attacking_team_id,
        method=method,
        params=params,
    )
    threat_orig, _ = _voronoi_threat(
        surface_orig,
        xt,
        frame,
        attacking_team_id=attacking_team_id,
        home_team_id=home_team_id,
    )

    # No short-circuit on threat_orig == 0.0: removing defenders could
    # increase threat from 0 to >0 (defenders fully suppressing PC).
    # The counterfactual handles this naturally.

    # Identify defenders to remove
    if defenders_to_remove is None:
        # Auto-identify lane-blockers (non-man-marking, non-GK defenders)
        cs_params = CoverShadowParams()
        players = frame[~frame["is_ball"].astype(bool)]
        defenders_outfield = players[
            (~ids_match(players["team_id"], attacking_team_id)) & (~players["is_goalkeeper"].astype(bool))
        ]
        attackers = players[ids_match(players["team_id"], attacking_team_id)]
        if same_id(attacking_team_id, home_team_id):
            goal_x_own = 105.0
        else:
            goal_x_own = 0.0
        man_markers = _classify_man_markers(
            defenders_outfield,
            attackers,
            goal_x_own=goal_x_own,
            params=cs_params,
        )
        lane_blocker_ids = [pid for pid in defenders_outfield["player_id"] if pid not in man_markers]
    else:
        lane_blocker_ids = list(defenders_to_remove)

    if not lane_blocker_ids:
        return BlockingScoreResult(0.0, threat_orig, threat_orig)

    # Counterfactual: remove lane-blockers
    frame_reduced = frame[~frame["player_id"].isin(lane_blocker_ids)].copy()
    surface_reduced = compute_pitch_control(
        frame_reduced,
        attacking_team_id,
        method=method,
        params=params,
    )
    threat_unblocked, _ = _voronoi_threat(
        surface_reduced,
        xt,
        frame_reduced,
        attacking_team_id=attacking_team_id,
        home_team_id=home_team_id,
    )

    score = max(threat_unblocked - threat_orig, 0.0)
    return BlockingScoreResult(score, threat_orig, threat_unblocked)


# ---------------------------------------------------------------------------
# Per-frame cover-shadow computation (shared by aggregator + VAEP factory)
# ---------------------------------------------------------------------------

_CS_COL_NAMES = [
    "n_blocked_receivers",
    "n_potential_receivers",
    "blocking_score",
    "blocked_threat_fraction",
    "max_single_defender_blocking_score",
]


def _compute_cover_shadow_dict(
    frame_data: pd.DataFrame,
    passer_xy: tuple[float, float],
    attacking_team_id: int | str,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    decision_rule: str = "majority",
    detailed: bool = False,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> dict[str, float | int] | None:
    """Compute 5 cover-shadow values for a single frame + passer position.

    Returns a dict keyed by _CS_COL_NAMES, or None on degenerate input
    (missing velocity columns, no ball, NaN coordinates, etc.).
    Used by both ``add_cover_shadows`` and ``cover_shadow_xfns`` to avoid
    duplicating the per-frame computation.

    The ``detailed=False`` ``max_single_defender_blocking_score`` uses a fixed-cast
    leave-one-out: man-markers are classified once on the full frame (provably a
    no-op to re-classify per lane-blocker removal — no ripple), per-player
    interception probabilities are precomputed once per receiver, and each blocker's
    removal re-runs only the clamped survival recurrence with that blocker's row
    masked (vectorized; INV-1: never a post-clamp subtraction). Bit-identical within
    rtol 1e-10 to the prior per-(d, receiver) ``lane_control`` loop.
    See docs/superpowers/specs/2026-05-28-cover-shadows-leave-one-out-decouple-design.md.
    """
    # Velocity columns are required for lane_control TTI race
    if "vx" not in frame_data.columns or "vy" not in frame_data.columns:
        return None

    players = frame_data[~frame_data["is_ball"].astype(bool)]
    attackers = players[ids_match(players["team_id"], attacking_team_id)]
    attackers_outfield = attackers[~attackers["is_goalkeeper"].astype(bool)]

    ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
    if ball_rows.empty or pd.isna(ball_rows.iloc[0]["x"]):
        return None
    ball_x = float(ball_rows.iloc[0]["x"])

    # After play_left_to_right, home team attacks toward high x.
    attacking_toward_high_x = same_id(attacking_team_id, home_team_id)
    if attacking_toward_high_x:
        dangerous = attackers_outfield[attackers_outfield["x"] > ball_x]
    else:
        dangerous = attackers_outfield[attackers_outfield["x"] < ball_x]

    n_potential = len(dangerous)

    if n_potential == 0:
        return {
            "n_blocked_receivers": 0,
            "n_potential_receivers": 0,
            "blocking_score": 0.0,
            "blocked_threat_fraction": 0.0,
            "max_single_defender_blocking_score": 0.0,
        }

    # Baseline pass: lane_control per receiver on the FULL frame, used only for the
    # n_blocked decision (kept unchanged so n_blocked_receivers stays provably
    # bit-identical). The max_single leave-one-out below recomputes its own baseline
    # via the vectorized precompute. See spec §5 (deliberate deviation note).
    cs_params = CoverShadowParams()
    decision_attr = f"is_blocked_{decision_rule}"
    n_blocked = 0
    for _, recv_row in dangerous.iterrows():
        recv_xy = (float(recv_row["x"]), float(recv_row["y"]))
        lc = lane_control(
            frame_data,
            passer_xy,
            recv_xy,
            home_team_id=home_team_id,
            attacking_team_id=attacking_team_id,
            params=cs_params,
        )
        if getattr(lc, decision_attr):
            n_blocked += 1

    # Identify lane-blockers from man-marking filter
    defenders_outfield = players[
        (~ids_match(players["team_id"], attacking_team_id)) & (~players["is_goalkeeper"].astype(bool))
    ]
    if same_id(attacking_team_id, home_team_id):
        goal_x_own = 105.0
    else:
        goal_x_own = 0.0
    man_markers = _classify_man_markers(
        defenders_outfield,
        attackers,
        goal_x_own=goal_x_own,
        params=cs_params,
    )
    lane_blocker_ids = [pid for pid in defenders_outfield["player_id"] if pid not in man_markers]

    # Short-circuit: no lane-blockers -> score is 0 without wasting a PC call
    if not lane_blocker_ids:
        return {
            "n_blocked_receivers": n_blocked,
            "n_potential_receivers": n_potential,
            "blocking_score": 0.0,
            "blocked_threat_fraction": 0.0,
            "max_single_defender_blocking_score": 0.0,
        }

    bs_result = compute_blocking_score(
        frame_data,
        attacking_team_id,
        xt,
        home_team_id=home_team_id,
        defenders_to_remove=lane_blocker_ids,
        method=method,
        pitch_control_cache=pitch_control_cache,
    )

    # Max single-defender blocking score
    if detailed:
        max_def = 0.0
        for d_pid in lane_blocker_ids:
            d_result = compute_blocking_score(
                frame_data,
                attacking_team_id,
                xt,
                home_team_id=home_team_id,
                defenders_to_remove=[d_pid],
                method=method,
                pitch_control_cache=pitch_control_cache,
            )
            max_def = max(max_def, d_result.blocking_score)
    else:
        # Lightweight: classify man-markers once (lane_blocker_ids, the fixed racer set),
        # precompute per-player interception probs once per receiver, then a single
        # vectorized leave-one-out (re-run the clamped survival per excluded lane-blocker).
        # Bit-identical to the prior per-(d, receiver) lane_control loop within rtol 1e-10:
        # man-marking is invariant under lane-blocker removal (no ripple), so removing d
        # only drops d's row from the fixed racer set. See spec §2.1 / INV-1.
        # ADR-041: the RAW interpolator preserves xT's inverted row storage (row 0 = TOP of
        # pitch), so reading it here y-mirrored every receiver's threat -- and, for an
        # RTL-attacking team, omitted the 180-degree point reflection entirely, matching
        # NEITHER orientation. This is the production default branch (`detailed=False`), so
        # the defect reached `max_single_defender_blocking_score` on every action. Its
        # sibling `_voronoi_threat` was repaired in the same PR; this read was missed and
        # found by final-review. `values_at_points` is the per-point authority (exact
        # `rate()` semantics) -- no grid needed for a handful of receivers.
        from silly_kicks.xthreat import values_at_points

        kept = defenders_outfield[defenders_outfield["player_id"].isin(lane_blocker_ids)]
        lb_pos = kept[["x", "y"]].to_numpy(dtype=np.float64)
        lb_vel = kept[["vx", "vy"]].to_numpy(dtype=np.float64)
        att_pos = attackers[["x", "y"]].to_numpy(dtype=np.float64)
        att_vel = attackers[["vx", "vy"]].to_numpy(dtype=np.float64)
        n_lb = lb_pos.shape[0]
        passer = np.array(passer_xy, dtype=np.float64)

        score_per_blocker = np.zeros(n_lb)
        for _, recv_row in dangerous.iterrows():
            recv_x = float(recv_row["x"])
            recv_y = float(recv_row["y"])
            # Frame coords -> action-LTR before the lookup (ADR-028 point reflection).
            q_x, q_y = (recv_x, recv_y) if attacking_toward_high_x else (105.0 - recv_x, 68.0 - recv_y)
            recv_xt = float(values_at_points(xt, np.array([q_x]), np.array([q_y]), require_fitted=False)[0])

            receiver = np.array([recv_x, recv_y], dtype=np.float64)
            pass_vec = receiver - passer
            pass_dist = np.linalg.norm(pass_vec)
            if pass_dist < 1e-6:
                continue
            u = pass_vec / pass_dist
            u_perp = np.array([-u[1], u[0]])
            half_width = cs_params.cone_width_factor * pass_dist / 2.0
            t = np.linspace(0.0, 1.0, cs_params.n_sample_points)
            center = passer[np.newaxis, :] + t[:, np.newaxis] * pass_vec[np.newaxis, :]
            left = center + t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]
            right = center - t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]

            old_recv = 0.0
            new_recv = np.zeros(n_lb)
            for lane in (center, left, right):
                p_int_def, p_int_att, _t_ball, p_ctrl = _lane_int_probs(
                    lane, lb_pos, lb_vel, att_pos, att_vel, params=cs_params
                )
                _pb, base_rec, loo_rec = _lane_received_batched(p_int_def, p_int_att, p_ctrl)
                old_recv += base_rec
                new_recv += loo_rec

            delta = np.maximum(new_recv - old_recv, 0.0)
            score_per_blocker += recv_xt * delta

        max_def = float(score_per_blocker.max()) if n_lb > 0 else 0.0

    return {
        "n_blocked_receivers": n_blocked,
        "n_potential_receivers": n_potential,
        "blocking_score": bs_result.blocking_score,
        "blocked_threat_fraction": bs_result.blocked_threat_fraction,
        "max_single_defender_blocking_score": max_def,
    }
