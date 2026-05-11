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

from .pitch_control import PitchControlParams, compute_pitch_control
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
    """Raise ValueError if frames contain non-LTR direction values."""
    if "team_attacking_direction" in frames.columns:
        directions = frames["team_attacking_direction"].dropna().unique()
        non_ltr = [d for d in directions if d != "ltr"]
        if non_ltr:
            raise ValueError(
                f"{caller}: frames must be LTR-normalized "
                "(play_left_to_right). Found non-'ltr' values in "
                f"team_attacking_direction: {non_ltr}"
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
    own goal.
    """
    if defenders.empty or attackers.empty:
        return set()

    # Unit toward own goal
    if goal_x_own < 52.5:
        toward_own_goal = np.array([-1.0, 0.0])
    else:
        toward_own_goal = np.array([1.0, 0.0])

    man_markers: set = set()
    att_pos = attackers[["x", "y"]].to_numpy()
    def_pos = defenders[["x", "y"]].to_numpy()
    def_ids = defenders["player_id"].to_numpy()

    for a_xy in att_pos:
        behind_point = a_xy + params.man_mark_behind_offset * toward_own_goal
        dists = np.linalg.norm(def_pos - behind_point, axis=1)
        close_mask = dists < params.man_mark_radius
        for pid in def_ids[close_mask]:
            man_markers.add(pid)

    return man_markers


# ---------------------------------------------------------------------------
# Lane control primitive
# ---------------------------------------------------------------------------


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

    Returns (p_blocked, p_received).
    """
    n_points = targets.shape[0]

    # Ball travel time to each sample point
    d_from_passer = np.linalg.norm(
        targets - targets[0:1],
        axis=1,
    )
    t_ball = ball_drag_time(d_from_passer, params)

    # Defender TTI
    tti_def = player_tti(
        defender_pos,
        defender_vel,
        targets,
        is_defender=True,
        params=params,
    )  # (n_defenders, n_points)

    # Attacker TTI (passer excluded — only the receiver matters,
    # but we pass all attackers for safety)
    tti_att = player_tti(
        attacker_pos,
        attacker_vel,
        targets,
        is_defender=False,
        params=params,
    )  # (n_attackers, n_points)

    # Sigmoid width
    s = np.sqrt(3.0) * params.sigma / np.pi

    # Per-player interception probability
    def _p_int(tti_matrix: np.ndarray) -> np.ndarray:
        """Sigmoid interception probability for each (player, point)."""
        dt = t_ball[np.newaxis, :] - tti_matrix  # positive = arrives before ball
        return 1.0 / (1.0 + np.exp(-dt / s))

    p_int_def = _p_int(tti_def)  # (n_defenders, n_points)
    p_int_att = _p_int(tti_att)  # (n_attackers, n_points)

    # Sequential integration along lane.
    # All players at point k share the same P_anyone_prior from points < k.
    # Accumulate all contributions at k, then update p_anyone_prior after.
    p_blocked = 0.0
    p_received = 0.0
    p_anyone_prior = 0.0

    for k in range(1, n_points):
        dt_k = t_ball[k] - t_ball[k - 1]
        if dt_k <= 0:
            continue
        p_ctrl = 1.0 - np.exp(-params.lambda_ctrl * dt_k)

        total_contrib_k = 0.0
        for j in range(len(defender_pos)):
            contrib = float(p_int_def[j, k]) * p_ctrl * (1.0 - p_anyone_prior)
            p_blocked += contrib
            total_contrib_k += contrib

        for j in range(len(attacker_pos)):
            contrib = float(p_int_att[j, k]) * p_ctrl * (1.0 - p_anyone_prior)
            p_received += contrib
            total_contrib_k += contrib

        p_anyone_prior = min(p_anyone_prior + total_contrib_k, 1.0)

    return p_blocked, p_received


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
    defenders_all = players[players["team_id"] != attacking_team_id].copy()
    attackers_all = players[players["team_id"] == attacking_team_id].copy()

    # Exclude GK from lane-blocking defenders
    defenders_outfield = defenders_all[~defenders_all["is_goalkeeper"].astype(bool)]

    # Man-marking filter
    if str(attacking_team_id) == str(home_team_id):
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
    attackers = players[players["team_id"] == attacking_team_id].copy()

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
    attacking_toward_high_x = str(attacking_team_id) == str(home_team_id)
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

    # xT grid (interpolated to PC grid coords)
    xt_interp = xt.interpolator()
    xt_vals = xt_interp(x_coords, y_coords)  # (ny, nx)
    if attacking_toward_high_x:
        threat_grid = xt_vals * surface.surface
    else:
        threat_grid = xt_vals[:, ::-1] * surface.surface

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

    # Original threat
    surface_orig = compute_pitch_control(
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
            (players["team_id"] != attacking_team_id) & (~players["is_goalkeeper"].astype(bool))
        ]
        attackers = players[players["team_id"] == attacking_team_id]
        if str(attacking_team_id) == str(home_team_id):
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
) -> dict[str, float | int] | None:
    """Compute 5 cover-shadow values for a single frame + passer position.

    Returns a dict keyed by _CS_COL_NAMES, or None on degenerate input
    (missing velocity columns, no ball, NaN coordinates, etc.).
    Used by both ``add_cover_shadows`` and ``cover_shadow_xfns`` to avoid
    duplicating the per-frame computation.
    """
    # Velocity columns are required for lane_control TTI race
    if "vx" not in frame_data.columns or "vy" not in frame_data.columns:
        return None

    players = frame_data[~frame_data["is_ball"].astype(bool)]
    attackers = players[players["team_id"] == attacking_team_id]
    attackers_outfield = attackers[~attackers["is_goalkeeper"].astype(bool)]

    ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
    if ball_rows.empty or pd.isna(ball_rows.iloc[0]["x"]):
        return None
    ball_x = float(ball_rows.iloc[0]["x"])

    # After play_left_to_right, home team attacks toward high x.
    attacking_toward_high_x = str(attacking_team_id) == str(home_team_id)
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

    # Lane control for each receiver
    cs_params = CoverShadowParams()
    decision_attr = f"is_blocked_{decision_rule}"
    n_blocked = 0
    lane_results: list[tuple] = []  # (receiver_pid, LaneControlResult)
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
        lane_results.append((recv_row["player_id"], lc))
        if getattr(lc, decision_attr):
            n_blocked += 1

    # Identify lane-blockers from man-marking filter
    defenders_outfield = players[(players["team_id"] != attacking_team_id) & (~players["is_goalkeeper"].astype(bool))]
    if str(attacking_team_id) == str(home_team_id):
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
            )
            max_def = max(max_def, d_result.blocking_score)
    else:
        # Lightweight approximation: for each lane-blocker d, re-run
        # lane_control without d to compute delta_P_received per receiver.
        # score_d = sum_r xT(r) * delta_P_received_r
        xt_interp = xt.interpolator()  # type: ignore[union-attr]
        max_approx = 0.0
        for d_pid in lane_blocker_ids:
            frame_without_d = frame_data[frame_data["player_id"] != d_pid]
            score_d = 0.0
            for recv_pid, lc_orig in lane_results:
                recv_rows = dangerous[dangerous["player_id"] == recv_pid]
                if recv_rows.empty:
                    continue
                recv_x = float(recv_rows.iloc[0]["x"])
                recv_y = float(recv_rows.iloc[0]["y"])
                recv_xt = float(
                    xt_interp(
                        np.array([recv_x]),
                        np.array([recv_y]),
                    )[0, 0]
                )
                lc_new = lane_control(
                    frame_without_d,
                    passer_xy,
                    (recv_x, recv_y),
                    home_team_id=home_team_id,
                    attacking_team_id=attacking_team_id,
                    params=cs_params,
                )
                old_recv = lc_orig.p_received_center + lc_orig.p_received_left + lc_orig.p_received_right
                new_recv = lc_new.p_received_center + lc_new.p_received_left + lc_new.p_received_right
                delta_p = max(new_recv - old_recv, 0.0)
                score_d += recv_xt * delta_p
            max_approx = max(max_approx, score_d)
        max_def = max_approx

    return {
        "n_blocked_receivers": n_blocked,
        "n_potential_receivers": n_potential,
        "blocking_score": bs_result.blocking_score,
        "blocked_threat_fraction": bs_result.blocked_threat_fraction,
        "max_single_defender_blocking_score": max_def,
    }
