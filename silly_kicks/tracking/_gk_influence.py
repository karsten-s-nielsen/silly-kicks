"""GK influence primitives (TF-15, GKDV Layer 1).

Three per-frame primitives measuring distinct aspects of GK spatial
contribution: threat-weighted pitch control share, uniquely reachable area,
and zone closing time.

See docs/superpowers/specs/2026-05-09-tf15-gk-influence-primitives-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._gk_resolve import GoalEndUnresolvedError, GoalMap

from ._defensive_line import select_back_line_players
from ._velocity_availability import velocity_unavailable_by_design, zero_velocity_if_unavailable
from .pitch_control import PitchControlCache, PitchControlParams, SpearmanParams
from .pitch_control._spearman import compute_tti

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat


# ---------------------------------------------------------------------------
# Goal geometry constants
# ---------------------------------------------------------------------------

_FIELD_WIDTH = spadlconfig.field_width  # 68.0
_POST_LEFT_Y = (_FIELD_WIDTH - 7.32) / 2  # 30.34
_POST_RIGHT_Y = (_FIELD_WIDTH + 7.32) / 2  # 37.66


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneClosingTime:
    """GK closing time to a single zone.

    Examples
    --------
    >>> zct = ZoneClosingTime(min_s=0.8, mean_s=1.2)
    """

    min_s: float
    mean_s: float


@dataclass(frozen=True)
class GkInfluence:
    """Per-frame GK influence measurement (all three primitives).

    Examples
    --------
    >>> gi = GkInfluence(
    ...     pitch_control_share_weighted=0.12,
    ...     reachable_area_m2=150.0,
    ...     closing_times={"six_yard_box": ZoneClosingTime(0.8, 1.2)},
    ... )
    """

    pitch_control_share_weighted: float
    reachable_area_m2: float
    closing_times: dict[str, ZoneClosingTime]


# ---------------------------------------------------------------------------
# Zone dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Zone:
    """A named set of target points for GK closing-time computation.

    Examples
    --------
    >>> zone = Zone.six_yard_box(goal_x=0.0)
    >>> zone.points.shape
    (9, 2)
    """

    name: str
    points: np.ndarray  # (N, 2) — x, y in meters, LTR-normalized

    def __post_init__(self) -> None:
        """Enforce array immutability."""
        self.points.flags.writeable = False

    @staticmethod
    def six_yard_box(goal_x: float) -> Zone:
        """~9 evenly-spaced points covering the six-yard box.

        Examples
        --------
        >>> Zone.six_yard_box(goal_x=0.0).points.shape
        (9, 2)
        """
        if goal_x == 0.0:
            xs = np.linspace(0.0, 5.5, 3)
        else:
            xs = np.linspace(99.5, 105.0, 3)
        ys = np.linspace(_POST_LEFT_Y, _POST_RIGHT_Y, 3)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        return Zone(name="six_yard_box", points=pts)

    @staticmethod
    def near_post(goal_x: float, ball_y: float | None = None) -> Zone:
        """~4 points near the goalpost closest to the ball.

        Examples
        --------
        >>> Zone.near_post(goal_x=0.0, ball_y=25.0).name
        'near_post'
        """
        near_y, _far_y = _resolve_near_far_post_y(ball_y)
        return _build_post_zone("near_post", goal_x, near_y)

    @staticmethod
    def far_post(goal_x: float, ball_y: float | None = None) -> Zone:
        """~4 points near the goalpost farthest from the ball.

        Examples
        --------
        >>> Zone.far_post(goal_x=0.0, ball_y=25.0).name
        'far_post'
        """
        _near_y, far_y = _resolve_near_far_post_y(ball_y)
        return _build_post_zone("far_post", goal_x, far_y)


def _resolve_near_far_post_y(ball_y: float | None) -> tuple[float, float]:
    """Determine near-post and far-post y based on ball position."""
    if ball_y is not None:
        d_left = abs(_POST_LEFT_Y - ball_y)
        d_right = abs(_POST_RIGHT_Y - ball_y)
        if d_left <= d_right:
            return _POST_LEFT_Y, _POST_RIGHT_Y
        else:
            return _POST_RIGHT_Y, _POST_LEFT_Y
    else:
        # Fixed proxy: left half -> near = left post
        return _POST_LEFT_Y, _POST_RIGHT_Y


def _build_post_zone(name: str, goal_x: float, post_y: float) -> Zone:
    """Build a ~4-point zone around one goalpost."""
    center_y = (_POST_LEFT_Y + _POST_RIGHT_Y) / 2
    if goal_x == 0.0:
        xs = np.array([0.0, 2.75])
    else:
        xs = np.array([102.25, 105.0])
    # Y corridor: from post_y toward center, 2 points
    mid_y = (post_y + center_y) / 2
    ys = np.array([post_y, mid_y])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return Zone(name=name, points=pts)


# ---------------------------------------------------------------------------
# Lightweight closing-time-only path (no pitch control overhead)
# ---------------------------------------------------------------------------


def compute_zone_closing_times(
    frame: pd.DataFrame,
    gk_player_id: int | str,
    zones: list[Zone],
    *,
    gk_reaction_time: float = 0.4,
    gk_max_acceleration: float = 7.0,
) -> dict[str, ZoneClosingTime]:
    """Compute GK closing time to zones WITHOUT pitch control overhead.

    Lightweight path for callers who only need closing times (not share
    or reachable area). Calls compute_tti directly.

    Examples
    --------
    Compute per-zone closing times for a keeper on a frame::

        from silly_kicks.tracking._gk_influence import compute_zone_closing_times, Zone
        cts = compute_zone_closing_times(frame, gk_player_id=1,
            zones=[Zone.six_yard_box(goal_x=0.0)])
    """
    players = frame[~frame["is_ball"].astype(bool)].dropna(subset=["x", "y"])
    # ADR-019: caller-supplied gk_player_id vs the frame's player_id column is a cross-source
    # compare -- route through ids_match (byte-identical on matched dtypes; resolves an Int64
    # frame id against a str/int query instead of silently raising 'not found').
    gk_mask = ids_match(players["player_id"], gk_player_id)
    if not gk_mask.any():
        raise ValueError(f"gk_player_id={gk_player_id!r} not found in frame")
    gk_row = players[gk_mask].iloc[0]
    gk_pos = np.array([[float(gk_row["x"]), float(gk_row["y"])]])
    gk_vel_x = float(gk_row.get("vx", 0.0)) if pd.notna(gk_row.get("vx")) else 0.0
    gk_vel_y = float(gk_row.get("vy", 0.0)) if pd.notna(gk_row.get("vy")) else 0.0
    gk_vel = np.array([[gk_vel_x, gk_vel_y]])

    result: dict[str, ZoneClosingTime] = {}
    for zone in zones:
        zone_tti = compute_tti(
            gk_pos,
            gk_vel,
            zone.points,
            gk_reaction_time,
            gk_max_acceleration,
        )[0]
        result[zone.name] = ZoneClosingTime(
            min_s=float(zone_tti.min()),
            mean_s=float(zone_tti.mean()),
        )
    return result


# ---------------------------------------------------------------------------
# Full per-frame GK influence computation
# ---------------------------------------------------------------------------


def compute_gk_influence(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    gk_player_id: int | str,
    xt: ExpectedThreat,
    *,
    goal_map: GoalMap,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
    zones: list[Zone] | None = None,
    tau_seconds: float = 1.0,
    gk_reaction_time: float = 0.4,
    gk_max_acceleration: float = 7.0,
    pitch_control_cache: PitchControlCache | None = None,
) -> GkInfluence:
    """Per-frame GK influence measurement (all three primitives).

    Uses back-line defenders only for reachable area (primitive b). This is
    an intentional Layer 1 approximation; full-team outfield coverage is
    deferred to GKDV Layer 2 (TF-19).

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data (TRACKING_FRAMES_COLUMNS schema).
    attacking_team_id : int | str
        Team currently in possession / attacking.
    gk_player_id : int | str
        The defending GK's player_id.
    xt : ExpectedThreat
        Pre-fit xT model for threat weighting.
    goal_map : GoalMap
        Defended-goal map for this match, from
        :func:`~silly_kicks.tracking.resolve_defended_goals`. **Required, with no default.**

        This is a per-FRAME function, so a default would let a caller omit the map and let the
        end be derived from whatever this one frame happens to show -- a different estimator,
        measured wrong for 7.1% of team-frames on SkillCorner and unresolvable for 35.7% of
        them (ADR-055; 0.0% on dense tracking -- the cost is provider-dependent). Build the map
        ONCE per match from the full frames and thread it in. The
        ``add_*`` aggregators accept ``goal_map=None`` precisely because they HAVE the full
        frames; this function does not.

        .. versionchanged:: ADR-055
           Replaces ``home_team_id``, which resolved the end as
           ``same_id(defending_team_id, home_team_id)`` -- identity-keyed direction, correct only
           while the frames are home-attacks-right.
    method : str, default "spearman"
        Pitch control model for primitive (a). Primitives (b) and (c)
        always use the Spearman kinematic TTI model regardless.
    params : PitchControlParams | None
        Optional pitch control params override.
    zones : list[Zone] | None
        Target zones for closing time. Default: [Zone.six_yard_box(goal_x)].
    tau_seconds : float, default 1.0
        TTI threshold for reachable area (primitive b).
    gk_reaction_time : float, default 0.4
        GK-specific reaction time (seconds).
    gk_max_acceleration : float, default 7.0
        GK-specific max acceleration (m/s^2).

    Returns
    -------
    GkInfluence

    Raises
    ------
    ValueError
        If gk_player_id is not found in the frame.
    GoalEndUnresolvedError
        If ``goal_map`` does not resolve the defending team's end, or the goal the attacking
        team attacks, in this (game, period). A ``ValueError`` subclass. Refusing is the point:
        both former guards spelled the miss as "attacking rightward", i.e. they failed OPEN.
        ``add_gk_influence`` catches this BY NAME at its edge and emits a NaN row.

    Examples
    --------
    Compute GK influence primitives for a keeper on a frame::

        from silly_kicks.tracking import resolve_defended_goals
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        goal_map = resolve_defended_goals(frames)   # ONCE per match, from the FULL frames
        gi = compute_gk_influence(
            frame, attacking_team_id=2, gk_player_id=1,
            xt=fitted_xt, goal_map=goal_map,
        )

    See NOTICE for full bibliographic citations.
    """
    # ADR-063: a declared-velocity-less provider (SB360 freeze frames) gets the zero-velocity
    # positional model; a frame merely missing vx/vy (a forgotten derive_velocities()) fails loud
    # here. The Tier-2 kinematic outputs (reachable area, closing time) are compute_tti-monotonic in
    # the along-target velocity component, so at zero velocity they are systematically BIASED
    # estimates of a physical fact -- SUPPRESSED to NaN below, the frame-level
    # validate_velocity_regime being the signal (PREFERRED D1). Only the model-relative Tier-1
    # pitch-control share is surfaced.
    frame = zero_velocity_if_unavailable(frame, method=method)
    _velocity_less = velocity_unavailable_by_design(frame)

    # --- Validate GK presence ---
    players = frame[~frame["is_ball"].astype(bool)]
    players = players.dropna(subset=["x", "y"])
    # ADR-019: caller-supplied gk_player_id vs the frame's player_id column is a cross-source
    # compare -- route through ids_match (byte-identical on matched dtypes; resolves an Int64
    # frame id against a str/int query instead of silently raising 'not found').
    gk_mask = ids_match(players["player_id"], gk_player_id)
    if not gk_mask.any():
        raise ValueError(
            f"gk_player_id={gk_player_id!r} not found in frame (available: {players['player_id'].tolist()})"
        )

    gk_row = players[gk_mask].iloc[0]
    gk_pos = np.array([[float(gk_row["x"]), float(gk_row["y"])]])
    gk_vel_x = float(gk_row.get("vx", 0.0)) if pd.notna(gk_row.get("vx")) else 0.0
    gk_vel_y = float(gk_row.get("vy", 0.0)) if pd.notna(gk_row.get("vy")) else 0.0
    gk_vel = np.array([[gk_vel_x, gk_vel_y]])

    # --- Goal-end resolution ---
    defending_team_id = gk_row["team_id"]
    _gid, _pid = frame["game_id"].iloc[0], frame["period_id"].iloc[0]
    goal_x = goal_map.get(_gid, _pid, defending_team_id, allow_guess=True)
    if goal_x is None:
        # PRECONDITION. The add_* edge catches this by name and emits a NaN row; reaching
        # here from a direct caller is a programming error, the same shape as the
        # "gk_player_id not found in frame" raise above.
        raise GoalEndUnresolvedError(
            f"compute_gk_influence: goal_map does not resolve team {defending_team_id!r} in "
            f"(game={_gid}, period={_pid})."
        )

    # --- Default zones ---
    if zones is None:
        zones = [Zone.six_yard_box(goal_x)]

    # --- Primitive (a): threat-weighted pitch control share ---
    # Canonical-frame surface — route through the shared cache so other
    # families (and other actions on this frame) reuse it (TF-7 shared surface).
    cache = pitch_control_cache if pitch_control_cache is not None else PitchControlCache()
    surface = cache.surface(
        frame,
        attacking_team_id,
        method=method,
        params=params,
        decompose=True,
    )

    gk_surface = surface.player_surface(gk_player_id)  # (ny, nx)

    # Team influence: sum over teammates (same team_id as GK)
    team_surface = np.zeros_like(gk_surface)
    if (
        surface.player_ids is not None
        and surface.player_team_ids is not None
        and surface.per_player_influence is not None
    ):
        # frame-vs-frame (both from the same surface/frame) -> dtype-consistent by
        # construction, raw compare (ADR-019 only governs cross-source/boundary seams).
        team_mask_arr = surface.player_team_ids == defending_team_id
        for idx in np.flatnonzero(team_mask_arr):
            team_surface += surface.per_player_influence[idx]

    # Per-cell share with threshold guard
    safe_team = np.where(team_surface < 1e-8, np.inf, team_surface)
    share_grid = np.where(team_surface < 1e-8, 0.0, gk_surface / safe_team)

    # Physically-oriented (ascending-y) threat grid -- ADR-041. The raw
    # xt.interpolator() output preserves xT's INVERTED row storage (row 0 = TOP of the
    # pitch), which silently y-mirrored this fusion against the ascending-y pitch-control
    # surface; physical_grid neutralizes the inversion once, at xthreat's boundary, and
    # fail-closes on an unfitted/None model. Lazy import: a module-level xthreat import
    # here closes a real cycle (see _player_influence.py).
    from silly_kicks.xthreat import physical_grid

    threat_grid = physical_grid(xt, surface.grid_x, surface.grid_y, require_fitted=False)  # (ny, nx)

    # Away-team attack: BOTH axes (ADR-028 is a 180-degree point reflection, x->105-x AND
    # y->68-y). An x-only mirror is exact only for a y-symmetric grid, which a fitted xT
    # merely APPROXIMATES -- the same incomplete repair ADR-041 corrected elsewhere.
    # Direction from the FRAMES via the seam. NOTE `acting_team_attacks_rtl` does NOT fit
    # here: it is (actions, frames) -> Series, i.e. per-ACTION, and this is a per-frame call.
    _attacked = goal_map.attacked_goal(_gid, _pid, attacking_team_id, allow_guess=True)
    if _attacked is None:
        # Explicit: `if _attacked == 0.0` alone would fail OPEN, silently choosing
        # 'attacking rightward' for an unresolved direction.
        raise GoalEndUnresolvedError(
            f"compute_gk_influence: goal_map does not resolve the goal attacked by "
            f"{attacking_team_id!r} in (game={_gid}, period={_pid})."
        )
    if _attacked == 0.0:
        threat_grid = threat_grid[::-1, ::-1]

    # Weighted average
    cell_area = surface.cell_area
    threat_weight = threat_grid * cell_area
    total_weight = threat_weight.sum()

    if total_weight < 1e-8:
        pitch_control_share_weighted = float("nan")
    else:
        pitch_control_share_weighted = float((share_grid * threat_weight).sum() / total_weight)

    # --- Primitive (b): reachable area ---
    # Defender TTI always uses Spearman kinematic model
    sp = SpearmanParams() if not isinstance(params, SpearmanParams) else params

    # GK TTI to all grid cells
    grid_x = surface.grid_x
    grid_y = surface.grid_y
    gx, gy = np.meshgrid(grid_x, grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])

    tti_gk = compute_tti(gk_pos, gk_vel, targets, gk_reaction_time, gk_max_acceleration)
    tti_gk = tti_gk[0]  # (n_targets,)

    # Back-line defenders TTI
    # Direction from the RESOLVED end, not from team identity. `goal_x` is this keeper's own
    # goal, looked up from the goal map above; `goal_x == 0.0` is therefore the same fact the
    # helper used to infer as `same_id(team_id, home_team_id)` -- minus the D3 assumption that
    # the frames are home-attacks-right.
    back_line = select_back_line_players(
        frame,
        team_id=defending_team_id,
        defends_x0=goal_x == 0.0,
    )

    if len(back_line) > 0:
        def_pos = back_line[["x", "y"]].to_numpy(dtype="float64")
        vx_col = back_line["vx"].to_numpy(dtype="float64") if "vx" in back_line.columns else np.zeros(len(back_line))
        vy_col = back_line["vy"].to_numpy(dtype="float64") if "vy" in back_line.columns else np.zeros(len(back_line))
        def_vel = np.column_stack([np.nan_to_num(vx_col), np.nan_to_num(vy_col)])

        tti_defenders = compute_tti(
            def_pos,
            def_vel,
            targets,
            sp.reaction_time,
            sp.max_acceleration,
        )
        min_tti_def = tti_defenders.min(axis=0)  # (n_targets,)

        # Cells where GK can reach within tau but no defender can
        gk_reachable = tti_gk <= tau_seconds
        def_not_reachable = min_tti_def > tau_seconds
        unique_cells = gk_reachable & def_not_reachable
    else:
        unique_cells = tti_gk <= tau_seconds

    # Tier-2 SUPPRESSION (ADR-063 PREFERRED D1): on a declared-velocity-less frame the zero-
    # velocity reachable area is a systematically biased physical estimate, withheld as NaN.
    reachable_area_m2 = float("nan") if _velocity_less else float(unique_cells.sum() * cell_area)

    # --- Primitive (c): zone closing times ---
    # The zone loop still runs on the velocity-less branch so the zone-keyed COLUMNS exist (as NaN)
    # rather than being absent -- a consumer sees an honest NaN, not a missing column.
    closing_times: dict[str, ZoneClosingTime] = {}
    for zone in zones:
        zone_tti = compute_tti(
            gk_pos,
            gk_vel,
            zone.points,
            gk_reaction_time,
            gk_max_acceleration,
        )[0]  # (n_zone_points,)
        closing_times[zone.name] = ZoneClosingTime(
            min_s=float("nan") if _velocity_less else float(zone_tti.min()),
            mean_s=float("nan") if _velocity_less else float(zone_tti.mean()),
        )

    return GkInfluence(
        pitch_control_share_weighted=pitch_control_share_weighted,
        reachable_area_m2=reachable_area_m2,
        closing_times=closing_times,
    )
