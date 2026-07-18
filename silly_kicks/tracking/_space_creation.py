"""Space Creation quantification (Fernandez & Bornn 2018).

Measures each player's contribution to the team's off-ball scoring
opportunity by computing differential OBSO: how much the team's
OBSO surface changes when that player is removed.

See NOTICE for full bibliographic citations.

References
----------
Fernandez, J. & Bornn, L. (2018). "Wide Open Spaces: A statistical
technique for measuring space creation in professional soccer."
MIT Sloan Sports Analytics Conference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._id_compat import ids_match

if TYPE_CHECKING:
    from .pitch_control import PitchControlCache


@dataclass(frozen=True)
class SpaceCreationParams:
    """Parameters for space creation computation.

    All spatial parameters are in meters on a [0, pitch_length] x [0, pitch_width]
    coordinate system.

    Examples
    --------
    >>> params = SpaceCreationParams()
    >>> params.pitch_length
    105.0
    """

    pitch_length: float = 105.0
    pitch_width: float = 68.0


def _unique_team_ids(frame: pd.DataFrame) -> np.ndarray:
    """Unique non-NaN team ids among non-ball rows of a single frame."""
    non_ball = frame.loc[frame["is_ball"] != True, "team_id"]  # noqa: E712
    return np.asarray(non_ball.dropna().unique())


def _resolve_opponent_team_id(frame: pd.DataFrame, attacking_team_id: int | str):
    """Resolve the opposing team id from a two-team frame (dtype-robust).

    Raises ``ValueError`` when the frame does not contain exactly two team
    ids (excluding ball rows) or when ``attacking_team_id`` does not uniquely
    match one of them — corrupt input must fail loud, not emit silent NaN.
    """
    uniq = _unique_team_ids(frame)
    if len(uniq) != 2:
        raise ValueError(
            "opponent perspective requires exactly two team ids in the frame "
            f"(excluding ball rows); found {list(uniq)!r}"
        )
    match_mask = ids_match(pd.Series(uniq), attacking_team_id).to_numpy()
    if match_mask.sum() != 1:
        raise ValueError(
            f"attacking_team_id {attacking_team_id!r} does not uniquely match the frame team ids {list(uniq)!r}"
        )
    return uniq[~match_mask][0]


def compute_space_created(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    *,
    ball_position: tuple[float, float] | None = None,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    params: SpaceCreationParams | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    obso_sigma_x: float = 26.25,
    obso_sigma_y: float = 17.0,
    pitch_control_cache: PitchControlCache | None = None,
    include_opponent_perspective: bool = False,
) -> pd.DataFrame:
    """Per-player space creation via leave-one-out differential OBSO.

    For each attacking-team outfield player, removes that player from the
    frame, re-computes pitch control, and measures the resulting OBSO gain
    attributable to the player's presence: ``space_created_m2`` (>= 0).

    With ``include_opponent_perspective=True``, the SAME leave-one-out is
    additionally evaluated on the opposing team's OBSO surface (the player
    acting as a defender of that surface), weighed by the opponent's OWN
    attacking geometry: the same transition/EPV grid artifacts MIRRORED along
    x to the goal the opponent attacks (the ball-distance weight stays
    ball-anchored). Grid resolution, sigmas, and pitch-control method are
    shared, so ``*_m2`` magnitudes are directly comparable — but the mirrored
    weighting makes the opponent LOO a genuine independent measurement (an
    unmirrored shared multiplier degenerates it to the exact pointwise
    negation of the team LOO; 4.23.0 defect). The opponent measurement is
    ``space_denied_m2_opponent`` (>= 0): opponent OBSO-weighted space the
    player's presence denies — the rest-defense reading.

    WHY exactly two columns (4.24.0 lean contract): the LOO is
    pointwise-monotone — removing a player can only DECREASE his own team's
    control and INCREASE the opponent's, everywhere, for every shipped
    pitch-control method. A team-side "destroyed" half and an opponent-side
    "created" half are therefore structurally 0, and net columns would be
    exact redundancies of the two live measurements; none of them are part
    of the contract (always-zero columns shipped 3.21.0-4.23.0 and were
    retired by lakehouse/owner decision).

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data with columns per TRACKING_FRAMES_COLUMNS.
    attacking_team_id : int or str
        Team in possession.
    ball_position : tuple[float, float] or None
        (x, y) in meters.  If None, inferred from ball row.
    transition_grid : np.ndarray or None
        Pre-computed ball transition probability grid.
    epv_grid : np.ndarray or None
        Pre-computed EPV grid.
    params : SpaceCreationParams or None
        Grid/pitch parameters.  None uses defaults.
    pitch_control_method : str
        Pitch control model (default ``"spearman"``).
    obso_sigma_x, obso_sigma_y : float
        Gaussian decay sigmas for OBSO distance weighting (meters).
    include_opponent_perspective : bool
        When True, also emit ``space_denied_m2_opponent`` (see above) and
        REQUIRE the frame to contain exactly two team ids (raises
        ``ValueError`` otherwise — corrupt input fails loud).

    Returns
    -------
    pd.DataFrame
        Columns: ``player_id``, ``team_id``, ``space_created_m2`` (+
        ``space_denied_m2_opponent`` when ``include_opponent_perspective=True``).

    Examples
    --------
    >>> result = compute_space_created(frame, attacking_team_id=1)
    >>> result.columns.tolist()
    ['player_id', 'team_id', 'space_created_m2']
    """
    from ._obso import _get_default_grids, _interpolate_grid
    from .pitch_control import PitchControlCache

    if params is None:
        params = SpaceCreationParams()

    opponent_team_id = None
    if include_opponent_perspective:
        # Loud two-team guard (corrupt frames never degrade to silent NaN).
        opponent_team_id = _resolve_opponent_team_id(frame, attacking_team_id)

    transition_grid, epv_grid = _get_default_grids(transition_grid, epv_grid)

    # Ensure velocity columns
    if "vx" not in frame.columns or "vy" not in frame.columns:
        frame = frame.copy()
        if "vx" not in frame.columns:
            frame["vx"] = 0.0
        if "vy" not in frame.columns:
            frame["vy"] = 0.0

    # Resolve ball position
    if ball_position is None:
        ball_rows = frame[frame["is_ball"] == True]  # noqa: E712
        if len(ball_rows) > 0:
            ball_position = (float(ball_rows.iloc[0]["x"]), float(ball_rows.iloc[0]["y"]))
        else:
            ball_position = (params.pitch_length / 2, params.pitch_width / 2)

    # 1. Compute baseline pitch control with per-player decomposition
    #    (decompose=True is free for Spearman/F&B — same influence computation,
    #    just retains the per-player arrays instead of discarding them)
    use_analytical = pitch_control_method in ("spearman", "fernandez_bornn")
    # Canonical-frame baseline via the shared cache (TF-7 shared surface). The
    # leave-one-out counterfactuals below operate on modified frames and stay
    # uncached.
    cache = pitch_control_cache if pitch_control_cache is not None else PitchControlCache()
    baseline_surface = cache.surface(
        frame,
        attacking_team_id,
        method=pitch_control_method,
        decompose=use_analytical,
        ball_position=ball_position,
    )

    grid_x = np.asarray(baseline_surface.grid_x)
    grid_y = np.asarray(baseline_surface.grid_y)
    ny, nx = baseline_surface.surface.shape

    # 2. Hoist loop-invariant: OBSO multiplier
    transition_interp = _interpolate_grid(transition_grid, (ny, nx))
    epv_interp = _interpolate_grid(epv_grid, (ny, nx))

    ball_x, ball_y = ball_position
    xx, yy = np.meshgrid(grid_x, grid_y)
    distance_weight = np.exp(
        -((xx - ball_x) ** 2) / (2.0 * obso_sigma_x**2) - (yy - ball_y) ** 2 / (2.0 * obso_sigma_y**2)
    )
    effective_transition = transition_interp * distance_weight
    max_trans = np.max(effective_transition)
    if max_trans > 1e-10:
        effective_transition = effective_transition / max_trans
    obso_multiplier = effective_transition * epv_interp  # (ny, nx) -- constant

    # Opponent perspective: the opponent's OWN attacking geometry — the SAME
    # transition/EPV artifacts mirrored along x (the opponent attacks the other
    # goal); the ball-distance weight stays ball-anchored. Without the mirror the
    # complementary PC surface makes the opponent LOO the exact pointwise negation
    # of the team LOO (informationally empty — lakehouse round-2 rejection).
    obso_multiplier_opponent = None
    if include_opponent_perspective:
        # Point reflection (ADR-041): the opponent attacks the other goal AND the y-axis
        # mirrors with it. Equivalent to the previous axis=1 flip for the y-SYMMETRIC
        # synthetic grids -- gated to rtol=1e-9 against a pre-change golden in
        # test_space_creation_mirror.py, where the measured float noise is 3.3e-16 -- but
        # CORRECT for an injected, y-asymmetric xT-derived surface (ADR-041 wires those in).
        # NOTE distance_weight below is deliberately NOT mirrored (it stays ball-anchored,
        # 4.24.0), and the two branches normalize by their OWN maxima; do not "fix" either.
        transition_opp = np.flip(transition_interp, axis=(0, 1))
        epv_opp = np.flip(epv_interp, axis=(0, 1))
        effective_transition_opp = transition_opp * distance_weight
        max_trans_opp = np.max(effective_transition_opp)
        if max_trans_opp > 1e-10:
            effective_transition_opp = effective_transition_opp / max_trans_opp
        obso_multiplier_opponent = effective_transition_opp * epv_opp

    # Baseline OBSO
    baseline_obso = np.clip(np.asarray(baseline_surface.surface) * obso_multiplier, 0.0, 1.0)

    # 3. Identify attacking-team players (including GK)
    atk_mask = ids_match(frame["team_id"], attacking_team_id) & (frame["is_ball"] != True)  # noqa: E712
    atk_players = frame.loc[atk_mask]

    if atk_players.empty:
        base_cols = ["player_id", "team_id", "space_created_m2"]
        if include_opponent_perspective:
            base_cols.append("space_denied_m2_opponent")
        return pd.DataFrame(columns=base_cols)

    # Cell area
    dx = float(grid_x[1] - grid_x[0]) if len(grid_x) > 1 else 1.0
    dy = float(grid_y[1] - grid_y[0]) if len(grid_y) > 1 else 1.0
    cell_area = dx * dy

    # 4. Leave-one-out: analytical delta (Spearman/F&B) or naive (Voronoi)
    if use_analytical:
        results = _analytical_leave_one_out(
            baseline_surface,
            baseline_obso,
            obso_multiplier,
            attacking_team_id,
            atk_players,
            cell_area,
            pitch_control_method,
            include_opponent=include_opponent_perspective,
            obso_multiplier_opponent=obso_multiplier_opponent,
        )
    else:
        results = _naive_leave_one_out(
            frame,
            baseline_obso,
            obso_multiplier,
            attacking_team_id,
            atk_players,
            cell_area,
            pitch_control_method,
            ball_position,
            opponent_team_id=opponent_team_id,
            obso_multiplier_opponent=obso_multiplier_opponent,
        )

    return pd.DataFrame(results)


def _analytical_leave_one_out(
    baseline_surface: object,
    baseline_obso: np.ndarray,
    obso_multiplier: np.ndarray,
    attacking_team_id: int | str,
    atk_players: pd.DataFrame,
    cell_area: float,
    method: Literal["spearman", "fernandez_bornn"],
    *,
    include_opponent: bool = False,
    obso_multiplier_opponent: np.ndarray | None = None,
) -> list[dict]:
    """Analytical per-player delta — 1 PC computation instead of N+1.

    Exploits additive decomposition of Spearman (ratio) and Fernandez-Bornn
    (sigmoid) pitch control models. Voronoi is NOT decomposable and must
    use the naive fallback.

    With ``include_opponent=True``, the opponent-attacking surface is the
    exact complement of the same decomposition (Spearman: def/(att+def);
    F&B: sigmoid(def-att)) — zero extra PC computations — but it is weighed by
    ``obso_multiplier_opponent`` (the x-MIRRORED transition/EPV geometry of the
    goal the opponent attacks). The mirror is what decouples the opponent LOO
    from the team LOO: under the shared unmirrored multiplier the complement
    made it the exact pointwise negation (4.23.0 defect).
    """
    # per_player_influence: (n_players, ny, nx) — post-GK-weighting influence
    ppi = np.asarray(baseline_surface.per_player_influence)  # type: ignore[union-attr]
    p_ids = np.asarray(baseline_surface.player_ids)  # type: ignore[union-attr]
    p_teams = np.asarray(baseline_surface.player_team_ids)  # type: ignore[union-attr]

    # Dtype-safe id match (ADR-019): canonical collapses 1.0/1/"1" consistently, so it is correct
    # across numeric/string callers (the old raw == broke when p_teams was string + team numeric).
    is_atk = ids_match(p_teams, attacking_team_id).to_numpy()
    att_total = ppi[is_atk].sum(axis=0)  # (ny, nx)
    def_total = ppi[~is_atk].sum(axis=0)  # (ny, nx)

    baseline_opp_obso = None
    if include_opponent:
        assert obso_multiplier_opponent is not None  # noqa: S101 — caller contract
        if method == "spearman":
            total = att_total + def_total
            safe_total = np.maximum(total, 1e-10)
            base_opp_pc = np.where(total > 1e-10, def_total / safe_total, 0.5)
        else:  # fernandez_bornn
            base_opp_pc = 1.0 / (1.0 + np.exp(-(def_total - att_total)))
        baseline_opp_obso = np.clip(base_opp_pc * obso_multiplier_opponent, 0.0, 1.0)

    results: list[dict] = []
    for player_row in atk_players.itertuples():
        pid = player_row.player_id

        # Find this player in the decomposed arrays
        pid_matches = np.flatnonzero(p_ids == pid)
        if len(pid_matches) == 0:
            # Player not in PC (dropped by NaN filter) → zero contribution
            row = {
                "player_id": pid,
                "team_id": attacking_team_id,
                "space_created_m2": 0.0,
            }
            if include_opponent:
                row["space_denied_m2_opponent"] = 0.0
            results.append(row)
            continue

        player_inf = ppi[pid_matches[0]]  # (ny, nx)
        removed_att = att_total - player_inf

        # Reconstruct PC surface without this player
        if method == "spearman":
            total = removed_att + def_total
            safe_total = np.maximum(total, 1e-10)
            removed_pc = np.where(total > 1e-10, removed_att / safe_total, 0.5)
        else:  # fernandez_bornn
            removed_pc = 1.0 / (1.0 + np.exp(-(removed_att - def_total)))

        removed_obso = np.clip(removed_pc * obso_multiplier, 0.0, 1.0)
        # Monotone LOO: removal only DECREASES own-team control, so the delta is
        # one-signed — created is the whole measurement (a "destroyed" half would
        # be structurally 0 and is deliberately NOT part of the contract; 4.24.0).
        delta = baseline_obso - removed_obso
        space_created = float(np.sum(np.maximum(delta, 0.0)) * cell_area)

        row = {
            "player_id": pid,
            "team_id": attacking_team_id,
            "space_created_m2": space_created,
        }

        if include_opponent:
            # Opponent-attacking surface without this player: the player leaves
            # the DEFENSE of that surface (same removal, complementary side).
            # Monotone in the other direction: removal only INCREASES opponent
            # control, so denial (the negative delta mass) is the whole measurement.
            if method == "spearman":
                total = removed_att + def_total
                safe_total = np.maximum(total, 1e-10)
                removed_opp_pc = np.where(total > 1e-10, def_total / safe_total, 0.5)
            else:  # fernandez_bornn
                removed_opp_pc = 1.0 / (1.0 + np.exp(-(def_total - removed_att)))
            removed_opp_obso = np.clip(removed_opp_pc * obso_multiplier_opponent, 0.0, 1.0)
            delta_opp = baseline_opp_obso - removed_opp_obso
            row["space_denied_m2_opponent"] = float(np.sum(np.abs(np.minimum(delta_opp, 0.0))) * cell_area)

        results.append(row)

    return results


def _naive_leave_one_out(
    frame: pd.DataFrame,
    baseline_obso: np.ndarray,
    obso_multiplier: np.ndarray,
    attacking_team_id: int | str,
    atk_players: pd.DataFrame,
    cell_area: float,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"],
    ball_position: tuple[float, float],
    *,
    opponent_team_id=None,
    obso_multiplier_opponent: np.ndarray | None = None,
) -> list[dict]:
    """Naive N-recompute fallback for non-decomposable models (Voronoi).

    When ``opponent_team_id`` is provided, the opponent-attacking surface is
    additionally recomputed per removal (explicit PC calls — correct for any
    method, including non-complementary ones) and weighed by
    ``obso_multiplier_opponent`` (the x-mirrored opponent attacking geometry —
    same semantics as the analytical path), yielding ``space_denied_m2_opponent``.
    """
    from .pitch_control import compute_pitch_control

    include_opponent = opponent_team_id is not None
    baseline_opp_obso = None
    if include_opponent:
        assert obso_multiplier_opponent is not None  # noqa: S101 — caller contract
        baseline_opp_surface = compute_pitch_control(
            frame,
            opponent_team_id,
            method=pitch_control_method,
            ball_position=ball_position,
        )
        baseline_opp_obso = np.clip(np.asarray(baseline_opp_surface.surface) * obso_multiplier_opponent, 0.0, 1.0)

    results: list[dict] = []
    for player_row in atk_players.itertuples():
        pid = player_row.player_id

        removed_frame = frame[
            ~((frame["player_id"] == pid) & (frame["is_ball"] != True))  # noqa: E712
        ]

        removed_surface = compute_pitch_control(
            removed_frame,
            attacking_team_id,
            method=pitch_control_method,
            ball_position=ball_position,
        )

        removed_obso = np.clip(np.asarray(removed_surface.surface) * obso_multiplier, 0.0, 1.0)
        delta = baseline_obso - removed_obso
        space_created = float(np.sum(np.maximum(delta, 0.0)) * cell_area)

        row = {
            "player_id": pid,
            "team_id": attacking_team_id,
            "space_created_m2": space_created,
        }

        if include_opponent:
            removed_opp_surface = compute_pitch_control(
                removed_frame,
                opponent_team_id,
                method=pitch_control_method,
                ball_position=ball_position,
            )
            removed_opp_obso = np.clip(np.asarray(removed_opp_surface.surface) * obso_multiplier_opponent, 0.0, 1.0)
            delta_opp = baseline_opp_obso - removed_opp_obso
            row["space_denied_m2_opponent"] = float(np.sum(np.abs(np.minimum(delta_opp, 0.0))) * cell_area)

        results.append(row)

    return results
