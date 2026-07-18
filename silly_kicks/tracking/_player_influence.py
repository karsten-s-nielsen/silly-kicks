"""Per-player influence primitives (TF-36 + TF-33).

Per-frame computation of off-ball xT (threat-weighted pitch control share)
and uniquely reachable area for all outfield players. Both metrics share a
single compute_pitch_control(decompose=True) call per frame.

See docs/superpowers/specs/2026-05-23-tf36-tf33-player-influence-off-ball-xt-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._id_compat import same_id
from .pitch_control import PitchControlCache, PitchControlParams, PitchControlSurface, SpearmanParams
from .pitch_control._spearman import compute_tti

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat


@dataclass(frozen=True)
class PlayerInfluence:
    """Per-player per-frame influence measurement.

    Examples
    --------
    >>> pi = PlayerInfluence(off_ball_xt=0.35, reachable_area_m2=120.0)
    """

    off_ball_xt: float
    reachable_area_m2: float


def compute_player_influence(
    frame: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    attacking_team_id: int | str,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
    surface: PitchControlSurface | None = None,
    tau_seconds: float = 1.0,
    reaction_time: float | None = None,
    max_acceleration: float | None = None,
    pitch_control_cache: PitchControlCache | None = None,
) -> dict[int | str, PlayerInfluence]:
    """Per-frame influence for all outfield players.

    Computes off-ball xT (threat-weighted pitch control share) and uniquely
    reachable area for every non-GK, non-ball player in the frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data (TRACKING_FRAMES_COLUMNS schema).
    xt : ExpectedThreat
        Pre-fit xT model for threat weighting.
    attacking_team_id : int | str
        Team currently in possession.
    home_team_id : int | str
        Home team identifier for goal-end / xT orientation.
    method : str, default "spearman"
        Pitch control model (ignored when ``surface`` is provided).
    params : PitchControlParams | None
        Pitch control params override (ignored when ``surface`` is provided).
    surface : PitchControlSurface | None
        Pre-computed decomposed pitch control surface. When provided,
        ``method`` and ``params`` are ignored. Must have been computed with
        ``decompose=True``.
    tau_seconds : float, default 1.0
        TTI threshold for uniquely reachable area.
    reaction_time : float | None
        Outfield reaction time. Defaults to ``SpearmanParams().reaction_time``.
    max_acceleration : float | None
        Outfield max acceleration. Defaults to ``SpearmanParams().max_acceleration``.

    Returns
    -------
    dict[int | str, PlayerInfluence]
        Mapping from player_id to influence metrics. GKs and ball excluded.

    Examples
    --------
    >>> from silly_kicks.tracking._player_influence import compute_player_influence
    >>> result = compute_player_influence(
    ...     frame, xt, attacking_team_id=1, home_team_id=1,
    ... )

    See NOTICE for full bibliographic citations.
    """
    sp_defaults = SpearmanParams()
    rt = reaction_time if reaction_time is not None else sp_defaults.reaction_time
    ma = max_acceleration if max_acceleration is not None else sp_defaults.max_acceleration

    # --- Pitch control surface ---
    if surface is not None:
        pc = surface
    else:
        # Canonical-frame surface via the shared cache (TF-7 shared surface).
        cache = pitch_control_cache if pitch_control_cache is not None else PitchControlCache()
        pc = cache.surface(
            frame,
            attacking_team_id,
            method=method,
            params=params,
            decompose=True,
        )

    # --- xT interpolation ---
    # Lazy import (ADR-041): a MODULE-level xthreat import here closes a real cycle --
    # xthreat/_grid imports spadl.config, spadl/__init__ imports tracking, and
    # tracking/__init__ imports this module, so `import silly_kicks.xthreat` would re-enter
    # xthreat while it is still initializing. Same idiom as tracking/_xt_gk.py.
    from silly_kicks.xthreat import physical_grid

    # Physically-oriented (ascending-y) threat grid -- ADR-041. The raw
    # xt.interpolator() output preserves xT's INVERTED row storage (row 0 = TOP of the
    # pitch), which silently y-mirrored this fusion against the ascending-y pitch-control
    # surfaces below; it stayed invisible only because a fitted xT surface is close to
    # y-symmetric. physical_grid neutralizes the inversion once, at xthreat's boundary,
    # and also fail-closes on an unfitted/None model.
    threat_grid = physical_grid(xt, pc.grid_x, pc.grid_y)  # (ny, nx)

    # xT reflection for away-team attack: BOTH axes (ADR-041 second pass). ADR-028's
    # action-LTR <-> frame relation is x->105-x AND y->68-y, so an x-only mirror is exact
    # only for a y-symmetric grid -- true of the ramp-style fixtures and nearly true of a
    # fitted xT, which is how it hid. Pinned by
    # test_player_influence_orientation.py::test_away_attack_reflects_the_threat_grid_on_BOTH_axes.
    if not same_id(attacking_team_id, home_team_id):
        threat_grid = threat_grid[::-1, ::-1]

    cell_area = pc.cell_area

    # --- Identify outfield players ---
    players = frame[~frame["is_ball"].astype(bool)].copy()
    players = players[~players["is_goalkeeper"].astype(bool)]
    players = players.dropna(subset=["x", "y", "team_id"])

    if len(players) == 0:
        return {}

    # --- Off-ball xT (from decomposed PC surface) ---
    off_ball_xt_map: dict[int | str, float] = {}
    for _, row in players.iterrows():
        pid = row["player_id"]
        try:
            ps = pc.player_surface(pid)  # (ny, nx)
        except ValueError:
            off_ball_xt_map[pid] = 0.0
            continue
        off_ball_xt_map[pid] = float((ps * threat_grid * cell_area).sum())

    # --- Uniquely reachable area (team-TTI-matrix optimization) ---
    reachable_map: dict[int | str, float] = {}

    # Build grid targets
    gx, gy = np.meshgrid(pc.grid_x, pc.grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])  # (n_cells, 2)

    # Process per-team
    for team_id in players["team_id"].unique():
        # frame-vs-frame (team_id is a value FROM players["team_id"]) -> raw compare.
        team_players = players[players["team_id"] == team_id]
        n_team = len(team_players)
        pids = team_players["player_id"].values

        # Build position + velocity arrays, NaN vx/vy -> 0.0
        pos = team_players[["x", "y"]].to_numpy(dtype="float64")
        vx_arr = team_players["vx"].to_numpy(dtype="float64") if "vx" in team_players.columns else np.zeros(n_team)
        vy_arr = team_players["vy"].to_numpy(dtype="float64") if "vy" in team_players.columns else np.zeros(n_team)
        vel = np.column_stack([np.nan_to_num(vx_arr), np.nan_to_num(vy_arr)])

        # Compute full-team TTI matrix: (n_team, n_cells)
        tti_matrix = compute_tti(pos, vel, targets, rt, ma)

        if n_team == 1:
            # Single player: every cell within tau is uniquely reachable
            unique_cells = tti_matrix[0] <= tau_seconds
            reachable_map[pids[0]] = float(unique_cells.sum() * cell_area)
        else:
            # argmin/second-min optimization
            global_argmin = np.argmin(tti_matrix, axis=0)  # (n_cells,)
            global_min = tti_matrix.min(axis=0)  # (n_cells,) — explicit, no partition ambiguity
            # np.partition: kth=1 gives second-smallest in position [1]
            partitioned = np.partition(tti_matrix, kth=1, axis=0)
            second_min = partitioned[1, :]  # (n_cells,)

            for idx in range(n_team):
                pid = pids[idx]
                player_tti = tti_matrix[idx]  # (n_cells,)
                # min TTI of teammates excluding this player
                min_excluding = np.where(
                    global_argmin == idx,
                    second_min,
                    global_min,
                )
                unique_cells = (player_tti <= tau_seconds) & (min_excluding > tau_seconds)
                reachable_map[pid] = float(unique_cells.sum() * cell_area)

    # --- Assemble result ---
    result: dict[int | str, PlayerInfluence] = {}
    for _, row in players.iterrows():
        pid = row["player_id"]
        result[pid] = PlayerInfluence(
            off_ball_xt=off_ball_xt_map.get(pid, 0.0),
            reachable_area_m2=reachable_map.get(pid, 0.0),
        )
    return result
