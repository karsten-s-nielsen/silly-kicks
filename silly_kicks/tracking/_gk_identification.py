"""Derived goalkeeper identification via positional behavior (B+ filtered algorithm).

Original empirical heuristic for cross-provider GK identification; thresholds and
stage shape are tuned against the 2026-05-04 cross-provider sweep documented in
ADR-007. No academic prior art directly maps to this algorithm — closest-to-goal
and dwell-time-in-region are general spatial-positional reasoning patterns.

See ADR-007 for full design rationale and threshold justification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows

# Module-level constants (locked thresholds from spec §4.2)
_GK_N_FRAMES_FRAC = 0.30  # candidate filter: >= 30% of team's max player-frame count
_GK_PA_DWELL_MIN = 0.40  # strict GK criterion: in PA for >= 40% of on-pitch frames
_GK_DIST_MAX_M = 20.0  # strict GK criterion: mean dist to nearest goal-line < 20m

# SPADL coordinate bounds with 15m off-pitch tolerance (players legitimately
# run off-pitch; the check detects CENTERED coords, not slight off-pitch)
_SPADL_X_MIN, _SPADL_X_MAX = -15.0, 120.0
_SPADL_Y_MIN, _SPADL_Y_MAX = -15.0, 83.0


def derive_goalkeepers(
    frames: pd.DataFrame,
    teams: pd.MultiIndex | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[str]]]:
    """Identify goalkeeper(s) per (game_id, team_id) from positional behaviour.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS-shaped output. Required columns: game_id,
        team_id, player_id, x, y, is_ball, is_goalkeeper. Coordinates must
        be in 0-105 / 0-68 SPADL convention (post pitch-dim normalisation).
    teams : pd.MultiIndex | None, default None
        (game_id, team_id) pairs to derive. None means: derive for all
        teams in `frames`.

    Returns
    -------
    frames_out : pd.DataFrame
        Copy of input with is_goalkeeper overwritten on rows belonging to
        identified GK player(s) for affected teams; other rows unchanged.
    derived_picks : dict[(game_id, team_id), list[player_id]]
        Audit trail: which player_id(s) were flagged per (game, team).

    Raises
    ------
    ValueError
        If required columns are missing, if NaN game_id/team_id is
        encountered on player rows, or if coordinate range falls outside
        SPADL bounds.

    Examples
    --------
    Derive GKs for all teams in a tracking DataFrame::

        from silly_kicks.tracking._gk_identification import derive_goalkeepers
        frames_out, picks = derive_goalkeepers(frames)
        # picks = {("game1", "teamA"): ["gk_player_id"], ...}
    """
    # Input validation: required columns
    required = {"game_id", "team_id", "player_id", "x", "y", "is_ball", "is_goalkeeper"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"derive_goalkeepers: frames missing columns {sorted(missing)}")

    # Short-circuit for empty input
    if len(frames) == 0:
        return frames.copy(), {}

    # Filter to player rows only (exclude ball)
    player_rows = frames[~frames["is_ball"]].copy()

    # Input validation: NaN game_id/team_id on player rows
    if player_rows["game_id"].isna().any() or player_rows["team_id"].isna().any():
        raise ValueError("derive_goalkeepers: NaN game_id/team_id encountered (pipeline integrity issue)")

    # Input validation: coordinate range (SPADL bounds with slack)
    valid_coords = player_rows[["x", "y"]].dropna()
    if len(valid_coords) > 0:
        x_min, x_max = valid_coords["x"].min(), valid_coords["x"].max()
        y_min, y_max = valid_coords["y"].min(), valid_coords["y"].max()
        if x_min < _SPADL_X_MIN or x_max > _SPADL_X_MAX or y_min < _SPADL_Y_MIN or y_max > _SPADL_Y_MAX:
            raise ValueError(
                f"derive_goalkeepers: coords must be SPADL 0-105/0-68; "
                f"got x in [{x_min:.1f},{x_max:.1f}] y in [{y_min:.1f},{y_max:.1f}] "
                "(caller must run to_pitch_dimensions first)"
            )

    # Core algorithm implementation
    frames_out = frames.copy()
    derived_picks: dict[tuple[str, str], list[str]] = {}

    # Determine teams to process
    if teams is None:
        teams_list = player_rows[["game_id", "team_id"]].drop_duplicates().values.tolist()
    else:
        teams_list = list(teams)

    # ADR-068: build the per-(game, team) row lookup ONCE instead of a full boolean scan of the
    # entire player_rows table on every team (was O(n_teams * n_player_rows) on a batch).
    team_groups = group_rows(player_rows, ("game_id", "team_id"))

    for game_id, team_id in teams_list:
        team_rows = team_groups.get(game_id, team_id).copy()

        if len(team_rows) == 0:
            continue

        # Pre-compute per-row features (safer than lambda closure in groupby)
        clipped_x = team_rows["x"].clip(0, 105)
        team_rows["_dist_to_goal"] = np.minimum(clipped_x, 105 - clipped_x)
        team_rows["_in_pa"] = ((team_rows["x"] < 16.5) | (team_rows["x"] > 88.5)) & (
            team_rows["y"].between(13.84, 54.16)
        )

        # Per-player feature aggregation
        agg = (
            team_rows.groupby("player_id")
            .agg(
                n_frames=("x", "count"),
                dist_mean=("_dist_to_goal", "mean"),
                pa_dwell=("_in_pa", "mean"),
            )
            .reset_index()
        )

        # Stage 1: candidate filter (n_frames >= 30% of team max)
        max_frames = agg["n_frames"].max()
        threshold_frames = _GK_N_FRAMES_FRAC * max_frames
        candidates = agg[agg["n_frames"] >= threshold_frames].copy()

        if len(candidates) == 0:
            # Should be impossible (max player always passes), but defensive
            raise AssertionError(
                f"derive_goalkeepers: zero candidates after n_frames filter for ({game_id}, {team_id}); this is a bug"
            )

        # Stage 2a: strict GK detection (multi-GK output natural)
        strict_mask = (candidates["pa_dwell"] >= _GK_PA_DWELL_MIN) & (candidates["dist_mean"] < _GK_DIST_MAX_M)
        strict_gks = candidates[strict_mask]

        if len(strict_gks) > 0:
            gk_player_ids = strict_gks["player_id"].tolist()
        else:
            # Stage 2b: sweeper-keeper fallback
            # Sort by player_id for deterministic ranking
            candidates = candidates.sort_values("player_id").reset_index(drop=True)
            # Rank-sum: lower dist is better (asc), higher pa_dwell is better (desc)
            candidates["rank_dist"] = candidates["dist_mean"].rank(method="first", ascending=True)
            candidates["rank_pa"] = candidates["pa_dwell"].rank(method="first", ascending=False)
            candidates["score"] = candidates["rank_dist"] + candidates["rank_pa"]
            # Pick lowest score (ties broken by first occurrence = lowest player_id)
            best_idx = candidates["score"].idxmin()
            gk_player_ids = [candidates.loc[best_idx, "player_id"]]

        # Store picks
        derived_picks[(game_id, team_id)] = gk_player_ids

    # ADR-068: set is_goalkeeper in a SINGLE positional pass over frames_out, instead of a full
    # 3-condition boolean scan of the whole table per GK (was O(n_teams * n_frames)). Grouping over
    # a reset-index copy yields 0..n-1 POSITIONS, so the write is index-safe (no reliance on the
    # caller's index being unique). Only-True writes over disjoint (game, team, player) rows -> the
    # post-loop order is byte-identical to the incremental per-team writes it replaces.
    gk_pos = group_rows(frames_out.reset_index(drop=True), ("game_id", "team_id", "player_id"))
    gk_positions: list[int] = []
    for (game_id, team_id), gk_player_ids in derived_picks.items():
        for pid in gk_player_ids:
            gk_positions.extend(int(p) for p in gk_pos.get(game_id, team_id, pid).index)
    if gk_positions:
        # Position-safe write (a boolean mask ignores index labels, so a non-unique caller index
        # cannot mis-target) -- byte-identical to the per-team incremental writes it replaces.
        gk_mask = np.zeros(len(frames_out), dtype=bool)
        gk_mask[gk_positions] = True
        frames_out.loc[gk_mask, "is_goalkeeper"] = True

    return frames_out, derived_picks
