"""Per-frame team shape envelope (TF-31, TF-44).

Computes centroid, convex hull area, length, width, stretch index,
defensive line height, inter-line gaps, and visible outfield player count
for a specified team per frame.

See spec: docs/superpowers/specs/2026-05-09-tf31-tf32-team-shape-line-breaking-design.md s1.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._collective import collective_from_positions, pack_groups

_RESULT_COLS = [
    "game_id",
    "period_id",
    "frame_id",
    "team_id",
    "n_outfield_players",
    "centroid_x",
    "centroid_y",
    "convex_hull_area",
    "team_length",
    "team_width",
    "stretch_index",
    "defensive_line_height",
    "inter_line_gap_1",
    "inter_line_gap_2",
]


def compute_team_shape(
    frames: pd.DataFrame,
    team_id: int | str,
    *,
    n_defensive_lines: int = 3,
) -> pd.DataFrame:
    """Per-(game_id, period_id, frame_id) team shape metrics for one team.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS schema).
    team_id : int | str
        Team to compute shape for.
    n_defensive_lines : int
        Number of defensive lines to identify via Ward clustering
        (default 3). Used for defensive_line_height and inter-line gaps.

    Returns
    -------
    pd.DataFrame
        One row per (game_id, period_id, frame_id) where the team has at
        least one visible outfield player. Frames with zero visible
        outfield players are omitted from output (consumers should LEFT
        JOIN and fill NaN). Columns: game_id, period_id, frame_id,
        team_id, n_outfield_players, centroid_x, centroid_y,
        convex_hull_area, team_length, team_width, stretch_index,
        defensive_line_height, inter_line_gap_1, inter_line_gap_2.

    Examples
    --------
    Compute team shape for a single team::

        from silly_kicks.tracking._team_shape import compute_team_shape
        shape = compute_team_shape(frames, team_id=1)

    See NOTICE for full bibliographic citations.
    """
    if len(frames) == 0:
        return pd.DataFrame(columns=_RESULT_COLS)

    # Filter to outfield players with valid coordinates
    mask = (
        ids_match(frames["team_id"], team_id)
        & (~frames["is_ball"].astype(bool))
        & (~frames["is_goalkeeper"].astype(bool))
        & frames["x"].notna()
        & frames["y"].notna()
    )
    outfield = frames[mask]
    if outfield.empty:
        return pd.DataFrame(columns=_RESULT_COLS)

    # Delegate centroid/length/width/stretch/hull to the single vectorised kernel (ADR D12); keep the Ward
    # line clustering per group (no batch form). Group order (sort=True) matches the legacy iteration order,
    # so the output is row-for-row byte-identical except convex_hull_area (<= 1e-9 relative; exactly 0.0 iff
    # exactly collinear). observed=True is deterministic across pandas majors (F1b-safe).
    gb = outfield.groupby(["game_id", "period_id", "frame_id"], dropna=False, sort=True, observed=True)
    codes = gb.ngroup().to_numpy()
    key_tuples = gb.size().index.tolist()
    pos, counts, first_row = pack_groups(
        codes,
        outfield["x"].to_numpy(dtype="float64"),
        outfield["y"].to_numpy(dtype="float64"),
        len(key_tuples),
    )
    cv = collective_from_positions(pos, counts)
    directions = (
        outfield["team_attacking_direction"].to_numpy()[first_row]
        if "team_attacking_direction" in outfield.columns
        else np.full(len(key_tuples), None)
    )
    def_line, gap_1, gap_2 = _ward_lines(pos, counts, directions, n_defensive_lines)

    result = pd.DataFrame(key_tuples, columns=["game_id", "period_id", "frame_id"])
    result["team_id"] = team_id
    result["n_outfield_players"] = counts
    result["centroid_x"] = cv["centroid_x"]
    result["centroid_y"] = cv["centroid_y"]
    result["convex_hull_area"] = cv["convex_hull_area"]
    result["team_length"] = cv["team_length"]
    result["team_width"] = cv["team_width"]
    result["stretch_index"] = cv["stretch_index"]
    result["defensive_line_height"] = def_line
    result["inter_line_gap_1"] = gap_1
    result["inter_line_gap_2"] = gap_2
    result = result[_RESULT_COLS]
    result["n_outfield_players"] = result["n_outfield_players"].astype("Int64")
    return result


def _ward_lines(
    pos: np.ndarray, counts: np.ndarray, directions: np.ndarray, n_defensive_lines: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group Ward line clustering (TF-44), verbatim legacy semantics. Reads the packed row (input row
    order), never rescans. Returns ``(defensive_line_height, inter_line_gap_1, inter_line_gap_2)`` arrays.

    The deepest line is the cluster NEAREST the defended goal (ADR-028): "rtl" defends x=105 (deepest =
    highest-x), else deepest = lowest-x. Defaults to "ltr" when direction is absent.
    """
    g = counts.shape[0]
    def_line = np.full(g, np.nan)
    gap_1 = np.full(g, np.nan)
    gap_2 = np.full(g, np.nan)
    for i in range(g):
        n = int(counts[i])
        xs = pos[i, :n, 0]
        defends_high_x = directions[i] == "rtl"
        n_eff = min(n_defensive_lines, n)
        if n < 2:
            def_line[i] = float(xs.max() if defends_high_x else xs.min())
            continue
        z = linkage(xs.reshape(-1, 1), method="ward")
        labels = fcluster(z, t=n_eff, criterion="maxclust")
        centroids = np.sort([float(np.mean(xs[labels == c])) for c in range(1, n_eff + 1) if np.any(labels == c)])
        if defends_high_x:
            centroids = centroids[::-1]
        n_actual = len(centroids)
        def_line[i] = float(centroids[0])
        if n_actual >= 2:
            gap_1[i] = float(abs(centroids[1] - centroids[0]))
        if n_actual >= 3:
            gap_2[i] = float(abs(centroids[2] - centroids[1]))
    return def_line, gap_1, gap_2
