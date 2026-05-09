"""Per-frame team shape envelope (TF-31).

Computes centroid, convex hull area, length, width, stretch index, and
visible outfield player count for a specified team per frame.

See spec: docs/superpowers/specs/2026-05-09-tf31-tf32-team-shape-line-breaking-design.md s1.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

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
]


def compute_team_shape(
    frames: pd.DataFrame,
    team_id: int | str,
) -> pd.DataFrame:
    """Per-(game_id, period_id, frame_id) team shape metrics for one team.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS schema).
    team_id : int | str
        Team to compute shape for.

    Returns
    -------
    pd.DataFrame
        One row per (game_id, period_id, frame_id) where the team has at
        least one visible outfield player. Columns: game_id, period_id,
        frame_id, team_id, n_outfield_players, centroid_x, centroid_y,
        convex_hull_area, team_length, team_width, stretch_index.

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
        (frames["team_id"] == team_id)
        & (~frames["is_ball"].astype(bool))
        & (~frames["is_goalkeeper"].astype(bool))
        & frames["x"].notna()
        & frames["y"].notna()
    )
    outfield = frames[mask]
    if outfield.empty:
        return pd.DataFrame(columns=_RESULT_COLS)

    rows: list[dict] = []
    groups = outfield.groupby(["game_id", "period_id", "frame_id"], dropna=False)

    for (game_id, period_id, frame_id), group in groups:
        xs = group["x"].to_numpy(dtype="float64")
        ys = group["y"].to_numpy(dtype="float64")
        n = len(xs)

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        team_length = float(np.max(xs) - np.min(xs))
        team_width = float(np.max(ys) - np.min(ys))

        # Stretch index: mean Euclidean distance from centroid
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        stretch = float(np.mean(dists))

        # Convex hull area
        if n < 3:
            hull_area = np.nan
        else:
            try:
                hull = ConvexHull(np.column_stack([xs, ys]))
                hull_area = float(hull.volume)  # 2D: volume = area
            except QhullError:
                hull_area = np.nan

        rows.append(
            {
                "game_id": game_id,
                "period_id": period_id,
                "frame_id": frame_id,
                "team_id": team_id,
                "n_outfield_players": n,
                "centroid_x": cx,
                "centroid_y": cy,
                "convex_hull_area": hull_area,
                "team_length": team_length,
                "team_width": team_width,
                "stretch_index": stretch,
            }
        )

    result = pd.DataFrame(rows, columns=_RESULT_COLS)
    result["n_outfield_players"] = result["n_outfield_players"].astype("Int64")
    return result
