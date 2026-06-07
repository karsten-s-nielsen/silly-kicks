"""Per-frame defensive-line geometry (TF-14).

Computes back-line geometry for both teams per frame. Foundational primitive
consumed by action-coupled VAEP features, GKDV stack, and line-break detection.

See spec: docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md s3.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ._id_compat import ids_match, same_id


def select_back_line_players(
    frames: pd.DataFrame,
    team_id: int | str,
    home_team_id: int | str,
    *,
    n: int | Literal["adaptive"] = 4,
    adaptive_max_n: int = 5,
) -> pd.DataFrame:
    """Select the N outfield players closest to their own goal.

    Returns a DataFrame of player rows (preserving x, y, vx, vy, player_id,
    etc.) sorted by proximity to own goal. Operates on a single frame.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frame (single frame expected, but multi-frame
        is tolerated — uses first frame group).
    team_id : int | str
        Team to select back-line players for.
    home_team_id : int | str
        Home team identifier for goal-end resolution.
    n : int | Literal["adaptive"], default 4
        Target back-line player count. Clamped to available outfield.
    adaptive_max_n : int, default 5
        Upper bound for adaptive N.

    Returns
    -------
    pd.DataFrame
        Player rows with all original columns preserved, sorted by
        proximity to own goal. Length = min(n_effective, available_outfield).

    Examples
    --------
    >>> from silly_kicks.tracking._defensive_line import select_back_line_players
    >>> back_line = select_back_line_players(frame, team_id=1, home_team_id=1)
    >>> back_line[["player_id", "x", "y"]].head()

    See NOTICE for full bibliographic citations.
    """
    outfield = frames[
        (~frames["is_ball"].astype(bool))
        & (~frames["is_goalkeeper"].astype(bool))
        & ids_match(frames["team_id"], team_id)
        & frames["x"].notna()
    ]

    if len(outfield) < 3:
        return outfield

    defends_x0 = same_id(team_id, home_team_id)
    xs = outfield["x"].to_numpy(dtype="float64")

    if defends_x0:
        order = np.argsort(xs)
    else:
        order = np.argsort(-xs)

    xs_sorted = xs[order]
    p = len(outfield)
    n_effective = _select_n(xs_sorted, n, adaptive_max_n, p)

    return outfield.iloc[order[:n_effective]]


def compute_defensive_line(
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
    adaptive_max_n: int = 5,
) -> pd.DataFrame:
    """Per-(game_id, period_id, frame_id, team_id): 6 back-line geometry columns.

    Computes for BOTH teams. home_team_id determines goal assignment
    (must match the value used in play_left_to_right).

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
        Must be LTR-normalized (play_left_to_right applied).
    home_team_id : int | str
        Home team identifier. After LTR normalization:
        - home_team_id defends goal at x=0 (back-line = lowest-x outfield)
        - other team defends goal at x=105 (back-line = highest-x outfield)
    n : int | Literal["adaptive"], default 4
        Target back-line player count (3, 4, or 5), clamped to available
        outfield players (minimum 3). Or "adaptive" for x-gap clustering.
    adaptive_max_n : int, default 5
        Upper bound for adaptive N. Must be in {3, 4, 5}.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, period_id, frame_id, team_id, defensive_line_x,
        back_line_high_x, compactness_x, lateral_width, max_lateral_gap,
        back_n_count.

    Raises
    ------
    ValueError
        If n is an int outside {3, 4, 5}, adaptive_max_n outside {3, 4, 5},
        frames missing required columns, or non-LTR direction values found.

    Examples
    --------
    Compute defensive-line geometry for both teams::

        from silly_kicks.tracking.features import compute_defensive_line
        dl = compute_defensive_line(frames, home_team_id=1, n=4)

    See NOTICE for full bibliographic citations.
    """
    # --- Validation ---
    if isinstance(n, int) and n not in (3, 4, 5):
        raise ValueError(f"n must be 3, 4, or 5 (got {n})")
    if adaptive_max_n not in (3, 4, 5):
        raise ValueError(f"adaptive_max_n must be in {{3, 4, 5}} (got {adaptive_max_n})")

    required_cols = {"game_id", "period_id", "frame_id", "team_id", "player_id", "is_ball", "is_goalkeeper", "x", "y"}
    missing = required_cols - set(frames.columns)
    if missing:
        raise ValueError(f"compute_defensive_line: frames missing columns {sorted(missing)}")

    # LTR guard: period-normalized frames have home="ltr", away="rtl"
    if "team_attacking_direction" in frames.columns:
        directions = set(frames["team_attacking_direction"].dropna().unique())
        valid = {"ltr", "rtl"}
        unexpected = directions - valid
        if unexpected:
            raise ValueError(
                "compute_defensive_line: frames have unexpected "
                f"team_attacking_direction values: {sorted(unexpected)}. "
                "Expected 'ltr'/'rtl' only."
            )
        if directions and "ltr" not in directions:
            raise ValueError(
                "compute_defensive_line: frames must be period-normalized "
                "(play_left_to_right). Found only 'rtl' direction values — "
                "no home-team rows with 'ltr'."
            )

    # --- Short-circuit ---
    result_cols = [
        "game_id",
        "period_id",
        "frame_id",
        "team_id",
        "defensive_line_x",
        "back_line_high_x",
        "compactness_x",
        "lateral_width",
        "max_lateral_gap",
        "back_n_count",
    ]
    if len(frames) == 0:
        return pd.DataFrame(columns=result_cols)

    # --- Core computation ---
    # Filter to outfield players with valid coordinates
    outfield = frames[(~frames["is_ball"]) & (~frames["is_goalkeeper"]) & frames["x"].notna()].copy()

    # Group by (game_id, period_id, frame_id, team_id)
    rows: list[dict] = []
    groups = outfield.groupby(["game_id", "period_id", "frame_id", "team_id"], dropna=False)

    for (game_id, period_id, frame_id, team_id), group in groups:
        p = len(group)
        if p < 3:
            rows.append(
                {
                    "game_id": game_id,
                    "period_id": period_id,
                    "frame_id": frame_id,
                    "team_id": team_id,
                    "defensive_line_x": np.nan,
                    "back_line_high_x": np.nan,
                    "compactness_x": np.nan,
                    "lateral_width": np.nan,
                    "max_lateral_gap": np.nan,
                    "back_n_count": pd.NA,
                }
            )
            continue

        # Sort by proximity to own goal
        defends_x0 = same_id(team_id, home_team_id)
        xs = group["x"].to_numpy(dtype="float64")
        ys = group["y"].to_numpy(dtype="float64")

        if defends_x0:
            order = np.argsort(xs)  # ascending: closest to x=0 first
        else:
            order = np.argsort(-xs)  # descending: closest to x=105 first

        xs_sorted = xs[order]
        ys_sorted = ys[order]

        # Determine N
        n_effective = _select_n(xs_sorted, n, adaptive_max_n, p)

        # Select back-line players
        sel_x = xs_sorted[:n_effective]
        sel_y = ys_sorted[:n_effective]

        # Compute 6 columns
        defensive_line_x = float(np.mean(sel_x))
        compactness_x = float(np.max(sel_x) - np.min(sel_x))

        if defends_x0:
            back_line_high_x = float(np.max(sel_x))  # furthest from x=0
        else:
            back_line_high_x = float(np.min(sel_x))  # furthest from x=105

        lateral_width = float(np.max(sel_y) - np.min(sel_y))

        # max_lateral_gap: sort by y, compute adjacent gaps
        y_sorted = np.sort(sel_y)
        if len(y_sorted) >= 2:
            y_gaps = np.diff(y_sorted)
            max_lateral_gap = float(np.max(y_gaps))
        else:
            max_lateral_gap = 0.0

        rows.append(
            {
                "game_id": game_id,
                "period_id": period_id,
                "frame_id": frame_id,
                "team_id": team_id,
                "defensive_line_x": defensive_line_x,
                "back_line_high_x": back_line_high_x,
                "compactness_x": compactness_x,
                "lateral_width": lateral_width,
                "max_lateral_gap": max_lateral_gap,
                "back_n_count": n_effective,
            }
        )

    result = pd.DataFrame(rows, columns=result_cols)
    result["back_n_count"] = result["back_n_count"].astype("Int64")
    return result


def _select_n(
    xs_sorted: np.ndarray,
    n: int | Literal["adaptive"],
    adaptive_max_n: int,
    p: int,
) -> int:
    """Determine how many players form the back line.

    Parameters
    ----------
    xs_sorted : sorted x-positions (closest to own goal first)
    n : target N or "adaptive"
    adaptive_max_n : upper bound for adaptive
    p : total available outfield players

    Returns
    -------
    int : effective N (3..5, clamped to available)
    """
    if isinstance(n, int):
        return min(n, p)

    # --- Adaptive algorithm ---
    if p == 3:
        return 3
    if p == 4:
        # Single cut-point; no relative comparison possible -> default N=4
        return 4

    # Examine cut-points: gaps between positions [2]->[3], [3]->[4], [4]->[5]
    gaps = np.diff(xs_sorted)  # gaps[i] = xs_sorted[i+1] - xs_sorted[i]

    # Available cut indices (0-indexed into gaps array):
    # cut at [2]->[3] means gaps[2]; corresponds to N=3
    # cut at [3]->[4] means gaps[3]; corresponds to N=4
    # cut at [4]->[5] means gaps[4]; corresponds to N=5
    cut_indices = []
    cut_ns = []
    for candidate_n in (3, 4, 5):
        gap_idx = candidate_n - 1  # gaps[2] = gap between sorted[2] and sorted[3] -> N=3
        if gap_idx < len(gaps) and candidate_n <= adaptive_max_n:
            cut_indices.append(gap_idx)
            cut_ns.append(candidate_n)

    if not cut_indices:
        return min(4, p)

    cut_gaps = [abs(float(gaps[i])) for i in cut_indices]

    # Degenerate: all gaps are 0
    if max(cut_gaps) == 0.0:
        return min(4, p)

    # Find dominant gap
    sorted_gaps = sorted(cut_gaps, reverse=True)
    max_gap = sorted_gaps[0]
    second_gap = sorted_gaps[1] if len(sorted_gaps) > 1 else 0.0

    if second_gap == 0.0 or max_gap >= 1.5 * second_gap:
        # Dominant gap found
        best_idx = cut_gaps.index(max_gap)
        return cut_ns[best_idx]

    # No dominant gap -> default to 4
    return min(4, p)
