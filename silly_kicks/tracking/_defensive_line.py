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

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._collective import back_line_batch, pack_groups
from silly_kicks.tracking._gk_resolve import GoalEndUnresolvedError, GoalMap


def select_back_line_players(
    frames: pd.DataFrame,
    team_id: int | str,
    defends_x0: bool,
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
    defends_x0 : bool
        Whether ``team_id`` defends the goal at x=0 on these frames — i.e. the
        DIRECTION, supplied by the caller rather than inferred here.

        This parameter used to be ``home_team_id``, from which the direction was
        derived as ``same_id(team_id, home_team_id)``. That is identity-keyed
        direction inference (ADR-051 D3): it is correct only while the frames are
        home-attacks-right, and it silently inverts otherwise. Every caller in the
        package now resolves the direction from the goal map instead; the D3 pin in
        ``tests/tracking/test_mirror_registry.py`` asserts that no module in this
        family computes ``same_id(..., home_team_id)`` at all.
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
    Select the back-line players for a team on a frame::

        from silly_kicks.tracking import resolve_defended_goals
        from silly_kicks.tracking._defensive_line import select_back_line_players

        goal_map = resolve_defended_goals(frames)   # ONCE per match, full frames
        back_line = select_back_line_players(
            frame,
            team_id=1,
            # `get` returns float | None; `None == 0.0` is False, so a bare `== 0.0`
            # fails OPEN. Resolve first, then compare.
            defends_x0=goal_map.get(game_id, period_id, 1, allow_guess=True) == 0.0,
        )
        back_line[["player_id", "x", "y"]].head()

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
    goal_map: GoalMap,
    n: int | Literal["adaptive"] = 4,
    adaptive_max_n: int = 5,
) -> pd.DataFrame:
    """Per-(game_id, period_id, frame_id, team_id): 6 back-line geometry columns.

    Computes for BOTH teams, so it needs BOTH ends -- which is why it takes the map
    rather than a single direction bool (the ADR-051 D3 rule: one team -> bool, both
    teams -> map).

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
        Must be LTR-normalized (play_left_to_right applied).
    goal_map : GoalMap
        Per-(game, period, team) defended-goal ends, built ONCE per match from the FULL
        frames (:func:`resolve_defended_goals`). REQUIRED, no default: a default would
        re-admit per-frame direction inference at exactly the call sites that forget it.
        An unresolved end RAISES :class:`GoalEndUnresolvedError` rather than guessing.
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
        goal_map = resolve_defended_goals(frames)   # ONCE per match, full frames
        dl = compute_defensive_line(frames, goal_map=goal_map, n=4)

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
    # Filter to outfield players with valid coordinates (x-valid only; NaN y propagates as before).
    outfield = frames[(~frames["is_ball"]) & (~frames["is_goalkeeper"]) & frames["x"].notna()].copy()

    # Delegate the six back-line columns to the vectorised kernel (ADR D13, byte-identical). Group order
    # (sort=True) matches the legacy iteration order. observed=True is deterministic across pandas majors.
    gb = outfield.groupby(["game_id", "period_id", "frame_id", "team_id"], dropna=False, sort=True, observed=True)
    codes = gb.ngroup().to_numpy()
    key_tuples = gb.size().index.tolist()
    pos, counts, _first = pack_groups(
        codes,
        outfield["x"].to_numpy(dtype="float64"),
        outfield["y"].to_numpy(dtype="float64"),
        len(key_tuples),
    )
    # Resolve the defended end ONCE per (game, period, team) with >= 3 players -- never per frame-team, never
    # from team IDENTITY (`same_id(team_id, home_team_id)` silently inverts off home-attacks-right; ADR-051 D3).
    # An unresolved end raises for the FIRST such group in key order, matching the legacy loop.
    ends: dict[tuple, float | None] = {}
    for i, k in enumerate(key_tuples):
        if counts[i] < 3:
            continue
        gpt = (k[0], k[1], k[3])
        if gpt not in ends:
            ends[gpt] = goal_map.get(k[0], k[1], k[3], allow_guess=True)
    for i, k in enumerate(key_tuples):
        # Explicit `is None`: `== 0.0` alone would fail OPEN, silently choosing 'defends x=0'.
        if counts[i] >= 3 and ends[(k[0], k[1], k[3])] is None:
            raise GoalEndUnresolvedError(
                f"defensive line: goal_map does not resolve the end defended by {k[3]!r} "
                f"in (game={k[0]!r}, period={k[1]!r})."
            )
    defends_x0 = np.array(
        [counts[i] >= 3 and ends[(k[0], k[1], k[3])] == 0.0 for i, k in enumerate(key_tuples)],
        dtype=bool,
    )
    bl = back_line_batch(pos, counts, defends_x0, n=n, adaptive_max_n=adaptive_max_n)

    result = pd.DataFrame(key_tuples, columns=["game_id", "period_id", "frame_id", "team_id"])
    for col in ("defensive_line_x", "back_line_high_x", "compactness_x", "lateral_width", "max_lateral_gap"):
        result[col] = bl[col]
    back_n = pd.array(bl["back_n_count"], dtype="Int64")
    back_n[~bl["valid"]] = pd.NA  # count < 3 -> NA, matching the legacy per-frame NaN row
    result["back_n_count"] = back_n
    return result[result_cols]


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
