"""Ward-clustering line-breaking detection (TF-32).

Identifies defensive lines via 1D Ward hierarchical clustering on opponent
x-coordinates, constructs line segments, and tests pass trajectory
intersection via cross-product straddle test.

Deviations from reference: Karakus & Arkadas (2025) use centroid +
vertical-span intersection test. This implementation uses polyline +
cross-product straddle test, which captures actual defensive geometry
(player positions form the line segments, not cluster centroids). The
straddle test is more geometrically precise and handles non-vertical
lines correctly.

Out-of-scope paper metrics (Karakus & Arkadas 2025): SBR (Successful
Ball Recovery), LBPCh1 (Line-Breaking Pass Chance 1st-half),
LBPCh2 (Line-Breaking Pass Chance 2nd-half). These are game-level
aggregates computed from the per-pass detection output and are not
implemented here.

See spec: docs/superpowers/specs/2026-05-09-tf31-tf32-team-shape-line-breaking-design.md s2.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

_PASS_CROSS_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[n] for n in ("pass", "cross") if n in spadlconfig.actiontype_id
)


@dataclass(frozen=True)
class LineBreakingParams:
    """Parameters for Ward-clustering line-breaking detection.

    Examples
    --------
    >>> from silly_kicks.tracking._line_breaking import LineBreakingParams
    >>> params = LineBreakingParams(min_opponents=3, n_clusters=3)
    """

    min_opponents: int = 3
    n_clusters: int = 3  # Design choice (defense/midfield/attack partition), not from reference paper
    min_pass_length: float = 3.0  # metres
    min_x_spread: float = 5.0  # metres
    pitch_y_min: float = 0.0  # SPADL y-coordinate of near sideline
    pitch_y_max: float = 68.0  # SPADL y-coordinate of far sideline


_RESULT_COLS = [
    "line_break__ward",
    "lines_broken__ward",
    "line_breaking_type__ward",
]


def detect_line_breaking(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: LineBreakingParams | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-action Ward-clustering line-breaking detection.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions (per-action coordinates where the acting team attacks x=105).
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS schema).
        Must be LTR-normalized (play_left_to_right applied).
    params : LineBreakingParams | None
        Algorithm parameters. Defaults to ``LineBreakingParams()``.

    Returns
    -------
    pd.DataFrame
        Aligned with ``actions.index``. Columns:
        - ``line_break__ward`` (boolean, nullable)
        - ``lines_broken__ward`` (Int64, 0-3)
        - ``line_breaking_type__ward`` (object: "between_lines", "around_line", or None)

    Examples
    --------
    Detect line-breaking passes::

        from silly_kicks.tracking._line_breaking import detect_line_breaking
        lb = detect_line_breaking(actions, frames)

    See NOTICE for full bibliographic citations.
    """
    if params is None:
        params = LineBreakingParams()

    n_actions = len(actions)
    empty = pd.DataFrame(
        {
            "line_break__ward": pd.array([pd.NA] * n_actions, dtype="boolean"),
            "lines_broken__ward": pd.array([pd.NA] * n_actions, dtype="Int64"),
            "line_breaking_type__ward": pd.array([None] * n_actions, dtype="object"),
        },
        index=actions.index,
    )

    if n_actions == 0 or len(frames) == 0:
        return empty

    from .utils import link_actions_to_frames

    # Link actions to frames
    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return empty

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    linked = linked.merge(
        actions[
            [
                "action_id",
                "team_id",
                "type_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "period_id",
                "game_id",
            ]
        ],
        on="action_id",
        how="left",
    )
    linked = linked.drop_duplicates("action_id", keep="first")

    # Pre-build grouped outfield opponent positions
    non_ball_non_gk = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))]
    frame_groups: dict = dict(
        iter(non_ball_non_gk.groupby(["game_id", "period_id", "frame_id", "team_id"], sort=False))
    )

    # Pre-build (game, period, frame) -> list of team_ids for O(1) opposing lookup
    frame_to_teams: dict[tuple, list] = {}
    for key in frame_groups:
        frame_key = key[:3]
        frame_to_teams.setdefault(frame_key, []).append(key[3])

    # Align game_id dtype between linked (from actions) and frame groupby keys.
    # Actions may carry int game_id while frames carry str (or vice versa),
    # causing the dict lookup to silently miss all frames.
    # Cast linked game_id to str for comparison, and rebuild frame dicts
    # with str keys if needed, since str is the universal safe direction.
    if len(frame_to_teams) > 0:
        sample_frame_gid = next(iter(frame_to_teams))[0]
        linked_gid_sample = linked["game_id"].iloc[0]
        if not isinstance(linked_gid_sample, type(sample_frame_gid)):
            # Rebuild dicts with str game_id keys
            frame_groups = {(str(k[0]), k[1], k[2], k[3]): v for k, v in frame_groups.items()}
            frame_to_teams = {}
            for key in frame_groups:
                frame_key = key[:3]
                frame_to_teams.setdefault(frame_key, []).append(key[3])
            linked["game_id"] = linked["game_id"].astype(str)

    # Build positional lookup
    aid_to_pos = {aid: pos for pos, aid in enumerate(actions["action_id"].values)}

    lb_arr = np.full(n_actions, np.nan)
    count_arr = np.full(n_actions, np.nan)
    type_arr: list[str | None] = [None] * n_actions

    # Direction per ACTION, from the frames' own labels -- never from team identity
    # (ADR-051 D3). This function holds both `actions` and `frames`, so it IS the edge where
    # the single orientation authority is consulted (ADR-028/041); per-frame callees below
    # receive the resolved bool, not the resolver.
    # The Series is NULLABLE boolean: pd.NA marks an action whose direction the frames do not
    # resolve. It is kept as NA rather than coerced, so the loop below can leave that action's
    # row NaN -- `.to_numpy(dtype=bool)` cannot represent the third state at all, and an
    # na_value=False would restore the exact defect this re-key removes (an unresolved
    # direction rendered indistinguishable from a resolved "does not flip").
    _flip_by_aid = dict(zip(actions["action_id"], acting_team_attacks_rtl(actions, frames), strict=False))

    for _, row in linked.iterrows():
        aid = row["action_id"]
        if aid not in aid_to_pos:
            continue
        pos = aid_to_pos[aid]

        action_team = row["team_id"]
        # NaN-actor action (e.g. the Gradient Sports null-actor duel/foul residue,
        # team_id <NA>): NaN-route to the pd.NA default -- never let an NA reach the
        # opponent-set `!=` below (`t != <NA>` -> "boolean value of NA is ambiguous").
        if pd.isna(action_team):
            continue
        action_type = row.get("type_id")
        if pd.notna(action_type) and int(action_type) not in _PASS_CROSS_TYPE_IDS:
            continue  # Non-pass/cross -> leave as pd.NA

        game_id = row["game_id"]
        period_id = row["period_id"]
        frame_id = int(row["frame_id_int"])
        start_x = float(row["start_x"])
        start_y = float(row["start_y"])
        end_x = float(row["end_x"])
        end_y = float(row["end_y"])

        # Find opposing team via O(1) lookup
        frame_key = (game_id, period_id, frame_id)
        teams_at_frame = frame_to_teams.get(frame_key, [])
        # ADR-019: dtype-safe id comparison (Int64 action team_id vs object-string
        # frame team_id -- e.g. GS-on-tracking -- where a raw `!=` is always True and
        # silently keeps the actor's OWN team as the "opponent"). action_team is
        # guaranteed non-NA here (NaN-routed above). Mirrors same_id below.
        opp_teams = [t for t in teams_at_frame if not same_id(t, action_team)]

        if not opp_teams:
            lb_arr[pos] = 0.0
            count_arr[pos] = 0
            continue

        opp_df = frame_groups[(game_id, period_id, frame_id, opp_teams[0])]
        valid_mask = opp_df["x"].notna() & opp_df["y"].notna()
        valid_opp = opp_df[valid_mask]
        opp_x = valid_opp["x"].to_numpy(dtype="float64")
        opp_y = valid_opp["y"].to_numpy(dtype="float64")

        # Convert SPADL action coords to tracking coords for intersection (opponents are
        # already tracking coords). An action-LTR point for a team attacking LEFT-TO-RIGHT in
        # the frames already IS the tracking point; one attacking RIGHT-TO-LEFT reflects.
        #
        # ⚠ NOTE THE DIRECTION OF THIS CONVERSION. `acting_team_attacks_rtl` documents its
        # bool as "frame-sampled positions must be flipped to land in the ACTION-LTR frame";
        # here the conversion runs the other way, action-LTR -> tracking. The two coincide
        # ONLY because (x, y) -> (105-x, 68-y) is an INVOLUTION -- its own inverse (the
        # 180-degree rotation recorded in `_geometry.py`). Stated because a future reader who
        # checks the helper's contract against this use will find them opposed, and the
        # natural "fix" is to invert something that is already correct.
        _flip = _flip_by_aid.get(aid)
        if pd.isna(_flip):
            # Unresolved direction -> leave this action's row NaN, the same route every other
            # uncomputable row above takes. Guessing "no flip" would mirror away-team geometry
            # into the wrong convention silently.
            continue
        if _flip:
            track_start_x = 105.0 - start_x
            track_start_y = 68.0 - start_y
            track_end_x = 105.0 - end_x
            track_end_y = 68.0 - end_y
        else:
            track_start_x = start_x
            track_start_y = start_y
            track_end_x = end_x
            track_end_y = end_y

        # The Ward-cluster + straddle geometry is single-sourced in _straddle_core (N3); the
        # pass and the opponents are both in tracking coords here, so it returns the same answer
        # the TF-51 defensive-credit signal gets by feeding it action-LTR coords instead.
        is_break, break_type, n_lines = _straddle_core(
            track_start_x, track_start_y, track_end_x, track_end_y, opp_x, opp_y, params
        )
        lb_arr[pos] = 1.0 if is_break else 0.0
        count_arr[pos] = n_lines
        type_arr[pos] = break_type

    return pd.DataFrame(
        {
            "line_break__ward": pd.array(
                [pd.NA if np.isnan(v) else bool(v) for v in lb_arr],
                dtype="boolean",
            ),
            "lines_broken__ward": pd.array(
                [pd.NA if np.isnan(v) else int(v) for v in count_arr],
                dtype="Int64",
            ),
            "line_breaking_type__ward": pd.array(type_arr, dtype="object"),
        },
        index=actions.index,
    )


def _straddle_core(
    pass_start_x: float,
    pass_start_y: float,
    pass_end_x: float,
    pass_end_y: float,
    opp_x: np.ndarray,
    opp_y: np.ndarray,
    params: LineBreakingParams,
) -> tuple[bool, str | None, int]:
    """Ward-cluster opponent x-coords into lines and straddle-test the pass trajectory.

    Coordinate-frame agnostic: the pass endpoints AND the opponent points must be in ONE
    consistent frame (tracking OR action-LTR -- the cross-product straddle test is invariant
    under a consistent rigid transform). This is the SINGLE straddle implementation (N3): both
    ``detect_line_breaking`` (tracking coords) and the TF-51 defensive-credit line-break signal
    (action-LTR) delegate here. Callers own opponent SELECTION and any coordinate flip.

    Returns ``(is_break, break_type, n_lines_broken)``; the min_pass_length / min_opponents /
    min_x_spread / <2-cluster short-circuits all return ``(False, None, 0)``.
    """
    pass_len = np.sqrt((pass_end_x - pass_start_x) ** 2 + (pass_end_y - pass_start_y) ** 2)
    if pass_len < params.min_pass_length:
        return False, None, 0
    if len(opp_x) < params.min_opponents:
        return False, None, 0
    x_spread = float(np.max(opp_x) - np.min(opp_x))
    if x_spread < params.min_x_spread:
        return False, None, 0
    n_eff_clusters = min(params.n_clusters, len(opp_x))
    if n_eff_clusters < 2:
        return False, None, 0

    # Ward clustering on 1D x-coordinates
    linkage_matrix = linkage(opp_x.reshape(-1, 1), method="ward")
    labels = fcluster(linkage_matrix, t=n_eff_clusters, criterion="maxclust")

    # Sort clusters by ascending mean x
    cluster_ids = np.unique(labels)
    cluster_means = [float(np.mean(opp_x[labels == c])) for c in cluster_ids]
    sorted_order = np.argsort(cluster_means)
    sorted_cluster_ids = cluster_ids[sorted_order]

    # Build segments per cluster and test intersection
    lines_broken = 0
    any_through = False

    for cid in sorted_cluster_ids:
        mask = labels == cid
        cx = opp_x[mask]
        cy = opp_y[mask]

        # Sort by y
        y_order = np.argsort(cy)
        cx_sorted = cx[y_order]
        cy_sorted = cy[y_order]

        # Extend to sidelines using nearest-player x
        points_x = np.concatenate([[cx_sorted[0]], cx_sorted, [cx_sorted[-1]]])
        points_y = np.concatenate([[params.pitch_y_min], cy_sorted, [params.pitch_y_max]])

        # Test each segment for intersection with pass trajectory
        cluster_broken = False
        cluster_has_through = False
        n_segments = len(points_x) - 1

        for si in range(n_segments):
            ax, ay = points_x[si], points_y[si]
            bx, by = points_x[si + 1], points_y[si + 1]

            if _segments_intersect(
                pass_start_x,
                pass_start_y,
                pass_end_x,
                pass_end_y,
                ax,
                ay,
                bx,
                by,
            ):
                cluster_broken = True
                # Extension segments are first and last
                if si != 0 and si != n_segments - 1:
                    cluster_has_through = True

        if cluster_broken:
            lines_broken += 1
            if cluster_has_through:
                any_through = True

    if lines_broken == 0:
        return False, None, 0
    # "between_lines" dominates (more tactically significant)
    return True, ("between_lines" if any_through else "around_line"), lines_broken


def _segments_intersect(
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> bool:
    """Cross-product straddle test for segment (C,D) vs segment (A,B).

    Returns True if the two segments intersect or touch (endpoint on
    the other segment). Excludes fully collinear cases where the pass
    trajectory is parallel to and on the defensive segment.
    """
    d1 = _cross(bx - ax, by - ay, cx - ax, cy - ay)
    d2 = _cross(bx - ax, by - ay, dx - ax, dy - ay)
    d3 = _cross(dx - cx, dy - cy, ax - cx, ay - cy)
    d4 = _cross(dx - cx, dy - cy, bx - cx, by - cy)
    # Use <= to include touching (pass through a player position).
    # Guard against fully collinear false positives (d1==d2==0).
    return (d1 * d2 <= 0) and (d3 * d4 <= 0) and not (d1 == 0 and d2 == 0)


def _cross(ux: float, uy: float, vx: float, vy: float) -> float:
    """2D cross product of vectors (ux, uy) and (vx, vy)."""
    return ux * vy - uy * vx
