"""Layer-1 rest-defense structure metrics (TF-60, ADR-080, spec §7.1).

Each metric is a pure function of one sample's frame slice + a :class:`SampleContext` (the scalars the
orchestrator resolves once per match and threads in: the acting team A, the opponent B, the ball's
frame-x, the own/attacked goal ends, and A's rearguard-line / shape values from
``compute_defensive_line`` / ``compute_team_shape``). Orientation is expressed as a distance from A's
OWN goal (``goal_relative_x = abs(x - own_goal_x)``), so every metric is identical for a scene and its
point-reflection -- direction never comes from team identity (ADR-055 / ADR-051 D3). ids compare via
``id_compat`` (ADR-019).

Column-source note (owner-ratified Option B, 2026-08-30; spec §7.1 corrected to match): both
``rd_compactness_x`` (rearguard x-range) and ``rd_width`` (rearguard lateral/y width) come from
``compute_defensive_line`` -- the back line, GK-excluded -- so they are genuinely REARGUARD-subset.
``rd_depth`` is the WHOLE-TEAM front-to-back ``team_length`` from ``compute_team_shape``: a back-line
depth would merely duplicate ``rd_compactness_x`` (a flat line has ~no independent depth), and the
team's vertical stretch is the informative counter-vulnerability signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

from ._columns import (
    RD_COMPACTNESS_X,
    RD_DEPTH,
    RD_GK_LINE_HEIGHT,
    RD_GK_TO_LINE_DISTANCE,
    RD_LINE_HEIGHT,
    RD_LINE_HEIGHT_RELATIVE,
    RD_NUM_SUPERIORITY,
    RD_NUM_SUPERIORITY_GK,
    RD_SHAPE_STAGGER,
    RD_WIDTH,
    RD_ZONE_OCCUPANCY,
)
from ._counting import bool_flag, count_goalside
from ._geometry import danger_zone_bounds


@dataclass(frozen=True)
class SampleContext:
    """Scalars resolved once per sample by the orchestrator (never re-derived per metric)."""

    team_id: object  # A -- the in-possession team
    opponent_id: object  # B -- the future counter-attacker
    ball_x: float  # ball x in the linked frame (frame-LTR)
    own_goal_x: float  # G_A -- the end A defends (0.0 or 105.0)
    attacked_goal_x: float  # G_B -- the end A attacks
    defensive_line_x: float  # A's rearguard line (compute_defensive_line, single source)
    compactness_x: float  # A's rearguard x-range (compute_defensive_line.compactness_x)
    lateral_width: float  # A's rearguard lateral/y width (compute_defensive_line.lateral_width)
    team_length: float  # A's whole-team front-to-back length (compute_team_shape.team_length)


def _is_missing(v) -> bool:
    """NA-safe missing check for a scalar id (int / str / pd.NA); unannotated arg mirrors _compute._f
    so ``pd.isna`` accepts an ``object``-typed ``SampleContext`` field."""
    return bool(pd.isna(v))


def _goal_relative_x(x: float, own_goal_x: float) -> float:
    """Distance from A's OWN goal along the into-pitch axis (0 at the own goal, 105 at the far goal)."""
    return abs(x - own_goal_x)


def _gk_x(frame_rows: pd.DataFrame, team_id) -> float:
    """The x of ``team_id``'s keeper in this frame (mean if several rows); NaN if none observed."""
    if "is_goalkeeper" not in frame_rows.columns:
        return math.nan
    mask = ids_match(frame_rows["team_id"], team_id).to_numpy() & bool_flag(frame_rows["is_goalkeeper"])
    xs = frame_rows.loc[mask, "x"].to_numpy(dtype=float)
    xs = xs[np.isfinite(xs)]
    return float(xs.mean()) if xs.size else math.nan


def rd_num_superiority(frame_rows: pd.DataFrame, ctx: SampleContext, *, include_a_gk: bool = False):
    """(# A behind the ball) - (# B behind the ball), both counted toward A's defended goal G_A.

    A missing/unresolvable opponent (a non-two-team frame set, where ``opponent_id`` is NA) yields
    ``pd.NA``, NOT a silent A-count: with no opponent the B-count would be a fabricated 0, reporting
    A's whole rearguard as "superiority" (IMPL-04). The count/gk metrics that do not need the
    opponent stay computed.
    """
    if _is_missing(ctx.opponent_id):
        return pd.NA
    a = count_goalside(
        frame_rows, team_id=ctx.team_id, ball_x=ctx.ball_x, goal_x=ctx.own_goal_x, include_gk=include_a_gk
    )
    b = count_goalside(frame_rows, team_id=ctx.opponent_id, ball_x=ctx.ball_x, goal_x=ctx.own_goal_x, include_gk=False)
    return a - b


def rd_zone_occupancy(frame_rows: pd.DataFrame, ctx: SampleContext, *, params):
    """Headcount of A's players (keeper included) inside the danger zone Z; ``pd.NA`` if unresolved."""
    if not math.isfinite(ctx.defensive_line_x) or not math.isfinite(ctx.own_goal_x):
        return pd.NA
    lo, hi = danger_zone_bounds(ctx.defensive_line_x, ctx.own_goal_x, zone_depth_m=params.zone_depth_m)
    return count_goalside(frame_rows, team_id=ctx.team_id, ball_x=hi, goal_x=lo, include_gk=True)


def rd_line_height(ctx: SampleContext) -> float:
    """A's rearguard line distance from its OWN goal (Dash 2025; FIFA EFI)."""
    return _goal_relative_x(ctx.defensive_line_x, ctx.own_goal_x)


def rd_line_height_relative(ctx: SampleContext) -> float:
    """Rearguard line height MINUS ball height (both from A's own goal); negative = line behind ball."""
    return _goal_relative_x(ctx.defensive_line_x, ctx.own_goal_x) - _goal_relative_x(ctx.ball_x, ctx.own_goal_x)


def rd_gk_line_height(frame_rows: pd.DataFrame, ctx: SampleContext) -> float:
    """A's keeper distance from its OWN goal (FIFA EFI; StatsBomb/Wyscout sweeper distance)."""
    return _goal_relative_x(_gk_x(frame_rows, ctx.team_id), ctx.own_goal_x)


def rd_gk_to_line_distance(frame_rows: pd.DataFrame, ctx: SampleContext) -> float:
    """A's keeper height MINUS rearguard-line height (the FIFA coupled-unit gap); usually negative."""
    return _goal_relative_x(_gk_x(frame_rows, ctx.team_id), ctx.own_goal_x) - _goal_relative_x(
        ctx.defensive_line_x, ctx.own_goal_x
    )


def _stagger_label(gr_positions: np.ndarray):
    """Largest-gap 2-line split of a behind-ball unit into a deeper and a shallower group -> "n-m".

    Sorts the members' distances-from-own-goal, splits at the single widest gap, and labels
    ``"{n_deeper}-{n_shallower}"`` (deeper = nearer the own goal). A unit smaller than 2 -> ``pd.NA``.
    A 5-player unit typically yields "2-3"/"3-2"; other sizes yield the generic "n-m" (spec §20.7).
    """
    xs = np.sort(gr_positions[np.isfinite(gr_positions)])
    if xs.size < 2:
        return pd.NA
    n_deeper = int(np.argmax(np.diff(xs))) + 1
    return f"{n_deeper}-{xs.size - n_deeper}"


def rd_shape_stagger(frame_rows: pd.DataFrame, ctx: SampleContext):
    """Stagger label of A's behind-the-ball OUTFIELD unit (players goal-side of the ball toward G_A)."""
    a = frame_rows[ids_match(frame_rows["team_id"], ctx.team_id)]
    if "is_goalkeeper" in a.columns:
        a = a.loc[~bool_flag(a["is_goalkeeper"])]
    x = a["x"].to_numpy(dtype=float)
    gr = np.abs(x - ctx.own_goal_x)
    gr = gr[np.isfinite(gr)]
    unit = gr[gr <= _goal_relative_x(ctx.ball_x, ctx.own_goal_x)]
    return _stagger_label(unit)


def layer1_metrics(frame_rows: pd.DataFrame, ctx: SampleContext, *, params) -> dict:
    """All Layer-1 metrics for one sample, keyed by the ``RD_*`` column names."""
    return {
        RD_NUM_SUPERIORITY: rd_num_superiority(frame_rows, ctx, include_a_gk=False),
        RD_NUM_SUPERIORITY_GK: rd_num_superiority(frame_rows, ctx, include_a_gk=True),
        RD_ZONE_OCCUPANCY: rd_zone_occupancy(frame_rows, ctx, params=params),
        RD_LINE_HEIGHT: rd_line_height(ctx),
        RD_LINE_HEIGHT_RELATIVE: rd_line_height_relative(ctx),
        RD_COMPACTNESS_X: float(ctx.compactness_x),
        RD_WIDTH: float(ctx.lateral_width),
        RD_DEPTH: float(ctx.team_length),
        RD_SHAPE_STAGGER: rd_shape_stagger(frame_rows, ctx),
        RD_GK_LINE_HEIGHT: rd_gk_line_height(frame_rows, ctx),
        RD_GK_TO_LINE_DISTANCE: rd_gk_to_line_distance(frame_rows, ctx),
    }
