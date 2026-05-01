"""Tracking-aware action_context features for atomic SPADL.

Mirrors silly_kicks.tracking.features with atomic-shaped column reads.
Shares the schema-agnostic kernels in silly_kicks.tracking._kernels.

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.tracking import _kernels
from silly_kicks.tracking.feature_framework import lift_to_states
from silly_kicks.tracking.utils import _resolve_action_frame_context

__all__ = [
    "actor_speed",
    "add_action_context",
    "atomic_tracking_default_xfns",
    "defenders_in_triangle_to_goal",
    "nearest_defender_distance",
    "receiver_zone_density",
]


def nearest_defender_distance(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: distance to nearest defender at action anchor (x, y).

    See NOTICE; matches silly_kicks.tracking.features.nearest_defender_distance.

    Examples
    --------
    Compute defender distance for an atomic action stream::

        from silly_kicks.atomic.tracking.features import nearest_defender_distance
        d = nearest_defender_distance(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._nearest_defender_distance(actions["x"], actions["y"], ctx)


def actor_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: actor's speed at the linked frame.

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import actor_speed
        s = actor_speed(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._actor_speed_from_ctx(ctx)


def receiver_zone_density(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    radius: float = 5.0,
) -> pd.Series:
    """Atomic-SPADL: defenders within radius of (x + dx, y + dy).

    Degenerate case: when dx == dy == 0 (instantaneous atomic actions like shots),
    density is computed at the anchor (x, y).

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import receiver_zone_density
        d = receiver_zone_density(atomic_actions, frames, radius=5.0)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    end_x = actions["x"] + actions["dx"]
    end_y = actions["y"] + actions["dy"]
    return _kernels._receiver_zone_density(end_x, end_y, ctx, radius=radius)


def defenders_in_triangle_to_goal(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.Series:
    """Atomic-SPADL: defenders in triangle from (x, y) to goal posts.

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import defenders_in_triangle_to_goal
        d = defenders_in_triangle_to_goal(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._defenders_in_triangle_to_goal(actions["x"], actions["y"], ctx)


@nan_safe_enrichment
def add_action_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    receiver_zone_radius: float = 5.0,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: enrich actions with the 4 features + 4 provenance cols.

    Parallels silly_kicks.tracking.features.add_action_context with atomic-shaped
    column reads (x, y, dx, dy).

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import add_action_context
        enriched = add_action_context(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    out = actions.copy()
    out["nearest_defender_distance"] = _kernels._nearest_defender_distance(actions["x"], actions["y"], ctx)
    out["actor_speed"] = _kernels._actor_speed_from_ctx(ctx)
    end_x = actions["x"] + actions["dx"]
    end_y = actions["y"] + actions["dy"]
    rz = _kernels._receiver_zone_density(end_x, end_y, ctx, radius=receiver_zone_radius)
    out["receiver_zone_density"] = rz.astype("Int64")
    dt = _kernels._defenders_in_triangle_to_goal(actions["x"], actions["y"], ctx)
    out["defenders_in_triangle_to_goal"] = dt.astype("Int64")
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_tracking_default_xfns = [
    lift_to_states(nearest_defender_distance),
    lift_to_states(actor_speed),
    lift_to_states(receiver_zone_density),
    lift_to_states(defenders_in_triangle_to_goal),
]
