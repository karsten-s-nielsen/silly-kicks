"""Tracking-aware action_context features for standard SPADL.

Public API:
- nearest_defender_distance(actions, frames) -> pd.Series
- actor_speed(actions, frames) -> pd.Series
- receiver_zone_density(actions, frames, *, radius=5.0) -> pd.Series
- defenders_in_triangle_to_goal(actions, frames) -> pd.Series
- add_action_context(actions, frames, *, receiver_zone_radius=5.0) -> pd.DataFrame
- pre_shot_gk_x(actions, frames) -> pd.Series         (PR-S21)
- pre_shot_gk_y(actions, frames) -> pd.Series         (PR-S21)
- pre_shot_gk_distance_to_goal(actions, frames) -> pd.Series   (PR-S21)
- pre_shot_gk_distance_to_shot(actions, frames) -> pd.Series   (PR-S21)
- add_pre_shot_gk_position(actions, frames) -> pd.DataFrame    (PR-S21)
- tracking_default_xfns: list[FrameAwareTransformer]
- pre_shot_gk_default_xfns: list[FrameAwareTransformer]   (PR-S21)

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
Spec: docs/superpowers/specs/2026-04-30-action-context-pr1-design.md (PR-S20)
      docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md (PR-S21)
"""

from __future__ import annotations

import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.spadl import config as spadlconfig

from . import _kernels
from .feature_framework import lift_to_states
from .utils import _resolve_action_frame_context

_STANDARD_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty"))

__all__ = [
    "actor_speed",
    "add_action_context",
    "add_pre_shot_gk_position",
    "defenders_in_triangle_to_goal",
    "nearest_defender_distance",
    "pre_shot_gk_default_xfns",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "receiver_zone_density",
    "tracking_default_xfns",
]


def nearest_defender_distance(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Meters to the closest opposing-team player at the linked frame.

    Anchor: ``(action.start_x, action.start_y)``. NaN if action couldn't link to a frame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    Compute defender distance for a SPADL action stream::

        from silly_kicks.tracking.features import nearest_defender_distance
        d = nearest_defender_distance(actions, frames)

    References
    ----------
    Lucey et al. (2014). "Quality vs Quantity: Improved Shot Prediction in Soccer
        using Strategic Features from Spatiotemporal Data." MIT Sloan SAC.
    Anzer & Bauer (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal."
        Frontiers in Sports and Active Living, 3, 624475.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._nearest_defender_distance(actions["start_x"], actions["start_y"], ctx)


def actor_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """m/s of the action's player_id at the linked frame.

    NaN if the action couldn't link, the actor's player_id is absent from the linked
    frame, or the frame's speed value is NaN.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import actor_speed
        s = actor_speed(actions, frames)

    References
    ----------
    Anzer & Bauer (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal."
        Frontiers in Sports and Active Living, 3, 624475.
    Bauer & Anzer (2021). "Data-driven detection of counterpressing in professional
        football." Data Mining and Knowledge Discovery, 35(5), 2009-2049.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._actor_speed_from_ctx(ctx)


def receiver_zone_density(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    radius: float = 5.0,
) -> pd.Series:
    """Count of opposing-team players within ``radius`` of (action.end_x, action.end_y).

    Integer-valued (0 if linked but no defenders within radius; NaN if unlinked).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import receiver_zone_density
        d = receiver_zone_density(actions, frames, radius=5.0)

    References
    ----------
    Spearman (2018). "Beyond Expected Goals." MIT Sloan SAC.
    Power et al. (2017). "Not all passes are created equal." KDD '17 (OBSO).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._receiver_zone_density(actions["end_x"], actions["end_y"], ctx, radius=radius)


def defenders_in_triangle_to_goal(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.Series:
    """Count of opposing-team players inside the triangle
    (action.start_x, action.start_y) -> goal-mouth posts at x=105.

    Goal-mouth: y in [30.34, 37.66] per spadl.config.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import defenders_in_triangle_to_goal
        d = defenders_in_triangle_to_goal(actions, frames)

    References
    ----------
    Lucey et al. (2014). "Quality vs Quantity: Improved Shot Prediction in Soccer
        using Strategic Features from Spatiotemporal Data." MIT Sloan SAC.
    Pollard & Reep (1997). "Measuring the effectiveness of playing strategies at
        soccer." J. Royal Statistical Society Series D, 46(4), 541-550.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._defenders_in_triangle_to_goal(actions["start_x"], actions["start_y"], ctx)


@nan_safe_enrichment
def add_action_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    receiver_zone_radius: float = 5.0,
) -> pd.DataFrame:
    """Enrich actions with 4 tracking-aware features + 4 linkage-provenance columns.

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - nearest_defender_distance (float64, meters)
        - actor_speed (float64, m/s)
        - receiver_zone_density (Int64, count; NaN unlinked, 0 = no defenders)
        - defenders_in_triangle_to_goal (Int64, count; NaN unlinked, 0 = none)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import add_action_context
        enriched = add_action_context(actions, frames, receiver_zone_radius=5.0)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    out = actions.copy()
    out["nearest_defender_distance"] = _kernels._nearest_defender_distance(actions["start_x"], actions["start_y"], ctx)
    out["actor_speed"] = _kernels._actor_speed_from_ctx(ctx)
    rz = _kernels._receiver_zone_density(actions["end_x"], actions["end_y"], ctx, radius=receiver_zone_radius)
    out["receiver_zone_density"] = rz.astype("Int64")
    dt = _kernels._defenders_in_triangle_to_goal(actions["start_x"], actions["start_y"], ctx)
    out["defenders_in_triangle_to_goal"] = dt.astype("Int64")
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


tracking_default_xfns = [
    lift_to_states(nearest_defender_distance),
    lift_to_states(actor_speed),
    lift_to_states(receiver_zone_density),
    lift_to_states(defenders_in_triangle_to_goal),
]


# ---------------------------------------------------------------------------
# PR-S21 — pre_shot_gk_* features
# ---------------------------------------------------------------------------


def pre_shot_gk_x(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Defending GK's x at the linked frame (m, LTR-normalized).

    NaN for non-shot rows, unlinked actions, pre-engagement (NaN
    ``defending_gk_player_id``), or GK-absent-from-frame (substitution) cases.

    REQUIRES the actions DataFrame to have a ``defending_gk_player_id``
    column (run ``silly_kicks.spadl.utils.add_pre_shot_gk_context`` first).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    Compute defending-GK x for a SPADL action stream after engagement-state enrichment::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_x
        actions = add_pre_shot_gk_context(actions)
        gk_x = pre_shot_gk_x(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal." Frontiers in
        Sports and Active Living, 3, 624475. (defending-GK-position as xG feature)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS
    )
    return df["pre_shot_gk_x"].rename("pre_shot_gk_x")


def pre_shot_gk_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Defending GK's y at the linked frame (m, LTR-normalized).

    NaN semantics identical to :func:`pre_shot_gk_x`. REQUIRES
    ``defending_gk_player_id`` column in ``actions``.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_y
        actions = add_pre_shot_gk_context(actions)
        gk_y = pre_shot_gk_y(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS
    )
    return df["pre_shot_gk_y"].rename("pre_shot_gk_y")


def pre_shot_gk_distance_to_goal(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Euclidean distance (m) from defending GK to goal-mouth center (105, 34).

    NaN semantics identical to :func:`pre_shot_gk_x`. REQUIRES
    ``defending_gk_player_id`` column in ``actions``.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_distance_to_goal
        actions = add_pre_shot_gk_context(actions)
        d = pre_shot_gk_distance_to_goal(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS
    )
    return df["pre_shot_gk_distance_to_goal"].rename("pre_shot_gk_distance_to_goal")


def pre_shot_gk_distance_to_shot(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Euclidean distance (m) from defending GK to shot anchor (action.start_x, action.start_y).

    NaN semantics identical to :func:`pre_shot_gk_x`. REQUIRES
    ``defending_gk_player_id`` column in ``actions``.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_distance_to_shot
        actions = add_pre_shot_gk_context(actions)
        d = pre_shot_gk_distance_to_shot(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS
    )
    return df["pre_shot_gk_distance_to_shot"].rename("pre_shot_gk_distance_to_shot")


@nan_safe_enrichment
def add_pre_shot_gk_position(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich actions with 4 GK-position columns + 4 linkage-provenance columns.

    REQUIRES the actions DataFrame to have a ``defending_gk_player_id`` column
    (run ``silly_kicks.spadl.utils.add_pre_shot_gk_context`` first).

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - pre_shot_gk_x (float64, m)
        - pre_shot_gk_y (float64, m)
        - pre_shot_gk_distance_to_goal (float64, m)
        - pre_shot_gk_distance_to_shot (float64, m)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)

    All 4 GK columns are NaN for non-shot / unlinked / pre-engagement /
    GK-absent-from-frame rows.

    Raises
    ------
    ValueError
        If ``defending_gk_player_id`` column is absent from ``actions``.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    Tag pre-shot defending-GK position via the tracking-namespace canonical compute::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import add_pre_shot_gk_position
        actions = add_pre_shot_gk_context(actions)            # populates defending_gk_player_id
        enriched = add_pre_shot_gk_position(actions, frames)  # adds 4 GK + 4 provenance columns
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_position: actions missing required column "
            "'defending_gk_player_id'. Run silly_kicks.spadl.utils.add_pre_shot_gk_context "
            "first to populate it."
        )
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS
    )
    out = actions.copy()
    for col in ("pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot"):
        out[col] = df[col]
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


pre_shot_gk_default_xfns = [
    lift_to_states(pre_shot_gk_x),
    lift_to_states(pre_shot_gk_y),
    lift_to_states(pre_shot_gk_distance_to_goal),
    lift_to_states(pre_shot_gk_distance_to_shot),
]
