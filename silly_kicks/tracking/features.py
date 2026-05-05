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
- defending_gk_from_frames(actions, frames) -> pd.Series       (PR-S27, TF-13)
- defensive_line_x / back_line_high_x / compactness_x / lateral_width /
  max_lateral_gap / back_n_count (actions, frames, *, home_team_id) (PR-S27, TF-14)
- add_defensive_line(actions, frames, *, home_team_id) -> pd.DataFrame  (PR-S27)
- defensive_line_xfns(home_team_id) -> list                    (PR-S27)

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
Spec: docs/superpowers/specs/2026-04-30-action-context-pr1-design.md (PR-S20)
      docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md (PR-S21)
      docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md (PR-S27)
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.spadl import config as spadlconfig

from . import _kernels
from ._ball_carrier import infer_ball_carrier
from ._gk_resolve import defending_gk_from_frames
from .feature_framework import lift_to_states
from .pressure import (
    AndrienkoParams,
    BekkersParams,
    LinkParams,
    Method,
    PressureParams,
    validate_params_for_method,
)
from .utils import _resolve_action_frame_context, link_actions_to_frames

_STANDARD_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty"))

__all__ = [
    "Method",
    "actor_arc_length_pre_window",
    "actor_displacement_pre_window",
    "actor_pre_window_default_xfns",
    "actor_speed",
    "add_action_context",
    "add_actor_pre_window",
    "add_defensive_line",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_pressure_on_actor",
    "back_line_high_x",
    "back_n_count",
    "ball_carrier_at_action",
    "compactness_x",
    "defenders_in_triangle_to_goal",
    "defending_gk_from_frames",
    "defensive_line_x",
    "defensive_line_xfns",
    "lateral_width",
    "max_lateral_gap",
    "nearest_defender_distance",
    "pre_shot_gk_angle_default_xfns",
    "pre_shot_gk_angle_off_goal_line",
    "pre_shot_gk_angle_to_shot_trajectory",
    "pre_shot_gk_default_xfns",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "pre_shot_gk_full_default_xfns",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pressure_default_xfns",
    "pressure_on_actor",
    "receiver_zone_density",
    "tracking_default_xfns",
]


def ball_carrier_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
    tolerance_m: float = 3.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> pd.Series:
    """Per-action ball carrier player_id resolved from tracking frames.

    Links actions to frames via ``link_actions_to_frames``, then looks up
    the ``infer_ball_carrier`` result at the linked frame.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds.
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link.
    tolerance_m : float, default 3.0
        Carrier-attribution radius passed to ``infer_ball_carrier``.
    beta : float, default 0.5
        Velocity weight passed to ``infer_ball_carrier``.
    gamma : float, default 1.0
        Hysteresis bonus passed to ``infer_ball_carrier``.

    Returns
    -------
    pd.Series
        Aligned with actions.index. dtype matches frames' player_id dtype.
        NaN where action couldn't link or no carrier found.

    Examples
    --------
    Get the ball carrier at each action::

        from silly_kicks.tracking.features import ball_carrier_at_action
        carrier = ball_carrier_at_action(actions, frames)

    See NOTICE for full bibliographic citations.
    """
    import numpy as np

    pid_dtype = frames["player_id"].dtype
    n = len(actions)
    out = pd.Series(np.full(n, np.nan), index=actions.index, dtype="object")

    if n == 0 or len(frames) == 0:
        return out

    # Compute per-frame carriers
    carriers = infer_ball_carrier(frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma)
    if carriers.empty:
        return out

    # Link actions to frames
    pointers, _report = link_actions_to_frames(actions, frames, tolerance_seconds=tolerance_seconds)

    # Join pointers with actions to get period_id + game_id
    ptr = pointers.merge(
        actions[["action_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )
    linked = ptr[ptr["frame_id"].notna()].copy()
    if linked.empty:
        return out

    linked["frame_id_int"] = linked["frame_id"].astype("int64")

    # Join with carriers on (game_id, period_id, frame_id)
    merged = linked.merge(
        carriers[["game_id", "period_id", "frame_id", "ball_carrier_player_id"]],
        left_on=["game_id", "period_id", "frame_id_int"],
        right_on=["game_id", "period_id", "frame_id"],
        how="left",
    )

    # Deduplicate: one carrier per action_id (take first)
    merged = merged.drop_duplicates("action_id", keep="first")

    # Map back to actions index
    action_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())
    for _, row in merged.iterrows():
        aid = row["action_id"]
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = row["ball_carrier_player_id"]

    # Cast to match frames dtype if numeric
    if pid_dtype == np.dtype("int64") or str(pid_dtype) == "Int64":
        out = pd.to_numeric(out, errors="coerce")
        if str(pid_dtype) == "Int64":
            out = out.astype("Int64")

    return out


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


# ---------------------------------------------------------------------------
# PR-S24 -- TF-12: pre_shot_gk_angle_*
# ---------------------------------------------------------------------------


def pre_shot_gk_angle_to_shot_trajectory(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Signed angle (rad) between (goal-centre->anchor) and (GK->anchor) at the linked frame.

    Zero ==> GK is on the shot trajectory line. Positive ==> GK to +y side; negative ==> -y side.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal." Frontiers in
        Sports and Active Living, 3, 624475.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_to_shot_trajectory"].rename("pre_shot_gk_angle_to_shot_trajectory")


def pre_shot_gk_angle_off_goal_line(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Signed angle (rad) of GK position relative to goal-line normal at goal-mouth centre.

    Zero ==> GK is on the goal-line normal. Positive ==> GK offset to +y side; negative ==> -y side.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_off_goal_line"].rename("pre_shot_gk_angle_off_goal_line")


@nan_safe_enrichment
def add_pre_shot_gk_angle(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """Add 2 GK-angle columns at the linked frame for each shot action.

    REQUIRES ``defending_gk_player_id`` column (run
    ``silly_kicks.spadl.utils.add_pre_shot_gk_context`` first).

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - pre_shot_gk_angle_to_shot_trajectory (float64, radians, signed)
        - pre_shot_gk_angle_off_goal_line (float64, radians, signed)

    NaN for non-shot / unlinked / pre-engagement / GK-absent rows. Standalone
    aggregator -- does NOT extend ``add_pre_shot_gk_position`` (preserves the
    PR-S21 4-column surface; primitive+assembly pattern).

    Raises
    ------
    ValueError
        If ``defending_gk_player_id`` column is absent from ``actions``.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_angle: actions missing required column 'defending_gk_player_id'. "
            "Run silly_kicks.spadl.utils.add_pre_shot_gk_context first."
        )
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS)
    out = actions.copy()
    for col in ("pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"):
        out[col] = df[col]
    return out


pre_shot_gk_angle_default_xfns = [
    lift_to_states(pre_shot_gk_angle_to_shot_trajectory),
    lift_to_states(pre_shot_gk_angle_off_goal_line),
]


pre_shot_gk_full_default_xfns = pre_shot_gk_default_xfns + pre_shot_gk_angle_default_xfns


# ---------------------------------------------------------------------------
# PR-S25 -- TF-3: actor_*_pre_window features
# ---------------------------------------------------------------------------


def actor_arc_length_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Geometric arc-length of actor's path over the pre-action window (m).

    Per-action sum of consecutive segment distances over frames in
    (action_time - pre_seconds, action_time], filtered to actor's player_id
    within the same period:

        sum_{k=1..N-1} sqrt((x_{k+1} - x_k)**2 + (y_{k+1} - y_k)**2)

    Consecutive segments computed AFTER sorting by frame timestamp ASC and
    dropping frames with NaN positions (bridge rule per spec section 3.2).
    NaN if fewer than 2 valid frames remain.

    The pre_seconds=0.5 default captures sub-second pre-action movement
    intensity. For longer windows like Bauer & Anzer 2021 counterpressing
    detection (5s), pass pre_seconds=5.0.

    NOT a re-implementation of any paper's filtered/threshold-based
    "covered distance" feature -- pure geometric arc-length, no
    sprint-intensity filtering. See NOTICE.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking.features import actor_arc_length_pre_window
    >>> actions = pd.DataFrame({
    ...     "action_id": [1], "period_id": [1], "time_seconds": [10.0],
    ...     "player_id": [42], "team_id": [1], "start_x": [50.0],
    ...     "start_y": [34.0], "type_id": [0],
    ... })
    >>> frames = pd.DataFrame()  # empty -> all-NaN; runnable example
    >>> _ = actor_arc_length_pre_window(actions, frames)
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_arc_length_pre_window"].rename("actor_arc_length_pre_window")


def actor_displacement_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Net Euclidean displacement (window-first to window-last valid position).

    Differs from arc-length: a player who runs in a circle has high
    arc-length but ~zero displacement.

    NaN semantics identical to :func:`actor_arc_length_pre_window`. See NOTICE.

    Examples
    --------
    >>> from silly_kicks.tracking.features import actor_displacement_pre_window
    >>> # See tests/tracking/test_pre_window_features.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_displacement_pre_window"].rename("actor_displacement_pre_window")


@nan_safe_enrichment
def add_actor_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.DataFrame:
    """Enrich actions with 2 TF-3 movement columns + 4 linkage-provenance columns.

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - actor_arc_length_pre_window (float64, m)
        - actor_displacement_pre_window (float64, m)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - n_candidate_frames (int64)
        - link_quality_score (float64; NaN if unlinked)

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_actor_pre_window
    >>> # See tests/tracking/test_pre_window_features.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    out = actions.copy()
    out["actor_arc_length_pre_window"] = df["actor_arc_length_pre_window"]
    out["actor_displacement_pre_window"] = df["actor_displacement_pre_window"]
    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


actor_pre_window_default_xfns = [lift_to_states(actor_arc_length_pre_window)]


# ---------------------------------------------------------------------------
# PR-S25 -- TF-2: pressure_on_actor multi-flavor feature
# ---------------------------------------------------------------------------


def _build_ball_xy_v_per_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    ctx,
) -> pd.DataFrame:
    """Build per-action ball position+velocity at the linked frame.

    Joins on ``(period_id, frame_id)`` jointly -- ``frame_id`` alone is not
    unique across periods (PR-S25 e2e regression).
    """
    pointers = ctx.pointers
    actions_with_period = actions[["action_id", "period_id"]]
    pointers_with_period = pointers.merge(actions_with_period, on="action_id", how="left")
    ball_rows = frames.loc[frames["is_ball"], ["period_id", "frame_id", "x", "y", "vx", "vy"]]
    merged = pointers_with_period.merge(ball_rows, on=["period_id", "frame_id"], how="left")
    return merged[["action_id", "x", "y", "vx", "vy"]]


def pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    method: Method = "andrienko_oval",
    params: PressureParams | None = None,
) -> pd.Series:
    """Pressure exerted on the action's actor at the linked frame.

    Three published methodologies via ``method=``:

    - ``"andrienko_oval"`` (default) - Andrienko et al. 2017 directional oval
      pressure; sum across opposing defenders. Output range [0, ~200%].
    - ``"link_zones"`` - Link et al. 2016 piecewise-zone pressure;
      saturating exponential aggregation. Output [0, 1].
    - ``"bekkers_pi"`` - Bekkers 2024 Pressing Intensity probabilistic TTI;
      requires velocity columns vx/vy in frames. Output [0, 1].

    Returns Series named ``pressure_on_actor__<method>`` (suffix-naming
    convention per ADR-005 section 8 multi-flavor xfn rule).

    NaN where action couldn't link; 0.0 where linked but no defenders
    contribute pressure. ``bekkers_pi`` raises ValueError if frames lack
    vx/vy or (when use_ball_carrier_max=True) if frames lack any ball rows.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import pressure_on_actor
    >>> # See tests/tracking/test_pressure_*.py for runnable examples per method.
    """
    validate_params_for_method(method, params)
    if method == "andrienko_oval":
        ap = params if isinstance(params, AndrienkoParams) else AndrienkoParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_andrienko(actions["start_x"], actions["start_y"], ctx, params=ap)
    elif method == "link_zones":
        lp = params if isinstance(params, LinkParams) else LinkParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_link(actions["start_x"], actions["start_y"], ctx, params=lp)
    elif method == "bekkers_pi":
        bp = params if isinstance(params, BekkersParams) else BekkersParams()
        if "vx" not in frames.columns or "vy" not in frames.columns:
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi'): frames missing velocity columns "
                "'vx'/'vy'. Run silly_kicks.tracking.preprocess.derive_velocities(frames) "
                "first, or use a provider that emits velocities natively."
            )
        if bp.use_ball_carrier_max and not frames["is_ball"].any():
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi', params.use_ball_carrier_max=True): "
                "frames missing is_ball=True rows in linked frames. Either set "
                "use_ball_carrier_max=False to compute pressure-on-player only, or "
                "use a provider that emits ball positions per frame."
            )
        ctx = _resolve_action_frame_context(actions, frames)
        ball_xy_v_per_action = _build_ball_xy_v_per_action(actions, frames, ctx)
        s = _kernels._pressure_bekkers(
            actions["start_x"],
            actions["start_y"],
            ctx,
            params=bp,
            ball_xy_v_per_action=ball_xy_v_per_action,
        )
    else:
        # Defensive; validate_params_for_method already raised
        raise ValueError(f"Unknown method '{method}'.")
    return s.rename(f"pressure_on_actor__{method}")


@nan_safe_enrichment
def add_pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    methods: tuple[Method, ...] = ("andrienko_oval",),
    params_per_method: dict[Method, PressureParams] | None = None,
) -> pd.DataFrame:
    """Enrich actions with one ``pressure_on_actor__<m>`` column per method
    + 4 linkage-provenance columns.

    Validates all (method, params) pairs BEFORE computing any column
    (transactional behavior per spec section 8.5).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_pressure_on_actor
    >>> # See tests/tracking/test_pressure_*.py for runnable examples.
    """
    if params_per_method is None:
        params_per_method = {}
    # Validate all upfront (transactional)
    for m in methods:
        validate_params_for_method(m, params_per_method.get(m))

    out = actions.copy()
    for m in methods:
        params = params_per_method.get(m)
        s = pressure_on_actor(actions, frames, method=m, params=params)
        out[f"pressure_on_actor__{m}"] = s.values

    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


pressure_default_xfns = [lift_to_states(pressure_on_actor)]


# ---------------------------------------------------------------------------
# PR-S27 -- TF-14: defensive-line features
# ---------------------------------------------------------------------------


def defensive_line_x(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """Mean x of the defending team's back-line at the linked frame (m).

    NaN where action is unlinked or defending team has <3 valid outfield players.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import defensive_line_x
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["defensive_line_x"].rename("defensive_line_x")


def back_line_high_x(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """x of the most advanced back-line player on the defending team (m).

    Approximates the offside line when the GK is behind the defensive line
    (typical case); NOT law-compliant for sweeper-keeper scenarios.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import back_line_high_x
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["back_line_high_x"].rename("back_line_high_x")


def compactness_x(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """x-spread of defending team's back-line (max - min, meters).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import compactness_x
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["compactness_x"].rename("compactness_x")


def lateral_width(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """y-spread of defending team's back-line (max - min, meters).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import lateral_width
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["lateral_width"].rename("lateral_width")


def max_lateral_gap(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """Largest y-gap between adjacent y-sorted back-line players (m).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import max_lateral_gap
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["max_lateral_gap"].rename("max_lateral_gap")


def back_n_count(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.Series:
    """Number of players in the defending team's back line (3/4/5).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import back_n_count
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    return df["back_n_count"].rename("back_n_count")


@nan_safe_enrichment
def add_defensive_line(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.DataFrame:
    """Enrich actions with 6 defensive-line columns + 4 linkage-provenance columns.

    Provenance columns (frame_id, time_offset_seconds, link_quality_score,
    n_candidate_frames) are skipped if they already exist on the input DataFrame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_defensive_line
    >>> # See tests/tracking/test_defensive_line_features.py for runnable examples.
    """
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n)
    out = actions.copy()
    for col in ("defensive_line_x", "back_line_high_x", "compactness_x", "lateral_width", "max_lateral_gap"):
        out[col] = df[col]
    out["back_n_count"] = df["back_n_count"].astype("Int64")

    # Provenance: skip if already present (idempotent with other add_* enrichments)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing_provenance = [c for c in provenance_cols if c in out.columns]
    if not existing_provenance:
        pointers, _report = link_actions_to_frames(actions, frames)
        pointer_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


def defensive_line_xfns(
    home_team_id: int | str,
    *,
    n: int | Literal["adaptive"] = 4,
) -> list:
    """Build VAEP xfn list bound to a specific home_team_id.

    Returns a list with ONE FrameAwareTransformer that emits all 6
    defensive-line columns x 3 game-states = 18 columns total. This ensures
    compute_defensive_line is called 3x (once per state), not 18x.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, defensive_line_xfns
        xfns = tracking_default_xfns + defensive_line_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    col_names = [
        "defensive_line_x",
        "back_line_high_x",
        "compactness_x",
        "lateral_width",
        "max_lateral_gap",
        "back_n_count",
    ]

    def _defensive_line_transformer(states, frames):
        """Multi-column defensive-line xfn (6 cols x nb_states)."""
        out = pd.DataFrame(index=states[0].index)
        for i, slot in enumerate(states[:3]):
            batch = _kernels._defensive_line_at_actions(slot, frames, home_team_id=home_team_id, n=n)
            for col in col_names:
                out[f"{col}_a{i}"] = batch[col].to_numpy()
        return out

    _defensive_line_transformer._frame_aware = True  # type: ignore[attr-defined]
    _defensive_line_transformer.__name__ = "defensive_line"
    return [_defensive_line_transformer]
