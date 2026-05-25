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
- pitch_control_at_action(actions, frames) -> pd.Series        (PR-S31, TF-7)
- add_pitch_control(actions, frames) -> pd.DataFrame           (PR-S31, TF-7)
- pitch_control_xfns(method) -> list                           (PR-S31, TF-7)
- pitch_control_default_xfns: list[FrameAwareTransformer]      (PR-S31, TF-7)
- gk_pitch_control_share_weighted / gk_reachable_area_m2 /
  gk_closing_time_min_s / gk_closing_time_mean_s               (PR-S34, TF-15)
- add_gk_influence(actions, frames, xt, *, home_team_id) -> pd.DataFrame (PR-S34)
- gk_influence_xfns(xt, *, home_team_id) -> list               (PR-S34)

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
Spec: docs/superpowers/specs/2026-04-30-action-context-pr1-design.md (PR-S20)
      docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md (PR-S21)
      docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md (PR-S27)
      docs/superpowers/specs/2026-05-09-tf15-gk-influence-primitives-design.md (PR-S34)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat

    from ._line_breaking import LineBreakingParams

import numpy as np
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
    "actor_reachable_area_m2",
    "actor_speed",
    "add_action_context",
    "add_actor_pre_window",
    "add_cover_shadows",
    "add_das",
    "add_defensive_line",
    "add_gk_influence",
    "add_line_break",
    "add_off_ball_context",
    "add_off_ball_runs",
    "add_pitch_control",
    "add_player_influence",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_pressure_on_actor",
    "add_team_shape",
    "back_line_high_x",
    "back_n_count",
    "ball_carrier_at_action",
    "compactness_x",
    "cover_shadow_xfns",
    "das_at_action",
    "das_xfns",
    "defenders_in_triangle_to_goal",
    "defending_gk_from_frames",
    "defensive_line_x",
    "defensive_line_xfns",
    "gk_closing_time_mean_s",
    "gk_closing_time_min_s",
    "gk_influence_xfns",
    "gk_pitch_control_share_weighted",
    "gk_reachable_area_m2",
    "lateral_width",
    "line_breaking_ward_xfns",
    "max_lateral_gap",
    "nearest_defender_distance",
    "off_ball_context_xfns",
    "off_ball_xt_opponent",
    "off_ball_xt_team",
    "pitch_control_at_action",
    "pitch_control_default_xfns",
    "pitch_control_xfns",
    "player_influence_xfns",
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
    "reachable_area_opponent",
    "reachable_area_team",
    "receiver_zone_density",
    "team_shape_xfns",
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

    # Align game_id dtype between linked (from actions) and carriers (from frames)
    # before the merge — pandas rejects merge on object vs int64 keys.
    carriers_proj = carriers[["game_id", "period_id", "frame_id", "ball_carrier_player_id"]].copy()
    if len(linked) > 0 and len(carriers_proj) > 0:
        if linked["game_id"].dtype != carriers_proj["game_id"].dtype:
            linked["game_id"] = linked["game_id"].astype(str)
            carriers_proj["game_id"] = carriers_proj["game_id"].astype(str)

    # Join with carriers on (game_id, period_id, frame_id)
    merged = linked.merge(
        carriers_proj,
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
    links: pd.DataFrame | None = None,
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
    ctx = _resolve_action_frame_context(actions, frames, links=links)
    out = actions.copy()
    out["nearest_defender_distance"] = _kernels._nearest_defender_distance(actions["start_x"], actions["start_y"], ctx)
    out["actor_speed"] = _kernels._actor_speed_from_ctx(ctx)
    rz = _kernels._receiver_zone_density(actions["end_x"], actions["end_y"], ctx, radius=receiver_zone_radius)
    out["receiver_zone_density"] = rz.astype("Int64")
    dt = _kernels._defenders_in_triangle_to_goal(actions["start_x"], actions["start_y"], ctx)
    out["defenders_in_triangle_to_goal"] = dt.astype("Int64")
    # Provenance: skip if already present (idempotent with other add_* enrichments)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        pointer_cols = ctx.pointers.set_index("action_id")[provenance_cols]
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
    *,
    links: pd.DataFrame | None = None,
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
    ctx = _resolve_action_frame_context(actions, frames, links=links)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS
    )
    out = actions.copy()
    for col in ("pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot"):
        out[col] = df[col]
    # Provenance: skip if already present (idempotent with other add_* enrichments)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        pointer_cols = ctx.pointers.set_index("action_id")[provenance_cols]
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
    links: pd.DataFrame | None = None,
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
    ctx = _resolve_action_frame_context(actions, frames, links=links)
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
    links: pd.DataFrame | None = None,
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
    # Provenance: skip if already present (idempotent with other add_* enrichments)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        if links is not None:
            pointers = links
        else:
            pointers, _report = link_actions_to_frames(actions, frames)
        pointer_cols = pointers.set_index("action_id")[provenance_cols]
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
    links: pd.DataFrame | None = None,
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
        ctx = _resolve_action_frame_context(actions, frames, links=links)
        s = _kernels._pressure_andrienko(actions["start_x"], actions["start_y"], ctx, params=ap)
    elif method == "link_zones":
        lp = params if isinstance(params, LinkParams) else LinkParams()
        ctx = _resolve_action_frame_context(actions, frames, links=links)
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
        ctx = _resolve_action_frame_context(actions, frames, links=links)
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
    links: pd.DataFrame | None = None,
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
        s = pressure_on_actor(actions, frames, method=m, params=params, links=links)
        out[f"pressure_on_actor__{m}"] = s.values

    # Provenance: skip if already present (idempotent with other add_* enrichments)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        if links is not None:
            pointers = links
        else:
            pointers, _report = link_actions_to_frames(actions, frames)
        pointer_cols = pointers.set_index("action_id")[provenance_cols]
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
    links: pd.DataFrame | None = None,
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
    df = _kernels._defensive_line_at_actions(actions, frames, home_team_id=home_team_id, n=n, links=links)
    out = actions.copy()
    for col in ("defensive_line_x", "back_line_high_x", "compactness_x", "lateral_width", "max_lateral_gap"):
        out[col] = df[col]
    out["back_n_count"] = df["back_n_count"].astype("Int64")

    # Provenance: skip if already present (idempotent with other add_* enrichments)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing_provenance = [c for c in provenance_cols if c in out.columns]
    if not existing_provenance:
        if links is not None:
            pointers = links
        else:
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


# ---------------------------------------------------------------------------
# PR-S30 -- TF-4: off-ball runs + line-break features
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_off_ball_runs(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Enrich actions with 4 off-ball-run columns.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_off_ball_runs
    >>> # See tests/tracking/test_off_ball_runs.py for runnable examples.
    """
    from ._off_ball_runs import _off_ball_runs_kernel

    df = _off_ball_runs_kernel(
        actions,
        frames,
        home_team_id=home_team_id,
        pre_seconds=pre_seconds,
        min_displacement_m=min_displacement_m,
    )
    out = actions.copy()
    for col in (
        "n_off_ball_runners_pre_window",
        "max_off_ball_run_displacement_pre_window",
        "mean_off_ball_run_speed_pre_window",
        "n_off_ball_runners_toward_goal_pre_window",
    ):
        out[col] = df[col]
    return out


@nan_safe_enrichment
def add_line_break(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    method: Literal["threshold", "ward"] = "threshold",
    n: int = 4,
    params: LineBreakingParams | None = None,
) -> pd.DataFrame:
    """Enrich actions with line-break columns.

    Two methods are available:

    - ``method="threshold"`` (default): Binary threshold test against the
      defending team's ``defensive_line_x``. Returns ``line_break`` (bool)
      and ``n_attackers_behind_line`` (Int64). Backward-compatible default.
      ``params`` is ignored.
    - ``method="ward"``: Ward-clustering line identification + segment
      intersection. Returns ``line_break__ward`` (bool),
      ``lines_broken__ward`` (Int64, 0-3), ``line_breaking_type__ward``
      (str: "between_lines"/"around_line"/None). ``n`` is ignored.

    Column sets are disjoint between methods (no collision). A consumer
    can call both methods if they want all 5 columns (note: each call
    performs its own ``link_actions_to_frames`` --- see §1.4 linkage
    cost note in the spec).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_line_break
    >>> # See tests/tracking/test_line_breaking.py for runnable examples.
    """
    if method == "threshold":
        from ._off_ball_runs import _line_break_kernel

        df = _line_break_kernel(actions, frames, home_team_id=home_team_id, n=n, links=links)
        out = actions.copy()
        out["line_break"] = df["line_break"]
        out["n_attackers_behind_line"] = df["n_attackers_behind_line"]
        return out

    # method == "ward"
    from ._line_breaking import detect_line_breaking

    result = detect_line_breaking(actions, frames, home_team_id=home_team_id, params=params, links=links)
    out = actions.copy()
    out["line_break__ward"] = result["line_break__ward"]
    out["lines_broken__ward"] = result["lines_broken__ward"]
    out["line_breaking_type__ward"] = result["line_breaking_type__ward"]
    return out


@nan_safe_enrichment
def add_off_ball_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    n: int = 4,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Umbrella: add all 6 off-ball-run + line-break columns.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_off_ball_context
    >>> # See tests/tracking/test_off_ball_runs.py for runnable examples.
    """
    from ._off_ball_runs import _line_break_kernel, _off_ball_runs_kernel

    runs = _off_ball_runs_kernel(
        actions,
        frames,
        home_team_id=home_team_id,
        pre_seconds=pre_seconds,
        min_displacement_m=min_displacement_m,
    )
    lb = _line_break_kernel(actions, frames, home_team_id=home_team_id, n=n, links=links)
    out = actions.copy()
    for col in runs.columns:
        out[col] = runs[col]
    for col in lb.columns:
        out[col] = lb[col]
    return out


def off_ball_context_xfns(
    home_team_id: int | str,
    *,
    n: int = 4,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> list:
    """Build VAEP xfn list bound to home_team_id for TF-4 features.

    Returns a list with ONE FrameAwareTransformer that emits all 6
    off-ball-run + line-break columns x 3 game-states = 18 columns total.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, off_ball_context_xfns
        xfns = tracking_default_xfns + off_ball_context_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._off_ball_runs import _LINE_BREAK_COLS, _OFF_BALL_RUNS_COLS, _line_break_kernel, _off_ball_runs_kernel

    def _off_ball_context_transformer(states, frames):
        """Multi-column off-ball-context xfn (6 cols x nb_states).

        Known optimization target: _line_break_kernel calls
        compute_defensive_line per slot, but defensive-line depends only
        on frames (not actions) — result is identical across all 3 slots.
        Acceptable for v1; hoist into shared pre-computation if profiling
        shows this as a bottleneck.
        """
        out = pd.DataFrame(index=states[0].index)
        for i, slot in enumerate(states[:3]):
            runs = _off_ball_runs_kernel(
                slot,
                frames,
                home_team_id=home_team_id,
                pre_seconds=pre_seconds,
                min_displacement_m=min_displacement_m,
            )
            lb = _line_break_kernel(slot, frames, home_team_id=home_team_id, n=n)
            for col in _OFF_BALL_RUNS_COLS:
                out[f"{col}_a{i}"] = runs[col].to_numpy()
            for col in _LINE_BREAK_COLS:
                out[f"{col}_a{i}"] = lb[col].to_numpy()
        return out

    _off_ball_context_transformer._frame_aware = True  # type: ignore[attr-defined]
    _off_ball_context_transformer.__name__ = "off_ball_context"
    return [_off_ball_context_transformer]


# ---------------------------------------------------------------------------
# PR-S33 -- TF-31: team shape envelope
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_team_shape(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Enrich actions with 14 team-shape columns (7 metrics x 2 teams).

    Provenance columns (frame_id, time_offset_seconds, link_quality_score,
    n_candidate_frames) are skipped if they already exist on the input.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_team_shape
    >>> # See tests/tracking/test_team_shape.py for runnable examples.
    """
    from ._team_shape import compute_team_shape

    out = actions.copy()

    # Compute shape for both teams (ONCE each)
    teams = frames[~frames["is_ball"].astype(bool)]["team_id"].dropna().unique()
    if len(teams) < 2:
        # Can't determine attacking/defending split — fill NaN
        for suffix in ("attacking", "defending"):
            for metric in (
                "n_outfield_players",
                "centroid_x",
                "centroid_y",
                "convex_hull_area",
                "team_length",
                "team_width",
                "stretch_index",
            ):
                out[f"team_shape_{metric}_{suffix}"] = np.nan
        return out

    # Pre-compute and index shape by (game_id, period_id, frame_id) for O(1) lookup
    shape_indexed: dict = {}
    for tid in teams:
        s = compute_team_shape(frames, team_id=tid)
        shape_indexed[tid] = s.set_index(["game_id", "period_id", "frame_id"])

    # Link actions to frames
    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()

    metrics = [
        "n_outfield_players",
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
    ]

    # Initialize output columns to NaN
    for suffix in ("attacking", "defending"):
        for metric in metrics:
            out[f"team_shape_{metric}_{suffix}"] = np.nan

    if linked.empty:
        return out

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    linked = linked.merge(
        actions[["action_id", "team_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )

    # Align game_id dtype between linked (from actions) and shape_indexed keys
    # (from frames) — int64 vs object mismatch causes silent dict-key miss.
    if len(linked) > 0 and shape_indexed:
        sample_sdf = next(iter(shape_indexed.values()))
        if len(sample_sdf) > 0:
            sample_key_gid = sample_sdf.index[0][0]
            linked_gid_sample = linked["game_id"].iloc[0]
            if not isinstance(linked_gid_sample, type(sample_key_gid)):
                linked["game_id"] = linked["game_id"].astype(str)

    aid_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())

    for _, row in linked.iterrows():
        aid = row["action_id"]
        if aid not in aid_to_idx.index:
            continue
        idx = aid_to_idx.loc[aid]
        action_team = row["team_id"]
        if pd.isna(action_team):
            continue
        key = (row["game_id"], row["period_id"], int(row["frame_id_int"]))

        for tid, sdf in shape_indexed.items():
            if key not in sdf.index:
                continue
            shape_row = sdf.loc[key]
            suffix = "attacking" if tid == action_team else "defending"
            for metric in metrics:
                out.at[idx, f"team_shape_{metric}_{suffix}"] = shape_row[metric]

    # Provenance: skip if already present
    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]
    existing_provenance = [c for c in provenance_cols if c in out.columns]
    if not existing_provenance:
        pointer_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


def team_shape_xfns(home_team_id: int | str) -> list:
    """Build VAEP xfn list for TF-31 team shape features.

    Returns a list with ONE FrameAwareTransformer that emits 12 features x 3
    game-states = 36 columns total. ``n_outfield_players`` is excluded (data-quality
    indicator, not a tactical feature).

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, team_shape_xfns
        xfns = tracking_default_xfns + team_shape_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._team_shape import compute_team_shape

    vaep_metrics = [
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
    ]

    col_names = []
    for metric in vaep_metrics:
        for suffix in ("attacking", "defending"):
            col_names.append(f"team_shape_{metric}_{suffix}")

    def _team_shape_transformer(states, frames):
        """Multi-column team-shape xfn (12 cols x nb_states)."""
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        teams = frames[~frames["is_ball"].astype(bool)]["team_id"].dropna().unique()
        shape_indexed = {}
        for tid in teams:
            s = compute_team_shape(frames, team_id=tid)
            shape_indexed[tid] = s.set_index(["game_id", "period_id", "frame_id"])

        for i, slot in enumerate(states[:3]):
            slot_result = _team_shape_at_actions(slot, frames, home_team_id, shape_indexed)
            for col in col_names:
                out[f"{col}_a{i}"] = slot_result[col].to_numpy()
        return out

    _team_shape_transformer._frame_aware = True  # type: ignore[attr-defined]
    _team_shape_transformer.__name__ = "team_shape"
    return [_team_shape_transformer]


def _team_shape_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    home_team_id: int | str,
    shape_indexed: dict,
) -> pd.DataFrame:
    """Join pre-indexed team shape to actions. Internal helper for xfn.

    ``shape_indexed`` is a dict of {team_id: DataFrame} where each DataFrame
    is indexed by (game_id, period_id, frame_id) for O(1) lookup.
    """
    vaep_metrics = [
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
    ]
    col_names = []
    for metric in vaep_metrics:
        for suffix in ("attacking", "defending"):
            col_names.append(f"team_shape_{metric}_{suffix}")

    n = len(actions)
    empty = pd.DataFrame({col: np.full(n, np.nan) for col in col_names}, index=actions.index)

    if n == 0 or len(frames) == 0:
        return empty

    actions_with_idx = actions.copy()
    actions_with_idx["_row_idx"] = np.arange(n)
    pointers, _report = link_actions_to_frames(actions_with_idx, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return empty

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    linked = linked.merge(
        actions_with_idx[["action_id", "_row_idx", "team_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )
    linked = linked.drop_duplicates("_row_idx", keep="first")

    # Align game_id dtype between linked (from actions) and shape_indexed keys
    # (from frames) — int64 vs object mismatch causes silent dict-key miss.
    if len(linked) > 0 and shape_indexed:
        sample_sdf = next(iter(shape_indexed.values()))
        if len(sample_sdf) > 0:
            sample_key_gid = sample_sdf.index[0][0]
            linked_gid_sample = linked["game_id"].iloc[0]
            if not isinstance(linked_gid_sample, type(sample_key_gid)):
                linked["game_id"] = linked["game_id"].astype(str)

    out = empty.copy()

    for _, row in linked.iterrows():
        pos = int(row["_row_idx"])
        idx = actions.index[pos]
        action_team = row["team_id"]
        if pd.isna(action_team):
            continue
        key = (row["game_id"], row["period_id"], int(row["frame_id_int"]))

        for tid, sdf in shape_indexed.items():
            if key not in sdf.index:
                continue
            shape_row = sdf.loc[key]
            suffix = "attacking" if tid == action_team else "defending"
            for metric in vaep_metrics:
                out.at[idx, f"team_shape_{metric}_{suffix}"] = shape_row[metric]

    return out


# ---------------------------------------------------------------------------
# PR-S33 -- TF-32: Ward line-breaking xfns
# ---------------------------------------------------------------------------


def line_breaking_ward_xfns(home_team_id: int | str) -> list:
    """Build VAEP xfn list for TF-32 Ward line-breaking features.

    Returns a list with ONE FrameAwareTransformer that emits 3 features x 3
    game-states = 9 columns total. ``line_break__ward`` is excluded (redundant
    with ``lines_broken__ward > 0``; VAEP should not waste a parameter on a
    linearly dependent feature).

    The ``line_breaking_type__ward`` categorical is one-hot encoded:
    ``line_breaking_type__ward_between_lines`` and ``line_breaking_type__ward_around_line``.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import (
            tracking_default_xfns,
            line_breaking_ward_xfns,
        )
        xfns = tracking_default_xfns + line_breaking_ward_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._line_breaking import detect_line_breaking

    col_names = [
        "lines_broken__ward",
        "line_breaking_type__ward_between_lines",
        "line_breaking_type__ward_around_line",
    ]

    def _line_breaking_ward_transformer(states, frames):
        """Multi-column Ward line-breaking xfn (3 cols x nb_states)."""
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        for i, slot in enumerate(states[:3]):
            lb = detect_line_breaking(slot, frames, home_team_id=home_team_id)
            out[f"lines_broken__ward_a{i}"] = lb["lines_broken__ward"].to_numpy()
            out[f"line_breaking_type__ward_between_lines_a{i}"] = (
                lb["line_breaking_type__ward"] == "between_lines"
            ).to_numpy()
            out[f"line_breaking_type__ward_around_line_a{i}"] = (
                lb["line_breaking_type__ward"] == "around_line"
            ).to_numpy()
        return out

    _line_breaking_ward_transformer._frame_aware = True  # type: ignore[attr-defined]
    _line_breaking_ward_transformer.__name__ = "line_breaking_ward"
    return [_line_breaking_ward_transformer]


# ---------------------------------------------------------------------------
# PR-S31 -- TF-7: pitch control at action
# ---------------------------------------------------------------------------


def pitch_control_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    links: pd.DataFrame | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.Series:
    """Pitch control value at ball position for the acting team at the linked frame.

    Returns a Series named ``pitch_control_at_ball__<method>`` with one value per action
    in [0, 1], representing the attacking team's spatial control at the ball location
    at the moment of the action.

    NaN where action couldn't link to a frame or ball position is unavailable.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import pitch_control_at_action
    >>> pc = pitch_control_at_action(actions, frames, method="spearman")
    """
    import numpy as np

    col_name = f"pitch_control_at_ball__{method}"

    # Introspection mode: VAEP fit-time calls with frames=None
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)

    from .pitch_control import compute_pitch_control

    # Ensure velocity columns exist (fill with zero if missing)
    if "vx" not in frames.columns or "vy" not in frames.columns:
        frames = frames.copy()
        if "vx" not in frames.columns:
            frames["vx"] = 0.0
        if "vy" not in frames.columns:
            frames["vy"] = 0.0

    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)

    results = np.full(len(actions), np.nan)

    # Merge pointers with action period_id for frame lookup
    pointer_lookup = pointers.set_index("action_id")

    # Group frames by (period_id, frame_id) for efficient lookup
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        action_id = action_row["action_id"]
        if action_id not in pointer_lookup.index:
            continue

        frame_id_raw = pointer_lookup.at[action_id, "frame_id"]
        if pd.isna(frame_id_raw):
            continue

        period_id = action_row["period_id"]
        frame_id_int = int(float(frame_id_raw))  # type: ignore[arg-type]

        try:
            frame_data = frame_groups.get_group((period_id, frame_id_int))
        except KeyError:
            continue

        team_id = action_row["team_id"]

        # Compute pitch control for this frame
        surface = compute_pitch_control(frame_data, attacking_team_id=team_id, method=method)

        # Query at action start position (proxy for ball position)
        start_x = action_row["start_x"]
        start_y = action_row["start_y"]
        if np.isnan(start_x) or np.isnan(start_y):
            continue

        results[i] = surface.at_point(start_x, start_y)

    return pd.Series(results, index=actions.index, name=col_name)


@nan_safe_enrichment
def add_pitch_control(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.DataFrame:
    """Enrich actions with ``pitch_control_at_ball__<method>`` column.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_pitch_control
    >>> enriched = add_pitch_control(actions, frames)
    """
    out = actions.copy()
    s = pitch_control_at_action(actions, frames, links=links, method=method)
    out[s.name] = s.values
    return out


def pitch_control_xfns(
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> list:
    """Factory returning a list with one FrameAwareTransformer for pitch control.

    Uses the ``<feature>__<method>`` suffix-naming convention (ADR-005 section 8).

    Examples
    --------
    >>> from silly_kicks.tracking.features import pitch_control_xfns
    >>> xfns = pitch_control_xfns("spearman")
    """

    def _pc_helper(actions, frames):
        return pitch_control_at_action(actions, frames, method=method)

    _pc_helper.__name__ = f"pitch_control_at_ball__{method}"
    return [lift_to_states(_pc_helper)]


pitch_control_default_xfns = pitch_control_xfns("spearman")


# ---------------------------------------------------------------------------
# DAS — Dangerous Accessible Space (TF-28)
#
# Architecture: single-pass precomputation. get_das() runs ONCE on the full
# frames DataFrame; a (period_id, frame_id) → {team_id: DAS} lookup dict is
# built from the result; action-coupled helpers and the VAEP transformer map
# into this lookup. This avoids the 12n redundant get_das calls that would
# result from 3 separate lift_to_states helpers x 3 gamestate slots.
# ---------------------------------------------------------------------------

import warnings as _warnings  # noqa: E402


def _precompute_das_lookup(
    frames: pd.DataFrame,
) -> dict[tuple, dict]:
    """Run get_individual_das ONCE on all frames, build per-frame team-level DAS lookup.

    Uses ``get_individual_das`` (per-player DAS) and sums per team.
    ``get_das`` returns per-frame scalars that are identical for both teams,
    which would make ``das_diff`` always zero.

    Returns a dict mapping ``(period_id, frame_id)`` to ``{team_id: DAS_value}``.
    """
    from ._das import get_individual_das

    das_frames = get_individual_das(frames, use_progress_bar=False)

    player_rows = das_frames[das_frames["is_ball"] != True]  # noqa: E712
    # Filter to rows with valid DAS — accessible-space may return NaN for some
    # frames (e.g. insufficient players, off-pitch data). Without this filter,
    # the lookup stores NaN, making all action-coupled results NaN.
    valid_rows = player_rows.dropna(subset=["DAS"])
    lookup: dict[tuple, dict] = {}
    for (pid, fid, tid), grp in valid_rows.groupby(["period_id", "frame_id", "team_id"]):
        lookup.setdefault((pid, fid), {})[tid] = float(grp["DAS"].sum())
    return lookup


def _map_das_to_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    das_lookup: dict[tuple, dict],
    *,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map precomputed DAS lookup to actions. Returns 3-column DataFrame."""
    import numpy as np

    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")

    team_vals = np.full(len(actions), np.nan)
    opp_vals = np.full(len(actions), np.nan)

    for i, (_idx, row) in enumerate(actions.iterrows()):
        aid = row["action_id"]
        if aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue
        key = (row["period_id"], int(float(fid_raw)))  # type: ignore[arg-type]
        if key not in das_lookup:
            continue

        team_id = row["team_id"]
        team_vals[i] = das_lookup[key].get(team_id, np.nan)
        # Football: exactly 2 teams per frame; take the sole opponent.
        opp = [v for k, v in das_lookup[key].items() if k != team_id]
        if opp:
            opp_vals[i] = opp[0]

    return pd.DataFrame(
        {
            "das_team": team_vals,
            "das_opponent": opp_vals,
            "das_diff": team_vals - opp_vals,
        },
        index=actions.index,
    )


def das_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    col_name: str = "das_team",
) -> pd.Series:
    """Team-level DAS at the linked frame for the acting team.

    Returns a Series with one value per action. NaN where action couldn't
    link to a frame or DAS computation failed.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import das_at_action
    >>> das = das_at_action(actions, frames)
    """
    import numpy as np

    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)

    try:
        lookup = _precompute_das_lookup(frames)
    except (ValueError, RuntimeError, ImportError) as exc:
        _warnings.warn(
            f"DAS computation failed ({type(exc).__name__}: {exc}); returning NaN for all actions",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series(np.nan, index=actions.index, name=col_name)

    mapped = _map_das_to_actions(actions, frames, lookup)
    s = mapped["das_team"]
    s.name = col_name
    return s


@nan_safe_enrichment
def add_das(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich actions with ``das_team``, ``das_opponent``, ``das_diff`` columns.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_das
    >>> enriched = add_das(actions, frames)
    """
    import numpy as np

    out = actions.copy()

    try:
        lookup = _precompute_das_lookup(frames)
    except (ValueError, RuntimeError, ImportError) as exc:
        _warnings.warn(
            f"DAS computation failed ({type(exc).__name__}: {exc}); returning NaN for all DAS columns",
            UserWarning,
            stacklevel=2,
        )
        out["das_team"] = np.nan
        out["das_opponent"] = np.nan
        out["das_diff"] = np.nan
        return out

    mapped = _map_das_to_actions(actions, frames, lookup, links=links)
    out["das_team"] = mapped["das_team"].values
    out["das_opponent"] = mapped["das_opponent"].values
    out["das_diff"] = mapped["das_diff"].values
    return out


def _make_das_transformer():
    """Build a single FrameAwareTransformer that emits all 9 DAS columns.

    Single-pass: calls get_das() ONCE on the full frames DataFrame, then
    looks up per-action across all 3 gamestate slots. Returns columns:
    das_team_a0..a2, das_opponent_a0..a2, das_diff_a0..a2.
    """
    import numpy as np

    das_cols = ("das_team", "das_opponent", "das_diff")

    def das_features(states, frames):
        nb = min(len(states), 3)
        out = pd.DataFrame(index=states[0].index)

        # Empty frames → column-name probing (feature_column_names)
        if len(frames) == 0:
            for i in range(nb):
                for col in das_cols:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Precompute DAS for ALL frames — single get_das call
        try:
            lookup = _precompute_das_lookup(frames)
        except (ValueError, RuntimeError, ImportError) as exc:
            _warnings.warn(
                f"DAS computation failed ({type(exc).__name__}: {exc}); returning NaN for all DAS features",
                UserWarning,
                stacklevel=2,
            )
            for i in range(nb):
                for col in das_cols:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Map per gamestate slot
        for i, slot in enumerate(states[:nb]):
            mapped = _map_das_to_actions(slot, frames, lookup)
            for col in das_cols:
                out[f"{col}_a{i}"] = mapped[col].to_numpy()
        return out

    das_features._frame_aware = True  # type: ignore[attr-defined]
    das_features.__name__ = "das_features"
    das_features.__qualname__ = "das_features"
    return das_features


das_xfns = [_make_das_transformer()]


# ---------------------------------------------------------------------------
# TF-15 -- GK influence primitives
# ---------------------------------------------------------------------------


def _gk_influence_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    method: str = "spearman",
    zone_names: list[str] | None = None,
    tau_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch kernel: compute GK influence for all actions at once.

    Caches compute_gk_influence per unique (period_id, frame_id, team_id)
    to avoid redundant pitch control computation.

    Returns (result_df, pointers) — result_df aligned with actions.index
    containing gk_pitch_control_share_weighted, gk_reachable_area_m2,
    gk_closing_time_{min,mean}_s__<zone_name>; pointers from
    link_actions_to_frames for caller reuse.
    """
    from ._gk_influence import GkInfluence, Zone, compute_gk_influence

    _zone_names = zone_names or ["six_yard_box"]

    # Initialize output columns
    col_names = ["gk_pitch_control_share_weighted", "gk_reachable_area_m2"]
    for zn in _zone_names:
        col_names.extend(
            [
                f"gk_closing_time_min_s__{zn}",
                f"gk_closing_time_mean_s__{zn}",
            ]
        )

    result = pd.DataFrame(
        {col: np.full(len(actions), np.nan) for col in col_names},
        index=actions.index,
    )

    if len(frames) == 0:
        return result, pd.DataFrame()

    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    # Cache: (period_id, frame_id, team_id) -> GkInfluence | None
    cache: dict[tuple, GkInfluence | None] = {}

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        aid = action_row["action_id"]
        tid = action_row["team_id"]
        if pd.isna(tid):
            continue
        if aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue

        pid = action_row["period_id"]
        fid = int(float(fid_raw))  # type: ignore[arg-type]
        cache_key = (pid, fid, tid)

        if cache_key not in cache:
            try:
                frame_data = frame_groups.get_group((pid, fid))
            except KeyError:
                cache[cache_key] = None
                continue

            gk_rows = frame_data[
                frame_data["is_goalkeeper"].astype(bool)
                & (~frame_data["is_ball"].astype(bool))
                & (frame_data["team_id"] != tid)
            ]
            if gk_rows.empty:
                cache[cache_key] = None
                continue

            gk_pid = gk_rows.iloc[0]["player_id"]
            gk_team = gk_rows.iloc[0]["team_id"]
            goal_x = 0.0 if gk_team == home_team_id else 105.0

            # Resolve ball position for near/far post zones
            ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
            ball_y = float(ball_rows.iloc[0]["y"]) if not ball_rows.empty and pd.notna(ball_rows.iloc[0]["y"]) else None

            # Build Zone instances per-action with resolved goal_x + ball_y
            zones = []
            for zn in _zone_names:
                if zn == "six_yard_box":
                    zones.append(Zone.six_yard_box(goal_x))
                elif zn == "near_post":
                    zones.append(Zone.near_post(goal_x, ball_y=ball_y))
                elif zn == "far_post":
                    zones.append(Zone.far_post(goal_x, ball_y=ball_y))
                else:
                    _warnings.warn(
                        f"Unknown zone name '{zn}'; skipping",
                        UserWarning,
                        stacklevel=2,
                    )

            try:
                gi = compute_gk_influence(
                    frame_data,
                    attacking_team_id=tid,
                    gk_player_id=gk_pid,
                    xt=xt,
                    home_team_id=home_team_id,
                    method=method,  # type: ignore[arg-type]
                    zones=zones,
                    tau_seconds=tau_seconds,
                )
                cache[cache_key] = gi
            except (ValueError, KeyError) as exc:
                _warnings.warn(
                    f"compute_gk_influence failed for frame=({pid},{fid}), team={tid}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                cache[cache_key] = None

        gi = cache[cache_key]
        if gi is None:
            continue

        idx = actions.index[i]
        result.at[idx, "gk_pitch_control_share_weighted"] = gi.pitch_control_share_weighted
        result.at[idx, "gk_reachable_area_m2"] = gi.reachable_area_m2
        for zn, zct in gi.closing_times.items():
            result.at[idx, f"gk_closing_time_min_s__{zn}"] = zct.min_s
            result.at[idx, f"gk_closing_time_mean_s__{zn}"] = zct.mean_s

    return result, pointers


def gk_pitch_control_share_weighted(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
) -> pd.Series:
    """Threat-weighted GK pitch control share at the linked frame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_pitch_control_share_weighted
    >>> share = gk_pitch_control_share_weighted(actions, frames, xt, home_team_id=1)
    """
    col_name = "gk_pitch_control_share_weighted"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _gk_influence_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        method=method,
    )
    return batch[col_name].rename(col_name)


def gk_reachable_area_m2(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: str = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """GK uniquely reachable area (m^2) at the linked frame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_reachable_area_m2
    >>> area = gk_reachable_area_m2(actions, frames, xt, home_team_id=1)
    """
    col_name = "gk_reachable_area_m2"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _gk_influence_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        method=method,
        tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


def gk_closing_time_min_s(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    home_team_id: int | str,
    zone_name: str = "six_yard_box",
) -> pd.Series:
    """GK minimum closing time (seconds) to the specified zone.

    Lightweight: uses compute_zone_closing_times directly (no pitch
    control computation). See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_closing_time_min_s
    >>> ct = gk_closing_time_min_s(actions, frames, home_team_id=1)
    """
    col_name = f"gk_closing_time_min_s__{zone_name}"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    return _closing_time_per_series(
        actions,
        frames,
        home_team_id=home_team_id,
        zone_name=zone_name,
        extract="min_s",
        col_name=col_name,
    )


def gk_closing_time_mean_s(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    home_team_id: int | str,
    zone_name: str = "six_yard_box",
) -> pd.Series:
    """GK mean closing time (seconds) to the specified zone.

    Lightweight: uses compute_zone_closing_times directly (no pitch
    control computation). See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import gk_closing_time_mean_s
    >>> ct = gk_closing_time_mean_s(actions, frames, home_team_id=1)
    """
    col_name = f"gk_closing_time_mean_s__{zone_name}"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    return _closing_time_per_series(
        actions,
        frames,
        home_team_id=home_team_id,
        zone_name=zone_name,
        extract="mean_s",
        col_name=col_name,
    )


def _closing_time_per_series(
    actions,
    frames,
    *,
    home_team_id,
    zone_name,
    extract,
    col_name,
) -> pd.Series:
    """Lightweight closing-time path — calls compute_zone_closing_times directly."""
    from ._gk_influence import Zone, compute_zone_closing_times

    pointers, _ = link_actions_to_frames(actions, frames)
    results = np.full(len(actions), np.nan)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for i, (_idx, row) in enumerate(actions.iterrows()):
        aid = row["action_id"]
        tid = row["team_id"]
        if pd.isna(tid) or aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue

        pid = row["period_id"]
        fid = int(float(fid_raw))  # type: ignore[arg-type]
        try:
            frame_data = frame_groups.get_group((pid, fid))
        except KeyError:
            continue

        gk_rows = frame_data[
            frame_data["is_goalkeeper"].astype(bool)
            & (~frame_data["is_ball"].astype(bool))
            & (frame_data["team_id"] != tid)
        ]
        if gk_rows.empty:
            continue
        gk_pid = gk_rows.iloc[0]["player_id"]
        gk_team = gk_rows.iloc[0]["team_id"]
        goal_x = 0.0 if gk_team == home_team_id else 105.0

        ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
        ball_y = float(ball_rows.iloc[0]["y"]) if not ball_rows.empty and pd.notna(ball_rows.iloc[0]["y"]) else None

        if zone_name == "six_yard_box":
            zone = Zone.six_yard_box(goal_x)
        elif zone_name == "near_post":
            zone = Zone.near_post(goal_x, ball_y=ball_y)
        elif zone_name == "far_post":
            zone = Zone.far_post(goal_x, ball_y=ball_y)
        else:
            continue

        try:
            cts = compute_zone_closing_times(
                frame_data,
                gk_player_id=gk_pid,
                zones=[zone],
            )
            zct = cts.get(zone_name)
            if zct is not None:
                results[i] = getattr(zct, extract)
        except (ValueError, KeyError) as exc:
            _warnings.warn(
                f"compute_zone_closing_times failed for action_id={aid}: {exc}",
                UserWarning,
                stacklevel=2,
            )

    return pd.Series(results, index=actions.index, name=col_name)


@nan_safe_enrichment
def add_gk_influence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    method: str = "spearman",
    zone_names: list[str] | None = None,
    tau_seconds: float = 1.0,
) -> pd.DataFrame:
    """Enrich actions with GK influence columns.

    Default zone_names (["six_yard_box"]) emit 4 columns. Additional zone
    names ("near_post", "far_post") add closing-time columns. Zones are
    constructed per-action with the correct goal_x and ball_y.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_gk_influence
    >>> enriched = add_gk_influence(actions, frames, xt, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    out = actions.copy()
    batch, pointers = _gk_influence_at_actions(
        actions,
        frames,
        xt,
        links=links,
        home_team_id=home_team_id,
        method=method,
        zone_names=zone_names,
        tau_seconds=tau_seconds,
    )
    for col in batch.columns:
        out[col] = batch[col].values

    # Provenance (reuse pointers from batch kernel)
    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(
            ptr_cols,
            left_on="action_id",
            right_index=True,
            how="left",
        )

    return out


def gk_influence_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    zone_names: list[str] | None = None,
    tau_seconds: float = 1.0,
) -> list:
    """Factory returning a list with one FrameAwareTransformer for GK influence.

    Default zones (six_yard_box only): 4 columns x 3 game states = 12 VAEP columns.
    With near_post + far_post: 8 columns x 3 states = 24 columns.

    The transformer precomputes compute_gk_influence per unique
    (period_id, frame_id, team_id), avoiding redundant pitch control computation
    across 3 game-state slots and repeated actions.

    Parameters
    ----------
    xt : ExpectedThreat
        Fitted xT model for threat weighting.
    home_team_id : int | str
        Home team identifier for goal-end orientation.
    method : {"spearman", "fernandez_bornn", "voronoi"}
        Pitch control model, default "spearman".
    zone_names : list[str] | None
        Zone factory names (e.g. ["six_yard_box", "near_post"]).
        Defaults to ["six_yard_box"]. Zones are constructed per-action
        with resolved goal_x + ball_y.
    tau_seconds : float
        TTI tau parameter, default 1.0.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, gk_influence_xfns
        xfns = tracking_default_xfns + gk_influence_xfns(xt, home_team_id=1)
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from . import _gk_influence as _gk_mod

    resolved_zone_names = zone_names if zone_names is not None else ["six_yard_box"]

    col_names = [
        "gk_pitch_control_share_weighted",
        "gk_reachable_area_m2",
    ]
    for zn in resolved_zone_names:
        col_names.append(f"gk_closing_time_min_s__{zn}")
        col_names.append(f"gk_closing_time_mean_s__{zn}")

    def _gk_influence_transformer(states, frames):
        """Multi-column GK influence xfn with frame precomputation cache."""
        out = pd.DataFrame(index=states[0].index)

        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Shared cache across all 3 slots: (period_id, frame_id, team_id) -> GkInfluence
        cache: dict[tuple, _gk_mod.GkInfluence | None] = {}
        frame_groups = frames.groupby(["period_id", "frame_id"])

        def _get_gi(period_id, frame_id_int, team_id):
            key = (period_id, frame_id_int, team_id)
            if key in cache:
                return cache[key]

            try:
                frame_data = frame_groups.get_group((period_id, frame_id_int))
            except KeyError:
                cache[key] = None
                return None

            gk_rows = frame_data[
                frame_data["is_goalkeeper"].astype(bool)
                & (~frame_data["is_ball"].astype(bool))
                & (frame_data["team_id"] != team_id)
            ]
            if gk_rows.empty:
                cache[key] = None
                return None
            gk_pid = gk_rows.iloc[0]["player_id"]
            gk_team = gk_rows.iloc[0]["team_id"]
            goal_x = 0.0 if gk_team == home_team_id else 105.0

            # Resolve ball_y from frame
            ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
            ball_y = float(ball_rows.iloc[0]["y"]) if not ball_rows.empty and pd.notna(ball_rows.iloc[0]["y"]) else 34.0

            # Build zones per-action with resolved goal_x + ball_y
            action_zones = [
                getattr(_gk_mod.Zone, zn)(goal_x, ball_y=ball_y)
                if zn in ("near_post", "far_post")
                else getattr(_gk_mod.Zone, zn)(goal_x)
                for zn in resolved_zone_names
            ]

            try:
                gi = _gk_mod.compute_gk_influence(
                    frame_data,
                    attacking_team_id=team_id,
                    gk_player_id=gk_pid,
                    xt=xt,
                    home_team_id=home_team_id,
                    method=method,
                    zones=action_zones,
                    tau_seconds=tau_seconds,
                )
            except (ValueError, KeyError) as exc:
                _warnings.warn(
                    f"compute_gk_influence failed for frame {frame_id_int}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                gi = None

            cache[key] = gi
            return gi

        # C-1 fix: link_actions_to_frames per-slot (each slot has different action_ids)
        for i, slot in enumerate(states[:3]):
            slot_results = {col: np.full(len(slot), np.nan) for col in col_names}

            pointers, _ = link_actions_to_frames(slot, frames)
            pointer_lookup = pointers.set_index("action_id")

            for j, (_idx, row) in enumerate(slot.iterrows()):
                aid = row["action_id"]
                tid = row["team_id"]
                if pd.isna(tid):
                    continue
                if aid not in pointer_lookup.index:
                    continue
                fid_raw = pointer_lookup.at[aid, "frame_id"]
                if pd.isna(fid_raw):
                    continue

                pid = row["period_id"]
                fid = int(float(fid_raw))  # type: ignore[arg-type]

                gi = _get_gi(pid, fid, tid)
                if gi is None:
                    continue

                slot_results["gk_pitch_control_share_weighted"][j] = gi.pitch_control_share_weighted
                slot_results["gk_reachable_area_m2"][j] = gi.reachable_area_m2
                for zn, zct in gi.closing_times.items():
                    if f"gk_closing_time_min_s__{zn}" in slot_results:
                        slot_results[f"gk_closing_time_min_s__{zn}"][j] = zct.min_s
                        slot_results[f"gk_closing_time_mean_s__{zn}"][j] = zct.mean_s

            for col in col_names:
                out[f"{col}_a{i}"] = slot_results[col]

        return out

    _gk_influence_transformer._frame_aware = True  # type: ignore[attr-defined]
    _gk_influence_transformer.__name__ = "gk_influence"
    return [_gk_influence_transformer]


# ---------------------------------------------------------------------------
# PR-S36 -- TF-30: Cover shadows — lane control + blocking score
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_cover_shadows(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    decision_rule: Literal["any", "majority", "all"] = "majority",
    detailed: bool = False,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.DataFrame:
    """Enrich actions with cover shadow columns.

    Computes lane-specific pass obstruction and blocking score for each action.
    Emits 5 columns: n_blocked_receivers, n_potential_receivers, blocking_score,
    blocked_threat_fraction, max_single_defender_blocking_score.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with standard columns.
    frames : pd.DataFrame
        Tracking frames (LTR-normalized).
    xt : ExpectedThreat
        Fitted xT model for threat weighting.
    home_team_id : int | str
        Home team identifier (defends x=0).
    decision_rule : {"any", "majority", "all"}
        Lane-blocking decision rule. Default "majority".
    detailed : bool
        If True, compute per-defender blocking score via full pitch control
        counterfactual. If False, use lightweight lane-control approximation.
    method : str
        Pitch control method.

    Returns
    -------
    pd.DataFrame
        Input actions with 5 additional columns.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_cover_shadows
    >>> enriched = add_cover_shadows(actions, frames, xt, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    from . import _cover_shadows as _cs_mod

    out = actions.copy()
    n = len(actions)
    col_n_blocked = np.full(n, pd.NA, dtype="object")
    col_n_potential = np.full(n, pd.NA, dtype="object")
    col_bs = np.full(n, np.nan)
    col_btf = np.full(n, np.nan)
    col_max_def = np.full(n, np.nan)

    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for j, (_idx, row) in enumerate(actions.iterrows()):
        aid = row["action_id"]
        tid = row["team_id"]
        if pd.isna(tid) or aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue

        pid_period = row["period_id"]
        fid = int(float(fid_raw))  # type: ignore[arg-type]

        try:
            frame_data = frame_groups.get_group((pid_period, fid))
        except KeyError:
            continue

        passer_xy = (float(row["start_x"]), float(row["start_y"]))

        cs = _cs_mod._compute_cover_shadow_dict(
            frame_data,
            passer_xy,
            tid,
            xt,
            home_team_id=home_team_id,
            decision_rule=decision_rule,
            detailed=detailed,
            method=method,
        )
        if cs is None:
            continue

        col_n_blocked[j] = cs["n_blocked_receivers"]
        col_n_potential[j] = cs["n_potential_receivers"]
        col_bs[j] = cs["blocking_score"]
        col_btf[j] = cs["blocked_threat_fraction"]
        col_max_def[j] = cs["max_single_defender_blocking_score"]

    out["n_blocked_receivers"] = pd.array(col_n_blocked, dtype="Int64")
    out["n_potential_receivers"] = pd.array(col_n_potential, dtype="Int64")
    out["blocking_score"] = col_bs
    out["blocked_threat_fraction"] = col_btf
    out["max_single_defender_blocking_score"] = col_max_def

    # Provenance columns
    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(
            ptr_cols,
            left_on="action_id",
            right_index=True,
            how="left",
        )

    return out


def cover_shadow_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    decision_rule: Literal["any", "majority", "all"] = "majority",
    detailed: bool = False,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> list:
    """Factory returning a list with one FrameAwareTransformer for cover shadows.

    5 columns x 3 game states = 15 VAEP columns. Frame-precomputation cache
    keyed on (period_id, frame_id, team_id, rounded_passer_xy).

    Parameters
    ----------
    xt : ExpectedThreat
        Fitted xT model for threat weighting.
    home_team_id : int | str
        Home team identifier for goal-end orientation.
    decision_rule : {"any", "majority", "all"}
        Lane-blocking decision rule. Default "majority".
    detailed : bool
        If True, per-defender blocking score via full PC counterfactual.
    method : str
        Pitch control method, default "spearman".

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, cover_shadow_xfns
        xfns = tracking_default_xfns + cover_shadow_xfns(xt, home_team_id=1)
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from . import _cover_shadows as _cs_mod

    col_names = _cs_mod._CS_COL_NAMES

    def _cover_shadow_transformer(states, frames):
        """Multi-column cover shadow xfn with frame precomputation cache."""
        import warnings as _warnings

        out = pd.DataFrame(index=states[0].index)

        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        cache: dict[tuple, dict | None] = {}
        frame_groups = frames.groupby(["period_id", "frame_id"])

        def _get_cs(period_id, frame_id_int, team_id, passer_xy):
            passer_key = (round(passer_xy[0], 0), round(passer_xy[1], 0))
            key = (period_id, frame_id_int, team_id, passer_key)
            if key in cache:
                return cache[key]

            try:
                frame_data = frame_groups.get_group(
                    (period_id, frame_id_int),
                )
            except KeyError:
                cache[key] = None
                return None

            try:
                result_dict = _cs_mod._compute_cover_shadow_dict(
                    frame_data,
                    passer_xy,
                    team_id,
                    xt,
                    home_team_id=home_team_id,
                    decision_rule=decision_rule,
                    detailed=detailed,
                    method=method,
                )
                cache[key] = result_dict
                return result_dict

            except (ValueError, KeyError) as exc:
                _warnings.warn(
                    f"cover_shadow computation failed for frame {frame_id_int}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                cache[key] = None
                return None

        for i, slot in enumerate(states[:3]):
            slot_results = {col: np.full(len(slot), np.nan) for col in col_names}
            pointers, _ = link_actions_to_frames(slot, frames)
            pointer_lookup = pointers.set_index("action_id")

            for j, (_idx, row) in enumerate(slot.iterrows()):
                aid = row["action_id"]
                tid = row["team_id"]
                if pd.isna(tid):
                    continue
                if aid not in pointer_lookup.index:
                    continue
                fid_raw = pointer_lookup.at[aid, "frame_id"]
                if pd.isna(fid_raw):
                    continue

                pid = row["period_id"]
                fid = int(float(fid_raw))  # type: ignore[arg-type]
                passer_xy = (
                    float(row["start_x"]),
                    float(row["start_y"]),
                )

                cs = _get_cs(pid, fid, tid, passer_xy)
                if cs is None:
                    continue

                for col in col_names:
                    slot_results[col][j] = cs[col]

            for col in col_names:
                out[f"{col}_a{i}"] = slot_results[col]

        return out

    _cover_shadow_transformer._frame_aware = True  # type: ignore[attr-defined]
    _cover_shadow_transformer.__name__ = "cover_shadows"
    return [_cover_shadow_transformer]


# ---------------------------------------------------------------------------
# PR-S51 -- TF-36 + TF-33: Per-player influence + Off-ball xT
# ---------------------------------------------------------------------------


def _player_influence_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
    links: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch kernel: compute player influence for all actions.

    Cache key: (period_id, frame_id, attacking_team_id). Returns
    (result_df, pointers).
    """
    from ._player_influence import PlayerInfluence, compute_player_influence

    col_names = [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "off_ball_xt_diff",
        "reachable_area_team",
        "reachable_area_opponent",
        "reachable_area_diff",
    ]

    result = pd.DataFrame(
        {col: np.full(len(actions), np.nan) for col in col_names},
        index=actions.index,
    )

    if len(frames) == 0:
        return result, pd.DataFrame()

    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    # Cache: (period_id, frame_id, attacking_team_id) -> dict | None
    cache: dict[tuple, dict[int | str, PlayerInfluence] | None] = {}

    # Build player -> team_id lookup from PC surface (populated on first call)
    player_team_lookup: dict[int | str, int | str] = {}

    # Pre-compute column indices for .iat[] (list.index returns plain int)
    _cols = list(result.columns)
    _ci_actor = _cols.index("actor_reachable_area_m2")
    _ci_xt_team = _cols.index("off_ball_xt_team")
    _ci_xt_opp = _cols.index("off_ball_xt_opponent")
    _ci_xt_diff = _cols.index("off_ball_xt_diff")
    _ci_area_team = _cols.index("reachable_area_team")
    _ci_area_opp = _cols.index("reachable_area_opponent")
    _ci_area_diff = _cols.index("reachable_area_diff")

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        aid = action_row["action_id"]
        tid = action_row["team_id"]
        actor_pid = action_row["player_id"]
        if pd.isna(tid):
            continue
        if aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue

        pid = action_row["period_id"]
        fid = int(float(str(fid_raw)))
        cache_key = (pid, fid, tid)

        if cache_key not in cache:
            try:
                frame_data = frame_groups.get_group((pid, fid))
            except KeyError:
                cache[cache_key] = None
                continue

            try:
                pi_dict = compute_player_influence(
                    frame_data,
                    xt,
                    attacking_team_id=tid,
                    home_team_id=home_team_id,
                    method=method,
                    tau_seconds=tau_seconds,
                )
            except (ValueError, KeyError) as exc:
                _warnings.warn(
                    f"compute_player_influence failed for frame {fid}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                pi_dict = None

            cache[cache_key] = pi_dict

            # Populate player->team lookup from frame data
            if pi_dict is not None:
                outfield = frame_data[~frame_data["is_ball"].astype(bool) & ~frame_data["is_goalkeeper"].astype(bool)]
                for _, prow in outfield.iterrows():
                    p_id = prow["player_id"]
                    if p_id not in player_team_lookup:
                        player_team_lookup[p_id] = prow["team_id"]

        pi_dict = cache[cache_key]
        if pi_dict is None:
            continue

        # Aggregate per-team
        actor_team = tid
        team_xt = 0.0
        opponent_xt = 0.0
        actor_area = 0.0
        team_area = 0.0
        opponent_area = 0.0

        for p_id, pi in pi_dict.items():
            p_team = player_team_lookup.get(p_id)
            if p_team is None:
                continue
            is_same_team = str(p_team) == str(actor_team)
            is_actor = str(p_id) == str(actor_pid)

            if is_same_team:
                team_area += pi.reachable_area_m2
                if is_actor:
                    actor_area = pi.reachable_area_m2
                else:
                    team_xt += pi.off_ball_xt
            else:
                opponent_xt += pi.off_ball_xt
                opponent_area += pi.reachable_area_m2

        result.iat[i, _ci_actor] = actor_area
        result.iat[i, _ci_xt_team] = team_xt
        result.iat[i, _ci_xt_opp] = opponent_xt
        result.iat[i, _ci_xt_diff] = team_xt - opponent_xt
        result.iat[i, _ci_area_team] = team_area
        result.iat[i, _ci_area_opp] = opponent_area
        result.iat[i, _ci_area_diff] = team_area - opponent_area

    return result, pointers


def actor_reachable_area_m2(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """Actor's uniquely reachable area (m^2) at the linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import actor_reachable_area_m2
    >>> area = actor_reachable_area_m2(actions, frames, xt, home_team_id=1)
    """
    col_name = "actor_reachable_area_m2"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        method=method,
        tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


def off_ball_xt_team(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.Series:
    """Sum of teammates' off-ball xT (excluding actor) at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import off_ball_xt_team
    >>> val = off_ball_xt_team(actions, frames, xt, home_team_id=1)
    """
    col_name = "off_ball_xt_team"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        method=method,
    )
    return batch[col_name].rename(col_name)


def off_ball_xt_opponent(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.Series:
    """Sum of opponents' off-ball xT at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import off_ball_xt_opponent
    >>> val = off_ball_xt_opponent(actions, frames, xt, home_team_id=1)
    """
    col_name = "off_ball_xt_opponent"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        method=method,
    )
    return batch[col_name].rename(col_name)


def reachable_area_team(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """Sum of acting team's uniquely reachable area (m^2) at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import reachable_area_team
    >>> val = reachable_area_team(actions, frames, xt, home_team_id=1)
    """
    col_name = "reachable_area_team"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        method=method,
        tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


def reachable_area_opponent(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
) -> pd.Series:
    """Sum of opponent team's uniquely reachable area (m^2) at linked frame.

    For multiple columns, prefer ``add_player_influence`` which computes
    all 7 columns in a single pass.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import reachable_area_opponent
    >>> val = reachable_area_opponent(actions, frames, xt, home_team_id=1)
    """
    col_name = "reachable_area_opponent"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)
    batch, _ = _player_influence_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        method=method,
        tau_seconds=tau_seconds,
    )
    return batch[col_name].rename(col_name)


@nan_safe_enrichment
def add_player_influence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
) -> pd.DataFrame:
    """Enrich actions with 7 player-influence columns + 4 provenance.

    Columns: actor_reachable_area_m2, off_ball_xt_team, off_ball_xt_opponent,
    off_ball_xt_diff, reachable_area_team, reachable_area_opponent,
    reachable_area_diff.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_player_influence
    >>> enriched = add_player_influence(actions, frames, xt, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    out = actions.copy()
    batch, pointers = _player_influence_at_actions(
        actions,
        frames,
        xt,
        links=links,
        home_team_id=home_team_id,
        method=method,
        tau_seconds=tau_seconds,
    )
    for col in batch.columns:
        out[col] = batch[col].values

    # Provenance (idempotent skip-guard)
    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(
            ptr_cols,
            left_on="action_id",
            right_index=True,
            how="left",
        )

    return out


def player_influence_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
) -> list:
    """Factory returning a FrameAwareTransformer for player influence.

    Emits 7 columns x 3 gamestate slots = 21 VAEP columns.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, player_influence_xfns
        xfns = tracking_default_xfns + player_influence_xfns(xt, home_team_id=1)
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._player_influence import PlayerInfluence, compute_player_influence

    col_names = [
        "actor_reachable_area_m2",
        "off_ball_xt_team",
        "off_ball_xt_opponent",
        "off_ball_xt_diff",
        "reachable_area_team",
        "reachable_area_opponent",
        "reachable_area_diff",
    ]

    def _player_influence_transformer(states, frames):
        """Multi-column player influence xfn with frame precomputation cache."""
        out = pd.DataFrame(index=states[0].index)

        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Shared cache across all 3 slots
        cache: dict[tuple, dict[int | str, PlayerInfluence] | None] = {}
        frame_groups = frames.groupby(["period_id", "frame_id"])
        player_team_lookup: dict[int | str, int | str] = {}

        def _get_pi(period_id, frame_id_int, team_id):
            key = (period_id, frame_id_int, team_id)
            if key in cache:
                return cache[key]

            try:
                frame_data = frame_groups.get_group((period_id, frame_id_int))
            except KeyError:
                cache[key] = None
                return None

            try:
                pi_dict = compute_player_influence(
                    frame_data,
                    xt,
                    attacking_team_id=team_id,
                    home_team_id=home_team_id,
                    method=method,
                    tau_seconds=tau_seconds,
                )
            except (ValueError, KeyError) as exc:
                _warnings.warn(
                    f"compute_player_influence failed for frame {frame_id_int}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                pi_dict = None

            cache[key] = pi_dict

            # Populate player->team lookup
            if pi_dict is not None:
                outfield = frame_data[~frame_data["is_ball"].astype(bool) & ~frame_data["is_goalkeeper"].astype(bool)]
                for _, prow in outfield.iterrows():
                    p_id = prow["player_id"]
                    if p_id not in player_team_lookup:
                        player_team_lookup[p_id] = prow["team_id"]

            return pi_dict

        def _aggregate(pi_dict, actor_team, actor_pid):
            """Aggregate per-player values into 7-column dict."""
            vals = {col: np.nan for col in col_names}
            if pi_dict is None:
                return vals

            team_xt = 0.0
            opp_xt = 0.0
            actor_area = 0.0
            team_area = 0.0
            opp_area = 0.0

            for p_id, pi in pi_dict.items():
                p_team = player_team_lookup.get(p_id)
                if p_team is None:
                    continue
                is_same = str(p_team) == str(actor_team)
                is_actor = str(p_id) == str(actor_pid)

                if is_same:
                    team_area += pi.reachable_area_m2
                    if is_actor:
                        actor_area = pi.reachable_area_m2
                    else:
                        team_xt += pi.off_ball_xt
                else:
                    opp_xt += pi.off_ball_xt
                    opp_area += pi.reachable_area_m2

            vals["actor_reachable_area_m2"] = actor_area
            vals["off_ball_xt_team"] = team_xt
            vals["off_ball_xt_opponent"] = opp_xt
            vals["off_ball_xt_diff"] = team_xt - opp_xt
            vals["reachable_area_team"] = team_area
            vals["reachable_area_opponent"] = opp_area
            vals["reachable_area_diff"] = team_area - opp_area
            return vals

        for i, slot in enumerate(states[:3]):
            slot_results = {col: np.full(len(slot), np.nan) for col in col_names}

            pointers, _ = link_actions_to_frames(slot, frames)
            pointer_lookup = pointers.set_index("action_id")

            for j, (_idx, row) in enumerate(slot.iterrows()):
                aid = row["action_id"]
                tid = row["team_id"]
                if pd.isna(tid):
                    continue
                if aid not in pointer_lookup.index:
                    continue
                fid_raw = pointer_lookup.at[aid, "frame_id"]
                if pd.isna(fid_raw):
                    continue

                period = row["period_id"]
                fid = int(float(str(fid_raw)))
                actor_pid = row["player_id"]

                pi_dict = _get_pi(period, fid, tid)
                agg = _aggregate(pi_dict, tid, actor_pid)
                for col in col_names:
                    slot_results[col][j] = agg[col]

            for col in col_names:
                out[f"{col}_a{i}"] = slot_results[col]

        return out

    _player_influence_transformer._frame_aware = True  # type: ignore[attr-defined]
    _player_influence_transformer.__name__ = "player_influence"
    return [_player_influence_transformer]
