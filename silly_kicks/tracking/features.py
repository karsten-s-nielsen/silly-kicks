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
- acting_gk_from_frames(actions, frames) -> pd.Series          (PR-S106; mirror + identity fallback)
- defensive_line_x / back_line_high_x / compactness_x / lateral_width /
  max_lateral_gap / back_n_count (actions, frames, *, home_team_id) (PR-S27, TF-14)
- add_defensive_line(actions, frames, *, home_team_id) -> pd.DataFrame  (PR-S27)
- defensive_line_xfns(home_team_id) -> list                    (PR-S27)
- pitch_control_at_target(actions, frames) -> pd.Series        (PR-S31, TF-7)
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

    from ._gk_completion import GkCompletionModel
    from ._line_breaking import LineBreakingParams
    from .pitch_control import PitchControlCache

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import align_join_keys, ids_differ, ids_match, restore_id_dtype, same_id
from silly_kicks.spadl import config as spadlconfig

from . import _kernels
from ._action_orientation import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    acting_team_attacks_rtl,
    reproject_to_action_ltr,
)
from ._ball_carrier import infer_ball_carrier
from ._das import (
    DAS_SOURCE_COMPUTED,
    DAS_SOURCE_TEAM_UNRESOLVED,
    DAS_SOURCE_UNLINKED,
    DAS_SOURCE_UNSCOREABLE_CALL,
    DAS_SOURCE_UNSCOREABLE_FRAME,
    DasUnscoreableError,
)
from ._gk_resolve import (
    acting_gk_from_frames,
    defended_goal_x,
    defending_gk_from_frames,
    gk_distribution_mask,
)
from ._packing import PackingParams, secured_reception
from ._shot_goalmouth import ShotGoalmouthParams, compute_shot_goalmouth
from ._structural_pass import StructuralPassParams
from ._warnings import IgnoredSurfaceInputsWarning, SyntheticEPVWarning
from ._xcross_attempt import xcross_attempt_xfns
from ._xshot_occurrence import xshot_occurrence_xfns
from ._xt_gk import XtGkParams, compute_xt_gk
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
    "acting_gk_from_frames",
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
    "add_elastic_sync",
    "add_ghost_gk",
    "add_gk_completion",
    "add_gk_influence",
    "add_line_break",
    "add_obso",
    "add_off_ball_context",
    "add_off_ball_run_values",
    "add_off_ball_runs",
    "add_packing",
    "add_pausa",
    "add_pitch_control",
    "add_player_influence",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_pressure_on_actor",
    "add_shape_graph",
    "add_shot_goalmouth",
    "add_space_creation",
    "add_team_shape",
    "back_line_high_x",
    "back_n_count",
    "ball_carrier_at_action",
    "compactness_x",
    "cover_shadow_xfns",
    "das_at_action",
    "das_xfns",
    "defended_goal_x",
    "defenders_in_triangle_to_goal",
    "defending_gk_from_frames",
    "defensive_line_x",
    "defensive_line_xfns",
    "elastic_sync_xfns",
    "ghost_gk_xfns",
    "gk_closing_time_mean_s",
    "gk_closing_time_min_s",
    "gk_distribution_mask",
    "gk_influence_xfns",
    "gk_pitch_control_share_weighted",
    "gk_reachable_area_m2",
    "lateral_width",
    "line_breaking_ward_xfns",
    "max_lateral_gap",
    "nearest_defender_distance",
    "obso_actual",
    "obso_optimal",
    "obso_peak",
    "obso_xfns",
    "off_ball_context_xfns",
    "off_ball_run_value_xfns",
    "off_ball_xt_opponent",
    "off_ball_xt_team",
    "packing_xfns",
    "pausa_xfns",
    "pitch_control_at_target",
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
    "shape_graph_xfns",
    "shot_crossing_y",
    "shot_crossing_z",
    "shot_on_target_derived",
    "shot_speed",
    "shot_time_to_goal_line",
    "space_creation_xfns",
    "team_shape_xfns",
    "tracking_default_xfns",
]


def ball_carrier_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
    tolerance_m: float = 3.0,
    beta: float = 0.0,
    gamma: float = 0.25,
    pre: dict | None = None,
    links: pd.DataFrame | None = None,
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
    beta : float, default 0.0
        Velocity weight passed to ``infer_ball_carrier`` (Optuna-calibrated, TF-24).
    gamma : float, default 0.25
        Hysteresis bonus passed to ``infer_ball_carrier`` (Optuna-calibrated, TF-24).
    pre : dict, optional
        Precomputed ``_pre_index_frames(frames)``, threaded to
        ``infer_ball_carrier`` to skip re-marshalling the frames. Both ``pre``
        and ``links`` are independent of the carrier-scoring params, so a caller
        re-resolving carriers on the same frames with different params (the TF-24
        sweep) can compute them once and reuse — bit-identical to recomputing.
    links : pd.DataFrame, optional
        Precomputed ``link_actions_to_frames(actions, frames)`` pointers. When
        supplied, the internal linking is skipped and these pointers are used
        (mirrors the ``links`` kwarg on the ``add_*`` aggregators).

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

    # Compute per-frame carriers (reusing a cached pre-index when supplied).
    carriers = infer_ball_carrier(frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma, pre=pre)
    if carriers.empty:
        return out

    # Link actions to frames (reuse caller-supplied pointers when provided —
    # linking is independent of the carrier-scoring params).
    if links is not None:
        pointers = links
    else:
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

    # Restore the frames dtype (shared rule -- see restore_id_dtype).
    out = restore_id_dtype(out, pid_dtype)

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


# PR-S80: xS (GKDV Layer 2) joins the GK/shot-context union ONLY -- NOT the general
# tracking_default_xfns (which stays model-free; adding a frame-time bundled-weights +
# [xgboost] dependency to the broad default would be a Hyrum break). Bundled model load is
# memoized (from_variant("default") -> _VARIANT_CACHE).
# PR-B: xCross (TF-17, GKDV Layer 2) joins the same GK-union list as xS (NOT the general default).
# Wired in regardless of the model's tf19_ready flag -- a VAEP feature factory exposes inputs to a
# learner; the column is not asserted "validated useful" (an inert-GK model still ships its column).
pre_shot_gk_full_default_xfns = (
    pre_shot_gk_default_xfns + pre_shot_gk_angle_default_xfns + xshot_occurrence_xfns() + xcross_attempt_xfns()
)


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
    out = merged[["action_id", "x", "y", "vx", "vy"]]

    # ADR-045 D2: the ball never entered ActionFrameContext, so _reproject_rows never
    # saw it -- it reached _pressure_bekkers in FRAME coordinates while the defenders
    # were in action-LTR. Position AND velocity both need re-projecting here.
    #
    # Read the flip from ctx (Task 5 put it there); do NOT recompute
    # acting_team_attacks_rtl. One orientation decision, one place -- otherwise the ball
    # and the players agree only by coincidence, which is the drift this PR removes.
    row_flip = out["action_id"].map(ctx.flip_by_action).fillna(False).astype(bool)
    row_flip.index = out.index
    return reproject_to_action_ltr(out, row_flip, x_cols=["x"], y_cols=["y"], vx_cols=["vx"], vy_cols=["vy"])


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
    # Dup-action_id safety (ADR): VAEP shifted gamestate slots repeat the boundary
    # action, so action_id is non-unique. The pressure kernels are action_id-indexed
    # (would raise on the dup). Re-key to a unique positional surrogate for the compute;
    # the output Series stays indexed by the (unique) frame index, so nothing leaks. Only
    # on the internal-link path (links is None) -- caller-supplied links own the ids.
    if links is None and actions["action_id"].duplicated().any():
        actions = actions.copy()
        actions["action_id"] = np.arange(len(actions))
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
        # No whole-batch ball-row guard: when ball rows are missing (entirely or per
        # action), _pressure_bekkers falls back per-action to the base model
        # (pressure-on-player only). ball-carrier-max is an improvement, not a
        # requirement (Bekkers 2024 section 2.4). (3.30.0)
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


@nan_safe_enrichment
def add_structural_pass(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    links: pd.DataFrame | None = None,
    params: StructuralPassParams | None = None,
) -> pd.DataFrame:
    """Append structural_lbs / structural_sgm / structural_sdi (NA for non-pass/cross).

    Per-pass structural primitives (TF-45, arXiv:2603.28916). Idempotent provenance
    columns. Accepts caller-supplied ``links`` (skips internal link_actions_to_frames).
    The body implements the NaN-safety contract; @nan_safe_enrichment + the CI gate
    only verify it.

    See NOTICE for full bibliographic citations.
    """
    batch = _kernels._structural_pass_at_actions(actions, frames, home_team_id=home_team_id, params=params, links=links)
    out = actions.copy()
    # House Int64 idiom (float Series w/ NaN -> Int64 <NA>) preserves the LBS=0-vs-NaN
    # distinction.
    out["structural_lbs"] = batch["structural_lbs"].astype("Int64")
    out["structural_sgm"] = batch["structural_sgm"].to_numpy()
    out["structural_sdi"] = batch["structural_sdi"].to_numpy()

    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing:
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            ptr_cols = pointers.set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out


def structural_pass_xfns(
    *,
    home_team_id: int | str,
    params: StructuralPassParams | None = None,
) -> list:
    """VAEP xfn factory: ONE FrameAwareTransformer emitting structural_lbs/sgm/sdi x 3
    gamestate slots = 9 columns (structural_lbs_a0 .. structural_sdi_a2). Calls the
    SHARED _kernels._structural_pass_at_actions once per slot (3x, not 9x).

    See NOTICE for full bibliographic citations.
    """
    col_names = ["structural_lbs", "structural_sgm", "structural_sdi"]

    def _structural_pass_transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(min(3, len(states))):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out
        for i, slot in enumerate(states[:3]):
            batch = _kernels._structural_pass_at_actions(slot, frames, home_team_id=home_team_id, params=params)
            for col in col_names:
                out[f"{col}_a{i}"] = batch[col].to_numpy()
        return out

    _structural_pass_transformer._frame_aware = True  # type: ignore[attr-defined]
    _structural_pass_transformer.__name__ = "structural_pass"
    return [_structural_pass_transformer]


@nan_safe_enrichment
def add_packing(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    links: pd.DataFrame | None = None,
    params: PackingParams | None = None,
) -> pd.DataFrame:
    """Append the five TF-49 packing columns (Impect-faithful bypass counts).

    Adds, per action (NaN/<NA> off-domain -- domain = ``params.action_types``
    AND result success):

    - ``packing_made`` (Int64) -- canon Impect count: eligible opponent
      defenders with ``start_x < d_x <= end_x`` (outfield-only by default;
      ``params.include_gk`` adds the keeper). Receiver double-credit is a
      consumer-side groupby on ``packing_receiver_player_id``.
    - ``packing_net`` (float64) -- direction-multiplied signed count over the
      ``(min_x, max_x]`` interval: +1 forward (theta <= ``forward_max_deg``),
      ``side_multiplier`` sideways, ``back_multiplier`` backward
      (football-packing multipliers).
    - ``packing_goal_threat`` (Int64) -- forward count restricted to the
      defending team's ``select_back_line_players`` back-``back_line_n``
      (MSC "goal threat packing").
    - ``packing_receiver_player_id`` -- next same-team touch's player_id
      (source dtype passthrough, never float64-upcast; ADR-019). <NA> for
      dribbles (no reception in packing semantics), off-domain rows,
      opponent-next, period end, or NaN source ids.
    - ``packing_secured`` (boolean) -- 'ball stays past the line'
      (:func:`silly_kicks.tracking.secured_reception`). <NA> unless a receiver
      resolved AND ``packing_made >= 1``.

    ``params.require_secured=True`` gates the three numeric columns on
    ``packing_secured`` for RECEIVER-BEARING types only (secured False -> 0,
    secured <NA> -> NaN); dribbles keep their raw counts (secured reception is
    a pass concept -- TF-49 review F3), and ``packing_made == 0`` rows keep
    their honest 0.

    Caveats: ``take_on`` (SPADL 7) is excluded -- a point event cannot express
    bypass under this inequality, so ``dribble`` does NOT cover 1v1s. The
    completion gate inherits each provider's result semantics (SkillCorner
    ``result_source`` tiers, sportec eval-allowlist) -- cross-provider
    ``packing_made`` sums are not provider-comparable without that caveat.
    Gradient Sports dribbles before silly-kicks 4.49.0 shipped placeholder
    ends (``start == end``) and are honestly NaN (spec s5.6); post-fix only
    period-last carries remain placeholders.

    Idempotent provenance columns; accepts caller-supplied ``links``. Returns
    a NEW frame (ADR-033). See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking import add_packing
    >>> out = add_packing(actions, frames, home_team_id=1)
    >>> out[["packing_made", "packing_net", "packing_goal_threat"]].describe()
    """
    if params is None:
        params = PackingParams()
    from silly_kicks.spadl.utils import _resolve_next_touch_positions, resolve_next_touch_receiver

    batch = _kernels._packing_at_actions(actions, frames, home_team_id=home_team_id, params=params, links=links)

    # Event-only assembly (TF-49 review major 7): receiver + secured live HERE, not in
    # the kernel -- packing_xfns calls the kernel on shifted gamestate slots where
    # next-row relationships are meaningless.
    receiver_pos = _resolve_next_touch_positions(actions)
    receiver = resolve_next_touch_receiver(actions, positions=receiver_pos)

    domain_ids = frozenset(spadlconfig.actiontype_id[n] for n in params.action_types)
    dribble_id = spadlconfig.actiontype_id["dribble"]
    success_id = spadlconfig.result_id["success"]
    type_arr = actions["type_id"].to_numpy()
    result_arr = actions["result_id"].to_numpy()
    receiver_bearing = np.isin(type_arr, list(domain_ids - {dribble_id})) & (result_arr == success_id)
    receiver = receiver.where(pd.Series(receiver_bearing, index=receiver.index))

    secured = secured_reception(actions, batch["line_x"], receiver_pos, params=params)
    made_ge1 = batch["packing_made"].to_numpy() >= 1  # NaN compares False
    secured = secured.where(pd.Series(receiver.notna().to_numpy() & made_ge1, index=secured.index))

    made = batch["packing_made"].copy()
    net = batch["packing_net"].copy()
    goal_threat = batch["packing_goal_threat"].copy()
    if params.require_secured:
        gate_rows = receiver_bearing & made_ge1
        secured_false = (~secured.astype("boolean")).fillna(False).to_numpy(dtype=bool)
        zero_rows = gate_rows & secured_false
        na_rows = gate_rows & secured.isna().to_numpy()
        for col in (made, net, goal_threat):
            col[zero_rows] = 0.0
            col[na_rows] = np.nan

    out = actions.copy()
    # House Int64 idiom preserves the 0-vs-NaN distinction on the count columns.
    out["packing_made"] = made.astype("Int64")
    out["packing_net"] = net.to_numpy()
    out["packing_goal_threat"] = goal_threat.astype("Int64")
    out["packing_receiver_player_id"] = receiver
    out["packing_secured"] = secured
    # line_x is kernel-internal (feeds secured); deliberately NOT an output column.

    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing:
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            ptr_cols = pointers.set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out


def packing_xfns(
    *,
    home_team_id: int | str,
    params: PackingParams | None = None,
) -> list:
    """VAEP xfn factory: ONE FrameAwareTransformer emitting packing_made /
    packing_net / packing_goal_threat x 3 gamestate slots = 9 columns. Calls the
    SHARED _kernels._packing_at_actions once per slot (3x, not 9x). Receiver id +
    secured are provenance, excluded (the _COORD_COLS precedent).

    ``params.require_secured=True`` raises ``ValueError``: receiver/secured
    resolution needs true next-row relationships, which shifted gamestate slots
    do not have -- secured gating is aggregator-only (ADR-039).

    .. warning::

        Result-leakage (TF-49 review F4, ADR-039): every packing column gates on
        the action's OWN ``result_id`` -- as an a0-slot feature that is exactly
        the result-leakage class HybridVAEP exists to strip, and the TF-48
        auto-discovering guard covers DEFAULT lists only (opt-in factories
        bypass it). ``packing_xfns`` MUST NOT enter HybridVAEP-class consumers
        without a0 exclusion. A result-free a0 variant is a recorded fork in
        ADR-039, not built.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking import packing_xfns
    >>> xfns = packing_xfns(home_team_id=1)
    >>> len(xfns)
    1
    """
    if params is not None and params.require_secured:
        raise ValueError(
            "packing_xfns: params.require_secured=True is not supported -- receiver/secured "
            "resolution needs true next-row relationships, which shifted gamestate slots do "
            "not have. Secured gating is aggregator-only (use add_packing). See ADR-039."
        )
    col_names = ["packing_made", "packing_net", "packing_goal_threat"]

    def _packing_transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(min(3, len(states))):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out
        for i, slot in enumerate(states[:3]):
            batch = _kernels._packing_at_actions(slot, frames, home_team_id=home_team_id, params=params)
            for col in col_names:
                out[f"{col}_a{i}"] = batch[col].to_numpy()
        return out

    _packing_transformer._frame_aware = True  # type: ignore[attr-defined]
    _packing_transformer.__name__ = "packing"
    return [_packing_transformer]


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

    Adds (all ``float64``; NaN when the pre-window has no linked frames):

    - ``n_off_ball_runners_pre_window`` — count of attacking teammates making
      an off-ball run (displacement > ``min_displacement_m``) in the pre-window.
    - ``max_off_ball_run_displacement_pre_window`` — largest such displacement (m).
    - ``mean_off_ball_run_speed_pre_window`` — mean runner speed (m/s).
    - ``n_off_ball_runners_toward_goal_pre_window`` — count whose run is
      goalward (toward the attacked goal).

    For all 6 off-ball-run + line-break columns at once use
    :func:`add_off_ball_context`.

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

    Emits the 4 off-ball-run columns of :func:`add_off_ball_runs`
    (``n_off_ball_runners_pre_window``, ``max_off_ball_run_displacement_pre_window``,
    ``mean_off_ball_run_speed_pre_window``, ``n_off_ball_runners_toward_goal_pre_window``
    — all ``float64``) plus the 2 threshold line-break columns of
    :func:`add_line_break`:

    - ``line_break`` (``bool``) — the action's pass/carry crosses the defending
      team's ``defensive_line_x``.
    - ``n_attackers_behind_line`` (``Int64``) — attackers beyond that line.

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
# PR-S119 -- TF-35: off-ball run valuation
# ---------------------------------------------------------------------------

#: The five wide columns emitted by :func:`add_off_ball_run_values`. Numeric-only
#: subset (``_RUN_VALUE_XFN_COLS``) feeds the VAEP factory.
_RUN_VALUE_COLS = [
    "run_value_target",
    "n_disruptive_runs",
    "run_value_disruptive_sum",
    "n_valued_disruptive_runs",
    "run_value_enabled_pass",
]

#: ``n_valued_disruptive_runs`` is EXCLUDED from the VAEP surface: it is a coverage
#: denominator (how many disruptive runs were observable), not a football quantity, and
#: as a model feature it would let tracking-visibility stand in for play quality.
_RUN_VALUE_XFN_COLS = [
    "run_value_target",
    "n_disruptive_runs",
    "run_value_disruptive_sum",
    "run_value_enabled_pass",
]


def _run_values_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt,
    *,
    home_team_id: int | str,
    links: pd.DataFrame | None = None,
    pitch_control_cache=None,
    params=None,
) -> pd.DataFrame:
    """Shared TF-35 kernel: detect -> value -> collapse to one row per action.

    Positional output aligned to ``actions`` (never keyed on ``action_id``), so the
    shifted gamestate slots that repeat a boundary action stay safe -- ADR-020.
    """
    from ._run_values import RunValuationParams, action_level_context, detect_off_ball_runs, value_off_ball_runs

    if params is None:
        params = RunValuationParams()

    n = len(actions)
    target = np.full(n, np.nan)
    n_disruptive = np.full(n, np.nan)
    disruptive_sum = np.full(n, np.nan)
    n_valued = np.full(n, np.nan)
    enabled = np.full(n, np.nan)

    if n == 0:
        return pd.DataFrame(
            {
                "run_value_target": target,
                "n_disruptive_runs": n_disruptive,
                "run_value_disruptive_sum": disruptive_sum,
                "n_valued_disruptive_runs": n_valued,
                "run_value_enabled_pass": enabled,
            },
            index=actions.index,
        )

    _receiver, on_domain, credit = action_level_context(actions, xt)
    # On-domain actions start at an honest 0 ("this pass had no qualifying runs"),
    # NOT NA ("we could not tell"). Off-domain rows stay NA on all five.
    target[on_domain] = 0.0
    n_disruptive[on_domain] = 0.0
    disruptive_sum[on_domain] = 0.0
    n_valued[on_domain] = 0.0
    enabled[on_domain] = credit[on_domain]

    runs = detect_off_ball_runs(actions, frames, home_team_id=home_team_id, params=params)
    if len(runs) == 0:
        return pd.DataFrame(
            {
                "run_value_target": target,
                "n_disruptive_runs": n_disruptive,
                "run_value_disruptive_sum": disruptive_sum,
                "n_valued_disruptive_runs": n_valued,
                "run_value_enabled_pass": enabled,
            },
            index=actions.index,
        )

    valued = value_off_ball_runs(
        runs,
        actions,
        frames,
        xt,
        links=links,
        pitch_control_cache=pitch_control_cache,
        params=params,
    )

    # Positional collapse keyed on (game_id, action_id): SPADL action_id restarts per
    # game, so an action_id-only key folds another game's runs into this one -- inflating
    # n_disruptive_runs and run_value_disruptive_sum (ADR-042 review finding 2).
    pos_by_key: dict = {}
    for pos, key in enumerate(zip(actions["game_id"].to_numpy(), actions["action_id"].to_numpy(), strict=True)):
        pos_by_key.setdefault(key, []).append(pos)

    for key, grp in valued.groupby(["game_id", "action_id"], sort=False):
        positions = pos_by_key.get(tuple(key))
        if not positions:
            continue
        # NA-safe: `role` is a nullable "string" column and is <NA> for every run whose
        # action is off-domain, so a bare `== "target"` yields a BooleanArray WITH NA and
        # `.to_numpy(dtype=bool)` raises. Reached only when a detected run belongs to an
        # off-domain action -- which the standard fixtures never produced and the atomic
        # mirror did.
        has_role = grp["role"].notna().to_numpy()
        is_target = (grp["role"] == "target").fillna(False).to_numpy(dtype=bool)
        tgt_vals = grp.loc[is_target, "run_value"].to_numpy(dtype="float64")
        dis_vals = grp.loc[has_role & ~is_target, "run_value"].to_numpy(dtype="float64")
        n_dis = len(dis_vals)
        n_dis_valued = int(np.isfinite(dis_vals).sum())
        for pos in positions:
            if not on_domain[pos]:
                continue
            if len(tgt_vals):
                # A detected-but-unvalued target run is NaN, never a fabricated 0.
                target[pos] = float(tgt_vals[0])
            n_disruptive[pos] = n_dis
            n_valued[pos] = n_dis_valued
            disruptive_sum[pos] = float(np.nansum(dis_vals)) if n_dis else 0.0

    return pd.DataFrame(
        {
            "run_value_target": target,
            "n_disruptive_runs": n_disruptive,
            "run_value_disruptive_sum": disruptive_sum,
            "n_valued_disruptive_runs": n_valued,
            "run_value_enabled_pass": enabled,
        },
        index=actions.index,
    )


@nan_safe_enrichment
def add_off_ball_run_values(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt,
    *,
    home_team_id: int | str,
    links: pd.DataFrame | None = None,
    pitch_control_cache=None,
    params=None,
) -> pd.DataFrame:
    """Append the five TF-35 off-ball-run-value columns (ADR-042).

    Adds, per action (all five ``<NA>``/NaN off-domain -- domain is a completed
    ``pass``/``cross`` whose next same-team touch resolves to a receiver):

    - ``run_value_target`` (float64) -- value of the RECEIVER's own off-ball run.
      ``0.0`` when the receiver made no qualifying run; NaN when they made one that
      could not be valued (a tracking-visibility gap).
    - ``n_disruptive_runs`` (Int64) -- qualifying runs by non-receiving teammates.
    - ``run_value_disruptive_sum`` (float64) -- sum of those runs' values, skipping
      unvalued ones.
    - ``n_valued_disruptive_runs`` (Int64) -- how many of them actually carried a
      value.
    - ``run_value_enabled_pass`` (float64) -- the pass's own floored xT gain, once
      per action, i.e. the threat the disruptive runs helped unlock.

    .. warning::

        **Mean disruptive value must divide by ``n_valued_disruptive_runs``, never by
        ``n_disruptive_runs``** (R2-3). The sum skips unvalued runs, so the second
        denominator silently deflates the mean in exact proportion to how poor the
        provider's tracking coverage was -- turning a data-quality gradient into an
        apparent tactical one.

    .. warning::

        Result-leakage (ADR-042, inheriting TF-49 review F4): the domain gates on the
        action's OWN ``result_id``. As an a0-slot feature that is the leakage class
        ``HybridVAEP`` exists to strip, so :func:`off_ball_run_value_xfns` MUST NOT
        enter a HybridVAEP-class consumer without a0 exclusion.

    Provider bias: run detection uses the frames' ``speed`` column when present and
    falls back to displacement/duration otherwise, which UNDER-states peak speed --
    see ``peak_speed_source`` on :func:`silly_kicks.tracking.detect_off_ball_runs`
    for the per-run record of which was used.

    Idempotent provenance columns; accepts caller-supplied ``links`` and
    ``pitch_control_cache``. Returns a NEW frame (ADR-033). See NOTICE for full
    bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking import add_off_ball_run_values
    >>> out = add_off_ball_run_values(actions, frames, xt, home_team_id=1)
    >>> out[["run_value_target", "n_disruptive_runs"]].head()
    """
    batch = _run_values_at_actions(
        actions,
        frames,
        xt,
        home_team_id=home_team_id,
        links=links,
        pitch_control_cache=pitch_control_cache,
        params=params,
    )

    out = actions.copy()
    out["run_value_target"] = batch["run_value_target"].to_numpy()
    out["run_value_disruptive_sum"] = batch["run_value_disruptive_sum"].to_numpy()
    out["run_value_enabled_pass"] = batch["run_value_enabled_pass"].to_numpy()
    # House Int64 idiom preserves the 0-vs-<NA> distinction on the count columns.
    for col in ("n_disruptive_runs", "n_valued_disruptive_runs"):
        out[col] = pd.array(
            [pd.NA if not np.isfinite(v) else int(v) for v in batch[col].to_numpy(dtype="float64")],
            dtype="Int64",
        )

    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            # drop_duplicates before the merge (ADR-020, matching add_obso / add_space_creation):
            # a non-unique action_id in the pointers FANS OUT the merge, returning more rows
            # than the caller passed in.
            ptr_cols = pointers.drop_duplicates("action_id").set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out


def off_ball_run_value_xfns(
    xt,
    *,
    home_team_id: int | str,
    params=None,
) -> list:
    """VAEP xfn factory: ONE FrameAwareTransformer emitting 4 numeric TF-35 columns
    x 3 gamestate slots = 12 columns. ``n_valued_disruptive_runs`` is excluded (it is
    a coverage denominator, not a football quantity -- see
    :func:`add_off_ball_run_values`).

    The ``xt`` model is validated at FACTORY time, so an unfitted grid fails when the
    xfn list is built rather than deep inside a VAEP fit.

    .. note::

        **Role assignment is exact on the a0 slot and approximate on a1/a2.** TF-35 splits
        runs into ``target`` (the receiver's own run) and ``disruptive`` using the next
        same-team touch. ``gamestates`` builds slot ``i`` by shifting the stream and filling
        the leading ``i`` rows with a REPEAT of the first action, so on those fill rows the
        "next touch" resolves to the action's own actor -- who is excluded from run
        candidacy, making every run there read as ``disruptive``. Away from the boundary the
        shifted rows still hold consecutive actions, so resolution is correct. The effect is
        bounded by ``i`` rows per game per slot. This is the same shifted-slot limitation
        that makes ``packing_xfns`` refuse ``require_secured`` (ADR-039); TF-35 degrades
        rather than refuses because the numeric columns remain meaningful.

    .. warning::

        Result-leakage (ADR-042): the TF-35 domain gates on the action's own
        ``result_id``, so this factory must not enter a HybridVAEP-class consumer
        without a0 exclusion. It is deliberately in NO default xfn list.

    Examples
    --------
    >>> from silly_kicks.tracking import off_ball_run_value_xfns
    >>> xfns = off_ball_run_value_xfns(xt, home_team_id=1)
    >>> len(xfns)
    1
    """
    from silly_kicks.xthreat import require_fitted_xt

    require_fitted_xt(xt, caller="off_ball_run_value_xfns")

    def _off_ball_run_value_transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(min(3, len(states))):
                for col in _RUN_VALUE_XFN_COLS:
                    out[f"{col}_a{i}"] = np.nan
            return out
        for i, slot in enumerate(states[:3]):
            batch = _run_values_at_actions(slot, frames, xt, home_team_id=home_team_id, params=params)
            for col in _RUN_VALUE_XFN_COLS:
                out[f"{col}_a{i}"] = batch[col].to_numpy(dtype="float64")
        return out

    _off_ball_run_value_transformer._frame_aware = True  # type: ignore[attr-defined]
    _off_ball_run_value_transformer.__name__ = "off_ball_run_values"
    return [_off_ball_run_value_transformer]


# ---------------------------------------------------------------------------
# PR-S33 -- TF-31: team shape envelope
# ---------------------------------------------------------------------------

# ADR-028: team-shape position columns to re-project into action-LTR (both teams).
# centroid_x / defensive_line_height are x-positions; centroid_y is a y-position.
# convex_hull_area / team_length / team_width / stretch_index / inter_line_gap_* /
# n_outfield_players are spans / counts / areas -> flip-invariant.
_TEAM_SHAPE_X_COLS = [
    "team_shape_centroid_x_attacking",
    "team_shape_centroid_x_defending",
    "team_shape_defensive_line_height_attacking",
    "team_shape_defensive_line_height_defending",
]
_TEAM_SHAPE_Y_COLS = [
    "team_shape_centroid_y_attacking",
    "team_shape_centroid_y_defending",
]


def _reproject_team_shape(out: pd.DataFrame, actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Mirror team-shape centroid / line-height positions into each action's LTR frame (ADR-028)."""
    flip = acting_team_attacks_rtl(actions, frames)
    return reproject_to_action_ltr(out, flip, x_cols=_TEAM_SHAPE_X_COLS, y_cols=_TEAM_SHAPE_Y_COLS)


@nan_safe_enrichment
def add_team_shape(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Enrich actions with 20 team-shape columns (10 metrics x 2 teams).

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
                "defensive_line_height",
                "inter_line_gap_1",
                "inter_line_gap_2",
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
        "defensive_line_height",
        "inter_line_gap_1",
        "inter_line_gap_2",
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
            suffix = "attacking" if same_id(tid, action_team) else "defending"
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
    out = _reproject_team_shape(out, actions, frames)  # ADR-028
    return out


def team_shape_xfns(home_team_id: int | str) -> list:
    """Build VAEP xfn list for TF-31/TF-44 team shape features.

    Returns a list with ONE FrameAwareTransformer that emits 18 features x 3
    game-states = 54 columns total. ``n_outfield_players`` is excluded (data-quality
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
        "defensive_line_height",
        "inter_line_gap_1",
        "inter_line_gap_2",
    ]

    col_names = []
    for metric in vaep_metrics:
        for suffix in ("attacking", "defending"):
            col_names.append(f"team_shape_{metric}_{suffix}")

    def _team_shape_transformer(states, frames):
        """Multi-column team-shape xfn (18 cols x nb_states)."""
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
        "defensive_line_height",
        "inter_line_gap_1",
        "inter_line_gap_2",
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
            suffix = "attacking" if same_id(tid, action_team) else "defending"
            for metric in vaep_metrics:
                out.at[idx, f"team_shape_{metric}_{suffix}"] = shape_row[metric]

    out = _reproject_team_shape(out, actions, frames)  # ADR-028
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


def pitch_control_at_target(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    links: pd.DataFrame | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.Series:
    """Pitch control value at the action DESTINATION for the acting team at the linked frame.

    Returns a Series named ``pitch_control_at_target__<method>`` with one value per action
    in [0, 1], the acting team's spatial control at the action's destination ``(end_x, end_y)``
    at the moment of the action. Sampling the destination (not the ball) makes the value live:
    ball-travel-time to the destination is positive, so players can contest it — whereas at the
    ball the Spearman PPCF is the degenerate ~0.5 reaction-time fallback (ADR-032 retired the old
    ``pitch_control_at_ball__<method>``). The query is re-projected into the frame's absolute-frame
    convention so away-team actions sample the correct cell (ADR-028).

    Interpretation varies by action type (a uniform raw feature): for passes/crosses/carries it is
    open-play control of the destination; for shots the destination is the shot target (GK/defender-
    dominated), so it reads as target-cell contestation. A model conditions on this via ``type_id``.

    NaN where the action couldn't link to a frame, or ``end_x``/``end_y`` is unavailable. In-place
    actions (``end ≈ start ≈ ball``) read ~0.5 by construction (no spatial destination — honest).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import pitch_control_at_target
    >>> pc = pitch_control_at_target(actions, frames, method="spearman")
    """
    import numpy as np

    col_name = f"pitch_control_at_target__{method}"

    # Introspection mode: VAEP fit-time calls with frames=None
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)

    from .pitch_control import PitchControlCache

    # One cache across all actions (TF-7 shared surface); a caller-supplied
    # cache extends reuse across feature families in a single pass.
    cache = pitch_control_cache if pitch_control_cache is not None else PitchControlCache()

    # Ensure velocity columns exist (fill with zero if missing)
    if "vx" not in frames.columns or "vy" not in frames.columns:
        frames = frames.copy()
        if "vx" not in frames.columns:
            frames["vx"] = 0.0
        if "vy" not in frames.columns:
            frames["vy"] = 0.0

    results = np.full(len(actions), np.nan)

    from ._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr

    # Re-aim from the ball (degenerate ~0.5) to the action DESTINATION, re-projected into the frame's
    # absolute-frame convention so away-team actions sample the correct cell (ADR-028 / ADR-032). The
    # cached per-frame surface stays absolute-frame (cache key unchanged) -- only the query point flips.
    #
    # INVOLUTION (do NOT "simplify" this): reproject_to_action_ltr is the 180-degree reflection
    # (x->105-x, y->68-y) on rtl rows; the reflection is its own inverse. The action's end point is in
    # action-LTR; applying the SAME per-action flip lands it on the absolute-frame cell the surface is
    # keyed in. Applying a "to_ltr"-named helper to ltr points to obtain ABSOLUTE reads backwards but is
    # correct -- the away ground-truth + multi-action tests go RED if anyone replaces this with a no-op.
    _flip = acting_team_attacks_rtl(actions, frames)
    _q = actions[["end_x", "end_y"]].rename(columns={"end_x": "_qx", "end_y": "_qy"}).copy()
    _q = reproject_to_action_ltr(_q, _flip, x_cols=["_qx"], y_cols=["_qy"])
    qx = _q["_qx"].to_numpy(dtype="float64")  # positional order == actions row order (== loop order below)
    qy = _q["_qy"].to_numpy(dtype="float64")

    # Dup-action_id-safe frame-id resolution by position (ADR: frame-aware xfns resolve
    # frame_id by position, never .at on a non-unique action_id). Equivalent to the old
    # .at lookup on unique action_ids (locked by test_resolve_frame_ids_by_position), so
    # the add_pitch_control aggregator path is unchanged.
    fid_by_pos = _kernels.resolve_frame_ids_by_position(actions, frames, links=links)

    # Group frames by (period_id, frame_id) for efficient lookup
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        if np.isnan(fid_by_pos[i]):
            continue

        period_id = int(action_row["period_id"])
        frame_id_int = int(fid_by_pos[i])

        try:
            frame_data = frame_groups.get_group((period_id, frame_id_int))
        except KeyError:
            continue

        team_id = action_row["team_id"]

        # Compute pitch control for this frame (canonical-frame surface, cached)
        surface = cache.surface(frame_data, team_id, method=method)

        # Query at the action destination (end), re-projected into the frame's absolute convention.
        if np.isnan(qx[i]) or np.isnan(qy[i]):
            continue

        results[i] = surface.at_point(float(qx[i]), float(qy[i]))

    return pd.Series(results, index=actions.index, name=col_name)


@nan_safe_enrichment
def add_pitch_control(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.DataFrame:
    """Enrich actions with the ``pitch_control_at_target__<method>`` column (ADR-032).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_pitch_control
    >>> enriched = add_pitch_control(actions, frames)
    """
    out = actions.copy()
    s = pitch_control_at_target(actions, frames, links=links, method=method, pitch_control_cache=pitch_control_cache)
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
        return pitch_control_at_target(actions, frames, method=method)

    _pc_helper.__name__ = f"pitch_control_at_target__{method}"
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


def _describe_das_group(keys: list[str], values) -> str:
    """Human-readable ``(game_id=.., period_id=..)`` label for a groupby key."""
    if not keys:
        return "(all frames)"
    if not isinstance(values, tuple):
        values = (values,)
    return "(" + ", ".join(f"{k}={v}" for k, v in zip(keys, values, strict=False)) + ")"


def _validate_per_frame_attacking_direction(
    frames: pd.DataFrame,
    col: str,
    *,
    link_frame_ids: set | None = None,
) -> None:
    """Fail-loud validation of a caller-supplied per-frame attacking-direction column.

    Contract (TF-28 ``add_das`` passthrough, Option A): when ``attacking_direction_col``
    is supplied it must hold ONE numeric (+1/-1) value per
    ``(game_id, period_id, frame_id)`` — the in-possession team's attacking
    direction — for every action-linked frame. silly-kicks does not interpret
    ``team_in_possession`` or any string convention; the caller owns the
    per-team→per-frame reduction. Violations raise (never a NaN fallback):

    * column missing                         -> ``ValueError``
    * non-numeric dtype (e.g. ``"ltr"``)     -> ``TypeError``
    * a ``(game_id, period_id)`` group all-NaN -> ``ValueError`` (names the group)
    * a group partially populated            -> ``ValueError`` (names group + frames)

    When ``link_frame_ids`` is given, only those frames are validated — unlinked
    frames are never simulated, so their direction is irrelevant (a dead-ball
    frame the caller left NaN must not trip validation).
    """
    if col not in frames.columns:
        raise ValueError(f"add_das: attacking_direction_col='{col}' not found in frames columns {list(frames.columns)}")
    if not pd.api.types.is_numeric_dtype(frames[col]):
        raise TypeError(
            f"add_das: attacking_direction_col='{col}' must be numeric (+1/-1 per frame); "
            f"got dtype '{frames[col].dtype}'. Map string labels (e.g. 'ltr'/'rtl') to ±1 first."
        )

    scope = frames
    if link_frame_ids is not None and "frame_id" in frames.columns:
        scope = frames[frames["frame_id"].isin(link_frame_ids)]
    if scope.empty:
        return

    group_keys = [k for k in ("game_id", "period_id") if k in scope.columns]
    groups = scope.groupby(group_keys, dropna=False) if group_keys else [((), scope)]

    for gkey, grp in groups:
        per_frame_count = grp.groupby("frame_id")[col].count()
        has_value = per_frame_count > 0
        if not bool(has_value.any()):
            raise ValueError(
                f"add_das: attacking_direction_col='{col}' is all-NaN for group "
                f"{_describe_das_group(group_keys, gkey)}; expected a numeric direction per frame."
            )
        if not bool(has_value.all()):
            missing = sorted(int(f) for f in per_frame_count.index[~has_value.to_numpy()])
            raise ValueError(
                f"add_das: attacking_direction_col='{col}' is partially populated for group "
                f"{_describe_das_group(group_keys, gkey)}: frames {missing} have no value. "
                "Populate the direction for every linked frame (no partial coverage)."
            )


def _precompute_das_lookup(
    frames: pd.DataFrame,
    *,
    chunk_size: int | None = None,
    link_frame_ids: set | None = None,
    attacking_direction_col: str | None = None,
) -> dict[tuple, dict]:
    """Run get_individual_das ONCE on all frames, build per-frame team-level DAS lookup.

    Uses ``get_individual_das`` (per-player DAS) and sums per team.
    ``get_das`` returns per-frame scalars that are identical for both teams,
    which would make ``das_diff`` always zero.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames.
    chunk_size : int or None, default None
        When set, passed through to ``accessible-space`` to process frames
        in chunks of this size. Useful for memory-constrained environments
        (e.g. Databricks ``applyInPandas`` with 1 GB group memory cap).
    link_frame_ids : set or None, default None
        When provided, restrict the (expensive) per-frame simulation to these
        action-linked ``frame_id``s — per-frame DAS is a snapshot, so a linked
        frame's value is independent of which other frames are present. The
        attacking direction is pinned on the FULL frames first (via
        ``_pin_attacking_direction``) so the restricted subset keeps the
        full-period sign, making the result bit-identical to the unrestricted
        computation. When None, all frames are simulated (direction inferred).
    attacking_direction_col : str or None, default None
        When supplied, the column on ``frames`` holding a caller-precomputed
        per-frame numeric (+1/-1) attacking direction (the in-possession team's
        direction). ``_pin_attacking_direction`` is skipped entirely — useful
        when the caller already knows the direction and the per-frame inference
        would assert or mis-infer (e.g. a dead-ball window with no non-NaN
        ``team_in_possession``). The column is validated (see
        ``_validate_per_frame_attacking_direction``) and threaded to
        ``get_individual_das``; the library's possession gate is untouched.
        Mutually exclusive with the ``_pin`` path: when given, it takes over
        regardless of ``link_frame_ids``.

    Returns a dict mapping ``(period_id, frame_id)`` to ``{team_id: DAS_value}``.
    """
    from ._das import get_individual_das

    kwargs: dict = {"use_progress_bar": False}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size

    if attacking_direction_col is not None:
        # Caller supplied a per-frame numeric direction. Restrict to the linked
        # frames (per-frame DAS is a snapshot), validate, and bypass _pin —
        # whose infer_playing_direction asserts on all-NaN team_in_possession.
        # The library's possession gate (which NaN-fills empty-possession
        # frames) is left untouched.
        if link_frame_ids is not None:
            frames = frames[frames["frame_id"].isin(link_frame_ids)]
        _validate_per_frame_attacking_direction(frames, attacking_direction_col)
        kwargs["attacking_direction_col"] = attacking_direction_col
    elif link_frame_ids is not None:
        from ._das import _pin_attacking_direction

        frames = _pin_attacking_direction(frames)
        frames = frames[frames["frame_id"].isin(link_frame_ids)]
        kwargs["attacking_direction_col"] = "attacking_direction"

    # Cross-repo: the lakehouse runs _fill_possession_from_set_piece_actions (possession
    # back-fill for set-piece restarts) BEFORE add_das, so this guard correctly fires only
    # when the link-restricted subset is still all-NaN AFTER that fill -- genuine dead-ball
    # (e.g. IDSSE ~33% dead frames), not a fillable set-piece gap. `frames` here is the
    # link-restricted subset (both branches above ran frames[frame_id.isin(link_frame_ids)]),
    # so this surfaces silly-kicks' clear message instead of accessible-space's generic
    # ValueError; add_das catches it and NaN-degrades.
    if (
        link_frame_ids is not None
        and "team_in_possession" in frames.columns
        and not frames["team_in_possession"].notna().any()
    ):
        msg = (
            "team_in_possession is all-NaN in the link-restricted frame subset (dead-ball "
            "window): DAS is undefined here. add_das degrades these actions to NaN "
            f"(das_source={DAS_SOURCE_UNSCOREABLE_CALL!r})."
        )
        raise DasUnscoreableError(msg)

    das_frames = get_individual_das(frames, **kwargs)

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
    """Map precomputed DAS lookup to actions. Returns a 4-column DataFrame.

    The 4th column is the ``das_source`` provenance (ADR-043): a NaN DAS value alone
    cannot say WHY it is NaN, so each row is classified against the closed
    ``DAS_SOURCE_VALUES`` vocabulary. ``unscoreable_call`` is never produced here --
    that token belongs to the caller's degrade path, which never reaches this mapper.

    Frame ids resolve POSITIONALLY via ``_kernels.resolve_frame_ids_by_position``
    (ADR-020), never ``.at`` on an ``action_id`` index: VAEP shifted gamestate slots
    repeat the boundary action, so that index is legitimately non-unique and ``.at``
    returns a Series ("truth value of a Series is ambiguous"). ``add_das``'s own path
    passes unique ids, where the two are byte-equivalent.
    """
    import numpy as np

    frame_ids = _kernels.resolve_frame_ids_by_position(actions, frames, links=links)

    team_vals = np.full(len(actions), np.nan)
    opp_vals = np.full(len(actions), np.nan)
    sources = np.full(len(actions), DAS_SOURCE_UNLINKED, dtype=object)

    for i, (_idx, row) in enumerate(actions.iterrows()):
        fid_raw = frame_ids[i]
        if np.isnan(fid_raw):
            continue
        key = (row["period_id"], int(fid_raw))
        if key not in das_lookup:
            # The action DID link; the frame carries no DAS (accessible-space returned
            # NaN there, or the _has_simulatable_frame guard NaN-ed the whole subset).
            sources[i] = DAS_SOURCE_UNSCOREABLE_FRAME
            continue

        team_id = row["team_id"]
        # Dtype-safe team match (ADR-019): das_lookup keys are frame-derived (team_in_possession),
        # team_id is action-derived -- canonical match instead of raw dict .get / != .
        matched = next((v for k, v in das_lookup[key].items() if same_id(k, team_id)), np.nan)
        team_vals[i] = matched
        # Football: exactly 2 teams per frame; take the sole opponent.
        opp = [v for k, v in das_lookup[key].items() if not same_id(k, team_id)]
        if opp:
            opp_vals[i] = opp[0]
        # das_lookup values are finite by construction (_precompute_das_lookup drops NaN
        # DAS rows before summing), so a NaN here means no lookup team matched the actor.
        sources[i] = DAS_SOURCE_COMPUTED if not pd.isna(matched) else DAS_SOURCE_TEAM_UNRESOLVED

    return pd.DataFrame(
        {
            "das_team": team_vals,
            "das_opponent": opp_vals,
            "das_diff": team_vals - opp_vals,
            "das_source": sources,
        },
        index=actions.index,
    )


def das_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    col_name: str = "das_team",
    chunk_size: int | None = None,
) -> pd.Series:
    """Team-level DAS at the linked frame for the acting team.

    Returns a Series with one value per action. NaN where the action couldn't link to a
    frame, where the linked frame carries no DAS, or where DAS was unscoreable for the
    whole call (:class:`~silly_kicks.tracking.DasUnscoreableError`).

    A bare Series carries no provenance, so those three NaN causes are indistinguishable
    here. Use :func:`add_das`, whose ``das_source`` column names the cause per row
    (ADR-043). Only ``DasUnscoreableError`` degrades; every other exception propagates.

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
        lookup = _precompute_das_lookup(frames, chunk_size=chunk_size)
    except DasUnscoreableError as exc:
        _warnings.warn(
            f"DAS is unscoreable for these frames ({exc}); returning NaN for all actions",
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
    chunk_size: int | None = None,
    attacking_direction_col: str | None = None,
) -> pd.DataFrame:
    """Enrich actions with ``das_team``, ``das_opponent``, ``das_diff``, ``das_source``.

    ``das_source`` is the provenance column (ADR-043). A NaN DAS value on its own cannot
    say WHY it is NaN, so every row is tagged with one of the closed
    :data:`~silly_kicks.tracking.DAS_SOURCE_VALUES` tokens:

    ``computed``
        A DAS value was resolved for the acting team at the linked frame.
    ``unlinked``
        The action resolved to no frame -- DAS is legitimately absent here.
    ``unscoreable_frame``
        The FRAMES are why DAS is absent. Either per-action (the action linked to a frame
        that carries no DAS -- accessible-space returned NaN at that instant, e.g. NaN
        ``team_in_possession``, or the ball/player-disjoint subset guard fired), or
        whole-call (every frame declares
        :data:`~silly_kicks.tracking.SPEED_SOURCE_UNAVAILABLE`, so the ``vx``/``vy`` DAS
        needs can never exist for this source -- the ``snapshot_to_tracking_frames``
        freeze-frame shape). An UNMARKED frame set missing ``vx``/``vy`` still RAISES:
        that is a forgotten ``derive_velocities()``, not a property of the data.
    ``team_unresolved``
        The linked frame carries DAS, but none of its teams matched the acting team.
    ``unscoreable_call``
        The DAS computation degraded with
        :class:`~silly_kicks.tracking.DasUnscoreableError` on frames that could in
        principle have carried DAS (dead-ball window, degenerate Voronoi, NaN
        coordinates); no action in this call has DAS. Together with
        ``unscoreable_frame`` these are the only degrades -- every other exception
        propagates.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions.
    frames : pd.DataFrame
        Long-form tracking frames.
    links : pd.DataFrame or None, default None
        Pre-computed action-frame link pointers.
    chunk_size : int or None, default None
        When set, passed through to ``accessible-space`` to process frames
        in chunks. Useful for memory-constrained environments (e.g.
        Databricks ``applyInPandas`` UDFs with 1 GB group memory cap).
    attacking_direction_col : str or None, default None
        When supplied, the column on ``frames`` holding a caller-precomputed
        per-frame **numeric** (+1/-1) attacking direction — one value per
        ``(game_id, period_id, frame_id)``, the in-possession team's direction.
        silly-kicks validates it (exists / numeric / fully covered per group,
        restricted to action-linked frames), then skips ``_pin_attacking_direction``
        and threads it straight to ``accessible-space``. Use this when the
        direction is already known and per-frame inference would assert or
        mis-infer — e.g. a dead-ball window with no non-NaN ``team_in_possession``
        (``_pin``'s ``infer_playing_direction`` asserts there). A misconfigured
        column fails loud (``ValueError``/``TypeError``); it is **not** degraded
        to NaN. The library's possession gate is unchanged: frames whose
        ``team_in_possession`` is NaN still yield NaN DAS. When None, behavior is
        bit-identical to before (direction inferred via ``_pin``).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_das
    >>> enriched = add_das(actions, frames)
    >>> # caller-supplied per-frame numeric direction (skips inference):
    >>> enriched = add_das(actions, frames, attacking_direction_col="attacking_direction")
    """
    import numpy as np

    out = actions.copy()

    # When links are supplied, restrict the per-frame simulation to the linked
    # frames (per-frame DAS is a snapshot; direction is pinned on full frames
    # first, so the result is bit-identical — see _precompute_das_lookup).
    link_frame_ids: set | None = None
    if links is not None and "frame_id" in links.columns:
        link_frame_ids = set(links["frame_id"].dropna().astype(int).tolist())

    # Fail loud on a misconfigured direction column BEFORE the try below (which
    # degrades library/runtime failures to NaN). A bad column is a caller
    # contract violation, not a runtime DAS failure — it must propagate.
    if attacking_direction_col is not None:
        _validate_per_frame_attacking_direction(frames, attacking_direction_col, link_frame_ids=link_frame_ids)

    try:
        lookup = _precompute_das_lookup(
            frames,
            chunk_size=chunk_size,
            link_frame_ids=link_frame_ids,
            attacking_direction_col=attacking_direction_col,
        )
    except DasUnscoreableError as exc:
        _warnings.warn(
            f"DAS is unscoreable for these frames ({exc}); returning NaN for all DAS columns",
            UserWarning,
            stacklevel=2,
        )
        out["das_team"] = np.nan
        out["das_opponent"] = np.nan
        out["das_diff"] = np.nan
        # The raiser names its own provenance (default ``unscoreable_call``); a frame source
        # that structurally cannot carry velocity reports ``unscoreable_frame`` instead.
        out["das_source"] = exc.das_source
        return out

    mapped = _map_das_to_actions(actions, frames, lookup, links=links)
    out["das_team"] = mapped["das_team"].values
    out["das_opponent"] = mapped["das_opponent"].values
    out["das_diff"] = mapped["das_diff"].values
    out["das_source"] = mapped["das_source"].values
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
        except DasUnscoreableError as exc:
            _warnings.warn(
                f"DAS is unscoreable for these frames ({exc}); returning NaN for all DAS features",
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
    pitch_control_cache: PitchControlCache | None = None,
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
                & (~ids_match(frame_data["team_id"], tid))
            ]
            if gk_rows.empty:
                cache[cache_key] = None
                continue

            gk_pid = gk_rows.iloc[0]["player_id"]
            gk_team = gk_rows.iloc[0]["team_id"]
            goal_x = 0.0 if same_id(gk_team, home_team_id) else 105.0

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
                    pitch_control_cache=pitch_control_cache,
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
            & (~ids_match(frame_data["team_id"], tid))
        ]
        if gk_rows.empty:
            continue
        gk_pid = gk_rows.iloc[0]["player_id"]
        gk_team = gk_rows.iloc[0]["team_id"]
        goal_x = 0.0 if same_id(gk_team, home_team_id) else 105.0

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
    pitch_control_cache: PitchControlCache | None = None,
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
        pitch_control_cache=pitch_control_cache,
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
                & (~ids_match(frame_data["team_id"], team_id))
            ]
            if gk_rows.empty:
                cache[key] = None
                return None
            gk_pid = gk_rows.iloc[0]["player_id"]
            gk_team = gk_rows.iloc[0]["team_id"]
            goal_x = 0.0 if same_id(gk_team, home_team_id) else 105.0

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

            # Dup-action_id-safe frame-id resolution by position (ADR: frame-aware xfns
            # resolve frame_id by position, never .at on a non-unique action_id).
            fid_by_pos = _kernels.resolve_frame_ids_by_position(slot, frames)

            for j, (_idx, row) in enumerate(slot.iterrows()):
                tid = row["team_id"]
                if pd.isna(tid) or np.isnan(fid_by_pos[j]):
                    continue

                pid = int(row["period_id"])
                fid = int(fid_by_pos[j])

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
    pitch_control_cache: PitchControlCache | None = None,
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
    from .pitch_control import PitchControlCache as _PitchControlCache

    # One cache across all actions + the per-defender counterfactual loop so the
    # canonical surface for a frame is computed once (TF-7 shared surface). A
    # caller-supplied cache extends reuse across feature families.
    cache = pitch_control_cache if pitch_control_cache is not None else _PitchControlCache()

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
            pitch_control_cache=cache,
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
            # Dup-action_id-safe frame-id resolution by position (ADR: frame-aware xfns
            # resolve frame_id by position, never .at on a non-unique action_id).
            fid_by_pos = _kernels.resolve_frame_ids_by_position(slot, frames)

            for j, (_idx, row) in enumerate(slot.iterrows()):
                tid = row["team_id"]
                if pd.isna(tid) or np.isnan(fid_by_pos[j]):
                    continue

                pid = int(row["period_id"])
                fid = int(fid_by_pos[j])
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
    pitch_control_cache: PitchControlCache | None = None,
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
                    pitch_control_cache=pitch_control_cache,
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
            is_same_team = same_id(p_team, actor_team)
            is_actor = same_id(p_id, actor_pid)

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
    pitch_control_cache: PitchControlCache | None = None,
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
        pitch_control_cache=pitch_control_cache,
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
                is_same = same_id(p_team, actor_team)
                is_actor = same_id(p_id, actor_pid)

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

            # Dup-action_id-safe frame-id resolution by position (ADR: frame-aware xfns
            # resolve frame_id by position, never .at on a non-unique action_id).
            fid_by_pos = _kernels.resolve_frame_ids_by_position(slot, frames)

            for j, (_idx, row) in enumerate(slot.iterrows()):
                tid = row["team_id"]
                if pd.isna(tid) or np.isnan(fid_by_pos[j]):
                    continue

                period = int(row["period_id"])
                fid = int(fid_by_pos[j])
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


# ---------------------------------------------------------------------------
# Ghost-GK positioning (TF-18, GKDV Layer 2)
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_ghost_gk(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model=None,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    actions_for_context: pd.DataFrame | None = None,
    carrier: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich actions with ghost-GK positioning columns.

    Adds ghost_gk_x, ghost_gk_y per action (defending GK's ghost position at the
    linked frame).

    Only adds ghost-GK columns — does NOT add link provenance columns.
    Callers wanting provenance should call link_actions_to_frames directly.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions.
    frames : pd.DataFrame
        Tracking frames (LTR-normalized).
    model : GhostGkModel | "default" | "full" | None
        ``"default"`` / ``None``: bundled lightweight model (~12 MB).
        ``"full"``: high-resolution model (~170 MB, downloaded from HF Hub).
        Or a pre-loaded ``GhostGkModel`` instance.
    links : pd.DataFrame | None
        Pre-computed link pointers.
    home_team_id : int | str
        Home team ID.
    actions_for_context : pd.DataFrame | None
        SPADL actions for score_diff and phase context resolution.
        If None, context defaults to 0 (backward-compatible).
    carrier : pd.DataFrame | None
        Optional precomputed carrier forwarded to ``compute_ghost_gk`` to avoid
        recomputing possession (see its docstring; mirrors ``links``).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_ghost_gk
    >>> enriched = add_ghost_gk(actions, frames, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    from ._ghost_gk import _resolve_model, compute_ghost_gk

    resolved_model = _resolve_model(model)
    out = actions.copy()

    # Link actions to frames
    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)

    # PR-S66: restrict the per-frame KDE to the frames these actions link to.
    # add_ghost_gk always has pointers (supplied or internally computed), so the
    # restriction applies regardless of source; the per-frame ghost is internal
    # and the action mapping reads only linked frames, so unrestricted frames
    # staying NaN changes no consumed value. Bit-identical (see compute_ghost_gk).
    link_frame_ids: set[int] | None = None
    if "frame_id" in pointers.columns:
        link_frame_ids = set(pointers["frame_id"].dropna().astype(int).tolist())

    # Short-circuit: skip compute if frames already have ghost columns
    if "ghost_gk_x" in frames.columns and frames["ghost_gk_x"].notna().any():
        ghost_frames = frames
    else:
        ghost_frames = compute_ghost_gk(
            frames,
            model=resolved_model,
            home_team_id=home_team_id,
            actions=actions_for_context,
            carrier=carrier,
            link_frame_ids=link_frame_ids,
        )

    # Extract ghost predictions from GK rows
    gk_ghost = ghost_frames[
        ghost_frames["is_goalkeeper"].astype(bool)
        & ~ghost_frames["is_ball"].astype(bool)
        & ghost_frames["ghost_gk_x"].notna()
    ][["game_id", "period_id", "frame_id", "team_id", "ghost_gk_x", "ghost_gk_y"]].copy()

    # Build linked lookup
    linked = pointers.merge(
        actions[["action_id", "game_id", "period_id", "team_id"]],
        on="action_id",
    )

    # Align id-valued join keys before the merge so a string-id caller does not raise (ADR-019;
    # replaces the prior ad-hoc game_id.astype(str) hand-patch).
    linked, gk_ghost = align_join_keys(linked, gk_ghost, ["game_id", "period_id", "frame_id"])

    # Merge: find defending GK (opposite team from action's team)
    merged = linked.merge(
        gk_ghost,
        on=["game_id", "period_id", "frame_id"],
        how="left",
        suffixes=("_action", "_gk"),
    )
    # Defending GK = opposite team (ids_differ requires both present: an unmatched left-join row
    # with NaN team_id_gk is correctly excluded, not treated as opposite).
    defending = merged[ids_differ(merged["team_id_action"], merged["team_id_gk"])]
    deduped = defending.drop_duplicates(subset=["action_id"], keep="first")

    # Join back to actions
    ghost_cols = deduped.set_index("action_id")[["ghost_gk_x", "ghost_gk_y"]]
    out = out.merge(ghost_cols, left_on="action_id", right_index=True, how="left")

    # ADR-028: emit in action-LTR. The model serves goal-relative coords (defended goal
    # at x=0, y kept in absolute-frame terms). In action-LTR the defended goal is x=105,
    # so x -> 105 - gr_x UNIFORMLY (gr_x already measures from the defended goal). y mirrors
    # only for away-team actions (the per-action 180-degree reflection). density_spread is
    # a dispersion magnitude -> invariant.
    flip = acting_team_attacks_rtl(actions, frames).reindex(out.index, fill_value=False).to_numpy(dtype=bool)
    gx = out["ghost_gk_x"].to_numpy(dtype="float64")
    gy = out["ghost_gk_y"].to_numpy(dtype="float64")
    out["ghost_gk_x"] = FIELD_LENGTH - gx
    out["ghost_gk_y"] = np.where(flip, FIELD_WIDTH - gy, gy)

    return out


def ghost_gk_xfns(
    *,
    model=None,
    home_team_id: int | str,
    carrier: pd.DataFrame | None = None,
) -> list:
    """Factory returning a FrameAwareTransformer for ghost-GK features.

    2 columns x 3 game states = 6 VAEP columns.

    Parameters
    ----------
    carrier : pd.DataFrame | None
        Optional precomputed carrier forwarded to ``compute_ghost_gk`` to avoid
        recomputing possession (see its docstring; mirrors ``links``).

    Examples
    --------
    >>> from silly_kicks.tracking.features import ghost_gk_xfns
    >>> xfns = ghost_gk_xfns(home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    # 2 metric columns (density_spread retired, spec 2026-07-20 §3.1); the range(3)
    # below is the gamestate-SLOT loop (a0/a1/a2), NOT the column count.
    col_names = ["ghost_gk_x", "ghost_gk_y"]

    def _ghost_gk_transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)

        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        from ._ghost_gk import _resolve_model, compute_ghost_gk

        resolved = _resolve_model(model)

        # PR-S66: link each gamestate slot once and restrict the single
        # compute_ghost_gk to the UNION of their linked frames. The union ⊇ every
        # slot's linked set, the KDE is byte-identical per sample, and each
        # per-slot add_ghost_gk reads only its own linked frames (union extras
        # stay NaN, unread). Reusing pointers as `links` avoids re-linking.
        slot_pointers: list[pd.DataFrame] = []
        link_frame_ids: set[int] = set()
        for slot in states[:3]:
            pointers, _ = link_actions_to_frames(slot, frames)
            slot_pointers.append(pointers)
            if "frame_id" in pointers.columns:
                link_frame_ids |= set(pointers["frame_id"].dropna().astype(int).tolist())

        ghost_frames = compute_ghost_gk(
            frames,
            model=resolved,
            home_team_id=home_team_id,
            carrier=carrier,
            link_frame_ids=link_frame_ids,
        )

        for i, (slot, pointers) in enumerate(zip(states[:3], slot_pointers, strict=False)):
            enriched = add_ghost_gk(
                slot,
                ghost_frames,
                model=resolved,
                home_team_id=home_team_id,
                links=pointers,
            )
            for col in col_names:
                out[f"{col}_a{i}"] = enriched[col].values if col in enriched.columns else np.nan

        return out

    _ghost_gk_transformer._frame_aware = True  # type: ignore[attr-defined]
    _ghost_gk_transformer.__name__ = "ghost_gk_xfn"
    return [_ghost_gk_transformer]


# ---------------------------------------------------------------------------
# PR-S57 -- TF-39: shape graph (Sotudeh 2026)
# ---------------------------------------------------------------------------

_SHAPE_GRAPH_METRICS = ("density", "n_edges", "mean_stability")


@nan_safe_enrichment
def add_shape_graph(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Enrich actions with 6 shape-graph columns (3 metrics x 2 teams).

    Metrics per team (attacking / defending):
    - ``shape_graph_density``: n_edges / max_possible_edges (float 0-1)
    - ``shape_graph_n_edges``: number of stable edges (int)
    - ``shape_graph_mean_stability``: mean angular stability in degrees

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_shape_graph
    >>> # See tests/tracking/test_shape_graph.py for runnable examples.
    """
    from ._shape_graph import compute_shape_graph

    out = actions.copy()
    col_names = []
    for metric in _SHAPE_GRAPH_METRICS:
        for suffix in ("attacking", "defending"):
            col_names.append(f"shape_graph_{metric}_{suffix}")
    for col in col_names:
        out[col] = np.nan

    teams = frames[~frames["is_ball"].astype(bool)]["team_id"].dropna().unique()
    if len(teams) < 2:
        return out

    # When links are supplied, the per-frame shape graph is a snapshot, so only
    # the action-linked frames are needed — restrict the (expensive) per-frame
    # loop to them. The metric depends solely on a single frame's positions, so
    # this is bit-identical. Gate on links is not None so the no-links path is
    # unchanged for other consumers.
    restrict_frame_ids: set | None = None
    if links is not None and "frame_id" in links.columns:
        restrict_frame_ids = set(links["frame_id"].dropna().astype("int64").tolist())

    # Pre-compute shape graph metrics indexed by (game_id, period_id, frame_id, team_id)
    sg_indexed: dict = {}
    for tid in teams:
        team_frames = frames[
            (frames["team_id"] == tid)
            & (~frames["is_ball"].astype(bool))
            & (~frames["is_goalkeeper"].astype(bool))
            & frames["x"].notna()
            & frames["y"].notna()
        ]
        if restrict_frame_ids is not None:
            team_frames = team_frames[team_frames["frame_id"].isin(restrict_frame_ids)]
        if team_frames.empty:
            continue
        frame_metrics: list[dict] = []
        for (gid, pid, fid), grp in team_frames.groupby(["game_id", "period_id", "frame_id"], dropna=False):
            positions = grp[["x", "y"]].to_numpy(dtype="float64")
            n = len(positions)
            if n < 3:
                frame_metrics.append(
                    {
                        "game_id": gid,
                        "period_id": pid,
                        "frame_id": fid,
                        "density": np.nan,
                        "n_edges": 0,
                        "mean_stability": np.nan,
                    }
                )
                continue
            sg = compute_shape_graph(positions)
            max_edges = n * (n - 1) / 2
            density = float(len(sg.edges)) / max_edges if max_edges > 0 else 0.0
            n_edges = len(sg.edges)
            mean_stab = float(np.mean(sg.stabilities)) if n_edges > 0 else np.nan
            frame_metrics.append(
                {
                    "game_id": gid,
                    "period_id": pid,
                    "frame_id": fid,
                    "density": density,
                    "n_edges": n_edges,
                    "mean_stability": mean_stab,
                }
            )
        if frame_metrics:
            sdf = pd.DataFrame(frame_metrics).set_index(["game_id", "period_id", "frame_id"])
            sg_indexed[tid] = sdf

    if not sg_indexed:
        return out

    # Link actions to frames
    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return out

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    linked = linked.merge(
        actions[["action_id", "team_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )

    # Align game_id dtype
    if len(linked) > 0 and sg_indexed:
        sample_sdf = next(iter(sg_indexed.values()))
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

        for tid, sdf in sg_indexed.items():
            if key not in sdf.index:
                continue
            sg_row = sdf.loc[key]
            suffix = "attacking" if same_id(tid, action_team) else "defending"
            for metric in _SHAPE_GRAPH_METRICS:
                out.at[idx, f"shape_graph_{metric}_{suffix}"] = sg_row[metric]

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


def shape_graph_xfns(home_team_id: int | str) -> list:
    """Build VAEP xfn list for TF-39 shape graph features.

    Returns a list with ONE FrameAwareTransformer that emits 6 features x 3
    game-states = 18 columns total.

    Examples
    --------
    Compose into HybridVAEP::

        from silly_kicks.tracking.features import tracking_default_xfns, shape_graph_xfns
        xfns = tracking_default_xfns + shape_graph_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    from ._shape_graph import compute_shape_graph

    col_names = []
    for metric in _SHAPE_GRAPH_METRICS:
        for suffix in ("attacking", "defending"):
            col_names.append(f"shape_graph_{metric}_{suffix}")

    def _shape_graph_transformer(states, frames):
        """Multi-column shape-graph xfn (6 cols x nb_states)."""
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(3):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out

        # Pre-compute shape graph metrics per (team, frame)
        teams = frames[~frames["is_ball"].astype(bool)]["team_id"].dropna().unique()
        sg_indexed: dict = {}
        for tid in teams:
            team_outfield = frames[
                (frames["team_id"] == tid)
                & (~frames["is_ball"].astype(bool))
                & (~frames["is_goalkeeper"].astype(bool))
                & frames["x"].notna()
                & frames["y"].notna()
            ]
            if team_outfield.empty:
                continue
            rows_list: list[dict] = []
            for (gid, pid, fid), grp in team_outfield.groupby(["game_id", "period_id", "frame_id"], dropna=False):
                positions = grp[["x", "y"]].to_numpy(dtype="float64")
                n = len(positions)
                if n < 3:
                    rows_list.append(
                        {
                            "game_id": gid,
                            "period_id": pid,
                            "frame_id": fid,
                            "density": np.nan,
                            "n_edges": 0,
                            "mean_stability": np.nan,
                        }
                    )
                    continue
                sg = compute_shape_graph(positions)
                max_edges = n * (n - 1) / 2
                rows_list.append(
                    {
                        "game_id": gid,
                        "period_id": pid,
                        "frame_id": fid,
                        "density": float(len(sg.edges)) / max_edges if max_edges > 0 else 0.0,
                        "n_edges": len(sg.edges),
                        "mean_stability": float(np.mean(sg.stabilities)) if len(sg.edges) > 0 else np.nan,
                    }
                )
            if rows_list:
                sg_indexed[tid] = pd.DataFrame(rows_list).set_index(["game_id", "period_id", "frame_id"])

        for i, slot in enumerate(states[:3]):
            slot_result = _shape_graph_at_actions(slot, frames, home_team_id, sg_indexed)
            for col in col_names:
                out[f"{col}_a{i}"] = slot_result[col].to_numpy()
        return out

    _shape_graph_transformer._frame_aware = True  # type: ignore[attr-defined]
    _shape_graph_transformer.__name__ = "shape_graph"
    return [_shape_graph_transformer]


def _shape_graph_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    home_team_id: int | str,
    sg_indexed: dict,
) -> pd.DataFrame:
    """Join pre-indexed shape graph metrics to actions. Internal helper for xfn."""
    col_names = []
    for metric in _SHAPE_GRAPH_METRICS:
        for suffix in ("attacking", "defending"):
            col_names.append(f"shape_graph_{metric}_{suffix}")

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

    # Align game_id dtype
    if len(linked) > 0 and sg_indexed:
        sample_sdf = next(iter(sg_indexed.values()))
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

        for tid, sdf in sg_indexed.items():
            if key not in sdf.index:
                continue
            sg_row = sdf.loc[key]
            suffix = "attacking" if same_id(tid, action_team) else "defending"
            for metric in _SHAPE_GRAPH_METRICS:
                out.at[idx, f"shape_graph_{metric}_{suffix}"] = sg_row[metric]

    return out


# ---------------------------------------------------------------------------
# OBSO — Off-Ball Scoring Opportunity (TF-40)
#
# Architecture: per-pass windowed computation. For each pass action, uses
# slice_around_event to extract a frame window, computes pitch control at
# each timestep, then calls compute_pass_obso to get the triplet
# (actual_obso, peak_obso, optimal_obso).
# ---------------------------------------------------------------------------

_OBSO_COLUMNS = ("obso_actual", "obso_peak", "obso_optimal")

#: Per-row provenance for the EPV surface an OBSO-family value was computed from.
#: A warning cannot survive into a mart; this column can (ADR-041).
_EPV_SOURCE_COLUMN = "obso_epv_source"

_PASS_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("pass", "cross") if n in spadlconfig.actiontype_id)


def _resolve_epv_grid(
    xt: ExpectedThreat | None,
    epv_grid: np.ndarray | None,
    *,
    caller: str,
    params: object | None = None,
) -> tuple[np.ndarray | None, str]:
    """Resolve ``xt=`` / ``epv_grid=`` to ``(grid, source_label)``.

    The CALLER emits the synthetic-surface warning, so ``stacklevel`` is correct per call
    site; this helper stays warning-free. ``caller`` is the public function name and
    appears in user-facing messages. ``params`` selects the build geometry -- None uses
    the default ``ObsoParams``; ``compute_pass_obso`` passes its OWN resolved params so a
    non-default geometry is honoured.

    Returns ``(None, "synthetic")`` when neither input is supplied, leaving the
    downstream synthetic fallback in place.
    """
    from silly_kicks.xthreat import physical_grid, require_fitted_xt

    from ._obso import ObsoParams

    if xt is not None and epv_grid is not None:
        raise ValueError(f"{caller}: pass either xt= or epv_grid=, not both")
    if xt is not None:
        require_fitted_xt(xt, caller=caller)
        p = params if params is not None else ObsoParams()
        # NODE registration, matching every consumer of this grid (ADR-041):
        #   * compute_pass_obso's index map  x / pitch_length * (grid_nx - 1)
        #   * PitchControlSurface's          np.linspace(0, 105, grid_cells_x)
        #   * _interpolate_grid's            np.linspace(0, src - 1, tgt)
        # An earlier draft built CELL CENTRES ((i + 0.5) * L / n), which is off by up to
        # +-0.505 m at the edges -- and uncorrected, because building at exactly
        # (grid_ny, grid_nx) makes _interpolate_grid's identity shortcut return the grid
        # unresampled. It was invisible on the synthetic path (a pure linear x-ramp barely
        # notices a half-cell shift) but not for a structured fitted-xT surface.
        gx = np.linspace(0.0, p.pitch_length, p.grid_nx)  # type: ignore[attr-defined]
        gy = np.linspace(0.0, p.pitch_width, p.grid_ny)  # type: ignore[attr-defined]
        return physical_grid(xt, gx, gy), "xt"
    if epv_grid is not None:
        return epv_grid, "injected"
    return None, "synthetic"


def _warn_synthetic_epv(caller: str) -> None:
    """Emit the synthetic-surface notice, attributed to the USER's call site.

    ``stacklevel=3`` because there are exactly two frames between ``warn`` and the caller
    we want blamed: this helper, then the public aggregator/factory that called it. All
    six OBSO-family entry points invoke this from their own body, so one constant is
    correct for every site; ``@nan_safe_enrichment`` is a pure marker that returns ``fn``
    unwrapped, so it adds no frame. Pointing at ``features.py`` instead of the caller is
    how a warning becomes noise.
    """
    _warnings.warn(
        f"{caller}: OBSO EPV is the synthetic linspace(0.01, 0.3) placeholder ramp -- "
        "pass xt= (fitted ExpectedThreat) or epv_grid= for production surfaces.",
        SyntheticEPVWarning,
        stacklevel=3,
    )


def obso_actual(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    home_team_id: int | str = 0,
    links: pd.DataFrame | None = None,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.Series:
    """OBSO at the actual pass target at the event frame.

    Only produces values for pass actions; NaN for all others.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import obso_actual
    >>> s = obso_actual(actions, frames, home_team_id=1)
    """
    col_name = "obso_actual"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)

    lookup = _precompute_obso_lookup(
        actions,
        frames,
        links=links,
        home_team_id=home_team_id,
        transition_grid=transition_grid,
        epv_grid=epv_grid,
        pitch_control_method=pitch_control_method,
    )
    return pd.Series(
        [lookup.get(i, {}).get("actual_obso", np.nan) for i in range(len(actions))],
        index=actions.index,
        name=col_name,
    )


def obso_peak(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    home_team_id: int | str = 0,
    links: pd.DataFrame | None = None,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.Series:
    """Peak OBSO at the pass target across the frame window.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import obso_peak
    >>> s = obso_peak(actions, frames, home_team_id=1)
    """
    col_name = "obso_peak"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)

    lookup = _precompute_obso_lookup(
        actions,
        frames,
        links=links,
        home_team_id=home_team_id,
        transition_grid=transition_grid,
        epv_grid=epv_grid,
        pitch_control_method=pitch_control_method,
    )
    return pd.Series(
        [lookup.get(i, {}).get("peak_obso", np.nan) for i in range(len(actions))],
        index=actions.index,
        name=col_name,
    )


def obso_optimal(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    home_team_id: int | str = 0,
    links: pd.DataFrame | None = None,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.Series:
    """Optimal OBSO across all teammate positions at the event frame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import obso_optimal
    >>> s = obso_optimal(actions, frames, home_team_id=1)
    """
    col_name = "obso_optimal"
    if frames is None:
        return pd.Series(np.nan, index=actions.index, name=col_name)

    lookup = _precompute_obso_lookup(
        actions,
        frames,
        links=links,
        home_team_id=home_team_id,
        transition_grid=transition_grid,
        epv_grid=epv_grid,
        pitch_control_method=pitch_control_method,
    )
    return pd.Series(
        [lookup.get(i, {}).get("optimal_obso", np.nan) for i in range(len(actions))],
        index=actions.index,
        name=col_name,
    )


def _precompute_obso_lookup(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str = 0,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pre_seconds: float = 3.0,
    post_seconds: float = 1.0,
    pitch_control_cache: PitchControlCache | None = None,
) -> dict[int, dict[str, float]]:
    """Run OBSO computation for all pass actions, return row-index lookup.

    Returns dict mapping row position (0-based) to OBSO triplet dict.
    """
    from ._obso import _get_default_grids, compute_pass_obso
    from .pitch_control import PitchControlCache as _PitchControlCache

    # One cache across all passes so overlapping pass windows reuse surfaces
    # (TF-7 shared surface). A caller-supplied cache extends reuse across
    # feature families in a single enrichment pass.
    cache = pitch_control_cache if pitch_control_cache is not None else _PitchControlCache()

    # Ensure velocity columns
    if "vx" not in frames.columns or "vy" not in frames.columns:
        frames = frames.copy()
        if "vx" not in frames.columns:
            frames["vx"] = 0.0
        if "vy" not in frames.columns:
            frames["vy"] = 0.0

    # Dup-action_id-safe frame-id resolution by position (ADR: frame-aware xfns resolve
    # frame_id by position, never .at on a non-unique action_id). Here it gates "is this
    # action linked"; the OBSO window is time-based below.
    fid_by_pos = _kernels.resolve_frame_ids_by_position(actions, frames, links=links)

    # --- ADR-028 per-action orientation (ADR-041 amendment; DEFECT A) -------------------
    # Frames are home-attacks-right; actions are acting-team-LTR. For an RTL-acting team
    # BOTH of these are needed, and they are jointly invariant at the target itself
    # (epv_rtl[col(105-x)] == epv[col(x)]) -- what actually moves is WHICH frame location's
    # pitch control is sampled, plus the orientation of the surface-wide peak/optimal scan:
    #   (a) the action-LTR target must be point-reflected into frame coordinates, and
    #   (b) the attack-LTR EPV grid must be POINT-reflected into frame orientation.
    # transition_grid is orientation-neutral (ball-anchored) and is NOT flipped.
    #
    # BOTH axes, not just x (ADR-041 second pass): ADR-028's relation is x->105-x AND
    # y->68-y, so the grid transform is [::-1, ::-1]. An x-only flip is exact only for a
    # y-symmetric grid -- which the synthetic ramp default is, and a fitted xT surface
    # nearly is, so the error hid behind both. Pinned by
    # tests/tracking/test_obso_orientation.py::TestEpvIsReflectedOnBothAxes.
    flip_rtl = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
    transition_grid, epv_grid = _get_default_grids(transition_grid, epv_grid)
    epv_grid_rtl = np.ascontiguousarray(np.asarray(epv_grid)[::-1, ::-1])

    # Group frames for windowing
    frame_groups = frames.groupby(["period_id", "frame_id"])

    # Identify pass actions
    pass_mask = actions["type_id"].isin(_PASS_TYPE_IDS)

    # Loop-invariant hoist: precompute one sorted (frame_id, time_seconds) table
    # per period once, instead of rebuilding it for every pass action (was
    # O(passes x frames)). Identical output.
    period_frame_times: dict = {
        pid: grp.drop_duplicates("frame_id")[["frame_id", "time_seconds"]].sort_values("time_seconds")
        for pid, grp in frames.groupby("period_id")
    }

    lookup: dict[int, dict[str, float]] = {}

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        if not pass_mask.iloc[i]:
            continue

        if np.isnan(fid_by_pos[i]):
            continue

        period_id = int(action_row["period_id"])
        action_time = action_row["time_seconds"]
        team_id = action_row["team_id"]
        target_x = action_row["end_x"]
        target_y = action_row["end_y"]

        if pd.isna(target_x) or pd.isna(target_y) or pd.isna(team_id):
            continue

        # Per-action orientation (see the block above the loop). NaN targets are skipped
        # first, so the reflection only ever runs on real coordinates.
        if flip_rtl[i]:
            target_x = FIELD_LENGTH - float(target_x)
            target_y = FIELD_WIDTH - float(target_y)
            epv_for_action = epv_grid_rtl
        else:
            epv_for_action = epv_grid

        # Build frame window around the pass (per-period table precomputed above)
        unique_frame_times = period_frame_times.get(period_id)
        if unique_frame_times is None:
            continue
        t_min = action_time - pre_seconds
        t_max = action_time + post_seconds
        window_fids = unique_frame_times[
            (unique_frame_times["time_seconds"] >= t_min) & (unique_frame_times["time_seconds"] <= t_max)
        ]["frame_id"].values

        if len(window_fids) == 0:
            continue

        # Build list of single-frame DataFrames for the window
        window_frames: list[pd.DataFrame] = []
        event_idx = 0
        closest_dist = float("inf")
        for w_idx, wfid in enumerate(window_fids):
            try:
                wf = frame_groups.get_group((period_id, int(wfid)))
                window_frames.append(wf)
            except KeyError:
                window_frames.append(pd.DataFrame())
                continue

            # Find closest frame to action time
            wf_time = wf["time_seconds"].iloc[0] if len(wf) > 0 else float("inf")
            dist = abs(wf_time - action_time)
            if dist < closest_dist:
                closest_dist = dist
                event_idx = w_idx

        # Filter out empty frames
        valid_window = [wf for wf in window_frames if len(wf) > 0]
        if not valid_window:
            continue

        # Recompute event_idx after filtering
        event_idx_adjusted = min(event_idx, len(valid_window) - 1)

        try:
            result = compute_pass_obso(
                valid_window,
                event_frame_idx=event_idx_adjusted,
                target_position=(float(target_x), float(target_y)),
                attacking_team_id=team_id,
                transition_grid=transition_grid,
                epv_grid=epv_for_action,
                pitch_control_method=pitch_control_method,
                pitch_control_cache=cache,
            )
            lookup[i] = result
        except (ValueError, KeyError, IndexError):
            # Frame-level failures (degenerate geometry, missing frame) are
            # non-fatal and skipped; unexpected errors now propagate so real
            # bugs surface rather than being masked as NaN. (ADR-002
            # no-silent-swallow; luxury-lakehouse is currently the only
            # downstream consumer, so narrowing the catch surprises no
            # third party.)
            continue

    return lookup


@nan_safe_enrichment
def add_obso(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str = 0,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    xt: ExpectedThreat | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.DataFrame:
    """Enrich actions with OBSO columns (actual, peak, optimal).

    Only pass actions receive values; all other actions are NaN.
    Uses ``compute_pass_obso`` from ``silly_kicks.tracking._obso``.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions.
    frames : pd.DataFrame
        Long-form tracking frames.
    links : pd.DataFrame or None
        Pre-computed link pointers (from ``link_actions_to_frames``).
    home_team_id : int or str
        Home team identifier. Retained for signature stability and LTR-consistency
        checks; per-action orientation is keyed on the frames' own
        ``team_attacking_direction`` via ``acting_team_attacks_rtl`` (ADR-028), NOT on
        this parameter. Before ADR-041 it was accepted and never read at all.
    transition_grid : np.ndarray or None
        Pre-computed ball transition probability grid. Orientation-neutral
        (ball-anchored), so it is never mirrored per action.
    epv_grid : np.ndarray or None
        Pre-computed expected possession value grid. Mutually exclusive with ``xt``.
    xt : ExpectedThreat or None
        A FITTED xT model; its surface is sampled onto the OBSO grid and used as the EPV
        factor (ADR-041). Mutually exclusive with ``epv_grid``. When neither is supplied
        the synthetic placeholder ramp is used and a
        :class:`~silly_kicks.tracking.SyntheticEPVWarning` is emitted.
    pitch_control_method : str
        Pitch control model (default ``"spearman"``).

    Returns
    -------
    pd.DataFrame
        Actions enriched with ``obso_actual``, ``obso_peak``, ``obso_optimal`` and the
        ``obso_epv_source`` provenance column (``"synthetic"`` / ``"xt"`` /
        ``"injected"``; NA off-domain).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_obso
    >>> enriched = add_obso(actions, frames, home_team_id=1)
    """
    epv_grid, epv_source = _resolve_epv_grid(xt, epv_grid, caller="add_obso")
    if epv_source == "synthetic":
        _warn_synthetic_epv("add_obso")

    out = actions.copy()

    # Provenance skip guard
    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "link_quality_score",
        "n_candidate_frames",
    ]
    has_provenance = any(c in out.columns for c in provenance_cols)

    lookup = _precompute_obso_lookup(
        actions,
        frames,
        links=links,
        home_team_id=home_team_id,
        transition_grid=transition_grid,
        epv_grid=epv_grid,
        pitch_control_method=pitch_control_method,
        pitch_control_cache=pitch_control_cache,
    )

    for col in _OBSO_COLUMNS:
        out[col] = np.nan
    # NOTE: seeded SEPARATELY from the _OBSO_COLUMNS float pre-seed on purpose. Folding
    # this into that `out[col] = np.nan` loop and writing a str via .at upcasts to OBJECT
    # dtype, not pandas "string". Keep the pd.array(..., dtype="string") seed + .loc fill.
    # First non-numeric column emitted by a tracking aggregator (ADR-041).
    out[_EPV_SOURCE_COLUMN] = pd.array([pd.NA] * len(out), dtype="string")

    for row_pos, triplet in lookup.items():
        idx = actions.index[row_pos]
        out.at[idx, "obso_actual"] = triplet["actual_obso"]
        out.at[idx, "obso_peak"] = triplet["peak_obso"]
        out.at[idx, "obso_optimal"] = triplet["optimal_obso"]

    # Provenance for every row that actually received a value. Deliberately OUTSIDE the
    # `links is None` provenance branch below: a caller passing links= (the documented
    # pipeline optimization) must still receive it.
    out.loc[out["obso_actual"].notna(), _EPV_SOURCE_COLUMN] = epv_source

    # Add provenance if not already present
    if not has_provenance and links is None:
        pointers, _report = link_actions_to_frames(actions, frames)
        # drop_duplicates: shifted gamestate slots repeat the boundary action_id, so a
        # raw merge would fan out (ADR: dup-action_id safety). One provenance row per id.
        pointers = pointers.drop_duplicates("action_id")
        for pc in provenance_cols:
            if pc in pointers.columns:
                merged = actions[["action_id"]].merge(
                    pointers[["action_id", pc]],
                    on="action_id",
                    how="left",
                )
                out[pc] = merged[pc].values

    return out


def obso_xfns(
    home_team_id: int | str = 0,
    *,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    xt: ExpectedThreat | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> list:
    """Factory returning 3 FrameAwareTransformers for OBSO features.

    Produces 3 features x 3 gamestates = 9 VAEP columns:
    ``obso_actual``, ``obso_peak``, ``obso_optimal``.

    Parameters
    ----------
    home_team_id : int or str
        Home team identifier.
    transition_grid, epv_grid : np.ndarray or None
        Pre-computed grids (None uses synthetic defaults).
    xt : ExpectedThreat or None
        A FITTED xT model used as the EPV factor (ADR-041); mutually exclusive with
        ``epv_grid``. Resolved ONCE here, at factory-call time.
    pitch_control_method : str
        Pitch control model.

    Returns
    -------
    list
        List of 3 lifted FrameAwareTransformers.

    Examples
    --------
    >>> from silly_kicks.tracking.features import obso_xfns
    >>> xfns = obso_xfns(home_team_id=1)
    >>> len(xfns)
    3
    """
    epv_grid, epv_source = _resolve_epv_grid(xt, epv_grid, caller="obso_xfns")
    if epv_source == "synthetic":
        _warn_synthetic_epv("obso_xfns")

    xfns_out = []
    for col_key, fn in [
        ("obso_actual", obso_actual),
        ("obso_peak", obso_peak),
        ("obso_optimal", obso_optimal),
    ]:

        def _helper(
            actions,
            frames,
            *,
            _fn=fn,
            _htid=home_team_id,
            _tg=transition_grid,
            _eg=epv_grid,
            _pcm: Literal["spearman", "fernandez_bornn", "voronoi"] = pitch_control_method,
        ):
            return _fn(
                actions,
                frames,
                home_team_id=_htid,
                transition_grid=_tg,
                epv_grid=_eg,
                pitch_control_method=_pcm,
            )

        _helper.__name__ = col_key
        _helper._frame_aware = True  # type: ignore[attr-defined]
        xfns_out.append(lift_to_states(_helper))

    return xfns_out


# ---------------------------------------------------------------------------
# TF-41 — Space Creation (Fernandez & Bornn 2018)
# ---------------------------------------------------------------------------

# Lean 2-column contract (4.24.0, lakehouse/owner decision): the LOO is
# pointwise-monotone, so a team-side "destroyed" and an opponent-side "created"
# are structurally 0 and nets are exact redundancies — always-zero columns
# (shipped 3.21.0-4.23.0) are retired rather than carried. The two live
# measurements: attack-side creation + rest-defense denial.
_SPACE_CREATION_COLUMNS = (
    "space_created_m2",
    "space_denied_m2_opponent",
)

_SPACE_CREATION_NAN_ROW: dict[str, float] = dict.fromkeys(_SPACE_CREATION_COLUMNS, np.nan)


def _compute_space_creation_for_action(
    action_row: pd.Series,
    frame: pd.DataFrame,
    *,
    home_team_id: int | str,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> dict[str, float]:
    """Compute both-perspective space creation for the actor at one action frame.

    A single ``compute_space_created(include_opponent_perspective=True)`` call
    yields both live measurements from the same frame and grid, so
    ``space_created_m2`` and ``space_denied_m2_opponent`` share an identical
    NaN mask by construction.
    """
    from ._space_creation import _unique_team_ids, compute_space_created

    team_id = action_row["team_id"]
    player_id = action_row["player_id"]

    # ADR-003: NaN actor identifiers route to the NaN-row default, never the guard.
    if pd.isna(team_id) or pd.isna(player_id):
        return dict(_SPACE_CREATION_NAN_ROW)

    # Loud two-team guard with the action/frame key (lakehouse contract: a
    # resolvable actor on a corrupt frame must raise, not silently emit NaN).
    team_ids_in_frame = _unique_team_ids(frame)
    if len(team_ids_in_frame) != 2:
        raise ValueError(
            "space_creation opponent perspective requires exactly two team ids "
            f"in the linked frame; found {list(team_ids_in_frame)!r} "
            f"(game_id={action_row.get('game_id')!r}, "
            f"period_id={action_row.get('period_id')!r}, "
            f"frame_id={frame['frame_id'].iloc[0]!r}, "
            f"action_id={action_row.get('action_id')!r})"
        )

    result = compute_space_created(
        frame,
        attacking_team_id=team_id,
        transition_grid=transition_grid,
        epv_grid=epv_grid,
        pitch_control_method=pitch_control_method,
        pitch_control_cache=pitch_control_cache,
        include_opponent_perspective=True,
    )

    actor_row = result[ids_match(result["player_id"], player_id)]
    if len(actor_row) == 0:
        return dict(_SPACE_CREATION_NAN_ROW)

    row = actor_row.iloc[0]
    return {
        "space_created_m2": float(row["space_created_m2"]),
        "space_denied_m2_opponent": float(row["space_denied_m2_opponent"]),
    }


@nan_safe_enrichment
def add_space_creation(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str = 0,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    xt: ExpectedThreat | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.DataFrame:
    """Enrich actions with per-actor space creation columns.

    Computes differential OBSO (leave-one-out) for the acting player at
    each action's linked frame.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions.
    frames : pd.DataFrame
        Long-form tracking frames.
    links : pd.DataFrame or None
        Pre-computed link pointers.
    home_team_id : int or str
        Home team identifier.
    transition_grid, epv_grid : np.ndarray or None
        Pre-computed grids.
    xt : ExpectedThreat or None
        A FITTED xT model used as the EPV factor (ADR-041); mutually exclusive with
        ``epv_grid``. Emits the shared ``obso_epv_source`` provenance column -- ONE name
        across the OBSO family, since space creation is OBSO-derived and a consumer
        joining the two must not meet conflicting columns.
    pitch_control_method : str
        Pitch control model.

    Returns
    -------
    pd.DataFrame
        Actions enriched with the lean 2-column contract (4.24.0):
        ``space_created_m2`` >= 0 — the actor's leave-one-out differential
        on their own team's OBSO surface (space existing because of the
        actor's presence; attacking value) — and ``space_denied_m2_opponent``
        >= 0 — the same leave-one-out evaluated on the opposing team's OBSO
        surface (actor as defender), weighed by the opponent's OWN attacking
        geometry (x-mirrored transition/EPV artifacts; shared
        grid/sigmas/method keep magnitudes comparable; rest-defense value).
        The LOO is pointwise-monotone, so the complementary halves
        (team-side destroyed, opponent-side created) are structurally 0 and
        deliberately NOT part of the contract; net columns would be exact
        redundancies and are likewise omitted. The two columns share an
        identical NaN mask (no linked frame / actor absent); a linked frame
        without exactly two team ids raises ``ValueError``.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_space_creation
    >>> enriched = add_space_creation(actions, frames, home_team_id=1)
    """
    epv_grid, epv_source = _resolve_epv_grid(xt, epv_grid, caller="add_space_creation")
    if epv_source == "synthetic":
        _warn_synthetic_epv("add_space_creation")

    out = actions.copy()

    from .pitch_control import PitchControlCache as _PitchControlCache

    # One cache across all actions (TF-7 shared surface); a caller-supplied
    # cache extends reuse across feature families in a single pass.
    cache = pitch_control_cache if pitch_control_cache is not None else _PitchControlCache()

    provenance_cols = [
        "frame_id",
        "time_offset_seconds",
        "link_quality_score",
        "n_candidate_frames",
    ]
    has_provenance = any(c in out.columns for c in provenance_cols)

    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)

    # Dup-action_id-safe frame-id resolution by position (ADR: frame-aware xfns resolve
    # frame_id by position; reached with non-unique action_id via space_creation_xfns).
    # Equivalent to the old .at lookup on unique ids (test_resolve_frame_ids_by_position).
    fid_by_pos = _kernels.resolve_frame_ids_by_position(actions, frames, links=links)
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for col in _SPACE_CREATION_COLUMNS:
        out[col] = np.nan
    # Seeded SEPARATELY from the float pre-seed above on purpose -- see the identical
    # note in add_obso: folding a str into that np.nan loop yields OBJECT dtype, not
    # pandas "string" (ADR-041).
    out[_EPV_SOURCE_COLUMN] = pd.array([pd.NA] * len(out), dtype="string")

    for i, (_idx, action_row) in enumerate(actions.iterrows()):
        if np.isnan(fid_by_pos[i]):
            continue

        period_id = int(action_row["period_id"])
        frame_id_int = int(fid_by_pos[i])

        try:
            frame = frame_groups.get_group((period_id, frame_id_int))
        except KeyError:
            continue

        result = _compute_space_creation_for_action(
            action_row,
            frame,
            home_team_id=home_team_id,
            transition_grid=transition_grid,
            epv_grid=epv_grid,
            pitch_control_method=pitch_control_method,
            pitch_control_cache=cache,
        )

        idx = actions.index[i]
        for col, val in result.items():
            out.at[idx, col] = val

    # EPV provenance for every valued row; OUTSIDE the links-gated block below so a
    # caller passing links= still receives it (ADR-041).
    out.loc[out[_SPACE_CREATION_COLUMNS[0]].notna(), _EPV_SOURCE_COLUMN] = epv_source

    if not has_provenance and links is None:
        # drop_duplicates: shifted gamestate slots repeat the boundary action_id, so a
        # raw merge would fan out (ADR: dup-action_id safety). One provenance row per id.
        pointers_unique = pointers.drop_duplicates("action_id")
        for pc in provenance_cols:
            if pc in pointers_unique.columns:
                merged = actions[["action_id"]].merge(
                    pointers_unique[["action_id", pc]],
                    on="action_id",
                    how="left",
                )
                out[pc] = merged[pc].values

    return out


def space_creation_xfns(
    home_team_id: int | str = 0,
    *,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    xt: ExpectedThreat | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> list:
    """Factory returning FrameAwareTransformers for space creation features.

    Produces 2 features x 3 gamestates = 6 VAEP columns (lockstep with
    ``_SPACE_CREATION_COLUMNS``): ``space_created_m2`` (attack-side creation)
    and ``space_denied_m2_opponent`` (rest-defense denial). The string
    ``obso_epv_source`` provenance column is deliberately NOT surfaced as a VAEP feature.

    Parameters
    ----------
    home_team_id : int or str
        Home team identifier.
    transition_grid, epv_grid : np.ndarray or None
        Pre-computed grids (None uses synthetic defaults).
    xt : ExpectedThreat or None
        A FITTED xT model used as the EPV factor (ADR-041); mutually exclusive with
        ``epv_grid``. Resolved ONCE here, at factory-call time.
    pitch_control_method : str
        Pitch control model.

    Examples
    --------
    >>> from silly_kicks.tracking.features import space_creation_xfns
    >>> xfns = space_creation_xfns(home_team_id=1)
    >>> len(xfns)
    2
    """
    epv_grid, epv_source = _resolve_epv_grid(xt, epv_grid, caller="space_creation_xfns")
    if epv_source == "synthetic":
        _warn_synthetic_epv("space_creation_xfns")

    xfns_out = []
    for col_name in _SPACE_CREATION_COLUMNS:

        def _helper(
            actions,
            frames,
            *,
            _col=col_name,
            _htid=home_team_id,
            _tg=transition_grid,
            _eg=epv_grid,
            _pcm: Literal["spearman", "fernandez_bornn", "voronoi"] = pitch_control_method,
        ):
            if frames is None:
                return pd.Series(np.nan, index=actions.index, name=_col)
            enriched = add_space_creation(
                actions,
                frames,
                home_team_id=_htid,
                transition_grid=_tg,
                epv_grid=_eg,
                pitch_control_method=_pcm,
            )
            return enriched[_col].rename(_col)

        _helper.__name__ = col_name
        _helper._frame_aware = True  # type: ignore[attr-defined]
        xfns_out.append(lift_to_states(_helper))

    return xfns_out


# ---------------------------------------------------------------------------
# TF-42 — PAUSA scoring (Lee et al. 2026)
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_pausa(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str = 0,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    xt: ExpectedThreat | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.DataFrame:
    """Enrich pass actions with PAUSA decomposition columns.

    Requires OBSO columns (``obso_actual``, ``obso_peak``, ``obso_optimal``).
    If missing, computes them first via ``add_obso``.

    Parameters
    ----------
    actions, frames : pd.DataFrame
        SPADL actions and long-form tracking frames.
    links : pd.DataFrame or None
        Pre-computed link pointers.
    home_team_id : int or str
        Home team identifier.
    transition_grid, epv_grid : np.ndarray or None
        Pre-computed grids, forwarded to ``add_obso``.
    xt : ExpectedThreat or None
        A FITTED xT model used as the EPV factor (ADR-041); mutually exclusive with
        ``epv_grid``. **Ignored when the OBSO columns are already present** -- PAUSA then
        reuses them and emits an
        :class:`~silly_kicks.tracking.IgnoredSurfaceInputsWarning`.
    pitch_control_method : str
        Pitch control model.
    pitch_control_cache : PitchControlCache or None
        Shared surface cache, forwarded to ``add_obso`` (signature symmetry with the
        other OBSO-family aggregators).

    Returns
    -------
    pd.DataFrame
        Actions enriched with ``pausa_temporal``, ``pausa_spatial``,
        ``pausa_composite`` columns (NaN for non-pass actions).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_pausa
    >>> enriched = add_pausa(actions, frames, home_team_id=1)
    """
    from ._pausa import compute_pausa_batch

    out = actions.copy()

    # Ensure OBSO columns exist
    required_obso = {"obso_actual", "obso_peak", "obso_optimal"}
    if not required_obso.issubset(out.columns):
        out = add_obso(
            out,
            frames,
            links=links,
            home_team_id=home_team_id,
            transition_grid=transition_grid,
            epv_grid=epv_grid,
            xt=xt,
            pitch_control_method=pitch_control_method,
            pitch_control_cache=pitch_control_cache,
        )
    elif xt is not None or epv_grid is not None or transition_grid is not None:
        # Reuse path: the supplied surface inputs are never consulted. A caller threading
        # the SAME xt= through a chained pipeline hits this legitimately, so it is a
        # warning (with its own category) rather than a raise -- the library cannot tell
        # that case from the bug case (ADR-041).
        _warnings.warn(
            "add_pausa: obso columns already present -- supplied xt=/epv_grid=/"
            "transition_grid= are ignored (PAUSA reuses the existing columns). Chain "
            "add_obso with the same surface inputs, or drop the obso columns to recompute.",
            IgnoredSurfaceInputsWarning,
            stacklevel=2,
        )

    # Apply PAUSA only to rows with OBSO values
    has_obso = out["obso_actual"].notna()
    if has_obso.any():
        obso_subset = out.loc[has_obso, ["obso_actual", "obso_peak", "obso_optimal"]]
        pausa_result = compute_pausa_batch(obso_subset)
        for col in ("pausa_temporal", "pausa_spatial", "pausa_composite"):
            out[col] = np.nan
            out.loc[has_obso, col] = pausa_result[col].values
    else:
        for col in ("pausa_temporal", "pausa_spatial", "pausa_composite"):
            out[col] = np.nan

    return out


def pausa_xfns(
    home_team_id: int | str = 0,
    *,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    xt: ExpectedThreat | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> list:
    """Factory returning 3 FrameAwareTransformers for PAUSA features.

    Produces 3 features x 3 gamestates = 9 VAEP columns:
    ``pausa_temporal``, ``pausa_spatial``, ``pausa_composite``.

    Parameters
    ----------
    home_team_id : int or str
        Home team identifier.
    transition_grid, epv_grid : np.ndarray or None
        Pre-computed grids (None uses synthetic defaults).
    xt : ExpectedThreat or None
        A FITTED xT model used as the EPV factor (ADR-041); mutually exclusive with
        ``epv_grid``. Resolved ONCE here, at factory-call time.
    pitch_control_method : str
        Pitch control model.

    Examples
    --------
    >>> from silly_kicks.tracking.features import pausa_xfns
    >>> xfns = pausa_xfns(home_team_id=1)
    >>> len(xfns)
    3
    """
    epv_grid, epv_source = _resolve_epv_grid(xt, epv_grid, caller="pausa_xfns")
    if epv_source == "synthetic":
        _warn_synthetic_epv("pausa_xfns")

    col_names = ["pausa_temporal", "pausa_spatial", "pausa_composite"]
    xfns_out = []
    for col_name in col_names:

        def _helper(
            actions,
            frames,
            *,
            _col=col_name,
            _htid=home_team_id,
            _tg=transition_grid,
            _eg=epv_grid,
            _pcm: Literal["spearman", "fernandez_bornn", "voronoi"] = pitch_control_method,
        ):
            if frames is None:
                return pd.Series(np.nan, index=actions.index, name=_col)
            enriched = add_pausa(
                actions,
                frames,
                home_team_id=_htid,
                transition_grid=_tg,
                epv_grid=_eg,
                pitch_control_method=_pcm,
            )
            return enriched[_col].rename(_col)

        _helper.__name__ = col_name
        _helper._frame_aware = True  # type: ignore[attr-defined]
        xfns_out.append(lift_to_states(_helper))

    return xfns_out


# ---------------------------------------------------------------------------
# TF-43 — ELASTIC Sync (Kim et al. 2025)
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_elastic_sync(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    window_seconds: float = 1.0,
    accel_weight: float = 0.6,
    proximity_weight: float = 0.4,
    min_confidence: float = 0.1,
    frame_rate: int = 25,
) -> pd.DataFrame:
    """Enrich actions with ELASTIC event-tracking alignment columns.

    Runs the ELASTIC algorithm (Kim et al. 2025) to find the best-matching
    tracking frame for each action, returning alternative frame pointers
    with confidence scores.

    Returns
    -------
    pd.DataFrame
        Actions enriched with ``elastic_frame_id``, ``elastic_confidence``,
        ``elastic_error_seconds`` columns.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_elastic_sync
    >>> enriched = add_elastic_sync(actions, frames)
    """
    from ._elastic_sync import ElasticSyncParams, align_events_to_frames

    params = ElasticSyncParams(
        window_seconds=window_seconds,
        accel_weight=accel_weight,
        proximity_weight=proximity_weight,
        min_confidence=min_confidence,
        frame_rate=frame_rate,
    )

    alignment = align_events_to_frames(actions, frames, params=params)
    out = actions.copy()

    _elastic_cols = ["elastic_frame_id", "elastic_confidence", "elastic_error_seconds"]
    for col in _elastic_cols:
        out[col] = np.nan

    if not alignment.empty:
        out = out.merge(
            alignment[["action_id", *_elastic_cols]].rename(columns={c: f"_{c}" for c in _elastic_cols}),
            on="action_id",
            how="left",
        )
        for col in _elastic_cols:
            mask = out[f"_{col}"].notna()
            out.loc[mask, col] = out.loc[mask, f"_{col}"]
            out.drop(columns=f"_{col}", inplace=True)

    return out


def elastic_sync_xfns(
    *,
    window_seconds: float = 1.0,
    accel_weight: float = 0.6,
    proximity_weight: float = 0.4,
    min_confidence: float = 0.1,
    frame_rate: int = 25,
) -> list:
    """Factory returning FrameAwareTransformers for ELASTIC sync features.

    Produces 2 features x 3 gamestates = 6 VAEP columns:
    ``elastic_confidence``, ``elastic_error_seconds``.

    Examples
    --------
    >>> from silly_kicks.tracking.features import elastic_sync_xfns
    >>> xfns = elastic_sync_xfns()
    >>> len(xfns)
    2
    """
    col_names = ["elastic_confidence", "elastic_error_seconds"]
    xfns_out = []
    for col_name in col_names:

        def _helper(
            actions,
            frames,
            *,
            _col=col_name,
            _ws=window_seconds,
            _aw=accel_weight,
            _pw=proximity_weight,
            _mc=min_confidence,
            _fr=frame_rate,
        ):
            if frames is None:
                return pd.Series(np.nan, index=actions.index, name=_col)
            enriched = add_elastic_sync(
                actions,
                frames,
                window_seconds=_ws,
                accel_weight=_aw,
                proximity_weight=_pw,
                min_confidence=_mc,
                frame_rate=_fr,
            )
            return enriched[_col].rename(_col)

        _helper.__name__ = col_name
        _helper._frame_aware = True  # type: ignore[attr-defined]
        xfns_out.append(lift_to_states(_helper))

    return xfns_out


@nan_safe_enrichment
def add_xt_gk(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    params: XtGkParams | None = None,
) -> pd.DataFrame:
    """Add xT-GK columns (xt_gk_base/pev/rav/dzv/pressure + composite xt_gk + the three
    provenance columns xt_gk_origin_source/xt_gk_dest_source/xt_gk_origin_confidence) per
    GK-distribution action. ``xt`` is a REQUIRED pre-fitted ExpectedThreat (no self-fit --
    leakage contract). ``home_team_id`` is accepted for GK-feature-family signature parity
    (and CI-gate construction); the xT-GK math operates on LTR-normalized SPADL action
    coordinates and does not consume it. RAV uses a fitted GkCompletionModel (the bundled GS
    ``default``); the [das] extra is no longer required.

    See NOTICE for full bibliographic citations (Eyestone xT-GK).
    """
    out = actions.copy()
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]

    comp = compute_xt_gk(actions, frames, xt=xt, params=params, links=pointers)
    for c in comp.columns:
        out[c] = comp[c].to_numpy()

    # Idempotent provenance merge (skip if any provenance column already present).
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out


def xt_gk_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    params: XtGkParams | None = None,
) -> list:
    """Factory returning one frame-aware VAEP transformer for xT-GK, closing over the
    caller-fitted ``xt`` (no self-fit -- leakage contract). Emits xt_gk_*_a{i} per
    gamestate slot. ``home_team_id`` accepted for family parity + CI-gate construction
    (see add_xt_gk).

    The per-slot ``action_id`` rekey to a unique positional surrogate is required because
    the sub-calls (pressure_on_actor / the completion-density linker) are action_id-keyed and
    would fan out on a shifted gamestate slot that repeats the boundary action -- this is the
    same mechanism resolve_frame_ids_by_position uses internally. Frame linkage is by time, so
    values are unchanged; action_id is ephemeral here (only the value columns are returned).
    """
    from ._xt_gk import _OUTPUT_COLS

    def _xt_gk_transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(3):
                for col in _OUTPUT_COLS:
                    out[f"{col}_a{i}"] = np.nan
            return out
        for i, slot in enumerate(states[:3]):
            safe = slot.copy()
            safe["action_id"] = np.arange(len(safe))
            comp = compute_xt_gk(safe, frames, xt=xt, params=params)
            for col in _OUTPUT_COLS:
                out[f"{col}_a{i}"] = comp[col].to_numpy()
        return out

    _xt_gk_transformer._frame_aware = True  # type: ignore[attr-defined]
    _xt_gk_transformer.__name__ = "xt_gk"
    return [_xt_gk_transformer]


@nan_safe_enrichment
def add_gk_completion(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: GkCompletionModel | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich SPADL actions with a ``gk_completion`` column: the GK-distribution
    pass-completion probability (xT-GK RAV's P(success) model) for each in-scope GK
    distribution (goalkick, GK-actor pass/throw_in); NaN for out-of-scope actions.
    ``model=None`` -> bundled GS ``default``. NaN identifiers route to NaN output
    (ADR-003); ``links`` skips internal linking.

    For the lakehouse wide action-context table. Reuses the EXACT scoring path xT-GK's
    RAV uses (``_completion_p``): geometry is resolved on the FULL action list (so a
    NaN-destination goalkick's next-event destination is the next ACTUAL action --
    train==serve parity) and the model scores only the masked GK-distribution rows, so
    this column equals the P(success) RAV consumes. Emits the four linkage-provenance
    columns idempotently.

    See NOTICE for full bibliographic citations (Eyestone xT-GK).
    """
    from ._gk_geometry import resolve_gk_geometry
    from ._xt_gk import _completion_p, _gk_distribution_mask

    out = actions.copy()
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]

    geom = resolve_gk_geometry(actions, frames=frames, links=pointers)
    mask = _gk_distribution_mask(actions, frames)
    vals = np.full(len(actions), np.nan, dtype=float)
    if mask.any():
        vals[mask] = _completion_p(actions, frames, geom, mask, pointers, model)
    out["gk_completion"] = vals

    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out


# ---------------------------------------------------------------------------
# TF-48: post-shot goalmouth crossing geometry (ADR-030)
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_shot_goalmouth(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params: ShotGoalmouthParams | None = None,
) -> pd.DataFrame:
    """Add TF-48 post-shot goalmouth crossing geometry per shot action; NaN/NA out-of-scope.

    Adds 11 columns (canonical attacked-goal frame, x=105). Metrics:

    - ``shot_crossing_y`` (``float64``) — y where the ball trajectory crosses the goal plane (m).
    - ``shot_crossing_z`` (``float64``) — height at the crossing (m).
    - ``shot_speed`` (``float64``) — contact-segment ball speed (m/s).
    - ``shot_time_to_goal_line`` (``float64``) — flight time contact→goal plane (s).
    - ``shot_on_target_derived`` (``boolean``) — posts/bar + ball-radius tolerance (width folded in).

    Provenance:

    - ``shot_crossing_source`` (``object``) — ``native`` / ``tracking`` / ``no_ball_frames`` /
      ``insufficient_frames`` / ``no_crossing`` / ``unresolved``.
    - ``shot_crossing_confidence`` (``float64``), ``shot_fit_n_frames`` (``int64``),
      ``shot_fit_rmse`` (``float64``), ``shot_fit_end_reason`` (``object``),
      ``shot_z_profile`` (``object`` — ``rolling`` / ``airborne`` / ``bounced``).

    Pure geometry over the ball trajectory -- NOT a VAEP feature
    (post-contact outcome leakage; ADR-030 guard). NaN identifiers resolve to
    "unresolved" rows (ADR-003 -- implemented here, the decorator is marker-only).

    Pointer resolution deliberately uses the add_xt_gk-verbatim path, NOT
    _resolve_action_frame_context (the context helper additionally builds
    actor/opponent row joins TF-48 never consumes). Decision recorded in ADR-030.

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_shot_goalmouth
    >>> enriched = add_shot_goalmouth(actions, frames)  # doctest: +SKIP
    >>> enriched[["shot_crossing_y", "shot_crossing_z", "shot_crossing_source"]]  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    out = actions.copy()
    comp = compute_shot_goalmouth(actions, frames, links=links, params=params)
    for c in comp.columns:
        out[c] = comp[c].to_numpy() if comp[c].dtype != "boolean" else comp[c].array
    # EDGE policy (spec section 4 -- warnings live here, never in the pure engine):
    # a mostly-unresolvable shot set is a data-quality signal (mirrors on_low_coverage).
    src = comp["shot_crossing_source"].dropna()
    if len(src) > 0:
        bad = float(src.isin(["no_ball_frames", "unresolved"]).mean())
        if bad > 0.5:
            _warnings.warn(
                f"add_shot_goalmouth: {bad:.0%} of shot rows could not be resolved "
                "(no ball frames / goal-end unresolved) -- check frames coverage and "
                "the GK map for this match.",
                stacklevel=2,
            )
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            ptr_cols = pointers.set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out


def shot_crossing_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Goal-plane crossing y (m, canonical attacked-goal-at-x=105). NaN out-of-scope.

    Examples
    --------
    >>> from silly_kicks.tracking.features import shot_crossing_y
    >>> shot_crossing_y(actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return compute_shot_goalmouth(actions, frames)["shot_crossing_y"].rename("shot_crossing_y")


def shot_crossing_z(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Goal-plane crossing z (m; bar = 2.44). NaN out-of-scope or when ball z unavailable.

    Examples
    --------
    >>> from silly_kicks.tracking.features import shot_crossing_z
    >>> shot_crossing_z(actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return compute_shot_goalmouth(actions, frames)["shot_crossing_z"].rename("shot_crossing_z")


def shot_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Fitted initial ball speed at contact (m/s; contact sub-segment -- see ADR-030 M-1;
    segment-average horizontal components, underestimates true contact speed under drag).

    Examples
    --------
    >>> from silly_kicks.tracking.features import shot_speed
    >>> shot_speed(actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return compute_shot_goalmouth(actions, frames)["shot_speed"].rename("shot_speed")


def shot_time_to_goal_line(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Elapsed seconds from (refined) contact to the goal-plane crossing.

    Examples
    --------
    >>> from silly_kicks.tracking.features import shot_time_to_goal_line
    >>> shot_time_to_goal_line(actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return compute_shot_goalmouth(actions, frames)["shot_time_to_goal_line"].rename("shot_time_to_goal_line")


def shot_on_target_derived(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Nullable boolean: crossing within posts+bar expanded by the ball-radius tolerance
    (post/bar physical width folded into the tolerance BY DECISION -- ADR-030). NA unless
    source is observed/extrapolated and crossing z is available.

    Examples
    --------
    >>> from silly_kicks.tracking.features import shot_on_target_derived
    >>> shot_on_target_derived(actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return compute_shot_goalmouth(actions, frames)["shot_on_target_derived"].rename("shot_on_target_derived")
