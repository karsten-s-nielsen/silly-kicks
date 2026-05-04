"""Tracking-aware action_context features for atomic SPADL.

Mirrors silly_kicks.tracking.features with atomic-shaped column reads.
Shares the schema-agnostic kernels in silly_kicks.tracking._kernels.

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import _kernels
from silly_kicks.tracking.feature_framework import lift_to_states
from silly_kicks.tracking.pressure import Method, PressureParams
from silly_kicks.tracking.utils import _resolve_action_frame_context

_ATOMIC_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty"))

__all__ = [
    "actor_arc_length_pre_window",
    "actor_displacement_pre_window",
    "actor_speed",
    "add_action_context",
    "add_actor_pre_window",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_pressure_on_actor",
    "atomic_actor_pre_window_default_xfns",
    "atomic_pre_shot_gk_angle_default_xfns",
    "atomic_pre_shot_gk_default_xfns",
    "atomic_pre_shot_gk_full_default_xfns",
    "atomic_pressure_default_xfns",
    "atomic_tracking_default_xfns",
    "defenders_in_triangle_to_goal",
    "nearest_defender_distance",
    "pre_shot_gk_angle_off_goal_line",
    "pre_shot_gk_angle_to_shot_trajectory",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pressure_on_actor",
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


# ---------------------------------------------------------------------------
# PR-S21 — atomic pre_shot_gk_* mirror
# ---------------------------------------------------------------------------


def pre_shot_gk_x(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: defending GK's x at the linked frame (m, LTR-normalized).

    Mirrors :func:`silly_kicks.tracking.features.pre_shot_gk_x` with atomic shot type ids
    (``{shot, shot_penalty}`` — atomic does NOT recognize ``shot_freekick``, which is
    collapsed into ``freekick``).

    REQUIRES ``defending_gk_player_id`` column in ``actions``
    (run ``silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`` first).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_x
        atomic = add_pre_shot_gk_context(atomic)
        gk_x = pre_shot_gk_x(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_x"].rename("pre_shot_gk_x")


def pre_shot_gk_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: defending GK's y at the linked frame.

    See :func:`pre_shot_gk_x` for NaN/REQUIRES contract. See NOTICE for full citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_y
        atomic = add_pre_shot_gk_context(atomic)
        gk_y = pre_shot_gk_y(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_y"].rename("pre_shot_gk_y")


def pre_shot_gk_distance_to_goal(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: Euclidean distance (m) from defending GK to goal-mouth center.

    See :func:`pre_shot_gk_x` for NaN/REQUIRES contract. See NOTICE for full citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_distance_to_goal
        atomic = add_pre_shot_gk_context(atomic)
        d = pre_shot_gk_distance_to_goal(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_distance_to_goal"].rename("pre_shot_gk_distance_to_goal")


def pre_shot_gk_distance_to_shot(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: Euclidean distance (m) from defending GK to shot anchor (action.x, action.y).

    See :func:`pre_shot_gk_x` for NaN/REQUIRES contract. See NOTICE for full citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_distance_to_shot
        atomic = add_pre_shot_gk_context(atomic)
        d = pre_shot_gk_distance_to_shot(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_distance_to_shot"].rename("pre_shot_gk_distance_to_shot")


@nan_safe_enrichment
def add_pre_shot_gk_position(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: 4 GK-position columns + 4 linkage-provenance columns.

    Mirrors :func:`silly_kicks.tracking.features.add_pre_shot_gk_position` with atomic
    column reads (``x``, ``y``) and atomic shot type ids (``{shot, shot_penalty}``).

    REQUIRES ``defending_gk_player_id`` column in ``actions``
    (run ``silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`` first).

    Returns
    -------
    pd.DataFrame
        Input atomic actions with the columns:
        - pre_shot_gk_x (float64, m)
        - pre_shot_gk_y (float64, m)
        - pre_shot_gk_distance_to_goal (float64, m)
        - pre_shot_gk_distance_to_shot (float64, m)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)

    Raises
    ------
    ValueError
        If ``defending_gk_player_id`` column is absent.

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import add_pre_shot_gk_position
        atomic = add_pre_shot_gk_context(atomic)
        enriched = add_pre_shot_gk_position(atomic, frames)
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_position: actions missing required column "
            "'defending_gk_player_id'. Run silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context "
            "first to populate it."
        )
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    out = actions.copy()
    for col in ("pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot"):
        out[col] = df[col]
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_pre_shot_gk_default_xfns = [
    lift_to_states(pre_shot_gk_x),
    lift_to_states(pre_shot_gk_y),
    lift_to_states(pre_shot_gk_distance_to_goal),
    lift_to_states(pre_shot_gk_distance_to_shot),
]


# ---------------------------------------------------------------------------
# PR-S24 -- TF-12: atomic mirror of pre_shot_gk_angle_*
# ---------------------------------------------------------------------------


def pre_shot_gk_angle_to_shot_trajectory(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: signed angle (rad) GK->shot vs goal-centre->shot at the linked frame.

    See :func:`silly_kicks.tracking.features.pre_shot_gk_angle_to_shot_trajectory` for full
    semantics. Atomic shot type ids are ``{shot, shot_penalty}``.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_to_shot_trajectory"].rename("pre_shot_gk_angle_to_shot_trajectory")


def pre_shot_gk_angle_off_goal_line(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: signed angle (rad) of GK relative to goal-line normal at goal-mouth centre.

    See :func:`silly_kicks.tracking.features.pre_shot_gk_angle_off_goal_line`.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_off_goal_line"].rename("pre_shot_gk_angle_off_goal_line")


@nan_safe_enrichment
def add_pre_shot_gk_angle(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: 2 GK-angle columns at the linked frame.

    Mirrors :func:`silly_kicks.tracking.features.add_pre_shot_gk_angle` with atomic
    column reads (``x``, ``y``) and atomic shot type ids (``{shot, shot_penalty}``).

    REQUIRES ``defending_gk_player_id`` column in ``actions``.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_angle: actions missing required column 'defending_gk_player_id'. "
            "Run silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context first."
        )
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    out = actions.copy()
    for col in ("pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"):
        out[col] = df[col]
    return out


atomic_pre_shot_gk_angle_default_xfns = [
    lift_to_states(pre_shot_gk_angle_to_shot_trajectory),
    lift_to_states(pre_shot_gk_angle_off_goal_line),
]


atomic_pre_shot_gk_full_default_xfns = atomic_pre_shot_gk_default_xfns + atomic_pre_shot_gk_angle_default_xfns


# ---------------------------------------------------------------------------
# PR-S25 -- atomic mirror: TF-3 actor_*_pre_window
# ---------------------------------------------------------------------------


def actor_arc_length_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Atomic-SPADL: geometric arc-length of actor's path over pre-action window.

    See :func:`silly_kicks.tracking.features.actor_arc_length_pre_window`.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import actor_arc_length_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_arc_length_pre_window"].rename("actor_arc_length_pre_window")


def actor_displacement_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Atomic-SPADL: net Euclidean displacement over pre-action window.

    See :func:`silly_kicks.tracking.features.actor_displacement_pre_window`.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import actor_displacement_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
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
    """Atomic-SPADL aggregator for TF-3 features.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_actor_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    out = actions.copy()
    out["actor_arc_length_pre_window"] = df["actor_arc_length_pre_window"]
    out["actor_displacement_pre_window"] = df["actor_displacement_pre_window"]
    from silly_kicks.tracking.utils import link_actions_to_frames

    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_actor_pre_window_default_xfns = [lift_to_states(actor_arc_length_pre_window)]


# ---------------------------------------------------------------------------
# PR-S25 -- atomic mirror: TF-2 pressure_on_actor multi-flavor
# ---------------------------------------------------------------------------


def pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    method: Method = "andrienko_oval",
    params: PressureParams | None = None,
) -> pd.Series:
    """Atomic-SPADL: multi-flavor pressure on actor at linked frame.

    Mirrors :func:`silly_kicks.tracking.features.pressure_on_actor` with
    atomic anchor (x, y) instead of (start_x, start_y).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import pressure_on_actor
    >>> # See tests/atomic/tracking/test_pressure_*_atomic.py for runnable examples per method.
    """
    from silly_kicks.tracking.pressure import (
        AndrienkoParams,
        BekkersParams,
        LinkParams,
        validate_params_for_method,
    )

    validate_params_for_method(method, params)
    if method == "andrienko_oval":
        ap = params if isinstance(params, AndrienkoParams) else AndrienkoParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_andrienko(actions["x"], actions["y"], ctx, params=ap)
    elif method == "link_zones":
        lp = params if isinstance(params, LinkParams) else LinkParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_link(actions["x"], actions["y"], ctx, params=lp)
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
                "frames missing is_ball=True rows in linked frames."
            )
        ctx = _resolve_action_frame_context(actions, frames)
        from silly_kicks.tracking.features import _build_ball_xy_v_per_action

        ball_xy_v = _build_ball_xy_v_per_action(actions, frames, ctx)
        s = _kernels._pressure_bekkers(
            actions["x"],
            actions["y"],
            ctx,
            params=bp,
            ball_xy_v_per_action=ball_xy_v,
        )
    else:
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
    """Atomic-SPADL aggregator for multi-flavor TF-2 pressure.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_pressure_on_actor
    >>> # See tests/atomic/tracking/test_pressure_*_atomic.py for runnable examples.
    """
    from silly_kicks.tracking.pressure import validate_params_for_method

    if params_per_method is None:
        params_per_method = {}
    for m in methods:
        validate_params_for_method(m, params_per_method.get(m))
    out = actions.copy()
    for m in methods:
        s = pressure_on_actor(actions, frames, method=m, params=params_per_method.get(m))
        out[f"pressure_on_actor__{m}"] = s.values
    from silly_kicks.tracking.utils import link_actions_to_frames

    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_pressure_default_xfns = [lift_to_states(pressure_on_actor)]
