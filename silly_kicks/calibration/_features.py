"""Augmented-VAEP feature enrichment for the TF-24 calibration harness.

Three entry points implementing the CachedObjective invariant/patch split (spec §4/§4a):

- ``enrich_invariant`` runs the 14 trial-INDEPENDENT enrichment steps once, leaving the 5
  trial-dependent columns as NaN placeholders.
- ``patch_trial_columns`` runs ONLY the 2 trial-dependent steps (link_zones pressure + off-ball
  runs), overwriting exactly those 5 columns.
- ``enrich_full`` runs ALL 16 steps with the trial params applied INLINE (the INDEPENDENT
  monolithic path the CachedObjective's ``evaluate`` uses, so ``assert_cache_equivalence`` is not
  tautological — lakehouse H1).

``ALL_FEATURES`` is the model's feature matrix — the proven set from the prior TC-3 monolith.
Line-break columns are deliberately NOT features (so they are never computed; the patch uses
``add_off_ball_runs``, not the ``add_off_ball_context`` umbrella). xT is a frozen exogenous
artifact passed in (``xt``), never fit here.

See NOTICE for the per-feature methodology citations.

Examples
--------
>>> from silly_kicks.calibration._features import enrich_invariant, patch_trial_columns
>>> # base, links = enrich_invariant(actions=a, frames=f, xt=xt, home_team_id=h,
>>> #     carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0})
>>> # patched = patch_trial_columns(base_actions=base, frames=f, links=links,
>>> #     home_team_id=h, k3=1.0, pre_seconds=1.5, min_displacement_m=3.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat

# The 5 columns written by the two trial-dependent steps (spec §4a).
_TRIAL_DEPENDENT_COLS = [
    "pressure_on_actor__link_zones",
    "n_off_ball_runners_pre_window",
    "max_off_ball_run_displacement_pre_window",
    "mean_off_ball_run_speed_pre_window",
    "n_off_ball_runners_toward_goal_pre_window",
]

_SPADL_FEATURES = [
    "type_id",
    "result_id",
    "bodypart_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]

_TRACKING_FEATURES = [
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "actor_arc_length_pre_window",
    "actor_displacement_pre_window",
    "pressure_on_actor__andrienko_oval",
    "pressure_on_actor__link_zones",
    "pressure_on_actor__bekkers_pi",
    "pitch_control_at_target__spearman",
    "pitch_control_at_target__fernandez_bornn",
    "pitch_control_at_target__voronoi",
    "defensive_line_x",
    "back_line_high_x",
    "compactness_x",
    "lateral_width",
    "max_lateral_gap",
    "back_n_count",
    "n_off_ball_runners_pre_window",
    "max_off_ball_run_displacement_pre_window",
    "mean_off_ball_run_speed_pre_window",
    "n_off_ball_runners_toward_goal_pre_window",
    "team_shape_centroid_x_attacking",
    "team_shape_centroid_y_attacking",
    "team_shape_convex_hull_area_attacking",
    "team_shape_team_length_attacking",
    "team_shape_team_width_attacking",
    "team_shape_stretch_index_attacking",
    "team_shape_centroid_x_defending",
    "team_shape_centroid_y_defending",
    "team_shape_convex_hull_area_defending",
    "team_shape_team_length_defending",
    "team_shape_team_width_defending",
    "team_shape_stretch_index_defending",
    "das_team",
    "das_opponent",
    "das_diff",
    "gk_pitch_control_share_weighted",
    "gk_reachable_area_m2",
    "gk_closing_time_mean_s__six_yard_box",
    "gk_closing_time_min_s__six_yard_box",
    "n_blocked_receivers",
    "n_potential_receivers",
    "blocking_score",
    "blocked_threat_fraction",
    "max_single_defender_blocking_score",
    "sync_score_min",
    "sync_score_mean",
    "sync_score_high_quality_frac",
]

ALL_FEATURES = _SPADL_FEATURES + _TRACKING_FEATURES


def _compute_das(
    actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame, carrier_params: dict
) -> pd.DataFrame:
    """Step 12: DAS team/opponent/diff + the public ``das_source`` provenance column.

    M8 (calibration must SURFACE silent DAS failures, not absorb them) is now served by the
    public provenance instead of a private ``das_ok`` flag: rows tagged
    ``DAS_SOURCE_UNSCOREABLE_CALL`` are the degrade the caller counts into the manifest
    (ADR-043). The private try/except that produced that flag is gone -- ``add_das`` owns
    the narrowed degrade, and every non-``DasUnscoreableError`` failure now propagates.

    The frames are pre-restricted to the action-linked ``(period_id, frame_id)`` pairs
    before ``add_das``, so the direction ``add_das`` pins is inferred on exactly the frame
    set the library would otherwise have inferred it on -- the DAS values are unchanged.
    """
    from silly_kicks.tracking import add_das, derive_team_in_possession, infer_ball_carrier

    carrier = infer_ball_carrier(
        frames,
        tolerance_m=carrier_params["tolerance_m"],
        beta=carrier_params["beta"],
        gamma=carrier_params["gamma"],
    )
    frames_with_tip = derive_team_in_possession(frames, carrier)
    del carrier
    linked = links[["action_id", "frame_id"]].dropna(subset=["frame_id"])
    linked = linked.merge(actions[["action_id", "period_id"]], on="action_id", how="left")
    linked_frame_ids = linked[["period_id", "frame_id"]].drop_duplicates()
    das_frames = frames_with_tip.merge(linked_frame_ids, on=["period_id", "frame_id"], how="inner")
    del linked, frames_with_tip
    return add_das(actions, das_frames, links=links, chunk_size=10)


def enrich_invariant(
    *,
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    home_team_id: int | str,
    carrier_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 14 trial-independent enrichment steps; leave the 5 trial cols as NaN.

    Returns ``(base_actions, links)``. ``xt`` is a frozen ``ExpectedThreat`` (consumed by
    gk-influence + cover-shadows only). ``carrier_params`` are the fixed Stage-1 optimum.
    A degraded DAS for this match (M8) is read off the public ``das_source`` column on
    ``base_actions`` (``DAS_SOURCE_UNSCOREABLE_CALL``), not a private flag (ADR-043).

    Examples
    --------
    Run once per match, then reuse the result across every Optuna trial -- the whole point of
    the invariant/patch split is that these 14 steps do not depend on the trial's parameters::

        base_actions, links = enrich_invariant(
            actions=actions,
            frames=frames,
            xt=frozen.model,                      # the frozen exogenous xT artifact
            home_team_id=home_team_id,
            carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
        )

        # per trial: only the 5 trial-dependent columns are recomputed
        trial_actions = patch_trial_columns(
            base_actions=base_actions,
            frames=frames,
            links=links,
            home_team_id=home_team_id,
            k3=1.0,
            pre_seconds=1.5,
            min_displacement_m=3.0,
        )

    A match whose DAS degraded is readable off ``base_actions["das_source"]``
    (``DAS_SOURCE_UNSCOREABLE_CALL``) rather than a private flag.
    """
    from silly_kicks.spadl.utils import add_pre_shot_gk_context
    from silly_kicks.tracking import (
        add_action_context,
        add_actor_pre_window,
        add_cover_shadows,
        add_defensive_line,
        add_gk_influence,
        add_pressure_on_actor,
        add_sync_score,
        add_team_shape,
        link_actions_to_frames,
        pitch_control_at_target,
    )

    actions = actions.copy()
    links, _report = link_actions_to_frames(actions, frames)

    actions = add_pre_shot_gk_context(actions, frames=frames)  # Step 1
    actions = add_action_context(actions, frames, links=links)  # Step 2
    actions = add_actor_pre_window(actions, frames, links=links)  # Step 3
    actions = add_pressure_on_actor(actions, frames, links=links, methods=("andrienko_oval",))  # 4a
    actions["pressure_on_actor__link_zones"] = np.nan  # Step 4b SKIPPED (k3)
    try:  # Step 4c
        actions = add_pressure_on_actor(actions, frames, links=links, methods=("bekkers_pi",))
    except ValueError as exc:
        # is_ball-only frames OR a velocity-less provider (no vx/vy) -> bekkers_pi can't compute.
        # Degrade that ONE feature to NaN rather than crashing the whole harness; velocities should
        # normally be derived upstream (loaders call derive_velocities). XGBoost handles the NaN.
        msg = str(exc)
        if "is_ball=True" in msg or "vx" in msg or "velocit" in msg.lower():
            actions["pressure_on_actor__bekkers_pi"] = np.nan
        else:
            raise
    for method in ("spearman", "fernandez_bornn", "voronoi"):  # Steps 5-7
        s = pitch_control_at_target(actions, frames, links=links, method=method)
        actions[s.name] = s.values
    actions = add_defensive_line(actions, frames, links=links, home_team_id=home_team_id)  # Step 8
    for col in _TRIAL_DEPENDENT_COLS[1:]:  # Step 9 SKIPPED (off-ball runs)
        actions[col] = np.nan
    # Step 10 (line-break) DELETED — not a feature (spec §4a).
    actions = add_team_shape(actions, frames, links=links, home_team_id=home_team_id)  # Step 11
    actions = _compute_das(actions, frames, links, carrier_params)  # Step 12
    # ADR-055: these two take an optional `goal_map` and no `home_team_id`. Left to default so
    # each derives the map from `frames` -- the same frames every other step here consumes.
    actions = add_gk_influence(actions, frames, xt, links=links)  # Step 13
    actions = add_cover_shadows(actions, frames, xt, links=links)  # 14
    actions = add_sync_score(actions, links)  # Step 15
    return actions, links


def patch_trial_columns(
    *,
    base_actions: pd.DataFrame,
    frames: pd.DataFrame,
    links: pd.DataFrame,
    home_team_id: int | str,
    k3: float,
    pre_seconds: float,
    min_displacement_m: float,
) -> pd.DataFrame:
    """Overwrite ONLY the 5 trial-dependent columns on a cached invariant base.

    Runs link_zones pressure (k3) + off-ball RUNS (pre_seconds, min_displacement_m). Uses
    ``add_off_ball_runs`` (RUN cols only), NOT the ``add_off_ball_context`` umbrella.

    Examples
    --------
    Patch only the 5 trial-dependent columns onto a cached invariant base::

        patched = patch_trial_columns(
            base_actions=base, frames=f, links=links,
            home_team_id=h, k3=1.0, pre_seconds=1.5, min_displacement_m=3.0,
        )
    """
    from silly_kicks.tracking import LinkParams, add_off_ball_runs, add_pressure_on_actor

    actions = base_actions.copy()
    actions = add_pressure_on_actor(
        actions,
        frames,
        links=links,
        methods=("link_zones",),
        params_per_method={"link_zones": LinkParams(k3=k3)},
    )
    actions = add_off_ball_runs(
        actions,
        frames,
        home_team_id=home_team_id,
        pre_seconds=pre_seconds,
        min_displacement_m=min_displacement_m,
    )
    return actions


def enrich_full(
    *,
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    home_team_id: int | str,
    carrier_params: dict,
    k3: float,
    pre_seconds: float,
    min_displacement_m: float,
) -> pd.DataFrame:
    """MONOLITHIC recompute: all 16 steps with the trial params applied INLINE at their natural
    positions (no NaN placeholders, no cached base).

    This is the INDEPENDENT full path the CachedObjective's ``evaluate`` uses. It must NOT be
    ``enrich_invariant`` + ``patch_trial_columns`` (that would make assert_cache_equivalence
    tautological — H1). Running 4b/9 inline BEFORE steps 11-15 is what lets the equivalence test
    catch a "trial-independent" step that secretly reads a trial-varying column.

    Examples
    --------
    Run the monolithic 16-step recompute with the trial params applied inline::

        full = enrich_full(
            actions=a, frames=f, xt=xt, home_team_id=h,
            carrier_params=cp, k3=1.0, pre_seconds=1.5, min_displacement_m=3.0,
        )
    """
    from silly_kicks.spadl.utils import add_pre_shot_gk_context
    from silly_kicks.tracking import (
        LinkParams,
        add_action_context,
        add_actor_pre_window,
        add_cover_shadows,
        add_defensive_line,
        add_gk_influence,
        add_off_ball_runs,
        add_pressure_on_actor,
        add_sync_score,
        add_team_shape,
        link_actions_to_frames,
        pitch_control_at_target,
    )

    actions = actions.copy()
    links, _report = link_actions_to_frames(actions, frames)
    actions = add_pre_shot_gk_context(actions, frames=frames)  # 1
    actions = add_action_context(actions, frames, links=links)  # 2
    actions = add_actor_pre_window(actions, frames, links=links)  # 3
    actions = add_pressure_on_actor(actions, frames, links=links, methods=("andrienko_oval",))  # 4a
    actions = add_pressure_on_actor(  # 4b INLINE (k3)
        actions,
        frames,
        links=links,
        methods=("link_zones",),
        params_per_method={"link_zones": LinkParams(k3=k3)},
    )
    try:  # 4c
        actions = add_pressure_on_actor(actions, frames, links=links, methods=("bekkers_pi",))
    except ValueError as exc:
        # is_ball-only frames OR a velocity-less provider (no vx/vy) -> bekkers_pi can't compute.
        # Degrade that ONE feature to NaN rather than crashing the whole harness; velocities should
        # normally be derived upstream (loaders call derive_velocities). XGBoost handles the NaN.
        msg = str(exc)
        if "is_ball=True" in msg or "vx" in msg or "velocit" in msg.lower():
            actions["pressure_on_actor__bekkers_pi"] = np.nan
        else:
            raise
    for method in ("spearman", "fernandez_bornn", "voronoi"):  # 5-7
        s = pitch_control_at_target(actions, frames, links=links, method=method)
        actions[s.name] = s.values
    actions = add_defensive_line(actions, frames, links=links, home_team_id=home_team_id)  # 8
    actions = add_off_ball_runs(  # 9 INLINE (pre_seconds, min_displacement_m)
        actions, frames, home_team_id=home_team_id, pre_seconds=pre_seconds, min_displacement_m=min_displacement_m
    )
    # Step 10 (line-break) DELETED — not a feature.
    actions = add_team_shape(actions, frames, links=links, home_team_id=home_team_id)  # 11
    actions = _compute_das(actions, frames, links, carrier_params)  # 12
    # ADR-055: see enrich_invariant -- optional `goal_map`, no `home_team_id`.
    actions = add_gk_influence(actions, frames, xt, links=links)  # 13
    actions = add_cover_shadows(actions, frames, xt, links=links)  # 14
    actions = add_sync_score(actions, links)  # 15
    return actions
