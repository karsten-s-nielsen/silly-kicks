"""The committed W9 link-population registry (spec 2026-11-08 tracking-batch §4.9 W9.6; ADR-056 derive-and-assert).

Every value here is asserted against its derivation in ``tests/test_link_population.py``:

- ``LINK_POPULATION`` equals ``derive_population()``;
- ``ADAPTERS`` covers exactly the public members that are neither registered ``add_*`` aggregators nor ADR-078
  shaped (``nonconforming_members()``);
- ``REFERENCE_ONLY`` is exactly the private members that no public member reaches, each with no package caller;
- ``BASE_MEMBERS`` is the population as derived at BASE_SHA, which the W0 driver's ``--members`` runs on both legs.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from tests._prod_fixtures import HOME_TEAM_ID

Call = Callable[[pd.DataFrame, pd.DataFrame, "pd.DataFrame | None"], object]


def _xt():
    """The battery's fitted, y-asymmetric xT surface (``scripts._sb_battery.audit_xt``): one surface for every
    xT-taking member, single-sourced with the add_* battery."""
    from scripts._sb_battery import audit_xt

    return audit_xt()


def _battery(adapter_name: str, member: str) -> Call:
    """A battery call-shape adapter (``scripts._sb_battery``) applied to a population member (no second copy)."""

    def call(actions, frames, links):
        import scripts._sb_battery as battery
        from tests.invariants._link_population import resolve_object

        return getattr(battery, adapter_name)(resolve_object(member))(actions, frames, links, HOME_TEAM_ID)

    return call


def _first(result):
    """The table half of a ``(table, report)`` result."""
    return result[0] if isinstance(result, tuple) else result


def _kw(member: str, **kw) -> Call:
    """``fn(actions, frames, **kw)``: ``frames`` by keyword when the member takes it keyword-only, ``links`` only
    where accepted, the sentinels resolved to the shared inputs. A ``(table, report)`` result yields the table."""

    def call(actions, frames, links):
        import inspect

        from tests.invariants._link_population import resolve_object

        fn = resolve_object(member)
        params = inspect.signature(fn).parameters
        args = {k: _resolve(v, frames) for k, v in kw.items()}
        if "links" in params:
            args["links"] = links
        if params["frames"].kind is inspect.Parameter.KEYWORD_ONLY:
            return _first(fn(actions, frames=frames, **args))
        return _first(fn(actions, frames, **args))

    return call


def _resolve(value, frames):
    if value is _XT:
        return _xt()
    if value is _GOAL_MAP:
        from silly_kicks.tracking import resolve_defended_goals

        return resolve_defended_goals(frames)
    if value is _GHOST_GK:
        from silly_kicks.tracking import GhostGkModel

        return GhostGkModel.from_variant("default")
    if value is _GHOST_OUTFIELD:
        from silly_kicks.tracking import GhostOutfieldModel

        return GhostOutfieldModel.from_variant("default")
    return value


def _links(actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame | None) -> pd.DataFrame:
    from silly_kicks.tracking.utils import link_actions_to_frames

    return links if links is not None else link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0]


def _with_xt(fn: Callable, *, takes_links: bool = True) -> Call:
    """``fn(actions, frames, xt, *, links=...)``."""

    def call(actions, frames, links):
        return fn(actions, frames, _xt(), links=links) if takes_links else fn(actions, frames, _xt())

    return call


def _features(xfns_of: Callable[[], list], *, atomic: bool = False) -> Call:
    """A frame-aware xfn list through the public VAEP path: ``VAEP(xfns=...).compute_features(game, actions,
    frames=...)``, one game-state slot set per action (duplicate ``action_id``s across slots by construction)."""

    def call(actions, frames, links):
        from silly_kicks.atomic.vaep.base import AtomicVAEP
        from silly_kicks.vaep.base import VAEP

        game = pd.Series({"game_id": actions["game_id"].iloc[0], "home_team_id": HOME_TEAM_ID})
        model = (AtomicVAEP if atomic else VAEP)(xfns=xfns_of())
        return model.compute_features(game, actions, frames=frames)

    return call


def _sync_score(actions, frames, links):
    from silly_kicks.tracking.utils import sync_score

    return sync_score(_links(actions, frames, links))


def _link_report_sync_scores(actions, frames, links):
    from silly_kicks.tracking.utils import link_actions_to_frames

    pointers, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    return report.sync_scores(links if links is not None else pointers)


def _value_off_ball_runs(actions, frames, links):
    from silly_kicks.tracking import detect_off_ball_runs
    from silly_kicks.tracking._run_values import value_off_ball_runs

    runs = detect_off_ball_runs(actions, frames)
    return value_off_ball_runs(runs, actions, frames, _xt(), links=links)


def _patch_trial_columns(actions, frames, links):
    from silly_kicks.calibration._features import patch_trial_columns

    return patch_trial_columns(
        base_actions=actions,
        frames=frames,
        links=_links(actions, frames, links),
        home_team_id=HOME_TEAM_ID,
        k3=1.0,
        pre_seconds=1.5,
        min_displacement_m=3.0,
    )


#: The calibration module's own example carrier parameters (calibration/_features.py:170).
_CARRIER_PARAMS = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}


def _atomic_gk_prereq(member: str) -> Call:
    """An atomic pre-shot GK member after its documented prerequisite ``add_pre_shot_gk_context`` (atomic)."""

    def call(actions, frames, links):
        import inspect

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from tests.invariants._link_population import resolve_object

        prepared = add_pre_shot_gk_context(actions, frames=frames, links=links)
        fn = resolve_object(member)
        if "links" in inspect.signature(fn).parameters:
            return fn(prepared, frames, links=links)
        return fn(prepared, frames)

    return call


def _enrich_invariant(actions, frames, links):
    from silly_kicks.calibration._features import enrich_invariant

    return _first(enrich_invariant(actions=actions, frames=frames, xt=_xt(), carrier_params=_CARRIER_PARAMS))


def _enrich_full(actions, frames, links):
    from silly_kicks.calibration._features import enrich_full

    return enrich_full(
        actions=actions,
        frames=frames,
        xt=_xt(),
        home_team_id=HOME_TEAM_ID,
        carrier_params=_CARRIER_PARAMS,
        k3=1.0,
        pre_seconds=1.5,
        min_displacement_m=3.0,
    )


def _coordination_signals(actions, frames, links):
    """The signals' per-window table (``CoordinationSignals.windows``), keyed by its own game."""
    from silly_kicks.coordination._signals import build_coordination_signals
    from silly_kicks.coordination._windows import period_windows

    signals = build_coordination_signals(frames, windows=period_windows(frames), actions=actions, links=links)
    return signals.windows


def _team_coordination(actions, frames, links):
    """The result's per-window table (``CoordinationResult.windows``), keyed by its own game."""
    from silly_kicks.coordination._compute import compute_team_coordination

    return compute_team_coordination(frames, actions=actions, links=links).windows


def _option_rows(actions, frames, links):
    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet

    keepers = frames.loc[frames["is_goalkeeper"].astype(bool), "player_id"].dropna().unique().tolist()
    option_set = ReconstructedOptionSet(
        actions,
        frames,
        xpass=PassCompletionModel.bundled(),
        params=GkDecisionParams(),
        keeper_ids=keepers,
        frame_convention="match_ltr",
        links=links,
    )
    return option_set.option_rows()


def _xfns(factory: str, **kw) -> Call:
    """A ``*_xfns`` factory (``module:name``) called with ``kw`` (sentinels resolved); an atomic factory runs through
    ``AtomicVAEP``."""
    return _features(_factory(factory, **kw), atomic=factory.startswith("silly_kicks.atomic."))


def _factory(factory: str, **kw) -> Callable[[], list]:
    def build():
        import importlib

        module, _, name = factory.partition(":")
        fn = getattr(importlib.import_module(module), name)
        positional = tuple(_xt() if v is _XT else v for v in kw.get("_positional", ()))
        args = {k: (_xt() if v is _XT else v) for k, v in kw.items() if k != "_positional"}
        return fn(*positional, **args)

    return build


_XT, _GOAL_MAP, _GHOST_GK, _GHOST_OUTFIELD = object(), object(), object(), object()

_F = "silly_kicks.tracking.features"
_AF = "silly_kicks.atomic.tracking.features"


#: The W9.6 population at BASE_SHA bb0c601 (the package .py files equal 0552f58; derived by
#: tests/invariants/_link_population.derive_population; owner
#: ruling 2026-11-09 accepted its size over the spec's 150 stop line). test_population_matches_registry asserts it
#: exactly.
LINK_POPULATION: frozenset[str] = frozenset(
    {
        "scripts._das_path_timing:_main",
        "scripts._das_path_timing:time_add_das_and_xfns",
        "scripts._das_path_timing:time_add_das_and_xfns.<locals>._add",
        "scripts._das_path_timing:time_add_das_and_xfns.<locals>._xfn",
        "scripts._receiver_validation:receiver_failed_pass_accuracy",
        "scripts._receiver_validation:trajectory_weak_labels",
        "scripts._rq_corpus:extract_played_passes",
        "scripts._sb_battery:call_aggregator",
        "scripts._sb_battery:defensive_credit",
        "scripts._sb_battery:defensive_credit.<locals>.call",
        "scripts._sb_battery:generic",
        "scripts._sb_battery:generic.<locals>.call",
        "scripts._sb_battery:gradientsports_player_ids",
        "scripts._sb_battery:gradientsports_player_ids.<locals>.call",
        "scripts._sb_battery:pre_shot_gk_angle",
        "scripts._sb_battery:pre_shot_gk_angle.<locals>.call",
        "scripts._sb_battery:pre_shot_gk_position",
        "scripts._sb_battery:pre_shot_gk_position.<locals>.call",
        "scripts._sb_battery:run_add_star_battery",
        "scripts._sb_battery:sync_score",
        "scripts._sb_battery:sync_score.<locals>.call",
        "scripts._sb_battery:visible_area_coverage",
        "scripts._sb_battery:visible_area_coverage.<locals>.call",
        "scripts._sb_battery:with_pressure_methods",
        "scripts._sb_battery:with_pressure_methods.<locals>.call",
        "scripts._sb_battery:with_xt",
        "scripts._sb_battery:with_xt.<locals>.call",
        "scripts._sb_battery:with_xt_keyword",
        "scripts._sb_battery:with_xt_keyword.<locals>.call",
        "scripts.build_rq_pass_scores:main",
        "scripts.build_rq_pass_scores:score_match",
        "scripts.derive_coordination_params:_pass_b",
        "scripts.derive_coordination_params:_pass_b.<locals>.work",
        "scripts.derive_coordination_params:main",
        "scripts.derive_coordination_params:pass_b_match",
        "scripts.hf_jobs.headtohead:_owner_eval_receiver",
        "scripts.measure_cover_shadow_argmax_agreement:main",
        "scripts.measure_cover_shadow_argmax_agreement:measure_match",
        "scripts.measure_f1b_feature_delta:_feat_receiver",
        "scripts.probe_action_context_baselines:_convert_and_write",
        "scripts.probe_action_context_baselines:_process_lakehouse_provider",
        "scripts.probe_action_context_baselines:_process_skillcorner",
        "scripts.probe_action_context_baselines:_verify_link_rate",
        "scripts.probe_action_context_baselines:main",
        "scripts.probe_preprocess_baseline:_enrich_from_slim_parquet",
        "scripts.probe_preprocess_baseline:main",
        "scripts.train_receiver_model:_deploy_work",
        "scripts.train_receiver_model:_deployment_counts_for_match",
        "scripts.train_receiver_model:_rows_work",
        "scripts.train_receiver_model:_rows_work.<locals>.work",
        "scripts.train_receiver_model:deployment_gate",
        "scripts.train_receiver_model:extract_candidate_rows",
        "scripts.tune_structural_pass_sigma:_match_records",
        "scripts.tune_structural_pass_sigma:main",
        "scripts.validate_das_native_parity:_bench_match",
        "scripts.validate_das_native_parity:_measure_match",
        "scripts.validate_das_native_parity:_scored_frames",
        "scripts.validate_das_native_parity:main",
        "scripts.validate_das_native_parity:run_benchmark",
        "scripts.validate_das_native_parity:run_corpus",
        "scripts.validate_das_native_parity:run_corpus.<locals>._work",
        "scripts.validate_sb360_licensed_corpus:main",
        "scripts.validate_sb360_licensed_corpus:measure_match",
        "silly_kicks.atomic.spadl.utils:add_pre_shot_gk_context",
        "silly_kicks.atomic.tracking.features:_aggregate_defensive_credit",
        "silly_kicks.atomic.tracking.features:actor_arc_length_pre_window",
        "silly_kicks.atomic.tracking.features:actor_speed",
        "silly_kicks.atomic.tracking.features:add_action_context",
        "silly_kicks.atomic.tracking.features:add_actor_pre_window",
        "silly_kicks.atomic.tracking.features:add_cover_shadows",
        "silly_kicks.atomic.tracking.features:add_defensive_credit",
        "silly_kicks.atomic.tracking.features:add_off_ball_run_values",
        "silly_kicks.atomic.tracking.features:add_packing",
        "silly_kicks.atomic.tracking.features:add_pitch_control",
        "silly_kicks.atomic.tracking.features:add_pre_shot_gk_angle",
        "silly_kicks.atomic.tracking.features:add_pre_shot_gk_position",
        "silly_kicks.atomic.tracking.features:add_press_commitment",
        "silly_kicks.atomic.tracking.features:add_pressure_on_actor",
        "silly_kicks.atomic.tracking.features:add_shot_goalmouth",
        "silly_kicks.atomic.tracking.features:add_structural_pass",
        "silly_kicks.atomic.tracking.features:atomic_pitch_control_xfns",
        "silly_kicks.atomic.tracking.features:atomic_pitch_control_xfns.<locals>._pc_helper",
        "silly_kicks.atomic.tracking.features:compute_defensive_credits",
        "silly_kicks.atomic.tracking.features:defenders_in_triangle_to_goal",
        "silly_kicks.atomic.tracking.features:nearest_defender_distance",
        "silly_kicks.atomic.tracking.features:off_ball_run_value_xfns",
        "silly_kicks.atomic.tracking.features:off_ball_run_value_xfns.<locals>._atomic_transformer",
        "silly_kicks.atomic.tracking.features:packing_xfns",
        "silly_kicks.atomic.tracking.features:packing_xfns.<locals>._atomic_transformer",
        "silly_kicks.atomic.tracking.features:pitch_control_at_target",
        "silly_kicks.atomic.tracking.features:pre_shot_gk_angle_off_goal_line",
        "silly_kicks.atomic.tracking.features:pre_shot_gk_angle_to_shot_trajectory",
        "silly_kicks.atomic.tracking.features:pre_shot_gk_distance_to_goal",
        "silly_kicks.atomic.tracking.features:pre_shot_gk_distance_to_shot",
        "silly_kicks.atomic.tracking.features:pre_shot_gk_x",
        "silly_kicks.atomic.tracking.features:pre_shot_gk_y",
        "silly_kicks.atomic.tracking.features:pressure_on_actor",
        "silly_kicks.atomic.tracking.features:receiver_zone_density",
        "silly_kicks.atomic.tracking.features:structural_pass_xfns",
        "silly_kicks.atomic.tracking.features:structural_pass_xfns.<locals>._atomic_transformer",
        "silly_kicks.atomic.vaep.features:xt_xfns",
        "silly_kicks.atomic.vaep.features:xt_xfns.<locals>._xt",
        "silly_kicks.calibration._carrier_objective:CarrierAccuracyObjective.evaluate",
        "silly_kicks.calibration._carrier_objective:_match_accuracy",
        "silly_kicks.calibration._carrier_objective:_prepare_match",
        "silly_kicks.calibration._features:_compute_das",
        "silly_kicks.calibration._features:enrich_full",
        "silly_kicks.calibration._features:enrich_invariant",
        "silly_kicks.calibration._features:patch_trial_columns",
        "silly_kicks.coordination._compute:compute_team_coordination",
        "silly_kicks.coordination._signals:_build_coordination_signals",
        "silly_kicks.coordination._signals:build_coordination_signals",
        "silly_kicks.coordination._windows:possession_windows_from_actions",
        "silly_kicks.gk_decision._reconstruct:ReconstructedOptionSet.__init__",
        "silly_kicks.gk_decision._reconstruct:ReconstructedOptionSet._frame_and_anchors_action_ltr",
        "silly_kicks.gk_decision._reconstruct:ReconstructedOptionSet._frame_id_for",
        "silly_kicks.gk_decision._reconstruct:ReconstructedOptionSet._rows_for_decision",
        "silly_kicks.gk_decision._reconstruct:ReconstructedOptionSet.option_rows",
        "silly_kicks.keeper_identity:_resolve_from_native",
        "silly_kicks.keeper_identity:resolve_keeper_identities",
        "silly_kicks.restdefense._arms:_deterrent",
        "silly_kicks.restdefense._arms:rest_defense_gk_deterrent",
        "silly_kicks.restdefense._arms:rest_defense_outfield_deterrent",
        "silly_kicks.restdefense._compute:compute_rest_defense",
        "silly_kicks.restdefense._windows:select_rest_defense_samples",
        "silly_kicks.spadl.utils:add_pre_shot_gk_context",
        "silly_kicks.spadl.utils:add_restart_coordinates",
        "silly_kicks.territorial_defense._compute:compute_territorial_defense",
        "silly_kicks.tracking._gk_geometry:_side_y",
        "silly_kicks.tracking._gk_geometry:_throwin_x",
        "silly_kicks.tracking._gk_geometry:_tracking_ball_xy",
        "silly_kicks.tracking._gk_geometry:_tracking_gk_xy",
        "silly_kicks.tracking._gk_geometry:_tracking_gk_xy_detected",
        "silly_kicks.tracking._gk_geometry:resolve_restart_geometry",
        "silly_kicks.tracking._gk_resolve:_gk_from_frames_linked",
        "silly_kicks.tracking._gk_resolve:acting_gk_from_frames",
        "silly_kicks.tracking._gk_resolve:defending_gk_from_frames",
        "silly_kicks.tracking._kernels:_defensive_line_at_actions",
        "silly_kicks.tracking._kernels:_packing_at_actions",
        "silly_kicks.tracking._kernels:_structural_pass_at_actions",
        "silly_kicks.tracking._kernels:resolve_frame_ids_by_position",
        "silly_kicks.tracking._line_breaking:detect_line_breaking",
        "silly_kicks.tracking._off_ball_runs:_line_break_kernel",
        "silly_kicks.tracking._press_commitment:compute_press_commitment",
        "silly_kicks.tracking._receiver:_link_and_index",
        "silly_kicks.tracking._receiver:intended_receiver_positions",
        "silly_kicks.tracking._receiver:resolve_intended_receiver",
        "silly_kicks.tracking._run_features:_call_family",
        "silly_kicks.tracking._run_features:run_tracking_features",
        "silly_kicks.tracking._run_values:value_off_ball_runs",
        "silly_kicks.tracking._shot_goalmouth:compute_shot_goalmouth",
        "silly_kicks.tracking._visibility:add_visible_area_coverage",
        "silly_kicks.tracking._xcross_attempt:add_xcross_attempt",
        "silly_kicks.tracking._xcross_attempt:xcross_attempt_xfns",
        "silly_kicks.tracking._xcross_attempt:xcross_attempt_xfns.<locals>._transformer",
        "silly_kicks.tracking._xshot_occurrence:add_xshot_occurrence",
        "silly_kicks.tracking._xshot_occurrence:xshot_occurrence_xfns",
        "silly_kicks.tracking._xshot_occurrence:xshot_occurrence_xfns.<locals>._transformer",
        "silly_kicks.tracking.defensive_credit._orchestration:_aggregate_defensive_credit",
        "silly_kicks.tracking.defensive_credit._orchestration:_defensive_credit_observation_rollup",
        "silly_kicks.tracking.defensive_credit._orchestration:_rollup_defending_aggregate",
        "silly_kicks.tracking.defensive_credit._orchestration:compute_defensive_credits",
        "silly_kicks.tracking.features:_append_engine_companions",
        "silly_kicks.tracking.features:_append_fixed_zone_companions",
        "silly_kicks.tracking.features:_closing_time_per_series",
        "silly_kicks.tracking.features:_gk_influence_at_actions",
        "silly_kicks.tracking.features:_make_das_transformer",
        "silly_kicks.tracking.features:_make_das_transformer.<locals>.das_features",
        "silly_kicks.tracking.features:_map_das_to_actions",
        "silly_kicks.tracking.features:_player_influence_at_actions",
        "silly_kicks.tracking.features:_precompute_obso_lookup",
        "silly_kicks.tracking.features:_run_values_at_actions",
        "silly_kicks.tracking.features:_shape_graph_at_actions",
        "silly_kicks.tracking.features:_team_shape_at_actions",
        "silly_kicks.tracking.features:actor_arc_length_pre_window",
        "silly_kicks.tracking.features:actor_reachable_area_m2",
        "silly_kicks.tracking.features:actor_speed",
        "silly_kicks.tracking.features:add_action_context",
        "silly_kicks.tracking.features:add_actor_pre_window",
        "silly_kicks.tracking.features:add_cover_shadows",
        "silly_kicks.tracking.features:add_das",
        "silly_kicks.tracking.features:add_defensive_credit",
        "silly_kicks.tracking.features:add_defensive_line",
        "silly_kicks.tracking.features:add_ghost_gk",
        "silly_kicks.tracking.features:add_gk_influence",
        "silly_kicks.tracking.features:add_line_break",
        "silly_kicks.tracking.features:add_obso",
        "silly_kicks.tracking.features:add_off_ball_context",
        "silly_kicks.tracking.features:add_off_ball_run_values",
        "silly_kicks.tracking.features:add_packing",
        "silly_kicks.tracking.features:add_pausa",
        "silly_kicks.tracking.features:add_pitch_control",
        "silly_kicks.tracking.features:add_player_influence",
        "silly_kicks.tracking.features:add_pre_shot_gk_angle",
        "silly_kicks.tracking.features:add_pre_shot_gk_position",
        "silly_kicks.tracking.features:add_press_commitment",
        "silly_kicks.tracking.features:add_pressure_on_actor",
        "silly_kicks.tracking.features:add_shape_graph",
        "silly_kicks.tracking.features:add_shot_goalmouth",
        "silly_kicks.tracking.features:add_space_creation",
        "silly_kicks.tracking.features:add_structural_pass",
        "silly_kicks.tracking.features:add_team_shape",
        "silly_kicks.tracking.features:back_line_high_x",
        "silly_kicks.tracking.features:back_n_count",
        "silly_kicks.tracking.features:ball_carrier_at_action",
        "silly_kicks.tracking.features:compactness_x",
        "silly_kicks.tracking.features:cover_shadow_xfns",
        "silly_kicks.tracking.features:cover_shadow_xfns.<locals>._cover_shadow_transformer",
        "silly_kicks.tracking.features:cover_shadow_xfns.<locals>._cover_shadow_transformer.<locals>._get_cs",
        "silly_kicks.tracking.features:das_at_action",
        "silly_kicks.tracking.features:das_xfns",
        "silly_kicks.tracking.features:defenders_in_triangle_to_goal",
        "silly_kicks.tracking.features:defensive_line_x",
        "silly_kicks.tracking.features:defensive_line_xfns",
        "silly_kicks.tracking.features:defensive_line_xfns.<locals>._defensive_line_transformer",
        "silly_kicks.tracking.features:elastic_sync_xfns",
        "silly_kicks.tracking.features:elastic_sync_xfns.<locals>._helper",
        "silly_kicks.tracking.features:ghost_gk_xfns",
        "silly_kicks.tracking.features:ghost_gk_xfns.<locals>._ghost_gk_transformer",
        "silly_kicks.tracking.features:gk_closing_time_mean_s",
        "silly_kicks.tracking.features:gk_closing_time_min_s",
        "silly_kicks.tracking.features:gk_influence_xfns",
        "silly_kicks.tracking.features:gk_influence_xfns.<locals>._gk_influence_transformer",
        "silly_kicks.tracking.features:gk_influence_xfns.<locals>._gk_influence_transformer.<locals>._get_gi",
        "silly_kicks.tracking.features:gk_pitch_control_share_weighted",
        "silly_kicks.tracking.features:gk_reachable_area_m2",
        "silly_kicks.tracking.features:lateral_width",
        "silly_kicks.tracking.features:line_breaking_ward_xfns",
        "silly_kicks.tracking.features:line_breaking_ward_xfns.<locals>._line_breaking_ward_transformer",
        "silly_kicks.tracking.features:max_lateral_gap",
        "silly_kicks.tracking.features:nearest_defender_distance",
        "silly_kicks.tracking.features:obso_actual",
        "silly_kicks.tracking.features:obso_optimal",
        "silly_kicks.tracking.features:obso_peak",
        "silly_kicks.tracking.features:obso_xfns",
        "silly_kicks.tracking.features:obso_xfns.<locals>._helper",
        "silly_kicks.tracking.features:off_ball_context_xfns",
        "silly_kicks.tracking.features:off_ball_context_xfns.<locals>._off_ball_context_transformer",
        "silly_kicks.tracking.features:off_ball_run_value_xfns",
        "silly_kicks.tracking.features:off_ball_run_value_xfns.<locals>._off_ball_run_value_transformer",
        "silly_kicks.tracking.features:off_ball_xt_opponent",
        "silly_kicks.tracking.features:off_ball_xt_team",
        "silly_kicks.tracking.features:packing_xfns",
        "silly_kicks.tracking.features:packing_xfns.<locals>._packing_transformer",
        "silly_kicks.tracking.features:pausa_xfns",
        "silly_kicks.tracking.features:pausa_xfns.<locals>._helper",
        "silly_kicks.tracking.features:pitch_control_at_target",
        "silly_kicks.tracking.features:pitch_control_xfns",
        "silly_kicks.tracking.features:pitch_control_xfns.<locals>._pc_helper",
        "silly_kicks.tracking.features:player_influence_xfns",
        "silly_kicks.tracking.features:player_influence_xfns.<locals>._player_influence_transformer",
        "silly_kicks.tracking.features:player_influence_xfns.<locals>._player_influence_transformer.<locals>._aggregate",
        "silly_kicks.tracking.features:player_influence_xfns.<locals>._player_influence_transformer.<locals>._get_pi",
        "silly_kicks.tracking.features:pre_shot_gk_angle_off_goal_line",
        "silly_kicks.tracking.features:pre_shot_gk_angle_to_shot_trajectory",
        "silly_kicks.tracking.features:pre_shot_gk_distance_to_goal",
        "silly_kicks.tracking.features:pre_shot_gk_distance_to_shot",
        "silly_kicks.tracking.features:pre_shot_gk_x",
        "silly_kicks.tracking.features:pre_shot_gk_y",
        "silly_kicks.tracking.features:pressure_on_actor",
        "silly_kicks.tracking.features:reachable_area_opponent",
        "silly_kicks.tracking.features:reachable_area_team",
        "silly_kicks.tracking.features:receiver_zone_density",
        "silly_kicks.tracking.features:shape_graph_xfns",
        "silly_kicks.tracking.features:shape_graph_xfns.<locals>._shape_graph_transformer",
        "silly_kicks.tracking.features:space_creation_xfns",
        "silly_kicks.tracking.features:space_creation_xfns.<locals>._helper",
        "silly_kicks.tracking.features:structural_pass_xfns",
        "silly_kicks.tracking.features:structural_pass_xfns.<locals>._structural_pass_transformer",
        "silly_kicks.tracking.features:team_shape_xfns",
        "silly_kicks.tracking.features:team_shape_xfns.<locals>._team_shape_transformer",
        "silly_kicks.tracking.schema:LinkReport.sync_scores",
        "silly_kicks.tracking.utils:_resolve_action_frame_context",
        "silly_kicks.tracking.utils:add_sync_score",
        "silly_kicks.tracking.utils:sync_score",
        "silly_kicks.tracking.utils:validate_fov",
        "silly_kicks.vaep.features.expected_threat:xt_xfns",
        "silly_kicks.vaep.features.expected_threat:xt_xfns.<locals>._xt",
    }
)


def _member_xt(member: str) -> Call:
    """A ``(actions, frames, xt, *, ...)`` member; ``links`` is passed only where the signature accepts it."""

    def call(actions, frames, links):
        import importlib
        import inspect

        module, _, name = member.partition(":")
        fn = getattr(importlib.import_module(module), name)
        if "links" in inspect.signature(fn).parameters:
            return fn(actions, frames, _xt(), links=links)
        return fn(actions, frames, _xt())

    return call


def _list_global(member: str) -> Call:
    """A module-level default xfn list (``module:NAME``)."""

    def build():
        import importlib

        module, _, name = member.partition(":")
        return list(getattr(importlib.import_module(module), name))

    return _features(build)


#: ADR-078-shaped members whose documented prerequisite must run first: an adapter overrides their generic call.
PREREQUISITE_ADAPTERS: dict[str, str] = {
    f"{_AF}:{name}": "needs defending_gk_player_id from atomic add_pre_shot_gk_context first"
    for name in (
        "add_pre_shot_gk_position",
        "add_pre_shot_gk_angle",
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
        "pre_shot_gk_angle_to_shot_trajectory",
        "pre_shot_gk_angle_off_goal_line",
    )
}

#: Public members that are neither registered ``add_*`` aggregators nor ADR-078 shaped: each gets its own call.
ADAPTERS: dict[str, Call] = {
    # (actions, frames, xt, ...) members
    **{
        m: _member_xt(m)
        for m in (
            f"{_AF}:add_cover_shadows",
            f"{_AF}:add_off_ball_run_values",
            f"{_F}:actor_reachable_area_m2",
            f"{_F}:gk_pitch_control_share_weighted",
            f"{_F}:gk_reachable_area_m2",
            f"{_F}:off_ball_xt_opponent",
            f"{_F}:off_ball_xt_team",
            f"{_F}:reachable_area_opponent",
            f"{_F}:reachable_area_team",
        )
    },
    # *_xfns factories, through the public VAEP feature path
    **{
        f"{mod}:{name}": _xfns(f"{mod}:{name}")
        for mod, name in (
            (_AF, "atomic_pitch_control_xfns"),
            (_AF, "packing_xfns"),
            (_AF, "structural_pass_xfns"),
            (_F, "defensive_line_xfns"),
            (_F, "elastic_sync_xfns"),
            (_F, "line_breaking_ward_xfns"),
            (_F, "off_ball_context_xfns"),
            (_F, "packing_xfns"),
            (_F, "pitch_control_xfns"),
            (_F, "shape_graph_xfns"),
            (_F, "structural_pass_xfns"),
            (_F, "team_shape_xfns"),
        )
    },
    **{
        f"{mod}:{name}": _xfns(f"{mod}:{name}", _positional=(_XT,))
        for mod, name in (
            (_AF, "off_ball_run_value_xfns"),
            (_F, "cover_shadow_xfns"),
            (_F, "gk_influence_xfns"),
            (_F, "off_ball_run_value_xfns"),
            (_F, "player_influence_xfns"),
        )
    },
    f"{_F}:obso_xfns": _xfns(f"{_F}:obso_xfns", xt=_XT),
    f"{_F}:pausa_xfns": _xfns(f"{_F}:pausa_xfns", xt=_XT),
    f"{_F}:space_creation_xfns": _xfns(f"{_F}:space_creation_xfns", home_team_id=HOME_TEAM_ID, xt=_XT),
    f"{_F}:ghost_gk_xfns": _xfns(f"{_F}:ghost_gk_xfns", home_team_id=HOME_TEAM_ID),
    "silly_kicks.tracking._xcross_attempt:xcross_attempt_xfns": _xfns(
        "silly_kicks.tracking._xcross_attempt:xcross_attempt_xfns", home_team_id=HOME_TEAM_ID
    ),
    "silly_kicks.tracking._xshot_occurrence:xshot_occurrence_xfns": _xfns(
        "silly_kicks.tracking._xshot_occurrence:xshot_occurrence_xfns", home_team_id=HOME_TEAM_ID
    ),
    "silly_kicks.vaep.features.expected_threat:xt_xfns": _xfns(
        "silly_kicks.vaep.features.expected_threat:xt_xfns", model=_XT
    ),
    "silly_kicks.atomic.vaep.features:xt_xfns": _xfns("silly_kicks.atomic.vaep.features:xt_xfns", model=_XT),
    f"{_F}:das_xfns": _list_global(f"{_F}:das_xfns"),
    **{m: _atomic_gk_prereq(m) for m in PREREQUISITE_ADAPTERS},
    # keyword frames, or required keyword inputs
    "silly_kicks.spadl.utils:add_pre_shot_gk_context": _kw("silly_kicks.spadl.utils:add_pre_shot_gk_context"),
    "silly_kicks.atomic.spadl.utils:add_pre_shot_gk_context": _kw(
        "silly_kicks.atomic.spadl.utils:add_pre_shot_gk_context"
    ),
    "silly_kicks.spadl.utils:add_restart_coordinates": _kw("silly_kicks.spadl.utils:add_restart_coordinates"),
    "silly_kicks.tracking._gk_geometry:resolve_restart_geometry": _kw(
        "silly_kicks.tracking._gk_geometry:resolve_restart_geometry"
    ),
    "silly_kicks.restdefense._windows:select_rest_defense_samples": _kw(
        "silly_kicks.restdefense._windows:select_rest_defense_samples", goal_map=_GOAL_MAP
    ),
    "silly_kicks.restdefense._arms:rest_defense_gk_deterrent": _kw(
        "silly_kicks.restdefense._arms:rest_defense_gk_deterrent",
        xt=_XT,
        ghost_gk_model=_GHOST_GK,
        home_team_id=HOME_TEAM_ID,
    ),
    "silly_kicks.restdefense._arms:rest_defense_outfield_deterrent": _kw(
        "silly_kicks.restdefense._arms:rest_defense_outfield_deterrent",
        xt=_XT,
        ghost_outfield_model=_GHOST_OUTFIELD,
        home_team_id=HOME_TEAM_ID,
    ),
    "silly_kicks.territorial_defense._compute:compute_territorial_defense": _kw(
        "silly_kicks.territorial_defense._compute:compute_territorial_defense", xt=_XT, frame_convention="match_ltr"
    ),
    "silly_kicks.keeper_identity:resolve_keeper_identities": _kw(
        "silly_kicks.keeper_identity:resolve_keeper_identities", identity="native"
    ),
    **{
        m: _battery("defensive_credit", m)
        for m in (
            f"{_AF}:add_defensive_credit",
            f"{_AF}:compute_defensive_credits",
            "silly_kicks.tracking.defensive_credit._orchestration:compute_defensive_credits",
        )
    },
    # the rest
    "silly_kicks.calibration._features:enrich_invariant": _enrich_invariant,
    "silly_kicks.calibration._features:enrich_full": _enrich_full,
    "silly_kicks.calibration._features:patch_trial_columns": _patch_trial_columns,
    "silly_kicks.coordination._compute:compute_team_coordination": _team_coordination,
    "silly_kicks.coordination._signals:build_coordination_signals": _coordination_signals,
    "silly_kicks.tracking._run_values:value_off_ball_runs": _value_off_ball_runs,
    "silly_kicks.tracking.utils:sync_score": _sync_score,
    "silly_kicks.tracking.schema:LinkReport.sync_scores": _link_report_sync_scores,
    "silly_kicks.gk_decision._reconstruct:ReconstructedOptionSet.__init__": _option_rows,
    "silly_kicks.gk_decision._reconstruct:ReconstructedOptionSet.option_rows": _option_rows,
}

#: Pooled outputs (one scalar or one model over all matches): exempt from the split property, with a reason. Each
#: one's per-match inputs come from per-action population members (spec W9.6).
EXEMPT_POOLED: dict[str, str] = {
    "silly_kicks.tracking.utils:validate_fov": (
        "one FOV regime diagnosis over the whole call (FovDiagnosis); its per-action input is the visible-area "
        "table that the per-action visibility members produce"
    ),
    "silly_kicks.calibration._carrier_objective:CarrierAccuracyObjective.evaluate": (
        "one pooled carrier accuracy over every prepared match; its per-match input is _prepare_match, a "
        "per-match member reached through the same public objective"
    ),
}

#: Per-match / per-key outputs: the rows of game a in f(a + b) equal f(a), keyed by the output's own game key.
PER_KEY: frozenset[str] = frozenset(
    {
        "silly_kicks.coordination._compute:compute_team_coordination",
        "silly_kicks.coordination._signals:build_coordination_signals",
    }
)

#: Private package members that no public member reaches, each with no package caller (a test-only reference).
REFERENCE_ONLY: dict[str, str] = {
    "silly_kicks.calibration._carrier_objective:_match_accuracy": (
        "the uncached one-shot reference of CarrierAccuracyObjective; only tests/calibration/test_carrier_objective.py "
        "calls it"
    ),
}

#: Members that cannot run without caller links: the links=None variant does not apply (it is a call shape the
#: function does not offer), and the caller-links variant covers them.
LINKS_REQUIRED: dict[str, str] = {
    "silly_kicks.tracking.utils:add_sync_score": "add_sync_score(actions, links) scores caller pointers; no frames",
}

_EVENT_ONLY = "event-only: its xfns read actions and the xT grid, never frames, so no cross-game frame link exists"
_NO_FRAMES = "reads no frames (the visible-area polygons only), so no cross-game frame link exists"
_GAME_AWARE = (
    "game-aware at the base: on the two-game fixture its output equals the per-game calls (measured; its frame "
    "lookup does not go through the game-blind pointer join)"
)

#: (variant, member) pairs whose split property cannot fail at the base, each with its reason. Measured on the
#: two-game fixture at BASE_SHA on Python 3.12 (2026-11-09); every other public pair is red at the base.
#: The scope is that fixture only. Whole gradientsports matches (W9.6, 2026-11-12) failed two of these pairs at the
#: base: caller_links ``add_visible_area_coverage`` (an ``action_id`` set pooled the link status over games) and
#: ``defensive_line_xfns`` (an ``Int64`` column left the xfn as int64 or float64, by the data). Both are fixed.
CANNOT_FAIL_AT_BASE: dict[tuple[str, str], str] = {
    ("caller_links", "silly_kicks.atomic.tracking.features:add_off_ball_run_values"): _GAME_AWARE,
    ("caller_links", "silly_kicks.atomic.vaep.features:xt_xfns"): _EVENT_ONLY,
    ("caller_links", "silly_kicks.coordination._compute:compute_team_coordination"): _GAME_AWARE,
    ("caller_links", "silly_kicks.coordination._signals:build_coordination_signals"): _GAME_AWARE,
    ("caller_links", "silly_kicks.spadl.utils:add_restart_coordinates"): _GAME_AWARE,
    ("caller_links", "silly_kicks.territorial_defense._compute:compute_territorial_defense"): _GAME_AWARE,
    ("caller_links", "silly_kicks.tracking._gk_geometry:resolve_restart_geometry"): _GAME_AWARE,
    ("caller_links", "silly_kicks.tracking._kernels:resolve_frame_ids_by_position"): _GAME_AWARE,
    ("caller_links", "silly_kicks.tracking._run_values:value_off_ball_runs"): _GAME_AWARE,
    ("caller_links", "silly_kicks.tracking._visibility:add_visible_area_coverage"): _NO_FRAMES,
    ("caller_links", "silly_kicks.tracking.features:add_off_ball_run_values"): _GAME_AWARE,
    ("caller_links", "silly_kicks.tracking.features:defensive_line_xfns"): _GAME_AWARE,
    ("caller_links", "silly_kicks.tracking.features:shape_graph_xfns"): _GAME_AWARE,
    ("caller_links", "silly_kicks.tracking.features:team_shape_xfns"): _GAME_AWARE,
    ("caller_links", "silly_kicks.vaep.features.expected_threat:xt_xfns"): _EVENT_ONLY,
    ("links_none", "silly_kicks.atomic.vaep.features:xt_xfns"): _EVENT_ONLY,
    ("links_none", "silly_kicks.coordination._compute:compute_team_coordination"): _GAME_AWARE,
    ("links_none", "silly_kicks.coordination._signals:build_coordination_signals"): _GAME_AWARE,
    ("links_none", "silly_kicks.coordination._windows:possession_windows_from_actions"): _GAME_AWARE,
    ("links_none", "silly_kicks.spadl.utils:add_restart_coordinates"): _GAME_AWARE,
    ("links_none", "silly_kicks.territorial_defense._compute:compute_territorial_defense"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking._gk_geometry:resolve_restart_geometry"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking._visibility:add_visible_area_coverage"): _NO_FRAMES,
    ("links_none", "silly_kicks.tracking.features:back_line_high_x"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:back_n_count"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:compactness_x"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:defensive_line_x"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:defensive_line_xfns"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:lateral_width"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:max_lateral_gap"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:shape_graph_xfns"): _GAME_AWARE,
    ("links_none", "silly_kicks.tracking.features:team_shape_xfns"): _GAME_AWARE,
    ("links_none", "silly_kicks.vaep.features.expected_threat:xt_xfns"): _EVENT_ONLY,
}

_LINKER_FIXED = (
    "red at the base; the game-split property holds once the linker is game-aware (Task 4, W9.1): measured as a "
    "strict XPASS on the two-game fixture on 2026-11-10. Task 5 still migrates its joins to action_row"
)

#: (variant, member) pairs that the Task 4 linker change alone turns green (measured 2026-11-10 on the two-game
#: fixture: strict XPASS). Their red-at-base marks come off in Task 4, not Task 5.
GREEN_AFTER_LINKER: dict[tuple[str, str], str] = {
    ("caller_links", "silly_kicks.atomic.tracking.features:off_ball_run_value_xfns"): _LINKER_FIXED,
    ("caller_links", "silly_kicks.coordination._windows:possession_windows_from_actions"): _LINKER_FIXED,
    ("caller_links", "silly_kicks.tracking.features:off_ball_run_value_xfns"): _LINKER_FIXED,
    ("caller_links", "silly_kicks.tracking.utils:add_sync_score"): _LINKER_FIXED,
    ("links_none", "silly_kicks.atomic.tracking.features:add_off_ball_run_values"): _LINKER_FIXED,
    ("links_none", "silly_kicks.atomic.tracking.features:off_ball_run_value_xfns"): _LINKER_FIXED,
    ("links_none", "silly_kicks.tracking._kernels:resolve_frame_ids_by_position"): _LINKER_FIXED,
    ("links_none", "silly_kicks.tracking._run_values:value_off_ball_runs"): _LINKER_FIXED,
    ("links_none", "silly_kicks.tracking.features:add_off_ball_run_values"): _LINKER_FIXED,
    ("links_none", "silly_kicks.tracking.features:off_ball_run_value_xfns"): _LINKER_FIXED,
}

#: The population as derived at BASE_SHA: the W0 driver's --members runs exactly these on both legs (TB-PLAN-27).
BASE_MEMBERS: frozenset[str] = LINK_POPULATION
