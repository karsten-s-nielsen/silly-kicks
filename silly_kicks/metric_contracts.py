"""Uniform public output-contract registry for silly-kicks metric families (SK-EXPORT).

Every column-emitting metric family -- ``team_metrics``, ``match_outcome``, ``shot_stopping``,
``gk_decision``, ``territory``, ``duels``, ``restdefense`` -- has one entry here, under one well-known
name: its mart-grain ``keys``, its value ``metric_columns``, its full ordered output ``columns``, and
-- where the package declares a typed ``dict[str, str]`` full-columns mapping -- its ``column_types``
(``None`` otherwise). This is the canonical surface for a consumer schema-drift parity guard: iterate
``METRIC_CONTRACTS`` and assert the mart's column set equals ``metric_columns | keys``, with zero
per-package name or type knowledge.

The registry is a plain ``dict`` valued by a :class:`MetricContract` ``TypedDict`` -- the house
"schemas are plain Python dicts" convention (cf. ``SPADL_COLUMNS``), with pyright-checked field keys
and no runtime class. Like ``feature_glossary``, it **imports no metric package** -- the values are
hardcoded literals, keeping each metric family a pure dependency-free LEAF (nothing in ``silly_kicks``
imports them; the ``tests/<pkg>/test_import_allowlist.py`` contracts stay intact). The literals are
kept honest by ``tests/test_metric_contracts.py``: its round-trip test imports each package's public
constants (a test may, being outside ``silly_kicks/``) and CI fails on any divergence, so the mirror
cannot drift. ``match_outcome``'s metric names are the keys of that package's ``dict`` metric constant;
its ``column_types`` mirrors the package's typed full-columns dict.

``xsuccess`` is deliberately ABSENT: it is a VAEP rating method (TF-61) that adds ``xsuccess`` /
``vaep_adjusted_value`` onto ``fct_action_values`` and emits no mart column-set constant. Its exclusion
is asserted-intentional in ``tests/test_metric_contracts.py`` (a ``*_METRIC_COLUMNS`` appearing on it
later would break the completeness gate and force a decision).

Enforcement (self-consistency, completeness, round-trip, output-faithfulness, ``column_types``
coverage): ``tests/test_metric_contracts.py``.

Examples
--------
A uniform schema-drift parity guard over every metric family::

    from silly_kicks.metric_contracts import METRIC_CONTRACTS

    for family, contract in METRIC_CONTRACTS.items():
        expected = set(contract["metric_columns"]) | set(contract["keys"])
        assert set(mart_columns(family)) == expected  # your mart-column accessor
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict


class MetricContract(TypedDict):
    """The uniform public output contract for one metric family.

    ``keys`` -- the mart grain; ``metric_columns`` -- value column names; ``columns`` -- the full
    ordered output names (keys + metric + provenance); ``column_types`` -- name -> dtype where the
    package declares a typed ``dict[str, str]`` full-columns mapping, else ``None``.

    Examples
    --------
    One family's contract (as stored in :data:`METRIC_CONTRACTS`)::

        {
            "keys": ("game_id", "team_id"),
            "metric_columns": ("ppda", "field_tilt"),
            "columns": ("game_id", "team_id", "ppda", "field_tilt"),
            "column_types": {"ppda": "double", "field_tilt": "double"},
        }
    """

    keys: tuple[str, ...]
    metric_columns: tuple[str, ...]
    columns: tuple[str, ...]
    column_types: Mapping[str, str] | None


#: The canonical output contract per column-emitting metric family. Hardcoded mirrors of each package's
#: public constants (see the module docstring); ``tests/test_metric_contracts.py`` CI-verifies equality.
METRIC_CONTRACTS: dict[str, MetricContract] = {
    "team_metrics": {
        "keys": ("game_id", "team_id"),
        "metric_columns": (
            "ppda",
            "defensive_intensity",
            "time_to_defensive_action_s",
            "time_to_recovery_s",
            "recoveries",
            "recoveries_within_ns_pct",
            "counterpress_regains",
            "counterpress_regain_pct",
            "field_tilt_pct",
            "pass_tempo",
            "long_ball_pct",
            "defensive_action_height_m",
            "recovery_line_height_m",
            "turnover_line_height_m",
            "poss_to_final_third_pct",
            "final_third_entries",
            "final_third_to_box_pct",
            "box_touches",
            "box_to_shot_pct",
            "shots",
            "high_opportunity_shots",
            "breakout_left",
            "breakout_center",
            "breakout_right",
            "breakout_left_pct",
            "breakout_center_pct",
            "breakout_right_pct",
            "possessions_retained_after_ns_pct",
            "buildup_final_quarter",
            "buildup_next_phase",
            "buildup_opp_int_own_half",
            "buildup_stayed_phase_one",
            "buildup_opp_won_own_half",
            "buildup_led_opp_shot",
            "buildup_success_pct",
            "post_regain_second_pass_pct",
            "post_regain_failed_first_passes",
            "post_regain_forward_first_pct",
            "switch_press_success_pct",
            "switch_press_n",
            "final_third_entries_post_recovery",
            "box_touches_post_recovery",
            "shots_post_recovery",
            "high_opportunity_shots_post_recovery",
        ),
        "columns": (
            "game_id",
            "team_id",
            "ppda",
            "defensive_intensity",
            "time_to_defensive_action_s",
            "time_to_recovery_s",
            "recoveries",
            "recoveries_within_ns_pct",
            "counterpress_regains",
            "counterpress_regain_pct",
            "field_tilt_pct",
            "pass_tempo",
            "long_ball_pct",
            "defensive_action_height_m",
            "recovery_line_height_m",
            "turnover_line_height_m",
            "poss_to_final_third_pct",
            "final_third_entries",
            "final_third_to_box_pct",
            "box_touches",
            "box_to_shot_pct",
            "shots",
            "high_opportunity_shots",
            "breakout_left",
            "breakout_center",
            "breakout_right",
            "breakout_left_pct",
            "breakout_center_pct",
            "breakout_right_pct",
            "possessions_retained_after_ns_pct",
            "buildup_final_quarter",
            "buildup_next_phase",
            "buildup_opp_int_own_half",
            "buildup_stayed_phase_one",
            "buildup_opp_won_own_half",
            "buildup_led_opp_shot",
            "buildup_success_pct",
            "post_regain_second_pass_pct",
            "post_regain_failed_first_passes",
            "post_regain_forward_first_pct",
            "switch_press_success_pct",
            "switch_press_n",
            "final_third_entries_post_recovery",
            "box_touches_post_recovery",
            "shots_post_recovery",
            "high_opportunity_shots_post_recovery",
        ),
        "column_types": {
            "game_id": "object",
            "team_id": "object",
            "ppda": "float64",
            "defensive_intensity": "float64",
            "time_to_defensive_action_s": "float64",
            "time_to_recovery_s": "float64",
            "recoveries": "Int64",
            "recoveries_within_ns_pct": "float64",
            "counterpress_regains": "Int64",
            "counterpress_regain_pct": "float64",
            "field_tilt_pct": "float64",
            "pass_tempo": "float64",
            "long_ball_pct": "float64",
            "defensive_action_height_m": "float64",
            "recovery_line_height_m": "float64",
            "turnover_line_height_m": "float64",
            "poss_to_final_third_pct": "float64",
            "final_third_entries": "Int64",
            "final_third_to_box_pct": "float64",
            "box_touches": "Int64",
            "box_to_shot_pct": "float64",
            "shots": "Int64",
            "high_opportunity_shots": "Int64",
            "breakout_left": "Int64",
            "breakout_center": "Int64",
            "breakout_right": "Int64",
            "breakout_left_pct": "float64",
            "breakout_center_pct": "float64",
            "breakout_right_pct": "float64",
            "possessions_retained_after_ns_pct": "float64",
            "buildup_final_quarter": "Int64",
            "buildup_next_phase": "Int64",
            "buildup_opp_int_own_half": "Int64",
            "buildup_stayed_phase_one": "Int64",
            "buildup_opp_won_own_half": "Int64",
            "buildup_led_opp_shot": "Int64",
            "buildup_success_pct": "float64",
            "post_regain_second_pass_pct": "float64",
            "post_regain_failed_first_passes": "Int64",
            "post_regain_forward_first_pct": "float64",
            "switch_press_success_pct": "float64",
            "switch_press_n": "Int64",
            "final_third_entries_post_recovery": "Int64",
            "box_touches_post_recovery": "Int64",
            "shots_post_recovery": "Int64",
            "high_opportunity_shots_post_recovery": "Int64",
        },
    },
    "match_outcome": {
        "keys": ("game_id", "team_id"),
        "metric_columns": ("p_win", "p_draw", "p_loss", "xpoints", "expected_goals"),
        "columns": ("game_id", "team_id", "p_win", "p_draw", "p_loss", "xpoints", "expected_goals"),
        "column_types": {
            "game_id": "object",
            "team_id": "object",
            "p_win": "float64",
            "p_draw": "float64",
            "p_loss": "float64",
            "xpoints": "float64",
            "expected_goals": "float64",
        },
    },
    "shot_stopping": {
        "keys": ("game_id", "player_id"),
        "metric_columns": (
            "shots_faced",
            "goals_conceded",
            "psxg_faced",
            "goals_prevented",
            "shots_faced_excl_penalties",
            "goals_conceded_excl_penalties",
            "psxg_faced_excl_penalties",
            "goals_prevented_excl_penalties",
        ),
        "columns": (
            "game_id",
            "player_id",
            "team_id",
            "shots_faced",
            "goals_conceded",
            "psxg_faced",
            "goals_prevented",
            "shots_faced_excl_penalties",
            "goals_conceded_excl_penalties",
            "psxg_faced_excl_penalties",
            "goals_prevented_excl_penalties",
        ),
        "column_types": {
            "game_id": "object",
            "player_id": "object",
            "team_id": "object",
            "shots_faced": "Int64",
            "goals_conceded": "Int64",
            "psxg_faced": "float64",
            "goals_prevented": "float64",
            "shots_faced_excl_penalties": "Int64",
            "goals_conceded_excl_penalties": "Int64",
            "psxg_faced_excl_penalties": "float64",
            "goals_prevented_excl_penalties": "float64",
        },
    },
    "gk_decision": {
        "keys": ("game_id", "keeper"),
        "metric_columns": ("decision_value", "chosen_ev", "best_ev", "sel_efficiency", "decision_pct"),
        "columns": (
            "game_id",
            "period_id",
            "decision_id",
            "keeper",
            "keeper_raw",
            "team_id",
            "decision_value",
            "chosen_ev",
            "best_ev",
            "sel_efficiency",
            "decision_pct",
            "n_options",
            "option_set_source",
        ),
        "column_types": None,
    },
    "territory": {
        "keys": ("game_id", "player_id"),
        "metric_columns": (
            "territory_xt_conceded",
            "territory_xt_prevented",
            "territory_xt_net",
            "territory_xt_conceded_forward",
            "territory_xt_prevented_forward",
            "territory_passes_into_hull",
            "territory_xt_conceded_rate",
            "territory_xt_prevented_rate",
            "territory_hull_area_m2",
            "territory_hull_centroid_x",
            "territory_hull_centroid_y",
            "territory_defensive_actions_in_hull",
        ),
        "columns": (
            "game_id",
            "player_id",
            "territory_xt_conceded",
            "territory_xt_prevented",
            "territory_xt_net",
            "territory_xt_conceded_forward",
            "territory_xt_prevented_forward",
            "territory_passes_into_hull",
            "territory_xt_conceded_rate",
            "territory_xt_prevented_rate",
            "territory_hull_area_m2",
            "territory_hull_centroid_x",
            "territory_hull_centroid_y",
            "territory_defensive_actions_in_hull",
            "territory_hull_source",
        ),
        "column_types": {
            "game_id": "object",
            "player_id": "object",
            "territory_xt_conceded": "float64",
            "territory_xt_prevented": "float64",
            "territory_xt_net": "float64",
            "territory_xt_conceded_forward": "float64",
            "territory_xt_prevented_forward": "float64",
            "territory_passes_into_hull": "Int64",
            "territory_xt_conceded_rate": "float64",
            "territory_xt_prevented_rate": "float64",
            "territory_hull_area_m2": "float64",
            "territory_hull_centroid_x": "float64",
            "territory_hull_centroid_y": "float64",
            "territory_defensive_actions_in_hull": "Int64",
            "territory_hull_source": "object",
        },
    },
    "duels": {
        "keys": ("game_id", "player_id"),
        "metric_columns": (
            "duel_rating",
            "duel_rating_deviation",
            "duel_volatility",
            "duels_contested",
            "duels_won",
            "duels_lost",
        ),
        "columns": (
            "game_id",
            "player_id",
            "duel_rating",
            "duel_rating_deviation",
            "duel_volatility",
            "duels_contested",
            "duels_won",
            "duels_lost",
            "duel_winner_source",
        ),
        "column_types": {
            "game_id": "object",
            "player_id": "object",
            "duel_rating": "float64",
            "duel_rating_deviation": "float64",
            "duel_volatility": "float64",
            "duels_contested": "Int64",
            "duels_won": "Int64",
            "duels_lost": "Int64",
            "duel_winner_source": "object",
        },
    },
    "restdefense": {
        "keys": ("game_id", "period_id", "team_id", "action_id"),
        "metric_columns": (
            "rd_num_superiority",
            "rd_num_superiority_gk",
            "rd_zone_occupancy",
            "rd_line_height",
            "rd_line_height_relative",
            "rd_compactness_x",
            "rd_width",
            "rd_depth",
            "rd_shape_2_3_vs_3_2",
            "rd_gk_line_height",
            "rd_gk_to_line_distance",
            "rd_attacker_space_control",
            "rd_danger_behind_line",
            "rd_danger_behind_line_gk",
            "rd_gk_coverage_behind_line",
            "rd_gk_reachable_coverage_m2",
        ),
        "columns": (
            "game_id",
            "period_id",
            "team_id",
            "action_id",
            "rd_num_superiority",
            "rd_num_superiority_gk",
            "rd_zone_occupancy",
            "rd_line_height",
            "rd_line_height_relative",
            "rd_compactness_x",
            "rd_width",
            "rd_depth",
            "rd_shape_2_3_vs_3_2",
            "rd_gk_line_height",
            "rd_gk_to_line_distance",
            "rd_attacker_space_control",
            "rd_danger_behind_line",
            "rd_danger_behind_line_gk",
            "rd_gk_coverage_behind_line",
            "rd_gk_reachable_coverage_m2",
        ),
        "column_types": None,
    },
    "positioning": {
        "keys": ("game_id", "team_id"),
        "metric_columns": ("positioning_gap", "threat_actual", "threat_optimum"),
        "columns": (
            "game_id",
            "period_id",
            "frame_id",
            "team_id",
            "positioning_gap",
            "threat_actual",
            "threat_optimum",
            "n_movable",
            "n_feasible_proposals",
            "sa_converged",
            "positioning_gap_source",
        ),
        "column_types": None,
    },
}

__all__ = ["METRIC_CONTRACTS", "MetricContract"]
