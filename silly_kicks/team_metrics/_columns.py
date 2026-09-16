"""team_metrics output columns (TF-52). Single source for the schema every gate iterates.

Grain: one output row per ``(game_id, team_id)`` -- two rows per match. Counts are nullable ``Int64``;
rates / times / heights are ``float64``; keys are ``object``-tolerant. The within-Ns post-recovery
companion columns (``*_post_recovery``) are appended by ``_compute`` (Task 6) from this same registry.
"""

from __future__ import annotations

#: Output grain keys (structural, not glossary metrics).
TEAM_KPI_KEYS = ["game_id", "team_id"]

# --- Pressing family (_pressing.py) ---
PRESSING_COLUMNS: dict[str, str] = {
    "ppda": "float64",
    "defensive_intensity": "float64",
    "time_to_defensive_action_s": "float64",
    "time_to_recovery_s": "float64",
    "recoveries": "Int64",
    "recoveries_within_ns_pct": "float64",
    "counterpress_regains": "Int64",
    "counterpress_regain_pct": "float64",
}

# --- Progression family (_progression.py) ---
PROGRESSION_COLUMNS: dict[str, str] = {
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
}

# --- Build-up family (_buildup.py) ---
BUILDUP_COLUMNS: dict[str, str] = {
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
}

#: The offensive OUTPUT metrics re-computed inside the post-recovery window (spec Section 4.5),
#: emitted as ``<name>_post_recovery`` companions by ``_compute`` -- the transition-output block a team
#: generates within N s of winning the ball: final-third entries, box touches, shots, high-value shots.
#: **Per-action COUNTS only.** A possession-level event (a *breakout* is one channel per possession) or
#: a possession-rate (field tilt, tempo, conversion %) has no meaningful restriction to a per-action
#: time-window, so neither is companioned -- excluded on correctness grounds, not as a scope cut.
POST_RECOVERY_SOURCE_COLUMNS = [
    "final_third_entries",
    "box_touches",
    "shots",
    "high_opportunity_shots",
]
POST_RECOVERY_COLUMNS: dict[str, str] = {
    f"{name}_post_recovery": BUILDUP_COLUMNS.get(name) or PROGRESSION_COLUMNS[name]
    for name in POST_RECOVERY_SOURCE_COLUMNS
}

#: All derived metric columns, in output order (documented in feature_glossary).
_METRIC_DICTS = (PRESSING_COLUMNS, PROGRESSION_COLUMNS, BUILDUP_COLUMNS, POST_RECOVERY_COLUMNS)
TEAM_KPI_METRIC_COLUMNS: list[str] = [name for d in _METRIC_DICTS for name in d]

#: Full output column order + dtype (keys object-tolerant; then every metric).
TEAM_KPI_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "team_id": "object",
    **PRESSING_COLUMNS,
    **PROGRESSION_COLUMNS,
    **BUILDUP_COLUMNS,
    **POST_RECOVERY_COLUMNS,
}
