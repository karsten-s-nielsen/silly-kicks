"""Column names + closed provenance vocab for the GK-decision metric (TF-62)."""

from __future__ import annotations

OPTION_SET_SOURCE_VALUES: tuple[str, ...] = ("native", "reconstructed")
GK_DECISION_DROP_REASONS: tuple[str, ...] = (
    "too_few_options",
    "no_unique_chosen",
    "chosen_unvalued",
    "no_frame",  # reconstruction: no linked freeze-frame / unresolvable frame (PR2)
    "fov_cropped",  # reconstruction: keeper neighbourhood under-observed (PR2; ADR-077)
)

#: uniform option-rows schema (every OptionSet adapter yields exactly these)
OPTION_ROW_COLUMNS: tuple[str, ...] = (
    "game_id",
    "period_id",
    "decision_id",
    "keeper_id",
    "team_id",
    "is_chosen",
    "completion",
    "opponents_bypassed",
    "option_set_source",
)

#: the DERIVED metric columns (glossaried, ADR-048) -- a subset of the samples (the keys/provenance
#: game_id/period_id/decision_id/keeper/keeper_raw/team_id/n_options/option_set_source are not features).
GK_DECISION_METRIC_COLUMNS: tuple[str, ...] = (
    "decision_value",
    "chosen_ev",
    "best_ev",
    "sel_efficiency",
    "decision_pct",
)

#: per-decision samples emitted by compute_gk_decision_value
GK_DECISION_SAMPLE_COLUMNS: tuple[str, ...] = (
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
)
