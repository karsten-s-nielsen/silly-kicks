"""Column-set constants for the per-action win-probability output.

Per-action (feature-grain), NOT a per-(entity, match) mart family -- so this constant is deliberately
named ``WIN_PROBABILITY_COLUMNS`` (NOT ``*_METRIC_COLUMNS``): the SK-EXPORT / ADR-098 metric-contracts
completeness gate enrolls only ``*_METRIC_COLUMNS`` exporters, so this package stays exempt (like
``xsuccess``). The per-action columns are documented in ``feature_glossary`` (ADR-048) instead.
"""

from __future__ import annotations

WIN_PROBABILITY_KEYS = ("game_id", "action_id")

# win_prob_source closed vocabulary: {"scored", "unresolved_state", "excluded_not_two_teams"}
WIN_PROBABILITY_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "action_id": "int64",
    "team_id": "object",
    "period_id": "int64",
    "p_win": "float64",
    "p_draw": "float64",
    "p_loss": "float64",
    "win_prob_leverage": "float64",
    "win_prob_source": "object",
}
