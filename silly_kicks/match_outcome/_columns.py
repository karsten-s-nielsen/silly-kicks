"""match_outcome output columns (TF-53). Single source for the schema every gate iterates.

Grain: one output row per ``(game_id, team_id)`` -- two rows per match. Keys are ``object``-tolerant;
the outcome probabilities + xPoints + expected goals are ``float64``.
"""

from __future__ import annotations

#: Output grain keys (structural, not glossary metrics).
MATCH_OUTCOME_KEYS = ["game_id", "team_id"]

#: Derived metric columns (documented in feature_glossary), in output order.
MATCH_OUTCOME_METRIC_COLUMNS: dict[str, str] = {
    "p_win": "float64",
    "p_draw": "float64",
    "p_loss": "float64",
    "xpoints": "float64",
    "expected_goals": "float64",
}

#: Full output column order + dtype (keys object-tolerant; then every metric).
MATCH_OUTCOME_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "team_id": "object",
    **MATCH_OUTCOME_METRIC_COLUMNS,
}
