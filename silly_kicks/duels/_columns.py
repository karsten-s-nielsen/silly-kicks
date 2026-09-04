"""Glicko-2 duel-rating output columns (TF-55). Single source for the schema every gate iterates.

Grain: one output row per (game_id, player_id) -- the per-match rating snapshot (rating period = match).
A window slices the trajectory (spec §5b.4).
"""

from __future__ import annotations

#: Grain keys.
DUEL_KEYS = ["game_id", "player_id"]

# Metric column names (spec §5b.5). Rating / RD / volatility -> float64; counts -> Int64.
DU_RATING = "duel_rating"
DU_RATING_DEVIATION = "duel_rating_deviation"
DU_VOLATILITY = "duel_volatility"
DU_CONTESTED = "duels_contested"
DU_WON = "duels_won"
DU_LOST = "duels_lost"

#: Provenance: native (sportec tackle_winner/loser) vs derived (tackle/take_on adjacency).
DU_WINNER_SOURCE = "duel_winner_source"

#: The derived metric columns (documented in feature_glossary; keys + provenance are structural).
DUEL_METRIC_COLUMNS = [
    DU_RATING,
    DU_RATING_DEVIATION,
    DU_VOLATILITY,
    DU_CONTESTED,
    DU_WON,
    DU_LOST,
]

#: Full output column order + dtype (keys/provenance object; counts nullable Int64; rating/RD/vol float64).
DUEL_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "player_id": "object",
    DU_RATING: "float64",
    DU_RATING_DEVIATION: "float64",
    DU_VOLATILITY: "float64",
    DU_CONTESTED: "Int64",
    DU_WON: "Int64",
    DU_LOST: "Int64",
    DU_WINNER_SOURCE: "object",
}

#: Closed provenance vocabulary for DU_WINNER_SOURCE.
DUEL_WINNER_SOURCE_VALUES = frozenset({"native", "derived"})
