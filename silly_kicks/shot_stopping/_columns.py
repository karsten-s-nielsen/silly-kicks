"""Shot-stopping metric output columns (TF-59 PR2). Single source for the schema every gate iterates."""

from __future__ import annotations

#: Grain: one output row per (game_id, defending keeper player_id).
SS_KEYS = ["game_id", "player_id"]

#: The keeper's team, carried onto each output row (authoritative, from defending_gk_team_id).
SS_TEAM_ID = "team_id"

# Metric column names (spec §6.1). Counts -> Int64; PSxG / goals-prevented -> float64.
SS_SHOTS_FACED = "shots_faced"
SS_GOALS_CONCEDED = "goals_conceded"
SS_PSXG_FACED = "psxg_faced"
SS_GOALS_PREVENTED = "goals_prevented"  # == GSAA: sum(psxg_faced) - goals_conceded
SS_SHOTS_FACED_EXCL_PEN = "shots_faced_excl_penalties"
SS_GOALS_CONCEDED_EXCL_PEN = "goals_conceded_excl_penalties"
SS_PSXG_FACED_EXCL_PEN = "psxg_faced_excl_penalties"
SS_GOALS_PREVENTED_EXCL_PEN = "goals_prevented_excl_penalties"

#: The derived metric columns (documented in feature_glossary; the sample keys + team are structural).
SHOT_STOPPING_METRIC_COLUMNS = [
    SS_SHOTS_FACED,
    SS_GOALS_CONCEDED,
    SS_PSXG_FACED,
    SS_GOALS_PREVENTED,
    SS_SHOTS_FACED_EXCL_PEN,
    SS_GOALS_CONCEDED_EXCL_PEN,
    SS_PSXG_FACED_EXCL_PEN,
    SS_GOALS_PREVENTED_EXCL_PEN,
]

#: Full output column order + dtype (counts nullable Int64; PSxG/GP float64; keys/team object-tolerant).
SHOT_STOPPING_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "player_id": "object",
    SS_TEAM_ID: "object",
    SS_SHOTS_FACED: "Int64",
    SS_GOALS_CONCEDED: "Int64",
    SS_PSXG_FACED: "float64",
    SS_GOALS_PREVENTED: "float64",
    SS_SHOTS_FACED_EXCL_PEN: "Int64",
    SS_GOALS_CONCEDED_EXCL_PEN: "Int64",
    SS_PSXG_FACED_EXCL_PEN: "float64",
    SS_GOALS_PREVENTED_EXCL_PEN: "float64",
}
