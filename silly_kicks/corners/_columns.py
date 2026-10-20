"""Corner metric output columns (TF-46 Arm 1). Single source for the schema every gate iterates.

Grain for every corner table is ``(game_id, corner_id)``. The run-encoding and routine-feature
outputs are DATA-DRIVEN width (``n_initial x n_target`` run columns; ``r`` routine-feature columns),
so they are NOT committed as a fixed ``*_METRIC_COLUMNS`` constant and are deliberately excluded from
``metric_contracts`` (cf. the ``xsuccess`` exemption) -- their liveness/documentation is carried by the
corners liveness gate + the ``feature_glossary`` ``_corners_columns()`` leg, not the schema-drift
parity guard. The one registered mart is the fixed-width per-routine CHANCE-QUALITY mart.
"""

from __future__ import annotations

#: Grain: one row per (game_id, corner_id). game_id is SPADL-native; corner_id is a per-game
#: chronological positional index (NOT a cross-table FK -- join cross-table on the stable natural
#: key (game_id, period_id, t_kick); spec §13 / plan Task 2).
CORNER_KEYS = ["game_id", "corner_id"]

# --- identification table (identify_corners) -------------------------------------------------
CORNER_SIDE = "corner_side"
CORNER_KIND = "corner_kind"
CORNER_SOURCE = "corner_source"

#: Closed vocabularies (validated by tests/corners/test_columns.py).
CORNER_SIDE_VALUES = frozenset({"left", "right"})
CORNER_KIND_VALUES = frozenset({"direct", "short", "unclear"})
# ``skillcorner_marker`` (not ``skillcorner_gi``): SkillCorner corners come from the interruption
# MARKER, not the GI reception token (decision C7, spec section 9.4).
CORNER_SOURCE_VALUES = frozenset({"skillcorner_marker", "spadl_corner", "sb360_event"})

#: Full identification-table column order + dtype. ``type_source`` travels from the SPADL action's base
#: signal column (spec section 9.4): native / inferred / default / ``<NA>``.
CORNER_IDENTIFY_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "corner_id": "int64",
    "period_id": "int64",
    "t_kick": "float64",
    "attacking_team_id": "object",
    CORNER_SIDE: "category",
    "kick_x": "float64",
    "kick_y": "float64",
    CORNER_KIND: "category",
    CORNER_SOURCE: "category",
    "type_source": "object",
}

# --- window columns (add_corner_windows, additive onto the identification table) -------------
TARGET_SNAPSHOT_SOURCE = "target_snapshot_source"
TARGET_SNAPSHOT_SOURCE_VALUES = frozenset({"first_contact", "kick_plus_2s", "sb360_event"})

#: The columns add_corner_windows appends (names only; dtypes in CORNER_WINDOW_COLUMNS).
CORNER_WINDOW_COLUMNS: dict[str, str] = {
    "t_init": "float64",
    "t_target": "float64",
    "t_outcome": "float64",
    "first_contact_time": "float64",
    "first_possession_time": "float64",
    "winner_team_id": "object",
    "winner_player_id": "object",
    TARGET_SNAPSHOT_SOURCE: "category",
}

# --- per-routine chance quality (compute_corner_chance_quality) -- the registered mart -------
CC_N_SHOTS = "n_shots"
CC_N_GOALS = "n_goals"
CC_SUM_XG = "sum_xg"
CC_SOURCE = "chance_quality_source"

#: The derived metric columns (documented in feature_glossary; keys are structural).
CORNER_CHANCE_METRIC_COLUMNS = [
    CC_N_SHOTS,
    CC_N_GOALS,
    CC_SUM_XG,
]

#: Full chance-quality mart column order + dtype (counts nullable Int64; xG float64).
CORNER_CHANCE_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "corner_id": "int64",
    CC_N_SHOTS: "Int64",
    CC_N_GOALS: "Int64",
    CC_SUM_XG: "float64",
    CC_SOURCE: "category",
}
