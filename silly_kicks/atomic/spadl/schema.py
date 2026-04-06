"""Atomic-SPADL output schema — plain Python constants."""

ATOMIC_SPADL_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "original_event_id": "object",
    "action_id": "int64",
    "period_id": "int64",
    "time_seconds": "float64",
    "team_id": "int64",
    "player_id": "int64",
    "x": "float64",
    "y": "float64",
    "dx": "float64",
    "dy": "float64",
    "type_id": "int64",
    "bodypart_id": "int64",
}

ATOMIC_SPADL_NAME_COLUMNS: dict[str, str] = {
    "type_name": "object",
    "bodypart_name": "object",
}
