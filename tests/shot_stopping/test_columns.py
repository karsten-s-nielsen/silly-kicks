from __future__ import annotations

from silly_kicks.shot_stopping import SHOT_STOPPING_COLUMNS, SHOT_STOPPING_METRIC_COLUMNS, SS_KEYS


def test_keys_and_metric_columns():
    assert SS_KEYS == ["game_id", "player_id"]
    assert len(SHOT_STOPPING_METRIC_COLUMNS) == 8
    assert list(SHOT_STOPPING_COLUMNS)[:3] == ["game_id", "player_id", "team_id"]
    for c in SHOT_STOPPING_METRIC_COLUMNS:
        assert c in SHOT_STOPPING_COLUMNS
    assert SHOT_STOPPING_COLUMNS["shots_faced"] == "Int64"
    assert SHOT_STOPPING_COLUMNS["goals_prevented"] == "float64"
