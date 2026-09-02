from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.keeper_identity import (
    KEEPER_APPEARANCE_COLUMNS,
    KEEPER_APPEARANCE_SOURCE_VALUES,
    validate_keeper_appearances,
)


def _appearances() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "team_id": pd.array([10, 20], dtype="Int64"),
            "player_id": pd.array([901, 902], dtype="Int64"),
            "period_id": [1, 1],
            "start_time_seconds": [0.0, 0.0],
            "end_time_seconds": [np.inf, np.inf],
            "source": ["starting_xi", "starting_xi"],
        }
    )


def test_valid_appearances_round_trip():
    df = validate_keeper_appearances(_appearances())
    assert list(df.columns) == list(KEEPER_APPEARANCE_COLUMNS)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing"):
        validate_keeper_appearances(_appearances().drop(columns=["period_id"]))


def test_negative_start_raises():
    bad = _appearances()
    bad.loc[0, "start_time_seconds"] = -1.0
    with pytest.raises(ValueError, match=r"(?i)start_time_seconds"):
        validate_keeper_appearances(bad)


def test_start_after_end_raises():
    bad = _appearances()
    bad.loc[0, "start_time_seconds"] = 100.0
    bad.loc[0, "end_time_seconds"] = 10.0
    with pytest.raises(ValueError, match=r"start.*end"):
        validate_keeper_appearances(bad)


def test_unknown_source_raises():
    bad = _appearances()
    bad.loc[0, "source"] = "not_a_real_source"
    with pytest.raises(ValueError, match=r"(?i)unknown.*source"):
        validate_keeper_appearances(bad)


def test_source_vocab_is_closed():
    assert set(KEEPER_APPEARANCE_SOURCE_VALUES) == {
        "native_intervals",
        "sub_events",
        "starting_xi",
        "emergency_keeper",
    }


def test_columns_schema_shape_and_object_ids():
    # The three ids are object (string-tolerant), NOT Int64 -- DFL/SkillCorner ids are strings.
    assert KEEPER_APPEARANCE_COLUMNS == {
        "game_id": "object",
        "team_id": "object",
        "player_id": "object",
        "period_id": "int64",
        "start_time_seconds": "float64",
        "end_time_seconds": "float64",
        "source": "object",
    }


def test_string_ids_are_tolerated():
    # DFL ids are strings (MatchInfo.gk_player_ids: frozenset[str]); the port must accept them un-coerced.
    df = _appearances().copy()
    df["team_id"] = ["DFL-CLU-00000G", "DFL-CLU-00000P"]
    df["player_id"] = ["DFL-OBJ-0027AX", "DFL-OBJ-0027V2"]
    out = validate_keeper_appearances(df)
    assert list(out["player_id"]) == ["DFL-OBJ-0027AX", "DFL-OBJ-0027V2"]  # no Int64 coercion / no raise
