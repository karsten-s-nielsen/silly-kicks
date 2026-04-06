"""Tests for output contract utilities, input validation, and add_names preservation."""

import warnings

import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlcfg
from silly_kicks.spadl.schema import SPADL_COLUMNS
from silly_kicks.spadl.utils import (
    _finalize_output,
    _validate_input_columns,
    validate_spadl,
)


def _make_valid_spadl_df(n: int = 3) -> pd.DataFrame:
    """Create a minimal valid SPADL DataFrame for testing."""
    return pd.DataFrame({
        "game_id": [1] * n,
        "original_event_id": ["ev1", "ev2", "ev3"][:n],
        "action_id": list(range(n)),
        "period_id": [1] * n,
        "time_seconds": [0.0, 1.0, 2.0][:n],
        "team_id": [100] * n,
        "player_id": [200] * n,
        "start_x": [50.0] * n,
        "start_y": [34.0] * n,
        "end_x": [60.0] * n,
        "end_y": [34.0] * n,
        "type_id": [spadlcfg.actiontype_id["pass"]] * n,
        "result_id": [spadlcfg.result_id["success"]] * n,
        "bodypart_id": [spadlcfg.bodypart_id["foot"]] * n,
    })


class TestFinalizeOutput:
    def test_selects_declared_columns_only(self):
        df = _make_valid_spadl_df()
        df["extra_col"] = "should_be_dropped"
        result = _finalize_output(df)
        assert list(result.columns) == list(SPADL_COLUMNS.keys())
        assert "extra_col" not in result.columns

    def test_enforces_dtypes(self):
        df = _make_valid_spadl_df()
        df["game_id"] = df["game_id"].astype(float)
        result = _finalize_output(df)
        assert result["game_id"].dtype == "int64"

    def test_original_event_id_is_object(self):
        df = _make_valid_spadl_df()
        result = _finalize_output(df)
        assert result["original_event_id"].dtype == object


class TestValidateInputColumns:
    def test_missing_column_raises(self):
        df = pd.DataFrame({"game_id": [1], "event_id": [1]})
        required = {"game_id", "event_id", "period_id"}
        with pytest.raises(ValueError, match="period_id"):
            _validate_input_columns(df, required, provider="Test")

    def test_extra_columns_accepted(self):
        df = pd.DataFrame({"game_id": [1], "event_id": [1], "bonus": [True]})
        required = {"game_id", "event_id"}
        _validate_input_columns(df, required, provider="Test")

    def test_error_message_includes_provider(self):
        df = pd.DataFrame({"game_id": [1]})
        required = {"game_id", "missing_col"}
        with pytest.raises(ValueError, match="StatsBomb"):
            _validate_input_columns(df, required, provider="StatsBomb")


class TestValidateSpadl:
    def test_valid_dataframe_passes(self):
        df = _make_valid_spadl_df()
        result = validate_spadl(df)
        assert result is df

    def test_missing_column_raises(self):
        df = _make_valid_spadl_df().drop(columns=["type_id"])
        with pytest.raises(ValueError, match="type_id"):
            validate_spadl(df)

    def test_dtype_mismatch_warns(self):
        df = _make_valid_spadl_df()
        df["game_id"] = df["game_id"].astype(float)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_spadl(df)
            dtype_warnings = [x for x in w if "dtype" in str(x.message).lower()]
            assert len(dtype_warnings) >= 1

    def test_extra_columns_accepted(self):
        df = _make_valid_spadl_df()
        df["custom_col"] = "hello"
        validate_spadl(df)


class TestAddNamesPreservesExtraColumns:
    def test_extra_columns_preserved(self):
        from silly_kicks.spadl.utils import add_names
        df = _make_valid_spadl_df()
        df["my_custom_col"] = [10, 20, 30]
        result = add_names(df)
        assert "my_custom_col" in result.columns
        assert list(result["my_custom_col"]) == [10, 20, 30]

    def test_name_columns_added(self):
        from silly_kicks.spadl.utils import add_names
        df = _make_valid_spadl_df()
        result = add_names(df)
        assert "type_name" in result.columns
        assert "result_name" in result.columns
        assert "bodypart_name" in result.columns

    def test_atomic_add_names_preserves_extra_columns(self):
        import silly_kicks.atomic.spadl.config as atomicconfig
        from silly_kicks.atomic.spadl.utils import add_names as atomic_add_names
        df = pd.DataFrame({
            "game_id": [1],
            "original_event_id": ["ev1"],
            "action_id": [0],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": [100],
            "player_id": [200],
            "x": [50.0],
            "y": [34.0],
            "dx": [10.0],
            "dy": [0.0],
            "type_id": [atomicconfig.actiontype_id["pass"]],
            "bodypart_id": [atomicconfig.bodypart_id["foot"]],
            "my_custom_col": [42],
        })
        result = atomic_add_names(df)
        assert "my_custom_col" in result.columns
        assert result["my_custom_col"].iloc[0] == 42
