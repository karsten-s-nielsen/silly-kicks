"""Tests for ``silly_kicks.atomic.spadl.utils.validate_atomic_spadl`` (1.5.0)."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from silly_kicks.atomic.spadl.schema import ATOMIC_SPADL_COLUMNS
from silly_kicks.atomic.spadl.utils import validate_atomic_spadl
from tests.atomic._atomic_test_fixtures import _df, _make_atomic_action


class TestValidateAtomicSpadl:
    def test_returns_input_unchanged(self):
        actions = _df([_make_atomic_action(action_id=0)])
        result = validate_atomic_spadl(actions)
        assert isinstance(result, pd.DataFrame)
        # Same row count + same columns.
        assert len(result) == len(actions)
        for col in ATOMIC_SPADL_COLUMNS:
            assert col in result.columns

    def test_passes_for_valid_input(self):
        actions = _df([_make_atomic_action(action_id=0), _make_atomic_action(action_id=1)])
        # Should not raise.
        validate_atomic_spadl(actions)

    def test_raises_on_missing_required_column(self):
        actions = _df([_make_atomic_action(action_id=0)]).drop(columns=["x"])
        with pytest.raises(ValueError, match=r"Missing"):
            validate_atomic_spadl(actions)

    def test_raises_on_missing_multiple_columns(self):
        actions = _df([_make_atomic_action(action_id=0)]).drop(columns=["x", "y"])
        with pytest.raises(ValueError, match=r"Missing"):
            validate_atomic_spadl(actions)

    def test_warns_on_dtype_mismatch(self):
        actions = _df([_make_atomic_action(action_id=0)])
        # Coerce x to int (should be float64).
        actions["x"] = actions["x"].astype("int64")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_atomic_spadl(actions)
            assert any("dtype" in str(warning.message).lower() for warning in w)

    def test_chains_for_pipeline_use(self):
        # validate_atomic_spadl returns its input — chains naturally with other helpers.
        actions = _df([_make_atomic_action(action_id=0)])
        result = validate_atomic_spadl(actions)
        # Same instance/values.
        assert result.equals(actions)

    def test_custom_schema(self):
        # Pass a custom schema: {"x": "float64"} only — passes when only x is present.
        actions = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        validate_atomic_spadl(actions, schema={"x": "float64"})

    def test_extra_columns_are_allowed(self):
        # validate_atomic_spadl checks REQUIRED columns — extra columns don't break it.
        actions = _df([_make_atomic_action(action_id=0)])
        actions["my_extra"] = "preserved"
        # Should not raise.
        validate_atomic_spadl(actions)
