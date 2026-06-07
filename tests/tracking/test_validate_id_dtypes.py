"""Tests for the public validate_id_dtypes guard + IdDtypeDiagnosis (ADR-019)."""

import numpy as np  # noqa: F401
import pandas as pd
import pytest

from silly_kicks.tracking import IdDtypeDiagnosis, validate_id_dtypes


def _actions(team_dtype, player_dtype):
    return pd.DataFrame(
        {
            "action_id": [0, 1],
            "period_id": [1, 1],
            "team_id": pd.Series([5, 5], dtype=team_dtype),
            "player_id": pd.Series([10, 11], dtype=player_dtype),
        }
    )


def _frames(team_dtype, player_dtype):
    return pd.DataFrame(
        {
            "period_id": [1, 1],
            "frame_id": [0, 0],
            "team_id": pd.Series([5, 6], dtype=team_dtype),
            "player_id": pd.Series([10, 20], dtype=player_dtype),
            "is_ball": [False, False],
        }
    )


def test_matched_dtypes_no_mismatch():
    diag = validate_id_dtypes(_actions("int64", "int64"), _frames("int64", "int64"), on_mismatch="raise")
    assert isinstance(diag, IdDtypeDiagnosis)
    assert not diag.has_mismatch


def test_mismatch_raises_by_default():
    with pytest.raises(ValueError, match="id dtype"):
        validate_id_dtypes(_actions("int64", "int64"), _frames("object", "object"))


def test_mismatch_warn_returns_diag():
    with pytest.warns(UserWarning, match="id dtype"):
        diag = validate_id_dtypes(_actions("int64", "int64"), _frames("object", "object"), on_mismatch="warn")
    assert diag.has_mismatch
    assert "team_id" in diag.coercion_required_columns


def test_home_team_id_axis_flagged():
    diag = validate_id_dtypes(
        _actions("int64", "int64"), _frames("int64", "int64"), home_team_id="5", on_mismatch="ignore"
    )
    assert diag.home_team_id_requires_coercion
