"""columns_for_method -- per-method column/dtype resolver (SPEC-04 Task 7)."""

from __future__ import annotations

from silly_kicks.territory._columns import TERRITORY_COLUMNS, columns_for_method


def test_completed_failed_columns_are_exactly_v1():
    assert columns_for_method("completed_failed") == dict(TERRITORY_COLUMNS)  # byte-identical shape


def test_counterfactual_adds_only_the_five_cf_columns():
    base = set(TERRITORY_COLUMNS)
    cf = set(columns_for_method("counterfactual"))
    assert base <= cf
    assert cf - base == {
        "territory_expected_threat_faced",
        "territory_xt_prevented_above_expectation",
        "territory_passes_aimed_into_hull",
        "territory_mean_completion_faced",
        "territory_target_source",
    }


def test_counterfactual_cf_column_dtypes():
    cols = columns_for_method("counterfactual")
    assert cols["territory_expected_threat_faced"] == "float64"
    assert cols["territory_xt_prevented_above_expectation"] == "float64"
    assert cols["territory_passes_aimed_into_hull"] == "Int64"
    assert cols["territory_mean_completion_faced"] == "float64"
    assert cols["territory_target_source"] == "object"


def test_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        columns_for_method("bogus")
