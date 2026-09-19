from silly_kicks.territory import _columns as C


def test_completed_failed_schema_is_v1_verbatim():
    assert C.columns_for_method("completed_failed") == dict(C.TERRITORY_COLUMNS)


def test_counterfactual_adds_exactly_five_columns():
    extra = set(C.columns_for_method("counterfactual")) - set(C.TERRITORY_COLUMNS)
    assert extra == {
        "territory_expected_threat_faced",
        "territory_xt_prevented_above_expectation",
        "territory_passes_aimed_into_hull",
        "territory_mean_completion_faced",
        "territory_target_source",
    }


def test_method_set_has_both():
    assert C.TERRITORY_METHODS == frozenset({"completed_failed", "counterfactual"})


def test_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        C.columns_for_method("nope")
