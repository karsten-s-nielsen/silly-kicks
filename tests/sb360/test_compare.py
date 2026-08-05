"""Unit tests for the paired-leg comparison primitive.

Everything downstream trusts this module, so its edge cases are pinned here rather than
discovered during the audit.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from tests.sb360 import _vocabulary as V
from tests.sb360._compare import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    DtypeMismatchError,
    ShapeMismatchError,
    aggregate_column,
    classify_row,
    compare_column,
)

_NAN = float("nan")


def test_row_classification_is_exhaustive_over_the_finite_nan_grid():
    """Every (Leg A, Leg B) combination lands in exactly one declared row class."""
    values = [1.0, 2.0, _NAN]
    seen = set()
    for a, b in itertools.product(values, values):
        cls = classify_row(a, b, is_float=True, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
        assert cls in V.ROW_CLASSES, f"({a}, {b}) produced undeclared class {cls!r}"
        seen.add(cls)
    assert seen == V.ROW_CLASSES, f"unreached row classes: {sorted(V.ROW_CLASSES - seen)}"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1.0, 1.0, "row_identical"),
        (1.0, 2.0, "row_differs"),
        (_NAN, 1.0, "row_nan_a"),
        (1.0, _NAN, "row_nan_b"),
        (_NAN, _NAN, "row_nan_both"),
    ],
)
def test_row_classification_cases(a, b, expected):
    assert classify_row(a, b, is_float=True, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL) == expected


def test_non_float_compares_exactly():
    """A tolerance on an integer count would silently absorb an off-by-one."""
    assert classify_row(3, 4, is_float=False, rtol=1.0, atol=1.0) == "row_differs"
    assert classify_row(3, 3, is_float=False, rtol=0.0, atol=0.0) == "row_identical"


def test_aggregation_precedence_leg_b_declined_beats_everything_else():
    counts = {
        "row_identical": 100,
        "row_differs": 0,
        "row_nan_a": 5,
        "row_nan_b": 1,
        "row_nan_both": 3,
    }
    assert aggregate_column(counts) == "leg_b_declined"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"row_nan_both": 4}, "no_signal"),
        ({"row_nan_a": 3, "row_nan_both": 2}, "all_nan"),
        ({"row_nan_a": 3, "row_identical": 2}, "partial_nan"),
        ({"row_differs": 1, "row_identical": 5}, "differs"),
        ({"row_identical": 5}, "identical"),
        # The sparse-domain column: NaN in BOTH legs off-domain, identical on-domain. An
        # earlier draft tightened `identical` and orphaned exactly this.
        ({"row_identical": 2, "row_nan_both": 40}, "identical"),
    ],
)
def test_aggregation_cases(counts, expected):
    full = {rc: 0 for rc in V.ROW_CLASSES}
    full.update(counts)
    assert aggregate_column(full) == expected


def test_aggregation_is_total_over_reachable_tallies():
    """No tally of row classes falls through to a default."""
    for combo in itertools.product([0, 1, 2], repeat=len(V.ROW_CLASSES)):
        counts = dict(zip(sorted(V.ROW_CLASSES), combo, strict=True))
        if sum(counts.values()) == 0:
            continue
        obs = aggregate_column(counts)
        assert obs in V.OBSERVATIONS, f"tally {counts} produced undeclared observation {obs!r}"


def test_dtype_mismatch_fails_loudly_rather_than_casting():
    """int64 vs object is the ADR-019 trap: an implicit cast makes a real defect read identical."""
    a = pd.Series([1, 2, 3], dtype="int64")
    b = pd.Series(["1", "2", "3"], dtype="object")
    with pytest.raises(DtypeMismatchError, match=r"numeric.*other"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_all_nan_upcast_is_not_a_dtype_mismatch():
    """An integer count that DECLINES on freeze-frames is the desirable `all_nan` outcome.

    pandas cannot hold NaN in int64, so Leg A upcasts to float64 against an int64 Leg B.
    """
    a = pd.Series([np.nan, np.nan, np.nan], dtype="float64")
    b = pd.Series([1, 2, 3], dtype="int64")
    obs, counts = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "all_nan"
    assert counts["row_nan_a"] == 3


def test_partial_nan_upcast_is_not_a_dtype_mismatch():
    """The ADJACENT case, and the one an all-NaN-only exemption misses.

    An integer column declining on SOME rows leaves BOTH legs populated with different
    declared dtypes, so a guard exempting only the all-NaN case fires here -- aborting on
    `partial_nan`, the expected outcome on the visibility axis.
    """
    a = pd.Series([1.0, np.nan, 3.0], dtype="float64")
    b = pd.Series([1, 2, 3], dtype="int64")
    obs, counts = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "partial_nan"
    assert counts["row_nan_a"] == 1
    assert counts["row_identical"] == 2


def test_bool_column_declining_to_object_is_not_a_dtype_mismatch():
    """numpy bool cannot hold NaN either, so a declining bool column becomes object."""
    a = pd.Series([True, None, False], dtype="object")
    b = pd.Series([True, True, False], dtype="bool")
    obs, _ = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "partial_nan"


def test_numeric_versus_string_still_raises_when_partially_nan():
    """The exemption must not swallow the trap it exists for."""
    a = pd.Series([1.0, np.nan, 3.0], dtype="float64")
    b = pd.Series(["1", "2", "3"], dtype="object")
    with pytest.raises(DtypeMismatchError, match=r"numeric.*other"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_length_mismatch_fails_rather_than_truncating():
    """zip() truncates to the shorter series and reports a confident observation computed
    from a PREFIX -- the audit's core primitive carrying the defect class the audit exists
    to find."""
    a = pd.Series([1.0, 2.0])
    b = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ShapeMismatchError, match=r"2 rows.*3 rows"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_index_mismatch_fails_even_at_equal_length():
    """Equal length is not enough: a re-indexed leg compares row i against a different action
    while every value still lines up positionally."""
    a = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    b = pd.Series([1.0, 2.0, 3.0], index=[0, 2, 1])
    with pytest.raises(ShapeMismatchError, match=r"index"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_compare_column_returns_observation_and_tally():
    a = pd.Series([1.0, np.nan, 3.0])
    b = pd.Series([1.0, 2.0, 3.0])
    obs, counts = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "partial_nan"
    assert counts["row_nan_a"] == 1
    assert counts["row_identical"] == 2


def test_both_all_nan_is_no_signal_not_identical():
    """A column that produced nothing anywhere is UNEXERCISED, not working."""
    a = pd.Series([np.nan, np.nan], dtype="float64")
    b = pd.Series([np.nan, np.nan], dtype="float64")
    obs, _ = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "no_signal"
