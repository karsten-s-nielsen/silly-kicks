"""Row classification and column aggregation for the paired-leg comparison.

Two levels, because a raise is not a row property: if a call raises there is no output frame,
so there are no rows to classify. Call outcomes are resolved by the harness BEFORE this module
is reached; everything here assumes ``both_succeeded``.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

import math

import pandas as pd

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


class DtypeMismatchError(AssertionError):
    """Legs disagree on a column's value KIND. Never reconciled silently -- see ADR-019."""


class ShapeMismatchError(AssertionError):
    """Legs disagree on row count or index. Comparing anyway would use a silent prefix."""


_NUMERIC_INFERRED = frozenset({"integer", "floating", "mixed-integer-float", "decimal", "complex"})
_BOOL_INFERRED = frozenset({"boolean"})


def _value_kind(s: pd.Series) -> str | None:
    """Kind of a column's NON-NULL values, or None when it has none.

    Content-inferred rather than dtype-declared, so a NaN-forced upcast (``int64`` ->
    ``float64``, ``bool`` -> ``object``) cannot reach the comparison at all.
    """
    vals = s.dropna()
    if vals.empty:
        return None
    inferred = pd.api.types.infer_dtype(vals, skipna=True)
    if inferred in _NUMERIC_INFERRED:
        return "numeric"
    if inferred in _BOOL_INFERRED:
        return "boolean"
    return "other"


def classify_row(a, b, *, is_float: bool, rtol: float, atol: float) -> str:
    # NOTE: `math.isclose` semantics, NOT numpy's. math uses a SYMMETRIC
    # max(rel_tol*max(|a|,|b|), abs_tol); np.isclose uses the asymmetric atol + rtol*|b|.
    # Immaterial at 1e-9, but do not assume numpy behaviour when tuning a per-column override.
    a_nan = pd.isna(a)
    b_nan = pd.isna(b)
    if a_nan and b_nan:
        return "row_nan_both"
    if a_nan:
        return "row_nan_a"
    if b_nan:
        return "row_nan_b"
    if is_float:
        same = math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    else:
        same = bool(a == b)
    return "row_identical" if same else "row_differs"


def aggregate_column(counts: dict[str, int]) -> str:
    """Collapse a row-class tally to one observation, by declared precedence.

    ``row_nan_both`` rows are UNINFORMATIVE -- neither leg said anything -- so they leave the
    denominator rather than counting as agreement. That single choice handles both the
    unexercised column (all uninformative -> ``no_signal``) and the sparse-domain column
    (uninformative off-domain, compared on-domain -> its real observation). An earlier draft
    instead TIGHTENED ``identical`` and orphaned the second case entirely.
    """

    def get(rc: str) -> int:
        return int(counts.get(rc, 0))

    if get("row_nan_b"):
        return "leg_b_declined"

    total = sum(int(v) for v in counts.values())
    informative = total - get("row_nan_both")
    if informative == 0:
        return "no_signal"

    nan_a = get("row_nan_a")
    if nan_a == informative:
        return "all_nan"
    if nan_a:
        return "partial_nan"
    if get("row_differs"):
        return "differs"
    return "identical"


def compare_column(leg_a: pd.Series, leg_b: pd.Series, *, rtol: float, atol: float) -> tuple[str, dict[str, int]]:
    """Compare one column across legs. Shape and kind are checked BEFORE any value comparison."""
    if len(leg_a) != len(leg_b):
        raise ShapeMismatchError(
            f"leg A has {len(leg_a)} rows, leg B has {len(leg_b)} rows. Refusing to compare: "
            f"zip() would truncate to the shorter and report a confident observation computed "
            f"from a PREFIX. An aggregator dropping unlinked actions is the likely cause."
        )
    if not leg_a.index.equals(leg_b.index):
        raise ShapeMismatchError(
            "legs have equal length but a different index, so row i would be compared against "
            "a different action. Re-align before comparing."
        )

    # Compare the KIND of the actual values, never the declared dtype.
    #
    # The trap this guard exists for is ADR-019's int64-vs-object -- a numeric-versus-
    # non-numeric difference. int64 vs float64 is not that, and it is exactly what a NaN
    # forces: pandas cannot hold NaN in int64, so an integer column that declines on SOME
    # Leg A rows upcasts to float64 while Leg B stays int64. A declared-dtype guard fires on
    # that, aborting the audit on `partial_nan` -- the EXPECTED outcome on the visibility
    # axis, not an edge case. Inferring the kind from values subsumes all-NaN, partial-NaN,
    # bool->object AND the real trap under ONE rule, instead of an exemption per case.
    kind_a, kind_b = _value_kind(leg_a), _value_kind(leg_b)
    if kind_a is not None and kind_b is not None and kind_a != kind_b:
        raise DtypeMismatchError(
            f"leg A values are {kind_a} (dtype {leg_a.dtype}), leg B values are {kind_b} "
            f"(dtype {leg_b.dtype}). Not reconciled: an implicit cast is how a real ADR-019 "
            f"dtype defect reads as `identical`."
        )

    # A leg with no values contributes only row_nan_a/row_nan_both, so no value comparison
    # runs and the reference dtype is safely taken from the populated leg.
    is_float = pd.api.types.is_float_dtype(leg_a.dtype if kind_a is not None else leg_b.dtype)

    counts = {
        "row_identical": 0,
        "row_differs": 0,
        "row_nan_a": 0,
        "row_nan_b": 0,
        "row_nan_both": 0,
    }
    for a, b in zip(leg_a.to_numpy(), leg_b.to_numpy(), strict=True):
        counts[classify_row(a, b, is_float=is_float, rtol=rtol, atol=atol)] += 1
    return aggregate_column(counts), counts
