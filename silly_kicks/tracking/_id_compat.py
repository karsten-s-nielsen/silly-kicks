"""Dtype-safe id identity for tracking-feature seams (ADR-019).

Defines "id identity" once: a single canonicalization truth (scalar + vectorized),
comparison helpers, and pre-merge join-key alignment. All comparison/align helpers
have a same-dtype fast path (zero cost when both sides already share a numpy kind, or
both are object), so pure-library matched pipelines pay nothing.

See docs/superpowers/specs/2026-06-06-tracking-id-dtype-contract-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _canonical(x):
    """Single-element canonical id key. NA/None/NaN -> pd.NA (never matches).

    Integral numerics collapse: 366, 366.0, np.int64(366), Int64(366), "366" -> "366".
    Genuine strings pass through. Non-integral floats stringify as-is (won't match).
    """
    if x is None or x is pd.NA:
        return pd.NA
    # floats / numpy floats: NaN -> NA; integral -> collapse to int string
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return pd.NA
        if float(x).is_integer():
            return str(int(x))
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def canonical_id(x):
    """Scalar entry point -- delegates to the single ``_canonical`` truth."""
    return _canonical(x)


def canonical_id_series(s: pd.Series) -> pd.Series:
    """Vectorized canonicalization, NaN-safe, object dtype. Matches ``canonical_id``
    element-wise. Avoids naive ``.astype(str)`` (which yields '366.0', not '366')."""
    s = pd.Series(s)
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    notna = s.notna()
    if not notna.any():
        return out
    vals = s[notna]
    kind = vals.dtype.kind
    if kind == "f":
        integral = vals == vals.round()
        out.loc[vals.index[integral]] = vals[integral].astype("Int64").astype("string").astype("object")
        out.loc[vals.index[~integral]] = vals[~integral].astype("string").astype("object")
    elif kind in ("i", "u") or str(vals.dtype) in ("Int64", "Int32", "Int16", "Int8"):
        out.loc[vals.index] = vals.astype("Int64").astype("string").astype("object")
    else:
        # object/string: VECTORIZED stringify (A1 -- no Python loop). Real object id columns
        # are homogeneous strings (genuine ids or stringified ints); astype("string") matches
        # ``_canonical`` for str/object-int/non-integral-float elements. The ONLY divergence is
        # an *integral float stored in an object column* (366.0 -> "366.0" vs _canonical's
        # "366") -- a doubly-pathological case (float ids in object dtype) that does not occur
        # for real id columns; documented, not Python-looped.
        out.loc[vals.index] = vals.astype("string").astype("object")
    return out


def _directly_comparable(a_dtype, b_dtype) -> bool:
    """True when two columns can be raw-compared/merged WITHOUT canonicalization:
    same numpy kind (Int64 nullable and int64 both report 'i'), OR both object.
    Both-object is safe under the same-provider invariant: two same-provider object
    id columns are both genuine strings, so raw ==/!= is correct -- and it avoids the
    object Python-loop regression for sportec/kloppy."""
    ak, bk = a_dtype.kind, b_dtype.kind
    if ak == "O" and bk == "O":
        return True
    return ak == bk and ak in ("i", "u", "f", "b")


def _merge_compatible(a_dtype, b_dtype) -> bool:
    """True when two columns can be used as a pd.merge key WITHOUT canonicalization.
    pandas raises only on numeric-vs-object keys; numeric-vs-numeric (int64/Int64/float64,
    e.g. an Int64 linker frame_id vs a float64 frames frame_id) merges fine on value, and
    object-vs-object merges fine. Broader than `_directly_comparable` (which governs
    element-wise comparison): stringifying a mergeable numeric key would needlessly change
    the merge representation (and, for non-integer float artifacts, its cardinality)."""
    ak, bk = a_dtype.kind, b_dtype.kind
    numeric = ("i", "u", "f", "b")
    if ak in numeric and bk in numeric:
        return True
    return ak == "O" and bk == "O"


def _as_bool(series: pd.Series) -> pd.Series:
    """Resolve any NA to False and pin non-nullable np.bool_."""
    return series.fillna(False).astype(bool)


def _positional(a: pd.Series, b: pd.Series):
    """Positional (not label) alignment -- seam columns share a frame; stay consistent."""
    return a.reset_index(drop=True), b.reset_index(drop=True)


def ids_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise id equality, dtype-safe, POSITIONAL. NA never equals anything -> False.
    Returns a non-nullable np.bool_ Series."""
    a, b = pd.Series(a), pd.Series(b)
    pa, pb = _positional(a, b)
    if _directly_comparable(a.dtype, b.dtype):
        eq = (pa == pb) & pa.notna() & pb.notna()
        return _as_bool(eq)
    ca, cb = canonical_id_series(pa), canonical_id_series(pb)
    eq = (ca == cb) & ca.notna() & cb.notna()
    return _as_bool(eq)


def ids_differ(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise id inequality requiring BOTH present (opponent mask), POSITIONAL.
    A NA on either side (e.g. an unmatched how='left' row) is NOT "differ".
    Returns a non-nullable np.bool_ Series."""
    a, b = pd.Series(a), pd.Series(b)
    pa, pb = _positional(a, b)
    if _directly_comparable(a.dtype, b.dtype):
        differ = pa.notna() & pb.notna() & (pa != pb)
        return _as_bool(differ)
    ca, cb = canonical_id_series(pa), canonical_id_series(pb)
    differ = ca.notna() & cb.notna() & (ca != cb)
    return _as_bool(differ)


def ids_match(series, scalar) -> pd.Series:
    """Vectorized ``series == scalar``, dtype-safe. ``series`` may be a pd.Series or
    np.ndarray. Returns a non-nullable np.bool_ Series."""
    s = pd.Series(series)
    key = canonical_id(scalar)
    if key is pd.NA:
        return pd.Series(np.zeros(len(s), dtype=bool), index=s.index)
    # fast path: directly comparable to a 1-element scalar series (incl. object x object)
    scal_s = pd.Series([scalar])
    if _directly_comparable(s.dtype, scal_s.dtype):
        return _as_bool(s == scalar)
    return _as_bool(canonical_id_series(s) == key)


def same_id(a, b) -> bool:
    """Scalar-vs-scalar id equality (groupby-loop comparisons). False if either is NA."""
    ca, cb = _canonical(a), _canonical(b)
    # isinstance narrows to str (NA -> not str -> False); also satisfies the type checker.
    if not isinstance(ca, str) or not isinstance(cb, str):
        return False
    return ca == cb


def align_join_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list):
    """Canonicalize id-valued join keys on both sides to a common dtype BEFORE merge.

    ``keys`` entries are either a ``str`` (same column name on both sides) or a
    ``(left_key, right_key)`` tuple for differently-named keys (e.g. the
    ``frame_id_int`` (left) vs ``frame_id`` (right) merge in ``_kernels.py``, which
    ``pd.merge(left_on=..., right_on=...)`` allows but a same-name aligner could not
    bridge without pair support).

    Per key: if both sides are merge-compatible (both numeric, OR both object -- pandas
    merges those fine) -> no-op (fast path, zero cost); else (numeric vs object) coerce
    both to canonical string. Returns (left, right) copies ready for the merge. Prevents
    the ValueError pandas raises on a numeric-vs-object merge key (ADR-019)."""
    left, right = left.copy(), right.copy()
    for k in keys:
        lk, rk = (k, k) if isinstance(k, str) else (k[0], k[1])
        if lk not in left.columns or rk not in right.columns:
            continue
        if _merge_compatible(left[lk].dtype, right[rk].dtype):
            continue
        left[lk] = canonical_id_series(left[lk])
        right[rk] = canonical_id_series(right[rk])
    return left, right
