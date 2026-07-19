"""Dtype-safe id identity, repo-wide (ADR-019).

Defines "id identity" once: a single canonicalization truth (scalar + vectorized),
comparison helpers, and pre-merge join-key alignment. ADR-019 requires **every** id
comparison in the codebase to route through this module -- `spadl/`, `vaep/`, `atomic/`,
`causal/`, `tracking/` and `gkdv/` all consume it -- which is why it is a public module
rather than a private tracking submodule (promoted in 4.53.0; it previously lived at
`silly_kicks/tracking/_id_compat.py`).

Cost model: comparison/align helpers keep a same-dtype fast path, so a matched NUMERIC
pipeline pays nothing beyond a dtype check. Object-vs-object is NOT free -- it costs a
content probe (see `_raw_comparable`), because "two object id columns are both genuine
strings" turned out to be false in a shipped seam. The probe is cheap relative to the
comparison it guards (`_all_genuine_strings` records the measurement) and correctness wins.

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
    """Scalar entry point -- delegates to the single ``_canonical`` truth.

    Examples
    --------
    Every integral spelling of the same id collapses to one key, which is what makes a
    cross-dtype comparison answerable at all:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import canonical_id
    >>> canonical_id(366), canonical_id(366.0), canonical_id("366")
    ('366', '366', '366')

    Missing ids canonicalize to ``pd.NA``, which never equals anything -- including another
    ``pd.NA`` (see :func:`ids_equal`). Genuine non-numeric ids pass through untouched:

    >>> canonical_id(None) is pd.NA, canonical_id(float("nan")) is pd.NA
    (True, True)
    >>> canonical_id("GK-7")
    'GK-7'
    """
    return _canonical(x)


#: int64 bounds as EXACTLY-representable float64 powers of two. Deliberately not
#: ``float(np.iinfo("int64").max)``: that rounds UP to 2**63, so a ``<=`` bound against it would
#: wave through 2**63 itself -- one past the largest int64 -- and re-open the cast error.
_INT64_FLOOR = -(2.0**63)  # == int64 min exactly, so the bound is inclusive
_INT64_CEIL = 2.0**63  # one PAST int64 max, so the bound is exclusive


def canonical_id_series(s: pd.Series) -> pd.Series:
    """Vectorized canonicalization, NaN-safe, object dtype. Matches ``canonical_id``
    element-wise. Avoids naive ``.astype(str)`` (which yields '366.0', not '366').

    Examples
    --------
    The float column an outer merge leaves behind canonicalizes to the SAME keys as the
    int column it came from, and nulls survive as ``pd.NA`` rather than the string
    ``'nan'``:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import canonical_id_series
    >>> canonical_id_series(pd.Series([366.0, None, 512.0])).tolist()
    ['366', <NA>, '512']

    This is exactly what the naive stringify gets wrong -- ``'366.0'`` never matches the
    ``'366'`` its int-dtyped counterpart produces:

    >>> pd.Series([366.0, 512.0]).astype(str).tolist()
    ['366.0', '512.0']

    An OBJECT column is not assumed to hold strings: ``infer_ball_carrier`` emits boxed
    floats, so mixed content is routed element-wise rather than stringified wholesale.

    >>> canonical_id_series(pd.Series([2.0, "3"], dtype=object)).tolist()
    ['2', '3']
    """
    s = pd.Series(s)
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    notna = s.notna()
    if not notna.any():
        return out
    vals = s[notna]
    kind = vals.dtype.kind
    if kind == "f":
        integral = vals == vals.round()
        # `integral` is True for +/-inf and for any large-magnitude float, but `.astype("Int64")`
        # can represent NEITHER: infinities raise OverflowError and out-of-range values raise
        # "cannot safely cast non-equivalent float64 to int64". Python ints are arbitrary-precision,
        # so the scalar `_canonical` renders both -- which is the parity this function promises.
        # Same shape as the object branch below: keep the vectorized cast for the values it can
        # hold, and route only the rest through the single `_canonical` truth.
        castable = integral & np.isfinite(vals) & (vals >= _INT64_FLOOR) & (vals < _INT64_CEIL)
        overflow = integral & ~castable
        out.loc[vals.index[castable]] = vals[castable].astype("Int64").astype("string").astype("object")
        out.loc[vals.index[overflow]] = vals[overflow].map(_canonical)
        out.loc[vals.index[~integral]] = vals[~integral].astype("string").astype("object")
    elif kind in ("i", "u") or str(vals.dtype) in ("Int64", "Int32", "Int16", "Int8"):
        out.loc[vals.index] = vals.astype("Int64").astype("string").astype("object")
    else:
        # object dtype. The COMMON case is a homogeneous genuine-string id column
        # (sportec/kloppy), where a VECTORIZED stringify matches ``_canonical`` exactly and
        # costs nothing -- keep that path (A1: no Python loop; pinned by the structural perf
        # guard).
        #
        # But object columns are NOT always strings. ``infer_ball_carrier`` emits
        # ``ball_carrier_team_id`` as an object column holding FLOATS, and a bare stringify
        # renders 2.0 as "2.0" while ``_canonical`` renders it "2" -- so a consumer joining
        # carrier ids to frame ids through ``ids_equal`` got a silent all-row miss. An earlier
        # revision of this comment asserted that case "does not occur for real id columns";
        # that was wrong, and it was wrong in a shipped public seam.
        #
        # ``infer_dtype`` is a cheap C-level probe. All-strings keeps the fast path; anything
        # else (boxed numerics, or a genuinely mixed column) routes element-wise through the
        # single ``_canonical`` truth, so this function delivers the element-wise match its
        # docstring promises. The Python loop is confined to the pathological columns.
        if pd.api.types.infer_dtype(vals, skipna=True) == "string":
            out.loc[vals.index] = vals.astype("string").astype("object")
        else:
            out.loc[vals.index] = vals.map(_canonical)
    return out


def _directly_comparable(a_dtype, b_dtype) -> bool:
    """DTYPE-ONLY comparability: same numpy kind (Int64 nullable and int64 both report
    'i'), OR both object.

    The both-object arm rests on a same-provider invariant ("two object id columns are
    both genuine strings") that is NOT universally true, so NOTHING calls this to decide a
    comparison: every caller goes through ``_raw_comparable``, which probes content before
    trusting the both-object arm. Kept as the dtype-level building block ``_raw_comparable``
    delegates its numeric case to -- it cannot see a boxed-numeric object column, and a
    caller that needs a comparison decision must not use it directly."""
    ak, bk = a_dtype.kind, b_dtype.kind
    if ak == "O" and bk == "O":
        return True
    return ak == bk and ak in ("i", "u", "f", "b")


def _all_genuine_strings(s: pd.Series) -> bool:
    """True when an object column holds only genuine strings (the common id case).

    ``infer_dtype`` is a C-level probe. Measured on a 500k-row object id column (pandas
    2.3.3): ~2.1 ms per probe against ~14 ms for the raw ``==`` it guards, i.e. **~15% per
    probe**. ``_raw_comparable`` probes BOTH sides, so the guard as actually paid costs
    **~30% of the comparison** -- about a third, not the ~25% an earlier revision claimed for
    the whole guard (that figure was roughly the ONE-probe cost, quoted as though it covered
    both). Probe cost is flat in string length, cardinality and match rate; only the ``==``
    side moves. Cheap enough that correctness wins, but not free -- which is why the module
    docstring no longer advertises object-vs-object as a zero-cost fast path.
    """
    return pd.api.types.infer_dtype(s, skipna=True) == "string"


def _raw_comparable(a: pd.Series, b: pd.Series) -> bool:
    """Whether ``a == b`` is trustworthy WITHOUT canonicalization.

    Numeric kinds are decided by dtype alone. Object-vs-object needs a CONTENT probe:
    ``_directly_comparable`` assumed "two object id columns are both genuine strings", but
    ``infer_ball_carrier`` emits an object column of FLOATS, so a boxed 2.0 raw-compared
    against the string "2" is False even though both canonicalize to "2" -- i.e. this module
    contradicted its own canonicalization. Probe, and fall through to the canonical path
    whenever either side is not genuinely string-typed.
    """
    if a.dtype.kind == "O" and b.dtype.kind == "O":
        return _all_genuine_strings(a) and _all_genuine_strings(b)
    return _directly_comparable(a.dtype, b.dtype)


_NULLABLE_INT_NAMES = ("Int64", "Int32", "Int16", "Int8", "UInt64", "UInt32", "UInt16", "UInt8")


def restore_id_dtype(values, source_dtype) -> pd.Series:
    """Put a kernel-built id column back on its SOURCE dtype, where that dtype can hold it.

    Kernels assemble id results into ``np.empty(n, dtype=object)`` because a slot may be
    missing. Handing that object column straight back leaks a *boxed-numeric* id: a
    ``float64`` team id round-trips as an object column of ``2.0``, which raw-compares
    False against the string ``"2"`` and writes a useless object column to parquet. This
    restores the source representation instead of special-casing one dtype.

    Restorability, not blanket casting -- a **numpy** integer dtype cannot hold NA, so a
    result with unmatched rows stays float (the long-standing deliberate behaviour at the
    two ``_gk_resolve``/``features`` sites); a **nullable** integer dtype can, so it is
    always restored. Object sources are left untouched: object IS their source dtype.

    Examples
    --------
    A fully-matched kernel result goes back onto its source numpy dtype:

    >>> import numpy as np
    >>> from silly_kicks.id_compat import restore_id_dtype
    >>> restore_id_dtype(np.array(["7", "9"], dtype=object), np.dtype("int64")).tolist()
    [7, 9]

    Add one unmatched slot and the SAME source dtype deliberately yields float, because a
    numpy integer cannot hold NA. This is the asymmetry to remember: ``restore_id_dtype``
    promises restorability, not a blanket cast -- it will not invent a sentinel id to
    honour the requested dtype.

    >>> restore_id_dtype(np.array(["7", None], dtype=object), np.dtype("int64")).dtype
    dtype('float64')

    A NULLABLE integer source can hold the gap, so it is restored exactly -- which is why
    an id column declared ``Int64`` upstream survives a kernel round trip and a plain
    ``int64`` one does not:

    >>> restore_id_dtype(np.array(["7", None], dtype=object), "Int64").tolist()
    [7, <NA>]
    """
    s = pd.Series(values)
    name = str(source_dtype)
    if name in _NULLABLE_INT_NAMES:
        return pd.to_numeric(s, errors="coerce").astype(name)
    if name == "string":
        return s.astype("string")
    kind = getattr(source_dtype, "kind", None)
    if kind in ("i", "u"):
        num = pd.to_numeric(s, errors="coerce")
        # numpy ints cannot represent NA -- only narrow back when nothing is missing.
        return num.astype(source_dtype) if not num.isna().any() else num
    if kind == "f":
        return pd.to_numeric(s, errors="coerce")
    return s


def _merge_compatible(a_dtype, b_dtype) -> bool:
    """True when two columns can be used as a pd.merge key WITHOUT canonicalization.
    pandas raises only on numeric-vs-object keys; numeric-vs-numeric (int64/Int64/float64,
    e.g. an Int64 linker frame_id vs a float64 frames frame_id) merges fine on value, and
    object-vs-object merges fine. Broader than the element-wise comparison rule
    (`_raw_comparable`): stringifying a mergeable numeric key would needlessly change the
    merge representation (and, for non-integer float artifacts, its cardinality).

    Note this arm is about MERGEABILITY, not equality semantics: pandas merges two object
    key columns without raising, which is all this answers. It deliberately does NOT carry
    the retired "both object implies both genuine strings" assumption -- a boxed-numeric
    object key merges fine and simply matches nothing, whereas the same assumption in the
    comparison path produced a silent wrong answer (hence the `_raw_comparable` probe)."""
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
    Returns a non-nullable np.bool_ Series.

    Examples
    --------
    POSITIONAL, not label-aligned. Two id columns sliced off different frames rarely share
    an index, and pandas' own ``==`` refuses that outright rather than comparing row 0 to
    row 0 -- so the alignment rule is a contract, not an implementation detail:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import ids_equal
    >>> a = pd.Series([7, 9], index=[10, 11])
    >>> b = pd.Series(["7", "9"], index=[20, 21])
    >>> try:
    ...     a == b
    ... except ValueError as exc:
    ...     print(type(exc).__name__)
    ValueError
    >>> ids_equal(a, b).tolist()
    [True, True]

    NA never equals anything, so an unmatched ``how="left"`` row is False rather than
    nullable -- the result is plain ``np.bool_`` and is always safe to use as a mask:

    >>> ids_equal(pd.Series([1, None], dtype="Int64"), pd.Series([1, None], dtype="Int64")).tolist()
    [True, False]
    """
    a, b = pd.Series(a), pd.Series(b)
    pa, pb = _positional(a, b)
    if _raw_comparable(a, b):
        eq = (pa == pb) & pa.notna() & pb.notna()
        return _as_bool(eq)
    ca, cb = canonical_id_series(pa), canonical_id_series(pb)
    eq = (ca == cb) & ca.notna() & cb.notna()
    return _as_bool(eq)


def ids_differ(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise id inequality requiring BOTH present (opponent mask), POSITIONAL.
    A NA on either side (e.g. an unmatched how='left' row) is NOT "differ".
    Returns a non-nullable np.bool_ Series.

    Examples
    --------
    This is NOT ``~ids_equal(...)``. A row where either id is missing is neither "same"
    nor "different" -- both helpers return False for it, so the two masks do not partition
    the frame:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import ids_differ, ids_equal
    >>> actor = pd.Series([1, 2, None], dtype="Int64")
    >>> other = pd.Series([2, 2, 2], dtype="Int64")
    >>> ids_equal(actor, other).tolist()
    [False, True, False]
    >>> ids_differ(actor, other).tolist()
    [True, False, False]

    That gap is the point: ``ids_differ`` builds OPPONENT masks, and a NaN-actor row
    (ADR-027 -- Gradient Sports emits genuinely team-less duel/foul events) must not be
    counted as an opponent just because its id fails to equal ours.
    """
    a, b = pd.Series(a), pd.Series(b)
    pa, pb = _positional(a, b)
    if _raw_comparable(a, b):
        differ = pa.notna() & pb.notna() & (pa != pb)
        return _as_bool(differ)
    ca, cb = canonical_id_series(pa), canonical_id_series(pb)
    differ = ca.notna() & cb.notna() & (ca != cb)
    return _as_bool(differ)


def ids_match(series, scalar) -> pd.Series:
    """Vectorized ``series == scalar``, dtype-safe. ``series`` may be a pd.Series or
    np.ndarray. Returns a non-nullable np.bool_ Series.

    Examples
    --------
    THE motivating failure. A tracking frame carries ``team_id`` as ``Int64`` while the
    caller holds ``home_team_id`` as a string; the raw comparison does not raise, it
    quietly matches nothing -- so an orientation seam labelled ZERO players and the frames
    silently came out mis-oriented (ADR-019):

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import ids_match
    >>> team_id = pd.Series([366, 366, 512], dtype="Int64")
    >>> (team_id == "366").tolist()
    [False, False, False]
    >>> ids_match(team_id, "366").tolist()
    [True, True, False]

    A missing scalar matches nothing rather than raising, so a caller holding an
    unresolved id gets an empty mask instead of a crash:

    >>> ids_match(team_id, None).tolist()
    [False, False, False]
    """
    s = pd.Series(series)
    key = canonical_id(scalar)
    if key is pd.NA:
        return pd.Series(np.zeros(len(s), dtype=bool), index=s.index)
    # fast path: raw-comparable against a 1-element scalar series. Object x object is NOT
    # waved through on dtype alone -- `_raw_comparable` content-probes both sides, so a boxed
    # numeric on either side falls through to the canonical path below.
    scal_s = pd.Series([scalar])
    if _raw_comparable(s, scal_s):
        return _as_bool(s == scalar)
    return _as_bool(canonical_id_series(s) == key)


def ids_isin(series, scalars) -> pd.Series:
    """Vectorized ``series.isin(scalars)``, dtype-safe. The COLLECTION sibling of
    :func:`ids_match`. ``series`` may be a pd.Series or np.ndarray; ``scalars`` is any
    iterable of ids (``None`` / empty -> an all-False mask). Returns a non-nullable
    np.bool_ Series.

    A caller-supplied id SET resolved against an id column carries the identical hazard as a
    scalar -- ``.isin`` compares by value, so a ``{"999"}`` set against an ``int64``
    ``player_id`` column matches nothing and the caller's declaration is silently discarded:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import ids_isin
    >>> player_id = pd.Series([999, 999, 12], dtype="int64")
    >>> player_id.isin({"999"}).tolist()
    [False, False, False]
    >>> ids_isin(player_id, {"999"}).tolist()
    [True, True, False]

    Not a ``.astype(str).isin(...)``: that renders a float-backed id column as ``'999.0'``
    and matches nothing either, which is the same silent failure wearing a fix's clothes.
    ``canonical_id`` collapses every integral spelling instead:

    >>> ids_isin(pd.Series([999.0, 12.0]), {999}).tolist()
    [True, False]

    Missing ids never match -- neither a NA in the column nor a NA offered in the set, so a
    caller holding an unresolved keeper id cannot accidentally claim every unattributed row:

    >>> ids_isin(pd.Series([999, None]), {None}).tolist()
    [False, False]
    """
    s = pd.Series(series)
    if scalars is None:
        return pd.Series(np.zeros(len(s), dtype=bool), index=s.index)
    keys = {k for k in (canonical_id(v) for v in scalars) if isinstance(k, str)}
    if not keys:
        return pd.Series(np.zeros(len(s), dtype=bool), index=s.index)
    return _as_bool(canonical_id_series(s).isin(keys))


def same_id(a, b) -> bool:
    """Scalar-vs-scalar id equality (groupby-loop comparisons). False if either is NA.

    Examples
    --------
    The scalar sibling of :func:`ids_match`, for the ``for team, group in
    frames.groupby(...)`` loops where the key arrives already unboxed:

    >>> from silly_kicks.id_compat import same_id
    >>> same_id(366, "366"), same_id(366, 512)
    (True, False)

    Two missing ids are NOT the same id -- an unresolved keeper must never be paired with
    another unresolved keeper just because both are absent:

    >>> same_id(None, None)
    False
    """
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
    the ValueError pandas raises on a numeric-vs-object merge key (ADR-019).

    Examples
    --------
    Unlike the comparison helpers, the numeric-vs-object case here FAILS LOUD -- pandas
    refuses the merge outright. That makes this the one seam whose absence is noisy rather
    than silent, and the fix is a pre-pass, not a cast at the call site:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import align_join_keys
    >>> left = pd.DataFrame({"team_id": [366, 512], "v": [1, 2]})
    >>> right = pd.DataFrame({"team_id": ["366", "512"], "w": [3, 4]})
    >>> try:
    ...     left.merge(right, on="team_id")
    ... except ValueError as exc:
    ...     print(type(exc).__name__)
    ValueError
    >>> lt, rt = align_join_keys(left, right, ["team_id"])
    >>> lt.merge(rt, on="team_id")["w"].tolist()
    [3, 4]

    Differently-named keys pair up as a tuple, and an already-mergeable pair is left
    ALONE rather than stringified -- coercing a numeric key that merges fine would change
    the merge representation for no gain:

    >>> lt, rt = align_join_keys(left, right, [("team_id", "team_id")])
    >>> lt["team_id"].tolist()
    ['366', '512']
    >>> same = align_join_keys(left, left, ["team_id"])[0]
    >>> same["team_id"].dtype
    dtype('int64')
    """
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
