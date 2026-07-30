"""Unit tests for the tracking id-dtype safety primitive (ADR-019)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks import id_compat as idc

# (input, expected canonical) — the single truth, asserted at BOTH entry points
CANON_CASES = [
    (366, "366"),
    (366.0, "366"),
    (np.int64(366), "366"),
    (pd.array([366], dtype="Int64")[0], "366"),
    ("366", "366"),
    ("DFL-CLU-A", "DFL-CLU-A"),
    (366.5, "366.5"),
]


@pytest.mark.parametrize("raw,expected", CANON_CASES)
def test_canonical_id_scalar(raw, expected):
    assert idc.canonical_id(raw) == expected


def test_canonical_id_scalar_na():
    assert idc.canonical_id(None) is pd.NA
    assert idc.canonical_id(np.nan) is pd.NA
    assert idc.canonical_id(pd.NA) is pd.NA


def test_canonical_id_series_matches_scalar():
    # single-truth: vectorized output == scalar `_canonical`, fed per NATURAL dtype.
    #
    # The last row used to be excluded as "the one documented pathological divergence": an
    # integral float boxed in an object column canonicalized to "366.0" vectorized vs "366"
    # scalar. That divergence is GONE -- the object branch content-probes and falls back to
    # `_canonical` element-wise -- so it is included here as coverage rather than described
    # as an exception. There is now no dtype for which the two entry points disagree.
    for dtype, vals, exp in [
        ("int64", [366, 7], ["366", "7"]),
        ("Int64", [366, 7], ["366", "7"]),
        ("float64", [366.0, 366.5], ["366", "366.5"]),
        ("object", ["366", "DFL-CLU-A"], ["366", "DFL-CLU-A"]),
        ("object", [366.0, 7], ["366", "7"]),  # boxed integral float, formerly divergent
    ]:
        out = idc.canonical_id_series(pd.Series(vals, dtype=dtype))
        assert out.tolist() == exp
        assert [idc.canonical_id(v) for v in vals] == exp  # scalar entry agrees


def test_canonical_id_series_integral_float_collapse():
    # the str(366.0)->"366.0" trap: must collapse
    out = idc.canonical_id_series(pd.Series([366.0, 366.5, np.nan], dtype="float64"))
    assert out.tolist()[:2] == ["366", "366.5"]
    assert out.tolist()[2] is pd.NA


def test_canonical_id_series_int64_and_Int64():
    assert idc.canonical_id_series(pd.Series([366], dtype="int64")).tolist() == ["366"]
    assert idc.canonical_id_series(pd.Series([366], dtype="Int64")).tolist() == ["366"]


# Floats the int64 cast provably cannot represent, with the scalar truth they must match.
# `inf`/`-inf` are integral by `x == round(x)`; 1e20 and 2**63 exceed int64's range while
# Python ints are arbitrary-precision, so the SCALAR path renders them exactly.
UNREPRESENTABLE_FLOATS = [
    (float("inf"), "inf"),
    (float("-inf"), "-inf"),
    (1e20, "100000000000000000000"),
    (float(2**63), "9223372036854775808"),
    (-1e25, "-10000000000000000905969664"),
]


@pytest.mark.parametrize("raw,expected", UNREPRESENTABLE_FLOATS)
def test_canonical_id_series_matches_scalar_on_int64_unrepresentable_floats(raw, expected):
    """The element-wise-parity claim, on the floats that break the vectorized ``Int64`` cast.

    ``integral = vals == vals.round()`` is True for ``inf`` and for any large-magnitude float,
    which routed them into ``.astype("Int64")`` -- raising ``OverflowError`` (infinities) or
    ``TypeError: cannot safely cast non-equivalent float64 to int64`` (out-of-range). The scalar
    ``canonical_id`` handled all of them, so the module contradicted its own docstring: an id
    column carrying one sentinel infinity crashed a seam every consumer is required to route
    through, rather than yielding a value that simply never matches.
    """
    assert idc.canonical_id(raw) == expected  # scalar truth, unchanged
    assert idc.canonical_id_series(pd.Series([raw], dtype="float64")).tolist() == [expected]


def test_canonical_id_series_mixes_representable_and_unrepresentable_floats():
    """The fast path must survive CONTACT with a bad value, not just isolation.

    A per-column all-or-nothing fallback would pass the parametrized test above (every input
    there is a pure-bad column) while still mangling the ordinary ids that share a real column
    with one sentinel. Ordinary values keep the vectorized cast; only the unrepresentable ones
    fall back element-wise.
    """
    s = pd.Series([366.0, float("inf"), 366.5, np.nan, 1e20, 7.0], dtype="float64")
    out = idc.canonical_id_series(s)
    assert out.tolist()[:3] == ["366", "inf", "366.5"]
    assert out.tolist()[3] is pd.NA
    assert out.tolist()[4:] == ["100000000000000000000", "7"]
    # ...and every element still agrees with the scalar entry point.
    assert [x for x in out.tolist() if x is not pd.NA] == [idc.canonical_id(v) for v in s.tolist() if not np.isnan(v)]


def test_ids_equal_survives_an_infinity_sentinel():
    """Behavioural consequence: the crash reached the public comparison helpers.

    ``ids_equal`` on a float id column against string ids takes the canonical path, so a single
    sentinel infinity took down the comparison rather than simply not matching.
    """
    a = pd.Series([366.0, float("inf")], dtype="float64")
    b = pd.Series(["366", "366"], dtype="object")
    assert idc.ids_equal(a, b).tolist() == [True, False]


# --- Task 2: comparison helpers ---


def test_ids_equal_cross_dtype_and_na():
    a = pd.Series([366, 7], dtype="int64")
    b = pd.Series(["366", "9"], dtype="object")  # asymmetric
    out = idc.ids_equal(a, b)
    assert out.tolist() == [True, False]
    assert out.dtype == np.bool_  # C1: non-nullable
    # NA never equals anything
    an = pd.Series([366, pd.NA], dtype="Int64")
    bn = pd.Series(["366", "366"], dtype="object")
    assert idc.ids_equal(an, bn).tolist() == [True, False]


def test_ids_differ_both_present_and_left_join_miss():
    # opponent mask: object NaN (unmatched left-join) must NOT count as "differ"
    a = pd.Series(["5", pd.NA, "9"], dtype="object")  # team_id_dl (NaN = unmatched)
    b = pd.Series(["5", "5", "5"], dtype="object")  # team_id_action
    out = idc.ids_differ(a, b)
    assert out.tolist() == [False, False, True]  # row1 NA -> excluded (N1)
    assert out.dtype == np.bool_


def test_ids_match_series_and_array():
    s = pd.Series([366, 7], dtype="Int64")
    assert idc.ids_match(s, "366").tolist() == [True, False]  # string scalar
    arr = np.array(["366", "7"], dtype=object)
    assert idc.ids_match(arr, 366).tolist() == [True, False]  # numpy array side


def test_ids_match_same_kind_fast_path():
    s = pd.Series([366, 7], dtype="int64")
    assert idc.ids_match(s, 366).tolist() == [True, False]


def test_ids_isin_resolves_a_caller_set_across_dtypes():
    """The wyscout/add_gk_role shape: a caller-supplied id SET against an id column.

    Both directions, because callers hold roster ids either way -- a wyscout caller with
    string ids against an int64 `player_id` column, and a `goalkeeper_ids={1}` against an
    object-string column.
    """
    assert idc.ids_isin(pd.Series([999, 999, 12], dtype="int64"), {"999"}).tolist() == [True, True, False]
    assert idc.ids_isin(pd.Series(["1", "77"], dtype=object), {1}).tolist() == [True, False]
    assert idc.ids_isin(pd.Series([999, 12], dtype="Int64"), {999}).tolist() == [True, False]
    # np.ndarray side, like ids_match
    assert idc.ids_isin(np.array(["999", "12"], dtype=object), {999}).tolist() == [True, False]
    assert idc.ids_isin(pd.Series([1, 2]), {1}).dtype == np.bool_


def test_ids_isin_is_not_a_naive_stringify():
    """THE discriminating case -- and the one the id-scalar registry gate CANNOT see.

    That gate's third axis (a float-valued scalar) fires only on a numeric `matched`, and an
    entity-id COLLECTION entry declares a set, so a `ids_isin` degraded to
    `s.astype(str).isin(keys)` passes the whole registry green. Measured: mutating the helper
    that way left all 103 registry tests passing.

    A float-backed id column is the case that separates them -- `.astype(str)` renders 999.0
    as "999.0", which matches no canonicalized key, so the naive form silently resolves
    NOTHING. Float-backed id columns are not exotic: they are what an outer merge leaves
    behind wherever a row failed to join.
    """
    float_col = pd.Series([999.0, 12.0], dtype="float64")
    assert float_col.astype(str).isin({"999"}).tolist() == [False, False]  # the naive form
    assert idc.ids_isin(float_col, {999}).tolist() == [True, False]
    assert idc.ids_isin(float_col, {"999"}).tolist() == [True, False]
    # object columns holding BOXED floats (what infer_ball_carrier emits) too
    assert idc.ids_isin(pd.Series([999.0, "12"], dtype=object), {999, 12}).tolist() == [True, True]


def test_ids_isin_missing_ids_never_match():
    """NA on either side is "unresolved", never a wildcard.

    A caller whose roster lookup returned a null must not thereby claim every unattributed
    row. The raw `.isin` does exactly that -- but only on SOME dtypes, which is the sharper
    problem: the null-matching behaviour is an accident of the column's storage, so the same
    caller code silently changes meaning when a provider's id column changes dtype.
    MEASURED, per dtype:

        object  + {None}   -> the null row MATCHES
        float64 + {nan}    -> the null row MATCHES
        Int64   + {None}   -> it does not

    `ids_isin` gives all three the same answer: unresolved never matches.
    """
    obj_col = pd.Series([999, None], dtype=object)
    float_col = pd.Series([999.0, np.nan], dtype="float64")
    nullable_col = pd.Series([999, None], dtype="Int64")

    assert obj_col.isin({None}).tolist() == [False, True]  # raw: the null row is claimed
    assert float_col.isin({np.nan}).tolist() == [False, True]  # raw: likewise
    assert nullable_col.isin({None}).tolist() == [False, False]  # raw: but NOT here

    for col in (obj_col, float_col, nullable_col):
        assert idc.ids_isin(col, {None}).tolist() == [False, False]
        assert idc.ids_isin(col, {np.nan}).tolist() == [False, False]
        assert idc.ids_isin(col, {np.nan, 999}).tolist() == [True, False]
        # Empty / absent collections resolve nothing, and never raise.
        assert idc.ids_isin(col, None).tolist() == [False, False]
        assert idc.ids_isin(col, set()).tolist() == [False, False]


def test_ids_isin_passes_genuine_string_ids_through():
    """Non-numeric provider ids (DFL-CLU-A, GK-7) are opaque tokens, not numbers."""
    col = pd.Series(["DFL-OBJ-1", "DFL-OBJ-2"], dtype=object)
    assert idc.ids_isin(col, {"DFL-OBJ-2"}).tolist() == [False, True]
    assert idc.ids_isin(col, {"DFL-OBJ-9"}).tolist() == [False, False]


def test_ids_isin_preserves_the_callers_index():
    """The mask is consumed positionally next to other masks on the same frame, so a
    re-indexed input must come back aligned, not silently re-based to 0..n."""
    col = pd.Series([999, 12], index=[7, 9], dtype="int64")
    assert idc.ids_isin(col, {999}).index.tolist() == [7, 9]
    assert idc.ids_isin(col, set()).index.tolist() == [7, 9]


def test_same_id_scalar():
    assert idc.same_id(366, "366") is True
    assert idc.same_id("366", 366) is True
    assert idc.same_id(366, 7) is False
    assert idc.same_id(pd.NA, 366) is False
    assert idc.same_id(366, np.nan) is False


def test_object_object_takes_raw_fast_path_no_canonicalize(monkeypatch):
    # A1: genuine-string providers (sportec/kloppy) must NOT Python-loop through
    # canonical_id_series -- object x object is directly comparable.
    calls = {"n": 0}
    real = idc.canonical_id_series
    monkeypatch.setattr(
        idc,
        "canonical_id_series",
        lambda s: (calls.__setitem__("n", calls["n"] + 1), real(s))[1],
    )
    a = pd.Series(["DFL-CLU-A", "DFL-CLU-B"], dtype="object")
    b = pd.Series(["DFL-CLU-A", "DFL-CLU-Z"], dtype="object")
    assert idc.ids_equal(a, b).tolist() == [True, False]
    assert idc.ids_differ(a, b).tolist() == [False, True]
    assert calls["n"] == 0  # zero canonicalization on the object x object path


# --- Task 3: align_join_keys ---


def test_align_join_keys_prevents_merge_raise():
    left = pd.DataFrame({"game_id": pd.Series([1, 2], dtype="int64"), "v": [10, 20]})
    right = pd.DataFrame({"game_id": pd.Series(["1", "2"], dtype="object"), "w": [100, 200]})
    # raw merge would raise ValueError on int64 x object key
    l2, r2 = idc.align_join_keys(left, right, ["game_id"])
    out = l2.merge(r2, on="game_id", how="left")
    assert out["w"].tolist() == [100, 200]


def test_align_join_keys_same_kind_noop():
    left = pd.DataFrame({"game_id": pd.Series([1, 2], dtype="int64")})
    right = pd.DataFrame({"game_id": pd.Series([1, 2], dtype="int64")})
    l2, r2 = idc.align_join_keys(left, right, ["game_id"])
    # fast path: dtype unchanged (no needless stringify)
    assert l2["game_id"].dtype == np.dtype("int64")
    assert r2["game_id"].dtype == np.dtype("int64")


def test_align_join_keys_object_object_noop():  # N-c: object x object is mergeable as-is
    left = pd.DataFrame({"game_id": pd.Series(["1", "2"], dtype="object")})
    right = pd.DataFrame({"game_id": pd.Series(["1", "2"], dtype="object")})
    l2, r2 = idc.align_join_keys(left, right, ["game_id"])
    assert l2["game_id"].dtype == object and r2["game_id"].dtype == object


def test_align_join_keys_boxed_object_vs_string_object_merges():
    """BOTH-object is not automatically merge-safe: one side may be BOXED NUMERICS.

    ``align_join_keys`` decided on DTYPE alone (``_merge_compatible``) and treated any
    object-vs-object pair as a no-op fast path -- "pandas merges those fine". Pandas does merge
    them; it just MATCHES NOTHING, because a boxed ``10.0`` is not equal to the string ``"10"``.

    That is the exact hazard ``_raw_comparable`` already content-probes for on the COMPARISON
    side (ADR-043): ``infer_ball_carrier`` emits object columns of floats. The module therefore
    contradicted itself -- ``ids_equal`` canonicalized this pair while ``align_join_keys`` did not.

    Found via the ADR-028 fail-loud seam: ``acting_team_attacks_rtl`` merges actions to frames on
    ``team_id``, and on this dtype pair the merge silently matched zero rows, yielding an
    all-False re-projection flip and away-team geometry in the wrong convention.
    """
    left = pd.DataFrame({"team_id": pd.Series([10.0, 11.0], dtype="float64").astype(object), "v": [1, 2]})
    right = pd.DataFrame({"team_id": pd.Series(["10", "11"], dtype="object"), "w": [3, 4]})

    # Non-vacuity: the RAW merge must genuinely lose the rows, or this proves nothing.
    raw = left.merge(right, on="team_id", how="left")
    assert raw["w"].isna().all(), "raw merge already matched -- fixture does not exercise the hazard"

    l2, r2 = idc.align_join_keys(left, right, ["team_id"])
    out = l2.merge(r2, on="team_id", how="left")
    assert out["w"].tolist() == [3, 4]


def test_align_join_keys_genuine_string_pair_stays_a_noop():
    """The fast path survives: two genuine-string object columns are still left ALONE.

    Non-vacuity partner for the test above -- without it, the fix could be "canonicalize every
    object pair", which would pay the probe AND the coercion on the common case for no gain.
    """
    left = pd.DataFrame({"team_id": pd.Series(["a", "b"], dtype="object")})
    right = pd.DataFrame({"team_id": pd.Series(["a", "b"], dtype="object")})
    l2, r2 = idc.align_join_keys(left, right, ["team_id"])
    assert l2["team_id"].tolist() == ["a", "b"] and r2["team_id"].tolist() == ["a", "b"]


def test_align_join_keys_pair_names():  # A3: differently-named keys (frame_id_int vs frame_id)
    left = pd.DataFrame({"frame_id_int": pd.Series([1, 2], dtype="int64"), "v": [10, 20]})
    right = pd.DataFrame({"frame_id": pd.Series(["1", "2"], dtype="object"), "w": [100, 200]})
    l2, r2 = idc.align_join_keys(left, right, [("frame_id_int", "frame_id")])
    out = l2.merge(r2, left_on="frame_id_int", right_on="frame_id", how="left")
    assert out["w"].tolist() == [100, 200]


# --- Boxed-numeric object columns (the infer_ball_carrier hazard) ----------------------
#
# `infer_ball_carrier` emits `ball_carrier_team_id` as an OBJECT column holding FLOATS.
# Two independent defects made that silently un-joinable through the ADR-019 seam:
#   1. `canonical_id_series`'s object branch bare-stringified it, so 2.0 -> "2.0" while the
#      scalar `_canonical` truth gives "2" -- the function contradicted its own docstring
#      promise to "match `canonical_id` element-wise".
#   2. `_directly_comparable` short-circuited object-vs-object to a raw `==`, so a boxed
#      2.0 never equalled the string "2" even once (1) agreed they were the same id.
# Both produced an all-False mask -- a silent all-row join miss, not an error.


def _boxed_carrier():
    """Object column of floats, exactly as `infer_ball_carrier` emits it."""
    return pd.Series([2.0, 1.0], dtype=object)


def test_canonical_series_matches_scalar_truth_on_boxed_numeric_object():
    boxed = _boxed_carrier()
    assert list(idc.canonical_id_series(boxed)) == [idc.canonical_id(v) for v in boxed]
    # and it agrees with the same ids carried as a real numeric column
    assert list(idc.canonical_id_series(boxed)) == list(idc.canonical_id_series(pd.Series([2.0, 1.0], dtype="float64")))


def test_canonical_series_matches_scalar_truth_on_MIXED_object():
    """Mixed object columns must still go element-wise, incl. non-integral floats."""
    raw = [2.0, "DFL-CLU-1", 3, 2.5, None]
    mixed = pd.Series(raw, dtype=object)
    assert list(idc.canonical_id_series(mixed)) == [idc.canonical_id(v) for v in raw]


def test_ids_equal_resolves_boxed_object_against_numeric_column():
    """The reported live shape: object-boxed carrier ids vs a float64 frames column."""
    assert list(idc.ids_equal(pd.Series([2.0, 1.0], dtype="float64"), _boxed_carrier())) == [True, True]


def test_ids_helpers_resolve_boxed_object_against_STRING_object():
    """Both sides object -> the old dtype-only fast path raw-compared and missed everything."""
    boxed, strings = _boxed_carrier(), pd.Series(["2", "1"], dtype=object)
    assert list(idc.ids_equal(boxed, strings)) == [True, True]
    assert list(idc.ids_differ(boxed, strings)) == [False, False]
    assert list(idc.ids_match(boxed, "2")) == [True, False]


def test_genuine_string_object_columns_keep_the_raw_fast_path():
    """Non-vacuity guard for the fix: the sportec/kloppy path must NOT be canonicalized.

    `_raw_comparable` must still answer True for two genuine-string object columns, or the
    fix would have silently traded the object Python-loop regression back in.
    """
    s = pd.Series(["DFL-CLU-A", "DFL-CLU-B"], dtype=object)
    t = pd.Series(["DFL-CLU-A", "DFL-CLU-C"], dtype=object)
    assert idc._raw_comparable(s, t) is True
    assert list(idc.ids_equal(s, t)) == [True, False]
    assert list(idc.ids_differ(s, t)) == [False, True]
    # ...and it must answer False once either side stops being genuinely string-typed
    assert idc._raw_comparable(s, _boxed_carrier()) is False


def test_na_semantics_survive_the_content_probe():
    """NA never equals anything; NA on either side is not "differ". Unchanged by the probe."""
    left = pd.Series([None, "2"], dtype=object)
    right = pd.Series(["2", "1"], dtype=object)
    assert list(idc.ids_equal(left, right)) == [False, False]
    assert list(idc.ids_differ(left, right)) == [False, True]
    boxed_na = pd.Series([None, 2.0], dtype=object)
    assert list(idc.ids_equal(boxed_na, pd.Series(["9", "2"], dtype=object))) == [False, True]
    assert list(idc.ids_differ(boxed_na, pd.Series(["9", "2"], dtype=object))) == [False, False]


# --- restore_id_dtype: kernel-built id columns go back on their SOURCE dtype ------------
#
# Kernels assemble id results into `np.empty(n, dtype=object)` because a slot may be missing.
# The restoration used to be keyed on the single literal "Int64", so int64 / float64 / string
# sources all fell through and shipped an OBJECT column of boxed values -- the shape that made
# `ball_carrier_team_id` un-joinable. All three sibling sites now share one rule.


def _int64_dtype():
    return pd.Series([1], dtype="int64").dtype


def test_restore_id_dtype_round_trips_every_source_dtype():
    boxed = pd.Series([10.0, 11.0], dtype=object)  # what a kernel hands back
    for name in ("Int64", "int64", "float64", "string"):
        src = pd.Series([10, 11], dtype=name) if name != "float64" else pd.Series([10.0, 11.0], dtype=name)
        out = idc.restore_id_dtype(boxed, src.dtype)
        assert str(out.dtype) == name, f"{name} source did not round-trip (got {out.dtype})"


def test_restore_id_dtype_leaves_object_sources_faithful():
    """Object IS the source dtype there -- restoring must not invent a numeric column."""
    src = pd.Series(["DFL-CLU-A", "DFL-CLU-B"], dtype=object)
    out = idc.restore_id_dtype(pd.Series(["DFL-CLU-A", "DFL-CLU-B"], dtype=object), src.dtype)
    assert out.dtype == object
    assert list(out) == ["DFL-CLU-A", "DFL-CLU-B"]


def test_restore_id_dtype_will_not_narrow_a_numpy_int_that_must_hold_NA():
    """A numpy integer dtype CANNOT represent NA.

    This is the deliberate long-standing behaviour at the `features.py` / `_gk_resolve.py`
    sites: an unmatched action leaves NaN, so the result stays float rather than raising or
    silently coercing. Restorability, not blanket casting.
    """
    with_na = pd.Series([10.0, None], dtype=object)
    assert str(idc.restore_id_dtype(with_na, _int64_dtype()).dtype) == "float64"
    # ...but a nullable Int64 CAN hold NA, so that one restores.
    assert str(idc.restore_id_dtype(with_na, pd.Series([1], dtype="Int64").dtype).dtype) == "Int64"
    # ...and with nothing missing, the numpy int narrows as intended.
    assert str(idc.restore_id_dtype(pd.Series([10.0, 11.0], dtype=object), _int64_dtype()).dtype) == "int64"


def test_infer_ball_carrier_emits_the_source_dtype_not_object():
    """End-to-end at the site that shipped the live defect."""
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier

    def frames(dtype_name):
        rows = []
        for f in (1, 2):
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=f,
                    player_id=None,
                    team_id=0,
                    x=50.0,
                    y=34.0,
                    is_ball=True,
                    ball_state="alive",
                    vx=0.0,
                    vy=0.0,
                )
            )
            for p, (t, x) in enumerate([(2, 50.2), (1, 60.0)]):
                rows.append(
                    dict(
                        game_id=1,
                        period_id=1,
                        frame_id=f,
                        player_id=p + 10,
                        team_id=t,
                        x=x,
                        y=34.0,
                        is_ball=False,
                        ball_state="alive",
                        vx=0.0,
                        vy=0.0,
                    )
                )
        df = pd.DataFrame(rows)
        df["team_id"] = df["team_id"].astype(dtype_name)
        return df

    for name in ("Int64", "int64", "float64", "string"):
        col = infer_ball_carrier(frames(name))["ball_carrier_team_id"]
        assert str(col.dtype) == name, f"{name} source leaked {col.dtype}"
        # non-vacuity: the column must actually carry a resolved carrier, not be all-NA
        assert col.notna().any(), f"{name}: fixture produced no carrier, test would be vacuous"
