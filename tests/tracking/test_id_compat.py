"""Unit tests for the tracking id-dtype safety primitive (ADR-019)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _id_compat as idc

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
    # single-truth: vectorized output == scalar `_canonical`, fed per NATURAL dtype (real id
    # columns are homogeneous, not a mixed object dump — an integral float in an object column
    # is the one documented pathological divergence, excluded here).
    for dtype, vals, exp in [
        ("int64", [366, 7], ["366", "7"]),
        ("Int64", [366, 7], ["366", "7"]),
        ("float64", [366.0, 366.5], ["366", "366.5"]),
        ("object", ["366", "DFL-CLU-A"], ["366", "DFL-CLU-A"]),
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


def test_align_join_keys_pair_names():  # A3: differently-named keys (frame_id_int vs frame_id)
    left = pd.DataFrame({"frame_id_int": pd.Series([1, 2], dtype="int64"), "v": [10, 20]})
    right = pd.DataFrame({"frame_id": pd.Series(["1", "2"], dtype="object"), "w": [100, 200]})
    l2, r2 = idc.align_join_keys(left, right, [("frame_id_int", "frame_id")])
    out = l2.merge(r2, left_on="frame_id_int", right_on="frame_id", how="left")
    assert out["w"].tolist() == [100, 200]
