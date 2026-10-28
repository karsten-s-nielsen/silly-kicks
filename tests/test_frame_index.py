"""Tests for the ``silly_kicks._frame_index`` row-group lookup seam (ADR-068)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks._frame_index import group_rows


def _frames():
    return pd.DataFrame(
        {
            "game_id": pd.array([1, 1, 1, 2, 2], dtype="Int64"),
            "frame_id": pd.array([10, 10, 11, 10, 10], dtype="Int64"),
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


def test_single_key_lookup_matches_boolean_filter():
    df = _frames()
    g = group_rows(df, "frame_id")
    # dtype-agnostic key: Python int / str / numpy int all resolve the Int64 group
    for key in (10, "10", np.int64(10)):
        got = g.get(key)
        exp = df[df["frame_id"] == 10]
        pd.testing.assert_frame_equal(got, exp)


def test_multi_key_lookup_matches_boolean_filter():
    df = _frames()
    g = group_rows(df, ("game_id", "frame_id"))
    got = g.get(2, 10)
    exp = df[(df["game_id"] == 2) & (df["frame_id"] == 10)]
    pd.testing.assert_frame_equal(got, exp)


def test_missing_key_returns_empty_frame_not_keyerror():
    df = _frames()
    g = group_rows(df, "frame_id")
    out = g.get(999)  # absent
    assert out.empty
    assert list(out.columns) == list(df.columns)
    assert out.dtypes.equals(df.dtypes)


def test_within_group_order_preserved():
    df = _frames()
    g = group_rows(df, "frame_id")
    # both games' frame_id==10 rows, in source order (x = 0.0, 1.0, 3.0, 4.0)
    assert g.get(10)["x"].tolist() == [0.0, 1.0, 3.0, 4.0]


def test_mixed_dtype_key_raises_not_silent_row_loss():
    # F1(a): int 366 and str "366" are distinct groups that canonicalize equal -> refuse loud.
    df = pd.DataFrame({"k": pd.array([366, "366"], dtype="object"), "x": [0.0, 1.0]})
    with pytest.raises(ValueError, match=r"collapsed under|mixes dtypes"):
        group_rows(df, "k")


def test_contains_single_and_multi_key():
    df = _frames()
    assert 10 in group_rows(df, "frame_id")
    assert 999 not in group_rows(df, "frame_id")
    g = group_rows(df, ("game_id", "frame_id"))
    assert (2, 10) in g  # multi-key membership: pass a tuple
    assert (2, 11) not in g


def test_keys_returns_canonical_keys_usable_in_get():
    df = _frames()
    # single-key: canonical scalars (Int64 -> str "10"/"11"); each round-trips through get()
    g = group_rows(df, "frame_id")
    assert set(g.keys()) == {"10", "11"}
    for k in g.keys():
        assert not g.get(k).empty
    # multi-key: canonical tuples, splattable into get() -> the same rows the boolean filter gives
    gm = group_rows(df, ("game_id", "frame_id"))
    assert set(gm.keys()) == {("1", "10"), ("1", "11"), ("2", "10")}
    got = gm.get(*("2", "10"))
    exp = df[(df["game_id"] == 2) & (df["frame_id"] == 10)]
    pd.testing.assert_frame_equal(got, exp)


def test_group_rows_category_key_is_observed_only_no_phantom_groups():
    """ADR-105 Task 6: a `category` key whose declared categories are WIDER than observed yields ONLY
    the observed groups (no phantom zero-row category keys) and emits no FutureWarning -- observed=True
    is pinned, so behaviour is deterministic across pandas majors (ADR-057). Defensive: ADR-103 made
    frame columns category, and id->category is a deferred retrain-cycle candidate."""
    import warnings

    from silly_kicks._frame_index import group_rows

    df = pd.DataFrame({"k": pd.Categorical(["a", "a", "b"], categories=["a", "b", "z"]), "v": [1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        g = group_rows(df, "k")
    assert set(g._indices) == {"a", "b"}  # NOT "z" -- the zero-row category is not a phantom group
    assert g.get("z").empty  # a genuinely-absent key still misses (empty frame)
    assert len(g.get("a")) == 2


def _same_key_part(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    return bool(a == b)


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("kind", ["int", "str", "Int64", "float_nan", "object_none", "object_mixed_none"])
def test_ordered_group_rows_is_the_groupby_walk(kind, sort, dropna):
    """Spec r4 D20: the extract loops walk ``groupby`` by positions. The walk must be the iteration's own:
    the same key objects (type and value, NA keys included), the same rows, in the same group order."""
    from silly_kicks._frame_index import ordered_group_rows

    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {
            "game_id": rng.integers(1, 3, n),
            "period_id": rng.integers(1, 3, n),
            "frame_id": rng.integers(0, 15, n),
            "x": rng.random(n),
        }
    )
    if kind == "str":
        df["game_id"] = df["game_id"].map({1: "g1", 2: "g2"}).astype(object)
    elif kind == "Int64":
        df["game_id"] = df["game_id"].astype("Int64")
        df.loc[df.index[::17], "game_id"] = pd.NA
    elif kind == "float_nan":
        df["frame_id"] = df["frame_id"].astype(float)
        df.loc[df.index[::13], "frame_id"] = np.nan
    elif kind in ("object_none", "object_mixed_none"):  # an object key with a missing value (B review R4I-04)
        values = {1: "g1", 2: "g2"} if kind == "object_none" else {1: 1, 2: "1"}
        df["game_id"] = df["game_id"].map(values).astype(object)
        df.loc[df.index[::11], "game_id"] = None
    df = df.sample(frac=1.0, random_state=5)  # shuffled index labels: the walk is positional
    by = ["game_id", "period_id", "frame_id"]
    marked = df.assign(__pos=np.arange(len(df)))
    expected = [(key, group["__pos"].to_numpy()) for key, group in marked.groupby(by, sort=sort, dropna=dropna)]
    got = ordered_group_rows(df, by, sort=sort, dropna=dropna)
    assert len(got) == len(expected) > 0
    for (key, rows), (want_key, want_rows) in zip(got, expected, strict=True):
        assert isinstance(key, tuple) and len(key) == len(want_key)
        assert all(_same_key_part(a, b) for a, b in zip(key, want_key, strict=True)), (key, want_key)
        assert np.array_equal(rows, want_rows)


def test_ordered_group_rows_refuses_one_key_column():
    """A one-column groupby iterates its keys as Python scalars, which ``indices`` does not hold: refuse rather than
    hand on keys of another type."""
    from silly_kicks._frame_index import ordered_group_rows

    with pytest.raises(ValueError, match="two or more key columns"):
        ordered_group_rows(_frames(), ["frame_id"])
