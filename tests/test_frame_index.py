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
