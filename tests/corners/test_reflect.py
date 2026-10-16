"""TF-46 corner-side reflection (Y-normalisation)."""

from __future__ import annotations

import pandas as pd

from silly_kicks.corners import reflect_corners_to_canonical_side
from silly_kicks.corners._reflect import CORNER_REFLECTION_KINDS


def _df():
    return pd.DataFrame(
        {
            "corner_side": ["left", "right"],
            "kick_x": [104.0, 104.0],
            "kick_y": [60.0, 8.0],
        }
    )


def test_right_side_y_flips_left_unchanged():
    out = reflect_corners_to_canonical_side(_df())
    assert out.loc[0, "kick_y"] == 60.0  # left: unchanged
    assert out.loc[1, "kick_y"] == 60.0  # right: 68 - 8 = 60


def test_x_is_invariant():
    out = reflect_corners_to_canonical_side(_df())
    assert list(out["kick_x"]) == [104.0, 104.0]


def test_pure_returns_new_frame():
    df = _df()
    before = df.copy(deep=True)
    out = reflect_corners_to_canonical_side(df)
    pd.testing.assert_frame_equal(df, before)
    assert out is not df


def test_missing_side_column_raises():
    try:
        reflect_corners_to_canonical_side(pd.DataFrame({"kick_y": [1.0]}))
    except KeyError as e:
        assert "corner_side" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_kinds_map_declares_only_known_kinds():
    assert set(CORNER_REFLECTION_KINDS.values()) <= {
        "point_x",
        "point_y",
        "vector_x",
        "vector_y",
        "invariant",
    }
