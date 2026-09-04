"""Trimmed defensive hull: exact area/centroid/membership, trimming, degenerate -> None."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.territory._hull import build_trimmed_hull


def test_square_hull_exact():
    # A 10x10 square, keep all (trim_fraction=1.0) -> area 100, centroid (5,5).
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    hull = build_trimmed_hull(square, trim_fraction=1.0)
    assert hull is not None
    assert hull.area == pytest.approx(100.0)
    assert hull.centroid == (5.0, 5.0)  # mean of the square is exact
    # membership (vectorized + single point)
    got = hull.contains(np.array([[5.0, 5.0], [15.0, 15.0], [0.0, 0.0]]))
    assert bool(got[0]) is True  # inside
    assert bool(got[1]) is False  # outside
    assert bool(hull.contains(np.array([2.0, 3.0]))) is True


def test_trimming_excludes_outliers():
    # 6 points tightly around a 2x2 box + 2 far outliers; trim 0.75 keeps 6 -> the outliers drop, so the
    # hull area is the inlier hull's, far smaller than the full 8-point hull.
    inliers = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 0.0], [1.0, 2.0]])
    outliers = np.array([[100.0, 100.0], [-100.0, -100.0]])
    pts = np.vstack([inliers, outliers])
    trimmed = build_trimmed_hull(pts, trim_fraction=0.75)
    full = build_trimmed_hull(pts, trim_fraction=1.0)
    assert trimmed is not None and full is not None
    assert trimmed.area == pytest.approx(4.0)  # the 2x2 inlier box
    assert full.area > trimmed.area  # the outliers blow the untrimmed hull up


def test_degenerate_returns_none():
    assert build_trimmed_hull(np.array([[0.0, 0.0], [1.0, 1.0]]), trim_fraction=1.0) is None  # 2 points
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert build_trimmed_hull(collinear, trim_fraction=1.0) is None  # collinear -> no area
    assert build_trimmed_hull(np.empty((0, 2)), trim_fraction=1.0) is None  # empty


def test_nan_rows_dropped():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [np.nan, 5.0]])
    hull = build_trimmed_hull(pts, trim_fraction=1.0)
    assert hull is not None and hull.area == pytest.approx(100.0)
