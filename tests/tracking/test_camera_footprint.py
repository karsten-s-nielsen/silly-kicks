"""Task 2.3b camera-footprint geometry (spec section 17.7, DL ruling). Both sides of every bar; the
point-in-quadrilateral test is corner-order-independent and degenerate-safe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._camera_footprint import in_camera_view_mask, orient_camera_footprint

# A 10x10 square footprint, corners (tl, tr, br, bl).
_SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def _tile(corners: np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(corners, (n, 4, 2)).copy()


def test_in_camera_view_inside_outside_edge_and_tol():
    x = np.array([5.0, 15.0, 0.0, -0.9, -1.1])
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    m = in_camera_view_mask(x, y, _tile(_SQUARE, 5), tol_m=1.0)
    assert list(m) == [True, False, True, True, False]  # inside / outside / on-edge / within tol / beyond tol


def test_in_camera_view_tol_zero_edge_is_inside():
    m = in_camera_view_mask(np.array([0.0]), np.array([5.0]), _tile(_SQUARE, 1), tol_m=0.0)
    assert bool(m[0]) is True  # exactly on the edge counts inside


def test_in_camera_view_corner_order_independent():
    reversed_sq = _SQUARE[::-1].copy()  # bl, br, tr, tl -- opposite winding
    x = np.array([5.0, 15.0])
    y = np.array([5.0, 5.0])
    m1 = in_camera_view_mask(x, y, _tile(_SQUARE, 2))
    m2 = in_camera_view_mask(x, y, _tile(reversed_sq, 2))
    assert list(m1) == list(m2) == [True, False]


def test_in_camera_view_na_on_nan_position_and_degenerate_quad():
    # NaN own position -> <NA>
    m = in_camera_view_mask(np.array([np.nan]), np.array([5.0]), _tile(_SQUARE, 1))
    assert m[0] is pd.NA
    # non-convex ("bowtie") quad -> <NA>
    bowtie = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    m = in_camera_view_mask(np.array([5.0]), np.array([5.0]), _tile(bowtie, 1))
    assert m[0] is pd.NA
    # zero-area (all corners coincident) -> <NA>
    degen = np.zeros((1, 4, 2))
    assert in_camera_view_mask(np.array([0.0]), np.array([0.0]), degen)[0] is pd.NA
    # NaN corner -> <NA>
    nan_c = _tile(_SQUARE, 1)
    nan_c[0, 1, 0] = np.nan
    assert in_camera_view_mask(np.array([5.0]), np.array([5.0]), nan_c)[0] is pd.NA


def test_orient_camera_footprint_reflects_flagged_period_only():
    fp = pd.DataFrame(
        {
            "game_id": ["g", "g"],
            "period_id": [1, 2],
            "frame_id": [1, 2],
            "cam_tl_x": [10.0, 10.0],
            "cam_tl_y": [20.0, 20.0],
            "cam_tr_x": [30.0, 30.0],
            "cam_tr_y": [20.0, 20.0],
            "cam_br_x": [30.0, 30.0],
            "cam_br_y": [40.0, 40.0],
            "cam_bl_x": [10.0, 10.0],
            "cam_bl_y": [40.0, 40.0],
        }
    )
    out = orient_camera_footprint(fp, {("g", 1): True, ("g", 2): False})
    # period 1 flipped (105 - x, 68 - y); period 2 unchanged.
    assert out.loc[0, "cam_tl_x"] == 105.0 - 10.0 and out.loc[0, "cam_tl_y"] == 68.0 - 20.0
    assert out.loc[1, "cam_tl_x"] == 10.0 and out.loc[1, "cam_tl_y"] == 20.0
