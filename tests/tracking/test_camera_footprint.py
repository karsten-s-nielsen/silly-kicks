"""Task 2.3b camera-footprint geometry (spec section 17.7, DL ruling). Both sides of every bar; the
point-in-quadrilateral test is corner-order-independent and degenerate-safe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._camera_footprint import (
    footprint_inside_share,
    in_camera_view_mask,
    keeper_undetected_split,
    orient_camera_footprint,
)

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


def _footprint_frame(share_inside: float, n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One frame; `n` detected players, `share_inside` of them inside the square footprint."""
    n_in = round(share_inside * n)
    xs = [5.0] * n_in + [50.0] * (n - n_in)  # inside the 10x10 square / far outside
    frames = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": 7,
            "x": xs,
            "y": [5.0] * n,
            "is_ball": False,
            "is_goalkeeper": False,
            "visibility": True,  # all detected
        }
    )
    fp = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "frame_id": [7],
            "cam_tl_x": [0.0],
            "cam_tl_y": [0.0],
            "cam_tr_x": [10.0],
            "cam_tr_y": [0.0],
            "cam_br_x": [10.0],
            "cam_br_y": [10.0],
            "cam_bl_x": [0.0],
            "cam_bl_y": [10.0],
        }
    )
    return frames, fp


def test_footprint_inside_share_both_sides_of_the_gate():
    frames, fp = _footprint_frame(0.99)
    assert footprint_inside_share(frames, fp, tol_m=1.0) >= 0.99
    frames, fp = _footprint_frame(0.98)
    assert footprint_inside_share(frames, fp, tol_m=1.0) < 0.99


def test_footprint_inside_share_excludes_undetected():
    frames, fp = _footprint_frame(0.0)  # everyone far outside
    frames.loc[frames.index[0], "visibility"] = True  # one detected, outside
    frames.loc[frames.index[1:], "visibility"] = False  # the rest undetected -> excluded
    # only the single detected (outside) row counts -> share 0.0 over 1 evaluable
    assert footprint_inside_share(frames, fp, tol_m=1.0) == 0.0


def test_footprint_inside_share_tol_boundary():
    frames = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": 7,
            "x": [-0.9, -1.1],  # 0.9 m outside / 1.1 m outside the left edge
            "y": [5.0, 5.0],
            "is_ball": False,
            "is_goalkeeper": False,
            "visibility": True,
        }
    )
    _, fp = _footprint_frame(1.0, n=1)
    assert footprint_inside_share(frames, fp, tol_m=1.0) == 0.5  # the 0.9 m row counts, the 1.1 m does not


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


def test_tripwire_unflipped_footprint_fails_the_inside_gate():
    # 10 detected players on the RIGHT half (x=95). A CORRECT right-half footprint contains them -> share
    # 1.0 >= 0.99; an UNFLIPPED left-half footprint puts them all outside -> 0.0 < 0.99. The spec section 16.1
    # gate thus catches a footprint in a different orientation from the frames (DL ruling tripwire).
    frames = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 2,
            "frame_id": 1,
            "x": [95.0] * 10,
            "y": [34.0] * 10,
            "is_ball": False,
            "is_goalkeeper": False,
            "visibility": True,
        }
    )
    right = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [2],
            "frame_id": [1],
            "cam_tl_x": [52.5],
            "cam_tl_y": [68.0],
            "cam_tr_x": [105.0],
            "cam_tr_y": [68.0],
            "cam_br_x": [105.0],
            "cam_br_y": [0.0],
            "cam_bl_x": [52.5],
            "cam_bl_y": [0.0],
        }
    )
    left = right.copy()
    for c in ("cam_tl_x", "cam_tr_x", "cam_br_x", "cam_bl_x"):
        left[c] = 105.0 - left[c]  # mirror to the left half: the wrong orientation for these right-side players
    assert footprint_inside_share(frames, right, tol_m=1.0) >= 0.99
    assert footprint_inside_share(frames, left, tol_m=1.0) < 0.99


def test_keeper_undetected_split_counts_in_view_vs_off_camera():
    frames = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": [1, 2, 3, 4],
            "is_ball": False,
            "is_goalkeeper": True,
            "visibility": [False, False, False, True],  # 3 undetected, 1 detected
            "in_camera_view": pd.array([True, False, pd.NA, True], dtype="boolean"),
        }
    )
    split = keeper_undetected_split(frames)
    assert split["n_undetected_keeper_frames"] == 3
    assert split["in_view"] == 1 and split["off_camera"] == 1 and split["unknown_view"] == 1
