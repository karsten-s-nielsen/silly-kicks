"""TF-79 follow-up: trajectory_dedup_mask collapses consecutive-duplicate ball positions."""

from __future__ import annotations

import numpy as np

from silly_kicks.tracking.preprocess._dedup import trajectory_dedup_mask


def test_collapses_consecutive_runs():
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
    y = np.zeros(6)
    assert trajectory_dedup_mask(x, y).tolist() == [True, False, False, True, False, True]


def test_preserves_a_b_a_oscillation():
    x = np.array([0.0, 1.0, 0.0])
    y = np.zeros(3)
    assert trajectory_dedup_mask(x, y).tolist() == [True, True, True]


def test_all_duplicates_collapse_to_one():
    x = np.full(5, 3.0)
    y = np.full(5, 4.0)
    assert trajectory_dedup_mask(x, y).tolist() == [True, False, False, False, False]


def test_float32_exact_duplicates():
    x = np.array([1.1, 1.1, 1.2], dtype=np.float32).astype(np.float64)
    y = np.zeros(3)
    assert trajectory_dedup_mask(x, y).tolist() == [True, False, True]


def test_tol_boundary_keeps_motion_above_tol():
    x = np.array([0.0, 2e-6, 4e-6])  # 2e-6 m steps > 1e-6 tol
    y = np.zeros(3)
    assert trajectory_dedup_mask(x, y).tolist() == [True, True, True]


def test_single_point():
    assert trajectory_dedup_mask(np.array([0.0]), np.array([0.0])).tolist() == [True]


def test_empty():
    assert trajectory_dedup_mask(np.array([]), np.array([])).tolist() == []


def test_nan_frames_are_kept_not_treated_as_duplicates():
    # NaN = non-detection, NOT a duplicate -> preserved (so detection-quality / gap handling is intact)
    x = np.array([0.0, np.nan, np.nan, 1.0, 1.0])
    y = np.array([0.0, np.nan, np.nan, 0.0, 0.0])
    assert trajectory_dedup_mask(x, y).tolist() == [True, True, True, True, False]
