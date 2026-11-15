"""TF-79 follow-up: trajectory_dedup_mask collapses consecutive-duplicate ball positions."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_rest_run_longer_than_rest_s_is_kept_whole():
    # ADR-134: a run that lasts longer than rest_s is a ball at rest (a measurement), not a stale repeat: all its
    # samples stay. A run that lasts exactly rest_s or less is still collapsed.
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    y = np.zeros(6)
    rest = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])  # run 0.0 .. 0.75: 0.75 s
    assert trajectory_dedup_mask(rest, y, t=t, rest_s=0.5).tolist() == [True] * 6
    assert trajectory_dedup_mask(rest, y, t=t, rest_s=0.75).tolist() == [True, False, False, False, True, True]
    stale = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 3.0])  # run 0.25 .. 0.75: exactly 0.5 s
    assert trajectory_dedup_mask(stale, y, t=t, rest_s=0.5).tolist() == [True, True, False, False, True, True]


def test_rest_s_needs_t():
    with pytest.raises(ValueError, match="together"):
        trajectory_dedup_mask(np.zeros(3), np.zeros(3), rest_s=0.5)
    with pytest.raises(ValueError, match="together"):
        trajectory_dedup_mask(np.zeros(3), np.zeros(3), t=np.arange(3.0))


def test_nan_frames_are_kept_not_treated_as_duplicates():
    # NaN = non-detection, NOT a duplicate -> preserved (so detection-quality / gap handling is intact)
    x = np.array([0.0, np.nan, np.nan, 1.0, 1.0])
    y = np.array([0.0, np.nan, np.nan, 0.0, 0.0])
    assert trajectory_dedup_mask(x, y).tolist() == [True, True, True, True, False]
