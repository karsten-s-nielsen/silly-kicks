"""Task 1: pure metric helpers for the RQ validation cycle (no corpus, no I/O)."""

from __future__ import annotations

import numpy as np

from scripts import _rq_metrics as M


def test_false_positive_rate_conditions_on_completed_only():
    # 4 passes: blocked/completed, blocked/completed, blocked/failed, open/completed
    is_blocked = np.array([True, True, True, False])
    is_completed = np.array([True, True, False, True])
    # P(blocked | completed): completed = idx 0,1,3 -> blocked at 0,1 -> 2/3
    assert abs(M.false_positive_rate(is_blocked, is_completed) - 2 / 3) < 1e-9


def test_false_alarm_rate_completed_only():
    control = np.array([0.05, 0.15, 0.5, 0.05])
    is_completed = np.array([True, True, True, False])  # last is failed -> excluded
    # tau=0.1: completed = 0,1,2 -> control<0.1 at idx0 -> 1/3
    assert abs(M.false_alarm_rate(control, is_completed, tau=0.1) - 1 / 3) < 1e-9


def test_auc_nan_safe_on_single_class():
    assert np.isnan(M.auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
    assert M.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_confusion_balanced_accuracy():
    pred = np.array([True, True, False, False])
    actual_pos = np.array([True, False, False, True])
    c = M.confusion(pred, actual_pos)
    assert c["tp"] == 1 and c["fp"] == 1 and c["tn"] == 1 and c["fn"] == 1
    assert abs(c["balanced_accuracy"] - 0.5) < 1e-9


def test_reliability_curve_bins_and_empirical_rate():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.05, 0.15, 0.85, 0.95])
    out = M.reliability_curve(y, score, n_bins=10)
    # two occupied low bins (rate 0) + two occupied high bins (rate 1)
    assert out["emp_rate"][0] == 0.0 and out["emp_rate"][-1] == 1.0
    assert sum(out["n"]) == 4


def test_low_control_completion_band_over_all_passes():
    control = np.array([0.05, 0.25, 0.05])
    is_success = np.array([True, False, False])
    band = M.low_control_completion_band(control, is_success, taus=(0.1,))
    # control<0.1 -> idx 0,2 -> success rate 1/2
    assert abs(band[0.1] - 0.5) < 1e-9


def test_metrics_are_nan_safe_on_score():
    """REGRESSION (real GS data): ~0.8% of control + a stray p_blocked are non-finite; the
    library ece/reliability_slope (np.polyfit) raise `SVD did not converge` on NaN. Every
    score-consuming metric must drop non-finite scores, not crash."""
    y = np.array([0, 0, 1, 1, 0])
    score = np.array([0.1, np.nan, 0.9, 0.8, np.inf])
    M.auc(y, score)  # these must NOT raise
    M.ece(y, score)
    M.reliability_slope(y, score)
    curve = M.reliability_curve(y, score)
    assert sum(curve["n"]) == 3  # only the 3 finite scores are binned
    # false_alarm_rate drops NaN control from the completed denominator
    fa = M.false_alarm_rate(np.array([0.05, np.nan, 0.5]), np.array([True, True, True]), tau=0.1)
    assert abs(fa - 0.5) < 1e-9  # finite completed = idx 0,2 -> control<0.1 at idx0 -> 1/2


def test_reliability_metrics_handle_p_blocked_above_one():
    """`p_blocked` is an unbounded blocking INTENSITY (real max ~2.3), not a [0,1] probability;
    the binning must clip it into the last bin rather than crash or drop it."""
    y = np.array([0, 1, 1, 0])
    score = np.array([0.2, 2.29, 1.8, 0.1])  # two scores > 1
    curve = M.reliability_curve(y, score)
    assert sum(curve["n"]) == 4  # all four kept (>1 clipped into the last bin)
