"""TF-65 Task 12: C1 default smoother stays the SG floor; method docstring lists all methods."""

from __future__ import annotations

from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames


def test_default_smoothing_method_is_savgol_floor():
    assert PreprocessConfig.default().smoothing_method == "savgol"


def test_method_docstring_lists_all_methods():
    doc = smooth_frames.__doc__ or ""
    for m in ("savgol", "ema", "butterworth", "kalman"):
        assert m in doc
