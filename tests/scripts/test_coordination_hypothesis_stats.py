"""A-53 (remaining): paired sign test excludes ties; TOST treats all-identical-within-margin as equivalent.

Both are standard corrections to the paired tests H1/H5 use (``_coordination_hypotheses``): a tie carries no sign,
so it must leave the trial count; and a zero-variance difference set that sits inside the equivalence margin is
trivially equivalent (a t-test cannot run on it, but the answer is not "not equivalent").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from _coordination_hypotheses import _sign_test_greater, _tost_equivalent


def test_sign_test_excludes_ties_from_the_trial_count():
    a = np.array([2.0, 2.0, 3.0, 1.0])
    b = np.array([1.0, 2.0, 2.0, 2.0])  # idx1 is a tie (2 == 2)
    p, n, share = _sign_test_greater(a, b)
    assert n == 3  # the tie is dropped, not counted as a non-win
    assert share == 2 / 3  # two wins out of three non-tied pairs
    assert np.isfinite(p)


def test_sign_test_all_tied_is_empty():
    a = np.array([1.0, 2.0, 3.0])
    p, n, share = _sign_test_greater(a, a)
    assert n == 0
    assert np.isnan(p) and np.isnan(share)


def test_tost_all_identical_within_margin_is_equivalent():
    diffs = np.array([0.01, 0.01, 0.01, 0.01])
    equiv, _p_lo, _p_hi = _tost_equivalent(diffs, margin=0.05, alpha=0.05)
    assert equiv is True  # a zero-variance difference set inside +-margin is trivially equivalent


def test_tost_all_identical_outside_margin_is_not_equivalent():
    diffs = np.array([0.1, 0.1, 0.1])
    equiv, _p_lo, _p_hi = _tost_equivalent(diffs, margin=0.05, alpha=0.05)
    assert equiv is False


def test_tost_normal_variance_path_still_runs():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.0, 0.005, 50)  # tightly around 0, well inside +-0.05
    equiv, p_lo, p_hi = _tost_equivalent(diffs, margin=0.05, alpha=0.05)
    assert equiv is True
    assert np.isfinite(p_lo) and np.isfinite(p_hi)
