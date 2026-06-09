"""Regular-suite unit tests for the pure NLL verdict helpers used by the owner-gated e2e."""

import math

from tests._xthreat_helpers import kde_clears_tripwire, nll_relative_win


def test_relative_win_positive_when_candidate_lower():
    assert math.isclose(nll_relative_win(4.0, 3.8), 0.05, rel_tol=1e-9)  # 5% improvement


def test_relative_win_negative_when_candidate_higher():
    assert nll_relative_win(4.0, 4.2) < 0


def test_relative_win_exactly_at_floor():
    assert math.isclose(nll_relative_win(4.0, 3.94), 0.015, rel_tol=1e-9)


def test_relative_win_nan_when_baseline_nonfinite_or_zero():
    assert math.isnan(nll_relative_win(float("nan"), 3.8))
    assert math.isnan(nll_relative_win(0.0, 3.8))


def test_relative_win_nan_when_candidate_nan():
    # empty-corpus holdout -> compute_holdout_nll returns nan for the candidate
    assert math.isnan(nll_relative_win(4.0, float("nan")))


def test_tripwire_clears_well_above_floor():
    assert kde_clears_tripwire(4.0, 3.8, floor=0.015) is True  # 5% >> 1.5%


def test_tripwire_fails_just_below_floor():
    assert kde_clears_tripwire(4.0, 3.95, floor=0.015) is False  # 1.25% < 1.5%


def test_tripwire_true_exactly_at_floor():
    assert kde_clears_tripwire(4.0, 3.94, floor=0.015) is True  # 1.5% == floor


def test_tripwire_false_when_kde_loses():
    assert kde_clears_tripwire(4.0, 4.2, floor=0.015) is False


def test_tripwire_false_on_nan():
    assert kde_clears_tripwire(float("nan"), 3.8, floor=0.015) is False
    assert kde_clears_tripwire(4.0, float("nan"), floor=0.015) is False


def test_tripwire_strict_beat_with_zero_floor():
    # floor=0.0 == strict-beat (the shipped-default KDE(1.0) contract): any win clears, tie/loss fails.
    assert kde_clears_tripwire(4.0, 3.99, floor=0.0) is True
    assert kde_clears_tripwire(4.0, 4.0, floor=0.0) is False
    assert kde_clears_tripwire(4.0, 4.1, floor=0.0) is False
