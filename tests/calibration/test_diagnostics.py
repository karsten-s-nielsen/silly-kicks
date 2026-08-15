import math

import pytest

from silly_kicks.calibration._diagnostics import exceeds_noise_floor, tf25_gate_fires


def test_tf25_fires_when_gap_exceeds_provider_se():
    # global k3 Brier worse than provider-best by more than that provider's CV SE => fire
    assert tf25_gate_fires(global_brier=0.060, provider_best_brier=0.050, provider_se=0.005) is True


def test_tf25_does_not_fire_within_se():
    assert tf25_gate_fires(global_brier=0.052, provider_best_brier=0.050, provider_se=0.005) is False


def test_tf25_nan_se_never_fires():
    # single-fold SE is nan => cannot justify provider-specific defaults
    assert tf25_gate_fires(global_brier=0.10, provider_best_brier=0.05, provider_se=float("nan")) is False


@pytest.mark.parametrize("se", [None, math.nan, math.inf])
def test_exceeds_noise_floor_non_finite_se_never_clears(se):
    assert exceeds_noise_floor(1.0, se) is False


def test_exceeds_noise_floor_strict_boundary():
    assert exceeds_noise_floor(0.06, 0.05) is True
    assert exceeds_noise_floor(0.05, 0.05) is False  # strict >


@pytest.mark.parametrize("se", [0.005, math.nan, math.inf])
def test_tf25_gate_verdict_unchanged_across_finite_nan_inf(se):
    # RED-first: pin the observable verdict is identical after the refactor.
    # gap = 0.06 - 0.05 = 0.01; finite se 0.005 -> True; nan/inf -> False.
    expected = se == 0.005
    assert tf25_gate_fires(global_brier=0.06, provider_best_brier=0.05, provider_se=se) is expected
