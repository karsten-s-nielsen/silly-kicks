from silly_kicks.calibration._diagnostics import tf25_gate_fires


def test_tf25_fires_when_gap_exceeds_provider_se():
    # global k3 Brier worse than provider-best by more than that provider's CV SE => fire
    assert tf25_gate_fires(global_brier=0.060, provider_best_brier=0.050, provider_se=0.005) is True


def test_tf25_does_not_fire_within_se():
    assert tf25_gate_fires(global_brier=0.052, provider_best_brier=0.050, provider_se=0.005) is False


def test_tf25_nan_se_never_fires():
    # single-fold SE is nan => cannot justify provider-specific defaults
    assert tf25_gate_fires(global_brier=0.10, provider_best_brier=0.05, provider_se=float("nan")) is False
