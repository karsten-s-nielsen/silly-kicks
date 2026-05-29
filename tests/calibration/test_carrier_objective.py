from ruthless import Candidate

from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective

_P = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}


def test_perfect_carrier_fixture_scores_one(synth_known_carrier):
    obj = CarrierAccuracyObjective(synth_known_carrier)
    metrics = obj.evaluate(Candidate(id="t0", params=_P))
    assert metrics["carrier_accuracy"] >= 0.99
    assert "carrier_accuracy__provA" in metrics  # per-provider attr present


def test_equal_provider_weighting(synth_two_providers_imbalanced):
    # provA: 100 matches @ acc 1.0; provB: 1 match @ acc 0.0 => equal-weight mean = 0.5
    obj = CarrierAccuracyObjective(synth_two_providers_imbalanced)
    metrics = obj.evaluate(Candidate(id="t0", params=_P))
    assert metrics["carrier_accuracy"] == 0.5  # NOT match-count-weighted (would be ~0.99)
