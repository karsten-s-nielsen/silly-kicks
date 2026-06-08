import pytest
from ruthless import Candidate

import silly_kicks.calibration._carrier_objective as carrier_obj
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


def test_unreachable_actor_counts_as_miss(synth_unreachable_actor):
    """A linked carrier-actor action whose actor is beyond tolerance_m (-> NaN inference)
    must count as a MISS, not be silently excluded.

    One hit + one tolerance-miss => 0.5. The old precision-only objective scored this 1.0
    (it averaged only over actions where a carrier was inferred), which is exactly why
    tolerance_m collapsed to the search lower bound. This guards the recall-aware fix.
    """
    obj = CarrierAccuracyObjective(synth_unreachable_actor)
    metrics = obj.evaluate(Candidate(id="t0", params=_P))
    assert metrics["carrier_accuracy"] == 0.5
    assert metrics["n_compared__provA"] == 2.0  # both linked actions are in the denominator


def test_link_failure_excluded_not_penalized(synth_link_failure):
    """A genuine link failure (no frame near the action time) is excluded from the
    denominator, never penalized — link success is independent of the swept carrier
    params. One hit + one unlinkable action => 1.0 over the single linked action.
    """
    obj = CarrierAccuracyObjective(synth_link_failure)
    metrics = obj.evaluate(Candidate(id="t0", params=_P))
    assert metrics["carrier_accuracy"] == 1.0
    assert metrics["n_compared__provA"] == 1.0  # only the linked action counts


@pytest.mark.slow
def test_prepare_cached_once_and_matches_uncached(synth_two_providers_imbalanced, monkeypatch):
    """The per-match invariant prepare runs ONCE per match and is reused across trials, and
    the cached evaluate is bit-identical to the uncached one-shot reference (_match_accuracy).

    This is the cache-equivalence guard for the invariant-prepare optimization: the pre-index +
    linking (param-invariant) are cached, so only the carrier kernel re-runs per candidate.
    """
    calls = {"n": 0}
    real_prepare = carrier_obj._prepare_match

    def spy(actions, frames):
        calls["n"] += 1
        return real_prepare(actions, frames)

    monkeypatch.setattr(carrier_obj, "_prepare_match", spy)

    fold = synth_two_providers_imbalanced  # provA: 100 matches, provB: 1 match => 101 matches
    obj = CarrierAccuracyObjective(fold)
    p1 = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}
    p2 = {"tolerance_m": 7.0, "beta": 0.0, "gamma": 0.1}

    m1 = obj.evaluate(Candidate(id="a", params=p1))
    m2 = obj.evaluate(Candidate(id="b", params=p2))

    n_matches = sum(len(v) for v in fold.values())
    assert calls["n"] == n_matches  # prepared exactly once each; 2nd evaluate reuses the cache

    # Cached metric equals the uncached one-shot reference for each candidate.
    for params, metric in [(p1, m1), (p2, m2)]:
        ref_per_provider = []
        for _provider, matches in fold.items():
            accs, weights = [], []
            for actions, frames, _home in matches:
                acc, n = carrier_obj._match_accuracy(actions, frames, **params)
                if n > 0:
                    accs.append(acc)
                    weights.append(n)
            if accs:
                import numpy as np

                ref_per_provider.append(float(np.average(accs, weights=weights)))
        import numpy as np

        assert metric["carrier_accuracy"] == float(np.mean(ref_per_provider))
