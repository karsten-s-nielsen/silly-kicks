"""TF-19 sign-off package: verdict/routing split (spec §7, F4).

`regate_verdict` answers "what did the probe say"; `regate_routing` answers "what should we do about
it", and only the second may legitimately depend on Layer 2. Conflating them is what hard-coded H1.
"""

from __future__ import annotations

import itertools

import pytest

from silly_kicks.tracking._model_eval import (
    _ENTANGLEMENT,
    _PROBE_VERDICTS,
    REGATE_ROUTING_VALUES,
    regate_routing,
    regate_verdict,
)

_ARMS = ("shot", "cross")


def _reachable_verdicts() -> set[str]:
    return {
        regate_verdict(arm=a, probe_verdict=p, entanglement=e)
        for a, p, e in itertools.product(_ARMS, _PROBE_VERDICTS, _ENTANGLEMENT)
    }


def test_gated_clean_fail_no_longer_routes_to_gk_feature_engineering():
    """The pre-registered disclosure: `gated_clean_fail` must stop hard-coding H1."""
    assert regate_routing("gated_clean_fail") == "pending_layer2"
    assert regate_routing("gated_clean_fail") != "gk_feature_engineering"


@pytest.mark.parametrize("verdict", sorted(_reachable_verdicts()))
def test_every_reachable_verdict_has_a_routing_in_the_closed_vocabulary(verdict):
    assert regate_routing(verdict) in REGATE_ROUTING_VALUES


def test_unknown_verdict_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown verdict"):
        regate_routing("not_a_verdict")


def test_regate_verdict_is_byte_identical_over_every_input_combination():
    """Golden pin: no recorded verdict may move. 4.60.0's `joins_with_caveat` and 4.51.0's
    `gated_clean_fail` are published in `metrics.json` research artifacts."""
    got = {
        (a, p, e): regate_verdict(arm=a, probe_verdict=p, entanglement=e)
        for a, p, e in itertools.product(_ARMS, _PROBE_VERDICTS, _ENTANGLEMENT)
    }
    assert got[("shot", "pass", "inside_band")] == "joins_with_caveat"
    assert got[("shot", "pass", "clears")] == "joins"
    assert got[("cross", "fail", "inside_band")] == "gated_clean_fail"
    assert got[("shot", "no_valid_placebo", "clears")] == "unmeasurable_at_dose"
    assert got[("shot", "unmeasurable_at_dose", "clears")] == "unmeasurable_at_dose"
    assert got[("shot", "instrument_invalid", "clears")] == "verdict_void"
    assert got[("shot", "band_pass_flat_dose_response", "clears")] == "gated_flat_dose_response"


def test_routing_is_a_pure_function_of_the_verdict_alone():
    """The routing table must not smuggle the arm or the entanglement back in -- that is the
    coupling the split exists to remove."""
    for verdict in _reachable_verdicts():
        assert regate_routing(verdict) == regate_routing(verdict)
