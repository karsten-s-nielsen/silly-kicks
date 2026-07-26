"""TF-19 sign-off package: the Layer 3 headroom derivation (spec §6).

The duty is "state `openGoal`'s units and observed range FIRST, then set the threshold as a stated
fraction of that range" -- so the arithmetic that turns a measured range into the registered
threshold is the load-bearing part, and it is pure.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.derive_opengoal_range as mod  # bare import: tests/scripts/ has NO __init__.py
from silly_kicks.gkdv._validate import LAYER3_HEADROOM_RANGE_FRACTION


def test_threshold_is_the_registered_fraction_of_the_OBSERVED_range():
    out = mod.summarise([0.0, 0.5, 1.0])
    assert out["observed_range"] == pytest.approx(1.0)
    assert out["layer3_headroom_threshold"] == pytest.approx(LAYER3_HEADROOM_RANGE_FRACTION)


def test_a_narrower_corpus_yields_a_proportionally_smaller_threshold():
    """The threshold TRACKS the range -- that is the whole point of expressing it as a fraction
    rather than as the bare 0.02 the spec guessed. A test on one corpus width cannot see this."""
    wide = mod.summarise([0.0, 1.0])
    narrow = mod.summarise([0.4, 0.6])
    assert narrow["observed_range"] == pytest.approx(0.2)
    assert narrow["layer3_headroom_threshold"] == pytest.approx(wide["layer3_headroom_threshold"] * 0.2)


def test_non_finite_values_are_dropped_not_propagated():
    """`_open_goal_fraction` returns NaN when the ball is on/behind the goal line, so the corpus
    carries them by design."""
    out = mod.summarise([0.2, np.nan, 0.8, np.inf])
    assert out["n"] == 2
    assert out["min"] == pytest.approx(0.2)
    assert out["max"] == pytest.approx(0.8)


def test_an_empty_corpus_refuses_rather_than_deriving_from_nothing():
    with pytest.raises(ValueError, match="refusing to derive"):
        mod.summarise([np.nan, np.inf])


def test_units_are_reported_because_the_duty_demands_them():
    """ "A reader cannot currently tell whether 0.02 is generous or unreachable" is the duty's own
    stated reason for requiring the units and range alongside the number."""
    out = mod.summarise([0.0, 1.0])
    assert "fraction" in out["units"]
    assert out["range_fraction"] == LAYER3_HEADROOM_RANGE_FRACTION
