"""TF-19 sign-off package: the constants registered BLIND, before the corpus run (spec §5.4, §6)."""

from __future__ import annotations

from silly_kicks.gkdv._validate import (
    ATT_RELATIVE_ANCHORS,
    ICC_ANCHORS,
    LAYER3_HEADROOM_RANGE_FRACTION,
    N_MIN_MATCHED,
)


def test_att_anchors_are_registered_as_a_range_mirroring_the_icc_anchors():
    assert ATT_RELATIVE_ANCHORS == (0.10, 0.15, 0.20)
    assert len(ATT_RELATIVE_ANCHORS) == len(ICC_ANCHORS)  # both report a curve, not a point


def test_layer3_fraction_is_committed_before_the_measurement():
    """The ordering is load-bearing: measuring `openGoal`'s range first and choosing the fraction
    afterwards would make the threshold tunable to any desired Layer 3 outcome."""
    assert LAYER3_HEADROOM_RANGE_FRACTION == 0.02


def test_att_anchors_are_relative_not_absolute():
    """Row 5 gates a spell-level ATT, not a keeper-level variance share -- the ICC anchor does not
    transfer. Relative per spec §1.3's own lesson on small-probability quantities."""
    assert all(0.0 < a < 1.0 for a in ATT_RELATIVE_ANCHORS)
    assert ATT_RELATIVE_ANCHORS != ICC_ANCHORS


def test_n_min_is_an_unfilled_VALUE_not_an_unregistered_RULE():
    """`N_MIN_MATCHED` is None until the locked-corpus run fills it. The RULE (which anchor, which
    outcome, how derived) is registered now; only the number is pending."""
    assert N_MIN_MATCHED is None or isinstance(N_MIN_MATCHED, int)
