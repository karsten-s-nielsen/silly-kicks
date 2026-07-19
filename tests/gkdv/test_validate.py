"""Registered validation constants + Layer 4 behavioural anchoring (spec 6.1-6.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.gkdv._validate import (
    EXPECTED_DIRECTION,
    ICC_ANCHORS,
    TERCILE_SEPARATION_M,
    behavioural_anchoring_verdict,
)


def test_registered_constants_are_frozen_values():
    assert ICC_ANCHORS == (0.015, 0.020, 0.026)
    assert TERCILE_SEPARATION_M == 0.5


def test_package_reexports_the_validation_surface():
    """The gate constants and the Layer-4 verdict are PACKAGE-public.

    House precedent is ``xtgk``, which re-exports its private ``_diagnostics`` gate
    surface (``GateConfig`` / ``run_deep_zone_gate`` / ``DeepZoneGateReport``) through
    ``__init__`` while its own tests still import the private path. An owner-run
    validation harness consumes these from the package, so dropping them from
    ``__all__`` would break it while every test in this file kept passing --
    which is exactly what this test exists to prevent.
    """
    import silly_kicks.gkdv as gkdv

    expected = {
        "EXPECTED_DIRECTION",
        "ICC_ANCHORS",
        "TERCILE_SEPARATION_M",
        "behavioural_anchoring_verdict",
    }
    assert expected <= set(gkdv.__all__)
    # Re-exported objects must be the SAME objects, not shadowing copies.
    assert gkdv.ICC_ANCHORS is ICC_ANCHORS
    assert gkdv.TERCILE_SEPARATION_M is TERCILE_SEPARATION_M
    assert gkdv.EXPECTED_DIRECTION is EXPECTED_DIRECTION
    assert gkdv.behavioural_anchoring_verdict is behavioural_anchoring_verdict


def test_expected_direction_is_negative_for_both_arms():
    """Both arms are ATTACKER-VALUE units, so a deterrent keeper reads NEGATIVE (spec 5).
    A flipped entry here would invert the sign panel's verdict silently."""
    assert EXPECTED_DIRECTION == {"delta_das": "negative", "delta_threat": "negative"}


def test_anchored_arm_passes():
    """Top and bottom terciles differ in mean signed depth by more than the threshold."""
    df = pd.DataFrame(
        {
            "player_id": range(9),
            "value": np.linspace(-0.05, 0.05, 9),
            "signed_dx": np.linspace(-3.0, 3.0, 9),
        }
    )
    assert behavioural_anchoring_verdict(df, value_col="value", depth_col="signed_dx") == "anchored"


def test_unanchored_arm_is_uninterpretable_not_evidence():
    """The PEV lesson: an arm not tracking a behaviour keepers vary is NOT evidence."""
    df = pd.DataFrame(
        {
            "player_id": range(9),
            "value": np.linspace(-0.05, 0.05, 9),
            "signed_dx": np.zeros(9),
        }
    )
    assert behavioural_anchoring_verdict(df, value_col="value", depth_col="signed_dx") == "uninterpretable"


def test_separation_just_below_the_threshold_is_uninterpretable():
    """The threshold is a real boundary, not decoration: a 0.4 m separation must FAIL and a
    0.6 m separation must PASS on otherwise identical inputs."""
    base = {"player_id": range(9), "value": np.linspace(-0.05, 0.05, 9)}
    below = np.repeat([-0.2, 0.0, 0.2], 3)  # outer terciles differ by 0.4 m
    above = np.repeat([-0.3, 0.0, 0.3], 3)  # ...by 0.6 m
    verdict_below = behavioural_anchoring_verdict(pd.DataFrame({**base, "d": below}), value_col="value", depth_col="d")
    verdict_above = behavioural_anchoring_verdict(pd.DataFrame({**base, "d": above}), value_col="value", depth_col="d")
    assert verdict_below == "uninterpretable"
    assert verdict_above == "anchored"


def test_verdict_is_direction_agnostic():
    """The separation is an absolute magnitude: reversing which tercile is deeper must not
    change the verdict (Layer 4 asks 'do keepers VARY', not 'in which direction')."""
    base = {"player_id": range(9), "value": np.linspace(-0.05, 0.05, 9)}
    fwd = behavioural_anchoring_verdict(
        pd.DataFrame({**base, "d": np.linspace(-3.0, 3.0, 9)}), value_col="value", depth_col="d"
    )
    rev = behavioural_anchoring_verdict(
        pd.DataFrame({**base, "d": np.linspace(3.0, -3.0, 9)}), value_col="value", depth_col="d"
    )
    assert fwd == rev == "anchored"


def test_nan_value_rows_do_not_contaminate_the_outer_terciles():
    """NaN sorts LAST, so an unfiltered NaN-VALUE row is ranked as a top-tercile keeper and
    its real depth is averaged into ``hi`` -- while also widening ``k``.

    NOT the naive "the mean becomes NaN" story: pandas' ``mean`` skips NaN, so the
    corruption is silent rather than loud. This fixture is calibrated so the distortion
    actually FLIPS the verdict (9 real keepers separate by 4.50 m; leaving the three
    NaN-value rows in collapses that to 0.375 m, below the 0.5 m threshold), which is what
    makes the assertion load-bearing rather than decorative.
    """
    df = pd.DataFrame(
        {
            "player_id": range(12),
            "value": [*np.linspace(-0.05, 0.05, 9), np.nan, np.nan, np.nan],
            "signed_dx": [*np.linspace(-3.0, 3.0, 9), -3.0, -3.0, -3.0],
        }
    )
    assert behavioural_anchoring_verdict(df, value_col="value", depth_col="signed_dx") == "anchored"


def test_empty_input_is_uninterpretable_not_a_crash():
    df = pd.DataFrame({"player_id": [], "value": [], "signed_dx": []})
    assert behavioural_anchoring_verdict(df, value_col="value", depth_col="signed_dx") == "uninterpretable"


def test_input_is_not_mutated():
    df = pd.DataFrame(
        {
            "player_id": range(9),
            "value": np.linspace(0.05, -0.05, 9),  # deliberately unsorted by value
            "signed_dx": np.linspace(3.0, -3.0, 9),
        }
    )
    before = df.copy(deep=True)
    behavioural_anchoring_verdict(df, value_col="value", depth_col="signed_dx")
    pd.testing.assert_frame_equal(df, before)


def test_band_is_correctly_SIZED_not_just_detecting():
    """BOTH SIDES: a no-effect fixture must land INSIDE the band AND a planted effect
    outside it.

    Only asserting the planted-effect side lets an anti-conservative instrument ship (it
    would flag structure that is not there); only asserting the no-effect side is satisfied
    by a function that always returns zero. Both directions, or the test proves nothing.
    """
    from silly_kicks._group_metrics import icc_one_way

    rng = np.random.default_rng(42)
    groups = rng.integers(0, 8, 400)

    null = icc_one_way(rng.normal(size=400), groups)
    assert abs(null) < 0.05, f"no-effect fixture produced ICC={null:.4f} -- band is anti-conservative"

    # Same groups, same within-keeper noise scale, PLUS a real between-keeper offset.
    planted = icc_one_way(rng.normal(size=400) + groups * 1.5, groups)
    assert planted > 0.3, f"planted keeper effect produced ICC={planted:.4f} -- instrument is dead"
