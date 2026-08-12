"""`n_flipped` must be ATTRIBUTABLE, not merely reported.

Spec 1.1: the migration has exactly two contributors -- the 1 cm band (20.15 -> 20.16) and the depth
boundary (`<` -> `<=`). A flip count that cannot be decomposed is a number that cannot be reasoned
about next cycle.

A consistency assertion (`n_flipped == sum of parts`) is NOT enough on its own: any partition of
`flipped` satisfies it, including one that labels every flip "boundary". Demonstrated. So the real
guard is PER-CASE attribution with hand-derived answers, plus the negative case.
"""

from __future__ import annotations

import numpy as np

from scripts.measure_box_constant_delta import classify_flips

_ZERO = {
    "n_flipped": 0,
    "n_flipped_band_only": 0,
    "n_flipped_boundary_only": 0,
    "n_flipped_both": 0,
}


def test_band_only():
    """y inside the 1 cm strip, x comfortably inside: in under 20.16, out under 20.15."""
    out = classify_flips(np.array([5.0]), np.array([34.0 + 20.155]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_band_only": 1}


def test_boundary_only():
    """Exactly on the depth line: `<` excludes, `<=` includes."""
    out = classify_flips(np.array([16.5]), np.array([34.0]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_boundary_only": 1}


def test_both_causes_is_its_own_bucket():
    """Both changes individually NECESSARY -- neither pure bucket may claim it, or the other
    becomes a systematic undercount."""
    out = classify_flips(np.array([16.5]), np.array([34.0 + 20.155]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_both": 1}


def test_unaffected_point_flips_nothing():
    assert classify_flips(np.array([5.0]), np.array([34.0])) == _ZERO


def test_y_in_strip_but_x_outside_does_not_flip():
    """Negative case: the band change is irrelevant when depth already excludes the point."""
    assert classify_flips(np.array([40.0]), np.array([34.0 + 20.155])) == _ZERO


def test_the_shipped_legacy_BAND_form_is_modelled_not_the_abs_form():
    """THE regression that matters.

    `y = 13.85` is the ONLY value separating the two legacy forms (spec 1.1 item 3): the shipped
    min/max band says OUTSIDE (13.85 sits fractionally below `(68-40.3)/2`), the abs form says
    INSIDE (`|13.85-34.0|` is exactly 20.15), and canonical says inside. So it IS a flip -- and a
    driver modelling legacy with the abs form reports 0, an UNDERCOUNT at the exact boundary this
    driver exists to measure.
    """
    out = classify_flips(np.array([5.0]), np.array([13.85]))
    assert out["n_flipped"] == 1, "the shipped band form was replaced by the abs form"
    assert out["n_flipped_band_only"] == 1


def test_the_buckets_partition_flipped_over_a_large_random_sample():
    """Consistency is necessary but NOT sufficient -- see the module docstring. Kept as a companion
    to the per-case tests above, never as a substitute."""
    rng = np.random.default_rng(0)
    gr_x = rng.uniform(-5.0, 25.0, 200_000)
    y = rng.uniform(0.0, 68.0, 200_000)
    out = classify_flips(gr_x, y)
    parts = out["n_flipped_band_only"] + out["n_flipped_boundary_only"] + out["n_flipped_both"]
    assert out["n_flipped"] == parts
    assert out["n_flipped"] > 0, "sample produced no flips; the attribution is untested here"
