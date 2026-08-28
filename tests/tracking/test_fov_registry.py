"""Byte-identical freeze of the ADR-062 ``add_action_context`` visibility companions (ADR-077).

Task 2 retires the hand-coded ``_append_visibility_companions`` into the neutral
``_fov_registry`` engine. This gate FREEZES the companion output first: it is captured from the
CURRENT implementation, confirmed passing before the refactor, and re-run after -- so any drift
in value, source token, NaN policy, OR column ORDER fails.

Strengthened over the plan's value-only sketch:

* the exact ORDERED list of the six companion column names is asserted (a column-order
  regression -- the byte-identical subtlety the retirement must preserve -- is caught);
* every ``_observed_source`` column and the NaN mask of every ``_observed_fraction`` column are
  asserted EXACTLY (strings + NaN policy are fully deterministic);
* the ``_observed_fraction`` VALUES are asserted to ``atol=1e-9`` -- the repo's own convention for
  these clip/trig-derived fractions (``pytest.approx(abs=1e-9)`` / ``assert_allclose(atol=1e-12)``
  throughout ``tests/tracking/test_visibility.py``), which absorbs libm ULP across the
  Windows/Linux CI legs while a real region/order regression (which moves a fraction by far more
  than 1e-9, or flips a source/NaN) is still caught exactly by the other three assertions.
"""

from __future__ import annotations

import numpy as np

from silly_kicks.tracking import add_action_context
from tests.tracking._fov_fixtures import tiny_actions, tiny_frames, tiny_visible_area

#: The three ADR-062 count features whose region-of-interest is companioned, in EMISSION order.
_COMPANIONED = (
    "nearest_defender_distance",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
)

#: Exact ordered companion-column names as they must appear in ``out.columns``: for each feature,
#: ``_observed_fraction`` immediately followed by ``_observed_source``, features in the order above.
_EXPECTED_COMPANION_ORDER = [f"{name}_observed_{suffix}" for name in _COMPANIONED for suffix in ("fraction", "source")]

#: Golden captured from the pre-refactor (ADR-062 hand-coded helper) implementation. Frozen before
#: touching ``features.py`` -- the fixture-generator discipline.
_GOLDEN_FRACTION = {
    "nearest_defender_distance": [1.0, 0.9999999999999906, 1.0, np.nan, 0.9999999999998743, 0.5000000000000071],
    "receiver_zone_density": [0.9999999999999059, 0.7497441211548401, 0.0, np.nan, 0.9999999999999059, 1.0],
    "defenders_in_triangle_to_goal": [
        0.8185941043083883,
        0.4982698961937732,
        0.25000000000001243,
        np.nan,
        0.7901234567901181,
        0.8520710059171598,
    ],
}
_GOLDEN_SOURCE = {
    "nearest_defender_distance": ["observed", "observed", "observed", "no_polygon", "observed", "observed"],
    "receiver_zone_density": ["observed", "observed", "observed", "no_polygon", "observed", "observed"],
    "defenders_in_triangle_to_goal": ["observed", "observed", "observed", "no_polygon", "observed", "observed"],
}


def test_adr062_companions_byte_identical_after_refactor():
    a, f, va = tiny_actions(), tiny_frames(), tiny_visible_area()
    out = add_action_context(a, f, visible_area=va)

    # (0) The fixture is a REAL crop, not all-no_polygon/all-unlinked: at least one companion
    #     source is 'observed' with a fraction strictly inside (0, 1). A parity gate over a
    #     degenerate fixture would freeze nothing.
    strict_observed = [
        v
        for name in _COMPANIONED
        for v in out[f"{name}_observed_fraction"].to_numpy()
        if isinstance(v, float) and 0.0 < v < 1.0
    ]
    assert strict_observed, "fixture produced no partial-observation fraction; the freeze is vacuous"

    # (1) Column order (byte-identical): the six companions appear in exactly this order.
    observed_order = [c for c in out.columns if c.endswith(("_observed_fraction", "_observed_source"))]
    assert observed_order == _EXPECTED_COMPANION_ORDER

    for name in _COMPANIONED:
        frac_col = f"{name}_observed_fraction"
        src_col = f"{name}_observed_source"
        assert frac_col in out.columns
        assert src_col in out.columns

        golden_frac = np.array(_GOLDEN_FRACTION[name], dtype=float)
        actual_frac = out[frac_col].to_numpy(dtype=float)

        # (2) NaN mask is EXACT (the degenerate/unlinked/no-polygon NaN policy is deterministic).
        np.testing.assert_array_equal(np.isnan(actual_frac), np.isnan(golden_frac))
        # (3) Fraction values match to 1e-9 (repo convention; libm-ULP-portable).
        np.testing.assert_allclose(actual_frac, golden_frac, rtol=0.0, atol=1e-9, equal_nan=True)
        # (4) Source token per row is EXACT (byte-identical strings).
        assert out[src_col].tolist() == _GOLDEN_SOURCE[name]
