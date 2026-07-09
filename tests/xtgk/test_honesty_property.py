import numpy as np
import pytest

from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import zone_of
from tests.xtgk.conftest import (
    flat_no_shot_cohort,
    mirror_x,
    mirror_y,
    offmidline_cohort,
    three_band_cohort,
)


def _fit(a):
    return MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure")


def test_surface_is_y_reflection_equivariant():
    # y->68-y preserves attack-LTR; a fit on y-mirrored data must be the row-reversed surface.
    # Uses an off-midline cohort so every y bins to its reflection partner cleanly (a y=34-on-
    # midline cohort rounds into cell 6, which the even-grid row-reversal cannot match). atol=1e-8
    # absorbs float non-associativity from the permuted value-iteration sums.
    a = offmidline_cohort()
    surf = _fit(a).surface(1)
    surf_m = _fit(mirror_y(a)).surface(1)
    assert np.allclose(surf_m, surf[::-1, :], atol=1e-8)


def test_attack_reversed_input_is_rejected_not_fit():
    with pytest.raises(ValueError, match="orientation"):
        _fit(mirror_x(three_band_cohort()))


def test_negative_control_flat_cohort_gives_flat_deep_value():
    m = _fit(flat_no_shot_cohort())
    z = zone_of(3.0, 34.0)
    assert m.value(z, 1) < 1e-6 and m.value(z, 3) < 1e-6


def test_honest_cohort_deep_gradient_positive():
    m = _fit(three_band_cohort())
    z = zone_of(3.0, 34.0)
    assert m.value(z, 1) > 0.0 and m.value(z, 1) > m.value(z, 3)
