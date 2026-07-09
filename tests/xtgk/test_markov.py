import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import State, zone_of
from tests.xtgk.conftest import three_band_cohort


def _fit():
    return MarkovPossessionValue().fit(three_band_cohort(), xg_column="xg", pressure_column="pressure")


def test_fit_returns_three_surfaces_of_grid_shape():
    m = _fit()
    for p in (1, 2, 3):
        assert m.surface(p).shape == (12, 16)


def test_all_three_levels_populated():
    m = _fit()
    for p in (1, 2, 3):
        assert m.support(p).sum() > 0


def test_value_before_fit_raises():
    with pytest.raises(NotFittedError):
        MarkovPossessionValue().value(0, 1)


def test_deep_value_nonzero_and_pressure_ordered():
    m = _fit()
    z = zone_of(3.0, 34.0)
    v1, v2, v3 = m.value(z, 1), m.value(z, 2), m.value(z, 3)
    assert v1 > 0.0  # deep build-up carries value via propagation
    assert v1 >= v2 >= v3  # xg decreases with pressure in the fixture
    assert v3 < v1  # a REAL gap (not because a stratum is empty)


def test_delta_v_shapley_identity():
    m = _fit()
    dv = m.delta_v(State(zone_of(3.0, 34.0), 1), State(zone_of(100.0, 34.0), 3))
    assert abs((dv.pressure_component + dv.position_component) - dv.delta) < 1e-12


def test_delta_v_on_unsupported_corner_is_finite():
    m = _fit()
    dv = m.delta_v(State(191, 1), State(0, 2))  # empty cells solve to 0.0, never NaN
    assert np.isfinite(dv.delta) and np.isfinite(dv.pressure_component)
