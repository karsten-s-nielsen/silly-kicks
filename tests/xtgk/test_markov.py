import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from silly_kicks.xtgk import PressureLevels
from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import State, zone_of
from silly_kicks.xthreat._grid import M, N, _get_flat_indexes
from tests.xtgk.conftest import three_band_cohort


def _fit():
    return MarkovPossessionValue().fit(three_band_cohort(), xg_column="xg", pressure_column="pressure")


def test_markov_fits_under_zone_conditional_and_roundtrips(tmp_path):
    actions = three_band_cohort()
    pl = PressureLevels(mode="zone_conditional")
    zones = _get_flat_indexes(actions.start_x, actions.start_y, N, M).to_numpy()
    pl.fit(actions["pressure"], zones=zones)
    mk = MarkovPossessionValue().fit(actions, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    v_lo = mk.value(0, 1)  # deep cell, low tercile
    assert np.isfinite(v_lo)

    mk.save(tmp_path / "surf")
    reloaded = MarkovPossessionValue.load(tmp_path / "surf")
    assert reloaded.pressure_levels is not None
    assert reloaded.pressure_levels.mode == "zone_conditional"
    assert np.isclose(reloaded.value(0, 1), v_lo)


def test_markov_global_metadata_byte_identical(tmp_path):
    actions = three_band_cohort()
    mk = MarkovPossessionValue().fit(actions, xg_column="xg", pressure_column="pressure")
    mk.save(tmp_path / "surf")
    meta = (tmp_path / "surf" / "metadata.json").read_text()
    assert "pressure_mode" not in meta  # global form must NOT gain the zone-conditional key
    assert '"cutpoints"' in meta


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


def test_fit_merges_reward_provenance():
    # Q3 (ADR-036 §6): the owner-run computes an OOD-rate / xg-CI summary from fct_shot_xg and
    # passes it as reward_provenance; fit records it (silly-kicks never hard-codes ood_flag semantics).
    prov = {"ood_rate": 1.0, "xg_ci_mean_width": 0.12, "xg_source": "fct_shot_xg.xg"}
    m = MarkovPossessionValue().fit(
        three_band_cohort(), xg_column="xg", pressure_column="pressure", reward_provenance=prov
    )
    assert m.provenance["reward_provenance"] == prov


def test_fit_without_reward_provenance_omits_key():
    m = _fit()
    assert "reward_provenance" not in m.provenance
