"""Task 2 gates: DAS/xC profiles match the frozen library constants; grids + quadrature weights."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from silly_kicks.tracking._das_params import DAS_PARAMS, XC_PARAMS, simulation_grids
from tests.tracking._das_golden import load_golden


def test_das_profile_matches_reference_constants():
    d = load_golden().metadata["accessible_space_defaults"]
    assert DAS_PARAMS.b0 == d["interface._DEFAULT_B0_FOR_DAS"]
    assert DAS_PARAMS.b1 == d["interface._DEFAULT_B1_FOR_DAS"] == -2000.0
    assert DAS_PARAMS.n_angles == int(d["interface._DEFAULT_N_ANGLES_FOR_DAS"]) == 30
    assert DAS_PARAMS.n_v0 == int(d["interface._DEFAULT_N_V0_FOR_DAS"]) == 15
    assert DAS_PARAMS.v0_min == d["interface._DEFAULT_V0_MIN_FOR_DAS"]
    assert DAS_PARAMS.v0_max == d["interface._DEFAULT_V0_MAX_FOR_DAS"]
    assert DAS_PARAMS.radial_gridsize == d["interface._DEFAULT_RADIAL_GRIDSIZE_FOR_DAS"]
    assert DAS_PARAMS.pass_start_location_offset == d["interface._DEFAULT_PASS_START_LOCATION_OFFSET_FOR_DAS"]
    assert DAS_PARAMS.time_offset_ball == d["interface._DEFAULT_TIME_OFFSET_BALL_FOR_DAS"]
    assert DAS_PARAMS.player_velocity == d["interface._DEFAULT_PLAYER_VELOCITY_FOR_DAS"]
    assert DAS_PARAMS.inertial_seconds == d["interface._DEFAULT_INERTIAL_SECONDS_FOR_DAS"]
    assert DAS_PARAMS.tol_distance == d["interface._DEFAULT_TOL_DISTANCE_FOR_DAS"]
    assert DAS_PARAMS.use_max == bool(d["interface._DEFAULT_USE_MAX_FOR_DAS"]) is False
    assert DAS_PARAMS.v_max == d["interface._DEFAULT_V_MAX_FOR_DAS"]
    assert DAS_PARAMS.a_max == d["interface._DEFAULT_A_MAX_FOR_DAS"]
    assert DAS_PARAMS.factor == d["interface._DEFAULT_FACTOR_FOR_DAS"]
    assert DAS_PARAMS.factor2 == d["interface._DEFAULT_FACTOR2_FOR_DAS"]
    assert DAS_PARAMS.normalize == bool(d["interface._DEFAULT_NORMALIZE_FOR_DAS"]) is True
    assert DAS_PARAMS.danger_weight == d["interface._DEFAULT_DANGER_WEIGHT"]
    assert DAS_PARAMS.quadrature == "periodic"


def test_xc_profile_matches_reference_constants():
    d = load_golden().metadata["accessible_space_defaults"]
    assert XC_PARAMS.b0 == d["core._DEFAULT_B0"]
    assert XC_PARAMS.b1 == d["core._DEFAULT_B1"]
    assert XC_PARAMS.v0_min == d["interface._DEFAULT_V0_MIN_FOR_XC"]
    assert XC_PARAMS.v0_max == d["interface._DEFAULT_V0_MAX_FOR_XC"]
    assert XC_PARAMS.n_v0 == round(d["interface._DEFAULT_N_V0_FOR_XC"]) == 14
    assert XC_PARAMS.radial_gridsize == d["core._DEFAULT_RADIAL_GRIDSIZE"]
    assert XC_PARAMS.pass_start_location_offset == d["core._DEFAULT_PASS_START_LOCATION_OFFSET"]
    assert XC_PARAMS.time_offset_ball == d["core._DEFAULT_TIME_OFFSET_BALL"]
    assert XC_PARAMS.player_velocity == d["core._DEFAULT_PLAYER_VELOCITY"]
    assert XC_PARAMS.inertial_seconds == d["core._DEFAULT_INERTIAL_SECONDS"]
    assert XC_PARAMS.tol_distance == d["core._DEFAULT_TOL_DISTANCE"]
    assert XC_PARAMS.use_max == bool(d["core._DEFAULT_USE_MAX"]) is True
    assert XC_PARAMS.normalize is False
    assert XC_PARAMS.respect_offside is False
    assert XC_PARAMS.exclude_passer is True


@pytest.mark.parametrize(
    "field,bad",
    [
        ("n_angles", 0),
        ("n_v0", 0),
        ("v0_min", 0.0),
        ("v0_max", 1.0),
        ("radial_gridsize", -1.0),
        ("player_velocity", 0.0),
        ("danger_weight", 0.0),
        ("quadrature", "trapezoid"),
    ],
)
def test_invalid_params_raise(field, bad):
    with pytest.raises(ValueError):
        dataclasses.replace(DAS_PARAMS, **{field: bad})


def test_das_radial_grid_has_46_points():
    assert simulation_grids(DAS_PARAMS).d.shape == (46,)
    assert simulation_grids(DAS_PARAMS).phi.shape == (30,)
    assert simulation_grids(DAS_PARAMS).v0.shape == (15,)


def test_t_ball_and_derived_grid_shapes():
    g = simulation_grids(DAS_PARAMS)
    assert g.t_ball.shape == (15, 46)
    assert g.dt.shape == (15, 45)
    assert g.dt0.shape == (15,)
    assert g.rate_divisor.shape == (15,)
    assert g.dr.shape == (46,)
    assert g.d_area.shape == (30, 46)


def test_quadrature_weights_differ_only_at_the_two_end_rays():
    per = simulation_grids(DAS_PARAMS).d_area
    ref = simulation_grids(dataclasses.replace(DAS_PARAMS, quadrature="reference")).d_area
    diff = per != ref
    assert diff[0].all(), "ray 0 must differ"
    assert diff[-1].all(), "ray n-1 must differ"
    assert not diff[1:-1].any(), "interior rays must be bit-identical across quadratures"
    # periodic restores the missing half wedges -> strictly larger at the two end rays.
    assert np.all(per[0] >= ref[0]) and np.all(per[-1] >= ref[-1])
    assert per[0].sum() > ref[0].sum() and per[-1].sum() > ref[-1].sum()


def test_grids_are_cached():
    assert simulation_grids(DAS_PARAMS) is simulation_grids(DAS_PARAMS)


def test_dataclass_is_frozen_and_hashable():
    assert hash(DAS_PARAMS) == hash(DAS_PARAMS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DAS_PARAMS.b1 = 1.0  # type: ignore[misc]
