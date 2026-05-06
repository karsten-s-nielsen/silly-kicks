"""Tests for pitch control parameter dataclasses and validation."""

from __future__ import annotations

import pytest

from silly_kicks.tracking.pitch_control._params import (
    FernandezBornnParams,
    SpearmanParams,
    VoronoiParams,
    validate_params_for_method,
)


class TestSpearmanParams:
    def test_defaults(self):
        p = SpearmanParams()
        assert p.reaction_time == 0.7
        assert p.max_acceleration == 7.0
        assert p.sigma == 0.45
        assert p.lambda_gk == 3.0
        assert p.average_ball_speed == 15.0
        assert p.grid_cells_x == 50
        assert p.grid_cells_y == 32

    def test_frozen(self):
        p = SpearmanParams()
        with pytest.raises(AttributeError):
            p.sigma = 1.0  # type: ignore[misc]


class TestFernandezBornnParams:
    def test_defaults(self):
        p = FernandezBornnParams()
        assert p.max_speed == 13.0
        assert p.min_radius == 4.0
        assert p.max_radius == 10.0
        assert p.grid_cells_x == 50
        assert p.grid_cells_y == 32

    def test_frozen(self):
        p = FernandezBornnParams()
        with pytest.raises(AttributeError):
            p.max_speed = 20.0  # type: ignore[misc]


class TestVoronoiParams:
    def test_defaults(self):
        p = VoronoiParams()
        assert p.grid_cells_x == 50
        assert p.grid_cells_y == 32


class TestValidateParamsForMethod:
    def test_none_params_accepted(self):
        validate_params_for_method("spearman", None)
        validate_params_for_method("fernandez_bornn", None)
        validate_params_for_method("voronoi", None)

    def test_correct_type_accepted(self):
        validate_params_for_method("spearman", SpearmanParams())
        validate_params_for_method("fernandez_bornn", FernandezBornnParams())
        validate_params_for_method("voronoi", VoronoiParams())

    def test_wrong_type_raises_typeerror(self):
        with pytest.raises(TypeError, match=r"spearman.*expects SpearmanParams"):
            validate_params_for_method("spearman", FernandezBornnParams())

    def test_unknown_method_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown method"):
            validate_params_for_method("bogus", None)  # type: ignore[arg-type]
