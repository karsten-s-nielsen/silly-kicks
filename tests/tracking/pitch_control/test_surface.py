"""Tests for PitchControlSurface dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking.pitch_control._surface import PitchControlSurface


def _make_surface(nx=10, ny=7, value=0.6, decompose=False):
    """Helper to build a test surface."""
    grid_x = np.linspace(0, 105, nx)
    grid_y = np.linspace(0, 68, ny)
    surface = np.full((ny, nx), value)
    per_player = None
    player_ids = None
    player_team_ids = None
    if decompose:
        per_player = np.full((3, ny, nx), value / 3)
        player_ids = np.array([1, 2, 3])
        player_team_ids = np.array([1, 1, 2])  # players 1,2 on team 1; player 3 on team 2
    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=surface,
        method="spearman",
        attacking_team_id=1,
        per_player_influence=per_player,
        player_ids=player_ids,
        player_team_ids=player_team_ids,
    )


class TestImmutability:
    def test_frozen_attribute(self):
        s = _make_surface()
        with pytest.raises(AttributeError):
            s.method = "voronoi"  # type: ignore[misc]

    def test_array_not_writeable(self):
        s = _make_surface()
        with pytest.raises(ValueError):
            s.surface[0, 0] = 999.0

    def test_grid_not_writeable(self):
        s = _make_surface()
        with pytest.raises(ValueError):
            s.grid_x[0] = -1.0


class TestCellArea:
    def test_cell_area_correct(self):
        s = _make_surface(nx=10, ny=7)
        dx = 105.0 / 9  # linspace(0, 105, 10) has 9 gaps
        dy = 68.0 / 6
        assert abs(s.cell_area - dx * dy) < 1e-10


class TestAtPoint:
    def test_uniform_surface(self):
        s = _make_surface(value=0.7)
        assert abs(s.at_point(50.0, 34.0) - 0.7) < 1e-10

    def test_at_points_batch(self):
        s = _make_surface(value=0.7)
        pts = np.array([[50.0, 34.0], [10.0, 10.0]])
        result = s.at_points(pts)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, 0.7, atol=1e-10)

    def test_edge_clamp(self):
        s = _make_surface(value=0.5)
        # Point at grid boundary should not crash
        val = s.at_point(0.0, 0.0)
        assert 0.0 <= val <= 1.0


class TestControlInRegion:
    def test_uniform_surface(self):
        s = _make_surface(value=0.8)
        val = s.control_in_region(0, 105, 0, 68)
        assert abs(val - 0.8) < 1e-10

    def test_half_pitch(self):
        s = _make_surface(value=0.8)
        val = s.control_in_region(52.5, 105, 0, 68)
        assert abs(val - 0.8) < 1e-10


class TestPlayerShare:
    def test_raises_without_decomposition(self):
        s = _make_surface(decompose=False)
        with pytest.raises(ValueError, match="decompose"):
            s.player_share(1)

    def test_equal_shares_within_team(self):
        s = _make_surface(decompose=True)
        # Players 1 & 2 on team 1 with equal influence -> 50% each of team total
        assert abs(s.player_share(1) - 0.5) < 1e-10
        assert abs(s.player_share(2) - 0.5) < 1e-10
        # Player 3 alone on team 2 -> 100% of team total
        assert abs(s.player_share(3) - 1.0) < 1e-10

    def test_unknown_player_raises(self):
        s = _make_surface(decompose=True)
        with pytest.raises(ValueError, match="not found"):
            s.player_share(999)


class TestPlayerSurface:
    def test_returns_correct_shape(self):
        s = _make_surface(nx=10, ny=7, decompose=True)
        ps = s.player_surface(1)
        assert ps.shape == (7, 10)

    def test_raises_without_decomposition(self):
        s = _make_surface(decompose=False)
        with pytest.raises(ValueError, match="decompose"):
            s.player_surface(1)


class TestToXarray:
    def test_raises_without_xarray(self):
        """If xarray not installed, should raise ImportError with message."""
        # This test may pass or skip depending on env
        s = _make_surface()
        try:
            import xarray  # noqa: F401

            # xarray is installed -- test it works
            da = s.to_xarray()
            assert da.dims == ("y", "x")
        except ImportError:
            with pytest.raises(ImportError, match="xarray"):
                s.to_xarray()
