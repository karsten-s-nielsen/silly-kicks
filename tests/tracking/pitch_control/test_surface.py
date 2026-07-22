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


def _make_decomposed(player_ids: np.ndarray) -> PitchControlSurface:
    """A 2-team decomposed surface with caller-chosen ``player_ids`` dtype (team layout fixed:
    players 0,1 on team 1 with equal influence; player 2 alone on team 2)."""
    nx, ny = 10, 7
    return PitchControlSurface(
        grid_x=np.linspace(0, 105, nx),
        grid_y=np.linspace(0, 68, ny),
        surface=np.full((ny, nx), 0.6),
        method="spearman",
        attacking_team_id=1,
        per_player_influence=np.full((3, ny, nx), 0.2),
        player_ids=player_ids,
        player_team_ids=np.array([1, 1, 2]),
    )


class TestPlayerIdDtypeInvariance:
    """ADR-019: ``player_share`` / ``player_surface`` resolve a caller-supplied id scalar
    dtype-invariantly. RED before the fix -- a raw ``player_ids == player_id`` matches nothing
    across dtypes, so both methods RAISED 'not found' on a value-equal id of a different dtype
    (the recorded ``_surface.py:140,167`` gap)."""

    def test_player_share_int_ids_queried_with_str(self):
        s = _make_decomposed(np.array([1, 2, 3]))  # int64 ids
        assert s.player_share("1") == pytest.approx(0.5)  # str "1" -> player 1 (team 1, half)
        assert s.player_share("3") == pytest.approx(1.0)  # player 3 alone on team 2

    def test_player_share_str_ids_queried_with_int(self):
        s = _make_decomposed(np.array(["1", "2", "3"], dtype=object))  # object string ids
        assert s.player_share(1) == pytest.approx(0.5)
        assert s.player_share(3) == pytest.approx(1.0)

    def test_player_surface_is_dtype_invariant(self):
        s_int = _make_decomposed(np.array([1, 2, 3]))
        np.testing.assert_allclose(s_int.player_surface("1"), s_int.player_surface(1))
        s_str = _make_decomposed(np.array(["1", "2", "3"], dtype=object))
        np.testing.assert_allclose(s_str.player_surface(1), s_str.player_surface("1"))

    def test_a_genuinely_absent_id_still_raises_not_found(self):
        # Discriminating power: the fix must resolve a value-equal id, NOT match everything.
        s = _make_decomposed(np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="not found"):
            s.player_share("999")
        with pytest.raises(ValueError, match="not found"):
            s.player_surface(999)


class TestToXarray:
    def test_raises_without_xarray(self):
        """If xarray not installed, should raise ImportError with message."""
        # This test may pass or skip depending on env
        s = _make_surface()
        try:
            import xarray  # noqa: F401  # type: ignore[import-not-found]

            # xarray is installed -- test it works
            da = s.to_xarray()
            assert da.dims == ("y", "x")  # type: ignore[attr-defined]
        except ImportError:
            with pytest.raises(ImportError, match="xarray"):
                s.to_xarray()
