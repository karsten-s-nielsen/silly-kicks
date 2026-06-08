"""Tests for silly_kicks.tracking._obso — OBSO surface computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._obso import (
    ObsoParams,
    _get_default_grids,
    _interpolate_grid,
    _make_synthetic_epv_grid,
    _make_synthetic_reachability_grid,
    compute_obso_surface,
    compute_pass_obso,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracking_frame(
    n_per_team: int = 5,
    team_ids: tuple[int, int] = (1, 2),
    include_ball: bool = True,
) -> pd.DataFrame:
    """Build a minimal tracking frame for pitch control tests."""
    rows = []
    rng = np.random.RandomState(42)
    for tid_idx, tid in enumerate(team_ids):
        for j in range(n_per_team):
            x = 20.0 + tid_idx * 60 + rng.uniform(-5, 5)
            y = 10.0 + j * 12 + rng.uniform(-2, 2)
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": 0,
                    "time_seconds": 0.0,
                    "player_id": tid * 100 + j,
                    "team_id": tid,
                    "x": x,
                    "y": y,
                    "vx": 0.5 * (1 - 2 * tid_idx),
                    "vy": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": j == 0,
                }
            )
    if include_ball:
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 0,
                "time_seconds": 0.0,
                "player_id": None,
                "team_id": None,
                "x": 52.5,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "is_goalkeeper": False,
            }
        )
    return pd.DataFrame(rows)


def _make_simple_pc_surface() -> object:
    """Create a mock PitchControlSurface-like object."""
    grid_x = np.linspace(0, 105, 50)
    grid_y = np.linspace(0, 68, 32)
    surface = np.random.RandomState(42).uniform(0.3, 0.7, (32, 50))

    class MockSurface:
        pass

    s = MockSurface()
    s.surface = surface
    s.grid_x = grid_x
    s.grid_y = grid_y
    return s


# ---------------------------------------------------------------------------
# TestInterpolateGrid
# ---------------------------------------------------------------------------


class TestInterpolateGrid:
    def test_identity(self):
        """Same-shape input returns a copy."""
        grid = np.arange(12, dtype=float).reshape(3, 4)
        result = _interpolate_grid(grid, (3, 4))
        np.testing.assert_array_equal(result, grid)
        assert result is not grid

    def test_upsample_shape(self):
        """Output matches target shape on upsample."""
        grid = np.ones((5, 5))
        result = _interpolate_grid(grid, (10, 10))
        assert result.shape == (10, 10)

    def test_downsample_shape(self):
        """Output matches target shape on downsample."""
        grid = np.ones((20, 30))
        result = _interpolate_grid(grid, (5, 8))
        assert result.shape == (5, 8)

    def test_corners_preserved(self):
        """Corner values are preserved after interpolation."""
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _interpolate_grid(grid, (10, 10))
        np.testing.assert_almost_equal(result[0, 0], 1.0)
        np.testing.assert_almost_equal(result[0, -1], 2.0)
        np.testing.assert_almost_equal(result[-1, 0], 3.0)
        np.testing.assert_almost_equal(result[-1, -1], 4.0)

    def test_constant_grid(self):
        """Uniform grid stays uniform after interpolation."""
        grid = np.full((4, 6), 0.5)
        result = _interpolate_grid(grid, (8, 12))
        np.testing.assert_array_almost_equal(result, 0.5)


# ---------------------------------------------------------------------------
# TestDefaultGrids
# ---------------------------------------------------------------------------


class TestGetDefaultGrids:
    def test_synthetic_generation(self):
        """Both grids are generated when no input provided."""
        reach, epv = _get_default_grids()
        assert reach.ndim == 2
        assert epv.ndim == 2
        assert reach.shape == (100, 64)
        assert epv.shape == (50, 32)

    def test_passthrough(self):
        """Pre-loaded arrays are returned directly."""
        r = np.ones((10, 10))
        e = np.ones((5, 5))
        reach, epv = _get_default_grids(r, e)
        assert reach is r
        assert epv is e

    def test_mixed(self):
        """One pre-loaded, one synthetic."""
        r = np.ones((10, 10))
        reach, epv = _get_default_grids(r, None)
        assert reach is r
        assert epv.shape == (50, 32)


class TestSyntheticGrids:
    def test_reachability_shape(self):
        grid = _make_synthetic_reachability_grid(50, 30)
        assert grid.shape == (50, 30)
        assert grid.min() >= 0
        assert grid.max() <= 1

    def test_epv_shape(self):
        grid = _make_synthetic_epv_grid(25, 16)
        assert grid.shape == (25, 16)
        assert grid.min() >= 0


# ---------------------------------------------------------------------------
# TestComputeObsoSurface
# ---------------------------------------------------------------------------


class TestComputeObsoSurface:
    def test_shape_matches_ppcf(self):
        """OBSO surface shape matches input PPCF grid."""
        pc = _make_simple_pc_surface()
        obso = compute_obso_surface(pc, (52.5, 34.0))
        assert obso.values.shape == pc.surface.shape

    def test_values_bounded(self):
        """All values in [0, 1]."""
        pc = _make_simple_pc_surface()
        obso = compute_obso_surface(pc, (52.5, 34.0))
        assert obso.values.min() >= 0.0
        assert obso.values.max() <= 1.0

    def test_zero_ppcf_gives_zero_obso(self):
        """Zero pitch control produces zero OBSO."""
        pc = _make_simple_pc_surface()
        pc.surface = np.zeros_like(pc.surface)
        obso = compute_obso_surface(pc, (52.5, 34.0))
        np.testing.assert_array_equal(obso.values, 0.0)

    def test_grid_axes_returned(self):
        """ObsoSurface includes grid coordinates."""
        pc = _make_simple_pc_surface()
        obso = compute_obso_surface(pc, (52.5, 34.0))
        assert len(obso.grid_x) == pc.surface.shape[1]
        assert len(obso.grid_y) == pc.surface.shape[0]

    def test_custom_params(self):
        """Custom ObsoParams are respected."""
        pc = _make_simple_pc_surface()
        params = ObsoParams(sigma_x=10.0, sigma_y=5.0)
        obso = compute_obso_surface(pc, (52.5, 34.0), params=params)
        # Tighter sigma = more concentrated near ball, faster decay
        assert obso.values.min() >= 0.0

    def test_near_ball_gradient(self):
        """OBSO should be higher near the ball than far from it (given uniform PPCF)."""
        grid_x = np.linspace(0, 105, 50)
        grid_y = np.linspace(0, 68, 32)

        class UniformPC:
            pass

        pc = UniformPC()
        pc.surface = np.full((32, 50), 0.5)
        pc.grid_x = grid_x
        pc.grid_y = grid_y

        ball_pos = (52.5, 34.0)
        obso = compute_obso_surface(pc, ball_pos)

        # Near-ball cell
        cx = np.argmin(np.abs(grid_x - ball_pos[0]))
        cy = np.argmin(np.abs(grid_y - ball_pos[1]))
        near_val = obso.values[cy, cx]

        # Far corner
        far_val = obso.values[0, 0]

        assert near_val > far_val

    def test_custom_transition_and_epv(self):
        """Custom grids are used when provided."""
        pc = _make_simple_pc_surface()
        transition = np.ones((20, 20))
        epv = np.full((15, 15), 0.5)
        obso = compute_obso_surface(pc, (52.5, 34.0), transition_grid=transition, epv_grid=epv)
        assert obso.values.shape == pc.surface.shape


# ---------------------------------------------------------------------------
# TestObsoParams
# ---------------------------------------------------------------------------


class TestObsoParams:
    def test_frozen(self):
        params = ObsoParams()
        with pytest.raises(AttributeError):
            params.sigma_x = 99.0  # type: ignore[misc]

    def test_defaults(self):
        params = ObsoParams()
        assert params.grid_nx == 104
        assert params.grid_ny == 68
        assert params.pitch_length == 105.0
        assert params.pitch_width == 68.0
        assert params.sigma_x == 26.25
        assert params.sigma_y == 17.0


# ---------------------------------------------------------------------------
# TestComputePassObso
# ---------------------------------------------------------------------------


class TestComputePassObso:
    @pytest.fixture()
    def window_frames(self) -> list[pd.DataFrame]:
        """3-frame window around a pass event."""
        frames = []
        for fid in range(3):
            frame = _make_tracking_frame()
            frame["frame_id"] = fid
            frame["time_seconds"] = float(fid) * 0.04
            frames.append(frame)
        return frames

    def test_schema(self, window_frames):
        """Output dict has the expected keys."""
        result = compute_pass_obso(
            window_frames,
            event_frame_idx=1,
            target_position=(80.0, 30.0),
            attacking_team_id=1,
        )
        assert set(result.keys()) == {"actual_obso", "peak_obso", "optimal_obso"}

    def test_values_bounded(self, window_frames):
        """All values in [0, 1]."""
        result = compute_pass_obso(
            window_frames,
            event_frame_idx=1,
            target_position=(80.0, 30.0),
            attacking_team_id=1,
        )
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_peak_geq_actual(self, window_frames):
        """Peak OBSO >= actual OBSO by definition."""
        result = compute_pass_obso(
            window_frames,
            event_frame_idx=1,
            target_position=(80.0, 30.0),
            attacking_team_id=1,
        )
        assert result["peak_obso"] >= result["actual_obso"] - 1e-10

    def test_optimal_geq_actual(self, window_frames):
        """Optimal OBSO >= actual OBSO by definition."""
        result = compute_pass_obso(
            window_frames,
            event_frame_idx=1,
            target_position=(80.0, 30.0),
            attacking_team_id=1,
        )
        assert result["optimal_obso"] >= result["actual_obso"] - 1e-10

    def test_empty_frames(self):
        """Empty frame list returns all NaN."""
        result = compute_pass_obso(
            [],
            event_frame_idx=0,
            target_position=(50.0, 34.0),
            attacking_team_id=1,
        )
        assert np.isnan(result["actual_obso"])
        assert np.isnan(result["peak_obso"])
        assert np.isnan(result["optimal_obso"])

    def test_out_of_range_idx(self):
        """Out-of-range event index returns all NaN."""
        frame = _make_tracking_frame()
        result = compute_pass_obso(
            [frame],
            event_frame_idx=5,
            target_position=(50.0, 34.0),
            attacking_team_id=1,
        )
        assert np.isnan(result["actual_obso"])


# ---------------------------------------------------------------------------
# TestAddObso + TestObsoXfns (wiring tests — after aggregator is added)
# ---------------------------------------------------------------------------


def _make_pass_actions_and_frames():
    """Build minimal pass actions + multi-frame tracking data for OBSO aggregator."""
    actions = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "action_id": [0, 1, 2],
            "period_id": [1, 1, 1],
            "time_seconds": [10.0, 15.0, 20.0],
            "team_id": [1, 2, 1],
            "player_id": [101, 201, 102],
            "start_x": [30.0, 70.0, 50.0],
            "start_y": [34.0, 34.0, 20.0],
            "end_x": [60.0, 40.0, 80.0],
            "end_y": [30.0, 40.0, 34.0],
            "type_id": [0, 0, 1],  # 0=pass, 1=other
            "type_name": ["pass", "pass", "dribble"],
            "result_id": [1, 1, 1],
            "result_name": ["success", "success", "success"],
            "bodypart_id": [0, 0, 0],
            "bodypart_name": ["foot", "foot", "foot"],
        }
    )

    # Build frames at 0.04s intervals covering action times
    frame_rows = []
    rng = np.random.RandomState(42)
    for fid in range(600):  # 24 seconds at 25 fps
        t = fid * 0.04
        for tid in [1, 2]:
            for j in range(5):
                frame_rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "time_seconds": t,
                        "source_provider": "test",
                        "player_id": tid * 100 + j,
                        "team_id": tid,
                        "x": 20 + (tid - 1) * 60 + rng.uniform(-3, 3),
                        "y": 10 + j * 12 + rng.uniform(-2, 2),
                        "vx": 0.5,
                        "vy": 0.0,
                        "is_ball": False,
                        "is_goalkeeper": j == 0,
                    }
                )
        # Ball row
        frame_rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "source_provider": "test",
                "player_id": None,
                "team_id": None,
                "x": 52.5,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "is_goalkeeper": False,
            }
        )

    frames = pd.DataFrame(frame_rows)
    return actions, frames


class TestAddObso:
    def test_enrichment_columns(self):
        """add_obso adds 3 OBSO columns."""
        from silly_kicks.tracking.features import add_obso

        actions, frames = _make_pass_actions_and_frames()
        result = add_obso(actions, frames, home_team_id=1)
        expected_cols = {"obso_actual", "obso_peak", "obso_optimal"}
        added = set(result.columns) - set(actions.columns)
        # Provenance columns may also be added
        assert expected_cols.issubset(added)

    @pytest.mark.slow
    def test_obso_bounded(self):
        """OBSO columns are in [0, 1] or NaN."""
        from silly_kicks.tracking.features import add_obso

        actions, frames = _make_pass_actions_and_frames()
        result = add_obso(actions, frames, home_team_id=1)
        for col in ["obso_actual", "obso_peak", "obso_optimal"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= 0.0
                assert vals.max() <= 1.0


class TestObsoXfns:
    def test_column_count(self):
        """obso_xfns produces 9 VAEP columns (3 features x 3 states)."""
        from silly_kicks.tracking.features import obso_xfns

        xfns = obso_xfns(home_team_id=1)
        # Each xfn produces 3 columns (3 states via lift_to_states)
        assert len(xfns) == 3  # 3 per-Series helpers, each lifted

    def test_introspection_nan(self):
        """xfns produce NaN in introspection mode (frames=None)."""
        from silly_kicks.tracking.features import obso_xfns

        xfns = obso_xfns(home_team_id=1)
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [0],
                "period_id": [1],
                "time_seconds": [10.0],
                "team_id": [1],
                "player_id": [101],
                "start_x": [30.0],
                "start_y": [34.0],
                "end_x": [60.0],
                "end_y": [30.0],
                "type_id": [0],
                "type_name": ["pass"],
                "result_id": [1],
                "result_name": ["success"],
                "bodypart_id": [0],
                "bodypart_name": ["foot"],
            }
        )
        gamestates = [actions, actions, actions]
        for xfn in xfns:
            result = xfn(gamestates, None)
            assert result.isna().all().all() or result.isna().all()
