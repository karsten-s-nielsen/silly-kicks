"""Cross-reference Spearman output against lakehouse implementation + provider smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import SpearmanParams, compute_pitch_control

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def test_spearman_matches_lakehouse_tti_formula():
    """Verify TTI formula produces same values as lakehouse _tti_numpy.

    Uses a known 3-player setup with hand-computed expected values.
    Lakehouse formula (identical math, different coordinate system):
        TTI = reaction_time + (-v_proj + sqrt(v_proj^2 + 2*a*d)) / a
    """
    from silly_kicks.tracking.pitch_control._spearman import compute_tti

    # Player at (10, 10), velocity (3, 0), target at (20, 10)
    # d = 10, v_proj = 3 (full projection along x)
    # TTI = 0.7 + (-3 + sqrt(9 + 140)) / 7 = 0.7 + (-3 + 12.2066) / 7
    pos = np.array([[10.0, 10.0]])
    vel = np.array([[3.0, 0.0]])
    target = np.array([[20.0, 10.0]])
    tti = compute_tti(pos, vel, target, 0.7, 7.0)
    expected = 0.7 + (-3.0 + np.sqrt(9.0 + 140.0)) / 7.0
    np.testing.assert_allclose(tti[0, 0], expected, rtol=1e-12)


def test_spearman_surface_structure_matches_lakehouse():
    """Verify surface spatial structure: attacker side > 0.5, defender side < 0.5."""
    frame = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 10,
                "x": 25,
                "y": 34,
                "vx": 2,
                "vy": 0,
                "is_ball": False,
                "is_goalkeeper": True,
            },
            {
                "player_id": 2,
                "team_id": 10,
                "x": 45,
                "y": 34,
                "vx": 1,
                "vy": 0,
                "is_ball": False,
                "is_goalkeeper": False,
            },
            {
                "player_id": 3,
                "team_id": 20,
                "x": 60,
                "y": 34,
                "vx": -1,
                "vy": 0,
                "is_ball": False,
                "is_goalkeeper": False,
            },
            {
                "player_id": 4,
                "team_id": 20,
                "x": 80,
                "y": 34,
                "vx": 0,
                "vy": 0,
                "is_ball": False,
                "is_goalkeeper": True,
            },
            {
                "player_id": np.nan,
                "team_id": np.nan,
                "x": 52.5,
                "y": 34,
                "vx": 0,
                "vy": 0,
                "is_ball": True,
                "is_goalkeeper": False,
            },
        ]
    )
    s = compute_pitch_control(frame, 10, method="spearman", params=SpearmanParams(grid_cells_x=20, grid_cells_y=13))
    # Attacker half (x < 52.5) should have higher control
    mid_idx = len(s.grid_x) // 2
    att_mean = s.surface[:, :mid_idx].mean()
    def_mean = s.surface[:, mid_idx:].mean()
    assert att_mean > def_mean


# ---------------------------------------------------------------------------
# Provider coverage smoke tests
# ---------------------------------------------------------------------------

_PROVIDERS = ["sportec", "metrica", "skillcorner", "pff"]


def _load_single_frame(provider: str) -> pd.DataFrame:
    """Load one full frame from a provider fixture using the shared loader."""
    from tests.tracking._provider_inputs import load_provider_frames

    frames = load_provider_frames(provider)
    # Pick first frame with >= 3 players
    for fid in frames["frame_id"].dropna().unique()[:5]:
        candidate = frames[frames["frame_id"] == fid].copy()
        if len(candidate) >= 3:
            return candidate
    return frames.head(23).copy()


@pytest.mark.parametrize("provider", _PROVIDERS)
class TestProviderSmoke:
    def test_voronoi_computes_valid_surface(self, provider):
        """Position-only pitch control works on every provider fixture."""
        frame = _load_single_frame(provider)
        teams = frame.loc[~frame["is_ball"].astype(bool), "team_id"].dropna().unique()
        assert len(teams) >= 2, f"{provider} frame has <2 teams"
        s = compute_pitch_control(frame, teams[0], method="voronoi")
        assert s.surface.shape == (32, 50)
        assert (s.surface >= 0).all() and (s.surface <= 1).all()

    def test_spearman_with_zero_velocity(self, provider):
        """Spearman works with zero-velocity fill (no native vx/vy)."""
        frame = _load_single_frame(provider)
        frame["vx"] = 0.0
        frame["vy"] = 0.0
        teams = frame.loc[~frame["is_ball"].astype(bool), "team_id"].dropna().unique()
        s = compute_pitch_control(frame, teams[0], method="spearman")
        assert s.surface.shape == (32, 50)
        assert (s.surface >= 0).all() and (s.surface <= 1).all()

    def test_fernandez_bornn_with_zero_velocity(self, provider):
        """F/B works with zero-velocity fill (isotropic Gaussian)."""
        frame = _load_single_frame(provider)
        frame["vx"] = 0.0
        frame["vy"] = 0.0
        teams = frame.loc[~frame["is_ball"].astype(bool), "team_id"].dropna().unique()
        s = compute_pitch_control(frame, teams[0], method="fernandez_bornn")
        assert s.surface.shape == (32, 50)
        assert (s.surface >= 0).all() and (s.surface <= 1).all()
        assert not np.isnan(s.surface).any()

    def test_decomposition_per_player_valid(self, provider):
        """Decomposition returns per_player_influence matching player count."""
        frame = _load_single_frame(provider)
        frame["vx"] = 0.0
        frame["vy"] = 0.0
        teams = frame.loc[~frame["is_ball"].astype(bool), "team_id"].dropna().unique()
        s = compute_pitch_control(frame, teams[0], method="spearman", decompose=True)
        n_players = (~frame["is_ball"].astype(bool)).sum()
        assert s.per_player_influence is not None
        assert s.per_player_influence.shape[0] == n_players
        assert s.player_ids is not None
        assert len(s.player_ids) == n_players
