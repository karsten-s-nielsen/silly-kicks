"""Provider fixture tests for GK influence primitives (TF-15).

Uses committed slim-parquet fixtures (Sportec, Metrica, SkillCorner) and Gradient Sports
synthetic fixtures for real-data validation of add_gk_influence. Exercises
string-typed IDs (Sportec/Metrica/SkillCorner), partial NaN velocities
(Metrica), and the full smooth→derive→link→compute pipeline.

Not marked @pytest.mark.e2e: all fixtures are committed to the repo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking import play_left_to_right
from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
from tests.tracking._goal_map_helpers import goal_map_like_home_team_id
from tests.tracking._provider_inputs import (
    GRADIENTSPORTS_DIR,
    SLIM_DIR,
    load_provider_frames,
    synthesize_actions,
)

_SLIM_PROVIDERS = sorted(p.stem.replace("_slim", "") for p in SLIM_DIR.glob("*_slim.parquet"))
_PROVIDERS = _SLIM_PROVIDERS + (["gradientsports"] if GRADIENTSPORTS_DIR.exists() else [])


def _prepare(provider: str) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Load frames, add velocities, determine home_team_id, synthesize actions."""
    frames = load_provider_frames(provider)
    if "vx" not in frames.columns:
        frames = derive_velocities(smooth_frames(frames))
    team_counts = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts()
    home_team_id = team_counts.index[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    actions = synthesize_actions(frames)
    return actions, frames, home_team_id


@pytest.fixture(params=_PROVIDERS)
def provider_data(request, fitted_xt):
    """Load + preprocess real provider data for GK influence tests."""
    provider = request.param
    actions, frames, home_team_id = _prepare(provider)
    return provider, actions, frames, home_team_id, fitted_xt


class TestGkInfluenceProviders:
    """Per-provider: add_gk_influence runs on real data and produces valid output."""

    def test_adds_expected_columns(self, provider_data):
        """All 4 GK influence columns present, output length matches input."""
        from silly_kicks.tracking.features import add_gk_influence

        provider, actions, frames, home_team_id, xt = provider_data
        result = add_gk_influence(actions, frames, xt, goal_map=goal_map_like_home_team_id(frames, home_team_id))
        expected_cols = {
            "gk_pitch_control_share_weighted",
            "gk_reachable_area_m2",
            "gk_closing_time_min_s__six_yard_box",
            "gk_closing_time_mean_s__six_yard_box",
        }
        assert expected_cols.issubset(set(result.columns)), f"{provider}: missing GK influence columns"
        assert len(result) == len(actions)

    def test_non_nan_coverage(self, provider_data):
        """At least 1 non-NaN value per primitive column on real data."""
        from silly_kicks.tracking.features import add_gk_influence

        provider, actions, frames, home_team_id, xt = provider_data
        result = add_gk_influence(actions, frames, xt, goal_map=goal_map_like_home_team_id(frames, home_team_id))
        for col in (
            "gk_pitch_control_share_weighted",
            "gk_reachable_area_m2",
            "gk_closing_time_min_s__six_yard_box",
            "gk_closing_time_mean_s__six_yard_box",
        ):
            n_valid = result[col].notna().sum()
            assert n_valid >= 1, f"{provider}: {col} has 0 non-NaN values out of {len(result)}"

    def test_physical_invariants(self, provider_data):
        """Physical invariants hold: share in [0,1], area >= 0, min_s <= mean_s."""
        from silly_kicks.tracking.features import add_gk_influence

        provider, actions, frames, home_team_id, xt = provider_data
        result = add_gk_influence(actions, frames, xt, goal_map=goal_map_like_home_team_id(frames, home_team_id))

        # Share in [0, 1]
        share = result["gk_pitch_control_share_weighted"].dropna()
        assert len(share) > 0, f"{provider}: no valid share values"
        assert (share >= 0.0).all(), f"{provider}: share < 0 found"
        assert (share <= 1.0).all(), f"{provider}: share > 1 found"

        # Area >= 0
        area = result["gk_reachable_area_m2"].dropna()
        assert len(area) > 0, f"{provider}: no valid area values"
        assert (area >= 0.0).all(), f"{provider}: area < 0 found"

        # min_s <= mean_s (with float tolerance)
        min_s = result["gk_closing_time_min_s__six_yard_box"]
        mean_s = result["gk_closing_time_mean_s__six_yard_box"]
        valid = min_s.notna() & mean_s.notna()
        assert valid.sum() > 0, f"{provider}: no valid closing time pairs"
        assert (min_s[valid] <= mean_s[valid] + 1e-9).all(), f"{provider}: min > mean closing time"
