# tests/tracking/test_player_influence_e2e.py
"""Provider fixture tests for player influence primitives (TF-36 + TF-33).

Uses committed slim-parquet fixtures (Sportec, Metrica, SkillCorner) and Gradient Sports
synthetic fixtures for real-data validation of add_player_influence. Exercises
string-typed IDs (Sportec/Metrica/SkillCorner), partial NaN velocities
(Metrica), and the full smooth→derive→link→compute pipeline.

Not marked @pytest.mark.e2e: all fixtures are committed to the repo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking import play_left_to_right
from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
from tests.tracking._provider_inputs import (
    GRADIENTSPORTS_DIR,
    SLIM_DIR,
    load_provider_frames,
    synthesize_actions,
)

_SLIM_PROVIDERS = sorted(p.stem.replace("_slim", "") for p in SLIM_DIR.glob("*_slim.parquet"))
_PROVIDERS = _SLIM_PROVIDERS + (["gradientsports"] if GRADIENTSPORTS_DIR.exists() else [])

_OUTPUT_COLS = [
    "actor_reachable_area_m2",
    "off_ball_xt_team",
    "off_ball_xt_opponent",
    "off_ball_xt_diff",
    "reachable_area_team",
    "reachable_area_opponent",
    "reachable_area_diff",
]


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
    """Load + preprocess real provider data for player influence tests."""
    provider = request.param
    actions, frames, home_team_id = _prepare(provider)
    return provider, actions, frames, home_team_id, fitted_xt


class TestPlayerInfluenceProviders:
    """Per-provider: add_player_influence runs on real data and produces valid output."""

    def test_adds_expected_columns(self, provider_data):
        """All 7 player influence columns present, output length matches input."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt)
        for col in _OUTPUT_COLS:
            assert col in result.columns, f"{provider}: missing column {col}"
        assert len(result) == len(actions)

    def test_non_nan_coverage(self, provider_data):
        """At least 1 non-NaN value per output column on real data."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt)
        for col in _OUTPUT_COLS:
            n_valid = result[col].notna().sum()
            assert n_valid >= 1, f"{provider}: {col} has 0 non-NaN values out of {len(result)}"

    def test_physical_invariants(self, provider_data):
        """Physical invariants: areas >= 0, off_ball_xt >= 0, diff identity."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt)

        # Areas >= 0
        for col in ["actor_reachable_area_m2", "reachable_area_team", "reachable_area_opponent"]:
            vals = result[col].dropna()
            assert (vals >= 0.0).all(), f"{provider}: {col} has negative values"

        # Off-ball xT >= 0
        for col in ["off_ball_xt_team", "off_ball_xt_opponent"]:
            vals = result[col].dropna()
            assert (vals >= 0.0).all(), f"{provider}: {col} has negative values"

        # Diff identity: _diff = _team - _opponent
        valid = result["off_ball_xt_team"].notna()
        if valid.any():
            pd.testing.assert_series_equal(
                result.loc[valid, "off_ball_xt_diff"],
                (result.loc[valid, "off_ball_xt_team"] - result.loc[valid, "off_ball_xt_opponent"]).rename(
                    "off_ball_xt_diff"
                ),
                check_exact=False,
                atol=1e-10,
            )
            pd.testing.assert_series_equal(
                result.loc[valid, "reachable_area_diff"],
                (result.loc[valid, "reachable_area_team"] - result.loc[valid, "reachable_area_opponent"]).rename(
                    "reachable_area_diff"
                ),
                check_exact=False,
                atol=1e-10,
            )

    def test_team_area_lte_pitch_area(self, provider_data):
        """Per-action team reachable area <= total pitch area."""
        from silly_kicks.tracking.features import add_player_influence

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_player_influence(actions, frames, xt)

        pitch_area = 105.0 * 68.0
        for col in ["reachable_area_team", "reachable_area_opponent"]:
            vals = result[col].dropna()
            assert (vals <= pitch_area).all(), f"{provider}: {col} exceeds pitch area (max={vals.max():.1f})"
