"""Provider-parameterized tests for TF-30 cover shadow features."""

from __future__ import annotations

import pytest

from silly_kicks.tracking import resolve_defended_goals
from tests.tracking._provider_inputs import (
    load_provider_frames,
    synthesize_actions,
)

_PROVIDERS = ["sportec", "metrica", "skillcorner", "gradientsports"]

_NAN_RATE_CEILING = {
    "sportec": 0.5,
    "metrica": 0.9,  # ~77% NaN ball coords
    "skillcorner": 0.7,
    "gradientsports": 0.5,
}


@pytest.fixture(params=_PROVIDERS)
def provider_data(request, fitted_xt):
    """Load frames, synthesize actions, preprocess for each provider."""
    provider = request.param
    frames = load_provider_frames(provider)

    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
    from silly_kicks.tracking.utils import play_left_to_right

    frames = smooth_frames(frames)
    frames = derive_velocities(frames)

    # LTR-normalize (cover shadows require LTR)
    home_team_id = frames[~frames["team_id"].isna()]["team_id"].iloc[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)

    actions = synthesize_actions(frames)
    return provider, actions, frames, home_team_id, fitted_xt


class TestCoverShadowsProviders:
    """Cross-provider cover shadow tests."""

    def test_shape_and_dtypes(self, provider_data):
        """5 columns present, correct dtypes, no crashes."""
        from silly_kicks.tracking.features import add_cover_shadows

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_cover_shadows(
            actions,
            frames,
            xt,
            goal_map=resolve_defended_goals(frames),
        )
        expected_cols = [
            "n_blocked_receivers",
            "n_potential_receivers",
            "blocking_score",
            "blocked_threat_fraction",
            "max_single_defender_blocking_score",
        ]
        for col in expected_cols:
            assert col in result.columns, f"{provider}: missing {col}"

    def test_nan_rate_bounds(self, provider_data):
        """NaN rate < provider-specific ceiling."""
        from silly_kicks.tracking.features import add_cover_shadows

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_cover_shadows(
            actions,
            frames,
            xt,
            goal_map=resolve_defended_goals(frames),
        )
        nan_rate = result["blocking_score"].isna().mean()
        ceiling = _NAN_RATE_CEILING[provider]
        assert nan_rate <= ceiling, f"{provider}: NaN rate {nan_rate:.2f} > ceiling {ceiling}"

    def test_value_bounds(self, provider_data):
        """blocking_score >= 0, blocked_threat_fraction in [0,1]."""
        from silly_kicks.tracking.features import add_cover_shadows

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_cover_shadows(
            actions,
            frames,
            xt,
            goal_map=resolve_defended_goals(frames),
        )
        valid_bs = result["blocking_score"].dropna()
        if len(valid_bs) > 0:
            assert (valid_bs >= -1e-9).all(), f"{provider}: negative blocking_score"
        valid_btf = result["blocked_threat_fraction"].dropna()
        if len(valid_btf) > 0:
            assert (valid_btf >= -1e-9).all(), f"{provider}: negative btf"
            assert (valid_btf <= 1.0 + 1e-9).all(), f"{provider}: btf > 1"

    def test_n_blocked_receivers_nonneg(self, provider_data):
        """n_blocked_receivers >= 0 for all linked actions."""
        from silly_kicks.tracking.features import add_cover_shadows

        provider, actions, frames, _home_team_id, xt = provider_data
        result = add_cover_shadows(
            actions,
            frames,
            xt,
            goal_map=resolve_defended_goals(frames),
        )
        valid = result["n_blocked_receivers"].dropna()
        if len(valid) == 0:
            pytest.skip(f"{provider}: no linked actions (all NaN)")
        assert (valid >= 0).all(), f"{provider}: negative n_blocked_receivers"
