"""Provider fixture tests for team shape envelope (TF-31)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking import play_left_to_right
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_PROVIDERS = ["sportec", "metrica", "gradientsports", "skillcorner"]


@pytest.fixture(params=_PROVIDERS)
def provider_data(request):
    """Load frames and synthesize actions for a provider."""
    provider = request.param
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    team_counts = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts()
    home_team_id = team_counts.index[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    return actions, frames, home_team_id


class TestTeamShapeProviders:
    def test_add_team_shape_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_team_shape

        actions, frames, home_team_id = provider_data
        result = add_team_shape(actions, frames, home_team_id=home_team_id)
        assert "team_shape_centroid_x_attacking" in result.columns
        assert "team_shape_centroid_x_defending" in result.columns
        assert len(result) == len(actions)
        assert result["team_shape_centroid_x_attacking"].notna().sum() >= 1, "expected >=1 non-NaN team_shape row"

    def test_team_shape_xfns_no_crash(self, provider_data):
        from silly_kicks.tracking.features import team_shape_xfns

        actions, frames, home_team_id = provider_data
        xfns = team_shape_xfns(home_team_id=home_team_id)
        xfn = xfns[0]

        states = [actions, actions, actions]
        result = xfn(states, frames)
        assert len(result.columns) == 54
        assert result["team_shape_centroid_x_attacking_a0"].notna().sum() >= 1
