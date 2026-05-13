"""Provider fixture tests for Ward line-breaking (TF-32)."""

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


class TestLineBreakingProviders:
    def test_add_line_break_ward_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_line_break

        actions, frames, home_team_id = provider_data
        result = add_line_break(actions, frames, home_team_id=home_team_id, method="ward")
        assert "line_break__ward" in result.columns
        assert "lines_broken__ward" in result.columns
        assert "line_breaking_type__ward" in result.columns
        assert len(result) == len(actions)
        # At least one action should have a valid result
        assert result["lines_broken__ward"].notna().sum() >= 1, "expected >=1 non-NaN lines_broken__ward row"

    def test_line_breaking_ward_xfns_no_crash(self, provider_data):
        from silly_kicks.tracking.features import line_breaking_ward_xfns

        actions, frames, home_team_id = provider_data
        xfns = line_breaking_ward_xfns(home_team_id=home_team_id)
        xfn = xfns[0]

        states = [actions, actions, actions]
        result = xfn(states, frames)
        assert len(result.columns) == 9

    def test_threshold_still_works(self, provider_data):
        """Existing method='threshold' unaffected by new method= kwarg."""
        from silly_kicks.tracking.features import add_line_break

        actions, frames, home_team_id = provider_data
        result = add_line_break(actions, frames, home_team_id=home_team_id, method="threshold")
        assert "line_break" in result.columns
        assert "n_attackers_behind_line" in result.columns
