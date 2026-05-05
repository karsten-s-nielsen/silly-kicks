"""Provider fixture tests for off-ball runs + line-break (TF-4)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking import play_left_to_right
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_PROVIDERS = ["sportec", "metrica", "pff"]


@pytest.fixture(params=_PROVIDERS)
def provider_data(request):
    """Load frames and synthesize actions for a provider."""
    provider = request.param
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    # Determine home_team_id from majority team in frames
    team_counts = frames[~frames["is_ball"].astype(bool)]["team_id"].value_counts()
    home_team_id = team_counts.index[0]
    # LTR-normalize (kernels require this)
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    return actions, frames, home_team_id


class TestOffBallRunsProviders:
    def test_off_ball_runs_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_off_ball_runs

        actions, frames, home_team_id = provider_data
        result = add_off_ball_runs(actions, frames, home_team_id=home_team_id)
        assert "n_off_ball_runners_pre_window" in result.columns
        assert len(result) == len(actions)
        assert result["n_off_ball_runners_pre_window"].notna().sum() >= 1, "expected >=1 non-NaN off-ball-runner row"

    def test_line_break_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_line_break

        actions, frames, home_team_id = provider_data
        result = add_line_break(actions, frames, home_team_id=home_team_id)
        assert "line_break" in result.columns
        assert "n_attackers_behind_line" in result.columns
        assert len(result) == len(actions)
        assert result["n_attackers_behind_line"].notna().sum() >= 1, "expected >=1 non-NaN line-break row"

    def test_off_ball_context_no_crash(self, provider_data):
        from silly_kicks.tracking.features import add_off_ball_context

        actions, frames, home_team_id = provider_data
        result = add_off_ball_context(actions, frames, home_team_id=home_team_id)
        expected_cols = {
            "n_off_ball_runners_pre_window",
            "max_off_ball_run_displacement_pre_window",
            "mean_off_ball_run_speed_pre_window",
            "n_off_ball_runners_toward_goal_pre_window",
            "line_break",
            "n_attackers_behind_line",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) == len(actions)
        assert result["n_off_ball_runners_pre_window"].notna().sum() >= 1, "expected >=1 non-NaN off-ball-runner row"
        assert result["n_attackers_behind_line"].notna().sum() >= 1, "expected >=1 non-NaN line-break row"
