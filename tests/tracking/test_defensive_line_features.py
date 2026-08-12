"""Tests for action-coupled defensive-line features (TF-14 s4)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tracking.test_defensive_line import _make_frame_rows


def _make_actions_for_defensive_line(team_id=1, time_seconds=1.0, period_id=1):
    """Actions by home team at known time."""
    return pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [1, 2],
            "period_id": [period_id, period_id],
            "time_seconds": [time_seconds, time_seconds],
            "team_id": [team_id, team_id],
            "player_id": [50, 51],
            "start_x": [50.0, 55.0],
            "start_y": [34.0, 34.0],
            "type_id": [0, 0],
        }
    )


class TestActionCoupledFeatures:
    def test_action_gets_opposing_team_line(self):
        from silly_kicks.tracking.features import defensive_line_x

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        # Home team acts -> should get AWAY team's defensive line
        actions = _make_actions_for_defensive_line(team_id=1)
        result = defensive_line_x(actions, frames)
        # Away back 4 (highest x): 95, 93, 91, 89 -> mean = 92.0
        assert result.iloc[0] == pytest.approx(92.0)

    def test_unlinked_action_nan(self):
        from silly_kicks.tracking.features import defensive_line_x

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            time_seconds=100.0,  # far from action
        )
        actions = _make_actions_for_defensive_line(time_seconds=1.0)
        result = defensive_line_x(actions, frames)
        assert pd.isna(result.iloc[0])

    def test_aggregator_column_count(self):
        from silly_kicks.tracking.features import add_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        original_cols = len(actions.columns)
        result = add_defensive_line(actions, frames)
        new_cols = len(result.columns) - original_cols
        assert new_cols == 10  # 6 feature + 4 provenance

    def test_aggregator_provenance_skip_if_exists(self):
        """Provenance cols already present -> not duplicated."""
        from silly_kicks.tracking.features import add_action_context, add_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        # add_action_context needs end_x/end_y for receiver_zone_density
        actions["end_x"] = [55.0, 60.0]
        actions["end_y"] = [34.0, 34.0]
        enriched = add_action_context(actions, frames)
        # Now add defensive line on top -- provenance already exists
        result = add_defensive_line(enriched, frames)
        # Should have exactly 6 new feature cols, no _x/_y suffixes
        assert "frame_id_x" not in result.columns
        assert "frame_id_y" not in result.columns

    def test_xfns_factory_produces_valid(self):
        from silly_kicks.tracking.features import defensive_line_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        xfns = defensive_line_xfns()
        assert len(xfns) == 1
        assert is_frame_aware(xfns[0])

    def test_xfns_factory_has_name(self):
        from silly_kicks.tracking.features import defensive_line_xfns

        xfns = defensive_line_xfns()
        assert xfns[0].__name__ == "defensive_line"

    def test_xfns_column_count(self):
        """Factory transformer emits 6 x 3 = 18 columns."""
        from silly_kicks.tracking.features import defensive_line_xfns
        from silly_kicks.vaep.feature_framework import gamestates

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        states = gamestates(actions, nb_prev_actions=3)
        xfn = defensive_line_xfns()[0]
        result = xfn(states, frames)
        assert result.shape[1] == 18  # 6 cols x 3 states

    def test_batch_kernel_called_once(self):
        """Verify compute_defensive_line is called 3x (once per state), not 18x."""
        from unittest.mock import patch

        from silly_kicks.tracking._defensive_line import compute_defensive_line
        from silly_kicks.tracking.features import defensive_line_xfns
        from silly_kicks.vaep.feature_framework import gamestates

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_actions_for_defensive_line(team_id=1)
        states = gamestates(actions, nb_prev_actions=3)
        xfn = defensive_line_xfns()[0]
        with patch(
            "silly_kicks.tracking._defensive_line.compute_defensive_line",
            wraps=compute_defensive_line,
        ) as mock_cdl:
            xfn(states, frames)
            assert mock_cdl.call_count == 3  # once per state slot
