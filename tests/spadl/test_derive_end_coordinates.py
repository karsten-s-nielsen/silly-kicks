"""Unit tests for _derive_end_coordinates (Bug #7)."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.base import _derive_end_coordinates


def _make_actions(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal SPADL-shaped DataFrame from row dicts."""
    defaults = {
        "game_id": 1,
        "period_id": 1,
        "team_id": 100,
        "player_id": 200,
        "bodypart_id": 0,
        "result_id": 1,
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


class TestPassClassDerivation:
    """Pass-class types get next-event end coordinates."""

    def test_pass_gets_next_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 10.0,
                    "start_x": 50.0,
                    "start_y": 30.0,
                    "end_x": 50.0,
                    "end_y": 30.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["tackle"],
                    "time_seconds": 12.0,
                    "start_x": 70.0,
                    "start_y": 40.0,
                    "end_x": 70.0,
                    "end_y": 40.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(70.0)
        assert result.loc[0, "end_y"] == pytest.approx(40.0)

    def test_cross_gets_next_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["cross"],
                    "time_seconds": 10.0,
                    "start_x": 80.0,
                    "start_y": 5.0,
                    "end_x": 80.0,
                    "end_y": 5.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["shot"],
                    "time_seconds": 12.0,
                    "start_x": 95.0,
                    "start_y": 34.0,
                    "end_x": 95.0,
                    "end_y": 34.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(95.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)

    def test_throw_in_gets_next_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["throw_in"],
                    "time_seconds": 10.0,
                    "start_x": 60.0,
                    "start_y": 0.0,
                    "end_x": 60.0,
                    "end_y": 0.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 11.0,
                    "start_x": 65.0,
                    "start_y": 10.0,
                    "end_x": 65.0,
                    "end_y": 10.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(65.0)
        assert result.loc[0, "end_y"] == pytest.approx(10.0)

    def test_clearance_gets_next_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["clearance"],
                    "time_seconds": 10.0,
                    "start_x": 15.0,
                    "start_y": 34.0,
                    "end_x": 15.0,
                    "end_y": 34.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 13.0,
                    "start_x": 55.0,
                    "start_y": 50.0,
                    "end_x": 55.0,
                    "end_y": 50.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(55.0)
        assert result.loc[0, "end_y"] == pytest.approx(50.0)

    def test_goalkick_gets_next_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["goalkick"],
                    "time_seconds": 10.0,
                    "start_x": 5.0,
                    "start_y": 34.0,
                    "end_x": 5.0,
                    "end_y": 34.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 12.0,
                    "start_x": 40.0,
                    "start_y": 20.0,
                    "end_x": 40.0,
                    "end_y": 20.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(40.0)
        assert result.loc[0, "end_y"] == pytest.approx(20.0)


class TestExclusions:
    """Types NOT in the derive set keep end = start."""

    def test_shot_keeps_end_equals_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["shot"],
                    "time_seconds": 10.0,
                    "start_x": 90.0,
                    "start_y": 34.0,
                    "end_x": 90.0,
                    "end_y": 34.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["keeper_save"],
                    "time_seconds": 11.0,
                    "start_x": 104.0,
                    "start_y": 34.0,
                    "end_x": 104.0,
                    "end_y": 34.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(90.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)

    def test_tackle_keeps_end_equals_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["tackle"],
                    "time_seconds": 10.0,
                    "start_x": 50.0,
                    "start_y": 30.0,
                    "end_x": 50.0,
                    "end_y": 30.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 11.0,
                    "start_x": 55.0,
                    "start_y": 35.0,
                    "end_x": 55.0,
                    "end_y": 35.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(50.0)
        assert result.loc[0, "end_y"] == pytest.approx(30.0)

    def test_keeper_save_keeps_end_equals_start(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["keeper_save"],
                    "time_seconds": 10.0,
                    "start_x": 104.0,
                    "start_y": 34.0,
                    "end_x": 104.0,
                    "end_y": 34.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["clearance"],
                    "time_seconds": 11.0,
                    "start_x": 100.0,
                    "start_y": 30.0,
                    "end_x": 100.0,
                    "end_y": 30.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(104.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)


class TestSourceDataGuard:
    """Rows where source already provided end != start are NOT overwritten."""

    def test_pass_with_source_end_preserved(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 10.0,
                    "start_x": 50.0,
                    "start_y": 30.0,
                    "end_x": 65.0,
                    "end_y": 35.0,
                },  # source provided different end
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["tackle"],
                    "time_seconds": 12.0,
                    "start_x": 70.0,
                    "start_y": 40.0,
                    "end_x": 70.0,
                    "end_y": 40.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        # Source end_x=65.0 must be preserved, NOT overwritten with 70.0
        assert result.loc[0, "end_x"] == pytest.approx(65.0)
        assert result.loc[0, "end_y"] == pytest.approx(35.0)

    def test_clearance_with_source_end_preserved(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["clearance"],
                    "time_seconds": 10.0,
                    "start_x": 15.0,
                    "start_y": 34.0,
                    "end_x": 55.0,
                    "end_y": 50.0,
                },  # source provided different end
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 13.0,
                    "start_x": 60.0,
                    "start_y": 20.0,
                    "end_x": 60.0,
                    "end_y": 20.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        # Source end preserved, not overwritten with 60.0/20.0
        assert result.loc[0, "end_x"] == pytest.approx(55.0)
        assert result.loc[0, "end_y"] == pytest.approx(50.0)


class TestPeriodBoundary:
    """Last action per period keeps end = start (no cross-period contamination)."""

    def test_last_action_period_1_not_contaminated(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 2700.0,
                    "start_x": 80.0,
                    "start_y": 34.0,
                    "end_x": 80.0,
                    "end_y": 34.0,
                    "period_id": 1,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 0.5,
                    "start_x": 50.0,
                    "start_y": 34.0,
                    "end_x": 50.0,
                    "end_y": 34.0,
                    "period_id": 2,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        # Period 1 last action keeps end = start (not contaminated by P2 start)
        assert result.loc[0, "end_x"] == pytest.approx(80.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)

    def test_period_2_action_gets_next_within_period(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 2700.0,
                    "start_x": 80.0,
                    "start_y": 34.0,
                    "end_x": 80.0,
                    "end_y": 34.0,
                    "period_id": 1,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 0.5,
                    "start_x": 50.0,
                    "start_y": 34.0,
                    "end_x": 50.0,
                    "end_y": 34.0,
                    "period_id": 2,
                },
                {
                    "action_id": 2,
                    "type_id": spadlconfig.actiontype_id["tackle"],
                    "time_seconds": 2.0,
                    "start_x": 55.0,
                    "start_y": 40.0,
                    "end_x": 55.0,
                    "end_y": 40.0,
                    "period_id": 2,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        # P2 first pass gets next action within P2
        assert result.loc[1, "end_x"] == pytest.approx(55.0)
        assert result.loc[1, "end_y"] == pytest.approx(40.0)


class TestEdgeCases:
    """Empty and single-row DataFrames."""

    def test_empty_dataframe(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 10.0,
                    "start_x": 50.0,
                    "start_y": 30.0,
                    "end_x": 50.0,
                    "end_y": 30.0,
                },
            ]
        ).iloc[0:0]
        result = _derive_end_coordinates(actions)
        assert len(result) == 0

    def test_single_action_keeps_end(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 10.0,
                    "start_x": 50.0,
                    "start_y": 30.0,
                    "end_x": 50.0,
                    "end_y": 30.0,
                },
            ]
        )
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(50.0)
        assert result.loc[0, "end_y"] == pytest.approx(30.0)

    def test_does_not_mutate_input(self):
        actions = _make_actions(
            [
                {
                    "action_id": 0,
                    "type_id": spadlconfig.actiontype_id["pass"],
                    "time_seconds": 10.0,
                    "start_x": 50.0,
                    "start_y": 30.0,
                    "end_x": 50.0,
                    "end_y": 30.0,
                },
                {
                    "action_id": 1,
                    "type_id": spadlconfig.actiontype_id["tackle"],
                    "time_seconds": 12.0,
                    "start_x": 70.0,
                    "start_y": 40.0,
                    "end_x": 70.0,
                    "end_y": 40.0,
                },
            ]
        )
        original_end_x = actions.loc[0, "end_x"]
        _derive_end_coordinates(actions)
        assert actions.loc[0, "end_x"] == original_end_x
