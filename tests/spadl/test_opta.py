import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlcfg
from silly_kicks.spadl import opta as opta
from silly_kicks.spadl.schema import SPADL_COLUMNS, ConversionReport

# ---------------------------------------------------------------------------
# Tests that use inline DataFrames (no external fixtures required)
# ---------------------------------------------------------------------------


class TestOptaPreserveNative:
    """Tests for the ``preserve_native`` kwarg added in 1.1.0."""

    @staticmethod
    def _events_with_extras() -> pd.DataFrame:
        base = {
            "game_id": 318175,
            "type_id": 1,
            "period_id": 1,
            "minute": 2,
            "second": 14,
            "timestamp": "2010-01-27 19:47:14",
            "player_id": 8786,
            "team_id": 157,
            "outcome": True,
            "start_x": 50.0,
            "start_y": 50.0,
            "end_x": 60.0,
            "end_y": 50.0,
            "assist": False,
            "keypass": False,
            "qualifiers": {1: True},
            "type_name": "pass",
        }
        return pd.DataFrame(
            [
                {**base, "event_id": 100001, "competition_id": "EPL", "match_phase": "open"},
                {**base, "event_id": 100002, "competition_id": "EPL", "match_phase": "open"},
            ]
        )

    def test_default_none_unchanged(self):
        events = self._events_with_extras()
        actions, _ = opta.convert_to_actions(events, home_team_id=157)
        assert "competition_id" not in actions.columns
        assert list(actions.columns) == list(SPADL_COLUMNS.keys())

    def test_single_field_preserved(self):
        events = self._events_with_extras()
        actions, _ = opta.convert_to_actions(events, home_team_id=157, preserve_native=["competition_id"])
        assert "competition_id" in actions.columns
        non_synthetic = actions[actions["original_event_id"].notna()]
        assert all(v == "EPL" for v in non_synthetic["competition_id"])

    def test_multiple_fields_preserved(self):
        events = self._events_with_extras()
        actions, _ = opta.convert_to_actions(
            events, home_team_id=157, preserve_native=["competition_id", "match_phase"]
        )
        assert "competition_id" in actions.columns
        assert "match_phase" in actions.columns

    def test_missing_field_raises(self):
        events = self._events_with_extras()
        with pytest.raises(ValueError, match="preserve_native"):
            opta.convert_to_actions(events, home_team_id=157, preserve_native=["does_not_exist"])

    def test_overlap_with_schema_raises(self):
        events = self._events_with_extras()
        with pytest.raises(ValueError, match=r"overlap|already"):
            opta.convert_to_actions(events, home_team_id=157, preserve_native=["team_id"])


def test_convert_goalkick() -> None:
    event = pd.DataFrame(
        [
            {
                "game_id": 318175,
                "event_id": 1619686768,
                "type_id": 1,
                "period_id": 1,
                "minute": 2,
                "second": 14,
                "timestamp": "2010-01-27 19:47:14",
                "player_id": 8786,
                "team_id": 157,
                "outcome": False,
                "start_x": 5.0,
                "start_y": 37.0,
                "end_x": 73.0,
                "end_y": 18.7,
                "assist": False,
                "keypass": False,
                "qualifiers": {
                    56: "Right",
                    141: "18.7",
                    124: True,
                    140: "73.0",
                    1: True,
                },
                "type_name": "pass",
            }
        ]
    )
    actions, _ = opta.convert_to_actions(event, 0)
    action = actions.iloc[0]
    assert action["type_id"] == spadlcfg.actiontypes.index("goalkick")


def test_convert_own_goal() -> None:
    event = pd.DataFrame(
        [
            {
                "game_id": 318175,
                "event_id": 1619686768,
                "type_id": 16,
                "period_id": 1,
                "minute": 2,
                "second": 14,
                "timestamp": "2010-01-27 19:47:14",
                "player_id": 8786,
                "team_id": 157,
                "outcome": 1,
                "start_x": 5.0,
                "start_y": 37.0,
                "end_x": 73.0,
                "end_y": 18.7,
                "assist": False,
                "keypass": False,
                "qualifiers": {28: True},
                "type_name": "goal",
            }
        ]
    )
    actions, _ = opta.convert_to_actions(event, 0)
    action = actions.iloc[0]
    assert action["type_id"] == spadlcfg.actiontypes.index("bad_touch")
    assert action["result_id"] == spadlcfg.results.index("owngoal")


def test_opta_card_events_mapped() -> None:
    """Bug #784: Card events should be mapped, not dropped."""
    from silly_kicks.spadl.opta import _get_result_id, _get_type_id

    # Yellow card (no qualifier 32)
    type_id = _get_type_id(("card", True, {}))
    assert type_id == spadlcfg.actiontype_id["foul"]

    result_id = _get_result_id(("card", True, {}))
    assert result_id == spadlcfg.result_id["yellow_card"]

    # Red card (qualifier 32 present)
    result_id_red = _get_result_id(("card", True, {32: True}))
    assert result_id_red == spadlcfg.result_id["red_card"]


def test_opta_returns_tuple():
    event = pd.DataFrame(
        [
            {
                "game_id": 1,
                "event_id": 100,
                "type_id": 1,
                "period_id": 1,
                "minute": 1,
                "second": 0,
                "team_id": 10,
                "player_id": 20,
                "outcome": False,
                "start_x": 50.0,
                "start_y": 50.0,
                "end_x": 60.0,
                "end_y": 50.0,
                "qualifiers": {124: True},
                "type_name": "pass",
            }
        ]
    )
    result = opta.convert_to_actions(event, home_team_id=10)
    assert isinstance(result, tuple)
    actions, report = result
    assert isinstance(actions, pd.DataFrame)
    assert isinstance(report, ConversionReport)
    assert report.provider == "Opta"


def test_opta_output_columns():
    event = pd.DataFrame(
        [
            {
                "game_id": 1,
                "event_id": 100,
                "type_id": 1,
                "period_id": 1,
                "minute": 1,
                "second": 0,
                "team_id": 10,
                "player_id": 20,
                "outcome": True,
                "start_x": 50.0,
                "start_y": 50.0,
                "end_x": 60.0,
                "end_y": 50.0,
                "qualifiers": {124: True},
                "type_name": "pass",
            }
        ]
    )
    actions, _ = opta.convert_to_actions(event, home_team_id=10)
    assert list(actions.columns) == list(SPADL_COLUMNS.keys())
    for col, dtype in SPADL_COLUMNS.items():
        assert str(actions[col].dtype) == dtype, f"{col}: expected {dtype}, got {actions[col].dtype}"


def test_opta_input_validation():
    df = pd.DataFrame({"game_id": [1]})
    with pytest.raises(ValueError, match="Opta"):
        opta.convert_to_actions(df, home_team_id=10)


# ---------------------------------------------------------------------------
# goalkeeper_ids — accepted as no-op for cross-provider API symmetry (1.10.0)
# ---------------------------------------------------------------------------


class TestOptaGoalkeeperIdsNoOp:
    @staticmethod
    def _events() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [1, 1],
                "event_id": ["e1", "e2"],
                "period_id": [1, 1],
                "minute": [0, 0],
                "second": [1, 2],
                "team_id": [100, 100],
                "player_id": [200, 201],
                "type_name": ["pass", "pass"],
                "outcome": [True, True],
                "start_x": [50.0, 60.0],
                "start_y": [50.0, 50.0],
                "end_x": [60.0, 70.0],
                "end_y": [50.0, 50.0],
                "qualifiers": [{}, {}],
            }
        )

    def test_goalkeeper_ids_parameter_is_accepted(self):
        actions, _ = opta.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids={200})
        assert isinstance(actions, pd.DataFrame)

    def test_goalkeeper_ids_output_identical_with_and_without(self):
        a_none, _ = opta.convert_to_actions(self._events(), home_team_id=100)
        a_set, _ = opta.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids={200})
        pd.testing.assert_frame_equal(a_none, a_set, check_dtype=True)

    def test_empty_goalkeeper_ids_set_equivalent_to_none(self):
        a_empty, _ = opta.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids=set())
        a_none, _ = opta.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids=None)
        pd.testing.assert_frame_equal(a_empty, a_none, check_dtype=True)
