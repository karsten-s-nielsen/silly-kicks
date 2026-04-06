import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlcfg
from silly_kicks.spadl import opta as opta
from silly_kicks.spadl.schema import SPADL_COLUMNS, ConversionReport

# ---------------------------------------------------------------------------
# Tests that use inline DataFrames (no external fixtures required)
# ---------------------------------------------------------------------------


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
# Tests below require Opta XML fixtures in tests/datasets/opta/ and the
# removed silly_kicks.data.opta loader.  They are marked e2e so they are
# skipped in normal CI runs.
# ---------------------------------------------------------------------------

pytestmark_e2e = pytest.mark.e2e


@pytest.mark.e2e
class TestSpadlConvertorE2E:
    """End-to-end SPADL converter tests that need Opta fixture files.

    These tests require:
    - tests/datasets/opta/f7-23-2018-1009316-matchresults.xml
    - tests/datasets/opta/f24-23-2018-1009316-eventdetails.xml
    - silly_kicks.data.opta.OptaLoader (removed)

    Skipped unless ``-m e2e`` is passed to pytest.
    """

    def test_placeholder(self) -> None:
        pytest.skip("Opta fixture data and data loaders are not available")
