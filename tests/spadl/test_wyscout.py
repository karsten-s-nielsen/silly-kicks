import pandas as pd
import pytest

from silly_kicks.spadl import config as spadl
from silly_kicks.spadl import wyscout as wy
from silly_kicks.spadl.schema import SPADL_COLUMNS, ConversionReport

# ---------------------------------------------------------------------------
# Tests that use inline DataFrames (no external fixtures required)
# ---------------------------------------------------------------------------


def test_insert_interception_passes() -> None:
    event = pd.DataFrame(
        [
            {
                "type_id": 8,
                "subtype_name": "Head pass",
                "tags": [{"id": 102}, {"id": 1401}, {"id": 1801}],  # own goal
                "player_id": 38093,
                "positions": [{"y": 56, "x": 5}, {"y": 100, "x": 100}],
                "game_id": 2499737,
                "type_name": "Pass",
                "team_id": 1610,
                "period_id": 2,
                "milliseconds": 2184.793924,
                "subtype_id": 82,
                "event_id": 180427412,
            }
        ]
    )
    actions, _ = wy.convert_to_actions(event, 1610)
    assert len(actions) == 2
    assert actions.at[0, "type_id"] == spadl.actiontypes.index("interception")
    assert actions.at[1, "type_id"] == spadl.actiontypes.index("bad_touch")
    assert actions.at[0, "result_id"] == spadl.results.index("success")
    assert actions.at[1, "result_id"] == spadl.results.index("owngoal")


def test_convert_own_goal_touches() -> None:
    """Tests conversion of own goals following a bad touch.

    Own goals resulting from bad touch events in the Wyscout event
    streams should be included in the SPADL representation.
    """
    # An own goal from the game between Leicester and Stoke on 24 Feb 2018.
    # Stoke's goalkeeper Jack Butland allows a low cross to bounce off his
    # gloves and into the net:
    event = pd.DataFrame(
        [
            {
                "type_id": 8,
                "subtype_name": "Cross",
                "tags": [{"id": 402}, {"id": 801}, {"id": 1802}],
                "player_id": 8013,
                "positions": [{"y": 89, "x": 97}, {"y": 0, "x": 0}],
                "game_id": 2499994,
                "type_name": "Pass",
                "team_id": 1631,
                "period_id": 2,
                "milliseconds": 1496.7290489999993,
                "subtype_id": 80,
                "event_id": 230320305,
            },
            {
                "type_id": 7,
                "subtype_name": "Touch",
                "tags": [{"id": 102}],
                "player_id": 8094,
                "positions": [{"y": 50, "x": 1}, {"y": 100, "x": 100}],
                "game_id": 2499994,
                "type_name": "Others on the ball",
                "team_id": 1639,
                "period_id": 2,
                "milliseconds": 1497.6330749999993,
                "subtype_id": 72,
                "event_id": 230320132,
            },
            {
                "type_id": 9,
                "subtype_name": "Reflexes",
                "tags": [{"id": 101}, {"id": 1802}],
                "player_id": 8094,
                "positions": [{"y": 100, "x": 100}, {"y": 50, "x": 1}],
                "game_id": 2499994,
                "type_name": "Save attempt",
                "team_id": 1639,
                "period_id": 2,
                "milliseconds": 1499.980547,
                "subtype_id": 90,
                "event_id": 230320135,
            },
        ]
    )
    actions, _ = wy.convert_to_actions(event, 1639)
    # FIXME: It adds a dribble between the bad touch of the goalkeeper and
    # his attempt to save the ball before crossing the line. Not sure
    # whether that is ideal.
    assert len(actions) == 4
    assert actions.at[1, "type_id"] == spadl.actiontypes.index("bad_touch")
    assert actions.at[1, "result_id"] == spadl.results.index("owngoal")


def test_convert_simulations_precede_by_take_on() -> None:
    events = pd.DataFrame(
        [
            {
                "type_id": 1,
                "subtype_name": "Ground attacking duel",
                "tags": [{"id": 503}, {"id": 701}, {"id": 1802}],
                "player_id": 8327,
                "positions": [{"y": 48, "x": 82}, {"y": 47, "x": 83}],
                "game_id": 2576263,
                "type_name": "Duel",
                "team_id": 3158,
                "period_id": 2,
                "milliseconds": 706.309475 * 1000,
                "subtype_id": 11,
                "event_id": 240828365,
            },
            {
                "type_id": 2,
                "subtype_name": "Simulation",
                "tags": [{"id": 1702}],
                "player_id": 8327,
                "positions": [{"y": 47, "x": 83}, {"y": 0, "x": 0}],
                "game_id": 2576263,
                "type_name": "Foul",
                "team_id": 3158,
                "period_id": 2,
                "milliseconds": 709.1020480000002 * 1000,
                "subtype_id": 25,
                "event_id": 240828368,
            },
        ]
    )

    actions, _ = wy.convert_to_actions(events, 3158)

    assert len(actions) == 1
    assert actions.at[0, "type_id"] == spadl.actiontypes.index("take_on")
    assert actions.at[0, "result_id"] == spadl.results.index("fail")


def test_convert_simulations() -> None:
    events = pd.DataFrame(
        [
            {
                "type_id": 8,
                "subtype_name": "Cross",
                "tags": [{"id": 402}, {"id": 801}, {"id": 1801}],
                "player_id": 20472,
                "positions": [{"y": 76, "x": 92}, {"y": 92, "x": 98}],
                "game_id": 2575974,
                "type_name": "Pass",
                "team_id": 3173,
                "period_id": 1,
                "milliseconds": 1010.5460250000001 * 1000,
                "subtype_id": 80,
                "event_id": 182640540,
            },
            {
                "type_id": 1,
                "subtype_name": "Ground loose ball duel",
                "tags": [{"id": 701}, {"id": 1802}],
                "player_id": 116171,
                "positions": [{"y": 92, "x": 98}, {"y": 43, "x": 87}],
                "game_id": 2575974,
                "type_name": "Duel",
                "team_id": 3173,
                "period_id": 1,
                "milliseconds": 1012.8018770000001 * 1000,
                "subtype_id": 13,
                "event_id": 182640541,
            },
            {
                "type_id": 2,
                "subtype_name": "Simulation",
                "tags": [{"id": 1702}],
                "player_id": 116171,
                "positions": [{"y": 43, "x": 87}, {"y": 100, "x": 100}],
                "game_id": 2575974,
                "type_name": "Foul",
                "team_id": 3173,
                "period_id": 1,
                "milliseconds": 1014.7540220000001 * 1000,
                "subtype_id": 25,
                "event_id": 182640542,
            },
        ]
    )

    actions, _ = wy.convert_to_actions(events, 3157)

    assert len(actions) == 3
    assert actions.at[2, "type_id"] == spadl.actiontypes.index("take_on")
    assert actions.at[2, "result_id"] == spadl.results.index("fail")


def test_wyscout_keeper_claim() -> None:
    """Bug #37/D44: Wyscout GK claim events must map to keeper_claim."""
    from silly_kicks.spadl.wyscout import _determine_type_id

    event = pd.Series({
        "type_id": 9, "subtype_id": 92,
        "fairplay": False, "own_goal": False, "high": False,
        "take_on_left": False, "take_on_right": False,
        "sliding_tackle": False, "interception": False,
    })
    assert _determine_type_id(event) == spadl.actiontype_id["keeper_claim"]


def test_wyscout_keeper_punch() -> None:
    """Bug #37/D44: Wyscout GK punch events must map to keeper_punch."""
    from silly_kicks.spadl.wyscout import _determine_type_id

    event = pd.Series({
        "type_id": 9, "subtype_id": 93,
        "fairplay": False, "own_goal": False, "high": False,
        "take_on_left": False, "take_on_right": False,
        "sliding_tackle": False, "interception": False,
    })
    assert _determine_type_id(event) == spadl.actiontype_id["keeper_punch"]


def test_wyscout_keeper_save_default() -> None:
    """Bug #37/D44: Wyscout GK reflexes/save events still map to keeper_save."""
    from silly_kicks.spadl.wyscout import _determine_type_id

    event = pd.Series({
        "type_id": 9, "subtype_id": 90,
        "fairplay": False, "own_goal": False, "high": False,
        "take_on_left": False, "take_on_right": False,
        "sliding_tackle": False, "interception": False,
    })
    assert _determine_type_id(event) == spadl.actiontype_id["keeper_save"]


def test_wyscout_returns_tuple():
    event = pd.DataFrame([{
        "type_id": 8, "subtype_name": "Simple pass", "subtype_id": 85,
        "tags": [{"id": 1801}],
        "player_id": 1, "positions": [{"y": 50, "x": 50}, {"y": 60, "x": 60}],
        "game_id": 1, "type_name": "Pass", "team_id": 100,
        "period_id": 1, "milliseconds": 1000.0, "event_id": 1,
    }])
    result = wy.convert_to_actions(event, 100)
    assert isinstance(result, tuple)
    _actions, report = result
    assert isinstance(report, ConversionReport)
    assert report.provider == "Wyscout"


def test_wyscout_output_columns():
    event = pd.DataFrame([{
        "type_id": 8, "subtype_name": "Simple pass", "subtype_id": 85,
        "tags": [{"id": 1801}],
        "player_id": 1, "positions": [{"y": 50, "x": 50}, {"y": 60, "x": 60}],
        "game_id": 1, "type_name": "Pass", "team_id": 100,
        "period_id": 1, "milliseconds": 1000.0, "event_id": 1,
    }])
    actions, _ = wy.convert_to_actions(event, 100)
    assert list(actions.columns) == list(SPADL_COLUMNS.keys())


def test_wyscout_input_validation():
    df = pd.DataFrame({"game_id": [1]})
    with pytest.raises(ValueError, match="Wyscout"):
        wy.convert_to_actions(df, home_team_id=100)


# ---------------------------------------------------------------------------
# Tests below require Wyscout fixture files in tests/datasets/wyscout_public/
# and the removed silly_kicks.data.wyscout loader.  They are marked e2e so
# they are skipped in normal CI runs.
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSpadlConvertorE2E:
    """End-to-end SPADL converter tests that need Wyscout fixture files.

    These tests require:
    - tests/datasets/wyscout_public/raw/ event data
    - silly_kicks.data.wyscout.PublicWyscoutLoader (removed)

    Skipped unless ``-m e2e`` is passed to pytest.
    """

    def test_placeholder(self) -> None:
        pytest.skip("Wyscout fixture data and data loaders are not available")
