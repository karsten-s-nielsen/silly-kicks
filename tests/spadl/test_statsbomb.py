"""StatsBomb SPADL converter tests."""

import inspect

import pandas as pd
import pytest

from silly_kicks.spadl import statsbomb
from silly_kicks.spadl.schema import SPADL_COLUMNS, ConversionReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_statsbomb_events() -> pd.DataFrame:
    """Minimal StatsBomb event DataFrame for testing."""
    return pd.DataFrame([
        {
            "game_id": 1,
            "event_id": "abc-123",
            "period_id": 1,
            "timestamp": "00:00:01.000",
            "team_id": 100,
            "player_id": 200,
            "type_name": "Pass",
            "location": [60.0, 40.0],
            "extra": {
                "pass": {
                    "end_location": [70.0, 40.0],
                    "outcome": {"name": "Complete"},
                    "height": {"name": "Ground Pass"},
                }
            },
        },
        {
            "game_id": 1,
            "event_id": "abc-456",
            "period_id": 1,
            "timestamp": "00:00:05.000",
            "team_id": 100,
            "player_id": 201,
            "type_name": "Pressure",
            "location": [50.0, 30.0],
            "extra": {},
        },
    ])


# ---------------------------------------------------------------------------
# Tests that use inline data (no external fixtures required)
# ---------------------------------------------------------------------------


def test_statsbomb_no_inplace_fillna() -> None:
    """Bug #946: fillna must not use inplace=True (pandas 3.0 compat)."""
    source = inspect.getsource(statsbomb.convert_to_actions)
    assert "inplace=True" not in source, "inplace=True is deprecated in pandas 2.x"


def test_statsbomb_returns_tuple():
    events = _make_statsbomb_events()
    result = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert isinstance(result, tuple)
    assert len(result) == 2
    actions, report = result
    assert isinstance(actions, pd.DataFrame)
    assert isinstance(report, ConversionReport)


def test_statsbomb_conversion_report():
    events = _make_statsbomb_events()
    _actions, report = statsbomb.convert_to_actions(
        events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
    )
    assert report.provider == "StatsBomb"
    assert report.total_events == 2
    assert "Pass" in report.mapped_counts
    assert "Pressure" in report.excluded_counts
    assert not report.has_unrecognized


def test_statsbomb_unrecognized_event_warning():
    events = _make_statsbomb_events()
    events = pd.concat([events, pd.DataFrame([{
        "game_id": 1,
        "event_id": "abc-789",
        "period_id": 1,
        "timestamp": "00:00:10.000",
        "team_id": 100,
        "player_id": 202,
        "type_name": "FutureEventType",
        "location": [50.0, 30.0],
        "extra": {},
    }])], ignore_index=True)
    import warnings as w_mod
    with w_mod.catch_warnings(record=True) as w:
        w_mod.simplefilter("always")
        _actions, report = statsbomb.convert_to_actions(
            events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
        )
    assert report.has_unrecognized
    assert "FutureEventType" in report.unrecognized_counts
    unrecognized_warnings = [x for x in w if "unrecognized" in str(x.message).lower()]
    assert len(unrecognized_warnings) >= 1


def test_statsbomb_output_columns():
    events = _make_statsbomb_events()
    actions, _ = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert list(actions.columns) == list(SPADL_COLUMNS.keys())
    for col, dtype in SPADL_COLUMNS.items():
        assert str(actions[col].dtype) == dtype, f"{col}: expected {dtype}, got {actions[col].dtype}"


def test_statsbomb_input_validation():
    df = pd.DataFrame({"game_id": [1]})
    with pytest.raises(ValueError, match="StatsBomb"):
        statsbomb.convert_to_actions(df, home_team_id=100)


# ---------------------------------------------------------------------------
# Tests below require StatsBomb event fixtures and are marked e2e.
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSpadlConvertor:
    """End-to-end SPADL converter tests that need StatsBomb fixture files.

    These tests require:
    - tests/datasets/statsbomb/raw/events/7584.json  (Japan vs Belgium)
    - tests/datasets/statsbomb/raw/events/7577.json  (Morocco game)
    - tests/datasets/statsbomb/raw/events/9912.json  (high-fidelity coords)
    - tests/datasets/statsbomb/raw/lineups/7584.json
    - silly_kicks.data.statsbomb.StatsBombLoader (removed)

    Skipped unless ``-m e2e`` is passed to pytest.
    """

    def test_placeholder(self) -> None:
        pytest.skip("StatsBomb fixture data and data loaders are not available")
