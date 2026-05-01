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
    return pd.DataFrame(
        [
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
        ]
    )


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
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    {
                        "game_id": 1,
                        "event_id": "abc-789",
                        "period_id": 1,
                        "timestamp": "00:00:10.000",
                        "team_id": 100,
                        "player_id": 202,
                        "type_name": "FutureEventType",
                        "location": [50.0, 30.0],
                        "extra": {},
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
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


def test_statsbomb_goal_keeper_key_fallback():
    """extra["goal_keeper"] (snake_cased) should work the same as extra["goalkeeper"]."""
    events = pd.DataFrame(
        [
            {
                "game_id": 1,
                "event_id": "gk-1",
                "period_id": 1,
                "timestamp": "00:00:30.000",
                "team_id": 100,
                "player_id": 300,
                "type_name": "Goal Keeper",
                "location": [10.0, 34.0],
                "extra": {
                    "goal_keeper": {
                        "type": {"name": "Shot Saved"},
                        "outcome": {"name": "Success"},
                        "body_part": {"name": "Right Hand"},
                    }
                },
            },
        ]
    )
    actions, _report = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1)
    # The keeper action should be produced (not dropped as non_action)
    assert len(actions) >= 1
    from silly_kicks.spadl.config import actiontype_id

    keeper_types = {actiontype_id["keeper_save"], actiontype_id["keeper_claim"], actiontype_id["keeper_punch"]}
    assert any(actions["type_id"].isin(keeper_types)), (
        f"Expected a keeper action, got type_ids: {actions['type_id'].tolist()}"
    )


class TestStatsbombPreserveNative:
    """Tests for the ``preserve_native`` kwarg added in 1.1.0.

    Surfaces provider-native fields (e.g. StatsBomb's top-level ``possession``
    sequence number, ``possession_team``, ``play_pattern``) through conversion
    as extra columns alongside the canonical SPADL output.
    """

    @staticmethod
    def _events_with_possession() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "event_id": "ev-1",
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
                    "possession": 1,
                    "possession_team": {"id": 100, "name": "Home"},
                },
                {
                    "game_id": 1,
                    "event_id": "ev-2",
                    "period_id": 1,
                    "timestamp": "00:00:05.000",
                    "team_id": 100,
                    "player_id": 201,
                    "type_name": "Pass",
                    "location": [65.0, 42.0],
                    "extra": {
                        "pass": {
                            "end_location": [75.0, 42.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 1,
                    "possession_team": {"id": 100, "name": "Home"},
                },
                {
                    "game_id": 1,
                    "event_id": "ev-3",
                    "period_id": 1,
                    "timestamp": "00:00:10.000",
                    "team_id": 200,
                    "player_id": 300,
                    "type_name": "Pass",
                    "location": [60.0, 40.0],
                    "extra": {
                        "pass": {
                            "end_location": [50.0, 40.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 2,
                    "possession_team": {"id": 200, "name": "Away"},
                },
            ]
        )

    def test_default_none_unchanged(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
        )
        assert "possession" not in actions.columns
        assert list(actions.columns) == list(SPADL_COLUMNS.keys())

    def test_empty_list_unchanged(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1, preserve_native=[]
        )
        assert list(actions.columns) == list(SPADL_COLUMNS.keys())

    def test_single_field_preserved(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events,
            home_team_id=100,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
            preserve_native=["possession"],
        )
        assert "possession" in actions.columns
        # Map original_event_id → preserved possession value (skipping synthetic dribbles which have NaN).
        non_synthetic = actions[actions["original_event_id"].notna() & (actions["original_event_id"] != "")]
        possession_for = dict(zip(non_synthetic["original_event_id"], non_synthetic["possession"], strict=True))
        assert possession_for.get("ev-1") == 1
        assert possession_for.get("ev-2") == 1
        assert possession_for.get("ev-3") == 2

    def test_multiple_fields_preserved(self):
        events = self._events_with_possession()
        actions, _ = statsbomb.convert_to_actions(
            events,
            home_team_id=100,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
            preserve_native=["possession", "possession_team"],
        )
        assert "possession" in actions.columns
        assert "possession_team" in actions.columns

    def test_missing_field_raises(self):
        events = self._events_with_possession()
        with pytest.raises(ValueError, match="preserve_native"):
            statsbomb.convert_to_actions(
                events,
                home_team_id=100,
                xy_fidelity_version=1,
                shot_fidelity_version=1,
                preserve_native=["does_not_exist"],
            )

    def test_overlap_with_schema_raises(self):
        events = self._events_with_possession()
        with pytest.raises(ValueError, match=r"overlap|already"):
            statsbomb.convert_to_actions(
                events,
                home_team_id=100,
                xy_fidelity_version=1,
                shot_fidelity_version=1,
                preserve_native=["team_id"],
            )

    def test_synthetic_dribbles_get_nan(self):
        # Two same-team passes with sufficient gap to trigger _add_dribbles.
        events = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "event_id": "ev-A",
                    "period_id": 1,
                    "timestamp": "00:00:01.000",
                    "team_id": 100,
                    "player_id": 200,
                    "type_name": "Pass",
                    "location": [10.0, 40.0],
                    "extra": {
                        "pass": {
                            "end_location": [30.0, 40.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 1,
                },
                {
                    "game_id": 1,
                    "event_id": "ev-B",
                    "period_id": 1,
                    "timestamp": "00:00:03.000",
                    "team_id": 100,
                    "player_id": 200,
                    "type_name": "Pass",
                    "location": [50.0, 40.0],
                    "extra": {
                        "pass": {
                            "end_location": [70.0, 40.0],
                            "outcome": {"name": "Complete"},
                            "height": {"name": "Ground Pass"},
                        }
                    },
                    "possession": 1,
                },
            ]
        )
        actions, _ = statsbomb.convert_to_actions(
            events,
            home_team_id=100,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
            preserve_native=["possession"],
        )
        # Synthetic dribble has NULL original_event_id (per CHANGELOG).
        synthetic = actions[actions["original_event_id"].isna()]
        if len(synthetic) > 0:
            assert synthetic["possession"].isna().all()


# ---------------------------------------------------------------------------
# goalkeeper_ids — accepted as no-op for cross-provider API symmetry (1.10.0)
# StatsBomb's source events natively mark GK actions, so the parameter has
# no effect on output. Asserting byte-for-byte equivalence catches drift.
# ---------------------------------------------------------------------------


class TestStatsBombGoalkeeperIdsNoOp:
    @staticmethod
    def _events() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [1, 1],
                "event_id": ["e1", "e2"],
                "period_id": [1, 1],
                "timestamp": ["00:00:01.000", "00:00:02.000"],
                "team_id": [100, 100],
                "player_id": [200, 201],
                "type_name": ["Pass", "Pass"],
                "location": [[60.0, 40.0], [70.0, 40.0]],
                "extra": [{}, {}],
            }
        )

    def test_goalkeeper_ids_parameter_is_accepted(self):
        actions, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids={200, 300})
        assert isinstance(actions, pd.DataFrame)

    def test_goalkeeper_ids_output_identical_with_and_without(self):
        a_none, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100)
        a_set, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids={200})
        pd.testing.assert_frame_equal(a_none, a_set, check_dtype=True)

    def test_empty_goalkeeper_ids_set_equivalent_to_none(self):
        a_empty, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids=set())
        a_none, _ = statsbomb.convert_to_actions(self._events(), home_team_id=100, goalkeeper_ids=None)
        pd.testing.assert_frame_equal(a_empty, a_none, check_dtype=True)
