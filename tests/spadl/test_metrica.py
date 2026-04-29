"""Metrica DataFrame SPADL converter tests."""

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import metrica as metrica_mod
from silly_kicks.spadl.schema import KLOPPY_SPADL_COLUMNS

_KLOPPY_FIXTURES_DIR = Path(__file__).parent.parent / "datasets" / "kloppy"

_REQUIRED_COLS = [
    "match_id",
    "event_id",
    "type",
    "subtype",
    "period",
    "start_time_s",
    "end_time_s",
    "player",
    "team",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def _df_minimal_pass() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["Sample_Game_1"],
            "event_id": [1],
            "type": ["PASS"],
            "subtype": [None],
            "period": [1],
            "start_time_s": [10.0],
            "end_time_s": [11.0],
            "player": ["Home_11"],
            "team": ["Home"],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [55.0],
            "end_y": [34.0],
        }
    )


def _df_metrica(typ: str, subtype: str | None = None, **overrides) -> pd.DataFrame:
    base = _df_minimal_pass()
    base["type"] = [typ]
    base["subtype"] = [subtype]
    for k, v in overrides.items():
        base[k] = [v]
    return base


class TestMetricaContract:
    def test_returns_tuple_dataframe_conversion_report(self):
        actions, report = metrica_mod.convert_to_actions(_df_minimal_pass(), home_team_id="Home")
        assert isinstance(actions, pd.DataFrame)
        assert report.provider == "Metrica"

    def test_output_schema_matches_kloppy_spadl_columns(self):
        actions, _ = metrica_mod.convert_to_actions(_df_minimal_pass(), home_team_id="Home")
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())

    def test_dtypes_match_schema(self):
        actions, _ = metrica_mod.convert_to_actions(_df_minimal_pass(), home_team_id="Home")
        for col, expected in KLOPPY_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected

    def test_empty_input_returns_empty_actions_with_schema(self):
        events = pd.DataFrame({c: [] for c in _REQUIRED_COLS})
        actions, _report = metrica_mod.convert_to_actions(events, home_team_id="Home")
        assert len(actions) == 0
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())

    def test_input_dataframe_not_mutated(self):
        events = _df_minimal_pass()
        original_columns = list(events.columns)
        original_len = len(events)
        _, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        assert list(events.columns) == original_columns
        assert len(events) == original_len


class TestMetricaRequiredColumns:
    @pytest.mark.parametrize("missing", _REQUIRED_COLS)
    def test_missing_required_column_raises(self, missing):
        events = _df_minimal_pass().drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            metrica_mod.convert_to_actions(events, home_team_id="Home")


class TestMetricaActionMapping:
    def test_pass_default(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("PASS"), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_pass_cross(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("PASS", "CROSS"), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["cross"]

    def test_pass_goalkick(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("PASS", "GOAL KICK"), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["goalkick"]

    def test_pass_head_bodypart(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("PASS", "HEAD"), home_team_id="Home")
        assert actions["bodypart_id"].iloc[0] == spadlconfig.bodypart_id["head"]

    @pytest.mark.parametrize(
        "subtype,expected_result",
        [
            ("ON TARGET", "fail"),
            ("OFF TARGET", "fail"),
            ("BLOCKED", "fail"),
            ("WOODWORK", "fail"),
            ("GOAL", "success"),
        ],
    )
    def test_shot_outcomes(self, subtype, expected_result):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("SHOT", subtype), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["shot"]
        assert actions["result_id"].iloc[0] == spadlconfig.result_id[expected_result]

    def test_recovery_maps_to_interception(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("RECOVERY"), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["interception"]

    def test_challenge_won_maps_to_tackle(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("CHALLENGE", "WON"), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["tackle"]

    def test_challenge_lost_dropped(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("CHALLENGE", "LOST"), home_team_id="Home")
        assert len(actions) == 0

    def test_ball_lost_maps_to_bad_touch_fail(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("BALL LOST"), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["bad_touch"]
        assert actions["result_id"].iloc[0] == spadlconfig.result_id["fail"]

    def test_fault_maps_to_foul_fail(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("FAULT"), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["foul"]
        assert actions["result_id"].iloc[0] == spadlconfig.result_id["fail"]

    @pytest.mark.parametrize(
        "subtype,expected_type",
        [
            ("FREE KICK", "freekick_short"),
            ("CORNER KICK", "corner_short"),
            ("THROW IN", "throw_in"),
            ("GOAL KICK", "goalkick"),
        ],
    )
    def test_set_piece_dispatch(self, subtype, expected_type):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("SET PIECE", subtype), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id[expected_type]

    def test_set_piece_kickoff_dropped(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("SET PIECE", "KICK OFF"), home_team_id="Home")
        assert len(actions) == 0

    def test_excluded_types_dropped(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("BALL OUT"), home_team_id="Home")
        assert len(actions) == 0
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("SUBSTITUTION"), home_team_id="Home")
        assert len(actions) == 0
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("FAULT RECEIVED"), home_team_id="Home")
        assert len(actions) == 0


class TestMetricaSetPieceShotComposition:
    def test_freekick_then_shot_within_5s_same_player_upgrades_shot(self):
        events = pd.DataFrame(
            {
                "match_id": ["G1", "G1"],
                "event_id": [1, 2],
                "type": ["SET PIECE", "SHOT"],
                "subtype": ["FREE KICK", "ON TARGET"],
                "period": [1, 1],
                "start_time_s": [10.0, 12.0],
                "end_time_s": [10.5, 12.5],
                "player": ["Home_11", "Home_11"],
                "team": ["Home", "Home"],
                "start_x": [50.0, 95.0],
                "start_y": [34.0, 34.0],
                "end_x": [80.0, 100.0],
                "end_y": [34.0, 34.0],
            }
        )
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["shot_freekick"]

    def test_freekick_then_shot_different_player_no_upgrade(self):
        events = pd.DataFrame(
            {
                "match_id": ["G1", "G1"],
                "event_id": [1, 2],
                "type": ["SET PIECE", "SHOT"],
                "subtype": ["FREE KICK", "ON TARGET"],
                "period": [1, 1],
                "start_time_s": [10.0, 12.0],
                "end_time_s": [10.5, 12.5],
                "player": ["Home_11", "Home_22"],
                "team": ["Home", "Home"],
                "start_x": [50.0, 95.0],
                "start_y": [34.0, 34.0],
                "end_x": [80.0, 100.0],
                "end_y": [34.0, 34.0],
            }
        )
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        type_set = set(actions["type_id"].tolist())
        assert spadlconfig.actiontype_id["freekick_short"] in type_set
        assert spadlconfig.actiontype_id["shot"] in type_set
        assert spadlconfig.actiontype_id["shot_freekick"] not in type_set

    def test_freekick_then_shot_over_5s_no_upgrade(self):
        events = pd.DataFrame(
            {
                "match_id": ["G1", "G1"],
                "event_id": [1, 2],
                "type": ["SET PIECE", "SHOT"],
                "subtype": ["FREE KICK", "ON TARGET"],
                "period": [1, 1],
                "start_time_s": [10.0, 16.5],
                "end_time_s": [10.5, 17.0],
                "player": ["Home_11", "Home_11"],
                "team": ["Home", "Home"],
                "start_x": [50.0, 95.0],
                "start_y": [34.0, 34.0],
                "end_x": [80.0, 100.0],
                "end_y": [34.0, 34.0],
            }
        )
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        assert len(actions) == 2

    def test_corner_then_shot_no_upgrade_corner_retained(self):
        events = pd.DataFrame(
            {
                "match_id": ["G1", "G1"],
                "event_id": [1, 2],
                "type": ["SET PIECE", "SHOT"],
                "subtype": ["CORNER KICK", "ON TARGET"],
                "period": [1, 1],
                "start_time_s": [10.0, 12.0],
                "end_time_s": [10.5, 12.5],
                "player": ["Home_11", "Home_11"],
                "team": ["Home", "Home"],
                "start_x": [105.0, 95.0],
                "start_y": [0.0, 34.0],
                "end_x": [95.0, 100.0],
                "end_y": [34.0, 34.0],
            }
        )
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        type_set = set(actions["type_id"].tolist())
        assert spadlconfig.actiontype_id["corner_short"] in type_set
        assert spadlconfig.actiontype_id["shot"] in type_set


class TestMetricaCoordinateClamping:
    def test_negative_start_x_clamped(self):
        events = _df_minimal_pass()
        events["start_x"] = [-1.5]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        assert actions["start_x"].iloc[0] >= 0

    def test_oversized_start_y_clamped(self):
        events = _df_minimal_pass()
        events["start_y"] = [70.0]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        assert actions["start_y"].iloc[0] <= 68


class TestMetricaDirectionOfPlay:
    def test_away_team_x_flipped(self):
        events = _df_minimal_pass()
        events["team"] = ["Away"]
        events["start_x"] = [30.0]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        assert actions["start_x"].iloc[0] == 75.0


class TestMetricaPreserveNative:
    def test_preserve_native_passes_through(self):
        events = _df_minimal_pass()
        events["my_extra"] = ["foo"]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", preserve_native=["my_extra"])
        assert actions["my_extra"].iloc[0] == "foo"

    def test_preserve_native_schema_overlap_raises(self):
        events = _df_minimal_pass()
        events["team_id"] = ["overlap"]
        with pytest.raises(ValueError, match=r"overlap|already"):
            metrica_mod.convert_to_actions(events, home_team_id="Home", preserve_native=["team_id"])


# ---------------------------------------------------------------------------
# Cross-path consistency: kloppy path vs dedicated Metrica DataFrame path
# ---------------------------------------------------------------------------

_KLOPPY_TO_METRICA_TYPE = {
    "PASS": ("PASS", None),
    "SHOT": ("SHOT", None),
    "RECOVERY": ("RECOVERY", None),
    "FOUL_COMMITTED": ("FAULT", None),
    "BALL_OUT": ("BALL OUT", None),
    "CARD": ("CARD", None),
    "DUEL": ("CHALLENGE", "WON"),
    "CARRY": ("PASS", None),
    "GENERIC": ("GENERIC", None),
}


def _kloppy_dataset_to_metrica_df(dataset) -> pd.DataFrame:
    """Bridge kloppy EventDataset to a Metrica bronze-shaped DataFrame."""
    rows = []
    for ev in dataset.events:
        et_name = ev.event_type.name if hasattr(ev.event_type, "name") else str(ev.event_type)
        m_type, m_subtype = _KLOPPY_TO_METRICA_TYPE.get(et_name, ("GENERIC", None))
        coords = ev.coordinates
        x = coords.x if coords is not None else 0.0
        y = coords.y if coords is not None else 0.0
        rows.append(
            {
                "match_id": "cross_test",
                "event_id": ev.event_id,
                "type": m_type,
                "subtype": m_subtype,
                "period": ev.period.id,
                "start_time_s": ev.timestamp.total_seconds(),
                "end_time_s": ev.timestamp.total_seconds() + 0.5,
                "player": ev.player.player_id if ev.player else None,
                "team": ev.team.team_id if ev.team else None,
                "start_x": x,
                "start_y": y,
                "end_x": x,
                "end_y": y,
            }
        )
    return pd.DataFrame(rows)


class TestMetricaCrossPathConsistency:
    """V1 cross-path proof: kloppy path and dedicated DataFrame path produce
    non-empty SPADL DataFrames with the same column schema."""

    def test_kloppy_and_dedicated_paths_produce_equivalent_shape(self, metrica_dataset):
        from silly_kicks.spadl import kloppy as kloppy_mod

        actions_kloppy, _ = kloppy_mod.convert_to_actions(metrica_dataset, game_id="cross_test")

        bronze = _kloppy_dataset_to_metrica_df(metrica_dataset)
        home_team_id = metrica_dataset.metadata.teams[0].team_id
        actions_dedicated, _ = metrica_mod.convert_to_actions(bronze, home_team_id=home_team_id)

        assert len(actions_kloppy) > 0
        assert len(actions_dedicated) > 0
        assert list(actions_kloppy.columns) == list(actions_dedicated.columns)
