"""Sportec (DFL) DataFrame SPADL converter tests."""

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import sportec as sportec_mod
from silly_kicks.spadl.schema import KLOPPY_SPADL_COLUMNS

_KLOPPY_FIXTURES_DIR = Path(__file__).parent.parent / "datasets" / "kloppy"

_REQUIRED_COLS = [
    "match_id",
    "event_id",
    "event_type",
    "period",
    "timestamp_seconds",
    "player_id",
    "team",
    "x",
    "y",
]


def _df_minimal_pass() -> pd.DataFrame:
    """One-pass DataFrame for smoke-testing the Contract."""
    return pd.DataFrame(
        {
            "match_id": ["J03WMX"],
            "event_id": ["e1"],
            "event_type": ["Pass"],
            "period": [1],
            "timestamp_seconds": [10.5],
            "player_id": ["DFL-OBJ-0001"],
            "team": ["DFL-CLU-A"],
            "x": [50.0],
            "y": [34.0],
        }
    )


class TestSportecContract:
    """Contract: return shape, schema, dtypes, no input mutation."""

    def test_returns_tuple_dataframe_conversion_report(self):
        events = _df_minimal_pass()
        result = sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A")
        assert isinstance(result, tuple) and len(result) == 2
        actions, report = result
        assert isinstance(actions, pd.DataFrame)
        assert report.provider == "Sportec"

    def test_output_schema_matches_kloppy_spadl_columns(self):
        events = _df_minimal_pass()
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A")
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())

    def test_dtypes_match_schema(self):
        events = _df_minimal_pass()
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A")
        for col, expected in KLOPPY_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected, f"{col}: got {actions[col].dtype}, expected {expected}"

    def test_empty_input_returns_empty_actions_with_schema(self):
        events = pd.DataFrame({c: [] for c in _REQUIRED_COLS})
        actions, report = sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A")
        assert len(actions) == 0
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())
        assert report.total_events == 0
        assert report.total_actions == 0

    def test_input_dataframe_not_mutated(self):
        events = _df_minimal_pass()
        original_columns = list(events.columns)
        original_len = len(events)
        _, _ = sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A")
        assert list(events.columns) == original_columns
        assert len(events) == original_len


class TestSportecRequiredColumns:
    """Missing any required input column must raise ValueError with column names."""

    @pytest.mark.parametrize("missing", _REQUIRED_COLS)
    def test_missing_required_column_raises(self, missing):
        events = _df_minimal_pass().drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A")


def _df_pass_default() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["Pass"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [50.0],
            "y": [34.0],
        }
    )


def _df_pass_cross() -> pd.DataFrame:
    df = _df_pass_default()
    df["play_height"] = ["cross"]
    return df


def _df_pass_flat_cross() -> pd.DataFrame:
    df = _df_pass_default()
    df["play_flat_cross"] = [True]
    return df


def _df_pass_head() -> pd.DataFrame:
    df = _df_pass_default()
    df["play_height"] = ["head"]
    return df


def _df_freekick() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["FreeKick"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [50.0],
            "y": [34.0],
        }
    )


def _df_freekick_cross() -> pd.DataFrame:
    df = _df_freekick()
    df["freekick_execution_mode"] = ["cross"]
    return df


def _df_corner() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["Corner"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [105.0],
            "y": [0.0],
        }
    )


def _df_corner_crossed() -> pd.DataFrame:
    df = _df_corner()
    df["corner_target_area"] = ["box"]
    return df


def _df_throwin() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["ThrowIn"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [50.0],
            "y": [0.0],
        }
    )


def _df_goalkick() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["GoalKick"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [5.0],
            "y": [34.0],
        }
    )


class TestSportecActionMappingPassAndSetPieces:
    def test_pass_default_maps_to_pass_foot(self):
        actions, _ = sportec_mod.convert_to_actions(_df_pass_default(), home_team_id="T-HOME")
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]
        assert actions["bodypart_id"].iloc[0] == spadlconfig.bodypart_id["foot"]

    def test_pass_play_height_cross_maps_to_cross(self):
        actions, _ = sportec_mod.convert_to_actions(_df_pass_cross(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["cross"]

    def test_pass_play_flat_cross_maps_to_cross(self):
        actions, _ = sportec_mod.convert_to_actions(_df_pass_flat_cross(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["cross"]

    def test_pass_play_height_head_uses_head_bodypart(self):
        actions, _ = sportec_mod.convert_to_actions(_df_pass_head(), home_team_id="T-HOME")
        assert actions["bodypart_id"].iloc[0] == spadlconfig.bodypart_id["head"]

    def test_freekick_default_maps_to_freekick_short(self):
        actions, _ = sportec_mod.convert_to_actions(_df_freekick(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["freekick_short"]

    def test_freekick_with_cross_execution_maps_to_freekick_crossed(self):
        actions, _ = sportec_mod.convert_to_actions(_df_freekick_cross(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["freekick_crossed"]

    def test_corner_default_maps_to_corner_short(self):
        actions, _ = sportec_mod.convert_to_actions(_df_corner(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["corner_short"]

    def test_corner_with_box_target_maps_to_corner_crossed(self):
        actions, _ = sportec_mod.convert_to_actions(_df_corner_crossed(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["corner_crossed"]

    def test_throwin_maps_to_throw_in_other_bodypart(self):
        actions, _ = sportec_mod.convert_to_actions(_df_throwin(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["throw_in"]
        assert actions["bodypart_id"].iloc[0] == spadlconfig.bodypart_id["other"]

    def test_goalkick_maps_to_goalkick(self):
        actions, _ = sportec_mod.convert_to_actions(_df_goalkick(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["goalkick"]


def _df_shot_default() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["ShotAtGoal"],
            "period": [1],
            "timestamp_seconds": [60.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [95.0],
            "y": [34.0],
        }
    )


def _df_shot_after_freekick() -> pd.DataFrame:
    df = _df_shot_default()
    df["shot_after_free_kick"] = ["true"]
    return df


def _df_shot_penalty() -> pd.DataFrame:
    df = _df_shot_default()
    df["penalty_team"] = ["T-HOME"]
    df["penalty_causing_player"] = ["P-OPP"]
    return df


def _df_shot_goal() -> pd.DataFrame:
    df = _df_shot_default()
    df["shot_outcome_type"] = ["goal"]
    return df


def _df_tackle_winner() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["TacklingGame"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P-LOSER"],
            "team": ["T-AWAY"],
            "x": [50.0],
            "y": [34.0],
            "tackle_winner": ["P-WINNER"],
            "tackle_winner_team": ["T-HOME"],
            "tackle_loser": ["P-LOSER"],
            "tackle_loser_team": ["T-AWAY"],
        }
    )


def _df_foul() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["Foul"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [50.0],
            "y": [34.0],
        }
    )


def _df_foul_with_yellow_caution_paired() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1", "M1"],
            "event_id": ["e1", "e2"],
            "event_type": ["Foul", "Caution"],
            "period": [1, 1],
            "timestamp_seconds": [10.0, 11.5],
            "player_id": ["P-FOULER", "P-FOULER"],
            "team": ["T-HOME", "T-HOME"],
            "x": [50.0, 50.0],
            "y": [34.0, 34.0],
            "foul_fouler": ["P-FOULER", None],
            "caution_player": [None, "P-FOULER"],
            "caution_card_color": [None, "yellow"],
        }
    )


def _df_play_gk_save() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["Play"],
            "period": [1],
            "timestamp_seconds": [60.0],
            "player_id": ["P-GK"],
            "team": ["T-AWAY"],
            "x": [3.0],
            "y": [34.0],
            "play_goal_keeper_action": ["save"],
        }
    )


class TestSportecActionMappingShotsTacklesFoulsGK:
    def test_shot_default_maps_to_shot(self):
        actions, _ = sportec_mod.convert_to_actions(_df_shot_default(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["shot"]

    def test_shot_after_freekick_maps_to_shot_freekick(self):
        actions, _ = sportec_mod.convert_to_actions(_df_shot_after_freekick(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["shot_freekick"]

    def test_shot_penalty_maps_to_shot_penalty(self):
        actions, _ = sportec_mod.convert_to_actions(_df_shot_penalty(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["shot_penalty"]

    def test_shot_goal_outcome_maps_to_success_result(self):
        actions, _ = sportec_mod.convert_to_actions(_df_shot_goal(), home_team_id="T-HOME")
        assert actions["result_id"].iloc[0] == spadlconfig.result_id["success"]

    def test_tackle_uses_winner_as_actor(self):
        actions, _ = sportec_mod.convert_to_actions(_df_tackle_winner(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["tackle"]
        assert actions["player_id"].iloc[0] == "P-WINNER"
        assert actions["team_id"].iloc[0] == "T-HOME"

    def test_foul_default_maps_to_foul_fail(self):
        actions, _ = sportec_mod.convert_to_actions(_df_foul(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["foul"]
        assert actions["result_id"].iloc[0] == spadlconfig.result_id["fail"]

    def test_foul_with_paired_caution_upgrades_result(self):
        actions, _ = sportec_mod.convert_to_actions(_df_foul_with_yellow_caution_paired(), home_team_id="T-HOME")
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["foul"]
        assert actions["result_id"].iloc[0] == spadlconfig.result_id["yellow_card"]

    @pytest.mark.parametrize(
        "gk_action,expected_type",
        [
            ("save", "keeper_save"),
            ("claim", "keeper_claim"),
            ("punch", "keeper_punch"),
            ("pickUp", "keeper_pick_up"),
        ],
    )
    def test_play_gk_action_maps(self, gk_action, expected_type):
        df = _df_play_gk_save()
        df["play_goal_keeper_action"] = [gk_action]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id[expected_type]


class TestSportecDirectionOfPlay:
    """Away-team coords must be flipped: x -> 105-x, y -> 68-y."""

    def test_away_team_x_flipped(self):
        events = pd.DataFrame(
            {
                "match_id": ["M1", "M1"],
                "event_id": ["e1", "e2"],
                "event_type": ["Pass", "Pass"],
                "period": [1, 1],
                "timestamp_seconds": [10.0, 20.0],
                "player_id": ["P1", "P2"],
                "team": ["T-HOME", "T-AWAY"],
                "x": [30.0, 30.0],
                "y": [34.0, 34.0],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        home_row = actions[actions["team_id"] == "T-HOME"].iloc[0]
        away_row = actions[actions["team_id"] == "T-AWAY"].iloc[0]
        assert home_row["start_x"] == 30.0
        assert away_row["start_x"] == 75.0
        assert away_row["start_y"] == 34.0


class TestSportecCoordinateClamping:
    """Off-pitch coords must be clamped to [0, 105] x [0, 68], not dropped."""

    def test_negative_x_clamped_to_zero(self):
        events = _df_pass_default()
        events["x"] = [-1.5]
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        assert len(actions) == 1
        assert actions["start_x"].iloc[0] >= 0.0

    def test_oversized_y_clamped_to_pitch_width(self):
        events = _df_pass_default()
        events["y"] = [70.5]
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        assert actions["start_y"].iloc[0] <= 68.0


class TestSportecActionId:
    """action_id must be range(len(actions))."""

    def test_action_id_is_zero_indexed_range(self):
        events = pd.DataFrame(
            {
                "match_id": ["M1"] * 3,
                "event_id": ["e1", "e2", "e3"],
                "event_type": ["Pass", "Pass", "ShotAtGoal"],
                "period": [1, 1, 1],
                "timestamp_seconds": [10.0, 20.0, 30.0],
                "player_id": ["P1", "P2", "P3"],
                "team": ["T-HOME"] * 3,
                "x": [50.0, 60.0, 95.0],
                "y": [34.0, 34.0, 34.0],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        assert actions["action_id"].tolist() == list(range(len(actions)))


class TestSportecAddDribbles:
    """Synthetic dribbles inserted between same-team passes with positional gap."""

    def test_dribble_inserted_between_distant_same_team_passes(self):
        events = pd.DataFrame(
            {
                "match_id": ["M1", "M1"],
                "event_id": ["e1", "e2"],
                "event_type": ["Pass", "Pass"],
                "period": [1, 1],
                "timestamp_seconds": [10.0, 12.0],
                "player_id": ["P1", "P2"],
                "team": ["T-HOME"] * 2,
                "x": [30.0, 60.0],
                "y": [34.0, 34.0],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        type_names = (
            spadlconfig.actiontypes_df().set_index("type_id").loc[actions["type_id"].tolist(), "type_name"].tolist()
        )
        assert "dribble" in type_names


class TestSportecPreserveNative:
    """preserve_native surfaces caller-attached columns."""

    def test_preserve_native_passes_through_extra_column(self):
        events = _df_pass_default()
        events["my_custom_col"] = ["custom_value"]
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME", preserve_native=["my_custom_col"])
        assert "my_custom_col" in actions.columns
        assert actions["my_custom_col"].iloc[0] == "custom_value"

    def test_preserve_native_with_schema_overlap_raises(self):
        events = _df_pass_default()
        events["team_id"] = ["overlap"]
        with pytest.raises(ValueError, match=r"overlap|already"):
            sportec_mod.convert_to_actions(events, home_team_id="T-HOME", preserve_native=["team_id"])

    def test_preserve_native_missing_column_raises(self):
        events = _df_pass_default()
        with pytest.raises(ValueError, match="missing"):
            sportec_mod.convert_to_actions(events, home_team_id="T-HOME", preserve_native=["nonexistent_col"])


# ---------------------------------------------------------------------------
# Cross-path consistency: kloppy path vs dedicated DataFrame path
# ---------------------------------------------------------------------------

# Map kloppy's normalized EventType enum names back to DFL XML tag names so
# the bridge helper can produce a bronze-shaped DataFrame from a kloppy
# EventDataset, enabling apples-to-apples comparison of resulting SPADL.
_KLOPPY_EVENT_TYPE_TO_DFL_NAME = {
    "PASS": "Pass",
    "SHOT": "ShotAtGoal",
    "DUEL": "TacklingGame",
    "FOUL_COMMITTED": "Foul",
    "CARRY": "Play",
    "RECOVERY": "Recovery",
    "BALL_OUT": "BallOut",
    "CARD": "Caution",
    "GENERIC": "Generic",
    "GOALKEEPER": "Play",
    "INTERCEPTION": "Recovery",
    "TAKE_ON": "Play",
    "MISCONTROL": "Play",
    "CLEARANCE": "Play",
}


def _kloppy_dataset_to_sportec_bronze_df(dataset) -> pd.DataFrame:
    """Bridge a kloppy EventDataset to a Sportec bronze-shaped DataFrame."""
    rows = []
    for ev in dataset.events:
        et_name = ev.event_type.name if hasattr(ev.event_type, "name") else str(ev.event_type)
        dfl_name = _KLOPPY_EVENT_TYPE_TO_DFL_NAME.get(et_name, "Unknown")
        coords = ev.coordinates
        x = coords.x if coords is not None else 0.0
        y = coords.y if coords is not None else 0.0
        row = {
            "match_id": "fixture_match",
            "event_id": ev.event_id,
            "event_type": dfl_name,
            "period": ev.period.id,
            "timestamp_seconds": ev.timestamp.total_seconds(),
            "player_id": ev.player.player_id if ev.player else None,
            "team": ev.team.team_id if ev.team else None,
            "x": x,
            "y": y,
        }
        if isinstance(ev.raw_event, dict):
            for k, v in ev.raw_event.items():
                col = k.lower().replace("-", "_")
                row[col] = v
        rows.append(row)
    return pd.DataFrame(rows)


class TestSportecCrossPathConsistency:
    """V1 cross-path proof: kloppy path and dedicated DataFrame path produce
    non-empty SPADL DataFrames with the same column schema. Stricter
    row-equivalence (per spec §7.3) is deferred to a follow-up — the bridge
    helper is heuristic (e.g., kloppy CARRY -> DFL Play is approximate)."""

    def test_kloppy_and_dedicated_paths_produce_equivalent_shape(self, sportec_dataset):
        from silly_kicks.spadl import kloppy as kloppy_mod

        actions_kloppy, _ = kloppy_mod.convert_to_actions(sportec_dataset, game_id="cross_test")

        bronze = _kloppy_dataset_to_sportec_bronze_df(sportec_dataset)
        home_team_id = sportec_dataset.metadata.teams[0].team_id
        actions_dedicated, _ = sportec_mod.convert_to_actions(bronze, home_team_id=home_team_id)

        assert len(actions_kloppy) > 0
        assert len(actions_dedicated) > 0
        assert list(actions_kloppy.columns) == list(actions_dedicated.columns)
