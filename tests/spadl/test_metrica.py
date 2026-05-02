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
        actions, report = metrica_mod.convert_to_actions(
            _df_minimal_pass(), home_team_id="Home", home_team_start_left=True
        )
        assert isinstance(actions, pd.DataFrame)
        assert report.provider == "Metrica"

    def test_output_schema_matches_kloppy_spadl_columns(self):
        actions, _ = metrica_mod.convert_to_actions(_df_minimal_pass(), home_team_id="Home", home_team_start_left=True)
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())

    def test_dtypes_match_schema(self):
        actions, _ = metrica_mod.convert_to_actions(_df_minimal_pass(), home_team_id="Home", home_team_start_left=True)
        for col, expected in KLOPPY_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected

    def test_empty_input_returns_empty_actions_with_schema(self):
        events = pd.DataFrame({c: [] for c in _REQUIRED_COLS})
        actions, _report = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
        assert len(actions) == 0
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())

    def test_input_dataframe_not_mutated(self):
        events = _df_minimal_pass()
        original_columns = list(events.columns)
        original_len = len(events)
        _, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
        assert list(events.columns) == original_columns
        assert len(events) == original_len


class TestMetricaRequiredColumns:
    @pytest.mark.parametrize("missing", _REQUIRED_COLS)
    def test_missing_required_column_raises(self, missing):
        events = _df_minimal_pass().drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)


class TestMetricaActionMapping:
    def test_pass_default(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica("PASS"), home_team_id="Home", home_team_start_left=True)
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_pass_cross(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("PASS", "CROSS"), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["cross"]

    def test_pass_goalkick(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("PASS", "GOAL KICK"), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["goalkick"]

    def test_pass_head_bodypart(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("PASS", "HEAD"), home_team_id="Home", home_team_start_left=True
        )
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
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("SHOT", subtype), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["shot"]
        assert actions["result_id"].iloc[0] == spadlconfig.result_id[expected_result]

    def test_recovery_maps_to_interception(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("RECOVERY"), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["interception"]

    def test_challenge_won_maps_to_tackle(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("CHALLENGE", "WON"), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["tackle"]

    def test_challenge_lost_dropped(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("CHALLENGE", "LOST"), home_team_id="Home", home_team_start_left=True
        )
        assert len(actions) == 0

    def test_ball_lost_maps_to_bad_touch_fail(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("BALL LOST"), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["bad_touch"]
        assert actions["result_id"].iloc[0] == spadlconfig.result_id["fail"]

    def test_fault_maps_to_foul_fail(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("FAULT"), home_team_id="Home", home_team_start_left=True
        )
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
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("SET PIECE", subtype), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id[expected_type]

    def test_set_piece_kickoff_dropped(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("SET PIECE", "KICK OFF"), home_team_id="Home", home_team_start_left=True
        )
        assert len(actions) == 0

    def test_excluded_types_dropped(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("BALL OUT"), home_team_id="Home", home_team_start_left=True
        )
        assert len(actions) == 0
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("SUBSTITUTION"), home_team_id="Home", home_team_start_left=True
        )
        assert len(actions) == 0
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica("FAULT RECEIVED"), home_team_id="Home", home_team_start_left=True
        )
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
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
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
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
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
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
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
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
        type_set = set(actions["type_id"].tolist())
        assert spadlconfig.actiontype_id["corner_short"] in type_set
        assert spadlconfig.actiontype_id["shot"] in type_set


class TestMetricaCoordinateClamping:
    def test_negative_start_x_clamped(self):
        events = _df_minimal_pass()
        events["start_x"] = [-1.5]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
        assert actions["start_x"].iloc[0] >= 0

    def test_oversized_start_y_clamped(self):
        events = _df_minimal_pass()
        events["start_y"] = [70.0]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
        assert actions["start_y"].iloc[0] <= 68


class TestMetricaDirectionOfPlay:
    def test_away_team_x_flipped(self):
        events = _df_minimal_pass()
        events["team"] = ["Away"]
        events["start_x"] = [30.0]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
        assert actions["start_x"].iloc[0] == 75.0


class TestMetricaPreserveNative:
    def test_preserve_native_passes_through(self):
        events = _df_minimal_pass()
        events["my_extra"] = ["foo"]
        actions, _ = metrica_mod.convert_to_actions(
            events, home_team_id="Home", preserve_native=["my_extra"], home_team_start_left=True
        )
        assert actions["my_extra"].iloc[0] == "foo"

    def test_preserve_native_schema_overlap_raises(self):
        events = _df_minimal_pass()
        events["team_id"] = ["overlap"]
        with pytest.raises(ValueError, match=r"overlap|already"):
            metrica_mod.convert_to_actions(
                events, home_team_id="Home", preserve_native=["team_id"], home_team_start_left=True
            )


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
        actions_dedicated, _ = metrica_mod.convert_to_actions(
            bronze, home_team_id=home_team_id, home_team_start_left=True
        )

        assert len(actions_kloppy) > 0
        assert len(actions_dedicated) > 0
        assert list(actions_kloppy.columns) == list(actions_dedicated.columns)


# ---------------------------------------------------------------------------
# Bug 3 — Metrica lacks native GK markers; goalkeeper_ids enables coverage
# (1.10.0; supersedes pre-1.10.0 where Metrica emitted zero keeper_* actions
# and the lakehouse adapter had no way to surface GK actions)
# ---------------------------------------------------------------------------


def _df_metrica_pass_by_gk() -> pd.DataFrame:
    df = _df_minimal_pass()
    df["player"] = ["GK_HOME"]
    df["team"] = ["Home"]
    return df


def _df_metrica_recovery_by_gk() -> pd.DataFrame:
    df = _df_metrica("RECOVERY", None)
    df["player"] = ["GK_HOME"]
    return df


def _df_metrica_aerial_won_by_gk() -> pd.DataFrame:
    df = _df_metrica("CHALLENGE", "AERIAL-WON")
    df["player"] = ["GK_HOME"]
    return df


def _df_metrica_aerial_lost_by_gk() -> pd.DataFrame:
    df = _df_metrica("CHALLENGE", "AERIAL-LOST")
    df["player"] = ["GK_HOME"]
    return df


class TestMetricaGoalkeeperIdsRouting:
    """Without goalkeeper_ids: zero keeper_* actions (1.9.0 default preserved).
    With goalkeeper_ids: PASS by GK → synth, RECOVERY by GK → keeper_pick_up,
    CHALLENGE-AERIAL-WON by GK → keeper_claim. Other events unchanged.
    """

    def test_no_goalkeeper_ids_pass_remains_pass(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_pass_by_gk(), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_no_goalkeeper_ids_recovery_remains_interception(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_recovery_by_gk(), home_team_id="Home", home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["interception"]

    def test_no_goalkeeper_ids_emits_zero_keeper_actions(self):
        # Bundle 4 events, all by GK_HOME. Without goalkeeper_ids,
        # output has zero keeper_* actions — preserves 1.9.0 behavior.
        events = pd.concat(
            [
                _df_metrica_pass_by_gk(),
                _df_metrica_recovery_by_gk(),
                _df_metrica_aerial_won_by_gk(),
                _df_metrica_aerial_lost_by_gk(),
            ],
            ignore_index=True,
        )
        events["event_id"] = list(range(len(events)))
        events["start_time_s"] = [10.0, 20.0, 30.0, 40.0]
        events["end_time_s"] = [10.5, 20.5, 30.5, 40.5]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
        keeper_ids = {
            spadlconfig.actiontype_id[t] for t in ("keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up")
        }
        keeper_count = actions["type_id"].isin(keeper_ids).sum()
        assert keeper_count == 0

    def test_pass_by_gk_synthesizes_two_actions(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_pass_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
        )
        assert len(actions) == 2
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]
        assert actions["type_id"].iloc[1] == spadlconfig.actiontype_id["pass"]

    def test_recovery_by_gk_maps_to_keeper_pick_up(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_recovery_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
        )
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]

    def test_aerial_won_by_gk_maps_to_keeper_claim(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_aerial_won_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
        )
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_claim"]

    def test_aerial_lost_by_gk_unchanged(self):
        # AERIAL-LOST → CHALLENGE not WON → dropped by default Metrica dispatch.
        # Adding goalkeeper_ids does NOT promote it to keeper_claim.
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_aerial_lost_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
        )
        assert len(actions) == 0

    def test_pass_by_non_gk_player_unchanged_with_goalkeeper_ids(self):
        df = _df_minimal_pass()
        df["player"] = ["NOT_GK"]
        actions, _ = metrica_mod.convert_to_actions(
            df, home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
        )
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_empty_goalkeeper_ids_set_equivalent_to_none(self):
        df = _df_metrica_pass_by_gk()
        actions_empty, _ = metrica_mod.convert_to_actions(
            df, home_team_id="Home", goalkeeper_ids=set(), home_team_start_left=True
        )
        actions_none, _ = metrica_mod.convert_to_actions(
            df, home_team_id="Home", goalkeeper_ids=None, home_team_start_left=True
        )
        assert len(actions_empty) == len(actions_none) == 1

    def test_set_piece_freekick_by_gk_unchanged(self):
        df = _df_metrica("SET PIECE", "FREE KICK")
        df["player"] = ["GK_HOME"]
        actions, _ = metrica_mod.convert_to_actions(
            df, home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["freekick_short"]

    def test_pass_by_gk_synthetic_pass_has_other_bodypart(self):
        # The synthesized pass uses bodypart=other (matching sportec throwOut shape).
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_pass_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}, home_team_start_left=True
        )
        # Action 1 (the synth pass) — bodypart = other
        assert actions["bodypart_id"].iloc[1] == spadlconfig.bodypart_id["other"]

    def test_preserve_native_propagates_to_synthesized_actions(self):
        df = _df_metrica_pass_by_gk()
        df["my_extra"] = ["custom"]
        actions, _ = metrica_mod.convert_to_actions(
            df,
            home_team_id="Home",
            goalkeeper_ids={"GK_HOME"},
            preserve_native=["my_extra"],
            home_team_start_left=True,
        )
        assert actions["my_extra"].iloc[0] == "custom"
        assert actions["my_extra"].iloc[1] == "custom"


class TestMetricaPerPeriodKwargContract:
    """PR-S23 / silly-kicks 3.0.1: per-period direction-of-play kwarg semantics."""

    def test_raises_when_no_per_period_kwarg_supplied(self):
        events = _df_minimal_pass()
        with pytest.raises(ValueError, match=r"3\.0\.1.*home_team_start_left"):
            metrica_mod.convert_to_actions(events, home_team_id="Home")

    def test_raises_when_both_kwargs_supplied(self):
        events = _df_minimal_pass()
        with pytest.raises(ValueError, match="not both"):
            metrica_mod.convert_to_actions(
                events,
                home_team_id="Home",
                home_team_start_left=True,
                home_attacks_right_per_period={1: True, 2: False},
            )

    def test_accepts_explicit_mapping_path(self):
        events = _df_minimal_pass()
        actions, _ = metrica_mod.convert_to_actions(
            events,
            home_team_id="Home",
            home_attacks_right_per_period={1: True, 2: False},
        )
        assert len(actions) >= 1

    def test_raises_when_extratime_supplied_without_start_left(self):
        events = _df_minimal_pass()
        with pytest.raises(ValueError, match="_extratime supplied without"):
            metrica_mod.convert_to_actions(
                events,
                home_team_id="Home",
                home_team_start_left_extratime=True,
            )

    def test_raises_when_et_periods_present_without_extratime(self):
        events = _df_minimal_pass().copy()
        events["period"] = [3]  # ET period
        with pytest.raises(ValueError, match="ET periods"):
            metrica_mod.convert_to_actions(
                events,
                home_team_id="Home",
                home_team_start_left=True,
            )
