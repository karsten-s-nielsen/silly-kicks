"""PFF FC DataFrame SPADL converter tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import pff as pff_mod
from silly_kicks.spadl.schema import PFF_SPADL_COLUMNS

# Minimum set of input columns to construct a one-row test DataFrame.
# Mirrors the EXPECTED_INPUT_COLUMNS frozenset in pff.py.
_REQUIRED_COLS = sorted(pff_mod.EXPECTED_INPUT_COLUMNS)

_SYNTHETIC_FIXTURE = Path(__file__).parent.parent / "datasets" / "pff" / "synthetic_match.json"


def _load_synthetic_events() -> pd.DataFrame:
    """Load the synthetic match JSON and flatten into the EXPECTED_INPUT_COLUMNS shape."""
    with _SYNTHETIC_FIXTURE.open("r", encoding="utf-8") as f:
        events_json = json.load(f)

    rows = []
    for ev in events_json:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        # Real PFF data carries `fouls` as a single dict per event (not a list).
        f0 = ev.get("fouls") or {}
        ball = (ev.get("ball") or [{}])[0] if ev.get("ball") else {}

        rows.append(
            {
                "game_id": ev["gameId"],
                "event_id": ev["gameEventId"],
                "possession_event_id": ev.get("possessionEventId"),
                "period_id": ge.get("period"),
                "time_seconds": ge.get("startGameClock") or 0.0,
                "team_id": ge.get("teamId"),
                "player_id": ge.get("playerId"),
                "game_event_type": ge.get("gameEventType"),
                "possession_event_type": pe.get("possessionEventType"),
                "set_piece_type": ge.get("setpieceType"),
                "ball_x": ball.get("x"),
                "ball_y": ball.get("y"),
                "body_type": pe.get("bodyType"),
                "ball_height_type": pe.get("ballHeightType"),
                "pass_outcome_type": pe.get("passOutcomeType"),
                "pass_type": pe.get("passType"),
                "incompletion_reason_type": pe.get("incompletionReasonType"),
                "cross_outcome_type": pe.get("crossOutcomeType"),
                "cross_type": pe.get("crossType"),
                "cross_zone_type": pe.get("crossZoneType"),
                "shot_outcome_type": pe.get("shotOutcomeType"),
                "shot_type": pe.get("shotType"),
                "shot_nature_type": pe.get("shotNatureType"),
                "shot_initial_height_type": pe.get("shotInitialHeightType"),
                "save_height_type": pe.get("saveHeightType"),
                "save_rebound_type": pe.get("saveReboundType"),
                "carry_type": pe.get("carryType"),
                "ball_carry_outcome": pe.get("ballCarryOutcome"),
                "carry_intent": pe.get("carryIntent"),
                "carry_defender_player_id": pe.get("carryDefenderPlayerId"),
                "challenge_type": pe.get("challengeType"),
                "challenge_outcome_type": pe.get("challengeOutcomeType"),
                "challenger_player_id": pe.get("challengerPlayerId"),
                "challenger_team_id": None,
                "challenge_winner_player_id": pe.get("challengeWinnerPlayerId"),
                "challenge_winner_team_id": None,
                "tackle_attempt_type": pe.get("tackleAttemptType"),
                "clearance_outcome_type": pe.get("clearanceOutcomeType"),
                "rebound_outcome_type": pe.get("reboundOutcomeType"),
                "keeper_touch_type": pe.get("keeperTouchType"),
                "touch_outcome_type": pe.get("touchOutcomeType"),
                "touch_type": pe.get("touchType"),
                "foul_type": f0.get("foulType"),
                "on_field_offense_type": f0.get("onFieldOffenseType"),
                "final_offense_type": f0.get("finalOffenseType"),
                "on_field_foul_outcome_type": f0.get("onFieldFoulOutcomeType"),
                "final_foul_outcome_type": f0.get("finalFoulOutcomeType"),
            }
        )
    df = pd.DataFrame(rows)

    # Roster join — synthetic teams: ids 1-11 = team 100, 12-22 = team 200.
    def _team_for(pid):
        if pid is None or pd.isna(pid):
            return pd.NA
        pid_int = int(pid)
        return 100 if 1 <= pid_int <= 11 else 200

    df["challenger_team_id"] = df["challenger_player_id"].map(_team_for)
    df["challenge_winner_team_id"] = df["challenge_winner_player_id"].map(_team_for)

    # Cast nullable Int64 columns the converter expects.
    for col in (
        "possession_event_id",
        "player_id",
        "carry_defender_player_id",
        "challenger_player_id",
        "challenger_team_id",
        "challenge_winner_player_id",
        "challenge_winner_team_id",
    ):
        df[col] = df[col].astype("Int64")
    df["game_id"] = df["game_id"].astype("int64")
    df["event_id"] = df["event_id"].astype("int64")
    df["period_id"] = df["period_id"].astype("int64")
    df["team_id"] = df["team_id"].astype("int64")
    df["time_seconds"] = df["time_seconds"].astype("float64")
    df["ball_x"] = df["ball_x"].astype("float64")
    df["ball_y"] = df["ball_y"].astype("float64")
    return df


def _df_minimal_pass() -> pd.DataFrame:
    """One-row open-play pass DataFrame; player 1 (home team 100) to player 2."""
    base = {col: [None] for col in _REQUIRED_COLS}
    overrides = {
        "game_id": [10502],
        "event_id": [1],
        "possession_event_id": [1],
        "period_id": [1],
        "time_seconds": [10.5],
        "team_id": [100],
        "player_id": [1],
        "game_event_type": ["OTB"],
        "possession_event_type": ["PA"],
        "set_piece_type": ["O"],
        "ball_x": [0.0],
        "ball_y": [0.0],
        "pass_outcome_type": ["C"],
        "body_type": ["R"],
    }
    base.update(overrides)
    df = pd.DataFrame(base)
    # Cast nullable-Int64 columns explicitly (the converter expects them).
    for col in (
        "possession_event_id",
        "player_id",
        "carry_defender_player_id",
        "challenger_player_id",
        "challenger_team_id",
        "challenge_winner_player_id",
        "challenge_winner_team_id",
    ):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


class TestPffContract:
    """Contract: return shape, schema, dtypes, no input mutation."""

    def test_returns_tuple_dataframe_conversion_report(self):
        events = _df_minimal_pass()
        result = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        actions, report = result
        assert isinstance(actions, pd.DataFrame)
        assert report.provider == "PFF"

    def test_output_schema_matches_pff_spadl_columns(self):
        events = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert list(actions.columns) == list(PFF_SPADL_COLUMNS.keys())

    def test_dtypes_match_schema(self):
        events = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        for col, expected in PFF_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected, f"{col}: got {actions[col].dtype}, expected {expected}"

    def test_empty_input_returns_empty_actions_with_schema(self):
        empty = pd.DataFrame({c: [] for c in _REQUIRED_COLS})
        actions, report = pff_mod.convert_to_actions(
            empty,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 0
        assert list(actions.columns) == list(PFF_SPADL_COLUMNS.keys())
        assert report.total_events == 0
        assert report.total_actions == 0

    def test_input_dataframe_not_mutated(self):
        events = _df_minimal_pass()
        original_columns = list(events.columns)
        original_len = len(events)
        _, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert list(events.columns) == original_columns
        assert len(events) == original_len


class TestPffRequiredColumns:
    """Missing any required input column must raise ValueError with column names."""

    @pytest.mark.parametrize("missing", _REQUIRED_COLS)
    def test_missing_required_column_raises(self, missing):
        events = _df_minimal_pass().drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            pff_mod.convert_to_actions(
                events,
                home_team_id=100,
                home_team_start_left=True,
            )


class TestPffCoordinateTranslation:
    """PFF centered meters → SPADL bottom-left meters."""

    def test_center_spot_translates_to_pitch_center(self):
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 0.0
        df.loc[0, "ball_y"] = 0.0
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # Home team in period 1 (start left=True) attacks right, no flip.
        # SPADL center: (52.5, 34.0).
        assert actions.iloc[0]["start_x"] == pytest.approx(52.5)
        assert actions.iloc[0]["start_y"] == pytest.approx(34.0)

    def test_corner_translates_to_pitch_corner(self):
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = -52.5  # PFF centered: left-side corner
        df.loc[0, "ball_y"] = -34.0
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # Home team period 1, no flip → SPADL (0, 0).
        assert actions.iloc[0]["start_x"] == pytest.approx(0.0)
        assert actions.iloc[0]["start_y"] == pytest.approx(0.0)


class TestPffDirectionOfPlay:
    """All teams attack left-to-right after conversion (per-period flip)."""

    def test_home_period1_no_flip(self):
        """Home team, period 1, home_team_start_left=True → no flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "ball_y"] = 0.0
        df.loc[0, "team_id"] = 100  # home
        df.loc[0, "period_id"] = 1
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # SPADL bottom-left: (26.25 + 52.5, 34) = (78.75, 34). No flip.
        assert actions.iloc[0]["start_x"] == pytest.approx(78.75)

    def test_away_period1_flips(self):
        """Away team, period 1, home_team_start_left=True → away attacks left, flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "ball_y"] = 0.0
        df.loc[0, "team_id"] = 200  # away
        df.loc[0, "period_id"] = 1
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # Pre-flip SPADL: 78.75. Away in P1 flips → 105 - 78.75 = 26.25.
        assert actions.iloc[0]["start_x"] == pytest.approx(26.25)

    def test_home_period2_flips(self):
        """Home team, period 2 → home attacks left in P2, flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "team_id"] = 100  # home
        df.loc[0, "period_id"] = 2
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["start_x"] == pytest.approx(26.25)

    def test_away_period2_no_flip(self):
        """Away team, period 2 → away attacks right in P2, no flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "team_id"] = 200  # away
        df.loc[0, "period_id"] = 2
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["start_x"] == pytest.approx(78.75)


class TestPffExtraTimeFallback:
    """ET data without explicit ET-direction param raises ValueError."""

    def test_period3_event_without_extratime_param_raises(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 3
        with pytest.raises(ValueError, match="home_team_start_left_extratime"):
            pff_mod.convert_to_actions(
                df,
                home_team_id=100,
                home_team_start_left=True,
            )

    def test_period4_event_without_extratime_param_raises(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 4
        with pytest.raises(ValueError, match="home_team_start_left_extratime"):
            pff_mod.convert_to_actions(
                df,
                home_team_id=100,
                home_team_start_left=True,
            )

    def test_period3_event_with_extratime_param_succeeds(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 3
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 1


class TestPffBodyPart:
    """body_type → SPADL bodypart_id mapping."""

    @pytest.mark.parametrize(
        "body_type, expected_name",
        [
            ("L", "foot_left"),
            ("R", "foot_right"),
            ("H", "head"),
            ("O", "other"),
            (None, "foot"),
        ],
    )
    def test_body_type_dispatch(self, body_type, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "body_type"] = body_type
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        expected_id = spadlconfig.bodypart_id[expected_name]
        assert actions.iloc[0]["bodypart_id"] == expected_id


class TestPffPassDispatch:
    """OTB+PA dispatched by set_piece_type."""

    @pytest.mark.parametrize(
        "set_piece, expected_name",
        [
            ("O", "pass"),
            ("K", "pass"),
            ("F", "freekick_short"),
            ("C", "corner_short"),
            ("T", "throw_in"),
            ("G", "goalkick"),
        ],
    )
    def test_pass_set_piece_composition(self, set_piece, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "set_piece_type"] = set_piece
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        expected_id = spadlconfig.actiontype_id[expected_name]
        assert actions.iloc[0]["type_id"] == expected_id

    def test_pass_outcome_complete_is_success(self):
        df = _df_minimal_pass()
        df.loc[0, "pass_outcome_type"] = "C"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["success"]

    def test_pass_outcome_fail_is_fail(self):
        df = _df_minimal_pass()
        df.loc[0, "pass_outcome_type"] = "F"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["fail"]


class TestPffCrossDispatch:
    """OTB+CR dispatched by set_piece_type."""

    @pytest.mark.parametrize(
        "set_piece, expected_name",
        [
            ("O", "cross"),
            ("F", "freekick_crossed"),
            ("C", "corner_crossed"),
        ],
    )
    def test_cross_set_piece_composition(self, set_piece, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "set_piece_type"] = set_piece
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        expected_id = spadlconfig.actiontype_id[expected_name]
        assert actions.iloc[0]["type_id"] == expected_id

    def test_cross_outcome_uses_cross_outcome_type(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "cross_outcome_type"] = "C"
        df.loc[0, "pass_outcome_type"] = None
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["success"]


class TestPffShotDispatch:
    """OTB+SH dispatched by set_piece_type, results from shot_outcome_type."""

    @pytest.mark.parametrize(
        "set_piece, expected_name",
        [
            ("O", "shot"),
            ("F", "shot_freekick"),
            ("P", "shot_penalty"),
        ],
    )
    def test_shot_set_piece_composition(self, set_piece, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "SH"
        df.loc[0, "set_piece_type"] = set_piece
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id[expected_name]

    @pytest.mark.parametrize(
        "shot_outcome, expected_result",
        [
            ("G", "success"),
            ("O", "owngoal"),
            ("S", "fail"),
            ("B", "fail"),
            ("W", "fail"),
            ("M", "fail"),
            (None, "fail"),
        ],
    )
    def test_shot_result_mapping(self, shot_outcome, expected_result):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "SH"
        df.loc[0, "shot_outcome_type"] = shot_outcome
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id[expected_result]


class TestPffRebound:
    """RE events disambiguate by keeper_touch_type → keeper_save / keeper_pick_up."""

    def test_rebound_default_is_keeper_save(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "keeper_touch_type"] = None
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]

    def test_rebound_catch_class_is_keeper_pick_up(self):
        """Catch-class keeper_touch_type → keeper_pick_up.

        NOTE: The exact PFF keeper_touch_type code letters are not enumerated
        by the spec; the test uses "C" as a placeholder catch code matching
        the catch_class set in pff.py. If the synthetic match generator
        (Task 19) authors a different vocabulary, update this code AND the
        catch_class set in pff.py simultaneously.
        """
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "keeper_touch_type"] = "C"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_pick_up"]


class TestPffTackle:
    """OTB+CH → SPADL tackle, with winner/loser passthrough columns."""

    def _df_tackle(self, winner_id, winner_team_id):
        """Carrier (player 1, team 100) is challenged by player 5 (team 200)."""
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CH"
        df.loc[0, "challenger_player_id"] = 5
        df.loc[0, "challenger_team_id"] = 200
        df.loc[0, "challenge_winner_player_id"] = winner_id
        df.loc[0, "challenge_winner_team_id"] = winner_team_id
        df["challenger_player_id"] = df["challenger_player_id"].astype("Int64")
        df["challenger_team_id"] = df["challenger_team_id"].astype("Int64")
        df["challenge_winner_player_id"] = df["challenge_winner_player_id"].astype("Int64")
        df["challenge_winner_team_id"] = df["challenge_winner_team_id"].astype("Int64")
        return df

    def test_tackle_type_id_set(self):
        df = self._df_tackle(winner_id=5, winner_team_id=200)
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]

    def test_tackle_winner_columns_populated_when_challenger_wins(self):
        """Challenger (5/200) wins → carrier (1/100) lost."""
        df = self._df_tackle(winner_id=5, winner_team_id=200)
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["tackle_winner_player_id"] == 5
        assert actions.iloc[0]["tackle_winner_team_id"] == 200
        assert actions.iloc[0]["tackle_loser_player_id"] == 1
        assert actions.iloc[0]["tackle_loser_team_id"] == 100

    def test_tackle_winner_columns_populated_when_carrier_holds(self):
        """Carrier (1/100) wins (== event_player_id) → challenger (5/200) lost."""
        df = self._df_tackle(winner_id=1, winner_team_id=100)
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["tackle_winner_player_id"] == 1
        assert actions.iloc[0]["tackle_winner_team_id"] == 100
        assert actions.iloc[0]["tackle_loser_player_id"] == 5
        assert actions.iloc[0]["tackle_loser_team_id"] == 200

    def test_tackle_passthrough_NaN_on_non_tackle_rows(self):
        """A pass row has NA on all four tackle columns."""
        df = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        for col in (
            "tackle_winner_player_id",
            "tackle_winner_team_id",
            "tackle_loser_player_id",
            "tackle_loser_team_id",
        ):
            assert pd.isna(actions.iloc[0][col]), f"{col} should be NA on a pass row"


class TestPffClearanceDribbleTouchControl:
    """OTB+CL → clearance, OTB+BC → dribble, OTB+TC → bad_touch."""

    def test_clearance(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CL"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["clearance"]

    def test_ball_carry(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "BC"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["dribble"]

    def test_touch_control(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "TC"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["bad_touch"]


class TestPffFoul:
    """Rows with foul_type non-null synthesize an extra SPADL foul action."""

    def test_foul_synthesizes_additional_action(self):
        df = _df_minimal_pass()
        df.loc[0, "foul_type"] = "STANDARD"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 2
        assert actions["type_id"].tolist() == [
            spadlconfig.actiontype_id["pass"],
            spadlconfig.actiontype_id["foul"],
        ]

    def test_foul_yellow_card(self):
        df = _df_minimal_pass()
        df.loc[0, "foul_type"] = "STANDARD"
        df.loc[0, "final_foul_outcome_type"] = "Y"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        foul_row = actions[actions["type_id"] == spadlconfig.actiontype_id["foul"]].iloc[0]
        assert foul_row["result_id"] == spadlconfig.result_id["yellow_card"]

    def test_foul_red_card(self):
        df = _df_minimal_pass()
        df.loc[0, "foul_type"] = "STANDARD"
        df.loc[0, "final_foul_outcome_type"] = "R"
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        foul_row = actions[actions["type_id"] == spadlconfig.actiontype_id["foul"]].iloc[0]
        assert foul_row["result_id"] == spadlconfig.result_id["red_card"]

    def test_no_foul_no_synthesis(self):
        df = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 1


class TestPffExclusions:
    """Structural / metadata events with no SPADL counterpart are excluded."""

    @pytest.mark.parametrize(
        "ge_type",
        [
            "OUT",
            "SUB",
            "FIRSTKICKOFF",
            "SECONDKICKOFF",
            "THIRDKICKOFF",
            "FOURTHKICKOFF",
            "END",
            "OFF",
            "ON",
            "G",
        ],
    )
    def test_excluded_game_event_types_drop_out(self, ge_type):
        df = _df_minimal_pass()
        df.loc[0, "game_event_type"] = ge_type
        df.loc[0, "possession_event_type"] = None
        actions, report = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 0
        assert report.excluded_counts.get(ge_type) == 1

    def test_otb_plus_it_excluded(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "IT"
        actions, report = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 0
        assert report.excluded_counts.get("OTB+IT") == 1

    def test_otb_plus_empty_pe_excluded(self):
        """OTB rows with empty possessionEventType are initialNonEvent markers."""
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = None
        actions, report = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 0
        assert report.excluded_counts.get("OTB+") == 1


class TestPffDedicatedFoulEvent:
    """Standalone FOUL gameEventType with possessionEventType='FO' converts
    in-place to a SPADL foul action (no phantom non_action row)."""

    def test_foul_event_in_place_conversion(self):
        df = _df_minimal_pass()
        df.loc[0, "game_event_type"] = "FOUL"
        df.loc[0, "possession_event_type"] = "FO"
        df.loc[0, "foul_type"] = "I"
        df.loc[0, "final_foul_outcome_type"] = "Y"
        actions, report = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # Exactly ONE row: the foul (no phantom non_action parent).
        assert len(actions) == 1
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["foul"]
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["yellow_card"]
        # NOT in unrecognized — handled correctly.
        assert "FOUL+FO" not in report.unrecognized_counts


class TestPffReportCounts:
    """ConversionReport.mapped_counts uses SPADL action-type names."""

    def test_mapped_counts_uses_spadl_names(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df.loc[1, "possession_event_type"] = "SH"
        df.loc[1, "shot_outcome_type"] = "G"
        df.loc[1, "time_seconds"] = 11.0
        _, report = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert report.mapped_counts.get("pass") == 1
        assert report.mapped_counts.get("shot") == 1


class TestPffEndCoordinates:
    """end_x/end_y of each action equals start_x/start_y of the next action
    in the same period (chained-event semantics)."""

    def test_pass_end_is_next_start(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df.loc[1, "ball_x"] = 20.0
        df.loc[1, "ball_y"] = 5.0
        df.loc[1, "time_seconds"] = 11.0
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["end_x"] == pytest.approx(actions.iloc[1]["start_x"])
        assert actions.iloc[0]["end_y"] == pytest.approx(actions.iloc[1]["start_y"])

    def test_last_action_end_equals_start(self):
        """Last action has no successor — end falls back to its own start."""
        df = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[-1]["end_x"] == pytest.approx(actions.iloc[-1]["start_x"])
        assert actions.iloc[-1]["end_y"] == pytest.approx(actions.iloc[-1]["start_y"])


class TestPffSyntheticMatchE2E:
    """End-to-end conversion against the committed synthetic match fixture."""

    def test_synthetic_match_converts_with_no_unrecognized(self):
        events = _load_synthetic_events()
        _, report = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert report.has_unrecognized is False, f"Unexpected unrecognized vocabulary: {report.unrecognized_counts}"
        assert report.total_actions > 20

    def test_synthetic_match_dispatch_coverage(self):
        """Every documented dispatch row produces at least one action."""
        events = _load_synthetic_events()
        _, report = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        expected_action_types = {
            "pass",
            "freekick_short",
            "corner_short",
            "throw_in",
            "goalkick",
            "cross",
            "freekick_crossed",
            "corner_crossed",
            "shot",
            "shot_freekick",
            "shot_penalty",
            "clearance",
            "dribble",
            "tackle",
            "bad_touch",
            "keeper_save",
            "keeper_pick_up",
            "foul",
        }
        produced = set(report.mapped_counts.keys())
        missing = expected_action_types - produced
        assert not missing, f"Synthetic match missing dispatch coverage: {missing}"

    def test_synthetic_match_excluded_counts_non_trivial(self):
        """The synthetic match exercises every excluded vocabulary category
        empirically validated against the WC 2022 dataset (12 game_event_types
        + 2 OTB-pair patterns). Asserting all of them here locks the parity
        between synthetic-fixture coverage and the converter's excluded set.
        """
        events = _load_synthetic_events()
        _, report = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        for key in (
            "OUT",
            "SUB",
            "FIRSTKICKOFF",
            "SECONDKICKOFF",
            "THIRDKICKOFF",
            "FOURTHKICKOFF",
            "END",
            "OFF",
            "ON",
            "G",
            "OTB+IT",
            "OTB+",
        ):
            assert report.excluded_counts.get(key) == 1, (
                f"Synthetic match expected exactly 1 excluded {key!r} event, got {report.excluded_counts.get(key)}"
            )

    def test_synthetic_match_yields_goal_actions(self):
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        is_shot = actions["type_id"].isin(
            [
                spadlconfig.actiontype_id["shot"],
                spadlconfig.actiontype_id["shot_penalty"],
                spadlconfig.actiontype_id["shot_freekick"],
            ]
        )
        is_goal = actions["result_id"] == spadlconfig.result_id["success"]
        assert int((is_shot & is_goal).sum()) >= 2

    def test_synthetic_match_yields_yellow_and_red_cards(self):
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        foul_actions = actions[actions["type_id"] == spadlconfig.actiontype_id["foul"]]
        assert (foul_actions["result_id"] == spadlconfig.result_id["yellow_card"]).any()
        assert (foul_actions["result_id"] == spadlconfig.result_id["red_card"]).any()

    def test_synthetic_match_tackle_winner_columns_populated(self):
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        tackles = actions[actions["type_id"] == spadlconfig.actiontype_id["tackle"]]
        assert len(tackles) >= 2
        winners_diff_from_actor = (tackles["tackle_winner_player_id"] != tackles["player_id"]).any()
        winners_eq_actor = (tackles["tackle_winner_player_id"] == tackles["player_id"]).any()
        assert winners_diff_from_actor and winners_eq_actor


class TestPffAtomicComposability:
    """PFF SPADL output composes cleanly with Atomic-SPADL."""

    def test_atomic_conversion_runs_without_error(self):
        from silly_kicks.atomic.spadl import convert_to_atomic

        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        atomic_actions = convert_to_atomic(actions)
        assert len(atomic_actions) > 0
        for col in (
            "game_id",
            "period_id",
            "time_seconds",
            "team_id",
            "player_id",
            "type_id",
        ):
            assert col in atomic_actions.columns

    def test_atomic_add_possessions_runs(self):
        from silly_kicks.atomic.spadl import add_possessions, convert_to_atomic

        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        atomic_actions = convert_to_atomic(actions)
        with_poss = add_possessions(atomic_actions)
        assert "possession_id" in with_poss.columns


class TestPffVaepComposability:
    """PFF SPADL output composes cleanly with VAEP labels."""

    def test_vaep_labels_scores(self):
        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import scores

        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        actions_named = add_names(actions)
        labels = scores(actions_named, nr_actions=5)
        assert len(labels) == len(actions)
        assert int(labels["scores"].sum()) > 0

    def test_vaep_labels_concedes_runs(self):
        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import concedes

        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        actions_named = add_names(actions)
        labels = concedes(actions_named, nr_actions=5)
        assert len(labels) == len(actions)
