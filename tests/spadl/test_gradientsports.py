"""Gradient Sports DataFrame SPADL converter tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import gradientsports as gs_mod
from silly_kicks.spadl.schema import GRADIENTSPORTS_SPADL_COLUMNS

# Minimum set of input columns to construct a one-row test DataFrame.
# Mirrors the EXPECTED_INPUT_COLUMNS frozenset in gradientsports.py.
_REQUIRED_COLS = sorted(gs_mod.EXPECTED_INPUT_COLUMNS)

_SYNTHETIC_FIXTURE = Path(__file__).parent.parent / "datasets" / "gradientsports" / "synthetic_match.json"


def _load_synthetic_events() -> pd.DataFrame:
    """Load the synthetic match JSON and flatten into the EXPECTED_INPUT_COLUMNS shape."""
    with _SYNTHETIC_FIXTURE.open("r", encoding="utf-8") as f:
        events_json = json.load(f)

    rows = []
    for ev in events_json:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        # Real Gradient Sports data carries `fouls` as a single dict per event (not a list).
        f0 = ev.get("fouls") or {}
        ball = (ev.get("ball") or [{}])[0] if ev.get("ball") else {}

        rows.append(
            {
                "game_id": ev["gameId"],
                "event_id": ev["gameEventId"],
                "possession_event_id": ev.get("possessionEventId"),
                "period_id": ge.get("period"),
                "time_seconds": ge.get("startGameClock"),
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
                "nonEvent": pe.get("nonEvent"),
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
        "team_id",
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


class TestGradientsportsContract:
    """Contract: return shape, schema, dtypes, no input mutation."""

    def test_returns_tuple_dataframe_conversion_report(self):
        events = _df_minimal_pass()
        result = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        actions, report = result
        assert isinstance(actions, pd.DataFrame)
        assert report.provider == "gradientsports"

    def test_output_schema_matches_gradientsports_spadl_columns(self):
        events = _df_minimal_pass()
        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert list(actions.columns) == list(GRADIENTSPORTS_SPADL_COLUMNS.keys())

    def test_dtypes_match_schema(self):
        events = _df_minimal_pass()
        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        for col, expected in GRADIENTSPORTS_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected, f"{col}: got {actions[col].dtype}, expected {expected}"

    def test_empty_input_returns_empty_actions_with_schema(self):
        empty = pd.DataFrame({c: [] for c in _REQUIRED_COLS})
        actions, report = gs_mod.convert_to_actions(
            empty,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 0
        assert list(actions.columns) == list(GRADIENTSPORTS_SPADL_COLUMNS.keys())
        assert report.total_events == 0
        assert report.total_actions == 0

    def test_input_dataframe_not_mutated(self):
        events = _df_minimal_pass()
        original_columns = list(events.columns)
        original_len = len(events)
        _, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert list(events.columns) == original_columns
        assert len(events) == original_len


class TestGradientsportsRequiredColumns:
    """Missing any required input column must raise ValueError with column names."""

    @pytest.mark.parametrize("missing", _REQUIRED_COLS)
    def test_missing_required_column_raises(self, missing):
        events = _df_minimal_pass().drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            gs_mod.convert_to_actions(
                events,
                home_team_id=100,
                home_team_start_left=True,
            )


class TestGradientsportsCoordinateTranslation:
    """Gradient Sports centered meters → SPADL bottom-left meters."""

    def test_center_spot_translates_to_pitch_center(self):
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 0.0
        df.loc[0, "ball_y"] = 0.0
        actions, _ = gs_mod.convert_to_actions(
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
        df.loc[0, "ball_x"] = -52.5  # centered: left-side corner
        df.loc[0, "ball_y"] = -34.0
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # Home team period 1, no flip → SPADL (0, 0).
        assert actions.iloc[0]["start_x"] == pytest.approx(0.0)
        assert actions.iloc[0]["start_y"] == pytest.approx(0.0)


class TestGradientsportsCoordinateClipping:
    """OOB coordinate clipping to SPADL pitch bounds [0, 105] x [0, 68].

    Lakehouse WC2022 evidence: 1,108/91,931 actions (1.2%) had OOB coords.
    Max overshoot: ~5m x, ~8m y (throw-ins, GK overruns, tracking noise).
    All other providers clip; GS is the only source that was missing it.
    """

    def test_high_oob_start_coordinates_clipped(self):
        """start_x > 105 and start_y > 68 are clipped to pitch bounds."""
        df = _df_minimal_pass()
        # ball_x=57.5 → SPADL 110.0 (OOB by 5m, matches lakehouse max_x=110.07)
        # ball_y=44.0 → SPADL 78.0 (OOB by 10m, matches lakehouse max_y=78.07)
        df.loc[0, "ball_x"] = 57.5
        df.loc[0, "ball_y"] = 44.0
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["start_x"] == pytest.approx(spadlconfig.field_length)
        assert actions.iloc[0]["start_y"] == pytest.approx(spadlconfig.field_width)

    def test_low_oob_start_coordinates_clipped(self):
        """start_x < 0 and start_y < 0 are clipped to zero."""
        df = _df_minimal_pass()
        # ball_x=-57.9 → SPADL -5.4 (matches lakehouse min_x=-5.4)
        # ball_y=-42.15 → SPADL -8.15 (matches lakehouse min_y=-8.15)
        df.loc[0, "ball_x"] = -57.9
        df.loc[0, "ball_y"] = -42.15
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["start_x"] == pytest.approx(0.0)
        assert actions.iloc[0]["start_y"] == pytest.approx(0.0)

    def test_end_coordinates_clipped_after_derive(self):
        """end_x/end_y (derived from next-action start) must also be clipped.

        _derive_end_coordinates sets end = next-action's start for pass-class
        types. If the next action's start is OOB, the derived end must be clipped.
        """
        df = _df_minimal_pass()
        # Two-row frame: pass at center, followed by pass at OOB location.
        # The first pass's end_x/end_y = second pass's start_x/start_y (OOB).
        row2 = df.iloc[0].copy()
        row2["event_id"] = 2
        row2["possession_event_id"] = 2
        row2["time_seconds"] = 12.0
        row2["ball_x"] = 57.5  # → SPADL 110.0 (OOB)
        row2["ball_y"] = 44.0  # → SPADL 78.0 (OOB)
        df = pd.concat([df, pd.DataFrame([row2])], ignore_index=True)
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # First action's end coords were derived from second action's start.
        assert actions.iloc[0]["end_x"] <= spadlconfig.field_length
        assert actions.iloc[0]["end_y"] <= spadlconfig.field_width

    def test_away_team_oob_clipped_after_ltr_flip(self):
        """Away team LTR flip doesn't produce or preserve OOB coordinates."""
        df = _df_minimal_pass()
        df.loc[0, "team_id"] = 200  # away
        df.loc[0, "ball_x"] = 57.5  # → SPADL 110.0, then LTR flip → 105-110=-5 (OOB low!)
        df.loc[0, "ball_y"] = 44.0  # → SPADL 78.0, then LTR y-flip → 68-78=-10 (OOB low!)
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["start_x"] >= 0.0
        assert actions.iloc[0]["start_y"] >= 0.0
        assert actions.iloc[0]["start_x"] <= spadlconfig.field_length
        assert actions.iloc[0]["start_y"] <= spadlconfig.field_width

    def test_inbounds_coordinates_unchanged(self):
        """Coordinates within [0, 105] x [0, 68] are not affected by clipping."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25  # → SPADL 78.75 (in bounds)
        df.loc[0, "ball_y"] = 10.0  # → SPADL 44.0 (in bounds)
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["start_x"] == pytest.approx(78.75)
        assert actions.iloc[0]["start_y"] == pytest.approx(44.0)

    def test_synthetic_fixture_all_coordinates_in_bounds(self):
        """Full synthetic fixture (including OOB events) produces zero OOB rows."""
        events = _load_synthetic_events()
        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        for col in ("start_x", "end_x"):
            oob = (actions[col] < 0) | (actions[col] > spadlconfig.field_length)
            assert not oob.any(), (
                f"{col} has {oob.sum()} OOB values: min={actions[col].min()}, max={actions[col].max()}"
            )
        for col in ("start_y", "end_y"):
            oob = (actions[col] < 0) | (actions[col] > spadlconfig.field_width)
            assert not oob.any(), (
                f"{col} has {oob.sum()} OOB values: min={actions[col].min()}, max={actions[col].max()}"
            )


class TestGradientsportsDirectionOfPlay:
    """All teams attack left-to-right after conversion (per-period flip)."""

    def test_home_period1_no_flip(self):
        """Home team, period 1, home_team_start_left=True → no flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "ball_y"] = 0.0
        df.loc[0, "team_id"] = 100  # home
        df.loc[0, "period_id"] = 1
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["start_x"] == pytest.approx(78.75)


class TestGradientsportsExtraTimeFallback:
    """ET data without explicit ET-direction param raises ValueError."""

    def test_period3_event_without_extratime_param_raises(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 3
        with pytest.raises(ValueError, match="home_team_start_left_extratime"):
            gs_mod.convert_to_actions(
                df,
                home_team_id=100,
                home_team_start_left=True,
            )

    def test_period4_event_without_extratime_param_raises(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 4
        with pytest.raises(ValueError, match="home_team_start_left_extratime"):
            gs_mod.convert_to_actions(
                df,
                home_team_id=100,
                home_team_start_left=True,
            )

    def test_period3_event_with_extratime_param_succeeds(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 3
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 1


class TestGradientsportsBodyPart:
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        expected_id = spadlconfig.bodypart_id[expected_name]
        assert actions.iloc[0]["bodypart_id"] == expected_id


class TestGradientsportsPassDispatch:
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["success"]

    def test_pass_outcome_fail_is_fail(self):
        df = _df_minimal_pass()
        df.loc[0, "pass_outcome_type"] = "F"
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["fail"]


class TestGradientsportsCrossDispatch:
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["success"]


class TestGradientsportsShotDispatch:
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
        actions, _ = gs_mod.convert_to_actions(
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
            # "O" is the off-target shot bucket (NOT own-goal). Verified against the
            # full PFF FC / Gradient Sports WC2022 feed (64 matches): a 0-0 match
            # (MAR-ESP) carries O=10, and O occurs 4-17x every match — impossible for
            # own goals. Own goals surface under "G". See _dispatch_actiontype_resultid.
            ("O", "fail"),
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id[expected_result]


class TestGradientsportsShotOutcomeRegression:
    """Realistic full-match regression for the ``shot_outcome_type == "O"`` →
    ``owngoal`` mis-map fixed in 4.12.2.

    ``"O"`` is the off-target shot bucket (alongside ``S``=saved / ``B``=blocked),
    NOT own-goal. The 4.12.2 bug turned such SHOTS into ``owngoal`` results, so the
    enduring guard is: **no shot-class action carries the owngoal result.** (Real
    own goals — the RE+G capture, 4.13.0 — are ``bad_touch``, not shots.) Exercised
    on the committed synthetic match, so it runs in the regular (non-e2e) suite.
    """

    def test_no_shot_class_action_is_owngoal_on_realistic_match(self):
        events = _load_synthetic_events()

        # Guard: the fixture must actually carry off-target "O" shots, else this
        # regression cannot bite (a no-change test must exercise the path that
        # CAN change the value).
        shot_outcomes = events.loc[events["possession_event_type"] == "SH", "shot_outcome_type"]
        n_off_target = int((shot_outcomes == "O").sum())
        assert n_off_target >= 1, "fixture must contain >=1 off-target 'O' shot for this regression"

        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )

        # 4.12.2 core: the "O" bug turned SHOTS into owngoals. No shot-class action may carry owngoal.
        shot_type_ids = {spadlconfig.actiontype_id[name] for name in ("shot", "shot_freekick", "shot_penalty")}
        owngoal_id = spadlconfig.result_id["owngoal"]
        shot_owngoals = actions[actions["type_id"].isin(shot_type_ids) & (actions["result_id"] == owngoal_id)]
        assert len(shot_owngoals) == 0, (
            "no shot-class action may carry the owngoal result — 'O' is off-target, not own-goal"
        )

        # Any owngoal present is the legitimate RE+G own-goal capture (bad_touch), never a shot.
        owngoals = actions[actions["result_id"] == owngoal_id]
        assert (owngoals["type_id"] == spadlconfig.actiontype_id["bad_touch"]).all()


class TestGradientsportsRebound:
    """RE events disambiguate by keeper_touch_type → keeper_save / keeper_pick_up."""

    def test_rebound_default_is_keeper_save(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "keeper_touch_type"] = None
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]

    def test_rebound_catch_class_is_keeper_pick_up(self):
        """Catch-class keeper_touch_type → keeper_pick_up.

        NOTE: The exact Gradient Sports keeper_touch_type code letters are not enumerated
        by the spec; the test uses "C" as a placeholder catch code matching
        the catch_class set in gradientsports.py. If the synthetic match generator
        (Task 19) authors a different vocabulary, update this code AND the
        catch_class set in gradientsports.py simultaneously.
        """
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "keeper_touch_type"] = "C"
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_pick_up"]


class TestGradientsportsTackle:
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]

    def test_tackle_winner_columns_populated_when_challenger_wins(self):
        """Challenger (5/200) wins → carrier (1/100) lost."""
        df = self._df_tackle(winner_id=5, winner_team_id=200)
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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


class TestGradientsportsClearanceDribbleTouchControl:
    """OTB+CL → clearance, OTB+BC → dribble, OTB+TC → bad_touch."""

    def test_clearance(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CL"
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["clearance"]

    def test_ball_carry(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "BC"
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["dribble"]

    def test_touch_control(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "TC"
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["bad_touch"]


class TestGradientsportsFoul:
    """Rows with foul_type non-null synthesize an extra SPADL foul action."""

    def test_foul_synthesizes_additional_action(self):
        df = _df_minimal_pass()
        df.loc[0, "foul_type"] = "STANDARD"
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        foul_row = actions[actions["type_id"] == spadlconfig.actiontype_id["foul"]].iloc[0]
        assert foul_row["result_id"] == spadlconfig.result_id["red_card"]

    def test_no_foul_no_synthesis(self):
        df = _df_minimal_pass()
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 1


class TestGradientsportsExclusions:
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
        actions, report = gs_mod.convert_to_actions(
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
        actions, report = gs_mod.convert_to_actions(
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
        actions, report = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 0
        assert report.excluded_counts.get("OTB+") == 1


class TestGradientsportsDedicatedFoulEvent:
    """Standalone FOUL gameEventType with possessionEventType='FO' converts
    in-place to a SPADL foul action (no phantom non_action row)."""

    def test_foul_event_in_place_conversion(self):
        df = _df_minimal_pass()
        df.loc[0, "game_event_type"] = "FOUL"
        df.loc[0, "possession_event_type"] = "FO"
        df.loc[0, "foul_type"] = "I"
        df.loc[0, "final_foul_outcome_type"] = "Y"
        actions, report = gs_mod.convert_to_actions(
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


class TestGradientsportsNullActorEvents:
    """Events with null teamId/playerId (OTB+CH challenges, FOUL+FO fouls)
    survive the exclusion filter in real GS data. The converter must not crash
    on NaN team_id — it should apply Int64→fillna(0)→int64, matching player_id.

    Root cause: WC 2022 has ~17 events/match (10 OTB+CH + 7 FOUL+FO) with
    gameEvents.teamId=NULL and gameEvents.playerId=NULL.
    """

    def _df_null_actor_challenge(self) -> pd.DataFrame:
        """OTB+CH event with null team_id and player_id (real-data pattern)."""
        base = {col: [None] for col in _REQUIRED_COLS}
        overrides = {
            "game_id": [10502],
            "event_id": [1],
            "possession_event_id": [1],
            "period_id": [1],
            "time_seconds": [10.5],
            "team_id": [None],  # null actor — the bug trigger
            "player_id": [None],  # null actor
            "game_event_type": ["OTB"],
            "possession_event_type": ["CH"],
            "set_piece_type": [None],
            "ball_x": [0.0],
            "ball_y": [0.0],
        }
        base.update(overrides)
        df = pd.DataFrame(base)
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
        # team_id stays as object with None — reproduces the real-data dtype
        return df

    def _df_null_actor_foul(self) -> pd.DataFrame:
        """FOUL+FO event with null team_id and player_id (real-data pattern)."""
        base = {col: [None] for col in _REQUIRED_COLS}
        overrides = {
            "game_id": [10502],
            "event_id": [2],
            "possession_event_id": [2],
            "period_id": [1],
            "time_seconds": [15.0],
            "team_id": [None],  # null actor
            "player_id": [None],  # null actor
            "game_event_type": ["FOUL"],
            "possession_event_type": ["FO"],
            "set_piece_type": [None],
            "ball_x": [5.0],
            "ball_y": [3.0],
            "foul_type": ["I"],  # indirect foul
            "final_foul_outcome_type": [None],
        }
        base.update(overrides)
        df = pd.DataFrame(base)
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

    def test_otb_ch_null_actor_does_not_crash(self):
        """OTB+CH challenge with null teamId must convert without error."""
        df = self._df_null_actor_challenge()
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 1
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]
        assert actions.iloc[0]["team_id"] == 0
        assert actions.iloc[0]["player_id"] == 0

    def test_foul_fo_null_actor_does_not_crash(self):
        """FOUL+FO foul with null teamId must convert without error."""
        df = self._df_null_actor_foul()
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 1
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["foul"]
        assert actions.iloc[0]["team_id"] == 0
        assert actions.iloc[0]["player_id"] == 0

    def test_mixed_null_and_valid_actors(self):
        """Batch with both null-actor and valid-actor events converts cleanly."""
        valid = _df_minimal_pass()
        null_ch = self._df_null_actor_challenge()
        null_ch.loc[0, "event_id"] = 2
        null_ch.loc[0, "possession_event_id"] = 2
        null_ch.loc[0, "time_seconds"] = 11.0
        df = pd.concat([valid, null_ch], ignore_index=True)
        # Re-cast nullable columns after concat
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
        actions, _report = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 2
        # First row: valid pass with real team_id
        assert actions.iloc[0]["team_id"] == 100
        # Second row: null-actor challenge with team_id=0
        assert actions.iloc[1]["team_id"] == 0


class TestGradientsportsNanTimeSeconds:
    """Dedicated FOUL events (gameEventType=FOUL, possessionEventType=FO) in
    real Gradient Sports data have NULL startGameClock — 28/28 across 13/64
    WC2022 matches. The converter must impute time_seconds via ffill+bfill
    within each period, not leak NaN into the output.
    """

    @staticmethod
    def _build_df(event_specs: list[tuple[str | None, float | None, dict]]) -> pd.DataFrame:
        """Build a multi-row input DataFrame from (pe_type, time_s, extras) specs."""
        rows: list[dict] = []
        for i, (pe_type, time_s, extra) in enumerate(event_specs, start=1):
            row: dict[str, object] = {col: None for col in _REQUIRED_COLS}
            row.update(
                {
                    "game_id": 10502,
                    "event_id": i,
                    "possession_event_id": i,
                    "period_id": 1,
                    "time_seconds": time_s,
                    "team_id": 100,
                    "player_id": 1,
                    "game_event_type": "FOUL" if pe_type is None else "OTB",
                    "possession_event_type": "FO" if pe_type is None else pe_type,
                    "ball_x": 0.0,
                    "ball_y": 0.0,
                }
            )
            if pe_type is None:
                row["foul_type"] = "I"
                row["final_foul_outcome_type"] = "Y"
            row.update(extra)
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in ("possession_event_id", "player_id"):
            df[col] = df[col].astype("Int64")
        return df

    def test_dedicated_foul_nan_time_seconds_imputed(self):
        """Single dedicated FOUL with NaN time_seconds between two valid events."""
        df = self._build_df(
            [
                ("PA", 60.0, {"pass_outcome_type": "C", "body_type": "R", "set_piece_type": "O"}),
                (None, None, {}),  # FOUL — NaN time_seconds
                ("PA", 70.0, {"pass_outcome_type": "C", "body_type": "R", "set_piece_type": "O"}),
            ]
        )

        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        # The foul row must have a valid (non-NaN) time_seconds.
        foul_mask = actions["type_id"] == spadlconfig.actiontype_id["foul"]
        assert foul_mask.any(), "Expected at least one foul action"
        assert actions.loc[foul_mask, "time_seconds"].notna().all(), (
            "Dedicated FOUL with NULL startGameClock must have imputed time_seconds, got NaN"
        )
        # Imputed value should be 60.0 (forward-fill from the preceding pass).
        assert actions.loc[foul_mask, "time_seconds"].iloc[0] == pytest.approx(60.0)

    def test_period_leading_foul_uses_bfill(self):
        """FOUL at the start of a period (no preceding event) uses back-fill."""
        df = self._build_df(
            [
                (None, None, {}),  # FOUL at period start — NaN time_seconds
                ("PA", 5.0, {"pass_outcome_type": "C", "body_type": "R", "set_piece_type": "O"}),
            ]
        )

        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        foul_mask = actions["type_id"] == spadlconfig.actiontype_id["foul"]
        assert foul_mask.any()
        assert actions.loc[foul_mask, "time_seconds"].notna().all(), (
            "Period-leading FOUL must use bfill when no preceding event exists"
        )
        assert actions.loc[foul_mask, "time_seconds"].iloc[0] == pytest.approx(5.0)

    def test_synthetic_fixture_no_nan_time_seconds(self):
        """Full synthetic fixture (with realistic NULL startGameClock on FOUL)
        must produce zero NaN time_seconds in the output."""
        events = _load_synthetic_events()
        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        nan_mask = actions["time_seconds"].isna()
        assert not nan_mask.any(), (
            f"Found {nan_mask.sum()} NaN time_seconds in output:\n"
            f"{actions.loc[nan_mask, ['action_id', 'period_id', 'type_id', 'time_seconds']]}"
        )


class TestGradientsportsReportCounts:
    """ConversionReport.mapped_counts uses SPADL action-type names."""

    def test_mapped_counts_uses_spadl_names(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df.loc[1, "possession_event_type"] = "SH"
        df.loc[1, "shot_outcome_type"] = "G"
        df.loc[1, "time_seconds"] = 11.0
        _, report = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert report.mapped_counts.get("pass") == 1
        assert report.mapped_counts.get("shot") == 1


class TestGradientsportsEndCoordinates:
    """end_x/end_y of each action equals start_x/start_y of the next action
    in the same period (chained-event semantics)."""

    def test_pass_end_is_next_start(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df.loc[1, "ball_x"] = 20.0
        df.loc[1, "ball_y"] = 5.0
        df.loc[1, "time_seconds"] = 11.0
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
            df,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert actions.iloc[-1]["end_x"] == pytest.approx(actions.iloc[-1]["start_x"])
        assert actions.iloc[-1]["end_y"] == pytest.approx(actions.iloc[-1]["start_y"])


class TestGradientsportsSyntheticMatchE2E:
    """End-to-end conversion against the committed synthetic match fixture."""

    def test_synthetic_match_converts_with_no_unrecognized(self):
        events = _load_synthetic_events()
        _, report = gs_mod.convert_to_actions(
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
        _, report = gs_mod.convert_to_actions(
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
        _, report = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
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

    def test_synthetic_match_null_actor_events_convert(self):
        """Null-actor OTB+CH and FOUL+FO events (real WC 2022 pattern) convert
        with team_id=0, player_id=0 instead of crashing."""
        events = _load_synthetic_events()
        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        null_actor_rows = actions[actions["team_id"] == 0]
        assert len(null_actor_rows) >= 2, f"Expected >=2 null-actor rows (OTB+CH + FOUL+FO), got {len(null_actor_rows)}"
        assert (null_actor_rows["player_id"] == 0).all()


class TestGradientsportsAtomicComposability:
    """Gradient Sports SPADL output composes cleanly with Atomic-SPADL."""

    def test_atomic_conversion_runs_without_error(self):
        from silly_kicks.atomic.spadl import convert_to_atomic

        events = _load_synthetic_events()
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        atomic_actions = convert_to_atomic(actions)
        with_poss = add_possessions(atomic_actions)
        assert "possession_id" in with_poss.columns


class TestGradientsportsVaepComposability:
    """Gradient Sports SPADL output composes cleanly with VAEP labels."""

    def test_vaep_labels_scores(self):
        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.labels import scores

        events = _load_synthetic_events()
        actions, _ = gs_mod.convert_to_actions(
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
        actions, _ = gs_mod.convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        actions_named = add_names(actions)
        labels = concedes(actions_named, nr_actions=5)
        assert len(labels) == len(actions)


class TestGradientsportsNonEventExclusion:
    """Component 4: possessionEvents.nonEvent==True voided events are excluded (observable no-op)."""

    def test_nonevent_true_excluded_and_tallied(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df["nonEvent"] = [False, True]
        actions, report = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert len(actions) == 1
        assert report.excluded_counts.get("nonEvent") == 1

    def test_nonevent_column_absent_warns_and_noops(self):
        df = _df_minimal_pass()
        with pytest.warns(UserWarning, match="nonEvent"):
            actions, report = gs_mod.convert_to_actions(
                df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
            )
        assert len(actions) == 1
        assert "nonEvent" not in report.excluded_counts

    def test_nonevent_stringified_false_not_excluded(self):
        # robust coercion: the string "false" must NOT be treated truthy (would invert exclusion).
        df = _df_minimal_pass()
        df["nonEvent"] = ["false"]
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert len(actions) == 1


class TestGradientsportsOwnGoalCapture:
    """Component 1: RE + shotOutcome G -> bad_touch + owngoal (conceding team / rebounder scorer)."""

    def test_re_g_is_bad_touch_owngoal_conceding_team(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "team_id"] = 100  # conceding (acting) team
        df.loc[0, "player_id"] = 7  # OG scorer (= gameEvents.playerId = rebounderPlayerId)
        df.loc[0, "ball_x"] = -45.0  # -> start_x 7.5 (own half), survives the Task-3 tripwire
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["bad_touch"]
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["owngoal"]
        assert actions.iloc[0]["team_id"] == 100
        assert actions.iloc[0]["player_id"] == 7

    def test_re_without_g_still_keeper_save(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "shot_outcome_type"] = None
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]


class TestGradientsportsOwnGoalTripwire:
    """Component 1 tripwire: an RE+G owngoal must sit in the conceding team's OWN half (post-LTR);
    else WARN + revert to keeper_save/fail. (start_x mapping verified: ball_x -45 -> 7.5 own half;
    +45 -> 97.5 attacking half, for team 100 / period 1 / start_left=True.)"""

    def _re_g(self, ball_x):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "team_id"] = 100
        df.loc[0, "ball_x"] = ball_x
        df["nonEvent"] = [False]  # silence the Component-4 absent-column warning
        return df

    def test_re_g_in_attacking_half_reverts_with_warning(self):
        df = self._re_g(ball_x=45.0)  # -> start_x 97.5 (attacking half) -> revert
        with pytest.warns(UserWarning, match="own-goal"):
            actions, _ = gs_mod.convert_to_actions(
                df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
            )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["fail"]
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]

    def test_re_g_in_own_half_kept_as_owngoal_no_warning(self, recwarn):
        df = self._re_g(ball_x=-45.0)  # -> start_x 7.5 (own half) -> kept
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["owngoal"]
        assert not [w for w in recwarn.list if "own-goal" in str(w.message)]


class TestGradientsportsCrossGoal:
    """Component 2: CR + shotOutcome G -> keep the cross + synthesize a shot by the crosser."""

    def test_cr_g_keeps_cross_and_synthesizes_shot(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "set_piece_type"] = "F"  # free-kick cross -> shot_freekick
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "pass_outcome_type"] = None  # CR is not a PA — clear the minimal-pass default
        df.loc[0, "cross_outcome_type"] = "I"  # cross-as-pass incomplete
        df.loc[0, "team_id"] = 100
        df.loc[0, "player_id"] = 9  # crosser = scorer
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert len(actions) == 2
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["freekick_crossed"]
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["fail"]
        assert actions.iloc[1]["type_id"] == spadlconfig.actiontype_id["shot_freekick"]
        assert actions.iloc[1]["result_id"] == spadlconfig.result_id["success"]
        assert actions.iloc[1]["player_id"] == 9
        assert actions.iloc[1]["team_id"] == 100
        assert list(actions["action_id"]) == [0, 1]

    def test_cross_goal_with_foul_orders_shot_before_foul(self):
        # same-parent edge: a CR+G that ALSO carries a foul -> .4 shot AND .5 foul, in order.
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "set_piece_type"] = "O"
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "pass_outcome_type"] = None
        df.loc[0, "team_id"] = 100
        df.loc[0, "player_id"] = 9
        df.loc[0, "foul_type"] = "I"
        df.loc[0, "final_foul_outcome_type"] = "Y"
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert [actions.iloc[i]["type_id"] for i in range(len(actions))] == [
            spadlconfig.actiontype_id["cross"],
            spadlconfig.actiontype_id["shot"],
            spadlconfig.actiontype_id["foul"],
        ]
        assert list(actions["action_id"]) == [0, 1, 2]


class TestGradientsportsGoalCaptureRealistic:
    """Components 1/2/4 together on the committed synthetic match (RE+G OG #52, CR+G #53,
    disallowed SH+G #54)."""

    def test_owngoal_crossgoal_captured_disallowed_excluded(self):
        events = _load_synthetic_events()
        actions, report = gs_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        # exactly one own goal (the RE+G), on the conceding team 100
        og = actions[actions["result_id"] == spadlconfig.result_id["owngoal"]]
        assert len(og) == 1
        assert (og["type_id"] == spadlconfig.actiontype_id["bad_touch"]).all()
        assert (og["team_id"] == 100).all()
        # the cross-goal's synthetic shot is present (a successful shot-class action by the crosser)
        shot_ids = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")]
        shots = actions[actions["type_id"].isin(shot_ids)]
        assert (shots["result_id"] == spadlconfig.result_id["success"]).sum() >= 1
        # the disallowed SH+G (nonEvent=True) was excluded
        assert report.excluded_counts.get("nonEvent") == 1

    def test_composition_dense_action_ids_and_order(self):
        events = _load_synthetic_events()
        actions, _ = gs_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert list(actions["action_id"]) == list(range(len(actions)))  # dense + contiguous
        # the cross-goal's synthetic shot sorts immediately after its cross (same player) -> .4 offset
        shot_ids = {spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")}
        cross_ids = {spadlconfig.actiontype_id[n] for n in ("cross", "freekick_crossed", "corner_crossed")}
        adjacency = [
            i
            for i in range(len(actions) - 1)
            if actions.iloc[i]["type_id"] in cross_ids
            and actions.iloc[i + 1]["type_id"] in shot_ids
            and actions.iloc[i + 1]["player_id"] == actions.iloc[i]["player_id"]
        ]
        assert adjacency, "expected a cross immediately followed by its synthetic shot (same player)"


class TestGradientsportsSyntheticProvenance:
    """`is_synthetic` marks converter-injected rows (cross-goal shot, synthesized foul) that share the
    parent's `original_event_id`, so consumers don't collapse/drop them on a dedup."""

    def test_is_synthetic_in_schema_and_default_false(self):
        from silly_kicks.spadl.schema import GRADIENTSPORTS_SPADL_COLUMNS

        assert "is_synthetic" in GRADIENTSPORTS_SPADL_COLUMNS
        actions, _ = gs_mod.convert_to_actions(
            _df_minimal_pass(), home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert "is_synthetic" in actions.columns
        assert bool(actions.iloc[0]["is_synthetic"]) is False  # a plain real pass

    def test_cross_goal_shot_flagged_synthetic_cross_not(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "set_piece_type"] = "F"
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "pass_outcome_type"] = None
        df.loc[0, "team_id"] = 100
        df.loc[0, "player_id"] = 9
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert bool(actions.iloc[0]["is_synthetic"]) is False  # the real cross
        assert bool(actions.iloc[1]["is_synthetic"]) is True  # the synthesized shot

    def test_synthesized_foul_flagged_parent_not(self):
        df = _df_minimal_pass()  # real PA with an inline foul -> parent kept + synth foul row
        df.loc[0, "foul_type"] = "I"
        df.loc[0, "final_foul_outcome_type"] = "Y"
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert len(actions) == 2
        assert bool(actions.iloc[0]["is_synthetic"]) is False  # the pass
        assert bool(actions.iloc[1]["is_synthetic"]) is True  # the synthesized foul
