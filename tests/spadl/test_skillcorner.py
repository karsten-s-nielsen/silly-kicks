"""SkillCorner SPADL converter tests."""

import json
from pathlib import Path

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import KLOPPY_SPADL_COLUMNS, SKILLCORNER_SPADL_COLUMNS, ConversionReport

_FIXTURE_DIR = Path(__file__).parent.parent / "datasets" / "skillcorner"


class TestSkillcornerSchema:
    """Schema constant structure."""

    def test_extends_kloppy_spadl_columns(self):
        for col, dtype in KLOPPY_SPADL_COLUMNS.items():
            assert col in SKILLCORNER_SPADL_COLUMNS
            assert SKILLCORNER_SPADL_COLUMNS[col] == dtype

    def test_has_action_provenance_column(self):
        assert "action_provenance" in SKILLCORNER_SPADL_COLUMNS
        assert SKILLCORNER_SPADL_COLUMNS["action_provenance"] == "object"

    def test_skillcorner_extra_columns(self):
        # action_provenance (native/derived) + result_source (completion-label tier, D-S8/G1)
        extra = set(SKILLCORNER_SPADL_COLUMNS) - set(KLOPPY_SPADL_COLUMNS)
        assert extra == {"action_provenance", "result_source"}
        assert SKILLCORNER_SPADL_COLUMNS["result_source"] == "object"


class TestCoordinateTransform:
    """Spec section 5.3: centered meters to SPADL 0-105 x 0-68 frame."""

    def test_center_spot_maps_to_pitch_center(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([0.0]),
            y=pd.Series([0.0]),
            pitch_length=105,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 52.5) < 0.01
        assert abs(y.iloc[0] - 34.0) < 0.01

    def test_positive_corner_maps_to_top_right(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([52.5]),
            y=pd.Series([34.0]),
            pitch_length=105,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 105.0) < 0.01
        assert abs(y.iloc[0] - 68.0) < 0.01

    def test_negative_corner_maps_to_bottom_left(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([-52.5]),
            y=pd.Series([-34.0]),
            pitch_length=105,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 0.0) < 0.01
        assert abs(y.iloc[0] - 0.0) < 0.01

    def test_non_standard_pitch_rescales(self):
        """104m pitch: half_length=52, so x=52 rescales to 105.0."""
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([52.0]),
            y=pd.Series([34.0]),
            pitch_length=104,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 105.0) < 0.01
        assert abs(y.iloc[0] - 68.0) < 0.01

    def test_106m_pitch_rescales(self):
        """106m pitch: half_length=53, so x=53 rescales to 105.0."""
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, _y = _transform_coords(
            x=pd.Series([53.0]),
            y=pd.Series([34.0]),
            pitch_length=106,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 105.0) < 0.01

    def test_nan_propagates(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([float("nan")]),
            y=pd.Series([float("nan")]),
            pitch_length=105,
            pitch_width=68,
        )
        assert pd.isna(x.iloc[0])
        assert pd.isna(y.iloc[0])


class TestTimeParsing:
    """Spec section 5.5: parse time_start MM:SS.d string."""

    def test_simple_time(self):
        from silly_kicks.spadl.skillcorner import _parse_time_start

        result = _parse_time_start(pd.Series(["00:01.8"]))
        assert abs(result.iloc[0] - 1.8) < 0.01

    def test_multi_minute(self):
        from silly_kicks.spadl.skillcorner import _parse_time_start

        result = _parse_time_start(pd.Series(["12:34.5"]))
        assert abs(result.iloc[0] - (12 * 60 + 34.5)) < 0.01

    def test_zero_fraction(self):
        from silly_kicks.spadl.skillcorner import _parse_time_start

        result = _parse_time_start(pd.Series(["05:00.0"]))
        assert abs(result.iloc[0] - 300.0) < 0.01


class TestBodyPartDispatch:
    """Spec section 7.1.1: is_header -> head, hand_pass -> other, else -> foot."""

    def test_default_is_foot(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([False]),
            hand_pass=pd.Series([False]),
        )
        assert result[0] == spadlconfig.bodypart_id["foot"]

    def test_header(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([True]),
            hand_pass=pd.Series([False]),
        )
        assert result[0] == spadlconfig.bodypart_id["head"]

    def test_hand_pass(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([False]),
            hand_pass=pd.Series([True]),
        )
        assert result[0] == spadlconfig.bodypart_id["other"]

    def test_header_takes_priority_over_hand(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([True]),
            hand_pass=pd.Series([True]),
        )
        assert result[0] == spadlconfig.bodypart_id["head"]


class TestCrossDetection:
    """Spec section 3.2: native channel/third columns with spatial fallback."""

    def test_attacking_third_wide_left_is_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series(["attacking_third"]),
            channel=pd.Series(["wide_left"]),
            start_x_spadl=pd.Series([80.0]),
            start_y_spadl=pd.Series([10.0]),
        )
        assert result.iloc[0]

    def test_attacking_third_center_is_not_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series(["attacking_third"]),
            channel=pd.Series(["center"]),
            start_x_spadl=pd.Series([80.0]),
            start_y_spadl=pd.Series([34.0]),
        )
        assert not result.iloc[0]

    def test_middle_third_wide_is_not_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series(["middle_third"]),
            channel=pd.Series(["wide_right"]),
            start_x_spadl=pd.Series([50.0]),
            start_y_spadl=pd.Series([60.0]),
        )
        assert not result.iloc[0]

    def test_nan_columns_fall_back_to_spatial(self):
        """When native columns are NaN, use spatial heuristic."""
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series([None]),
            channel=pd.Series([None]),
            start_x_spadl=pd.Series([80.0]),
            start_y_spadl=pd.Series([10.0]),
        )
        assert result.iloc[0]

    def test_nan_columns_not_in_zone_is_not_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series([None]),
            channel=pd.Series([None]),
            start_x_spadl=pd.Series([50.0]),
            start_y_spadl=pd.Series([34.0]),
        )
        assert not result.iloc[0]


# --- Fixture loaders ---


def _load_basic_fixture():
    events = pd.read_csv(_FIXTURE_DIR / "basic_possessions.csv")
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        metadata = json.load(f)
    return events, metadata


def _load_derived_fixture():
    events = pd.read_csv(_FIXTURE_DIR / "derived_actions.csv")
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        metadata = json.load(f)
    return events, metadata


def test_block_columns_all_na_basic_and_derived():
    # SkillCorner records no shot/cross-block signal (real-data verified) -> both columns all pd.NA.
    # Cover BOTH fixtures: the derived-actions path exercises the `[actions.columns]` merge that
    # would crash if the columns were added to the native dict instead of after the concat.
    from silly_kicks.spadl.skillcorner import convert_to_actions

    for events, meta in (_load_basic_fixture(), _load_derived_fixture()):
        actions, _ = convert_to_actions(events, meta)
        for col in ("shot_blocked", "cross_blocked"):
            assert str(actions[col].dtype) == "boolean"
            assert actions[col].isna().all()


# --- Main converter tests ---


class TestConvertToActionsContract:
    """Contract: return shape, schema, dtypes."""

    def test_returns_tuple(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        result = convert_to_actions(events, meta)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], ConversionReport)

    def test_output_schema_matches(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        expected_cols = set(SKILLCORNER_SPADL_COLUMNS.keys())
        assert set(actions.columns) == expected_cols

    def test_dtypes_match_schema(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        for col, expected_dtype in SKILLCORNER_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected_dtype, (
                f"Column {col}: expected {expected_dtype}, got {actions[col].dtype}"
            )

    def test_provenance_values(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        valid = {"native", "derived"}
        assert set(actions["action_provenance"].unique()) <= valid

    def test_conversion_report_provider(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        _, report = convert_to_actions(events, meta)
        assert report.provider == "skillcorner"


class TestActionDispatch:
    """Spec section 3.1: end_type + game_interruption_before dispatch."""

    def test_pass_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "pass" in named["type_name"].values

    def test_shot_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "shot" in named["type_name"].values

    def test_goal_is_success(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]]
        goal_shots = shots[shots["result_id"] == spadlconfig.result_id["success"]]
        assert len(goal_shots) >= 1

    def test_clearance_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "clearance" in named["type_name"].values

    def test_goalkick_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "goalkick" in named["type_name"].values

    def test_throw_in_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "throw_in" in named["type_name"].values

    def test_foul_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "foul" in named["type_name"].values

    def test_cross_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "cross" in named["type_name"].values

    def test_possession_loss_is_non_action(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "non_action" in named["type_name"].values

    def test_header_bodypart(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        head_actions = actions[actions["bodypart_id"] == spadlconfig.bodypart_id["head"]]
        assert len(head_actions) >= 1

    def test_hand_pass_bodypart(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        other_actions = actions[actions["bodypart_id"] == spadlconfig.bodypart_id["other"]]
        assert len(other_actions) >= 1


class TestEndCoordinates:
    """Spec section 5.4: pass uses player_targeted_x_reception, fallback to x_end."""

    def test_pass_with_targeted_reception_uses_reception(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        # Row 1 is a pass with targeted_x_reception=15.0 vs x_end=8.0
        passes = actions[actions["type_id"] == spadlconfig.actiontype_id["pass"]]
        first_pass = passes.iloc[0]
        expected_end_x = (15.0 / 52.5) * 52.5 + 52.5  # = 67.5
        assert abs(first_pass["end_x"] - expected_end_x) < 0.5

    def test_pass_without_targeted_reception_uses_x_end(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        # Row 2 has no targeted_x_reception, x_end=18.0
        passes = actions[actions["type_id"] == spadlconfig.actiontype_id["pass"]]
        p2_passes = passes[passes["player_id"] == "p2"]
        if len(p2_passes) > 0:
            expected_end_x = (18.0 / 52.5) * 52.5 + 52.5  # = 70.5
            assert abs(p2_passes.iloc[0]["end_x"] - expected_end_x) < 0.5


class TestDerivedActions:
    """Spec section 4: dual-action production, start_type interceptions, OBE tackles, keeper saves."""

    def test_interception_with_obe_upgraded_to_tackle(self):
        """Row 2 + OBE row 3: pass_interception + direct_regain -> tackle."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        tackles = actions[actions["type_id"] == spadlconfig.actiontype_id["tackle"]]
        assert len(tackles) >= 1

    def test_recovery_with_obe_upgraded_to_tackle(self):
        """Row 4 + OBE row 5: recovery + direct_regain -> tackle."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        tackles = actions[actions["type_id"] == spadlconfig.actiontype_id["tackle"]]
        assert len(tackles) >= 2

    def test_interception_without_obe_stays_interception(self):
        """Rows 8-9: interceptions with no nearby OBE -> interception (not tackle)."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        interceptions = actions[actions["type_id"] == spadlconfig.actiontype_id["interception"]]
        assert len(interceptions) >= 1

    def test_keeper_save_after_shot(self):
        """Rows 6-7: shot followed by opponent possession -> keeper_save."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        ks = actions[actions["type_id"] == spadlconfig.actiontype_id["keeper_save"]]
        assert len(ks) >= 1
        assert (ks["action_provenance"] == "derived").all()

    def test_dual_action_interception_before_native(self):
        """Spec section 4 dual-action: derived defensive action ordered before its native action."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")

        derived = named[named["action_provenance"] == "derived"]
        defensive = derived[derived["type_name"].isin({"interception", "tackle"})]
        assert len(defensive) >= 1
        # Each derived defensive action must have a later native action in the stream
        for idx in defensive.index:
            pos = actions.index.get_loc(idx)
            remaining = named.iloc[pos + 1 :]  # type: ignore[operator]
            native_after = remaining[remaining["action_provenance"] == "native"]
            assert len(native_after) > 0, f"No native action after defensive at pos {pos}"

    def test_derived_actions_have_provenance(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        derived = actions[actions["action_provenance"] == "derived"]
        assert len(derived) >= 3

    def test_dribbles_are_derived(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        dribbles = actions[actions["type_id"] == spadlconfig.actiontype_id["dribble"]]
        if len(dribbles) > 0:
            assert (dribbles["action_provenance"] == "derived").all()
