"""SkillCorner derived action inference tests."""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig


class TestStartTypeInterceptions:
    """Spec section 4.1: start_type-based interception detection."""

    def test_pass_interception_produces_interception(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p12"],
                "start_type": ["pass_reception", "pass_interception"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]
        assert result.iloc[0]["player_id"] == "p12"

    def test_recovery_produces_interception(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p12"],
                "start_type": ["pass_reception", "recovery"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]

    def test_throw_in_interception_produces_interception(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p12"],
                "start_type": ["pass_reception", "throw_in_interception"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]

    def test_pass_reception_produces_no_defensive_action(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_a"],
                "player_id": ["p1", "p2"],
                "start_type": ["pass_reception", "pass_reception"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 0


class TestOBETackleUpgrade:
    """Spec section 4.1: OBE direct_regain upgrades interception -> tackle."""

    def test_interception_upgraded_to_tackle_with_obe(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p12"],
                "start_type": ["pass_reception", "pass_interception"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(
            {
                "period": [1],
                "time_seconds": [14.8],
                "team_id": ["team_b"],
                "player_id": ["p13"],
                "end_type": ["direct_regain"],
                "x_start": [14.0],
                "y_start": [9.0],
            }
        )
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]
        assert result.iloc[0]["player_id"] == "p13"
        assert abs(result.iloc[0]["start_x"] - 14.0) < 0.01

    def test_recovery_upgraded_to_tackle_with_obe(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p12"],
                "start_type": ["pass_reception", "recovery"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(
            {
                "period": [1],
                "time_seconds": [14.5],
                "team_id": ["team_b"],
                "player_id": ["p14"],
                "end_type": ["direct_regain"],
                "x_start": [16.0],
                "y_start": [11.0],
            }
        )
        result = infer_defensive_actions(pp, obe)
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]

    def test_no_upgrade_when_obe_too_far_in_time(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p12"],
                "start_type": ["pass_reception", "pass_interception"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(
            {
                "period": [1],
                "time_seconds": [12.0],
                "team_id": ["team_b"],
                "player_id": ["p13"],
                "end_type": ["direct_regain"],
                "x_start": [14.0],
                "y_start": [9.0],
            }
        )
        result = infer_defensive_actions(pp, obe)
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]

    def test_indirect_regain_does_not_upgrade(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [10.0, 15.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p12"],
                "start_type": ["pass_reception", "pass_interception"],
                "x_start": [5.0, 15.0],
                "y_start": [3.0, 10.0],
            }
        )
        obe = pd.DataFrame(
            {
                "period": [1],
                "time_seconds": [14.8],
                "team_id": ["team_b"],
                "player_id": ["p13"],
                "end_type": ["indirect_regain"],
                "x_start": [14.0],
                "y_start": [9.0],
            }
        )
        result = infer_defensive_actions(pp, obe)
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]


class TestKeeperSaves:
    """Spec section 4.2: shot -> opponent possession = keeper_save."""

    def test_shot_followed_by_opponent_produces_keeper_save(self):
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [25.0, 27.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p15"],
                "end_type": ["shot", "pass"],
                "x_start": [30.0, -40.0],
                "y_start": [2.0, 0.0],
            }
        )
        result = infer_keeper_saves(pp)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]
        assert result.iloc[0]["player_id"] == "p15"
        assert result.iloc[0]["result_id"] == spadlconfig.result_id["success"]

    def test_shot_followed_by_goal_no_keeper_save(self):
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [25.0, 27.0],
                "team_id": ["team_a", "team_a"],
                "player_id": ["p1", "p2"],
                "end_type": ["shot", "pass"],
                "game_interruption_after": ["goal_for", None],
                "x_start": [30.0, 5.0],
                "y_start": [2.0, 3.0],
            }
        )
        result = infer_keeper_saves(pp)
        assert len(result) == 0

    def test_shot_followed_by_same_team_no_keeper_save(self):
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [25.0, 27.0],
                "team_id": ["team_a", "team_a"],
                "player_id": ["p1", "p2"],
                "end_type": ["shot", "pass"],
                "x_start": [30.0, 5.0],
                "y_start": [2.0, 3.0],
            }
        )
        result = infer_keeper_saves(pp)
        assert len(result) == 0

    def test_goal_kick_against_no_keeper_save(self):
        """goal_kick_against = shot missed wide/over bar -- not a save."""
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [25.0, 27.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p15"],
                "end_type": ["shot", "pass"],
                "game_interruption_after": ["goal_kick_against", None],
                "x_start": [30.0, -40.0],
                "y_start": [2.0, 0.0],
            }
        )
        result = infer_keeper_saves(pp)
        assert len(result) == 0

    def test_corner_for_produces_keeper_save(self):
        """corner_for after shot = deflected behind goal line -- plausible save."""
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [25.0, 27.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p15"],
                "end_type": ["shot", "pass"],
                "game_interruption_after": ["corner_for", None],
                "x_start": [30.0, -40.0],
                "y_start": [2.0, 0.0],
            }
        )
        result = infer_keeper_saves(pp)
        assert len(result) == 1

    @pytest.mark.parametrize(
        "gi_after_val",
        [
            "free_kick_against",
            "throw_in_against",
            "throw_in_for",
        ],
    )
    def test_non_save_gi_after_excluded(self, gi_after_val):
        """gi_after values that indicate non-save outcomes produce no keeper_save."""
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame(
            {
                "event_id": ["pp_1", "pp_2"],
                "period": [1, 1],
                "time_seconds": [25.0, 27.0],
                "team_id": ["team_a", "team_b"],
                "player_id": ["p1", "p15"],
                "end_type": ["shot", "pass"],
                "game_interruption_after": [gi_after_val, None],
                "x_start": [30.0, -40.0],
                "y_start": [2.0, 0.0],
            }
        )
        result = infer_keeper_saves(pp)
        assert len(result) == 0
