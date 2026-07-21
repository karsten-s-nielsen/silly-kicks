"""ADR-045: enrichment coordinate columns must mirror with the canonical ones."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl.utils import play_left_to_right


def test_enriched_restart_coordinates_are_mirrored():
    actions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": 2,
                "player_id": 7,
                "start_x": 10.0,
                "start_y": 20.0,
                "end_x": 30.0,
                "end_y": 40.0,
                "enriched_start_x": 10.0,
                "enriched_start_y": 20.0,
                "enriched_end_x": 30.0,
                "enriched_end_y": 40.0,
                "type_id": 0,
                "result_id": 1,
                "bodypart_id": 0,
            }
        ]
    )
    out = play_left_to_right(actions, home_team_id=1)  # acting team is AWAY -> mirrored
    assert out.loc[0, "start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_end_y"] == pytest.approx(28.0)


# Task 9d: D8 -- the SPADL orienter (to_spadl_ltr), both branches
def test_to_spadl_ltr_absolute_frame_mirrors_enrichment_columns():
    from silly_kicks.spadl.orientation import ABSOLUTE_FRAME_HOME_RIGHT, to_spadl_ltr

    actions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": 2,
                "player_id": 7,
                "start_x": 10.0,
                "start_y": 20.0,
                "end_x": 30.0,
                "end_y": 40.0,
                "enriched_start_x": 10.0,
                "enriched_start_y": 20.0,
                "enriched_end_x": 30.0,
                "enriched_end_y": 40.0,
                "type_id": 0,
                "result_id": 1,
                "bodypart_id": 0,
            }
        ]
    )
    out = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=1)
    assert out.loc[0, "start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_start_x"] == pytest.approx(95.0)  # the D8 trap
    assert out.loc[0, "enriched_end_y"] == pytest.approx(28.0)


def test_to_spadl_ltr_per_period_mirrors_enrichment_columns():
    from silly_kicks.spadl.orientation import PER_PERIOD_ABSOLUTE, to_spadl_ltr

    actions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": 1,
                "player_id": 7,
                "start_x": 10.0,
                "start_y": 20.0,
                "end_x": 30.0,
                "end_y": 40.0,
                "enriched_start_x": 10.0,
                "enriched_start_y": 20.0,
                "enriched_end_x": 30.0,
                "enriched_end_y": 40.0,
                "type_id": 0,
                "result_id": 1,
                "bodypart_id": 0,
            }
        ]
    )
    # home team, period where home attacks LEFT -> row must mirror
    out = to_spadl_ltr(
        actions,
        input_convention=PER_PERIOD_ABSOLUTE,
        home_team_id=1,
        home_attacks_right_per_period={1: False},
    )
    assert out.loc[0, "start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_start_x"] == pytest.approx(95.0)


def test_to_spadl_ltr_preserves_NA_team_as_away():
    """BOTH-SIDES partner for the NA semantics the migration must not change."""
    from silly_kicks.spadl.orientation import ABSOLUTE_FRAME_HOME_RIGHT, to_spadl_ltr

    actions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "action_id": 0,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": None,
                "player_id": 7,
                "start_x": 10.0,
                "start_y": 20.0,
                "end_x": 30.0,
                "end_y": 40.0,
                "type_id": 0,
                "result_id": 1,
                "bodypart_id": 0,
            },
        ]
    )
    out = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=1)
    assert out.loc[0, "start_x"] == pytest.approx(95.0), "NA team_id must mirror as AWAY"
