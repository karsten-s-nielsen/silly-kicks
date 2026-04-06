"""Wyscout event-fixing pipeline: position extraction, duel/interception/touch transforms."""

import pandas as pd

from . import config as spadlconfig
from .base import min_dribble_length
from .wyscout import (
    _WS_SUBTYPE_ACCELERATION,
    _WS_SUBTYPE_AIR_DUEL,
    _WS_SUBTYPE_BALL_OUT_OF_FIELD,
    _WS_SUBTYPE_CLEARANCE,
    _WS_SUBTYPE_FK_SHOT,
    _WS_SUBTYPE_GROUND_ATT_DUEL,
    _WS_SUBTYPE_HEAD_PASS_BP,
    _WS_SUBTYPE_SHOT_ON_TARGET,
    _WS_SUBTYPE_SIMPLE_PASS,
    _WS_SUBTYPE_SIMULATION,
    _WS_SUBTYPE_TOUCH,
    _WS_TYPE_DUEL,
    _WS_TYPE_OFFSIDE,
    _WS_TYPE_PASS,
    _WS_TYPE_TAKE_ON,
    wyscout_tags,
)

# ---------------------------------------------------------------------------
# Position extraction (vectorized)
# ---------------------------------------------------------------------------


def _make_new_positions(events: pd.DataFrame) -> pd.DataFrame:
    """Extract the start and end coordinates for each action.

    Parameters
    ----------
    events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with start and end coordinates for each action.
    """
    pos_list = events["positions"].tolist()
    empty_pos: dict[str, float | None] = {"x": None, "y": None}

    start_pos = [p[0] if len(p) >= 1 else empty_pos for p in pos_list]
    end_pos = [p[1] if len(p) >= 2 else (p[0] if len(p) >= 1 else empty_pos) for p in pos_list]

    events["start_x"] = pd.Series([p.get("x") for p in start_pos], index=events.index, dtype=float)
    events["start_y"] = pd.Series([p.get("y") for p in start_pos], index=events.index, dtype=float)
    events["end_x"] = pd.Series([p.get("x") for p in end_pos], index=events.index, dtype=float)
    events["end_y"] = pd.Series([p.get("y") for p in end_pos], index=events.index, dtype=float)

    events = events.drop("positions", axis=1)  # type: ignore[reportAssignmentType]
    return events


# ---------------------------------------------------------------------------
# Event fixes
# ---------------------------------------------------------------------------


def _fix_wyscout_events(df_events: pd.DataFrame) -> pd.DataFrame:
    """Perform some fixes on the Wyscout events such that the spadl action dataframe can be built.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with an extra column 'offside'
    """
    df_events = _create_shot_coordinates(df_events)
    df_events = _convert_duels(df_events)
    df_events = _insert_interceptions(df_events)
    df_events = _add_offside_variable(df_events)
    df_events = _convert_touches(df_events)
    df_events = _convert_simulations(df_events)
    return df_events


def _create_shot_coordinates(df_events: pd.DataFrame) -> pd.DataFrame:
    """Create shot coordinates (estimates) from the Wyscout tags.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with end coordinates for shots
    """
    shot = df_events.subtype_id.isin([_WS_SUBTYPE_FK_SHOT, _WS_SUBTYPE_SHOT_ON_TARGET])
    pas = df_events.type_id == _WS_TYPE_PASS

    goal_center_idx = (
        df_events["position_goal_low_center"]
        | df_events["position_goal_mid_center"]
        | df_events["position_goal_high_center"]
    )
    df_events.loc[shot & goal_center_idx, "end_x"] = 100.0
    df_events.loc[shot & goal_center_idx, "end_y"] = 50.0

    goal_right_idx = (
        df_events["position_goal_low_right"]
        | df_events["position_goal_mid_right"]
        | df_events["position_goal_high_right"]
    )
    df_events.loc[shot & goal_right_idx, "end_x"] = 100.0
    df_events.loc[shot & goal_right_idx, "end_y"] = 55.0

    goal_left_idx = (
        df_events["position_goal_mid_left"] | df_events["position_goal_low_left"] | df_events["position_goal_high_left"]
    )
    df_events.loc[shot & goal_left_idx, "end_x"] = 100.0
    df_events.loc[shot & goal_left_idx, "end_y"] = 45.0

    out_center_idx = df_events["position_out_high_center"] | df_events["position_post_high_center"]
    df_events.loc[shot & out_center_idx, "end_x"] = 100.0
    df_events.loc[shot & out_center_idx, "end_y"] = 50.0

    out_right_idx = (
        df_events["position_out_low_right"] | df_events["position_out_mid_right"] | df_events["position_out_high_right"]
    )
    df_events.loc[shot & out_right_idx, "end_x"] = 100.0
    df_events.loc[shot & out_right_idx, "end_y"] = 60.0

    out_left_idx = (
        df_events["position_out_mid_left"] | df_events["position_out_low_left"] | df_events["position_out_high_left"]
    )
    df_events.loc[shot & out_left_idx, "end_x"] = 100.0
    df_events.loc[shot & out_left_idx, "end_y"] = 40.0

    post_left_idx = (
        df_events["position_post_mid_left"] | df_events["position_post_low_left"] | df_events["position_post_high_left"]
    )
    df_events.loc[shot & post_left_idx, "end_x"] = 100.0
    df_events.loc[shot & post_left_idx, "end_y"] = 55.38

    post_right_idx = (
        df_events["position_post_low_right"]
        | df_events["position_post_mid_right"]
        | df_events["position_post_high_right"]
    )
    df_events.loc[shot & post_right_idx, "end_x"] = 100.0
    df_events.loc[shot & post_right_idx, "end_y"] = 44.62

    blocked_idx = df_events["blocked"]
    df_events.loc[(shot | pas) & blocked_idx, "end_x"] = df_events.loc[blocked_idx, "start_x"]
    df_events.loc[(shot | pas) & blocked_idx, "end_y"] = df_events.loc[blocked_idx, "start_y"]

    return df_events


def _convert_duels(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert duel events.

    This function converts Wyscout duels that end with the ball out of field
    (subtype_id 50) into a pass for the player winning the duel to the location
    of where the ball went out of field. The remaining duels are removed as
    they are not on-the-ball actions.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe in which the duels are either removed or
        transformed into a pass
    """
    # Shift events dataframe by one and two time steps
    df_events1 = df_events.shift(-1)
    df_events2 = df_events.shift(-2)

    # Define selector for same period id
    selector_same_period = df_events["period_id"] == df_events2["period_id"]

    # Define selector for duels that are followed by an 'out of field' event
    selector_duel_out_of_field = (
        (df_events["type_id"] == _WS_TYPE_DUEL)
        & (df_events1["type_id"] == _WS_TYPE_DUEL)
        & (df_events2["subtype_id"] == _WS_SUBTYPE_BALL_OUT_OF_FIELD)
        & selector_same_period
    )

    # Define selectors for current time step
    selector0_duel_won = selector_duel_out_of_field & (df_events["team_id"] != df_events2["team_id"])
    selector0_duel_won_air = selector0_duel_won & (df_events["subtype_id"] == _WS_SUBTYPE_AIR_DUEL)
    selector0_duel_won_not_air = selector0_duel_won & (df_events["subtype_id"] != _WS_SUBTYPE_AIR_DUEL)

    # Define selectors for next time step
    selector1_duel_won = selector_duel_out_of_field & (df_events1["team_id"] != df_events2["team_id"])
    selector1_duel_won_air = selector1_duel_won & (df_events1["subtype_id"] == _WS_SUBTYPE_AIR_DUEL)
    selector1_duel_won_not_air = selector1_duel_won & (df_events1["subtype_id"] != _WS_SUBTYPE_AIR_DUEL)

    # Aggregate selectors
    selector_duel_won = selector0_duel_won | selector1_duel_won
    selector_duel_won_air = selector0_duel_won_air | selector1_duel_won_air
    selector_duel_won_not_air = selector0_duel_won_not_air | selector1_duel_won_not_air

    # Set types and subtypes
    df_events.loc[selector_duel_won, "type_id"] = _WS_TYPE_PASS
    df_events.loc[selector_duel_won_air, "subtype_id"] = _WS_SUBTYPE_HEAD_PASS_BP
    df_events.loc[selector_duel_won_not_air, "subtype_id"] = _WS_SUBTYPE_SIMPLE_PASS

    # set end location equal to ball out of field location
    df_events.loc[selector_duel_won, "accurate"] = False
    df_events.loc[selector_duel_won, "not_accurate"] = True
    df_events.loc[selector_duel_won, "end_x"] = 100 - df_events2.loc[selector_duel_won, "start_x"]
    df_events.loc[selector_duel_won, "end_y"] = 100 - df_events2.loc[selector_duel_won, "start_y"]

    # Define selector for ground attacking duels with take on
    selector_attacking_duel = df_events["subtype_id"] == _WS_SUBTYPE_GROUND_ATT_DUEL
    selector_take_on = (df_events["take_on_left"]) | (df_events["take_on_right"])
    selector_att_duel_take_on = selector_attacking_duel & selector_take_on

    # Set take ons type to 0
    df_events.loc[selector_att_duel_take_on, "type_id"] = _WS_TYPE_TAKE_ON

    # Set sliding tackles type to 0
    df_events.loc[df_events["sliding_tackle"], "type_id"] = _WS_TYPE_TAKE_ON

    # Remove the remaining duels
    df_events = df_events[df_events["type_id"] != _WS_TYPE_DUEL]  # type: ignore[reportAssignmentType]

    # Reset the index
    df_events = df_events.reset_index(drop=True)  # type: ignore[reportAssignmentType]

    return df_events


def _insert_interceptions(df_events: pd.DataFrame) -> pd.DataFrame:
    """Insert interception actions before passes, clearances and dribbles.

    This function converts passes (type_id 8), clearances (subtype_id 71) and
    accelerations (subtype_id 70) that are also interceptions (tag
    interception) in the Wyscout event data into two separate events, first an
    interception and then a pass/clearance/dribble.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe in which passes that were also denoted as
        interceptions in the Wyscout notation are transformed into two events
    """
    df_events_interceptions = df_events[
        df_events["interception"]
        & (
            (df_events["type_id"] == _WS_TYPE_PASS)
            | (df_events["subtype_id"] == _WS_SUBTYPE_ACCELERATION)
            | (df_events["subtype_id"] == _WS_SUBTYPE_CLEARANCE)
        )
    ].copy()

    if not df_events_interceptions.empty:
        df_events_interceptions.loc[:, [t[1] for t in wyscout_tags]] = False
        df_events_interceptions["interception"] = True
        df_events_interceptions["type_id"] = _WS_TYPE_TAKE_ON
        df_events_interceptions["subtype_id"] = _WS_TYPE_TAKE_ON
        df_events_interceptions[["end_x", "end_y"]] = df_events_interceptions[["start_x", "start_y"]]

        df_events = pd.concat([df_events_interceptions, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "milliseconds"], kind="mergesort")
        df_events = df_events.reset_index(drop=True)  # type: ignore[reportAssignmentType]

    return df_events


def _add_offside_variable(df_events: pd.DataFrame) -> pd.DataFrame:
    """Attach offside events to the previous action.

    This function removes the offside events in the Wyscout event data and adds
    sets offside to 1 for the previous event (if this was a passing event)

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe with an extra column 'offside'
    """
    # Create a new column for the offside variable
    df_events["offside"] = 0

    # Shift events dataframe by one timestep
    df_events1 = df_events.shift(-1)

    # Select offside passes
    selector_offside = (df_events1["type_id"] == _WS_TYPE_OFFSIDE) & (df_events["type_id"] == _WS_TYPE_PASS)

    # Set variable 'offside' to 1 for all offside passes
    df_events.loc[selector_offside, "offside"] = 1

    # Remove offside events
    df_events = df_events[df_events["type_id"] != _WS_TYPE_OFFSIDE]  # type: ignore[reportAssignmentType]

    # Reset index
    df_events = df_events.reset_index(drop=True)  # type: ignore[reportAssignmentType]

    return df_events


def _convert_simulations(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert simulations to failed take-ons.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe


    Returns
    -------
        pd.DataFrame
        Wyscout event dataframe in which simulation events are either
        transformed into a failed take-on
    """
    prev_events = df_events.shift(1)

    # Select simulations
    selector_simulation = df_events["subtype_id"] == _WS_SUBTYPE_SIMULATION

    # Select actions preceded by a failed take-on
    selector_previous_is_failed_take_on = (
        (prev_events["take_on_left"]) | (prev_events["take_on_right"]) & prev_events["not_accurate"]
    )

    # Transform simulations not preceded by a failed take-on to a failed take-on
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "type_id"] = _WS_TYPE_TAKE_ON
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "subtype_id"] = _WS_TYPE_TAKE_ON
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "accurate"] = False
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "not_accurate"] = True
    # Set take_on_left or take_on_right to True
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "take_on_left"] = True

    # Remove simulation events which are preceded by a failed take-on
    df_events = df_events[~(selector_simulation & selector_previous_is_failed_take_on)]  # type: ignore[reportAssignmentType]

    # Reset index
    df_events = df_events.reset_index(drop=True)  # type: ignore[reportAssignmentType]

    return df_events


def _convert_touches(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert touch events to dribbles or passes.

    This function converts the Wyscout 'touch' event (sub_type_id 72) into either
    a dribble or a pass (accurate or not depending on receiver)

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        Wyscout event dataframe without any touch events
    """
    df_events1 = df_events.shift(-1)

    selector_touch = (
        (df_events["subtype_id"] == _WS_SUBTYPE_TOUCH) & ~df_events["interception"] & ~df_events["missed_ball"]
    )

    selector_same_player = df_events["player_id"] == df_events1["player_id"]
    selector_same_team = df_events["team_id"] == df_events1["team_id"]

    # selector_touch_same_player = selector_touch & selector_same_player
    selector_touch_same_team = selector_touch & ~selector_same_player & selector_same_team
    selector_touch_other = selector_touch & ~selector_same_player & ~selector_same_team

    same_x = abs(df_events["end_x"] - df_events1["start_x"]) < min_dribble_length
    same_y = abs(df_events["end_y"] - df_events1["start_y"]) < min_dribble_length
    same_loc = same_x & same_y

    df_events.loc[selector_touch_same_team & same_loc, "type_id"] = _WS_TYPE_PASS
    df_events.loc[selector_touch_same_team & same_loc, "subtype_id"] = _WS_SUBTYPE_SIMPLE_PASS
    df_events.loc[selector_touch_same_team & same_loc, "accurate"] = True
    df_events.loc[selector_touch_same_team & same_loc, "not_accurate"] = False

    df_events.loc[selector_touch_other & same_loc, "type_id"] = _WS_TYPE_PASS
    df_events.loc[selector_touch_other & same_loc, "subtype_id"] = _WS_SUBTYPE_SIMPLE_PASS
    df_events.loc[selector_touch_other & same_loc, "accurate"] = False
    df_events.loc[selector_touch_other & same_loc, "not_accurate"] = True

    return df_events


# ---------------------------------------------------------------------------
# Action fixes (coordinate adjustments, goalkick results, keeper saves)
# ---------------------------------------------------------------------------


def _fix_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix the generated actions.

    Parameters
    ----------
    df_actions : pd.DataFrame
        SPADL actions dataframe

    Returns
    -------
    pd.DataFrame
        SpADL actions dataframe with end coordinates for shots
    """
    df_actions["start_x"] = (df_actions["start_x"] * spadlconfig.field_length / 100).clip(0, spadlconfig.field_length)
    df_actions["start_y"] = (
        (100 - df_actions["start_y"]) * spadlconfig.field_width / 100
        # y is from top to bottom in Wyscout
    ).clip(0, spadlconfig.field_width)
    df_actions["end_x"] = (df_actions["end_x"] * spadlconfig.field_length / 100).clip(0, spadlconfig.field_length)
    df_actions["end_y"] = (
        (100 - df_actions["end_y"]) * spadlconfig.field_width / 100
        # y is from top to bottom in Wyscout
    ).clip(0, spadlconfig.field_width)
    df_actions = _fix_goalkick_coordinates(df_actions)
    df_actions = _adjust_goalkick_result(df_actions)
    df_actions = _fix_foul_coordinates(df_actions)
    df_actions = _fix_keeper_save_coordinates(df_actions)
    df_actions = _remove_keeper_goal_actions(df_actions)
    df_actions.reset_index(drop=True, inplace=True)

    return df_actions


def _fix_goalkick_coordinates(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix goalkick coordinates.

    This function sets the goalkick start coordinates to (5,34)

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with start coordinates for goalkicks in the
        corner of the pitch

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe including start coordinates for goalkicks
    """
    goalkicks_idx = df_actions["type_id"] == spadlconfig.actiontype_id["goalkick"]
    df_actions.loc[goalkicks_idx, "start_x"] = 5.0
    df_actions.loc[goalkicks_idx, "start_y"] = 34.0

    return df_actions


def _fix_foul_coordinates(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix fould coordinates.

    This function sets foul end coordinates equal to the foul start coordinates

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with no end coordinates for fouls

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe including start coordinates for goalkicks
    """
    fouls_idx = df_actions["type_id"] == spadlconfig.actiontype_id["foul"]
    df_actions.loc[fouls_idx, "end_x"] = df_actions.loc[fouls_idx, "start_x"]
    df_actions.loc[fouls_idx, "end_y"] = df_actions.loc[fouls_idx, "start_y"]

    return df_actions


def _fix_keeper_save_coordinates(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Fix keeper save coordinates.

    This function sets keeper_save start coordinates equal to
    keeper_save end coordinates. It also inverts the shot coordinates to the own goal.

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with start coordinates in the corner of the pitch

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe with correct keeper_save coordinates
    """
    saves_idx = df_actions["type_id"] == spadlconfig.actiontype_id["keeper_save"]
    # invert the coordinates
    df_actions.loc[saves_idx, "end_x"] = spadlconfig.field_length - df_actions.loc[saves_idx, "end_x"]
    df_actions.loc[saves_idx, "end_y"] = spadlconfig.field_width - df_actions.loc[saves_idx, "end_y"]
    # set start coordinates equal to start coordinates
    df_actions.loc[saves_idx, "start_x"] = df_actions.loc[saves_idx, "end_x"]
    df_actions.loc[saves_idx, "start_y"] = df_actions.loc[saves_idx, "end_y"]

    return df_actions


def _remove_keeper_goal_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Remove keeper goal-saving actions.

    This function removes keeper_save actions that appear directly after a goal

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with keeper actions directly after a goal

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe without keeper actions directly after a goal
    """
    prev_actions = df_actions.shift(1)
    same_phase = prev_actions.time_seconds + 10 > df_actions.time_seconds
    shot_goals = (prev_actions.type_id == spadlconfig.actiontype_id["shot"]) & (prev_actions.result_id == 1)
    penalty_goals = (prev_actions.type_id == spadlconfig.actiontype_id["shot_penalty"]) & (prev_actions.result_id == 1)
    freekick_goals = (prev_actions.type_id == spadlconfig.actiontype_id["shot_freekick"]) & (
        prev_actions.result_id == 1
    )
    goals = shot_goals | penalty_goals | freekick_goals
    keeper_save = df_actions["type_id"] == spadlconfig.actiontype_id["keeper_save"]
    goals_keepers_idx = same_phase & goals & keeper_save
    df_actions = df_actions.drop(df_actions.index[goals_keepers_idx])  # type: ignore[reportAssignmentType]
    df_actions = df_actions.reset_index(drop=True)  # type: ignore[reportAssignmentType]

    return df_actions


def _adjust_goalkick_result(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Adjust goalkick results.

    This function adjusts goalkick results depending on whether
    the next action is performed by the same team or not

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe with incorrect goalkick results

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe with correct goalkick results
    """
    nex_actions = df_actions.shift(-1)
    goalkicks = df_actions["type_id"] == spadlconfig.actiontype_id["goalkick"]
    same_team = df_actions["team_id"] == nex_actions["team_id"]
    accurate = same_team & goalkicks
    not_accurate = ~same_team & goalkicks
    df_actions.loc[accurate, "result_id"] = 1
    df_actions.loc[not_accurate, "result_id"] = 0

    return df_actions
