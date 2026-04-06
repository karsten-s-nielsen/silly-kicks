"""Wyscout event stream data to SPADL converter."""

import warnings
from collections import Counter
from typing import Any, Optional

import pandas as pd

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_direction_of_play,
    min_dribble_length,
)
from .schema import ConversionReport
from .utils import _finalize_output, _validate_input_columns

# ---------------------------------------------------------------------------
# Wyscout event type IDs
# ---------------------------------------------------------------------------
_WS_TYPE_TAKE_ON: int = 0       # synthetic type assigned to take-ons / tackles
_WS_TYPE_DUEL: int = 1
_WS_TYPE_FOUL: int = 2
_WS_TYPE_OFFSIDE: int = 6
_WS_TYPE_PASS: int = 8
_WS_TYPE_GK: int = 9
_WS_TYPE_SHOT: int = 10

# ---------------------------------------------------------------------------
# Wyscout event subtype IDs
# ---------------------------------------------------------------------------
_WS_SUBTYPE_AIR_DUEL: int = 10
_WS_SUBTYPE_GROUND_ATT_DUEL: int = 11
_WS_SUBTYPE_GROUND_DEF_DUEL: int = 12
_WS_SUBTYPE_GROUND_LOOSE_BALL: int = 13
_WS_SUBTYPE_HEAD_PASS: int = 21
_WS_SUBTYPE_HAND_FOUL: int = 22
_WS_SUBTYPE_LATE_CARD_FOUL: int = 23
_WS_SUBTYPE_OUT_OF_GAME_FOUL: int = 24
_WS_SUBTYPE_SIMULATION: int = 25
_WS_SUBTYPE_VIOLENT_FOUL: int = 26
_WS_SUBTYPE_CORNER: int = 30
_WS_SUBTYPE_FK_SHORT: int = 31
_WS_SUBTYPE_FK_CROSSED: int = 32
_WS_SUBTYPE_FK_SHOT: int = 33
_WS_SUBTYPE_GOALKICK: int = 34
_WS_SUBTYPE_PENALTY: int = 35
_WS_SUBTYPE_THROW_IN: int = 36
_WS_SUBTYPE_BALL_OUT_OF_FIELD: int = 50
_WS_SUBTYPE_ACCELERATION: int = 70
_WS_SUBTYPE_CLEARANCE: int = 71
_WS_SUBTYPE_TOUCH: int = 72
_WS_SUBTYPE_CROSS: int = 80
_WS_SUBTYPE_THROW_IN_BP: int = 81  # throw-in (bodypart context)
_WS_SUBTYPE_HEAD_PASS_BP: int = 82  # headed pass (bodypart context)
_WS_SUBTYPE_SIMPLE_PASS: int = 85
_WS_SUBTYPE_LAUNCH: int = 90
_WS_SUBTYPE_HIGH_PASS: int = 91
_WS_SUBTYPE_GK_REFLEXES: int = 90
_WS_SUBTYPE_GK_SAVE: int = 91
_WS_SUBTYPE_GK_CLAIM: int = 92
_WS_SUBTYPE_GK_PUNCH: int = 93
_WS_SUBTYPE_SHOT_ON_TARGET: int = 100

# ---------------------------------------------------------------------------
# Wyscout tag IDs referenced as column names
# (tag_id -> column mappings live in wyscout_tags list below)
# ---------------------------------------------------------------------------
_WS_TAG_FAIRPLAY: int = 1001
_WS_TAG_OWN_GOAL: int = 102

EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "milliseconds",
    "team_id", "player_id", "type_id", "subtype_id",
    "positions", "tags",
}

_MAPPED_WS_TYPE_IDS: frozenset[int] = frozenset({
    _WS_TYPE_TAKE_ON,        # 0
    _WS_TYPE_DUEL,           # 1
    _WS_TYPE_FOUL,           # 2
    7,                        # Others on the ball
    _WS_TYPE_PASS,           # 8
    _WS_TYPE_GK,             # 9
    _WS_TYPE_SHOT,           # 10
})

_EXCLUDED_WS_TYPE_IDS: frozenset[int] = frozenset({
    3,   # Free kick
    4,   # Goal kick
    5,   # Interruption
    _WS_TYPE_OFFSIDE,  # 6
})


def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> tuple[pd.DataFrame, ConversionReport]:
    """
    Convert Wyscout events to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing Wyscout events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.

    """
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Wyscout")
    _event_type_counts = Counter(events["type_id"])

    events = pd.concat([events, _get_tagsdf(events)], axis=1)
    events = _make_new_positions(events)
    events = _fix_wyscout_events(events)
    actions = _create_df_actions(events)
    actions = _fix_actions(actions)
    actions = _fix_direction_of_play(actions, home_team_id)
    actions = _fix_clearances(actions)
    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)

    actions = _finalize_output(actions)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for ws_type_id, count in _event_type_counts.items():
        label = str(ws_type_id)
        if ws_type_id in _MAPPED_WS_TYPE_IDS:
            mapped_counts[label] = count
        elif ws_type_id in _EXCLUDED_WS_TYPE_IDS:
            excluded_counts[label] = count
        else:
            unrecognized_counts[label] = count
    if unrecognized_counts:
        warnings.warn(
            f"Wyscout: {sum(unrecognized_counts.values())} unrecognized event type_ids "
            f"dropped: {dict(unrecognized_counts)}"
        )
    report = ConversionReport(
        provider="Wyscout",
        total_events=sum(_event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report


def _get_tag_set(tags: list[dict[str, Any]]) -> set[int]:
    return {tag["id"] for tag in tags}


def _get_tagsdf(events: pd.DataFrame) -> pd.DataFrame:
    """Represent Wyscout tags as a boolean dataframe.

    Parameters
    ----------
    events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for each tag.
    """
    tags = events.tags.apply(_get_tag_set)
    tagsdf = pd.DataFrame()
    for tag_id, column in wyscout_tags:
        tagsdf[column] = tags.apply(lambda x, tag=tag_id: tag in x)
    return tagsdf


wyscout_tags = [
    (101, "goal"),
    (102, "own_goal"),
    (301, "assist"),
    (302, "key_pass"),
    (1901, "counter_attack"),
    (401, "left_foot"),
    (402, "right_foot"),
    (403, "head/body"),
    (1101, "direct"),
    (1102, "indirect"),
    (2001, "dangerous_ball_lost"),
    (2101, "blocked"),
    (801, "high"),
    (802, "low"),
    (1401, "interception"),
    (1501, "clearance"),
    (201, "opportunity"),
    (1301, "feint"),
    (1302, "missed_ball"),
    (501, "free_space_right"),
    (502, "free_space_left"),
    (503, "take_on_left"),
    (504, "take_on_right"),
    (1601, "sliding_tackle"),
    (601, "anticipated"),
    (602, "anticipation"),
    (1701, "red_card"),
    (1702, "yellow_card"),
    (1703, "second_yellow_card"),
    (1201, "position_goal_low_center"),
    (1202, "position_goal_low_right"),
    (1203, "position_goal_mid_center"),
    (1204, "position_goal_mid_left"),
    (1205, "position_goal_low_left"),
    (1206, "position_goal_mid_right"),
    (1207, "position_goal_high_center"),
    (1208, "position_goal_high_left"),
    (1209, "position_goal_high_right"),
    (1210, "position_out_low_right"),
    (1211, "position_out_mid_left"),
    (1212, "position_out_low_left"),
    (1213, "position_out_mid_right"),
    (1214, "position_out_high_center"),
    (1215, "position_out_high_left"),
    (1216, "position_out_high_right"),
    (1217, "position_post_low_right"),
    (1218, "position_post_mid_left"),
    (1219, "position_post_low_left"),
    (1220, "position_post_mid_right"),
    (1221, "position_post_high_center"),
    (1222, "position_post_high_left"),
    (1223, "position_post_high_right"),
    (901, "through"),
    (1001, "fairplay"),
    (701, "lost"),
    (702, "neutral"),
    (703, "won"),
    (1801, "accurate"),
    (1802, "not_accurate"),
]


def _make_position_vars(event_id: int, positions: list[dict[str, Optional[float]]]) -> pd.Series:
    if len(positions) == 2:  # if less than 2 then action is removed
        start_x = positions[0]["x"]
        start_y = positions[0]["y"]
        end_x = positions[1]["x"]
        end_y = positions[1]["y"]
    elif len(positions) == 1:
        start_x = positions[0]["x"]
        start_y = positions[0]["y"]
        end_x = start_x
        end_y = start_y
    else:
        start_x = None
        start_y = None
        end_x = None
        end_y = None
    return pd.Series([event_id, start_x, start_y, end_x, end_y])


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
    new_positions = events[["event_id", "positions"]].apply(
        lambda row: _make_position_vars(row["event_id"], row["positions"]), axis=1
    )
    new_positions.columns = ["event_id", "start_x", "start_y", "end_x", "end_y"]
    events = pd.merge(events, new_positions, left_on="event_id", right_on="event_id")
    events[["start_x", "end_x"]] = events[["start_x", "end_x"]].astype(float)
    events[["start_y", "end_y"]] = events[["start_y", "end_y"]].astype(float)
    events = events.drop("positions", axis=1)
    return events


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
        df_events["position_goal_mid_left"]
        | df_events["position_goal_low_left"]
        | df_events["position_goal_high_left"]
    )
    df_events.loc[shot & goal_left_idx, "end_x"] = 100.0
    df_events.loc[shot & goal_left_idx, "end_y"] = 45.0

    out_center_idx = df_events["position_out_high_center"] | df_events["position_post_high_center"]
    df_events.loc[shot & out_center_idx, "end_x"] = 100.0
    df_events.loc[shot & out_center_idx, "end_y"] = 50.0

    out_right_idx = (
        df_events["position_out_low_right"]
        | df_events["position_out_mid_right"]
        | df_events["position_out_high_right"]
    )
    df_events.loc[shot & out_right_idx, "end_x"] = 100.0
    df_events.loc[shot & out_right_idx, "end_y"] = 60.0

    out_left_idx = (
        df_events["position_out_mid_left"]
        | df_events["position_out_low_left"]
        | df_events["position_out_high_left"]
    )
    df_events.loc[shot & out_left_idx, "end_x"] = 100.0
    df_events.loc[shot & out_left_idx, "end_y"] = 40.0

    post_left_idx = (
        df_events["position_post_mid_left"]
        | df_events["position_post_low_left"]
        | df_events["position_post_high_left"]
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
    selector0_duel_won = selector_duel_out_of_field & (
        df_events["team_id"] != df_events2["team_id"]
    )
    selector0_duel_won_air = selector0_duel_won & (df_events["subtype_id"] == _WS_SUBTYPE_AIR_DUEL)
    selector0_duel_won_not_air = selector0_duel_won & (df_events["subtype_id"] != _WS_SUBTYPE_AIR_DUEL)

    # Define selectors for next time step
    selector1_duel_won = selector_duel_out_of_field & (
        df_events1["team_id"] != df_events2["team_id"]
    )
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

    # df_events.loc[selector_duel_won, 'end_x'] = df_events2.loc[selector_duel_won, 'start_x']
    # df_events.loc[selector_duel_won, 'end_y'] = df_events2.loc[selector_duel_won, 'start_y']

    # Define selector for ground attacking duels with take on
    selector_attacking_duel = df_events["subtype_id"] == _WS_SUBTYPE_GROUND_ATT_DUEL
    selector_take_on = (df_events["take_on_left"]) | (df_events["take_on_right"])
    selector_att_duel_take_on = selector_attacking_duel & selector_take_on

    # Set take ons type to 0
    df_events.loc[selector_att_duel_take_on, "type_id"] = _WS_TYPE_TAKE_ON

    # Set sliding tackles type to 0
    df_events.loc[df_events["sliding_tackle"], "type_id"] = _WS_TYPE_TAKE_ON

    # Remove the remaining duels
    df_events = df_events[df_events["type_id"] != _WS_TYPE_DUEL]

    # Reset the index
    df_events = df_events.reset_index(drop=True)

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
        df_events_interceptions[["end_x", "end_y"]] = df_events_interceptions[
            ["start_x", "start_y"]
        ]

        df_events = pd.concat([df_events_interceptions, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "milliseconds"], kind="mergesort")
        df_events = df_events.reset_index(drop=True)

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
    df_events = df_events[df_events["type_id"] != _WS_TYPE_OFFSIDE]

    # Reset index
    df_events = df_events.reset_index(drop=True)

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
        (prev_events["take_on_left"])
        | (prev_events["take_on_right"]) & prev_events["not_accurate"]
    )

    # Transform simulations not preceded by a failed take-on to a failed take-on
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "type_id"] = _WS_TYPE_TAKE_ON
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "subtype_id"] = _WS_TYPE_TAKE_ON
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "accurate"] = False
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "not_accurate"] = (
        True
    )
    # Set take_on_left or take_on_right to True
    df_events.loc[selector_simulation & ~selector_previous_is_failed_take_on, "take_on_left"] = (
        True
    )

    # Remove simulation events which are preceded by a failed take-on
    df_events = df_events[~(selector_simulation & selector_previous_is_failed_take_on)]

    # Reset index
    df_events = df_events.reset_index(drop=True)

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

    # df_events.loc[selector_touch_same_player & same_loc, 'subtype_id'] = 70
    # df_events.loc[selector_touch_same_player & same_loc, 'accurate'] = True
    # df_events.loc[selector_touch_same_player & same_loc, 'not_accurate'] = False

    df_events.loc[selector_touch_same_team & same_loc, "type_id"] = _WS_TYPE_PASS
    df_events.loc[selector_touch_same_team & same_loc, "subtype_id"] = _WS_SUBTYPE_SIMPLE_PASS
    df_events.loc[selector_touch_same_team & same_loc, "accurate"] = True
    df_events.loc[selector_touch_same_team & same_loc, "not_accurate"] = False

    df_events.loc[selector_touch_other & same_loc, "type_id"] = _WS_TYPE_PASS
    df_events.loc[selector_touch_other & same_loc, "subtype_id"] = _WS_SUBTYPE_SIMPLE_PASS
    df_events.loc[selector_touch_other & same_loc, "accurate"] = False
    df_events.loc[selector_touch_other & same_loc, "not_accurate"] = True

    return df_events


def _create_df_actions(df_events: pd.DataFrame) -> pd.DataFrame:
    """Create the SciSports action dataframe.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe
    """
    df_events["time_seconds"] = df_events["milliseconds"] / 1000
    df_actions = df_events[
        [
            "game_id",
            "period_id",
            "time_seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
    ].copy()
    df_actions["original_event_id"] = df_events["event_id"].astype(object)
    df_actions["bodypart_id"] = df_events.apply(_determine_bodypart_id, axis=1)
    df_actions["type_id"] = df_events.apply(_determine_type_id, axis=1)
    df_actions["result_id"] = df_events.apply(_determine_result_id, axis=1)

    df_actions = _remove_non_actions(df_actions)  # remove all non-actions left

    return df_actions


def _determine_bodypart_id(event: pd.DataFrame) -> int:
    """Determint eht body part for each action.

    Parameters
    ----------
    event : pd.Series
        Wyscout event Series

    Returns
    -------
    int
        id of the body part used for the action
    """
    if event["subtype_id"] in [
        _WS_SUBTYPE_THROW_IN_BP, _WS_SUBTYPE_THROW_IN, _WS_SUBTYPE_HEAD_PASS,
        _WS_SUBTYPE_LAUNCH, _WS_SUBTYPE_HIGH_PASS,
    ]:
        body_part = "other"
    elif event["subtype_id"] == _WS_SUBTYPE_HEAD_PASS_BP:
        body_part = "head"
    elif event["type_id"] == _WS_TYPE_SHOT and event["head/body"]:
        body_part = "head/other"
    elif event["left_foot"]:
        body_part = "foot_left"
    elif event["right_foot"]:
        body_part = "foot_right"
    else:  # all other cases
        body_part = "foot"
    return spadlconfig.bodypart_id[body_part]


def _determine_type_id(event: pd.DataFrame) -> int:  # noqa: C901
    """Determine the type of each action.

    This function transforms the Wyscout events, sub_events and tags
    into the corresponding SciSports action type

    Parameters
    ----------
    event : pd.Series
        A series from the Wyscout event dataframe

    Returns
    -------
    int
        id of the action type
    """
    if event["fairplay"]:
        action_type = "non_action"
    elif event["own_goal"]:
        action_type = "bad_touch"
    elif event["type_id"] == _WS_TYPE_PASS:
        if event["subtype_id"] == _WS_SUBTYPE_CROSS:
            action_type = "cross"
        else:
            action_type = "pass"
    elif event["subtype_id"] == _WS_SUBTYPE_THROW_IN:
        action_type = "throw_in"
    elif event["subtype_id"] == _WS_SUBTYPE_CORNER:
        if event["high"]:
            action_type = "corner_crossed"
        else:
            action_type = "corner_short"
    elif event["subtype_id"] == _WS_SUBTYPE_FK_CROSSED:
        action_type = "freekick_crossed"
    elif event["subtype_id"] == _WS_SUBTYPE_FK_SHORT:
        action_type = "freekick_short"
    elif event["subtype_id"] == _WS_SUBTYPE_GOALKICK:
        action_type = "goalkick"
    elif event["type_id"] == _WS_TYPE_FOUL and (event["subtype_id"] not in [
        _WS_SUBTYPE_HAND_FOUL, _WS_SUBTYPE_LATE_CARD_FOUL,
        _WS_SUBTYPE_OUT_OF_GAME_FOUL, _WS_SUBTYPE_VIOLENT_FOUL,
    ]):
        action_type = "foul"
    elif event["type_id"] == _WS_TYPE_SHOT:
        action_type = "shot"
    elif event["subtype_id"] == _WS_SUBTYPE_PENALTY:
        action_type = "shot_penalty"
    elif event["subtype_id"] == _WS_SUBTYPE_FK_SHOT:
        action_type = "shot_freekick"
    elif event["type_id"] == _WS_TYPE_GK:
        subtype = event["subtype_id"]
        if subtype == _WS_SUBTYPE_GK_CLAIM:
            action_type = "keeper_claim"
        elif subtype == _WS_SUBTYPE_GK_PUNCH:
            action_type = "keeper_punch"
        else:
            action_type = "keeper_save"
    elif event["subtype_id"] == _WS_SUBTYPE_CLEARANCE:
        action_type = "clearance"
    elif event["subtype_id"] == _WS_SUBTYPE_TOUCH and event["not_accurate"]:
        action_type = "bad_touch"
    elif event["subtype_id"] == _WS_SUBTYPE_ACCELERATION:
        action_type = "dribble"
    elif event["take_on_left"] or event["take_on_right"]:
        action_type = "take_on"
    elif event["sliding_tackle"]:
        action_type = "tackle"
    elif event["interception"] and (event["subtype_id"] in [
        _WS_TYPE_TAKE_ON, _WS_SUBTYPE_AIR_DUEL, _WS_SUBTYPE_GROUND_ATT_DUEL,
        _WS_SUBTYPE_GROUND_DEF_DUEL, _WS_SUBTYPE_GROUND_LOOSE_BALL, _WS_SUBTYPE_TOUCH,
    ]):
        action_type = "interception"
    else:
        action_type = "non_action"
    return spadlconfig.actiontype_id[action_type]


def _determine_result_id(event: pd.DataFrame) -> int:  # noqa: C901
    """Determine the result of each event.

    Parameters
    ----------
    event : pd.Series
        Wyscout event Series

    Returns
    -------
    int
        result of the action
    """
    if event["offside"] == 1:
        return spadlconfig.result_id["offside"]
    if event["type_id"] == _WS_TYPE_FOUL:
        if event["yellow_card"]:
            return spadlconfig.result_id["yellow_card"]
        elif event["second_yellow_card"] or event["red_card"]:
            return spadlconfig.result_id["red_card"]
        return spadlconfig.result_id["fail"]
    if event["goal"]:
        return spadlconfig.result_id["success"]
    if event["own_goal"]:
        return spadlconfig.result_id["owngoal"]
    if event["subtype_id"] in [_WS_SUBTYPE_SHOT_ON_TARGET, _WS_SUBTYPE_FK_SHOT, _WS_SUBTYPE_PENALTY]:
        return spadlconfig.result_id["fail"]
    if event["accurate"]:
        return spadlconfig.result_id["success"]
    if event["not_accurate"]:
        return spadlconfig.result_id["fail"]
    if (
        event["interception"] or event["clearance"] or event["subtype_id"] == _WS_SUBTYPE_CLEARANCE
    ):  # interception or clearance always success
        return spadlconfig.result_id["success"]
    if event["type_id"] == _WS_TYPE_GK:  # keeper save always success
        return spadlconfig.result_id["success"]
    # no idea, assume it was successful
    return spadlconfig.result_id["success"]


def _remove_non_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Remove the remaining non_actions from the action dataframe.

    Parameters
    ----------
    df_actions : pd.DataFrame
        SciSports action dataframe

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe without non-actions
    """
    df_actions = df_actions[df_actions["type_id"] != spadlconfig.actiontype_id["non_action"]]
    # remove remaining ball out of field, whistle and goalkeeper from line
    df_actions = df_actions.reset_index(drop=True)
    return df_actions


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
    df_actions["start_x"] = (df_actions["start_x"] * spadlconfig.field_length / 100).clip(
        0, spadlconfig.field_length
    )
    df_actions["start_y"] = (
        (100 - df_actions["start_y"]) * spadlconfig.field_width / 100
        # y is from top to bottom in Wyscout
    ).clip(0, spadlconfig.field_width)
    df_actions["end_x"] = (df_actions["end_x"] * spadlconfig.field_length / 100).clip(
        0, spadlconfig.field_length
    )
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
    df_actions.loc[saves_idx, "end_x"] = (
        spadlconfig.field_length - df_actions.loc[saves_idx, "end_x"]
    )
    df_actions.loc[saves_idx, "end_y"] = (
        spadlconfig.field_width - df_actions.loc[saves_idx, "end_y"]
    )
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
    shot_goals = (prev_actions.type_id == spadlconfig.actiontype_id["shot"]) & (
        prev_actions.result_id == 1
    )
    penalty_goals = (prev_actions.type_id == spadlconfig.actiontype_id["shot_penalty"]) & (
        prev_actions.result_id == 1
    )
    freekick_goals = (prev_actions.type_id == spadlconfig.actiontype_id["shot_freekick"]) & (
        prev_actions.result_id == 1
    )
    goals = shot_goals | penalty_goals | freekick_goals
    keeper_save = df_actions["type_id"] == spadlconfig.actiontype_id["keeper_save"]
    goals_keepers_idx = same_phase & goals & keeper_save
    df_actions = df_actions.drop(df_actions.index[goals_keepers_idx])
    df_actions = df_actions.reset_index(drop=True)

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
