"""Wyscout-to-SPADL action mapping: vectorized type/result/bodypart dispatch."""

import numpy as np
import pandas as pd

from . import config as spadlconfig
from .wyscout import (
    _WS_SUBTYPE_ACCELERATION,
    _WS_SUBTYPE_AIR_DUEL,
    _WS_SUBTYPE_CLEARANCE,
    _WS_SUBTYPE_CORNER,
    _WS_SUBTYPE_CROSS,
    _WS_SUBTYPE_FK_CROSSED,
    _WS_SUBTYPE_FK_SHORT,
    _WS_SUBTYPE_FK_SHOT,
    _WS_SUBTYPE_GK_CLAIM,
    _WS_SUBTYPE_GK_PUNCH,
    _WS_SUBTYPE_GOALKICK,
    _WS_SUBTYPE_GROUND_ATT_DUEL,
    _WS_SUBTYPE_GROUND_DEF_DUEL,
    _WS_SUBTYPE_GROUND_LOOSE_BALL,
    _WS_SUBTYPE_HAND_FOUL,
    _WS_SUBTYPE_HEAD_PASS,
    _WS_SUBTYPE_HEAD_PASS_BP,
    _WS_SUBTYPE_HIGH_PASS,
    _WS_SUBTYPE_LATE_CARD_FOUL,
    _WS_SUBTYPE_LAUNCH,
    _WS_SUBTYPE_OUT_OF_GAME_FOUL,
    _WS_SUBTYPE_PENALTY,
    _WS_SUBTYPE_SHOT_ON_TARGET,
    _WS_SUBTYPE_THROW_IN,
    _WS_SUBTYPE_THROW_IN_BP,
    _WS_SUBTYPE_TOUCH,
    _WS_SUBTYPE_VIOLENT_FOUL,
    _WS_TYPE_FOUL,
    _WS_TYPE_GK,
    _WS_TYPE_PASS,
    _WS_TYPE_SHOT,
    _WS_TYPE_TAKE_ON,
)

# ---------------------------------------------------------------------------
# Vectorized dispatch functions (replace row-wise apply)
# ---------------------------------------------------------------------------


def _vectorized_bodypart_id(df_events: pd.DataFrame) -> pd.Series:
    """Determine the body part for each action (vectorized).

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.Series
        Body part id for each action
    """
    bid = spadlconfig.bodypart_id
    sid = df_events["subtype_id"]
    tid = df_events["type_id"]

    conditions = [
        sid.isin(
            [
                _WS_SUBTYPE_THROW_IN_BP,
                _WS_SUBTYPE_THROW_IN,
                _WS_SUBTYPE_HEAD_PASS,
                _WS_SUBTYPE_LAUNCH,
                _WS_SUBTYPE_HIGH_PASS,
            ]
        ),
        sid == _WS_SUBTYPE_HEAD_PASS_BP,
        (tid == _WS_TYPE_SHOT) & df_events["head/body"],
        df_events["left_foot"],
        df_events["right_foot"],
    ]
    choices = [
        bid["other"],
        bid["head"],
        bid["head/other"],
        bid["foot_left"],
        bid["foot_right"],
    ]
    return pd.Series(
        np.select(conditions, choices, default=bid["foot"]),
        index=df_events.index,
        dtype="int64",
    )


def _vectorized_type_id(df_events: pd.DataFrame) -> pd.Series:
    """Determine the type of each action (vectorized).

    Translates the Wyscout events, sub_events and tags into the
    corresponding SPADL action type.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.Series
        Type id for each action
    """
    aid = spadlconfig.actiontype_id
    tid = df_events["type_id"]
    sid = df_events["subtype_id"]

    # Build conditions in the same priority order as the original if/elif chain
    conditions = [
        df_events["fairplay"],
        df_events["own_goal"],
        (tid == _WS_TYPE_PASS) & (sid == _WS_SUBTYPE_CROSS),
        tid == _WS_TYPE_PASS,
        sid == _WS_SUBTYPE_THROW_IN,
        (sid == _WS_SUBTYPE_CORNER) & df_events["high"],
        sid == _WS_SUBTYPE_CORNER,
        sid == _WS_SUBTYPE_FK_CROSSED,
        sid == _WS_SUBTYPE_FK_SHORT,
        sid == _WS_SUBTYPE_GOALKICK,
        (tid == _WS_TYPE_FOUL)
        & ~sid.isin(
            [
                _WS_SUBTYPE_HAND_FOUL,
                _WS_SUBTYPE_LATE_CARD_FOUL,
                _WS_SUBTYPE_OUT_OF_GAME_FOUL,
                _WS_SUBTYPE_VIOLENT_FOUL,
            ]
        ),
        tid == _WS_TYPE_SHOT,
        sid == _WS_SUBTYPE_PENALTY,
        sid == _WS_SUBTYPE_FK_SHOT,
        (tid == _WS_TYPE_GK) & (sid == _WS_SUBTYPE_GK_CLAIM),
        (tid == _WS_TYPE_GK) & (sid == _WS_SUBTYPE_GK_PUNCH),
        tid == _WS_TYPE_GK,
        sid == _WS_SUBTYPE_CLEARANCE,
        (sid == _WS_SUBTYPE_TOUCH) & df_events["not_accurate"],
        sid == _WS_SUBTYPE_ACCELERATION,
        df_events["take_on_left"] | df_events["take_on_right"],
        df_events["sliding_tackle"],
        df_events["interception"]
        & sid.isin(
            [
                _WS_TYPE_TAKE_ON,
                _WS_SUBTYPE_AIR_DUEL,
                _WS_SUBTYPE_GROUND_ATT_DUEL,
                _WS_SUBTYPE_GROUND_DEF_DUEL,
                _WS_SUBTYPE_GROUND_LOOSE_BALL,
                _WS_SUBTYPE_TOUCH,
            ]
        ),
    ]
    choices = [
        aid["non_action"],
        aid["bad_touch"],
        aid["cross"],
        aid["pass"],
        aid["throw_in"],
        aid["corner_crossed"],
        aid["corner_short"],
        aid["freekick_crossed"],
        aid["freekick_short"],
        aid["goalkick"],
        aid["foul"],
        aid["shot"],
        aid["shot_penalty"],
        aid["shot_freekick"],
        aid["keeper_claim"],
        aid["keeper_punch"],
        aid["keeper_save"],
        aid["clearance"],
        aid["bad_touch"],
        aid["dribble"],
        aid["take_on"],
        aid["tackle"],
        aid["interception"],
    ]
    return pd.Series(
        np.select(conditions, choices, default=aid["non_action"]),
        index=df_events.index,
        dtype="int64",
    )


def _vectorized_result_id(df_events: pd.DataFrame) -> pd.Series:
    """Determine the result of each event (vectorized).

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe

    Returns
    -------
    pd.Series
        Result id for each action
    """
    rid = spadlconfig.result_id
    tid = df_events["type_id"]
    sid = df_events["subtype_id"]

    # Build conditions in the same priority order as the original if chain.
    # The original function returns early, so later conditions must exclude
    # rows already matched by earlier conditions.
    conditions = [
        df_events["offside"] == 1,
        (tid == _WS_TYPE_FOUL) & df_events["yellow_card"],
        (tid == _WS_TYPE_FOUL) & (df_events["second_yellow_card"] | df_events["red_card"]),
        tid == _WS_TYPE_FOUL,
        df_events["goal"],
        df_events["own_goal"],
        sid.isin([_WS_SUBTYPE_SHOT_ON_TARGET, _WS_SUBTYPE_FK_SHOT, _WS_SUBTYPE_PENALTY]),
        df_events["accurate"],
        df_events["not_accurate"],
        df_events["interception"] | df_events["clearance"] | (sid == _WS_SUBTYPE_CLEARANCE),
        tid == _WS_TYPE_GK,
    ]
    choices = [
        rid["offside"],
        rid["yellow_card"],
        rid["red_card"],
        rid["fail"],
        rid["success"],
        rid["owngoal"],
        rid["fail"],
        rid["success"],
        rid["fail"],
        rid["success"],
        rid["success"],
    ]
    return pd.Series(
        np.select(conditions, choices, default=rid["success"]),
        index=df_events.index,
        dtype="int64",
    )


# ---------------------------------------------------------------------------
# Action DataFrame construction
# ---------------------------------------------------------------------------


def _create_df_actions(
    df_events: pd.DataFrame,
    preserve_native: list[str] | None = None,
) -> pd.DataFrame:
    """Create the SciSports action dataframe.

    Parameters
    ----------
    df_events : pd.DataFrame
        Wyscout event dataframe
    preserve_native : list[str], optional
        Provider-native fields to copy from ``df_events`` onto the output
        actions df alongside the canonical SPADL columns. Used by the
        public ``convert_to_actions(preserve_native=...)`` parameter.

    Returns
    -------
    pd.DataFrame
        SciSports action dataframe
    """
    df_events["time_seconds"] = df_events["milliseconds"] / 1000
    base_cols = [
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
    extras = list(preserve_native) if preserve_native else []
    df_actions = df_events[[*base_cols, *extras]].copy()
    df_actions["original_event_id"] = df_events["event_id"].astype(object)
    df_actions["bodypart_id"] = _vectorized_bodypart_id(df_events)
    df_actions["type_id"] = _vectorized_type_id(df_events)
    df_actions["result_id"] = _vectorized_result_id(df_events)

    df_actions = _remove_non_actions(df_actions)

    return df_actions


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
    df_actions = df_actions[df_actions["type_id"] != spadlconfig.actiontype_id["non_action"]]  # type: ignore[reportAssignmentType]
    # remove remaining ball out of field, whistle and goalkeeper from line
    df_actions = df_actions.reset_index(drop=True)  # type: ignore[reportAssignmentType]
    return df_actions
