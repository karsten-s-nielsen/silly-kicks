"""Opta event stream data to SPADL converter."""

import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_direction_of_play,
    min_dribble_length,
)
from .schema import ConversionReport
from .utils import _finalize_output, _validate_input_columns, _validate_preserve_native

EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id",
    "event_id",
    "period_id",
    "minute",
    "second",
    "team_id",
    "player_id",
    "type_name",
    "outcome",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "qualifiers",
}

_MAPPED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "pass",
        "offside pass",
        "take on",
        "foul",
        "tackle",
        "interception",
        "blocked pass",
        "miss",
        "post",
        "attempt saved",
        "goal",
        "save",
        "claim",
        "punch",
        "keeper pick-up",
        "clearance",
        "card",
        "ball touch",
        "ball recovery",
    }
)

_EXCLUDED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "end",
        "start",
        "formation change",
        "resume",
        "deleted event",
        "shield ball opp",
        "offside provoked",
        "player off",
        "player on",
        "player retired",
        "chance missed",
        "attendance",
        "referee stop",
        "referee drop ball",
        "50/50",
        "cross not claimed",
        "goalkeeper position",
    }
)


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """
    Convert Opta events to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing Opta events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.
    preserve_native : list[str], optional
        Provider-native event fields to preserve alongside the canonical SPADL
        output as extra columns. Each field must be present on the input
        ``events`` DataFrame and must not overlap with the SPADL schema.
        Synthetic actions inserted by ``_add_dribbles`` get NaN.
    goalkeeper_ids : set[int] or None, default ``None``
        Accepted for cross-provider API symmetry with the
        ``sportec`` / ``metrica`` converters (silly-kicks 1.10.0+); has
        no effect on Opta output because Opta's source events natively
        mark GK actions via the dedicated ``save`` / ``claim`` /
        ``punch`` / ``keeper pick-up`` event types. The parameter is
        silently accepted; the output is byte-for-byte identical with
        and without it. Empty set ≡ ``None``.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions, plus any
        ``preserve_native`` columns appended.

    Examples
    --------
    Preserve a top-level Opta field alongside the SPADL output::

        actions, report = convert_to_actions(
            events, home_team_id=157, preserve_native=["competition_id"]
        )

    """
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Opta")
    _validate_preserve_native(events, preserve_native, provider="Opta")
    # goalkeeper_ids: accepted for cross-provider API symmetry; no-op for
    # Opta because source events natively mark GK actions.
    _ = goalkeeper_ids
    _event_type_counts = Counter(events["type_name"])

    actions = pd.DataFrame()

    actions["game_id"] = events.game_id
    actions["original_event_id"] = events.event_id.astype(object)
    actions["period_id"] = events.period_id

    actions["time_seconds"] = (
        60 * events.minute
        + events.second
        - ((events.period_id > 1) * 45 * 60)
        - ((events.period_id > 2) * 45 * 60)
        - ((events.period_id > 3) * 15 * 60)
        - ((events.period_id > 4) * 15 * 60)
    )
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    for col in ["start_x", "end_x"]:
        actions[col] = events[col].clip(0, 100) / 100 * spadlconfig.field_length
    for col in ["start_y", "end_y"]:
        actions[col] = events[col].clip(0, 100) / 100 * spadlconfig.field_width

    # Provider-native passthroughs (per preserve_native): copy alongside the
    # canonical source-to-action columns so they survive filter/sort/dribble
    # insertion. Synthetic dribbles produced downstream by _add_dribbles get
    # NaN in these columns automatically.
    for _col in preserve_native or []:
        actions[_col] = events[_col].values

    # Pre-explode qualifier flags for vectorized lookups
    _used_qualifier_ids = [1, 2, 3, 5, 6, 9, 15, 20, 21, 26, 28, 32, 72, 107, 124, 155, 168, 238]
    qual_sets = [set(q.keys()) if isinstance(q, dict) else set() for q in events["qualifiers"]]
    qual_data = np.array([[qid in qs for qid in _used_qualifier_ids] for qs in qual_sets], dtype=bool)
    qual_df = pd.DataFrame(qual_data, columns=[f"q_{qid}" for qid in _used_qualifier_ids], index=events.index)
    for col in qual_df.columns:
        events[col] = qual_df[col]

    actions["type_id"] = _vectorized_type_id(events)
    actions["result_id"] = _vectorized_result_id(events)
    actions["bodypart_id"] = _vectorized_bodypart_id(events)

    actions = _fix_recoveries(actions, events.type_name)
    actions = _fix_unintentional_ball_touches(actions, events.type_name, events.outcome)
    actions = (
        actions[actions.type_id != spadlconfig.actiontype_id["non_action"]]
        .sort_values(["game_id", "period_id", "time_seconds"], kind="mergesort")  # type: ignore[reportCallIssue]
        .reset_index(drop=True)
    )
    actions = _fix_owngoals(actions)
    actions = _fix_direction_of_play(actions, home_team_id)
    actions = _fix_clearances(actions)
    actions = _fix_interceptions(actions)
    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)

    actions = _finalize_output(actions, extra_columns=preserve_native)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for etype, count in _event_type_counts.items():
        if etype in _MAPPED_EVENT_TYPES:
            mapped_counts[etype] = count
        elif etype in _EXCLUDED_EVENT_TYPES:
            excluded_counts[etype] = count
        else:
            unrecognized_counts[etype] = count
    if unrecognized_counts:
        warnings.warn(
            f"Opta: {sum(unrecognized_counts.values())} unrecognized event types dropped: {dict(unrecognized_counts)}",
            stacklevel=2,
        )
    report = ConversionReport(
        provider="Opta",
        total_events=sum(_event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report


def _vectorized_type_id(events: pd.DataFrame) -> pd.Series:
    """Vectorized replacement for row-wise _get_type_id."""
    tn = events["type_name"]
    outcome = events["outcome"]
    aid = spadlconfig.actiontype_id
    is_pass = tn.isin(["pass", "offside pass"])
    is_shot = tn.isin(["miss", "post", "attempt saved", "goal"])

    conditions = [
        events["q_238"],  # fairplay → non_action
        is_pass & events["q_107"],  # throw_in
        is_pass & events["q_5"] & (events["q_2"] | events["q_1"] | events["q_155"]),  # freekick_crossed
        is_pass & events["q_5"],  # freekick_short
        is_pass & events["q_6"] & events["q_2"],  # corner_crossed
        is_pass & events["q_6"],  # corner_short
        is_pass & events["q_2"],  # cross
        is_pass & events["q_124"],  # goalkick
        is_pass,  # pass (default for pass events)
        tn == "take on",  # take_on
        (tn == "foul") & (outcome == False),  # foul  # noqa: E712
        tn == "tackle",  # tackle
        tn.isin(["interception", "blocked pass"]),  # interception
        is_shot & events["q_9"],  # shot_penalty
        is_shot & events["q_26"],  # shot_freekick
        is_shot,  # shot
        tn == "save",  # keeper_save
        tn == "claim",  # keeper_claim
        tn == "punch",  # keeper_punch
        tn == "keeper pick-up",  # keeper_pick_up
        tn == "clearance",  # clearance
        tn == "card",  # foul (card events are mapped to foul)
        (tn == "ball touch") & (outcome == False),  # bad_touch  # noqa: E712
    ]
    choices = [
        aid["non_action"],
        aid["throw_in"],
        aid["freekick_crossed"],
        aid["freekick_short"],
        aid["corner_crossed"],
        aid["corner_short"],
        aid["cross"],
        aid["goalkick"],
        aid["pass"],
        aid["take_on"],
        aid["foul"],
        aid["tackle"],
        aid["interception"],
        aid["shot_penalty"],
        aid["shot_freekick"],
        aid["shot"],
        aid["keeper_save"],
        aid["keeper_claim"],
        aid["keeper_punch"],
        aid["keeper_pick_up"],
        aid["clearance"],
        aid["foul"],
        aid["bad_touch"],
    ]
    return pd.Series(
        np.select(conditions, choices, default=aid["non_action"]),
        index=events.index,
        dtype="int64",
    )


def _vectorized_result_id(events: pd.DataFrame) -> pd.Series:
    """Vectorized replacement for row-wise _get_result_id."""
    tn = events["type_name"]
    outcome = events["outcome"]
    rid = spadlconfig.result_id

    conditions = [
        tn == "offside pass",  # offside
        (tn == "card") & events["q_32"],  # red_card
        tn == "card",  # yellow_card (no q_32)
        tn == "foul",  # fail
        tn.isin(["attempt saved", "miss", "post"]),  # fail
        (tn == "goal") & events["q_28"],  # owngoal
        tn == "goal",  # success
        tn == "ball touch",  # fail
        outcome == True,  # success  # noqa: E712
    ]
    choices = [
        rid["offside"],
        rid["red_card"],
        rid["yellow_card"],
        rid["fail"],
        rid["fail"],
        rid["owngoal"],
        rid["success"],
        rid["fail"],
        rid["success"],
    ]
    return pd.Series(
        np.select(conditions, choices, default=rid["fail"]),
        index=events.index,
        dtype="int64",
    )


def _vectorized_bodypart_id(events: pd.DataFrame) -> pd.Series:
    """Vectorized replacement for row-wise _get_bodypart_id."""
    tn = events["type_name"]
    bid = spadlconfig.bodypart_id

    conditions = [
        events["q_15"] | events["q_3"] | events["q_168"],  # head
        events["q_21"],  # other
        events["q_20"],  # foot_right
        events["q_72"],  # foot_left
        events["q_107"],  # other (throw-in)
        tn.isin(["save", "claim", "punch", "keeper pick-up"]),  # other
    ]
    choices = [
        bid["head"],
        bid["other"],
        bid["foot_right"],
        bid["foot_left"],
        bid["other"],
        bid["other"],
    ]
    return pd.Series(
        np.select(conditions, choices, default=bid["foot"]),
        index=events.index,
        dtype="int64",
    )


def _get_bodypart_id(args: tuple[str, bool, dict[int, Any]]) -> int:
    e, _outcome, q = args
    if 15 in q or 3 in q or 168 in q:
        b = "head"
    elif 21 in q:
        b = "other"
    elif 20 in q:
        b = "foot_right"
    elif 72 in q:
        b = "foot_left"
    elif 107 in q:  # throw-in
        b = "other"
    else:
        if e in ["save", "claim", "punch", "keeper pick-up"]:
            b = "other"
        else:
            b = "foot"
    return spadlconfig.bodypart_id[b]


def _get_result_id(args: tuple[str, bool, dict[int, Any]]) -> int:
    e, outcome, q = args
    if e == "offside pass":
        r = "offside"  # offside
    elif e == "card":
        if 32 in q:
            r = "red_card"
        else:
            r = "yellow_card"
    elif e == "foul":
        r = "fail"
    elif e in ["attempt saved", "miss", "post"]:
        r = "fail"
    elif e == "goal":
        if 28 in q:
            r = "owngoal"  # own goal, x and y must be switched
        else:
            r = "success"
    elif e == "ball touch":
        r = "fail"
    elif outcome:
        r = "success"
    else:
        r = "fail"
    return spadlconfig.result_id[r]


def _get_type_id(args: tuple[str, bool, dict[int, Any]]) -> int:
    eventname, outcome, q = args
    fairplay = 238 in q
    if fairplay:
        a = "non_action"
    elif eventname in ("pass", "offside pass"):
        cross = 2 in q
        longball = 1 in q
        chipped = 155 in q
        freekick = 5 in q
        corner = 6 in q
        throw_in = 107 in q
        goalkick = 124 in q
        if throw_in:
            a = "throw_in"
        elif freekick and (cross or longball or chipped):
            a = "freekick_crossed"
        elif freekick:
            a = "freekick_short"
        elif corner and cross:
            a = "corner_crossed"
        elif corner:
            a = "corner_short"
        elif cross:
            a = "cross"
        elif goalkick:
            a = "goalkick"
        else:
            a = "pass"
    elif eventname == "take on":
        a = "take_on"
    elif eventname == "foul" and outcome is False:
        a = "foul"
    elif eventname == "tackle":
        a = "tackle"
    elif eventname in ("interception", "blocked pass"):
        a = "interception"
    elif eventname in ["miss", "post", "attempt saved", "goal"]:
        if 9 in q:
            a = "shot_penalty"
        elif 26 in q:
            a = "shot_freekick"
        else:
            a = "shot"
    elif eventname == "save":
        a = "keeper_save"
    elif eventname == "claim":
        a = "keeper_claim"
    elif eventname == "punch":
        a = "keeper_punch"
    elif eventname == "keeper pick-up":
        a = "keeper_pick_up"
    elif eventname == "clearance":
        a = "clearance"
    elif eventname == "card":
        a = "foul"
    elif eventname == "ball touch" and outcome is False:
        a = "bad_touch"
    else:
        a = "non_action"
    return spadlconfig.actiontype_id[a]


def _fix_owngoals(actions: pd.DataFrame) -> pd.DataFrame:
    owngoals_idx = (actions.result_id == spadlconfig.result_id["owngoal"]) & (
        actions.type_id == spadlconfig.actiontype_id["shot"]
    )
    actions.loc[owngoals_idx, "end_x"] = spadlconfig.field_length - actions[owngoals_idx].end_x.to_numpy()
    actions.loc[owngoals_idx, "end_y"] = spadlconfig.field_width - actions[owngoals_idx].end_y.to_numpy()
    actions.loc[owngoals_idx, "type_id"] = spadlconfig.actiontype_id["bad_touch"]
    return actions


def _fix_recoveries(df_actions: pd.DataFrame, opta_types: pd.Series) -> pd.DataFrame:
    """Convert ball recovery events to dribbles.

    This function converts the Opta 'ball recovery' event (type_id 49) into
    a dribble.

    Parameters
    ----------
    df_actions : pd.DataFrame
        Opta actions dataframe
    opta_types : pd.Series
        Original Opta event types

    Returns
    -------
    pd.DataFrame
        Opta event dataframe without any ball recovery events
    """
    df_actions_next = df_actions.shift(-1)
    df_actions_next = df_actions_next.mask(df_actions_next.type_id == spadlconfig.actiontype_id["non_action"]).bfill()

    selector_recovery = opta_types == "ball recovery"

    same_x = abs(df_actions["end_x"] - df_actions_next["start_x"]) < min_dribble_length
    same_y = abs(df_actions["end_y"] - df_actions_next["start_y"]) < min_dribble_length
    same_loc = same_x & same_y

    df_actions.loc[selector_recovery & ~same_loc, "type_id"] = spadlconfig.actiontype_id["dribble"]
    df_actions.loc[selector_recovery & same_loc, "type_id"] = spadlconfig.actiontype_id["non_action"]
    df_actions.loc[selector_recovery, ["end_x", "end_y"]] = df_actions_next.loc[
        selector_recovery, ["start_x", "start_y"]
    ].values

    return df_actions


def _fix_interceptions(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Set the result of interceptions to 'fail' if they do not regain possession.

    Parameters
    ----------
    df_actions : pd.DataFrame
        Opta actions dataframe.

    Returns
    -------
    pd.DataFrame
        Opta event dataframe without any ball recovery events
    """
    mask_interception = df_actions.type_id == spadlconfig.actiontype_id["interception"]
    same_team = df_actions.team_id == df_actions.shift(-1).team_id
    df_actions.loc[mask_interception & ~same_team, "result_id"] = spadlconfig.result_id["fail"]
    return df_actions


def _fix_unintentional_ball_touches(
    df_actions: pd.DataFrame, opta_type: pd.Series, opta_outcome: pd.Series
) -> pd.DataFrame:
    """Discard unintentional ball touches.

    Passes that are deflected but still reach their target are registered as
    successful passes. The (unintentional) deflection is not recored as an
    action, because players should not be credited for it.

    Parameters
    ----------
    df_actions : pd.DataFrame
        Opta actions dataframe
    opta_type : pd.Series
        Original Opta event types
    opta_outcome : pd.Series
        Original Opta event outcomes

    Returns
    -------
    pd.DataFrame
        Opta event dataframe without any unintentional ball touches.
    """
    df_actions_next = df_actions.shift(-2)
    selector_pass = df_actions["type_id"] == spadlconfig.actiontype_id["pass"]
    selector_deflected = (opta_type.shift(-1) == "ball touch") & (opta_outcome.shift(-1))
    selector_same_team = df_actions["team_id"] == df_actions_next["team_id"]
    df_actions.loc[selector_deflected, ["end_x", "end_y"]] = df_actions_next.loc[
        selector_deflected, ["start_x", "start_y"]
    ].values
    df_actions.loc[selector_pass & selector_deflected & selector_same_team, "result_id"] = spadlconfig.result_id[
        "success"
    ]
    return df_actions
