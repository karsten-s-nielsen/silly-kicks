"""Wyscout event stream data to SPADL converter."""

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_direction_of_play,
)
from .schema import ConversionReport
from .utils import _finalize_output, _validate_input_columns

# ---------------------------------------------------------------------------
# Wyscout event type IDs
# ---------------------------------------------------------------------------
_WS_TYPE_TAKE_ON: int = 0  # synthetic type assigned to take-ons / tackles
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
    "game_id",
    "event_id",
    "period_id",
    "milliseconds",
    "team_id",
    "player_id",
    "type_id",
    "subtype_id",
    "positions",
    "tags",
}

_MAPPED_WS_TYPE_IDS: frozenset[int] = frozenset(
    {
        _WS_TYPE_TAKE_ON,  # 0
        _WS_TYPE_DUEL,  # 1
        _WS_TYPE_FOUL,  # 2
        7,  # Others on the ball
        _WS_TYPE_PASS,  # 8
        _WS_TYPE_GK,  # 9
        _WS_TYPE_SHOT,  # 10
    }
)

_EXCLUDED_WS_TYPE_IDS: frozenset[int] = frozenset(
    {
        3,  # Free kick
        4,  # Goal kick
        5,  # Interruption
        _WS_TYPE_OFFSIDE,  # 6
    }
)


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


# ---------------------------------------------------------------------------
# Lazy imports from submodules (avoids circular import at module level)
# ---------------------------------------------------------------------------


def _lazy_import_events():
    from ._wyscout_events import _fix_actions, _fix_wyscout_events, _make_new_positions

    return _fix_wyscout_events, _make_new_positions, _fix_actions


def _lazy_import_mappings():
    from ._wyscout_mappings import _create_df_actions

    return (_create_df_actions,)


# ---------------------------------------------------------------------------
# Tag extraction (optimized batch constructor)
# ---------------------------------------------------------------------------


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
    tag_ids = [tid for tid, _ in wyscout_tags]
    tag_names = [name for _, name in wyscout_tags]
    # Single pass: convert each row's tag list to a set, then build boolean matrix
    tag_set_list = [{tag["id"] for tag in tags} for tags in events.tags]
    data = np.array([[tid in ts for tid in tag_ids] for ts in tag_set_list], dtype=bool)
    return pd.DataFrame(data, columns=tag_names, index=events.index)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    goalkeeper_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """
    Convert Wyscout events to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing Wyscout events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.
    goalkeeper_ids : set[int] or None, default=None
        If provided, aerial duels by these player IDs are mapped to
        ``keeper_claim`` instead of the default duel dispatch.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.

    """
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Wyscout")
    _event_type_counts = Counter(events["type_id"])

    _fix_wyscout_events, _make_new_positions, _fix_actions = _lazy_import_events()
    (_create_df_actions,) = _lazy_import_mappings()

    events = pd.concat([events, _get_tagsdf(events)], axis=1)  # type: ignore[reportAssignmentType]
    events = _make_new_positions(events)

    # Reclassify aerial duels by known goalkeepers as GK claims before
    # _fix_wyscout_events removes unmatched duels.
    if goalkeeper_ids:
        gk_aerial_mask = (
            (events["type_id"] == _WS_TYPE_DUEL)
            & (events["subtype_id"] == _WS_SUBTYPE_AIR_DUEL)
            & events["player_id"].isin(goalkeeper_ids)
        )
        events.loc[gk_aerial_mask, "type_id"] = _WS_TYPE_GK
        events.loc[gk_aerial_mask, "subtype_id"] = _WS_SUBTYPE_GK_CLAIM

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
            f"dropped: {dict(unrecognized_counts)}",
            stacklevel=2,
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


# ---------------------------------------------------------------------------
# Backwards-compatible alias: tests import _determine_type_id from here
# ---------------------------------------------------------------------------


def _determine_type_id(event: pd.Series) -> int:
    """Determine the type of a single Wyscout event (compatibility wrapper).

    This wraps the vectorized implementation so that existing tests calling
    ``_determine_type_id(series)`` continue to work.

    Parameters
    ----------
    event : pd.Series
        A single Wyscout event as a pandas Series.

    Returns
    -------
    int
        The SPADL action type id.
    """
    from ._wyscout_mappings import _vectorized_type_id

    df = event.to_frame().T
    # Ensure boolean columns expected by _vectorized_type_id exist with defaults
    for col in ("not_accurate",):
        if col not in df.columns:
            df[col] = False
    # Convert column dtypes so np.select receives proper boolean/int arrays
    # (Series.to_frame().T yields object dtype for mixed-type Series)
    df = df.infer_objects()
    return int(_vectorized_type_id(df).iloc[0])
