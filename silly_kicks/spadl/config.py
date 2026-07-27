"""Configuration of the SPADL language.

Attributes
----------
field_length : float
    The length of a pitch (in meters).
field_width : float
    The width of a pitch (in meters).
penalty_area_half_width : float
    Half the width of the penalty area (in meters). FIFA Laws: the area is 40.32 m wide.
penalty_area_depth : float
    The depth of the penalty area from the goal line (in meters). FIFA Laws: 16.5 m.
bodyparts : list(str)
    The bodyparts used in the SPADL language.
results : list(str)
    The action results used in the SPADL language.
actiontypes : list(str)
    The action types used in the SPADL language.
actiontype_id : dict[str, int]
    Reverse lookup: action type name to integer ID.
result_id : dict[str, int]
    Reverse lookup: result name to integer ID.
bodypart_id : dict[str, int]
    Reverse lookup: body part name to integer ID.

"""

import functools

import pandas as pd  # type: ignore

field_length: float = 105.0  # unit: meters
field_width: float = 68.0  # unit: meters

# FIFA Laws of the Game: the penalty area is 40.32 m wide and 16.5 m deep. CANONICAL -- do NOT
# re-derive these locally. `tracking/_ghost_gk.py` deliberately still uses 40.3 (half-width 20.15)
# until its re-fit: its bundled weights were trained on that value, and flipping the constant
# without re-fitting would skew a trained feature. Its artifact records the divergence in its
# feature contract, so the flip cannot happen silently.
penalty_area_half_width: float = 20.16
penalty_area_depth: float = 16.5

bodyparts: list[str] = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]
results: list[str] = [
    "fail",
    "success",
    "offside",
    "owngoal",
    "yellow_card",
    "red_card",
]
actiontypes: list[str] = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "non_action",
    "dribble",
    "goalkick",
]

# Precomputed lookups — O(1) vs O(k) list.index() scans
actiontype_id: dict[str, int] = {name: i for i, name in enumerate(actiontypes)}
result_id: dict[str, int] = {name: i for i, name in enumerate(results)}
bodypart_id: dict[str, int] = {name: i for i, name in enumerate(bodyparts)}


@functools.cache
def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])


@functools.cache
def results_df() -> pd.DataFrame:
    """Return a dataframe with the result id and result name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'result_id' and 'result_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(results)), columns=["result_id", "result_name"])


@functools.cache
def bodyparts_df() -> pd.DataFrame:
    """Return a dataframe with the bodypart id and bodypart name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'bodypart_id' and 'bodypart_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(bodyparts)), columns=["bodypart_id", "bodypart_name"])
