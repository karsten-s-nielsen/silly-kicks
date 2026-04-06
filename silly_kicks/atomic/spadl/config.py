"""Configuration of the Atomic-SPADL language.

Attributes
----------
field_length : float
    The length of a pitch (in meters).
field_width : float
    The width of a pitch (in meters).
bodyparts : list(str)
    The bodyparts used in the Atomic-SPADL language.
actiontypes : list(str)
    The action types used in the Atomic-SPADL language.

"""

import functools

import pandas as pd

import silly_kicks.spadl.config as _spadl

field_length = _spadl.field_length
field_width = _spadl.field_width

bodyparts = _spadl.bodyparts
bodyparts_df = _spadl.bodyparts_df

actiontypes = _spadl.actiontypes + [
    "receival",
    "interception",
    "out",
    "offside",
    "goal",
    "owngoal",
    "yellow_card",
    "red_card",
    "corner",
    "freekick",
]

# Precomputed lookups — O(1) vs O(k) list.index() scans
actiontype_id: dict[str, int] = {name: i for i, name in enumerate(actiontypes)}
result_id: dict[str, int] = _spadl.result_id
bodypart_id: dict[str, int] = _spadl.bodypart_id


@functools.lru_cache(maxsize=None)
def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each Atomic-SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each Atomic-SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])
