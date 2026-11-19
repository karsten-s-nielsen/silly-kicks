"""The library's one goal rule (ADR-135).

A goal is a successful row of a shot type (``shot``, ``shot_penalty``, ``shot_freekick``). An own goal is a row
with the ``owngoal`` result, on any type: converters emit own goals as ``bad_touch``, not as shots (ADR-018).

Each place that decides "this row changes the score" reads these two masks and keeps no copy: the VAEP labels,
the ``goalscore`` feature, the VAEP formula, ``add_game_state``, the win-probability state and model, the ghost
score lookup, the stoppage spans, the coordination event windows, the ELASTIC goal termination, the atomic
conversion and the match-outcome census. A shot measure that reads the result of a shot by its own definition
(shot stopping, corner effectiveness, the defensive credit of a shot, the xT grid) is not such a place.
"""

from __future__ import annotations

import pandas as pd

from . import config as spadlconfig

SHOT_TYPE_NAMES: frozenset[str] = frozenset({"shot", "shot_penalty", "shot_freekick"})
_SHOT_TYPE_IDS: frozenset[int] = frozenset(spadlconfig.actiontype_id[n] for n in SHOT_TYPE_NAMES)


def goal_mask(actions: pd.DataFrame) -> pd.Series:
    """True on a successful row of a shot type; reads ``type_id``, else ``type_name`` (and ``result_id``, else
    ``result_name``).

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.spadl import config as c
    >>> a = pd.DataFrame({"type_id": [c.actiontype_id["shot_penalty"], c.actiontype_id["pass"]],
    ...                   "result_id": [c.result_id["success"]] * 2})
    >>> goal_mask(a).tolist()
    [True, False]
    """
    if "type_id" in actions.columns:
        is_shot = actions["type_id"].isin(_SHOT_TYPE_IDS)
    else:
        is_shot = actions["type_name"].isin(SHOT_TYPE_NAMES)
    return is_shot & _result_is(actions, "success")


def owngoal_mask(actions: pd.DataFrame) -> pd.Series:
    """True on a row with the ``owngoal`` result, on any type (no type gate); reads ``result_id``, else
    ``result_name``.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.spadl import config as c
    >>> a = pd.DataFrame({"type_id": [c.actiontype_id["bad_touch"]], "result_id": [c.result_id["owngoal"]]})
    >>> owngoal_mask(a).tolist()
    [True]
    """
    return _result_is(actions, "owngoal")


def _result_is(actions: pd.DataFrame, name: str) -> pd.Series:
    """The result test: reads ``result_id``, else ``result_name`` (a name-only table, as for the type)."""
    if "result_id" in actions.columns:
        return actions["result_id"] == spadlconfig.result_id[name]
    return actions["result_name"] == name
