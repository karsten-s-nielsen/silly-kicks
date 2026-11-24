"""The ONE visible-area polygon join: the ``visible_area`` polygon of each action ROW.

A NEUTRAL root module, like ``_polygon`` and ``id_compat``: ``tracking``, ``territorial_defense`` and
``restdefense`` all read a provider ``visible_area`` table (``action_id`` -> ``polygon``, from
``silly_kicks.providers.statsbomb.shape_snapshots``), and none may import another's private module. Before this
module, each kept its own ``action_id`` dict. That join has two traps:

- ADR-019: ``action_id`` joins separately-sourced frames, so a raw-id key MISSES SILENTLY across dtypes and reads
  as a genuine absence. Every key here is canonical.
- Spec W9.4: an ``action_id`` names one action only inside a game. Two games' ``visible_area`` tables in one call
  collide on ``action_id``, and the last polygon silently wins (owner ruling 2026-11-12).

So a ``visible_area`` with a ``game_id`` column joins on ``(game_id, action_id)``. Without one, the ``action_id``
join is allowed only when it is unambiguous (the W9.5 legacy-links rule): the actions span one game, and when they
have no ``game_id`` column to show that, their ``action_id`` values are unique. A repeated ``visible_area`` key, or an
ambiguous join, raises ``ValueError``: an ambiguous join is never guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id_series

_NO_GAME_ID = (
    "visible_area has no game_id column and the actions span {n} games: an action_id names one action only inside "
    "a game. Add a game_id column to visible_area, or call once per game."
)
_REPEATED_ACTION_ID = (
    "action_id repeats in actions that have no game_id column, so they may span more than one game: the action_id "
    "join is ambiguous. Add a game_id column to both frames, or call once per game."
)
_REPEATED_KEY = "visible_area repeats the key {key!r}: one action has more than one polygon."
_REPEATED_KEY_NO_GAME = (
    "visible_area repeats the action_id {key!r} and has no game_id column: two games' tables collide on action_id. "
    "Add a game_id column to visible_area, or call once per game."
)
_REPEATED_KEY_ACTIONS_NO_GAME = (
    "visible_area repeats the action_id {key!r} in two games, and actions has no game_id column to tell them apart. "
    "Add a game_id column to actions, or call once per game."
)


def _keys(values: pd.Series) -> np.ndarray:
    """The canonical id string of each value (``None`` for a missing one), as an object array."""
    keys = canonical_id_series(values.reset_index(drop=True)).to_numpy(dtype=object, copy=True)  # written below (CoW)
    keys[pd.isna(keys)] = None
    return keys


def _row_keys(frame: pd.DataFrame, by_game: bool) -> list:
    action_keys = _keys(frame["action_id"])
    if not by_game:
        return list(action_keys)
    return list(zip(_keys(frame["game_id"]), action_keys, strict=True))


def polygons_by_action_row(actions: pd.DataFrame, visible_area: pd.DataFrame | None) -> list:
    """The ``visible_area`` polygon of each ``actions`` row, in row order; ``None`` where there is none.

    The join is on ``(game_id, action_id)`` when both frames have a ``game_id`` column, else on ``action_id``.
    The ``action_id`` join raises ``ValueError`` when the actions span more than one game, or repeat an
    ``action_id`` with no ``game_id`` column. A repeated ``visible_area`` key raises ``ValueError`` too. Keys are
    canonical (ADR-019).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> actions = pd.DataFrame({"game_id": [1, 2], "action_id": [0, 0]})
    >>> visible = pd.DataFrame({"game_id": [2], "action_id": [0], "polygon": [np.zeros((3, 2))]})
    >>> [p is None for p in polygons_by_action_row(actions, visible)]
    [True, False]
    """
    n = len(actions)
    if visible_area is None or len(visible_area) == 0:
        return [None] * n
    by_game = "game_id" in visible_area.columns and "game_id" in actions.columns
    if not by_game:
        if "game_id" in actions.columns:
            n_games = len(set(_keys(actions["game_id"])))
            if n_games > 1:
                raise ValueError(_NO_GAME_ID.format(n=n_games))
        elif pd.Series(_keys(actions["action_id"])).duplicated().any():
            raise ValueError(_REPEATED_ACTION_ID)
    # the game of each visible_area key when only visible_area has a game_id column: a repeat across two games
    # is the actions' missing game_id, a repeat inside one game is a repeated key
    vis_games = _keys(visible_area["game_id"]) if not by_game and "game_id" in visible_area.columns else None
    table: dict = {}
    game_of: dict = {}
    for i, (key, polygon) in enumerate(zip(_row_keys(visible_area, by_game), visible_area["polygon"], strict=True)):
        if key in table:
            if by_game:
                message = _REPEATED_KEY
            elif vis_games is None:
                message = _REPEATED_KEY_NO_GAME
            else:
                message = _REPEATED_KEY if game_of[key] == vis_games[i] else _REPEATED_KEY_ACTIONS_NO_GAME
            raise ValueError(message.format(key=key))
        table[key] = polygon
        if vis_games is not None:
            game_of[key] = vis_games[i]
    return [table.get(key) for key in _row_keys(actions, by_game)]
