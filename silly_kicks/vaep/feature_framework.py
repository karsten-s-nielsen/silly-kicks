"""Shared VAEP feature framework primitives.

Both ``silly_kicks.vaep.features`` (standard VAEP) and
``silly_kicks.atomic.vaep.features`` (atomic VAEP) build on this module's
type aliases and helpers. Extracted in 2.4.0 to give the cross-package
boundary a named home and to close TODO A9. See ADR-002.

Public surface:

- Type aliases: ``Actions``, ``Features``, ``FeatureTransfomer``, ``GameStates``
- Gamestate construction: ``gamestates``
- Decorator: ``simple``
- SPADL-config-parameterized helper: ``actiontype_categorical``
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, no_type_check

import pandas as pd  # type: ignore

__all__ = [
    "Actions",
    "FeatureTransfomer",
    "Features",
    "GameStates",
    "actiontype_categorical",
    "gamestates",
    "simple",
]

Actions = pd.DataFrame
GameStates = list[pd.DataFrame]
Features = pd.DataFrame
FeatureTransfomer = Callable[[GameStates], Features]


def gamestates(actions: Actions, nb_prev_actions: int = 3) -> GameStates:
    r"""Convert a dataframe of actions to gamestates.

    Each gamestate is represented as the <nb_prev_actions> previous actions.

    The list of gamestates is internally represented as a list of actions
    dataframes :math:`[a_0,a_1,\ldots]` where each row in the a_i dataframe contains the
    previous action of the action in the same row in the :math:`a_{i-1}` dataframe.

    Parameters
    ----------
    actions : Actions
        A DataFrame with the actions of a game.
    nb_prev_actions : int, default=3  # noqa: DAR103
        The number of previous actions included in the game state.

    Raises
    ------
    ValueError
        If the number of actions is smaller 1.

    Returns
    -------
    GameStates
         The <nb_prev_actions> previous actions for each action.

    Examples
    --------
    Build a 3-step gamestate stream from a SPADL action DataFrame::

        from silly_kicks.vaep.feature_framework import gamestates

        states = gamestates(actions, nb_prev_actions=3)
        # ``states`` is a list of 3 DataFrames — states[0] is the current action,
        # states[1] is the previous action aligned by row, states[2] is the one before.
    """
    if nb_prev_actions < 1:
        raise ValueError("The game state should include at least one preceding action.")
    states = [actions]
    group_keys = ["game_id", "period_id"]
    # Precompute group-first values once for boundary filling (excludes groupby keys)
    first_in_group = actions.groupby(group_keys, sort=False).transform("first")
    for i in range(1, nb_prev_actions):
        prev = actions.shift(i)
        # Detect period/game boundaries: where shifted row crosses a group
        boundary = (actions["game_id"] != actions["game_id"].shift(i)) | (
            actions["period_id"] != actions["period_id"].shift(i)
        )
        # At boundaries, fill groupby-key columns with current row values
        for col in group_keys:
            prev.loc[boundary, col] = actions.loc[boundary, col]
        # At boundaries, fill remaining columns with the first row of the current group
        for col in first_in_group.columns:
            prev.loc[boundary, col] = first_in_group.loc[boundary, col]
        prev.index = actions.index.copy()
        states.append(prev)
    return states


@no_type_check
def simple(actionfn: Callable) -> FeatureTransfomer:
    """Make a function decorator to apply actionfeatures to game states.

    Parameters
    ----------
    actionfn : Callable
        A feature transformer that operates on actions.

    Returns
    -------
    FeatureTransfomer
        A feature transformer that operates on game states.

    Examples
    --------
    Lift an action-level feature function to a gamestate-level transformer::

        from silly_kicks.vaep.feature_framework import simple

        @simple
        def my_action_feature(actions):
            return actions[['x']]

        feats = my_action_feature(states)
    """

    @wraps(actionfn)
    def _wrapper(gamestates: list[Actions]) -> pd.DataFrame:
        if not isinstance(gamestates, (list,)):
            gamestates = [gamestates]
        X = []
        for i, a in enumerate(gamestates):
            Xi = actionfn(a)
            Xi.columns = [c + "_a" + str(i) for c in Xi.columns]
            X.append(Xi)
        return pd.concat(X, axis=1)  # type: ignore[reportReturnType]

    return _wrapper


def actiontype_categorical(actions: Actions, spadl_cfg: Any) -> Features:
    """SPADL-config-parameterized categorical actiontype helper.

    Both standard-VAEP and atomic-VAEP wrap this with ``@simple`` to
    produce their respective ``actiontype`` feature transformers.

    Parameters
    ----------
    actions : Actions
        The actions of a game.
    spadl_cfg : module
        A SPADL config module exposing ``actiontypes`` (list of type names)
        and ``actiontypes_df()`` (DataFrame with ``type_id`` / ``type_name``).
        Pass ``silly_kicks.spadl.config`` for standard SPADL or
        ``silly_kicks.atomic.spadl.config`` for atomic SPADL.

    Returns
    -------
    Features
        A single-column DataFrame with column ``actiontype`` of dtype
        ``pd.Categorical`` whose categories are the SPADL config's
        action-type names in original order.

    Examples
    --------
    Build a per-action categorical actiontype feature for standard SPADL::

        from silly_kicks.vaep.feature_framework import actiontype_categorical
        import silly_kicks.spadl.config as spadlcfg

        feats = actiontype_categorical(actions, spadlcfg)
        # feats has one column 'actiontype' of dtype Categorical.
    """
    X = pd.DataFrame(index=actions.index)
    categories = list(dict.fromkeys(spadl_cfg.actiontypes))  # dedupe, preserve order
    X["actiontype"] = pd.Categorical(
        actions["type_id"].replace(spadl_cfg.actiontypes_df().type_name.to_dict()),
        categories=categories,
        ordered=False,
    )
    return X
