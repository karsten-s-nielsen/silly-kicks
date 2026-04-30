"""Action-type feature transformers.

Two transformers: integer-typed and one-hot-encoded action-type per gamestate
slot. The integer-typed variant delegates to ``actiontype_categorical`` in
:mod:`silly_kicks.vaep.feature_framework` so standard and atomic VAEP share
the same SPADL-config-parameterized helper.
"""

import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg
from silly_kicks.vaep.feature_framework import (
    Actions,
    Features,
    actiontype_categorical,
    simple,
)

__all__ = ["actiontype", "actiontype_onehot"]


@simple
def actiontype(actions: Actions) -> Features:
    """Get the type of each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'type_id' of each action.

    Examples
    --------
    Extract action-type integer codes per gamestate slot::

        from silly_kicks.vaep.features import actiontype

        feats = actiontype(states)
    """
    return actiontype_categorical(actions, spadlcfg)


@simple
def actiontype_onehot(actions: Actions) -> Features:
    """Get the one-hot-encoded type of each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        A one-hot encoding of each action's type.

    Examples
    --------
    Extract action-type one-hot features per gamestate slot::

        from silly_kicks.vaep.features import actiontype_onehot

        feats = actiontype_onehot(states)
    """
    X = {}
    for type_id, type_name in enumerate(spadlcfg.actiontypes):
        col = "actiontype_" + type_name
        X[col] = actions["type_id"] == type_id
    return pd.DataFrame(X, index=actions.index)
