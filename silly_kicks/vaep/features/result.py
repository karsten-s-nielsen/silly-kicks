"""Result feature transformers.

Five transformers: integer-typed result, one-hot result, joined actiontype +
result one-hot, and prev-only variants (zero-out the current action's result
to avoid leakage in models like HybridVAEP).
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .actiontype import actiontype_onehot
from .core import Actions, Features, GameStates, simple

__all__ = [
    "actiontype_result_onehot",
    "actiontype_result_onehot_prev_only",
    "result",
    "result_onehot",
    "result_onehot_prev_only",
]


@simple
def result(actions: Actions) -> Features:
    """Get the result of each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'result_id' of each action.

    Examples
    --------
    Extract action-result integer codes per gamestate slot::

        from silly_kicks.vaep.features import result

        feats = result(states)
    """
    X = pd.DataFrame(index=actions.index)
    X["result"] = pd.Categorical(
        actions["result_id"].replace(spadlcfg.results_df().result_name.to_dict()),
        categories=spadlcfg.results,
        ordered=False,
    )
    return X


@simple
def result_onehot(actions: Actions) -> Features:
    """Get the one-hot-encode result of each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The one-hot encoding of each action's result.

    Examples
    --------
    Extract action-result one-hot features per gamestate slot::

        from silly_kicks.vaep.features import result_onehot

        feats = result_onehot(states)
    """
    X = {}
    for result_id, result_name in enumerate(spadlcfg.results):
        col = "result_" + result_name
        X[col] = actions["result_id"] == result_id
    return pd.DataFrame(X, index=actions.index)


@simple
def actiontype_result_onehot(actions: Actions) -> Features:
    """Get a one-hot encoding of the combination between the type and result of each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The one-hot encoding of each action's type and result.

    Examples
    --------
    Extract joint action-type x result one-hot features per gamestate slot::

        from silly_kicks.vaep.features import actiontype_result_onehot

        feats = actiontype_result_onehot(states)
    """
    res = result_onehot.__wrapped__(actions)  # type: ignore
    tys = actiontype_onehot.__wrapped__(actions)  # type: ignore
    cross = tys.values[:, :, np.newaxis] & res.values[:, np.newaxis, :]
    cols = [f"{tc}_{rc}" for tc in tys.columns for rc in res.columns]
    return pd.DataFrame(cross.reshape(len(actions), -1), columns=cols, index=actions.index)


def result_onehot_prev_only(gamestates: GameStates) -> Features:
    """Result one-hot encoding for previous actions only (a1, a2, ...).

    Excludes a0 to prevent result leakage in Hybrid-VAEP mode.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        The one-hot encoding of each previous action's result, excluding a0.

    Examples
    --------
    Result one-hot features for previous actions only (HybridVAEP — removes
    result leakage on the current action)::

        from silly_kicks.vaep.features import result_onehot_prev_only

        feats = result_onehot_prev_only(states)
        # feats has columns for a1, a2, ... but not a0.
    """
    dfs = []
    for i, actions in enumerate(gamestates):
        if i == 0:
            continue
        result_df = result_onehot.__wrapped__(actions)  # type: ignore
        result_df.columns = [c + "_a" + str(i) for c in result_df.columns]
        dfs.append(result_df)
    return pd.concat(dfs, axis=1)  # type: ignore[reportReturnType]


def actiontype_result_onehot_prev_only(gamestates: GameStates) -> Features:
    """Action type x result cross-product for previous actions only (a1, a2, ...).

    Excludes a0 to prevent result leakage in Hybrid-VAEP mode.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        The one-hot encoding of each previous action's type-result cross, excluding a0.

    Examples
    --------
    Joint action-type x result one-hot for previous actions only::

        from silly_kicks.vaep.features import actiontype_result_onehot_prev_only

        feats = actiontype_result_onehot_prev_only(states)
    """
    dfs = []
    for i, actions in enumerate(gamestates):
        if i == 0:
            continue
        cross_df = actiontype_result_onehot.__wrapped__(actions)  # type: ignore
        cross_df.columns = [c + "_a" + str(i) for c in cross_df.columns]
        dfs.append(cross_df)
    return pd.concat(dfs, axis=1)  # type: ignore[reportReturnType]
