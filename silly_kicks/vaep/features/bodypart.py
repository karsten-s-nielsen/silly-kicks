"""Bodypart feature transformers.

Four transformers: integer-typed bodypart, integer-typed detailed bodypart
(``foot_left`` / ``foot_right`` resolved), one-hot bodypart, one-hot detailed
bodypart.
"""

import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, simple

__all__ = [
    "bodypart",
    "bodypart_detailed",
    "bodypart_detailed_onehot",
    "bodypart_onehot",
]


@simple
def bodypart(actions: Actions) -> Features:
    """Get the body part used to perform each action.

    This feature generator does not distinguish between the left and right foot.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'bodypart_id' of each action.

    See Also
    --------
    bodypart_detailed :
        An alternative version that splits between the left and right foot.

    Examples
    --------
    Extract bodypart integer codes per gamestate slot::

        from silly_kicks.vaep.features import bodypart

        feats = bodypart(states)
    """
    X = pd.DataFrame(index=actions.index)
    foot_id = spadlcfg.bodypart_id["foot"]
    left_foot_id = spadlcfg.bodypart_id["foot_left"]
    right_foot_id = spadlcfg.bodypart_id["foot_right"]
    X["bodypart"] = pd.Categorical(
        actions["bodypart_id"]
        .replace([left_foot_id, right_foot_id], foot_id)
        .replace(spadlcfg.bodyparts_df().bodypart_name.to_dict()),
        categories=["foot", "head", "other", "head/other"],
        ordered=False,
    )
    return X


@simple
def bodypart_detailed(actions: Actions) -> Features:
    """Get the body part with split by foot used to perform each action.

    This feature generator distinguishes between the left and right foot, if
    supported by the dataprovider.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'bodypart_id' of each action.

    See Also
    --------
    bodypart :
        An alternative version that does not split between the left and right foot.

    Examples
    --------
    Extract detailed bodypart integer codes (foot_left / foot_right resolved)::

        from silly_kicks.vaep.features import bodypart_detailed

        feats = bodypart_detailed(states)
    """
    X = pd.DataFrame(index=actions.index)
    X["bodypart"] = pd.Categorical(
        actions["bodypart_id"].replace(spadlcfg.bodyparts_df().bodypart_name.to_dict()),
        categories=spadlcfg.bodyparts,
        ordered=False,
    )
    return X


@simple
def bodypart_onehot(actions: Actions) -> Features:
    """Get the one-hot-encoded bodypart of each action.

    This feature generator does not distinguish between the left and right foot.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The one-hot encoding of each action's bodypart.

    See Also
    --------
    bodypart_detailed_onehot :
        An alternative version that splits between the left and right foot.

    Examples
    --------
    Extract bodypart one-hot features per gamestate slot::

        from silly_kicks.vaep.features import bodypart_onehot

        feats = bodypart_onehot(states)
    """
    X = {}
    for bodypart_id, bodypart_name in enumerate(spadlcfg.bodyparts):
        if bodypart_name in ("foot_left", "foot_right"):
            continue
        col = "bodypart_" + bodypart_name
        if bodypart_name == "foot":
            foot_id = spadlcfg.bodypart_id["foot"]
            left_foot_id = spadlcfg.bodypart_id["foot_left"]
            right_foot_id = spadlcfg.bodypart_id["foot_right"]
            X[col] = actions["bodypart_id"].isin([foot_id, left_foot_id, right_foot_id])
        elif bodypart_name == "head/other":
            head_id = spadlcfg.bodypart_id["head"]
            other_id = spadlcfg.bodypart_id["other"]
            head_other_id = spadlcfg.bodypart_id["head/other"]
            X[col] = actions["bodypart_id"].isin([head_id, other_id, head_other_id])
        else:
            X[col] = actions["bodypart_id"] == bodypart_id
    return pd.DataFrame(X, index=actions.index)


@simple
def bodypart_detailed_onehot(actions: Actions) -> Features:
    """Get the one-hot-encoded bodypart with split by foot of each action.

    This feature generator distinguishes between the left and right foot, if
    supported by the dataprovider.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The one-hot encoding of each action's bodypart.

    See Also
    --------
    bodypart_onehot :
        An alternative version that does not split between the left and right foot.

    Examples
    --------
    Extract detailed bodypart one-hot features per gamestate slot::

        from silly_kicks.vaep.features import bodypart_detailed_onehot

        feats = bodypart_detailed_onehot(states)
    """
    X = {}
    for bodypart_id, bodypart_name in enumerate(spadlcfg.bodyparts):
        col = "bodypart_" + bodypart_name
        if bodypart_name == "foot":
            foot_id = spadlcfg.bodypart_id["foot"]
            left_foot_id = spadlcfg.bodypart_id["foot_left"]
            right_foot_id = spadlcfg.bodypart_id["foot_right"]
            X[col] = actions["bodypart_id"].isin([foot_id, left_foot_id, right_foot_id])
        elif bodypart_name == "head/other":
            head_id = spadlcfg.bodypart_id["head"]
            other_id = spadlcfg.bodypart_id["other"]
            head_other_id = spadlcfg.bodypart_id["head/other"]
            X[col] = actions["bodypart_id"].isin([head_id, other_id, head_other_id])
        else:
            X[col] = actions["bodypart_id"] == bodypart_id
    return pd.DataFrame(X, index=actions.index)
