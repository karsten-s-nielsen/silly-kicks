"""Implements the label tranformers of the Atomic-VAEP framework."""

from __future__ import annotations

import pandas as pd

import silly_kicks.atomic.spadl.config as atomicspadl


def scores(actions: pd.DataFrame, nr_actions: int = 10, xg_column: str | None = None) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.
    xg_column : str or None, default=None  # noqa: DAR103
        If provided, return xG-weighted scoring probability instead of boolean labels.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "scores" labels on an atomic-SPADL stream for VAEP training::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import scores

        atomic = convert_to_atomic(actions)
        y_scores = scores(atomic, nr_actions=10)
        # y_scores["scores"] is bool: True iff the team in possession scores
        # within the next 10 atomic actions.
    """
    if xg_column is not None:
        return _scores_xg(actions, nr_actions, xg_column)

    goal = actions["type_id"] == atomicspadl.actiontype_id["goal"]
    owngoal = actions["type_id"] == atomicspadl.actiontype_id["owngoal"]
    team_id = actions["team_id"]

    result = goal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        result = result | (shifted_goal & same_team) | (shifted_owngoal & ~same_team)

    return pd.DataFrame(result, columns=["scores"])


def concedes(actions: pd.DataFrame, nr_actions: int = 10, xg_column: str | None = None) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.
    xg_column : str or None, default=None  # noqa: DAR103
        If provided, return xG-weighted conceding probability instead of boolean labels.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action set to
        True if a goal was conceded by the team possessing the ball within the
        next x actions; otherwise False. If xg_column is provided, the column
        contains the maximum xG value instead of a boolean.

    Examples
    --------
    Compute "concedes" labels on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import concedes

        atomic = convert_to_atomic(actions)
        y_concedes = concedes(atomic, nr_actions=10)
        # y_concedes["concedes"] is bool: True iff the team in possession
        # concedes within the next 10 atomic actions.
    """
    if xg_column is not None:
        return _concedes_xg(actions, nr_actions, xg_column)

    goal = actions["type_id"] == atomicspadl.actiontype_id["goal"]
    owngoal = actions["type_id"] == atomicspadl.actiontype_id["owngoal"]
    team_id = actions["team_id"]

    result = owngoal.copy()
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        result = result | (shifted_goal & ~same_team) | (shifted_owngoal & same_team)

    return pd.DataFrame(result, columns=["concedes"])


def _scores_xg(actions: pd.DataFrame, nr_actions: int, xg_column: str) -> pd.DataFrame:
    """Compute xG-weighted scoring labels using shift-based vectorization."""
    goal = actions["type_id"] == atomicspadl.actiontype_id["goal"]
    owngoal = actions["type_id"] == atomicspadl.actiontype_id["owngoal"]
    xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]
    team_id = actions["team_id"]

    result = pd.Series(0.0, index=actions.index)
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_xg = xg.shift(-i).fillna(0.0)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        score_xg = shifted_xg.where(shifted_goal & same_team, 0.0)
        owngoal_xg = shifted_xg.where(shifted_owngoal & ~same_team, 0.0)
        result = result.combine(score_xg + owngoal_xg, max, fill_value=0.0)  # type: ignore[reportArgumentType]
    return pd.DataFrame({"scores": result})


def _concedes_xg(actions: pd.DataFrame, nr_actions: int, xg_column: str) -> pd.DataFrame:
    """Compute xG-weighted conceding labels using shift-based vectorization."""
    goal = actions["type_id"] == atomicspadl.actiontype_id["goal"]
    owngoal = actions["type_id"] == atomicspadl.actiontype_id["owngoal"]
    xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)  # type: ignore[reportOptionalMemberAccess]
    team_id = actions["team_id"]

    result = pd.Series(0.0, index=actions.index)
    for i in range(1, nr_actions):
        shifted_goal = goal.shift(-i).fillna(False)
        shifted_owngoal = owngoal.shift(-i).fillna(False)
        shifted_xg = xg.shift(-i).fillna(0.0)
        shifted_team = team_id.shift(-i)
        same_team = team_id == shifted_team
        concede_xg = shifted_xg.where(shifted_goal & ~same_team, 0.0)
        owngoal_xg = shifted_xg.where(shifted_owngoal & same_team, 0.0)
        result = result.combine(concede_xg + owngoal_xg, max, fill_value=0.0)  # type: ignore[reportArgumentType]
    return pd.DataFrame({"concedes": result})


def goal_from_shot(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine whether a goal was scored from the current action.

    This label can be use to train an xG model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.

    Examples
    --------
    Build per-action goal labels on an atomic-SPADL stream for an xG model.
    Atomic decomposes a successful shot into a ``shot`` row immediately
    followed by a ``goal`` row, so the label is True only on the ``shot``
    row whose successor is ``goal``::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import goal_from_shot

        atomic = convert_to_atomic(actions)
        y = goal_from_shot(atomic)
        # y["goal"] is True only on shot rows followed by a goal row.
    """
    goals = (actions["type_id"] == atomicspadl.actiontype_id["shot"]) & (
        actions["type_id"].shift(-1) == atomicspadl.actiontype_id["goal"]
    )

    return pd.DataFrame(goals.rename("goal"))


def save_from_shot(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine whether the goalkeeper saved the current shot.

    This label can be used to train an Expected Saves (xS) model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'save_from_shot' and a row for each action
        set to True if the action is a keeper save; otherwise False.

    Examples
    --------
    Build per-action keeper-save labels on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import save_from_shot

        atomic = convert_to_atomic(actions)
        y = save_from_shot(atomic)
        # y["save_from_shot"] is True on every keeper_save row.
    """
    saves = actions["type_id"] == atomicspadl.actiontype_id["keeper_save"]
    return pd.DataFrame(saves.rename("save_from_shot"))


def claim_from_cross(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine whether the goalkeeper claimed the current cross.

    This label can be used to train an Expected Claims (xC) model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'claim_from_cross' and a row for each action
        set to True if the action is a keeper claim; otherwise False.

    Examples
    --------
    Build per-action keeper-claim labels on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.vaep.labels import claim_from_cross

        atomic = convert_to_atomic(actions)
        y = claim_from_cross(atomic)
        # y["claim_from_cross"] is True on every keeper_claim row.
    """
    claims = actions["type_id"] == atomicspadl.actiontype_id["keeper_claim"]
    return pd.DataFrame(claims.rename("claim_from_cross"))
