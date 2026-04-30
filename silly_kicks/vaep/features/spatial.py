"""Spatial feature transformers.

Six transformers: start/end location coordinates, start/end polar (distance +
angle to opponent goal), movement (per-action delta), and ``space_delta``
(gamestate-to-gamestate spatial change).
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, GameStates, simple

__all__ = [
    "endlocation",
    "endpolar",
    "movement",
    "space_delta",
    "startlocation",
    "startpolar",
]


@simple
def startlocation(actions: Actions) -> Features:
    """Get the location where each action started.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'start_x' and 'start_y' location of each action.

    Examples
    --------
    Extract action start-location features per gamestate slot::

        from silly_kicks.vaep.features import startlocation

        feats = startlocation(states)
    """
    return actions[["start_x", "start_y"]]  # type: ignore[reportReturnType]


@simple
def endlocation(actions: Actions) -> Features:
    """Get the location where each action ended.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'end_x' and 'end_y' location of each action.

    Examples
    --------
    Extract action end-location features per gamestate slot::

        from silly_kicks.vaep.features import endlocation

        feats = endlocation(states)
    """
    return actions[["end_x", "end_y"]]  # type: ignore[reportReturnType]


@simple
def startpolar(actions: Actions) -> Features:
    """Get the polar coordinates of each action's start location.

    The center of the opponent's goal is used as the origin.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'start_dist_to_goal' and 'start_angle_to_goal' of each action.

    Examples
    --------
    Extract polar (distance/angle to goal) start features per gamestate slot::

        from silly_kicks.vaep.features import startpolar

        feats = startpolar(states)
    """
    polardf = pd.DataFrame(index=actions.index)
    dx = (spadlcfg.field_length - actions["start_x"]).abs().to_numpy()
    dy = (spadlcfg.field_width / 2 - actions["start_y"]).abs().to_numpy()
    polardf["start_dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["start_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@simple
def endpolar(actions: Actions) -> Features:
    """Get the polar coordinates of each action's end location.

    The center of the opponent's goal is used as the origin.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'end_dist_to_goal' and 'end_angle_to_goal' of each action.

    Examples
    --------
    Extract polar (distance/angle to goal) end features per gamestate slot::

        from silly_kicks.vaep.features import endpolar

        feats = endpolar(states)
    """
    polardf = pd.DataFrame(index=actions.index)
    dx = (spadlcfg.field_length - actions["end_x"]).abs().to_numpy()
    dy = (spadlcfg.field_width / 2 - actions["end_y"]).abs().to_numpy()
    polardf["end_dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["end_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@simple
def movement(actions: Actions) -> Features:
    """Get the distance covered by each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The horizontal ('dx'), vertical ('dy') and total ('movement') distance
        covered by each action.

    Examples
    --------
    Extract action displacement (dx, dy, total) features per gamestate slot::

        from silly_kicks.vaep.features import movement

        feats = movement(states)
    """
    mov = pd.DataFrame(index=actions.index)
    mov["dx"] = actions.end_x - actions.start_x
    mov["dy"] = actions.end_y - actions.start_y
    mov["movement"] = np.sqrt(mov.dx**2 + mov.dy**2)
    return mov


def space_delta(gamestates: GameStates) -> Features:
    """Get the distance covered between the last and previous actions.

    Parameters
    ----------
    gamestates : GameStates
        The gamestates of a game.

    Returns
    -------
    Features
        A dataframe with a column for the horizontal ('dx_a0i'), vertical
        ('dy_a0i') and total ('mov_a0i') distance covered between each
        <nb_prev_actions> action ai and action a0.

    Examples
    --------
    Extract space-since-previous-action features per gamestate slot::

        from silly_kicks.vaep.features import space_delta

        feats = space_delta(states)
    """
    a0 = gamestates[0]
    spaced = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a0.start_x
        spaced["dx_a0" + (str(i + 1))] = dx
        dy = a.end_y - a0.start_y
        spaced["dy_a0" + (str(i + 1))] = dy
        spaced["mov_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2)
    return spaced
