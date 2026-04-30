"""Temporal feature transformers.

Three transformers: ``time`` (period + clock), ``time_delta``
(gamestate-to-gamestate dt), ``speed`` (movement / dt).
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .core import Actions, Features, GameStates, simple

__all__ = ["speed", "time", "time_delta"]


@simple
def time(actions: Actions) -> Features:
    """Get the time when each action was performed.

    This generates the following features:
        :period_id:
            The ID of the period.
        :time_seconds:
            Seconds since the start of the period.
        :time_seconds_overall:
            Seconds since the start of the game. Stoppage time during previous
            periods is ignored.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'period_id', 'time_seconds' and 'time_seconds_overall' when each
        action was performed.

    Examples
    --------
    Extract clock-time features per gamestate slot::

        from silly_kicks.vaep.features import time

        feats = time(states)
    """
    match_time_at_period_start = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
    timedf = actions[["period_id", "time_seconds"]].copy()
    timedf["time_seconds_overall"] = (timedf.period_id.map(match_time_at_period_start) * 60) + timedf.time_seconds
    return timedf


def time_delta(gamestates: GameStates) -> Features:
    """Get the number of seconds between the last and previous actions.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with a column 'time_delta_i' for each <nb_prev_actions>
        containing the number of seconds between action ai and action a0.

    Examples
    --------
    Extract time-since-previous-action features per gamestate slot::

        from silly_kicks.vaep.features import time_delta

        feats = time_delta(states)
    """
    a0 = gamestates[0]
    dt = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dt["time_delta_" + (str(i + 1))] = a0.time_seconds - a.time_seconds
    return dt


def speed(gamestates: GameStates) -> Features:
    """Get the speed at which the ball moved during the previous actions.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with columns 'speedx_a0i', 'speedy_a0i', 'speed_a0i'
        for each <nb_prev_actions> containing the ball speed in m/s  between
        action ai and action a0.

    Examples
    --------
    Extract speed (space_delta / time_delta) features per gamestate slot::

        from silly_kicks.vaep.features import speed

        feats = speed(states)
    """
    a0 = gamestates[0]
    speed = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a0.start_x
        dy = a.end_y - a0.start_y
        dt = a0.time_seconds - a.time_seconds
        dt[dt <= 0] = 1e-6
        speed["speedx_a0" + (str(i + 1))] = dx.abs() / dt
        speed["speedy_a0" + (str(i + 1))] = dy.abs() / dt
        speed["speed_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2) / dt
    return speed
