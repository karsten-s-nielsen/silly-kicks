"""Context feature transformers.

Three transformers: ``team`` (same-team-as-actor mask per gamestate slot),
``player_possession_time`` (per-player rolling possession seconds),
``goalscore`` (cumulative score / opponent / diff at each action).
"""

import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, GameStates, simple

__all__ = ["goalscore", "player_possession_time", "team"]


@simple
def player_possession_time(actions: Actions) -> Features:
    """Get the time (sec) a player was in ball possession before attempting the action.

    We only look at the dribble preceding the action and reset the possession
    time after a defensive interception attempt or a take-on.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'player_possession_time' of each action.

    Examples
    --------
    Extract per-action player-possession duration (seconds) features::

        from silly_kicks.vaep.features import player_possession_time

        feats = player_possession_time(states)
    """
    cur_action = actions[["period_id", "time_seconds", "player_id", "type_id"]]
    prev_action = actions[["period_id", "time_seconds", "player_id", "type_id"]].shift(1)
    df = cur_action.join(prev_action, rsuffix="_prev")
    same_player = df.player_id == df.player_id_prev
    same_period = df.period_id == df.period_id_prev
    prev_dribble = df.type_id_prev == spadlcfg.actiontype_id["dribble"]
    mask = same_period & same_player & prev_dribble
    df.loc[mask, "player_possession_time"] = df.loc[mask, "time_seconds"] - df.loc[mask, "time_seconds_prev"]
    return df[["player_possession_time"]].fillna(0.0)


def team(gamestates: GameStates) -> Features:
    """Check whether the possession changed during the game state.

    For each action in the game state, True if the team that performed the
    action is the same team that performed the last action of the game state;
    otherwise False.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with a column 'team_ai' for each <nb_prev_actions> indicating
        whether the team that performed action a0 is in possession.

    Examples
    --------
    Extract team-of-actor features per gamestate slot::

        from silly_kicks.vaep.features import team

        feats = team(states)
    """
    a0 = gamestates[0]
    teamdf = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        teamdf["team_" + (str(i + 1))] = a.team_id == a0.team_id
    return teamdf


def goalscore(gamestates: GameStates) -> Features:
    """Get the number of goals scored by each team after the action.

    Parameters
    ----------
    gamestates : GameStates
        The gamestates of a game.

    Returns
    -------
    Features
        The number of goals scored by the team performing the last action of the
        game state ('goalscore_team'), by the opponent ('goalscore_opponent'),
        and the goal difference between both teams ('goalscore_diff').

    Examples
    --------
    Extract per-action goal-difference features (own goals scored vs conceded)::

        from silly_kicks.vaep.features import goalscore

        feats = goalscore(states)
    """
    actions = gamestates[0]
    teamA = actions["team_id"].values[0]
    goals = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadlcfg.result_id["success"])
    owngoals = actions["type_name"].str.contains("shot") & (actions["result_id"] == spadlcfg.result_id["owngoal"])
    teamisA = actions["team_id"] == teamA
    teamisB = ~teamisA
    goalsteamA = (goals & teamisA) | (owngoals & teamisB)
    goalsteamB = (goals & teamisB) | (owngoals & teamisA)
    goalscoreteamA = goalsteamA.cumsum() - goalsteamA
    goalscoreteamB = goalsteamB.cumsum() - goalsteamB

    scoredf = pd.DataFrame(index=actions.index)
    scoredf["goalscore_team"] = (goalscoreteamA * teamisA) + (goalscoreteamB * teamisB)
    scoredf["goalscore_opponent"] = (goalscoreteamB * teamisA) + (goalscoreteamA * teamisB)
    scoredf["goalscore_diff"] = scoredf["goalscore_team"] - scoredf["goalscore_opponent"]
    return scoredf
