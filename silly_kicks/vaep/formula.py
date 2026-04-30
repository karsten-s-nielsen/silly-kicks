"""Implements the formula of the VAEP framework."""

import pandas as pd


def _prev(x: pd.Series) -> pd.Series:
    prev_x = x.shift(1)
    prev_x[:1] = x.values[0]
    return prev_x  # type: ignore[reportReturnType]


_SAMEPHASE_NB: int = 10  # number of subsequent actions considered same phase
_PENALTY_SCORING_PROB: float = 0.792453  # empirical penalty conversion rate
_CORNER_SCORING_PROB: float = 0.046500  # empirical corner scoring rate


def offensive_value(actions: pd.DataFrame, scores: pd.Series, concedes: pd.Series) -> pd.Series:
    r"""Compute the offensive value of each action.

    VAEP defines the *offensive value* of an action as the change in scoring
    probability before and after the action.

    .. math::

      \Delta P_{score}(a_{i}, t) = P^{k}_{score}(S_i, t) - P^{k}_{score}(S_{i-1}, t)

    where :math:`P_{score}(S_i, t)` is the probability that team :math:`t`
    which possesses the ball in state :math:`S_i` will score in the next 10
    actions.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action.
    scores : pd.Series
        The probability of scoring from each corresponding game state.
    concedes : pd.Series
        The probability of conceding from each corresponding game state.

    Returns
    -------
    pd.Series
        The offensive value of each action.

    Examples
    --------
    Compute the offensive component of VAEP from estimated probabilities::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.formula import offensive_value

        actions_with_names = add_names(actions)
        # p_scores, p_concedes: pd.Series, one row per action, e.g. from
        # VAEP._estimate_probabilities or any binary classifier of choice.
        ov = offensive_value(actions_with_names, p_scores, p_concedes)
        # Returns a pd.Series of per-action offensive values.
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_scores = (_prev(scores) * sameteam + _prev(concedes) * (~sameteam)).astype(float)

    # if the previous action was too long ago, the odds of scoring are now 0
    toolong_idx = abs(actions.time_seconds - _prev(actions.time_seconds)) > _SAMEPHASE_NB
    prev_scores[toolong_idx] = 0.0

    # if the previous action was a goal, the odds of scoring are now 0
    prevgoal_idx = (_prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])) & (
        _prev(actions.result_name) == "success"
    )
    prev_scores[prevgoal_idx] = 0.0

    # fixed odds of scoring when penalty
    penalty_idx = actions.type_name == "shot_penalty"
    prev_scores[penalty_idx] = _PENALTY_SCORING_PROB

    # fixed odds of scoring when corner
    corner_idx = actions.type_name.isin(["corner_crossed", "corner_short"])
    prev_scores[corner_idx] = _CORNER_SCORING_PROB

    return scores - prev_scores


def defensive_value(actions: pd.DataFrame, scores: pd.Series, concedes: pd.Series) -> pd.Series:
    r"""Compute the defensive value of each action.

    VAEP defines the *defensive value* of an action as the change in conceding
    probability.

    .. math::

      \Delta P_{concede}(a_{i}, t) = P^{k}_{concede}(S_i, t) - P^{k}_{concede}(S_{i-1}, t)

    where :math:`P_{concede}(S_i, t)` is the probability that team :math:`t`
    which possesses the ball in state :math:`S_i` will concede in the next 10
    actions.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action.
    scores : pd.Series
        The probability of scoring from each corresponding game state.
    concedes : pd.Series
        The probability of conceding from each corresponding game state.

    Returns
    -------
    pd.Series
        The defensive value of each action.

    Examples
    --------
    Compute the defensive component of VAEP from estimated probabilities::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.formula import defensive_value

        actions_with_names = add_names(actions)
        dv = defensive_value(actions_with_names, p_scores, p_concedes)
        # Returns a pd.Series of per-action defensive values (sign convention:
        # a successful defensive action that lowers conceding probability is
        # positive).
    """
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_concedes = (_prev(concedes) * sameteam + _prev(scores) * (~sameteam)).astype(float)

    toolong_idx = abs(actions.time_seconds - _prev(actions.time_seconds)) > _SAMEPHASE_NB
    prev_concedes[toolong_idx] = 0.0

    # if the previous action was a goal, the odds of conceding are now 0
    prevgoal_idx = (_prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])) & (
        _prev(actions.result_name) == "success"
    )
    prev_concedes[prevgoal_idx] = 0.0

    return -(concedes - prev_concedes)


def value(actions: pd.DataFrame, Pscores: pd.Series, Pconcedes: pd.Series) -> pd.DataFrame:
    r"""Compute the offensive, defensive and VAEP value of each action.

    The total VAEP value of an action is the difference between that action's
    offensive value and defensive value.

    .. math::

      V_{VAEP}(a_i) = \Delta P_{score}(a_{i}, t) - \Delta P_{concede}(a_{i}, t)

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action.
    Pscores : pd.Series
        The probability of scoring from each corresponding game state.
    Pconcedes : pd.Series
        The probability of conceding from each corresponding game state.

    Returns
    -------
    pd.DataFrame
        The 'offensive_value', 'defensive_value' and 'vaep_value' of each action.

    See Also
    --------
    :func:`~silly_kicks.vaep.formula.offensive_value`: The offensive value
    :func:`~silly_kicks.vaep.formula.defensive_value`: The defensive value

    Examples
    --------
    Compute per-action VAEP values directly from probabilities (without going
    through ``VAEP.rate()``)::

        from silly_kicks.spadl import add_names
        from silly_kicks.vaep.formula import value

        actions_with_names = add_names(actions)
        v = value(actions_with_names, p_scores, p_concedes)
        # v has columns 'offensive_value', 'defensive_value', 'vaep_value';
        # vaep_value = offensive_value + defensive_value per row.
    """
    v = pd.DataFrame()
    v["offensive_value"] = offensive_value(actions, Pscores, Pconcedes)
    v["defensive_value"] = defensive_value(actions, Pscores, Pconcedes)
    v["vaep_value"] = v["offensive_value"] + v["defensive_value"]
    return v
