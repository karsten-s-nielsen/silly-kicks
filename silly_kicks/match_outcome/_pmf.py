"""Exact Poisson-binomial goal PMF + the two-team outcome simplex (TF-53 Rung 1).

``goal_count_pmf`` is the exact per-team goal distribution over per-shot Bernoullis (an O(n^2) DP
convolution) -- NOT ``Poisson(sum(xg))``, which discards chance quality. ``match_outcome_probabilities``
builds the joint scoreline distribution and sums the win/draw/loss simplex; the cross-team dependence
(``team_dependence``) is applied here (Rung 3a wires ``dixon_coles``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._config import MatchOutcomeParams

_DEFAULT_PARAMS = MatchOutcomeParams.default()


def goal_count_pmf(shot_xgs: Sequence[float] | np.ndarray) -> np.ndarray:
    """Exact Poisson-binomial PMF: ``pmf[k] = P(exactly k goals)``.

    An empty shot list yields ``array([1.0])`` (a team with no shots scores 0 with probability 1).

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks.match_outcome import goal_count_pmf
    >>> pmf = goal_count_pmf([0.3, 0.4])
    >>> bool(np.isclose(pmf.sum(), 1.0)) and pmf.shape == (3,)
    True
    """
    pmf = np.array([1.0], dtype="float64")
    for xg in shot_xgs:
        p = float(xg)
        pmf = np.convolve(pmf, np.array([1.0 - p, p], dtype="float64"))
    return pmf


def _independent_joint(home_pmf: np.ndarray, away_pmf: np.ndarray) -> np.ndarray:
    """Joint scoreline P(home=i, away=j) under team independence -- the outer product of marginals."""
    return np.outer(home_pmf, away_pmf)


def match_outcome_probabilities(
    home_pmf: np.ndarray, away_pmf: np.ndarray, *, params: MatchOutcomeParams = _DEFAULT_PARAMS
) -> tuple[float, float, float]:
    """``(p_home_win, p_draw, p_away_win)`` from two goal PMFs.

    ``params.team_dependence`` selects the joint: ``"independent"`` = the product of marginals;
    ``"dixon_coles"`` = the fitted low-score tau reweighting (Rung 3a).

    Examples
    --------
    >>> from silly_kicks.match_outcome import goal_count_pmf, match_outcome_probabilities
    >>> ph, pd_, pa = match_outcome_probabilities(goal_count_pmf([0.5]), goal_count_pmf([0.5]))
    >>> bool(abs(ph - pa) < 1e-12) and bool(abs(ph + pd_ + pa - 1.0) < 1e-12)
    True
    """
    home = np.asarray(home_pmf, dtype="float64")
    away = np.asarray(away_pmf, dtype="float64")
    if params.team_dependence == "independent":
        joint = _independent_joint(home, away)
    else:  # "dixon_coles" -- Rung 3a (lazy import keeps the core dependency-free)
        from ._dependence import apply_dependence, resolve_rho

        joint = apply_dependence(home, away, rho=resolve_rho(params))
    i = np.arange(joint.shape[0])[:, None]
    j = np.arange(joint.shape[1])[None, :]
    p_home = float(joint[i > j].sum())
    p_draw = float(joint[i == j].sum())
    p_away = float(joint[i < j].sum())
    return p_home, p_draw, p_away
