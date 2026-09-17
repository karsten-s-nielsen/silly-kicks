"""MatchOutcomeReport -- match/shot census for compute_match_outcome (TF-53).

Field names mirror ``ShotStoppingReport``. A game with exactly two team ids is SCORED; a game whose
actions carry != 2 distinct team ids is EXCLUDED and COUNTED (never silently dropped -- ADR-042).
Conservation (``n_matches_scored + n_matches_excluded_not_two_teams == n_matches_in`` and
``n_shots_with_xg + n_shots_null_xg == n_shots``) is asserted by a CI gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._config import MatchOutcomeParams


@dataclass(frozen=True)
class MatchOutcomeReport:
    """Per-``compute_match_outcome`` census over the match + shot-xG population.

    ``n_own_goals`` (result-only, ADR-018) is reported so a consumer can reconcile the pure-xG
    outcome probabilities to the realized scoreline (own goals carry no xG; spec §7, option A).

    Examples
    --------
    >>> from silly_kicks.match_outcome import MatchOutcomeParams, MatchOutcomeReport
    >>> r = MatchOutcomeReport(MatchOutcomeParams(), 10, 9, 1, 240, 235, 5, 3)
    >>> r.n_matches_scored + r.n_matches_excluded_not_two_teams == r.n_matches_in
    True
    >>> r.n_shots_with_xg + r.n_shots_null_xg == r.n_shots
    True
    """

    params: MatchOutcomeParams
    n_matches_in: int
    n_matches_scored: int
    n_matches_excluded_not_two_teams: int
    n_shots: int
    n_shots_with_xg: int
    n_shots_null_xg: int
    n_own_goals: int
