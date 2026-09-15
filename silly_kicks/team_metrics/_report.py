"""TeamKpiReport -- match/shot-attribution census for compute_team_kpis (TF-52).

Field names mirror ``ShotStoppingReport``. A game with exactly two team ids is SCORED; a game whose
actions carry != 2 distinct team ids is EXCLUDED and COUNTED (never silently dropped -- ADR-042).
Conservation (``n_matches_scored + n_matches_excluded_not_two_teams == n_matches_in`` and
``n_shots_with_xg + n_shots_null_xg == n_shots``) is asserted by a CI gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._config import TeamKpiParams


@dataclass(frozen=True)
class TeamKpiReport:
    """Per-``compute_team_kpis`` census over the match population + the high-opportunity-shot domain.

    Examples
    --------
    >>> from silly_kicks.team_metrics import TeamKpiParams, TeamKpiReport
    >>> r = TeamKpiReport(TeamKpiParams(), 10, 9, 1, 25, 23, 2)
    >>> r.n_matches_scored + r.n_matches_excluded_not_two_teams == r.n_matches_in
    True
    >>> r.n_shots_with_xg + r.n_shots_null_xg == r.n_shots
    True
    """

    params: TeamKpiParams
    n_matches_in: int
    n_matches_scored: int
    n_matches_excluded_not_two_teams: int
    n_shots: int
    n_shots_with_xg: int
    n_shots_null_xg: int
