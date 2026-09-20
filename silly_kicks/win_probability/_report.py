"""Conservation census for ``compute_win_probability`` (ADR-042)."""

from __future__ import annotations

import dataclasses

from ._config import WinProbabilityParams


@dataclasses.dataclass(frozen=True)
class WinProbabilityReport:
    """Match + action census. ``n_matches_scored + n_matches_excluded_not_two_teams == n_matches_in``.

    Examples
    --------
    >>> from silly_kicks.win_probability import WinProbabilityParams, WinProbabilityReport
    >>> r = WinProbabilityReport(
    ...     WinProbabilityParams.default(),
    ...     n_matches_in=2,
    ...     n_matches_scored=2,
    ...     n_matches_excluded_not_two_teams=0,
    ...     n_actions=50,
    ...     n_actions_unresolved=0,
    ... )
    >>> r.n_matches_scored + r.n_matches_excluded_not_two_teams == r.n_matches_in
    True
    """

    params: WinProbabilityParams
    n_matches_in: int
    n_matches_scored: int
    n_matches_excluded_not_two_teams: int
    n_actions: int
    n_actions_unresolved: int

    def __post_init__(self) -> None:
        if self.n_matches_scored + self.n_matches_excluded_not_two_teams != self.n_matches_in:
            raise ValueError(
                "WinProbabilityReport: match conservation violated "
                f"({self.n_matches_scored} + {self.n_matches_excluded_not_two_teams} != {self.n_matches_in})"
            )
