"""TerritoryReport -- coverage census for compute_territorial_dominance (TF-54).

A defender with a resolvable trimmed hull is SCORED; one whose hull is degenerate (< 3 non-collinear
own-half defensive actions) is DEGENERATE and its row is dropped-and-counted here (never a fabricated
0/NaN -- ADR-042). Conservation (``n_scored + n_degenerate_hull + n_no_actions == n_players_in``) is
asserted by the compute tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._config import TerritoryParams


@dataclass(frozen=True)
class TerritoryReport:
    """Per-``compute_territorial_dominance`` census over the defender population.

    Examples
    --------
    >>> from silly_kicks.territory import TerritoryParams
    >>> from silly_kicks.territory._report import TerritoryReport
    >>> r = TerritoryReport(TerritoryParams(), n_players_in=5, n_scored=4, n_degenerate_hull=1,
    ...                     n_no_actions=0, n_passes_considered=200, n_passes_into_hull=37)
    >>> r.n_scored + r.n_degenerate_hull + r.n_no_actions == r.n_players_in
    True
    """

    params: TerritoryParams
    n_players_in: int
    n_scored: int
    n_degenerate_hull: int
    n_no_actions: int
    n_passes_considered: int
    n_passes_into_hull: int
