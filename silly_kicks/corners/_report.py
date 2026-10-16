"""CornersReport -- identification/attribution census for the TF-46 corner pipeline (Arm 1).

Field names mirror ``ShotStoppingReport`` / ``RestDefenseReport``. A corner excluded as short/unclear
is counted here, never silently dropped (ADR-042). Conservation
(``n_direct + n_short + n_unclear == n_corners``) is asserted by a doctest + a unit gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CornersReport:
    """Per-pipeline-run census over the identified corner population.

    Examples
    --------
    >>> from silly_kicks.corners import CornersReport
    >>> r = CornersReport(n_corners=10, n_direct=7, n_short=2, n_unclear=1, n_winner_unresolved=3)
    >>> r.n_direct + r.n_short + r.n_unclear == r.n_corners
    True
    """

    n_corners: int
    n_direct: int
    n_short: int
    n_unclear: int
    n_winner_unresolved: int
