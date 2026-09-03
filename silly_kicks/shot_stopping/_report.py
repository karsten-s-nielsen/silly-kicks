"""ShotStoppingReport -- attribution-coverage census for compute_shot_stopping (TF-59 PR2).

Field names mirror ``RestDefenseReport`` / ``GkdvReport``. A shot faced with a resolved defending
keeper is ATTRIBUTED; one with a ``pd.NA`` ``defending_gk_player_id`` is UNATTRIBUTED (surfaced here,
never silently dropped nor misattributed -- ADR-042). Conservation
(``n_shots_attributed + n_shots_unattributed == n_shots_faced``) is asserted by a CI gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._config import ShotStoppingParams


@dataclass(frozen=True)
class ShotStoppingReport:
    """Per-``compute_shot_stopping`` attribution census over the on-target-shots-faced population.

    Examples
    --------
    >>> from silly_kicks.shot_stopping import ShotStoppingParams, ShotStoppingReport
    >>> r = ShotStoppingReport(ShotStoppingParams(), 20, 18, 2)
    >>> r.n_shots_attributed + r.n_shots_unattributed == r.n_shots_faced
    True
    """

    params: ShotStoppingParams
    n_shots_faced: int
    n_shots_attributed: int
    n_shots_unattributed: int
