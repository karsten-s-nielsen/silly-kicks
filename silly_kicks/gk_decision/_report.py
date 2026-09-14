"""GkDecisionReport -- conserving census for compute_gk_decision_value (TF-62).

Every GK decision is dropped-and-counted, never a fabricated 0 (ADR-042); the report is a frozen
conserving Report (ADR-043 ``GkdvReport`` idiom):
``n_decisions_in == n_scored + n_too_few_options + n_no_unique_chosen + n_chosen_unvalued
+ n_no_frame + n_fov_cropped`` (the last two are the reconstruction adapter's own drops, PR2).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._config import GkDecisionParams


@dataclass(frozen=True)
class GkDecisionReport:
    """Per-``compute_gk_decision_value`` census over the GK-decision population.

    Examples
    --------
    >>> from silly_kicks.gk_decision import GkDecisionParams, GkDecisionReport
    >>> r = GkDecisionReport(GkDecisionParams(), n_decisions_in=10, n_scored=6,
    ...                      n_too_few_options=2, n_no_unique_chosen=1, n_chosen_unvalued=1)
    >>> (r.n_scored + r.n_too_few_options + r.n_no_unique_chosen
    ...  + r.n_chosen_unvalued == r.n_decisions_in)
    True
    """

    params: GkDecisionParams
    n_decisions_in: int
    n_scored: int
    n_too_few_options: int
    n_no_unique_chosen: int
    n_chosen_unvalued: int  # the (unique) chosen option had a non-finite EV -> cannot score
    n_no_frame: int = 0  # reconstruction: no linked freeze-frame / unresolvable frame (PR2)
    n_fov_cropped: int = 0  # reconstruction: keeper neighbourhood under-observed (PR2; ADR-077)
