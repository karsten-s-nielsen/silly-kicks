"""RestDefenseGhostReport -- per-frame conservation for the Layer-3 counterfactual (TF-60, ADR-089).

Field names mirror ``GkdvReport`` / ``RestDefenseReport`` (``n_frames_in`` / ``n_frames_scored`` /
``drop_reasons``). Conservation (``n_frames_scored + sum(drop_reasons.values()) == n_frames_in``) is a
CI gate, not a dataclass property (as in gkdv) -- a non-simulatable frame is dropped-AND-COUNTED,
never scored as Delta = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._config import RestDefenseParams


@dataclass(frozen=True)
class RestDefenseGhostReport:
    """Run-level audit for :func:`build_restdefense_ghost_frames`.

    Examples
    --------
    Conservation holds exactly -- every frame is scored or counted under one drop reason:

    >>> from silly_kicks.restdefense import RestDefenseParams, RestDefenseGhostReport
    >>> r = RestDefenseGhostReport(RestDefenseParams(), 10, 6, {"not_committed_forward": 4})
    >>> r.n_frames_scored + sum(r.drop_reasons.values()) == r.n_frames_in
    True
    """

    params: RestDefenseParams
    n_frames_in: int
    n_frames_scored: int
    drop_reasons: dict = field(default_factory=dict)
