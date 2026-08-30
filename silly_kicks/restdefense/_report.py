"""RestDefenseReport -- conservation bookkeeping for compute_rest_defense (TF-60, ADR-080).

Field names mirror ``GkdvReport`` (``n_frames_in`` / ``n_frames_scored`` / ``drop_reasons``).
Conservation (``n_frames_scored + sum(drop_reasons.values()) == n_frames_in``) is asserted by a CI
gate, not by a dataclass property (as in gkdv) -- an unscoreable sample is dropped-AND-COUNTED, never
silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._config import RestDefenseParams


@dataclass(frozen=True)
class RestDefenseReport:
    """Per-``compute_rest_defense`` conservation report over the action-grid population.

    Examples
    --------
    Conservation holds exactly -- every action is either scored or counted under one drop reason:

    >>> from silly_kicks.restdefense import RestDefenseParams, RestDefenseReport
    >>> r = RestDefenseReport(RestDefenseParams(), 10, 7, {"not_committed_forward": 3})
    >>> r.n_frames_scored + sum(r.drop_reasons.values()) == r.n_frames_in
    True
    """

    params: RestDefenseParams
    n_frames_in: int
    n_frames_scored: int
    drop_reasons: dict = field(default_factory=dict)
