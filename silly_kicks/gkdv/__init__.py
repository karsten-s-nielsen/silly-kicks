"""GKDV -- GK Deterrent Value (TF-19, ADR-043).

Counterfactual valuation of goalkeeper positioning: how much does the ACTUAL keeper's
position change the attacking team's accessible space and threat, relative to a
league-average "ghost" keeper in the same frame state? Arms are defined in
attacker-value units as ``actual - ghost``, so **negative = deterrent** uniformly.

Depends on ``silly_kicks.tracking`` PUBLIC seams, on the repo-wide public
``silly_kicks.id_compat`` (ADR-019 requires every consumer to route id comparisons through
it), and on exactly ONE private tracking symbol: ``_das._pin_attacking_direction``, confined
to ``_das_port.py``, which has no public meaning because it encodes what the optional
``accessible-space`` dependency expects of its input. Never the reverse -- ``tracking`` must
not import ``gkdv``; the probe consumes ghost positions as DATA. Both directions are pinned
by ``tests/gkdv/test_import_allowlist.py``.

See NOTICE for full bibliographic citations.
"""

from ._arms import delta_das, delta_threat_suppression
from ._engine import GkdvParams, GkdvReport, build_ghost_frames, provenance_to_targets
from ._metric import aggregate_by_keeper
from ._validate import (
    EXPECTED_DIRECTION,
    ICC_ANCHORS,
    TERCILE_SEPARATION_M,
    behavioural_anchoring_verdict,
)

__all__ = [
    "EXPECTED_DIRECTION",
    "ICC_ANCHORS",
    "TERCILE_SEPARATION_M",
    "GkdvParams",
    "GkdvReport",
    "aggregate_by_keeper",
    "behavioural_anchoring_verdict",
    "build_ghost_frames",
    "delta_das",
    "delta_threat_suppression",
    "provenance_to_targets",
]
