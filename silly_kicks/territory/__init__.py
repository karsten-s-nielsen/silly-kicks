"""silly-kicks territorial-dominance metric (TF-54).

Event-only, per-(player, match): the "Van Dijk" territorial-dominance lens. For a defender, build the
trimmed convex hull of their own-half defensive-action locations, then value the OPPONENT passes whose
destination lands inside that hull -- threat CONCEDED (completed) vs threat PREVENTED (failed), by an
INJECTED fitted ``ExpectedThreat`` (silly-kicks ships no xT model; port pattern).

Hexagonal / event-only: imports ``silly_kicks.spadl`` + ``silly_kicks.id_compat`` +
``silly_kicks.xthreat`` (the injected model type) + scipy ONLY; NEVER ``silly_kicks.tracking`` (pinned
by ``tests/territory/test_import_allowlist.py``). Additive -- no VAEP/tracking retrain.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._columns import (
    TERRITORY_COLUMNS,
    TERRITORY_HULL_SOURCE_VALUES,
    TERRITORY_METHODS,
    TERRITORY_METRIC_COLUMNS,
)
from ._compute import compute_territorial_dominance
from ._config import TerritoryParams
from ._report import TerritoryReport

__all__ = [
    "TERRITORY_COLUMNS",
    "TERRITORY_HULL_SOURCE_VALUES",
    "TERRITORY_METHODS",
    "TERRITORY_METRIC_COLUMNS",
    "TerritoryParams",
    "TerritoryReport",
    "compute_territorial_dominance",
]
