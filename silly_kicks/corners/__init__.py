"""silly-kicks corner analysis -- identification + offence routine inference (TF-46 Arm 1).

Follows the Shaw & Gopaladesikan "Routine Inspection" playbook: identify corners and their windows,
reflect-normalise to one flag, encode each attacking player's run as a probabilistic (initial x target)
zone pair, factor the per-corner run vectors into co-occurring routine features (NMF), cluster them into
a team's distinct routines, and report a descriptive per-routine chance quality. Ships RAW primitives
(ADR-110); archetype naming and which-routine-beats-which ranking stay consumer-side.

Hexagonal but TRACKING-CONSUMING (unlike the event-only ``shot_stopping``): imports
``silly_kicks.spadl``, ``silly_kicks.id_compat``, ``silly_kicks.reflection`` and ``silly_kicks.tracking``
(frames + the TF-79 ball-contact primitive). NOTHING in ``silly_kicks`` imports ``corners`` (a leaf;
pinned by ``tests/corners/test_import_allowlist.py``). Fitted models (``RunZoneModel`` GMM,
``CornerRoutineModel`` NMF) are ADR-011 params-only (sklearn at fit, pure-numpy serve).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._columns import (
    CORNER_CHANCE_COLUMNS,
    CORNER_CHANCE_METRIC_COLUMNS,
    CORNER_IDENTIFY_COLUMNS,
    CORNER_KEYS,
    CORNER_WINDOW_COLUMNS,
)
from ._config import (
    ChanceQualityParams,
    CornerWindowParams,
    RoutineParams,
    RunEncodingParams,
)
from ._effectiveness import aggregate_chance_quality_by_feature, compute_corner_chance_quality
from ._encode import compute_corner_run_encoding
from ._identify import identify_corners
from ._reflect import reflect_corners_to_canonical_side
from ._report import CornersReport
from ._routines import (
    CornerRoutineModel,
    cluster_corner_routines,
    compute_corner_routine_features,
)
from ._windows import add_corner_windows
from ._zones import RunZoneModel

__all__ = [
    "CORNER_CHANCE_COLUMNS",
    "CORNER_CHANCE_METRIC_COLUMNS",
    "CORNER_IDENTIFY_COLUMNS",
    "CORNER_KEYS",
    "CORNER_WINDOW_COLUMNS",
    "ChanceQualityParams",
    "CornerRoutineModel",
    "CornerWindowParams",
    "CornersReport",
    "RoutineParams",
    "RunEncodingParams",
    "RunZoneModel",
    "add_corner_windows",
    "aggregate_chance_quality_by_feature",
    "cluster_corner_routines",
    "compute_corner_chance_quality",
    "compute_corner_routine_features",
    "compute_corner_run_encoding",
    "identify_corners",
    "reflect_corners_to_canonical_side",
]
