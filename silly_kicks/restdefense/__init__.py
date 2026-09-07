"""silly-kicks rest-defense structure metrics (TF-60, ADR-080).

Rest defense (*Restverteidigung*) is the defensive rearguard an IN-POSSESSION team keeps while
attacking, to blunt the opponent's counter after a loss. This package ships the descriptive
Layer-1 structure KPIs (numerical superiority behind the ball, rest-defense zone occupancy,
rearguard shape, GK line-height and GK-to-line distance), sampled at the in-possession team's
on-ball action grid so it works on both continuous tracking and StatsBomb-360 freeze-frames.

Hexagonal: consumes ``silly_kicks.tracking`` (and, in later cycles, ``silly_kicks.gkdv``) PUBLIC
seams only; NOTHING imports ``restdefense`` and ``tracking`` must never import it (pinned by
``tests/restdefense/test_import_allowlist.py``). Additive -- no existing feature changes, no VAEP
retrain.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._arms import (
    merge_rest_defense,
    rest_defense_gk_deterrent,
    rest_defense_outfield_deterrent,
)
from ._columns import (
    RD_ARM_COLUMNS,
    RD_ARM_SOURCE_VALUES,
    RD_GK_ARM_COLUMNS,
    RD_GK_DETER_SPACE,
    RD_GK_DETER_THREAT,
    RD_GK_SOURCE,
    RD_LAYER1_COLUMNS,
    RD_LAYER2_COLUMNS,
    RD_METRIC_COLUMNS,
    RD_OUTFIELD_ARM_COLUMNS,
    RD_OUTFIELD_DETER_SPACE,
    RD_OUTFIELD_DETER_THREAT,
    RD_OUTFIELD_SOURCE,
    RD_SAMPLE_KEYS,
)
from ._compute import compute_rest_defense, summarize_rest_defense
from ._config import RestDefenseParams
from ._counterfactual import build_restdefense_ghost_frames
from ._ghost_report import RestDefenseGhostReport
from ._report import RestDefenseReport
from ._wfield import WFieldParams

__all__ = [
    "RD_ARM_COLUMNS",
    "RD_ARM_SOURCE_VALUES",
    "RD_GK_ARM_COLUMNS",
    "RD_GK_DETER_SPACE",
    "RD_GK_DETER_THREAT",
    "RD_GK_SOURCE",
    "RD_LAYER1_COLUMNS",
    "RD_LAYER2_COLUMNS",
    "RD_METRIC_COLUMNS",
    "RD_OUTFIELD_ARM_COLUMNS",
    "RD_OUTFIELD_DETER_SPACE",
    "RD_OUTFIELD_DETER_THREAT",
    "RD_OUTFIELD_SOURCE",
    "RD_SAMPLE_KEYS",
    "RestDefenseGhostReport",
    "RestDefenseParams",
    "RestDefenseReport",
    "WFieldParams",
    "build_restdefense_ghost_frames",
    "compute_rest_defense",
    "merge_rest_defense",
    "rest_defense_gk_deterrent",
    "rest_defense_outfield_deterrent",
    "summarize_rest_defense",
]
