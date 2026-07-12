"""xT-GK v2 — honest possession-value surface V(z,p). See ADR-036 / NOTICE."""

from silly_kicks.xtgk._diagnostics import (
    DeepZoneGateReport,
    GateConfig,
    frame_present_null_pressure_count,
    ood_rate_by_source,
    run_deep_zone_gate,
    run_gate_both_orientations,
    run_gate_with_ladder,
)
from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._metric import compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import (
    DeltaV,
    PossessionValue,
    PressureLevel,
    State,
    finite_coord_mask,
    mirror_zone,
    zone_of,
)
from silly_kicks.xtgk._pressure_levels import PressureLevels, coalesce_frame_present_null_pressure
from silly_kicks.xtgk._resolved_geometry import (
    GK_GEOMETRY_SOURCE_COLUMN,
    GK_GEOMETRY_SOURCES,
    apply_resolved_gk_geometry,
)
from silly_kicks.xtgk._retention import GkRetentionModel, RetentionModel
from silly_kicks.xtgk._retention_features import extract_retention_features
from silly_kicks.xtgk._retention_labels import retains
from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue, MirroredTurnoverCost, TurnoverCost
from silly_kicks.xtgk._validate import (
    PossessionValueInputDiagnosis,
    validate_possession_value_input,
)

__all__ = [
    "GK_GEOMETRY_SOURCES",
    "GK_GEOMETRY_SOURCE_COLUMN",
    "DeepZoneGateReport",
    "DeltaV",
    "EmpiricalPossessionValue",
    "EmpiricalTurnoverValue",
    "GateConfig",
    "GkRetentionModel",
    "MarkovPossessionValue",
    "MirroredTurnoverCost",
    "PossessionValue",
    "PossessionValueInputDiagnosis",
    "PressureLevel",
    "PressureLevels",
    "RetentionModel",
    "State",
    "TurnoverCost",
    "apply_resolved_gk_geometry",
    "coalesce_frame_present_null_pressure",
    "compute_xt_gk_v2",
    "extract_retention_features",
    "finite_coord_mask",
    "frame_present_null_pressure_count",
    "mirror_zone",
    "ood_rate_by_source",
    "retains",
    "run_deep_zone_gate",
    "run_gate_both_orientations",
    "run_gate_with_ladder",
    "validate_possession_value_input",
    "zone_of",
]
