"""xT-GK v2 — honest possession-value surface V(z,p). See ADR-036 / NOTICE."""

from silly_kicks.xtgk._diagnostics import DeepZoneGateReport, GateConfig, run_deep_zone_gate
from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._possession_value import (
    DeltaV,
    PossessionValue,
    PressureLevel,
    State,
    zone_of,
)
from silly_kicks.xtgk._pressure_levels import PressureLevels
from silly_kicks.xtgk._validate import (
    PossessionValueInputDiagnosis,
    validate_possession_value_input,
)

__all__ = [
    "DeepZoneGateReport",
    "DeltaV",
    "EmpiricalPossessionValue",
    "GateConfig",
    "MarkovPossessionValue",
    "PossessionValue",
    "PossessionValueInputDiagnosis",
    "PressureLevel",
    "PressureLevels",
    "State",
    "run_deep_zone_gate",
    "validate_possession_value_input",
    "zone_of",
]
