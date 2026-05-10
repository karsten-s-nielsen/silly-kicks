"""Pitch control models — three-flavor spatial control computation.

Public API:
- compute_pitch_control(frame, attacking_team_id, ...) -> PitchControlSurface
- compute_pitch_control_at_points(frame, targets, ...) -> np.ndarray
- PitchControlSurface — rich frozen dataclass return type
- SpearmanParams / FernandezBornnParams / VoronoiParams
- Method type alias

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md
and ADR-008 for architectural decisions.
"""

from __future__ import annotations

from ._dispatch import compute_pitch_control, compute_pitch_control_at_points
from ._params import (
    FernandezBornnParams,
    Method,
    PitchControlParams,
    SpearmanParams,
    VoronoiParams,
    validate_params_for_method,
)
from ._spearman import compute_tti
from ._surface import PitchControlSurface

__all__ = [
    "FernandezBornnParams",
    "Method",
    "PitchControlParams",
    "PitchControlSurface",
    "SpearmanParams",
    "VoronoiParams",
    "compute_pitch_control",
    "compute_pitch_control_at_points",
    "compute_tti",
    "validate_params_for_method",
]
