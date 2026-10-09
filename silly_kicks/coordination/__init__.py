"""TF-58 team-coordination dynamics.

Relative phase, cross-correlation, vector coding, spectral median frequency, cluster-phase synchrony and
relative stretch between the two teams (and their players), each scored against a surrogate baseline. Pure
kernels live in ``_kernels/`` (numpy/scipy/stdlib/numba only); this module is the whole public surface (spec
§7.2). Everything else in the package is private (a leading ``_``).
"""

from __future__ import annotations

from silly_kicks.coordination._catalog import DEFAULT_PAIRS, PairSpec
from silly_kicks.coordination._columns import (
    COORD_LEVELS,
    COORD_SIGNALS,
    COORD_SOURCE_VALUES,
    COORD_SURROGATE_SOURCE_VALUES,
    COORD_WINDOW_COLUMNS,
    COORDINATION_CLUSTER_PLAYER_COLUMNS,
    COORDINATION_CLUSTER_PLAYER_KEYS,
    COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    COORDINATION_CLUSTER_TEAM_COLUMNS,
    COORDINATION_CLUSTER_TEAM_KEYS,
    COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
    COORDINATION_PAIR_COLUMNS,
    COORDINATION_PAIR_KEYS,
    COORDINATION_PAIR_METRIC_COLUMNS,
    COORDINATION_PAIR_PHASE_COLUMNS,
    COORDINATION_PAIR_PHASE_KEYS,
    COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
    COORDINATION_RSI_COLUMNS,
    COORDINATION_RSI_KEYS,
    COORDINATION_RSI_METRIC_COLUMNS,
    COORDINATION_SPECTRAL_COLUMNS,
    COORDINATION_SPECTRAL_KEYS,
    COORDINATION_SPECTRAL_METRIC_COLUMNS,
    COORDINATION_TEAM_SYNC_COLUMNS,
    COORDINATION_TEAM_SYNC_KEYS,
    COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
    DEFAULT_LEVELS,
)
from silly_kicks.coordination._compute import (
    CoordinationResult,
    compute_cluster_phase,
    compute_coherence,
    compute_cross_correlation,
    compute_relative_phase,
    compute_relative_stretch,
    compute_spectral,
    compute_team_coordination,
    compute_vector_coding,
)
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._report import CoordinationCoverageWarning, CoordinationReport
from silly_kicks.coordination._series import compute_coordination_series
from silly_kicks.coordination._signals import CoordinationSignals, build_coordination_signals
from silly_kicks.coordination._windows import (
    period_windows,
    possession_windows_from_actions,
    possession_windows_from_frames,
)

# sorted (ruff RUF022), so no section comments here: the import block above groups the surface.
__all__ = [
    "COORDINATION_CLUSTER_PLAYER_COLUMNS",
    "COORDINATION_CLUSTER_PLAYER_KEYS",
    "COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS",
    "COORDINATION_CLUSTER_TEAM_COLUMNS",
    "COORDINATION_CLUSTER_TEAM_KEYS",
    "COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS",
    "COORDINATION_PAIR_COLUMNS",
    "COORDINATION_PAIR_KEYS",
    "COORDINATION_PAIR_METRIC_COLUMNS",
    "COORDINATION_PAIR_PHASE_COLUMNS",
    "COORDINATION_PAIR_PHASE_KEYS",
    "COORDINATION_PAIR_PHASE_METRIC_COLUMNS",
    "COORDINATION_RSI_COLUMNS",
    "COORDINATION_RSI_KEYS",
    "COORDINATION_RSI_METRIC_COLUMNS",
    "COORDINATION_SPECTRAL_COLUMNS",
    "COORDINATION_SPECTRAL_KEYS",
    "COORDINATION_SPECTRAL_METRIC_COLUMNS",
    "COORDINATION_TEAM_SYNC_COLUMNS",
    "COORDINATION_TEAM_SYNC_KEYS",
    "COORDINATION_TEAM_SYNC_METRIC_COLUMNS",
    "COORD_LEVELS",
    "COORD_SIGNALS",
    "COORD_SOURCE_VALUES",
    "COORD_SURROGATE_SOURCE_VALUES",
    "COORD_WINDOW_COLUMNS",
    "DEFAULT_LEVELS",
    "DEFAULT_PAIRS",
    "CoordinationCoverageWarning",
    "CoordinationParams",
    "CoordinationReport",
    "CoordinationResult",
    "CoordinationSignals",
    "PairSpec",
    "build_coordination_signals",
    "compute_cluster_phase",
    "compute_coherence",
    "compute_coordination_series",
    "compute_cross_correlation",
    "compute_relative_phase",
    "compute_relative_stretch",
    "compute_spectral",
    "compute_team_coordination",
    "compute_vector_coding",
    "period_windows",
    "possession_windows_from_actions",
    "possession_windows_from_frames",
]
