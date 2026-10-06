"""Coordination vocabularies, signal catalogue and output schemas (single source; every gate iterates these).

Schemas are plain ``dict[str, str]`` name -> dtype in emitted order (the ``TERRITORY_COLUMNS`` precedent; no
pandera). The dtype vocabulary is ``object`` / ``int64`` / ``Int64`` / ``float64``. Id columns are ``object``
("id-valued"): pair keys are canonical ids, single-entity ids are ``restore_id_dtype``-restored (C22).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from silly_kicks.coordination._kernels._circular import HIST_BIN_LABELS

# --------------------------------------------------------------------------- closed vocabularies
COORD_LEVELS = ("team_team", "cross_variable", "intra_team", "dyad")
DEFAULT_LEVELS = COORD_LEVELS
COORD_WINDOW_KINDS = ("period", "sliding", "possession")
COORD_WINDOW_SOURCES = ("period", "possession_events", "possession_tracking", "caller")
COORD_AXES = ("x", "y", "scalar", "mixed")  # C7 (+ "mixed" for a caller's x-vs-y pair)
COORD_METHOD_FAMILIES = (
    "relative_phase",
    "cross_correlation",
    "vector_coding",
    "coherence",
    "spectral",
    "cluster",
    "team_sync",
    "rsi",
)
COORD_SOURCE_VALUES = (
    "scored",
    "too_short",
    "insufficient_detection",
    "insufficient_players",
    "goal_end_unresolved",
    "not_commensurate",
    "no_possession_role",
    "degenerate_constant",
    "entropy_undefined",
)
COORD_SURROGATE_SOURCE_VALUES = (
    "computed",
    "computed_nonconverged",
    "disabled",
    "segment_too_short",
    "not_scored",
)
COORD_DETECTION_SOURCE_VALUES = ("fully_observed", "detection_aware")
COORD_STOPPAGE_SOURCE_VALUES = ("ball_state", "events", "unavailable")


# --------------------------------------------------------------------------- signal catalogue
@dataclass(frozen=True)
class SignalSpec:
    """A coordination signal's physical unit, kind, axis and scope."""

    unit: Literal["metres", "m^2", "dimensionless"]
    kind: Literal["positional", "magnitude", "binary"]
    axis: Literal["x", "y", "scalar"]
    scope: Literal["team", "player", "match"]


COORD_SIGNALS: MappingProxyType[str, SignalSpec] = MappingProxyType(
    {
        "centroid_x": SignalSpec("metres", "positional", "x", "team"),
        "centroid_y": SignalSpec("metres", "positional", "y", "team"),
        "team_length": SignalSpec("metres", "magnitude", "x", "team"),
        "team_width": SignalSpec("metres", "magnitude", "y", "team"),
        "stretch_index": SignalSpec("metres", "magnitude", "scalar", "team"),
        "stretch_x": SignalSpec("metres", "magnitude", "x", "team"),
        "stretch_y": SignalSpec("metres", "magnitude", "y", "team"),
        "spread": SignalSpec("metres", "magnitude", "scalar", "team"),
        "convex_hull_area": SignalSpec("m^2", "magnitude", "scalar", "team"),
        "defensive_line_x": SignalSpec("metres", "positional", "x", "team"),
        "compactness_x": SignalSpec("metres", "magnitude", "x", "team"),
        "back_line_high_x": SignalSpec("metres", "positional", "x", "team"),
        "player_x": SignalSpec("metres", "positional", "x", "player"),
        "player_y": SignalSpec("metres", "positional", "y", "player"),
        "possession": SignalSpec("dimensionless", "binary", "scalar", "match"),
    }
)
#: The 12 team-scope signals -- the spectral / cluster / team-sync set.
TEAM_SIGNALS = tuple(s for s, spec in COORD_SIGNALS.items() if spec.scope == "team")


# --------------------------------------------------------------------------- metric groups (exact tuples)
def _triple(metric: str) -> tuple[str, str, str]:
    return (f"{metric}_surrogate_mean", f"{metric}_percentile", f"{metric}_excess")


RP_CORE = ("coord_rp_mean_deg", "coord_rp_circ_sd_deg", "coord_rp_resultant_length", "coord_rp_pct_near_in_phase")
RP_HIST = tuple(f"coord_rp_hist_bin_{lab}" for lab in HIST_BIN_LABELS)  # 12
RP_VALID = ("coord_rp_phase_valid_fraction_a", "coord_rp_phase_valid_fraction_b")
RP = RP_CORE + RP_HIST + RP_VALID + _triple("coord_rp_resultant_length") + _triple("coord_rp_pct_near_in_phase")
XC = ("coord_xc_max_abs_r", "coord_xc_lag_s", "coord_xc_r_at_max", "coord_xc_r_lag0", *_triple("coord_xc_max_abs_r"))
VC_CORE = (
    "coord_vc_pct_in_phase",
    "coord_vc_pct_anti_phase",
    "coord_vc_pct_a_phase",
    "coord_vc_pct_b_phase",
    "coord_vc_mean_angle_deg",
    "coord_vc_angle_variability_deg",
    "coord_vc_n_stationary",
)
VC = VC_CORE + _triple("coord_vc_pct_in_phase") + _triple("coord_vc_pct_anti_phase")
COH = ("coord_coh_band_mean", "coord_coh_peak_freq_cpm", "coord_coh_n_segments", *_triple("coord_coh_band_mean"))
COVERAGE = (
    "coord_duration_s",
    "coord_n_samples",
    "coord_n_segments",
    "coord_observed_fraction_a",
    "coord_observed_fraction_b",
    "coord_detected_share",  # the min side share the insufficient_detection gate tests (spec 7.11, A-08)
)

# --------------------------------------------------------------------------- dtype assignment
_OBJECT_IDS = frozenset(
    {
        "game_id",
        "team_id",
        "player_id",
        "team_a_id",
        "team_b_id",
        "player_a_id",
        "player_b_id",
        "attacking_team_id",
        "terminal_team_id",
    }
)
_OBJECT_CATEGORICALS = frozenset(
    {"window_kind", "window_source", "level", "signal", "signal_a", "signal_b", "axis", "terminal_action"}
)
_INT64_COUNTS = frozenset({"coord_n_samples", "coord_n_segments", "coord_vc_n_stationary", "coord_coh_n_segments"})
_NULLABLE_INT = frozenset({"window_id", "phase_index", "n_phases"})


def _dtype_for(name: str) -> str:
    if name in _OBJECT_IDS or name in _OBJECT_CATEGORICALS or name.endswith("_source"):
        return "object"
    if name == "period_id":
        return "int64"
    if name in _NULLABLE_INT or name in _INT64_COUNTS:
        return "Int64"
    return "float64"


def _schema(columns: tuple[str, ...]) -> dict[str, str]:
    return {c: _dtype_for(c) for c in columns}


# --------------------------------------------------------------------------- window schema
COORD_WINDOW_COLUMNS = _schema(
    (
        "game_id",
        "period_id",
        "window_kind",
        "window_id",
        "window_source",
        "start_time_s",
        "end_time_s",
        "attacking_team_id",
        "terminal_action",
        "terminal_team_id",
        "n_phases",
    )
)

# --------------------------------------------------------------------------- 7 contract families
COORDINATION_PAIR_KEYS = (
    "game_id",
    "period_id",
    "window_kind",
    "window_id",
    "level",
    "signal_a",
    "signal_b",
    "axis",
    "team_a_id",
    "team_b_id",
    "player_a_id",
    "player_b_id",
)
COORDINATION_PAIR_METRIC_COLUMNS = RP + XC + VC + COH + COVERAGE
COORDINATION_PAIR_COLUMNS = _schema(
    COORDINATION_PAIR_KEYS
    + COORDINATION_PAIR_METRIC_COLUMNS
    + (
        "coord_rp_source",
        "coord_xc_source",
        "coord_vc_source",
        "coord_coh_source",
        "coord_rp_surrogate_source",
        "coord_xc_surrogate_source",
        "coord_vc_surrogate_source",
        "coord_coh_surrogate_source",
        "coord_detection_source",
        "coord_stoppage_source",
    )
)

COORDINATION_PAIR_PHASE_KEYS = (*COORDINATION_PAIR_KEYS, "phase_index")
COORDINATION_PAIR_PHASE_METRIC_COLUMNS = (
    RP_CORE + RP_HIST + VC_CORE + ("coord_duration_s", "coord_n_samples", "coord_detected_share")
)
COORDINATION_PAIR_PHASE_COLUMNS = _schema(
    COORDINATION_PAIR_PHASE_KEYS
    + COORDINATION_PAIR_PHASE_METRIC_COLUMNS
    + ("coord_rp_source", "coord_vc_source", "coord_detection_source", "coord_stoppage_source")
)

COORDINATION_SPECTRAL_KEYS = ("game_id", "period_id", "window_kind", "window_id", "team_id", "signal")
COORDINATION_SPECTRAL_METRIC_COLUMNS = (
    "coord_median_freq_cpm",
    "coord_duration_s",
    "coord_n_segments",
    "coord_detected_share",
)
COORDINATION_SPECTRAL_COLUMNS = _schema(
    COORDINATION_SPECTRAL_KEYS
    + COORDINATION_SPECTRAL_METRIC_COLUMNS
    + ("coord_spectral_source", "coord_detection_source", "coord_stoppage_source")
)

COORDINATION_CLUSTER_TEAM_KEYS = ("game_id", "period_id", "window_kind", "window_id", "team_id", "axis")
COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS = (
    "coord_rho_group_mean",
    "coord_rho_group_sd",
    "coord_rho_group_sampen",
    "coord_n_players_mean",
    "coord_rho_group_mean_surrogate_mean",
    "coord_rho_group_mean_percentile",
    "coord_rho_group_mean_excess",
    "coord_duration_s",
    "coord_n_samples",
    "coord_observed_fraction",
    "coord_detected_share",
)
COORDINATION_CLUSTER_TEAM_COLUMNS = _schema(
    COORDINATION_CLUSTER_TEAM_KEYS
    + COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS
    + (
        "coord_cluster_source",
        "coord_cluster_sampen_source",  # A-20: SampEn's own token, never the row's
        "coord_cluster_surrogate_source",
        "coord_detection_source",
        "coord_stoppage_source",
    )
)

COORDINATION_CLUSTER_PLAYER_KEYS = (*COORDINATION_CLUSTER_TEAM_KEYS[:5], "player_id", "axis")
COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS = (
    "coord_phi_mean_deg",
    "coord_rho_k",
    "coord_phi_sd_deg",
    "coord_phi_sampen",
    "coord_on_pitch_s",
    "coord_detected_share",
)
COORDINATION_CLUSTER_PLAYER_COLUMNS = _schema(
    COORDINATION_CLUSTER_PLAYER_KEYS
    + COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS
    + (
        "coord_cluster_player_source",
        "coord_cluster_player_sampen_source",  # A-20
        "coord_detection_source",
        "coord_stoppage_source",
    )
)

COORDINATION_TEAM_SYNC_KEYS = ("game_id", "period_id", "window_kind", "window_id", "axis")
COORDINATION_TEAM_SYNC_METRIC_COLUMNS = (
    "coord_team_sync_pearson_r",
    "coord_team_sync_cross_sampen",
    "coord_team_sync_pearson_r_surrogate_mean",
    "coord_team_sync_pearson_r_percentile",
    "coord_team_sync_pearson_r_excess",
    "coord_detected_share",
)
COORDINATION_TEAM_SYNC_COLUMNS = _schema(
    (
        *COORDINATION_TEAM_SYNC_KEYS,
        "team_a_id",
        "team_b_id",
        *COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
        "coord_team_sync_source",
        "coord_team_sync_sampen_source",  # A-20
        "coord_team_sync_surrogate_source",
        "coord_detection_source",
        "coord_stoppage_source",
    )
)

COORDINATION_RSI_KEYS = ("game_id", "period_id", "window_kind", "window_id", "axis")
COORDINATION_RSI_METRIC_COLUMNS = (
    "coord_rsi_mean_m",
    "coord_rsi_fraction_positive",
    "coord_rsi_switch_rate_per_min",
    "coord_rsi_bimodality_coefficient",
    "coord_detected_share",
)
COORDINATION_RSI_COLUMNS = _schema(
    (
        *COORDINATION_RSI_KEYS,
        "team_a_id",
        "team_b_id",
        *COORDINATION_RSI_METRIC_COLUMNS,
        "coord_rsi_source",
        "coord_detection_source",
        "coord_stoppage_source",
    )
)


# --------------------------------------------------------------------------- A-09: circular reliability registry
#: The circular-MEAN metric columns: a direction on the circle (degrees). Their D3 binding reliability is the
#: rotation-invariant circular statistic (``_kernels._circular.circular_reliability``), NEVER a plain linear ICC --
#: a linear ICC on a circular mean is origin-dependent (spec §8.5, ADR-111). The dispersion columns below are
#: MAGNITUDES (a spread in degrees) and stay linear despite the ``_deg`` suffix. Gated: ``test_circular_registry``
#: derives the degree-valued metric columns from the emitted schemas and asserts they partition EXACTLY into these
#: two tuples, so a new ``*_deg`` column cannot ship without being classified (the anti-rot meta-test, §9.4).
CIRCULAR_MEAN_COLUMNS: tuple[str, ...] = ("coord_rp_mean_deg", "coord_vc_mean_angle_deg", "coord_phi_mean_deg")
CIRCULAR_DISPERSION_COLUMNS: tuple[str, ...] = (
    "coord_rp_circ_sd_deg",
    "coord_vc_angle_variability_deg",
    "coord_phi_sd_deg",
)


def reliability_kind(column: str) -> str:
    """Return ``"circular"`` for a circular-mean metric column, else ``"linear"`` (spec §8.5 / ADR-111)."""
    return "circular" if column in CIRCULAR_MEAN_COLUMNS else "linear"


#: Coverage / count columns: denominators and sample diagnostics, NOT signals ("a coverage denominator must not
#: masquerade as a signal", AGENTS / ADR-042). Excluded from the per-construct reliability sweep (A-09) and flagged
#: as coverage in the glossary (A-38). Single-sourced here so both the sweep and the glossary read the same set.
COVERAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "coord_duration_s",
        "coord_n_samples",
        "coord_n_segments",
        "coord_coh_n_segments",
        "coord_observed_fraction",
        "coord_observed_fraction_a",
        "coord_observed_fraction_b",
        "coord_detected_share",
        "coord_n_players_mean",
        "coord_vc_n_stationary",
        "coord_on_pitch_s",
        # B-R3-01: the relative-phase valid-sample FRACTIONS are coverage denominators ("Reported, never used to
        # filter", spec §7; glossary "Coverage:"), not signals -- excluded from the reliability sweep like every
        # other observed-fraction column.
        "coord_rp_phase_valid_fraction_a",
        "coord_rp_phase_valid_fraction_b",
    }
)
