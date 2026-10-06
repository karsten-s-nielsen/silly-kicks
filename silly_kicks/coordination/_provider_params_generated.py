"""GENERATED -- coordination Tier-B base + per-provider parameter values (A3 single source).

Commit 1 renders the interim R4 table with ``BASE_SOURCE = "interim"``; Task 19's codegen reproduces this
file for empty inputs, and commit 2 regenerates it with ``BASE_SOURCE = "derivation"`` from D1's pooled
values. This module imports NOTHING from the package (no import cycle) -- the key lists are literal, and
``tests/coordination/test_config.py`` asserts they match the canonical vocabularies.

Do not hand-edit: regenerate via ``scripts/derive_coordination_params.py`` (Task 20)."""

from __future__ import annotations

BASE_SOURCE: str = "interim"

BASE_COORDINATION_PARAMS: dict[str, object] = {
    "band_high_cpm": 0.83,
    "band_low_cpm": 0.22,
    "butterworth_cutoff_hz": 0.4,
    "max_detection_gap_s": 0.5,
    "min_observed_fraction": {
        "cluster": 0.5,
        "coherence": 0.5,
        "cross_correlation": 0.5,
        "relative_phase": 0.5,
        "rsi": 0.5,
        "spectral": 0.5,
        "team_sync": 0.5,
        "vector_coding": 0.5,
    },
    "min_shift_s": {
        "back_line_high_x": 60.0,
        "centroid_x": 60.0,
        "centroid_y": 60.0,
        "cluster_amplitude": 60.0,
        "compactness_x": 60.0,
        "convex_hull_area": 60.0,
        "defensive_line_x": 60.0,
        "player_x": 60.0,
        "player_y": 60.0,
        "spread": 60.0,
        "stretch_index": 60.0,
        "stretch_x": 60.0,
        "stretch_y": 60.0,
        "team_length": 60.0,
        "team_width": 60.0,
    },
    "possession_gap_s": 2.0,
    "vc_epsilon": {
        "back_line_high_x": 0.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "compactness_x": 0.0,
        "convex_hull_area": 0.0,
        "defensive_line_x": 0.0,
        "spread": 0.0,
        "stretch_index": 0.0,
        "stretch_x": 0.0,
        "stretch_y": 0.0,
        "team_length": 0.0,
        "team_width": 0.0,
    },
    "welch_segment_s": 400.0,
}

PROVIDER_COORDINATION_PARAMS: dict[str, dict[str, object]] = {}
