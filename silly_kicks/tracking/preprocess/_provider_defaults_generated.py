"""AUTO-GENERATED -- DO NOT EDIT.

Source: tests/fixtures/baselines/preprocess_baseline.json
Regen:  uv run python scripts/regenerate_provider_defaults.py

PR-S24 lakehouse review S1 fix -- codegen pipeline replaces manual sync hand-edit.
"""

from __future__ import annotations

from ._config_dataclass import PreprocessConfig

_PROVIDER_DEFAULTS: dict[str, PreprocessConfig] = {
    "sportec": PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.48,
        derive_velocity=True,
        link_quality_high_threshold=0.85,
    ),
    "pff": PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=0.333,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
        derive_velocity=True,
        link_quality_high_threshold=0.85,
    ),
    "metrica": PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.56,
        derive_velocity=True,
        link_quality_high_threshold=0.85,
    ),
    "skillcorner": PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=1.0,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.6,
        derive_velocity=True,
        link_quality_high_threshold=0.85,
    ),
}
