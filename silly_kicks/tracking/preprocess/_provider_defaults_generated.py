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
        # sg_window_seconds: unverified: 10 / sampling_rate_hz (equals the placeholder; databricks_sql
        # fct_tracking_frames (recorded in probe_sources) sets it when its source value is present)
        sg_window_seconds=0.4,
        # sg_poly_order: placeholder: the _PLACEHOLDERS default
        sg_poly_order=3,
        # ema_alpha: placeholder: the _PLACEHOLDERS default
        ema_alpha=0.3,
        interpolation_method="linear",
        # max_gap_seconds: unverified: gap_length_p99_frames / sampling_rate_hz (the _PLACEHOLDERS default; equals the
        # placeholder; databricks_sql fct_tracking_frames (recorded in probe_sources) sets it when its source value is
        # present)
        max_gap_seconds=0.48,
        derive_velocity=True,
        # link_quality_high_threshold: placeholder: the domain default 0.85: the synthesized actions saturate link
        # quality (p10 >= 0.99)
        link_quality_high_threshold=0.85,
    ),
    "gradientsports": PreprocessConfig(
        smoothing_method="savgol",
        # sg_window_seconds: unverified: 10 / sampling_rate_hz (equals the placeholder;
        # gradientsports_local:10502.jsonl.bz2 (recorded in probe_sources) sets it when its source value is present)
        sg_window_seconds=0.333,
        # sg_poly_order: placeholder: the _PLACEHOLDERS default
        sg_poly_order=3,
        # ema_alpha: placeholder: the _PLACEHOLDERS default
        ema_alpha=0.3,
        interpolation_method="linear",
        # max_gap_seconds: unverified: gap_length_p99_frames / sampling_rate_hz (the _PLACEHOLDERS default; equals the
        # placeholder; gradientsports_local:10502.jsonl.bz2 (recorded in probe_sources) sets it when its source value
        # is present)
        max_gap_seconds=0.5,
        derive_velocity=True,
        # link_quality_high_threshold: placeholder: the domain default 0.85: the synthesized actions saturate link
        # quality (p10 >= 0.99)
        link_quality_high_threshold=0.85,
    ),
    "metrica": PreprocessConfig(
        smoothing_method="savgol",
        # sg_window_seconds: unverified: 10 / sampling_rate_hz (equals the placeholder; databricks_sql
        # fct_tracking_frames (recorded in probe_sources) sets it when its source value is present)
        sg_window_seconds=0.4,
        # sg_poly_order: placeholder: the _PLACEHOLDERS default
        sg_poly_order=3,
        # ema_alpha: placeholder: the _PLACEHOLDERS default
        ema_alpha=0.3,
        interpolation_method="linear",
        # max_gap_seconds: unverified: gap_length_p99_frames / sampling_rate_hz (the _PLACEHOLDERS default; equals the
        # placeholder; databricks_sql fct_tracking_frames (recorded in probe_sources) sets it when its source value is
        # present)
        max_gap_seconds=0.56,
        derive_velocity=True,
        # link_quality_high_threshold: placeholder: the domain default 0.85: the synthesized actions saturate link
        # quality (p10 >= 0.99)
        link_quality_high_threshold=0.85,
    ),
    "skillcorner": PreprocessConfig(
        smoothing_method="savgol",
        # sg_window_seconds: unverified: 10 / sampling_rate_hz (equals the placeholder; databricks_sql
        # fct_tracking_frames (recorded in probe_sources) sets it when its source value is present)
        sg_window_seconds=1.0,
        # sg_poly_order: placeholder: the _PLACEHOLDERS default
        sg_poly_order=3,
        # ema_alpha: placeholder: the _PLACEHOLDERS default
        ema_alpha=0.3,
        interpolation_method="linear",
        # max_gap_seconds: unverified: gap_length_p99_frames / sampling_rate_hz (the _PLACEHOLDERS default; equals the
        # placeholder; databricks_sql fct_tracking_frames (recorded in probe_sources) sets it when its source value is
        # present)
        max_gap_seconds=0.6,
        derive_velocity=True,
        # link_quality_high_threshold: placeholder: the domain default 0.85: the synthesized actions saturate link
        # quality (p10 >= 0.99)
        link_quality_high_threshold=0.85,
    ),
}
