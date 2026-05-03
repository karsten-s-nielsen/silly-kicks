"""resolve_preprocess -- provider-aware auto-promotion + S5 fallback.

PR-S24 / lakehouse-review S5: when the provider is not in
get_provider_defaults() (e.g. kloppy 'tracab'), fall back to
PreprocessConfig.default(force_universal=True) + UserWarning rather than
crash with KeyError.
"""

from __future__ import annotations

import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig
from silly_kicks.tracking.preprocess._resolve import resolve_preprocess


def test_default_auto_promotes_to_supported_provider():
    cfg_in = PreprocessConfig.default()
    cfg_out = resolve_preprocess(cfg_in, provider="sportec")
    assert cfg_out.is_default() is False
    # Sportec for_provider has sg_window_seconds=0.4 (10 frames @ 25 Hz)
    assert cfg_out.sg_window_seconds == 0.4


def test_unsupported_provider_falls_back_with_warning():
    """S5: provider not in baselines -> UserWarning + universal fallback.

    N6 defensive note: ``has no per-provider PreprocessConfig`` is a literal
    substring of the warning text and is NOT regex-special; safe for
    pytest.warns(match=). For parametrised provider names, wrap in re.escape.
    """
    cfg_in = PreprocessConfig.default()
    with pytest.warns(UserWarning, match="has no per-provider PreprocessConfig"):
        cfg_out = resolve_preprocess(cfg_in, provider="tracab")
    assert cfg_out.is_default() is False  # force_universal=True path
    # Universal-safe field values still present
    assert cfg_out.sg_window_seconds == 0.4


def test_explicit_config_passes_through_unchanged():
    """Caller explicitly built config (non-default) -- pass through, no auto-promotion."""
    cfg_in = PreprocessConfig(
        smoothing_method="ema",
        sg_window_seconds=1.0,
        sg_poly_order=3,
        ema_alpha=0.5,
        interpolation_method="linear",
        max_gap_seconds=2.0,
        derive_velocity=True,
    )
    cfg_out = resolve_preprocess(cfg_in, provider="sportec")
    assert cfg_out is cfg_in
    assert cfg_out.smoothing_method == "ema"
    assert cfg_out.sg_window_seconds == 1.0


def test_force_universal_passes_through_unchanged():
    """force_universal=True opts out of auto-promotion deliberately."""
    cfg_in = PreprocessConfig.default(force_universal=True)
    cfg_out = resolve_preprocess(cfg_in, provider="sportec")
    assert cfg_out is cfg_in
    assert cfg_out.sg_window_seconds == 0.4
