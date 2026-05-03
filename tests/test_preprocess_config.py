"""PreprocessConfig dataclass -- flag-based is_default(); per-provider defaults.

PR-S24 / lakehouse-review C1: __post_init__ rejects derive_velocity=True +
smoothing_method=None at construction.
PR-S24 / lakehouse-review N5: get_provider_defaults() public getter (no private
submodule import).
"""

from __future__ import annotations

import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig, get_provider_defaults


def test_default_factory_marks_universal():
    cfg = PreprocessConfig.default()
    assert cfg.is_default() is True


def test_force_universal_does_not_mark_default():
    cfg = PreprocessConfig.default(force_universal=True)
    assert cfg.is_default() is False


def test_for_provider_does_not_mark_default():
    cfg = PreprocessConfig.for_provider("sportec")
    assert cfg.is_default() is False


def test_hand_constructed_with_default_values_does_not_mark_default():
    """Flag-based, not value-equality (PR-S24 N5)."""
    cfg = PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
        derive_velocity=True,
    )
    assert cfg.is_default() is False


def test_eq_excludes_provenance_flag():
    """Two configs with identical fields are equal regardless of factory provenance."""
    a = PreprocessConfig.default()
    b = PreprocessConfig(
        smoothing_method="savgol",
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
        derive_velocity=True,
    )
    assert a == b


def test_unknown_provider_raises():
    with pytest.raises(KeyError):
        PreprocessConfig.for_provider("statsbomb")  # tracking-side unsupported


def test_derive_velocity_without_smoothing_rejected_at_construction():
    """C1: catch the impossible combination at construction time, not mid-pipeline."""
    with pytest.raises(ValueError, match="derive_velocity=True requires smoothing_method"):
        PreprocessConfig(
            smoothing_method=None,
            derive_velocity=True,
            sg_window_seconds=0.4,
            sg_poly_order=3,
            ema_alpha=0.3,
            interpolation_method="linear",
            max_gap_seconds=0.5,
        )


def test_derive_velocity_false_without_smoothing_is_fine():
    """Caller wants raw positions only -- no smoothing, no velocity. Allowed."""
    cfg = PreprocessConfig(
        smoothing_method=None,
        derive_velocity=False,
        sg_window_seconds=0.4,
        sg_poly_order=3,
        ema_alpha=0.3,
        interpolation_method="linear",
        max_gap_seconds=0.5,
    )
    assert cfg.smoothing_method is None
    assert cfg.derive_velocity is False


def test_get_provider_defaults_returns_all_four():
    defaults = get_provider_defaults()
    assert set(defaults.keys()) == {"sportec", "pff", "metrica", "skillcorner"}
    for name, cfg in defaults.items():
        assert isinstance(cfg, PreprocessConfig), name


def test_get_provider_defaults_returns_independent_copy():
    a = get_provider_defaults()
    b = get_provider_defaults()
    assert a is not b
    assert a == b
