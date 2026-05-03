"""PreprocessConfig dataclass.

Lives in its own submodule (separate from ``_config.py``) to avoid a circular
import: the auto-generated ``_provider_defaults_generated.py`` imports
``PreprocessConfig`` from here; ``_config.py`` then orchestrates and re-exports
both. Without the split, ``_config.py`` -> ``_provider_defaults_generated.py`` ->
``_config.py`` would deadlock at import time.

PR-S24 / spec section 4.3 + lakehouse review S1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SmoothingMethod = Literal["savgol", "ema", None]
# Cubic interpolation is intentionally NOT in the API surface for PR-S24.
# Lakehouse review N3: the implementation only does linear math; "cubic" was
# accepted in v1 but produced linear output. Restrict the Literal to linear-only
# until a concrete consumer asks for cubic -- at that point we ship cubic via
# scipy.interpolate.CubicSpline as TF-9-cubic.
InterpolationMethod = Literal["linear", None]


@dataclass(frozen=True)
class PreprocessConfig:
    """Shared preprocessing config dataclass.

    PR-S24 lakehouse review C1: rejects ``derive_velocity=True`` +
    ``smoothing_method=None`` at construction (impossible combination --
    velocity derivation reads ``x_smoothed``/``y_smoothed`` produced by
    ``smooth_frames``).

    Examples
    --------
    >>> from silly_kicks.tracking.preprocess import PreprocessConfig
    >>> cfg = PreprocessConfig.default()
    >>> cfg.is_default()
    True
    """

    smoothing_method: SmoothingMethod = "savgol"
    sg_window_seconds: float = 0.4
    sg_poly_order: int = 3
    ema_alpha: float = 0.3
    interpolation_method: InterpolationMethod = "linear"
    max_gap_seconds: float = 0.5
    derive_velocity: bool = True
    link_quality_high_threshold: float = 0.85
    # Provenance flag -- set by default() factory only. Excluded from __eq__/__hash__/repr
    # so two configs with the same field values are still equal regardless of which
    # factory built them. Read via is_default() (flag-based, NOT value-equality --
    # see feedback_default_config_auto_promotion).
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Reject impossible combinations at construction time (lakehouse review C1).

        ``derive_velocity=True`` requires smoothed positions (``smoothing_method`` not None);
        ``derive_velocities()`` raises if ``x_smoothed``/``y_smoothed`` are absent.
        Catching the inconsistency here means the converter / direct callers never run a
        partial pipeline that crashes mid-flight.
        """
        if self.derive_velocity and self.smoothing_method is None:
            raise ValueError(
                "PreprocessConfig: derive_velocity=True requires smoothing_method != None -- "
                "velocity derivation reads x_smoothed/y_smoothed (which are produced by "
                "smooth_frames). Either set smoothing_method='savgol'/'ema' or "
                "derive_velocity=False."
            )

    @classmethod
    def default(cls, *, force_universal: bool = False) -> PreprocessConfig:
        """Universal-safe defaults. Per-provider tuning via :meth:`for_provider`.

        ``force_universal=True`` is an escape hatch for the rare consumer that
        passes ``default()`` to a provider-aware caller AND genuinely wants
        universal-safe values (debugging cross-provider comparisons under
        fixed config). When True, the returned config's ``is_default()``
        returns False so the provider-aware caller does NOT auto-promote.

        Examples
        --------
        >>> from silly_kicks.tracking.preprocess import PreprocessConfig
        >>> PreprocessConfig.default().is_default()
        True
        >>> PreprocessConfig.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> PreprocessConfig:
        """Provider-tuned defaults from preprocess_baseline.json.

        Raises ``KeyError`` for unsupported providers -- caller should pre-check
        via ``provider in get_provider_defaults()`` or use
        ``PreprocessConfig.default(force_universal=True)`` as a fallback.

        Examples
        --------
        >>> from silly_kicks.tracking.preprocess import PreprocessConfig
        >>> cfg = PreprocessConfig.for_provider("sportec")
        >>> cfg.sg_window_seconds  # 10 frames @ 25 Hz
        0.4
        """
        # Local import to break the cycle (see module docstring)
        from ._config import get_provider_defaults

        return get_provider_defaults()[provider]

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        Examples
        --------
        >>> from silly_kicks.tracking.preprocess import PreprocessConfig
        >>> PreprocessConfig.default().is_default()
        True
        """
        return self._is_universal_default
