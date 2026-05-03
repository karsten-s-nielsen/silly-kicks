"""Provider-aware auto-promotion of PreprocessConfig.default() with safe fallback.

Lakehouse-review S5: when ``provider`` is not in ``get_provider_defaults()``
(e.g., kloppy returns a provider like "tracab" or "statsperform" not yet
profiled), fall back to ``PreprocessConfig.default(force_universal=True)``
rather than raising KeyError. Emits a UserWarning so the operator knows
universal-safe values are in play.

Memory: feedback_provider_aware_config_fallback.
"""

from __future__ import annotations

import warnings

from ._config import get_provider_defaults
from ._config_dataclass import PreprocessConfig


def resolve_preprocess(cfg: PreprocessConfig, *, provider: str) -> PreprocessConfig:
    """Auto-promote ``PreprocessConfig.default()`` to ``for_provider(provider)``.

    If ``cfg.is_default()`` returns False (i.e., the caller built the config
    explicitly or passed ``default(force_universal=True)``), pass through
    unchanged. If True and ``provider`` is unsupported, fall back to
    ``PreprocessConfig.default(force_universal=True)`` + ``UserWarning``.

    Examples
    --------
    >>> from silly_kicks.tracking.preprocess import PreprocessConfig
    >>> from silly_kicks.tracking.preprocess._resolve import resolve_preprocess
    >>> cfg = resolve_preprocess(PreprocessConfig.default(), provider="sportec")
    >>> cfg.is_default()
    False
    """
    if not cfg.is_default():
        return cfg
    defaults = get_provider_defaults()
    if provider in defaults:
        return defaults[provider]
    warnings.warn(
        f"resolve_preprocess: provider {provider!r} has no per-provider PreprocessConfig "
        "in tests/fixtures/baselines/preprocess_baseline.json -- falling back to "
        "PreprocessConfig.default(force_universal=True). To suppress this warning, "
        "either add a provider block to the baseline JSON + regen, or pass "
        "PreprocessConfig.default(force_universal=True) / a hand-built config explicitly.",
        UserWarning,
        stacklevel=3,
    )
    return PreprocessConfig.default(force_universal=True)
