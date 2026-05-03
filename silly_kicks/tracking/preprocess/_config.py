"""Public-API surface for the preprocess config.

Provides ``get_provider_defaults()`` -- the public-stable read of the codegen'd
provider-defaults table. PR-S24 N5 fix: callers must use this getter instead
of importing ``_PROVIDER_DEFAULTS`` directly from a private submodule.
"""

from __future__ import annotations

from ._config_dataclass import PreprocessConfig
from ._provider_defaults_generated import _PROVIDER_DEFAULTS


def get_provider_defaults() -> dict[str, PreprocessConfig]:
    """Return a shallow copy of the per-provider PreprocessConfig defaults.

    The returned dict is a new object on every call; mutating it does not
    affect the canonical generated table. The contained ``PreprocessConfig``
    instances are frozen so they can be safely shared without copying.

    Examples
    --------
    >>> from silly_kicks.tracking.preprocess import get_provider_defaults
    >>> defaults = get_provider_defaults()
    >>> "sportec" in defaults
    True
    """
    return dict(_PROVIDER_DEFAULTS)
