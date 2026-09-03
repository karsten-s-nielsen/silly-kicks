"""ShotStoppingParams -- frozen params for the TF-59 GK shot-stopping metric (PR2).

Mirrors ``restdefense.RestDefenseParams``: a frozen dataclass with ``.default`` / ``.for_provider`` /
``.is_default`` and an EMPTY per-provider override map until an ADR-009 apply-gate clears. The metric
has no *calibratable* parameter (GP/GSAA is deterministic over an injected PSxG); the sole structural
knob is which period id is the penalty shootout (excluded).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShotStoppingParams:
    """Parameters for the shot-stopping metric.

    Attributes
    ----------
    shootout_period_id: The period id treated as the penalty shootout and EXCLUDED entirely (spec §6.2).

    Examples
    --------
    >>> from silly_kicks.shot_stopping import ShotStoppingParams
    >>> ShotStoppingParams().shootout_period_id
    5
    """

    shootout_period_id: int = 5
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> ShotStoppingParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch (mirrors restdefense).

        >>> ShotStoppingParams.default().is_default()
        True
        >>> ShotStoppingParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> ShotStoppingParams:
        """Per-provider params; returns the base config for an unlisted provider (ADR-009).

        Examples
        --------
        The override map ships EMPTY until a calibration apply-gate clears, so every provider
        currently resolves to the base config:

        >>> ShotStoppingParams.for_provider("statsbomb") == ShotStoppingParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_SHOT_STOPPING_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        Examples
        --------
        A hand-built config is distinguishable from a factory one even with identical fields:

        >>> ShotStoppingParams().is_default()
        False
        >>> ShotStoppingParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_SHOT_STOPPING_PARAMS: dict[str, dict] = {}
