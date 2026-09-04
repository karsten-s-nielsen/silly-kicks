"""TerritoryParams -- frozen params for the TF-54 territorial-dominance metric.

Mirrors ``restdefense.RestDefenseParams`` / ``ShotStoppingParams``: a frozen dataclass with
``.default`` / ``.for_provider`` / ``.is_default`` and an EMPTY per-provider override map until an
ADR-009 calibration apply-gate clears.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TerritoryParams:
    """Parameters for the territorial-dominance metric (spec §5.2, §5.4).

    Attributes
    ----------
    trim_fraction: Fraction of a player's own-half defensive-action locations nearest their centroid
        kept for the hull (a trimmed convex hull -- robust to the odd deep recovery).
    forward_threshold_m: A pass into the hull is "forward" iff its destination x exceeds its origin x
        by more than this (metres, opponent frame).
    defensive_action_types: The SPADL action types whose locations build the hull.
    own_half_max_x: A defensive action counts toward the hull only if its ``start_x`` is below this
        (the defender's own half, action-LTR frame).

    Examples
    --------
    >>> from silly_kicks.territory import TerritoryParams
    >>> TerritoryParams().trim_fraction
    0.7
    """

    trim_fraction: float = 0.70
    forward_threshold_m: float = 0.0
    defensive_action_types: tuple[str, ...] = ("tackle", "interception", "clearance")
    own_half_max_x: float = 52.5
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> TerritoryParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch (mirrors restdefense).

        >>> TerritoryParams.default().is_default()
        True
        >>> TerritoryParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> TerritoryParams:
        """Per-provider params; returns the base config for an unlisted provider (ADR-009).

        Examples
        --------
        The override map ships EMPTY until a calibration apply-gate clears, so every provider
        currently resolves to the base config:

        >>> TerritoryParams.for_provider("statsbomb") == TerritoryParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_TERRITORY_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        Examples
        --------
        A hand-built config is distinguishable from a factory one even with identical fields:

        >>> TerritoryParams().is_default()
        False
        >>> TerritoryParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_TERRITORY_PARAMS: dict[str, dict] = {}
