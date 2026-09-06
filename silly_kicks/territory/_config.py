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


@dataclass(frozen=True)
class CounterfactualParams:
    """Parameters for the reserved ``method="counterfactual"`` territorial valuation (SPEC-04 §5.4).

    Attributes
    ----------
    direction_cone_degrees: Half-angle (degrees) of the directional cone, centred on a failed pass's
        death direction from its origin, used to identify the plausible "aimed at" target zones when
        modeling its intended destination (spec §5.2 -- the target distribution ``q`` is renormalized
        transition mass restricted to this cone).
    min_transition_support: Minimum modeled-transition probability mass (``Sigma T`` over the cone) below
        which a failed pass's target is treated as unresolvable (``territory_target_source="unresolved"``,
        dropped-and-counted -- never a fabricated 0).

    ``target_zone_grid`` is deliberately NOT a field here -- it defaults to the injected ``xt.grid`` at
    compute time (the method has no xT of its own; silly-kicks ships no xG/xT model, ADR-022 port pattern).

    Examples
    --------
    >>> from silly_kicks.territory import CounterfactualParams
    >>> CounterfactualParams().direction_cone_degrees
    45.0
    """

    direction_cone_degrees: float = 45.0
    min_transition_support: float = 1e-6
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> CounterfactualParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch (mirrors TerritoryParams).

        >>> CounterfactualParams.default().is_default()
        True
        >>> CounterfactualParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> CounterfactualParams:
        """Per-provider params; returns the base config for an unlisted provider (ADR-009).

        Examples
        --------
        The override map ships EMPTY until a calibration apply-gate clears, so every provider
        currently resolves to the base config:

        >>> CounterfactualParams.for_provider("statsbomb") == CounterfactualParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_COUNTERFACTUAL_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        Examples
        --------
        >>> CounterfactualParams().is_default()
        False
        >>> CounterfactualParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_COUNTERFACTUAL_PARAMS: dict[str, dict] = {}
