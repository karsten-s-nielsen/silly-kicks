"""RestDefenseParams -- frozen params for the TF-60 rest-defense metrics (ADR-080).

Combines the ``CoverShadowParams.for_provider`` empty-override-map pattern (ADR-066: a base for
every unlisted provider) with the flag-based ``PreprocessConfig.is_default`` (a config built by
:meth:`default` is distinguishable from one hand-built with the same field values, so a future
provider-aware caller does not auto-promote a hand-built config). All calibratable defaults ship
un-tuned with an EMPTY per-provider override map until an ADR-009 apply-gate clears.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ._wfield import WFieldParams


@dataclass(frozen=True)
class RestDefenseParams:
    """Parameters for the rest-defense structure metrics.

    Attributes
    ----------
    n_rearguard:
        Back-line size for the rearguard-line geometry (TF-14 ``compute_defensive_line``); NOT
        the rest-defense unit size (the behind-the-ball unit is dynamic, typically ~5).
    min_ball_advance_m:
        Committed-forward gate: rest defense is only meaningful when the in-possession team is
        advanced. An action whose ball is closer than this to its own goal is dropped-and-counted.
        Default 52.5 m (past halfway). *(calibratable)*
    zone_depth_m:
        ``None`` => the danger zone is the full strip ``[rearguard line, own goal]``; a value caps
        the strip depth (metres from the own goal). *(calibratable)*
    danger_field_weight:
        Opt-in OBPV ``w_field`` re-weighting of the deep-zone threat (Layer 2, PR2). *(calibratable)*
    w_field_params:
        Shape parameters for the OBPV ``w_field`` (applied only when ``danger_field_weight``); a
        frozen :class:`WFieldParams` with un-tuned spec-time defaults (ADR-009). *(calibratable)*
    possession_stride:
        Sample every Nth in-possession action (cost control; 1 = every action).

    Examples
    --------
    >>> from silly_kicks.restdefense import RestDefenseParams
    >>> p = RestDefenseParams()
    >>> p.n_rearguard, p.min_ball_advance_m
    (4, 52.5)
    """

    n_rearguard: int = 4
    min_ball_advance_m: float = 52.5
    zone_depth_m: float | None = None
    danger_field_weight: bool = False
    w_field_params: WFieldParams = field(default_factory=WFieldParams)
    possession_stride: int = 1
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> RestDefenseParams:
        """Universal-safe defaults.

        Mirrors :meth:`PreprocessConfig.default`: ``force_universal=True`` is the escape hatch for
        the rare consumer that passes ``default()`` to a (future) provider-aware caller AND
        genuinely wants universal-safe values -- so its ``is_default()`` is ``False`` and the
        caller does not auto-promote it.

        >>> RestDefenseParams.default().is_default()
        True
        >>> RestDefenseParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> RestDefenseParams:
        """Per-provider params; returns the base config for an unlisted provider (ADR-066).

        Examples
        --------
        The override map ships EMPTY until a calibration apply-gate clears, so every provider
        currently resolves to the base config:

        >>> RestDefenseParams.for_provider("skillcorner") == RestDefenseParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_REST_DEFENSE_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        Examples
        --------
        A hand-built config is distinguishable from a factory one even with identical fields:

        >>> RestDefenseParams().is_default()
        False
        >>> RestDefenseParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-066-style calibration apply-gate clears (ADR-009: a per-provider tune is a
#: separate gated apply PR, never this cycle).
_PROVIDER_REST_DEFENSE_PARAMS: dict[str, dict] = {}
