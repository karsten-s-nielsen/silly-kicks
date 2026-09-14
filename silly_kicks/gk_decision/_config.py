"""GkDecisionParams -- frozen params for the TF-62 GK-decision metric.

Mirrors ``TerritoryParams`` / ``RestDefenseParams``: a frozen dataclass with ``default`` /
``for_provider`` / ``is_default`` and an EMPTY per-provider override map until an ADR-009 calibration
apply-gate clears.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GkDecisionParams:
    """Parameters for the GK build-up decision-quality metric (spec §4, §12).

    Attributes
    ----------
    min_options: A decision is scored only if it offers at least this many (valued) options.
    value_fn: The option-value strategy key; PR1 ships only ``"completion_progression"``.
    reachability_min_xpass: The reachability floor (min xPass) for the reconstruction tiers -- a candidate
        is a genuine ALTERNATIVE only if its completion probability is at least this. The native tier never
        filters (its options are curated), so this is inert there. Default 0.85 ("reachable =
        high-completion"): the SB360 reachability sweep found the metric INVERTS at 0.5 (the bundled WC2022
        xPass runs generous on SB360 -- ~95% of reconstructed options >= 0.5, so a 0.5 floor cannot bite)
        and is responsive at ~0.85, so 0.5 is a useless default. See NOTICE / the construct-validity report.
    fov_radius_m: Reconstruction FOV-completeness disk radius (m) around the keeper (PR2; ADR-077).
    fov_min_observed_fraction: Reconstruction FOV floor -- a decision whose keeper neighbourhood is
        observed below this is dropped ``fov_cropped`` (PR2; ADR-077).
    receiver_exclusion_m: Reconstruction -- the nearest teammate within this distance of the pass end is
        the presumed receiver, excluded from the ALTERNATIVES to avoid double-counting (PR2).

    Examples
    --------
    >>> from silly_kicks.gk_decision import GkDecisionParams
    >>> GkDecisionParams().min_options
    3
    """

    min_options: int = 3
    value_fn: str = "completion_progression"
    reachability_min_xpass: float = 0.85  # reconstruction reachability floor: "reachable = high-completion"
    fov_radius_m: float = 10.0  # reconstruction FOV-completeness disk radius, m (PR2; ADR-077)
    fov_min_observed_fraction: float = 0.7  # reconstruction FOV-completeness floor (PR2; ADR-077)
    receiver_exclusion_m: float = 5.0  # reconstruction receiver-exclusion radius, m (PR2)
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> GkDecisionParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch (mirrors territory).

        >>> GkDecisionParams.default().is_default()
        True
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> GkDecisionParams:
        """Per-provider params; the base config for an unlisted provider (ADR-009).

        The override map ships EMPTY until a calibration apply-gate clears:

        >>> GkDecisionParams.for_provider("skillcorner") == GkDecisionParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        Examples
        --------
        >>> GkDecisionParams.default().is_default()
        True
        >>> GkDecisionParams().is_default()
        False
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_PARAMS: dict[str, dict] = {}
