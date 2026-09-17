"""MatchOutcomeParams -- frozen params for the TF-53 match-outcome metric.

Mirrors ``shot_stopping.ShotStoppingParams``: a frozen dataclass with ``.default`` / ``.for_provider``
/ ``.is_default`` and an EMPTY per-provider override map until an ADR-009 apply-gate clears. The two
knobs are the ORTHOGONAL honesty corrections (spec §4): ``same_possession`` (Rung 3b -- collapse
same-possession shots to one opportunity) and ``team_dependence`` (Rung 3a -- Dixon-Coles low-score
correlation). **Both default ON** (ADR-097): ``collapse`` on correctness (same-possession shots are not
independent Bernoulli trials -- a save->rebound->goal is ONE chance, so independent-shots double-counts)
and ``dixon_coles`` on measured evidence (the full-corpus calibration study found it improves the paired
per-match Brier on 77% of matches, Wilcoxon p~1e-124; both-corrections beats independent on 68.7%,
p~1e-71 -- ``docs/research/tf53_match_outcome_calibration/``). ``"independent"`` on either axis recovers
the naive exact Poisson-binomial baseline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

SamePossession = Literal["independent", "collapse"]
TeamDependence = Literal["independent", "dixon_coles"]

_SAME_POSSESSION: frozenset[str] = frozenset({"independent", "collapse"})
_TEAM_DEPENDENCE: frozenset[str] = frozenset({"independent", "dixon_coles"})


@dataclass(frozen=True)
class MatchOutcomeParams:
    """Parameters for the match-outcome metric.

    Attributes
    ----------
    same_possession :
        Rung-3b within-team correction. ``"collapse"`` (default, ADR-097) combines same-possession
        shots into ONE opportunity (``1 - prod(1 - xg)``) via ``spadl.add_possessions``, so a
        save->rebound->goal is one chance; ``"independent"`` treats every shot as its own Bernoulli.
    team_dependence :
        Rung-3a cross-team correction. ``"dixon_coles"`` (default, ADR-097) applies the fitted low-score
        tau reweighting (bundled rho artifact -- the default compute path loads it fail-closed);
        ``"independent"`` is the product of marginals.
    possession_max_gap_seconds :
        ``add_possessions`` gap used only when ``same_possession == "collapse"``.

    Examples
    --------
    >>> from silly_kicks.match_outcome import MatchOutcomeParams
    >>> MatchOutcomeParams.default().is_default()
    True
    >>> MatchOutcomeParams.for_provider("statsbomb") == MatchOutcomeParams()
    True
    """

    same_possession: SamePossession = "collapse"
    team_dependence: TeamDependence = "dixon_coles"
    possession_max_gap_seconds: float = 7.0
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.same_possession not in _SAME_POSSESSION:
            raise ValueError(f"same_possession must be one of {sorted(_SAME_POSSESSION)}; got {self.same_possession!r}")
        if self.team_dependence not in _TEAM_DEPENDENCE:
            raise ValueError(f"team_dependence must be one of {sorted(_TEAM_DEPENDENCE)}; got {self.team_dependence!r}")

    @classmethod
    def default(cls, *, force_universal: bool = False) -> MatchOutcomeParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch (mirrors shot_stopping).

        >>> MatchOutcomeParams.default().is_default()
        True
        >>> MatchOutcomeParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> MatchOutcomeParams:
        """Per-provider params; returns the base config for an unlisted provider (ADR-009).

        The override map ships EMPTY until a calibration apply-gate clears, so every provider
        currently resolves to the base config:

        >>> MatchOutcomeParams.for_provider("wyscout") == MatchOutcomeParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_MATCH_OUTCOME_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        >>> MatchOutcomeParams().is_default()
        False
        >>> MatchOutcomeParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_MATCH_OUTCOME_PARAMS: dict[str, dict] = {}
