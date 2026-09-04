"""DuelRatingParams + GlickoState -- frozen config for the TF-55 Glicko-2 duel-rating metric.

Mirrors the ``restdefense.RestDefenseParams`` idiom: a frozen dataclass with ``.default`` /
``.for_provider`` / ``.is_default`` and an EMPTY per-provider override map until an ADR-009 apply-gate.
Constants are Glickman's Glicko-2 defaults.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlickoState:
    """One player's Glicko-2 state: rating (Glicko scale), rating deviation, and volatility.

    Examples
    --------
    >>> from silly_kicks.duels import GlickoState
    >>> s = GlickoState(rating=1500.0, rd=350.0, volatility=0.06)
    >>> (s.rating, s.rd, s.volatility)
    (1500.0, 350.0, 0.06)
    """

    rating: float
    rd: float
    volatility: float


@dataclass(frozen=True)
class DuelRatingParams:
    """Parameters for the Glicko-2 duel-rating metric (spec §5b.3).

    Attributes
    ----------
    initial_rating / initial_rd / initial_volatility: The seed Glicko-2 state for an unseen player.
    tau: The system constant constraining volatility change over one rating period (Glickman: 0.3-1.2).
    apply_inactivity_rd_growth: Widen a player's RD in a rating period they contest no duel (gold-standard
        Glicko-2 -- uncertainty grows with inactivity).

    Examples
    --------
    >>> from silly_kicks.duels import DuelRatingParams
    >>> DuelRatingParams().initial_rating
    1500.0
    """

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_volatility: float = 0.06
    tau: float = 0.5
    apply_inactivity_rd_growth: bool = True
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    def initial_state(self) -> GlickoState:
        """The seed state for a player never seen before.

        >>> DuelRatingParams().initial_state()
        GlickoState(rating=1500.0, rd=350.0, volatility=0.06)
        """
        return GlickoState(self.initial_rating, self.initial_rd, self.initial_volatility)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> DuelRatingParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch (mirrors restdefense).

        >>> DuelRatingParams.default().is_default()
        True
        >>> DuelRatingParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> DuelRatingParams:
        """Per-provider params; returns the base config for an unlisted provider (ADR-009).

        Examples
        --------
        The override map ships EMPTY until a calibration apply-gate clears, so every provider
        currently resolves to the base config:

        >>> DuelRatingParams.for_provider("sportec") == DuelRatingParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_DUEL_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        Examples
        --------
        A hand-built config is distinguishable from a factory one even with identical fields:

        >>> DuelRatingParams().is_default()
        False
        >>> DuelRatingParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_DUEL_PARAMS: dict[str, dict] = {}
