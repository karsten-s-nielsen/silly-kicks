"""Frozen params for the TF-46 corner offence pipeline (Arm 1).

Mirrors ``shot_stopping.ShotStoppingParams`` / ``restdefense.RestDefenseParams``: each is a frozen
dataclass with ``.default`` / ``.for_provider`` / ``.is_default`` and an EMPTY per-provider override map
until an ADR-009 apply-gate clears. The knobs here are STRUCTURAL (window leads, encoder/cluster
settings), not calibratable tunes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CornerWindowParams:
    """Window geometry for corner identification + snapshot timing (spec §6.2).

    Attributes
    ----------
    init_lead_s: Seconds before the kick for the initial snapshot (Shaw: the speed-rise onset).
    target_contact_lead_s: Seconds after first contact for the target snapshot.
    target_cap_s: Hard cap after the kick for the target snapshot (the ``whichever occurs first`` bound).
    outcome_window_s: Seconds after the kick for the shot/goal outcome window (Shaw §5.2).
    short_corner_max_m: A next-same-team action within this distance marks a short corner.

    Examples
    --------
    >>> from silly_kicks.corners import CornerWindowParams
    >>> CornerWindowParams().target_cap_s
    2.0
    """

    init_lead_s: float = 2.0
    target_contact_lead_s: float = 1.0
    target_cap_s: float = 2.0
    outcome_window_s: float = 6.0
    short_corner_max_m: float = 15.0
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> CornerWindowParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch.

        >>> CornerWindowParams.default().is_default()
        True
        >>> CornerWindowParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> CornerWindowParams:
        """Per-provider params; the base config for an unlisted provider (ADR-009).

        Examples
        --------
        The override map ships EMPTY, so every provider resolves to the base config:

        >>> CornerWindowParams.for_provider("skillcorner") == CornerWindowParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_WINDOW_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """True iff built by :meth:`default` without ``force_universal=True``.

        >>> CornerWindowParams().is_default()
        False
        >>> CornerWindowParams.default().is_default()
        True
        """
        return self._is_universal_default


@dataclass(frozen=True)
class RunEncodingParams:
    """Settings for the probabilistic (initial x target)-zone run encoder.

    Attributes
    ----------
    min_run_distance_m: Players whose init->target move is shorter than this are inactive (ignored).

    Examples
    --------
    >>> from silly_kicks.corners import RunEncodingParams
    >>> RunEncodingParams().min_run_distance_m
    2.0
    """

    min_run_distance_m: float = 2.0
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> RunEncodingParams:
        """Universal-safe defaults.

        >>> RunEncodingParams.default().is_default()
        True
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> RunEncodingParams:
        """Per-provider params; base config for an unlisted provider.

        >>> RunEncodingParams.for_provider("sportec") == RunEncodingParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_ENCODING_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """True iff built by :meth:`default`.

        >>> RunEncodingParams.default().is_default()
        True
        """
        return self._is_universal_default


@dataclass(frozen=True)
class RoutineParams:
    """Settings for the agglomerative routine-clustering primitive.

    Attributes
    ----------
    linkage: scipy linkage method.
    n_clusters: Fixed cluster count (``None`` = cut by ``distance_threshold``).
    distance_threshold: Cophenetic cut distance when ``n_clusters`` is ``None``.

    Examples
    --------
    >>> from silly_kicks.corners import RoutineParams
    >>> RoutineParams().linkage
    'ward'
    """

    linkage: str = "ward"
    n_clusters: int | None = None
    distance_threshold: float = 1.0
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> RoutineParams:
        """Universal-safe defaults.

        >>> RoutineParams.default().is_default()
        True
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> RoutineParams:
        """Per-provider params; base config for an unlisted provider.

        >>> RoutineParams.for_provider("skillcorner") == RoutineParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_ROUTINE_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """True iff built by :meth:`default`.

        >>> RoutineParams.default().is_default()
        True
        """
        return self._is_universal_default


@dataclass(frozen=True)
class ChanceQualityParams:
    """Settings for the descriptive per-routine chance-quality primitive.

    Attributes
    ----------
    include_own_goals: Whether own goals count toward conceded/created goals (default False).

    Examples
    --------
    >>> from silly_kicks.corners import ChanceQualityParams
    >>> ChanceQualityParams().include_own_goals
    False
    """

    include_own_goals: bool = False
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> ChanceQualityParams:
        """Universal-safe defaults.

        >>> ChanceQualityParams.default().is_default()
        True
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> ChanceQualityParams:
        """Per-provider params; base config for an unlisted provider.

        >>> ChanceQualityParams.for_provider("statsbomb") == ChanceQualityParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_CHANCE_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """True iff built by :meth:`default`.

        >>> ChanceQualityParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_WINDOW_PARAMS: dict[str, dict] = {}
_PROVIDER_ENCODING_PARAMS: dict[str, dict] = {}
_PROVIDER_ROUTINE_PARAMS: dict[str, dict] = {}
_PROVIDER_CHANCE_PARAMS: dict[str, dict] = {}
