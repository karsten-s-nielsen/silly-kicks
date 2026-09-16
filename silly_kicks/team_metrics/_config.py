"""TeamKpiParams + CounterpressWindow -- frozen params for the TF-52 team-KPI module.

Mirrors ``shot_stopping.ShotStoppingParams``: a frozen dataclass with ``.default`` / ``.for_provider``
/ ``.is_default`` and an EMPTY per-provider override map until an ADR-009 apply-gate clears. Geometry
defaults come from ``spadlconfig`` (ADR-050); the counter-press window is a seconds-XOR-passes value
object (spec Section 4.1).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from silly_kicks.spadl import config as spadlconfig


@dataclass(frozen=True)
class CounterpressWindow:
    """A counter-press regain window: exactly ONE of ``seconds=`` or ``passes=`` (spec Section 4.1 XOR).

    Examples
    --------
    >>> from silly_kicks.team_metrics import CounterpressWindow
    >>> CounterpressWindow(seconds=6.0).seconds
    6.0
    >>> CounterpressWindow(passes=3).passes
    3
    """

    seconds: float | None = None
    passes: int | None = None

    def __post_init__(self) -> None:
        if (self.seconds is None) == (self.passes is None):
            raise ValueError("CounterpressWindow requires exactly one of seconds= or passes=")


@dataclass(frozen=True)
class TeamKpiParams:
    """Frozen params for the team-KPI module.

    Note
    ----
    ``counterpress_seconds`` (v1 recoveries-within-Ns, a FIXED window) and ``counterpress_window``
    (v2 configurable seconds-XOR-passes) are DISTINCT by design and both default to 5 s -- the v1
    Twelve-glossary metric and the v2 practitioner window are separate KPIs, not a duplicate.

    Examples
    --------
    >>> from silly_kicks.team_metrics import TeamKpiParams
    >>> TeamKpiParams.default().is_default()
    True
    >>> TeamKpiParams.for_provider("statsbomb") == TeamKpiParams()
    True
    """

    possession_max_gap_seconds: float = 7.0  # -> add_possessions(max_gap_seconds=)
    possession_retain_on_set_pieces: bool = True  # -> add_possessions(retain_on_set_pieces=)
    defensive_action_types: tuple[str, ...] = ("tackle", "interception", "foul")
    ppda_zone_fraction: float = 0.6
    long_ball_distance_m: float = 32.0
    counterpress_seconds: float = 5.0  # v1 recoveries-within-Ns window (fixed)
    counterpress_window: CounterpressWindow = CounterpressWindow(seconds=5.0)  # v2 configurable window
    post_recovery_window_seconds: float = 10.0
    retained_after_seconds: float = 5.0
    high_opportunity_xg: float = 0.15
    switch_min_lateral_m: float = 30.0
    # spec Section 5 geometry-derived fields (from spadlconfig at import time; immutable):
    channel_boundaries: tuple[float, float] = (
        spadlconfig.field_width / 3.0,
        2.0 * spadlconfig.field_width / 3.0,
    )
    build_up_zone_max_x: float = spadlconfig.field_length / 3.0
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> TeamKpiParams:
        """Universal-safe defaults; ``force_universal=True`` is the escape hatch (mirrors shot_stopping).

        >>> TeamKpiParams.default().is_default()
        True
        >>> TeamKpiParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> TeamKpiParams:
        """Per-provider params; returns the base config for an unlisted provider (ADR-009).

        The override map ships EMPTY until a calibration apply-gate clears, so every provider
        currently resolves to the base config:

        >>> TeamKpiParams.for_provider("wyscout") == TeamKpiParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_TEAM_KPI_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """Flag-based: True iff built by :meth:`default` without ``force_universal=True``.

        >>> TeamKpiParams().is_default()
        False
        >>> TeamKpiParams.default().is_default()
        True
        """
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_TEAM_KPI_PARAMS: dict[str, dict] = {}

#: Documented named counter-press windows (seconds XOR passes). Constant, not a mutable param.
#: Plug into ``TeamKpiParams(counterpress_window=COUNTERPRESS_PRESETS["tigres_hunt"])``.
COUNTERPRESS_PRESETS: dict[str, CounterpressWindow] = {
    "barcelona": CounterpressWindow(seconds=6.0),
    "coventry": CounterpressWindow(seconds=5.0),
    "leipzig": CounterpressWindow(seconds=10.0),
    "hammarby": CounterpressWindow(seconds=5.0),
    "tigres_hunt": CounterpressWindow(passes=3),
}
