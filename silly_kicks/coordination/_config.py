"""CoordinationParams -- frozen parameters for the TF-58 coordination metrics.

Tier-A and convention defaults are literal here; Tier-B base defaults are SINGLE-SOURCED from the generated
``_provider_params_generated.BASE_COORDINATION_PARAMS`` (amendment A3: no interim placeholder survives into a
release -- Task 28 gates ``BASE_SOURCE``). Map fields freeze to ``MappingProxyType`` and their key sets are
asserted exactly. Follows the ``RestDefenseParams`` default/for_provider/is_default idiom (ADR-066).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from silly_kicks.coordination import _provider_params_generated
from silly_kicks.coordination._columns import COORD_METHOD_FAMILIES, TEAM_SIGNALS

_BASE = _provider_params_generated.BASE_COORDINATION_PARAMS
_MIN_OBSERVED_KEYS = frozenset(COORD_METHOD_FAMILIES)
_VC_EPSILON_KEYS = frozenset(TEAM_SIGNALS)
_MIN_SHIFT_KEYS = frozenset(TEAM_SIGNALS) | {"player_x", "player_y", "cluster_amplitude"}


def _base_map(name: str) -> dict[str, float]:
    """A fresh mutable copy of a generated base map (frozen later by ``__post_init__``)."""
    return dict(_BASE[name])  # type: ignore[arg-type]


def _default_include_goalkeeper() -> dict[str, bool]:
    # team signals: outfield (Moura); dyad: outfield (Folgado); cluster: 11 incl. GK (Duarte).
    return {"team_signals": False, "dyad": False, "cluster": True}


def _merge_override(base: CoordinationParams, override: Mapping[str, object]) -> CoordinationParams:
    """``base`` with a per-provider ``override`` merged KEY-WISE (a map value updates the base map, a scalar replaces).

    The ONE merge rule: :meth:`CoordinationParams.for_provider` applies it to the generated module's override, and the
    coordination drivers' artifact handoff (``scripts/_coordination_params_codegen.params_from_artifacts``) applies it
    to the same override read from ``derivation.json`` -- so both give identical params (ADR-111, M-5).
    """
    if not override:
        return base
    kwargs: dict[str, object] = {}
    for key, value in override.items():
        current = getattr(base, key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged = dict(current)
            merged.update(value)
            kwargs[key] = merged
        else:
            kwargs[key] = value
    return dataclasses.replace(base, **kwargs)


@dataclass(frozen=True)
class CoordinationParams:
    """Parameters for the coordination-dynamics metrics.

    Examples
    --------
    >>> from silly_kicks.coordination._config import CoordinationParams
    >>> p = CoordinationParams()
    >>> p.xcorr_max_lag_s, p.n_phases, p.min_players
    (15.0, 3, 6)
    >>> p.butterworth_cutoff_hz  # single-sourced from the generated interim base
    0.4
    """

    # Tier A -- primary sources, never tuned
    xcorr_max_lag_s: float = 15.0
    # what the POSSESSION window builders stamp (Moura 2016's possession thirds); each window's own n_phases decides
    # its subdivision, and NA (period/sliding windows) means none (spec 7.6, D3; owner ruling 2026-10-04, A-22)
    n_phases: int = 3
    near_in_phase_deg: float = 30.0
    max_stoppage_s: float = 25.0
    sampen_m: int = 1
    sampen_r_sd: float = 0.2
    butterworth_order: int = 3
    analysis_hz: float = 10.0
    # Conventions -- fixed, with rationale
    min_players: int = 6
    n_surrogates: int = 199
    iaaft_max_iter: int = 100
    coverage_warn_fraction: float = 0.25
    # Tier B -- base values single-sourced from the generated file (A3); no literal lives here.
    butterworth_cutoff_hz: float = _BASE["butterworth_cutoff_hz"]  # type: ignore[assignment]
    max_detection_gap_s: float = _BASE["max_detection_gap_s"]  # type: ignore[assignment]
    min_observed_fraction: Mapping[str, float] = field(default_factory=lambda: _base_map("min_observed_fraction"))
    vc_epsilon: Mapping[str, float] = field(default_factory=lambda: _base_map("vc_epsilon"))
    min_shift_s: Mapping[str, float] = field(default_factory=lambda: _base_map("min_shift_s"))
    band_low_cpm: float = _BASE["band_low_cpm"]  # type: ignore[assignment]
    band_high_cpm: float = _BASE["band_high_cpm"]  # type: ignore[assignment]
    welch_segment_s: float = _BASE["welch_segment_s"]  # type: ignore[assignment]
    possession_gap_s: float = _BASE["possession_gap_s"]  # type: ignore[assignment]
    # Other
    surrogate_method: Literal["time_shift", "iaaft"] = "time_shift"
    surrogate_seed: int = 0
    include_goalkeeper: Mapping[str, bool] = field(default_factory=_default_include_goalkeeper)
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._reject_scalars()
        self._freeze_and_check_maps()

    def _reject_scalars(self) -> None:
        checks: list[tuple[bool, str]] = [
            (self.n_phases < 2, "n_phases must be >= 2 (one phase only duplicates the window; A-22)"),
            (self.xcorr_max_lag_s <= 0, "xcorr_max_lag_s must be > 0"),
            (not (0 < self.near_in_phase_deg < 180), "near_in_phase_deg must be in (0, 180)"),
            (self.max_stoppage_s <= 0, "max_stoppage_s must be > 0"),
            (self.sampen_m < 1, "sampen_m must be >= 1"),
            (self.sampen_r_sd <= 0, "sampen_r_sd must be > 0"),
            (self.butterworth_order < 1, "butterworth_order must be >= 1"),
            (self.analysis_hz <= 0, "analysis_hz must be > 0"),
            (self.butterworth_cutoff_hz <= 0, "butterworth_cutoff_hz must be > 0"),
            (self.min_players < 2, "min_players must be >= 2"),
            (self.n_surrogates < 0, "n_surrogates must be >= 0"),
            (
                self.surrogate_method not in ("time_shift", "iaaft"),
                f"surrogate_method must be 'time_shift' or 'iaaft'; got {self.surrogate_method!r}",
            ),
            (self.iaaft_max_iter < 1, "iaaft_max_iter must be >= 1"),
            (not (0 <= self.coverage_warn_fraction <= 1), "coverage_warn_fraction must be in [0, 1]"),
            (self.max_detection_gap_s < 0, "max_detection_gap_s must be >= 0"),
            (
                self.max_detection_gap_s >= self.max_stoppage_s,
                # A-28: a long stoppage ends a player run only because the dead samples leave a gap longer than
                # max_detection_gap_s; if the gap tolerance reached the stoppage threshold the run would bridge a
                # stoppage the team segments split on.
                f"max_detection_gap_s ({self.max_detection_gap_s}) must be < max_stoppage_s ({self.max_stoppage_s}) "
                f"so a long stoppage splits a player run",
            ),
            (not (0 < self.band_low_cpm < self.band_high_cpm), "require 0 < band_low_cpm < band_high_cpm"),
            (self.welch_segment_s <= 0, "welch_segment_s must be > 0"),
            (self.possession_gap_s < 0, "possession_gap_s must be >= 0"),
            (self.surrogate_seed < 0, "surrogate_seed must be >= 0"),
        ]
        for bad, message in checks:
            if bad:
                raise ValueError(message)

    def _freeze_and_check_maps(self) -> None:
        specs = (
            ("min_observed_fraction", _MIN_OBSERVED_KEYS, lambda v: not (0 <= v <= 1), "must be in [0, 1]"),
            ("vc_epsilon", _VC_EPSILON_KEYS, lambda v: v < 0, "must be >= 0"),
            ("min_shift_s", _MIN_SHIFT_KEYS, lambda v: v <= 0, "must be > 0"),
        )
        for name, keys, is_bad, why in specs:
            m = dict(getattr(self, name))
            if set(m) != set(keys):
                raise ValueError(f"{name} keys must be exactly {sorted(keys)}; got {sorted(m)}")
            for k, v in m.items():
                if is_bad(v):
                    raise ValueError(f"{name}[{k!r}] {why}; got {v}")
            object.__setattr__(self, name, MappingProxyType(m))
        gk = dict(self.include_goalkeeper)
        if set(gk) != {"team_signals", "dyad", "cluster"}:
            raise ValueError(
                f"include_goalkeeper keys must be exactly ['cluster', 'dyad', 'team_signals']; got {sorted(gk)}"
            )
        object.__setattr__(self, "include_goalkeeper", MappingProxyType(gk))

    def __hash__(self) -> int:
        def canon(value: object) -> object:
            if isinstance(value, Mapping):
                return tuple(sorted((k, canon(v)) for k, v in value.items()))
            return value

        fields = (f.name for f in dataclasses.fields(self) if f.name != "_is_universal_default")
        return hash(tuple(canon(getattr(self, name)) for name in fields))

    @classmethod
    def default(cls, *, force_universal: bool = False) -> CoordinationParams:
        """Universal-safe defaults (``RestDefenseParams.default`` idiom).

        >>> from silly_kicks.coordination._config import CoordinationParams
        >>> CoordinationParams.default().is_default()
        True
        >>> CoordinationParams.default(force_universal=True).is_default()
        False
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> CoordinationParams:
        """Per-provider params: the base merged KEY-WISE with the generated override (base for an unlisted provider).

        >>> from silly_kicks.coordination._config import CoordinationParams
        >>> CoordinationParams.for_provider("skillcorner") == CoordinationParams()
        True
        """
        return _merge_override(cls(), _provider_params_generated.PROVIDER_COORDINATION_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        """True iff built by :meth:`default` without ``force_universal=True``.

        >>> from silly_kicks.coordination._config import CoordinationParams
        >>> CoordinationParams().is_default()
        False
        >>> CoordinationParams.default().is_default()
        True
        """
        return self._is_universal_default
