"""Coordination run report: coverage warning, per-run provenance, and conservation accounting.

The report is a pure accounting object -- built per shard, merged across shards, and checked for
conservation (windows in == scored + dropped; per-family scored rows == surrogate rows). Its warning category
is distinct from every other coordination/tracking warning (spec 7.13).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silly_kicks.coordination._config import CoordinationParams

#: Drop-reason precedence for a window with no scored row in any method (deterministic attribution, spec 7.13).
DROP_REASON_PRECEDENCE = (
    "goal_end_unresolved",
    "insufficient_detection",
    "insufficient_players",
    "too_short",
    "no_possession_role",
    "not_commensurate",
    "degenerate_constant",
    "entropy_undefined",
)


class CoordinationCoverageWarning(UserWarning):
    """A window/metric was emitted with NaN because observed coverage fell below the threshold (spec 7.13) -- or a
    window builder skipped input the frames do not cover (actions in a period with no tracking get no window).

    Subclasses no other warning category and is subclassed by none -- callers filter it precisely.

    Examples
    --------
    Silence only this category while scoring a low-coverage match::

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CoordinationCoverageWarning)
            result = compute_team_coordination(frames)
    """


def first_drop_reason(reasons: Iterable[str]) -> str | None:
    """The highest-precedence token in ``reasons`` (per :data:`DROP_REASON_PRECEDENCE`), or ``None`` if empty.

    Examples
    --------
    >>> first_drop_reason({"too_short", "insufficient_players"})
    'insufficient_players'
    >>> first_drop_reason(set()) is None
    True
    """
    present = set(reasons)
    for token in DROP_REASON_PRECEDENCE:
        if token in present:
            return token
    return None


def _sum_counts(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _sum_nested(a: Mapping[str, Mapping[str, int]], b: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {k: dict(v) for k, v in a.items()}
    for family, tokens in b.items():
        out[family] = _sum_counts(out.get(family, {}), tokens)
    return out


@dataclass(frozen=True)
class CoordinationReport:
    """Per-run coordination provenance and conservation counters.

    Examples
    --------
    Returned alongside every family table; check its conservation + per-source counts::

        pair, phase, report = compute_relative_phase(signals)
        assert report.conservation_errors() == []           # windows/rows accounting balances
        report.rows_by_source["coordination_pair"]           # {source token -> row count}
    """

    params: CoordinationParams
    provider: str
    window_source: str
    detection_source: str
    stoppage_source: str
    native_hz: float
    effective_hz: float
    rate_capped: bool
    n_windows_in: int
    n_windows_scored: int
    windows_dropped: Mapping[str, int] = field(default_factory=dict)  # reason token -> windows with no scored row
    n_segments: int = 0
    n_runs_too_short: int = 0
    samples_stationary: int = 0
    samples_unobserved: int = 0
    samples_below_min_players: int = 0
    rows_by_source: Mapping[str, Mapping[str, int]] = field(default_factory=dict)  # family -> source token -> rows
    surrogate_rows_by_source: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )  # family -> surr token -> rows
    dead_seconds: float = 0.0
    n_stoppage_splits: int = 0
    iaaft_nonconverged: int = 0

    _PROVENANCE = (
        "params",
        "provider",
        "window_source",
        "detection_source",
        "stoppage_source",
        "native_hz",
        "effective_hz",
        "rate_capped",
    )

    def merge(self, other: CoordinationReport) -> CoordinationReport:
        """Combine two shard reports; provenance must match, counters sum, family maps union key-wise.

        Examples
        --------
        Fold per-shard reports into one run-level report::

            total = shard_reports[0]
            for r in shard_reports[1:]:
                total = total.merge(r)
        """
        for name in self._PROVENANCE:
            if getattr(self, name) != getattr(other, name):
                raise ValueError(f"cannot merge coordination reports: {name} differs")
        return CoordinationReport(
            params=self.params,
            provider=self.provider,
            window_source=self.window_source,
            detection_source=self.detection_source,
            stoppage_source=self.stoppage_source,
            native_hz=self.native_hz,
            effective_hz=self.effective_hz,
            rate_capped=self.rate_capped,
            n_windows_in=self.n_windows_in + other.n_windows_in,
            n_windows_scored=self.n_windows_scored + other.n_windows_scored,
            windows_dropped=_sum_counts(self.windows_dropped, other.windows_dropped),
            n_segments=self.n_segments + other.n_segments,
            n_runs_too_short=self.n_runs_too_short + other.n_runs_too_short,
            samples_stationary=self.samples_stationary + other.samples_stationary,
            samples_unobserved=self.samples_unobserved + other.samples_unobserved,
            samples_below_min_players=self.samples_below_min_players + other.samples_below_min_players,
            rows_by_source=_sum_nested(self.rows_by_source, other.rows_by_source),
            surrogate_rows_by_source=_sum_nested(self.surrogate_rows_by_source, other.surrogate_rows_by_source),
            dead_seconds=self.dead_seconds + other.dead_seconds,
            n_stoppage_splits=self.n_stoppage_splits + other.n_stoppage_splits,
            iaaft_nonconverged=self.iaaft_nonconverged + other.iaaft_nonconverged,
        )

    def conservation_errors(self) -> list[str]:
        """Empty iff windows balance (in == scored + dropped) and each family's surrogate rows match its scored rows.

        Examples
        --------
        >>> r = CoordinationReport(params=None, provider="sportec", window_source="period",
        ...     detection_source="fully_observed", stoppage_source="ball_state", native_hz=25.0,
        ...     effective_hz=10.0, rate_capped=True, n_windows_in=3, n_windows_scored=2,
        ...     windows_dropped={"too_short": 1})
        >>> r.conservation_errors()
        []
        """
        errors: list[str] = []
        if self.n_windows_in != self.n_windows_scored + sum(self.windows_dropped.values()):
            errors.append("window conservation: n_windows_in != n_windows_scored + sum(windows_dropped)")
        for family, tokens in self.surrogate_rows_by_source.items():
            scored = sum(self.rows_by_source.get(family, {}).values())
            if sum(tokens.values()) != scored:
                errors.append(f"surrogate conservation: {family} surrogate rows != scored rows")
        return errors
