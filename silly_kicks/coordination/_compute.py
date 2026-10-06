"""Coordination family computes + orchestrator (spec 7.2, 7.8, 7.9).

Each ``compute_*`` runs one method family over the prepared :class:`CoordinationSignals`, per (pair|team, window),
emitting the family's ``*_COLUMNS`` table(s) plus a :class:`CoordinationReport`. The four pair families share the
COORD_PAIR schema (each fills only its own columns; the orchestrator coalesces them). Positional signals live in
team A's goal-relative frame; a window whose reference team is B negates them (C24). Surrogates (spec 7.9) draw per
segment from a key-derived generator, so results are order-independent.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from silly_kicks.coordination._catalog import METHODS_BY_LEVEL, PairSpec, resolve_pairs
from silly_kicks.coordination._columns import (
    COORD_SIGNALS,
    COORDINATION_CLUSTER_PLAYER_COLUMNS,
    COORDINATION_CLUSTER_TEAM_COLUMNS,
    COORDINATION_PAIR_COLUMNS,
    COORDINATION_PAIR_KEYS,
    COORDINATION_PAIR_PHASE_COLUMNS,
    COORDINATION_RSI_COLUMNS,
    COORDINATION_SPECTRAL_COLUMNS,
    COORDINATION_TEAM_SYNC_COLUMNS,
    DEFAULT_LEVELS,
    HIST_BIN_LABELS,
    TEAM_SIGNALS,
)
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._detection import DetectionCounts, insufficient_detection, row_detected_share
from silly_kicks.coordination._kernels import _cluster_reference
from silly_kicks.coordination._kernels._circular import circular_summary, hist_bin_index, near_in_phase
from silly_kicks.coordination._kernels._cluster import (
    ClusterWindowStats,
    ShiftRun,
    cluster_phase,
    pearson_rows,
    phasor_parts,
    shifted_rho_group_means,
    window_cluster_stats,
)
from silly_kicks.coordination._kernels._entropy import cross_sampen_over_runs, sampen_over_runs
from silly_kicks.coordination._kernels._phase import analytic_phase, pad_length, phasor
from silly_kicks.coordination._kernels._spectral import (
    cross_spectra_batch,
    median_frequency_cpm,
    pooled_coherence,
    pooled_median_frequency,
    welch_spectra,
)
from silly_kicks.coordination._kernels._surrogates import (
    draw_shifts,
    iaaft,
    lagged_pearson_b_side,
    near_in_phase_counts_at,
    phasor_spectrum,
    shifted_phasor_sums,
    shifted_slice_lagged_pearson,
    shifted_window,
    surrogate_rng,
    surrogate_triple,
)
from silly_kicks.coordination._kernels._vector_coding import classify, coupling_angle_deg, stationary_mask
from silly_kicks.coordination._kernels._xcorr import (
    fisher_pool,
    lagged_pearson,
    lagged_pearson_batch,
    min_slice_samples,
    xcorr_summary,
)
from silly_kicks.coordination._report import CoordinationCoverageWarning, CoordinationReport, first_drop_reason
from silly_kicks.coordination._signals import (
    CoordinationSignals,
    PeriodSignals,
    PlayerSeries,
    build_coordination_signals,
)
from silly_kicks.coordination._windows import (
    period_windows,
    phase_assignment,
    possession_windows_from_actions,
    possession_windows_from_frames,
)
from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.tracking import GoalMap


# --------------------------------------------------------------------------- generic helpers
def _cast(rows: list[dict[str, Any]], columns: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(columns))
    for col, dtype in columns.items():
        if dtype == "Int64":
            df[col] = df[col].astype("Int64")
        elif dtype == "int64":
            df[col] = df[col].astype("int64")
        elif dtype == "float64":
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype(object)
    return df


def _cid(v: Any) -> object:  # Any: pandas isna overloads reject `object` (the pandas-Scalar friction)
    return pd.NA if pd.isna(v) else canonical_id(v)


def _orientation_signs(spec: PairSpec, reference_is_b: bool) -> tuple[float, float]:
    """Per-signal sign for the window frame: -1 for a positional signal when the reference team is B (C24)."""

    def sign(sig: str) -> float:
        return -1.0 if (reference_is_b and COORD_SIGNALS[sig].kind == "positional") else 1.0

    return sign(spec.signal_a), sign(spec.signal_b)


#: Signals that depend on the team's DEFENDED END: the positional ones (oriented by the goal) AND the back-line-derived
#: ``compactness_x`` -- a MAGNITUDE signal, but built from ``defends0`` (the defended-end mask) in ``_team_signals``,
#: so it too is computed against a GUESSED end when the goal is unresolved (review A-24). ``defensive_line_x`` /
#: ``back_line_high_x`` are already positional and so covered by the kind check.
_GOAL_END_DEPENDENT = frozenset({"compactness_x"})


def _sig_goal_unresolved(ps: PeriodSignals, sig: str, team: object) -> bool:
    """True iff ``sig`` needs ``team``'s defended end and that end is unresolved (``goal_x`` is None)."""
    end_dependent = COORD_SIGNALS[sig].kind == "positional" or sig in _GOAL_END_DEPENDENT
    return end_dependent and ps.goal_x.get(team) is None


def _goal_unresolved(ps: PeriodSignals, spec: PairSpec, b: _Binding) -> bool:
    """A signal whose team's defended end could not be resolved (goal_x is None) is computed against a guessed end
    -- every positional signal and the back-line-derived ``compactness_x`` (review A-24)."""
    return _sig_goal_unresolved(ps, spec.signal_a, b.team_a_id) or _sig_goal_unresolved(ps, spec.signal_b, b.team_b_id)


def _is_possession(window_source: object) -> bool:
    return window_source in ("possession_events", "possession_tracking")


def _ref_is_b(ps: PeriodSignals, wrow: pd.Series) -> bool:
    if not _is_possession(wrow["window_source"]):
        return False
    att = wrow["attacking_team_id"]
    return same_id(att, ps.team_ids[1])  # an NA attacking team is no team (ADR-019)


def _iter_windows(signals: CoordinationSignals):
    for ps in signals.periods:
        for row_idx, s, e in ps.window_ranges:
            yield ps, signals.windows.loc[row_idx], int(s), int(e)


def _seed_game(ps: PeriodSignals) -> str:
    """The game component of every surrogate seed key: the CANONICAL id (spec 7.9, ADR-019), so a game id stored as
    1.0 draws exactly what the same game stored as 1 draws."""
    return str(canonical_id(ps.game_id))


def _wkey(ps: PeriodSignals, wrow: pd.Series) -> tuple:
    return (str(canonical_id(ps.game_id)), int(ps.period_id), wrow["window_kind"], int(wrow["window_id"]))


def _f(v) -> float:
    return float(v)


# --------------------------------------------------------------------------- report accumulation
@dataclass
class _Accum:
    family: str
    rows_by_source: dict[str, int]
    surrogate_rows_by_source: dict[str, int]
    windows_scored: set
    windows_drop_reason: dict
    #: Per-family coverage diagnostics the report sums (review A-19; spec 7.13). Vector coding fills
    #: ``samples_stationary`` (consecutive-diff samples omitted as stationary); the cluster family fills
    #: ``samples_below_min_players`` (team present but fewer than ``min_players`` valid phasors). Others leave them 0.
    samples_stationary: int = 0
    samples_below_min_players: int = 0

    @classmethod
    def new(cls, family: str) -> _Accum:
        return cls(family, {}, {}, set(), {})

    def note(self, wkey, source: str) -> None:
        self.rows_by_source[source] = self.rows_by_source.get(source, 0) + 1
        if source == "scored":
            self.windows_scored.add(wkey)
        else:
            self.windows_drop_reason.setdefault(wkey, set()).add(source)

    def note_surrogate(self, source: str) -> None:
        if source is not None and not (isinstance(source, float) and np.isnan(source)):
            self.surrogate_rows_by_source[source] = self.surrogate_rows_by_source.get(source, 0) + 1


def _report_from(
    signals: CoordinationSignals, accums: Sequence[_Accum], iaaft_nonconverged: int = 0
) -> CoordinationReport:
    windows_in = len(signals.windows)
    rows_by_source: dict[str, dict[str, int]] = {}
    surrogate_rows: dict[str, dict[str, int]] = {}
    for a in accums:  # same-family accums (e.g. the 4 pair methods) merge, not overwrite
        fam = rows_by_source.setdefault(a.family, {})
        for src, n in a.rows_by_source.items():
            fam[src] = fam.get(src, 0) + n
        if a.surrogate_rows_by_source:
            sfam = surrogate_rows.setdefault(a.family, {})
            for src, n in a.surrogate_rows_by_source.items():
                sfam[src] = sfam.get(src, 0) + n
    scored: set = set()
    dropped: dict = {}
    for a in accums:
        scored |= a.windows_scored
        for wkey, reasons in a.windows_drop_reason.items():
            dropped.setdefault(wkey, set()).update(reasons)
    windows_dropped: dict[str, int] = {}
    for wkey, reasons in dropped.items():
        if wkey in scored:
            continue
        tok = first_drop_reason(reasons) or "too_short"
        windows_dropped[tok] = windows_dropped.get(tok, 0) + 1
    # A window that emitted NO row at all (e.g. a possession window under `levels=("dyad",)`, which dyads skip) is in
    # neither `scored` nor `dropped`; it must be counted under a reason, never silently as scored (review A-30).
    w = signals.windows
    all_keys = {
        (str(_cid(g)), int(p), k, int(wid))
        for g, p, k, wid in zip(
            w["game_id"].tolist(),
            w["period_id"].tolist(),
            w["window_kind"].tolist(),
            w["window_id"].tolist(),
            strict=True,
        )
    }
    no_row = len(all_keys - scored - set(dropped))
    if no_row:
        windows_dropped["no_row"] = windows_dropped.get("no_row", 0) + no_row
    n_scored = len(scored)
    return CoordinationReport(
        params=signals.params,
        provider=signals.provider,
        window_source=signals.window_regime,
        detection_source=signals.detection_source,
        stoppage_source=signals.stoppage.source,
        native_hz=signals.native_hz,
        effective_hz=signals.fs,
        rate_capped=signals.rate_capped,
        n_windows_in=windows_in,
        n_windows_scored=n_scored,
        windows_dropped=windows_dropped,
        n_segments=int(signals.counters.get("n_segments", 0)),
        n_runs_too_short=int(signals.counters.get("n_runs_too_short", 0)),
        # A-19: the coverage counters are now populated -- unobserved at signal build, the other two by the families.
        samples_unobserved=int(signals.counters.get("samples_unobserved", 0)),
        samples_stationary=sum(a.samples_stationary for a in accums),
        samples_below_min_players=sum(a.samples_below_min_players for a in accums),
        rows_by_source=rows_by_source,
        surrogate_rows_by_source=surrogate_rows,
        dead_seconds=float(signals.stoppage.dead_seconds),
        n_stoppage_splits=int(signals.stoppage.n_splits),
        iaaft_nonconverged=iaaft_nonconverged,
    )


def _accum_from_rows(family: str, rows: list[dict[str, Any]], source_col: str, surr_col: str | None) -> _Accum:
    """Derive a family accum from its emitted rows (one source + one surrogate token per row)."""
    acc = _Accum.new(family)
    for row in rows:
        wk = (str(_cid(row["game_id"])), int(row["period_id"]), row["window_kind"], int(row["window_id"]))
        acc.note(wk, row[source_col])
        if surr_col is not None:
            acc.note_surrogate(row[surr_col])
    return acc


def _coverage_message(report: CoordinationReport) -> str | None:
    """Spec 7.13's one call-level trigger: a message when the call's dropped share of WINDOWS (windows with no scored
    row) exceeds ``coverage_warn_fraction``, else ``None``.

    Each public compute warns with it itself, ``stacklevel=2``, so the warning names the code that called the public
    function -- a helper that warned would name this module instead.
    """
    n_in = report.n_windows_in
    dropped = n_in - report.n_windows_scored
    if not n_in or dropped / n_in <= report.params.coverage_warn_fraction:
        return None
    return (
        f"{dropped}/{n_in} coordination windows have no scored row (> coverage_warn_fraction "
        f"{report.params.coverage_warn_fraction}); by reason: {dict(report.windows_dropped)}"
    )


# --------------------------------------------------------------------------- pair bindings
@dataclass(frozen=True)
class _Binding:
    team_a_id: object
    team_b_id: object
    player_a_id: object
    player_b_id: object
    ref_is_b: bool


def _dyad_bindings(
    ps: PeriodSignals, spec: PairSpec, ref_is_b: bool, *, include_goalkeeper: bool, team_order: tuple[object, object]
) -> list[_Binding]:
    """Every same-team or cross-team player pair; goalkeepers take part iff ``include_goalkeeper`` (the
    ``include_goalkeeper["dyad"]`` flag, spec 7.14; the player series hold the keeper whenever the flag is set).

    ``team_order`` is the window's spec-7.7 pair order ``(first, second)`` (attacking, defending on possession
    windows; else canonical): a CROSS-team dyad is ordered first-team-as-A, so A = the attacking team wherever that
    is defined (review A-23). Same-team pairs are unaffected (no attacking side within a team)."""
    ta, tb = ps.team_ids
    roster: dict[object, list] = {ta: [], tb: []}
    for tm, pid in ps.players:
        if (include_goalkeeper or not ps.players[(tm, pid)].is_goalkeeper) and tm in roster:
            roster[tm].append(pid)
    for tm in roster:
        roster[tm].sort(key=lambda p: str(canonical_id(p)))
    out: list[_Binding] = []
    if spec.role == "same_team":
        for tm in (ta, tb):
            lst = roster[tm]
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    out.append(_Binding(tm, tm, lst[i], lst[j], ref_is_b))
    else:
        first, second = team_order
        for pa in roster[first]:
            for pb in roster[second]:
                out.append(_Binding(first, second, pa, pb, ref_is_b))
    return out


def _attacking_defending(attacking_team_id: object, ta: object, tb: object) -> tuple[object, object]:
    """Map the windows table's ``attacking_team_id`` to the PERIOD's (attacking, defending) team objects. The returned
    ids are the frames' team ids (``ta``/``tb``), never the raw window value, because they key into the frame-keyed
    signal maps -- a str/int dtype drift between the actions-derived window and the frames would otherwise KeyError
    (review A-26). Matching is via ``id_compat`` (ADR-019), never a raw ``==``."""
    return (ta, tb) if same_id(attacking_team_id, ta) else (tb, ta)


def _pair_teams(ps: PeriodSignals, wrow: pd.Series) -> tuple[object, object]:
    """The spec 7.7 pair order: on a possession window with a known attacker, (attacking, defending); otherwise the
    PERIOD's canonical ``(team_a, team_b)``. One source for every team-team pair order -- team-team / cross-variable
    pairs, cross-team dyads and the relative stretch index -- so A = the attacking team wherever Moura's x-axis is
    defined (review A-23)."""
    ta, tb = ps.team_ids
    if _is_possession(wrow["window_source"]) and not pd.isna(wrow["attacking_team_id"]):
        return _attacking_defending(wrow["attacking_team_id"], ta, tb)
    return ta, tb


def _bindings_for(ps: PeriodSignals, spec: PairSpec, wrow: pd.Series, *, dyad_goalkeeper: bool) -> list[_Binding] | str:
    """The pair bindings of ``spec`` in the window ``wrow``; ``dyad_goalkeeper`` is ``include_goalkeeper["dyad"]``
    (required, so no caller can silently drop the flag)."""
    ref_is_b = _ref_is_b(ps, wrow)
    ta, tb = ps.team_ids
    possession = _is_possession(wrow["window_source"])
    if spec.level == "dyad":
        return _dyad_bindings(ps, spec, ref_is_b, include_goalkeeper=dyad_goalkeeper, team_order=_pair_teams(ps, wrow))
    if spec.role == "attacking_defending":
        if not possession or pd.isna(wrow["attacking_team_id"]):
            return "no_possession_role"
        att, deff = _attacking_defending(wrow["attacking_team_id"], ta, tb)
        return [_Binding(att, deff, pd.NA, pd.NA, ref_is_b)]
    if spec.role == "same_team":
        return [_Binding(ta, ta, pd.NA, pd.NA, ref_is_b), _Binding(tb, tb, pd.NA, pd.NA, ref_is_b)]
    if possession and not pd.isna(wrow["attacking_team_id"]):
        att, other = _attacking_defending(wrow["attacking_team_id"], ta, tb)
        return [_Binding(att, other, pd.NA, pd.NA, ref_is_b)]
    return [_Binding(ta, tb, pd.NA, pd.NA, ref_is_b)]


def _team_det(ps: PeriodSignals, team: object) -> DetectionCounts:
    """A team side's detection counts; a team with no team-signal player in the period has none (share NaN)."""
    det = ps.team_detection.get(team)
    if det is None:
        none = np.zeros(ps.t.size, dtype=bool)
        det = DetectionCounts.from_masks(none, none)
    return det


def _side(ps: PeriodSignals, spec: PairSpec, b: _Binding, side: str):
    """(values, phasor, inst_freq_pos, detection counts, segments) for one side."""
    sig = spec.signal_a if side == "a" else spec.signal_b
    team = b.team_a_id if side == "a" else b.team_b_id
    pid = b.player_a_id if side == "a" else b.player_b_id
    if spec.level == "dyad":
        player = ps.players[(team, pid)]
        is_x = sig == "player_x"
        return (
            player.x if is_x else player.y,
            player.phasor_x if is_x else player.phasor_y,
            player.inst_freq_pos_x if is_x else player.inst_freq_pos_y,
            player.detection,
            player.runs,
        )
    return (
        ps.team_signal[(team, sig)],
        ps.team_phasor[(team, sig)],
        ps.team_inst_freq_pos[(team, sig)],
        _team_det(ps, team),
        ps.segments[team],
    )


def _clipped_sorted(segs: np.ndarray, s: int, e: int) -> list[tuple[int, int]]:
    """The segments overlapping ``[s, e)``, clipped to it and ordered by start (one vectorised overlap test over all of
    them, not a Python pass: fragmented tracking has ~140 segments per side)."""
    segs = np.asarray(segs).reshape(-1, 2)
    hit = segs[(segs[:, 1] > s) & (segs[:, 0] < e)]
    hit = hit[np.argsort(hit[:, 0], kind="stable")]
    return [(max(int(lo), s), min(int(hi), e)) for lo, hi in hit]


def _both_runs(seg_a: np.ndarray, seg_b: np.ndarray, s: int, e: int) -> list[tuple[int, int]]:
    """The window ∩ A-segment ∩ B-segment slices of ``[s, e)``, in order (spec 7.4 step 2 + D15).

    A segment is cut with NO gap where the team's on-pitch count changes (a red card), so two segments can touch; they
    stay two slices here -- never one run across the cut -- and no windowed statistic (cross-correlation, vector-coding
    differences, coherence, the relative stretch index and its switch events) spans the count step.
    """
    a, b = _clipped_sorted(seg_a, s, e), _clipped_sorted(seg_b, s, e)
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


@dataclass(frozen=True)
class _Ctx:
    idx: np.ndarray  # window valid grid indices (both sides in-segment)
    runs: list[tuple[int, int]]
    va: np.ndarray
    vb: np.ndarray
    pha: np.ndarray
    phb: np.ndarray
    ipa: np.ndarray
    ipb: np.ndarray
    det_a: DetectionCounts  # each side's detection counts (spec 7.11, A-08)
    det_b: DetectionCounts
    seg_a: np.ndarray
    seg_b: np.ndarray
    sa: float
    sb: float


def _context(ps: PeriodSignals, spec: PairSpec, b: _Binding, s: int, e: int) -> _Ctx:
    va, pha, ipa, det_a, seg_a = _side(ps, spec, b, "a")
    vb, phb, ipb, det_b, seg_b = _side(ps, spec, b, "b")
    runs = _both_runs(seg_a, seg_b, s, e)
    idx = np.concatenate([np.arange(lo, hi) for lo, hi in runs]) if runs else np.empty(0, dtype=np.int64)
    sa, sb = _orientation_signs(spec, b.ref_is_b)
    return _Ctx(idx, runs, va, vb, pha, phb, ipa, ipb, det_a, det_b, seg_a, seg_b, sa, sb)


def _coverage(ctx: _Ctx, fs: float, s: int, e: int) -> dict[str, Any]:
    """A pair row's coverage: the samples it scores, and each side's raw-detection share over the window with their
    minimum -- the number the ``insufficient_detection`` gate tests (spec 7.11, A-08)."""
    idx = ctx.idx
    return {
        "coord_duration_s": len(idx) / fs,
        "coord_n_samples": len(idx),
        "coord_n_segments": len(ctx.runs),
        "coord_observed_fraction_a": ctx.det_a.share(s, e),
        "coord_observed_fraction_b": ctx.det_b.share(s, e),
        "coord_detected_share": row_detected_share((ctx.det_a, ctx.det_b), s, e),
    }


def _pair_gate(ps: PeriodSignals, spec: PairSpec, b: _Binding, row: dict[str, Any], params, family: str) -> str | None:
    """The tokens that precede a pair family's own scoring, in the Task 14 precedence: ``goal_end_unresolved``, then
    ``insufficient_detection`` (the row's ``coord_detected_share`` below the family threshold); ``None`` to score."""
    if _goal_unresolved(ps, spec, b):
        return "goal_end_unresolved"
    if insufficient_detection(row["coord_detected_share"], params.min_observed_fraction[family]):
        return "insufficient_detection"
    return None


def _phase_children(parent: dict[str, Any], wrow: pd.Series, source_col: str) -> list[dict[str, Any]]:
    """A degraded pair row's phase rows (spec 7.13 / ADR-042; owner ruling 2026-10-04): exactly what a healthy parent
    would emit -- the window's ``n_phases`` rows, none on an NA window -- with the parent's keys and detection share,
    NaN metrics and the PARENT's token. Emitted for any non-scored parent whatever the reason, so a new degradation
    token cannot reintroduce the drop."""
    n = _window_n_phases(wrow)
    if n is None:
        return []
    out: list[dict[str, Any]] = []
    for k in range(1, n + 1):
        child: dict[str, Any] = {c: parent[c] for c in COORDINATION_PAIR_KEYS}
        child["phase_index"] = k
        for c, dt in COORDINATION_PAIR_PHASE_COLUMNS.items():
            if c not in child:
                child[c] = pd.NA if dt in ("Int64", "object") else float("nan")
        for c in ("coord_detected_share", source_col, "coord_detection_source", "coord_stoppage_source"):
            child[c] = parent[c]
        out.append(child)
    return out


def _blank_pair_row(ps: PeriodSignals, wrow: pd.Series, spec: PairSpec, b: _Binding, signals) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": ps.game_id,
        "period_id": int(ps.period_id),
        "window_kind": wrow["window_kind"],
        "window_id": wrow["window_id"],
        "level": spec.level,
        "signal_a": spec.signal_a,
        "signal_b": spec.signal_b,
        "axis": spec.axis,
        "team_a_id": _cid(b.team_a_id),
        "team_b_id": _cid(b.team_b_id),
        "player_a_id": _cid(b.player_a_id),
        "player_b_id": _cid(b.player_b_id),
    }
    for c, dt in COORDINATION_PAIR_COLUMNS.items():
        if c not in row:
            row[c] = pd.NA if dt in ("Int64", "object") else float("nan")
    row["coord_detection_source"] = signals.detection_source
    row["coord_stoppage_source"] = signals.stoppage.source
    return row


# --------------------------------------------------------------------------- surrogate driver
#: Elements per draw-chunk array in the batched pair nulls: bounds peak memory on whole-period windows while a
#: short window scores all its draws in one chunk.
_NULL_CHUNK_ELEMENTS = 1 << 21

#: Set to ``"1"`` to score through the AS-BUILT reference numerics (ADR-111): every surrogate null by the DIRECT
#: estimators -- the batched per-draw loop -- instead of the spec 7.9 identities (D2(b)), and the cluster/team-sync
#: family by its as-built complex arithmetic (``_cluster_reference``) instead of D4's. The parity oracle of the tests
#: and the reference leg of the corpus no-flip gate (``scripts/validate_coordination_numerics.py``); production never
#: sets it. Read per call, like ``SILLY_KICKS_COORDINATION_FORCE_NUMPY``.
REFERENCE_NUMERICS_ENV = "SILLY_KICKS_COORDINATION_REFERENCE_NUMERICS"


def _reference_numerics() -> bool:
    return os.environ.get(REFERENCE_NUMERICS_ENV) == "1"


#: Relative-phase dispatch crossover (ADR-111, owner ruling A 2026-09-28): the spec 7.9 identity runs only where it is
#: cheaper than the direct null. The identity costs three complex FFTs per touched B segment (~3 ns per n log2 n unit
#: each) plus the direct near-in-phase count (~1.5 ns per draw x idx row); the direct null costs ~40 ns per draw x
#: idx row (numpy 2.4 / scipy 1.18, x86-64, measured on the three reference matches). The identity wins iff
#: ``sum_g n_g * ceil(log2 n_g) <= (40 - 1.5) / 9 * K * |idx|``; 38.5 / 9 = 4.28 is fixed at 17/4. A FIXED size rule:
#: it reads sizes only, never values, in exact rational arithmetic (no platform-dependent float ``log2`` at the
#: boundary), and both branches are parity-equal (tests), so it only chooses speed, never a result.
RP_IDENTITY_CROSSOVER = Fraction(17, 4)


def _rp_identity_is_cheaper(idx: np.ndarray, segs: Sequence[tuple[int, int]], n_draws: int) -> bool:
    """Decision A's size rule: does the relative-phase identity beat the direct null for this window's sizes?

    FFT work counts only the B segments the window's rows touch (``_rp_identity`` skips the others), as
    ``n * ceil(log2 n)`` integer units; the direct work is ``n_draws * |idx|`` products.
    """
    fft_units = 0
    for lo, hi in segs:
        i0, i1 = np.searchsorted(idx, (lo, hi))
        if i1 > i0:
            n = int(hi) - int(lo)
            fft_units += n * (n - 1).bit_length()
    return fft_units <= RP_IDENTITY_CROSSOVER * n_draws * int(idx.size)


def _draw_chunks(n_draws: int, width: int) -> list[tuple[int, int]]:
    """``[d0, d1)`` draw ranges whose ``(d1 - d0) x width`` arrays stay within ``_NULL_CHUNK_ELEMENTS``."""
    per = max(1, _NULL_CHUNK_ELEMENTS // max(1, width))
    return [(d0, min(n_draws, d0 + per)) for d0 in range(0, n_draws, per)]


class _SurrogateB(Protocol):
    """Series B under every surrogate draw, read per row range and draw chunk -- the batched pair nulls' only view
    of it (never a per-draw full-period copy). ``values``/``phasors`` return ``(d1 - d0, hi - lo)`` arrays."""

    @property
    def n_draws(self) -> int: ...  # read-only: the frozen _ShiftedB field and _StackedB's property both satisfy it

    def values(self, lo: int, hi: int, d0: int, d1: int) -> np.ndarray: ...

    def phasors(self, lo: int, hi: int, d0: int, d1: int) -> np.ndarray: ...


@dataclass(frozen=True)
class _ShiftedB:
    """Time-shift draws: each segment of B circularly shifted by its draw, gathered row by row (``shifted_window``)."""

    ctx: _Ctx
    segs: list[tuple[int, int]]
    shifts: list[np.ndarray]  # per segment, (n_draws,)
    n_draws: int

    def values(self, lo: int, hi: int, d0: int, d1: int) -> np.ndarray:
        return shifted_window(self.ctx.vb, self.segs, [sh[d0:d1] for sh in self.shifts], lo, hi, d1 - d0)

    def phasors(self, lo: int, hi: int, d0: int, d1: int) -> np.ndarray:
        return shifted_window(self.ctx.phb, self.segs, [sh[d0:d1] for sh in self.shifts], lo, hi, d1 - d0)


@dataclass(frozen=True)
class _StackedB:
    """IAAFT draws (not shifts, so materialised once per draw) over the window rows ``[offset, offset + width)``."""

    vb: np.ndarray  # (n_draws, width)
    phb: np.ndarray  # (n_draws, width) complex
    offset: int

    @property
    def n_draws(self) -> int:
        return self.vb.shape[0]

    def values(self, lo: int, hi: int, d0: int, d1: int) -> np.ndarray:
        return self.vb[d0:d1, lo - self.offset : hi - self.offset]

    def phasors(self, lo: int, hi: int, d0: int, d1: int) -> np.ndarray:
        return self.phb[d0:d1, lo - self.offset : hi - self.offset]


def _iaaft_draws(ps, ctx: _Ctx, segs, params: CoordinationParams, fs: float, descriptor, family: str):
    """Every IAAFT draw of B over the window's rows (``[idx.min(), idx.max()]``), with the non-convergence count.

    Per draw, each overlapping segment's SIGNAL is IAAFT-surrogated from its own generator (draw after draw, as the
    per-draw loop consumed it) and its phase recomputed (estimator identity, A1); other rows keep B's observed rows.
    """
    w0, w1 = int(ctx.idx.min()), int(ctx.idx.max()) + 1
    seg_rngs = [
        surrogate_rng(params.surrogate_seed, (_seed_game(ps), int(ps.period_id), int(lo), *descriptor, family))
        for lo, _hi in segs
    ]
    vb = np.empty((params.n_surrogates, w1 - w0))
    phb = np.empty((params.n_surrogates, w1 - w0), dtype=np.complex128)
    nonconv = 0
    for k in range(params.n_surrogates):
        vb[k] = ctx.vb[w0:w1]
        phb[k] = ctx.phb[w0:w1]
        for (lo, hi), rng in zip(segs, seg_rngs, strict=True):
            surr, converged = iaaft(ctx.vb[lo:hi], rng, params.iaaft_max_iter)
            if not converged:
                nonconv += 1
            ph = phasor(analytic_phase(surr, pad_length(hi - lo, fs, params.band_low_cpm)))
            a, c = max(lo, w0), min(hi, w1)
            vb[k, a - w0 : c - w0] = surr[a - lo : c - lo]
            phb[k, a - w0 : c - w0] = ph[a - lo : c - lo]
    return _StackedB(vb, phb, w0), nonconv


def _segments_holding(segments, rows: np.ndarray) -> list[tuple[int, int]]:
    """The segments (sorted, disjoint ``(lo, hi)`` rows) holding at least one of ``rows`` -- the rows a statistic
    reads. These are plan Task 17's CONTRIBUTING segments: the only ones its surrogate shifts and checks. A segment that
    merely overlaps the window's span, holding none of the rows, takes no part -- the null never reads it, so it must
    not refuse the row as ``segment_too_short`` (or mark it ``computed_nonconverged``) either. A read row outside every
    segment breaks the slicing invariant (every slice lies inside one segment of each side) and says so."""
    segs = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return []
    g = np.searchsorted(segs[:, 0], rows, side="right") - 1
    inside = (g >= 0) & (rows < segs[np.maximum(g, 0), 1])
    if not inside.all():
        raise RuntimeError(f"{int((~inside).sum())} scored row(s) lie outside every B segment (a bug)")
    return [(int(lo), int(hi)) for lo, hi in segs[np.unique(g)]]


#: The ONE source of truth for which surrogate-null method each family implements (review A-17). The pair families
#: honour both the time-shift (default) and the IAAFT opt-in; the cluster and team-sync nulls implement only the
#: time shift, and REFUSE IAAFT fail-loud (never silently time-shift and report `computed`). The union is exactly the
#: globally accepted ``surrogate_method`` values (asserted in ``test_surrogate_methods_map_covers_every_null_family``).
SURROGATE_METHODS_BY_FAMILY: dict[str, frozenset[str]] = {
    "relative_phase": frozenset({"time_shift", "iaaft"}),
    "cross_correlation": frozenset({"time_shift", "iaaft"}),
    "vector_coding": frozenset({"time_shift", "iaaft"}),
    "coherence": frozenset({"time_shift", "iaaft"}),
    "cluster": frozenset({"time_shift"}),
    "team_sync": frozenset({"time_shift"}),
}


def _require_surrogate_method(family: str, method: str) -> None:
    """Refuse a surrogate method a family does not implement (review A-17): fail loud rather than silently substitute
    the time shift and stamp ``computed`` -- a surrogate method is an inferential choice, not a cosmetic one."""
    supported = SURROGATE_METHODS_BY_FAMILY[family]
    if method not in supported:
        raise NotImplementedError(
            f"surrogate_method={method!r} is not implemented for the {family} null; supported: {sorted(supported)}"
        )


def _surrogate(ps, ctx, signal_b, params, fs, descriptor, family, null_fn, obs_vals, slices):
    """Return ({metric: triple}, source, n_nonconverged) for the metrics in ``obs_vals``.

    ``slices`` are the rows the observed statistic reads; the null draws from the B segments holding them
    (:func:`_segments_holding`), so a too-short B segment the statistic never reads does not refuse the row.
    ``null_fn(draws)`` scores EVERY surrogate draw of series B at once (one float per draw per metric), byte-identical
    to re-scoring a per-draw copy of B: time-shift draws are gathered rows (``_ShiftedB``), IAAFT draws materialised
    once each (``_StackedB``).
    """
    triples = {m: (float("nan"), float("nan"), float("nan")) for m in obs_vals}
    if params.n_surrogates == 0:
        return triples, "disabled", 0
    if all(np.isnan(v) for v in obs_vals.values()):
        return triples, "not_scored", 0
    segs = _segments_holding(ctx.seg_b, np.concatenate([np.arange(lo, hi) for lo, hi in slices]))
    nonconv = 0
    if params.surrogate_method == "time_shift":
        tau = round(params.min_shift_s[signal_b if signal_b in params.min_shift_s else "cluster_amplitude"] * fs)
        shift_cols: list[np.ndarray] = []
        for lo, hi in segs:
            rng = surrogate_rng(
                params.surrogate_seed, (_seed_game(ps), int(ps.period_id), int(lo), *descriptor, family)
            )
            sh = draw_shifts(rng, hi - lo, tau, params.n_surrogates)
            if sh is None:
                return triples, "segment_too_short", 0
            shift_cols.append(sh)
        dists = null_fn(_ShiftedB(ctx, segs, shift_cols, params.n_surrogates))
        src = "computed"
    else:  # iaaft: surrogate the B SIGNAL per segment; recompute its phase (estimator identity, A1)
        stacked, nonconv = _iaaft_draws(ps, ctx, segs, params, fs, descriptor, family)
        dists = null_fn(stacked)
        src = "computed_nonconverged" if nonconv > 0 else "computed"
    for m in obs_vals:
        triples[m] = surrogate_triple(obs_vals[m], dists[m])
    return triples, src, nonconv


# --------------------------------------------------------------------------- batched pair nulls (one per family)
class _SegmentCache:
    """One compute call's memo of a B segment's identity side (R: ``phasor_spectrum``; cross-correlation:
    ``lagged_pearson_b_side``), shared by every window rolled within that segment (ADR-111 ruling C). Keyed by the
    series object and the segment bounds; each entry keeps its series alive, so the ``id`` key cannot be reused."""

    def __init__(self, build: Callable[[np.ndarray], Any]) -> None:
        self._build = build
        self._memo: dict[tuple[int, int, int], tuple[np.ndarray, Any]] = {}

    def get(self, series: np.ndarray, lo: int, hi: int) -> Any:
        key = (id(series), int(lo), int(hi))
        hit = self._memo.get(key)
        if hit is None or hit[0] is not series:
            hit = (series, self._build(series[lo:hi]))
            self._memo[key] = hit
        return hit[1]


def _rp_identity(
    ctx: _Ctx, a: np.ndarray, draws: _ShiftedB, cos_thr: float, b_sides: _SegmentCache | None = None
) -> dict[str, np.ndarray] | None:
    """R and % near-in-phase for every time-shift draw through the spec 7.9 identities, or ``None`` to decline.

    Per B segment ``[lo, hi)`` touching the idx rows (draw ``d`` rolls it by its shift): the R phasor sum over the idx
    rows inside it is ONE circular cross-correlation of A -- zero outside those rows -- with the segment
    (``shifted_phasor_sums``, C11), and the near-in-phase counts are direct (``near_in_phase_counts_at``, numba
    optional). Declines when a touched segment holds a non-finite phasor: the FFT would spread it over every shift,
    while the direct null NaNs only the draws that roll it onto an idx row.
    """
    idx = ctx.idx
    sums = np.zeros(draws.n_draws, dtype=np.complex128)
    counts = np.zeros(draws.n_draws, dtype=np.int64)
    for (lo, hi), shifts in zip(draws.segs, draws.shifts, strict=True):
        i0, i1 = (int(v) for v in np.searchsorted(idx, (lo, hi)))
        if i1 == i0:
            continue  # the direct null rolls this segment too, but reads none of its rows
        seg = ctx.phb[lo:hi]
        if not np.isfinite(seg).all():
            return None
        t_loc = idx[i0:i1] - lo
        a_seg = np.zeros(hi - lo, dtype=np.complex128)
        a_seg[t_loc] = a[i0:i1]
        spectrum = None if b_sides is None else b_sides.get(ctx.phb, lo, hi)
        sums += shifted_phasor_sums(a_seg, seg, shifts, zb_spectrum=spectrum)
        counts += near_in_phase_counts_at(a[i0:i1], t_loc, seg, shifts, cos_thr)
    return {
        "coord_rp_resultant_length": np.abs(sums) / idx.size,
        "coord_rp_pct_near_in_phase": counts / idx.size,
    }


def _rp_null(ctx: _Ctx, cos_thr: float, b_sides: _SegmentCache | None = None):
    """Relative phase R and % near-in-phase for every draw. Time-shift draws score through the spec 7.9 identities
    (:func:`_rp_identity`) where the size rule says they are cheaper (:func:`_rp_identity_is_cheaper`); IAAFT draws,
    the other windows, a declined identity and the reference mode take the direct null: one product
    ``(sa*sb) * pha * conj(phb)`` per chunk, C-contiguous row sums (pairwise, as the 1-D ``sum``), the same scalar
    ``abs``, exact near-in-phase counts -- bit-identical to the per-draw loop."""
    idx = ctx.idx
    a = (ctx.sa * ctx.sb) * ctx.pha[idx]
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    cols = idx - lo

    def run(draws: _SurrogateB) -> dict[str, np.ndarray]:
        if (
            isinstance(draws, _ShiftedB)
            and not _reference_numerics()
            and _rp_identity_is_cheaper(idx, draws.segs, draws.n_draws)
        ):
            scored = _rp_identity(ctx, a, draws, cos_thr, b_sides)
            if scored is not None:
                return scored
        r = np.empty(draws.n_draws)
        near = np.empty(draws.n_draws)
        for d0, d1 in _draw_chunks(draws.n_draws, hi - lo):
            g = np.ascontiguousarray(np.take(draws.phasors(lo, hi, d0, d1), cols, axis=1))
            zz = np.ascontiguousarray(a[None, :] * np.conj(g))
            sums = zz.sum(axis=1)
            for i in range(d1 - d0):
                r[d0 + i] = float(abs(sums[i]) / idx.size)
            near[d0:d1] = np.mean(np.real(zz) >= cos_thr, axis=1)
        return {"coord_rp_resultant_length": r, "coord_rp_pct_near_in_phase": near}

    return run


def _xc_identity(
    ctx: _Ctx, slices: list[tuple[int, int]], lag: int, draws: _ShiftedB, b_sides: _SegmentCache | None = None
) -> np.ndarray | None:
    """The Fisher-pooled lag profile of every time-shift draw, ``(n_draws, 2 * lag + 1)``, through the spec 7.9
    identity (``shifted_slice_lagged_pearson``: one circular FFT cross-correlation per slice, the exact edge
    correction, prefix sums) -- or ``None`` to decline when a slice's B segment (or the slice's A values) holds a
    non-finite value, which the FFT would spread over every shift."""
    sign = ctx.sa * ctx.sb
    rr: list[np.ndarray] = []
    nn: list[np.ndarray] = []
    for lo, hi in slices:
        held = [g for g, (glo, ghi) in enumerate(draws.segs) if glo <= lo and hi <= ghi]
        if len(held) != 1:  # `_both_runs` slices lie inside ONE segment of each side -- touching segments included
            raise RuntimeError(f"cross-correlation slice [{lo}, {hi}) lies in {len(held)} B segments (a bug)")
        glo, ghi = draws.segs[held[0]]
        seg, a = ctx.vb[glo:ghi], ctx.va[lo:hi]
        if not (np.isfinite(seg).all() and np.isfinite(a).all()):
            return None
        b_side = None if b_sides is None else b_sides.get(ctx.vb, glo, ghi)
        r, n = shifted_slice_lagged_pearson(a, seg, lo - glo, draws.shifts[held[0]], lag, b_side=b_side)
        rr.append(sign * r)
        nn.append(n)
    return fisher_pool(np.ascontiguousarray(np.stack(rr, axis=1)), np.array(nn))


def _peak_abs_r(pooled: np.ndarray) -> np.ndarray:
    """``xcorr_summary(row, fs)[0]`` of every row of ``pooled`` ``(draws, 2L + 1)``: the peak |r| over the finite lags,
    NaN for a row without one. A max -- exact in any evaluation order, so the vectorised form is bit-identical."""
    finite = np.isfinite(pooled)
    peak = np.where(finite, np.abs(pooled), -1.0).max(axis=1)
    return np.where(finite.any(axis=1), peak, np.nan)


def _xc_null(ctx: _Ctx, slices: list[tuple[int, int]], lag: int, b_sides: _SegmentCache | None = None):
    """Peak |r| of the Fisher-pooled lag profile for every draw. Time-shift draws pool the spec 7.9 identity's
    profiles (:func:`_xc_identity`); IAAFT draws, a declined identity and the reference mode take the direct null:
    ``lagged_pearson_batch`` per slice, ``fisher_pool`` over the slice axis per draw -- bit-identical to the per-draw
    loop. The peak is :func:`_peak_abs_r` (``xcorr_summary``'s max) either way."""
    sign = ctx.sa * ctx.sb
    width = 4 * max(hi - lo for lo, hi in slices)  # FFT work buffers ~ a few times the longest slice

    def run(draws: _SurrogateB) -> dict[str, np.ndarray]:
        if isinstance(draws, _ShiftedB) and not _reference_numerics():
            pooled_all = _xc_identity(ctx, slices, lag, draws, b_sides)
            if pooled_all is not None:
                return {"coord_xc_max_abs_r": _peak_abs_r(pooled_all)}
        out = np.empty(draws.n_draws)
        for d0, d1 in _draw_chunks(draws.n_draws, width):
            rr, nn = [], []
            for lo, hi in slices:
                r2, n2 = lagged_pearson_batch(ctx.va[lo:hi], draws.values(lo, hi, d0, d1), lag)
                rr.append(sign * r2)
                nn.append(n2)
            out[d0:d1] = _peak_abs_r(fisher_pool(np.ascontiguousarray(np.stack(rr, axis=1)), np.array(nn)))
        return {"coord_xc_max_abs_r": out}

    return run


def _vc_runs(ctx: _Ctx) -> list[tuple[int, int]]:
    """The slices vector coding reads -- every slice with a step (>= 2 samples) -- for its observed statistic
    (:func:`_vc_diffs`), its null (:func:`_vc_null`) and the B segments that null draws from (:func:`_surrogate`)."""
    return [(lo, hi) for lo, hi in ctx.runs if hi - lo >= 2]


def _vc_null(ctx: _Ctx, eps_a: float, eps_b: float):
    """Vector-coding in-phase and anti-phase fractions for every draw: A's steps once, B's steps per draw, the same
    elementwise stationary mask / coupling angle / class, and exact kept-step counts."""
    runs = _vc_runs(ctx)
    da = np.concatenate([ctx.sa * np.diff(ctx.va[lo:hi]) for lo, hi in runs]) if runs else np.empty(0)

    def run(draws: _SurrogateB) -> dict[str, np.ndarray]:
        in_phase = np.full(draws.n_draws, np.nan)
        anti_phase = np.full(draws.n_draws, np.nan)
        if not runs:
            return {"coord_vc_pct_in_phase": in_phase, "coord_vc_pct_anti_phase": anti_phase}
        for d0, d1 in _draw_chunks(draws.n_draws, da.size):
            db = np.ascontiguousarray(
                np.concatenate([ctx.sb * np.diff(draws.values(lo, hi, d0, d1), axis=1) for lo, hi in runs], axis=1)
            )
            da_rows = np.ascontiguousarray(np.broadcast_to(da, db.shape))
            keep = ~stationary_mask(da_rows, db, eps_a, eps_b)
            cls = classify(coupling_angle_deg(da_rows, db))
            n_keep = keep.sum(axis=1)
            for code, dest in ((0, in_phase), (1, anti_phase)):
                hits = ((cls == code) & keep).sum(axis=1)
                dest[d0:d1] = np.where(n_keep >= 3, hits / np.where(n_keep > 0, n_keep, 1), np.nan)
        return {"coord_vc_pct_in_phase": in_phase, "coord_vc_pct_anti_phase": anti_phase}

    return run


def _coh_null(ctx: _Ctx, slices: list[tuple[int, int]], nperseg: int, fs: float, params: CoordinationParams):
    """Band-mean coherence for every draw: A's auto-spectrum once per slice, B's auto- and cross-spectra batched over
    draws (``cross_spectra_batch``), then ``pooled_coherence`` per draw."""
    width = 4 * sum(hi - lo for lo, hi in slices)

    def run(draws: _SurrogateB) -> dict[str, np.ndarray]:
        out = np.empty(draws.n_draws)
        for d0, d1 in _draw_chunks(draws.n_draws, width):
            batches = [
                cross_spectra_batch(ctx.va[lo:hi], draws.values(lo, hi, d0, d1), fs, nperseg) for lo, hi in slices
            ]
            for i in range(d1 - d0):
                spectra = [batch.row(i) for batch in batches]
                out[d0 + i] = pooled_coherence(spectra, params.band_low_cpm, params.band_high_cpm)[0]
        return {"coord_coh_band_mean": out}

    return run


def _descriptor(spec: PairSpec, b: _Binding) -> tuple:
    return (
        spec.signal_a,
        spec.signal_b,
        str(_cid(b.team_a_id)),
        str(_cid(b.team_b_id)),
        str(_cid(b.player_a_id)),
        str(_cid(b.player_b_id)),
    )


def _blank_phase_row(ps, wrow, spec, b, phase_index, signals) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": ps.game_id,
        "period_id": int(ps.period_id),
        "window_kind": wrow["window_kind"],
        "window_id": wrow["window_id"],
        "level": spec.level,
        "signal_a": spec.signal_a,
        "signal_b": spec.signal_b,
        "axis": spec.axis,
        "team_a_id": _cid(b.team_a_id),
        "team_b_id": _cid(b.team_b_id),
        "player_a_id": _cid(b.player_a_id),
        "player_b_id": _cid(b.player_b_id),
        "phase_index": int(phase_index),
    }
    for c, dt in COORDINATION_PAIR_PHASE_COLUMNS.items():
        if c not in row:
            row[c] = pd.NA if dt in ("Int64", "object") else float("nan")
    row["coord_detection_source"] = signals.detection_source
    row["coord_stoppage_source"] = signals.stoppage.source
    return row


def _nanmean_or_nan(x: np.ndarray) -> float:
    """``nanmean`` without the empty-slice warning: NaN when nothing is finite (a window of only run-first samples)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _rp_stats_into(row: dict[str, Any], z: np.ndarray, ipa: np.ndarray, ipb: np.ndarray, near_deg: float) -> None:
    n = z.size
    mean_deg, r, sd = circular_summary(z.sum(), float(n))
    hist = np.bincount(hist_bin_index(z), minlength=12) / n
    row["coord_rp_mean_deg"] = float(mean_deg)
    row["coord_rp_circ_sd_deg"] = float(sd)
    row["coord_rp_resultant_length"] = float(r)
    row["coord_rp_pct_near_in_phase"] = float(np.mean(near_in_phase(z, near_deg)))
    for lab, frac in zip(HIST_BIN_LABELS, hist, strict=False):
        row[f"coord_rp_hist_bin_{lab}"] = float(frac)
    if ipa is not None:
        # nan-mean: each run's first sample is NaN in the advance indicator (A-45), excluded from num and denom.
        row["coord_rp_phase_valid_fraction_a"] = _nanmean_or_nan(ipa)
        row["coord_rp_phase_valid_fraction_b"] = _nanmean_or_nan(ipb)


def _window_n_phases(wrow: pd.Series) -> int | None:
    """The window's OWN subdivision count (spec 7.6, D3: "default 3 on possession windows"); NA = no subdivision
    (owner ruling 2026-10-04, review A-22 -- every window used to get ``params.n_phases``)."""
    n = wrow["n_phases"]
    return None if pd.isna(n) else int(n)


def _rp_pair_phase(ps, wrow, spec, b, ctx, s, e, fs, cos_thr, params, signals):
    """Per-phase RP rows for the window (C5): the window's own ``n_phases`` subdivisions, none when it is NA."""
    out = []
    m = e - s
    n_phases = _window_n_phases(wrow)
    if n_phases is None or m <= 0:
        return out
    phase_of = phase_assignment(m, n_phases)  # (m,), 1..n_phases over grid samples
    valid_mask = np.zeros(m, dtype=bool)
    valid_mask[ctx.idx - s] = True
    for k in range(1, n_phases + 1):
        sub = np.flatnonzero((phase_of == k) & valid_mask) + s
        row = _blank_phase_row(ps, wrow, spec, b, k, signals)
        row["coord_duration_s"] = len(sub) / fs
        row["coord_n_samples"] = len(sub)
        if len(sub) < 3:
            row["coord_rp_source"] = "too_short"
            out.append((row, "too_short"))
            continue
        z = (ctx.sa * ctx.sb) * ctx.pha[sub] * np.conj(ctx.phb[sub])
        _rp_stats_into(row, z, ctx.ipa[sub], ctx.ipb[sub], params.near_in_phase_deg)
        row["coord_rp_source"] = "scored"
        out.append((row, "scored"))
    return out


def compute_relative_phase(
    signals: CoordinationSignals,
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    pairs: Sequence[PairSpec] | None = None,
    dyad_windows: Literal["period", "all"] = "period",
) -> tuple[pd.DataFrame, pd.DataFrame, CoordinationReport]:
    """Relative-phase pair + per-phase tables + report (spec 7.8.1, C6, C24).

    Examples
    --------
    Score team-team relative phase from a built signals object (needs a real match)::

        pair, phase, report = compute_relative_phase(signals)
        pair["coord_rp_mean_deg"]   # circular-mean relative phase per (pair, window); ~0 = in-phase
    """
    params = signals.params
    fs = signals.fs
    cos_thr = float(np.cos(np.radians(params.near_in_phase_deg)))
    specs = [p for p in resolve_pairs(pairs, levels) if "relative_phase" in METHODS_BY_LEVEL[p.level]]
    acc = _Accum.new("coordination_pair")
    acc_ph = _Accum.new("coordination_pair_phase")
    rows: list[dict[str, Any]] = []
    prows: list[dict[str, Any]] = []
    iaaft_nc = 0
    b_sides = _SegmentCache(phasor_spectrum)
    for ps, wrow, s, e in _iter_windows(signals):
        wk = _wkey(ps, wrow)
        for spec in specs:
            if spec.level == "dyad" and not (dyad_windows == "all" or wrow["window_kind"] == "period"):
                continue
            binds = _bindings_for(ps, spec, wrow, dyad_goalkeeper=params.include_goalkeeper["dyad"])
            if isinstance(binds, str):
                blank_b = _Binding(ps.team_ids[0], ps.team_ids[1], pd.NA, pd.NA, False)
                row = _blank_pair_row(ps, wrow, spec, blank_b, signals)
                row["coord_rp_source"] = binds
                row["coord_rp_surrogate_source"] = "not_scored"
                rows.append(row)
                prows.extend(_phase_children(row, wrow, "coord_rp_source"))
                acc.note((wk, spec, None), binds)
                continue
            for b in binds:
                row = _blank_pair_row(ps, wrow, spec, b, signals)
                ctx = _context(ps, spec, b, s, e)
                row.update(_coverage(ctx, fs, s, e))
                pkey = (wk, spec, (str(_cid(b.team_a_id)), str(_cid(b.player_a_id))))
                reason = _pair_gate(ps, spec, b, row, params, "relative_phase")
                if reason is None and len(ctx.idx) < 3:
                    reason = "too_short"
                if reason is not None:
                    row["coord_rp_source"] = reason
                    row["coord_rp_surrogate_source"] = "not_scored"
                    rows.append(row)
                    prows.extend(_phase_children(row, wrow, "coord_rp_source"))
                    acc.note(pkey, reason)
                    continue
                idx = ctx.idx
                z = (ctx.sa * ctx.sb) * ctx.pha[idx] * np.conj(ctx.phb[idx])
                _rp_stats_into(row, z, ctx.ipa[idx], ctx.ipb[idx], params.near_in_phase_deg)
                row["coord_rp_source"] = "scored"
                obs = {
                    "coord_rp_resultant_length": row["coord_rp_resultant_length"],
                    "coord_rp_pct_near_in_phase": row["coord_rp_pct_near_in_phase"],
                }

                triples, src, _nc = _surrogate(
                    ps,
                    ctx,
                    spec.signal_b,
                    params,
                    fs,
                    _descriptor(spec, b),
                    "relative_phase",
                    _rp_null(ctx, cos_thr, b_sides),
                    obs,
                    ctx.runs,
                )
                iaaft_nc += _nc
                for metric, (sm, pc, ex) in triples.items():
                    row[f"{metric}_surrogate_mean"] = sm
                    row[f"{metric}_percentile"] = pc
                    row[f"{metric}_excess"] = ex
                row["coord_rp_surrogate_source"] = src
                rows.append(row)
                acc.note(pkey, "scored")
                acc.note_surrogate(src)
                for prow, _psrc in _rp_pair_phase(ps, wrow, spec, b, ctx, s, e, fs, cos_thr, params, signals):
                    prow["coord_detected_share"] = row["coord_detected_share"]
                    prows.append(prow)
    acc = _accum_from_rows("coordination_pair", rows, "coord_rp_source", "coord_rp_surrogate_source")
    acc_ph = _accum_from_rows("coordination_pair_phase", prows, "coord_rp_source", None)
    report = _report_from(signals, [acc, acc_ph], iaaft_nonconverged=iaaft_nc)
    object.__setattr__(report, "_family_accums", [acc, acc_ph])
    if (message := _coverage_message(report)) is not None:
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return _cast(rows, COORDINATION_PAIR_COLUMNS), _cast(prows, COORDINATION_PAIR_PHASE_COLUMNS), report


def _dropped_pair(ps, wrow, spec, signals, source_col, surr_col=None, source="no_possession_role"):
    blank_b = _Binding(ps.team_ids[0], ps.team_ids[1], pd.NA, pd.NA, False)
    row = _blank_pair_row(ps, wrow, spec, blank_b, signals)
    row[source_col] = source
    if surr_col is not None:
        row[surr_col] = "not_scored"
    return row


# --------------------------------------------------------------------------- cross-correlation
def compute_cross_correlation(
    signals: CoordinationSignals,
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    pairs: Sequence[PairSpec] | None = None,
    dyad_windows: Literal["period", "all"] = "period",
) -> tuple[pd.DataFrame, CoordinationReport]:
    """Lagged cross-correlation pair table + report (spec 7.8.2).

    Examples
    --------
    Lagged co-movement of the two teams' signals (needs a real match)::

        pair, report = compute_cross_correlation(signals)
        pair[["coord_xc_max_abs_r", "coord_xc_lag_s"]]   # peak |r| and its lag (+ = team A leads)
    """
    params = signals.params
    fs = signals.fs
    lag = round(params.xcorr_max_lag_s * fs)
    specs = [p for p in resolve_pairs(pairs, levels) if "cross_correlation" in METHODS_BY_LEVEL[p.level]]
    acc = _Accum.new("coordination_pair")
    rows: list[dict[str, Any]] = []
    iaaft_nc = 0
    b_sides = _SegmentCache(lagged_pearson_b_side)
    for ps, wrow, s, e in _iter_windows(signals):
        wk = _wkey(ps, wrow)
        for spec in specs:
            if spec.level == "dyad" and not (dyad_windows == "all" or wrow["window_kind"] == "period"):
                continue
            binds = _bindings_for(ps, spec, wrow, dyad_goalkeeper=params.include_goalkeeper["dyad"])
            if isinstance(binds, str):
                rows.append(
                    _dropped_pair(ps, wrow, spec, signals, "coord_xc_source", "coord_xc_surrogate_source", binds)
                )
                acc.note((wk, spec, None), binds)
                continue
            for b in binds:
                row = _blank_pair_row(ps, wrow, spec, b, signals)
                ctx = _context(ps, spec, b, s, e)
                row.update(_coverage(ctx, fs, s, e))
                pkey = (wk, spec, (str(_cid(b.team_a_id)), str(_cid(b.player_a_id))))
                reason = _pair_gate(ps, spec, b, row, params, "cross_correlation")
                if reason is not None:
                    row["coord_xc_source"] = reason
                    row["coord_xc_surrogate_source"] = "not_scored"
                    rows.append(row)
                    acc.note(pkey, reason)
                    continue
                slices = [(lo, hi) for lo, hi in ctx.runs if hi - lo >= min_slice_samples(lag)]
                if not slices:
                    row["coord_xc_source"] = "too_short"
                    row["coord_xc_surrogate_source"] = "not_scored"
                    rows.append(row)
                    acc.note(pkey, "too_short")
                    continue
                pooled = _xcorr_pooled(ctx, slices, lag)
                if np.isnan(pooled).all():
                    row["coord_xc_source"] = "degenerate_constant"
                    row["coord_xc_surrogate_source"] = "not_scored"
                    rows.append(row)
                    acc.note(pkey, "degenerate_constant")
                    continue
                max_abs_r, lag_s, r_at_max, r_lag0 = xcorr_summary(pooled, fs)
                row["coord_xc_max_abs_r"] = max_abs_r
                row["coord_xc_lag_s"] = lag_s
                row["coord_xc_r_at_max"] = r_at_max
                row["coord_xc_r_lag0"] = r_lag0
                row["coord_xc_source"] = "scored"

                triples, src, _nc = _surrogate(
                    ps,
                    ctx,
                    spec.signal_b,
                    params,
                    fs,
                    _descriptor(spec, b),
                    "cross_correlation",
                    _xc_null(ctx, slices, lag, b_sides),
                    {"coord_xc_max_abs_r": max_abs_r},
                    slices,
                )
                iaaft_nc += _nc
                sm, pc, ex = triples["coord_xc_max_abs_r"]
                row["coord_xc_max_abs_r_surrogate_mean"] = sm
                row["coord_xc_max_abs_r_percentile"] = pc
                row["coord_xc_max_abs_r_excess"] = ex
                row["coord_xc_surrogate_source"] = src
                rows.append(row)
    acc = _accum_from_rows("coordination_pair", rows, "coord_xc_source", "coord_xc_surrogate_source")
    report = _report_from(signals, [acc], iaaft_nonconverged=iaaft_nc)
    object.__setattr__(report, "_family_accums", [acc])
    if (message := _coverage_message(report)) is not None:
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return _cast(rows, COORDINATION_PAIR_COLUMNS), report


def _xcorr_pooled(ctx: _Ctx, slices: list[tuple[int, int]], lag: int) -> np.ndarray:
    rr = []
    nn = []
    for lo, hi in slices:
        r, n = lagged_pearson(ctx.va[lo:hi], ctx.vb[lo:hi], lag)
        rr.append((ctx.sa * ctx.sb) * r)
        nn.append(n)
    return fisher_pool(np.array(rr), np.array(nn))


# --------------------------------------------------------------------------- vector coding
def _vc_diffs(ctx: _Ctx) -> tuple[np.ndarray, np.ndarray]:
    das, dbs = [], []
    for lo, hi in _vc_runs(ctx):
        das.append(ctx.sa * np.diff(ctx.va[lo:hi]))
        dbs.append(ctx.sb * np.diff(ctx.vb[lo:hi]))
    if not das:
        return np.empty(0), np.empty(0)
    return np.concatenate(das), np.concatenate(dbs)


def _vc_stats_into(row: dict[str, Any], da: np.ndarray, db: np.ndarray, eps_a: float, eps_b: float) -> str:
    stat = stationary_mask(da, db, eps_a, eps_b)
    row["coord_vc_n_stationary"] = int(stat.sum())
    keep = ~stat
    if int(keep.sum()) < 3:
        return "too_short"
    ang = coupling_angle_deg(da[keep], db[keep])
    cls = classify(ang)
    n = int(keep.sum())
    row["coord_vc_pct_in_phase"] = float(np.mean(cls == 0))
    row["coord_vc_pct_anti_phase"] = float(np.mean(cls == 1))
    row["coord_vc_pct_a_phase"] = float(np.mean(cls == 2))
    row["coord_vc_pct_b_phase"] = float(np.mean(cls == 3))
    mean_deg, _r, sd = circular_summary(np.exp(1j * np.radians(ang)).sum(), float(n))
    row["coord_vc_mean_angle_deg"] = float(mean_deg) % 360.0
    row["coord_vc_angle_variability_deg"] = float(sd)
    return "scored"


def compute_vector_coding(
    signals: CoordinationSignals,
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    pairs: Sequence[PairSpec] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, CoordinationReport]:
    """Vector-coding pair + per-phase tables + report (spec 7.8.3).

    Examples
    --------
    Coupling-angle pattern fractions between the two teams (needs a real match)::

        pair, phase, report = compute_vector_coding(signals)
        pair[["coord_vc_pct_in_phase", "coord_vc_pct_anti_phase"]]   # the four fractions sum to 1
    """
    params = signals.params
    fs = signals.fs
    specs = [p for p in resolve_pairs(pairs, levels) if "vector_coding" in METHODS_BY_LEVEL[p.level]]
    acc = _Accum.new("coordination_pair")
    acc_ph = _Accum.new("coordination_pair_phase")
    rows: list[dict[str, Any]] = []
    prows: list[dict[str, Any]] = []
    iaaft_nc = 0
    n_stationary = 0  # A-19: consecutive-diff samples omitted as stationary, summed over every scored-diff pair
    for ps, wrow, s, e in _iter_windows(signals):
        wk = _wkey(ps, wrow)
        for spec in specs:
            binds = _bindings_for(ps, spec, wrow, dyad_goalkeeper=params.include_goalkeeper["dyad"])
            if isinstance(binds, str):
                row = _dropped_pair(ps, wrow, spec, signals, "coord_vc_source", "coord_vc_surrogate_source", binds)
                rows.append(row)
                prows.extend(_phase_children(row, wrow, "coord_vc_source"))
                acc.note((wk, spec, None), binds)
                continue
            for b in binds:
                row = _blank_pair_row(ps, wrow, spec, b, signals)
                ctx = _context(ps, spec, b, s, e)
                row.update(_coverage(ctx, fs, s, e))
                pkey = (wk, spec, (str(_cid(b.team_a_id)), str(_cid(b.player_a_id))))
                reason = _pair_gate(ps, spec, b, row, params, "vector_coding")  # Task 14: before not_commensurate
                if reason is None and not spec.commensurate:
                    reason = "not_commensurate"
                if reason is not None:
                    row["coord_vc_source"] = reason
                    row["coord_vc_surrogate_source"] = "not_scored"
                    rows.append(row)
                    prows.extend(_phase_children(row, wrow, "coord_vc_source"))
                    acc.note(pkey, reason)
                    continue
                da, db = _vc_diffs(ctx)
                eps_a = params.vc_epsilon[spec.signal_a]
                eps_b = params.vc_epsilon[spec.signal_b]
                if da.size:
                    n_stationary += int(stationary_mask(da, db, eps_a, eps_b).sum())  # A-19 diagnostic
                status = _vc_stats_into(row, da, db, eps_a, eps_b) if da.size else "too_short"
                row["coord_vc_source"] = status
                row["coord_vc_surrogate_source"] = "not_scored"
                if status == "scored":
                    obs = {
                        "coord_vc_pct_in_phase": row["coord_vc_pct_in_phase"],
                        "coord_vc_pct_anti_phase": row["coord_vc_pct_anti_phase"],
                    }

                    triples, src, _nc = _surrogate(
                        ps,
                        ctx,
                        spec.signal_b,
                        params,
                        fs,
                        _descriptor(spec, b),
                        "vector_coding",
                        _vc_null(ctx, eps_a, eps_b),
                        obs,
                        _vc_runs(ctx),
                    )
                    iaaft_nc += _nc
                    for metric, (sm, pc, ex) in triples.items():
                        row[f"{metric}_surrogate_mean"] = sm
                        row[f"{metric}_percentile"] = pc
                        row[f"{metric}_excess"] = ex
                    row["coord_vc_surrogate_source"] = src
                    acc.note_surrogate(src)
                rows.append(row)
                acc.note(pkey, status)
                if status != "scored":  # a degraded parent: its phase rows carry its token (ADR-042)
                    prows.extend(_phase_children(row, wrow, "coord_vc_source"))
                    continue
                for prow, _psrc in _vc_pair_phase(ps, wrow, spec, b, ctx, s, e, fs, eps_a, eps_b, params, signals):
                    prow["coord_detected_share"] = row["coord_detected_share"]
                    prows.append(prow)
    acc = _accum_from_rows("coordination_pair", rows, "coord_vc_source", "coord_vc_surrogate_source")
    acc.samples_stationary = n_stationary  # A-19
    acc_ph = _accum_from_rows("coordination_pair_phase", prows, "coord_vc_source", None)
    report = _report_from(signals, [acc, acc_ph], iaaft_nonconverged=iaaft_nc)
    object.__setattr__(report, "_family_accums", [acc, acc_ph])
    if (message := _coverage_message(report)) is not None:
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return _cast(rows, COORDINATION_PAIR_COLUMNS), _cast(prows, COORDINATION_PAIR_PHASE_COLUMNS), report


def _vc_pair_phase(ps, wrow, spec, b, ctx, s, e, fs, eps_a, eps_b, params, signals):
    """Per-phase VC rows for the window (C5): the window's own ``n_phases`` subdivisions, none when it is NA."""
    out = []
    m = e - s
    n_phases = _window_n_phases(wrow)
    if n_phases is None or m <= 0:
        return out
    phase_of = phase_assignment(m, n_phases)
    for k in range(1, n_phases + 1):
        row = _blank_phase_row(ps, wrow, spec, b, k, signals)
        # diffs restricted to this phase's grid samples within each run
        das, dbs = [], []
        for lo, hi in ctx.runs:
            local = np.arange(lo, hi)
            ph = phase_of[local - s]
            mask = ph == k
            if mask.sum() >= 2:
                das.append(ctx.sa * np.diff(ctx.va[local[mask]]))
                dbs.append(ctx.sb * np.diff(ctx.vb[local[mask]]))
        if not das:
            row["coord_vc_source"] = "too_short"
            out.append((row, "too_short"))
            continue
        da = np.concatenate(das)
        db = np.concatenate(dbs)
        status = _vc_stats_into(row, da, db, eps_a, eps_b)
        row["coord_vc_source"] = status
        out.append((row, status))
    return out


# --------------------------------------------------------------------------- coherence
def compute_coherence(
    signals: CoordinationSignals,
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    pairs: Sequence[PairSpec] | None = None,
) -> tuple[pd.DataFrame, CoordinationReport]:
    """Pooled-Welch coherence pair table + report (spec 7.8.5).

    Examples
    --------
    Frequency-domain coupling over the analysis band (needs a long, contiguous match)::

        pair, report = compute_coherence(signals)
        pair["coord_coh_band_mean"]   # mean magnitude-squared coherence in [0, 1] over the band
    """
    params = signals.params
    fs = signals.fs
    nperseg = round(params.welch_segment_s * fs)
    specs = [p for p in resolve_pairs(pairs, levels) if "coherence" in METHODS_BY_LEVEL[p.level]]
    acc = _Accum.new("coordination_pair")
    rows: list[dict[str, Any]] = []
    iaaft_nc = 0
    for ps, wrow, s, e in _iter_windows(signals):
        wk = _wkey(ps, wrow)
        for spec in specs:
            binds = _bindings_for(ps, spec, wrow, dyad_goalkeeper=params.include_goalkeeper["dyad"])
            if isinstance(binds, str):
                rows.append(
                    _dropped_pair(ps, wrow, spec, signals, "coord_coh_source", "coord_coh_surrogate_source", binds)
                )
                acc.note((wk, spec, None), binds)
                continue
            for b in binds:
                row = _blank_pair_row(ps, wrow, spec, b, signals)
                ctx = _context(ps, spec, b, s, e)
                row.update(_coverage(ctx, fs, s, e))
                pkey = (wk, spec, (str(_cid(b.team_a_id)), str(_cid(b.player_a_id))))
                reason = _pair_gate(ps, spec, b, row, params, "coherence")
                if reason is not None:
                    row["coord_coh_source"] = reason
                    row["coord_coh_surrogate_source"] = "not_scored"
                    rows.append(row)
                    acc.note(pkey, reason)
                    continue
                slices = [(lo, hi) for lo, hi in ctx.runs if hi - lo >= nperseg]
                spectra = [welch_spectra(ctx.va[lo:hi], ctx.vb[lo:hi], fs, nperseg) for lo, hi in slices]
                if not spectra:
                    row["coord_coh_source"] = "too_short"
                    row["coord_coh_surrogate_source"] = "not_scored"
                    rows.append(row)
                    acc.note(pkey, "too_short")
                    continue
                band_mean, peak, k = pooled_coherence(spectra, params.band_low_cpm, params.band_high_cpm)
                f_cpm = 60.0 * spectra[0].f
                in_band = bool(((f_cpm >= params.band_low_cpm) & (f_cpm <= params.band_high_cpm)).any())
                # A-20: no bin in the band = the segment is too short to resolve it; a band of zero power has no
                # coherence -- either way a NaN carries its token, never "scored"
                bad = (
                    "too_short"
                    if (k < 4 or not in_band)
                    else (None if np.isfinite(band_mean) else "degenerate_constant")
                )
                if bad is not None:
                    row["coord_coh_source"] = bad
                    row["coord_coh_surrogate_source"] = "not_scored"
                    rows.append(row)
                    acc.note(pkey, bad)
                    continue
                row["coord_coh_band_mean"] = band_mean
                row["coord_coh_peak_freq_cpm"] = peak
                row["coord_coh_n_segments"] = int(k)
                row["coord_coh_source"] = "scored"

                triples, src, _nc = _surrogate(
                    ps,
                    ctx,
                    spec.signal_b,
                    params,
                    fs,
                    _descriptor(spec, b),
                    "coherence",
                    _coh_null(ctx, slices, nperseg, fs, params),
                    {"coord_coh_band_mean": band_mean},
                    slices,
                )
                iaaft_nc += _nc
                sm, pc, ex = triples["coord_coh_band_mean"]
                row["coord_coh_band_mean_surrogate_mean"] = sm
                row["coord_coh_band_mean_percentile"] = pc
                row["coord_coh_band_mean_excess"] = ex
                row["coord_coh_surrogate_source"] = src
                rows.append(row)
    acc = _accum_from_rows("coordination_pair", rows, "coord_coh_source", "coord_coh_surrogate_source")
    report = _report_from(signals, [acc], iaaft_nonconverged=iaaft_nc)
    object.__setattr__(report, "_family_accums", [acc])
    if (message := _coverage_message(report)) is not None:
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return _cast(rows, COORDINATION_PAIR_COLUMNS), report


def _segments_in(segments: np.ndarray, s: int, e: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in segments:
        a, c = max(int(lo), s), min(int(hi), e)
        if c > a:
            out.append((a, c))
    return out


# --------------------------------------------------------------------------- spectral
def compute_spectral(signals: CoordinationSignals) -> tuple[pd.DataFrame, CoordinationReport]:
    """Median-frequency spectral table: one row per (team, signal, window) + a possession row (spec 7.8.4).

    Examples
    --------
    Dominant oscillation rate of each team signal (needs a long, contiguous match)::

        spectral, report = compute_spectral(signals)
        spectral["coord_median_freq_cpm"]   # spectral median frequency in cycles per minute
    """
    from silly_kicks.coordination._kernels._spectral import min_spectral_samples

    params = signals.params
    fs = signals.fs
    min_n = min_spectral_samples(fs, params.band_low_cpm)
    rows: list[dict[str, Any]] = []
    threshold = params.min_observed_fraction["spectral"]
    for ps, wrow, s, e in _iter_windows(signals):
        for tm in ps.team_ids:
            segs = _segments_in(ps.segments[tm], s, e)
            share = _team_det(ps, tm).share(s, e)  # the team side's raw-detection share (spec 7.11, A-08)
            for sig in TEAM_SIGNALS:
                if _sig_goal_unresolved(ps, sig, tm):  # Task 14 precedence: goal_end_unresolved before detection (A-24)
                    row = _spectral_base(ps, wrow, tm, sig, signals)
                    row["coord_spectral_source"] = "goal_end_unresolved"
                elif insufficient_detection(share, threshold):
                    row = _spectral_base(ps, wrow, tm, sig, signals)
                    row["coord_spectral_source"] = "insufficient_detection"
                else:
                    values = ps.team_signal[(tm, sig)]
                    mfs, durs = [], []
                    for lo, hi in segs:
                        if hi - lo >= min_n:
                            mfs.append(median_frequency_cpm(values[lo:hi], fs))
                            durs.append((hi - lo) / fs)
                    row = _spectral_row(ps, wrow, tm, sig, mfs, durs, signals)
                row["coord_detected_share"] = share
                rows.append(row)
        # the possession row has no per-side detection gate and carries no share (owner ruling 2026-10-04: D1 reports
        # its occlusion error as the evidence)
        rows.append(_possession_spectral_row(ps, wrow, s, e, fs, min_n, signals))
    acc = _accum_from_rows("coordination_spectral", rows, "coord_spectral_source", None)
    report = _report_from(signals, [acc])
    object.__setattr__(report, "_family_accums", [acc])  # the orchestrator's merged report counts these rows too
    if (message := _coverage_message(report)) is not None:
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return _cast(rows, COORDINATION_SPECTRAL_COLUMNS), report


def _spectral_base(ps, wrow, team_id, signal, signals) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": ps.game_id,
        "period_id": int(ps.period_id),
        "window_kind": wrow["window_kind"],
        "window_id": wrow["window_id"],
        "team_id": _cid(team_id) if not pd.isna(team_id) else pd.NA,
        "signal": signal,
    }
    for c, dt in COORDINATION_SPECTRAL_COLUMNS.items():
        if c not in row:
            row[c] = pd.NA if dt in ("Int64", "object") else float("nan")
    row["coord_detection_source"] = signals.detection_source
    row["coord_stoppage_source"] = signals.stoppage.source
    return row


def _pool_spectral_slices(row: dict[str, Any], mfs: list[float], durs: list[float]) -> dict[str, Any]:
    """Pool the slices' median frequencies into ``row`` (spec 7.8.4): ``too_short`` with no slice, and (A-20)
    ``degenerate_constant`` when every slice is constant (zero power, no median) -- never ``scored`` with a NaN.
    The coverage counts the slices that went into the value."""
    if not mfs:
        row["coord_duration_s"] = 0.0
        row["coord_n_segments"] = 0
        row["coord_spectral_source"] = "too_short"
        return row
    m, d = np.asarray(mfs, dtype=np.float64), np.asarray(durs, dtype=np.float64)
    live = np.isfinite(m)
    if not live.any():
        row["coord_duration_s"] = float(d.sum())
        row["coord_n_segments"] = len(mfs)
        row["coord_spectral_source"] = "degenerate_constant"
        return row
    row["coord_median_freq_cpm"] = pooled_median_frequency(m, d)
    row["coord_duration_s"] = float(d[live].sum())
    row["coord_n_segments"] = int(live.sum())
    row["coord_spectral_source"] = "scored"
    return row


def _spectral_row(ps, wrow, tm, sig, mfs, durs, signals) -> dict[str, Any]:
    return _pool_spectral_slices(_spectral_base(ps, wrow, tm, sig, signals), mfs, durs)


def _possession_spectral_row(ps, wrow, s, e, fs, min_n, signals) -> dict[str, Any]:
    """The possession series' spectrum (1 = team A in possession, 0 = team B; Moura 2013 Table II), sliced like every
    team signal (spec 7.8.4): per window AND both teams' segments, then per run of samples with a known possession
    (NA before a period's first possession). A long stoppage therefore splits it, although the series holds."""
    row = _spectral_base(ps, wrow, pd.NA, "possession", signals)
    poss = ps.possession_team[s:e]
    ta, tb = ps.team_ids
    numeric = np.full(e - s, np.nan)
    for i, v in enumerate(poss):
        if not pd.isna(v):
            numeric[i] = 1.0 if same_id(v, ta) else 0.0
    if not np.isfinite(numeric).any():
        row["coord_duration_s"] = 0.0
        row["coord_n_segments"] = 0
        row["coord_spectral_source"] = "no_possession_role"
        return row
    mfs, durs = [], []
    for lo, hi in _both_runs(ps.segments[ta], ps.segments[tb], s, e):
        finite = np.isfinite(numeric[lo - s : hi - s])
        d = np.diff(np.concatenate(([0], finite.astype(np.int8), [0])))
        for a, c in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
            if c - a >= min_n:
                mfs.append(median_frequency_cpm(numeric[lo - s + a : lo - s + c], fs))
                durs.append((c - a) / fs)
    return _pool_spectral_slices(row, mfs, durs)


# --------------------------------------------------------------------------- cluster phase + team sync
def _cluster_roster(ps: PeriodSignals, tm: object, params: CoordinationParams) -> list[tuple[object, PlayerSeries]]:
    """The team's cluster players (goalkeeper per ``include_goalkeeper["cluster"]``) in column order."""
    include_gk = params.include_goalkeeper["cluster"]
    players = [
        (pid, pl) for (t2, pid), pl in ps.players.items() if same_id(t2, tm) and (include_gk or not pl.is_goalkeeper)
    ]
    players.sort(key=lambda kp: str(canonical_id(kp[0])))
    return players


def _cluster_inputs(ps: PeriodSignals, tm: object, axis: str, params: CoordinationParams):
    players = _cluster_roster(ps, tm, params)
    grid_n = ps.t.size
    k = len(players)
    z = np.full((grid_n, k), np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.zeros((grid_n, k), dtype=bool)
    on_pitch = np.zeros((grid_n, k), dtype=bool)
    pids = []
    for j, (pid, pl) in enumerate(players):
        ph = pl.phasor_x if axis == "x" else pl.phasor_y
        z[:, j] = ph
        valid[:, j] = np.isfinite(ph)
        on_pitch[:, j] = pl.on_pitch
        pids.append(pid)
    return z, valid, pids, on_pitch


def _read_only(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class _ClusterKernels:
    """The cluster-family arithmetic of one scoring call: ADR-111 D4's (production) or the as-built (reference)."""

    cluster_phase: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
    window_cluster_stats: Callable[[np.ndarray, np.ndarray, np.ndarray, int, int], ClusterWindowStats]
    shifted_rho_group_means: Callable[..., np.ndarray]
    pearson_rows: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def _reference_rho_null(z, valid, usable, start, end, runs, n_draws, *, parts=None) -> np.ndarray:
    """The as-built null behind the production call shape: it takes the complex ``z`` and splits nothing."""
    del parts
    return _cluster_reference.shifted_rho_group_means(z, valid, usable, start, end, runs, n_draws)


def _cluster_kernels() -> _ClusterKernels:
    """This call's cluster kernels -- the as-built ones under ``REFERENCE_NUMERICS_ENV``. The production names are
    resolved at call time, so a structural guard that wraps them (``test_scale_guards``) still sees every call."""
    if _reference_numerics():
        ref = _cluster_reference
        return _ClusterKernels(ref.cluster_phase, ref.window_cluster_stats, _reference_rho_null, ref.pearson_rows)
    return _ClusterKernels(cluster_phase, window_cluster_stats, shifted_rho_group_means, pearson_rows)


@dataclass(frozen=True)
class _ClusterCtx:
    """One (period, team, axis) cluster-phase context, built ONCE per period and shared by every window.

    The arrays are read-only (a stray in-place write fails loud instead of corrupting later windows). ``runs[j]``
    are player ``j``'s phase runs -- the continuous stretches its phasor was computed on (D15), i.e. the surrogate
    shift unit of plan R1. ``run_shifts`` memoises each run's draws: they are keyed by (game, period, run start,
    team, axis, player) -- never by the window -- so every window touching a run reuses them. The memo is bounded
    by this team-period's run count and dies with the context.
    """

    z: np.ndarray  # (T, K) player phasors
    parts: tuple[np.ndarray, np.ndarray]  # (real, imag) of z, split ONCE for the numba null (never per window)
    valid: np.ndarray  # (T, K) finite-phasor mask
    usable: np.ndarray  # (T,) at least min_players valid
    rel: np.ndarray  # (T, K) relative phasors z * conj(q)
    pids: list
    runs: list[list[tuple[int, int]]]
    on_pitch: np.ndarray
    amp: np.ndarray  # segment-level rho_group,i (C20; the team-sync input)
    detections: list[DetectionCounts]  # each roster player's own detection counts (A-08: min(team, player))
    run_shifts: dict[tuple[int, int, int], np.ndarray | None] = field(default_factory=dict)


def _phase_runs(runs: np.ndarray, valid_col: np.ndarray) -> list[tuple[int, int]]:
    """The player's runs whose phase was computed (``_phasors_over_runs`` fills a run whole or not at all)."""
    return [(int(lo), int(hi)) for lo, hi in runs if int(hi) > int(lo) and valid_col[int(lo) : int(hi)].all()]


def _cluster_ctx(
    ps: PeriodSignals, tm: object, axis: str, params: CoordinationParams, kern: _ClusterKernels
) -> _ClusterCtx | None:
    z, valid, pids, on_pitch = _cluster_inputs(ps, tm, axis, params)
    if z.shape[1] == 0:
        return None
    _q, rel, usable = kern.cluster_phase(z, valid, params.min_players)
    amp = _amplitude_series(rel, valid, usable, ps.segments[tm], kern.window_cluster_stats)
    roster = _cluster_roster(ps, tm, params)
    runs = [_phase_runs(pl.runs, valid[:, j]) for j, (_pid, pl) in enumerate(roster)]
    z_re, z_im = phasor_parts(z)
    return _ClusterCtx(
        z=_read_only(z),
        parts=(_read_only(z_re), _read_only(z_im)),
        valid=_read_only(valid),
        usable=_read_only(usable),
        rel=_read_only(rel),
        pids=pids,
        runs=runs,
        on_pitch=_read_only(on_pitch),
        amp=_read_only(amp),
        detections=[pl.detection for _pid, pl in roster],
    )


def _run_shift_draws(
    ps: PeriodSignals,
    tm: object,
    axis: str,
    c: _ClusterCtx,
    player: int,
    lo: int,
    hi: int,
    tau: int,
    params: CoordinationParams,
) -> np.ndarray | None:
    """Player ``player``'s shift draws for its run ``[lo, hi)``, ``(n_surrogates,)``; ``None`` if the run is too
    short to shift by at least ``tau`` (``shift_bounds``)."""
    key = (player, lo, hi)
    if key not in c.run_shifts:
        rng = surrogate_rng(
            params.surrogate_seed,
            (
                _seed_game(ps),
                int(ps.period_id),
                lo,
                str(canonical_id(tm)),
                axis,
                str(canonical_id(c.pids[player])),
                "cluster",
            ),
        )
        sh = draw_shifts(rng, hi - lo, tau, params.n_surrogates)
        c.run_shifts[key] = None if sh is None else _read_only(sh)
    return c.run_shifts[key]


def _amplitude_series(
    rel: np.ndarray,
    valid: np.ndarray,
    usable: np.ndarray,
    segments: np.ndarray,
    stats: Callable[[np.ndarray, np.ndarray, np.ndarray, int, int], ClusterWindowStats],
) -> np.ndarray:
    """The team's instantaneous group synchrony per usable sample, segment by segment (C20; the team-sync input),
    scored by ``stats`` -- the call's ``window_cluster_stats``."""
    amp = np.full(usable.size, np.nan)
    for lo, hi in segments:
        st = stats(rel, valid, usable, int(lo), int(hi))
        u_idx = np.flatnonzero(usable[int(lo) : int(hi)]) + int(lo)
        if u_idx.size == st.rho_group_i.size and u_idx.size:
            amp[u_idx] = st.rho_group_i
    return amp


def _cluster_base(ps, wrow, tm, axis, columns, signals) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": ps.game_id,
        "period_id": int(ps.period_id),
        "window_kind": wrow["window_kind"],
        "window_id": wrow["window_id"],
        "team_id": _cid(tm) if not pd.isna(tm) else pd.NA,
        "axis": axis,
    }
    for c, dt in columns.items():
        if c not in row:
            row[c] = pd.NA if dt in ("Int64", "object") else float("nan")
    row["coord_detection_source"] = signals.detection_source
    row["coord_stoppage_source"] = signals.stoppage.source
    return row


def _split_collapsed(values: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Split a COLLAPSED series (one value per True of ``mask``) back into its contiguous runs -- the gaps the mask
    skipped. ``len(values) == mask.sum()``; SampEn templates then never span a gap (review A-25)."""
    m = np.asarray(mask, dtype=bool)
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    lengths = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()
    return list(np.split(np.asarray(values), np.cumsum(lengths)[:-1])) if lengths else []


def _sampen_runs_token(runs: list[np.ndarray], params) -> tuple[float, str]:
    val, a, b = sampen_over_runs(runs, m=params.sampen_m, r_sd=params.sampen_r_sd)
    return val, "entropy_undefined" if (a == 0 or b == 0) else "scored"


def _unwrap_phase_runs(angle_runs: list[np.ndarray]) -> list[np.ndarray]:
    """Unwrap each phase run to a continuous series (review A-25), reset per run, consistent with the instantaneous-
    frequency unwrap in ``_phasors_over_runs``. Precondition (checked, not assumed): the wrapped consecutive step
    stays below pi -- a step at/over pi means the analysis rate is too low to unwrap reliably, so fail loud with the
    offending step rather than emit a silently-wrong SampEn."""
    out: list[np.ndarray] = []
    for a in angle_runs:
        if a.size >= 2:
            step = float(np.abs((np.diff(a) + np.pi) % (2.0 * np.pi) - np.pi).max())
            if step >= np.pi - 1e-9:
                raise ValueError(
                    f"phi_k phase step {step:.6f} rad >= pi within a run: the analysis rate is too low to unwrap "
                    f"this relative phase reliably (review A-25)"
                )
        out.append(np.unwrap(a) if a.size else a)
    return out


def _cluster_player_row(ps, wrow, tm, axis, pid, signals) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": ps.game_id,
        "period_id": int(ps.period_id),
        "window_kind": wrow["window_kind"],
        "window_id": wrow["window_id"],
        "team_id": _cid(tm),
        "player_id": _cid(pid),
        "axis": axis,
    }
    for c, dt in COORDINATION_CLUSTER_PLAYER_COLUMNS.items():
        if c not in row:
            row[c] = pd.NA if dt in ("Int64", "object") else float("nan")
    row["coord_detection_source"] = signals.detection_source
    row["coord_stoppage_source"] = signals.stoppage.source
    return row


def _cluster_surrogate(ps, tm, axis, c: _ClusterCtx, s, e, params, fs, obs_mean: float, kern: _ClusterKernels):
    """The window's rho_group_mean surrogate triple (spec 7.9): each player's phasor circularly shifted within each
    of its own phase runs (plan R1's shift unit), independently per player and run, scored by the call's
    ``shifted_rho_group_means`` (the observed estimator per draw).

    ``segment_too_short`` when a CONTRIBUTING run -- one holding a usable ``rho_group`` sample the statistic reads -- is
    shorter than ``2 * tau + 1`` samples (R1; B m3, Task-17 rule 3). A run that merely TOUCHES the window but holds no
    usable sample (e.g. the team is below ``min_players`` there) takes no part and is skipped, exactly as the pair
    families do via ``_segments_holding``.
    """
    if params.n_surrogates == 0:
        return (float("nan"), float("nan"), float("nan")), "disabled"
    _require_surrogate_method("cluster", params.surrogate_method)  # A-17: refuse iaaft here, never silently time-shift
    tau = round(params.min_shift_s["player_x" if axis == "x" else "player_y"] * fs)
    usable_win = c.usable[s:e]
    runs: list[ShiftRun] = []
    for player, player_runs in enumerate(c.runs):
        pr = np.asarray(player_runs, dtype=np.int64).reshape(-1, 2)
        if not len(pr):
            continue
        # the samples this player CONTRIBUTES to the window's rho_group: usable AND its own phasor valid, in [s, e)
        contrib = np.flatnonzero(usable_win & c.valid[s:e, player]) + s
        if not contrib.size:
            continue  # touches the window but reads no usable sample -> takes no part (B m3)
        for lo, hi in _segments_holding(pr, contrib):  # only this player's runs holding a contributing sample
            sh = _run_shift_draws(ps, tm, axis, c, player, lo, hi, tau, params)
            if sh is None:
                return (float("nan"), float("nan"), float("nan")), "segment_too_short"
            runs.append(ShiftRun(player, lo, hi, sh))
    if not runs:  # a usable sample is a valid phasor, and every valid phasor lies in one of its player's runs
        raise RuntimeError(f"cluster window [{s}, {e}) has usable samples but no contributing player run (a bug)")
    dist = kern.shifted_rho_group_means(c.z, c.valid, c.usable, s, e, runs, params.n_surrogates, parts=c.parts)
    return surrogate_triple(obs_mean, dist), "computed"


#: Fewest usable samples a window needs for the cluster family (spec 7.8.6, owner ruling 2026-10-02). With ONE usable
#: sample each player's window-mean relative phasor IS its one relative phasor, so rho_group is exactly 1 for the data
#: and for every shift draw: no information, and a percentile decided by float rounding (the 12 no-flip crossings on
#: 0.1 s IDSSE possession windows). Such a window is ``too_short``, like the other families' minimum-length rules.
MIN_CLUSTER_SAMPLES = 2


def _cluster_child(ps, wrow, tm, axis, pid, reason: str, share: float, signals) -> dict[str, Any]:
    """A degraded team window's player row (spec 7.13 / ADR-042; owner ruling 2026-10-04): real keys, the player's
    own detection share, NaN metrics, the PARENT's token in both source columns."""
    prow = _cluster_player_row(ps, wrow, tm, axis, pid, signals)
    prow["coord_detected_share"] = share
    prow["coord_cluster_player_source"] = reason
    prow["coord_cluster_player_sampen_source"] = reason
    return prow


def _cluster_team_and_players(
    ps, wrow, tm, axis, c: _ClusterCtx | None, s, e, fs, params, signals, kern: _ClusterKernels
):
    """The cluster-team row and its roster's player rows for one window (spec 7.8.6).

    Tokens in the Task 14 precedence: ``insufficient_detection`` (the team's raw-detection share below the cluster
    threshold, A-08), ``insufficient_players`` (no sample with ``min_players`` valid phases), ``too_short`` (fewer than
    ``MIN_CLUSTER_SAMPLES`` usable samples). A degraded window still emits one tokenised row per roster member
    (reason-independent). A player row tests the LOWER of its team's and its own share, and is ``too_short`` below
    ``MIN_CLUSTER_SAMPLES`` usable samples of its own (A-21). SampEn's token lives in its own column (A-20).
    """
    t_row = _cluster_base(ps, wrow, tm, axis, COORDINATION_CLUSTER_TEAM_COLUMNS, signals)
    team_det = _team_det(ps, tm)
    team_share = team_det.share(s, e)
    t_row["coord_observed_fraction"] = team_share
    t_row["coord_detected_share"] = team_share
    threshold = params.min_observed_fraction["cluster"]
    pids = c.pids if c is not None else []
    dets = c.detections if c is not None else []
    shares = [row_detected_share((team_det, d), s, e) for d in dets]
    reason: str | None = None
    if insufficient_detection(team_share, threshold):
        reason = "insufficient_detection"
    elif c is None or not c.usable[s:e].any():
        reason = "insufficient_players"
    if c is not None:
        n_usable = int(c.usable[s:e].sum())
        t_row["coord_duration_s"] = n_usable / fs
        t_row["coord_n_samples"] = n_usable
        if reason is None and n_usable < MIN_CLUSTER_SAMPLES:
            reason = "too_short"
    if reason is not None:
        t_row["coord_cluster_source"] = reason
        t_row["coord_cluster_sampen_source"] = reason
        t_row["coord_cluster_surrogate_source"] = "not_scored"
        return t_row, [
            _cluster_child(ps, wrow, tm, axis, pid, reason, sh, signals) for pid, sh in zip(pids, shares, strict=True)
        ]
    if c is None:  # unreachable: a missing context is insufficient_players, returned above
        raise RuntimeError("cluster context missing on a scored window (a bug)")
    rel, valid, usable, on_pitch = c.rel, c.valid, c.usable, c.on_pitch
    st = kern.window_cluster_stats(rel, valid, usable, s, e)
    t_row["coord_rho_group_mean"] = st.rho_group_mean
    t_row["coord_rho_group_sd"] = st.rho_group_sd
    t_row["coord_n_players_mean"] = st.n_players_mean
    samp, samp_tok = _sampen_runs_token(_split_collapsed(st.rho_group_i, usable[s:e]), params)  # within-run (A-25)
    t_row["coord_rho_group_sampen"] = samp
    t_row["coord_cluster_source"] = "scored"  # A-20: an undefined SampEn no longer demotes a valid rho_group
    t_row["coord_cluster_sampen_source"] = samp_tok
    triple, src = _cluster_surrogate(ps, tm, axis, c, s, e, params, fs, st.rho_group_mean, kern)
    t_row["coord_rho_group_mean_surrogate_mean"] = triple[0]
    t_row["coord_rho_group_mean_percentile"] = triple[1]
    t_row["coord_rho_group_mean_excess"] = triple[2]
    t_row["coord_cluster_surrogate_source"] = src
    players = []
    for j, pid in enumerate(pids):
        prow = _cluster_player_row(ps, wrow, tm, axis, pid, signals)
        prow["coord_detected_share"] = shares[j]
        prow["coord_on_pitch_s"] = int(on_pitch[s:e, j].sum()) / fs
        sel = valid[s:e, j] & usable[s:e]
        p_reason = None
        if insufficient_detection(shares[j], threshold):
            p_reason = "insufficient_detection"
        elif int(sel.sum()) < MIN_CLUSTER_SAMPLES:  # A-21: one sample IS its own window mean -> rho_k == 1
            p_reason = "too_short"
        if p_reason is not None:
            prow["coord_cluster_player_source"] = p_reason
            prow["coord_cluster_player_sampen_source"] = p_reason
            players.append(prow)
            continue
        prow["coord_phi_mean_deg"] = float(np.degrees(st.phi_bar[j])) if np.isfinite(st.phi_bar[j]) else float("nan")
        prow["coord_rho_k"] = float(st.rho_k[j]) if np.isfinite(st.rho_k[j]) else float("nan")
        prow["coord_phi_sd_deg"] = (
            float(np.degrees(np.sqrt(-2.0 * np.log(min(float(st.rho_k[j]), 1.0)))))
            if np.isfinite(st.rho_k[j]) and st.rho_k[j] > 0
            else float("nan")
        )
        # phi_k is a wrapped angle: split into runs, unwrap each (precondition: step < pi), then SampEn (A-25)
        phi_runs = _unwrap_phase_runs(_split_collapsed(np.angle(rel[s:e, j][sel]), sel))
        samp_p, tok_p = _sampen_runs_token(phi_runs, params)
        prow["coord_phi_sampen"] = samp_p
        prow["coord_cluster_player_source"] = "scored"
        prow["coord_cluster_player_sampen_source"] = tok_p
        players.append(prow)
    return t_row, players


def _team_sync_row(ps, wrow, axis, cache, s, e, fs, params, signals, kern: _ClusterKernels):
    row = _cluster_base(ps, wrow, pd.NA, axis, COORDINATION_TEAM_SYNC_COLUMNS, signals)
    ta, tb = ps.team_ids
    row["team_a_id"] = _cid(ta)
    row["team_b_id"] = _cid(tb)
    share = row_detected_share((_team_det(ps, ta), _team_det(ps, tb)), s, e)  # spec 7.11, A-08
    row["coord_detected_share"] = share
    ca = cache[(str(canonical_id(ta)), axis)]
    cb = cache[(str(canonical_id(tb)), axis)]
    reason: str | None = None
    if insufficient_detection(share, params.min_observed_fraction["team_sync"]):
        reason = "insufficient_detection"
    elif ca is None or cb is None:
        reason = "insufficient_players"
    else:
        amp_a = ca.amp[s:e]
        amp_b = cb.amp[s:e]
        both = np.isfinite(amp_a) & np.isfinite(amp_b)
        if int(both.sum()) < 4:
            reason = "too_short"
        elif np.std(amp_a[both]) == 0 or np.std(amp_b[both]) == 0:
            reason = "degenerate_constant"
    if reason is not None:
        row["coord_team_sync_source"] = reason
        row["coord_team_sync_sampen_source"] = reason
        row["coord_team_sync_surrogate_source"] = "not_scored"
        return row
    if ca is None or cb is None:  # unreachable: a missing context is insufficient_players, returned above
        raise RuntimeError("team-sync context missing on a scored window (a bug)")
    r = float(kern.pearson_rows(amp_a, amp_b[None, :], both, 4)[0])  # the estimator every surrogate draw takes
    row["coord_team_sync_pearson_r"] = r
    xs, a_count, b_count = cross_sampen_over_runs(  # within-run templates, no gap-spanning (A-25)
        _split_collapsed(amp_a[both], both),
        _split_collapsed(amp_b[both], both),
        m=params.sampen_m,
        r=params.sampen_r_sd,
    )
    row["coord_team_sync_cross_sampen"] = xs
    row["coord_team_sync_source"] = "scored"
    row["coord_team_sync_sampen_source"] = "entropy_undefined" if (a_count == 0 or b_count == 0) else "scored"
    triple, src = _team_sync_surrogate(ps, tb, axis, cb, amp_a, s, e, both, r, params, fs, kern)
    row["coord_team_sync_pearson_r_surrogate_mean"] = triple[0]
    row["coord_team_sync_pearson_r_percentile"] = triple[1]
    row["coord_team_sync_pearson_r_excess"] = triple[2]
    row["coord_team_sync_surrogate_source"] = src
    return row


def _team_sync_surrogate(ps, tb, axis, cb: _ClusterCtx, amp_a, s, e, both, obs_r, params, fs, kern: _ClusterKernels):
    """The team-sync Pearson r triple (spec 7.9): team B's amplitude circularly shifted within each of its segments
    holding a row the r reads (``both``; :func:`_segments_holding`), every draw scored by the call's ``pearson_rows``
    over the rows finite on both sides."""
    if params.n_surrogates == 0:
        return (float("nan"), float("nan"), float("nan")), "disabled"
    _require_surrogate_method("team_sync", params.surrogate_method)  # A-17: refuse iaaft here
    segs = _segments_holding(ps.segments[tb], s + np.flatnonzero(both))
    tau = round(params.min_shift_s["cluster_amplitude"] * fs)
    shift_cols = []
    for lo, hi in segs:
        rng = surrogate_rng(
            params.surrogate_seed,
            (_seed_game(ps), int(ps.period_id), int(lo), str(canonical_id(tb)), axis, "team_sync"),
        )
        sh = draw_shifts(rng, hi - lo, tau, params.n_surrogates)
        if sh is None:
            return (float("nan"), float("nan"), float("nan")), "segment_too_short"
        shift_cols.append(sh)
    draws = shifted_window(cb.amp, segs, shift_cols, s, e, params.n_surrogates)
    return surrogate_triple(obs_r, kern.pearson_rows(amp_a, draws, both, 4)), "computed"


def compute_cluster_phase(
    signals: CoordinationSignals,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CoordinationReport]:
    """Cluster-team, cluster-player and team-sync tables + report (spec 7.8.6, C20).

    Examples
    --------
    Kuramoto group synchrony per team + per player (needs a real match)::

        cluster_team, cluster_player, team_sync, report = compute_cluster_phase(signals)
        cluster_team["coord_rho_group_mean"]   # group synchronisation index rho in [0, 1]
    """
    params = signals.params
    fs = signals.fs
    kern = _cluster_kernels()
    t_rows: list[dict[str, Any]] = []
    p_rows: list[dict[str, Any]] = []
    s_rows: list[dict[str, Any]] = []
    n_below_min = 0  # A-19: samples with the team present but fewer than min_players valid phasors (per team, axis)
    for ps in signals.periods:
        cache: dict[tuple[str, str], _ClusterCtx | None] = {}
        for tm in ps.team_ids:
            for axis in ("x", "y"):
                ctx = _cluster_ctx(ps, tm, axis, params, kern)
                cache[(str(canonical_id(tm)), axis)] = ctx
                if ctx is not None:
                    n_below_min += int((ctx.on_pitch.any(axis=1) & ~ctx.usable).sum())
        for row_idx, s, e in ps.window_ranges:
            wrow = signals.windows.loc[row_idx]
            for tm in ps.team_ids:
                for axis in ("x", "y"):
                    c = cache[(str(canonical_id(tm)), axis)]
                    t_row, players = _cluster_team_and_players(ps, wrow, tm, axis, c, s, e, fs, params, signals, kern)
                    t_rows.append(t_row)
                    p_rows.extend(players)
            for axis in ("x", "y"):
                s_rows.append(_team_sync_row(ps, wrow, axis, cache, s, e, fs, params, signals, kern))
    acc_t = _accum_from_rows(
        "coordination_cluster_team", t_rows, "coord_cluster_source", "coord_cluster_surrogate_source"
    )
    acc_t.samples_below_min_players = n_below_min  # A-19
    acc_p = _accum_from_rows("coordination_cluster_player", p_rows, "coord_cluster_player_source", None)
    acc_s = _accum_from_rows(
        "coordination_team_sync", s_rows, "coord_team_sync_source", "coord_team_sync_surrogate_source"
    )
    report = _report_from(signals, [acc_t, acc_p, acc_s])
    object.__setattr__(report, "_family_accums", [acc_t, acc_p, acc_s])
    if (message := _coverage_message(report)) is not None:
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return (
        _cast(t_rows, COORDINATION_CLUSTER_TEAM_COLUMNS),
        _cast(p_rows, COORDINATION_CLUSTER_PLAYER_COLUMNS),
        _cast(s_rows, COORDINATION_TEAM_SYNC_COLUMNS),
        report,
    )


# --------------------------------------------------------------------------- relative stretch (RSI)
def compute_relative_stretch(signals: CoordinationSignals) -> tuple[pd.DataFrame, CoordinationReport]:
    """Relative stretch index (SI_A - SI_B) per axis + report (spec 7.8.7).

    Examples
    --------
    Which team is more stretched, and how often the lead alternates (needs a real match)::

        rsi, report = compute_relative_stretch(signals)
        rsi[["coord_rsi_mean_m", "coord_rsi_switch_rate_per_min"]]   # metres; lead switches per minute
    """
    from scipy.stats import kurtosis, skew

    fs = signals.fs
    threshold = signals.params.min_observed_fraction["rsi"]
    rows: list[dict[str, Any]] = []
    axis_signal = {"x": "stretch_x", "y": "stretch_y"}
    for ps, wrow, s, e in _iter_windows(signals):
        # spec 7.7 pair order (A-23): A = the attacking team on possession windows, so RSI = SI_attacking - SI_defending
        ta, tb = _pair_teams(ps, wrow)
        share = row_detected_share((_team_det(ps, ta), _team_det(ps, tb)), s, e)  # spec 7.11, A-08
        for axis, sig in axis_signal.items():
            row = _rsi_base(ps, wrow, axis, ta, tb, signals)
            row["coord_detected_share"] = share
            if insufficient_detection(share, threshold):
                row["coord_rsi_source"] = "insufficient_detection"
                rows.append(row)
                continue
            si_a = ps.team_signal[(ta, sig)]
            si_b = ps.team_signal[(tb, sig)]
            runs = _both_runs(ps.segments[ta], ps.segments[tb], s, e)
            idx = np.concatenate([np.arange(lo, hi) for lo, hi in runs]) if runs else np.empty(0, dtype=np.int64)
            rsi = (si_a[idx] - si_b[idx]) if idx.size else np.empty(0)
            if rsi.size < 4:
                row["coord_rsi_source"] = "too_short"
                rows.append(row)
                continue
            if np.std(rsi) == 0:
                row["coord_rsi_source"] = "degenerate_constant"
                rows.append(row)
                continue
            row["coord_rsi_mean_m"] = float(np.mean(rsi))
            row["coord_rsi_fraction_positive"] = float(np.mean(rsi > 0))
            row["coord_rsi_switch_rate_per_min"] = _switch_count(rsi, runs) / (rsi.size / fs / 60.0)
            g = float(skew(rsi, bias=False))
            k = float(kurtosis(rsi, fisher=True, bias=False))
            n = rsi.size
            row["coord_rsi_bimodality_coefficient"] = (g**2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            row["coord_rsi_source"] = "scored"
            rows.append(row)
    acc = _accum_from_rows("coordination_rsi", rows, "coord_rsi_source", None)
    report = _report_from(signals, [acc])
    object.__setattr__(report, "_family_accums", [acc])  # the orchestrator's merged report counts these rows too
    if (message := _coverage_message(report)) is not None:
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return _cast(rows, COORDINATION_RSI_COLUMNS), report


def _rsi_base(ps, wrow, axis, ta, tb, signals) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": ps.game_id,
        "period_id": int(ps.period_id),
        "window_kind": wrow["window_kind"],
        "window_id": wrow["window_id"],
        "axis": axis,
        "team_a_id": _cid(ta),
        "team_b_id": _cid(tb),
    }
    for c, dt in COORDINATION_RSI_COLUMNS.items():
        if c not in row:
            row[c] = pd.NA if dt in ("Int64", "object") else float("nan")
    row["coord_detection_source"] = signals.detection_source
    row["coord_stoppage_source"] = signals.stoppage.source
    return row


def _sign_change_local_indices(rsi: np.ndarray, runs: list[tuple[int, int]]) -> list[int]:
    """Indices into the concatenated ``rsi`` where the sign changes vs the previous sample IN THE SAME run.

    The single source of the RSI sign-switch detection: :func:`_switch_count` takes its length,
    :func:`rsi_switch_times` maps each index to its grid time.
    """
    sign = np.sign(rsi)
    out: list[int] = []
    pos = 0
    for lo, hi in runs:
        n = hi - lo
        seg = sign[pos : pos + n]
        for local in np.flatnonzero(seg[1:] * seg[:-1] < 0) + 1:
            out.append(pos + int(local))
        pos += n
    return out


def _switch_count(rsi: np.ndarray, runs: list[tuple[int, int]]) -> int:
    """Sign changes between consecutive valid samples in the same run."""
    return len(_sign_change_local_indices(rsi, runs))


def rsi_switch_times(signals: CoordinationSignals) -> pd.DataFrame:
    """Per (game, period, axis), the grid TIMES where the relative-stretch index (SI_A - SI_B) changes sign
    within a valid run -- the RSI sign-switch events H7 tests against possession changes (spec 8.5).

    Single-sourced with :func:`compute_relative_stretch` (the same ``SI_A - SI_B`` series over ``_both_runs``) and
    :func:`_switch_count` (the same sign-change rule); computed over the period (match-half) windows only.

    Examples
    --------
    The H7 input: when the more-stretched team changes, per half and axis (needs a real match)::

        switches = rsi_switch_times(signals)
        switches[["period_id", "axis", "time"]]   # one row per sign change of SI_A - SI_B (grid seconds)
    """
    axis_signal = {"x": "stretch_x", "y": "stretch_y"}
    rows: list[dict[str, Any]] = []
    for ps, wrow, s, e in _iter_windows(signals):
        if wrow["window_kind"] != "period":
            continue
        ta, tb = ps.team_ids
        for axis, sig in axis_signal.items():
            si_a, si_b = ps.team_signal[(ta, sig)], ps.team_signal[(tb, sig)]
            runs = _both_runs(ps.segments[ta], ps.segments[tb], s, e)
            idx = np.concatenate([np.arange(lo, hi) for lo, hi in runs]) if runs else np.empty(0, dtype=np.int64)
            rsi = (si_a[idx] - si_b[idx]) if idx.size else np.empty(0)
            if rsi.size < 4:
                continue
            for local in _sign_change_local_indices(rsi, runs):
                rows.append(
                    {
                        "game_id": ps.game_id,
                        "period_id": int(ps.period_id),
                        "axis": axis,
                        "time": float(ps.t[idx[local]]),
                    }
                )
    return pd.DataFrame(rows, columns=["game_id", "period_id", "axis", "time"])


# --------------------------------------------------------------------------- orchestrator
@dataclass(frozen=True)
class CoordinationResult:
    """The full coordination output for a match: seven tables + one merged report.

    Examples
    --------
    The object :func:`compute_team_coordination` returns (needs a real match)::

        result = compute_team_coordination(frames, actions=actions)
        result.pair          # the combined per-pair table (rp + xc + vc + coh columns)
        result.report.conservation_errors()   # [] when the windows/rows accounting balances
    """

    windows: pd.DataFrame
    pair: pd.DataFrame
    pair_phase: pd.DataFrame
    spectral: pd.DataFrame
    cluster_team: pd.DataFrame
    cluster_player: pd.DataFrame
    team_sync: pd.DataFrame
    rsi: pd.DataFrame
    report: CoordinationReport


def _combine_on_keys(frames: list[pd.DataFrame], keys: tuple[str, ...], columns: dict[str, str]) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    if not frames:
        return _cast([], columns)
    base = frames[0].set_index(list(keys))
    for f in frames[1:]:
        base = base.combine_first(f.set_index(list(keys)))
    out = base.reset_index()
    return _cast(cast("list[dict[str, Any]]", out.to_dict("records")), columns)


def compute_team_coordination(
    frames: pd.DataFrame,
    *,
    windows: pd.DataFrame | None = None,
    actions: pd.DataFrame | None = None,
    params: CoordinationParams | None = None,
    goal_map: GoalMap | None = None,
    links: pd.DataFrame | None = None,
    levels: Sequence[str] = DEFAULT_LEVELS,
    dyad_windows: Literal["period", "all"] = "period",
) -> CoordinationResult:
    """Build signals once and run all seven coordination families (spec 7.2).

    Examples
    --------
    The one-call entry point (needs a real match's ``frames``; ``actions`` optional)::

        result = compute_team_coordination(frames, actions=actions)
        result.pair, result.spectral, result.cluster_team, result.rsi   # the family tables
    """
    if params is None:
        params = CoordinationParams.for_provider(str(frames["source_provider"].astype(object).iloc[0]))
    if windows is None:
        built = [period_windows(frames)]
        if actions is not None:
            built.append(possession_windows_from_actions(actions, frames, links=links, n_phases=params.n_phases))
        else:
            built.append(possession_windows_from_frames(frames, n_phases=params.n_phases, params=params))
        windows = pd.concat(built, ignore_index=True)
    signals = build_coordination_signals(
        frames, windows=windows, params=params, actions=actions, goal_map=goal_map, links=links
    )
    result = _result_from_signals(signals, levels=levels, dyad_windows=dyad_windows)
    if (message := _coverage_message(result.report)) is not None:  # ONE call-level warning (spec 7.13)
        warnings.warn(message, CoordinationCoverageWarning, stacklevel=2)
    return result


def _no_stage(_name: str) -> AbstractContextManager[None]:
    return nullcontext()


def _result_from_signals(
    signals: CoordinationSignals,
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    dyad_windows: Literal["period", "all"] = "period",
    on_stage: Callable[[str], AbstractContextManager[Any]] | None = None,
) -> CoordinationResult:
    """Run all seven coordination families on ALREADY-PREPARED signals and assemble the result.

    Split out of :func:`compute_team_coordination` so a caller that prepares signals once can re-score
    them under a different *post-preparation* ``params`` (Welch segment length, vector-coding epsilon,
    surrogate shift, analysis band) via ``dataclasses.replace(signals, params=...)`` without rebuilding
    the filtered/resampled series -- the D2 Tier-C sweep reuse (spec 8.4) and the shared corpus plumbing
    (``scripts/_coordination_corpus.match_tables``). ``compute_team_coordination`` is exactly
    ``build_coordination_signals`` followed by this, so its output is unchanged.

    ``on_stage(name)``, when given, returns a context manager wrapped around each family (``relative_phase``,
    ``cross_correlation``, ``vector_coding``, ``coherence``, ``spectral``, ``cluster_phase``, ``relative_stretch``)
    and the table ``combine`` -- the drivers' per-stage timing hook (R8). The result never depends on it.

    It never warns: the families' coverage warnings are suppressed here, and the one call-level warning (spec 7.13)
    is the public caller's to raise, so it names that caller's code.
    """
    stage = on_stage if on_stage is not None else _no_stage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        with stage("relative_phase"):
            rp_pair, rp_phase, rep_rp = compute_relative_phase(signals, levels=levels, dyad_windows=dyad_windows)
        with stage("cross_correlation"):
            xc_pair, rep_xc = compute_cross_correlation(signals, levels=levels, dyad_windows=dyad_windows)
        with stage("vector_coding"):
            vc_pair, vc_phase, rep_vc = compute_vector_coding(signals, levels=levels)
        with stage("coherence"):
            coh_pair, rep_coh = compute_coherence(signals, levels=levels)
        with stage("spectral"):
            spectral, rep_sp = compute_spectral(signals)
        with stage("cluster_phase"):
            cluster_team, cluster_player, team_sync, rep_cl = compute_cluster_phase(signals)
        with stage("relative_stretch"):
            rsi, rep_rsi = compute_relative_stretch(signals)
    from silly_kicks.coordination._columns import COORDINATION_PAIR_KEYS, COORDINATION_PAIR_PHASE_KEYS

    with stage("combine"):
        pair = _combine_on_keys(
            [rp_pair, xc_pair, vc_pair, coh_pair], COORDINATION_PAIR_KEYS, COORDINATION_PAIR_COLUMNS
        )
        pair_phase = _combine_on_keys(
            [rp_phase, vc_phase], COORDINATION_PAIR_PHASE_KEYS, COORDINATION_PAIR_PHASE_COLUMNS
        )
        accums: list[_Accum] = []
        iaaft_nc = 0
        for rep in (rep_rp, rep_xc, rep_vc, rep_coh, rep_sp, rep_cl, rep_rsi):
            accums.extend(getattr(rep, "_family_accums", []))
            iaaft_nc += rep.iaaft_nonconverged
        report = _report_from(signals, accums, iaaft_nonconverged=iaaft_nc)
    return CoordinationResult(
        windows=signals.windows,
        pair=pair,
        pair_phase=pair_phase,
        spectral=spectral,
        cluster_team=cluster_team,
        cluster_player=cluster_player,
        team_sync=team_sync,
        rsi=rsi,
        report=report,
    )
