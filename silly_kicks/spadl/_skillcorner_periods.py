"""SkillCorner Game Intelligence (GI) period-label repair (TF-46 Arm 1 Part 2, spec section 26; DL ruling 2026-10-25).

GI can label a possession block with the wrong ``period`` and compute its clock on that wrong period's basis
(measured: the second-half kick-off possession and its passing options labelled period 1 in 7 of 895 matches). The
GI frame id is the independent fact. For the frame-clock event types (``player_possession``, ``passing_option``,
``off_ball_run``) the GI clock is an exact function of the frame::

    clock = offset(period) + (frame_start - S_period) / 10

so GI alone recovers each period's start frame ``S_p``, and an event whose frame lies in another period is relabelled
and its clock re-based (the delta form, in integer deciseconds). ``on_ball_engagement`` is not on the frame clock: a
conflicting OBE is flagged, never repaired.

One pure helper, single-sourced: the SPADL converter (``spadl.skillcorner.convert_to_actions``) and
``providers.skillcorner.parse_passing_options`` both read :func:`repair_gi_periods`. The repair decision uses GI facts
only; ``match.json`` ``match_periods`` is a diagnostic. Match-level guards G0-G4 are all-or-nothing: a failed guard
repairs nothing (the output stays the unrepaired one), records the reason and warns; it never raises.

Imports numpy, pandas and ``silly_kicks.id_compat`` (itself numpy + pandas; the AGENTS.md id rule) only -- never
``silly_kicks.tracking`` (pinned by ``tests/spadl/test_skillcorner_periods.py``).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..id_compat import canonical_id_series

# SkillCorner's `time_start` is the CONTINUOUS broadcast clock (the 2nd half shows 45:00+,
# extra time 90:00+/105:00+), so each period's nominal start offset must be subtracted to get
# the period-relative seconds the tracking-frame contract requires (ADR-017). The offsets are
# the regulation period boundaries (45/90/105/120 min), matching the Gradient Sports adapter.
_PERIOD_START_SECONDS = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0, 5: 120 * 60.0}

#: The GI frame clock: 10 frames per second, so one frame = one decisecond of clock (a provider fact; guard G2
#: verifies it per match through the clock identity).
GI_FRAMES_PER_SECOND = 10

#: The GI event types on the frame clock (the identity holds for 100 % of them in 895 matches): repaired.
FRAME_CLOCK_EVENT_TYPES: tuple[str, ...] = ("player_possession", "passing_option", "off_ball_run")

#: ``period_source`` values: ``native`` = the period is the GI label; ``frame`` = the period and the time come from
#: the GI frame (spec section 26.5).
PERIOD_SOURCE_VALUES: tuple[str, ...] = ("native", "frame")

#: ``period_repair_status`` values (spec section 26.3).
PERIOD_REPAIR_STATUS_VALUES: tuple[str, ...] = (
    "clean",
    "repaired",
    "skipped_g1",
    "skipped_g2",
    "skipped_g3",
    "skipped_g4",
    "not_evaluable",
)

#: ``period_repair_shape`` values: the repaired possessions sit at the period boundary, or not.
PERIOD_REPAIR_SHAPE_VALUES: tuple[str, ...] = ("boundary_block", "interior")

_POSSESSION = "player_possession"
_OBE = "on_ball_engagement"
_LINKED = ("passing_option", "off_ball_run")
_ASSOC = "associated_player_possession_event_id"


class GiPeriodRepairWarning(UserWarning):
    """A SkillCorner GI period-label repair was skipped by a guard, had an interior shape, or found a GI period start
    that disagrees with ``match_periods`` (spec section 26). A separate category (AGENTS.md: never one umbrella)."""


def _parse_time_start(time_start: pd.Series) -> pd.Series:
    """Parse ``MM:SS.d`` time strings to float seconds (CONTINUOUS match clock).

    NOTE: SkillCorner's ``time_start`` is the continuous broadcast clock (2nd-half values are
    45:00+), so this returns seconds since MATCH start, NOT since period start. Callers must
    re-base to period-relative via ``spadl.skillcorner._to_period_relative`` to satisfy the period-relative
    frame contract (ADR-017).

    Parameters
    ----------
    time_start : pd.Series
        String series in ``"MM:SS.d"`` format (e.g. ``"12:34.5"``).

    Returns
    -------
    pd.Series
        Float64 seconds since match start (continuous broadcast clock).
    """
    parts = time_start.str.split(":", expand=True)
    minutes = parts[0].astype("float64")
    seconds = parts[1].astype("float64")
    return minutes * 60 + seconds


@dataclasses.dataclass(frozen=True)
class GiPeriodRepair:
    """The outcome of :func:`repair_gi_periods` for one match. Every Series is index-aligned to the input events.

    ``period`` is the GI ``period`` column with the repaired labels; ``clock`` the continuous clock in seconds (the
    :func:`_parse_time_start` value, re-based on a repaired row); ``period_source`` ``native`` / ``frame`` per row;
    ``obe_conflict`` True on an ``on_ball_engagement`` whose label conflicts with its frame period (withheld from
    tackle enrichment; never repaired). ``status`` in :data:`PERIOD_REPAIR_STATUS_VALUES`; ``relabelled`` = repaired
    events per frame-clock type; ``shape`` in :data:`PERIOD_REPAIR_SHAPE_VALUES` (``None`` unless repaired);
    ``start_frames`` = the GI-implied ``S_p``; ``period_start_mismatch`` = a ``match_periods`` start differs."""

    period: pd.Series
    clock: pd.Series
    period_source: pd.Series
    obe_conflict: pd.Series
    status: str
    relabelled: Mapping[str, int]
    shape: str | None
    start_frames: Mapping[int, int]
    period_start_mismatch: bool


def _clock_seconds(events: pd.DataFrame) -> pd.Series:
    """The continuous clock per row: :func:`_parse_time_start` on every row with a ``MM:SS.d`` value, NaN elsewhere
    (the same arithmetic, so an unrepaired row is byte-identical to the converter's former parse)."""
    out = pd.Series(np.nan, index=events.index, dtype="float64")
    if "time_start" not in events.columns or len(events) == 0:
        return out
    ts = events["time_start"]
    ok = ts.notna() & ts.astype(str).str.contains(":", regex=False)
    if ok.any():
        out.loc[ok] = _parse_time_start(ts[ok].astype(str)).to_numpy("float64")
    return out


def _mode_smallest(values: np.ndarray) -> int:
    """The most frequent value; a tie resolves to the smallest (order-insensitive, ADR-065)."""
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.flatnonzero(counts == counts.max())[0]])


def _warn(message: str) -> None:
    # stacklevel=3 = the AGENTS.md stacklevel=2 seen from repair_gi_periods (this helper adds one frame): the warning
    # points at repair_gi_periods' caller (the converter or parse_passing_options).
    warnings.warn(f"repair_gi_periods: {message}", GiPeriodRepairWarning, stacklevel=3)


def _is_boundary_block(
    pp_rows: np.ndarray, frame: np.ndarray, before: np.ndarray, after: np.ndarray, repaired: np.ndarray
) -> bool:
    """True when, for every (old -> new) period move, the repaired possessions are the possessions of the new
    period nearest the old one in frame order (the first ones when the new period follows, the last when it
    precedes): a block at the period boundary. ``pp_rows`` are the positions of the evaluable possessions."""
    moves = {(int(before[i]), int(after[i])) for i in pp_rows[repaired[pp_rows]]}
    if not moves:
        return False  # only linked events moved: not a possession block
    for old, new in moves:
        group = {int(i) for i in pp_rows if repaired[i] and int(before[i]) == old and int(after[i]) == new}
        in_new = [int(i) for i in pp_rows if int(after[i]) == new]
        in_new.sort(key=lambda i: (frame[i], i))
        k = len(group)
        edge = in_new[:k] if new > old else in_new[-k:]
        if set(edge) != group:
            return False
    return True


def repair_gi_periods(
    events: pd.DataFrame, *, match_periods: Sequence[Mapping[str, object]] | None = None
) -> GiPeriodRepair:
    """Repair the GI period labels of ONE match (spec section 26.2) under the match-level guards (section 26.3).

    ``events`` is the GI dynamic-events table (all event types; the columns ``event_type``, ``frame_start``,
    ``period``, ``time_start`` and, for guard G4, ``associated_player_possession_event_id``). ``match_periods`` is the
    optional ``match.json`` ``match_periods`` list: a diagnostic only (section 26.4), never a gate.

    Examples
    --------
    Repair the labels, then read the repaired period and clock of the possessions::

        repair = repair_gi_periods(events, match_periods=match_metadata.get("match_periods"))
        repair.status                      # "clean" / "repaired" / "skipped_g2" / ...
        pp = events["event_type"] == "player_possession"
        repair.period[pp], repair.clock[pp], repair.period_source[pp]
    """
    idx = events.index
    clock = _clock_seconds(events)
    period = events["period"].copy() if "period" in events.columns else pd.Series(np.nan, index=idx)
    source = pd.Series(PERIOD_SOURCE_VALUES[0], index=idx, dtype=object)
    obe_conflict = pd.Series(False, index=idx, dtype=bool)
    relabelled = dict.fromkeys(FRAME_CLOCK_EVENT_TYPES, 0)

    def _result(status: str, **kw) -> GiPeriodRepair:
        fields = {
            "period": period,
            "clock": clock,
            "period_source": source,
            "obe_conflict": obe_conflict,
            "relabelled": relabelled,
            "shape": None,
            "start_frames": {},
            "period_start_mismatch": False,
            **kw,
        }
        return GiPeriodRepair(status=status, **fields)

    # G0: the GI frame facts are present (an input without them is not evaluable; no warning).
    if any(c not in events.columns for c in ("event_type", "frame_start", "period", "time_start")):
        return _result("not_evaluable")
    etype = events["event_type"].astype(str).to_numpy()
    frame = pd.to_numeric(events["frame_start"], errors="coerce").to_numpy("float64")
    label = pd.to_numeric(events["period"], errors="coerce").to_numpy("float64")
    ck = clock.to_numpy("float64")
    framed = np.isfinite(frame) & np.isfinite(label)
    on_clock = np.isin(etype, FRAME_CLOCK_EVENT_TYPES) & framed & np.isfinite(ck)
    if not on_clock.any():
        return _result("not_evaluable")
    c10 = np.where(np.isfinite(ck), np.round(ck * GI_FRAMES_PER_SECOND), 0.0).astype("int64")
    fr = np.where(framed, frame, 0.0).astype("int64")
    lab = np.where(framed, label, 0.0).astype("int64")
    is_pp = on_clock & (etype == _POSSESSION)

    # G1: every labelled period with GI events has a possession (so S_p exists) and a known offset.
    labelled = sorted({int(p) for p in lab[framed]})
    missing = [p for p in labelled if p not in _PERIOD_START_SECONDS or not (is_pp & (lab == p)).any()]
    if missing:
        _warn(f"guard G1 failed (period(s) {missing} carry GI events but no possession / known offset); not repaired")
        return _result("skipped_g1")
    off10 = {p: round(_PERIOD_START_SECONDS[p] * GI_FRAMES_PER_SECOND) for p in labelled}
    starts = {p: _mode_smallest(fr[is_pp & (lab == p)] - (c10[is_pp & (lab == p)] - off10[p])) for p in labelled}

    # The match_periods diagnostic (section 26.4): start frames only; a mismatch warns and the repair still runs.
    mismatch = False
    for mp in match_periods or ():
        p, s = mp.get("period"), mp.get("start_frame")
        if p is not None and s is not None and int(p) in starts and int(s) != starts[int(p)]:  # type: ignore[arg-type]
            mismatch = True
    if mismatch:
        _warn(f"GI period start frames {starts} disagree with match_periods (diagnostic; the repair still runs)")

    # G2: the clock identity holds for EVERY frame-clock event under its own label, within one decisecond.
    s_lab = np.array([starts.get(int(p), 0) for p in lab], dtype="int64")
    o_lab = np.array([off10.get(int(p), 0) for p in lab], dtype="int64")
    residual = np.abs(c10 - (o_lab + fr - s_lab))
    if bool((residual[on_clock] > 1).any()):
        n_bad = int((residual[on_clock] > 1).sum())
        _warn(f"guard G2 failed ({n_bad} frame-clock event(s) off the GI clock identity by > 1 ds); not repaired")
        return _result("skipped_g2", start_frames=starts, period_start_mismatch=mismatch)

    # G3: the start frames strictly increase with the period.
    order = sorted(starts)
    s_sorted = np.array([starts[p] for p in order], dtype="int64")
    if len(s_sorted) > 1 and not bool((np.diff(s_sorted) > 0).all()):
        _warn(f"guard G3 failed (period start frames {starts} do not increase); not repaired")
        return _result("skipped_g3", start_frames=starts, period_start_mismatch=mismatch)

    # The frame's period: the last period whose start frame is at or before it (none before the first start).
    pos = np.searchsorted(s_sorted, fr, side="right") - 1
    frame_period = np.where(framed & (pos >= 0), np.array(order, dtype="int64")[np.clip(pos, 0, None)], -1)
    conflict = framed & (frame_period >= 0) & (frame_period != lab)
    fix = on_clock & conflict
    new_lab = np.where(fix, frame_period, lab)

    # G2, the clockless case: a frame-clock event whose frame lies in another period but whose clock cannot be read
    # cannot be re-based, so the identity cannot hold for it -- the match is not repaired (all-or-nothing).
    clockless = np.isin(etype, FRAME_CLOCK_EVENT_TYPES) & conflict & ~np.isfinite(ck)
    if bool(clockless.any()):
        n_bad = int(clockless.sum())
        _warn(
            f"guard G2 failed ({n_bad} frame-clock event(s) in another period without a readable clock); not repaired"
        )
        return _result("skipped_g2", start_frames=starts, period_start_mismatch=mismatch)

    # G4 (post-condition): every passing_option / off_ball_run has the period of its associated possession. Ids join
    # through id_compat (AGENTS.md: never astype(str) a join id -- "8.0" from a NaN-promoted column must meet "8").
    linked = on_clock & np.isin(etype, _LINKED)
    if _ASSOC in events.columns and "event_id" in events.columns:
        eid = canonical_id_series(events["event_id"]).to_numpy(dtype=object)
        assoc = canonical_id_series(events[_ASSOC]).to_numpy(dtype=object)
        pp_period = {eid[i]: int(new_lab[i]) for i in np.flatnonzero(is_pp) if not pd.isna(eid[i])}
        bad = [
            i
            for i in np.flatnonzero(linked)
            if not pd.isna(assoc[i]) and assoc[i] in pp_period and pp_period[assoc[i]] != int(new_lab[i])
        ]
        if bad:
            _warn(f"guard G4 failed ({len(bad)} linked event(s) off their possession's period); not repaired")
            return _result("skipped_g4", start_frames=starts, period_start_mismatch=mismatch)
    elif bool((fix & linked).any()):
        _warn(
            "guard G4 failed (linked events would be relabelled but the possession association cannot be read: no "
            f"event_id / {_ASSOC} column); not repaired"
        )
        return _result("skipped_g4", start_frames=starts, period_start_mismatch=mismatch)

    obe_conflict = pd.Series((etype == _OBE) & conflict, index=idx, dtype=bool)
    if not fix.any():
        return _result("clean", obe_conflict=obe_conflict, start_frames=starts, period_start_mismatch=mismatch)

    # The repair: relabel + delta re-base in integer deciseconds (section 26.2.4), divided by 10 once.
    s_new = np.array([starts.get(int(p), 0) for p in new_lab], dtype="int64")
    o_new = np.array([off10.get(int(p), 0) for p in new_lab], dtype="int64")
    c10_new = c10 + (o_new - o_lab) - (s_new - s_lab)
    rows = idx[fix]
    period = period.copy()
    period.loc[rows] = new_lab[fix]
    clock = clock.copy()
    clock.loc[rows] = c10_new[fix] / float(GI_FRAMES_PER_SECOND)
    source = source.copy()
    source.loc[rows] = PERIOD_SOURCE_VALUES[1]
    relabelled = {t: int((fix & (etype == t)).sum()) for t in FRAME_CLOCK_EVENT_TYPES}
    pp_rows = np.flatnonzero(is_pp)
    shape = "boundary_block" if _is_boundary_block(pp_rows, fr, lab, new_lab, fix) else "interior"
    if shape == "interior":
        _warn(f"repaired {relabelled} with an interior shape (not a block at a period boundary)")
    return _result(
        "repaired",
        period=period,
        clock=clock,
        period_source=source,
        obe_conflict=obe_conflict,
        relabelled=relabelled,
        shape=shape,
        start_frames=starts,
        period_start_mismatch=mismatch,
    )
