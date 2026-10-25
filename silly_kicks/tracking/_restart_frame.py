"""The tracking<->action frame transform and the GI direction check / per-match pairing (TF-46 Arm 1 Part 2,
spec sections 4.3, 6.1; CL-2).

One transform maps tracking-frame coordinates to the action frame, with the direction taken from ``GoalMap``
(ADR-055) and the ADR-045 180 deg point reflection; it adds NO new direction logic. The GI direction check
compares, per team and period, the GI ``attacking_side`` majority with ``GoalMap``. The per-match pairing
decides ``gi_pairing`` from three tests (Range, Alignment = proximity + lag, Direction).

Pure numpy / pandas. SkillCorner frame-level possession (``sc_possession_player_id``) serves the pairing
alignment ONLY (spec section 17.7); it is never truth for a gated signal.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id, canonical_id_series

from ._gk_resolve import GoalMap
from ._provider_visibility import detected_mask
from ._restart_validation import RATIFIED
from .schema import FRAME_FLIPS_ATTR, PERIOD_FRAME_RANGES_ATTR


def _i(v) -> int:
    """pandas Scalar / Hashable -> int; the untyped arg sidesteps the stub friction."""
    return int(v)


_PITCH_LENGTH_M = 105.0
_PITCH_WIDTH_M = 68.0


class GiDirectionMismatchWarning(UserWarning):
    """A SkillCorner period whose GI ``attacking_side`` majority disagrees with ``GoalMap`` (spec section 4.3).
    A separate category (AGENTS.md: never one umbrella warning)."""


# ---------------------------------------------------------------------------
# The frame transform (spec section 4.3)
# ---------------------------------------------------------------------------


def team_attacks_rtl(rows: pd.DataFrame, goal_map: GoalMap, *, team_col: str = "team_id") -> pd.Series:
    """Nullable boolean per row: True when the row's team attacks right-to-left in its period, i.e.
    ``GoalMap.attacked_goal(game, period, team) == 0.0``; ``<NA>`` when the direction is unresolved
    (ADR-051). No new direction logic -- the answer is ``GoalMap``'s."""
    out: list[object] = []
    for gid, pid, team in zip(rows["game_id"], rows["period_id"], rows[team_col], strict=True):
        g = goal_map.attacked_goal(gid, _i(pid), team, allow_guess=False)
        out.append(pd.NA if g is None else bool(g == 0.0))
    return pd.Series(out, index=rows.index, dtype="boolean")


def to_action_frame(
    x: np.ndarray, y: np.ndarray, rtl: pd.Series | pd.arrays.BooleanArray
) -> tuple[np.ndarray, np.ndarray]:
    """Map tracking coordinates to the action frame: the ADR-045 180 deg point reflection
    ``(105 - x, 68 - y)`` where ``rtl`` is True, the identity where False, and NaN where ``rtl`` is ``<NA>``
    (a withheld direction nulls the geometry)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    vals = pd.Series(rtl).astype("boolean")
    is_na = np.asarray(vals.isna(), dtype=bool)
    is_true = np.asarray(vals.fillna(False).to_numpy(dtype=bool)) & ~is_na
    xout = np.where(is_true, _PITCH_LENGTH_M - x, x)
    yout = np.where(is_true, _PITCH_WIDTH_M - y, y)
    xout = np.where(is_na, np.nan, xout)
    yout = np.where(is_na, np.nan, yout)
    return xout, yout


# ---------------------------------------------------------------------------
# GI direction check (spec section 4.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GiDirectionCheck:
    withheld: pd.Series  # bool, index-aligned to actions: geometry withheld for the row
    mismatched_periods: tuple[tuple[object, int], ...]
    n_noise_possessions: int
    n_insufficient_team_periods: int
    n_periods: int
    #: every (game, period) with a GI-backed majority check (the corpus guard's denominator, spec 4.3)
    checked_periods: tuple[tuple[object, int], ...] = ()
    #: every (game, period) with GI possessions but no recorded frame flip: direction UNRESOLVED, the period is
    #: withheld and counted, never compared (raw GI against oriented frames is the bug class, R.2 2026-10-23)
    unresolved_periods: tuple[tuple[object, int], ...] = ()
    #: every (game, period) where at least one team-period reached the GoalMap comparison (resolved flip AND
    #: at least ``min_possessions``): the pairing Direction test needs one per match (DL ruling 2026-10-23)
    compared_periods: tuple[tuple[object, int], ...] = ()
    #: every (game, period, team) below the possession floor (withheld, counted per match by the pairing)
    insufficient_team_periods: tuple[tuple[object, int, object], ...] = ()


#: The frames' per-period orientation flip, keyed ``(game_id, period_id)``: ``True`` = the period was
#: point-reflected to left-to-right, ``False`` = left as delivered, ``None`` = no decision (a no-anchor period).
#: The SkillCorner adapter records it per period (``TrackingConversionReport.ltr_flip_by_period``, keyed by period
#: only -- re-key it with ``silly_kicks.tracking.skillcorner.frame_flips_from_report``; ``frames.attrs``).
FrameFlips = Mapping[tuple[Any, int], "bool | None"]  # Any: a Mapping key is invariant (str / int game ids)


def _id_str(v: object) -> str:
    """The canonical string form of an id (ADR-019): an int and a string game id meet."""
    return str(canonical_id(v))


def _flip_key(game_id: object, period_id: object) -> tuple[str, int]:
    return _id_str(game_id), int(period_id)  # type: ignore[arg-type]


def resolve_frame_flips(frames: pd.DataFrame | None, flips: FrameFlips | None = None) -> dict[tuple[str, int], Any]:
    """The frames' per-period orientation flips: the declared ``flips`` first, else the adapter's
    ``frames.attrs[FRAME_FLIPS_ATTR]`` stamp, else ``{}`` (every period UNRESOLVED -- fail closed). Keys are
    canonicalised to ``(canonical game id, int period)`` so an int and a string game id meet."""
    if flips is not None:
        bad = [k for k in flips if not (isinstance(k, tuple) and len(k) == 2)]
        if bad:
            # the adapter report is keyed by period only; a raw report dict would otherwise crash on unpacking
            raise TypeError(
                f"flips must be keyed (game_id, period_id); got key {bad[0]!r}. Build it from a SkillCorner tracking "
                "report with silly_kicks.tracking.skillcorner.frame_flips_from_report(report)."
            )
        return {_flip_key(g, p): v for (g, p), v in flips.items()}
    stamp = getattr(frames, "attrs", {}).get(FRAME_FLIPS_ATTR) if frames is not None else None
    if not stamp:
        return {}
    return {_flip_key(g, p): (None if v is None else bool(v)) for g, p, v in stamp}


#: GI ``attacking_side`` strings that mean "attacking right-to-left" (toward x = 0, the attacked goal at 0.0).
_RTL_SIDES = frozenset({"rtl", "right_to_left", "left"})
_LTR_SIDES = frozenset({"ltr", "left_to_right", "right"})


def _side_is_rtl(side: object) -> bool | None:
    s = None if side is None or (isinstance(side, float) and np.isnan(side)) else str(side).strip().lower()
    if s in _RTL_SIDES:
        return True
    if s in _LTR_SIDES:
        return False
    return None


def gi_direction_check(
    actions: pd.DataFrame, goal_map: GoalMap, *, flips: FrameFlips | None, min_possessions: int = 5
) -> GiDirectionCheck:
    """Per team and period, the majority GI ``attacking_side`` (over at least ``min_possessions``
    possessions) is compared with ``GoalMap`` (spec section 4.3). A team-period below the floor is withheld
    and counted; a possession disagreeing with its period majority is withheld and counted; a period whose
    majority disagrees with ``GoalMap`` withholds the whole period, is counted, and warns (never raises).

    GI ``attacking_side`` is in the AS-DELIVERED tracking frame, ``GoalMap`` in the frames' (oriented) frame, so
    ``flips`` (:data:`FrameFlips`, the adapter's recorded per-period flip; :func:`resolve_frame_flips`) maps the GI
    side into the frames' frame before the comparison: oriented rtl = delivered rtl XOR flip. A period without a
    recorded flip (absent, or ``None``; ``flips=None`` = none recorded) is direction-UNRESOLVED: withheld,
    counted in ``unresolved_periods``, never compared."""
    withheld = pd.Series(False, index=actions.index)
    if len(actions) == 0 or "gi_attacking_side" not in actions.columns:
        return GiDirectionCheck(withheld, (), 0, 0, 0)

    flip_lut = resolve_frame_flips(None, flips) if flips is not None else {}
    rtl_gi = actions["gi_attacking_side"].map(_side_is_rtl)  # True/False/None per row (delivered frame)
    backed = rtl_gi.notna()  # rows backed by a readable GI possession
    mismatched: list[tuple[object, int]] = []
    unresolved: list[tuple[object, int]] = []
    n_noise = 0
    n_insufficient = 0
    periods = set()
    # ADR-068 / ADR-019: the (game, period) row groups are built ONCE with canonical ids (no per-group raw-`==` scan)
    period_rows = group_rows(actions, ("game_id", "period_id"))
    compared: list[tuple[object, int]] = []
    insufficient: list[tuple[object, int, object]] = []

    for (gid, pid, team), grp in actions[backed].groupby(["game_id", "period_id", "team_id"], sort=False):
        flip = flip_lut.get(_flip_key(gid, pid))
        if flip is None:
            # no recorded flip -> the GI side cannot be mapped into the frames' frame: unresolved, withhold the period
            withheld.loc[period_rows.get(gid, _i(pid)).index] = True
            unresolved.append((gid, _i(pid)))
            continue
        periods.add((gid, _i(pid)))
        sides = rtl_gi.loc[grp.index].astype(bool) ^ bool(flip)  # into the frames' (oriented) frame
        if len(sides) < min_possessions:
            withheld.loc[grp.index] = True
            n_insufficient += 1
            insufficient.append((gid, _i(pid), team))
            continue
        compared.append((gid, _i(pid)))
        majority_rtl = bool(sides.sum() * 2 > len(sides)) if sides.sum() * 2 != len(sides) else bool(sides.iloc[0])
        # possessions disagreeing with the period majority are GI noise -> withhold those rows
        noise = sides != majority_rtl
        if bool(noise.any()):
            withheld.loc[grp.index[noise.to_numpy()]] = True
            n_noise += int(noise.sum())
        gm = goal_map.attacked_goal(gid, _i(pid), team, allow_guess=False)
        gm_rtl = None if gm is None else bool(gm == 0.0)
        if gm_rtl is None or gm_rtl != majority_rtl:
            # the period majority disagrees with GoalMap (or GoalMap unresolved) -> withhold the whole period
            withheld.loc[period_rows.get(gid, _i(pid)).index] = True
            mismatched.append((gid, _i(pid)))
            warnings.warn(
                f"gi_direction_check: game={gid} period={pid} team={team} GI majority disagrees with GoalMap "
                f"(gi_rtl={majority_rtl} after the recorded flip={bool(flip)}, goalmap_rtl={gm_rtl}) -- geometry "
                "withheld for the period.",
                GiDirectionMismatchWarning,
                stacklevel=2,
            )

    return GiDirectionCheck(
        withheld=withheld,
        mismatched_periods=tuple(dict.fromkeys(mismatched)),
        n_noise_possessions=n_noise,
        n_insufficient_team_periods=n_insufficient,
        n_periods=len(periods),
        checked_periods=tuple(sorted(periods, key=repr)),
        unresolved_periods=tuple(sorted(dict.fromkeys(unresolved), key=repr)),
        compared_periods=tuple(sorted(dict.fromkeys(compared), key=repr)),
        insufficient_team_periods=tuple(insufficient),
    )


# ---------------------------------------------------------------------------
# Per-match GI / tracking pairing (spec section 6.1)
# ---------------------------------------------------------------------------

GI_PAIRING_VALUES = ("paired", "unpaired")
GI_PAIRING_FAILED_TESTS = ("range", "proximity", "lag", "direction")


@dataclass(frozen=True)
class GiPairing:
    game_id: object
    pairing: str  # GI_PAIRING_VALUES
    failed_test: str | None  # first failing test in GI_PAIRING_FAILED_TESTS order; None when paired
    prox_share: float  # share within pair_prox_m at the possession start, detected balls only (NaN if none)
    alignment_offset: int | None  # the offset maximising interval agreement; None when unevaluable
    alignment_agreement: float  # the agreement at alignment_offset
    n_tied_offsets: int  # offsets sharing the maximum agreement (B-15)
    alignment_source: str  # "sc_possession" | "proximity_fallback"
    n_evaluable: int  # de-duplicated possession starts with possessor AND ball detected
    gi_source: str | None  # provenance; recorded, never decisive
    tracking_source: str | None
    provenance_mismatch: bool  # recorded flag only
    # Every test's verdict (DL ruling 2026-10-23), evaluated even after an earlier test fails; ``None`` = not
    # evaluable (no GI possession / no evaluable start). The decision order and ``failed_test`` are unchanged:
    # paired only when all four are True, ``failed_test`` = the first that is not.
    pair_range_ok: bool | None = None
    pair_proximity_ok: bool | None = None
    pair_lag_ok: bool | None = None
    pair_direction_ok: bool | None = None
    n_unresolved_periods: int = 0  # this match's periods with no recorded frame flip (never compared)
    n_insufficient_team_periods: int = 0  # this match's team-periods below the GI possession floor
    #: the period range the Range test read (RANGE_BASIS_VALUES): "provider" = the raw provider period range (R.2
    #: requires it); "retained" = the frames with objects (the stricter fallback; spec section 6.1, amended 2026-10-25)
    range_basis: str = "retained"
    #: every Range violation as (period_id, frame_start, frame_end, class), class in RANGE_VIOLATION_CLASSES; any
    #: violation fails the match (no clipping; the 2026-10-23 edge rule is retired, DL ruling 2026-10-25)
    range_violations: tuple[tuple[int, int, int, str], ...] = ()


#: ``GiPairing.range_basis`` values (spec section 6.1, amended 2026-10-25).
RANGE_BASIS_VALUES: tuple[str, ...] = ("provider", "retained")
#: ``GiPairing.range_violations`` classes: inside ANOTHER period's range; outside every range; partly over its own
#: range's edge; a label period with no range (spec section 6.1, amended 2026-10-25). Each fails the match.
RANGE_VIOLATION_CLASSES: tuple[str, ...] = ("other_period", "outside", "partial_edge", "no_period_range")

#: The provider period range per ``(game_id, period_id)``: ``(first_frame, last_frame)`` of every raw tracking frame
#: the provider labels with that period, empty frames included. Build it from the SkillCorner tracking report with
#: ``silly_kicks.tracking.skillcorner.period_frame_ranges_from_report`` (the report field is keyed by period only).
PeriodFrameRanges = Mapping[tuple[Any, int], tuple[int, int]]


def resolve_period_frame_ranges(
    frames: pd.DataFrame | None, period_ranges: PeriodFrameRanges | None = None
) -> dict[tuple[str, int], tuple[int, int]]:
    """The provider period ranges: the declared ``period_ranges`` first, else the adapter's
    ``frames.attrs[PERIOD_FRAME_RANGES_ATTR]`` stamp, else ``{}`` (every match then falls back to its RETAINED
    range, ``range_basis = "retained"``). Keys canonicalised to ``(canonical game id, int period)``. A
    period-only key raises ``TypeError`` naming the re-key helper (the C2-IMPL-28 lesson)."""
    if period_ranges is not None:
        bad = [k for k in period_ranges if not (isinstance(k, tuple) and len(k) == 2)]
        if bad:
            raise TypeError(
                f"period_ranges must be keyed (game_id, period_id); got key {bad[0]!r}. Build it from a SkillCorner "
                "tracking report with silly_kicks.tracking.skillcorner.period_frame_ranges_from_report(report)."
            )
        return {_flip_key(g, p): (int(lo), int(hi)) for (g, p), (lo, hi) in period_ranges.items()}
    stamp = getattr(frames, "attrs", {}).get(PERIOD_FRAME_RANGES_ATTR) if frames is not None else None
    if not stamp:
        return {}
    return {_flip_key(g, p): (int(lo), int(hi)) for g, p, lo, hi in stamp}


#: The negative-control time shifts (seconds, both directions) of the proximity bar derivation (OD-6 amendment
#: 2026-10-23, spec sections 6.1 / 16.1): realistic clock-offset failures; they need no truth.
PAIRING_NEGATIVE_SHIFTS_S: tuple[float, ...] = (-30.0, -10.0, -5.0, 5.0, 10.0, 30.0)


class _DetectedIndex:
    """One match's DETECTED ball and player positions keyed by ``(period, frame)`` / ``(period, frame, player)``,
    built ONCE (ADR-068: the per-possession lookup never rescans the frames). The first detected row wins, as the
    row-order ``iloc[0]`` of the per-frame filter it replaces."""

    def __init__(self, frames: pd.DataFrame) -> None:
        self.ball: dict[tuple[int, int], tuple[float, float]] = {}
        self.player: dict[tuple[int, int, object], tuple[float, float]] = {}
        if len(frames) == 0:
            return
        det = np.asarray(detected_mask(frames["visibility"], provider="skillcorner"), dtype=bool)
        is_ball = frames["is_ball"].astype(bool).to_numpy()
        p = frames["period_id"].astype(int).to_numpy()
        f = frames["frame_id"].astype(int).to_numpy()
        x = frames["x"].to_numpy("float64")
        y = frames["y"].to_numpy("float64")
        for i in np.flatnonzero(is_ball & det):
            self.ball.setdefault((int(p[i]), int(f[i])), (float(x[i]), float(y[i])))
        pid = canonical_id_series(frames["player_id"]).to_numpy(dtype=object)
        for i in np.flatnonzero(~is_ball & det):
            self.player.setdefault((int(p[i]), int(f[i]), pid[i]), (float(x[i]), float(y[i])))

    def at(self, period: int, frame_id: int, player: object):
        """The (detected-ball xy or None, possessor xy or None) at one (period, frame)."""
        return self.ball.get((period, frame_id)), self.player.get((period, frame_id, canonical_id(player)))


def _frames_per_second(frames: pd.DataFrame) -> float:
    """The frame rate from the frame-id / clock relation (SkillCorner: 10 fps)."""
    g = frames.drop_duplicates(["period_id", "frame_id"]).sort_values(["period_id", "frame_id"])
    df = np.diff(g["frame_id"].to_numpy("float64"))
    dt = np.diff(g["time_seconds"].to_numpy("float64"))
    ok = (df > 0) & (dt > 0) & (np.diff(g["period_id"].to_numpy("int64")) == 0)
    return float(np.median(df[ok] / dt[ok])) if ok.any() else float("nan")


def pairing_proximity_profile(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    prox_ms: tuple[float, ...],
    shifts_s: tuple[float, ...] = PAIRING_NEGATIVE_SHIFTS_S,
) -> pd.DataFrame:
    """The pairing proximity share per match at each distance in ``prox_ms`` and each GI time shift (OD-6
    amendment 2026-10-23): ``shift_s == 0`` is the match itself; a non-zero shift moves every GI possession start
    by ``shift_s`` seconds (the negative controls the bar is derived from). Columns ``game_id``, ``prox_m``,
    ``shift_s``, ``prox_share``, ``n_evaluable``. SkillCorner GI-bracketed actions only; empty otherwise."""
    cols = ["game_id", "prox_m", "shift_s", "prox_share", "n_evaluable"]
    rows: list[dict] = []
    frame_groups = group_rows(frames, "game_id") if "game_id" in frames.columns else None  # ONCE, canonical ids
    for gid, agrp in actions.groupby("game_id", sort=False):
        mframes = frame_groups.get(gid) if frame_groups is not None else frames
        poss = _possessions(agrp)  # no clipping (the 2026-10-23 edge rule is retired); a start without tracking
        # data is simply not evaluable (no detected possessor / ball at that frame)
        index = _DetectedIndex(mframes)
        fps = _frames_per_second(mframes) if len(mframes) else float("nan")
        for m in prox_ms:
            for s in (0.0, *shifts_s):
                if s != 0.0 and not np.isfinite(fps):
                    share, n = float("nan"), 0  # no frame clock -> no shifted control (never a mislabelled 0)
                else:
                    share, n = _proximity_share(poss, index, float(m), offset=0 if s == 0.0 else round(s * fps))
                rows.append(
                    {"game_id": gid, "prox_m": float(m), "shift_s": float(s), "prox_share": share, "n_evaluable": n}
                )
    return pd.DataFrame(rows, columns=cols)


def interval_agreement(
    gi_intervals: pd.DataFrame,
    possessor_by_frame: pd.DataFrame,
    *,
    offsets: np.ndarray,
) -> np.ndarray:
    """Per offset ``o``: the share of GI-interval frames ``[frame_start + o, frame_end + o]`` whose frame-level
    possessor equals the GI possessor, pooled over every interval (spec section 6.1).

    ``gi_intervals`` columns: ``period_id``, ``frame_start``, ``frame_end``, ``possessor``.
    ``possessor_by_frame`` columns: ``period_id``, ``frame_id``, ``possessor`` -- one row per frame PRESENT in the
    tracking (a ``None`` possessor = present, nobody possessing). Ids compared with id_compat.

    A frame with no row (absent from the tracking) is missing data, not disagreement: it leaves the denominator
    (DL ruling 2026-10-25, Q4; counting it as a miss biased the argmax toward offsets that avoid gaps). A present
    frame with no possessor stays a miss. An interval set with every frame absent gives 0.0 at every offset.
    """
    lut: dict[tuple[int, int], object] = {}
    for p, f, poss in zip(
        possessor_by_frame["period_id"].astype(int),
        possessor_by_frame["frame_id"].astype(int),
        possessor_by_frame["possessor"],
        strict=True,
    ):
        lut[(p, f)] = canonical_id(poss) if poss is not None and not _isna(poss) else None

    out = np.zeros(len(offsets), dtype="float64")
    for oi, o in enumerate(offsets):
        total = 0
        hit = 0
        for p, fs, fe, poss in zip(
            gi_intervals["period_id"].astype(int),
            gi_intervals["frame_start"].astype(int),
            gi_intervals["frame_end"].astype(int),
            gi_intervals["possessor"],
            strict=True,
        ):
            cposs = canonical_id(poss) if poss is not None and not _isna(poss) else None
            for f in range(fs + int(o), fe + int(o) + 1):
                if (p, f) not in lut:
                    continue  # absent from the tracking: missing data, out of the denominator
                total += 1
                fp = lut[(p, f)]
                if fp is not None and cposs is not None and fp == cposs:
                    hit += 1
        out[oi] = (hit / total) if total else 0.0
    return out


def alignment_offset(agreement: np.ndarray, offsets: np.ndarray) -> tuple[int, int]:
    """The offset maximising ``agreement``; ties resolve to the smallest ``|o|``, then the negative ``o``
    (B-15). Returns ``(offset, n_tied)``."""
    if len(agreement) == 0:
        return 0, 0
    peak = float(np.max(agreement))
    tied = [int(o) for o, a in zip(offsets, agreement, strict=True) if abs(a - peak) < 1e-12]
    best = sorted(tied, key=lambda o: (abs(o), o))[0]
    return best, len(tied)


def _isna(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def assess_gi_pairing(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    provider: str,
    direction: GiDirectionCheck,
    gi_source: str | None = None,
    tracking_source: str | None = None,
    constants: Mapping[str, float] | None = None,
    period_ranges: PeriodFrameRanges | None = None,
) -> dict[object, GiPairing]:
    """Decide ``gi_pairing`` per SkillCorner match from the three tests (spec section 6.1), in order
    Range -> proximity -> lag -> Direction; ``failed_test`` names the first that fails. Non-SkillCorner
    providers get an empty dict (pairing applies only to GI-bracketed providers).

    ``constants`` overrides individual ratified pairing constants (run (2) re-computes the verdict with each at
    its ``sensitivity_values``, one at a time). Default ``None`` = the ratified values.

    ``period_ranges`` (:data:`PeriodFrameRanges`, keyed ``(game_id, period_id)``; build it with
    ``skillcorner.period_frame_ranges_from_report``) is the PROVIDER period range the Range test reads (spec section
    6.1, amended 2026-10-25): the declared keyword first, else the ``frames.attrs`` stamp, else -- per match -- the
    RETAINED range (the frames with objects; stricter, so it can only fail more). ``GiPairing.range_basis`` records
    which. A period-only key raises ``TypeError``."""
    if provider != "skillcorner" or len(actions) == 0:
        return {}
    c = {**RATIFIED, **(constants or {})}
    prox_m = c["pair_prox_m"]
    prox_share_bar = c["pair_prox_share"]
    lag_tol = round(c["pair_lag_tol_frames"])
    scan = round(c["pair_lag_scan_frames"])
    offsets = np.arange(-scan, scan + 1)
    provider_ranges = resolve_period_frame_ranges(frames, period_ranges)

    out: dict[object, GiPairing] = {}
    frame_groups = group_rows(frames, "game_id") if "game_id" in frames.columns else None  # ONCE, canonical ids
    for gid, agrp in actions.groupby("game_id", sort=False):
        mframes = frame_groups.get(gid) if frame_groups is not None else frames
        key = _id_str(gid)
        mranges = {p: lohi for (g, p), lohi in provider_ranges.items() if g == key}
        basis = RANGE_BASIS_VALUES[0] if mranges else RANGE_BASIS_VALUES[1]
        if not mranges:  # the fallback: the retained range (frames with objects), recorded as such
            mranges = {
                _i(p): (int(f.min()), int(f.max()))
                for p, f in ((p, g["frame_id"].to_numpy("int64")) for p, g in mframes.groupby("period_id"))
            }
        result = _pair_one_match(
            gid,
            agrp,
            mframes,
            direction,
            offsets,
            prox_m,
            prox_share_bar,
            lag_tol,
            gi_source,
            tracking_source,
            ranges=mranges,
            range_basis=basis,
        )
        out[gid] = result
    return out


def _possessions(agrp: pd.DataFrame) -> pd.DataFrame:
    """One row per GI possession for the match, de-duplicated by ``gi_event_id`` (C-PLAN-17)."""
    need = ["gi_event_id", "gi_possession_frame_start", "gi_possession_frame_end", "period_id", "player_id"]
    have = [c for c in need if c in agrp.columns]
    if "gi_event_id" not in have:
        return pd.DataFrame(columns=["period_id", "frame_start", "frame_end", "possessor", "gi_event_id"])
    df = agrp[have].dropna(subset=["gi_event_id", "gi_possession_frame_start", "gi_possession_frame_end"])
    df = df.drop_duplicates(subset=["gi_event_id"])
    return pd.DataFrame(
        {
            "period_id": df["period_id"].astype(int).to_numpy(),
            "frame_start": df["gi_possession_frame_start"].astype(int).to_numpy(),
            "frame_end": df["gi_possession_frame_end"].astype(int).to_numpy(),
            "possessor": df["player_id"].to_numpy(),
            "gi_event_id": df["gi_event_id"].to_numpy(),
        }
    )


def _pair_one_match(
    gid,
    agrp,
    frames,
    direction,
    offsets,
    prox_m,
    prox_share_bar,
    lag_tol,
    gi_source,
    tracking_source,
    *,
    ranges: Mapping[int, tuple[int, int]],
    range_basis: str,
) -> GiPairing:
    """Every test is evaluated (DL ruling 2026-10-23: a run shows lag and direction even when proximity fails);
    the decision keeps the order Range -> proximity -> lag -> Direction, ``failed_test`` = the first not True."""
    poss = _possessions(agrp)
    prov_mismatch = bool(gi_source is not None and tracking_source is not None and gi_source != tracking_source)
    index = _DetectedIndex(frames)

    # 1. Range (the provider period range, DL ruling 2026-10-25): every possession inside its period's range; any
    # violation fails the match and is recorded with its class. No clipping, no margin.
    range_ok, violations = _range_check(poss, ranges)
    has_poss = len(poss) > 0

    # 2a. Proximity: at each possession start, the possessor within prox_m of a DETECTED ball.
    prox_share, n_evaluable = _proximity_share(poss, index, prox_m)
    proximity_ok = None if n_evaluable == 0 else bool(prox_share >= prox_share_bar)

    # 2b. Lag: the alignment offset within lag_tol of 0 (interval agreement; sc_possession or proximity fallback).
    source, poss_by_frame = _possessor_by_frame(frames, poss, prox_m, index)
    off: int | None = None
    agree_at, n_tied = float("nan"), 0
    lag_ok: bool | None = None
    if has_poss:
        agreement = interval_agreement(poss, poss_by_frame, offsets=offsets)
        off, n_tied = alignment_offset(agreement, offsets)
        agree_at = float(agreement[int(np.where(offsets == off)[0][0])])
        # zero agreement at every offset carries no alignment evidence: not evaluable (the all-tie break would
        # otherwise pick offset 0 and pass a desynchronised file)
        if float(np.max(agreement)) <= 0.0:
            # no offset carries evidence: record none (the field contract), so the lag table and its median guard
            # never count a desynchronised file as aligned at 0 (C2 delta-2 A-02)
            lag_ok, off, n_tied = None, None, 0
        else:
            lag_ok = abs(off) <= lag_tol

    # 3. Direction: no period of this match mismatched AND at least one period actually compared (resolved flip,
    # enough possessions) -- zero compared periods is the absence of evidence, not agreement (fail closed).
    key = _id_str(gid)
    mismatched = any(_id_str(g) == key for g, _ in direction.mismatched_periods)
    compared = any(_id_str(g) == key for g, _ in direction.compared_periods)
    direction_ok = bool(compared and not mismatched)

    verdicts = (("range", range_ok), ("proximity", proximity_ok), ("lag", lag_ok), ("direction", direction_ok))
    failed = next((name for name, ok in verdicts if ok is not True), None)
    return GiPairing(
        game_id=gid,
        pairing="paired" if failed is None else "unpaired",
        failed_test=failed,
        prox_share=prox_share,
        alignment_offset=off,
        alignment_agreement=agree_at,
        n_tied_offsets=n_tied,
        alignment_source=source,
        n_evaluable=n_evaluable,
        gi_source=gi_source,
        tracking_source=tracking_source,
        provenance_mismatch=prov_mismatch,
        pair_range_ok=range_ok,
        pair_proximity_ok=proximity_ok,
        pair_lag_ok=lag_ok,
        pair_direction_ok=direction_ok,
        n_unresolved_periods=sum(1 for g, _ in direction.unresolved_periods if _id_str(g) == key),
        n_insufficient_team_periods=sum(1 for g, _, _ in direction.insufficient_team_periods if _id_str(g) == key),
        range_basis=range_basis,
        range_violations=violations,
    )


def _range_check(
    poss: pd.DataFrame, ranges: Mapping[int, tuple[int, int]]
) -> tuple[bool | None, tuple[tuple[int, int, int, str], ...]]:
    """The Range test (the provider period range; spec section 6.1, amended 2026-10-25, DL ruling). Returns
    ``(range_ok, violations)``: each possession's ``[frame_start, frame_end]`` must lie inside the range of its
    ``period_id``; a violation is ``(period_id, frame_start, frame_end, class)`` with class ``no_period_range`` (its
    period has no range), ``partial_edge`` (it overlaps its range but crosses an edge), ``other_period`` (wholly
    inside another period's range) or ``outside`` (wholly outside every range). Any violation fails; nothing is
    clipped and there is no frame margin. ``(None, ())`` without possessions."""
    if len(poss) == 0:
        return None, ()
    violations: list[tuple[int, int, int, str]] = []
    for p, fs, fe in zip(poss["period_id"], poss["frame_start"], poss["frame_end"], strict=True):
        p, fs, fe = int(p), int(fs), int(fe)
        lo_hi = ranges.get(p)
        if lo_hi is None:
            cls = "no_period_range"
        elif lo_hi[0] <= fs and fe <= lo_hi[1]:
            continue
        elif fe >= lo_hi[0] and fs <= lo_hi[1]:
            cls = "partial_edge"
        elif any(q != p and lo <= fs and fe <= hi for q, (lo, hi) in ranges.items()):
            cls = "other_period"
        else:
            cls = "outside"
        violations.append((p, fs, fe, cls))
    return not violations, tuple(violations)


def _proximity_share(poss: pd.DataFrame, index: _DetectedIndex, prox_m: float, *, offset: int = 0):
    """The share of evaluable GI possession starts (possessor AND ball detected at ``frame_start + offset``) with
    the possessor within ``prox_m`` of the ball; ``(NaN, 0)`` when none is evaluable."""
    near = 0
    evaluable = 0
    for p, fs, player in zip(poss["period_id"], poss["frame_start"], poss["possessor"], strict=True):
        ball_xy, player_xy = index.at(int(p), int(fs) + offset, player)
        if ball_xy is None or player_xy is None:
            continue  # possessor or ball not detected at the start -> not evaluable (detected balls only)
        evaluable += 1
        if np.hypot(ball_xy[0] - player_xy[0], ball_xy[1] - player_xy[1]) <= prox_m:
            near += 1
    return (near / evaluable if evaluable else float("nan")), evaluable


def _possessor_by_frame(
    frames: pd.DataFrame, poss: pd.DataFrame, prox_m: float, index: _DetectedIndex
) -> tuple[str, pd.DataFrame]:
    """The per-frame possessor for the lag test: ``sc_possession_player_id`` when present, else the marked
    proximity fallback (the player within prox_m of a detected ball)."""
    if "sc_possession_player_id" in frames.columns:
        # One definition of a PRESENT frame on both paths (any tracking row at (period, frame); DL ruling Q4): the
        # frame-level possessor from the first player row, else (a ball-only frame) from its ball row.
        is_ball = frames["is_ball"].astype(bool).to_numpy()
        keys = ["period_id", "frame_id"]
        players = frames[~is_ball].drop_duplicates(subset=keys)
        balls = frames[is_ball].drop_duplicates(subset=keys)
        have = pd.MultiIndex.from_frame(players[keys])
        one = pd.concat([players, balls[~pd.MultiIndex.from_frame(balls[keys]).isin(have)]], ignore_index=True)
        return "sc_possession", pd.DataFrame(
            {
                "period_id": one["period_id"].astype(int).to_numpy(),
                "frame_id": one["frame_id"].astype(int).to_numpy(),
                "possessor": one["sc_possession_player_id"].to_numpy(),
            }
        )
    # Fallback: the GI possessor within prox_m of a detected ball at each frame of each interval. EVERY frame present
    # in the tracking gets a row (a frame outside every interval: possessor None, a miss when a shifted interval reads
    # it); an absent frame gets none -- missing data, out of the agreement denominator (DL ruling 2026-10-25, Q4). A
    # frame inside two intervals keeps the later interval's answer (the former row order, last write wins).
    keys = frames[["period_id", "frame_id"]].drop_duplicates()
    at: dict[tuple[int, int], object] = {
        (_i(p), _i(f)): None for p, f in zip(keys["period_id"].to_numpy(), keys["frame_id"].to_numpy(), strict=True)
    }
    for p, fs, fe, player in zip(
        poss["period_id"], poss["frame_start"], poss["frame_end"], poss["possessor"], strict=True
    ):
        for f in range(int(fs), int(fe) + 1):
            if (int(p), f) not in at:
                continue
            ball_xy, player_xy = index.at(int(p), f, player)
            at[(int(p), f)] = (
                player
                if (ball_xy and player_xy and np.hypot(ball_xy[0] - player_xy[0], ball_xy[1] - player_xy[1]) <= prox_m)
                else None
            )
    return "proximity_fallback", pd.DataFrame(
        [{"period_id": p, "frame_id": f, "possessor": v} for (p, f), v in at.items()],
        columns=["period_id", "frame_id", "possessor"],
    )
