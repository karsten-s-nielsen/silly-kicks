"""Utility functions for silly_kicks.tracking.

Includes:
  - _derive_speed: per-row derived speed where provider doesn't supply it
  - play_left_to_right: tracking-variant L-to-R direction normalization
  - link_actions_to_frames: action <-> frame 1:1 nearest-time linkage
  - slice_around_event: action <-> frame 1:many windowed slice
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import (
    _as_bool,
    _raw_comparable,
    align_join_keys,
    canonical_id_series,
    ids_match,
)
from silly_kicks.reflection import TRACKING_REFLECTION_KINDS, reflect

from ._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr
from .schema import IdDtypeDiagnosis, LinkReport, TimeBaseDiagnosis

MISMATCH_OVERLAP_FLOOR: float = 0.2
"""Per-period action/frame range overlap below this is flagged a suspected
time-base mismatch (period-relative vs absolute). Decoupled from the linker's
min_link_rate: this governs the *cause hypothesis*, not the *symptom*. 0.2 is
specific to near-disjoint ranges (the GS bug was ~0.14) and stays quiet on
ordinary sparsity. See ADR-017."""


def filter_extratime_frames(frames: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Drop extra-time periods (3/4) from a frames/events DataFrame.

    **Calibration / sampling ONLY.** Calibration is sample-based, so dropping ET
    frames loses a little signal but never correctness. **Production must NOT
    filter ET** --- it must source ``home_team_start_left_extratime`` from provider
    metadata (e.g. via ``MatchMeta``) and pass it to the converter, validated by
    the public guard :func:`silly_kicks.tracking.require_et_direction`. Dropping ET
    in production would silently discard real match data. See ADR-010 §4 / spec §7.

    A no-op (no warning) when no ET periods are present. When ET periods are
    dropped, emits a ``UserWarning`` naming ``label`` so the skip is auditable.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form frames or events with a period column (``period_id`` for tracking
        frames; ``period`` for some events-input shapes --- both are accepted).
    label : str
        Human-readable context (e.g. ``"gradientsports 10517"``) for the warning.

    Returns
    -------
    pd.DataFrame
        ``frames`` with period ``in {3, 4}`` rows removed (a copy when any were
        dropped; otherwise the input is returned unchanged).

    Examples
    --------
    Drop ET for a calibration sample::

        from silly_kicks.tracking.utils import filter_extratime_frames
        rt_only = filter_extratime_frames(frames, label="gradientsports 10517")
    """
    period_col = "period_id" if "period_id" in frames.columns else "period"
    et_mask = frames[period_col].isin([3, 4])
    if not et_mask.any():
        return frames
    warnings.warn(
        f"{label}: extra-time periods (period_id in {{3, 4}}) present but dropped for "
        "calibration/sampling. Production must source home_team_start_left_extratime "
        "from metadata (see require_et_direction), not drop ET.",
        UserWarning,
        stacklevel=2,
    )
    return frames.loc[~et_mask].copy()


def _derive_speed(frames: pd.DataFrame) -> pd.DataFrame:
    """Compute speed = sqrt(dx^2 + dy^2) * frame_rate per (period, is_ball, player) group.

    Modifies a copy of ``frames``:
      - Where ``speed`` is NaN, fill with derived value and set
        ``speed_source="derived"``.
      - Where ``speed`` is populated, leave both columns unchanged.
      - First frame of each (player, period) group: speed remains NaN,
        speed_source unchanged (None / NaN).

    Vectorized via groupby+diff. Ball rows are treated as a single logical
    entity (their ``player_id`` is NaN; ``dropna=False`` puts them all in
    one group keyed on ``is_ball=True``).
    """
    out = frames.copy()
    sort_cols = ["period_id", "is_ball", "player_id", "frame_id"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    grp_keys = ["period_id", "is_ball", "player_id"]
    dx = out.groupby(grp_keys, dropna=False)["x"].diff()
    dy = out.groupby(grp_keys, dropna=False)["y"].diff()
    derived = pd.Series(np.sqrt(dx**2 + dy**2) * out["frame_rate"], index=out.index)

    fill_mask = out["speed"].isna() & derived.notna()
    out.loc[fill_mask, "speed"] = derived[fill_mask]
    out.loc[fill_mask, "speed_source"] = "derived"
    return out


def play_left_to_right(frames: pd.DataFrame, home_team_id) -> pd.DataFrame:
    """Normalize tracking frames so the home team attacks left-to-right in every period.

    Performs **per-period** normalization: in any period where the home team's
    ``team_attacking_direction`` is ``"rtl"``, ALL rows in that period (home
    players, away players, AND ball) are mirrored around the SPADL pitch center
    (105/2, 68/2). Direction labels swap (``"ltr"`` <-> ``"rtl"``) for player
    rows in flipped periods; ball direction stays ``None``.

    This ensures all entities remain in a single consistent coordinate frame
    per period, preserving ball-player distances. After the call, the home team
    attacks toward high x in every period; the away team attacks toward low x.

    .. versionchanged:: 3.15.3
       Changed from per-team flip (broke ball-player spatial relationships) to
       per-period flip. See CHANGELOG for migration notes.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS.
    home_team_id : int | str
        ID of the home team. Used to identify which periods need flipping
        (periods where home-team player rows have direction ``"rtl"``).

    Returns
    -------
    pd.DataFrame
        Frames with per-period normalization applied. Home-team player rows
        have ``team_attacking_direction = "ltr"``; away-team rows have
        ``"rtl"``; ball rows have ``None``.

    Examples
    --------
    Normalize tracking frames so the home team always attacks left-to-right::

        from silly_kicks.tracking import sportec
        from silly_kicks.tracking.utils import play_left_to_right
        frames, _ = sportec.convert_to_frames(
            raw, home_team_id="DFL-CLU-A", home_team_start_left=True,
        )
        ltr_frames = play_left_to_right(frames, home_team_id="DFL-CLU-A")
        # Home-team rows now have team_attacking_direction == "ltr" in
        # all periods; away-team rows have "rtl"; ball rows have None.
    """
    out = frames.copy()
    if len(out) == 0 or "is_ball" not in out.columns or "team_id" not in out.columns:
        return out

    # Identify periods where the home team has "rtl" direction → need flipping
    is_ball = out["is_ball"].astype(bool)
    home_player_mask = (~is_ball) & ids_match(out["team_id"], home_team_id)
    home_rtl_mask = home_player_mask & (out["team_attacking_direction"] == "rtl")
    home_rtl_idx = np.flatnonzero(home_rtl_mask.to_numpy())
    rtl_periods = set(out["period_id"].iloc[home_rtl_idx].unique())

    if not rtl_periods:
        return out  # Already period-normalized; no-op

    period_flip = out["period_id"].isin(rtl_periods).to_numpy()

    # ADR-045: reflect by DECLARED KIND. Previously this transformed x/y only, so vx/vy
    # (a vector), x_smoothed/y_smoothed (a point pair) and the direction label rode through
    # untransformed -- none is in TRACKING_FRAMES_COLUMNS, so all were invisible to a
    # schema-driven author. `direction_label` handles the ltr<->rtl swap: ball rows carry a
    # null label and _DIRECTION_SWAP.get(None, None) is None, so the swap is already a no-op
    # on them and no player/ball split is needed.
    return reflect(out, period_flip, kinds=TRACKING_REFLECTION_KINDS)


_ORIENT_REQUIRED_COLUMNS = ("x", "y", "team_id", "period_id", "is_ball", "team_attacking_direction")


def orient_frames_to_ltr(
    frames: pd.DataFrame,
    *,
    home_team_id,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> pd.DataFrame:
    """Orient *unlabeled* absolute-orientation tracking frames into the canonical
    home-attacks-right (LTR) frame, per period.

    This is the unlabeled-input sibling of :func:`play_left_to_right`. It populates
    ``team_attacking_direction`` from ``home_team_start_left`` (the physical pre-flip
    direction) and then per-period flips so the home team attacks x=105 in every
    period and the away team attacks x=0 --- byte-identical to
    ``convert_to_frames(output_convention="ltr")`` and exactly the convention the
    per-action geometry layer (ADR-028) expects.

    Intended for consumers that build frames from a non-kloppy source (e.g. the
    lakehouse metrica/skillcorner bronze builders) in absolute orientation. For frames
    that ALREADY carry a populated ``team_attacking_direction`` (labeled, e.g.
    ``kloppy.convert_to_frames(output_convention="absolute_frame")`` output), use
    :func:`play_left_to_right` directly --- this helper raises on labeled input.

    Parameters
    ----------
    frames : pd.DataFrame
        Unlabeled absolute tracking frames. Required columns: ``x``, ``y``,
        ``team_id``, ``period_id``, ``is_ball``, ``team_attacking_direction`` (which
        must be all-null on entry). ``team_id`` may be any dtype --- comparisons route
        through the ADR-019 dtype-safe ``ids_match``.
    home_team_id : int | str
        Identifies the home team in ``team_id``. The caller derives this; silly-kicks
        does not infer it.
    home_team_start_left : bool
        True iff the home team's own goal is on the left (x=0) in period 1, i.e. it
        attacks toward x=105 in period 1. Source of truth for the orientation; the
        helper is only as correct as this flag (validate it per game --- see ADR-029).
    home_team_start_left_extratime : bool | None, default None
        Required only when ET periods (3/4) are present.

    Returns
    -------
    pd.DataFrame
        A new DataFrame in home-attacks-right convention. Not idempotent --- a second
        call raises (the first populated ``team_attacking_direction``).

    Raises
    ------
    ValueError
        Missing required columns; ``team_attacking_direction`` non-null on entry (use
        ``play_left_to_right``); ``home_team_id`` matches zero player rows; ET periods
        present without ``home_team_start_left_extratime``.

    See ADR-029 for the single-source-of-truth orientation contract.

    Examples
    --------
    Orient absolute metrica/skillcorner frames into the canonical LTR convention::

        from silly_kicks.tracking import orient_frames_to_ltr
        ltr_frames = orient_frames_to_ltr(
            abs_frames, home_team_id=57, home_team_start_left=True,
        )
    """
    out = frames.copy()
    if len(out) == 0:
        return out

    missing = [c for c in _ORIENT_REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"orient_frames_to_ltr: frames missing required columns: {missing}")

    is_ball = out["is_ball"].astype(bool)
    players = out[~is_ball]
    if players.empty:
        return out

    # C2: labeled-input guard. Unlabeled absolute frames carry an all-null direction;
    # any non-null means the frames are already labeled -> route to play_left_to_right.
    if out["team_attacking_direction"].notna().any():
        raise ValueError(
            "orient_frames_to_ltr: frames already carry a populated "
            "team_attacking_direction (labeled). This helper is for UNLABELED absolute "
            "frames; use silly_kicks.tracking.play_left_to_right for labeled frames."
        )

    # C1: zero-match guard (ADR-019 dtype-safe compare). Zero home-player match means
    # play_left_to_right cannot identify flip periods -> definitely-wrong output.
    is_home = ids_match(players["team_id"], home_team_id).fillna(False)
    if not bool(is_home.any()):
        raise ValueError(
            f"orient_frames_to_ltr: home_team_id={home_team_id!r} matched ZERO player rows "
            "(id dtype mismatch vs frame team_id?) -- orientation would be wrong."
        )

    from .direction import compute_attacking_direction, require_et_direction

    require_et_direction(out["period_id"], home_team_start_left_extratime, source="orient_frames_to_ltr")

    out["team_attacking_direction"] = compute_attacking_direction(
        team_id=out["team_id"],
        period_id=out["period_id"],
        is_ball=out["is_ball"],
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    return play_left_to_right(out, home_team_id)


def link_actions_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    tolerance_seconds: float = 0.2,
    *,
    min_link_rate: float = 0.5,
    on_low_coverage: Literal["warn", "raise", "ignore"] = "warn",
) -> tuple[pd.DataFrame, LinkReport]:
    """Link each action to the nearest tracking frame in time within tolerance.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with at least ``action_id``, ``period_id``,
        ``time_seconds``.
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS.
        Multiple rows per (period_id, frame_id) --- internally
        deduplicated to one row per frame before merge.
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link. NaN frame_id otherwise.
    min_link_rate : float, default 0.5
        Per-period link-rate floor for the coverage guard. Evaluated
        **per period** (the worst period), never the match aggregate -- a
        match-aggregate floor would launder a catastrophically-unlinked period
        behind a healthy one (e.g. GS 10503: 60.6% whole-match vs 19% in p2).
        0.5 fires on structural defects (which crater coverage to 13-19%) while
        staying quiet on legitimate sparsity (0.7-0.95). Tighten for stricter
        consumers. See ADR-017.
    on_low_coverage : {"warn", "raise", "ignore"}, default "warn"
        Policy when any period's link rate is below ``min_link_rate``.
        ``"warn"`` emits one ``UserWarning`` per offending period (low coverage
        is a quality continuum, not a structurally-impossible input, so the
        default does not raise); ``"raise"`` raises ``ValueError``; ``"ignore"``
        is silent (the report is still populated). The message carries the
        per-period rate, unlinked count, and -- when the period's action/frame
        ranges are near-disjoint -- a suspected time-base mismatch hint.

    Notes
    -----
    **Time-base contract.** ``actions`` and ``frames`` MUST share a per-period
    time base. silly_kicks' canonical convention is that ``time_seconds`` is
    **seconds since the start of its period, resetting to 0 each period** --- NOT
    absolute match-clock / continuous across periods. Linking is per-period
    (``merge_asof`` within each ``period_id``), so cross-period continuity is
    irrelevant, but a period whose actions and frames use different origins
    (e.g. period-relative frames vs absolute actions) will not link and trips
    ``on_low_coverage``. For consumers that pre-filter / window / batch actions
    by time before linking, call :func:`validate_time_base` on the **unfiltered**
    inputs --- the guard here cannot see actions a pre-filter already dropped.
    See ADR-017.

    Returns
    -------
    pointers : pd.DataFrame
        Columns:
        - ``action_id`` (int64)
        - ``frame_id`` (Int64; NaN if unlinked)
        - ``time_offset_seconds`` (float64; ``action_time - frame_time``;
          NaN if unlinked)
        - ``n_candidate_frames`` (int64; frames in same period within
          tolerance)
        - ``link_quality_score`` (float64; ``1 - |dt|/tolerance``;
          NaN if unlinked)
    report : LinkReport
        Audit trail.

    Examples
    --------
    Find the nearest frame for each SPADL action and inspect link rate::

        from silly_kicks.tracking.utils import link_actions_to_frames
        pointers, report = link_actions_to_frames(
            actions, frames, tolerance_seconds=0.1,
        )
        assert report.link_rate >= 0.95
    """
    if len(actions) == 0:
        empty = pd.DataFrame(
            {
                "action_id": pd.Series([], dtype="int64"),
                "frame_id": pd.Series([], dtype="Int64"),
                "time_offset_seconds": pd.Series([], dtype="float64"),
                "n_candidate_frames": pd.Series([], dtype="int64"),
                "link_quality_score": pd.Series([], dtype="float64"),
            }
        )
        return empty, LinkReport(0, 0, 0, 0, {}, 0.0, tolerance_seconds)

    frame_index = (
        frames[["period_id", "frame_id", "time_seconds", "source_provider"]]
        .drop_duplicates(["period_id", "frame_id"])
        .sort_values(["period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )

    actions_sorted = (
        actions[["action_id", "period_id", "time_seconds"]]
        .sort_values(["period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )

    parts: list[pd.DataFrame] = []
    for period, a_group in actions_sorted.groupby("period_id", sort=False):
        f_group = frame_index[frame_index["period_id"] == period]
        if len(f_group) == 0:
            unlinked = a_group.copy()
            unlinked["frame_id"] = pd.array([pd.NA] * len(a_group), dtype="Int64")
            unlinked["frame_time"] = float("nan")
            unlinked["source_provider"] = None
            parts.append(unlinked)
            continue
        merged = pd.merge_asof(
            a_group.sort_values("time_seconds"),
            f_group[["frame_id", "time_seconds", "source_provider"]]
            .rename(columns={"time_seconds": "frame_time"})
            .sort_values("frame_time"),
            left_on="time_seconds",
            right_on="frame_time",
            direction="nearest",
            tolerance=tolerance_seconds,  # type: ignore[arg-type]  # numeric-on-column accepts float; pandas-stubs limitation
        )
        parts.append(merged)

    merged_all = pd.concat(parts, ignore_index=True)

    time_offset = merged_all["time_seconds"] - merged_all["frame_time"]
    quality = 1.0 - time_offset.abs() / tolerance_seconds
    quality = quality.where(merged_all["frame_id"].notna(), other=float("nan"))
    time_offset = time_offset.where(merged_all["frame_id"].notna(), other=float("nan"))

    n_cand = _count_candidates_within_tolerance(
        actions_sorted,
        frame_index,
        tolerance_seconds,
    )

    pointers = pd.DataFrame(
        {
            "action_id": merged_all["action_id"].astype("int64"),
            "frame_id": merged_all["frame_id"].astype("Int64"),
            "time_offset_seconds": time_offset.astype("float64"),
            "n_candidate_frames": n_cand.astype("int64"),
            "link_quality_score": quality.astype("float64"),
        }
    )

    per_period_link_rate: dict[int, float] = {
        int(p): float(s.notna().mean())  # type: ignore[arg-type]  # groupby key is Hashable; period_id is int
        for p, s in merged_all.groupby("period_id")["frame_id"]
    }

    n_in = len(actions)
    n_linked = int(pointers["frame_id"].notna().sum())
    n_unlinked = n_in - n_linked
    n_multi = int((pointers["n_candidate_frames"] > 1).sum())
    per_provider: dict[str, float] = {}
    if n_linked > 0:
        provider_col = merged_all.loc[merged_all["frame_id"].notna(), "source_provider"]
        for prov, count in provider_col.value_counts().items():
            per_provider[str(prov)] = float(count) / n_in
    max_off = float(time_offset.abs().max()) if n_linked > 0 else 0.0

    report = LinkReport(
        n_actions_in=n_in,
        n_actions_linked=n_linked,
        n_actions_unlinked=n_unlinked,
        n_actions_multi_candidate=n_multi,
        per_provider_link_rate=per_provider,
        max_time_offset_seconds=max_off,
        tolerance_seconds=tolerance_seconds,
        per_period_link_rate=per_period_link_rate,
    )
    _enforce_link_coverage(
        actions,
        frames,
        report,
        min_link_rate=min_link_rate,
        on_low_coverage=on_low_coverage,
    )
    return pointers, report


def _enforce_link_coverage(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    report: LinkReport,
    *,
    min_link_rate: float,
    on_low_coverage: Literal["warn", "raise", "ignore"],
) -> None:
    """Per-period low-coverage policy for link_actions_to_frames. See ADR-017."""
    if on_low_coverage == "ignore" or not report.per_period_link_rate:
        return
    offending = {p: r for p, r in report.per_period_link_rate.items() if r < min_link_rate}
    if not offending:
        return

    diag = _diagnose_time_base(actions, frames)  # lazy: only on a tripped guard
    suspected = set(diag.suspected_mismatch_periods)
    worst_first = sorted(offending, key=lambda p: offending[p])

    def _line(p: int) -> str:
        n_total = int((actions["period_id"] == p).sum())
        n_unlinked = round((1.0 - offending[p]) * n_total)
        msg = (
            f"link_actions_to_frames: period {p} link_rate {offending[p]:.2f} "
            f"({n_total} actions, {n_unlinked} unlinked) below min_link_rate {min_link_rate:g}."
        )
        if p in suspected:
            a_min, a_max = diag.per_period_action_range[p]
            frng = diag.per_period_frame_range.get(p)
            frames_desc = f"frames [{frng[0]:g}, {frng[1]:g}]" if frng else "no frames"
            msg += (
                f" period {p}: actions [{a_min:g}, {a_max:g}] vs {frames_desc} — "
                f"near-disjoint (overlap {diag.per_period_overlap_fraction[p]:.2f}); "
                "suspected period-relative/absolute time-base mismatch. "
                "See the time-base contract in the docstring."
            )
        return msg

    if on_low_coverage == "raise":
        raise ValueError(" ".join(_line(p) for p in worst_first))
    for p in worst_first:  # one warning per offending period (deduped per period)
        # stacklevel=3: warn site is _enforce_link_coverage (1) -> link_actions_to_frames (2)
        # -> the user's call site (3). NOT 2 — that would blame the linker's own internals.
        # (Contrast validate_time_base, which warns in its own body, so stacklevel=2 is correct
        # there. The project's "stacklevel=2" convention means "point at the user"; the literal
        # value depends on call-nesting depth.)
        warnings.warn(_line(p), UserWarning, stacklevel=3)


def _diagnose_time_base(actions: pd.DataFrame, frames: pd.DataFrame) -> TimeBaseDiagnosis:
    """Pure per-period action-vs-frame time-range diagnosis. No warn/raise/I/O.

    Vectorized: per-period ranges via a single groupby().agg on each side
    (NOT the iterrows pattern in _count_candidates_within_tolerance). NaN
    time_seconds rows are dropped before computing ranges.
    """
    a = actions[["period_id", "time_seconds"]].dropna(subset=["time_seconds"])
    f = frames[["period_id", "time_seconds"]].dropna(subset=["time_seconds"])
    a_rng = a.groupby("period_id")["time_seconds"].agg(["min", "max"])
    f_rng = f.groupby("period_id")["time_seconds"].agg(["min", "max"])

    per_action: dict[int, tuple[float, float]] = {}
    per_frame: dict[int, tuple[float, float]] = {}
    overlap_frac: dict[int, float] = {}
    suspected: list[int] = []

    for p in a_rng.index:
        # pandas-stubs types .loc[scalar, col] as Scalar (incl. complex); these are real floats.
        a_min, a_max = float(a_rng.loc[p, "min"]), float(a_rng.loc[p, "max"])  # type: ignore[arg-type]
        per_action[int(p)] = (a_min, a_max)
        if p in f_rng.index:
            f_min, f_max = float(f_rng.loc[p, "min"]), float(f_rng.loc[p, "max"])  # type: ignore[arg-type]
            per_frame[int(p)] = (f_min, f_max)
            span = a_max - a_min
            if span <= 0.0:  # degenerate single-point action span
                frac = 1.0 if (f_min <= a_min <= f_max) else 0.0
            else:
                overlap = max(0.0, min(a_max, f_max) - max(a_min, f_min))
                frac = overlap / span
        else:
            frac = 0.0  # actions in this period but no frames at all
        overlap_frac[int(p)] = frac
        if frac < MISMATCH_OVERLAP_FLOOR:
            suspected.append(int(p))

    suspected.sort(key=lambda p: overlap_frac[p])  # worst (lowest overlap) first
    message = _format_diagnosis(per_action, per_frame, overlap_frac, tuple(suspected))
    return TimeBaseDiagnosis(per_action, per_frame, overlap_frac, tuple(suspected), message)


def _format_diagnosis(
    per_action: dict[int, tuple[float, float]],
    per_frame: dict[int, tuple[float, float]],
    overlap_frac: dict[int, float],
    suspected: tuple[int, ...],
) -> str:
    """Human-readable summary; enumerates suspected periods worst-first."""
    if not suspected:
        return "no time-base mismatch detected (all periods overlap)"
    parts = []
    for p in suspected:
        a_min, a_max = per_action[p]
        if p in per_frame:
            f_min, f_max = per_frame[p]
            frames_desc = f"frames [{f_min:g}, {f_max:g}]"
        else:
            frames_desc = "no frames"
        parts.append(
            f"period {p}: actions [{a_min:g}, {a_max:g}] vs {frames_desc} "
            f"— near-disjoint (overlap {overlap_frac[p]:.2f})"
        )
    return (
        "; ".join(parts) + "; suspected period-relative/absolute time-base mismatch "
        "(time_seconds must be period-relative; see the time-base contract)"
    )


def validate_time_base(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    on_mismatch: Literal["warn", "raise", "ignore"] = "raise",
) -> TimeBaseDiagnosis:
    """Pre-link assertion that actions + frames share a per-period time base.

    silly_kicks' canonical ``time_seconds`` convention is **period-relative**
    (resets to 0 each period; see the link_actions_to_frames docstring). This
    helper runs the pure per-period range diagnosis and, on a suspected
    mismatch, raises (default), warns, or returns silently.

    **This is the primary guard for any consumer that pre-filters / windows /
    batches actions by time before linking.** ``link_actions_to_frames``'s own
    ``on_low_coverage`` guard only sees the actions that reach it -- a pre-filter
    that drops out-of-range actions upstream leaves the linker with
    ~100%-linkable survivors and the guard silent (exactly how the original GS
    period-2 bug stayed invisible). Call this on the **unfiltered** inputs at
    work-unit entry. See ADR-017.

    Parameters
    ----------
    actions, frames : pd.DataFrame
        SPADL actions / long-form tracking frames (need ``period_id`` +
        ``time_seconds``).
    on_mismatch : {"raise", "warn", "ignore"}, default "raise"
        Policy when a suspected mismatch is found. Default ``"raise"`` -- an
        explicitly-invoked assertion should fail loud (the asymmetry with the
        linker's ``warn`` default is intentional).

    Returns
    -------
    TimeBaseDiagnosis
        The per-period diagnosis (returned in all policies, including "raise"
        when no mismatch is found).

    Raises
    ------
    ValueError
        If ``on_mismatch="raise"`` and a suspected mismatch is found.

    Examples
    --------
    >>> from silly_kicks.tracking import validate_time_base
    >>> diag = validate_time_base(actions, frames, on_mismatch="warn")  # doctest: +SKIP
    >>> diag.has_suspected_mismatch  # doctest: +SKIP
    """
    diag = _diagnose_time_base(actions, frames)
    if diag.has_suspected_mismatch:
        if on_mismatch == "raise":
            raise ValueError(f"validate_time_base: {diag.message}")
        if on_mismatch == "warn":
            warnings.warn(f"validate_time_base: {diag.message}", UserWarning, stacklevel=2)
    return diag


_ID_COLUMNS = ("player_id", "team_id", "defending_gk_player_id")


def _diagnose_id_dtypes(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    home_team_id=None,
) -> IdDtypeDiagnosis:
    """Pure action-vs-frame id-dtype compatibility diagnosis (ADR-019)."""
    per_column: dict[str, tuple[str, str]] = {}
    coercion: list[str] = []
    for col in _ID_COLUMNS:
        if col in actions.columns and col in frames.columns:
            ad, fd = actions[col].dtype, frames[col].dtype
            per_column[col] = (str(ad), str(fd))
            if ad.kind != fd.kind:
                coercion.append(col)
    ht_dtype = None
    ht_coerce = False
    if home_team_id is not None and "team_id" in frames.columns:
        ht_dtype = type(home_team_id).__name__
        scal_kind = pd.Series([home_team_id]).dtype.kind
        ht_coerce = scal_kind != frames["team_id"].dtype.kind
    bits = [f"{c}: action={per_column[c][0]} vs frame={per_column[c][1]}" for c in coercion]
    if ht_coerce:
        bits.append(f"home_team_id={ht_dtype} vs frame team_id={frames['team_id'].dtype}")
    message = "id dtype mismatch (coercion applied at seams): " + "; ".join(bits) if bits else "id dtypes compatible"
    return IdDtypeDiagnosis(per_column, tuple(coercion), ht_dtype, ht_coerce, message)


def validate_id_dtypes(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id=None,
    on_mismatch: Literal["warn", "raise", "ignore"] = "raise",
) -> IdDtypeDiagnosis:
    """Pre-flight guard that actions + frames share comparable id dtypes (ADR-019).

    The tracking-feature seams coerce id dtypes transparently, so this is an OPT-IN
    loud guard, not a required call. Mirrors :func:`validate_time_base`: ``on_mismatch``
    defaults to ``"raise"`` (an explicitly-invoked assertion fails loud); ``"warn"`` /
    ``"ignore"`` available. The diagnosis is returned under all policies.

    Parameters
    ----------
    actions, frames : pd.DataFrame
        SPADL actions / long-form tracking frames (id columns ``player_id`` /
        ``team_id`` / optional ``defending_gk_player_id``).
    home_team_id : int | str | None
        Optional scalar to check against the frame ``team_id`` dtype (the
        scalar-arg failure axis).
    on_mismatch : {"raise", "warn", "ignore"}, default "raise"
        Policy when an id-dtype mismatch is found.

    Returns
    -------
    IdDtypeDiagnosis
        The diagnosis (returned in all policies).

    Examples
    --------
    >>> from silly_kicks.tracking import validate_id_dtypes
    >>> diag = validate_id_dtypes(actions, frames, on_mismatch="warn")  # doctest: +SKIP
    >>> diag.has_mismatch  # doctest: +SKIP
    """
    diag = _diagnose_id_dtypes(actions, frames, home_team_id=home_team_id)
    if diag.has_mismatch:
        if on_mismatch == "raise":
            raise ValueError(f"validate_id_dtypes: {diag.message}")
        if on_mismatch == "warn":
            warnings.warn(f"validate_id_dtypes: {diag.message}", UserWarning, stacklevel=2)
    return diag


def _count_candidates_within_tolerance(
    actions_sorted: pd.DataFrame,
    frame_index: pd.DataFrame,
    tolerance: float,
) -> pd.Series:
    """For each action, count distinct frame_ids within +/-tolerance in same period."""
    counts = np.zeros(len(actions_sorted), dtype="int64")
    for i, row in actions_sorted.iterrows():
        f_period = frame_index[frame_index["period_id"] == row["period_id"]]
        if len(f_period) == 0:
            continue
        in_window = (f_period["time_seconds"] - row["time_seconds"]).abs() <= tolerance
        counts[int(i)] = int(in_window.sum())  # type: ignore[arg-type]  # iterrows returns Hashable label
    return pd.Series(counts, index=actions_sorted.index)


def _resolve_action_frame_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
) -> ActionFrameContext:  # type: ignore[name-defined]  # noqa: F821
    """Build the linked-context structure used by all 4 action_context features.

    Calls link_actions_to_frames once, then derives:
      - actor_rows: one row per action where the actor's player_id appears in the linked frame.
      - opposite_rows_per_action: long-form (action_id, opposite-team frame row) pairs,
        excluding ball rows (is_ball=True).

    After the inner join with frames using ``suffixes=("_action", "_frame")``, the
    overlapping columns ``team_id`` and ``player_id`` are renamed to
    ``team_id_action`` / ``team_id_frame`` and ``player_id_action`` /
    ``player_id_frame``. Per-feature kernels read the suffixed names.

    Internal helper. Public per-feature surface lives in silly_kicks.tracking.features
    and silly_kicks.atomic.tracking.features.

    Examples
    --------
    Build the linked context once and consume across multiple feature kernels::

        from silly_kicks.tracking.utils import _resolve_action_frame_context
        ctx = _resolve_action_frame_context(actions, frames)
        # ctx.actor_rows / ctx.opposite_rows_per_action drive the kernels.
    """
    from .feature_framework import ActionFrameContext  # avoid module-import cycle

    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)

    # Inner-join pointers <-> frames on (period_id, frame_id) to materialize linked frames per action
    projection_cols = ["action_id", "period_id", "team_id", "player_id"]
    if "defending_gk_player_id" in actions.columns:
        projection_cols.append("defending_gk_player_id")
    actions_with_period = actions[projection_cols]
    pointer_with_period = pointers.merge(actions_with_period, on="action_id", how="left", suffixes=("", "_action"))
    # Align id-valued join keys before the merge so a string-id caller does not raise (ADR-019).
    pointer_with_period, frames = align_join_keys(pointer_with_period, frames, ["period_id", "frame_id"])
    long = pointer_with_period.merge(
        frames,
        on=["period_id", "frame_id"],
        how="inner",
        suffixes=("_action", "_frame"),
    )

    # ADR-019: dtype-safe id comparisons at the actor / opponent / GK masks. Canonicalize each
    # suffixed id column AT MOST ONCE and reuse across masks (A1 de-dup); the fast path leaves a
    # same-kind/both-object pair raw-compared at zero cost.
    not_ball = ~long["is_ball"].astype(bool)
    _canon_cache: dict[str, pd.Series] = {}

    def _canon_col(col: str) -> pd.Series:
        # explicit membership check -- NOT setdefault, whose default arg is eagerly
        # evaluated every call and would defeat the de-dup (A1).
        if col not in _canon_cache:
            _canon_cache[col] = canonical_id_series(long[col])
        return _canon_cache[col]

    # `_raw_comparable`, NOT `_directly_comparable`: these compare an ACTION column against a
    # FRAME column, so the two sides come from different sources and a dtype-level answer is not
    # enough. `_directly_comparable` short-circuits object-vs-object to a raw `==`, which is the
    # exact shape that mis-resolved a boxed-numeric object column (an object id column holding
    # 2.0 raw-compares False against the string "2"). `_raw_comparable` probes content first, so
    # these masks agree with `ids_equal`/`ids_differ` on every input.
    def _ids_equal_cols(lcol: str, rcol: str) -> pd.Series:
        a, b = long[lcol], long[rcol]
        if _raw_comparable(a, b):
            return _as_bool((a == b) & a.notna() & b.notna())
        ca, cb = _canon_col(lcol), _canon_col(rcol)
        return _as_bool((ca == cb) & ca.notna() & cb.notna())

    def _ids_differ_cols(lcol: str, rcol: str) -> pd.Series:
        a, b = long[lcol], long[rcol]
        if _raw_comparable(a, b):
            return _as_bool(a.notna() & b.notna() & (a != b))
        ca, cb = _canon_col(lcol), _canon_col(rcol)
        return _as_bool(ca.notna() & cb.notna() & (ca != cb))

    # actor_rows: filter to rows where frame.player_id == action.player_id (and not ball)
    if "player_id_frame" in long.columns:
        actor_mask = _ids_equal_cols("player_id_frame", "player_id_action") & not_ball
        actor_long = long.loc[actor_mask].copy()
    else:
        actor_long = long.iloc[0:0].copy()

    # Build per-action actor row; left-join on action_id so unlinked actions also appear (with NaN cols)
    actor_rows = pd.DataFrame({"action_id": actions["action_id"]}).merge(actor_long, on="action_id", how="left")

    # opposite_rows_per_action: filter to rows where frame.team_id != action.team_id and not ball.
    # ids_differ requires BOTH present, so an unmatched/NaN id is never mis-classified as opponent.
    if "team_id_frame" in long.columns:
        opp_mask = _ids_differ_cols("team_id_frame", "team_id_action") & not_ball
        opposite = long.loc[opp_mask].copy()
    else:
        opposite = long.iloc[0:0].copy()

    # defending_gk_rows (PR-S21): rows where frame.player_id == action.defending_gk_player_id
    # AND defending_gk_player_id is not NaN AND not ball. ids_equal's both-present rule subsumes
    # the explicit .notna() guard.
    if "defending_gk_player_id" in long.columns and "player_id_frame" in long.columns:
        gk_mask = _ids_equal_cols("player_id_frame", "defending_gk_player_id") & not_ball
        defending_gk_rows = long.loc[gk_mask].copy()
    else:
        defending_gk_rows = long.iloc[0:0].copy()

    # ADR-028: re-project the sampled frame positions into each action's LTR frame.
    # Frames are home-attacks-right; actions are per-acting-team-LTR. They are a
    # 180-degree mirror apart for away-team actions. After this, the kernels'
    # hardcoded goal at (105, 34) is correct because the acting team attacks x=105.
    flip = acting_team_attacks_rtl(actions, frames)  # index: actions.index
    # Key by action_id (same precondition as the action_id merges above). Dedupe defensively:
    # gamestate-shifted slots repeat the SAME action (identical game/period/team) so first-wins
    # is correct, and a duplicate index would otherwise make .map() raise.
    flip_by_action = pd.Series(flip.to_numpy(dtype=bool), index=actions["action_id"].to_numpy())
    flip_by_action = flip_by_action[~flip_by_action.index.duplicated(keep="first")]

    def _reproject_rows(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty or "action_id" not in rows.columns:
            return rows
        row_flip = rows["action_id"].map(flip_by_action)
        row_flip = row_flip.fillna(False).astype(bool)
        row_flip.index = rows.index
        # ADR-045: velocities MUST be negated alongside positions. Omitting them made
        # _pressure_bekkers read action-LTR positions against frame-convention velocity,
        # modelling away defenders as running backwards (-38.9% on away actions).
        #
        # x_smoothed/y_smoothed are enumerated too, and that is NOT belt-and-braces:
        # derive_velocities REQUIRES them (preprocess/_velocity.py:41 raises without
        # them), so every frame that carries vx/vy -- i.e. every frame where this fix
        # matters at all -- also carries the smoothed pair. Enumerating x/y/vx/vy alone
        # would leave a mirrored position sitting next to an unmirrored copy of itself,
        # which is D3b reconstituted inside D1's own fix.
        return reproject_to_action_ltr(
            rows,
            row_flip,
            x_cols=["x", "x_smoothed"],
            y_cols=["y", "y_smoothed"],
            vx_cols=["vx"],
            vy_cols=["vy"],
        )

    actor_rows = _reproject_rows(actor_rows)
    opposite = _reproject_rows(opposite)
    defending_gk_rows = _reproject_rows(defending_gk_rows)

    return ActionFrameContext(
        actions=actions,
        pointers=pointers,
        actor_rows=actor_rows,
        opposite_rows_per_action=opposite,
        flip_by_action=flip_by_action,
        defending_gk_rows=defending_gk_rows,
    )


def slice_around_event(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    pre_seconds: float = 0.0,
    post_seconds: float = 0.0,
) -> pd.DataFrame:
    """Return all frames within ``[t - pre_seconds, t + post_seconds]`` per action.

    Constrained to the same period; window does not cross period boundaries.
    Output is long-form (one row per (action_id, frame_id, player_or_ball))
    with ``action_id`` and ``time_offset_seconds`` joined in. Assumes the
    per-period ``time_seconds`` convention (resets each period); see
    :func:`link_actions_to_frames` / ADR-017.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with ``action_id``, ``period_id``, ``time_seconds``.
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS.
    pre_seconds, post_seconds : float
        Window extents on either side of the action time. Both default
        to 0.0, returning frames whose time exactly equals the action
        time (typically yields no rows unless action timestamps line up
        on a frame boundary).

    Returns
    -------
    pd.DataFrame
        Long-form slice with ``action_id`` and
        ``time_offset_seconds = frame_time - action_time``.

    Examples
    --------
    Pull the 0.5 s pre/post window around every shot::

        from silly_kicks.tracking.utils import slice_around_event
        shots = actions[actions["type_id"] == 11]  # shot type
        ctx = slice_around_event(shots, frames, pre_seconds=0.5, post_seconds=0.5)
    """
    if len(actions) == 0 or len(frames) == 0:
        cols = [*frames.columns, "action_id", "time_offset_seconds"]
        return pd.DataFrame(columns=cols)

    parts: list[pd.DataFrame] = []
    a_proj = actions[["action_id", "period_id", "time_seconds"]]

    for period_id, a_grp in a_proj.groupby("period_id", sort=False):
        f_period = frames[frames["period_id"] == period_id]
        if len(f_period) == 0:
            continue

        # One row per unique frame, sorted by time for searchsorted
        frame_index = (
            f_period[["frame_id", "time_seconds"]]
            .drop_duplicates("frame_id")
            .sort_values("time_seconds", kind="mergesort")
            .reset_index(drop=True)
        )
        ft = frame_index["time_seconds"].to_numpy()
        fids = frame_index["frame_id"].to_numpy()

        action_times = a_grp["time_seconds"].to_numpy()
        action_ids = a_grp["action_id"].to_numpy()

        # O(A * log F) instead of O(A * F) cartesian merge
        lo = np.searchsorted(ft, action_times - pre_seconds, side="left")
        hi = np.searchsorted(ft, action_times + post_seconds, side="right")
        counts = hi - lo
        total = counts.sum()
        if total == 0:
            continue

        # Vectorized expansion of (action_id, frame_id, action_time) triples
        a_idx = np.repeat(np.arange(len(action_times)), counts)
        cumstarts = np.empty(len(counts), dtype=np.intp)
        cumstarts[0] = 0
        np.cumsum(counts[:-1], out=cumstarts[1:])
        within = np.arange(total) - np.repeat(cumstarts, counts)
        frame_offsets = np.repeat(lo, counts) + within

        pair_df = pd.DataFrame(
            {
                "action_id": action_ids[a_idx],
                "_frame_id_key": fids[frame_offsets],
                "_action_time": action_times[a_idx],
            }
        )

        merged = pair_df.merge(
            f_period,
            left_on="_frame_id_key",
            right_on="frame_id",
            how="inner",
        )
        merged["time_offset_seconds"] = (merged["time_seconds"] - merged["_action_time"]).astype("float64")
        merged = merged.drop(columns=["_action_time", "_frame_id_key"])
        parts.append(merged)

    if not parts:
        cols = [*frames.columns, "action_id", "time_offset_seconds"]
        return pd.DataFrame(columns=cols)

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# PR-S24 -- TF-6: sync_score per-action tracking<->events sync-quality
# ---------------------------------------------------------------------------


def sync_score(
    links: pd.DataFrame,
    *,
    high_quality_threshold: float = 0.85,
) -> pd.DataFrame:
    """Per-action sync-quality scores (3 aggregations).

    Returns a DataFrame indexed by ``action_id`` with columns:
      - ``sync_score_min`` -- min(link_quality_score) per action.
      - ``sync_score_mean`` -- mean(link_quality_score) per action.
      - ``sync_score_high_quality_frac`` -- fraction of links with
        ``link_quality_score >= high_quality_threshold``.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking.utils import sync_score
    >>> links = pd.DataFrame({
    ...     "action_id": [1, 1],
    ...     "link_quality_score": [0.9, 0.8],
    ...     "frame_id": [10, 11],
    ...     "time_offset_seconds": [0.0, 0.04],
    ...     "n_candidate_frames": [2, 2],
    ... })
    >>> df = sync_score(links, high_quality_threshold=0.85)
    >>> float(df.loc[1, "sync_score_min"])
    0.8
    """
    if "link_quality_score" not in links.columns or "action_id" not in links.columns:
        raise ValueError("sync_score: links must contain 'action_id' and 'link_quality_score'")
    grp = links.groupby("action_id", dropna=False)["link_quality_score"]
    out = pd.DataFrame(
        {
            "sync_score_min": grp.min(),
            "sync_score_mean": grp.mean(),
            "sync_score_high_quality_frac": grp.apply(lambda s: float((s >= high_quality_threshold).mean())),
        }
    )
    return out


def add_sync_score(
    actions: pd.DataFrame,
    links: pd.DataFrame,
    *,
    high_quality_threshold: float = 0.85,
) -> pd.DataFrame:
    """Enrich ``actions`` with three ``sync_score_*`` columns merged on ``action_id``.

    Examples
    --------
    >>> # See tests/test_sync_score.py for runnable example.
    """
    if "action_id" not in actions.columns:
        raise ValueError("add_sync_score: actions must contain 'action_id'")
    scores = sync_score(links, high_quality_threshold=high_quality_threshold)
    return actions.merge(scores, left_on="action_id", right_index=True, how="left")
