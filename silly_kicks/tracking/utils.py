"""Utility functions for silly_kicks.tracking.

Includes:
  - _derive_speed: per-row derived speed where provider doesn't supply it
  - play_left_to_right: tracking-variant L-to-R direction normalization
  - link_actions_to_frames: action <-> frame 1:1 nearest-time linkage
  - slice_around_event: action <-> frame 1:many windowed slice
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .schema import LinkReport


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
    home_player_mask = (~is_ball) & (out["team_id"] == home_team_id)
    home_rtl_mask = home_player_mask & (out["team_attacking_direction"] == "rtl")
    home_rtl_idx = np.flatnonzero(home_rtl_mask.to_numpy())
    rtl_periods = set(out["period_id"].iloc[home_rtl_idx].unique())

    if not rtl_periods:
        return out  # Already period-normalized; no-op

    # Flip ALL rows (player + ball) in periods where home attacks RTL
    period_flip = out["period_id"].isin(rtl_periods).to_numpy()
    out.loc[period_flip, "x"] = 105.0 - out.loc[period_flip, "x"]
    out.loc[period_flip, "y"] = 68.0 - out.loc[period_flip, "y"]

    # Swap direction labels for player rows in flipped periods
    player_in_flip = period_flip & (~is_ball).to_numpy()
    old_dir = out.loc[player_in_flip, "team_attacking_direction"].copy()
    out.loc[player_in_flip, "team_attacking_direction"] = old_dir.map({"ltr": "rtl", "rtl": "ltr"})
    return out


def link_actions_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    tolerance_seconds: float = 0.2,
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
    )
    return pointers, report


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
    long = pointer_with_period.merge(
        frames,
        on=["period_id", "frame_id"],
        how="inner",
        suffixes=("_action", "_frame"),
    )

    # actor_rows: filter to rows where frame.player_id == action.player_id (and not ball)
    if "player_id_frame" in long.columns:
        actor_mask = (long["player_id_frame"] == long["player_id_action"]) & (~long["is_ball"])
        actor_long = long.loc[actor_mask].copy()
    else:
        actor_long = long.iloc[0:0].copy()

    # Build per-action actor row; left-join on action_id so unlinked actions also appear (with NaN cols)
    actor_rows = pd.DataFrame({"action_id": actions["action_id"]}).merge(actor_long, on="action_id", how="left")

    # opposite_rows_per_action: filter to rows where frame.team_id != action.team_id and not ball
    if "team_id_frame" in long.columns:
        opp_mask = (long["team_id_frame"] != long["team_id_action"]) & (~long["is_ball"])
        opposite = long.loc[opp_mask].copy()
    else:
        opposite = long.iloc[0:0].copy()

    # defending_gk_rows (PR-S21): rows where frame.player_id == action.defending_gk_player_id
    # AND defending_gk_player_id is not NaN AND not ball.
    # Provider-native player_ids may be strings (sportec) or numeric — comparing them
    # element-wise via pd.Series.eq handles both cases; the .notna() mask guards NaN.
    if "defending_gk_player_id" in long.columns and "player_id_frame" in long.columns:
        gk_id_action = long["defending_gk_player_id"]
        pid_frame = long["player_id_frame"]
        gk_mask = (pid_frame == gk_id_action) & gk_id_action.notna() & (~long["is_ball"])
        defending_gk_rows = long.loc[gk_mask].copy()
    else:
        defending_gk_rows = long.iloc[0:0].copy()

    return ActionFrameContext(
        actions=actions,
        pointers=pointers,
        actor_rows=actor_rows,
        opposite_rows_per_action=opposite,
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
    with ``action_id`` and ``time_offset_seconds`` joined in.

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
