"""Utility functions for silly_kicks.tracking.

Includes:
  - _derive_speed: per-row derived speed where provider doesn't supply it
  - play_left_to_right: tracking-variant L-to-R direction normalization
  - link_actions_to_frames: action <-> frame 1:1 nearest-time linkage
  - slice_around_event: action <-> frame 1:many windowed slice
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import LinkReport


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
    """Mirror tracking frames so the home team attacks left-to-right in every period.

    Operates on long-form rows (player rows AND ball rows). For rows where
    ``team_attacking_direction == "rtl"``, mirrors x and y around the SPADL
    pitch center (105/2, 68/2). Sets ``team_attacking_direction = "ltr"`` for
    flipped rows. NaN coordinates pass through unchanged.

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS.
    home_team_id : int | str
        ID of the home team. Currently reserved (the flip decision is taken
        from ``team_attacking_direction`` which adapters precompute); kept in
        signature for API parity with ``spadl.utils.play_left_to_right`` and
        to support future direction-from-roster-only callers.

    Returns
    -------
    pd.DataFrame
        Frames with x/y mirrored where direction was "rtl" and
        ``team_attacking_direction`` reset to "ltr" on flipped rows.

    Examples
    --------
    Normalize tracking frames so the home team always attacks left-to-right::

        from silly_kicks.tracking import sportec
        from silly_kicks.tracking.utils import play_left_to_right
        frames, _ = sportec.convert_to_frames(
            raw, home_team_id="DFL-CLU-A", home_team_start_left=True,
        )
        ltr_frames = play_left_to_right(frames, home_team_id="DFL-CLU-A")
        # All rows now have team_attacking_direction == "ltr".
    """
    _ = home_team_id  # reserved
    out = frames.copy()
    flip_mask = (out["team_attacking_direction"] == "rtl").to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]
    out.loc[flip_mask, "team_attacking_direction"] = "ltr"
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

    a = actions[["action_id", "period_id", "time_seconds"]].rename(
        columns={"time_seconds": "action_time"},
    )
    merged = frames.merge(a, on="period_id", how="inner")
    delta = merged["time_seconds"] - merged["action_time"]
    in_window = (delta >= -pre_seconds) & (delta <= post_seconds)
    out = merged.loc[in_window].copy()
    out["time_offset_seconds"] = (out["time_seconds"] - out["action_time"]).astype("float64")
    out = out.drop(columns=["action_time"])
    return out.reset_index(drop=True)
