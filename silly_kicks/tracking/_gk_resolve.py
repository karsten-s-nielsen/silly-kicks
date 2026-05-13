"""Frame-based defending-GK resolution (TF-13).

Resolves the defending team's goalkeeper player_id from tracking frames
for every action. Standalone composable utility -- callers use for fillna
on events-based defending_gk_player_id or as direct lookup.

See spec: docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md s2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import link_actions_to_frames


def defending_gk_from_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action defending-GK player_id resolved from tracking frames.

    For each action, links to the nearest frame (within tolerance), finds
    the opposing team's is_goalkeeper=True row, and returns that player_id.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds, team_id.
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link.

    Returns
    -------
    pd.Series
        Aligned with actions.index. dtype matches frames' player_id dtype
        (object for kloppy/sportec, int64/Int64 for Gradient Sports).
        NaN where action couldn't link, no opposing-team GK in linked frame,
        or action.team_id is NaN.

    Examples
    --------
    Fill NaN from events-based GK resolution::

        from silly_kicks.tracking.features import defending_gk_from_frames
        gk_series = defending_gk_from_frames(actions, frames)
        actions["defending_gk_player_id"] = (
            actions["defending_gk_player_id"].fillna(gk_series)
        )

    See NOTICE for full bibliographic citations.
    """
    # Determine output dtype from frames' player_id
    pid_dtype = frames["player_id"].dtype

    n = len(actions)
    out = pd.Series(np.full(n, np.nan), index=actions.index, dtype="object")

    if n == 0 or len(frames) == 0:
        return out

    pointers, _report = link_actions_to_frames(actions, frames, tolerance_seconds=tolerance_seconds)

    # Build lookup: for each (period_id, frame_id), the GK player_id per team
    gk_rows = frames[(frames["is_goalkeeper"] == True) & (~frames["is_ball"])].copy()  # noqa: E712
    if gk_rows.empty:
        return out

    # Join pointers with actions to get action team_id + period_id
    ptr = pointers.merge(
        actions[["action_id", "team_id", "period_id"]],
        on="action_id",
        how="left",
    )
    # Filter to linked actions only
    linked = ptr[ptr["frame_id"].notna()].copy()
    if linked.empty:
        return out

    # Join with GK rows on (period_id, frame_id) to find GKs in linked frame
    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    gk_in_frame = linked.merge(
        gk_rows[["period_id", "frame_id", "team_id", "player_id"]].rename(
            columns={"team_id": "gk_team_id", "player_id": "gk_player_id"}
        ),
        left_on=["period_id", "frame_id_int"],
        right_on=["period_id", "frame_id"],
        how="inner",
    )

    # Filter to opposing team's GK (gk_team_id != action team_id)
    # Handle NaN team_id on actions: comparison with NaN is False -> filtered out
    opposing = gk_in_frame[gk_in_frame["gk_team_id"] != gk_in_frame["team_id"]]

    if opposing.empty:
        return out

    # Deterministic tiebreak: lowest player_id per action
    opposing_sorted = opposing.sort_values("gk_player_id")
    best = opposing_sorted.drop_duplicates("action_id", keep="first")

    # Map back to actions index
    action_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())
    for _, row in best.iterrows():
        aid = row["action_id"]
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = row["gk_player_id"]

    # Cast to match frames dtype if numeric
    if pid_dtype == np.dtype("int64") or str(pid_dtype) == "Int64":
        out = pd.to_numeric(out, errors="coerce")
        if str(pid_dtype) == "Int64":
            out = out.astype("Int64")

    return out
