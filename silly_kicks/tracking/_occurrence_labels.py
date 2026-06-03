"""Shared time-windowed occurrence label for trained-frame models (xS, xCross).

A frame is positive iff an event by the same (game, period, team) occurs with
``time_seconds`` in ``[t, t + horizon]``. No ``frame_id`` arithmetic (providers are
not frame-contiguous); per-period ``searchsorted`` over sorted ``time_seconds``.

The frames-side team column (``team_in_possession``) and the events-side team column
(``team_id``) differ in the house schema, so both are parameters. ``groupby(dropna=False)``
+ platform-int output mirror the pre-extraction ``build_xshot_labels`` byte-for-byte (R2-L2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_occurrence_labels(
    frames_index: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: float,
    frame_team_col: str,
    event_team_col: str = "team_id",
) -> np.ndarray:
    y = np.zeros(len(frames_index), dtype=int)  # platform int -- matches xS exactly
    if len(events) == 0:
        return y
    ev_groups: dict[tuple, np.ndarray] = {}
    for key, grp in events.groupby(["game_id", "period_id", event_team_col], dropna=False):
        ev_groups[key] = np.sort(grp["time_seconds"].to_numpy(dtype=float))
    gids = frames_index["game_id"].to_numpy()
    pids = frames_index["period_id"].to_numpy()
    tcol = frames_index[frame_team_col].to_numpy()
    ts = frames_index["time_seconds"].to_numpy(dtype=float)
    for i in range(len(frames_index)):
        arr = ev_groups.get((gids[i], pids[i], tcol[i]))
        if arr is None:
            continue
        lo = float(ts[i])
        left = np.searchsorted(arr, lo, side="left")
        if left < len(arr) and arr[left] <= lo + horizon:
            y[i] = 1
    return y
