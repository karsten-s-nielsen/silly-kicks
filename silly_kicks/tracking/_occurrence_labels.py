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


def _occurrence_censor_mask(
    frames_index: pd.DataFrame,
    uncertain_events: pd.DataFrame,
    *,
    horizon: float,
    frame_team_col: str,
    event_team_col: str = "team_id",
) -> np.ndarray:
    """Boolean per frame: True where the frame must be CENSORED (TF-46 Arm 1 Part 2, spec section 18.2.1).

    An event with an unvalidated time or type is NOT dropped from the label build -- dropping it would turn
    the frames before it into false negatives. Instead, every frame of that event's team in
    ``[t_lo - horizon, t_hi]`` is censored (removed from training and evaluation). ``uncertain_events`` carries
    ``t_lo`` / ``t_hi`` (the uncertainty interval), ``game_id``, ``period_id`` and the team column.
    """
    censor = np.zeros(len(frames_index), dtype=bool)
    if frames_index.empty or uncertain_events is None or uncertain_events.empty:
        return censor
    from silly_kicks.id_compat import canonical_id

    fg = frames_index["game_id"].astype(object).to_numpy()
    fp = frames_index["period_id"].to_numpy()
    ft = frames_index["time_seconds"].to_numpy(dtype=float)
    fteam = np.array([canonical_id(v) for v in frames_index[frame_team_col].astype(object).to_numpy()], dtype=object)
    for _, ev in uncertain_events.iterrows():
        lo, hi = float(ev["t_lo"]), float(ev["t_hi"])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        team = canonical_id(ev[event_team_col])
        same = (fg == ev["game_id"]) & (fp == ev["period_id"]) & (fteam == team)
        censor |= same & (ft >= lo - horizon) & (ft <= hi)
    return censor


def _apply_effective_time(events: pd.DataFrame) -> pd.DataFrame:
    """Substitute ``t_kick`` for ``time_seconds`` where a gated kick time shipped (spec section 18.2.1): a
    gated derived restart uses its resolved kick time as the event time. Other rows keep ``time_seconds``."""
    if "t_kick" not in events.columns or "kick_time_source" not in events.columns:
        return events
    out = events.copy()
    kts = out["kick_time_source"]
    gated = (~kts.isin(["withheld", "none"]) & kts.notna() & out["t_kick"].notna()).to_numpy(dtype=bool)
    if gated.any():
        out.loc[gated, "time_seconds"] = out.loc[gated, "t_kick"].to_numpy()
    return out


def occurrence_labels_and_censor(
    frames_index: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: float,
    frame_team_col: str,
    event_team_col: str = "team_id",
    reads_type: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y, censor)`` under the provenance contract (spec section 18.2.1).

    A VALIDATED event (its time -- and ``type`` when ``reads_type`` -- is validated) labels frames at its
    effective time (``t_kick`` for a gated derived restart). An UNVALIDATED event is NOT dropped (that would
    turn its pre-event frames into false negatives); instead its uncertainty interval ``[t_lo, t_hi]`` censors
    every frame of its team in ``[t_lo - horizon, t_hi]``. The caller drops the censored frames from training
    and evaluation and reports the censored-frame count.
    """
    from silly_kicks.spadl._training_exclusion import provenance_masks, uncertainty_interval

    ev = events.reset_index(drop=True)
    pm = provenance_masks(ev)
    unval = pm["time"].to_numpy(dtype=bool)
    if reads_type:
        unval = unval | pm["type"].to_numpy(dtype=bool)
    ui = uncertainty_interval(ev)
    y = _build_occurrence_labels(
        frames_index,
        _apply_effective_time(ev[~unval]),
        horizon=horizon,
        frame_team_col=frame_team_col,
        event_team_col=event_team_col,
    )
    unc = ev[unval].copy()
    unc["t_lo"] = ui["t_lo"].to_numpy()[unval]
    unc["t_hi"] = ui["t_hi"].to_numpy()[unval]
    censor = _occurrence_censor_mask(
        frames_index, unc, horizon=horizon, frame_team_col=frame_team_col, event_team_col=event_team_col
    )
    return y, censor
