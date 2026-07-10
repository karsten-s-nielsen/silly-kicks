"""retains() label for the rho classifier (ADR-036 §Part 3, GENUINELY NEW -- not a copy).

Per action, within window_seconds either (a) the actor's team still holds the ball at window end
(no opponent possession boundary intervenes) OR (b) the actor's team takes a shot -> label 1.0;
if the opponent takes over before either -> label 0.0. A window TRUNCATED by end-of-period data
with no decisive event -> NaN (retention was NOT observed; excluded from training) rather than a
falsely-optimistic 1.0. Returns FLOAT (1.0/0.0/NaN). Searchsorted boundary skeleton borrowed from
vaep.labels._scores_time; the retain/loss payload + add_possessions coupling + truncation-NaN are new.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import add_possessions

_SHOT = spadlconfig.actiontype_id["shot"]


def retains(actions: pd.DataFrame, *, window_seconds: float = 10.0) -> pd.Series:
    a = actions
    if "possession_id" not in a.columns:
        a = add_possessions(a)
    team = np.asarray(a["team_id"].values)
    typ = a["type_id"].to_numpy()
    poss = a["possession_id"].to_numpy()
    time_s = np.asarray(a["time_seconds"].values, dtype=np.float64)
    result = np.full(len(a), np.nan, dtype=float)

    group_keys = [k for k in ("game_id", "period_id") if k in a.columns]
    groups = a.groupby(group_keys) if group_keys else [(None, a)]
    for _key, grp in groups:
        idx = np.asarray(grp.index)
        t = time_s[idx]
        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError("time_seconds must be non-decreasing within each (game_id, period_id) group")
        boundaries = np.searchsorted(t, t + window_seconds, side="left")
        t_last = t[-1] if len(t) else 0.0
        for li in range(len(idx)):
            gi = idx[li]
            end = min(boundaries[li], len(idx))
            label: float | None = None
            for lj in range(li + 1, end):
                gj = idx[lj]
                if typ[gj] == _SHOT and team[gj] == team[gi]:
                    label = 1.0  # (b) actor's team shoots -> decisive retain
                    break
                if team[gj] != team[gi] and poss[gj] != poss[gi]:
                    label = 0.0  # opponent possession boundary intervened -> decisive loss
                    break
            if label is None:
                # No decisive event observed. If the FULL window was observable (>= window_seconds of
                # subsequent data), the team retained -> 1.0; if the window was truncated by the end
                # of the (game, period) data, retention was NOT observed -> NaN.
                label = 1.0 if (t_last - t[li]) >= window_seconds - 1e-9 else np.nan
            result[gi] = label
    return pd.Series(result, index=a.index, name="retains")
