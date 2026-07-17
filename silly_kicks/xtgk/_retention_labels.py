"""retains() label for the rho classifier (ADR-036 §Part 3, GENUINELY NEW -- not a copy).

Per action, within window_seconds either (a) the actor's team still holds the ball at window end
(no opponent possession boundary intervenes) OR (b) the actor's team takes a shot -> label 1.0;
if the opponent takes over before either -> label 0.0. A window TRUNCATED by end-of-period data
with no decisive event -> NaN (retention was NOT observed; excluded from training) rather than a
falsely-optimistic 1.0. Returns FLOAT (1.0/0.0/NaN). Searchsorted boundary skeleton borrowed from
vaep.labels._scores_time; the retain/loss payload + add_possessions coupling + truncation-NaN are new.

PR-S117 hardening (the packing secured-seam rules, ADR-039 relay item 1): non_action / foul /
NaN-team rows never DECIDE another row's label (none is a possession-implying ball event --
winning a foul is not losing the ball, and a GS null-actor row must never read as "opponent",
ADR-027), a NaN-team ANCHOR row is itself UNDECIDABLE -> NaN (an unknown team has no knowable
"whose team retained" answer; delta-review-added so the ADR-027 contract holds anchor-side too,
not just decider-side), and the possession-boundary test requires BOTH ids attested (an NA
possession never decides). A
2026-07-17 read-only probe measured these rules as a label NO-OP on both live rho training
cohorts (0/3451 GS + 0/5483 SkillCorner flips -- the gold-mart possession ids stay continuous
through foul rows and carry no NAs), so the bundled rho weights required NO retrain; the rules
protect the add_possessions self-heal path, where the foul-row boundary bias is live. The scan
additionally runs in canonical (time_seconds, action_id) order per group -- the rho loader's
own sort, so a stable no-op on the live cohorts -- making labels input-row-order-insensitive
at time ties (9,649 tied pairs on the GS cohort alone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import add_possessions

_SHOT = spadlconfig.actiontype_id["shot"]
_SKIP_TYPES = frozenset({spadlconfig.actiontype_id["non_action"], spadlconfig.actiontype_id["foul"]})


def _same(a, b) -> bool:
    """Both-present scalar equality (NA never equals anything)."""
    return (not pd.isna(a)) and (not pd.isna(b)) and a == b


def retains(actions: pd.DataFrame, *, window_seconds: float = 10.0) -> pd.Series:
    a = actions
    if "possession_id" not in a.columns:
        a = add_possessions(a)
    team = np.asarray(a["team_id"].values)
    team_na = pd.isna(team)
    typ = a["type_id"].to_numpy()
    poss = a["possession_id"].to_numpy()
    poss_na = pd.isna(poss)
    time_s = np.asarray(a["time_seconds"].values, dtype=np.float64)
    result = np.full(len(a), np.nan, dtype=float)

    group_keys = [k for k in ("game_id", "period_id") if k in a.columns]
    groups = a.groupby(group_keys) if group_keys else [(None, a)]
    for _key, grp in groups:
        # Canonical (time_seconds, action_id) scan order (PR-S117): time ties break
        # deterministically, so labels are input-row-order-INSENSITIVE. This is exactly
        # the rho loader's sort -> a stable no-op on the live cohorts (gate-verified;
        # 9,649 GS time-tie pairs would otherwise expose positional-order sensitivity).
        # NOTE: sorting by bare action_id instead would RAISE on the live GS cohort --
        # the mart's action_id order genuinely disagrees with time_seconds there.
        sort_keys = ["time_seconds"] + (["action_id"] if "action_id" in grp.columns else [])
        idx = np.asarray(grp.sort_values(sort_keys, kind="stable").index)
        t = time_s[idx]
        # Post-sort, the guard is a NaN-time catcher (NaN sorts last -> diff is NaN ->
        # raises); mis-ORDERED input is now canonicalized rather than rejected.
        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError("time_seconds must be non-decreasing within each (game_id, period_id) group")
        boundaries = np.searchsorted(t, t + window_seconds, side="left")
        t_last = t[-1] if len(t) else 0.0
        for li in range(len(idx)):
            gi = idx[li]
            if team_na[gi]:
                continue  # unknown anchor team -> no knowable "whose team retained" -> NaN (ADR-027)
            end = min(boundaries[li], len(idx))
            label: float | None = None
            for lj in range(li + 1, end):
                gj = idx[lj]
                if typ[gj] in _SKIP_TYPES or team_na[gj]:
                    continue  # fouls / non_action / null-actor rows never decide (PR-S117)
                if typ[gj] == _SHOT and _same(team[gj], team[gi]):
                    label = 1.0  # (b) actor's team shoots -> decisive retain
                    break
                if (not _same(team[gj], team[gi])) and not poss_na[gj] and not poss_na[gi] and poss[gj] != poss[gi]:
                    label = 0.0  # opponent possession boundary intervened -> decisive loss
                    break
            if label is None:
                # No decisive event observed. If the FULL window was observable (>= window_seconds of
                # subsequent data), the team retained -> 1.0; if the window was truncated by the end
                # of the (game, period) data, retention was NOT observed -> NaN.
                label = 1.0 if (t_last - t[li]) >= window_seconds - 1e-9 else np.nan
            result[gi] = label
    return pd.Series(result, index=a.index, name="retains")
