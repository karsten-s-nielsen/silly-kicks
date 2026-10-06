"""Shared pure reliability-reduce kernels for the owner-run construct-validity drivers.

Single source (the ``_id_compat`` precedent, ``docs/PRIVATE_CONSUMERS.md``) for the reliability
statistics that more than one driver computes -- the TF-52 team-KPI study
(``scripts/validate_team_kpi_reliability.py``) and the TF-62 GK build-up decision battery
(``scripts/validate_gk_decision.py``) both reduce per-shard samples into the SAME ICC(1). Two copies
of ``icc1`` would drift silently; a driver's reliability number must not depend on which file it lives
in. Pure numpy/pandas, no I/O, no corpus dependency -- the CI kernels exercise these directly.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_MIN_TEAMS = 3  # a correlation / ICC below this team count is not reported
_COMPARABILITY_ICC_TOL = 0.20  # max cross-provider ICC spread for a KPI to be flagged poolable


def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1): between-group / total variance. Pure numpy."""
    df = pd.DataFrame({"v": np.asarray(values, dtype="float64"), "g": groups}).dropna(subset=["v"])
    k = df["g"].nunique()
    n = len(df)
    if k < 2 or n <= k:
        return float("nan")
    grand = df["v"].mean()
    gm = df.groupby("g")["v"]
    ni = gm.count().to_numpy(dtype="float64")
    mi = gm.mean().to_numpy()
    ssb = float(np.sum(ni * (mi - grand) ** 2))
    ssw = float(np.sum((df["v"].to_numpy() - df.groupby("g")["v"].transform("mean").to_numpy()) ** 2))
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - np.sum(ni**2) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return (msb - msw) / denom if denom > 0 else float("nan")


def _stable_half(game_id) -> int:
    """Deterministic 0/1 split of a match id (stable across runs / platforms)."""
    return int(hashlib.sha256(str(game_id).encode("utf-8")).hexdigest(), 16) % 2


def split_half_reliability(samples: pd.DataFrame, kpi: str, *, team_col="team_id", id_col="game_id") -> dict:
    """Split each team's matches odd/even, mean the KPI per half, Pearson r of the two halves across teams."""
    df = samples[[team_col, id_col, kpi]].dropna()
    if df.empty:
        return {"r": float("nan"), "n_teams": 0}
    df = df.assign(_h=df[id_col].map(_stable_half))
    means = df.groupby([team_col, "_h"])[kpi].mean().unstack("_h")
    if 0 not in means.columns or 1 not in means.columns:
        return {"r": float("nan"), "n_teams": 0}
    pair = means.dropna(subset=[0, 1])
    if len(pair) < _MIN_TEAMS or pair[0].std() == 0 or pair[1].std() == 0:
        return {"r": float("nan"), "n_teams": len(pair)}
    r = float(np.corrcoef(pair[0].to_numpy(), pair[1].to_numpy())[0, 1])
    return {"r": r, "n_teams": len(pair)}


def type_ii_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Standardised / reduced major-axis (SMA / RMA) regression slope = sign(corr) * sd(y) / sd(x).

    NOT orthogonal (Deming) major-axis regression -- ``sd(y) / sd(x)`` is the RMA/SMA estimator (it is
    scale-free in each axis' own units), which is the Type-II slope reported for a symmetric reliability
    relationship (A-36; the historical "orthogonal" label was wrong, the arithmetic is unchanged and pinned
    by ``test_type_ii_slope_is_the_sma_rma_value``). The name is kept for its downstream consumers.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < _MIN_TEAMS or x.std() == 0 or y.std() == 0:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return float(np.sign(r) * y.std() / x.std())


def compare_providers(reports: dict[str, dict]) -> dict:
    """Per-KPI cross-provider comparability: POOLABLE iff every provider reports finite same-sign ICC
    within ``_COMPARABILITY_ICC_TOL``. ``reports`` maps provider -> its ``verdicts`` dict."""
    providers = list(reports)
    if len(providers) < 2:
        return {"n_providers": len(providers), "per_kpi": {}}
    kpis: set[str] = set()
    for r in reports.values():
        kpis |= set(r.get("reliability", {}).get("per_kpi", {}))
    out: dict = {}
    for k in sorted(kpis):
        vals = {p: reports[p].get("reliability", {}).get("per_kpi", {}).get(k, {}) for p in providers}
        iccs = [v.get("icc") for v in vals.values()]
        finite = [x for x in iccs if x is not None and np.isfinite(x)]
        poolable = (
            len(finite) == len(providers)
            and all(np.sign(finite[0]) == np.sign(x) for x in finite)
            and (max(finite) - min(finite)) <= _COMPARABILITY_ICC_TOL
        )
        out[k] = {
            "providers": {
                p: {"icc": vals[p].get("icc"), "split_half_r": vals[p].get("split_half_r")} for p in providers
            },
            "icc_spread": float(max(finite) - min(finite)) if len(finite) == len(providers) else None,
            "poolable": bool(poolable),
        }
    return {"n_providers": len(providers), "per_kpi": out}
