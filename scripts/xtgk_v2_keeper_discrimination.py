"""W5 (4.45.0): keeper discrimination -- the real SP5 instrument (Jeff's Bravo/Navas reranking mode).

The program exists to fix v1's keeper NON-discrimination (near-constant per-keeper mean). Here we ask,
on the FAITHFUL metric: does xt_gk_v2 separate keepers where v1 was flat? Descriptive (V_opp fit on the
FULL cohort, R4). Discrimination = ICC on the ACTION-level values grouped by player_key (within-keeper
replication), NOT collapsed per-keeper means (that would be degenerate, R2). Per-keeper mean is used only
for the ranking. CV is reported secondary (unstable near zero mean). Honest-reporting: report whatever it
shows (§3) -- if v2 is still keeper-flat, that is the finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_N = 20  # a-priori: min distributions per keeper for a stable within-keeper term


def icc_one_way(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1) from ACTION-level values grouped by keeper: between-keeper variance
    as a fraction of total. Higher => the metric separates keepers. Partitions variance from the raw
    action-level values (NOT per-group means -- that has no within-group term)."""
    df = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    g_sizes = df.groupby("g")["v"].transform("size")
    df = df[g_sizes >= 2]  # a group needs >=2 actions to contribute a within term
    grp = df.groupby("g")["v"]
    ng, means = grp.count().to_numpy(float), grp.mean().to_numpy(float)
    n, g = len(df), len(ng)
    if g < 2 or n <= g:
        return float("nan")
    grand = df["v"].mean()
    ssb = float((ng * (means - grand) ** 2).sum())
    ssw = float(grp.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum())
    msb, msw = ssb / (g - 1), ssw / (n - g)
    n0 = (n - (ng**2).sum() / n) / (g - 1)  # unbalanced correction
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom != 0 else float("nan")


def keeper_spread(values: np.ndarray, keys: np.ndarray, *, min_n: int = _MIN_N) -> dict:
    """ICC (action-level) + CV (per-keeper means, secondary/unstable) + per-keeper mean ranking."""
    df = pd.DataFrame({"v": np.asarray(values, float), "k": np.asarray(keys)}).dropna()
    cnt = df.groupby("k")["v"].transform("size")
    df = df[cnt >= min_n]
    if df["k"].nunique() < 2:
        return {"icc": float("nan"), "cv": float("nan"), "n_keepers": int(df["k"].nunique()), "ranking": []}
    icc = icc_one_way(df["v"].to_numpy(), df["k"].to_numpy())
    per = df.groupby("k")["v"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    m = per["mean"].to_numpy()
    cv = float(np.std(m) / abs(np.mean(m))) if np.mean(m) != 0 else float("nan")
    ranking = [(str(k), float(r["mean"]), int(r["count"])) for k, r in per.iterrows()]
    return {"icc": icc, "cv": cv, "n_keepers": len(per), "ranking": ranking}


def _report(provider: str, variant: str, n_actions: int, v2: dict, v1: dict) -> str:
    from pathlib import Path

    def _rank(rows, top=8):
        return "".join(f"| {i + 1} | `{k}` | {mean:+.4f} | {n} |\n" for i, (k, mean, n) in enumerate(rows[:top]))

    lines = [
        f"# xT-GK v2 keeper discrimination — {provider} (FAITHFUL V_opp)\n",
        f"- rho variant: `{variant}` * GK-distribution actions: **{n_actions}** * min {_MIN_N} dist/keeper * "
        f"V_opp fit on FULL cohort (descriptive spread)\n",
        "\n| metric | ICC (action-level) | CV (means, unstable) | n keepers |\n|---|---|---|---|\n",
        f"| **xt_gk_v2** | **{v2['icc']:.4f}** | {v2['cv']:.3f} | {v2['n_keepers']} |\n",
        f"| v1 (c.xt_gk) | {v1['icc']:.4f} | {v1['cv']:.3f} | {v1['n_keepers']} |\n",
        "\n> ICC = between-keeper variance ÷ total (action-level; NOT per-keeper means). Higher = separates "
        "keepers more. CV = std/|mean| of per-keeper means (secondary; unstable when the metric mean ~ 0). "
        "**§3: report whatever it shows** — if v2's ICC ~ v1's, v2 is still keeper-flat on the faithful metric.\n",
        "\n### xt_gk_v2 top keepers (per-action mean; face validity — the owner's coaching eye)\n"
        "| # | player_key | v2 mean | n |\n|---|---|---|---|\n" + _rank(v2["ranking"]),
    ]
    out = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "research"
        / "xtgk_v2_construct_validity"
        / "keeper_discrimination.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = out.read_text(encoding="utf-8") if out.exists() else ""
    out.write_text(prior + "".join(lines) + "\n---\n", encoding="utf-8")
    return str(out)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="xT-GK v2 keeper discrimination (W5).")
    ap.add_argument("--provider", default="gradientsports")
    a = ap.parse_args()

    from _loader_databricks import load_xtgk_cohort  # type: ignore[import-not-found]
    from validate_xtgk_possession_value import (  # type: ignore[import-not-found]
        _FRAME_PRESENT_COLUMN,
        _PRESSURE_COLUMN,
        _XG_COLUMN,
        prepare_cohort,
    )

    from silly_kicks.xtgk import EmpiricalTurnoverValue, MarkovPossessionValue, PressureLevels, compute_xt_gk_v2
    from silly_kicks.xtgk._retention import GkRetentionModel, variant_key_for_provider
    from silly_kicks.xtgk._retention_features import extract_retention_features

    raw, _ = load_xtgk_cohort(a.provider)
    full = prepare_cohort(raw, pressure_column=_PRESSURE_COLUMN, frame_present_column=_FRAME_PRESENT_COLUMN)
    pl = PressureLevels().fit(full[_PRESSURE_COLUMN])
    v = MarkovPossessionValue().fit(full, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, pressure_levels=pl)
    tc = EmpiricalTurnoverValue(min_support=30).fit(
        full, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, pressure_levels=pl
    )
    variant = variant_key_for_provider(a.provider)
    rho = GkRetentionModel.from_variant(variant)

    gk = full[full["is_gk_distribution"].fillna(False)].reset_index(drop=True)
    feats = extract_retention_features(gk, pressure_column=_PRESSURE_COLUMN)
    v2 = compute_xt_gk_v2(
        gk,
        possession_value=v,
        retention=rho,
        turnover_cost=tc,
        pressure_column=_PRESSURE_COLUMN,
        pressure_levels=pl,
        retention_features=feats,
    )["xt_gk_v2"].to_numpy()
    keys = gk["player_key"].to_numpy()
    v1 = pd.to_numeric(gk["xt_gk"], errors="coerce").to_numpy()

    sv2, sv1 = keeper_spread(v2, keys), keeper_spread(v1, keys)
    print(f"provider={a.provider} n_actions={len(gk)} keepers(v2)={sv2['n_keepers']}")
    print(f"  v2 ICC={sv2['icc']:.4f} CV={sv2['cv']:.3f} | v1 ICC={sv1['icc']:.4f} CV={sv1['cv']:.3f}")
    print("wrote", _report(a.provider, variant, len(gk), sv2, sv1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
