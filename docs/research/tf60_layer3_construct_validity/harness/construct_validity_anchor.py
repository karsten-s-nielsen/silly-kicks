#!/usr/bin/env python
"""TF-60 Layer-3 construct-validity anchor (LOCKED, owner-ratified 2026-09-06).

DRAFT for the box. Reads the arm-values shards produced by build_tf60_layer3_arm_values.py
and executes the pre-registered anchor VERBATIM (does NOT re-define it). Reported-not-gated.

Outfield arm:
  one-sided Spearman rho between the per-possession outfield arm
  (rd_outfield_deter_threat, and separately rd_outfield_deter_space) and
    (a) rd_num_superiority
    (b) -rd_compactness_x   (more compact = smaller x-range -> negate)
  Criterion: rho < 0 (higher superiority -> more deterrent), one-sided p < 0.05, |rho| >= 0.10.
  rho + CI reported regardless.

Keeper arm:
  one-sided Mann-Whitney of the named set {Alisson, Neuer} per-keeper rd_gk_deter_threat
  vs the rest; criterion: named set BELOW corpus median (more deterrent), one-sided p < 0.05.
  Plus an ADR-060 exceeds_noise_floor-style paired-difference SE report.

Usage:
  python construct_validity_anchor.py --arm-values <dir> [--keeper-names-json <map>]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARM_COLS = ("game_id", "period_id", "team_id", "action_id", "keeper_key",
            "rd_num_superiority", "rd_compactness_x",
            "rd_outfield_deter_threat", "rd_outfield_deter_space",
            "rd_gk_deter_threat", "rd_gk_deter_space",
            "rd_outfield_source", "rd_gk_source")

NAMED_KEEPERS = ("Alisson", "Neuer")  # LOCKED 2026-09-06; identical to TF-19 NAMED_KEEPER_PRIOR


def load_arm_values(arm_dir: Path) -> pd.DataFrame:
    shards = [p for p in arm_dir.rglob("*.parquet") if "manifest" not in p.name]
    if not shards:
        raise SystemExit(f"no shards under {arm_dir}")
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    return df


def spearman_one_sided(x: pd.Series, y: pd.Series, *, direction: str = "less") -> dict:
    """One-sided Spearman. direction='less' tests rho<0."""
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(float), y[m].to_numpy(float)
    n = len(x)
    if n < 10 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": n, "rho": None, "p_one_sided": None, "note": "degenerate/too-few"}
    rho, p_two = stats.spearmanr(x, y)
    # one-sided p for rho<0
    p_one = (p_two / 2) if rho < 0 else (1 - p_two / 2)
    # Fisher-z CI
    z = math.atanh(rho) if abs(rho) < 1 else float("inf")
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    passes = (rho < 0) and (p_one < 0.05) and (abs(rho) >= 0.10)
    return {"n": n, "rho": round(rho, 4), "p_one_sided": p_one, "ci95": [round(lo, 4), round(hi, 4)],
            "criterion_rho<0_p<0.05_|rho|>=0.10": bool(passes)}


def per_keeper_table(df: pd.DataFrame, names: dict | None) -> pd.DataFrame:
    g = df[df["rd_gk_deter_threat"].notna()].groupby("keeper_key")["rd_gk_deter_threat"]
    t = g.agg(["median", "mean", "count"]).reset_index()
    if names:
        t["name"] = t["keeper_key"].astype(str).map({str(k): v for k, v in names.items()})
    else:
        t["name"] = None
    return t


def keeper_mwu(t: pd.DataFrame, *, names_supplied: bool) -> dict:
    if not names_supplied:
        return {"note": "no keeper-name map supplied; MWU {Alisson,Neuer} not runnable", "runnable": False}
    if t["name"].isna().all():
        return {"note": "keeper-name map supplied but NONE of its keepers appear in this corpus slice "
                        "(Alisson=32/Neuer=4602 need the Brazil/Germany matches); MWU not runnable here",
                "runnable": False, "n_keepers": int(len(t))}
    named = t[t["name"].isin(NAMED_KEEPERS)]
    rest = t[~t["name"].isin(NAMED_KEEPERS)]
    if len(named) == 0 or len(rest) < 5:
        return {"note": f"named={len(named)} rest={len(rest)}; insufficient", "runnable": False,
                "named_found": named["name"].tolist()}
    # one-sided: named per-keeper median BELOW rest (more deterrent = more negative)
    u, p_two = stats.mannwhitneyu(named["median"], rest["median"], alternative="less")
    corpus_median = float(t["median"].median())
    # ADR-060-style paired-difference SE report (named mean vs rest mean)
    diff = float(named["median"].mean() - rest["median"].mean())
    pooled_se = math.sqrt(named["median"].var(ddof=1) / max(len(named), 1) +
                          rest["median"].var(ddof=1) / max(len(rest), 1)) if len(named) > 1 else float("nan")
    return {
        "runnable": True,
        "named_found": named[["name", "median", "count"]].to_dict("records"),
        "n_named": int(len(named)), "n_rest": int(len(rest)),
        "corpus_median": round(corpus_median, 5),
        "mwu_U": float(u), "p_one_sided_less": float(p_two),
        "criterion_below_median_p<0.05": bool(p_two < 0.05 and named["median"].median() < corpus_median),
        "adr060_diff_named_minus_rest": round(diff, 5),
        "adr060_pooled_se": (round(pooled_se, 5) if not math.isnan(pooled_se) else None),
        "adr060_effect_over_se": (round(diff / pooled_se, 3) if pooled_se and not math.isnan(pooled_se) and pooled_se > 0 else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-values", required=True, type=Path)
    ap.add_argument("--keeper-names-json", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = load_arm_values(args.arm_values)
    names = json.loads(args.keeper_names_json.read_text()) if args.keeper_names_json else None

    result = {"n_rows": int(len(df)), "n_matches": int(df["game_id"].nunique()),
              "n_keepers": int(df["keeper_key"].nunique())}
    result["outfield"] = {
        "threat_vs_num_superiority": spearman_one_sided(df["rd_outfield_deter_threat"], df["rd_num_superiority"]),
        "threat_vs_neg_compactness": spearman_one_sided(df["rd_outfield_deter_threat"], -df["rd_compactness_x"]),
        "space_vs_num_superiority": spearman_one_sided(df["rd_outfield_deter_space"], df["rd_num_superiority"]),
        "space_vs_neg_compactness": spearman_one_sided(df["rd_outfield_deter_space"], -df["rd_compactness_x"]),
    }
    t = per_keeper_table(df, names)
    result["keeper"] = keeper_mwu(t, names_supplied=names is not None)
    result["keeper_table_top"] = t.sort_values("median").head(15).to_dict("records")

    print(json.dumps(result, indent=2, default=str))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
