#!/usr/bin/env python
"""TF-60 Layer-3 predictive-validity of the rest-defense deterrent arms.

Does a more-deterrent arm predict a LESS-dangerous opponent counter after a committed-forward
turnover? This is the outcome-based, counterfactual-compatible criterion the convergent anchor
(superiority / compactness) lacked -- it tests what the arm CLAIMS.

Unit: a committed-forward possession by A (>=1 arm179 sample) that ends in a LIVE-BALL turnover to B.
Predictor: the arm aggregated (mean) over that possession's arm179 samples (more negative = more
deterrent). Outcome: B's counter danger in the immediately-following possession -- the PEAK xT-surface
value B reaches, the total positive xT-gain, and a shot-in-counter binary. Validity: more-negative arm
=> less danger => POSITIVE corr(arm, counter-danger).

Reads the tc3 _actions (action_id-consistent with arm179), the arm179 xt.npz (the SAME fitted surface),
and the arm179 combined table. Provider-split + a clean full-tracking (sportec/idsse) subset for the
keeper arm (GS 27.5m clamp caveat, ADR-083).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from silly_kicks.id_compat import canonical_id, canonical_id_series
from silly_kicks.spadl.utils import add_possessions
from silly_kicks.xthreat._physical import values_at_points

LIVE_BALL_GAP = 5.0  # s: p+1 within this of p's end == a live-ball counter (not a stoppage restart)
ARM_COLS = ["rd_outfield_deter_threat", "rd_gk_deter_threat", "rd_outfield_deter_space", "rd_gk_deter_space"]
_SHOT_TYPE = None  # resolved from spadlconfig at runtime


def load_xt(npz_path: str, driver_path: str):
    spec = importlib.util.spec_from_file_location("armdrv", driver_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    xt, _ids = m._load_xt_npz(Path(npz_path))
    return xt


def match_rows(game_id, provider, actions, arm_game, xt):
    global _SHOT_TYPE
    if _SHOT_TYPE is None:
        import silly_kicks.spadl.config as cfg

        _SHOT_TYPE = {cfg.actiontypes.index(t) for t in cfg.actiontypes if "shot" in t}
    a = add_possessions(actions.copy())
    a = a.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True)
    a["_end_xt"] = values_at_points(xt, a["end_x"].to_numpy(float), a["end_y"].to_numpy(float))
    a["_gain"] = np.clip(np.asarray(xt.rate(a), dtype=float), 0, None)
    a["_is_shot"] = a["type_id"].isin(_SHOT_TYPE)
    grp = a.groupby("possession_id")
    poss = pd.DataFrame(
        {
            "team": grp["team_id"].first().map(canonical_id),
            "t0": grp["time_seconds"].min(),
            "t1": grp["time_seconds"].max(),
            "period": grp["period_id"].first(),
            "start_x": grp["start_x"].first(),
            "max_att_x": grp["end_x"].max(),  # A's COMMITMENT: deepest attacking x this possession
            "max_xt": grp["_end_xt"].max(),
            "sum_gain": grp["_gain"].sum(),
            "any_shot": grp["_is_shot"].any(),
        }
    ).sort_values(["period", "t0"])
    aid2poss = dict(zip(a["action_id"].astype(int), a["possession_id"], strict=True))
    ag = arm_game.copy()
    ag["possession_id"] = ag["action_id"].astype(int).map(aid2poss)
    arm_by_poss = ag.dropna(subset=["possession_id"]).groupby("possession_id")[ARM_COLS + ["rd_num_superiority"]].mean()
    rows = []
    plist = poss.index.tolist()
    for i in range(len(plist) - 1):
        pA, pB = plist[i], plist[i + 1]
        if pA not in arm_by_poss.index:  # A's possession must be committed-forward (carry arm samples)
            continue
        teamA, teamB = poss.loc[pA, "team"], poss.loc[pB, "team"]
        if pd.isna(teamA) or pd.isna(teamB) or teamA == teamB:  # need two real, DISTINCT teams (a turnover)
            continue
        if poss.loc[pB, "period"] != poss.loc[pA, "period"]:
            continue
        gap = poss.loc[pB, "t0"] - poss.loc[pA, "t1"]
        if not (0 <= gap <= LIVE_BALL_GAP):  # live-ball counter only
            continue
        arm = arm_by_poss.loc[pA]
        rows.append(
            {
                "game_id": game_id,
                "provider": provider,
                "possession_id": int(pA),
                **{k: float(arm[k]) for k in ARM_COLS},
                "counter_max_xt": float(poss.loc[pB, "max_xt"]),
                "counter_sum_gain": float(poss.loc[pB, "sum_gain"]),
                "counter_shot": int(bool(poss.loc[pB, "any_shot"])),
                "turnover_x_B": float(poss.loc[pB, "start_x"]),
                "poss_max_x": float(poss.loc[pA, "max_att_x"]),  # A's commitment (deepest attack)
                "minute": float(poss.loc[pA, "t1"]) / 60.0 + (int(poss.loc[pA, "period"]) - 1) * 45,
                "poss_len": float(poss.loc[pA, "t1"] - poss.loc[pA, "t0"]),
                "gap": float(gap),
            }
        )
    return rows


def analyse(df, arm_col, out_col):
    sub = df[[arm_col, out_col, "turnover_x_B", "minute", "poss_len"]].dropna()
    if len(sub) < 30 or sub[arm_col].std() == 0 or sub[out_col].std() == 0:
        return {"n": len(sub), "note": "too-few/degenerate"}
    rho, p = stats.spearmanr(sub[arm_col], sub[out_col])
    # controlled OLS: outcome ~ arm + turnover_x + minute + poss_len
    x_mat = np.column_stack(
        [np.ones(len(sub)), sub[arm_col], sub["turnover_x_B"], sub["minute"], sub["poss_len"]]
    )
    y = sub[out_col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    resid = y - x_mat @ beta
    dof = len(sub) - x_mat.shape[1]
    s2 = float(resid @ resid) / dof
    se = math.sqrt(float((s2 * np.linalg.inv(x_mat.T @ x_mat))[1, 1]))
    t = float(beta[1]) / se if se > 0 else float("nan")
    # VALIDITY: more-negative arm -> less danger -> positive rho / positive OLS coef
    return {
        "n": int(len(sub)),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p),
        "ols_arm_coef": round(float(beta[1]), 6),
        "ols_arm_t": round(t, 3),
        "valid_direction (rho>0 & p<0.05)": bool(rho > 0 and p < 0.05),
    }


def att_analysis(df, arm_col, out_col="counter_max_xt", *, cov_cols=("turnover_x_B", "minute", "poss_len")):
    """Matched ATT: effect of a MORE-deterrent rearguard (arm below median = more negative) on the
    counter danger, matched on cov_cols + provider (Abadie-Imbens SE). VALID = ATT < 0 (high-deterrent
    lowers counter danger) and significant. Run with/without turnover_x_B to probe confounder-vs-mediator."""
    from silly_kicks.causal.matching import estimate_att, fit_propensity

    sub = df[[arm_col, out_col, *cov_cols, "provider"]].dropna()
    if len(sub) < 60 or sub[arm_col].std() == 0:
        return {"n": int(len(sub)), "note": "too-few/degenerate"}
    med = sub[arm_col].median()
    z = (sub[arm_col] < med).astype(int).to_numpy()  # treatment = MORE deterrent (below median)
    if z.sum() < 20 or (1 - z).sum() < 20:
        return {"n": int(len(sub)), "note": "degenerate-treatment"}
    prov = pd.get_dummies(sub["provider"], drop_first=True).to_numpy(float)
    base = sub[list(cov_cols)].to_numpy(float)
    x_mat = np.column_stack([base, prov]) if prov.size else base
    y = sub[out_col].to_numpy(float)
    ps, _ = fit_propensity(x_mat, z, seed=0)
    est = estimate_att(y, z, ps, x_mat)
    over_se = float(est.estimate / est.se) if est.se > 0 else float("nan")
    smd = est.balance["smd_post"].abs().max() if "smd_post" in getattr(est, "balance", {}) else None
    return {
        "n": int(len(sub)),
        "n_treated": int(z.sum()),
        "att": round(float(est.estimate), 6),
        "se": round(float(est.se), 6),
        "att_over_se": round(over_se, 3),
        "valid_ATT<0_sig": bool(est.estimate < 0 and abs(over_se) > 1.96),
        "max_abs_smd_post": (round(float(smd), 3) if smd is not None else None),
    }


def _ols(y, x_cols):
    xc = np.column_stack([np.ones(len(y)), x_cols])
    beta, *_ = np.linalg.lstsq(xc, y, rcond=None)
    resid = y - xc @ beta
    dof = max(len(y) - xc.shape[1], 1)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(xc.T @ xc)))
    return beta, se


def mediation(df, arm_col, med="turnover_x_B", out="counter_max_xt", ctrl=("minute", "poss_len")):
    """Linear natural direct/indirect decomposition around the turnover position (the possible
    mediator). a = arm->turnover_x; b = turnover_x->counter | arm; NDE = arm->counter | turnover_x;
    NIE = a*b (Sobel z). If NDE ~0 while NIE carries the total, the arm's association with counter
    danger runs THROUGH where the turnover happens."""
    sub = df[[arm_col, med, out, *ctrl, "provider"]].dropna()
    if len(sub) < 60:
        return {"n": int(len(sub)), "note": "too-few"}
    prov = pd.get_dummies(sub["provider"], drop_first=True).to_numpy(float)
    c = np.column_stack([sub[list(ctrl)].to_numpy(float), prov]) if prov.size else sub[list(ctrl)].to_numpy(float)
    av, mv, yv = (sub[arm_col].to_numpy(float), sub[med].to_numpy(float), sub[out].to_numpy(float))
    bm, sem = _ols(mv, np.column_stack([av, c]))
    a, a_se = bm[1], sem[1]
    by, sey = _ols(yv, np.column_stack([av, mv, c]))
    cprime, cp_se, b, b_se = by[1], sey[1], by[2], sey[2]
    bt, _ = _ols(yv, np.column_stack([av, c]))
    total = bt[1]
    nie = a * b
    sobel = math.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
    return {
        "n": int(len(sub)),
        "a_arm_to_turnoverx": [round(float(a), 5), round(float(a / a_se), 2)],
        "b_turnoverx_to_counter": [round(float(b), 6), round(float(b / b_se), 2)],
        "total_effect": round(float(total), 6),
        "NDE_direct": [round(float(cprime), 6), round(float(cprime / cp_se), 2)],
        "NIE_indirect": round(float(nie), 6),
        "NIE_sobel_z": (round(float(nie / sobel), 2) if sobel > 0 else None),
        "prop_mediated": (round(float(nie / total), 3) if total != 0 else None),
    }


def report(df, label):
    out = {"label": label, "n_turnovers": int(len(df)), "n_games": int(df["game_id"].nunique())}
    for ac in ARM_COLS:
        out[f"{ac}__spearman"] = analyse(df, ac, "counter_max_xt")
        out[f"{ac}__ATT_ctrl_turnoverx"] = att_analysis(df, ac, cov_cols=("turnover_x_B", "minute", "poss_len"))
        out[f"{ac}__ATT_ctrl_commitment"] = att_analysis(df, ac, cov_cols=("poss_max_x", "minute", "poss_len"))
        out[f"{ac}__ATT_ctrl_both"] = att_analysis(df, ac, cov_cols=("turnover_x_B", "poss_max_x", "minute", "poss_len"))
        out[f"{ac}__ATT_ctrl_none"] = att_analysis(df, ac, cov_cols=("minute", "poss_len"))
        out[f"{ac}__mediation"] = mediation(df, ac)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-values", required=True)
    ap.add_argument("--xt-npz", required=True)
    ap.add_argument("--actions-dir", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    arm = pd.read_parquet(args.arm_values)
    arm["game_id"] = canonical_id_series(arm["game_id"])
    xt = load_xt(args.xt_npz, args.driver)

    # tc3 file index: the game_id COLUMN matches arm179, but the FILENAME game-part differs for IDSSE
    # (arm179 "J03WPY" vs file "idsse__DFL-MAT-J03WPY.parquet"). Index by the file game-part AND its
    # DFL suffix so GS/SC (numeric) and IDSSE (DFL) ids both resolve; O(1) lookup, no suffix collisions.
    tc3_index = {}
    for f in glob.glob(f"{args.actions_dir}/*__*.parquet"):
        gp = Path(f).name.split("__", 1)[1][: -len(".parquet")]
        tc3_index[canonical_id(gp)] = f
        if gp.startswith("DFL-MAT-"):
            tc3_index[canonical_id(gp[len("DFL-MAT-") :])] = f
    allrows = []
    for g, arm_game in arm.groupby("game_id"):
        f = tc3_index.get(g)
        if f is None:
            continue
        provider = Path(f).name.split("__")[0]
        actions = pd.read_parquet(f)
        allrows += match_rows(g, provider, actions, arm_game, xt)
    df = pd.DataFrame(allrows)
    if not len(df):
        print(json.dumps({"error": "no turnover rows"}))
        return

    result = {
        "n_turnovers": int(len(df)),
        "n_games": int(df["game_id"].nunique()),
        "by_provider_counts": df["provider"].value_counts().to_dict(),
        "counter_danger_summary": {
            "max_xt_mean": round(float(df["counter_max_xt"].mean()), 5),
            "shot_rate": round(float(df["counter_shot"].mean()), 4),
        },
        "ALL": report(df, "all-179"),
        "full_tracking_only (sportec/idsse)": report(df[df["provider"].isin(["sportec", "idsse"])], "full-tracking"),
        "gradientsports_only": report(df[df["provider"] == "gradientsports"], "GS"),
        "skillcorner_only": report(df[df["provider"] == "skillcorner"], "SC"),
    }
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
