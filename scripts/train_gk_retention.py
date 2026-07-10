"""Train the GK-distribution retention model (rho) for xT-GK v2 (ADR-036 §Part 3).

Mirror of train_gk_completion.py, but the label is retains() (NOT completion) and EVERY shipped
variant is calibration-gated (ece<=0.10 AND |reliability_slope-1|<=0.25). GS(WC2022) 'default' +
SkillCorner variant via the same GS-transfer-or-bundle decision. The __main__ owner-run (real
pining/Databricks load, per-variant fit + gate, save under _retention_weights/) is NOT exercised in
CI -- only the pure prepare/gate/CV helpers below are unit-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._calibration_metrics import ece, reliability_slope
from silly_kicks.xtgk._retention import GkRetentionModel
from silly_kicks.xtgk._retention_features import extract_retention_features
from silly_kicks.xtgk._retention_labels import retains

_ECE_MAX = 0.10
_SLOPE_TOL = 0.25


def prepare_retention_training_data(
    actions: pd.DataFrame, *, pressure_column: str = "pressure"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (features, labels, groups) from a full attack-LTR action stream (marts-native).

    Domain = **goal-kicks** (the mart-reliable GK-distribution subset) UNION the materialized
    ``is_gk_distribution`` flag (= tracking.gk_distribution_mask, resolve_gk="robust"; NULLs coalesced
    to False -- the rollout population is out of scope, not dropped). ``retains`` is computed on the FULL
    stream then masked. Drops geometry-unscoreable + truncated-window (NaN-label) rows.
    """
    import silly_kicks.spadl.config as spadlconfig

    mask = actions["type_id"].to_numpy() == spadlconfig.actiontype_id["goalkick"]
    if "is_gk_distribution" in actions.columns:
        mask = mask | actions["is_gk_distribution"].fillna(False).to_numpy(dtype=bool)
    X_full = extract_retention_features(actions, pressure_column=pressure_column)
    y_full = retains(actions).to_numpy(dtype=float)  # 1.0 / 0.0 / NaN (truncated windows)
    domain = np.asarray(mask, dtype=bool)
    X = X_full.loc[domain].reset_index(drop=True)
    y = y_full[domain]
    groups = (actions["game_id"].to_numpy() if "game_id" in actions.columns else np.zeros(len(actions)))[domain]
    keep = np.isfinite(X["length"].to_numpy()) & np.isfinite(X["dest_x"].to_numpy()) & np.isfinite(y)
    return X.loc[keep].reset_index(drop=True), y[keep].astype(int), groups[keep]


def calibration_gate(y: np.ndarray, oof: np.ndarray) -> tuple[bool, dict]:
    e = ece(y, oof)
    s = reliability_slope(y, oof)
    ok = (e <= _ECE_MAX) and (np.isfinite(s) and abs(s - 1.0) <= _SLOPE_TOL)
    return bool(ok), {"ece": e, "reliability_slope": s, "ece_max": _ECE_MAX, "slope_tol": _SLOPE_TOL}


def cross_val_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    if n_splits < 2:
        return GkRetentionModel().fit(X, pd.Series(y)).predict_proba(X)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = GkRetentionModel().fit(X.iloc[tr], pd.Series(y[tr]))
        oof[te] = m.predict_proba(X.iloc[te])
    return oof


# --- owner-run trainer (gold marts -> features -> calibrate -> fit -> bundle) ---------------------
# NOT exercised in CI (needs Databricks read access). Marts-native (tracking-frames deprecated):
# fct_action_values (geometry/type/result/possession) + fct_action_context (pressure + is_gk_distribution).


def main() -> int:
    import argparse
    import json
    from pathlib import Path

    from _loader_databricks import load_retention_cohort  # type: ignore[import-not-found]

    ap = argparse.ArgumentParser(description="Train + bundle the GK retention (rho) model (marts-native).")
    ap.add_argument("--provider", required=True)
    ap.add_argument("--variant", required=True, help="weights dir under _retention_weights/ (default|skillcorner)")
    a = ap.parse_args()

    actions = load_retention_cohort(a.provider)
    print(f"loaded {len(actions)} actions ({actions['game_id'].nunique()} matches) for {a.provider}")
    X, y, groups = prepare_retention_training_data(actions)
    print(f"\nCORPUS rows={len(X)} pos={int(y.sum())} ({y.mean():.3f}) n_games={len(np.unique(groups))}")
    oof = cross_val_oof(X, y, groups)
    ok, metrics = calibration_gate(y, oof)
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
    print(
        f"OOF AUC={auc:.3f} ECE={metrics['ece']:.3f} "
        f"slope={metrics['reliability_slope']:.2f} GATE={'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        print("CALIBRATION GATE FAILED -- not bundling (escalate: plain logistic under-calibrated).")
        return 3
    model = GkRetentionModel().fit(X, pd.Series(y))
    model.shipped_variant = a.variant
    model.provider_list = [a.provider]
    wdir = Path(__file__).resolve().parent.parent / "silly_kicks" / "xtgk" / "_retention_weights" / a.variant
    model.save(wdir)
    (wdir / "metrics.json").write_text(
        json.dumps(
            {"auc": auc, **metrics, "n_rows": len(X), "n_games": len(np.unique(groups)), "provider": a.provider},
            indent=2,
        ),
        encoding="utf-8",
    )
    (wdir / "MODEL_CARD.md").write_text(
        f"# GK retention (rho) model — variant `{a.variant}`\n\n"
        f"P(retain | GK distribution) for xT-GK v2 (ADR-036 §Part 3). Logistic, pure-numpy serve.\n\n"
        f"- Provider: **{a.provider}** ({len(np.unique(groups))} matches, {len(X)} GK-distribution actions)\n"
        f"- Label: `retains(window_seconds=10)` (truncated windows excluded)\n"
        f"- Marts-native 8 features (geometry + `pressure_on_actor__andrienko_oval`); tracking-frames "
        f"deprecated so the frames-only receiver-density feature is absent\n"
        f"- Out-of-fold (GroupKFold by match): **AUC {auc:.3f}**, **ECE {metrics['ece']:.3f}**, "
        f"reliability slope {metrics['reliability_slope']:.2f}\n"
        f"- Calibration gate (ECE<=0.10 AND |slope-1|<=0.25): **PASS**\n",
        encoding="utf-8",
    )
    print(f"bundled -> {wdir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    raise SystemExit(main())
