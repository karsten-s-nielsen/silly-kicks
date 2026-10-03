"""Receiver 30 -> 327 widening gate -- PRE-REGISTERED (combined-cycle-completion spec section 7).

Run at C1 on a clean tree:
    python scripts/validate_receiver_widening.py --rows <candidate_rows.parquet> --out <dir>
Step 1 identifies the committed model's 30 training matches (the first 30 of the statsbomb manifest,
verified by refit reproduction, with a negative control). Step 2 compares a fresh per-fold refit on the
327 rows against the committed model on identical held-out passes. The output carries provenance and
aggregate numbers only -- never a match id (statsbomb is manifest-private, ADR-062).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.train_receiver_model import _feature_names, _namespace_game_ids, cv_top1
from silly_kicks.tracking._receiver import ReceiverModel

FEATURE_SET = "public"
_COMMITTED = Path(__file__).resolve().parents[1] / "silly_kicks" / "tracking" / "_receiver_weights" / "default"
ID_TOP1, ID_TOP1_TOL, ID_PARAM_RTOL = 0.5097663185813329, 0.005, 1e-2
#: D8 (owner-approved 2026-10-02): the trainer's own Q3 data-earns-inclusion rule (pooling_gate: pooled >=
#: primary) AND a non-inferiority bound that stops a noisy pass. MARGIN = 1 percentage point of top-1: below
#: the committed model's fold-to-fold SD (~0.013) and near the margin that rejected the GS pool (-0.009).
MARGIN = 0.01
DECISION_RULE = "point_estimate_ge_0_and_boot_lb95_gt_-0.01"


def decide(diff: float, lb: float) -> bool:
    """Ship iff the point estimate new - old is >= 0 AND the 95% bootstrap lower bound is > -MARGIN."""
    return diff >= 0.0 and lb > -MARGIN


def per_pass_hits(model, rows: pd.DataFrame) -> pd.DataFrame:
    """One row per pass: ``hit`` = the argmax candidate is the labelled receiver (the trainer's rule)."""
    names = _feature_names(FEATURE_SET)
    test = rows.reset_index(drop=True).copy()
    test["_p"] = model.predict_candidates(test[names])
    out = [
        (g, a, int(grp.loc[grp["_p"].idxmax(), "label"] == 1)) for (g, a), grp in test.groupby(["game_id", "action_id"])
    ]
    return pd.DataFrame(out, columns=["game_id", "action_id", "hit"])


def _params(m: ReceiverModel) -> np.ndarray:
    coef, intercept, mean, std = m._coef, m._intercept, m._mean, m._std
    if coef is None or intercept is None or mean is None or std is None:
        raise ValueError("ReceiverModel is not fitted -- no parameters to compare")
    return np.concatenate([np.ravel(coef), np.ravel(intercept), np.ravel(mean), np.ravel(std)])


def _reproduces(
    rows: pd.DataFrame, ids: list[str], committed: ReceiverModel, expected_top1: float
) -> tuple[bool, dict]:
    names = _feature_names(FEATURE_SET)
    sub = rows[rows["game_id"].isin(ids)]
    if sub["game_id"].nunique() != len(ids):
        return False, {"n_present": int(sub["game_id"].nunique())}
    refit = ReceiverModel(FEATURE_SET).fit(sub[names], sub["label"])
    top1, _folds = cv_top1(sub, FEATURE_SET)
    a, b = _params(refit), _params(committed)
    rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
    ok = abs(top1 - expected_top1) <= ID_TOP1_TOL and rel <= ID_PARAM_RTOL
    return ok, {"n_present": len(ids), "refit_top1_cv": float(top1), "max_rel_param_diff": rel}


def identify(rows, candidate_ids, committed, *, expected_top1=ID_TOP1, seed=0, n_controls=20):
    """Step 1 + its negative control. Returns (report, game ids to exclude from every test fold)."""
    ids = [f"statsbomb:{m}" for m in candidate_ids]
    ok, detail = _reproduces(rows, ids, committed, expected_top1)
    rest = sorted(set(rows["game_id"]) - set(ids))
    rng = np.random.default_rng(seed)
    passing = 0
    for _ in range(n_controls):
        ctrl = list(rng.choice(rest, size=len(ids), replace=False)) if len(rest) >= len(ids) else []
        if ctrl and _reproduces(rows, ctrl, committed, expected_top1)[0]:
            passing += 1
    discriminating = passing == 0
    report = {
        "identified": bool(ok and discriminating),
        "candidate_reproduces": bool(ok),
        "n_controls": n_controls,
        "n_controls_passing": passing,
        "criteria": {"top1": expected_top1, "top1_tol": ID_TOP1_TOL, "param_rtol": ID_PARAM_RTOL},
        **detail,
    }
    return report, (ids if report["identified"] else [])


def gate(rows, committed, exclude, *, n_splits=5, seed=0, n_boot=2000) -> dict:
    names = _feature_names(FEATURE_SET)
    news, olds = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(rows, groups=rows["game_id"].to_numpy()):
        train, test = rows.iloc[tr], rows.iloc[te]
        test = test[~test["game_id"].isin(exclude)]
        if test.empty:
            continue
        m = ReceiverModel(FEATURE_SET).fit(train[names], train["label"])
        news.append(per_pass_hits(m, test))
        olds.append(per_pass_hits(committed, test))
    both = pd.concat(news).merge(
        pd.concat(olds), on=["game_id", "action_id"], suffixes=("_new", "_old"), validate="one_to_one"
    )
    g = both.groupby("game_id").agg(n=("hit_new", "size"), d=("hit_new", "sum"), o=("hit_old", "sum"))
    n, d, o = g["n"].to_numpy(), g["d"].to_numpy(), g["o"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    boots = (d[idx].sum(1) - o[idx].sum(1)) / n[idx].sum(1)
    diff = float((d.sum() - o.sum()) / n.sum())
    lb, ub = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "n_test_matches": len(g),
        "n_test_passes": int(n.sum()),
        "top1_new": float(d.sum() / n.sum()),
        "top1_old": float(o.sum() / n.sum()),
        "diff_new_minus_old": diff,
        "boot_ci_95": [lb, ub],
        "decision_rule": DECISION_RULE,
        "margin": MARGIN,
        "ship": decide(diff, lb),
        "seed": seed,
        "n_boot": n_boot,
        "excluded_from_test": len(exclude),
    }


def _committed_model() -> ReceiverModel:
    return ReceiverModel.load(_COMMITTED)


def _statsbomb_first30() -> list[str]:
    """The first 30 ids of the statsbomb manifest (owner token). Read in-process, never written out."""
    from scripts._loader_pining import _base_url, _list_matches, _resolve_token

    return [str(m["id"]) for m in _list_matches("statsbomb", _resolve_token(None), _base_url())[:30]]


def run(rows_path: Path, out: Path, *, prov: dict, n_boot: int = 2000, n_controls: int = 20) -> dict:
    """The gate over an already-checked provenance (``main`` refuses a dirty tree before calling this)."""
    rows = _namespace_game_ids(pd.read_parquet(rows_path), "statsbomb")
    committed = _committed_model()
    ident, exclude = identify(rows, _statsbomb_first30(), committed, n_controls=n_controls)
    result = {
        "identification": ident,
        "gate": gate(rows, committed, exclude, n_boot=n_boot),
        "inputs_sha256": {
            "rows": hashlib.sha256(Path(rows_path).read_bytes()).hexdigest(),
            "committed_model": hashlib.sha256((_COMMITTED / "model.json").read_bytes()).hexdigest(),
        },
        "n_matches": int(rows["game_id"].nunique()),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "receiver_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    from scripts._provenance import git_provenance, require_clean_tree

    # FIRST, in main() itself: the provenance entry-point gate (test_provenance_wiring) requires this call
    # here, not behind a helper (B r4 CCC-PLAN-29).
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    print(json.dumps(run(args.rows, args.out, prov=prov), indent=2))


if __name__ == "__main__":
    main()
