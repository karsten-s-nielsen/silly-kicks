"""OpenEvolve feature-representation discovery for xSuccess (TF-61, ADR-093 discipline).

Stage A of the ruthless stack: search the ``xsuccess_features`` construction for a representation that
beats the END-BLIND seed on held-out CALIBRATED log-loss, WITHIN a leakage-safe, END-FREE input
allowlist. The winning function is human-reviewed and committed as deterministic code; this driver
records the run's provenance + fitness (the OUTPUT is committed, the process is documented, not
required to be bit-reproducible -- the TF-57 pattern).

Corpus load is sharded with ``for_each`` (ADR-052); the OpenEvolve LLM search runs in the reduce and
is executed on the box (Task 12) with the ``openevolve`` package + proposer API keys. If ``openevolve``
is absent this driver still assembles + caches the eval dataset and can score a single ``--candidate``
module (the evaluator OpenEvolve itself calls), so the fitness/leakage contract is runnable offline.

Usage (on the box):
  python scripts/evolve_xsuccess_features.py --out <DIR> [--iterations 120 --population 40]
  python scripts/evolve_xsuccess_features.py --out <DIR> --candidate path/to/features_candidate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

#: END-FREE, leakage-safe input columns a candidate feature-builder may read (spec sec 6.4). NEVER
#: end_x/end_y (target leakage) NOR result_id (the label).
_ALLOWED_INPUT_COLUMNS = ["game_id", "period_id", "type_id", "bodypart_id", "start_x", "start_y", "time_seconds"]
_SHARD_SCHEMA_VERSION = "xsuccess-evolve-shard-1"


def onball_inputs_for_match(actions: pd.DataFrame, match_id: object) -> pd.DataFrame:
    """One match's END-FREE feature inputs + the success label -- its eval shard (pure)."""
    non_action = spadlconfig.actiontype_id["non_action"]
    p = actions[actions["type_id"] != non_action]
    out = p[["period_id", "type_id", "bodypart_id", "start_x", "start_y", "time_seconds"]].copy()
    out.insert(0, "game_id", str(match_id))
    out["success"] = (p["result_id"].to_numpy() == spadlconfig.result_id["success"]).astype(int)
    return out.reset_index(drop=True)


def _load_candidate_features(candidate_path: Path):
    """Import a candidate ``xsuccess_features`` from a module file (the OpenEvolve program)."""
    spec = importlib.util.spec_from_file_location("_xsuccess_candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import candidate {candidate_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.xsuccess_features


def _assert_end_blind(features_fn, pooled: pd.DataFrame) -> None:
    """Reject a candidate that reads the realized end (target leakage). The eval shards carry NO
    end columns, so we inject synthetic end_x/end_y and require the features to be byte-identical."""
    base = features_fn(pooled)
    probe = pooled.copy()
    probe["end_x"] = 1.0
    probe["end_y"] = 1.0
    perturbed = features_fn(probe)
    if not np.array_equal(np.asarray(base, float), np.asarray(perturbed, float), equal_nan=True):
        raise ValueError("candidate feature-builder is NOT end-blind: it reads end_x/end_y (target leakage)")


def evaluate_candidate(features_fn, pooled: pd.DataFrame) -> dict:
    """Held-out fitness of a candidate representation: GroupKFold calibrated log-loss + calibration-in-
    the-large + per-type reliability. Lower ``logloss`` is better. Raises if the candidate leaks the end."""
    import xgboost as xgb
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    _assert_end_blind(features_fn, pooled)
    X = np.asarray(features_fn(pooled), dtype=float)
    y = pooled["success"].to_numpy().astype(int)
    groups = pooled["game_id"].to_numpy().astype(str)
    finite = np.isfinite(X).all(axis=1)
    X, y, groups = X[finite], y[finite], groups[finite]
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        n_splits = 2
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lls, brs = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        brs.append(brier_score_loss(y[te], p))
    ll = float(np.mean(lls)) if lls else float("inf")
    return {"logloss": ll, "brier": float(np.mean(brs)) if brs else float("nan"), "n_features": int(X.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="run dir OUTSIDE the repo (shards + evolved features + provenance)")
    ap.add_argument("--competition-id", type=int, default=43)
    ap.add_argument("--season-id", type=int, default=106)
    ap.add_argument(
        "--competitions",
        default=None,
        help="comma-sep comp:season pairs for the FULL redistributable open-data corpus (spec S13); "
        "overrides --competition-id/--season-id. e.g. '43:106,55:43'.",
    )
    ap.add_argument("--max-per-provider", type=int, default=None, help="cap matches (tractable per-candidate eval)")
    ap.add_argument("--match-ids-json", default=None)
    ap.add_argument("--iterations", type=int, default=120, help="OpenEvolve iterations (Task-12 run)")
    ap.add_argument("--population", type=int, default=40, help="OpenEvolve population (Task-12 run)")
    ap.add_argument("--candidate", default=None, help="score a single candidate features module and exit")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; marked dirty)")
    args = ap.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    from scripts._driver import for_each
    from scripts._sb_open_data import load_open_data_matches

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)
    import itertools

    pairs = (
        [tuple(int(x) for x in p.split(":")) for p in args.competitions.split(",")]
        if args.competitions
        else [(args.competition_id, args.season_id)]
    )
    ids = (match_ids or {}).get("statsbomb")
    matches_iter = itertools.chain.from_iterable(
        load_open_data_matches(competition_id=c, season_id=s, match_ids=ids, max_matches=args.max_per_provider)
        for c, s in pairs
    )

    res = for_each(
        matches_iter,
        key=lambda item: (str(item[0]), str(item[1])),
        work=lambda item: onball_inputs_for_match(item[2], item[1]),
        shard_root=dest / "shards",
        token_inputs={
            "model": "xsuccess-evolve",
            "schema": _SHARD_SCHEMA_VERSION,
            "allowed": _ALLOWED_INPUT_COLUMNS,
            "competitions": [list(p) for p in pairs],
        },
        label="match",
    )
    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    pooled = pd.concat([pd.read_parquet(s) for s in shard_files], ignore_index=True) if shard_files else pd.DataFrame()
    if not len(pooled):
        raise SystemExit("no rows collected from the corpus; nothing to evolve")

    from silly_kicks.xsuccess._features import xsuccess_features as seed_features

    seed_fitness = evaluate_candidate(seed_features, pooled)

    if args.candidate:  # offline: score one candidate module (the evaluator OpenEvolve calls)
        cand_fitness = evaluate_candidate(_load_candidate_features(Path(args.candidate)), pooled)
        verdict = "beats_seed" if cand_fitness["logloss"] < seed_fitness["logloss"] else "keep_seed"
        print(json.dumps({"seed": seed_fitness, "candidate": cand_fitness, "verdict": verdict}, indent=2))
        return

    # Stage-A OpenEvolve search (Task 12, on the box). The winning program is human-reviewed + committed;
    # if it does not beat the seed by the pre-registered margin, the seed ships (spec sec 6.1 seed-fallback).
    try:
        import openevolve  # noqa: F401 # pyright: ignore[reportMissingImports]  (Task-12 dependency; run on the box)
    except ImportError as exc:
        raise SystemExit(
            "openevolve not installed. Run the Stage-A search on the box (Task 12) with the openevolve "
            "package + proposer API keys; the seed fitness + eval dataset are cached under --out. "
            f"seed_fitness={seed_fitness}"
        ) from exc

    # OpenEvolve orchestration is finalized at the Task-12 run (config: iterations/population/islands,
    # proposer models, fitness = evaluate_candidate's logloss with gain/regression penalties). The
    # committed OUTPUT is the winning silly_kicks/xsuccess/_features.py + this provenance.
    provenance = {
        "seed_fitness": seed_fitness,
        "iterations": args.iterations,
        "population": args.population,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "n_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        **res.manifest(),
    }
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(json.dumps(provenance, indent=2, default=str))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
