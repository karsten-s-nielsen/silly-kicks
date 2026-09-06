"""Owner-run trainer: bundle the default PassCompletionModel weights (TF-54b).

Fits ``silly_kicks.expected_passing.PassCompletionModel`` on the PUBLIC corpus and writes the
pickle-free JSON + SHA256SUMS artifact via ``model.save(...)``, plus an out-of-fold ``metrics.json``
stamping ``training_commit`` (ADR-052 / ADR-011 discipline, mirroring the five weight trainers).
Inference imports no sklearn; sklearn is used only during the fit.

The expensive per-match corpus load is sharded with ``for_each`` (ADR-052): one shard per match
holding that match's finite-coordinate pass rows, resumable on a crash. The pooled fit + the
GroupKFold-by-match out-of-fold metrics + the ``save`` happen in the reduce, off the network.

``--out`` is a run directory OUTSIDE the repo (it holds the shards, the fitted ``weights/`` artifact,
and ``metrics.json``); the owner copies ``<out>/weights`` into
``silly_kicks/expected_passing/weights/`` for the bundled-weights commit.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/train_pass_completion.py --out <DIR> [--providers statsbomb] \
      [--max-per-provider N] [--tracking-limit N] [--match-ids-json FILE]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

#: The columns one match's pass-row shard carries (the fit corpus for that match). ``type_id`` and
#: ``result_id`` are kept so ``PassCompletionModel.fit`` (which filters ``type_id == pass``) and the
#: success label both work directly off the pooled shards.
_SHARD_COLUMNS = ["game_id", "type_id", "result_id", "start_x", "start_y", "end_x", "end_y"]


def pass_rows_for_match(actions: pd.DataFrame, match_id: object) -> pd.DataFrame:
    """The finite-coordinate ``pass`` rows for one match -- its fit shard (pure)."""
    pass_id = spadlconfig.actiontype_id["pass"]
    p = actions[actions["type_id"] == pass_id].dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    out = p[["type_id", "result_id", "start_x", "start_y", "end_x", "end_y"]].copy()
    out.insert(0, "game_id", str(match_id))
    return out.reset_index(drop=True)


def _predict(model, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.predict_completion(
            rows["start_x"].to_numpy(),
            rows["start_y"].to_numpy(),
            rows["end_x"].to_numpy(),
            rows["end_y"].to_numpy(),
        ),
        dtype=float,
    )


def cross_val_metrics(pooled: pd.DataFrame) -> dict:
    """GroupKFold-by-match out-of-fold AUC / ECE / Brier + the held-out base rate (pure).

    The base rate is the label mean over the finite-prediction rows -- the ``brier_noskill`` a Brier
    skill score is measured against downstream (spec PLAN-09).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from silly_kicks._calibration_metrics import ece
    from silly_kicks.expected_passing import PassCompletionModel

    success = spadlconfig.result_id["success"]
    y = (pooled["result_id"].to_numpy() == success).astype(int)
    groups = pooled["game_id"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    if n_splits < 2:
        oof = _predict(PassCompletionModel().fit(pooled), pooled)
    else:
        for tr, te in GroupKFold(n_splits=n_splits).split(pooled, y, groups):
            m = PassCompletionModel().fit(pooled.iloc[tr])
            oof[te] = _predict(m, pooled.iloc[te])

    keep = np.isfinite(oof)
    yk = y[keep]
    ok = oof[keep]
    auc = float(roc_auc_score(yk, ok)) if len(np.unique(yk)) > 1 else float("nan")
    brier = float(np.mean((ok - yk) ** 2)) if len(ok) else float("nan")
    base_rate = float(yk.mean()) if len(yk) else float("nan")
    return {
        "auc": auc,
        "ece": float(ece(yk, ok)) if len(ok) else float("nan"),
        "brier": brier,
        "base_rate": base_rate,
        "n_oof": len(ok),
    }


def render_model_card(metrics: dict) -> str:
    """The bundled ``PassCompletionModel`` MODEL_CARD.md, generated from the out-of-fold metrics so it
    ships next to the weights and cannot drift (ADR-088: every bundled model carries a card). ASCII-only.
    """
    auc = float(metrics.get("auc", float("nan")))
    ece = float(metrics.get("ece", float("nan")))
    brier = float(metrics.get("brier", float("nan")))
    base = float(metrics.get("base_rate", float("nan")))
    n_rows = metrics.get("n_pass_rows", "?")
    n_matches = metrics.get("n_matches", "?")
    providers = metrics.get("providers", "?")
    commit = metrics.get("training_commit", "?")
    noskill = base * (1.0 - base)
    bss_line = ""
    if np.isfinite(brier) and np.isfinite(noskill) and noskill > 0.0:
        bss_line = f" (Brier skill score {1.0 - brier / noskill:.3f} vs the base-rate baseline)"
    return f"""# Pass-completion model -- `default` variant (TF-54b)

**What it is.** The event-only pass-completion probability `P(complete | origin -> target geometry)`
that the TF-54b territorial counterfactual (`silly_kicks.territory`, `method="counterfactual"`) uses to
weight the threat a defender's territory prevented. Logistic regression; sklearn at fit, pure-numpy
`sigmoid(Xb)` at serve (no runtime sklearn). Loaded via
`silly_kicks.expected_passing.PassCompletionModel.bundled()`; injected into
`compute_territorial_dominance(..., completion_model=)`.

**Label construct.** SPADL `result_id == success` = the pass reached a teammate. Completed passes are
labelled at their real end; failed passes at their SPADL death/recovery location (the field-standard
expected-passing label).

**Features (event-only, 10).** distance, angle-to-goal, forward and lateral components, origin/target
x and y, origin/target pitch-third. No tracking, no teammate positions. At serve for a FAILED pass the
model is evaluated at the HYPOTHESISED target (a cone-restricted xT grid zone), within the geometry
range completed passes already cover.

**Training corpus + metrics.** {n_matches} match(es) from `{providers}` ({n_rows} finite-coordinate
pass rows). GroupKFold-by-match out-of-fold: AUC {auc:.3f}, ECE {ece:.3f}, Brier {brier:.3f} vs base
rate {base:.3f}{bss_line}. See `metrics.json`.

**Missing-value policy.** A non-finite coordinate yields an all-NaN feature row and a NaN probability
(never a fabricated value); the counterfactual seam drops-and-counts such a target.

**Provenance + reproduction.** Reproduce with `python scripts/train_pass_completion.py --out <DIR>
--providers {providers}`. `metrics.json` records `training_commit` ({commit}) and the tree state (this
bundle was produced from a clean tree). Pickle-free JSON + SHA256 envelope (`model.json` +
`SHA256SUMS`) with a feature contract + chirality probe; `load()` is fail-closed
(ADR-011/016/040/044/050). Every bundled model carries a card (ADR-088). Attribution:
expected-passing / pass-completion modelling -- see NOTICE.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="run dir OUTSIDE the repo (shards + weights/ + metrics.json)")
    ap.add_argument(
        "--source",
        choices=("open-data", "pining"),
        default="open-data",
        help="'open-data' = PUBLIC StatsBomb open data (redistributable; the bundled default) via statsbombpy; "
        "'pining' = the pining providers (owner-tier cross-check).",
    )
    ap.add_argument("--competition-id", type=int, default=43, help="open-data competition (default 43 = World Cup)")
    ap.add_argument("--season-id", type=int, default=106, help="open-data season (default 106 = 2022)")
    ap.add_argument("--providers", default="statsbomb", help="comma-separated pining providers (--source pining only)")
    ap.add_argument("--max-per-provider", type=int, default=None, help="cap the number of matches (both sources)")
    ap.add_argument("--tracking-limit", type=int, default=None, help="cap frames parsed per match (pining only)")
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON {"statsbomb": ["3869685", ...]} pinning WHICH matches this process handles.',
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact marked dirty)")
    args = ap.parse_args()

    # Clean-tree guard FIRST, before any corpus work. This trainer writes BUNDLED weights, and an
    # artifact whose provenance is unknown is one nobody can reproduce or audit later (ADR-052).
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    from scripts._driver import for_each
    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.expected_passing._features import FEATURE_NAMES

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)

    # The bundled default trains on PUBLIC StatsBomb open data (redistributable -> publicly reproducible);
    # --source pining is the owner-tier cross-check. Both yield the same (provider, id, actions, frames,
    # home) tuple so for_each is source-agnostic.
    if args.source == "open-data":
        from scripts._sb_open_data import load_open_data_matches

        corpus_label = f"statsbomb-open (competition {args.competition_id}, season {args.season_id})"
        matches_iter = load_open_data_matches(
            competition_id=args.competition_id,
            season_id=args.season_id,
            match_ids=(match_ids or {}).get("statsbomb"),
            max_matches=args.max_per_provider,
        )
        source_token = {"source": "open-data", "competition_id": args.competition_id, "season_id": args.season_id}
    else:
        from scripts._loader_pining import load_matches
        from scripts._partition import providers_for_slice

        # IMPL-05 footgun guard: pining weights are NOT guaranteed public. The bundled default must be
        # public-corpus-trained (the redistributability contract), so name the trap loudly here.
        warnings.warn(
            "Training from --source pining: these weights are NOT guaranteed public/redistributable and "
            "MUST NOT be bundled or published. The pining 'statsbomb' provider is a PRIVATE women's corpus, "
            "not the public men's WC2022 the bundled default uses -- pass --source open-data instead.",
            stacklevel=2,
        )
        corpus_label = f"pining:{args.providers}"
        matches_iter = load_matches(
            providers=providers_for_slice(args.providers.split(","), match_ids),
            match_ids=match_ids,
            max_per_provider=args.max_per_provider,
            tracking_limit=args.tracking_limit,
        )
        source_token = {"source": "pining", "providers": args.providers, "tracking_limit": args.tracking_limit}

    def _work(item):
        _provider, match_id, actions, _frames, _home = item
        return pass_rows_for_match(actions, match_id)

    res = for_each(
        matches_iter,
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=dest / "shards",
        token_inputs={"model": "PassCompletionModel", "feature_names": list(FEATURE_NAMES), **source_token},
        label="match",
    )

    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    frames = [pd.read_parquet(s) for s in shard_files]
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)
    if not len(pooled):
        raise SystemExit("no pass rows collected from the corpus; nothing to fit")

    metrics = cross_val_metrics(pooled)
    model = PassCompletionModel().fit(pooled)
    wdir = dest / "weights"
    model.save(wdir)

    out = {
        **metrics,
        "n_pass_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": corpus_label,
        # ADR-052: which code produced these weights.
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **res.manifest(),
    }
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    # The bundled model carries a MODEL_CARD.md next to its weights (ADR-088), generated from the
    # metrics so it ships with the weights (into silly_kicks/expected_passing/weights/) and cannot drift.
    (wdir / "MODEL_CARD.md").write_text(render_model_card(out), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"bundled weights + MODEL_CARD.md -> {wdir}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
