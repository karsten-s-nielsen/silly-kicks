"""Owner-run trainer: bundle the default PassCompletionModel weights (TF-54b).

Fits ``silly_kicks.expected_passing.PassCompletionModel`` on the PUBLIC corpus and writes the
pickle-free JSON + SHA256SUMS artifact via ``model.save(...)``, plus an out-of-fold ``metrics.json``
stamping ``training_commit`` (ADR-052 / ADR-011 discipline, mirroring the five weight trainers).
Inference imports no sklearn; sklearn is used only during the fit.

An ADR-128 pipeline (HFU spec 15.5a): ``refs`` -> ``extract`` (one store-cached unit per match: its
finite-coordinate pass rows) -> ``fit`` (the GroupKFold-by-match out-of-fold metrics, the pooled fit, the
bundle). ``python -m scripts.train_pass_completion --run-all --store <STORE> --all-competitions --out <DIR>``.

The owner-tier pining cross-check (DGX only) runs from ``scripts/_pass_completion_pining.py``, which injects its
source into ``pining_main``; it shares ``fit_bundle``. This module runs on HF, so it never imports the pining client:
the I1 gate walks its static import closure (ADR-123 Amendment B).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from scripts import _open_data_stages as od
from scripts._stages import FileSet, ItemStage, Pipeline, Stage, main
from silly_kicks.expected_passing._features import FEATURE_NAMES

#: The owner-tier pining corpus as ``(refs, load, corpus_label, source_token)``, built from the parsed
#: arguments and the ``--match-ids-json`` selection. ``scripts/_pass_completion_pining.py`` injects it, so
#: the HF job's import closure never reaches the pining client (I1, ADR-123 Amendment B).
PiningSource = Callable[[argparse.Namespace, dict | None], tuple[list[Any], Callable[[Any], Any], str, dict]]

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
        import time

        from scripts._study_shared import print_fold

        t0 = time.monotonic()
        for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(pooled, y, groups), start=1):
            m = PassCompletionModel().fit(pooled.iloc[tr])
            oof[te] = _predict(m, pooled.iloc[te])
            print_fold("pass_completion", k, n_splits, t0)

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
    n_competitions = metrics.get("n_competitions")
    digest = metrics.get("corpus_match_ids_sha256")  # the corpus identity verify checks (HFU spec 14.13)
    identity = f"\nCorpus identity: {metrics.get('corpus_n_matches')} matches, sha256 {digest}." if digest else ""
    noskill = base * (1.0 - base)
    bss_line = ""
    if np.isfinite(brier) and np.isfinite(noskill) and noskill > 0.0:
        bss_line = f" (Brier skill score {1.0 - brier / noskill:.3f} vs the base-rate baseline)"
    # A source-faithful reproduce command on the pipeline CLI: the full public open-data corpus uses
    # --all-competitions, a single tournament --competition-id/--season-id; the pining cross-check runs
    # from its own entry with --source pining --providers.
    m_open = re.match(r"statsbomb-open \(competition (\d+), season (\d+)\)", str(providers))
    corpus_note = ""
    entry = "-m scripts.train_pass_completion --run-all --store <STORE>"
    if n_competitions is not None:
        reproduce_args = "--all-competitions"
        corpus_note = (
            f" This bundle was fit on the full public open-data corpus "
            f"({n_competitions} (competition, season) releases in the StatsBomb open-data manifest)."
        )
    elif m_open:
        reproduce_args = f"--competition-id {m_open.group(1)} --season-id {m_open.group(2)}"
    else:
        # The corpus label is "pining:<providers>"; --providers takes the bare list.
        entry = "scripts/_pass_completion_pining.py"
        reproduce_args = f"--source pining --providers {str(providers).removeprefix('pining:')}"
    return f"""# Pass-completion model -- `default` variant (TF-54b)

**What it is.** An event-only pass-completion probability `P(complete | origin -> target geometry)`,
shipped as a reusable seam. Logistic regression; sklearn at fit, pure-numpy `sigmoid(Xb)` at serve
(no runtime sklearn). Loaded via `silly_kicks.expected_passing.PassCompletionModel.bundled()`. Carried
from the unmerged `ab9001c` alongside `xthreat.destination_profiles`; the event-only "threat prevented"
counterfactual cone it was built to weight was not carried into the TF-54b cycle, so no shipped feature
consumes it yet -- it is library infrastructure for a future consumer.

**Label construct.** SPADL `result_id == success` = the pass reached a teammate. Completed passes are
labelled at their real end; failed passes at their SPADL death/recovery location (the field-standard
expected-passing label).

**Features (event-only, 10).** distance, angle-to-goal, forward and lateral components, origin/target
x and y, origin/target pitch-third. No tracking, no teammate positions. A consumer scoring a FAILED
pass evaluates the model at a HYPOTHESISED target (e.g. an xT-grid destination), within the geometry
range completed passes already cover.

**Training corpus + metrics.** {n_matches} match(es) from `{providers}` ({n_rows} finite-coordinate
pass rows).{corpus_note} GroupKFold-by-match out-of-fold: AUC {auc:.3f}, ECE {ece:.3f}, Brier
{brier:.3f} vs base rate {base:.3f}{bss_line}. See `metrics.json`.{identity}

**Missing-value policy.** A non-finite coordinate yields an all-NaN feature row and a NaN probability
(never a fabricated value); a consumer drops-and-counts such a target.

**Provenance + reproduction.** Reproduce with `python {entry} --out <DIR>
{reproduce_args}`. `metrics.json` records `training_commit` ({commit}) and the tree state (this
bundle was produced from a clean tree). Pickle-free JSON + SHA256 envelope (`model.json` +
`SHA256SUMS`) with a feature contract + chirality probe; `load()` is fail-closed
(ADR-011/016/040/044/050). Every bundled model carries a card (ADR-088). Attribution:
expected-passing / pass-completion modelling -- see NOTICE.
"""


def fit_bundle(pooled: pd.DataFrame, *, prov: dict, corpus_label: str, extra: dict) -> dict[str, bytes]:
    """The bundle files from the pooled pass rows: the fitted model's files (``model.save``), its
    ``MODEL_CARD.md`` (ADR-088) and the out-of-fold ``metrics.json``. The pipeline and the pining CLI share it."""
    import tempfile

    from silly_kicks.expected_passing import PassCompletionModel

    if not len(pooled):
        raise SystemExit("no pass rows collected from the corpus; nothing to fit")
    metrics = cross_val_metrics(pooled)
    model = PassCompletionModel().fit(pooled)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(Path(tmp))
        files = {f.name: f.read_bytes() for f in sorted(Path(tmp).iterdir()) if f.is_file()}
    out = {
        **metrics,
        "n_pass_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": corpus_label,
        "training_commit": prov["commit"],  # ADR-052: which code produced these weights
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **extra,
    }
    files["MODEL_CARD.md"] = render_model_card(out).encode("utf-8")
    files["metrics.json"] = json.dumps(out, indent=2, default=str).encode("utf-8")
    print(json.dumps(out, indent=2, default=str))
    return files


def _extract_reduce(results, ctx) -> pd.DataFrame:
    """The per-match pass rows in sorted key order (the legacy sorted shard glob)."""
    from scripts._stages import combine_frames

    frames = list(results.values())
    return combine_frames(frames, ids=("game_id",)) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


def _fit_work(key, ctx):
    """The pooled fit and its bundle; the corpus identity and the competitions verify reads (HFU spec 14.3)."""
    from scripts._corpus_identity import open_data_identity
    from scripts._open_data_stages import corpus_label

    refs = ctx.upstream("refs")
    comps = [tuple(c) for c in refs["competitions"]]
    extra = open_data_identity([r["match_id"] for r in refs["refs"]]) | od.pass_fields(ctx)
    if ctx.params["all_competitions"]:
        extra |= {"n_competitions": len(comps), "competitions": [list(c) for c in comps]}
    label = corpus_label(ctx.params, comps)
    return FileSet(fit_bundle(ctx.upstream("extract"), prov=ctx.prov, corpus_label=label, extra=extra))


PIPELINE = Pipeline(
    name="pass_completion",
    module="scripts.train_pass_completion",
    params=od.PARAMS,
    stages=(
        Stage(
            name="refs",
            units=lambda ctx: ["only"],
            work=od.refs_work,
            reduce=lambda r, ctx: r["only"],
            params=od.REFS_PARAMS,
            uses_dataset=True,
            dataset_files=lambda ctx: (),  # open data from the snapshot: no dataset file
        ),
        ItemStage(
            name="extract",
            refs=od.extract_refs,
            key=lambda ref: ref.key,
            load=od.load_open_match,
            work=lambda lm, ctx: pass_rows_for_match(lm.actions, lm.match_id),
            needs=("refs",),
            reduce=_extract_reduce,
            declare=lambda p: {"model": "PassCompletionModel", "feature_names": list(FEATURE_NAMES)},
            cross_commit_version="pass-completion-shard-1",
            uses_dataset=True,
            max_failures=None,  # the legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("refs", "extract"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="fit",
)


def pining_main(argv: list[str] | None = None, *, pining_source: PiningSource) -> None:
    """The owner-tier pining cross-check CLI (DGX only), run from ``scripts/_pass_completion_pining.py``, which
    injects ``pining_source``. HF runs the pipeline (``python -m scripts.train_pass_completion``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="run dir OUTSIDE the repo (shards + weights/ + metrics.json)")
    ap.add_argument("--source", choices=("pining",), default="pining", help="the pining providers only")
    ap.add_argument("--providers", default="statsbomb", help="comma-separated pining providers")
    ap.add_argument("--max-per-provider", type=int, default=None, help="cap the number of matches")
    ap.add_argument("--tracking-limit", type=int, default=None, help="cap frames parsed per match")
    ap.add_argument("--match-ids-json", default=None, help='JSON {"statsbomb": ["3869685", ...]}')
    ap.add_argument("--cache-dir", default=None, help="raw-artifact cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact marked dirty)")
    args = ap.parse_args(argv)

    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)  # before any corpus work (ADR-052)

    from scripts._driver import for_each
    from silly_kicks.expected_passing._features import FEATURE_NAMES

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)
    refs, load, label, source_token = pining_source(args, match_ids)
    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=lambda item: pass_rows_for_match(item[2], item[1]),
        shard_root=dest / "shards",
        token_inputs={"model": "PassCompletionModel", "feature_names": list(FEATURE_NAMES), **source_token},
        label="match",
    )
    frames = [pd.read_parquet(f) for f in sorted(res.shard_dir.glob("*.parquet"))]
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)
    files = fit_bundle(pooled, prov=prov, corpus_label=label, extra=res.manifest())
    wdir = dest / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_bytes(files.pop("metrics.json"))
    for name, data in files.items():
        (wdir / name).write_bytes(data)
    print(f"weights + MODEL_CARD.md -> {wdir} (pining: never bundle or publish)")


if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
