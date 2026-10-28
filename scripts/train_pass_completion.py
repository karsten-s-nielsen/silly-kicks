"""Owner-run trainer: bundle the default PassCompletionModel weights (TF-54b).

Fits ``silly_kicks.expected_passing.PassCompletionModel`` on the PUBLIC corpus and writes the
pickle-free JSON + SHA256SUMS artifact via ``model.save(...)``, plus an out-of-fold ``metrics.json``
stamping ``training_commit`` (ADR-052 / ADR-011 discipline, mirroring the five weight trainers).
Inference imports no sklearn; sklearn is used only during the fit.

ADR-128 stage contract (HFU R3-10 / D12; it replaced the one-unit adapter): ``refs`` (the competition set +
match references, one unit) -> ``extract[match]`` (an ``ItemStage``: one shard of finite-coordinate pass rows
per open-data match, fetched per unit, resume-before-download) -> ``cv[fold]`` (one unit per GroupKFold fold:
its out-of-fold predictions; the reduce assembles the OOF array in row order and computes the metrics) ->
``fit`` (one unit: the pooled fit and the bundle). ``--run-all --out <DIR>`` writes the flat bundle into
``<DIR>``: the model files, ``MODEL_CARD.md`` and ``metrics.json`` (owner ruling 2026-10-30: the bundle ships
``metrics.json``, as every other campaign-trained bundle does); ``adopt`` copies it into
``silly_kicks/expected_passing/weights/``. The legacy CLI is kept verbatim as the test oracle
(``tests/scripts/_legacy_trainers``).

Usage (StatsBomb credentials UNSET so open-data mode is used):
  python -m scripts.train_pass_completion --store <DIR> --run-all --out <DIR> \
      [--all-competitions | --competition-id C --season-id S] [--max-per-provider N] [--match-ids-json FILE]

The owner-tier pining cross-check (``--source pining``, DGX only) runs from
``scripts/_pass_completion_pining.py``, which owns its own ``for_each`` flow over this module's pure functions.
This module runs on HF, so it never imports the pining client: the I1 gate walks its static import closure
(ADR-123 Amendment B).
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, cli_main, int_key

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


def _labels(pooled: pd.DataFrame) -> np.ndarray:
    return (pooled["result_id"].to_numpy() == spadlconfig.result_id["success"]).astype(int)


def cv_folds(pooled: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """The GroupKFold-by-match ``(train, test)`` folds, or None when there are fewer than 2 groups (then the
    out-of-fold predictions are the in-sample predictions of one fit on everything)."""
    from sklearn.model_selection import GroupKFold

    groups = pooled["game_id"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return None
    return list(GroupKFold(n_splits=n_splits).split(pooled, _labels(pooled), groups))


def fold_predictions(pooled: pd.DataFrame, fold: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """The out-of-fold predictions of ONE fold (its test rows), or of the single in-sample fit (``None``)."""
    from silly_kicks.expected_passing import PassCompletionModel

    if fold is None:
        return _predict(PassCompletionModel().fit(pooled), pooled)
    tr, te = fold
    return _predict(PassCompletionModel().fit(pooled.iloc[tr]), pooled.iloc[te])


def cross_val_metrics(pooled: pd.DataFrame) -> dict:
    """GroupKFold-by-match out-of-fold AUC / ECE / Brier + the held-out base rate (pure).

    The base rate is the label mean over the finite-prediction rows -- the ``brier_noskill`` a Brier
    skill score is measured against downstream (spec PLAN-09).
    """
    folds = cv_folds(pooled)
    if folds is None:
        return metrics_from_oof(pooled, fold_predictions(pooled, None))
    oof = np.full(len(pooled), np.nan)
    for tr, te in folds:
        oof[te] = fold_predictions(pooled, (tr, te))
    return metrics_from_oof(pooled, oof)


def metrics_from_oof(pooled: pd.DataFrame, oof: np.ndarray) -> dict:
    """The out-of-fold metrics' arithmetic (``cross_val_metrics`` before the HFU R3-10 split, unchanged)."""
    from sklearn.metrics import roc_auc_score

    from silly_kicks._calibration_metrics import ece

    y = _labels(pooled)
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
    noskill = base * (1.0 - base)
    bss_line = ""
    if np.isfinite(brier) and np.isfinite(noskill) and noskill > 0.0:
        bss_line = f" (Brier skill score {1.0 - brier / noskill:.3f} vs the base-rate baseline)"
    # A source-faithful reproduce command: the full public open-data corpus uses --all-competitions;
    # a single tournament uses --source open-data --competition-id/--season-id; the pining cross-check
    # runs from its own entry with --source pining --providers.
    m_open = re.match(r"statsbomb-open \(competition (\d+), season (\d+)\)", str(providers))
    corpus_note = ""
    # Owner ruling 2026-10-30 (HFU R3-10): the open-data bundle names the PIPELINE command (the legacy CLI is
    # gone); the pining cross-check keeps its own entry and its exact text.
    command = "python -m scripts.train_pass_completion --store <DIR> --run-all --out <DIR>"
    if n_competitions is not None:
        reproduce_args = "--all-competitions"
        corpus_note = (
            f" This bundle was fit on the full public open-data corpus "
            f"({n_competitions} (competition, season) releases in the StatsBomb open-data manifest)."
        )
    elif m_open:
        reproduce_args = f"--source open-data --competition-id {m_open.group(1)} --season-id {m_open.group(2)}"
    else:
        # The corpus label is "pining:<providers>"; --providers takes the bare list.
        command = "python scripts/_pass_completion_pining.py --out <DIR>"
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
{brier:.3f} vs base rate {base:.3f}{bss_line}. See `metrics.json`.

**Missing-value policy.** A non-finite coordinate yields an all-NaN feature row and a NaN probability
(never a fabricated value); a consumer drops-and-counts such a target.

**Provenance + reproduction.** Reproduce with `{command}
{reproduce_args}`. `metrics.json` records `training_commit` ({commit}) and the tree state (this
bundle was produced from a clean tree). Pickle-free JSON + SHA256 envelope (`model.json` +
`SHA256SUMS`) with a feature contract + chirality probe; `load()` is fail-closed
(ADR-011/016/040/044/050). Every bundled model carries a card (ADR-088). Attribution:
expected-passing / pass-completion modelling -- see NOTICE.
"""


#: WC2022 (competition 43, season 106) -- the single-competition default corpus (open-data).
_DEFAULT_COMPETITION_ID = 43
_DEFAULT_SEASON_ID = 106


_PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param(
        "source",
        str,
        "open-data",
        choices=("open-data", "pining"),
        help="'open-data' only here; 'pining' runs from scripts/_pass_completion_pining.py (owner, DGX only)",
    ),
    Param("competition_id", int, None, help="open-data competition (default 43 = World Cup)"),
    Param("season_id", int, None, help="open-data season (default 106 = 2022)"),
    Param(
        "all_competitions",
        flag=True,
        help="re-fit on the FULL public open-data corpus (every (competition, season) in the open-data manifest)",
    ),
    Param("max_per_provider", int, None, help="cap the number of matches"),
    Param("match_ids_json", str, None, help='JSON {"statsbomb": ["3869685", ...]} pinning WHICH matches'),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)


def _match_ids(params) -> list[str] | None:
    """The ``--match-ids-json`` statsbomb selection, or None (no selection)."""
    path = params["match_ids_json"]
    if path is None:
        return None
    return (json.loads(Path(path).read_text(encoding="utf-8")) or {}).get("statsbomb")


def _refuse_bad_source(params) -> None:
    """I1 severance (ADR-123 Amendment B): this module runs on HF and never imports the pining client; the
    mutual exclusion of ``--all-competitions`` with an explicit competition is the legacy CLI's."""
    if params["source"] == "pining":
        raise ValueError(
            "--source pining runs from scripts/_pass_completion_pining.py (owner, DGX only): this module runs "
            "on HF and never imports the pining client (I1 gate, ADR-123 Amendment B)"
        )
    if params["all_competitions"] and (params["competition_id"] is not None or params["season_id"] is not None):
        raise ValueError("--all-competitions is mutually exclusive with --competition-id/--season-id")


def _refs_work(key, ctx) -> dict:
    """The corpus and every match reference, as one JSON unit (the legacy corpus selection: the public-only
    guard runs FIRST, before any manifest call or load)."""
    import scripts._sb_open_data as sbod

    p = ctx.params
    _refuse_bad_source(p)
    sbod.assert_statsbomb_open_data_mode()
    if p["all_competitions"]:
        comps = [(int(c[0]), int(c[1])) for c in sbod.all_open_competitions()]
        label = f"statsbomb-open (all {len(comps)} open-data competitions)"
    else:
        cid = _DEFAULT_COMPETITION_ID if p["competition_id"] is None else p["competition_id"]
        sid = _DEFAULT_SEASON_ID if p["season_id"] is None else p["season_id"]
        comps = [(cid, sid)]
        label = f"statsbomb-open (competition {cid}, season {sid})"
    refs = sbod.list_open_data_refs(comps, match_ids=_match_ids(p), max_matches=p["max_per_provider"])
    return {
        "competitions": [list(c) for c in comps],
        "all_competitions": bool(p["all_competitions"]),
        "label": label,
        "refs": [dataclasses.asdict(r) for r in refs],
    }


def _extract_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def _load_match(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))


def _extract_work(item, ctx) -> pd.DataFrame:
    return pass_rows_for_match(item.actions, item.match_id)


def _extract_reduce(results, ctx) -> pd.DataFrame:
    from scripts._stages import combine_frames

    frames = list(results.values())  # sorted key order == the legacy sorted shard glob
    return combine_frames(frames, ids=("game_id",)) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


_POOLED: dict = {}


def _pooled(ctx) -> pd.DataFrame:
    """The pooled pass rows, loaded ONCE per worker; refuses an empty corpus. Keyed by the store AND the extract
    token: one token can name different corpora in different stores (``--all-competitions`` lists the manifest at
    run time), so a process that serves two stores must never reuse the first one's rows."""
    tok = (str(getattr(ctx.store, "root", getattr(ctx.store, "prefix", ""))), ctx.tokens["extract"])
    if tok not in _POOLED:
        pooled = ctx.upstream("extract")
        if not len(pooled):
            raise RuntimeError("no pass rows collected from the corpus; nothing to fit")
        _POOLED.clear()
        _POOLED[tok] = pooled
    return _POOLED[tok]


def _cv_units(ctx) -> list[str]:
    """``max(1, min(5, n_done))`` fold keys, from the extract SUMMARY, never from its output (the controller's
    ``--plan`` reads the bucket every step; the ghost_outfield precedent). The real split count comes from the
    pooled rows in ``_cv_work``: a key past it returns an empty part, and with fewer than 2 groups key ``0000``
    carries the single in-sample fit."""
    n = int(ctx.summary("extract")["n_done"])
    return [int_key(k) for k in range(max(1, min(5, n)))]


def _cv_work(key, ctx) -> pd.DataFrame:
    """One fold's out-of-fold predictions as ``(row, pred)``; empty for a key past the real split."""
    pooled = _pooled(ctx)
    k = int(key)
    folds = cv_folds(pooled)
    if folds is None:
        rows, preds = (np.arange(len(pooled)), fold_predictions(pooled, None)) if k == 0 else ([], [])
    elif k < len(folds):
        rows, preds = folds[k][1], fold_predictions(pooled, folds[k])
    else:
        rows, preds = [], []
    return pd.DataFrame({"row": np.asarray(rows, dtype=np.int64), "pred": np.asarray(preds, dtype=float)})


def _cv_reduce(results, ctx) -> dict:
    """The OOF array in row order (each fold writes its own test rows; the single fit writes every row), then
    the legacy metrics arithmetic."""
    pooled = _pooled(ctx)
    if cv_folds(pooled) is None:
        oof = results[int_key(0)]["pred"].to_numpy(dtype=float)
    else:
        oof = np.full(len(pooled), np.nan)
        for part in results.values():
            oof[part["row"].to_numpy()] = part["pred"].to_numpy(dtype=float)
    return metrics_from_oof(pooled, oof)


def _fit_work(key, ctx) -> FileSet:
    """The legacy reduce tail: the pooled fit, ``metrics.json`` (legacy key order) and the card, as ONE flat
    bundle (an artifact is flat: ``adopt.retrieve`` refuses a nested path). Owner ruling 2026-10-30: the bundle
    ships ``metrics.json`` beside the weights and the card, as every other campaign-trained bundle does."""
    import tempfile

    from silly_kicks.expected_passing import PassCompletionModel

    pooled = _pooled(ctx)
    corpus = ctx.upstream("refs")
    model = PassCompletionModel().fit(pooled)
    prov = ctx.prov
    summary = ctx.summary("extract")
    # Decision P4 (ADR-128): CORPUS-scoped counters, never pass-scoped ones (the legacy `res.manifest()` keys).
    pass_fields = {
        "generation": ctx.tokens["extract"],
        "n_attempted": summary["n_units"],
        "n_failed": summary["n_failed"],
        "n_counters_unrecorded": 0,
        "n_excluded": summary["n_excluded"],
    }
    out = {
        **ctx.upstream("cv"),
        "n_pass_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": corpus["label"],
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **pass_fields,
    }
    if corpus["all_competitions"]:
        comps = [tuple(c) for c in corpus["competitions"]]
        out["n_competitions"] = len(comps)
        out["competitions"] = [list(c) for c in sorted(comps)]
    payload = json.dumps(out, indent=2, default=str)
    with tempfile.TemporaryDirectory() as tmp:
        wdir = Path(tmp) / "weights"
        model.save(wdir)
        files = {p.name: p.read_bytes() for p in sorted(wdir.iterdir()) if p.is_file()}
    files["MODEL_CARD.md"] = render_model_card(out).encode("utf-8")
    files["metrics.json"] = payload.encode("utf-8")
    print(payload)
    return FileSet(files)


#: HFU plan R3-8 S3 item 12: the reviewed serial reasons of this pipeline's SERIAL stages, verbatim from the
#: plan's SERIAL annex (owner ruling 2026-10-30, "too small to pay"); `scripts/hf_jobs/serial_reasons.py`
#: holds the same texts and `tests/scripts/test_serial_reasons.py` holds the two equal.
_SERIAL_REASONS = {
    "refs": ("Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."),
    "extract": (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 64 units, 28 s of CPU in all; W=16 "
        "measured slower than W=8 (14.8 s against 13.3 s) and spent 58 % more CPU."
    ),
    "cv": ("Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): 5 folds of 0.3 s, 1 s of CPU in all."),
    "fit": ("Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, under 1 s of CPU."),
}


PIPELINE = Pipeline(
    name="pass_completion",
    module="scripts.train_pass_completion",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            serial_reason=_SERIAL_REASONS["refs"],  # S3 item 12 (SERIAL annex)
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=("source", "competition_id", "season_id", "all_competitions", "max_per_provider"),
            # the match selection is declared by CONTENT, never by path (the 4.77.1 stale-shard rule)
            declare=lambda p: {"match_ids": _match_ids(p)},
            uses_dataset=True,
            dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
        ),
        ItemStage(
            name="extract",
            serial_reason=_SERIAL_REASONS["extract"],  # S3 item 12 (SERIAL annex)
            refs=_extract_refs,
            key=lambda ref: ref.key,
            load=_load_match,
            work=_extract_work,
            needs=("refs",),
            reduce=_extract_reduce,
            declare=lambda p: {"model": "PassCompletionModel", "cols": list(_SHARD_COLUMNS)},
            cross_commit_version="pass-completion-shard-1",
            uses_dataset=True,
            max_failures=None,  # legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="cv",
            serial_reason=_SERIAL_REASONS["cv"],  # S3 item 12 (SERIAL annex)
            units=_cv_units,
            work=_cv_work,
            needs=("extract",),
            reduce=_cv_reduce,
        ),
        Stage(
            name="fit",
            serial_reason=_SERIAL_REASONS["fit"],  # S3 item 12 (SERIAL annex)
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("refs", "extract", "cv"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="fit",
)

if __name__ == "__main__":
    cli_main(PIPELINE)
