#!/usr/bin/env python
"""Train a ghost-outfield rearguard-positioning model (TF-60 PR5).

Usage (ADR-128 stage contract; the uniform CLI from ``scripts._stages.main``)::

    python -m scripts.train_ghost_outfield --dataset tc3 --variant default \
        --store <DIR> --data-root <DATA> --run-all --out <BUNDLE_DIR>

Pipeline: ``extract[match]`` (an ``ItemStage`` over the staged flat dataset: ``shards/<gen>/`` frames,
``_actions/`` and ``home_teams.json``) -> ``cv[fold]`` -> ``fit`` -> ``assemble`` (the bundle). Each frames
parquet must carry a ``team_in_possession`` column or a resolvable ball carrier. The extract token includes
``feature_set`` (the 4.77.1 stale-shard rule: a position_only run drops 4 columns, so its shards are NOT
interchangeable with a faithful run).

Requires: silly-kicks[train].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._ghost_outfield_stages import STAGES as _REUSABLE  # ADR-129: the reusable extract stage
from scripts._ghost_outfield_stages import (
    _actions_for,  # noqa: F401 -- re-export: tests
    _extract_reduce,  # noqa: F401 -- re-export: tests
    _home_team_for,  # noqa: F401 -- re-export: tests
    _subsample_frames,  # noqa: F401 -- re-export: tests
    extract_match,  # noqa: F401 -- re-export: tests
    extraction_inputs,  # noqa: F401 -- re-export: tests
    feature_set_for_variant,
)
from scripts._stages import Param, Pipeline, Stage, main


def _cv_splits(data: pd.DataFrame, cv_folds: int):
    """The deterministic CV split list + the provider array. ``(None, _)`` when the corpus has < 2 games
    (CV infeasible).

    Match-stratified: **StratifiedGroupKFold(groups=game_id, y=source_provider)** keeps every game
    intact while balancing the provider mix across folds (mirroring the ghost-GK trainer). Falls back to
    ``GroupKFold`` only if the stratification is infeasible (e.g. a provider with fewer games than
    folds), warning so the reduction is visible, never silent. Every ``cv`` unit and the reduce compute
    the same split, so a fanned CV equals a serial one for every (W, N).
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    groups = data["game_id"].astype(str).to_numpy()
    providers = (
        data["source_provider"].astype(str).to_numpy()
        if "source_provider" in data.columns
        else np.array(["unknown"] * len(data))
    )
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        return None, providers
    n_splits = min(cv_folds, n_groups)
    try:
        splits = list(StratifiedGroupKFold(n_splits=n_splits).split(data, y=providers, groups=groups))
    except ValueError as exc:
        import warnings

        warnings.warn(
            f"StratifiedGroupKFold infeasible ({exc}); falling back to GroupKFold (per-provider "
            "balance not enforced this run).",
            stacklevel=2,
        )
        splits = list(GroupKFold(n_splits=n_splits).split(data, groups=groups))
    return splits, providers


def _run_one_fold_of(data, providers, tr, te, *, feature_set, n_estimators, max_depth) -> dict:
    """One CV fold's euclidean-MAE partials (overall + per-slot/provider/possession), as plain floats so a
    ``cv`` unit can persist them to JSON and the reduce can aggregate byte-identically."""
    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel

    train, test = data.iloc[tr], data.iloc[te]
    m = GhostOutfieldModel(n_estimators=n_estimators, max_depth=max_depth, feature_set=feature_set)._fit_extracted(
        train
    )
    preds = m.predict_mean(test)
    d = np.sqrt((preds[:, 0] - test["target_x"].to_numpy()) ** 2 + (preds[:, 1] - test["target_y"].to_numpy()) ** 2)
    te_slots = test["slot_index"].to_numpy()
    te_provs = providers[te]
    te_poss = test["team_in_possession"].to_numpy()
    return {
        "fold_euclid": float(np.mean(d)),
        "per_slot": {str(int(s)): float(np.mean(d[te_slots == s])) for s in sorted(set(te_slots.tolist()))},
        "per_provider": {p: float(np.mean(d[te_provs == p])) for p in sorted(set(te_provs.tolist()))},
        "per_possession": {
            ("in_possession" if pv == 1.0 else "out_of_possession"): float(np.mean(d[te_poss == pv]))
            for pv in sorted(set(te_poss.tolist()))
        },
    }


def _aggregate_folds(fold_dicts: list[dict]):
    """Mean the per-fold partials into (cv_mae, per_slot, per_provider, per_possession) -- the same
    reduction the inline loop did, so serial == fanned."""
    slot_euclid: dict[str, list[float]] = {}
    prov_euclid: dict[str, list[float]] = {}
    poss_euclid: dict[str, list[float]] = {}
    for fd in fold_dicts:
        for s, v in fd["per_slot"].items():
            slot_euclid.setdefault(s, []).append(v)
        for p, v in fd["per_provider"].items():
            prov_euclid.setdefault(p, []).append(v)
        for k, v in fd["per_possession"].items():
            poss_euclid.setdefault(k, []).append(v)
    per_slot = {s: float(np.mean(v)) for s, v in sorted(slot_euclid.items(), key=lambda kv: int(kv[0]))}
    per_provider = {p: float(np.mean(v)) for p, v in sorted(prov_euclid.items())}
    per_possession = {k: float(np.mean(v)) for k, v in sorted(poss_euclid.items())}
    return float(np.mean([fd["fold_euclid"] for fd in fold_dicts])), per_slot, per_provider, per_possession


# ---- ADR-128 stage contract --------------------------------------------------------------------------
#
# extract[match] -> cv[fold] -> fit (one unit) -> assemble (one unit, the bundle). No whole-corpus cache
# stage (ADR-123 section 1): `data` is a cheap concat of the extract shards, loaded once per worker.

_PARAMS = (
    Param(
        "dataset",
        str,
        None,
        required=True,
        help="the staged flat dataset (shards/<gen>/ + _actions/ + home_teams.json)",
    ),
    Param(
        "variant",
        str,
        "default",
        choices=("default", "position_only"),
        help="default = faithful; position_only drops velocity",
    ),
    Param("n_estimators", int, 500),
    Param("max_depth", int, 8),
    Param("cv_folds", int, 5),
    Param("n_rearguard", int, 4),
    Param(
        "subsample_fps",
        float,
        1.0,
        help="Frames/sec to keep per (game, period) (default 1.0, mirroring train_ghost_gk); a large value keeps all",
    ),
    Param("training_platform", str, None),
)


_DATA: dict = {}


def _data(ctx) -> pd.DataFrame:
    """The extract output, loaded ONCE per worker (keyed by the extract token)."""
    tok = ctx.tokens["extract"]
    if tok not in _DATA:
        _DATA.clear()
        _DATA[tok] = ctx.upstream("extract")
    return _DATA[tok]


def _cv_units(ctx) -> list[str]:
    """``min(cv_folds, n_extracted_units)`` fold keys, from the extract SUMMARY.

    Never from the extract OUTPUT: the controller's ``--plan`` reads a bucket store, and the whole-corpus
    row table is gigabytes on tc3. The summary bounds the real split count (a game whose extraction is
    empty yields no group), and a key past the real split returns no fold (see `_cv_work`)."""
    from scripts._stages import int_key

    n = int(ctx.summary("extract")["n_done"])
    return [int_key(k) for k in range(min(int(ctx.params["cv_folds"]), n))]


def _cv_work(key, ctx) -> dict:
    from scripts._train_guard import require_training_sklearn

    require_training_sklearn()
    data = _data(ctx)
    if not len(data):
        return {"skipped": "no extracted rows"}
    splits, providers = _cv_splits(data, ctx.params["cv_folds"])
    if splits is None or int(key) >= len(splits):
        return {"skipped": "CV infeasible or past the real split count (an empty game)"}
    tr, te = splits[int(key)]
    p = ctx.params
    return _run_one_fold_of(
        data,
        providers,
        tr,
        te,
        feature_set=feature_set_for_variant(p["variant"]),
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
    )


def _cv_reduce(results, ctx) -> list:
    """``[cv_mae, per_slot, per_provider, per_possession]`` from the folds in FOLD order (``_aggregate_folds``);
    the ``None`` tuple when CV is infeasible (< 2 games), exactly as the pre-contract serial CV
    (``_cross_validate`` at ``cd2ebcb``)."""
    folds = [results[k] for k in sorted(results) if "skipped" not in results[k]]
    if not folds:
        return [None, {}, {}, {}]
    return list(_aggregate_folds(folds))


def _fit_work(key, ctx):
    import tempfile

    from scripts._stages import FileSet
    from scripts._train_guard import require_training_sklearn
    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel, ghost_rearguard_coherence

    require_training_sklearn()
    p = ctx.params
    data = _data(ctx)
    if not len(data):
        raise RuntimeError("extractor produced no training rows across the corpus")
    fs = feature_set_for_variant(p["variant"])
    final = GhostOutfieldModel(n_estimators=p["n_estimators"], max_depth=p["max_depth"], feature_set=fs)._fit_extracted(
        data
    )
    # S5.10: a stage reads no ctx.prov; adopt.stamp_provenance stamps training_commit at adopt.
    final.training_platform = p["training_platform"]
    final.corpus_provenance = {
        "n_games": int(data["game_id"].astype(str).nunique()),
        "n_rows": len(data),
        "variant": p["variant"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / p["variant"]
        final.save(artifact_dir)
        # Round-trip verify the serialized parameters (parameters-only artifact).
        loaded = GhostOutfieldModel.load(artifact_dir)
        ft, lt = final._tree_nodes, loaded._tree_nodes
        if ft is None or lt is None:
            raise RuntimeError("round-trip verification: the saved model carries no tree nodes")
        for a, b in zip(ft, lt, strict=True):
            np.testing.assert_array_equal(a, b)
        if loaded._baseline_x != final._baseline_x:
            raise RuntimeError("round-trip verification: the baseline differs after save/load")
        files = {q.name: q.read_bytes() for q in sorted(artifact_dir.iterdir()) if q.is_file()}
    # Coherence (reported, not gated -- spec 9): served ghost positions on the training rows.
    served = data[["game_id", "period_id", "frame_id", "team_id", "slot_index"]].copy()
    fpred = final.predict_mean(data)
    served["ghost_gr_x"] = fpred[:, 0]
    served["ghost_gr_y"] = fpred[:, 1]
    files["coherence.json"] = json.dumps(ghost_rearguard_coherence(served)).encode("utf-8")
    return FileSet(files)


def _assemble_work(key, ctx):
    from scripts._stages import FileSet

    p = ctx.params
    data = _data(ctx)
    cv_mae, cv_mae_by_slot, cv_mae_by_provider, cv_mae_by_possession = ctx.upstream("cv")
    fit = ctx.upstream("fit")
    coherence = json.loads(fit.files["coherence.json"])
    metrics = {
        "variant": p["variant"],
        "feature_set": feature_set_for_variant(p["variant"]),
        "subsample_fps": p["subsample_fps"],
        "n_games": int(data["game_id"].astype(str).nunique()),
        "n_rows": len(data),
        "cv_mae": cv_mae,
        "cv_mae_by_slot": cv_mae_by_slot,
        "cv_mae_by_provider": cv_mae_by_provider,
        "cv_mae_by_possession": cv_mae_by_possession,
        "coherence": coherence,
        "training_platform": p["training_platform"],
    }
    files = {name: data_ for name, data_ in fit.files.items() if name != "coherence.json"}
    files["metrics.json"] = json.dumps(metrics, indent=2).encode("utf-8")  # the legacy json.dump bytes
    from scripts.hf_jobs.identity import commit_free

    return FileSet(commit_free(files))  # S5.10: commit-free stage output


PIPELINE = Pipeline(
    name="ghost_outfield",
    module="scripts.train_ghost_outfield",
    params=_PARAMS,
    stages=(
        _REUSABLE["extract"],  # ADR-129: declared in scripts._ghost_outfield_stages
        Stage(
            name="cv",
            units=_cv_units,
            work=_cv_work,
            needs=("extract",),
            reduce=_cv_reduce,
            params=("variant", "n_estimators", "max_depth", "cv_folds"),
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],
            params=("variant", "n_estimators", "max_depth", "training_platform"),
        ),
        Stage(
            name="assemble",
            units=lambda ctx: ["only"],
            work=_assemble_work,
            needs=("extract", "cv", "fit"),
            reduce=lambda r, ctx: r["only"],
            params=("variant", "subsample_fps", "training_platform"),
        ),
    ),
    output="assemble",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
