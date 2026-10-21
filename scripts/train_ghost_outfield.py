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
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._stages import ItemStage, Param, Pipeline, Stage, main

_FeatureSet = Literal["faithful", "position_only"]


def feature_set_for_variant(variant: str) -> _FeatureSet:
    """Map the CLI ``--variant`` onto the model feature set."""
    return "position_only" if variant == "position_only" else "faithful"


def extraction_inputs(variant: str, subsample_fps: float | None = 1.0) -> dict:
    """DECLARED shard-generation inputs (the digest ``for_each`` keys shards on).

    ``feature_set`` MUST be here (the 4.77.1 stale-shard rule): a position_only run drops the 4
    velocity features, so its shards carry a different column set and are NOT interchangeable with a
    faithful run's. ``subsample_fps`` MUST be here too: it changes the ROW SET (fewer, temporally
    thinned frames), so a 1 fps shard is not interchangeable with a 25 fps one. Bumping the
    extractor's output schema means bumping ``extractor`` here.
    """
    return {
        "schema": "ghost-outfield-1",
        "feature_set": feature_set_for_variant(variant),
        # v2 = possession-conditioned extraction (BOTH teams per frame, live team_in_possession); a v1
        # in-possession-only shard is NOT interchangeable, so the token change forces a fresh generation.
        "extractor": "ghost_outfield_v2",
        "subsample_fps": subsample_fps,
    }


def _subsample_frames(frames: pd.DataFrame, subsample_fps: float | None) -> pd.DataFrame:
    """Keep ~``subsample_fps`` frames/sec per ``(game_id, period_id)`` --- a VERBATIM mirror of the
    ghost-GK trainer's subsampler (``prepare_ghost_gk_training_data``).

    Tracking is ~25 fps and consecutive frames are near-duplicates, so a league-average positioning
    model trains on 1 fps by default: ~25x fewer, highly-redundant rows -> a tractable fit with no
    meaningful signal loss. No-op when ``subsample_fps`` is ``None``/non-positive, ``frame_rate`` is
    absent/non-positive, or the computed step is 1.
    """
    if subsample_fps is None or "frame_rate" not in frames.columns or len(frames) == 0:
        return frames
    fr = frames["frame_rate"].iloc[0]
    if not (fr > 0 and subsample_fps > 0):
        return frames
    step = max(1, round(fr / subsample_fps))
    if step == 1:
        return frames
    uniq = (
        frames[["game_id", "period_id", "frame_id"]].drop_duplicates().sort_values(["game_id", "period_id", "frame_id"])
    )
    keep_mask = uniq.groupby(["game_id", "period_id"]).cumcount() % step == 0
    keep = uniq[keep_mask.values]
    return frames.merge(keep, on=["game_id", "period_id", "frame_id"])


def extract_match(
    frames: pd.DataFrame,
    actions: pd.DataFrame | None,
    feature_set: _FeatureSet,
    *,
    n_rearguard: int = 4,
    home_team_id: int | str | None = None,
    subsample_fps: float | None = None,
):
    """Extract per-(frame, rearguard-slot) features for ONE match; ``None`` on zero rows.

    Frames are thinned to ``subsample_fps`` frames/sec FIRST (``None`` -> keep all), mirroring the
    ghost-GK trainer -- 25 fps tracking is far more (near-duplicate) rows than a mean-positioning model
    needs. Then the in-possession team is resolved from ``team_in_possession`` when present, else the
    ball carrier is inferred (the rearguard's owner). ``home_team_id`` (score perspective) is
    caller-supplied (the tc3 ``meta.json`` home); when ``None`` it falls back to the first non-ball
    team --- consulted only when ``actions`` carry a goal, and only for the sign of ``score_diff``
    (never orientation).
    """
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier
    from silly_kicks.tracking._ghost_outfield import _extract_all_ghost_outfield_features

    frames = _subsample_frames(frames, subsample_fps)
    non_ball = frames[~frames["is_ball"].astype(bool)]
    ids = non_ball["team_id"].dropna()
    if ids.empty:
        return None
    home = home_team_id if home_team_id is not None else ids.iloc[0]
    carrier = None
    if "team_in_possession" not in frames.columns:
        carrier_params: dict = dict(DEFAULT_CARRIER_PARAMS)
        c = infer_ball_carrier(frames, **carrier_params)
        carrier = c[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]
    feats = _extract_all_ghost_outfield_features(
        frames,
        actions,
        home_team_id=home,
        carrier=carrier,
        feature_set=feature_set,
        n_rearguard=n_rearguard,
        both_teams=True,  # possession-conditioned training: model BOTH teams' deepest-n per frame
    )
    if feats is None or len(feats) == 0:
        return None
    # Tag the match's provider for the StratifiedGroupKFold(match+provider) CV + per-provider MAE
    # (spec 6). A match has ONE provider; absent -> "unknown".
    prov = "unknown"
    if "source_provider" in frames.columns:
        present = frames["source_provider"].dropna()
        if len(present):
            prov = str(present.iloc[0])
    feats = feats.copy()
    feats["source_provider"] = prov
    return feats


def _actions_for(frames_path: Path, args) -> pd.DataFrame | None:
    """Locate the SPADL actions parquet mirroring ``frames_path`` under ``--actions-dir`` (or None).

    The tc3 ghost cache is ``{provider}/{game}/frames.parquet`` with a SEPARATE flat
    ``_actions/{game}.parquet`` (the game DIRECTORY name is the actions key --- dir name == game_id ==
    actions filename across gradientsports / skillcorner / idsse, verified on the ghost_cache corpus).
    An earlier resolver only tried ``_actions/{provider}/{game}/actions.parquet`` and
    ``_actions/frames.parquet`` (``rel.stem`` is ``"frames"`` under the nested layout), so it matched
    NOTHING on the real corpus and every ``phase`` / ``score_diff`` feature trained constant-0. The
    flat ``{game}.parquet`` toy layout masked this. Candidates are tried in order.
    """
    if args.actions_dir is None:
        return None
    rel = frames_path.relative_to(args.data_dir)
    candidates: list[Path] = []
    if len(rel.parts) > 1:
        # tc3 cache: {provider}/{game}/frames.parquet -> flat _actions/{game}.parquet (game = dir name)
        candidates.append(args.actions_dir / f"{rel.parts[-2]}.parquet")
        # nested variant: _actions/{provider}/{game}/actions.parquet
        candidates.append(args.actions_dir / rel.parent / "actions.parquet")
    # flat frames layout: {game}.parquet -> _actions/{game}.parquet
    candidates.append(args.actions_dir / f"{rel.stem}.parquet")
    from scripts._driver_sources import read_dataset_parquet

    for cand in candidates:
        if cand.exists():
            return read_dataset_parquet(cand)
    return None


def _home_team_for(frames_path: Path, frames: pd.DataFrame, home_map: dict | None = None) -> int | str | None:
    """The authoritative home team for the ``score_diff`` perspective: a ``--home-teams`` map keyed by
    ``game_id`` (a FLAT for_each generation has no per-match ``meta.json``), else the ``meta.json``
    sibling of a tc3-tree frames parquet, else the first non-ball ``team_id``.

    ``score_diff`` is signed from HOME's perspective, so an arbitrary first-team fallback flips the
    sign on any match whose frame row order leads with the away team (measured on IDSSE:
    ``ids.iloc[0]`` disagreed with ``meta.home_team_id``). ``home_team_id`` is consulted ONLY for the
    score lookup --- orientation is GoalMap-based (ADR-055) --- so a wrong home corrupts the sign of
    one feature, never the geometry. Returns ``None`` on an all-ball / empty frame (the caller drops it).
    """
    if home_map is not None:
        gids = frames["game_id"].dropna()
        if len(gids):
            hv = home_map.get(str(gids.iloc[0]))
            if hv is not None:
                return hv
    meta = frames_path.parent / "meta.json"
    if meta.exists():
        from scripts._driver_sources import read_dataset_json

        try:
            return read_dataset_json(meta)["home_team_id"]
        except (KeyError, ValueError, OSError):
            pass
    ids = frames[~frames["is_ball"].astype(bool)]["team_id"].dropna()
    return ids.iloc[0] if len(ids) else None


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


def _home_map(ctx) -> dict | None:
    from scripts._driver_sources import home_teams

    return home_teams(ctx.data_root, ctx.params["dataset"]) or None


def _extract_refs(ctx):
    from scripts._driver_sources import staged_flat_refs

    return staged_flat_refs(ctx.data_root, ctx.params["dataset"])


def _extract_load(ref, ctx):
    """``(frames_path, legacy args namespace)``: the legacy ``_work`` reads the frames and resolves the
    actions / home team through ``_actions_for`` / ``_home_team_for`` unchanged (decision P16)."""
    import types

    from scripts._driver_sources import dataset_dir

    root = dataset_dir(ctx.data_root, ctx.params["dataset"])
    frames_path = root / ref.files[0]
    ns = types.SimpleNamespace(data_dir=frames_path.parent, actions_dir=root / "_actions")
    return frames_path, ns


def _extract_work(item, ctx):
    from scripts._driver_sources import read_dataset_parquet

    p, ns = item
    frames = read_dataset_parquet(p)
    return extract_match(
        frames,
        _actions_for(p, ns),
        feature_set_for_variant(ctx.params["variant"]),
        n_rearguard=ctx.params["n_rearguard"],
        home_team_id=_home_team_for(p, frames, _home_map(ctx)),
        subsample_fps=ctx.params["subsample_fps"],
    )


def _extract_reduce(results, ctx):
    """The legacy combine: NON-EMPTY shards in sorted key order (``res.shard_keys`` order)."""
    from scripts._stages import combine_frames

    # a multi-provider corpus mixes int and str ids (gradientsports vs idsse / skillcorner): stored as the
    # canonical strings the legacy grouped on
    return combine_frames([f for f in results.values() if len(f)], ids=("game_id", "player_id", "team_id"))


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
    training_commit = None if ctx.prov["commit"] == "unknown" else ctx.prov["commit"]
    final = GhostOutfieldModel(n_estimators=p["n_estimators"], max_depth=p["max_depth"], feature_set=fs)._fit_extracted(
        data
    )
    final.training_commit = training_commit
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
        "run_commit": None if ctx.prov["commit"] == "unknown" else ctx.prov["commit"],
        "run_tree_dirty": bool(ctx.prov["dirty"]),
        "training_platform": p["training_platform"],
    }
    files = {name: data_ for name, data_ in fit.files.items() if name != "coherence.json"}
    files["metrics.json"] = json.dumps(metrics, indent=2).encode("utf-8")  # the legacy json.dump bytes
    return FileSet(files)


PIPELINE = Pipeline(
    name="ghost_outfield",
    module="scripts.train_ghost_outfield",
    params=_PARAMS,
    stages=(
        ItemStage(
            name="extract",
            refs=_extract_refs,
            key=lambda ref: ref.stem,  # the legacy flat-layout `_key`: the shard stem; _KEY_EXCEPTIONS
            load=_extract_load,
            work=_extract_work,
            reduce=_extract_reduce,
            # extraction_inputs is the legacy token; n_rearguard is declared too, so a changed rearguard
            # size can never reuse stale shards.
            params=("n_rearguard",),
            declare=lambda p: extraction_inputs(p["variant"], p["subsample_fps"]),
            cross_commit_version="ghost-outfield-extract-1",
            uses_dataset=True,
            max_failures=None,  # legacy WARNs and continues
        ),
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
