#!/usr/bin/env python
"""Train a ghost-outfield rearguard-positioning model (TF-60 PR5).

Usage::

    uv run python scripts/train_ghost_outfield.py \
        --data-dir /path/to/tc3_cache/ \
        --variant default \
        --output-dir models/

Mirrors ``scripts/train_ghost_gk.py`` (``for_each`` corpus sharding + ``require_clean_tree``
provenance + parameters-only save) for the rest-defense rearguard model. ``--data-dir`` holds
tracking parquets (``{provider}/{game}/frames.parquet`` or flat ``*.parquet``); each must carry a
``team_in_possession`` column or a resolvable ball carrier. The shard-generation token includes
``feature_set`` (the 4.77.1 stale-shard rule: a position_only run drops 4 columns, so its shards are
NOT interchangeable with a faithful run).

Requires: silly-kicks[train].
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train a ghost-outfield rearguard-positioning model.")
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory of tracking parquets")
    ap.add_argument("--output-dir", type=Path, default=Path("models"), help="Where to save the artifact")
    ap.add_argument("--actions-dir", type=Path, default=None, help="Optional: directory of SPADL actions parquets")
    ap.add_argument(
        "--home-teams",
        type=Path,
        default=None,
        help="Optional JSON {game_id: home_team_id} (the materialize home_teams.json). REQUIRED for a "
        "FLAT for_each generation (no per-match meta.json) to sign score_diff correctly; falls back to "
        "the meta.json sibling, then the first-team heuristic.",
    )
    ap.add_argument(
        "--variant",
        choices=["default", "position_only"],
        default="default",
        help="Which variant this run produces (default = faithful; position_only drops velocity).",
    )
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-rearguard", type=int, default=4)
    ap.add_argument(
        "--subsample-fps",
        type=float,
        default=1.0,
        help="Frames/sec to keep per (game, period) for training (default 1.0, mirroring "
        "train_ghost_gk). Tracking is ~25 fps and consecutive frames are near-duplicates; 1 fps is "
        "~25x fewer rows with no meaningful signal loss for a mean-positioning model. Pass a large "
        "value (e.g. 999) to keep all frames.",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a dev run on a dirty tree; the artifact still records dirty:true (never launders it).",
    )
    ap.add_argument("--training-platform", default=None)
    # Map/reduce fan-out (ADR-123), mirroring train_ghost_gk: extract --shard-index -> --fold -> --assemble.
    # No --build-cache phase (unlike ghost_gk): this trainer keeps no whole-corpus feature cache; `data` is a
    # cheap concat of the (resumed) shards, rebuilt identically by the fold workers and the reduce.
    ap.add_argument(
        "--shard-index",
        default=None,
        help="I/N: extract only this disjoint subset of the discovered parquet files (fan-out worker); "
        "writes its shards to the shared shard_root and exits before the concat/CV.",
    )
    ap.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run ONLY CV fold K over the (resumed) shards, write {shard_root}/folds/fold_K.json, then exit.",
    )
    ap.add_argument(
        "--assemble",
        action="store_true",
        help="Reduce: concat shards, aggregate folds/fold_*.json, final fit + save + metrics, then exit.",
    )
    return ap.parse_args(argv)


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
    for cand in candidates:
        if cand.exists():
            return pd.read_parquet(cand)
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
        try:
            return json.loads(meta.read_text())["home_team_id"]
        except (KeyError, ValueError, OSError):
            pass
    ids = frames[~frames["is_ball"].astype(bool)]["team_id"].dropna()
    return ids.iloc[0] if len(ids) else None


def _cross_validate(data: pd.DataFrame, *, feature_set: _FeatureSet, n_estimators: int, max_depth: int, cv_folds: int):
    """CV euclidean MAE: overall + per-slot + **per-provider** (spec 6). ``None`` overall when < 2 games.

    Match-stratified: **StratifiedGroupKFold(groups=game_id, y=source_provider)** keeps every game
    intact while balancing the provider mix across folds (mirroring the ghost-GK trainer). Falls back to
    ``GroupKFold`` only if the stratification is infeasible (e.g. a provider with fewer games than
    folds), warning so the reduction is visible, never silent.
    """
    splits, providers = _cv_splits(data, cv_folds)
    if splits is None:
        return None, {}, {}, {}
    fold_dicts = [
        _run_one_fold_of(
            data, providers, tr, te, feature_set=feature_set, n_estimators=n_estimators, max_depth=max_depth
        )
        for tr, te in splits
    ]
    return _aggregate_folds(fold_dicts)


def _cv_splits(data: pd.DataFrame, cv_folds: int):
    """The deterministic CV split list + the provider array (ADR-123: computed identically by the serial
    run, the --fold workers and the --assemble reduce, so a fanned CV equals a serial one). ``(None, _)``
    when the corpus has < 2 games (CV infeasible)."""
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
    fanned fold worker can persist them to JSON and the reduce can aggregate byte-identically."""
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


def main(argv=None):
    args = parse_args(argv)

    # Refuse to fit a SHIPPABLE artifact under a scikit-learn outside the supported fit range: HGBR
    # trees differ across sklearn versions, so bundled weights must be fit on the pinned range
    # (mirrors train_ghost_gk; the @slow smoke skips where sklearn is out of range).
    from scripts._train_guard import require_training_sklearn

    print(f"scikit-learn {require_training_sklearn()} (within the supported fit range for bundled weights)")

    # FIRST, before any parquet is read: this trainer stamps `training_commit` into the SHIPPED
    # artifact, and a bare HEAD SHA is identical on a dirty tree -- a verifiable-looking false claim.
    from scripts._provenance import git_provenance, require_clean_tree

    run_prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fs = feature_set_for_variant(args.variant)
    training_commit = None if run_prov["commit"] == "unknown" else run_prov["commit"]
    tree_dirty = bool(run_prov["dirty"])
    training_platform = args.training_platform
    print(
        f"variant={args.variant} (feature_set={fs}); subsample_fps={args.subsample_fps}; "
        f"training_commit={training_commit} (tree_dirty={tree_dirty})"
    )

    parquets = sorted(args.data_dir.glob("**/frames.parquet")) or sorted(args.data_dir.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: No .parquet files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Discovered {len(parquets)} tracking parquet(s) in {args.data_dir}")

    # Map/reduce extract fan-out (ADR-123): slice to this worker's disjoint file subset (each flat parquet
    # is one game -> a file-index partition is a disjoint cover). --fold / --assemble run the FULL list.
    if args.shard_index is not None:
        from scripts._driver import shard_select

        _si, _sn = (int(x) for x in args.shard_index.split("/"))
        parquets = shard_select(parquets, _si, _sn)
        print(f"Shard {_si}/{_sn}: extracting {len(parquets)} of the discovered parquet files")
        if not parquets:
            print(f"ERROR: shard {args.shard_index} selects no files", file=sys.stderr)
            sys.exit(1)

    from scripts._driver import for_each, shard_path

    home_map = json.loads(args.home_teams.read_text()) if args.home_teams else None

    def _items():
        yield from parquets

    def _key(p: Path) -> str:
        # FLAT, injective shard key (no path separators -- for_each writes <gen>/<key>.parquet).
        # tc3 layout ({..}/{game}/frames.parquet) -> the dir path joined; flat *.parquet -> the stem.
        rel = p.relative_to(args.data_dir)
        return "__".join(rel.parts[:-1]) if len(rel.parts) > 1 else rel.stem

    def _work(p: Path):
        frames = pd.read_parquet(p)
        return extract_match(
            frames,
            _actions_for(p, args),
            fs,
            n_rearguard=args.n_rearguard,
            home_team_id=_home_team_for(p, frames, home_map),
            subsample_fps=args.subsample_fps,
        )

    shard_root = args.output_dir / "_shards"
    t0 = time.time()
    res = for_each(
        _items(),
        key=_key,
        work=_work,
        shard_root=shard_root,
        token_inputs=extraction_inputs(args.variant, args.subsample_fps),
    )
    if res.failures:
        print(f"WARN: {len(res.failures)} match(es) failed during extraction: {res.failures}")
    # Map/reduce extract fan-out (ADR-123): a --shard-index worker has written its subset's shards to the
    # shared shard_root; concat + CV + fit belong to --assemble. Exit so N workers compose without racing.
    if args.shard_index is not None and not args.assemble:
        print(f"Shard {args.shard_index}: extraction complete ({res.skipped} resumed); exiting before CV.")
        return
    # `res.shard_keys`, not `res.keys`: a failed match (this driver WARNs and continues rather than
    # raising) has no shard, so reading `res.keys` would FileNotFoundError on it (ADR-052 D13 / the
    # cycle's interp-11 combine contract). Byte-identical to `res.keys` when nothing failed.
    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys) if len(f)]
    if not parts:
        print("ERROR: extractor produced no training rows across the corpus", file=sys.stderr)
        sys.exit(1)
    data = pd.concat(parts, ignore_index=True)
    print(
        f"Extracted {len(data)} rows across {data['game_id'].astype(str).nunique()} game(s) in {time.time() - t0:.1f}s"
    )

    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel, ghost_rearguard_coherence

    # Map/reduce CV (ADR-123): --fold K runs one fold from the identically-computed split and persists it;
    # --assemble aggregates the fanned fold_*.json; a plain serial run uses the inline _cross_validate.
    fold_dir = shard_root / "folds"
    if args.fold is not None:
        splits, providers = _cv_splits(data, args.cv_folds)
        if splits is None:
            print("ERROR: CV infeasible (<2 games); --fold needs >=2 games", file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.fold < len(splits)):
            print(f"ERROR: --fold {args.fold} out of range [0,{len(splits)})", file=sys.stderr)
            sys.exit(2)
        fold_dir.mkdir(parents=True, exist_ok=True)
        _tr, _te = splits[args.fold]
        _d = _run_one_fold_of(
            data, providers, _tr, _te, feature_set=fs, n_estimators=args.n_estimators, max_depth=args.max_depth
        )
        (fold_dir / f"fold_{args.fold}.json").write_text(json.dumps(_d), encoding="utf-8")
        print(f"Fold {args.fold} done -> {fold_dir / f'fold_{args.fold}.json'}")
        return

    if args.assemble:
        splits, _providers = _cv_splits(data, args.cv_folds)
        if splits is None:
            cv_mae, cv_mae_by_slot, cv_mae_by_provider, cv_mae_by_possession = None, {}, {}, {}
        else:
            _fold_dicts = [
                json.loads((fold_dir / f"fold_{k}.json").read_text(encoding="utf-8")) for k in range(len(splits))
            ]
            cv_mae, cv_mae_by_slot, cv_mae_by_provider, cv_mae_by_possession = _aggregate_folds(_fold_dicts)
    else:
        cv_mae, cv_mae_by_slot, cv_mae_by_provider, cv_mae_by_possession = _cross_validate(
            data, feature_set=fs, n_estimators=args.n_estimators, max_depth=args.max_depth, cv_folds=args.cv_folds
        )
    print(f"CV euclidean MAE: {cv_mae}  per-slot: {cv_mae_by_slot}")
    print(f"  per-provider: {cv_mae_by_provider}  per-possession: {cv_mae_by_possession}")

    # --- Final fit + provenance ---
    final = GhostOutfieldModel(n_estimators=args.n_estimators, max_depth=args.max_depth, feature_set=fs)._fit_extracted(
        data
    )
    final.training_commit = training_commit
    final.training_platform = training_platform
    final.corpus_provenance = {
        "n_games": int(data["game_id"].astype(str).nunique()),
        "n_rows": len(data),
        "variant": args.variant,
    }
    artifact_dir = args.output_dir / args.variant
    final.save(artifact_dir)
    print(f"Model saved to {artifact_dir}")

    # Round-trip verify the serialized parameters (parameters-only artifact).
    loaded = GhostOutfieldModel.load(artifact_dir)
    ft, lt = final._tree_nodes, loaded._tree_nodes
    assert ft is not None and lt is not None  # noqa: S101
    for a, b in zip(ft, lt, strict=True):
        np.testing.assert_array_equal(a, b)
    assert loaded._baseline_x == final._baseline_x  # noqa: S101
    print("Round-trip verification: PASS")

    # Coherence (reported, not gated -- spec 9): served ghost positions on the training rows.
    served = data[["game_id", "period_id", "frame_id", "team_id", "slot_index"]].copy()
    fpred = final.predict_mean(data)
    served["ghost_gr_x"] = fpred[:, 0]
    served["ghost_gr_y"] = fpred[:, 1]
    coherence = ghost_rearguard_coherence(served)
    print(f"Ghost-rearguard coherence: {coherence}")

    metrics = {
        "variant": args.variant,
        "feature_set": fs,
        "subsample_fps": args.subsample_fps,
        "n_games": int(data["game_id"].astype(str).nunique()),
        "n_rows": len(data),
        "cv_mae": cv_mae,
        "cv_mae_by_slot": cv_mae_by_slot,
        "cv_mae_by_provider": cv_mae_by_provider,
        "cv_mae_by_possession": cv_mae_by_possession,
        "coherence": coherence,
        "run_commit": training_commit,
        "run_tree_dirty": tree_dirty,
        "training_platform": training_platform,
    }
    with open(artifact_dir / "metrics.json", "w", newline="\n") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {artifact_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
