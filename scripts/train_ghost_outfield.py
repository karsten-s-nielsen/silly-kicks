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


def _home_team_for(frames_path: Path, frames: pd.DataFrame) -> int | str | None:
    """The authoritative home team for the ``score_diff`` perspective: the ``meta.json`` sibling of a
    tc3 cache frames parquet, else the first non-ball ``team_id``.

    ``score_diff`` is signed from HOME's perspective, so an arbitrary first-team fallback flips the
    sign on any match whose frame row order leads with the away team (measured on IDSSE:
    ``ids.iloc[0]`` disagreed with ``meta.home_team_id``). ``home_team_id`` is consulted ONLY for the
    score lookup --- orientation is GoalMap-based (ADR-055) --- so a wrong home corrupts the sign of
    one feature, never the geometry. Returns ``None`` on an all-ball / empty frame (the caller drops it).
    """
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
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel

    groups = data["game_id"].astype(str).to_numpy()
    providers = (
        data["source_provider"].astype(str).to_numpy()
        if "source_provider" in data.columns
        else np.array(["unknown"] * len(data))
    )
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        return None, {}, {}, {}
    n_splits = min(cv_folds, n_groups)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        splits = list(splitter.split(data, y=providers, groups=groups))
    except ValueError as exc:
        import warnings

        warnings.warn(
            f"StratifiedGroupKFold infeasible ({exc}); falling back to GroupKFold (per-provider "
            "balance not enforced this run).",
            stacklevel=2,
        )
        splits = list(GroupKFold(n_splits=n_splits).split(data, groups=groups))

    fold_euclid: list[float] = []
    slot_euclid: dict[int, list[float]] = {}
    prov_euclid: dict[str, list[float]] = {}
    poss_euclid: dict[str, list[float]] = {}
    for tr, te in splits:
        train, test = data.iloc[tr], data.iloc[te]
        m = GhostOutfieldModel(n_estimators=n_estimators, max_depth=max_depth, feature_set=feature_set)._fit_extracted(
            train
        )
        preds = m.predict_mean(test)
        d = np.sqrt((preds[:, 0] - test["target_x"].to_numpy()) ** 2 + (preds[:, 1] - test["target_y"].to_numpy()) ** 2)
        fold_euclid.append(float(np.mean(d)))
        te_slots = test["slot_index"].to_numpy()
        for slot in sorted(set(te_slots.tolist())):
            slot_euclid.setdefault(int(slot), []).append(float(np.mean(d[te_slots == slot])))
        te_provs = providers[te]
        for prov in sorted(set(te_provs.tolist())):
            prov_euclid.setdefault(prov, []).append(float(np.mean(d[te_provs == prov])))
        # Per possession regime: the model serves the IN-POSSESSION slice for rest defense, so its
        # in-possession MAE is the acceptance-relevant number (distinct from the out-of-possession line).
        te_poss = test["team_in_possession"].to_numpy()
        for pv in sorted(set(te_poss.tolist())):
            label = "in_possession" if pv == 1.0 else "out_of_possession"
            poss_euclid.setdefault(label, []).append(float(np.mean(d[te_poss == pv])))
    per_slot = {str(int(s)): float(np.mean(v)) for s, v in sorted(slot_euclid.items())}
    per_provider = {p: float(np.mean(v)) for p, v in sorted(prov_euclid.items())}
    per_possession = {k: float(np.mean(v)) for k, v in sorted(poss_euclid.items())}
    return float(np.mean(fold_euclid)), per_slot, per_provider, per_possession


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

    from scripts._driver import for_each, shard_path

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
            home_team_id=_home_team_for(p, frames),
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
    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys) if len(f)]
    if not parts:
        print("ERROR: extractor produced no training rows across the corpus", file=sys.stderr)
        sys.exit(1)
    data = pd.concat(parts, ignore_index=True)
    print(
        f"Extracted {len(data)} rows across {data['game_id'].astype(str).nunique()} game(s) in {time.time() - t0:.1f}s"
    )

    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel, ghost_rearguard_coherence

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
