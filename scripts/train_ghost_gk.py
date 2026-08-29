#!/usr/bin/env python
"""Train Ghost-GK positioning model (TF-18).

Usage:
    uv run python scripts/train_ghost_gk.py \
        --data-dir /path/to/tc3_cache/ \
        --output-dir models/ \
        --subsample-fps 1.0 \
        --n-estimators 500 \
        --max-depth 8 \
        --cv-folds 5

Supports two directory layouts (prefers TC3 when both exist):
  - TC3 cache: data-dir/{provider}/{game_id}/frames.parquet
    (auto-reads meta.json siblings for home_team_id)
  - Flat: data-dir/*.parquet

Override home team mapping with --home-teams JSON file.

Requires: silly-kicks installed (uv run handles this).

See docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def cache_token() -> str:
    """Feature-cache identity, DERIVED from the geometry constants the extractor consumes.

    Deriving rather than hand-bumping is the whole point: a literal version string goes stale
    inside the very re-fit cycle it exists to protect. The re-fit sequence is *extract features,
    flip the penalty-area constant, re-run* -- and with a bare file-existence cache check the
    second run silently reuses the first run's 40.3 m features while stamping a 20.16 m feature
    contract. Deriving the token from the constants makes that impossible with zero discipline.

    Reads the CANONICAL source directly. Ghost used to own `_PENALTY_AREA_Y_MIN/_MAX/_X` and this
    token derived from those; ADR-050 section 6's closure deleted them, so the token follows the constants
    to `spadlconfig` rather than pointing at names that no longer exist. The band form is preserved
    so the token stays comparable in shape -- and its VALUE changes (13.8500 -> 13.8400), which is
    precisely the invalidation this function exists to produce for the box-constant re-fit.
    """
    import silly_kicks.spadl.config as _spc
    from silly_kicks.tracking import _geometry as _geo

    lo = _geo.GOAL_Y - _spc.penalty_area_half_width
    hi = _geo.GOAL_Y + _spc.penalty_area_half_width
    return f"v3-box{lo:.4f}-{hi:.4f}-{_spc.penalty_area_depth:.4f}"


#: One game's labels ride its shard under this prefix, so a game is ONE tidy frame -- the shape
#: `_driver.for_each` persists. Stripped again on read.
_LABEL_PREFIX = "_lab_"

#: Per-row arrays that ride the same shard as columns. Underscore-prefixed, and collision-checked
#: against the real feature names before any assembly: a feature named `_lab_gk_x` would be read
#: back as the LABEL and the model would be fitted on its own target. The 26 GHOST_GK_FEATURE_NAMES
#: carry no leading underscore today -- `_assert_no_column_collision` proves that per run rather
#: than trusting it, because the extractor is free to widen.
_SIDE_COLS = ("_game_id", "_provider", "_keeper")

#: The selection-bias diagnostic, carried as (sum, count) pairs through `for_each`'s `counters`
#: channel. See `_work` in `main` for why it cannot ride the frame, and why a list would be wrong.
_BIAS_COUNTERS = (
    "bias_depth_detected_sum",
    "bias_depth_detected_n",
    "bias_depth_undetected_sum",
    "bias_depth_undetected_n",
    "bias_b2k_detected_sum",
    "bias_b2k_detected_n",
    "bias_b2k_undetected_sum",
    "bias_b2k_undetected_n",
)


def _assert_no_column_collision(feature_columns) -> None:
    """Refuse to pack a feature whose name is indistinguishable from a label or a side column."""
    bad = sorted(c for c in feature_columns if c in _SIDE_COLS or str(c).startswith(_LABEL_PREFIX))
    if bad:
        raise ValueError(
            f"feature column(s) {bad} collide with the shard's reserved names "
            f"({_LABEL_PREFIX!r} prefix, {list(_SIDE_COLS)}). Unpacking would treat a FEATURE as a "
            f"label, and the model would be fitted on its own target. Rename the reserved columns."
        )


def _pack(feats: pd.DataFrame, labs: pd.DataFrame, *, game_id, provider: str, keepers: pd.Series) -> pd.DataFrame:
    """One game's features + labels + per-row side data as a single tidy frame."""
    _assert_no_column_collision(feats.columns)
    return pd.concat([feats, labs.add_prefix(_LABEL_PREFIX)], axis=1).assign(
        _game_id=game_id, _provider=provider, _keeper=keepers.to_numpy()
    )


def _unpack(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Invert :func:`_pack` over the combined table.

    The feature frame is reconstructed by REMOVAL, never by selecting ``GHOST_GK_FEATURE_NAMES``:
    the pre-migration code concatenated whatever ``prepare_ghost_gk_training_data`` returned, so a
    positive selection would silently change what the model is fitted on the day the extractor
    widens -- a different model from the same code, with nothing saying so.

    The three arrays are rebuilt via ``np.array(<list>)`` rather than ``Series.to_numpy()`` so their
    dtypes match the pre-migration ones exactly (``np.array(["a"])`` is a ``<U1`` array, not object);
    ``groups`` and ``provider_labels`` reach ``StratifiedGroupKFold`` and ``keepers`` reaches the
    keeper-domain code, none of which should see a dtype change from a storage decision.
    """
    lab_cols = [c for c in combined.columns if str(c).startswith(_LABEL_PREFIX)]
    if not lab_cols:
        raise ValueError(f"no {_LABEL_PREFIX!r} columns in the combined shards -- labels were lost in packing")
    features = combined.drop(columns=[*lab_cols, *_SIDE_COLS]).reset_index(drop=True)
    labels = combined[lab_cols].rename(columns=lambda c: c[len(_LABEL_PREFIX) :]).reset_index(drop=True)
    groups = np.array(combined["_game_id"].tolist())
    provider_labels = np.array(combined["_provider"].astype(str).tolist())
    keepers = np.array(combined["_keeper"].astype(str).tolist(), dtype=object)
    return features, labels, groups, provider_labels, keepers


def _mean_from_counters(total: float, n: float) -> float:
    """``sum / n``, or NaN on an empty population -- matching what ``np.mean([])`` reports."""
    return float(total) / float(n) if n else float("nan")


def validate_corpus_providers(providers: list[str]) -> None:
    """Fail on an unclassified provider BEFORE any loading or fitting.

    Without this the check fires inside ``keeper_detection_mask``, i.e. after the full per-game
    extraction -- the expensive part. Same rule, same single source
    (``_provider_visibility.validate_provider``); only the moment it fires changes, from "after an
    hour" to "immediately".
    """
    from silly_kicks.tracking._provider_visibility import validate_provider

    for provider in providers:
        validate_provider(provider)


def validate_corpus_visibility(provider_by_path: dict[Path, str]) -> None:
    """Fail BEFORE extraction on a detection-aware shard whose ``visibility`` was discarded.

    The build-time guard (``materialize_tc3_frames._guard_provider_frames``) is the primary defense,
    but a corpus can predate it or be hand-assembled, so the trainer re-checks at consume time -- the
    same class of pre-flight as :func:`validate_corpus_providers`, one stage earlier than the per-frame
    :func:`keeper_detection_mask` (which fires only after the expensive extraction).

    Reads parquet ``null_count`` METADATA (zero data pages) for EVERY detection-aware shard, so a
    MIXED corpus (one tail-kloppy shard among good ones) is caught, not just a systematic one. Raises
    the shared remedy message (rebuild via ``tracking.skillcorner``); also raises if a detection-aware
    shard dropped the ``visibility`` column entirely (M2, consume side). A shard whose statistics are
    unavailable falls back to reading the one column.
    """
    import pyarrow.parquet as pq

    from silly_kicks.tracking._provider_visibility import (
        _DETECTION_AWARE_PROVIDERS,
        _detection_discarded_message,
    )

    for path, provider in provider_by_path.items():
        if provider not in _DETECTION_AWARE_PROVIDERS:
            continue
        pf = pq.ParquetFile(path)
        if "visibility" not in pf.schema_arrow.names:
            raise ValueError(
                f"{path.name}: provider {provider!r} carries a detection flag, but the shard has NO "
                "`visibility` column -- the pipeline dropped it. Build these frames with "
                "tracking.skillcorner instead (spec 4.3)."
            )
        meta = pf.metadata
        num_rows = meta.num_rows
        if num_rows == 0:
            continue  # an empty shard is not a discarded-flag signal
        # Flat tc3 schema: arrow field order == parquet leaf-column order.
        col_idx = pf.schema_arrow.names.index("visibility")
        null_count = 0
        stats_ok = True
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(col_idx).statistics
            nc = getattr(stats, "null_count", None) if stats is not None else None
            if nc is None:
                stats_ok = False
                break
            null_count += nc
        if not stats_ok:
            # No usable metadata -> read the one column and decide honestly (rare path).
            vis = pf.read(columns=["visibility"]).column("visibility").to_pandas()
            if vis.isna().all():
                raise ValueError(f"{path.name}: {_detection_discarded_message(provider)}")
            continue
        if null_count == num_rows:
            raise ValueError(f"{path.name}: {_detection_discarded_message(provider)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ghost-GK model")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of tracking parquets")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Where to save model artifact")
    parser.add_argument("--actions-dir", type=Path, default=None, help="Optional: directory of SPADL actions parquets")
    parser.add_argument(
        "--home-teams",
        type=Path,
        default=None,
        help="JSON file: {game_id: home_team_id, ...} (auto-read from meta.json if omitted)",
    )
    parser.add_argument("--subsample-fps", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    # PR-S81: carrier params (single source -> prepare AND fit; recorded in metadata, R3).
    parser.add_argument(
        "--carrier-beta", type=float, default=None, help="Carrier velocity weight (default: library default)"
    )
    parser.add_argument("--carrier-gamma", type=float, default=None, help="Carrier hysteresis (default: library)")
    parser.add_argument("--carrier-tolerance", type=float, default=None, help="Carrier radius m (default: library)")
    parser.add_argument(
        "--variant",
        choices=["default", "full"],
        default="full",
        help="Which variant this run produces (recorded in metrics/metadata)",
    )
    parser.add_argument(
        "--feature-set",
        choices=["faithful", "position_only"],
        default="faithful",
        help="'faithful' (velocity-bearing, 26 feats) or 'position_only' (5 velocity feats dropped, 21) "
        "for a model that scores on lone velocity-less SB360 freeze frames.",
    )
    parser.add_argument(
        "--subsample-cap",
        type=int,
        default=None,
        help="Cap total training samples (default None=all; ~36000 for the bundled 'default')",
    )
    parser.add_argument(
        "--training-platform", type=str, default=None, help="Recorded in metadata (e.g. 'dgx-spark-aarch64')"
    )
    parser.add_argument(
        "--skip-permutation-importance",
        action="store_true",
        help="Skip the (slow, metrics-only) permutation importance pass. The artifact + CV "
        "metrics + acceptance criteria are unaffected; only the printed feature-importance "
        "ranking is omitted. At full scale (887k) the pass dominates wall-clock.",
    )
    parser.add_argument(
        "--perm-importance-sample",
        type=int,
        default=150000,
        help="Cap the permutation-importance EVAL rows to a seeded subsample (importance is a "
        "statistical estimate; the ranking is stable on a representative sample). The full "
        "corpus (887k) is memory-bandwidth-bound and intractable even with n_jobs. Default "
        "150000 (~4x the default variant's whole training set). Set 0 for all rows.",
    )
    # spec 4.3: keeper-grouped CV over a common keeper domain. Default OFF so the shipped
    # artifact's headline metrics stay match-grouped and comparable with 4.14.0.
    parser.add_argument(
        "--keeper-grouped",
        action="store_true",
        default=False,
        help="CV by KEEPER over the common domain (spec 4.3) instead of StratifiedGroupKFold by "
        "game_id. The two corpora share keepers (e.g. Courtois), so match-grouped CV leaks a "
        "keeper across train/test folds. Requires --expansion-keepers. Default OFF.",
    )
    parser.add_argument(
        "--expansion-keepers",
        type=Path,
        default=None,
        help="An .npy of keeper ids present in the 98-match expansion cohort. A --keeper-grouped "
        "baseline run consumes it to build the common evaluation domain (baseline keepers MINUS "
        "anyone in the 98). Mandatory when --keeper-grouped is set.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Train from a modified working tree. The run still records run_tree_dirty=true in "
        "metrics.json -- the hatch permits a dev run, it never launders the fact.",
    )
    return parser.parse_args()


def keeper_detection_mask_or_none(visibility: pd.Series) -> pd.Series | None:
    """Diagnostic-only detected mask for the selection-bias report (spec 4.3 rev 5).

    Unlike :func:`keeper_detection_mask` (the TRAINING filter, which fail-closes on an all-null
    ``visibility``), this returns ``None`` so the diagnostic simply SKIPS a match whose flag is
    entirely null -- the RAISE belongs to the training filter, not the report. Otherwise it mirrors
    the training mask exactly (``fillna(False).astype(bool)``) so the detected/undetected split
    matches what the filter would keep.
    """
    if visibility.isna().all():
        return None
    return visibility.fillna(False).astype(bool)


def resolve_training_platform(explicit: str | None, prov: dict) -> str:
    """The platform to stamp: an explicit label if given, else the detected one.

    Defaulting to `None` was the defect. `_xshot_occurrence` and `_xcross_attempt` both record
    `platform.platform()` unconditionally, so ghost was the only trained artifact whose machine
    identity depended on the operator remembering a flag -- and the flag was in fact forgotten,
    leaving artifacts that could not say where they ran. Detection is the DEFAULT and the flag is now
    only an override for a human-meaningful label ("dgx-spark-aarch64" reads better than
    "Linux-6.11.0-aarch64-with-glibc2.39" in a model card).

    Never returns None: an absent platform and an unknown platform are the same useless value
    downstream, and `platform.platform()` always answers.
    """
    return explicit or prov["platform"]


def main() -> None:
    # Force unbuffered stdout so background tasks show progress immediately
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    args = parse_args()

    # Refuse to fit a SHIPPABLE ghost artifact under a scikit-learn outside the supported fit range
    # (VSSOT-IMPL-03). HistGradientBoosting produces different trees across sklearn versions (measured:
    # same corpus/commit/pandas under 1.7.2 vs 1.9.0 -> different weights, different sha256). The
    # `[train]` extra pins `scikit-learn>=1.9,<2` but marker-gates it to python>=3.11 (sklearn 1.9
    # dropped 3.10), so `pip install .[train]` on py3.10 silently resolves an older sklearn. This makes
    # that footgun LOUD at the fit-time entry point rather than shipping mismatched weights. Kept here,
    # not in GhostGkModel.fit(): the non-slow library unit tests fit toy models on the leg's own
    # sklearn (1.7.2 on py3.10) and a library-level raise would redden those legs; this trainer's own
    # smokes are all @slow (primary 3.12 leg, sklearn 1.9.0).
    from scripts._train_guard import require_training_sklearn

    print(f"scikit-learn {require_training_sklearn()} (within the supported fit range for bundled weights)")

    # FIRST, before any parquet is read or any model is fitted. This trainer stamps
    # `training_commit` into the SHIPPED artifact's metadata.json, and it used to read that SHA
    # from a bare `git rev-parse HEAD` -- which returns the same commit whether or not the tree is
    # modified. A bundled weights file therefore carried a verifiable-looking claim about code that
    # may never have existed at that commit, which is strictly worse than carrying none.
    #
    # This is the one trainer with that specific defect: the others stamp no commit at all, so they
    # make no false claim. Refusing by default is what makes `training_commit` true by
    # construction, so the artifact's own field needs no new schema to be trustworthy;
    # `--allow-dirty` still records the fact in metrics.json beside it.
    #
    # `run_prov`, not `prov`: `main` already binds `prov` twice as a per-provider loop variable
    # (the CV per-provider MAE, and the metrics aggregation), and the first CLI run after this was
    # written died on `TypeError: string indices must be integers` -- the loop had rebound it to a
    # provider name by the time the metrics dict was built.
    from scripts._provenance import git_provenance, require_clean_tree

    run_prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # PR-S81: resolve ONE carrier cp from CLI (default = library) and pass the SAME dict
    # to both prepare (compute) and fit (record) so metadata records exactly what was used.
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    cp = dict(DEFAULT_CARRIER_PARAMS)
    if args.carrier_tolerance is not None:
        cp["tolerance_m"] = args.carrier_tolerance
    if args.carrier_beta is not None:
        cp["beta"] = args.carrier_beta
    if args.carrier_gamma is not None:
        cp["gamma"] = args.carrier_gamma
    print(f"Carrier params (single source, recorded + used): {cp}")

    # `None` rather than the helper's "unknown" sentinel, so metadata.json keeps the `str | None`
    # shape that `GhostGkModel.load` and the published artifacts already carry.
    training_commit = None if run_prov["commit"] == "unknown" else run_prov["commit"]
    training_platform = resolve_training_platform(args.training_platform, run_prov)
    print(f"training_commit={training_commit} (tree_dirty={run_prov['dirty']}), training_platform={training_platform}")

    print(f"Config: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"Data: {args.data_dir}, subsample_fps={args.subsample_fps}")
    _cv_regime = (
        "GroupKFold by keeper (common domain)" if args.keeper_grouped else "StratifiedGroupKFold (match+provider)"
    )
    print(f"CV: {args.cv_folds}-fold {_cv_regime}")
    print(f"Output: {args.output_dir}")

    # --- 1. Discover tracking parquets ---
    # Support both tc3 cache ({provider}/{game_id}/frames.parquet) and flat (*.parquet) layouts.
    # Prefer tc3 layout (more specific) to avoid picking up stale non-tracking parquets in the root.
    parquets = sorted(args.data_dir.glob("**/frames.parquet"))
    if not parquets:
        parquets = sorted(args.data_dir.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: No .parquet files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Fail fast on an unclassified provider, BEFORE any extraction or fitting. The membership rule
    # lives in _provider_visibility.validate_provider (single source); only the moment it fires
    # changes. It previously fired inside keeper_detection_mask -- i.e. after the full per-game
    # extraction, the expensive part -- so a typo'd provider cost an hour before anything said so.
    #
    # Reads ONE column from ONE row group per file: seconds, against an extraction measured in tens
    # of minutes. A file with no `source_provider` column contributes nothing here; that case is
    # already handled downstream (prov = "unknown"), which validate_provider will reject at the
    # point it becomes real rather than being guessed at now.
    import pyarrow.parquet as pq

    discovered: set[str] = set()
    provider_by_path: dict[Path, str] = {}
    for p in parquets:
        pf = pq.ParquetFile(p)
        if "source_provider" not in pf.schema_arrow.names or pf.num_row_groups == 0:
            continue
        col = pf.read_row_group(0, columns=["source_provider"])["source_provider"]
        if len(col):
            prov = str(col[0].as_py())
            discovered.add(prov)
            provider_by_path[p] = prov
    if discovered:
        validate_corpus_providers(sorted(discovered))
        print(f"Providers validated up front: {sorted(discovered)}")
        # And, for every detection-aware shard, that its visibility flag actually survived -- BEFORE
        # extraction, so a kloppy-discarded corpus fails HERE, not an hour in via keeper_detection_mask.
        # Reads null_count metadata only, so it also catches a MIXED corpus (one tail-kloppy shard).
        validate_corpus_visibility(provider_by_path)
    else:
        print("No source_provider column present; provider validation happens during extraction.")

    # Schema validation on first file only (all files share the same pipeline)
    required = {
        "game_id",
        "period_id",
        "frame_id",
        "time_seconds",
        "player_id",
        "team_id",
        "is_ball",
        "is_goalkeeper",
        "x",
        "y",
    }
    import pyarrow.parquet as pq

    probe_cols = set(pq.read_schema(parquets[0]).names)
    missing = required - probe_cols
    if missing:
        print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    if "vx" not in probe_cols or "vy" not in probe_cols:
        print("ERROR: vx/vy columns missing. Run smooth_frames + derive_velocities first.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parquets)} parquet files in {args.data_dir}")

    # --- 2. Load actions (optional, small -- OK to hold in memory) ---
    actions_by_game: dict[str, pd.DataFrame] = {}
    if args.actions_dir is not None:
        action_parquets = sorted(args.actions_dir.glob("*.parquet"))
        if action_parquets:
            actions = pd.concat(
                [pd.read_parquet(p) for p in action_parquets],
                ignore_index=True,
            )
            actions_by_game = dict(list(actions.groupby("game_id")))  # type: ignore[reportAssignmentType]
            del actions  # Release concatenated copy
            print(f"Loaded actions for {len(actions_by_game)} games")

    # --- 3. Load home team mapping ---
    # Auto-discover from meta.json siblings when --home-teams is not provided.
    home_team_map: dict[str, str] = {}
    if args.home_teams is not None:
        with open(args.home_teams) as f:
            home_team_map = json.load(f)
    else:
        for pq_path in parquets:
            meta_path = pq_path.parent / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                home_val = str(meta["home_team_id"])
                # Key by directory name (works for providers where dir name == game_id)
                home_team_map[pq_path.parent.name] = home_val
                # Also key by actual game_id from the parquet (handles SkillCorner
                # where dir name is match_id but game_id column is a kloppy hash)
                actual_ids = pq.read_table(pq_path, columns=["game_id"]).column("game_id").unique().to_pylist()
                for gid in actual_ids:
                    home_team_map[str(gid)] = home_val
    if not home_team_map:
        print(
            "ERROR: No home team mapping. Provide --home-teams or use tc3 cache layout with meta.json.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Home team mapping: {len(home_team_map)} games")

    # --- 4. Per-game feature extraction (with disk cache) ---
    # Cache extracted features to avoid re-reading 78 x 4M-row parquets on re-runs.
    cache_dir = args.output_dir / "ghost_gk_v1" / "_feature_cache"
    cache_feats = cache_dir / "features.parquet"
    cache_labels = cache_dir / "labels.parquet"
    cache_groups = cache_dir / "groups.npy"
    cache_provs = cache_dir / "providers.npy"
    cache_keepers = cache_dir / "keepers.npy"
    cache_token_path = cache_dir / "cache_token.txt"

    # Selection-bias diagnostic (spec 4.3 rev 5), carried as (sum, count) pairs. Defined at a scope
    # visible to BOTH cache branches: the fresh-extract path fills it (it holds the raw
    # visibility); a whole-corpus cache hit leaves it empty (raw visibility is not cached), so the
    # diagnostic block is simply omitted. Two bias axes: keeper DEPTH (gk_x_gr) and BALL-to-keeper
    # distance (both goal-relative).
    bias: dict[str, float] = {}
    bias_unrecorded = 0

    # WHAT THE RECORDED TOKEN COVERS -- widened here, and the widening is a bug fix. It used to be
    # `cache_token()` alone (the penalty-area geometry), so a re-run with a different
    # `--subsample-fps`, different `--carrier-*`, or `--actions-dir` newly supplied silently reused
    # the previous run's feature matrix while `metadata.json` recorded the NEW carrier params. That
    # is the recorded==used invariant PR-S81 exists to hold, broken by the cache underneath it.
    # The token is now the shard GENERATION digest, which folds in `cache_token()` plus every other
    # declared input, so the two layers can no longer disagree about what the features are.
    #
    # `token_inputs` is built once and used twice (here and by `for_each` below) so the two cannot
    # drift into naming different generations for the same run.
    from scripts._driver import generation_dir

    extraction_inputs: dict[str, object] = {
        "extractor": "prepare_ghost_gk_training_data",
        # feature_set changes the shard's feature columns (26 vs 21) -> MUST key the generation (4.77.1).
        "feature_set": args.feature_set,
        "geometry": cache_token(),
        "subsample_fps": args.subsample_fps,
        "carrier_params": dict(cp),
        # Presence only. The score/phase context depends on the actions THEMSELVES, which no digest
        # this driver can afford would capture; `generation_dir` states that ceiling -- a token
        # closes silent omission, not mis-declaration.
        "with_actions": bool(actions_by_game),
        "detected_targets_only": True,
    }
    shard_root = cache_dir / "shards"
    generation = generation_dir(shard_root, token_inputs=extraction_inputs)

    # A cache is trusted only if every array is present AND its recorded token matches this run's
    # generation. A missing or differing token is a MISS, so a cache extracted under different
    # geometry constants -- or different extraction parameters -- can never be silently reused
    # (that is exactly the re-fit-cycle failure it guards).
    _recorded_token = cache_token_path.read_text(encoding="utf-8").strip() if cache_token_path.exists() else None
    if (
        cache_feats.exists()
        and cache_labels.exists()
        and cache_groups.exists()
        and cache_provs.exists()
        and cache_keepers.exists()
        and _recorded_token == generation.name
    ):
        print(f"\nLoading cached features from {cache_dir}")
        t0 = time.time()
        features = pd.read_parquet(cache_feats)
        labels = pd.read_parquet(cache_labels)
        groups = np.load(cache_groups, allow_pickle=True)
        provider_labels = np.load(cache_provs, allow_pickle=True)
        keepers = np.load(cache_keepers, allow_pickle=True)
        elapsed = time.time() - t0
        print(f"Loaded {len(features)} samples in {elapsed:.1f}s (cached)")
    else:
        # Following lakehouse TC-3 pattern: load frames per-file, extract features,
        # then delete frames immediately.  Only the extracted feature matrix (small)
        # stays in memory -- raw frames (large) are never held simultaneously.
        from scripts._driver import for_each, shard_path
        from silly_kicks.tracking import prepare_ghost_gk_training_data
        from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, keeper_detection_mask

        # Fail before a single 4M-row parquet is read, not on the first packed game.
        _assert_no_column_collision(GHOST_GK_FEATURE_NAMES)

        n_skipped = 0
        t0 = time.time()

        def _items():
            """Yield one ``(provider, game_id, frames, home)`` item at a time.

            A GENERATOR, not a materialised list: `for_each` streams, so at most ONE file's frames
            are alive -- today's memory profile exactly. Enumerating ``(file, game)`` pairs up
            front was rejected because a game id is only knowable from inside the file, so `work`
            would have to read the parquet itself and a multi-game file would be read once PER GAME.

            The consequence, stated rather than hidden: a RESUMED extraction still re-READS every
            parquet, because `for_each` resumes ``work``, not the production of its items. What it
            skips is the extraction, which is the expensive half here (the provider fail-fast above
            exists precisely because the per-game extraction is what costs an hour). The
            whole-corpus feature cache above remains the fast path for a run that already finished.
            """
            nonlocal n_skipped
            for pq_idx, pq_path in enumerate(parquets):
                file_frames = pd.read_parquet(pq_path)
                game_ids_in_file = sorted(file_frames["game_id"].unique())
                print(
                    f"  [{pq_idx + 1}/{len(parquets)}] {pq_path.name}:"
                    f" {len(game_ids_in_file)} game(s), {len(file_frames)} rows"
                )
                # provider is a per-file constant, so it is read once here rather than once per
                # game; the pre-migration code recomputed the same file-level value inside the game
                # loop. It is needed BEFORE prepare so the detected-only filter and the
                # selection-bias diagnostic can key on it.
                prov = (
                    str(file_frames["source_provider"].iloc[0])
                    if "source_provider" in file_frames.columns
                    else "unknown"
                )
                for game_id in game_ids_in_file:
                    home = home_team_map.get(str(game_id))
                    if home is None:
                        print(f"    SKIP game {game_id}: no home_team_id in mapping")
                        n_skipped += 1
                        continue
                    yield prov, game_id, file_frames[file_frames["game_id"] == game_id], home
                del file_frames  # Release entire file's frames before loading next

        # Per-item bias sums the tidy frame CANNOT carry: they are measured BEFORE the
        # detected-only filter, so they have a different row count than the shard's rows. They ride
        # the `counters` channel instead, as (sum, count) pairs -- summable, so `for_each` replays
        # them from the counters sidecar and a PARTIALLY resumed pass still reports means over the
        # whole corpus. A list accumulator here would have been silently wrong on resume: the
        # reported means would describe only the games this pass happened to redo, while looking
        # exactly like a corpus figure. (The whole-corpus cache branch is different and safe: it is
        # all-or-nothing, so it omits the block entirely rather than narrowing it.)
        _bias: dict[str, float] = {}

        def _work(item):
            prov, game_id, game_frames, home = item
            game_actions = actions_by_game.get(game_id) if actions_by_game else None

            # return_meta=True -> the 3-tuple overload; pyright narrows on the Literal flag.
            feats, labs, meta = prepare_ghost_gk_training_data(
                game_frames,
                home_team_id=home,
                actions=game_actions,
                subsample_fps=args.subsample_fps,
                carrier_params=cp,
                feature_set=args.feature_set,
                return_meta=True,
            )
            # Zeroed, not merely cleared: `counters` runs for every ATTEMPTED item, and a
            # non-skillcorner game must contribute an explicit 0 rather than a missing key.
            _bias.clear()
            _bias.update(dict.fromkeys(_BIAS_COUNTERS, 0.0))

            if not len(feats):
                return None  # nothing extracted (meta empty -> stop before the masks)

            # --- Selection-bias diagnostic (spec 4.3 rev 5): compute BEFORE the detected-only
            # filter, because the filter drops the undetected frames the bias compares against.
            # keeper DEPTH = gk_x_gr (goal-relative x, already in meta); detected vs undetected
            # by gk_visibility. This is a REPORTED limitation, not a gate.
            if prov == "skillcorner" and len(meta):
                _vis = keeper_detection_mask_or_none(meta["gk_visibility"])
                if _vis is not None:
                    _depth = meta["gk_x_gr"].to_numpy(dtype=float)
                    # Ball-to-keeper distance (the OTHER bias axis). feats.ball_x/ball_y and
                    # meta.gk_x_gr/gk_y_gr share the SAME goal-relative frame (identical to_gr_x
                    # flip; y unflipped) and are row-aligned pre-filter, so this is a single
                    # consistent coordinate system. NaN where the ball is absent.
                    _b2k = np.hypot(
                        feats["ball_x"].to_numpy() - meta["gk_x_gr"].to_numpy(),
                        feats["ball_y"].to_numpy() - meta["gk_y_gr"].to_numpy(),
                    )
                    _vm = _vis.to_numpy()
                    _bias.update(
                        {
                            # PLAIN sum + row count for depth: the report takes a mean (not a
                            # nanmean) of these, so a NaN must still poison it rather than vanish.
                            "bias_depth_detected_sum": float(np.sum(_depth[_vm])),
                            "bias_depth_detected_n": int(_vm.sum()),
                            "bias_depth_undetected_sum": float(np.sum(_depth[~_vm])),
                            "bias_depth_undetected_n": int((~_vm).sum()),
                            # nansum over a NON-NaN count reproduces nanmean exactly.
                            "bias_b2k_detected_sum": float(np.nansum(_b2k[_vm])),
                            "bias_b2k_detected_n": int((~np.isnan(_b2k[_vm])).sum()),
                            "bias_b2k_undetected_sum": float(np.nansum(_b2k[~_vm])),
                            "bias_b2k_undetected_n": int((~np.isnan(_b2k[~_vm])).sum()),
                        }
                    )

            # --- Detected-keeper targets ONLY (spec 4.3). RAISES if a detection-aware
            # provider's flag was discarded upstream (fail-closed on the ambiguous null).
            keep = keeper_detection_mask(meta["gk_visibility"], provider=prov)
            feats = feats[keep].reset_index(drop=True)
            labs = labs[keep].reset_index(drop=True)
            meta = meta[keep].reset_index(drop=True)
            if not len(feats):
                print(f"  SKIP {prov}/{game_id}: no detected-keeper frames")
                return None  # an EMPTY shard: "ran, produced no usable row", so it is not redone
            return _pack(feats, labs, game_id=game_id, provider=prov, keepers=meta["gk_player_id"].astype(str))

        res = for_each(
            _items(),
            # Provider first, mirroring every other migrated driver: providers in this corpus
            # demonstrably share game ids. The parquet FILENAME is deliberately not a component --
            # in the tc3 layout every file is named `frames.parquet`, so it distinguishes nothing.
            #
            # `home` is the THIRD component, and it belongs in the KEY rather than in
            # `token_inputs`, because it is a PER-ITEM input while the token is per-PASS.
            # `home_team_id` drives the goal-relative flip of every feature and label in the shard,
            # so a corrected mapping must invalidate the games it corrects -- but declaring the whole
            # `{game_id: home}` map would invalidate EVERY shard the moment one match is added to
            # `--data-dir`, which is exactly the over-invalidation the selector rule exists to avoid.
            # Per-item input -> per-item identity. Still injective: one home per game per run.
            #
            # The asymmetry this closes: ADDING a previously-missing mapping was always safe (the
            # game was skipped, so no shard existed), but CHANGING an existing one was silent -- the
            # shard was skipped, the correction never reached the features, and because the
            # whole-corpus cache token is this generation's digest, the cache accepted it too. The
            # model would have trained on wrong-handed data with no signal anywhere.
            key=lambda item: (str(item[0]), str(item[1]), str(item[3])),
            work=_work,
            counters=lambda _item, _frame: dict(_bias),
            shard_root=shard_root,
            token_inputs=extraction_inputs,
            tag="ghost_gk_features",
            label="game",
        )
        if res.failures:
            raise RuntimeError(
                f"{len(res.failures)} game(s) failed during extraction: {res.failures}. "
                f"Re-run to retry only those -- the games that succeeded are already sharded."
            )

        # Combined from THIS PASS'S keys, not `_driver.reconcile`: this driver has no partition
        # surface, so a whole-generation read would fold in games from a wider earlier `--data-dir`.
        parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys) if len(f)]
        if not parts:
            print("ERROR: No training samples extracted.", file=sys.stderr)
            sys.exit(1)

        combined = pd.concat(parts, ignore_index=True)
        del parts
        features, labels, groups, provider_labels, keepers = _unpack(combined)
        del combined
        bias = dict(res.counters)
        bias_unrecorded = res.counters_unrecorded
        elapsed = time.time() - t0
        print(
            f"\nExtracted {len(features)} samples from {len(set(groups.tolist()))} games"
            f" ({n_skipped} skipped, {res.skipped} already sharded) in {elapsed:.1f}s"
        )

        # Save cache for subsequent runs
        cache_dir.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_feats)
        labels.to_parquet(cache_labels)
        np.save(cache_groups, groups)
        np.save(cache_provs, provider_labels)
        np.save(cache_keepers, keepers)
        # Written LAST, deliberately: the token is what makes the cache trustworthy, so it must
        # not exist until every array beside it does. A crash mid-write then leaves a tokenless
        # directory, which the hit predicate treats as a MISS.
        cache_token_path.write_text(generation.name, encoding="utf-8")
        print(f"Cached features to {cache_dir} (generation {generation.name})")

    # PR-S81: variant axis = sample count -> wheel size. Cap AFTER extraction so the
    # bundled "default" stays small while "full" keeps all in-domain samples.
    if args.subsample_cap is not None and len(features) > args.subsample_cap:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(features), size=args.subsample_cap, replace=False)
        keep.sort()
        features = features.iloc[keep].reset_index(drop=True)
        labels = labels.iloc[keep].reset_index(drop=True)
        groups = groups[keep]
        provider_labels = provider_labels[keep]
        keepers = keepers[keep]
        print(f"Subsampled to {len(features)} samples (variant={args.variant}, cap={args.subsample_cap})")

    # --- 5. CV ---
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    # Two CV regimes (spec 4.3):
    #  - default: StratifiedGroupKFold by game_id (match-grouped; headline metrics comparable
    #    with 4.14.0).
    #  - --keeper-grouped: GroupKFold by KEEPER over the common domain (baseline keepers MINUS the
    #    98-cohort keepers), because the corpora share keepers so a match fold would leak one.
    dreport = None
    if args.keeper_grouped:
        from _ghost_domain import common_keeper_domain, keeper_folds

        if args.expansion_keepers is None:
            print(
                "ERROR: --keeper-grouped requires --expansion-keepers (the keeper ids present in "
                "the 98-match cohort). The common-domain exclusion is mandatory for a "
                "keeper-grouped run -- see spec 4.3.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        expansion_keepers = set(np.load(args.expansion_keepers, allow_pickle=True).astype(str).tolist())
        domain, dreport = common_keeper_domain(keepers, expansion_keepers=expansion_keepers, n_splits=args.cv_folds)
        print(
            f"Keeper-grouped CV: {dreport.n_domain_keepers} domain keepers "
            f"({dreport.n_excluded_keepers} excluded as expansion-cohort), "
            f"underpowered={dreport.underpowered}"
        )
        # No test-fold keeper may be in the 98 (the whole point of the common domain).
        assert not (set(keepers[domain].tolist()) & expansion_keepers), (  # noqa: S101
            "a domain keeper is in the expansion cohort -- the common-domain exclusion failed"
        )
        folds = list(keeper_folds(keepers, domain, n_splits=args.cv_folds))
    else:
        from sklearn.model_selection import StratifiedGroupKFold

        cv = StratifiedGroupKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=42,
        )
        folds = list(cv.split(features, provider_labels, groups))

    fold_metrics: list[dict] = []

    cv_t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")
        print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model = GhostGkModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            verbose=1,
            feature_set=args.feature_set,
        )
        fit_t0 = time.time()
        model.fit(X_train, y_train, carrier_params=cp)
        fit_elapsed = time.time() - fit_t0
        print(f"  Fit: {fit_elapsed:.1f}s")

        pred_t0 = time.time()
        preds = model.predict_mean(X_test)  # shape (n, 2) -- exact boosted HGBR mean (Option A)
        pred_elapsed = time.time() - pred_t0
        print(f"  Predict (boosted mean): {pred_elapsed:.1f}s")

        mae_x = float(np.mean(np.abs(preds[:, 0] - y_test["gk_x"].values)))
        mae_y = float(np.mean(np.abs(preds[:, 1] - y_test["gk_y"].values)))
        mae_euclid = float(
            np.mean(np.sqrt((preds[:, 0] - y_test["gk_x"].values) ** 2 + (preds[:, 1] - y_test["gk_y"].values) ** 2))
        )

        # Per-provider MAE
        test_provs = provider_labels[test_idx]
        per_prov: dict[str, float] = {}
        for prov in np.unique(test_provs):
            mask = test_provs == prov
            per_prov[prov] = float(
                np.mean(
                    np.sqrt(
                        (preds[mask, 0] - y_test["gk_x"].values[mask]) ** 2
                        + (preds[mask, 1] - y_test["gk_y"].values[mask]) ** 2
                    )
                )
            )

        fold_wall = time.time() - cv_t0
        avg_per_fold = fold_wall / (fold + 1)
        remaining = avg_per_fold * (args.cv_folds - fold - 1)
        print(f"  MAE x={mae_x:.3f}m  y={mae_y:.3f}m  euclid={mae_euclid:.3f}m")
        print(f"  Per-provider: {per_prov}")
        print(f"  CV elapsed: {fold_wall:.0f}s, ETA remaining: {remaining:.0f}s")
        fold_metrics.append(
            {
                "mae_x": mae_x,
                "mae_y": mae_y,
                "mae_euclidean": mae_euclid,
                "per_provider": per_prov,
            }
        )

    # Aggregate CV
    mae_x_vals = [m["mae_x"] for m in fold_metrics]
    mae_y_vals = [m["mae_y"] for m in fold_metrics]
    mae_e_vals = [m["mae_euclidean"] for m in fold_metrics]
    print("\n=== CV Summary ===")
    print(f"MAE x: {np.mean(mae_x_vals):.3f} +/- {np.std(mae_x_vals):.3f}")
    print(f"MAE y: {np.mean(mae_y_vals):.3f} +/- {np.std(mae_y_vals):.3f}")
    print(f"MAE euclid: {np.mean(mae_e_vals):.3f} +/- {np.std(mae_e_vals):.3f}")

    # --- 6. Feature importance ---
    from sklearn.inspection import permutation_importance

    print("\n--- Feature importance (full model, x-coordinate only) ---")
    print("NOTE: Importance measured for gk_x predictions only.")
    print("Features primarily influencing gk_y may show artificially low importance.")
    print("Training final model on all data...")
    final_model = GhostGkModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        verbose=1,
        feature_set=args.feature_set,
    )
    final_t0 = time.time()
    final_model.fit(features, labels, carrier_params=cp)
    print(f"Final model fit: {time.time() - final_t0:.1f}s")

    # --- BLOCKING parity-on-fresh-fit gate (Option A, ADR-016) ---
    # predict_mean (pickle-free numpy reconstruction) must equal the live sklearn
    # regressors' .predict() to <=1e-6 on the ACTUAL fitted model. The regressors are
    # transient (not serialized), so this can only run on the fresh fit, here. Abort
    # before the expensive permutation importance + publish if parity fails.
    _par_n = min(20000, len(features))
    _par_idx = np.random.default_rng(0).choice(len(features), _par_n, replace=False)
    _par_Xv = features.iloc[_par_idx][features.columns].values
    # The sklearn regressors + boosted tree-node arrays are populated by the fresh fit above
    # (this gate runs only on the just-fitted model); narrow off Optional for the type checker.
    _sk_reg_x, _sk_reg_y = final_model._sk_reg_x, final_model._sk_reg_y
    _tree_nodes, _tree_nodes_y = final_model._tree_nodes, final_model._tree_nodes_y
    assert _sk_reg_x is not None and _sk_reg_y is not None  # noqa: S101
    assert _tree_nodes is not None and _tree_nodes_y is not None  # noqa: S101
    _par_sk = np.column_stack([_sk_reg_x.predict(_par_Xv), _sk_reg_y.predict(_par_Xv)])
    _par_err = float(np.abs(final_model.predict_mean(features.iloc[_par_idx]) - _par_sk).max())
    _par_ncat = sum(int(t["is_categorical"].sum()) for t in _tree_nodes) + sum(
        int(t["is_categorical"].sum()) for t in _tree_nodes_y
    )
    print(f"PARITY GATE: max|predict_mean - sklearn| = {_par_err:.2e} over {_par_n} rows; n_cat = {_par_ncat}")
    if _par_err > 1e-6 or _par_ncat != 0:
        msg = f"BLOCKING parity gate FAILED (err={_par_err:.2e}, ncat={_par_ncat}); refusing to publish."
        raise RuntimeError(msg)
    print("PARITY GATE: PASS (boosted reconstruction is exact; safe to publish)")

    # Permutation importance is metrics-only (printed, not saved to metrics.json) and
    # dominates wall-clock at full scale (887k x 5 repeats x 26 features). Skippable.
    if args.skip_permutation_importance:
        print("Skipping permutation importance (--skip-permutation-importance).")
    else:
        # Use a simple sklearn wrapper for permutation importance
        from sklearn.base import BaseEstimator, RegressorMixin

        class _SklearnWrapper(BaseEstimator, RegressorMixin):  # type: ignore[misc]
            def __init__(self, m: GhostGkModel | None = None) -> None:
                self.m = m

            def fit(self, X: np.ndarray, y: np.ndarray) -> _SklearnWrapper:
                return self  # already fitted

            def predict(self, X: np.ndarray) -> np.ndarray:
                assert self.m is not None  # noqa: S101
                return self.m.predict_mean(pd.DataFrame(X, columns=features.columns))[:, 0]

        # Subsample the EVAL rows (importance is a statistical estimate; the ranking is stable
        # on a representative sample). The full 887k corpus is memory-bandwidth-bound -- each
        # boosted predict_mean scans the full leaf arrays, so 20 workers contend for bandwidth
        # and n_jobs gives no speedup. A 150k subsample is ~8x less traffic so n_jobs parallelizes.
        _pi_cap = args.perm_importance_sample
        if _pi_cap and _pi_cap < len(features):
            _pi_idx = np.random.default_rng(42).choice(len(features), _pi_cap, replace=False)
            _pi_X = features.values[_pi_idx]
            _pi_y = labels["gk_x"].values[_pi_idx]
        else:
            _pi_X, _pi_y = features.values, labels["gk_x"].values
        print(f"Running permutation importance (5 repeats, n_jobs=-1) on {len(_pi_X)} eval rows...")
        pi_t0 = time.time()
        # n_jobs=-1: parallelize the per-(feature, repeat) scorer calls; pair with
        # OMP_NUM_THREADS=1 in the launch env so each loky worker stays single-threaded.
        pi = permutation_importance(
            _SklearnWrapper(m=final_model),
            _pi_X,
            _pi_y,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        print(f"Permutation importance: {time.time() - pi_t0:.1f}s")
        importances = sorted(
            zip(features.columns, pi.importances_mean, strict=True),  # type: ignore[reportAttributeAccessIssue]
            key=lambda x: -x[1],
        )
        print("Top 10 features:")
        for name, imp in importances[:10]:
            print(f"  {name}: {imp:.4f}")

    # --- 7. Save final model ---
    final_model.training_commit = training_commit
    final_model.training_platform = training_platform
    # Aggregate corpus provenance (providers + counts ONLY; spec 2026-07-20 S6). Providers come
    # from the per-file source_provider column (already collected into provider_labels);
    # n_games from the training groups; n_rows from the retained sample count. NEVER a per-match
    # id list, NEVER a public/restricted split (owner decision). No match->registered
    # classification join is performed, so there is no join to guard.
    final_model.corpus_provenance = {
        "providers": sorted({str(p) for p in provider_labels.tolist()}),
        "n_games": len(set(groups.tolist())),
        "n_rows": len(features),
    }
    artifact_dir = args.output_dir / "ghost_gk_v1"
    final_model.save(artifact_dir)
    print(f"\nModel saved to {artifact_dir}")

    # Round-trip verify. Post-2026-07-20 the artifact is parameters-only: the per-sample arrays
    # are NOT persisted, so a loaded model's _training_* are None by design. Verify the SERIALIZED
    # PARAMETERS (tree ensembles + baselines) round-trip exactly instead.
    loaded = GhostGkModel.load(artifact_dir)
    for attr in ("_tree_nodes", "_tree_nodes_y"):
        orig = getattr(final_model, attr)
        back = getattr(loaded, attr)
        for i, (a, b) in enumerate(zip(orig, back, strict=True)):
            np.testing.assert_array_equal(a, b, err_msg=f"{attr}[{i}]")
    assert loaded._baseline_x == final_model._baseline_x, "baseline_x drift"  # noqa: S101
    assert loaded._baseline_y == final_model._baseline_y, "baseline_y drift"  # noqa: S101
    assert loaded.carrier_params == cp, f"carrier_params drift: {loaded.carrier_params} != {cp}"  # noqa: S101
    print(f"Round-trip verification: PASS (R3 carrier_params={loaded.carrier_params})")

    # --- 8. Metrics JSON ---
    # Aggregate per-provider MAE across folds
    all_provs_set: set[str] = set()
    for m in fold_metrics:
        all_provs_set.update(m["per_provider"].keys())
    per_prov_agg: dict[str, float] = {}
    for prov in sorted(all_provs_set):
        vals = [m["per_provider"].get(prov, np.nan) for m in fold_metrics]
        per_prov_agg[prov] = float(np.nanmean(vals))

    # Measure the SHIPPED file set, not a directory walk (reviewer m3). The feature cache lives
    # inside ghost_gk_v1/ (~220 MB) and rglob swept it in, making the gate meaningless (2026-07-13
    # reported FAIL while the real payload was 14.64 MB). The bundled payload is exactly these
    # files (compare silly_kicks/tracking/_ghost_gk_weights/default/).
    _SHIPPED = ("rfcde_weights.npz", "metadata.json", "SHA256SUMS")
    artifact_bytes = sum((artifact_dir / f).stat().st_size for f in _SHIPPED if (artifact_dir / f).exists())

    # Derive game/provider counts from groups/provider_labels (always defined in BOTH
    # the fresh-extract and cache-load branches, and subsample-cap-aware) -- all_game_ids/
    # all_providers exist only on the fresh-extract path (PR-S81).
    metrics = {
        "n_games": len(set(groups.tolist())),
        "n_samples": len(features),
        "n_providers": len(set(provider_labels.tolist())),
        "providers": sorted({str(p) for p in provider_labels.tolist()}),
        "cv_folds": args.cv_folds,
        "subsample_fps": args.subsample_fps,
        "variant": args.variant,
        "carrier_params": cp,
        "training_commit": training_commit,
        # The artifact's own metadata.json records only the SHA. These two say whether that SHA
        # describes the code that ran: `--allow-dirty` permits a dev run, and this is where the
        # fact survives instead of living in someone's memory of how the run was invoked.
        "run_commit": run_prov["commit"],
        "run_tree_dirty": run_prov["dirty"],
        "run_tree_state": run_prov["tree_state"],
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
        "cv_mae_x_mean": float(np.mean(mae_x_vals)),
        "cv_mae_x_std": float(np.std(mae_x_vals)),
        "cv_mae_y_mean": float(np.mean(mae_y_vals)),
        "cv_mae_y_std": float(np.std(mae_y_vals)),
        "cv_mae_euclidean_mean": float(np.mean(mae_e_vals)),
        "cv_mae_euclidean_std": float(np.std(mae_e_vals)),
        "per_provider_mae_euclidean": per_prov_agg,
        "acceptance": {
            "overall_mae_lt_2m": float(np.mean(mae_e_vals)) < 2.0,
            "per_provider_mae_lt_3m": all(v < 3.0 for v in per_prov_agg.values()),
            "cross_fold_std_lt_05m": float(np.std(mae_e_vals)) < 0.5,
            "artifact_size_lt_15mb": artifact_bytes < 15_000_000,
        },
        "artifact_size_bytes": artifact_bytes,
    }

    # spec 4.3: keeper-grouped CV block (only on a --keeper-grouped run; dreport is None otherwise).
    if dreport is not None:
        metrics["keeper_grouped"] = {
            "n_domain_keepers": dreport.n_domain_keepers,
            "n_excluded_keepers": dreport.n_excluded_keepers,
            "underpowered": dreport.underpowered,
            "per_fold_mae_euclidean": mae_e_vals,
        }

    # spec 4.3 rev 5: measured selection-bias limitation (REPORTED, not a gate). Only populated on
    # a fresh SkillCorner extract (the cache-load path does not carry raw visibility). Computed
    # from summed (sum, count) counters rather than pooled lists, so a resumed extraction still
    # reports the whole corpus; `n_games_counters_unrecorded` says when that is not true.
    if bias.get("bias_depth_detected_n") and bias.get("bias_depth_undetected_n"):
        metrics["detection_selection_bias"] = {
            "keeper_depth_detected_mean": _mean_from_counters(
                bias["bias_depth_detected_sum"], bias["bias_depth_detected_n"]
            ),
            "keeper_depth_undetected_mean": _mean_from_counters(
                bias["bias_depth_undetected_sum"], bias["bias_depth_undetected_n"]
            ),
            "ball_to_keeper_dist_detected_mean": _mean_from_counters(
                bias["bias_b2k_detected_sum"], bias["bias_b2k_detected_n"]
            ),
            "ball_to_keeper_dist_undetected_mean": _mean_from_counters(
                bias["bias_b2k_undetected_sum"], bias["bias_b2k_undetected_n"]
            ),
            "n_detected": int(bias["bias_depth_detected_n"]),
            "n_undetected": int(bias["bias_depth_undetected_n"]),
            # Non-zero means some skipped game's counters could not be replayed (a pre-sidecar
            # shard generation, or a sidecar truncated by a kill), so the four means above
            # UNDER-cover the corpus. Reported rather than left to be inferred from a figure that
            # looks complete.
            "n_games_counters_unrecorded": int(bias_unrecorded),
            "note": (
                "Detected keeper frames are a SELECTION-BIASED sample (the camera sees the keeper "
                "when the ball is near him), so they over-represent the engaged/advanced keeper and "
                "under-sample the deep sweeper regime GKDV cares about. Depth = goal-relative x. "
                "This is a stated limitation, not a gate -- no rule in this cycle detects it. "
                "Both bias axes are measured here: keeper depth AND ball-to-keeper distance "
                "(goal-relative; detected frames are expected to show a SMALLER ball-to-keeper "
                "distance, which IS the selection mechanism). See spec 4.3 rev 5."
            ),
        }

    metrics_path = args.output_dir / "ghost_gk_v1" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Acceptance criteria
    print("\n=== Acceptance Criteria ===")
    for key, passed in metrics["acceptance"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
