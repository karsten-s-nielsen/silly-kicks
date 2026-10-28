#!/usr/bin/env python
"""Train the Ghost-GK positioning model (TF-18).

Usage (ADR-128 stage contract; the uniform CLI from ``scripts._stages.main``)::

    python -m scripts.train_ghost_gk --dataset tc3 --variant default \
        --store <DIR> --data-root <DATA> --run-all --out <BUNDLE_DIR>

Pipeline: ``validate`` (provider + visibility + schema fail-fast, before any extraction) ->
``extract[match]`` (an ``ItemStage`` over the staged flat dataset: ``shards/<gen>/`` frames, ``_actions/``
and ``home_teams.json``) -> ``cache`` (the whole-corpus matrix) -> ``cv[fold]`` -> ``fit`` (final fit +
the BLOCKING parity gate) -> ``importance`` (metrics-only; one ``inner_parallel`` unit) -> ``assemble``
(the bundle). The tc3 / ``--data-dir`` / ``--actions-dir`` / ``--home-teams`` modes are gone (decision
P17): a corpus is staged first (``scripts/hf_jobs/datasets.toml``).

Requires: silly-kicks[train] (scikit-learn in [1.9, 2) for a shippable fit).

See docs/superpowers/specs/2026-05-26-tf18-training-hub-publish-design.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._corpus_visibility import validate_corpus_visibility
from scripts._stages import ItemStage, Param, Pipeline, Stage, cli_main, int_key


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


def high_sweeper_stratum_mae(preds, labels, *, threshold: float = 30.0) -> float:
    """Euclidean MAE over the HIGH-SWEEPER stratum only (labels with ``gk_x > threshold``).

    TF-60 PR3 sec 7.3: the sweeper re-fit must be validated to predict the >30 m regime, not merely
    lower the overall error. NaN on an empty stratum (nothing to average) -- reported, not gated.
    """
    gx = np.asarray(labels["gk_x"], dtype=float)
    gy = np.asarray(labels["gk_y"], dtype=float)
    mask = gx > threshold
    if not mask.any():
        return float("nan")
    dx = preds[mask, 0] - gx[mask]
    dy = preds[mask, 1] - gy[mask]
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def per_provider_high_sweeper_coverage(labels, provider_labels, *, threshold: float = 30.0) -> dict:
    """Per-provider fraction of retained keeper labels above ``threshold`` (goal-relative x).

    TF-60 PR3 sec 6: the high-sweeper stratum leans on the full-tracking providers (sportec/GS); this
    reports each provider's >30 m coverage so a consumer knows the provenance of the high regime.
    """
    gx = np.asarray(labels["gk_x"], dtype=float)
    provs = np.asarray(provider_labels)
    out: dict[str, float] = {}
    for p in sorted({str(x) for x in provs.tolist()}):
        sel = provs == p
        out[p] = float(np.mean(gx[sel] > threshold)) if sel.any() else float("nan")
    return out


def grid_spec_from_args(grid_x_max):
    """The `GhostGridSpec` a run trains on: DEFAULT_GHOST_GRID unless `--grid-x-max` extends the
    ceiling (TF-60 PR3). Only x_max is swept; y and resolution stay at the default (sec 5)."""
    from silly_kicks.tracking._ghost_gk import DEFAULT_GHOST_GRID, GhostGridSpec

    if grid_x_max is None:
        return DEFAULT_GHOST_GRID
    return GhostGridSpec(
        x_min=DEFAULT_GHOST_GRID.x_min,
        x_max=float(grid_x_max),
        y_min=DEFAULT_GHOST_GRID.y_min,
        y_max=DEFAULT_GHOST_GRID.y_max,
        resolution=DEFAULT_GHOST_GRID.resolution,
    )


def build_extraction_inputs(*, feature_set, grid_x_max, subsample_fps, carrier_params, with_actions):
    """The DECLARED shard-generation inputs (the digest `for_each`/`generation_dir` keys shards on).

    Extracted from `main()` so the stale-shard-token behavior is BEHAVIOURALLY testable without a full
    training run (TF-60 PR3): `grid_x_max` changes the shard's label ROWS (which keepers survive the
    domain filter), so it MUST key the generation or a re-run with a lifted ceiling silently reuses the
    stale <=30 m shards (the 4.77.1 trap; `feature_set` is here for the same reason -- 26 vs 21 cols).
    `geometry` is `cache_token()` (the penalty-area constants). main() no longer builds the dict inline.
    """
    return {
        "extractor": "prepare_ghost_gk_training_data",
        "feature_set": feature_set,
        "grid_x_max": grid_x_max,
        "geometry": cache_token(),
        "subsample_fps": subsample_fps,
        "carrier_params": dict(carrier_params),
        "with_actions": with_actions,
        "detected_targets_only": True,
    }


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


# ---- ADR-128 stage contract --------------------------------------------------------------------------
#
# validate (one unit) -> extract[match] -> cache (one unit) -> cv[fold] -> fit (one unit) ->
# importance (one inner_parallel unit) -> assemble (one unit, the bundle).
#
# Variant sharing (evidence E11): extract/cache declare only the extraction inputs, so `default` and
# `full` share them; cv/fit/importance declare the model inputs but NOT `variant`, so those two also share
# CV and the final fit. Only `assemble` (the `variant` label in metrics.json) differs.


def _to_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


_PARAMS = (
    Param(
        "dataset",
        str,
        None,
        required=True,
        help="the staged flat dataset (shards/<gen>/ + _actions/ + home_teams.json)",
    ),
    Param("variant", str, "full", choices=("default", "full", "sweeper", "sweeper_position_only")),
    Param("feature_set", str, "faithful", choices=("faithful", "position_only")),
    Param("grid_x_max", float, None, help="TF-60 PR3 extended label ceiling (e.g. 52.5); None = DEFAULT_GHOST_GRID"),
    Param("subsample_fps", float, 1.0),
    Param("n_estimators", int, 500),
    Param("max_depth", int, 8),
    Param("cv_folds", int, 5),
    Param("carrier_beta", float, None),
    Param("carrier_gamma", float, None),
    Param("carrier_tolerance", float, None),
    Param("subsample_cap", int, None, help="cap total training samples (~36000 for the bundled 'default')"),
    Param("skip_permutation_importance", flag=True),
    Param("perm_importance_sample", int, 150000),
    Param("keeper_grouped", flag=True),
    Param("expansion_keepers", str, None, help="an .npy of 98-cohort keeper ids (relative = the dataset root)"),
    # Every staged flat dataset carries _actions/; this keys the extraction token exactly as the legacy
    # `bool(actions_by_game)` did.
    Param("with_actions", _to_bool, True),
)


def _carrier(p) -> dict:
    """PR-S81: ONE carrier cp from the parameters (default = library), passed to prepare AND fit."""
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    cp = dict(DEFAULT_CARRIER_PARAMS)
    if p["carrier_tolerance"] is not None:
        cp["tolerance_m"] = p["carrier_tolerance"]
    if p["carrier_beta"] is not None:
        cp["beta"] = p["carrier_beta"]
    if p["carrier_gamma"] is not None:
        cp["gamma"] = p["carrier_gamma"]
    return cp


def _extraction_declare(p) -> dict:
    return build_extraction_inputs(
        feature_set=p["feature_set"],
        grid_x_max=grid_spec_from_args(p["grid_x_max"]).x_max,
        subsample_fps=p["subsample_fps"],
        carrier_params=_carrier(p),
        with_actions=bool(p["with_actions"]),
    )


def _model_declare(p) -> dict:
    """What cv/fit/importance numbers depend on beyond the extraction (NOT `variant`, E11)."""
    return {
        "feature_set": p["feature_set"],
        "grid_x_max": grid_spec_from_args(p["grid_x_max"]).x_max,
        "carrier_params": _carrier(p),
        "n_estimators": p["n_estimators"],
        "max_depth": p["max_depth"],
        "cv_folds": p["cv_folds"],
        "subsample_cap": p["subsample_cap"],
        "keeper_grouped": bool(p["keeper_grouped"]),
        "expansion_keepers": p["expansion_keepers"],
    }


def _root(ctx):
    from scripts._driver_sources import dataset_dir

    return dataset_dir(ctx.data_root, ctx.params["dataset"])


def _refs(ctx):
    from scripts._driver_sources import staged_flat_refs

    return staged_flat_refs(ctx.data_root, ctx.params["dataset"])


def _home_map(ctx) -> dict:
    from scripts._driver_sources import home_teams

    # The JSON values as written (the legacy `--home-teams` read them raw; only the keys are str game ids).
    return {str(k): v for k, v in home_teams(ctx.data_root, ctx.params["dataset"]).items()}


def _validate_work(key, ctx) -> dict:
    """Fail fast BEFORE any extraction (the legacy up-front checks): every provider classified, every
    detection-aware shard's visibility flag present, the frame schema, velocity, and a home-team map."""
    from scripts._driver_sources import read_dataset_schema

    root = _root(ctx)
    refs = _refs(ctx)
    if not refs:
        raise RuntimeError(f"no frame shards in the staged dataset {ctx.params['dataset']!r}")
    # `source_provider` is the shard's first row-group value (the staging lock on HF, the file locally) --
    # the legacy per-file read; "unknown" (no column) contributes nothing here, as before.
    provider_by_path = {root / r.files[0]: r.source_provider for r in refs if r.source_provider != "unknown"}
    if provider_by_path:
        validate_corpus_providers(sorted(set(provider_by_path.values())))
        validate_corpus_visibility(provider_by_path)
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
    probe_cols = set(read_dataset_schema(root / refs[0].files[0]).names)
    missing = required - probe_cols
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")
    if "vx" not in probe_cols or "vy" not in probe_cols:
        raise RuntimeError("vx/vy columns missing. Run smooth_frames + derive_velocities first.")
    if not _home_map(ctx):
        raise RuntimeError("no home team mapping: the staged dataset carries no home_teams.json")
    return {p.stem: prov for p, prov in sorted(provider_by_path.items())}


def _validate_files(ctx) -> list[str]:
    """The dataset files `_validate_work` reads (its ``dataset_files``): the first shard's schema, every
    detection-aware shard's footer (`validate_corpus_visibility`) and the home-team map."""
    from silly_kicks.tracking._provider_visibility import _DETECTION_AWARE_PROVIDERS

    refs = _refs(ctx)
    files = {
        r.files[0] for r in refs if r.source_provider != "unknown" and r.source_provider in _DETECTION_AWARE_PROVIDERS
    }
    if refs:
        files.add(refs[0].files[0])
    return sorted(files | {"home_teams.json"})


def _extract_load(ref, ctx):
    from scripts._driver_sources import read_dataset_parquet

    root = _root(ctx)
    frames = read_dataset_parquet(root / ref.files[0])
    apath = root / ref.files[1]
    actions = read_dataset_parquet(apath) if (ctx.params["with_actions"] and apath.exists()) else None
    return ref, frames, actions


#: The per-unit selection-bias sums `counters` reads (ADR-052: replayed from the sidecar on resume). Zeroed
#: at the start of every unit, so a unit that contributes nothing reports explicit 0s, never a stale sum.
_UNIT_BIAS: dict[str, float] = {}


def _extract_work(item, ctx):
    """The legacy per-game ``_work`` for every game in the unit's file (sorted), packs concatenated.

    The bias counters are summed over the file's games (one game per staged flat file, so the sums are
    the legacy per-game sums). A game with no home-team mapping is skipped, as the legacy did; a unit
    with NO mapped game is an exclusion, so the skip is reported in the stage summary."""
    from scripts._item_outcome import ItemExcluded
    from silly_kicks.tracking import prepare_ghost_gk_training_data
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, keeper_detection_mask

    _assert_no_column_collision(GHOST_GK_FEATURE_NAMES)
    _ref, file_frames, actions = item
    p = ctx.params
    cp = _carrier(p)
    grid_spec = grid_spec_from_args(p["grid_x_max"])
    home_map = _home_map(ctx)
    actions_by_game = dict(list(actions.groupby("game_id"))) if actions is not None and len(actions) else {}
    prov = str(file_frames["source_provider"].iloc[0]) if "source_provider" in file_frames.columns else "unknown"
    _UNIT_BIAS.clear()
    _UNIT_BIAS.update(dict.fromkeys(_BIAS_COUNTERS, 0.0))
    totals = _UNIT_BIAS
    packs = []
    game_ids = sorted(file_frames["game_id"].unique())
    unmapped = [g for g in game_ids if home_map.get(str(g)) is None]
    for game_id in unmapped:
        print(f"    SKIP game {game_id}: no home_team_id in mapping")
    if game_ids and len(unmapped) == len(game_ids):
        raise ItemExcluded("no home_team_id in mapping", details={"game_ids": [str(g) for g in unmapped]})
    for game_id in game_ids:
        home = home_map.get(str(game_id))
        if home is None:
            continue
        game_frames = file_frames[file_frames["game_id"] == game_id]
        game_actions = actions_by_game.get(game_id) if actions_by_game else None
        feats, labs, meta = prepare_ghost_gk_training_data(
            game_frames,
            home_team_id=home,
            actions=game_actions,
            subsample_fps=p["subsample_fps"],
            carrier_params=cp,
            feature_set=p["feature_set"],
            grid_spec=grid_spec,
            return_meta=True,
        )
        if not len(feats):
            continue
        if prov == "skillcorner" and len(meta):
            _vis = keeper_detection_mask_or_none(meta["gk_visibility"])
            if _vis is not None:
                _depth = meta["gk_x_gr"].to_numpy(dtype=float)
                _b2k = np.hypot(
                    feats["ball_x"].to_numpy() - meta["gk_x_gr"].to_numpy(),
                    feats["ball_y"].to_numpy() - meta["gk_y_gr"].to_numpy(),
                )
                _vm = _vis.to_numpy()
                for k, v in {
                    "bias_depth_detected_sum": float(np.sum(_depth[_vm])),
                    "bias_depth_detected_n": int(_vm.sum()),
                    "bias_depth_undetected_sum": float(np.sum(_depth[~_vm])),
                    "bias_depth_undetected_n": int((~_vm).sum()),
                    "bias_b2k_detected_sum": float(np.nansum(_b2k[_vm])),
                    "bias_b2k_detected_n": int((~np.isnan(_b2k[_vm])).sum()),
                    "bias_b2k_undetected_sum": float(np.nansum(_b2k[~_vm])),
                    "bias_b2k_undetected_n": int((~np.isnan(_b2k[~_vm])).sum()),
                }.items():
                    totals[k] += v
        keep = keeper_detection_mask(meta["gk_visibility"], provider=prov)
        feats = feats[keep].reset_index(drop=True)
        labs = labs[keep].reset_index(drop=True)
        meta = meta[keep].reset_index(drop=True)
        if not len(feats):
            print(f"  SKIP {prov}/{game_id}: no detected-keeper frames")
            continue
        packs.append(_pack(feats, labs, game_id=game_id, provider=prov, keepers=meta["gk_player_id"].astype(str)))
    return pd.concat(packs, ignore_index=True) if packs else None


def _cache_work(key, ctx) -> pd.DataFrame:
    """The whole-corpus combined table: every non-empty extract shard in sorted key order (the legacy
    `res.keys` order: sorted files, then games). Downstream stages `_unpack` it."""
    from scripts._stages import read_result, sorted_units, unit_states

    st = ctx.pipeline.stage("extract")
    gen = ctx.generation("extract")
    done = [k for k, v in unit_states(ctx, st, sorted_units(ctx, st), verify=False).items() if v == "done"]
    parts = [f for f in (read_result(ctx.store, f"{gen}/{k}") for k in done) if len(f)]
    if not parts:
        raise RuntimeError("no training samples extracted")
    return _combine_packs(parts)


def _combine_packs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """The packs concatenated, with ``_game_id`` storable: a multi-provider corpus mixes int and str ids,
    and the legacy grouped on ``np.array(_game_id.tolist())``, which made every id a string -- the same
    strings `_stages.storable_id_column` stores, so the CV folds do not move."""
    from scripts._stages import combine_frames

    return combine_frames(parts, ids=("_game_id",))


_CORPUS: dict = {}


def _corpus(ctx):
    """``(features, labels, groups, provider_labels, keepers)`` from the cache, with the seeded subsample
    cap applied (PR-S81) -- loaded ONCE per worker (keyed by the cache token)."""
    tok = ctx.tokens["cache"]
    if tok not in _CORPUS:
        features, labels, groups, provider_labels, keepers = _unpack(ctx.upstream("cache"))
        cap = ctx.params["subsample_cap"]
        if cap is not None and len(features) > cap:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(features), size=cap, replace=False)
            keep.sort()
            features = features.iloc[keep].reset_index(drop=True)
            labels = labels.iloc[keep].reset_index(drop=True)
            groups = groups[keep]
            provider_labels = provider_labels[keep]
            keepers = keepers[keep]
        _CORPUS.clear()
        _CORPUS[tok] = (features, labels, groups, provider_labels, keepers)
    return _CORPUS[tok]


def _folds(ctx):
    """The seeded CV split (either regime, spec 4.3) and the keeper-domain report (or None)."""
    features, _labels, groups, provider_labels, keepers = _corpus(ctx)
    p = ctx.params
    if p["keeper_grouped"]:
        from _ghost_domain import common_keeper_domain, keeper_folds

        if p["expansion_keepers"] is None:
            raise RuntimeError(
                "--keeper-grouped requires --expansion-keepers (the keeper ids present in the 98-match cohort)"
            )
        ek = Path(p["expansion_keepers"])
        ek = ek if ek.is_absolute() else _root(ctx) / ek
        expansion_keepers = set(np.load(ek, allow_pickle=True).astype(str).tolist())
        domain, dreport = common_keeper_domain(keepers, expansion_keepers=expansion_keepers, n_splits=p["cv_folds"])
        if set(keepers[domain].tolist()) & expansion_keepers:
            raise RuntimeError("a domain keeper is in the expansion cohort -- the common-domain exclusion failed")
        return list(keeper_folds(keepers, domain, n_splits=p["cv_folds"])), dreport
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=p["cv_folds"], shuffle=True, random_state=42)
    return list(cv.split(features, provider_labels, groups)), None


def _cv_work(key, ctx) -> dict:
    from scripts._train_guard import require_training_sklearn
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    require_training_sklearn()
    features, labels, _groups, provider_labels, _keepers = _corpus(ctx)
    folds, _dreport = _folds(ctx)
    train_idx, test_idx = folds[int(key)]
    p = ctx.params
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    model = GhostGkModel(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        verbose=1,
        feature_set=p["feature_set"],
        grid_spec=grid_spec_from_args(p["grid_x_max"]),
    )
    model.fit(X_train, y_train, carrier_params=_carrier(p))
    preds = model.predict_mean(X_test)
    mae_x = float(np.mean(np.abs(preds[:, 0] - y_test["gk_x"].values)))
    mae_y = float(np.mean(np.abs(preds[:, 1] - y_test["gk_y"].values)))
    mae_euclid = float(
        np.mean(np.sqrt((preds[:, 0] - y_test["gk_x"].values) ** 2 + (preds[:, 1] - y_test["gk_y"].values) ** 2))
    )
    test_provs = provider_labels[test_idx]
    per_prov: dict[str, float] = {}
    for prov in np.unique(test_provs):
        mask = test_provs == prov
        per_prov[str(prov)] = float(
            np.mean(
                np.sqrt(
                    (preds[mask, 0] - y_test["gk_x"].values[mask]) ** 2
                    + (preds[mask, 1] - y_test["gk_y"].values[mask]) ** 2
                )
            )
        )
    return {
        "mae_x": mae_x,
        "mae_y": mae_y,
        "mae_euclidean": mae_euclid,
        "per_provider": per_prov,
        # TF-60 PR3: out-of-sample euclid MAE over the >30 m high-sweeper stratum (NaN if none).
        "high_sweeper_mae": high_sweeper_stratum_mae(
            preds, {"gk_x": y_test["gk_x"].values, "gk_y": y_test["gk_y"].values}
        ),
    }


def _fit_work(key, ctx):
    """Final fit + the BLOCKING parity-on-fresh-fit gate (Option A, ADR-016) + save + round-trip."""
    import tempfile

    from scripts._stages import FileSet
    from scripts._train_guard import require_training_sklearn
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    require_training_sklearn()
    features, labels, groups, provider_labels, _keepers = _corpus(ctx)
    p = ctx.params
    cp = _carrier(p)
    final_model = GhostGkModel(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        verbose=1,
        feature_set=p["feature_set"],
        grid_spec=grid_spec_from_args(p["grid_x_max"]),
    )
    final_model.fit(features, labels, carrier_params=cp)
    _par_n = min(20000, len(features))
    _par_idx = np.random.default_rng(0).choice(len(features), _par_n, replace=False)
    _par_Xv = features.iloc[_par_idx][features.columns].values
    _sk_reg_x, _sk_reg_y = final_model._sk_reg_x, final_model._sk_reg_y
    _tree_nodes, _tree_nodes_y = final_model._tree_nodes, final_model._tree_nodes_y
    if _sk_reg_x is None or _sk_reg_y is None or _tree_nodes is None or _tree_nodes_y is None:
        raise RuntimeError("the fresh fit did not populate the regressors / tree arrays")
    _par_sk = np.column_stack([_sk_reg_x.predict(_par_Xv), _sk_reg_y.predict(_par_Xv)])
    _par_err = float(np.abs(final_model.predict_mean(features.iloc[_par_idx]) - _par_sk).max())
    _par_ncat = sum(int(t["is_categorical"].sum()) for t in _tree_nodes) + sum(
        int(t["is_categorical"].sum()) for t in _tree_nodes_y
    )
    print(f"PARITY GATE: max|predict_mean - sklearn| = {_par_err:.2e} over {_par_n} rows; n_cat = {_par_ncat}")
    if _par_err > 1e-6 or _par_ncat != 0:
        raise RuntimeError(f"BLOCKING parity gate FAILED (err={_par_err:.2e}, ncat={_par_ncat}); refusing to publish.")
    training_commit = None if ctx.prov["commit"] == "unknown" else ctx.prov["commit"]
    final_model.training_commit = training_commit
    # Aggregate corpus provenance (providers + counts ONLY; spec 2026-07-20 S6).
    final_model.corpus_provenance = {
        "providers": sorted({str(x) for x in provider_labels.tolist()}),
        "n_games": len(set(groups.tolist())),
        "n_rows": len(features),
    }
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "ghost_gk_v1"
        final_model.save(artifact_dir)
        loaded = GhostGkModel.load(artifact_dir)
        for attr in ("_tree_nodes", "_tree_nodes_y"):
            for i, (a, b) in enumerate(zip(getattr(final_model, attr), getattr(loaded, attr), strict=True)):
                np.testing.assert_array_equal(a, b, err_msg=f"{attr}[{i}]")
        if loaded._baseline_x != final_model._baseline_x or loaded._baseline_y != final_model._baseline_y:
            raise RuntimeError("round-trip verification: a baseline drifted")
        if loaded.carrier_params != cp:
            raise RuntimeError(f"carrier_params drift: {loaded.carrier_params} != {cp}")
        files = {q.name: q.read_bytes() for q in sorted(artifact_dir.iterdir()) if q.is_file()}
    return FileSet(files)


def _importance_work(key, ctx) -> dict:
    """Permutation importance (metrics-only: printed, never in metrics.json). ONE inner_parallel unit that
    calls scikit-learn with ``n_jobs = ctx.pool.workers`` (decision P11), so scikit-learn's own seed
    stream is unchanged; the model is the fit stage's saved artifact (bit-identical predictions)."""
    import tempfile

    from scripts._stages import materialize_fileset
    from scripts._train_guard import require_training_sklearn
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    p = ctx.params
    if p["skip_permutation_importance"]:
        print("Skipping permutation importance (--skip-permutation-importance).")
        return {"skipped": True, "top": []}
    require_training_sklearn()
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.inspection import permutation_importance

    features, labels, _groups, _provs, _keepers = _corpus(ctx)
    with tempfile.TemporaryDirectory() as tmp:
        model = GhostGkModel.load(materialize_fileset(ctx, "fit", "_output", Path(tmp)))
    columns = features.columns
    pid_log = os.environ.get("SILLY_KICKS_IMPORTANCE_PID_LOG")  # test hook (decision P11); never set in a job

    class _SklearnWrapper(BaseEstimator, RegressorMixin):  # type: ignore[misc]
        def __init__(self, m: GhostGkModel | None = None) -> None:
            self.m = m

        def fit(self, X: np.ndarray, y: np.ndarray) -> _SklearnWrapper:
            return self  # already fitted

        def predict(self, X: np.ndarray) -> np.ndarray:
            if pid_log:
                with open(pid_log, "a", encoding="ascii") as fh:
                    fh.write(f"{os.getpid()}\n")
            if self.m is None:
                raise RuntimeError("the importance wrapper holds no model")
            return self.m.predict_mean(pd.DataFrame(X, columns=columns))[:, 0]

    cap = p["perm_importance_sample"]
    if cap and cap < len(features):
        idx = np.random.default_rng(42).choice(len(features), cap, replace=False)
        X, y = features.values[idx], labels["gk_x"].values[idx]
    else:
        X, y = features.values, labels["gk_x"].values
    n_jobs = ctx.pool.workers if ctx.pool is not None else 1
    print(f"Running permutation importance (5 repeats, n_jobs={n_jobs}) on {len(X)} eval rows...")
    pi = permutation_importance(
        _SklearnWrapper(m=model),
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=n_jobs,
    )
    importances = sorted(
        zip(columns, pi.importances_mean, strict=True),  # type: ignore[reportAttributeAccessIssue]
        key=lambda x: -x[1],
    )
    print("Top 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")
    return {"skipped": False, "top": [[str(n), float(v)] for n, v in importances]}


def _assemble_work(key, ctx):
    """The legacy metrics block (fold aggregation, bias block, acceptance) + the bundle FileSet."""
    from scripts._corpus_identity import reproducibility
    from scripts._stages import FileSet, item_counters

    p = ctx.params
    features, labels, groups, provider_labels, _keepers = _corpus(ctx)
    folds_out = ctx.upstream("cv")
    fold_metrics = [folds_out[k] for k in sorted(folds_out)]
    _folds_unused, dreport = _folds(ctx) if p["keeper_grouped"] else (None, None)
    fit = ctx.upstream("fit")
    cp = _carrier(p)
    grid_spec = grid_spec_from_args(p["grid_x_max"])
    training_commit = None if ctx.prov["commit"] == "unknown" else ctx.prov["commit"]
    mae_x_vals = [m["mae_x"] for m in fold_metrics]
    mae_y_vals = [m["mae_y"] for m in fold_metrics]
    mae_e_vals = [m["mae_euclidean"] for m in fold_metrics]
    all_provs_set: set[str] = set()
    for m in fold_metrics:
        all_provs_set.update(m["per_provider"].keys())
    per_prov_agg: dict[str, float] = {}
    for prov in sorted(all_provs_set):
        vals = [m["per_provider"].get(prov, np.nan) for m in fold_metrics]
        per_prov_agg[prov] = float(np.nanmean(vals))
    _SHIPPED = ("rfcde_weights.npz", "metadata.json", "SHA256SUMS")
    artifact_bytes = sum(len(fit.files[f]) for f in _SHIPPED if f in fit.files)
    metrics = {
        "n_games": len(set(groups.tolist())),
        "n_samples": len(features),
        "n_providers": len(set(provider_labels.tolist())),
        "providers": sorted({str(x) for x in provider_labels.tolist()}),
        "cv_folds": p["cv_folds"],
        "subsample_fps": p["subsample_fps"],
        "variant": p["variant"],
        "carrier_params": cp,
        "training_commit": training_commit,
        "run_commit": ctx.prov["commit"],
        "run_tree_dirty": ctx.prov["dirty"],
        "run_tree_state": ctx.prov["tree_state"],
        # ADR-067 M4, emitted by the trainer: a staged corpus carries no public-visibility proof, so it is
        # restricted (fail-closed, like the public-row rule's default).
        **reproducibility("restricted", provider_labels.tolist(), training_commit=training_commit),
        "hyperparameters": {"n_estimators": p["n_estimators"], "max_depth": p["max_depth"]},
        "cv_mae_x_mean": float(np.mean(mae_x_vals)),
        "cv_mae_x_std": float(np.std(mae_x_vals)),
        "cv_mae_y_mean": float(np.mean(mae_y_vals)),
        "cv_mae_y_std": float(np.std(mae_y_vals)),
        "cv_mae_euclidean_mean": float(np.mean(mae_e_vals)),
        "cv_mae_euclidean_std": float(np.std(mae_e_vals)),
        "per_provider_mae_euclidean": per_prov_agg,
        "grid_x_max": grid_spec.x_max,
        "high_sweeper_stratum_mae_mean": float(np.nanmean([m["high_sweeper_mae"] for m in fold_metrics])),
        "per_provider_high_sweeper_coverage": per_provider_high_sweeper_coverage(
            {"gk_x": labels["gk_x"].to_numpy()}, provider_labels
        ),
        "acceptance": {
            "overall_mae_lt_2m": float(np.mean(mae_e_vals)) < 2.0,
            "per_provider_mae_lt_3m": all(v < 3.0 for v in per_prov_agg.values()),
            "cross_fold_std_lt_05m": float(np.std(mae_e_vals)) < 0.5,
            "artifact_size_lt_15mb": artifact_bytes < 15_000_000,
        },
        "artifact_size_bytes": artifact_bytes,
    }
    if dreport is not None:
        metrics["keeper_grouped"] = {
            "n_domain_keepers": dreport.n_domain_keepers,
            "n_excluded_keepers": dreport.n_excluded_keepers,
            "underpowered": dreport.underpowered,
            "per_fold_mae_euclidean": mae_e_vals,
        }
    # spec 4.3 rev 5: the selection-bias limitation from the CORPUS-scoped counters (summed over every
    # extract unit in sorted key order), so a resumed extraction still reports the whole corpus.
    bias, bias_unrecorded = item_counters(ctx, "extract")
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
    files = dict(fit.files)
    files["metrics.json"] = json.dumps(metrics, indent=2).encode("utf-8")  # the legacy json.dump bytes
    print("\n=== Acceptance Criteria ===")
    for k, passed in metrics["acceptance"].items():
        print(f"  {k}: {'PASS' if passed else 'FAIL'}")
    return FileSet(files)


PIPELINE = Pipeline(
    name="ghost_gk",
    module="scripts.train_ghost_gk",
    params=_PARAMS,
    stages=(
        Stage(
            name="validate",
            units=lambda ctx: ["only"],
            work=_validate_work,
            reduce=lambda r, ctx: r["only"],
            uses_dataset=True,
            dataset_files=_validate_files,
            # Upstream of `extract`: without its own cross-commit version its commit-keyed token would
            # fork the extract generation on every commit (decision P9).
            cross_commit_version="ghost-gk-validate-1",
        ),
        ItemStage(
            name="extract",
            refs=_refs,
            key=lambda ref: ref.stem,  # one game per staged flat file; the dataset identity keys `home`
            load=_extract_load,
            work=_extract_work,
            counters=lambda _item, _frame: dict(_UNIT_BIAS),
            needs=("validate",),
            declare=_extraction_declare,
            cross_commit_version="ghost-gk-extract-1",
            uses_dataset=True,
            max_failures=0,  # the legacy raised on any failed game
        ),
        Stage(
            name="cache",
            units=lambda ctx: ["only"],
            work=_cache_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],
        ),
        Stage(
            name="cv",
            units=lambda ctx: [int_key(k) for k in range(int(ctx.params["cv_folds"]))],
            work=_cv_work,
            needs=("cache",),
            reduce=lambda r, ctx: dict(r),
            declare=_model_declare,
            thread_safe=True,  # HFU T11: each fold fits a HistGB; W=1 gets the full vCPU OMP budget
            thread_lever="omp",  # HistGB OpenMP; BLAS stays serial (B38) -- identity test_cv_thread_identity_ghost_gk
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("cache",),
            reduce=lambda r, ctx: r["only"],
            declare=_model_declare,
            thread_safe=True,  # ADR-128 INV-2 opt-in (test_thread_identity; test_thread_safe_registry)
            thread_lever="omp",  # HistGB OpenMP; BLAS stays serial (B38)
        ),
        Stage(
            name="importance",
            units=lambda ctx: ["only"],
            work=_importance_work,
            needs=("cache", "fit"),
            reduce=lambda r, ctx: r["only"],
            declare=_model_declare,
            params=("skip_permutation_importance", "perm_importance_sample"),
            inner_parallel=True,
            inner_width=8,
        ),
        Stage(
            name="assemble",
            units=lambda ctx: ["only"],
            work=_assemble_work,
            needs=("extract", "cache", "cv", "fit", "importance"),
            reduce=lambda r, ctx: r["only"],
            declare=_model_declare,
            params=("variant", "subsample_fps"),
        ),
    ),
    output="assemble",
)

if __name__ == "__main__":
    cli_main(PIPELINE)
