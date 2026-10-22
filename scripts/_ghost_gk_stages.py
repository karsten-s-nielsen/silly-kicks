"""ADR-129: the reusable stages of `scripts.train_ghost_gk` (`validate`, `extract`); their closure must
not contain the trainer (`test_stage_modules`). The fit-side stages (cache/cv/fit/importance/assemble)
stay in the trainer and import the shared helpers (`_carrier`, `grid_spec_from_args`, `cache_token`,
`_LABEL_PREFIX`, `_SIDE_COLS`) from here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts._corpus_visibility import validate_corpus_visibility
from scripts._stages import ItemStage, Stage


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


#: The reusable stages (ADR-129). The lambdas are declared HERE, so every callable's ``__module__`` is this
#: stage module (``test_stage_modules``). The trainer imports STAGES and threads these into its PIPELINE.
_VALIDATE = Stage(
    name="validate",
    units=lambda ctx: ["only"],
    work=_validate_work,
    reduce=lambda r, ctx: r["only"],
    uses_dataset=True,
    dataset_files=_validate_files,
    # Upstream of `extract`: without its own cross-commit version its commit-keyed token would
    # fork the extract generation on every commit (decision P9).
)
_EXTRACT = ItemStage(
    name="extract",
    refs=_refs,
    key=lambda ref: ref.stem,  # one game per staged flat file; the dataset identity keys `home`
    load=_extract_load,
    work=_extract_work,
    counters=lambda _item, _frame: dict(_UNIT_BIAS),
    needs=("validate",),
    declare=_extraction_declare,
    uses_dataset=True,
    max_failures=0,  # the legacy raised on any failed game
)
STAGES = {"validate": _VALIDATE, "extract": _EXTRACT}
