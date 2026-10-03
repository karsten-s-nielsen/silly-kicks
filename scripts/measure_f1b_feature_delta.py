"""Measure the F1b float32-storage feature delta, per bundled frame-geometry model.

ADR-106 stores tracking-frame coordinates as float32 and computes in float64 (every kernel
upcasts the coord slice at its boundary). The storage rounding is a deterministic ~1e-5 m error
that is far below sensor precision but ABOVE the trained-model feature contract's atol=1e-6 -- so
it is a real feature-value change and a real retrain trigger. This driver measures that change.

For each frame-geometry model it runs the model's shared train/serve feature extractor on the SAME
real match twice -- once with float64 coordinates, once with the coordinates rounded to float32
storage -- and reports, per feature, the max and mean absolute delta plus the fraction of rows whose
delta exceeds atol=1e-6. It answers three questions the F1b retrain decision needs numbers for:

1. The drift is BOUNDED at storage-eps (a sanity check on the float32-storage/float64-compute design).
2. The drift EXCEEDS atol=1e-6 (which is what makes the retrain necessary, not optional).
3. gk_completion is FRAME-GEOMETRY, not action-only: its `dest_defender_density` feature reads
   tracking-frame player positions (via `receiver_zone_density`), so it moves under float32 while its
   seven action-coordinate features do not. This settles the classify-then-include question (spec
   4.3) with a measurement rather than an assumption.

The measurement is corpus-agnostic in character (the delta is a property of the storage rounding
propagated through each extractor), but it must read frames whose coordinates are stored as float64:
a float32-materialized corpus cannot expose the rounding (its baseline arm is already rounded). The
driver asserts the loaded coordinates are float64 and fails loud otherwise.

The two arms are aligned by a stable per-row KEY (each adapter returns ``(features, keys)``), matched
by inner join -- NOT by row position. The coord-selection-sensitive models (xshot/xcross via the
attacking-third gate, receiver via link_actions_to_frames) can select slightly different frame SETS
under float32 vs float64; a positional comparison would then subtract different frames and report a
spurious delta. The per-feature delta is measured over the COMMON keys; rows present in only one arm
are reported separately as ``selection_instability`` (they are a domain-filter flip, not a
feature-value change).

Shards are LONG format -- one ``status="selection"`` row per (match, model) carrying the only-in-one-
arm counts, then one ``status="ok"`` row per (match, model, feature) -- so the fixed-column ADR-052
schema holds while the feature set varies across models. Only the FAITHFUL (superset) feature set is
measured per model: a position_only set drops one velocity feature and shares every other feature's
delta exactly, so measuring the superset covers both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

from scripts._driver import for_each, reconcile
from scripts._provenance import git_provenance, require_clean_tree

#: The trained-model feature-contract tolerance (silly_kicks/tracking/_feature_contract.py). A per-row
#: delta above this is a contract-relevant change; the fraction above it justifies the retrain.
_ATOL = 1e-6

#: Frame coordinate/kinematic columns whose storage dtype F1b changes to float32.
_COORD_COLS = ("x", "y", "z", "vx", "vy", "speed", "x_smoothed", "y_smoothed")

#: SPADL shot action type names xShot labels/curates on (build_xshot_labels default).
_SHOT_TYPES = ("shot", "shot_penalty", "shot_freekick")


def frame_parquets(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Every TRACKING-FRAME parquet under `data_dir`, sidecar directories excluded.

    Accepts both the `for_each` shard layout (`shard_root/<token>/<key>.parquet`) and the named
    `{provider}/{id}/frames.parquet` tc3 tree, and excludes underscore-prefixed sidecars
    (`_actions/`, `_home/`) which are not frames. Mirrors `measure_box_constant_delta.frame_parquets`
    so the two measurement drivers read one corpus the same way.
    """
    named = sorted(data_dir.glob("**/frames.parquet"))
    if named:
        return named
    return sorted(
        p
        for p in data_dir.glob("**/*.parquet")
        if not any(part.startswith("_") for part in p.relative_to(data_dir).parts[:-1])
    )


def _match_key(fp: pathlib.Path, data_dir: pathlib.Path) -> str:
    """An injective `for_each` key from the full relative path (the `frames` stem collides)."""
    return "__".join(fp.relative_to(data_dir).with_suffix("").parts)


def _sidecar_key(fp: pathlib.Path) -> str:
    """The `_actions/`/`_home/` sidecar key: the stem, or the id dir in the named layout."""
    return fp.parent.name if fp.stem == "frames" else fp.stem


def _load_home_team_id(data_dir: pathlib.Path, key: str):
    """Per-match home_team_id from the tc3 `_home/<key>.json` sidecar, or None."""
    p = data_dir / "_home" / f"{key}.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("home_team_id", data) if isinstance(data, dict) else data


def _assert_float64_coords(frames: pd.DataFrame, fp: pathlib.Path) -> None:
    """A float32-stored corpus cannot expose the rounding -- its baseline arm is already rounded."""
    for c in ("x", "y"):
        if c in frames.columns and frames[c].dtype != np.float64:
            raise SystemExit(
                f"{fp}: coordinate '{c}' is {frames[c].dtype}, not float64. This driver measures the "
                f"float64->float32 storage rounding, so it needs a float64-stored corpus (materialized "
                f"before the ADR-106 schema change). Re-materialize from a float64 source."
            )


def _subsample_frames(frames: pd.DataFrame, fps: float = 1.0) -> pd.DataFrame:
    """Keep every step-th unique frame_id per (game_id, period_id), step = round(frame_rate/fps).

    The ghost models train on frames subsampled to ~1 fps (`prepare_ghost_gk_training_data` does this
    internally; `train_ghost_outfield` does it before extraction). Measuring the delta at full fps would
    (a) cost ~25x and (b) score frames the model never trains on. The keep is by frame_id, so the float64
    and float32 arms select the SAME frames and align row-for-row.
    """
    if fps <= 0 or "frame_rate" not in frames.columns or not len(frames):
        return frames
    fr = float(frames["frame_rate"].iloc[0])
    if fr <= 0:
        return frames
    step = max(1, round(fr / fps))
    if step == 1:
        return frames
    uniq = (
        frames[["game_id", "period_id", "frame_id"]].drop_duplicates().sort_values(["game_id", "period_id", "frame_id"])
    )
    keep = uniq[(uniq.groupby(["game_id", "period_id"]).cumcount() % step == 0).to_numpy()]
    return frames.merge(keep, on=["game_id", "period_id", "frame_id"], how="inner")


def _arm(frames: pd.DataFrame, dtype: str) -> pd.DataFrame:
    """A copy of `frames` with the coordinate/kinematic columns cast to `dtype`.

    The float32 arm mirrors F1b storage; every extractor then upcasts the slice to float64 at its
    kernel boundary (ADR-106), so the measured delta is exactly the storage-rounding effect.
    """
    target = np.dtype(dtype)  # a DtypeObj; bare `str` is not an astype overload under pandas-stubs
    out = frames.copy()
    for c in _COORD_COLS:
        if c in out.columns:
            out[c] = out[c].astype(target)
    return out


# --------------------------------------------------------------------------------------------
# Per-model feature adapters: each returns ``(features, keys)`` for one match's frames + actions, or
# None. ``keys`` is a per-row identity DataFrame ROW-ALIGNED to ``features``. The driver compares the
# two arms by KEY (inner-join), NOT by row position: the coord-selection-sensitive models (xshot /
# xcross via the attacking-third gate; receiver via link_actions_to_frames) can select slightly
# different frame SETS under float32 vs float64, so a positional row-for-row comparison would subtract
# DIFFERENT frames and report a spurious delta (the float32-storage feature delta is measured on the
# COMMON keys; rows present in only one arm are counted separately as selection-instability).


def _feat_xshot(frames, actions, home_team_id):
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, prepare_xshot_training_data

    if actions is None:
        return None
    feats, _y, _g, keys = prepare_xshot_training_data(
        frames, actions, home_team_id=home_team_id, feature_set="faithful", return_keys=True
    )
    if len(feats) == 0:
        return None
    return feats[list(XSHOT_FEATURE_NAMES_FAITHFUL)], keys[["game_id", "period_id", "frame_id"]]


def _feat_xcross(frames, actions, home_team_id):
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    if actions is None:
        return None
    # return_meta=True surfaces the row-aligned frame index (game_id/period_id/frame_id); the box-detail
    # columns it also carries are ignored -- only the selected-frame key is needed here.
    feats, _y, _g, meta = prepare_xcross_training_data(
        frames, actions, home_team_id=home_team_id, feature_set="faithful", return_meta=True
    )
    if len(feats) == 0:
        return None
    return feats[list(XCROSS_FEATURE_NAMES_FAITHFUL)], meta[["game_id", "period_id", "frame_id"]]


def _feat_ghost_gk(frames, actions, home_team_id):
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, prepare_ghost_gk_training_data

    # prepare_ghost_gk returns (features, meta); meta carries the row-aligned identity columns.
    feats, meta = prepare_ghost_gk_training_data(
        frames, home_team_id=home_team_id, actions=actions, feature_set="faithful"
    )
    if len(feats) == 0:
        return None
    return feats[list(GHOST_GK_FEATURE_NAMES)], meta[["game_id", "period_id", "frame_id", "gk_team_id"]]


def _feat_ghost_outfield(frames, actions, home_team_id):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier
    from silly_kicks.tracking._ghost_outfield import GHOST_OUTFIELD_FEATURE_NAMES, _extract_all_ghost_outfield_features

    if actions is None:
        return None
    frames = _subsample_frames(frames, fps=1.0)  # the training regime; infer_ball_carrier on full fps is ~25x cost
    carrier = None
    if "team_in_possession" not in frames.columns:
        c = infer_ball_carrier(frames, **dict(DEFAULT_CARRIER_PARAMS))  # type: ignore[arg-type]  # kwargs unpack of the carrier-params dict
        carrier = c[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]
    data = _extract_all_ghost_outfield_features(
        frames, actions, home_team_id=home_team_id, feature_set="faithful", carrier=carrier, both_teams=True
    )
    if len(data) == 0:
        return None
    key_cols = [c for c in ("game_id", "period_id", "frame_id", "team_id", "player_id") if c in data.columns]
    keys = data[key_cols].copy()
    # A slot can be empty (player_id NaN) or otherwise repeat; a per-key cumcount makes the identity unique.
    keys["_slot"] = keys.groupby(key_cols, dropna=False).cumcount()
    return data[list(GHOST_OUTFIELD_FEATURE_NAMES)], keys


def _feat_gk_completion(frames, actions, home_team_id):
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES, prepare_gk_completion_training_data

    if actions is None:
        return None
    feats, _y, _g = prepare_gk_completion_training_data(actions, frames=frames)
    if len(feats) == 0:
        return None
    # gk-pass selection is action-domain based (coord-independent), so both arms yield identical rows in
    # identical order -- a positional key is a valid identity, and only-in-one-arm counts stay 0 (a rare
    # geometry-edge count change surfaces as selection-instability, never a silent misalignment).
    keys = pd.DataFrame({"_row": np.arange(len(feats))})
    return feats[list(GK_COMPLETION_FEATURE_NAMES)], keys


def _feat_receiver(frames, actions, home_team_id):
    """One row per (pass, teammate): the public positions-only receiver features, keyed by
    ``(action_id, candidate_id)``.

    Passes are linked to their pre-pass frame via `link_actions_to_frames`; the per-pass extractor is
    the shared serve/train entry (`receiver_candidate_features`), so this is the model's real feature.
    """
    from silly_kicks.tracking._receiver import _PUBLIC_COLS, receiver_candidate_features
    from silly_kicks.tracking.utils import link_actions_to_frames

    if actions is None:
        return None
    passes = actions[actions.get("type_name", pd.Series(index=actions.index, dtype=object)) == "pass"]
    if len(passes) == 0:
        return None
    links, _report = link_actions_to_frames(passes, frames, on_low_coverage="ignore")
    fid_by_action = dict(zip(links["action_id"].to_numpy(), links["frame_id"].to_numpy(), strict=False))
    feat_rows = []
    key_rows = []
    for _, action in passes.iterrows():
        fid = fid_by_action.get(action["action_id"])
        if fid is None:
            continue
        frame = frames[frames["frame_id"] == fid]
        if frame.empty:
            continue
        feats = receiver_candidate_features(action, frame, feature_set="public")
        if len(feats):
            feat_rows.append(feats[list(_PUBLIC_COLS)])
            key_rows.append(
                pd.DataFrame({"action_id": action["action_id"], "candidate_id": feats["candidate_id"].to_numpy()})
            )
    if not feat_rows:
        return None
    return pd.concat(feat_rows, ignore_index=True), pd.concat(key_rows, ignore_index=True)


_MODEL_ADAPTERS = {
    "xshot": _feat_xshot,
    "xcross": _feat_xcross,
    "ghost_gk": _feat_ghost_gk,
    "ghost_outfield": _feat_ghost_outfield,
    "gk_completion": _feat_gk_completion,
    "receiver": _feat_receiver,
}

_SHARD_SCHEMA_VERSION = "f1b-feature-delta-2"  # bumped: key-aligned comparison + selection-instability counts
_EMITTED_SHARD_COLUMNS = (
    "match_key",
    "model",
    "feature",
    "status",
    "n_rows",
    "sum_abs",
    "max_abs",
    "n_gt_atol",
    "n_only_f64",
    "n_only_f32",
)


def _delta_rows(match_key: str, model: str, adapter, frames_f64, frames_f32, actions, home) -> list[dict]:
    """Per-feature delta rows for one (match, model), KEY-ALIGNED across the two arms.

    Each adapter returns ``(features, keys)``; the two arms are matched by an inner join on ``keys``,
    not by row position, so a coord-driven selection flip cannot masquerade as a feature delta. Emits
    one ``status="selection"`` row carrying the only-in-one-arm counts (``n_only_f64``/``n_only_f32``),
    then one ``status="ok"`` row per feature with the delta over the COMMON keys. Single status row on
    empty / error / duplicate-or-mismatched keys.
    """

    def _status(status: str) -> list[dict]:
        return [
            dict(
                match_key=match_key,
                model=model,
                feature="",
                status=status,
                n_rows=0,
                sum_abs=0.0,
                max_abs=0.0,
                n_gt_atol=0,
                n_only_f64=0,
                n_only_f32=0,
            )
        ]

    try:
        r64 = adapter(frames_f64, actions, home)
        r32 = adapter(frames_f32, actions, home)
    except Exception as exc:  # a per-model extractor failure must not abort the corpus pass
        return _status(f"error:{type(exc).__name__}")
    if r64 is None or r32 is None:
        return _status("empty")
    f64, k64 = r64
    f32, k32 = r32
    if len(f64) == 0 or len(f32) == 0:
        return _status("empty")
    if list(k64.columns) != list(k32.columns):
        return _status("key_schema_mismatch")

    # Per-row identity tuples. NaN is filled to a sentinel so an empty-slot key (e.g. ghost_outfield's
    # NaN player_id) self-matches across arms; ids are NOT cast between arms, so identical keys give
    # identical tuples. Match by KEY, not by row position (works for any key width, incl. 1 column).
    kk64 = k64.reset_index(drop=True).fillna("__NA__")
    kk32 = k32.reset_index(drop=True).fillna("__NA__")
    key64 = [tuple(r) for r in kk64.to_numpy()]
    key32 = [tuple(r) for r in kk32.to_numpy()]
    if len(set(key64)) != len(key64) or len(set(key32)) != len(key32):
        return _status("dup_keys")  # a non-unique key would make the join ambiguous -- fail honestly
    set32 = set(key32)
    pos32 = {k: i for i, k in enumerate(key32)}
    sel_f64 = [i for i, k in enumerate(key64) if k in set32]  # common keys, in f64 order
    n_only_f64 = len(key64) - len(sel_f64)
    n_only_f32 = len(key32) - len(sel_f64)
    if not sel_f64:
        return _status("no_common_keys")
    n_common = len(sel_f64)
    a = f64.reset_index(drop=True).iloc[sel_f64].reset_index(drop=True)
    b = f32.reset_index(drop=True).iloc[[pos32[key64[i]] for i in sel_f64]].reset_index(drop=True)

    out: list[dict] = [
        dict(
            match_key=match_key,
            model=model,
            feature="",
            status="selection",
            n_rows=n_common,
            sum_abs=0.0,
            max_abs=0.0,
            n_gt_atol=0,
            n_only_f64=n_only_f64,
            n_only_f32=n_only_f32,
        )
    ]
    for col in f64.columns:
        av = a[col].to_numpy(dtype="float64")
        bv = b[col].to_numpy(dtype="float64")
        both_finite = np.isfinite(av) & np.isfinite(bv)
        d = np.abs(av[both_finite] - bv[both_finite])
        out.append(
            dict(
                match_key=match_key,
                model=model,
                feature=str(col),
                status="ok",
                n_rows=int(d.size),
                sum_abs=float(d.sum()),
                max_abs=float(d.max()) if d.size else 0.0,
                n_gt_atol=int((d > _ATOL).sum()),
                n_only_f64=0,
                n_only_f32=0,
            )
        )
    return out


def _with_type_name(actions: pd.DataFrame | None) -> pd.DataFrame | None:
    """Restore the canonical SPADL ``type_name`` column from ``type_id`` when absent.

    tc3 shards store ``type_id`` only (space), but the extractors' domain filters key on
    ``type_name`` (shot/cross/pass/goalkick). The mapping is the single-sourced spadl config table.
    """
    if actions is None or "type_name" in actions.columns or "type_id" not in actions.columns:
        return actions
    import silly_kicks.spadl.config as spadlconfig

    names = dict(spadlconfig.actiontypes_df().itertuples(index=False, name=None))
    out = actions.copy()
    out["type_name"] = out["type_id"].map(names)
    return out


def _measure_one_match(fp: pathlib.Path, data_dir: pathlib.Path) -> pd.DataFrame:
    frames = pd.read_parquet(fp)
    _assert_float64_coords(frames, fp)
    skey = _sidecar_key(fp)
    actions_p = data_dir / "_actions" / f"{skey}.parquet"
    actions = _with_type_name(pd.read_parquet(actions_p)) if actions_p.is_file() else None
    home = _load_home_team_id(data_dir, skey)
    match_key = _match_key(fp, data_dir)

    frames_f64 = _arm(frames, "float64")
    frames_f32 = _arm(frames, "float32")

    rows: list[dict] = []
    for model, adapter in _MODEL_ADAPTERS.items():
        rows.extend(_delta_rows(match_key, model, adapter, frames_f64, frames_f32, actions, home))

    df = pd.DataFrame(rows, columns=list(_EMITTED_SHARD_COLUMNS))
    if set(df.columns) != set(_EMITTED_SHARD_COLUMNS):
        raise AssertionError(
            f"shard schema drift: {sorted(set(df.columns) ^ set(_EMITTED_SHARD_COLUMNS))} -- bump "
            f"_SHARD_SCHEMA_VERSION (ADR-052)."
        )
    return df


def _aggregate(combined: pd.DataFrame) -> dict:
    """Corpus per-(model, feature) rollup from summed counts (a mean of per-match means is wrong)."""
    ok = combined[combined["status"] == "ok"]
    models: dict[str, dict] = {}
    for model in sorted(combined["model"].unique()):
        m_ok = ok[ok["model"] == model]
        features: dict[str, dict] = {}
        for feat in sorted(m_ok["feature"].unique()):
            f = m_ok[m_ok["feature"] == feat]
            n = int(f["n_rows"].sum())
            features[feat] = {
                "n_rows": n,
                "mean_abs_delta": float(f["sum_abs"].sum() / n) if n else 0.0,
                "max_abs_delta": float(f["max_abs"].max()) if len(f) else 0.0,
                "frac_gt_atol": float(f["n_gt_atol"].sum() / n) if n else 0.0,
                "n_gt_atol": int(f["n_gt_atol"].sum()),
            }
        max_over_features = max((v["max_abs_delta"] for v in features.values()), default=0.0)
        moved = sorted(k for k, v in features.items() if v["max_abs_delta"] > _ATOL)
        unmoved = sorted(k for k, v in features.items() if v["max_abs_delta"] <= _ATOL)
        m_all = combined[combined["model"] == model]
        statuses = m_all["status"].value_counts().to_dict()
        # Selection-instability: rows selected in only ONE arm (a float32-driven domain-filter flip),
        # summed over the per-match `selection` rows. Kept SEPARATE from the per-feature delta so an
        # alignment artifact can never inflate a feature's max|delta| (the reason this driver was fixed).
        sel = m_all[m_all["status"] == "selection"]
        only_f64 = int(sel["n_only_f64"].sum())
        only_f32 = int(sel["n_only_f32"].sum())
        n_common = int(sel["n_rows"].sum())
        denom = only_f64 + only_f32 + n_common
        models[model] = {
            "max_abs_delta_over_features": max_over_features,
            "moved_features": moved,
            "unmoved_features": unmoved,
            "features": features,
            "match_status_counts": {str(k): int(v) for k, v in statuses.items()},
            "selection_instability": {
                "n_only_f64": only_f64,
                "n_only_f32": only_f32,
                "n_common": n_common,
                "frac": float((only_f64 + only_f32) / denom) if denom else 0.0,
            },
        }
    return models


def _token_inputs(paths: list[pathlib.Path], data_dir: pathlib.Path, commit: str) -> dict:
    """The for_each generation key: driver schema + the RUN COMMIT + the corpus identity (a digest of
    the FULL sorted key list, never a worker's subset). Shards are therefore attributable to a commit
    even when a worker dies before writing its manifest, and a worker resumed at another commit lands
    in another generation (combined-cycle spec section 6)."""
    keys = sorted(_match_key(p, data_dir) for p in paths)
    return {
        "schema": _SHARD_SCHEMA_VERSION,
        "driver": "f1b-feature-delta",
        "atol": _ATOL,
        "commit": commit,
        "corpus": hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest(),
    }


def _select(paths: list[pathlib.Path], data_dir: pathlib.Path, keys_json: str | None) -> list[pathlib.Path]:
    """The worker's subset: ``paths`` filtered to the keys in ``keys_json`` (a JSON list), corpus order."""
    if keys_json is None:
        return paths
    wanted = set(json.loads(pathlib.Path(keys_json).read_text(encoding="utf-8")))
    known = {_match_key(p, data_dir) for p in paths}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"--match-keys-json names {len(unknown)} key(s) absent from --data-dir: {unknown[:5]}")
    return [p for p in paths if _match_key(p, data_dir) in wanted]


def _map(paths: list[pathlib.Path], data_dir: pathlib.Path, out: pathlib.Path, *, token: dict):
    return for_each(
        paths,
        key=lambda fp: _match_key(fp, data_dir),
        work=lambda fp: _measure_one_match(fp, data_dir),
        shard_root=out / "_shards",
        token_inputs=token,
        label="match",
    )


def _write_worker_manifest(res, *, prov: dict, worker_tag: str) -> None:
    """Persist THIS worker's manifest beside its shards; the reduce reads every worker's."""
    (res.shard_dir / f"manifest_{worker_tag}.json").write_text(
        json.dumps({**res.manifest(), "run_commit": prov["commit"], "run_tree_dirty": prov["dirty"]}, default=str),
        encoding="utf-8",
    )


def _artifact(
    combined: pd.DataFrame, *, n_matches: int, manifest: dict, prov: dict, dirty: bool, n_accounted: int
) -> dict:
    """The metrics.json body -- ONE schema for the serial run and the sharded reduce (+ n_accounted)."""
    out: dict[str, object] = {"atol": _ATOL, "n_matches": n_matches, "models": _aggregate(combined)}
    out.update(manifest)
    out["n_accounted"] = n_accounted  # keys with a shard or exclusion marker; n_attempted counts only this pass
    out["run_commit"] = prov["commit"]
    out["run_tree_dirty"] = dirty
    return out


def reduce_t10(paths: list[pathlib.Path], data_dir: pathlib.Path, out: pathlib.Path, *, prov: dict) -> dict:
    """Reduce every worker's shards into the corpus artifact (combined-cycle spec section 6).

    Completeness is by ACCOUNTED KEYS (shard or exclusion marker for every listed key) in the ONE
    generation this commit's token inputs produce; every worker manifest present must name this commit.
    """
    from scripts._driver import _token, exclusion_path, shard_path
    from scripts._partition import aggregate_manifests

    root = out / "_shards"
    gens = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    expected = _token(_token_inputs(paths, data_dir, prov["commit"]), None)
    if [g.name for g in gens] != [expected]:
        raise SystemExit(
            f"expected exactly the generation {expected} (this commit + this corpus) under {root}, found "
            f"{[g.name for g in gens]}; a worker ran at another commit or on another corpus -- use a fresh --out"
        )
    gen = gens[0]
    keys = [_match_key(p, data_dir) for p in paths]
    missing = [k for k in keys if not shard_path(gen, k).is_file() and not exclusion_path(gen, k).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(keys)} listed matches have no shard (first: {missing[:3]}); a worker has "
            "not finished -- re-run it with the same --worker-tag (it resumes)."
        )
    agg = aggregate_manifests(gen, defaults=("n_attempted", "n_failed", "n_counters_unrecorded", "n_excluded"))
    foreign = sorted(set(agg["commits_seen"]) - {prov["commit"]})
    if foreign:
        raise SystemExit(f"worker manifest(s) from another commit {foreign}; this reduce runs at {prov['commit']}")
    combined = reconcile(gen, out / "f1b_feature_delta.parquet", tag="all")
    if not len(combined):
        raise SystemExit("every shard was empty -- the corpus yielded no measurable frames.")
    manifest = {
        "generation": gen.name,
        "n_attempted": agg["n_attempted"],
        "n_failed": agg["n_failed"],
        "n_counters_unrecorded": agg["n_counters_unrecorded"],
        "n_excluded": agg["n_excluded"],
    }
    return _artifact(
        combined,
        n_matches=len(paths),
        manifest=manifest,
        prov=prov,
        dirty=bool(prov["dirty"] or agg["run_tree_dirty"]),
        n_accounted=len(keys) - len(missing),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the F1b float32-storage per-feature delta by model.")
    ap.add_argument("--data-dir", type=pathlib.Path, required=True, help="float64-stored tracking-frame corpus")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="artifact directory (metrics.json written here)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--list-match-keys", action="store_true", help="print the corpus match keys as JSON and exit")
    ap.add_argument("--match-keys-json", default=None, help="JSON list of match keys this --shards-only worker handles")
    ap.add_argument(
        "--shards-only", action="store_true", help="MAP only: shards + manifest_<worker-tag>.json, no metrics.json"
    )
    ap.add_argument("--worker-tag", default=None, help="unique per-worker manifest tag (required with --shards-only)")
    ap.add_argument("--reduce-only", action="store_true", help="REDUCE only: metrics.json from every worker's shards")
    args = ap.parse_args()

    paths = frame_parquets(args.data_dir)
    if not paths:
        raise SystemExit(f"no frame parquets under {args.data_dir}. Point --data-dir at a float64 frame corpus.")
    if args.list_match_keys:
        print(json.dumps(sorted(_match_key(p, args.data_dir) for p in paths), indent=2))
        return
    if args.out is None:
        ap.error("--out is required unless --list-match-keys is given")
    if args.shards_only and args.reduce_only:
        ap.error("--shards-only and --reduce-only are mutually exclusive")
    if args.shards_only and not args.worker_tag:
        ap.error("--shards-only needs a unique --worker-tag")
    if args.match_keys_json and not args.shards_only:
        ap.error("--match-keys-json is a --shards-only worker flag; the reduce always covers the whole corpus")

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.reduce_only:
        out = reduce_t10(paths, args.data_dir, args.out, prov=prov)
    else:
        token = _token_inputs(paths, args.data_dir, prov["commit"])
        res = _map(_select(paths, args.data_dir, args.match_keys_json), args.data_dir, args.out, token=token)
        _write_worker_manifest(res, prov=prov, worker_tag=args.worker_tag or "serial")
        if args.shards_only:
            print(json.dumps({"shards_only": True, "worker_tag": args.worker_tag, **res.manifest()}, default=str))
            return
        combined = reconcile(res.shard_dir, args.out / "f1b_feature_delta.parquet", tag="all")
        if not len(combined):
            raise SystemExit("every shard was empty -- the corpus yielded no measurable frames.")
        out = _artifact(
            combined,
            n_matches=len(paths),
            manifest=res.manifest(),
            prov=prov,
            dirty=bool(prov["dirty"]),
            n_accounted=len(paths),
        )

    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
