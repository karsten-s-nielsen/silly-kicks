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

Shards are LONG format -- one row per (match, model, feature) -- so the fixed-column ADR-052 schema
holds while the feature set varies across models. Only the FAITHFUL (superset) feature set is
measured per model: a position_only set drops one velocity feature and shares every other feature's
delta exactly, so measuring the superset covers both.
"""

from __future__ import annotations

import argparse
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
# Per-model feature adapters: each returns a numeric feature DataFrame (declared features only) for
# one match's frames + actions. Only frame COORDINATES differ between the two arms, so the row
# selection is identical and the frames align row-for-row.


def _feat_xshot(frames, actions, home_team_id):
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, prepare_xshot_training_data

    shots = actions if actions is not None else None
    if shots is None:
        return None
    feats, _y, _g = prepare_xshot_training_data(frames, shots, home_team_id=home_team_id, feature_set="faithful")
    return feats[list(XSHOT_FEATURE_NAMES_FAITHFUL)]


def _feat_xcross(frames, actions, home_team_id):
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    if actions is None:
        return None
    feats, _y, _g = prepare_xcross_training_data(frames, actions, home_team_id=home_team_id, feature_set="faithful")
    return feats[list(XCROSS_FEATURE_NAMES_FAITHFUL)]


def _feat_ghost_gk(frames, actions, home_team_id):
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, prepare_ghost_gk_training_data

    feats, _y = prepare_ghost_gk_training_data(
        frames, home_team_id=home_team_id, actions=actions, feature_set="faithful"
    )
    return feats[list(GHOST_GK_FEATURE_NAMES)]


def _feat_ghost_outfield(frames, actions, home_team_id):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier
    from silly_kicks.tracking._ghost_outfield import GHOST_OUTFIELD_FEATURE_NAMES, _extract_all_ghost_outfield_features

    if actions is None:
        return None
    # The extractor needs the in-possession team per frame -- from a `team_in_possession` column, else
    # an inferred ball-carrier (train_ghost_outfield does exactly this; tc3 frames lack the column).
    # Ball-carrier is nearest-player-to-ball (discrete), so float32 storage rounding does not flip it;
    # each arm infers its own and they align, or the driver records a row_mismatch honestly.
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
    return data[list(GHOST_OUTFIELD_FEATURE_NAMES)]


def _feat_gk_completion(frames, actions, home_team_id):
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES, prepare_gk_completion_training_data

    if actions is None:
        return None
    feats, _y, _g = prepare_gk_completion_training_data(actions, frames=frames)
    if len(feats) == 0:
        return None
    return feats[list(GK_COMPLETION_FEATURE_NAMES)]


def _feat_receiver(frames, actions, home_team_id):
    """One row per (pass, teammate): the public positions-only receiver features.

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
    rows = []
    for _, action in passes.iterrows():
        fid = fid_by_action.get(action["action_id"])
        if fid is None:
            continue
        frame = frames[frames["frame_id"] == fid]
        if frame.empty:
            continue
        feats = receiver_candidate_features(action, frame, feature_set="public")
        if len(feats):
            rows.append(feats[list(_PUBLIC_COLS)])
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


_MODEL_ADAPTERS = {
    "xshot": _feat_xshot,
    "xcross": _feat_xcross,
    "ghost_gk": _feat_ghost_gk,
    "ghost_outfield": _feat_ghost_outfield,
    "gk_completion": _feat_gk_completion,
    "receiver": _feat_receiver,
}

_SHARD_SCHEMA_VERSION = "f1b-feature-delta-1"
_EMITTED_SHARD_COLUMNS = ("match_key", "model", "feature", "status", "n_rows", "sum_abs", "max_abs", "n_gt_atol")


def _delta_rows(match_key: str, model: str, adapter, frames_f64, frames_f32, actions, home) -> list[dict]:
    """Per-feature delta rows for one (match, model), or a single status row on skip/mismatch/error."""

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
            )
        ]

    try:
        f64 = adapter(frames_f64, actions, home)
        f32 = adapter(frames_f32, actions, home)
    except Exception as exc:  # a per-model extractor failure must not abort the corpus pass
        return _status(f"error:{type(exc).__name__}")
    if f64 is None or f32 is None or len(f64) == 0:
        return _status("empty")
    if len(f64) != len(f32):
        return _status("row_mismatch")

    out: list[dict] = []
    for col in f64.columns:
        a = f64[col].to_numpy(dtype="float64")
        b = f32[col].to_numpy(dtype="float64")
        both_finite = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a[both_finite] - b[both_finite])
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
        statuses = combined[combined["model"] == model]["status"].value_counts().to_dict()
        models[model] = {
            "max_abs_delta_over_features": max_over_features,
            "moved_features": moved,
            "unmoved_features": unmoved,
            "features": features,
            "match_status_counts": {str(k): int(v) for k, v in statuses.items()},
        }
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the F1b float32-storage per-feature delta by model.")
    ap.add_argument("--data-dir", type=pathlib.Path, required=True, help="float64-stored tracking-frame corpus")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="artifact directory (metrics.json written here)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    paths = frame_parquets(args.data_dir)
    if not paths:
        raise SystemExit(f"no frame parquets under {args.data_dir}. Point --data-dir at a float64 frame corpus.")

    args.out.mkdir(parents=True, exist_ok=True)
    res = for_each(
        paths,
        key=lambda fp: _match_key(fp, args.data_dir),
        work=lambda fp: _measure_one_match(fp, args.data_dir),
        shard_root=args.out / "_shards",
        token_inputs={"schema": _SHARD_SCHEMA_VERSION, "driver": "f1b-feature-delta", "atol": _ATOL},
        label="match",
    )
    combined = reconcile(res.shard_dir, args.out / "f1b_feature_delta.parquet", tag="all")
    if not len(combined):
        raise SystemExit("every shard was empty -- the corpus yielded no measurable frames.")

    out: dict[str, object] = {
        "atol": _ATOL,
        "n_matches": len(paths),
        "models": _aggregate(combined),
    }
    out.update(res.manifest())
    out["run_commit"] = prov["commit"]
    out["run_tree_dirty"] = prov["dirty"]

    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
