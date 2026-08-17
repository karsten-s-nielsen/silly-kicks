"""Measure what the ghost box-constant unification actually changes, before re-fitting.

Spec D2: the re-fit happens either way (the contract raises on an unaccompanied flip), but the SHIP
CLAIM depends on this count. Zero-versus-nonzero is the whole question, so no threshold is needed --
but the count must be ATTRIBUTABLE to the band vs the depth boundary, or it is a number nobody can
reason about next cycle.

Also counts the behind-the-line population (`gr_x < 0`) while the rows are in hand: ADR-050 parks
"should a behind-the-line point count as in-box", and answering it later would otherwise cost a
second corpus pass over data we are already holding.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
import silly_kicks.tracking._geometry as _geo
from scripts._driver import for_each, reconcile
from scripts._offpitch import OFF_PITCH_MARGIN_M
from scripts._provenance import git_provenance, require_clean_tree

_LEGACY_BOX_WIDTH = 40.3
_LEGACY_DEPTH = 16.5


def frame_parquets(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Every TRACKING-FRAME parquet under `data_dir`, and nothing else.

    Accepts BOTH layouts. `materialize_tc3_frames` adopts ADR-052's `for_each`, which writes
    `shard_root/<token>/<key>.parquet` -- NOT `{provider}/{id}/frames.parquet`. A
    `**/frames.parquet` glob silently finds nothing there and this driver exits having measured no
    rows, which on a remote box reads as "the corpus is empty" rather than "you pointed at the wrong
    directory".

    **Sidecar directories are EXCLUDED, and that is a regression fix, not tidiness.** The bare
    `**/*.parquet` fallback was safe only while frames were the sole parquet under `--out`. Once
    `materialize_tc3_frames` began emitting `_actions/<key>.parquet` for the trainer, the fallback
    swept those up too and this driver died mid-corpus on
    `ArrowInvalid: No match for FieldRef.Name(x)` -- SPADL actions carry `start_x`/`end_x`, never
    `x`. Measured on the real 179-match pass.

    The rule is the leading underscore: `_actions/`, `_home/` and any future sidecar are siblings of
    the shard generation under one `--out`, and none of them holds frames. Excluding by NAME rather
    than by probing each file's schema keeps this cheap and, more importantly, keeps a genuinely
    malformed FRAME shard loud instead of silently skipped.
    """
    named = sorted(data_dir.glob("**/frames.parquet"))
    if named:
        return named
    return sorted(
        p
        for p in data_dir.glob("**/*.parquet")
        if not any(part.startswith("_") for part in p.relative_to(data_dir).parts[:-1])
    )


_GOAL_Y = 34.0


def _legacy_y_in_band(y: np.ndarray) -> np.ndarray:
    """The MIN/MAX BAND form ghost actually shipped -- NOT the abs form.

    Do not "simplify" this to `np.abs(y - 34.0) <= 40.3/2`. Spec 1.1 item 3 proves the two forms
    equivalent at the CANONICAL constant and explicitly records that they DISAGREE at the LEGACY
    one, at exactly `y = 13.85`: the double sits fractionally below `(68-40.3)/2`, so the band says
    outside while the abs form says inside. Modelling legacy with the abs form makes that row a
    no-flip when it is a real flip -- an undercount at precisely the boundary this driver measures.
    """
    return (y >= (68.0 - _LEGACY_BOX_WIDTH) / 2.0) & (y <= (68.0 + _LEGACY_BOX_WIDTH) / 2.0)


def classify_flips(gr_x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Split the legacy-vs-canonical disagreement by CAUSE, three ways.

    A three-way split states a fact; a two-way split forces a convention. A row where both changes
    are individually NECESSARY (y in the 1 cm strip AND x exactly on the depth line) belongs to
    neither pure bucket, and folding it into one makes the other a systematic undercount.
    """
    legacy_y = _legacy_y_in_band(y)
    canon_y = np.abs(y - _GOAL_Y) <= spadlconfig.penalty_area_half_width
    legacy_x = gr_x < _LEGACY_DEPTH
    canon_x = gr_x <= spadlconfig.penalty_area_depth

    flipped = (legacy_x & legacy_y) != (canon_x & canon_y)
    y_agrees = legacy_y == canon_y
    x_agrees = legacy_x == canon_x
    return {
        "n_flipped": int(flipped.sum()),
        "n_flipped_band_only": int((flipped & ~y_agrees & x_agrees).sum()),
        "n_flipped_boundary_only": int((flipped & y_agrees & ~x_agrees).sum()),
        "n_flipped_both": int((flipped & ~y_agrees & ~x_agrees).sum()),
    }


# --------------------------------------------------------------------------------------------
# Phase B (basis A): the training-feature-delta measurement under a `gr_x >= 0` clamp.


@contextlib.contextmanager
def _scoped_gr_x_clamp():
    """Measurement-only: patch the MODULE ATTRIBUTE ``in_penalty_area_goal_relative_array``.

    Both consuming extractors (``_ghost_gk``, ``_xcross_attempt``) call it via attribute access on
    the ``_geometry`` module (``_geo.in_penalty_area_goal_relative_array``), so a patched attribute
    is seen by both (N1). This ships NO clamp -- the real clamp is a declared constant that lands
    only if the decision warrants it (spec 5.4). The bound is ``0.0``, the value
    ``penalty_area_min_gr_x`` would take (N2).
    """
    original = _geo.in_penalty_area_goal_relative_array

    def _clamped(gr_x, y):
        return original(gr_x, y) & (np.asarray(gr_x) >= 0.0)

    _geo.in_penalty_area_goal_relative_array = _clamped
    try:
        yield
    finally:
        _geo.in_penalty_area_goal_relative_array = original


def _ghost_behind_line_grx_per_example(frames: pd.DataFrame, meta: pd.DataFrame) -> list:
    """Behind-line-in-band ATTACKER gr_x per ghost example, aligned row-for-row to ``meta``.

    Mirrors the extractor's selection -- ``attacking = other-team, non-GK`` (``_ghost_gk.py:653``) --
    and its goal-relative transform (defended goal via the ADR-055 map), so the reconstruction
    describes the SAME points as ``attackers_in_box`` and the N-1 coherence holds.
    """
    from silly_kicks.id_compat import ids_match
    from silly_kicks.tracking._geometry import GOAL_Y, to_goal_relative_x
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals

    hw = float(spadlconfig.penalty_area_half_width)
    gmap = resolve_defended_goals(frames)
    is_ball = frames["is_ball"].fillna(False).astype(bool)
    is_gk = frames["is_goalkeeper"].fillna(False).astype(bool)
    outfield = frames[~is_ball & ~is_gk]

    per_example: list = []
    for _, m in meta.iterrows():
        goal_x = gmap.get(m["game_id"], m["period_id"], m["gk_team_id"])
        atk = outfield[(outfield["frame_id"] == m["frame_id"]) & ~ids_match(outfield["team_id"], m["gk_team_id"])]
        if goal_x is None or atk.empty:
            per_example.append(np.array([], dtype=float))
            continue
        xs = atk["x"].to_numpy(dtype=float)
        ys = atk["y"].to_numpy(dtype=float)
        grx = np.array([to_goal_relative_x(float(x), goal_x=float(goal_x)) for x in xs])
        behind = (grx < 0.0) & (np.abs(ys - GOAL_Y) <= hw)
        per_example.append(grx[behind])
    return per_example


def _summarize_flip(base: np.ndarray, clamp: np.ndarray, per_example_grx: list, margin_m: float) -> dict:
    """Combine the seam delta with the reconstruction, asserting the N-1 row-set coherence."""
    base = np.asarray(base, dtype=float)
    clamp = np.asarray(clamp, dtype=float)
    changed = ~np.isclose(base, clamp, equal_nan=True)
    n = len(base)
    n_recon = np.array([len(a) for a in per_example_grx])

    # N-1: every training row whose box feature CHANGED under the clamp must have a reconstructed
    # behind-line point -- the ratio (xCross) or count (ghost) only moves when a behind-line box
    # player is removed. Subset, not equality: a ratio can cancel (equal offsetting removals leave
    # it unchanged), so a behind-line point may exist WITHOUT a change; a change without a point is
    # drift. Fail loud on drift.
    if not (set(np.flatnonzero(changed).tolist()) <= set(np.flatnonzero(n_recon > 0).tolist())):
        raise AssertionError(
            "N-1 coherence: a training row whose box feature changed under the clamp has NO "
            "reconstructed behind-line point -- the reconstruction has drifted from the extractor."
        )

    grx_all = np.concatenate(per_example_grx) if any(len(a) for a in per_example_grx) else np.array([], dtype=float)
    if len(grx_all):
        near_mask = (grx_all >= -margin_m) & (grx_all < 0.0)
        off_mask = grx_all < -margin_m
        hist = np.histogram(np.abs(grx_all), bins=[0.0, 1.0, 2.0, 5.0, 10.0, 105.0])[0].tolist()
        n_real_near, n_off = int(near_mask.sum()), int(off_mask.sum())
        real_near, offpitch = float(near_mask.mean()), float(off_mask.mean())
    else:
        hist, n_real_near, n_off, real_near, offpitch = [0, 0, 0, 0, 0], 0, 0, 0.0, 0.0
    # Counts are returned alongside the fractions so a corpus driver can aggregate across matches
    # (a mean-of-per-match-fractions is wrong when matches have different example/point counts).
    return {
        "changed_fraction": float(changed.mean()) if n else 0.0,
        "train_behind_line_base_rate": float((n_recon > 0).mean()) if n else 0.0,
        "real_near_line_fraction": real_near,
        "offpitch_fraction": offpitch,
        "dist_to_goal_hist": hist,
        "n_examples": int(n),
        "n_changed": int(changed.sum()),
        "n_rows_with_behind_line": int((n_recon > 0).sum()),
        "n_behind_line": int(n_recon.sum()),
        "n_real_near_line": n_real_near,
        "n_offpitch": n_off,
    }


def _ghost_flip(frames: pd.DataFrame, home_team_id, margin_m: float) -> dict:
    from silly_kicks.tracking import prepare_ghost_gk_training_data

    feats_base, _lab, meta = prepare_ghost_gk_training_data(frames, home_team_id=home_team_id, return_meta=True)
    with _scoped_gr_x_clamp():
        feats_clamp, _lab2, _meta2 = prepare_ghost_gk_training_data(frames, home_team_id=home_team_id, return_meta=True)
    per_example = _ghost_behind_line_grx_per_example(frames, meta)
    base = feats_base["attackers_in_box"].to_numpy()
    clamp = feats_clamp["attackers_in_box"].to_numpy()
    # Ghost stronger check: attackers_in_box is a COUNT, so the total delta equals the total number
    # of behind-line attackers removed. (A ratio -- xCross -- has no such identity; row-set only.)
    if int((base - clamp).sum()) != int(sum(len(a) for a in per_example)):
        raise AssertionError("N-1 count incoherence: sum(attackers_in_box delta) != reconstructed behind-line count")
    return _summarize_flip(base, clamp, per_example, margin_m)


def _xcross_flip(frames: pd.DataFrame, actions, home_team_id, margin_m: float) -> dict:
    """xCross: ONE ``return_meta=True`` call sources both decision inputs (owner elevation).

    ``changed_fraction`` = box_off_def_ratio (X) vs box_off_def_ratio_clamped (meta); real-vs-garbage
    = the same rows' ``behind_line_box_gr_x``. The clamped ratio is the extractor's own value with
    behind-line box players removed (proven identical to the scoped clamp), so the two legs are the
    same row-set by construction -- no independent reconstruction to drift.
    """
    from silly_kicks.tracking import prepare_xcross_training_data

    if actions is None:
        raise ValueError("xcross needs `actions` (cross opportunities); pass the match's SPADL actions")
    feats, _y, _g, meta = prepare_xcross_training_data(frames, actions, home_team_id=home_team_id, return_meta=True)
    if not len(feats):
        return _summarize_flip(np.array([]), np.array([]), [], margin_m)
    base = feats["box_off_def_ratio"].to_numpy(dtype=float)
    clamp = meta["box_off_def_ratio_clamped"].to_numpy(dtype=float)
    per_example = [np.asarray(a, dtype=float) for a in meta["behind_line_box_gr_x"]]
    return _summarize_flip(base, clamp, per_example, margin_m)


def measure_training_flip(
    frames: pd.DataFrame, actions, home_team_id, *, model: str, margin_m: float | None = None
) -> dict:
    """The Phase-B (basis A) measurement for one model on one match.

    ``changed_fraction`` is the tau input and comes from the SEAM (recompute the box feature under
    the shipped predicate and under ``_scoped_gr_x_clamp``). The descriptive fields describe the
    behind-line points via a reconstruction aligned to the extractor's examples (N-1).
    """
    if margin_m is None:
        margin_m = OFF_PITCH_MARGIN_M
    if model == "ghost":
        return _ghost_flip(frames, home_team_id, margin_m)
    if model == "xcross":
        return _xcross_flip(frames, actions, home_team_id, margin_m)
    raise ValueError(f"unknown model {model!r}; expected 'ghost' or 'xcross'")


def _load_home_team_id(data_dir: pathlib.Path, key: str):
    """Per-match home_team_id from the tc3 `_home/<key>.json` sidecar, or None."""
    p = data_dir / "_home" / f"{key}.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("home_team_id", data) if isinstance(data, dict) else data


_FLIP_COUNT_KEYS = (
    "n_examples",
    "n_changed",
    "n_rows_with_behind_line",
    "n_behind_line",
    "n_real_near_line",
    "n_offpitch",
)
_GEOM_KEYS = ("n_flipped", "n_flipped_band_only", "n_flipped_boundary_only", "n_flipped_both")
_MODELS = ("ghost", "xcross")
_N_HIST_BINS = 5  # _summarize_flip bins |gr_x| into [0,1,2,5,10,105] -> 5 counts

#: Bumped whenever the per-match shard columns change. Digested into the `for_each` generation token,
#: so a re-run against a stale generation with the OLD schema would be caught -- but the real
#: protection is the run-time assertion in `_measure_one_match`, which fails at the FIRST shard if the
#: built row keys diverge from `_EMITTED_SHARD_COLUMNS` (ADR-052; never `pd.DataFrame(rows, columns=)`,
#: which selects-to-declaration and hides both a dropped and a missing key).
_SHARD_SCHEMA_VERSION = "box-constant-delta-1"


def _emitted_shard_columns() -> tuple[str, ...]:
    cols = ["match_key", "geom_n_rows", "geom_n_behind_line", *(f"geom_{k}" for k in _GEOM_KEYS)]
    for m in _MODELS:
        cols.append(f"{m}__present")
        cols += [f"{m}__{k}" for k in _FLIP_COUNT_KEYS]
        cols += [f"{m}__hist{b}" for b in range(_N_HIST_BINS)]
    return tuple(cols)


_EMITTED_SHARD_COLUMNS = _emitted_shard_columns()


def _match_key(fp: pathlib.Path, data_dir: pathlib.Path) -> str:
    """An INJECTIVE `for_each` key from the full relative path.

    The shard layout's stem is already `provider__id` (unique), but the named `{provider}/{id}/
    frames.parquet` layout has stem `frames` for every match -- so keying on the stem alone would
    collide and `for_each._require_injective` would (correctly) raise. The whole relative path is
    unique in both layouts.
    """
    return "__".join(fp.relative_to(data_dir).with_suffix("").parts)


def _sidecar_key(fp: pathlib.Path) -> str:
    """The `_actions/`/`_home/` sidecar key: the stem, or the id dir in the named layout.

    Shards are `_actions/<provider>__<id>.parquet`; the named tc3 tree is `_actions/<id>.parquet`
    beside `<provider>/<id>/frames.parquet`, so a `frames` stem resolves to its parent (`<id>`).
    """
    return fp.parent.name if fp.stem == "frames" else fp.stem


def _measure_one_match(fp: pathlib.Path, data_dir: pathlib.Path, margin_m: float) -> pd.DataFrame:
    """One per-match shard row: the geometry classification plus the per-model training-flip counts.

    Counts, never fractions -- a mean of per-match fractions is wrong when matches differ in
    example/point counts, so the corpus fractions are computed from summed counts in
    `_aggregate_training_flip`.
    """
    frames = pd.read_parquet(fp)
    skey = _sidecar_key(fp)
    actions_p = data_dir / "_actions" / f"{skey}.parquet"
    actions = pd.read_parquet(actions_p) if actions_p.is_file() else None
    home = _load_home_team_id(data_dir, skey)

    gx = frames["x"].to_numpy(dtype=float)
    gy = frames["y"].to_numpy(dtype=float)
    row: dict[str, object] = {
        "match_key": _match_key(fp, data_dir),
        "geom_n_rows": int(gx.size),
        "geom_n_behind_line": int((gx < 0.0).sum()),
    }
    row.update({f"geom_{k}": int(v) for k, v in classify_flips(gx, gy).items()})
    for model in _MODELS:
        present = model == "ghost" or actions is not None  # xcross needs cross opportunities
        if present:
            res = measure_training_flip(frames, actions, home, model=model, margin_m=margin_m)
        else:
            res = {**{k: 0 for k in _FLIP_COUNT_KEYS}, "dist_to_goal_hist": [0] * _N_HIST_BINS}
        row[f"{model}__present"] = int(present)
        row.update({f"{model}__{k}": int(res[k]) for k in _FLIP_COUNT_KEYS})
        row.update({f"{model}__hist{b}": int(res["dist_to_goal_hist"][b]) for b in range(_N_HIST_BINS)})

    if set(row) != set(_EMITTED_SHARD_COLUMNS):  # keys the row ACTUALLY carries, not a select-to-declared
        raise AssertionError(
            f"shard schema drift: {sorted(set(row) ^ set(_EMITTED_SHARD_COLUMNS))} differ from "
            f"_EMITTED_SHARD_COLUMNS -- bump _SHARD_SCHEMA_VERSION with the change (ADR-052)."
        )
    return pd.DataFrame([row], columns=list(_EMITTED_SHARD_COLUMNS))


def _aggregate_geometry(combined: pd.DataFrame) -> dict:
    """Corpus geometry attribution: sum the per-match `classify_flips` counts (exact -- a per-row
    classification counted is additive across matches)."""
    out: dict[str, int] = {k: int(combined[f"geom_{k}"].sum()) for k in _GEOM_KEYS}
    out["n_rows"] = int(combined["geom_n_rows"].sum())
    out["n_behind_line"] = int(combined["geom_n_behind_line"].sum())
    return out


def _aggregate_training_flip(combined: pd.DataFrame) -> dict:
    """Corpus training-feature-delta, per model, from the summed per-match counts."""
    out: dict[str, dict] = {}
    for model in _MODELS:
        c = {k: int(combined[f"{model}__{k}"].sum()) for k in _FLIP_COUNT_KEYS}
        ne, nb = c["n_examples"], c["n_behind_line"]
        out[model] = {
            **c,
            "n_matches": int(combined[f"{model}__present"].sum()),
            "changed_fraction": c["n_changed"] / ne if ne else 0.0,
            "train_behind_line_base_rate": c["n_rows_with_behind_line"] / ne if ne else 0.0,
            "real_near_line_fraction": c["n_real_near_line"] / nb if nb else 0.0,
            "offpitch_fraction": c["n_offpitch"] / nb if nb else 0.0,
            "dist_to_goal_hist": [int(combined[f"{model}__hist{b}"].sum()) for b in range(_N_HIST_BINS)],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the ghost box-constant feature delta.")
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    paths = frame_parquets(args.data_dir)
    if not paths:
        raise SystemExit(
            f"no *.parquet under {args.data_dir}. Run materialize_tc3_frames first, and point "
            f"--data-dir at its --out (the shard generation directory beneath it is searched too)."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    margin_m = OFF_PITCH_MARGIN_M
    # ADR-052: this pass runs the ghost + xcross trainers per match (minutes each), so it writes one
    # shard per match through `for_each` -- a crash resumes, an existing shard is skipped, and each
    # item prints progress -- rather than holding every result in memory and writing once at the end.
    res = for_each(
        paths,
        key=lambda fp: _match_key(fp, args.data_dir),
        work=lambda fp: _measure_one_match(fp, args.data_dir, margin_m),
        shard_root=args.out / "_shards",  # underscore-prefixed so `frame_parquets` never re-reads it
        token_inputs={"schema": _SHARD_SCHEMA_VERSION, "driver": "box-constant-delta", "margin_m": margin_m},
        label="match",
    )
    combined = reconcile(res.shard_dir, args.out / "box_constant_delta.parquet", tag="all")
    if not len(combined):
        raise SystemExit("every shard was empty -- the corpus yielded no measurable frames.")

    out: dict[str, object] = {}
    out.update(_aggregate_geometry(combined))
    # Phase B (basis A): the training-feature-delta decision inputs, per trained model.
    out["training_flip"] = _aggregate_training_flip(combined)
    out["off_pitch_margin_m"] = margin_m
    out.update(res.manifest())
    # `git_provenance()` returns a DICT, not an object -- `prov.commit` raises AttributeError. pyright
    # caught that once; the unit tests could not, because only `test_main_...` exercises `main()`.
    out["run_commit"] = prov["commit"]
    out["run_tree_dirty"] = prov["dirty"]

    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
