"""Fit the TF-46 Arm 1 corner-offence models on the owner-tier corpus.

Extracts, per corner, the attacking players' reflected initial (t_kick-2s) and target (min(first
contact+1s, t_kick+2s)) positions from tracking frames (TF-79 for first contact), fits the RunZoneModel
GMMs over the pooled snapshots, encodes each corner's run vector, then fits the CornerRoutineModel NMF
basis. Writes the two params-only npz artifacts + a metrics.json fit summary.

Clean-tree guard runs FIRST (ADR-037); the input contract declares the symbols the fit depends on
(ADR-056). Aggregate-only outputs under the reversibility policy -- no reversible ids committed. The
real corpus run is opt-in / e2e (run on the DGX or media-PC); tests/scripts cover the pure kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_SHARD_SCHEMA_VERSION = "corner-snapshots-1"
_EMITTED_SHARD_COLUMNS = ("game_id", "corner_id", "player_id", "init_x", "init_y", "target_x", "target_y")


def _snapshot_rows(item) -> pd.DataFrame:
    """Per-(corner, attacking player) reflected initial + target positions for one match."""
    from silly_kicks.corners import add_corner_windows, identify_corners
    from silly_kicks.corners._encode import _nearest_frame_xy, _reflect_y
    from silly_kicks.tracking._ball_contact import detect_ball_contacts

    actions, frames = item.actions, item.frames
    if frames is None or frames.empty:
        return pd.DataFrame(columns=list(_EMITTED_SHARD_COLUMNS))
    corners = identify_corners(actions, provider=item.provider)
    corners = corners[corners["corner_kind"] == "direct"]
    if corners.empty:
        return pd.DataFrame(columns=list(_EMITTED_SHARD_COLUMNS))
    contacts = detect_ball_contacts(frames)
    windowed = add_corner_windows(corners, frames, contacts)

    rows: list[dict] = []
    # SB360 (statsbomb) is freeze-frame only: its initial snapshot is the KICK freeze frame (t_kick),
    # not the -2s continuous snapshot (spec section 7; the -2s-vs-kick divergence is measured in section 12.7).
    for _, c in windowed.iterrows():
        period, team, side = int(c["period_id"]), c["attacking_team_id"], str(c["corner_side"])
        init_t = float(c["t_kick"]) if item.provider == "statsbomb" else float(c["t_init"])
        init = _nearest_frame_xy(frames, period, init_t, team)
        target = _nearest_frame_xy(frames, period, float(c["t_target"]), team)
        if init is None or target is None:
            continue
        init, target = _reflect_y(init, side), _reflect_y(target, side)
        merged = init.merge(target, on="player_id", suffixes=("_i", "_t"))
        for _, p in merged.iterrows():
            rows.append(
                {
                    "game_id": c["game_id"],
                    "corner_id": int(c["corner_id"]),
                    "player_id": p["player_id"],
                    "init_x": float(p["x_i"]),
                    "init_y": float(p["y_i"]),
                    "target_x": float(p["x_t"]),
                    "target_y": float(p["y_t"]),
                }
            )
    if rows and tuple(rows[0]) != _EMITTED_SHARD_COLUMNS:
        raise AssertionError(
            f"shard columns drifted from _EMITTED_SHARD_COLUMNS; bump _SHARD_SCHEMA_VERSION "
            f"(currently {_SHARD_SCHEMA_VERSION!r}). emitted={tuple(rows[0])}"
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(_EMITTED_SHARD_COLUMNS))


def _combine_shards(res) -> pd.DataFrame:
    from scripts._driver import shard_path

    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys) if not f.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(_EMITTED_SHARD_COLUMNS))


def fit_models(snapshots: pd.DataFrame, out_dir: Path) -> dict:
    """Fit RunZoneModel + CornerRoutineModel from the pooled per-player snapshots; save both. Pure-ish
    (writes two bundle dirs under ``out_dir``). Returns a JSON-safe fit summary."""
    from silly_kicks.corners import CornerRoutineModel, RunZoneModel
    from silly_kicks.corners._encode import _run_column_names

    init_xy = snapshots[["init_x", "init_y"]].to_numpy(dtype=float)
    target_xy = snapshots[["target_x", "target_y"]].to_numpy(dtype=float)
    zone_model = RunZoneModel().fit(init_xy, target_xy)
    zone_model.save(out_dir / "run_zone")

    # per-corner term matrix: encode each corner's aligned players through the fitted zone model
    run_cols = _run_column_names(zone_model)
    term_rows = []
    for (_g, _c), grp in snapshots.groupby(["game_id", "corner_id"], sort=False):
        vec = zone_model.encode_runs(
            grp[["init_x", "init_y"]].to_numpy(dtype=float),
            grp[["target_x", "target_y"]].to_numpy(dtype=float),
            active_mask=np.ones(len(grp), dtype=bool),
        )
        term_rows.append(vec)
    term_matrix = np.vstack(term_rows) if term_rows else np.zeros((0, len(run_cols)))
    routine_model = CornerRoutineModel().fit(term_matrix)
    routine_model.save(out_dir / "routine")

    return {
        "n_corners": int(snapshots[["game_id", "corner_id"]].drop_duplicates().shape[0]),
        "n_player_snapshots": len(snapshots),
        "run_zone": zone_model.selected,
        "routine": routine_model.selected,
        "n_runs": len(run_cols),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--cache-dir", default=None)
    # Multi-provider pooled fit (spec D4): continuous tracking (skillcorner + idsse, the DFL/Sportec
    # optical source served under the "idsse" pining key) + SB360 freeze (statsbomb). "sportec" is NOT
    # a separate pining provider -- it is the same data as "idsse" and 404s the listing; Metrica is
    # excluded (3 ancient demo games). The library still accepts any provider passed explicitly.
    ap.add_argument("--providers", nargs="*", default=["skillcorner", "idsse", "statsbomb"])
    ap.add_argument("--match-ids-json", default=None, help='per-provider pins: JSON {"<provider>": ["<id>", ...]}')
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    from scripts._driver import for_each
    from scripts._input_contract import declare_inputs
    from scripts._loader_pining import pining_source
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)

    refs, load = pining_source(
        providers=args.providers,
        match_ids=match_ids,
        max_per_provider=args.max_matches,
        cache_dir=args.cache_dir,
        token=args.token,
        with_velocity=False,  # corners read x/y only; skip the ~32s/match velocity derivation
    )
    res = for_each(
        refs,
        key=lambda r: r.key,
        load=load,
        work=_snapshot_rows,
        shard_root=dest / "shards",
        token_inputs={"metric": "corner_snapshots", "schema": _SHARD_SCHEMA_VERSION},
        label="match",
    )
    snapshots = _combine_shards(res)
    summary = fit_models(snapshots, dest) if not snapshots.empty else {"n_corners": 0}

    contract = declare_inputs(
        driver="build_corner_routines",
        emitted_shard_columns=list(_EMITTED_SHARD_COLUMNS),
        schema_version=_SHARD_SCHEMA_VERSION,
        models=["RunZoneModel", "CornerRoutineModel"],
    )
    report = {
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "input_contract": contract,
        "verdicts": summary,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
