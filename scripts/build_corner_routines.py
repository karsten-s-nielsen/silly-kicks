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
    from silly_kicks.tracking._provider_visibility import classified_provider

    actions, frames = item.actions, item.frames
    if frames is None or frames.empty:
        return pd.DataFrame(columns=list(_EMITTED_SHARD_COLUMNS))
    corners = identify_corners(actions, provider=item.provider)
    corners = corners[corners["corner_kind"] == "direct"]
    if corners.empty:
        return pd.DataFrame(columns=list(_EMITTED_SHARD_COLUMNS))
    contacts = detect_ball_contacts(frames, provider=classified_provider(item.provider))
    windowed = add_corner_windows(corners, frames, contacts)

    from silly_kicks.tracking._gk_resolve import resolve_defended_goals

    goal_map = resolve_defended_goals(frames)  # ADR-055: built once per match, threaded in (F0)
    rows: list[dict] = []
    # SB360 (statsbomb) is freeze-frame only: its initial snapshot is the KICK freeze frame (t_kick),
    # not the -2s continuous snapshot (spec section 7; the -2s-vs-kick divergence is measured in section 12.7).
    for _, c in windowed.iterrows():
        period, team, side = int(c["period_id"]), c["attacking_team_id"], str(c["corner_side"])
        init_t = float(c["t_kick"]) if item.provider == "statsbomb" else float(c["t_init"])
        init = _nearest_frame_xy(frames, period, init_t, team, goal_map=goal_map)
        target = _nearest_frame_xy(frames, period, float(c["t_target"]), team, goal_map=goal_map)
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


#: TF-46 Arm 1 Part 2 run (16) decision bar (section 18.1 amendment, C-04): refit without the course tier only
#: when the leave-one-tier-out zone-assignment agreement is >= this for BOTH corner models.
_TIER_AGREEMENT_BAR = 0.95


def zone_assignment_agreement(snapshots: pd.DataFrame, model_a, model_b) -> float:
    """Share of player snapshots whose (initial, target) zone argmax is identical under two RunZoneModels.

    The leave-one-tier-out stability statistic for run (16): a refit that reassigns few snapshots agrees with
    the full fit. NaN when there are no snapshots."""
    if snapshots.empty:
        return float("nan")
    from silly_kicks.corners._zones import _gaussian_responsibilities

    init = snapshots[["init_x", "init_y"]].to_numpy(dtype=float)
    target = snapshots[["target_x", "target_y"]].to_numpy(dtype=float)

    def _zones(m):
        ri = _gaussian_responsibilities(init, m.init_means, m.init_covs, m.init_weights).argmax(axis=1)
        rt = _gaussian_responsibilities(target, m.target_means, m.target_covs, m.target_weights).argmax(axis=1)
        return ri, rt

    ai, at = _zones(model_a)
    bi, bt = _zones(model_b)
    return float(((ai == bi) & (at == bt)).mean())


def refit_without_course_decision(
    zone_agreement: float, routine_agreement: float, *, bar: float = _TIER_AGREEMENT_BAR
) -> bool:
    """Run (16) pre-registered rule: refit without the course tier only when BOTH models' leave-one-tier-out
    agreement clears ``bar`` (0.95). A low agreement on either model keeps the course tier in."""
    return bool(zone_agreement >= bar and routine_agreement >= bar)


def leave_one_tier_out(snapshots: pd.DataFrame, tier: pd.Series, out_dir: Path, *, training_commit: str | None) -> dict:
    """Per left-out tier, the zone-assignment agreement vs the full fit for both models (run 16)."""
    from silly_kicks.corners import RunZoneModel
    from silly_kicks.corners._encode import _run_column_names

    full_zone = RunZoneModel().fit(
        snapshots[["init_x", "init_y"]].to_numpy(float), snapshots[["target_x", "target_y"]].to_numpy(float)
    )
    report: dict[str, dict] = {}
    for t in sorted(tier.dropna().unique()):
        kept = snapshots[tier.to_numpy() != t]
        if kept.empty:
            continue
        loo_zone = RunZoneModel().fit(
            kept[["init_x", "init_y"]].to_numpy(float), kept[["target_x", "target_y"]].to_numpy(float)
        )
        z_agree = zone_assignment_agreement(snapshots, full_zone, loo_zone)
        # routine agreement: the NMF dominant-basis assignment per corner under each term matrix
        r_agree = _routine_agreement(snapshots, kept, full_zone, loo_zone, _run_column_names(full_zone))
        report[str(t)] = {
            "zone_agreement": z_agree,
            "routine_agreement": r_agree,
            "refit_without_tier": refit_without_course_decision(z_agree, r_agree),
        }
    return report


def _routine_agreement(snapshots, kept, full_zone, loo_zone, run_cols) -> float:
    """Per-corner dominant-routine agreement between the full and leave-one-out NMF fits (run 16)."""
    from silly_kicks.corners import CornerRoutineModel

    def _terms(zone):
        rows = []
        for (_g, _c), grp in snapshots.groupby(["game_id", "corner_id"], sort=True):
            rows.append(
                zone.encode_runs(
                    grp[["init_x", "init_y"]].to_numpy(float), grp[["target_x", "target_y"]].to_numpy(float)
                )
            )
        return np.vstack(rows) if rows else np.zeros((0, len(run_cols)))

    tm_full, tm_loo = _terms(full_zone), _terms(loo_zone)
    if not len(tm_full):
        return float("nan")
    rm_full = CornerRoutineModel().fit(tm_full)
    rm_loo = CornerRoutineModel().fit(tm_loo)
    dom_full = rm_full.transform(tm_full).argmax(axis=1)
    dom_loo = rm_loo.transform(tm_loo).argmax(axis=1)
    return float((dom_full == dom_loo).mean())


def _extra_tier_paths(match_dir: Path) -> dict[str, str]:
    """The raw SkillCorner artifacts of ONE course-tier match directory: ``match.json``, the GI dynamic-events
    CSV and the tracking file (``.jsonl`` / ``.jsonl.gz`` / ``.json`` / tracking ``.parquet``)."""
    meta = match_dir / "match.json"
    events = sorted(p for p in match_dir.iterdir() if p.is_file() and "event" in p.name and p.suffix == ".csv")
    tracking = sorted(p for p in match_dir.iterdir() if p.is_file() and "tracking" in p.name)
    if not meta.is_file() or len(events) != 1 or len(tracking) != 1:
        raise ValueError(
            f"{match_dir}: a course-tier match needs exactly match.json + one *event*.csv + one *tracking* file "
            f"(found events={[p.name for p in events]}, tracking={[p.name for p in tracking]})"
        )
    return {"metadata": str(meta), "events": str(events[0]), "tracking": str(tracking[0])}


def load_extra_tier(extra_dir: Path, revision: str) -> tuple[list, dict]:
    """Convert the raw SkillCorner GI + tracking files of each course-tier match in ``extra_dir`` IN-RUN through
    the library (C-12): ``spadl.skillcorner.convert_to_actions`` + the native ``tracking.skillcorner`` builder,
    the SAME path the pining loader uses. Layout: one sub-directory per match, named by the match id.

    Refuses pre-converted input (a parquet file that is not a raw tracking table): the course tier must convert
    through the library, not be injected. Returns the loaded matches (``LoadedMatch``, provider ``course``) and
    a sidecar (revision, file-list hash, data profile, match count) so the run records provenance."""
    import hashlib

    from scripts._driver_sources import LoadedMatch
    from scripts._loader_pining import _build_skillcorner

    if not revision:
        raise ValueError("--extra-tier-dir requires --extra-tier-revision")
    files = sorted(p for p in Path(extra_dir).rglob("*") if p.is_file())
    if any(p.suffix == ".parquet" and "tracking" not in p.name for p in files):
        raise ValueError(
            "--extra-tier-dir must hold RAW GI + tracking files; a directory of parquet actions is refused"
        )
    file_hash = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    items = []
    for match_dir in sorted(p for p in Path(extra_dir).iterdir() if p.is_dir()):
        actions, frames, home_team_id, report = _build_skillcorner(
            _extra_tier_paths(match_dir), match_dir.name, None, with_velocity=False
        )
        items.append(LoadedMatch("course", match_dir.name, actions, frames, home_team_id, None, report))
    sidecar = {
        "extra_tier_revision": revision,
        "file_list_sha256": file_hash,
        "n_files": len(files),
        "n_matches": len(items),
        "data_profile": "course",
    }
    return items, sidecar


def fit_models(snapshots: pd.DataFrame, out_dir: Path, *, training_commit: str | None = None) -> dict:
    """Fit RunZoneModel + CornerRoutineModel from the pooled per-player snapshots; save both. Pure-ish
    (writes two bundle dirs under ``out_dir``). Returns a JSON-safe fit summary.

    ``training_commit`` stamps both bundles' metadata so a shipped artifact traces back to a commit
    (ADR-011 provenance; mirrors the tracking trainers). The corpus run passes the clean-HEAD commit."""
    from silly_kicks.corners import CornerRoutineModel, RunZoneModel
    from silly_kicks.corners._encode import _run_column_names

    init_xy = snapshots[["init_x", "init_y"]].to_numpy(dtype=float)
    target_xy = snapshots[["target_x", "target_y"]].to_numpy(dtype=float)
    zone_model = RunZoneModel().fit(init_xy, target_xy)
    zone_model.training_commit = training_commit
    zone_model.save(out_dir / "run_zone")

    # per-corner term matrix: encode each corner's aligned players through the fitted zone model
    run_cols = _run_column_names(zone_model)
    term_rows = []
    for (_g, _c), grp in snapshots.groupby(["game_id", "corner_id"], sort=False):
        # Occupancy default (no active_mask): only players whose target is in the danger+approach region
        # contribute to the term matrix -- stay-back/short players are excluded (spec section 6, 2026-10-18).
        vec = zone_model.encode_runs(
            grp[["init_x", "init_y"]].to_numpy(dtype=float),
            grp[["target_x", "target_y"]].to_numpy(dtype=float),
        )
        term_rows.append(vec)
    term_matrix = np.vstack(term_rows) if term_rows else np.zeros((0, len(run_cols)))
    routine_model = CornerRoutineModel().fit(term_matrix)
    routine_model.training_commit = training_commit
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
    # TF-46 Arm 1 Part 2 run (16) stability (section 18.1 amendment; committed in C2, run in C4 -- B-01 / CL-7).
    ap.add_argument(
        "--leave-one-tier-out",
        action="store_true",
        help="report per-tier zone/routine-assignment agreement vs the full fit and the refit decision (run 16)",
    )
    ap.add_argument(
        "--extra-tier-dir", default=None, help="RAW SkillCorner GI + tracking dir for the course tier (C-12)"
    )
    ap.add_argument(
        "--extra-tier-revision", default=None, help="REQUIRED with --extra-tier-dir: the course-tier revision"
    )
    ap.add_argument(
        "--owner-ids-json", default=None, help="owner-tier fixtures for the course-tier overlap de-dup (C-04)"
    )
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
    g2p = {str(r.match_id): r.provider for r in refs}
    tier = snapshots["game_id"].astype(str).map(g2p)

    # run (16): the course tier always ENTERS the fit (spec section 18.1 amendment, C-04), de-duplicated against
    # the owner tier first (a course match the owner tier already holds would be double-counted).
    extra_sidecar = None
    if args.extra_tier_dir:
        from scripts._tf46p2_common import dedup_course_tier

        if not args.owner_ids_json:
            raise SystemExit("--extra-tier-dir requires --owner-ids-json (the course tier must be de-duplicated)")
        items, extra_sidecar = load_extra_tier(Path(args.extra_tier_dir), args.extra_tier_revision)
        raw = json.loads(Path(args.owner_ids_json).read_text(encoding="utf-8"))
        kept, overlap = dedup_course_tier(pd.DataFrame(raw.get("course", [])), pd.DataFrame(raw.get("owner", [])))
        kept_ids = set(kept["match_id"].astype(str)) if "match_id" in kept.columns else set()
        course_parts = [_snapshot_rows(it) for it in items if str(it.match_id) in kept_ids]
        course = pd.concat(course_parts, ignore_index=True) if course_parts else snapshots.iloc[:0]
        extra_sidecar["overlap_with_owner"] = overlap
        extra_sidecar["n_matches_kept"] = len(course_parts)
        snapshots = pd.concat([snapshots, course], ignore_index=True)
        tier = pd.concat([tier, pd.Series(["course"] * len(course))], ignore_index=True)
    summary = fit_models(snapshots, dest, training_commit=prov["commit"]) if not snapshots.empty else {"n_corners": 0}

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
    # leave-one-tier-out reports the stability + the refit decision, but does not itself refit (CL-8).
    if extra_sidecar is not None:
        report["extra_tier"] = extra_sidecar
    if args.leave_one_tier_out and not snapshots.empty:
        report["leave_one_tier_out"] = leave_one_tier_out(snapshots, tier, dest, training_commit=prov["commit"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
