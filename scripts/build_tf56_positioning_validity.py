"""Maintainer driver: TF-56 prescriptive positioning-gap construct-validity battery (AMENDED 2026-09-22).

Owner-run corpus pass (reported-not-gated for applied numbers; the metric's COLUMN ship-status IS
gated on the composite verdict this produces). The original dose/responsiveness battery was RETIRED
(a commit-1 spike MEASURED positioning_gap non-monotonic in any dose -- it re-anchors to each
defender's reachable set -- and orthogonal to concurrent shape-badness). Per SCORED defensive frame it
measures the positioning gap under the shipped ThreatObjective and PAIRS it with the corpus-derived
conceded threat over the next window (the attacking team's realized xT progression in [t, t+dt); no
external xG). The pooled reduce computes:

* PREDICTIVE (the ship gate, ``predictive_verdict``): corr(positioning_gap_t, conceded_threat_{t+dt})
  positive AND significant -- the meaningful construct test, precisely because the gap is orthogonal
  to concurrent threat. Reported at two windows (5 s, 10 s); the primary gate is the 10 s window.
* DISCRIMINATION (``discrimination_verdict``): the gap is non-degenerate across (game_id, team_id)
  units (not all ~0 / noise).
* NON-DEGENERACY: the fraction of scored frames whose SA found a feasible proposal, + the converged rate.

Optimizer STABILITY (the other instrument-validity leg) is a CHEAP FIXTURE property (it needs no
corpus), gated by tests/positioning/test_probe.py::optimizer_stability at the shipped init_sigma_m=2.0,
not recomputed here.

**GO** (predictive AND discriminating AND non-degenerate) -> the glossaried column stays. **NO-GO**
-> the metric is DEMOTED in the same release (territorial_defense precedent); this driver only reports
the verdict, it does not edit code.

The corpus map is ``for_each`` (ADR-052: per-match shards, resumable, conserving); the pooled predictive
/ discrimination / non-degeneracy census is computed in a REDUCE over ALL shards, never per shard.
Frame-level correctness is validated by the OWNER RUN; the unit tests pin the pooled reduce, the verdict
discrimination and the schema (``tests/scripts/test_build_tf56_positioning_validity.py``).

silly-kicks ships no xT model, so ``xt`` is fit WITHIN the corpus (a first streaming pass over the
matches' SPADL actions) and injected -- the gap is a RELATIVE measure and the conceded-threat leg is a
within-corpus xT progression, so a within-corpus xT is appropriate for a construct-validity battery. The
SA scoring dominates cost, so the actions-fit pass is negligible beside it.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/build_tf56_positioning_validity.py --out <DIR> [--providers skillcorner,idsse,gradientsports] \\
      [--max-per-provider N] [--tracking-limit N] [--sa-iterations N] [--match-ids-json FILE | --list-matches]

For a PARALLEL run, xT MUST be fit ONCE and SHARED across workers -- a per-slice refit would score
each shard under a DIFFERENT surface, making the pooled predictive correlation incoherent:
  1. FIT step: ``--fit-only --xt-path XT.json`` (NO --match-ids-json -> full corpus actions,
     tracking_limit=0, cheap) fits the corpus xT once and saves it.
  2. WORKERS: split ``--match-ids-json`` across N processes sharing one ``--out``, each with
     ``--xt-path XT.json`` (LOADS the shared xT; writes shards + a per-worker manifest, NO verdict).
  3. REDUCE: run ONCE more UNPARTITIONED over the same ``--out`` with ``--xt-path XT.json`` to produce
     the authoritative ``metrics.json`` (resumes existing shards, reduces over ALL).
A SERIAL run (no ``--xt-path``) fits xT in-corpus and scores in one process -- coherent but unparallelised.
``--out docs/research/tf56_positioning/`` puts the cited ``metrics.json`` where the ADR-056 staleness
detector reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._input_contract import declare_inputs
from silly_kicks.positioning._probe import (
    MIN_DOMAIN_FRAMES,
    PREDICTIVE_ALPHA,
    discrimination_verdict,
    predictive_verdict,
)

_FRAME_KEYS = ["game_id", "period_id", "frame_id"]

#: Default provider tokens = the pining API + cache keys. ``idsse`` is the DFL/Sportec key -- NOT
#: "sportec" (that 404s at list-matches AND makes providers_for_slice() drop an idsse worker slice).
_DEFAULT_PROVIDERS = "skillcorner,idsse,gradientsports"
#: Conceded-threat windows (seconds) paired with each scored gap; the primary predictive gate is 10 s.
_CONCEDED_WINDOWS_S: tuple[float, float] = (5.0, 10.0)
#: Discrimination floor: the gap must be non-constant across at least this many (game, team) units.
_MIN_UNITS = 2

#: The columns a per-frame shard carries. Bump _SHARD_SCHEMA_VERSION (and it is referenced in the
#: token_inputs) whenever this changes -- the 4.77.1 stale-shard trap.
_SHARD_COLUMNS = [
    "game_id",
    "period_id",
    "frame_id",
    "team_id",
    "positioning_gap",
    "conceded_threat_5s",
    "conceded_threat_10s",
    "n_feasible_proposals",
    "sa_converged",
]
_SHARD_SCHEMA_VERSION = "tf56-positioning-2"


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056)."""
    from dataclasses import asdict

    from silly_kicks.positioning import PositioningParams

    return declare_inputs(
        driver="build_tf56_positioning_validity",
        params={
            "positioning": asdict(PositioningParams()),
            "min_domain_frames": MIN_DOMAIN_FRAMES,
            "predictive_alpha": PREDICTIVE_ALPHA,
            "conceded_windows_s": list(_CONCEDED_WINDOWS_S),
            "min_units": _MIN_UNITS,
            "schema": _SHARD_SCHEMA_VERSION,
        },
        extractors=(
            "silly_kicks.positioning._compute",
            "silly_kicks.positioning._probe",
            "silly_kicks.positioning._solve",
        ),
        models=(),
    )


def _nanmed(a: np.ndarray) -> float:
    return float(np.nanmedian(a)) if a.size and bool(np.isfinite(a).any()) else float("nan")


def _xt_fingerprint(xt) -> str:
    """A stable 16-hex digest of the FITTED xT surface -- keys the shard cache on the surface.

    ``for_each`` shards are resumable across runs; without this, a re-fit over a different corpus
    composition + a resume would silently POOL shards scored under two different xT surfaces -- the
    exact incoherence ``--xt-path`` prevents WITHIN a run, reopened ACROSS runs. Feeding it into
    ``token_inputs`` makes a changed surface a NEW shard generation, so stale shards are never reused.
    """
    return hashlib.sha256(np.ascontiguousarray(np.asarray(xt.xT, dtype=float)).tobytes()).hexdigest()[:16]


def _conceded_threat(actions, xt, *, period_id, t0: float, window_s: float, defending_team_id) -> float:
    """Attacking-team xT progression conceded in the STRICTLY-FUTURE ``(t0, t0 + window_s)`` of ``period_id``.

    The window is strictly after the frame instant ``t0`` (the positioning decision) -- an action AT
    ``t0`` is simultaneous with the decision, not conceded by it. The attacking team is everyone NOT the
    defending team (the scored sample's team_id). Sums the positive per-action ``xt.rate`` (xT added by
    a move) over the window; NaN ratings (non-move actions) are dropped. Empty actions / no in-window
    attack -> 0.0 (a real "nothing conceded"); a non-finite ``t0`` -> NaN (frame time unknown).
    """
    if actions is None or not len(actions) or "time_seconds" not in actions.columns:
        return 0.0
    if not np.isfinite(t0):
        return float("nan")
    from silly_kicks.id_compat import ids_match

    t = actions["time_seconds"].to_numpy(dtype=float)
    same_period = actions["period_id"].to_numpy() == period_id
    in_window = same_period & (t > float(t0)) & (t < float(t0) + float(window_s))
    if not bool(in_window.any()):
        return 0.0
    attacking = in_window & (~ids_match(actions["team_id"], defending_team_id).to_numpy())
    if not bool(attacking.any()):
        return 0.0
    vals = np.asarray(xt.rate(actions.loc[attacking]), dtype=float)
    pos = vals[np.isfinite(vals) & (vals > 0.0)]
    return float(pos.sum()) if pos.size else 0.0


def _measure_match(item, *, xt, params) -> tuple[pd.DataFrame, dict]:
    """One per-scored-frame (gap, conceded_threat_{t+dt}) shard for a match + conservation counts.

    ``load_matches`` yields ``(provider, match_id, actions, frames, home_team_id)``. Scores the match
    once via ``compute_positioning_gap`` (the threat objective, injected ``xt``), then per scored frame
    derives the conceded threat over each window from the SAME match's SPADL actions. Returns
    ``(shard, counts)`` where ``counts`` carries the PositioningReport totals so the driver conserves.
    """
    from silly_kicks.positioning import compute_positioning_gap

    _provider, _match_id, actions, frames, _home = item

    samples, report = compute_positioning_gap(frames, xt=xt, params=params)
    counts = {
        "n_frames_in": report.n_frames_in,
        "n_frames_scored": report.n_frames_scored,
        "n_frames_dropped": sum(report.drop_reasons.values()),
        "n_matches": 1,
    }
    scored = samples[samples["positioning_gap_source"] == "scored"]
    if not len(scored):
        return pd.DataFrame(columns=_SHARD_COLUMNS), counts

    # ADR-068: group the frames ONCE (never a per-scored-frame boolean-mask rescan); the `_compute.py`
    # groupby idiom, not `group_rows` (a scripts driver need not enroll in the ADR-073 SCALE_GUARDED
    # registry, and the SA scoring dominates cost ~1000:1 regardless).
    frame_by_key = {k: g for k, g in frames.groupby(_FRAME_KEYS, sort=False)}
    rows: list[dict] = []
    for _, srow in scored.iterrows():
        g, p, f, team = srow["game_id"], srow["period_id"], srow["frame_id"], srow["team_id"]
        frame = frame_by_key.get((g, p, f))
        if frame is None:  # every scored key is present by construction; defensive narrowing
            continue
        t0 = float(frame["time_seconds"].iloc[0]) if "time_seconds" in frame.columns else float("nan")
        c5 = _conceded_threat(actions, xt, period_id=p, t0=t0, window_s=_CONCEDED_WINDOWS_S[0], defending_team_id=team)
        c10 = _conceded_threat(actions, xt, period_id=p, t0=t0, window_s=_CONCEDED_WINDOWS_S[1], defending_team_id=team)
        rows.append(
            {
                "game_id": g,
                "period_id": p,
                "frame_id": f,
                "team_id": team,
                "positioning_gap": float(srow["positioning_gap"]),
                "conceded_threat_5s": c5,
                "conceded_threat_10s": c10,
                "n_feasible_proposals": srow["n_feasible_proposals"],
                "sa_converged": srow["sa_converged"],
            }
        )
    shard = pd.DataFrame(rows, columns=_SHARD_COLUMNS)
    from silly_kicks.id_compat import canonical_id_series

    for _idc in ("game_id", "team_id"):
        shard[_idc] = canonical_id_series(shard[_idc])
    return shard, counts


def pool_positioning_shards(shards: list[pd.DataFrame]) -> dict:
    """Concatenate per-frame shards -> the POOLED-corpus statistics the verdicts consume.

    Returns ``{gap, conceded_5s, conceded_10s, n_domain, gap_median, n_feasible_fraction,
    converged_fraction, n_units, unit_gaps, unit_gap_std}`` -- the paired conceded-threat legs feed the
    predictive verdict, the per-(game, team) unit means feed discrimination.
    """
    if not shards:
        return {}
    df = pd.concat(shards, ignore_index=True)
    gap = df["positioning_gap"].to_numpy(dtype=float)
    c5 = df["conceded_threat_5s"].to_numpy(dtype=float)
    c10 = df["conceded_threat_10s"].to_numpy(dtype=float)
    feas = df["n_feasible_proposals"].to_numpy(dtype=float)
    conv = df["sa_converged"].astype("boolean")
    unit = df.groupby(["game_id", "team_id"], dropna=False)["positioning_gap"].mean()
    return {
        "gap": gap,
        "conceded_5s": c5,
        "conceded_10s": c10,
        "gap_median": _nanmed(gap),
        "n_domain": len(df),
        "n_feasible_fraction": float(np.nanmean(feas > 0)) if feas.size else float("nan"),
        "converged_fraction": float(conv.mean()) if len(conv) else float("nan"),
        "n_units": int(unit.shape[0]),
        "unit_gaps": unit.to_numpy(dtype=float),
        "unit_gap_std": float(np.nanstd(unit.to_numpy(dtype=float))) if unit.shape[0] else float("nan"),
    }


def reduce_positioning_verdicts(pooled: dict) -> dict:
    """The pooled PREDICTIVE + DISCRIMINATION + NON-DEGENERACY verdicts + the GO/NO-GO composite.

    GO = predictive (10 s window, positive AND significant) AND discriminating AND non-degenerate.
    ``arm_unscoreable`` (thin domain) propagates to the composite as ``no_go``. Optimizer STABILITY is
    the fixture instrument (tests/positioning/test_probe.py), not a corpus statistic.
    """
    if not pooled or pooled.get("n_domain", 0) == 0:
        return {"composite": "no_go", "reason": "empty corpus", "n_domain": 0}

    n_domain = pooled["n_domain"]
    pred5 = predictive_verdict(pooled["gap"], pooled["conceded_5s"], n_min=MIN_DOMAIN_FRAMES, alpha=PREDICTIVE_ALPHA)
    pred10 = predictive_verdict(pooled["gap"], pooled["conceded_10s"], n_min=MIN_DOMAIN_FRAMES, alpha=PREDICTIVE_ALPHA)
    discriminating = bool(
        discrimination_verdict(pooled["unit_gaps"]) == "discriminating"
        and pooled["n_units"] >= _MIN_UNITS
        and np.isfinite(pooled["gap_median"])
        and pooled["gap_median"] > 0.0
    )
    non_degenerate = bool(np.isfinite(pooled["n_feasible_fraction"]) and pooled["n_feasible_fraction"] > 0.0)
    predictive = pred10["verdict"] == "predictive"  # primary gate = 10 s window
    go = predictive and discriminating and non_degenerate
    return {
        "predictive": pred10["verdict"],
        "predictive_5s": pred5["verdict"],
        "predictive_r": pred10["r"],
        "predictive_p": pred10["p"],
        "predictive_n": pred10["n"],
        "predictive_5s_r": pred5["r"],
        "predictive_5s_p": pred5["p"],
        "discriminating": discriminating,
        "non_degenerate": non_degenerate,
        "composite": "go" if go else "no_go",
        "n_domain": n_domain,
        "gap_median": pooled["gap_median"],
        "n_units": pooled["n_units"],
        "unit_gap_std": pooled["unit_gap_std"],
        "n_feasible_fraction": pooled["n_feasible_fraction"],
        "converged_fraction": pooled["converged_fraction"],
        "note_optimizer_stability": (
            "fixture instrument (tests/positioning/test_probe.py::optimizer_stability); not a corpus statistic"
        ),
    }


def _fit_corpus_xt(items):
    """Fit a within-corpus ExpectedThreat over the matches' pooled SPADL actions (first pass)."""
    from silly_kicks.xthreat import ExpectedThreat

    frames_of_actions = [actions for (_p, _m, actions, _f, _h) in items]
    if not frames_of_actions:
        raise SystemExit("no matches loaded to fit the corpus xT")
    pooled = pd.concat(frames_of_actions, ignore_index=True)
    return ExpectedThreat().fit(pooled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--providers", default=_DEFAULT_PROVIDERS)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument("--sa-iterations", type=int, default=None, help="override SAParams.num_iterations (dev speed)")
    ap.add_argument(
        "--match-ids-json", default=None, help='JSON {"skillcorner": ["..."]} pinning this worker (parallel split).'
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    ap.add_argument(
        "--xt-path",
        default=None,
        help="Load a pre-fit ExpectedThreat (JSON) so PARALLEL workers score under ONE coherent xT; "
        "if the file is absent it is fit on the corpus and SAVED here (the fit step). A per-slice "
        "refit would make the pooled predictive correlation incoherent across shards.",
    )
    ap.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit the corpus xT to --xt-path and exit (the fit step before a parallel worker fan-out).",
    )
    args = ap.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    if args.fit_only and not args.xt_path:
        raise SystemExit("--fit-only requires --xt-path (the destination for the fit xT)")
    if not args.list_matches and not args.fit_only and not args.out:
        raise SystemExit("--out is required unless --list-matches or --fit-only is given")

    prov = (
        {"commit": "n/a", "dirty": False, "tree_state": "clean", "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    if args.list_matches:
        from scripts._partition import list_match_ids

        print(json.dumps(list_match_ids(args.providers.split(",")), indent=2))
        return

    import dataclasses

    from scripts._driver import for_each
    from scripts._loader_pining import load_matches
    from scripts._partition import providers_for_slice, worker_tag
    from silly_kicks.positioning import PositioningParams, SAParams

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out) if args.out else None  # None only on the --fit-only path (returns before use)
    worker = worker_tag(args.match_ids_json)
    providers = providers_for_slice(args.providers.split(","), match_ids)

    params = PositioningParams()
    if args.sa_iterations is not None:
        params = dataclasses.replace(
            params, sa=SAParams(num_iterations=args.sa_iterations, patience=args.sa_iterations)
        )

    # Pass 1: obtain the within-corpus xT. Fit ONCE on the corpus actions (the gap is a RELATIVE
    # measure; see the docstring), OR load a pre-fit xT via --xt-path so PARALLEL workers all score
    # under ONE coherent surface -- a per-slice refit would make the pooled predictive correlation
    # incoherent across shards. When --xt-path is given but absent, fit here and SAVE it (the fit
    # step); the worker fan-out then loads it (SK-XT-SER, ADR-100: pickle-free JSON + fail-closed load).
    from silly_kicks.xthreat import ExpectedThreat

    if args.xt_path and Path(args.xt_path).exists():
        xt = ExpectedThreat.load(args.xt_path)
    else:
        fit_items = list(
            load_matches(
                providers=providers, match_ids=match_ids, max_per_provider=args.max_per_provider, tracking_limit=0
            )
        )
        xt = _fit_corpus_xt(fit_items)
        if args.xt_path:
            xt.save(args.xt_path)

    if args.fit_only:  # the fit step: xT persisted, nothing to score/reduce here
        print(json.dumps({"fit_only": True, "xt_path": args.xt_path, "run_commit": prov["commit"]}, default=str))
        return

    if dest is None:  # unreachable: non-fit-only runs require --out (guarded above) -- narrows the type
        raise SystemExit("--out is required")

    xt_fp = _xt_fingerprint(xt)  # keys the shard cache on the xT SURFACE (cross-run coherence; SHOULD-FIX 09)

    _last_counts: dict = {}

    def _work(item):
        shard, counts = _measure_match(item, xt=xt, params=params)
        _last_counts.clear()
        _last_counts.update(counts)
        return shard

    res = for_each(
        load_matches(
            providers=providers,
            match_ids=match_ids,
            max_per_provider=args.max_per_provider,
            tracking_limit=args.tracking_limit,
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        counters=lambda _item, _frame: dict(_last_counts),
        shard_root=dest / "shards",
        token_inputs={
            "objective": "threat",
            "pitch_control_method": "spearman",
            "xt": xt_fp,  # the xT SURFACE digest: a changed surface => new generation, no stale-shard pooling
            "conceded_windows_s": list(_CONCEDED_WINDOWS_S),
            "sa_iterations": args.sa_iterations,
            "tracking_limit": args.tracking_limit,
            "schema": _SHARD_SCHEMA_VERSION,
        },
        tag=worker,
        label="match",
    )

    if args.match_ids_json is not None:
        worker_manifest = {
            **res.manifest(),
            "run_commit": prov["commit"],
            "run_tree_dirty": prov["dirty"],
            "run_tree_state": prov.get("tree_state"),
            "partition": worker,
            "note": "partition worker: shards written; run an UNPARTITIONED pass for the authoritative verdict.",
        }
        (dest / f"manifest_{worker}.json").write_text(
            json.dumps(worker_manifest, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(worker_manifest, indent=2, default=str))
        return

    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    shards = [pd.read_parquet(s) for s in shard_files]
    verdict = reduce_positioning_verdicts(pool_positioning_shards(shards))

    out = {
        "verdict": verdict,
        "registered_constants": {
            "MIN_DOMAIN_FRAMES": MIN_DOMAIN_FRAMES,
            "PREDICTIVE_ALPHA": PREDICTIVE_ALPHA,
            "CONCEDED_WINDOWS_S": list(_CONCEDED_WINDOWS_S),
            "MIN_UNITS": _MIN_UNITS,
        },
        "conservation": {
            "n_frames_in": res.counters.get("n_frames_in"),
            "n_frames_scored": res.counters.get("n_frames_scored"),
            "n_frames_dropped": res.counters.get("n_frames_dropped"),
            "n_matches": res.counters.get("n_matches"),
        },
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "partition": worker,
        "input_contract": input_contract(),
    }
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
