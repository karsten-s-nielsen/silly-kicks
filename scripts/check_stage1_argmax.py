#!/usr/bin/env python
"""TF-24 Stage-1 confirmation: does the recorded carrier optimum still hold on corrected geometry?

ADR-028 and its follow-ups moved away-team geometry after every Stage-1 sweep on record, so the
question is whether the recorded optimum survives. Two prongs, answering DIFFERENT questions:

1. INVARIANCE -- score the corpus, score its exact point reflection, compare carrier assignments.
   Carrier inference should be orientation-invariant; the threshold (>= 99.9%) is pre-registered in
   spec D5, fixed before any corrected-geometry data was scored.
2. ARGMAX -- re-score the recorded optimum and its nearest trials and report whether the best point
   moved BY MORE THAN THE NOISE.

**BOTH PRONGS USE THE HARNESS'S OWN MACHINERY, and the first draft of this script did not.** That
draft hand-rolled a corpus loop (which `pd.concat`-ed 727M rows and would have OOM'd a 119 GB box),
hand-rolled disjoint folds, and invented a "sentinel delta vs margin" ratio to decide whether a
difference mattered. All three already exist here and are used now:

* `scripts/_driver.py::for_each` (ADR-052) streams the corpus BY CONSTRUCTION, shards per match,
  resumes after a crash and asserts conservation. The OOM was unreachable through this seam.
* `calibration._cv.match_cv_splits` gives match-stratified folds -- GroupKFold(5) above 7 matches,
  leave-one-match-out below -- so no match appears in both sides.
* `calibration._cv.cv_standard_error` is the noise floor: `std(ddof=1)/sqrt(n_folds)`.

**THE SHIPPED DEFAULT IS NOT THE STORE'S OPTIMUM.** `_ball_carrier.DEFAULT_CARRIER_PARAMS` ships
`beta=0.0, gamma=0.25`; the store recorded `0.000194, 0.22096` -- the optimum ROUNDED. "Does the
SHIPPED default still win?" and "has the ARGMAX moved?" are different questions, so both points are
scored and labelled separately.

`tolerance_m` is held at 3.0: only `beta`/`gamma` vary in the store, so sweeping it here would
answer a question nobody asked and make the neighbour set incomparable to the recorded one. Whether
it SHOULD be swept is a live question -- it moves the objective by ~0.42 across its plausible range
while beta/gamma move it by ~4e-4 -- but that is a design decision, not a knob to turn here.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from scripts._driver import for_each
from scripts._provenance import git_provenance, require_clean_tree

#: Pre-registered in spec D5 BEFORE any corrected-geometry data was scored. A gate whose threshold
#: is chosen after seeing the result is not a gate.
_INVARIANCE_THRESHOLD = 0.999

_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0

#: Held, not swept -- see the module docstring.
_TOLERANCE_M = 3.0

#: The shipped library default, which is the recorded optimum ROUNDED. Scored as its own point.
_SHIPPED_POINT = {"beta": 0.0, "gamma": 0.25}

_VELOCITY_COLUMNS = ("vx", "vy")

#: Bump when the SHAPE of an invariance shard changes, so a resumed pass cannot combine old rows
#: with new ones (the 4.77.1 defect).
_SHARD_SCHEMA_VERSION = "tf24-invariance-1"


def invariance_verdict(*, same: int, total: int) -> str:
    """`"stands"` iff agreement clears the pre-registered threshold.

    Raises on an empty comparison rather than returning `"stands"`: zero frames compared would read
    as perfect invariance, i.e. a gate reporting success having tested nothing.
    """
    if total <= 0:
        raise ValueError("no frames compared; an empty comparison cannot be a pass")
    return "stands" if (same / total) >= _INVARIANCE_THRESHOLD else "sweep"


def stage1_metric_and_direction() -> tuple[str, object]:
    """The metric name and optimisation direction, READ FROM the sweep's own config.

    `stage1_config` is the single source for both (`metric="carrier_accuracy"`,
    `direction=Direction.MAXIMIZE`). Hard-coding "higher is better" into a `>` here would leave this
    confirmation silently disagreeing with the sweep the day either changes -- the confirmation would
    keep reporting on a design that no longer exists, which is the exact failure mode the whole
    corrected-geometry re-check exists to catch one level up.
    """
    from silly_kicks.calibration._spaces import stage1_config

    cfg = stage1_config(n_trials=1, store_path=":memory:")
    return cfg.metric, cfg.direction


def moved_beyond_noise(*, recorded: float, best_alternative: float, se: float, maximize: bool = True) -> bool:
    """True iff the gain over the recorded point exceeds the between-fold standard error.

    Mirrors `calibration._diagnostics.tf25_gate_fires`, which decides the same question for
    provider-specific defaults: *a difference only counts when it clears that fold set's own SE.*
    Deliberately a separate function rather than a call into it -- `tf25_gate_fires`'s parameters
    are named for a MINIMISED Brier, and routing a maximised accuracy through `global_brier` /
    `provider_best_brier` would make every call site read as its own opposite.

    A `nan` SE (single fold) can never justify "moved", exactly as a nan SE can never justify a
    provider-specific default there.
    """
    if se is None or not np.isfinite(se):
        return False
    gain = (best_alternative - recorded) if maximize else (recorded - best_alternative)
    return gain > se


def require_velocity(frames: pd.DataFrame) -> None:
    """Raise unless the frames carry USABLE `vx`/`vy`.

    `_ball_carrier` substitutes `pvx = 0.0` when velocity is absent, which makes `beta` inert so
    every candidate scores identically -- an argmax that "cannot move" for a reason that has nothing
    to do with geometry. Checks CONTENT, not just presence: an all-NaN column passes a
    `in frames.columns` test and is exactly as inert.
    """
    missing = [c for c in _VELOCITY_COLUMNS if c not in frames.columns]
    if missing:
        raise ValueError(
            f"frames lack {missing}; `infer_ball_carrier` would substitute 0.0 and make `beta` "
            f"inert, so the argmax could not move regardless of geometry. Refusing to score."
        )
    empty = [c for c in _VELOCITY_COLUMNS if not np.isfinite(frames[c].to_numpy(dtype=float)).any()]
    if empty:
        raise ValueError(
            f"frames carry {empty} but every value is non-finite; `beta` is inert for the same "
            f"reason as an absent column. Refusing to score."
        )


def reflect_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """An exact 180-degree point reflection: positions mirrored, VELOCITIES NEGATED.

    Mirroring positions while leaving velocities pointing the original way is not a reflection --
    `infer_ball_carrier` scores `cand_dists[ci] - beta * v_toward`, so the velocity term is live.
    Prong 1 is structurally BLIND to getting this wrong (beta ~ 0 kills the term); prong 2 is what
    gets corrupted, its neighbours having `beta != 0` by construction. Hence the dedicated test.

    Returns a NEW frame -- a mutating reflection would make the "factual" leg the reflected one.
    """
    out = frames.copy(deep=True)
    out["x"] = _PITCH_LENGTH - frames["x"]
    out["y"] = _PITCH_WIDTH - frames["y"]
    for col in _VELOCITY_COLUMNS:
        if col in frames.columns:
            out[col] = -frames[col]
    return out


def _carrier_series(frames: pd.DataFrame, *, beta: float, gamma: float) -> pd.Series:
    """Carrier assignment keyed by frame, for one parameter point."""
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier

    out = infer_ball_carrier(frames, tolerance_m=_TOLERANCE_M, beta=beta, gamma=gamma)
    idx = pd.MultiIndex.from_frame(out[["game_id", "period_id", "frame_id"]].astype(str))
    return pd.Series(out["ball_carrier_player_id"].astype(str).to_numpy(), index=idx)


def compare_assignments(factual: pd.Series, reflected: pd.Series, *, count_no_carrier_as_agreement: bool) -> dict:
    """Agreement between two carrier assignments.

    **The no-carrier rule is stated, not implied.** Treating `no-carrier == no-carrier` as agreement
    inflates the fraction by every dead-ball frame; excluding them changes the denominator. Either is
    defensible, silence is not. The default EXCLUDES them: the claim under test is about carrier
    CHOICE, and a frame with no carrier expresses none.
    """
    joined = pd.DataFrame({"a": factual, "b": reflected}).dropna(how="all")
    both_none = joined["a"].isin(["nan", "None", "<NA>"]) & joined["b"].isin(["nan", "None", "<NA>"])
    n_no_carrier = int(both_none.sum())
    scored = joined if count_no_carrier_as_agreement else joined[~both_none]
    return {
        "n_frames": len(scored),
        "n_same": int((scored["a"] == scored["b"]).sum()),
        "n_no_carrier": n_no_carrier,
    }


def invariance_rows(shard: pathlib.Path, *, points: dict, count_no_carrier_as_agreement: bool) -> pd.DataFrame:
    """One match's invariance counts, as the tidy frame `for_each` requires.

    This is the `work` callable: it sees ONE match, so the corpus is streamed by construction and
    the 727M-row concatenation that OOM'd the first draft is unreachable here.
    """
    frames = pd.read_parquet(shard)
    require_velocity(frames)
    mirrored = reflect_frames(frames)
    rows = []
    for label, point in points.items():
        cmp = compare_assignments(
            _carrier_series(frames, **point),
            _carrier_series(mirrored, **point),
            count_no_carrier_as_agreement=count_no_carrier_as_agreement,
        )
        rows.append({"match": shard.stem, "point": label, **cmp})
    return pd.DataFrame.from_records(rows)


def build_fold(
    shards: list[pathlib.Path],
    *,
    actions_dir: pathlib.Path,
    home_teams: pathlib.Path,
    max_per_provider: int | None = None,
) -> dict:
    """`{provider: [(actions, frames, home_team_id)]}` -- the shape `CarrierAccuracyObjective` takes.

    A match missing either actions or a home id is SKIPPED and counted, never defaulted: a
    fabricated `home_team_id` would silently mis-orient one match's geometry inside an objective
    whose whole purpose here is to detect geometry-driven change.

    Capped because `_PreparedMatch` retains `frames` plus a dense pre-index for every match in the
    fold. That is the harness's normal setting rather than a compromise -- Stage 1 scores
    match-stratified CV folds, never the whole corpus at once.
    """
    home_map = json.loads(home_teams.read_text(encoding="utf-8"))
    fold: dict[str, list] = {}
    skipped = {"no_actions": 0, "no_home": 0}
    for shard in shards:
        provider = shard.stem.split("__")[0]
        if max_per_provider is not None and len(fold.get(provider, [])) >= max_per_provider:
            continue
        apath = actions_dir / f"{shard.stem}.parquet"
        if not apath.is_file():
            skipped["no_actions"] += 1
            continue
        frames = pd.read_parquet(shard)
        gids = {str(g) for g in frames["game_id"].dropna().unique()}
        home = next((home_map[g] for g in gids if g in home_map), None)
        if home is None:
            skipped["no_home"] += 1
            continue
        fold.setdefault(provider, []).append((pd.read_parquet(apath), frames, home))
    if any(skipped.values()):
        print(f"  fold skips (counted, not defaulted): {skipped}", flush=True)
    return fold


def score_points_by_cv_fold(fold: dict, points: dict) -> dict[str, list[float]]:
    """Per-CV-fold carrier accuracy for each candidate point.

    Uses `match_cv_splits` -- the harness's own match-stratified scheme (GroupKFold(5) above 7
    matches, leave-one-match-out below) -- rather than a hand-partitioned fold set. The per-fold
    metrics are what `cv_standard_error` turns into the noise floor, so the split must be the one
    the rest of the harness uses or the SE describes a different design.

    Test sets are disjoint and cover the corpus, so each match is prepared exactly once across all
    folds -- the caching in `CarrierAccuracyObjective` is not wasted by scoring fold-wise.
    """
    from ruthless.result import Candidate

    from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
    from silly_kicks.calibration._cv import match_cv_splits

    flat = [(prov, m) for prov, matches in fold.items() for m in matches]
    match_ids = np.array([f"{prov}__{i}" for i, (prov, _m) in enumerate(flat)])
    out: dict[str, list[float]] = {name: [] for name in points}
    for _train_idx, test_idx in match_cv_splits(match_ids):
        sub: dict[str, list] = {}
        for i in test_idx:
            prov, m = flat[i]
            sub.setdefault(prov, []).append(m)
        obj = CarrierAccuracyObjective(sub)
        for name, p in points.items():
            m = obj.evaluate(Candidate(id=name, params={"tolerance_m": _TOLERANCE_M, **p}))
            out[name].append(float(m["carrier_accuracy"]))
    return out


def load_neighbours(store: pathlib.Path, *, optimum: dict, k: int) -> list[dict]:
    """The K nearest completed trials to `optimum` in NORMALISED parameter space.

    Normalised because `beta` and `gamma` have different ranges; a raw euclidean distance would let
    the wider parameter dominate and the probe would see one axis only. `beta` sits ON a boundary at
    ~0, so neighbours exist on one side -- the count FOUND is recorded rather than assumed symmetric.
    """
    import optuna

    study = optuna.load_study(study_name=None, storage=f"sqlite:///{store}")
    trials = [t for t in study.trials if t.state.name == "COMPLETE" and {"beta", "gamma"} <= set(t.params)]
    if not trials:
        raise SystemExit(f"{store}: no COMPLETE trials carrying beta+gamma; wrong store?")
    betas = np.array([t.params["beta"] for t in trials], dtype=float)
    gammas = np.array([t.params["gamma"] for t in trials], dtype=float)
    span_b = float(betas.max() - betas.min()) or 1.0
    span_g = float(gammas.max() - gammas.min()) or 1.0
    d = np.hypot((betas - optimum["beta"]) / span_b, (gammas - optimum["gamma"]) / span_g)
    out = []
    for i in np.argsort(d):
        t = trials[i]
        if abs(t.params["beta"] - optimum["beta"]) < 1e-12 and abs(t.params["gamma"] - optimum["gamma"]) < 1e-12:
            continue  # the optimum is not its own neighbour: a guaranteed tie hides real movement
        out.append({"beta": float(t.params["beta"]), "gamma": float(t.params["gamma"]), "recorded_value": t.value})
        if len(out) == k:
            break
    return out


def shard_key(shard: pathlib.Path) -> tuple[str, str]:
    """`(provider, match_id)` for a `{provider}__{match_id}.parquet` shard.

    The match id must NOT keep the `provider__` prefix: `join_key` rejects a key component that
    contains its own `__` separator, because `("a__b", "c")` and `("a", "b__c")` would both join to
    `"a__b__c"` and two distinct items would silently share one shard. Splitting on the FIRST
    separator only, so a match id containing `__` still round-trips.
    """
    provider, _, match = shard.stem.partition("__")
    return (provider, match or shard.stem)


def frame_parquets(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Frame shards only -- underscore-prefixed sidecars (`_actions/`, `_home/`) are not frames."""
    named = sorted(data_dir.glob("**/frames.parquet"))
    if named:
        return named
    return sorted(
        p
        for p in data_dir.glob("**/*.parquet")
        if not any(part.startswith("_") for part in p.relative_to(data_dir).parts[:-1])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-24 Stage-1 confirmation on corrected geometry.")
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--actions-dir", type=pathlib.Path, required=True)
    ap.add_argument("--home-teams", type=pathlib.Path, required=True)
    ap.add_argument("--store", type=pathlib.Path, required=True, help="The PRIOR Stage-1 Optuna store (s1.db).")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--k-neighbours", type=int, default=4)
    ap.add_argument("--objective-matches-per-provider", type=int, default=5)
    ap.add_argument("--max-matches", type=int, default=None, help="Dev smoke only; a capped run is not a result.")
    ap.add_argument("--count-no-carrier-as-agreement", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    best = json.loads((args.store.parent / "carrier_best.json").read_text(encoding="utf-8"))
    recorded = {"beta": float(best["beta"]), "gamma": float(best["gamma"])}
    points = {"shipped_point": _SHIPPED_POINT, "recorded_optimum": recorded}
    print(f"recorded optimum: {recorded} (tolerance_m held at {_TOLERANCE_M})")
    print(f"shipped point:    {_SHIPPED_POINT}")

    paths = frame_parquets(args.data_dir)
    if args.max_matches:
        paths = paths[: args.max_matches]
    if not paths:
        raise SystemExit(f"no frame parquets under {args.data_dir}")

    # PRONG 1, through the ADR-052 seam: streams by construction, shards per match, resumes.
    res = for_each(
        paths,
        key=shard_key,
        work=lambda p: invariance_rows(
            p, points=points, count_no_carrier_as_agreement=args.count_no_carrier_as_agreement
        ),
        shard_root=args.out / "shards",
        token_inputs={
            "points": {k: dict(sorted(v.items())) for k, v in sorted(points.items())},
            "tolerance_m": _TOLERANCE_M,
            "no_carrier": bool(args.count_no_carrier_as_agreement),
            "schema": _SHARD_SCHEMA_VERSION,
        },
        label="match",
    )
    counts = pd.concat([pd.read_parquet(p) for p in sorted(res.shard_dir.glob("*.parquet"))], ignore_index=True)
    invariance = {}
    for label in points:
        sub = counts[counts["point"] == label]
        n_frames, n_same = int(sub["n_frames"].sum()), int(sub["n_same"].sum())
        invariance[label] = {
            "n_frames": n_frames,
            "n_same": n_same,
            "n_no_carrier": int(sub["n_no_carrier"].sum()),
            "no_carrier_convention": ("counted_as_agreement" if args.count_no_carrier_as_agreement else "excluded"),
            "point": points[label],
            "invariance_fraction": n_same / n_frames if n_frames else 0.0,
            "verdict": invariance_verdict(same=n_same, total=n_frames),
        }
        print(f"  {label}: {invariance[label]['invariance_fraction']:.6f} -> {invariance[label]['verdict']}")

    # PRONG 2: the harness's CV, and its standard error as the noise floor.
    neighbours = load_neighbours(args.store, optimum=recorded, k=args.k_neighbours)
    fold = build_fold(
        paths,
        actions_dir=args.actions_dir,
        home_teams=args.home_teams,
        max_per_provider=args.objective_matches_per_provider,
    )
    n_scored = sum(len(v) for v in fold.values())
    if not n_scored:
        raise SystemExit("the objective fold is empty; the argmax prong cannot run")
    candidates = {**points, **{f"nb{i}": {"beta": n["beta"], "gamma": n["gamma"]} for i, n in enumerate(neighbours)}}
    per_fold = score_points_by_cv_fold(fold, candidates)

    from ruthless import Direction

    from silly_kicks.calibration._cv import cv_scheme_for, cv_standard_error

    metric_name, direction = stage1_metric_and_direction()
    maximize = direction == Direction.MAXIMIZE

    summary = {
        name: {
            "mean": float(np.mean(v)),
            "se": cv_standard_error(v),
            "per_fold": v,
            "params": candidates[name],
        }
        for name, v in per_fold.items()
    }
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {s['mean']:.6f} +/- {s['se']:.6f}  {name}")

    rec = summary["recorded_optimum"]
    alts = {k: v for k, v in summary.items() if k.startswith("nb")}
    best_alt = max(alts.values(), key=lambda s: s["mean"]) if alts else None
    argmax_moved = (
        moved_beyond_noise(recorded=rec["mean"], best_alternative=best_alt["mean"], se=rec["se"], maximize=maximize)
        if best_alt
        else False
    )

    out = {
        "invariance": invariance,
        "invariance_threshold": _INVARIANCE_THRESHOLD,
        "n_matches_invariance": len(paths),
        "objective_matches_per_provider_cap": args.objective_matches_per_provider,
        "n_matches_objective": n_scored,
        "cv_scheme": cv_scheme_for(n_scored),
        "objective": f"CarrierAccuracyObjective.{metric_name}",
        "direction": str(direction),
        "points": summary,
        "k_neighbours_requested": args.k_neighbours,
        "k_neighbours_found": len(neighbours),
        "argmax_moved": argmax_moved,
        "argmax_rule": "best neighbour mean - recorded mean > recorded CV standard error",
        "store": str(args.store),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"argmax_moved={argmax_moved} (rule: {out['argmax_rule']})")


if __name__ == "__main__":
    main()
