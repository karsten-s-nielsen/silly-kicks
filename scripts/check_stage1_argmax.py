#!/usr/bin/env python
"""TF-24 Stage-1 confirmation: does the recorded carrier optimum still hold on corrected geometry?

ADR-028 and its follow-ups moved away-team geometry after every Stage-1 sweep on record, so the
question is whether the recorded optimum survives. Two prongs, and they answer DIFFERENT questions:

1. INVARIANCE -- score the corpus, then score an exact point reflection of it, and compare carrier
   assignments frame by frame. Carrier inference should be orientation-invariant; the pre-registered
   threshold is >= 99.9% agreement (spec D5, fixed before any corrected-geometry data was scored).
2. ARGMAX -- re-score the recorded optimum and its immediate neighbours on the corrected frames and
   report whether the best point is still the recorded one.

**THE SHIPPED DEFAULT IS NOT THE STORE'S OPTIMUM, and scoring one while reporting the other is the
trap this script exists to avoid.** `_ball_carrier.DEFAULT_CARRIER_PARAMS` ships
`beta=0.0, gamma=0.25`; `calibration_runs/balanced_confirm_tol3` recorded
`beta=0.000194, gamma=0.22096`. The shipped values are the Optuna optimum ROUNDED. "Does the SHIPPED
default still win?" and "has the ARGMAX moved?" are both legitimate and they are not the same
question, so both points are scored and labelled separately in the artifact.

`tolerance_m` is held at 3.0 throughout: it is an engineering default that Stage 1 never calibrated
(only `beta` and `gamma` vary in the store), so sweeping it here would answer a question nobody
asked and would make the neighbour set incomparable to the recorded one.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from scripts._provenance import git_provenance, require_clean_tree

#: Pre-registered in the design BEFORE any corrected-geometry data was scored (spec D5). A gate
#: whose threshold is chosen after seeing the result is not a gate; `test_check_stage1_argmax.py`
#: pins both sides of it.
_INVARIANCE_THRESHOLD = 0.999

_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0

#: Held, not swept -- see the module docstring.
_TOLERANCE_M = 3.0

#: The shipped library default (`_ball_carrier.DEFAULT_CARRIER_PARAMS`), which is the recorded
#: optimum ROUNDED. Scored as its own labelled point.
_SHIPPED_POINT = {"beta": 0.0, "gamma": 0.25}

_VELOCITY_COLUMNS = ("vx", "vy")

#: A deliberately DISPLACED point, scored only to prove the objective responds to these parameters
#: at all. Measured on a 3-match fold: recorded 0.31456, this 0.30460 -- so prong 2 is not inert.
#: Without it, "the argmax did not move" is indistinguishable from "nothing could have moved it",
#: which is the failure `require_velocity` guards on the input side and this guards on the output
#: side. The neighbourhood around the optimum is genuinely FLAT (four neighbours scored identically
#: to 16 digits), so the distinction is not hypothetical.
_SENTINEL_POINT = {"beta": 5.0, "gamma": 2.0}


def invariance_verdict(*, same: int, total: int) -> str:
    """`"stands"` iff agreement clears the pre-registered threshold.

    Raises on an empty comparison rather than returning `"stands"`: zero frames compared would
    otherwise read as perfect invariance, which is the silent no-op-gate failure -- the gate reports
    success having tested nothing.
    """
    if total <= 0:
        raise ValueError("no frames compared; an empty comparison cannot be a pass")
    return "stands" if (same / total) >= _INVARIANCE_THRESHOLD else "sweep"


def require_velocity(frames: pd.DataFrame) -> None:
    """Raise unless the frames carry USABLE `vx`/`vy`.

    `_ball_carrier` sets `has_velocity = "vx" in frames.columns and "vy" in frames.columns` and
    silently substitutes `pvx = 0.0` otherwise. With velocity zeroed, `beta` becomes inert and every
    neighbour scores identically to the optimum -- the argmax "cannot move" for a reason that has
    nothing to do with geometry, and prong 2 reports a clean pass having tested nothing.

    Checks CONTENT, not just presence: an all-NaN `vx` passes a `in frames.columns` test and leaves
    beta exactly as inert. Present-but-empty is the same defect wearing a different shape.
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
    """An exact 180-degree point reflection of the pitch: positions mirrored, VELOCITIES NEGATED.

    Mirroring positions while leaving velocities pointing the original way is not a reflection --
    it is a physically incoherent frame, and `infer_ball_carrier` scores
    `cand_dists[ci] - beta * v_toward`, so the velocity term is live.

    **Prong 1 is structurally BLIND to getting this wrong**: the recorded optimum has `beta` at (or
    rounded to) zero, the term vanishes, and the invariance fraction is unaffected. Prong 2 is what
    gets corrupted, because its neighbours have `beta != 0` by construction -- exactly the scores
    the argmax comparison depends on. Hence the dedicated unit test.

    Returns a NEW frame; the caller scores both legs and a mutating reflection would make the
    "factual" leg the reflected one.
    """
    out = frames.copy(deep=True)
    out["x"] = _PITCH_LENGTH - frames["x"]
    out["y"] = _PITCH_WIDTH - frames["y"]
    for col in _VELOCITY_COLUMNS:
        if col in frames.columns:
            out[col] = -frames[col]
    return out


def _carrier_series(frames: pd.DataFrame, *, beta: float, gamma: float, pre: dict | None = None) -> pd.Series:
    """Carrier assignment keyed by frame, for one parameter point."""
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier

    out = infer_ball_carrier(frames, tolerance_m=_TOLERANCE_M, beta=beta, gamma=gamma, pre=pre)
    idx = pd.MultiIndex.from_frame(out[["game_id", "period_id", "frame_id"]].astype(str))
    return pd.Series(out["ball_carrier_player_id"].astype(str).to_numpy(), index=idx)


def compare_assignments(factual: pd.Series, reflected: pd.Series, *, count_no_carrier_as_agreement: bool) -> dict:
    """Agreement between two carrier assignments over the frames BOTH resolved.

    **The no-carrier rule is stated, not implied.** Frames where inference returns no carrier are
    counted explicitly: treating `None == None` as agreement inflates the fraction by however many
    dead-ball frames the corpus holds, and excluding them changes the denominator. Either is
    defensible; silence is not, in a pre-registered gate. The default here EXCLUDES them, because
    the claim under test is about carrier CHOICE and a frame with no carrier expresses none.
    """
    joined = pd.DataFrame({"a": factual, "b": reflected}).dropna(how="all")
    a_none = joined["a"].isin(["nan", "None", "<NA>"])
    b_none = joined["b"].isin(["nan", "None", "<NA>"])
    both_none = a_none & b_none
    n_no_carrier = int(both_none.sum())
    if count_no_carrier_as_agreement:
        scored = joined
        same = int((joined["a"] == joined["b"]).sum())
    else:
        scored = joined[~both_none]
        same = int((scored["a"] == scored["b"]).sum())
    return {
        "n_frames": len(scored),
        "n_same": same,
        "n_no_carrier": n_no_carrier,
        "no_carrier_convention": ("counted_as_agreement" if count_no_carrier_as_agreement else "excluded"),
    }


def load_neighbours(store: pathlib.Path, *, optimum: dict, k: int) -> list[dict]:
    """The K nearest completed trials to `optimum` in NORMALISED parameter space.

    Normalised because `beta` and `gamma` have different ranges, so a raw euclidean distance would
    make the wider parameter dominate "nearest" and the neighbour set would probe one axis only.

    **`beta` sits ON a boundary at ~0, so neighbours exist on one side only.** The count actually
    found is recorded rather than assumed symmetric -- a K that silently returns fewer points is a
    weaker test than it looks.
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
    order = np.argsort(d)
    out = []
    for i in order[: k + 1]:
        t = trials[i]
        if abs(t.params["beta"] - optimum["beta"]) < 1e-12 and abs(t.params["gamma"] - optimum["gamma"]) < 1e-12:
            continue  # the optimum itself
        out.append({"beta": float(t.params["beta"]), "gamma": float(t.params["gamma"]), "recorded_value": t.value})
        if len(out) == k:
            break
    return out


def build_fold(
    shards: list[pathlib.Path],
    *,
    actions_dir: pathlib.Path,
    home_teams: pathlib.Path,
    max_per_provider: int | None = None,
) -> dict:
    """`{provider: [(actions, frames, home_team_id)]}` -- the shape `CarrierAccuracyObjective` takes.

    Reconstructed from the corpus the materializer wrote: shards are `{provider}__{match}.parquet`,
    their actions sit at `<actions_dir>/{provider}__{match}.parquet`, and the home map is keyed by
    the `game_id` the FRAMES carry (SkillCorner's is a kloppy hash unrelated to its match id).

    A match missing either actions or a home id is SKIPPED and counted, never defaulted: a
    fabricated `home_team_id` would silently mis-orient one match's geometry inside an objective
    whose whole purpose here is to detect geometry-driven change.
    """
    home_map = json.loads(home_teams.read_text(encoding="utf-8"))
    fold: dict[str, list] = {}
    skipped = {"no_actions": 0, "no_home": 0}
    for shard in shards:
        stem = shard.stem
        provider = stem.split("__")[0]
        if max_per_provider is not None and len(fold.get(provider, [])) >= max_per_provider:
            continue
        apath = actions_dir / f"{stem}.parquet"
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


def _frame_parquets(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Frames only -- excludes `_actions/` and `_home/` sidecars, which are not frames."""
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
    ap.add_argument("--store", type=pathlib.Path, required=True, help="The PRIOR Stage-1 Optuna store (s1.db).")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--actions-dir", type=pathlib.Path, required=True, help="The corpus `_actions/` dir.")
    ap.add_argument("--home-teams", type=pathlib.Path, required=True, help="The corpus `home_teams.json`.")
    ap.add_argument("--k-neighbours", type=int, default=4)
    ap.add_argument(
        "--objective-matches-per-provider",
        type=int,
        default=5,
        help="Cap on matches per provider in the ARGMAX fold (the objective retains each match). "
        "The invariance prong always streams the FULL corpus; this bounds only the argmax fold.",
    )
    ap.add_argument("--max-matches", type=int, default=None, help="Dev smoke only; a capped run is not a result.")
    ap.add_argument(
        "--count-no-carrier-as-agreement",
        action="store_true",
        help="Count frames where BOTH legs found no carrier as agreement (default: exclude).",
    )
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    best = json.loads((args.store.parent / "carrier_best.json").read_text(encoding="utf-8"))
    recorded = {"beta": float(best["beta"]), "gamma": float(best["gamma"])}
    print(f"recorded optimum: {recorded} (tolerance_m held at {_TOLERANCE_M})")
    print(f"shipped point:    {_SHIPPED_POINT}")

    paths = _frame_parquets(args.data_dir)
    if args.max_matches:
        paths = paths[: args.max_matches]
    if not paths:
        raise SystemExit(f"no frame parquets under {args.data_dir}")
    # STREAM the invariance prong: one match in memory at a time. The corpus is ~727M rows across
    # 179 matches; a `pd.concat` of all of them, plus the deep copy `reflect_frames` makes, would
    # need roughly twice the box's RAM and would OOM hours into an unattended run. The counts are
    # additive, so streaming costs nothing in fidelity -- this prong still sees EVERY match.
    results: dict[str, dict] = {}
    for label, point in (("shipped_point", _SHIPPED_POINT), ("recorded_optimum", recorded)):
        acc = {"n_frames": 0, "n_same": 0, "n_no_carrier": 0}
        for i, path in enumerate(paths, 1):
            fr = pd.read_parquet(path)
            if i == 1 and label == "shipped_point":
                require_velocity(fr)
            fact = _carrier_series(fr, beta=point["beta"], gamma=point["gamma"])
            refl = _carrier_series(reflect_frames(fr), beta=point["beta"], gamma=point["gamma"])
            cmp1 = compare_assignments(fact, refl, count_no_carrier_as_agreement=args.count_no_carrier_as_agreement)
            for k in ("n_frames", "n_same", "n_no_carrier"):
                acc[k] += cmp1[k]
            del fr, fact, refl
            if i % 25 == 0:
                print(f"    {label}: {i}/{len(paths)} matches", flush=True)
        fraction = acc["n_same"] / acc["n_frames"] if acc["n_frames"] else 0.0
        verdict = invariance_verdict(same=acc["n_same"], total=acc["n_frames"])
        results[label] = {
            **acc,
            "point": point,
            "no_carrier_convention": ("counted_as_agreement" if args.count_no_carrier_as_agreement else "excluded"),
            "invariance_fraction": fraction,
            "verdict": verdict,
        }
        print(f"  {label}: {fraction:.6f} -> {verdict} (n={acc['n_frames']})", flush=True)

    neighbours = load_neighbours(args.store, optimum=recorded, k=args.k_neighbours)
    print(f"neighbours found: {len(neighbours)} of k={args.k_neighbours} requested", flush=True)

    # Re-score the optimum and its neighbours on the CORRECTED frames. The score is agreement with
    # the optimum's own assignment: a neighbour that reproduces it is indistinguishable here, which
    # is the honest reading -- this prong asks whether the argmax MOVED, not what its objective was.
    # Score with the SAME objective the store's values came from. An agreement-with-the-optimum
    # proxy was written here first and removed: it answers "do these parameters assign the same
    # carrier?", which is not the quantity that was maximised, so an argmax verdict derived from it
    # would compare two different metrics and call the result a confirmation.
    # The objective RETAINS each match (`_PreparedMatch` keeps `frames` plus a dense pre-index), so
    # its fold is provider-CAPPED and the cap is recorded. This is the harness's normal setting, not
    # a compromise: Stage 1 scores match-stratified CV folds, never the whole corpus at once.
    fold = build_fold(
        paths,
        actions_dir=args.actions_dir,
        home_teams=args.home_teams,
        max_per_provider=args.objective_matches_per_provider,
    )
    n_matches_scored = sum(len(v) for v in fold.values())
    if not n_matches_scored:
        raise SystemExit(
            "the objective fold is empty -- no match yielded (actions, frames, home_team_id). "
            "Without it the argmax prong cannot run, and reporting invariance alone as a "
            "confirmation would overstate what was tested."
        )
    print(f"objective fold: {n_matches_scored} matches across {sorted(fold)}", flush=True)

    from ruthless.result import Candidate

    from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective

    obj = CarrierAccuracyObjective(fold)

    def _score(label: str, beta: float, gamma: float) -> float:
        m = obj.evaluate(Candidate(id=label, params={"tolerance_m": _TOLERANCE_M, "beta": beta, "gamma": gamma}))
        return float(m["carrier_accuracy"])

    recorded_score = _score("recorded", recorded["beta"], recorded["gamma"])
    shipped_score = _score("shipped", _SHIPPED_POINT["beta"], _SHIPPED_POINT["gamma"])
    print(f"  recorded optimum score: {recorded_score:.6f}", flush=True)
    print(f"  shipped point score:    {shipped_score:.6f}", flush=True)

    scored = []
    for i, nb in enumerate(neighbours):
        s = _score(f"nb{i}", nb["beta"], nb["gamma"])
        scored.append({**nb, "score_on_corrected": s})
        print(f"  neighbour beta={nb['beta']:.6g} gamma={nb['gamma']:.6g}: {s:.6f}", flush=True)

    # The argmax MOVED iff some neighbour now scores strictly higher than the recorded optimum.
    # Strictly: a tie leaves the recorded point winning, which is the conservative reading for a
    # confirmation whose purpose is to avoid an unnecessary sweep.
    best_neighbour = max((s["score_on_corrected"] for s in scored), default=float("-inf"))
    argmax_moved = bool(best_neighbour > recorded_score)
    score_margin = best_neighbour - recorded_score if scored else None

    # Non-vacuity, on the OUTPUT side. A flat neighbourhood makes `argmax_moved` turn on a margin of
    # ~1e-4, which is a handful of actions; reported as a bare boolean that overstates the finding.
    # Scoring a deliberately displaced point separates "the optimum held" from "nothing could have
    # moved it", and records the margin so the verdict can be read rather than trusted.
    sentinel_score = _score("sentinel", _SENTINEL_POINT["beta"], _SENTINEL_POINT["gamma"])
    sentinel_delta = abs(sentinel_score - recorded_score)
    print(f"  sentinel {_SENTINEL_POINT}: {sentinel_score:.6f} (delta {sentinel_delta:.6f})", flush=True)
    if sentinel_delta == 0.0:
        raise SystemExit(
            "the objective did not move for a deliberately displaced point, so it is INSENSITIVE to "
            "these parameters here and no argmax verdict is meaningful. Refusing to write an "
            "artifact that would read as a confirmation."
        )

    out = {
        "shipped_point": results["shipped_point"],
        "recorded_optimum": results["recorded_optimum"],
        "tolerance_m_held": _TOLERANCE_M,
        "invariance_threshold": _INVARIANCE_THRESHOLD,
        "k_neighbours_requested": args.k_neighbours,
        "k_neighbours_found": len(neighbours),
        "neighbours": scored,
        "recorded_optimum_score": recorded_score,
        "shipped_point_score": shipped_score,
        "best_neighbour_score": best_neighbour if scored else None,
        "objective": "CarrierAccuracyObjective.carrier_accuracy",
        "n_matches_scored": n_matches_scored,
        "argmax_moved": argmax_moved,
        "score_margin": score_margin,
        "sentinel_point": _SENTINEL_POINT,
        "sentinel_score": sentinel_score,
        "sentinel_delta": sentinel_delta,
        "n_matches_invariance": len(paths),
        "objective_matches_per_provider_cap": args.objective_matches_per_provider,
        "store": str(args.store),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "neighbours"}, indent=2))


if __name__ == "__main__":
    main()
