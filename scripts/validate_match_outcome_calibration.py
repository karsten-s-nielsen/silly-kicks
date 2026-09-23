"""TF-53 Rung-2 match-outcome calibration study (owner-run, reported-not-gated, PUBLIC corpus).

Scores the win/draw/loss simplex against realized results on the StatsBomb open-data corpus and writes
a report -- it changes NO library default (the bundled rho stays an opt-in method; ADR-009). Per method
config (``independent`` / ``collapse`` / ``dixon_coles`` / ``both``) it emits the predicted simplex +
realized outcome per ``(game_id, team_id)``, then reduces to a 3-way Brier score, a calibration slope,
and an xPoints-vs-points bias. The two dependence arms (``dixon_coles`` / ``both``) use a PER-FOLD rho
fitted by :func:`cv_rho_by_fold` grouped by ``game_id`` -- evaluated held-out, NEVER on the fit data and
NEVER from the bundled weights, so the study stays a clean commit-2 producer.

The corpus map is ``for_each`` (ADR-052: per-match shards, resumable, conserving); the clean-tree guard
runs FIRST (ADR-037); the input contract declares which symbols the numbers depend on (ADR-056);
``load_open_data_matches`` is fail-closed public-only (``assert_statsbomb_open_data_mode``).

Usage (owner):
    python scripts/validate_match_outcome_calibration.py --out docs/research/tf53_match_outcome_calibration
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._input_contract import declare_inputs
from scripts.train_match_outcome_dependence import fit_rho
from silly_kicks.id_compat import ids_match
from silly_kicks.match_outcome import MatchOutcomeParams, apply_dependence, goal_count_pmf
from silly_kicks.match_outcome._collapse import collapse_team_xgs
from silly_kicks.spadl import config as spadlconfig

_SHOT_TYPE_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_SUCCESS = spadlconfig.result_id["success"]
_OWNGOAL = spadlconfig.result_id["owngoal"]

#: The four method configs the study scores side by side (the two orthogonal correction axes).
CONFIGS: tuple[str, ...] = ("independent", "collapse", "dixon_coles", "both")

#: Emitted shard schema + its generation token; MUST move together (ADR-052 / the 4.77.1 stale-shard
#: rule) -- the for_each fingerprint digests token_inputs only.
_SHARD_SCHEMA_VERSION = "match-outcome-calibration-1"
_EMITTED_SHARD_COLUMNS = [
    "config",
    "game_id",
    "team_id",
    "p_win",
    "p_draw",
    "p_loss",
    "xpoints",
    "outcome",  # realized: 0 win / 1 draw / 2 loss (team perspective)
    "points",  # realized: 3 / 1 / 0
]

#: Number of CV folds for the per-fold rho (grouped by game_id).
DEFAULT_FOLDS = 5


# --------------------------------------------------------------------------------------------------
# Pure kernels (CI-tested).
# --------------------------------------------------------------------------------------------------
def three_way_brier(probs, outcomes) -> float:
    """Multi-class Brier score: ``mean_i sum_k (p_ik - y_ik)^2`` over one-hot realized outcomes.

    ``probs`` is ``(n, 3)`` (``p_win`` / ``p_draw`` / ``p_loss``); ``outcomes`` is ``(n,)`` in
    ``{0, 1, 2}``. A perfect prediction scores 0; a uniform ``(1/3, 1/3, 1/3)`` scores ``2/3``.
    """
    p = np.asarray(probs, dtype="float64")
    o = np.asarray(outcomes, dtype="int64")
    y = np.zeros_like(p)
    y[np.arange(p.shape[0]), o] = 1.0
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def calibration_slope(probs, outcomes) -> float:
    """Logistic recalibration slope over the flattened per-class (predicted, realized) pairs.

    Fits ``y ~ b0 + b1 * logit(p)`` by IRLS across all ``n * 3`` (class-vs-not) pairs; ``b1`` is the
    slope. A perfectly-calibrated set recovers ``b1 ~= 1.0``; over-confident predictions give ``b1 < 1``.
    """
    p = np.asarray(probs, dtype="float64")
    o = np.asarray(outcomes, dtype="int64")
    y = np.zeros_like(p)
    y[np.arange(p.shape[0]), o] = 1.0
    pf = np.clip(p.ravel(), 1e-6, 1.0 - 1e-6)
    yf = y.ravel()
    x = np.log(pf / (1.0 - pf))  # logit of the predicted probability
    b0, b1 = 0.0, 1.0
    for _ in range(100):
        eta = np.clip(b0 + b1 * x, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu) + 1e-9
        z = eta + (yf - mu) / w
        sw = float(w.sum())
        swx = float((w * x).sum())
        swxx = float((w * x * x).sum())
        swz = float((w * z).sum())
        swxz = float((w * x * z).sum())
        det = sw * swxx - swx * swx
        if abs(det) < 1e-12:
            break
        nb0 = (swxx * swz - swx * swxz) / det
        nb1 = (sw * swxz - swx * swz) / det
        if abs(nb1 - b1) < 1e-10 and abs(nb0 - b0) < 1e-10:
            b0, b1 = nb0, nb1
            break
        b0, b1 = nb0, nb1
    return float(b1)


def team_outcome_probabilities(own_pmf, opp_pmf, *, rho: float | None = None) -> tuple[float, float, float]:
    """``(p_win, p_draw, p_loss)`` for the team whose PMF is ``own_pmf``.

    ``rho=None`` = team independence (the outer product); a float applies the Dixon-Coles low-score
    dependence (Rung 3a) with that rho.
    """
    own = np.asarray(own_pmf, dtype="float64")
    opp = np.asarray(opp_pmf, dtype="float64")
    joint = np.outer(own, opp) if rho is None else apply_dependence(own, opp, rho=rho)
    i = np.arange(joint.shape[0])[:, None]
    j = np.arange(joint.shape[1])[None, :]
    return float(joint[i > j].sum()), float(joint[i == j].sum()), float(joint[i < j].sum())


def cv_rho_by_fold(matches: Sequence[tuple], folds: int) -> list[dict]:
    """Fit one Dixon-Coles rho PER FOLD on train games; report held-out Brier (never on the fit data).

    ``matches`` is a sequence of ``(game_id, home_xgs, away_xgs, home_goals, away_goals)``. Games are
    partitioned into ``folds`` deterministic folds by sorted-index modulo. For each fold, rho is fit on
    the OTHER folds' games and the held-out fold is scored under both ``dixon_coles(rho)`` and
    ``independent``; ``train_game_ids`` / ``test_game_ids`` are disjoint by construction.
    """
    game_ids = sorted({str(m[0]) for m in matches})
    by_game = {str(m[0]): m for m in matches}
    out: list[dict] = []
    for f in range(folds):
        test_ids = [g for k, g in enumerate(game_ids) if k % folds == f]
        train_ids = [g for k, g in enumerate(game_ids) if k % folds != f]
        if not test_ids or not train_ids:
            continue
        train = [by_game[g][1:] for g in train_ids]  # drop game_id -> (home_xgs, away_xgs, hg, ag)
        rho = fit_rho(train)
        dc_probs, indep_probs, outcomes = [], [], []
        for g in test_ids:
            _gid, hx, ax, hg, ag = by_game[g]
            hp, ap = goal_count_pmf(hx), goal_count_pmf(ax)
            dc_probs.append(team_outcome_probabilities(hp, ap, rho=rho))
            indep_probs.append(team_outcome_probabilities(hp, ap, rho=None))
            outcomes.append(0 if hg > ag else (1 if hg == ag else 2))  # home perspective
        out.append(
            {
                "fold": f,
                "rho": rho,
                "n_train_games": len(train_ids),
                "n_test_games": len(test_ids),
                "train_game_ids": train_ids,
                "test_game_ids": test_ids,
                "heldout_brier_dixon": three_way_brier(dc_probs, outcomes),
                "heldout_brier_indep": three_way_brier(indep_probs, outcomes),
            }
        )
    return out


def reduce_calibration(shards: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Per-config 3-way Brier + calibration slope + xPoints-vs-points bias from the emitted shards."""
    if not shards:
        return pd.DataFrame(
            columns=["config", "n", "brier", "calibration_slope", "mean_xpoints", "mean_points", "xpoints_bias"]
        )
    df = pd.concat(shards, ignore_index=True)
    rows = []
    for config, cg in df.groupby("config", sort=True):
        probs = cg[["p_win", "p_draw", "p_loss"]].to_numpy(dtype="float64")
        outcomes = cg["outcome"].to_numpy(dtype="int64")
        mean_xp = float(cg["xpoints"].mean())
        mean_pts = float(cg["points"].mean())
        rows.append(
            {
                "config": config,
                "n": len(cg),
                "brier": three_way_brier(probs, outcomes),
                "calibration_slope": calibration_slope(probs, outcomes),
                "mean_xpoints": mean_xp,
                "mean_points": mean_pts,
                "xpoints_bias": mean_xp - mean_pts,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Corpus extraction (owner-run; NOT CI-exercised beyond the kernels above).
# --------------------------------------------------------------------------------------------------
def _team_stats(actions: pd.DataFrame, xg_column: str, *, gap: float) -> dict | None:
    """Per-team ``{team_id: {indep, collapse, goals}}`` for a 2-team match, else ``None``.

    ``indep`` = per-shot xGs (NaN dropped); ``collapse`` = the possession-collapsed xGs (Rung 3b);
    ``goals`` = successful shots + the OTHER team's own goals (ADR-018 result-coded).
    """
    from silly_kicks import spadl

    teams = list(pd.unique(actions["team_id"].dropna()))
    if len(teams) != 2:
        return None
    ctx = spadl.add_possessions(actions, max_gap_seconds=gap)
    succ, og, stats = {}, {}, {}
    for t in teams:
        tm = ids_match(ctx["team_id"], t).to_numpy()
        ta = ctx.iloc[tm]
        shots = ta[ta["type_id"].isin(_SHOT_TYPE_IDS)]
        succ[t] = int((shots["result_id"] == _SUCCESS).sum())
        og[t] = int((ta["result_id"] == _OWNGOAL).sum())
        stats[t] = {
            "indep": shots[xg_column].dropna().to_numpy(dtype="float64")
            if xg_column in shots.columns
            else np.array([]),
            "collapse": collapse_team_xgs(shots, xg_column=xg_column),
        }
    for idx, t in enumerate(teams):
        stats[t]["goals"] = succ[t] + og[teams[1 - idx]]  # own scored + opponent own goal
    return stats


def _build_rows(game_id, *, stats: dict, rho: float | None) -> pd.DataFrame:
    """Per-``(config, team)`` predicted simplex + realized outcome rows for ONE match."""
    teams = list(stats.keys())
    rows = []
    for idx, t in enumerate(teams):
        opp = teams[1 - idx]
        own, other = stats[t], stats[opp]
        outcome = 0 if own["goals"] > other["goals"] else (1 if own["goals"] == other["goals"] else 2)
        points = (3, 1, 0)[outcome]
        for config in CONFIGS:
            use_collapse = config in ("collapse", "both")
            use_rho = rho if config in ("dixon_coles", "both") else None
            own_pmf = goal_count_pmf(own["collapse"] if use_collapse else own["indep"])
            opp_pmf = goal_count_pmf(other["collapse"] if use_collapse else other["indep"])
            p_win, p_draw, p_loss = team_outcome_probabilities(own_pmf, opp_pmf, rho=use_rho)
            rows.append(
                {
                    "config": config,
                    "game_id": game_id,
                    "team_id": t,
                    "p_win": p_win,
                    "p_draw": p_draw,
                    "p_loss": p_loss,
                    "xpoints": 3.0 * p_win + p_draw,
                    "outcome": outcome,
                    "points": points,
                }
            )
    return pd.DataFrame(rows).reindex(columns=_EMITTED_SHARD_COLUMNS)


#: The prepass shard: one match's per-team xG arrays + goals (the ADR-052 U-shape prepass, Task 13).
_STATS_SHARD_SCHEMA_VERSION = "match-outcome-calibration-stats-1"
_STATS_SHARD_COLUMNS = [
    "game_id",
    "team0_id",
    "team1_id",
    "team0_indep",
    "team1_indep",
    "team0_collapse",
    "team1_collapse",
    "team0_goals",
    "team1_goals",
]


def extract_stats_slice(match_id, actions: pd.DataFrame, *, gap: float) -> pd.DataFrame:
    """One match's per-team xG arrays + goals as a shard (the corpus prepass; pure). An EMPTY frame is
    a non-two-team match, dropped from the CV corpus ("ran, produced no scoreline"; ADR-052). The xG
    arrays are stored as lists so the shard round-trips through parquet."""
    stats = _team_stats(actions, "xg", gap=gap)
    if stats is None:
        return pd.DataFrame(columns=_STATS_SHARD_COLUMNS)
    t0, t1 = list(stats.keys())
    return pd.DataFrame(
        [
            {
                "game_id": str(match_id),
                "team0_id": t0,
                "team1_id": t1,
                "team0_indep": [float(x) for x in stats[t0]["indep"]],
                "team1_indep": [float(x) for x in stats[t1]["indep"]],
                "team0_collapse": [float(x) for x in stats[t0]["collapse"]],
                "team1_collapse": [float(x) for x in stats[t1]["collapse"]],
                "team0_goals": int(stats[t0]["goals"]),
                "team1_goals": int(stats[t1]["goals"]),
            }
        ]
    )


def stats_from_shards(frames: Sequence[pd.DataFrame]) -> tuple[dict, list]:
    """Rebuild ``(by_game, cv_matches)`` from the prepass shards (the whole-corpus barrier). Order does
    NOT matter -- ``cv_rho_by_fold`` sorts the game ids, so a resumed/partitioned run fits the same
    per-fold rho as the pre-migration in-memory pass."""
    by_game: dict[str, dict] = {}
    cv: list[tuple] = []
    for f in frames:
        for r in f.to_dict("records"):
            mid, t0, t1 = str(r["game_id"]), r["team0_id"], r["team1_id"]
            stats = {
                t0: {
                    "indep": np.asarray(r["team0_indep"], dtype="float64"),
                    "collapse": np.asarray(r["team0_collapse"], dtype="float64"),
                    "goals": int(r["team0_goals"]),
                },
                t1: {
                    "indep": np.asarray(r["team1_indep"], dtype="float64"),
                    "collapse": np.asarray(r["team1_collapse"], dtype="float64"),
                    "goals": int(r["team1_goals"]),
                },
            }
            by_game[mid] = stats
            cv.append((mid, stats[t0]["indep"], stats[t1]["indep"], stats[t0]["goals"], stats[t1]["goals"]))
    return by_game, cv


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056)."""
    from dataclasses import asdict

    return declare_inputs(
        driver="validate_match_outcome_calibration",
        params={
            "match_outcome": asdict(MatchOutcomeParams()),
            "configs": list(CONFIGS),
            "default_folds": DEFAULT_FOLDS,
        },
        extractors=["silly_kicks.match_outcome._pmf", "silly_kicks.match_outcome._collapse"],
        models=["silly_kicks.match_outcome._dependence"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument(
        "--match-ids-json", default=None, help='JSON ["3857276", ...] pinning WHICH matches (parallel split).'
    )
    ap.add_argument(
        "--cache-dir", default=None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    args = ap.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    prov = (
        {"commit": "n/a", "dirty": False, "tree_state": "clean", "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None

    from scripts._driver import for_each
    from scripts._sb_open_data import all_open_competitions, open_data_source

    gap = MatchOutcomeParams().possession_max_gap_seconds
    refs, load = open_data_source(
        all_open_competitions(), match_ids=match_ids, max_matches=args.max_matches, cache_dir=args.cache_dir
    )

    if args.list_matches:
        # LIST the refs -- never build every match just to print its id (spec section 4.5).
        print(json.dumps([ref.match_id for ref in refs], indent=2))
        return

    dest = Path(args.out)
    # Phase 1 (ADR-052 prepass): shard each match's per-team xG stats behind for_each's resume check --
    # before this, a crash re-downloaded the whole open-data corpus. The reduce rebuilds the CV corpus.
    stats_res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=lambda item: extract_stats_slice(item[1], item[2], gap=gap),
        shard_root=dest / "stats_shards",
        token_inputs={"metric": "match_outcome_calibration_stats", "schema": _STATS_SHARD_SCHEMA_VERSION},
        label="match",
    )
    stats_shards = [pd.read_parquet(s) for s in sorted(stats_res.shard_dir.glob("*.parquet"))]
    by_game, cv_matches = stats_from_shards(stats_shards)

    # Phase 2: per-fold rho (grouped by game_id, evaluated held-out; NEVER the bundled weights).
    folds_out = cv_rho_by_fold(cv_matches, args.folds)
    heldout_rho = {g: fo["rho"] for fo in folds_out for g in fo["test_game_ids"]}

    # Phase 3: for_each over the loaded match ids (work is trivial next to Phase-1's load).
    def _work(mid: str) -> pd.DataFrame:
        return _build_rows(mid, stats=by_game[mid], rho=heldout_rho.get(mid))

    res = for_each(
        list(by_game.keys()),
        key=lambda mid: str(mid),
        work=_work,
        shard_root=dest / "shards",
        token_inputs={
            "metric": "match_outcome_calibration",
            "schema": _SHARD_SCHEMA_VERSION,
            "folds": args.folds,
            "configs": list(CONFIGS),
        },
        label="match",
    )

    shards = [pd.read_parquet(s) for s in sorted(res.shard_dir.glob("*.parquet"))]
    summary = reduce_calibration(shards)

    out = {
        "n_matches": len(by_game),
        "per_config": summary.to_dict(orient="records"),
        "cv_rho_by_fold": [
            {k: v for k, v in fo.items() if k not in ("train_game_ids", "test_game_ids")} for fo in folds_out
        ],
        "note": (
            "Reported-not-gated (ADR-009): the study changes NO library default. The dixon_coles/both "
            "arms use a per-fold rho (grouped by game_id, held-out), never the bundled weights."
        ),
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "input_contract": input_contract(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if not summary.empty:
        summary.to_parquet(dest / "per_config_calibration.parquet", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "input_contract"}, indent=2, default=str))


if __name__ == "__main__":
    main()
