"""Fit the TF-53 Dixon-Coles dependence rho on the PUBLIC StatsBomb open-data corpus (owner-run).

1-D MLE (scipy ``minimize_scalar``, bounded) of the realized scoreline under the tau-corrected
Poisson-binomial joint (spec section 5). Writes a pickle-free ``weights/{model.json, SHA256SUMS}`` stamping
``training_commit`` on a CLEAN tree (ADR-037). ``statsbombpy`` is a scripts-only network dep; no Optuna
(the likelihood is smooth and one-dimensional). NOT exercised in CI beyond the pure ``fit_rho`` kernel.

Usage (owner):
    python scripts/train_match_outcome_dependence.py --out silly_kicks/match_outcome/weights
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def _nll(rho: float, matches: Sequence[tuple]) -> float:
    """Negative log-likelihood of realized scorelines under the tau(rho)-corrected PB joint."""
    from silly_kicks.match_outcome import apply_dependence, goal_count_pmf

    total = 0.0
    for home_xgs, away_xgs, hg, ag in matches:
        joint = apply_dependence(goal_count_pmf(home_xgs), goal_count_pmf(away_xgs), rho=float(rho))
        p = joint[hg, ag] if hg < joint.shape[0] and ag < joint.shape[1] else 0.0
        total -= float(np.log(max(p, 1e-12)))
    return total


def fit_rho(matches: Sequence[tuple], *, bounds: tuple[float, float] = (-0.5, 0.5)) -> float:
    """MLE of the Dixon-Coles rho over ``(home_xgs, away_xgs, home_goals, away_goals)`` tuples."""
    from scipy.optimize import minimize_scalar  # scipy is fit-time only (never at serve)

    res = minimize_scalar(lambda r: _nll(r, matches), bounds=bounds, method="bounded")
    return float(res.x)


_SHOT_NAMES = ("shot", "shot_penalty", "shot_freekick")


def _match_tuples(actions: pd.DataFrame, xg_column: str) -> list[tuple]:
    """Per-match ``(home_xgs, away_xgs, home_goals, away_goals)`` from SPADL actions with injected xg."""
    from silly_kicks.spadl import config as spadlconfig

    shot_ids = [spadlconfig.actiontype_id[n] for n in _SHOT_NAMES]
    success = spadlconfig.result_id["success"]
    owngoal = spadlconfig.result_id["owngoal"]
    out: list[tuple] = []
    for _gid, ga in actions.groupby("game_id", sort=False):
        teams = list(pd.unique(ga["team_id"].dropna()))
        if len(teams) != 2:
            continue
        per = []
        for t in teams:
            ta = ga[ga["team_id"] == t]
            oa = ga[ga["team_id"] != t]
            xgs = ta.loc[ta["type_id"].isin(shot_ids), xg_column].dropna().to_numpy(dtype="float64")
            goals = int(((ta["type_id"].isin(shot_ids)) & (ta["result_id"] == success)).sum())
            goals += int((oa["result_id"] == owngoal).sum())  # opponent own goal credits this team
            per.append((xgs, goals))
        out.append((per[0][0], per[1][0], per[0][1], per[1][1]))
    return out


def extract_match_slice(actions: pd.DataFrame, xg_column: str = "xg") -> pd.DataFrame:
    """One match's per-game ``(home_xgs, away_xgs, home_goals, away_goals)`` rows as a shard (pure).

    The xg arrays are stored as lists so the shard round-trips through parquet; the reduce
    (``matches_from_shards``) rebuilds the ``fit_rho`` tuples. An empty frame (non-two-team match) is a
    valid shard: "ran, produced no scoreline" (ADR-052)."""
    rows = [
        {
            "home_xgs": [float(x) for x in hx],
            "away_xgs": [float(x) for x in ax],
            "home_goals": int(hg),
            "away_goals": int(ag),
        }
        for hx, ax, hg, ag in _match_tuples(actions, xg_column)
    ]
    return pd.DataFrame(rows, columns=["home_xgs", "away_xgs", "home_goals", "away_goals"])


def matches_from_shards(frames: Iterable[pd.DataFrame]) -> list[tuple]:
    """Reconstruct the ``fit_rho`` tuple list from per-match shards (the whole-corpus reduce). Order
    does not matter -- ``fit_rho`` sums the per-match NLL, so a resumed/partitioned run fits the same
    rho as the pre-migration in-memory pass."""
    out: list[tuple] = []
    for f in frames:
        for r in f.to_dict("records"):
            out.append(
                (
                    np.asarray(r["home_xgs"], dtype="float64"),
                    np.asarray(r["away_xgs"], dtype="float64"),
                    int(r["home_goals"]),
                    int(r["away_goals"]),
                )
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="silly_kicks/match_outcome/weights")
    ap.add_argument("--competitions-json", default=None, help="JSON [[competition_id, season_id], ...]")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument(
        "--shard-root",
        default=None,
        help="per-match shard dir (default: a temp dir OUTSIDE the repo; resumable across runs).",
    )
    ap.add_argument(
        "--cache-dir", default=None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"
    )
    ap.add_argument("--allow-dirty", action="store_true", help="dev only; artifact records dirty:true")
    args = ap.parse_args()

    from scripts._driver import for_each
    from scripts._provenance import git_provenance, require_clean_tree
    from scripts._sb_open_data import all_open_competitions, open_data_source

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    competitions = (
        [tuple(c) for c in json.loads(Path(args.competitions_json).read_text(encoding="utf-8"))]
        if args.competitions_json
        else all_open_competitions()
    )
    # ADR-052 resume: shard each match's scoreline slice behind for_each's resume check (a 3,961-match
    # load re-downloaded everything on a crash before this). The whole-corpus rho fit is the reduce.
    refs, load = open_data_source(competitions, max_matches=args.max_matches, cache_dir=args.cache_dir)
    shard_root = (
        Path(args.shard_root) if args.shard_root else Path(tempfile.gettempdir()) / "sk_match_outcome_dependence_shards"
    )
    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=lambda item: extract_match_slice(item[2], "xg"),
        shard_root=shard_root,
        token_inputs={"model": "match_outcome_dependence", "competitions": sorted(tuple(c) for c in competitions)},
        label="match",
    )
    shards = [pd.read_parquet(s) for s in sorted(res.shard_dir.glob("*.parquet"))]
    matches = matches_from_shards(shards)

    rho = fit_rho(matches)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "rho": rho,
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "corpus": "statsbomb-open",
        "n_matches": len(matches),
    }
    model_path = out / "model.json"
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{sha}  model.json\n", encoding="utf-8")
    print(json.dumps({**payload, "sha256": sha}, indent=2))


if __name__ == "__main__":
    main()
