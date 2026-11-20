"""Fit the TF-53 Dixon-Coles dependence rho on the PUBLIC StatsBomb open-data corpus (owner-run).

1-D MLE (scipy ``minimize_scalar``, bounded) of the realized scoreline under the tau-corrected
Poisson-binomial joint (spec section 5). Writes a pickle-free ``{model.json, SHA256SUMS}`` stamping
``training_commit`` on a CLEAN tree (ADR-037). ``statsbombpy`` is a scripts-only network dep; no Optuna
(the likelihood is smooth and one-dimensional).

An ADR-128 stage pipeline (HFU spec section 5): ``refs`` (the competition set + every match ref) -> ``extract`` (one
unit per match: its scoreline slice) -> ``fit`` (the whole-corpus rho). The pure kernels below are unchanged
from the pre-pipeline trainer; ``validate_match_outcome_calibration.py`` and the tests import them.

Usage (owner):
    python -m scripts.train_match_outcome_dependence --run-all --store <store> --out silly_kicks/match_outcome/weights
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main


def _nll(rho: float, matches: Sequence[tuple]) -> float:
    """Negative log-likelihood of realized scorelines under the tau(rho)-corrected PB joint."""
    from silly_kicks.match_outcome import apply_dependence, goal_count_pmf

    total = 0.0
    for home_xgs, away_xgs, hg, ag in matches:
        joint = apply_dependence(goal_count_pmf(home_xgs), goal_count_pmf(away_xgs), rho=float(rho))
        p = joint[hg, ag] if hg < joint.shape[0] and ag < joint.shape[1] else 0.0
        total -= float(np.log(max(p, 1e-12)))
    return total


#: The rho resolution of the bounded fit (scipy's default ``xatol`` for ``method="bounded"``, stated so that the
#: acceptance gate reads the same number).
FIT_XATOL = 1e-5


def fit_rho(matches: Sequence[tuple], *, bounds: tuple[float, float] = (-0.5, 0.5)) -> float:
    """MLE of the Dixon-Coles rho over ``(home_xgs, away_xgs, home_goals, away_goals)`` tuples."""
    from scipy.optimize import OptimizeResult, minimize_scalar  # scipy is fit-time only (never at serve)

    res = cast(
        OptimizeResult,
        minimize_scalar(lambda r: _nll(r, matches), bounds=bounds, method="bounded", options={"xatol": FIT_XATOL}),
    )
    return float(res.x)


#: The folds of the out-of-fold likelihood that the acceptance detail reports (information, not a gate).
ACCEPTANCE_FOLDS = 5
#: A fit within this distance of a bound counts as stuck at the bound.
BOUND_MARGIN = 1e-3
#: The fit check probes the likelihood this far to each side of the fresh rho: twice the fit's own tolerance, so a fit
#: within its tolerance of the optimum passes and a fit farther from it fails.
OPTIMUM_PROBE = 2 * FIT_XATOL
#: The relative rounding noise of the likelihood (measured: 4e-11 on 10544 between match orders); a drop must exceed it.
NLL_NOISE_RTOL = 1e-12


def acceptance_gates(
    matches: Sequence[tuple],
    *,
    fresh_rho: float,
    committed_rho: float,
    bounds: tuple[float, float] = (-0.5, 0.5),
    folds: int = ACCEPTANCE_FOLDS,
) -> tuple[dict[str, bool], dict[str, float]]:
    """The acceptance gates of a refit (ADR-130: the fresh rho replaces the committed one on purpose).

    ``rho_inside_bounds``: the fresh rho is not stuck at a bound of the bounded fit.
    ``fit_reaches_the_likelihood_optimum``: a FIT CHECK (owner ruling R106 of the action-time repair spec), which tests
    the optimum directly and does not depend on the committed rho. The negative log-likelihood must not drop when rho
    moves ``OPTIMUM_PROBE`` (twice the fit's tolerance ``FIT_XATOL``) to either side inside the bounds, beyond the
    rounding noise. A fit within ``FIT_XATOL`` of the optimum passes in any match order; a fit farther than
    ``FIT_XATOL`` from it fails. It never fails on a change of the data. The detail reports, as information only,
    the likelihood under the committed rho and under independence (rho 0), and the out-of-fold likelihood of the fit
    (each fold scored with the rho fitted on the other folds; folds round-robin over the corpus order).
    """
    lo, hi = bounds
    fresh = _nll(fresh_rho, matches)
    neighbours = [r for r in (fresh_rho - OPTIMUM_PROBE, fresh_rho + OPTIMUM_PROBE) if lo <= r <= hi]
    drop = max((fresh - _nll(r, matches) for r in neighbours), default=0.0)
    oof = 0.0
    for k in range(folds):
        train = [m for i, m in enumerate(matches) if i % folds != k]
        test = [m for i, m in enumerate(matches) if i % folds == k]
        if test:
            oof += _nll(fit_rho(train, bounds=bounds), test)
    gates = {
        "rho_inside_bounds": bool(lo + BOUND_MARGIN < fresh_rho < hi - BOUND_MARGIN),
        "fit_reaches_the_likelihood_optimum": bool(drop <= NLL_NOISE_RTOL * max(1.0, abs(fresh))),
    }
    detail = {
        "nll_fresh": fresh,
        "nll_drop_at_neighbours": drop,
        "nll_committed": _nll(committed_rho, matches),
        "nll_rho_zero": _nll(0.0, matches),
        "oof_nll_fresh": oof,
        "committed_rho": float(committed_rho),
    }
    return gates, detail


_SHOT_NAMES = ("shot", "shot_penalty", "shot_freekick")


def _match_tuples(actions: pd.DataFrame, xg_column: str) -> list[tuple]:
    """Per-match ``(home_xgs, away_xgs, home_goals, away_goals)`` from SPADL actions with injected xg."""
    from silly_kicks.id_compat import ids_match
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.spadl._goals import goal_mask, owngoal_mask

    shot_ids = [spadlconfig.actiontype_id[n] for n in _SHOT_NAMES]
    out: list[tuple] = []
    for _gid, ga in actions.groupby("game_id", sort=False):
        # A shoot-out (period 5) row is no part of the match, for its shots and its goals; nor is a row with no period.
        ga = ga[(ga["period_id"].notna() & ga["period_id"].ne(5)).fillna(False).to_numpy(dtype=bool)]
        teams = list(pd.unique(ga["team_id"].dropna()))
        if len(teams) != 2:
            continue
        # The library's goal rule; a row with a missing result is no goal.
        scored = goal_mask(ga).fillna(False).astype(bool)
        own = owngoal_mask(ga).fillna(False).astype(bool)
        per = []
        for t in teams:
            is_t = ids_match(ga["team_id"], t)
            ta = ga[is_t]
            xgs = ta.loc[ta["type_id"].isin(shot_ids), xg_column].dropna().to_numpy(dtype="float64")
            goals = int((scored & is_t).sum())
            goals += int((own & ~is_t & ga["team_id"].notna()).sum())  # opponent own goal credits this team
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


#: The columns of one match's scoreline slice (`extract_match_slice`).
_SLICE_COLUMNS = ["home_xgs", "away_xgs", "home_goals", "away_goals"]

_PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param("competitions_json", str, None, help="JSON file [[competition_id, season_id], ...] (default: every release)"),
    Param("max_matches", int, None, help="cap the number of matches"),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)


def _competitions_file(path: str | None) -> list[list[int]] | None:
    """The CONTENT of ``--competitions-json``, in file order (the legacy walk order); None without one."""
    if not path:
        return None
    return [[int(c), int(s)] for c, s in json.loads(Path(path).read_text(encoding="utf-8"))]


def _refs_work(key, ctx) -> dict:
    """The competition set + every match reference, as one JSON unit (the legacy `open_data_source` walk)."""
    import scripts._sb_open_data as sbod

    sbod.assert_statsbomb_open_data_mode()
    listed = _competitions_file(ctx.params["competitions_json"])
    comps = [tuple(c) for c in listed] if listed is not None else [tuple(c) for c in sbod.all_open_competitions()]
    refs = sbod.list_open_data_refs(comps, max_matches=ctx.params["max_matches"])
    return {"competitions": [list(c) for c in comps], "refs": [dataclasses.asdict(r) for r in refs]}


def _extract_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def _load_match(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    import scripts._sb_open_data as sbod
    from scripts._driver_sources import resolve_cache_dir

    sbod.assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return sbod.load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))


def _extract_reduce(results: Mapping[str, pd.DataFrame], ctx) -> pd.DataFrame:
    """The per-match slices in sorted key order (the legacy sorted shard glob); none -> an empty slice."""
    from scripts._stages import combine_frames

    frames = list(results.values())
    return combine_frames(frames, ids=()) if frames else pd.DataFrame(columns=_SLICE_COLUMNS)


def fit_artifact(pooled: pd.DataFrame, prov: Mapping[str, Any], *, match_ids: Sequence[str]) -> FileSet:
    """The ``{model.json, SHA256SUMS, metrics.json}`` artifact from the pooled scoreline slices: the legacy fit
    and its exact serialization (``json.dumps(indent=2)``, no trailing newline; ``SHA256SUMS`` names
    ``model.json``), plus ``metrics.json``: ``rho`` and the corpus identity of ``match_ids`` (the listed refs),
    which verify compares with the lock (HFU spec 14.3)."""
    from scripts._corpus_identity import open_data_identity
    from silly_kicks import match_outcome
    from silly_kicks.match_outcome._dependence import DependenceModel

    matches = matches_from_shards([pooled])
    payload = {
        "rho": fit_rho(matches),
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "corpus": "statsbomb-open",
        "n_matches": len(matches),
    }
    model = json.dumps(payload, indent=2).encode("utf-8")
    sha = hashlib.sha256(model).hexdigest()
    print(json.dumps({**payload, "sha256": sha}, indent=2))
    committed_rho = DependenceModel.load(Path(match_outcome.__file__).parent / "weights").rho
    gates, detail = acceptance_gates(matches, fresh_rho=payload["rho"], committed_rho=committed_rho)
    metrics = json.dumps(
        {"rho": payload["rho"], "acceptance": gates, "acceptance_detail": detail, **open_data_identity(match_ids)},
        indent=2,
    ).encode("utf-8")
    return FileSet({"model.json": model, "SHA256SUMS": f"{sha}  model.json\n".encode("ascii"), "metrics.json": metrics})


PIPELINE = Pipeline(
    name="match_outcome_dependence",
    module="scripts.train_match_outcome_dependence",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=("competitions_json", "max_matches"),
            # The CONTENT of the competitions file keys the token (a path alone would not key a changed file).
            declare=lambda p: {"competitions": _competitions_file(p["competitions_json"])},
            uses_dataset=True,
            dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
        ),
        ItemStage(
            name="extract",
            refs=_extract_refs,
            key=lambda ref: ref.key,
            load=_load_match,
            work=lambda lm, ctx: extract_match_slice(lm.actions, "xg"),
            needs=("refs",),
            reduce=_extract_reduce,
            cross_commit_version="match-outcome-shard-2",
            uses_dataset=True,
            max_failures=None,  # the legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=lambda key, ctx: fit_artifact(
                ctx.upstream("extract"), ctx.prov, match_ids=[r["match_id"] for r in ctx.upstream("refs")["refs"]]
            ),
            needs=("refs", "extract"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="fit",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
