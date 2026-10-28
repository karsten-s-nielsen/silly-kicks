"""Fit the TF-53 Dixon-Coles dependence rho on the PUBLIC StatsBomb open-data corpus (owner-run).

1-D MLE (scipy ``minimize_scalar``, bounded) of the realized scoreline under the tau-corrected
Poisson-binomial joint (spec section 5). Writes a pickle-free ``model.json`` + ``SHA256SUMS`` stamping
``training_commit`` (ADR-037). ``statsbombpy`` is a scripts-only network dep; no Optuna (the likelihood is
smooth and one-dimensional).

ADR-128 stage contract (HFU R3-10 / D12; it replaced the one-unit adapter): ``refs`` (the competition set +
match references, one unit) -> ``extract[match]`` (an ``ItemStage``: one scoreline slice per open-data match,
fetched per unit, resume-before-download) -> ``fit`` (one unit: ``matches_from_shards`` in sorted key order,
the legacy sorted shard glob, -> ``fit_rho`` -> the bundle FileSet). ``--run-all --out <DIR>`` writes
``model.json`` + ``SHA256SUMS`` into ``<DIR>``; ``adopt`` copies them into ``silly_kicks/match_outcome/weights/``.
The legacy CLI is kept verbatim as the test oracle (``tests/scripts/_legacy_trainers``).

Usage (StatsBomb credentials UNSET so open-data mode is used; every open competition unless
``--competitions-json``):
    python -m scripts.train_match_outcome_dependence --store <DIR> --run-all --out <DIR>
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, cli_main


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
    from scipy.optimize import OptimizeResult, minimize_scalar  # scipy is fit-time only (never at serve)

    # `minimize_scalar` is untyped; whole-repo pyright (scipy 1.18) can infer its result as `object`, so the
    # result type is declared here. Runtime is unchanged.
    res = cast(OptimizeResult, minimize_scalar(lambda r: _nll(r, matches), bounds=bounds, method="bounded"))
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


#: The columns one match's scoreline shard carries (``extract_match_slice``).
_SHARD_COLUMNS = ["home_xgs", "away_xgs", "home_goals", "away_goals"]

_PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param(
        "competitions_json",
        str,
        None,
        help="JSON file [[competition_id, season_id], ...] (default: every open-data competition)",
    ),
    Param("max_matches", int, None, help="cap the number of matches"),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)


def _competitions_file(params) -> list[tuple[int, int]] | None:
    """The ``--competitions-json`` competitions in file order, or None (every open competition)."""
    path = params["competitions_json"]
    if path is None:
        return None
    return [(int(c[0]), int(c[1])) for c in json.loads(Path(path).read_text(encoding="utf-8"))]


def _refs_work(key, ctx) -> dict:
    """The competition set + every match reference, as one JSON unit (the legacy ``open_data_source`` walk:
    the competitions in their given order, so a ``--max-matches`` cap selects the matches the legacy run did)."""
    import scripts._sb_open_data as sbod

    sbod.assert_statsbomb_open_data_mode()
    comps = _competitions_file(ctx.params)
    if comps is None:
        comps = [(int(c[0]), int(c[1])) for c in sbod.all_open_competitions()]
    refs = sbod.list_open_data_refs(comps, max_matches=ctx.params["max_matches"])
    return {"competitions": [list(c) for c in sorted(comps)], "refs": [dataclasses.asdict(r) for r in refs]}


def _extract_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def _load_match(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))


def _extract_work(item, ctx) -> pd.DataFrame:
    return extract_match_slice(item.actions, "xg")


def _extract_reduce(results, ctx) -> pd.DataFrame:
    from scripts._stages import combine_frames

    frames = list(results.values())  # sorted key order == the legacy sorted shard glob
    return combine_frames(frames, ids=()) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


def _fit_work(key, ctx) -> FileSet:
    """The legacy reduce: the whole-corpus rho fit and the bundle, byte for byte (``model.json`` keeps the
    legacy key order and ``indent=2``; ``SHA256SUMS`` hashes those exact bytes)."""
    matches = matches_from_shards([ctx.upstream("extract")])
    rho = fit_rho(matches)
    prov = ctx.prov
    payload = {
        "rho": rho,
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "corpus": "statsbomb-open",
        "n_matches": len(matches),
    }
    model = json.dumps(payload, indent=2).encode("utf-8")
    sha = hashlib.sha256(model).hexdigest()
    print(json.dumps({**payload, "sha256": sha}, indent=2))
    return FileSet({"model.json": model, "SHA256SUMS": f"{sha}  model.json\n".encode()})


#: HFU plan R3-8 S3 item 12: the reviewed serial reasons of this pipeline's SERIAL stages, verbatim from the
#: plan's SERIAL annex (owner ruling 2026-10-30, "too small to pay"); `scripts/hf_jobs/serial_reasons.py`
#: holds the same texts and `tests/scripts/test_serial_reasons.py` holds the two equal.
_SERIAL_REASONS = {
    "refs": (
        "Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 3 s of CPU in a 7 s wall, "
        "listing the match references."
    ),
    "fit": ("Too small to pay (under 60 s of unit CPU, owner ruling 2026-10-30): one unit, 4 s of CPU."),
}


PIPELINE = Pipeline(
    name="match_outcome_dependence",
    module="scripts.train_match_outcome_dependence",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            serial_reason=_SERIAL_REASONS["refs"],  # S3 item 12 (SERIAL annex)
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=("max_matches",),
            # the competitions file is declared by CONTENT, never by path (the 4.77.1 stale-shard rule)
            declare=lambda p: {"competitions": _competitions_file(p)},
            uses_dataset=True,
            dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
        ),
        ItemStage(
            name="extract",
            refs=_extract_refs,
            key=lambda ref: ref.key,
            load=_load_match,
            work=_extract_work,
            needs=("refs",),
            reduce=_extract_reduce,
            declare=lambda p: {"model": "match_outcome_dependence", "xg_column": "xg", "cols": list(_SHARD_COLUMNS)},
            cross_commit_version="match-outcome-dependence-shard-1",
            uses_dataset=True,
            max_failures=None,  # legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="fit",
            serial_reason=_SERIAL_REASONS["fit"],  # S3 item 12 (SERIAL annex)
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("extract",),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="fit",
)

if __name__ == "__main__":
    cli_main(PIPELINE)
