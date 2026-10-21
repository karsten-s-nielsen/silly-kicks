"""Trainer: bundle the default WinProbabilityModel weights (TF-63, ADR-101).

Fits ``silly_kicks.win_probability.WinProbabilityModel`` on the PUBLIC StatsBomb open-data corpus and
writes the pickle-free JSON + SHA256SUMS artifact via ``model.save(...)``, plus ``metrics.json``
(``training_commit``, calibration ``ece`` / ``reliability_slope``, ``corpus_goals_per_match`` +
``expected_goals_tol`` for the sec. 9.1 expected-goals gate) and a ``MODEL_CARD.md`` (ADR-088). Inference
imports no sklearn; sklearn is used only during the fit. v1 ships the base GLM+chain (the isotonic
layer is reserved in the model infra, spec sec. 5.1, and fit only if the base OOF ece exceeds the gate).

The expensive per-match corpus load is sharded with ``for_each`` (ADR-052): one shard per match holding
that match's win-prob action rows (the columns ``WinProbabilityModel.fit`` reads) + ``match_date`` +
``home_team_id`` + per-shot ``xg`` (for the leakage-free strength). The chronological strength rating,
the pooled fit, the GroupKFold-by-match out-of-fold calibration, and ``save`` happen in the reduce.

The strength rating is **leakage-free by DATE** (spec sec. 5.2): ``base_strength(match m, team t)`` = the
mean xG-supremacy (``xg_for - xg_against``) over team ``t``'s matches STRICTLY EARLIER than ``m``.

ADR-128 stage contract: ``refs`` (the competition set + match references, one unit) ->
``extract[match]`` (an ``ItemStage``, one shard per open-data match, fetched per unit: resume-before-
download) -> ``fit`` (one unit: strength, calibration, fit, gates, the bundle FileSet). The uniform CLI
comes from ``scripts._stages.main``; ``--run-all --out <DIR>`` is the serial parity reference and writes
the four bundle files into ``<DIR>`` (copy them into ``silly_kicks/win_probability/weights/``; ``adopt``
does this).

Usage (StatsBomb credentials UNSET so open-data mode is used):
  python -m scripts.train_win_probability --store <DIR> --run-all --out <DIR> [--all-competitions |
      --competition-id N --season-id N]
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main

_SHOT_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_SUCCESS = spadlconfig.result_id["success"]
_OWNGOAL = spadlconfig.result_id["owngoal"]

#: Columns one match's shard carries -- exactly what WinProbabilityModel.fit reads (via games +
#: strength_column) plus match_date (chronology) + home_team_id + per-shot xg (strength).
_SHARD_COLUMNS = [
    "game_id",
    "action_id",
    "period_id",
    "team_id",
    "player_id",
    "time_seconds",
    "type_id",
    "result_id",
    "xg",
    "home_team_id",
    "match_date",
]


def match_shard(actions: pd.DataFrame, home_team_id: int | str, match_date: str) -> pd.DataFrame:
    """One match's win-prob action rows + constants (pure). ``action_id`` is REQUIRED by
    ``compute_win_probability`` (a WIN_PROBABILITY_KEYS column) -- omitting it is what the DGX
    validation run surfaced as a KeyError the synthetic fixtures never hit."""
    cols = [
        c
        for c in (
            "game_id",
            "action_id",
            "period_id",
            "team_id",
            "player_id",
            "time_seconds",
            "type_id",
            "result_id",
            "xg",
        )
        if c in actions.columns
    ]
    out = actions[cols].copy()
    if "player_id" not in out.columns:
        out["player_id"] = np.nan
    if "xg" not in out.columns:
        out["xg"] = np.nan
    out["home_team_id"] = home_team_id
    out["match_date"] = str(match_date)
    return out.reset_index(drop=True)[_SHARD_COLUMNS]


def _team_final_goals(g: pd.DataFrame) -> dict:
    """Final id-based goal count per team for one match (scored shots + own goals credited to opp)."""
    teams = list(pd.unique(g["team_id"].dropna()))
    goals = {t: 0 for t in teams}
    is_scored = g["type_id"].isin(_SHOT_IDS) & (g["result_id"] == _SUCCESS)
    is_og = g["result_id"] == _OWNGOAL
    tcol = g["team_id"].to_numpy()
    sc = is_scored.to_numpy()
    og = is_og.to_numpy()
    for i in range(len(g)):
        t = tcol[i]
        if sc[i]:
            goals[t] = goals.get(t, 0) + 1
        elif og[i]:
            opp = [x for x in teams if x != t]
            if len(opp) == 1:
                goals[opp[0]] = goals.get(opp[0], 0) + 1
    return goals


def chronological_strength(pooled: pd.DataFrame) -> dict:
    """{(game_id, team_id): base_strength} = mean xG-supremacy over the team's STRICTLY-EARLIER matches.

    Leakage-free by match_date (spec sec. 5.2): a match's strength uses only prior matches. Ties on date
    keep insertion (manifest) order, still strictly-earlier-only within the running accumulator.
    """
    # per (game, team): xg_for, and the match_date; opponent's xg_for is xg_against.
    per = []
    for gid, g in pooled.groupby("game_id", sort=False):
        teams = list(pd.unique(g["team_id"].dropna()))
        date = str(g["match_date"].iloc[0])
        shots = g[g["type_id"].isin(_SHOT_IDS)]
        xg_for = {t: float(shots[shots["team_id"] == t]["xg"].fillna(0.0).sum()) for t in teams}
        for t in teams:
            opp = [x for x in teams if x != t]
            xa = xg_for.get(opp[0], 0.0) if len(opp) == 1 else 0.0
            per.append({"game_id": gid, "team_id": t, "date": date, "supremacy": xg_for.get(t, 0.0) - xa})
    df = pd.DataFrame(per).sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    out: dict = {}
    running: dict = {}  # team -> [sum, n] over strictly-earlier matches
    # process in date order; within a date, all same-date matches see only strictly-earlier (prior dates)
    for _date, block in df.groupby("date", sort=True):
        for _, r in block.iterrows():
            s, n = running.get(r["team_id"], (0.0, 0))
            out[(r["game_id"], r["team_id"])] = s / n if n else 0.0
        for _, r in block.iterrows():
            s, n = running.get(r["team_id"], (0.0, 0))
            running[r["team_id"]] = (s + float(r["supremacy"]), n + 1)
    return out


def _actual_win(pooled: pd.DataFrame) -> dict:
    """{(game_id, team_id): 1.0 if team won the match else 0.0} (binary win-indicator for calibration)."""
    out: dict = {}
    for gid, g in pooled.groupby("game_id", sort=False):
        goals = _team_final_goals(g)
        teams = list(goals)
        for t in teams:
            opp = [x for x in teams if x != t]
            og = goals.get(opp[0], 0) if len(opp) == 1 else 0
            out[(gid, t)] = 1.0 if goals[t] > og else 0.0
    return out


def _attach_strength(pooled: pd.DataFrame, strength: dict) -> pd.DataFrame:
    out = pooled.copy()
    out["base_strength"] = [strength.get((gid, t), 0.0) for gid, t in zip(out["game_id"], out["team_id"], strict=False)]
    return out


def _games_frame(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"game_id": gid, "home_team_id": g["home_team_id"].iloc[0]} for gid, g in pooled.groupby("game_id", sort=False)
    ]
    return pd.DataFrame(rows)


def calibration_metrics(pooled: pd.DataFrame, games: pd.DataFrame, params) -> dict:
    """GroupKFold-by-match OOF calibration of the served per-action p_win vs the match win-indicator.

    Returns ece, reliability_slope (OLS of win-indicator on p_win), brier, n_oof. base_strength is a
    pre-match feature (leakage-free by DATE), so it is the SAME column in every fold -- only the GLM is
    refit per fold.
    """
    from sklearn.model_selection import GroupKFold

    from silly_kicks._calibration_metrics import ece
    from silly_kicks.win_probability import WinProbabilityModel, compute_win_probability

    win = _actual_win(pooled)
    game_ids = pd.unique(pooled["game_id"])
    groups = pooled["game_id"].to_numpy()
    n_splits = min(5, len(game_ids))
    if n_splits < 2:
        return {"ece": float("nan"), "reliability_slope": float("nan"), "brier": float("nan"), "n_oof": 0}

    preds: list[float] = []
    targets: list[float] = []
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(pooled))
    for tr, te in gkf.split(idx, groups=groups):
        train_games = set(pd.unique(pooled.iloc[tr]["game_id"]))
        tr_pool = pooled[pooled["game_id"].isin(train_games)]
        m = WinProbabilityModel(params=params).fit(tr_pool, games=games, strength_column="base_strength")
        te_games = set(pd.unique(pooled.iloc[te]["game_id"]))
        te_pool = pooled[pooled["game_id"].isin(te_games)]
        samples, _ = compute_win_probability(te_pool, model=m, games=games, strength_column="base_strength")
        s = samples.dropna(subset=["p_win"])
        for gid, t, pw in zip(s["game_id"], s["team_id"], s["p_win"], strict=False):
            preds.append(float(pw))
            targets.append(win.get((gid, t), 0.0))

    p = np.asarray(preds)
    y = np.asarray(targets)
    if not len(p):
        return {"ece": float("nan"), "reliability_slope": float("nan"), "brier": float("nan"), "n_oof": 0}
    slope = float(np.polyfit(p, y, 1)[0]) if np.std(p) > 1e-9 else float("nan")
    return {"ece": float(ece(y, p)), "reliability_slope": slope, "brier": float(np.mean((p - y) ** 2)), "n_oof": len(p)}


def render_model_card(metrics: dict) -> str:
    ece = float(metrics.get("ece", float("nan")))
    slope = float(metrics.get("reliability_slope", float("nan")))
    gpm = float(metrics.get("corpus_goals_per_match", float("nan")))
    eg = float(metrics.get("model_expected_goals", float("nan")))
    n_matches = metrics.get("n_matches", "?")
    n_comp = metrics.get("n_competitions", "?")
    commit = metrics.get("training_commit", "?")
    return f"""# In-game win-probability model -- `default` variant (TF-63, ADR-101)

**What it is.** A per-action in-game win-probability model: an interval-hazard logistic on
`(score_diff, minutes_remaining, base_strength, home, man_advantage)` feeds a forward Markov chain on
the score difference. `goal_leverage` reads the exact per-state dP(win | goal); `VAEP.rate_ximpact`
weights `VAEP_adjusted` by it. Logistic at fit, pure-numpy at serve (no runtime sklearn; the isotonic
recalibration layer is reserved and unused in this base-model v1).
Loaded via `silly_kicks.win_probability.WinProbabilityModel.bundled()`.

**Strength.** `base_strength(match, team)` = mean xG-supremacy over the team's STRICTLY-EARLIER matches
(leakage-free by date). At serve it is an injected pre-match prior (odds-derived supremacy); optional,
default even.

**Corpus + metrics.** {n_matches} public StatsBomb open-data matches ({n_comp} (competition, season)
releases). GroupKFold-by-match out-of-fold: ECE {ece:.3f}, reliability slope {slope:.3f}. Corpus
goals/match {gpm:.3f}; model expected goals (0-0, full match) {eg:.3f}. See `metrics.json`.

**Gates.** `certify_coherence` (leverage>=0, monotone in score_diff) passes on these weights; the
expected-goals gate checks the model's expected goals match the corpus rate; calibration `ece<=0.10`
and `|slope-1|<=0.25`.

**Provenance.** `metrics.json` records `training_commit` ({commit}) + tree state (clean). Pickle-free
JSON + SHA256 + feature-contract probe; `load()` is fail-closed (chirality N/A -- no geometric
features). Every bundled model carries a card (ADR-088). Attribution: Paul, Klemp & Memmert (2025);
Dixon & Robinson (1998); Robberechts, Van Haaren & Davis (2019) -- see NOTICE.
"""


_DEFAULT_COMPETITION_ID = 43
_DEFAULT_SEASON_ID = 106

_PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param("all_competitions", flag=True, help="full public open-data corpus (every release)"),
    Param("competition_id", int, None, help="open-data competition (default 43 = World Cup)"),
    Param("season_id", int, None, help="open-data season (default 106 = 2022)"),
    Param("max_matches", int, None, help="cap the number of matches"),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)


def _competitions(params) -> list[tuple[int, int]]:
    import scripts._sb_open_data as sbod

    if params["all_competitions"] and (params["competition_id"] is not None or params["season_id"] is not None):
        raise ValueError("--all-competitions is mutually exclusive with --competition-id/--season-id")
    sbod.assert_statsbomb_open_data_mode()
    if params["all_competitions"]:
        return [(int(c[0]), int(c[1])) for c in sbod.all_open_competitions()]
    return [
        (
            _DEFAULT_COMPETITION_ID if params["competition_id"] is None else params["competition_id"],
            _DEFAULT_SEASON_ID if params["season_id"] is None else params["season_id"],
        )
    ]


def _refs_work(key, ctx) -> dict:
    """The competition set + every match reference, as one JSON unit (the legacy listing: ``:259-291`` at
    ``cd2ebcb``)."""
    import scripts._sb_open_data as sbod

    comps = _competitions(ctx.params)
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
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"])), ref.match_date


def _extract_work(item, ctx):
    lm, match_date = item
    # The legacy date_map came from the same sb.matches manifest the ref was built from.
    return match_shard(lm.actions, lm.home_team_id, match_date)


def _extract_reduce(results, ctx):
    from scripts._stages import combine_frames

    frames = list(results.values())  # sorted key order == the legacy sorted shard glob
    return combine_frames(frames, ids=("game_id",)) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


def _fit_work(key, ctx) -> FileSet:
    import tempfile

    from silly_kicks.win_probability import WinProbabilityModel, WinProbabilityParams

    pooled = ctx.upstream("extract")
    if not len(pooled):
        raise RuntimeError("no action rows collected from the corpus; nothing to fit")
    comps = [tuple(c) for c in ctx.upstream("refs")["competitions"]]
    params = WinProbabilityParams.default()

    strength = chronological_strength(pooled)
    pooled = _attach_strength(pooled, strength)
    games = _games_frame(pooled)

    metrics = calibration_metrics(pooled, games, params)
    # v1 ships the BASE GLM+chain (model._isotonic stays None -> identity serve). The metrics.json ece is
    # the HONEST base-model OOF calibration, consistent with what serves. The isotonic layer is reserved
    # in the model infra (spec sec. 5.1); if the full-corpus base OOF ece exceeds ece_max it is fit then,
    # with a proper nested-CV measurement rather than an in-sample-OOF one.
    model = WinProbabilityModel(params=params).fit(pooled, games=games, strength_column="base_strength")
    model.certify_coherence()  # fail-closed: a non-monotone fit must not be bundled

    # corpus expected-goals gate inputs
    n_matches = int(pooled["game_id"].nunique())
    total_goals = sum(sum(_team_final_goals(g).values()) for _, g in pooled.groupby("game_id", sort=False))
    corpus_goals_per_match = total_goals / n_matches if n_matches else float("nan")
    from silly_kicks.win_probability._chain import expected_total_goals

    n_int = params.regulation_minutes // params.interval_minutes

    def hz_home(d, mm):
        return model._hazard(score_diff=d, minutes_remaining=mm, base_strength=0.0, home=True, man_advantage=0)

    def hz_away(d, mm):
        return model._hazard(score_diff=-d, minutes_remaining=mm, base_strength=0.0, home=False, man_advantage=0)

    model_expected_goals = expected_total_goals(hz_home, hz_away, n_intervals=n_int, K=params.lattice_pad)

    prov = ctx.prov
    summary = ctx.summary("extract")
    # Decision P4 (ADR-128): CORPUS-scoped counters (the values a single fresh serial pass reports), never
    # pass-scoped ones, so resume history cannot change metrics.json (INV-1). `generation` is the token.
    pass_fields = {
        "generation": ctx.tokens["extract"],
        "n_attempted": summary["n_units"],
        "n_failed": summary["n_failed"],
        "n_counters_unrecorded": 0,
        "n_excluded": summary["n_excluded"],
    }
    out = {
        **metrics,
        "corpus_goals_per_match": corpus_goals_per_match,
        "model_expected_goals": float(model_expected_goals),
        "expected_goals_tol": 0.5,
        "n_matches": n_matches,
        "n_action_rows": len(pooled),
        "n_competitions": len(comps),
        "competitions": [list(c) for c in sorted(comps)],
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **pass_fields,
    }
    # metrics.json ships INSIDE the bundle (test_bundled's calibration + expected-goals gates read it from
    # the shipped artifact).
    payload = json.dumps(out, indent=2, default=str)
    with tempfile.TemporaryDirectory() as tmp:
        wdir = Path(tmp) / "weights"
        model.save(wdir)
        files = {p.name: p.read_bytes() for p in sorted(wdir.iterdir()) if p.is_file()}
    files["metrics.json"] = payload.encode("utf-8")
    files["MODEL_CARD.md"] = render_model_card(out).encode("utf-8")
    print(payload)
    return FileSet(files)


PIPELINE = Pipeline(
    name="win_probability",
    module="scripts.train_win_probability",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=("all_competitions", "competition_id", "season_id", "max_matches"),
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
            declare=lambda p: {"model": "WinProbabilityModel", "cols": list(_SHARD_COLUMNS)},
            cross_commit_version="win-probability-shard-1",
            uses_dataset=True,
            max_failures=None,  # legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="fit",
            units=lambda ctx: ["only"],
            work=_fit_work,
            needs=("refs", "extract"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="fit",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
