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

``--out`` is a run directory OUTSIDE the repo; copy ``<out>/weights`` into
``silly_kicks/win_probability/weights/`` for the bundled-weights commit (commit-2).

Usage (on the box, scripts/ on sys.path, StatsBomb credentials UNSET so open-data mode is used):
  python scripts/train_win_probability.py --out <DIR> [--all-competitions | --competition-id N --season-id N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="run dir OUTSIDE the repo (shards + weights/ + metrics.json)")
    ap.add_argument("--competition-id", type=int, default=None, help="open-data competition (default 43 = World Cup)")
    ap.add_argument("--season-id", type=int, default=None, help="open-data season (default 106 = 2022)")
    ap.add_argument("--all-competitions", action="store_true", help="full public open-data corpus (every release)")
    ap.add_argument("--max-matches", type=int, default=None, help="cap the number of matches")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact marked dirty)")
    args = ap.parse_args(argv)

    if args.all_competitions and (args.competition_id is not None or args.season_id is not None):
        ap.error("--all-competitions is mutually exclusive with --competition-id/--season-id")

    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    import scripts._sb_open_data as sbod
    from scripts._driver import for_each
    from silly_kicks.win_probability import WinProbabilityModel, WinProbabilityParams

    params = WinProbabilityParams.default()
    dest = Path(args.out)

    # Match-date map (chronology for the leakage-free strength) + the match iterator, both over the
    # SAME competition set. sb.matches carries match_date; the loader does not, so fetch it here.
    from statsbombpy import sb  # type: ignore[import-not-found]

    sbod.assert_statsbomb_open_data_mode()
    if args.all_competitions:
        comps = [(int(c[0]), int(c[1])) for c in sbod.all_open_competitions()]
    else:
        comps = [
            (
                _DEFAULT_COMPETITION_ID if args.competition_id is None else args.competition_id,
                _DEFAULT_SEASON_ID if args.season_id is None else args.season_id,
            )
        ]

    date_map: dict[str, str] = {}
    for cid, sid in comps:
        for mid, m in sb.matches(competition_id=cid, season_id=sid, fmt="dict").items():
            date_map[str(mid)] = str(m.get("match_date", ""))

    def _matches():
        for cid, sid in comps:
            yield from sbod.load_open_data_matches(competition_id=cid, season_id=sid, max_matches=args.max_matches)

    def _work(item):
        _provider, mid, actions, _frames, home = item
        return match_shard(actions, home, date_map.get(str(mid), ""))

    res = for_each(
        _matches(),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=dest / "shards",
        token_inputs={"model": "WinProbabilityModel", "cols": list(_SHARD_COLUMNS), "competitions": sorted(comps)},
        label="match",
    )

    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    pooled = (
        pd.concat([pd.read_parquet(s) for s in shard_files], ignore_index=True)
        if shard_files
        else pd.DataFrame(columns=_SHARD_COLUMNS)
    )
    if not len(pooled):
        raise SystemExit("no action rows collected from the corpus; nothing to fit")

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

    wdir = dest / "weights"
    model.save(wdir)

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
        **res.manifest(),
    }
    # metrics.json ships INSIDE weights/ (so it commits with the bundle and test_bundled's calibration +
    # expected-goals gates read it from the shipped artifact); a copy in dest/ for the run log.
    payload = json.dumps(out, indent=2, default=str)
    (wdir / "metrics.json").write_text(payload, encoding="utf-8")
    (dest / "metrics.json").write_text(payload, encoding="utf-8")
    (wdir / "MODEL_CARD.md").write_text(render_model_card(out), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"bundled weights + MODEL_CARD.md -> {wdir}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
