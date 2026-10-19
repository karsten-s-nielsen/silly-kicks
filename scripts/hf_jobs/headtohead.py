"""Head-to-head scoring: committed vs fresh model on the SAME held-out eval, emitted as a pair.

In the job, ``main`` builds the REAL eval from the corpus (``build_eval``: owner-tier from ``--corpus-dir``,
public via statsbombpy) and scores BOTH the committed model (shipped in the cloned Commit-1 ref under
``silly_kicks/<weights_path>/``) and the freshly-fit model (``/tmp/out/weights``) on it, emitting
``{f"committed_{m}": ..., f"fresh_{m}": ...}`` + the platform ``stamp()`` into the run's ``metrics.json``.
The floor (floor.py) then compares the pair. No committed ``metrics.json`` is read (3 public trainers ship
none).

**Eval data is injected, scoring is pure.** Each ``_score_*`` takes an eval BUNDLE and is unit-tested with
a ``synthetic_bundle`` (``tests/scripts/hf_jobs/test_headtohead.py``) so the load + predict + metric + fold
PATH is proven at build time with no corpus / statsbombpy. ``build_eval`` assembles the REAL bundle and is
**fail-loud**: if the data source is unavailable it raises ``EvalDataUnavailableError`` — it NEVER silently
falls back to synthetic data (a synthetic eval would make the no-regression floor vacuous). The real
``build_eval`` execution is first verified at P1 (statsbombpy / owner corpus exist on HF / DGX, not in the
local .venv); the feasibility + fail-loud tests guard everything that can run locally.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


class EvalDataUnavailableError(RuntimeError):
    """The real head-to-head eval corpus is not reachable here (fail-loud; never synthetic in the job)."""


def stamp() -> dict:
    def _v(pkg: str) -> str | None:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "training_platform": f"{platform.system()}-{platform.machine()}",
        "python_version": sys.version.split()[0],
        "sklearn_version": _v("scikit-learn"),
        "xgboost_version": _v("xgboost"),
        "numpy_version": _v("numpy"),
        "numba_version": _v("numba"),
    }


# --- metric helpers --------------------------------------------------------------------------------
def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y, dtype=int), np.asarray(p, dtype=float)))


def log_loss_metric(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def top1(correct: np.ndarray) -> float:
    return float(np.mean(np.asarray(correct, dtype=float)))


def mae_euclidean(pred_xy: np.ndarray, true_xy: np.ndarray) -> float:
    pred_xy = np.asarray(pred_xy, dtype=float)
    true_xy = np.asarray(true_xy, dtype=float)
    return float(np.mean(np.hypot(pred_xy[:, 0] - true_xy[:, 0], pred_xy[:, 1] - true_xy[:, 1])))


# --- shared helpers --------------------------------------------------------------------------------
def _feature_cols(model) -> list[str]:
    """The model's fit-time feature order (public ``feature_names`` / private ``_feature_names()`` / booster)."""
    cols = getattr(model, "feature_names", None)
    if cols:
        return list(cols)
    fn = getattr(model, "_feature_names", None)
    if callable(fn):
        return list(cast("list[str]", fn()))
    booster = getattr(model, "_booster", None)  # xgboost trainers (xshot/xcross) carry names on the booster
    bcols = getattr(booster, "feature_names", None) if booster is not None else None
    if bcols:
        return list(bcols)
    raise AttributeError(f"{type(model).__name__}: no feature_names surface")


def _heldout_fold(game_ids: np.ndarray, fold: int = 0, n_splits: int = 5) -> np.ndarray:
    """Index of ONE GroupKFold-by-match fold (deterministic) — the shared eval split both models score on."""
    from sklearn.model_selection import GroupKFold

    game_ids = np.asarray(game_ids)
    idx = np.arange(len(game_ids))
    n = min(n_splits, len(np.unique(game_ids)))
    if n < 2:
        return idx
    return list(GroupKFold(n_splits=n).split(idx, groups=game_ids))[fold][1]


def _fold_frame(bundle: dict) -> np.ndarray:
    """Row index of the held-out slice for a feature-frame bundle (uses ``groups`` when present)."""
    groups = bundle.get("groups")
    n = len(bundle["features"])
    return _heldout_fold(np.asarray(groups)) if groups is not None else np.arange(n)


# --- per-trainer scorers (pure; score committed + fresh on the SAME injected eval bundle) ----------
def _final_winner(actions: pd.DataFrame) -> dict:
    """Per game: the team that won (more goals), or None on a draw. Goals = scored shots + own goals to opp."""
    import silly_kicks.spadl.config as spadlconfig

    shot_ids = {spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")}
    succ, owngoal = spadlconfig.result_id["success"], spadlconfig.result_id["owngoal"]
    winner: dict = {}
    for gid, grp in actions.groupby(actions["game_id"].astype(str), sort=False):
        teams = list(pd.unique(grp["team_id"].dropna()))
        goals = dict.fromkeys(teams, 0)
        scored = grp["type_id"].isin(shot_ids) & (grp["result_id"] == succ)
        for t in grp.loc[scored, "team_id"]:
            goals[t] = goals.get(t, 0) + 1
        for t in grp.loc[grp["result_id"] == owngoal, "team_id"]:
            opp = [x for x in teams if x != t]
            if opp:
                goals[opp[0]] = goals.get(opp[0], 0) + 1
        if goals:
            mx = max(goals.values())
            top = [t for t, v in goals.items() if v == mx]
            winner[gid] = top[0] if len(top) == 1 else None
    return winner


def _score_win_probability(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.win_probability import WinProbabilityModel, compute_win_probability

    actions, games = bundle["actions"], bundle.get("games")
    strength_column = bundle.get("strength_column")
    winner = _final_winner(actions)
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        model = WinProbabilityModel.load(d)
        samples, _ = compute_win_probability(actions, model=model, games=games, strength_column=strength_column)
        res = samples[samples["p_win"].notna()]
        if len(res) == 0:
            out[f"{tag}_ece"] = float("nan")
            continue
        y = np.array(
            [1.0 if winner.get(str(g)) == t else 0.0 for g, t in zip(res["game_id"], res["team_id"], strict=False)]
        )
        out[f"{tag}_ece"] = ece(y, res["p_win"].to_numpy())
    return out


def _score_pass_completion(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.expected_passing import PassCompletionModel

    ox, oy, tx, ty = (np.asarray(bundle[k], dtype=float) for k in ("ox", "oy", "tx", "ty"))
    y = np.asarray(bundle["y"], dtype=float)
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        m = PassCompletionModel.load(d)
        out[f"{tag}_ece"] = ece(y, np.asarray(m.predict_completion(ox, oy, tx, ty), dtype=float))
    return out


def _score_xsuccess(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.xsuccess import XSuccessModel

    actions, y = bundle["actions"], np.asarray(bundle["y"], dtype=float)
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        m = XSuccessModel.load(d)
        p = np.asarray(m.predict_success(actions), dtype=float)
        mask = np.isfinite(p)
        out[f"{tag}_ece"] = ece(y[mask], p[mask]) if mask.any() else float("nan")
    return out


def _score_match_outcome(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.match_outcome._dependence import DependenceModel

    matches = bundle["matches"]  # list of (home_pmf, away_pmf, home_win 0/1) — one per match
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        model = DependenceModel.load(d)
        p, y = [], []
        for home_pmf, away_pmf, home_win in matches:
            joint = model.apply(np.asarray(home_pmf, dtype=float), np.asarray(away_pmf, dtype=float))
            p.append(float(np.tril(joint, -1).sum()))  # P(home goals > away goals) under the fitted rho
            y.append(float(home_win))
        out[f"{tag}_log_loss"] = log_loss_metric(np.array(y), np.array(p)) if p else float("nan")
    return out


def _score_ghost_gk(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    te = _fold_frame(bundle)
    feats = bundle["features"].iloc[te]
    true_xy = np.asarray(bundle["true_xy"], dtype=float)[te]
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        out[f"{tag}_cv_mae_euclidean"] = mae_euclidean(GhostGkModel.load(d).predict_mean(feats), true_xy)
    return out


def _score_ghost_outfield(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel

    te = _fold_frame(bundle)
    feats = bundle["features"].iloc[te]
    true = np.asarray(bundle["true"], dtype=float)[te]
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        pred = np.asarray(GhostOutfieldModel.load(d).predict_mean(feats), dtype=float)
        out[f"{tag}_cv_mae"] = mae_euclidean(pred, true)  # (x, y) target -> euclidean, matches the trainer
    return out


def _score_xshot(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    te = _fold_frame(bundle)
    feats, y = bundle["features"].iloc[te], np.asarray(bundle["y"])[te]
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        out[f"{tag}_pr_auc"] = pr_auc(y, np.asarray(XShotOccurrenceModel.load(d).predict_proba(feats), dtype=float))
    return out


def _score_xcross(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    te = _fold_frame(bundle)
    feats, y = bundle["features"].iloc[te], np.asarray(bundle["y"])[te]
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        out[f"{tag}_pr_auc"] = pr_auc(y, np.asarray(XCrossAttemptModel.load(d).predict_proba(feats), dtype=float))
    return out


def _score_receiver(committed_dir: Path, fresh_dir: Path, bundle: dict) -> dict:
    from silly_kicks.tracking._receiver import ReceiverModel

    rows = bundle["rows"]  # one row per (decision, candidate): features + label (1 = actual receiver)
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        model = ReceiverModel.load(d)
        hits = []
        for _key, grp in rows.groupby(["game_id", "action_id"], sort=False):
            scores = np.asarray(model.predict_candidates(grp[model.feature_names]), dtype=float)
            labels = np.asarray(grp["label"])
            hits.append(1.0 if labels[int(np.argmax(scores))] == 1 else 0.0)  # top-1 candidate is the receiver
        out[f"{tag}_top1"] = top1(np.array(hits)) if hits else float("nan")
    return out


SCORERS: dict[str, Callable[[Path, Path, dict], dict[str, float]]] = {
    "win_probability": _score_win_probability,
    "pass_completion": _score_pass_completion,
    "match_outcome_dependence": _score_match_outcome,
    "xsuccess": _score_xsuccess,
    "ghost_gk": _score_ghost_gk,
    "ghost_outfield": _score_ghost_outfield,
    "xshot_occurrence": _score_xshot,
    "xcross_attempt": _score_xcross,
    "receiver_model": _score_receiver,
}


# --- synthetic eval bundles (feasibility test ONLY; never used by the job) -------------------------
def _synthetic_feature_frame(committed_dir: Path, n: int, loader) -> pd.DataFrame:
    model = loader(committed_dir)
    cols = _feature_cols(model)
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((n, len(cols))), columns=cols)


def synthetic_bundle(name: str, committed_dir: Path) -> dict:
    """A tiny in-shape eval bundle for the per-trainer feasibility test — NOT real data, test-only."""
    rng = np.random.default_rng(0)
    if name == "win_probability":
        import silly_kicks.spadl.config as spadlconfig

        n = 12
        team = np.where(np.arange(n) % 2 == 0, 1, 2)  # two teams, alternating
        typ = np.full(n, spadlconfig.actiontype_id["pass"])
        typ[4] = spadlconfig.actiontype_id["shot"]  # one scored shot by team 1 -> 1-0
        actions = pd.DataFrame(
            {
                "game_id": ["g1"] * n,
                "action_id": np.arange(n),
                "team_id": team,
                "period_id": np.ones(n, dtype=int),
                "time_seconds": np.linspace(0.0, 2700.0, n),
                "type_id": typ,
                "result_id": np.full(n, spadlconfig.result_id["success"]),
                "player_id": np.arange(n),
                "xg": np.full(n, np.nan),
            }
        )
        games = pd.DataFrame({"game_id": ["g1"], "home_team_id": [1]})
        return {"actions": actions, "games": games}
    if name == "pass_completion":
        return {
            "ox": [20.0, 40.0, 60.0, 80.0],
            "oy": [34.0, 20.0, 48.0, 10.0],
            "tx": [60.0, 70.0, 85.0, 95.0],
            "ty": [34.0, 30.0, 38.0, 20.0],
            "y": [1, 0, 1, 1],
        }
    if name == "xsuccess":
        import silly_kicks.spadl.config as spadlconfig

        k = 8
        return {
            "actions": pd.DataFrame(
                {
                    "type_id": [spadlconfig.actiontype_id["pass"]] * k,
                    "result_id": [spadlconfig.result_id["success"]] * k,
                    "bodypart_id": [spadlconfig.bodypart_id["foot"]] * k,
                    "start_x": np.linspace(20, 80, k),
                    "start_y": np.linspace(10, 60, k),
                    "end_x": np.linspace(30, 95, k),
                    "end_y": np.linspace(20, 50, k),
                    "time_seconds": np.linspace(0, 2700, k),
                    "period_id": [1] * k,
                }
            ),
            "y": (np.arange(k) % 2),
        }
    if name == "match_outcome_dependence":
        from scipy.stats import poisson

        k = np.arange(11)
        matches = [
            (poisson.pmf(k, 1.6), poisson.pmf(k, 1.0), 1),
            (poisson.pmf(k, 0.8), poisson.pmf(k, 1.4), 0),
            (poisson.pmf(k, 1.2), poisson.pmf(k, 1.2), 0),
        ]
        return {"matches": matches}
    if name == "ghost_gk":
        from silly_kicks.tracking._ghost_gk import GhostGkModel

        feats = _synthetic_feature_frame(committed_dir, 16, GhostGkModel.load)
        return {"features": feats, "true_xy": rng.random((16, 2)) * 10.0, "groups": np.arange(16) % 4}
    if name == "ghost_outfield":
        from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel

        feats = _synthetic_feature_frame(committed_dir, 16, GhostOutfieldModel.load)
        return {"features": feats, "true": rng.random((16, 2)) * 10.0, "groups": np.arange(16) % 4}
    if name == "xshot_occurrence":
        from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

        feats = _synthetic_feature_frame(committed_dir, 16, XShotOccurrenceModel.load)
        return {"features": feats, "y": np.arange(16) % 2, "groups": np.arange(16) % 4}
    if name == "xcross_attempt":
        from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

        feats = _synthetic_feature_frame(committed_dir, 16, XCrossAttemptModel.load)
        return {"features": feats, "y": np.arange(16) % 2, "groups": np.arange(16) % 4}
    if name == "receiver_model":
        from silly_kicks.tracking._receiver import ReceiverModel

        rows = _synthetic_feature_frame(committed_dir, 5, ReceiverModel.load)  # 5 candidates, one decision
        rows["game_id"] = "g1"
        rows["action_id"] = 0
        rows["label"] = 0
        rows.loc[0, "label"] = 1  # candidate 0 is the actual receiver
        return {"rows": rows}
    raise KeyError(name)


# --- real eval builders (job path; fail-loud; first verified at P1 on HF/DGX) ----------------------
def _require_statsbombpy() -> None:
    try:
        import statsbombpy  # noqa: F401  # type: ignore[import-not-found]  # absent on CI (importorskip-only)
    except ImportError as e:  # pragma: no cover - statsbombpy absent in the local .venv
        raise EvalDataUnavailableError(
            "public head-to-head eval needs statsbombpy (open data); unavailable in the local .venv. "
            "Runs at P1 on HF where [train] + statsbombpy are installed."
        ) from e


def _require_corpus(corpus_dir: Path | None) -> Path:
    if corpus_dir is None or not Path(corpus_dir).exists():
        raise EvalDataUnavailableError(
            f"owner-tier head-to-head eval needs the staged corpus (corpus_dir={corpus_dir!r}); "
            "absent locally. Runs at P1/P2 on HF after stage.py pushes it."
        )
    return Path(corpus_dir)


def build_eval(name: str, corpus_dir: Path | None, *, feature_set: str = "faithful") -> dict:
    """Assemble the REAL eval bundle (the one the floor's δ comparison is meaningful on).

    Fail-loud: raises ``EvalDataUnavailableError`` when the data source is not reachable — the job must
    never score the head-to-head on synthetic data (that would make the no-regression floor vacuous).
    Delegates to the trainers' own feature extractors / loaders. First executed + verified at P1 on HF
    (statsbombpy) and P2 on HF/DGX (owner corpus); the local suite proves the scoring path with
    ``synthetic_bundle`` and proves THIS stays fail-loud when the source is absent.

    ``feature_set`` selects the eval feature vintage for the xgboost occurrence classifiers
    (xshot/xcross), so a velocity-free ``position_only`` booster is scored on a matching frame
    (``predict_proba`` is strict on ``feature_names``; ADR-123 / spec §5). Ignored by the public evals
    and the ghost/receiver builders (ghost's ``predict_mean`` reindexes to the model's own names).
    """
    from scripts.hf_jobs.registry import get

    spec = get(name)
    if spec.certification == "public":  # ADR-128: the certification policy (the `source` field is gone)
        _require_statsbombpy()
        return _build_public_eval(name)
    return _build_owner_eval(name, _require_corpus(corpus_dir), feature_set=feature_set)


def _public_eval_win_probability() -> dict:  # pragma: no cover - needs statsbombpy + scripts/ (P1 on HF)
    """Real open-data eval: a few StatsBomb open matches -> the trainer's own win-prob action rows +
    leakage-free strength + games frame (reuses scripts._sb_open_data + scripts.train_win_probability)."""
    import scripts._sb_open_data as sbod
    from scripts.train_win_probability import (
        _attach_strength,
        _games_frame,
        chronological_strength,
        match_shard,
    )

    refs, load = sbod.open_data_source([(43, 106)], max_matches=3, cache_dir=None)
    rows = []
    for ref in refs:
        _provider, _mid, actions, _frames, home, *_ = load(ref)
        rows.append(match_shard(actions, cast("int | str", home), ""))
    pooled = pd.concat(rows, ignore_index=True)
    pooled = _attach_strength(pooled, chronological_strength(pooled))
    return {"actions": pooled, "games": _games_frame(pooled), "strength_column": "base_strength"}


def _public_actions(max_matches: int = 3) -> pd.DataFrame:
    """Pooled SPADL actions for a few StatsBomb open matches (via the trainer's loader)."""
    import scripts._sb_open_data as sbod

    refs, load = sbod.open_data_source([(43, 106)], max_matches=max_matches, cache_dir=None)
    rows = []
    for ref in refs:
        _provider, _mid, actions, _frames, _home, *_ = load(ref)
        rows.append(actions)
    return pd.concat(rows, ignore_index=True)


def _public_eval_pass_completion() -> dict:  # pragma: no cover - needs statsbombpy + scripts/ (DGX/P1)
    import silly_kicks.spadl.config as spadlconfig

    acts = _public_actions()
    passes = acts[acts["type_id"] == spadlconfig.actiontype_id["pass"]]
    succ = spadlconfig.result_id["success"]
    return {
        "ox": passes["start_x"].to_numpy(dtype=float),
        "oy": passes["start_y"].to_numpy(dtype=float),
        "tx": passes["end_x"].to_numpy(dtype=float),
        "ty": passes["end_y"].to_numpy(dtype=float),
        "y": (passes["result_id"] == succ).to_numpy(dtype=float),
    }


def _public_eval_xsuccess() -> dict:  # pragma: no cover - needs statsbombpy + scripts/ (DGX/P1)
    import silly_kicks.spadl.config as spadlconfig

    acts = _public_actions()
    return {"actions": acts, "y": (acts["result_id"] == spadlconfig.result_id["success"]).to_numpy(dtype=float)}


def _public_eval_match_outcome() -> dict:  # pragma: no cover - needs statsbombpy + scripts/ (DGX/P1)
    """Per open match: home/away expected goals (summed per-shot xg) -> Poisson goal pmfs + the realized
    home-win outcome. The scorer applies the fitted Dixon-Coles rho to the pmf pair."""
    from scipy.stats import poisson

    import scripts._sb_open_data as sbod
    import silly_kicks.spadl.config as spadlconfig
    from silly_kicks.id_compat import same_id

    refs, load = sbod.open_data_source([(43, 106)], max_matches=8, cache_dir=None)
    shot_ids = {spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")}
    succ = spadlconfig.result_id["success"]
    k = np.arange(16)
    matches = []
    for ref in refs:
        _provider, _mid, actions, _frames, home, *_ = load(ref)
        teams = [t for t in pd.unique(actions["team_id"].dropna())]
        away = next((t for t in teams if not same_id(t, home)), None)
        if away is None or "xg" not in actions.columns:
            continue
        xg = actions.groupby("team_id")["xg"].sum()
        home_xg, away_xg = float(xg.get(home, 0.0) or 0.0), float(xg.get(away, 0.0) or 0.0)
        scored = actions[actions["type_id"].isin(shot_ids) & (actions["result_id"] == succ)]
        goals = scored.groupby("team_id").size()
        home_win = 1.0 if int(goals.get(home, 0)) > int(goals.get(away, 0)) else 0.0
        matches.append((poisson.pmf(k, max(home_xg, 0.05)), poisson.pmf(k, max(away_xg, 0.05)), home_win))
    if not matches:
        raise EvalDataUnavailableError("match_outcome: no scoreable matches (need xg + two teams)")
    return {"matches": matches}


_PUBLIC_EVAL_BUILDERS: dict[str, Callable[[], dict]] = {
    "win_probability": _public_eval_win_probability,
    "pass_completion": _public_eval_pass_completion,
    "xsuccess": _public_eval_xsuccess,
    "match_outcome_dependence": _public_eval_match_outcome,
}


def _build_public_eval(name: str) -> dict:
    builder = _PUBLIC_EVAL_BUILDERS.get(name)
    if builder is None:
        raise EvalDataUnavailableError(
            f"{name}: public eval follows the verified win_probability pattern (statsbombpy open data -> "
            "the trainer's action rows); implemented + verified at P1 on HF (gated)."
        )
    return builder()


def _owner_shards(corpus_dir: Path) -> tuple[list[Path], dict, Path]:
    """The staged corpus's per-game frame shards (``shards/<gen>/<provider>__<game>.parquet``, or flat
    ``*.parquet``), the ``{game_id: home_team_id}`` map, and the ``_actions`` dir."""
    import glob
    import json

    shards = [Path(p) for p in sorted(glob.glob(str(corpus_dir / "shards" / "*" / "*.parquet")))]
    if not shards:
        shards = [Path(p) for p in sorted(glob.glob(str(corpus_dir / "*.parquet")))]
    if not shards:
        raise EvalDataUnavailableError(f"no frame shards under {corpus_dir}")
    hp = corpus_dir / "home_teams.json"
    home = json.loads(hp.read_text()) if hp.exists() else {}
    return shards, home, corpus_dir / "_actions"


def _home_for(game_id: str, home: dict) -> str | None:
    return home.get(game_id) or home.get(game_id.split("__")[-1])


def _owner_eval_ghost_gk(shards: list[Path], home: dict, actions_dir: Path) -> dict:
    from silly_kicks.tracking._ghost_gk import prepare_ghost_gk_training_data

    feats_list, true_list, groups = [], [], []
    for f in shards:
        frames = pd.read_parquet(f)
        gid = str(frames["game_id"].iloc[0])
        hid = _home_for(gid, home)
        if hid is None:
            continue
        apath = actions_dir / f.name
        actions = pd.read_parquet(apath) if apath.exists() else None
        feats, labs, _meta = prepare_ghost_gk_training_data(
            frames,
            home_team_id=hid,
            actions=actions,
            subsample_fps=1.0,
            feature_set="faithful",
            return_meta=True,
        )
        if len(feats) == 0:
            continue
        feats_list.append(feats.reset_index(drop=True))
        true_list.append(labs[["gk_x", "gk_y"]].to_numpy(dtype=float))
        groups.append(np.full(len(feats), gid))
    if not feats_list:
        raise EvalDataUnavailableError("ghost_gk: no shards with a resolvable home_team_id + GK labels")
    return {
        "features": pd.concat(feats_list, ignore_index=True),
        "true_xy": np.concatenate(true_list),
        "groups": np.concatenate(groups),
    }


def _with_type_name(actions: pd.DataFrame) -> pd.DataFrame:
    """The xshot/xcross extractors filter shots/crosses by ``type_name``; add it from ``type_id`` if absent."""
    if "type_name" in actions.columns:
        return actions
    import silly_kicks.spadl.config as spadlconfig

    name_by_id = {v: k for k, v in spadlconfig.actiontype_id.items()}
    return actions.assign(type_name=actions["type_id"].map(name_by_id))


def _owner_eval_classifier(shards, home, actions_dir, prepare, feature_set: str = "faithful") -> dict:
    """Shared owner-eval for the xgboost occurrence classifiers (xshot/xcross): per game, frames + actions
    -> ``prepare_<model>_training_data`` -> (features, 0/1 labels, game-id groups).

    ``feature_set`` is threaded to ``prepare`` so the eval frame matches the scored variant's vintage
    (26-feat position_only vs 27-feat faithful); xgboost ``predict_proba`` is strict on feature names."""
    feats_list, y_list, groups = [], [], []
    for f in shards:
        frames = pd.read_parquet(f)
        gid = str(frames["game_id"].iloc[0])
        hid = _home_for(gid, home)
        apath = actions_dir / f.name
        if hid is None or not apath.exists():
            continue
        feats, labels, grp = prepare(
            frames, _with_type_name(pd.read_parquet(apath)), home_team_id=hid, feature_set=feature_set
        )
        if len(feats) == 0:
            continue
        feats_list.append(feats.reset_index(drop=True))
        y_list.append(np.asarray(labels))
        groups.append(np.asarray(grp).astype(str))
    if not feats_list:
        raise EvalDataUnavailableError("no shards with a resolvable home_team_id + labels")
    return {
        "features": pd.concat(feats_list, ignore_index=True),
        "y": np.concatenate(y_list),
        "groups": np.concatenate(groups),
    }


def _owner_eval_xshot(shards, home, actions_dir, feature_set: str = "faithful") -> dict:
    from silly_kicks.tracking._xshot_occurrence import prepare_xshot_training_data

    return _owner_eval_classifier(shards, home, actions_dir, prepare_xshot_training_data, feature_set)


def _owner_eval_xcross(shards, home, actions_dir, feature_set: str = "faithful") -> dict:
    from silly_kicks.tracking._xcross_attempt import prepare_xcross_training_data

    return _owner_eval_classifier(shards, home, actions_dir, prepare_xcross_training_data, feature_set)


def _owner_eval_ghost_outfield(shards, home, actions_dir) -> dict:
    from scripts.train_ghost_outfield import extract_match  # the trainer's per-match extractor

    parts, true_list, groups = [], [], []
    for f in shards:
        frames = pd.read_parquet(f)
        gid = str(frames["game_id"].iloc[0])
        hid = _home_for(gid, home)
        if hid is None:
            continue
        apath = actions_dir / f.name
        actions = pd.read_parquet(apath) if apath.exists() else None
        data = extract_match(frames, actions, "faithful", home_team_id=hid, subsample_fps=1.0)
        if data is None or len(data) == 0:
            continue
        parts.append(data.reset_index(drop=True))
        true_list.append(data[["target_x", "target_y"]].to_numpy(dtype=float))
        groups.append(np.asarray(data["game_id"]).astype(str))
    if not parts:
        raise EvalDataUnavailableError("ghost_outfield: no shards with a resolvable home_team_id + rows")
    return {
        "features": pd.concat(parts, ignore_index=True),  # predict_mean reindexes to its own feature names
        "true": np.concatenate(true_list),
        "groups": np.concatenate(groups),
    }


def _owner_eval_receiver(shards, home, actions_dir) -> dict:
    from scripts.train_receiver_model import extract_candidate_rows  # per-pass candidate rows + id-label

    parts = []
    for f in shards:
        apath = actions_dir / f.name
        if not apath.exists():
            continue
        rows = extract_candidate_rows(
            pd.read_parquet(apath), pd.read_parquet(f), feature_set="public", labeling_strategy="id"
        )
        if len(rows):
            parts.append(rows)
    if not parts:
        raise EvalDataUnavailableError("receiver: no candidate rows (no completed passes with a labeled receiver)")
    return {"rows": pd.concat(parts, ignore_index=True)}


# One real owner-eval builder per model; each reads the staged shards + the trainer's LIBRARY extractor.
# ghost_gk / xshot / xcross reuse the public prepare_*_training_data extractors. ghost_outfield + receiver
# have no library prepare_* yet (their extraction is located + wired, then verified on the DGX corpus).
# Every builder is verified against ~/tc3-kinematics on the DGX before its re-fit is trusted; fail-loud
# (EvalDataUnavailableError) until a builder is wired.
_OWNER_EVAL_BUILDERS: dict[str, Callable[[list[Path], dict, Path], dict]] = {
    "ghost_gk": _owner_eval_ghost_gk,
    "xshot_occurrence": _owner_eval_xshot,
    "xcross_attempt": _owner_eval_xcross,
    "ghost_outfield": _owner_eval_ghost_outfield,
    "receiver_model": _owner_eval_receiver,
}


#: Owner evals whose feature frame must match the scored variant's vintage (xgboost strict feature_names).
#: The others (ghost_gk/ghost_outfield via predict_mean reindex, receiver) ignore feature_set.
_FEATURE_SET_EVALS = frozenset({"xshot_occurrence", "xcross_attempt"})


def _build_owner_eval(name: str, corpus_dir: Path, *, feature_set: str = "faithful") -> dict:
    builder = _OWNER_EVAL_BUILDERS.get(name)
    if builder is None:
        raise EvalDataUnavailableError(
            f"{name}: owner-tier eval follows the verified ghost_gk pattern (read shards -> "
            "prepare_<model>_training_data -> features+labels+groups); implemented + verified at P2 (gated)."
        )
    shards, home, actions_dir = _owner_shards(corpus_dir)
    if name in _FEATURE_SET_EVALS:
        vintage_builder = cast("Callable[[list[Path], dict, Path, str], dict]", builder)
        return vintage_builder(shards, home, actions_dir, feature_set)
    return builder(shards, home, actions_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trainer")
    ap.add_argument("--committed-dir", type=Path, required=True)
    ap.add_argument("--fresh-dir", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, default=None)
    ap.add_argument("--feature-set", default="faithful", help="Eval feature vintage (faithful|position_only).")
    ap.add_argument("--metrics-json", type=Path, required=True)
    a = ap.parse_args()
    # fail-loud: real data or EvalDataUnavailableError, never synthetic; feature_set matches the variant vintage.
    bundle = build_eval(a.trainer, a.corpus_dir, feature_set=a.feature_set)
    paired = SCORERS[a.trainer](a.committed_dir, a.fresh_dir, bundle)
    existing = json.loads(a.metrics_json.read_text()) if a.metrics_json.exists() else {}
    existing.update(paired)
    existing.update(stamp())
    a.metrics_json.write_text(json.dumps(existing, indent=2))
    print("HF_JOBS_HEADTOHEAD_OK " + json.dumps(paired))


if __name__ == "__main__":
    main()
