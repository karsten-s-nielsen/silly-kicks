"""HFU plan R3-8 S3 item 8 (b)-(f): the xcross TF-19 block as units (row 28; owner ruling 2026-10-30).

`_tf19_block` ran the GK-block ablation, the CV-held-out permutation importance and the substitution probe in ONE
serial unit. The trainer now runs them as units: `tf19_ablation` (one per (arm, fold)), `tf19_perm_fit` (one per
fold: the fold model), `tf19_perm` (one per (fold, column)) and `tf19_probe`; `assemble` combines them. The
`2474b73` bodies are kept below VERBATIM as the oracles (the D20 method; only the names differ). Every result goes
through the store's JSON before the combine, as in a job.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

import scripts.train_xcross_attempt as T
from scripts._stages import _encode, int_key, join_key
from scripts.train_xcross_attempt import _gated_probe_matches
from silly_kicks.tracking import _xcross_eval as ev
from silly_kicks.tracking._xcross_attempt import XCROSS_GK_BLOCK, _pinned_params
from silly_kicks.tracking._xcross_eval import _cv_score

PARAMS = {"n_estimators": 15, "max_depth": 3, "learning_rate": 0.3}


# ---- the 2474b73 bodies, VERBATIM ------------------------------------------------------------------------


def _gk_block_ablation_2474b73(
    X: pd.DataFrame,  # noqa: N803 -- the verbatim oracle keeps sklearn's `X`
    y,
    groups,
    params: dict,
    *,
    seed: int = 42,
    negative_subsample: float | None = None,
) -> dict:
    """Reported context: held-out PR-AUC + log-loss WITH vs WITHOUT XCROSS_GK_BLOCK, scored via
    the SAME ``_cv_score`` the acceptance gate uses (so deltas are gate-comparable for any seed/ns)."""
    # Feature-set-agnostic: the caller passes the shipped model's feature matrix (faithful 16 /
    # position_only 15), so derive the columns from X rather than hardcoding FAITHFUL (which would
    # KeyError on ['ball_speed'] for a position_only model). XCROSS_GK_BLOCK is velocity-free, so it
    # is present in both sets and the "without GK" drop is correct either way.
    feature_cols = list(X.columns)
    with_ = _cv_score(X[feature_cols], y, groups, params, seed=seed, negative_subsample=negative_subsample)
    base_cols = [c for c in feature_cols if c not in XCROSS_GK_BLOCK]
    wo_ = _cv_score(X[base_cols], y, groups, params, seed=seed, negative_subsample=negative_subsample)
    return {
        "with_gk_pr_auc": with_["pr_auc"],
        "without_gk_pr_auc": wo_["pr_auc"],
        "with_gk_log_loss": with_["log_loss"],
        "without_gk_log_loss": wo_["log_loss"],
        "delta_pr_auc": with_["pr_auc"] - wo_["pr_auc"],
        "delta_log_loss": with_["log_loss"] - wo_["log_loss"],
        "note": "reported context (marginal predictive value); NOT the tf19_ready gate",
    }


def _permutation_importance_report_2474b73(
    X: pd.DataFrame,  # noqa: N803 -- the verbatim oracle keeps sklearn's `X`
    y,
    groups,
    params: dict,
    *,
    n_repeats: int = 10,
    seed: int = 42,
) -> dict:
    """Reported context: CV-HELD-OUT permutation importance (C2 -- never permuted on the all-data
    shipped model's own training data). For each fold: fit on K-1, permute+score on fold K with
    scoring='average_precision' (B3); average importances across folds. Also report
    score_differential coverage (non-NaN fraction over the full matrix, B2)."""
    import xgboost as xgb
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.utils import Bunch

    # Feature-set-agnostic (see gk_block_ablation): derive columns from X, never hardcode FAITHFUL.
    feature_cols = list(X.columns)
    X = X[feature_cols]
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    per_fold: list[np.ndarray] = []
    for tr, te in skf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        p_ = dict(_pinned_params(params))
        p_["base_score"] = float(y[tr].mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(X.iloc[tr].to_numpy(float), y[tr])
        r = permutation_importance(
            clf, X.iloc[te].to_numpy(float), y[te], scoring="average_precision", n_repeats=n_repeats, random_state=seed
        )
        # A single scoring string returns one Bunch; sklearn returns a dict[str, Bunch] only for
        # multi-metric scoring. Narrow without an `assert` (S101) -- handle both for safety.
        bunch = r if isinstance(r, Bunch) else next(iter(r.values()))
        per_fold.append(bunch.importances_mean)

    if per_fold:
        mean_imp = np.mean(np.vstack(per_fold), axis=0)
        importances = {f: float(v) for f, v in zip(feature_cols, mean_imp, strict=True)}
    else:
        importances = {f: float("nan") for f in feature_cols}

    coverage = float(X["score_differential"].notna().mean())
    return {
        "importances": importances,
        "score_differential_importance": importances["score_differential"],
        "score_differential_coverage": coverage,
        "scoring": "average_precision",
        "n_repeats": n_repeats,
        "n_folds_used": len(per_fold),
        "held_out": True,
        "note": "CV-held-out, architecture-representative; NOT measured on the production "
        "weights' own training data. score_differential importance is qualified by coverage.",
    }


def _tf19_block_2474b73(ctx, inp, model, shipped, candidates) -> dict:
    """The headline GK validations on the SHIPPED candidate (PR-B; legacy assemble `run_probe`): the GK-block
    ablation, the CV-held-out permutation importance and the TF-19 substitution probe on the held-out cohort."""
    import tempfile

    from scripts._stages import materialize_fileset
    from silly_kicks.tracking import _xcross_eval as ev

    cfg = inp.config
    ns, seed = cfg["negative_subsample"], cfg["seed"]
    X, y, groups = inp.X, inp.y, inp.groups
    ship_mask = inp.arms[shipped]
    shipped_params = candidates[shipped]["params"]
    gk_ablation = ev.gk_block_ablation(
        X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, seed=seed, negative_subsample=ns
    )
    perm_imp = ev.permutation_importance_report(
        X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, n_repeats=10, seed=seed
    )
    with tempfile.TemporaryDirectory() as tmp:
        ps = materialize_fileset(ctx, "probe", "_output", Path(tmp) / "_probe_sample")
        if not (ps / "frames.parquet").exists():
            raise RuntimeError(
                "the probe stage captured no cohort -> cannot run the TF-19 substitution probe (the headline "
                "deliverable). Check --probe-dataset / --probe-providers. Refusing to ship a spurious "
                "tf19_ready=False."
            )
        pf = pd.read_parquet(ps / "frames.parquet")
        pa = pd.read_parquet(ps / "actions.parquet")
        probe_meta = json.loads((ps / "meta.json").read_text(encoding="utf-8"))
    run_paired = cfg["run_paired"]
    phome = probe_meta["home_team_id"]
    probe_matches = probe_meta.get("probe_matches", [])
    match_groups = probe_meta.get("match_groups", {})
    train_groups = set(groups[ship_mask].tolist())
    in_training = {mid: bool(set(g) & train_groups) for mid, g in match_groups.items()}
    probe_meta["in_training_folds"] = in_training
    shipped_providers = set(candidates[shipped].get("providers", []))
    probe_provs_seen = {prov for prov, _mid in probe_matches}
    admitted = bool(run_paired and (not probe_provs_seen or probe_provs_seen & shipped_providers))
    try:
        gated_matches = _gated_probe_matches(probe_meta, admitted)
    except SystemExit as exc:  # the legacy refusal; a stage unit fails on an Exception, never exits the worker
        raise RuntimeError(str(exc)) from None
    if admitted:
        held_groups = {g for _prov, mid in gated_matches for g in match_groups.get(str(mid), [])}
        pf = pf[pf["game_id"].astype(str).isin(held_groups)]
        pa = pa[pa["game_id"].astype(str).isin(held_groups)]
        if pf.empty:
            raise RuntimeError(
                "Held-out probe matches resolved ZERO frames -- the probe sample is inconsistent with its "
                "meta.json match_groups. Re-run the probe stage."
            )
    elif probe_matches and all(in_training.get(str(m[1]), False) for m in probe_matches):
        print(
            "NOTE: every probe match sits in the shipped training corpus (single-candidate path) -- "
            "the substitution-probe statistic below is NOT held-out.",
            file=sys.stderr,
        )
    probe = ev.gk_substitution_probe(model, pf, actions=pa, home_team_id=phome, n_frames=200, seed=seed)
    if not probe.get("tf19_ready", False):
        print(
            f"NOTE: tf19_ready=False ({probe.get('tf19_reason')}) -- surface ships, but flagged "
            "NOT TF-19-ready (loud, not silent).",
            file=sys.stderr,
        )
    return {
        "gk_block_ablation": gk_ablation,
        "gk_substitution_probe": probe,
        "permutation_importance": perm_imp,
        "score_differential_range_probe": cfg["score_differential_range_probe"],
        "probe_sample_matches": probe_matches,
        "probe_sample_in_training_folds": in_training,
        "probe_gated_on_held_out": bool(admitted),
        "tf19_ready": probe.get("tf19_ready", False),
    }


# ---- fixtures -----------------------------------------------------------------------------------------------------


def _corpus(n_games: int, *, rows: int = 50, seed: int = 0):
    from tests.scripts.hf_jobs.fixtures import make_synthetic_xcross_fit

    X, y = make_synthetic_xcross_fit(n_samples=n_games * rows, seed=seed)
    rng = np.random.default_rng(seed)
    X.loc[rng.random(len(X)) < 0.2, "score_differential"] = np.nan  # coverage below 1
    groups = np.repeat([f"g{i:02d}" for i in range(n_games)], rows)
    return X, y.to_numpy(int), groups


def _stored(obj):
    """A unit result as the store writes and reads it (`_stages._encode`: key order kept, tuples become lists)."""
    suffix, data, _files = _encode(obj)
    assert suffix == ".json"
    return json.loads(data)


# ---- (b) the GK-block ablation: one unit per (arm, fold) ----------------------------------------------------------


@pytest.mark.parametrize("n_games", [12, 3])
def test_the_ablation_units_combine_to_the_serial_gk_block_ablation(n_games):
    """12 games make 5 folds per arm; 3 make 3 folds and 2 absent units per arm."""
    X, y, groups = _corpus(n_games)
    results = {
        join_key((arm, int_key(f))): _stored(
            T.tf19_ablation_fold(X, y, groups, PARAMS, arm, f, seed=7, negative_subsample=0.5)
        )
        for arm in T.TF19_ABLATION_ARMS
        for f in range(T.TF19_FOLDS)
    }
    oracle = _gk_block_ablation_2474b73(X, y, groups, PARAMS, seed=7, negative_subsample=0.5)
    assert json.dumps(T.tf19_ablation_report(results)) == json.dumps(oracle)
    # each unit equals the serial block's fold value
    for i, tr, te in ev._cv_score_plan(y, groups):
        serial = ev._cv_score_fold(X, y, i, tr, te, PARAMS, seed=7, negative_subsample=0.5)
        assert results[join_key(("with", int_key(i)))]["fold"] == (None if serial is None else list(serial))
    absent = [k for k, v in results.items() if v.get("absent")]
    assert len(absent) == 2 * (T.TF19_FOLDS - min(5, n_games))


# ---- (c) the permutation importance: the fold fits, then one unit per (fold, column) ------------------------------


@pytest.mark.parametrize("n_games", [12, 3])
def test_the_permutation_units_combine_to_the_serial_permutation_importance_report(n_games):
    X, y, groups = _corpus(n_games)
    fits = {f: T.tf19_perm_fold_fit(X, y, groups, PARAMS, f, seed=7) for f in range(T.TF19_FOLDS)}
    results = {
        join_key((int_key(f), int_key(c))): _stored(T.tf19_perm_column(fits[f], X, y, groups, f, c, seed=7))
        for f in range(T.TF19_FOLDS)
        for c in range(X.shape[1])
    }
    report = T.tf19_perm_report(results, list(X.columns), T.tf19_coverage(X))
    oracle = _permutation_importance_report_2474b73(X, y, groups, PARAMS, n_repeats=10, seed=7)
    assert json.dumps(report) == json.dumps(oracle)
    assert report["n_folds_used"] == min(5, n_games)  # non-vacuity: every real fold scored


def test_a_fold_with_one_class_is_skipped_exactly_as_the_serial_report_skips_it():
    X, y, groups = _corpus(12)
    y = y.copy()
    y[groups == "g00"] = 0  # still two classes overall; the oracle decides which folds it skips
    fits = {f: T.tf19_perm_fold_fit(X, y, groups, PARAMS, f, seed=7) for f in range(T.TF19_FOLDS)}
    results = {
        join_key((int_key(f), int_key(c))): _stored(T.tf19_perm_column(fits[f], X, y, groups, f, c, seed=7))
        for f in range(T.TF19_FOLDS)
        for c in range(X.shape[1])
    }
    report = T.tf19_perm_report(results, list(X.columns), T.tf19_coverage(X))
    assert json.dumps(report) == json.dumps(_permutation_importance_report_2474b73(X, y, groups, PARAMS, seed=7))


# ---- the pipeline wiring --------------------------------------------------------------------------------------------


def test_the_tf19_block_runs_as_units_before_assemble():
    P = T.PIPELINE
    ctx: Any = types.SimpleNamespace(params={"feature_set": "faithful"})  # the units read only the params
    assert len(P.stage("tf19_ablation").units(ctx)) == 2 * T.TF19_FOLDS == 10
    assert len(P.stage("tf19_perm_fit").units(ctx)) == T.TF19_FOLDS == 5
    assert len(P.stage("tf19_perm").units(ctx)) == T.TF19_FOLDS * len(T._feature_names("faithful")) == 80
    assert P.stage("tf19_probe").units(ctx) == ["only"]
    assert {"tf19_ablation", "tf19_perm", "tf19_probe"} <= set(P.stage("assemble").needs)
    perm_fit = P.stage("tf19_perm_fit")  # the fold models take the LEVER budget (item 8 (f)): W=5 x 2 on HF
    assert perm_fit.thread_safe and perm_fit.thread_lever == "n_jobs"
    # owner rulings 2026-10-31 (S3B-04): `tf19_ablation` runs pinned (W=8 x 1 beat the lever shape on HF) and
    # `tf19_probe` is one unit; both carry a reviewed reason
    ablation, probe = P.stage("tf19_ablation"), P.stage("tf19_probe")
    assert not ablation.thread_safe
    assert ablation.serial_reason and probe.serial_reason
    assert not hasattr(T, "_tf19_block")  # no serial copy left in production


# ---- (e) the whole block on the toy corpus, against the serial `_tf19_block` ----------------------------------------


@pytest.mark.slow  # slow: one serial toy-corpus run of the whole xcross pipeline (ADR-023)
def test_the_assembled_tf19_block_equals_the_serial_block_on_the_toy_corpus(tmp_path):
    """The toy pipeline runs the units; the oracle runs the `2474b73` `_tf19_block` on the same stored inputs (the
    same context, select output and shipped model). The block keys of `metrics.json` are byte-identical, in order."""
    pytest.importorskip("ruthless")
    import tempfile

    from scripts._provenance import code_identity, env_provenance, git_provenance
    from scripts._stages import LocalStore, build_parser, make_context, materialize_fileset
    from scripts._toy_corpora import build_xcross_corpus, build_xcross_probe_corpus, xshot_arms
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    from tests.scripts._parity import run_serial

    data = tmp_path / "data"
    build_xcross_corpus(data / "toyxc")
    build_xcross_probe_corpus(data / "toyxcprobe")
    args = [
        "--dataset",
        "toyxc",
        "--probe-dataset",
        "toyxcprobe",
        "--arms-json",
        xshot_arms(data / "toyxc", paired=True),
    ]
    args += ["--n-trials", "4"]
    run_serial(T.__name__, tmp_path / "s", data, args, tmp_path / "o")
    metrics = json.loads((tmp_path / "o" / "metrics.json").read_text(encoding="utf-8"))

    ns = build_parser(T.PIPELINE).parse_args(
        ["--plan", "--store", str(tmp_path / "s"), "--data-root", str(data), *args]
    )
    params = {p.name: getattr(ns, p.name) for p in T.PIPELINE.params}
    git = git_provenance()
    ctx = make_context(
        T.PIPELINE,
        params,
        {**git, "env": env_provenance(), "code_identity": code_identity(git)},  # as `_stages.main` builds it
        store=LocalStore(tmp_path / "s"),
        data_root=data,
    )
    _root, inp = T._inputs(ctx)
    sel = ctx.upstream("select")
    with tempfile.TemporaryDirectory() as tmp:
        model = XCrossAttemptModel.load(materialize_fileset(ctx, "fit", "_output", Path(tmp) / "fit"))
    oracle = _tf19_block_2474b73(ctx, inp, model, sel["shipped"], sel["candidates"])
    assert list(metrics)[-len(oracle) :] == list(oracle)  # the block keys, in the serial order
    assert json.dumps({k: metrics[k] for k in oracle}, indent=2) == json.dumps(oracle, indent=2)
