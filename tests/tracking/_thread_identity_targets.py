"""The n_jobs-lever identity targets (HFU plan R3-8 S3 item 1 (c)): each runs in a process started with the
production worker env (`tests.scripts._parity.assert_thread_axis_identical_in_env`), reads its thread budget the
way the trainer does (`runtime_n_jobs_from_env`), and writes what the stage writes. The fixtures are the
synthetic ones of `tests/tracking/test_thread_identity.py` (same seeds), so the child and the parent's
non-vacuity precondition see the same data.
"""

from __future__ import annotations

import pathlib

import numpy as np


def _write_json(out: pathlib.Path, obj) -> None:
    """The result as the STORE encodes a unit result (`_stages._encode`; review S3B-09: the arms compare the
    stage's encoded unit results, byte for byte)."""
    from scripts._stages import _encode

    out.mkdir(parents=True, exist_ok=True)
    suffix, payload, files = _encode(obj)
    (out / f"result{suffix}").write_bytes(payload)
    for name, data in files.items():
        (out / name).write_bytes(data)


def _even_odd(n: int):
    return np.arange(0, n, 2), np.arange(1, n, 2)


def xshot_fit(out: pathlib.Path, threads: int) -> None:
    """The xShot `fit` stage's model: an XShotOccurrenceModel with the runtime n_jobs read trainer-side."""
    from scripts._thread_pin import runtime_n_jobs_from_env
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel
    from tests.scripts.hf_jobs.fixtures import make_synthetic_xshot_fit

    assert runtime_n_jobs_from_env() == threads  # the transport the trainer reads
    X, y = make_synthetic_xshot_fit(seed=0)
    XShotOccurrenceModel().fit(X, y, runtime_n_jobs=runtime_n_jobs_from_env()).save(out)


def xcross_fit(out: pathlib.Path, threads: int) -> None:
    """The xCross `fit` stage's model: an XCrossAttemptModel with the runtime n_jobs read trainer-side."""
    from scripts._thread_pin import runtime_n_jobs_from_env
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    from tests.scripts.hf_jobs.fixtures import make_synthetic_xcross_fit

    assert runtime_n_jobs_from_env() == threads
    X, y = make_synthetic_xcross_fit(seed=0)
    XCrossAttemptModel().fit(X, y, runtime_n_jobs=runtime_n_jobs_from_env()).save(out)


def xshot_cv_fold(out: pathlib.Path, threads: int) -> None:
    """The xShot `cv` stage's fold (an XShotOccurrenceModel xgb fit through `_cv_metrics_fold`, which reads the
    n_jobs-lever budget from the env itself)."""
    from scripts._thread_pin import runtime_n_jobs_from_env
    from scripts.train_xshot_occurrence import _cv_metrics_fold
    from tests.scripts.hf_jobs.fixtures import make_synthetic_xshot_fit

    assert runtime_n_jobs_from_env() == threads
    X, y = make_synthetic_xshot_fit(seed=0)
    tr, te = _even_odd(len(X))
    fold = _cv_metrics_fold(X, y.to_numpy(), 0, tr, te, {"n_estimators": 20})
    assert fold is not None  # the synthetic fold is non-degenerate
    _write_json(out, list(fold))


def xcross_cv_fold(out: pathlib.Path, threads: int) -> None:
    """The xCross `cv` stage's fold (an XCrossAttemptModel xgb fit through `_cv_score_fold`, which takes the
    trainer-side budget as `runtime_n_jobs`)."""
    from scripts._thread_pin import runtime_n_jobs_from_env
    from silly_kicks.tracking import _xcross_eval as ev
    from tests.scripts.hf_jobs.fixtures import make_synthetic_xcross_fit

    assert runtime_n_jobs_from_env() == threads
    X, y = make_synthetic_xcross_fit(seed=0)
    tr, te = _even_odd(len(X))
    fold = ev._cv_score_fold(X, y.to_numpy(), 0, tr, te, {"n_estimators": 20}, runtime_n_jobs=runtime_n_jobs_from_env())
    assert fold is not None
    _write_json(out, list(fold))


def _tf19_corpus():
    """A shipped-arm stand-in for the xcross TF-19 block: 12 games, 50 rows each, coverage below 1."""
    from tests.scripts.hf_jobs.fixtures import make_synthetic_xcross_fit

    X, y = make_synthetic_xcross_fit(n_samples=600, seed=0)
    rng = np.random.default_rng(0)
    X.loc[rng.random(len(X)) < 0.2, "score_differential"] = np.nan
    return X, y.to_numpy(int), np.repeat([f"g{i:02d}" for i in range(12)], 50)


_TF19_PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3}


def xcross_tf19_perm(out: pathlib.Path, threads: int) -> None:
    """The xcross `tf19_perm_fit` stage (HFU S3 item 8 (f)): each fold model fitted by `tf19_perm_fold_fit` at the
    n_jobs-lever budget, then every (fold, column) permutation unit, combined to the `permutation_importance` dict."""
    from scripts._stages import int_key, join_key
    from scripts._thread_pin import runtime_n_jobs_from_env
    from scripts.train_xcross_attempt import (
        TF19_FOLDS,
        tf19_coverage,
        tf19_perm_column,
        tf19_perm_fold_fit,
        tf19_perm_report,
    )

    assert runtime_n_jobs_from_env() == threads
    X, y, groups = _tf19_corpus()
    fits = {
        f: tf19_perm_fold_fit(X, y, groups, _TF19_PARAMS, f, seed=7, runtime_n_jobs=threads) for f in range(TF19_FOLDS)
    }
    results = {
        join_key((int_key(f), int_key(c))): tf19_perm_column(fits[f], X, y, groups, f, c, seed=7)
        for f in range(TF19_FOLDS)
        for c in range(X.shape[1])
    }
    _write_json(out, tf19_perm_report(results, list(X.columns), tf19_coverage(X)))


_STUDY_TRIALS = (
    {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3},
    {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.1},
    {"n_estimators": 15, "max_depth": 2, "learning_rate": 0.5},
    {"n_estimators": 25, "max_depth": 5, "learning_rate": 0.2},
)


def _study_children(module: str, objective: str, out: pathlib.Path, threads: int, make) -> None:
    """Four trials of a `study` stage, each trial's folds in two InnerPool children that inherit this process's
    production env (the child budget ``threads``) and fit through the trainer's `_study_fold_task`."""
    import importlib
    import tempfile

    from scripts._corpus_mmap import persist_design_matrix
    from scripts._stages import int_key
    from scripts._thread_pin import runtime_n_jobs_from_env
    from scripts.hf_jobs.executor import InnerPool

    assert runtime_n_jobs_from_env() == threads
    fold_task = importlib.import_module(module)._study_fold_task
    obj = importlib.import_module(objective)
    X, y = make(n_samples=600, seed=0)
    y = y.to_numpy(int)
    groups = np.repeat([f"g{i:02d}" for i in range(12)], 50)
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        corpus = persist_design_matrix(X, y, groups, root / "corpus")
        rows, plan_path = root / "rows.npy", root / "plan.npz"
        np.save(rows, np.arange(len(y)))
        plan = obj.cv_fold_plan(y, groups)
        np.savez(plan_path, *[a for _i, tr, te in plan for a in (tr, te)])
        pool = InnerPool(2)
        try:
            values = []
            for params in _STUDY_TRIALS:
                items = {int_key(i): (corpus, rows, plan_path, i, params, None, 42) for i, _tr, _te in plan}
                res = pool.map_keyed(fold_task, items)
                values.append(list(obj.combine_folds([res[k] for k in sorted(res)])))
        finally:
            pool.shutdown()
    _write_json(out, values)


def xshot_study_children(out: pathlib.Path, threads: int) -> None:
    """The xShot `study` stage (HFU S3 item 9): `_study_fold_task` in the InnerPool children at the child budget."""
    from tests.scripts.hf_jobs.fixtures import make_synthetic_xshot_fit

    _study_children(
        "scripts.train_xshot_occurrence",
        "silly_kicks.tracking._xshot_occurrence_objective",
        out,
        threads,
        make_synthetic_xshot_fit,
    )
