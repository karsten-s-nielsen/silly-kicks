"""ADR-128 INV-2 opt-in gate (Cycle 1, T4): EVERY ``thread_safe`` fit's artifact is byte-identical across the
production thread range {1,2,4,8}, whether driven by OpenMP (HistGB) or the runtime ``n_jobs`` (xgb). One test
per opted-in model (``test_thread_safe_registry`` binds each stage to the test named here, fail-closed -- B42).

Runs on x86 in CI (the HF ``cpu-upgrade`` production arch); no aarch64/DGX leg (DGX is a backup, never the
benchmark -- P15). Thread counts are applied via ``threadpoolctl`` logical limits, so the full {1,2,4,8} range
is exercised regardless of the runner's core count (B44); for the omp leg BLAS is pinned to 1 in-process so the
env matches production (B53).

The fixtures are SYNTHETIC at the models' real feature width (never owner-tier raw positions, B16) and above
sklearn's 10_000-row early-stopping threshold so the gate takes the production ES path (B43). A non-vacuity
precondition proves the fixture engages both sklearn parallel axes (else the identity test is vacuous).
"""

from __future__ import annotations

import pytest

from silly_kicks.tracking._ghost_gk import GhostGkModel
from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel
from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel
from tests.scripts import _parity
from tests.scripts.hf_jobs.fixtures import (
    assert_parallel_engages,
    make_synthetic_ghost_fit,
    make_synthetic_ghost_outfield_table,
    make_synthetic_xcross_fit,
    make_synthetic_xshot_fit,
)

#: The production thread range the gate proves identity across (B44), applied via threadpoolctl logical limits.
THREAD_COUNTS = (1, 2, 4, 8)
_BUDGET = max(THREAD_COUNTS)

#: The models whose byte-identity across thread counts is proven in THIS module. test_thread_safe_registry
#: binds each thread_safe stage to one of these; a stage whose model is not covered here fails the gate.
IDENTITY_COVERED = frozenset({"GhostGkModel", "GhostOutfieldModel", "XShotOccurrenceModel", "XCrossAttemptModel"})


@pytest.mark.slow  # slow: ~40 HistGB/xgb fits on a 12000-row fixture; invariant, runs in the dedicated slow job
def test_ghost_gk_histgb_byte_identical_across_omp_threads(tmp_path):
    X, labels = make_synthetic_ghost_fit(seed=0)
    assert_parallel_engages(n_samples=len(X), n_features=X.shape[1], budget=_BUDGET)

    def fit_and_save(out, threads):  # omp threads applied by the harness via OpenMP limits, not here
        GhostGkModel(n_estimators=30).fit(X, labels).save(out)

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="omp")


@pytest.mark.slow
def test_the_ghost_fixture_engages_the_production_early_stopping_path():
    # B43/B54: the >10_000-row fixture must take sklearn's early_stopping="auto" path (the production code
    # path), not a sub-threshold one. Assert the fitted HistGB enabled ES -- else B43 is assumed, not proven.
    X, labels = make_synthetic_ghost_fit()
    m = GhostGkModel(n_estimators=30).fit(X, labels)
    assert getattr(m._sk_reg_x, "do_early_stopping_", False) is True


@pytest.mark.slow
def test_ghost_outfield_histgb_byte_identical_across_omp_threads(tmp_path):
    table = make_synthetic_ghost_outfield_table(seed=0)
    n_features = table.shape[1] - 2  # minus target_x/target_y
    assert_parallel_engages(n_samples=len(table), n_features=n_features, budget=_BUDGET)

    def fit_and_save(out, threads):
        GhostOutfieldModel(n_estimators=30)._fit_extracted(table).save(out)

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="omp")


@pytest.mark.slow
def test_xshot_xgb_byte_identical_across_runtime_n_jobs(tmp_path):
    X, y = make_synthetic_xshot_fit(seed=0)
    assert_parallel_engages(n_samples=len(X), n_features=X.shape[1], budget=_BUDGET)

    def fit_and_save(out, threads):
        XShotOccurrenceModel().fit(X, y, runtime_n_jobs=threads).save(out)

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="n_jobs")


@pytest.mark.slow
def test_xcross_xgb_byte_identical_across_runtime_n_jobs(tmp_path):
    X, y = make_synthetic_xcross_fit(seed=0)
    assert_parallel_engages(n_samples=len(X), n_features=X.shape[1], budget=_BUDGET)

    def fit_and_save(out, threads):
        XCrossAttemptModel().fit(X, y, runtime_n_jobs=threads).save(out)

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="n_jobs")


def test_non_vacuity_precondition_is_red_below_budget_on_the_row_axis():
    # fewer rows than threads -> the row-parallel loops (histogram.pyx:160,164; splitting.pyx:353,395) idle
    with pytest.raises(AssertionError, match="row axis"):
        assert_parallel_engages(n_samples=7, n_features=26, budget=8)


def test_non_vacuity_precondition_is_red_below_budget_on_the_feature_axis():
    # fewer features than threads -> the feature-parallel loops (histogram.pyx:170,270; splitting.pyx:519) idle
    with pytest.raises(AssertionError, match="feature axis"):
        assert_parallel_engages(n_samples=512, n_features=7, budget=8)


# ---- HFU T11: the cv stages are thread_safe too; each opted-in cv stage proves its fold result is -----------
# ---- byte-identical across the production thread range (serialized fold floats, NO model files). -------------


def _write_json(out, obj) -> None:
    import json
    import pathlib

    p = pathlib.Path(out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "cv.json").write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")


def _even_odd(n: int):
    import numpy as np

    return np.arange(0, n, 2), np.arange(1, n, 2)


@pytest.mark.slow
def test_cv_thread_identity_ghost(tmp_path):
    """The ghost_outfield `cv` stage fits a GhostOutfieldModel HistGB per fold; its serialized fold result
    (`_run_one_fold_of`: euclidean-MAE partials) is byte-identical across the OMP thread range {1,2,4,8}."""
    import numpy as np

    from scripts.train_ghost_outfield import _run_one_fold_of

    table = make_synthetic_ghost_outfield_table(seed=0)
    n_features = table.shape[1] - 2
    assert_parallel_engages(n_samples=len(table), n_features=n_features, budget=_BUDGET)
    tr, te = _even_odd(len(table))
    providers = np.array(["p"] * len(table))

    def fit_and_save(out, threads):  # omp threads applied by the harness; HistGB reads the OMP env
        fold = _run_one_fold_of(table, providers, tr, te, feature_set="faithful", n_estimators=20, max_depth=3)
        _write_json(out, fold)

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="omp")


@pytest.mark.slow
def test_cv_thread_identity_ghost_gk(tmp_path):
    """The ghost_gk `cv` stage fits a GhostGkModel HistGB per fold; the fold's euclidean MAE is byte-identical
    across the OMP thread range {1,2,4,8} (the cv fold is a fit-plus-score, so the fit arm's omp invariance
    carries to the cv result)."""
    import numpy as np

    X, labels = make_synthetic_ghost_fit(seed=0)
    assert_parallel_engages(n_samples=len(X), n_features=X.shape[1], budget=_BUDGET)
    tr, te = _even_odd(len(X))

    def fit_and_save(out, threads):
        # Match `_cv_work`'s HistGB-shaping config (max_depth=8, the production default), not a bare default
        # model (reviewer B D1-IMPL-16): the fold's thread identity must cover the config the cv stage runs.
        # grid_spec/carrier_params are extraction/predict-time (the synthetic fixture is pre-extracted), so they
        # do not shape this fold FIT's OMP behaviour, which is what this arm proves.
        model = GhostGkModel(n_estimators=20, max_depth=8).fit(X.iloc[tr], labels.iloc[tr])
        preds = model.predict_mean(X.iloc[te])
        yte = labels.iloc[te]
        mae = float(
            np.mean(np.sqrt((preds[:, 0] - yte["gk_x"].to_numpy()) ** 2 + (preds[:, 1] - yte["gk_y"].to_numpy()) ** 2))
        )
        _write_json(out, {"mae_euclid": mae})

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="omp")


@pytest.mark.slow
def test_cv_thread_identity_xgb(tmp_path):
    """The xshot_occurrence `cv` stage fits an XShotOccurrenceModel xgb per fold through `_cv_metrics_fold`,
    which reads the n_jobs-lever budget from the env; the fold metrics are byte-identical across {1,2,4,8}."""
    import os

    from scripts._thread_pin import RUNTIME_THREADS_VAR
    from scripts.train_xshot_occurrence import _cv_metrics_fold

    X, y = make_synthetic_xshot_fit(seed=0)
    assert_parallel_engages(n_samples=len(X), n_features=X.shape[1], budget=_BUDGET)
    tr, te = _even_odd(len(X))

    def fit_and_save(out, threads):  # the n_jobs lever: the budget travels through the env var the fold reads
        prev = os.environ.get(RUNTIME_THREADS_VAR)
        os.environ[RUNTIME_THREADS_VAR] = str(threads)
        try:
            fold = _cv_metrics_fold(X, y.to_numpy(), 0, tr, te, {"n_estimators": 20})
        finally:
            if prev is None:
                os.environ.pop(RUNTIME_THREADS_VAR, None)
            else:
                os.environ[RUNTIME_THREADS_VAR] = prev
        assert fold is not None  # the synthetic fold is non-degenerate
        _write_json(out, list(fold))

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="n_jobs")


@pytest.mark.slow
def test_cv_thread_identity_xcross(tmp_path):
    """The xcross_attempt `cv` stage fits an XCrossAttemptModel xgb per fold through `_cv_score_fold`, which
    takes the n_jobs-lever budget as `runtime_n_jobs`; the fold metrics are byte-identical across {1,2,4,8}."""
    from silly_kicks.tracking import _xcross_eval as ev

    X, y = make_synthetic_xcross_fit(seed=0)
    assert_parallel_engages(n_samples=len(X), n_features=X.shape[1], budget=_BUDGET)
    tr, te = _even_odd(len(X))

    def fit_and_save(out, threads):  # the n_jobs lever: the budget is the runtime arg
        fold = ev._cv_score_fold(X, y.to_numpy(), 0, tr, te, {"n_estimators": 20}, runtime_n_jobs=threads)
        assert fold is not None
        _write_json(out, list(fold))

    _parity.assert_thread_axis_identical(fit_and_save, thread_counts=THREAD_COUNTS, tmp_root=tmp_path, lever="n_jobs")


def test_xgb_cv_fit_reads_runtime_n_jobs(monkeypatch):
    """HFU T11 S3/B-06: the xgb cv fold fit reads the n_jobs-lever budget (not the constant pinned 1) -- else
    the cv stage stays 1-thread while the identity arm passes vacuously. `_cv_metrics_fold` must build its
    XGBClassifier with the env budget's n_jobs."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    from scripts.train_xshot_occurrence import _cv_metrics_fold

    seen: dict = {}
    real = xgb.XGBClassifier

    def recording(**kw):
        seen["n_jobs"] = kw.get("n_jobs")
        return real(**kw)

    monkeypatch.setattr(xgb, "XGBClassifier", recording)
    monkeypatch.setenv("SILLY_KICKS_RUNTIME_THREADS", "7")
    n = 40
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float)[::-1]})
    y = np.array([0, 1] * (n // 2))
    tr, te = np.arange(0, n // 2), np.arange(n // 2, n)  # contiguous halves: each keeps both classes
    _cv_metrics_fold(X, y, 0, tr, te, {"n_estimators": 5})
    assert seen["n_jobs"] == 7  # the cv fit read the runtime budget, not the pinned 1


def test_xcross_cv_fold_fit_reads_runtime_n_jobs(monkeypatch):
    """HFU T11 / reviewer B D1-IMPL-16: the xcross cv fold fit applies the n_jobs-lever budget passed as
    `runtime_n_jobs` (the trainer's `_cv_work` feeds it `runtime_n_jobs_from_env()`) to the xgb fit, not the
    pinned 1 -- else the xcross cv stays 1-thread while its identity arm passes vacuously. Mirrors the xshot S3
    gate for the xcross trainer's n_jobs wiring."""
    import xgboost as xgb

    from silly_kicks.tracking import _xcross_eval as ev

    seen: dict = {}
    real = xgb.XGBClassifier

    def recording(**kw):
        seen["n_jobs"] = kw.get("n_jobs")
        return real(**kw)

    monkeypatch.setattr(xgb, "XGBClassifier", recording)
    X, y = make_synthetic_xcross_fit(seed=0)
    tr, te = _even_odd(len(X))
    ev._cv_score_fold(X, y.to_numpy(), 0, tr, te, {"n_estimators": 5}, runtime_n_jobs=7)
    assert seen["n_jobs"] == 7  # the xcross cv fit read the runtime budget, not the pinned 1
