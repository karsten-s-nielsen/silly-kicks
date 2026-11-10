"""The xsuccess `fit` stage as 36 units gives the bundle of the 4.140.0 `fit` (HFU fit-units spec 3.2, 3.3).

Each test seeds the `refs`, `extract` and `hpo` outputs of a real `RunContext`, runs every `fit` unit, writes each
result to the store and reads it back (so every result crosses the store encoding), runs the reduce, and compares
the bundle with the oracle: `tests/scripts/_xsuccess_fit_oracle.py`, the old `_fit_work` on the old model code.
The comparison is the RAW bytes of every file. The two runs share one context, so no provenance differs: a key
order, an indent or a dtype change must fail. This is the store path of `--run-all` without the open-data
server; `tests/scripts/test_parity_xsuccess.py` runs the real `--run-all`.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as cfg

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

import scripts.train_xsuccess as T
from scripts import _stages as s
from silly_kicks.xsuccess import _model as M
from silly_kicks.xsuccess._features import xsuccess_features
from tests.scripts._stage_harness import PROV, params_from_args
from tests.scripts._xsuccess_fit_oracle import oracle_fit_work

_SUCCESS, _FAIL = cfg.result_id["success"], cfg.result_id["fail"]


def _pooled(sizes: dict[str, int], *, seed: int = 5, nan_rows: dict[str, int] | None = None) -> pd.DataFrame:
    """The `extract` output shape: on-ball rows of the matches ``sizes`` (``{game_id: rows}``), in match order.
    ``nan_rows``: ``{game_id: n}`` rows of that match get a NaN ``start_x`` (a non-finite feature)."""
    rng = np.random.default_rng(seed)
    frames = []
    for game, n in sizes.items():
        x = rng.uniform(0, 105, n)
        ok = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(2.0 - 0.04 * (105.0 - x))))
        f = pd.DataFrame(
            {
                "game_id": game,
                "period_id": 1,
                "type_id": cfg.actiontype_id["pass"],
                "result_id": np.where(ok, _SUCCESS, _FAIL),
                "bodypart_id": cfg.bodypart_id["foot"],
                "start_x": x,
                "start_y": rng.uniform(0, 68, n),
                "time_seconds": rng.uniform(0, 3000, n),
            }
        )
        f.loc[f.index[: (nan_rows or {}).get(game, 0)], "start_x"] = np.nan
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


_SIX = {"m1": 90, "m2": 80, "m3": 42, "m4": 41, "m5": 30, "m6": 20}
# Study parameters, as production has them (`study.best_params`: the search keys only, never None there).
_STUDY = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.07, "subsample": 0.8, "min_child_weight": 2}


def _calibrated() -> pd.DataFrame:
    """Seven matches. m3 has 42 rows and 40 finite rows; m4 has 41. So GroupKFold puts the smallest match,
    m7, with m4 in the row space of all pooled rows and with m3 in the row space of the finite rows: the
    two row spaces give different fold PARTITIONS (A-R2)."""
    return _pooled({**_SIX, "m7": 15}, nan_rows={"m3": 2})


def _few_failures() -> pd.DataFrame:
    """Four failures in four matches: the final fit has 4 calibration splits, a fold fit has fewer."""
    df = _pooled(_SIX)
    df["result_id"] = _SUCCESS
    for game in ("m1", "m2", "m3", "m4"):
        df.loc[df.index[df["game_id"] == game][0], "result_id"] = _FAIL
    return df


def _all_fail() -> pd.DataFrame:
    df = _pooled(_SIX)
    df["result_id"] = _FAIL
    return df


def _ctx(tmp_path, pooled: pd.DataFrame, *, family: str = "xgboost", best: dict | None = None) -> s.RunContext:
    """A real context of the xsuccess pipeline with its `refs`, `extract` and `hpo` outputs seeded."""
    params = params_from_args(T.PIPELINE, ["--family", family])
    ctx = s.make_context(T.PIPELINE, params, PROV, store=s.LocalStore(tmp_path), data_root=None)
    games = sorted(pooled["game_id"].unique()) if len(pooled) else []
    refs = {"competitions": [[43, 106]], "refs": [{"match_id": g} for g in games]}
    s.write_result(ctx.store, f"{ctx.generation('refs')}/_output", refs)
    s.write_result(ctx.store, f"{ctx.generation('extract')}/_output", pooled)
    summary = {"n_units": len(games), "n_failed": 0, "n_excluded": 0}
    ctx.store.write_bytes(f"{ctx.generation('extract')}/_summary.json", json.dumps(summary).encode("utf-8"))
    s.write_result(ctx.store, f"{ctx.generation('hpo')}/_output", {"best_params": best})
    return ctx


def _unit_results(ctx) -> dict:
    """Every `fit` unit, written to the store and read back through the verified read."""
    gen = ctx.generation("fit")
    for key in T.FIT_UNITS:
        s.write_result(ctx.store, f"{gen}/{key}", T._fit_unit_work(key, ctx))
        ctx.store.write_bytes(f"{gen}/{key}.ledger.json", b"{}")
    results = s.done_results(ctx, "fit")
    assert sorted(results) == list(T.FIT_UNITS)
    return results


def _absent(results: dict) -> set[str]:
    return {k for k, v in results.items() if isinstance(v, dict) and v == T.ABSENT}


def _run(tmp_path, pooled, *, family: str = "xgboost", best: dict | None = None):
    ctx = _ctx(tmp_path, pooled, family=family, best=best)
    results = _unit_results(ctx)
    new = T._fit_reduce(results, ctx)
    old = oracle_fit_work("only", ctx)
    return results, new, old


def _partition(frame: pd.DataFrame) -> set[frozenset[str]]:
    """The matches of each test fold of `_cv_splits` in the row space of ``frame``."""
    frame = frame.reset_index(drop=True)
    _y, splits = T._cv_splits(frame)
    games = frame["game_id"].to_numpy()
    return {frozenset(str(g) for g in games[te]) for _tr, te in splits}


def test_the_unit_grid_is_36_sorted_keys():
    assert len(T.FIT_UNITS) == 36 and list(T.FIT_UNITS) == sorted(T.FIT_UNITS)
    assert {"final-main", "final-cal0", "final-cal4", "fold0-main", "fold4-cal4"} <= set(T.FIT_UNITS)


def test_calibrated_xgboost_equals_the_oracle(tmp_path):
    pooled = _calibrated()
    # non-vacuity (A-R2): unequal matches, non-finite rows, and a fold that DEPENDS on the row space
    assert len(set(pooled["game_id"].value_counts())) > 1
    finite = np.isfinite(xsuccess_features(pooled)).all(axis=1)
    assert (~finite).sum() == 2

    # The PARTITION of the matches into folds differs (a fold NUMBER alone changes no output byte).
    assert _partition(pooled) != _partition(pooled[finite])

    results, new, old = _run(tmp_path, pooled, best=_STUDY)
    assert _absent(results) == set()  # all 36 units exist
    assert results["final-cal0"]["p"].dtype == np.float32  # the store keeps the calibration dtype
    assert new.files == old.files
    assert json.loads(new.files["metadata.json"])["isotonic"] is not None


def test_fewer_calibration_splits_equals_the_oracle(tmp_path):
    results, new, old = _run(tmp_path, _few_failures())
    absent = _absent(results)
    assert "final-cal4" in absent and "final-cal0" not in absent and "final-main" not in absent
    assert new.files == old.files


def test_one_class_equals_the_oracle(tmp_path):
    """Review focus 5: a fit whose kept rows hold one class has no calibration."""
    results, new, old = _run(tmp_path, _all_fail())
    assert {k for k in T.FIT_UNITS if "-cal" in k} <= _absent(results)
    assert new.files == old.files
    assert json.loads(new.files["metadata.json"])["isotonic"] is None


def test_per_type_logistic_equals_the_oracle(tmp_path):
    results, new, old = _run(tmp_path, _calibrated(), family="per_type_logistic")
    assert _absent(results) == {k for k in T.FIT_UNITS if "-cal" in k}  # all 30 calibration units
    assert new.files == old.files


def test_one_match_equals_the_oracle(tmp_path):
    results, new, old = _run(tmp_path, _pooled({"m1": 120}))
    assert {k for k in T.FIT_UNITS if k.startswith("fold")} <= _absent(results)
    assert "in_sample_raw.npy" in results["final-main"].files
    assert new.files == old.files


def test_fit_reduce_does_not_import_xgboost(tmp_path, monkeypatch):
    pooled = _calibrated()
    ctx = _ctx(tmp_path, pooled, best=_STUDY)
    results = _unit_results(ctx)
    monkeypatch.setitem(sys.modules, "xgboost", None)  # an import of xgboost now raises ImportError
    metrics, files, feature_set = T.fit_reduce(pooled, results, family="xgboost", params=_STUDY)
    assert feature_set == "xgboost" and set(files) == {"model.json", "metadata.json", "SHA256SUMS"}
    assert metrics["n_oof"] == len(pooled) - 2


def test_an_empty_corpus_stops_with_todays_message(tmp_path):
    ctx = _ctx(tmp_path, _calibrated().iloc[0:0])
    with pytest.raises(SystemExit, match="no on-ball rows collected from the corpus; nothing to fit"):
        T._fit_unit_work("final-main", ctx)
    with pytest.raises(SystemExit, match="no on-ball rows collected from the corpus; nothing to fit"):
        T._fit_reduce({}, ctx)


def test_the_shape_test_detects_a_fault(tmp_path, monkeypatch):
    """A deliberate fault in the reduce path (1e-9 on a calibrated value) must change the bundle bytes."""
    ctx = _ctx(tmp_path, _calibrated(), best=_STUDY)
    results = _unit_results(ctx)
    old = oracle_fit_work("only", ctx)
    real = M._calibrate
    monkeypatch.setattr(M, "_calibrate", lambda raw, x, y: real(raw, x, y) + 1e-9)
    assert T._fit_reduce(results, ctx).files != old.files


def test_the_shape_test_detects_folds_from_the_finite_row_space(tmp_path, monkeypatch):
    """A-R2: folds taken in the row space of the finite rows (not of all pooled rows) must change the bundle
    bytes. The fault keeps every other step: only the fold of a match can move."""
    pooled = _calibrated()
    ctx = _ctx(tmp_path, pooled, best=_STUDY)
    old = oracle_fit_work("only", ctx)
    real = T._cv_splits

    def finite_row_space(frame: pd.DataFrame):
        y, _splits = real(frame)
        pos = np.flatnonzero(np.isfinite(xsuccess_features(frame)).all(axis=1))
        _y, splits = real(frame.iloc[pos].reset_index(drop=True))
        return y, [(pos[tr], pos[te]) for tr, te in splits]

    monkeypatch.setattr(T, "_cv_splits", finite_row_space)
    assert T._fit_reduce(_unit_results(ctx), ctx).files != old.files
