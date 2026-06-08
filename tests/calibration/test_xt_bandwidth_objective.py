"""SK-xT-3 Task 3/4: XtBandwidthObjective unit behavior + cache/perf/round-trip guards."""

import math

import numpy as np
import pytest
from ruthless import Candidate

from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective
from tests._xthreat_helpers import _sparse_overfit_corpus


def _cand(bandwidth=1.0, adaptive=True, grid="16x12"):
    return Candidate(id="t0", params={"bandwidth": bandwidth, "adaptive": adaptive, "grid": grid})


def test_evaluate_returns_finite_nll_and_singh_baseline():
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=1, n_games=20), seed=42)
    m = obj.evaluate(_cand())
    assert math.isfinite(m["xt_holdout_nll"])
    assert math.isfinite(m["singh_holdout_nll"])
    assert m["n_folds"] >= 2
    assert m["n_holdout_moves"] > 0


def test_evaluate_is_deterministic():
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=1, n_games=20), seed=42)
    assert obj.evaluate(_cand())["xt_holdout_nll"] == obj.evaluate(_cand())["xt_holdout_nll"]


def test_grid_axis_changes_the_score():
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=1, n_games=20), seed=42)
    a = obj.evaluate(_cand(grid="12x8"))["xt_holdout_nll"]
    b = obj.evaluate(_cand(grid="24x16"))["xt_holdout_nll"]
    assert a != b


def test_string_and_int_game_id_both_work():
    # provider-asymmetric game_id dtype must not crash CV grouping / NLL.
    df = _sparse_overfit_corpus(seed=2, n_games=20)
    int_score = XtBandwidthObjective(df, seed=42).evaluate(_cand())["xt_holdout_nll"]
    df_str = df.copy()
    df_str["game_id"] = df_str["game_id"].astype(str)
    str_score = XtBandwidthObjective(df_str, seed=42).evaluate(_cand())["xt_holdout_nll"]
    assert math.isfinite(int_score) and math.isfinite(str_score)


def test_mixed_dtype_game_id_does_not_crash():
    # Real-corpus regression (DGX pining run): a multi-provider corpus concatenates int + str
    # game_ids into ONE object column; match_cv_splits -> np.unique -> sort() raised
    # "'<' not supported between instances of 'int' and 'str'". astype(str) in __init__ guards it.
    import pandas as pd

    df = _sparse_overfit_corpus(seed=4, n_games=20)
    gids = df["game_id"].to_numpy(dtype=object)
    uniq = list(pd.unique(gids))
    str_set = set(uniq[: len(uniq) // 2])
    df["game_id"] = [str(g) if g in str_set else int(g) for g in gids]
    assert df["game_id"].map(type).nunique() == 2  # genuinely mixed int/str object column
    m = XtBandwidthObjective(df, seed=42).evaluate(_cand())
    assert math.isfinite(m["xt_holdout_nll"])


def test_no_signal_corpus_scores_inf_not_crash():
    # A corpus with no eligible holdout MOVES competes honestly as the worst score, never crashes.
    # NOTE: needs >=2 games — match_cv_splits uses LeaveOneGroupOut for <=7 games, which RAISES on a
    # single group. Two shot-only games => every fold's holdout has 0 moves => all excluded => inf.
    import pandas as pd

    import silly_kicks.spadl.config as cfg

    cols = [
        "game_id",
        "action_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "bodypart_id",
        "type_id",
        "result_id",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]
    shot, fail = cfg.actiontype_id["shot"], cfg.result_id["fail"]
    rows = [
        [1, 0, 1, 0.0, 1, 1, 0, shot, fail, 95.0, 34.0, 105.0, 34.0],
        [2, 1, 1, 0.0, 1, 1, 0, shot, fail, 95.0, 34.0, 105.0, 34.0],
    ]
    m = XtBandwidthObjective(pd.DataFrame(rows, columns=cols), seed=42).evaluate(_cand())
    assert m["xt_holdout_nll"] == float("inf")


def test_cache_equivalence_warm_equals_cold():
    df = _sparse_overfit_corpus(seed=5, n_games=20)
    cold = XtBandwidthObjective(df, seed=42).evaluate(_cand(bandwidth=2.0, adaptive=False, grid="20x14"))
    warm_obj = XtBandwidthObjective(df, seed=42)
    warm_obj.evaluate(_cand(bandwidth=0.5, adaptive=True, grid="20x14"))  # populate cache at this grid
    warm = warm_obj.evaluate(_cand(bandwidth=2.0, adaptive=False, grid="20x14"))  # reuse cache
    assert warm["xt_holdout_nll"] == pytest.approx(cold["xt_holdout_nll"], abs=1e-12)


def test_binning_cached_once_per_grid_fold_seam_runs_per_trial(monkeypatch):
    # Structural perf guard: over N trials at ONE grid, the (expensive) binning runs once per fold,
    # while the (cheap) gaussian seam runs once per (trial, fold). Patch in the objective module's
    # namespace (it imports the helpers by name).
    import silly_kicks.calibration._xt_bandwidth_objective as mod
    from tests._perf_structural import call_counter

    bin_calls = call_counter(monkeypatch, mod, "_bin_destinations_by_source")
    seam_calls = call_counter(monkeypatch, mod, "_gaussian_transition_from_grouped")
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=6, n_games=20), seed=42)
    n_folds = len(obj._folds)
    n_trials = 4
    for bw in (0.5, 1.0, 2.0, 4.0):
        obj.evaluate(_cand(bandwidth=bw, grid="16x12"))
    assert bin_calls["n"] == n_folds  # binning NOT re-run per trial (cache works)
    assert seam_calls["n"] == n_trials * n_folds  # seam re-runs per (trial, fold)


def test_recommendation_round_trips_into_expected_threat():
    # The recommended config must construct a usable fitted xT (not just emit numbers).
    from silly_kicks.xthreat import ExpectedThreat, KDEParams
    from tests._xthreat_helpers import _corpus_with_shots

    grid = "20x14"
    nx, ny = (int(v) for v in grid.split("x"))
    xt = ExpectedThreat(l=nx, w=ny, method="kde_smoothed", params=KDEParams(bandwidth=1.5, adaptive=True))
    xt.fit(_corpus_with_shots(n_per_zone=40))
    assert np.any(xt.xT > 0)
    assert np.all(np.isfinite(xt.xT))
