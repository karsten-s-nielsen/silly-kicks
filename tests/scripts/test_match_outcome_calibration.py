"""TF-53 Rung-2 calibration kernels (3-way Brier / calibration slope / CV rho / reduce)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.validate_match_outcome_calibration import (
    calibration_slope,
    cv_rho_by_fold,
    reduce_calibration,
    team_outcome_probabilities,
    three_way_brier,
)
from silly_kicks.match_outcome import apply_dependence, goal_count_pmf
from tests.scripts.test_match_outcome_train import _fake_corpus, _patch_open_corpus


def test_three_way_brier_perfect_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert three_way_brier(probs, [0, 1, 2]) == 0.0


def test_three_way_brier_uniform_is_two_thirds():
    probs = np.full((5, 3), 1.0 / 3.0)
    assert abs(three_way_brier(probs, [0, 1, 2, 0, 1]) - 2.0 / 3.0) < 1e-12


def test_calibration_slope_recovers_one_on_calibrated_data():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=6000)
    outcomes = np.array([rng.choice(3, p=p) for p in probs])  # y ~ Categorical(p) -> calibrated
    assert abs(calibration_slope(probs, outcomes) - 1.0) < 0.15


def test_calibration_slope_below_one_when_overconfident():
    # Concentrated base (no near-0/1 probs -> no separability artifact); outcomes from the base truth,
    # predictions MILDLY sharpened -> over-confident, so recalibration must shrink the logits (slope<1).
    rng = np.random.default_rng(1)
    base = rng.dirichlet(np.full(3, 4.0), size=6000)
    outcomes = np.array([rng.choice(3, p=p) for p in base])
    sharp = base**1.5
    sharp = sharp / sharp.sum(axis=1, keepdims=True)
    slope = calibration_slope(sharp, outcomes)
    assert 0.3 < slope < 0.95


def _simulate_cv_matches(rho_true: float, *, n: int = 120, seed: int = 0) -> list[tuple]:
    rng = np.random.default_rng(seed)
    matches = []
    for k in range(n):
        hx = rng.uniform(0.05, 0.4, size=int(rng.integers(3, 9))).tolist()
        ax = rng.uniform(0.05, 0.4, size=int(rng.integers(3, 9))).tolist()
        joint = apply_dependence(goal_count_pmf(hx), goal_count_pmf(ax), rho=rho_true)
        flat = joint.flatten()
        i, j = np.unravel_index(int(rng.choice(flat.size, p=flat / flat.sum())), joint.shape)
        matches.append((f"g{k}", hx, ax, int(i), int(j)))
    return matches


def test_cv_rho_folds_are_disjoint_and_cover_all():
    folds = cv_rho_by_fold(_simulate_cv_matches(0.08), 3)
    assert len(folds) == 3
    all_games = {f"g{k}" for k in range(120)}
    covered_test = set()
    for fo in folds:
        train, test = set(fo["train_game_ids"]), set(fo["test_game_ids"])
        assert train.isdisjoint(test)  # held-out never in the fit set
        assert train | test == all_games  # each fold splits the full corpus
        covered_test |= test
        assert np.isfinite(fo["rho"])
    assert covered_test == all_games  # every game is held out in exactly one fold


def test_cv_rho_recovers_positive_dependence():
    folds = cv_rho_by_fold(_simulate_cv_matches(0.12, n=200), 4)
    assert float(np.mean([fo["rho"] for fo in folds])) > 0.03


def test_team_outcome_probabilities_independent_matches_outer():
    own, opp = goal_count_pmf([0.5, 0.3]), goal_count_pmf([0.4])
    pw, pd_, pl = team_outcome_probabilities(own, opp, rho=None)
    assert abs(pw + pd_ + pl - 1.0) < 1e-12


def test_reduce_calibration_per_config():
    df = pd.DataFrame(
        {
            "config": ["independent"] * 2 + ["dixon_coles"] * 2,
            "game_id": ["g0", "g0", "g0", "g0"],
            "team_id": [1, 2, 1, 2],
            "p_win": [1.0, 0.0, 0.9, 0.05],
            "p_draw": [0.0, 0.0, 0.05, 0.05],
            "p_loss": [0.0, 1.0, 0.05, 0.9],
            "xpoints": [3.0, 0.0, 2.75, 0.2],
            "outcome": [0, 2, 0, 2],
            "points": [3, 0, 3, 0],
        }
    )
    summary = reduce_calibration([df])
    out = summary.set_index("config").to_dict("index")
    assert out["independent"]["brier"] == 0.0  # perfect predictions
    assert out["independent"]["n"] == 2
    assert out["dixon_coles"]["brier"] > 0.0
    assert set(summary.columns) >= {"brier", "calibration_slope", "mean_xpoints", "mean_points", "xpoints_bias"}


@pytest.mark.slow
def test_calibration_main_writes_metrics(tmp_path, monkeypatch):
    """Does-it-run smoke for main() end-to-end on a synthetic corpus (no network; --allow-dirty)."""
    from scripts import validate_match_outcome_calibration as cal

    _patch_open_corpus(monkeypatch, _fake_corpus())
    out = tmp_path / "cal"
    monkeypatch.setattr(
        sys, "argv", ["validate_match_outcome_calibration.py", "--out", str(out), "--folds", "2", "--allow-dirty"]
    )
    cal.main()

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_matches"] == 6
    configs = {r["config"] for r in metrics["per_config"]}
    assert configs == {"independent", "collapse", "dixon_coles", "both"}
    assert metrics["input_contract"]["driver"] == "validate_match_outcome_calibration"
    assert "run_commit" in metrics and len(metrics["cv_rho_by_fold"]) == 2


def test_stats_prepass_reduce_equals_in_memory():
    """ADR-052 U-shape: the per-match stats prepass + reduce rebuild the SAME CV corpus the
    pre-migration in-memory pass built, so cv_rho_by_fold fits identical per-fold rho (order-independent
    -- the reduce sorts game ids)."""
    from scripts.validate_match_outcome_calibration import (
        _team_stats,
        extract_stats_slice,
        stats_from_shards,
    )
    from silly_kicks.match_outcome import MatchOutcomeParams

    gap = MatchOutcomeParams().possession_max_gap_seconds
    corpus = _fake_corpus(6)

    in_memory = []
    for _p, mid, actions, _f, _h in corpus:
        stats = _team_stats(actions, "xg", gap=gap)
        if stats is None:
            continue
        teams = list(stats.keys())
        in_memory.append(
            (
                mid,
                stats[teams[0]]["indep"],
                stats[teams[1]]["indep"],
                stats[teams[0]]["goals"],
                stats[teams[1]]["goals"],
            )
        )

    shards = [extract_stats_slice(mid, actions, gap=gap) for _p, mid, actions, _f, _h in corpus]
    by_game, cv = stats_from_shards(shards)
    assert len(by_game) == len(in_memory)
    assert [f["rho"] for f in cv_rho_by_fold(cv, 2)] == [f["rho"] for f in cv_rho_by_fold(in_memory, 2)]
    assert [f["rho"] for f in cv_rho_by_fold(list(reversed(cv)), 2)] == [f["rho"] for f in cv_rho_by_fold(in_memory, 2)]
