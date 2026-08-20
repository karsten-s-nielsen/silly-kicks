"""Task 5: cover-shadow RQ1 consumer (`scripts/validate_cover_shadow_rq1.py`)."""

from __future__ import annotations

import json

import pandas as pd
import pytest


def test_headline_fp_rate_is_pass_only_completed():
    from scripts.validate_cover_shadow_rq1 import compute_cover_shadow_metrics

    df = pd.DataFrame(
        {
            "is_blocked_majority": [True, True, True, False, True],
            "is_completed": [True, True, False, True, True],
            "is_fail": [False, False, True, False, False],
            "is_cross": [False, False, False, False, True],  # idx4 CROSS -> excluded from headline
            "p_blocked_center": [0.9, 0.8, 0.7, 0.1, 0.9],
            "p_blocked_mean": [0.8, 0.7, 0.6, 0.1, 0.9],
            "p_blocked_max": [0.95, 0.9, 0.8, 0.2, 0.95],
            "p_received_center": [0.3, 0.3, 0.4, 0.6, 0.3],
            "p_received_left": [0.3, 0.3, 0.4, 0.6, 0.3],
            "p_received_right": [0.3, 0.3, 0.4, 0.6, 0.3],
            "n_blocked": [3, 3, 2, 0, 3],
        }
    )
    m = compute_cover_shadow_metrics(df)
    # PASS-ONLY completed = idx 0,1,3 (cross idx4 dropped) -> blocked-majority at 0,1 -> 2/3
    assert abs(m["headline_fp_rate"]["majority"] - 2 / 3) < 1e-9
    assert "pass_plus_cross_secondary" in m  # paper-comparable cut retains the cross
    assert m["paper_reconciliation"]["required_sentence"]  # Q1
    assert "OVER-PREDICTION" in m["scope_note"]
    # optimistic AUC leads with the discriminating margin score, keeps absolute p_blocked alongside
    assert set(m["optimistic"]["auc"]) == {"n_blocked", "margin_mean", "abs_p_blocked"}


def test_margin_score_discriminates_where_absolute_p_blocked_does_not():
    """The enhancement: the discriminating continuous score is `n_blocked` / the margin the majority
    rule counts, NOT the absolute `p_blocked` magnitude. Here `n_blocked` perfectly separates fail
    from complete while `p_blocked_mean` is constant (AUC 0.5) -- the real-run finding, in miniature."""
    from scripts.validate_cover_shadow_rq1 import compute_cover_shadow_metrics

    n = 6
    df = pd.DataFrame(
        {
            "is_blocked_majority": [True, True, True, False, False, False],
            "is_completed": [False, False, False, True, True, True],
            "is_fail": [True, True, True, False, False, False],
            "is_cross": [False] * n,
            "p_blocked_center": [0.5] * n,
            "p_blocked_mean": [0.5] * n,  # constant magnitude -> AUC 0.5
            "p_blocked_max": [0.5] * n,
            "p_received_center": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],  # margin +0.4 (fail) vs -0.4 (complete)
            "p_received_left": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            "p_received_right": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            "n_blocked": [3, 3, 3, 0, 0, 0],  # perfectly separates
        }
    )
    auc = compute_cover_shadow_metrics(df)["optimistic"]["auc"]
    assert auc["n_blocked"] == 1.0  # the count the majority rule thresholds discriminates perfectly
    assert auc["margin_mean"] == 1.0  # p_blocked - p_received: +0.4 vs -0.4
    assert abs(auc["abs_p_blocked"]["mean"] - 0.5) < 1e-9  # the magnitude alone does NOT (the ~0.5 finding)


def test_consumer_refuses_dirty_upstream(tmp_path, monkeypatch):
    import sys

    from scripts import validate_cover_shadow_rq1 as V

    scores = tmp_path / "pass_scores.parquet"
    pd.DataFrame(
        {
            "is_blocked_majority": [True],
            "is_completed": [True],
            "is_fail": [False],
            "is_cross": [False],
            "p_blocked_center": [0.9],
            "p_blocked_mean": [0.8],
            "p_blocked_max": [0.95],
        }
    ).to_parquet(scores)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "rq-scores-1", "run_tree_dirty": True, "run_commit": "abc", "n_passes": 1})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_cover_shadow_rq1.py", "--pass-scores", str(scores), "--out", str(tmp_path / "art"), "--allow-dirty"],
    )
    with pytest.raises(SystemExit) as exc:
        V.main()
    assert exc.value.code != 0  # refuses a dirty upstream artifact (ADR-037)
