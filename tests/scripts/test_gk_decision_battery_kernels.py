"""Known-truth tests for the pure stat kernels of the owner-run gk_decision battery.

The battery's corpus ORCHESTRATION is owner-run (needs the owner-tier corpus) and is NOT exercised
here -- only its pure kernels (icc1 / club_adjusted_residuals / transfer_signs), plus the module loads.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "_vgd", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate_gk_decision.py"
)
assert _spec is not None and _spec.loader is not None
vgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vgd)


def test_icc1_no_positive_signal_when_group_means_identical():
    # identical group means -> between-group variance 0 -> ICC(1) is NON-POSITIVE (msb < msw), i.e. no
    # positive discrimination signal (the property the permutation-null gate relies on: obs !> null).
    v = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    g = np.repeat([0, 1, 2], 3)
    assert vgd.icc1(v, g) < 0.05


def test_icc1_high_when_groups_separated():
    v = np.array([0.0, 0.1, 0.0, 5.0, 5.1, 4.9, 10.0, 10.1, 9.9])
    g = np.repeat([0, 1, 2], 3)
    assert vgd.icc1(v, g) > 0.9


def test_club_adjusted_removes_team_level():
    # two teams, constant per-team offset -> leave-one-keeper-out residuals center at 0
    df = pd.DataFrame(
        {
            "team": ["a"] * 4 + ["b"] * 4,
            "keeper": [1, 1, 2, 2, 3, 3, 4, 4],
            "x": [10, 10, 12, 12, 20, 20, 22, 22],
        }
    )
    r = vgd.club_adjusted_residuals(df, "x")
    assert abs(r.dropna().mean()) < 1e-9


def test_transfer_signs_guard_below_three_crossing_keepers():
    # < 3 usable transfer keepers -> the sign-agreement / corr are meaningless -> NaN (guard fires),
    # but the count is still reported (dropped-and-counted, not a crash).
    df = pd.DataFrame(
        {
            "team": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "keeper": [1, 1, 1, 2, 1, 1, 1, 3],  # only keeper 1 crosses a->b
            "x": [9.0, 9.0, 9.0, 1.0, 19.0, 19.0, 19.0, 11.0],
        }
    )
    out = vgd.transfer_signs(df, "x", min_per_team=3)
    assert out["n_transfer_keepers"] == 1
    assert np.isnan(out["sign_agreement"]) and np.isnan(out["resid_corr"])


def test_transfer_signs_all_travel_when_ranks_consistent():
    # 3 keepers crossing a<->b, each holding a consistent rank at BOTH clubs (club offset +10):
    # keeper 1 always above its two teammates, keeper 3 always below -> residuals same sign at both
    # clubs -> sign-agreement 1.0 and a perfect residual correlation.
    rows = []
    for club, offset in [("a", 0.0), ("b", 10.0)]:
        for keeper, val in [(1, 10.0), (2, 6.0), (3, 2.0)]:
            rows += [{"team": club, "keeper": keeper, "x": val + offset}] * 3  # >= min_per_team
    out = vgd.transfer_signs(pd.DataFrame(rows), "x", min_per_team=3)
    assert out["n_transfer_keepers"] == 3
    assert out["sign_agreement"] == pytest.approx(1.0)
    assert out["resid_corr"] == pytest.approx(1.0)


def test_icc_vs_permutation_detects_separated_keepers():
    # two keepers whose values are cleanly separated -> observed ICC exceeds the permutation null p95.
    df = pd.DataFrame(
        {
            "keeper": ["A"] * 6 + ["B"] * 6,
            "m": [0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 5.0, 5.1, 4.9, 5.05, 4.95, 5.0],
        }
    )
    out = vgd.icc_vs_permutation(df, "m", n_perm=300)
    assert out["n_keepers"] == 2
    assert out["icc"] > out["null_p95"] and out["p"] < 0.05


def test_reduce_samples_verdict_structure_and_counts():
    # end-to-end pooling of per-decision native samples into the aggregate verdicts (all kernels wired).
    rows = []
    for keeper, team, base in [("A", "t1", 0.6), ("B", "t1", 0.4), ("C", "t2", 0.55), ("D", "t2", 0.45)]:
        for i in range(6):  # >= _MIN_PER_KEEPER so the ICC filter keeps every keeper
            rows.append(
                dict(
                    game_id="g",
                    period_id=1,
                    decision_id=f"{keeper}{i}",
                    keeper=keeper,
                    keeper_raw=keeper,
                    team_id=team,
                    decision_value=base - 0.5,
                    chosen_ev=1.0,
                    best_ev=2.0,
                    sel_efficiency=base,
                    decision_pct=base,
                    n_options=3,
                    option_set_source="native",
                )
            )
    v = vgd.reduce_samples(pd.DataFrame(rows))
    assert v["n_decisions"] == 24 and v["n_keepers"] == 4 and v["n_teams"] == 2
    assert set(v) >= {"responsiveness", "discrimination_one_way", "net_of_team", "transfer"}
    assert set(v["responsiveness"]) >= {
        "decision_pct_mean",
        "decision_pct_t_vs_0.5",
        "decision_value_mean",
        "decision_value_t_vs_0",
    }
    for col in ("decision_value", "sel_efficiency", "decision_pct"):
        assert "icc" in v["discrimination_one_way"][col]
        assert {"club_adjusted", "team_fixed_effect", "team_one_way"} <= set(v["net_of_team"][col])


def test_fidelity_spearman_perfect_reversed_and_shared_key_join():
    recon = pd.DataFrame({"keeper": ["A", "B", "C", "D"], "game_id": ["g"] * 4, "decision_value": [0.1, 0.2, 0.3, 0.4]})
    agree = pd.DataFrame({"keeper": ["A", "B", "C", "D"], "game_id": ["g"] * 4, "decision_value": [1.0, 2.0, 3.0, 4.0]})
    out = vgd.fidelity_spearman(recon, agree, value_col="decision_value", keys=["keeper", "game_id"])
    assert out["rho"] == pytest.approx(1.0) and out["n"] == 4  # monotone-agreeing ranking
    rev = pd.DataFrame({"keeper": ["A", "B", "C", "D"], "game_id": ["g"] * 4, "decision_value": [4.0, 3.0, 2.0, 1.0]})
    assert vgd.fidelity_spearman(recon, rev, value_col="decision_value", keys=["keeper", "game_id"])[
        "rho"
    ] == pytest.approx(-1.0)
    # non-shared keys are dropped from the join (E is only in native) -> n == 3 shared, counts reported
    partial = pd.DataFrame(
        {"keeper": ["A", "B", "C", "E"], "game_id": ["g"] * 4, "decision_value": [1.0, 2.0, 3.0, 9.0]}
    )
    p = vgd.fidelity_spearman(recon, partial, value_col="decision_value", keys=["keeper", "game_id"])
    assert p["n"] == 3 and p["n_recon"] == 4 and p["n_native"] == 4


def test_reachability_sweep_threshold_grid_recommends_crossing():
    # per decision: a short chosen (byp 0, high completion) + 2 near alts (byp 0) it beats + 3 far alts
    # (completion 0.72, byp 4 -> high EV via the progression term) that beat it. At low threshold the far
    # options invert decision_pct; at 0.85 they are pruned (0.72 < 0.85) leaving chosen + 2 near (3 options)
    # -> chosen beats both -> responsive. 0.90 prunes a near (0.88 < 0.90) -> < min_options -> no decisions.
    def _dec(did):
        base = dict(game_id="g", decision_id=did)
        return [
            {**base, "is_chosen": True, "completion": 0.95, "opponents_bypassed": 0},
            {**base, "is_chosen": False, "completion": 0.90, "opponents_bypassed": 0},
            {**base, "is_chosen": False, "completion": 0.88, "opponents_bypassed": 0},
            {**base, "is_chosen": False, "completion": 0.72, "opponents_bypassed": 4},
            {**base, "is_chosen": False, "completion": 0.73, "opponents_bypassed": 4},
            {**base, "is_chosen": False, "completion": 0.71, "opponents_bypassed": 4},
        ]

    rows = pd.DataFrame(_dec("d1") + _dec("d2"))
    out = vgd.reachability_sweep(rows, thresholds=(0.5, 0.85, 0.9))
    by = {c["threshold"]: c for c in out["curve"]}
    assert by[0.5]["decision_pct_mean"] < 0.5  # far high-EV options invert the signal
    assert by[0.85]["decision_pct_mean"] > 0.5 and by[0.85]["n_decisions"] == 2  # pruned -> responsive
    assert by[0.9]["n_decisions"] == 0  # 0.90 prunes a near option -> < min_options -> dropped
    assert out["recommended_threshold"] == 0.85 and out["shipped_default"] == 0.85
