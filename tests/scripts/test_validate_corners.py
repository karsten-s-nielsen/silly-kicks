"""CI-safe unit slice for scripts/validate_corners.py (pure stat kernels; corpus battery is e2e)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.validate_corners import icc_vs_permutation, sb360_init_divergence


def test_icc_detects_team_structure_vs_null():
    # three teams, each with its own tight value level -> high ICC, clears the permutation null
    rng = np.random.default_rng(0)
    rows = []
    for team, level in (("A", 0.1), ("B", 0.5), ("C", 0.9)):
        for _ in range(20):
            rows.append({"team_id": team, "sum_xg": level + rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)
    res = icc_vs_permutation(df, "sum_xg", "team_id", n_perm=200)
    assert res["icc"] > res["null_p95"]  # real between-team structure clears the null
    assert res["p"] < 0.05
    assert res["n"] == 60


def test_icc_null_on_unstructured_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"team_id": rng.integers(0, 3, size=60), "sum_xg": rng.normal(0.5, 0.1, size=60)})
    res = icc_vs_permutation(df, "sum_xg", "team_id", n_perm=200)
    assert res["icc"] <= res["null_p95"] or res["p"] > 0.05  # no real structure -> does not clear the null


def test_sb360_init_divergence_bounds_the_reassignment():
    from silly_kicks.corners._zones import RunZoneModel

    rng = np.random.default_rng(0)
    init = np.vstack([rng.normal([90.0, 25.0], 1.0, size=(60, 2)), rng.normal([90.0, 43.0], 1.0, size=(60, 2))])
    target = np.vstack(
        [
            rng.normal([100.0, 28.0], 1.0, size=(40, 2)),
            rng.normal([100.0, 40.0], 1.0, size=(40, 2)),
            rng.normal([80.0, 10.0], 1.0, size=(40, 2)),
        ]
    )
    model = RunZoneModel().fit(init, target, init_k=2, target_k=3)
    init_xy = np.array([[90.0, 25.0], [90.0, 43.0]])
    kick_xy = init_xy + 0.5  # small shift -> low divergence
    res = sb360_init_divergence(init_xy, kick_xy, model)
    assert res["n_players"] == 2
    assert res["delta_mean_m"] == pytest.approx(np.hypot(0.5, 0.5))
    assert 0.0 <= res["zone_reassignment_rate"] <= 1.0


def test_injection_firewall_no_outcome_vector():
    # the power harness takes an InjectionSpec recipe, never an observed-outcome vector (unrepresentable)
    import inspect

    from silly_kicks.causal.power import att_power_curve

    params = set(inspect.signature(att_power_curve).parameters)
    assert "injection" in params
    assert "Y" not in params and "outcomes" not in params  # observed outcome is structurally unrepresentable
