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


def test_reduce_samples_model_free_backbone_and_per_provider_xg():
    # Tier 1: n_goals / n_shots are the model-free backbone (always present, provider-uniform); sum_xg
    # is a per-provider native-xG companion, honest-NaN where a provider supplies no xG (SkillCorner).
    from scripts.validate_corners import reduce_samples

    rows = [
        ("A", "idsse", 1, 3, 0.4),
        ("A", "idsse", 0, 2, 0.2),
        ("B", "statsbomb", 0, 1, 0.1),
        ("B", "statsbomb", 1, 4, 0.6),
        ("A", "skillcorner", 0, 2, float("nan")),
        ("B", "skillcorner", 1, 3, float("nan")),
    ]
    samples = pd.DataFrame(rows, columns=["team_id", "provider", "n_goals", "n_shots", "sum_xg"])
    samples["n_goals"] = samples["n_goals"].astype("Int64")
    samples["n_shots"] = samples["n_shots"].astype("Int64")

    out = reduce_samples(samples)
    assert set(out["responsiveness"]) == {"n_goals_mean", "n_shots_mean", "sum_xg_mean"}
    assert out["responsiveness"]["n_shots_mean"] == pytest.approx(2.5)  # model-free, all six corners
    assert set(out["discrimination_one_way"]) == {"n_goals", "n_shots", "sum_xg"}
    cp = out["cross_provider"]
    assert set(cp) == {"idsse", "statsbomb", "skillcorner"}
    assert cp["skillcorner"]["n_shots_mean"] == pytest.approx(2.5)  # model-free leg present per provider
    assert np.isnan(cp["skillcorner"]["sum_xg_mean"])  # honest-NaN: no native SkillCorner shot xG, never faked


def test_reduce_samples_handles_mixed_dtype_team_ids_across_providers():
    # Real corpus: SkillCorner team_id is str, statsbomb/idsse int -> pooled mixed dtype. A bare
    # np.unique(team_id) sort crashes ("'<' not supported between 'int' and 'str'"); reduce must
    # canonicalize + provider-qualify the team key (ADR-019), and never merge same-spelling ids.
    from scripts.validate_corners import reduce_samples

    rows = [
        ("7", "skillcorner", 0, 2, float("nan")),  # str id
        ("7", "skillcorner", 1, 3, float("nan")),
        (7, "statsbomb", 0, 1, 0.1),  # int id, SAME spelling, DIFFERENT provider/team
        (7, "statsbomb", 1, 2, 0.3),
    ]
    samples = pd.DataFrame(rows, columns=["team_id", "provider", "n_goals", "n_shots", "sum_xg"])
    out = reduce_samples(samples)  # must NOT raise
    assert out["n_teams"] == 2  # skillcorner:7 and statsbomb:7 stay distinct, never merged
    assert "n_shots" in out["discrimination_one_way"] and "n_goals" in out["discrimination_one_way"]


def test_injection_firewall_no_outcome_vector():
    # the power harness takes an InjectionSpec recipe, never an observed-outcome vector (unrepresentable)
    import inspect

    from silly_kicks.causal.power import att_power_curve

    params = set(inspect.signature(att_power_curve).parameters)
    assert "injection" in params
    assert "Y" not in params and "outcomes" not in params  # observed outcome is structurally unrepresentable
