"""Task 7: the reliability kernels are a SHARED single source (``scripts._reliability``).

Two drivers reduce per-shard samples into the same ICC(1) / split-half / Type-II statistics -- the TF-52
team-KPI study and the TF-62 GK decision battery. This proves both drivers now bind the ONE
implementation (object identity) and that the move changed no value (seeded expectations captured from the
unmoved functions before the extraction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import _reliability as r
from scripts import validate_gk_decision as g
from scripts import validate_team_kpi_reliability as v


def test_reliability_functions_are_shared():
    assert v.icc1 is r.icc1
    assert g.icc1 is r.icc1
    assert v.split_half_reliability is r.split_half_reliability
    assert v.type_ii_slope is r.type_ii_slope
    assert v.compare_providers is r.compare_providers


def _seeded_samples() -> pd.DataFrame:
    """5 teams x 6 matches: a team-structured KPI + a structureless one (deterministic PCG64 stream)."""
    rng = np.random.default_rng(20260927)
    labels = [f"t{i}" for i in range(5)]
    teams = np.repeat(labels, 6)
    team_eff = {f"t{i}": float(val) for i, val in enumerate(rng.normal(0, 2, 5))}
    reliable = np.array([team_eff[t] + rng.normal(0, 0.3) for t in teams])
    noise = rng.normal(0, 2, len(teams))
    games = np.array([f"{t}_g{j}" for t in labels for j in range(6)])
    return pd.DataFrame({"team_id": teams, "game_id": games, "reliable": reliable, "noise": noise})


# Literal expectations captured from the UNMOVED functions BEFORE the extraction (plan Task 7 Step 0).
_XS = [
    -0.1270538547599662,
    0.1366446099104383,
    -0.30271324883916967,
    0.39573104837258927,
    -0.20815544484712356,
    0.7870700922012389,
    -0.7192199086682897,
    0.004960316350584543,
    -0.3095729468837365,
    0.6434234195019936,
    -0.028128859003025062,
    -0.3990585799653824,
]
_YS = [
    -0.2627026610410541,
    0.2427463076043863,
    -0.3764404756978399,
    0.8377063736611425,
    -0.46558302308095467,
    1.4517618929067024,
    -1.5502740748656623,
    0.03428527399912257,
    -0.5624189658812434,
    1.3195155879847953,
    -0.21084231743339962,
    -0.7857794767989867,
]


def test_reliability_values_unchanged():
    samples = _seeded_samples()
    teams = samples["team_id"].to_numpy()
    assert r.icc1(samples["reliable"].to_numpy(), teams) == pytest.approx(0.960127560812556)
    assert r.icc1(samples["noise"].to_numpy(), teams) == pytest.approx(0.030337583820999516)

    sh_rel = r.split_half_reliability(samples, "reliable")
    assert sh_rel["r"] == pytest.approx(0.9886331051031944)
    assert sh_rel["n_teams"] == 5
    sh_noise = r.split_half_reliability(samples, "noise")
    assert sh_noise["r"] == pytest.approx(-0.8845126568016232)
    assert sh_noise["n_teams"] == 5

    assert r.type_ii_slope(np.array(_XS), np.array(_YS)) == pytest.approx(1.9863317877940767)
    # A-36: it is the SMA/RMA slope sign(r)*sd(y)/sd(x), NOT the OLS slope r*sd(y)/sd(x) (the mislabel was fixed).
    x, y = np.array(_XS, dtype=float), np.array(_YS, dtype=float)
    rho = float(np.corrcoef(x, y)[0, 1])
    sma = float(np.sign(rho) * y.std() / x.std())
    ols = float(rho * y.std() / x.std())
    assert r.type_ii_slope(x, y) == pytest.approx(sma)
    assert abs(r.type_ii_slope(x, y) - ols) > 1e-6  # distinct from OLS (|r| < 1 here)

    reports = {
        "pa": {
            "reliability": {
                "per_kpi": {"k1": {"icc": 0.6, "split_half_r": 0.5}, "k2": {"icc": 0.2, "split_half_r": 0.1}}
            }
        },
        "pb": {
            "reliability": {
                "per_kpi": {"k1": {"icc": 0.65, "split_half_r": 0.55}, "k2": {"icc": -0.1, "split_half_r": 0.0}}
            }
        },
    }
    cmp = r.compare_providers(reports)
    assert cmp["n_providers"] == 2
    assert cmp["per_kpi"]["k1"]["poolable"] is True
    assert cmp["per_kpi"]["k1"]["icc_spread"] == pytest.approx(0.05)
    assert cmp["per_kpi"]["k2"]["poolable"] is False
    assert cmp["per_kpi"]["k2"]["icc_spread"] == pytest.approx(0.30)


def test_shared_icc1_matches_both_drivers_numerically():
    """Non-vacuity: the shared icc1 reproduces the gk_decision leg's numbers too (not only team-KPI)."""
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(6), 8)
    eff = rng.normal(0, 1, 6)
    vals = eff[groups] + rng.normal(0, 0.5, groups.size)
    # both driver namespaces expose the SAME object, so this is one value, asserted from both bindings.
    assert v.icc1(vals, groups) == g.icc1(vals, groups) == r.icc1(vals, groups)
