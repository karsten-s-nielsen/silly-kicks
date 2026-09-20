"""TF-63 win-probability trainer: smoke + the leakage-free-strength gate (spec §5.2, load-bearing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as C
from scripts.train_win_probability import (
    _attach_strength,
    _games_frame,
    calibration_metrics,
    chronological_strength,
    main,
    match_shard,
)
from silly_kicks.win_probability import (
    WinProbabilityModel,
    WinProbabilityParams,
    compute_win_probability,
)
from silly_kicks.win_probability._chain import expected_total_goals

SHOT = C.actiontype_id["shot"]
PASS = C.actiontype_id["pass"]
SUCC = C.result_id["success"]
FAIL = C.result_id["fail"]


def test_module_imports_without_statsbombpy_and_main_is_argparse_guarded():
    # statsbombpy / _sb_open_data / _driver / _provenance are function-local imports, so the module
    # imports clean in CI (no network dep) and main() does not run on import.
    assert callable(main)
    assert callable(chronological_strength) and callable(match_shard)


def _pooled(m3_a_xg: float) -> pd.DataFrame:
    """3 matches (A=1 vs B=2) on ascending dates; one shot each carrying xg. m3's A-xg is the knob."""
    rows = []
    spec = [
        ("m1", "2020-01-01", 1.0, 0.0),  # A supremacy +1
        ("m2", "2020-01-02", 0.5, 0.5),  # A supremacy  0
        ("m3", "2020-01-03", m3_a_xg, 0.0),  # A supremacy +m3_a_xg (LATER than m1/m2)
    ]
    for gid, date, a_xg, b_xg in spec:
        for team, xg in ((1, a_xg), (2, b_xg)):
            rows.append(
                {
                    "game_id": gid,
                    "period_id": 1,
                    "team_id": team,
                    "player_id": None,
                    "time_seconds": 10.0,
                    "type_id": SHOT,
                    "result_id": SUCC,
                    "xg": xg,
                    "home_team_id": 1,
                    "match_date": date,
                }
            )
    return pd.DataFrame(rows)


def test_strength_is_leakage_free_by_date():
    # spec §5.2: base_strength(match m, team t) uses ONLY strictly-earlier matches. Perturbing a LATER
    # match (m3) must leave an EARLIER match's (m1, m2) strength byte-identical.
    s_lo = chronological_strength(_pooled(0.2))
    s_hi = chronological_strength(_pooled(3.0))
    # earlier matches unchanged under the future perturbation
    assert s_lo[("m1", 1)] == s_hi[("m1", 1)] == 0.0  # no prior -> 0
    assert s_lo[("m2", 1)] == s_hi[("m2", 1)]  # m2's prior is only m1 (unaffected by m3)
    assert abs(s_lo[("m2", 1)] - 1.0) < 1e-12  # supremacy(m1, A) = +1
    # m3 itself DOES change (its prior mean is fixed, but m3's OWN supremacy differs -> future matches
    # would differ; m3's strength is the mean of m1,m2 and is identical, proving m3 uses only priors)
    assert s_lo[("m3", 1)] == s_hi[("m3", 1)]  # strength(m3) = mean(prior m1,m2) -> independent of m3's xg
    assert abs(s_lo[("m3", 1)] - 0.5) < 1e-12  # mean(+1, 0)


def _synthetic_corpus(n_matches: int = 6) -> pd.DataFrame:
    """A tiny shard-shaped pooled corpus (the columns match_shard emits) for the reduce path."""
    rows = []
    for g in range(n_matches):
        gid = f"m{g}"
        home, away = 1, 2
        date = f"2020-01-{g + 1:02d}"
        aid = 0
        # a scored shot for each team at different minutes + filler + a late pass (final minute ~93)
        specs = [
            (1, home, 300.0, SHOT, SUCC, 0.3 + 0.05 * g),  # home scores p1
            (1, away, 1500.0, SHOT, FAIL, 0.2),
            (2, home, 600.0, PASS, SUCC, float("nan")),
            (2, away, 900.0, SHOT, SUCC, 0.25),  # away scores p2
            (2, home, 2850.0, PASS, SUCC, float("nan")),  # sets final minute ~92.5
        ]
        for period, team, t, typ, res, xg in specs:
            rows.append(
                {
                    "game_id": gid,
                    "action_id": aid,
                    "period_id": period,
                    "team_id": team,
                    "player_id": None,
                    "time_seconds": t,
                    "type_id": typ,
                    "result_id": res,
                    "xg": xg,
                    "home_team_id": home,
                    "match_date": date,
                }
            )
            aid += 1
    return pd.DataFrame(rows)


def test_reduce_path_runs_end_to_end_on_synthetic_corpus():
    # The gate that SHOULD precede any DGX run: exercise the trainer's reduce (strength -> fit ->
    # calibration_metrics -> compute_win_probability -> certify -> expected-goals) on a tiny synthetic
    # corpus WITHOUT statsbombpy. This is the path where the action_id shard-omission crashed on the DGX;
    # a local run of it catches that class of bug in ~1s instead of via a live job.
    pooled = _synthetic_corpus(6)
    params = WinProbabilityParams.default()
    pooled = _attach_strength(pooled, chronological_strength(pooled))
    games = _games_frame(pooled)

    metrics = calibration_metrics(pooled, games, params)  # <- this call KeyError'd on action_id on the DGX
    assert metrics["n_oof"] > 0 and np.isfinite(metrics["ece"])

    model = WinProbabilityModel(params=params).fit(pooled, games=games, strength_column="base_strength")
    model.certify_coherence()  # must not raise
    _samples, report = compute_win_probability(pooled, model=model, games=games, strength_column="base_strength")
    assert report.n_matches_scored == 6 and report.n_actions == len(pooled)
    n = params.regulation_minutes // params.interval_minutes

    def hz(d, mm):
        return model._hazard(score_diff=d, minutes_remaining=mm, base_strength=0.0, home=True, man_advantage=0)

    assert np.isfinite(expected_total_goals(hz, hz, n_intervals=n, K=params.lattice_pad))


def test_reduce_path_would_have_caught_action_id_omission():
    # Non-vacuity: the reduce path FAILS if the shard drops action_id (the exact DGX bug). Proves the
    # end-to-end test above is a real guard, not a shape that happens to pass.
    pooled = _synthetic_corpus(6).drop(columns=["action_id"])
    pooled = _attach_strength(pooled, chronological_strength(pooled))
    with pytest.raises(KeyError):
        calibration_metrics(pooled, _games_frame(pooled), WinProbabilityParams.default())


def test_match_shard_shape():
    actions = pd.DataFrame(
        {
            "game_id": ["g"] * 2,
            "action_id": [0, 1],
            "period_id": [1, 1],
            "team_id": [1, 2],
            "player_id": [10, 20],
            "time_seconds": [1.0, 2.0],
            "type_id": [SHOT, PASS],
            "result_id": [SUCC, SUCC],
            "xg": [0.3, float("nan")],
        }
    )
    shard = match_shard(actions, home_team_id=1, match_date="2020-05-05")
    assert list(shard.columns) == [
        "game_id",
        "action_id",
        "period_id",
        "team_id",
        "player_id",
        "time_seconds",
        "type_id",
        "result_id",
        "xg",
        "home_team_id",
        "match_date",
    ]
    assert (shard["home_team_id"] == 1).all() and (shard["match_date"] == "2020-05-05").all()


def test_shard_carries_the_columns_compute_win_probability_needs():
    # Regression (DGX run): the shard MUST carry every column the reduce feeds compute_win_probability
    # + derive_match_states, or the pooled corpus KeyErrors. action_id (a WIN_PROBABILITY_KEYS column)
    # was the one dropped -- synthetic fixtures always had it, so only the real trainer run caught it.
    from silly_kicks.win_probability import WIN_PROBABILITY_KEYS

    actions = pd.DataFrame(
        {
            "game_id": ["g"],
            "action_id": [0],
            "period_id": [1],
            "team_id": [1],
            "player_id": [10],
            "time_seconds": [1.0],
            "type_id": [SHOT],
            "result_id": [SUCC],
            "xg": [0.3],
        }
    )
    shard = match_shard(actions, home_team_id=1, match_date="2020-05-05")
    required = set(WIN_PROBABILITY_KEYS) | {"period_id", "team_id", "time_seconds", "type_id", "result_id", "player_id"}
    assert required <= set(shard.columns), (
        f"shard missing compute-required columns: {sorted(required - set(shard.columns))}"
    )
