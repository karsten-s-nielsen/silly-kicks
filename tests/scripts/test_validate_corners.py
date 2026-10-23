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
            rng.normal([86.0, 34.0], 1.0, size=(40, 2)),  # top-of-box / second-ball edge (in-region)
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


def test_dominant_routine_handles_all_nan_rows():
    # A short/unclear corner is an all-NaN coefficient row -> -1 bucket, never an "All-NaN slice" crash.
    from scripts.validate_corners import dominant_routine

    coeffs = np.array([[0.1, 0.9, 0.2], [np.nan, np.nan, np.nan], [0.5, 0.0, 0.0]])
    assert dominant_routine(coeffs).tolist() == [1, -1, 0]


def test_f6_float_gi_join_helpers_are_gone():
    # F6 (spec 14.3): the exact-float / whole-second-pad GI join is REMOVED; GI times ride the restart rows.
    import scripts.validate_corners as vc

    assert not hasattr(vc, "gi_first_possession") and not hasattr(vc, "gi_possession_times")


def test_f6_resolve_corners_for_match_carries_first_possession_on_the_restart_row():
    # F6: resolve_kicks runs first, so first_possession_time / _status come WITH the corner row (restart
    # identity). A native corner -> status native_anchor, time NaN; a native match is never "gi_unpaired".
    from scripts._driver_sources import LoadedMatch
    from scripts.validate_corners import resolve_corners_for_match
    from silly_kicks.spadl import config as spadlconfig

    actions = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "original_event_id": ["e0", "e1"],
            "action_id": [0, 1],
            "period_id": [1, 1],
            "time_seconds": [100.0, 102.0],
            "team_id": ["tA", "tA"],
            "player_id": ["pa", "pb"],
            "start_x": [104.5, 95.0],
            "start_y": [0.5, 30.0],
            "end_x": [95.0, 100.0],
            "end_y": [30.0, 34.0],
            "type_id": [spadlconfig.actiontype_id["corner_crossed"], spadlconfig.actiontype_id["shot"]],
            "result_id": [spadlconfig.result_id["success"], spadlconfig.result_id["fail"]],
            "bodypart_id": [0, 0],
        }
    )
    rows = []
    for i in range(40):
        t = 98.0 + 0.1 * i
        for pid, team, gk, x, y in (
            ("pa", "tA", False, 104.0, 1.0),
            ("pb", "tA", False, 95.0, 30.0),
            ("ka", "tA", True, 2.0, 34.0),
            ("kb", "tB", True, 103.0, 34.0),
        ):
            rows.append(
                {
                    "game_id": "g1",
                    "frame_id": 1000 + i,
                    "period_id": 1,
                    "time_seconds": t,
                    "team_id": team,
                    "player_id": pid,
                    "x": x,
                    "y": y,
                    "is_ball": False,
                    "is_goalkeeper": gk,
                }
            )
        rows.append(
            {
                "game_id": "g1",
                "frame_id": 1000 + i,
                "period_id": 1,
                "time_seconds": t,
                "team_id": None,
                "player_id": None,
                "x": 104.5,
                "y": 0.5,
                "is_ball": True,
                "is_goalkeeper": False,
            }
        )
    frames = pd.DataFrame(rows)
    item = LoadedMatch("sportec", "g1", actions, frames, "tA", None, None)
    windowed, goal_map, _contacts, gi_unpaired = resolve_corners_for_match(item)
    assert {"first_possession_time", "first_possession_status"} <= set(windowed.columns)
    assert len(windowed) == 1 and windowed.iloc[0]["first_possession_status"] == "native_anchor"
    assert np.isnan(windowed.iloc[0]["first_possession_time"]) and gi_unpaired is False
    assert goal_map.attacked_goal("g1", 1, "tA") == 105.0


def test_derive_season_july_split():
    from scripts.validate_corners import derive_season

    assert derive_season("2025-02-01") == "2024/2025"  # Feb -> prior July's season
    assert derive_season("2024-08-10") == "2024/2025"  # Aug -> that July's season
    assert derive_season("2025-07-01") == "2025/2026"  # July boundary flips forward
    assert derive_season("2025-06-30") == "2024/2025"  # June stays in the prior season
    assert derive_season(None) is None
    assert derive_season("not-a-date") is None


def test_opponent_adjusted_removes_defence_baseline():
    # Leave-one-attacker-out defending-team residual: three attackers all face "D" at the same level ->
    # each equals D's other-attacker baseline -> residual ~ 0 (the opponent-strength confound is removed).
    from scripts.validate_corners import opponent_adjusted

    rows = [{"_team_key": atk, "_def_key": "D", "v": 0.5} for atk in ("A", "C", "E") for _ in range(6)]
    rows += [{"_team_key": "A", "_def_key": "W", "v": 0.9} for _ in range(6)]  # W faced ONLY by A
    s = opponent_adjusted(pd.DataFrame(rows), "v", "_team_key", "_def_key")
    assert s.loc[s["_def_key"] == "D", "_resid"].abs().max() < 1e-9  # equal-level vs D -> residual 0
    assert s.loc[s["_def_key"] == "W", "_resid"].isna().all()  # no leave-A-out baseline for W -> NaN


def test_team_season_transfer_nets_out_league_baseline():
    # TF46-IMPL-04: the test must bite on the LEAVE-TEAM-OUT league netting, not just sign persistence
    # (chance metrics are non-negative, so a raw-mean sign trivially persists across seasons). T_cross is
    # ABOVE its league in 2023/2024 but BELOW it in 2024/2025, so its NETTED residual flips sign while its
    # raw mean (0.7 then 0.4, both positive) does not. With the netting sign_agreement = 2/3; drop the
    # baseline (resid = raw mean) and it becomes 3/3 -> this assertion goes RED. T_hi/T_lo stay one side.
    from scripts.validate_corners import team_season_transfer

    cells = {
        ("T_cross", "2023/2024"): 0.7,
        ("T_cross", "2024/2025"): 0.4,
        ("T_hi", "2023/2024"): 0.8,
        ("T_hi", "2024/2025"): 0.9,
        ("T_lo", "2023/2024"): 0.2,
        ("T_lo", "2024/2025"): 0.5,
    }
    rows = [{"team": t, "season": s, "v": v} for (t, s), v in cells.items() for _ in range(5)]
    res = team_season_transfer(pd.DataFrame(rows), "v", "team", "season")
    assert res["n_transfer_teams"] == 3
    assert res["sign_agreement"] == pytest.approx(2 / 3)  # only T_cross flips sign once netted of the league


def test_team_season_transfer_degenerate_below_three_pairs():
    from scripts.validate_corners import team_season_transfer

    rows = [
        {"team": t, "season": s, "v": lvl}
        for s in ("2023/2024", "2024/2025")
        for t, lvl in (("T1", 0.8), ("T2", 0.2))
        for _ in range(6)
    ]
    res = team_season_transfer(pd.DataFrame(rows), "v", "team", "season")
    assert res["n_transfer_teams"] == 2  # <3 pairs -> honest NaN, never a fabricated correlation
    assert np.isnan(res["sign_agreement"]) and np.isnan(res["resid_corr"])


def test_convergent_timing_correlated_with_offset():
    from scripts.validate_corners import convergent_timing

    rng = np.random.default_rng(0)
    t = rng.uniform(100.0, 200.0, size=30)
    df = pd.DataFrame({"first_contact_time": t, "first_possession_time": t + 0.3})
    res = convergent_timing(df, "first_contact_time", "first_possession_time")
    assert res["n"] == 30
    assert res["rho"] > 0.95  # the two onset measures agree
    assert res["offset_mean_s"] == pytest.approx(0.3)


def test_convergent_timing_kick_relative_per_provider_and_event_coded_native_leg():
    # F5 (spec section 14.2): per-provider Spearman on kick-relative onsets; a NATIVE provider's contact leg
    # uses only winner_source == "event_coded" rows.
    from scripts.validate_corners import convergent_timing_kick_relative

    rng = np.random.default_rng(1)
    tk = rng.uniform(100.0, 200.0, size=20)
    c_lag = rng.uniform(0.2, 0.8, size=20)  # varying contact onset (kick-relative) so rho is defined
    sc = pd.DataFrame(
        {
            "provider": ["skillcorner"] * 20,
            "t_kick": tk,
            "first_contact_time": tk + c_lag,
            "first_possession_time": tk + c_lag + 0.4,  # possession tracks contact -> high rho
            "winner_source": ["tracking_inferred"] * 20,
        }
    )
    # native: 6 event_coded rows (used, varying onsets) + 6 tracking_inferred rows with a BROKEN contact leg.
    tkn = rng.uniform(100.0, 200.0, size=12)
    nlag = rng.uniform(0.2, 0.8, size=6)
    native = pd.DataFrame(
        {
            "provider": ["sportec"] * 12,
            "t_kick": tkn,
            "first_contact_time": np.concatenate([tkn[:6] + nlag, tkn[6:] * 0.0 + 999.0]),
            "first_possession_time": np.concatenate([tkn[:6] + nlag + 0.3, tkn[6:] + 0.8]),
            "winner_source": ["event_coded"] * 6 + ["tracking_inferred"] * 6,
        }
    )
    out = convergent_timing_kick_relative(pd.concat([sc, native], ignore_index=True))
    assert set(out) == {"skillcorner", "sportec"}
    assert out["skillcorner"]["n"] == 20
    assert out["sportec"]["n"] == 6  # only the event_coded native rows survive
    assert out["sportec"]["rho"] > 0.9  # the surviving native leg agrees once the broken rows are gone


def test_reduce_samples_net_of_team_and_transfer_when_columns_present():
    # With the confound columns present, reduce adds net_of_team (team/opponent/routine legs) + transfer.
    from scripts.validate_corners import reduce_samples

    rows = [
        {
            "team_id": atk,
            "defending_team_id": "D",
            "provider": "idsse",
            "dom_routine": i % 3,
            "season": season,
            "n_goals": ngoals,
            "n_shots": nshots,
            "sum_xg": lvl,  # distinct, persistent team levels -> non-degenerate residuals both seasons
        }
        for season in ("2023/2024", "2024/2025")
        for atk, lvl, ngoals, nshots in (("A", 0.7, 1, 3), ("B", 0.2, 0, 1), ("C", 0.45, 0, 2))
        for i in range(8)
    ]
    out = reduce_samples(pd.DataFrame(rows))
    assert {"team_one_way", "opponent_adjusted", "routine_fixed_effect"} <= set(out["net_of_team"]["sum_xg"])
    assert "transfer" in out and "sum_xg" in out["transfer"]


def test_reduce_counts_gi_unpaired_matches():
    # F6 / TF46-IMPL-06: a dark GI timing leg is attributed to a measured UNPAIRED match, counted per match.
    from scripts.validate_corners import reduce_samples

    samples = pd.DataFrame(
        {
            "team_id": ["A", "A", "B", "B"],
            "provider": "skillcorner",
            "game_id": ["g1", "g1", "g2", "g2"],
            "gi_unpaired": [True, True, False, False],
            "n_goals": [0, 1, 0, 1],
            "n_shots": [2, 3, 1, 2],
        }
    )
    out = reduce_samples(samples)
    assert out["n_matches_gi_unpaired"] == 1  # g1 unpaired, g2 paired


def test_reduce_samples_omits_net_of_team_without_confound_columns():
    # The minimal sample frame (no defending_team_id/season) must still reduce -- legs are conditional.
    from scripts.validate_corners import reduce_samples

    samples = pd.DataFrame(
        {"team_id": ["A", "A", "B", "B"], "provider": "idsse", "n_goals": [0, 1, 0, 1], "n_shots": [2, 3, 1, 2]}
    )
    out = reduce_samples(samples)
    assert "net_of_team" not in out and "transfer" not in out  # absent without the confound columns


def test_injection_firewall_no_outcome_vector():
    # the power harness takes an InjectionSpec recipe, never an observed-outcome vector (unrepresentable)
    import inspect

    from silly_kicks.causal.power import att_power_curve

    params = set(inspect.signature(att_power_curve).parameters)
    assert "injection" in params
    assert "Y" not in params and "outcomes" not in params  # observed outcome is structurally unrepresentable
