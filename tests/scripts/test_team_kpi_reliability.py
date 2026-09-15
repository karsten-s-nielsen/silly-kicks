"""Pure reliability-reduce kernels for the TF-52 team-KPI reliability driver (owner-run corpus, CI kernels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts._sb_open_data import assert_statsbomb_open_data_mode
from scripts.validate_team_kpi_reliability import (
    _load_wyscout_pappalardo_matches,
    _shape_pappalardo_event,
    aggregate_possession_ground_truth,
    compare_providers,
    icc1,
    reduce_possession_ground_truth,
    reduce_reliability,
    split_half_reliability,
    type_ii_slope,
)


def _synthetic(seed: int = 0) -> pd.DataFrame:
    """6 teams x 8 matches: a 'reliable' KPI carries team structure; 'noise' has none."""
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(6)]
    team_level = {t: float(rng.normal(0.0, 2.0)) for t in teams}
    rows = []
    for t in teams:
        for m in range(8):
            rows.append(
                {
                    "team_id": t,
                    "game_id": f"{t}_g{m}",
                    "reliable": team_level[t] + float(rng.normal(0.0, 0.25)),
                    "noise": float(rng.normal(0.0, 2.0)),
                }
            )
    return pd.DataFrame(rows)


def test_icc1_separates_structured_from_noise():
    df = _synthetic()
    icc_reliable = icc1(df["reliable"].to_numpy(), df["team_id"].to_numpy())
    icc_noise = icc1(df["noise"].to_numpy(), df["team_id"].to_numpy())
    assert icc_reliable > 0.7
    assert icc_noise < 0.3


def test_split_half_reliability_recovers_structure():
    df = _synthetic()
    assert split_half_reliability(df, "reliable")["r"] > 0.7
    assert split_half_reliability(df, "reliable")["n_teams"] == 6
    # noise has no per-team structure -> low correlation
    assert split_half_reliability(df, "noise")["r"] < 0.5


def test_type_ii_slope_known_value():
    x = np.linspace(0, 10, 50)
    y = 2.0 * x
    assert abs(type_ii_slope(x, y) - 2.0) < 1e-9
    # perfectly anti-correlated -> negative slope
    assert type_ii_slope(x, -3.0 * x) < 0


def test_reduce_reliability_structure_and_values():
    df = _synthetic()
    out = reduce_reliability(df, ["reliable", "noise", "not_a_column"])
    assert out["n_teams"] == 6 and out["n_matches"] == 48
    assert "not_a_column" not in out["per_kpi"]  # absent columns skipped
    assert out["per_kpi"]["reliable"]["icc"] > 0.7
    assert out["per_kpi"]["reliable"]["split_half_r"] > 0.7
    assert out["per_kpi"]["reliable"]["n_observed"] == 48


def test_reduce_reliability_empty():
    out = reduce_reliability(pd.DataFrame(columns=["team_id", "game_id"]), ["reliable"])
    assert out == {"n_teams": 0, "n_matches": 0, "per_kpi": {}}


def test_aggregate_possession_ground_truth():
    per_match = [
        {"recall": 0.9, "precision": 0.4, "f1": 0.55},
        {"recall": 0.8, "precision": 0.5, "f1": 0.62},
    ]
    agg = aggregate_possession_ground_truth(per_match)
    assert agg["n_matches"] == 2
    assert abs(agg["recall_mean"] - 0.85) < 1e-9
    assert np.isnan(aggregate_possession_ground_truth([])["f1_mean"])


def test_reduce_possession_ground_truth_dedups_per_match():
    # 2 matches, 2 team rows each; poss_* repeated on both rows of a match -> deduped per game_id.
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "team_id": [1, 2, 3, 4],
            "poss_recall": [0.9, 0.9, 0.8, 0.8],
            "poss_precision": [0.4, 0.4, 0.5, 0.5],
            "poss_f1": [0.55, 0.55, 0.62, 0.62],
        }
    )
    agg = reduce_possession_ground_truth(df)
    assert agg["n_matches"] == 2
    assert abs(agg["recall_mean"] - 0.85) < 1e-9
    # no possession columns (e.g. Wyscout, no native id) -> empty
    assert reduce_possession_ground_truth(pd.DataFrame({"game_id": [1], "team_id": [1]}))["n_matches"] == 0


def test_compare_providers_poolable_and_not():
    reports = {
        "statsbomb": {"reliability": {"per_kpi": {"ppda": {"icc": 0.60, "split_half_r": 0.7}, "noise": {"icc": 0.01}}}},
        "wyscout": {"reliability": {"per_kpi": {"ppda": {"icc": 0.55, "split_half_r": 0.6}, "noise": {"icc": -0.30}}}},
    }
    out = compare_providers(reports)
    assert out["n_providers"] == 2
    assert out["per_kpi"]["ppda"]["poolable"] is True  # 0.60 vs 0.55: same sign, spread 0.05 < 0.20
    assert out["per_kpi"]["noise"]["poolable"] is False  # 0.01 vs -0.30: opposite sign
    assert compare_providers({"statsbomb": reports["statsbomb"]})["per_kpi"] == {}  # single provider


def test_shape_pappalardo_event_matches_wyscout_contract():
    from silly_kicks.spadl.wyscout import EXPECTED_INPUT_COLUMNS

    e = {
        "matchId": 2576335,
        "id": 111,
        "matchPeriod": "2H",
        "eventSec": 12.5,
        "teamId": 1609,
        "playerId": 25413,
        "eventId": 8,
        "subEventId": 85,
        "positions": [{"x": 49, "y": 49}, {"x": 31, "y": 78}],
        "tags": [{"id": 1801}],
    }
    row = _shape_pappalardo_event(e)
    assert set(row) == EXPECTED_INPUT_COLUMNS  # exactly the wyscout input contract
    assert row["game_id"] == 2576335 and row["event_id"] == 111
    assert row["period_id"] == 2 and row["milliseconds"] == 12500
    assert row["team_id"] == 1609 and row["player_id"] == 25413
    assert row["type_id"] == 8 and row["subtype_id"] == 85
    assert row["positions"] == [{"x": 49, "y": 49}, {"x": 31, "y": 78}]
    assert row["tags"] == [{"id": 1801}]


def test_assert_statsbomb_open_data_mode_fails_closed_on_credentials(monkeypatch):
    # no credentials -> open-data (public) mode -> no raise
    monkeypatch.delenv("SB_USERNAME", raising=False)
    monkeypatch.delenv("SB_PASSWORD", raising=False)
    assert_statsbomb_open_data_mode()
    # credentials set -> statsbombpy would pull the PRIVATE API -> fail closed
    monkeypatch.setenv("SB_USERNAME", "someone")
    with pytest.raises(SystemExit, match="public-only"):
        assert_statsbomb_open_data_mode()


def test_wyscout_reader_rejects_non_public_competition(tmp_path):
    # a --wyscout-dir file naming a competition outside the public Pappalardo set fails closed.
    (tmp_path / "events_PrivateLeague.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="public-only"):
        list(_load_wyscout_pappalardo_matches(str(tmp_path)))


def test_wyscout_reader_accepts_public_competition(tmp_path):
    # a public Pappalardo competition passes the guard (no matching matches file -> skipped, no raise).
    (tmp_path / "events_England.json").write_text("[]", encoding="utf-8")
    assert list(_load_wyscout_pappalardo_matches(str(tmp_path))) == []
