"""Every team_metrics METRIC column is documented in feature_glossary and homes to its family (ADR-048)."""

from __future__ import annotations

from silly_kicks.feature_glossary import FEATURE_GLOSSARY
from silly_kicks.team_metrics import TEAM_KPI_METRIC_COLUMNS

_EXPECTED_MODULE = {
    "silly_kicks.team_metrics._pressing": {
        "ppda",
        "defensive_intensity",
        "time_to_defensive_action_s",
        "time_to_recovery_s",
        "recoveries",
        "recoveries_within_ns_pct",
        "counterpress_regains",
        "counterpress_regain_pct",
    },
    "silly_kicks.team_metrics._compute": {
        "high_opportunity_shots_post_recovery",
        "box_touches_post_recovery",
    },
}


def test_all_metric_columns_glossaried():
    missing = [c for c in TEAM_KPI_METRIC_COLUMNS if c not in FEATURE_GLOSSARY]
    assert not missing, f"undocumented team_metrics metric columns: {missing}"


def test_entries_home_to_a_team_metrics_module():
    for c in TEAM_KPI_METRIC_COLUMNS:
        home = FEATURE_GLOSSARY[c].emitting_module
        assert home.startswith("silly_kicks.team_metrics."), f"{c} homes to {home}"


def test_pressing_and_companion_modules():
    for module, cols in _EXPECTED_MODULE.items():
        for c in cols:
            assert FEATURE_GLOSSARY[c].emitting_module == module, c
