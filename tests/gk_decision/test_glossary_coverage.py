"""Every gk_decision METRIC column (not the keys/provenance) is documented in feature_glossary (ADR-048)."""

from __future__ import annotations

from silly_kicks.feature_glossary import FEATURE_GLOSSARY

_METRIC_COLUMNS = ("decision_value", "sel_efficiency", "decision_pct", "chosen_ev", "best_ev")


def test_gk_decision_metric_columns_glossaried():
    missing = [c for c in _METRIC_COLUMNS if c not in FEATURE_GLOSSARY]
    assert not missing, f"undocumented gk_decision metric columns: {missing}"


def test_gk_decision_entries_home_to_compute_module():
    for c in _METRIC_COLUMNS:
        assert FEATURE_GLOSSARY[c].emitting_module == "silly_kicks.gk_decision._compute"
