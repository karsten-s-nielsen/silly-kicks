"""Liveness gate for restdefense (TF-60, ADR-032 idiom): non-null + non-constant float metrics."""

from silly_kicks.restdefense import RD_LAYER1_COLUMNS
from silly_kicks.restdefense._compute import compute_rest_defense
from tests.restdefense._fixtures import make_rest_defense_fixture

# Count columns (Int64) + the categorical stagger are exempt from the non-constant check (float only).
_COUNT_COLS = {"rd_num_superiority", "rd_num_superiority_gk", "rd_zone_occupancy"}
_STAGGER = {"rd_shape_2_3_vs_3_2"}
_FLOAT_METRIC_COLS = [c for c in RD_LAYER1_COLUMNS if c not in _COUNT_COLS and c not in _STAGGER]


def _resolved():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)
    return samples[samples["rd_geometry_source"] == "resolved"]


def test_every_layer1_column_non_null_on_resolved_rows():
    resolved = _resolved()
    assert len(resolved) >= 2
    for c in RD_LAYER1_COLUMNS:
        assert resolved[c].notna().all(), f"{c} has a NaN on a resolved row"


def test_float_metrics_are_non_constant():
    resolved = _resolved()
    for c in _FLOAT_METRIC_COLS:
        vals = resolved[c].dropna()
        if len(vals) >= 2:
            assert vals.nunique() > 1, f"float metric {c} is constant across resolved samples"
