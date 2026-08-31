"""Layer-2 column constants (TF-60 PR2, ADR-081)."""

from silly_kicks.restdefense import RD_LAYER1_COLUMNS, RD_LAYER2_COLUMNS, RD_METRIC_COLUMNS


def test_layer2_columns():
    assert RD_LAYER2_COLUMNS == [
        "rd_attacker_space_control",
        "rd_danger_behind_line",
        "rd_danger_behind_line_gk",
        "rd_gk_coverage_behind_line",
        "rd_gk_reachable_coverage_m2",
    ]


def test_metric_columns_is_layer1_then_layer2_disjoint():
    assert RD_METRIC_COLUMNS == [*RD_LAYER1_COLUMNS, *RD_LAYER2_COLUMNS]
    assert set(RD_LAYER1_COLUMNS).isdisjoint(RD_LAYER2_COLUMNS)
