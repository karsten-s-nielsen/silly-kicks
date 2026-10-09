"""A-09: the gated circular-column registry and its anti-rot meta-test (spec §8.5 / §9.4).

The three circular-MEAN constructs are scored with rotation-invariant circular reliability, not a plain
(origin-dependent) linear ICC. This gate DERIVES the degree-valued metric columns from the emitted schemas and
asserts they partition EXACTLY into the two declared registries -- so a new ``*_deg`` column cannot ship without
being classified as a circular mean or a linear dispersion magnitude.
"""

from __future__ import annotations

from silly_kicks.coordination._columns import (
    CIRCULAR_DISPERSION_COLUMNS,
    CIRCULAR_MEAN_COLUMNS,
    COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
    COORDINATION_PAIR_METRIC_COLUMNS,
    COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
    COORDINATION_RSI_METRIC_COLUMNS,
    COORDINATION_SPECTRAL_METRIC_COLUMNS,
    COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
    reliability_kind,
)


def _all_emitted_metric_columns() -> frozenset[str]:
    return frozenset(
        COORDINATION_PAIR_METRIC_COLUMNS
        + COORDINATION_PAIR_PHASE_METRIC_COLUMNS
        + COORDINATION_SPECTRAL_METRIC_COLUMNS
        + COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS
        + COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS
        + COORDINATION_TEAM_SYNC_METRIC_COLUMNS
        + COORDINATION_RSI_METRIC_COLUMNS
    )


def test_degree_columns_partition_exactly_into_the_two_registries():
    deg_columns = {c for c in _all_emitted_metric_columns() if c.endswith("_deg")}
    registered = set(CIRCULAR_MEAN_COLUMNS) | set(CIRCULAR_DISPERSION_COLUMNS)
    # The anti-rot assertion: every degree-valued metric column is classified, and nothing is classified
    # that is not emitted. A new *_deg column breaks this until it is declared mean or dispersion.
    assert deg_columns == registered


def test_mean_and_dispersion_registries_are_disjoint_and_nonempty():
    assert set(CIRCULAR_MEAN_COLUMNS).isdisjoint(CIRCULAR_DISPERSION_COLUMNS)
    assert len(CIRCULAR_MEAN_COLUMNS) > 0
    assert len(CIRCULAR_DISPERSION_COLUMNS) > 0


def test_every_registered_column_is_actually_emitted_and_degree_valued():
    emitted = _all_emitted_metric_columns()
    for c in (*CIRCULAR_MEAN_COLUMNS, *CIRCULAR_DISPERSION_COLUMNS):
        assert c in emitted, c
        assert c.endswith("_deg"), c


def test_reliability_kind_maps_each_column():
    for c in CIRCULAR_MEAN_COLUMNS:
        assert reliability_kind(c) == "circular"
    for c in CIRCULAR_DISPERSION_COLUMNS:
        assert reliability_kind(c) == "linear"
    assert reliability_kind("coord_xc_max_abs_r") == "linear"
