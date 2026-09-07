"""Liveness gate for restdefense (TF-60, ADR-032 idiom): non-null + non-constant float metrics."""

from silly_kicks.restdefense import RD_METRIC_COLUMNS
from silly_kicks.restdefense._compute import compute_rest_defense
from tests.restdefense._fixtures import make_fitted_xt, make_rest_defense_fixture

# Count columns (Int64) + the categorical stagger are exempt from the non-constant check (float only).
# All five Layer-2 columns are floats, so they fall into _FLOAT_METRIC_COLS automatically.
_COUNT_COLS = {"rd_num_superiority", "rd_num_superiority_gk", "rd_zone_occupancy"}
_STAGGER = {"rd_shape_2_3_vs_3_2"}
_FLOAT_METRIC_COLS = [c for c in RD_METRIC_COLUMNS if c not in _COUNT_COLS and c not in _STAGGER]


def _resolved():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, xt=make_fitted_xt())  # xt -> Layer 2 live
    return samples[samples["rd_geometry_source"] == "resolved"]


def test_every_metric_column_non_null_on_resolved_rows():
    resolved = _resolved()
    assert len(resolved) >= 2
    for c in RD_METRIC_COLUMNS:
        assert resolved[c].notna().all(), f"{c} has a NaN on a resolved row"


def test_float_metrics_are_non_constant():
    resolved = _resolved()
    for c in _FLOAT_METRIC_COLS:
        vals = resolved[c].dropna()
        if len(vals) >= 2:
            assert vals.nunique() > 1, f"float metric {c} is constant across resolved samples"


def test_layer3_arm_columns_non_null_and_non_constant():
    """Spec 10 liveness for the 4 Layer-3 deterrent arm columns (ADR-089): every emitted arm column is
    non-NaN and non-constant on the `computed` rows of the multi-domain fixture. Checked on the
    `computed`-source rows (a `ghost_missing` row is an honest NaN, not a liveness failure), mirroring
    the resolved-rows convention above."""
    from silly_kicks.restdefense._arms import rest_defense_gk_deterrent, rest_defense_outfield_deterrent
    from silly_kicks.restdefense._columns import (
        RD_GK_ARM_COLUMNS,
        RD_GK_SOURCE,
        RD_OUTFIELD_ARM_COLUMNS,
        RD_OUTFIELD_SOURCE,
    )
    from tests.tracking.test_ghost_gk import _fitted_model
    from tests.tracking.test_ghost_outfield_model import _fit_toy

    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    of_arm, _ = rest_defense_outfield_deterrent(
        actions, frames, xt=xt, ghost_outfield_model=_fit_toy()[0], home_team_id=1
    )
    gk_arm, _ = rest_defense_gk_deterrent(actions, frames, xt=xt, ghost_gk_model=_fitted_model()[0], home_team_id=1)
    for arm, cols, source in (
        (of_arm, RD_OUTFIELD_ARM_COLUMNS, RD_OUTFIELD_SOURCE),
        (gk_arm, RD_GK_ARM_COLUMNS, RD_GK_SOURCE),
    ):
        computed = arm[arm[source] == "computed"]
        assert len(computed) >= 2, f"fixture scored <2 computed rows for {source}"
        for c in cols:
            assert computed[c].notna().all(), f"arm column {c} has a NaN on a computed row"
            vals = computed[c].dropna()
            assert vals.nunique() > 1, f"arm column {c} is constant across computed samples"
