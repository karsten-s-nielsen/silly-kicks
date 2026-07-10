from typing import cast

import numpy as np

from silly_kicks.xtgk._diagnostics import DEEP_ZONE_CELLS, GateConfig, run_deep_zone_gate
from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._pressure_levels import PressureLevels
from tests.xtgk.conftest import flat_no_shot_cohort, three_band_cohort


class _StubValue:
    """Minimal PossessionValue stub: constant per-tercile deep value + full support."""

    def __init__(self, per_level: dict[int, float], support: int = 100):
        self._v = per_level
        self._support = support

    def value(self, zone, p):
        return self._v[p]

    def support(self, p):
        return np.full((12, 16), self._support, dtype=int)


def _stub(per_level):
    # the gate only calls .value / .support; cast the stub to the concrete types the signature wants
    s = _StubValue(per_level)
    return cast(MarkovPossessionValue, s), cast(EmpiricalPossessionValue, s)


def test_relative_effect_floor_fails_a_trivial_gradient():
    # absolute effect passes (0.02 >= 0.005) but relative is tiny: 0.02 / mean(1.00,0.98)=~0.02 < 0.25
    mk, emp = _stub({1: 1.00, 2: 0.99, 3: 0.98})
    cfg = GateConfig(
        effect_floor=0.005, relative_effect_floor=0.25, n_min=10, min_occupied_cells=2, expected_direction="decreasing"
    )
    report = run_deep_zone_gate(mk, emp, cfg)
    assert report.relative_effect < 0.25
    assert report.passed is False
    assert "relative" in report.stop_reason


def test_relative_effect_floor_passes_a_real_gradient():
    mk, emp = _stub({1: 0.030, 2: 0.020, 3: 0.010})  # rel = 0.02/0.02 = 1.0 >= 0.25
    cfg = GateConfig(
        effect_floor=0.005, relative_effect_floor=0.25, n_min=10, min_occupied_cells=2, expected_direction="decreasing"
    )
    report = run_deep_zone_gate(mk, emp, cfg)
    assert report.relative_effect >= 0.25
    assert report.passed is True


def _fit_pair(a):
    pl = PressureLevels().fit(a["pressure"])  # ONE tercile fit shared by both
    mk = MarkovPossessionValue().fit(a, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    emp = EmpiricalPossessionValue().fit(a, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    return mk, emp


def test_deep_zone_cells_are_first_two_columns():
    assert len(DEEP_ZONE_CELLS) == 24 and all((c % 16) in (0, 1) for c in DEEP_ZONE_CELLS)


def test_gate_passes_on_honest_cohort():
    mk, emp = _fit_pair(three_band_cohort())
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=3, min_occupied_cells=2))
    assert rep.passed is True
    assert rep.n_occupied_cells >= 2
    assert rep.effect_size > 0.005
    assert rep.observed_direction == "decreasing"


def test_gate_stops_on_too_few_occupied_cells():
    mk, emp = _fit_pair(three_band_cohort())
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=10_000_000, min_occupied_cells=2))
    assert rep.passed is False and "support" in rep.stop_reason.lower()


def test_gate_fails_on_flat_negative_control():
    mk, emp = _fit_pair(flat_no_shot_cohort())
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=3, min_occupied_cells=2))
    assert rep.passed is False


def test_gate_fails_when_crosscheck_disagrees_on_buildup():
    # real mk passes effect + monotonicity; a stub empirical surface with an OPPOSITE build-up
    # gradient must flip crosscheck_agrees to False and fail the gate (pins G1's mechanism).
    mk, _ = _fit_pair(three_band_cohort())

    class _DisagreeingEmp:
        def value(self, zone, p):
            return {1: 0.0, 2: 0.05, 3: 0.2}[p]  # rises with pressure -> opposite sign to mk

    emp = cast(EmpiricalPossessionValue, _DisagreeingEmp())  # gate only calls emp.value(zone, p)
    rep = run_deep_zone_gate(mk, emp, GateConfig(effect_floor=0.005, n_min=3, min_occupied_cells=2))
    assert rep.crosscheck_agrees is False and rep.passed is False


def test_expected_direction_configurable_and_reported():
    mk, emp = _fit_pair(three_band_cohort())
    rep = run_deep_zone_gate(
        mk,
        emp,
        GateConfig(effect_floor=0.005, n_min=3, min_occupied_cells=2, expected_direction="increasing"),
    )
    assert rep.observed_direction == "decreasing" and rep.passed is False


# --- Q3 / G8 pre-gate input-QC reports (owner-run, ADR-036 §6) ---


def test_ood_rate_by_source():
    import pandas as pd

    from silly_kicks.xtgk._diagnostics import ood_rate_by_source

    df = pd.DataFrame(
        {
            "data_source": ["gradientsports", "gradientsports", "skillcorner", "skillcorner"],
            "ood_flag": [False, False, True, True],
        }
    )
    rep = ood_rate_by_source(df)
    assert rep["gradientsports"] == 0.0 and rep["skillcorner"] == 1.0


def test_ladder_falls_to_zone_conditional_when_global_starves_deep_high_tercile():
    from silly_kicks.xtgk._diagnostics import _fit_pair as _diag_fit_pair
    from silly_kicks.xtgk._diagnostics import _occupied, run_gate_with_ladder
    from tests.xtgk.conftest import deep_low_rest_high_cohort

    actions = deep_low_rest_high_cohort()
    cfg = GateConfig(
        effect_floor=0.0, relative_effect_floor=0.0, n_min=5, min_occupied_cells=2, expected_direction="either"
    )
    # rung 1 (global) alone starves the deep high tercile -> < 2 occupied deep cells (would STOP)
    _pl_g, mk_g, _emp_g = _diag_fit_pair(actions, xg_column="xg", pressure_column="pressure", mode="global")
    assert len(_occupied(mk_g, cfg)) < cfg.min_occupied_cells
    # the ladder therefore falls through to zone-conditional and RUNS there
    result = run_gate_with_ladder(actions, xg_column="xg", pressure_column="pressure", cfg=cfg)
    assert result["rung"] == "zone_conditional"


def test_both_orientations_equivariant_under_mirror_y_and_flags_mirror_x():
    from silly_kicks.xtgk._diagnostics import run_gate_both_orientations

    actions = three_band_cohort()
    cfg = GateConfig(
        effect_floor=0.0, relative_effect_floor=0.0, n_min=5, min_occupied_cells=2, expected_direction="either"
    )
    out = run_gate_both_orientations(actions, xg_column="xg", pressure_column="pressure", cfg=cfg)
    # mirror_y (attack direction preserved) must reproduce the fit effect within tolerance
    assert abs(out["fit"]["report"].effect_size - out["mirror_y"]["report"].effect_size) < 1e-6
    # mirror_x reverses attack direction -> the fit validator rejects it (not attack-LTR)
    assert out["mirror_x"]["orientation_rejected"] is True


def test_frame_present_null_pressure_count():
    import numpy as np
    import pandas as pd

    from silly_kicks.xtgk._diagnostics import frame_present_null_pressure_count

    df = pd.DataFrame(
        {
            "data_source": ["gradientsports"] * 4,
            "pressure": [0.5, np.nan, np.nan, 0.2],
            "frame_present": [True, True, False, True],  # row1 = unpressured restart; row2 = gap
        }
    )
    rep = frame_present_null_pressure_count(df, pressure_col="pressure", frame_present_col="frame_present")
    assert rep["gradientsports"] == 1  # only the frame-present + null row counts
