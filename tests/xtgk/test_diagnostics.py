from typing import cast

from silly_kicks.xtgk._diagnostics import DEEP_ZONE_CELLS, GateConfig, run_deep_zone_gate
from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue
from silly_kicks.xtgk._pressure_levels import PressureLevels
from tests.xtgk.conftest import flat_no_shot_cohort, three_band_cohort


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
