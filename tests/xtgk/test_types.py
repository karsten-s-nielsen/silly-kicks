from silly_kicks.xtgk._possession_value import DeltaV, State, zone_of


def test_zone_of_matches_flat_convention():
    # x~3 -> xi=0, y=34 -> yj=6 -> flat=(12-1-6)*16+0 = 80
    assert zone_of(3.0, 34.0) == (11 - 6) * 16 + 0


def test_deltav_identity_holds_by_construction():
    dv = DeltaV(delta=0.5, pressure_component=0.2, position_component=0.3)
    assert abs((dv.pressure_component + dv.position_component) - dv.delta) < 1e-12


def test_state_carries_zone_and_pressure():
    s = State(zone=80, pressure_level=1)
    assert s.zone == 80 and s.pressure_level == 1
