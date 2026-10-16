"""TF-46 RunZoneModel mechanics: GMM fit, active-zone selection, encoding non-vacuity."""

from __future__ import annotations

import numpy as np

from silly_kicks.corners._zones import RunZoneModel
from silly_kicks.tracking._geometry import in_penalty_area_absolute

_FIELD_LENGTH = 105.0


def _synthetic():
    rng = np.random.default_rng(0)
    # initial: two clusters
    init = np.vstack([rng.normal([90.0, 25.0], 1.0, size=(60, 2)), rng.normal([90.0, 43.0], 1.0, size=(60, 2))])
    # target: two IN the penalty area (x in [88.5,105], y in [13.84,54.16]), one OUTSIDE
    target = np.vstack(
        [
            rng.normal([100.0, 28.0], 1.0, size=(40, 2)),  # inside PA
            rng.normal([100.0, 40.0], 1.0, size=(40, 2)),  # inside PA
            rng.normal([80.0, 10.0], 1.0, size=(40, 2)),  # outside PA
        ]
    )
    return init, target


def _toy_model() -> RunZoneModel:
    init, target = _synthetic()
    return RunZoneModel().fit(init, target, init_k=2, target_k=3)


def test_fit_selects_two_active_zones():
    m = _toy_model()
    assert m.n_initial_zones == 2
    assert m.n_target_zones == 2  # two of the three target components are inside the penalty area


def test_active_zones_are_inside_penalty_area():
    m = _toy_model()
    assert m.active_target_idx is not None and m.target_means is not None
    for j in m.active_target_idx:
        mean = m.target_means[j]
        assert in_penalty_area_absolute(float(mean[0]), float(mean[1]), attacked_goal_x=_FIELD_LENGTH)


def test_fit_raises_when_no_active_zone():
    rng = np.random.default_rng(1)
    init = rng.normal([50.0, 34.0], 1.0, size=(40, 2))
    target = rng.normal([30.0, 10.0], 1.0, size=(40, 2))  # all far outside the penalty area
    try:
        RunZoneModel().fit(init, target, init_k=2, target_k=2)
    except ValueError as e:
        assert "penalty area" in str(e)
    else:
        raise AssertionError("expected ValueError for no active target zone")


def test_encode_length_matches_dims():
    m = _toy_model()
    init_xy = np.array([[90.0, 25.0], [90.0, 43.0]])
    target_xy = np.array([[100.0, 28.0], [100.0, 40.0]])
    vec = m.encode_runs(init_xy, target_xy, active_mask=np.array([True, True]))
    assert vec.shape == (m.n_initial_zones * m.n_target_zones,)  # 2*2 = 4


def test_encode_is_non_vacuous():
    m = _toy_model()
    # two players running to the two different active zones from the two initial clusters
    init_xy = np.array([[90.0, 25.0], [90.0, 43.0]])
    target_xy = np.array([[100.0, 28.0], [100.0, 40.0]])
    vec = m.encode_runs(init_xy, target_xy, active_mask=np.array([True, True]))
    assert vec.sum() > 0  # mass present
    assert (vec > 1e-6).sum() > 1  # populates more than one run-pair (not degenerate)
