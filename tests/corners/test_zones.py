"""TF-46 RunZoneModel mechanics: goal-radius target region, occupancy encoding, non-vacuity."""

from __future__ import annotations

import numpy as np

from silly_kicks.corners._zones import RunZoneModel, _within_goal_radius


def _synthetic():
    rng = np.random.default_rng(0)
    # initial: two clusters (unrestricted -- late-runner origins may be deep)
    init = np.vstack([rng.normal([90.0, 25.0], 1.0, size=(60, 2)), rng.normal([90.0, 43.0], 1.0, size=(60, 2))])
    # target: three clusters inside the <=25 m danger+approach region (near post, far post, and the
    # top-of-box / second-ball EDGE at ~19 m), plus one at midfield that the region filter drops.
    target = np.vstack(
        [
            rng.normal([100.0, 28.0], 1.0, size=(40, 2)),  # in-region
            rng.normal([100.0, 40.0], 1.0, size=(40, 2)),  # in-region
            rng.normal([86.0, 34.0], 1.0, size=(40, 2)),  # top-of-box / second-ball edge, in-region
            rng.normal([60.0, 10.0], 1.0, size=(40, 2)),  # midfield -> OUTSIDE 25 m, dropped
        ]
    )
    return init, target


def _toy_model() -> RunZoneModel:
    init, target = _synthetic()
    return RunZoneModel().fit(init, target, init_k=2, target_k=3)


def test_fit_target_zones_are_in_region():
    m = _toy_model()
    assert m.n_initial_zones == 2
    assert m.n_target_zones == 3  # midfield cluster dropped; 3 in-region zones fit, all active
    assert m.active_target_idx is not None and m.active_target_idx.tolist() == [0, 1, 2]


def test_active_zone_means_within_goal_radius():
    m = _toy_model()
    assert m.target_means is not None
    assert _within_goal_radius(m.target_means, m.goal_radius_m).all()  # every target zone in the danger region


def test_fit_raises_when_no_target_in_region():
    rng = np.random.default_rng(1)
    init = rng.normal([50.0, 34.0], 1.0, size=(40, 2))
    target = rng.normal([30.0, 10.0], 1.0, size=(40, 2))  # all far outside the goal-radius region
    try:
        RunZoneModel().fit(init, target, init_k=2, target_k=2)
    except ValueError as e:
        assert "within" in str(e) and "goal" in str(e)
    else:
        raise AssertionError("expected ValueError for no in-region target zone")


def test_encode_length_matches_dims():
    m = _toy_model()
    init_xy = np.array([[90.0, 25.0], [90.0, 43.0]])
    target_xy = np.array([[100.0, 28.0], [100.0, 40.0]])
    vec = m.encode_runs(init_xy, target_xy, active_mask=np.array([True, True]))
    assert vec.shape == (m.n_initial_zones * m.n_target_zones,)  # 2*3 = 6


def test_occupancy_default_excludes_deep_stayback():
    m = _toy_model()
    # p0 runs into a box zone (in-region); p1 is a deep stay-back whose target is at midfield (out-of-region)
    init_xy = np.array([[90.0, 25.0], [70.0, 34.0]])
    target_xy = np.array([[100.0, 28.0], [60.0, 34.0]])
    occ = m.encode_runs(init_xy, target_xy)  # default occupancy
    forced = m.encode_runs(init_xy, target_xy, active_mask=np.array([True, True]))
    assert occ.sum() > 0  # the box runner is counted
    assert forced.sum() > occ.sum()  # forcing the deep player active adds mass -> occupancy excluded it


def test_occupancy_counts_stationary_second_ball_holder():
    # Owner's case: a second-ball holder sits at the top-of-box edge (in-region) and barely moves.
    m = _toy_model()
    edge = np.array([[86.0, 34.0]])  # ~19 m from goal, in-region, zero displacement
    vec = m.encode_runs(edge, edge)  # default occupancy -- no displacement threshold
    assert vec.sum() > 0  # counted despite zero movement


def test_encode_is_non_vacuous():
    m = _toy_model()
    init_xy = np.array([[90.0, 25.0], [90.0, 43.0]])
    target_xy = np.array([[100.0, 28.0], [100.0, 40.0]])
    vec = m.encode_runs(init_xy, target_xy)
    assert vec.sum() > 0
    assert (vec > 1e-6).sum() > 1  # populates more than one run-pair (not degenerate)
