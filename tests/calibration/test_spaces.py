from ruthless import Direction, OptunaConfig

from silly_kicks.calibration._spaces import stage1_config, stage2_config


def test_stage1_config_is_maximize_with_two_params():
    # ADR-060: tolerance_m is held at DEFAULT_CARRIER_PARAMS, not swept -> beta/gamma only.
    cfg = stage1_config(n_trials=10, store_path="s1.db")
    assert isinstance(cfg, OptunaConfig)
    assert cfg.metric == "carrier_accuracy"
    assert cfg.direction is Direction.MAXIMIZE
    assert set(cfg.param_space) == {"beta", "gamma"}
    assert set(cfg.warm_start) == {"beta", "gamma"}  # current defaults
    assert cfg.store.path == "s1.db"  # type: ignore[union-attr]


def test_stage1_config_does_not_sweep_tolerance_m():
    # ADR-060: tolerance_m is held at DEFAULT_CARRIER_PARAMS, not swept (under-determined by the
    # carrier-actor objective, which has no loose-ball negatives). Only beta/gamma are searched.
    cfg = stage1_config(n_trials=1, store_path=":memory:")
    assert set(cfg.param_space) == {"beta", "gamma"}
    assert "tolerance_m" not in cfg.warm_start


def test_stage2_config_is_minimize_with_three_params():
    cfg = stage2_config(n_trials=10, store_path="s2.db")
    assert cfg.metric == "brier"
    assert cfg.direction is Direction.MINIMIZE
    assert set(cfg.param_space) == {"k3", "pre_seconds", "min_displacement_m"}
    assert cfg.param_space["k3"].log is True  # type: ignore[attr-defined]  # log-uniform


def test_warm_start_subset_of_param_space_enforced_by_ruthless():
    # OptunaConfig validates warm_start subset of param_space; our builders must satisfy it.
    stage1_config(n_trials=1, store_path="x.db")  # must not raise
    stage2_config(n_trials=1, store_path="y.db")  # must not raise


def test_xt_bandwidth_config_minimizes_nll_over_three_axes():
    from silly_kicks.calibration._spaces import xt_bandwidth_config

    cfg = xt_bandwidth_config(n_trials=10, store_path="xt.db")
    assert cfg.metric == "xt_holdout_nll"
    assert cfg.direction is Direction.MINIMIZE
    assert set(cfg.param_space) == {"bandwidth", "adaptive", "grid"}
    assert cfg.param_space["bandwidth"].log is True  # type: ignore[attr-defined]
    assert set(cfg.warm_start) == {"bandwidth", "adaptive", "grid"}
    assert cfg.warm_start["grid"] == "16x12"


def test_every_grid_member_parses_to_valid_gridspec():
    from silly_kicks.calibration._spaces import _GRIDS, grid_from_str
    from silly_kicks.xthreat import GridSpec

    for s in _GRIDS:
        g = grid_from_str(s)
        assert isinstance(g, GridSpec)
        assert g.n_zones_x >= 1 and g.n_zones_y >= 1
    assert "16x12" in _GRIDS


def test_xt_bandwidth_public_exports():
    from silly_kicks.calibration import XtBandwidthObjective, grid_from_str, xt_bandwidth_config

    assert callable(xt_bandwidth_config)
    assert callable(grid_from_str)
    assert XtBandwidthObjective.__name__ == "XtBandwidthObjective"
