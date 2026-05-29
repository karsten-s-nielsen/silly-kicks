from ruthless import Direction, OptunaConfig

from silly_kicks.calibration._spaces import stage1_config, stage2_config


def test_stage1_config_is_maximize_with_three_params():
    cfg = stage1_config(n_trials=10, store_path="s1.db")
    assert isinstance(cfg, OptunaConfig)
    assert cfg.metric == "carrier_accuracy"
    assert cfg.direction is Direction.MAXIMIZE
    assert set(cfg.param_space) == {"tolerance_m", "beta", "gamma"}
    assert set(cfg.warm_start) == {"tolerance_m", "beta", "gamma"}  # current defaults
    assert cfg.store.path == "s1.db"


def test_stage2_config_is_minimize_with_three_params():
    cfg = stage2_config(n_trials=10, store_path="s2.db")
    assert cfg.metric == "brier"
    assert cfg.direction is Direction.MINIMIZE
    assert set(cfg.param_space) == {"k3", "pre_seconds", "min_displacement_m"}
    assert cfg.param_space["k3"].log is True  # log-uniform


def test_warm_start_subset_of_param_space_enforced_by_ruthless():
    # OptunaConfig validates warm_start subset of param_space; our builders must satisfy it.
    stage1_config(n_trials=1, store_path="x.db")  # must not raise
    stage2_config(n_trials=1, store_path="y.db")  # must not raise
