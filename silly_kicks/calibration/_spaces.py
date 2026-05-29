"""OptunaConfig builders for the two calibration stages (spec §3/§4).

Search spaces + warm-starts (current library defaults) from the verified signatures:
infer_ball_carrier(tolerance_m=3.0, beta=0.5, gamma=1.0); LinkParams(k3=1.0);
add_off_ball_runs(pre_seconds=1.5, min_displacement_m=3.0).

Examples
--------
>>> from silly_kicks.calibration._spaces import stage1_config, stage2_config
>>> cfg = stage2_config(n_trials=60, store_path="tc3_stage2.db")
>>> cfg.metric
'brier'
"""

from __future__ import annotations

from typing import Literal

from ruthless import Direction, FloatRange, OptunaConfig
from ruthless.config.common import StoreConfig


def stage1_config(*, n_trials: int, store_path: str, sampler: Literal["tpe", "random"] = "tpe") -> OptunaConfig:
    """Stage 1 — carrier accuracy (maximize): tolerance_m, beta, gamma.

    Examples
    --------
    >>> from silly_kicks.calibration._spaces import stage1_config
    >>> stage1_config(n_trials=10, store_path="/tmp/s1.db").metric
    'carrier_accuracy'
    """
    return OptunaConfig(
        kind="optuna",
        metric="carrier_accuracy",
        direction=Direction.MAXIMIZE,
        n_trials=n_trials,
        sampler=sampler,
        param_space={
            "tolerance_m": FloatRange(kind="float", lo=1.0, hi=8.0),
            "beta": FloatRange(kind="float", lo=0.0, hi=2.0),
            "gamma": FloatRange(kind="float", lo=0.0, hi=3.0),
        },
        warm_start={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
        store=StoreConfig(kind="sqlite", path=store_path),
    )


def stage2_config(*, n_trials: int, store_path: str, sampler: Literal["tpe", "random"] = "tpe") -> OptunaConfig:
    """Stage 2 — augmented-VAEP held-out Brier (minimize): k3, pre_seconds, min_displacement_m.

    Examples
    --------
    >>> from silly_kicks.calibration._spaces import stage2_config
    >>> stage2_config(n_trials=10, store_path="/tmp/s2.db").param_space["k3"].log
    True
    """
    return OptunaConfig(
        kind="optuna",
        metric="brier",
        direction=Direction.MINIMIZE,
        n_trials=n_trials,
        sampler=sampler,
        param_space={
            "k3": FloatRange(kind="float", lo=0.1, hi=5.0, log=True),
            "pre_seconds": FloatRange(kind="float", lo=0.5, hi=5.0),
            "min_displacement_m": FloatRange(kind="float", lo=1.0, hi=8.0),
        },
        warm_start={"k3": 1.0, "pre_seconds": 1.5, "min_displacement_m": 3.0},
        store=StoreConfig(kind="sqlite", path=store_path),
    )
