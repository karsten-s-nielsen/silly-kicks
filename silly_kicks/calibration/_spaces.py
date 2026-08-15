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

from ruthless import Choice, Direction, FloatRange, OptunaConfig
from ruthless.config.common import StoreConfig

from silly_kicks.xthreat import GridSpec


def stage1_config(*, n_trials: int, store_path: str, sampler: Literal["tpe", "random"] = "tpe") -> OptunaConfig:
    """Stage 1 — carrier accuracy (maximize): beta, gamma.

    tolerance_m is held at DEFAULT_CARRIER_PARAMS — under-determined by this objective (ADR-060),
    so it is not swept here.

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
            "beta": FloatRange(kind="float", lo=0.0, hi=2.0),
            "gamma": FloatRange(kind="float", lo=0.0, hi=3.0),
        },
        warm_start={"beta": 0.5, "gamma": 1.0},
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


# Aspect-sane grids near the pitch's ~1.54 ratio (105x68). Resolution is SWEPT (SK-xT-3) over this
# curated discrete set rather than two independent IntRanges (~475 cells, admits non-physical 32x6).
_GRIDS: tuple[str, ...] = ("12x8", "16x12", "20x14", "24x16", "28x18", "32x20")


def grid_from_str(s: str) -> GridSpec:
    """Parse a ``"<nx>x<ny>"`` grid string into a ``GridSpec`` (e.g. ``"16x12"`` -> 16x12).

    Examples
    --------
    >>> from silly_kicks.calibration._spaces import grid_from_str
    >>> grid_from_str("16x12").n_zones
    192
    """
    nx, ny = s.lower().split("x")
    return GridSpec(n_zones_x=int(nx), n_zones_y=int(ny))


def xt_bandwidth_config(*, n_trials: int, store_path: str, sampler: Literal["tpe", "random"] = "tpe") -> OptunaConfig:
    """SK-xT-3 — held-out xT transition-NLL sweep (minimize): bandwidth x adaptive x grid.

    Examples
    --------
    >>> from silly_kicks.calibration._spaces import xt_bandwidth_config
    >>> xt_bandwidth_config(n_trials=10, store_path="/tmp/xt.db").metric
    'xt_holdout_nll'
    """
    return OptunaConfig(
        kind="optuna",
        metric="xt_holdout_nll",
        direction=Direction.MINIMIZE,
        n_trials=n_trials,
        sampler=sampler,
        param_space={
            "bandwidth": FloatRange(kind="float", lo=0.1, hi=20.0, log=True),
            "adaptive": Choice(kind="choice", choices=(True, False)),
            "grid": Choice(kind="choice", choices=_GRIDS),
        },
        warm_start={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"},
        store=StoreConfig(kind="sqlite", path=store_path),
    )
