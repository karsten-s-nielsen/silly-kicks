"""silly_kicks.calibration — Optuna calibration harness for tracking defaults (TF-24).

Optional subpackage; requires the ``[calibration]`` extra
(``pip install silly-kicks[calibration]``). NOT imported by ``silly_kicks/__init__`` —
import members directly from ``silly_kicks.calibration``.

See ``docs/superpowers/specs/2026-05-29-tf24-optuna-calibration-harness-design.md``.
"""

from __future__ import annotations

from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
from silly_kicks.calibration._cv import cv_scheme_for, cv_standard_error, match_cv_splits
from silly_kicks.calibration._diagnostics import exceeds_noise_floor, tf25_gate_fires
from silly_kicks.calibration._features import ALL_FEATURES, enrich_full, enrich_invariant, patch_trial_columns
from silly_kicks.calibration._gates import default_feature_variances, h1_penalty_fires, signal_sanity
from silly_kicks.calibration._selection import (
    MIN_EFFECT_SIZE,
    PointScore,
    Selection,
    build_selection_artifact,
    select_recommended_point,
)
from silly_kicks.calibration._spaces import grid_from_str, stage1_config, stage2_config, xt_bandwidth_config
from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective
from silly_kicks.calibration._xt import FrozenXt, fit_frozen_xt, load_xt, save_xt
from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective

__all__ = [
    "ALL_FEATURES",
    "MIN_EFFECT_SIZE",
    "AugmentedVaepBrierObjective",
    "CarrierAccuracyObjective",
    "FrozenXt",
    "PointScore",
    "Selection",
    "XtBandwidthObjective",
    "build_selection_artifact",
    "cv_scheme_for",
    "cv_standard_error",
    "default_feature_variances",
    "enrich_full",
    "enrich_invariant",
    "exceeds_noise_floor",
    "fit_frozen_xt",
    "grid_from_str",
    "h1_penalty_fires",
    "load_xt",
    "match_cv_splits",
    "patch_trial_columns",
    "save_xt",
    "select_recommended_point",
    "signal_sanity",
    "stage1_config",
    "stage2_config",
    "tf25_gate_fires",
    "xt_bandwidth_config",
]
