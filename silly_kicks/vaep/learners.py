"""Gradient boosting learner implementations for the VAEP framework.

Each learner function trains a binary classifier and returns the fitted model.
Default hyperparameters are exposed as module-level constants so callers can
inspect or override them.
"""

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

try:
    import xgboost
except ImportError:
    xgboost = None  # type: ignore
try:
    import catboost
except ImportError:
    catboost = None  # type: ignore
try:
    import lightgbm
except ImportError:
    lightgbm = None  # type: ignore


_XGBOOST_DEFAULTS: dict[str, object] = {
    "n_estimators": 100,
    "max_depth": 3,
    "eval_metric": "auc",
    "early_stopping_rounds": 10,
    "enable_categorical": True,
}

_CATBOOST_DEFAULTS: dict[str, object] = {
    "eval_metric": "BrierScore",
    "loss_function": "Logloss",
    "iterations": 100,
}

_LIGHTGBM_DEFAULTS: dict[str, object] = {
    "n_estimators": 100,
    "max_depth": 3,
}


def _fit_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    eval_set: Optional[list[tuple[pd.DataFrame, pd.Series]]] = None,
    tree_params: Optional[dict[str, Any]] = None,
    fit_params: Optional[dict[str, Any]] = None,
) -> "xgboost.XGBClassifier":
    """Train an XGBoost classifier.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training labels.
    eval_set : list of (DataFrame, Series), optional
        Validation data for early stopping.
    tree_params : dict, optional
        Constructor parameters. Uses :data:`_XGBOOST_DEFAULTS` when *None*.
    fit_params : dict, optional
        Parameters passed to ``model.fit()``.

    Returns
    -------
    xgboost.XGBClassifier
        The fitted model.
    """
    if xgboost is None:
        raise ImportError("xgboost is not installed.")
    if tree_params is None:
        tree_params = dict(_XGBOOST_DEFAULTS)
    if fit_params is None:
        fit_params = {"verbose": True}
    if eval_set is not None:
        fit_params = {**fit_params, "eval_set": eval_set}
    model = xgboost.XGBClassifier(**tree_params)
    return model.fit(X, y, **fit_params)


def _fit_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    eval_set: Optional[list[tuple[pd.DataFrame, pd.Series]]] = None,
    tree_params: Optional[dict[str, Any]] = None,
    fit_params: Optional[dict[str, Any]] = None,
) -> "catboost.CatBoostClassifier":
    """Train a CatBoost classifier.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training labels.
    eval_set : list of (DataFrame, Series), optional
        Validation data for early stopping.
    tree_params : dict, optional
        Constructor parameters. Uses :data:`_CATBOOST_DEFAULTS` when *None*.
    fit_params : dict, optional
        Parameters passed to ``model.fit()``.

    Returns
    -------
    catboost.CatBoostClassifier
        The fitted model.
    """
    if catboost is None:
        raise ImportError("catboost is not installed.")
    if tree_params is None:
        tree_params = dict(_CATBOOST_DEFAULTS)
    if fit_params is None:
        is_cat_feature = [c.dtype.name == "category" for (_, c) in X.items()]
        fit_params = {
            "cat_features": np.nonzero(is_cat_feature)[0].tolist(),
            "verbose": True,
        }
    if eval_set is not None:
        fit_params = {**fit_params, "early_stopping_rounds": 10, "eval_set": eval_set}
    model = catboost.CatBoostClassifier(**tree_params)
    return model.fit(X, y, **fit_params)


def _fit_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    eval_set: Optional[list[tuple[pd.DataFrame, pd.Series]]] = None,
    tree_params: Optional[dict[str, Any]] = None,
    fit_params: Optional[dict[str, Any]] = None,
) -> "lightgbm.LGBMClassifier":
    """Train a LightGBM classifier.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training labels.
    eval_set : list of (DataFrame, Series), optional
        Validation data for early stopping.
    tree_params : dict, optional
        Constructor parameters. Uses :data:`_LIGHTGBM_DEFAULTS` when *None*.
    fit_params : dict, optional
        Parameters passed to ``model.fit()``.

    Returns
    -------
    lightgbm.LGBMClassifier
        The fitted model.
    """
    if lightgbm is None:
        raise ImportError("lightgbm is not installed.")
    if tree_params is None:
        tree_params = dict(_LIGHTGBM_DEFAULTS)
    if fit_params is None:
        fit_params = {"eval_metric": "auc", "verbose": True}
    if eval_set is not None:
        fit_params = {**fit_params, "early_stopping_rounds": 10, "eval_set": eval_set}
    model = lightgbm.LGBMClassifier(**tree_params)
    return model.fit(X, y, **fit_params)


_LEARNER_REGISTRY: dict[str, Callable[..., Any]] = {
    "xgboost": _fit_xgboost,
    "catboost": _fit_catboost,
    "lightgbm": _fit_lightgbm,
}
