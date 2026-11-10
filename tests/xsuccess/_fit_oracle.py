"""The oracle of the xsuccess fit (HFU fit-units spec 3.3, A-R1).

`OracleXSuccessModel` holds the `XSuccessModel` methods `fit`, `_fit_xgboost`, `_fit_per_type`,
`predict_success`, `_predict_per_type`, `to_dict`, `_chirality_block`, `_sha` and `save` as they were at
`bb0c601c` (4.140.0), copied VERBATIM (`silly_kicks/xsuccess/_model.py:141-352`). So no oracle path runs the
split fit steps: the kept-row selection, the `cross_val_predict` calibration, the metadata payload and the file
writes are all the old code. The new model must give the same saved bytes and the same predictions. Do not edit
the method bodies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from silly_kicks.xsuccess._features import FEATURE_NAMES, feature_contract_block, xsuccess_features
from silly_kicks.xsuccess._model import (
    _DEFAULT_PARAMS,
    _NON_ACTION,
    _SUCCESS,
    XSuccessIntegrityError,
    XSuccessModel,
    _chirality_probe_actions,
)


class OracleXSuccessModel(XSuccessModel):
    """`XSuccessModel` with its 4.140.0 fit, predict, metadata and save methods."""

    def fit(
        self,
        actions,
        *,
        family: str = "xgboost",
        calibrate: bool = True,
        params: dict | None = None,
        seed: int = 42,
    ) -> XSuccessModel:
        """Fit an action-completion model on all real on-ball actions.

        ``family="xgboost"`` (default) fits a calibrated XGBoost; ``family="per_type_logistic"`` fits
        one standardized logistic per action_type on the shared feature set (calibrated-by-
        construction — the ADR-009 fallback when XGBoost fails the per-type calibration gate; one-
        sided / thin types serve the per-type base rate). sklearn/xgboost are imported here ONLY.
        ``non_action`` rows and rows with a non-finite feature are dropped. Label = ``result_id ==
        success``.

        Examples
        --------
        .. code-block:: python

            model = XSuccessModel().fit(actions)                          # calibrated XGBoost
            fallback = XSuccessModel().fit(actions, family="per_type_logistic")
            assert model.is_fitted
        """
        sub = actions[actions["type_id"] != _NON_ACTION]
        X = xsuccess_features(sub)
        y = (np.asarray(sub["result_id"]) == _SUCCESS).astype(int)
        tid = np.asarray(sub["type_id"])
        finite = np.isfinite(X).all(axis=1)
        X, y, tid = X[finite], y[finite], tid[finite]

        self.feature_names = list(FEATURE_NAMES)
        self.feature_set = family
        self._booster = None
        self._iso_x = self._iso_y = None
        self._pt_params = None
        self._pt_base = {}
        self._pt_base_global = 0.0

        if family == "xgboost":
            self._fit_xgboost(X, y, params=params, calibrate=calibrate, seed=seed)
        elif family == "per_type_logistic":
            self._fit_per_type(X, y, tid)
        else:
            raise XSuccessIntegrityError(f"unknown family {family!r} (expected 'xgboost' or 'per_type_logistic')")
        return self

    def _fit_xgboost(self, X, y, *, params, calibrate, seed) -> None:
        import xgboost as xgb

        p = dict(params or _DEFAULT_PARAMS)
        p["random_state"] = seed
        clf = xgb.XGBClassifier(**p)
        clf.fit(X, y)
        self._booster = clf.get_booster()
        self._params = p
        if calibrate and len(np.unique(y)) == 2 and len(y) >= 50:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_predict

            n_splits = min(5, int(np.bincount(y).min()))
            if n_splits >= 2:
                oof = np.asarray(
                    cross_val_predict(
                        xgb.XGBClassifier(**p),
                        X,
                        y,
                        method="predict_proba",
                        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
                    )
                )[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
                self._iso_x = np.asarray(iso.X_thresholds_, dtype=float)
                self._iso_y = np.asarray(iso.y_thresholds_, dtype=float)

    def _fit_per_type(self, X, y, tid) -> None:
        from sklearn.linear_model import LogisticRegression

        pt: dict[int, dict] = {}
        base: dict[int, float] = {}
        for t in np.unique(tid):
            mask = tid == t
            yt = y[mask]
            base[int(t)] = float(yt.mean()) if len(yt) else 0.0
            if len(np.unique(yt)) < 2 or int(mask.sum()) < 30:
                continue  # one-sided / thin type -> serve the per-type base rate
            Xt = X[mask]
            mean = Xt.mean(axis=0)
            scale = Xt.std(axis=0)
            scale[scale == 0] = 1.0
            clf = LogisticRegression(max_iter=1000).fit((Xt - mean) / scale, yt)
            pt[int(t)] = {
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
                "mean": mean.tolist(),
                "scale": scale.tolist(),
            }
        self._pt_params = pt
        self._pt_base = base
        self._pt_base_global = float(y.mean()) if len(y) else 0.0

    def predict_success(self, actions) -> np.ndarray:
        """Pure-of-sklearn ``P(success)`` in ``[0, 1]`` per action; NaN features -> NaN (never fabricated).

        Examples
        --------
        .. code-block:: python

            p = model.predict_success(actions)   # one probability per SPADL action row
        """
        if not self.is_fitted:
            raise XSuccessIntegrityError("model is not fitted")
        X = xsuccess_features(actions)
        if self.feature_set == "per_type_logistic":
            return self._predict_per_type(X, np.asarray(actions["type_id"]))

        import xgboost as xgb

        if self._booster is None:
            raise XSuccessIntegrityError("xgboost family requires a fitted booster")
        finite = np.isfinite(X).all(axis=1)
        out = np.full(X.shape[0], np.nan, dtype=float)
        if finite.any():
            prob = np.asarray(self._booster.predict(xgb.DMatrix(X[finite])), dtype=float)
            if self._iso_x is not None and self._iso_y is not None:
                prob = np.interp(prob, self._iso_x, self._iso_y)
            out[finite] = prob
        return out

    def _predict_per_type(self, X, tid) -> np.ndarray:
        """Per-type logistic serve (pure-numpy): dispatch each row to its action_type's logistic,
        or the per-type base rate for a one-sided / unseen type. NaN features -> NaN."""
        out = np.full(X.shape[0], np.nan, dtype=float)
        finite = np.isfinite(X).all(axis=1)
        pt = self._pt_params or {}
        for i in np.nonzero(finite)[0]:
            t = int(tid[i])
            pr = pt.get(t)
            if pr is None:
                out[i] = self._pt_base.get(t, self._pt_base_global)
            else:
                z = (X[i] - np.asarray(pr["mean"])) / np.asarray(pr["scale"])
                out[i] = 1.0 / (1.0 + np.exp(-float(z @ np.asarray(pr["coef"]) + pr["intercept"])))
        return out

    # --- serialization (booster JSON + metadata.json + SHA256SUMS; fail-closed load) ---
    def to_dict(self) -> dict:
        """Metadata payload (everything but the booster, which is its own ``model.json`` file).

        Examples
        --------
        .. code-block:: python

            payload = XSuccessModel().fit(actions).to_dict()
            assert payload["feature_set"] == "xgboost"
        """
        if not self.is_fitted:
            raise XSuccessIntegrityError("model is not fitted; nothing to serialize")
        return {
            "feature_names": list(self.feature_names),
            "feature_set": self.feature_set,
            "params": self._params,
            "isotonic": (
                {"x": self._iso_x.tolist(), "y": self._iso_y.tolist()}
                if self._iso_x is not None and self._iso_y is not None
                else None
            ),
            "feature_contract": feature_contract_block(),
            "chirality": self._chirality_block(),
        }

    def _chirality_block(self) -> dict:
        return {"probe_prediction": self.predict_success(_chirality_probe_actions()).tolist()}

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256((path / "model.json").read_bytes()).hexdigest()

    def save(self, path) -> None:
        """Write the pickle-free artifact (``model.json`` booster + ``metadata.json`` + ``SHA256SUMS``).

        Examples
        --------
        .. code-block:: python

            XSuccessModel().fit(actions).save("weights/")
        """
        if not self.is_fitted:
            raise XSuccessIntegrityError("model is not fitted; nothing to serialize")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.feature_set == "per_type_logistic":
            # model.json = the per-type logistic params (pure JSON; keys stringified for JSON).
            (path / "model.json").write_text(
                json.dumps(
                    {
                        "per_type": {str(k): v for k, v in (self._pt_params or {}).items()},
                        "base": {str(k): float(v) for k, v in self._pt_base.items()},
                        "base_global": float(self._pt_base_global),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            if self._booster is None:
                raise XSuccessIntegrityError("xgboost family requires a fitted booster")
            self._booster.save_model(str(path / "model.json"))
        (path / "metadata.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        (path / "SHA256SUMS").write_text(f"{self._sha(path)}  model.json\n", encoding="utf-8")
