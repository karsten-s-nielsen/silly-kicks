"""XSuccessModel (TF-61) — bundled, calibrated, END-BLIND action-completion model.

``P(success | pre-action context)`` over all on-ball action types. ``fit`` trains xgboost + an
optional isotonic calibrator (both imported function-locally — training only); the serve path uses
xgboost (``[xgboost]`` extra, the ``XShotOccurrenceModel`` precedent) and pure-numpy isotonic
interpolation, so no sklearn at inference. Serialization (added in the serialization task) is
pickle-free (booster JSON + metadata + SHA256SUMS) with a chirality probe + feature contract, loaded
fail-closed (ADR-011/016/040/050).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

import silly_kicks.spadl.config as cfg

from ._features import FEATURE_NAMES, feature_contract_block, xsuccess_features

_NON_ACTION = cfg.actiontype_id["non_action"]
_SUCCESS = cfg.result_id["success"]

_DEFAULT_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "min_child_weight": 1,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
}


def _chirality_probe_actions():
    import pandas as pd

    return pd.DataFrame(
        dict(
            type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]],
            bodypart_id=[cfg.bodypart_id["foot"], cfg.bodypart_id["head"]],
            start_x=[25.0, 90.0],
            start_y=[40.0, 30.0],
            end_x=[25.0, 90.0],
            end_y=[40.0, 30.0],
            result_id=[cfg.result_id["success"], cfg.result_id["fail"]],
            time_seconds=[100.0, 2700.0],
            period_id=[1, 2],
        )
    )


def _load_booster_base_score_safe(model_json: Path):
    """Load an xgboost Booster, normalizing the bracketed ``base_score`` xgboost 3.x writes (2.x
    cannot parse). COPIED (not imported) from ``tracking/_xshot_occurrence.py``: xsuccess is
    event-only and must not import ``silly_kicks.tracking`` (allowlist gate); the small helper is
    duplicated rather than promoting a shared module out of the frozen tracking model."""
    import xgboost as xgb

    obj = json.loads(model_json.read_text(encoding="utf-8"))
    try:
        lmp = obj["learner"]["learner_model_param"]
        bs = lmp.get("base_score")
        if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
            lmp["base_score"] = str(float(bs.strip("[]")))
    except (KeyError, TypeError, ValueError):
        pass
    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(obj), "utf-8"))
    return booster


# ---- the fit steps (HFU fit-units spec 3.2) ----------------------------------------------------------
# `XSuccessModel.fit`, `predict_success` and `to_dict` compose these functions in sequence. The trainer
# (`scripts/train_xsuccess.py`) composes the SAME functions, one xgboost fit per stage unit, so the library and
# the trainer run one composition and give the same bytes.


def _training_arrays(actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, type_id)`` of the kept rows: ``non_action`` rows and rows with a non-finite feature are dropped."""
    sub = actions[actions["type_id"] != _NON_ACTION]
    X = xsuccess_features(sub)
    y = (np.asarray(sub["result_id"]) == _SUCCESS).astype(int)
    tid = np.asarray(sub["type_id"])
    finite = np.isfinite(X).all(axis=1)
    return X[finite], y[finite], tid[finite]


def _xgb_params(params: dict | None, seed: int) -> dict:
    """The xgboost parameters of one fit: ``params`` (default `_DEFAULT_PARAMS`) with ``random_state = seed``."""
    p = dict(params or _DEFAULT_PARAMS)
    p["random_state"] = seed
    return p


def _fit_booster(X, y, p: dict):
    """One xgboost fit; returns the Booster."""
    import xgboost as xgb

    clf = xgb.XGBClassifier(**p)
    clf.fit(X, y)
    return clf.get_booster()


def _calibration_splits(X, y, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``(train, test)`` index pairs of the isotonic calibration, in order; ``[]`` when no calibration
    applies (one class, fewer than 50 rows, or fewer than 2 rows of the rarer class)."""
    if not (len(np.unique(y)) == 2 and len(y) >= 50):
        return []
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return []
    from sklearn.model_selection import StratifiedKFold

    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))


def _calibration_fold_proba(X, y, p: dict, train, test) -> np.ndarray:
    """One calibration split: fit on its train rows, return ``P(success)`` of its test rows (xgboost's dtype)."""
    import xgboost as xgb

    clf = xgb.XGBClassifier(**p)
    clf.fit(X[train], y[train])
    return clf.predict_proba(X[test])[:, 1]


def _calibration_oof(n: int, splits, probas) -> np.ndarray:
    """The out-of-fold vector of ``n`` rows: each split's ``probas`` entry placed at its test rows. The dtype is
    the splits' own (what ``cross_val_predict`` gave by concatenation)."""
    oof = np.empty(n, dtype=np.result_type(*[np.asarray(q).dtype for q in probas]))
    for (_train, test), proba in zip(splits, probas, strict=True):
        oof[test] = proba
    return oof


def _isotonic_thresholds(oof, y) -> tuple[np.ndarray, np.ndarray]:
    """The isotonic breakpoints ``(x, y)`` fitted on the out-of-fold vector."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    return np.asarray(iso.X_thresholds_, dtype=float), np.asarray(iso.y_thresholds_, dtype=float)


def _raw_predict(booster, X) -> np.ndarray:
    """The booster's ``P(success)`` BEFORE the isotonic step: float64, NaN for a row with a non-finite feature."""
    finite = np.isfinite(X).all(axis=1)
    out = np.full(X.shape[0], np.nan, dtype=float)
    if finite.any():
        import xgboost as xgb

        out[finite] = np.asarray(booster.predict(xgb.DMatrix(X[finite])), dtype=float)
    return out


def _calibrate(raw: np.ndarray, iso_x: np.ndarray | None, iso_y: np.ndarray | None) -> np.ndarray:
    """The isotonic step on a raw prediction vector (pure numpy); NaN stays NaN. Without thresholds: ``raw``."""
    if iso_x is None or iso_y is None:
        return raw
    out = np.full(raw.shape[0], np.nan, dtype=float)
    finite = ~np.isnan(raw)
    out[finite] = np.interp(raw[finite], iso_x, iso_y)
    return out


def _metadata_payload(*, feature_names, feature_set: str, params, iso_x, iso_y, probe_prediction: list) -> dict:
    """The ``metadata.json`` payload from its parts (everything but the booster), in the fixed key order."""
    return {
        "feature_names": list(feature_names),
        "feature_set": feature_set,
        "params": params,
        "isotonic": ({"x": iso_x.tolist(), "y": iso_y.tolist()} if iso_x is not None and iso_y is not None else None),
        "feature_contract": feature_contract_block(),
        "chirality": {"probe_prediction": probe_prediction},
    }


def _write_metadata_and_sums(path: Path, payload: dict) -> None:
    """Write ``metadata.json`` (the payload) and ``SHA256SUMS`` (the sha256 of the ``model.json`` already in
    ``path``). `XSuccessModel.save` and the trainer's reduce both write the two files with this step."""
    (path / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sha = hashlib.sha256((path / "model.json").read_bytes()).hexdigest()
    (path / "SHA256SUMS").write_text(f"{sha}  model.json\n", encoding="utf-8")


class XSuccessIntegrityError(RuntimeError):
    """Raised on an unfitted serve or a load-time SHA / chirality / feature-contract mismatch.

    Examples
    --------
    An unfitted model refuses to predict:

    .. code-block:: python

        import pandas as pd
        from silly_kicks.xsuccess import XSuccessModel, XSuccessIntegrityError

        try:
            XSuccessModel().predict_success(pd.DataFrame())
        except XSuccessIntegrityError:
            print("refused")
    """


class XSuccessModel:
    """Event-only ``P(success | context)`` over all on-ball action types (END-BLIND features).

    Examples
    --------
    Fit on SPADL actions and score, or load the packaged default weights instead of fitting:

    .. code-block:: python

        from silly_kicks.xsuccess import XSuccessModel

        model = XSuccessModel().fit(actions)      # actions: SPADL on-ball rows
        p = model.predict_success(actions)        # P(success) per action, in [0, 1]
        # or, once weights are bundled:
        model = XSuccessModel.bundled()
    """

    def __init__(self) -> None:
        self._booster = None  # xgboost.Booster after fit/load (family="xgboost")
        self._iso_x: np.ndarray | None = None  # isotonic breakpoints (pure-numpy serve)
        self._iso_y: np.ndarray | None = None
        self._pt_params: dict[int, dict] | None = None  # per-type logistic coeffs (family="per_type_logistic")
        self._pt_base: dict[int, float] = {}  # per-type base rate (degenerate / one-sided types)
        self._pt_base_global: float = 0.0  # base rate for a type unseen in training
        self._params: dict = dict(_DEFAULT_PARAMS)
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.feature_set: str = "xgboost"

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` (or :meth:`load`) has populated a model (either family).

        Examples
        --------
        >>> from silly_kicks.xsuccess import XSuccessModel
        >>> XSuccessModel().is_fitted
        False
        """
        return self._booster is not None or self._pt_params is not None

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
        X, y, tid = _training_arrays(actions)

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
        """The model fit, then one fit per calibration split and the isotonic fit on their out-of-fold vector
        (the fit steps above, in sequence: 1 + up to 5 xgboost fits)."""
        p = _xgb_params(params, seed)
        self._booster = _fit_booster(X, y, p)
        self._params = p
        splits = _calibration_splits(X, y, seed) if calibrate else []
        if splits:
            probas = [_calibration_fold_proba(X, y, p, train, test) for train, test in splits]
            self._iso_x, self._iso_y = _isotonic_thresholds(_calibration_oof(len(y), splits, probas), y)

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

        if self._booster is None:
            raise XSuccessIntegrityError("xgboost family requires a fitted booster")
        return _calibrate(_raw_predict(self._booster, X), self._iso_x, self._iso_y)

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
        return _metadata_payload(
            feature_names=self.feature_names,
            feature_set=self.feature_set,
            params=self._params,
            iso_x=self._iso_x,
            iso_y=self._iso_y,
            probe_prediction=self._chirality_block()["probe_prediction"],
        )

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
        _write_metadata_and_sums(path, self.to_dict())

    @classmethod
    def load(cls, path, *, legacy_override: bool = False) -> XSuccessModel:
        """Load + verify (SHA -> feature-contract -> chirality); fail-closed on tamper/drift.

        SHA covers ``model.json`` (the booster); a ``metadata.json`` tamper is caught by the
        contract/chirality checks. A MISSING contract/fingerprint warns (pre-contract artifact);
        a name/constant drift or chirality mismatch RAISES ``XSuccessIntegrityError``.

        Examples
        --------
        .. code-block:: python

            model = XSuccessModel.load("weights/")
        """
        path = Path(path)
        want = (path / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
        if want != cls._sha(path):
            raise XSuccessIntegrityError(f"integrity check failed (SHA mismatch) at {path}")
        d = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        m = cls()
        m.feature_names = list(d["feature_names"])
        m.feature_set = d.get("feature_set", "xgboost")
        m._params = d.get("params", dict(_DEFAULT_PARAMS))
        if m.feature_set == "per_type_logistic":
            mj = json.loads((path / "model.json").read_text(encoding="utf-8"))
            m._pt_params = {int(k): v for k, v in mj.get("per_type", {}).items()}
            m._pt_base = {int(k): float(v) for k, v in mj.get("base", {}).items()}
            m._pt_base_global = float(mj.get("base_global", 0.0))
        else:
            iso = d.get("isotonic")
            if iso:
                m._iso_x = np.asarray(iso["x"], dtype=float)
                m._iso_y = np.asarray(iso["y"], dtype=float)
            m._booster = _load_booster_base_score_safe(path / "model.json")
        m._verify_feature_contract(d)
        m._verify_chirality(d, legacy_override=legacy_override)
        return m

    def _verify_feature_contract(self, d: dict) -> None:
        fc = d.get("feature_contract")
        if fc is None:
            warnings.warn("XSuccessModel: artifact carries no feature contract; skipping contract check.", stacklevel=2)
            return
        if list(fc.get("feature_names", [])) != list(FEATURE_NAMES):
            raise XSuccessIntegrityError(
                "feature-contract mismatch: stored feature_names differ from the current FEATURE_NAMES"
            )
        want = feature_contract_block()["geometry"]
        got = fc.get("geometry", {})
        for k, v in want.items():
            if k not in got or not np.isclose(float(got[k]), float(v), atol=1e-6, rtol=0):
                raise XSuccessIntegrityError(f"feature-contract mismatch: declared geometry constant {k!r} differs")

    def _verify_chirality(self, d: dict, *, legacy_override: bool) -> None:
        ch = d.get("chirality")
        if not ch or "probe_prediction" not in ch:
            warnings.warn(
                "XSuccessModel: artifact carries no chirality fingerprint; cannot verify served output.", stacklevel=2
            )
            return
        recomputed = self.predict_success(_chirality_probe_actions())
        stored = np.asarray(ch["probe_prediction"], dtype=float)
        if not np.allclose(recomputed, stored, atol=1e-6, rtol=0, equal_nan=True):
            if legacy_override:
                warnings.warn("XSuccessModel: chirality mismatch overridden (legacy_override=True).", stacklevel=2)
                return
            raise XSuccessIntegrityError("chirality mismatch: served predictions differ from the stored fingerprint")

    @classmethod
    def bundled(cls) -> XSuccessModel:
        """Load the packaged public-corpus weights. Raises ``FileNotFoundError`` until weights ship.

        Examples
        --------
        .. code-block:: python

            model = XSuccessModel.bundled()   # FileNotFoundError until weights ship (Commit 2)
        """
        import importlib.resources as ir

        weights = ir.files("silly_kicks.xsuccess") / "weights"
        return cls.load(Path(str(weights)))
