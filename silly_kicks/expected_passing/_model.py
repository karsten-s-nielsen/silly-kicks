"""PassCompletionModel -- event-only expected passing (TF-54b).

Logistic ``P(complete | origin -> target geometry)``. Trained with sklearn (imported
function-locally inside :meth:`PassCompletionModel.fit` ONLY); served pure-numpy from a
pickle-free JSON + SHA256SUMS artifact with a chirality probe and a feature contract, so
inference imports no sklearn and loading is fail-closed (ADR-011 / ADR-040 / ADR-050 idiom, mirrored
from ``silly_kicks/tracking/_gk_completion.py`` but kept event-only -- no tracking import).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

from ._features import FEATURE_NAMES, feature_contract_block, pass_completion_features

_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]

# A fixed chirality probe: a pair of asymmetric origin -> target passes at OPPOSITE ends of the
# pitch. Stored predictions are re-computed at load and compared, so a coefficient/scale corruption
# (or a mis-served left/right-handed model) fails closed rather than serving silently.
_CHIRALITY_PROBE_INPUT: dict[str, list[float]] = {
    "origin_x": [20.0, 80.0],
    "origin_y": [34.0, 10.0],
    "target_x": [60.0, 88.0],
    "target_y": [40.0, 30.0],
}
_CHIRALITY_ATOL = 1e-6  # served-prediction fingerprint tolerance
_CONTRACT_ATOL = 1e-6  # declared geometry-constant drift tolerance (distinct axis from chirality)


class PassCompletionIntegrityError(RuntimeError):
    """Raised on an unfitted serve or a load-time chirality / feature-contract mismatch.

    Mirrors the ADR-011 / ADR-040 / ADR-050 fail-closed load discipline: a tampered artifact
    (SHA mismatch), a served-prediction chirality mismatch, or a declared-constant / feature-name
    drift all raise this rather than silently serving the wrong model.

    Examples
    --------
    An unfitted model refuses to predict:

    >>> import numpy as np
    >>> from silly_kicks.expected_passing import PassCompletionModel, PassCompletionIntegrityError
    >>> try:
    ...     PassCompletionModel().predict_completion(
    ...         np.array([1.0]), np.array([1.0]), np.array([2.0]), np.array([2.0]))
    ... except PassCompletionIntegrityError:
    ...     print("refused")
    refused
    """


class PassCompletionModel:
    """Event-only logistic pass-completion model: ``P(complete | origin -> target geometry)``.

    ``fit`` trains a standardized logistic regression on every ``pass``-type row (completed = 1 at
    the real end, failed = 0 at the death ``end``); ``predict_completion`` is
    ``sigmoid(standardize(X) @ coef + intercept)`` in pure numpy, so the serve path never imports
    sklearn. A non-finite input coordinate yields a NaN probability (never a fabricated value).

    Examples
    --------
    Fit on a table of SPADL ``pass`` actions, then score a candidate origin -> target, or load the
    packaged default weights via :meth:`bundled` instead of fitting:

    .. code-block:: python

        import numpy as np
        from silly_kicks.expected_passing import PassCompletionModel

        model = PassCompletionModel().fit(actions)           # actions: SPADL pass rows
        p = model.predict_completion(                        # P(complete) in [0, 1]
            np.array([20.0]), np.array([34.0]),              # origin x, y
            np.array([60.0]), np.array([40.0]),              # target x, y
        )
        # or, once weights are bundled (Commit 2):
        model = PassCompletionModel.bundled()
    """

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.feature_names = list(FEATURE_NAMES)

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` (or :meth:`load`) has populated the coefficients.

        Examples
        --------
        >>> from silly_kicks.expected_passing import PassCompletionModel
        >>> PassCompletionModel().is_fitted
        False
        """
        return self._coef is not None

    def fit(self, actions: pd.DataFrame) -> PassCompletionModel:
        """Fit the standardized logistic coefficients on all ``pass``-type rows.

        sklearn is imported here and ONLY here -- the serve path is pure numpy. Rows with a
        non-finite origin/target coordinate are dropped before fitting.

        Parameters
        ----------
        actions : pandas.DataFrame
            SPADL actions with ``type_id`` / ``result_id`` / ``start_x`` / ``start_y`` /
            ``end_x`` / ``end_y``.

        Examples
        --------
        .. code-block:: python

            from silly_kicks.expected_passing import PassCompletionModel

            model = PassCompletionModel().fit(actions)   # actions: SPADL pass rows
            assert model.is_fitted
        """
        from sklearn.linear_model import LogisticRegression  # training-only import

        p = actions[actions["type_id"] == _PASS].dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        X = pass_completion_features(
            p["start_x"].to_numpy(),
            p["start_y"].to_numpy(),
            p["end_x"].to_numpy(),
            p["end_y"].to_numpy(),
        )
        y = (p["result_id"].to_numpy() == _SUCCESS).astype(int)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Z = (X - mean) / scale
        clf = LogisticRegression(max_iter=1000).fit(Z, y)
        self._coef = clf.coef_[0].astype(float)
        self._intercept = float(clf.intercept_[0])
        self._mean = mean
        self._scale = scale
        return self

    def predict_completion(self, origin_x, origin_y, target_x, target_y) -> np.ndarray:
        """Pure-numpy ``P(complete)`` in ``[0, 1]`` for each origin -> target pair.

        A row whose features are non-finite scores NaN (never a fabricated probability). Raises
        :class:`PassCompletionIntegrityError` if the model is not fitted/loaded.

        Examples
        --------
        .. code-block:: python

            import numpy as np

            p = model.predict_completion(
                np.array([20.0]), np.array([34.0]),
                np.array([60.0]), np.array([40.0]),
            )
        """
        coef, mean, scale, intercept = self._coef, self._mean, self._scale, self._intercept
        if coef is None or mean is None or scale is None or intercept is None:
            raise PassCompletionIntegrityError("model is not fitted")
        X = pass_completion_features(origin_x, origin_y, target_x, target_y)
        Z = (X - mean) / scale
        logit = Z @ coef + intercept
        out = 1.0 / (1.0 + np.exp(-logit))
        out[~np.isfinite(X).all(axis=1)] = np.nan  # NaN feature -> NaN (never a fabricated prob)
        return out

    # ---- serialization (pickle-free JSON envelope + SHA256, fail-closed load) ----
    def _chirality_block(self) -> dict:
        pin = _CHIRALITY_PROBE_INPUT
        pred = self.predict_completion(
            np.asarray(pin["origin_x"], float),
            np.asarray(pin["origin_y"], float),
            np.asarray(pin["target_x"], float),
            np.asarray(pin["target_y"], float),
        )
        return {"probe_input": {k: list(v) for k, v in pin.items()}, "probe_prediction": pred.tolist()}

    def to_dict(self) -> dict:
        """The pickle-free JSON artifact payload (coefficients + contract + chirality fingerprint).

        Examples
        --------
        .. code-block:: python

            payload = PassCompletionModel().fit(actions).to_dict()
            assert payload["feature_names"][0] == "distance"
        """
        coef, mean, scale, intercept = self._coef, self._mean, self._scale, self._intercept
        if coef is None or mean is None or scale is None or intercept is None:
            raise PassCompletionIntegrityError("model is not fitted; nothing to serialize")
        return {
            "feature_names": list(self.feature_names),
            "coef": coef.tolist(),
            "intercept": intercept,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "feature_contract": feature_contract_block(),
            "chirality": self._chirality_block(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PassCompletionModel:
        """Rebuild a served model from a JSON payload, without touching the filesystem.

        Examples
        --------
        .. code-block:: python

            fitted = PassCompletionModel().fit(actions)
            clone = PassCompletionModel.from_dict(fitted.to_dict())
            assert clone.is_fitted
        """
        m = cls()
        m.feature_names = list(d["feature_names"])
        m._coef = np.asarray(d["coef"], dtype=float)
        m._intercept = float(d["intercept"])
        m._mean = np.asarray(d["mean"], dtype=float)
        m._scale = np.asarray(d["scale"], dtype=float)
        return m

    @staticmethod
    def _sha(path: Path) -> str:
        text = (path / "model.json").read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save(self, path: Path | str) -> None:
        """Write the pickle-free artifact (``model.json`` + ``SHA256SUMS``) to a directory.

        Examples
        --------
        .. code-block:: python

            PassCompletionModel().fit(actions).save("weights/")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        (path / "SHA256SUMS").write_text(f"{self._sha(path)}  model.json\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str, *, legacy_override: bool = False) -> PassCompletionModel:
        """Load and verify an artifact directory, fail-closed on tamper / drift.

        In order: verify the ``SHA256SUMS`` digest, verify the served-prediction chirality
        fingerprint (``atol=1e-6``, ``equal_nan=True``), then verify the feature names and declared
        geometry constants. A tamper (SHA), a chirality mismatch, or a name/constant drift raises
        :class:`PassCompletionIntegrityError`; a MISSING fingerprint or feature contract only warns
        (a pre-contract artifact is undeclared, not known-bad). ``legacy_override=True`` downgrades
        a chirality mismatch to a warning.

        Examples
        --------
        .. code-block:: python

            model = PassCompletionModel.load("weights/")
        """
        path = Path(path)
        want = (path / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
        if want != cls._sha(path):
            raise PassCompletionIntegrityError(f"integrity check failed (SHA mismatch) at {path}")
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        m = cls.from_dict(d)
        m._verify_feature_contract(d)
        m._verify_chirality(d, legacy_override=legacy_override)
        return m

    def _verify_feature_contract(self, d: dict) -> None:
        fc = d.get("feature_contract")
        if fc is None:
            warnings.warn(
                "PassCompletionModel: artifact carries no feature contract; skipping contract check.",
                stacklevel=2,
            )
            return
        if list(fc.get("feature_names", [])) != list(FEATURE_NAMES):
            raise PassCompletionIntegrityError(
                "feature-contract mismatch: stored feature_names differ from the current FEATURE_NAMES"
            )
        want_geom = feature_contract_block()["geometry"]
        got_geom = fc.get("geometry", {})
        for key, value in want_geom.items():
            if key not in got_geom or not np.isclose(float(got_geom[key]), float(value), atol=_CONTRACT_ATOL, rtol=0):
                raise PassCompletionIntegrityError(
                    f"feature-contract mismatch: declared geometry constant {key!r} differs from the current value"
                )

    def _verify_chirality(self, d: dict, *, legacy_override: bool) -> None:
        ch = d.get("chirality")
        if not ch or "probe_input" not in ch or "probe_prediction" not in ch:
            warnings.warn(
                "PassCompletionModel: artifact carries no chirality fingerprint; cannot verify served output.",
                stacklevel=2,
            )
            return
        pin = ch["probe_input"]
        recomputed = self.predict_completion(
            np.asarray(pin["origin_x"], float),
            np.asarray(pin["origin_y"], float),
            np.asarray(pin["target_x"], float),
            np.asarray(pin["target_y"], float),
        )
        stored = np.asarray(ch["probe_prediction"], float)
        if not np.allclose(recomputed, stored, atol=_CHIRALITY_ATOL, rtol=0, equal_nan=True):
            if legacy_override:
                warnings.warn(
                    "PassCompletionModel: chirality mismatch overridden (legacy_override=True).",
                    stacklevel=2,
                )
                return
            raise PassCompletionIntegrityError(
                "chirality mismatch: served predictions differ from the stored fingerprint"
            )

    @classmethod
    def bundled(cls) -> PassCompletionModel:
        """Load the packaged default weights (public-corpus-trained).

        The convenience path so a caller can inject a completion model without fitting one. Raises
        ``FileNotFoundError`` before Commit 2 (weights are bundled then).

        Examples
        --------
        .. code-block:: python

            model = PassCompletionModel.bundled()   # FileNotFoundError until weights ship
            p = model.predict_completion(
                np.array([20.0]), np.array([34.0]), np.array([60.0]), np.array([40.0]))
        """
        import importlib.resources as ir

        weights = ir.files("silly_kicks.expected_passing") / "weights"
        return cls.load(Path(str(weights)))
