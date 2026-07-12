"""RetentionModel port + GkRetentionModel adapter (ADR-036 §Part 3).

P(retain | s,a) for GK distributions. Injected into the v2 metric (same discipline as
compute_xt_gk's completion=). Jeffrey's xR-GK later = a second adapter satisfying this port.
Logistic, sklearn at fit, pure-numpy serve, pickle-free JSON+SHA256 -- mirrors GkCompletionModel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES

_WEIGHTS_ROOT = Path(__file__).parent / "_retention_weights"
_VARIANT_CACHE: dict = {}
_VARIANT_KEY_ALIASES = {"gs": "default"}
# GS `default` + a SkillCorner variant ship (PR-S111, 4.44.0). SkillCorner goal-kick-only retention was
# near-chance (AUC ~0.54, calibration-failing); the broadened is_gk_distribution domain (goal-kicks + GK
# open-play passes, 5477 rows) makes it viable -- AUC 0.650 / ECE 0.020 / slope 0.92, GATE=PASS -- so
# SkillCorner routes to its own weights; every other provider falls back to `default` (ADR-036 §Part 3).
# Add a provider key here only when a passing per-provider variant is bundled.
_PROVIDER_VARIANT: dict[str, str] = {"skillcorner": "skillcorner"}


@runtime_checkable
class RetentionModel(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> npt.NDArray[np.float64]: ...


def variant_key_for_provider(source_provider: str | None) -> str:
    """Map a tracking ``source_provider`` to a retention-model variant key (PURE, no IO).
    ``skillcorner`` -> its own bundled variant; every other provider -> ``"gs"`` (aliased to the bundled
    ``default``). Extend ``_PROVIDER_VARIANT`` when a passing per-provider variant ships."""
    return _PROVIDER_VARIANT.get(str(source_provider).lower() if source_provider is not None else "", "gs")


class GkRetentionModel:
    """Logistic P(retain) for GK distributions. sklearn at fit; pure-numpy at serve."""

    VERSION = "1.0.0"

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.feature_names: list[str] = list(RETENTION_FEATURE_NAMES)
        self.shipped_variant: str | None = None
        self.provider_list: list | None = None

    # ---- fit ----
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> GkRetentionModel:
        from sklearn.linear_model import LogisticRegression

        X_raw = features[self.feature_names].to_numpy(float)
        y = np.asarray(labels, dtype=int)
        mean = np.nanmean(X_raw, axis=0)
        std_raw = np.nanstd(X_raw, axis=0)
        std = np.where(std_raw > 1e-9, std_raw, 1.0)
        self._mean, self._std = mean, std
        X = np.where(np.isfinite(X_raw), X_raw, mean[None, :])
        Xs = (X - mean) / std
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(Xs, y)
        self._coef = clf.coef_[0].astype(float)
        self._intercept = float(clf.intercept_[0])
        return self

    # ---- serve (pure numpy) ----
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """P(retain) per row. Pure numpy; no sklearn at serve.

        Non-finite features are imputed to the **training mean** (neutral post-standardisation).
        That is deliberate and unchanged — but it means a NaN-geometry row would silently receive a
        *no-information* rho, which then multiplies every term of the metric. ``compute_xt_gk_v2``
        therefore masks non-finite-coordinate rows **upstream** and never calls this on them, rather
        than relying on the imputation. See ADR-036 (4.46.0 amendment).
        """
        coef, mean, std = self._coef, self._mean, self._std
        if coef is None or mean is None or std is None:
            raise RuntimeError("GkRetentionModel not fitted/loaded.")
        X = features[self.feature_names].to_numpy(float)
        Xf = np.where(np.isfinite(X), X, mean[None, :])
        Xs = (Xf - mean) / std
        return 1.0 / (1.0 + np.exp(-(Xs @ coef + self._intercept)))

    # ---- serialization (pickle-free JSON envelope) ----
    def to_dict(self) -> dict:
        import sklearn

        if self._coef is None or self._mean is None or self._std is None:
            raise RuntimeError("GkRetentionModel not fitted/loaded; nothing to serialize.")
        return {
            "version": self.VERSION,
            "feature_names": self.feature_names,
            "coef": self._coef.tolist(),
            "intercept": self._intercept,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "sklearn_version": sklearn.__version__,
            "shipped_variant": self.shipped_variant,
            "provider_list": self.provider_list,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GkRetentionModel:
        m = cls()
        m.feature_names = list(d["feature_names"])
        m._coef = np.asarray(d["coef"], dtype=float)
        m._intercept = float(d["intercept"])
        m._mean = np.asarray(d["mean"], dtype=float)
        m._std = np.asarray(d["std"], dtype=float)
        m.shipped_variant = d.get("shipped_variant")
        m.provider_list = d.get("provider_list")
        return m

    @staticmethod
    def _sha(path: Path) -> str:
        text = (path / "model.json").read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        (path / "SHA256SUMS").write_text(f"{self._sha(path)}  model.json\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> GkRetentionModel:
        path = Path(path)
        want = (path / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
        if want != cls._sha(path):
            raise ValueError(f"GkRetentionModel integrity check failed at {path}")
        return cls.from_dict(json.loads((path / "model.json").read_text(encoding="utf-8")))

    @classmethod
    def from_variant(cls, variant: str = "default") -> GkRetentionModel:
        variant = _VARIANT_KEY_ALIASES.get(variant, variant)
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        wdir = _WEIGHTS_ROOT / variant
        if not (wdir / "SHA256SUMS").exists():
            raise FileNotFoundError(
                f"No bundled retention weights for {variant!r} at {wdir}. Train via scripts/train_gk_retention.py."
            )
        m = cls.load(wdir)
        _VARIANT_CACHE[variant] = m
        return m
