"""GK-distribution pass-completion model for xT-GK RAV (Eyestone). Logistic regression,
pure-numpy serve, tiny JSON coefficient artifact. Replaces the open-play accessible-space xC
(OOD on goal-kicks: ~31% coverage). Trained on observed SPADL ``result_id``. See NOTICE / ADR-024."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]
_WEIGHTS_ROOT = Path(__file__).parent / "_gk_completion_weights"
_VARIANT_CACHE: dict = {}
_GEOM_FEATURES = ("length", "forwardness", "dest_x", "dest_y_off")  # geometry-unscoreable iff any NaN

_GATE_LCB_FLOOR = 0.5  # serve the model only if a type's held-out AUC LCB strictly exceeds chance
_GATE_N_MIN = 50  # below this per-type sample a bootstrap LCB is too unstable to trust -> base_rate


def serve_mode_from_lcb(
    lcb: float | None, n: int, *, lcb_floor: float = _GATE_LCB_FLOOR, n_min: int = _GATE_N_MIN
) -> str:
    """Per-type serve-gate decision (the ONE place the rule lives; unit-tested at the boundaries).

    Returns ``"model"`` iff the type's held-out AUC lower-confidence-bound strictly exceeds ``lcb_floor``
    on a large-enough sample; else ``"base_rate"``. A ``None``/NaN ``lcb`` (undefined/degenerate AUC --
    e.g. a near-empty positive class like GK throw-ins) or ``n < n_min`` -> ``"base_rate"``. See the
    per-type-base-rate spec (2026-06-09) Decision 2: serve uses the conservative LCB while *bundling*
    uses the point estimate -- different questions ("beats chance with confidence for THIS type" vs
    "good enough to ship the variant")."""
    if lcb is None or not math.isfinite(lcb) or n < n_min:
        return "base_rate"
    return "model" if lcb > lcb_floor else "base_rate"


GK_COMPLETION_FEATURE_NAMES = [
    "length",
    "forwardness",
    "dy_abs",
    "dest_x",
    "dest_y_off",
    "dest_defender_density",
    "is_goalkick",
    "is_throw_in",
]


def extract_gk_completion_features(geom: pd.DataFrame, *, defender_density: pd.Series | None = None) -> pd.DataFrame:
    """Feature rows from resolved geometry (origin_x/y, dest_x/y, type_id). The SINGLE code path
    used at BOTH train and serve (train==serve parity)."""
    ox = geom["origin_x"].to_numpy(float)
    oy = geom["origin_y"].to_numpy(float)
    dx = geom["dest_x"].to_numpy(float) - ox
    dy = geom["dest_y"].to_numpy(float) - oy
    length = np.hypot(dx, dy)
    dens = defender_density.to_numpy(float) if defender_density is not None else np.full(len(geom), np.nan)
    # density NaN is LEFT NaN for the model to MEAN-impute (neutral after standardization). A finite
    # sentinel would be an OOD extrapolation in a linear model AND bypass the base-rate fallback.
    tid = geom["type_id"].to_numpy()
    return pd.DataFrame(
        {
            "length": length,
            "forwardness": np.divide(dx, length, out=np.zeros_like(dx), where=length > 0),
            "dy_abs": np.abs(dy),
            "dest_x": geom["dest_x"].to_numpy(float),
            "dest_y_off": np.abs(geom["dest_y"].to_numpy(float) - spadlconfig.field_width / 2),
            "dest_defender_density": dens,
            "is_goalkick": (tid == _GOALKICK).astype(float),
            "is_throw_in": (tid == _THROW_IN).astype(float),
        },
        index=geom.index,
    )


class GkCompletionModel:
    """Logistic P(success) for GK distributions. sklearn at fit; pure-numpy at serve."""

    VERSION = "1.1.0"

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.feature_names: list[str] = list(GK_COMPLETION_FEATURE_NAMES)
        self._base_rates: dict[str, float] = {}
        # Per-type serve gate (computed at train time from held-out CV; empty -> fail-open all-"model").
        self._type_serve_mode: dict[str, str] = {}  # {goalkick|throw_in|other: "model"|"base_rate"}
        self._type_gate_metrics: dict[str, dict] = {}  # {type: {auc, lcb, n}} -- transparency, not read at serve
        self.shipped_variant: str | None = None
        self.provider_list: list | None = None

    # ---- fit ----
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> GkCompletionModel:
        from sklearn.linear_model import LogisticRegression

        X_raw = features[self.feature_names].to_numpy(float)  # geometry finite (P2); density may be NaN
        y = np.asarray(labels, dtype=int)
        # m-a: standardization stats over PRESENT values only (nanmean/nanstd) so mean-imputed
        # density rows do NOT compress _std as the NaN fraction grows.
        mean = np.nanmean(X_raw, axis=0)
        std_raw = np.nanstd(X_raw, axis=0)
        std = np.where(std_raw > 1e-9, std_raw, 1.0)
        self._mean = mean
        self._std = std
        # P3: density NaN -> training mean (neutral after standardization); stored _mean makes serve
        # impute identically. (Geometry is finite -- prepare dropped geometry-unscoreable rows, P2.)
        X = np.where(np.isfinite(X_raw), X_raw, mean[None, :])
        if not np.isfinite(X).all():
            raise ValueError(
                "fit received non-finite GEOMETRY features; prepare must drop geometry-unscoreable rows (review P2)."
            )
        Xs = (X - mean) / std
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(Xs, y)
        self._coef = clf.coef_[0].astype(float)
        self._intercept = float(clf.intercept_[0])
        gk = features["is_goalkick"].to_numpy() == 1.0
        ti = features["is_throw_in"].to_numpy() == 1.0
        self._base_rates = {
            "goalkick": float(y[gk].mean()) if gk.any() else float(y.mean()),
            "throw_in": float(y[ti].mean()) if ti.any() else float(y.mean()),
            "other": float(y[~(gk | ti)].mean()) if (~(gk | ti)).any() else float(y.mean()),
            "global": float(y.mean()),
        }
        return self

    # ---- serve (pure numpy) ----
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        coef, mean, std = self._coef, self._mean, self._std
        if coef is None or mean is None or std is None:
            raise RuntimeError("GkCompletionModel not fitted/loaded.")
        X = features[self.feature_names].to_numpy(float)
        geom_idx = [self.feature_names.index(c) for c in _GEOM_FEATURES]
        geom_bad = ~np.isfinite(X[:, geom_idx]).all(axis=1)
        # per-FEATURE missing (density NaN) -> training mean (neutral); the row is still scored on
        # its geometry (P3 -- NOT a sentinel, NOT whole-row base-rate).
        Xf = np.where(np.isfinite(X), X, mean[None, :])
        Xs = (Xf - mean) / std
        p = 1.0 / (1.0 + np.exp(-(Xs @ coef + self._intercept)))
        # whole-row geometry-unscoreable -> per-type base rate (R5/M4).
        if geom_bad.any():
            gk = features["is_goalkick"].to_numpy()
            ti = features["is_throw_in"].to_numpy()
            for i in np.flatnonzero(geom_bad):
                p[i] = self._base_rate_for_type(float(gk[i]), float(ti[i]))
        return p

    def _base_rate_for_type(self, is_gk: float, is_ti: float) -> float:
        if is_gk == 1.0:
            return self._base_rates.get("goalkick", self._base_rates.get("global", 0.5))
        if is_ti == 1.0:
            return self._base_rates.get("throw_in", self._base_rates.get("global", 0.5))
        return self._base_rates.get("other", self._base_rates.get("global", 0.5))

    @staticmethod
    def _type_key(type_id: int) -> str:
        if type_id == _GOALKICK:
            return "goalkick"
        if type_id == _THROW_IN:
            return "throw_in"
        return "other"

    def serve_mode_for_types(self, type_ids: np.ndarray) -> np.ndarray:
        """Per-row ``"model"``/``"base_rate"`` from the stored per-type gate; absent type -> ``"model"``
        (fail-open). Pure; the gate is computed at train time (held-out CV)."""
        return np.array([self._type_serve_mode.get(self._type_key(int(t)), "model") for t in type_ids], dtype=object)

    def base_rate_for_types(self, type_ids: np.ndarray) -> np.ndarray:
        """Vectorized per-type calibrated base rate (reuses ``_base_rate_for_type``)."""
        return np.array(
            [self._base_rate_for_type(float(t == _GOALKICK), float(t == _THROW_IN)) for t in type_ids],
            dtype=float,
        )

    # ---- serialization (pickle-free JSON envelope) ----
    def to_dict(self) -> dict:
        import sklearn

        if self._coef is None or self._mean is None or self._std is None:
            raise RuntimeError("GkCompletionModel not fitted/loaded; nothing to serialize.")
        return {
            "version": self.VERSION,
            "feature_names": self.feature_names,
            "coef": self._coef.tolist(),
            "intercept": self._intercept,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "base_rates": self._base_rates,
            "type_serve_mode": self._type_serve_mode,
            "type_gate_metrics": self._type_gate_metrics,
            "sklearn_version": sklearn.__version__,
            "shipped_variant": self.shipped_variant,
            "provider_list": self.provider_list,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GkCompletionModel:
        m = cls()
        m.feature_names = list(d["feature_names"])
        m._coef = np.asarray(d["coef"], dtype=float)
        m._intercept = float(d["intercept"])
        m._mean = np.asarray(d["mean"], dtype=float)
        m._std = np.asarray(d["std"], dtype=float)
        m._base_rates = dict(d["base_rates"])
        m._type_serve_mode = dict(d.get("type_serve_mode", {}))  # fail-open: absent -> all "model"
        m._type_gate_metrics = dict(d.get("type_gate_metrics", {}))
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
    def load(cls, path: Path | str) -> GkCompletionModel:
        path = Path(path)
        want = (path / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
        if want != cls._sha(path):
            raise ValueError(f"GkCompletionModel integrity check failed at {path}")
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        sv = d.get("sklearn_version")
        if sv:
            import sklearn

            if sv.split(".")[0] != sklearn.__version__.split(".")[0]:
                warnings.warn(
                    "GkCompletionModel: sklearn major version differs from training; serve is "
                    "numpy-only so inference is unaffected.",
                    stacklevel=2,
                )
        return cls.from_dict(d)

    @classmethod
    def from_variant(cls, variant: str = "default") -> GkCompletionModel:
        # NOTE (m3): _VARIANT_CACHE returns a SHARED instance -- loaded models are treated IMMUTABLE
        # post-load (predict-only). A future mutator must clone, not mutate in place.
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        wdir = _WEIGHTS_ROOT / variant
        if not (wdir / "SHA256SUMS").exists():
            raise FileNotFoundError(
                f"No bundled Gk-completion weights for {variant!r} at {wdir}. Train via scripts/train_gk_completion.py."
            )
        m = cls.load(wdir)
        _VARIANT_CACHE[variant] = m
        return m


_PROVIDER_VARIANT = {"skillcorner": "skillcorner"}  # everything else -> the native-completion gs model


def variant_key_for_provider(source_provider: str | None) -> str:
    """Map a tracking ``source_provider`` to a GK-completion variant key (D-S2/C4 -- PURE, no IO).

    ``gs`` / ``sportec`` / ``snapshot`` / ``metrica`` / unknown / ``None`` -> ``"gs"`` (all share the
    native-completion construct once SkillCorner ``result_id`` is fixed, D-S8); ``skillcorner`` -> its
    own key (whose weights may equal ``gs`` -- the D-S1 GS-transfer re-measurement decides). Kept
    artifact-free so the mapping is exhaustively unit-testable; the IO seam is ``from_variant(key)``."""
    return _PROVIDER_VARIANT.get(str(source_provider).lower() if source_provider is not None else "", "gs")


def _gk_completion_density(
    actions: pd.DataFrame, frames: pd.DataFrame | None, geom: pd.DataFrame, links=None
) -> pd.Series:
    """receiver_zone_density at the RESOLVED destination -- the SINGLE shared producer used by
    BOTH prepare and the serve path (the divergence-prone step). NaN where unlinked -> the model
    mean-imputes."""
    if frames is None:
        return pd.Series(np.nan, index=actions.index)
    from .features import receiver_zone_density

    a = actions.copy()
    a["end_x"] = geom["dest_x"].to_numpy()
    a["end_y"] = geom["dest_y"].to_numpy()
    return receiver_zone_density(a, frames)


def compute_gk_completion(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    model: GkCompletionModel | None = None,
    links: pd.DataFrame | None = None,
) -> pd.Series:
    """Standalone GK-distribution completion probability per action (the RAV P(success) model,
    exposed for inspection / the lakehouse wide table). ``model=None`` -> bundled GS default.
    Uses the SAME geometry + shared density + extract path as training (parity)."""
    from ._gk_geometry import resolve_gk_geometry

    m = model if isinstance(model, GkCompletionModel) else GkCompletionModel.from_variant("default")
    geom = resolve_gk_geometry(actions, frames=frames, links=links)
    dens = _gk_completion_density(actions, frames, geom, links)
    X = extract_gk_completion_features(geom.assign(type_id=actions["type_id"].to_numpy()), defender_density=dens)
    return pd.Series(m.predict_proba(X), index=actions.index, name="gk_completion")


def prepare_gk_completion_training_data(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None,
    links: pd.DataFrame | None = None,
    min_class_fraction: float = 0.02,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (features, labels, groups) for the completion model. Train==serve by construction:
    the SAME serve-domain predicate (P1), resolve-geom-on-FULL-then-mask (P1-residual), the SAME
    shared density helper (C1), and the SAME extract_gk_completion_features (M2). Label =
    result_id==success. Drops geometry-unscoreable + NaN-id rows (P2/#2). Fails loud on a
    degenerate label distribution (m6). X carries the ``origin_source`` metadata column (for the
    native-origin gate; the model ignores it)."""
    from ._gk_geometry import resolve_gk_geometry
    from ._xt_gk import _gk_distribution_mask

    SUCCESS = spadlconfig.result_id["success"]
    # P1: the EXACT serve domain. Frames are needed for GK identity; without frames (unit tests)
    # fall back to goalkicks only.
    if frames is not None:
        mask = _gk_distribution_mask(actions, frames)
    else:
        mask = actions["type_id"].to_numpy() == _GOALKICK
    # P1-residual: resolve geometry on the FULL action list THEN mask, mirroring serve (the
    # positional next_event shift is frame-size-dependent).
    geom_full = resolve_gk_geometry(actions, frames=frames, links=links)
    domain = actions.loc[mask].copy()
    geom = geom_full.loc[mask]
    dens = _gk_completion_density(domain, frames, geom, links)
    X = extract_gk_completion_features(geom.assign(type_id=domain["type_id"].to_numpy()), defender_density=dens)
    X["origin_source"] = geom["origin_source"].to_numpy()  # metadata (NOT a feature) for the native gate
    y = (domain["result_id"].to_numpy() == SUCCESS).astype(int)
    groups = domain["game_id"].to_numpy() if "game_id" in domain.columns else np.zeros(len(domain))
    # P2: drop geometry-unscoreable rows (serve base-rates them); #2: also drop NaN-id rows (serve's
    # id_ok gate would route them to default) -> strict domain parity.
    geom_ok = np.isfinite(X["length"].to_numpy()) & np.isfinite(X["dest_x"].to_numpy())
    id_ok = domain["player_id"].notna().to_numpy() & domain["team_id"].notna().to_numpy()
    keep = geom_ok & id_ok
    # F1+G1: train ONLY on the native (pass_outcome) label -- the one construct giving BOTH classes.
    # inferred (received==True / next-action==targeted) is positive-only -> would bias the intercept
    # high -> mis-calibrate p (the primary hard gate). stopgap is the weak proxy. result_id keeps both
    # values for VAEP coverage; the completion model never trains on either. result_source is a
    # SkillCorner-only column (absent for GS / kloppy providers -> no-op).
    if "result_source" in domain.columns:
        keep = keep & (domain["result_source"] == "native").to_numpy()
    X = X.loc[keep].reset_index(drop=True)
    y, groups = y[keep], groups[keep]
    frac = float(y.mean()) if len(y) else 0.0
    if len(y) == 0 or min(frac, 1.0 - frac) < min_class_fraction or len(np.unique(y)) < 2:
        raise ValueError(
            f"degenerate label distribution (success rate={frac:.3f}); check provider "
            "result_id semantics (m6/D5) before training."
        )
    return X, y, groups
