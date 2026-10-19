"""CornerRoutineModel (NMF) + agglomerative routine clustering (TF-46 Arm 1, Shaw §3.2).

``CornerRoutineModel`` factors the per-corner run vectors (the term matrix, corners x runs) into a
basis of co-occurring run-combinations (NMF components) and a per-corner coefficient vector. sklearn
NMF at FIT only; ``transform`` is a pure-numpy non-negative multiplicative-update solve against the
FROZEN basis, so ``load()`` imports no sklearn.

Unlike ``RunZoneModel``, this model carries NO chirality fingerprint: it transforms orientation-neutral
run vectors (the reflection / y-handedness is resolved upstream in ``RunZoneModel``), not a frame, so
the ADR-037 y-mirror class does not apply. Serialization is ``model.npz`` + ``metadata.json`` (LF) +
``SHA256SUMS`` with a load-time integrity check.

``cluster_corner_routines`` groups corners by their coefficient vectors (agglomerative, raw structure
only -- archetype naming / ranking stay consumer-side, ADR-110).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._columns import CORNER_KEYS
from ._config import RoutineParams

_MODEL_VERSION = "corner-routine-1"
_DEFAULT_ROUTINE_PARAMS = RoutineParams()


class IntegrityError(Exception):
    """Raised when a CornerRoutineModel artifact fails a load-time integrity check (ADR-011)."""


def _nmf_transform(x: np.ndarray, basis: np.ndarray, *, n_iter: int = 300, eps: float = 1e-9) -> np.ndarray:
    """Pure-numpy NMF transform: solve ``min_{W>=0} ||X - W H||_F`` for W given a frozen basis ``H``.

    Deterministic multiplicative update from a fixed non-negative init (no random state).
    ``x``: (n, n_runs); ``basis`` H: (r, n_runs); returns W: (n, r).
    """
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(basis, dtype=np.float64)
    n, r = x.shape[0], h.shape[0]
    hht = h @ h.T  # (r, r)
    xht = x @ h.T  # (n, r)
    w = np.full((n, r), np.abs(xht).mean() + eps, dtype=np.float64)  # fixed non-negative init
    for _ in range(n_iter):
        w *= xht / (w @ hht + eps)
    return np.clip(w, 0.0, None)


class CornerRoutineModel:
    """NMF routine-feature model. See module docstring + NOTICE.

    Examples
    --------
    Fit the routine basis on the corpus term matrix, then transform one match's corners::

        model = CornerRoutineModel().fit(term_matrix)   # (n_corners, n_runs); rank elbow-selected
        coeffs = model.transform(match_term_matrix)      # (n_corners, rank)
    """

    def __init__(self) -> None:
        self.basis: np.ndarray | None = None  # H: (rank, n_runs)
        self.rank: int | None = None
        self.n_runs: int | None = None
        self.alpha: float = 0.0
        self.selected: dict = {}
        self.training_commit: str | None = None

    def fit(
        self,
        term_matrix: np.ndarray,
        *,
        rank: int | None = None,
        rank_range: tuple[int, int] = (2, 30),
        alpha: float = 0.0,
        random_state: int = 0,
    ) -> CornerRoutineModel:
        """Fit the NMF basis. ``rank`` is reconstruction-elbow-selected over ``rank_range`` unless pinned.

        Examples
        --------
        Pin the rank, or leave it ``None`` to elbow-select::

            model = CornerRoutineModel().fit(term_matrix, rank=7)  # or leave rank/alpha default
        """
        from sklearn.decomposition import NMF

        x = np.asarray(term_matrix, dtype=np.float64)
        self.n_runs = int(x.shape[1])
        self.alpha = float(alpha)

        def _fit_rank(k: int):
            # kwargs dict: sklearn's NMF carries only a partial type stub (mis-types n_components /
            # alpha_H); a **dict keeps the call pyright-clean without a blanket type: ignore.
            nmf_kwargs = {
                "n_components": k,
                "init": "nndsvda",
                "solver": "mu",
                "beta_loss": "frobenius",
                "alpha_W": alpha,
                "alpha_H": alpha,
                "l1_ratio": 1.0,
                "max_iter": 1000,
                "random_state": random_state,
            }
            model = NMF(**nmf_kwargs)
            w = model.fit_transform(x)
            err = float(np.linalg.norm(x - w @ model.components_, ord="fro"))
            return model.components_, err

        if rank is not None:
            basis, err = _fit_rank(rank)
            self.rank, self.basis = rank, basis
            self.selected = {"selected_rank": rank, "reconstruction": {str(rank): err}}
            return self

        lo, hi = rank_range
        hi = min(hi, max(lo, min(x.shape)))  # nndsvda requires rank <= min(n_samples, n_features)
        curve: dict[str, float] = {}
        bases: dict[int, np.ndarray] = {}
        for k in range(lo, hi + 1):
            basis, err = _fit_rank(k)
            curve[str(k)] = err
            bases[k] = basis
        self.rank = _elbow(curve)
        self.basis = bases[self.rank]
        self.selected = {"selected_rank": self.rank, "reconstruction": curve}
        return self

    def transform(self, term_matrix: np.ndarray) -> np.ndarray:
        """Per-corner coefficient vectors W (n_corners x rank) against the frozen basis (pure numpy).

        Examples
        --------
        Project run vectors onto the fitted routine basis::

            coeffs = model.transform(term_matrix)   # (n_corners, rank), non-negative
        """
        if self.basis is None:
            raise RuntimeError("CornerRoutineModel is not fitted/loaded.")
        return _nmf_transform(np.asarray(term_matrix, dtype=np.float64), self.basis)

    def save(self, path: Path) -> None:
        """Serialise to ``model.npz`` + ``metadata.json`` + ``SHA256SUMS`` (no pickle; ADR-011).

        Examples
        --------
        Persist the fitted basis to a bundle directory::

            model.save(weights_dir)
        """
        if self.basis is None:
            raise RuntimeError("CornerRoutineModel is not fitted.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "model.npz", basis=self.basis)
        metadata = {
            "version": _MODEL_VERSION,
            "rank": self.rank,
            "n_runs": self.n_runs,
            "alpha": self.alpha,
            "selected": self.selected,
            "training_commit": self.training_commit,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), newline="\n")
        with open(path / "SHA256SUMS", "w", newline="\n") as f:
            for fname in ["model.npz", "metadata.json"]:
                raw = (path / fname).read_bytes()
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")

    @classmethod
    def load(cls, path: Path) -> CornerRoutineModel:
        """Load a bundle, verifying checksums (ADR-011); raises ``IntegrityError`` on tamper.

        Examples
        --------
        Load the fitted routine basis::

            model = CornerRoutineModel.load(weights_dir)
        """
        path = Path(path)
        sums = path / "SHA256SUMS"
        if not sums.exists():
            raise IntegrityError(f"SHA256SUMS not found in {path}")
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            expected, fname = line.split("  ", 1)
            raw = (path / fname).read_bytes()
            if fname.endswith(".json"):
                raw = raw.replace(b"\r\n", b"\n")
            if hashlib.sha256(raw).hexdigest() != expected:
                raise IntegrityError(f"Integrity check failed for {fname}")
        meta = json.loads((path / "metadata.json").read_text())
        with np.load(path / "model.npz", allow_pickle=False) as z:
            model = cls()
            model.basis = z["basis"]
        model.rank = meta.get("rank")
        model.n_runs = meta.get("n_runs")
        model.alpha = meta.get("alpha", 1.0)
        model.selected = meta.get("selected", {})
        model.training_commit = meta.get("training_commit")
        return model


def _elbow(curve: dict[str, float]) -> int:
    """The rank at the reconstruction-error elbow (max distance to the chord between endpoints)."""
    ks = sorted(int(k) for k in curve)
    errs = np.array([curve[str(k)] for k in ks], dtype=np.float64)
    if len(ks) <= 2:
        return ks[0]
    x = np.array(ks, dtype=np.float64)
    x0, x1, y0, y1 = x[0], x[-1], errs[0], errs[-1]
    # perpendicular distance of each point to the line (x0,y0)-(x1,y1)
    num = np.abs((y1 - y0) * x - (x1 - x0) * errs + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0) + 1e-12
    return int(ks[int(np.argmax(num / den))])


def _routine_feature_names(model: CornerRoutineModel) -> list[str]:
    """The ``routine_feat_<k>`` column names."""
    return [f"routine_feat_{k}" for k in range(int(model.rank or 0))]


def cluster_corner_routines(features: pd.DataFrame, *, params: RoutineParams = _DEFAULT_ROUTINE_PARAMS) -> pd.DataFrame:
    """Group corners by their routine-feature coefficients (agglomerative). Pure.

    Returns ``(game_id, corner_id, cluster_id)`` -- RAW cluster structure only; archetype naming and
    which-routine-beats-which ranking stay consumer-side (ADR-110). A NaN-coefficient corner (short /
    unclear) is kept with ``cluster_id = -1`` (unclustered), never dropped (ADR-042). The scipy linkage
    matrix is attached to ``df.attrs["linkage"]`` for a consumer dendrogram.

    Examples
    --------
    Group corners into a fixed number of routine clusters::

        clusters = cluster_corner_routines(routine_features, params=RoutineParams(n_clusters=8))
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    feat_cols = [c for c in features.columns if c.startswith("routine_feat_")]
    out = features[CORNER_KEYS].copy()
    coeffs = features[feat_cols].to_numpy(dtype=np.float64)
    valid = ~np.isnan(coeffs).any(axis=1)
    cluster_id = np.full(len(features), -1, dtype=np.int64)
    link = None
    n_valid = int(valid.sum())
    if n_valid == 1:
        cluster_id[valid] = 0
    elif n_valid >= 2:
        link = linkage(coeffs[valid], method=params.linkage)
        if params.n_clusters is not None:
            labels = fcluster(link, t=params.n_clusters, criterion="maxclust")
        else:
            labels = fcluster(link, t=params.distance_threshold, criterion="distance")
        cluster_id[valid] = labels.astype(np.int64)
    out["cluster_id"] = cluster_id
    out.attrs["linkage"] = link
    return out


def compute_corner_routine_features(encoding: pd.DataFrame, *, model: CornerRoutineModel) -> pd.DataFrame:
    """Per-corner NMF routine-feature coefficients at grain ``(game_id, corner_id)``. Pure.

    Examples
    --------
    Project the per-corner run encoding onto the fitted routine basis::

        features = compute_corner_routine_features(encoding, model=routine_model)
    """
    feat_cols = _routine_feature_names(model)
    run_cols = [c for c in encoding.columns if c.startswith("run_")]
    out = encoding[CORNER_KEYS].copy()
    x = encoding[run_cols].to_numpy(dtype=np.float64)
    valid = ~np.isnan(x).any(axis=1)
    coeffs = np.full((len(encoding), len(feat_cols)), np.nan, dtype=np.float64)
    if valid.any():
        coeffs[valid] = model.transform(x[valid])
    for j, col in enumerate(feat_cols):
        out[col] = coeffs[:, j]
    return out
