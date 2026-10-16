"""RunZoneModel -- probabilistic (initial x target)-zone run encoder (TF-46 Arm 1, Shaw §3.1).

A params-only ADR-011 fitted model: sklearn ``GaussianMixture`` at FIT only; serve is pure numpy
(manual Gaussian responsibilities), so ``load()`` imports no sklearn. Two GMMs -- one over initial
positions (``t_kick - 2s``), one over target positions -- with component counts BIC-selected unless
pinned. The target components whose means fall inside the penalty area are the ACTIVE zones (a-g, Shaw);
a run is encoded probabilistically as ``P(initial zone i) * P(target zone t)`` summed over active
attacking players.

Serialization: ``model.npz`` (raw arrays) + ``metadata.json`` (LF) + ``SHA256SUMS``. ``load()`` verifies
the checksums, the chirality fingerprint (ADR-037 -- zones are y-sensitive) and the feature contract
(ADR-050 -- the penalty-area geometry the active-zone selection consumed), raising ``IntegrityError``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._geometry import in_penalty_area_absolute

_MODEL_VERSION = "run-zone-1"
_ATTACKED_GOAL_X = float(spadlconfig.field_length)  # frame-LTR canonical: attacking goal at x=105


class IntegrityError(Exception):
    """Raised when a RunZoneModel artifact fails a load-time integrity check (ADR-011)."""


def _gaussian_responsibilities(xy: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Soft component assignment P(component | point) for a 2D GMM -- pure numpy (no sklearn).

    Returns ``(n_points, n_components)``; a point with zero total density gets a uniform row.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    k = means.shape[0]
    n = xy.shape[0]
    comp = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        cov = covs[j] + np.eye(2) * 1e-9  # jitter for numerical safety
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        diff = xy - means[j]
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        comp[:, j] = weights[j] * np.exp(-0.5 * maha) / (2.0 * np.pi * np.sqrt(max(det, 1e-300)))
    total = comp.sum(axis=1, keepdims=True)
    zero = (total <= 0).ravel()
    resp = np.divide(comp, total, out=np.full_like(comp, 1.0 / k), where=total > 0)
    if zero.any():
        resp[zero] = 1.0 / k
    return resp


class RunZoneModel:
    """Probabilistic run encoder over (initial, target) zones. See module docstring + NOTICE.

    Examples
    --------
    Fit on corpus snapshots, then encode one corner's runs::

        model = RunZoneModel().fit(init_xy, target_xy)   # (n_players, 2) arrays over the corpus
        run_vector = model.encode_runs(corner_init_xy, corner_target_xy)
    """

    def __init__(self) -> None:
        self.init_means: np.ndarray | None = None
        self.init_covs: np.ndarray | None = None
        self.init_weights: np.ndarray | None = None
        self.target_means: np.ndarray | None = None
        self.target_covs: np.ndarray | None = None
        self.target_weights: np.ndarray | None = None
        self.active_target_idx: np.ndarray | None = None  # indices of in-penalty-area target components
        self.selected: dict = {}  # fit provenance (BIC curves, selected k)
        self.training_commit: str | None = None

    # --- fit (sklearn, local import) ----------------------------------------------------------
    def fit(
        self,
        init_xy: np.ndarray,
        target_xy: np.ndarray,
        *,
        init_k: int | None = None,
        target_k: int | None = None,
        init_k_range: tuple[int, int] = (2, 6),
        target_k_range: tuple[int, int] = (8, 16),
        random_state: int = 0,
    ) -> RunZoneModel:
        """Fit the two GMMs. Component counts BIC-selected over the ranges unless pinned.

        Examples
        --------
        Pin the component counts (BIC-select by leaving them ``None``)::

            model = RunZoneModel().fit(init_xy, target_xy, init_k=6, target_k=15)
        """
        from sklearn.mixture import GaussianMixture

        init_xy = np.asarray(init_xy, dtype=np.float64).reshape(-1, 2)
        target_xy = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)

        def _fit_bic(data: np.ndarray, k_fixed: int | None, k_range: tuple[int, int]):
            if k_fixed is not None:
                g = GaussianMixture(n_components=k_fixed, covariance_type="full", random_state=random_state)
                g.fit(data)
                return g, {"selected_k": k_fixed, "bic": {str(k_fixed): float(g.bic(data))}}
            lo, hi = k_range
            hi = min(hi, max(lo, len(data)))  # can't fit more components than points
            curve: dict[str, float] = {}
            best = None
            best_bic = np.inf
            for k in range(lo, hi + 1):
                g = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
                g.fit(data)
                b = float(g.bic(data))
                curve[str(k)] = b
                if b < best_bic:
                    best_bic, best = b, g
            if best is None:
                raise ValueError("RunZoneModel.fit: empty BIC range (k_range below the data size)")
            return best, {"selected_k": int(best.n_components), "bic": curve}

        gi, si = _fit_bic(init_xy, init_k, init_k_range)
        gt, st = _fit_bic(target_xy, target_k, target_k_range)

        tgt_means = np.asarray(gt.means_, dtype=np.float64)  # local: non-Optional for the active-zone scan
        self.init_means, self.init_covs, self.init_weights = gi.means_, gi.covariances_, gi.weights_
        self.target_means, self.target_covs, self.target_weights = tgt_means, gt.covariances_, gt.weights_
        self.active_target_idx = np.array(
            [
                j
                for j in range(tgt_means.shape[0])
                if in_penalty_area_absolute(
                    float(tgt_means[j, 0]), float(tgt_means[j, 1]), attacked_goal_x=_ATTACKED_GOAL_X
                )
            ],
            dtype=np.int64,
        )
        if self.active_target_idx.size == 0:
            raise ValueError("RunZoneModel.fit: no target component falls inside the penalty area (no active zones).")
        self.selected = {"init": si, "target": st, "n_active_target": int(self.active_target_idx.size)}
        return self

    # --- serve (pure numpy) -------------------------------------------------------------------
    @property
    def n_initial_zones(self) -> int:
        """Number of fitted initial zones.

        Examples
        --------
        Read the fitted initial-zone count::

            model = RunZoneModel.load(path)
            n_initial = model.n_initial_zones
        """
        return 0 if self.init_means is None else int(self.init_means.shape[0])

    @property
    def n_target_zones(self) -> int:
        """Number of active (in-penalty-area) target zones.

        Examples
        --------
        Read the active target-zone count::

            model = RunZoneModel.load(path)
            n_target = model.n_target_zones
        """
        return 0 if self.active_target_idx is None else int(self.active_target_idx.size)

    def encode_runs(
        self, init_xy: np.ndarray, target_xy: np.ndarray, active_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Probabilistic run vector of length ``n_initial_zones * n_target_zones``.

        ``active_mask`` selects the attacking players counted (default: those with >=0.5 target-mass on
        the active zones -- Shaw's "active players near an active zone").

        Examples
        --------
        Encode one corner's aligned initial/target player positions::

            run_vector = model.encode_runs(init_xy, target_xy)   # (n_init * n_target,) weights
        """
        if (
            self.init_means is None
            or self.init_covs is None
            or self.init_weights is None
            or self.target_means is None
            or self.target_covs is None
            or self.target_weights is None
            or self.active_target_idx is None
        ):
            raise RuntimeError("RunZoneModel is not fitted/loaded.")
        r_init = _gaussian_responsibilities(init_xy, self.init_means, self.init_covs, self.init_weights)
        r_tgt = _gaussian_responsibilities(target_xy, self.target_means, self.target_covs, self.target_weights)
        r_tgt_active = r_tgt[:, self.active_target_idx]
        if active_mask is None:
            active_mask = r_tgt_active.sum(axis=1) >= 0.5
        active_mask = np.asarray(active_mask, dtype=bool)
        vec = np.zeros((self.n_initial_zones, self.n_target_zones), dtype=np.float64)
        for p in np.flatnonzero(active_mask):
            vec += np.outer(r_init[p], r_tgt_active[p])
        return vec.ravel()

    # --- serialization (ADR-011) --------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialise to ``model.npz`` + ``metadata.json`` + ``SHA256SUMS`` (no pickle; ADR-011).

        Examples
        --------
        Persist a fitted model to a bundle directory::

            model.save(weights_dir)   # writes model.npz, metadata.json, SHA256SUMS
        """
        if self.init_means is None:
            raise RuntimeError("RunZoneModel is not fitted.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "model.npz",
            init_means=self.init_means,
            init_covs=self.init_covs,
            init_weights=self.init_weights,
            target_means=self.target_means,
            target_covs=self.target_covs,
            target_weights=self.target_weights,
            active_target_idx=self.active_target_idx,
        )
        metadata = {
            "version": _MODEL_VERSION,
            "pitch_length": float(spadlconfig.field_length),
            "pitch_width": float(spadlconfig.field_width),
            "selected": self.selected,
            "chirality": _chirality_block(self),
            "feature_contract": _feature_contract_block(self),
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
    def load(cls, path: Path, *, legacy_override: bool = False) -> RunZoneModel:
        """Load a bundle, verifying checksums + chirality + feature contract (ADR-011).

        Examples
        --------
        Load a fitted model (raises ``IntegrityError`` on a tampered or mis-chiral artifact)::

            model = RunZoneModel.load(weights_dir)
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
        rec_len, rec_wid = meta.get("pitch_length"), meta.get("pitch_width")
        if rec_len is not None and (
            rec_len != float(spadlconfig.field_length) or rec_wid != float(spadlconfig.field_width)
        ):
            raise IntegrityError(
                f"Pitch-dimension mismatch: model {rec_len}x{rec_wid} m vs library "
                f"{spadlconfig.field_length}x{spadlconfig.field_width} m; refusing to load (retrain)."
            )
        with np.load(path / "model.npz", allow_pickle=False) as z:
            model = cls()
            model.init_means = z["init_means"]
            model.init_covs = z["init_covs"]
            model.init_weights = z["init_weights"]
            model.target_means = z["target_means"]
            model.target_covs = z["target_covs"]
            model.target_weights = z["target_weights"]
            model.active_target_idx = z["active_target_idx"]
        model.selected = meta.get("selected", {})
        model.training_commit = meta.get("training_commit")

        from silly_kicks.tracking._chirality import verify_chirality
        from silly_kicks.tracking._feature_contract import verify_feature_contract

        verify_chirality(
            _chirality_block(model),
            meta.get("chirality"),
            legacy_override=legacy_override,
            model_name="RunZoneModel",
            error_cls=IntegrityError,
        )
        verify_feature_contract(
            _feature_contract_block(model),
            meta.get("feature_contract"),
            legacy_override=legacy_override,
            model_name="RunZoneModel",
            error_cls=IntegrityError,
        )
        return model


def _probe_xy(frame: pd.DataFrame) -> np.ndarray:
    """Non-ball player (x, y) from a probe frame."""
    sub = frame[~frame["is_ball"].astype(bool)] if "is_ball" in frame.columns else frame
    return sub[["x", "y"]].to_numpy(dtype=np.float64)


def _chirality_block(model: RunZoneModel) -> dict:
    """Behavioral chirality (ADR-037): the encoder's output on the y-asymmetric probe frame."""
    from silly_kicks.tracking._chirality import chirality_fingerprint

    def _predict(frame: pd.DataFrame) -> np.ndarray:
        xy = _probe_xy(frame)
        return model.encode_runs(xy, xy, active_mask=np.ones(len(xy), dtype=bool))

    return chirality_fingerprint(_predict)


def _feature_contract_block(model: RunZoneModel) -> dict:
    """Feature contract (ADR-050): the encoder's vector on the fixed contract probe frame, plus the
    penalty-area constants the active-zone selection consumed (a geometry change -> re-fit)."""
    from silly_kicks.tracking._feature_contract import contract_probe_frame, feature_contract

    def _vec() -> np.ndarray:
        xy = _probe_xy(contract_probe_frame())
        return model.encode_runs(xy, xy, active_mask=np.ones(len(xy), dtype=bool))

    return feature_contract(
        _vec,
        constants={
            "penalty_area_half_width": float(spadlconfig.penalty_area_half_width),
            "penalty_area_depth": float(spadlconfig.penalty_area_depth),
        },
    )
