"""Ghost-GK positioning model (TF-18, GKDV Layer 2).

Predicts where a league-average goalkeeper would position themselves
given the current game state, using RFCDE density estimation over
HistGradientBoostingRegressor leaf assignments.

Input frames MUST be in LTR-normalized convention (home team attacks
right in all periods — the standard silly-kicks tracking output after
play_left_to_right normalization).

See docs/superpowers/specs/2026-05-25-tf18-ghost-gk-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import gaussian_kde

from silly_kicks.spadl import config as spadlconfig

# ---------------------------------------------------------------------------
# Grid constants (fixed for API stability — see spec Density Grid)
# ---------------------------------------------------------------------------

GRID_X_MIN = 0.0
GRID_X_MAX = 30.0
GRID_Y_MIN = 18.0
GRID_Y_MAX = 50.0
GRID_NX = 60
GRID_NY = 64
GRID_RESOLUTION = 0.5  # meters per cell

# Cell centers
_GRID_X = np.linspace(
    GRID_X_MIN + GRID_RESOLUTION / 2,
    GRID_X_MAX - GRID_RESOLUTION / 2,
    GRID_NX,
)
_GRID_Y = np.linspace(
    GRID_Y_MIN + GRID_RESOLUTION / 2,
    GRID_Y_MAX - GRID_RESOLUTION / 2,
    GRID_NY,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GhostGkDensity:
    """Per-frame ghost-GK density prediction.

    Attributes
    ----------
    mode_x : float
        Joint 2D mode x (argmax of probabilities grid), goal-relative.
    mode_y : float
        Joint 2D mode y (argmax of probabilities grid), goal-relative.
    mean_x : float
        Density-weighted mean x.
    mean_y : float
        Density-weighted mean y.
    spread : float
        Effective area (entropy-based measure of density dispersion).
    grid_x : np.ndarray
        Shape (60,) — x-axis cell centers covering [0, 30].
    grid_y : np.ndarray
        Shape (64,) — y-axis cell centers covering [18, 50].
    probabilities : np.ndarray
        Shape (60, 64) — probability mass per cell, sums to ~1.0.
        Read-only (writeable=False).

    Examples
    --------
    >>> density = model.predict_density(features)[0]
    >>> density.mode_x, density.mode_y
    (5.25, 34.25)
    >>> density.probabilities.sum()
    1.0
    """

    mode_x: float
    mode_y: float
    mean_x: float
    mean_y: float
    spread: float
    grid_x: np.ndarray
    grid_y: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        # Prevent mutation of numpy arrays after construction
        object.__setattr__(self, "grid_x", self.grid_x.copy())
        object.__setattr__(self, "grid_y", self.grid_y.copy())
        object.__setattr__(self, "probabilities", self.probabilities.copy())
        self.grid_x.flags.writeable = False
        self.grid_y.flags.writeable = False
        self.probabilities.flags.writeable = False


# ---------------------------------------------------------------------------
# Errors + constants
# ---------------------------------------------------------------------------


class IntegrityError(Exception):
    """Raised when artifact SHA-256 verification fails."""


_ENV_VAR = "SILLY_KICKS_GHOST_GK_PATH"


def _resolve_model(model: GhostGkModel | None) -> GhostGkModel:
    """Resolve model parameter: load default from Hub if None.

    Checks huggingface_hub availability BEFORE attempting download.
    Respects SILLY_KICKS_GHOST_GK_PATH env var for offline use.

    Examples
    --------
    >>> resolved = _resolve_model(None)  # downloads from Hub
    >>> resolved = _resolve_model(my_model)  # pass-through
    """
    if model is not None:
        return model

    # Check huggingface_hub availability before download attempt
    try:
        import huggingface_hub  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        msg = "Ghost GK requires: pip install silly-kicks[ghost-gk]"
        raise ImportError(msg) from None

    # Check env var override
    env_path = os.environ.get(_ENV_VAR)
    if env_path is not None:
        return GhostGkModel.load(Path(env_path))

    return GhostGkModel.from_hub()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_FIELD_LENGTH = spadlconfig.field_length  # 105.0
_FIELD_WIDTH = spadlconfig.field_width  # 68.0
_GOAL_Y = _FIELD_WIDTH / 2.0  # 34.0
_PENALTY_AREA_X = 16.5
_PENALTY_AREA_Y_MIN = (_FIELD_WIDTH - 40.3) / 2.0
_PENALTY_AREA_Y_MAX = (_FIELD_WIDTH + 40.3) / 2.0
_VELOCITY_WINDOW_S = 0.5

GHOST_GK_FEATURE_NAMES: list[str] = [
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "ball_distance_to_goal",
    "defensive_line_x",
    "defensive_line_depth",
    "defensive_line_width",
    "attackers_in_box",
    "nearest_attacker_to_goal_x",
    "attacker_centroid_x",
    "attacker_centroid_y",
    "defenders_behind_ball",
    "deepest_defender_x",
    "phase",
    "team_in_possession",
    "score_diff",
    "time_seconds",
    "period_id",
    "ball_to_goal_angle",
    "ball_to_nearest_attacker_dist",
    "defending_team_compactness",
    "ball_in_own_half",
    "ball_speed",
    "defensive_line_speed",
    "defending_centroid_vx",
]


def extract_ghost_gk_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id: int | str,
    goal_x: float = 0.0,
    score_diff: float = 0.0,
    phase: int = 0,
    ball_carrier_team_id: int | str | None = None,
    prev_defensive_line_x: float | None = None,
    prev_defending_centroid_x: float | None = None,
    dt: float = _VELOCITY_WINDOW_S,
) -> pd.DataFrame:
    """Extract 26 ghost-GK features from frame data in goal-relative coords.

    Accepts a single frame (one game_id/period_id/frame_id group) and
    returns a single-row DataFrame. Called in a vectorized batch by
    compute_ghost_gk.

    Input frames MUST be in LTR-normalized convention (home attacks right
    in all periods). The goal_x parameter specifies which end the GK
    defends: 0.0 for the home GK, 105.0 for the away GK.

    Parameters
    ----------
    frame_data : pd.DataFrame
        All rows for one frame — players + ball.
    gk_team_id : int | str
        Team ID of the GK whose ghost position we predict.
    goal_x : float
        x-coordinate of the defending goal (0.0 or 105.0).
    score_diff : float
        GK's team score minus opponent.
    phase : int
        0 = open_play, 1 = set_piece, 2 = goal_kick.
    ball_carrier_team_id : int | str | None
        Team currently in possession.
    prev_defensive_line_x : float | None
        Previous frame's defensive line x (for velocity).
    prev_defending_centroid_x : float | None
        Previous frame's defending centroid x.
    dt : float
        Time delta for velocity computation.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 26 columns (GHOST_GK_FEATURE_NAMES).

    Examples
    --------
    >>> features = extract_ghost_gk_features(frame, gk_team_id=1, goal_x=0.0)
    >>> features.shape
    (1, 26)
    """
    # --- Coordinate transform ---
    flip = goal_x > 50.0

    def to_gr_x(x: float) -> float:
        return (_FIELD_LENGTH - x) if flip else x

    def to_gr_vx(vx: float) -> float:
        return -vx if flip else vx

    # --- Ball ---
    ball = frame_data[frame_data["is_ball"].astype(bool)]
    if len(ball) > 0:
        bx_raw = float(ball["x"].iloc[0])
        by_raw = float(ball["y"].iloc[0])
        bvx_raw = float(ball["vx"].iloc[0]) if "vx" in ball.columns else np.nan
        bvy_raw = float(ball["vy"].iloc[0]) if "vy" in ball.columns else np.nan
    else:
        bx_raw = by_raw = bvx_raw = bvy_raw = np.nan

    ball_x = to_gr_x(bx_raw)
    ball_y = by_raw
    ball_vx = to_gr_vx(bvx_raw)
    ball_vy = bvy_raw
    ball_speed = float(np.sqrt(bvx_raw**2 + bvy_raw**2)) if not np.isnan(bvx_raw) else np.nan
    ball_dist = float(np.sqrt(ball_x**2 + (ball_y - _GOAL_Y) ** 2)) if not np.isnan(ball_x) else np.nan

    # --- Player splits ---
    players = frame_data[~frame_data["is_ball"].astype(bool)]
    defending = players[(players["team_id"] == gk_team_id) & (~players["is_goalkeeper"].astype(bool))]
    attacking = players[(players["team_id"] != gk_team_id) & (~players["is_goalkeeper"].astype(bool))]
    gk_rows = players[(players["team_id"] == gk_team_id) & (players["is_goalkeeper"].astype(bool))]

    # --- Defensive line ---
    if len(defending) > 0:
        def_xs = np.asarray(defending["x"].apply(to_gr_x).values)
        sorted_xs = np.sort(def_xs)
        n_back = min(4, len(sorted_xs))
        defensive_line_x = float(np.median(sorted_xs[:n_back]))
        deepest_defender_x = float(sorted_xs[0])
        defensive_line_width = float(defending["y"].max() - defending["y"].min())
    else:
        defensive_line_x = deepest_defender_x = defensive_line_width = np.nan

    if len(gk_rows) > 0 and not np.isnan(defensive_line_x):
        gk_x_gr = to_gr_x(float(gk_rows["x"].iloc[0]))
        defensive_line_depth = abs(defensive_line_x - gk_x_gr)
    else:
        defensive_line_depth = np.nan

    # --- Attacking players ---
    if len(attacking) > 0:
        atk_xs = np.asarray(attacking["x"].apply(to_gr_x).values)
        atk_ys = np.asarray(attacking["y"].values)
        nearest_atk_x = float(np.min(atk_xs))
        atk_cx = float(np.mean(atk_xs))
        atk_cy = float(np.mean(atk_ys))
        in_box = (atk_xs < _PENALTY_AREA_X) & (atk_ys >= _PENALTY_AREA_Y_MIN) & (atk_ys <= _PENALTY_AREA_Y_MAX)
        attackers_in_box = int(np.sum(in_box))
    else:
        nearest_atk_x = atk_cx = atk_cy = np.nan
        attackers_in_box = 0

    # --- Defenders behind ball ---
    if len(defending) > 0 and not np.isnan(ball_x):
        def_xs_arr = np.asarray(defending["x"].apply(to_gr_x).values)
        defenders_behind_ball = int(np.sum(def_xs_arr < ball_x))
    else:
        defenders_behind_ball = 0

    # --- Spatial geometry ---
    ball_to_goal_angle = float(np.arctan2(ball_y - _GOAL_Y, ball_x)) if not np.isnan(ball_x) else np.nan

    if len(attacking) > 0 and not np.isnan(ball_x):
        atk_xs_rel = np.asarray(attacking["x"].apply(to_gr_x).values)
        dists = np.sqrt((atk_xs_rel - ball_x) ** 2 + (np.asarray(attacking["y"].values) - ball_y) ** 2)
        ball_to_nearest_atk = float(np.min(dists))
    else:
        ball_to_nearest_atk = np.nan

    if len(defending) >= 3:
        coords = np.column_stack([np.asarray(defending["x"].apply(to_gr_x).values), np.asarray(defending["y"].values)])
        try:
            hull = ConvexHull(coords)
            compactness = float(hull.volume)
        except QhullError:
            compactness = np.nan
    else:
        compactness = np.nan

    ball_in_own_half = 1.0 if (not np.isnan(ball_x) and ball_x < _FIELD_LENGTH / 2) else 0.0
    team_in_poss = 1.0 if ball_carrier_team_id == gk_team_id else 0.0
    period_clamped = min(int(frame_data["period_id"].iloc[0]), 2)
    time_s = float(frame_data["timestamp"].iloc[0]) if "timestamp" in frame_data.columns else 0.0

    # --- Velocity (actual dt from consecutive frames) ---
    if prev_defensive_line_x is not None and not np.isnan(defensive_line_x):
        def_line_speed = (defensive_line_x - prev_defensive_line_x) / dt
    else:
        def_line_speed = np.nan

    if prev_defending_centroid_x is not None and len(defending) > 0:
        def_cx = float(np.mean(np.asarray(defending["x"].apply(to_gr_x).values)))
        def_centroid_vx = (def_cx - prev_defending_centroid_x) / dt
    else:
        def_centroid_vx = np.nan

    row = [
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        ball_dist,
        defensive_line_x,
        defensive_line_depth,
        defensive_line_width,
        attackers_in_box,
        nearest_atk_x,
        atk_cx,
        atk_cy,
        defenders_behind_ball,
        deepest_defender_x,
        phase,
        team_in_poss,
        score_diff,
        time_s,
        period_clamped,
        ball_to_goal_angle,
        ball_to_nearest_atk,
        compactness,
        ball_in_own_half,
        ball_speed,
        def_line_speed,
        def_centroid_vx,
    ]
    return pd.DataFrame([row], columns=GHOST_GK_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Vectorized tree traversal
# ---------------------------------------------------------------------------


def _vectorized_leaf_indices(nodes_list: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Vectorized tree traversal for all trees at once.

    Parameters
    ----------
    nodes_list : list[np.ndarray]
        Each element is a structured array of tree nodes with fields:
        left, right, feature_idx, num_threshold, missing_go_to_left.
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_trees) — leaf node index per sample per tree.
    """
    n_samples = X.shape[0]
    n_trees = len(nodes_list)
    leaves = np.zeros((n_samples, n_trees), dtype=np.intp)

    for t_idx, nodes in enumerate(nodes_list):
        current = np.zeros(n_samples, dtype=np.intp)

        for _ in range(100):  # depth bound
            node_data = nodes[current]
            is_leaf = node_data["left"] == 0
            if np.all(is_leaf):
                break

            feature_idx = node_data["feature_idx"]
            thresholds = node_data["num_threshold"]
            go_left_missing = node_data["missing_go_to_left"]

            # Gather feature values for each sample's current split feature
            feat_vals = X[np.arange(n_samples), feature_idx]

            # Decision: NaN handling + threshold comparison
            is_nan = np.isnan(feat_vals)
            go_left = np.where(
                is_nan,
                go_left_missing.astype(bool),
                feat_vals <= thresholds,
            )

            # Advance non-leaf samples
            next_node = np.where(go_left, node_data["left"], node_data["right"])
            current = np.where(is_leaf, current, next_node)

        leaves[:, t_idx] = current

    return leaves


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class GhostGkModel:
    """League-average GK positioning model using RFCDE density estimation.

    Uses HistGradientBoostingRegressor leaf-assignment (NaN-tolerant)
    + weighted 2D KDE for joint density estimation. The regressor is
    trained on gk_x as a proxy target — its role is purely to partition
    feature space into regions of similar game state.

    Thread-safe for concurrent predict/predict_density calls after fit/load.

    Examples
    --------
    >>> model = GhostGkModel()
    >>> model.fit(X_train, labels_train)
    >>> positions = model.predict(X_test)  # shape (n, 2)
    >>> densities = model.predict_density(X_test)  # list[GhostGkDensity]

    See NOTICE for full bibliographic citations.
    """

    def __init__(self, *, n_estimators: int = 500, max_depth: int = 8):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._tree_nodes: list[np.ndarray] | None = None
        self._training_gk_x: np.ndarray | None = None
        self._training_gk_y: np.ndarray | None = None
        self._training_leaves: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> GhostGkModel:
        """Train on feature matrix + (gk_x, gk_y) labels.

        Parameters
        ----------
        features : pd.DataFrame
            Shape (n_samples, 26).
        labels : pd.DataFrame
            Columns "gk_x", "gk_y" — actual GK positions (goal-relative).

        Examples
        --------
        >>> model = GhostGkModel()
        >>> model.fit(X_train, labels_train)
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = features.values.astype(np.float64)
        y_x = labels["gk_x"].values.astype(np.float64)

        # Determine categorical feature index (phase)
        phase_idx = list(features.columns).index("phase") if "phase" in features.columns else None
        cat_features = [phase_idx] if phase_idx is not None else []

        cat_arg: list[int] | str = cat_features if cat_features else "from_dtype"
        regressor = HistGradientBoostingRegressor(
            max_iter=self._n_estimators,
            max_depth=self._max_depth,
            categorical_features=cat_arg,  # type: ignore[arg-type]
            random_state=42,
        )
        regressor.fit(X, y_x)

        # Extract tree node arrays for serialization + inference
        self._tree_nodes = []
        for tree_list in regressor._predictors:
            tree = tree_list[0]
            self._tree_nodes.append(tree.nodes.copy())

        # Compute training leaves
        self._training_leaves = _vectorized_leaf_indices(self._tree_nodes, X)
        self._training_gk_x = np.array(y_x, copy=True)
        self._training_gk_y = np.asarray(labels["gk_y"].values, dtype=np.float64).copy()

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict joint (x, y) mode for each sample.

        Returns shape (n_samples, 2) — argmax of joint 2D density grid.

        Examples
        --------
        >>> model.predict(X_test).shape
        (100, 2)
        """
        densities = self.predict_density(features)
        return np.array([[d.mode_x, d.mode_y] for d in densities])

    def predict_density(self, features: pd.DataFrame) -> list[GhostGkDensity]:
        """Full density prediction per sample.

        Computes leaf co-occurrence weights, weighted 2D KDE, grid evaluation.

        Examples
        --------
        >>> densities = model.predict_density(X_test)
        >>> densities[0].probabilities.sum()
        1.0
        """
        if (
            self._tree_nodes is None
            or self._training_leaves is None
            or self._training_gk_x is None
            or self._training_gk_y is None
        ):
            msg = "Model not fitted. Call .fit() or .load() first."
            raise RuntimeError(msg)

        training_gk_x = self._training_gk_x
        training_gk_y = self._training_gk_y

        X = features.values.astype(np.float64)
        query_leaves = _vectorized_leaf_indices(self._tree_nodes, X)
        n_trees = query_leaves.shape[1]

        # Precompute grid mesh
        grid_xx, grid_yy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
        grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()])

        results: list[GhostGkDensity] = []
        for i in range(len(X)):
            # Weight = fraction of trees with matching leaf
            matches = self._training_leaves == query_leaves[i]
            weights = matches.sum(axis=1).astype(np.float64) / n_trees

            nonzero = weights > 0
            w = weights[nonzero]
            gk_x_w = training_gk_x[nonzero]
            gk_y_w = training_gk_y[nonzero]

            if len(w) < 2:
                probs = np.ones((GRID_NX, GRID_NY)) / (GRID_NX * GRID_NY)
            else:
                try:
                    kde = gaussian_kde(
                        np.vstack([gk_x_w, gk_y_w]),
                        weights=w,
                        bw_method="scott",
                    )
                    density_vals = kde(grid_points)
                    probs = density_vals.reshape(GRID_NX, GRID_NY)
                    total = probs.sum()
                    if total > 0:
                        probs = probs / total
                    else:
                        probs = np.ones((GRID_NX, GRID_NY)) / (GRID_NX * GRID_NY)
                except np.linalg.LinAlgError:
                    probs = np.ones((GRID_NX, GRID_NY)) / (GRID_NX * GRID_NY)

            # Joint 2D mode
            flat_idx = int(np.argmax(probs))
            ix, iy = np.unravel_index(flat_idx, probs.shape)
            mode_x = float(_GRID_X[ix])
            mode_y = float(_GRID_Y[iy])

            # Mean
            mean_x = float(np.sum(probs * grid_xx))
            mean_y = float(np.sum(probs * grid_yy))

            # Spread (entropy)
            nz = probs[probs > 0]
            entropy = float(-np.sum(nz * np.log(nz)))
            spread = float(np.exp(entropy) * GRID_RESOLUTION**2)

            results.append(
                GhostGkDensity(
                    mode_x=mode_x,
                    mode_y=mode_y,
                    mean_x=mean_x,
                    mean_y=mean_y,
                    spread=spread,
                    grid_x=_GRID_X,
                    grid_y=_GRID_Y,
                    probabilities=probs,
                )
            )

        return results

    def save(self, path: Path) -> None:
        """Serialize to npz + metadata.json + SHA256SUMS (no pickle).

        Artifact structure:
        - rfcde_weights.npz: tree_nodes_* arrays + training_gk_x/y + training_leaves
        - metadata.json: feature_names, grid_spec, hyperparams, version
        - SHA256SUMS: per-file integrity hashes

        Examples
        --------
        >>> model.save(Path("models/ghost_gk_v1"))
        """
        if (
            self._tree_nodes is None
            or self._training_gk_x is None
            or self._training_gk_y is None
            or self._training_leaves is None
        ):
            msg = "Model not fitted. Call .fit() first."
            raise RuntimeError(msg)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Weights npz
        save_dict: dict[str, np.ndarray] = {
            "training_gk_x": self._training_gk_x,
            "training_gk_y": self._training_gk_y,
            "training_leaves": self._training_leaves,
            "n_trees": np.array([len(self._tree_nodes)]),
        }
        for i, nodes in enumerate(self._tree_nodes):
            save_dict[f"tree_nodes_{i}"] = nodes.view(np.uint8)
            save_dict[f"tree_dtype_{i}"] = np.array([str(nodes.dtype)], dtype="U2000")

        npz_path = path / "rfcde_weights.npz"
        np.savez_compressed(str(npz_path), **save_dict)  # type: ignore[arg-type]

        # 2. Metadata
        metadata = {
            "feature_names": GHOST_GK_FEATURE_NAMES,
            "grid_spec": {
                "x_min": GRID_X_MIN,
                "x_max": GRID_X_MAX,
                "y_min": GRID_Y_MIN,
                "y_max": GRID_Y_MAX,
                "nx": GRID_NX,
                "ny": GRID_NY,
                "resolution": GRID_RESOLUTION,
            },
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "version": "1.0.0",
        }
        meta_path = path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 3. SHA-256
        sums_path = path / "SHA256SUMS"
        with open(sums_path, "w") as f:
            for fname in ["rfcde_weights.npz", "metadata.json"]:
                h = hashlib.sha256((path / fname).read_bytes()).hexdigest()
                f.write(f"{h}  {fname}\n")

    @classmethod
    def load(cls, path: Path) -> GhostGkModel:
        """Load from local directory with SHA-256 verification.

        No sklearn or onnxruntime needed — tree traversal uses stored
        node arrays with numpy.

        Examples
        --------
        >>> model = GhostGkModel.load(Path("models/ghost_gk_v1"))
        """
        path = Path(path)

        # Verify integrity
        sums_path = path / "SHA256SUMS"
        if not sums_path.exists():
            raise IntegrityError(f"SHA256SUMS not found in {path}")

        with open(sums_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                expected_hash, fname = line.split("  ", 1)
                actual_hash = hashlib.sha256((path / fname).read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    raise IntegrityError(
                        f"Integrity check failed for {fname}: expected {expected_hash}, got {actual_hash}"
                    )

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load npz (use context manager to release file handle on Windows)
        with np.load(path / "rfcde_weights.npz", allow_pickle=False) as data:
            n_trees = int(data["n_trees"][0])
            tree_nodes = []
            for i in range(n_trees):
                raw_bytes = np.array(data[f"tree_nodes_{i}"])
                dtype_str = str(data[f"tree_dtype_{i}"][0])
                # ast.literal_eval safely parses the dtype descriptor (list of tuples)
                dtype = np.dtype(ast.literal_eval(dtype_str))
                nodes = raw_bytes.view(dtype)
                tree_nodes.append(nodes)

            training_gk_x = np.array(data["training_gk_x"])
            training_gk_y = np.array(data["training_gk_y"])
            training_leaves = np.array(data["training_leaves"])

        model = cls(
            n_estimators=metadata.get("n_estimators", 500),
            max_depth=metadata.get("max_depth", 8),
        )
        model._tree_nodes = tree_nodes
        model._training_gk_x = training_gk_x
        model._training_gk_y = training_gk_y
        model._training_leaves = training_leaves

        return model

    @classmethod
    def from_hub(cls, repo_id: str = "karsten-s-nielsen/ghost-gk-v1") -> GhostGkModel:
        """Download from HuggingFace Hub and load.

        Examples
        --------
        >>> model = GhostGkModel.from_hub()
        """
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError:
            msg = "Ghost GK requires: pip install silly-kicks[ghost-gk]"
            raise ImportError(msg) from None

        local_dir = snapshot_download(repo_id=repo_id)
        return cls.load(Path(local_dir))


# ---------------------------------------------------------------------------
# Per-frame primitive
# ---------------------------------------------------------------------------


def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | None = None,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Per-frame ghost-GK primitive (batched).

    Adds ghost_gk_x, ghost_gk_y, ghost_gk_spread columns.
    One prediction per (frame, GK team). Results written to GK rows.

    Input frames MUST be in LTR-normalized convention (home team attacks
    right in all periods — standard silly-kicks tracking output).

    Parameters
    ----------
    frames : pd.DataFrame
        Tracking frames (TRACKING_FRAMES_COLUMNS schema, LTR-normalized).
    model : GhostGkModel | None
        Pre-loaded model. None = lazy download from Hub.
    home_team_id : int | str
        Home team ID (attacks right -> defends at x=0).

    Returns
    -------
    pd.DataFrame
        Copy of frames with ghost_gk_x, ghost_gk_y, ghost_gk_spread added.

    Examples
    --------
    >>> from silly_kicks.tracking._ghost_gk import compute_ghost_gk
    >>> result = compute_ghost_gk(frames, home_team_id=1)
    """
    resolved = _resolve_model(model)
    out = frames.copy()
    out["ghost_gk_x"] = np.nan
    out["ghost_gk_y"] = np.nan
    out["ghost_gk_spread"] = np.nan

    group_keys = ["game_id", "period_id", "frame_id"]
    grouped = list(frames.groupby(group_keys, sort=True))

    # Collect all (frame_group, gk_team, goal_x) + track velocity state
    feature_rows: list[pd.DataFrame] = []
    index_map: list[tuple] = []  # (game_id, period_id, frame_id, gk_team_id)

    # Per-team velocity state tracking
    prev_state: dict[tuple, tuple[float, float]] = {}
    prev_timestamps: dict[tuple, float] = {}

    for (gid, pid, fid), frame_data in grouped:
        gk_rows = frame_data[frame_data["is_goalkeeper"].astype(bool) & ~frame_data["is_ball"].astype(bool)]
        for _, gk_row in gk_rows.iterrows():
            gk_team = gk_row["team_id"]
            # Direction: home team defends at x=0, away at x=105 (LTR convention)
            goal_x = 0.0 if str(gk_team) == str(home_team_id) else _FIELD_LENGTH

            # Velocity from previous frame with actual dt
            state_key = (gid, gk_team)
            prev_dl_x, prev_dc_x = prev_state.get(state_key, (None, None))
            current_ts = float(frame_data["timestamp"].iloc[0]) if "timestamp" in frame_data.columns else 0.0
            prev_ts = prev_timestamps.get(state_key)
            actual_dt = (current_ts - prev_ts) if prev_ts is not None and current_ts > prev_ts else _VELOCITY_WINDOW_S

            feat = extract_ghost_gk_features(
                frame_data,
                gk_team_id=gk_team,
                goal_x=goal_x,
                prev_defensive_line_x=prev_dl_x,
                prev_defending_centroid_x=prev_dc_x,
                dt=actual_dt,
            )
            feature_rows.append(feat)
            index_map.append((gid, pid, fid, gk_team))

            # Update velocity state for next frame
            defending = frame_data[
                (frame_data["team_id"] == gk_team)
                & ~frame_data["is_goalkeeper"].astype(bool)
                & ~frame_data["is_ball"].astype(bool)
            ]
            if len(defending) > 0:
                flip = goal_x > 50.0
                def_cx = float(np.mean((_FIELD_LENGTH - defending["x"]) if flip else defending["x"]))
            else:
                def_cx = np.nan
            prev_state[state_key] = (
                float(feat["defensive_line_x"].iloc[0]),
                def_cx,
            )
            prev_timestamps[state_key] = current_ts

    if not feature_rows:
        return out

    # Batch predict
    batch_features = pd.concat(feature_rows, ignore_index=True)
    densities = resolved.predict_density(batch_features)

    # Build result DataFrame from predictions (single merge, not O(n*m) loop)
    result_df = pd.DataFrame(
        {
            "game_id": [t[0] for t in index_map],
            "period_id": [t[1] for t in index_map],
            "frame_id": [t[2] for t in index_map],
            "team_id": [t[3] for t in index_map],
            "ghost_gk_x": [d.mode_x for d in densities],
            "ghost_gk_y": [d.mode_y for d in densities],
            "ghost_gk_spread": [d.spread for d in densities],
        }
    )

    # Merge into GK rows via single join
    gk_mask = out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)
    gk_rows_df = out.loc[gk_mask, ["game_id", "period_id", "frame_id", "team_id"]].copy()
    gk_rows_df = gk_rows_df.merge(
        result_df,
        on=["game_id", "period_id", "frame_id", "team_id"],
        how="left",
    )
    out.loc[gk_mask, "ghost_gk_x"] = gk_rows_df["ghost_gk_x"].values
    out.loc[gk_mask, "ghost_gk_y"] = gk_rows_df["ghost_gk_y"].values
    out.loc[gk_mask, "ghost_gk_spread"] = gk_rows_df["ghost_gk_spread"].values

    return out
