# TF-18: Ghost-GK Positioning Model

**Date:** 2026-05-25
**Status:** Approved
**Layer:** GKDV Layer 2 (TF-18)
**Prereqs:** TF-15 GK influence primitives (shipped 3.9.0), TF-7 pitch control (shipped 3.6.0), TF-14 defensive line (shipped 3.4.0), TF-5 ball carrier (shipped 3.5.0)

## Purpose

Predict where a league-average goalkeeper would position themselves given the current game state. The model answers: "In this situation, where would any GK typically stand?" This enables TF-19 (GK Deterrent Value) to compute the counterfactual "what threat reduction does this GK's actual position provide vs the expected position?"

## Module Structure

**Production module:** `silly_kicks/tracking/_ghost_gk.py`
**Training script:** `scripts/train_ghost_gk.py`
**Artifact format:** ONNX (tree model for leaf assignment) + `.npz` (RFCDE training coordinates) + `metadata.json` — zero pickle/joblib
**Artifact location:** Lazy-downloaded from HuggingFace Hub on first use (no binary in package)

## Public API

### Core Class

```python
class GhostGkModel:
    """League-average GK positioning model using RFCDE density estimation."""

    def fit(self, frames: pd.DataFrame, labels: pd.DataFrame) -> Self:
        """Train on tracking frames with known GK positions."""

    def predict(self, frames: pd.DataFrame) -> np.ndarray:
        """Predict mode (x, y) for each frame. Returns shape (n_frames, 2)."""

    def predict_density(self, frames: pd.DataFrame) -> list[GhostGkDensity]:
        """Full density prediction per frame."""

    def save(self, path: Path) -> None:
        """Serialize to ONNX + npz + metadata.json (no pickle)."""

    @classmethod
    def load(cls, path: Path) -> GhostGkModel:
        """Load from local directory path.

        SHA-256 hash verification: checks each file against SHA256SUMS;
        raises IntegrityError on mismatch.
        """

    @classmethod
    def from_hub(cls, repo_id: str = "karsten-s-nielsen/ghost-gk-v1") -> GhostGkModel:
        """Download from HuggingFace Hub and load.

        Uses huggingface_hub.hf_hub_download; caches locally.
        SHA-256 verification applied after download.
        """
```

### Density Dataclass

```python
@dataclasses.dataclass(frozen=True)
class GhostGkDensity:
    mode_x: float         # Joint 2D mode x (argmax of probabilities grid)
    mode_y: float         # Joint 2D mode y (argmax of probabilities grid)
    mean_x: float         # Density-weighted mean x
    mean_y: float         # Density-weighted mean y
    spread: float         # Effective area (entropy-based)
    grid_x: np.ndarray    # Shape (60,) — x-axis cell centers [0, 30]
    grid_y: np.ndarray    # Shape (64,) — y-axis cell centers [18, 50]
    probabilities: np.ndarray  # Shape (60, 64), sums to ~1.0
```

### ADR-005 Surface

```python
def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | None = None,
) -> pd.DataFrame:
    """Per-frame primitive. Adds ghost_gk_x, ghost_gk_y, ghost_gk_spread columns."""

@nan_safe_enrichment
def add_ghost_gk(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Action-coupled aggregator. Provenance skip guard, links kwarg."""

def ghost_gk_xfns(gamestates: list[pd.DataFrame]) -> pd.DataFrame:
    """VAEP factory. Silent NaN on introspection (missing extension columns)."""
```

### Conventions

- `model=None` default: first checks `onnxruntime` availability — if missing, raises `ImportError("Ghost GK requires: pip install silly-kicks[ghost-gk]")` BEFORE attempting download. Then lazy-downloads subsampled model from HuggingFace Hub via `huggingface_hub.hf_hub_download` (cached locally after first call). Override path via `SILLY_KICKS_GHOST_GK_PATH` env var for offline/air-gapped environments.
- `model=GhostGkModel.from_hub("karsten-s-nielsen/ghost-gk-full-v1")`: explicit Hub download (separate classmethod, no URI parsing ambiguity)
- `model=GhostGkModel.load(Path("/local/path/"))`: explicit local path (must be `Path` object or existing directory — no string-type overloading)
- `links` kwarg: pre-linking optimization (skip internal `link_actions_to_frames`)
- Provenance skip guard: idempotent when provenance columns already present
- `@nan_safe_enrichment`: NaN identifiers route to NaN output
- Atomic mirror: `atomic.tracking.features.add_ghost_gk`
- Thread safety: `GhostGkModel` is thread-safe for concurrent `predict()`/`predict_density()` calls. All inference state (ONNX session, training coordinates, leaf matrix) is read-only after `load()`; per-frame KDE computation uses local arrays.

## Feature Engineering

### 26-Feature Candidate Set (Goal-Relative Coordinates)

All spatial features are transformed to goal-relative coordinates where the defending goal is at x=0. This doubles effective training data by removing LTR/RTL asymmetry.

**Important:** Features are computed from the actual frame state including the actual GK position. The ghost model answers "given this situation, where would an average GK be?" — it does not remove the GK from the game state. The TF-19 counterfactual evaluates positioning QUALITY (actual vs expected), not a GK-absent scenario.

**Ball (5):**
- `ball_x`, `ball_y` — goal-relative position
- `ball_vx`, `ball_vy` — velocity components
- `ball_distance_to_goal` — Euclidean distance to goal center

**Defensive line (3):**
- `defensive_line_x` — back-line x position
- `defensive_line_depth` — distance between deepest defender and GK
- `defensive_line_width` — lateral spread of back line

**Attacking players (4):**
- `attackers_in_box` — count inside penalty area
- `nearest_attacker_to_goal_x` — deepest attacker x
- `attacker_centroid_x`, `attacker_centroid_y` — attacking team center of mass

**Defending players (2):**
- `defenders_behind_ball` — count between ball and goal
- `deepest_defender_x` — closest defender to own goal

**Game state (5):**
- `phase` — categorical feature via HistGBRT `categorical_features` parameter: open_play / set_piece / goal_kick (no ordinal encoding)
- `team_in_possession` — binary: 1 = GK's team has ball, 0 = opponent (derived from ball_carrier team_id via TF-5)
- `score_diff` — GK's team score minus opponent
- `time_seconds` — elapsed time in period
- `period_id` — 1 or 2 (ET frames filtered during training; at inference period_id > 2 clamped to 2)

**Spatial geometry (4):**
- `ball_to_goal_angle` — angle from ball to goal center
- `ball_to_nearest_attacker_dist` — proximity of closest threat
- `defending_team_compactness` — convex hull area of defending outfield
- `ball_in_own_half` — binary indicator

**Velocity (3):**
All velocity features computed at a canonical 0.5s backward-difference window regardless of source frame rate. This normalizes velocity semantics across 1fps subsampled and 25fps full-resolution tiers, preventing train/serve distribution mismatch.
- `ball_speed` — magnitude of ball velocity
- `defensive_line_speed` — rate of change of defensive line x
- `defending_centroid_vx` — team movement direction

### Target Variables

- `gk_x`, `gk_y` — actual GK position in goal-relative coordinates (from `is_goalkeeper` identification)

## Model Architecture

### Single Joint-Density Model (No Independent x/y Regressors)

GK x and y are spatially correlated (sweeper-keepers at high x tend toward central y; near-post positioning has low x and low y). Independent x/y models can predict physically impossible combinations.

**Architecture:** A single `HistGradientBoostingRegressor` trained on `target = gk_x` (arbitrary — used purely for leaf-assignment quality, not point prediction). The forest's role is to partition the feature space into regions of similar game state. All spatial predictions come from the joint 2D density layer.

- Native NaN handling via `HistGradientBoostingRegressor` (velocity features may be NaN on first frames of period)
- `categorical_features` parameter for `phase` column (no ordinal encoding)
- `.apply(X)` method provides leaf indices for RFCDE weight computation

**Methodology departure:** HistGBRT chosen over the original RFCDE paper's RandomForest (Pospisil & Lee 2018) for native NaN handling of velocity features. Boosted trees optimize leaves sequentially (correcting residuals) vs RF's independently-grown trees, producing different partition geometry. Leaf-assignment quality validated via RFCDE calibration metrics (see Model Acceptance Criteria).

### RFCDE Density Layer

1. **Forest leaf assignment:** `model.apply(X)` → leaf indices for each tree
2. **Weight computation:** For prediction frame i, weight of training sample j = fraction of trees where i and j land in the same leaf
3. **Weighted 2D KDE:** Apply weights to training (gk_x, gk_y) joint positions
4. **Grid evaluation:** Evaluate KDE on the output grid
5. **Mode extraction:** `argmax` of 2D probability grid → joint (x, y) mode preserving spatial correlation
6. **Bandwidth:** Scott's rule, computed per-frame from effective weighted sample count

`predict()` returns the joint 2D mode (not two independent 1D predictions). This guarantees physically plausible (x, y) combinations.

### Density Grid Specification

- **Domain:** [0, 30] x [18, 50] in goal-relative coordinates
  - x: 0 (goal line) to 30m (accommodates sweeper-keepers: Neuer/Ederson operate at 20-25m)
  - y: 18m to 50m (half-width of pitch, centered on goal)
- **Resolution:** ~0.5m per cell
- **Grid size:** 60 x 64 = 3,840 cells
- **Output:** Probability mass per cell, sums to ~1.0
- **Note:** Grid bounds are fixed at [0, 30] × [18, 50] for API stability (consumers depend on shape (60, 64) and axis ranges). This was validated against training data: 99th percentile of observed GK positions falls within [0, 28] × [20, 48], confirming the 30m x-limit covers >99.5% of positions including sweeper-keeper extremes.

### Serialization Format (No Pickle)

silly-kicks has zero pickle/joblib usage. This model will not introduce one.

**Artifact structure** (directory or `.zip`):
```
ghost_gk_v1/
├── model.onnx          # Tree structure for leaf assignment (via skl2onnx)
├── rfcde_weights.npz   # training_gk_x, training_gk_y, leaf_assignments
├── metadata.json       # feature_names, grid_spec, version, sha256_manifest
└── SHA256SUMS          # Per-file integrity hashes
```

- **ONNX** for tree model: deterministic, non-executable format. Inference via `onnxruntime` (hard dependency of the `[ghost-gk]` extra — no fallback path). Users who install `silly_kicks[ghost-gk]` get onnxruntime; the complexity of reimplementing tree inference from stored arrays is not justified when dependency management solves the problem. Note: `skl2onnx` is a training-only dependency (used in `scripts/train_ghost_gk.py`), not required at inference time.
- **NPZ** for RFCDE weights: pure numpy arrays (training coordinates + precomputed leaf matrix).
- **SHA-256 verification** on load: check each file against `SHA256SUMS`; reject artifact if any hash mismatches. Shipped model hash embedded as constant in source.

### Tiered Deployment

| Tier | Training data | Artifact size | Location |
|------|---------------|---------------|----------|
| Default | Subsampled ~1fps | ~10-15MB | `karsten-s-nielsen/ghost-gk-v1` (lazy download via `from_hub()`) |
| Full | All frames (~25fps) | ~50-100MB | `karsten-s-nielsen/ghost-gk-full-v1` (explicit `from_hub(repo_id)`) |

Package ships zero binary artifacts. First call to `compute_ghost_gk(model=None)` triggers `huggingface_hub.hf_hub_download` to local cache (`~/.cache/huggingface/hub/...`). Offline override: `SILLY_KICKS_GHOST_GK_PATH=/path/to/ghost_gk_v1/`.

Subsampling rationale: consecutive frames at 25fps are nearly identical; 1fps retains all meaningful state transitions while reducing artifact size by ~25x.

## Training Pipeline

### Data Sources

| Provider | Role | Rationale |
|----------|------|-----------|
| Gradient Sports (WC 2022) | Primary | Stadium optical, reliable ball + GK, international tactical variety |
| Sportec (DFL) | Secondary | Stadium optical, native GK ID, Bundesliga tactical patterns |
| SkillCorner | Tertiary | Multi-league diversity (EPL, La Liga, Serie A, Ligue 1), broadcast CV |

### Cross-Validation

Match-level `GroupKFold` with 5 folds:
- Groups = `game_id`
- Prevents temporal autocorrelation leakage (never train and test on same match)
- Reports per-fold metrics for stability assessment

### Training Script Interface

```bash
python scripts/train_ghost_gk.py \
    --data-dir /path/to/tracking/parquet/ \
    --output-dir models/ \
    --subsample-fps 1.0 \
    --n-estimators 500 \
    --max-depth 8 \
    --cv-folds 5
```

### Feature Importance Pruning

Post-training, report permutation importance. Process is human-in-loop:
1. Train → generate importance report (sorted ranking + % contribution)
2. Human reviews: features below 1% contribution flagged as removal candidates
3. Retrain without flagged features → compare CV metrics
4. Accept pruned model if metrics hold within 0.1m MAE degradation; reject otherwise
5. Document final feature set in `metadata.json`

Initial 26 features are deliberately broad; expect convergence to 15-18 informative features.

## TF-19 Integration Path

TF-19 needs full density grids, not just point predictions. The integration path is:

**Frame-level density access:** TF-19's `compute_gk_deterrent_value` calls `model.predict_density(frames)` directly — bypassing the DataFrame surface (`compute_ghost_gk`) which only emits scalar summary columns. This is the clean architectural boundary: DataFrame API for action-coupled consumers, object API for spatial-composition consumers.

```python
def compute_gk_deterrent_value(
    actual_gk_pos: tuple[float, float],
    ghost_density: GhostGkDensity,
    shot_reachable_region: np.ndarray,  # same grid shape (60, 64)
) -> float:
    """
    Probability mass of ghost density in shot-reachable region that
    the actual GK covers better/worse than expected.
    """
```

The density grid uses the same 0.5m resolution and meshgrid convention as TF-7 pitch control, enabling element-wise operations after cropping the pitch control grid (full pitch ~105×68m) to the ghost GK domain ([0,30]×[18,50]). Cropping is a simple index slice with no interpolation required.

## Testing Strategy

### Unit Tests (`tests/tracking/test_ghost_gk.py`)

| Test | Validates |
|------|-----------|
| `test_extract_ghost_gk_features` | 26 feature columns, correct dtypes, goal-relative normalization, `team_in_possession` present |
| `test_ghost_gk_model_fit_predict` | `.fit()` + `.predict()` returns (n, 2) joint mode |
| `test_ghost_gk_density` | `.predict_density()` grid sums to ~1.0, mode in bounds |
| `test_ghost_gk_density_joint_mode` | Mode is argmax of 2D grid, not two independent 1D argmaxes |
| `test_ghost_gk_model_save_load` | Round-trip: save → load → predict matches (ONNX+npz, no pickle) |
| `test_ghost_gk_model_sha256_verification` | Tampered artifact raises `IntegrityError` on load |
| `test_compute_ghost_gk_no_model` | Clear error when no model available and Hub unreachable |
| `test_add_ghost_gk_aggregator` | Expected columns, links kwarg, provenance skip guard |
| `test_ghost_gk_xfns_factory` | Correct column names, silent NaN on dummy gamestate |
| `test_goal_relative_transform_symmetry` | LTR vs RTL frames produce identical goal-relative coords |
| `test_density_grid_bounds` | grid_x covers [0,30], grid_y covers [18,50], probabilities shape (60,64) |
| `test_ghost_gk_predict_with_nan_features` | NaN ball velocity (first frame) + NaN defensive_line_speed → finite predictions in bounds |
| `test_velocity_canonical_window` | Features computed at 0.5s window regardless of source fps (1fps vs 25fps input → `assert_allclose(rtol=0.05)` — sub-frame interpolation differences tolerated) |
| `test_ghost_gk_missing_onnxruntime_extra` | Mock onnxruntime unavailable → `ImportError` with "pip install silly-kicks[ghost-gk]" message before any download attempt |
| `test_defending_team_compactness_degenerate` | Collinear defenders (< 3 or all on a line) → NaN compactness (QhullError caught), predictions still finite |

### Integration Tests (`tests/tracking/test_ghost_gk_integration.py`)

| Test | Validates |
|------|-----------|
| `test_add_ghost_gk_dtype_mismatch` | int64 actions + str frames → no crash (PR-S53 pattern) |
| `test_ghost_gk_with_gk_deterrent_interface` | Density output compatible with TF-19 stub |
| `test_atomic_mirror` | `atomic.tracking.features.add_ghost_gk` same columns |
| `test_ghost_gk_from_hub_download` | `@pytest.mark.network` — `from_hub()` → download → load → predictions finite and in bounds. Catches SHA-256 manifest mismatches, missing Hub files, onnxruntime version incompatibilities. |

### E2e Tests (`@pytest.mark.e2e`)

| Test | Validates |
|------|-----------|
| `test_ghost_gk_gradientsports_wc2022` | Full pipeline + **programmatic acceptance assertions**: `assert mae < 2.0`, `assert nll < uniform_nll`, `assert std_mae_across_folds < 0.5` |
| `test_ghost_gk_sportec_dfl` | Cross-provider generalization, `assert mae < 2.5` (relaxed vs GS — fewer available matches increases estimate variance) |
| `test_ghost_gk_metrica_degraded` | Graceful NaN on sparse ball data: no crash, output shape correct (not a quality gate) |

### Model Acceptance Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Mode MAE | < 2.0m | ~1 body width; GK positioning is coarse |
| Per-provider MAE | < 3.0m (no single provider) | Catches provider-specific regression masked by aggregate |
| Cross-fold MAE std | < 0.5m | Stability across match groups (no provider/match bias) |
| Mean NLL | < uniform baseline | Density is informative |
| Calibration | > 45% true in top-50% region | Sanity check |
| Artifact size | < 15MB | Reasonable download on first use |
| Predict latency | < 5ms/frame | Batch of 22 frames |
| No data leakage | Match-level split | GroupKFold enforced |
| Feature diversity | >= 3 features > 5% importance | Model not degenerate |
| SHA-256 integrity | All files pass | Security: reject tampered artifacts |

### Quality Metrics (Reported in Training Output)

- Mode MAE (overall + per-phase breakdown)
- Mean negative log-likelihood vs uniform baseline
- Calibration curve (% of true positions in top-N% density regions)
- Per-provider MAE (detect provider-specific bias)
- Feature permutation importance ranking

## Academic Attribution

NOTICE entry required:

- Le, Yue, Carr & Lucey (2017) — "Data-Driven Ghosting Using Deep Imitation Learning" (MIT Sloan SAC) — ghosting methodology ancestor
- Dutta, Yurko, Ventura (2024) — "NFL Ghosts: A framework for evaluating defender positioning with conditional density estimation" (arXiv:2406.17220) — RFCDE-for-positioning framework, direct methodological source
- Pospisil & Lee (2018) — "RFCDE: Random Forests for Conditional Density Estimation" — originating paper for the density estimation technique

## Version & Release

- **Version:** TBD (next minor after 3.18.x)
- **CHANGELOG section:** `### Added` — Ghost-GK positioning model (TF-18)
- **ADR:** Amendment to ADR-005 (new feature kind: externally-hosted ML model artifact with lazy download). Documents `[ghost-gk]` extra isolation — onnxruntime (~50-200MB) kept out of base install via extras to avoid inflating `pip install silly-kicks` for users who don't use tracking ML features.
