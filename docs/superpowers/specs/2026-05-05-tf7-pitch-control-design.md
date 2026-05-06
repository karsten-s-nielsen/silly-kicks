# TF-7: Pitch Control Models — Design Specification

| Field | Value |
|---|---|
| **Date** | 2026-05-05 |
| **Status** | Draft |
| **Target release** | silly-kicks 3.7.0 |
| **Size** | Monstah |
| **Depends on** | PR-S24 (preprocess/derive_velocities), PR-S19 (tracking schema) |
| **Unblocks** | TF-15, TF-16, TF-17, TF-18, TF-19 (entire GKDV research program) |

## 1. Purpose

Pitch control is the foundational spatial primitive for the GKDV research
program (TF-15..TF-19) and a general-purpose building block for pass
evaluation, space creation analysis, and EPV computation. It answers: "for
each point on the pitch, what is the probability that team T controls the ball
if it arrives there?"

This PR ships three published pitch control formulations as a first-class
subpackage, with per-player decomposition (needed by TF-15), optional numba
acceleration, and full VAEP integration.

## 2. References

- Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017).
  "Physics-Based Modeling of Pass Probabilities in Soccer." MIT Sloan SAC.
- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan SAC.
- Fernandez, J., & Bornn, L. (2018). "Wide Open Spaces: A statistical
  technique for measuring space creation in professional soccer." MIT Sloan SAC.
- Shaw, L. (2020). Friends of Tracking — LaurieOnTracking reference
  implementation. GitHub: Friends-of-Tracking-Data-FoTD/LaurieOnTracking.
- DataBallPy space_occupation documentation (Fernandez/Bornn implementation).
- Lakehouse `src/analytics/pitch_control.py` (Spearman, JAX/NumPy) — used for
  cross-reference parity testing. The lakehouse is a consumer of silly-kicks;
  silly-kicks is the reference implementation, not the other way around.

## 3. Architecture

### 3.1 Module structure

```
silly_kicks/tracking/pitch_control/
├── __init__.py              # Public API re-exports
├── _surface.py              # PitchControlSurface frozen dataclass + methods
├── _params.py               # SpearmanParams, FernandezBornnParams, VoronoiParams
├── _spearman.py             # Kinematic TTI + logistic influence + sum aggregation
├── _fernandez_bornn.py      # Velocity-scaled bivariate normal + sigmoid aggregation
├── _voronoi.py              # Nearest-player tessellation (broadcast argmin)
├── _dispatch.py             # compute_pitch_control() router + ball-travel-time
└── _numba_kernels.py        # Optional @njit TTI + influence + Gaussian kernels
```

### 3.2 Dependency direction

```
features.py (action-coupled xfns)
    └── pitch_control._dispatch.compute_pitch_control()
            ├── _spearman.compute_spearman()
            │       └── _numba_kernels (optional, try/except)
            ├── _fernandez_bornn.compute_fernandez_bornn()
            │       └── _numba_kernels (optional, try/except)
            ├── _voronoi.compute_voronoi()
            └── _surface.PitchControlSurface (return type for all)
```

### 3.3 Key invariants

- **Zero I/O, zero global state** — hexagonal contract per ADR-004.
- **Input**: silly-kicks tracking DataFrame (20-column long-form, 105x68 meters).
  No coordinate conversion needed.
- **Output**: `PitchControlSurface` frozen dataclass.
- **numba is optional** — `ImportError` falls through to NumPy silently.
- **scipy is already a transitive dep** (via scikit-learn) — Voronoi adds no
  new runtime deps.
- **xarray is optional** — `.to_xarray()` raises `ImportError` with helpful
  message if not installed.

### 3.4 Public API

```python
from silly_kicks.tracking.pitch_control import (
    # Core compute
    compute_pitch_control,            # dispatch router (grid surface)
    compute_pitch_control_at_points,  # batch point queries (no grid)

    # Return type
    PitchControlSurface,

    # Params
    SpearmanParams,
    FernandezBornnParams,
    VoronoiParams,

    # Type
    Method,  # Literal["spearman", "fernandez_bornn", "voronoi"]
)
```

## 4. PitchControlSurface dataclass

```python
@dataclass(frozen=True)
class PitchControlSurface:
    """Spatial pitch control field for a single frame.

    Values in [0, 1]: 1.0 = full attacking-team control,
    0.0 = full defending-team control, 0.5 = contested.
    """

    grid_x: np.ndarray             # (nx,) cell centers in meters [0, 105]
    grid_y: np.ndarray             # (ny,) cell centers in meters [0, 68]
    surface: np.ndarray            # (ny, nx) control values [0, 1]
    method: str                    # "spearman" | "fernandez_bornn" | "voronoi"
    attacking_team_id: int | str   # which team_id maps to 1.0

    # Optional decomposition (computed when decompose=True)
    per_player_influence: np.ndarray | None = None  # (n_players, ny, nx)
    player_ids: np.ndarray | None = None            # (n_players,) aligning axis 0
    player_team_ids: np.ndarray | None = None       # (n_players,) team membership

    # --- Properties ---
    @property
    def cell_area(self) -> float: ...

    # --- Convenience methods ---
    def at_point(self, x: float, y: float) -> float: ...
    def at_points(self, xy: np.ndarray) -> np.ndarray: ...
    def control_in_region(self, x_min, x_max, y_min, y_max) -> float: ...
    def player_share(self, player_id: int | str) -> float: ...
    def player_surface(self, player_id: int | str) -> np.ndarray: ...
    def to_xarray(self) -> "xr.DataArray": ...
```

### 4.1 Design choices

- **`attacking_team_id`**: surface always oriented relative to one team. Caller
  decides (possession team or LTR-normalized acting team for VAEP).
- **`per_player_influence`**: `None` by default. Computed only on
  `decompose=True` to avoid 22x memory overhead in the common case.
- **Array immutability**: all numpy array fields have `flags.writeable = False`
  set at construction time (in `__post_init__`). This enforces true
  immutability beyond what `frozen=True` provides (which only prevents
  attribute reassignment, not internal array mutation).
- **`at_point()`**: bilinear interpolation — more accurate than nearest-cell
  for action-coupled queries where coordinates fall between cell centers.
- **`player_share()`**: returns player's fraction of their *team's* total spatial
  influence (denominator is teammates only, not all players). TF-15 computes the
  threat-weighted variant externally.
- **`to_xarray()`**: bridges to labelled-array ecosystem. Dimensions: (y, x)
  for surface; (player_id, y, x) for decomposed. Optional dep.

## 5. Params dataclasses

### 5.1 SpearmanParams

```python
@dataclass(frozen=True)
class SpearmanParams:
    reaction_time: float = 0.7        # seconds before player begins moving
    max_acceleration: float = 7.0     # m/s^2 peak acceleration
    sigma: float = 0.45               # logistic curve steepness (seconds)
    lambda_gk: float = 3.0            # GK control-rate multiplier (Shaw default)
    average_ball_speed: float = 15.0  # m/s for ball-travel-time filter
    grid_cells_x: int = 50
    grid_cells_y: int = 32
```

#### Parameter provenance

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| `reaction_time` | 0.7 s | Shaw (2020), Bekkers (2025) | Consistent across implementations |
| `max_acceleration` | 7.0 m/s² | Lakehouse calibration | NOT from Shaw or Spearman 2017 (see §6.1 Note) |
| `sigma` | 0.45 s | Shaw (2020) | Logistic width; consistent with Bekkers |
| `lambda_gk` | 3.0 | Shaw (2020) | GK intercept rate = 3× outfield |
| `average_ball_speed` | 15.0 m/s | Shaw (2020) | For ball-travel-time filter |
| `grid_cells_x/y` | 50/32 | Lakehouse default | ~2.1m × ~2.1m cell size |

### 5.2 FernandezBornnParams

```python
@dataclass(frozen=True)
class FernandezBornnParams:
    max_speed: float = 13.0    # m/s normalizes velocity scaling
    min_radius: float = 4.0    # minimum influence radius (meters)
    max_radius: float = 10.0   # maximum influence radius (meters)
    grid_cells_x: int = 50
    grid_cells_y: int = 32
```

Defaults: DataBallPy + paper description. `max_speed = 13.0` is elite sprint
ceiling for scaling factor alpha = (speed/max_speed)^2.

**Provenance note:** The radius formula `R_i = min(min_r + d³/972, max_r)` is
NOT from the original paper — Fernandez & Bornn 2018 describes the concept
qualitatively but does not publish the exact mathematical formula for the
distance-to-ball scaling. DataBallPy's documentation explicitly states this
was "derived by visual inspection of the figure given in the appendix." We
adopt this as the best available approximation; if a more authoritative
formula is published, it should replace this.

### 5.3 VoronoiParams

```python
@dataclass(frozen=True)
class VoronoiParams:
    grid_cells_x: int = 50
    grid_cells_y: int = 32
```

Minimal — Voronoi is position-only, no physics parameters.

### 5.4 Dispatch validation

Same pattern as `tracking/pressure.py`:

```python
Method = Literal["spearman", "fernandez_bornn", "voronoi"]
PitchControlParams = SpearmanParams | FernandezBornnParams | VoronoiParams

def validate_params_for_method(method, params) -> None:
    """Raise TypeError if method/params combination is invalid."""
```

## 6. Model implementations

### 6.1 Spearman (`_spearman.py`)

Three stages: TTI → per-player influence → team aggregation.

#### Note on model lineage

This implementation uses the **ratio approximation** of Spearman's pitch
control, not Shaw's full temporal-integration ODE (dPC/dt = (1 - PC_att -
PC_def) · λ · P(T) integrated with dt=0.04s). The differences:

| Aspect | Shaw ODE (Friends of Tracking) | This implementation (ratio) |
|--------|-------------------------------|----------------------------|
| Aggregation | Temporal integration until convergence | Static ratio: att / (att + def) |
| Probability bound | Structural via (1 - PC) factor | By construction (ratio ∈ [0,1]) |
| GK treatment | λ_gk = 3× λ_outfield | `lambda_gk` multiplier on influence |
| Computational cost | Iterative (10-100 steps per cell) | Single-pass (closed-form) |

The ratio approximation is chosen for:
1. **Computational efficiency** — single-pass is 10-100× faster, critical for
   batch computation and numba vectorization.
2. **Decomposition** — per-player influence is naturally additive in the
   ratio numerator, enabling TF-15's player-share computation. Shaw's ODE
   couples all players through the (1 - PC) factor, making decomposition
   require re-integration with each player removed (N+1 integrations).
3. **Lakehouse validation** — the ratio approach has been validated against
   real tracking data in production. Numerical values differ from Shaw by
   ~5-15% at contested cells but preserve the same spatial structure.

The TTI formula uses **kinematic acceleration** (not Shaw's constant-velocity
max_speed=5.0 model). This originates from the lakehouse implementation and
models realistic player movement more accurately (players cannot instantly
reach top speed). It is NOT from Spearman 2017 or Shaw — it extends their
framework with better physics.

**Stage 1: Kinematic TTI (acceleration-based)**

```
displacement = target - player_pos              # (n_players, n_targets, 2)
distance = ||displacement||                     # (n_players, n_targets)
v_proj = dot(player_vel, unit(displacement))    # velocity toward target
TTI = reaction_time + (-v_proj + sqrt(v_proj^2 + 2 * a_max * distance)) / a_max
```

Players moving toward target arrive sooner (v_proj > 0 reduces TTI). Fully
broadcast-vectorized — no Python loops.

**Stage 2: Per-player logistic influence**

```
k = pi / (sqrt(3) * sigma)
raw_influence_ij = 1 / (1 + exp(-k * (opponent_min_tti_j - team_tti_ij)))
```

**GK weighting:** GK rows (identified via `is_goalkeeper.astype(bool)` in the
tracking frame) have their raw influence output scaled by `lambda_gk` before
aggregation:

```
influence_ij = raw_influence_ij * lambda_gk   if player i is GK
influence_ij = raw_influence_ij               otherwise
```

This is the ratio-approximation analogue of Shaw's ODE rate parameter
(lambda_gk = 12.9 = 3 × 4.3): in Shaw's formalism the GK "claims" territory
faster via a higher dPC/dt rate; in the ratio model the GK contributes 3× the
influence magnitude to the team sum, achieving the same territorial dominance
effect. Default `lambda_gk = 3.0` matches Shaw's 3× multiplier.

Returns per-player matrix `(n_players, n_targets)` — NOT summed. This enables
decomposition. The sum is deferred to aggregation.

**Stage 3: Team aggregation (ratio)**

```
control = sum(attacking_influence_i) / (sum(attacking_influence_i) + sum(defending_influence_j))
```

Where each `influence_i` already includes the GK weighting from Stage 2.
Safe division: returns 0.5 when total < epsilon. The ratio guarantees
control ∈ [0, 1] by construction.

**Ball-travel-time filter** (when `ball_position` supplied):

```
ball_travel_time = distance(ball, target) / average_ball_speed
# Zero influence for players whose TTI > ball_travel_time at each cell
```

### 6.2 Fernandez/Bornn (`_fernandez_bornn.py`)

Each player projects a **directional Gaussian influence field**:

```
1. mu_i = pos_i + 0.5 * vel_i                    # anticipation shift
2. R_i = min(min_radius + dist_to_ball^3/972, max_radius)  # ball-aware radius
3. alpha_i = (speed_i / max_speed)^2              # velocity scaling
4. S_i = [[R*(1+alpha), 0], [0, R*(1-alpha)]]    # elongation matrix
5. theta_i = arctan2(vy_i, vx_i)                  # velocity heading
6. Sigma_i = R_theta * S * S^T * R_theta^T        # covariance
7. influence_i(t) = N(t; mu_i, Sigma_i) / max(N)  # normalized to [0, 1]
```

Team aggregation:

```
team_att = sum(attacking_influence)
team_def = sum(defending_influence)
control = sigmoid(team_att - team_def)
```

**Decomposition semantics:** Because sigmoid is nonlinear, per-player
influences do NOT additively reconstruct the surface (unlike Spearman's ratio
aggregation). For Fernandez/Bornn, `per_player_influence` stores the
**pre-sigmoid raw Gaussian influence** per player. `player_share()` returns
`player_gaussian_sum / team_gaussian_sum` — the player's fraction of their
team's total raw influence input to the sigmoid. This is the meaningful
decomposition: it answers "what proportion of the team's spatial presence is
this player responsible for?" without requiring expensive Shapley-value
recomputation.

**Velocity guards (stationary + high-speed):**

```
SPEED_FLOOR = 0.1   # m/s — below meaningful movement
ALPHA_CEIL  = 0.99  # prevents singular covariance (minor eigenvalue >= 0.01*R^2)

speed = np.linalg.norm(vel, axis=-1)
alpha = np.clip((speed / max_speed) ** 2, 0.0, ALPHA_CEIL)
alpha[speed < SPEED_FLOOR] = 0.0  # isotropic for stationary
```

- **Low-speed guard** (`< 0.1 m/s`): treats player as stationary (`alpha=0`,
  isotropic circular). Prevents tiny velocity noise from creating a spuriously
  elongated ellipse in an arbitrary direction.
- **High-speed guard** (`ALPHA_CEIL = 0.99`): when `speed → max_speed`,
  `alpha → 1.0` which makes `S = [[2R, 0], [0, 0]]` — a rank-1 singular
  matrix whose inverse is undefined. Clamping alpha at 0.99 ensures the
  minor eigenvalue is always `>= 0.01 * R^2`, keeping `inv_cov` well-
  conditioned. Sprints of 10-12 m/s (alpha 0.59-0.85) are unaffected;
  only the extreme tail near exactly `max_speed` (rare in real data, ~47 km/h)
  is clamped.

Note: DataBallPy has the same latent singularity bug (relies on
`multivariate_normal.pdf()` to absorb it silently). Our explicit `inv_cov`
via `einsum` surfaces the failure as NaN — hence the guard is mandatory.

NumPy vectorized via `einsum` for Mahalanobis distance with per-player
covariance:

```python
mahal = np.einsum("pti,pij,ptj->pt", diff, inv_cov, diff)
influence = np.exp(-0.5 * mahal)
```

When `ball_position` supplied: each player's influence radius scales with
their distance to the ball (closer to ball = tighter zone, per the paper's
"distance-to-ball" factor in R_i). When omitted: all players get `min_radius`
(conservative uniform assumption — equivalent to "ball is equidistant from
all players").

### 6.3 Voronoi (`_voronoi.py`)

Nearest-player tessellation — binary assignment:

```python
distances = ||targets - player_pos||  # (n_cells, n_players) via broadcast
nearest_player_idx = argmin(distances, axis=1)
control[cell] = 1.0 if nearest is attacking else 0.0
```

Uses broadcast argmin (not scipy.spatial.Voronoi) — works with 1+ players,
no collinearity requirement, faster for typical grid sizes.

`ball_position` accepted for API consistency but ignored.

## 7. Dispatch layer (`_dispatch.py`)

### 7.1 Main entry point

```python
def compute_pitch_control(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    decompose: bool = False,
    ball_position: tuple[float, float] | None = None,
) -> PitchControlSurface:
```

### 7.2 Batch point-query API

```python
def compute_pitch_control_at_points(
    frame: pd.DataFrame,
    target_points: np.ndarray,
    attacking_team_id: int | str,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    ball_position: tuple[float, float] | None = None,
) -> np.ndarray:
```

Optimized for action-coupled queries: no grid allocation.

### 7.3 Input extraction

- Filters ball rows via `frame["is_ball"].astype(bool)` — NOT `== True`.
  Slim-parquet providers store `is_ball` as object dtype where Python `~`
  yields -1/-2 ints rather than logical negation (established pattern in
  `_off_ball_runs.py:743`, `_kernels.py:741`). Same `.astype(bool)` pattern
  applies to `is_goalkeeper` when identifying GK rows for `lambda_gk`.
  Column guaranteed present by ADR-004 invariant 5.
- Splits by `team_id` (attacking vs defending)
- Extracts positions `(x, y)` and velocities `(vx, vy)`
- NaN velocities → zero (stationary assumption)
- NaN positions → filter out player (treat as absent from frame)
- Raises `ValueError` if method requires velocities but `vx`/`vy` absent

### 7.4 Ball position inference

Priority: explicit kwarg > ball row in frame > None (no conditioning).

### 7.5 Velocity requirement

```python
if method in ("spearman", "fernandez_bornn") and "vx" not in frame.columns:
    raise ValueError(
        f"method='{method}' requires velocity columns (vx, vy). "
        "Call derive_velocities() or smooth_frames() first, "
        "or use method='voronoi' for position-only control."
    )
```

## 8. Numba acceleration

### 8.1 Strategy

Optional `@njit(cache=True)` kernels mirroring NumPy implementations exactly.
Dispatch tries numba first, falls back silently.

### 8.2 Kernels

```python
@numba.njit(cache=True)
def tti_numba(player_pos, player_vel, targets, reaction_time, max_accel): ...

@numba.njit(cache=True)
def influence_numba(team_tti, opponent_min_tti, sigma): ...

@numba.njit(cache=True)
def gaussian_influence_numba(targets, mu, inv_cov, det_cov): ...
```

### 8.3 Performance expectations

| Grid | NumPy (est.) | Numba (est.) | Speedup |
|------|-------------|-------------|---------|
| 50x32 (1,600 cells) | ~5 ms | ~0.5 ms | ~10x |
| 104x68 (7,072 cells) | ~20 ms | ~1.5 ms | ~13x |
| Full-match batch | ~11 min | ~1 min | ~10x |

### 8.4 Dependency declaration

```toml
[project.optional-dependencies]
numba = ["numba>=0.58"]
xarray = ["xarray>=2023.1"]
```

**CI integration:** The existing `golden-master` extra demonstrates the
pattern (env-conditional, test-gated). CI matrix adds:
- One job with `pip install .[numba]` to exercise numba parity tests
- One job with `pip install .[xarray]` to exercise `.to_xarray()` tests
- Default jobs run without either (testing NumPy-only path)

The numba parity tests use `@pytest.mark.skipif(not _HAS_NUMBA, ...)` so
they pass (skip) in the default matrix but exercise in the numba variant.

## 9. Action-coupled VAEP integration

### 9.1 Per-Series helper

```python
def pitch_control_at_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
) -> pd.Series:
    """Pitch control at action's start_x/start_y. NaN for unlinked."""
```

### 9.2 Aggregator

```python
def add_pitch_control(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
) -> pd.DataFrame:
    """Append pitch_control_at_ball__<method> column."""
```

### 9.3 VAEP xfn factory

```python
def pitch_control_xfns(
    method: Method = "spearman",
    params: PitchControlParams | None = None,
) -> list[FrameAwareTransformer]:
    """Returns frame-aware xfn emitting: pitch_control_at_ball__<method>_a0/a1/a2"""

pitch_control_default_xfns: list[FrameAwareTransformer] = pitch_control_xfns("spearman")
```

Ships Spearman only in the default list (per ADR-005 section 8: "default xfn list
ships exactly ONE flavor"). Consumers wanting multiple flavors register
additional `pitch_control_xfns("fernandez_bornn")` explicitly.

**Import-time binding note:** `pitch_control_default_xfns` captures default
`SpearmanParams()` at import time. If future versions change param defaults,
this binding does not auto-update. Consumers needing specific params should
call `pitch_control_xfns(method, params)` directly. This mirrors the existing
`tracking_default_xfns` pattern and will be documented in the docstring.

### 9.4 Gamestates introspection tolerance

Per `feedback_vaep_feature_column_names_introspection`: when frames=None
(VAEP fit-time dummy), emit all-NaN columns silently for shape discovery.

### 9.5 Batch optimization

Actions grouped by linked frame_id: one `compute_pitch_control_at_points()`
call per unique frame, results scattered back to action indices.

## 10. Testing strategy

### 10.1 Test structure

```
tests/tracking/pitch_control/
├── test_surface_dataclass.py       # PitchControlSurface methods + invariants
├── test_spearman.py                # Spearman unit tests
├── test_fernandez_bornn.py         # Fernandez/Bornn unit tests
├── test_voronoi.py                 # Voronoi unit tests
├── test_dispatch.py                # Routing, validation, ball inference
├── test_numba_parity.py            # NumPy == numba golden-master
├── test_action_coupled.py          # VAEP xfn + introspection
└── test_lakehouse_parity.py        # Cross-reference (Spearman)

tests/invariants/
└── test_pitch_control_invariants.py
```

### 10.2 Physical invariants (all flavors)

1. **Bounds**: surface values in [0, 1]
2. **Self-dominance** (with distant opponents): single player on a cell,
   all opponents > 30m away → control exceeds threshold. Thresholds
   per method: Spearman/Voronoi > 0.95, Fernandez/Bornn > 0.80 (sigmoid
   aggregation means even distant Gaussian tails reduce the value below
   what ratio aggregation produces).
3. **Symmetry**: mirrored teams → surface ~ 0.5 everywhere
4. **Monotonicity**: closer player → higher control
5. **Velocity effect** (Spearman/FB): moving toward cell → higher control
6. **Decomposition consistency** (method-qualified):
   - Spearman/Voronoi: per_player influence sums reconstruct surface
     (linear aggregation — ratio or binary assignment)
   - Fernandez/Bornn: per_player raw Gaussian sums equal team_input to
     sigmoid (pre-sigmoid consistency). The surface itself is NOT
     additively decomposable due to sigmoid nonlinearity.
7. **Grid bounds**: grid_x in [0, 105], grid_y in [0, 68]
8. **No NaN/inf** (Fernandez/Bornn): player at `speed = max_speed - 0.01`
   produces a valid surface with no NaN or inf values (guards the
   high-speed covariance singularity).

### 10.3 Provider coverage

All 4 providers (PFF, Sportec, Metrica, SkillCorner) tested. Existing synthetic
fixtures used where sufficient. If fixtures lack velocity data or frame density
for pitch control, regenerate from lakehouse + local PFF data using the
probe-driven fixture parameterization pattern (per
`feedback_probe_driven_fixture_parameterization`).

### 10.4 Numba parity

Golden-master pattern: fixed-seed input → assert numpy_result == numba_result
within ULP tolerance. Skipped when numba not installed.

### 10.5 Lakehouse parity

Port a known lakehouse input/output pair (Spearman, converted to 105x68 meters)
and verify numerical agreement. One-time validation test.

### 10.6 Performance budget

```python
_BUDGET = 0.05 if sys.platform != "win32" else 0.075  # seconds
```

Single-frame Spearman on 50x32 grid, 22 players, NumPy-only (no numba).

**Note:** This library budget (50ms) is deliberately relaxed relative to the
lakehouse's pipeline-critical 5ms/frame target. The lakehouse achieves sub-5ms
via JAX JIT on pre-warmed accelerators in a production pipeline. silly-kicks
targets correctness-first with NumPy; consumers requiring sub-5ms latency
should install numba (expected ~0.5ms/frame) or integrate the lakehouse's
JAX path directly.

## 11. Edge cases

| Situation | Behavior |
|-----------|----------|
| Frame with < 2 players (one team empty) | All-1.0 or all-0.0 |
| Frame with 0 players | All-0.5 |
| NaN positions in frame | Filter out; warn if >50% missing |
| Missing vx/vy + physics method | Raise ValueError with fix instructions |
| `decompose=True` + Voronoi | Binary per-player |
| Ball position off-pitch or NaN | Treat as None (no ball conditioning). Off-pitch = outside [0, 105] x [0, 68] per TRACKING_CONSTRAINTS. |
| Single-team frame | Valid — sole team controls everything |

## 12. ADR

This PR ships **ADR-008** codifying:
- Subpackage pattern for complex spatial computations
- `PitchControlSurface` as a rich-return-type precedent (convenience methods
  on frozen dataclasses)
- Optional numba acceleration contract
- Optional xarray bridge contract

## 13. Scope boundaries

**In scope:**
- Three-flavor pitch control engine (Spearman, Fernandez/Bornn, Voronoi)
- PitchControlSurface dataclass with convenience methods
- Numba optional acceleration
- xarray `.to_xarray()` bridge
- Action-coupled VAEP integration (xfn factory)
- ADR-008
- NOTICE entries
- All-provider test coverage

**Out of scope (deferred to future PRs):**
- TF-15 threat-weighted GK share (consumes PitchControlSurface.player_share)
- OBSO / EPV composition (consumes PitchControlSurface)
- Space creation player-removal differential (future PR)
- Ghost trajectory / counterfactual analysis (lakehouse pattern, future PR)
- Optuna calibration of params (TF-24)
- Dense-grid batch computation for full-match pipelines (future optimization)
